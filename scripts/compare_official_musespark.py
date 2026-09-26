#!/usr/bin/env python
"""Ground-truth comparison of `musespark.forward` against the OFFICIAL sglang PyTorch Muse Spark model, on CPU.

Runs the first `--layers` decoder layers (+ embedding norm, final norm, lm_head) of both implementations on one
prompt with the real bf16 checkpoint weights, taps every sub-step (attention core / attention output / gated
residual / pre-FFN norm / router logits / top-k / expert input / expert mix / FFN output / residual stream /
final hidden / softcapped logits) and prints per-layer max-abs and relative-RMS differences.

The official side imports `sglang/srt/layers/muse_spark_layers.py` VERBATIM from an sglang checkout
(`--sglang-python`, the `python/` directory of branch `dev`; its three sglang imports are stubbed) and is
otherwise a line-by-line transcription of `models/muse_spark.py`, the `forward_native` fallbacks of the
`kernels/ops/muse_spark_v12/fused_*.py` ops, `layers/muse_spark_bf16_moe.py` (torch expert loop),
`layers/attention/torch_native_backend.py` and `layers/logits_processor.py`.  Parameters are created under
bf16 default dtype as sglang does, and the loader-side q/k row permutation + interleaved RoPE are kept.

Requirements: torch (cpu), jax (cpu), safetensors, numpy, and `musespark` importable (run from the repo root).
Memory: ~25 GB per layer per implementation (dense bf16 experts) plus the 6.6 GB lm_head twice.

Example (4 layers, ~15 min on a 224-core host, weights read from page cache):
    JAX_PLATFORMS=cpu python scripts/compare_official_musespark.py \
        --sglang-python /path/to/sglang-dev/python --ids prompt_ids.npy --layers 4 --out /tmp/cmp
Window-semantics test (layer 0 only, 300 ids, window 256 in BOTH implementations):
    ... --layers 1 --sliding-window 256 --no-head --ids ids_300.npy

Noise floors measured on the real model (layer 0): SDPA-bf16 vs explicit-fp32 attention 1.9e-3 rel RMS on the
residual stream; bf16 vs fp32 accumulation of the 8 expert outputs 1.1e-3.  Anything above ~1e-2 is a real
formula difference; `topk_scores` rel RMS is inflated by near-tie top-8 flips whenever the inputs differ at all.
"""

import argparse
import dataclasses
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
from jax import lax
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import musespark as ms  # noqa: E402
from musespark import (  # noqa: E402
    BF16, F32, HIGHEST, allowed_keys, expert_weights, layer_weights, mm, r16, rms, rope_rotate_half, route,
)
from musespark.load import Checkpoint  # noqa: E402

PREFIX = "model.language_model."


# ---- import the official layer primitives verbatim ---------------------------------------------
def import_official_layers(sglang_python_dir):
    for name, attrs in {
        "sglang": {},
        "sglang.srt": {},
        "sglang.srt.distributed": {"divide": lambda a, b: a // b},
        "sglang.srt.layers": {},
        "sglang.srt.layers.linear": {"ColumnParallelLinear": torch.nn.Module},
        "sglang.srt.utils": {"is_cpu": lambda: False},
    }.items():
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        mod.__path__ = []
        sys.modules[name] = mod
    spec = importlib.util.spec_from_file_location(
        "muse_spark_layers", Path(sglang_python_dir) / "sglang" / "srt" / "layers" / "muse_spark_layers.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


L = None  # set by main() once --sglang-python is known


# ---- checkpoint access --------------------------------------------------------------------------
class Shards:
    def __init__(self, path):
        self.path = path
        self.index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        self._open = {}

    def get(self, key):
        fn = self.index[key]
        if fn not in self._open:
            self._open[fn] = safe_open(self.path / fn, framework="pt", device="cpu")
        return self._open[fn].get_tensor(key)


# ---- model -------------------------------------------------------------------------------------
class OfficialAttention:
    """MuseSpark1xAttention (models/muse_spark.py) with tp=1 and explicit torch maths."""

    def __init__(self, cfg, layer_idx, shards, prefix):
        self.layer_idx = layer_idx
        self.head_dim = cfg["head_dim"]
        self.num_heads = cfg["num_attention_heads"]
        self.num_kv_heads = cfg["num_key_value_heads"]
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.use_rope = cfg["layer_rope_theta"][layer_idx] > 0
        # M: sliding_window = config.sliding_window - 1 if layer_types[l] == "sliding_attention" else None
        self.sliding_window = (
            cfg["sliding_window"] - 1 if cfg["layer_types"][layer_idx] == "sliding_attention" else None
        )
        # M: self.scaling = head_dim**-0.5; self.scaling *= qk_scale_factor
        self.scaling = self.head_dim**-0.5
        self.scaling *= cfg["qk_scale_factor"]
        self.norm_eps = cfg["rms_norm_eps"]

        def permute_qk(w):
            # M load_weights: q/k rows -> reshape(heads, 2, D/2, in).transpose(1, 2).reshape(shape)
            heads = self.num_heads if w.shape[0] == self.q_size else self.num_kv_heads
            shape = w.shape
            return w.reshape(heads, 2, shape[0] // heads // 2, *shape[1:]).transpose(1, 2).reshape(shape).contiguous()

        q = permute_qk(shards.get(prefix + "self_attn.q_proj.weight"))
        k = permute_qk(shards.get(prefix + "self_attn.k_proj.weight"))
        v = shards.get(prefix + "self_attn.v_proj.weight")
        g = shards.get(prefix + "self_attn.gate_proj.weight")
        # QKVOGateParallelLinear: one [q | k | v | g] weight
        self.qkv_weight = torch.cat([q, k, v, g], dim=0)
        self.o_weight = shards.get(prefix + "self_attn.o_proj.weight")
        # rotary_embedding/base.py: inv_freq, cos_sin_cache = cat(cos, sin) fp32
        if self.use_rope:
            base = cfg["layer_rope_theta"][layer_idx]
            inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float) / self.head_dim))
            t = torch.arange(0, 8192, dtype=torch.float)
            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            self.cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        # _o_norm_module = MuseSpark1xRMSNorm(head_dim, eps=norm_eps, has_weight=False)
        self.o_norm = L.MuseSpark1xRMSNorm(self.head_dim, eps=self.norm_eps, has_weight=False)

    def _rope_native(self, normed, positions):
        # fused_qk_norm.py::_rope_native (interleaved pairs on the permuted layout)
        d = self.head_dim
        cos, sin = self.cos_sin_cache[positions].float().chunk(2, dim=-1)
        out = []
        for t in normed:
            x = t.float().reshape(t.shape[0], -1, d // 2, 2)
            c = cos[:, None, :, None]
            s_ = sin[:, None, :, None]
            x0, x1 = x[..., 0:1], x[..., 1:2]
            rot = torch.cat([x0 * c - x1 * s_, x0 * s_ + x1 * c], dim=-1)
            out.append(rot.reshape(t.shape).to(t.dtype))
        return out[0], out[1]

    def forward(self, positions, hidden_states, taps=None, attn_impl="sdpa"):
        qkv = F.linear(hidden_states, self.qkv_weight)
        split_sizes = [self.q_size, self.kv_size, self.kv_size]
        q, k, v, o_gate = qkv.split(split_sizes + [self.q_size], dim=-1)
        # FusedQKNorm.forward_native: per-head F.rms_norm (weightless), then interleaved rope
        d = self.head_dim
        normed = [F.rms_norm(t.reshape(-1, d), (d,), None, self.norm_eps).reshape(t.shape) for t in (q, k)]
        if self.use_rope:
            normed = list(self._rope_native(normed, positions))
        q, k = normed
        T = q.shape[0]
        qh = q.reshape(T, self.num_heads, d)
        kh = k.reshape(T, self.num_kv_heads, d)
        vh = v.reshape(T, self.num_kv_heads, d)
        # torch_native_backend: sliding mask (k_pos <= q_pos) & (k_pos >= q_pos - sliding_window_size)
        q_pos = positions[:, None]
        k_pos = positions[None, :]
        if self.sliding_window is not None:
            attn_mask = (k_pos <= q_pos) & (k_pos >= q_pos - self.sliding_window)
        else:
            attn_mask = k_pos <= q_pos  # is_causal=True (prefix-free prefill, same thing)
        if attn_impl == "sdpa":
            out = F.scaled_dot_product_attention(
                qh.movedim(0, 1).unsqueeze(0),
                kh.movedim(0, 1).unsqueeze(0),
                vh.movedim(0, 1).unsqueeze(0),
                attn_mask=attn_mask,
                enable_gqa=True,
                scale=self.scaling,
            ).squeeze(0).movedim(1, 0)
        else:  # explicit fp32 maths (cross-check of the SDPA kernel numerics)
            group = self.num_heads // self.num_kv_heads
            qf = qh.float().reshape(T, self.num_kv_heads, group, d)
            s = torch.einsum("tkgd,skd->kgts", qf, kh.float()) * self.scaling
            s = s.masked_fill(~attn_mask[None, None], float("-inf"))
            p = torch.softmax(s, dim=-1)
            out = torch.einsum("kgts,skd->tkgd", p, vh.float()).reshape(T, self.num_heads, d).to(qh.dtype)
        output = out.reshape(T, self.num_heads * d).contiguous()
        if taps is not None:
            taps["attn_core"] = output
        # FusedRMSNormGate.forward_native: x = o_norm(x.reshape(-1, D)).reshape(x.shape); x = g.sigmoid() * x
        x = self.o_norm(output.reshape(-1, d)).reshape(output.shape)
        x = o_gate.sigmoid() * x
        output = F.linear(x, self.o_weight)
        return output


class OfficialMoE:
    """MuseSpark1xMoE with tp=1/ep=1, the bf16 torch expert loop and the native routing op."""

    def __init__(self, cfg, layer_idx, shards, prefix):
        self.layer_idx = layer_idx
        self.top_k = cfg["num_experts_per_tok"]
        self.route_scale = cfg["routed_scaling_factor"]
        self.route_eps = 1e-15
        self.router_multiplier = cfg["output_multiplier"]
        self.num_experts = cfg["num_local_experts"]
        self.gate_weight = shards.get(prefix + "mlp.gate.weight").float()  # fp32 in the checkpoint
        self.expert_bias = shards.get(prefix + "mlp.gate.e_score_correction_bias").float()
        self.pre_weight = shards.get(prefix + "mlp.pre_expert_proj.weight")
        self.post_weight = shards.get(prefix + "mlp.post_expert_proj.weight")
        Hm = cfg["moe_hidden_size"]
        self.pre_expert_norm = L.MuseSpark1xRMSNorm(
            Hm, eps=cfg["rms_norm_eps"], has_weight=True, zero_centered_gamma=True, residual_id=2 * layer_idx + 2
        )
        self.post_expert_norm = L.MuseSpark1xRMSNormWithInputScale(
            Hm,
            eps=cfg["post_norm_eps"],
            has_weight=True,
            is_post_norm=True,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=2 * layer_idx + 2,
        )
        with torch.no_grad():
            self.pre_expert_norm.weight.copy_(shards.get(prefix + "mlp.pre_expert_norm.weight"))
            self.post_expert_norm.weight.copy_(shards.get(prefix + "mlp.experts.post_expert_norm.weight"))
        self.gate_up_proj = shards.get(prefix + "mlp.experts.gate_up_proj")  # [E, 2I, Hm]
        self.down_proj = shards.get(prefix + "mlp.experts.down_proj")  # [E, Hm, I]

    def routing_native(self, gating_output):
        # FusedMuseSparkRouting.forward_native + _finalize
        if self.router_multiplier is not None:
            gating_output = gating_output * self.router_multiplier
        scores = torch.sigmoid(gating_output.float())
        scores_for_topk = scores + self.expert_bias
        _, topk_indices = torch.topk(scores_for_topk, self.top_k, dim=-1)  # fast_topk
        topk_indices = topk_indices.to(torch.int32)
        topk_scores = scores.gather(1, topk_indices.long())
        score_sum = topk_scores.sum(dim=-1, keepdim=True)
        topk_scores = topk_scores / (score_sum + self.route_eps) * self.route_scale
        return topk_scores, topk_indices

    def experts_loop(self, hidden_states, topk_scores, topk_ids, post_expert_norm, accumulate_fp32=False):
        # MuseSpark1xBf16Experts.forward (torch loop)
        output = torch.zeros_like(hidden_states, dtype=torch.float32 if accumulate_fp32 else hidden_states.dtype)
        for local_id in range(self.num_experts):
            global_id = local_id
            token_slot = (topk_ids == global_id).nonzero(as_tuple=False)
            if token_slot.numel() == 0:
                continue
            token_ids = token_slot[:, 0]
            slots = token_slot[:, 1]
            selected = hidden_states.index_select(0, token_ids)
            gate_up = F.linear(selected, self.gate_up_proj[local_id])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = F.linear(F.silu(gate) * up, self.down_proj[local_id])
            if post_expert_norm is not None:
                expert_output = post_expert_norm(expert_output)
            expert_output = expert_output * topk_scores[token_ids, slots].unsqueeze(-1)
            output.index_add_(0, token_ids, expert_output.to(output.dtype))
        return output.to(hidden_states.dtype)

    def forward(self, hidden_states, router_input, taps=None, accumulate_fp32=False):
        # _forward_experts (CPU branch): pre_expert_proj -> router -> routing -> pre_expert_norm -> experts
        expert_input = F.linear(hidden_states, self.pre_weight)
        router_logits = F.linear(router_input, self.gate_weight)
        topk_scores, topk_ids = self.routing_native(router_logits)
        expert_input = self.pre_expert_norm(expert_input)
        if taps is not None:
            taps["router_logits"] = router_logits
            taps["topk_ids"] = topk_ids
            taps["topk_scores"] = topk_scores
            taps["expert_input"] = expert_input
        output = self.experts_loop(expert_input, topk_scores, topk_ids, self.post_expert_norm, accumulate_fp32)
        if taps is not None:
            taps["moe_mix"] = output
        # _reduce_and_unpool (CPU): post_expert_proj
        output = F.linear(output, self.post_weight)
        return output


class OfficialLayer:
    def __init__(self, cfg, layer_idx, shards):
        prefix = f"{PREFIX}layers.{layer_idx}."
        H = cfg["hidden_size"]
        self.layer_idx = layer_idx
        self.self_attn = OfficialAttention(cfg, layer_idx, shards, prefix)
        self.post_attn_norm = L.MuseSpark1xRMSNorm(
            H, eps=cfg["post_norm_eps"], has_weight=False, is_post_norm=True, zero_centered_gamma=True,
            post_norm_gain_center_type="one", residual_id=2 * layer_idx + 1,
        )
        self.pre_feedforward_layernorm = L.MuseSpark1xRMSNorm(
            H, eps=cfg["rms_norm_eps"], has_weight=True, zero_centered_gamma=True, residual_id=2 * layer_idx + 2
        )
        self.post_feedforward_layernorm = L.MuseSpark1xRMSNormWithInputScale(
            H, eps=cfg["post_norm_eps"], has_weight=True, is_post_norm=True, zero_centered_gamma=True,
            post_norm_gain_center_type="one", residual_id=2 * layer_idx + 2,
        )
        self.input_layernorm = L.MuseSpark1xRMSNorm(
            H, eps=cfg["rms_norm_eps"], has_weight=True, zero_centered_gamma=True, residual_id=2 * layer_idx + 1
        )
        self.mlp = OfficialMoE(cfg, layer_idx, shards, prefix)
        self.post_attention_residual_gate = L.MuseSpark1xNormPreservingResidualConnection(
            H, temperature=cfg["residual_gate_temperature"]
        )
        self.post_feedforward_residual_gate = L.MuseSpark1xNormPreservingResidualConnection(
            H, temperature=cfg["residual_gate_temperature"]
        )
        with torch.no_grad():
            self.input_layernorm.weight.copy_(shards.get(prefix + "input_layernorm.weight"))
            self.pre_feedforward_layernorm.weight.copy_(shards.get(prefix + "pre_feedforward_layernorm.weight"))
            self.post_feedforward_layernorm.weight.copy_(shards.get(prefix + "post_feedforward_layernorm.weight"))
            self.post_attention_residual_gate.gate.copy_(shards.get(prefix + "post_attention_residual_gate.gate"))
            self.post_feedforward_residual_gate.gate.copy_(shards.get(prefix + "post_feedforward_residual_gate.gate"))
        self.operator_dtype = torch.bfloat16

    def modules(self):
        return [
            self.self_attn.o_norm, self.post_attn_norm, self.pre_feedforward_layernorm,
            self.post_feedforward_layernorm, self.input_layernorm, self.mlp.pre_expert_norm,
            self.mlp.post_expert_norm,
        ]

    def boundary(self, post_norm, residual_add, pre_norm, stream, branch, branch_dtype, router_dtype=None):
        # FusedPostNormResidualPreNorm.forward_native
        normed_branch = post_norm(branch)
        new_stream = residual_add(stream, normed_branch)
        normed = pre_norm(new_stream)
        output = normed.to(branch_dtype)
        router_input = output.to(router_dtype) if router_dtype is not None else None
        return output, new_stream, router_input

    def forward(self, positions, stream, attn_input, next_layer, taps, **kw):
        # MuseSpark1xDecoderLayer.forward
        attn_output = self.self_attn.forward(positions, attn_input, taps=taps, attn_impl=kw.get("attn_impl", "sdpa"))
        taps["attn_out"] = attn_output
        ffn_input, stream, router_input = self.boundary(
            self.post_attn_norm, self.post_attention_residual_gate, self.pre_feedforward_layernorm,
            stream, attn_output, self.operator_dtype, torch.float32,
        )
        taps["stream_after_attn"] = stream
        taps["ffn_input"] = ffn_input
        ffn_output = self.mlp.forward(ffn_input, router_input, taps=taps, accumulate_fp32=kw.get("accumulate_fp32", False))
        taps["ffn_out"] = ffn_output
        if next_layer is None:
            return self.post_feedforward_residual_gate(stream, self.post_feedforward_layernorm(ffn_output)), None
        next_attn_input, stream, _ = self.boundary(
            self.post_feedforward_layernorm, self.post_feedforward_residual_gate, next_layer.input_layernorm,
            stream, ffn_output, self.operator_dtype,
        )
        return stream, next_attn_input


class OfficialModel:
    def __init__(self, num_layers=4, shards=None, cfg=None, load_head=True):
        self.cfg = cfg
        H = cfg["hidden_size"]
        t0 = time.time()
        # sglang builds the model under the checkpoint dtype (bf16): every norm weight / gate parameter is bf16,
        # so `weight + gain_center` (precompute_effective_weight) rounds to bf16.
        torch.set_default_dtype(torch.bfloat16)
        self.embed_tokens = shards.get(PREFIX + "embed_tokens.weight")
        self.embed_norm = L.MuseSpark1xRMSNorm(H, eps=cfg["rms_norm_eps"], has_weight=False)
        self.layers = []
        for i in range(num_layers):
            self.layers.append(OfficialLayer(cfg, i, shards))
            print(f"  loaded layer {i} ({time.time() - t0:.0f}s)", flush=True)
        self.norm = L.MuseSpark1xRMSNorm(
            H, eps=cfg["rms_norm_eps"], has_weight=True, zero_centered_gamma=True,
            output_norm_gain_center_type="zero", residual_id=-1,
        )
        with torch.no_grad():
            self.norm.weight.copy_(shards.get(PREFIX + "norm.weight"))
        self.lm_head = shards.get("lm_head.weight") if load_head else None
        torch.set_default_dtype(torch.float32)
        self.operator_dtype = torch.bfloat16
        # post_load_weights
        with torch.no_grad():
            for layer in self.layers:
                layer.post_attention_residual_gate.precompute_coefficients()
                layer.post_feedforward_residual_gate.precompute_coefficients()
                for m in layer.modules():
                    m.precompute_effective_weight()
            self.norm.precompute_effective_weight()
            self.embed_norm.precompute_effective_weight()

    @torch.no_grad()
    def forward(self, input_ids, positions, **kw):
        taps = {}
        hidden_states = self.embed_tokens[input_ids]
        hidden_states = self.embed_norm(hidden_states)
        stream = hidden_states.to(torch.float32)
        taps["hidden"] = [stream.clone()]
        attn_input = self.layers[0].input_layernorm(stream).to(self.operator_dtype)
        taps["layers"] = []
        for i, layer in enumerate(self.layers):
            lt = {}
            nxt = self.layers[i + 1] if i + 1 < len(self.layers) else None
            stream, attn_input = layer.forward(positions, stream, attn_input, nxt, lt, **kw)
            taps["layers"].append(lt)
            taps["hidden"].append(stream.clone())
        norm_input = stream.to(self.operator_dtype)
        hidden_states = self.norm(norm_input).to(self.operator_dtype)
        taps["final_hidden"] = hidden_states
        if self.lm_head is not None:
            # MuseSpark1xLogitsProcessor._compute_lm_head + logit_scale + softcap
            # torch.mm(..., out_dtype=fp32) is CUDA-only; bf16 x bf16 with fp32 accumulate/output == fp32 mm of
            # the bf16 values (products exact in fp32).
            logits = torch.mm(hidden_states.to(self.lm_head.dtype).float(), self.lm_head.float().T)
            logits.mul_(self.cfg["output_multiplier"])
            cap = self.cfg["final_logit_softcapping"]
            logits = cap * torch.tanh(logits / cap)
            taps["logits"] = logits
        else:
            taps["logits"] = torch.zeros(1, 1)
        return taps


def official_taps_to_numpy(taps):
    out = {}
    out["hidden"] = np.stack([h.float().numpy() for h in taps["hidden"]])
    for i, lt in enumerate(taps["layers"]):
        for k, v in lt.items():
            out[f"L{i}/{k}"] = v.float().numpy() if v.dtype != torch.int32 else v.numpy()
    out["final_hidden"] = taps["final_hidden"].float().numpy()
    if "logits" in taps:
        out["logits"] = taps["logits"].numpy()
    return out




# ---- tapped copies of musespark.attention_branch / moe_branch / decoder_layer -------------------
def attention_branch_tapped(cfg, lw, x_attn, positions, k_cache, v_cache, layer, taps):
    B, T, _ = x_attn.shape
    nH, nKV, D = cfg.heads, cfg.kv_heads, cfg.head_dim
    q = mm(x_attn, lw["q"]).reshape(B, T, nH, D)
    k = mm(x_attn, lw["k"]).reshape(B, T, nKV, D)
    v = mm(x_attn, lw["v"]).reshape(B, T, nKV, D)
    g = mm(x_attn, lw["gate"]).reshape(B, T, nH, D)
    q = r16(rms(q, None, cfg.rms_eps))
    k = r16(rms(k, None, cfg.rms_eps))
    if not cfg.is_full_attention(layer):
        q = rope_rotate_half(q, positions, cfg.rope_theta)
        k = rope_rotate_half(k, positions, cfg.rope_theta)
    rows = jnp.arange(B)[:, None]
    k_cache = k_cache.at[rows, positions].set(k.astype(BF16))
    v_cache = v_cache.at[rows, positions].set(v.astype(BF16))
    group = nH // nKV
    qg = q.reshape(B, T, nKV, group, D)
    scores = jnp.einsum("btkgd,bskd->bkgts", qg, k_cache.astype(F32), precision=HIGHEST) * cfg.softmax_scale
    slots = jnp.arange(k_cache.shape[1])
    ok = allowed_keys(cfg, positions, slots, layer)
    scores = jnp.where(ok[:, None, None], scores, -jnp.inf)
    p = jax.nn.softmax(scores, axis=-1)
    o = jnp.einsum("bkgts,bskd->btkgd", p, v_cache.astype(F32), precision=HIGHEST)
    o = o.reshape(B, T, nH, D)
    taps["attn_core"] = r16(o).reshape(B, T, nH * D)
    o = rms(o, None, cfg.rms_eps) * jax.nn.sigmoid(g)
    attn_out = mm(r16(o).reshape(B, T, nH * D), lw["o"])
    return attn_out, k_cache, v_cache


def moe_branch_tapped(cfg, lw, x_ffn, taps):
    I = cfg.expert_hidden
    logits = jnp.dot(x_ffn, jnp.asarray(lw["router"], F32), precision=HIGHEST)
    idx, w = route(logits * cfg.output_multiplier, lw["router_bias"], cfg.top_k, cfg.route_eps)
    h1 = r16(rms(mm(x_ffn, lw["pre"]), lw["pre_expert_norm"], cfg.rms_eps))
    taps["router_logits"], taps["topk_ids"], taps["topk_scores"], taps["expert_input"] = logits, idx, w, h1
    gate_up, down = expert_weights(lw)
    w_post = jnp.asarray(lw["post_expert_norm"], F32)
    m = jnp.zeros(h1.shape, F32)
    for slot in range(cfg.top_k):
        e = idx[..., slot]
        gu = r16(jnp.einsum("bth,btho->bto", h1.astype(BF16), gate_up[e], preferred_element_type=F32))
        act = r16(jax.nn.silu(gu[..., :I]) * gu[..., I:])
        y = r16(jnp.einsum("bti,btio->bto", act.astype(BF16), down[e], preferred_element_type=F32))
        t = r16(y * w_post)
        yn = t * lax.rsqrt(jnp.mean(t * t, axis=-1, keepdims=True) + cfg.post_eps)
        m = m + w[..., slot : slot + 1] * yn
    taps["moe_mix"] = r16(m)
    return mm(r16(m), lw["post"])


def decoder_layer_tapped(cfg, lw, s, x_attn, positions, k_cache, v_cache, layer, taps):
    attn_out, k_cache, v_cache = attention_branch_tapped(cfg, lw, x_attn, positions, k_cache, v_cache, layer, taps)
    taps["attn_out"] = attn_out
    nb = r16(rms(attn_out, None, cfg.post_eps))
    s = lw["attn_gate_alpha"] * s + lw["attn_gate_beta"] * nb
    taps["stream_after_attn"] = s
    x_ffn = r16(rms(s, lw["ffn_norm"], cfg.rms_eps))
    taps["ffn_input"] = x_ffn
    ffn_out = moe_branch_tapped(cfg, lw, x_ffn, taps)
    taps["ffn_out"] = ffn_out
    t = r16(ffn_out * jnp.asarray(lw["post_ffn_norm"], F32))
    nb = r16(t * lax.rsqrt(jnp.mean(t * t, axis=-1, keepdims=True) + cfg.post_eps))
    s = lw["ffn_gate_alpha"] * s + lw["ffn_gate_beta"] * nb
    return s, k_cache, v_cache


def forward_tapped(cfg, weights, tokens, start_positions, caches):
    tokens = jnp.asarray(tokens, jnp.int32)
    T = tokens.shape[1]
    positions = jnp.asarray(start_positions, jnp.int32)[:, None] + jnp.arange(T, dtype=jnp.int32)
    k_cache, v_cache = caches
    s = ms.embed(weights, tokens, cfg.rms_eps)
    x = r16(rms(s, weights["attn_norm"][0], cfg.rms_eps))
    hidden, layer_taps = [s], []
    for layer in range(cfg.layers):
        lw = layer_weights(weights, layer)
        taps = {}
        s, k_l, v_l = decoder_layer_tapped(cfg, lw, s, x, positions, k_cache[layer], v_cache[layer], layer, taps)
        k_cache, v_cache = k_cache.at[layer].set(k_l), v_cache.at[layer].set(v_l)
        hidden.append(s)
        layer_taps.append(taps)
        if layer + 1 < cfg.layers:
            x = r16(rms(s, weights["attn_norm"][layer + 1], cfg.rms_eps))
        print(f"  layer {layer} done", flush=True)
    hN = r16(rms(r16(s), weights["final_norm"], cfg.rms_eps))
    raw = jnp.dot(hN.astype(BF16), jnp.asarray(weights["lm_head"]), preferred_element_type=F32)
    logits = ms.softcap_logits(cfg, raw)
    return logits, jnp.stack(hidden), layer_taps, hN


# ---- weights -----------------------------------------------------------------------------------
def load_weights(cfg, ckpt, num_layers, load_head=True, experts=True):
    t0 = time.time()
    get = ckpt.read
    w = ms.canonical_global_from_checkpoint(cfg, get)
    if not load_head:
        w["lm_head"] = np.zeros((cfg.hidden, 8), ms.NP_BF16)
    per_layer = []
    for l in range(num_layers):
        lw = ms.canonical_layer_from_checkpoint(cfg, l, get, experts=False)
        if experts:
            p = f"{ms.CHECKPOINT_PREFIX}layers.{l}.mlp.experts."
            # dense bf16 experts: gate_up [E, Hm, 2I] = gate_up_proj.transpose(0, 2, 1); down likewise
            lw["gate_up"] = np.ascontiguousarray(get(p + "gate_up_proj").transpose(0, 2, 1))
            lw["down"] = np.ascontiguousarray(get(p + "down_proj").transpose(0, 2, 1))
        per_layer.append(lw)
        print(f"  loaded layer {l} ({time.time() - t0:.0f}s)", flush=True)
    # per-layer entries as lists (indexed by layer exactly like a stacked [L, ...] array)
    for name in per_layer[0]:
        w[name] = [jnp.asarray(lw[name]) for lw in per_layer]
    for name in ("embed", "lm_head", "final_norm"):
        w[name] = jnp.asarray(w[name])
    return w


def jax_taps_to_numpy(logits, hidden, layer_taps, hN):
    out = {"hidden": np.asarray(hidden)[:, 0], "logits": np.asarray(logits)[0], "final_hidden": np.asarray(hN)[0]}
    for i, lt in enumerate(layer_taps):
        for k, v in lt.items():
            out[f"L{i}/{k}"] = np.asarray(v)[0]
    return out




STEPS = [
    "attn_core", "attn_out", "stream_after_attn", "ffn_input", "router_logits", "topk_ids", "topk_scores",
    "expert_input", "moe_mix", "ffn_out",
]


def stats(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = a - b
    rel_rms = np.sqrt((d * d).mean()) / (np.sqrt((b * b).mean()) + 1e-30)
    return dict(max_abs=float(np.abs(d).max()), rel_rms=float(rel_rms), rms_b=float(np.sqrt((b * b).mean())))


def worst_rows(a, b, n=3):
    d = np.sqrt(((a - b) ** 2).mean(-1)) / (np.sqrt((b * b).mean(-1)) + 1e-30)
    idx = np.argsort(-d)[:n]
    return [(int(i), float(d[i])) for i in idx]


def compare(o, j):
    if "copy_vs_module_max_abs" in j:
        print("jax tapped-copy vs musespark.forward max abs (hidden, logits):", j["copy_vs_module_max_abs"])
    print("\n== residual stream (after embedding / after each layer) ==")
    for l in range(o["hidden"].shape[0]):
        s = stats(j["hidden"][l], o["hidden"][l])
        print(f"hidden[{l}]: max_abs={s['max_abs']:.3e} rel_rms={s['rel_rms']:.3e} (rms={s['rms_b']:.3f})")
    print("\n== per-layer sub-steps (rel RMS of ours vs official; > 1e-2 flagged) ==")
    L = o["hidden"].shape[0] - 1
    for l in range(L):
        print(f"-- layer {l}")
        for st in STEPS:
            ko = f"L{l}/{st}"
            if ko not in o or ko not in j:
                continue
            a, b = j[ko], o[ko]
            if st == "topk_ids":
                same_set = np.mean([set(a[t].tolist()) == set(b[t].tolist()) for t in range(a.shape[0])])
                same_order = np.mean((a == b).all(-1))
                print(f"  {st:18s} same-set={same_set:.4f} same-order={same_order:.4f}")
                continue
            if st == "topk_scores":
                # compare as per-token expert->weight maps (order may differ)
                E = 256
                wa = np.zeros((a.shape[0], E)); wb = np.zeros((a.shape[0], E))
                np.put_along_axis(wa, j[f"L{l}/topk_ids"].astype(int), a, axis=1)
                np.put_along_axis(wb, o[f"L{l}/topk_ids"].astype(int), b, axis=1)
                s = stats(wa, wb)
            else:
                s = stats(a, b)
            flag = "  <-- FLAG" if s["rel_rms"] > 1e-2 else ""
            wr = worst_rows(np.asarray(a, np.float64), np.asarray(b, np.float64)) if s["rel_rms"] > 1e-2 else ""
            print(f"  {st:18s} max_abs={s['max_abs']:.3e} rel_rms={s['rel_rms']:.3e} rms={s['rms_b']:.3f}{flag} {wr}")
    print("\n== final ==")
    s = stats(j["final_hidden"], o["final_hidden"])
    print(f"final_hidden: max_abs={s['max_abs']:.3e} rel_rms={s['rel_rms']:.3e}")
    lo, lj = o["logits"], j["logits"]
    s = stats(lj, lo)
    top1 = np.mean(lo.argmax(-1) == lj.argmax(-1))
    # top-1 margin agreement: how often our top-1 is within official's top-5
    top5 = np.argsort(-lo, axis=-1)[:, :5]
    in5 = np.mean([lj[t].argmax() in top5[t] for t in range(lo.shape[0])])
    print(f"logits: max_abs={s['max_abs']:.3e} rel_rms={s['rel_rms']:.3e} top1-agree={top1:.4f} ours-top1-in-official-top5={in5:.4f}")
    print("official top1 last 8:", lo.argmax(-1)[-8:].tolist())
    print("ours     top1 last 8:", lj.argmax(-1)[-8:].tolist())




def main():
    global L
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sglang-python", required=True, help="sglang checkout's python/ dir (branch dev)")
    ap.add_argument("--checkpoint", default="/filestore/weights/Muse-Spark-1.2-816B-A42B-open")
    ap.add_argument("--ids", required=True, help=".npy of int token ids (one prompt)")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--sliding-window", type=int, default=0, help="override sliding_window in BOTH implementations")
    ap.add_argument("--no-head", action="store_true", help="skip lm_head (saves 13 GB and time)")
    ap.add_argument("--out", default=None, help="directory for official_taps.npz / jax_taps.npz")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    L = import_official_layers(args.sglang_python)
    ckpt_dir = Path(args.checkpoint)
    ids = np.load(args.ids).astype(np.int64)
    print(f"ids: {ids.shape}", flush=True)

    # ---- official
    cfg = json.loads((ckpt_dir / "config.json").read_text())["text_config"]
    if args.sliding_window:
        cfg["sliding_window"] = args.sliding_window
    print("== official (torch) ==", flush=True)
    model = OfficialModel(args.layers, shards=Shards(ckpt_dir), cfg=cfg, load_head=not args.no_head)
    t0 = time.time()
    taps = model.forward(torch.from_numpy(ids), torch.arange(len(ids)))
    print(f"official forward {time.time() - t0:.0f}s", flush=True)
    o = official_taps_to_numpy(taps)
    del model, taps

    # ---- ours
    print("== ours (musespark.forward, JAX cpu) ==", flush=True)
    ck = Checkpoint(ckpt_dir)
    jcfg = dataclasses.replace(ck.config, layers=args.layers)
    if args.sliding_window:
        jcfg = dataclasses.replace(jcfg, sliding_window=args.sliding_window)
    w = load_weights(jcfg, ck, args.layers, load_head=not args.no_head)
    T = len(ids)
    caches = ms.init_caches(jcfg, 1, T)
    tok = ids.astype(np.int32)[None]
    t0 = time.time()
    logits, hidden, layer_taps, hN = forward_tapped(jcfg, w, tok, jnp.zeros(1, jnp.int32), caches)
    logits2, _, hidden2 = ms.forward(jcfg, w, tok, jnp.zeros(1, jnp.int32), caches, return_hidden=True)
    jax.block_until_ready(logits2)
    print(f"jax forward {time.time() - t0:.0f}s", flush=True)
    j = jax_taps_to_numpy(logits2, hidden2, layer_taps, hN)
    j["copy_vs_module_max_abs"] = np.array(
        [float(jnp.max(jnp.abs(hidden - hidden2))), float(jnp.max(jnp.abs(logits - logits2)))]
    )
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        np.savez(Path(args.out) / "official_taps.npz", **o)
        np.savez(Path(args.out) / "jax_taps.npz", **j)
    if args.no_head:
        o["logits"] = np.zeros((T, 1), np.float32)
        j["logits"] = np.zeros((T, 1), np.float32)
    compare(o, j)


if __name__ == "__main__":
    main()
