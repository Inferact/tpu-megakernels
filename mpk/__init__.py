"""Muse-Spark-Open configuration and reference model equations.

Mirrors the torchTPU reference (p-vllm-torchtpu .../muse_spark/{model,layers,
weights}.py) for the open meta-models/Muse-Spark-Open-816B-A42B-NVFP4
checkpoint: FP32 residual stream with gated blends, zero-centered pre-norms,
weight-first post-norms, weightless per-head QK/o norms, interleaved RoPE on
sliding-window layers, NoPE on full-attention layers, sigmoid+bias top-8
routing, pooled MoE with per-expert post-norm before the FP32 weighted sum,
and a tanh soft-capped output projection.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


F32 = jnp.float32
HIGHEST = jax.lax.Precision.HIGHEST

# E2M1 code table (sign bit 3); negative zero is preserved by the table.
NVFP4_LUT = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float32,
)


def dequantize_nvfp4(packed, block_scale, global_scale):
    """Linear-layout E2M1 * block-16 E4M3 * global scale -> fp32 (numpy).

    packed: [..., K/2] uint8, element 2k in the low nibble of byte k.
    block_scale: [..., K/16] float8_e4m3fn (or raw uint8 bits).
    global_scale: scalar-like fp32 broadcast against block_scale.
    """
    import ml_dtypes

    packed = np.asarray(packed, dtype=np.uint8)
    if packed.shape[-1] != block_scale.shape[-1] * 8:
        raise ValueError("NVFP4 block scales must cover 16 unpacked values")
    if packed.shape[:-1] != block_scale.shape[:-1]:
        raise ValueError("NVFP4 scale and weight row shapes disagree")
    if block_scale.dtype == np.uint8:
        block_scale = block_scale.view(ml_dtypes.float8_e4m3fn)
    low = NVFP4_LUT[packed & 15]
    high = NVFP4_LUT[packed >> 4]
    values = np.stack((low, high), axis=-1).reshape(*packed.shape[:-1], -1)
    scales = block_scale.astype(np.float32) * np.asarray(global_scale, dtype=np.float32)
    return (values.reshape(*block_scale.shape, 16) * scales[..., None]).reshape(values.shape)


@dataclass(frozen=True)
class Config:
    dim: int = 8192
    layers: int = 62
    heads: int = 128
    kv_heads: int = 16
    head_dim: int = 64
    vocab: int = 202048
    eps: float = 1e-5
    post_eps: float = 1e-8
    moe_hidden: int = 4096
    moe_intermediate: int = 4096
    experts: int = 256
    top_k: int = 8
    qk_scale_factor: float = 5.4730064863875
    output_multiplier: float = 0.17677669529663687
    soft_cap: float = 20.0
    gate_temperature: float = 0.3
    sliding_window: int = 2048
    rope_theta: float = 500000.0
    router_eps: float = 1e-15
    full_attention: tuple[int, ...] = tuple(range(1, 62, 4))
    bf16_expert_layers: tuple[int, ...] = (0, 61)
    bos: int = 200000
    eos: tuple[int, ...] = (200001, 200008)
    pad: int = 200018

    @classmethod
    def from_checkpoint(cls, directory):
        raw = json.loads((Path(directory) / "config.json").read_text())
        c = raw["text_config"]
        required = {
            "attention_bias": False,
            "hidden_act": "silu",
            "num_shared_experts": 0,
            "tie_word_embeddings": False,
            "routed_scaling_factor": 1.0,
        }
        for name, expected in required.items():
            if c[name] != expected:
                raise ValueError(f"Unsupported Muse-Spark setting {name}={c[name]!r}")
        if c["rope_parameters"]["rope_type"] != "default":
            raise ValueError(f"Unsupported Muse-Spark rope {c['rope_parameters']!r}")
        layers = c["num_hidden_layers"]
        full = tuple(i for i in range(layers) if c["layer_types"][i] == "full_attention")
        for i in range(layers):
            theta = c["layer_rope_theta"][i]
            if c["layer_types"][i] == "full_attention":
                if theta != 0:
                    raise ValueError("Muse-Spark full-attention layers must be NoPE")
            elif c["layer_types"][i] != "sliding_attention" or theta != c["rope_parameters"]["rope_theta"]:
                raise ValueError(f"Unsupported Muse-Spark layer {i} layout")
        if any(kind != "sparse" for kind in c["mlp_layer_types"]):
            raise ValueError("Muse-Spark megakernel expects routed MoE in every layer")
        return cls(
            dim=c["hidden_size"],
            layers=layers,
            heads=c["num_attention_heads"],
            kv_heads=c["num_key_value_heads"],
            head_dim=c["head_dim"],
            vocab=c["vocab_size"],
            eps=c["rms_norm_eps"],
            post_eps=c["post_norm_eps"],
            moe_hidden=c["moe_hidden_size"],
            moe_intermediate=c["moe_intermediate_size"],
            experts=c["num_local_experts"],
            top_k=c["num_experts_per_tok"],
            qk_scale_factor=c["qk_scale_factor"],
            output_multiplier=c["output_multiplier"],
            soft_cap=c["final_logit_softcapping"],
            gate_temperature=c["residual_gate_temperature"],
            sliding_window=c["sliding_window"],
            rope_theta=c["rope_parameters"]["rope_theta"],
            full_attention=full,
            bos=c["bos_token_id"],
            eos=(200001, 200008),
            pad=c["pad_token_id"],
        )

    @property
    def scale(self):
        return self.head_dim**-0.5 * self.qk_scale_factor

    def use_rope(self, index):
        return index not in self.full_attention

    def use_window(self, index):
        return index not in self.full_attention


def rms_gain(x, gain, eps):
    """RMSNorm with the (already center-folded) gain applied after normalize."""
    y = x.astype(F32)
    y = y * jax.lax.rsqrt(jnp.mean(y * y, axis=-1, keepdims=True) + eps)
    return y * gain.astype(F32)


def rms_gain_first(x, gain, eps):
    """Post-norm variant: the gain multiplies the input before normalize."""
    y = x.astype(F32) * gain.astype(F32)
    return y * jax.lax.rsqrt(jnp.mean(y * y, axis=-1, keepdims=True) + eps)


def rms_plain(x, eps):
    y = x.astype(F32)
    return y * jax.lax.rsqrt(jnp.mean(y * y, axis=-1, keepdims=True) + eps)


def residual_coefficients(gate, temperature):
    """(alpha, beta) channel gates, materialized once (matches MuseResidual)."""
    w = np.asarray(gate, dtype=np.float32) / temperature

    def sigmoid(v):
        return 1.0 / (1.0 + np.exp(-v))

    beta = sigmoid(w)
    alpha = np.sqrt(np.clip(sigmoid(-w) * (1.0 + beta), 1e-3, None))
    return alpha.astype(np.float32), beta.astype(np.float32)


def linear(x, weight):
    return jnp.dot(x, weight, preferred_element_type=F32).astype(x.dtype)


def rope_tables(c, positions):
    """[P] positions -> (cos, sin) [P, head_dim//2] fp32 (interleaved-pair tables)."""
    inv = 1.0 / (c.rope_theta ** (jnp.arange(0, c.head_dim, 2, dtype=F32) / c.head_dim))
    angles = positions.astype(F32)[:, None] * inv
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x, cos, sin):
    """x [..., head_dim]; split-half rotation: (i, i + head_dim/2), fp32 in/out."""
    real, imag = jnp.split(x.astype(F32), 2, axis=-1)
    return jnp.concatenate((real * cos - imag * sin, real * sin + imag * cos), axis=-1)


def route(x, w, c):
    """FP32 sigmoid router; bias for selection only; unbiased-score normalization."""
    logits = jnp.dot(x.astype(F32), w["router"].astype(F32), precision=HIGHEST)
    scores = jax.nn.sigmoid(logits * c.output_multiplier)
    ranked = scores + w["router_bias"].astype(F32)
    ids = jax.lax.top_k(ranked, c.top_k)[1]
    weights = jnp.take_along_axis(scores, ids, axis=-1)
    weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + c.router_eps)
    return weights, ids


def expert_outputs(pooled, ids, experts, c):
    """Batched BF16 expert math: [..., K] x top_k -> [..., top_k, K] BF16 rows."""
    w13 = experts["gate_up"][ids]  # [..., top_k, K, 2I]
    w2 = experts["down"][ids]  # [..., top_k, I, K]
    gate_up = jnp.einsum("...k,...tki->...ti", pooled, w13, preferred_element_type=F32)
    gate, up = jnp.split(gate_up, 2, axis=-1)
    activated = (jax.nn.silu(gate.astype(F32)) * up.astype(F32)).astype(pooled.dtype)
    rows = jnp.einsum("...ti,...tik->...tk", activated, w2, preferred_element_type=F32)
    return rows.astype(pooled.dtype)


def moe(x, w, c):
    """One MoE block on a BF16 branch input; returns the BF16 unpooled output."""
    weights, ids = route(x, w, c)
    pooled = linear(x, w["pool"])
    pooled = rms_gain(pooled, w["norm_pre_expert"], c.eps).astype(x.dtype)
    rows = expert_outputs(pooled, ids, w["experts"], c)
    rows = rms_gain_first(rows, w["norm_post_expert"], c.post_eps)
    mixed = jnp.sum(rows * weights.astype(F32)[..., None], axis=-2).astype(x.dtype)
    return linear(mixed, w["unpool"])


def attention_step(x, w, cache, position, c, index):
    """One GQA token; cache = (keys, values) [kv_heads, context, head_dim] BF16."""
    query = linear(x, w["q"]).reshape(c.heads, c.head_dim)
    key = linear(x, w["k"]).reshape(c.kv_heads, c.head_dim)
    value = linear(x, w["v"]).reshape(c.kv_heads, c.head_dim)
    gate = linear(x, w["gate"]).reshape(c.heads, c.head_dim)
    query = rms_plain(query, c.eps).astype(x.dtype)
    key = rms_plain(key, c.eps).astype(x.dtype)
    if c.use_rope(index):
        cos, sin = rope_tables(c, position[None])
        query = apply_rope(query, cos[0][None, :], sin[0][None, :]).astype(x.dtype)
        key = apply_rope(key, cos[0][None, :], sin[0][None, :]).astype(x.dtype)
    keys, values = cache
    keys, values = keys.at[:, position].set(key), values.at[:, position].set(value)
    grouped = query.reshape(c.kv_heads, c.heads // c.kv_heads, c.head_dim)
    scores = jnp.einsum("gjd,gtd->gjt", grouped.astype(F32), keys.astype(F32), precision=HIGHEST)
    scores = scores * c.scale
    slots = jnp.arange(keys.shape[1])
    keep = slots <= position
    if c.use_window(index):
        keep = keep & (slots > position - c.sliding_window)
    scores = jnp.where(keep, scores, -jnp.inf)
    probs = jax.nn.softmax(scores, axis=-1).astype(x.dtype)
    out = jnp.einsum("gjt,gtd->gjd", probs.astype(F32), values.astype(F32), precision=HIGHEST)
    out = out.reshape(c.heads, c.head_dim)
    out = rms_plain(out, c.eps) * jax.nn.sigmoid(gate.astype(F32))
    return linear(out.astype(x.dtype).reshape(-1), w["o"]), (keys, values)


def layer_step(stream, w, cache, position, c, index):
    """FP32 stream in/out; returns (stream, new_cache)."""
    x = rms_gain(stream, w["norm_in"], c.eps).astype(jnp.bfloat16)
    branch, cache = attention_step(x, w, cache, position, c, index)
    stream = w["alpha_a"] * stream + w["beta_a"] * rms_plain(branch, c.post_eps)
    x = rms_gain(stream, w["norm_pre_ffn"], c.eps).astype(jnp.bfloat16)
    branch = moe(x, w, c)
    stream = w["alpha_f"] * stream + w["beta_f"] * rms_gain_first(
        branch, w["norm_post_ffn"], c.post_eps
    )
    return stream, cache


def initial_states(c, context, dtype=jnp.bfloat16):
    return tuple(
        (
            jnp.zeros((c.kv_heads, context, c.head_dim), dtype),
            jnp.zeros((c.kv_heads, context, c.head_dim), dtype),
        )
        for _ in range(c.layers)
    )


def logits_from_stream(stream, weights, c):
    hidden = rms_gain(stream, weights["norm_final"], c.eps).astype(jnp.bfloat16)
    logits = jnp.dot(hidden, weights["lm_head"], preferred_element_type=F32)
    logits = logits * c.output_multiplier
    return c.soft_cap * jnp.tanh(logits / c.soft_cap)


def forward(token, weights, states, position, c, return_hidden=False):
    """Full decoder oracle with the FP32 residual stream."""
    stream = rms_plain(weights["embedding"][token], c.eps)
    new_states = []
    for index, (w, state) in enumerate(zip(weights["layers"], states, strict=True)):
        stream, state = layer_step(stream, w, state, position, c, index)
        new_states.append(state)
    logits = logits_from_stream(stream, weights, c)
    if return_hidden:
        return logits, tuple(new_states), stream
    return logits, tuple(new_states)


def generate(weights, prompt, steps, c, context):
    """Greedy token-by-token generation; returns the produced token ids."""
    states = initial_states(c, context)
    tokens = []
    todo = list(prompt) + [None] * steps
    for position, token in enumerate(todo):
        if token is None:
            token = int(jnp.argmax(logits))
            tokens.append(token)
        logits, states = forward(token, weights, states, jnp.array(position, jnp.int32), c)
    return tokens


def attention_prefill(x, w, npad, c, index):
    """Batched causal GQA over B padded rows; returns (out, (key, value))."""
    L = x.shape[0]
    query = linear(x, w["q"]).reshape(L, c.heads, c.head_dim)
    key = linear(x, w["k"]).reshape(L, c.kv_heads, c.head_dim)
    value = linear(x, w["v"]).reshape(L, c.kv_heads, c.head_dim)
    gate = linear(x, w["gate"]).reshape(L, c.heads, c.head_dim)
    query = rms_plain(query, c.eps).astype(x.dtype)
    key = rms_plain(key, c.eps).astype(x.dtype)
    positions = jnp.arange(L, dtype=jnp.int32) - npad
    if c.use_rope(index):
        cos, sin = rope_tables(c, positions)
        query = apply_rope(query, cos[:, None, :], sin[:, None, :]).astype(x.dtype)
        key = apply_rope(key, cos[:, None, :], sin[:, None, :]).astype(x.dtype)
    grouped = query.transpose(1, 0, 2).reshape(c.kv_heads, c.heads // c.kv_heads, L, c.head_dim)
    keys_t = key.transpose(1, 0, 2)
    scores = jnp.einsum("gjld,gtd->gjlt", grouped.astype(F32), keys_t.astype(F32))
    scores = scores * c.scale
    real = jnp.arange(L) >= npad
    key_abs = jnp.arange(L) - npad  # absolute position of the token at row j
    keep = (key_abs[None, :] <= positions[:, None]) & (key_abs[None, :] >= 0)
    if c.use_window(index):
        keep = keep & (key_abs[None, :] > (positions - c.sliding_window)[:, None])
    scores = jnp.where(keep[None, None], scores, -jnp.inf)
    scores = jnp.where(real[None, None, :, None], scores, 0.0)
    probs = jax.nn.softmax(scores, axis=-1).astype(x.dtype)
    out = jnp.einsum("gjlt,gtd->gjld", probs.astype(F32), value.transpose(1, 0, 2).astype(F32))
    out = out.reshape(c.heads, L, c.head_dim).transpose(1, 0, 2)
    out = rms_plain(out, c.eps) * jax.nn.sigmoid(gate.astype(F32))
    return linear(out.astype(x.dtype).reshape(L, -1), w["o"]), (key, value)


def prefill(weights, states, tokens, c, npad=0):
    """Padded-prompt oracle; states get the prompt's KV written at [0, L)."""
    L = len(tokens)
    real = jnp.arange(L) >= npad
    positions = jnp.arange(L, dtype=jnp.int32) - npad
    stream = rms_plain(weights["embedding"][jnp.asarray(tokens)], c.eps)
    stream = jnp.where(real[:, None], stream, jnp.zeros((), stream.dtype))
    new_states = []
    for index, (w, state) in enumerate(zip(weights["layers"], states, strict=True)):
        x = rms_gain(stream, w["norm_in"], c.eps).astype(jnp.bfloat16)
        branch, (key, value) = attention_prefill(x, w, npad, c, index)
        stream = w["alpha_a"] * stream + w["beta_a"] * rms_plain(branch, c.post_eps)
        x = rms_gain(stream, w["norm_pre_ffn"], c.eps).astype(jnp.bfloat16)
        weights_r, ids = route(x, w, c)
        pooled = jnp.dot(x, w["pool"], preferred_element_type=F32).astype(x.dtype)
        pooled = rms_gain(pooled, w["norm_pre_expert"], c.eps).astype(x.dtype)
        rows = expert_outputs(pooled, ids, w["experts"], c)
        rows = rms_gain_first(rows, w["norm_post_expert"], c.post_eps)
        mixed = jnp.sum(rows * weights_r.astype(F32)[..., None], axis=-2).astype(x.dtype)
        branch = jnp.dot(mixed, w["unpool"], preferred_element_type=F32).astype(x.dtype)
        stream = jnp.where(real[:, None], stream, jnp.zeros((), stream.dtype))
        branch = jnp.where(real[:, None], branch, jnp.zeros((), branch.dtype))
        stream = w["alpha_f"] * stream + w["beta_f"] * rms_gain_first(
            branch, w["norm_post_ffn"], c.post_eps
        )
        stream = jnp.where(real[:, None], stream, jnp.zeros((), stream.dtype))
        kc, vc = state
        slots = positions  # absolute slot per row; pad rows drop out
        kc = kc.at[:, slots].set(key.transpose(1, 0, 2).astype(kc.dtype), mode="drop")
        vc = vc.at[:, slots].set(value.transpose(1, 0, 2).astype(vc.dtype), mode="drop")
        new_states.append((kc, vc))
    last = stream[-1]
    logits = logits_from_stream(last, weights, c)
    return tuple(new_states), logits
