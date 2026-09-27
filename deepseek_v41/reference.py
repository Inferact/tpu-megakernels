"""Pure-JAX reference implementation of DeepSeek-V4.1-Flash and DSpark.

Matches `REF/model.py`, `REF/kernel.py`, and `REF/engram.py`:
- Single-pass 4-stream mHC with 20-iteration Sinkhorn-Knopp (`sinkhorn_iters` configurable).
- All 6 backbone attention layer kinds (`SWA`, `FULL_R2`, `REUSE_R2`, `FULL_R1_CAND`, `REINDEX_R1`,
  `REUSE_R1`) plus `DSPARK` (`mtp.0..n_mtp_layers-1`).
- Both `index_k_mode="reference"` (`REF/model.py:537-554`) and `index_k_mode="intended"`.
- Single-token decode, full-sequence prefill, and multi-token block verification (`[B, q]` with
  causal sliding-window + compressed masks and non-destructive rollback).
- DSpark 3-layer bidirectional draft block and sequential Markov chain head.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from deepseek_v41.config import DSV41Config
from deepseek_v41.engram_hash import EngramHashState, EngramLayout, SyntheticTokenizer
from deepseek_v41.quant import (
    act_quant_jax,
    dequant_fp8_block32x32_jax,
    dequant_mxfp4_block32_jax,
    e8m0_to_f32_jax,
    fp4_act_quant_jax,
)
from deepseek_v41.rope import apply_rotary_emb_jax, precompute_layer_rope_tables


@functools.partial(jax.jit, static_argnames=("eps",))
def rms_norm(x: jax.Array, weight: jax.Array, eps: float = 1e-20) -> jax.Array:
    """RMSNorm in `float32` rounded back to `x.dtype` (`REF/model.py:281-294`)."""
    orig_dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    y = x_f32 * jax.lax.rsqrt(var + jnp.float32(eps))
    w_f32 = weight.astype(jnp.bfloat16).astype(jnp.float32)
    return (y * w_f32).astype(orig_dtype)


@functools.partial(jax.jit, static_argnames=("hc_mult", "sinkhorn_iters", "eps"))
def hc_split_sinkhorn(
    mixes: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Splits `mixes` (`[..., 24]`) into `(pre, post, comb)` and runs Sinkhorn (`REF/kernel.py:406-474`).

    When `sinkhorn_iters <= 0`, skips row-softmax and Sinkhorn
    iterations on `comb`.
    """
    m = mixes.astype(jnp.float32)
    s = hc_scale.astype(jnp.float32)
    b = hc_base.astype(jnp.float32)
    eps_f32 = jnp.float32(eps)

    pre = jax.nn.sigmoid(m[..., :hc_mult] * s[0] + b[:hc_mult]) + eps_f32
    post = jnp.float32(2.0) * jax.nn.sigmoid(m[..., hc_mult : 2 * hc_mult] * s[1] + b[hc_mult : 2 * hc_mult])
    comb = (m[..., 2 * hc_mult :] * s[2] + b[2 * hc_mult :]).reshape(*m.shape[:-1], hc_mult, hc_mult)

    if sinkhorn_iters > 0:
        row_max = jnp.max(comb, axis=-1, keepdims=True)
        comb = jnp.exp(comb - row_max)
        comb = comb / jnp.sum(comb, axis=-1, keepdims=True) + eps_f32
        comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + eps_f32)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (jnp.sum(comb, axis=-1, keepdims=True) + eps_f32)
            comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + eps_f32)
    return pre, post, comb


@functools.partial(jax.jit, static_argnames=("hc_mult", "sinkhorn_iters", "eps", "norm_eps"))
def hc_mixes(
    x: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
    norm_eps: float = 1e-20,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Computes `(pre, post, comb)` from `x` (`[B, S, hc_mult, D]`) (`REF/model.py:948-955`)."""
    x_flat = x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1]).astype(jnp.float32)
    rsqrt = jax.lax.rsqrt(jnp.mean(jnp.square(x_flat), axis=-1, keepdims=True) + jnp.float32(norm_eps))
    mixes = jnp.einsum(
        "...d,md->...m",
        x_flat,
        hc_fn.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    ) * rsqrt
    return hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=hc_mult, sinkhorn_iters=sinkhorn_iters, eps=eps)


@jax.jit
def hc_pre(x: jax.Array, pre_mix: jax.Array) -> jax.Array:
    """Collapses `hc_mult` copies `[B, S, hc_mult, D]` -> `[B, S, D]` (`REF/model.py:957-960`)."""
    y = jnp.sum(pre_mix[..., None].astype(jnp.float32) * x.astype(jnp.float32), axis=-2)
    return y.astype(x.dtype)


@jax.jit
def hc_post(
    x: jax.Array,
    residual: jax.Array,
    post: jax.Array,
    comb: jax.Array,
) -> jax.Array:
    """Expands `x` (`[B, S, D]`) and mixes `residual` (`[B, S, hc, D]`) through `comb` (`REF/model.py:962-966`)."""
    y = post[..., None].astype(jnp.float32) * x[..., None, :].astype(jnp.float32) + jnp.sum(
        comb[..., None].astype(jnp.float32) * residual[..., :, None, :].astype(jnp.float32),
        axis=-3,
    )
    return y.astype(x.dtype)


@functools.partial(jax.jit, static_argnames=("block_size", "out_dtype"))
def fp8_linear(
    x: jax.Array,
    weight: jax.Array,
    scale: jax.Array,
    block_size: int = 32,
    out_dtype: Any = jnp.bfloat16,
) -> jax.Array:
    """Quantizes `x` per-32 with `act_quant` and runs block-scaled FP8 GEMM (`REF/model.py:196-205`)."""
    x_q, x_s = act_quant_jax(x, block_size=block_size, scale_fmt="ue8m0", inplace=False)
    x_deq = x_q.astype(jnp.float32) * jnp.repeat(x_s, block_size, axis=-1)
    w_deq = dequant_fp8_block32x32_jax(weight, scale, block_size=block_size, out_dtype=jnp.float32)
    out = jnp.einsum("...k,nk->...n", x_deq, w_deq, precision=jax.lax.Precision.HIGHEST)
    return out.astype(out_dtype)


@functools.partial(jax.jit, static_argnames=("block_size", "out_dtype"))
def fp4_linear(
    x: jax.Array,
    weight: jax.Array,
    scale: jax.Array,
    block_size: int = 32,
    out_dtype: Any = jnp.bfloat16,
) -> jax.Array:
    """Quantizes `x` per-32 with `act_quant` and runs MXFP4 GEMM (`REF/model.py:186-195`)."""
    x_q, x_s = act_quant_jax(x, block_size=block_size, scale_fmt="ue8m0", inplace=False)
    x_deq = x_q.astype(jnp.float32) * jnp.repeat(x_s, block_size, axis=-1)
    w_deq = dequant_mxfp4_block32_jax(weight, scale, block_size=32, out_dtype=jnp.float32)
    out = jnp.einsum("...k,nk->...n", x_deq, w_deq, precision=jax.lax.Precision.HIGHEST)
    return out.astype(out_dtype)


@functools.partial(jax.jit, static_argnames=("out_dtype",))
def bf16_linear(x: jax.Array, weight: jax.Array, out_dtype: Any = None) -> jax.Array:
    """Unquantized linear projection with `float32` accumulation (`REF/model.py:207`)."""
    target_dtype = x.dtype if out_dtype is None else out_dtype
    out = jnp.einsum(
        "...k,nk->...n",
        x.astype(jnp.float32),
        weight.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )
    return out.astype(target_dtype)


@functools.partial(jax.jit, static_argnames=("softmax_scale",))
def sparse_attn(
    q: jax.Array,
    kv: jax.Array,
    attn_sink: jax.Array,
    topk_idxs: jax.Array,
    softmax_scale: float | None = None,
) -> jax.Array:
    """Online 64-block sparse MQA latent attention matching `REF/kernel.py:310-403`."""
    b, m, h, d = q.shape
    if softmax_scale is None:
        softmax_scale = float(d**-0.5)
    topk = topk_idxs.shape[-1]
    block = 64
    num_blocks = (topk + block - 1) // block
    pad_k = num_blocks * block - topk
    if pad_k > 0:
        idxs_padded = jnp.pad(topk_idxs.astype(jnp.int32), ((0, 0), (0, 0), (0, pad_k)), constant_values=-1)
    else:
        idxs_padded = topk_idxs.astype(jnp.int32)

    q_bf16 = q.astype(jnp.bfloat16)
    kv_bf16 = kv.astype(jnp.bfloat16)
    acc_o = jnp.zeros((b, m, h, d), dtype=jnp.float32)
    sum_exp = jnp.zeros((b, m, h), dtype=jnp.float32)
    scores_max = jnp.full((b, m, h), -1e30, dtype=jnp.float32)
    batch_idx = jnp.arange(b, dtype=jnp.int32)[:, None, None]

    for t in range(num_blocks):
        blk_idxs = idxs_padded[:, :, t * block : (t + 1) * block]
        valid = blk_idxs != -1
        safe_idxs = jnp.maximum(blk_idxs, 0)
        kv_blk = kv_bf16[batch_idx, safe_idxs]  # [B, M, 64, D]
        kv_blk = jnp.where(valid[..., None], kv_blk, jnp.bfloat16(0.0))

        acc_s = (
            jnp.einsum(
                "bmhd,bmkd->bmhk",
                q_bf16.astype(jnp.float32),
                kv_blk.astype(jnp.float32),
                precision=jax.lax.Precision.HIGHEST,
            )
            * jnp.float32(softmax_scale)
        )
        acc_s = jnp.where(valid[:, :, None, :], acc_s, jnp.float32(-jnp.inf))

        scores_max_prev = scores_max
        blk_max = jnp.max(acc_s, axis=-1)
        scores_max = jnp.maximum(scores_max_prev, blk_max)
        scores_scale = jnp.exp(scores_max_prev - scores_max)
        p_f32 = jnp.exp(acc_s - scores_max[..., None])
        scores_sum = jnp.sum(p_f32, axis=-1)
        sum_exp = sum_exp * scores_scale + scores_sum

        p_bf16 = p_f32.astype(jnp.bfloat16)
        acc_o = acc_o * scores_scale[..., None] + jnp.einsum(
            "bmhk,bmkd->bmhd",
            p_bf16.astype(jnp.float32),
            kv_blk.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        )

    sum_exp = sum_exp + jnp.exp(attn_sink.astype(jnp.float32).reshape(1, 1, h) - scores_max)
    return (acc_o / sum_exp[..., None]).astype(q.dtype)


def select_candidate_blocks(
    logits: jax.Array,
    compress_lens: jax.Array | int,
    topk_blocks: int,
    block_size: int,
) -> jax.Array:
    """Level-1 candidate block selection matching `REF/model.py:583-610`."""
    width = logits.shape[-1]
    pad_len = (-width) % block_size
    if pad_len > 0:
        padded = jnp.pad(
            logits.astype(jnp.float32),
            [(0, 0)] * (logits.ndim - 1) + [(0, pad_len)],
            constant_values=-jnp.inf,
        )
    else:
        padded = logits.astype(jnp.float32)
    scores = jnp.max(padded.reshape(*logits.shape[:-1], -1, block_size), axis=-1)
    num_blocks = scores.shape[-1]

    last = (jnp.asarray(compress_lens, dtype=jnp.int32) - 1) // block_size
    block_arange = jnp.arange(num_blocks, dtype=jnp.int32)
    scores = jnp.where(block_arange == last, jnp.float32(jnp.inf), scores)

    k = min(topk_blocks, num_blocks)
    top_vals, top_idxs = jax.lax.top_k(scores, k)
    valid_top = top_vals > -jnp.inf

    # Scatter valid_top into boolean mask of shape scores.shape.
    one_hot = jnp.arange(num_blocks, dtype=jnp.int32) == top_idxs[..., None]  # [..., k, num_blocks]
    keep = jnp.any(one_hot & valid_top[..., None], axis=-2)
    return jnp.repeat(keep, block_size, axis=-1)[..., :width]


def engram_forward(
    x: jax.Array,
    hash_ids: jax.Array,
    weights: dict[str, Any],
    l_prefix: str,
    cfg: DSV41Config,
    num_embeddings: int,
    token_mask: jax.Array | None = None,
) -> jax.Array:
    """Engram table lookup and gated residual addition matching `REF/model.py:296-366`."""
    indices = jnp.asarray(hash_ids, dtype=jnp.int32)
    mask = (indices < 0) | (indices >= num_embeddings)
    safe_indices = jnp.where(mask, 0, indices)

    emb_w = weights[f"{l_prefix}.engram.embed.weight"]
    emb_s = weights[f"{l_prefix}.engram.embed.scale"]
    vals_fp8 = emb_w[safe_indices].astype(jnp.float32)
    scales_f32 = e8m0_to_f32_jax(emb_s[safe_indices])
    vals = (vals_fp8.reshape(*vals_fp8.shape[:-1], -1, 32) * scales_f32[..., None]).reshape(vals_fp8.shape)
    vals_bf16 = jnp.where(mask[..., None], jnp.bfloat16(0.0), vals.astype(jnp.bfloat16))
    flat_e = vals_bf16.reshape(*vals_bf16.shape[:-2], vals_bf16.shape[-2] * vals_bf16.shape[-1])

    kv = fp8_linear(
        flat_e,
        weights[f"{l_prefix}.engram.wkv.weight"],
        weights[f"{l_prefix}.engram.wkv.scale"],
        block_size=32,
    )
    hc_dim = cfg.hc_mult * cfg.dim
    key = kv[..., :hc_dim].astype(jnp.float32).reshape(*x.shape[:-2], cfg.hc_mult, cfg.dim)
    value = kv[..., hc_dim:]

    q_w = weights[f"{l_prefix}.engram.q_weight"].astype(jnp.float32)
    k_w = weights[f"{l_prefix}.engram.k_weight"].astype(jnp.float32)
    wgt = q_w * k_w

    h = x.astype(jnp.float32)
    eps = jnp.float32(cfg.rms_norm_eps)
    rstd = jax.lax.rsqrt(jnp.mean(jnp.square(h), axis=-1) + eps) * jax.lax.rsqrt(
        jnp.mean(jnp.square(key), axis=-1) + eps
    )
    dot = jnp.sum(h * wgt * key, axis=-1) * rstd * jnp.float32(cfg.dim**-0.5)
    signed_sqrt = jnp.copysign(jnp.sqrt(jnp.maximum(jnp.abs(dot), jnp.float32(1e-6))), dot)
    gate = jax.nn.sigmoid(signed_sqrt)
    if token_mask is not None:
        gate = jnp.where(token_mask[..., None], gate, jnp.float32(0.0))
    out = h + gate[..., None] * value.astype(jnp.float32)[..., None, :]
    return out.astype(x.dtype)


def _softplus_f32(x: jax.Array) -> jax.Array:
    """Matches `torch.nn.functional.softplus(x, beta=1.0, threshold=20.0)` in `float32`."""
    return jnp.where(x > jnp.float32(20.0), x, jnp.log1p(jnp.exp(x)))


@functools.partial(jax.jit, static_argnames=("swiglu_limit", "is_fp4"))
def _expert_swiglu(
    x_bf16: jax.Array,
    w1_w: jax.Array,
    w1_s: jax.Array,
    w2_w: jax.Array,
    w2_s: jax.Array,
    w3_w: jax.Array,
    w3_s: jax.Array,
    swiglu_limit: float,
    routing_weight: jax.Array | None = None,
    *,
    is_fp4: bool = True,
) -> jax.Array:
    """Single SwiGLU expert (`REF/model.py:830-852`)."""
    lin_fn = fp4_linear if is_fp4 else fp8_linear
    gate = lin_fn(x_bf16, w1_w, w1_s, block_size=32).astype(jnp.float32)
    up = lin_fn(x_bf16, w3_w, w3_s, block_size=32).astype(jnp.float32)
    if swiglu_limit > 0:
        lim = jnp.float32(swiglu_limit)
        up = jnp.clip(up, -lim, lim)
        gate = jnp.minimum(gate, lim)
    h = jax.nn.silu(gate) * up
    if routing_weight is not None:
        h = routing_weight.astype(jnp.float32) * h
    return lin_fn(h.astype(jnp.bfloat16), w2_w, w2_s, block_size=32)


def moe_forward(
    x: jax.Array,
    weights: dict[str, Any],
    l_prefix: str,
    cfg: DSV41Config,
    layer_id: int,
) -> jax.Array:
    """MoE router + top-k FP4 routed experts + FP8 shared expert (`REF/model.py:854-905`)."""
    orig_shape = x.shape
    tokens = x.reshape(-1, cfg.dim)
    num_tokens = tokens.shape[0]
    is_backbone = layer_id < cfg.n_layers
    n_routed = cfg.n_routed_experts if is_backbone else cfg.dspark_n_routed_experts
    topk = cfg.n_activated_experts if is_backbone else cfg.dspark_n_activated_experts

    gate_w = weights[f"{l_prefix}.ffn.gate.weight"].astype(jnp.float32)
    gate_b = weights[f"{l_prefix}.ffn.gate.bias"].astype(jnp.float32)
    raw_logits = jnp.einsum(
        "td,ed->te",
        tokens.astype(jnp.float32),
        gate_w,
        precision=jax.lax.Precision.HIGHEST,
    )
    scores = jnp.sqrt(_softplus_f32(raw_logits))
    _, top_indices = jax.lax.top_k(scores + gate_b[None, :], topk)
    top_weights = jnp.take_along_axis(scores, top_indices, axis=-1)
    if topk > 1:
        top_weights = top_weights / (jnp.sum(top_weights, axis=-1, keepdims=True) + jnp.float32(1e-20))
    top_weights = top_weights * jnp.float32(cfg.route_scale)

    y_f32 = jnp.zeros((num_tokens, cfg.dim), dtype=jnp.float32)
    indices_np = np.asarray(top_indices)
    for e in range(n_routed):
        tok_idx, slot_idx = np.where(indices_np == e)
        if tok_idx.size == 0:
            continue
        tok_j = jnp.asarray(tok_idx, dtype=jnp.int32)
        slot_j = jnp.asarray(slot_idx, dtype=jnp.int32)
        x_e = tokens[tok_j]
        w_e = top_weights[tok_j, slot_j][:, None]
        ep = f"{l_prefix}.ffn.experts.{e}"
        out_e = _expert_swiglu(
            x_e,
            weights[f"{ep}.w1.weight"],
            weights[f"{ep}.w1.scale"],
            weights[f"{ep}.w2.weight"],
            weights[f"{ep}.w2.scale"],
            weights[f"{ep}.w3.weight"],
            weights[f"{ep}.w3.scale"],
            cfg.swiglu_limit,
            routing_weight=w_e,
            is_fp4=True,
        ).astype(jnp.float32)
        y_f32 = y_f32.at[tok_j].add(out_e)

    sp = f"{l_prefix}.ffn.shared_experts"
    shared_out = _expert_swiglu(
        tokens,
        weights[f"{sp}.w1.weight"],
        weights[f"{sp}.w1.scale"],
        weights[f"{sp}.w2.weight"],
        weights[f"{sp}.w2.scale"],
        weights[f"{sp}.w3.weight"],
        weights[f"{sp}.w3.scale"],
        cfg.swiglu_limit,
        routing_weight=None,
        is_fp4=False,
    ).astype(jnp.float32)
    return (y_f32 + shared_out).astype(x.dtype).reshape(orig_shape)


@dataclass
class DSV41ReferenceState:
    """Mutable KV, compressor, and Engram hash state for `DeepSeekV41Reference`."""

    batch_size: int
    max_seq_len: int
    window_kv_cache: dict[int, jax.Array] = field(default_factory=dict)
    compress_kv_cache: dict[int, jax.Array] = field(default_factory=dict)
    index_k_cache: dict[int, jax.Array] = field(default_factory=dict)
    compressor_kv_state: dict[int, jax.Array] = field(default_factory=dict)
    compressor_score_state: dict[int, jax.Array] = field(default_factory=dict)
    engram_hash_state: EngramHashState | None = None
    shared_index_k_source: int | None = None

    def clone(self) -> DSV41ReferenceState:
        return DSV41ReferenceState(
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            window_kv_cache={k: jnp.array(v, copy=True) for k, v in self.window_kv_cache.items()},
            compress_kv_cache={k: jnp.array(v, copy=True) for k, v in self.compress_kv_cache.items()},
            index_k_cache={k: jnp.array(v, copy=True) for k, v in self.index_k_cache.items()},
            compressor_kv_state={k: jnp.array(v, copy=True) for k, v in self.compressor_kv_state.items()},
            compressor_score_state={k: jnp.array(v, copy=True) for k, v in self.compressor_score_state.items()},
            engram_hash_state=copy.deepcopy(self.engram_hash_state),
            shared_index_k_source=self.shared_index_k_source,
        )


class DeepSeekV41Reference:
    """Pure-JAX reference implementation of DeepSeek-V4.1-Flash and DSpark (`INVENTORY.md`)."""

    def __init__(
        self,
        cfg: DSV41Config,
        weights: dict[str, Any],
        *,
        max_batch_size: int = 4,
        max_seq_len: int = 256,
        index_k_mode: str | None = None,
        sinkhorn_iters: int | None = None,
        tokenizer: Any = None,
    ):
        self.cfg = cfg
        self.weights = weights
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.index_k_mode = index_k_mode if index_k_mode is not None else cfg.index_k_mode
        self.sinkhorn_iters = sinkhorn_iters if sinkhorn_iters is not None else cfg.hc_sinkhorn_iters
        self.engram_layout = EngramLayout.from_config(cfg)
        if tokenizer is None and self.engram_layout is not None:
            tokenizer = SyntheticTokenizer(cfg.vocab_size, cfg.vocab_size if cfg.vocab_size != 129280 else None)
        self.tokenizer = tokenizer
        self.rope_tables = precompute_layer_rope_tables(cfg, max_seq_len)
        self.state = self.init_state(max_batch_size, max_seq_len)

    def init_state(
        self,
        batch_size: int | None = None,
        max_seq_len: int | None = None,
    ) -> DSV41ReferenceState:
        bsz = batch_size or self.max_batch_size
        seq_len = max_seq_len or self.max_seq_len
        cfg = self.cfg
        if seq_len > self.rope_tables["swa"][0].shape[0]:
            self.rope_tables = precompute_layer_rope_tables(cfg, seq_len)

        win_kv: dict[int, jax.Array] = {}
        comp_kv: dict[int, jax.Array] = {}
        idx_k: dict[int, jax.Array] = {}
        comp_kv_st: dict[int, jax.Array] = {}
        comp_sc_st: dict[int, jax.Array] = {}

        total_layers = cfg.n_layers + cfg.n_mtp_layers
        for lid in range(total_layers):
            win_kv[lid] = jnp.zeros((bsz, cfg.sliding_window, cfg.head_dim), dtype=jnp.bfloat16)
            if lid < cfg.n_layers and lid in cfg.kv_source_layer_ids:
                r = cfg.compress_ratios[lid]
                comp_kv[lid] = jnp.zeros((bsz, seq_len // r, cfg.head_dim), dtype=jnp.bfloat16)
                idx_k[lid] = jnp.zeros((bsz, seq_len // r, cfg.index_head_dim), dtype=jnp.bfloat16)
                if r > 1:
                    comp_kv_st[lid] = jnp.zeros((bsz, r, cfg.head_dim), dtype=jnp.float32)
                    comp_sc_st[lid] = jnp.full((bsz, r, cfg.head_dim), -jnp.inf, dtype=jnp.float32)

        eng_state = (
            EngramHashState(cfg, self.engram_layout, self.tokenizer, max_batch_size=bsz, max_seq_len=seq_len)
            if self.engram_layout is not None
            else None
        )
        return DSV41ReferenceState(
            batch_size=bsz,
            max_seq_len=seq_len,
            window_kv_cache=win_kv,
            compress_kv_cache=comp_kv,
            index_k_cache=idx_k,
            compressor_kv_state=comp_kv_st,
            compressor_score_state=comp_sc_st,
            engram_hash_state=eng_state,
            shared_index_k_source=None,
        )

    def reset(self) -> None:
        self.state = self.init_state(self.max_batch_size, self.max_seq_len)

    def _get_layer_rope(self, layer_id: int) -> tuple[jax.Array, jax.Array]:
        r = self.cfg.compress_ratios[layer_id]
        return self.rope_tables["yarn"] if r > 0 else self.rope_tables["swa"]

    def _window_topk_idxs(self, bsz: int, seqlen: int, start_pos: int) -> jax.Array:
        win = self.cfg.sliding_window
        if start_pos == 0:
            end = jnp.arange(seqlen, dtype=jnp.int32)[:, None]
            idxs = jnp.maximum(end - win + 1, 0) + jnp.arange(min(seqlen, win), dtype=jnp.int32)[None, :]
            idxs = jnp.where(idxs > end, -1, idxs)
        else:
            oldest = start_pos % win + 1
            idxs = jnp.concatenate(
                [jnp.arange(oldest, win, dtype=jnp.int32), jnp.arange(oldest, dtype=jnp.int32)]
            )[None, :]
            idxs = jnp.where(idxs > start_pos, -1, idxs)
        return jnp.broadcast_to(idxs[None, :, :], (bsz, seqlen, idxs.shape[-1]))

    def _compressor_step(
        self,
        layer_id: int,
        x: jax.Array,
        start_pos: int,
        state: DSV41ReferenceState,
    ) -> jax.Array | None:
        """Runs `Compressor.forward` (`REF/model.py:458-485`) for `[B, S, D]`."""
        cfg = self.cfg
        bsz, seqlen, _ = x.shape
        r = cfg.compress_ratios[layer_id]
        l_prefix = f"layers.{layer_id}"
        wkv_w = self.weights[f"{l_prefix}.attn.compressor.wkv.weight"]
        norm_w = self.weights[f"{l_prefix}.attn.compressor.norm.weight"]

        if r == 1:
            return rms_norm(bf16_linear(x, wkv_w), norm_w, cfg.rms_norm_eps)

        wgate_w = self.weights[f"{l_prefix}.attn.compressor.wgate.weight"]
        kv = bf16_linear(x.astype(jnp.float32), wkv_w.astype(jnp.float32), out_dtype=jnp.float32)
        score = bf16_linear(x.astype(jnp.float32), wgate_w.astype(jnp.float32), out_dtype=jnp.float32)

        if start_pos == 0:
            should_compress = seqlen >= r
            remainder = seqlen % r
            cutoff = seqlen - remainder
            if remainder:
                state.compressor_kv_state[layer_id] = (
                    state.compressor_kv_state[layer_id].at[:bsz, :remainder].set(kv[:, cutoff:])
                )
                state.compressor_score_state[layer_id] = (
                    state.compressor_score_state[layer_id].at[:bsz, :remainder].set(score[:, cutoff:])
                )
                kv = kv[:, :cutoff]
                score = score[:, :cutoff]
            if not should_compress:
                return None
            kv_grp = kv.reshape(bsz, -1, r, cfg.head_dim)
            sc_grp = score.reshape(bsz, -1, r, cfg.head_dim)
            pooled = jnp.sum(kv_grp * jax.nn.softmax(sc_grp, axis=2), axis=2)
            return rms_norm(pooled.astype(x.dtype), norm_w, cfg.rms_norm_eps)

        # Decode (`seqlen == 1`) at `start_pos > 0`.
        should_compress = (start_pos + 1) % r == 0
        slot = start_pos % r
        state.compressor_kv_state[layer_id] = (
            state.compressor_kv_state[layer_id].at[:bsz, slot].set(kv[:, 0])
        )
        state.compressor_score_state[layer_id] = (
            state.compressor_score_state[layer_id].at[:bsz, slot].set(score[:, 0])
        )
        if not should_compress:
            return None
        kv_st = state.compressor_kv_state[layer_id][:bsz]
        sc_st = state.compressor_score_state[layer_id][:bsz]
        pooled = jnp.sum(kv_st * jax.nn.softmax(sc_st, axis=1), axis=1, keepdims=True)
        return rms_norm(pooled.astype(x.dtype), norm_w, cfg.rms_norm_eps)

    def _indexer_step(
        self,
        layer_id: int,
        x: jax.Array,
        qr: jax.Array,
        latent: jax.Array | None,
        start_pos: int,
        offset: int,
        state: DSV41ReferenceState,
        shared_runtime: dict[str, Any],
        index_k_mode: str,
    ) -> jax.Array:
        """Runs `Indexer.forward` (`REF/model.py:527-580`)."""
        cfg = self.cfg
        bsz, seqlen, _ = x.shape
        r = cfg.compress_ratios[layer_id]
        end_pos = start_pos + seqlen
        l_prefix = f"layers.{layer_id}"
        cos_all, sin_all = self._get_layer_rope(layer_id)
        owns_k = layer_id in cfg.kv_source_layer_ids

        if owns_k and latent is not None:
            if start_pos == 0:
                cos_k = cos_all[: seqlen - seqlen % r : r]
                sin_k = sin_all[: seqlen - seqlen % r : r]
            else:
                grp_pos = start_pos + 1 - r
                cos_k = cos_all[grp_pos : grp_pos + 1]
                sin_k = sin_all[grp_pos : grp_pos + 1]
            wk_w = self.weights[f"{l_prefix}.attn.indexer.wk.weight"]
            knorm_w = self.weights[f"{l_prefix}.attn.indexer.k_norm.weight"]
            k = rms_norm(bf16_linear(latent, wk_w), knorm_w, cfg.rms_norm_eps)
            k = apply_rotary_emb_jax(k, (cos_k, sin_k), inverse=False)
            k = fp4_act_quant_jax(k, block_size=32, inplace=True, scale_dtype="e8m0")
            c_start = start_pos // r
            state.index_k_cache[layer_id] = (
                state.index_k_cache[layer_id].at[:bsz, c_start : c_start + k.shape[1]].set(k)
            )
            state.shared_index_k_source = layer_id

        if index_k_mode == "intended":
            kv_src = cfg.kv_source_for(layer_id)
            assert kv_src is not None
            k_source_cache = state.index_k_cache[kv_src]
        else:
            src_id = state.shared_index_k_source
            if src_id is None:
                src_id = cfg.kv_source_for(layer_id)
            assert src_id is not None
            k_source_cache = state.index_k_cache[src_id]

        wq_b_w = self.weights[f"{l_prefix}.attn.indexer.wq_b.weight"]
        wq_b_s = self.weights[f"{l_prefix}.attn.indexer.wq_b.scale"]
        q = fp8_linear(qr, wq_b_w, wq_b_s, block_size=32).reshape(
            bsz, seqlen, cfg.index_n_heads, cfg.index_head_dim
        )
        q = apply_rotary_emb_jax(q, (cos_all[start_pos:end_pos], sin_all[start_pos:end_pos]), inverse=False)
        q = fp4_act_quant_jax(q, block_size=32, inplace=True, scale_dtype="e8m0")

        index_k = k_source_cache[:bsz, : end_pos // r]
        w_proj = self.weights[f"{l_prefix}.attn.indexer.weights_proj.weight"]
        scale_w = float((cfg.index_head_dim**-0.5) * (cfg.index_n_heads**-0.5))
        weights = (bf16_linear(x, w_proj).astype(jnp.float32) * jnp.float32(scale_w)).astype(jnp.bfloat16)

        # Match `REF/model.py:556-557` rounding points (einsum -> bf16, relu, * weights -> bf16, sum -> bf16).
        raw_score = jnp.einsum(
            "bshd,btd->bsht",
            q.astype(jnp.float32),
            index_k.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        ).astype(jnp.bfloat16)
        gated = (jnp.maximum(raw_score, jnp.bfloat16(0.0)).astype(jnp.float32) * weights[..., None].astype(jnp.float32)).astype(
            jnp.bfloat16
        )
        index_score = jnp.sum(gated.astype(jnp.float32), axis=2).astype(jnp.bfloat16)

        if start_pos == 0:
            compress_lens = (jnp.arange(1, seqlen + 1, dtype=jnp.int32) // r)[:, None]
            t_arange = jnp.arange(seqlen // r, dtype=jnp.int32)[None, :]
            index_score = jnp.where(t_arange >= compress_lens, jnp.bfloat16(-jnp.inf), index_score)
        else:
            compress_lens = end_pos // r

        if layer_id == cfg.candidate_source_layer_id:
            shared_runtime["candidates"] = select_candidate_blocks(
                index_score,
                compress_lens,
                cfg.candidate_topk_blocks,
                cfg.candidate_block_size,
            )
        elif 0 <= cfg.candidate_source_layer_id < layer_id:
            cand_mask = shared_runtime["candidates"]
            index_score = jnp.where(cand_mask, index_score, jnp.bfloat16(-jnp.inf))

        topk = min(cfg.index_topk, end_pos // r)
        _, top_idxs = jax.lax.top_k(index_score.astype(jnp.float32), topk)
        sorted_idxs = jnp.sort(top_idxs, axis=-1)
        return jnp.where(sorted_idxs < compress_lens, sorted_idxs + offset, -1).astype(jnp.int32)

    def _attention_step(
        self,
        layer_id: int,
        x: jax.Array,
        start_pos: int,
        state: DSV41ReferenceState,
        shared_runtime: dict[str, Any],
        index_k_mode: str,
    ) -> jax.Array:
        """Runs `Attention.forward` (`REF/model.py:765-790`) for `[B, S, D]`."""
        cfg = self.cfg
        bsz, seqlen, _ = x.shape
        l_prefix = f"layers.{layer_id}"
        cos_all, sin_all = self._get_layer_rope(layer_id)
        cos_cur = cos_all[start_pos : start_pos + seqlen]
        sin_cur = sin_all[start_pos : start_pos + seqlen]

        qr = rms_norm(
            fp8_linear(x, self.weights[f"{l_prefix}.attn.wq_a.weight"], self.weights[f"{l_prefix}.attn.wq_a.scale"]),
            self.weights[f"{l_prefix}.attn.q_norm.weight"],
            cfg.rms_norm_eps,
        )
        q = fp8_linear(
            qr,
            self.weights[f"{l_prefix}.attn.wq_b.weight"],
            self.weights[f"{l_prefix}.attn.wq_b.scale"],
        ).reshape(bsz, seqlen, cfg.n_heads, cfg.head_dim)
        q = apply_rotary_emb_jax(q, (cos_cur, sin_cur), inverse=False)

        # Sliding-window KV (`REF/model.py:700-720`).
        win = cfg.sliding_window
        kv = rms_norm(
            fp8_linear(x, self.weights[f"{l_prefix}.attn.wkv.weight"], self.weights[f"{l_prefix}.attn.wkv.scale"]),
            self.weights[f"{l_prefix}.attn.kv_norm.weight"],
            cfg.rms_norm_eps,
        )
        kv = apply_rotary_emb_jax(kv, (cos_cur, sin_cur), inverse=False)
        kv = act_quant_jax(kv, block_size=32, scale_fmt="ue8m0", inplace=True)

        if start_pos == 0:
            if seqlen <= win:
                state.window_kv_cache[layer_id] = state.window_kv_cache[layer_id].at[:bsz, :seqlen].set(kv)
            else:
                cutoff = seqlen % win
                tail_win = kv[:, -win:]
                state.window_kv_cache[layer_id] = (
                    state.window_kv_cache[layer_id]
                    .at[:bsz, cutoff:win]
                    .set(tail_win[:, : win - cutoff])
                    .at[:bsz, :cutoff]
                    .set(tail_win[:, win - cutoff :])
                )
            window_kv = kv
        else:
            state.window_kv_cache[layer_id] = (
                state.window_kv_cache[layer_id].at[:bsz, start_pos % win].set(kv[:, 0])
            )
            window_kv = state.window_kv_cache[layer_id][:bsz]

        topk_idxs = self._window_topk_idxs(bsz, seqlen, start_pos)

        r = cfg.compress_ratios[layer_id]
        if r > 0:
            compress_len = (start_pos + seqlen) // r
            latent = None
            if layer_id in cfg.kv_source_layer_ids:
                latent = self._compressor_step(layer_id, x, start_pos, state)
                shared_runtime["compress_kv_source"] = layer_id

            if layer_id in cfg.index_source_layer_ids:
                if compress_len == 0:
                    compress_idxs = jnp.zeros((bsz, seqlen, 0), dtype=jnp.int32)
                else:
                    compress_idxs = self._indexer_step(
                        layer_id,
                        x,
                        qr,
                        latent,
                        start_pos,
                        window_kv.shape[1],
                        state,
                        shared_runtime,
                        index_k_mode,
                    )
                shared_runtime["topk_idxs"] = compress_idxs
            else:
                compress_idxs = shared_runtime["topk_idxs"]

            if latent is not None:
                if start_pos == 0:
                    cos_c = cos_all[: seqlen - seqlen % r : r]
                    sin_c = sin_all[: seqlen - seqlen % r : r]
                else:
                    grp_pos = start_pos + 1 - r
                    cos_c = cos_all[grp_pos : grp_pos + 1]
                    sin_c = sin_all[grp_pos : grp_pos + 1]
                latent_rot = apply_rotary_emb_jax(latent, (cos_c, sin_c), inverse=False)
                latent_q = fp4_act_quant_jax(latent_rot, block_size=16, inplace=True, scale_dtype="e4m3")
                c_start = start_pos // r
                state.compress_kv_cache[layer_id] = (
                    state.compress_kv_cache[layer_id]
                    .at[:bsz, c_start : c_start + latent_q.shape[1]]
                    .set(latent_q)
                )

            kv_src = shared_runtime["compress_kv_source"]
            compress_kv = state.compress_kv_cache[kv_src][:bsz, :compress_len]
            kv_all = jnp.concatenate([window_kv, compress_kv], axis=1)
            topk_idxs = jnp.concatenate([topk_idxs, compress_idxs], axis=-1)
        else:
            kv_all = window_kv

        attn_sink = self.weights[f"{l_prefix}.attn.attn_sink"]
        o = sparse_attn(q, kv_all, attn_sink, topk_idxs, softmax_scale=float(cfg.head_dim**-0.5))
        o = apply_rotary_emb_jax(o, (cos_cur, sin_cur), inverse=True)

        o_grouped = o.reshape(bsz, seqlen, cfg.o_groups, -1)
        wo_a_raw = self.weights[f"{l_prefix}.attn.wo_a.weight"]
        wo_a_s_key = f"{l_prefix}.attn.wo_a.scale"
        if wo_a_s_key in self.weights and wo_a_raw.ndim == 2:
            wo_a_deq = dequant_fp8_block32x32_jax(wo_a_raw, self.weights[wo_a_s_key], out_dtype=jnp.bfloat16)
        else:
            wo_a_deq = wo_a_raw.astype(jnp.bfloat16)
        wo_a = wo_a_deq.reshape(cfg.o_groups, cfg.o_lora_rank, -1)
        o_proj = jnp.einsum(
            "bsgd,grd->bsgr",
            o_grouped.astype(jnp.float32),
            wo_a.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        ).astype(jnp.bfloat16)
        return fp8_linear(
            o_proj.reshape(bsz, seqlen, cfg.o_groups * cfg.o_lora_rank),
            self.weights[f"{l_prefix}.attn.wo_b.weight"],
            self.weights[f"{l_prefix}.attn.wo_b.scale"],
        )

    def _block_step(
        self,
        layer_id: int,
        x: jax.Array,
        start_pos: int,
        pre_mix: jax.Array,
        state: DSV41ReferenceState,
        shared_runtime: dict[str, Any],
        index_k_mode: str,
        sinkhorn_iters: int,
    ) -> tuple[jax.Array, jax.Array]:
        """Runs one backbone `Block.forward` (`REF/model.py:968-994`)."""
        cfg = self.cfg
        l_prefix = f"layers.{layer_id}"
        residual = x
        attn_pre, attn_post, attn_comb = hc_mixes(
            x,
            self.weights[f"{l_prefix}.hc_attn_fn"],
            self.weights[f"{l_prefix}.hc_attn_scale"],
            self.weights[f"{l_prefix}.hc_attn_base"],
            hc_mult=cfg.hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            eps=cfg.hc_eps,
            norm_eps=cfg.rms_norm_eps,
        )
        u = rms_norm(
            hc_pre(x, pre_mix),
            self.weights[f"{l_prefix}.attn_norm.weight"],
            cfg.rms_norm_eps,
        )
        attn_out = self._attention_step(layer_id, u, start_pos, state, shared_runtime, index_k_mode)
        x = hc_post(attn_out, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = hc_mixes(
            x,
            self.weights[f"{l_prefix}.hc_ffn_fn"],
            self.weights[f"{l_prefix}.hc_ffn_scale"],
            self.weights[f"{l_prefix}.hc_ffn_base"],
            hc_mult=cfg.hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            eps=cfg.hc_eps,
            norm_eps=cfg.rms_norm_eps,
        )
        v = rms_norm(
            hc_pre(x, attn_pre),
            self.weights[f"{l_prefix}.ffn_norm.weight"],
            cfg.rms_norm_eps,
        )
        ffn_out = moe_forward(v, self.weights, l_prefix, cfg, layer_id)
        x = hc_post(ffn_out, residual, ffn_post, ffn_comb)
        return x, ffn_pre

    def _forward_single_or_prefill(
        self,
        ids: jax.Array,
        start_pos: int,
        state: DSV41ReferenceState,
        *,
        full_logits: bool = False,
        index_k_mode: str,
        sinkhorn_iters: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array | None]:
        cfg = self.cfg
        bsz, seqlen = ids.shape
        engram_hashes = None
        if state.engram_hash_state is not None:
            engram_hashes = jnp.asarray(state.engram_hash_state(np.asarray(ids), start_pos=start_pos))

        h = self.weights["embed.weight"][ids].astype(jnp.bfloat16)
        h = jnp.repeat(h[:, :, None, :], cfg.hc_mult, axis=2)
        pre_mix = jnp.zeros((bsz, seqlen, cfg.hc_mult), dtype=jnp.float32).at[:, :, 0].set(1.0)

        shared_runtime: dict[str, Any] = {}
        main_hiddens: list[jax.Array] = []
        for layer_id in range(cfg.n_layers):
            if self.engram_layout is not None and layer_id in self.engram_layout.layer_ids:
                eng_idx = self.engram_layout.layer_ids.index(layer_id)
                h = engram_forward(
                    h,
                    engram_hashes[:, :, eng_idx, :],
                    self.weights,
                    f"layers.{layer_id}",
                    cfg,
                    self.engram_layout.num_embeddings[eng_idx],
                )
            if layer_id in cfg.dspark_target_layer_ids:
                main_hiddens.append(jnp.mean(h.astype(jnp.float32), axis=2).astype(jnp.bfloat16))
            h, pre_mix = self._block_step(
                layer_id,
                h,
                start_pos,
                pre_mix,
                state,
                shared_runtime,
                index_k_mode,
                sinkhorn_iters,
            )

        y = hc_pre(h, pre_mix)
        normed = rms_norm(y, self.weights["norm.weight"], cfg.rms_norm_eps)
        if not full_logits:
            normed = normed[:, -1]
        logits = jnp.einsum(
            "...d,vd->...v",
            normed.astype(jnp.float32),
            self.weights["head.weight"].astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        )
        output_ids = jnp.argmax(logits, axis=-1)
        main_hidden = jnp.concatenate(main_hiddens, axis=-1) if main_hiddens else None
        return output_ids, logits, main_hidden

    def forward(
        self,
        input_ids: Any,
        start_pos: int = 0,
        state: DSV41ReferenceState | None = None,
        *,
        full_logits: bool = False,
        index_k_mode: str | None = None,
        sinkhorn_iters: int | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array | None]:
        """Runs prefill (`start_pos == 0`), single-token decode (`seqlen == 1`), or multi-token block verify (`start_pos > 0, seqlen > 1`)."""
        st = state if state is not None else self.state
        ik_mode = index_k_mode if index_k_mode is not None else self.index_k_mode
        s_iters = sinkhorn_iters if sinkhorn_iters is not None else self.sinkhorn_iters

        ids = jnp.asarray(input_ids, dtype=jnp.int32)
        if ids.ndim == 1:
            ids = ids[None, :]
        _, seqlen = ids.shape

        if start_pos == 0 or seqlen == 1:
            return self._forward_single_or_prefill(
                ids,
                start_pos=start_pos,
                state=st,
                full_logits=full_logits,
                index_k_mode=ik_mode,
                sinkhorn_iters=s_iters,
            )

        # Multi-token block verification (`start_pos > 0, seqlen = q > 1`):
        # By `INVENTORY.md` §8.2, a verify block of q tokens at positions `start_pos .. start_pos + q - 1`
        # has the exact causal window and compressed KV/indexer semantics of q sequential single-token steps.
        step_logits: list[jax.Array] = []
        step_hiddens: list[jax.Array] = []
        for i in range(seqlen):
            _, l_i, mh_i = self._forward_single_or_prefill(
                ids[:, i : i + 1],
                start_pos=start_pos + i,
                state=st,
                full_logits=False,
                index_k_mode=ik_mode,
                sinkhorn_iters=s_iters,
            )
            step_logits.append(l_i)
            if mh_i is not None:
                step_hiddens.append(mh_i)
        all_logits = jnp.stack(step_logits, axis=1)  # [B, q, vocab_size]
        all_main_hidden = jnp.concatenate(step_hiddens, axis=1) if step_hiddens else None
        if full_logits:
            return jnp.argmax(all_logits, axis=-1), all_logits, all_main_hidden
        last_logits = all_logits[:, -1, :]
        return jnp.argmax(last_logits, axis=-1), last_logits, all_main_hidden

    def __call__(self, input_ids: Any, start_pos: int = 0, **kwargs: Any):
        return self.forward(input_ids, start_pos=start_pos, **kwargs)

    def _dspark_attention_step(
        self,
        layer_id: int,
        x: jax.Array,
        start_pos: int,
        main_x: jax.Array,
        state: DSV41ReferenceState,
    ) -> jax.Array:
        """Runs `DSparkAttention.forward` (`REF/model.py:1032-1074`)."""
        cfg = self.cfg
        bsz, seqlen, _ = main_x.shape
        win = cfg.sliding_window
        stage_id = layer_id - cfg.n_layers
        l_prefix = f"mtp.{stage_id}"
        cos_all, sin_all = self._get_layer_rope(layer_id)

        cos_m = cos_all[start_pos : start_pos + seqlen]
        sin_m = sin_all[start_pos : start_pos + seqlen]
        main_kv = rms_norm(
            fp8_linear(main_x, self.weights[f"{l_prefix}.attn.wkv.weight"], self.weights[f"{l_prefix}.attn.wkv.scale"]),
            self.weights[f"{l_prefix}.attn.kv_norm.weight"],
            cfg.rms_norm_eps,
        )
        main_kv = apply_rotary_emb_jax(main_kv, (cos_m, sin_m), inverse=False)
        main_kv = act_quant_jax(main_kv, block_size=32, scale_fmt="ue8m0", inplace=True)

        if start_pos == 0:
            if seqlen <= win:
                state.window_kv_cache[layer_id] = state.window_kv_cache[layer_id].at[:bsz, :seqlen].set(main_kv)
            else:
                cutoff = seqlen % win
                tail_win = main_kv[:, -win:]
                state.window_kv_cache[layer_id] = (
                    state.window_kv_cache[layer_id]
                    .at[:bsz, cutoff:win]
                    .set(tail_win[:, : win - cutoff])
                    .at[:bsz, :cutoff]
                    .set(tail_win[:, win - cutoff :])
                )
            return x

        for s_idx in range(seqlen):
            slot = (start_pos + s_idx) % win
            state.window_kv_cache[layer_id] = (
                state.window_kv_cache[layer_id].at[:bsz, slot].set(main_kv[:, s_idx])
            )

        _, block_size, _ = x.shape
        draft_start = start_pos + seqlen
        cos_b = cos_all[draft_start : draft_start + block_size]
        sin_b = sin_all[draft_start : draft_start + block_size]

        qr = rms_norm(
            fp8_linear(x, self.weights[f"{l_prefix}.attn.wq_a.weight"], self.weights[f"{l_prefix}.attn.wq_a.scale"]),
            self.weights[f"{l_prefix}.attn.q_norm.weight"],
            cfg.rms_norm_eps,
        )
        q = fp8_linear(
            qr,
            self.weights[f"{l_prefix}.attn.wq_b.weight"],
            self.weights[f"{l_prefix}.attn.wq_b.scale"],
        ).reshape(bsz, block_size, cfg.n_heads, cfg.head_dim)
        q = apply_rotary_emb_jax(q, (cos_b, sin_b), inverse=False)

        kv = rms_norm(
            fp8_linear(x, self.weights[f"{l_prefix}.attn.wkv.weight"], self.weights[f"{l_prefix}.attn.wkv.scale"]),
            self.weights[f"{l_prefix}.attn.kv_norm.weight"],
            cfg.rms_norm_eps,
        )
        kv = apply_rotary_emb_jax(kv, (cos_b, sin_b), inverse=False)
        kv = act_quant_jax(kv, block_size=32, scale_fmt="ue8m0", inplace=True)

        valid_ring = min(win, start_pos + seqlen)
        idxs_1d = jnp.concatenate(
            [
                jnp.arange(valid_ring, dtype=jnp.int32),
                win + jnp.arange(block_size, dtype=jnp.int32),
            ]
        )
        topk_idxs = jnp.broadcast_to(idxs_1d[None, None, :], (bsz, block_size, idxs_1d.shape[0]))
        kv_all = jnp.concatenate([state.window_kv_cache[layer_id][:bsz], kv], axis=1)

        attn_sink = self.weights[f"{l_prefix}.attn.attn_sink"]
        o = sparse_attn(q, kv_all, attn_sink, topk_idxs, softmax_scale=float(cfg.head_dim**-0.5))
        o = apply_rotary_emb_jax(o, (cos_b, sin_b), inverse=True)

        o_grouped = o.reshape(bsz, block_size, cfg.o_groups, -1)
        wo_a_raw = self.weights[f"{l_prefix}.attn.wo_a.weight"]
        wo_a_s_key = f"{l_prefix}.attn.wo_a.scale"
        if wo_a_s_key in self.weights and wo_a_raw.ndim == 2:
            wo_a_deq = dequant_fp8_block32x32_jax(wo_a_raw, self.weights[wo_a_s_key], out_dtype=jnp.bfloat16)
        else:
            wo_a_deq = wo_a_raw.astype(jnp.bfloat16)
        wo_a = wo_a_deq.reshape(cfg.o_groups, cfg.o_lora_rank, -1)
        o_proj = jnp.einsum(
            "bsgd,grd->bsgr",
            o_grouped.astype(jnp.float32),
            wo_a.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        ).astype(jnp.bfloat16)
        return fp8_linear(
            o_proj.reshape(bsz, block_size, cfg.o_groups * cfg.o_lora_rank),
            self.weights[f"{l_prefix}.attn.wo_b.weight"],
            self.weights[f"{l_prefix}.attn.wo_b.scale"],
        )

    def forward_spec(
        self,
        input_ids: Any,
        main_hidden: jax.Array,
        start_pos: int = 0,
        state: DSV41ReferenceState | None = None,
        *,
        sinkhorn_iters: int | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array] | None:
        """Runs DSpark (`mtp.0..n_mtp_layers-1`) forward and Markov chain drafting (`REF/model.py:1275-1283`)."""
        cfg = self.cfg
        if cfg.dspark_block_size <= 0 or cfg.n_mtp_layers <= 0:
            return None
        st = state if state is not None else self.state
        s_iters = sinkhorn_iters if sinkhorn_iters is not None else self.sinkhorn_iters

        ids = jnp.asarray(input_ids, dtype=jnp.int32)
        if ids.ndim == 2:
            ids_1d = ids[:, -1]
        else:
            ids_1d = ids
        bsz = ids_1d.shape[0]
        block_size = cfg.dspark_block_size

        # forward_embed (`REF/model.py:1128-1135`)
        main_x = rms_norm(
            fp8_linear(
                main_hidden,
                self.weights["mtp.0.main_proj.weight"],
                self.weights["mtp.0.main_proj.scale"],
            ),
            self.weights["mtp.0.main_norm.weight"],
            cfg.rms_norm_eps,
        )
        draft_ids = jnp.full((bsz, block_size), cfg.dspark_noise_token_id, dtype=jnp.int32)
        draft_ids = draft_ids.at[:, 0].set(ids_1d)
        h = self.weights["embed.weight"][draft_ids].astype(jnp.bfloat16)
        h = jnp.repeat(h[:, :, None, :], cfg.hc_mult, axis=2)
        pre_mix = jnp.zeros((bsz, block_size, cfg.hc_mult), dtype=jnp.float32).at[:, :, 0].set(1.0)

        for stage_id in range(cfg.n_mtp_layers):
            layer_id = cfg.n_layers + stage_id
            l_prefix = f"mtp.{stage_id}"
            if start_pos == 0:
                self._dspark_attention_step(layer_id, h[:, :, 0, :], start_pos, main_x, st)
                continue

            residual = h
            attn_pre, attn_post, attn_comb = hc_mixes(
                h,
                self.weights[f"{l_prefix}.hc_attn_fn"],
                self.weights[f"{l_prefix}.hc_attn_scale"],
                self.weights[f"{l_prefix}.hc_attn_base"],
                hc_mult=cfg.hc_mult,
                sinkhorn_iters=s_iters,
                eps=cfg.hc_eps,
                norm_eps=cfg.rms_norm_eps,
            )
            u = rms_norm(
                hc_pre(h, pre_mix),
                self.weights[f"{l_prefix}.attn_norm.weight"],
                cfg.rms_norm_eps,
            )
            attn_out = self._dspark_attention_step(layer_id, u, start_pos, main_x, st)
            h = hc_post(attn_out, residual, attn_post, attn_comb)

            residual = h
            ffn_pre, ffn_post, ffn_comb = hc_mixes(
                h,
                self.weights[f"{l_prefix}.hc_ffn_fn"],
                self.weights[f"{l_prefix}.hc_ffn_scale"],
                self.weights[f"{l_prefix}.hc_ffn_base"],
                hc_mult=cfg.hc_mult,
                sinkhorn_iters=s_iters,
                eps=cfg.hc_eps,
                norm_eps=cfg.rms_norm_eps,
            )
            v = rms_norm(
                hc_pre(h, attn_pre),
                self.weights[f"{l_prefix}.ffn_norm.weight"],
                cfg.rms_norm_eps,
            )
            ffn_out = moe_forward(v, self.weights, l_prefix, cfg, layer_id)
            h = hc_post(ffn_out, residual, ffn_post, ffn_comb)
            pre_mix = ffn_pre

        if start_pos == 0:
            return None

        # forward_head (`REF/model.py:1137-1156`)
        last_prefix = f"mtp.{cfg.n_mtp_layers - 1}"
        x_collapsed = hc_pre(h, pre_mix)
        normed = rms_norm(x_collapsed, self.weights[f"{last_prefix}.norm.weight"], cfg.rms_norm_eps)
        logits = jnp.einsum(
            "bsd,vd->bsv",
            normed.astype(jnp.float32),
            self.weights["head.weight"].astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        )

        markov_emb_w = self.weights[f"{last_prefix}.markov_head.embed.weight"].astype(jnp.bfloat16)
        markov_head_w = self.weights[f"{last_prefix}.markov_head.head.weight"].astype(jnp.float32)
        conf_proj_w = self.weights[f"{last_prefix}.confidence_head.proj.weight"].astype(jnp.float32)

        out_tokens = [ids_1d]
        updated_logits = []
        markov_embeds = []
        for i in range(block_size):
            cur_tok = out_tokens[i]
            m_emb = markov_emb_w[cur_tok]  # [B, markov_rank] bf16
            logits_bias = jnp.einsum(
                "br,vr->bv",
                m_emb.astype(jnp.float32),
                markov_head_w,
                precision=jax.lax.Precision.HIGHEST,
            )
            l_i = logits[:, i, :] + logits_bias
            next_tok = jnp.argmax(l_i, axis=-1).astype(jnp.int32)
            out_tokens.append(next_tok)
            updated_logits.append(l_i)
            markov_embeds.append(m_emb)

        output_ids = jnp.stack(out_tokens, axis=1)  # [B, block_size + 1]
        logits_out = jnp.stack(updated_logits, axis=1)  # [B, block_size, vocab_size]
        markov_embed_all = jnp.stack(markov_embeds, axis=1)  # [B, block_size, markov_rank]
        conf_in = jnp.concatenate([x_collapsed, markov_embed_all], axis=-1).astype(jnp.float32)
        confidence = jnp.einsum(
            "bsd,od->bso",
            conf_in,
            conf_proj_w,
            precision=jax.lax.Precision.HIGHEST,
        ).squeeze(-1)
        return output_ids, logits_out, confidence

    def verify_block(
        self,
        block_ids: Any,
        start_pos: int,
        state: DSV41ReferenceState | None = None,
        *,
        force_num_accepted: int | None = None,
        index_k_mode: str | None = None,
        sinkhorn_iters: int | None = None,
    ) -> dict[str, Any]:
        """Verifies a candidate block `[t_{p}, d_{p+1}, ..., d_{p+q-1}]` (`[1, q]`) at `start_pos = p`.

        Snapshots per-step states across the `q` positions, finds the longest prefix where greedy
        target predictions match the draft tokens (`target_tokens[i] == block_ids[i + 1]`), commits
        the state immediately after the last accepted position (`n_accepted + 1` tokens consumed),
        and returns the accepted + bonus tokens along with the `main_hidden` sequence needed by DSpark.
        """
        st = state if state is not None else self.state
        ik_mode = index_k_mode if index_k_mode is not None else self.index_k_mode
        s_iters = sinkhorn_iters if sinkhorn_iters is not None else self.sinkhorn_iters

        ids = jnp.asarray(block_ids, dtype=jnp.int32)
        if ids.ndim == 1:
            ids = ids[None, :]
        bsz, q = ids.shape
        if bsz != 1:
            raise ValueError(f"verify_block rollback expects batch_size == 1, got {bsz}")

        working = st.clone()
        post_step_states: list[DSV41ReferenceState] = []
        step_logits: list[jax.Array] = []
        step_preds: list[int] = []
        step_hiddens: list[jax.Array] = []

        for i in range(q):
            pred_i, logits_i, mh_i = self._forward_single_or_prefill(
                ids[:, i : i + 1],
                start_pos=start_pos + i,
                state=working,
                full_logits=False,
                index_k_mode=ik_mode,
                sinkhorn_iters=s_iters,
            )
            post_step_states.append(working.clone())
            step_logits.append(logits_i)
            step_preds.append(int(pred_i[0]))
            if mh_i is not None:
                step_hiddens.append(mh_i)

        if force_num_accepted is not None:
            n_accepted = int(force_num_accepted)
        else:
            n_accepted = 0
            for i in range(q - 1):
                if step_preds[i] == int(ids[0, i + 1]):
                    n_accepted += 1
                else:
                    break

        committed = post_step_states[n_accepted]
        st.window_kv_cache = committed.window_kv_cache
        st.compress_kv_cache = committed.compress_kv_cache
        st.index_k_cache = committed.index_k_cache
        st.compressor_kv_state = committed.compressor_kv_state
        st.compressor_score_state = committed.compressor_score_state
        st.engram_hash_state = committed.engram_hash_state
        st.shared_index_k_source = committed.shared_index_k_source

        bonus_token = step_preds[n_accepted]
        emitted_tokens = [int(ids[0, i]) for i in range(1, n_accepted + 1)] + [bonus_token]
        accepted_main_hidden = jnp.concatenate(step_hiddens[: n_accepted + 1], axis=1) if step_hiddens else None
        return {
            "num_draft_tokens": q - 1,
            "num_accepted": n_accepted,
            "bonus_token": bonus_token,
            "emitted_tokens": emitted_tokens,
            "next_start_pos": start_pos + n_accepted + 1,
            "logits": jnp.stack(step_logits, axis=1),
            "accepted_main_hidden": accepted_main_hidden,
        }


DeepSeekV41 = DeepSeekV41Reference
