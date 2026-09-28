"""Same-run XLA-jitted decode & verify control for DeepSeek-V4.1-Flash (`@jax.jit` over `shard_map`).

Implements the exact DeepSeek-V4.1-Flash decode step (`[B]` or `[B, 1]`) and multi-token
speculative verify block (`[B, q]`, `q <= 6` or chunked prefill `q = 8`) with the same sharded
weights and KV cache layout as the Pallas decode megakernel:
- Single-pass 4-stream mHC with 20-iteration Sinkhorn (`eps=1e-6`, `RMSNorm eps=1e-20`)
  and `sinkhorn_iters=0` ablation switch
- All 6 backbone layer kinds (`SWA`, `FULL_R2`, `REUSE_R2`, `FULL_R1_CAND`, `REINDEX_R1`, `REUSE_R1`)
- Both `index_k_mode="reference"` (`REF/model.py:537-554`) and `index_k_mode="intended"`
- Routed MoE (top-6 of 384 with `sqrt(softplus)` gate, clamps `[-10, 10]` / `<= 10`, `route_scale=1.5`,
  weight applied before `w2` activation quant) + shared expert
- Engram (`L1, L14` / `cfg.engram_layer_ids`) gated n-gram injection before the block
"""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from deepseek_v41.checkpoint import generate_synthetic_weights
from deepseek_v41.config import DSV41Config
from deepseek_v41.engram_hash import (
    EngramLayout,
    build_compressed_token_map,
    compute_hash_multipliers,
)
from deepseek_v41.quant import (
    act_quant_jax,
    dequant_fp8_block32x32,
    e8m0_to_f32_jax,
    fp4_act_quant_jax,
    pack_checkpoint_mxfp4_for_v7x_bitcast,
    pack_mxfp4_for_v7x_bitcast,
    unpack_mxfp4_v7x_bitcast,
)
from deepseek_v41.rope import apply_rotary_emb_jax, precompute_layer_rope_tables


# ---------------------------------------------------------------------------
# 1. Exact Quantization & Linear Helpers (REF/kernel.py, REF/convert.py)
# ---------------------------------------------------------------------------

def _cpu_pad_barrier(x: Any) -> Any:
    return jax.lax.optimization_barrier(x) if jax.default_backend() == "cpu" else x


def e8m0_to_f32(scale_u8: jax.Array) -> jax.Array:
    """Convert uint8 E8M0 biased exponent to FP32 power-of-two scale `2^(e - 127)`."""
    return e8m0_to_f32_jax(scale_u8)


def act_quant(
    x: jax.Array,
    block_size: int = 32,
    *,
    inplace: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Block-wise FP8 E4M3 quantization with power-of-two (E8M0) scale (`REF/kernel.py:74-91`)."""
    return act_quant_jax(
        x,
        block_size=block_size,
        scale_fmt="ue8m0",
        inplace=inplace,
        return_u8_scale=(not inplace) and (jax.default_backend() == "cpu"),
    )


def fp4_act_quant(
    x: jax.Array,
    block_size: int = 32,
    *,
    inplace: bool = True,
    scale_fmt: str = "e8m0",
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Block-wise FP4 E2M1 quantization with E8M0 (block=32) or E4M3 (block=16) scale (`REF/kernel.py:160-166`)."""
    return fp4_act_quant_jax(
        x,
        block_size=block_size,
        inplace=inplace,
        scale_dtype=scale_fmt,
    )


def pack_mxfp4_u32(codes_out_in: np.ndarray | jax.Array) -> np.ndarray:
    """Pack 4-bit E2M1 codes `[out_dim, in_dim]` into `[in_dim // 8, out_dim]` uint32 for `pltpu.bitcast`."""
    codes = np.asarray(codes_out_in, dtype=np.uint8) & np.uint8(0x0F)
    return pack_mxfp4_for_v7x_bitcast(codes.T, axis=0)


def unpack_mxfp4_u32(packed_u32: jax.Array, scale_u8: jax.Array) -> jax.Array:
    """Unpack `[..., in_dim // 8, out_dim]` uint32 + `[..., in_dim // 32, out_dim]` E8M0 to `[..., in_dim, out_dim]` bf16."""
    return unpack_mxfp4_v7x_bitcast(packed_u32, scale_u8, axis=-2, out_dtype=jnp.bfloat16)


def pad8_matmul_nk(x: jax.Array, w: jax.Array) -> jax.Array:
    """Compute `einsum('...k,nk->...n', x, w)` padded to a multiple of 8 rows so M=1 and M=6 use identical MXU lowering."""
    orig_shape = x.shape[:-1]
    k = x.shape[-1]
    n = w.shape[0]
    x2d = x.astype(jnp.float32).reshape(-1, k)
    m = x2d.shape[0]
    pad_m = ((max(m, 8) + 7) // 8) * 8
    if pad_m > m:
        x2d = jnp.pad(x2d, ((0, pad_m - m), (0, 0)))
    x2d = _cpu_pad_barrier(x2d)
    out2d = jnp.einsum("mk,nk->mn", x2d, w.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
    out2d = _cpu_pad_barrier(out2d)
    return out2d[:m].reshape(*orig_shape, n)


def pad8_matmul_kn(x: jax.Array, w: jax.Array) -> jax.Array:
    """Compute `einsum('...k,kn->...n', x, w)` padded to a multiple of 8 rows so M=1 and M=6 use identical MXU lowering."""
    orig_shape = x.shape[:-1]
    k = x.shape[-1]
    n = w.shape[1]
    x2d = x.astype(jnp.float32).reshape(-1, k)
    m = x2d.shape[0]
    pad_m = ((max(m, 8) + 7) // 8) * 8
    if pad_m > m:
        x2d = jnp.pad(x2d, ((0, pad_m - m), (0, 0)))
    x2d = _cpu_pad_barrier(x2d)
    out2d = jnp.einsum("mk,kn->mn", x2d, w.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
    out2d = _cpu_pad_barrier(out2d)
    return out2d[:m].reshape(*orig_shape, n)


def fp8_linear(
    x: jax.Array,
    weight: jax.Array,
    scale: jax.Array | None = None,
    *,
    act_block_size: int = 32,
    out_dtype: Any = jnp.bfloat16,
) -> jax.Array:
    """Reference-faithful FP8 linear (`REF/model.py:186-205`): `act_quant(x, 32)` then FP8xFP8 -> `out_dtype`."""
    orig_shape = x.shape[:-1]
    k = x.shape[-1]
    n = weight.shape[0]
    x2d = x.reshape(-1, k)
    m = x2d.shape[0]
    pad_m = ((max(m, 8) + 7) // 8) * 8
    if pad_m > m:
        x2d = jnp.pad(x2d, ((0, pad_m - m), (0, 0)))
    x2d = _cpu_pad_barrier(x2d)
    if scale is None and weight.dtype in (jnp.bfloat16, jnp.float32):
        out2d = _cpu_pad_barrier(pad8_matmul_nk(x2d, weight).astype(out_dtype))
        return out2d[:m].reshape(*orig_shape, n)
    x_q, s_x = act_quant(x2d, act_block_size, inplace=False)
    n_k_blocks = k // act_block_size
    s_x_f32 = e8m0_to_f32_jax(s_x)  # [pad_m, K//32]
    s_w_f32 = e8m0_to_f32_jax(scale)  # [N//block_n, K//block_k]
    n_out_blocks = s_w_f32.shape[0]
    out_blk = n // n_out_blocks
    in_blk = k // s_w_f32.shape[1]
    if in_blk != act_block_size:
        s_w_f32 = jnp.repeat(s_w_f32, in_blk // act_block_size, axis=1)
    x_blk = x_q.astype(jnp.float32).reshape(pad_m, n_k_blocks, act_block_size)
    w_blk = weight.astype(jnp.float32).reshape(n_out_blocks, out_blk, n_k_blocks, act_block_size)
    x_deq = (x_blk * s_x_f32[..., :, None]).reshape(pad_m, k)
    w_deq = (w_blk * s_w_f32[:, None, :, None]).reshape(n, k)
    out2d = _cpu_pad_barrier(pad8_matmul_nk(x_deq, w_deq).astype(out_dtype))
    return out2d[:m].reshape(*orig_shape, n)


def fp4_linear(
    x: jax.Array,
    packed_u32: jax.Array,
    scale_u8: jax.Array | None = None,
    *,
    act_block_size: int = 32,
    out_dtype: Any = jnp.bfloat16,
) -> jax.Array:
    """Reference-faithful FP4 expert linear (`REF/kernel.py:477-591`): `act_quant(x, 32)` x MXFP4 weight."""
    orig_shape = x.shape[:-1]
    k = x.shape[-1]
    x2d = x.reshape(-1, k)
    m = x2d.shape[0]
    pad_m = ((max(m, 8) + 7) // 8) * 8
    if pad_m > m:
        x2d = jnp.pad(x2d, ((0, pad_m - m), (0, 0)))
    x2d = _cpu_pad_barrier(x2d)
    if scale_u8 is None or packed_u32.dtype in (jnp.bfloat16, jnp.float32):
        w = packed_u32.astype(jnp.float32)
        x_deq = act_quant(x2d, act_block_size, inplace=True).astype(jnp.float32)
        if w.shape[-1] == k:
            out2d = _cpu_pad_barrier(pad8_matmul_nk(x_deq, w).astype(out_dtype))
        else:
            out2d = _cpu_pad_barrier(pad8_matmul_kn(x_deq, w).astype(out_dtype))
        return out2d[:m].reshape(*orig_shape, out2d.shape[-1])

    x_q, s_x = act_quant(x2d, act_block_size, inplace=False)
    n_k_blocks = k // 32
    w_fp8 = unpack_mxfp4_v7x_bitcast(packed_u32, None, axis=0, out_dtype=jnp.float8_e4m3fn)
    n = w_fp8.shape[1]
    w_blk = w_fp8.astype(jnp.float32).reshape(n_k_blocks, 32, n)
    x_blk = x_q.astype(jnp.float32).reshape(pad_m, n_k_blocks, 32)
    s_x_f32 = e8m0_to_f32_jax(s_x)
    s_w_f32 = e8m0_to_f32_jax(scale_u8)
    if s_w_f32.shape == (n, n_k_blocks) and n != n_k_blocks:
        s_w_f32 = s_w_f32.T
    x_deq = (x_blk * s_x_f32[..., :, None]).reshape(pad_m, k)
    w_deq = (w_blk * s_w_f32[:, None, :]).reshape(k, n)
    out2d = _cpu_pad_barrier(pad8_matmul_kn(x_deq, w_deq).astype(out_dtype))
    return out2d[:m].reshape(*orig_shape, n)


# ---------------------------------------------------------------------------
# 2. RoPE & Single-Pass 4-Stream mHC (REF/model.py, REF/kernel.py)
# ---------------------------------------------------------------------------

def apply_rotary_emb(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    rope_dim: int = 64,
    *,
    inverse: bool = False,
) -> jax.Array:
    """Apply interleaved complex RoPE to the last `rope_dim` elements of `x` (`REF/model.py:392-406`)."""
    del rope_dim
    return apply_rotary_emb_jax(x, cos, sin, inverse=inverse)


def rms_norm(x: jax.Array, weight: jax.Array, eps: float = 1e-20) -> jax.Array:
    """RMSNorm in FP32 with `eps=1e-20`, returning `x.dtype` (`REF/model.py:281-293`)."""
    orig_shape = x.shape
    d = orig_shape[-1]
    x2d = x.astype(jnp.float32).reshape(-1, d)
    m = x2d.shape[0]
    pad_m = ((max(m, 8) + 7) // 8) * 8
    if pad_m > m:
        x2d = jnp.pad(x2d, ((0, pad_m - m), (0, 0)))
    x2d = _cpu_pad_barrier(x2d)
    rstd = jax.lax.rsqrt(jnp.mean(x2d * x2d, axis=-1, keepdims=True) + jnp.float32(eps))
    out2d = _cpu_pad_barrier((x2d * rstd * weight.astype(jnp.float32)).astype(x.dtype))
    return out2d[:m].reshape(orig_shape)


def hc_split_sinkhorn(
    mixes: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Split `[..., 24]` mixes into `(pre, post, comb)` and run 20-iter Sinkhorn (`REF/kernel.py:406-474`)."""
    m = mixes.astype(jnp.float32)
    s = hc_scale.astype(jnp.float32)
    b = hc_base.astype(jnp.float32)
    hc = hc_mult
    eps32 = jnp.float32(eps)

    pre = jax.nn.sigmoid(m[..., :hc] * s[0] + b[:hc]) + eps32
    post = jnp.float32(2.0) * jax.nn.sigmoid(m[..., hc : 2 * hc] * s[1] + b[hc : 2 * hc])
    comb_raw = (m[..., 2 * hc :] * s[2] + b[2 * hc :]).reshape(*m.shape[:-1], hc, hc)

    if sinkhorn_iters <= 0:
        # Skip Sinkhorn doubly-stochastic normalization when sinkhorn_iters <= 0
        iota = jnp.arange(hc, dtype=jnp.int32)
        eye_rev = (iota[:, None] + iota[None, :] == (hc - 1)).astype(jnp.float32)
        comb = jnp.tanh(comb_raw) - jnp.float32(0.5) * eye_rev
        return pre, post, comb

    row_max = jnp.max(comb_raw, axis=-1, keepdims=True)
    comb = jnp.exp(comb_raw - row_max)
    comb = comb / jnp.sum(comb, axis=-1, keepdims=True) + eps32
    comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + eps32)

    for _ in range(sinkhorn_iters - 1):
        comb = comb / (jnp.sum(comb, axis=-1, keepdims=True) + eps32)
        comb = comb / (jnp.sum(comb, axis=-2, keepdims=True) + eps32)

    return pre, post, comb


def hc_mixes(
    x: jax.Array,
    hc_fn: jax.Array,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    *,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-20,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute `(pre, post, comb)` from 4-stream residual `x` `[B, S, 4, D]` (`REF/model.py:948-955`)."""
    orig_prefix = x.shape[:-2]
    hc, d = x.shape[-2], x.shape[-1]
    flat = x.reshape(-1, hc * d).astype(jnp.float32)
    m = flat.shape[0]
    pad_m = ((max(m, 8) + 7) // 8) * 8
    if pad_m > m:
        flat = jnp.pad(flat, ((0, pad_m - m), (0, 0)))
    flat = _cpu_pad_barrier(flat)
    rstd = jax.lax.rsqrt(jnp.mean(flat * flat, axis=-1, keepdims=True) + jnp.float32(norm_eps))
    mixes = pad8_matmul_nk(flat, hc_fn) * rstd
    pre, post, comb = _cpu_pad_barrier(
        hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=hc_mult, sinkhorn_iters=sinkhorn_iters, eps=hc_eps)
    )
    return (
        pre[:m].reshape(*orig_prefix, hc_mult),
        post[:m].reshape(*orig_prefix, hc_mult),
        comb[:m].reshape(*orig_prefix, hc_mult, hc_mult),
    )


def hc_pre(x: jax.Array, pre_mix: jax.Array) -> jax.Array:
    """Collapse 4-stream residual `x` `[B, S, 4, D]` with `pre_mix` `[B, S, 4]` -> `[B, S, D]` BF16."""
    orig_prefix = x.shape[:-2]
    hc, d = x.shape[-2], x.shape[-1]
    x3d = x.reshape(-1, hc, d)
    pm2d = pre_mix.reshape(-1, hc)
    m = x3d.shape[0]
    pad_m = ((max(m, 8) + 7) // 8) * 8
    if pad_m > m:
        x3d = jnp.pad(x3d, ((0, pad_m - m), (0, 0), (0, 0)))
        pm2d = jnp.pad(pm2d, ((0, pad_m - m), (0, 0)))
    x3d, pm2d = _cpu_pad_barrier((x3d, pm2d))
    y = jnp.sum(pm2d.astype(jnp.float32)[..., None] * x3d.astype(jnp.float32), axis=-2)
    y = _cpu_pad_barrier(y.astype(x.dtype))
    return y[:m].reshape(*orig_prefix, d)


def hc_post(
    y: jax.Array,
    residual: jax.Array,
    post: jax.Array,
    comb: jax.Array,
) -> jax.Array:
    """Expand sublayer output `y` `[B, S, D]` back into 4 streams with `comb` `[B, S, 4, 4]` -> BF16."""
    orig_prefix = y.shape[:-1]
    hc, d = residual.shape[-2], y.shape[-1]
    y2d = y.reshape(-1, d)
    r3d = residual.reshape(-1, hc, d)
    p2d = post.reshape(-1, hc)
    c3d = comb.reshape(-1, hc, hc)
    m = y2d.shape[0]
    pad_m = ((max(m, 8) + 7) // 8) * 8
    if pad_m > m:
        y2d = jnp.pad(y2d, ((0, pad_m - m), (0, 0)))
        r3d = jnp.pad(r3d, ((0, pad_m - m), (0, 0), (0, 0)))
        p2d = jnp.pad(p2d, ((0, pad_m - m), (0, 0)))
        c3d = jnp.pad(c3d, ((0, pad_m - m), (0, 0), (0, 0)))
    y2d, r3d, p2d, c3d = _cpu_pad_barrier((y2d, r3d, p2d, c3d))
    y32 = y2d.astype(jnp.float32)
    r32 = r3d.astype(jnp.float32)
    # REF/model.py:965: torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    mixed_res = jnp.sum(c3d.astype(jnp.float32)[..., None] * r32[..., :, None, :], axis=-3)
    out = p2d.astype(jnp.float32)[..., :, None] * y32[..., None, :] + mixed_res
    out = _cpu_pad_barrier(out.astype(y.dtype))
    return out[:m].reshape(*orig_prefix, hc, d)


# ---------------------------------------------------------------------------
# 3. Engram Hashing & Gated Injection (REF/engram.py, REF/model.py:328-365)
# ---------------------------------------------------------------------------

def build_engram_tables(cfg: DSV41Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute `(primes, offsets, multipliers)` for `cfg.engram_layer_ids` via `engram_hash.py`."""
    layout = EngramLayout.from_config(cfg)
    if layout is None:
        return (
            np.zeros((0, cfg.engram_max_ngram_size - 1, cfg.engram_n_heads), dtype=np.int64),
            np.zeros((0, (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads), dtype=np.int64),
            np.zeros((0, cfg.engram_max_ngram_size), dtype=np.int64),
        )
    default_c = getattr(cfg, "engram_compressed_vocab_size", None)
    if default_c is None:
        default_c = 99092 if cfg.vocab_size == 129280 else cfg.vocab_size
    _, comp_vocab = build_compressed_token_map(None, vocab_size=cfg.vocab_size, compressed_vocab_size=default_c)
    multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, comp_vocab)
    return layout.primes_np, layout.offsets_np, multipliers


def engram_forward(
    x: jax.Array,
    embed_flat: jax.Array,
    wkv_weight: jax.Array,
    wkv_scale: jax.Array | None,
    q_weight: jax.Array,
    k_weight: jax.Array,
    cfg: DSV41Config,
    *,
    sharded_tp: bool = False,
    axis_name: str = "tp",
) -> jax.Array:
    """Apply Engram gated injection to 4-stream `x` `[B, S, 4, D]` before the block (`REF/model.py:350-365`)."""
    orig_prefix = x.shape[:-2]
    m = int(np.prod(orig_prefix))
    pad_m = ((max(m, 8) + 7) // 8) * 8
    x_pad = x.reshape(m, cfg.hc_mult, cfg.dim)
    e_pad = embed_flat.reshape(m, embed_flat.shape[-1])
    if pad_m > m:
        x_pad = jnp.pad(x_pad, ((0, pad_m - m), (0, 0), (0, 0)))
        e_pad = jnp.pad(e_pad, ((0, pad_m - m), (0, 0)))
    x_pad, e_pad = _cpu_pad_barrier((x_pad, e_pad))
    kv = fp8_linear(e_pad, wkv_weight, wkv_scale, act_block_size=32)
    if sharded_tp and kv.shape[-1] < (cfg.hc_mult + 1) * cfg.dim:
        kv = jax.lax.all_gather(kv, axis_name, axis=-1, tiled=True)
    kv = _cpu_pad_barrier(kv)
    key = kv[:, : cfg.hc_mult * cfg.dim].astype(jnp.float32).reshape(pad_m, cfg.hc_mult, cfg.dim)
    value = kv[:, cfg.hc_mult * cfg.dim : (cfg.hc_mult + 1) * cfg.dim].astype(jnp.bfloat16)

    wgt = q_weight.astype(jnp.float32) * k_weight.astype(jnp.float32)
    h = x_pad.astype(jnp.float32)
    eps = jnp.float32(cfg.rms_norm_eps)
    rstd = jax.lax.rsqrt(jnp.mean(h * h, axis=-1) + eps) * jax.lax.rsqrt(jnp.mean(key * key, axis=-1) + eps)
    dot = jnp.sum(h * wgt * key, axis=-1) * rstd * jnp.float32(cfg.dim**-0.5)
    signed_sqrt = jnp.where(dot >= 0.0, 1.0, -1.0) * jnp.sqrt(jnp.maximum(jnp.abs(dot), jnp.float32(1e-6)))
    gate = jax.nn.sigmoid(signed_sqrt)
    out = h + gate[..., None] * value.astype(jnp.float32)[..., None, :]
    out_pad = _cpu_pad_barrier(out.astype(x.dtype))
    return out_pad[:m].reshape(*orig_prefix, cfg.hc_mult, cfg.dim)


# ---------------------------------------------------------------------------
# 4. Sparse Attention, Candidate Blocks, and MoE (INVENTORY.md §2–§3)
# ---------------------------------------------------------------------------

def sparse_attn(
    q: jax.Array,
    kv: jax.Array,
    attn_sink: jax.Array,
    topk_idxs: jax.Array,
    softmax_scale: float,
) -> jax.Array:
    """Reference-faithful MQA latent sparse attention (`REF/kernel.py:310-403`)."""
    bsz, _seqlen, _h_local, _d = q.shape
    valid = topk_idxs >= 0
    safe_idxs = jnp.clip(topk_idxs, 0, kv.shape[1] - 1)
    batch_iota = jnp.arange(bsz)[:, None, None]
    kv_gathered = jnp.where(valid[..., None], kv[batch_iota, safe_idxs], jnp.bfloat16(0.0))

    scores = (
        jnp.einsum(
            "bshd,bskd->bhsk",
            q.astype(jnp.float32),
            kv_gathered.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        )
        * jnp.float32(softmax_scale)
    )
    scores = jnp.where(valid[:, None, :, :], scores, jnp.float32(-jnp.inf))

    m = jnp.maximum(jnp.max(scores, axis=-1, keepdims=True), jnp.float32(-1e30))
    p_f32 = jnp.where(valid[:, None, :, :], jnp.exp(scores - m), jnp.float32(0.0))
    sum_exp = jnp.sum(p_f32, axis=-1, keepdims=True) + jnp.exp(attn_sink.astype(jnp.float32)[None, :, None, None] - m)

    p_bf16 = p_f32.astype(jnp.bfloat16)
    acc_o = jnp.einsum(
        "bhsk,bskd->bshd",
        p_bf16.astype(jnp.float32),
        kv_gathered.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )
    out = acc_o / sum_exp.transpose(0, 2, 1, 3)
    return out.astype(jnp.bfloat16)


def select_candidate_blocks(
    scores: jax.Array,
    compress_lens: jax.Array,
    topk_blocks: int,
    block_size: int,
) -> jax.Array:
    """Two-level candidate block selection at Layer 20 (`REF/model.py:583-610`)."""
    width = scores.shape[-1]
    pad_len = (-width) % block_size
    padded = jnp.pad(scores, ((0, 0), (0, pad_len)), constant_values=-jnp.inf) if pad_len > 0 else scores
    num_blocks = padded.shape[-1] // block_size
    block_scores = jnp.max(padded.reshape(scores.shape[0], num_blocks, block_size), axis=-1)

    last_blk = (compress_lens - 1) // block_size
    blk_iota = jnp.arange(num_blocks, dtype=jnp.int32)[None, :]
    block_scores = jnp.where((compress_lens[:, None] > 0) & (blk_iota == last_blk[:, None]), jnp.inf, block_scores)

    k_blk = min(topk_blocks, num_blocks)
    top_vals, top_idxs = jax.lax.top_k(block_scores, k_blk)
    valid_top = top_vals > -jnp.inf
    keep_blk = jnp.any(
        (blk_iota[:, :, None] == top_idxs[:, None, :]) & valid_top[:, None, :],
        axis=-1,
    )
    return jnp.repeat(keep_blk, block_size, axis=-1)[:, :width]


def moe_gate(
    v: jax.Array,
    gate_weight: jax.Array,
    gate_bias: jax.Array,
    topk: int,
    route_scale: float = 1.5,
) -> tuple[jax.Array, jax.Array]:
    """MoE router (`REF/model.py:809-827`): `sqrt(softplus(s))`, top-k on `scores + bias`, normalized."""
    orig_prefix = v.shape[:-1]
    d = v.shape[-1]
    v2d = v.reshape(-1, d)
    m = v2d.shape[0]
    pad_m = ((max(m, 8) + 7) // 8) * 8
    if pad_m > m:
        v2d = jnp.pad(v2d, ((0, pad_m - m), (0, 0)))
    v2d = _cpu_pad_barrier(v2d)
    s = pad8_matmul_nk(v2d, gate_weight)
    softplus_s = jnp.where(s > 20.0, s, jnp.log1p(jnp.exp(jnp.minimum(s, 20.0))))
    scores = jnp.sqrt(jnp.maximum(softplus_s, 0.0))
    _, top_indices = jax.lax.top_k(scores + gate_bias.astype(jnp.float32), topk)
    top_weights = jnp.take_along_axis(scores, top_indices, axis=-1)
    if topk > 1:
        top_weights = top_weights / (jnp.sum(top_weights, axis=-1, keepdims=True) + jnp.float32(1e-20))
    top_weights = top_weights * jnp.float32(route_scale)
    top_weights, top_indices = _cpu_pad_barrier((top_weights, top_indices))
    return top_weights[:m].reshape(*orig_prefix, topk), top_indices[:m].reshape(*orig_prefix, topk)


def run_single_expert_fp4(
    x: jax.Array,
    w_e: jax.Array,
    w1_u32: jax.Array,
    w1_s: jax.Array,
    w3_u32: jax.Array,
    w3_s: jax.Array,
    w2_u32: jax.Array,
    w2_s: jax.Array,
    swiglu_limit: float = 10.0,
    *,
    out_dtype: Any = jnp.bfloat16,
) -> jax.Array:
    """One SwiGLU routed expert with clamps and pre-w2 weight scaling (`REF/model.py:841-851`)."""
    orig_shape = x.shape[:-1]
    k = x.shape[-1]
    x2d = x.reshape(-1, k)
    we1d = w_e.astype(jnp.float32).reshape(-1)
    m = x2d.shape[0]
    pad_m = ((max(m, 8) + 7) // 8) * 8
    if pad_m > m:
        x2d = jnp.pad(x2d, ((0, pad_m - m), (0, 0)))
        we1d = jnp.pad(we1d, (0, pad_m - m))
    x2d, we1d = _cpu_pad_barrier((x2d, we1d))
    gate = fp4_linear(x2d, w1_u32, w1_s, act_block_size=32).astype(jnp.float32)
    up = fp4_linear(x2d, w3_u32, w3_s, act_block_size=32).astype(jnp.float32)
    if swiglu_limit > 0.0:
        lim = jnp.float32(swiglu_limit)
        up = jnp.clip(up, -lim, lim)
        gate = jnp.minimum(gate, lim)
    h = jax.nn.silu(gate) * up * we1d[:, None]
    out2d = fp4_linear(h.astype(jnp.bfloat16), w2_u32, w2_s, act_block_size=32, out_dtype=out_dtype)
    out2d = _cpu_pad_barrier(out2d)
    return out2d[:m].reshape(*orig_shape, out2d.shape[-1])


def run_shared_expert_fp8(
    x: jax.Array,
    w1: jax.Array,
    w1_s: jax.Array | None,
    w3: jax.Array,
    w3_s: jax.Array | None,
    w2: jax.Array,
    w2_s: jax.Array | None,
    swiglu_limit: float = 10.0,
    *,
    out_dtype: Any = jnp.bfloat16,
) -> jax.Array:
    """Shared SwiGLU expert with FP8 weights (`REF/model.py:841-851, 903`)."""
    orig_shape = x.shape[:-1]
    k = x.shape[-1]
    x2d = x.reshape(-1, k)
    m = x2d.shape[0]
    pad_m = ((max(m, 8) + 7) // 8) * 8
    if pad_m > m:
        x2d = jnp.pad(x2d, ((0, pad_m - m), (0, 0)))
    x2d = _cpu_pad_barrier(x2d)
    gate = fp8_linear(x2d, w1, w1_s, act_block_size=32).astype(jnp.float32)
    up = fp8_linear(x2d, w3, w3_s, act_block_size=32).astype(jnp.float32)
    if swiglu_limit > 0.0:
        lim = jnp.float32(swiglu_limit)
        up = jnp.clip(up, -lim, lim)
        gate = jnp.minimum(gate, lim)
    h = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
    out2d = fp8_linear(h, w2, w2_s, act_block_size=32, out_dtype=out_dtype)
    out2d = _cpu_pad_barrier(out2d)
    return out2d[:m].reshape(*orig_shape, out2d.shape[-1])


# ---------------------------------------------------------------------------
# 5. Weight Normalization, Generation, Cache Layout, and TP/EP Sharding
# ---------------------------------------------------------------------------

def normalize_weights(
    cfg: DSV41Config,
    raw_weights: Mapping[str, Any],
    *,
    max_seq_len: int = 256,
) -> dict[str, jax.Array]:
    """Normalize a raw checkpoint dict (e.g. from `generate_synthetic_weights`) into stacked JAX arrays.

    - Pads `embed.weight` and `head.weight` to `cfg.padded_vocab_size` if needed.
    - Dequantizes `wo_a.weight` + `wo_a.scale` to `[o_groups, o_lora_rank, group_head_dim]` `bfloat16`
      matching `REF/convert.py:157-173`.
    - Packs per-expert `ffn.experts.{e}.{w1,w2,w3}.weight` (`int8` `[out, in // 2]`) into stacked
      `ffn.experts.{w1,w2,w3}.u32` (`[E, in // 8, out]` `uint32`) and `.scale` (`[E, in // 32, out]` `uint8`).
    - Populates `rope.{swa,yarn}.{cos,sin}` tables if not already present.
    """
    w: dict[str, Any] = dict(raw_weights)

    # Pad embed and head to padded_vocab_size
    emb = np.asarray(w["embed.weight"])
    if emb.shape[0] < cfg.padded_vocab_size:
        pad_emb = np.zeros((cfg.padded_vocab_size - emb.shape[0], emb.shape[1]), dtype=emb.dtype)
        emb = np.concatenate([emb, pad_emb], axis=0)
    w["embed.weight"] = jnp.asarray(emb, dtype=jnp.bfloat16)

    head = np.asarray(w["head.weight"])
    if head.shape[0] < cfg.padded_vocab_size:
        pad_head = np.full((cfg.padded_vocab_size - head.shape[0], head.shape[1]), -1e4, dtype=np.float32).astype(
            head.dtype
        )
        head = np.concatenate([head, pad_head], axis=0)
    w["head.weight"] = jnp.asarray(head, dtype=jnp.bfloat16)
    w["norm.weight"] = jnp.asarray(w["norm.weight"], dtype=jnp.bfloat16)

    if "rope.swa.cos" not in w:
        tables = precompute_layer_rope_tables(cfg, max_seq_len=max_seq_len)
        w["rope.swa.cos"], w["rope.swa.sin"] = tables["swa"]
        w["rope.yarn.cos"], w["rope.yarn.sin"] = tables["yarn"]

    total_layers = cfg.n_layers + cfg.n_mtp_layers
    for l_id in range(total_layers):
        is_mtp = l_id >= cfg.n_layers
        prefix = f"mtp.{l_id - cfg.n_layers}." if is_mtp else f"layers.{l_id}."

        # Convert wo_a from FP8+scale to [o_groups, o_lora_rank, group_head_dim] bf16 if needed
        wo_a_w = w[prefix + "attn.wo_a.weight"]
        wo_a_s_key = prefix + "attn.attn.wo_a.scale" if (prefix + "attn.attn.wo_a.scale") in w else (prefix + "attn.wo_a.scale")
        if wo_a_s_key in w and wo_a_w.ndim == 2:
            deq = dequant_fp8_block32x32(np.asarray(wo_a_w), np.asarray(w.pop(wo_a_s_key)), out_dtype=np.float32)
            w[prefix + "attn.wo_a.weight"] = jnp.asarray(
                deq.reshape(cfg.o_groups, cfg.o_lora_rank, -1), dtype=jnp.bfloat16
            )
        else:
            w[prefix + "attn.wo_a.weight"] = jnp.asarray(wo_a_w, dtype=jnp.bfloat16).reshape(
                cfg.o_groups, cfg.o_lora_rank, -1
            )

        # Stack per-expert I8 weights into [E, K // 8, N] uint32 + [E, K // 32, N] uint8
        fp = prefix + "ffn."
        n_exp = cfg.dspark_n_routed_experts if is_mtp else cfg.n_routed_experts
        if (fp + "experts.w1.u32") not in w and (fp + "experts.0.w1.weight") in w:
            for proj in ("w1", "w2", "w3"):
                u32_list = []
                scale_list = []
                for e in range(n_exp):
                    wi8 = np.asarray(w.pop(f"{fp}experts.{e}.{proj}.weight"))
                    ws = np.asarray(w.pop(f"{fp}experts.{e}.{proj}.scale"), dtype=np.uint8)
                    u32_list.append(pack_checkpoint_mxfp4_for_v7x_bitcast(wi8, transpose=True))
                    scale_list.append(ws.T)
                w[f"{fp}experts.{proj}.u32"] = jnp.asarray(np.stack(u32_list, axis=0), dtype=jnp.uint32)
                w[f"{fp}experts.{proj}.scale"] = jnp.asarray(np.stack(scale_list, axis=0), dtype=jnp.uint8)

        if is_mtp and (l_id - cfg.n_layers) == cfg.n_mtp_layers - 1:
            for mkey in ("markov_head.embed.weight", "markov_head.head.weight"):
                arr = np.asarray(w[prefix + mkey])
                if arr.shape[0] < cfg.padded_vocab_size:
                    pad_m = np.zeros((cfg.padded_vocab_size - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
                    arr = np.concatenate([arr, pad_m], axis=0)
                w[prefix + mkey] = jnp.asarray(arr, dtype=jnp.bfloat16)

    # Convert any remaining numpy arrays to JAX arrays
    out: dict[str, jax.Array] = {}
    for k, v in w.items():
        if isinstance(v, jax.Array):
            out[k] = v
        else:
            arr = np.asarray(v)
            if str(arr.dtype) == "bfloat16":
                out[k] = jnp.asarray(arr, dtype=jnp.bfloat16)
            elif str(arr.dtype) == "float8_e4m3fn":
                out[k] = jnp.asarray(arr, dtype=jnp.float8_e4m3fn)
            else:
                out[k] = jnp.asarray(arr)
    return out


def make_random_weights(cfg: DSV41Config, seed: int = 0, max_seq_len: int = 256) -> dict[str, jax.Array]:
    """Create a complete deterministic quantized weight dictionary for `cfg` via `checkpoint.py`."""
    raw = generate_synthetic_weights(cfg, seed=seed)
    for k, v in list(raw.items()):
        if k.endswith(".scale") and v.dtype == np.uint8:
            raw[k] = np.clip(v.astype(np.int32) - 5, 1, 254).astype(np.uint8)
    return normalize_weights(cfg, raw, max_seq_len=max_seq_len)


def replicate_to_mesh(x: Any, mesh: Mesh) -> jax.Array:
    """Place `x` on `NamedSharding(mesh, P())` across single-host or multi-host meshes."""
    rep_sharding = NamedSharding(mesh, P())
    if isinstance(x, jax.Array) and x.sharding == rep_sharding:
        return x
    if len(mesh.local_devices) == mesh.size:
        return jax.device_put(jnp.asarray(x), rep_sharding)
    arr_np = np.asarray(x)
    j_local = jnp.asarray(arr_np)
    shards = [jax.device_put(j_local, d) for d in mesh.local_devices]
    return jax.make_array_from_single_device_arrays(arr_np.shape, rep_sharding, shards)


def init_cache(
    cfg: DSV41Config,
    batch_size: int = 1,
    max_seq_len: int = 256,
    mesh: Mesh | None = None,
) -> dict[str, jax.Array]:
    """Initialize KV, compressor, indexer, Engram, and DSpark state buffers."""
    n_kv_sources = len(cfg.kv_source_layer_ids)
    cache: dict[str, jax.Array] = {
        "window_kv": jnp.zeros((cfg.n_layers, batch_size, cfg.sliding_window, cfg.head_dim), dtype=jnp.bfloat16),
        "compress_kv": jnp.zeros((n_kv_sources, batch_size, max_seq_len, cfg.head_dim), dtype=jnp.bfloat16),
        "index_k": jnp.zeros((n_kv_sources, batch_size, max_seq_len, cfg.index_head_dim), dtype=jnp.bfloat16),
        "comp_kv_state": jnp.zeros((n_kv_sources, batch_size, 2, cfg.head_dim), dtype=jnp.float32),
        "comp_score_state": jnp.full((n_kv_sources, batch_size, 2, cfg.head_dim), -jnp.inf, dtype=jnp.float32),
        "comp_kv_ring": jnp.zeros((n_kv_sources, batch_size, 8, cfg.head_dim), dtype=jnp.float32),
        "comp_score_ring": jnp.full((n_kv_sources, batch_size, 8, cfg.head_dim), -jnp.inf, dtype=jnp.float32),
        "last_index_k_owner": jnp.int32(n_kv_sources - 1),
        "token_history": jnp.full((batch_size, max_seq_len), cfg.engram_pad_id, dtype=jnp.int32),
        "dspark_window_kv": jnp.zeros(
            (cfg.n_mtp_layers, batch_size, cfg.sliding_window, cfg.head_dim), dtype=jnp.bfloat16
        ),
    }
    if mesh is not None:
        cache = {k: replicate_to_mesh(v, mesh) for k, v in cache.items()}
    return cache


def shard_weights(
    cfg_or_weights: DSV41Config | Mapping[str, Any],
    weights_or_cfg: Mapping[str, Any] | DSV41Config,
    mesh: Mesh,
) -> dict[str, jax.Array]:
    """Shard an unsharded weight dict across `mesh` (`N = mesh.size`) following `spec.md` §1."""
    if isinstance(cfg_or_weights, DSV41Config):
        cfg = cfg_or_weights
        weights = weights_or_cfg  # type: ignore[assignment]
    else:
        weights = cfg_or_weights
        cfg = weights_or_cfg  # type: ignore[assignment]
    norm_w = normalize_weights(cfg, weights) if "rope.swa.cos" not in weights or "layers.0.ffn.experts.w1.u32" not in weights else dict(weights)
    num_devices = mesh.size
    tp_sharding = NamedSharding(mesh, P("tp"))
    sharded: dict[str, jax.Array] = {}

    def _put_sharded(arr: np.ndarray | jax.Array) -> jax.Array:
        return jax.device_put(jnp.asarray(arr), tp_sharding)

    def _replicate(arr: np.ndarray | jax.Array) -> jax.Array:
        a = jnp.asarray(arr)
        return _put_sharded(jnp.broadcast_to(a[None], (num_devices, *a.shape)))

    for key in ("embed.weight", "head.weight"):
        arr = jnp.asarray(norm_w[key])
        if arr.shape[0] % num_devices == 0:
            sharded[key] = _put_sharded(arr.reshape(num_devices, arr.shape[0] // num_devices, arr.shape[1]))
        else:
            sharded[key] = _replicate(arr)

    for key in ("norm.weight", "rope.swa.cos", "rope.swa.sin", "rope.yarn.cos", "rope.yarn.sin"):
        sharded[key] = _replicate(norm_w[key])

    total_layers = cfg.n_layers + cfg.n_mtp_layers
    ep_lanes = min(num_devices, 8)
    tp_hosts = max(1, num_devices // 8)

    for l_id in range(total_layers):
        is_mtp = l_id >= cfg.n_layers
        prefix = f"mtp.{l_id - cfg.n_layers}." if is_mtp else f"layers.{l_id}."
        r = cfg.compress_ratios[l_id]

        for rk in (
            "hc_attn_fn",
            "hc_ffn_fn",
            "hc_attn_base",
            "hc_ffn_base",
            "hc_attn_scale",
            "hc_ffn_scale",
            "attn_norm.weight",
            "ffn_norm.weight",
            "attn.wq_a.weight",
            "attn.wq_a.scale",
            "attn.q_norm.weight",
            "attn.wkv.weight",
            "attn.wkv.scale",
            "attn.kv_norm.weight",
        ):
            sharded[prefix + rk] = _replicate(norm_w[prefix + rk])

        wq_b_w = jnp.asarray(norm_w[prefix + "attn.wq_b.weight"])
        wq_b_s = jnp.asarray(norm_w[prefix + "attn.wq_b.scale"])
        sharded[prefix + "attn.wq_b.weight"] = _put_sharded(
            wq_b_w.reshape(num_devices, wq_b_w.shape[0] // num_devices, wq_b_w.shape[1])
        )
        sharded[prefix + "attn.wq_b.scale"] = _put_sharded(
            wq_b_s.reshape(num_devices, wq_b_s.shape[0] // num_devices, wq_b_s.shape[1])
        )
        sink = jnp.asarray(norm_w[prefix + "attn.attn_sink"])
        sharded[prefix + "attn.attn_sink"] = _put_sharded(sink.reshape(num_devices, sink.shape[0] // num_devices))

        wo_a = jnp.asarray(norm_w[prefix + "attn.wo_a.weight"])
        if num_devices >= cfg.o_groups:
            ranks_per_group = num_devices // cfg.o_groups
            cols_per_rank = wo_a.shape[-1] // ranks_per_group
            wo_a_sharded = (
                wo_a.reshape(cfg.o_groups, cfg.o_lora_rank, ranks_per_group, cols_per_rank)
                .transpose(0, 2, 1, 3)
                .reshape(num_devices, cfg.o_lora_rank, cols_per_rank)
            )
            sharded[prefix + "attn.wo_a.weight"] = _put_sharded(wo_a_sharded)
        else:
            groups_per_rank = cfg.o_groups // num_devices
            sharded[prefix + "attn.wo_a.weight"] = _put_sharded(
                wo_a.reshape(num_devices, groups_per_rank, cfg.o_lora_rank, wo_a.shape[-1])
            )

        wo_b_w = jnp.asarray(norm_w[prefix + "attn.wo_b.weight"])
        wo_b_s = jnp.asarray(norm_w[prefix + "attn.wo_b.scale"])
        sharded[prefix + "attn.wo_b.weight"] = _put_sharded(
            wo_b_w.reshape(wo_b_w.shape[0], num_devices, wo_b_w.shape[1] // num_devices).transpose(1, 0, 2)
        )
        sharded[prefix + "attn.wo_b.scale"] = _put_sharded(
            wo_b_s.reshape(wo_b_s.shape[0], num_devices, wo_b_s.shape[1] // num_devices).transpose(1, 0, 2)
        )

        if not is_mtp and l_id in cfg.kv_source_layer_ids:
            cp = prefix + "attn.compressor."
            sharded[cp + "wkv.weight"] = _replicate(norm_w[cp + "wkv.weight"])
            if r > 1:
                sharded[cp + "wgate.weight"] = _replicate(norm_w[cp + "wgate.weight"])
            sharded[cp + "norm.weight"] = _replicate(norm_w[cp + "norm.weight"])

        if not is_mtp and l_id in cfg.index_source_layer_ids:
            ip = prefix + "attn.indexer."
            iwq_w = jnp.asarray(norm_w[ip + "wq_b.weight"])
            iwq_s = jnp.asarray(norm_w[ip + "wq_b.scale"])
            iw_proj = jnp.asarray(norm_w[ip + "weights_proj.weight"])
            sharded[ip + "wq_b.weight"] = _put_sharded(
                iwq_w.reshape(num_devices, iwq_w.shape[0] // num_devices, iwq_w.shape[1])
            )
            sharded[ip + "wq_b.scale"] = _put_sharded(
                iwq_s.reshape(num_devices, iwq_s.shape[0] // num_devices, iwq_s.shape[1])
            )
            sharded[ip + "weights_proj.weight"] = _put_sharded(
                iw_proj.reshape(num_devices, iw_proj.shape[0] // num_devices, iw_proj.shape[1])
            )
            if l_id in cfg.kv_source_layer_ids:
                sharded[ip + "wk.weight"] = _replicate(norm_w[ip + "wk.weight"])
                sharded[ip + "k_norm.weight"] = _replicate(norm_w[ip + "k_norm.weight"])

        fp = prefix + "ffn."
        sharded[fp + "gate.weight"] = _replicate(norm_w[fp + "gate.weight"])
        sharded[fp + "gate.bias"] = _replicate(norm_w[fp + "gate.bias"])

        n_exp = cfg.dspark_n_routed_experts if is_mtp else cfg.n_routed_experts
        exp_per_lane = n_exp // ep_lanes
        inter_per_host = cfg.moe_inter_dim // tp_hosts

        w1_u32 = np.asarray(norm_w[fp + "experts.w1.u32"])
        w1_s = np.asarray(norm_w[fp + "experts.w1.scale"])
        w3_u32 = np.asarray(norm_w[fp + "experts.w3.u32"])
        w3_s = np.asarray(norm_w[fp + "experts.w3.scale"])
        w2_u32 = np.asarray(norm_w[fp + "experts.w2.u32"])
        w2_s = np.asarray(norm_w[fp + "experts.w2.scale"])

        rank_w1_u32, rank_w1_s = [], []
        rank_w3_u32, rank_w3_s = [], []
        rank_w2_u32, rank_w2_s = [], []
        for rank in range(num_devices):
            lane = rank % ep_lanes
            host = rank // ep_lanes
            e_slice = slice(lane * exp_per_lane, (lane + 1) * exp_per_lane)
            i_slice = slice(host * inter_per_host, (host + 1) * inter_per_host)
            i_u32_slice = slice(host * (inter_per_host // 8), (host + 1) * (inter_per_host // 8))
            i_s_slice = slice(host * (inter_per_host // 32), (host + 1) * (inter_per_host // 32))

            rank_w1_u32.append(w1_u32[e_slice, :, i_slice])
            rank_w1_s.append(w1_s[e_slice, :, i_slice])
            rank_w3_u32.append(w3_u32[e_slice, :, i_slice])
            rank_w3_s.append(w3_s[e_slice, :, i_slice])
            rank_w2_u32.append(w2_u32[e_slice, i_u32_slice, :])
            rank_w2_s.append(w2_s[e_slice, i_s_slice, :])

        sharded[fp + "experts.w1.u32"] = _put_sharded(np.stack(rank_w1_u32, axis=0))
        sharded[fp + "experts.w1.scale"] = _put_sharded(np.stack(rank_w1_s, axis=0))
        sharded[fp + "experts.w3.u32"] = _put_sharded(np.stack(rank_w3_u32, axis=0))
        sharded[fp + "experts.w3.scale"] = _put_sharded(np.stack(rank_w3_s, axis=0))
        sharded[fp + "experts.w2.u32"] = _put_sharded(np.stack(rank_w2_u32, axis=0))
        sharded[fp + "experts.w2.scale"] = _put_sharded(np.stack(rank_w2_s, axis=0))

        sp = fp + "shared_experts."
        shared_tp = min(ep_lanes, cfg.moe_inter_dim // 32)
        s_inter = cfg.moe_inter_dim // shared_tp
        sw1 = np.asarray(norm_w[sp + "w1.weight"])
        sw1_s = np.asarray(norm_w[sp + "w1.scale"])
        sw3 = np.asarray(norm_w[sp + "w3.weight"])
        sw3_s = np.asarray(norm_w[sp + "w3.scale"])
        sw2 = np.asarray(norm_w[sp + "w2.weight"])
        sw2_s = np.asarray(norm_w[sp + "w2.scale"])

        r_sw1, r_sw1_s, r_sw3, r_sw3_s, r_sw2, r_sw2_s = [], [], [], [], [], []
        for rank in range(num_devices):
            s_idx = rank % shared_tp
            sl = slice(s_idx * s_inter, (s_idx + 1) * s_inter)
            sl_blk = slice(s_idx * (s_inter // 32), (s_idx + 1) * (s_inter // 32))
            r_sw1.append(sw1[sl, :])
            r_sw1_s.append(sw1_s[sl_blk, :])
            r_sw3.append(sw3[sl, :])
            r_sw3_s.append(sw3_s[sl_blk, :])
            r_sw2.append(sw2[:, sl])
            r_sw2_s.append(sw2_s[:, sl_blk])
        sharded[sp + "w1.weight"] = _put_sharded(np.stack(r_sw1, axis=0))
        sharded[sp + "w1.scale"] = _put_sharded(np.stack(r_sw1_s, axis=0))
        sharded[sp + "w3.weight"] = _put_sharded(np.stack(r_sw3, axis=0))
        sharded[sp + "w3.scale"] = _put_sharded(np.stack(r_sw3_s, axis=0))
        sharded[sp + "w2.weight"] = _put_sharded(np.stack(r_sw2, axis=0))
        sharded[sp + "w2.scale"] = _put_sharded(np.stack(r_sw2_s, axis=0))

        if not is_mtp and cfg.has_engram(l_id):
            ep = prefix + "engram."
            sharded[ep + "embed.weight"] = _replicate(norm_w[ep + "embed.weight"])
            sharded[ep + "embed.scale"] = _replicate(norm_w[ep + "embed.scale"])
            ewkv_w = np.asarray(norm_w[ep + "wkv.weight"])
            ewkv_s = np.asarray(norm_w[ep + "wkv.scale"])
            if (ewkv_w.shape[0] // num_devices) % 32 == 0 and ewkv_w.shape[0] % num_devices == 0:
                rows_r = ewkv_w.shape[0] // num_devices
                blk_r = ewkv_s.shape[0] // num_devices
                sharded[ep + "wkv.weight"] = _put_sharded(ewkv_w.reshape(num_devices, rows_r, ewkv_w.shape[1]))
                sharded[ep + "wkv.scale"] = _put_sharded(ewkv_s.reshape(num_devices, blk_r, ewkv_s.shape[1]))
            else:
                sharded[ep + "wkv.weight"] = _replicate(ewkv_w)
                sharded[ep + "wkv.scale"] = _replicate(ewkv_s)
            sharded[ep + "q_weight"] = _replicate(norm_w[ep + "q_weight"])
            sharded[ep + "k_weight"] = _replicate(norm_w[ep + "k_weight"])

        if is_mtp:
            stage_id = l_id - cfg.n_layers
            if stage_id == 0:
                sharded[prefix + "main_proj.weight"] = _replicate(norm_w[prefix + "main_proj.weight"])
                sharded[prefix + "main_proj.scale"] = _replicate(norm_w[prefix + "main_proj.scale"])
                sharded[prefix + "main_norm.weight"] = _replicate(norm_w[prefix + "main_norm.weight"])
            if stage_id == cfg.n_mtp_layers - 1:
                sharded[prefix + "norm.weight"] = _replicate(norm_w[prefix + "norm.weight"])
                sharded[prefix + "markov_head.embed.weight"] = _replicate(norm_w[prefix + "markov_head.embed.weight"])
                m_head = jnp.asarray(norm_w[prefix + "markov_head.head.weight"])
                sharded[prefix + "markov_head.head.weight"] = _put_sharded(
                    m_head.reshape(num_devices, m_head.shape[0] // num_devices, m_head.shape[1])
                )
                sharded[prefix + "confidence_head.proj.weight"] = _replicate(
                    norm_w[prefix + "confidence_head.proj.weight"]
                )

    return sharded


# ---------------------------------------------------------------------------
# 6. Core Layer & Step Execution (Shared by Reference and Sharded XLA Control)
# ---------------------------------------------------------------------------

def _lookup_engram_rows(
    hash_ids: jax.Array,
    embed_w: jax.Array,
    embed_s: jax.Array,
    *,
    num_embeddings: int | None = None,
    sharded_tp: bool = False,
    num_devices: int = 1,
    axis_name: str = "tp",
) -> jax.Array:
    """Lookup 24 FP8 rows + E8M0 scales and dequantize to `[B, S, 24 * head_dim]` BF16."""
    orig_prefix = hash_ids.shape[:-1]
    n_hashes = hash_ids.shape[-1]
    m = int(np.prod(orig_prefix))
    pad_m = ((max(m, 8) + 7) // 8) * 8
    h_pad = hash_ids.reshape(m, n_hashes)
    if pad_m > m:
        h_pad = jnp.pad(h_pad, ((0, pad_m - m), (0, 0)))
    h_pad = _cpu_pad_barrier(h_pad)
    part_n = (
        (int(num_embeddings) + num_devices - 1) // num_devices
        if (num_embeddings is not None and num_devices > 0)
        else 0
    )
    if sharded_tp and num_devices > 1 and num_embeddings is not None and embed_w.shape[0] == part_n:
        rank = jax.lax.axis_index(axis_name)
        start_idx = rank * part_n
        end_idx = jnp.minimum(start_idx + part_n, jnp.int32(num_embeddings))
        in_range = (h_pad >= start_idx) & (h_pad < end_idx)
        local_ids = jnp.where(in_range, h_pad - start_idx, jnp.int32(0))
        rows_fp8 = embed_w[local_ids]
        rows_s = e8m0_to_f32_jax(embed_s[local_ids])
        deq = (rows_fp8.astype(jnp.float32).reshape(*rows_s.shape, 32) * rows_s[..., None]).reshape(
            pad_m, n_hashes, -1
        ).astype(jnp.bfloat16)
        deq = jnp.where(in_range[..., None], deq, jnp.bfloat16(0.0))
        flat = deq.reshape(pad_m, -1)
        out_pad = _cpu_pad_barrier(jax.lax.psum(flat.astype(jnp.float32), axis_name).astype(jnp.bfloat16))
        return out_pad[:m].reshape(*orig_prefix, -1)
    limit = embed_w.shape[0] if num_embeddings is None else min(int(embed_w.shape[0]), int(num_embeddings))
    in_range = (h_pad >= 0) & (h_pad < limit)
    safe_ids = jnp.where(in_range, h_pad, jnp.int32(0))
    rows_fp8 = embed_w[safe_ids]
    rows_s = e8m0_to_f32_jax(embed_s[safe_ids])
    deq = (rows_fp8.astype(jnp.float32).reshape(*rows_s.shape, 32) * rows_s[..., None]).reshape(
        pad_m, n_hashes, -1
    ).astype(jnp.bfloat16)
    deq = jnp.where(in_range[..., None], deq, jnp.bfloat16(0.0))
    out_pad = _cpu_pad_barrier(deq.reshape(pad_m, -1))
    return out_pad[:m].reshape(*orig_prefix, -1)


def _compute_engram_hashes_jax(
    cfg: DSV41Config,
    token_history: jax.Array,
    start_pos: jax.Array,
    seqlen: int,
    token_map: jax.Array | None = None,
) -> jax.Array:
    """Pure JAX uint32 exact 64-bit Engram hashing inside jit for `[B, seqlen]` starting at `start_pos`."""
    primes_np, offsets_np, mult_np = build_engram_tables(cfg)
    primes_u32 = jnp.asarray(primes_np, dtype=jnp.uint32)
    offsets_u32 = jnp.asarray(offsets_np, dtype=jnp.uint32)
    m0 = jnp.asarray((mult_np >> 0) & 0xFFFF, dtype=jnp.uint32)
    m1 = jnp.asarray((mult_np >> 16) & 0xFFFF, dtype=jnp.uint32)
    m2 = jnp.asarray((mult_np >> 32) & 0xFFFF, dtype=jnp.uint32)
    m3 = jnp.asarray((mult_np >> 48) & 0xFFFF, dtype=jnp.uint32)
    if token_map is None:
        default_c = getattr(cfg, "engram_compressed_vocab_size", None)
        if default_c is None:
            default_c = 99092 if cfg.vocab_size == 129280 else cfg.vocab_size
        token_map_np, _ = build_compressed_token_map(None, vocab_size=cfg.vocab_size, compressed_vocab_size=default_c)
        token_map = jnp.asarray(token_map_np, dtype=jnp.int32)
    else:
        token_map = token_map.astype(jnp.int32)

    bsz = token_history.shape[0]
    max_n = cfg.engram_max_ngram_size
    pad_id = token_map[cfg.engram_pad_id]
    positions = (start_pos + jnp.arange(seqlen, dtype=jnp.int32))[None, :]
    tokens = []
    blocked = jnp.zeros((bsz, seqlen), dtype=jnp.bool_)
    for shift in range(max_n):
        idx = jnp.clip(positions - shift, 0, token_history.shape[1] - 1)
        raw_tok = jnp.take_along_axis(token_history, jnp.broadcast_to(idx, (bsz, seqlen)), axis=1)
        blocked = blocked | (positions < shift) | (raw_tok < 0)
        comp_tok = token_map[jnp.clip(raw_tok, 0, cfg.vocab_size - 1)]
        tokens.append(jnp.where(blocked, pad_id, comp_tok))
    tok_arr = jnp.stack(tokens, axis=-1).astype(jnp.uint32)[:, :, None, :]  # [B, S, 1, max_n]

    t_lo = tok_arr & jnp.uint32(0xFFFF)
    t_hi = tok_arr >> jnp.uint32(16)
    p0 = t_lo * m0[None, None, :, :]
    p1 = t_lo * m1[None, None, :, :] + t_hi * m0[None, None, :, :] + (p0 >> jnp.uint32(16))
    p2 = t_lo * m2[None, None, :, :] + t_hi * m1[None, None, :, :] + (p1 >> jnp.uint32(16))
    p3 = t_lo * m3[None, None, :, :] + t_hi * m2[None, None, :, :] + (p2 >> jnp.uint32(16))
    prod_lo = (p0 & jnp.uint32(0xFFFF)) | ((p1 & jnp.uint32(0xFFFF)) << jnp.uint32(16))
    prod_hi = (p2 & jnp.uint32(0xFFFF)) | ((p3 & jnp.uint32(0xFFFF)) << jnp.uint32(16))

    roll_lo = prod_lo[..., 0]
    roll_hi = prod_hi[..., 0]
    hashes = []
    for i in range(1, max_n):
        roll_lo = jnp.bitwise_xor(roll_lo, prod_lo[..., i])
        roll_hi = jnp.bitwise_xor(roll_hi, prod_hi[..., i])
        p_mod = primes_u32[None, None, :, i - 1, :]  # [1, 1, L_e, H]
        r_hi = roll_hi[..., None]
        r_lo = roll_lo[..., None]
        rem = ((r_hi >> jnp.uint32(24)) & jnp.uint32(0xFF)) % p_mod
        for shift_b in (16, 8, 0):
            b = (r_hi >> jnp.uint32(shift_b)) & jnp.uint32(0xFF)
            rem = (rem * jnp.uint32(256) + b) % p_mod
        for shift_b in (24, 16, 8, 0):
            b = (r_lo >> jnp.uint32(shift_b)) & jnp.uint32(0xFF)
            rem = (rem * jnp.uint32(256) + b) % p_mod
        hashes.append(rem)
    raw_ids = jnp.concatenate(hashes, axis=-1) + offsets_u32[None, None, :, :]
    return raw_ids.astype(jnp.int32)


def _single_token_step_body(
    cfg: DSV41Config,
    weights: dict[str, jax.Array],
    cache: dict[str, jax.Array],
    tokens_1: jax.Array,
    pos: jax.Array,
    *,
    sharded: bool,
    num_devices: int,
    sinkhorn_iters: int,
    index_k_mode: str,
    axis_name: str = "tp",
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Run one token step `[B, 1]` at scalar `pos` across all `cfg.n_layers` backbone layers."""
    bsz = tokens_1.shape[0]
    pos_i32 = pos.astype(jnp.int32)

    token_history = cache["token_history"].at[:, pos_i32].set(tokens_1[:, 0])
    engram_hashes = _compute_engram_hashes_jax(
        cfg, token_history, pos_i32, 1, token_map=weights.get("engram.token_map")
    )

    embed_w = weights["embed.weight"]
    if sharded and embed_w.shape[0] == cfg.padded_vocab_size // num_devices:
        rank = jax.lax.axis_index(axis_name)
        part_v = embed_w.shape[0]
        local_id = tokens_1 - rank * part_v
        in_range = (local_id >= 0) & (local_id < part_v)
        local_emb = jnp.where(in_range[..., None], embed_w[jnp.clip(local_id, 0, part_v - 1)], jnp.bfloat16(0.0))
        h0 = jax.lax.psum(local_emb.astype(jnp.float32), axis_name).astype(jnp.bfloat16)
    else:
        h0 = embed_w[tokens_1]

    h = jnp.repeat(h0[:, :, None, :], cfg.hc_mult, axis=2)
    pre_mix = jnp.zeros((bsz, 1, cfg.hc_mult), dtype=jnp.float32).at[:, :, 0].set(1.0)

    window_kv_all = cache["window_kv"]
    compress_kv_all = cache["compress_kv"]
    index_k_all = cache["index_k"]
    comp_kv_state_all = cache["comp_kv_state"]
    comp_score_state_all = cache["comp_score_state"]
    comp_kv_ring_all = cache.get("comp_kv_ring")
    comp_score_ring_all = cache.get("comp_score_ring")
    last_index_k_owner = cache["last_index_k_owner"]

    shared_compress_kv: jax.Array | None = None
    shared_topk_idxs: jax.Array | None = None
    shared_candidates: jax.Array | None = None
    main_hiddens = []

    ep_lanes = min(num_devices, 8) if sharded else 1
    shared_tp = min(ep_lanes, cfg.moe_inter_dim // 32) if sharded else 1

    for l_id in range(cfg.n_layers):
        prefix = f"layers.{l_id}."
        r = cfg.compress_ratios[l_id]

        if cfg.has_engram(l_id):
            ep = prefix + "engram."
            e_idx = cfg.engram_layer_ids.index(l_id)
            e_flat = _lookup_engram_rows(
                engram_hashes[:, :, e_idx, :],
                weights[ep + "embed.weight"],
                weights[ep + "embed.scale"],
                num_embeddings=cfg.engram_num_embeddings[e_idx],
                sharded_tp=sharded,
                num_devices=num_devices,
                axis_name=axis_name,
            )
            h = engram_forward(
                h,
                e_flat,
                weights[ep + "wkv.weight"],
                weights[ep + "wkv.scale"],
                weights[ep + "q_weight"],
                weights[ep + "k_weight"],
                cfg,
                sharded_tp=sharded,
                axis_name=axis_name,
            )

        if l_id in cfg.dspark_target_layer_ids:
            main_hiddens.append(jnp.mean(h.astype(jnp.float32), axis=2).astype(jnp.bfloat16))

        residual = h
        a_pre, a_post, a_comb = hc_mixes(
            h,
            weights[prefix + "hc_attn_fn"],
            weights[prefix + "hc_attn_scale"],
            weights[prefix + "hc_attn_base"],
            hc_mult=cfg.hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            hc_eps=cfg.hc_eps,
            norm_eps=cfg.rms_norm_eps,
        )
        u = rms_norm(hc_pre(h, pre_mix), weights[prefix + "attn_norm.weight"], cfg.rms_norm_eps)

        ap = prefix + "attn."
        rope_key = "rope.yarn." if r > 0 else "rope.swa."
        cos_pos = weights[rope_key + "cos"][pos_i32][None]
        sin_pos = weights[rope_key + "sin"][pos_i32][None]

        qr = rms_norm(
            fp8_linear(u, weights[ap + "wq_a.weight"], weights[ap + "wq_a.scale"]),
            weights[ap + "q_norm.weight"],
            cfg.rms_norm_eps,
        )
        q_raw = fp8_linear(qr, weights[ap + "wq_b.weight"], weights[ap + "wq_b.scale"])
        n_local_heads = cfg.n_heads // num_devices if sharded else cfg.n_heads
        q = q_raw.reshape(bsz, 1, n_local_heads, cfg.head_dim)
        q = apply_rotary_emb(q, cos_pos, sin_pos, cfg.qk_rope_head_dim, inverse=False)

        win = cfg.sliding_window
        kv_raw = rms_norm(
            fp8_linear(u, weights[ap + "wkv.weight"], weights[ap + "wkv.scale"]),
            weights[ap + "kv_norm.weight"],
            cfg.rms_norm_eps,
        )
        kv_rot = apply_rotary_emb(kv_raw, cos_pos, sin_pos, cfg.qk_rope_head_dim, inverse=False)
        kv_quant = act_quant(kv_rot, 32, inplace=True)
        win_layer = window_kv_all[l_id].at[:, pos_i32 % win].set(kv_quant[:, 0])
        window_kv_all = window_kv_all.at[l_id].set(win_layer)

        oldest = (pos_i32 % win) + 1
        ring_order = (oldest + jnp.arange(win, dtype=jnp.int32)) % win
        win_idxs_1d = jnp.where(ring_order > pos_i32, jnp.int32(-1), ring_order)
        topk_idxs = jnp.broadcast_to(win_idxs_1d[None, None, :], (bsz, 1, win))
        full_kv = win_layer

        if r > 0:
            is_kv_src = l_id in cfg.kv_source_layer_ids
            is_idx_src = l_id in cfg.index_source_layer_ids
            kv_src_idx = cfg.kv_source_layer_ids.index(cfg.kv_source_for(l_id))
            compress_len = (pos_i32 + 1) // r

            has_new_latent = jnp.bool_(False)
            latent_pre_rope = jnp.zeros((bsz, 1, cfg.head_dim), dtype=jnp.bfloat16)

            if is_kv_src:
                cp = ap + "compressor."
                if r == 1:
                    has_new_latent = jnp.bool_(True)
                    wkv_c = weights[cp + "wkv.weight"].astype(jnp.float32)
                    lat = jnp.einsum("bsd,hd->bsh", u.astype(jnp.float32), wkv_c, precision=jax.lax.Precision.HIGHEST)
                    latent_pre_rope = rms_norm(lat.astype(jnp.bfloat16), weights[cp + "norm.weight"], cfg.rms_norm_eps)
                else:
                    slot = pos_i32 % r
                    ring_slot = pos_i32 % 8
                    wkv_c = weights[cp + "wkv.weight"].astype(jnp.float32)
                    wgate_c = weights[cp + "wgate.weight"].astype(jnp.float32)
                    kv_c = jnp.einsum("bsd,hd->bsh", u.astype(jnp.float32), wkv_c, precision=jax.lax.Precision.HIGHEST)
                    sc_c = jnp.einsum(
                        "bsd,hd->bsh", u.astype(jnp.float32), wgate_c, precision=jax.lax.Precision.HIGHEST
                    )
                    cur_kv_state = comp_kv_state_all[kv_src_idx].at[:, slot].set(kv_c[:, 0])
                    cur_sc_state = comp_score_state_all[kv_src_idx].at[:, slot].set(sc_c[:, 0])
                    comp_kv_state_all = comp_kv_state_all.at[kv_src_idx].set(cur_kv_state)
                    comp_score_state_all = comp_score_state_all.at[kv_src_idx].set(cur_sc_state)
                    if comp_kv_ring_all is not None and comp_score_ring_all is not None:
                        comp_kv_ring_all = comp_kv_ring_all.at[kv_src_idx, :, ring_slot].set(kv_c[:, 0])
                        comp_score_ring_all = comp_score_ring_all.at[kv_src_idx, :, ring_slot].set(sc_c[:, 0])
                    has_new_latent = ((pos_i32 + 1) % r) == 0
                    pooled = jnp.sum(cur_kv_state * jax.nn.softmax(cur_sc_state, axis=1), axis=1, keepdims=True)
                    latent_pre_rope = rms_norm(
                        pooled.astype(jnp.bfloat16), weights[cp + "norm.weight"], cfg.rms_norm_eps
                    )
                shared_compress_kv = compress_kv_all[kv_src_idx]

            if is_idx_src:
                ip = ap + "indexer."
                if is_kv_src:
                    grp_pos = jnp.maximum(pos_i32 + 1 - r, 0)
                    cos_grp = weights[rope_key + "cos"][grp_pos][None]
                    sin_grp = weights[rope_key + "sin"][grp_pos][None]
                    wk_i = weights[ip + "wk.weight"].astype(jnp.float32)
                    k_raw = jnp.einsum(
                        "bsd,hd->bsh",
                        latent_pre_rope.astype(jnp.float32),
                        wk_i,
                        precision=jax.lax.Precision.HIGHEST,
                    ).astype(jnp.bfloat16)
                    k_normed = rms_norm(k_raw, weights[ip + "k_norm.weight"], cfg.rms_norm_eps)
                    k_rot = apply_rotary_emb(k_normed, cos_grp, sin_grp, cfg.qk_rope_head_dim, inverse=False)
                    k_quant = fp4_act_quant(k_rot, 32, inplace=True, scale_fmt="e8m0")
                    updated_k = jnp.where(
                        has_new_latent,
                        index_k_all[kv_src_idx].at[:, pos_i32 // r].set(k_quant[:, 0]),
                        index_k_all[kv_src_idx],
                    )
                    index_k_all = index_k_all.at[kv_src_idx].set(updated_k)
                    last_index_k_owner = jnp.where(has_new_latent, jnp.int32(kv_src_idx), last_index_k_owner)

                active_owner = last_index_k_owner if index_k_mode == "reference" else jnp.int32(kv_src_idx)
                active_k_cache = index_k_all[active_owner]

                n_local_idx_heads = cfg.index_n_heads // num_devices if sharded else cfg.index_n_heads
                qi_raw = fp8_linear(qr, weights[ip + "wq_b.weight"], weights[ip + "wq_b.scale"]).reshape(
                    bsz, 1, n_local_idx_heads, cfg.index_head_dim
                )
                qi_rot = apply_rotary_emb(qi_raw, cos_pos, sin_pos, cfg.qk_rope_head_dim, inverse=False)
                qi_quant = fp4_act_quant(qi_rot, 32, inplace=True, scale_fmt="e8m0")

                w_proj = weights[ip + "weights_proj.weight"].astype(jnp.float32)
                idx_scale = (cfg.index_head_dim**-0.5) * (cfg.index_n_heads**-0.5)
                w_heads = (
                    jnp.einsum("bsd,hd->bsh", u.astype(jnp.float32), w_proj, precision=jax.lax.Precision.HIGHEST)
                    * jnp.float32(idx_scale)
                ).astype(jnp.bfloat16)

                raw_dot = jnp.einsum(
                    "bshd,btd->bsht",
                    qi_quant.astype(jnp.float32),
                    active_k_cache.astype(jnp.float32),
                    precision=jax.lax.Precision.HIGHEST,
                ).astype(jnp.bfloat16)
                head_weighted = (jnp.maximum(raw_dot, jnp.bfloat16(0.0)) * w_heads[..., None]).astype(jnp.float32)
                score_sum = jnp.sum(head_weighted, axis=2)
                if sharded:
                    score_sum = jax.lax.psum(score_sum, axis_name)
                index_score = score_sum.astype(jnp.bfloat16).astype(jnp.float32)[:, 0, :]

                t_iota = jnp.arange(active_k_cache.shape[1], dtype=jnp.int32)[None, :]
                index_score = jnp.where(t_iota < compress_len, index_score, -jnp.inf)

                if l_id == cfg.candidate_source_layer_id:
                    shared_candidates = select_candidate_blocks(
                        index_score,
                        jnp.full((bsz,), compress_len, dtype=jnp.int32),
                        cfg.candidate_topk_blocks,
                        cfg.candidate_block_size,
                    )
                elif l_id > cfg.candidate_source_layer_id and shared_candidates is not None:
                    index_score = jnp.where(shared_candidates, index_score, -jnp.inf)

                _, top_pos = jax.lax.top_k(index_score, cfg.index_topk)
                k_keep = jnp.minimum(jnp.int32(cfg.index_topk), compress_len)
                rank_iota = jnp.arange(cfg.index_topk, dtype=jnp.int32)[None, :]
                masked_for_sort = jnp.where(rank_iota < k_keep, top_pos, jnp.int32(1 << 29))
                sorted_pos = jnp.sort(masked_for_sort, axis=-1)
                comp_idxs = jnp.where(
                    (rank_iota < k_keep) & (sorted_pos < compress_len),
                    sorted_pos + win,
                    jnp.int32(-1),
                )
                shared_topk_idxs = comp_idxs[:, None, :]

            if is_kv_src:
                grp_pos = jnp.maximum(pos_i32 + 1 - r, 0)
                cos_grp = weights[rope_key + "cos"][grp_pos][None]
                sin_grp = weights[rope_key + "sin"][grp_pos][None]
                lat_rot = apply_rotary_emb(latent_pre_rope, cos_grp, sin_grp, cfg.qk_rope_head_dim, inverse=False)
                lat_quant = fp4_act_quant(lat_rot, 16, inplace=True, scale_fmt="e4m3")
                updated_ckv = jnp.where(
                    has_new_latent,
                    compress_kv_all[kv_src_idx].at[:, pos_i32 // r].set(lat_quant[:, 0]),
                    compress_kv_all[kv_src_idx],
                )
                compress_kv_all = compress_kv_all.at[kv_src_idx].set(updated_ckv)
                shared_compress_kv = updated_ckv

            assert shared_compress_kv is not None and shared_topk_idxs is not None
            full_kv = jnp.concatenate([win_layer, shared_compress_kv], axis=1)
            topk_idxs = jnp.concatenate([topk_idxs, shared_topk_idxs], axis=-1)

        o = sparse_attn(q, full_kv, weights[ap + "attn_sink"], topk_idxs, cfg.head_dim**-0.5)
        o = apply_rotary_emb(o, cos_pos, sin_pos, cfg.qk_rope_head_dim, inverse=True)

        wo_a = weights[ap + "wo_a.weight"]
        if sharded and num_devices >= cfg.o_groups:
            ranks_per_group = num_devices // cfg.o_groups
            rank = jax.lax.axis_index(axis_name)
            o_flat = o.reshape(bsz, 1, -1).astype(jnp.float32)
            o_group_partial = jnp.einsum(
                "bsd,rd->bsr", o_flat, wo_a.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST
            )
            if ranks_per_group > 1:
                gathered_partials = jax.lax.all_gather(o_group_partial, axis_name, axis=0)
                grp_base = (rank // ranks_per_group) * ranks_per_group
                grp_partials = jax.lax.dynamic_slice_in_dim(gathered_partials, grp_base, ranks_per_group, axis=0)
                o_group_full = jnp.sum(grp_partials, axis=0).astype(jnp.bfloat16)
                sub_idx = rank % ranks_per_group
                sub_width = cfg.o_lora_rank // ranks_per_group
                o_mid = jax.lax.dynamic_slice_in_dim(o_group_full, sub_idx * sub_width, sub_width, axis=-1)
            else:
                o_mid = o_group_partial.astype(jnp.bfloat16)
        elif sharded:
            groups_per_rank = cfg.o_groups // num_devices
            o_grp = o.reshape(bsz, 1, groups_per_rank, -1).astype(jnp.float32)
            o_mid = jnp.einsum(
                "bsgd,grd->bsgr", o_grp, wo_a.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST
            ).reshape(bsz, 1, -1).astype(jnp.bfloat16)
        else:
            o_grp = o.reshape(bsz, 1, cfg.o_groups, -1).astype(jnp.float32)
            o_mid = jnp.einsum(
                "bsgd,grd->bsgr", o_grp, wo_a.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST
            ).reshape(bsz, 1, -1).astype(jnp.bfloat16)

        attn_out_f32 = fp8_linear(
            o_mid, weights[ap + "wo_b.weight"], weights[ap + "wo_b.scale"], out_dtype=jnp.float32
        )
        if sharded:
            attn_out_f32 = jax.lax.psum(attn_out_f32, axis_name)
        attn_out = attn_out_f32.astype(jnp.bfloat16)

        h = hc_post(attn_out, residual, a_post, a_comb)

        residual = h
        f_pre, f_post, f_comb = hc_mixes(
            h,
            weights[prefix + "hc_ffn_fn"],
            weights[prefix + "hc_ffn_scale"],
            weights[prefix + "hc_ffn_base"],
            hc_mult=cfg.hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            hc_eps=cfg.hc_eps,
            norm_eps=cfg.rms_norm_eps,
        )
        v = rms_norm(hc_pre(h, a_pre), weights[prefix + "ffn_norm.weight"], cfg.rms_norm_eps)

        fp = prefix + "ffn."
        top_w, top_idx = moe_gate(
            v[:, 0, :],
            weights[fp + "gate.weight"],
            weights[fp + "gate.bias"],
            cfg.n_activated_experts,
            cfg.route_scale,
        )

        w1_u32 = weights[fp + "experts.w1.u32"]
        w1_s = weights[fp + "experts.w1.scale"]
        w3_u32 = weights[fp + "experts.w3.u32"]
        w3_s = weights[fp + "experts.w3.scale"]
        w2_u32 = weights[fp + "experts.w2.u32"]
        w2_s = weights[fp + "experts.w2.scale"]
        n_local_exp = w1_u32.shape[0]
        if sharded:
            lane = jax.lax.axis_index(axis_name) % ep_lanes
            exp_start = lane * n_local_exp
        else:
            exp_start = jnp.int32(0)

        e_local_all = top_idx - exp_start  # [B, k_act]

        def _scan_local_expert(acc: jax.Array, exp_inputs: tuple[jax.Array, ...]) -> tuple[jax.Array, None]:
            e_idx, w1_e, w1_se, w3_e, w3_se, w2_e, w2_se = exp_inputs
            match = e_local_all == e_idx
            is_active = jnp.any(match, axis=-1)
            w_e = jnp.sum(jnp.where(match, top_w, jnp.float32(0.0)), axis=-1)
            out_e = run_single_expert_fp4(
                v[:, 0, :],
                w_e,
                w1_e,
                w1_se,
                w3_e,
                w3_se,
                w2_e,
                w2_se,
                cfg.swiglu_limit,
                out_dtype=jnp.bfloat16 if num_devices <= 8 else jnp.float32,
            ).astype(jnp.float32)
            return acc + jnp.where(is_active[:, None], out_e, jnp.float32(0.0)), None

        routed_acc, _ = jax.lax.scan(
            _scan_local_expert,
            jnp.zeros((bsz, cfg.dim), dtype=jnp.float32),
            (jnp.arange(n_local_exp, dtype=jnp.int32), w1_u32, w1_s, w3_u32, w3_s, w2_u32, w2_s),
        )

        if sharded:
            routed_acc = jax.lax.psum(routed_acc, axis_name)

        sp = fp + "shared_experts."
        shared_part = run_shared_expert_fp8(
            v[:, 0, :],
            weights[sp + "w1.weight"],
            weights[sp + "w1.scale"],
            weights[sp + "w3.weight"],
            weights[sp + "w3.scale"],
            weights[sp + "w2.weight"],
            weights[sp + "w2.scale"],
            cfg.swiglu_limit,
            out_dtype=jnp.float32,
        ).astype(jnp.float32)
        if sharded:
            shared_rep = num_devices // shared_tp
            shared_full = (jax.lax.psum(shared_part, axis_name) / jnp.float32(shared_rep)).astype(jnp.bfloat16).astype(jnp.float32)
        else:
            shared_full = shared_part.astype(jnp.bfloat16).astype(jnp.float32)

        ffn_out = (routed_acc + shared_full).astype(jnp.bfloat16)[:, None, :]
        h = hc_post(ffn_out, residual, f_post, f_comb)
        pre_mix = f_pre

    y_final = rms_norm(hc_pre(h, pre_mix), weights["norm.weight"], cfg.rms_norm_eps)
    head_w = weights["head.weight"].astype(jnp.float32)
    logits_local = jnp.einsum(
        "bd,vd->bv", y_final[:, 0, :].astype(jnp.float32), head_w, precision=jax.lax.Precision.HIGHEST
    )
    if sharded and head_w.shape[0] == cfg.padded_vocab_size // num_devices:
        logits_full = jax.lax.all_gather(logits_local, axis_name, axis=-1, tiled=True)
    else:
        logits_full = logits_local
    logits = logits_full[:, : cfg.vocab_size]
    main_hidden = jnp.concatenate(main_hiddens, axis=-1)[:, 0, :]

    new_cache = {
        **cache,
        "window_kv": window_kv_all,
        "compress_kv": compress_kv_all,
        "index_k": index_k_all,
        "comp_kv_state": comp_kv_state_all,
        "comp_score_state": comp_score_state_all,
        "last_index_k_owner": last_index_k_owner,
        "token_history": token_history,
    }
    if comp_kv_ring_all is not None and comp_score_ring_all is not None:
        new_cache["comp_kv_ring"] = comp_kv_ring_all
        new_cache["comp_score_ring"] = comp_score_ring_all
    return logits, main_hidden, new_cache


def reference_decode_step(
    cfg: DSV41Config,
    weights: Mapping[str, Any],
    cache: dict[str, jax.Array],
    tokens: jax.Array,
    start_pos: int | jax.Array,
    *,
    sinkhorn_iters: int | None = None,
    index_k_mode: str | None = None,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Unsharded single-device JAX reference step for `[B]` or `[B, q]` tokens."""
    norm_w = (
        normalize_weights(cfg, weights, max_seq_len=cache["compress_kv"].shape[2])
        if "rope.swa.cos" not in weights or "layers.0.ffn.experts.w1.u32" not in weights
        else dict(weights)
    )
    s_iters = cfg.hc_sinkhorn_iters if sinkhorn_iters is None else int(sinkhorn_iters)
    ik_mode = cfg.index_k_mode if index_k_mode is None else index_k_mode
    tok = jnp.asarray(tokens, dtype=jnp.int32)
    squeeze_seq = tok.ndim == 1
    if squeeze_seq:
        tok = tok[:, None]
    pos0 = jnp.asarray(start_pos, dtype=jnp.int32)

    q_len = tok.shape[1]
    logits_list = []
    hidden_list = []
    cur_cache = cache
    for k in range(q_len):
        lg, mh, cur_cache = _single_token_step_body(
            cfg,
            norm_w,
            cur_cache,
            tok[:, k : k + 1],
            pos0 + k,
            sharded=False,
            num_devices=1,
            sinkhorn_iters=s_iters,
            index_k_mode=ik_mode,
        )
        logits_list.append(lg)
        hidden_list.append(mh)

    if squeeze_seq:
        return logits_list[0], hidden_list[0], cur_cache
    return jnp.stack(logits_list, axis=1), jnp.stack(hidden_list, axis=1), cur_cache


def make_xla_decode(
    cfg: DSV41Config,
    mesh: Mesh,
    *,
    sinkhorn_iters: int | None = None,
    index_k_mode: str | None = None,
    donate: bool = False,
):
    """Compile the sharded XLA decode / verify step (`@jax.jit` over `jax.shard_map` on `"tp"`)."""
    num_devices = mesh.size
    s_iters = cfg.hc_sinkhorn_iters if sinkhorn_iters is None else int(sinkhorn_iters)
    ik_mode = cfg.index_k_mode if index_k_mode is None else index_k_mode

    def _local_fn(sharded_weights, cache, tokens, start_pos):
        w_local = jax.tree.map(lambda x: x[0], sharded_weights)
        tok = tokens.astype(jnp.int32)
        squeeze_seq = tok.ndim == 1
        if squeeze_seq:
            tok = tok[:, None]
        pos0 = start_pos.astype(jnp.int32)

        logits_list = []
        hidden_list = []
        cur_cache = cache
        for k in range(tok.shape[1]):
            lg, mh, cur_cache = _single_token_step_body(
                cfg,
                w_local,
                cur_cache,
                tok[:, k : k + 1],
                pos0 + k,
                sharded=True,
                num_devices=num_devices,
                sinkhorn_iters=s_iters,
                index_k_mode=ik_mode,
                axis_name="tp",
            )
            logits_list.append(lg)
            hidden_list.append(mh)

        if squeeze_seq:
            return logits_list[0], hidden_list[0], cur_cache
        return jnp.stack(logits_list, axis=1), jnp.stack(hidden_list, axis=1), cur_cache

    sharded_step = jax.shard_map(
        _local_fn,
        mesh=mesh,
        in_specs=(P("tp"), P(), P(), P()),
        out_specs=(P(), P(), P()),
        check_vma=False,
    )
    jitted = jax.jit(sharded_step, donate_argnums=(1,) if donate else ())
    if len(mesh.local_devices) == mesh.size:
        return jitted

    def _multihost_step(sharded_weights, cache, tokens, start_pos):
        return jitted(
            sharded_weights,
            cache,
            replicate_to_mesh(jnp.asarray(tokens, dtype=jnp.int32), mesh),
            replicate_to_mesh(jnp.asarray(start_pos, dtype=jnp.int32), mesh),
        )

    return _multihost_step


make_xla_decode_step = make_xla_decode
shard_weights_for_mesh = shard_weights


