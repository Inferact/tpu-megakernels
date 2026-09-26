"""Symmetric int4 group quantization for the Muse Spark routed experts (design.md section 2).

Layout: a weight is `[..., K, N]` with K the contraction axis (`y = x @ w`). Groups are `G`
consecutive rows of K per output column. For each group

    scale = bf16_round(absmax * f32(1/7))        (f32 maths, round-to-nearest-even to bf16)
    q     = clip(round_half_even(w * (1/scale)), -8, 7)     (zero groups: scale = 1)

so `dequant = q * scale` is exact in f32 and `|w - dequant| <= scale/2` (up to one f32 ulp of
the reciprocal multiply). The multiplications by rounded reciprocals are spelled out because XLA
rewrites `x / c` and `x / broadcast(s)` into exactly that; the numpy and jnp quantizers are
bit-identical.

Scales are returned as **f32** holding bf16-representable values, flat `[..., K/G, N]`
("group-major"). The kernels index scales on a leading untiled axis, so the per-rank layout
stores them chunked: `[..., K/KC, KC/G, N]` with `KC = k_chunk(K) = min(512, K)` the static K
chunk of the in-kernel dot (`scales_to_chunked` / `scales_from_chunked`). Quantized values live
in `jnp.int4` (`ml_dtypes.int4`, one byte per element on the host); the numpy quantizer returns
int8 with the same values. On disk int4 values are packed two per byte along the LAST axis, low
nibble first (`pack_int4` / `unpack_int4`).
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax import lax

BF16 = ml_dtypes.bfloat16
QMIN, QMAX = -8, 7
LEVELS = 7
INV_LEVELS = np.float32(1.0 / LEVELS)
K_CHUNK = 512  # static K chunk of the in-kernel int4 dot (hw report section 3)


def k_chunk(K):
    """K chunk used to lay out the scales of a `[K, N]` matrix: `min(512, K)`."""
    if K % min(K_CHUNK, K):
        raise ValueError(f"K={K} is not a multiple of the K chunk {min(K_CHUNK, K)}")
    return min(K_CHUNK, K)


def _group(shape, group):
    K = shape[-2]
    if K % group:
        raise ValueError(f"group size {group} does not divide K={K}")
    return shape[:-2] + (K // group, group, shape[-1])


def quantize_int4_np(w, group=128):
    """numpy: f32 `[.., K, N]` -> (`q` int8 `[.., K, N]` in [-8,7], `scales` f32 `[.., K/G, N]`)."""
    w = np.asarray(w, np.float32)
    g = w.reshape(_group(w.shape, group))
    amax = np.max(np.abs(g), axis=-2)  # [..., K/G, N]
    scale = (amax * INV_LEVELS).astype(BF16).astype(np.float32)
    scale = np.where(scale == 0, np.float32(1), scale)
    q = np.rint(g * (np.float32(1) / scale)[..., None, :])
    q = np.clip(q, QMIN, QMAX).astype(np.int8).reshape(w.shape)
    return q, scale


def _quantize_int4_jnp(w, group):
    w = jnp.asarray(w, jnp.float32)
    g = w.reshape(_group(w.shape, group))
    amax = jnp.max(jnp.abs(g), axis=-2)
    # reduce_precision is the bf16 rounding XLA cannot elide (a bf16->f32 convert pair may be
    # optimised away under the default --xla_allow_excess_precision=true).
    scale = lax.reduce_precision(amax * INV_LEVELS, exponent_bits=8, mantissa_bits=7)
    scale = jnp.where(scale == 0, jnp.float32(1), scale)
    q = jnp.round(g * (jnp.float32(1) / scale)[..., None, :])  # round half to even, like np.rint
    q = jnp.clip(q, QMIN, QMAX).astype(jnp.int8).astype(jnp.int4).reshape(w.shape)
    return q, scale


quantize_int4_jnp = jax.jit(_quantize_int4_jnp, static_argnames=("group",))
quantize_int4_jnp.__doc__ = (
    "jnp (jitted): f32 `[..., K, N]` -> (`q` jnp.int4 `[..., K, N]`, `scales` f32 `[..., K/G, N]`)."
)


def quantize_int4(w, group=128):
    """Dispatch on the input: numpy in -> (int8, f32) numpy; jax Array in -> (int4, f32) jax."""
    if isinstance(w, jax.Array):
        return quantize_int4_jnp(w, group=group)
    return quantize_int4_np(w, group)


def _flat_scales(q_shape, scales):
    """Accept flat `[..., K/G, N]` or chunked `[..., K/KC, KC/G, N]` scales; return flat."""
    if scales.ndim == len(q_shape) + 1:
        return scales.reshape(scales.shape[:-3] + (-1, scales.shape[-1]))
    return scales


def dequantize_int4_np(q, scales):
    """numpy: `q * scale` in f32 (exact), `[..., K, N]`; scales flat or chunked, any float dtype."""
    q = np.asarray(q)
    if q.dtype == np.dtype(jnp.int4):
        q = q.astype(np.int8)
    scales = _flat_scales(q.shape, np.asarray(scales)).astype(np.float32)
    s = np.repeat(scales, q.shape[-2] // scales.shape[-2], axis=-2)
    return q.astype(np.float32) * s


def dequantize_int4_jnp(q, scales):
    """jnp: `q * scale` in f32 (exact), `[..., K, N]`; scales flat or chunked, any float dtype."""
    scales = _flat_scales(q.shape, jnp.asarray(scales)).astype(jnp.float32)
    s = jnp.repeat(scales, q.shape[-2] // scales.shape[-2], axis=-2)
    return q.astype(jnp.float32) * s


def dequantize_int4(q, scales):
    """Dispatch on the input type; returns f32 `[..., K, N]` (numpy or jax)."""
    if isinstance(q, jax.Array) or isinstance(scales, jax.Array):
        return dequantize_int4_jnp(jnp.asarray(q), jnp.asarray(scales))
    return dequantize_int4_np(q, scales)


def scales_to_chunked(scales, K, kc=None):
    """Flat `[..., K/G, N]` -> kernel layout `[..., K/KC, KC/G, N]` f32 (`KC = k_chunk(K)`)."""
    kc = k_chunk(K) if kc is None else kc
    groups = scales.shape[-2]
    if groups % (K // kc):
        raise ValueError(f"{groups} groups cannot be split into {K // kc} chunks of K={K}")
    shape = scales.shape[:-2] + (K // kc, groups // (K // kc), scales.shape[-1])
    return scales.reshape(shape).astype(
        np.float32 if not isinstance(scales, jax.Array) else jnp.float32
    )


def scales_from_chunked(scales):
    """Kernel layout `[..., K/KC, KC/G, N]` -> flat `[..., K/G, N]` (dtype preserved)."""
    return scales.reshape(scales.shape[:-3] + (-1, scales.shape[-1]))


def pack_int4(q):
    """int8/int4 `[..., N]` (N even) -> uint8 `[..., N/2]`; element 2i in the low nibble."""
    q = np.asarray(q)
    if q.dtype == np.dtype(jnp.int4):
        q = q.astype(np.int8)
    if q.shape[-1] % 2:
        raise ValueError("pack_int4 needs an even last axis")
    lo = q[..., 0::2].astype(np.uint8) & 0xF
    hi = q[..., 1::2].astype(np.uint8) & 0xF
    return (lo | (hi << 4)).astype(np.uint8)


def unpack_int4(packed, dtype=np.int8):
    """uint8 `[..., N/2]` -> `[..., N]` of `dtype` (int8 by default, or `jnp.int4`)."""
    packed = np.asarray(packed, np.uint8)
    lo = (packed << 4).astype(np.int8) >> 4  # sign-extend the low nibble
    hi = packed.astype(np.int8) >> 4  # arithmetic shift sign-extends the high nibble
    out = np.stack((lo, hi), axis=-1).reshape(packed.shape[:-1] + (packed.shape[-1] * 2,))
    return out.astype(dtype)


def unpack_int4_jnp(packed):
    """On-device unpack: uint8 `[..., N/2]` -> jnp.int4 `[..., N]` (low nibble first).

    Uses int32 arithmetic (unsigned vector ops are unsafe on TPU7x Mosaic; plain XLA is fine).
    """
    p = jnp.asarray(packed).astype(jnp.int32)
    lo = ((p & 0xF) ^ 0x8) - 0x8  # sign-extend the low nibble
    hi = (((p >> 4) & 0xF) ^ 0x8) - 0x8
    out = jnp.stack((lo, hi), axis=-1).reshape(p.shape[:-1] + (p.shape[-1] * 2,))
    return out.astype(jnp.int8).astype(jnp.int4)
