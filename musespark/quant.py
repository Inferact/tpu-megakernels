"""Weight quantizers of the Muse Spark kernels: int4 g128 (routed experts, container v1), NVFP4
(experts, container v2) and int8 per-output-channel (dense projections + lm_head, `dense_format
int8`, see the int8 section below).

Symmetric int4 group quantization for the Muse Spark routed experts (design.md section 2).

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


# ---------------------------------------------------------------------------------------------
# int8 per-output-channel (the dense projections and the lm_head)
# ---------------------------------------------------------------------------------------------
#
# A `[..., K, N]` weight (K the contraction axis) gets ONE f32 scale per output column:
#
#     scale[n] = absmax_k |w[k, n]| * f32(1/127)        (all-zero columns: scale = 1)
#     q[k, n]  = clip(round_half_even(w[k, n] * (1/scale[n])), -127, 127)      int8
#
# so `dequant = q * scale` (exact in f32: q has <= 7 significant bits). The kernel never forms
# the dequantized weight: it computes `dot(x_bf16, q_int8) -> f32` on the MXU (int8 values are
# exact in bf16, so the products are the same as a bf16 x bf16 dot of the dequantized weight
# would give BEFORE its rounding) and multiplies the f32 column sums by `scale` afterwards.
# The reciprocal multiplies are spelled out (see the int4 note above) so the numpy and jnp
# quantizers are bit-identical.

INT8_MAX = 127
INV_INT8_MAX = np.float32(1.0 / INT8_MAX)


def quantize_int8_np(w):
    """numpy: f32/bf16 `[..., K, N]` -> (`q` int8 `[..., K, N]` in [-127, 127], `scale` f32
    `[..., 1, N]`), symmetric per output column (absmax / 127, round half to even)."""
    w = np.asarray(w)
    if w.dtype != np.float32:
        w = w.astype(np.float32)
    amax = np.max(np.abs(w), axis=-2, keepdims=True)  # [..., 1, N]
    scale = (amax * INV_INT8_MAX).astype(np.float32)
    scale = np.where(scale == 0, np.float32(1), scale).astype(np.float32)
    q = np.rint(w * (np.float32(1) / scale))
    q = np.clip(q, -INT8_MAX, INT8_MAX).astype(np.int8)
    return q, scale


def _quantize_int8_jnp(w):
    w = jnp.asarray(w, jnp.float32)
    amax = jnp.max(jnp.abs(w), axis=-2, keepdims=True)
    scale = amax * INV_INT8_MAX
    scale = jnp.where(scale == 0, jnp.float32(1), scale)
    q = jnp.round(w * (jnp.float32(1) / scale))  # round half to even, like np.rint
    q = jnp.clip(q, -INT8_MAX, INT8_MAX).astype(jnp.int8)
    return q, scale


quantize_int8_jnp = jax.jit(_quantize_int8_jnp)
quantize_int8_jnp.__doc__ = (
    "jnp (jitted): `[..., K, N]` -> (`q` int8 `[..., K, N]`, `scale` f32 `[..., 1, N]`); "
    "bit-identical to `quantize_int8_np`."
)


def quantize_int8(w):
    """Dispatch on the input: numpy in -> numpy out, jax Array in -> jax out."""
    if isinstance(w, jax.Array):
        return quantize_int8_jnp(w)
    return quantize_int8_np(w)


def dequantize_int8_np(q, scale):
    """numpy: `q * scale` in f32 (exact), `[..., K, N]`."""
    return np.asarray(q).astype(np.float32) * np.asarray(scale, np.float32)


def dequantize_int8_jnp(q, scale):
    """jnp: `q * scale` in f32 (exact), `[..., K, N]`."""
    return jnp.asarray(q).astype(jnp.float32) * jnp.asarray(scale, jnp.float32)


def dequantize_int8(q, scale):
    """Dispatch on the input type; returns f32 `[..., K, N]` (numpy or jax)."""
    if isinstance(q, jax.Array) or isinstance(scale, jax.Array):
        return dequantize_int8_jnp(q, scale)
    return dequantize_int8_np(q, scale)


def int8_dot(x, q, scale):
    """jnp: the kernel's int8 dense maths, `dot(bf16(x), bf16(q)) * scale` -> f32 `[..., N]`
    (exact int8 -> bf16 conversion; XLA/MXU f32 accumulation)."""
    y = jnp.dot(jnp.asarray(x).astype(jnp.bfloat16), jnp.asarray(q).astype(jnp.bfloat16),
                preferred_element_type=jnp.float32)
    return y * jnp.asarray(scale, jnp.float32)


# ---------------------------------------------------------------------------------------------
# NVFP4 (e2m1 values, block-16 e4m3 scales, per-tensor f32 global scale)
# ---------------------------------------------------------------------------------------------
#
# Vendor format (TensorRT-Model-Optimizer `NVFP4QTensor`, checkpoint
# `meta-models/Muse-Spark-1.2-816B-A42B-NVFP4-open`), for an nn.Linear weight `[N, K]`:
#
#     w[n, k] = e2m1(code[n, k]) * e4m3(weight_scale[n, k // 16]) * weight_scale_2   (f32)
#
# with `code` packed two per byte along K (low nibble = even k), `weight_scale_2 = amax / (6 * 448)`
# and `weight_scale = e4m3(amax_block / (6 * weight_scale_2))` clamped to [2^-9, 448].
#
# Kernel/container form of a `[K, N]` (`y = x @ w`) slab: the codes are packed EIGHT per int32
# along K, `packed[k', n]` holds rows `8k' .. 8k'+7`, row `8k'+j` in bits `4j .. 4j+3` (low nibble
# first), which is exactly how `pltpu.bitcast(int32 [K/8, N] -> float4_e2m1fn [K, N])` and
# `lax.bitcast_convert_type` unpack it (`unpack_fp4_rows`). Because the checkpoint also packs
# low-nibble-first along K, `nvfp4_rows_to_packed` (checkpoint `[N, K/2]` uint8 -> `[K/8, N]` int32)
# is a pure transpose of the bytes viewed as little-endian uint32 -- no nibble maths. Block
# scales stay e4m3 (`[K/16, N]`, or chunked `[K/KC, KC/16, N]` like the int4 scales) and the
# global scale is a separate f32; `e2m1 * e4m3` has <= 6 significant bits, so the product is exact
# in bf16/f32 and only the global multiply rounds.

FP4_BLOCK = 16
FP4_PER_WORD = 8  # e2m1 codes per int32 along K
E4M3 = np.dtype(ml_dtypes.float8_e4m3fn)
FP4 = np.dtype(ml_dtypes.float4_e2m1fn)
E2M1_MAX = np.float32(6.0)
E4M3_MAX = np.float32(448.0)
E4M3_MIN_SCALE = np.float32(2.0**-9)  # smallest e4m3 subnormal
E2M1_MAGNITUDES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)
E2M1_TABLE = np.concatenate([E2M1_MAGNITUDES, -E2M1_MAGNITUDES]).astype(np.float32)  # code -> value
E2M1_BOUNDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], np.float32)  # midpoints


def e2m1_values(codes):
    """uint8 e2m1 codes (`sign << 3 | ordinal`) -> f32 values (numpy)."""
    return E2M1_TABLE[np.asarray(codes, np.uint8) & 0xF]


def e2m1_values_jnp(codes):
    """jnp: int codes -> f32 values (16-entry table gather; `-0` for code 8)."""
    return jnp.asarray(E2M1_TABLE)[jnp.asarray(codes).astype(jnp.int32) & 0xF]


def e2m1_encode(v):
    """f32 (already divided by its scales) -> uint8 e2m1 codes, modelopt rounding: nearest, ties
    to the even ordinal (`0.25 -> 0`, `0.75 -> 1.0`, `1.25 -> 1.0`, `1.75 -> 2`, `2.5 -> 2`,
    `3.5 -> 4`, `5 -> 4`), magnitudes above 6 clip to 6."""
    v = np.asarray(v, np.float32)
    a = np.abs(v)
    ordinal = np.zeros(a.shape, np.uint8)
    for bound in E2M1_BOUNDS:  # = searchsorted(bounds, a, side="left")
        ordinal += a > bound
    for bound in E2M1_BOUNDS[1::2]:  # ties at the odd bounds round up (to the even ordinal)
        ordinal += a == bound
    return ((v < 0).astype(np.uint8) << 3) | ordinal


def unpack_nvfp4_codes(packed):
    """Checkpoint packing: uint8 `[..., K/2]` -> uint8 codes `[..., K]` (low nibble = even k)."""
    p = np.asarray(packed, np.uint8)
    out = np.stack((p & 0xF, p >> 4), axis=-1)
    return out.reshape(p.shape[:-1] + (p.shape[-1] * 2,))


def pack_fp4_rows(codes):
    """uint8 codes `[..., K, N]` (K % 8 == 0) -> int32 `[..., K/8, N]`, row `8k'+j` in nibble `j`."""
    c = np.asarray(codes, np.uint8)
    K = c.shape[-2]
    if K % FP4_PER_WORD:
        raise ValueError(f"K={K} is not a multiple of {FP4_PER_WORD}")
    g = c.reshape(c.shape[:-2] + (K // FP4_PER_WORD, FP4_PER_WORD, c.shape[-1])).astype(np.uint32)
    word = np.zeros(g.shape[:-2] + (g.shape[-1],), np.uint32)
    for j in range(FP4_PER_WORD):
        word |= (g[..., j, :] & 0xF) << np.uint32(4 * j)
    return word.view(np.int32)


def unpack_fp4_rows(packed):
    """int32 `[..., K/8, N]` -> uint8 codes `[..., K, N]` (inverse of `pack_fp4_rows`)."""
    p = np.asarray(packed).view(np.uint32)
    parts = [((p >> np.uint32(4 * j)) & 0xF).astype(np.uint8) for j in range(FP4_PER_WORD)]
    out = np.stack(parts, axis=-2)  # [..., K/8, 8, N]
    return out.reshape(p.shape[:-2] + (p.shape[-2] * FP4_PER_WORD, p.shape[-1]))


def nvfp4_rows_to_packed(rows):
    """Checkpoint `[..., N, K/2]` uint8 (nn.Linear rows, K packed low-nibble-first) -> kernel
    `[..., K/8, N]` int32: the four bytes of rows `8k'..8k'+7` ARE the little-endian word."""
    r = np.asarray(rows, np.uint8)
    if r.shape[-1] % 4:
        raise ValueError("K/2 must be a multiple of 4 bytes")
    words = np.ascontiguousarray(r).view(np.uint32)  # [..., N, K/8]
    return np.ascontiguousarray(np.swapaxes(words, -1, -2)).view(np.int32)


def fp4_scales_to_chunked(bs, K, kc=None):
    """Flat block scales `[..., K/16, N]` -> `[..., K/KC, KC/16, N]` (dtype preserved)."""
    kc = k_chunk(K) if kc is None else kc
    groups = bs.shape[-2]
    if groups * FP4_BLOCK != K or kc % FP4_BLOCK:
        raise ValueError(f"{groups} block scales do not match K={K} (chunk {kc})")
    return bs.reshape(bs.shape[:-2] + (K // kc, kc // FP4_BLOCK, bs.shape[-1]))


def _flat_fp4_scales(packed_shape, bs):
    """Accept flat `[..., K/16, N]` or chunked `[..., K/KC, KC/16, N]` block scales."""
    if bs.ndim == len(packed_shape) + 1:
        return bs.reshape(bs.shape[:-3] + (-1, bs.shape[-1]))
    return bs


def _e4m3_np(bs):
    bs = np.asarray(bs)
    return bs.view(E4M3) if bs.dtype == np.uint8 else bs


def dequant_fp4_np(packed, bs, gs=None):
    """numpy: `[..., K/8, N]` int32 codes, e4m3 block scales (flat or chunked; uint8 bits accepted)
    and an optional f32 global scale (broadcastable to `[..., K, N]`) -> f32 `[..., K, N]`
    `= e2m1 * e4m3 (exact) * gs`."""
    packed = np.asarray(packed)
    v = e2m1_values(unpack_fp4_rows(packed))
    s = _flat_fp4_scales(packed.shape, _e4m3_np(bs)).astype(np.float32)
    w = v * np.repeat(s, FP4_BLOCK, axis=-2)
    return w if gs is None else w * np.asarray(gs, np.float32)


def dequant_fp4_jnp(packed, bs, gs=None):
    """jnp: same as `dequant_fp4_np` (int32 packed codes, e4m3/uint8 block scales, f32 gs)."""
    packed = jnp.asarray(packed)
    if packed.dtype != jnp.int32:
        packed = lax.bitcast_convert_type(packed, jnp.int32)
    p = lax.bitcast_convert_type(packed, jnp.float4_e2m1fn)  # [..., K/8, N, 8], nibble j -> [..., j]
    v = jnp.swapaxes(p, -1, -2).astype(jnp.float32)  # [..., K/8, 8, N]
    v = v.reshape(packed.shape[:-2] + (packed.shape[-2] * FP4_PER_WORD, packed.shape[-1]))
    bs = jnp.asarray(bs)
    if bs.dtype == jnp.uint8:
        bs = lax.bitcast_convert_type(bs, jnp.float8_e4m3fn)
    s = _flat_fp4_scales(packed.shape, bs).astype(jnp.float32)
    w = v * jnp.repeat(s, FP4_BLOCK, axis=-2)
    return w if gs is None else w * jnp.asarray(gs, jnp.float32)


def dequant_fp4(packed, bs, gs=None):
    """Dispatch on the input type; returns f32 `[..., K, N]` (numpy or jax)."""
    if isinstance(packed, jax.Array) or isinstance(bs, jax.Array):
        return dequant_fp4_jnp(packed, bs, gs)
    return dequant_fp4_np(packed, bs, gs)


def nvfp4_global_scale(w):
    """modelopt `weight_scale_2 = amax / (6 * 448)` of an f32 array (over its trailing 2 axes)."""
    w = np.asarray(w, np.float32)
    amax = np.max(np.abs(w), axis=(-2, -1))
    return (amax / (E2M1_MAX * E4M3_MAX)).astype(np.float32)


def quantize_nvfp4_np(w, gs=None):
    """numpy modelopt NVFP4 quantization of f32 `[..., K, N]` (K the contraction axis, blocks
    of 16 consecutive k per column) -> `(packed int32 [..., K/8, N], bs e4m3 [..., K/16, N],
    gs f32 [...])`. `gs` (per leading index) defaults to `nvfp4_global_scale(w)`; block scales
    are `e4m3(amax_block / (6 * gs))` clamped to [2^-9, 448], codes `e2m1_encode(w / (bs * gs))`.
    An all-zero tensor gets gs = 1 (nothing to scale)."""
    w = np.asarray(w, np.float32)
    K = w.shape[-2]
    if K % FP4_BLOCK:
        raise ValueError(f"K={K} is not a multiple of the fp4 block {FP4_BLOCK}")
    gs = nvfp4_global_scale(w) if gs is None else np.asarray(gs, np.float32)
    gs = np.where(gs == 0, np.float32(1), gs).astype(np.float32)
    g = w.reshape(w.shape[:-2] + (K // FP4_BLOCK, FP4_BLOCK, w.shape[-1]))
    amax = np.max(np.abs(g), axis=-2)  # [..., K/16, N]
    bs = amax / (E2M1_MAX * gs[..., None, None])
    bs = np.clip(bs, E4M3_MIN_SCALE, E4M3_MAX).astype(E4M3)
    denom = bs.astype(np.float32) * gs[..., None, None]
    codes = e2m1_encode(g / denom[..., None, :]).reshape(w.shape)
    return pack_fp4_rows(codes), bs, gs
