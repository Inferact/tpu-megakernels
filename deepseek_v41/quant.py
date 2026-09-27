"""Reference-faithful quantization, dequantization, and TPU v7x MXFP4 bitcast helpers.

Implements the exact decode-time quantization contracts from `INVENTORY.md` §6 and
`REF/kernel.py` / `REF/convert.py` in both NumPy and JAX:
- `act_quant`: per-32 FP8 (`float8_e4m3fn`) with `ue8m0` power-of-2 scales (`amax >= 1e-4`,
  `s = 2^ceil(log2(amax * float32(1/448)))` computed via IEEE-754 exponent/mantissa bits).
- `fp4_act_quant`:
  - `block_size=32`, `scale_dtype="e8m0"`: `amax >= 6 * 2^-126`, `s = 2^ceil(log2(amax * float32(1/6)))`.
  - `block_size=16`, `scale_dtype="e4m3"`: `amax >= 6 * 2^-9`, `s = float32(e4m3(amax / 6))`.
  - Uses exact midpoint comparison (`0.25*s, 0.75*s, 1.25*s, 1.75*s, 2.5*s, 3.5*s, 5.0*s`)
    with ties-to-even so E2M1 codes never flip from float32 division drift.
- `dequant_fp8_block32x32` and `dequant_mxfp4_block32`.
- `pack_mxfp4_for_v7x_bitcast` and `unpack_mxfp4_v7x_bitcast`: packs 8 E2M1 nibbles per `uint32`
  with `codes[j::8]` into nibble `j` (`<< (4 * j)`) so that
  `pltpu.bitcast(u32, jnp.float4_e2m1fn).astype(jnp.float8_e4m3fn)` recovers `FP4_TABLE[codes]`
  bit-for-bit on TPU v7x and in Pallas interpret mode.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

FP8_MIN = np.float32(-448.0)
FP8_MAX = np.float32(448.0)
FP8_MAX_INV = np.float32(1.0 / 448.0)
ACT_QUANT_AMAX_FLOOR = np.float32(1e-4)

FP4_MIN = np.float32(-6.0)
FP4_MAX = np.float32(6.0)
FP4_MAX_INV = np.float32(1.0 / 6.0)
FP4_E8M0_AMAX_FLOOR = np.float32(6.0 * (2.0**-126))
FP4_E4M3_AMAX_FLOOR = np.float32(6.0 * (2.0**-9))

FP4_POS_VALUES_NP = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
FP4_TABLE_NP = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)
FP4_TABLE = FP4_TABLE_NP


def _fp4_table_jax() -> jax.Array:
    return jnp.asarray(FP4_TABLE_NP, dtype=jnp.float32)



def _is_jax(x: Any) -> bool:
    return isinstance(x, jax.Array)


def e8m0_to_f32_np(u8_or_e8m0: np.ndarray) -> np.ndarray:
    """Converts E8M0 biased-exponent bytes (`uint8` or `float8_e8m0fnu`) to `float32`."""
    arr = np.asarray(u8_or_e8m0)
    if arr.dtype == ml_dtypes.float8_e8m0fnu:
        return arr.astype(np.float32)
    if arr.dtype in (np.float32, np.float64, ml_dtypes.bfloat16, ml_dtypes.float8_e4m3fn):
        return arr.astype(np.float32)
    u8 = arr.view(np.uint8)
    bits = u8.astype(np.uint32) << np.uint32(23)
    return bits.view(np.float32)


def e8m0_to_f32_jax(u8_or_e8m0: jax.Array) -> jax.Array:
    """Converts E8M0 biased-exponent bytes (`uint8` or `float8_e8m0fnu`) to `float32` in JAX."""
    arr = jnp.asarray(u8_or_e8m0)
    if arr.dtype == jnp.float8_e8m0fnu:
        return arr.astype(jnp.float32)
    if arr.dtype in (jnp.float32, jnp.bfloat16, jnp.float8_e4m3fn):
        return arr.astype(jnp.float32)
    u8 = jax.lax.bitcast_convert_type(arr, jnp.uint8) if arr.dtype != jnp.uint8 else arr
    bits = u8.astype(jnp.uint32) << jnp.uint32(23)
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def e8m0_to_f32(u8_or_e8m0: Any) -> Any:
    """Converts E8M0 scales to `float32` (supports both NumPy and JAX arrays)."""
    return e8m0_to_f32_jax(u8_or_e8m0) if _is_jax(u8_or_e8m0) else e8m0_to_f32_np(u8_or_e8m0)


def f32_to_e8m0_u8_np(s_f32: np.ndarray) -> np.ndarray:
    """Extracts the IEEE-754 biased exponent byte (`uint8`) from a power-of-2 `float32` scale."""
    bits = np.asarray(s_f32, dtype=np.float32).view(np.uint32)
    return ((bits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.uint8)


def f32_to_e8m0_u8_jax(s_f32: jax.Array) -> jax.Array:
    """Extracts the IEEE-754 biased exponent byte (`uint8`) from a power-of-2 `float32` scale in JAX."""
    bits = jax.lax.bitcast_convert_type(jnp.asarray(s_f32, dtype=jnp.float32), jnp.uint32)
    return ((bits >> jnp.uint32(23)) & jnp.uint32(0xFF)).astype(jnp.uint8)


def fast_round_scale_np(amax: np.ndarray, max_inv: np.float32) -> tuple[np.ndarray, np.ndarray]:
    """Computes `2^ceil(log2(amax * max_inv))` via IEEE-754 bits (`REF/kernel.py:22-38`).

    Returns `(s_f32, s_u8)` where `s_u8` is the E8M0 biased exponent byte (`exp + 127`).
    """
    prod = np.asarray(amax, dtype=np.float32) * np.float32(max_inv)
    bits = prod.view(np.uint32)
    exp_x = ((bits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int32)
    man_bits = bits & np.uint32((1 << 23) - 1)
    log2_ceil = exp_x - np.int32(127) + np.where(man_bits != 0, np.int32(1), np.int32(0))
    biased = np.clip(log2_ceil + np.int32(127), 0, 254).astype(np.uint32)
    s_f32 = (biased << np.uint32(23)).view(np.float32)
    return s_f32, biased.astype(np.uint8)


def fast_round_scale_jax(amax: jax.Array, max_inv: float) -> tuple[jax.Array, jax.Array]:
    """Computes `2^ceil(log2(amax * max_inv))` via IEEE-754 bits in JAX (`REF/kernel.py:22-38`)."""
    prod = jnp.asarray(amax, dtype=jnp.float32) * jnp.float32(max_inv)
    bits = jax.lax.bitcast_convert_type(prod, jnp.uint32)
    exp_x = ((bits >> jnp.uint32(23)) & jnp.uint32(0xFF)).astype(jnp.int32)
    man_bits = bits & jnp.uint32((1 << 23) - 1)
    log2_ceil = exp_x - jnp.int32(127) + jnp.where(man_bits != 0, jnp.int32(1), jnp.int32(0))
    biased = jnp.clip(log2_ceil + jnp.int32(127), 0, 254).astype(jnp.uint32)
    s_f32 = jax.lax.bitcast_convert_type(biased << jnp.uint32(23), jnp.float32)
    return s_f32, biased.astype(jnp.uint8)


def act_quant_np(
    x: np.ndarray,
    block_size: int = 32,
    scale_fmt: str | None = "ue8m0",
    inplace: bool = False,
    return_u8_scale: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Block-wise FP8 (`float8_e4m3fn`) quantization in NumPy matching `REF/kernel.py:40-124`."""
    arr = np.asarray(x)
    orig_dtype = arr.dtype
    n = arr.shape[-1]
    if n % block_size != 0:
        raise ValueError(f"Last dimension {n} must be divisible by block_size={block_size}")
    x_f32 = arr.astype(np.float32).reshape(*arr.shape[:-1], n // block_size, block_size)
    amax = np.maximum(np.max(np.abs(x_f32), axis=-1), ACT_QUANT_AMAX_FLOOR)
    if scale_fmt is not None:
        s_f32, s_u8 = fast_round_scale_np(amax, FP8_MAX_INV)
    else:
        s_f32 = amax * FP8_MAX_INV
        s_u8 = f32_to_e8m0_u8_np(s_f32)
    scaled = np.clip(x_f32 / s_f32[..., None], FP8_MIN, FP8_MAX)
    q_e4m3 = scaled.astype(ml_dtypes.float8_e4m3fn)
    if inplace:
        deq = (q_e4m3.astype(np.float32) * s_f32[..., None]).reshape(arr.shape)
        out = deq.astype(orig_dtype)
        if isinstance(x, np.ndarray) and x.dtype == out.dtype and x.shape == out.shape:
            np.copyto(x, out)
            return x
        return out
    q_out = q_e4m3.reshape(arr.shape)
    return (q_out, s_u8) if return_u8_scale else (q_out, s_f32)


def _cpu_pad_barrier(x: Any) -> Any:
    return jax.lax.optimization_barrier(x) if jax.default_backend() == "cpu" else x


def act_quant_jax(
    x: jax.Array,
    block_size: int = 32,
    scale_fmt: str | None = "ue8m0",
    inplace: bool = False,
    return_u8_scale: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Block-wise FP8 (`float8_e4m3fn`) quantization in JAX matching `REF/kernel.py:40-124`."""
    arr = jnp.asarray(x)
    orig_dtype = arr.dtype
    orig_shape = arr.shape
    n = orig_shape[-1]
    if n % block_size != 0:
        raise ValueError(f"Last dimension {n} must be divisible by block_size={block_size}")
    x2d = arr.astype(jnp.float32).reshape(-1, n)
    m = x2d.shape[0]
    pad_m = ((max(m, 8) + 7) // 8) * 8
    if pad_m > m:
        x2d = jnp.pad(x2d, ((0, pad_m - m), (0, 0)))
    x2d = _cpu_pad_barrier(x2d)
    x_f32 = x2d.reshape(pad_m, n // block_size, block_size)
    amax = jnp.maximum(jnp.max(jnp.abs(x_f32), axis=-1), jnp.float32(ACT_QUANT_AMAX_FLOOR))
    if scale_fmt is not None:
        prod = jnp.asarray(amax, dtype=jnp.float32) * jnp.float32(FP8_MAX_INV)
        bits = jax.lax.bitcast_convert_type(prod, jnp.uint32)
        exp_x = ((bits >> jnp.uint32(23)) & jnp.uint32(0xFF)).astype(jnp.int32)
        man_bits = bits & jnp.uint32((1 << 23) - 1)
        log2_ceil = exp_x - jnp.int32(127) + jnp.where(man_bits != 0, jnp.int32(1), jnp.int32(0))
        biased = jnp.clip(log2_ceil + jnp.int32(127), 0, 254).astype(jnp.uint32)
        s_f32 = jax.lax.bitcast_convert_type(biased << jnp.uint32(23), jnp.float32)
    else:
        s_f32 = amax * jnp.float32(FP8_MAX_INV)
    scaled = jnp.clip(x_f32 / s_f32[..., None], jnp.float32(FP8_MIN), jnp.float32(FP8_MAX))
    q_e4m3 = scaled.astype(jnp.float8_e4m3fn)
    if inplace:
        deq = _cpu_pad_barrier((q_e4m3.astype(jnp.float32) * s_f32[..., None]).reshape(pad_m, n).astype(orig_dtype))
        return deq[:m].reshape(orig_shape)
    q_e4m3, s_f32 = _cpu_pad_barrier((q_e4m3, s_f32))
    q_out = q_e4m3.reshape(pad_m, n)[:m].reshape(orig_shape)
    if return_u8_scale:
        s_u8 = f32_to_e8m0_u8_jax(s_f32)
        s_u8 = _cpu_pad_barrier(s_u8)
        s_u8_out = s_u8[:m].reshape(*orig_shape[:-1], n // block_size)
        return q_out, s_u8_out
    s_f32_out = s_f32[:m].reshape(*orig_shape[:-1], n // block_size)
    return q_out, s_f32_out


def act_quant(
    x: Any,
    block_size: int = 32,
    scale_fmt: str | None = "ue8m0",
    inplace: bool = False,
    return_u8_scale: bool = False,
) -> Any:
    """Polymorphic (NumPy / JAX) reference-faithful `act_quant`."""
    if _is_jax(x):
        return act_quant_jax(
            x,
            block_size=block_size,
            scale_fmt=scale_fmt,
            inplace=inplace,
            return_u8_scale=return_u8_scale,
        )
    return act_quant_np(
        x,
        block_size=block_size,
        scale_fmt=scale_fmt,
        inplace=inplace,
        return_u8_scale=return_u8_scale,
    )


def _is_e4m3_scale_dtype(scale_dtype: Any) -> bool:
    if isinstance(scale_dtype, str):
        return "e4m3" in scale_dtype.lower() or scale_dtype.lower() == "fp8"
    return scale_dtype in (ml_dtypes.float8_e4m3fn, jnp.float8_e4m3fn)


def _quantize_e2m1_midpoint_np(x_f32: np.ndarray, s_f32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantizes `x_f32` (`[..., B]`) with scale `s_f32` (`[...]`) to E2M1 via exact midpoints.

    Returns `(q_f32, codes_u8)` where `q_f32` is in `{-6..-0, 0..6}` and `codes_u8` is in `0..15`.
    """
    ax = np.abs(x_f32)
    s = s_f32[..., None]
    mag_code = (
        (ax > np.float32(0.25) * s).astype(np.uint8)
        + (ax >= np.float32(0.75) * s).astype(np.uint8)
        + (ax > np.float32(1.25) * s).astype(np.uint8)
        + (ax >= np.float32(1.75) * s).astype(np.uint8)
        + (ax > np.float32(2.5) * s).astype(np.uint8)
        + (ax >= np.float32(3.5) * s).astype(np.uint8)
        + (ax > np.float32(5.0) * s).astype(np.uint8)
    )
    neg = np.signbit(x_f32)
    codes_u8 = mag_code | (neg.astype(np.uint8) << np.uint8(3))
    q_f32 = FP4_POS_VALUES_NP[mag_code] * np.where(neg, np.float32(-1.0), np.float32(1.0))
    return q_f32, codes_u8


def _quantize_e2m1_midpoint_jax(x_f32: jax.Array, s_f32: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Quantizes `x_f32` (`[..., B]`) with scale `s_f32` (`[...]`) to E2M1 via exact midpoints in JAX."""
    ax = jnp.abs(x_f32)
    s = s_f32[..., None]
    c1 = (ax > jnp.float32(0.25) * s).astype(jnp.int32)
    c2 = (ax >= jnp.float32(0.75) * s).astype(jnp.int32)
    c3 = (ax > jnp.float32(1.25) * s).astype(jnp.int32)
    c4 = (ax >= jnp.float32(1.75) * s).astype(jnp.int32)
    c5 = (ax > jnp.float32(2.5) * s).astype(jnp.int32)
    c6 = (ax >= jnp.float32(3.5) * s).astype(jnp.int32)
    c7 = (ax > jnp.float32(5.0) * s).astype(jnp.int32)
    mag_code = c1 + c2 + c3 + c4 + c5 + c6 + c7
    neg = jnp.signbit(x_f32)
    codes_i32 = mag_code | (neg.astype(jnp.int32) << jnp.int32(3))
    pos_val = (
        jnp.float32(0.5) * (c1 + c2 + c3 + c4).astype(jnp.float32)
        + (c5 + c6).astype(jnp.float32)
        + jnp.float32(2.0) * c7.astype(jnp.float32)
    )
    q_f32 = pos_val * jnp.where(neg, jnp.float32(-1.0), jnp.float32(1.0))
    return q_f32, codes_i32


def fp4_act_quant_np(
    x: np.ndarray,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: Any = "e8m0",
    return_codes: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Block-wise FP4 (`float4_e2m1fn`) quantization in NumPy matching `REF/kernel.py:128-205`."""
    arr = np.asarray(x)
    orig_dtype = arr.dtype
    n = arr.shape[-1]
    if n % block_size != 0:
        raise ValueError(f"Last dimension {n} must be divisible by block_size={block_size}")
    x_f32 = arr.astype(np.float32).reshape(*arr.shape[:-1], n // block_size, block_size)
    amax = np.max(np.abs(x_f32), axis=-1)
    if _is_e4m3_scale_dtype(scale_dtype):
        amax = np.maximum(amax, FP4_E4M3_AMAX_FLOOR)
        s_scale = (amax / FP4_MAX).astype(ml_dtypes.float8_e4m3fn)
        s_f32 = s_scale.astype(np.float32)
    else:
        amax = np.maximum(amax, FP4_E8M0_AMAX_FLOOR)
        s_f32, s_u8 = fast_round_scale_np(amax, FP4_MAX_INV)
        s_scale = s_u8 if scale_dtype is np.uint8 else s_f32
    q_f32, codes_u8 = _quantize_e2m1_midpoint_np(x_f32, s_f32)
    if inplace:
        deq = (q_f32 * s_f32[..., None]).reshape(arr.shape).astype(orig_dtype)
        if isinstance(x, np.ndarray) and x.dtype == deq.dtype and x.shape == deq.shape:
            np.copyto(x, deq)
            return x
        return deq
    if return_codes:
        return codes_u8.reshape(arr.shape), s_scale
    return q_f32.reshape(arr.shape).astype(ml_dtypes.float4_e2m1fn), s_scale


def fp4_act_quant_jax(
    x: jax.Array,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: Any = "e8m0",
    return_codes: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Block-wise FP4 (`float4_e2m1fn`) quantization in JAX matching `REF/kernel.py:128-205`."""
    arr = jnp.asarray(x)
    orig_dtype = arr.dtype
    orig_shape = arr.shape
    n = orig_shape[-1]
    if n % block_size != 0:
        raise ValueError(f"Last dimension {n} must be divisible by block_size={block_size}")
    x2d = arr.astype(jnp.float32).reshape(-1, n)
    m = x2d.shape[0]
    pad_m = ((max(m, 8) + 7) // 8) * 8
    if pad_m > m:
        x2d = jnp.pad(x2d, ((0, pad_m - m), (0, 0)))
    x2d = _cpu_pad_barrier(x2d)
    x_f32 = x2d.reshape(pad_m, n // block_size, block_size)
    amax = jnp.max(jnp.abs(x_f32), axis=-1)
    if _is_e4m3_scale_dtype(scale_dtype):
        amax = jnp.maximum(amax, jnp.float32(FP4_E4M3_AMAX_FLOOR))
        s_scale = (amax / jnp.float32(FP4_MAX)).astype(jnp.float8_e4m3fn)
        s_f32 = s_scale.astype(jnp.float32)
    else:
        amax = jnp.maximum(amax, jnp.float32(FP4_E8M0_AMAX_FLOOR))
        prod = jnp.asarray(amax, dtype=jnp.float32) * jnp.float32(FP4_MAX_INV)
        bits = jax.lax.bitcast_convert_type(prod, jnp.uint32)
        exp_x = ((bits >> jnp.uint32(23)) & jnp.uint32(0xFF)).astype(jnp.int32)
        man_bits = bits & jnp.uint32((1 << 23) - 1)
        log2_ceil = exp_x - jnp.int32(127) + jnp.where(man_bits != 0, jnp.int32(1), jnp.int32(0))
        biased = jnp.clip(log2_ceil + jnp.int32(127), 0, 254).astype(jnp.uint32)
        s_f32 = jax.lax.bitcast_convert_type(biased << jnp.uint32(23), jnp.float32)
        s_scale = biased.astype(jnp.uint8) if scale_dtype is jnp.uint8 else s_f32
    q_f32, codes_i32 = _quantize_e2m1_midpoint_jax(x_f32, s_f32)
    if inplace:
        deq = _cpu_pad_barrier((q_f32 * s_f32[..., None]).reshape(pad_m, n).astype(orig_dtype))
        return deq[:m].reshape(orig_shape)
    q_f32, s_scale = _cpu_pad_barrier((q_f32, s_scale))
    s_scale_out = s_scale[:m].reshape(*orig_shape[:-1], n // block_size)
    if return_codes:
        codes_u8 = _cpu_pad_barrier(codes_i32.astype(jnp.uint8))
        return codes_u8.reshape(pad_m, n)[:m].reshape(orig_shape), s_scale_out
    return q_f32.reshape(pad_m, n)[:m].reshape(orig_shape).astype(jnp.float4_e2m1fn), s_scale_out


def fp4_act_quant(
    x: Any,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: Any = "e8m0",
    return_codes: bool = False,
) -> Any:
    """Polymorphic (NumPy / JAX) reference-faithful `fp4_act_quant`."""
    if _is_jax(x):
        return fp4_act_quant_jax(
            x,
            block_size=block_size,
            inplace=inplace,
            scale_dtype=scale_dtype,
            return_codes=return_codes,
        )
    return fp4_act_quant_np(
        x,
        block_size=block_size,
        inplace=inplace,
        scale_dtype=scale_dtype,
        return_codes=return_codes,
    )


def dequant_fp8_block32x32_np(
    weight: np.ndarray,
    scale: np.ndarray,
    block_size: int = 32,
    out_dtype: Any = np.float32,
) -> np.ndarray:
    """Dequantizes a 2D FP8 (`float8_e4m3fn` or `uint8` bits) tensor with `[32, 32]` (or `[1, 32]`) E8M0 scales."""
    w = np.asarray(weight)
    if w.dtype == np.uint8 or w.dtype == np.int8:
        w = w.view(ml_dtypes.float8_e4m3fn)
    w_f32 = w.astype(np.float32)
    s_f32 = e8m0_to_f32_np(scale)
    m, k = w_f32.shape[-2], w_f32.shape[-1]
    sm, sk = s_f32.shape[-2], s_f32.shape[-1]
    rep_m = 1 if sm == m else block_size
    rep_k = 1 if sk == k else block_size
    s_exp = np.repeat(np.repeat(s_f32, rep_m, axis=-2)[..., :m, :], rep_k, axis=-1)[..., :, :k]
    return (w_f32 * s_exp).astype(out_dtype)


def dequant_fp8_block32x32_jax(
    weight: jax.Array,
    scale: jax.Array,
    block_size: int = 32,
    out_dtype: Any = jnp.bfloat16,
) -> jax.Array:
    """Dequantizes a 2D FP8 (`float8_e4m3fn` or `uint8` bits) tensor with `[32, 32]` (or `[1, 32]`) E8M0 scales in JAX."""
    w = jnp.asarray(weight)
    if w.dtype in (jnp.uint8, jnp.int8):
        w = jax.lax.bitcast_convert_type(w, jnp.float8_e4m3fn)
    w_f32 = w.astype(jnp.float32)
    s_f32 = e8m0_to_f32_jax(scale)
    m, k = w_f32.shape[-2], w_f32.shape[-1]
    sm, sk = s_f32.shape[-2], s_f32.shape[-1]
    rep_m = 1 if sm == m else block_size
    rep_k = 1 if sk == k else block_size
    s_exp = jnp.repeat(jnp.repeat(s_f32, rep_m, axis=-2)[..., :m, :], rep_k, axis=-1)[..., :, :k]
    return (w_f32 * s_exp).astype(out_dtype)


def dequant_fp8_block32x32(
    weight: Any,
    scale: Any,
    block_size: int = 32,
    out_dtype: Any = None,
) -> Any:
    """Polymorphic (NumPy / JAX) `dequant_fp8_block32x32`."""
    if _is_jax(weight) or _is_jax(scale):
        return dequant_fp8_block32x32_jax(
            weight,
            scale,
            block_size=block_size,
            out_dtype=jnp.bfloat16 if out_dtype is None else out_dtype,
        )
    return dequant_fp8_block32x32_np(
        weight,
        scale,
        block_size=block_size,
        out_dtype=np.float32 if out_dtype is None else out_dtype,
    )


def unpack_checkpoint_mxfp4_codes_np(packed_i8: np.ndarray) -> np.ndarray:
    """Unpacks checkpoint `int8`/`uint8` `[..., K // 2]` (low nibble = even K, high nibble = odd K) to `[..., K]` `uint8` codes `0..15`."""
    u8 = np.asarray(packed_i8).view(np.uint8)
    low = u8 & np.uint8(0x0F)
    high = (u8 >> np.uint8(4)) & np.uint8(0x0F)
    return np.stack([low, high], axis=-1).reshape(*u8.shape[:-1], u8.shape[-1] * 2)


def dequant_mxfp4_block32_np(
    packed_or_codes: np.ndarray,
    scale: np.ndarray,
    block_size: int = 32,
    out_dtype: Any = np.float32,
) -> np.ndarray:
    """Dequantizes MXFP4 weights (`[N, K // 2]` packed bytes or `[N, K]` codes) with `[N, K // 32]` E8M0 scales."""
    arr = np.asarray(packed_or_codes)
    s_f32 = e8m0_to_f32_np(scale)
    expected_k = s_f32.shape[-1] * block_size
    if arr.shape[-1] == expected_k // 2:
        codes = unpack_checkpoint_mxfp4_codes_np(arr)
    elif arr.shape[-1] == expected_k:
        codes = arr.astype(np.uint8) & np.uint8(0x0F)
    else:
        raise ValueError(
            f"Packed/code shape {arr.shape} incompatible with scale shape {s_f32.shape} and block_size={block_size}"
        )
    values = FP4_TABLE_NP[codes]
    values = values.reshape(*codes.shape[:-1], s_f32.shape[-1], block_size) * s_f32[..., None]
    return values.reshape(codes.shape).astype(out_dtype)


def dequant_mxfp4_block32_jax(
    packed_or_codes: jax.Array,
    scale: jax.Array,
    block_size: int = 32,
    out_dtype: Any = jnp.bfloat16,
) -> jax.Array:
    """Dequantizes MXFP4 weights (`[N, K // 2]` packed bytes or `[N, K]` codes) with `[N, K // 32]` E8M0 scales in JAX."""
    arr = jnp.asarray(packed_or_codes)
    s_f32 = e8m0_to_f32_jax(scale)
    expected_k = s_f32.shape[-1] * block_size
    if arr.shape[-1] == expected_k // 2:
        u8 = jax.lax.bitcast_convert_type(arr, jnp.uint8) if arr.dtype != jnp.uint8 else arr
        low = u8 & jnp.uint8(0x0F)
        high = (u8 >> jnp.uint8(4)) & jnp.uint8(0x0F)
        codes = jnp.stack([low, high], axis=-1).reshape(*u8.shape[:-1], u8.shape[-1] * 2)
    elif arr.shape[-1] == expected_k:
        codes = arr.astype(jnp.uint8) & jnp.uint8(0x0F)
    else:
        raise ValueError(
            f"Packed/code shape {arr.shape} incompatible with scale shape {s_f32.shape} and block_size={block_size}"
        )
    values = _fp4_table_jax()[codes.astype(jnp.int32)]
    values = values.reshape(*codes.shape[:-1], s_f32.shape[-1], block_size) * s_f32[..., None]
    return values.reshape(codes.shape).astype(out_dtype)


def dequant_mxfp4_block32(
    packed_or_codes: Any,
    scale: Any,
    block_size: int = 32,
    out_dtype: Any = None,
) -> Any:
    """Polymorphic (NumPy / JAX) `dequant_mxfp4_block32`."""
    if _is_jax(packed_or_codes) or _is_jax(scale):
        return dequant_mxfp4_block32_jax(
            packed_or_codes,
            scale,
            block_size=block_size,
            out_dtype=jnp.bfloat16 if out_dtype is None else out_dtype,
        )
    return dequant_mxfp4_block32_np(
        packed_or_codes,
        scale,
        block_size=block_size,
        out_dtype=np.float32 if out_dtype is None else out_dtype,
    )


def pack_mxfp4_for_v7x_bitcast(codes: np.ndarray | jax.Array, *, axis: int = 0) -> np.ndarray:
    """Packs 4-bit E2M1 codes `0..15` into `uint32` for `pltpu.bitcast(..., jnp.float4_e2m1fn)`.

    Packs 8 nibbles per `uint32` along `axis` (default `0`, i.e. contraction `K` in `[K, N]`):
    `packed |= codes[j::8] << (4 * j)` for `j in range(8)`.

    When unpacked inside Pallas via `pltpu.bitcast(u32, jnp.float4_e2m1fn).astype(jnp.float8_e4m3fn)`,
    this recovers `FP4_TABLE[codes]` of shape `[K, N]` bit-for-bit (`probe_v7x.py:213-240`).
    """
    c = np.asarray(codes, dtype=np.uint8) & np.uint8(0x0F)
    axis = axis % c.ndim
    k = c.shape[axis]
    if k % 8 != 0:
        raise ValueError(f"Dimension {k} along axis={axis} must be divisible by 8")
    out_shape = list(c.shape)
    out_shape[axis] = k // 8
    packed = np.zeros(out_shape, dtype=np.uint32)
    for j in range(8):
        slc = [slice(None)] * c.ndim
        slc[axis] = slice(j, None, 8)
        packed |= c[tuple(slc)].astype(np.uint32) << np.uint32(4 * j)
    return packed


def pack_checkpoint_mxfp4_for_v7x_bitcast(weight_i8: np.ndarray, *, transpose: bool = True) -> np.ndarray:
    """Unpacks checkpoint `[N, K // 2]` `int8` MXFP4 weights and packs them into `[K // 8, N]` `uint32`."""
    if transpose:
        contig = np.ascontiguousarray(weight_i8)
        if contig.shape[-1] % 4 == 0:
            return np.ascontiguousarray(np.swapaxes(contig.view(np.uint32), -2, -1))
    codes_nk = unpack_checkpoint_mxfp4_codes_np(weight_i8)
    codes = np.swapaxes(codes_nk, -2, -1) if transpose else codes_nk
    return pack_mxfp4_for_v7x_bitcast(codes, axis=-2 if codes.ndim >= 2 else 0)


def unpack_mxfp4_v7x_bitcast(
    packed_u32: Any,
    scales: Any | None = None,
    *,
    axis: int = 0,
    out_dtype: Any = None,
    return_codes: bool = False,
) -> Any:
    """Unpacks `uint32` packed via `pack_mxfp4_for_v7x_bitcast` back to E2M1 values (or 4-bit codes).

    Matches `pltpu.bitcast(packed_u32, jnp.float4_e2m1fn).astype(jnp.float8_e4m3fn)` bit-for-bit.
    If `scales` (`[K // 32, N]` E8M0 or `float32`) is provided, multiplies each 32-element group along
    `axis` by its scale.
    """
    if _is_jax(packed_u32) or (scales is not None and _is_jax(scales)):
        p = jnp.asarray(packed_u32, dtype=jnp.uint32)
        axis = axis % p.ndim
        # Expand nibble j into position 8*r + j along `axis`.
        shifts = jnp.arange(8, dtype=jnp.uint32) * jnp.uint32(4)
        shift_shape = [1] * (p.ndim + 1)
        shift_shape[axis + 1] = 8
        codes = (jnp.expand_dims(p, axis=axis + 1) >> shifts.reshape(shift_shape)) & jnp.uint32(0x0F)
        new_shape = list(p.shape)
        new_shape[axis] = p.shape[axis] * 8
        codes = codes.reshape(new_shape).astype(jnp.uint8)
        if return_codes:
            return codes
        values = _fp4_table_jax()[codes.astype(jnp.int32)]
        if scales is not None:
            s_f32 = e8m0_to_f32_jax(scales)
            s_exp = jnp.repeat(s_f32, 32, axis=axis)
            values = values * s_exp
            target_dtype = jnp.bfloat16 if out_dtype is None else out_dtype
            return values.astype(target_dtype)
        target_dtype = jnp.float8_e4m3fn if out_dtype is None else out_dtype
        return values.astype(target_dtype)

    p = np.asarray(packed_u32, dtype=np.uint32)
    axis = axis % p.ndim
    new_shape = list(p.shape)
    new_shape[axis] = p.shape[axis] * 8
    codes = np.empty(new_shape, dtype=np.uint8)
    for j in range(8):
        slc = [slice(None)] * p.ndim
        slc[axis] = slice(j, None, 8)
        codes[tuple(slc)] = ((p >> np.uint32(4 * j)) & np.uint32(0x0F)).astype(np.uint8)
    if return_codes:
        return codes
    values = FP4_TABLE_NP[codes]
    if scales is not None:
        s_f32 = e8m0_to_f32_np(scales)
        s_exp = np.repeat(s_f32, 32, axis=axis)
        values = values * s_exp
        target_dtype = np.float32 if out_dtype is None else out_dtype
        return values.astype(target_dtype)
    target_dtype = ml_dtypes.float8_e4m3fn if out_dtype is None else out_dtype
    return values.astype(target_dtype)


def pallas_bitcast_mxfp4_to_fp8(packed_u32: jax.Array, *, interpret: bool = True) -> jax.Array:
    """Runs `pltpu.bitcast(u32, jnp.float4_e2m1fn).astype(jnp.float8_e4m3fn)` inside `pl.pallas_call`."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    p = jnp.asarray(packed_u32, dtype=jnp.uint32)
    if p.ndim != 2:
        raise ValueError(f"Expected 2D [K // 8, N] uint32 array, got shape {p.shape}")
    k8, n = p.shape

    def _kernel(p_ref, o_ref):
        o_ref[...] = pltpu.bitcast(p_ref[...], jnp.float4_e2m1fn).astype(jnp.float8_e4m3fn)

    interp_params = pltpu.InterpretParams() if interpret else False
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((k8 * 8, n), jnp.float8_e4m3fn),
        interpret=interp_params,
    )(p)
