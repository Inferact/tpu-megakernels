"""Interleaved complex RoPE (base and YaRN) for DeepSeek-V4.1-Flash.

Matches `REF/model.py:368-406` and `INVENTORY.md` §0, §2.2, §8.1:
- Applied to the last `rope_head_dim` (`64`) dimensions `x[..., -64:]`, pairing adjacent elements
  `(2i, 2i+1)` as complex numbers in `float32` and rounding back to `x.dtype` (`bfloat16`).
- `r == 0` layers (`L0, L1`, `mtp.0..2`): `theta = 10000.0`, `original_seq_len = 0` (no YaRN).
- `r > 0` layers (`L2..39`): `theta = 160000.0`, YaRN (`factor = 16.0, original_seq_len = 65536,
  beta_fast = 32, beta_slow = 1`, linear ramp dims `low = 15, high = 25`).
- `inverse=True` conjugates the rotation (`cos - i*sin`) for the attention output `o[..., -64:]`.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from deepseek_v41.config import DSV41Config


def _is_jax(x: Any) -> bool:
    return isinstance(x, jax.Array)


def corrected_dim(dim: int, rotations: float, original_seq_len: int, base: float) -> float:
    """Dimension whose wavelength completes `rotations` turns over `original_seq_len` (`REF/model.py:379-380`)."""
    return dim * math.log(original_seq_len / (rotations * 2.0 * math.pi)) / (2.0 * math.log(base))


def compute_rope_freqs_np(
    dim: int = 64,
    original_seq_len: int = 0,
    base: float = 10000.0,
    factor: float = 16.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
    dtype: Any = np.float32,
) -> tuple[np.ndarray, int | None, int | None]:
    """Computes the `dim // 2` inverse frequencies (`REF/model.py:376-386`).

    Returns `(freqs, low, high)` where `low, high` are `(15, 25)` when YaRN is active (`original_seq_len > 0`)
    and `(None, None)` when YaRN is disabled (`original_seq_len == 0`).
    """
    idx = np.arange(0, dim, 2, dtype=dtype)
    freqs = dtype(1.0) / (dtype(base) ** (idx / dtype(dim)))
    if original_seq_len > 0:
        low = max(math.floor(corrected_dim(dim, beta_fast, original_seq_len, base)), 0)
        high = min(math.ceil(corrected_dim(dim, beta_slow, original_seq_len, base)), dim - 1)
        denom = dtype(max(high - low, 1e-3))
        ramp = np.clip((np.arange(dim // 2, dtype=dtype) - dtype(low)) / denom, dtype(0.0), dtype(1.0))
        smooth = dtype(1.0) - ramp
        freqs = freqs / dtype(factor) * (dtype(1.0) - smooth) + freqs * smooth
        return freqs, low, high
    return freqs, None, None


def precompute_freqs_cis_np(
    dim: int,
    seqlen: int,
    original_seq_len: int = 0,
    base: float = 10000.0,
    factor: float = 16.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
    *,
    return_complex: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Precomputes RoPE `(cos, sin)` tables of shape `[seqlen, dim // 2]` in `float32` (`REF/model.py:369-389`)."""
    freqs, _, _ = compute_rope_freqs_np(
        dim=dim,
        original_seq_len=original_seq_len,
        base=base,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        dtype=np.float32,
    )
    positions = np.arange(seqlen, dtype=np.float32)
    angles = np.outer(positions, freqs).astype(np.float32)
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    if return_complex:
        return (cos + 1j * sin).astype(np.complex64)
    return cos, sin


def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int = 0,
    base: float = 10000.0,
    factor: float = 16.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
    *,
    return_complex: bool = False,
) -> tuple[jax.Array, jax.Array] | jax.Array:
    """Precomputes RoPE `(cos, sin)` tables as JAX `float32` arrays of shape `[seqlen, dim // 2]`."""
    out = precompute_freqs_cis_np(
        dim=dim,
        seqlen=seqlen,
        original_seq_len=original_seq_len,
        base=base,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        return_complex=return_complex,
    )
    if return_complex:
        return jnp.asarray(out, dtype=jnp.complex64)
    cos, sin = out
    return jnp.asarray(cos, dtype=jnp.float32), jnp.asarray(sin, dtype=jnp.float32)


def precompute_layer_rope_tables(
    cfg: DSV41Config,
    max_seq_len: int,
) -> dict[str, tuple[jax.Array, jax.Array]]:
    """Precomputes both `(cos, sin)` tables (`"swa"` for `r==0`, `"yarn"` for `r>0`) for `cfg`."""
    cos_swa, sin_swa = precompute_freqs_cis(
        dim=cfg.qk_rope_head_dim,
        seqlen=max_seq_len,
        original_seq_len=0,
        base=cfg.rope_theta,
        factor=cfg.rope_factor,
        beta_fast=cfg.beta_fast,
        beta_slow=cfg.beta_slow,
    )
    cos_yarn, sin_yarn = precompute_freqs_cis(
        dim=cfg.qk_rope_head_dim,
        seqlen=max_seq_len,
        original_seq_len=cfg.original_seq_len,
        base=cfg.compress_rope_theta,
        factor=cfg.rope_factor,
        beta_fast=cfg.beta_fast,
        beta_slow=cfg.beta_slow,
    )
    return {"swa": (cos_swa, sin_swa), "yarn": (cos_yarn, sin_yarn)}


def _unpack_cos_sin(
    freqs_or_cos: Any,
    sin: Any | None = None,
) -> tuple[Any, Any]:
    if sin is not None:
        return freqs_or_cos, sin
    if isinstance(freqs_or_cos, tuple) and len(freqs_or_cos) == 2:
        return freqs_or_cos[0], freqs_or_cos[1]
    if _is_jax(freqs_or_cos):
        arr = jnp.asarray(freqs_or_cos)
        if jnp.iscomplexobj(arr):
            return jnp.real(arr).astype(jnp.float32), jnp.imag(arr).astype(jnp.float32)
    else:
        arr = np.asarray(freqs_or_cos)
        if np.iscomplexobj(arr):
            return np.real(arr).astype(np.float32), np.imag(arr).astype(np.float32)
    raise ValueError("Expected `(cos, sin)` tuple, two `(cos, sin)` arguments, or complex `freqs_cis` tensor")


def _broadcast_cos_sin(x_even: Any, cos: Any, sin: Any, xp: Any) -> tuple[Any, Any]:
    """Broadcasts `cos, sin` (`[half_rd]` or `[S, half_rd]` or `[B, S, half_rd]`) against `x_even`."""
    c = xp.asarray(cos, dtype=xp.float32)
    s = xp.asarray(sin, dtype=xp.float32)
    if c.ndim == 1:
        # [half_rd] -> [1, ..., 1, half_rd]
        shape = (1,) * (x_even.ndim - 1) + (c.shape[-1],)
        return c.reshape(shape), s.reshape(shape)
    if c.ndim == 2:
        if x_even.ndim == 2:
            return c, s
        if x_even.ndim == 3:
            # x_even is [B, S, half_rd] or [S, H, half_rd]
            if x_even.shape[1] == c.shape[0]:
                return c[None, :, :], s[None, :, :]
            return c[:, None, :], s[:, None, :]
        if x_even.ndim == 4:
            # x_even is [B, S, H, half_rd]
            return c[None, :, None, :], s[None, :, None, :]
    if c.ndim == 3 and x_even.ndim == 4:
        # c is [B, S, half_rd], x_even is [B, S, H, half_rd]
        return c[:, :, None, :], s[:, :, None, :]
    return c, s


def apply_rotary_emb_np(
    x: np.ndarray,
    freqs_or_cos: Any,
    sin: np.ndarray | None = None,
    *,
    inverse: bool = False,
) -> np.ndarray:
    """Applies interleaved complex RoPE on the last `2 * (rd // 2)` dims of `x` in NumPy (`REF/model.py:392-406`)."""
    arr = np.asarray(x)
    orig_dtype = arr.dtype
    cos_arr, sin_arr = _unpack_cos_sin(freqs_or_cos, sin)
    half_rd = cos_arr.shape[-1]
    rd = 2 * half_rd
    if arr.shape[-1] < rd:
        raise ValueError(f"Input last dim {arr.shape[-1]} smaller than RoPE dim {rd}")
    tail = arr[..., -rd:].astype(np.float32)
    x_even = tail[..., 0::2]
    x_odd = tail[..., 1::2]
    c, s = _broadcast_cos_sin(x_even, cos_arr, sin_arr, np)
    if inverse:
        s = -s
    y_even = x_even * c - x_odd * s
    y_odd = x_even * s + x_odd * c
    rot_tail = np.stack([y_even, y_odd], axis=-1).reshape(tail.shape).astype(orig_dtype)
    if arr.shape[-1] == rd:
        return rot_tail
    return np.concatenate([arr[..., :-rd], rot_tail], axis=-1)


def _cpu_pad_barrier(x: Any) -> Any:
    return jax.lax.optimization_barrier(x) if jax.default_backend() == "cpu" else x


def apply_rotary_emb_jax(
    x: jax.Array,
    freqs_or_cos: Any,
    sin: jax.Array | None = None,
    *,
    inverse: bool = False,
) -> jax.Array:
    """Applies interleaved complex RoPE on the last `2 * (rd // 2)` dims of `x` in JAX (`REF/model.py:392-406`)."""
    arr = jnp.asarray(x)
    orig_dtype = arr.dtype
    cos_arr, sin_arr = _unpack_cos_sin(freqs_or_cos, sin)
    half_rd = cos_arr.shape[-1]
    rd = 2 * half_rd
    if arr.shape[-1] < rd:
        raise ValueError(f"Input last dim {arr.shape[-1]} smaller than RoPE dim {rd}")
    tail = arr[..., -rd:].astype(jnp.float32)
    x_even = tail[..., 0::2]
    x_odd = tail[..., 1::2]
    c, s = _broadcast_cos_sin(x_even, cos_arr, sin_arr, jnp)
    c = jnp.broadcast_to(c, x_even.shape)
    s = jnp.broadcast_to(s, x_even.shape)
    if x_even.ndim >= 4:
        # [B, S, H, half_rd] -> pad along token axis m = B * S so m=1 and m=6 both become pad_m=8
        h_dim = x_even.shape[-2]
        m = int(np.prod(x_even.shape[:-2]))
        pad_m = ((max(m, 8) + 7) // 8) * 8
        xe = x_even.reshape(m, h_dim, half_rd)
        xo = x_odd.reshape(m, h_dim, half_rd)
        cp = c.reshape(m, h_dim, half_rd)
        sp = s.reshape(m, h_dim, half_rd)
        if pad_m > m:
            pad_spec = ((0, pad_m - m), (0, 0), (0, 0))
            xe = jnp.pad(xe, pad_spec)
            xo = jnp.pad(xo, pad_spec)
            cp = jnp.pad(cp, pad_spec)
            sp = jnp.pad(sp, pad_spec)
        xe, xo, cp, sp = _cpu_pad_barrier((xe, xo, cp, sp))
        if inverse:
            sp = -sp
        ye = xe * cp - xo * sp
        yo = xe * sp + xo * cp
        rot_pad = _cpu_pad_barrier(
            jnp.stack([ye, yo], axis=-1).reshape(pad_m, h_dim, rd).astype(orig_dtype)
        )
        rot_tail = rot_pad[:m].reshape(tail.shape)
    else:
        m = int(np.prod(x_even.shape[:-1]))
        pad_m = ((max(m, 8) + 7) // 8) * 8
        xe = x_even.reshape(m, half_rd)
        xo = x_odd.reshape(m, half_rd)
        cp = c.reshape(m, half_rd)
        sp = s.reshape(m, half_rd)
        if pad_m > m:
            pad_spec = ((0, pad_m - m), (0, 0))
            xe = jnp.pad(xe, pad_spec)
            xo = jnp.pad(xo, pad_spec)
            cp = jnp.pad(cp, pad_spec)
            sp = jnp.pad(sp, pad_spec)
        xe, xo, cp, sp = _cpu_pad_barrier((xe, xo, cp, sp))
        if inverse:
            sp = -sp
        ye = xe * cp - xo * sp
        yo = xe * sp + xo * cp
        rot_pad = _cpu_pad_barrier(
            jnp.stack([ye, yo], axis=-1).reshape(pad_m, rd).astype(orig_dtype)
        )
        rot_tail = rot_pad[:m].reshape(tail.shape)
    if arr.shape[-1] == rd:
        return rot_tail
    return jnp.concatenate([arr[..., :-rd], rot_tail], axis=-1)


def apply_rotary_emb(
    x: Any,
    freqs_or_cos: Any,
    sin: Any | None = None,
    inverse: bool = False,
) -> Any:
    """Polymorphic (NumPy / JAX) interleaved complex RoPE on the last `rope_head_dim` dims."""
    if _is_jax(x):
        return apply_rotary_emb_jax(x, freqs_or_cos, sin, inverse=inverse)
    return apply_rotary_emb_np(x, freqs_or_cos, sin, inverse=inverse)
