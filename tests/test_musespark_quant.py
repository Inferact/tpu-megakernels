"""CPU tests for the int4 group quantization used by the Muse Spark experts."""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from musespark import quant


def _weights(shape, seed, dead_group=None):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(shape, np.float32) * rng.choice([1e-3, 1.0, 30.0], shape)
    if dead_group is not None:
        w[..., dead_group[0] : dead_group[1], 0] = 0
    return w.astype(np.float32)


@pytest.mark.parametrize(
    "shape,group", [((512, 256), 128), ((3, 256, 128), 64), ((4096, 512), 128)]
)
def test_round_trip_and_error_bound(shape, group):
    w = _weights(shape, 1, dead_group=(0, group))
    q, s = quant.quantize_int4(w, group)
    assert q.dtype == np.int8 and q.shape == w.shape
    assert s.dtype == np.float32 and s.shape == shape[:-2] + (shape[-2] // group, shape[-1])
    assert q.min() >= -8 and q.max() <= 7
    assert np.array_equal(s, s.astype(ml_dtypes.bfloat16).astype(np.float32))  # bf16 values
    assert (s[..., 0, 0] == 1).all()  # zero group -> scale 1, q == 0
    assert not q[..., :group, 0].any()
    d = quant.dequantize_int4(q, s)
    bound = np.repeat(s, group, axis=-2) / 2
    assert (np.abs(w - d) <= bound * (1 + 2.0**-20)).all()
    # every group uses the full range on its abs-max element
    groups = q.reshape(shape[:-2] + (shape[-2] // group, group, shape[-1]))
    assert (np.abs(groups).max(axis=-2)[..., 1:] == 7).all()
    # dequant is exactly q * scale (products of a 4-bit int and a bf16 are exact in f32)
    assert np.array_equal(d, q.astype(np.float32) * np.repeat(s, group, axis=-2))


@pytest.mark.parametrize("shape,group", [((4096, 8192), 128), ((16, 512, 1024), 64)])
def test_numpy_equals_jnp(shape, group):
    w = _weights(shape, 2, dead_group=(group, 2 * group))
    q, s = quant.quantize_int4(w, group)
    qj, sj = quant.quantize_int4(jnp.asarray(w), group=group)
    assert isinstance(qj, jax.Array) and qj.dtype == jnp.int4 and sj.dtype == jnp.float32
    assert np.array_equal(np.asarray(qj).astype(np.int8), q)
    assert np.array_equal(np.asarray(sj), s)
    dj = quant.dequantize_int4(qj, sj)
    assert isinstance(dj, jax.Array) and dj.dtype == jnp.float32
    assert np.array_equal(np.asarray(dj), quant.dequantize_int4(q, s))
    # int4-typed numpy input dequantizes too
    assert np.array_equal(quant.dequantize_int4(q.astype(jnp.int4), s), quant.dequantize_int4(q, s))


def test_quantize_is_fast_enough():
    import time

    w = _weights((4096, 8192), 3)
    quant.quantize_int4(w)  # warm the page cache / allocator
    t = time.perf_counter()
    quant.quantize_int4(w)
    assert time.perf_counter() - t < 1.0


def test_pack_unpack_inverse():
    q = np.arange(-8, 8, dtype=np.int8)
    q = np.stack([q, q[::-1], np.zeros(16, np.int8), np.full(16, -8, np.int8)])  # [4, 16]
    p = quant.pack_int4(q)
    assert p.dtype == np.uint8 and p.shape == (4, 8)
    assert p[0, 0] == (0x8 | (0x9 << 4))  # -8 -> 0x8 low nibble, -7 -> 0x9 high nibble
    assert np.array_equal(quant.unpack_int4(p), q)
    assert quant.unpack_int4(p, jnp.int4).dtype == np.dtype(jnp.int4)
    assert np.array_equal(quant.unpack_int4(p, jnp.int4).astype(np.int8), q)
    assert np.array_equal(quant.pack_int4(q.astype(jnp.int4)), p)
    uj = quant.unpack_int4_jnp(jnp.asarray(p))
    assert uj.dtype == jnp.int4 and np.array_equal(np.asarray(uj).astype(np.int8), q)
    big, _ = quant.quantize_int4(_weights((2, 256, 512), 4), 128)
    assert np.array_equal(quant.unpack_int4(quant.pack_int4(big)), big)
    with pytest.raises(ValueError):
        quant.pack_int4(np.zeros((3, 5), np.int8))


def test_chunked_scales():
    w = _weights((4096, 1024), 5)
    q, s = quant.quantize_int4(w, 128)
    assert quant.k_chunk(4096) == 512 and quant.k_chunk(64) == 64
    ch = quant.scales_to_chunked(s, 4096)
    assert ch.shape == (8, 4, 1024) and ch.dtype == np.float32
    assert np.array_equal(quant.scales_from_chunked(ch), s)
    assert np.array_equal(ch[3, 2], s[3 * 4 + 2])
    assert np.array_equal(quant.dequantize_int4(q, ch), quant.dequantize_int4(q, s))
    assert np.array_equal(
        np.asarray(quant.dequantize_int4(jnp.asarray(q), jnp.asarray(ch))),
        quant.dequantize_int4(q, s),
    )
    short = quant.scales_to_chunked(s[:1, :], 64)  # K=64 (MINI down slice): one chunk, one group
    assert short.shape == (1, 1, 1024)
    with pytest.raises(ValueError):
        quant.quantize_int4(w, 100)
