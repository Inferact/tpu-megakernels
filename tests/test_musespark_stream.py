"""Tests for `musespark.stream` (dense-weight bank ring + vector helpers).

CPU: `JAX_PLATFORMS=cpu pytest tests/test_musespark_stream.py` streams a 3-layer MINI
schedule under the Pallas interpreter and checks every `gemv` against `jnp.dot`.
TPU (single chip): `tpu_run.sh 2 python -m pytest tests/test_musespark_stream.py -s` adds
the real per-rank shapes (`[L, 8192, 1024]` ...) with the achieved weight bandwidth.
"""

import dataclasses
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from musespark import layout, stream
from musespark.config import MINI, Config

ON_TPU = jax.default_backend() == "tpu"
COMPUTE = os.environ.get("MUSESPARK_STREAM_DMA_ONLY", "") == ""  # profiling: skip the dots
LAG = int(os.environ.get("MUSESPARK_STREAM_LAG", "0"))  # ring refill lag under test
BANKS = int(os.environ.get("MUSESPARK_STREAM_BANKS", str(layout.BANKS)))
BLOCKDIAG = os.environ.get("MUSESPARK_STREAM_BLOCKDIAG", "1") == "1"
FAMILIES = layout.STREAMED_FAMILIES
MIB = 1 << 20


def _inputs(cfg, batch, seed, tp=8, dense_format="bf16"):
    """Random bf16 activations for every K width plus the per-rank weights (keyed by the
    logical family; int8 families as `(q int8, scale f32 [L, 1, N])` pairs in int8)."""
    shapes = layout.rank_shapes(cfg, tp, dense_format=dense_format)
    key = jax.random.PRNGKey(seed)
    weights = {}
    for i, name in enumerate(FAMILIES + ("lm_head",)):
        shape, dtype = shapes[layout.dense_ref_name(name, dense_format)]
        k = jax.random.fold_in(key, i)
        if dtype == jnp.int8:
            q = jax.random.randint(k, shape, -127, 128, jnp.int32).astype(jnp.int8)
            s = jax.random.uniform(jax.random.fold_in(k, 1), shape[:-2] + (1, shape[-1]),
                                   jnp.float32, 0.5, 2.0) / 127
            weights[name] = (q, s)
        else:
            weights[name] = jax.random.normal(k, shape, jnp.float32).astype(dtype)
    widths = sorted({shapes[layout.dense_ref_name(n, dense_format)][0][-2]
                     for n in FAMILIES + ("lm_head",)})
    xs = {
        k: jax.random.normal(jax.random.fold_in(key, 100 + k), (batch, k), jnp.float32).astype(
            jnp.bfloat16
        )
        for k in widths
    }
    return xs, weights


def _dense(w):
    """The f32 matrix a weight entry stands for (dequantized int8 pair or bf16)."""
    if isinstance(w, tuple):
        return w[0].astype(jnp.float32) * w[1]
    return w.astype(jnp.float32)


def _make_kernel(cfg, batch, widths, with_lm_head, interpret, dense_format="bf16"):
    shapes = layout.rank_shapes(cfg, dense_format=dense_format)
    names = FAMILIES + (("lm_head",) if with_lm_head else ())
    scaled = [n for n in names if isinstance(shapes.get(n + "_s"), tuple)]
    mp = max(8, -(-batch // 8) * 8)

    def kernel(*refs):
        x_refs = dict(zip(widths, refs[: len(widths)]))
        refs = refs[len(widths) :]
        w_refs, s_refs = {}, {}
        for name in names:
            w_refs[layout.dense_ref_name(name, dense_format)], refs = refs[0], refs[1:]
            if name in scaled:
                s_refs[name], refs = refs[0], refs[1:]
        outs = dict(zip(names, refs[: len(names)]))
        scratch = refs[len(names) :]
        lm = w_refs.get(layout.dense_ref_name("lm_head", dense_format)) if with_lm_head else None
        ring = stream.make_ring(cfg, scratch, w_refs, lm, dense_format=dense_format,
                                bitcast=not interpret, lag=LAG, blockdiag=BLOCKDIAG)
        stream.prime(ring)

        def layer(l, carry):
            for name in FAMILIES:
                k = shapes[layout.dense_ref_name(name, dense_format)][0][1]
                scale = s_refs[name][l] if name in scaled else None
                outs[name][l] = stream.gemv(ring, x_refs[k], name, l, scale=scale,
                                            compute=COMPUTE)
            return carry

        lax.fori_loop(0, cfg.layers, layer, 0)
        if with_lm_head:
            stream.gemv(ring, x_refs[cfg.hidden], "lm_head", cfg.layers, acc=outs["lm_head"],
                        scale=s_refs.get("lm_head"), compute=COMPUTE)

    out_shapes, in_specs = [], []
    vm = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    for name in names:
        (shape, _) = shapes[layout.dense_ref_name(name, dense_format)]
        in_specs.append(hbm)
        if name in scaled:
            in_specs.append(vm)  # the scales ride along in VMEM ([L, 1, N] / [1, Vp] f32)
        if name == "lm_head":
            out_shapes.append(jax.ShapeDtypeStruct((mp, shape[1]), jnp.float32))
        else:
            out_shapes.append(jax.ShapeDtypeStruct((cfg.layers, batch, shape[2]), jnp.float32))
    return pl.pallas_call(
        kernel,
        out_shape=out_shapes,
        in_specs=[vm] * len(widths) + in_specs,
        out_specs=[vm] * len(out_shapes),
        scratch_shapes=list(stream.scratch_shapes(cfg, banks=BANKS, dense_format=dense_format,
                                                  bitcast=not interpret)),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * MIB),
        interpret=pltpu.InterpretParams() if interpret else False,
    ), names


def _check(cfg, batch, xs, weights, outs, names, rtol=1e-2):
    for name, out in zip(names, outs):
        w = _dense(weights[name])
        if name == "lm_head":
            ref = jnp.dot(xs[cfg.hidden].astype(jnp.float32), w, precision="highest")
            got = out[:batch]
        else:
            ref = jnp.einsum("bk,lkn->lbn", xs[w.shape[1]].astype(jnp.float32), w,
                             precision="highest")
            got = out
        ref, got = np.asarray(ref), np.asarray(got)
        scale = np.abs(ref).max()
        err = np.abs(got - ref).max()
        assert err <= rtol * scale, f"{name}: max err {err} vs scale {scale}"


def _args(xs, weights, names):
    out = [xs[k] for k in sorted(xs)]
    for n in names:
        out += list(weights[n]) if isinstance(weights[n], tuple) else [weights[n]]
    return out


def _run(cfg, batch, with_lm_head, interpret, seed=0, dense_format="bf16"):
    xs, weights = _inputs(cfg, batch, seed, dense_format=dense_format)
    widths = sorted(xs)
    f, names = _make_kernel(cfg, batch, widths, with_lm_head, interpret, dense_format)
    args = _args(xs, weights, names)
    outs = jax.jit(f)(*args)
    jax.block_until_ready(outs)
    if COMPUTE:
        _check(cfg, batch, xs, weights, outs, names)
    return f, args, names


MINI3 = dataclasses.replace(MINI, layers=3)
# hidden 2048 with 1024-row banks: q/kv/gate/pre/post/router tiles pack 2 K-tiles per load
PACK2 = dataclasses.replace(MINI, layers=2, hidden=2048, heads=32, kv_heads=8, moe_hidden=1024,
                            expert_hidden=512, experts=8, top_k=2, vocab=2048, vocab_used=2000)


@pytest.mark.parametrize("dense_format", ["bf16", "int8"])
@pytest.mark.parametrize("batch", [1, 4, 8])
def test_ring_gemv_mini_interpret(batch, dense_format):
    if ON_TPU:
        pytest.skip("interpret-mode test; see test_ring_gemv_tpu")
    _run(MINI3, batch, with_lm_head=True, interpret=True, dense_format=dense_format)


@pytest.mark.parametrize("dense_format", ["bf16", "int8"])
def test_ring_gemv_packed_interpret(dense_format):
    """Loads holding several K-tiles (block-diagonal dot path) under the interpreter."""
    if ON_TPU:
        pytest.skip("interpret-mode test")
    sched = layout.tile_schedule(PACK2, 8, dense_format)
    assert any(len({t.k0 for t in sched.layer if t.family == f}) > 1 for f in FAMILIES)
    _run(PACK2, 3, with_lm_head=True, interpret=True, dense_format=dense_format)


def test_ring_gemv_mini_interpret_no_lm_head():
    if ON_TPU:
        pytest.skip("interpret-mode test")
    _run(MINI3, 2, with_lm_head=False, interpret=True)


def test_fetch_dynamic_matches_static_interpret():
    """`fetch(ring, traced g)` issues the same tiles as the static path (a ring primed by
    a traced loop yields identical gemv results)."""
    if ON_TPU:
        pytest.skip("interpret-mode test")
    cfg = MINI3
    xs, weights = _inputs(cfg, 2, 3)
    shapes = layout.rank_shapes(cfg)

    def kernel(x_ref, *refs):
        w_refs = dict(zip(FAMILIES, refs[: len(FAMILIES)]))
        lm = refs[len(FAMILIES)]
        out = refs[len(FAMILIES) + 1]
        ring = stream.make_ring(cfg, refs[len(FAMILIES) + 2 :], w_refs, lm)

        def issue(g, c):
            stream.fetch(ring, g)
            return c

        lax.fori_loop(0, layout.BANKS, issue, 0)

        def layer(l, c):
            acc = c
            for name in FAMILIES:
                if shapes[name][0][1] == cfg.hidden:
                    acc = acc + jnp.sum(stream.gemv(ring, x_ref, name, l), axis=1, keepdims=True)
                else:
                    stream.gemv(ring, jnp.zeros((2, shapes[name][0][1]), jnp.bfloat16), name, l)
            return acc

        total = lax.fori_loop(0, cfg.layers, layer, jnp.zeros((2, 1), jnp.float32))
        out[...] = jnp.broadcast_to(total, (2, 128))
        stream.drain(ring, cfg.layers * ring.per)

    vm = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((2, 128), jnp.float32),
        in_specs=[vm] + [hbm] * (len(FAMILIES) + 1),
        out_specs=vm,
        scratch_shapes=list(stream.scratch_shapes(cfg)),
        interpret=pltpu.InterpretParams(),
    )
    got = np.asarray(f(xs[cfg.hidden], *[weights[n] for n in FAMILIES], weights["lm_head"]))
    exp = 0.0
    for name in FAMILIES:
        if shapes[name][0][1] == cfg.hidden:
            y = jnp.einsum(
                "bk,lkn->lbn", xs[cfg.hidden], weights[name], preferred_element_type=jnp.float32
            )
            exp = exp + np.asarray(y.sum(axis=(0, 2)))
    np.testing.assert_allclose(got[:, 0], exp, rtol=1e-2)


def test_vector_helpers():
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 256), jnp.float32)
    w = jax.random.normal(jax.random.PRNGKey(1), (1, 256), jnp.float32).astype(jnp.bfloat16)
    ref = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-5) * w.astype(jnp.float32)
    np.testing.assert_allclose(stream.rms(x, w, 1e-5), ref, rtol=1e-6)
    np.testing.assert_array_equal(stream.norm_to_bf16(x, w, 1e-5), ref.astype(jnp.bfloat16))
    np.testing.assert_array_equal(stream.r16(x), x.astype(jnp.bfloat16).astype(jnp.float32))
    a, b = 0.25 * jnp.ones((1, 256)), 0.5 * jnp.ones((1, 256))
    np.testing.assert_allclose(stream.gated_residual(x, ref, a, b), 0.25 * x + 0.5 * ref)
    np.testing.assert_allclose(stream.silu(x), x * jax.nn.sigmoid(x))
    assert stream.mxu_rows(x[:1]).shape == (8, 256)
    assert stream.mxu_rows(x[:3]).shape == (8, 256)
    assert stream.mxu_rows(jnp.zeros((8, 256))).shape == (8, 256)
    assert stream.mxu_rows(jnp.zeros((12, 256))).shape == (16, 256)


def test_prefetch_vector_interpret():
    if ON_TPU:
        pytest.skip("interpret-mode test")
    L, W = 3, 256
    v = jax.random.normal(jax.random.PRNGKey(5), (L, 1, W), jnp.float32).astype(jnp.bfloat16)

    def kernel(v_hbm, o_ref, buf, sems):
        stream.prefetch_vector(v_hbm, buf, sems, 0)

        def layer(l, c):
            stream.wait_vector(v_hbm, buf, sems, l)

            @pl.when(l + 1 < L)
            def _():
                stream.prefetch_vector(v_hbm, buf, sems, l + 1)

            o_ref[l] = stream.vector(buf, l).astype(jnp.float32) * 2.0
            return c

        lax.fori_loop(0, L, layer, 0)

    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((L, 1, W), jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        scratch_shapes=list(stream.vector_scratch(W)),
        interpret=pltpu.InterpretParams(),
    )
    np.testing.assert_array_equal(np.asarray(f(v)), 2.0 * np.asarray(v.astype(jnp.float32)))


# ---------------------------------------------------------------------------------------
# TPU: MINI and real per-rank shapes, weight-stream bandwidth
# ---------------------------------------------------------------------------------------
REAL16 = dataclasses.replace(Config(), layers=16)


def _time(f, args, iters=5):
    out = f(*args)
    jax.block_until_ready(out)
    jax.block_until_ready(f(*args))
    t0 = time.perf_counter()
    for _ in range(iters):
        out = f(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters


@pytest.mark.skipif(not ON_TPU, reason="needs a TPU")
@pytest.mark.parametrize("dense_format", ["bf16", "int8"])
@pytest.mark.parametrize("batch", [1, 8])
def test_ring_gemv_mini_tpu(batch, dense_format):
    _run(MINI, batch, with_lm_head=True, interpret=False, dense_format=dense_format)


@pytest.mark.skipif(not ON_TPU, reason="needs a TPU")
@pytest.mark.parametrize("dense_format", ["bf16", "int8"])
def test_ring_gemv_real_shapes_tpu(dense_format):
    cfg = REAL16
    bytes_ = layout.bytes_per_rank(cfg, dense_format=dense_format)
    dense = bytes_["dense"]
    lm = bytes_["vocab"] - bytes_["embed"]
    f_lm, args_lm, _ = _run(cfg, 8, with_lm_head=True, interpret=False, dense_format=dense_format)
    f_no, args_no, _ = _run(cfg, 1, with_lm_head=False, interpret=False, dense_format=dense_format)
    f_b1, args_b1, _ = _run(cfg, 1, with_lm_head=True, interpret=False, dense_format=dense_format)
    t_lm = _time(jax.jit(f_lm), args_lm)
    t_no = _time(jax.jit(f_no), args_no)
    t_b1 = _time(jax.jit(f_b1), args_b1)
    per_layer = t_no / cfg.layers
    print(
        f"\n[{dense_format}] weight stream, {cfg.layers} layers x {dense / cfg.layers / MIB:.1f} MiB"
        f" dense (+ lm_head {lm / MIB:.0f} MiB):"
        f"\n  B=1 dense only : {t_no * 1e6:8.1f} us -> {dense / t_no / 1e9:7.0f} GB/s,"
        f" {per_layer * 1e6:.1f} us per layer"
        f"\n  B=1 + lm_head  : {t_b1 * 1e6:8.1f} us -> {(dense + lm) / t_b1 / 1e9:7.0f} GB/s"
        f" (lm_head alone ~{(t_b1 - t_no) * 1e6:.0f} us)"
        f"\n  B=8 + lm_head  : {t_lm * 1e6:8.1f} us -> {(dense + lm) / t_lm / 1e9:7.0f} GB/s"
        f" (lm_head alone ~{(t_lm - t_no) * 1e6:.0f} us)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-s", "-q"])
