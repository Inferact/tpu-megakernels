"""Tests for `musespark.collectives` (8 ranks over mesh axis "tp").

CPU: `XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu pytest ...`
runs the kernels under `pltpu.InterpretParams` (remote DMAs + semaphores are emulated).
TPU: `tpu_run.sh all python -m pytest tests/test_musespark_collectives.py` also runs the
bit-exactness and timing checks (`-s` prints the per-op microseconds).
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from musespark import collectives as col

TP = 8
ON_TPU = jax.default_backend() == "tpu"
INTERPRET = pltpu.InterpretParams(dma_execution_mode="eager")


def _mesh():
    devices = sorted(jax.devices(), key=lambda d: d.id)
    if len(devices) < TP:
        pytest.skip("needs 8 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)")
    return Mesh(np.array(devices[:TP]), (col.AXIS,))


def _call(kernel, out_shapes, in_specs, scratch, collective_id, interpret):
    vm = pl.BlockSpec(memory_space=pltpu.VMEM)
    return pl.pallas_call(
        kernel,
        out_shape=out_shapes,
        in_specs=in_specs,
        out_specs=[vm for _ in out_shapes] if isinstance(out_shapes, list) else vm,
        scratch_shapes=scratch,
        compiler_params=col.compiler_params(collective_id),
        interpret=INTERPRET if interpret else False,
    )


def _sharded(mesh, f, n_in, n_out):
    return jax.jit(
        jax.shard_map(
            f,
            mesh=mesh,
            in_specs=tuple(P(col.AXIS) for _ in range(n_in)),
            out_specs=tuple(P(col.AXIS) for _ in range(n_out)) if n_out > 1 else P(col.AXIS),
            check_vma=False,
        )
    )


def _put(mesh, x):
    return jax.device_put(x, NamedSharding(mesh, P(col.AXIS)))


# ---------------------------------------------------------------------------------------
# mixed kinds back to back: all_reduce [R, W] (phase 0), all_gather [R, w] (phase 1),
# all_reduce [R2, W2] (phase 2), all_gather bf16 (phase 3)
# ---------------------------------------------------------------------------------------
def _mixed_kernel(a_ref, g_ref, b_ref, gb_ref, oa_ref, og_ref, ob_ref, ogb_ref, o16_ref, *scratch):
    ws = col.workspace(*scratch)
    col.barrier()
    oa_ref[...] = col.all_reduce_rows(a_ref, ws, jnp.int32(0))
    og_ref[...] = col.all_gather_rows(g_ref[...], ws, jnp.int32(1))
    ob_ref[...] = col.all_reduce_rows(b_ref[...] * 2.0, ws, jnp.int32(2))
    ogb_ref[...] = col.all_gather_rows(gb_ref[...], ws, jnp.int32(3))
    o16_ref[...] = col.all_reduce_rows(a_ref, ws, jnp.int32(4), wire=jnp.bfloat16)


def _mixed_case(interpret, ra=8, wa=8192, rb=64, wb=4096, gw=1024, gbw=512, seed=0):
    mesh = _mesh()
    key = jax.random.PRNGKey(seed)
    ka, kg, kb, kgb = jax.random.split(key, 4)
    a = jax.random.normal(ka, (TP * ra, wa), jnp.float32)
    g = jax.random.normal(kg, (TP * ra, gw), jnp.float32)
    b = jax.random.normal(kb, (TP * rb, wb), jnp.float32)
    gb = jax.random.normal(kgb, (TP * ra, gbw), jnp.float32).astype(jnp.bfloat16)
    outs = [
        jax.ShapeDtypeStruct((ra, wa), jnp.float32),
        jax.ShapeDtypeStruct((ra, TP * gw), jnp.float32),
        jax.ShapeDtypeStruct((rb, wb), jnp.float32),
        jax.ShapeDtypeStruct((ra, TP * gbw), jnp.bfloat16),
        jax.ShapeDtypeStruct((ra, wa), jnp.float32),
    ]
    vm = pl.BlockSpec(memory_space=pltpu.VMEM)
    scratch = col.scratch_shapes(
        max(ra, rb), max(wa, wb), gather_width=max(gw, gbw), bf16_wire=True
    )
    f = _call(_mixed_kernel, outs, [vm] * 4, scratch, 11, interpret)
    run = _sharded(mesh, f, 4, 5)
    oa, og, ob, ogb, o16 = run(_put(mesh, a), _put(mesh, g), _put(mesh, b), _put(mesh, gb))
    oa, og, ob, ogb, o16 = (np.asarray(t.astype(jnp.float32)) for t in (oa, og, ob, ogb, o16))
    an, gn, bn, gbn = (np.asarray(t.astype(jnp.float32)) for t in (a, g, b, gb))
    exp_a = an.reshape(TP, ra, wa).sum(0)
    exp_b = 2.0 * bn.reshape(TP, rb, wb).sum(0)
    exp_g = gn.reshape(TP, ra, gw).transpose(1, 0, 2).reshape(ra, TP * gw)
    exp_gb = gbn.reshape(TP, ra, gbw).transpose(1, 0, 2).reshape(ra, TP * gbw)
    oa = oa.reshape(TP, ra, wa)
    og = og.reshape(TP, ra, TP * gw)
    ob = ob.reshape(TP, rb, wb)
    ogb = ogb.reshape(TP, ra, TP * gbw)
    o16 = o16.reshape(TP, ra, wa)
    exp_16 = np.asarray(a.astype(jnp.bfloat16).astype(jnp.float32)).reshape(TP, ra, wa).sum(0)
    exp_16 = np.asarray(jnp.asarray(exp_16).astype(jnp.bfloat16).astype(jnp.float32))
    for r in range(TP):
        np.testing.assert_allclose(o16[r], exp_16, rtol=2e-2, atol=1e-2)
        np.testing.assert_array_equal(o16[r], o16[0])
        np.testing.assert_allclose(oa[r], exp_a, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(ob[r], exp_b, rtol=1e-5, atol=1e-4)
        np.testing.assert_array_equal(og[r], exp_g)
        np.testing.assert_array_equal(ogb[r], exp_gb)
        # bit-identical across ranks
        np.testing.assert_array_equal(oa[r], oa[0])
        np.testing.assert_array_equal(ob[r], ob[0])


def test_mixed_collectives_interpret():
    if ON_TPU:
        pytest.skip("interpret-mode test; the TPU variant is test_mixed_collectives_tpu")
    _mixed_case(interpret=True)


def test_mini_shapes_interpret():
    """MINI config widths: all-reduce [8, 1024] (128-lane blocks), gathers of 64/128 lanes."""
    if ON_TPU:
        pytest.skip("interpret-mode test")
    _mixed_case(interpret=True, ra=8, wa=1024, rb=32, wb=1024, gw=128, gbw=64, seed=1)


@pytest.mark.skipif(not ON_TPU, reason="needs 8 TPU cores")
def test_mixed_collectives_tpu():
    _mixed_case(interpret=False)
    _mixed_case(interpret=False, ra=8, wa=1024, rb=32, wb=1024, gw=128, gbw=64, seed=1)


# ---------------------------------------------------------------------------------------
# reduce vs psum, gather vs all_gather, all_reduce output identical on every rank
# ---------------------------------------------------------------------------------------
def _psum_kernel(x_ref, o_ref, *scratch):
    ws = col.workspace(*scratch)
    col.barrier()
    o_ref[...] = col.all_reduce_rows(x_ref[...], ws, jnp.int32(0))


@pytest.mark.parametrize("rows,width", [(8, 8192), (64, 4096), (16, 2048)])
def test_all_reduce_vs_psum(rows, width):
    mesh = _mesh()
    x = jax.random.normal(jax.random.PRNGKey(2), (TP * rows, width), jnp.float32)
    vm = pl.BlockSpec(memory_space=pltpu.VMEM)
    f = _call(
        _psum_kernel,
        jax.ShapeDtypeStruct((rows, width), jnp.float32),
        [vm],
        col.scratch_shapes(rows, width),
        12,
        not ON_TPU,
    )
    got = np.asarray(_sharded(mesh, f, 1, 1)(_put(mesh, x))).reshape(TP, rows, width)
    ref = jax.jit(
        jax.shard_map(
            lambda v: lax.psum(v, col.AXIS), mesh=mesh, in_specs=P(col.AXIS), out_specs=P(col.AXIS)
        )
    )(_put(mesh, x))
    ref = np.asarray(ref).reshape(TP, rows, width)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-4)
    for r in range(1, TP):
        np.testing.assert_array_equal(got[r], got[0])


# ---------------------------------------------------------------------------------------
# TPU timing: 200 in-kernel iterations of each collective
# ---------------------------------------------------------------------------------------
REPS = 200


def _timed_reduce_kernel(x_ref, o_ref, *scratch, wire=jnp.float32):
    ws = col.workspace(*scratch)
    col.barrier()

    def body(i, carry):
        o_ref[...] = col.all_reduce_rows(x_ref, ws, i, wire=wire)
        return carry

    lax.fori_loop(0, REPS, body, 0)


def _timed_gather_kernel(x_ref, o_ref, *scratch):
    ws = col.workspace(*scratch)
    col.barrier()

    def body(i, carry):
        o_ref[...] = col.all_gather_rows(x_ref[...], ws, i)
        return carry

    lax.fori_loop(0, REPS, body, 0)


def _time(run, x, iters=5):
    out = run(x)
    out.block_until_ready()
    run(x).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = run(x)
    out.block_until_ready()
    return (time.perf_counter() - t0) / iters / REPS, out


@pytest.mark.skipif(not ON_TPU, reason="timing needs the TPU")
def test_timing_tpu():
    mesh = _mesh()
    vm = pl.BlockSpec(memory_space=pltpu.VMEM)
    report = []
    for rows, width, wire in [
        (8, 8192, jnp.float32),
        (64, 4096, jnp.float32),
        (8, 8192, jnp.bfloat16),
        (64, 4096, jnp.bfloat16),
    ]:
        x = jax.random.normal(jax.random.PRNGKey(3), (TP * rows, width), jnp.float32)
        if wire == jnp.bfloat16:
            x = x.astype(jnp.bfloat16).astype(jnp.float32)
        f = _call(
            functools.partial(_timed_reduce_kernel, wire=wire),
            jax.ShapeDtypeStruct((rows, width), jnp.float32),
            [vm],
            col.scratch_shapes(rows, width, bf16_wire=wire == jnp.bfloat16),
            13,
            False,
        )
        us, out = _time(_sharded(mesh, f, 1, 1), _put(mesh, x))
        got = np.asarray(out).reshape(TP, rows, width)
        exp = np.asarray(x).reshape(TP, rows, width).sum(0)
        tol = 2e-2 if wire == jnp.bfloat16 else 1e-5
        np.testing.assert_allclose(got[0], exp, rtol=tol, atol=tol)
        for r in range(1, TP):
            np.testing.assert_array_equal(got[r], got[0])
        report.append(
            f"all_reduce_rows [{rows}, {width}] wire {jnp.dtype(wire).name}: {us * 1e6:.2f} us"
        )
    for rows, w, dtype in [(8, 1024, jnp.bfloat16), (8, 1024, jnp.float32), (8, 512, jnp.bfloat16)]:
        x = jax.random.normal(jax.random.PRNGKey(4), (TP * rows, w), jnp.float32).astype(dtype)
        f = _call(
            _timed_gather_kernel,
            jax.ShapeDtypeStruct((rows, TP * w), dtype),
            [vm],
            col.scratch_shapes(rows, TP * w),
            14,
            False,
        )
        us, out = _time(_sharded(mesh, f, 1, 1), _put(mesh, x))
        got = np.asarray(out.astype(jnp.float32)).reshape(TP, rows, TP * w)
        exp = np.asarray(x.astype(jnp.float32)).reshape(TP, rows, w).transpose(1, 0, 2)
        exp = exp.reshape(rows, TP * w)
        for r in range(TP):
            np.testing.assert_array_equal(got[r], exp)
        report.append(f"all_gather_rows [{rows}, {w}] {jnp.dtype(dtype).name}: {us * 1e6:.2f} us")
    print("\n" + "\n".join(report))


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-s", "-q"])
