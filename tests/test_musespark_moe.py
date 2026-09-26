"""Muse Spark MoE mini-kernel tests (`musespark/moe.py`).

CPU (`JAX_PLATFORMS=cpu`): the real Pallas bodies run with ``interpret=True`` against
independent numpy/jnp references. TPU (single chip): the same correctness checks on real
per-rank shapes plus per-layer timings (printed with ``-s``).
"""

import functools
import importlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from musespark.config import MINI, Config
from musespark.quant import dequantize_int4, quantize_int4, scales_to_chunked

# `musespark/__init__.py` also defines a reference function named `moe`, so the submodule
# must be fetched by name rather than via `from musespark import moe`.
moe = importlib.import_module("musespark.moe")

ON_TPU = jax.devices()[0].platform == "tpu"
INTERPRET = not ON_TPU
tpu_only = pytest.mark.skipif(not ON_TPU, reason="needs a TPU")

TP = 8
GAP_ITERS = 40  # busy-loop rounds between early start and expert_stream in the timing test
VM = pl.BlockSpec(memory_space=pltpu.VMEM)
ANY = pl.BlockSpec(memory_space=pl.ANY)


def real_cfg(experts, layers=1):
    """Real per-rank expert shapes (Hm 4096, Is 512, G 128) with fewer experts/layers."""
    return Config(layers=layers, experts=experts)


def r16(x):
    return np.asarray(np.asarray(x, np.float32).astype(jnp.bfloat16), np.float32)


# ---------------------------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------------------------


def np_route(cfg, logits, bias):
    """Independent routing reference: iterative argmax (numpy argmax = first max index)."""
    z = np.asarray(logits, np.float32) * np.float32(cfg.output_multiplier)
    sc = (1 / (1 + np.exp(-z))).astype(np.float32)
    sel = sc + np.asarray(bias, np.float32).reshape(1, -1)
    idx = np.zeros((logits.shape[0], cfg.top_k), np.int32)
    w = np.zeros((logits.shape[0], cfg.top_k), np.float32)
    for b in range(logits.shape[0]):
        s = sel[b].copy()
        for k in range(cfg.top_k):
            j = int(np.argmax(s))
            idx[b, k] = j
            w[b, k] = sc[b, j]
            s[j] = -np.inf
        w[b] = w[b] / (w[b].sum(dtype=np.float32) + np.float32(cfg.route_eps))
    return idx, w


def random_expert_weights(cfg, key, tp=TP, gu_scale=0.004, dn_scale=0.01):
    """Random already-quantized per-rank expert arrays in the kernel layout (jax arrays)."""
    L, E, Hm, G = cfg.layers, cfg.experts, cfg.moe_hidden, cfg.group_size
    Is = cfg.expert_hidden // tp
    k1, k2, k3, k4 = jax.random.split(key, 4)

    def scales(k, K, N, mag):
        s = jax.random.uniform(k, (L, E, K // G, N), jnp.float32, 0.5, 1.5) * mag
        return scales_to_chunked(s.astype(jnp.bfloat16).astype(jnp.float32), K)

    gu_q = jax.random.randint(k1, (L, E, Hm, 2 * Is), -8, 8, jnp.int8).astype(jnp.int4)
    dn_q = jax.random.randint(k2, (L, E, Is, Hm), -8, 8, jnp.int8).astype(jnp.int4)
    return moe.ExpertWeights(
        gu_q, scales(k3, Hm, 2 * Is, gu_scale), dn_q, scales(k4, Is, Hm, dn_scale)
    )


@functools.partial(jax.jit, static_argnames=("Is",))
def _ref_expert(h1_row, gu_q, gu_s, dn_q, dn_s, Is):
    """One row through one expert on dequantized f32 weights -> this rank's partial y."""
    hi = lax.Precision.HIGHEST
    gu = jnp.dot(h1_row[None], dequantize_int4(gu_q, gu_s), precision=hi)[0]
    gate, up = gu[:Is], gu[Is:]
    a = moe.r16(gate * jax.nn.sigmoid(gate) * up)
    return jnp.dot(a[None], dequantize_int4(dn_q, dn_s), precision=hi)[0]


def ref_partial_outputs(cfg, layer, h1, idx, weights, tp=TP):
    """`y_out` reference: `[K*B, Hm]` f32 with row k*B + b = partial output of expert idx[b, k]."""
    B, K = idx.shape
    Is = cfg.expert_hidden // tp
    Y = np.zeros((K * B, cfg.moe_hidden), np.float32)
    h1 = jnp.asarray(h1, jnp.float32)
    for b in range(B):
        for k in range(K):
            e = int(idx[b, k])
            y = _ref_expert(
                h1[b],
                weights.gate_up_q[layer, e],
                weights.gate_up_s[layer, e],
                weights.down_q[layer, e],
                weights.down_s[layer, e],
                Is=Is,
            )
            Y[k * B + b] = np.asarray(y)
    return Y


def ref_finalize(cfg, Y, w, post):
    """Spec 3.5 finalize in numpy: r16 the bf16 GEMM output, input-scaled RMS norm, weighted sum."""
    B, K = w.shape
    m = np.zeros((B, Y.shape[1]), np.float32)
    for k in range(K):
        t = r16(r16(Y[k * B : (k + 1) * B]) * post.reshape(1, -1))
        yn = t / np.sqrt(np.mean(t * t, axis=1, keepdims=True) + np.float32(cfg.post_eps))
        m += w[:, k : k + 1] * yn
    return r16(m)


# ---------------------------------------------------------------------------------------------
# Kernel wrappers
# ---------------------------------------------------------------------------------------------


def run_int4_group_dot(x_bd, w_q, s):
    def kernel(x_ref, w_ref, s_ref, o_ref):
        o_ref[...] = moe.int4_group_dot(x_ref, w_ref, s_ref)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((moe.MR, w_q.shape[1]), jnp.float32),
        in_specs=[VM, VM, VM],
        out_specs=VM,
        interpret=INTERPRET,
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=48 << 20),
    )(x_bd, w_q, s)


def run_route(cfg, x, hi, lo, bias):
    def kernel(x_ref, hi_ref, lo_ref, b_ref, idx_ref, w_ref):
        idx_ref[...], w_ref[...] = moe.route(cfg, x_ref[...], hi_ref[...], lo_ref[...], b_ref[...])

    B = x.shape[0]
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((B, moe.LANES), jnp.int32),
            jax.ShapeDtypeStruct((B, moe.LANES), jnp.float32),
        ),
        in_specs=[VM] * 4,
        out_specs=(VM, VM),
        interpret=INTERPRET,
    )(x, hi, lo, bias)


def run_route_from_logits(cfg, logits, bias):
    def kernel(l_ref, b_ref, idx_ref, w_ref):
        idx_ref[...], w_ref[...] = moe.route_from_logits(cfg, l_ref[...], b_ref[...])

    B = logits.shape[0]
    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((B, moe.LANES), jnp.int32),
            jax.ShapeDtypeStruct((B, moe.LANES), jnp.float32),
        ),
        in_specs=[VM, VM],
        out_specs=(VM, VM),
        interpret=INTERPRET,
    )(logits, bias)


def make_moe_call(cfg, batch, layers, slots=moe.SLOTS, early_start=False, gap_iters=0):
    """`(idx_tile, w_tile, h1, post, gate_up_q, gate_up_s, down_q, down_s) -> (y_out, m)` running
    route materialisation + prefetch + expert_stream + (fake no-op all-reduce) + finalize for
    `layers` layers looped in-kernel; the outputs are those of the last layer.

    `early_start` issues the next layer's first wave of DMAs at the end of the current layer,
    followed by `gap_iters` iterations of a VPU busy loop (see `gap_time`), emulating the
    integrator, which calls `start_expert_stream` right after routing (before
    pre_expert_proj / all-gather / pre_expert_norm) so the first wave has landed by the time
    `expert_stream` runs."""
    K, Hm = cfg.top_k, cfg.moe_hidden

    def kernel(idx_ref, w_ref, h1_ref, post_ref, gu_q, gu_s, dn_q, dn_s, y_ref, m_ref, *scratch):
        sc = moe.MoeScratch.bind(scratch)
        weights = moe.ExpertWeights(gu_q, gu_s, dn_q, dn_s)

        def start(layer):
            moe.route_to_scratch(cfg, sc, idx_ref[...], w_ref[...])
            moe.start_expert_stream(cfg, layer, weights, sc)

        if early_start:
            start(0)

        @pl.loop(0, layers)
        def _layer(layer):
            if not early_start:
                start(layer)
            moe.expert_stream(cfg, layer, h1_ref[...], weights, sc, started=True)
            y_ref[...] = sc.y_out[...]  # the integrator all-reduces this across ranks
            m_ref[...] = moe.finalize(cfg, sc.y_out, sc.w, post_ref[...], batch)
            if early_start:

                @pl.when(layer + 1 < layers)
                def _prefetch_next():
                    start(layer + 1)

            if gap_iters:
                m_ref[...] += busy_loop(gap_iters, m_ref[...])

    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((K * batch, Hm), jnp.float32),
            jax.ShapeDtypeStruct((batch, Hm), jnp.float32),
        ),
        in_specs=[VM, VM, VM, VM, ANY, ANY, ANY, ANY],
        out_specs=(VM, VM),
        scratch_shapes=moe.scratch_shapes(cfg, batch, TP, slots),
        interpret=INTERPRET,
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 << 20),
    )
    return jax.jit(call)


def busy_loop(iters, x):
    """VPU busy work (`iters` dependent rounds of sin on `x`), used to emulate the integrator's
    work between `start_expert_stream` and `expert_stream`."""
    return lax.fori_loop(0, iters, lambda i, a: jnp.sin(a) * 0.5, x * 1e-3)


def gap_time(shape, iters, layers=8):
    """Seconds per kernel of `layers` x `busy_loop(iters)` on a `shape` f32 tile."""

    def kernel(x_ref, o_ref):
        @pl.loop(0, layers)
        def _l(layer):
            o_ref[...] += busy_loop(iters, x_ref[...])

    call = jax.jit(pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(shape, jnp.float32)))
    return _timeit(call, jnp.ones(shape, jnp.float32))


def route_tiles(cfg, idx, w):
    B = idx.shape[0]
    it = np.zeros((B, moe.LANES), np.int32)
    wt = np.zeros((B, moe.LANES), np.float32)
    it[:, : cfg.top_k] = idx
    wt[:, : cfg.top_k] = w
    return jnp.asarray(it), jnp.asarray(wt)


def random_routes(cfg, batch, rng, distinct=None):
    """`[B, K]` distinct-per-row expert ids; `distinct` groups rows so that the batch uses that
    many distinct experts in total (None: random)."""
    B, K, E = batch, cfg.top_k, cfg.experts
    if distinct is None:
        idx = np.stack([rng.choice(E, K, replace=False) for _ in range(B)])
    else:
        rows_per_group = max(1, B * K // distinct)
        idx = np.zeros((B, K), np.int64)
        for b in range(B):
            g = b // rows_per_group
            idx[b] = (np.arange(K) + g * K) % E
    w = rng.uniform(0.1, 1, (B, K)).astype(np.float32)
    w = w / w.sum(1, keepdims=True)
    return idx.astype(np.int32), w


def check_stream(cfg, batch, layers, weights, idx, w, rng, verbose=False):
    Hm = cfg.moe_hidden
    h1 = r16(rng.standard_normal((batch, Hm)).astype(np.float32))
    post = r16(1 + 0.1 * rng.standard_normal((1, Hm)).astype(np.float32))
    call = make_moe_call(cfg, batch, layers)
    it, wt = route_tiles(cfg, idx, w)
    y, m = call(it, wt, jnp.asarray(h1), jnp.asarray(post), *vars(weights).values())
    y, m = np.asarray(y), np.asarray(m)
    Y_ref = ref_partial_outputs(cfg, layers - 1, h1, idx, weights)
    m_ref = ref_finalize(cfg, Y_ref, w, post)
    y_err = np.abs(y - Y_ref).max() / np.abs(Y_ref).max()
    m_err = np.abs(m - m_ref).max()
    distinct = len(np.unique(idx))
    if verbose:
        print(
            f"  B={batch} distinct={distinct}: rel y_out err {y_err:.2e}, max |m| err {m_err:.2e}"
        )
    assert y_err <= 2e-3, y_err
    assert m_err <= 2e-2, m_err
    return y_err, m_err


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "K,N,G",
    [(4096, 1024, 128), (512, 4096, 128), (512, 128, 64), (64, 512, 64)],
    ids=["real-gate_up", "real-down", "mini-gate_up", "mini-down"],
)
def test_int4_group_dot(K, N, G):
    rng = np.random.default_rng(K + N)
    x = r16(rng.standard_normal((moe.MR, K)).astype(np.float32))
    q = rng.integers(-8, 8, (K, N), dtype=np.int8)
    s = r16(rng.uniform(0.5, 1.5, (K // G, N)).astype(np.float32))
    out = run_int4_group_dot(
        moe.block_diag(jnp.asarray(x), G),
        jnp.asarray(q).astype(jnp.int4),
        jnp.asarray(scales_to_chunked(s, K)),
    )
    ref = x @ (q.astype(np.float32) * np.repeat(s, G, axis=0))
    err = np.abs(np.asarray(out) - ref).max() / np.abs(ref).max()
    assert err <= 1e-3, err


def test_int4_group_dot_quantized_layout():
    """Kernel dot on `quantize_int4` output (chunked scales) == `dequantize_int4` reference."""
    K, N, G = 512, 256, 64
    rng = np.random.default_rng(3)
    w = rng.standard_normal((K, N)).astype(np.float32)
    q, s = quantize_int4(w, group=G)
    x = r16(rng.standard_normal((moe.MR, K)).astype(np.float32))
    s_chunked = scales_to_chunked(s, K)
    out = run_int4_group_dot(
        moe.block_diag(jnp.asarray(x), G), jnp.asarray(q).astype(jnp.int4), jnp.asarray(s_chunked)
    )
    ref = x @ dequantize_int4(q, s_chunked)
    err = np.abs(np.asarray(out) - ref).max() / np.abs(ref).max()
    assert err <= 1e-3, err


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("cfg", [MINI, Config()], ids=["mini", "real"])
def test_route_from_logits(cfg, batch):
    rng = np.random.default_rng(batch)
    B, E, K = batch, cfg.experts, cfg.top_k
    logits = rng.standard_normal((B, E)).astype(np.float32) * 3
    bias = rng.standard_normal((1, E)).astype(np.float32) * 0.1
    # Exact ties: same logit AND same bias -> the lowest index must win first.
    logits[:, 5] = logits[:, 3] = logits.max() + 1
    bias[0, 5] = bias[0, 3] = bias.max() + 1
    if B > 1:
        logits[1] = 0  # a whole row of logit ties -> ordered by bias, 3 before 5
    idx, w = run_route_from_logits(cfg, jnp.asarray(logits), jnp.asarray(bias))
    idx, w = np.asarray(idx), np.asarray(w)
    idx_ref, w_ref = np_route(cfg, logits, bias)
    assert (idx[:, :K] == idx_ref).all(), (idx[:, :K], idx_ref)
    assert (idx[:, K:] == E).all() and (w[:, K:] == 0).all()
    assert idx[0, 0] == 3 and idx[0, 1] == 5
    if B > 1:
        assert idx[1, 0] == 3 and idx[1, 1] == 5
    np.testing.assert_allclose(w[:, :K], w_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(w[:, :K].sum(1), 1, rtol=1e-5)


@pytest.mark.parametrize("batch", [1, 8])
def test_route(batch):
    """Router hi/lo split dot + selection vs an fp32 numpy router on the same fp32 weights."""
    cfg = Config(experts=64)  # E=64 keeps the [H, E] matrices small; H stays 8192
    rng = np.random.default_rng(10 + batch)
    B, H, E = batch, cfg.hidden, cfg.experts
    x = r16(rng.standard_normal((B, H)).astype(np.float32))
    W = (rng.standard_normal((H, E)) / np.sqrt(H)).astype(np.float32) * 4
    W[:, 7] = W[:, 2]  # identical columns -> exact logit tie between experts 2 and 7
    bias = rng.standard_normal((1, E)).astype(np.float32) * 0.05
    bias[0, 7] = bias[0, 2]
    hi = W.astype(jnp.bfloat16)
    lo = (W - np.asarray(hi, np.float32)).astype(jnp.bfloat16)
    idx, w = run_route(cfg, jnp.asarray(x), jnp.asarray(hi), jnp.asarray(lo), jnp.asarray(bias))
    idx, w = np.asarray(idx), np.asarray(w)
    logits = x @ (np.asarray(hi, np.float32) + np.asarray(lo, np.float32))
    idx_ref, w_ref = np_route(cfg, logits, bias)
    assert (idx[:, : cfg.top_k] == idx_ref).all(), (idx[:, : cfg.top_k], idx_ref)
    np.testing.assert_allclose(w[:, : cfg.top_k], w_ref, rtol=1e-4, atol=1e-5)
    for b in range(B):
        row = list(idx_ref[b])
        if 2 in row and 7 in row:
            assert row.index(2) < row.index(7)


@pytest.mark.parametrize("batch", [1, 8])
def test_route_matches_package_reference(batch):
    """Kernel routing vs the pure-JAX reference `musespark.route` (A1), incl. exact ties."""
    from musespark import route as ref_route

    cfg = Config()
    rng = np.random.default_rng(20 + batch)
    logits = rng.standard_normal((batch, cfg.experts)).astype(np.float32) * 3
    bias = rng.standard_normal((1, cfg.experts)).astype(np.float32) * 0.1
    logits[:, 17] = logits[:, 11] = logits.max() + 1
    bias[0, 17] = bias[0, 11] = bias.max() + 1
    idx, w = run_route_from_logits(cfg, jnp.asarray(logits), jnp.asarray(bias))
    idx_ref, w_ref = ref_route(logits * cfg.output_multiplier, bias, cfg.top_k, cfg.route_eps)
    assert (np.asarray(idx)[:, : cfg.top_k] == np.asarray(idx_ref)).all()
    np.testing.assert_allclose(
        np.asarray(w)[:, : cfg.top_k], np.asarray(w_ref), rtol=1e-5, atol=1e-6
    )


def test_scratch_budget():
    real = moe.scratch_bytes(Config(), 8, TP)  # 8 packed slots x 3.1 MiB + scales + buffers
    assert real <= 28 << 20, real / 2**20
    assert moe.scratch_bytes(MINI, 8, TP) < 1 << 20


def test_finalize_matches_reference():
    cfg = MINI
    rng = np.random.default_rng(1)
    B, K, Hm = 4, cfg.top_k, cfg.moe_hidden
    Y = rng.standard_normal((K * B, Hm)).astype(np.float32)
    w = rng.uniform(size=(B, K)).astype(np.float32)
    post = r16(1 + 0.1 * rng.standard_normal((1, Hm)).astype(np.float32))
    wt = np.zeros((B, moe.LANES), np.float32)
    wt[:, :K] = w
    m = moe.finalize(cfg, jnp.asarray(Y), jnp.asarray(wt), jnp.asarray(post), B)
    np.testing.assert_allclose(np.asarray(m), ref_finalize(cfg, Y, w, post), atol=1e-2)


@pytest.mark.parametrize("batch", [1, 4, 8])
def test_expert_stream_mini(batch):
    cfg = MINI  # E=16, K=4, G=64, per-rank gate_up [512, 128] / down [64, 512]
    rng = np.random.default_rng(100 + batch)
    weights = random_expert_weights(cfg, jax.random.PRNGKey(batch), gu_scale=0.02, dn_scale=0.03)
    # Duplicated experts across rows: two groups of rows share their experts, plus one row
    # that overlaps both groups (exercises the dedupe and partial waves).
    idx, w = random_routes(cfg, batch, rng, distinct=min(2, batch) * cfg.top_k)
    if batch >= 4:
        idx[2] = np.array([1, 2, 5, 9])
        idx[3] = np.array([0, 3, 12, 15])
    check_stream(cfg, batch, 2, weights, idx, w, rng)


@pytest.mark.parametrize("batch,distinct", [(1, None), (8, None), (8, 8), (4, 32)])
def test_expert_stream_real_shapes(batch, distinct):
    cfg = real_cfg(experts=32)
    rng = np.random.default_rng(batch)
    weights = random_expert_weights(cfg, jax.random.PRNGKey(7))
    idx, w = random_routes(cfg, batch, rng, distinct)
    check_stream(cfg, batch, 1, weights, idx, w, rng)


# ---------------------------------------------------------------------------------------------
# TPU: correctness on real per-rank shapes + timing
# ---------------------------------------------------------------------------------------------


def _timeit(f, *args, iters=10, warmup=2):
    for _ in range(warmup):
        jax.block_until_ready(f(*args))
    t0 = time.perf_counter()
    for _ in range(iters):
        r = f(*args)
    jax.block_until_ready(r)
    return (time.perf_counter() - t0) / iters


@tpu_only
def test_tpu_expert_stream_real_shapes_and_timing():
    layers = 8
    cfg = real_cfg(experts=64, layers=layers)
    weights = random_expert_weights(cfg, jax.random.PRNGKey(42))
    rng = np.random.default_rng(0)
    print()
    for batch, distinct in [(1, None), (8, 8), (8, 32), (8, 64), (8, None), (4, None)]:
        idx, w = random_routes(cfg, batch, rng, distinct)
        check_stream(cfg, batch, layers, weights, idx, w, rng, verbose=True)
    # Timing: layers looped in-kernel, per-layer time per core.
    Hm = cfg.moe_hidden
    for batch, distinct in [(1, None), (8, 8), (8, 32), (8, 64)]:
        idx, w = random_routes(cfg, batch, rng, distinct)
        it, wt = route_tiles(cfg, idx, w)
        h1 = jnp.asarray(r16(rng.standard_normal((batch, Hm)).astype(np.float32)))
        post = jnp.ones((1, Hm), jnp.float32)
        n = len(np.unique(idx))
        mib = n * (Hm * 2 * 512 + 512 * Hm) / 2 / 2**20
        line = f"  B={batch} distinct={n:2d}:"
        gap = gap_time((batch, Hm), GAP_ITERS, layers) / layers
        for early in (False, True):
            call = make_moe_call(cfg, batch, layers, early_start=early, gap_iters=GAP_ITERS * early)
            t = _timeit(call, it, wt, h1, post, *vars(weights).values()) / layers - gap * early
            y, _ = call(it, wt, h1, post, *vars(weights).values())
            Y_ref = ref_partial_outputs(cfg, layers - 1, np.asarray(h1), idx, weights)
            assert np.abs(np.asarray(y) - Y_ref).max() / np.abs(Y_ref).max() <= 2e-3
            tag = f"early start + {gap * 1e6:.0f} us gap" if early else "cold start"
            line += f"  {tag} {t * 1e6:6.1f} us/layer ({mib * 2**20 / t / 1e9:5.0f} GB/s)"
        print(line)
