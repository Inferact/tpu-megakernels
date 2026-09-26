"""Tests for the Muse Spark attention mini-kernel (`musespark.attention`).

CPU: the standalone `pallas_call` in interpret mode vs the pure-jnp reference on the MINI config
and on the real per-rank config (16 q heads / 2 kv heads). TPU (chip 0, skipped otherwise): the
same checks on hardware plus per-layer timings.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import musespark

# Importing the `musespark.attention` submodule rebinds the package attribute `attention`
# (A1's whole-model reference function) to the module; keep the function under its own name.
MODEL_ATTENTION = musespark.attention_branch
from musespark.attention import (
    make_attention_call,
    reference_attention,
    rope_table,
    rope_table_np,
)
from musespark.config import MINI, Config
from musespark.layout import cache_lanes

TP = 8
REAL = Config()  # per rank: 16 q heads, 2 kv heads, window 2048
REAL_POSITIONS = (0, 5, 127, 128, 2046, 2047, 2048, 2100, 4095)
MINI_POSITIONS = (0, 5, 127, 128, 254, 255, 256, 300, 511)
ON_TPU = jax.devices()[0].platform == "tpu"


def _random_case(cfg, batch, context, layers, seed):
    """Random bf16-valued q/kv/g, random bf16 caches (unused lanes zero)."""
    hq, hkv, D = cfg.heads // TP, cfg.kv_heads // TP, cfg.head_dim
    lanes = cache_lanes(cfg, TP)
    rng = np.random.default_rng(seed)

    def bf(shape, scale=1.0):
        x = rng.standard_normal(shape).astype(np.float32) * scale
        return jnp.asarray(x).astype(jnp.bfloat16).astype(jnp.float32)

    q = bf((batch, hq * D))
    kv = bf((batch, 2 * hkv * D))
    g = bf((batch, hq * D), 2.0)
    cache = rng.standard_normal((2, layers, batch, context, lanes)).astype(np.float32)
    cache[..., hkv * D :] = 0.0
    k_cache = jnp.asarray(cache[0]).astype(jnp.bfloat16)
    v_cache = jnp.asarray(cache[1]).astype(jnp.bfloat16)
    return q, kv, g, k_cache, v_cache


def _check(cfg, batch, context, layer, positions, seed, interpret, layers=2):
    """Run kernel and reference on one case; return (max |o err|, k slots exact, v slots exact)."""
    q, kv, g, k_cache, v_cache = _random_case(cfg, batch, context, layers, seed)
    pos = jnp.asarray(np.resize(np.asarray(positions, np.int32), batch))
    rope = jnp.asarray(rope_table_np(cfg, np.asarray(pos)))
    o_ref, k_ref, v_ref = reference_attention(
        cfg, layer, q, kv, g, k_cache, v_cache, pos, rope=rope, tp=TP
    )
    call = make_attention_call(cfg, batch, context, tp=TP, interpret=interpret)
    o, k_out, v_out = call(jnp.asarray([layer], jnp.int32), pos, q, kv, g, rope, k_cache, v_cache)
    o, o_ref = np.asarray(o), np.asarray(o_ref)
    assert not np.isnan(o).any()
    err = float(np.abs(o - o_ref).max())
    # The written slots (and nothing else) must match the reference caches exactly.
    k_same = np.array_equal(np.asarray(k_out).view(np.uint16), np.asarray(k_ref).view(np.uint16))
    v_same = np.array_equal(np.asarray(v_out).view(np.uint16), np.asarray(v_ref).view(np.uint16))
    return err, k_same, v_same


def _cases(cfg, positions, batches):
    for layer in (0, 1):  # sliding + RoPE, full + NoPE
        for batch in batches:
            if batch == 1:
                for p in positions:
                    yield layer, batch, (p,)
            else:
                yield layer, batch, positions[:batch]
                yield layer, batch, positions[::-1][:batch]


def _run_matrix(cfg, context, positions, batches, interpret):
    worst = 0.0
    for layer, batch, pos in _cases(cfg, positions, batches):
        err, k_same, v_same = _check(cfg, batch, context, layer, pos, 7 + batch, interpret)
        assert k_same, (layer, batch, pos, "k cache mismatch")
        assert v_same, (layer, batch, pos, "v cache mismatch")
        assert err <= 2e-2, (layer, batch, pos, err)
        worst = max(worst, err)
    return worst


@pytest.mark.skipif(ON_TPU, reason="CPU interpret test")
def test_mini_interpret():
    worst = _run_matrix(MINI, 512, MINI_POSITIONS, (1, 4, 8), interpret=True)
    print(f"MINI interpret max abs err {worst:.3e}")


@pytest.mark.skipif(ON_TPU, reason="CPU interpret test")
def test_real_rank_interpret():
    worst = _run_matrix(REAL, 4096, REAL_POSITIONS, (1, 4, 8), interpret=True)
    print(f"real-per-rank interpret max abs err {worst:.3e}")


@pytest.mark.skipif(ON_TPU, reason="CPU interpret test")
def test_layer_index_arithmetic_interpret():
    # Layers 4 (sliding) and 5 (full) of a 6-layer cache: the kind comes from the index.
    for layer in (4, 5):
        err, k_same, v_same = _check(MINI, 2, 512, layer, (300, 511), 3, True, layers=6)
        assert k_same and v_same and err <= 2e-2


def test_rope_table_matches_float64():
    pos = np.array([0, 1, 5, 2047, 8191, 65535], np.int32)
    table = np.asarray(rope_table(REAL, jnp.asarray(pos)))
    exact = rope_table_np(REAL, pos)
    assert table.shape == (6, 2 * REAL.head_dim)
    # The f32 angle `pos * inv_freq` itself is only resolved to ~pos * 2**-24 rad (any f32
    # implementation, sglang included); on TPU the large-argument cos/sin sit at that limit.
    tol = 1e-3 + 2.0 * pos.astype(np.float64) * 2.0**-24
    assert (np.abs(table - exact).max(axis=1) <= tol).all()


@pytest.mark.skipif(ON_TPU, reason="CPU interpret test")
def test_matches_model_reference_interpret():
    """8 per-rank kernels (design.md section 2 shards) == A1's canonical `musespark.attention`."""
    cfg, batch, context = MINI, 4, 512
    H, nH, nKV, D = cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim
    hq, hkv = nH // TP, nKV // TP
    rng = np.random.default_rng(11)

    def bf(shape, scale=1.0):
        return jnp.asarray(rng.standard_normal(shape).astype(np.float32) * scale).astype(
            jnp.bfloat16
        )

    lw = {
        "q": bf((H, nH * D), H**-0.5),
        "k": bf((H, nKV * D), H**-0.5),
        "v": bf((H, nKV * D), H**-0.5),
        "gate": bf((H, nH * D), 2 * H**-0.5),
        "o": jnp.eye(H, dtype=jnp.bfloat16),  # attn_out == r16(gated attention output)
    }
    x = bf((batch, 1, H)).astype(jnp.float32)
    pos = np.array([0, 300, 511, 255], np.int32)
    k0 = bf((cfg.layers, batch, context, nKV, D))
    v0 = bf((cfg.layers, batch, context, nKV, D))
    q = musespark.mm(x[:, 0], lw["q"])
    k = musespark.mm(x[:, 0], lw["k"])
    v = musespark.mm(x[:, 0], lw["v"])
    g = musespark.mm(x[:, 0], lw["gate"])
    rope = rope_table(cfg, jnp.asarray(pos))  # same jnp cos/sin as A1's reference
    for layer in (0, 1):
        want, k_want, v_want = MODEL_ATTENTION(
            cfg, lw, x, pos[:, None], k0[layer], v0[layer], layer
        )
        sharded = musespark.shard_caches(cfg, (k0, v0), TP)
        k_out = np.asarray(sharded["k_cache"]).copy()
        v_out = np.asarray(sharded["v_cache"]).copy()
        got = np.zeros((batch, H), np.float32)
        call = make_attention_call(cfg, batch, context, tp=TP, interpret=True)
        for r in range(TP):
            qs = q[:, r * hq * D : (r + 1) * hq * D]
            gs = g[:, r * hq * D : (r + 1) * hq * D]
            kvs = jnp.concatenate(
                [k[:, r * hkv * D : (r + 1) * hkv * D], v[:, r * hkv * D : (r + 1) * hkv * D]], -1
            )
            o, kc, vc = call(
                jnp.asarray([layer], jnp.int32),
                jnp.asarray(pos),
                qs,
                kvs,
                gs,
                rope,
                jnp.asarray(k_out[r]),
                jnp.asarray(v_out[r]),
            )
            got[:, r * hq * D : (r + 1) * hq * D] = np.asarray(o)
            k_out[r], v_out[r] = np.asarray(kc), np.asarray(vc)
        err = float(np.abs(got - np.asarray(want)[:, 0]).max())
        assert err <= 2e-2, (layer, err)
        k_got, v_got = musespark.unshard_caches(cfg, {"k_cache": k_out, "v_cache": v_out}, TP)
        assert np.array_equal(k_got[layer].view(np.uint16), np.asarray(k_want).view(np.uint16))
        assert np.array_equal(v_got[layer].view(np.uint16), np.asarray(v_want).view(np.uint16))


@pytest.mark.skipif(not ON_TPU, reason="needs a TPU")
def test_tpu_correctness():
    worst_mini = _run_matrix(MINI, 512, MINI_POSITIONS, (1, 4, 8), interpret=False)
    worst_real = _run_matrix(REAL, 4096, REAL_POSITIONS, (1, 4, 8), interpret=False)
    print(f"TPU max abs err: MINI {worst_mini:.3e}, real-per-rank {worst_real:.3e}")


def _time_layer(cfg, batch, context, layer, positions, reps=100, iters=5):
    q, kv, g, k_cache, v_cache = _random_case(cfg, batch, context, 2, 1)
    pos = jnp.asarray(np.resize(np.asarray(positions, np.int32), batch))
    rope = jnp.asarray(rope_table_np(cfg, np.asarray(pos)))
    layer_v = jnp.asarray([layer], jnp.int32)
    call = make_attention_call(cfg, batch, context, tp=TP, reps=reps)
    o, k_cache, v_cache = call(layer_v, pos, q, kv, g, rope, k_cache, v_cache)
    jax.block_until_ready(o)
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        o, k_cache, v_cache = call(layer_v, pos, q, kv, g, rope, k_cache, v_cache)
        jax.block_until_ready(o)
        best = min(best, (time.perf_counter() - t0) / reps)
    return best * 1e6


@pytest.mark.skipif(not ON_TPU, reason="needs a TPU")
def test_tpu_timing():
    rows = [
        ("B=1 sliding pos 2047 (16 blocks)", 1, 8192, 0, (2047,)),
        ("B=1 full pos 8191 (64 blocks)", 1, 8192, 1, (8191,)),
        ("B=1 sliding pos 8191 (17 blocks)", 1, 8192, 0, (8191,)),
        ("B=8 sliding pos 2047 (16 blocks)", 8, 8192, 0, (2047,)),
        ("B=8 full pos 8191 (64 blocks)", 8, 8192, 1, (8191,)),
        ("B=8 sliding mixed pos", 8, 8192, 0, (0, 5, 127, 2047, 2048, 4095, 8000, 8191)),
        ("B=4 sliding pos 2047", 4, 8192, 0, (2047,)),
    ]
    for name, batch, context, layer, pos in rows:
        us = _time_layer(REAL, batch, context, layer, pos)
        print(f"{name}: {us:.1f} us per layer")
