"""CPU tests of the XLA prefill (`musespark.prefill`) against the pure-JAX reference.

Needs eight host devices: ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8``.
MINI config, random int4-quantized canonical weights sharded with `shard_canonical`, so the
prefill and the reference see exactly the same weights.

Test weights: `random_canonical_weights` with two changes that keep the comparison meaningful.
The residual gates are damped (`beta ~ 0.27`) because with the default random gates the network
is chaotic (a one-ulp bf16 perturbation grows ~2.5x per layer in the reference itself, so TP
summation-order noise alone reaches O(1) after four layers), and in the `top4` variant the
routing is made decisive through the selection bias (four experts per layer get +2) because the
flat random router flips near-tie top-k selections under one-ulp noise. The `all_experts`
variant (`top_k = experts`) has no selection at all and populates every ragged group. Real bugs
(layout, RoPE, masks, sharding, grouping, cache writes) produce O(0.3) residual errors at the
layer where they occur; bf16 noise stays around 1e-2.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import musespark
import musespark.prefill as prefill_mod
from musespark import MINI, Config
from musespark.load import zero_caches

TP = 8
CONTEXT = 512
T = 37  # real prompt tokens (padded to 64 by the prefill)
CONFIGS = {"top4": MINI, "all_experts": Config(**{**MINI.__dict__, "top_k": MINI.experts})}
LOGIT_TOL = 5e-2  # design.md section 7: max |softcapped logit diff|
RESIDUAL_TOL = 0.1  # f32 residual stream (unit RMS): a few bf16 ulps of TP summation-order noise
CACHE_TOL = 0.1  # bf16 K/V (values up to ~4): one or two ulps


def _forward(cfg, canonical, tokens, hidden=False):
    """Reference prompt from position 0: (logits [T, V], caches, hidden [L+1, T, H] or None)."""
    caches = musespark.init_caches(cfg, 1, CONTEXT)
    run = jax.jit(
        lambda w, t, c: musespark.forward(
            cfg, w, t, jnp.zeros(1, jnp.int32), c, return_hidden=hidden
        )
    )
    out = run(musespark.to_device(canonical), jnp.asarray(tokens)[None], caches)
    return np.asarray(out[0][0]), out[1], (np.asarray(out[2])[:, 0] if hidden else None)


def _cache_diff(a, b, layers):
    """Per-layer max |diff| of two canonical `(k, v)` caches over the prompt slots."""
    return tuple(
        np.abs(np.asarray(x).astype(np.float32) - np.asarray(y).astype(np.float32))[:, 0, :T]
        .reshape(layers, -1)
        .max(axis=1)
        for x, y in zip(a, b)
    )


@pytest.fixture(scope="module")
def mesh():
    devices = jax.devices()
    if len(devices) < TP:
        pytest.skip("needs XLA_FLAGS=--xla_force_host_platform_device_count=8")
    return Mesh(np.array(devices[:TP]), ("tp",))


def _make_weights(cfg, decisive_routing, key=1):
    """Random canonical weights with damped residual gates (and decisive routing biases)."""
    canonical = musespark.random_canonical_weights(cfg, key=key, quantized=True)
    rng = np.random.default_rng(key + 100)
    for name in ("attn_gate", "ffn_gate"):
        g = (-0.3 + 0.05 * rng.standard_normal((cfg.layers, cfg.hidden))).astype(np.float32)
        alpha, beta = musespark.gate_coeffs(jnp.asarray(g), cfg.gate_temperature)
        canonical[name + "_alpha"], canonical[name + "_beta"] = np.asarray(alpha), np.asarray(beta)
    if decisive_routing:
        bias = np.asarray(canonical["router_bias"], np.float32).copy()
        for layer in range(cfg.layers):
            chosen = (layer * 5 + np.arange(cfg.top_k) * (cfg.experts // cfg.top_k)) % cfg.experts
            bias[layer, chosen] += 2.0  # sigmoid scores are in (0, 1): always selected
        canonical["router_bias"] = bias
    return canonical


@pytest.fixture(scope="module", params=sorted(CONFIGS))
def case(request, mesh):
    """(name, cfg, canonical numpy weights, sharded device weights, tokens, reference outputs)."""
    cfg = CONFIGS[request.param]
    canonical = _make_weights(cfg, decisive_routing=request.param == "top4")
    sharded = musespark.shard_canonical(cfg, canonical, TP)
    sharding = NamedSharding(mesh, P("tp"))
    weights = {
        n: jax.device_put(jnp.asarray(musespark.to_device(a)), sharding) for n, a in sharded.items()
    }
    tokens = np.random.default_rng(0).integers(0, cfg.vocab_used, T).astype(np.int32)
    reference = _forward(cfg, canonical, tokens, hidden=True)
    return request.param, cfg, canonical, weights, tokens, reference


def _run_prefill(mesh, cfg, weights, tokens, row=0, batch=2, taps=False):
    prefill = prefill_mod.make_prefill(mesh, cfg, CONTEXT, TP, taps=taps)
    caches = zero_caches(mesh, cfg, batch, CONTEXT)
    padded, length = prefill_mod.pad_prompt(cfg, tokens)
    assert padded.shape == (64,) and length == T
    started = time.perf_counter()
    out = prefill(weights, caches, padded, length, row)
    jax.block_until_ready(out)
    print(f"prefill(T={T}) first call (compile + run): {time.perf_counter() - started:.1f} s")
    return out


def test_prefill_logits_and_residuals_match_reference(mesh, case):
    name, cfg, _, weights, tokens, (ref_logits, _, ref_hidden) = case
    logits, _, hidden = _run_prefill(mesh, cfg, weights, tokens, taps=True)
    logits, hidden = np.asarray(logits), np.asarray(hidden)[:, :T]
    assert logits.shape == (cfg.vocab,) and hidden.shape == ref_hidden.shape
    assert np.all(np.abs(logits) <= cfg.softcap)
    per_layer = np.abs(hidden - ref_hidden).reshape(cfg.layers + 1, -1).max(axis=1)
    print(
        f"[{name}] residual max |diff| per layer {np.round(per_layer, 3)}, "
        f"exact after layer 0: {(hidden[1] == ref_hidden[1]).mean():.3f}"
    )
    assert per_layer[0] == 0  # embedding + embed norm: bit-exact
    assert np.all(per_layer <= RESIDUAL_TOL)  # bf16 summation-order noise only
    diff = np.abs(logits - ref_logits[T - 1]).max()
    print(f"[{name}] max |logit diff| at the last position: {diff:.4f}")
    assert diff <= LOGIT_TOL
    assert int(np.argmax(logits)) == int(np.argmax(ref_logits[T - 1]))


def test_prefill_writes_the_kernel_cache_layout(mesh, case):
    name, cfg, _, weights, tokens, (_, ref_caches, _) = case
    row = 1
    _, caches = _run_prefill(mesh, cfg, weights, tokens, row=row, batch=2)
    got = musespark.unshard_caches(cfg, {k: np.asarray(v) for k, v in caches.items()}, TP)
    lanes = caches["k_cache"].shape[-1]
    kvw = (cfg.kv_heads // TP) * cfg.head_dim
    assert caches["k_cache"].shape == (TP, cfg.layers, 2, CONTEXT, lanes)
    mine = tuple(c[:, row : row + 1] for c in got)  # row `row` as a batch-1 canonical cache
    diffs = _cache_diff(mine, ref_caches, cfg.layers)
    for kind, per_layer, cache in zip("kv", diffs, got):
        print(f"[{name}] {kind}-cache max |diff| per layer {np.round(per_layer, 3)}")
        assert per_layer[0] == 0  # layer 0: bit-exact (post-norm, post-RoPE keys / raw values)
        assert np.all(per_layer <= CACHE_TOL)
        assert not np.any(cache[:, row, T:])  # pad slots and the untouched row stay zero
        assert not np.any(cache[:, 1 - row])
    if lanes > kvw:  # MINI: lanes 64:128 are padding and stay zero
        assert not np.any(np.asarray(caches["k_cache"])[..., kvw:])


def test_prefill_then_reference_decode_matches_sequential_reference(mesh, case):
    name, cfg, canonical, weights, tokens, (_, ref_caches, _) = case
    steps = 4
    extra = np.random.default_rng(3).integers(0, cfg.vocab_used, steps).astype(np.int32)
    canonical_dev = musespark.to_device(canonical)
    step = jax.jit(lambda w, t, p, c: musespark.decode_step(cfg, w, t, p, c))

    def continue_from(caches):
        out = []
        for i in range(steps):
            token, pos = jnp.asarray(extra[i : i + 1]), jnp.asarray([T + i], jnp.int32)
            logits, caches = step(canonical_dev, token, pos, caches)
            out.append(np.asarray(logits[0]))
        return out

    ref_out = continue_from(ref_caches)
    _, caches = _run_prefill(mesh, cfg, weights, tokens, row=0, batch=1)
    mine = musespark.unshard_caches(cfg, {k: np.asarray(v) for k, v in caches.items()}, TP)
    got_out = continue_from(tuple(jnp.asarray(c) for c in mine))
    for i in range(steps):
        diff = np.abs(got_out[i] - ref_out[i]).max()
        print(f"[{name}] decode step {i}: max |logit diff| {diff:.4f}")
        assert diff <= LOGIT_TOL
        assert int(np.argmax(got_out[i])) == int(np.argmax(ref_out[i]))


def test_prefill_length_bucket():
    assert prefill_mod.prefill_length_bucket(1) == 64
    assert prefill_mod.prefill_length_bucket(64) == 64
    assert prefill_mod.prefill_length_bucket(65) == 128
    with pytest.raises(ValueError):
        prefill_mod.prefill_length_bucket(0)
    tokens, length = prefill_mod.pad_prompt(MINI, [5, 6, 7])
    assert length == 3 and tokens.shape == (64,) and int(tokens[3]) == MINI.pad
