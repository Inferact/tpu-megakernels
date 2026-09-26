"""CPU tests of `musespark.sampling`: softcap, sharded greedy and top-k / top-p sampling.

The sharded tests need ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from musespark import MINI, Config, layout, sampling

ROWS = 4


def _nucleus(logits, temperature, top_k, top_p):
    """Reference top-k / top-p distribution per row (numpy)."""
    z = logits / temperature
    out = np.zeros_like(z)
    for r in range(z.shape[0]):
        order = np.argsort(-z[r], kind="stable")[:top_k]
        p = np.exp(z[r, order] - z[r, order].max())
        p /= p.sum()
        cumulative = np.cumsum(p)
        keep = np.concatenate(([True], cumulative[:-1] < top_p))
        kept = np.where(keep, p, 0.0)
        out[r, order] = kept / kept.sum()
    return out


def _hist(tokens, vocab):
    return np.bincount(tokens, minlength=vocab) / len(tokens)


def _draws(logits, n, seed=0, **params):
    keys = jax.random.split(jax.random.key(seed), n)
    run = jax.vmap(lambda key: sampling.sample(jnp.asarray(logits), key, **params))
    return np.asarray(run(keys))  # [n, rows]


def test_softcap_matches_spec():
    cfg = Config()
    raw = np.linspace(-400, 400, 9, dtype=np.float32)
    got = np.asarray(sampling.softcap_logits(raw, cfg))
    want = 20.0 * np.tanh(raw * 0.17677669529663687 / 20.0)
    assert np.allclose(got, want, atol=1e-6)
    assert np.all(np.abs(got) <= 20.0)


def test_sample_follows_the_top_k_top_p_distribution():
    vocab, n = 40, 30000
    logits = (3.0 * np.random.default_rng(0).standard_normal((ROWS, vocab))).astype(np.float32)
    for temperature, top_k, top_p in ((1.0, 64, 1.0), (0.7, 8, 0.85), (1.3, 5, 0.5)):
        draws = _draws(logits, n, temperature=temperature, top_k=top_k, top_p=top_p)
        expected = _nucleus(logits, temperature, top_k, top_p)
        for r in range(ROWS):
            observed = _hist(draws[:, r], vocab)
            assert observed[expected[r] == 0].sum() == 0  # never outside the nucleus
            assert np.abs(observed - expected[r]).max() < 0.012, (temperature, top_k, top_p, r)


def test_sample_is_deterministic_and_greedy_at_zero_temperature():
    vocab = 50
    logits = np.random.default_rng(1).standard_normal((ROWS, vocab)).astype(np.float32)
    key = jax.random.key(7)
    a = np.asarray(sampling.sample(jnp.asarray(logits), key))
    b = np.asarray(sampling.sample(jnp.asarray(logits), key))
    assert np.array_equal(a, b)
    greedy = np.asarray(sampling.sample(jnp.asarray(logits), key, temperature=0.0))
    assert np.array_equal(greedy, np.argmax(logits, axis=1))
    # top_k=1 is greedy too, whatever the temperature
    assert np.array_equal(
        np.asarray(sampling.sample(jnp.asarray(logits), key, top_k=1)), np.argmax(logits, axis=1)
    )


def test_sample_never_picks_masked_ids():
    cfg = MINI
    vocab = cfg.vocab
    logits = np.zeros((ROWS, vocab), np.float32)
    logits[:, cfg.vocab_used - 1] = 5.0  # the last real id dominates ...
    masked = sampling.mask_unused(
        jnp.asarray(logits), Config(**{**cfg.__dict__, "vocab_used": cfg.vocab_used - 1})
    )
    draws = _draws(np.asarray(masked), 2000, top_k=0)  # ... but it is masked: full-vocab sampling
    assert np.all(draws < cfg.vocab_used - 1)


def _mesh():
    devices = jax.devices()
    if len(devices) < 8:
        pytest.skip("needs XLA_FLAGS=--xla_force_host_platform_device_count=8")
    return Mesh(np.array(devices[:8]), ("tp",))


def _shards(mesh, cfg, logits):
    """Full `[B, V]` raw logits -> the kernel's `[tp, B, Vp]` shards (zero-padded)."""
    tp = mesh.size
    vp = layout.vocab_pad(cfg, tp)
    padded = np.zeros((logits.shape[0], tp * vp), np.float32)
    padded[:, : cfg.vocab] = logits
    shards = padded.reshape(logits.shape[0], tp, vp).transpose(1, 0, 2)
    return jax.device_put(shards, NamedSharding(mesh, P("tp")))


def test_sharded_greedy_matches_masked_argmax_with_ties():
    mesh = _mesh()
    cfg = Config(**{**MINI.__dict__, "vocab": 2048, "vocab_used": 1900})
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((8, cfg.vocab)).astype(np.float32)
    logits[0, 1950] = 100.0  # an untrained id must never win
    logits[1, [5, 700, 1500]] = 50.0  # ties across ranks -> lowest id
    logits[2, [300, 301]] = 50.0  # ties inside a rank -> lowest id
    greedy = sampling.make_sharded_greedy(mesh, cfg)
    got = np.asarray(greedy(_shards(mesh, cfg, logits)))
    want = np.argmax(np.where(np.arange(cfg.vocab) < cfg.vocab_used, logits, -np.inf), axis=1)
    assert np.array_equal(got, want)
    assert got[0] != 1950 and got[1] == 5 and got[2] == 300


def test_gather_logits_softcaps_and_masks():
    mesh = _mesh()
    cfg = Config(**{**MINI.__dict__, "vocab": 2048, "vocab_used": 2000})
    logits = (30.0 * np.random.default_rng(3).standard_normal((2, cfg.vocab))).astype(np.float32)
    gather = sampling.make_gather_logits(mesh, cfg)
    got = np.asarray(gather(_shards(mesh, cfg, logits)))
    assert got.shape == (2, cfg.vocab)
    want = cfg.softcap * np.tanh(logits * cfg.output_multiplier / cfg.softcap)
    assert np.allclose(got[:, : cfg.vocab_used], want[:, : cfg.vocab_used], atol=1e-5)
    assert np.all(np.isneginf(got[:, cfg.vocab_used :]))
    # end to end: greedy over the gathered logits == the sharded greedy
    greedy = sampling.make_sharded_greedy(mesh, cfg)
    assert np.array_equal(np.asarray(greedy(_shards(mesh, cfg, logits))), np.argmax(got, axis=1))
