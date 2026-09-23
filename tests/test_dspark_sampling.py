"""CPU tests of the demo's speculative top-p sampling (``kimi.dspark.nucleus_accept``
and the vocabulary-sharded ``make_row_sampler``).

The mesh test needs ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=32``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kimi import dspark as draft

ROWS = 8


def _nucleus(logits, top_p):
    """Reference top-p distribution per row from dense logits (numpy)."""
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    out = np.zeros_like(probs)
    for r in range(probs.shape[0]):
        order = np.argsort(-probs[r], kind="stable")
        cumulative = np.cumsum(probs[r, order])
        keep = np.concatenate(([True], cumulative[:-1] < top_p))
        kept = np.where(keep, probs[r, order], 0.0)
        out[r, order] = kept / kept.sum()
    return out


def _dense_inputs(seed, vocab=40, scale=3.0):
    rng = np.random.default_rng(seed)
    logits = (scale * rng.standard_normal((ROWS, vocab))).astype(np.float32)
    lse = np.log(np.exp(logits).sum(axis=1)).astype(np.float32)
    ids = np.broadcast_to(np.arange(vocab, dtype=np.int32), (ROWS, vocab)).copy()
    return logits, ids, lse


def _batched(logits, ids, lse, drafts, forced_count, top_p, uniforms):
    run = jax.vmap(lambda u: draft.nucleus_accept(logits, ids, lse, drafts, forced_count, top_p, u))
    count, bonus = jax.jit(run)(uniforms)
    return np.asarray(count), np.asarray(bonus)


def _hist(values, vocab):
    return np.bincount(values, minlength=vocab) / len(values)


def test_forced_count_bonus_follows_the_nucleus():
    logits, ids, lse = _dense_inputs(0)
    top_p = 0.8
    expected = _nucleus(logits, top_p)
    uniforms = np.random.default_rng(1).random((40000, ROWS, 2), np.float32)
    drafts = np.full((ROWS,), -1, np.int32)
    for forced in (0, 3, 7):
        count, bonus = _batched(logits, ids, lse, drafts, jnp.int32(forced), top_p, uniforms)
        assert (count == forced).all()
        observed = _hist(bonus, logits.shape[1])
        assert observed[expected[forced] == 0].sum() == 0  # never outside the nucleus
        assert np.abs(observed - expected[forced]).max() < 0.012


def _check_positions(logits, drafts, top_p, uniforms):
    """Runs the sampler over ``uniforms`` and checks the acceptance rule and, per position k, that the
    output token (draft k when accepted, else the bonus) is distributed as the nucleus p'_k given that
    the previous drafts were accepted."""
    ids = np.broadcast_to(np.arange(logits.shape[1], dtype=np.int32), logits.shape).copy()
    lse = np.log(np.exp(logits - logits.max(axis=1, keepdims=True)).sum(axis=1)) + logits.max(axis=1)
    expected = _nucleus(logits, top_p)
    count, bonus = _batched(logits, ids, lse.astype(np.float32), drafts, jnp.int32(-1), top_p, uniforms)
    draft_prob = np.array([expected[k, drafts[k]] if drafts[k] >= 0 else 0.0 for k in range(ROWS - 1)])
    accepted = uniforms[:, : ROWS - 1, 0] < draft_prob[None]
    first_reject = np.where(accepted.all(axis=1), ROWS - 1, np.argmin(accepted, axis=1))
    assert (count == first_reject).all()
    rejected = (count < ROWS - 1) & (drafts[np.minimum(count, ROWS - 2)] >= 0)
    assert not (bonus[rejected] == drafts[count[rejected]]).any()  # the rejected draft is never re-drawn
    checked = 0
    for k in range(ROWS):
        rows = count >= k
        if rows.sum() < 3000:
            break
        token = np.where(count[rows] > k, drafts[min(k, ROWS - 2)], bonus[rows])
        observed = _hist(token, logits.shape[1])
        assert observed[expected[k] == 0].sum() == 0
        tolerance = 0.015 + 3.0 / np.sqrt(rows.sum())
        assert np.abs(observed - expected[k]).max() < tolerance, (k, np.abs(observed - expected[k]).max(), tolerance)
        checked += 1
    return checked


def test_speculative_output_is_distributed_as_the_nucleus():
    logits = _dense_inputs(2)[0]
    uniforms = np.random.default_rng(3).random((60000, ROWS, 2), np.float32)
    # Good and poorer drafts for rows 0..2 (argmax, second-best, argmax), none afterwards: the count
    # stops at 3 at the latest, so the first positions have plenty of samples.
    drafts = np.full((ROWS,), -1, np.int32)
    drafts[0], drafts[1], drafts[2] = np.argmax(logits[0]), np.argsort(-logits[1])[1], np.argmax(logits[2])
    assert _check_positions(logits, drafts, 0.9, uniforms) >= 3
    # Greedy drafts on every row of a peaked distribution: the chain reaches the later rows often.
    logits = _dense_inputs(7, scale=5.0)[0]
    drafts = np.concatenate((np.argmax(logits, axis=1)[:-1], [-1])).astype(np.int32)
    assert _check_positions(logits, drafts, 0.95, uniforms) >= 5


def test_full_nucleus_and_near_greedy_limits():
    logits, ids, lse = _dense_inputs(4)
    uniforms = np.random.default_rng(5).random((20000, ROWS, 2), np.float32)
    drafts = np.full((ROWS,), -1, np.int32)
    # top_p = 1: the full distribution.
    _, bonus = _batched(logits, ids, lse, drafts, jnp.int32(2), 1.0, uniforms)
    expected = _nucleus(logits, 1.0)[2]
    assert np.abs(_hist(bonus, logits.shape[1]) - expected).max() < 0.015
    # A tiny temperature makes the nucleus (and the sample) the argmax; the greedy drafts are always accepted.
    sharp = logits * 1000.0
    sharp_lse = np.log(np.exp(sharp - sharp.max(axis=1, keepdims=True)).sum(axis=1)) + sharp.max(axis=1)
    greedy = np.argmax(logits, axis=1).astype(np.int32)
    greedy_drafts = np.concatenate((greedy[:-1], [-1])).astype(np.int32)  # draft k is checked against row k
    count, bonus = _batched(sharp, ids, sharp_lse.astype(np.float32), greedy_drafts, jnp.int32(-1), 0.95, uniforms[:200])
    assert (count == ROWS - 1).all() and (bonus == greedy[ROWS - 1]).all()


def test_row_sampler_matches_dense_nucleus_accept():
    devices = jax.devices()
    if len(devices) < 32:
        pytest.skip("needs XLA_FLAGS=--xla_force_host_platform_device_count=32")
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    mesh = Mesh(np.array(devices[:32]), ("tp",))
    ranks, per_rank = 32, 8
    vocab = ranks * per_rank
    rng = np.random.default_rng(6)
    logits = (2.0 * rng.standard_normal((ROWS, vocab))).astype(np.float32)
    temperature, top_p = 0.7, 0.85
    dense = logits / temperature
    lse = np.log(np.exp(dense - dense.max(axis=1, keepdims=True)).sum(axis=1)) + dense.max(axis=1)
    ids = np.broadcast_to(np.arange(vocab, dtype=np.int32), (ROWS, vocab)).copy()
    drafts = np.concatenate((rng.integers(0, vocab, ROWS - 1), [-1])).astype(np.int32)
    sampler = draft.make_row_sampler(mesh, ROWS, vocab, candidates=per_rank)  # whole shards: exact
    # Rank r holds vocabulary [r * per_rank, (r + 1) * per_rank) of every row, as the verify emits it.
    sharded_logits = jax.device_put(logits.reshape(ROWS, ranks, per_rank).transpose(1, 0, 2), NamedSharding(mesh, P("tp")))
    for trial in range(5):
        uniforms = rng.random((ROWS, 2), np.float32)
        for forced in (-1, 4):
            count, bonus = sampler(sharded_logits, jnp.asarray(drafts), jnp.int32(forced), jnp.float32(temperature),
                                   jnp.float32(top_p), jnp.asarray(uniforms))
            ref_count, ref_bonus = draft.nucleus_accept(
                jnp.asarray(dense), jnp.asarray(ids), jnp.asarray(lse.astype(np.float32)), jnp.asarray(drafts),
                jnp.int32(forced), jnp.float32(top_p), jnp.asarray(uniforms),
            )
            assert int(count) == int(ref_count) and int(bonus) == int(ref_bonus), (trial, forced)
