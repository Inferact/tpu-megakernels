"""Logit post-processing and token selection for Muse Spark (XLA glue of design.md section 4).

The kernel emits vocabulary-sharded raw lm_head outputs `[tp, B, Vp]` (rank `r` holds ids
`r * Vp .. (r + 1) * Vp - 1`, the last shard zero-padded beyond `V`). This module turns them
into tokens:

* `softcap_logits(shard, cfg)`: `softcap * tanh(shard * output_multiplier / softcap)`.
* `valid_ids(cfg, tp, rank, width)`: which columns of a shard are real, sampleable ids
  (`< cfg.vocab_used`; the untrained rows `>= vocab_used` and the pad columns are never chosen).
* `greedy_local(shard, rank, cfg, tp)`: inside a `shard_map` body, the batch's greedy tokens
  from per-rank `(max, argmax)` pairs and one all-gather (ties -> lowest id). Softcapping is
  monotonic, so the raw shard can be used.
* `make_sharded_greedy(mesh, cfg)` / `make_gather_logits(mesh, cfg)`: jitted programs over
  `[tp, B, Vp]` shards -> greedy tokens `[B]` / softcapped, masked full logits `[B, V]`.
* `sample(logits, key, temperature, top_k, top_p)`: temperature -> top-k -> top-p nucleus
  sampling on full `[B, V]` logits (Muse Spark defaults: temperature 1.0, top_k 64, top_p 1.0);
  deterministic given `key`, and greedy at temperature 0.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

from musespark import layout
from musespark.config import Config

F32 = jnp.float32
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 64
DEFAULT_TOP_P = 1.0


def softcap_logits(shard, cfg: Config):
    """`softcap * tanh(x * output_multiplier / softcap)` on raw lm_head outputs (any shape)."""
    return cfg.softcap * jnp.tanh(jnp.asarray(shard, F32) * (cfg.output_multiplier / cfg.softcap))


def valid_ids(cfg: Config, tp, rank, width):
    """Bool `[width]`: column `c` of rank `rank` is a real id (`rank * Vp + c < vocab_used`)."""
    return rank * layout.vocab_pad(cfg, tp) + jnp.arange(width) < cfg.vocab_used


def greedy_local(shard, rank, cfg: Config, tp, axis="tp"):
    """Greedy tokens `[B]` from this rank's logits shard `[B, Vp]` (call inside a shard_map).

    Every rank takes its masked `(max, argmax)`; the pairs are all-gathered and the winner is
    the first rank holding the global maximum, so ties resolve to the lowest id. The result is
    identical on every rank.
    """
    width = shard.shape[-1]
    shard = jnp.where(valid_ids(cfg, tp, rank, width), jnp.asarray(shard, F32), -jnp.inf)
    local_id = jnp.argmax(shard, axis=-1).astype(jnp.int32)  # first maximum
    local_max = jnp.take_along_axis(shard, local_id[:, None], axis=-1)[:, 0]
    maxima, ids = lax.all_gather(
        (local_max, local_id + rank * layout.vocab_pad(cfg, tp)), axis, axis=0
    )  # [tp, B]
    winner = jnp.argmax(maxima, axis=0)  # first (lowest) rank holding the maximum
    return jnp.take_along_axis(ids, winner[None], axis=0)[0]


def make_sharded_greedy(mesh, cfg: Config):
    """Jitted `(shards [tp, B, Vp] f32) -> tokens [B] int32` (replicated)."""
    tp = mesh.size

    def local(shards):
        return greedy_local(shards[0], lax.axis_index("tp"), cfg, tp)

    return jax.jit(
        jax.shard_map(local, mesh=mesh, in_specs=(P("tp"),), out_specs=P(), check_vma=False)
    )


def gather_logits_local(shard, rank, cfg: Config, tp, axis="tp"):
    """Inside a shard_map: `[B, Vp]` raw shard -> softcapped, masked full logits `[B, V]`."""
    full = lax.all_gather(jnp.asarray(shard, F32), axis, axis=1, tiled=True)[:, : cfg.vocab]
    return jnp.where(jnp.arange(cfg.vocab) < cfg.vocab_used, softcap_logits(full, cfg), -jnp.inf)


def make_gather_logits(mesh, cfg: Config):
    """Jitted `(shards [tp, B, Vp]) -> logits [B, V] f32`: softcapped, unused ids -inf."""
    tp = mesh.size

    def local(shards):
        return gather_logits_local(shards[0], lax.axis_index("tp"), cfg, tp)

    return jax.jit(
        jax.shard_map(local, mesh=mesh, in_specs=(P("tp"),), out_specs=P(), check_vma=False)
    )


def mask_unused(logits, cfg: Config):
    """-inf on ids `>= cfg.vocab_used` of full `[..., V]` logits."""
    return jnp.where(jnp.arange(logits.shape[-1]) < cfg.vocab_used, logits, -jnp.inf)


@partial(jax.jit, static_argnames=("top_k",))
def sample(logits, key, temperature=DEFAULT_TEMPERATURE, top_k=DEFAULT_TOP_K, top_p=DEFAULT_TOP_P):
    """Tokens `[B]` int32 from full logits `[B, V]` (already softcapped and masked).

    `logits / temperature` -> the `top_k` largest (ties -> lowest id; `top_k` static, `None` or
    `0` = the whole vocabulary) -> top-p nucleus (the smallest prefix of the descending top-k
    whose probability mass reaches `top_p`, at least one token) -> one categorical draw per row
    from the renormalised nucleus. `temperature` and `top_p` may be traced. `temperature == 0`
    (checked as a traced value) selects the argmax; otherwise the draw is `jax.random.categorical`
    on `key`, so the result is a deterministic function of `(logits, key, params)`.
    """
    logits = jnp.asarray(logits, F32)
    vocab = logits.shape[-1]
    k = vocab if not top_k else min(int(top_k), vocab)
    temperature = jnp.asarray(temperature, F32)
    top_p = jnp.asarray(top_p, F32)
    # temperature 0 selects the argmax below; dividing by `tiny` there would saturate to inf
    # and break the top-k tie order, so the scaling is bypassed instead.
    scaled = jnp.where(temperature > 0, logits / jnp.maximum(temperature, jnp.finfo(F32).tiny), logits)
    values, ids = lax.top_k(scaled, k)  # descending, ties -> lower index
    probs = jax.nn.softmax(values, axis=-1)
    cumulative = jnp.cumsum(probs, axis=-1)
    keep = (cumulative - probs) < top_p  # the first candidate is always kept
    keep = keep & jnp.isfinite(values)
    choice = jax.random.categorical(key, jnp.where(keep, values, -jnp.inf), axis=-1)
    choice = jnp.where(temperature > 0, choice, 0)  # greedy: the largest candidate
    return jnp.take_along_axis(ids, choice[:, None], axis=-1)[:, 0].astype(jnp.int32)
