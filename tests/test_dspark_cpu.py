"""TP32 DSpark programs compared with the dense JAX reference on CPU."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from kimi import dspark


SMALL = dspark.DraftConfig(
    hidden=32,
    layers=2,
    heads=32,
    kv_heads=8,
    head_dim=8,
    intermediate=32,
    vocab=64,
    aux_layers=(1, 2),
    mask_token=61,
    window=16,
    markov_rank=4,
    markov_top_m=2,
)


def _cpu_mesh():
    devices = jax.devices("cpu")
    if len(devices) < 32:
        pytest.skip("requires XLA_FLAGS=--xla_force_host_platform_device_count=32")
    return Mesh(np.asarray(devices[:32]), ("tp",))


@pytest.mark.cpu32
def test_sharded_context_and_draft_step_match_dense_reference():
    mesh = _cpu_mesh()
    config = SMALL
    weights = dspark.random_weights(config, seed=5)
    rng = np.random.default_rng(7)
    context_tokens, block_size = 5, 3
    aux = jnp.asarray(
        rng.normal(size=(context_tokens, config.context_width)), jnp.float32
    ).astype(jnp.bfloat16)
    positions = jnp.arange(context_tokens, dtype=jnp.int32)
    anchor = jnp.int32(rng.integers(0, config.vocab - 4))
    anchor_position = jnp.int32(context_tokens)

    context = dspark.reference_context_features(weights, config, aux)
    reference_keys, reference_values = dspark.reference_context_kv(
        weights, config, context, positions
    )
    hidden = dspark.reference_block_hidden(
        weights,
        config,
        reference_keys,
        reference_values,
        positions,
        anchor,
        anchor_position,
        block_size,
    )
    expected = dspark.reference_markov_greedy(weights, config, hidden, anchor)

    sharded_weights = dspark.shard_weights_to_mesh(config, weights, mesh)
    cache = dspark.empty_cache(config, mesh)
    update = dspark.make_context_update(config, mesh, context_tokens)
    aux_by_layer = aux.reshape(
        context_tokens, len(config.aux_layers), config.hidden
    ).transpose(1, 0, 2)
    keys, values, cache_positions = update(
        sharded_weights, cache, cache[2], aux_by_layer, positions
    )

    kv_rank_sharing = 32 // config.kv_heads
    for rank in (0, 5, 31):
        head = rank // kv_rank_sharing
        np.testing.assert_allclose(
            np.asarray(keys[rank, :, :context_tokens], dtype=np.float32),
            np.asarray(reference_keys[:, :, head], dtype=np.float32),
            rtol=1e-2,
            atol=1e-2,
        )
        np.testing.assert_allclose(
            np.asarray(values[rank, :, :context_tokens], dtype=np.float32),
            np.asarray(reference_values[:, :, head], dtype=np.float32),
            rtol=1e-2,
            atol=1e-2,
        )
    np.testing.assert_array_equal(cache_positions[:context_tokens], positions)

    draft_step = dspark.make_draft_step(config, mesh, block_size)
    actual = draft_step(
        sharded_weights,
        (keys, values, cache_positions),
        cache_positions,
        anchor,
        anchor_position,
    )
    np.testing.assert_array_equal(actual, expected)
