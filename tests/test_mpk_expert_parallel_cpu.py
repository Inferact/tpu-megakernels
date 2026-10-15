import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

import mpk as model
from mpk import decode_megakernel_32 as megakernel
from mpk import load as mpk_load
from mpk.decode_megakernel import get_tp32_backend
from tests.test_mpk_tp32_cpu import MINI32, MINI32_PACKED, _cpu32, _packed_inputs32


@pytest.mark.cpu32
@pytest.mark.parametrize("config,seed", ((MINI32, 7), (MINI32_PACKED, 11)))
def test_expert_parallel_prefill_matches_oracle(config, seed):
    c, context = config, 256
    base, dense, oracle = _packed_inputs32(c, mpk_load.canonical_weights(c, seed=seed))
    mesh = _cpu32()
    weights = megakernel.prepare_decode_weights(mesh, mpk_load.build_sharded(mesh, {**base, **dense}))
    for key in ("gu_pack", "dw_pack", "gu_dense", "dw_dense"):
        assert weights[key].shape[2] == c.experts // 4
    states = {"kvcache": jax.device_put(jnp.zeros((32, c.layers, 1, context, 128), jnp.bfloat16), NamedSharding(mesh, P("tp")))}
    tokens = [5, 11, 900, 17]
    prefill = megakernel.make_prefill(mesh, c, context, 32)
    states, logits = jax.jit(prefill)(weights, states, jnp.asarray(tokens, jnp.int32), jnp.int32(0))
    expected_states, expected_logits = model.prefill(
        jax.tree.map(jnp.asarray, oracle),
        jax.tree.map(jnp.asarray, mpk_load.zero_states(c, context)), tokens, c,
    )
    np.testing.assert_allclose(
        np.asarray(logits).reshape(32, -1)[:, :c.vocab // 32].reshape(-1),
        np.asarray(expected_logits).reshape(-1), rtol=.05, atol=.04,
    )
    cache = np.asarray(states["kvcache"]).astype(np.float32)
    for layer in range(c.layers):
        for rank in range(32):
            head = rank * c.kv_heads // 32
            for offset, reference in ((0, expected_states[layer][0]), (64, expected_states[layer][1])):
                np.testing.assert_allclose(
                    cache[rank, layer, 0, :len(tokens), offset:offset + 64],
                    np.asarray(reference)[head, :len(tokens)].astype(np.float32), rtol=.1, atol=.1,
                )


def test_tp32_layout_selection(monkeypatch):
    monkeypatch.delenv("MPK_TP32_EXPERT_PARALLEL", raising=False)
    monkeypatch.delenv("MPK_EXPERT_TP", raising=False)
    assert get_tp32_backend().REPLICATED_EXPERTS
    monkeypatch.setenv("MPK_TP32_EXPERT_PARALLEL", "1")
    assert get_tp32_backend() is megakernel
    monkeypatch.setenv("MPK_TP32_EXPERT_PARALLEL", "0")
    assert get_tp32_backend().REPLICATED_EXPERTS
    monkeypatch.setenv("MPK_EXPERT_TP", "16")
    with pytest.raises(ValueError, match="requires MPK_EXPERT_TP=8"):
        get_tp32_backend()
    monkeypatch.setenv("MPK_TP32_EXPERT_PARALLEL", "typo")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        get_tp32_backend()
