"""Fast CPU checks for Kimi model semantics and weight encoding."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

import kimi


def test_router_bias_changes_selection_but_not_mixture_weights():
    logits = jnp.array([-2.0, 0.0, 2.0, 3.0])
    bias = jnp.array([2.0, 1.0, 0.0, 0.0])

    expert_ids, weights = kimi.route(logits, bias, 2)

    np.testing.assert_array_equal(expert_ids, [0, 1])
    uncorrected = 1 / (1 + np.exp(-np.array([-2.0, 0.0])))
    np.testing.assert_allclose(weights, uncorrected / uncorrected.sum(), rtol=1e-6)


def test_mxfp4_unpack_covers_all_codes_and_scale_groups():
    packed = np.array([0x76543210, 0xFEDCBA98] * 4, dtype=np.uint32)[:, None]
    scales = jnp.array([[127], [128]], jnp.uint8)

    actual = kimi.unpack_mxfp4(jnp.asarray(packed), scales)

    positive = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    expected = np.tile(np.r_[positive, -positive], 4)
    expected[32:] *= 2
    np.testing.assert_array_equal(actual.astype(jnp.float32)[:, 0], expected)


def test_attention_residual_matches_direct_formula():
    blocks = jnp.array([[1.0, 2.0, 3.0], [3.0, 0.0, -1.0]], jnp.bfloat16)
    prefix = jnp.array([2.0, -1.0, 1.0], jnp.bfloat16)
    folded = jnp.array([0.2, -0.3, 0.4], jnp.bfloat16)

    values = np.concatenate(
        (np.asarray(blocks, dtype=np.float32), np.asarray(prefix, dtype=np.float32)[None])
    )
    scores = (values @ np.asarray(folded, dtype=np.float32)) / np.sqrt(
        np.mean(values * values, axis=1) + 1e-5
    )
    probabilities = np.exp(scores - scores.max())
    expected = probabilities @ values / probabilities.sum()

    np.testing.assert_allclose(
        np.asarray(kimi.attention_residual(prefix, blocks, folded), dtype=np.float32),
        expected,
        rtol=4e-3,
        atol=4e-3,
    )


def test_kda_step_matches_direct_delta_rule():
    config = replace(kimi.Config(), heads=2, head_dim=8)
    rng = np.random.default_rng(918)

    def random(shape):
        return jnp.asarray(rng.normal(size=shape) * 0.3, jnp.float32)

    weights = {
        "conv": random((4, 3, 2, 8)),
        "a_log": random((2,)),
        "dt_bias": random((2, 8)),
        "out_norm": random((8,)) + 1,
    }
    convolution_state = random((3, 3, 2, 8))
    recurrent_state = random((2, 8, 8))
    qkv = random((3, 2, 8))
    raw_gate = random((2, 8))
    update_rate = random((2,))
    output_gate = random((2, 8))

    actual, actual_convolution, actual_state = kimi.kda_step(
        qkv,
        raw_gate,
        update_rate,
        output_gate,
        convolution_state,
        recurrent_state,
        weights,
        config,
    )

    history = jnp.concatenate((convolution_state, qkv[None]), axis=0)
    query, key, value = jax.nn.silu(
        jnp.sum(history.astype(jnp.float32) * weights["conv"], axis=0)
    )
    query *= jax.lax.rsqrt(jnp.sum(query * query, axis=-1, keepdims=True) + 1e-6)
    query *= config.head_dim**-0.5
    key *= jax.lax.rsqrt(jnp.sum(key * key, axis=-1, keepdims=True) + 1e-6)
    decay = config.gate_lower_bound * jax.nn.sigmoid(
        jnp.exp(weights["a_log"])[:, None] * (raw_gate + weights["dt_bias"])
    )
    expected_state = recurrent_state * jnp.exp(decay)[..., None]
    prediction = jnp.einsum("hk,hkv->hv", key, expected_state, precision="highest")
    delta = jax.nn.sigmoid(update_rate)[:, None] * (value - prediction)
    expected_state += key[..., None] * delta[:, None, :]
    attended = jnp.einsum("hk,hkv->hv", query, expected_state, precision="highest")
    expected = kimi.rms(attended, weights["out_norm"], config.eps) * kimi.sigmoid(
        output_gate
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(actual_convolution, history[1:])
    np.testing.assert_allclose(actual_state, expected_state, rtol=2e-5, atol=2e-6)


def test_bfloat16_sigmoid_rounds_once():
    values = jnp.linspace(-8, 8, 257).astype(jnp.bfloat16)
    expected = (1 / (1 + np.exp(-np.asarray(values, dtype=np.float32)))).astype(
        values.dtype
    )
    np.testing.assert_array_equal(kimi.sigmoid(values), expected)
