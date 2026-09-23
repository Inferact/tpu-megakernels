"""CPU checks that execute arithmetic from the production megakernel.

The expert tests run the real Pallas body with ``interpret=True``. This checks
values and ref mutation, but not TPU lowering, layouts, VMEM, or DMA timing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from kimi import decode_megakernel as megakernel
import kimi
from kimi import load as kimi_load


@pytest.mark.parametrize("rows", [1, 4, 8])
def test_mxu_dot_padding_matches_unpadded_matmul(rows):
    rng = np.random.default_rng(rows)
    left = jnp.asarray(rng.normal(size=(rows, 256)), jnp.float32).astype(jnp.bfloat16)
    right = jnp.asarray(rng.normal(size=(256, 128)), jnp.float32).astype(jnp.bfloat16)

    actual = megakernel._dot(left, right)
    expected = jax.lax.dot_general(
        left, right, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32
    )

    assert actual.shape == (rows, 128)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("mode", ["bf16", "hilo"])
def test_probability_value_mxu_matches_split_dot(rows, mode):
    rng = np.random.default_rng(rows)
    probabilities = jnp.asarray(rng.uniform(size=(rows, 128)), jnp.float32)
    values = jnp.asarray(rng.normal(size=(128, 128)), jnp.float32).astype(
        jnp.bfloat16
    )

    actual = megakernel._probability_value_mxu(probabilities, values, mode)
    high = probabilities.astype(jnp.bfloat16)
    expected = jax.lax.dot_general(
        high,
        values,
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    if mode == "hilo":
        low = (probabilities - high.astype(jnp.float32)).astype(jnp.bfloat16)
        expected += jax.lax.dot_general(
            low,
            values,
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    assert actual.shape == (rows, 128)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)


def test_attention_scores_pad_partial_mxu_row_tile():
    rng = np.random.default_rng(4)
    queries = jnp.asarray(rng.normal(size=(5, 256)), jnp.float32).astype(jnp.bfloat16)
    keys = jnp.asarray(rng.normal(size=(128, 256)), jnp.float32).astype(jnp.bfloat16)

    actual = megakernel._attention_scores(queries, keys)
    expected = jax.lax.dot_general(
        queries, keys, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )

    assert actual.shape == (5, 128)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("relaxed", [False, True])
def test_residual_mixture_matches_direct_formula(relaxed):
    rng = np.random.default_rng(7)
    blocks = jnp.asarray(rng.normal(size=(8, 256)), jnp.float32).astype(jnp.bfloat16)
    prefix = jnp.asarray(rng.normal(size=(1, 256)), jnp.float32).astype(jnp.bfloat16)
    folded = jnp.asarray(rng.normal(size=(1, 256)), jnp.float32).astype(jnp.bfloat16)

    actual = megakernel._residual_mixture_row(
        blocks,
        prefix,
        folded,
        5,
        epsilon=1e-5,
        relaxed_reduction_order=relaxed,
    )
    values = jnp.concatenate((blocks[:5], prefix), axis=0).astype(jnp.float32)
    scores = jnp.sum(values * folded.astype(jnp.float32), axis=1)
    scores *= jax.lax.rsqrt(jnp.mean(values * values, axis=1) + 1e-5)
    expected = jnp.sum(jax.nn.softmax(scores)[:, None] * values, axis=0, keepdims=True)

    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=5e-3,
        atol=5e-3,
    )


def _expert_fixture(seed):
    rng = np.random.default_rng(seed)
    source = jnp.asarray(rng.normal(size=(2, 256)), jnp.float32).astype(jnp.bfloat16)
    packed = jnp.asarray(
        rng.integers(0, np.iinfo(np.uint32).max, size=(32, 128), dtype=np.uint32)
    )
    scales = jnp.asarray(rng.integers(124, 130, size=(8, 128), dtype=np.uint8))
    return source, packed, scales


def _interpreted_expert_dot(source, weights, scales):
    def kernel(source_ref, weights_ref, scales_ref, output_ref):
        megakernel._quantized_expert_dot(
            source_ref, weights_ref, scales_ref, output_ref, 256, 128
        )

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((2, 128), jnp.float32),
        interpret=True,
    )(source, weights, scales)


@pytest.mark.parametrize("storage", ["mxfp4", "reblocked_fp8"])
def test_interpreted_quantized_expert_dot_matches_dense_dequant(storage):
    source, packed, scales = _expert_fixture(seed=11)
    expected = jax.lax.dot_general(
        source,
        kimi.unpack_mxfp4(packed, scales),
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    if storage == "reblocked_fp8":
        weights, dot_scales, lossless = kimi_load._reblock_one(packed, scales, 256)
        assert bool(lossless)
    else:
        weights, dot_scales = packed, scales

    actual = _interpreted_expert_dot(source, weights, dot_scales)

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
