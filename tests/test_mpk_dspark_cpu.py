import jax.numpy as jnp
import numpy as np
import pytest

from mpk.dspark import acceptance, norm, rope


@pytest.mark.parametrize("heads", (1, 2))
def test_verify_cache_preserves_each_kv_head(heads):
    from mpk.decode_megakernel import prepare_verify_cache

    shape = (2, 3, heads, 16, 128)
    key = jnp.arange(np.prod(shape), dtype=jnp.int32).reshape(shape)
    value = key + 100000
    result = prepare_verify_cache({"kcache": key, "vcache": value})["kvcache"]
    assert result.shape == (2, 3, 1, 16, heads * 128)
    for head in range(heads):
        np.testing.assert_array_equal(result[:, :, 0, :, head * 128:head * 128 + 64], key[:, :, head, :, :64])
        np.testing.assert_array_equal(result[:, :, 0, :, head * 128 + 64:(head + 1) * 128], value[:, :, head, :, :64])


@pytest.mark.parametrize("accepted", range(8))
def test_greedy_prefix_acceptance_and_bonus(accepted):
    proposals = jnp.arange(10, 17, dtype=jnp.int32)
    targets = jnp.arange(10, 18, dtype=jnp.int32)
    if accepted < 7:
        targets = targets.at[accepted].set(99)
    count, bonus, output = acceptance(proposals, targets)
    assert int(count) == accepted
    assert int(bonus) == int(targets[accepted])
    np.testing.assert_array_equal(np.asarray(output)[:accepted], np.asarray(proposals)[:accepted])
    assert int(output[accepted]) == int(bonus)
    assert np.all(np.asarray(output)[accepted + 1:] == -1)


@pytest.mark.parametrize("length", range(1, 9))
def test_synthetic_acceptance_includes_bonus(length):
    proposals = jnp.arange(7, dtype=jnp.int32)
    targets = jnp.arange(100, 108, dtype=jnp.int32)
    count, bonus, output = acceptance(proposals, targets, length)
    assert int(count) == length - 1
    assert int(bonus) == 100 + length - 1
    assert int(jnp.sum(output >= 0)) == length


def test_interleaved_muse_rope():
    x = jnp.array([[[1., 2., 3., 4.]]], jnp.bfloat16)
    result = rope(x, jnp.array([2]), 10000.)
    angles = np.array([2., .02])
    expected = np.array([1., 3.]) * np.cos(angles) - np.array([2., 4.]) * np.sin(angles)
    expected_odd = np.array([1., 3.]) * np.sin(angles) + np.array([2., 4.]) * np.cos(angles)
    np.testing.assert_allclose(np.asarray(result, np.float32)[0, 0, ::2], expected, atol=.016)
    np.testing.assert_allclose(np.asarray(result, np.float32)[0, 0, 1::2], expected_odd, atol=.016)


def test_input_norm_does_not_apply_gain_before_statistics():
    x = jnp.array([[1., 2., 4., 8.]], jnp.bfloat16)
    gain = jnp.array([1., 3., 2., 4.])
    expected = np.asarray(x, np.float32) / np.sqrt(np.mean(np.asarray(x, np.float32) ** 2) + 1e-5) * np.asarray(gain)
    np.testing.assert_allclose(norm(x, 1e-5, gain), expected, rtol=1e-6)
