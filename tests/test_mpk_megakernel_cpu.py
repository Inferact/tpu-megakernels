"""CPU tests for the Muse-Spark sharded prefill (plain XLA) against the oracle.

Runs make_prefill's shard_map on a single CPU device (tp=1): validates the
sharded math path end to end, including NVFP4 packed-expert dequantization.
The Pallas decode kernel is validated on TPU (see dev/tpu_mini_check.py).
"""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

import mpk as model
from mpk import decode_megakernel as megakernel
from mpk import load as mpk_load

_MINI_KW = dict(
    dim=512,
    layers=4,
    heads=8,
    kv_heads=4,
    head_dim=64,
    vocab=2048,
    moe_hidden=128,
    moe_intermediate=256,
    experts=16,
    top_k=4,
    sliding_window=8,
    full_attention=(1,),
)
MINI = model.Config(**_MINI_KW, bf16_expert_layers=(0, 1, 2, 3))
MINI_PACKED = model.Config(**_MINI_KW, bf16_expert_layers=(1,))  # 0/2/3 use the NVFP4 path


def quantize_nvfp4(w, block=16):
    """Lossy NVFP4 encoding of [..., K] -> (packed u8 [..., K/2], e4m3 [..., K/16])."""
    w = np.asarray(w, np.float32)
    k = w.shape[-1]
    blocked = w.reshape(*w.shape[:-1], k // block, block)
    amax = np.abs(blocked).max(axis=-1, keepdims=True)
    scales = (amax / 6.0).astype(ml_dtypes.float8_e4m3fn)
    s32 = scales.astype(np.float32)
    codes = blocked / np.where(s32 == 0, 1.0, s32)
    mags = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)
    idx = np.argmin(np.abs(np.abs(codes)[..., None] - mags), axis=-1).astype(np.uint8)
    nib = ((codes < 0).astype(np.uint8) << 3) | idx
    packed = ((nib[..., 0::2] & 15) | (nib[..., 1::2] << 4)).reshape(*w.shape[:-1], k // 2)
    return packed.astype(np.uint8), scales[..., 0]


def _peaked_router(weights, c, scale=30.0):
    """Scale up router logits so sigmoid top-k selections are stable under bf16 noise."""
    layers = []
    for w in weights["layers"]:
        w = dict(w)
        w["router"] = np.asarray(w["router"], np.float32) * scale
        layers.append(w)
    return {**weights, "layers": tuple(layers)}


def _packed_inputs(c, weights):
    """(base, dense) host dicts with the packed families filled for non-dense layers.

    Also returns the oracle weights with those experts replaced by their
    dequantized BF16 round-trip, so both paths compute with identical values.
    """
    base, dense = mpk_load.pack_host(weights, c, tp=1)
    packed_layers = [i for i in range(c.layers) if i not in c.bf16_expert_layers]
    oracle_layers = list(weights["layers"])
    gu_p, gu_s, gu_g, dw_p, dw_s, dw_g = [], [], [], [], [], []
    for li in packed_layers:
        gu = np.asarray(weights["layers"][li]["experts"]["gate_up"], np.float32)
        dw = np.asarray(weights["layers"][li]["experts"]["down"], np.float32)
        gp, gs = quantize_nvfp4(gu.transpose(0, 2, 1))  # [E, 2I, K] N-major
        dp, ds = quantize_nvfp4(dw.transpose(0, 2, 1))  # [E, K, I] -> packed along I
        gp, gs = gp.transpose(0, 2, 1), gs.transpose(0, 2, 1)  # -> K-major [E, K/2, 2I]
        dp, ds = dp.transpose(0, 2, 1), ds.transpose(0, 2, 1)  # -> K-major [E, I/2, K]
        gu_p.append(gp)
        gu_s.append(gs.view(np.uint8))
        gu_g.append(np.ones((c.experts, 2), np.float32))
        dw_p.append(dp)
        dw_s.append(ds.view(np.uint8))
        dw_g.append(np.ones((c.experts,), np.float32))
        oracle_layers[li] = dict(oracle_layers[li])
        oracle_layers[li]["experts"] = {
            "gate_up": mpk_load.dequantize_experts_bf16(gp, gs, np.ones((c.experts, 2), np.float32)),
            "down": mpk_load.dequantize_experts_bf16(dp, ds, np.ones((c.experts,), np.float32)),
        }
    if packed_layers:
        # pack_host packed every layer into the dense families; the kernel's
        # dense families hold only bf16_expert_layers (dense_ord indexing).
        keep = list(c.bf16_expert_layers)
        base["gu_dense"] = base["gu_dense"][:, keep]
        base["dw_dense"] = base["dw_dense"][:, keep]
        base["gu_packed"] = np.stack(gu_p, 0)[None]
        base["gu_scales"] = np.stack(gu_s, 0)[None]
        base["gu_global"] = np.stack(gu_g, 0)[None]
        base["dw_packed"] = np.stack(dw_p, 0)[None]
        base["dw_scales"] = np.stack(dw_s, 0)[None]
        base["dw_global"] = np.stack(dw_g, 0)[None]
    oracle_w = {**weights, "layers": tuple(oracle_layers)}
    return base, dense, oracle_w


def _prefill_states(c, context, tp=1):
    return {
        "kcache": jnp.zeros((tp, c.layers, c.kv_heads // tp, context, 128), jnp.bfloat16),
        "vcache": jnp.zeros((tp, c.layers, c.kv_heads // tp, context, 128), jnp.bfloat16),
    }


def _run_prefill(c, base, dense, tokens, context, tp=1):
    w = jax.tree.map(jnp.asarray, {**base, **dense})
    states = _prefill_states(c, context, tp)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:tp]), ("tp",))
    prefill = megakernel.make_prefill(mesh, c, context, tp)
    return jax.jit(prefill)(
        w, states, jnp.asarray(tokens, jnp.int32), jnp.array(0, jnp.int32)
    )


def test_nvfp4_quantize_dequant_roundtrip():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((3, 256), dtype=np.float32)
    packed, scales = quantize_nvfp4(w)
    back = model.dequantize_nvfp4(packed, scales, np.float32(1.0))
    rel = np.abs(back - w) / np.maximum(np.abs(w), 1e-3)
    assert np.median(rel) < 0.15  # 4-bit is lossy; blocks should track closely


def test_sharded_prefill_matches_oracle_dense():
    c = MINI
    context = 256
    weights = mpk_load.canonical_weights(c, seed=7)
    weights = _peaked_router(weights, c)
    base, dense, oracle_w = _packed_inputs(c, weights)
    tokens = [5, 11, 900, 17]
    (states_p, logits_p) = _run_prefill(c, base, dense, tokens, context)
    states_o, logits_o = model.prefill(
        jax.tree.map(jnp.asarray, oracle_w),
        jax.tree.map(jnp.asarray, mpk_load.zero_states(c, context)),
        tokens,
        c,
    )
    np.testing.assert_allclose(
        np.asarray(logits_p).reshape(-1)[: c.vocab],
        np.asarray(logits_o),
        rtol=3e-2,
        atol=3e-2,
    )
    for li in range(c.layers):
        np.testing.assert_allclose(
            np.asarray(states_p["kcache"])[0, li, :, : len(tokens), :64].astype(np.float32),
            np.asarray(states_o[li][0])[:, : len(tokens)].astype(np.float32),
            rtol=5e-2,
            atol=5e-2,
        )


def test_sharded_prefill_matches_oracle_packed():
    c = MINI_PACKED
    context = 256
    weights = mpk_load.canonical_weights(c, seed=11)
    weights = _peaked_router(weights, c)
    base, dense, oracle_w = _packed_inputs(c, weights)
    tokens = [5, 11, 900, 17]
    (states_p, logits_p) = _run_prefill(c, base, dense, tokens, context)
    states_o, logits_o = model.prefill(
        jax.tree.map(jnp.asarray, oracle_w),
        jax.tree.map(jnp.asarray, mpk_load.zero_states(c, context)),
        tokens,
        c,
    )
    np.testing.assert_allclose(
        np.asarray(logits_p).reshape(-1)[: c.vocab], np.asarray(logits_o), rtol=3e-2, atol=3e-2
    )
