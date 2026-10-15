"""CPU tests for the Muse-Spark TP32 prefill (plain XLA) against the oracle.

Runs make_prefill's shard_map on 32 forced CPU devices (kv_heads=2 < 32 ranks:
rank groups share one KV head). Needs
``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=32``.
The Pallas decode kernel is validated on TPU (see dev/tpu_mini_check_tp32.py).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mpk as model
from mpk import decode_megakernel_32
from mpk import load as mpk_load
from tests.test_mpk_megakernel_cpu import _peaked_router, quantize_nvfp4

_MINI_KW = dict(
    dim=512,
    layers=4,
    heads=64,  # nh=2 per rank at tp=32 (the kernel packs two heads per row)
    kv_heads=2,
    head_dim=64,
    vocab=2048,
    moe_hidden=128,
    moe_intermediate=512,
    experts=32,
    top_k=8,  # the decode kernel's row-transpose exchange maps one row per lane
    sliding_window=8,
    full_attention=(1,),
)
MINI32 = model.Config(**_MINI_KW, bf16_expert_layers=(0, 1, 2, 3))
MINI32_PACKED = model.Config(**_MINI_KW, bf16_expert_layers=(1,))  # 0/2/3 NVFP4


def _cpu32():
    devices = jax.devices("cpu")
    if len(devices) < 32:
        pytest.skip("requires XLA_FLAGS=--xla_force_host_platform_device_count=32")
    return jax.sharding.Mesh(np.asarray(devices[:32]), ("tp",))


def test_pack_pair_share_math():
    """kv_heads=2 over tp=32: rank r holds q head r and shares kv head r//16."""
    c = MINI32
    weights = mpk_load.canonical_weights(c, seed=3)
    ranks = [mpk_load.rank_weights(weights, c, r, 32) for r in range(32)]
    qh, hd = c.heads // 32, c.head_dim
    for r in (0, 1, 15, 16, 17, 31):
        qwkv = np.asarray(ranks[r]["qwkv"][0], np.float32)  # [dim, 2*qh*hd + 2*hd]
        q_ref = np.asarray(weights["layers"][0]["q"][:, r * qh * hd : (r + 1) * qh * hd], np.float32)
        np.testing.assert_array_equal(qwkv[:, : qh * hd], q_ref)
        k_ref = np.asarray(weights["layers"][0]["k"], np.float32)
        head = (r * c.kv_heads) // 32
        np.testing.assert_array_equal(
            qwkv[:, 2 * qh * hd : 2 * qh * hd + hd], k_ref[:, head * hd : (head + 1) * hd]
        )
    base, dense = mpk_load.pack_host(weights, c, tp=32)
    tiled = mpk_load.tile_dense(dense, c, 32)
    assert tiled["qwkv"].shape[0] == 32


def _packed_inputs32(c, weights):
    """(base, dense) tp=32 host dicts with packed families; oracle with dequant experts."""
    # A gap larger than the sigmoid's full range fixes the selected expert IDs.
    ladder = -2 * np.arange(c.experts, dtype=np.float32)
    weights = {**weights, "layers": tuple(dict(l, router_bias=ladder) for l in weights["layers"])}
    base, dense = mpk_load.pack_host(weights, c, tp=32)
    packed_layers = [i for i in range(c.layers) if i not in c.bf16_expert_layers]
    oracle_layers = list(weights["layers"])
    w32 = c.moe_intermediate // 32
    if packed_layers:
        gu_p = np.zeros((32, len(packed_layers), c.experts, c.moe_hidden // 2, 2 * w32), np.uint8)
        gu_s = np.zeros((32, len(packed_layers), c.experts, c.moe_hidden // 16, 2 * w32), np.uint8)
        gu_g = np.ones((32, len(packed_layers), c.experts, 2), np.float32)
        dw_p = np.zeros((32, len(packed_layers), c.experts, w32 // 2, c.moe_hidden), np.uint8)
        dw_s = np.zeros((32, len(packed_layers), c.experts, w32 // 16, c.moe_hidden), np.uint8)
        dw_g = np.ones((32, len(packed_layers), c.experts), np.float32)
        for o, li in enumerate(packed_layers):
            gu = np.asarray(weights["layers"][li]["experts"]["gate_up"], np.float32)
            dw = np.asarray(weights["layers"][li]["experts"]["down"], np.float32)
            gp, gs = quantize_nvfp4(gu.transpose(0, 2, 1))
            dp, ds = quantize_nvfp4(dw.transpose(0, 2, 1))
            gp, gs = gp.transpose(0, 2, 1), gs.transpose(0, 2, 1)  # K-major [E, K/2, 2I]
            dp, ds = dp.transpose(0, 2, 1), ds.transpose(0, 2, 1)  # K-major [E, I/2, K]
            oracle_layers[li] = dict(oracle_layers[li])
            oracle_layers[li]["experts"] = {
                "gate_up": mpk_load.dequantize_experts_bf16(gp, gs, np.ones((c.experts, 2), np.float32)),
                "down": mpk_load.dequantize_experts_bf16(dp, ds, np.ones((c.experts,), np.float32)),
            }
            for r in range(32):
                lo, hi = r * w32, (r + 1) * w32
                up_lo, up_hi = c.moe_intermediate + lo, c.moe_intermediate + hi
                gu_p[r, o] = np.concatenate((gp[:, :, lo:hi], gp[:, :, up_lo:up_hi]), axis=2)
                gu_s[r, o] = np.concatenate((gs[:, :, lo:hi], gs[:, :, up_lo:up_hi]), axis=2).view(np.uint8)
                dw_p[r, o] = dp[:, lo // 2 : hi // 2, :]
                dw_s[r, o] = ds[:, lo // 16 : hi // 16, :].view(np.uint8)
        keep = list(c.bf16_expert_layers)
        base["gu_dense"] = base["gu_dense"][:, keep]
        base["dw_dense"] = base["dw_dense"][:, keep]
        base["gu_packed"] = gu_p
        base["gu_scales"] = gu_s
        base["gu_global"] = gu_g
        base["dw_packed"] = dw_p
        base["dw_scales"] = dw_s
        base["dw_global"] = dw_g
    oracle_w = {**weights, "layers": tuple(oracle_layers)}
    return base, dense, oracle_w


def _run_prefill32(c, base, dense, tokens, context, megakernel):
    mesh = _cpu32()
    w = jax.tree.map(jnp.asarray, {**mpk_load.merge_expert_leaves(base), **dense})
    kvh = 1  # pair-shared KV head per rank
    states = {
        "kcache": jnp.zeros((32, c.layers, kvh, context, 128), jnp.bfloat16),
        "vcache": jnp.zeros((32, c.layers, kvh, context, 128), jnp.bfloat16),
    }
    prefill = megakernel.make_prefill(mesh, c, context, 32)
    return jax.jit(prefill)(w, states, jnp.asarray(tokens, jnp.int32), jnp.array(0, jnp.int32))


def _check_prefill32(c, seed):
    megakernel = decode_megakernel_32._replicated_tp8_backend()
    context = 256
    weights = mpk_load.canonical_weights(c, seed=seed)
    weights = _peaked_router(weights, c)
    base, dense, oracle_w = _packed_inputs32(c, weights)
    tokens = [5, 11, 900, 17]
    states_p, logits_p = _run_prefill32(c, base, dense, tokens, context, megakernel)
    states_o, logits_o = model.prefill(
        jax.tree.map(jnp.asarray, oracle_w),
        jax.tree.map(jnp.asarray, mpk_load.zero_states(c, context)),
        tokens,
        c,
    )
    # 32-way bf16 partial sums (the reference allreduce dtype) sit well above
    # the tp=1 noise floor the TP8 tolerances were tuned for; top_k=8 doubles
    # the expert terms again, so near-zero logits carry ~0.2 abs noise.
    np.testing.assert_allclose(
        np.asarray(logits_p).reshape(32, -1)[:, : c.vocab // 32].reshape(-1),
        np.asarray(logits_o).reshape(-1),
        rtol=0.15,
        atol=0.25,
    )
    heads_per = megakernel.ATTENTION_TP // c.kv_heads
    for li in range(c.layers):
        for r in (0, 1, 15, 16, 31):
            np.testing.assert_allclose(
                np.asarray(states_p["kcache"])[r, li, 0, : len(tokens), :64].astype(np.float32),
                np.asarray(states_o[li][0])[(r % megakernel.ATTENTION_TP) // heads_per, : len(tokens)].astype(np.float32),
                rtol=0.1,
                atol=0.1,
            )


@pytest.mark.cpu32
@pytest.mark.parametrize("attention_tp", (16, 32))
def test_sharded_prefill32_matches_oracle_dense(monkeypatch, attention_tp):
    monkeypatch.setenv("MPK_ATTENTION_TP", str(attention_tp))
    _check_prefill32(MINI32, seed=7)


@pytest.mark.cpu32
@pytest.mark.parametrize("attention_tp", (16, 32))
def test_sharded_prefill32_matches_oracle_packed(monkeypatch, attention_tp):
    monkeypatch.setenv("MPK_ATTENTION_TP", str(attention_tp))
    _check_prefill32(MINI32_PACKED, seed=11)
