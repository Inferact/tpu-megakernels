"""CPU tests for the Muse-Spark reference equations and TP packing (no TPU required)."""

import json
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from model_paths import resolve_model_source
import mpk as model
from mpk import load as mpk_load

MINI = model.Config(
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
    bf16_expert_layers=(0, 1, 2, 3),
)

MUSE_REPO = "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"
MUSE_REVISION = "ac059dd0bf13d962bb668149b95761fd3dc34b4d"


def test_place_pack_merges_expert_bytes_before_placement(tmp_path):
    shape = (1, 2, 3, 16, 128)
    packed = np.arange(np.prod(shape), dtype=np.uint32).astype(np.uint8).reshape(shape)
    scales = np.full((1, 2, 3, 2, 128), 0x38, np.uint8)
    source = {"gu_packed": packed, "gu_scales": scales,
              "dw_packed": packed ^ np.uint8(0x55), "dw_scales": scales + np.uint8(1),
              "norm": np.ones((1, 128), np.float32)}
    mpk_load.save_pack(tmp_path, source)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("tp",))
    actual = mpk_load.place_pack(mesh, tmp_path, merge_experts=True)
    expected = mpk_load.merge_expert_leaves(source)
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    original = mpk_load.place_pack(mesh, tmp_path)
    for name in source:
        np.testing.assert_array_equal(original[name], source[name])


def test_config_from_checkpoint():
    try:
        checkpoint = resolve_model_source(MUSE_REPO, revision=MUSE_REVISION, local_files_only=True)
    except (FileNotFoundError, OSError):
        return
    c = model.Config.from_checkpoint(checkpoint)
    assert c.dim == 8192 and c.layers == 62 and c.vocab == 202048
    assert c.full_attention == tuple(range(1, 62, 4))
    assert c.bf16_expert_layers == (0, 61)
    assert c.experts == 256 and c.top_k == 8
    assert c.sliding_window == 2048
    assert c.scale == c.head_dim**-0.5 * c.qk_scale_factor


def test_nvfp4_dequant_nibble_order():
    # byte 0x1A: low nibble 0xA = -1.0, high nibble 0x1 = +0.5
    packed = np.array([[0x1A, 0, 0, 0, 0, 0, 0, 0]], np.uint8)
    scales = np.full((1, 1), 0x38, np.uint8).view(ml_dtypes.float8_e4m3fn)  # E4M3 1.0
    out = model.dequantize_nvfp4(packed, scales, np.float32(2.0))
    want = np.zeros((1, 16), np.float32)
    want[0, 0], want[0, 1] = -2.0, 1.0
    np.testing.assert_array_equal(out, want)


def test_residual_gate_matches_reference_formula():
    gate = np.linspace(-2, 2, 17, dtype=np.float32)
    alpha, beta = model.residual_coefficients(gate, 0.3)
    w = gate / 0.3
    sig = lambda v: 1.0 / (1.0 + np.exp(-v))
    np.testing.assert_allclose(beta, sig(w), rtol=1e-6)
    np.testing.assert_allclose(alpha, np.sqrt(np.clip(sig(-w) * (1 + sig(w)), 1e-3, None)), rtol=1e-6)


def test_router_bias_selects_but_does_not_scale():
    c = MINI
    weights = mpk_load.canonical_weights(c, seed=11)
    w = weights["layers"][0]
    w["router_bias"] = np.zeros(c.experts, np.float32)
    w["router_bias"][3] = 100.0  # force expert 3 into the top-k
    x = jnp.asarray(np.random.default_rng(0).normal(size=(c.dim,)).astype(np.float32)).astype(
        jnp.bfloat16
    )
    rw, ids = model.route(x, jax.tree.map(jnp.asarray, w), c)
    assert 3 in [int(i) for i in ids]
    # weights are the UNBIASED sigmoid scores, normalized
    scores = jax.nn.sigmoid(
        jnp.dot(x.astype(jnp.float32), jnp.asarray(w["router"]), precision=model.HIGHEST)
        * c.output_multiplier
    )
    for j, i in enumerate(ids):
        np.testing.assert_allclose(float(rw[j]) * float(jnp.sum(jnp.take_along_axis(scores[None], ids[None], 1), 1)[0] + c.router_eps), float(scores[i]), rtol=1e-5)


def test_pack_shapes_and_slices():
    c = MINI
    weights = mpk_load.canonical_weights(c, seed=3)
    base, dense = mpk_load.pack_host(weights, c, tp=2)
    assert dense["qwkv"].shape == (2, 4, 512, 2 * 4 * 64 + 2 * 2 * 64)
    assert base["router"].shape == (2, 4, 512, 8)
    assert base["gu_dense"].shape == (2, 4, 16, 128, 2 * 128)  # [tp, L, E, K, 2I/tp]
    assert base["dw_dense"].shape == (2, 4, 16, 128, 128)  # [tp, L, E, I/tp, K]
    # rank 1's q columns are the second head-group's; gate rides after q
    layer0 = weights["layers"][0]
    qs = 4 * 64
    np.testing.assert_array_equal(
        np.asarray(dense["qwkv"][1, 0, :, :qs]), np.asarray(layer0["q"][:, qs:])
    )
    np.testing.assert_array_equal(
        np.asarray(dense["qwkv"][1, 0, :, qs : 2 * qs]), np.asarray(layer0["gate"][:, qs:])
    )
    # scalars row layout
    np.testing.assert_array_equal(base["scalars"][0, 0, 0], np.asarray(layer0["norm_in"]))
    np.testing.assert_array_equal(base["scalars"][0, 0, 3], layer0["alpha_a"])
    # dense experts are K-major like the oracle: [E, K, N]
    gu = weights["layers"][0]["experts"]["gate_up"]  # [E, K, 2I]
    want = np.concatenate([gu[0, :, 128:256], gu[0, :, 384:]], axis=1)  # rank1 [gate|up]
    np.testing.assert_array_equal(np.asarray(base["gu_dense"][1, 0, 0]), np.asarray(want))


def test_container_roundtrip(tmp_path):
    c = MINI
    weights = mpk_load.canonical_weights(c, seed=9)
    base, dense = mpk_load.pack_host(weights, c, tp=2)
    tiled = mpk_load.tile_dense(dense, c, 2)
    mpk_load.save_bundle(tmp_path, base, dense, tiled)
    prefill_w, decode_w = mpk_load.load_bundle(tmp_path)
    for a, b in zip(jax.tree.leaves(prefill_w), jax.tree.leaves({**base, **dense})):
        assert np.array_equal(
            np.asarray(a).view(np.uint16) if a.dtype != np.float32 else np.asarray(a),
            np.asarray(b).view(np.uint16) if b.dtype != np.float32 else np.asarray(b),
        )
    # tiled dense families untile back to the originals
    for name, bk, bn, nk, nb, nt in mpk_load.dense_schedule(c, 2):
        k = {"qwkv": c.dim, "ow": 4 * 64, "pool": c.dim, "unpool": 64, "lmw": c.dim}[name]
        n = {"qwkv": 768, "ow": c.dim, "pool": 64, "unpool": c.dim, "lmw": mpk_load.lm_width(c, 2)}[name]
        got = mpk_load.until_matrix(decode_w[name], k, n)
        np.testing.assert_array_equal(
            np.asarray(got).view(np.uint16), np.asarray(dense[name]).view(np.uint16)
        )


def test_prefill_matches_decode_oracle():
    """Batched left-padded prefill == token-by-token decode, states and logits."""
    c = MINI
    context = 128
    weights = mpk_load.canonical_weights(c, seed=5)
    states = mpk_load.zero_states(c, context)
    tokens = [3, 41, 7, 900, 12, 55, 260, 1023]
    width = 16
    npad = width - len(tokens)
    padded = [c.pad] * npad + tokens
    s_pre, logits_pre = model.prefill(
        jax.tree.map(jnp.asarray, weights),
        jax.tree.map(jnp.asarray, states),
        padded,
        c,
        npad=npad,
    )
    s_ref = jax.tree.map(jnp.asarray, states)
    for i, t in enumerate(tokens):
        logits_ref, s_ref = model.forward(
            jnp.array(t, jnp.int32), jax.tree.map(jnp.asarray, weights), s_ref,
            jnp.array(i, jnp.int32), c,
        )
    maxrel = 0.0
    for a, b in zip(jax.tree.leaves(s_pre), jax.tree.leaves(s_ref)):
        a32 = np.asarray(a, np.float32)[:, : len(tokens)]
        b32 = np.asarray(b, np.float32)[:, : len(tokens)]
        denom = np.abs(b32).max()
        maxrel = max(maxrel, float(np.abs(a32 - b32).max() / max(denom, 1e-8)))
    print("prefill-vs-decode max rel:", maxrel)
    assert maxrel < 3e-2
    assert (
        np.abs(np.asarray(logits_pre) - np.asarray(logits_ref)).max()
        / np.abs(np.asarray(logits_ref)).max()
        < 3e-2
    )


def test_generate_runs_and_stops():
    c = MINI
    weights = mpk_load.canonical_weights(c, seed=2)
    out = model.generate(jax.tree.map(jnp.asarray, weights), [5, 6, 7], 8, c, 128)
    assert len(out) == 8
    assert all(0 <= t < c.vocab for t in out)
