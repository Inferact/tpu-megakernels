"""CPU tests for the Qwen3.8 reference equations and the TP packing (no TPU required)."""

import json
import jax
import jax.numpy as jnp
import numpy as np

from model_paths import resolve_model_source
import qwen as model
from qwen import load as qwen_load

MINI = model.Config(
    dim=512,
    intermediate=1024,
    layers=4,
    heads=4,
    kv_heads=2,
    head_dim=256,
    rotary_dim=64,
    k_heads=2,
    v_heads=6,
    vocab=1024,
    full_attention=(3,),
)

QWEN_REPO = "Qwen/Qwen3.8-27B"
QWEN_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def test_config_from_checkpoint():
    try:
        checkpoint = resolve_model_source(
            QWEN_REPO, revision=QWEN_REVISION, local_files_only=True
        )
    except (FileNotFoundError, OSError):
        return
    c = model.Config.from_checkpoint(checkpoint)
    assert c.dim == 5120 and c.layers == 64 and c.vocab == 248320
    assert c.full_attention == tuple(range(3, 64, 4))
    assert (c.k_heads, c.v_heads, c.head_dim) == (16, 48, 256)
    assert c.rotary_dim == 64


def test_pack_shapes_and_zero_padding():
    c = MINI
    weights = qwen_load.canonical_weights(c, seed=3)
    packed = qwen_load.pack(weights, c)
    # weights are tile-blocked [tp, L, T, bk, bn]; until back to compare.
    lin, full, lm, *_ = qwen_load.schedule(c, 2, qwen_load.lm_width(c, 2))
    meta = {name: (bk, bn) for name, _, bk, bn, *_ in lin + full + [lm]}
    bk, bn = meta["qkvz"]
    qkv0 = qwen_load.until_matrix(packed["qkvz"][0, 0], c.dim, 1280)
    assert qkv0.shape == (c.dim, 1280)
    assert packed["emb"].shape == (2, c.vocab // 2, c.dim)
    # lm head pad stays zero (checked after untiling back to [dim, lm_width])
    bk, bn = meta["lmw"]
    lm0 = qwen_load.until_matrix(packed["lmw"][0], c.dim, qwen_load.lm_width(c, 2))
    assert not np.any(np.asarray(lm0[:, c.vocab // 2 :]))
    # Replicated families are identical across ranks.
    for name in ("norm_in", "norm_post", "onorm", "qn", "kn", "fnorm"):
        assert np.array_equal(np.asarray(packed[name][0]), np.asarray(packed[name][1]))


def test_pack_matches_canonical_slices():
    c = MINI
    weights = qwen_load.canonical_weights(c, seed=4)
    packed = qwen_load.pack(weights, c)
    lin = weights["layers"][0]["linear"]
    lin_, full_, lm_, *_ = qwen_load.schedule(c, 2, qwen_load.lm_width(c, 2))
    meta = {name: (bk, bn) for name, _, bk, bn, *_ in lin_ + full_ + [lm_]}
    bk, bn = meta["qkvz"]
    qkv1 = qwen_load.until_matrix(packed["qkvz"][1, 0], c.dim, 1280)
    # Rank 1's q slice equals the second half of the canonical q block.
    assert np.array_equal(np.asarray(qkv1[:, :128]), np.asarray(lin["qkv"][:, 128:256]))
    # Rank 0's MLP gate occupies the first half of gu (before the pad tail).
    bk, bn = meta["gu"]
    g0 = qwen_load.until_matrix(packed["gu"][0, 0], c.dim, qwen_load.gu_width(c, 2))
    assert np.array_equal(
        np.asarray(g0[:, : c.intermediate // 2]),
        np.asarray(weights["layers"][0]["mlp"]["gate"][:, :512]),
    )


def test_attention_first_token_is_value_passthrough():
    c = MINI
    weights = qwen_load.canonical_weights(c, seed=5)
    states = qwen_load.zero_states(c, 256)
    x = jnp.asarray(weights["embedding"][7])
    w = weights["layers"][3]
    mixed = model.rms(x, jnp.asarray(w["input_norm"]), c.eps)
    out, _ = model.attention_step(
        mixed, jax.tree.map(jnp.asarray, w["attention"]), jax.tree.map(jnp.asarray, states[3]),
        jnp.array(0, jnp.int32), c,
    )
    # With a single cached token the softmax is one-hot: output == o(v * sigmoid(gate)).
    qg = model.linear(mixed, jnp.asarray(w["attention"]["q"])).reshape(c.heads, 2 * c.head_dim)
    gate = jax.nn.sigmoid(qg[:, c.head_dim :].astype(jnp.float32)).astype(jnp.bfloat16)
    v = model.linear(mixed, jnp.asarray(w["attention"]["v"])).reshape(c.kv_heads, c.head_dim)
    expanded = jnp.repeat(v, c.heads // c.kv_heads, axis=0)
    want = model.linear((expanded * gate).reshape(-1), jnp.asarray(w["attention"]["o"]))
    assert jnp.allclose(out.astype(jnp.float32), want.astype(jnp.float32), atol=2e-2)


def test_delta_step_state_grows():
    c = MINI
    weights = qwen_load.canonical_weights(c, seed=6)
    states = qwen_load.zero_states(c, 256)
    x = jnp.asarray(weights["embedding"][11])
    w = weights["layers"][0]["linear"]
    mixed = model.rms(x, jnp.asarray(weights["layers"][0]["input_norm"]), c.eps)
    _, (conv, rec) = model.delta_step(
        mixed, jax.tree.map(jnp.asarray, w), jax.tree.map(jnp.asarray, states[0]), c
    )
    assert conv.shape == (c.conv_dim, c.conv_size)
    assert rec.shape == (c.v_heads, c.v_dim, c.k_dim)
    assert float(jnp.abs(rec).sum()) > 0


def test_rope_matches_direct_formula():
    c = model.Config()
    cos, sin = model.rope_cos_sin(c, jnp.array(7, jnp.int32))
    half = c.rotary_dim // 2
    # inv_freq exponent is 2j/rotary_dim (the HF convention)
    inv = 1.0 / (c.rope_theta ** (np.arange(half, dtype=np.float32) * 2.0 / c.rotary_dim))
    want_cos = np.concatenate([np.cos(7 * inv)] * 2)
    np.testing.assert_allclose(np.asarray(cos, np.float32), want_cos, atol=2e-3)
    assert cos.shape == (c.rotary_dim,)


def test_pack_container_roundtrip(tmp_path):
    c = MINI
    weights = qwen_load.canonical_weights(c, seed=9)
    packed = qwen_load.pack(weights, c)
    states = qwen_load.pack_states(qwen_load.zero_states(c, 256), c, 2)
    qwen_load.save_pack(tmp_path / "w", packed)
    qwen_load.save_pack(tmp_path / "s", states)
    got_w = qwen_load.load_pack(tmp_path / "w")
    got_s = qwen_load.load_pack(tmp_path / "s")
    for a, b in zip(jax.tree.leaves(packed), jax.tree.leaves(got_w)):
        assert np.array_equal(np.asarray(a).view(np.uint16), np.asarray(b).view(np.uint16))
    for a, b in zip(jax.tree.leaves(states), jax.tree.leaves(got_s)):
        assert np.array_equal(np.asarray(a).view(np.uint16), np.asarray(b).view(np.uint16))


def test_tp8_kv_pair_replication():
    c8 = model.Config(
        dim=512, intermediate=1024, layers=4, heads=8, kv_heads=4, head_dim=256,
        rotary_dim=64, k_heads=8, v_heads=24, vocab=2048, full_attention=(3,),
    )
    weights = qwen_load.canonical_weights(c8, seed=13)
    packed = qwen_load.pack(weights, c8, tp=8)
    # 4 KV heads over 8 ranks: rank pair (2g, 2g+1) shares head g. The kv part
    # rides fused into qwkv after the q columns; compare the untiled kv region.
    lin, full, lm, *_ = qwen_load.schedule(c8, 8, qwen_load.lm_width(c8, 8))
    bk, bn = {n: (b, bb) for n, _, b, bb, *_ in full}["qwkv"]
    w = qwen_load.until_matrix(packed["qwkv"], 512, 1024)
    kv0 = w[0, 0, :, 512:]
    kv1 = w[1, 0, :, 512:]
    kv2 = w[2, 0, :, 512:]
    assert np.array_equal(np.asarray(kv0), np.asarray(kv1))
    assert not np.array_equal(np.asarray(kv0), np.asarray(kv2))
    states = qwen_load.pack_states(qwen_load.zero_states(c8, 256), c8, tp=8)
    assert states["kcache"].shape == (8, 1, 1, 256, 256)


def test_prefill_matches_recurrent_decode():
    """Chunked prefill == token-by-token decode, states and logits."""
    c = MINI
    weights = qwen_load.canonical_weights(c, seed=5)
    states = qwen_load.zero_states(c, 256)
    tokens = [3, 41, 7, 900, 12, 55, 260, 1023]
    s_pre, logits_pre = model.prefill(
        jax.tree.map(jnp.asarray, weights),
        jax.tree.map(jnp.asarray, states),
        tokens,
        c,
    )
    # reference: decode the same tokens one at a time
    s_ref = jax.tree.map(jnp.asarray, states)
    for i, t in enumerate(tokens):
        logits_ref, s_ref = model.forward(
            jnp.array(t, jnp.int32), jax.tree.map(jnp.asarray, weights), s_ref,
            jnp.array(i, jnp.int32), c,
        )
    maxrel = 0.0
    for a, b in zip(jax.tree.leaves(s_pre), jax.tree.leaves(s_ref)):
        a32, b32 = np.asarray(a, np.float32), np.asarray(b, np.float32)
        denom = np.abs(b32).max()
        maxrel = max(maxrel, float(np.abs(a32 - b32).max() / max(denom, 1e-8)))
    print("prefill-vs-decode max rel:", maxrel)
    assert maxrel < 3e-2
    assert np.abs(np.asarray(logits_pre) - np.asarray(logits_ref)).max() / np.abs(
        np.asarray(logits_ref)
    ).max() < 3e-2


def test_lm_and_zba_widths_divisible():
    c = model.Config()
    assert qwen_load.lm_width(c) % 1280 == 0
    assert qwen_load.lm_width(MINI) % 1280 == 0
    assert qwen_load.zba_width(c) % 128 == 0
