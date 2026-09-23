"""CPU tests for the DFlash2 draft reference and packing (no TPU required)."""

import numpy as np
import jax
import jax.numpy as jnp

from qwen import dflash


def tiny_cfg():
    return dflash.DraftConfig(
        {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "intermediate_size": 128,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
            "vocab_size": 256,
            "max_position_embeddings": 512,
            "dflash_config": {
                "block_size": 8,
                "conv_group_size": 16,
                "conv_kernel_size": 2,
                "mask_token_id": 250,
                "selector_rank": 8,
                "selector_top_k": 4,
                "target_layer_ids": [0, 1],
            },
            "layer_types": ["sliding_attention", "sliding_attention"],
            "max_window_layers": 2,
            "sliding_window": 64,
        }
    )


def test_draft_config_fields():
    cfg = tiny_cfg()
    assert cfg.block_size == 8 and cfg.selector_top_k == 4
    assert cfg.hidden == 64 and cfg.layers == 2
    assert cfg.mask_token_id == 250 and cfg.selector_rank == 8
    assert cfg.target_layer_ids == [0, 1]


def test_rope_tables_direct_formula():
    cfg = tiny_cfg()
    positions = jnp.arange(5, dtype=jnp.int32)
    cos, sin = dflash.rope_tables(cfg.theta, cfg.head_dim, positions)
    half = cfg.head_dim // 2
    inv = 1.0 / (cfg.theta ** (np.arange(half, dtype=np.float32) * 2.0 / cfg.head_dim))
    want = np.concatenate([np.cos(3 * inv)] * 2)  # position 3 row
    np.testing.assert_allclose(np.asarray(cos[3], np.float32), want, atol=2e-3)


def test_conv_matches_direct_shift_sum():
    cfg = tiny_cfg()
    L, H = 6, cfg.hidden
    rng = np.random.default_rng(0)
    hidden = jnp.asarray(rng.normal(size=(L, H)).astype(np.float32))
    K, G = cfg.conv_kernel, H // cfg.conv_group
    cw = {
        "kernel_projection": jnp.asarray(rng.normal(size=(2 * K * G, H)).astype(np.float32)),
        "base_kernel": jnp.asarray(rng.normal(size=(2, K, H)).astype(np.float32)),
    }
    convolved, dynamic = dflash.conv_prepare(hidden, cw, cfg)

    # direct: out[t] = sum_tap (base[tap][c] + dyn[t, tap, g(c)]) * x[t - tap]
    dyn = np.asarray(dynamic)  # [L, K, G]
    base = np.asarray(cw["base_kernel"][1])  # [K, H]
    x = np.asarray(hidden)
    want = np.zeros_like(x)
    for tap in range(K):
        shifted = x if tap == 0 else np.pad(x[:-tap], ((tap, 0), (0, 0)))
        group = np.repeat(np.arange(G), cfg.conv_group)
        want += (base[tap][None, :] + dyn[:, tap, group]) * shifted
    got = np.asarray(dflash.conv_finish(hidden, dynamic, cw, cfg), np.float32)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
    assert convolved.shape == (L, H)


def test_select_path_greedy_consistency():
    cfg = tiny_cfg()
    rng = np.random.default_rng(1)
    B, H, V, K, R = cfg.block_size, cfg.hidden, 96, cfg.selector_top_k, cfg.selector_rank
    hidden = jnp.asarray(rng.normal(size=(B, H)).astype(np.float32))
    logits = jnp.asarray(rng.normal(size=(B, V)).astype(np.float32))
    selector = {
        "hidden_projection": jnp.asarray(rng.normal(size=(R, H)).astype(np.float32)),
        "predecessor_codebook": jnp.asarray(rng.normal(size=(V, R)).astype(np.float32)),
        "successor_codebook": jnp.asarray(rng.normal(size=(V, R)).astype(np.float32)),
        "top_k": K,
    }
    anchor = 5
    path, cand, unary = dflash.select_path(hidden, logits, anchor, selector)

    # naive: per position, greedy over the top-K candidates' codebook scores
    unary_np = np.asarray(unary)
    cand_np = np.asarray(cand)
    hproj = np.asarray(hidden) @ np.asarray(selector["hidden_projection"]).T
    pred = anchor
    want = []
    for i in range(B):
        cond = np.asarray(selector["predecessor_codebook"])[pred] * hproj[i]
        scores = unary_np[i] + np.asarray(selector["successor_codebook"])[cand_np[i]] @ cond
        pred = int(cand_np[i, int(np.argmax(scores))])
        want.append(pred)
    assert np.array_equal(np.asarray(path), np.asarray(want, np.int32))
    # top-K candidates are the K largest logits per row
    for i in range(B):
        top = set(np.argsort(-np.asarray(logits[i]))[:K].tolist())
        assert set(cand_np[i].tolist()) == top
