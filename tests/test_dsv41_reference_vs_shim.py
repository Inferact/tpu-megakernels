"""Suite T2 — Parity tests between `REF/model.py` + `torch_cpu_shim` and JAX `deepseek_v41/reference.py`."""

from __future__ import annotations

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
import torch

from deepseek_v41.config import tiny_config
from deepseek_v41.golden.torch_cpu_shim import (
    build_torch_reference_model,
    hc_split_sinkhorn as torch_hc_split_sinkhorn,
    make_random_reference_weights,
)
from deepseek_v41.reference import (
    DeepSeekV41Reference,
    hc_mixes as jax_hc_mixes,
    hc_post as jax_hc_post,
    hc_pre as jax_hc_pre,
    hc_split_sinkhorn as jax_hc_split_sinkhorn,
)


@pytest.fixture(scope="module")
def tiny_fixture():
    """Creates the shared tiny_config and deterministic quantized weights for Torch and JAX."""
    from deepseek_v41.golden.torch_cpu_shim import resolve_reference_inference_dir

    try:
        resolve_reference_inference_dir()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    cfg = tiny_config()
    torch_sd, jax_w = make_random_reference_weights(cfg, seed=42)
    return cfg, torch_sd, jax_w


def test_sublayer_parity_mhc_sinkhorn(tiny_fixture):
    """Checks `hc_mixes`, 20-iter `hc_split_sinkhorn`, `hc_pre`, and `hc_post` between Torch and JAX."""
    cfg, torch_sd, jax_w = tiny_fixture
    torch_model = build_torch_reference_model(cfg, torch_sd, max_batch_size=2, max_seq_len=32)
    block0_t = torch_model.layers[0]

    rng = np.random.default_rng(101)
    x_np = (rng.standard_normal((2, 4, cfg.hc_mult, cfg.dim)) * 0.5).astype(ml_dtypes.bfloat16)
    sub_np = (rng.standard_normal((2, 4, cfg.dim)) * 0.5).astype(ml_dtypes.bfloat16)

    x_t = torch.from_numpy(x_np.view(np.uint16).copy()).view(torch.bfloat16)
    sub_t = torch.from_numpy(sub_np.view(np.uint16).copy()).view(torch.bfloat16)
    x_j = jnp.asarray(x_np, dtype=jnp.bfloat16)
    sub_j = jnp.asarray(sub_np, dtype=jnp.bfloat16)

    # 1. Direct `hc_split_sinkhorn` parity
    mixes_np = rng.standard_normal((2, 4, 24)).astype(np.float32)
    pre_t, post_t, comb_t = torch_hc_split_sinkhorn(
        torch.from_numpy(mixes_np),
        block0_t.hc_attn_scale,
        block0_t.hc_attn_base,
        hc_mult=cfg.hc_mult,
        sinkhorn_iters=cfg.hc_sinkhorn_iters,
        eps=cfg.hc_eps,
    )
    pre_j, post_j, comb_j = jax_hc_split_sinkhorn(
        jnp.asarray(mixes_np),
        jax_w["layers.0.hc_attn_scale"],
        jax_w["layers.0.hc_attn_base"],
        hc_mult=cfg.hc_mult,
        sinkhorn_iters=cfg.hc_sinkhorn_iters,
        eps=cfg.hc_eps,
    )
    assert float(np.max(np.abs(pre_t.detach().numpy() - np.asarray(pre_j)))) < 1e-6
    assert float(np.max(np.abs(post_t.detach().numpy() - np.asarray(post_j)))) < 1e-6
    assert float(np.max(np.abs(comb_t.detach().numpy() - np.asarray(comb_j)))) < 1e-6

    # Verify doubly-stochastic row and column sums are ~1.0 after 20 Sinkhorn iterations.
    np.testing.assert_allclose(np.asarray(comb_j).sum(axis=-1), 1.0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(comb_j).sum(axis=-2), 1.0, atol=1e-4)

    # 2. Full `hc_mixes`, `hc_pre`, and `hc_post` parity
    pre_m_t, post_m_t, comb_m_t = block0_t.hc_mixes(
        x_t, block0_t.hc_attn_fn, block0_t.hc_attn_scale, block0_t.hc_attn_base
    )
    pre_m_j, post_m_j, comb_m_j = jax_hc_mixes(
        x_j,
        jax_w["layers.0.hc_attn_fn"],
        jax_w["layers.0.hc_attn_scale"],
        jax_w["layers.0.hc_attn_base"],
        hc_mult=cfg.hc_mult,
        sinkhorn_iters=cfg.hc_sinkhorn_iters,
        eps=cfg.hc_eps,
        norm_eps=cfg.rms_norm_eps,
    )
    assert float(np.max(np.abs(pre_m_t.detach().numpy() - np.asarray(pre_m_j)))) < 1e-5
    assert float(np.max(np.abs(post_m_t.detach().numpy() - np.asarray(post_m_j)))) < 1e-5
    assert float(np.max(np.abs(comb_m_t.detach().numpy() - np.asarray(comb_m_j)))) < 1e-5

    collapsed_t = block0_t.hc_pre(x_t, pre_m_t)
    collapsed_j = jax_hc_pre(x_j, pre_m_j)
    c_t_u16 = collapsed_t.detach().view(torch.uint16).numpy().astype(np.int32)
    c_j_u16 = np.asarray(collapsed_j).view(np.uint16).astype(np.int32)
    assert int(np.max(np.abs(c_t_u16 - c_j_u16))) <= 1

    expanded_exact_t = block0_t.hc_post(sub_t, x_t, post_t, comb_t)
    expanded_exact_j = jax_hc_post(sub_j, x_j, pre_j * 0.0 + post_j, comb_j)
    ee_t_u16 = expanded_exact_t.detach().view(torch.uint16).numpy().astype(np.int32)
    ee_j_u16 = np.asarray(expanded_exact_j).view(np.uint16).astype(np.int32)
    assert int(np.max(np.abs(ee_t_u16 - ee_j_u16))) <= 1

    expanded_t = block0_t.hc_post(sub_t, x_t, post_m_t, comb_m_t)
    expanded_j = jax_hc_post(sub_j, x_j, post_m_j, comb_m_j)
    e_t_f32 = expanded_t.detach().float().numpy()
    e_j_f32 = np.asarray(expanded_j, dtype=np.float32)
    rel_l2 = float(np.linalg.norm(e_t_f32 - e_j_f32) / (np.linalg.norm(e_t_f32) + 1e-12))
    assert rel_l2 < 1e-4

    # 3. Isolated single-block parity across all 8 backbone layers (all 6 attention kinds + Engram + MoE)
    jm = DeepSeekV41Reference(cfg, jax_w, max_batch_size=2, max_seq_len=32)
    tokens = rng.integers(0, cfg.vocab_size - 4, size=(2, 8), dtype=np.int64)
    tok_t = torch.from_numpy(tokens.copy()).long()
    with torch.no_grad():
        eh_t = torch_model.engram_hash(tok_t, 0, None)
        eh_j = jm.state.engram_hash_state(tokens, 0)
        np.testing.assert_array_equal(eh_t.numpy(), eh_j)
        h_t = torch_model.embed(tok_t).unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        pm_t = torch.zeros((2, 8, cfg.hc_mult), dtype=torch.float32)
        pm_t[:, :, 0] = 1.0
        sr: dict = {}
        for lid, layer in enumerate(torch_model.layers):
            if layer.engram is not None:
                h_t = layer.engram(h_t, eh_t[:, :, layer.engram.layer_hash_index, :], None)
            h_in_j = jnp.asarray(h_t.view(torch.uint16).numpy().view(ml_dtypes.bfloat16))
            pm_in_j = jnp.asarray(pm_t.numpy())
            h_out_j, _ = jm._block_step(lid, h_in_j, 0, pm_in_j, jm.state, sr, "reference", cfg.hc_sinkhorn_iters)
            h_t, pm_t = layer(h_t, 0, pm_t, None)
            ht_f32 = h_t.float().numpy()
            hj_f32 = np.asarray(h_out_j, dtype=np.float32)
            blk_rel_l2 = float(np.linalg.norm(ht_f32 - hj_f32) / (np.linalg.norm(ht_f32) + 1e-12))
            assert blk_rel_l2 < 5e-3, f"Layer {lid} isolated block rel_l2 {blk_rel_l2:.6f} >= 5e-3"


@pytest.mark.parametrize("index_k_mode", ["reference", "intended"])
def test_all_layer_kinds_multi_step_decode(tiny_fixture, index_k_mode):
    """Runs 12 decode steps (`p = 0..11`) across all layer kinds and checks Torch vs JAX parity."""
    cfg, torch_sd, jax_w = tiny_fixture
    bsz = 2
    n_steps = 12
    rng = np.random.default_rng(2026)
    token_seq = rng.integers(0, cfg.vocab_size - 4, size=(bsz, n_steps), dtype=np.int64)

    torch_model = build_torch_reference_model(
        cfg,
        torch_sd,
        index_k_mode=index_k_mode,
        max_batch_size=bsz,
        max_seq_len=64,
    )
    jax_model = DeepSeekV41Reference(
        cfg,
        jax_w,
        index_k_mode=index_k_mode,
        max_batch_size=bsz,
        max_seq_len=64,
    )

    top1_matches = 0
    total_positions = 0
    max_rel_l2 = 0.0

    for p in range(n_steps):
        tok_p = token_seq[:, p : p + 1]
        tok_t = torch.from_numpy(tok_p.copy()).long()
        out_t, logits_t, mh_t = torch_model.forward(tok_t, start_pos=p)
        out_j, logits_j, mh_j = jax_model.forward(tok_p, start_pos=p)

        l_t = logits_t.detach().float().numpy()
        l_j = np.asarray(logits_j, dtype=np.float32)

        rel_l2 = float(np.linalg.norm(l_t - l_j) / (np.linalg.norm(l_t) + 1e-12))
        max_rel_l2 = max(max_rel_l2, rel_l2)

        pred_t = out_t.detach().numpy()
        pred_j = np.asarray(out_j)
        top1_matches += int(np.sum(pred_t == pred_j))
        total_positions += bsz

        # Also check DSpark draft pass (`forward_spec`) at both p=0 (cache seed) and p>0 (Markov draft).
        spec_t = torch_model.forward_spec(out_t, mh_t, start_pos=p)
        spec_j = jax_model.forward_spec(out_t.detach().numpy(), mh_j, start_pos=p)
        if p == 0:
            assert spec_t is None and spec_j is None
        else:
            assert spec_t is not None and spec_j is not None
            _, draft_logits_t, _ = spec_t
            _, draft_logits_j, _ = spec_j
            dl0_t = draft_logits_t[:, 0].detach().float().numpy()
            dl0_j = np.asarray(draft_logits_j[:, 0], dtype=np.float32)
            assert float(np.linalg.norm(dl0_t - dl0_j) / (np.linalg.norm(dl0_t) + 1e-12)) < 0.15

    top1_agreement = top1_matches / total_positions
    assert top1_agreement >= 0.90, f"Expected >= 90% top-1 agreement, got {top1_agreement:.4f}"
    assert max_rel_l2 < 0.12, f"Expected relative L2 logit error < 0.12, got {max_rel_l2:.6f}"


def test_prefill_parity(tiny_fixture):
    """Checks full-sequence prefill (`seqlen = 12`, `start_pos = 0`) parity between Torch and JAX."""
    cfg, torch_sd, jax_w = tiny_fixture
    bsz = 2
    seqlen = 12
    rng = np.random.default_rng(2026)
    tokens = rng.integers(0, cfg.vocab_size - 4, size=(bsz, seqlen), dtype=np.int64)

    torch_model = build_torch_reference_model(cfg, torch_sd, max_batch_size=bsz, max_seq_len=64)
    jax_model = DeepSeekV41Reference(cfg, jax_w, max_batch_size=bsz, max_seq_len=64)

    out_t, logits_t, mh_t = torch_model.forward(torch.from_numpy(tokens.copy()).long(), start_pos=0)
    out_j, logits_j, mh_j = jax_model.forward(tokens, start_pos=0)

    l_t = logits_t.detach().float().numpy()
    l_j = np.asarray(logits_j, dtype=np.float32)
    rel_l2 = float(np.linalg.norm(l_t - l_j) / (np.linalg.norm(l_t) + 1e-12))
    assert rel_l2 < 0.12
    top1_agreement = float(np.mean(out_t.detach().numpy() == np.asarray(out_j)))
    assert top1_agreement >= 0.90

    mh_t_np = mh_t.detach().float().numpy()
    mh_j_np = np.asarray(mh_j, dtype=np.float32)
    assert float(np.linalg.norm(mh_t_np - mh_j_np) / (np.linalg.norm(mh_t_np) + 1e-12)) < 0.12


def test_block_verification_and_rollback(tiny_fixture):
    """Verifies `[1, q=4]` multi-token block verification and non-destructive rollback (`INVENTORY.md` §8.2)."""
    cfg, _, jax_w = tiny_fixture
    bsz = 1
    q = 4
    rng = np.random.default_rng(777)
    seq = rng.integers(0, cfg.vocab_size - 4, size=(bsz, 9), dtype=np.int64)

    seq_model = DeepSeekV41Reference(cfg, jax_w, max_batch_size=bsz, max_seq_len=64)
    blk_model = DeepSeekV41Reference(cfg, jax_w, max_batch_size=bsz, max_seq_len=64)

    # Seed first 3 tokens (positions 0, 1, 2 -> leaves r=2 compressor with 1 pending token at slot 0!).
    for p in range(3):
        tok_p = seq[:, p : p + 1]
        seq_model.forward(tok_p, start_pos=p)
        blk_model.forward(tok_p, start_pos=p)

    # Run q=4 block verification at start_pos=3 (positions 3, 4, 5, 6), accepting 1 draft token (2 tokens consumed: p=3, 4).
    block_tokens = seq[:, 3 : 3 + q]
    verify_res = blk_model.verify_block(block_tokens, start_pos=3, force_num_accepted=1)
    blk_logits = np.asarray(verify_res["logits"], dtype=np.float32)

    # Compare all q=4 verified positions against step-by-step sequential decode.
    seq_logits_list = []
    out_after_k2_seq = None
    log_after_k2_seq = None
    for i in range(q):
        pos = 3 + i
        _, log_s, _ = seq_model.forward(seq[:, pos : pos + 1], start_pos=pos)
        seq_logits_list.append(np.asarray(log_s, dtype=np.float32))
        if i == 1:
            # Capture sequential state after consuming 2 tokens (positions 3 and 4, so next start_pos=5).
            probe_state = seq_model.state.clone()
            out_after_k2_seq, log_after_k2_seq, _ = seq_model.forward(
                seq[:, 5:6], start_pos=5, state=probe_state
            )

    seq_logits = np.stack(seq_logits_list, axis=1)
    rel_l2 = float(np.linalg.norm(seq_logits - blk_logits) / (np.linalg.norm(seq_logits) + 1e-12))
    assert rel_l2 < 1e-5
    assert verify_res["next_start_pos"] == 5
    assert verify_res["accepted_main_hidden"] is not None

    # After rollback inside verify_block, stepping blk_model at start_pos=5 must match sequential decode bit-exactly.
    out_after_k2_blk, log_after_k2_blk, _ = blk_model.forward(seq[:, 5:6], start_pos=5)

    np.testing.assert_array_equal(np.asarray(out_after_k2_seq), np.asarray(out_after_k2_blk))
    np.testing.assert_allclose(
        np.asarray(log_after_k2_seq, dtype=np.float32),
        np.asarray(log_after_k2_blk, dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_negative_control_sinkhorn_zero_breaks_parity(tiny_fixture):
    """Verifies that setting `sinkhorn_iters=0` drops top-1 agreement dramatically (< 90%)."""
    cfg, _, jax_w = tiny_fixture
    bsz = 2
    n_steps = 12
    rng = np.random.default_rng(999)
    token_seq = rng.integers(0, cfg.vocab_size - 4, size=(bsz, n_steps), dtype=np.int64)

    ref_model = DeepSeekV41Reference(
        cfg,
        jax_w,
        index_k_mode="reference",
        sinkhorn_iters=cfg.hc_sinkhorn_iters,
        max_batch_size=bsz,
        max_seq_len=64,
    )
    neg_model = DeepSeekV41Reference(
        cfg,
        jax_w,
        index_k_mode="reference",
        sinkhorn_iters=0,
        max_batch_size=bsz,
        max_seq_len=64,
    )

    matches = 0
    total = 0
    for p in range(n_steps):
        tok_p = token_seq[:, p : p + 1]
        out_ref, _, _ = ref_model.forward(tok_p, start_pos=p)
        out_neg, _, _ = neg_model.forward(tok_p, start_pos=p)
        matches += int(np.sum(np.asarray(out_ref) == np.asarray(out_neg)))
        total += bsz

    agreement = matches / total
    assert agreement < 0.50, f"Expected sinkhorn_iters=0 negative control agreement < 50%, got {agreement:.4f}"
