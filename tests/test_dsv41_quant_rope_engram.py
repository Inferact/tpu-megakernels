"""Suite T1 — Unit tests for DeepSeek-V4.1-Flash quantization, RoPE/YaRN, and Engram hashing."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch

from deepseek_v41.config import full_config, tiny_config
from deepseek_v41.engram_hash import (
    DEFAULT_COMPRESSED_VOCAB_SIZE,
    EngramHashState,
    EngramLayout,
    SyntheticTokenizer,
    compute_hash_multipliers,
)
from deepseek_v41.quant import (
    FP4_TABLE_NP,
    act_quant,
    dequant_fp8_block32x32,
    dequant_mxfp4_block32,
    fp4_act_quant,
    pack_checkpoint_mxfp4_for_v7x_bitcast,
    pack_mxfp4_for_v7x_bitcast,
    pallas_bitcast_mxfp4_to_fp8,
    unpack_mxfp4_v7x_bitcast,
)
from deepseek_v41.rope import (
    apply_rotary_emb,
    compute_rope_freqs_np,
    precompute_freqs_cis,
    precompute_freqs_cis_np,
)


def _ref_fast_round_scale(amax_f32: np.ndarray, max_inv: np.float32) -> np.ndarray:
    """Direct IEEE-754 bit manipulation from `REF/kernel.py:22-38`."""
    prod = np.asarray(amax_f32, dtype=np.float32) * np.float32(max_inv)
    bits_x = prod.view(np.uint32)
    exp_x = ((bits_x >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int32)
    man_bits = bits_x & np.uint32((1 << 23) - 1)
    e = exp_x - 127 + np.where(man_bits != 0, 1, 0)
    return ((e + 127).astype(np.uint32) << np.uint32(23)).view(np.float32)


def test_act_quant_pow2_exact_boundaries():
    """Verifies `act_quant` (e4m3 + ue8m0 per-32) on zeros, boundaries `448 * 2^k`, and random inputs."""
    # 1. All-zero input hits the 1e-4 amax floor.
    zeros = np.zeros((2, 64), dtype=ml_dtypes.bfloat16)
    q_np, s_np = act_quant(zeros, block_size=32, scale_fmt="ue8m0", inplace=False)
    q_jax, s_jax = act_quant(jnp.asarray(zeros), block_size=32, scale_fmt="ue8m0", inplace=False)
    expected_zero_scale = _ref_fast_round_scale(np.array([1e-4], dtype=np.float32), np.float32(1.0 / 448.0))[0]
    np.testing.assert_array_equal(s_np, expected_zero_scale)
    np.testing.assert_array_equal(np.asarray(s_jax), expected_zero_scale)
    np.testing.assert_array_equal(q_np.astype(np.float32), 0.0)
    np.testing.assert_array_equal(np.asarray(q_jax, dtype=np.float32), 0.0)

    # 2. Exact boundary values amax = 448 * 2^k for k in [-20, 20): scale must equal 2^k exactly.
    ks = list(range(-20, 20))
    blocks = np.zeros((len(ks), 32), dtype=np.float32)
    for idx, k in enumerate(ks):
        blocks[idx, 0] = 448.0 * (2.0**k)
        blocks[idx, 1] = -224.0 * (2.0**k)
    blocks_bf16 = blocks.astype(ml_dtypes.bfloat16)
    q_b_np, s_b_np = act_quant(blocks_bf16, block_size=32, scale_fmt="ue8m0", inplace=False)
    q_b_jax, s_b_jax = act_quant(jnp.asarray(blocks_bf16), block_size=32, scale_fmt="ue8m0", inplace=False)
    expected_scales = np.array([max(2.0**k, float(expected_zero_scale)) for k in ks], dtype=np.float32)[:, None]
    np.testing.assert_array_equal(s_b_np, expected_scales)
    np.testing.assert_array_equal(np.asarray(s_b_jax), expected_scales)
    np.testing.assert_array_equal(q_b_np.astype(np.float32), np.asarray(q_b_jax, dtype=np.float32))
    # For k where 448 * 2^k >= 1e-4, quantized element 0 must be 448.0 and element 1 must be -224.0.
    for idx, k in enumerate(ks):
        if 448.0 * (2.0**k) >= 1e-4:
            assert float(q_b_np[idx, 0]) == 448.0
            assert float(q_b_np[idx, 1]) == -224.0

    # 3. Random bf16 inputs: NumPy and JAX match `REF/kernel.py:74-91` in both inplace and out-of-place modes.
    rng = np.random.default_rng(1234)
    x = (rng.standard_normal((4, 128)) * 12.0).astype(ml_dtypes.bfloat16)
    q_np, s_np = act_quant(x, block_size=32, scale_fmt="ue8m0", inplace=False)
    q_jax, s_jax = act_quant(jnp.asarray(x), block_size=32, scale_fmt="ue8m0", inplace=False)
    np.testing.assert_array_equal(s_np, np.asarray(s_jax))
    np.testing.assert_array_equal(q_np.astype(np.float32), np.asarray(q_jax, dtype=np.float32))

    deq_np = act_quant(x.copy(), block_size=32, scale_fmt="ue8m0", inplace=True)
    deq_jax = act_quant(jnp.asarray(x), block_size=32, scale_fmt="ue8m0", inplace=True)
    np.testing.assert_array_equal(deq_np.astype(np.float32), np.asarray(deq_jax, dtype=np.float32))


def test_fp4_act_quant_e8m0_and_e4m3():
    """Verifies `fp4_act_quant` for both block=32 E8M0 and block=16 E4M3 (including exact midpoints)."""
    # 1. All-zero floors: E8M0 floor is 6 * 2^-126 -> scale 2^-126; E4M3 floor is 6 * 2^-9 -> scale 2^-9.
    z32 = np.zeros((2, 64), dtype=ml_dtypes.bfloat16)
    _, s32_np = fp4_act_quant(z32, block_size=32, inplace=False, scale_dtype="e8m0")
    _, s32_jax = fp4_act_quant(jnp.asarray(z32), block_size=32, inplace=False, scale_dtype="e8m0")
    np.testing.assert_array_equal(s32_np, np.float32(2.0**-126))
    np.testing.assert_array_equal(np.asarray(s32_jax), np.float32(2.0**-126))

    z16 = np.zeros((2, 32), dtype=ml_dtypes.bfloat16)
    _, s16_np = fp4_act_quant(z16, block_size=16, inplace=False, scale_dtype="e4m3")
    _, s16_jax = fp4_act_quant(jnp.asarray(z16), block_size=16, inplace=False, scale_dtype="e4m3")
    np.testing.assert_array_equal(s16_np.astype(np.float32), np.float32(2.0**-9))
    np.testing.assert_array_equal(np.asarray(s16_jax, dtype=np.float32), np.float32(2.0**-9))

    # 2. Exact boundary values amax = 6 * 2^k for E8M0: scale must equal 2^k.
    ks = list(range(-15, 15))
    blocks = np.zeros((len(ks), 32), dtype=np.float32)
    for idx, k in enumerate(ks):
        blocks[idx, 0] = 6.0 * (2.0**k)
    _, s_k_np = fp4_act_quant(blocks.astype(ml_dtypes.bfloat16), block_size=32, inplace=False, scale_dtype="e8m0")
    _, s_k_jax = fp4_act_quant(jnp.asarray(blocks, dtype=jnp.bfloat16), block_size=32, inplace=False, scale_dtype="e8m0")
    expected_s = np.array([2.0**k for k in ks], dtype=np.float32)[:, None]
    np.testing.assert_array_equal(s_k_np, expected_s)
    np.testing.assert_array_equal(np.asarray(s_k_jax), expected_s)

    # 3. Exact midpoint ties-to-even test under non-pow2 E4M3 scale (s = 1.5 -> amax = 9.0).
    # Midpoints of E2M1 {0, 0.5, 1, 1.5, 2, 3, 4, 6} are {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}.
    # Under round-to-nearest-even, ties at 0.25->0, 0.75->1.0, 1.25->1.0, 1.75->2.0, 2.5->2.0, 3.5->4.0, 5.0->4.0.
    s_val = np.float32(1.5)
    mids = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32) * s_val
    row = np.zeros((1, 16), dtype=np.float32)
    row[0, 0] = 6.0 * s_val  # sets amax = 9.0 -> s = e4m3(9.0 / 6.0) = 1.5
    row[0, 1:8] = mids
    row[0, 8:15] = -mids
    row_bf16 = row.astype(ml_dtypes.bfloat16)
    deq_np = fp4_act_quant(row_bf16.copy(), block_size=16, inplace=True, scale_dtype="e4m3")
    deq_jax = fp4_act_quant(jnp.asarray(row_bf16), block_size=16, inplace=True, scale_dtype="e4m3")
    expected_ties = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0], dtype=np.float32) * s_val
    np.testing.assert_array_equal(deq_np[0, 1:8].astype(np.float32), expected_ties)
    np.testing.assert_array_equal(deq_np[0, 8:15].astype(np.float32), -expected_ties)
    np.testing.assert_array_equal(deq_np.astype(np.float32), np.asarray(deq_jax, dtype=np.float32))


def test_mxfp4_unpack_and_repack_matches_v7x_bitcast():
    """Verifies that `pack_mxfp4_for_v7x_bitcast` unpacked by `pltpu.bitcast` matches `FP4_TABLE` bit-for-bit."""
    rng = np.random.default_rng(2026)
    k, n = 64, 16
    # Include all 16 codes 0..15 in the first 16 rows, then random codes.
    codes = rng.integers(0, 16, size=(k, n), dtype=np.uint8)
    for c in range(16):
        codes[c, :] = c

    packed_u32 = pack_mxfp4_for_v7x_bitcast(codes, axis=0)
    assert packed_u32.shape == (k // 8, n)
    assert packed_u32.dtype == np.uint32

    # Unpack via Pallas `pltpu.bitcast(u32, float4_e2m1fn).astype(float8_e4m3fn)` in interpret mode.
    pallas_fp8 = pallas_bitcast_mxfp4_to_fp8(jnp.asarray(packed_u32), interpret=True)
    expected_f32 = FP4_TABLE_NP[codes]
    np.testing.assert_array_equal(np.asarray(pallas_fp8, dtype=np.float32), expected_f32)

    # Also verify pure NumPy and pure JAX `unpack_mxfp4_v7x_bitcast`.
    unpacked_np = unpack_mxfp4_v7x_bitcast(packed_u32)
    unpacked_jax = unpack_mxfp4_v7x_bitcast(jnp.asarray(packed_u32))
    np.testing.assert_array_equal(unpacked_np.astype(np.float32), expected_f32)
    np.testing.assert_array_equal(np.asarray(unpacked_jax, dtype=np.float32), expected_f32)

    # Also verify checkpoint `[N, K // 2]` int8 -> v7x `[K // 8, N]` uint32 round-trip and block32 dequant.
    low = codes[0::2, :].T  # Wait: checkpoint packs along K (last dim of [N, K])
    codes_nk = codes.T  # [N, K]
    ckpt_u8 = (codes_nk[:, 0::2] & 0x0F) | ((codes_nk[:, 1::2] & 0x0F) << 4)
    ckpt_i8 = ckpt_u8.view(np.int8)
    packed_from_ckpt = pack_checkpoint_mxfp4_for_v7x_bitcast(ckpt_i8, transpose=True)
    np.testing.assert_array_equal(packed_from_ckpt, packed_u32)

    scales_u8 = rng.integers(120, 132, size=(n, k // 32), dtype=np.uint8)
    deq_ckpt = dequant_mxfp4_block32(ckpt_i8, scales_u8, out_dtype=np.float32)
    deq_v7x = unpack_mxfp4_v7x_bitcast(packed_u32, scales_u8.T, out_dtype=np.float32).T
    np.testing.assert_array_equal(deq_ckpt, deq_v7x)

    # Verify `dequant_fp8_block32x32` round-trip consistency between NumPy and JAX.
    w_fp8 = rng.standard_normal((64, 64)).astype(ml_dtypes.float8_e4m3fn)
    s_fp8 = rng.integers(124, 130, size=(2, 2), dtype=np.uint8)
    np.testing.assert_array_equal(
        dequant_fp8_block32x32(w_fp8, s_fp8, out_dtype=np.float32),
        np.asarray(dequant_fp8_block32x32(jnp.asarray(w_fp8), jnp.asarray(s_fp8), out_dtype=jnp.float32)),
    )


def test_rope_interleaved_and_yarn():
    """Verifies base (`r=0`) and YaRN (`r>0`) frequencies and forward/inverse interleaved RoPE."""
    cfg = full_config()
    # 1. Check YaRN ramp bounds low=15, high=25 and frequencies against `REF/model.py:368-389`.
    freqs_yarn, low, high = compute_rope_freqs_np(
        dim=cfg.qk_rope_head_dim,
        original_seq_len=cfg.original_seq_len,
        base=cfg.compress_rope_theta,
        factor=cfg.rope_factor,
        beta_fast=cfg.beta_fast,
        beta_slow=cfg.beta_slow,
    )
    assert (low, high) == (15, 25)

    freqs_swa, low_swa, high_swa = compute_rope_freqs_np(
        dim=cfg.qk_rope_head_dim,
        original_seq_len=0,
        base=cfg.rope_theta,
    )
    assert (low_swa, high_swa) == (None, None)

    # Compare against PyTorch `REF/model.py:precompute_freqs_cis`.
    def _torch_precompute(dim, seqlen, orig, base, factor, bf, bs):
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if orig > 0:
            c_fast = dim * math.log(orig / (bf * 2 * math.pi)) / (2 * math.log(base))
            c_slow = dim * math.log(orig / (bs * 2 * math.pi)) / (2 * math.log(base))
            lo = max(math.floor(c_fast), 0)
            hi = min(math.ceil(c_slow), dim - 1)
            ramp = ((torch.arange(dim // 2, dtype=torch.float32) - lo) / max(hi - lo, 1e-3)).clamp(0, 1)
            smooth = 1 - ramp
            freqs = freqs / factor * (1 - smooth) + freqs * smooth
        angles = torch.outer(torch.arange(seqlen), freqs)
        return torch.polar(torch.ones_like(angles), angles)

    def _torch_apply(x_t, freqs_cis, inverse=False):
        y = x_t.clone()
        tail = y[..., -64:]
        c = torch.view_as_complex(tail.float().unflatten(-1, (-1, 2)))
        fc = freqs_cis.conj() if inverse else freqs_cis
        fc = fc.view(1, c.size(1), 1, c.size(-1)) if c.ndim == 4 else fc.view(1, c.size(1), c.size(-1))
        tail.copy_(torch.view_as_real(c * fc).flatten(-2))
        return y

    seqlen = 32
    for orig, theta in [(0, cfg.rope_theta), (cfg.original_seq_len, cfg.compress_rope_theta)]:
        fc_t = _torch_precompute(64, seqlen, orig, theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)
        cos_np, sin_np = precompute_freqs_cis_np(64, seqlen, orig, theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)
        cos_jax, sin_jax = precompute_freqs_cis(64, seqlen, orig, theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)

        rng = np.random.default_rng(77)
        x_np = rng.standard_normal((2, seqlen, 4, 512)).astype(ml_dtypes.bfloat16)
        x_t = torch.from_numpy(x_np.view(np.uint16)).view(torch.bfloat16)

        for inv in (False, True):
            want_t = _torch_apply(x_t, fc_t, inverse=inv)
            want_np = want_t.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
            got_np = apply_rotary_emb(x_np, (cos_np, sin_np), inverse=inv)
            got_jax = apply_rotary_emb(jnp.asarray(x_np), (cos_jax, sin_jax), inverse=inv)
            # Non-RoPE prefix [..., :448] must be untouched bit-for-bit.
            np.testing.assert_array_equal(got_np[..., :448], x_np[..., :448])
            np.testing.assert_array_equal(np.asarray(got_jax[..., :448]), x_np[..., :448])
            # RoPE tail [..., -64:] must match PyTorch within <= 1 bf16 ULP.
            np.testing.assert_allclose(
                got_np[..., -64:].astype(np.float32),
                want_np[..., -64:].astype(np.float32),
                rtol=1e-2,
                atol=1e-2,
            )
            np.testing.assert_array_equal(got_np.astype(np.float32), np.asarray(got_jax, dtype=np.float32))

        # Verify forward followed by inverse in float32 is the exact identity.
        x_f32 = rng.standard_normal((2, seqlen, 4, 64)).astype(np.float32)
        rot_f32 = apply_rotary_emb(x_f32, (cos_np, sin_np), inverse=False)
        unrot_f32 = apply_rotary_emb(rot_f32, (cos_np, sin_np), inverse=True)
        np.testing.assert_allclose(unrot_f32, x_f32, rtol=1e-5, atol=1e-5)


def test_engram_hash_primes_and_multipliers():
    """Verifies prime sums (384,006,168 and 384,016,682), PCG64 multipliers, and 24-col hash IDs."""
    cfg = full_config()
    layout = EngramLayout.from_config(cfg)
    assert layout is not None
    assert layout.layer_ids == (1, 14)
    assert layout.n_hash_cols == 24

    primes_np = layout.primes_np
    assert primes_np.shape == (2, 3, 8)
    assert int(primes_np[0].sum()) == 384006168 == cfg.engram_num_embeddings[0]
    assert int(primes_np[1].sum()) == 384016682 == cfg.engram_num_embeddings[1]
    assert int(primes_np[0, 0, 0]) == 16000057
    assert int(primes_np[0, 2, 7]) == 16000463
    assert int(primes_np[1, 0, 0]) == 16000477
    assert int(primes_np[1, 2, 7]) == 16000889

    mults = compute_hash_multipliers(cfg.engram_layer_ids, cfg.engram_max_ngram_size, DEFAULT_COMPRESSED_VOCAB_SIZE)
    assert mults.tolist() == [
        [76632096046245, 4839876093313, 35959672319349, 73987337458391],
        [67716810739261, 51510806800915, 30921347202721, 82619226485591],
    ]

    # Verify streaming decode vs full-sequence prefill hashing and pad propagation at p < s.
    tcfg = tiny_config()
    tok = SyntheticTokenizer(tcfg.vocab_size, tcfg.vocab_size)
    state_prefill = EngramHashState(tcfg, tokenizer=tok)
    state_decode = EngramHashState(tcfg, tokenizer=tok)

    seq = np.array([[10, 25, 300, 42, 7, 99], [5, 6, 7, 8, 9, 10]], dtype=np.int64)
    hashes_full = state_prefill(seq, start_pos=0)
    assert hashes_full.shape == (2, 6, 2, 24)
    assert hashes_full.dtype == np.int64

    hashes_step = []
    for p in range(seq.shape[1]):
        hashes_step.append(state_decode(seq[:, p : p + 1], start_pos=p))
    hashes_cat = np.concatenate(hashes_step, axis=1)
    np.testing.assert_array_equal(hashes_full, hashes_cat)

    # At p = 0, shifts 1, 2, 3 must use pad_id = token_map[2] = 2.
    pad_c = state_prefill.pad_id
    c0 = state_prefill.token_map[seq[0, 0]]
    for l_idx in range(2):
        m = state_prefill.multipliers[l_idx]
        roll = int(c0) * int(m[0])
        for n_idx in range(1, 4):
            roll ^= int(pad_c) * int(m[n_idx])
            for h_idx in range(8):
                col = (n_idx - 1) * 8 + h_idx
                p_mod = int(state_prefill.primes[l_idx, n_idx - 1, h_idx])
                off = int(state_prefill.offsets[l_idx, col])
                assert int(hashes_full[0, 0, l_idx, col]) == (roll % p_mod) + off
