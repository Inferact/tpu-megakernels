"""PyTorch CPU shim for DeepSeek-V4.1-Flash's TileLang GPU kernels (`REF/kernel.py`).

Replaces the 6 TileLang functions in `REF/kernel.py`:
- `act_quant`
- `fp4_act_quant`
- `fp8_gemm`
- `fp4_gemm`
- `sparse_attn`
- `hc_split_sinkhorn`

with exact PyTorch CPU implementations matching `INVENTORY.md`, so that unmodified `REF/model.py`
runs on CPU (`c4-highmem-192` or local `tiny_config`). Also provides `make_random_reference_weights`
producing matching PyTorch and JAX weight dictionaries.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import types
from typing import Any

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch
import torch.nn.functional as F

from deepseek_v41.config import DSV41Config
from deepseek_v41.engram_hash import DEFAULT_COMPRESSED_VOCAB_SIZE, SyntheticTokenizer
from deepseek_v41.quant import (
    FP4_POS_VALUES_NP,
    FP4_TABLE_NP,
    act_quant_np,
    fp4_act_quant_np,
)

DEFAULT_REF_INFERENCE_DIRS = tuple(
    p
    for p in (
        Path(os.environ["DSV41_REF_INFERENCE_DIR"]) if "DSV41_REF_INFERENCE_DIR" in os.environ else None,
        Path(__file__).resolve().parents[2] / "reference-inference",
    )
    if p is not None
)

_FP4_TABLE_TORCH = torch.tensor(FP4_TABLE_NP, dtype=torch.float32)
_FP4_POS_TORCH = torch.tensor(FP4_POS_VALUES_NP, dtype=torch.float32)


def _fast_round_scale_torch(amax: torch.Tensor, max_inv: float) -> torch.Tensor:
    """Computes `2^ceil(log2(amax * float32(max_inv)))` via IEEE-754 bits in PyTorch (`REF/kernel.py:22-38`)."""
    prod = amax.to(torch.float32) * float(np.float32(max_inv))
    bits = prod.view(torch.int32)
    exp_x = (bits >> 23) & 0xFF
    man_bits = bits & ((1 << 23) - 1)
    log2_ceil = exp_x - 127 + (man_bits != 0).to(torch.int32)
    biased = torch.clamp(log2_ceil + 127, min=0, max=254)
    return (biased << 23).view(torch.float32)


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """PyTorch CPU implementation of `REF/kernel.py:act_quant` (lines 40-124)."""
    n = x.size(-1)
    assert n % block_size == 0, f"Last dim {n} must be divisible by block_size={block_size}"
    z = x.contiguous()
    x_f32 = z.float().view(*z.shape[:-1], n // block_size, block_size)
    amax = torch.clamp(x_f32.abs().amax(dim=-1), min=1e-4)
    if scale_fmt is not None:
        s_f32 = _fast_round_scale_torch(amax, float(np.float32(1.0 / 448.0)))
    else:
        s_f32 = amax * float(np.float32(1.0 / 448.0))
    scaled = torch.clamp(x_f32 / s_f32.unsqueeze(-1), min=-448.0, max=448.0)
    q_fp8 = scaled.to(torch.float8_e4m3fn)
    if inplace:
        deq = (q_fp8.float() * s_f32.unsqueeze(-1)).reshape(x.shape).to(x.dtype)
        x.copy_(deq)
        return x
    return q_fp8.reshape(x.shape), s_f32.to(scale_dtype)


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: torch.dtype = torch.float8_e8m0fnu,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """PyTorch CPU implementation of `REF/kernel.py:fp4_act_quant` (lines 128-205)."""
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    n = x.size(-1)
    assert n % block_size == 0, f"Last dim {n} must be divisible by block_size={block_size}"
    z = x.contiguous()
    x_f32 = z.float().view(*z.shape[:-1], n // block_size, block_size)
    amax = x_f32.abs().amax(dim=-1)
    if scale_dtype == torch.float8_e4m3fn:
        amax = torch.clamp(amax, min=float(6.0 * (2.0**-9)))
        s_out = (amax / 6.0).to(torch.float8_e4m3fn)
        s_f32 = s_out.float()
    else:
        amax = torch.clamp(amax, min=float(6.0 * (2.0**-126)))
        s_f32 = _fast_round_scale_torch(amax, float(np.float32(1.0 / 6.0)))
        s_out = s_f32.to(torch.float8_e8m0fnu)

    ax = x_f32.abs()
    s = s_f32.unsqueeze(-1)
    mag_code = (
        (ax > 0.25 * s).long()
        + (ax >= 0.75 * s).long()
        + (ax > 1.25 * s).long()
        + (ax >= 1.75 * s).long()
        + (ax > 2.5 * s).long()
        + (ax >= 3.5 * s).long()
        + (ax > 5.0 * s).long()
    )
    pos_table = _FP4_POS_TORCH.to(x.device)
    neg = torch.signbit(x_f32)
    q_f32 = pos_table[mag_code] * torch.where(neg, -1.0, 1.0)
    if inplace:
        deq = (q_f32 * s).reshape(x.shape).to(x.dtype)
        x.copy_(deq)
        return x
    codes_u8 = (mag_code.to(torch.uint8) | (neg.to(torch.uint8) << 3)).reshape(*x.shape[:-1], n // 2, 2)
    packed_u8 = codes_u8[..., 0] | (codes_u8[..., 1] << 4)
    return packed_u8.view(torch.float4_e2m1fn_x2), s_out


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    block_size: int = 128,
) -> torch.Tensor:
    """PyTorch CPU implementation of `REF/kernel.py:fp8_gemm` (lines 207-307)."""
    del scale_dtype
    assert block_size in (32, 128)
    k = a.size(-1)
    n = b.size(0)
    a_deq = a.float().view(*a.shape[:-1], k // block_size, block_size) * a_s.float().unsqueeze(-1)
    a_deq = a_deq.reshape(*a.shape[:-1], k)
    b_s_f32 = b_s.float().repeat_interleave(block_size, dim=0)[:n, :].repeat_interleave(block_size, dim=1)[:, :k]
    b_deq = b.float() * b_s_f32
    out = F.linear(a_deq, b_deq)
    return out.to(torch.bfloat16)


def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    act_block_size: int = 128,
) -> torch.Tensor:
    """PyTorch CPU implementation of `REF/kernel.py:fp4_gemm` (lines 477-591)."""
    del scale_dtype
    assert act_block_size in (32, 128)
    k = a.size(-1)
    n = b.size(0)
    a_deq = a.float().view(*a.shape[:-1], k // act_block_size, act_block_size) * a_s.float().unsqueeze(-1)
    a_deq = a_deq.reshape(*a.shape[:-1], k)

    b_u8 = b.view(torch.uint8)
    low = (b_u8 & 0x0F).long()
    high = ((b_u8 >> 4) & 0x0F).long()
    table = _FP4_TABLE_TORCH.to(a.device)
    b_fp4 = torch.stack([table[low], table[high]], dim=-1).reshape(n, k)
    # Match `REF/kernel.py:540`: FP4 -> FP32 -> FP8 (exact for E2M1) then scale per 32 along K.
    b_deq = (b_fp4.view(n, k // 32, 32) * b_s.float().unsqueeze(-1)).reshape(n, k)
    out = F.linear(a_deq, b_deq)
    return out.to(torch.bfloat16)


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """PyTorch CPU implementation of `REF/kernel.py:sparse_attn` (lines 310-403).

    Replicates the online 64-index block loop, `scores_max` initialized to `-1e30`,
    `P` cast to `bfloat16` before `P @ V`, and unscaled `attn_sink` added to `sum_exp` at the end.
    """
    b, m, h, d = q.size()
    if softmax_scale is None:
        softmax_scale = float(d**-0.5)
    topk = topk_idxs.size(-1)
    block = 64
    num_blocks = (topk + block - 1) // block
    pad_k = num_blocks * block - topk
    if pad_k > 0:
        idxs_padded = F.pad(topk_idxs.to(torch.int64), (0, pad_k), value=-1)
    else:
        idxs_padded = topk_idxs.to(torch.int64)

    q_bf16 = q.to(torch.bfloat16)
    kv_bf16 = kv.to(torch.bfloat16)
    acc_o = torch.zeros((b, m, h, d), dtype=torch.float32, device=q.device)
    sum_exp = torch.zeros((b, m, h), dtype=torch.float32, device=q.device)
    scores_max = torch.full((b, m, h), -1e30, dtype=torch.float32, device=q.device)
    batch_idx = torch.arange(b, device=q.device)[:, None, None]

    for t in range(num_blocks):
        blk_idxs = idxs_padded[:, :, t * block : (t + 1) * block]  # [B, M, 64]
        valid = blk_idxs != -1
        safe_idxs = blk_idxs.clamp_min(0)
        kv_blk = kv_bf16[batch_idx, safe_idxs]  # [B, M, 64, D] bf16
        kv_blk = torch.where(valid.unsqueeze(-1), kv_blk, torch.zeros_like(kv_blk))

        # fp32 dot product of bf16 operands, scaled by softmax_scale; -inf where idx == -1.
        acc_s = torch.einsum("bmhd,bmkd->bmhk", q_bf16.float(), kv_blk.float()) * float(softmax_scale)
        acc_s = torch.where(valid.unsqueeze(2), acc_s, torch.full_like(acc_s, -torch.inf))

        scores_max_prev = scores_max
        blk_max = acc_s.amax(dim=-1)
        scores_max = torch.maximum(scores_max_prev, blk_max)
        scores_scale = torch.exp(scores_max_prev - scores_max)
        p_f32 = torch.exp(acc_s - scores_max.unsqueeze(-1))
        scores_sum = p_f32.sum(dim=-1)
        sum_exp = sum_exp * scores_scale + scores_sum

        # Cast P to bfloat16 before P @ V (`REF/kernel.py:377-380`).
        p_bf16 = p_f32.to(torch.bfloat16)
        acc_o = acc_o * scores_scale.unsqueeze(-1) + torch.einsum(
            "bmhk,bmkd->bmhd", p_bf16.float(), kv_blk.float()
        )

    sum_exp = sum_exp + torch.exp(attn_sink.float().view(1, 1, h) - scores_max)
    out = (acc_o / sum_exp.unsqueeze(-1)).to(q.dtype)
    return out


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch CPU implementation of `REF/kernel.py:hc_split_sinkhorn` (lines 406-474).

    When `sinkhorn_iters <= 0`, skips the row-softmax and
    Sinkhorn row/column normalizations on `comb`.
    """
    m = mixes.float()
    s = hc_scale.float()
    b = hc_base.float()
    pre = torch.sigmoid(m[..., :hc_mult] * s[0] + b[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(m[..., hc_mult : 2 * hc_mult] * s[1] + b[hc_mult : 2 * hc_mult])
    comb = (m[..., 2 * hc_mult :] * s[2] + b[2 * hc_mult :]).unflatten(-1, (hc_mult, hc_mult))
    if sinkhorn_iters > 0:
        row_max = comb.amax(dim=-1, keepdim=True)
        comb = torch.exp(comb - row_max)
        comb = comb / comb.sum(dim=-1, keepdim=True) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def install_torch_cpu_kernel_shim() -> types.ModuleType:
    """Registers a synthetic `kernel` module in `sys.modules` so `REF/model.py` imports on CPU."""
    try:
        import PIL  # noqa: F401
    except ModuleNotFoundError:
        pil_mod = types.ModuleType("PIL")
        pil_img = types.ModuleType("PIL.Image")
        pil_img.Image = object  # type: ignore[attr-defined]
        pil_ops = types.ModuleType("PIL.ImageOps")
        pil_mod.Image = pil_img  # type: ignore[attr-defined]
        pil_mod.ImageOps = pil_ops  # type: ignore[attr-defined]
        sys.modules.setdefault("PIL", pil_mod)
        sys.modules.setdefault("PIL.Image", pil_img)
        sys.modules.setdefault("PIL.ImageOps", pil_ops)

    shim = types.ModuleType("kernel")
    shim.act_quant = act_quant
    shim.fp4_act_quant = fp4_act_quant
    shim.fp8_gemm = fp8_gemm
    shim.fp4_gemm = fp4_gemm
    shim.sparse_attn = sparse_attn
    shim.hc_split_sinkhorn = hc_split_sinkhorn
    sys.modules["kernel"] = shim
    if "model" in sys.modules:
        mod = sys.modules["model"]
        for fn_name in ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm", "sparse_attn", "hc_split_sinkhorn"):
            setattr(mod, fn_name, getattr(shim, fn_name))
    return shim


def resolve_reference_inference_dir(ref_dir: str | Path | None = None) -> Path:
    if ref_dir is not None:
        p = Path(ref_dir)
        if p.is_dir():
            return p
    env_dir = os.environ.get("DSV41_REF_INFERENCE_DIR")
    if env_dir and Path(env_dir).is_dir():
        return Path(env_dir)
    for candidate in DEFAULT_REF_INFERENCE_DIRS:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not locate `research/reference-inference` directory")


def load_reference_model_module(ref_dir: str | Path | None = None) -> types.ModuleType:
    """Installs the CPU kernel shim and imports `REF/model.py` unmodified."""
    install_torch_cpu_kernel_shim()
    rdir = str(resolve_reference_inference_dir(ref_dir))
    if rdir not in sys.path:
        sys.path.insert(0, rdir)
    import model as ref_model  # type: ignore[import-not-found]

    for fn_name in ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm", "sparse_attn", "hc_split_sinkhorn"):
        setattr(ref_model, fn_name, globals()[fn_name])
    return ref_model


def config_to_model_args(
    cfg: DSV41Config,
    *,
    max_batch_size: int = 4,
    max_seq_len: int = 256,
    temperature: float = 0.0,
    compressed_vocab_size: int | None = None,
    ref_dir: str | Path | None = None,
) -> Any:
    """Converts a `DSV41Config` into `REF/model.py`'s `ModelArgs`."""
    ref_model = load_reference_model_module(ref_dir)
    c_vocab = compressed_vocab_size or (
        DEFAULT_COMPRESSED_VOCAB_SIZE if cfg.vocab_size == 129280 else cfg.vocab_size
    )
    return ref_model.ModelArgs(
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        temperature=temperature,
        dtype="fp8",
        expert_dtype="fp4",
        vocab_size=cfg.vocab_size,
        dim=cfg.dim,
        moe_inter_dim=cfg.moe_inter_dim,
        n_layers=cfg.n_layers,
        n_mtp_layers=cfg.n_mtp_layers,
        n_heads=cfg.n_heads,
        n_routed_experts=cfg.n_routed_experts,
        n_shared_experts=cfg.n_shared_experts,
        n_activated_experts=cfg.n_activated_experts,
        score_func="sqrtsoftplus",
        gate_temp=1.0,
        norm_topk_prob=True,
        route_scale=cfg.route_scale,
        swiglu_limit=cfg.swiglu_limit,
        q_lora_rank=cfg.q_lora_rank,
        head_dim=cfg.head_dim,
        rope_head_dim=cfg.qk_rope_head_dim,
        norm_eps=cfg.rms_norm_eps,
        o_groups=cfg.o_groups,
        o_lora_rank=cfg.o_lora_rank,
        window_size=cfg.sliding_window,
        compress_ratios=cfg.compress_ratios,
        kv_source_layers=cfg.kv_source_layer_ids,
        index_source_layers=cfg.index_source_layer_ids,
        compress_rope_theta=cfg.compress_rope_theta,
        original_seq_len=cfg.original_seq_len,
        rope_theta=cfg.rope_theta,
        rope_factor=cfg.rope_factor,
        beta_fast=cfg.beta_fast,
        beta_slow=cfg.beta_slow,
        index_n_heads=cfg.index_n_heads,
        index_head_dim=cfg.index_head_dim,
        index_topk=cfg.index_topk,
        candidate_source_layer=cfg.candidate_source_layer_id,
        candidate_topk_blocks=cfg.candidate_topk_blocks,
        candidate_block_size=cfg.candidate_block_size,
        hc_mult=cfg.hc_mult,
        hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
        hc_eps=cfg.hc_eps,
        engram_layer_ids=cfg.engram_layer_ids,
        engram_num_embeddings=cfg.engram_num_embeddings,
        engram_max_ngram_size=cfg.engram_max_ngram_size,
        engram_vocab_size=cfg.engram_vocab_size,
        engram_n_heads=cfg.engram_n_heads,
        engram_head_dim=cfg.engram_head_dim,
        engram_pad_id=cfg.engram_pad_id,
        engram_compressed_vocab_size=c_vocab if cfg.engram_layer_ids else 0,
        vision_n_layers=0,
        dspark_block_size=cfg.dspark_block_size,
        dspark_noise_token_id=cfg.dspark_noise_token_id,
        dspark_target_layer_ids=cfg.dspark_target_layer_ids,
        dspark_markov_rank=cfg.markov_rank,
        dspark_n_routed_experts=cfg.dspark_n_routed_experts,
        dspark_n_activated_experts=cfg.dspark_n_activated_experts,
    )


def _np_bf16_to_torch(arr: np.ndarray) -> torch.Tensor:
    u16 = np.asarray(arr, dtype=ml_dtypes.bfloat16).view(np.uint16).copy()
    return torch.from_numpy(u16).view(torch.bfloat16)


def _np_fp8_to_torch(arr: np.ndarray) -> torch.Tensor:
    u8 = np.asarray(arr, dtype=ml_dtypes.float8_e4m3fn).view(np.uint8).copy()
    return torch.from_numpy(u8).view(torch.float8_e4m3fn)


def _np_e8m0_to_torch(arr: np.ndarray) -> torch.Tensor:
    u8 = np.asarray(arr, dtype=np.uint8).copy()
    return torch.from_numpy(u8).view(torch.float8_e8m0fnu)


def _np_fp4x2_to_torch(arr_i8: np.ndarray) -> torch.Tensor:
    u8 = np.asarray(arr_i8).view(np.uint8).copy()
    return torch.from_numpy(u8).view(torch.float4_e2m1fn_x2)


def make_random_reference_weights(
    cfg: DSV41Config,
    seed: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Generates matching deterministic quantized weights for PyTorch (`REF/model.py`) and JAX (`reference.py`).

    Returns `(torch_state_dict, jax_weights)` keyed by official checkpoint tensor names.
    """
    rng = np.random.default_rng(seed)
    torch_sd: dict[str, torch.Tensor] = {}
    jax_w: dict[str, Any] = {}

    def _add_bf16(name: str, shape: tuple[int, ...], std: float = 0.08, offset: float = 0.0, as_fp32_in_torch: bool = False):
        arr = (rng.standard_normal(shape).astype(np.float32) * std + offset).astype(ml_dtypes.bfloat16)
        t = _np_bf16_to_torch(arr)
        torch_sd[name] = t.float() if as_fp32_in_torch else t
        jax_w[name] = jnp.asarray(arr, dtype=jnp.bfloat16)

    def _add_fp32(name: str, shape: tuple[int, ...], std: float = 0.05, offset: float = 0.0):
        arr = rng.standard_normal(shape).astype(np.float32) * std + offset
        torch_sd[name] = torch.from_numpy(arr.copy())
        jax_w[name] = jnp.asarray(arr, dtype=jnp.float32)

    def _add_fp8_linear(prefix: str, out_features: int, in_features: int, std: float = 0.08):
        raw = (rng.standard_normal((out_features, in_features)).astype(np.float32) * std).astype(ml_dtypes.bfloat16)
        # Quantize per 32x32 block using power-of-2 E8M0 scale so weights are realistic FP8+E8M0.
        bm, bk = out_features // 32, in_features // 32
        blocks = raw.astype(np.float32).reshape(bm, 32, bk, 32).transpose(0, 2, 1, 3).reshape(bm * bk, 1024)
        q_blk, s_u8_blk = act_quant_np(blocks, block_size=1024, scale_fmt="ue8m0", inplace=False, return_u8_scale=True)
        w_fp8 = q_blk.reshape(bm, bk, 32, 32).transpose(0, 2, 1, 3).reshape(out_features, in_features)
        s_u8 = s_u8_blk.reshape(bm, bk)
        torch_sd[f"{prefix}.weight"] = _np_fp8_to_torch(w_fp8)
        torch_sd[f"{prefix}.scale"] = _np_e8m0_to_torch(s_u8)
        jax_w[f"{prefix}.weight"] = jnp.asarray(w_fp8, dtype=jnp.float8_e4m3fn)
        jax_w[f"{prefix}.scale"] = jnp.asarray(s_u8, dtype=jnp.uint8)

    def _add_fp4_linear(prefix: str, out_features: int, in_features: int, std: float = 0.08):
        raw = (rng.standard_normal((out_features, in_features)).astype(np.float32) * std).astype(ml_dtypes.bfloat16)
        codes_u8, s_u8 = fp4_act_quant_np(
            raw, block_size=32, inplace=False, scale_dtype=np.uint8, return_codes=True
        )
        packed_u8 = (codes_u8[:, 0::2] & np.uint8(0x0F)) | ((codes_u8[:, 1::2] & np.uint8(0x0F)) << np.uint8(4))
        packed_i8 = packed_u8.view(np.int8)
        torch_sd[f"{prefix}.weight"] = _np_fp4x2_to_torch(packed_i8)
        torch_sd[f"{prefix}.scale"] = _np_e8m0_to_torch(s_u8)
        jax_w[f"{prefix}.weight"] = jnp.asarray(packed_i8, dtype=jnp.int8)
        jax_w[f"{prefix}.scale"] = jnp.asarray(s_u8, dtype=jnp.uint8)

    _add_bf16("embed.weight", (cfg.vocab_size, cfg.dim), std=0.12)
    _add_bf16("norm.weight", (cfg.dim,), std=0.05, offset=1.0)
    _add_bf16("head.weight", (cfg.vocab_size, cfg.dim), std=0.12, as_fp32_in_torch=True)

    total_layers = cfg.n_layers + cfg.n_mtp_layers
    for layer_id in range(total_layers):
        is_backbone = layer_id < cfg.n_layers
        l_prefix = f"layers.{layer_id}" if is_backbone else f"mtp.{layer_id - cfg.n_layers}"
        r = cfg.compress_ratios[layer_id]

        _add_bf16(f"{l_prefix}.attn_norm.weight", (cfg.dim,), std=0.05, offset=1.0)
        _add_bf16(f"{l_prefix}.ffn_norm.weight", (cfg.dim,), std=0.05, offset=1.0)

        mix_hc = (2 + cfg.hc_mult) * cfg.hc_mult
        hc_dim = cfg.hc_mult * cfg.dim
        _add_fp32(f"{l_prefix}.hc_attn_fn", (mix_hc, hc_dim), std=0.08)
        _add_fp32(f"{l_prefix}.hc_ffn_fn", (mix_hc, hc_dim), std=0.08)
        _add_fp32(f"{l_prefix}.hc_attn_base", (mix_hc,), std=0.2)
        _add_fp32(f"{l_prefix}.hc_ffn_base", (mix_hc,), std=0.2)
        _add_fp32(f"{l_prefix}.hc_attn_scale", (3,), std=0.1, offset=0.8)
        _add_fp32(f"{l_prefix}.hc_ffn_scale", (3,), std=0.1, offset=0.8)

        # Attention weights
        _add_fp32(f"{l_prefix}.attn.attn_sink", (cfg.n_heads,), std=0.2, offset=-1.0)
        _add_fp8_linear(f"{l_prefix}.attn.wq_a", cfg.q_lora_rank, cfg.dim, std=0.1)
        _add_bf16(f"{l_prefix}.attn.q_norm.weight", (cfg.q_lora_rank,), std=0.05, offset=1.0)
        _add_fp8_linear(f"{l_prefix}.attn.wq_b", cfg.n_heads * cfg.head_dim, cfg.q_lora_rank, std=0.1)
        _add_fp8_linear(f"{l_prefix}.attn.wkv", cfg.head_dim, cfg.dim, std=0.1)
        _add_bf16(f"{l_prefix}.attn.kv_norm.weight", (cfg.head_dim,), std=0.05, offset=1.0)
        _add_bf16(
            f"{l_prefix}.attn.wo_a.weight",
            (cfg.o_groups * cfg.o_lora_rank, cfg.n_heads * cfg.head_dim // cfg.o_groups),
            std=0.08,
        )
        _add_fp8_linear(f"{l_prefix}.attn.wo_b", cfg.dim, cfg.o_groups * cfg.o_lora_rank, std=0.08)

        if is_backbone and layer_id in cfg.kv_source_layer_ids:
            _add_bf16(f"{l_prefix}.attn.compressor.norm.weight", (cfg.head_dim,), std=0.05, offset=1.0)
            _add_bf16(
                f"{l_prefix}.attn.compressor.wkv.weight",
                (cfg.head_dim, cfg.dim),
                std=0.1,
                as_fp32_in_torch=(r > 1),
            )
            if r > 1:
                _add_bf16(
                    f"{l_prefix}.attn.compressor.wgate.weight",
                    (cfg.head_dim, cfg.dim),
                    std=0.1,
                    as_fp32_in_torch=True,
                )

        if is_backbone and layer_id in cfg.index_source_layer_ids:
            _add_fp8_linear(
                f"{l_prefix}.attn.indexer.wq_b",
                cfg.index_n_heads * cfg.index_head_dim,
                cfg.q_lora_rank,
                std=0.25,
            )
            _add_bf16(
                f"{l_prefix}.attn.indexer.weights_proj.weight",
                (cfg.index_n_heads, cfg.dim),
                std=0.2,
                offset=0.05,
            )
            if layer_id in cfg.kv_source_layer_ids:
                _add_bf16(f"{l_prefix}.attn.indexer.wk.weight", (cfg.index_head_dim, cfg.head_dim), std=0.25)
                _add_bf16(f"{l_prefix}.attn.indexer.k_norm.weight", (cfg.index_head_dim,), std=0.05, offset=1.0)

        if is_backbone and layer_id in cfg.engram_layer_ids:
            eng_idx = cfg.engram_layer_ids.index(layer_id)
            n_emb = cfg.engram_num_embeddings[eng_idx]
            h_dim = cfg.engram_head_dim
            raw_emb = (rng.standard_normal((n_emb, h_dim)).astype(np.float32) * 0.15).astype(ml_dtypes.bfloat16)
            q_emb, s_emb_u8 = act_quant_np(raw_emb, block_size=32, scale_fmt="ue8m0", inplace=False, return_u8_scale=True)
            torch_sd[f"{l_prefix}.engram.embed.weight"] = _np_fp8_to_torch(q_emb)
            torch_sd[f"{l_prefix}.engram.embed.scale"] = _np_e8m0_to_torch(s_emb_u8)
            jax_w[f"{l_prefix}.engram.embed.weight"] = jnp.asarray(q_emb, dtype=jnp.float8_e4m3fn)
            jax_w[f"{l_prefix}.engram.embed.scale"] = jnp.asarray(s_emb_u8, dtype=jnp.uint8)

            n_hash_cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
            _add_fp8_linear(
                f"{l_prefix}.engram.wkv",
                cfg.dim * (cfg.hc_mult + 1),
                n_hash_cols * h_dim,
                std=0.08,
            )
            _add_bf16(f"{l_prefix}.engram.q_weight", (cfg.hc_mult, cfg.dim), std=0.05, offset=0.5)
            _add_bf16(f"{l_prefix}.engram.k_weight", (cfg.hc_mult, cfg.dim), std=0.05, offset=0.5)

        # MoE weights
        n_exp = cfg.n_routed_experts if is_backbone else cfg.dspark_n_routed_experts
        _add_bf16(f"{l_prefix}.ffn.gate.weight", (n_exp, cfg.dim), std=0.05)
        # Distinct spacing on gate.bias prevents 1-ULP bfloat16 rounding from flipping expert top-k ties.
        perm = rng.permutation(n_exp)
        bias_arr = np.linspace(-2.0, 2.0, n_exp, dtype=np.float32)[perm]
        torch_sd[f"{l_prefix}.ffn.gate.bias"] = torch.from_numpy(bias_arr.copy())
        jax_w[f"{l_prefix}.ffn.gate.bias"] = jnp.asarray(bias_arr, dtype=jnp.float32)

        _add_fp8_linear(f"{l_prefix}.ffn.shared_experts.w1", cfg.moe_inter_dim, cfg.dim, std=0.08)
        _add_fp8_linear(f"{l_prefix}.ffn.shared_experts.w2", cfg.dim, cfg.moe_inter_dim, std=0.08)
        _add_fp8_linear(f"{l_prefix}.ffn.shared_experts.w3", cfg.moe_inter_dim, cfg.dim, std=0.08)

        for e in range(n_exp):
            _add_fp4_linear(f"{l_prefix}.ffn.experts.{e}.w1", cfg.moe_inter_dim, cfg.dim, std=0.08)
            _add_fp4_linear(f"{l_prefix}.ffn.experts.{e}.w2", cfg.dim, cfg.moe_inter_dim, std=0.08)
            _add_fp4_linear(f"{l_prefix}.ffn.experts.{e}.w3", cfg.moe_inter_dim, cfg.dim, std=0.08)

        if not is_backbone:
            stage_id = layer_id - cfg.n_layers
            if stage_id == 0:
                _add_fp8_linear(
                    f"{l_prefix}.main_proj",
                    cfg.dim,
                    cfg.dim * len(cfg.dspark_target_layer_ids),
                    std=0.08,
                )
                _add_bf16(f"{l_prefix}.main_norm.weight", (cfg.dim,), std=0.05, offset=1.0)
            if stage_id == cfg.n_mtp_layers - 1:
                _add_bf16(f"{l_prefix}.norm.weight", (cfg.dim,), std=0.05, offset=1.0)
                _add_bf16(f"{l_prefix}.markov_head.embed.weight", (cfg.vocab_size, cfg.markov_rank), std=0.12)
                _add_bf16(
                    f"{l_prefix}.markov_head.head.weight",
                    (cfg.vocab_size, cfg.markov_rank),
                    std=0.12,
                    as_fp32_in_torch=True,
                )
                _add_bf16(
                    f"{l_prefix}.confidence_head.proj.weight",
                    (1, cfg.dim + cfg.markov_rank),
                    std=0.1,
                    as_fp32_in_torch=True,
                )

    return torch_sd, jax_w


def load_weights_into_torch_model(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Loads `state_dict` into `REF/model.py`'s `Transformer` and re-links `weight.scale = scale`."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [
        k
        for k in missing
        if not (k.startswith("mtp.") and (k.endswith(".embed.weight") or k.endswith(".head.weight")))
    ]
    if missing:
        raise ValueError(f"Missing keys when loading state_dict into Torch reference model: {missing[:10]}")
    if unexpected:
        raise ValueError(f"Unexpected keys when loading state_dict into Torch reference model: {unexpected[:10]}")
    for module in model.modules():
        if hasattr(module, "weight") and hasattr(module, "scale") and getattr(module, "scale") is not None:
            module.weight.scale = module.scale


def set_torch_model_modes(
    model: torch.nn.Module,
    *,
    index_k_mode: str = "reference",
    sinkhorn_iters: int | None = None,
) -> None:
    """Configures `index_k_mode` (`"reference"` vs `"intended"`) and `sinkhorn_iters` on a Torch `Transformer`."""
    if index_k_mode not in ("reference", "intended"):
        raise ValueError(f"index_k_mode must be 'reference' or 'intended', got {index_k_mode!r}")

    ref_model = sys.modules.get("model")
    for handle in getattr(model, "_index_k_mode_hooks", []):
        handle.remove()
    model._index_k_mode_hooks = []  # type: ignore[attr-defined]

    if index_k_mode == "intended" and ref_model is not None:
        for layer in model.layers:
            indexer = getattr(layer.attn, "indexer", None)
            if indexer is not None and getattr(indexer, "owns_k", False):
                def _pre_hook(mod, _inputs):
                    ref_model.shared_attn.index_k = mod.k_cache

                model._index_k_mode_hooks.append(indexer.register_forward_pre_hook(_pre_hook))  # type: ignore[attr-defined]

    if sinkhorn_iters is not None:
        for layer in list(model.layers) + list(getattr(model, "mtp", [])):
            layer.hc_sinkhorn_iters = int(sinkhorn_iters)


def build_torch_reference_model(
    cfg: DSV41Config,
    torch_state_dict: dict[str, torch.Tensor] | None = None,
    *,
    seed: int = 0,
    index_k_mode: str | None = None,
    sinkhorn_iters: int | None = None,
    max_batch_size: int = 4,
    max_seq_len: int = 256,
    temperature: float = 0.0,
    tokenizer: Any = None,
    ref_dir: str | Path | None = None,
) -> Any:
    """Instantiates unmodified `REF/model.py` `Transformer` on CPU with the CPU kernel shim."""
    ref_model = load_reference_model_module(ref_dir)
    args = config_to_model_args(
        cfg,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        temperature=temperature,
        ref_dir=ref_dir,
    )
    if tokenizer is None and cfg.engram_layer_ids:
        tokenizer = SyntheticTokenizer(cfg.vocab_size, args.engram_compressed_vocab_size)

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cpu"):
            model = ref_model.Transformer(args, tokenizer=tokenizer)
    finally:
        torch.set_default_dtype(prev_dtype)

    if torch_state_dict is None:
        torch_state_dict, _ = make_random_reference_weights(cfg, seed=seed)
    load_weights_into_torch_model(model, torch_state_dict)
    set_torch_model_modes(
        model,
        index_k_mode=index_k_mode or cfg.index_k_mode,
        sinkhorn_iters=sinkhorn_iters if sinkhorn_iters is not None else cfg.hc_sinkhorn_iters,
    )
    ref_model.get_window_topk_idxs.cache_clear()
    ref_model.get_dspark_topk_idxs.cache_clear()
    return model
