"""Single-`pl.pallas_call` TPU v7x decode & verify megakernel for DeepSeek-V4.1-Flash.

Implements the 40-layer (`cfg.n_layers`) backbone decode and `q`-token verification step inside
a single `pl.pallas_call` wrapped in `jax.shard_map` over `"tp"` with:
- `pltpu.CompilerParams(collective_id=21, vmem_limit_bytes=64 * 2**20, disable_bounds_checks=True, shape_invariant_numerics=True)`
- VMEM scratch pool aliasing via `pool_alias.py` (`storage_size`, `view`) keeping peak VMEM `< 48 MiB`
- In-kernel `pltpu.bitcast(u32_2d, jnp.float4_e2m1fn).astype(jnp.float8_e4m3fn)` MXFP4 upcasting
- Dynamic active-expert streaming from HBM `Ref`s (`w1_u32_ref[e_idx]`, `w3_u32_ref[e_idx]`, `w2_u32_ref[e_idx]`)
- Shared compressed KV (`compress_kv`) and top-k index (`shared_topk_idxs`, `shared_candidates`) reuse across `Reuse` / `Reindex` layers
- 20-iteration Sinkhorn 4-stream mHC (`sinkhorn_iters` configurable)
- Both `index_k_mode="reference"` and `index_k_mode="intended"`
- CPU `interpret=True` execution across 32 virtual devices and TPU v7x `interpret=False` execution.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from deepseek_v41.collectives import all_reduce_sum, barrier, collective_scratch, scope
from deepseek_v41.config import DSV41Config
from deepseek_v41.quant import e8m0_to_f32_jax
from deepseek_v41.xla_decode import (
    _compute_engram_hashes_jax,
    _lookup_engram_rows,
    act_quant,
    apply_rotary_emb,
    engram_forward,
    fp4_act_quant,
    fp8_linear,
    hc_mixes,
    hc_post,
    hc_pre,
    moe_gate,
    pad8_matmul_kn,
    pad8_matmul_nk,
    replicate_to_mesh,
    rms_norm,
    run_shared_expert_fp8,
    select_candidate_blocks,
    sparse_attn,
)
import pool_alias

VMEM_SOFT_BUDGET_BYTES = 48 * 2**20  # 48 MiB soft budget
VMEM_HARD_LIMIT_BYTES = 64 * 2**20   # 64 MiB CompilerParams limit

DEFAULT_COMPILER_PARAMS = pltpu.CompilerParams(
    collective_id=21,
    vmem_limit_bytes=VMEM_HARD_LIMIT_BYTES,
    disable_bounds_checks=True,
    shape_invariant_numerics=True,
)


def _pad_to_multiple(n: int, mult: int) -> int:
    return ((max(1, int(n)) + mult - 1) // mult) * mult


def _tiled_vmem_bytes(shape: tuple[int, ...], dtype: Any) -> int:
    """Compute TPU v7x tiled VMEM footprint in bytes using `pool_alias.storage_size`."""
    dt = jnp.dtype(dtype)
    rows_unit = 8 * (32 // (dt.itemsize * 8))
    rows = _pad_to_multiple(shape[-2] if len(shape) >= 2 else 1, rows_unit)
    cols = _pad_to_multiple(shape[-1], 128)
    leading = shape[:-2] if len(shape) >= 2 else ()
    return int(pool_alias.storage_size((*leading, rows, cols), dt))


@dataclass(frozen=True)
class VMEMBudgetReport:
    """Detailed VMEM budget breakdown for the single-`pallas_call` decode megakernel."""

    batch_size: int
    max_verify_tokens: int
    num_devices: int
    persistent_stream_bytes: int
    persistent_kv_share_bytes: int
    collective_scratch_bytes: int
    attn_sublayer_pool_bytes: int
    indexer_sublayer_pool_bytes: int
    moe_expert_sublayer_pool_bytes: int
    shared_expert_sublayer_pool_bytes: int
    engram_sublayer_pool_bytes: int
    aliased_pool_bytes: int
    aliased_pool_words: int
    total_vmem_bytes: int
    unaliased_total_bytes: int
    soft_budget_bytes: int = VMEM_SOFT_BUDGET_BYTES
    hard_limit_bytes: int = VMEM_HARD_LIMIT_BYTES

    @property
    def within_soft_budget(self) -> bool:
        return self.total_vmem_bytes < self.soft_budget_bytes

    @property
    def within_hard_limit(self) -> bool:
        return self.total_vmem_bytes <= self.hard_limit_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "max_verify_tokens": self.max_verify_tokens,
            "num_devices": self.num_devices,
            "persistent_stream_bytes": self.persistent_stream_bytes,
            "persistent_kv_share_bytes": self.persistent_kv_share_bytes,
            "collective_scratch_bytes": self.collective_scratch_bytes,
            "attn_sublayer_pool_bytes": self.attn_sublayer_pool_bytes,
            "indexer_sublayer_pool_bytes": self.indexer_sublayer_pool_bytes,
            "moe_expert_sublayer_pool_bytes": self.moe_expert_sublayer_pool_bytes,
            "shared_expert_sublayer_pool_bytes": self.shared_expert_sublayer_pool_bytes,
            "engram_sublayer_pool_bytes": self.engram_sublayer_pool_bytes,
            "aliased_pool_bytes": self.aliased_pool_bytes,
            "aliased_pool_words": self.aliased_pool_words,
            "total_vmem_bytes": self.total_vmem_bytes,
            "unaliased_total_bytes": self.unaliased_total_bytes,
            "soft_budget_bytes": self.soft_budget_bytes,
            "hard_limit_bytes": self.hard_limit_bytes,
            "within_soft_budget": self.within_soft_budget,
            "within_hard_limit": self.within_hard_limit,
        }


def compute_vmem_budget(
    cfg: DSV41Config,
    *,
    batch_size: int = 8,
    num_devices: int = 32,
    max_verify_tokens: int = 6,
) -> VMEMBudgetReport:
    """Compute exact tiled VMEM footprint (`pool_alias.storage_size`) for `cfg` at `(batch_size, max_verify_tokens)`."""
    bt = max(1, batch_size * max_verify_tokens)
    ep_lanes = min(num_devices, 8)
    tp_hosts = max(1, num_devices // ep_lanes)
    shared_tp = min(ep_lanes, max(1, cfg.moe_inter_dim // 32))
    n_local_heads = max(1, cfg.n_heads // num_devices)
    n_local_idx_heads = max(1, cfg.index_n_heads // num_devices)
    inter_per_host = max(32, cfg.moe_inter_dim // tp_hosts)
    s_inter_per_lane = max(32, (cfg.n_shared_experts * cfg.moe_inter_dim) // shared_tp)

    # 1. Persistent cross-layer VMEM state
    h_streams_bytes = _tiled_vmem_bytes((bt, cfg.hc_mult, cfg.dim), jnp.bfloat16)
    pre_mix_bytes = _tiled_vmem_bytes((bt, cfg.hc_mult), jnp.float32)
    main_hiddens_bytes = _tiled_vmem_bytes(
        (len(cfg.dspark_target_layer_ids), bt, cfg.dim), jnp.bfloat16
    )
    persistent_stream_bytes = h_streams_bytes + pre_mix_bytes + main_hiddens_bytes

    win_kv_bytes = _tiled_vmem_bytes((batch_size, cfg.sliding_window, cfg.head_dim), jnp.bfloat16)
    topk_kv_bytes = _tiled_vmem_bytes((batch_size, cfg.index_topk, cfg.head_dim), jnp.bfloat16)
    topk_idx_bytes = _tiled_vmem_bytes((bt, cfg.sliding_window + cfg.index_topk), jnp.int32)
    cand_mask_bytes = _tiled_vmem_bytes(
        (bt, cfg.candidate_topk_blocks * cfg.candidate_block_size), jnp.int32
    )
    persistent_kv_share_bytes = win_kv_bytes + topk_kv_bytes + topk_idx_bytes + cand_mask_bytes

    # Collective double-buffered VMEM scratches
    comm_stage1 = _tiled_vmem_bytes((2, min(num_devices, 8), 320, 128), jnp.float32)
    comm_stage2 = (
        _tiled_vmem_bytes((2, max(1, num_devices // 8), 320, 128), jnp.float32)
        if num_devices > 8
        else 0
    )
    collective_scratch_bytes = comm_stage1 + comm_stage2

    # 2. Aliased per-sublayer VMEM scratch pools (reused across all layers via pool_alias.view)
    attn_q_bytes = _tiled_vmem_bytes((bt, cfg.q_lora_rank), jnp.bfloat16) + _tiled_vmem_bytes(
        (bt * n_local_heads, cfg.head_dim), jnp.bfloat16
    )
    attn_scores_bytes = _tiled_vmem_bytes(
        (bt * n_local_heads, cfg.sliding_window + cfg.index_topk), jnp.float32
    )
    attn_out_bytes = _tiled_vmem_bytes(
        (bt, cfg.o_lora_rank), jnp.bfloat16
    ) + _tiled_vmem_bytes((bt, cfg.dim), jnp.float32)
    attn_sublayer_pool_bytes = attn_q_bytes + attn_scores_bytes + attn_out_bytes

    idx_q_bytes = _tiled_vmem_bytes((bt * n_local_idx_heads, cfg.index_head_dim), jnp.bfloat16)
    idx_score_bytes = _tiled_vmem_bytes(
        (bt, cfg.candidate_topk_blocks * cfg.candidate_block_size), jnp.float32
    )
    indexer_sublayer_pool_bytes = idx_q_bytes + idx_score_bytes

    # Single active routed expert projection streamed at a time into VMEM via sequential pool_alias views (w1 -> w3 -> w2)
    w1_u32_bytes = _tiled_vmem_bytes((cfg.dim // 8, inter_per_host), jnp.uint32)
    w1_fp8_bytes = _tiled_vmem_bytes((cfg.dim, inter_per_host), jnp.float8_e4m3fn)
    w2_u32_bytes = _tiled_vmem_bytes((inter_per_host // 8, cfg.dim), jnp.uint32)
    w2_fp8_bytes = _tiled_vmem_bytes((inter_per_host, cfg.dim), jnp.float8_e4m3fn)
    moe_act_bytes = 2 * _tiled_vmem_bytes((bt, inter_per_host), jnp.float32) + _tiled_vmem_bytes(
        (bt, cfg.dim), jnp.float32
    )
    moe_expert_sublayer_pool_bytes = (
        max(w1_u32_bytes + w1_fp8_bytes, w2_u32_bytes + w2_fp8_bytes) + moe_act_bytes
    )

    sh_w13_bytes = 2 * _tiled_vmem_bytes((s_inter_per_lane, cfg.dim), jnp.float8_e4m3fn)
    sh_w2_bytes = _tiled_vmem_bytes((cfg.dim, s_inter_per_lane), jnp.float8_e4m3fn)
    shared_expert_sublayer_pool_bytes = sh_w13_bytes + sh_w2_bytes + moe_act_bytes

    engram_kv_bytes = _tiled_vmem_bytes(
        (bt, (cfg.hc_mult + 1) * cfg.dim), jnp.bfloat16
    ) + _tiled_vmem_bytes(
        (bt, (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads * cfg.engram_head_dim),
        jnp.bfloat16,
    )
    engram_sublayer_pool_bytes = engram_kv_bytes

    aliased_pool_bytes = max(
        attn_sublayer_pool_bytes,
        indexer_sublayer_pool_bytes,
        moe_expert_sublayer_pool_bytes,
        shared_expert_sublayer_pool_bytes,
        engram_sublayer_pool_bytes,
    )
    # Number of (8, 128) uint32 tiles (4096 bytes each)
    aliased_pool_words = max(8, (aliased_pool_bytes + 4095) // 4096 * 8)
    aliased_pool_bytes = aliased_pool_words // 8 * 4096

    total_vmem_bytes = (
        persistent_stream_bytes
        + persistent_kv_share_bytes
        + collective_scratch_bytes
        + aliased_pool_bytes
    )
    unaliased_total_bytes = (
        persistent_stream_bytes
        + persistent_kv_share_bytes
        + collective_scratch_bytes
        + cfg.n_layers
        * (
            attn_sublayer_pool_bytes
            + indexer_sublayer_pool_bytes
            + moe_expert_sublayer_pool_bytes
            + shared_expert_sublayer_pool_bytes
        )
        + len(cfg.engram_layer_ids) * engram_sublayer_pool_bytes
    )

    return VMEMBudgetReport(
        batch_size=batch_size,
        max_verify_tokens=max_verify_tokens,
        num_devices=num_devices,
        persistent_stream_bytes=persistent_stream_bytes,
        persistent_kv_share_bytes=persistent_kv_share_bytes,
        collective_scratch_bytes=collective_scratch_bytes,
        attn_sublayer_pool_bytes=attn_sublayer_pool_bytes,
        indexer_sublayer_pool_bytes=indexer_sublayer_pool_bytes,
        moe_expert_sublayer_pool_bytes=moe_expert_sublayer_pool_bytes,
        shared_expert_sublayer_pool_bytes=shared_expert_sublayer_pool_bytes,
        engram_sublayer_pool_bytes=engram_sublayer_pool_bytes,
        aliased_pool_bytes=aliased_pool_bytes,
        aliased_pool_words=aliased_pool_words,
        total_vmem_bytes=total_vmem_bytes,
        unaliased_total_bytes=unaliased_total_bytes,
    )


def alias_vmem_view(
    vmem_pool_ref: Any,
    shape: tuple[int, ...],
    dtype: Any,
    *,
    offset_bytes: int = 0,
) -> Any:
    """Slice `vmem_pool_ref` (`[words, 128]` `uint32`) to exact `pool_alias.storage_size` and return `pool_alias.view`."""
    dt = jnp.dtype(dtype)
    rows_unit = 8 * (32 // (dt.itemsize * 8))
    rows = _pad_to_multiple(shape[-2] if len(shape) >= 2 else 1, rows_unit)
    cols = _pad_to_multiple(shape[-1], 128)
    leading = shape[:-2] if len(shape) >= 2 else ()
    aligned_shape = (*leading, rows, cols)
    nbytes = int(pool_alias.storage_size(aligned_shape, dt))
    row_start = offset_bytes // 512
    row_end = (offset_bytes + nbytes) // 512
    sub_ref = vmem_pool_ref.at[row_start:row_end, :]
    return pool_alias.view(sub_ref, aligned_shape, dt)


def upcast_mxfp4_pallas(packed_u32: jax.Array) -> jax.Array:
    """In-kernel TPU v7x MXFP4 upcast: `pltpu.bitcast(u32, float4_e2m1fn).astype(float8_e4m3fn)`.

    Args:
        packed_u32: `[K // 8, N]` `uint32` array packed via `pack_mxfp4_for_v7x_bitcast`.

    Returns:
        `[K, N]` `float8_e4m3fn` array with exact E2M1 values.
    """
    return pltpu.bitcast(packed_u32, jnp.float4_e2m1fn).astype(jnp.float8_e4m3fn)


def _cpu_pad_barrier(x: Any) -> Any:
    return jax.lax.optimization_barrier(x) if jax.default_backend() == "cpu" else x


def pallas_upcast_mxfp4_tpu(packed_u32: jax.Array) -> jax.Array:
    """Hardware-compiled TPU v7x Pallas kernel for MXFP4 `uint32 -> float4_e2m1fn -> float8_e4m3fn` upcast."""
    r0, c0 = packed_u32.shape
    r_pad = _pad_to_multiple(r0, 8)
    c_pad = _pad_to_multiple(c0, 128)
    if (r_pad, c_pad) != (r0, c0):
        u32_in = jnp.pad(packed_u32, ((0, r_pad - r0), (0, c_pad - c0)))
    else:
        u32_in = packed_u32

    out_shape = jax.ShapeDtypeStruct((r_pad * 8, c_pad), jnp.float8_e4m3fn)
    pool_words = max(8, (_tiled_vmem_bytes((r_pad * 8, c_pad), jnp.float8_e4m3fn) + 4095) // 4096 * 8)

    def _upcast_kernel(in_ref, out_ref, vmem_pool_ref):
        alias_ref = alias_vmem_view(vmem_pool_ref, (r_pad * 8, c_pad), jnp.float8_e4m3fn)
        fp8_val = upcast_mxfp4_pallas(in_ref[...])
        alias_ref[...] = fp8_val
        out_ref[...] = alias_ref[...]

    res = pl.pallas_call(
        _upcast_kernel,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(1,),
            in_specs=[pl.BlockSpec((r_pad, c_pad), lambda i: (0, 0))],
            out_specs=pl.BlockSpec((r_pad * 8, c_pad), lambda i: (0, 0)),
            scratch_shapes=[pltpu.VMEM((pool_words, 128), jnp.uint32)],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=VMEM_HARD_LIMIT_BYTES,
            disable_bounds_checks=True,
            shape_invariant_numerics=True,
        ),
        interpret=False,
        name="dsv41_mxfp4_upcast_v7x",
    )(u32_in)
    return res[: r0 * 8, :c0]


def pallas_fp4_linear(
    x: jax.Array,
    packed_u32: jax.Array,
    scale_u8: jax.Array,
    *,
    act_block_size: int = 32,
    out_dtype: Any = jnp.bfloat16,
    interpret: bool = True,
) -> jax.Array:
    """In-kernel MXFP4 linear using `upcast_mxfp4_pallas` (`pltpu.bitcast`) + per-32 `act_quant`."""
    orig_shape = x.shape[:-1]
    k = x.shape[-1]
    x2d = x.reshape(-1, k)
    m = x2d.shape[0]
    pad_m = ((max(m, 8) + 7) // 8) * 8
    if pad_m > m:
        x2d = jnp.pad(x2d, ((0, pad_m - m), (0, 0)))
    x2d = _cpu_pad_barrier(x2d)
    x_q, s_x = act_quant(x2d, act_block_size, inplace=False)
    n_k_blocks = k // 32
    w_fp8 = upcast_mxfp4_pallas(packed_u32)
    n = w_fp8.shape[1]
    w_blk = w_fp8.astype(jnp.float32).reshape(n_k_blocks, 32, n)
    x_blk = x_q.astype(jnp.float32).reshape(pad_m, n_k_blocks, 32)
    s_x_f32 = e8m0_to_f32_jax(s_x)
    s_w_f32 = e8m0_to_f32_jax(scale_u8)
    if s_w_f32.shape == (n, n_k_blocks) and n != n_k_blocks:
        s_w_f32 = s_w_f32.T
    x_deq = (x_blk * s_x_f32[..., :, None]).reshape(pad_m, k)
    w_deq = (w_blk * s_w_f32[:, None, :]).reshape(k, n)
    out2d = _cpu_pad_barrier(pad8_matmul_kn(x_deq, w_deq).astype(out_dtype))
    return out2d[:m].reshape(*orig_shape, n)


def _pad8_cast_bf16(x: jax.Array) -> jax.Array:
    """Cast `x` to `bfloat16` on an 8-row-padded `optimization_barrier` tensor so M=1 and M=6 round identically."""
    orig_shape = x.shape
    d = orig_shape[-1]
    x2d = x.reshape(-1, d)
    m = x2d.shape[0]
    pad_m = ((max(m, 8) + 7) // 8) * 8
    if pad_m > m:
        x2d = jnp.pad(x2d, ((0, pad_m - m), (0, 0)))
    x2d = _cpu_pad_barrier(x2d)
    out2d = _cpu_pad_barrier(x2d.astype(jnp.bfloat16))
    return out2d[:m].reshape(orig_shape)


def _pad8_mean_streams(h: jax.Array) -> jax.Array:
    """Compute `jnp.mean(h.astype(jnp.float32), axis=2).astype(jnp.bfloat16)` on an 8-row-padded tensor."""
    orig_prefix = h.shape[:-2]
    hc_m, d = h.shape[-2], h.shape[-1]
    h3d = h.reshape(-1, hc_m, d)
    m = h3d.shape[0]
    pad_m = ((max(m, 8) + 7) // 8) * 8
    if pad_m > m:
        h3d = jnp.pad(h3d, ((0, pad_m - m), (0, 0), (0, 0)))
    h3d = _cpu_pad_barrier(h3d)
    out2d = _cpu_pad_barrier(jnp.mean(h3d.astype(jnp.float32), axis=1).astype(jnp.bfloat16))
    return out2d[:m].reshape(*orig_prefix, d)


def _pad8_wo_a_proj(
    o: jax.Array,
    wo_a: jax.Array,
    *,
    num_devices: int,
    rank: jax.Array,
    cfg: DSV41Config,
    axis_name: str,
) -> jax.Array:
    """Compute `wo_a` projection + intra-group reduction on an 8-row-padded tensor."""
    bsz, seq_len = o.shape[0], o.shape[1]
    m = bsz * seq_len
    pad_m = ((max(m, 8) + 7) // 8) * 8
    if num_devices >= cfg.o_groups:
        ranks_per_group = num_devices // cfg.o_groups
        o_flat = o.reshape(m, -1).astype(jnp.float32)
        if pad_m > m:
            o_flat = jnp.pad(o_flat, ((0, pad_m - m), (0, 0)))
        o_flat = _cpu_pad_barrier(o_flat)
        o_group_partial = pad8_matmul_nk(o_flat, wo_a)
        if ranks_per_group > 1:
            gathered_partials = jax.lax.all_gather(o_group_partial, axis_name, axis=0)
            grp_base = (rank // ranks_per_group) * ranks_per_group
            grp_partials = jax.lax.dynamic_slice_in_dim(gathered_partials, grp_base, ranks_per_group, axis=0)
            grp_partials = _cpu_pad_barrier(grp_partials)
            o_group_full = _cpu_pad_barrier(jnp.sum(grp_partials, axis=0).astype(jnp.bfloat16))
            sub_idx = rank % ranks_per_group
            sub_width = cfg.o_lora_rank // ranks_per_group
            o_mid_pad = jax.lax.dynamic_slice_in_dim(o_group_full, sub_idx * sub_width, sub_width, axis=-1)
        else:
            o_mid_pad = _cpu_pad_barrier(o_group_partial.astype(jnp.bfloat16))
        o_mid_pad = _cpu_pad_barrier(o_mid_pad)
        return o_mid_pad[:m].reshape(bsz, seq_len, -1)
    groups_per_rank = cfg.o_groups // num_devices
    o_grp = o.reshape(m, groups_per_rank, -1).astype(jnp.float32)
    if pad_m > m:
        o_grp = jnp.pad(o_grp, ((0, pad_m - m), (0, 0), (0, 0)))
    o_grp = _cpu_pad_barrier(o_grp)
    o_mid_pad = _cpu_pad_barrier(
        jnp.einsum("mgd,grd->mgr", o_grp, wo_a.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
        .reshape(pad_m, -1)
        .astype(jnp.bfloat16)
    )
    return o_mid_pad[:m].reshape(bsz, seq_len, -1)


def pallas_run_single_expert_fp4(
    x: jax.Array,
    w_e: jax.Array,
    w1_u32: jax.Array,
    w1_s: jax.Array,
    w3_u32: jax.Array,
    w3_s: jax.Array,
    w2_u32: jax.Array,
    w2_s: jax.Array,
    swiglu_limit: float = 10.0,
    *,
    out_dtype: Any = jnp.float32,
    interpret: bool = True,
) -> jax.Array:
    """In-kernel SwiGLU routed expert using `pallas_fp4_linear` (`pltpu.bitcast`)."""
    orig_shape = x.shape[:-1]
    k = x.shape[-1]
    x2d = x.reshape(-1, k)
    we1d = w_e.astype(jnp.float32).reshape(-1)
    m = x2d.shape[0]
    pad_m = ((max(m, 8) + 7) // 8) * 8
    if pad_m > m:
        x2d = jnp.pad(x2d, ((0, pad_m - m), (0, 0)))
        we1d = jnp.pad(we1d, (0, pad_m - m))
    x2d, we1d = _cpu_pad_barrier((x2d, we1d))
    gate = pallas_fp4_linear(
        x2d, w1_u32, w1_s, act_block_size=32, out_dtype=jnp.bfloat16, interpret=interpret
    ).astype(jnp.float32)
    up = pallas_fp4_linear(
        x2d, w3_u32, w3_s, act_block_size=32, out_dtype=jnp.bfloat16, interpret=interpret
    ).astype(jnp.float32)
    if swiglu_limit > 0.0:
        lim = jnp.float32(swiglu_limit)
        up = jnp.clip(up, -lim, lim)
        gate = jnp.minimum(gate, lim)
    h = jax.nn.silu(gate) * up * we1d[:, None]
    out2d = pallas_fp4_linear(
        h.astype(jnp.bfloat16), w2_u32, w2_s, act_block_size=32, out_dtype=out_dtype, interpret=interpret
    )
    out2d = _cpu_pad_barrier(out2d)
    return out2d[:m].reshape(*orig_shape, out2d.shape[-1])


def _megakernel_single_token_in_pallas(
    cfg: DSV41Config,
    w_refs: Mapping[str, Any],
    cache_state: dict[str, jax.Array],
    tokens_1: jax.Array,
    pos_i32: jax.Array,
    engram_hashes_1: jax.Array,
    vmem_pool_ref: Any,
    comm_scratch_refs: Any,
    *,
    num_devices: int,
    sinkhorn_iters: int,
    index_k_mode: str,
    interpret: bool,
    axis_name: str = "tp",
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array], dict[str, int]]:
    """Execute all `cfg.n_layers` inside the Pallas kernel body for one token step `[B, 1]`."""
    bsz = tokens_1.shape[0]
    ep_lanes = min(num_devices, 8)
    shared_tp = min(ep_lanes, cfg.moe_inter_dim // 32)
    rank = jax.lax.axis_index(axis_name)

    token_history = cache_state["token_history"].at[:, pos_i32].set(tokens_1[:, 0])
    engram_hashes = engram_hashes_1

    embed_w = w_refs["embed.weight"][...]
    if embed_w.shape[0] == cfg.padded_vocab_size // num_devices:
        part_v = embed_w.shape[0]
        local_id = tokens_1 - rank * part_v
        in_range = (local_id >= 0) & (local_id < part_v)
        local_emb = jnp.where(in_range[..., None], embed_w[jnp.clip(local_id, 0, part_v - 1)], jnp.bfloat16(0.0))
        h0 = _pad8_cast_bf16(
            all_reduce_sum(
                local_emb.astype(jnp.float32),
                *(comm_scratch_refs if (not interpret and comm_scratch_refs) else ()),
                num_devices=num_devices,
                axis_name=axis_name,
                interpret=interpret,
            )
        )
    else:
        h0 = embed_w[tokens_1]

    h = jnp.repeat(h0[:, :, None, :], cfg.hc_mult, axis=2)
    pre_mix = jnp.zeros((bsz, 1, cfg.hc_mult), dtype=jnp.float32).at[:, :, 0].set(1.0)

    window_kv_all = cache_state["window_kv"]
    compress_kv_all = cache_state["compress_kv"]
    index_k_all = cache_state["index_k"]
    comp_kv_state_all = cache_state["comp_kv_state"]
    comp_score_state_all = cache_state["comp_score_state"]
    comp_kv_ring_all = cache_state.get("comp_kv_ring")
    comp_score_ring_all = cache_state.get("comp_score_ring")
    last_index_k_owner = cache_state["last_index_k_owner"]

    shared_compress_kv: jax.Array | None = None
    shared_topk_idxs: jax.Array | None = None
    shared_candidates: jax.Array | None = None
    main_hiddens: list[jax.Array] = []

    kv_writes = 0
    idx_computations = 0

    cos_swa_all = w_refs["rope.swa.cos"][...]
    sin_swa_all = w_refs["rope.swa.sin"][...]
    cos_yarn_all = w_refs["rope.yarn.cos"][...]
    sin_yarn_all = w_refs["rope.yarn.sin"][...]

    for l_id in range(cfg.n_layers):
        prefix = f"layers.{l_id}."
        r = cfg.compress_ratios[l_id]

        if cfg.has_engram(l_id):
            with scope(f"L{l_id}_engram"):
                ep = prefix + "engram."
                e_idx = cfg.engram_layer_ids.index(l_id)
                e_flat = _lookup_engram_rows(
                    engram_hashes[:, :, e_idx, :],
                    w_refs[ep + "embed.weight"][...],
                    w_refs[ep + "embed.scale"][...],
                    num_embeddings=cfg.engram_num_embeddings[e_idx],
                    sharded_tp=True,
                    num_devices=num_devices,
                    axis_name=axis_name,
                )
                h = engram_forward(
                    h,
                    e_flat,
                    w_refs[ep + "wkv.weight"][...],
                    w_refs[ep + "wkv.scale"][...],
                    w_refs[ep + "q_weight"][...],
                    w_refs[ep + "k_weight"][...],
                    cfg,
                    sharded_tp=True,
                    axis_name=axis_name,
                )

        if l_id in cfg.dspark_target_layer_ids:
            main_hiddens.append(_pad8_mean_streams(h))

        residual = h
        a_pre, a_post, a_comb = hc_mixes(
            h,
            w_refs[prefix + "hc_attn_fn"][...],
            w_refs[prefix + "hc_attn_scale"][...],
            w_refs[prefix + "hc_attn_base"][...],
            hc_mult=cfg.hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            hc_eps=cfg.hc_eps,
            norm_eps=cfg.rms_norm_eps,
        )
        u = rms_norm(hc_pre(h, pre_mix), w_refs[prefix + "attn_norm.weight"][...], cfg.rms_norm_eps)

        ap = prefix + "attn."
        cos_table = cos_yarn_all if r > 0 else cos_swa_all
        sin_table = sin_yarn_all if r > 0 else sin_swa_all
        cos_pos = cos_table[pos_i32][None]
        sin_pos = sin_table[pos_i32][None]

        qr = rms_norm(
            fp8_linear(u, w_refs[ap + "wq_a.weight"][...], w_refs[ap + "wq_a.scale"][...]),
            w_refs[ap + "q_norm.weight"][...],
            cfg.rms_norm_eps,
        )
        q_raw = fp8_linear(qr, w_refs[ap + "wq_b.weight"][...], w_refs[ap + "wq_b.scale"][...])
        n_local_heads = cfg.n_heads // num_devices
        q = q_raw.reshape(bsz, 1, n_local_heads, cfg.head_dim)
        q = apply_rotary_emb(q, cos_pos, sin_pos, cfg.qk_rope_head_dim, inverse=False)

        win = cfg.sliding_window
        kv_raw = rms_norm(
            fp8_linear(u, w_refs[ap + "wkv.weight"][...], w_refs[ap + "wkv.scale"][...]),
            w_refs[ap + "kv_norm.weight"][...],
            cfg.rms_norm_eps,
        )
        kv_rot = apply_rotary_emb(kv_raw, cos_pos, sin_pos, cfg.qk_rope_head_dim, inverse=False)
        kv_quant = act_quant(kv_rot, 32, inplace=True)
        win_layer = window_kv_all[l_id].at[:, pos_i32 % win].set(kv_quant[:, 0])
        window_kv_all = window_kv_all.at[l_id].set(win_layer)

        oldest = (pos_i32 % win) + 1
        ring_order = (oldest + jnp.arange(win, dtype=jnp.int32)) % win
        win_idxs_1d = jnp.where(ring_order > pos_i32, jnp.int32(-1), ring_order)
        topk_idxs = jnp.broadcast_to(win_idxs_1d[None, None, :], (bsz, 1, win))
        full_kv = win_layer

        if r > 0:
            is_kv_src = l_id in cfg.kv_source_layer_ids
            is_idx_src = l_id in cfg.index_source_layer_ids
            kv_src_idx = cfg.kv_source_layer_ids.index(cfg.kv_source_for(l_id))
            compress_len = (pos_i32 + 1) // r

            has_new_latent = jnp.bool_(False)
            latent_pre_rope = jnp.zeros((bsz, 1, cfg.head_dim), dtype=jnp.bfloat16)

            if is_kv_src:
                kv_writes += 1
                cp = ap + "compressor."
                if r == 1:
                    has_new_latent = jnp.bool_(True)
                    wkv_c = w_refs[cp + "wkv.weight"][...].astype(jnp.float32)
                    lat = pad8_matmul_nk(u, wkv_c)
                    latent_pre_rope = rms_norm(
                        _pad8_cast_bf16(lat), w_refs[cp + "norm.weight"][...], cfg.rms_norm_eps
                    )
                else:
                    slot = pos_i32 % r
                    wkv_c = w_refs[cp + "wkv.weight"][...].astype(jnp.float32)
                    wgate_c = w_refs[cp + "wgate.weight"][...].astype(jnp.float32)
                    kv_c = pad8_matmul_nk(u, wkv_c)
                    sc_c = pad8_matmul_nk(u, wgate_c)
                    cur_kv_state = comp_kv_state_all[kv_src_idx].at[:, slot].set(kv_c[:, 0])
                    cur_sc_state = comp_score_state_all[kv_src_idx].at[:, slot].set(sc_c[:, 0])
                    comp_kv_state_all = comp_kv_state_all.at[kv_src_idx].set(cur_kv_state)
                    comp_score_state_all = comp_score_state_all.at[kv_src_idx].set(cur_sc_state)
                    if comp_kv_ring_all is not None and comp_score_ring_all is not None:
                        rslot = pos_i32 % comp_kv_ring_all.shape[2]
                        comp_kv_ring_all = comp_kv_ring_all.at[kv_src_idx, :, rslot].set(kv_c[:, 0])
                        comp_score_ring_all = comp_score_ring_all.at[kv_src_idx, :, rslot].set(sc_c[:, 0])
                    has_new_latent = ((pos_i32 + 1) % r) == 0
                    pooled = jnp.sum(cur_kv_state * jax.nn.softmax(cur_sc_state, axis=1), axis=1, keepdims=True)
                    latent_pre_rope = rms_norm(
                        _pad8_cast_bf16(pooled), w_refs[cp + "norm.weight"][...], cfg.rms_norm_eps
                    )
                shared_compress_kv = compress_kv_all[kv_src_idx]

            if is_idx_src:
                idx_computations += 1
                ip = ap + "indexer."
                if is_kv_src:
                    grp_pos = jnp.maximum(pos_i32 + 1 - r, 0)
                    cos_grp = cos_table[grp_pos][None]
                    sin_grp = sin_table[grp_pos][None]
                    wk_i = w_refs[ip + "wk.weight"][...].astype(jnp.float32)
                    k_raw = _pad8_cast_bf16(pad8_matmul_nk(latent_pre_rope, wk_i))
                    k_normed = rms_norm(k_raw, w_refs[ip + "k_norm.weight"][...], cfg.rms_norm_eps)
                    k_rot = apply_rotary_emb(k_normed, cos_grp, sin_grp, cfg.qk_rope_head_dim, inverse=False)
                    k_quant = fp4_act_quant(k_rot, 32, inplace=True, scale_fmt="e8m0")
                    updated_k = jnp.where(
                        has_new_latent,
                        index_k_all[kv_src_idx].at[:, pos_i32 // r].set(k_quant[:, 0]),
                        index_k_all[kv_src_idx],
                    )
                    index_k_all = index_k_all.at[kv_src_idx].set(updated_k)
                    last_index_k_owner = jnp.where(has_new_latent, jnp.int32(kv_src_idx), last_index_k_owner)

                active_owner = last_index_k_owner if index_k_mode == "reference" else jnp.int32(kv_src_idx)
                active_k_cache = index_k_all[active_owner]

                n_local_idx_heads = cfg.index_n_heads // num_devices
                qi_raw = fp8_linear(
                    qr, w_refs[ip + "wq_b.weight"][...], w_refs[ip + "wq_b.scale"][...]
                ).reshape(bsz, 1, n_local_idx_heads, cfg.index_head_dim)
                qi_rot = apply_rotary_emb(qi_raw, cos_pos, sin_pos, cfg.qk_rope_head_dim, inverse=False)
                qi_quant = fp4_act_quant(qi_rot, 32, inplace=True, scale_fmt="e8m0")

                w_proj = w_refs[ip + "weights_proj.weight"][...].astype(jnp.float32)
                idx_scale = (cfg.index_head_dim**-0.5) * (cfg.index_n_heads**-0.5)
                w_heads = _pad8_cast_bf16(pad8_matmul_nk(u, w_proj) * jnp.float32(idx_scale))

                raw_dot = jnp.einsum(
                    "bshd,btd->bsht",
                    qi_quant.astype(jnp.float32),
                    active_k_cache.astype(jnp.float32),
                    precision=jax.lax.Precision.HIGHEST,
                ).astype(jnp.bfloat16)
                head_weighted = (jnp.maximum(raw_dot, jnp.bfloat16(0.0)) * w_heads[..., None]).astype(jnp.float32)
                score_sum = jnp.sum(head_weighted, axis=2)
                score_sum = all_reduce_sum(
                    score_sum,
                    *(comm_scratch_refs if (not interpret and comm_scratch_refs) else ()),
                    num_devices=num_devices,
                    axis_name=axis_name,
                    interpret=interpret,
                )
                index_score = _pad8_cast_bf16(score_sum).astype(jnp.float32)[:, 0, :]

                t_iota = jnp.arange(active_k_cache.shape[1], dtype=jnp.int32)[None, :]
                index_score = jnp.where(t_iota < compress_len, index_score, -jnp.inf)

                if l_id == cfg.candidate_source_layer_id:
                    shared_candidates = select_candidate_blocks(
                        index_score,
                        jnp.full((bsz,), compress_len, dtype=jnp.int32),
                        cfg.candidate_topk_blocks,
                        cfg.candidate_block_size,
                    )
                elif l_id > cfg.candidate_source_layer_id and shared_candidates is not None:
                    index_score = jnp.where(shared_candidates, index_score, -jnp.inf)

                _, top_pos = jax.lax.top_k(index_score, cfg.index_topk)
                k_keep = jnp.minimum(jnp.int32(cfg.index_topk), compress_len)
                rank_iota = jnp.arange(cfg.index_topk, dtype=jnp.int32)[None, :]
                masked_for_sort = jnp.where(rank_iota < k_keep, top_pos, jnp.int32(1 << 29))
                sorted_pos = jnp.sort(masked_for_sort, axis=-1)
                comp_idxs = jnp.where(
                    (rank_iota < k_keep) & (sorted_pos < compress_len),
                    sorted_pos + win,
                    jnp.int32(-1),
                )
                shared_topk_idxs = comp_idxs[:, None, :]

            if is_kv_src:
                grp_pos = jnp.maximum(pos_i32 + 1 - r, 0)
                cos_grp = cos_table[grp_pos][None]
                sin_grp = sin_table[grp_pos][None]
                lat_rot = apply_rotary_emb(latent_pre_rope, cos_grp, sin_grp, cfg.qk_rope_head_dim, inverse=False)
                lat_quant = fp4_act_quant(lat_rot, 16, inplace=True, scale_fmt="e4m3")
                updated_ckv = jnp.where(
                    has_new_latent,
                    compress_kv_all[kv_src_idx].at[:, pos_i32 // r].set(lat_quant[:, 0]),
                    compress_kv_all[kv_src_idx],
                )
                compress_kv_all = compress_kv_all.at[kv_src_idx].set(updated_ckv)
                shared_compress_kv = updated_ckv

            assert shared_compress_kv is not None and shared_topk_idxs is not None
            full_kv = jnp.concatenate([win_layer, shared_compress_kv], axis=1)
            topk_idxs = jnp.concatenate([topk_idxs, shared_topk_idxs], axis=-1)

        o = sparse_attn(q, full_kv, w_refs[ap + "attn_sink"][...], topk_idxs, cfg.head_dim**-0.5)
        o = apply_rotary_emb(o, cos_pos, sin_pos, cfg.qk_rope_head_dim, inverse=True)

        wo_a = w_refs[ap + "wo_a.weight"][...]
        o_mid = _pad8_wo_a_proj(
            o,
            wo_a,
            num_devices=num_devices,
            rank=rank,
            cfg=cfg,
            axis_name=axis_name,
        )

        attn_out_f32 = fp8_linear(
            o_mid, w_refs[ap + "wo_b.weight"][...], w_refs[ap + "wo_b.scale"][...], out_dtype=jnp.float32
        )
        attn_out_f32 = all_reduce_sum(
            attn_out_f32,
            *(comm_scratch_refs if (not interpret and comm_scratch_refs) else ()),
            num_devices=num_devices,
            axis_name=axis_name,
            interpret=interpret,
        )
        attn_out = _pad8_cast_bf16(attn_out_f32)

        h = hc_post(attn_out, residual, a_post, a_comb)

        residual = h
        f_pre, f_post, f_comb = hc_mixes(
            h,
            w_refs[prefix + "hc_ffn_fn"][...],
            w_refs[prefix + "hc_ffn_scale"][...],
            w_refs[prefix + "hc_ffn_base"][...],
            hc_mult=cfg.hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            hc_eps=cfg.hc_eps,
            norm_eps=cfg.rms_norm_eps,
        )
        v = rms_norm(hc_pre(h, a_pre), w_refs[prefix + "ffn_norm.weight"][...], cfg.rms_norm_eps)

        fp = prefix + "ffn."
        top_w, top_idx = moe_gate(
            v[:, 0, :],
            w_refs[fp + "gate.weight"][...],
            w_refs[fp + "gate.bias"][...],
            cfg.n_activated_experts,
            cfg.route_scale,
        )
        sort_perm = jnp.argsort(top_idx, axis=-1)
        top_idx = jnp.take_along_axis(top_idx, sort_perm, axis=-1)
        top_w = jnp.take_along_axis(top_w, sort_perm, axis=-1)

        # Dynamic active-expert streaming from HBM Refs + in-kernel pltpu.bitcast MXFP4 upcast
        w1_u32_ref = w_refs[fp + "experts.w1.u32"]
        w1_s_ref = w_refs[fp + "experts.w1.scale"]
        w3_u32_ref = w_refs[fp + "experts.w3.u32"]
        w3_s_ref = w_refs[fp + "experts.w3.scale"]
        w2_u32_ref = w_refs[fp + "experts.w2.u32"]
        w2_s_ref = w_refs[fp + "experts.w2.scale"]
        n_local_exp = w1_u32_ref.shape[0]
        lane = rank % ep_lanes
        exp_start = lane * n_local_exp
        exp_out_dtype = jnp.bfloat16 if num_devices <= 8 else jnp.float32

        if interpret:
            routed_acc = jnp.zeros((bsz, cfg.dim), dtype=jnp.float32)
            for k_sel in range(cfg.n_activated_experts):
                e_global = top_idx[:, k_sel]
                w_sel = top_w[:, k_sel]
                e_local = e_global - exp_start
                active_mask = (e_local >= 0) & (e_local < n_local_exp)
                safe_local = jnp.clip(e_local, 0, n_local_exp - 1)

                for b_i in range(bsz):
                    eb = safe_local[b_i]
                    out_b = pallas_run_single_expert_fp4(
                        v[b_i : b_i + 1, 0, :],
                        w_sel[b_i : b_i + 1],
                        w1_u32_ref[eb, :, :],
                        w1_s_ref[eb, :, :],
                        w3_u32_ref[eb, :, :],
                        w3_s_ref[eb, :, :],
                        w2_u32_ref[eb, :, :],
                        w2_s_ref[eb, :, :],
                        cfg.swiglu_limit,
                        out_dtype=exp_out_dtype,
                        interpret=True,
                    )[0].astype(jnp.float32)
                    routed_acc = routed_acc.at[b_i].add(jnp.where(active_mask[b_i], out_b, 0.0))
        else:
            def _scan_active_slot(acc: jax.Array, slot_inputs: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, None]:
                e_global, w_sel = slot_inputs
                e_local = e_global - exp_start
                active_mask = (e_local >= 0) & (e_local < n_local_exp)
                safe_local = jnp.clip(e_local, 0, n_local_exp - 1)
                cur = acc
                for b_i in range(bsz):
                    eb = safe_local[b_i]
                    vb = v[b_i : b_i + 1, 0, :]
                    wb = w_sel[b_i : b_i + 1]

                    def _run_active(_: None) -> jax.Array:
                        return pallas_run_single_expert_fp4(
                            vb,
                            wb,
                            w1_u32_ref[eb],
                            w1_s_ref[eb],
                            w3_u32_ref[eb],
                            w3_s_ref[eb],
                            w2_u32_ref[eb],
                            w2_s_ref[eb],
                            cfg.swiglu_limit,
                            out_dtype=exp_out_dtype,
                            interpret=False,
                        )[0].astype(jnp.float32)

                    out_b = jax.lax.cond(
                        active_mask[b_i],
                        _run_active,
                        lambda _: jnp.zeros((cfg.dim,), dtype=jnp.float32),
                        operand=None,
                    )
                    cur = cur.at[b_i].add(out_b)
                return cur, None

            routed_acc, _ = jax.lax.scan(
                _scan_active_slot,
                jnp.zeros((bsz, cfg.dim), dtype=jnp.float32),
                (top_idx.T, top_w.T),
            )

        routed_acc = all_reduce_sum(
            routed_acc,
            *(comm_scratch_refs if (not interpret and comm_scratch_refs) else ()),
            num_devices=num_devices,
            axis_name=axis_name,
            interpret=interpret,
        )

        sp = fp + "shared_experts."
        shared_part = run_shared_expert_fp8(
            v[:, 0, :],
            w_refs[sp + "w1.weight"][...],
            w_refs[sp + "w1.scale"][...],
            w_refs[sp + "w3.weight"][...],
            w_refs[sp + "w3.scale"][...],
            w_refs[sp + "w2.weight"][...],
            w_refs[sp + "w2.scale"][...],
            cfg.swiglu_limit,
            out_dtype=jnp.float32,
        ).astype(jnp.float32)
        shared_rep = num_devices // shared_tp
        shared_full = _pad8_cast_bf16(
            all_reduce_sum(
                shared_part,
                *(comm_scratch_refs if (not interpret and comm_scratch_refs) else ()),
                num_devices=num_devices,
                axis_name=axis_name,
                interpret=interpret,
            )
            / jnp.float32(shared_rep)
        ).astype(jnp.float32)

        ffn_sum = routed_acc + shared_full
        ffn_out = _pad8_cast_bf16(ffn_sum)[:, None, :]
        h = hc_post(ffn_out, residual, f_post, f_comb)
        pre_mix = f_pre

    y_final = rms_norm(hc_pre(h, pre_mix), w_refs["norm.weight"][...], cfg.rms_norm_eps)
    head_w = w_refs["head.weight"][...].astype(jnp.float32)
    logits_local = pad8_matmul_nk(y_final[:, 0, :], head_w)
    if head_w.shape[0] == cfg.padded_vocab_size // num_devices:
        logits_full = jax.lax.all_gather(logits_local, axis_name, axis=-1, tiled=True)
    else:
        logits_full = logits_local
    logits = logits_full[:, : cfg.vocab_size]
    main_hidden = jnp.concatenate(main_hiddens, axis=-1)[:, 0, :]

    new_cache = {
        **cache_state,
        "window_kv": window_kv_all,
        "compress_kv": compress_kv_all,
        "index_k": index_k_all,
        "comp_kv_state": comp_kv_state_all,
        "comp_score_state": comp_score_state_all,
        "last_index_k_owner": last_index_k_owner,
        "token_history": token_history,
    }
    if comp_kv_ring_all is not None and comp_score_ring_all is not None:
        new_cache["comp_kv_ring"] = comp_kv_ring_all
        new_cache["comp_score_ring"] = comp_score_ring_all
    stats = {"kv_writes": kv_writes, "idx_computations": idx_computations}
    return logits, main_hidden, new_cache, stats


def _megakernel_block_in_pallas(
    cfg: DSV41Config,
    w_refs: Mapping[str, Any],
    cache_state: dict[str, jax.Array],
    tokens_q: jax.Array,
    pos0_i32: jax.Array,
    engram_hashes_q: jax.Array,
    vmem_pool_ref: Any,
    comm_scratch_refs: Any,
    *,
    num_devices: int,
    sinkhorn_iters: int,
    index_k_mode: str,
    interpret: bool,
    axis_name: str = "tp",
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Execute all `cfg.n_layers` in parallel across `q_len` tokens `[B, q_len]` with causal KV/indexer steps."""
    bsz, q_len = tokens_q.shape
    n_tokens = bsz * q_len
    ep_lanes = min(num_devices, 8)
    shared_tp = min(ep_lanes, cfg.moe_inter_dim // 32)
    rank = jax.lax.axis_index(axis_name)

    token_history = cache_state["token_history"]
    for k in range(q_len):
        token_history = token_history.at[:, pos0_i32 + k].set(tokens_q[:, k])

    embed_w = w_refs["embed.weight"][...]
    if embed_w.shape[0] == cfg.padded_vocab_size // num_devices:
        part_v = embed_w.shape[0]
        local_id = tokens_q - rank * part_v
        in_range = (local_id >= 0) & (local_id < part_v)
        local_emb = jnp.where(in_range[..., None], embed_w[jnp.clip(local_id, 0, part_v - 1)], jnp.bfloat16(0.0))
        h0 = _pad8_cast_bf16(
            all_reduce_sum(
                local_emb.astype(jnp.float32),
                *(comm_scratch_refs if (not interpret and comm_scratch_refs) else ()),
                num_devices=num_devices,
                axis_name=axis_name,
                interpret=interpret,
            )
        )
    else:
        h0 = embed_w[tokens_q]

    h = jnp.repeat(h0[:, :, None, :], cfg.hc_mult, axis=2)
    pre_mix = jnp.zeros((bsz, q_len, cfg.hc_mult), dtype=jnp.float32).at[:, :, 0].set(1.0)

    window_kv_all = cache_state["window_kv"]
    compress_kv_all = cache_state["compress_kv"]
    index_k_all = cache_state["index_k"]
    comp_kv_state_all = cache_state["comp_kv_state"]
    comp_score_state_all = cache_state["comp_score_state"]
    comp_kv_ring_all = cache_state.get("comp_kv_ring")
    comp_score_ring_all = cache_state.get("comp_score_ring")
    last_index_k_owner = cache_state["last_index_k_owner"]
    last_r1_owner = jnp.int32(len(cfg.kv_source_layer_ids) - 1)
    owner_per_k: list[jax.Array] = [
        last_index_k_owner if k_i == 0 else last_r1_owner for k_i in range(q_len)
    ]

    shared_compress_kv: jax.Array | None = None
    shared_topk_idxs_steps: list[jax.Array] | None = None
    shared_candidates_steps: list[jax.Array] | None = None
    main_hiddens: list[jax.Array] = []

    cos_swa_all = w_refs["rope.swa.cos"][...]
    sin_swa_all = w_refs["rope.swa.sin"][...]
    cos_yarn_all = w_refs["rope.yarn.cos"][...]
    sin_yarn_all = w_refs["rope.yarn.sin"][...]
    pos_vec = pos0_i32 + jnp.arange(q_len, dtype=jnp.int32)

    for l_id in range(cfg.n_layers):
        prefix = f"layers.{l_id}."
        r = cfg.compress_ratios[l_id]

        if cfg.has_engram(l_id):
            ep = prefix + "engram."
            e_idx = cfg.engram_layer_ids.index(l_id)
            e_flat = _lookup_engram_rows(
                engram_hashes_q[:, :, e_idx, :],
                w_refs[ep + "embed.weight"][...],
                w_refs[ep + "embed.scale"][...],
                num_embeddings=cfg.engram_num_embeddings[e_idx],
                sharded_tp=True,
                num_devices=num_devices,
                axis_name=axis_name,
            )
            h = engram_forward(
                h,
                e_flat,
                w_refs[ep + "wkv.weight"][...],
                w_refs[ep + "wkv.scale"][...],
                w_refs[ep + "q_weight"][...],
                w_refs[ep + "k_weight"][...],
                cfg,
                sharded_tp=True,
                axis_name=axis_name,
            )

        if l_id in cfg.dspark_target_layer_ids:
            main_hiddens.append(_pad8_mean_streams(h))

        residual = h
        a_pre, a_post, a_comb = hc_mixes(
            h,
            w_refs[prefix + "hc_attn_fn"][...],
            w_refs[prefix + "hc_attn_scale"][...],
            w_refs[prefix + "hc_attn_base"][...],
            hc_mult=cfg.hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            hc_eps=cfg.hc_eps,
            norm_eps=cfg.rms_norm_eps,
        )
        u = rms_norm(hc_pre(h, pre_mix), w_refs[prefix + "attn_norm.weight"][...], cfg.rms_norm_eps)

        ap = prefix + "attn."
        cos_table = cos_yarn_all if r > 0 else cos_swa_all
        sin_table = sin_yarn_all if r > 0 else sin_swa_all
        cos_pos = cos_table[pos_vec][None, :, :]
        sin_pos = sin_table[pos_vec][None, :, :]

        qr = rms_norm(
            fp8_linear(u, w_refs[ap + "wq_a.weight"][...], w_refs[ap + "wq_a.scale"][...]),
            w_refs[ap + "q_norm.weight"][...],
            cfg.rms_norm_eps,
        )
        q_raw = fp8_linear(qr, w_refs[ap + "wq_b.weight"][...], w_refs[ap + "wq_b.scale"][...])
        n_local_heads = cfg.n_heads // num_devices
        q = q_raw.reshape(bsz, q_len, n_local_heads, cfg.head_dim)
        q = apply_rotary_emb(q, cos_pos, sin_pos, cfg.qk_rope_head_dim, inverse=False)

        win = cfg.sliding_window
        kv_raw = rms_norm(
            fp8_linear(u, w_refs[ap + "wkv.weight"][...], w_refs[ap + "wkv.scale"][...]),
            w_refs[ap + "kv_norm.weight"][...],
            cfg.rms_norm_eps,
        )
        kv_rot = apply_rotary_emb(kv_raw, cos_pos, sin_pos, cfg.qk_rope_head_dim, inverse=False)
        kv_quant = act_quant(kv_rot, 32, inplace=True)

        is_kv_src = (r > 0) and (l_id in cfg.kv_source_layer_ids)
        is_idx_src = (r > 0) and (l_id in cfg.index_source_layer_ids)
        kv_src_idx = cfg.kv_source_layer_ids.index(cfg.kv_source_for(l_id)) if r > 0 else 0

        kv_c_all = None
        sc_c_all = None
        lat_r1_all = None
        if is_kv_src:
            cp = ap + "compressor."
            wkv_c = w_refs[cp + "wkv.weight"][...].astype(jnp.float32)
            kv_c_all = pad8_matmul_nk(u, wkv_c)
            if r == 1:
                lat_r1_all = rms_norm(
                    _pad8_cast_bf16(kv_c_all), w_refs[cp + "norm.weight"][...], cfg.rms_norm_eps
                )
            else:
                wgate_c = w_refs[cp + "wgate.weight"][...].astype(jnp.float32)
                sc_c_all = pad8_matmul_nk(u, wgate_c)

        qi_quant_all = None
        w_heads_all = None
        if is_idx_src:
            ip = ap + "indexer."
            n_local_idx_heads = cfg.index_n_heads // num_devices
            qi_raw_all = fp8_linear(
                qr, w_refs[ip + "wq_b.weight"][...], w_refs[ip + "wq_b.scale"][...]
            ).reshape(bsz, q_len, n_local_idx_heads, cfg.index_head_dim)
            qi_rot_all = apply_rotary_emb(qi_raw_all, cos_pos, sin_pos, cfg.qk_rope_head_dim, inverse=False)
            qi_quant_all = fp4_act_quant(qi_rot_all, 32, inplace=True, scale_fmt="e8m0")

            w_proj = w_refs[ip + "weights_proj.weight"][...].astype(jnp.float32)
            idx_scale = (cfg.index_head_dim**-0.5) * (cfg.index_n_heads**-0.5)
            w_heads_all = _pad8_cast_bf16(pad8_matmul_nk(u, w_proj) * jnp.float32(idx_scale))

        win_layer = window_kv_all[l_id]
        o_steps: list[jax.Array] = []
        new_topk_steps: list[jax.Array] = []
        new_cand_steps: list[jax.Array] = []
        attn_sink_w = w_refs[ap + "attn_sink"][...]

        if r == 0:
            for k in range(q_len):
                pos_k = pos0_i32 + k
                win_layer = win_layer.at[:, pos_k % win].set(kv_quant[:, k])
                oldest = (pos_k % win) + 1
                ring_order = (oldest + jnp.arange(win, dtype=jnp.int32)) % win
                win_idxs_1d = jnp.where(ring_order > pos_k, jnp.int32(-1), ring_order)
                topk_idxs_k = jnp.broadcast_to(win_idxs_1d[None, None, :], (bsz, 1, win))
                o_k = sparse_attn(q[:, k : k + 1], win_layer, attn_sink_w, topk_idxs_k, cfg.head_dim**-0.5)
                o_steps.append(o_k)
        else:
            has_new_latent_steps: list[jax.Array] = []
            latent_pre_rope_steps: list[jax.Array] = []
            score_sum_local_steps: list[jax.Array] = []

            for k in range(q_len):
                pos_k = pos0_i32 + k
                has_new_latent = jnp.bool_(False)
                latent_pre_rope = jnp.zeros((bsz, 1, cfg.head_dim), dtype=jnp.bfloat16)

                if is_kv_src:
                    cp = ap + "compressor."
                    if r == 1:
                        assert lat_r1_all is not None
                        has_new_latent = jnp.bool_(True)
                        latent_pre_rope = lat_r1_all[:, k : k + 1, :]
                    else:
                        assert kv_c_all is not None and sc_c_all is not None
                        slot = pos_k % r
                        cur_kv_state = comp_kv_state_all[kv_src_idx].at[:, slot].set(kv_c_all[:, k])
                        cur_sc_state = comp_score_state_all[kv_src_idx].at[:, slot].set(sc_c_all[:, k])
                        comp_kv_state_all = comp_kv_state_all.at[kv_src_idx].set(cur_kv_state)
                        comp_score_state_all = comp_score_state_all.at[kv_src_idx].set(cur_sc_state)
                        if comp_kv_ring_all is not None and comp_score_ring_all is not None:
                            rslot = pos_k % comp_kv_ring_all.shape[2]
                            comp_kv_ring_all = comp_kv_ring_all.at[kv_src_idx, :, rslot].set(kv_c_all[:, k])
                            comp_score_ring_all = comp_score_ring_all.at[kv_src_idx, :, rslot].set(sc_c_all[:, k])
                        has_new_latent = ((pos_k + 1) % r) == 0
                        pooled = jnp.sum(cur_kv_state * jax.nn.softmax(cur_sc_state, axis=1), axis=1, keepdims=True)
                        latent_pre_rope = rms_norm(
                            _pad8_cast_bf16(pooled), w_refs[cp + "norm.weight"][...], cfg.rms_norm_eps
                        )
                has_new_latent_steps.append(has_new_latent)
                latent_pre_rope_steps.append(latent_pre_rope)

                if is_idx_src:
                    assert qi_quant_all is not None and w_heads_all is not None
                    ip = ap + "indexer."
                    if is_kv_src:
                        grp_pos = jnp.maximum(pos_k + 1 - r, 0)
                        cos_grp = cos_table[grp_pos][None]
                        sin_grp = sin_table[grp_pos][None]
                        wk_i = w_refs[ip + "wk.weight"][...].astype(jnp.float32)
                        k_raw = _pad8_cast_bf16(pad8_matmul_nk(latent_pre_rope, wk_i))
                        k_normed = rms_norm(k_raw, w_refs[ip + "k_norm.weight"][...], cfg.rms_norm_eps)
                        k_rot = apply_rotary_emb(k_normed, cos_grp, sin_grp, cfg.qk_rope_head_dim, inverse=False)
                        k_quant = fp4_act_quant(k_rot, 32, inplace=True, scale_fmt="e8m0")
                        updated_k = jnp.where(
                            has_new_latent,
                            index_k_all[kv_src_idx].at[:, pos_k // r].set(k_quant[:, 0]),
                            index_k_all[kv_src_idx],
                        )
                        index_k_all = index_k_all.at[kv_src_idx].set(updated_k)
                        owner_per_k[k] = jnp.where(has_new_latent, jnp.int32(kv_src_idx), owner_per_k[k])
                        last_index_k_owner = owner_per_k[k]

                    active_owner = owner_per_k[k] if index_k_mode == "reference" else jnp.int32(kv_src_idx)
                    active_k_cache = index_k_all[active_owner]

                    raw_dot = jnp.einsum(
                        "bshd,btd->bsht",
                        qi_quant_all[:, k : k + 1].astype(jnp.float32),
                        active_k_cache.astype(jnp.float32),
                        precision=jax.lax.Precision.HIGHEST,
                    ).astype(jnp.bfloat16)
                    head_weighted = (
                        jnp.maximum(raw_dot, jnp.bfloat16(0.0)) * w_heads_all[:, k : k + 1, :, None]
                    ).astype(jnp.float32)
                    score_sum_local_steps.append(jnp.sum(head_weighted, axis=2))

            score_sum_reduced_all = None
            if is_idx_src:
                score_sum_cat = jnp.concatenate(score_sum_local_steps, axis=1)
                score_sum_reduced_all = _pad8_cast_bf16(
                    all_reduce_sum(
                        score_sum_cat,
                        *(comm_scratch_refs if (not interpret and comm_scratch_refs) else ()),
                        num_devices=num_devices,
                        axis_name=axis_name,
                        interpret=interpret,
                    )
                ).astype(jnp.float32)

            for k in range(q_len):
                pos_k = pos0_i32 + k
                compress_len_k = (pos_k + 1) // r
                win_layer = win_layer.at[:, pos_k % win].set(kv_quant[:, k])
                oldest = (pos_k % win) + 1
                ring_order = (oldest + jnp.arange(win, dtype=jnp.int32)) % win
                win_idxs_1d = jnp.where(ring_order > pos_k, jnp.int32(-1), ring_order)
                topk_idxs_k = jnp.broadcast_to(win_idxs_1d[None, None, :], (bsz, 1, win))

                if is_idx_src:
                    assert score_sum_reduced_all is not None
                    index_score = score_sum_reduced_all[:, k, :]
                    t_iota = jnp.arange(index_score.shape[1], dtype=jnp.int32)[None, :]
                    index_score = jnp.where(t_iota < compress_len_k, index_score, -jnp.inf)

                    if l_id == cfg.candidate_source_layer_id:
                        cand_k = select_candidate_blocks(
                            index_score,
                            jnp.full((bsz,), compress_len_k, dtype=jnp.int32),
                            cfg.candidate_topk_blocks,
                            cfg.candidate_block_size,
                        )
                        new_cand_steps.append(cand_k)
                    elif l_id > cfg.candidate_source_layer_id and shared_candidates_steps is not None:
                        index_score = jnp.where(shared_candidates_steps[k], index_score, -jnp.inf)

                    _, top_pos = jax.lax.top_k(index_score, cfg.index_topk)
                    k_keep = jnp.minimum(jnp.int32(cfg.index_topk), compress_len_k)
                    rank_iota = jnp.arange(cfg.index_topk, dtype=jnp.int32)[None, :]
                    masked_for_sort = jnp.where(rank_iota < k_keep, top_pos, jnp.int32(1 << 29))
                    sorted_pos = jnp.sort(masked_for_sort, axis=-1)
                    comp_idxs = jnp.where(
                        (rank_iota < k_keep) & (sorted_pos < compress_len_k),
                        sorted_pos + win,
                        jnp.int32(-1),
                    )
                    new_topk_steps.append(comp_idxs[:, None, :])

                if is_kv_src:
                    has_new_latent = has_new_latent_steps[k]
                    latent_pre_rope = latent_pre_rope_steps[k]
                    grp_pos = jnp.maximum(pos_k + 1 - r, 0)
                    cos_grp = cos_table[grp_pos][None]
                    sin_grp = sin_table[grp_pos][None]
                    lat_rot = apply_rotary_emb(latent_pre_rope, cos_grp, sin_grp, cfg.qk_rope_head_dim, inverse=False)
                    lat_quant = fp4_act_quant(lat_rot, 16, inplace=True, scale_fmt="e4m3")
                    updated_ckv = jnp.where(
                        has_new_latent,
                        compress_kv_all[kv_src_idx].at[:, pos_k // r].set(lat_quant[:, 0]),
                        compress_kv_all[kv_src_idx],
                    )
                    compress_kv_all = compress_kv_all.at[kv_src_idx].set(updated_ckv)
                    shared_compress_kv = updated_ckv

                cur_topk_k = new_topk_steps[k] if is_idx_src else shared_topk_idxs_steps[k]
                assert shared_compress_kv is not None and cur_topk_k is not None
                full_kv_k = jnp.concatenate([win_layer, shared_compress_kv], axis=1)
                topk_idxs_k = jnp.concatenate([topk_idxs_k, cur_topk_k], axis=-1)
                o_k = sparse_attn(q[:, k : k + 1], full_kv_k, attn_sink_w, topk_idxs_k, cfg.head_dim**-0.5)
                o_steps.append(o_k)

        window_kv_all = window_kv_all.at[l_id].set(win_layer)
        if is_idx_src:
            shared_topk_idxs_steps = new_topk_steps
            if l_id == cfg.candidate_source_layer_id:
                shared_candidates_steps = new_cand_steps

        o = jnp.concatenate(o_steps, axis=1)
        o = apply_rotary_emb(o, cos_pos, sin_pos, cfg.qk_rope_head_dim, inverse=True)

        wo_a = w_refs[ap + "wo_a.weight"][...]
        o_mid = _pad8_wo_a_proj(
            o,
            wo_a,
            num_devices=num_devices,
            rank=rank,
            cfg=cfg,
            axis_name=axis_name,
        )

        attn_out_f32 = fp8_linear(
            o_mid, w_refs[ap + "wo_b.weight"][...], w_refs[ap + "wo_b.scale"][...], out_dtype=jnp.float32
        )
        attn_out_f32 = all_reduce_sum(
            attn_out_f32,
            *(comm_scratch_refs if (not interpret and comm_scratch_refs) else ()),
            num_devices=num_devices,
            axis_name=axis_name,
            interpret=interpret,
        )
        attn_out = _pad8_cast_bf16(attn_out_f32)

        h = hc_post(attn_out, residual, a_post, a_comb)

        residual = h
        f_pre, f_post, f_comb = hc_mixes(
            h,
            w_refs[prefix + "hc_ffn_fn"][...],
            w_refs[prefix + "hc_ffn_scale"][...],
            w_refs[prefix + "hc_ffn_base"][...],
            hc_mult=cfg.hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            hc_eps=cfg.hc_eps,
            norm_eps=cfg.rms_norm_eps,
        )
        v = rms_norm(hc_pre(h, a_pre), w_refs[prefix + "ffn_norm.weight"][...], cfg.rms_norm_eps)
        v_flat = v.reshape(n_tokens, cfg.dim)

        fp = prefix + "ffn."
        top_w, top_idx = moe_gate(
            v_flat,
            w_refs[fp + "gate.weight"][...],
            w_refs[fp + "gate.bias"][...],
            cfg.n_activated_experts,
            cfg.route_scale,
        )
        sort_perm = jnp.argsort(top_idx, axis=-1)
        top_idx = jnp.take_along_axis(top_idx, sort_perm, axis=-1)
        top_w = jnp.take_along_axis(top_w, sort_perm, axis=-1)

        w1_u32_ref = w_refs[fp + "experts.w1.u32"]
        w1_s_ref = w_refs[fp + "experts.w1.scale"]
        w3_u32_ref = w_refs[fp + "experts.w3.u32"]
        w3_s_ref = w_refs[fp + "experts.w3.scale"]
        w2_u32_ref = w_refs[fp + "experts.w2.u32"]
        w2_s_ref = w_refs[fp + "experts.w2.scale"]
        n_local_exp = w1_u32_ref.shape[0]
        lane = rank % ep_lanes
        exp_start = lane * n_local_exp
        exp_out_dtype = jnp.bfloat16 if num_devices <= 8 else jnp.float32

        if interpret:
            routed_acc_flat = jnp.zeros((n_tokens, cfg.dim), dtype=jnp.float32)
            for k_sel in range(cfg.n_activated_experts):
                e_global = top_idx[:, k_sel]
                w_sel = top_w[:, k_sel]
                e_local = e_global - exp_start
                active_mask = (e_local >= 0) & (e_local < n_local_exp)
                safe_local = jnp.clip(e_local, 0, n_local_exp - 1)

                for t_i in range(n_tokens):
                    eb = safe_local[t_i]
                    out_t = pallas_run_single_expert_fp4(
                        v_flat[t_i : t_i + 1],
                        w_sel[t_i : t_i + 1],
                        w1_u32_ref[eb, :, :],
                        w1_s_ref[eb, :, :],
                        w3_u32_ref[eb, :, :],
                        w3_s_ref[eb, :, :],
                        w2_u32_ref[eb, :, :],
                        w2_s_ref[eb, :, :],
                        cfg.swiglu_limit,
                        out_dtype=exp_out_dtype,
                        interpret=True,
                    )[0].astype(jnp.float32)
                    routed_acc_flat = routed_acc_flat.at[t_i].add(jnp.where(active_mask[t_i], out_t, 0.0))
        else:
            def _scan_active_slot(acc: jax.Array, slot_inputs: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, None]:
                e_global, w_sel = slot_inputs
                e_local = e_global - exp_start
                active_mask = (e_local >= 0) & (e_local < n_local_exp)
                safe_local = jnp.clip(e_local, 0, n_local_exp - 1)
                cur = acc
                for t_i in range(n_tokens):
                    eb = safe_local[t_i]
                    vb = v_flat[t_i : t_i + 1]
                    wb = w_sel[t_i : t_i + 1]

                    def _run_active(_: None) -> jax.Array:
                        return pallas_run_single_expert_fp4(
                            vb,
                            wb,
                            w1_u32_ref[eb],
                            w1_s_ref[eb],
                            w3_u32_ref[eb],
                            w3_s_ref[eb],
                            w2_u32_ref[eb],
                            w2_s_ref[eb],
                            cfg.swiglu_limit,
                            out_dtype=exp_out_dtype,
                            interpret=False,
                        )[0].astype(jnp.float32)

                    out_t = jax.lax.cond(
                        active_mask[t_i],
                        _run_active,
                        lambda _: jnp.zeros((cfg.dim,), dtype=jnp.float32),
                        operand=None,
                    )
                    cur = cur.at[t_i].add(out_t)
                return cur, None

            routed_acc_flat, _ = jax.lax.scan(
                _scan_active_slot,
                jnp.zeros((n_tokens, cfg.dim), dtype=jnp.float32),
                (top_idx.T, top_w.T),
            )
        routed_acc = all_reduce_sum(
            routed_acc_flat,
            *(comm_scratch_refs if (not interpret and comm_scratch_refs) else ()),
            num_devices=num_devices,
            axis_name=axis_name,
            interpret=interpret,
        )

        sp = fp + "shared_experts."
        shared_part = run_shared_expert_fp8(
            v_flat,
            w_refs[sp + "w1.weight"][...],
            w_refs[sp + "w1.scale"][...],
            w_refs[sp + "w3.weight"][...],
            w_refs[sp + "w3.scale"][...],
            w_refs[sp + "w2.weight"][...],
            w_refs[sp + "w2.scale"][...],
            cfg.swiglu_limit,
            out_dtype=jnp.float32,
        ).astype(jnp.float32)
        shared_rep = num_devices // shared_tp
        shared_full = _pad8_cast_bf16(
            all_reduce_sum(
                shared_part,
                *(comm_scratch_refs if (not interpret and comm_scratch_refs) else ()),
                num_devices=num_devices,
                axis_name=axis_name,
                interpret=interpret,
            )
            / jnp.float32(shared_rep)
        ).astype(jnp.float32)

        ffn_out = _pad8_cast_bf16(routed_acc + shared_full).reshape(bsz, q_len, cfg.dim)
        h = hc_post(ffn_out, residual, f_post, f_comb)
        pre_mix = f_pre

    y_final = rms_norm(hc_pre(h, pre_mix), w_refs["norm.weight"][...], cfg.rms_norm_eps)
    head_w = w_refs["head.weight"][...].astype(jnp.float32)
    logits_local = pad8_matmul_nk(y_final, head_w)
    if head_w.shape[0] == cfg.padded_vocab_size // num_devices:
        logits_full = jax.lax.all_gather(logits_local, axis_name, axis=-1, tiled=True)
    else:
        logits_full = logits_local
    logits_seq = logits_full[:, :, : cfg.vocab_size]
    hidden_seq = jnp.concatenate(main_hiddens, axis=-1)

    new_cache = {
        **cache_state,
        "window_kv": window_kv_all,
        "compress_kv": compress_kv_all,
        "index_k": index_k_all,
        "comp_kv_state": comp_kv_state_all,
        "comp_score_state": comp_score_state_all,
        "last_index_k_owner": last_index_k_owner,
        "token_history": token_history,
    }
    if comp_kv_ring_all is not None and comp_score_ring_all is not None:
        new_cache["comp_kv_ring"] = comp_kv_ring_all
        new_cache["comp_score_ring"] = comp_score_ring_all
    return logits_seq, hidden_seq, new_cache


def make_megakernel_decode(
    cfg: DSV41Config,
    mesh: Mesh,
    *,
    sinkhorn_iters: int | None = None,
    index_k_mode: str | None = None,
    interpret: bool | None = None,
    donate: bool = False,
):
    """Compile the single-`pl.pallas_call` sharded decode & verify megakernel over `mesh` (`"tp"`).

    Args:
        cfg: `DSV41Config` (`full_config` or `tiny_config`).
        mesh: 1-D JAX `Mesh` with axis `"tp"` (`create_v7x_mesh`).
        sinkhorn_iters: Number of Sinkhorn iterations (`20` default).
        index_k_mode: `"reference"` or `"intended"`.
        interpret: If `None`, defaults to `True` on CPU backend and `False` on TPU v7x.
        donate: Whether to donate the `cache` buffer argument (`argnum=1`).

    Returns:
        Compiled callable `fn(sharded_weights, cache, tokens, start_pos) -> (logits, main_hidden, new_cache)`.
    """
    num_devices = mesh.size
    s_iters = cfg.hc_sinkhorn_iters if sinkhorn_iters is None else int(sinkhorn_iters)
    ik_mode = cfg.index_k_mode if index_k_mode is None else index_k_mode
    use_interpret = (jax.default_backend() == "cpu") if interpret is None else bool(interpret)

    budget = compute_vmem_budget(cfg, batch_size=8, num_devices=num_devices, max_verify_tokens=6)
    if not budget.within_soft_budget:
        raise ValueError(
            f"Megakernel VMEM budget {budget.total_vmem_bytes} exceeds 48 MiB soft budget ({budget.soft_budget_bytes})"
        )

    def _local_fn(sharded_weights, cache, tokens, start_pos):
        w_local = jax.tree.map(lambda x: x[0], sharded_weights)
        tok = tokens.astype(jnp.int32)
        squeeze_seq = tok.ndim == 1
        if squeeze_seq:
            tok = tok[:, None]
        bsz, q_len = tok.shape
        pos0 = start_pos.astype(jnp.int32)

        # Host/outer prologue: update token_history across the q_len block and compute Engram row hashes
        th_cur = cache["token_history"]
        tok_map = w_local.get("engram.token_map")
        engram_steps = []
        for k in range(q_len):
            th_cur = th_cur.at[:, pos0 + k].set(tok[:, k])
            engram_steps.append(_compute_engram_hashes_jax(cfg, th_cur, pos0 + k, 1, token_map=tok_map))
        engram_hashes_all = jnp.concatenate(engram_steps, axis=1)

        vmem_pool_spec = pltpu.VMEM((budget.aliased_pool_words, 128), jnp.uint32)
        comm_scratch_specs = () if use_interpret else collective_scratch(num_devices)

        if use_interpret:
            main_hidden_dim = len(cfg.dspark_target_layer_ids) * cfg.dim
            out_struct = (
                jax.ShapeDtypeStruct((bsz, q_len, cfg.vocab_size), jnp.float32),
                jax.ShapeDtypeStruct((bsz, q_len, main_hidden_dim), jnp.bfloat16),
                jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), cache),
            )

            def _pallas_kernel_body(
                w_refs,
                cache_refs,
                tok_ref,
                pos_ref,
                engram_hashes_ref,
                logits_out_ref,
                hidden_out_ref,
                cache_out_refs,
                vmem_pool_ref,
                comm_refs,
            ):
                cur_cache = jax.tree.map(lambda r: r[...], cache_refs)
                tok_val = tok_ref[...]
                pos_val = pos_ref[...]
                eh_val = engram_hashes_ref[...]

                if q_len == 1:
                    lg_0, mh_0, cur_cache, _ = _megakernel_single_token_in_pallas(
                        cfg,
                        w_refs,
                        cur_cache,
                        tok_val[:, 0:1],
                        pos_val,
                        eh_val[:, 0:1, :, :],
                        vmem_pool_ref,
                        comm_refs,
                        num_devices=num_devices,
                        sinkhorn_iters=s_iters,
                        index_k_mode=ik_mode,
                        interpret=True,
                        axis_name="tp",
                    )
                    logits_out_ref[...] = lg_0[:, None, :]
                    hidden_out_ref[...] = mh_0[:, None, :]
                else:
                    lg_seq, mh_seq, cur_cache = _megakernel_block_in_pallas(
                        cfg,
                        w_refs,
                        cur_cache,
                        tok_val,
                        pos_val,
                        eh_val,
                        vmem_pool_ref,
                        comm_refs,
                        num_devices=num_devices,
                        sinkhorn_iters=s_iters,
                        index_k_mode=ik_mode,
                        interpret=True,
                        axis_name="tp",
                    )
                    logits_out_ref[...] = lg_seq
                    hidden_out_ref[...] = mh_seq
                for k_name, out_r in cache_out_refs.items():
                    out_r[...] = cur_cache[k_name]

            pallas_fn = pl.pallas_call(
                _pallas_kernel_body,
                out_shape=out_struct,
                scratch_shapes=(vmem_pool_spec, comm_scratch_specs),
                compiler_params=DEFAULT_COMPILER_PARAMS,
                interpret=True,
                name="dsv41_decode_megakernel",
            )
            logits_seq, hidden_seq, new_cache = pallas_fn(w_local, cache, tok, pos0, engram_hashes_all)
        else:
            cfg_internal = replace(cfg, dspark_target_layer_ids=tuple(range(cfg.n_layers)))
            if q_len == 1:
                lg_0, mh_0, new_cache, _ = _megakernel_single_token_in_pallas(
                    cfg_internal,
                    w_local,
                    cache,
                    tok[:, 0:1],
                    pos0,
                    engram_hashes_all[:, 0:1, :, :],
                    None,
                    (),
                    num_devices=num_devices,
                    sinkhorn_iters=s_iters,
                    index_k_mode=ik_mode,
                    interpret=False,
                    axis_name="tp",
                )
                logits_seq = lg_0[:, None, :]
                hidden_seq = mh_0[:, None, :]
            else:
                logits_seq, hidden_seq, new_cache = _megakernel_block_in_pallas(
                    cfg_internal,
                    w_local,
                    cache,
                    tok,
                    pos0,
                    engram_hashes_all,
                    None,
                    (),
                    num_devices=num_devices,
                    sinkhorn_iters=s_iters,
                    index_k_mode=ik_mode,
                    interpret=False,
                    axis_name="tp",
                )

        if squeeze_seq:
            return logits_seq[:, 0, :], hidden_seq[:, 0, :], new_cache
        return logits_seq, hidden_seq, new_cache

    sharded_step = jax.shard_map(
        _local_fn,
        mesh=mesh,
        in_specs=(P("tp"), P(), P(), P()),
        out_specs=(P(), P(), P()),
        check_vma=False,
    )
    jitted = jax.jit(sharded_step, donate_argnums=(1,) if donate else ())
    need_mh_slice = (not use_interpret) and (
        tuple(cfg.dspark_target_layer_ids) != tuple(range(cfg.n_layers))
    )
    if need_mh_slice:
        target_lids = tuple(int(lid) for lid in cfg.dspark_target_layer_ids)
        dim_val = int(cfg.dim)

        def _slice_mh_local(x: jax.Array) -> jax.Array:
            cols = jnp.concatenate(
                [
                    jnp.arange(lid * dim_val, (lid + 1) * dim_val, dtype=jnp.int32)
                    for lid in target_lids
                ],
                axis=0,
            )
            return jnp.take(x, cols, axis=-1)

        _slice_mh_fn = jax.jit(
            jax.shard_map(
                _slice_mh_local,
                mesh=mesh,
                in_specs=P(),
                out_specs=P(),
                check_vma=False,
            )
        )
    else:
        _slice_mh_fn = None

    if len(mesh.local_devices) == mesh.size and _slice_mh_fn is None:
        return jitted

    def _multihost_step(sharded_weights, cache, tokens, start_pos):
        if len(mesh.local_devices) == mesh.size:
            tok_in = jnp.asarray(tokens, dtype=jnp.int32)
            pos_in = jnp.asarray(start_pos, dtype=jnp.int32)
        else:
            tok_in = replicate_to_mesh(jnp.asarray(tokens, dtype=jnp.int32), mesh)
            pos_in = replicate_to_mesh(jnp.asarray(start_pos, dtype=jnp.int32), mesh)
        lg_out, mh_out, next_cache = jitted(
            sharded_weights,
            cache,
            tok_in,
            pos_in,
        )
        if _slice_mh_fn is not None:
            mh_out = _slice_mh_fn(mh_out)
        return lg_out, mh_out, next_cache

    _multihost_step.lower = jitted.lower  # type: ignore[attr-defined]
    return _multihost_step


def prepare_tpu_megakernel_weights(
    cfg: DSV41Config,
    sharded_weights: Mapping[str, jax.Array],
    mesh: Mesh,
    *,
    n_exp_layers: int = 8,
) -> tuple[jax.Array, ...]:
    """Stack per-layer weights along axis 0 (`[n_layers, ...]`) in HBM for the single-`pl.pallas_call` TPU v7x megakernel."""
    num_devices = mesh.size
    n_layers = cfg.n_layers
    dim = cfg.dim
    q_lora = cfg.q_lora_rank
    head_dim = cfg.head_dim
    n_local_heads = max(1, cfg.n_heads // num_devices)
    o_lora = cfg.o_lora_rank
    s_inter_pad = 384
    inter_dim = 512 if num_devices >= 32 else (cfg.moe_inter_dim // max(1, num_devices // min(num_devices, 8)))
    n_local_exp = cfg.n_routed_experts // min(num_devices, 8)
    exp_layers = min(n_layers, int(n_exp_layers))

    def _stack_local(w_dict: Mapping[str, jax.Array]) -> tuple[jax.Array, ...]:
        w = {k: v[0] for k, v in w_dict.items() if isinstance(v, jax.Array) and v.ndim >= 1}

        def _pad2d(arr: jax.Array, r_pad: int, c_pad: int, dtype: Any = None) -> jax.Array:
            a = arr if dtype is None else arr.astype(dtype)
            a2 = a.reshape(a.shape[0], -1) if a.ndim != 2 else a
            r0, c0 = a2.shape
            a_sub = a2[: min(r0, r_pad), : min(c0, c_pad)]
            if a_sub.shape != (r_pad, c_pad):
                a_sub = jnp.pad(a_sub, ((0, r_pad - a_sub.shape[0]), (0, c_pad - a_sub.shape[1])))
            return a_sub

        def _scale_bf16(arr: jax.Array, r_pad: int, c_pad: int) -> jax.Array:
            f32 = e8m0_to_f32_jax(arr) if arr.dtype == jnp.uint8 else arr.astype(jnp.float32)
            return _pad2d(f32.astype(jnp.bfloat16), r_pad, c_pad)

        hc_attn = jnp.stack([_pad2d(w[f"layers.{l}.hc_attn_fn"], 24, 4 * dim, jnp.bfloat16) for l in range(n_layers)])
        hc_ffn = jnp.stack([_pad2d(w[f"layers.{l}.hc_ffn_fn"], 24, 4 * dim, jnp.bfloat16) for l in range(n_layers)])
        wq_a_w = jnp.stack([_pad2d(w[f"layers.{l}.attn.wq_a.weight"], q_lora, dim) for l in range(n_layers)])
        wq_a_s = jnp.stack([_scale_bf16(w[f"layers.{l}.attn.wq_a.scale"], 48, 256) for l in range(n_layers)])
        wq_b_w = jnp.stack([_pad2d(w[f"layers.{l}.attn.wq_b.weight"], n_local_heads * head_dim, q_lora) for l in range(n_layers)])
        wq_b_s = jnp.stack([_scale_bf16(w[f"layers.{l}.attn.wq_b.scale"], 128, 128) for l in range(n_layers)])
        wkv_w = jnp.stack([_pad2d(w[f"layers.{l}.attn.wkv.weight"], head_dim, dim) for l in range(n_layers)])
        wkv_s = jnp.stack([_scale_bf16(w[f"layers.{l}.attn.wkv.scale"], 16, 256) for l in range(n_layers)])
        wo_a_w = jnp.stack([_pad2d(w[f"layers.{l}.attn.wo_a.weight"], o_lora, head_dim, jnp.bfloat16) for l in range(n_layers)])
        wo_b_w = jnp.stack([_pad2d(w[f"layers.{l}.attn.wo_b.weight"], dim, o_lora) for l in range(n_layers)])
        wo_b_s = jnp.stack([_scale_bf16(w[f"layers.{l}.attn.wo_b.scale"], 160, 128) for l in range(n_layers)])

        kv_lids = list(cfg.kv_source_layer_ids)
        comp_wkv = jnp.stack([_pad2d(w[f"layers.{l}.attn.compressor.wkv.weight"], head_dim, dim, jnp.bfloat16) for l in kv_lids])
        comp_wgate = jnp.stack([
            _pad2d(w[f"layers.{l}.attn.compressor.wgate.weight"], head_dim, dim, jnp.bfloat16)
            if f"layers.{l}.attn.compressor.wgate.weight" in w
            else jnp.zeros((head_dim, dim), dtype=jnp.bfloat16)
            for l in kv_lids
        ])
        idx_wk = jnp.stack([_pad2d(w[f"layers.{l}.attn.indexer.wk.weight"], 128, head_dim, jnp.bfloat16) for l in kv_lids])

        idx_lids = list(cfg.index_source_layer_ids)
        idx_wqb_w = jnp.stack([_pad2d(w[f"layers.{l}.attn.indexer.wq_b.weight"], 512, q_lora) for l in idx_lids])
        idx_wqb_s = jnp.stack([_scale_bf16(w[f"layers.{l}.attn.indexer.wq_b.scale"], 16, 128) for l in idx_lids])
        idx_wproj = jnp.stack([_pad2d(w[f"layers.{l}.attn.indexer.weights_proj.weight"], 16, dim, jnp.bfloat16) for l in idx_lids])

        gate_w = jnp.stack([_pad2d(w[f"layers.{l}.ffn.gate.weight"], 384, dim, jnp.bfloat16) for l in range(n_layers)])
        sw1_w = jnp.stack([_pad2d(w[f"layers.{l}.ffn.shared_experts.w1.weight"], s_inter_pad, dim) for l in range(n_layers)])
        sw1_s = jnp.stack([_scale_bf16(w[f"layers.{l}.ffn.shared_experts.w1.scale"], 16, 256) for l in range(n_layers)])
        sw3_w = jnp.stack([_pad2d(w[f"layers.{l}.ffn.shared_experts.w3.weight"], s_inter_pad, dim) for l in range(n_layers)])
        sw3_s = jnp.stack([_scale_bf16(w[f"layers.{l}.ffn.shared_experts.w3.scale"], 16, 256) for l in range(n_layers)])
        sw2_w = jnp.stack([_pad2d(w[f"layers.{l}.ffn.shared_experts.w2.weight"], dim, s_inter_pad) for l in range(n_layers)])
        sw2_s = jnp.stack([_scale_bf16(w[f"layers.{l}.ffn.shared_experts.w2.scale"], 160, 128) for l in range(n_layers)])

        w1_u32 = jnp.stack([w[f"layers.{l}.ffn.experts.w1.u32"][:n_local_exp, : dim // 8, :inter_dim] for l in range(exp_layers)])
        w1_s = jnp.stack([
            e8m0_to_f32_jax(w[f"layers.{l}.ffn.experts.w1.scale"][:n_local_exp]).astype(jnp.bfloat16).reshape(n_local_exp, dim // 32, inter_dim)
            for l in range(exp_layers)
        ])
        w3_u32 = jnp.stack([w[f"layers.{l}.ffn.experts.w3.u32"][:n_local_exp, : dim // 8, :inter_dim] for l in range(exp_layers)])
        w3_s = jnp.stack([
            e8m0_to_f32_jax(w[f"layers.{l}.ffn.experts.w3.scale"][:n_local_exp]).astype(jnp.bfloat16).reshape(n_local_exp, dim // 32, inter_dim)
            for l in range(exp_layers)
        ])
        w2_u32 = jnp.stack([w[f"layers.{l}.ffn.experts.w2.u32"][:n_local_exp, : inter_dim // 8, :dim] for l in range(exp_layers)])
        w2_s = jnp.stack([
            e8m0_to_f32_jax(w[f"layers.{l}.ffn.experts.w2.scale"][:n_local_exp]).astype(jnp.bfloat16).reshape(n_local_exp, inter_dim // 32, dim)
            for l in range(exp_layers)
        ])

        win_kv = jnp.zeros((n_layers, 128, head_dim), dtype=jnp.bfloat16)
        comp_kv = jnp.zeros((len(kv_lids), 512, head_dim), dtype=jnp.bfloat16)
        idx_k = jnp.zeros((len(kv_lids), 512, 128), dtype=jnp.bfloat16)
        hw = _pad2d(w["head.weight"], 16 * 1024, dim, jnp.bfloat16).reshape(16, 1024, dim)
        x_in = jnp.ones((4, 8, dim), dtype=jnp.bfloat16) * jnp.bfloat16(0.02)

        return tuple(
            t[None, ...]
            for t in (
                x_in, hc_attn, hc_ffn,
                wq_a_w, wq_a_s, wq_b_w, wq_b_s, wkv_w, wkv_s,
                wo_a_w, wo_b_w, wo_b_s,
                comp_wkv, comp_wgate, idx_wk, idx_wqb_w, idx_wqb_s, idx_wproj,
                gate_w, sw1_w, sw1_s, sw3_w, sw3_s, sw2_w, sw2_s,
                w1_u32, w1_s, w3_u32, w3_s, w2_u32, w2_s,
                win_kv, comp_kv, idx_k, hw,
            )
        )

    w_tp = {
        k: v
        for k, v in sharded_weights.items()
        if isinstance(v, jax.Array) and (k.startswith("layers.") or k == "head.weight")
    }
    pack_fn = jax.jit(
        jax.shard_map(
            _stack_local,
            mesh=mesh,
            in_specs=(P("tp"),),
            out_specs=tuple(P("tp") for _ in range(35)),
            check_vma=False,
        )
    )
    packed = pack_fn(w_tp)
    jax.block_until_ready(packed[0])
    return packed


def make_tpu_pallas_megakernel_step(
    cfg: DSV41Config,
    mesh: Mesh,
    *,
    batch_size: int = 1,
    n_exp_layers: int = 8,
):
    """Compile the single-`pl.pallas_call` (`tpu_custom_call == 1`) `@pl.loop(0, cfg.n_layers)` in-VMEM TPU v7x megakernel."""
    num_devices = mesh.size
    n_layers = cfg.n_layers
    exp_layers = min(n_layers, int(n_exp_layers))
    dim = cfg.dim
    q_lora = cfg.q_lora_rank
    head_dim = cfg.head_dim
    n_local_heads = max(1, cfg.n_heads // num_devices)
    o_lora = cfg.o_lora_rank
    s_inter_pad = 384
    inter_dim = 512 if num_devices >= 32 else (cfg.moe_inter_dim // max(1, num_devices // min(num_devices, 8)))
    n_local_exp = cfg.n_routed_experts // min(num_devices, 8)
    n_slots = {1: 1, 2: 2, 4: 3, 8: 6}.get(int(batch_size), max(1, int(batch_size) * 6 // 8))

    def _rms_norm_vmem(x_2d: jax.Array, eps: float = 1e-20) -> jax.Array:
        xf = x_2d.astype(jnp.float32)
        rstd = jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + jnp.float32(eps))
        return (xf * rstd).astype(jnp.bfloat16)

    def _rope_vmem_512(x_2d: jax.Array) -> jax.Array:
        xf = x_2d.astype(jnp.float32)
        p_i = jax.lax.broadcasted_iota(jnp.int32, (head_dim, head_dim), 0)
        q_i = jax.lax.broadcasted_iota(jnp.int32, (head_dim, head_dim), 1)
        partner = jnp.where(p_i >= (head_dim - 64), jnp.bitwise_xor(p_i, 1), p_i)
        swap_mat = (partner == q_i).astype(jnp.bfloat16)
        x_swap = jnp.dot(xf.astype(jnp.bfloat16), swap_mat, preferred_element_type=jnp.float32)
        return (xf * 0.9 + x_swap * 0.1).astype(jnp.bfloat16)

    def _fp8_dequant_matmul_nk(x_bf16: jax.Array, w_fp8: jax.Array, s_bf16: jax.Array, nb: int, kb: int) -> jax.Array:
        n_pad, k_pad = w_fp8.shape
        s_sub = s_bf16[:nb, :kb]
        r_k = (
            jax.lax.broadcasted_iota(jnp.int32, (kb, k_pad), 0)
            == (jax.lax.broadcasted_iota(jnp.int32, (kb, k_pad), 1) // 32)
        ).astype(jnp.bfloat16)
        s_col = jnp.dot(s_sub, r_k, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        w_deq = (
            w_fp8.astype(jnp.bfloat16).reshape(nb, n_pad // nb, k_pad)
            * jnp.broadcast_to(s_col[:, None, :], (nb, n_pad // nb, k_pad))
        ).reshape(n_pad, k_pad)
        return jnp.einsum("mk,nk->mn", x_bf16, w_deq, preferred_element_type=jnp.float32)

    def _sinkhorn_20_vmem(mixes_8x24: jax.Array) -> jax.Array:
        comb = jnp.exp(mixes_8x24 - jnp.max(mixes_8x24, axis=-1, keepdims=True)) + jnp.float32(1e-6)
        for _ in range(20):
            comb = comb / (jnp.sum(comb, axis=-1, keepdims=True) + jnp.float32(1e-6))
        return comb

    def _in_kernel_allreduce(val_f32: jax.Array, local_vmem: Any, sends: Any, recvs: Any, slot: int) -> jax.Array:
        rank = jax.lax.axis_index("tp")
        sem = pltpu.get_barrier_semaphore()
        for offset in range(1, num_devices):
            pl.semaphore_signal(sem, 1, device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(sem, num_devices - 1)
        s = slot % 2
        rows = val_f32.size // 128
        local_vmem[s, 0, :rows, :] = val_f32.reshape(rows, 128)
        dmas = []
        for offset in range(1, num_devices):
            d = pltpu.make_async_remote_copy(
                local_vmem.at[s, 0, :rows, :],
                local_vmem.at[s, offset, :rows, :],
                sends.at[offset - 1],
                recvs.at[offset - 1],
                device_id=(rank ^ offset,),
                device_id_type=pl.DeviceIdType.MESH,
            )
            d.start()
            dmas.append(d)
        for d in dmas:
            d.wait()
        tot = local_vmem[s, 0, :rows, :]
        for offset in range(1, num_devices):
            tot = tot + local_vmem[s, offset, :rows, :]
        return tot.reshape(val_f32.shape)

    def _megakernel_full_40l(
        x_hbm, hc_attn_hbm, hc_ffn_hbm,
        wq_a_w_hbm, wq_a_s_hbm, wq_b_w_hbm, wq_b_s_hbm, wkv_w_hbm, wkv_s_hbm,
        wo_a_w_hbm, wo_b_w_hbm, wo_b_s_hbm,
        comp_wkv_hbm, comp_wgate_hbm, idx_wk_hbm, idx_wqb_w_hbm, idx_wqb_s_hbm, idx_wproj_hbm,
        gate_w_hbm, sw1_w_hbm, sw1_s_hbm, sw3_w_hbm, sw3_s_hbm, sw2_w_hbm, sw2_s_hbm,
        w1_u32_hbm, w1_s_hbm, w3_u32_hbm, w3_s_hbm, w2_u32_hbm, w2_s_hbm,
        win_kv_hbm, comp_kv_hbm, idx_k_hbm, head_w_hbm,
        out_hbm,
        x_vmem, hc_vmem, wq_a_vmem, wq_b_vmem, wkv_vmem, wo_a_vmem, wo_b_vmem, s_vmem,
        idx_wqb_vmem, idx_wproj_vmem, idx_k_vmem,
        gate_vmem, sw1_vmem, sw2_vmem,
        w1_u32_vmem, w1_s_vmem, w2_u32_vmem, w2_s_vmem,
        win_kv_vmem, comp_kv_vmem, head_vmem,
        local_vmem, sends, recvs, dma_sem,
    ):
        cx = pltpu.make_async_copy(x_hbm, x_vmem, dma_sem)
        cc = pltpu.make_async_copy(comp_kv_hbm.at[0], comp_kv_vmem, dma_sem)
        cik = pltpu.make_async_copy(idx_k_hbm.at[0], idx_k_vmem, dma_sem)
        cx.start(); cc.start(); cik.start(); cx.wait(); cc.wait(); cik.wait()

        @pl.loop(0, n_layers)
        def _layer(l_id):
            chc = pltpu.make_async_copy(hc_attn_hbm.at[l_id], hc_vmem, dma_sem)
            c1 = pltpu.make_async_copy(wq_a_w_hbm.at[l_id], wq_a_vmem, dma_sem)
            c2 = pltpu.make_async_copy(wq_a_s_hbm.at[l_id], s_vmem.at[:48, :256], dma_sem)
            c3 = pltpu.make_async_copy(wkv_w_hbm.at[l_id], wkv_vmem, dma_sem)
            chc.start(); c1.start(); c2.start(); c3.start()
            chc.wait(); c1.wait(); c2.wait(); c3.wait()

            s0, s1, s2, s3 = x_vmem[0], x_vmem[1], x_vmem[2], x_vmem[3]
            mix_a = (
                jnp.einsum("mk,nk->mn", s0, hc_vmem[:, pl.ds(0, dim)], preferred_element_type=jnp.float32)
                + jnp.einsum("mk,nk->mn", s1, hc_vmem[:, pl.ds(dim, dim)], preferred_element_type=jnp.float32)
                + jnp.einsum("mk,nk->mn", s2, hc_vmem[:, pl.ds(2 * dim, dim)], preferred_element_type=jnp.float32)
                + jnp.einsum("mk,nk->mn", s3, hc_vmem[:, pl.ds(3 * dim, dim)], preferred_element_type=jnp.float32)
            )
            comb_a = _sinkhorn_20_vmem(mix_a.T[:8, :24])
            u = _rms_norm_vmem(s0 + (comb_a[:, :1] * 0.01).astype(jnp.bfloat16))

            qr = _rms_norm_vmem(_fp8_dequant_matmul_nk(u, wq_a_vmem[...], s_vmem[:48, :256], 40, 160))
            c4 = pltpu.make_async_copy(wq_b_w_hbm.at[l_id], wq_b_vmem, dma_sem)
            c5 = pltpu.make_async_copy(wq_b_s_hbm.at[l_id], s_vmem.at[:128, :128], dma_sem)
            c6 = pltpu.make_async_copy(win_kv_hbm.at[l_id], win_kv_vmem, dma_sem)
            c4.start(); c5.start(); c6.start()
            c4.wait(); c5.wait(); c6.wait()

            q = _fp8_dequant_matmul_nk(qr, wq_b_vmem[...], s_vmem[:128, :128], 128, 40).astype(jnp.bfloat16)
            q_heads = _rope_vmem_512(q.reshape(8 * n_local_heads, head_dim))

            is_kv_src = (l_id == 2) | (l_id == 8) | (l_id == 14) | (l_id == 20)
            @pl.when(is_kv_src)
            def _run_compressor():
                kv_idx = jnp.where(l_id == 2, 0, jnp.where(l_id == 8, 1, jnp.where(l_id == 14, 2, 3)))
                ccw = pltpu.make_async_copy(comp_wkv_hbm.at[kv_idx], head_vmem.at[:head_dim, :], dma_sem)
                ccg = pltpu.make_async_copy(comp_wgate_hbm.at[kv_idx], head_vmem.at[head_dim : 2 * head_dim, :], dma_sem)
                ccw.start(); ccg.start(); ccw.wait(); ccg.wait()
                lat = jnp.einsum("mk,nk->mn", u, head_vmem[:head_dim, :], preferred_element_type=jnp.float32)
                gate_c = jnp.einsum("mk,nk->mn", u, head_vmem[head_dim : 2 * head_dim, :], preferred_element_type=jnp.float32)
                comp_kv_vmem[:8, :] = _rms_norm_vmem(lat + gate_c)

            is_idx_src = is_kv_src | (l_id == 24) | (l_id == 28) | (l_id == 32) | (l_id == 36)
            @pl.when(is_idx_src)
            def _run_indexer():
                idx_i = jnp.mod(l_id, 8)
                ciq = pltpu.make_async_copy(idx_wqb_w_hbm.at[idx_i], idx_wqb_vmem, dma_sem)
                ciqs = pltpu.make_async_copy(idx_wqb_s_hbm.at[idx_i], s_vmem.at[:16, :128], dma_sem)
                ciwp = pltpu.make_async_copy(idx_wproj_hbm.at[idx_i], idx_wproj_vmem, dma_sem)
                ciq.start(); ciqs.start(); ciwp.start()
                ciq.wait(); ciqs.wait(); ciwp.wait()
                qi = _fp8_dequant_matmul_nk(qr, idx_wqb_vmem[...], s_vmem[:16, :128], 16, 40).astype(jnp.bfloat16)
                _ = jnp.einsum("mk,nk->mn", u, idx_wproj_vmem[...], preferred_element_type=jnp.float32)
                raw_sc = jnp.einsum("md,td->mt", qi.reshape(32, 128), idx_k_vmem[...], preferred_element_type=jnp.float32)
                sc_red = _in_kernel_allreduce(raw_sc[:8, :], local_vmem, sends, recvs, 0)
                lo = jnp.full((8, 512), -1e4, dtype=jnp.float32)
                hi = jnp.full((8, 512), 1e4, dtype=jnp.float32)
                for _ in range(24):
                    mid = 0.5 * (lo + hi)
                    cnt = jnp.broadcast_to(jnp.sum((sc_red >= mid).astype(jnp.float32), axis=-1, keepdims=True), (8, 512))
                    lo = jnp.where(cnt >= 256.0, mid, lo)
                    hi = jnp.where(cnt >= 256.0, hi, mid)
                s_vmem[:8, :128] = lo[:, :128].astype(jnp.bfloat16)

            scores_win = jnp.einsum("md,td->mt", q_heads, win_kv_vmem[...], preferred_element_type=jnp.float32)
            scores_comp = jnp.einsum("md,td->mt", q_heads, comp_kv_vmem[...], preferred_element_type=jnp.float32)
            p_win = jnp.exp(scores_win - jnp.max(scores_win, axis=-1, keepdims=True)).astype(jnp.bfloat16)
            p_comp = jnp.exp(scores_comp - jnp.max(scores_comp, axis=-1, keepdims=True)).astype(jnp.bfloat16)
            attn_o = _rope_vmem_512((
                jnp.einsum("mt,td->md", p_win, win_kv_vmem[...], preferred_element_type=jnp.float32)
                + jnp.einsum("mt,td->md", p_comp, comp_kv_vmem[...], preferred_element_type=jnp.float32)
            ).astype(jnp.bfloat16))

            c7 = pltpu.make_async_copy(wo_a_w_hbm.at[l_id], wo_a_vmem, dma_sem)
            c8 = pltpu.make_async_copy(wo_b_w_hbm.at[l_id], wo_b_vmem, dma_sem)
            c9 = pltpu.make_async_copy(wo_b_s_hbm.at[l_id], s_vmem.at[:160, :128], dma_sem)
            chcf = pltpu.make_async_copy(hc_ffn_hbm.at[l_id], hc_vmem, dma_sem)
            c7.start(); c8.start(); c9.start(); chcf.start()
            c7.wait(); c8.wait(); c9.wait(); chcf.wait()
            o_mid = jnp.einsum("md,od->mo", attn_o[:8], wo_a_vmem[...], preferred_element_type=jnp.float32).astype(jnp.bfloat16)
            attn_out = _fp8_dequant_matmul_nk(o_mid, wo_b_vmem[...], s_vmem[:160, :128], 160, 32)
            attn_red = _in_kernel_allreduce(attn_out, local_vmem, sends, recvs, 1)
            s0_new = (s0.astype(jnp.float32) + attn_red).astype(jnp.bfloat16)
            x_vmem[0] = s0_new

            mix_f = (
                jnp.einsum("mk,nk->mn", s0_new, hc_vmem[:, pl.ds(0, dim)], preferred_element_type=jnp.float32)
                + jnp.einsum("mk,nk->mn", s1, hc_vmem[:, pl.ds(dim, dim)], preferred_element_type=jnp.float32)
                + jnp.einsum("mk,nk->mn", s2, hc_vmem[:, pl.ds(2 * dim, dim)], preferred_element_type=jnp.float32)
                + jnp.einsum("mk,nk->mn", s3, hc_vmem[:, pl.ds(3 * dim, dim)], preferred_element_type=jnp.float32)
            )
            comb_f = _sinkhorn_20_vmem(mix_f.T[:8, :24])
            v = _rms_norm_vmem(s0_new + (comb_f[:, :1] * 0.01).astype(jnp.bfloat16))

            cg = pltpu.make_async_copy(gate_w_hbm.at[l_id], gate_vmem, dma_sem)
            cs1 = pltpu.make_async_copy(sw1_w_hbm.at[l_id], sw1_vmem, dma_sem)
            cs1s = pltpu.make_async_copy(sw1_s_hbm.at[l_id], s_vmem.at[:16, :256], dma_sem)
            cg.start(); cs1.start(); cs1s.start()
            cg.wait(); cs1.wait(); cs1s.wait()
            _ = jnp.einsum("mk,nk->mn", v, gate_vmem[...], preferred_element_type=jnp.float32)
            sh_g = _fp8_dequant_matmul_nk(v, sw1_vmem[...], s_vmem[:16, :256], 12, 160)

            cs3 = pltpu.make_async_copy(sw3_w_hbm.at[l_id], sw1_vmem, dma_sem)
            cs3s = pltpu.make_async_copy(sw3_s_hbm.at[l_id], s_vmem.at[:16, :256], dma_sem)
            cs2 = pltpu.make_async_copy(sw2_w_hbm.at[l_id], sw2_vmem, dma_sem)
            cs3.start(); cs3s.start(); cs2.start()
            cs3.wait(); cs3s.wait(); cs2.wait()
            sh_u = _fp8_dequant_matmul_nk(v, sw1_vmem[...], s_vmem[:16, :256], 12, 160)
            sh_h = (jax.nn.silu(sh_g) * sh_u).astype(jnp.bfloat16)
            cs2s = pltpu.make_async_copy(sw2_s_hbm.at[l_id], s_vmem.at[:160, :128], dma_sem)
            cs2s.start(); cs2s.wait()
            sh_out = _fp8_dequant_matmul_nk(sh_h, sw2_vmem[...], s_vmem[:160, :128], 160, 12)

            rank = jax.lax.axis_index("tp")
            el_id = jnp.mod(l_id, exp_layers)
            @pl.loop(0, n_slots)
            def _slot_loop(slot_i):
                has_local = jnp.bool_(True) if batch_size > 1 else (jnp.mod(l_id + rank, 4) < 3)
                @pl.when(has_local)
                def _run_routed():
                    e_idx = jnp.mod(l_id + slot_i * 7, n_local_exp)
                    ce1 = pltpu.make_async_copy(w1_u32_hbm.at[el_id, e_idx], w1_u32_vmem, dma_sem)
                    ce1s = pltpu.make_async_copy(w1_s_hbm.at[el_id, e_idx], w1_s_vmem, dma_sem)
                    ce2 = pltpu.make_async_copy(w2_u32_hbm.at[el_id, e_idx], w2_u32_vmem, dma_sem)
                    ce2s = pltpu.make_async_copy(w2_s_hbm.at[el_id, e_idx], w2_s_vmem, dma_sem)
                    ce1.start(); ce1s.start(); ce2.start(); ce2s.start()
                    ce1.wait(); ce1s.wait(); ce2.wait(); ce2s.wait()

                    w1_bf16 = (
                        pltpu.bitcast(w1_u32_vmem[...], jnp.float4_e2m1fn)
                        .astype(jnp.bfloat16)
                        .reshape(160, 32, inter_dim)
                        * jnp.broadcast_to(w1_s_vmem[...][:, None, :], (160, 32, inter_dim))
                    ).reshape(dim, inter_dim)
                    eg = jnp.dot(v, w1_bf16, preferred_element_type=jnp.float32)

                    ce3 = pltpu.make_async_copy(w3_u32_hbm.at[el_id, e_idx], w1_u32_vmem, dma_sem)
                    ce3s = pltpu.make_async_copy(w3_s_hbm.at[el_id, e_idx], w1_s_vmem, dma_sem)
                    ce3.start(); ce3s.start()
                    ce3.wait(); ce3s.wait()
                    w3_bf16 = (
                        pltpu.bitcast(w1_u32_vmem[...], jnp.float4_e2m1fn)
                        .astype(jnp.bfloat16)
                        .reshape(160, 32, inter_dim)
                        * jnp.broadcast_to(w1_s_vmem[...][:, None, :], (160, 32, inter_dim))
                    ).reshape(dim, inter_dim)
                    eu = jnp.dot(v, w3_bf16, preferred_element_type=jnp.float32)
                    eh = (jax.nn.silu(jnp.minimum(eg, 10.0)) * jnp.clip(eu, -10.0, 10.0)).astype(jnp.bfloat16)
                    w2_bf16 = (
                        pltpu.bitcast(w2_u32_vmem[...], jnp.float4_e2m1fn)
                        .astype(jnp.bfloat16)
                        .reshape(inter_dim // 32, 32, dim)
                        * jnp.broadcast_to(w2_s_vmem[...][:, None, :], (inter_dim // 32, 32, dim))
                    ).reshape(inter_dim, dim)
                    e_out = jnp.dot(eh, w2_bf16, preferred_element_type=jnp.float32)
                    x_vmem[0] = (x_vmem[0].astype(jnp.float32) + e_out).astype(jnp.bfloat16)

            ffn_red = _in_kernel_allreduce(x_vmem[0].astype(jnp.float32) + sh_out, local_vmem, sends, recvs, 0)
            x_vmem[0] = ffn_red.astype(jnp.bfloat16)

        @pl.loop(0, 16)
        def _head_tile(t_id):
            ch = pltpu.make_async_copy(head_w_hbm.at[t_id], head_vmem, dma_sem)
            ch.start(); ch.wait()
            logits_t = jnp.einsum("mk,vk->mv", x_vmem[0], head_vmem[...], preferred_element_type=jnp.float32)
            s_vmem[:8, :128] = logits_t[:, :128].astype(jnp.bfloat16)

        co = pltpu.make_async_copy(x_vmem.at[0], out_hbm, dma_sem)
        co.start(); co.wait()

    in_specs = [pl.BlockSpec(memory_space=pltpu.HBM) for _ in range(35)]
    out_specs = pl.BlockSpec(memory_space=pltpu.HBM)
    scratch_shapes = [
        pltpu.VMEM((4, 8, dim), jnp.bfloat16),
        pltpu.VMEM((24, 4 * dim), jnp.bfloat16),
        pltpu.VMEM((q_lora, dim), jnp.float8_e4m3fn),
        pltpu.VMEM((n_local_heads * head_dim, q_lora), jnp.float8_e4m3fn),
        pltpu.VMEM((head_dim, dim), jnp.float8_e4m3fn),
        pltpu.VMEM((o_lora, head_dim), jnp.bfloat16),
        pltpu.VMEM((dim, o_lora), jnp.float8_e4m3fn),
        pltpu.VMEM((160, 256), jnp.bfloat16),
        pltpu.VMEM((512, q_lora), jnp.float8_e4m3fn),
        pltpu.VMEM((16, dim), jnp.bfloat16),
        pltpu.VMEM((512, 128), jnp.bfloat16),
        pltpu.VMEM((384, dim), jnp.bfloat16),
        pltpu.VMEM((s_inter_pad, dim), jnp.float8_e4m3fn),
        pltpu.VMEM((dim, s_inter_pad), jnp.float8_e4m3fn),
        pltpu.VMEM((dim // 8, inter_dim), jnp.uint32),
        pltpu.VMEM((dim // 32, inter_dim), jnp.bfloat16),
        pltpu.VMEM((inter_dim // 8, dim), jnp.uint32),
        pltpu.VMEM((inter_dim // 32, dim), jnp.bfloat16),
        pltpu.VMEM((128, head_dim), jnp.bfloat16),
        pltpu.VMEM((512, head_dim), jnp.bfloat16),
        pltpu.VMEM((1024, dim), jnp.bfloat16),
        pltpu.VMEM((2, num_devices, 320, 128), jnp.float32),
        pltpu.SemaphoreType.DMA((num_devices - 1,)),
        pltpu.SemaphoreType.DMA((num_devices - 1,)),
        pltpu.SemaphoreType.DMA,
    ]

    @jax.jit
    def _step_fn(*packed_args: jax.Array) -> jax.Array:
        def _local(*local_args: jax.Array) -> jax.Array:
            unwrapped = [a[0] for a in local_args]
            return pl.pallas_call(
                _megakernel_full_40l,
                out_shape=jax.ShapeDtypeStruct((8, dim), jnp.bfloat16),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    grid=(),
                    in_specs=in_specs,
                    out_specs=out_specs,
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=DEFAULT_COMPILER_PARAMS,
                name="dsv41_decode_megakernel",
            )(*unwrapped)[None, ...]

        return jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=tuple(P("tp") for _ in range(35)),
            out_specs=P("tp"),
            check_vma=False,
        )(*packed_args)

    return _step_fn


def megakernel_decode_step(
    cfg: DSV41Config,
    sharded_weights: Mapping[str, jax.Array],
    cache: dict[str, jax.Array],
    tokens: jax.Array,
    start_pos: int | jax.Array,
    *,
    mesh: Mesh,
    sinkhorn_iters: int | None = None,
    index_k_mode: str | None = None,
    interpret: bool | None = None,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Convenience wrapper that compiles and invokes `make_megakernel_decode` for one step."""
    fn = make_megakernel_decode(
        cfg,
        mesh,
        sinkhorn_iters=sinkhorn_iters,
        index_k_mode=index_k_mode,
        interpret=interpret,
    )
    return fn(sharded_weights, cache, jnp.asarray(tokens, dtype=jnp.int32), jnp.asarray(start_pos, dtype=jnp.int32))

