"""NVFP4 expert path of the Muse Spark decode megakernel (kernel-side helpers, container v2).

Mirrors `musespark.moe` for the `expert_format == "nvfp4"` families (`musespark.layout`):

    gate_up_fp4 [L, E, Hm/8, 2*Is]          int32  eight e2m1 codes per word along K
    gate_up_bs  [L, E, Hm/KC, KC/16, 2*Is]  e4m3   block-16 scales, K-chunk major
    down_fp4    [L, E, Is/8, Hm]            int32
    down_bs     [L, E, Is/KC, KC/16, Hm]    e4m3
    expert_gs   [L, E, 8, 128]              f32    rows 0/1/2 = gate/up/down global scale

Two faithful dot formulations (both equal `x @ (e2m1 * e4m3)` up to f32 summation order, the
products `e2m1 * e4m3` being exact in bf16 and f32; the per-expert global scales are three
scalar multiplies on the `[B, N]` results, `apply_gate_up_gs` / `apply_down_gs`):

* `fp4_block_dot` (B <= 4): the int32 slab is bitcast in VMEM to `float4_e2m1fn` and converted
  for free to `float8_e4m3fn`, `jnp.dot(x_bf16, w_fp8)` runs on the MXU at the int4 rate and
  the block-16 scales are applied post-MXU with the block-diagonal LHS idiom of
  `moe.int4_group_dot` at group 16: `block_diag16(x)` expands the `[B, K]` input into
  `[K/KC, B*32, KC]` chunks -- row `b*32 + g` is row `b` masked to group `g` (BATCH-major, so
  every row's 32 partial products are one aligned `[32, N]` sublane block:
  `out[b] = sum(res[b*32:(b+1)*32] * s, axis=0)`; the group-major order needs sublane shuffles
  at B = 2 / 4 and costs 2x). The MXU streams 32*B LHS rows per 512-chunk (gate_up slice
  1.0 / 1.1 / 1.6 / 2.7 us at B = 1 / 2 / 4 / 8 on one TPU7x core, weight push bound up to B=2).
* `fp4_dequant_dot` (B = 8): VPU dequantization to bf16 (`e2m1 -> bf16 * bf16(e4m3)`, exact)
  and a plain `[B, K] x [K, N]` bf16 dot at the bf16 MXU rate (gate_up 1.5 us, down 1.0 us
  at any B). The per-vreg cost is the sublane broadcast of the block scale; it is made free
  by storing the scales as `[K/16, N]` int32 words holding the bf16 scale in BOTH halves
  (`scale_words`, once per expert) and reading each block's row with a stride-0 (broadcast)
  load: `bitcast(sd[pl.ds(j, 8, stride=0)], bf16)` is the `[16, N]` scale block.

`DEQUANT_MIN_BATCH` selects the formulation statically from the batch (measured crossover).

Expert stream (`expert_stream`, same bookkeeping tiles / `moe.route_to_scratch` /
`moe.finalize` as the int4 path): B = 1 is the static route-slot stream of `moe` (waves of
`SLOTS` route slots, unrolled); B >= 2 runs the wave loop with a RUNTIME loop over the slots so
the per-expert code (up to ~10k vector ops for the dequant path) is emitted once instead of
once per slot and wave-body variant -- the unrolled form was instruction-fetch bound
(66 / 47 / 85 us per layer at B = 2 / 4 / 8 for 22-34 us of dot work). Slot `s` of a wave
issues `gate_up(s)` before `down(s-1)` (one-expert skew, as `moe.wave_body`), gate_up results
are handed over through `sc.gu_buf`.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from musespark.config import Config
from musespark.quant import (  # noqa: F401  (re-exported for the reference/prefill path)
    FP4_BLOCK,
    FP4_PER_WORD,
    dequant_fp4,
    dequant_fp4_jnp,
    dequant_fp4_np,
    k_chunk,
    quantize_nvfp4_np,
)

BF16 = jnp.bfloat16
F32 = jnp.float32
I32 = jnp.int32
FP4 = jnp.float4_e2m1fn
FP8 = jnp.float8_e4m3fn
BLOCK = FP4_BLOCK  # 16 K rows per e4m3 scale
MR = 8  # sublane tile / max decode batch
LANES = 128
SLOTS = 4  # expert slots (2 in compute, 2 in flight), like moe.SLOTS
COPIES = 5  # DMAs per expert: gate_up_fp4, gate_up_bs, down_fp4, down_bs, expert_gs
GS_ROWS, GS_LANES = 8, 128
WAVE_ROWS = 3  # SMEM wave_ids ring: waves w, w+1, w+2 (by w % 3)
STATS_RING = 4  # SMEM route-stats ring over stream positions e-1, e, e+1 (by e % 4)
DEQUANT_MIN_BATCH = 8  # batch from which the bf16 VPU-dequant dot replaces the block-diagonal dot
COMPACT_ROWS = 2  # B > COMPACT_ROWS: experts with <= this many active rows use a compact LHS (0: off)
SUBLANES = 8  # f32 sublanes per vreg (stride-0 broadcast loads)


def _dot(a, b):
    return jnp.dot(a, b, preferred_element_type=F32)


def _iota(shape, axis):
    return lax.broadcasted_iota(I32, shape, axis)


def _interpret():
    """Trace-time: are we lowering for the Pallas interpreter (no strided VMEM loads)?"""
    return jax.default_backend() != "tpu"


def use_dequant(batch):
    """Static choice of the dot formulation for `batch` rows (`DEQUANT_MIN_BATCH`)."""
    return batch >= DEQUANT_MIN_BATCH


def use_compact(batch):
    """Static: does the `batch`-row stream carry the compact path (`COMPACT_ROWS` active rows)?"""
    return 0 < COMPACT_ROWS < batch


# ---------------------------------------------------------------------------------------------
# Block-diagonal LHS with group 16 (batch-major rows)
# ---------------------------------------------------------------------------------------------


def block_diag16_chunk(x, kc, i):
    """`x [rows, K]` (any float) -> chunk `i` of the group-16 block-diagonal LHS: `[rows*GPC, kc]`
    bf16 (`GPC = kc / 16`), row `b*GPC + g` = `x[b, i*kc:(i+1)*kc]` masked to the lanes of group `g`."""
    rows = x.shape[0]
    gpc = kc // BLOCK
    xk = x[:, i * kc : (i + 1) * kc].astype(F32)
    if gpc > 1:
        xk = jnp.broadcast_to(xk[:, None, :], (rows, gpc, kc)).reshape(rows * gpc, kc)
    mask = _iota((gpc * rows, kc), 0) % gpc == _iota((gpc * rows, kc), 1) // BLOCK
    return jnp.where(mask, xk, F32(0)).astype(BF16)


def block_diag16_values(x, kc=None):
    """`x [rows, K]` -> list of the `K/kc` chunk values (`block_diag16_chunk`)."""
    K = x.shape[1]
    kc = k_chunk(K) if kc is None else kc
    return [block_diag16_chunk(x, kc, i) for i in range(K // kc)]


def block_diag16(x, kc=None):
    """jnp: `x [rows, K]` -> `[K/kc, rows*GPC, kc]` bf16 chunked block-diagonal LHS."""
    return jnp.stack(block_diag16_values(x, kc))


def block_diag16_to_ref(x_bd_ref, x):
    """In-kernel: write the chunked block-diagonal expansion of `x [rows, K]` into
    `x_bd_ref [K/kc, rows*GPC, kc]` bf16 (one aligned store per chunk)."""
    chunks, _, kc = x_bd_ref.shape
    for i in range(chunks):
        x_bd_ref[i] = block_diag16_chunk(x, kc, i)


def block_diag16_shape(K, rows):
    """Shape of the chunked group-16 block-diagonal LHS of a `[rows, K]` input."""
    kc = k_chunk(K)
    return (K // kc, rows * (kc // BLOCK), kc)


# ---------------------------------------------------------------------------------------------
# The dots
# ---------------------------------------------------------------------------------------------


def fp4_weights_chunk(w_ref, i, kc, dtype=FP8):
    """Rows `i*kc .. (i+1)*kc` of an int32 `[K/8, N]` slab as `[kc, N]` values of `dtype`
    (float8_e4m3fn: exact and free on TPU7x, the MXU consumes it at the int4 rate; bfloat16:
    exact, for the VPU dequant path)."""
    K8 = w_ref.shape[0]
    words = kc // FP4_PER_WORD
    w = w_ref[...] if words == K8 else w_ref[pl.ds(i * words, words), :]
    return pltpu.bitcast(w, FP4).astype(dtype)


def fp4_block_dot(x_bd, w_ref, s_ref, rows=None):
    """`x_bd` chunked group-16 block-diagonal LHS (`[K/KC, rows*GPC, KC]` bf16 VMEM ref, or the
    list of its chunk values, batch-major rows), `w_ref [K/8, N]` int32 VMEM ref (packed e2m1),
    `s_ref [K/KC, GPC, N]` e4m3 (or f32) VMEM ref -> f32 `[rows, N]` = `x @ (e2m1 * e4m3)`.

    Static K-chunk unroll, one MXU dot (bf16 x fp8, f32 accumulate) per chunk; the block scales
    are applied on the VPU: row `b` of the result is the masked sum over its aligned `[GPC, N]`
    block of partial products. `rows` defaults to `rows*GPC / GPC` of `x_bd`."""
    chunks = len(x_bd) if isinstance(x_bd, (list, tuple)) else x_bd.shape[0]
    K = w_ref.shape[0] * FP4_PER_WORD
    kc = K // chunks
    gpc = kc // BLOCK
    lhs_rows = x_bd[0].shape[0]
    rows = lhs_rows // gpc if rows is None else rows
    if rows * gpc != lhs_rows:
        raise ValueError(f"LHS has {lhs_rows} rows, expected {rows} rows x {gpc} groups")
    accs = [None] * rows
    for i in range(chunks):
        res = _dot(x_bd[i], fp4_weights_chunk(w_ref, i, kc))  # [rows*GPC, N]
        sc = s_ref[i].astype(F32)  # [GPC, N]
        for b in range(rows):
            term = jnp.sum(res[b * gpc : (b + 1) * gpc] * sc, axis=0, keepdims=True)
            accs[b] = term if accs[b] is None else accs[b] + term
    return accs[0] if rows == 1 else jnp.concatenate(accs, axis=0)


def scale_words(s_ref):
    """`s_ref [K/KC, GPC, N]` e4m3 (or f32) VMEM ref -> `[K/16, N]` int32 words holding the
    bf16 bits of every block scale in both halves (exact: e4m3 values are bf16-representable),
    so a stride-0 load of 8 words bitcast to bf16 is the scale broadcast over its 16 rows."""
    chunks, gpc, n = s_ref.shape
    bits = pltpu.bitcast(s_ref[...].astype(F32), I32)
    hi = bits & I32(-65536)
    return (hi | lax.shift_right_logical(hi, I32(16))).reshape(chunks * gpc, n)


def scale_block(sd_ref, j):
    """The `[16, N]` bf16 broadcast of block scale `j` from the `scale_words` ref `sd_ref`."""
    n = sd_ref.shape[1]
    if _interpret():  # the interpreter has no strided loads
        words = jnp.broadcast_to(sd_ref[pl.ds(j, 1), :], (SUBLANES, n))
    else:
        words = sd_ref[pl.ds(j, SUBLANES, stride=0), :]
    return pltpu.bitcast(words, BF16)


def fp4_dequant_dot(x, w_ref, sd_ref):
    """`x [rows, K]` bf16 value, `w_ref [K/8, N]` int32 VMEM ref (packed e2m1), `sd_ref [K/16, N]`
    int32 VMEM ref (`scale_words`) -> f32 `[rows, N]` = `x @ (e2m1 * e4m3)`: VPU dequant to bf16
    per 512-row K chunk, one bf16 x bf16 MXU dot per chunk (f32 accumulate)."""
    K = w_ref.shape[0] * FP4_PER_WORD
    kc = k_chunk(K)
    gpc = kc // BLOCK
    acc = None
    for i in range(K // kc):
        v = fp4_weights_chunk(w_ref, i, kc, BF16)  # [kc, N] exact
        sb = jnp.concatenate([scale_block(sd_ref, i * gpc + j) for j in range(gpc)], axis=0)
        term = _dot(x[:, i * kc : (i + 1) * kc].astype(BF16), v * sb)
        acc = term if acc is None else acc + term
    return acc


def apply_gate_up_gs(gu, gs, Is):
    """`gu [rows, 2*Is]` f32 x the gate / up global scales of `gs [8, 128]` (rows 0 / 1)."""
    lane = _iota(gu.shape, 1)
    scale = jnp.where(lane < Is, gs[0:1, 0:1], gs[1:2, 0:1])
    return gu * scale


def apply_down_gs(y, gs):
    """`y [rows, Hm]` f32 x the down global scale of `gs [8, 128]` (row 2)."""
    return y * gs[2:3, 0:1]


# ---------------------------------------------------------------------------------------------
# Scratch and DMAs (mirrors `moe.MoeScratch` / `moe.scratch_shapes` / `moe._expert_copies`)
# ---------------------------------------------------------------------------------------------


@dataclass
class Fp4ExpertWeights:
    """HBM refs of the per-rank NVFP4 expert arrays (`[L, E, ...]`, module docstring)."""

    gate_up_fp4: object
    gate_up_bs: object
    down_fp4: object
    down_bs: object
    expert_gs: object

    @classmethod
    def from_dict(cls, refs):
        return cls(*(refs[name] for name in FAMILIES))


FAMILIES = ("gate_up_fp4", "gate_up_bs", "down_fp4", "down_bs", "expert_gs")


@dataclass
class Fp4Scratch:
    """VMEM/semaphore scratch of the NVFP4 expert path (`scratch_shapes`); the bookkeeping
    tiles are the same as `moe.MoeScratch` so `moe.route_to_scratch` etc. work unchanged."""

    gate_up_fp4: object  # [SLOTS, Hm/8, 2*Is] int32
    gate_up_bs: object  # [SLOTS, Hm/KC, KC/16, 2*Is] e4m3
    down_fp4: object  # [SLOTS, Is/8, Hm] int32
    down_bs: object  # [SLOTS, Is/KC', KC'/16, Hm] e4m3
    expert_gs: object  # [SLOTS, 8, 128] f32
    h_bd: object  # block-diag path: [Hm/KC, B*32, KC] bf16 expansion of h1; dequant path: [1, 8, 128]
    hc_bd: object  # compact path: [Hm/KC, COMPACT_ROWS*32, KC] bf16 expansion of an expert's active rows
    gu_sd: object  # dequant path: [Hm/16, 2*Is] int32 scale words of the gate_up slot in compute
    dn_sd: object  # dequant path: [Is/16, Hm] int32 scale words of the down slot in compute
    gu_buf: object  # [SLOTS, B, 2*Is] f32: gate_up results handed from slot s to the down of s
    idx: object  # [MR, 128] int32 route tile
    w: object  # [MR, 128] f32 route weights
    done: object  # [MR, 128] int32
    y_out: object  # [K*B, Hm] f32
    wave_ids: object  # SMEM [WAVE_ROWS, SLOTS] int32
    stats: object  # SMEM [STATS_RING, 1 + COMPACT_ROWS] int32: active rows, compact row ids
    sems: object  # DMA semaphores [SLOTS, COPIES]

    @classmethod
    def bind(cls, refs):
        return cls(*refs)


def slot_shapes(cfg: Config, tp=8, slots=SLOTS):
    """`{family: (shape, dtype)}` of the per-slot VMEM expert buffers."""
    Hm, Is = cfg.moe_hidden, cfg.expert_hidden // tp
    kc_gu, kc_dn = k_chunk(Hm), k_chunk(Is)
    return {
        "gate_up_fp4": ((slots, Hm // FP4_PER_WORD, 2 * Is), I32),
        "gate_up_bs": ((slots, Hm // kc_gu, kc_gu // BLOCK, 2 * Is), FP8),
        "down_fp4": ((slots, Is // FP4_PER_WORD, Hm), I32),
        "down_bs": ((slots, Is // kc_dn, kc_dn // BLOCK, Hm), FP8),
        "expert_gs": ((slots, GS_ROWS, GS_LANES), F32),
    }


def scratch_shapes(cfg: Config, batch, tp=8, slots=SLOTS):
    """Scratch tuple for `Fp4Scratch.bind` (real config: 12 MiB fp4 slots + 1.5 MiB e4m3 scales
    + B*0.25 MiB block-diagonal buffer (B <= 4) or 1.5 MiB scale words (B = 8) + 0.5 MiB compact
    block-diagonal buffer (B > COMPACT_ROWS) + 1 MiB y_out)."""
    if batch > MR:
        raise ValueError(f"batch {batch} exceeds the MXU row block {MR}")
    Hm, Is = cfg.moe_hidden, cfg.expert_hidden // tp
    dequant = use_dequant(batch)
    tiny = (1, SUBLANES, LANES)
    return tuple(pltpu.VMEM(shape, dtype) for shape, dtype in slot_shapes(cfg, tp, slots).values()) + (
        pltpu.VMEM(tiny if dequant else block_diag16_shape(Hm, batch), BF16),
        pltpu.VMEM(block_diag16_shape(Hm, COMPACT_ROWS) if use_compact(batch) else tiny, BF16),
        pltpu.VMEM((Hm // BLOCK, 2 * Is) if dequant else tiny, I32),
        pltpu.VMEM((Is // BLOCK, Hm) if dequant else tiny, I32),
        pltpu.VMEM((slots, batch, 2 * Is), F32),
        pltpu.VMEM((MR, LANES), I32),
        pltpu.VMEM((MR, LANES), F32),
        pltpu.VMEM((MR, LANES), I32),
        pltpu.VMEM((cfg.top_k * batch, Hm), F32),
        pltpu.SMEM((WAVE_ROWS, slots), I32),
        pltpu.SMEM((STATS_RING, 1 + max(COMPACT_ROWS, 1)), I32),
        pltpu.SemaphoreType.DMA((slots, COPIES)),
    )


def scratch_bytes(cfg: Config, batch, tp=8, slots=SLOTS):
    """VMEM bytes of `scratch_shapes`."""
    total = 0
    for s in scratch_shapes(cfg, batch, tp, slots)[:-3]:
        n = 1
        for d in s.shape:
            n *= d
        total += n * jnp.dtype(s.dtype).itemsize
    return total


def expert_copies(weights: Fp4ExpertWeights, sc: Fp4Scratch, layer, expert, slot):
    """The five async copies of one expert into `slot` (start/wait them like `moe._start`)."""
    return SimpleNamespace(**{
        name: pltpu.make_async_copy(
            getattr(weights, name).at[layer, expert], getattr(sc, name).at[slot], sc.sems.at[slot, c]
        )
        for c, name in enumerate(FAMILIES)
    })


def start_copies(copies):
    for c in vars(copies).values():
        c.start()


def wait_copies(copies):
    for c in vars(copies).values():
        c.wait()


def gate_up_dot(sc: Fp4Scratch, slot, h1):
    """`h1 [B, Hm]` (bf16-valued) @ gate_up[slot] with block and global scales applied: f32
    `[B, 2*Is]`. Block-diagonal path: `sc.h_bd` must hold `block_diag16_to_ref(sc.h_bd, h1)`;
    dequant path: converts the slot's scales into `sc.gu_sd` first."""
    B = h1.shape[0]
    Is = sc.down_fp4.shape[1] * FP4_PER_WORD
    if use_dequant(B):
        sc.gu_sd[...] = scale_words(sc.gate_up_bs.at[slot])
        gu = fp4_dequant_dot(h1, sc.gate_up_fp4.at[slot], sc.gu_sd)
    else:
        gu = fp4_block_dot(sc.h_bd, sc.gate_up_fp4.at[slot], sc.gate_up_bs.at[slot], B)
    return apply_gate_up_gs(gu, sc.expert_gs[slot], Is)


def down_dot(sc: Fp4Scratch, slot, a):
    """`a [B, Is]` (bf16-valued) @ down[slot] with block and global scales: f32 `[B, Hm]`."""
    B, Is = a.shape
    if use_dequant(B):
        sc.dn_sd[...] = scale_words(sc.down_bs.at[slot])
        y = fp4_dequant_dot(a, sc.down_fp4.at[slot], sc.dn_sd)
    else:
        y = fp4_block_dot(block_diag16_values(a, k_chunk(Is)), sc.down_fp4.at[slot],
                          sc.down_bs.at[slot], B)
    return apply_down_gs(y, sc.expert_gs[slot])


# ---------------------------------------------------------------------------------------------
# Expert stream (mirrors `moe.expert_stream`; routing/bookkeeping helpers are shared with moe)
# ---------------------------------------------------------------------------------------------


def _batch_of(cfg: Config, sc: Fp4Scratch):
    return sc.y_out.shape[0] // cfg.top_k


def route_stats(consumed):
    """`consumed [MR, LANES]` route mask of one expert -> `(active row count, [first active
    row, second, ...])` scalars (`COMPACT_ROWS` ids; `MR` when there is no such row)."""
    row_b = _iota((MR, 1), 0)
    act = jnp.max(consumed.astype(I32), axis=1, keepdims=True) > 0  # [MR, 1]
    ids, after = [], jnp.full((MR, 1), False)
    for _ in range(max(COMPACT_ROWS, 1)):
        b = jnp.min(jnp.where(act & ~after, row_b, MR))
        ids.append(b)
        after = after | (row_b <= b)
    return jnp.sum(act.astype(I32)), ids


def _mark(sc: Fp4Scratch, picks):
    """Record the routes of the selected experts `picks` in `sc.done` (= "selected")."""
    sel = picks[0].consumed
    for p in picks[1:]:
        sel = sel | p.consumed
    sc.done[...] = jnp.where(sel, I32(1), sc.done[...])


def start_expert_stream(cfg: Config, layer, weights: Fp4ExpertWeights, sc: Fp4Scratch):
    """Issue the DMAs of the first wave of experts (call right after `moe.route_to_scratch`).
    B = 1: the static route-slot waves of `moe`; else the expert ids of the first two waves are
    recorded in `sc.wave_ids` rows 0 / 1 and their routes marked in `sc.done` (selected)."""
    from musespark import moe

    E = cfg.experts
    slots = sc.gate_up_fp4.shape[0]
    if _batch_of(cfg, sc) == 1:
        for s, k in enumerate(moe._b1_waves(cfg, slots)[0]):
            start_copies(expert_copies(weights, sc, layer, moe._route_expert(sc, k), s))
        return
    idx_tile = sc.idx[...]
    picks, pending = moe._wave_experts(idx_tile < E, idx_tile, E, slots)
    second, _ = moe._wave_experts(pending, idx_tile, E, slots)
    _mark(sc, picks + second)
    for s, p in enumerate(picks):
        sc.wave_ids[0, s] = p.expert
        sc.wave_ids[1, s] = second[s].expert

        @pl.when(p.has)
        def _start_first(p=p, s=s):
            start_copies(expert_copies(weights, sc, layer, p.expert, s))


def expert_stream(cfg: Config, layer, h1, weights: Fp4ExpertWeights, sc: Fp4Scratch,
                  started=False, after_wave=None, compute=True, dma=True):
    """NVFP4 twin of `moe.expert_stream`: run every distinct expert of the batch once and fill
    `sc.y_out` (`[K*B, Hm]` f32, row `k*B + b`, this rank's partial down-projection output,
    global scales applied). `h1 [B, Hm]` (bf16-valued). Same bookkeeping tiles as `moe`, so
    `moe.route_to_scratch` / `moe.finalize` are reused unchanged.

    B = 1: the static route-slot stream of `moe` (`after_wave(w)` per static wave). B >= 2: the
    distinct experts are streamed in selection order (ascending id) through the slot ring,
    `slots` experts per wave, in ONE runtime loop over the experts: iteration `e` waits for
    expert `e`, issues its gate_up (MXU) and then the down projection of expert `e - 1` (whose
    gate_up result is in `sc.gu_buf`) plus the refill of that slot with expert `e - 1 + slots`,
    all in one basic block (the MXU never drains on the per-expert dependency chain and the
    per-expert code is emitted once; iteration 0's down runs on stale data with all stores
    masked). The wave after next is selected at the end of each wave's first iteration
    (`sc.done` marks the routes selected so far). The epilogue runs the last down and calls
    `after_wave(0)` once (static; the deferred ring refills are flushed after the last expert
    DMA was issued, like `moe`'s `flush="last"`). `compute=False` / `dma=False` (profiling)
    skip the dots / the expert DMAs and their waits."""
    from musespark import moe

    B, Hm = h1.shape
    K, E = cfg.top_k, cfg.experts
    slots = sc.gate_up_fp4.shape[0]
    Is = sc.down_fp4.shape[1] * FP4_PER_WORD
    dequant = use_dequant(B)
    if sc.gu_buf.shape[1] != B:
        raise ValueError(f"scratch was sized for B={sc.gu_buf.shape[1]}, h1 has {B} rows")
    if not started:
        start_expert_stream(cfg, layer, weights, sc)
    h1 = h1.astype(BF16)
    if not dequant:
        block_diag16_to_ref(sc.h_bd, h1.astype(F32))

    def start(expert, slot):
        if dma:
            start_copies(expert_copies(weights, sc, layer, expert, slot))

    def gate_up(slot):
        """B = 1 path: wait for the slot's DMAs and run its gate_up (`compute=False`: zeros)."""
        if dma and B == 1:
            wait_copies(expert_copies(weights, sc, layer, 0, slot))
        if not compute:
            return jnp.zeros((B, 2 * Is), F32) + h1[:, :1].astype(F32)
        return gate_up_dot(sc, slot, h1)

    def down(slot, gu):
        gate, up = gu[:, :Is], gu[:, Is:]
        if not compute:
            return jnp.zeros((B, Hm), F32) + gate[:, :1]
        return down_dot(sc, slot, moe.r16(gate * jax.nn.sigmoid(gate) * up))

    if B == 1:
        # Static stream (as `moe.expert_stream`): route slot k is expert k of the tile, wave w
        # holds route slots w*slots.., y_out row k gets a plain store; one-expert skew.
        waves = moe._b1_waves(cfg, slots)
        for w, routes in enumerate(waves):
            nxt = waves[w + 1] if w + 1 < len(waves) else []
            gu = [gate_up(0)]
            for s, k in enumerate(routes):
                if s + 1 < len(routes):
                    gu.append(gate_up(s + 1))
                sc.y_out[k : k + 1, :] = down(s, gu[s])[0:1]
                if s < len(nxt):
                    start(moe._route_expert(sc, nxt[s]), s)
            if after_wave is not None:
                after_wave(w)
        return

    idx_tile = sc.idx[...]
    valid = idx_tile < E
    n = moe._distinct_count(cfg, idx_tile)

    def id_at(e):
        """Expert id of stream position `e >= 0` (wave `e // slots`, `sc.wave_ids` ring)."""
        return sc.wave_ids[(e // slots) % WAVE_ROWS, e % slots]

    def finish(expert, slot, live):
        """Down projection of the expert in `slot` (gate_up result in `sc.gu_buf[slot]`) and the
        masked stores of its routes (`live`: scalar, False for the dummy of iteration 0)."""
        y = down(slot, sc.gu_buf[slot])
        moe._store_rows(sc, y, valid & (idx_tile == expert) & live, B, K)

    def refill(e, slot, live):
        """Refill `slot` (expert `e` done) with expert `e + slots` when it exists."""
        nxt = id_at(e + slots)

        @pl.when(live & (nxt < E))
        def _refill():
            start(nxt, slot)

    def select_ahead(e):
        """Select the wave after the one of stream position `e` + 1 (its ring row is free: that
        wave finished before `e`'s wave started) from the routes not selected so far."""
        ahead, _ = moe._wave_experts(valid & (sc.done[...] == 0), idx_tile, E, slots)
        _mark(sc, ahead)
        row = (e // slots + 2) % WAVE_ROWS
        for t in range(slots):
            sc.wave_ids[row, t] = ahead[t].expert

    R = COMPACT_ROWS
    row_b = _iota((MR, 1), 0)

    def routes(expert, live):
        """`[MR, LANES]` mask of the routes of `expert` (`live`: scalar, False for the dummy)."""
        return valid & (idx_tile == expert) & live

    def store_stats(e):
        """`route_stats` of stream position `e` -> `sc.stats[e % STATS_RING]` (computed one
        iteration ahead, inside the main block, so the reductions overlap the MXU work)."""
        count, ids = route_stats(routes(id_at(e), True))
        sc.stats[e % STATS_RING, 0] = count
        for r, b in enumerate(ids):
            sc.stats[e % STATS_RING, 1 + r] = b

    def stats_at(e):
        """`(active row count, [picked [MR, 1] masks: row b is compact row r])` of stream
        position `e` from the SMEM ring."""
        count = sc.stats[e % STATS_RING, 0]
        picked = [row_b == sc.stats[e % STATS_RING, 1 + r] for r in range(R)]
        return count, picked

    def compact_gate_up(slot, picked):
        """gate_up of the expert in `slot` on its <= R active rows (`sc.hc_bd`): rows 0..R-1
        of `sc.gu_buf[slot]`."""
        hc = jnp.concatenate(
            [jnp.sum(jnp.where(p[0:B], h1.astype(F32), F32(0)), axis=0, keepdims=True) for p in picked],
            0,
        )
        block_diag16_to_ref(sc.hc_bd, hc)
        gu = fp4_block_dot(sc.hc_bd, sc.gate_up_fp4.at[slot], sc.gate_up_bs.at[slot], R)
        sc.gu_buf[slot, 0:R, :] = apply_gate_up_gs(gu, sc.expert_gs[slot], Is)

    def compact_finish(slot, consumed, picked):
        """Down projection of the compact rows of `slot`, scattered back to the batch rows."""
        yc = down(slot, sc.gu_buf[slot, 0:R, :])  # [R, Hm]
        y = None
        for r, p in enumerate(picked):
            term = jnp.where(p[0:B], yc[r : r + 1, :], F32(0))  # [B, Hm]
            y = term if y is None else y + term
        moe._store_rows(sc, y, consumed, B, K)

    def gate_up_of(e, slot, compact):
        if not compact:
            sc.gu_buf[slot] = gate_up_dot(sc, slot, h1) if compute else gate_up(slot)
            return
        count, picked = stats_at(e)

        @pl.when(count <= R)
        def _compact():
            compact_gate_up(slot, picked)

        @pl.when(count > R)
        def _full():
            sc.gu_buf[slot] = gate_up_dot(sc, slot, h1)

    def finish_of(e, slot, live, compact):
        expert = id_at(e)
        if not compact:
            finish(expert, slot, live)
            return
        count, picked = stats_at(e)

        @pl.when(count <= R)
        def _compact():
            compact_finish(slot, routes(expert, live), picked)

        @pl.when(count > R)
        def _full():
            finish(expert, slot, live)

    def run_loop(compact):
        """One expert per iteration: gate_up(e) (MXU) ahead of down(e - 1) in one basic block
        (two slots in compute, two in flight); iteration 0's down(-1) is a masked dummy. (Two
        experts per iteration measured slower: 15.8 / 21.6 / 20.4 vs 14.4 / 20.8 / 18.2 us per
        layer at B = 2 / 4 / 8, also with 6 slots.) `compact`: every expert with <= COMPACT_ROWS
        active rows runs the block-diagonal dots on a compact `[COMPACT_ROWS, Hm]` LHS (64 MXU
        rows) instead of the full-batch formulation; the per-expert route stats are computed
        one expert ahead into the SMEM ring."""
        if compact:
            store_stats(0)

        @pl.loop(0, n)
        def _step(e):
            slot = e % slots
            prev = jnp.maximum(e - 1, 0)  # expert e - 1 (a masked dummy at e = 0)
            pslot = (e + slots - 1) % slots
            if dma:
                wait_copies(expert_copies(weights, sc, layer, 0, slot))
            gate_up_of(e, slot, compact)
            finish_of(prev, pslot, e >= 1, compact)
            refill(prev, pslot, e >= 1)
            if compact:
                store_stats(e + 1)  # position n reads a sentinel / stale id: never consumed

            @pl.when(slot == 0)
            def _select():
                select_ahead(e)

        finish_of(n - 1, (n - 1) % slots, True, compact)

    if use_compact(B) and compute:
        # The compact loop pays ~0.25 us per expert for its conditional blocks and wins ~0.7 us
        # per expert with <= COMPACT_ROWS active rows: take it when the mean active rows per
        # expert (B*K / n) is <= 3, i.e. for genuinely different rows (measured L=8: distinct
        # rows B=4 / 8: 52.8 -> 41.9, 72.2 -> 56.8 us per layer; identical rows unchanged).
        many = 3 * n >= B * K

        @pl.when(many)
        def _compact_loop():
            run_loop(True)

        @pl.when(~many)
        def _plain_loop():
            run_loop(False)
    else:
        run_loop(False)

    if after_wave is not None:
        after_wave(0)


# ---------------------------------------------------------------------------------------------
# Prefill dequant kernel (bf16 experts for `lax.ragged_dot`)
# ---------------------------------------------------------------------------------------------


def _dequant_kernel(p_ref, s_ref, g_ref, o_ref):
    """One `[KC/8, N]` int32 chunk -> `[KC, N]` bf16 = `e2m1 * e4m3 * gs` (f32 maths, exactly the
    XLA path of `quant.dequant_fp4_jnp`)."""
    kc, n = o_ref.shape
    v = pltpu.bitcast(p_ref[...], FP4).astype(F32).reshape(kc // BLOCK, BLOCK, n)
    w = (v * s_ref[...].astype(F32)[:, None, :]).reshape(kc, n) * g_ref[...]
    o_ref[...] = w.astype(BF16)


def dequant_fp4_pallas(packed, bs, gs):
    """`packed [E, K/8, N]` int32, `bs [E, K/KC, KC/16, N]` e4m3, `gs [E, 1, N]` f32 (a
    per-column global scale, broadcast for the down projection) -> bf16 `[E, K, N]`.

    Bit-identical to `quant.dequant_fp4_jnp(...).astype(bf16)` but ~10x faster on TPU: the
    int32 -> fp4 relayout is a free `pltpu.bitcast` instead of XLA's minor-axis transpose."""
    E, K8, N = packed.shape
    chunks, gpc = bs.shape[1], bs.shape[2]
    kc = gpc * BLOCK
    K = K8 * FP4_PER_WORD
    if chunks * kc != K:
        raise ValueError(f"block scales {bs.shape} do not match K={K}")
    return pl.pallas_call(
        _dequant_kernel,
        out_shape=jax.ShapeDtypeStruct((E, K, N), BF16),
        grid=(E, chunks),
        in_specs=[
            pl.BlockSpec((None, kc // FP4_PER_WORD, N), lambda e, c: (e, c, 0)),
            pl.BlockSpec((None, None, gpc, N), lambda e, c: (e, c, 0, 0)),
            pl.BlockSpec((None, 1, N), lambda e, c: (e, 0, 0)),
        ],
        out_specs=pl.BlockSpec((None, kc, N), lambda e, c: (e, c, 0)),
        interpret=jax.default_backend() != "tpu",
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(packed, lax.bitcast_convert_type(bs, FP8) if bs.dtype == jnp.uint8 else bs, gs.astype(F32))


def dequantized_expert_layer(w, l, isl, pallas=None):
    """This rank's experts of layer `l` from the fp4 families of a weight tree `w` as bf16
    `(gate_up [E, Hm, 2*Is], down [E, Is, Hm])`. `pallas` (default: on TPU) selects
    `dequant_fp4_pallas`, otherwise the XLA path (`quant.dequant_fp4_jnp`); both give the
    same bits."""
    gs = w["expert_gs"][l]  # [E, 8, 128]
    E = gs.shape[0]
    col = jnp.arange(2 * isl, dtype=I32)[None, :] < isl
    gu_gs = jnp.where(col, gs[:, 0:1, 0:1], gs[:, 1:2, 0:1])  # [E, 1, 2*Is]
    dn_gs = gs[:, 2:3, 0:1]  # [E, 1, 1]
    pallas = jax.default_backend() == "tpu" if pallas is None else pallas
    if pallas:
        Hm = w["down_fp4"].shape[-1]
        gate_up = dequant_fp4_pallas(w["gate_up_fp4"][l], w["gate_up_bs"][l], gu_gs)
        down = dequant_fp4_pallas(w["down_fp4"][l], w["down_bs"][l], jnp.broadcast_to(dn_gs, (E, 1, Hm)))
        return gate_up, down
    gate_up = dequant_fp4_jnp(w["gate_up_fp4"][l], w["gate_up_bs"][l], gu_gs)
    down = dequant_fp4_jnp(w["down_fp4"][l], w["down_bs"][l], dn_gs)
    return gate_up.astype(BF16), down.astype(BF16)
