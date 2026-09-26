"""NVFP4 expert path of the Muse Spark decode megakernel (kernel-side helpers, container v2).

Mirrors `musespark.moe` for the `expert_format == "nvfp4"` families (`musespark.layout`):

    gate_up_fp4 [L, E, Hm/8, 2*Is]          int32  eight e2m1 codes per word along K
    gate_up_bs  [L, E, Hm/KC, KC/16, 2*Is]  e4m3   block-16 scales, K-chunk major
    down_fp4    [L, E, Is/8, Hm]            int32
    down_bs     [L, E, Is/KC, KC/16, Hm]    e4m3
    expert_gs   [L, E, 8, 128]              f32    rows 0/1/2 = gate/up/down global scale

The fastest faithful path measured in the feasibility study (scratchpad notes, variant (c2)):
the int32 slab is bitcast in VMEM to `float4_e2m1fn` and converted for free to `float8_e4m3fn`
(`pltpu.bitcast(w, fp4).astype(fp8)`), `jnp.dot(x_bf16, w_fp8)` runs on the MXU at the int4 rate
with f32 accumulation, and the block-16 e4m3 scales are applied post-MXU with the block-diagonal
LHS idiom of `moe.int4_group_dot` at group 16: `block_diag16(x)` expands the `[B, K]` input into
`[K/KC, 32*B, KC]` chunks (32 groups of 16 per 512-chunk), each chunk dot yields every group's
partial product as a separate row block, `acc += res[j*B:(j+1)*B] * s[j]` with the e4m3 scale
chunk converted to f32 in-kernel (B = 1: one `sum(res * s, axis=0)`). Products `e2m1 * e4m3`
are exact, so the result equals `x @ (e2m1 * e4m3)` up to f32 summation order; the per-expert
global scales are three scalar multiplies (`apply_gate_up_gs`, `apply_down_gs`) on the
`[B, N]` results. Unlike the int4 path the LHS uses the REAL batch rows B (not MR = 8): the MXU
cost of the tall block-diagonal LHS grows with 32*B (0.85 us per gate_up slice at B=1, 2.6 us at
B=8 on one TPU7x core), so the block-diagonal buffers are sized by the batch.

Integration into `moe.expert_stream` (per expert, `sc` holding the fp4 scratch of
`scratch_shapes`, five DMAs per expert from `Fp4ExpertWeights` via `expert_copies`):

    gu = gate_up_dot(sc, slot, rows=B)               # [B, 2*Is] f32, global scales applied
    a = r16(silu(gu[:, :Is]) * gu[:, Is:])
    y = down_dot(sc, slot, a)                         # [B, Hm] f32 partial, global scale applied
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


def _dot(a, b):
    return jnp.dot(a, b, preferred_element_type=F32)


def _iota(shape, axis):
    return lax.broadcasted_iota(I32, shape, axis)


# ---------------------------------------------------------------------------------------------
# Block-diagonal LHS with group 16
# ---------------------------------------------------------------------------------------------


def block_diag16_chunk(x, kc, i):
    """`x [rows, K]` (any float) -> chunk `i` of the group-16 block-diagonal LHS: `[GPC*rows, kc]`
    bf16 (`GPC = kc / 16`), row block `g` = `x[:, i*kc:(i+1)*kc]` masked to the lanes of group `g`."""
    rows = x.shape[0]
    gpc = kc // BLOCK
    xk = x[:, i * kc : (i + 1) * kc].astype(F32)
    tiled = jnp.concatenate([xk] * gpc, axis=0) if gpc > 1 else xk
    mask = _iota((gpc * rows, kc), 0) // rows == _iota((gpc * rows, kc), 1) // BLOCK
    return jnp.where(mask, tiled, F32(0)).astype(BF16)


def block_diag16_values(x, kc=None):
    """`x [rows, K]` -> list of the `K/kc` chunk values (`block_diag16_chunk`)."""
    K = x.shape[1]
    kc = k_chunk(K) if kc is None else kc
    return [block_diag16_chunk(x, kc, i) for i in range(K // kc)]


def block_diag16(x, kc=None):
    """jnp: `x [rows, K]` -> `[K/kc, GPC*rows, kc]` bf16 chunked block-diagonal LHS."""
    return jnp.stack(block_diag16_values(x, kc))


def block_diag16_to_ref(x_bd_ref, x):
    """In-kernel: write the chunked block-diagonal expansion of `x [rows, K]` into
    `x_bd_ref [K/kc, GPC*rows, kc]` bf16 (one aligned store per chunk)."""
    chunks, _, kc = x_bd_ref.shape
    for i in range(chunks):
        x_bd_ref[i] = block_diag16_chunk(x, kc, i)


def block_diag16_shape(K, rows):
    """Shape of the chunked group-16 block-diagonal LHS of a `[rows, K]` input."""
    kc = k_chunk(K)
    return (K // kc, (kc // BLOCK) * rows, kc)


# ---------------------------------------------------------------------------------------------
# The dot
# ---------------------------------------------------------------------------------------------


def fp4_weights_chunk(w_ref, i, kc):
    """Rows `i*kc .. (i+1)*kc` of an int32 `[K/8, N]` slab as `[kc, N]` float8_e4m3fn values
    (exact: e2m1 is a subset of e4m3); a free bitcast + convert on TPU7x."""
    K8 = w_ref.shape[0]
    words = kc // FP4_PER_WORD
    w = w_ref[...] if words == K8 else w_ref[pl.ds(i * words, words), :]
    return pltpu.bitcast(w, FP4).astype(FP8)


def fp4_block_dot(x_bd, w_ref, s_ref, rows=None):
    """`x_bd` chunked group-16 block-diagonal LHS (`[K/KC, GPC*rows, KC]` bf16 VMEM ref, or the
    list of its chunk values), `w_ref [K/8, N]` int32 VMEM ref (packed e2m1), `s_ref
    [K/KC, GPC, N]` e4m3 (or f32) VMEM ref -> f32 `[rows, N]` = `x @ (e2m1 * e4m3)`.

    Static K-chunk unroll, one MXU dot (bf16 x fp8, f32 accumulate) per chunk; the block scales
    are applied on the VPU per row block (B = 1: one masked reduction over the 32 row blocks).
    `rows` defaults to `GPC*rows / GPC` of `x_bd`."""
    chunks = len(x_bd) if isinstance(x_bd, (list, tuple)) else x_bd.shape[0]
    K = w_ref.shape[0] * FP4_PER_WORD
    kc = K // chunks
    gpc = kc // BLOCK
    lhs_rows = x_bd[0].shape[0]
    rows = lhs_rows // gpc if rows is None else rows
    if rows * gpc != lhs_rows:
        raise ValueError(f"LHS has {lhs_rows} rows, expected {gpc} groups x {rows} rows")
    acc = None
    for i in range(chunks):
        res = _dot(x_bd[i], fp4_weights_chunk(w_ref, i, kc))  # [GPC*rows, N]
        sc = s_ref[i].astype(F32)  # [GPC, N]
        if rows == 1:
            term = jnp.sum(res * sc, axis=0, keepdims=True)
            acc = term if acc is None else acc + term
            continue
        for j in range(gpc):
            term = res[j * rows : (j + 1) * rows] * sc[j : j + 1]
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
    h_bd: object  # [Hm/KC, 32*B, KC] bf16: group-16 block-diagonal expansion of h1 (B rows)
    idx: object  # [MR, 128] int32 route tile
    w: object  # [MR, 128] f32 route weights
    done: object  # [MR, 128] int32
    y_out: object  # [K*B, Hm] f32
    wave_ids: object  # SMEM [2, SLOTS] int32
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
    """Scratch tuple for `Fp4Scratch.bind` (real config, B = 8: 12 MiB fp4 slots + 1.5 MiB
    e4m3 scales + 2 MiB block-diagonal buffer + 1 MiB y_out)."""
    if batch > MR:
        raise ValueError(f"batch {batch} exceeds the MXU row block {MR}")
    Hm = cfg.moe_hidden
    return tuple(pltpu.VMEM(shape, dtype) for shape, dtype in slot_shapes(cfg, tp, slots).values()) + (
        pltpu.VMEM(block_diag16_shape(Hm, batch), BF16),
        pltpu.VMEM((MR, LANES), I32),
        pltpu.VMEM((MR, LANES), F32),
        pltpu.VMEM((MR, LANES), I32),
        pltpu.VMEM((cfg.top_k * batch, Hm), F32),
        pltpu.SMEM((2, slots), I32),
        pltpu.SemaphoreType.DMA((slots, COPIES)),
    )


def scratch_bytes(cfg: Config, batch, tp=8, slots=SLOTS):
    """VMEM bytes of `scratch_shapes`."""
    total = 0
    for s in scratch_shapes(cfg, batch, tp, slots)[:-2]:
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


def gate_up_dot(sc: Fp4Scratch, slot, rows):
    """`h1_bd @ gate_up[slot]` with block and global scales applied: f32 `[rows, 2*Is]`
    (`sc.h_bd` must hold `block_diag16_to_ref(sc.h_bd, h1)` of the `[rows, Hm]` input)."""
    Is = sc.down_fp4.shape[1] * FP4_PER_WORD
    gu = fp4_block_dot(sc.h_bd, sc.gate_up_fp4.at[slot], sc.gate_up_bs.at[slot], rows)
    return apply_gate_up_gs(gu, sc.expert_gs[slot], Is)


def down_dot(sc: Fp4Scratch, slot, a):
    """`a [rows, Is]` (bf16-valued) @ down[slot] with block and global scales: f32 `[rows, Hm]`."""
    Is = sc.down_fp4.shape[1] * FP4_PER_WORD
    y = fp4_block_dot(block_diag16_values(a, k_chunk(Is)), sc.down_fp4.at[slot], sc.down_bs.at[slot],
                      a.shape[0])
    return apply_down_gs(y, sc.expert_gs[slot])


# ---------------------------------------------------------------------------------------------
# Expert stream (mirrors `moe.expert_stream`; routing/bookkeeping helpers are shared with moe)
# ---------------------------------------------------------------------------------------------


def start_expert_stream(cfg: Config, layer, weights: Fp4ExpertWeights, sc: Fp4Scratch):
    """Issue the DMAs of the first wave of experts (call right after `moe.route_to_scratch`)
    and record the expert ids of the first two waves in `sc.wave_ids` (as `moe.start_expert_stream`)."""
    from musespark import moe

    E = cfg.experts
    slots = sc.gate_up_fp4.shape[0]
    idx_tile = sc.idx[...]
    picks, pending = moe._wave_experts(idx_tile < E, idx_tile, E, slots)
    second, _ = moe._wave_experts(pending, idx_tile, E, slots)
    for s, p in enumerate(picks):
        sc.wave_ids[0, s] = p.expert
        sc.wave_ids[1, s] = second[s].expert

        @pl.when(p.has)
        def _start_first(p=p, s=s):
            start_copies(expert_copies(weights, sc, layer, p.expert, s))


def expert_stream(cfg: Config, layer, h1, weights: Fp4ExpertWeights, sc: Fp4Scratch,
                  started=False, after_wave=None):
    """NVFP4 twin of `moe.expert_stream`: run every distinct expert of the batch once and fill
    `sc.y_out` (`[K*B, Hm]` f32, row `k*B + b`, this rank's partial down-projection output,
    global scales applied). Same wave/slot/refill structure and the same bookkeeping tiles, so
    `moe.route_to_scratch` / `moe.finalize` and the integrator's `after_wave` hook are reused
    unchanged. `h1 [B, Hm]` (bf16-valued): the block-diagonal expansion uses the real B rows
    (`sc.h_bd` is `[Hm/KC, 32*B, KC]`)."""
    from musespark import moe

    B, Hm = h1.shape
    K, E = cfg.top_k, cfg.experts
    slots = sc.gate_up_fp4.shape[0]
    Is = sc.down_fp4.shape[1] * FP4_PER_WORD
    if sc.h_bd.shape[1] != (sc.h_bd.shape[2] // BLOCK) * B:
        raise ValueError(f"h_bd was sized for {sc.h_bd.shape[1] // (sc.h_bd.shape[2] // BLOCK)} "
                         f"rows, h1 has {B}")
    if not started:
        start_expert_stream(cfg, layer, weights, sc)

    block_diag16_to_ref(sc.h_bd, h1.astype(F32))
    idx_tile = sc.idx[...]
    n_waves = (moe._distinct_count(cfg, idx_tile) + slots - 1) // slots

    def gate_up(slot):
        wait_copies(expert_copies(weights, sc, layer, 0, slot))
        return gate_up_dot(sc, slot, B)

    def finish(slot, gu, consumed):
        gate, up = gu[:, :Is], gu[:, Is:]
        a = moe.r16(gate * jax.nn.sigmoid(gate) * up)
        moe._store_rows(sc, down_dot(sc, slot, a), consumed, B, K)

    def wave_body(cur, nxt, refill, select_ahead=None):
        gu = [gate_up(0)]
        for s in range(slots):
            if s + 1 < slots:
                gu.append(gate_up(s + 1))
            if s == 0 and select_ahead is not None:
                select_ahead()
            finish(s, gu[s], cur[s].consumed)
            if refill == "always":
                start_copies(expert_copies(weights, sc, layer, nxt[s].expert, s))
            elif refill == "cond":

                @pl.when(nxt[s].has)
                def _refill(s=s):
                    start_copies(expert_copies(weights, sc, layer, nxt[s].expert, s))

    @pl.loop(0, n_waves)
    def _wave(w):
        pending = (idx_tile < E) & (sc.done[...] == 0)
        parity = w % 2
        cur, nxt = [], []
        for group, slot_ids in ((cur, sc.wave_ids.at[parity]), (nxt, sc.wave_ids.at[1 - parity])):
            for s in range(slots):
                expert = slot_ids[s]
                consumed = pending & (idx_tile == expert)
                pending = pending & ~consumed
                group.append(SimpleNamespace(expert=expert, has=expert < E, consumed=consumed))

        def select_ahead(pending=pending):
            ahead, _ = moe._wave_experts(pending, idx_tile, E, slots)
            for s in range(slots):
                sc.wave_ids[parity, s] = ahead[s].expert

        full, next_full, next_any = cur[-1].has, nxt[-1].has, nxt[0].has

        @pl.when(full & next_full)
        def _steady():
            wave_body(cur, nxt, "always", select_ahead)

        @pl.when(full & next_any & ~next_full)
        def _before_tail():
            wave_body(cur, nxt, "cond")

        @pl.when(full & ~next_any)
        def _last_full():
            wave_body(cur, nxt, "never")

        @pl.when(~full)
        def _tail():
            for s in range(slots):

                @pl.when(cur[s].has)
                def _one(s=s):
                    finish(s, gate_up(s), cur[s].consumed)

        consumed = cur[0].consumed
        for p in cur[1:]:
            consumed = consumed | p.consumed
        sc.done[...] = jnp.where(consumed, I32(1), sc.done[...])
        if after_wave is not None:
            after_wave(w)
