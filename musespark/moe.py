"""MoE mini-kernel for the Muse Spark decode megakernel (design.md section 5.4).

Everything here runs inside the grid-less decode `pallas_call` (or a test wrapper around it)
and is generic in `Config`; the MINI config exercises the identical code.

Per-rank expert layout (TP8 inside every expert, `Is = I / tp`, `KC = quant.k_chunk(K)`,
`G = cfg.group_size`, `GPC = KC / G` groups per chunk):

    gate_up_q [L, E, Hm, 2*Is]             int4  columns [0, Is) gate, [Is, 2*Is) up
    gate_up_s [L, E, Hm/KC, KC/G, 2*Is]    f32   group scales, K-chunk major
    down_q    [L, E, Is, Hm]               int4
    down_s    [L, E, Is/KC', KC'/G, Hm]    f32

The int4 slabs are DMA'd as int4 into VMEM slots and fed straight to the MXU
(`jnp.dot(bf16, int4)`); group scales are applied with the block-diagonal-LHS trick:
`block_diag(x)` expands the `[MR, K]` bf16 input into `[K/KC, GPC*MR, KC]` (chunk `i` holds the
block-diagonal `[(g, m), k]` = `x[m, k]` if `k` is in group `g` of that chunk, else 0), so one
`[GPC*MR, KC] x [KC, N]` dot yields every group's partial product as a separate row block and
the scales become a VPU multiply (hw report section 3). `MR = 8` rows are always computed
(rows `>= B` are zero padding); the MXU cost is flat for M <= 8.

Routing (spec 3.5): `logits = dot(x, hi) + dot(x, lo)` in f32 (bf16 hi/lo split of the fp32
router), `sc = sigmoid(logits * output_multiplier)`, selection on `sc + bias` by K iterations of
argmax (ties -> lowest index, winner masked to -inf), mixing weights = the UNBIASED `sc` of the
winners divided by `(sum + route_eps)`. Results are lane-dense `[B, 128]` tiles (lane `k < K`
holds route slot `k`; other lanes hold the sentinel `E` / weight 0) because the expert loop is a
runtime loop. In scratch the tiles are `[MR, 128]` with rows `>= B` set to the sentinel.

Expert stream: each DISTINCT expert of the batch is processed exactly once, in ascending
expert id (min-pending-id stream over the route tile), in waves of `SLOTS = 4` experts whose
DMAs are issued as early as possible (the first wave right after routing, each slot refilled
as soon as its expert's dots have been issued). Per expert
`gu = int4_group_dot(h1_bd, gate_up)`, `a = r16(silu(gu[:, :Is]) * gu[:, Is:])`,
`y = int4_group_dot(block_diag(a), down)`; `y[b]` is written to `y_out[k*B + b]` for every
route `(b, k)` of that expert (masked stores, no per-row control flow).

    y_out [K*B, Hm] f32 : row k*B + b = this rank's PARTIAL down-projection output of the
                          expert routed to batch row b at route slot k (k-major).

`y_out` must be all-reduced across the 8 ranks by the integrator; `finalize` then applies the
post_expert_norm and the routing weights (the norm needs the full sum, spec 3.5):
`t = r16(r16(Y) * w_post)`, `yn = t * rsqrt(mean(t^2) + post_eps)`, `m[b] = r16(sum_k w[b,k] yn)`.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from musespark.config import Config
from musespark.quant import k_chunk

BF16 = jnp.bfloat16
F32 = jnp.float32
I32 = jnp.int32

MR = 8  # row block of every MXU LHS (the decode batch is padded to this many rows)
LANES = 128  # lane-dense bookkeeping tiles are [MR, LANES]
SLOTS = 4  # int4 expert slots (2 in compute, 2 in flight)
COPIES = 4  # DMAs per expert: gate_up_q, gate_up_s, down_q, down_s


def r16(x):
    """Round an f32 array to bf16 precision (stays f32)."""
    return x.astype(BF16).astype(F32)


def _dot(a, b):
    return jnp.dot(a, b, preferred_element_type=F32)


def _iota(shape, axis):
    return lax.broadcasted_iota(I32, shape, axis)


# ---------------------------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------------------------


def router_logits(x_ffn, router_hi, router_lo):
    """`[B, H]` bf16-valued x, `[H, E]` bf16 hi/lo -> f32 `[B, E]` logits (dot(x,hi) + dot(x,lo))."""
    x = x_ffn.astype(BF16)
    return _dot(x, router_hi) + _dot(x, router_lo)


def route_from_logits(cfg: Config, logits, bias, width=LANES):
    """Top-k routing from f32 logits `[B, E]` and selection bias `[1, E]` f32 (spec 3.5).

    Returns `(idx, w)`, both `[B, width]`: lane `k < cfg.top_k` holds the k-th selected expert
    (int32) and its mixing weight (f32, unbiased sigmoid score / (sum + route_eps)); lanes
    `>= top_k` hold `cfg.experts` (the "no expert" sentinel) and 0. Ties resolve to the lowest
    expert index. With `width=None` the result is `[B, top_k]`.
    """
    B, E = logits.shape
    K = cfg.top_k
    width = K if width is None else width
    sc = jax.nn.sigmoid(logits * F32(cfg.output_multiplier))
    sel = sc + bias.astype(F32)
    expert_ids = _iota((B, E), 1)
    lane = _iota((B, width), 1)
    idx = jnp.full((B, width), E, I32)
    w = jnp.zeros((B, width), F32)
    total = jnp.zeros((B, 1), F32)
    for k in range(K):
        best = jnp.max(sel, axis=1, keepdims=True)
        winner = jnp.min(jnp.where(sel == best, expert_ids, E), axis=1, keepdims=True)
        winner_mask = expert_ids == winner
        score = jnp.sum(jnp.where(winner_mask, sc, F32(0)), axis=1, keepdims=True)
        sel = jnp.where(winner_mask, -jnp.inf, sel)
        total = total + score
        idx = jnp.where(lane == k, winner, idx)
        w = jnp.where(lane == k, score, w)
    return idx, w / (total + F32(cfg.route_eps))


def route(cfg: Config, x_ffn, router_hi, router_lo, bias, width=LANES):
    """`x_ffn [B, H]` (bf16-valued), `router_hi/lo [H, E]` bf16, `bias [1, E]` f32 ->
    `(idx, w)` `[B, width]` tiles as in `route_from_logits`."""
    return route_from_logits(cfg, router_logits(x_ffn, router_hi, router_lo), bias, width)


# ---------------------------------------------------------------------------------------------
# Block-diagonal group-scaled int4 dot
# ---------------------------------------------------------------------------------------------


def block_diag_chunk(x, group, kc, i):
    """`x [MR, K]` (any float) -> chunk `i` of the block-diagonal LHS: `[GPC*MR, kc]` bf16 with
    row block `g` = `x[:, i*kc:(i+1)*kc]` masked to the lanes of group `g`."""
    rows = x.shape[0]
    gpc = kc // group
    xk = x[:, i * kc : (i + 1) * kc].astype(F32)
    tiled = jnp.concatenate([xk] * gpc, axis=0) if gpc > 1 else xk
    mask = _iota((gpc * rows, kc), 0) // rows == _iota((gpc * rows, kc), 1) // group
    return jnp.where(mask, tiled, F32(0)).astype(BF16)


def block_diag_values(x, group, kc=None):
    """`x [MR, K]` -> list of `K/kc` chunk values (`block_diag_chunk` for every chunk)."""
    K = x.shape[1]
    kc = k_chunk(K) if kc is None else kc
    return [block_diag_chunk(x, group, kc, i) for i in range(K // kc)]


def block_diag(x, group, kc=None):
    """jnp: `x [MR, K]` -> `[K/kc, GPC*MR, kc]` bf16 chunked block-diagonal LHS."""
    return jnp.stack(block_diag_values(x, group, kc))


def block_diag_to_ref(x_bd_ref, x, group):
    """In-kernel: write the chunked block-diagonal expansion of `x [MR, K]` into
    `x_bd_ref [K/kc, GPC*MR, kc]` bf16 (one aligned store per chunk)."""
    chunks, _, kc = x_bd_ref.shape
    for i in range(chunks):
        x_bd_ref[i] = block_diag_chunk(x, group, kc, i)


def block_diag_shape(K, group, rows=MR):
    """Shape of the chunked block-diagonal LHS of a `[rows, K]` input."""
    kc = k_chunk(K)
    return (K // kc, (kc // group) * rows, kc)


def int4_group_dot(x_bd, w_ref, s_ref, rows=MR):
    """`x_bd` chunked block-diagonal LHS (`[K/KC, GPC*rows, KC]` bf16 VMEM ref, or the list of
    its chunk values), `w_ref [K, N]` int4 VMEM ref, `s_ref [K/KC, GPC, N]` f32 VMEM ref ->
    f32 `[rows, N]` = `x @ dequant(w)`. Static K-chunk unroll; one MXU dot per chunk."""
    chunks = len(x_bd) if isinstance(x_bd, (list, tuple)) else x_bd.shape[0]
    K = w_ref.shape[0]
    kc = K // chunks
    acc = None
    for i in range(chunks):
        lhs = x_bd[i]
        w = w_ref[...] if chunks == 1 else w_ref[pl.ds(i * kc, kc), :]
        res = _dot(lhs, w)  # [GPC*rows, N]; row block j = x_group_j @ w_group_j
        sc = s_ref[i]  # [GPC, N]
        for j in range(res.shape[0] // rows):
            term = res[j * rows : (j + 1) * rows] * sc[j : j + 1]
            acc = term if acc is None else acc + term
    return acc


# ---------------------------------------------------------------------------------------------
# Scratch
# ---------------------------------------------------------------------------------------------


@dataclass
class MoeScratch:
    """VMEM/semaphore scratch of the MoE mini-kernel (see `scratch_shapes`)."""

    gate_up_q: object  # [SLOTS, Hm, 2*Is] int4
    gate_up_s: object  # [SLOTS, Hm/KC, KC/G, 2*Is] f32
    down_q: object  # [SLOTS, Is, Hm] int4
    down_s: object  # [SLOTS, Is/KC', KC'/G, Hm] f32
    h_pad: object  # [MR, Hm] f32: h1 padded to MR rows
    h_bd: object  # [Hm/KC, GPC*MR, KC] bf16: block-diagonal expansion of h_pad
    idx: object  # [MR, 128] int32 route tile (sentinel E outside [0:B, 0:K])
    w: object  # [MR, 128] f32 route weights (0 outside [0:B, 0:K])
    done: object  # [MR, 128] int32: 1 where the route has been processed
    y_out: object  # [K*B, Hm] f32 per-(row, slot) partial expert outputs (k-major rows)
    wave_ids: object  # SMEM [2, SLOTS] int32: expert ids of waves w and w+1 (by wave parity)
    sems: object  # DMA semaphores [SLOTS, COPIES]

    @classmethod
    def bind(cls, refs):
        return cls(*refs)


def scratch_shapes(cfg: Config, batch, tp=8, slots=SLOTS):
    """Scratch tuple for `MoeScratch.bind` (real config: 12 MiB slots + 0.75 MiB scales +
    0.4 MiB block-diagonal/padding buffers + 1 MiB y_out for B=8)."""
    if batch > MR:
        raise ValueError(f"batch {batch} exceeds the MXU row block {MR}")
    if cfg.top_k > LANES:
        raise ValueError("top_k must fit in one 128-lane bookkeeping tile")
    Hm, G = cfg.moe_hidden, cfg.group_size
    Is = cfg.expert_hidden // tp
    kc_gu, kc_dn = k_chunk(Hm), k_chunk(Is)
    return (
        pltpu.VMEM((slots, Hm, 2 * Is), jnp.int4),
        pltpu.VMEM((slots, Hm // kc_gu, kc_gu // G, 2 * Is), F32),
        pltpu.VMEM((slots, Is, Hm), jnp.int4),
        pltpu.VMEM((slots, Is // kc_dn, kc_dn // G, Hm), F32),
        pltpu.VMEM((MR, Hm), F32),
        pltpu.VMEM(block_diag_shape(Hm, G), BF16),
        pltpu.VMEM((MR, LANES), I32),
        pltpu.VMEM((MR, LANES), F32),
        pltpu.VMEM((MR, LANES), I32),
        pltpu.VMEM((cfg.top_k * batch, Hm), F32),
        pltpu.SMEM((2, slots), I32),
        pltpu.SemaphoreType.DMA((slots, COPIES)),
    )


def scratch_bytes(cfg: Config, batch, tp=8, slots=SLOTS):
    """VMEM bytes of `scratch_shapes` (int4 counted at 4 bits)."""
    total = 0
    for s in scratch_shapes(cfg, batch, tp, slots)[:-2]:
        n = 1
        for d in s.shape:
            n *= d
        total += n // 2 if s.dtype == jnp.int4 else n * jnp.dtype(s.dtype).itemsize
    return total


# ---------------------------------------------------------------------------------------------
# Expert stream
# ---------------------------------------------------------------------------------------------


@dataclass
class ExpertWeights:
    """HBM refs of the per-rank expert arrays (`[L, E, ...]`, see the module docstring)."""

    gate_up_q: object
    gate_up_s: object
    down_q: object
    down_s: object


def _expert_copies(weights: ExpertWeights, sc: MoeScratch, layer, expert, slot):
    return SimpleNamespace(
        gate_up_q=pltpu.make_async_copy(
            weights.gate_up_q.at[layer, expert], sc.gate_up_q.at[slot], sc.sems.at[slot, 0]
        ),
        gate_up_s=pltpu.make_async_copy(
            weights.gate_up_s.at[layer, expert], sc.gate_up_s.at[slot], sc.sems.at[slot, 1]
        ),
        down_q=pltpu.make_async_copy(
            weights.down_q.at[layer, expert], sc.down_q.at[slot], sc.sems.at[slot, 2]
        ),
        down_s=pltpu.make_async_copy(
            weights.down_s.at[layer, expert], sc.down_s.at[slot], sc.sems.at[slot, 3]
        ),
    )


def _start(copies):
    for c in vars(copies).values():
        c.start()


def _wait(copies):
    for c in vars(copies).values():
        c.wait()


def route_to_scratch(cfg: Config, sc: MoeScratch, idx_tile, w_tile):
    """Materialise `[B, 128]` route tiles (from `route`) into the `[MR, 128]` scratch tiles
    (sentinel `E` / weight 0 outside `[0:B, 0:top_k]`) and reset the per-layer bookkeeping."""
    B = idx_tile.shape[0]
    E, K = cfg.experts, cfg.top_k
    lane = _iota(idx_tile.shape, 1)
    sc.idx[...] = jnp.full(sc.idx.shape, E, I32)
    sc.w[...] = jnp.zeros(sc.w.shape, F32)
    sc.idx[0:B, :] = jnp.where(lane < K, idx_tile, E)
    sc.w[0:B, :] = jnp.where(lane < K, w_tile, F32(0))
    sc.done[...] = jnp.zeros(sc.done.shape, I32)


def _pick(pending, idx_tile, E):
    """Lowest pending expert id: `(expert, has, consumed_routes, pending_after)`.

    The selection chain stays in the vector unit (`[1, 1]` reductions broadcast back); only the
    DMA index / `has` flag take the slow vector-to-scalar path, off the chain's critical path.
    """
    candidates = jnp.where(pending, idx_tile, E)
    lowest = jnp.min(jnp.min(candidates, axis=1, keepdims=True), axis=0, keepdims=True)
    consumed = pending & (idx_tile == lowest)
    expert = jnp.min(candidates)
    return expert, expert < E, consumed, pending & ~consumed


def _wave_experts(pending, idx_tile, E, slots):
    picks = []
    for _ in range(slots):
        expert, has, consumed, pending = _pick(pending, idx_tile, E)
        picks.append(SimpleNamespace(expert=expert, has=has, consumed=consumed))
    return picks, pending


def _distinct_count(cfg: Config, idx_tile):
    """Number of distinct experts among the routes of the tile (scalar int32)."""
    E = cfg.experts
    lane = _iota(idx_tile.shape, 1)
    ids = _iota((idx_tile.shape[0], E), 1)
    present = jnp.zeros((1, E), jnp.bool_)
    for k in range(cfg.top_k):
        col = jnp.sum(jnp.where(lane == k, idx_tile, 0), axis=1, keepdims=True)
        present = present | jnp.any(ids == col, axis=0, keepdims=True)
    return jnp.sum(present.astype(I32))


def start_expert_stream(cfg: Config, layer, weights: ExpertWeights, sc: MoeScratch):
    """Issue the DMAs of the first wave of experts (call right after `route_to_scratch`).

    Also records the expert ids of the first two waves in `sc.wave_ids`."""
    E = cfg.experts
    slots = sc.gate_up_q.shape[0]
    idx_tile = sc.idx[...]
    picks, pending = _wave_experts(idx_tile < E, idx_tile, E, slots)
    second, _ = _wave_experts(pending, idx_tile, E, slots)
    for s, p in enumerate(picks):
        sc.wave_ids[0, s] = p.expert
        sc.wave_ids[1, s] = second[s].expert

        @pl.when(p.has)
        def _start_first(p=p, s=s):
            _start(_expert_copies(weights, sc, layer, p.expert, s))


def _store_rows(sc: MoeScratch, y, consumed, batch, K):
    """Write `y[b]` to `y_out[k*B + b]` for every consumed route `(b, k)` of this expert
    (one masked `[B, Hm]` store per route slot; no per-row control flow)."""
    B, Hm = batch, y.shape[1]
    y = y[0:B]
    for k in range(K):
        mask = jnp.broadcast_to(consumed[0:B, k : k + 1], (B, Hm))
        pltpu.store(sc.y_out.at[pl.ds(k * B, B), :], y, mask=mask)


def expert_stream(cfg: Config, layer, h1, weights: ExpertWeights, sc: MoeScratch, started=False):
    """Run every distinct expert of the batch once; fills `sc.y_out` (see the module docstring).

    `h1 [B, Hm]` (bf16-valued f32 or bf16): pre_expert_norm output, identical on every rank.
    `sc.idx` must hold the route tile (`route_to_scratch`). With `started=True` the first
    wave's DMAs were already issued by `start_expert_stream`.
    """
    B, Hm = h1.shape
    K, E, G = cfg.top_k, cfg.experts, cfg.group_size
    slots = sc.gate_up_q.shape[0]
    Is = sc.down_q.shape[1]
    kc_dn = k_chunk(Is)

    if not started:
        start_expert_stream(cfg, layer, weights, sc)

    # Block-diagonal expansion of the (row-padded) expert input, once per layer.
    if B == MR:
        h = h1.astype(F32)
    else:
        sc.h_pad[...] = jnp.zeros((MR, Hm), F32)
        sc.h_pad[0:B, :] = h1.astype(F32)
        h = sc.h_pad[...]
    block_diag_to_ref(sc.h_bd, h, G)

    idx_tile = sc.idx[...]
    n_waves = (_distinct_count(cfg, idx_tile) + slots - 1) // slots

    def gate_up(slot):
        _wait(_expert_copies(weights, sc, layer, 0, slot))
        return int4_group_dot(sc.h_bd, sc.gate_up_q.at[slot], sc.gate_up_s.at[slot])

    def finish(slot, gu, consumed):
        gate, up = gu[:, :Is], gu[:, Is:]
        a = r16(gate * jax.nn.sigmoid(gate) * up)
        y = int4_group_dot(block_diag_values(a, G, kc_dn), sc.down_q.at[slot], sc.down_s.at[slot])
        _store_rows(sc, y, consumed, B, K)

    def wave_body(cur, nxt, refill, select_ahead=None):
        """One wave of `slots` experts; `refill` in {"always", "cond", "never"}.

        Program order is skewed by one expert (gate_up of slot s+1 is issued before the
        down projection of slot s) so the MXU never drains on the per-expert dependency chain.
        `select_ahead` (the selection of the wave after next) is placed behind the first two
        gate_up dots so its reductions overlap MXU work instead of delaying the wave.
        """
        gu = [gate_up(0)]
        for s in range(slots):
            if s + 1 < slots:
                gu.append(gate_up(s + 1))
            if s == 0 and select_ahead is not None:
                select_ahead()
            finish(s, gu[s], cur[s].consumed)
            if refill == "always":
                _start(_expert_copies(weights, sc, layer, nxt[s].expert, s))
            elif refill == "cond":

                @pl.when(nxt[s].has)
                def _refill(s=s):
                    _start(_expert_copies(weights, sc, layer, nxt[s].expert, s))

    @pl.loop(0, n_waves)
    def _wave(w):
        # The experts of this wave (DMAs in flight) and of the next one are known from SMEM
        # (`start_expert_stream` / the previous wave); this wave selects the wave after next.
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
            ahead, _ = _wave_experts(pending, idx_tile, E, slots)
            for s in range(slots):
                sc.wave_ids[parity, s] = ahead[s].expert

        full = cur[-1].has
        next_full = nxt[-1].has
        next_any = nxt[0].has

        # Steady state: the next wave is full, refills are unconditional (one basic block).
        @pl.when(full & next_full)
        def _steady():
            wave_body(cur, nxt, "always", select_ahead)

        # Full wave before a partial one: per-slot conditional refills.
        @pl.when(full & next_any & ~next_full)
        def _before_tail():
            wave_body(cur, nxt, "cond")

        # Last full wave: nothing to refill.
        @pl.when(full & ~next_any)
        def _last_full():
            wave_body(cur, nxt, "never")

        # Partial last wave.
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


# ---------------------------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------------------------


def finalize(cfg: Config, Y, w, post_expert_norm, batch):
    """`Y [K*B, Hm]` f32 (all-reduced `y_out`, row `k*B + b`), `w [>= B, >= K]` route weights
    (the `w` scratch tile or the `route` output), `post_expert_norm [1, Hm]` bf16-valued ->
    `m [B, Hm]` bf16-valued f32 (spec 3.5).

    Per route: `t = r16(r16(Y) * w_post)`, `yn = t * rsqrt(mean(t^2) + post_eps)`; then
    `m[b] = r16(sum_k w[b, k] * yn[k*B + b])` in f32. `Y` and `w` may be refs or values.
    """
    B, K = batch, cfg.top_k
    w_post = post_expert_norm.astype(F32)

    def normed(rows):
        t = r16(r16(rows) * w_post)
        return t * lax.rsqrt(jnp.mean(t * t, axis=1, keepdims=True) + F32(cfg.post_eps))

    if B == 1 and K == MR:
        # One tile-row holds all K routes: normalise once, weight by a sublane column, reduce.
        yn = normed(Y[0:K, :])
        w_row = jnp.broadcast_to(w[0:1, :], (K, w.shape[1]))
        w_col = jnp.sum(
            jnp.where(_iota(w_row.shape, 1) == _iota(w_row.shape, 0), w_row, F32(0)),
            axis=1,
            keepdims=True,
        )
        return r16(jnp.sum(w_col * yn, axis=0, keepdims=True))
    m = None
    for k in range(K):
        term = w[0:B, k : k + 1] * normed(Y[k * B : (k + 1) * B, :])
        m = term if m is None else m + term
    return r16(m)
