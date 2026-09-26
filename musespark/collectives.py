"""In-kernel collectives over the 8 TensorCores of the "tp" mesh axis (design.md 5.1).

All ranks are peers `rank ^ offset` addressed with `pl.DeviceIdType.MESH`; every collective
is symmetric (each rank sends to the 7 others and waits for its 7 sends and 7 receives), so
the DMA semaphores are the only synchronisation. The kernel must set
`pltpu.CompilerParams(collective_id=..., has_side_effects=True)` and call `barrier()` once
at entry (it uses the barrier semaphore that `collective_id` allocates).

Payload layouts (f32 unless noted, `R` a multiple of 8 rows):

    all_reduce_rows(x [R, W])         -> [R, W]      W a multiple of tp*128; every rank gets the
                                                     bit-identical sum (fixed slot order 0..tp-1)
    all_gather_rows(x [R, w] f32|bf16)-> [R, tp*w]   rank r's shard in columns r*w:(r+1)*w

Wire layout. A column block `[R, wb]` travels as a "wire tile" `[fr, fl]` window of the
`[..., wire_rows, 1024]` buffers: lane chunks of a block wider than 1024 are stacked along
rows, and tile-aligned row groups of a block narrower than 1024 are placed side by side
(`Wire`); both are free relayouts (128-lane / 8-row aligned slices). Blocks that fold
neither way use a zero-padded `[roundup(R, tile), roundup(wb, 128)]` window.

Hierarchical all-reduce (`all_reduce_rows(..., hierarchical=True)`, needs
`scratch_shapes(..., hierarchical=True)`): the 8 cores are 4 chips x 2 cores and the sibling
core `rank ^ 1` is reachable at 100-600 GB/s while every ICI link is ~45 GB/s per core
(hwbench/hier_bench.py). Rank `r` owns half `r % 2` of the columns: (1) pair reduce-scatter
of the halves over the sibling link, (2) reduce-scatter of the half's 4 quarters across the
4 chips (one message per link: quarter `j` -> core `2j + r % 2`), (3) all-gather of the
reduced quarters across the chips, (4) pair all-gather of the halves. Cross-chip bytes halve
(the sibling's contribution is pre-reduced) at the price of 4 latency phases instead of 2, so
it only pays for the large payloads (`[64, 4096]`: 13.5 vs 14.5 us bf16 wire, 19.9 vs 25.4 us
f32; `[8, 8192]`: 9.3 vs 7.3 us bf16). Summation order: sibling pair (commutative, 2
operands), then chips 0..3 -- fixed, so every rank gets the identical result.

Two slots suffice (`phase % 2`). Every collective ends only after all 7 sends AND 7 receives
of the calling rank completed, so when rank A starts collective c+1 it has already received
rank B's messages of c, i.e. B has *started* c, but B may still be reading its receive
buffers of c: that is why c+1 uses the other slot. Collective c+2 from A starts after A
finished c+1, which needed B's c+1 sends, which B issued only after it finished c (and thus
consumed the buffers of c), so the slot of c is free again. The argument needs only that the
caller passes a counter incremented by exactly one per collective on every rank (the kinds
may interleave arbitrarily: each kind has its own buffers and semaphores anyway).
"""

from dataclasses import dataclass
from types import SimpleNamespace

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

AXIS = "tp"
TP = 8
WIRE_LANES = 1024
_MESH = pl.DeviceIdType.MESH


def _peer_offsets(tp):
    # Descending offsets: the far peers first, the sibling core (offset 1) last.
    return tuple(range(tp - 1, 0, -1))


def barrier(tp=TP):
    """Signal the tp-1 peers on the barrier semaphore and wait for their signals."""
    rank = lax.axis_index(AXIS)
    sem = pltpu.get_barrier_semaphore()
    for offset in _peer_offsets(tp):
        pl.semaphore_signal(sem, 1, device_id=(rank ^ offset,), device_id_type=_MESH)
    pl.semaphore_wait(sem, tp - 1)


def _round_up(x, m):
    return -(-x // m) * m


@dataclass(frozen=True)
class Wire:
    """How a `[rows, width]` block is laid out in a `[fr, fl]` buffer window.

    `stack`: lane chunks of `fl` stacked along rows; `fold`: `fr`-row groups side by side;
    `pad`: zero padding to `[fr, fl]`. Rows are `row_tile`-aligned (8 f32 / 16 bf16).
    """

    rows: int
    width: int
    fr: int
    fl: int
    mode: str

    def to_wire(self, block):
        if self.mode == "stack":
            chunks = self.width // self.fl
            return jnp.concatenate(
                [block[:, c * self.fl : (c + 1) * self.fl] for c in range(chunks)], 0
            )
        if self.mode == "fold":
            groups = self.fl // self.width
            return jnp.concatenate(
                [block[g * self.fr : (g + 1) * self.fr] for g in range(groups)], 1
            )
        if (self.fr, self.fl) == (self.rows, self.width):
            return block
        return jnp.pad(block, ((0, self.fr - self.rows), (0, self.fl - self.width)))

    def from_wire(self, tile):
        if self.mode == "stack":
            chunks = self.width // self.fl
            return jnp.concatenate(
                [tile[c * self.rows : (c + 1) * self.rows] for c in range(chunks)], 1
            )
        if self.mode == "fold":
            groups = self.fl // self.width
            return jnp.concatenate(
                [tile[:, g * self.width : (g + 1) * self.width] for g in range(groups)], 0
            )
        return tile[: self.rows, : self.width]


def _wire(rows, width, row_tile=8):
    """Choose the wire layout of a `[rows, width]` block (see `Wire`)."""
    if rows % 8:
        raise ValueError(f"payload rows must be a multiple of 8, got {rows}")
    if width > WIRE_LANES:
        if width % WIRE_LANES or (rows * width // WIRE_LANES) % row_tile:
            raise ValueError(f"block [{rows}, {width}] cannot be stacked into {WIRE_LANES} lanes")
        return Wire(rows, width, rows * width // WIRE_LANES, WIRE_LANES, "stack")
    fold = WIRE_LANES // width
    if width % 128 == 0 and fold * width == WIRE_LANES and rows % (row_tile * fold) == 0:
        return Wire(rows, width, rows // fold, WIRE_LANES, "fold")
    if rows % row_tile == 0:
        return Wire(rows, width, rows, _round_up(width, 128), "pad")
    chunks = row_tile // rows
    if rows * chunks == row_tile and width % (chunks * 128) == 0:
        return Wire(rows, width, row_tile, width // chunks, "stack")
    return Wire(rows, width, _round_up(rows, row_tile), _round_up(width, 128), "pad")


def wire_rows(rows, width, tp=TP, row_tile=8):
    """Wire rows needed by an all-reduce of `[rows, width]` (its blocks are `[rows, width/tp]`)."""
    return _wire(rows, width // tp, row_tile).fr


def hier_shapes(rows, width, tp=TP, wire=jnp.bfloat16):
    """Buffers + semaphores of the hierarchical all-reduce of `[rows, width]` (`wire` dtype)."""
    if tp != 8:
        raise ValueError("the hierarchical all-reduce assumes 8 cores = 4 chips x 2")
    rt = 16 if wire == jnp.bfloat16 else 8
    half = _wire(rows, width // 2, rt)
    quarter = _wire(rows, width // 8, rt)
    return (
        pltpu.VMEM((half.fr, half.fl), wire),  # h_send: the sibling's half of my partial
        pltpu.VMEM((2, half.fr, half.fl), wire),  # h_recv[slot]: the sibling's copy of my half
        pltpu.VMEM((4, quarter.fr, quarter.fl), wire),  # q_send[chip j]
        pltpu.VMEM((2, 4, quarter.fr, quarter.fl), wire),  # q_recv[slot, source chip]
        pltpu.VMEM((2, 4, quarter.fr, quarter.fl), wire),  # q_ag[slot, owner chip]
        pltpu.VMEM((2, half.fr, half.fl), wire),  # h_ag[slot]: my reduced half (AG source)
        pltpu.SemaphoreType.DMA((2, 1)),  # pair RS send / recv
        pltpu.SemaphoreType.DMA((2, 1)),
        pltpu.SemaphoreType.DMA((2, 3)),  # chip RS
        pltpu.SemaphoreType.DMA((2, 3)),
        pltpu.SemaphoreType.DMA((2, 3)),  # chip AG
        pltpu.SemaphoreType.DMA((2, 3)),
        pltpu.SemaphoreType.DMA((2, 1)),  # pair AG
        pltpu.SemaphoreType.DMA((2, 1)),
    )


_HIER_NAMES = (
    "h_send", "h_recv", "q_send", "q_recv", "q_ag", "h_ag",
    "hp_send_sems", "hp_recv_sems", "hq_send_sems", "hq_recv_sems",
    "ha_send_sems", "ha_recv_sems", "hg_send_sems", "hg_recv_sems",
)


def scratch_shapes(
    rows, width=8192, tp=TP, gather_rows=8, gather_width=None, bf16_wire=False, f32_wire=True,
    hierarchical=False,
):
    """Scratch for the collectives, in the order `workspace()` consumes it.

    `rows x width` is the largest all-reduce payload (`[64, 4096]` for the expert outputs at
    B=8 also covers `[8, 8192]`); `gather_rows x gather_width` the largest all-gather shard
    (default `[8, width // tp]`). Sizes for (64, 4096): 1 + 2 + 2 + 0.25 + 0.25 MiB with the
    f32 wire, plus / or half of that with `bf16_wire` (the bf16-payload all-reduce buffers;
    `f32_wire=False` drops the f32 ones, then only `wire=jnp.bfloat16` reductions are legal).
    `hierarchical=True` appends the `hier_shapes` of the largest payload (bf16 wire unless
    only the f32 wire is allocated).
    """
    if gather_width is None:
        gather_width = width // tp
    wr = max(8, wire_rows(rows, width, tp))
    gw = _wire(gather_rows, gather_width)
    gr, gl = max(8, gw.fr), max(128, gw.fl)
    gw16 = _wire(gather_rows, gather_width, 16)
    gr16, gl16 = max(16, gw16.fr), max(128, gw16.fl)
    shapes = ()
    if f32_wire:
        shapes += (
            pltpu.VMEM((tp, wr, WIRE_LANES), jnp.float32),  # send: my column blocks, block-major
            pltpu.VMEM((2, tp, wr, WIRE_LANES), jnp.float32),  # rs_recv[slot, source rank]
            pltpu.VMEM((2, tp, wr, WIRE_LANES), jnp.float32),  # ag_recv[slot, owner rank]
        )
    shapes += (
        pltpu.VMEM((2, tp, gr, gl), jnp.float32),  # g_recv_f32[slot, owner rank]
        pltpu.VMEM((2, tp, gr16, gl16), jnp.bfloat16),  # g_recv_bf16
        pltpu.SemaphoreType.DMA((2, tp - 1)),  # rs_send
        pltpu.SemaphoreType.DMA((2, tp - 1)),  # rs_recv
        pltpu.SemaphoreType.DMA((2, tp - 1)),  # ag_send
        pltpu.SemaphoreType.DMA((2, tp - 1)),  # ag_recv
        pltpu.SemaphoreType.DMA((2, tp - 1)),  # g_send
        pltpu.SemaphoreType.DMA((2, tp - 1)),  # g_recv
    )
    if bf16_wire:
        wr16 = max(16, wire_rows(rows, width, tp, 16))
        shapes += (
            pltpu.VMEM((tp, wr16, WIRE_LANES), jnp.bfloat16),  # send16
            pltpu.VMEM((2, tp, wr16, WIRE_LANES), jnp.bfloat16),  # rs_recv16
            pltpu.VMEM((2, tp, wr16, WIRE_LANES), jnp.bfloat16),  # ag_recv16
        )
    if hierarchical:
        shapes += hier_shapes(rows, width, tp, jnp.bfloat16 if bf16_wire else jnp.float32)
    return shapes


_WS_NAMES32 = ("send", "rs_recv", "ag_recv")
_WS_NAMES = (
    "g_recv_f32",
    "g_recv_bf16",
    "rs_send_sems",
    "rs_recv_sems",
    "ag_send_sems",
    "ag_recv_sems",
    "g_send_sems",
    "g_recv_sems",
)
_WS_NAMES16 = ("send16", "rs_recv16", "ag_recv16")


def workspace(*refs, f32_wire=True, hierarchical=False):
    """Bind the refs allocated from `scratch_shapes(..., f32_wire=f32_wire,
    hierarchical=hierarchical)` (same order) into a namespace; whether the bf16 wire buffers
    are present is inferred from the count."""
    names = (_WS_NAMES32 if f32_wire else ()) + _WS_NAMES
    tail = _HIER_NAMES if hierarchical else ()
    if len(refs) == len(names) + len(_WS_NAMES16) + len(tail):
        names += _WS_NAMES16
    elif len(refs) != len(names) + len(tail):
        raise ValueError(
            f"workspace expects {len(names) + len(tail)} or "
            f"{len(names) + len(_WS_NAMES16) + len(tail)} refs, got {len(refs)}"
        )
    names += tail
    ws = SimpleNamespace(**dict(zip(names, refs)))
    for name in _WS_NAMES32 + _WS_NAMES16 + _HIER_NAMES:
        if not hasattr(ws, name):
            setattr(ws, name, None)
    ws.tp = ws.g_recv_f32.shape[1]
    return ws


def _is_ref(x):
    from jax._src.state import types as state_types

    return isinstance(x, state_types.TransformedRef) or isinstance(
        getattr(x, "aval", None), state_types.AbstractRef
    )


def _value(x):
    return x[...] if _is_ref(x) else x


def _exchange(src_view, dst_view, send_sems, recv_sems, slot, rank, tp, peer_src=None):
    """Copy `src_view` to `dst_view` on every peer; wait for the sends and the receives.

    `peer_src(peer) -> view` selects a peer-specific source (reduce-scatter); otherwise the
    same `src_view` goes to everyone (all-gather).
    """
    copies = []
    for j, offset in enumerate(_peer_offsets(tp)):
        peer = rank ^ offset
        src = src_view if peer_src is None else peer_src(peer)
        copy = pltpu.make_async_remote_copy(
            src,
            dst_view,
            send_sems.at[slot, j],
            recv_sems.at[slot, j],
            device_id=(peer,),
            device_id_type=_MESH,
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()


def _exchange_pairs(pairs, send_sems, recv_sems, slot):
    """`pairs`: `(src_view, dst_view, peer)` remote copies; start all, wait all."""
    copies = []
    for j, (src, dst, peer) in enumerate(pairs):
        copy = pltpu.make_async_remote_copy(
            src, dst, send_sems.at[slot, j], recv_sems.at[slot, j], device_id=(peer,),
            device_id_type=_MESH,
        )
        copy.start()
        copies.append(copy)
    for copy in copies:
        copy.wait()


def _all_reduce_hier(x, ws, phase, wire):
    """The hierarchical all-reduce (module docstring); `x` f32 `[R, W]`, W % 1024 == 0."""
    if ws.h_send is None:
        raise ValueError("hierarchical all-reduce needs scratch_shapes(..., hierarchical=True)")
    if jnp.dtype(ws.h_send.dtype) != jnp.dtype(wire):
        raise ValueError(f"hierarchical scratch was allocated for the {ws.h_send.dtype} wire")
    rank = lax.axis_index(AXIS)
    even = (rank % 2) == 0
    chip = rank // 2
    slot = phase % 2
    rows, width = x.shape
    rt = 16 if wire == jnp.bfloat16 else 8
    hw = _wire(rows, width // 2, rt)
    qw = _wire(rows, width // 8, rt)
    if hw.fr > ws.h_send.shape[0] or qw.fr > ws.q_send.shape[1]:
        raise ValueError(f"payload [{rows}, {width}] exceeds the hierarchical scratch")
    hwin = (pl.ds(0, hw.fr), pl.ds(0, hw.fl))
    qwin = (pl.ds(0, qw.fr), pl.ds(0, qw.fl))
    w2, w8 = width // 2, width // 8
    xw = x.astype(wire)
    lo, hi = xw[:, :w2], xw[:, w2:]
    mine, other = jnp.where(even, lo, hi), jnp.where(even, hi, lo)
    # (1) pair reduce-scatter: the sibling receives its half of my partial
    ws.h_send[hwin[0], hwin[1]] = hw.to_wire(other)
    _exchange_pairs(
        [(ws.h_send.at[hwin[0], hwin[1]], ws.h_recv.at[slot, hwin[0], hwin[1]], rank ^ 1)],
        ws.hp_send_sems, ws.hp_recv_sems, slot,
    )
    half = mine.astype(jnp.float32) + hw.from_wire(ws.h_recv[slot, hwin[0], hwin[1]]).astype(
        jnp.float32
    )
    # (2) chip reduce-scatter of the quarters: quarter j -> core (2j + rank % 2)
    for j in range(4):
        ws.q_send[j, qwin[0], qwin[1]] = qw.to_wire(half[:, j * w8 : (j + 1) * w8].astype(wire))
    pairs = []
    for offset in (2, 4, 6):
        peer = rank ^ offset
        pairs.append(
            (ws.q_send.at[peer // 2, qwin[0], qwin[1]], ws.q_recv.at[slot, chip, qwin[0], qwin[1]],
             peer)
        )
    _exchange_pairs(pairs, ws.hq_send_sems, ws.hq_recv_sems, slot)
    ws.q_recv[slot, chip, qwin[0], qwin[1]] = ws.q_send[chip, qwin[0], qwin[1]]
    total = ws.q_recv[slot, 0, qwin[0], qwin[1]].astype(jnp.float32)
    for j in range(1, 4):
        total = total + ws.q_recv[slot, j, qwin[0], qwin[1]].astype(jnp.float32)
    ws.q_ag[slot, chip, qwin[0], qwin[1]] = total.astype(wire)
    # (3) chip all-gather of the reduced quarters
    src = ws.q_ag.at[slot, chip, qwin[0], qwin[1]]
    _exchange_pairs(
        [(src, src, rank ^ offset) for offset in (2, 4, 6)], ws.ha_send_sems, ws.ha_recv_sems,
        slot,
    )
    half_red = jnp.concatenate(
        [qw.from_wire(ws.q_ag[slot, j, qwin[0], qwin[1]]) for j in range(4)], axis=1
    ).astype(jnp.float32)
    # (4) pair all-gather of the halves (lands in h_recv, free again after step 1)
    ws.h_ag[slot, hwin[0], hwin[1]] = hw.to_wire(half_red.astype(wire))
    _exchange_pairs(
        [(ws.h_ag.at[slot, hwin[0], hwin[1]], ws.h_recv.at[slot, hwin[0], hwin[1]], rank ^ 1)],
        ws.hg_send_sems, ws.hg_recv_sems, slot,
    )
    other_red = hw.from_wire(ws.h_recv[slot, hwin[0], hwin[1]]).astype(jnp.float32)
    return jnp.concatenate(
        [jnp.where(even, half_red, other_red), jnp.where(even, other_red, half_red)], axis=1
    )


def all_reduce_rows(x, ws, phase, wire=jnp.float32, hierarchical=False):
    """Sum an f32 `[R, W]` partial over the tp ranks; bit-identical result everywhere.

    Reduce-scatter over the tp column blocks (rank j sums block j of all ranks in slot order
    0..tp-1) followed by an all-gather of the reduced blocks. `x` may be a value or a VMEM
    ref; `phase` is the traced collective counter (see the module docstring). With
    `wire=jnp.bfloat16` (needs `scratch_shapes(..., bf16_wire=True)`) the partials and the
    reduced blocks travel as bf16: half the bytes, but every partial is rounded to bf16
    before the f32 summation and the result is bf16-valued (the caller's `r16` is then a
    no-op). `hierarchical=True` uses the chip-aware 4-phase scheme (module docstring).
    """
    tp = ws.tp
    rank = lax.axis_index(AXIS)
    slot = phase % 2
    x = _value(x).astype(jnp.float32)
    rows, width = x.shape
    if width % (tp * 128):
        raise ValueError(f"all_reduce_rows width {width} must be a multiple of {tp * 128}")
    if hierarchical:
        return _all_reduce_hier(x, ws, phase, wire)
    wb = width // tp
    if wire == jnp.bfloat16:
        if ws.send16 is None:
            raise ValueError("bf16 wire needs scratch_shapes(..., bf16_wire=True)")
        send, rs_recv, ag_recv, row_tile = ws.send16, ws.rs_recv16, ws.ag_recv16, 16
        x = x.astype(jnp.bfloat16)
    else:
        if ws.send is None:
            raise ValueError("f32 wire needs scratch_shapes(..., f32_wire=True)")
        send, rs_recv, ag_recv, row_tile = ws.send, ws.rs_recv, ws.ag_recv, 8
    w = _wire(rows, wb, row_tile)
    if w.fr > send.shape[1] or w.fl > send.shape[2]:
        raise ValueError(f"payload [{rows}, {width}] exceeds the collectives scratch")
    win = (pl.ds(0, w.fr), pl.ds(0, w.fl))

    # Stage my column blocks in wire layout, block-major, so peers are addressed by index.
    for j in range(tp):
        send[j, win[0], win[1]] = w.to_wire(x[:, j * wb : (j + 1) * wb])
    own = rs_recv.at[slot, rank, win[0], win[1]]
    _exchange(
        None,
        own,
        ws.rs_send_sems,
        ws.rs_recv_sems,
        slot,
        rank,
        tp,
        peer_src=lambda peer: send.at[peer, win[0], win[1]],
    )
    # My own block goes into slot `rank` locally (peers never write that slot here), then
    # the blocks are summed in strict slot order 0..tp-1 on every rank.
    rs_recv[slot, rank, win[0], win[1]] = send[rank, win[0], win[1]]
    total = rs_recv[slot, 0, win[0], win[1]].astype(jnp.float32)
    for j in range(1, tp):
        total = total + rs_recv[slot, j, win[0], win[1]].astype(jnp.float32)
    ag_recv[slot, rank, win[0], win[1]] = total.astype(ag_recv.dtype)
    mine = ag_recv.at[slot, rank, win[0], win[1]]
    _exchange(mine, mine, ws.ag_send_sems, ws.ag_recv_sems, slot, rank, tp)
    blocks = [w.from_wire(ag_recv[slot, j, win[0], win[1]]) for j in range(tp)]
    return jnp.concatenate(blocks, axis=1).astype(jnp.float32)


def all_gather_rows(x, ws, phase):
    """All-gather an `[R, w]` shard (f32 or bf16) into `[R, tp*w]`, rank r at columns r*w."""
    tp = ws.tp
    rank = lax.axis_index(AXIS)
    slot = phase % 2
    x = _value(x)
    rows, width = x.shape
    if x.dtype == jnp.bfloat16:
        buf = ws.g_recv_bf16
    elif x.dtype == jnp.float32:
        buf = ws.g_recv_f32
    else:
        raise TypeError(f"all_gather_rows supports f32 and bf16 payloads, got {x.dtype}")
    w = _wire(rows, width, 16 if x.dtype == jnp.bfloat16 else 8)
    if w.fr > buf.shape[2] or w.fl > buf.shape[3]:
        raise ValueError(f"shard [{rows}, {width}] {x.dtype} exceeds the collectives scratch")
    win = (pl.ds(0, w.fr), pl.ds(0, w.fl))
    buf[slot, rank, win[0], win[1]] = w.to_wire(x)
    mine = buf.at[slot, rank, win[0], win[1]]
    _exchange(mine, mine, ws.g_send_sems, ws.g_recv_sems, slot, rank, tp)
    shards = [w.from_wire(buf[slot, j, win[0], win[1]]) for j in range(tp)]
    return jnp.concatenate(shards, axis=1)


def compiler_params(collective_id, vmem_limit_bytes=64 << 20, **kw):
    """CompilerParams every kernel using these collectives needs."""
    return pltpu.CompilerParams(
        collective_id=collective_id,
        has_side_effects=True,
        vmem_limit_bytes=vmem_limit_bytes,
        **kw,
    )


__all__ = [
    "AXIS",
    "TP",
    "Wire",
    "all_gather_rows",
    "all_reduce_rows",
    "barrier",
    "compiler_params",
    "hier_shapes",
    "scratch_shapes",
    "wire_rows",
    "workspace",
]
