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


def scratch_shapes(
    rows, width=8192, tp=TP, gather_rows=8, gather_width=None, bf16_wire=False, f32_wire=True
):
    """Scratch for the collectives, in the order `workspace()` consumes it.

    `rows x width` is the largest all-reduce payload (`[64, 4096]` for the expert outputs at
    B=8 also covers `[8, 8192]`); `gather_rows x gather_width` the largest all-gather shard
    (default `[8, width // tp]`). Sizes for (64, 4096): 1 + 2 + 2 + 0.25 + 0.25 MiB with the
    f32 wire, plus / or half of that with `bf16_wire` (the bf16-payload all-reduce buffers;
    `f32_wire=False` drops the f32 ones, then only `wire=jnp.bfloat16` reductions are legal).
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


def workspace(*refs, f32_wire=True):
    """Bind the refs allocated from `scratch_shapes(..., f32_wire=f32_wire)` (same order) into a
    namespace; whether the bf16 wire buffers are present is inferred from the count."""
    names = (_WS_NAMES32 if f32_wire else ()) + _WS_NAMES
    if len(refs) == len(names) + len(_WS_NAMES16):
        names += _WS_NAMES16
    elif len(refs) != len(names):
        raise ValueError(
            f"workspace expects {len(names)} or {len(names) + len(_WS_NAMES16)} refs, "
            f"got {len(refs)}"
        )
    ws = SimpleNamespace(**dict(zip(names, refs)))
    for name in _WS_NAMES32 + _WS_NAMES16:
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


def all_reduce_rows(x, ws, phase, wire=jnp.float32):
    """Sum an f32 `[R, W]` partial over the tp ranks; bit-identical result everywhere.

    Reduce-scatter over the tp column blocks (rank j sums block j of all ranks in slot order
    0..tp-1) followed by an all-gather of the reduced blocks. `x` may be a value or a VMEM
    ref; `phase` is the traced collective counter (see the module docstring). With
    `wire=jnp.bfloat16` (needs `scratch_shapes(..., bf16_wire=True)`) the partials and the
    reduced blocks travel as bf16: half the bytes, but every partial is rounded to bf16
    before the f32 summation and the result is bf16-valued (the caller's `r16` is then a
    no-op).
    """
    tp = ws.tp
    rank = lax.axis_index(AXIS)
    slot = phase % 2
    x = _value(x).astype(jnp.float32)
    rows, width = x.shape
    if width % (tp * 128):
        raise ValueError(f"all_reduce_rows width {width} must be a multiple of {tp * 128}")
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
    "scratch_shapes",
    "wire_rows",
    "workspace",
]
