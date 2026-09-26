"""Dense-weight bank ring and small vector helpers for the Muse Spark kernels (design.md 5.2).

The ring streams every dense per-rank matrix of every layer through `BANKS` VMEM banks of
`[bank_k, bank_n]` (`layout.tile_geometry(cfg, dense_format)`: bf16 1024 x 1024 = 2 MiB for
the real model, or int8 1024 x 2048 = the same 2 MiB with `dense_format="int8"`; `min(1024,
H)`-sided for MINI) in the static order of `layout.tile_schedule(cfg, tp, dense_format)`:
per layer q, kv, gate, o, pre, router_hi, router_lo, post (N-tiles outer, K-tiles inner
inside each matrix), then the lm_head tiles after the last layer. Tiles narrower than the
bank (kv/router `[1024, 256]`, pre `[1024, 512]`) are packed `pack = bank_n // bn`
consecutive K-tiles per bank *load* (side by side in the bank's lanes, one DMA each) so that
every ring slot carries a full 2 MiB and the in-flight bytes never drop below BANKS x 2 MiB
(38 loads per layer for the real model in bf16, 21 in int8). Global load `g` lives in bank
`g % BANKS`, is fetched by `fetch(ring, g)` and consumed by `gemv`, which re-issues
`g + BANKS` right after the dot so the ring always keeps BANKS loads (24 MiB) in flight,
across layer boundaries and into the lm_head.

int8 dense format: the `INT8_DENSE` families and the lm_head are `[.., K, N]` int8 with one
f32 scale per output column (`quant.quantize_int8`); their tiles are fed to the MXU as
`jnp.dot(x_bf16, tile_int8) -> f32` directly (hw report section 3: int8 weights at the bf16
push rate, never `.astype`) and `gemv(..., scale=s [1, N])` multiplies each N-block's f32
column sums by its scales after the K sweep -- the maths of `dequantize_int8` up to f32
summation order. The router hi/lo tiles stay bf16 and live in the int8 bank through its
`.bitcast(bf16)` view (`[512, 2048]`: half the rows); in interpret mode (no ref bitcast) they
get an exact-shape side bank instead (`bitcast=False`), like the narrow MINI tiles.

Source windows are 2-D slices `w.at[layer, k0:k0+bk, n0:n0+bn]` of the plain `[L, K, N]`
HBM arrays (strided DMAs run at full speed, hw report section 2). The identity of the tile
`gemv` prefetches is static up to the layer number (`layer`, `layer + 1`, or the lm_head
when the next layer does not exist), so no scalar division is needed on the hot path; the
generic `fetch(ring, g)` with a traced `g` resolves the family with a `pl.when` chain.

`gemv(ring, x [B, K] bf16, family, layer[, scale])` returns `f32 [B, N]` (or accumulates
into an f32 VMEM ref for wide N, e.g. the logits). Rows are padded/broadcast to 8 for the MXU.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from musespark import layout
from musespark.config import Config

BANKS = layout.BANKS
LM_HEAD = "lm_head"
STATIC_N_BLOCKS = 8  # matrices with more N-tiles than this loop over N-blocks (lm_head)
BF16 = jnp.bfloat16


def _is_ref(x):
    from jax._src.state import types as state_types

    return isinstance(x, state_types.TransformedRef) or isinstance(
        getattr(x, "aval", None), state_types.AbstractRef
    )


def _value(x):
    return x[...] if _is_ref(x) else x


# --------------------------------------------------------------------------------------
# ring construction
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Family:
    """Static tile grid of one streamed matrix: tiles `(ki, ni)` at `[ki*bk, ni*bn]`, streamed
    as loads `t = ni * nkl + li` of the `pack` K-tiles `ki = li*pack + j` (N-outer, K-inner)."""

    name: str
    offset: int  # index of the first load within the layer schedule (0 for lm_head)
    nk: int
    nb: int
    bk: int
    bn: int
    narrow: int | None = None  # index of the exact-shape side bank (bn % 128 != 0, no bitcast)
    pack: int = 1  # K-tiles per bank load (lane window j holds tile li*pack + j)
    dtype: object = BF16  # tile element type (int8 for the int8 dense families)

    @property
    def nkl(self):
        """K-loads per N-block."""
        return -(-self.nk // self.pack)

    @property
    def loads(self):
        return self.nkl * self.nb

    def sub_tiles(self, li):
        """`(j, ki)` of the K-tiles in load `li` (static `li`; the last load may be partial)."""
        return [(j, li * self.pack + j) for j in range(self.pack) if li * self.pack + j < self.nk]

    def window(self, ki, ni):
        """`(k slice, n slice)` of tile `(ki, ni)`; `ki`/`ni` static ints or traced int32."""

        def span(i, n, size):
            if n == 1 or isinstance(i, int):
                return pl.ds(0 if n == 1 else i * size, size)
            return pl.ds(pl.multiple_of(i * size, size), size)

        return span(ki, self.nk, self.bk), span(ni, self.nb, self.bn)


def _side_key(tile, dtype):
    return (tile.bk, tile.bn, np.dtype(dtype).name)


def narrow_tiles(cfg: Config, tp: int = 8, dense_format="bf16", bitcast=True):
    """Distinct `(bk, bn, dtype name)` of scheduled tiles that need an exact-shape side bank:
    narrower than 128 lanes (MINI only) or, without `bitcast`, of a dtype other than the
    bank's (bf16 router tiles in an int8 ring under the interpreter). Sorted."""
    sched = layout.tile_schedule(cfg, tp, dense_format)
    keys = set()
    for t in sched.layer + sched.lm_head:
        dtype = sched.dtype(t.family)
        if t.bn % 128 or (not bitcast and np.dtype(dtype) != np.dtype(sched.geometry.dtype)):
            keys.add(_side_key(t, dtype))
    return sorted(keys)


def scratch_shapes(cfg: Config, tp: int = 8, banks: int = BANKS, dense_format="bf16",
                   bitcast=True):
    """`(VMEM banks [banks, bank_k, bank_n] of the bank dtype, DMA semaphores [banks], *side
    banks)`.

    Tiles narrower than 128 lanes cannot be DMA'd into a lane window of the main banks
    (Mosaic requires tile-aligned slices), so each such `(bk, bn, dtype)` gets its own
    exact-shape `[banks, bk, bn]` bank array (`narrow_tiles`); none exist for the real model.
    `bitcast=False` (interpret mode) also gives the bf16 tiles of an int8 ring a side bank.
    """
    geo = layout.tile_geometry(cfg, dense_format)
    return (
        pltpu.VMEM((banks, geo.bank_k, geo.bank_n), geo.dtype),
        pltpu.SemaphoreType.DMA((banks,)),
    ) + tuple(
        pltpu.VMEM((banks, bk, bn), np.dtype(name))
        for bk, bn, name in narrow_tiles(cfg, tp, dense_format, bitcast)
    )


def _pack(bk, bn, nk, geo, narrow, dtype):
    """K-tiles per bank load: fill the bank's lanes with narrow tiles (never for exact-shape
    side banks, never across a partial bank row)."""
    bank_k, bank_n = geo.tile_max(dtype)
    if narrow is not None or bk != bank_k or bank_n % bn:
        return 1
    return max(1, min(bank_n // bn, nk))


def _families(sched: layout.Schedule, narrow):
    """Group the static per-layer schedule by family and check the regular grid order."""
    families = {}
    tiles = list(sched.layer)
    i = 0
    offset = 0
    while i < len(tiles):
        name = tiles[i].family
        group = [t for t in tiles[i:] if t.family == name]
        group = group[: next((j for j, t in enumerate(group) if t.family != name), len(group))]
        bk, bn = group[0].bk, group[0].bn
        dtype = sched.dtype(name)
        nk = len({t.k0 for t in group})
        nb = len({t.n0 for t in group})
        side = narrow.get(_side_key(group[0], dtype))
        pack = _pack(bk, bn, nk, sched.geometry, side, dtype)
        fam = Family(name, offset, nk, nb, bk, bn, side, pack, dtype)
        expect = [
            layout.Tile(name, ki * bk, ni * bn, bk, bn) for ni in range(nb) for ki in range(nk)
        ]
        if group != expect:
            raise ValueError(f"{name}: schedule is not the N-outer/K-inner grid stream.py expects")
        families[name] = fam
        i += len(group)
        offset += fam.loads
    return families


def make_ring(cfg: Config, scratch, weights, lm_head=None, tp: int = 8, dense_format="bf16",
              bitcast=True, lag=0, blockdiag=True):
    """Bind the ring scratch and the HBM weight refs.

    `scratch`: the refs allocated from `scratch_shapes(cfg, tp, banks, dense_format,
    bitcast)` (same order); `weights`: mapping or namespace with an HBM ref `[L, K, N]` for
    every name in `layout.STREAMED_FAMILIES` under its array name of `dense_format`
    (`layout.dense_ref_name`: `q_i8` for `q` in int8); `lm_head`: HBM ref `[H, Vp]` (bf16, or
    the int8 `lm_head_i8`) streamed after the last layer (None if the kernel does not compute
    logits). `lag=1` issues the refill of a slot one load LATER but BEFORE the dot: consuming
    load `g` first refills slot `g - 1` (already consumed), then computes; the DMA start no
    longer waits for the matmul that reads slot `g`, at the price of one fewer load in
    flight (`prime` issues `BANKS - lag` loads). `blockdiag` computes every packed load with
    one block-diagonal dot (`_packed_dot`) instead of one dot per K-tile.
    """
    if lag not in (0, 1):
        raise ValueError("lag must be 0 or 1")
    sched = layout.tile_schedule(cfg, tp, dense_format)
    geo = sched.geometry
    banks, sems, *narrow_banks = scratch
    narrow = {key: i for i, key in enumerate(narrow_tiles(cfg, tp, dense_format, bitcast))}
    if tuple(banks.shape[1:]) != (geo.bank_k, geo.bank_n) or len(narrow_banks) != len(narrow):
        raise ValueError(f"ring scratch does not match scratch_shapes for {geo}")
    if np.dtype(banks.dtype) != np.dtype(geo.dtype):
        raise ValueError(f"ring banks are {banks.dtype}, the {dense_format} geometry needs {geo}")
    n_banks = banks.shape[0]  # the ring depth is whatever scratch_shapes allocated
    refs = {}
    for name in layout.STREAMED_FAMILIES:
        array = layout.dense_ref_name(name, dense_format)
        refs[name] = weights[array] if isinstance(weights, dict) else getattr(weights, array)
    families = _families(sched, narrow)
    for name, fam in families.items():
        L, K, N = refs[name].shape
        if (L, K, N) != (cfg.layers, fam.nk * fam.bk, fam.nb * fam.bn):
            raise ValueError(f"{name}: HBM ref {refs[name].shape} does not match the schedule")
        if np.dtype(refs[name].dtype) != np.dtype(fam.dtype):
            raise ValueError(f"{name}: HBM ref is {refs[name].dtype}, the schedule streams {fam.dtype}")
    lm = None
    if lm_head is not None:
        lm_tiles = sched.lm_head
        bk, bn = lm_tiles[0].bk, lm_tiles[0].bn
        dtype = sched.dtype(LM_HEAD)
        nk, nb = len({t.k0 for t in lm_tiles}), len({t.n0 for t in lm_tiles})
        side = narrow.get(_side_key(lm_tiles[0], dtype))
        pack = _pack(bk, bn, nk, geo, side, dtype)
        lm = Family(LM_HEAD, 0, nk, nb, bk, bn, side, pack, dtype)
        if tuple(lm_head.shape) != (nk * bk, nb * bn) or np.dtype(lm_head.dtype) != np.dtype(dtype):
            raise ValueError(f"lm_head {lm_head.shape} {lm_head.dtype} does not match the schedule")
    per = sum(f.loads for f in families.values())
    total = cfg.layers * per + (lm.loads if lm is not None else 0)
    return SimpleNamespace(
        banks=banks,
        narrow_banks=tuple(narrow_banks),
        sems=sems,
        refs=refs,
        lm_head=lm_head,
        families=families,
        lm=lm,
        layers=cfg.layers,
        per=per,
        total=total,
        banks_count=n_banks,
        deferred=[],
        dtype=np.dtype(geo.dtype),
        dense_format=dense_format,
        lag=lag,
        blockdiag=blockdiag,
    )


# --------------------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------------------
def _bank_view(ring, fam: Family, bank, j=0):
    """The `[bk, bn]` VMEM window of sub-tile `j` of a load of `fam` in slot `bank` (through
    the bank's `.bitcast(fam.dtype)` view when the tile dtype differs from the bank's)."""
    if fam.narrow is not None:
        return ring.narrow_banks[fam.narrow].at[bank]
    if np.dtype(fam.dtype) == ring.dtype:
        return ring.banks.at[bank, pl.ds(0, fam.bk), pl.ds(j * fam.bn, fam.bn)]
    view = ring.banks.at[bank].bitcast(fam.dtype)
    return view.at[pl.ds(0, fam.bk), pl.ds(j * fam.bn, fam.bn)]


def _bank_window(ring, fam: Family, bank, n):
    """The `[bk, n * bn]` VMEM window holding sub-tiles `0 .. n-1` of a load in slot `bank`."""
    if fam.narrow is not None:
        raise ValueError("side banks hold one tile per load")
    base = ring.banks.at[bank]
    if np.dtype(fam.dtype) != ring.dtype:
        base = base.bitcast(fam.dtype)
    return base.at[pl.ds(0, fam.bk), pl.ds(0, n * fam.bn)]


def _stacked_lhs(lhs, fam: Family):
    """`X [pack*rows, nkl*bk]` with `X[j*rows:(j+1)*rows, li*bk:(li+1)*bk] = lhs[:, K-tile
    li*pack + j]` (zeros past the last K-tile): lane window `li` is the block-diagonal LHS of
    load `li` (`_packed_dot`). Built once per gemv."""
    rows, p, bk = lhs.shape[0], fam.pack, fam.bk
    zeros = jnp.zeros((rows, bk), lhs.dtype)
    blocks = []
    for j in range(p):
        pieces = []
        for li in range(fam.nkl):
            ki = li * p + j
            pieces.append(lhs[:, ki * bk : (ki + 1) * bk] if ki < fam.nk else zeros)
        blocks.append(jnp.concatenate(pieces, axis=1) if len(pieces) > 1 else pieces[0])
    return jnp.concatenate(blocks, axis=0)


def _packed_dot(ring, fam: Family, bank, x_bd, li, rows, n):
    """One MXU op for the `n` K-tiles of packed load `li` (hw report section 3, the
    block-diagonal LHS): the K-slices of x stacked as row blocks `X [pack*rows, bk]`
    (`_stacked_lhs` lane window `li`) times the whole bank window `[bk, n*bn]`; the wanted
    products are the diagonal `[rows, bn]` blocks of the result, the off-diagonal ones are
    free (the MXU is weight-push bound). One wide dot replaces `n` narrow ones, whose issue
    latency and int8 conversion dominated the stream at 2 MiB per load."""
    x = x_bd[:, li * fam.bk : (li + 1) * fam.bk]
    w = _bank_window(ring, fam, bank, n)[...]
    r = jnp.dot(x, w, preferred_element_type=jnp.float32)  # [pack*rows, n*bn]
    d = None
    for j in range(n):
        dj = r[j * rows : (j + 1) * rows, j * fam.bn : (j + 1) * fam.bn]
        d = dj if d is None else d + dj
    return d


def _bank_copy(ring, src, fam: Family, bank, j=0):
    return pltpu.make_async_copy(src, _bank_view(ring, fam, bank, j), ring.sems.at[bank])


def _wait_bank(ring, fam: Family, bank, li=None):
    """Wait for the load in `bank`: one wait per sub-tile (semaphores count bytes, so any
    same-size descriptor works). `li` (static) tells how many sub-tiles a partial load has."""
    n = fam.pack if li is None else len(fam.sub_tiles(li))
    for j in range(n):
        _bank_copy(ring, _bank_view(ring, fam, bank, j), fam, bank, j).wait()


def _layer_tile_copy(ring, layer, ki, ni, fam: Family, bank, j=0):
    k0, n0 = fam.window(ki, ni)
    return _bank_copy(ring, ring.refs[fam.name].at[layer, k0, n0], fam, bank, j)


def _start_layer_load(ring, layer, li, ni, fam: Family, bank):
    """Start the DMAs of load `(li, ni)` of `fam`; `li`/`ni` static ints or traced int32."""
    if isinstance(li, int):
        for j, ki in fam.sub_tiles(li):
            _layer_tile_copy(ring, layer, ki, ni, fam, bank, j).start()
        return
    for j in range(fam.pack):
        ki = li * fam.pack + j
        if fam.nk % fam.pack == 0:
            _layer_tile_copy(ring, layer, ki, ni, fam, bank, j).start()
        else:

            @pl.when(ki < fam.nk)
            def _(ki=ki, j=j):
                _layer_tile_copy(ring, layer, ki, ni, fam, bank, j).start()


def _lm_tile_copy(ring, ki, ni, bank, j=0):
    k0, n0 = ring.lm.window(ki, ni)
    return _bank_copy(ring, ring.lm_head.at[k0, n0], ring.lm, bank, j)


def _start_lm_load(ring, li, ni, bank):
    lm = ring.lm
    if isinstance(li, int):
        for j, ki in lm.sub_tiles(li):
            _lm_tile_copy(ring, ki, ni, bank, j).start()
        return
    for j in range(lm.pack):
        ki = li * lm.pack + j
        if lm.nk % lm.pack == 0:
            _lm_tile_copy(ring, ki, ni, bank, j).start()
        else:

            @pl.when(ki < lm.nk)
            def _(ki=ki, j=j):
                _lm_tile_copy(ring, ki, ni, bank, j).start()


def _fetch_layer_index(ring, layer, j):
    """Start the copies of static load `j` of the (traced) `layer`; no-op past the last layer."""
    per, nb = ring.per, ring.banks_count
    fam = next(f for f in ring.families.values() if f.offset <= j < f.offset + f.loads)
    t = j - fam.offset

    @pl.when(layer < ring.layers)
    def _():
        _start_layer_load(ring, layer, t % fam.nkl, t // fam.nkl, fam, (layer * per + j) % nb)


def _fetch_lm_index(ring, j):
    """Start the copies of static lm_head load `j` (no-op if lm_head is absent or j is past it)."""
    if ring.lm is None or j >= ring.lm.loads:
        return
    bank = (ring.layers * ring.per + j) % ring.banks_count
    _start_lm_load(ring, j % ring.lm.nkl, j // ring.lm.nkl, bank)


def _fetch_lm_dynamic(ring, t):
    """Start the copies of traced lm_head load `t` (guarded against the end of the stream)."""
    if ring.lm is None:
        return
    lm = ring.lm

    @pl.when(t < lm.loads)
    def _():
        bank = (ring.layers * ring.per + t) % ring.banks_count
        _start_lm_load(ring, t % lm.nkl, t // lm.nkl, bank)


def _fetch_ahead(ring, layer, idx):
    """Start the load `idx` (static, may exceed the layer) positions into the traced `layer`."""
    per = ring.per
    ahead, j = divmod(idx, per)
    _fetch_layer_index(ring, layer + ahead, j)
    for m in range(ahead):  # layer + ahead == layers + m: lm_head load m*per + j
        if ring.lm is not None and m * per + j < ring.lm.loads:

            @pl.when(layer + ahead == ring.layers + m)
            def _(m=m):
                _fetch_lm_index(ring, m * per + j)


def fetch(ring, g):
    """Start the DMAs of global load `g` (static int or traced int32) into bank `g % BANKS`."""
    per, nb = ring.per, ring.banks_count
    if isinstance(g, int):
        if g >= ring.total:
            return
        if g < ring.layers * per:
            _fetch_layer_index(ring, g // per, g % per)
        else:
            _fetch_lm_index(ring, g - ring.layers * per)
        return

    @pl.when(g < ring.layers * per)
    def _():
        layer, i, bank = g // per, g % per, g % nb
        for fam in ring.families.values():

            @pl.when((i >= fam.offset) & (i < fam.offset + fam.loads))
            def _(fam=fam):
                t = i - fam.offset
                _start_layer_load(ring, layer, t % fam.nkl, t // fam.nkl, fam, bank)

    if ring.lm is not None:

        @pl.when(g >= ring.layers * per)
        def _():
            _fetch_lm_dynamic(ring, g - ring.layers * per)


def post_offset(ring, family="post"):
    """Static index of the first load of `family` in the layer schedule (for `defer_from`)."""
    return ring.families[family].offset


def flush_deferred(ring):
    """Issue the refills held back by `gemv(..., defer_from=...)`, in their original order."""
    for layer, idx in ring.deferred:
        _fetch_ahead(ring, layer, idx)
    ring.deferred.clear()


def prime(ring):
    """Issue the first BANKS (- lag) loads of the stream (call once before the layer loop)."""
    for g in range(min(ring.banks_count - ring.lag, ring.total)):
        fetch(ring, g)


def drain(ring, g_next):
    """Wait for the loads `g_next .. g_next + BANKS - lag` still in flight (static `g_next`)."""
    for g in range(g_next, min(g_next + ring.banks_count - ring.lag, ring.total)):
        if g < ring.layers * ring.per:
            j = g % ring.per
            fam = next(f for f in ring.families.values() if f.offset <= j < f.offset + f.loads)
            t = j - fam.offset
        else:
            fam, t = ring.lm, g - ring.layers * ring.per
        _wait_bank(ring, fam, g % ring.banks_count, t % fam.nkl)


# --------------------------------------------------------------------------------------
# gemv
# --------------------------------------------------------------------------------------
def mxu_rows(x, rows=8):
    """Pad (or broadcast, for one row) `x [B, K]` to `rows` rows for the MXU."""
    b = x.shape[0]
    if b == 1:
        return jnp.broadcast_to(x, (rows, x.shape[1]))
    if b % rows:
        return jnp.pad(x, ((0, rows - b % rows), (0, 0)))
    return x


def _scale_block(scale, ni, bn, static):
    """`[1, bn]` f32 column scales of N-block `ni` from `scale` (`[1, N]` value or VMEM ref)."""
    if static:
        return _value(scale)[:, ni * bn : (ni + 1) * bn].astype(jnp.float32)
    if not _is_ref(scale):
        raise ValueError("looped N-blocks need the scales as a VMEM ref")
    return scale[:, pl.ds(pl.multiple_of(ni * bn, bn), bn)].astype(jnp.float32)


def gemv(
    ring, x, family, layer, *, out_f32=True, acc=None, g0=None, compute=True, defer_from=None,
    scale=None,
):
    """`x [B, K] bf16 @ W[family][layer]` streamed through the ring -> f32 `[B, N]`.

    With `acc` (an f32 VMEM ref `[rows >= B, N]`) the result is stored there and None is
    returned; this is required for matrices with more than `STATIC_N_BLOCKS` N-tiles (the
    lm_head). `family == "lm_head"` streams the vocab tiles (`layer` is ignored). `g0`
    is accepted for API symmetry and must equal the family's first global tile if given.
    `compute=False` (profiling) keeps the DMA waits / refills but skips the dots (zeros).
    `defer_from` (a static tile index of the layer schedule, e.g. `post_offset(ring)`) holds
    back the refills whose target tile index is `>= defer_from` (including the next layer's
    tiles) until `flush_deferred(ring)`: used to let the expert slab DMAs of the layer enter
    the (FIFO) DMA queue ahead of the post / next-layer prefetch.
    `scale` (`[1, N]` f32 value or VMEM ref; a ref for the lm_head) multiplies the f32 column
    sums of each N-block after its K sweep: the int8 families' per-output-channel scales
    (`quant.dequantize_int8` maths up to summation order).
    """
    x = _value(x)
    b = x.shape[0]
    lhs = mxu_rows(x.astype(jnp.bfloat16))
    nb_ = ring.banks_count
    lag = ring.lag  # refill target of the load consumed now: `t + nb_ - lag`
    if family == LM_HEAD:
        fam = ring.lm
        if fam is None:
            raise ValueError("ring was built without an lm_head")
        base = ring.layers * ring.per
        ahead = lambda t: _fetch_lm_index(ring, t + nb_ - lag)
        ahead_dyn = lambda t: _fetch_lm_dynamic(ring, t + nb_ - lag)
    else:
        fam = ring.families[family]
        base = layer * ring.per + fam.offset

        def ahead(t):
            idx = fam.offset + t + nb_ - lag
            if defer_from is not None and idx >= defer_from:
                ring.deferred.append((layer, idx))
            else:
                _fetch_ahead(ring, layer, idx)

        ahead_dyn = None
    if g0 is not None and isinstance(g0, int) and isinstance(base, int) and g0 != base:
        raise ValueError(f"g0={g0} does not match the schedule position {base}")
    if x.shape[1] != fam.nk * fam.bk:
        raise ValueError(
            f"{family}: x has {x.shape[1]} columns, the matrix has K={fam.nk * fam.bk}"
        )

    packed = ring.blockdiag and fam.pack > 1 and compute
    x_bd = _stacked_lhs(lhs, fam) if packed else None

    def k_sweep(ni, static):
        block = None
        for li in range(fam.nkl):
            t = ni * fam.nkl + li
            bank = (base + t) % nb_
            _wait_bank(ring, fam, bank, li)
            if lag:  # refill the slot of the previous load before this load's dots
                if static:
                    ahead(t)
                else:
                    ahead_dyn(t)
            subs = fam.sub_tiles(li)
            if not compute:
                d = jnp.zeros((lhs.shape[0], fam.bn), jnp.float32)
            elif packed:
                d = _packed_dot(ring, fam, bank, x_bd, li, lhs.shape[0], len(subs))
            else:
                d = None
                for j, ki in subs:
                    tile = _bank_view(ring, fam, bank, j)[...]
                    dj = jnp.dot(
                        lhs[:, ki * fam.bk : (ki + 1) * fam.bk], tile,
                        preferred_element_type=jnp.float32,
                    )
                    d = dj if d is None else d + dj
            block = d if block is None else block + d
            if not lag:
                if static:
                    ahead(t)
                else:
                    ahead_dyn(t)
        if scale is not None:
            block = block * _scale_block(scale, ni, fam.bn, static)
        return block

    if fam.nb <= STATIC_N_BLOCKS:
        blocks = [k_sweep(ni, True) for ni in range(fam.nb)]
        if acc is None:
            out = jnp.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]
            out = out[:b]
            return out if out_f32 else out.astype(jnp.bfloat16)
        rows = acc.shape[0]
        for ni, block in enumerate(blocks):
            acc[:, pl.ds(ni * fam.bn, fam.bn)] = block[:rows]
        return None
    if acc is None:
        raise ValueError(f"{family}: {fam.nb} N-tiles; pass an f32 VMEM `acc` ref")
    if ahead_dyn is None:
        raise ValueError("looped N-blocks are only supported for the lm_head")
    rows = acc.shape[0]

    def body(ni, carry):
        block = k_sweep(ni, False)
        acc[:, pl.ds(pl.multiple_of(ni * fam.bn, fam.bn), fam.bn)] = block[:rows]
        return carry

    lax.fori_loop(0, fam.nb, body, 0)
    return None


# --------------------------------------------------------------------------------------
# small-vector helpers shared by the attention and MoE kernels
# --------------------------------------------------------------------------------------
def r16(x):
    """Round an f32 value to bf16 and back (the spec's `r16`)."""
    return x.astype(jnp.bfloat16).astype(jnp.float32)


def rms(x, w=None, eps=1e-5):
    """RMS-normalise the rows of `x [B, W]` in f32 and scale by `w [1, W]` (None = 1)."""
    x = _value(x).astype(jnp.float32)
    y = x * lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    if w is not None:
        y = y * _value(w).astype(jnp.float32)
    return y


def norm_to_bf16(x, w=None, eps=1e-5):
    """`rms` followed by the bf16 rounding every norm output gets."""
    return rms(x, w, eps).astype(jnp.bfloat16)


def gated_residual(s, nb, alpha, beta):
    """Residual gate: `alpha * s + beta * nb` in f32 (`alpha`, `beta` are `[1, W]`)."""
    return _value(alpha).astype(jnp.float32) * _value(s).astype(jnp.float32) + _value(beta).astype(
        jnp.float32
    ) * _value(nb).astype(jnp.float32)


def silu(x):
    return x * jax.nn.sigmoid(x)


def vector_scratch(width, dtype=jnp.bfloat16):
    """Double-buffered per-layer vector slots `[2, 1, width]` and their DMA semaphores."""
    return pltpu.VMEM((2, 1, width), dtype), pltpu.SemaphoreType.DMA((2,))


def _vector_copy(hbm, buf, sems, l):
    return pltpu.make_async_copy(hbm.at[l], buf.at[l % 2], sems.at[l % 2])


def prefetch_vector(hbm, buf, sems, l):
    """Start the copy of layer `l` of `hbm [L, 1, W]` into slot `l % 2` of `buf [2, 1, W]`."""
    _vector_copy(hbm, buf, sems, l).start()


def wait_vector(hbm, buf, sems, l):
    """Wait for `prefetch_vector(..., l)`."""
    _vector_copy(hbm, buf, sems, l).wait()


def vector(buf, l):
    """The `[1, W]` slot of layer `l` (after `wait_vector`)."""
    return buf[l % 2]


__all__ = [
    "BANKS",
    "LM_HEAD",
    "Family",
    "drain",
    "fetch",
    "flush_deferred",
    "gated_residual",
    "gemv",
    "make_ring",
    "mxu_rows",
    "norm_to_bf16",
    "post_offset",
    "prefetch_vector",
    "prime",
    "r16",
    "rms",
    "scratch_shapes",
    "silu",
    "vector",
    "vector_scratch",
    "wait_vector",
]
