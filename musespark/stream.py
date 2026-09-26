"""Dense-weight bank ring and small vector helpers for the Muse Spark kernels (design.md 5.2).

The ring streams every dense per-rank matrix of every layer through `BANKS` VMEM banks of
`[bank_k, bank_n]` bf16 (`layout.tile_geometry(cfg)`: 1024 x 1024 = 2 MiB for the real
model, `min(1024, H)` square for MINI) in the static order of `layout.tile_schedule(cfg)`:
per layer q, kv, gate, o, pre, router_hi, router_lo, post (N-tiles outer, K-tiles inner
inside each matrix), then the lm_head tiles after the last layer. Tiles narrower than the
bank (kv/router `[1024, 256]`, pre `[1024, 512]`) are packed `pack = bank_n // bn`
consecutive K-tiles per bank *load* (side by side in the bank's lanes, one DMA each) so that
every ring slot carries a full 2 MiB and the in-flight bytes never drop below BANKS x 2 MiB
(38 loads per layer for the real model). Global load `g` lives in bank `g % BANKS`, is
fetched by `fetch(ring, g)` and consumed by `gemv`, which re-issues `g + BANKS` right after
the dot so the ring always keeps BANKS loads (24 MiB) in flight, across layer boundaries and
into the lm_head.

Source windows are 2-D slices `w.at[layer, k0:k0+bk, n0:n0+bn]` of the plain `[L, K, N]`
HBM arrays (strided DMAs run at full speed, hw report section 2). The identity of the tile
`gemv` prefetches is static up to the layer number (`layer`, `layer + 1`, or the lm_head
when the next layer does not exist), so no scalar division is needed on the hot path; the
generic `fetch(ring, g)` with a traced `g` resolves the family with a `pl.when` chain.

`gemv(ring, x [B, K] bf16, family, layer)` returns `f32 [B, N]` (or accumulates into an f32
VMEM ref for wide N, e.g. the logits). Rows are padded/broadcast to 8 for the MXU.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from musespark import layout
from musespark.config import Config

BANKS = layout.BANKS
LM_HEAD = "lm_head"
STATIC_N_BLOCKS = 8  # matrices with more N-tiles than this loop over N-blocks (lm_head)


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
    narrow: int | None = None  # index of the exact-width bank array for bn % 128 != 0
    pack: int = 1  # K-tiles per bank load (lane window j holds tile li*pack + j)

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


def narrow_tiles(cfg: Config, tp: int = 8):
    """Distinct `(bk, bn)` of scheduled tiles narrower than 128 lanes (MINI only), sorted."""
    sched = layout.tile_schedule(cfg, tp)
    return sorted({(t.bk, t.bn) for t in sched.layer + sched.lm_head if t.bn % 128})


def scratch_shapes(cfg: Config, tp: int = 8, banks: int = BANKS):
    """`(VMEM banks [banks, bank_k, bank_n] bf16, DMA semaphores [banks], *narrow banks)`.

    Tiles narrower than 128 lanes cannot be DMA'd into a lane window of the main banks
    (Mosaic requires tile-aligned slices), so each such `(bk, bn)` gets its own exact-width
    `[banks, bk, bn]` bank array (`narrow_tiles`); none exist for the real model.
    """
    geo = layout.tile_geometry(cfg)
    return (
        pltpu.VMEM((banks, geo.bank_k, geo.bank_n), geo.dtype),
        pltpu.SemaphoreType.DMA((banks,)),
    ) + tuple(pltpu.VMEM((banks, bk, bn), geo.dtype) for bk, bn in narrow_tiles(cfg, tp))


def _pack(bk, bn, nk, geo, narrow):
    """K-tiles per bank load: fill the bank's lanes with narrow tiles (never for exact-width
    narrow banks, never across a partial bank row)."""
    if narrow is not None or bk != geo.bank_k or geo.bank_n % bn:
        return 1
    return max(1, min(geo.bank_n // bn, nk))


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
        nk = len({t.k0 for t in group})
        nb = len({t.n0 for t in group})
        pack = _pack(bk, bn, nk, sched.geometry, narrow.get((bk, bn)))
        fam = Family(name, offset, nk, nb, bk, bn, narrow.get((bk, bn)), pack)
        expect = [
            layout.Tile(name, ki * bk, ni * bn, bk, bn) for ni in range(nb) for ki in range(nk)
        ]
        if group != expect:
            raise ValueError(f"{name}: schedule is not the N-outer/K-inner grid stream.py expects")
        families[name] = fam
        i += len(group)
        offset += fam.loads
    return families


def make_ring(cfg: Config, scratch, weights, lm_head=None, tp: int = 8):
    """Bind the ring scratch and the HBM weight refs.

    `scratch`: the refs allocated from `scratch_shapes(cfg)` (same order); `weights`: mapping
    or namespace with an HBM ref `[L, K, N]` for every name in `layout.STREAMED_FAMILIES`;
    `lm_head`: HBM ref `[H, Vp]` streamed after the last layer (None if the kernel does not
    compute logits).
    """
    sched = layout.tile_schedule(cfg, tp)
    geo = sched.geometry
    banks, sems, *narrow_banks = scratch
    narrow = {key: i for i, key in enumerate(narrow_tiles(cfg, tp))}
    if tuple(banks.shape[1:]) != (geo.bank_k, geo.bank_n) or len(narrow_banks) != len(narrow):
        raise ValueError(f"ring scratch does not match scratch_shapes for {geo}")
    n_banks = banks.shape[0]  # the ring depth is whatever scratch_shapes allocated
    refs = {}
    for name in layout.STREAMED_FAMILIES:
        ref = weights[name] if isinstance(weights, dict) else getattr(weights, name)
        refs[name] = ref
    families = _families(sched, narrow)
    for name, fam in families.items():
        L, K, N = refs[name].shape
        if (L, K, N) != (cfg.layers, fam.nk * fam.bk, fam.nb * fam.bn):
            raise ValueError(f"{name}: HBM ref {refs[name].shape} does not match the schedule")
    lm = None
    if lm_head is not None:
        lm_tiles = sched.lm_head
        bk, bn = lm_tiles[0].bk, lm_tiles[0].bn
        nk, nb = len({t.k0 for t in lm_tiles}), len({t.n0 for t in lm_tiles})
        pack = _pack(bk, bn, nk, geo, narrow.get((bk, bn)))
        lm = Family(LM_HEAD, 0, nk, nb, bk, bn, narrow.get((bk, bn)), pack)
        if tuple(lm_head.shape) != (nk * bk, nb * bn):
            raise ValueError(f"lm_head {lm_head.shape} does not match the schedule")
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
    )


# --------------------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------------------
def _bank_view(ring, fam: Family, bank, j=0):
    """The `[bk, bn]` VMEM window of sub-tile `j` of a load of `fam` in slot `bank`."""
    if fam.narrow is None:
        return ring.banks.at[bank, pl.ds(0, fam.bk), pl.ds(j * fam.bn, fam.bn)]
    return ring.narrow_banks[fam.narrow].at[bank]


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
    """Issue the first BANKS loads of the stream (call once before the layer loop)."""
    for g in range(min(ring.banks_count, ring.total)):
        fetch(ring, g)


def drain(ring, g_next):
    """Wait for the loads `g_next .. g_next + BANKS` still in flight (static `g_next`)."""
    for g in range(g_next, min(g_next + ring.banks_count, ring.total)):
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


def gemv(
    ring, x, family, layer, *, out_f32=True, acc=None, g0=None, compute=True, defer_from=None
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
    """
    x = _value(x)
    b = x.shape[0]
    lhs = mxu_rows(x.astype(jnp.bfloat16))
    nb_ = ring.banks_count
    if family == LM_HEAD:
        fam = ring.lm
        if fam is None:
            raise ValueError("ring was built without an lm_head")
        base = ring.layers * ring.per
        ahead = lambda t: _fetch_lm_index(ring, t + nb_)
        ahead_dyn = lambda t: _fetch_lm_dynamic(ring, t + nb_)
    else:
        fam = ring.families[family]
        base = layer * ring.per + fam.offset

        def ahead(t):
            idx = fam.offset + t + nb_
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

    def k_sweep(ni, static):
        block = None
        for li in range(fam.nkl):
            t = ni * fam.nkl + li
            bank = (base + t) % nb_
            _wait_bank(ring, fam, bank, li)
            for j, ki in fam.sub_tiles(li):
                if compute:
                    tile = _bank_view(ring, fam, bank, j)[...]
                    d = jnp.dot(
                        lhs[:, ki * fam.bk : (ki + 1) * fam.bk], tile,
                        preferred_element_type=jnp.float32,
                    )
                else:
                    d = jnp.zeros((lhs.shape[0], fam.bn), jnp.float32)
                block = d if block is None else block + d
            if static:
                ahead(t)
            else:
                ahead_dyn(t)
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
