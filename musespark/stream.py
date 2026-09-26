"""Dense-weight bank ring and small vector helpers for the Muse Spark kernels (design.md 5.2).

The ring streams every dense per-rank matrix of every layer through `BANKS` VMEM banks of
`[bank_k, bank_n]` bf16 (`layout.tile_geometry(cfg)`: 1024 x 1024 = 2 MiB for the real
model, `min(1024, H)` square for MINI) in the static order of `layout.tile_schedule(cfg)`:
per layer q, kv, gate, o, pre, router_hi, router_lo, post (N-tiles outer, K-tiles inner
inside each matrix), then the lm_head tiles after the last layer. Global tile `g` lives in
bank `g % BANKS`, is fetched by `fetch(ring, g)` and consumed by `gemv`, which re-issues
`g + BANKS` right after the dot so the ring always keeps BANKS tiles (24 MiB) in flight,
across layer boundaries and into the lm_head.

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
    """Static tile grid of one streamed matrix: tiles `t = ni * nk + ki` at `[ki*bk, ni*bn]`."""

    name: str
    offset: int  # index of the first tile within the layer schedule (0 for lm_head)
    nk: int
    nb: int
    bk: int
    bn: int
    narrow: int | None = None  # index of the exact-width bank array for bn % 128 != 0

    @property
    def tiles(self):
        return self.nk * self.nb

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


def scratch_shapes(cfg: Config, tp: int = 8):
    """`(VMEM banks [BANKS, bank_k, bank_n] bf16, DMA semaphores [BANKS], *narrow banks)`.

    Tiles narrower than 128 lanes cannot be DMA'd into a lane window of the main banks
    (Mosaic requires tile-aligned slices), so each such `(bk, bn)` gets its own exact-width
    `[BANKS, bk, bn]` bank array (`narrow_tiles`); none exist for the real model.
    """
    geo = layout.tile_geometry(cfg)
    return (
        pltpu.VMEM((geo.banks, geo.bank_k, geo.bank_n), geo.dtype),
        pltpu.SemaphoreType.DMA((geo.banks,)),
    ) + tuple(pltpu.VMEM((geo.banks, bk, bn), geo.dtype) for bk, bn in narrow_tiles(cfg, tp))


def _families(sched: layout.Schedule, narrow):
    """Group the static per-layer schedule by family and check the regular grid order."""
    families = {}
    tiles = list(sched.layer)
    i = 0
    while i < len(tiles):
        name = tiles[i].family
        group = [t for t in tiles[i:] if t.family == name]
        group = group[: next((j for j, t in enumerate(group) if t.family != name), len(group))]
        bk, bn = group[0].bk, group[0].bn
        nk = len({t.k0 for t in group})
        nb = len({t.n0 for t in group})
        fam = Family(name, i, nk, nb, bk, bn, narrow.get((bk, bn)))
        expect = [
            layout.Tile(name, ki * bk, ni * bn, bk, bn) for ni in range(nb) for ki in range(nk)
        ]
        if group != expect:
            raise ValueError(f"{name}: schedule is not the N-outer/K-inner grid stream.py expects")
        families[name] = fam
        i += len(group)
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
    if tuple(banks.shape) != (geo.banks, geo.bank_k, geo.bank_n) or len(narrow_banks) != len(
        narrow
    ):
        raise ValueError(f"ring scratch does not match scratch_shapes for {geo}")
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
        lm = Family(LM_HEAD, 0, nk, nb, bk, bn, narrow.get((bk, bn)))
        if tuple(lm_head.shape) != (nk * bk, nb * bn):
            raise ValueError(f"lm_head {lm_head.shape} does not match the schedule")
    per = sched.tiles_per_layer
    total = cfg.layers * per + (lm.tiles if lm is not None else 0)
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
        banks_count=geo.banks,
    )


# --------------------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------------------
def _bank_view(ring, fam: Family, bank):
    """The `[bk, bn]` VMEM window that holds a tile of `fam` in slot `bank`."""
    if fam.narrow is None:
        return ring.banks.at[bank, pl.ds(0, fam.bk), pl.ds(0, fam.bn)]
    return ring.narrow_banks[fam.narrow].at[bank]


def _bank_copy(ring, src, fam: Family, bank):
    return pltpu.make_async_copy(src, _bank_view(ring, fam, bank), ring.sems.at[bank])


def _wait_bank(ring, fam: Family, bank):
    """Wait for the tile in `bank` (a same-size descriptor is enough: semaphores count bytes)."""
    _bank_copy(ring, _bank_view(ring, fam, bank), fam, bank).wait()


def _layer_tile_copy(ring, layer, ki, ni, fam: Family, bank):
    k0, n0 = fam.window(ki, ni)
    return _bank_copy(ring, ring.refs[fam.name].at[layer, k0, n0], fam, bank)


def _lm_tile_copy(ring, ki, ni, bank):
    k0, n0 = ring.lm.window(ki, ni)
    return _bank_copy(ring, ring.lm_head.at[k0, n0], ring.lm, bank)


def _fetch_layer_index(ring, layer, j):
    """Start the copy of static tile `j` of the (traced) `layer`; no-op past the last layer."""
    per, nb = ring.per, ring.banks_count
    fam = next(f for f in ring.families.values() if f.offset <= j < f.offset + f.tiles)
    t = j - fam.offset

    @pl.when(layer < ring.layers)
    def _():
        _layer_tile_copy(ring, layer, t % fam.nk, t // fam.nk, fam, (layer * per + j) % nb).start()


def _fetch_lm_index(ring, j):
    """Start the copy of static lm_head tile `j` (no-op if lm_head is absent or j is past it)."""
    if ring.lm is None or j >= ring.lm.tiles:
        return
    bank = (ring.layers * ring.per + j) % ring.banks_count
    _lm_tile_copy(ring, j % ring.lm.nk, j // ring.lm.nk, bank).start()


def _fetch_lm_dynamic(ring, t):
    """Start the copy of traced lm_head tile `t` (guarded against the end of the stream)."""
    if ring.lm is None:
        return
    lm = ring.lm

    @pl.when(t < lm.tiles)
    def _():
        bank = (ring.layers * ring.per + t) % ring.banks_count
        _lm_tile_copy(ring, t % lm.nk, t // lm.nk, bank).start()


def _fetch_ahead(ring, layer, idx):
    """Start the tile `idx` (static, may exceed the layer) positions into the traced `layer`."""
    per = ring.per
    ahead, j = divmod(idx, per)
    _fetch_layer_index(ring, layer + ahead, j)
    for m in range(ahead):  # layer + ahead == layers + m: lm_head tile m*per + j
        if ring.lm is not None and m * per + j < ring.lm.tiles:

            @pl.when(layer + ahead == ring.layers + m)
            def _(m=m):
                _fetch_lm_index(ring, m * per + j)


def fetch(ring, g):
    """Start the DMA of global tile `g` (static int or traced int32) into bank `g % BANKS`."""
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

            @pl.when((i >= fam.offset) & (i < fam.offset + fam.tiles))
            def _(fam=fam):
                t = i - fam.offset
                _layer_tile_copy(ring, layer, t % fam.nk, t // fam.nk, fam, bank).start()

    if ring.lm is not None:

        @pl.when(g >= ring.layers * per)
        def _():
            _fetch_lm_dynamic(ring, g - ring.layers * per)


def prime(ring):
    """Issue the first BANKS tiles of the stream (call once before the layer loop)."""
    for g in range(min(ring.banks_count, ring.total)):
        fetch(ring, g)


def drain(ring, g_next):
    """Wait for the tiles `g_next .. g_next + BANKS` still in flight (static `g_next`)."""
    for g in range(g_next, min(g_next + ring.banks_count, ring.total)):
        if g < ring.layers * ring.per:
            j = g % ring.per
            fam = next(f for f in ring.families.values() if f.offset <= j < f.offset + f.tiles)
            _wait_bank(ring, fam, g % ring.banks_count)
        else:
            _wait_bank(ring, ring.lm, g % ring.banks_count)


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


def gemv(ring, x, family, layer, *, out_f32=True, acc=None, g0=None):
    """`x [B, K] bf16 @ W[family][layer]` streamed through the ring -> f32 `[B, N]`.

    With `acc` (an f32 VMEM ref `[rows >= B, N]`) the result is stored there and None is
    returned; this is required for matrices with more than `STATIC_N_BLOCKS` N-tiles (the
    lm_head). `family == "lm_head"` streams the vocab tiles (`layer` is ignored). `g0`
    is accepted for API symmetry and must equal the family's first global tile if given.
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
        ahead = lambda t: _fetch_ahead(ring, layer, fam.offset + t + nb_)
        ahead_dyn = None
    if g0 is not None and isinstance(g0, int) and isinstance(base, int) and g0 != base:
        raise ValueError(f"g0={g0} does not match the schedule position {base}")
    if x.shape[1] != fam.nk * fam.bk:
        raise ValueError(
            f"{family}: x has {x.shape[1]} columns, the matrix has K={fam.nk * fam.bk}"
        )

    def k_sweep(ni, static):
        block = None
        for ki in range(fam.nk):
            t = ni * fam.nk + ki
            bank = (base + t) % nb_
            _wait_bank(ring, fam, bank)
            tile = _bank_view(ring, fam, bank)[...]
            d = jnp.dot(
                lhs[:, ki * fam.bk : (ki + 1) * fam.bk], tile, preferred_element_type=jnp.float32
            )
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
    "gated_residual",
    "gemv",
    "make_ring",
    "mxu_rows",
    "norm_to_bf16",
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
