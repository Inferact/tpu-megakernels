"""Per-rank kernel weight layout for the Muse Spark decode megakernel (design.md section 2).

Single source of truth for the shapes/dtypes of every per-rank array, the KV-cache layout
and the dense-weight tile schedule. Every number is derived from `Config`, so the MINI
config and the real model share the code. All matrices are `[in, out]` (`y = x @ W`).

Per-rank arrays (rank `r` of `tp`; the global array has a leading `tp` axis, sharded
`P("tp")`). With `L` layers, `H` hidden, `Hm` moe_hidden, `I` expert_hidden, `E` experts,
`D` head_dim, `qh = heads / tp`, `kvh = kv_heads / tp`, `Is = I / tp`, `G` group_size and
`Vp = vocab_pad(cfg, tp)`:

    embed            [Vp, H]        bf16  rows r*Vp:(r+1)*Vp of embed_tokens, zero-padded
    lm_head          [H, Vp]        bf16  columns r*Vp:(r+1)*Vp of lm_head.T, zero-padded
    final_norm       [1, H]         bf16  model.norm.weight as-is (gain center 0: NO +1)
    attn_norm        [L, 1, H]      bf16  r16(1 + input_layernorm)
    ffn_norm         [L, 1, H]      bf16  r16(1 + pre_feedforward_layernorm)
    post_ffn_norm    [L, 1, H]      bf16  r16(1 + post_feedforward_layernorm), applied BEFORE norm
    attn_gate_alpha  [L, 1, H]      f32   gate_coeffs(post_attention_residual_gate)
    attn_gate_beta   [L, 1, H]      f32
    ffn_gate_alpha   [L, 1, H]      f32   gate_coeffs(post_feedforward_residual_gate)
    ffn_gate_beta    [L, 1, H]      f32
    pre_expert_norm  [L, 1, Hm]     bf16  r16(1 + mlp.pre_expert_norm)
    post_expert_norm [L, 1, Hm]     bf16  r16(1 + experts.post_expert_norm), applied BEFORE the norm
    router_bias      [L, 1, E]      f32   e_score_correction_bias (selection only)
    q                [L, H, qh*D]   bf16  this rank's query heads r*qh:(r+1)*qh, head-major lanes
    kv               [L, H, 2*kvh*D] bf16 lanes [0,kvh*D) = k heads r*kvh.., [kvh*D,2*kvh*D) = v
    gate             [L, H, qh*D]   bf16  attention output-gate logits, same head layout as q
    o                [L, qh*D, H]   bf16  input-sharded o_proj rows -> partial sums, all-reduce
    pre              [L, H, Hm/tp]  bf16  output-sharded pre_expert_proj columns, all-gather
    router_hi        [L, H, E]      bf16  hi = bf16(W)          (W = f32 router weight, replicated)
    router_lo        [L, H, E]      bf16  lo = bf16(W - f32(hi)); logits = dot(x,hi)+dot(x,lo) f32
    post             [L, Hm, H/tp]  bf16  output-sharded post_expert_proj columns, all-gather
    gate_up_q        [L, E, Hm, 2*Is] int4  cols [0,Is) = gate cols r*Is.., [Is,2*Is) = up I+r*Is..
    gate_up_s        [L, E, Hm/KC, KC/G, 2*Is] f32  group scales, chunked (KC = min(512, Hm))
    down_q           [L, E, Is, Hm] int4  down_proj rows r*Is:(r+1)*Is
    down_s           [L, E, Is/KC, KC/G, Hm] f32   (KC = min(512, Is))

  or, with `expert_format="nvfp4"` (container format v2; the int4 families are then absent):

    gate_up_fp4      [L, E, Hm/8, 2*Is] int32  e2m1 codes, 8 K-rows per word (`quant.pack_fp4_rows`)
    gate_up_bs       [L, E, Hm/KC, KC/16, 2*Is] float8_e4m3fn  block-16 scales, K-chunk major
    down_fp4         [L, E, Is/8, Hm]   int32
    down_bs          [L, E, Is/KC, KC/16, Hm] float8_e4m3fn
    expert_gs        [L, E, 8, 128]     f32   per-expert global scales, one per ROW replicated over
                                             the lanes: row 0 gate half, row 1 up half, row 2 down;
                                             rows 3..7 zero. `w = e2m1 * e4m3 * gs` (quant.py).

int4 storage: arrays of dtype `jnp.int4` (= `ml_dtypes.int4`, one byte per element on the host)
with shape `[..., K, N]`, K = contraction axis (a multiple of 64 rows, the int4 sublane tile);
groups of `G` consecutive rows of K share one scale. Scales are f32 (bf16-representable values)
stored chunked `[..., K/KC, KC/G, N]` with `KC = quant.k_chunk(K) = min(512, K)` so the kernel
indexes them on a leading untiled axis; the flat group-major form is `[..., K/G, N]`
(`quant.scales_to_chunked` / `scales_from_chunked`);
`dequant = q.astype(f32) * repeat(flat_scales, G, axis=-2)`. On disk the int4 values are packed
two per byte along the LAST axis, low nibble first (`quant.pack_int4`).

KV cache (design.md section 3): `[L, B, context, lanes]` bf16 for K and for V, with
`lanes = cache_lanes(cfg, tp) = roundup(kvh * D, 128)`; lanes `[h*D, (h+1)*D)` hold local kv
head `h` (`h < kvh`; MINI: lanes 64:128 are zero); K is post-QK-norm, post-RoPE.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from musespark.config import Config
from musespark.quant import k_chunk

BF16 = jnp.bfloat16
F32 = jnp.float32
INT4 = jnp.int4
I32 = jnp.int32
E4M3 = jnp.float8_e4m3fn

BANKS = 12  # dense-weight ring depth (design.md 5.2)
BANK_TILE = 1024  # maximum bank tile side, in elements
VOCAB_ALIGN = 1024  # per-rank vocab shard padded to a multiple of this
CACHE_LANES = 128  # KV-cache minor axis is padded to a multiple of this
INT4_ROWS = 64  # int4 sublane tile on TPU7x: every int4 slab has K % 64 == 0
FP4_ROWS = 64  # 8 int32 sublanes x 8 codes per word: every fp4 slab has K % 64 == 0
FP4_BLOCK = 16  # NVFP4 block size along K
GS_ROWS, GS_LANES = 8, 128  # expert_gs tile
EXPERT_FORMATS = ("int4", "nvfp4")

# Dense families streamed through the bank ring, in per-layer consumption order.
STREAMED_FAMILIES = ("q", "kv", "gate", "o", "pre", "router_hi", "router_lo", "post")
# Small replicated per-layer vectors.
VECTOR_FAMILIES = (
    "attn_norm",
    "ffn_norm",
    "post_ffn_norm",
    "attn_gate_alpha",
    "attn_gate_beta",
    "ffn_gate_alpha",
    "ffn_gate_beta",
    "pre_expert_norm",
    "post_expert_norm",
    "router_bias",
)
EXPERT_FAMILIES = ("gate_up_q", "gate_up_s", "down_q", "down_s")  # expert_format == "int4"
FP4_EXPERT_FAMILIES = ("gate_up_fp4", "gate_up_bs", "down_fp4", "down_bs", "expert_gs")
GLOBAL_FAMILIES = ("embed", "lm_head", "final_norm")


def expert_families(expert_format="int4"):
    """The per-rank expert families of an expert format (`EXPERT_FORMATS`)."""
    if expert_format not in EXPERT_FORMATS:
        raise ValueError(f"unknown expert_format {expert_format!r}; choose from {EXPERT_FORMATS}")
    return EXPERT_FAMILIES if expert_format == "int4" else FP4_EXPERT_FAMILIES


def expert_format_of(names):
    """`"int4"` or `"nvfp4"` from the family names of a weight tree / layout document."""
    names = set(names)
    if set(FP4_EXPERT_FAMILIES) <= names:
        return "nvfp4"
    if set(EXPERT_FAMILIES) <= names:
        return "int4"
    raise ValueError("no complete expert family set among the names")


def _round_up(x, m):
    return -(-x // m) * m


def check_tp(cfg: Config, tp: int, expert_format="int4"):
    """Raise unless `tp` ranks split every sharded axis evenly."""
    expert_families(expert_format)
    for name, size in (
        ("heads", cfg.heads),
        ("kv_heads", cfg.kv_heads),
        ("expert_hidden", cfg.expert_hidden),
        ("hidden", cfg.hidden),
        ("moe_hidden", cfg.moe_hidden),
    ):
        if size % tp:
            raise ValueError(f"tp={tp} does not divide {name}={size}")
    for name, K in (("moe_hidden", cfg.moe_hidden), ("expert_hidden/tp", cfg.expert_hidden // tp)):
        if expert_format == "nvfp4":
            if K % FP4_ROWS:
                raise ValueError(f"fp4 slab K={name}={K} must be a multiple of {FP4_ROWS} rows")
            if k_chunk(K) % FP4_BLOCK:
                raise ValueError(f"the fp4 block {FP4_BLOCK} must divide the K chunk {k_chunk(K)}")
            continue
        if K % cfg.group_size:
            raise ValueError(f"group_size={cfg.group_size} must divide {name}={K}")
        if K % INT4_ROWS:
            raise ValueError(f"int4 slab K={name}={K} must be a multiple of {INT4_ROWS} rows")
        if k_chunk(K) % cfg.group_size:
            raise ValueError(f"group_size={cfg.group_size} must divide the K chunk {k_chunk(K)}")


def vocab_pad(cfg: Config, tp: int = 8) -> int:
    """Per-rank vocab shard `Vp = ceil(V / tp / 1024) * 1024` (25600 for the real model)."""
    return _round_up(-(-cfg.vocab // tp), VOCAB_ALIGN)


def cache_lanes(cfg: Config, tp: int = 8) -> int:
    """Minor axis of the per-rank KV caches: local kv heads x head_dim, padded to 128."""
    return _round_up((cfg.kv_heads // tp) * cfg.head_dim, CACHE_LANES)


def rank_shapes(
    cfg: Config, tp: int = 8, expert_format="int4"
) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """`name -> (per-rank shape, dtype)` in the order of the table in the module docstring
    (`expert_format` selects the int4 g128 or the NVFP4 expert families)."""
    check_tp(cfg, tp, expert_format)
    L, H, Hm, E, G = cfg.layers, cfg.hidden, cfg.moe_hidden, cfg.experts, cfg.group_size
    qw = (cfg.heads // tp) * cfg.head_dim
    kvw = (cfg.kv_heads // tp) * cfg.head_dim
    isl = cfg.expert_hidden // tp
    kc_gu, kc_dn = k_chunk(Hm), k_chunk(isl)
    vp = vocab_pad(cfg, tp)
    if expert_format == "nvfp4":
        experts = {
            "gate_up_fp4": ((L, E, Hm // 8, 2 * isl), I32),
            "gate_up_bs": ((L, E, Hm // kc_gu, kc_gu // FP4_BLOCK, 2 * isl), E4M3),
            "down_fp4": ((L, E, isl // 8, Hm), I32),
            "down_bs": ((L, E, isl // kc_dn, kc_dn // FP4_BLOCK, Hm), E4M3),
            "expert_gs": ((L, E, GS_ROWS, GS_LANES), F32),
        }
    else:
        experts = {
            "gate_up_q": ((L, E, Hm, 2 * isl), INT4),
            "gate_up_s": ((L, E, Hm // kc_gu, kc_gu // G, 2 * isl), F32),
            "down_q": ((L, E, isl, Hm), INT4),
            "down_s": ((L, E, isl // kc_dn, kc_dn // G, Hm), F32),
        }
    return {
        "embed": ((vp, H), BF16),
        "lm_head": ((H, vp), BF16),
        "final_norm": ((1, H), BF16),
        "attn_norm": ((L, 1, H), BF16),
        "ffn_norm": ((L, 1, H), BF16),
        "post_ffn_norm": ((L, 1, H), BF16),
        "attn_gate_alpha": ((L, 1, H), F32),
        "attn_gate_beta": ((L, 1, H), F32),
        "ffn_gate_alpha": ((L, 1, H), F32),
        "ffn_gate_beta": ((L, 1, H), F32),
        "pre_expert_norm": ((L, 1, Hm), BF16),
        "post_expert_norm": ((L, 1, Hm), BF16),
        "router_bias": ((L, 1, E), F32),
        "q": ((L, H, qw), BF16),
        "kv": ((L, H, 2 * kvw), BF16),
        "gate": ((L, H, qw), BF16),
        "o": ((L, qw, H), BF16),
        "pre": ((L, H, Hm // tp), BF16),
        "router_hi": ((L, H, E), BF16),
        "router_lo": ((L, H, E), BF16),
        "post": ((L, Hm, H // tp), BF16),
        **experts,
    }


def kv_cache_shapes(cfg: Config, batch: int, context: int, tp: int = 8):
    """`{"k_cache": (shape, dtype), "v_cache": ...}` per rank; context must be a 128-multiple."""
    if context % 128:
        raise ValueError("context must be a multiple of 128")
    shape = (cfg.layers, batch, context, cache_lanes(cfg, tp))
    return {"k_cache": (shape, BF16), "v_cache": (shape, BF16)}


class Tile(NamedTuple):
    """One `[bk, bn]` window `w[layer, k0:k0+bk, n0:n0+bn]` of a streamed per-rank matrix."""

    family: str
    k0: int
    n0: int
    bk: int
    bn: int


@dataclass(frozen=True)
class TileGeometry:
    bank_k: int  # bank rows (max K tile)
    bank_n: int  # bank lanes (max N tile)
    banks: int = BANKS
    dtype: object = BF16

    @property
    def bank_bytes(self):
        return self.bank_k * self.bank_n * np.dtype(self.dtype).itemsize


def tile_geometry(cfg: Config) -> TileGeometry:
    """Bank tile `[min(1024, H), min(1024, H)]` bf16; every scheduled tile fits inside it."""
    side = min(BANK_TILE, cfg.hidden)
    return TileGeometry(bank_k=side, bank_n=side)


def _matrix_tiles(family, K, N, geo):
    bk, bn = min(geo.bank_k, K), min(geo.bank_n, N)
    if K % bk or N % bn:
        raise ValueError(f"{family}: [{K}, {N}] is not tileable by [{bk}, {bn}]")
    # N-tiles outer, K-tiles inner: the consumer accumulates a full K sweep per output block.
    return [Tile(family, k0, n0, bk, bn) for n0 in range(0, N, bn) for k0 in range(0, K, bk)]


@dataclass(frozen=True)
class Schedule:
    layer: tuple[Tile, ...]  # tiles of one layer, in consumption order
    lm_head: tuple[Tile, ...]  # tiles of lm_head, streamed after the last layer
    geometry: TileGeometry

    @property
    def tiles_per_layer(self):
        return len(self.layer)

    def total(self, layers):
        return layers * self.tiles_per_layer + len(self.lm_head)

    def tile(self, g, layers):
        """Global tile counter `g` -> (layer, Tile); lm_head tiles report layer == layers."""
        per = self.tiles_per_layer
        if g < layers * per:
            return g // per, self.layer[g % per]
        return layers, self.lm_head[g - layers * per]


def tile_schedule(cfg: Config, tp: int = 8) -> Schedule:
    """Dense-weight ring order: q, kv, gate, o, pre, router_hi, router_lo, post per layer."""
    shapes = rank_shapes(cfg, tp)
    geo = tile_geometry(cfg)
    layer = []
    for family in STREAMED_FAMILIES:
        (_, K, N), _ = shapes[family]
        layer += _matrix_tiles(family, K, N, geo)
    (K, N), _ = shapes["lm_head"]
    return Schedule(tuple(layer), tuple(_matrix_tiles("lm_head", K, N, geo)), geo)


def nbytes(shape, dtype) -> int:
    """Byte size with int4 counted as half a byte per element."""
    n = int(np.prod(shape))
    return n // 2 if np.dtype(dtype) == np.dtype(INT4) else n * np.dtype(dtype).itemsize


def bytes_per_rank(cfg: Config, tp: int = 8, expert_format="int4") -> dict[str, int]:
    """Per-name byte counts plus the aggregates `experts`, `expert_scales`, `dense`,
    `vectors`, `vocab` and `total` (int4 counted at 4 bits)."""
    shapes = rank_shapes(cfg, tp, expert_format)
    out = {name: nbytes(shape, dtype) for name, (shape, dtype) in shapes.items()}
    if expert_format == "nvfp4":
        out["experts"] = out["gate_up_fp4"] + out["down_fp4"]
        out["expert_scales"] = out["gate_up_bs"] + out["down_bs"] + out["expert_gs"]
    else:
        out["experts"] = out["gate_up_q"] + out["down_q"]
        out["expert_scales"] = out["gate_up_s"] + out["down_s"]
    out["dense"] = sum(out[f] for f in STREAMED_FAMILIES)
    out["vectors"] = sum(out[f] for f in VECTOR_FAMILIES)
    out["vocab"] = out["embed"] + out["lm_head"]
    out["total"] = sum(out[name] for name in shapes)
    return out
