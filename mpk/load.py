"""Muse-Spark checkpoint loading, NVFP4 slicing, TP weight layouts, packed containers.

Kernel weight layout (per rank, stacked [tp, ...] in containers):

- ``emb`` [vocab/8, 8192] bf16; ``lmw`` [8192, lm_width] bf16 (padded); ``fnorm`` [8192] f32.
- ``qwkv`` [62, 8192, 2560] bf16: [q | gate | k | v] column-fused head shards.
- ``ow`` [62, 1024, 8192] bf16 (row-parallel output projection).
- ``scalars`` [62, 8, 8192] f32: norm_in / norm_pre_ffn / norm_post_ffn gains
  (center-folded) then alpha_a / beta_a / alpha_f / beta_f residual gates, then 0.
- ``router`` [62, 8192, 32] f32 (this rank's expert columns); ``router_bias``
  [62, 256] f32 (replicated); ``expert_norms`` [62, 2, 4096] f32 (pre/post, folded).
- ``pool`` [62, 8192, 512] / ``unpool`` [62, 512, 8192] bf16 (pooler/unpooler).
- NVFP4 experts (60 layers), N-major: ``gu_packed`` [60, E, 1024, 2048] u8,
  ``gu_scales`` [60, E, 1024, 256] u8(E4M3), ``gu_global`` [60, E, 2] f32;
  ``dw_packed`` [60, E, 4096, 256] u8, ``dw_scales`` [60, E, 4096, 32] u8,
  ``dw_global`` [60, E] f32.
- BF16 expert layers (0, 61), N-major: ``gu_dense`` [2, E, 1024, 4096] bf16,
  ``dw_dense`` [2, E, 4096, 512] bf16.
"""

import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from . import dequantize_nvfp4, residual_coefficients


_SAFETENSORS_DTYPES = {
    "BF16": (np.uint16, 2),
    "F32": (np.float32, 4),
    "U8": (np.uint8, 1),
    "F8_E4M3": (np.uint8, 1),
}

BASE_FAMILIES = (
    "router router_bias scalars expert_norms emb fnorm "
    "gu_packed gu_scales gu_global dw_packed dw_scales dw_global gu_dense dw_dense"
).split()
DENSE_FAMILIES = ("qwkv", "ow", "pool", "unpool", "lmw")


class Checkpoint:
    """Raw-byte safetensors reader; BF16/F8 stay bit-faithful (numpy views)."""

    def __init__(self, path):
        path = Path(path)
        index = json.loads((path / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self.root = path
        self._headers = {}

    def _header(self, shard):
        if shard not in self._headers:
            with open(self.root / shard, "rb") as fh:
                size = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(size))
            self._headers[shard] = (header, 8 + size)
        return self._headers[shard]

    def info(self, name):
        header, _ = self._header(self.weight_map[name])
        entry = header[name]
        return entry["dtype"], entry["shape"], entry["data_offsets"]

    def get(self, name):
        dtype, shape, (start, end) = self.info(name)
        np_dtype, itemsize = _SAFETENSORS_DTYPES[dtype]
        _, base = self._header(self.weight_map[name])
        count = (end - start) // itemsize
        raw = np.fromfile(
            self.root / self.weight_map[name], dtype=np_dtype, count=count, offset=base + start
        ).reshape(shape)
        if dtype == "BF16":
            return raw.view(ml_dtypes.bfloat16)
        if dtype == "F8_E4M3":
            return raw.view(ml_dtypes.float8_e4m3fn)
        return raw


def canonical_weights(c, seed=0):
    """Random oracle-layout weights (BF16 experts, K-major) for tests."""
    rng = np.random.default_rng(seed)

    def normal(*shape, scale=0.02):
        return (rng.standard_normal(shape, dtype=np.float32) * scale).astype(ml_dtypes.bfloat16)

    def gains(*shape, scale=0.1):
        return (rng.standard_normal(shape, dtype=np.float32) * scale).astype(np.float32) + 1.0

    layers = []
    for _ in range(c.layers):
        gate_a = rng.standard_normal(c.dim, dtype=np.float32) * 0.3
        gate_f = rng.standard_normal(c.dim, dtype=np.float32) * 0.3
        alpha_a, beta_a = residual_coefficients(gate_a, c.gate_temperature)
        alpha_f, beta_f = residual_coefficients(gate_f, c.gate_temperature)
        layers.append(
            {
                "norm_in": gains(c.dim),
                "alpha_a": alpha_a,
                "beta_a": beta_a,
                "q": normal(c.dim, c.heads * c.head_dim),
                "k": normal(c.dim, c.kv_heads * c.head_dim),
                "v": normal(c.dim, c.kv_heads * c.head_dim),
                "gate": normal(c.dim, c.heads * c.head_dim),
                "o": normal(c.heads * c.head_dim, c.dim),
                "norm_pre_ffn": gains(c.dim),
                "alpha_f": alpha_f,
                "beta_f": beta_f,
                "norm_post_ffn": gains(c.dim),
                "router": rng.standard_normal((c.dim, c.experts), dtype=np.float32) * 0.02,
                "router_bias": np.zeros(c.experts, np.float32),
                "pool": normal(c.dim, c.moe_hidden),
                "unpool": normal(c.moe_hidden, c.dim),
                "norm_pre_expert": gains(c.moe_hidden),
                "norm_post_expert": gains(c.moe_hidden),
                "experts": {
                    "gate_up": normal(c.experts, c.moe_hidden, 2 * c.moe_intermediate),
                    "down": normal(c.experts, c.moe_intermediate, c.moe_hidden),
                },
            }
        )
    return {
        "embedding": normal(c.vocab, c.dim),
        "layers": tuple(layers),
        "norm_final": gains(c.dim),
        "lm_head": normal(c.dim, c.vocab),
    }


def zero_states(c, context):
    return tuple(
        (
            np.zeros((c.kv_heads, context, c.head_dim), ml_dtypes.bfloat16),
            np.zeros((c.kv_heads, context, c.head_dim), ml_dtypes.bfloat16),
        )
        for _ in range(c.layers)
    )


def lm_pad(c, tp):
    return -(c.vocab // tp) % 1280


def lm_width(c, tp):
    return c.vocab // tp + lm_pad(c, tp)


def _kv_span(c, rank, tp):
    """This rank's KV-head slice [start, count). Past tp > kv_heads, rank pairs
    share one KV head (both members compute and cache it identically)."""
    count = max(c.kv_heads // tp, 1)
    return (rank * c.kv_heads) // tp, count


def _gu_shard(n_w, rank, tp, inter):
    """[E, 2I, K...] N-major rows -> [E, 2*I/tp, K...] with [gate|up] halves."""
    width = inter // tp
    gate, up = n_w[:, :inter], n_w[:, inter:]
    return np.concatenate(
        (gate[:, rank * width : (rank + 1) * width], up[:, rank * width : (rank + 1) * width]),
        axis=1,
    )


def _scalars_row(w):
    """Pack a layer's 8192-wide FP32 vectors into the [8, dim] kernel row."""
    row = np.zeros((8, w["norm_in"].shape[0]), np.float32)
    row[0] = np.asarray(w["norm_in"], np.float32)
    row[1] = np.asarray(w["norm_pre_ffn"], np.float32)
    row[2] = np.asarray(w["norm_post_ffn"], np.float32)
    row[3] = w["alpha_a"]
    row[4] = w["beta_a"]
    row[5] = w["alpha_f"]
    row[6] = w["beta_f"]
    return row


def rank_weights(weights, c, rank, tp=8):
    """One rank's kernel layout from canonical (oracle-layout) weights."""
    out = {}
    vshard = c.vocab // tp
    out["emb"] = weights["embedding"][rank * vshard : (rank + 1) * vshard]
    lm = weights["lm_head"][:, rank * vshard : (rank + 1) * vshard]
    out["lmw"] = np.pad(lm, ((0, 0), (0, lm_pad(c, tp)))).astype(ml_dtypes.bfloat16)

    qh = c.heads // tp
    kvs, kvh = _kv_span(c, rank, tp)
    eshard = c.experts // tp
    mh_r = c.moe_hidden // tp
    width = c.moe_intermediate // tp
    qwkv, ow, scalars, router, router_bias = [], [], [], [], []
    pool, unpool, expert_norms = [], [], []
    gu_dense, dw_dense = [], []
    for w in weights["layers"]:
        scalars.append(_scalars_row(w))
        qs = rank * qh * c.head_dim
        ks = kvs * c.head_dim
        qwkv.append(
            np.concatenate(
                (
                    w["q"][:, qs : qs + qh * c.head_dim],
                    w["gate"][:, qs : qs + qh * c.head_dim],
                    w["k"][:, ks : ks + kvh * c.head_dim],
                    w["v"][:, ks : ks + kvh * c.head_dim],
                ),
                axis=1,
            ).astype(ml_dtypes.bfloat16)
        )
        ow.append(np.ascontiguousarray(w["o"][qs : qs + qh * c.head_dim]))
        router.append(np.ascontiguousarray(w["router"][:, rank * eshard : (rank + 1) * eshard]))
        router_bias.append(w["router_bias"])
        pool.append(np.ascontiguousarray(w["pool"][:, rank * mh_r : (rank + 1) * mh_r]))
        unpool.append(np.ascontiguousarray(w["unpool"][rank * mh_r : (rank + 1) * mh_r, :]))
        expert_norms.append(
            np.stack(
                (
                    np.asarray(w["norm_pre_expert"], np.float32),
                    np.asarray(w["norm_post_expert"], np.float32),
                )
            )
        )
        # Canonical experts are K-major ([E, K, N]) like the kernel layout.
        gu = w["experts"]["gate_up"]  # [E, K, 2I]
        gu_n = np.concatenate(
            (
                gu[:, :, rank * width : (rank + 1) * width],
                gu[:, :, c.moe_intermediate + rank * width : c.moe_intermediate + (rank + 1) * width],
            ),
            axis=2,
        )
        gu_dense.append(np.ascontiguousarray(gu_n).astype(ml_dtypes.bfloat16))
        dw = w["experts"]["down"][:, rank * width : (rank + 1) * width, :]  # [E, I/tp, K]
        dw_dense.append(np.ascontiguousarray(dw).astype(ml_dtypes.bfloat16))
    out["qwkv"] = np.stack(qwkv)
    out["ow"] = np.stack(ow)
    out["scalars"] = np.stack(scalars)
    out["router"] = np.stack(router).astype(np.float32)
    out["router_bias"] = np.stack(router_bias).astype(np.float32)
    out["pool"] = np.stack(pool)
    out["unpool"] = np.stack(unpool)
    out["expert_norms"] = np.stack(expert_norms)
    out["gu_dense"] = np.stack(gu_dense)  # [L, E, 2I/tp, K] N-major
    out["dw_dense"] = np.stack(dw_dense)  # [L, E, K(H), I/tp] N-major
    out["fnorm"] = np.asarray(weights["norm_final"], np.float32)[None, :]
    return out


def _layer_rank_pieces(ckpt, c, i, tp):
    """Read one layer's tensors once; return per-rank kernel-layout pieces."""
    p = f"model.language_model.layers.{i}."
    a = p + "self_attn."
    m = p + "mlp."
    qh = c.heads // tp
    kvh = max(c.kv_heads // tp, 1)
    eshard = c.experts // tp
    mh_r = c.moe_hidden // tp
    width = c.moe_intermediate // tp
    q_w = ckpt.get(a + "q_proj.weight")  # [out, in] rows = heads
    g_w = ckpt.get(a + "gate_proj.weight")
    k_w = ckpt.get(a + "k_proj.weight")
    v_w = ckpt.get(a + "v_proj.weight")
    o_w = ckpt.get(a + "o_proj.weight")
    norm_in = np.asarray(ckpt.get(p + "input_layernorm.weight"), np.float32) + 1.0
    alpha_a, beta_a = residual_coefficients(
        ckpt.get(p + "post_attention_residual_gate.gate"), c.gate_temperature
    )
    norm_pre = np.asarray(ckpt.get(p + "pre_feedforward_layernorm.weight"), np.float32) + 1.0
    norm_post = np.asarray(ckpt.get(p + "post_feedforward_layernorm.weight"), np.float32) + 1.0
    alpha_f, beta_f = residual_coefficients(
        ckpt.get(p + "post_feedforward_residual_gate.gate"), c.gate_temperature
    )
    row = np.zeros((8, c.dim), np.float32)
    row[0], row[1], row[2] = norm_in, norm_pre, norm_post
    row[3], row[4], row[5], row[6] = alpha_a, beta_a, alpha_f, beta_f
    router = np.asarray(ckpt.get(m + "gate.weight"), np.float32)  # [E, dim]
    router_bias = np.asarray(ckpt.get(m + "gate.e_score_correction_bias"), np.float32)
    pool_w = ckpt.get(m + "pre_expert_proj.weight")  # [moe_hidden, dim]
    unpool_w = ckpt.get(m + "post_expert_proj.weight")  # [dim, moe_hidden]
    norm_pre_x = np.asarray(ckpt.get(m + "pre_expert_norm.weight"), np.float32) + 1.0
    norm_post_x = np.asarray(ckpt.get(m + "experts.post_expert_norm.weight"), np.float32) + 1.0

    pieces = []
    for rank in range(tp):
        qs = rank * qh * c.head_dim
        ks = ((rank * c.kv_heads) // tp) * c.head_dim
        qwkv = np.concatenate(
            (
                q_w[qs : qs + qh * c.head_dim].T,
                g_w[qs : qs + qh * c.head_dim].T,
                k_w[ks : ks + kvh * c.head_dim].T,
                v_w[ks : ks + kvh * c.head_dim].T,
            ),
            axis=1,
        ).astype(ml_dtypes.bfloat16)
        ow = np.ascontiguousarray(o_w[:, qs : qs + qh * c.head_dim].T)
        r = {
            "qwkv": qwkv,
            "ow": ow,
            "scalars": row,
            "router": np.ascontiguousarray(router[rank * eshard : (rank + 1) * eshard].T),
            "router_bias": router_bias,
            "pool": np.ascontiguousarray(pool_w.T[:, rank * mh_r : (rank + 1) * mh_r]),
            "unpool": np.ascontiguousarray(unpool_w.T[rank * mh_r : (rank + 1) * mh_r, :]),
            "expert_norms": np.stack((norm_pre_x, norm_post_x)),
        }
        pieces.append(r)

    if i in c.bf16_expert_layers:
        gu = ckpt.get(m + "experts.gate_up_proj")  # [E, 2I, K] bf16
        dw = ckpt.get(m + "experts.down_proj")  # [E, K, I] bf16
        for rank in range(tp):
            pieces[rank]["gu_dense"] = np.ascontiguousarray(
                _gu_shard(gu, rank, tp, c.moe_intermediate).transpose(0, 2, 1)
            )
            pieces[rank]["dw_dense"] = np.ascontiguousarray(
                dw[..., rank * width : (rank + 1) * width].transpose(0, 2, 1)
            )
    else:
        gu = ckpt.get(m + "experts.gate_up_proj")  # [E, 2I, K/2] u8
        gus = ckpt.get(m + "experts.gate_up_proj_weight_scale")
        gug = np.asarray(ckpt.get(m + "experts.gate_up_proj_weight_scale_2"), np.float32)
        dw = ckpt.get(m + "experts.down_proj")  # [E, K, I/2] u8
        dws = ckpt.get(m + "experts.down_proj_weight_scale")
        dwg = np.asarray(ckpt.get(m + "experts.down_proj_weight_scale_2"), np.float32)
        for rank in range(tp):
            pieces[rank]["gu_packed"] = np.ascontiguousarray(
                _gu_shard(gu, rank, tp, c.moe_intermediate).transpose(0, 2, 1)
            )
            pieces[rank]["gu_scales"] = _gu_shard(gus, rank, tp, c.moe_intermediate).transpose(0, 2, 1).view(np.uint8)
            pieces[rank]["gu_global"] = gug
            pieces[rank]["dw_packed"] = np.ascontiguousarray(
                dw[..., rank * (width // 2) : (rank + 1) * (width // 2)].transpose(0, 2, 1)
            )
            pieces[rank]["dw_scales"] = dws[
                ..., rank * (width // 16) : (rank + 1) * (width // 16)
            ].transpose(0, 2, 1).view(np.uint8)
            pieces[rank]["dw_global"] = dwg
    return pieces


def pack_from_checkpoint(path, c, tp=8, workers=8):
    """Stream the real checkpoint into (base, dense) packed host dicts."""
    ckpt = Checkpoint(path)
    vshard = c.vocab // tp
    n_packed = c.layers - len(c.bf16_expert_layers)
    n_dense = len(c.bf16_expert_layers)
    e = c.experts
    h = c.dim
    mh_r = c.moe_hidden // tp
    width = c.moe_intermediate // tp
    packed_shape = {
        "gu_packed": (n_packed, e, c.moe_hidden // 2, 2 * width),
        "gu_scales": (n_packed, e, c.moe_hidden // 16, 2 * width),
        "gu_global": (n_packed, e, 2),
        "dw_packed": (n_packed, e, width // 2, c.moe_hidden),
        "dw_scales": (n_packed, e, width // 16, c.moe_hidden),
        "dw_global": (n_packed, e),
        "gu_dense": (n_dense, e, c.moe_hidden, 2 * width),
        "dw_dense": (n_dense, e, width, c.moe_hidden),
    }
    base = {
        "router": np.empty((tp, c.layers, h, c.experts // tp), np.float32),
        "router_bias": np.empty((tp, c.layers, e), np.float32),
        "scalars": np.empty((tp, c.layers, 8, h), np.float32),
        "expert_norms": np.empty((tp, c.layers, 2, c.moe_hidden), np.float32),
        "emb": np.empty((tp, vshard, h), ml_dtypes.bfloat16),
        "fnorm": np.empty((1, h), np.float32),
        "gu_packed": np.empty((tp, *packed_shape["gu_packed"]), np.uint8),
        "gu_scales": np.empty((tp, *packed_shape["gu_scales"]), np.uint8),
        "gu_global": np.empty((tp, *packed_shape["gu_global"]), np.float32),
        "dw_packed": np.empty((tp, *packed_shape["dw_packed"]), np.uint8),
        "dw_scales": np.empty((tp, *packed_shape["dw_scales"]), np.uint8),
        "dw_global": np.empty((tp, *packed_shape["dw_global"]), np.float32),
        "gu_dense": np.empty((tp, *packed_shape["gu_dense"]), ml_dtypes.bfloat16),
        "dw_dense": np.empty((tp, *packed_shape["dw_dense"]), ml_dtypes.bfloat16),
    }
    dense = {
        "qwkv": np.empty((tp, c.layers, h, 2 * (c.heads // tp) * c.head_dim + 2 * max(c.kv_heads // tp, 1) * c.head_dim), ml_dtypes.bfloat16),
        "ow": np.empty((tp, c.layers, (c.heads // tp) * c.head_dim, h), ml_dtypes.bfloat16),
        "pool": np.empty((tp, c.layers, h, mh_r), ml_dtypes.bfloat16),
        "unpool": np.empty((tp, c.layers, mh_r, h), ml_dtypes.bfloat16),
        "lmw": np.empty((tp, h, lm_width(c, tp)), ml_dtypes.bfloat16),
    }
    packed_ord = {layer: i for i, layer in enumerate(j for j in range(c.layers) if j not in c.bf16_expert_layers)}
    dense_ord = {layer: i for i, layer in enumerate(c.bf16_expert_layers)}

    def do_layer(i):
        pieces = _layer_rank_pieces(ckpt, c, i, tp)
        for rank, r in enumerate(pieces):
            for key in ("qwkv", "ow", "pool", "unpool"):
                dense[key][rank, i] = r[key]
            for key in ("router", "router_bias", "scalars", "expert_norms"):
                base[key][rank, i] = r[key]
            if i in dense_ord:
                o = dense_ord[i]
                base["gu_dense"][rank, o] = r["gu_dense"]
                base["dw_dense"][rank, o] = r["dw_dense"]
            else:
                o = packed_ord[i]
                for key in ("gu_packed", "gu_scales", "gu_global", "dw_packed", "dw_scales", "dw_global"):
                    base[key][rank, o] = r[key]
        return i

    with ThreadPoolExecutor(workers) as pool_exec:
        for n, _ in enumerate(pool_exec.map(do_layer, range(c.layers))):
            if (n + 1) % 4 == 0:
                print(f"  packed {n + 1}/{c.layers} layers", flush=True)

    emb = ckpt.get("model.language_model.embed_tokens.weight")
    for rank in range(tp):
        base["emb"][rank] = emb[rank * vshard : (rank + 1) * vshard]
    lm = ckpt.get("lm_head.weight")  # [vocab, dim]
    for rank in range(tp):
        dense["lmw"][rank, :, : lm_width(c, tp) - lm_pad(c, tp)] = lm[
            rank * vshard : (rank + 1) * vshard
        ].T
    base["fnorm"][:] = np.asarray(ckpt.get("model.language_model.norm.weight"), np.float32)[None, :]
    return base, dense


def pack_host(weights, c, tp=8):
    """Canonical (test) weights -> (base, dense) packed host dicts."""
    ranks = [rank_weights(weights, c, r, tp) for r in range(tp)]
    merged = {k: np.stack([r[k] for r in ranks]) for k in ranks[0]}
    base = {k: merged[k] for k in BASE_FAMILIES if k in merged}
    dense = {k: merged[k] for k in DENSE_FAMILIES if k in merged}
    # Test weights carry every expert in the dense families; the NVFP4 families
    # get empty-shell arrays so the kernel's refs exist (never indexed).
    e = c.experts
    width = c.moe_intermediate // tp
    base.setdefault("gu_packed", np.zeros((tp, 1, e, c.moe_hidden // 2, 2 * width), np.uint8))
    base.setdefault("gu_scales", np.zeros((tp, 1, e, c.moe_hidden // 16, 2 * width), np.uint8))
    base.setdefault("gu_global", np.zeros((tp, 1, e, 2), np.float32))
    base.setdefault("dw_packed", np.zeros((tp, 1, e, width // 2, c.moe_hidden), np.uint8))
    base.setdefault("dw_scales", np.zeros((tp, 1, e, width // 16, c.moe_hidden), np.uint8))
    base.setdefault("dw_global", np.zeros((tp, 1, e), np.float32))
    return base, dense


def until_matrix(t, k, n):
    """Inverse of tile_matrix: [..., T, bk, bn] -> [..., K, N]."""
    *lead, T, bk, bn = t.shape
    nk, nb = k // bk, n // bn
    L = len(lead)
    order = list(range(L)) + [L + 1, L + 2, L + 0, L + 3]
    w = t.reshape(*lead, nb, nk, bk, bn).transpose(order)
    return w.reshape(*lead, nk * bk, nb * bn)


def tile_matrix(w, bk, bn):
    """[..., K, N] -> [..., T, bk, bn] with each tile contiguous (ti = ni*nk + ki)."""
    *lead, K, N = w.shape
    nk, nb = K // bk, N // bn
    L = len(lead)
    order = list(range(L)) + [L + 2, L + 0, L + 1, L + 3]
    tiled = w.reshape(*lead, nk, bk, nb, bn).transpose(order)
    return np.ascontiguousarray(tiled.reshape(*lead, nb * nk, bk, bn))


def tile_hw(k, n, max_k=None, max_n=None):
    """(bk, bn) tile for a [k, n] matrix: largest 128-multiples within caps.

    Test-scale dims without a 128-multiple divisor fall back to the largest
    plain divisor within the cap (CPU-only configs; TPU dims always qualify).
    """
    import os

    max_k = max_k or int(os.environ.get("MPK_MAXK", "1024"))
    max_n = max_n or int(os.environ.get("MPK_MAXN", "1280"))

    def pick(d, cap):
        divisors = [b for b in range(128, min(d, cap) + 1, 128) if d % b == 0]
        if not divisors:
            divisors = [b for b in range(min(d, cap), 0, -1) if d % b == 0]
        return max(divisors)

    return pick(k, max_k), pick(n, max_n)


def dense_schedule(c, tp):
    """Tiling per dense family: (name, bk, bn, nk, nb, tiles)."""
    h = c.dim
    qh = (c.heads // tp) * c.head_dim
    kvh = max(c.kv_heads // tp, 1) * c.head_dim
    out = []
    for name, k, n in (
        ("qwkv", h, 2 * qh + 2 * kvh),
        ("ow", qh, h),
        ("pool", h, c.moe_hidden // tp),
        ("unpool", c.moe_hidden // tp, h),
        ("lmw", h, lm_width(c, tp)),
    ):
        bk, bn = tile_hw(k, n)
        out.append((name, bk, bn, k // bk, n // bn, (k // bk) * (n // bn)))
    return out


def tile_dense(dense, c, tp):
    tiled = {}
    for name, bk, bn, *_ in dense_schedule(c, tp):
        tiled[name] = tile_matrix(dense[name], bk, bn)
    return tiled


def save_pack(directory, tree, progress=False):
    """Write ``tree`` (pytree of host arrays) as directory/pack.bin + manifest."""
    import pickle
    import uuid

    directory = Path(directory)
    marker = directory / "manifest.json"
    if marker.exists():
        return
    tmp = directory.with_suffix(f".tmp.{uuid.uuid4().hex[:8]}")
    tmp.mkdir(parents=True, exist_ok=True)
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    manifest = {"treedef": pickle.dumps(treedef).hex(), "leaves": []}
    offset = 0
    with open(tmp / "pack.bin", "wb") as bf:
        for index, leaf in enumerate(leaves):
            a = np.asarray(leaf)
            dtype = str(a.dtype)
            if a.dtype == ml_dtypes.bfloat16:
                a = a.view(np.uint16)
                dtype = "bf16"
            elif a.dtype == ml_dtypes.float8_e4m3fn:
                a = a.view(np.uint8)
                dtype = "f8e4m3"
            view = memoryview(a)
            if not a.flags["C_CONTIGUOUS"]:
                a = np.ascontiguousarray(a)
                view = memoryview(a)
            bview = view.cast("B")  # zero-copy (tobytes would double peak RAM)
            for start in range(0, len(bview), 1 << 30):
                bf.write(bview[start : start + (1 << 30)])
            manifest["leaves"].append(
                {"offset": offset, "nbytes": len(bview), "dtype": dtype, "shape": list(a.shape)}
            )
            offset += len(bview)
            if progress:
                print(f"  wrote leaf {index + 1}/{len(leaves)} ({len(bview) >> 20} MiB)", flush=True)
            del bview, view, a
    (tmp / "manifest.json").write_text(json.dumps(manifest))
    tmp.rename(directory)


def _pread_direct(fd, nbytes, offset):
    """O_DIRECT read into a page-aligned numpy buffer (no page-cache pollution)."""
    align = 4096
    start = offset & ~(align - 1)
    end = (offset + nbytes + align - 1) & ~(align - 1)
    buf = np.empty(end - start + 2 * align, np.uint8)
    addr = buf.ctypes.data
    off = (-addr) % align
    window = buf[off : off + (end - start)]
    need = offset - start + nbytes
    got = 0
    while got < need:
        n = os.preadv(fd, [memoryview(window)[got:]], start + got)
        if n == 0:
            break  # EOF past the aligned tail is fine
        got += n
    if got < need:
        raise IOError("short O_DIRECT read")
    return buf[off + (offset - start) : off + (offset - start) + nbytes]


def load_pack(directory, workers=16):
    """Read a ``save_pack`` container back into a host pytree (O_DIRECT)."""
    import pickle

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    treedef = pickle.loads(bytes.fromhex(manifest["treedef"]))
    fd = os.open(directory / "pack.bin", os.O_RDONLY | os.O_DIRECT)
    leaves = []
    try:
        for leaf in manifest["leaves"]:
            a = _pread_direct(fd, leaf["nbytes"], leaf["offset"])
            dtype = {"bf16": np.uint16, "f8e4m3": np.uint8}.get(leaf["dtype"], leaf["dtype"])
            a = a.view(dtype).reshape(leaf["shape"])
            if leaf["dtype"] == "bf16":
                a = a.view(ml_dtypes.bfloat16)
            elif leaf["dtype"] == "f8e4m3":
                a = a.view(ml_dtypes.float8_e4m3fn)
            leaves.append(a)
    finally:
        os.close(fd)
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _read_range(fd_path, offset, want, buf, start):
    """Read [offset, offset+want) into buf[start:] with a dedicated fd."""
    fd = os.open(fd_path, os.O_RDONLY)
    got = 0
    try:
        while got < want:
            n = os.preadv(fd, [memoryview(buf)[start + got : start + want]], offset + got)
            if n == 0:
                break
            got += n
    finally:
        os.close(fd)
    if got != want:
        raise IOError(f"short read: {got} < {want}")


def place_pack(mesh, directory, progress=False, read_workers=8, *, merge_experts=False):
    """Stream a pack container straight to devices, pipelining parallel reads.

    Each leaf is read in ~1 GiB ranges spread over the thread pool (one fd per
    call; the GCS fuse streams per-fd), placed per rank, and the host buffer
    freed after the transfer completes — host RAM stays ~2 leaves in flight.
    """
    import pickle

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    treedef = pickle.loads(bytes.fromhex(manifest["treedef"]))
    indices = jax.tree_util.tree_unflatten(treedef, list(range(len(manifest["leaves"]))))
    merge_pairs = {}
    if merge_experts:
        for name in ("gu", "dw"):
            if f"{name}_packed" in indices and f"{name}_scales" in indices:
                merge_pairs[indices[f"{name}_packed"]] = indices[f"{name}_scales"]
    devices = list(mesh.devices)
    # Multi-process: each host reads/places only its own ranks' shards.
    local_ranks = [r for r, d in enumerate(devices) if d.process_index == jax.process_index()]
    spec = NamedSharding(mesh, P("tp"))
    path = str(directory / "pack.bin")

    def submit_leaf(pool, i):
        leaf = manifest["leaves"][i]
        buf = bytearray(leaf["nbytes"])
        chunk = 1 << 30
        futs = [
            pool.submit(
                _read_range,
                path,
                leaf["offset"] + off,
                min(chunk, leaf["nbytes"] - off),
                buf,
                off,
            )
            for off in range(0, leaf["nbytes"], chunk)
        ]
        return buf, futs

    def submit_rank(pool, leaf, r):
        """Read one rank's contiguous byte range of a [tp, ...] leaf."""
        per_rank = leaf["nbytes"] // len(devices)
        buf = bytearray(per_rank)
        chunk = 1 << 30
        base_off = leaf["offset"] + r * per_rank
        futs = [
            pool.submit(
                _read_range,
                path,
                base_off + off,
                min(chunk, per_rank - off),
                buf,
                off,
            )
            for off in range(0, per_rank, chunk)
        ]
        return buf, futs

    def view_of(leaf, buf):
        dtype = {"bf16": np.uint16, "f8e4m3": np.uint8}.get(leaf["dtype"], leaf["dtype"])
        a = np.frombuffer(buf, dtype=dtype).reshape(leaf["shape"])
        if leaf["dtype"] == "bf16":
            a = a.view(ml_dtypes.bfloat16)
        elif leaf["dtype"] == "f8e4m3":
            a = a.view(ml_dtypes.float8_e4m3fn)
        return a

    with ThreadPoolExecutor(read_workers) as pool:
        placed = []
        rank_bytes = 8 << 30  # big [tp, ...] leaves stream rank-by-rank
        for i, leaf in enumerate(manifest["leaves"]):
            if i in merge_pairs.values():
                placed.append(None)
                continue
            if i in merge_pairs:
                scale_leaf = manifest["leaves"][merge_pairs[i]]
                arrays = []
                for r in local_ranks:
                    values_buf, values_futures = submit_rank(pool, leaf, r)
                    scales_buf, scales_futures = submit_rank(pool, scale_leaf, r)
                    for future in (*values_futures, *scales_futures):
                        future.result()
                    values = view_of(dict(leaf, shape=[1, *leaf["shape"][1:]]), values_buf)
                    scales = view_of(dict(scale_leaf, shape=[1, *scale_leaf["shape"][1:]]), scales_buf)
                    merged = np.concatenate((values, scales), axis=-2)
                    shard = jax.device_put(merged, jax.sharding.SingleDeviceSharding(devices[r]))
                    arrays.append(jax.block_until_ready(shard))
                    del merged, values, scales, values_buf, scales_buf
                shape = (len(devices), *arrays[0].shape[1:])
                placed.append(jax.make_array_from_single_device_arrays(shape, spec, arrays))
                if progress:
                    print(f"  placed merged leaf {i + 1}/{len(manifest['leaves'])}", flush=True)
            elif leaf["shape"][0] == len(devices) and leaf["nbytes"] > rank_bytes:
                pending_r = submit_rank(pool, leaf, local_ranks[0])
                arrays = []
                for pos, r in enumerate(local_ranks):
                    buf, futs = pending_r
                    for f in futs:
                        f.result()
                    sub_leaf = dict(leaf, shape=[1, *leaf["shape"][1:]], nbytes=len(buf))
                    shard = jax.device_put(
                        view_of(sub_leaf, buf), jax.sharding.SingleDeviceSharding(devices[r])
                    )
                    if pos + 1 < len(local_ranks):
                        pending_r = submit_rank(pool, leaf, local_ranks[pos + 1])
                    arrays.append(shard)
                    del buf
                jax.block_until_ready(arrays)
                arr = jax.make_array_from_single_device_arrays(leaf["shape"], spec, arrays)
                placed.append(arr)
                if progress:
                    print(
                        f"  placed leaf {i + 1}/{len(manifest['leaves'])} ({leaf['nbytes'] >> 20} MiB, per-rank)",
                        flush=True,
                    )
            else:
                buf, futs = submit_leaf(pool, i)
                for f in futs:
                    f.result()
                a = _tile_replicated(view_of(leaf, buf), len(devices))
                per = [
                    jax.device_put(a[r : r + 1], jax.sharding.SingleDeviceSharding(devices[r]))
                    for r in local_ranks
                ]
                jax.block_until_ready(per)
                arr = jax.make_array_from_single_device_arrays(a.shape, spec, per)
                placed.append(arr)
                del buf
                if progress:
                    print(
                        f"  placed leaf {i + 1}/{len(manifest['leaves'])} ({leaf['nbytes'] >> 20} MiB)",
                        flush=True,
                    )
    jax.block_until_ready(placed)
    result = jax.tree_util.tree_unflatten(treedef, placed)
    if merge_pairs:
        for name in ("gu", "dw"):
            if f"{name}_scales" in result:
                result[f"{name}_pack"] = result.pop(f"{name}_packed")
                del result[f"{name}_scales"]
    return result


def save_bundle(directory, base, dense_untiled, dense_tiled, progress=False):
    directory = Path(directory)
    save_pack(directory / "base", base, progress=progress)
    save_pack(directory / "prefill", dense_untiled, progress=progress)
    save_pack(directory / "tiled", dense_tiled, progress=progress)


def load_bundle(directory):
    """-> (prefill_weights, decode_weights); base leaves are shared references."""
    directory = Path(directory)
    base = load_pack(directory / "base")
    prefill = {**base, **load_pack(directory / "prefill")}
    decode = {**base, **load_pack(directory / "tiled")}
    return prefill, decode


def load_target(checkpoint, c, tp, packed=None):
    """(prefill_weights, decode_weights) host dicts; parse HF or load a bundle."""
    if packed is not None and (Path(packed) / "base" / "manifest.json").exists():
        return load_bundle(packed)
    base, dense = pack_from_checkpoint(checkpoint, c, tp)
    tiled = tile_dense(dense, c, tp)
    if packed is not None:
        save_bundle(packed, base, dense, tiled, progress=True)
    return {**base, **dense}, {**base, **tiled}


def load_tokenizer(checkpoint):
    """The checkpoint's tokenizer.json as a minimal encode/decode adapter.

    ``encode`` never adds special tokens implicitly (matching the reference
    server's completions path: this tokenizer's post-processor is inert).
    """
    from tokenizers import Tokenizer

    class _Tokenizer:
        def __init__(self, inner):
            self.inner = inner

        def encode(self, text, **unused):
            return self.inner.encode(text, add_special_tokens=False).ids

        def decode(self, ids, **unused):
            return self.inner.decode([int(t) for t in ids])

    return _Tokenizer(Tokenizer.from_file(str(Path(checkpoint) / "tokenizer.json")))


def _tile_replicated(a, tp):
    """Replicated leaves get a [tp, 1, ...] leading layout (uniform tp-shaping)."""
    if a.shape[0] == tp:
        return a
    return np.ascontiguousarray(np.broadcast_to(a[None], (tp,) + a.shape))


def build_sharded(mesh, packed):
    """Place [tp, ...] leaves per-rank (avoids a monolithic staging compile)."""
    devices = list(mesh.devices)
    local_ranks = [r for r, d in enumerate(devices) if d.process_index == jax.process_index()]
    spec = NamedSharding(mesh, P("tp"))

    def put(a):
        a = _tile_replicated(np.asarray(a), len(devices))
        per = [
            jax.device_put(a[r : r + 1], jax.sharding.SingleDeviceSharding(devices[r]))
            for r in local_ranks
        ]
        return jax.make_array_from_single_device_arrays(a.shape, spec, per)

    return jax.tree.map(put, packed)


def merge_expert_leaves(w):
    """Merge the packed-expert families into single-copy blobs (gu_pack, dw_pack).

    [..., K/2, N] + [..., K/16, N] -> [..., K/2 + K/16, N] along the K-ish axis,
    so the TP32 decode kernel streams one copy per expert instead of two. Works
    on host (numpy) or device (jax) arrays; the prefill path slices them back out.
    """
    w = dict(w)
    cat = jnp.concatenate if isinstance(w["gu_packed"], jax.Array) else np.concatenate
    w["gu_pack"] = cat([w["gu_packed"], w["gu_scales"]], axis=-2)
    w["dw_pack"] = cat([w["dw_packed"], w["dw_scales"]], axis=-2)
    for k in ("gu_packed", "gu_scales", "dw_packed", "dw_scales"):
        del w[k]
    return w


def prepare_tp8_weights(mesh, weights):
    """Prepare output-column projections for single-host decode."""
    if mesh.size != 8:
        raise ValueError("TP8 preparation requires eight devices")
    if "ow_group8" in weights:
        return weights
    weights = dict(weights)
    if "gu_pack" not in weights:
        weights = merge_expert_leaves(weights)
    hidden = weights["scalars"].shape[-1]
    experts = weights["router_bias"].shape[-1]
    shard = NamedSharding(mesh, P("tp"))

    @jax.jit(out_shardings=shard)
    def columns(value):
        _, layers, width, _ = value.shape
        return value.reshape(8, layers, width, 8, hidden // 8).transpose(3, 1, 0, 2, 4).reshape(8, layers, 8 * width, hidden // 8)

    @jax.jit(out_shardings=(shard, shard))
    def router_order(value, bias):
        _, layers, width, _ = value.shape
        order = jnp.arange(experts, dtype=jnp.int32)
        if experts in (128, 256):
            order = (127 - order % 128) * (experts // 128) + order // 128
        full = value.transpose(1, 2, 0, 3).reshape(layers, width, experts)
        return full[..., order].reshape(layers, width, 8, experts // 8).transpose(2, 0, 3, 1), bias[..., order]

    router, bias = router_order(weights["router"], weights["router_bias"])
    return jax.block_until_ready({**weights, "ow_group8": columns(weights["ow"]),
        "pool_local": weights["pool"], "unpool_group8": columns(weights["unpool"]),
        "router_ordered": router, "router_bias_ordered": bias})


def prepare_tp16_weights(mesh, weights):
    """Keep one TP16 expert partition and prepare the two-host dense layouts."""
    if mesh.size != 16:
        raise ValueError("TP16 preparation requires sixteen devices")
    if "ow_group8" in weights:
        return weights
    weights = dict(weights)
    if "gu_pack" not in weights:
        if "gu_scales" in weights:
            weights = merge_expert_leaves(weights)
        else:
            weights["gu_pack"] = weights.pop("gu_packed")
            weights["dw_pack"] = weights.pop("dw_packed")
    hidden = weights["scalars"].shape[-1]
    experts = weights["router_bias"].shape[-1]
    moe_hidden = weights["expert_norms"].shape[-1]
    shard = NamedSharding(mesh, P("tp"))

    @jax.jit(out_shardings=shard)
    def split_columns(value):
        _, layers, rows, columns = value.shape
        return value[:8].reshape(8, layers, rows, 2, columns // 2).transpose(0, 3, 1, 2, 4).reshape(16, layers, rows, columns // 2)

    if weights["router"].shape[-1] == experts // 8:
        weights["router"] = split_columns(weights["router"])
    if weights["pool"].shape[-1] == moe_hidden // 8:
        weights["pool"] = split_columns(weights["pool"])

    @jax.jit(out_shardings=shard)
    def local_columns(value):
        _, layers, rows, columns = value.shape
        value = value.reshape(8, 2, layers, rows, columns).transpose(0, 2, 3, 1, 4)
        return jnp.tile(value.reshape(8, layers, rows, 2 * columns), (2, 1, 1, 1))

    @jax.jit(out_shardings=shard)
    def output_groups(value):
        _, layers, width, _ = value.shape
        value = value.reshape(2, 8, layers, width, 8, hidden // 8)
        return value.transpose(0, 4, 2, 1, 3, 5).reshape(16, layers, 8 * width, hidden // 8)

    @jax.jit(out_shardings=shard)
    def unpool_columns(value):
        _, layers, width, _ = value.shape
        value = value.transpose(1, 0, 2, 3).reshape(layers, 16 * width, 8, hidden // 8)
        return jnp.tile(value.transpose(2, 0, 1, 3), (2, 1, 1, 1))

    @jax.jit(out_shardings=(shard, shard))
    def router_order(value, bias):
        _, layers, rows, columns = value.shape
        order = jnp.arange(experts, dtype=jnp.int32)
        if experts in (128, 256):
            order = (127 - order % 128) * (experts // 128) + order // 128
        full = value.transpose(1, 2, 0, 3).reshape(layers, rows, experts)
        full = full[..., order].reshape(layers, rows, 8, experts // 8).transpose(2, 0, 3, 1)
        return jnp.tile(full, (2, 1, 1, 1)), bias[..., order]

    router, bias = router_order(weights["router"], weights["router_bias"])
    return jax.block_until_ready({**weights, "ow_group8": output_groups(weights["ow"]),
        "pool_local": local_columns(weights["pool"]), "unpool_group8": unpool_columns(weights["unpool"]),
        "router_ordered": router, "router_bias_ordered": bias})


def prepare_expert_parallel_weights(mesh, weights):
    """Prepare one TP8 x EP4 partition shared by prefill and decode."""
    if mesh.size != 32:
        raise ValueError("expert-parallel inference requires 32 devices")
    experts = weights["router_bias"].shape[-1]
    if "gu_pack" in weights and weights["gu_pack"].shape[2] == experts // 4:
        return weights
    weights = merge_expert_leaves(weights) if "gu_scales" in weights else dict(weights)
    sharding = NamedSharding(mesh, P("tp"))

    def owned(value, contraction):
        _, layers, count, rows, columns = value.shape
        value = value.reshape(8, 4, layers, count, rows, columns)
        if contraction:
            value = value.transpose(0, 2, 3, 1, 4, 5).reshape(8, layers, count, 4 * rows, columns)
        else:
            value = value.transpose(0, 2, 3, 4, 1, 5).reshape(8, layers, count, rows, 4 * columns)
        return jnp.concatenate([value[:, :, host::4] for host in range(4)])

    def gate_up(value):
        gate, up = jnp.split(value, 2, axis=-1)
        return jnp.concatenate((owned(gate, False), owned(up, False)), axis=-1)

    def down_packed(value):
        rows = value.shape[-2] * 8 // 9
        return jnp.concatenate((owned(value[..., :rows, :], True), owned(value[..., rows:, :], True)), axis=-2)

    def convert_layers(value, transform):
        shape = (32, 1, *value.shape[2:])
        result = jax.eval_shape(transform, jax.ShapeDtypeStruct(shape, value.dtype))
        output = jax.jit(lambda: jnp.zeros((32, value.shape[1], *result.shape[2:]), value.dtype), out_shardings=sharding)()

        @jax.jit(donate_argnums=(0,), out_shardings=sharding)
        def convert(output, source, index):
            start = (0, index, 0, 0, 0)
            return jax.lax.dynamic_update_slice(output, transform(jax.lax.dynamic_slice(source, start, shape)), start)

        for index in range(value.shape[1]):
            output = convert(output, value, jnp.int32(index)).block_until_ready()
        return output

    @jax.jit(out_shardings=sharding)
    def attention_output(value):
        _, layers, width, hidden = value.shape
        value = value.reshape(4, 8, layers, width, 8, hidden // 8)
        return value.transpose(0, 4, 2, 1, 3, 5).reshape(32, layers, 8 * width, hidden // 8)

    @jax.jit(out_shardings=sharding)
    def local_columns(value):
        _, layers, hidden, width = value.shape
        value = value.reshape(8, 4, layers, hidden, width).transpose(0, 2, 3, 1, 4)
        return jnp.tile(value.reshape(8, layers, hidden, 4 * width), (4, 1, 1, 1))

    @jax.jit(out_shardings=sharding)
    def unpool(value):
        _, layers, width, hidden = value.shape
        value = value.transpose(1, 0, 2, 3).reshape(layers, 32 * width, 8, hidden // 8)
        return jnp.tile(value.transpose(2, 0, 1, 3), (4, 1, 1, 1))

    @jax.jit(out_shardings=(sharding, sharding))
    def router(value, bias):
        _, layers, hidden, width = value.shape
        order = jnp.arange(experts)
        if experts in (128, 256):
            order = (127 - order % 128) * (experts // 128) + order // 128
        value = value.transpose(1, 2, 0, 3).reshape(layers, hidden, experts)[..., order]
        value = value.reshape(layers, hidden, 8, experts // 8).transpose(2, 0, 3, 1)
        return jnp.tile(value, (4, 1, 1, 1)), bias[..., order]

    @jax.jit(out_shardings=sharding)
    def embedding(value):
        _, rows, hidden = value.shape
        return jnp.tile(value.reshape(1, 32 * rows, hidden), (32, 1, 1))

    result = {key: weights[key] for key in ("scalars", "expert_norms", "gu_global", "dw_global", "lmw", "fnorm")}
    result["gu_pack"] = convert_layers(weights["gu_pack"], gate_up)
    result["dw_pack"] = convert_layers(weights["dw_pack"], down_packed)
    result["gu_dense"] = convert_layers(weights["gu_dense"], gate_up)
    result["dw_dense"] = convert_layers(weights["dw_dense"], lambda value: owned(value, True))
    result["qwkv"], result["ow"] = weights["qwkv"], attention_output(weights["ow"])
    result["pool"], result["unpool"] = local_columns(weights["pool"]), unpool(weights["unpool"])
    result["router"], result["router_bias"] = router(weights["router"], weights["router_bias"])
    result["emb"] = embedding(weights["emb"])
    return jax.block_until_ready(result)


def dequantize_experts_bf16(packed_leaf, scales_leaf, global_leaf):
    """Kernel-layout K-major packed experts [E, K/2, N] -> BF16 [E, K, N] (oracle layout).

    ``global_leaf`` is [E] (down) or [E, 2] (gate/up: one global per half of N).
    """
    packed_leaf = np.asarray(packed_leaf)
    scales_leaf = np.asarray(scales_leaf)
    global_leaf = np.asarray(global_leaf, dtype=np.float32)
    e, _, n = packed_leaf.shape
    globals_2d = global_leaf.reshape(e, -1)
    halves = globals_2d.shape[1]
    out = np.empty((e, packed_leaf.shape[1] * 2, n), ml_dtypes.bfloat16)
    for i in range(e):
        columns = []
        for hh in range(halves):
            lo, hi = hh * n // halves, (hh + 1) * n // halves
            decoded = dequantize_nvfp4(
                packed_leaf[i, :, lo:hi].T,  # [N/2, K/2]
                scales_leaf[i, :, lo:hi].T,  # [N/2, K/16]
                globals_2d[i, hh],
            )
            columns.append(decoded.T)  # [K, N/2]
        out[i] = np.concatenate(columns, axis=1).astype(ml_dtypes.bfloat16)
    return out
