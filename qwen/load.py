"""Qwen3.8 checkpoint loading, TP weight layouts, and device placement."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from safetensors import safe_open


def canonical_weights(c, seed=0):
    rng = np.random.default_rng(seed)

    def normal(*shape, scale=0.02):
        return (rng.standard_normal(shape, dtype=np.float32) * scale).astype(ml_dtypes.bfloat16)

    layers = []
    for index in range(c.layers):
        w = {
            "input_norm": normal(c.dim, scale=0.1),
            "post_norm": normal(c.dim, scale=0.1),
            "mlp": {
                "gate": normal(c.dim, c.intermediate),
                "up": normal(c.dim, c.intermediate),
                "down": normal(c.intermediate, c.dim),
            },
        }
        if index in c.full_attention:
            w["attention"] = {
                "q": normal(c.dim, c.heads * c.head_dim * 2),
                "k": normal(c.dim, c.kv_heads * c.head_dim),
                "v": normal(c.dim, c.kv_heads * c.head_dim),
                "o": normal(c.heads * c.head_dim, c.dim),
                "q_norm": normal(c.head_dim, scale=0.1),
                "k_norm": normal(c.head_dim, scale=0.1),
            }
        else:
            w["linear"] = {
                "qkv": normal(c.dim, c.conv_dim),
                "z": normal(c.dim, c.value_dim),
                "b": normal(c.dim, c.v_heads),
                "a": normal(c.dim, c.v_heads),
                "conv": normal(c.conv_dim, c.conv_size),
                "a_log": np.log(rng.uniform(0.01, 16.0, c.v_heads).astype(np.float32)).astype(
                    ml_dtypes.bfloat16
                ),
                "dt_bias": np.ones(c.v_heads, np.float32).astype(ml_dtypes.bfloat16),
                "norm": normal(c.v_dim, scale=0.1),
                "out": normal(c.value_dim, c.dim),
            }
        layers.append(w)
    return {
        "embedding": normal(c.vocab, c.dim),
        "layers": tuple(layers),
        "norm": normal(c.dim, scale=0.1),
        "lm_head": normal(c.dim, c.vocab),
    }


def canonical_states(c, context, seed=1):
    rng = np.random.default_rng(seed)
    states = []
    for index in range(c.layers):
        if index in c.full_attention:
            states.append(
                tuple(
                    rng.standard_normal((c.kv_heads, context, c.head_dim), dtype=np.float32)
                    .astype(ml_dtypes.bfloat16)
                    for _ in range(2)
                )
            )
        else:
            states.append(
                (
                    rng.standard_normal((c.conv_dim, c.conv_size), dtype=np.float32).astype(
                        ml_dtypes.bfloat16
                    ),
                    (
                        rng.standard_normal((c.v_heads, c.v_dim, c.k_dim), dtype=np.float32) * 0.01
                    ).astype(np.float32),
                )
            )
    return tuple(states)


def _linear_ordinal(c, index):
    return index - (index + 1) // 4


def _full_ordinal(c, index):
    return (index + 1) // 4 - 1


def rank_weights(weights, c, rank, tp=2):
    """One rank's kernel layout from canonical weights (head-aligned slices)."""
    vocab_pad = lm_pad(c, tp)
    v_half = c.vocab // tp
    out = {}
    out["emb"] = weights["embedding"][rank * v_half : (rank + 1) * v_half]
    lm = weights["lm_head"][:, rank * v_half : (rank + 1) * v_half]
    out["lmw"] = np.pad(lm, ((0, 0), (0, vocab_pad))).astype(ml_dtypes.bfloat16)

    qkv, outw, convw, alog, dtb, onorm = [], [], [], [], [], []
    qw, ow, qn, kn = [], [], [], []
    gu, dw, norm_in, norm_post = [], [], [], []
    q_heads = c.heads // tp
    if tp <= c.kv_heads:
        kvh = c.kv_heads // tp
        kv_src = rank * kvh
    else:  # each KV head replicated across tp // kv_heads adjacent ranks
        kvh = 1
        kv_src = rank // (tp // c.kv_heads)
    for index, w in enumerate(weights["layers"]):
        norm_in.append(w["input_norm"])
        norm_post.append(w["post_norm"])
        ms = rank * (c.intermediate // tp)
        gur = np.concatenate(
            (
                w["mlp"]["gate"][:, ms : ms + c.intermediate // tp],
                w["mlp"]["up"][:, ms : ms + c.intermediate // tp],
            ),
            axis=1,
        )
        gu.append(
            np.pad(gur, ((0, 0), (0, gu_width(c, tp) - gur.shape[1]))).astype(ml_dtypes.bfloat16)
        )
        dwr = w["mlp"]["down"][ms : ms + c.intermediate // tp]
        dw.append(np.pad(dwr, ((0, dw_k(c, tp) - dwr.shape[0]), (0, 0))).astype(ml_dtypes.bfloat16))
        if index in c.full_attention:
            hs = rank * q_heads * c.head_dim * 2
            ks = kv_src * c.head_dim
            qw.append(
                np.concatenate(
                    (
                        w["attention"]["q"][:, hs : hs + q_heads * c.head_dim * 2],
                        w["attention"]["k"][:, ks : ks + kvh * c.head_dim],
                        w["attention"]["v"][:, ks : ks + kvh * c.head_dim],
                    ),
                    axis=1,
                )
            )
            rs = rank * q_heads * c.head_dim
            ow.append(w["attention"]["o"][rs : rs + q_heads * c.head_dim])
            qn.append(w["attention"]["q_norm"])
            kn.append(w["attention"]["k_norm"])
        else:
            lin = w["linear"]
            qs = rank * (c.k_heads // tp) * c.k_dim
            vs = rank * (c.v_heads // tp) * c.v_dim
            hs = rank * (c.v_heads // tp)
            qkv.append(
                np.concatenate(
                    (
                        lin["qkv"][:, qs : qs + (c.k_heads // tp) * c.k_dim],
                        lin["qkv"][:, c.key_dim + qs : c.key_dim + qs + (c.k_heads // tp) * c.k_dim],
                        lin["qkv"][:, 2 * c.key_dim + vs : 2 * c.key_dim + vs + (c.v_heads // tp) * c.v_dim],
                        # zba rides along as extra output columns of the fused gemv
                        lin["z"][:, vs : vs + (c.v_heads // tp) * c.v_dim],
                        lin["b"][:, hs : hs + c.v_heads // tp],
                        lin["a"][:, hs : hs + c.v_heads // tp],
                        np.zeros((c.dim, zba_pad(c, tp)), np.float32),
                    ),
                    axis=1,
                ).astype(ml_dtypes.bfloat16)
            )
            convw.append(
                np.concatenate(
                    (
                        lin["conv"][qs : qs + (c.k_heads // tp) * c.k_dim],
                        lin["conv"][c.key_dim + qs : c.key_dim + qs + (c.k_heads // tp) * c.k_dim],
                        lin["conv"][2 * c.key_dim + vs : 2 * c.key_dim + vs + (c.v_heads // tp) * c.v_dim],
                    ),
                    axis=0,
                )
            )
            hs = rank * (c.v_heads // tp)
            alog.append(lin["a_log"][hs : hs + c.v_heads // tp])
            dtb.append(lin["dt_bias"][hs : hs + c.v_heads // tp])
            onorm.append(lin["norm"])
            outw.append(lin["out"][vs : vs + (c.v_heads // tp) * c.v_dim])
    nl = c.layers - len(c.full_attention)
    nf = len(c.full_attention)
    out["qkvz"] = np.stack(qkv)
    out["outw"] = np.stack(outw)
    out["convw"] = np.stack(convw)
    out["alog"] = np.stack(alog)
    out["dtb"] = np.stack(dtb)
    out["onorm"] = np.stack(onorm)
    out["qwkv"] = np.stack(qw)
    out["ow"] = np.stack(ow)
    out["qn"] = np.stack(qn)
    out["kn"] = np.stack(kn)
    out["gu"] = np.stack(gu)
    out["dw"] = np.stack(dw)
    out["norm_in"] = np.stack(norm_in)
    out["norm_post"] = np.stack(norm_post)
    out["fnorm"] = weights["norm"]
    assert out["qkvz"].shape[0] == nl and out["qwkv"].shape[0] == nf
    return out


def zba_pad(c, tp=2):
    import os

    override = os.environ.get("QWEN_ZBA_PAD")
    raw = (c.v_heads // tp) * c.v_dim + 2 * (c.v_heads // tp)
    if override:
        return int(override) - raw
    return -raw % 640


def zba_width(c, tp=2):
    return (c.v_heads // tp) * c.v_dim + 2 * (c.v_heads // tp) + zba_pad(c, tp)


def until_matrix(t, k, n):
    """Inverse of tile_matrix: [..., T, bk, bn] -> [..., K, N]."""
    *lead, T, bk, bn = t.shape
    nk, nb = k // bk, n // bn
    L = len(lead)
    order = list(range(L)) + [L + 1, L + 2, L + 0, L + 3]
    w = t.reshape(*lead, nb, nk, bk, bn).transpose(order)
    return w.reshape(*lead, nk * bk, nb * bn)


def schedule(c, tp, lm_width):
    """The kernel tile schedule: (name, index_mode, bk, bn, k-tiles, n-tiles, tiles)."""
    h = c.dim
    kl = (c.k_heads // tp) * c.k_dim
    vl = (c.v_heads // tp) * c.v_dim
    qh = (c.heads // tp) * c.head_dim
    kvd = max(1, c.kv_heads // tp) * c.head_dim
    inter = c.intermediate // tp

    def slot(k, n):
        bk, bn = tile_hw(k, n)
        return bk, bn, k // bk, n // bn, (k // bk) * (n // bn)

    qkvz = kl * 2 + vl + zba_width(c, tp)
    lin = [
        ("qkvz", "lin", *slot(h, qkvz)),
        ("outw", "lin", *slot(vl, h)),
        ("gu", "all", *slot(h, gu_width(c, tp))),
        ("dw", "all", *slot(dw_k(c, tp), h)),
    ]
    qwkv = qh * 2 + 2 * kvd
    full = [
        ("qwkv", "full", *slot(h, qwkv)),
        ("ow", "full", *slot(qh, h)),
        ("gu", "all", *slot(h, gu_width(c, tp))),
        ("dw", "all", *slot(dw_k(c, tp), h)),
    ]

    def offsets(slots):
        out, total = [], 0
        for s_ in slots:
            out.append(total)
            total += s_[-1]
        return out, total

    off_lin, t_lin = offsets(lin)
    off_full, t_full = offsets(full)
    lm = ("lmw", "lm", *slot(h, lm_width))
    return lin, full, lm, off_lin, off_full, t_lin, t_full, 3 * t_lin + t_full


def tile_hw(k, n, max_k=None, max_n=None):
    """(bk, bn) tile for a [k, n] matrix: largest 128-multiples within caps."""
    import os

    max_k = max_k or int(os.environ.get("QWEN_MAXK", "1024"))
    max_n = max_n or int(os.environ.get("QWEN_MAXN", "1280"))
    bk = max(b for b in range(128, min(k, max_k) + 1, 128) if k % b == 0)
    bn = max(b for b in range(128, min(n, max_n) + 1, 128) if n % b == 0)
    return bk, bn


def tile_matrix(w, bk, bn):
    """[..., K, N] -> [..., T, bk, bn] with each tile contiguous (ti = ni*nk + ki)."""
    *lead, K, N = w.shape
    nk, nb = K // bk, N // bn
    L = len(lead)
    order = list(range(L)) + [L + 2, L + 0, L + 1, L + 3]
    tiled = w.reshape(*lead, nk, bk, nb, bn).transpose(order)
    return np.ascontiguousarray(tiled.reshape(*lead, nb * nk, bk, bn))


def gu_width(c, tp=2):
    """Per-rank [gate|up] width, padded only when the pad fraction is small."""
    import os

    override = os.environ.get("QWEN_GU_PAD")
    if override:
        return int(override)
    w = 2 * (c.intermediate // tp)
    if w % 1024 == 0:
        return w
    padded = w + (-w % 1280)
    if padded <= 1.06 * w:
        return padded
    padded = w + (-w % 512)
    return padded if padded <= 1.06 * w else w


def dw_k(c, tp=2):
    """Per-rank down-proj K, padded to a 512- or 256-multiple tile depth."""
    import os

    override = os.environ.get("QWEN_DW_PAD")
    if override:
        return int(override)
    k = c.intermediate // tp
    padded = k + (-k % 512)
    return padded if padded <= 1.06 * k else k + (-k % 256)


def lm_pad(c, tp=2):
    return -(c.vocab // tp) % 1280


def lm_width(c, tp=2):
    return c.vocab // tp + lm_pad(c, tp)


def pack(weights, c, tp=2, tiled=True):
    """All ranks stacked along a leading tp axis, schedule families tile-blocked."""
    ranks = [rank_weights(weights, c, r, tp) for r in range(tp)]
    packed = {k: np.stack([r[k] for r in ranks]) for k in ranks[0]}
    if not tiled:
        return packed
    lin, full, lm, *_ = schedule(c, tp, lm_width(c, tp))
    tiles = {}
    for name, _, bk, bn, nk, nb, nt in lin + full + [lm]:
        tiles[name] = (bk, bn)
    for name, (bk, bn) in tiles.items():
        packed[name] = tile_matrix(packed[name], bk, bn)
    return packed


def pack_states(states, c, tp=2):
    """Per-rank family arrays: conv/rec [nl], kcache/vcache [nf] (head split)."""
    conv, rec, keys, values = [], [], [], []
    for index, state in enumerate(states):
        if index in c.full_attention:
            keys.append(state[0])
            values.append(state[1])
        else:
            conv.append(state[0])
            rec.append(state[1])
    conv = np.stack(conv)
    rec = np.stack(rec)
    keys = np.stack(keys)
    values = np.stack(values)
    packed = {"conv": [], "rec": [], "kcache": [], "vcache": []}
    for rank in range(tp):
        qs = rank * (c.k_heads // tp) * c.k_dim
        vs = rank * (c.v_heads // tp) * c.v_dim
        packed["conv"].append(
            np.concatenate(
                (
                    conv[:, qs : qs + (c.k_heads // tp) * c.k_dim],
                    conv[:, c.key_dim + qs : c.key_dim + qs + (c.k_heads // tp) * c.k_dim],
                    conv[:, 2 * c.key_dim + vs : 2 * c.key_dim + vs + (c.v_heads // tp) * c.v_dim],
                ),
                axis=1,
            )
        )
        hs = rank * (c.v_heads // tp)
        packed["rec"].append(rec[:, hs : hs + c.v_heads // tp])
        if tp <= c.kv_heads:
            kvh, kv_src = c.kv_heads // tp, rank
        else:
            kvh, kv_src = 1, rank // (tp // c.kv_heads)
        ks = kv_src * kvh
        packed["kcache"].append(keys[:, ks : ks + kvh])
        packed["vcache"].append(values[:, ks : ks + kvh])
    return {k: np.stack(v) for k, v in packed.items()}


def zero_states(c, context):
    """Fresh decode states in canonical layout (conv/rec/KV all zero)."""
    states = []
    for index in range(c.layers):
        if index in c.full_attention:
            states.append(
                (
                    np.zeros((c.kv_heads, context, c.head_dim), ml_dtypes.bfloat16),
                    np.zeros((c.kv_heads, context, c.head_dim), ml_dtypes.bfloat16),
                )
            )
        else:
            states.append(
                (
                    np.zeros((c.conv_dim, c.conv_size), ml_dtypes.bfloat16),
                    np.zeros((c.v_heads, c.v_dim, c.k_dim), np.float32),
                )
            )
    return tuple(states)


def _flatten(tree, prefix=""):
    for key, value in tree.items():
        if isinstance(value, dict):
            yield from _flatten(value, prefix + key + "/")
        else:
            yield prefix + key, value


def _unflatten(items):
    out = {}
    for key, value in items.items():
        node = out
        *path, leaf = key.split("/")
        for part in path:
            node = node.setdefault(part, {})
        node[leaf] = value
    return out


class Checkpoint:
    def __init__(self, path):
        path = Path(path)
        index = json.loads((path / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self.handles = {}
        self.root = path

    def get(self, name):
        shard = self.weight_map[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(
                str(self.root / shard), framework="numpy"
            )
        return self.handles[shard].get_tensor(name)

    def has(self, name):
        return name in self.weight_map


def load_canonical(path, c):
    """HF checkpoint -> canonical layout ([in, out] matrices, BF16)."""
    ckpt = Checkpoint(path)
    layers = []
    for index in range(c.layers):
        p = f"model.language_model.layers.{index}."
        w = {
            "input_norm": ckpt.get(p + "input_layernorm.weight"),
            "post_norm": ckpt.get(p + "post_attention_layernorm.weight"),
            "mlp": {
                n: ckpt.get(p + f"mlp.{n}_proj.weight").T.copy()
                for n in ("gate", "up", "down")
            },
        }
        if index in c.full_attention:
            a = f"{p}self_attn."
            w["attention"] = {
                "q": ckpt.get(a + "q_proj.weight").T.copy(),
                "k": ckpt.get(a + "k_proj.weight").T.copy(),
                "v": ckpt.get(a + "v_proj.weight").T.copy(),
                "o": ckpt.get(a + "o_proj.weight").T.copy(),
                "q_norm": ckpt.get(a + "q_norm.weight"),
                "k_norm": ckpt.get(a + "k_norm.weight"),
            }
        else:
            a = f"{p}linear_attn."
            w["linear"] = {
                "qkv": ckpt.get(a + "in_proj_qkv.weight").T.copy(),
                "z": ckpt.get(a + "in_proj_z.weight").T.copy(),
                "b": ckpt.get(a + "in_proj_b.weight").T.copy(),
                "a": ckpt.get(a + "in_proj_a.weight").T.copy(),
                "conv": ckpt.get(a + "conv1d.weight").squeeze(1),
                "a_log": ckpt.get(a + "A_log"),
                "dt_bias": ckpt.get(a + "dt_bias"),
                "norm": ckpt.get(a + "norm.weight"),
                "out": ckpt.get(a + "out_proj.weight").T.copy(),
            }
        layers.append(w)
        if (index + 1) % 8 == 0:
            print(f"  loaded {index + 1}/{c.layers} layers", flush=True)
    return {
        "embedding": ckpt.get("model.language_model.embed_tokens.weight"),
        "layers": tuple(layers),
        "norm": ckpt.get("model.language_model.norm.weight"),
        "lm_head": ckpt.get("lm_head.weight").T.copy(),
    }


def save_pack(directory, tree):
    """Write ``tree`` (a pytree of host arrays) as ``directory/pack.bin`` + manifest."""
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
        for leaf in leaves:
            a = np.asarray(leaf)
            dtype = str(a.dtype)
            if a.dtype == ml_dtypes.bfloat16:
                a = a.view(np.uint16)  # numpy has no bf16; store the raw bits
                dtype = "bf16"
            raw = a.tobytes(order="C")
            bf.write(raw)
            manifest["leaves"].append(
                {"offset": offset, "nbytes": len(raw), "dtype": dtype, "shape": list(a.shape)}
            )
            offset += len(raw)
    (tmp / "manifest.json").write_text(json.dumps(manifest))
    tmp.rename(directory)


def load_pack(directory):
    """Read a ``save_pack`` container back into a host pytree."""
    import pickle

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    treedef = pickle.loads(bytes.fromhex(manifest["treedef"]))
    raw = np.fromfile(directory / "pack.bin", dtype=np.uint8)  # one sequential read
    leaves = []
    for leaf in manifest["leaves"]:
        a = np.frombuffer(
            raw[leaf["offset"] : leaf["offset"] + leaf["nbytes"]],
            dtype=np.uint16 if leaf["dtype"] == "bf16" else leaf["dtype"],
        ).reshape(leaf["shape"])
        if leaf["dtype"] == "bf16":
            a = a.view(ml_dtypes.bfloat16)
        leaves.append(a)
    return jax.tree_util.tree_unflatten(treedef, leaves)


def load_target(checkpoint, c, tp, packed=None):
    """(untiled, tiled) packed per-rank weights; HF parse or a packed directory.

    ``packed`` may name a directory holding the ``untiled/`` and ``tiled/``
    containers to skip the safetensors parse. States are zeros either way.
    """
    if packed is not None and (Path(packed) / "untiled" / "manifest.json").exists():
        untiled = load_pack(Path(packed) / "untiled")
        tiled = load_pack(Path(packed) / "tiled")
    else:
        canonical = load_canonical(checkpoint, c)
        untiled = pack(canonical, c, tp, tiled=False)
        tiled = pack(canonical, c, tp, tiled=True)
        del canonical
    return untiled, tiled


def load_tokenizer(checkpoint):
    """The checkpoint's tokenizer.json as a minimal encode/decode adapter.

    ``encode`` never adds special tokens implicitly (the chat renderer adds
    them explicitly); special-token strings in the text are still recognized.
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


def build_sharded(mesh, packed):
    spec = NamedSharding(mesh, P("tp"))

    def put(a):
        return jax.device_put(jnp.asarray(a), spec)

    return jax.tree.map(put, packed)
