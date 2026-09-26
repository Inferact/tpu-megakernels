"""Muse Spark 1.2 checkpoint streaming, per-rank conversion and device placement.

Three stages, each usable on its own:

1. `Checkpoint` streams tensors of the Hugging Face safetensors checkpoint with large
   parallel `os.preadv` reads (the shards live on NFS; one file streams at ~1 GB/s with
   8-32 threads, a single thread only ~0.3 GB/s).
2. `convert_presharded` turns the checkpoint into the per-rank kernel layout of
   `musespark.layout.rank_shapes` (design.md section 2): dense bf16 `[in, out]` slices,
   effective norm weights, residual-gate coefficients, the bf16 hi/lo split of the f32
   router and int4 group-quantized expert slices with f32 (bf16-valued) group scales in
   the kernel's chunked `[K/KC, KC/G, N]` layout. The result is an on-disk container:

       <dst>/layout.json        format, config, tp, group size, checkpoint revision and, per
                                array name, the per-rank shape/dtype and the on-disk
                                (packed) shape/dtype/byte size
       <dst>/progress.json      layers written so far (conversion is resumable/idempotent)
       <dst>/rank{r}/<name>.bin raw little-endian C-order per-rank array; layer-stacked
                                arrays keep the layer axis outermost so layer `l` is the
                                byte range [l*S, (l+1)*S); int4 arrays are stored as uint8
                                with two int4 values per byte along the LAST axis, low
                                nibble first (`quant.pack_int4`)

   The conversion streams the checkpoint layer by layer (two layers of expert tensors in
   flight in shared memory: one being read, one being quantized) and quantizes on a
   process pool so the NFS read bandwidth, not numpy, is the bottleneck.
3. `load_presharded` places every rank's arrays on its own device (`jax.device_put` +
   `jax.make_array_from_single_device_arrays`, the Kimi idiom) and unpacks the int4 nibbles
   on the device, so host RAM only ever sees one layer of packed bytes per rank.

Container format v2 (`FORMAT_V2`, `expert_format == "nvfp4"`, `convert_presharded_nvfp4`): the
same dense/vector/global families, but the experts are the vendor's NVFP4 tensors re-laid out
per rank without any float maths (`quant.nvfp4_rows_to_packed`): `gate_up_fp4 [L, E, Hm/8, 2*Is]`
int32 (eight e2m1 codes per word along K), `gate_up_bs [L, E, Hm/KC, KC/16, 2*Is]` e4m3 block
scales (stored as their uint8 bits), `down_fp4 [L, E, Is/8, Hm]`, `down_bs [L, E, Is/KC, KC/16,
Hm]` and `expert_gs [L, E, 8, 128]` f32 (row 0 gate, row 1 up, row 2 down global scale,
replicated over the lanes). Layers whose experts the vendor left in bf16 (0 and 61) are quantized
to the SAME format by `quant.quantize_nvfp4_np` (per-half global amax, modelopt rounding) so the
kernel has one expert format. The source checkpoint (533 GB) is streamed shard by shard from
the Hub (`ShardSource`): a shard is downloaded, every text-model tensor it holds is converted
into the container, and the shard is deleted, keeping at most a few shards on disk; progress is
per conversion unit (`progress.json`) and resumable.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import multiprocessing
from multiprocessing import shared_memory
import os
from pathlib import Path
import queue
import struct
import sys
import threading
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P, SingleDeviceSharding

from . import (
    Config,
    canonical_global_from_checkpoint,
    canonical_layer_from_checkpoint,
    eff_weight,
    gate_coeffs,
    layout,
    quant,
    split_hi_lo,
)


BF16 = np.dtype(ml_dtypes.bfloat16)
INT4 = np.dtype(ml_dtypes.int4)
E4M3 = np.dtype(ml_dtypes.float8_e4m3fn)
PREFIX = "model.language_model."
FORMAT = "musespark-presharded-v1"
FORMAT_V2 = "musespark-presharded-v2"
FORMATS = (FORMAT, FORMAT_V2)
INT4_ENCODING = (
    "uint8, two int4 values per byte along the last axis, low nibble = even index; "
    "value = ((byte >> (4 * (i % 2))) & 0xF) sign-extended from 4 bits"
)
FP4_ENCODING = (
    "int32 [K/8, N]: e2m1 code of row 8k'+j in bits 4j..4j+3 (low nibble first; "
    "pltpu.bitcast(int32 -> float4_e2m1fn) order); block scales float8_e4m3fn stored as "
    "uint8 bits, one per 16 rows, K-chunk major [K/KC, KC/16, N]; expert_gs [8, 128] f32 rows "
    "0/1/2 = gate/up/down per-expert global scale; w = e2m1 * e4m3 * gs"
)
NVFP4_REPO = "meta-models/Muse-Spark-1.2-816B-A42B-NVFP4-open"

# Names of the per-layer checkpoint tensors (suffixes of `PREFIX + f"layers.{l}."`).
LAYER_SUFFIXES = (
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.gate_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_residual_gate.gate",
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
    "post_feedforward_residual_gate.gate",
    "mlp.gate.weight",
    "mlp.gate.e_score_correction_bias",
    "mlp.pre_expert_proj.weight",
    "mlp.pre_expert_norm.weight",
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
    "mlp.experts.post_expert_norm.weight",
    "mlp.post_expert_proj.weight",
)
EXPERT_SUFFIXES = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")
DENSE_SUFFIXES = tuple(s for s in LAYER_SUFFIXES if s not in EXPERT_SUFFIXES)
# NVFP4 checkpoint expert tensors (`*_input_scale` exist too and are ignored: bf16 activations).
NVFP4_SUFFIXES = {
    "gate_up_fp4": "mlp.experts.gate_up_proj",
    "gate_up_bs": "mlp.experts.gate_up_proj_weight_scale",
    "gate_up_gs": "mlp.experts.gate_up_proj_weight_scale_2",
    "down_fp4": "mlp.experts.down_proj",
    "down_bs": "mlp.experts.down_proj_weight_scale",
    "down_gs": "mlp.experts.down_proj_weight_scale_2",
}
GLOBAL_KEYS = {
    "embed": PREFIX + "embed_tokens.weight",
    "lm_head": "lm_head.weight",
    "final_norm": PREFIX + "norm.weight",
}
INT8_ENCODING = (
    "int8 per output column: w[k, n] ~= q[k, n] * s[0, n] with s = absmax_k |w| / 127 (f32; "
    "all-zero columns s = 1) and q = round-half-even(w / s) in [-127, 127]; o_s is the absmax "
    "over the FULL o_proj contraction axis (all ranks hold the same o_s)"
)
EXPERT_NAMES = ("gate_up_q", "gate_up_s", "down_q", "down_s")
FP4_EXPERT_NAMES = layout.FP4_EXPERT_FAMILIES
GLOBAL_NAMES = ("embed", "lm_head", "final_norm")
# Dense per-layer families grouped into pool tasks (each task writes all ranks); the small
# vector families use the reference's jax helpers and are converted in the main process.
DENSE_TASKS = (
    ("q",),
    ("kv",),
    ("gate",),
    ("o",),
    ("pre",),
    ("post",),
    ("router_hi", "router_lo"),
)
VECTOR_NAMES = (
    "attn_norm", "ffn_norm", "post_ffn_norm", "attn_gate_alpha", "attn_gate_beta",
    "ffn_gate_alpha", "ffn_gate_beta", "pre_expert_norm", "post_expert_norm", "router_bias",
)

READ_CHUNK = 32 << 20
READ_THREADS = 32


def layer_key(layer, suffix):
    return f"{PREFIX}layers.{layer}.{suffix}"


# ---------------------------------------------------------------------------------------
# Checkpoint streaming
# ---------------------------------------------------------------------------------------


def _read_ranges(jobs, threads=READ_THREADS, chunk=READ_CHUNK):
    """Fill each `(path, offset, memoryview)` job with the file bytes, chunked over threads.

    Returns the number of bytes read. File descriptors are shared between threads (pread
    is positional), so a job costs one `open` no matter how many chunks it has.
    """
    fds = {}
    pieces = []
    for path, offset, view in jobs:
        path = os.fspath(path)
        if path not in fds:
            fds[path] = os.open(path, os.O_RDONLY)
        for start in range(0, len(view), chunk):
            pieces.append((fds[path], offset + start, view[start:start + chunk]))

    def run(piece):
        fd, offset, view = piece
        done = 0
        while done < len(view):
            got = os.preadv(fd, [view[done:]], offset + done)
            if got <= 0:
                raise EOFError(f"short read at offset {offset + done}")
            done += got
        return done

    try:
        if len(pieces) <= 1:
            total = sum(run(piece) for piece in pieces)
        else:
            with ThreadPoolExecutor(min(threads, len(pieces))) as pool:
                total = sum(pool.map(run, pieces))
    finally:
        for fd in fds.values():
            os.close(fd)
    return total


class Checkpoint:
    """Safetensors index + header parsing and streaming reads of one HF checkpoint."""

    DTYPES = {"BF16": BF16, "F32": np.dtype(np.float32), "F16": np.dtype(np.float16),
              "U8": np.dtype(np.uint8), "I8": np.dtype(np.int8), "F8_E4M3": E4M3}
    NAMES = {v: k for k, v in DTYPES.items()}

    def __init__(self, path, config=None):
        self.path = Path(path)
        if config is None:
            raw = json.loads((self.path / "config.json").read_text())
            vocab = raw.get("text_config", raw)["vocab_size"]
            config = Config.from_checkpoint(self.path, vocab_used=min(Config().vocab_used, vocab))
        self.config = config
        index_path = self.path / "model.safetensors.index.json"
        self.index = json.loads(index_path.read_text())["weight_map"]
        missing = sorted(f for f in set(self.index.values()) if not (self.path / f).is_file())
        if missing:
            raise FileNotFoundError(f"{self.path}: missing shards {missing[:3]}...")
        self.revision = hashlib.sha256(index_path.read_bytes()).hexdigest()[:16]
        self._layouts = {}
        self._lock = threading.Lock()
        self.bytes_read = 0
        self.read_seconds = 0.0

    def _layout(self, filename):
        """(header dict, data start) of one shard, cached."""
        with self._lock:
            if filename not in self._layouts:
                with open(self.path / filename, "rb") as f:
                    size = struct.unpack("<Q", f.read(8))[0]
                    self._layouts[filename] = (json.loads(f.read(size)), 8 + size)
            return self._layouts[filename]

    def meta(self, key):
        """`(path, byte offset, nbytes, np dtype, shape)` of one tensor."""
        filename = self.index[key]
        header, base = self._layout(filename)
        entry = header[key]
        start, end = entry["data_offsets"]
        return (self.path / filename, base + start, end - start, self.DTYPES[entry["dtype"]],
                tuple(entry["shape"]))

    def read_into(self, key, out, rows=None, threads=READ_THREADS):
        """Stream `key` (or its leading-axis row slice) into the byte buffer `out`."""
        path, offset, nbytes, dtype, shape = self.meta(key)
        if rows is not None:
            r0, r1, step = rows.indices(shape[0])
            if step != 1:
                raise ValueError("only unit-step leading slices are supported")
            row_bytes = nbytes // shape[0]
            offset, nbytes = offset + r0 * row_bytes, (r1 - r0) * row_bytes
        view = memoryview(out).cast("B")
        if len(view) != nbytes:
            raise ValueError(f"{key}: buffer holds {len(view)} bytes, tensor has {nbytes}")
        started = time.perf_counter()
        _read_ranges([(path, offset, view)], threads)
        with self._lock:
            self.bytes_read += nbytes
            self.read_seconds += time.perf_counter() - started

    def read(self, key, rows=None, threads=READ_THREADS):
        """One tensor, or `tensor[rows]` for a leading-axis slice, as a numpy array."""
        _, _, nbytes, dtype, shape = self.meta(key)
        shape = list(shape)
        if rows is not None:
            r0, r1, _ = rows.indices(shape[0])
            shape[0] = r1 - r0
        out = np.empty(shape, dtype)
        if out.nbytes:
            self.read_into(key, out.view(np.uint8).reshape(-1), rows, threads)
        return out

    def read_many(self, keys, threads=READ_THREADS):
        """`{key: array}` for several tensors, all chunks in flight on one thread pool."""
        arrays, jobs = {}, []
        for key in keys:
            path, offset, nbytes, dtype, shape = self.meta(key)
            arrays[key] = np.empty(shape, dtype)
            if nbytes:
                jobs.append((path, offset, memoryview(arrays[key].view(np.uint8).reshape(-1))))
        started = time.perf_counter()
        total = _read_ranges(jobs, threads)
        with self._lock:
            self.bytes_read += total
            self.read_seconds += time.perf_counter() - started
        return arrays

    def layer_keys(self, layer):
        return [layer_key(layer, s) for s in LAYER_SUFFIXES]

    def read_layer(self, layer, threads=READ_THREADS):
        """All tensors of one layer keyed by suffix (24 GiB for the real model)."""
        arrays = self.read_many(self.layer_keys(layer), threads)
        return {s: arrays[layer_key(layer, s)] for s in LAYER_SUFFIXES}

    def gbps(self):
        return self.bytes_read / max(self.read_seconds, 1e-9) / 1e9


# ---------------------------------------------------------------------------------------
# Per-rank conversion (numpy; bit-identical to `musespark.shard_canonical`)
# ---------------------------------------------------------------------------------------


def _bf16(x):
    return np.asarray(x).astype(BF16)


def _f32(x):
    return np.asarray(x).astype(np.float32)


def effective_norm(w):
    """`r16(1 + w)` (zero-centered gamma, `musespark.eff_weight`) as bf16 `[1, n]`."""
    return np.asarray(eff_weight(jnp.asarray(_bf16(w))).astype(jnp.bfloat16)).reshape(1, -1)


def gate_arrays(g, temperature):
    """`(alpha, beta)` f32 `[1, n]` of a residual gate parameter (`musespark.gate_coeffs`)."""
    alpha, beta = gate_coeffs(jnp.asarray(_bf16(g)), temperature)
    return _f32(alpha).reshape(1, -1), _f32(beta).reshape(1, -1)


def router_split(w):
    """f32 `[E, H]` router weight -> `(hi, lo)` bf16 `[H, E]` with `hi + lo ~= W.T`."""
    return split_hi_lo(np.ascontiguousarray(_f32(w).T))


def _rows_t(w, rank, width):
    """`[out, in]` checkpoint matrix -> this rank's `[in, width]` output slice."""
    return np.ascontiguousarray(_bf16(w)[rank * width:(rank + 1) * width].T)


def dense_rank_arrays(cfg, tp, rank, t, names=None):
    """One rank's dense arrays (`names`, default all) of one layer from the layer's
    checkpoint tensors `t` (keyed by `LAYER_SUFFIXES`; expert tensors are not needed)."""
    qw, kw = (cfg.heads // tp) * cfg.head_dim, (cfg.kv_heads // tp) * cfg.head_dim
    hm_r, h_r = cfg.moe_hidden // tp, cfg.hidden // tp

    def gate(suffix, which):
        return gate_arrays(t[suffix], cfg.gate_temperature)[which]

    makers = {
        "q": lambda: _rows_t(t["self_attn.q_proj.weight"], rank, qw),
        "kv": lambda: np.concatenate([_rows_t(t["self_attn.k_proj.weight"], rank, kw),
                                      _rows_t(t["self_attn.v_proj.weight"], rank, kw)], axis=1),
        "gate": lambda: _rows_t(t["self_attn.gate_proj.weight"], rank, qw),
        "o": lambda: np.ascontiguousarray(
            _bf16(t["self_attn.o_proj.weight"])[:, rank * qw:(rank + 1) * qw].T),
        "pre": lambda: _rows_t(t["mlp.pre_expert_proj.weight"], rank, hm_r),
        "post": lambda: _rows_t(t["mlp.post_expert_proj.weight"], rank, h_r),
        "router_hi": lambda: router_split(t["mlp.gate.weight"])[0],
        "router_lo": lambda: router_split(t["mlp.gate.weight"])[1],
        "attn_norm": lambda: effective_norm(t["input_layernorm.weight"]),
        "ffn_norm": lambda: effective_norm(t["pre_feedforward_layernorm.weight"]),
        "post_ffn_norm": lambda: effective_norm(t["post_feedforward_layernorm.weight"]),
        "pre_expert_norm": lambda: effective_norm(t["mlp.pre_expert_norm.weight"]),
        "post_expert_norm": lambda: effective_norm(t["mlp.experts.post_expert_norm.weight"]),
        "attn_gate_alpha": lambda: gate("post_attention_residual_gate.gate", 0),
        "attn_gate_beta": lambda: gate("post_attention_residual_gate.gate", 1),
        "ffn_gate_alpha": lambda: gate("post_feedforward_residual_gate.gate", 0),
        "ffn_gate_beta": lambda: gate("post_feedforward_residual_gate.gate", 1),
        "router_bias": lambda: _f32(t["mlp.gate.e_score_correction_bias"]).reshape(1, -1),
    }
    return {name: makers[name]() for name in (names or makers)}


def expert_rank_arrays(cfg, tp, rank, gate_up, down, packed=True):
    """One rank's int4 slices of ONE expert: `gate_up [2I, Hm]`, `down [Hm, I]` (checkpoint
    layout, `[out, in]`) -> `gate_up_q [Hm, 2*Is]`, `gate_up_s [Hm/KC, KC/G, 2*Is]` f32,
    `down_q [Is, Hm]`, `down_s [Is/KC, KC/G, Hm]` f32 (chunked scales, `quant.k_chunk`).
    With `packed` the int4 values are nibble-packed uint8 (the on-disk form)."""
    i_s, i_full, group = cfg.expert_hidden // tp, cfg.expert_hidden, cfg.group_size
    gu = _bf16(gate_up)
    w = np.concatenate(
        [gu[rank * i_s:(rank + 1) * i_s].T, gu[i_full + rank * i_s:i_full + (rank + 1) * i_s].T],
        axis=1,
    )
    gq, gs = quant.quantize_int4(w, group)
    dq, ds = quant.quantize_int4(_bf16(down)[:, rank * i_s:(rank + 1) * i_s].T, group)
    if packed:
        gq, dq = quant.pack_int4(gq), quant.pack_int4(dq)
    return {
        "gate_up_q": gq,
        "gate_up_s": quant.scales_to_chunked(gs, cfg.moe_hidden),
        "down_q": dq,
        "down_s": quant.scales_to_chunked(ds, i_s),
    }


def global_rank_arrays(cfg, tp, rank, embed_rows, lm_head_rows, final_norm):
    """`embed`, `lm_head`, `final_norm` of one rank from this rank's rows of the vocab
    tables (`embed_rows`/`lm_head_rows` = rows `r*Vp:(r+1)*Vp`, possibly short at the end)."""
    vp = layout.vocab_pad(cfg, tp)
    embed = np.zeros((vp, cfg.hidden), BF16)
    embed[:embed_rows.shape[0]] = _bf16(embed_rows)
    lm = np.zeros((vp, cfg.hidden), BF16)
    lm[:lm_head_rows.shape[0]] = _bf16(lm_head_rows)
    return {"embed": embed, "lm_head": np.ascontiguousarray(lm.T),
            "final_norm": _bf16(final_norm).reshape(1, -1)}


# ---------------------------------------------------------------------------------------
# On-disk container
# ---------------------------------------------------------------------------------------


def _shapes(cfg, tp, expert_format="int4", dense_format="bf16"):
    """`{name: (per-rank shape, np dtype)}` from `layout.rank_shapes`."""
    out = {}
    for name, (shape, dtype) in layout.rank_shapes(cfg, tp, expert_format, dense_format).items():
        out[name] = (tuple(int(s) for s in shape), np.dtype(dtype))
    return out


def disk_spec(shape, dtype):
    """`(disk shape, disk dtype)`: int4 is nibble-packed along the last axis, e4m3 block
    scales are stored as their uint8 bits."""
    if np.dtype(dtype) == INT4:
        if shape[-1] % 2:
            raise ValueError("int4 arrays need an even last axis to pack")
        return (*shape[:-1], shape[-1] // 2), np.dtype(np.uint8)
    if np.dtype(dtype) == E4M3:
        return tuple(shape), np.dtype(np.uint8)
    return tuple(shape), np.dtype(dtype)


def _dtype_name(dtype):
    dtype = np.dtype(dtype)
    return {BF16: "bfloat16", INT4: "int4", E4M3: "float8_e4m3fn"}.get(dtype, dtype.name)


def _np_dtype(name):
    return {"bfloat16": BF16, "int4": INT4, "float8_e4m3fn": E4M3}.get(name, np.dtype(name))


def make_layout(cfg, tp, revision=None, expert_format="int4", expert_source=None,
                dense_formats=("bf16",)):
    """The `layout.json` document of a container for `cfg`/`tp` (`expert_format` "int4" ->
    format v1, "nvfp4" -> format v2; `expert_source` documents, per layer, where the nvfp4
    experts came from: "vendor" or "rtn" (quantized from bf16 by `quant.quantize_nvfp4_np`)).
    `dense_formats` lists the dense-projection formats present: `("bf16",)` (the conversion
    output) or `("bf16", "int8")` after `quantize_dense`, which appends the int8 pairs to
    `arrays` and records `dense_format: int8` as the preferred format."""
    arrays = {}
    dense = "both" if "int8" in dense_formats else "bf16"
    for name, (shape, dtype) in _shapes(cfg, tp, expert_format, dense).items():
        dshape, ddtype = disk_spec(shape, dtype)
        arrays[name] = {
            "shape": list(shape),
            "dtype": _dtype_name(dtype),
            "file": f"{name}.bin",
            "disk_shape": list(dshape),
            "disk_dtype": ddtype.name,
            "nbytes": int(np.prod(dshape)) * ddtype.itemsize,
            "layer_axis": 0 if name not in GLOBAL_NAMES else None,
        }
    cfg_dict = {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()}
    doc = {
        "format": FORMAT if expert_format == "int4" else FORMAT_V2,
        "config": cfg_dict,
        "tp": tp,
        "group": cfg.group_size,
        "revision": revision,
        "int4": INT4_ENCODING,
        "ranks": [f"rank{r}" for r in range(tp)],
        "arrays": arrays,
        "total_bytes": tp * sum(a["nbytes"] for a in arrays.values()),
    }
    if expert_format != "int4":
        doc["expert_format"] = expert_format
        doc["fp4"] = FP4_ENCODING
        doc["fp4_block"] = layout.FP4_BLOCK
        doc["expert_source"] = expert_source or {}
        doc["layer_kinds"] = ["nvfp4"] * cfg.layers
    if "int8" in dense_formats:
        doc["dense_formats"] = ["bf16", "int8"]
        doc["dense_format"] = "int8"
        doc["int8"] = INT8_ENCODING
    return doc


def read_layout(directory):
    directory = Path(directory)
    doc = json.loads((directory / "layout.json").read_text())
    if doc.get("format") not in FORMATS:
        raise ValueError(f"{directory}: unexpected container format {doc.get('format')!r}")
    return doc


def layout_expert_format(doc):
    """`"int4"` (format v1) or `"nvfp4"` (format v2) of a layout document."""
    return doc.get("expert_format", "int4")


def container_expert_format(directory):
    """The expert format of the container at `directory` (`layout_expert_format`)."""
    return layout_expert_format(read_layout(directory))


def layout_dense_formats(doc):
    """The dense-projection formats a container holds: `("bf16",)` or `("bf16", "int8")`."""
    return tuple(doc.get("dense_formats", ["bf16"]))


def layout_dense_format(doc):
    """The container's preferred dense format (`"int8"` once `quantize_dense` ran, else bf16)."""
    return doc.get("dense_format", "bf16")


def container_dense_format(directory):
    return layout_dense_format(read_layout(directory))


def weight_array_names(doc, dense_format=None):
    """The arrays `load_presharded` loads for `dense_format` (default: the preferred one):
    every array of the container except the dense projections of the other format."""
    dense_format = dense_format or layout_dense_format(doc)
    if dense_format not in layout_dense_formats(doc):
        raise ValueError(
            f"container holds dense formats {layout_dense_formats(doc)}, not {dense_format!r}"
        )
    other = "int8" if dense_format == "bf16" else "bf16"
    skip = set(layout.dense_families(other))
    return [name for name in doc["arrays"] if name not in skip]


def effective_layout(directory, doc=None):
    """`layout.json`, or -- when `quantize_dense(..., publish=False)` completed the int8
    families without publishing them -- the document `quantize_dense` would publish, so an
    explicit `dense_format="int8"` can already load them."""
    directory = Path(directory)
    doc = doc or read_layout(directory)
    state = read_progress(directory).get("dense_int8") or {}
    if "int8" in layout_dense_formats(doc) or not state.get("complete"):
        return doc
    cfg, tp = config_from_layout(doc), doc["tp"]
    full = make_layout(cfg, tp, doc.get("revision"), layout_expert_format(doc),
                       doc.get("expert_source"), dense_formats=("bf16", "int8"))
    out = dict(doc)
    out["arrays"], out["total_bytes"] = full["arrays"], full["total_bytes"]
    out["dense_formats"], out["int8"] = full["dense_formats"], full["int8"]
    out["dense_format"] = layout_dense_format(doc)  # unpublished: bf16 stays preferred
    return out


def config_from_layout(doc):
    c = dict(doc["config"])
    c["eos"] = tuple(c["eos"])
    return Config(**c)


def is_presharded(directory):
    return (Path(directory) / "layout.json").is_file()


def read_progress(directory):
    path = Path(directory) / "progress.json"
    if path.is_file():
        return json.loads(path.read_text())
    return {"layers": [], "globals": False, "complete": False}


def _write_json(path, doc):
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, path)


def is_complete(directory):
    return is_presharded(directory) and read_progress(directory).get("complete", False)


def rank_file(directory, rank, name):
    return Path(directory) / f"rank{rank}" / f"{name}.bin"


def _pwrite_all(fd, data, offset):
    view = memoryview(data).cast("B")
    done = 0
    while done < len(view):
        done += os.pwrite(fd, view[done:], offset + done)


def _array_bytes(a):
    """A C-contiguous array as a writable-buffer view (bf16/int4 have no PEP 3118 format)."""
    return np.ascontiguousarray(a).view(np.uint8).reshape(-1)


# State of the conversion workers (set by `_worker_init` in every pool process).
_SHARED = SimpleNamespace(buffers=None, shms=[], fds={})


def _worker_init(shm_names, plan, cfg, tp, dst, doc):
    _SHARED.shms = [shared_memory.SharedMemory(name) for name in shm_names]  # attach only
    _SHARED.buffers = [shm.buf for shm in _SHARED.shms]
    _SHARED.plan, _SHARED.cfg, _SHARED.tp = plan, cfg, tp
    _SHARED.dst, _SHARED.layout, _SHARED.fds = dst, doc, {}


def _buffer_view(slot, suffix):
    offset, dtype, shape = _SHARED.plan[suffix]
    count = int(np.prod(shape))
    return np.frombuffer(_SHARED.buffers[slot], _np_dtype(dtype), count, offset).reshape(shape)


def _worker_fd(rank, name):
    key = (rank, name)
    if key not in _SHARED.fds:
        _SHARED.fds[key] = os.open(rank_file(_SHARED.dst, rank, name), os.O_WRONLY)
    return _SHARED.fds[key]


def _write_rank_array(rank, name, layer, value, expert=None):
    """Write one (layer[, expert]) chunk of a per-rank array at its container offset."""
    spec = _SHARED.layout["arrays"][name]
    dshape, ddtype = spec["disk_shape"], np.dtype(spec["disk_dtype"])
    data = _array_bytes(value)
    if expert is None:
        per_layer = int(np.prod(dshape[1:])) * ddtype.itemsize
        if len(data) != per_layer:
            raise ValueError(f"{name}: layer chunk has {len(data)} bytes, expected {per_layer}")
        offset = layer * per_layer
    else:
        per_expert = int(np.prod(dshape[2:])) * ddtype.itemsize
        if len(data) != per_expert:
            raise ValueError(f"{name}: expert chunk has {len(data)} bytes, expected {per_expert}")
        offset = (layer * dshape[1] + expert) * per_expert
    _pwrite_all(_worker_fd(rank, name), data, offset)
    return len(data)


def _expert_task(args):
    slot, layer, e0, e1 = args
    cfg, tp = _SHARED.cfg, _SHARED.tp
    gate_up = _buffer_view(slot, "mlp.experts.gate_up_proj")
    down = _buffer_view(slot, "mlp.experts.down_proj")
    written = 0
    for e in range(e0, e1):
        for rank in range(tp):
            for name, value in expert_rank_arrays(cfg, tp, rank, gate_up[e], down[e]).items():
                written += _write_rank_array(rank, name, layer, value, expert=e)
    return written


def _dense_task(args):
    slot, layer, names = args
    cfg, tp = _SHARED.cfg, _SHARED.tp
    tensors = {s: _buffer_view(slot, s) for s in LAYER_SUFFIXES if s not in EXPERT_SUFFIXES}
    written = 0
    for rank in range(tp):
        for name, value in dense_rank_arrays(cfg, tp, rank, tensors, names).items():
            written += _write_rank_array(rank, name, layer, value)
    return written


def _layer_plan(checkpoint, layer):
    """`{suffix: (buffer offset, dtype name, shape)}` for one layer's tensors, expert
    tensors first (they are the bulk), plus the total size."""
    plan, offset = {}, 0
    ordered = list(EXPERT_SUFFIXES) + [s for s in LAYER_SUFFIXES if s not in EXPERT_SUFFIXES]
    for suffix in ordered:
        _, _, nbytes, dtype, shape = checkpoint.meta(layer_key(layer, suffix))
        plan[suffix] = (offset, _dtype_name(dtype), list(shape))
        offset += (nbytes + 4095) // 4096 * 4096
    return plan, offset


def _read_layer_into(checkpoint, layer, plan, buffer, threads):
    """Stream every tensor of `layer` into its planned slot of the shared `buffer`."""
    jobs = []
    view = memoryview(buffer)
    for suffix, (offset, _, _) in plan.items():
        path, file_offset, nbytes, _, _ = checkpoint.meta(layer_key(layer, suffix))
        jobs.append((path, file_offset, view[offset:offset + nbytes]))
    started = time.perf_counter()
    total = _read_ranges(jobs, threads)
    with checkpoint._lock:
        checkpoint.bytes_read += total
        checkpoint.read_seconds += time.perf_counter() - started
    return total


def _allocate_files(dst, doc):
    """Create (sparse) every `rank{r}/<name>.bin` at its final size; check existing ones."""
    for r in range(doc["tp"]):
        (Path(dst) / f"rank{r}").mkdir(parents=True, exist_ok=True)
        for name, spec in doc["arrays"].items():
            path = rank_file(dst, r, name)
            if path.exists():
                if path.stat().st_size != spec["nbytes"]:
                    raise ValueError(f"{path}: size {path.stat().st_size} != {spec['nbytes']}")
                continue
            with open(path, "wb") as f:
                f.truncate(spec["nbytes"])


def _convert_globals(checkpoint, cfg, tp, dst, doc, threads, log):
    """embed / lm_head / final_norm: read each rank's rows of the vocab tables directly."""
    vp = layout.vocab_pad(cfg, tp)
    norm = checkpoint.read(GLOBAL_KEYS["final_norm"])
    for rank in range(tp):
        rows = slice(rank * vp, min((rank + 1) * vp, cfg.vocab))
        if rows.start >= cfg.vocab:
            embed_rows = lm_rows = np.zeros((0, cfg.hidden), BF16)
        else:
            embed_rows = checkpoint.read(GLOBAL_KEYS["embed"], rows, threads)
            lm_rows = checkpoint.read(GLOBAL_KEYS["lm_head"], rows, threads)
        arrays = global_rank_arrays(cfg, tp, rank, embed_rows, lm_rows, norm)
        for name, value in arrays.items():
            data = _array_bytes(value)
            if len(data) != doc["arrays"][name]["nbytes"]:
                raise ValueError(f"{name}: {len(data)} bytes != {doc['arrays'][name]['nbytes']}")
            fd = os.open(rank_file(dst, rank, name), os.O_WRONLY)
            try:
                _pwrite_all(fd, data, 0)
                os.fsync(fd)
            finally:
                os.close(fd)
        log(f"globals: rank {rank} written ({checkpoint.gbps():.2f} GB/s read so far)")


def convert_presharded(src_dir, dst_dir, tp=8, group=128, layers=None, workers=None,
                       read_threads=READ_THREADS, experts_per_task=2, log=print):
    """Convert the HF checkpoint at `src_dir` into the pre-sharded container `dst_dir`.

    `layers` restricts the work to those layer indices (default: all); the globals
    (`embed`, `lm_head`, `final_norm`) are converted on the first run. Re-running resumes
    from `progress.json`; finished layers are never rewritten. Returns the progress doc.
    Uses a spawned process pool: call it from an importable module or a guarded `__main__`.
    """
    started = time.perf_counter()
    checkpoint = Checkpoint(src_dir)
    cfg = checkpoint.config
    if group != cfg.group_size:
        cfg = Config(**{**cfg.__dict__, "group_size": group})
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    doc = make_layout(cfg, tp, revision=checkpoint.revision)
    if is_presharded(dst):
        existing = read_layout(dst)
        for key in ("config", "tp", "group", "arrays"):
            if existing[key] != doc[key]:
                raise ValueError(f"{dst}: existing container differs in {key}; use a new dst")
        if existing.get("revision") != doc["revision"]:
            raise ValueError(f"{dst}: existing container was converted from another revision")
    else:
        _write_json(dst / "layout.json", doc)
    _allocate_files(dst, doc)
    progress = read_progress(dst)
    done = set(progress["layers"])
    todo = [l for l in (range(cfg.layers) if layers is None else layers) if l not in done]
    log(f"convert {src_dir} -> {dst}: tp={tp} group={cfg.group_size} layers={len(todo)} todo "
        f"({len(done)} done), {doc['total_bytes'] / 1e9:.1f} GB total")

    def save_progress(**updates):
        progress.update(updates)
        progress["layers"] = sorted(done)
        progress["complete"] = progress["globals"] and len(done) == cfg.layers
        progress["elapsed"] = progress.get("elapsed", 0.0)
        _write_json(dst / "progress.json", progress)

    if not progress["globals"]:
        _convert_globals(checkpoint, cfg, tp, dst, doc, read_threads, log)
        save_progress(globals=True)
    if not todo:
        save_progress()
        return progress

    # Two shared-memory layer buffers: the reader thread fills one while the worker pool
    # quantizes the other. Workers are spawned (no jax/thread state of this process is
    # inherited; a script calling this must guard its `__main__` as usual with spawn) and
    # attach the buffers by name; they only run numpy.
    plan, slot_bytes = _layer_plan(checkpoint, todo[0])
    workers = workers or max(1, min(os.cpu_count() - 8, 128))
    shms = [shared_memory.SharedMemory(create=True, size=slot_bytes) for _ in range(2)]
    log(f"layer buffers: 2 x {slot_bytes / 2**30:.1f} GiB, {workers} quantization workers")
    free_slots, ready = queue.Queue(), queue.Queue()
    for slot in range(2):
        free_slots.put(slot)
    failure = []

    def reader():
        try:
            for layer in todo:
                slot = free_slots.get()
                t0 = time.perf_counter()
                nbytes = _read_layer_into(checkpoint, layer, plan, shms[slot].buf, read_threads)
                ready.put((layer, slot, nbytes, time.perf_counter() - t0))
        except BaseException as error:  # noqa: BLE001 - propagate to the main thread
            failure.append(error)
            ready.put(None)

    fds = {(r, n): os.open(rank_file(dst, r, n), os.O_WRONLY)
           for r in range(tp) for n in doc["arrays"]}
    e_chunks = [(e, min(e + experts_per_task, cfg.experts))
                for e in range(0, cfg.experts, experts_per_task)]
    layer_times, last = [], time.perf_counter()
    ctx = multiprocessing.get_context("spawn")
    init_args = ([shm.name for shm in shms], plan, cfg, tp, dst, doc)
    try:
        with ctx.Pool(workers, initializer=_worker_init, initargs=init_args) as pool:
            _worker_init(*init_args)  # the main process converts the vector families itself
            threading.Thread(target=reader, daemon=True).start()
            for i, layer in enumerate(todo):
                item = ready.get()
                if item is None:
                    raise failure[0]
                layer_got, slot, nbytes, read_time = item
                assert layer_got == layer
                t0 = time.perf_counter()
                results = [pool.apply_async(_expert_task, ((slot, layer, e0, e1),))
                           for e0, e1 in e_chunks]
                results += [pool.apply_async(_dense_task, ((slot, layer, names),))
                            for names in DENSE_TASKS]
                written = _dense_task((slot, layer, VECTOR_NAMES))
                written += sum(r.get() for r in results)
                free_slots.put(slot)
                for fd in fds.values():
                    os.fsync(fd)
                done.add(layer)
                now = time.perf_counter()
                layer_times.append(now - last)
                progress["elapsed"] = progress.get("elapsed", 0.0) + now - last
                last = now
                save_progress()
                eta = np.mean(layer_times[-4:]) * (len(todo) - i - 1)
                log(f"layer {layer}: read {nbytes / 1e9:.1f} GB in {read_time:.1f} s "
                    f"({nbytes / read_time / 1e9:.2f} GB/s), quantized+wrote "
                    f"{written / 1e9:.1f} GB in {now - t0:.1f} s; {len(done)}/{cfg.layers} "
                    f"layers, avg read {checkpoint.gbps():.2f} GB/s, ETA {eta / 60:.1f} min")
    finally:
        for fd in list(fds.values()) + list(_SHARED.fds.values()):
            os.close(fd)
        for shm in _SHARED.shms + shms:
            shm.close()
        for shm in shms:
            shm.unlink()
        _SHARED.buffers, _SHARED.shms, _SHARED.fds = None, [], {}
    save_progress()
    log(f"done: {len(done)}/{cfg.layers} layers, complete={progress['complete']}, "
        f"{progress['elapsed'] / 60:.1f} min total, {checkpoint.bytes_read / 1e9:.0f} GB read "
        f"at {checkpoint.gbps():.2f} GB/s")
    return progress


# ---------------------------------------------------------------------------------------
# Reading the container back (numpy, for tests/verification)
# ---------------------------------------------------------------------------------------


def read_rank_array(directory, rank, name, index=None, doc=None, unpack=True):
    """One per-rank array (or the leading-axes selection `index`, e.g. a layer or a
    `(layer, expert)` tuple) from the container as numpy; int4 arrays are unpacked to
    `ml_dtypes.int4` unless `unpack=False` (raw packed uint8)."""
    doc = doc or read_layout(directory)
    spec = doc["arrays"][name]
    dshape, ddtype = tuple(spec["disk_shape"]), np.dtype(spec["disk_dtype"])
    mm = np.memmap(rank_file(directory, rank, name), ddtype, mode="r", shape=dshape)
    value = np.array(mm if index is None else mm[index])
    if spec["dtype"] == "int4" and unpack:
        value = quant.unpack_int4(value, INT4)
    elif spec["dtype"] == "float8_e4m3fn" and unpack:
        value = value.view(E4M3)
    return value


def read_presharded(directory, names=None, layers=None):
    """`{name: array[tp, ...]}` host arrays of the whole container (small containers)."""
    doc = read_layout(directory)
    names = names or list(doc["arrays"])
    return {
        name: np.stack([read_rank_array(directory, r, name, layers, doc) for r in range(doc["tp"])])
        for name in names
    }


# ---------------------------------------------------------------------------------------
# Device placement
# ---------------------------------------------------------------------------------------


def _read_file(path, offset=0, nbytes=None, dtype=np.uint8, threads=READ_THREADS):
    nbytes = os.path.getsize(path) - offset if nbytes is None else nbytes
    out = np.empty(nbytes, np.uint8)
    if nbytes:
        _read_ranges([(path, offset, memoryview(out))], threads)
    return out.view(dtype)


def _load_rank(directory, doc, rank, device, layer_chunk_bytes, log, names=None):
    """All arrays (or `names`) of one rank as single-device arrays on `device`."""
    placement = SingleDeviceSharding(device)
    arrays = {}
    for name in names if names is not None else doc["arrays"]:
        spec = doc["arrays"][name]
        path = rank_file(directory, rank, name)
        shape, dtype = tuple(spec["shape"]), _np_dtype(spec["dtype"])
        dshape, ddtype = tuple(spec["disk_shape"]), np.dtype(spec["disk_dtype"])
        per_layer = int(np.prod(dshape[1:])) * ddtype.itemsize if spec["layer_axis"] == 0 else 0
        host_dtype = E4M3 if dtype == E4M3 else ddtype  # e4m3 bits are viewed, never converted
        if dtype != INT4 and (spec["nbytes"] <= layer_chunk_bytes or spec["layer_axis"] != 0):
            value = _read_file(path, dtype=host_dtype).reshape(dshape)
            arrays[name] = jax.device_put(value[None], device)
            del value
        elif dtype != INT4:
            # Stream one layer at a time into a preallocated device buffer (donated).
            jdtype = jnp.dtype(host_dtype)
            buffer = jax.jit(lambda: jnp.zeros((1, *shape), jdtype), out_shardings=placement)()

            def update(buffer, chunk, index):
                start = (0, index) + (0,) * (buffer.ndim - 2)
                return jax.lax.dynamic_update_slice(buffer, chunk[None, None], start)

            update = jax.jit(update, donate_argnums=0, out_shardings=placement)
            for l in range(shape[0]):
                chunk = _read_file(path, l * per_layer, per_layer, host_dtype).reshape(dshape[1:])
                buffer = update(buffer, jax.device_put(chunk, device), jnp.int32(l))
                del chunk
            arrays[name] = buffer
        elif spec["nbytes"] <= layer_chunk_bytes or spec["layer_axis"] != 0:
            packed = jax.device_put(_read_file(path).reshape(dshape)[None], device)
            arrays[name] = jax.jit(quant.unpack_int4_jnp, out_shardings=placement)(packed)
            del packed
        else:
            # Stream one layer of packed nibbles at a time, unpack on the device and drop
            # it into a preallocated int4 buffer (donated, so no second copy exists).
            buffer = jax.jit(lambda: jnp.zeros((1, *shape), jnp.int4), out_shardings=placement)()

            def update(buffer, packed, index):
                chunk = quant.unpack_int4_jnp(packed)[None, None]
                start = (0, index) + (0,) * (buffer.ndim - 2)
                return jax.lax.dynamic_update_slice(buffer, chunk, start)

            update = jax.jit(update, donate_argnums=0, out_shardings=placement)
            for l in range(shape[0]):
                chunk = _read_file(path, l * per_layer, per_layer).reshape(dshape[1:])
                buffer = update(buffer, jax.device_put(chunk, device), jnp.int32(l))
                del chunk
            arrays[name] = buffer
        log(f"rank {rank}: {name} resident")
    jax.block_until_ready(arrays)
    return arrays


def load_presharded(mesh, directory, cfg=None, *, ranks_in_flight=8, layer_chunk_bytes=1 << 30,
                    log=None, dense_format=None):
    """The container at `directory` as `{name: jax.Array[tp, ...]}` sharded `P("tp")` over
    `mesh` (rank r's arrays live on `mesh.devices.flat[r]`). int4 arrays are `jnp.int4`; a v2
    (nvfp4) container yields int32 packed codes, `float8_e4m3fn` block scales and f32 `expert_gs`
    (`layout.FP4_EXPERT_FAMILIES`) instead of the int4 families. `dense_format` ("bf16" /
    "int8", default: the container's preferred `dense_format`, int8 once `quantize_dense`
    ran) selects which dense-projection arrays are loaded (`weight_array_names`): the bf16
    families `q .. post`, `lm_head`, or their int8 `_i8` / `_s` pairs."""
    directory = Path(directory)
    doc = read_layout(directory)
    if not read_progress(directory).get("complete"):
        raise ValueError(f"{directory}: conversion is not complete (see progress.json)")
    if dense_format == "int8":
        doc = effective_layout(directory, doc)
    tp = doc["tp"]
    fmt = layout_expert_format(doc)
    if cfg is not None:
        expected = make_layout(cfg, tp, expert_format=fmt, dense_formats=layout_dense_formats(doc))
    if cfg is not None and expected["arrays"] != doc["arrays"]:
        raise ValueError(f"{directory}: container layout does not match the requested config")
    names = weight_array_names(doc, dense_format)
    devices = list(mesh.devices.flat)
    if len(devices) != tp:
        raise ValueError(f"mesh has {len(devices)} devices, container has tp={tp}")
    log = log or (lambda _: None)
    started = time.perf_counter()
    per_rank = {}
    with ThreadPoolExecutor(max(1, ranks_in_flight)) as pool:
        futures = {r: pool.submit(_load_rank, directory, doc, r, devices[r], layer_chunk_bytes,
                                  log, names) for r in range(tp)}
        for r, future in futures.items():
            per_rank[r] = future.result()
            log(f"rank {r} resident after {time.perf_counter() - started:.0f} s")
    sharding = NamedSharding(mesh, P("tp"))
    weights = {}
    for name in names:
        spec = doc["arrays"][name]
        weights[name] = jax.make_array_from_single_device_arrays(
            (tp, *spec["shape"]), sharding, [per_rank[r][name] for r in range(tp)]
        )
    total = tp * sum(doc["arrays"][name]["nbytes"] for name in names)
    log(f"{total / 1e9:.0f} GB resident after {time.perf_counter() - started:.0f} s "
        f"({total / 1e9 / (time.perf_counter() - started):.2f} GB/s)")
    return weights


def abstract_weights(mesh, cfg, tp=None, expert_format="int4", dense_format="bf16"):
    """`jax.ShapeDtypeStruct` tree matching `load_presharded`, for AOT compilation
    (`expert_format` / `dense_format` of the container: `container_expert_format`,
    `container_dense_format`)."""
    tp = tp or mesh.size
    sharding = NamedSharding(mesh, P("tp"))
    out = {}
    for name, (shape, dtype) in _shapes(cfg, tp, expert_format, dense_format).items():
        jdtype = jnp.int4 if dtype == INT4 else jnp.dtype(dtype)
        out[name] = jax.ShapeDtypeStruct((tp, *shape), jdtype, sharding=sharding)
    return out


def zero_caches(mesh, cfg, batch, context):
    """`{"k_cache", "v_cache"}` zero `[tp, L, B, context, lanes]` bf16 arrays sharded `P("tp")`."""
    tp = mesh.size
    sharding = NamedSharding(mesh, P("tp"))
    out = {}
    for name, (shape, dtype) in layout.kv_cache_shapes(cfg, batch, context, tp).items():
        out[name] = jax.jit(lambda: jnp.zeros((tp, *shape), dtype), out_shardings=sharding)()
    return out


# ---------------------------------------------------------------------------------------
# int8 dense projections written in place into an existing container (`quantize-dense`)
# ---------------------------------------------------------------------------------------
# One pool task per (layer, family): every rank's `[K, N_r]` bf16 layer slab is memmapped,
# quantized with `quant.quantize_int8_np` (per output column) and written at the layer offset
# of the preallocated `rank{r}/<family>_i8.bin` / `<family>_s.bin`. The row-sharded `o` is
# quantized over the concatenated rows of all ranks (one shared `o_s`); the lm_head is one task
# per rank. The bf16 families are never touched, so the container keeps serving both formats.

_Q = SimpleNamespace(directory=None, doc=None, tp=None, fds={})


def _q_init(directory, doc):
    _Q.directory, _Q.doc, _Q.tp, _Q.fds = Path(directory), doc, doc["tp"], {}


def _q_memmap(rank, name):
    spec = _Q.doc["arrays"][name]
    return np.memmap(rank_file(_Q.directory, rank, name), np.dtype(spec["disk_dtype"]), mode="r",
                     shape=tuple(spec["disk_shape"]))


def _q_write(rank, name, layer, value):
    key = (rank, name)
    if key not in _Q.fds:
        _Q.fds[key] = os.open(rank_file(_Q.directory, rank, name), os.O_WRONLY)
    data = _array_bytes(value)
    _pwrite_all(_Q.fds[key], data, 0 if layer is None else layer * data.nbytes)


def _quantize_dense_task(args):
    index, family = args
    tp = _Q.tp
    if family == "lm_head":  # `index` is the rank
        q, s = quant.quantize_int8_np(np.asarray(_q_memmap(index, "lm_head")))
        _q_write(index, layout.int8_family(family), None, q)
        _q_write(index, layout.scale_family(family), None, s)
        return args
    if family == "o":  # row-sharded: one scale per column over all ranks' rows
        parts = [np.asarray(_q_memmap(r, "o")[index]) for r in range(tp)]
        q, s = quant.quantize_int8_np(np.concatenate(parts, axis=0))
        rows = parts[0].shape[0]
        for r in range(tp):
            _q_write(r, "o_i8", index, np.ascontiguousarray(q[r * rows:(r + 1) * rows]))
            _q_write(r, "o_s", index, s)
        return args
    for r in range(tp):
        q, s = quant.quantize_int8_np(np.asarray(_q_memmap(r, family)[index]))
        _q_write(r, layout.int8_family(family), index, q)
        _q_write(r, layout.scale_family(family), index, s)
    return args


def quantize_dense(directory, workers=None, layers=None, log=print, publish=True):
    """Add the int8 dense families (`layout.dense_families("int8")`) to the complete container
    at `directory`, in place: new `.bin` files next to the bf16 ones, `progress.json`
    (`dense_int8`: resumable per layer) and, once every layer and the lm_head are done,
    `layout.json` with the new `arrays` entries, `dense_formats: [bf16, int8]` and
    `dense_format: int8` (the format `load_presharded` picks by default from then on).
    `layers` restricts the run to some layers (the lm_head is always done). With
    `publish=False` the files and the progress are written but `layout.json` is left alone
    (readers keep getting the bf16 tree); a later call publishes a complete set. Returns the
    progress document."""
    directory = Path(directory)
    doc = read_layout(directory)
    progress = read_progress(directory)
    if not progress.get("complete"):
        raise ValueError(f"{directory}: conversion is not complete (see progress.json)")
    cfg, tp = config_from_layout(doc), doc["tp"]
    full = make_layout(cfg, tp, doc.get("revision"), layout_expert_format(doc),
                       doc.get("expert_source"), dense_formats=("bf16", "int8"))
    if {n: full["arrays"][n] for n in doc["arrays"]} != doc["arrays"]:
        raise ValueError(f"{directory}: container layout does not match its config")
    new_names = [n for n in full["arrays"] if n not in doc["arrays"]]
    state = progress.get("dense_int8") or {"layers": [], "lm_head": False, "complete": False}
    if state["complete"] and layers is None:
        log(f"{directory}: int8 dense families already complete")
        if publish and layout_dense_format(doc) != "int8":
            _publish_int8(directory, doc, full, log)
        return progress
    for r in range(tp):
        for name in new_names:
            path, size = rank_file(directory, r, name), full["arrays"][name]["nbytes"]
            if not path.exists() or path.stat().st_size != size:
                with open(path, "wb") as f:
                    f.truncate(size)
    wanted = list(range(cfg.layers)) if layers is None else sorted(set(int(l) for l in layers))
    todo = [l for l in wanted if l not in state["layers"]]
    tasks = [(l, f) for l in todo for f in layout.INT8_DENSE]
    if not state["lm_head"]:
        tasks += [(r, "lm_head") for r in range(tp)]
    workers = workers or max(1, min(32, os.cpu_count() or 1, len(tasks)))
    log(f"{directory}: quantizing {len(todo)} layers x {len(layout.INT8_DENSE)} families"
        f"{'' if state['lm_head'] else ' + lm_head'} with {workers} workers ({len(tasks)} tasks)")

    def save():
        progress["dense_int8"] = state
        _write_json(directory / "progress.json", progress)

    started = time.perf_counter()
    done = {l: 0 for l in todo}
    lm_done = 0
    if tasks:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(workers, initializer=_q_init, initargs=(directory, doc)) as pool:
            for i, (index, family) in enumerate(pool.imap_unordered(_quantize_dense_task, tasks)):
                if family == "lm_head":
                    lm_done += 1
                    if lm_done == tp:
                        state["lm_head"] = True
                        save()
                else:
                    done[index] += 1
                    if done[index] == len(layout.INT8_DENSE):
                        state["layers"] = sorted(set(state["layers"]) | {index})
                        save()
                if (i + 1) % 32 == 0 or i + 1 == len(tasks):
                    log(f"  {i + 1}/{len(tasks)} tasks after {time.perf_counter() - started:.0f} s")
    state["complete"] = bool(state["lm_head"]) and len(state["layers"]) == cfg.layers
    save()
    if state["complete"]:
        log(f"{directory}: int8 dense families complete in {time.perf_counter() - started:.0f} s")
        if publish:
            _publish_int8(directory, doc, full, log)
    return progress


def _publish_int8(directory, doc, full, log):
    """Rewrite `layout.json` with the int8 arrays and `dense_format: int8`."""
    doc = dict(doc)
    doc["arrays"] = full["arrays"]
    doc["total_bytes"] = full["total_bytes"]
    doc["dense_formats"], doc["dense_format"], doc["int8"] = (
        full["dense_formats"], full["dense_format"], full["int8"])
    _write_json(Path(directory) / "layout.json", doc)
    log(f"{directory}: layout.json now prefers dense_format int8")


# ---------------------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------------------


class TokenizerAdapter:
    """Thin wrapper of the checkpoint's `tokenizers.Tokenizer`: `encode` never adds special
    tokens implicitly (the chat renderer emits BOS itself), special-token strings in the
    text are still recognised; `decode` skips nothing."""

    def __init__(self, inner):
        self.inner = inner

    def encode(self, text, **unused):
        return self.inner.encode(text, add_special_tokens=False).ids

    def decode(self, ids, **unused):
        return self.inner.decode([int(t) for t in ids], skip_special_tokens=False)

    def token_to_id(self, token):
        return self.inner.token_to_id(token)


def load_tokenizer(directory):
    from tokenizers import Tokenizer

    return TokenizerAdapter(Tokenizer.from_file(str(Path(directory) / "tokenizer.json")))


# ---------------------------------------------------------------------------------------
# Synthetic checkpoints (tests)
# ---------------------------------------------------------------------------------------


def nvfp4_layers(cfg):
    """Layers whose experts the vendor NVFP4 checkpoint quantizes: all but the first and last."""
    return tuple(range(1, cfg.layers - 1))


def checkpoint_tensor_specs(cfg, nvfp4=False):
    """`{name: (shape, np dtype)}` of every text-model tensor of an HF checkpoint for `cfg`
    (`nvfp4`: the experts of `nvfp4_layers(cfg)` in the vendor's NVFP4 tensor set)."""
    h, hm, i, e = cfg.hidden, cfg.moe_hidden, cfg.expert_hidden, cfg.experts
    f32, u8 = np.dtype(np.float32), np.dtype(np.uint8)
    nvfp4_experts = {
        "mlp.experts.gate_up_proj": ((e, 2 * i, hm // 2), u8),
        "mlp.experts.gate_up_proj_weight_scale": ((e, 2 * i, hm // 16), E4M3),
        "mlp.experts.gate_up_proj_weight_scale_2": ((e, 2), f32),
        "mlp.experts.gate_up_proj_input_scale": ((e, 2), f32),
        "mlp.experts.down_proj": ((e, hm, i // 2), u8),
        "mlp.experts.down_proj_weight_scale": ((e, hm, i // 16), E4M3),
        "mlp.experts.down_proj_weight_scale_2": ((e,), f32),
        "mlp.experts.down_proj_input_scale": ((e,), f32),
    }
    per_layer = {
        "input_layernorm.weight": ((h,), BF16),
        "self_attn.q_proj.weight": ((cfg.q_width, h), BF16),
        "self_attn.k_proj.weight": ((cfg.kv_width, h), BF16),
        "self_attn.v_proj.weight": ((cfg.kv_width, h), BF16),
        "self_attn.gate_proj.weight": ((cfg.q_width, h), BF16),
        "self_attn.o_proj.weight": ((h, cfg.q_width), BF16),
        "post_attention_residual_gate.gate": ((h,), BF16),
        "pre_feedforward_layernorm.weight": ((h,), BF16),
        "post_feedforward_layernorm.weight": ((h,), BF16),
        "post_feedforward_residual_gate.gate": ((h,), BF16),
        "mlp.gate.weight": ((e, h), np.dtype(np.float32)),
        "mlp.gate.e_score_correction_bias": ((e,), np.dtype(np.float32)),
        "mlp.pre_expert_proj.weight": ((hm, h), BF16),
        "mlp.pre_expert_norm.weight": ((hm,), BF16),
        "mlp.experts.gate_up_proj": ((e, 2 * i, hm), BF16),
        "mlp.experts.down_proj": ((e, hm, i), BF16),
        "mlp.experts.post_expert_norm.weight": ((hm,), BF16),
        "mlp.post_expert_proj.weight": ((h, hm), BF16),
    }
    specs = {
        GLOBAL_KEYS["embed"]: ((cfg.vocab, h), BF16),
        GLOBAL_KEYS["lm_head"]: ((cfg.vocab, h), BF16),
        GLOBAL_KEYS["final_norm"]: ((h,), BF16),
    }
    for layer in range(cfg.layers):
        for suffix in LAYER_SUFFIXES:
            specs[layer_key(layer, suffix)] = per_layer[suffix]
    return specs


def random_checkpoint_tensors(cfg, seed=0, nvfp4=False):
    """Deterministic random HF-named tensors for `cfg` (bf16 weights ~N(0, 0.05); the
    router f32 with a non-bf16-representable fraction; norms/gates around zero). With
    `nvfp4` the experts of `nvfp4_layers(cfg)` are NVFP4 tensors: random Gaussian experts
    quantized with `quant.quantize_nvfp4_np` in the checkpoint's `[out, in]` orientation."""
    rng = np.random.default_rng(seed)
    tensors = {}
    for name, (shape, dtype) in checkpoint_tensor_specs(cfg, nvfp4).items():
        if dtype == np.uint8 or dtype == E4M3 or name.endswith(("_scale_2", "_scale")):
            continue  # filled below from the quantizer
        if dtype == np.float32:
            value = rng.standard_normal(shape, np.float32) * np.float32(0.05)
        elif len(shape) == 1:
            value = rng.standard_normal(shape, np.float32) * np.float32(0.5)
        else:
            value = rng.standard_normal(shape, np.float32) * np.float32(0.05)
        tensors[name] = value.astype(dtype)
    if nvfp4:
        e, i, hm = cfg.experts, cfg.expert_hidden, cfg.moe_hidden
        for layer in nvfp4_layers(cfg):
            p = f"{PREFIX}layers.{layer}.mlp.experts."
            for kind, out_rows, in_cols, halves in (("gate_up", 2 * i, hm, 2), ("down", hm, i, 1)):
                # Blocks run along `in`: quantize the transposed `[E, in, out]` (K = in) per
                # half (gate / up global scales) and pack the codes back along `in`.
                w = rng.standard_normal((e, out_rows, in_cols), np.float32) * np.float32(0.05)
                w = w.astype(BF16).astype(np.float32)
                wt = np.ascontiguousarray(w.transpose(0, 2, 1))  # [E, in, out]
                packed_parts, bs_parts, gs_parts = [], [], []
                width = out_rows // halves
                for hf in range(halves):
                    pk, bs, gs = quant.quantize_nvfp4_np(wt[..., hf * width:(hf + 1) * width])
                    packed_parts.append(pk)
                    bs_parts.append(bs)
                    gs_parts.append(gs)
                codes = quant.unpack_fp4_rows(np.concatenate(packed_parts, axis=-1))  # [E, in, out]
                codes = codes.transpose(0, 2, 1)  # [E, out, in]
                lo, hi = codes[..., 0::2], codes[..., 1::2]
                tensors[p + f"{kind}_proj"] = np.ascontiguousarray(lo | (hi << 4)).astype(np.uint8)
                bs = np.concatenate(bs_parts, axis=-1).transpose(0, 2, 1)  # [E, out, in/16]
                tensors[p + f"{kind}_proj_weight_scale"] = np.ascontiguousarray(bs)
                gs = np.stack(gs_parts, axis=-1) if halves > 1 else gs_parts[0]
                tensors[p + f"{kind}_proj_weight_scale_2"] = gs.astype(np.float32)
                tensors[p + f"{kind}_proj_input_scale"] = np.full(gs.shape, 0.01, np.float32)
    return tensors


def save_safetensors(path, tensors):
    """Minimal safetensors writer (numpy, incl. ml_dtypes bf16/e4m3 which `safetensors.numpy`
    may not know): header JSON + raw little-endian data."""
    header, offset = {}, 0
    order = sorted(tensors)
    for name in order:
        a = np.ascontiguousarray(tensors[name])
        header[name] = {"dtype": Checkpoint.NAMES[np.dtype(a.dtype)], "shape": list(a.shape),
                        "data_offsets": [offset, offset + a.nbytes]}
        offset += a.nbytes
    header["__metadata__"] = {"format": "np"}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for name in order:
            f.write(_array_bytes(tensors[name]).tobytes())


def write_checkpoint(directory, cfg, tensors, shards=3, nvfp4=False):
    """Write `tensors` as an HF safetensors checkpoint (`model-0000k-of-0000n.safetensors` +
    `model.safetensors.index.json` + a `config.json` that `Config.from_checkpoint` accepts;
    `nvfp4` adds the vendor's `hf_quant_config.json`)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = sorted(tensors)
    weight_map, total = {}, 0
    for k in range(shards):
        filename = f"model-{k + 1:05d}-of-{shards:05d}.safetensors"
        part = {n: tensors[n] for n in names[k::shards]}
        save_safetensors(directory / filename, part)
        for n in part:
            weight_map[n] = filename
            total += tensors[n].nbytes
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map})
    )
    every, offset = cfg.full_attention_every, cfg.full_attention_offset
    text = {
        "attention_bias": False, "hidden_act": "silu", "num_shared_experts": 0,
        "routed_scaling_factor": 1.0, "tie_word_embeddings": False,
        "rope_parameters": {"rope_theta": cfg.rope_theta, "rope_type": "default"},
        "num_hidden_layers": cfg.layers, "mlp_layer_types": ["sparse"] * cfg.layers,
        "layer_types": ["full_attention" if l % every == offset else "sliding_attention"
                        for l in range(cfg.layers)],
        "layer_rope_theta": [0 if l % every == offset else cfg.rope_theta
                             for l in range(cfg.layers)],
        "hidden_size": cfg.hidden, "moe_hidden_size": cfg.moe_hidden,
        "moe_intermediate_size": cfg.expert_hidden, "num_local_experts": cfg.experts,
        "num_experts_per_tok": cfg.top_k, "num_attention_heads": cfg.heads,
        "num_key_value_heads": cfg.kv_heads, "head_dim": cfg.head_dim, "vocab_size": cfg.vocab,
        "sliding_window": cfg.sliding_window, "rms_norm_eps": cfg.rms_eps,
        "post_norm_eps": cfg.post_eps, "residual_gate_temperature": cfg.gate_temperature,
        "qk_scale_factor": cfg.qk_scale_factor, "output_multiplier": cfg.output_multiplier,
        "final_logit_softcapping": cfg.softcap, "eos_token_id": cfg.eos[0], "bos_token_id": cfg.bos,
        "pad_token_id": cfg.pad, "model_type": "muse_spark_text",
    }
    (directory / "config.json").write_text(json.dumps({"text_config": text}))
    (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": list(cfg.eos)}))
    if nvfp4:
        (directory / "hf_quant_config.json").write_text(json.dumps({
            "quant_method": "modelopt",
            "quantization": {"quant_algo": "NVFP4", "group_size": 16, "kv_cache_quant_algo": None,
                             "exclude_modules": [f"layers.{l}." for l in range(cfg.layers)
                                                 if l not in nvfp4_layers(cfg)]},
        }))
    return directory


def canonical_from_tensors(cfg, tensors):
    """HF-named tensors (a whole small checkpoint in memory) -> the reference's canonical
    dict (`musespark.shard_canonical` input), via the reference's own checkpoint mapping."""
    get = tensors.__getitem__
    out = canonical_global_from_checkpoint(cfg, get)
    layers = [canonical_layer_from_checkpoint(cfg, layer, get) for layer in range(cfg.layers)]
    for name in layers[0]:
        out[name] = np.stack([lw[name] for lw in layers])
    return out


def synthetic_checkpoint(directory, cfg, seed=0, shards=3):
    """Write a random HF checkpoint for `cfg`; returns its tensors (HF names)."""
    tensors = random_checkpoint_tensors(cfg, seed)
    write_checkpoint(directory, cfg, tensors, shards)
    return tensors


def synthetic_presharded(tmpdir, cfg, tp, seed=0, **convert_kwargs):
    """Synthetic HF checkpoint -> converted container under `tmpdir`.

    Returns `(checkpoint_dir, container_dir, tensors)` where `tensors` are the HF-named
    source arrays (feed them to `canonical_from_tensors` for `shard_canonical`)."""
    tmpdir = Path(tmpdir)
    src, dst = tmpdir / "checkpoint", tmpdir / f"presharded-tp{tp}"
    tensors = synthetic_checkpoint(src, cfg, seed)
    convert_presharded(src, dst, tp=tp, group=cfg.group_size, log=lambda _: None,
                       **convert_kwargs)
    return src, dst, tensors


def synthetic_presharded_nvfp4(tmpdir, cfg, tp, seed=0, shards=5, **convert_kwargs):
    """Synthetic NVFP4-format HF checkpoint -> format-v2 container under `tmpdir`;
    returns `(checkpoint_dir, container_dir, tensors)`."""
    tmpdir = Path(tmpdir)
    src, dst = tmpdir / "checkpoint-nvfp4", tmpdir / f"presharded-nvfp4-tp{tp}"
    tensors = random_checkpoint_tensors(cfg, seed, nvfp4=True)
    write_checkpoint(src, cfg, tensors, shards, nvfp4=True)
    convert_presharded_nvfp4(src, dst, tp=tp, config=cfg, log=lambda _: None, **convert_kwargs)
    return src, dst, tensors


# ---------------------------------------------------------------------------------------
# NVFP4 (container format v2): per-rank expert re-layout and RTN quantization of bf16 experts
# ---------------------------------------------------------------------------------------


def expert_gs_tile(gate, up, down):
    """`[8, 128]` f32 `expert_gs` tile: row 0 = gate, 1 = up, 2 = down global scale."""
    tile = np.zeros((layout.GS_ROWS, layout.GS_LANES), np.float32)
    tile[0], tile[1], tile[2] = np.float32(gate), np.float32(up), np.float32(down)
    return tile


def nvfp4_expert_rank_arrays(cfg, tp, rank, gate_up=None, gate_up_scale=None, down=None,
                             down_scale=None):
    """One rank's fp4 families of ONE vendor-quantized expert (pure byte re-layout).

    Checkpoint tensors (nn.Linear `[out, in]`, K packed low-nibble-first): `gate_up [2I, Hm/2]`
    uint8, `gate_up_scale [2I, Hm/16]` e4m3, `down [Hm, I/2]` uint8, `down_scale [Hm, I/16]`
    e4m3 -> `gate_up_fp4 [Hm/8, 2*Is]` int32, `gate_up_bs [Hm/KC, KC/16, 2*Is]` e4m3 (uint8 bits
    on disk), `down_fp4 [Is/8, Hm]`, `down_bs [Is/KC, KC/16, Hm]`; only the families whose
    inputs are given are returned."""
    i_s, i_full, hm = cfg.expert_hidden // tp, cfg.expert_hidden, cfg.moe_hidden
    gate_rows = slice(rank * i_s, (rank + 1) * i_s)
    up_rows = slice(i_full + rank * i_s, i_full + (rank + 1) * i_s)
    out = {}
    if gate_up is not None:
        rows = np.concatenate([gate_up[gate_rows], gate_up[up_rows]], axis=0)  # [2*Is, Hm/2]
        out["gate_up_fp4"] = quant.nvfp4_rows_to_packed(rows)  # [Hm/8, 2*Is]
    if gate_up_scale is not None:
        sc = np.asarray(gate_up_scale).view(np.uint8)
        sc = np.ascontiguousarray(np.concatenate([sc[gate_rows], sc[up_rows]], axis=0).T)
        out["gate_up_bs"] = quant.fp4_scales_to_chunked(sc, hm).view(E4M3)
    if down is not None:
        cols = np.asarray(down, np.uint8)[:, rank * (i_s // 2):(rank + 1) * (i_s // 2)]
        out["down_fp4"] = quant.nvfp4_rows_to_packed(cols)  # [Is/8, Hm]
    if down_scale is not None:
        sc = np.asarray(down_scale).view(np.uint8)
        sc = np.ascontiguousarray(sc[:, rank * (i_s // 16):(rank + 1) * (i_s // 16)].T)
        out["down_bs"] = quant.fp4_scales_to_chunked(sc, i_s).view(E4M3)
    return out


def nvfp4_expert_rank_arrays_slow(cfg, tp, rank, gate_up=None, gate_up_scale=None, down=None,
                                  down_scale=None):
    """Reference for `nvfp4_expert_rank_arrays` via explicit nibble unpacking/transposes."""
    i_s, i_full, hm = cfg.expert_hidden // tp, cfg.expert_hidden, cfg.moe_hidden
    cols = np.r_[rank * i_s:(rank + 1) * i_s, i_full + rank * i_s:i_full + (rank + 1) * i_s]
    out = {}
    if gate_up is not None:
        codes = quant.unpack_nvfp4_codes(gate_up)[cols].T  # [Hm, 2*Is]
        out["gate_up_fp4"] = quant.pack_fp4_rows(codes)
    if gate_up_scale is not None:
        gs = np.ascontiguousarray(np.asarray(gate_up_scale).view(np.uint8)[cols].T)  # [Hm/16, 2*Is]
        out["gate_up_bs"] = quant.fp4_scales_to_chunked(gs, hm).view(E4M3)
    if down is not None:
        dn = quant.unpack_nvfp4_codes(down)[:, rank * i_s:(rank + 1) * i_s].T  # [Is, Hm]
        out["down_fp4"] = quant.pack_fp4_rows(dn)
    if down_scale is not None:
        ds = np.asarray(down_scale).view(np.uint8)[:, rank * (i_s // 16):(rank + 1) * (i_s // 16)].T
        out["down_bs"] = quant.fp4_scales_to_chunked(np.ascontiguousarray(ds), i_s).view(E4M3)
    return out


def bf16_expert_to_nvfp4(cfg, tp, gate_up, down):
    """Quantize ONE bf16 expert (`gate_up [2I, Hm]`, `down [Hm, I]`, checkpoint layout) to the
    NVFP4 container format with `quant.quantize_nvfp4_np` (modelopt formula: one global scale
    per gate half / up half / down tensor from the FULL tensor's amax, like the vendor) ->
    `{rank: {family: array}}` for all ranks including `expert_gs`."""
    i_s, i_full, hm = cfg.expert_hidden // tp, cfg.expert_hidden, cfg.moe_hidden
    gu = np.ascontiguousarray(_bf16(gate_up).T, dtype=np.float32)  # [Hm, 2I]
    halves = [quant.quantize_nvfp4_np(gu[:, :i_full]), quant.quantize_nvfp4_np(gu[:, i_full:])]
    dn_packed, dn_bs, dn_gs = quant.quantize_nvfp4_np(
        np.ascontiguousarray(_bf16(down).T, dtype=np.float32)  # [I, Hm]
    )
    tile = expert_gs_tile(halves[0][2], halves[1][2], dn_gs)
    out = {}
    for rank in range(tp):
        cols = slice(rank * i_s, (rank + 1) * i_s)
        gu_packed = np.concatenate([halves[0][0][:, cols], halves[1][0][:, cols]], axis=1)
        gu_bs = np.concatenate([halves[0][1][:, cols], halves[1][1][:, cols]], axis=1)
        out[rank] = {
            "gate_up_fp4": np.ascontiguousarray(gu_packed),
            "gate_up_bs": quant.fp4_scales_to_chunked(np.ascontiguousarray(gu_bs), hm),
            "down_fp4": np.ascontiguousarray(dn_packed[rank * (i_s // 8):(rank + 1) * (i_s // 8)]),
            "down_bs": quant.fp4_scales_to_chunked(
                np.ascontiguousarray(dn_bs[rank * (i_s // 16):(rank + 1) * (i_s // 16)]), i_s
            ),
            "expert_gs": tile,
        }
    return out


# ---------------------------------------------------------------------------------------
# NVFP4 streaming converter
# ---------------------------------------------------------------------------------------


def safetensors_header(path):
    """`(header dict, data start offset)` of one safetensors file."""
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(size)), 8 + size


def _meta_from_header(path, header, base, key):
    entry = header[key]
    start, end = entry["data_offsets"]
    return (os.fspath(path), base + start, end - start, Checkpoint.DTYPES[entry["dtype"]],
            tuple(entry["shape"]))


def read_meta(meta, rows=None, threads=8):
    """A tensor (or its leading-axis slice `rows`) described by a `(path, offset, nbytes,
    dtype, shape)` tuple as a numpy array (positional reads, no mmap)."""
    path, offset, nbytes, dtype, shape = meta
    shape = list(shape)
    if rows is not None:
        r0, r1, step = rows.indices(shape[0])
        if step != 1:
            raise ValueError("only unit-step leading slices are supported")
        row_bytes = nbytes // shape[0]
        offset, nbytes, shape[0] = offset + r0 * row_bytes, (r1 - r0) * row_bytes, r1 - r0
    out = np.empty(shape, dtype)
    if nbytes:
        _read_ranges([(path, offset, memoryview(out.view(np.uint8).reshape(-1)))], threads)
    return out


def _lazy_tensors(metas):
    class Lazy(dict):
        def __missing__(self, suffix):
            self[suffix] = read_meta(metas[suffix])
            return self[suffix]

    return Lazy()


class ShardSource:
    """Shards of an HF checkpoint: a local directory (all files present, never deleted) or a
    Hub repo id (`fetch` downloads into the HF cache, `release` deletes the blob again)."""

    METADATA = ("config.json", "generation_config.json", "model.safetensors.index.json",
                "hf_quant_config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "chat_template.jinja")

    def __init__(self, src, revision=None, log=print):
        self.log = log
        self.local = Path(src).is_dir()
        self.sizes = {}
        if self.local:
            self.dir = Path(src)
            self.repo = None
        else:
            from huggingface_hub import HfApi, hf_hub_download

            self.repo, self._download = src, hf_hub_download
            info = HfApi().model_info(src, revision=revision, files_metadata=True)
            self.sizes = {f.rfilename: int(f.size or 0) for f in info.siblings}
            self.hub_revision = info.sha
            for name in self.METADATA:
                if name in self.sizes:
                    path = hf_hub_download(src, name, revision=self.hub_revision)
            self.dir = Path(path).parent
        self.index = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.revision = hashlib.sha256(
            (self.dir / "model.safetensors.index.json").read_bytes()).hexdigest()[:16]

    def size(self, filename):
        if self.local:
            return (self.dir / filename).stat().st_size
        return self.sizes.get(filename, 0)

    def present(self, filename):
        path = self.dir / filename
        return path.is_file() and (self.local or path.stat().st_size == self.size(filename))

    def fetch(self, filename):
        """Path of the shard, downloading it first in Hub mode (resumable)."""
        if self.local:
            return self.dir / filename
        return Path(self._download(self.repo, filename, revision=self.hub_revision))

    def release(self, filename):
        """Delete a downloaded shard (symlink and blob); local sources are untouched."""
        if self.local:
            return
        link = self.dir / filename
        if link.is_symlink() or link.exists():
            target = link.resolve()
            link.unlink(missing_ok=True)
            if target.exists() and target != link:
                target.unlink()
        blobs = self.dir.parent.parent / "blobs"
        if blobs.is_dir():
            for part in blobs.glob("*.incomplete"):
                if part.stat().st_mtime < time.time() - 3600:
                    part.unlink(missing_ok=True)


def _is_nvfp4_layer(index, layer):
    return layer_key(layer, NVFP4_SUFFIXES["gate_up_bs"]) in index


def nvfp4_units(cfg, index):
    """Conversion units of an NVFP4 checkpoint: `{unit id: (kind, layer, [keys])}`; every unit
    runs once all of its checkpoint tensors are on disk and writes a disjoint part of the
    container."""
    units = {}
    for name in GLOBAL_NAMES:
        units[f"global:{name}"] = ("global", name, [GLOBAL_KEYS[name]])
    for layer in range(cfg.layers):
        units[f"dense:{layer}"] = ("dense", layer, [layer_key(layer, s) for s in DENSE_SUFFIXES])
        if _is_nvfp4_layer(index, layer):
            for family in ("gate_up_fp4", "gate_up_bs", "down_fp4", "down_bs"):
                units[f"fp4:{layer}:{family}"] = (
                    "fp4", layer, [layer_key(layer, NVFP4_SUFFIXES[family])], family)
            units[f"gs:{layer}"] = ("gs", layer, [layer_key(layer, NVFP4_SUFFIXES["gate_up_gs"]),
                                                  layer_key(layer, NVFP4_SUFFIXES["down_gs"])])
        else:
            units[f"bf16:{layer}"] = ("bf16", layer, [layer_key(layer, s) for s in EXPERT_SUFFIXES])
    missing = sorted({k for u in units.values() for k in u[2] if k not in index})
    if missing:
        raise KeyError(f"checkpoint index lacks {len(missing)} tensors, e.g. {missing[:3]}")
    return units


def _worker_init_v2(cfg, tp, dst, doc):
    _SHARED.buffers, _SHARED.shms, _SHARED.plan = None, [], None
    _SHARED.cfg, _SHARED.tp, _SHARED.dst, _SHARED.layout, _SHARED.fds = cfg, tp, dst, doc, {}


def _fp4_task(args):
    """Vendor fp4 tensors of experts `e0:e1` of one family -> every rank's slice."""
    family, layer, e0, e1, meta = args
    cfg, tp = _SHARED.cfg, _SHARED.tp
    arg = FP4_TASK_ARGS[family]
    written = 0
    for e in range(e0, e1):
        tensor = read_meta(meta, slice(e, e + 1))[0]
        for rank in range(tp):
            value = nvfp4_expert_rank_arrays(cfg, tp, rank, **{arg: tensor})[family]
            written += _write_rank_array(rank, family, layer, value, expert=e)
    return written


def _bf16_expert_task(args):
    """bf16 experts `e0:e1` of a non-quantized layer -> NVFP4 (RTN) for every rank."""
    layer, e0, e1, meta_gu, meta_dn = args
    cfg, tp = _SHARED.cfg, _SHARED.tp
    written = 0
    for e in range(e0, e1):
        gu = read_meta(meta_gu, slice(e, e + 1))[0]
        dn = read_meta(meta_dn, slice(e, e + 1))[0]
        for rank, arrays in bf16_expert_to_nvfp4(cfg, tp, gu, dn).items():
            for name, value in arrays.items():
                written += _write_rank_array(rank, name, layer, value, expert=e)
    return written


def _dense_task_v2(args):
    layer, names, metas = args
    cfg, tp = _SHARED.cfg, _SHARED.tp
    tensors = _lazy_tensors(metas)
    written = 0
    for rank in range(tp):
        for name, value in dense_rank_arrays(cfg, tp, rank, tensors, names).items():
            written += _write_rank_array(rank, name, layer, value)
    return written


def _global_task(args):
    name, meta = args
    cfg, tp = _SHARED.cfg, _SHARED.tp
    written = 0
    vp = layout.vocab_pad(cfg, tp)
    for rank in range(tp):
        if name == "final_norm":
            value = _bf16(read_meta(meta)).reshape(1, -1)
        else:
            rows = slice(rank * vp, min((rank + 1) * vp, cfg.vocab))
            value = np.zeros((vp, cfg.hidden), BF16)
            if rows.start < cfg.vocab:
                value[:rows.stop - rows.start] = _bf16(read_meta(meta, rows, threads=16))
            if name == "lm_head":
                value = np.ascontiguousarray(value.T)
        data = _array_bytes(value)
        spec = _SHARED.layout["arrays"][name]
        if len(data) != spec["nbytes"]:
            raise ValueError(f"{name}: {len(data)} bytes != {spec['nbytes']}")
        _pwrite_all(_worker_fd(rank, name), data, 0)
        written += len(data)
    return written


def _free_bytes(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


FP4_TASK_ARGS = {"gate_up_fp4": "gate_up", "gate_up_bs": "gate_up_scale", "down_fp4": "down",
                 "down_bs": "down_scale"}


def _fp4_self_check(cfg, tp, dst, doc, layer, family, meta, experts=(0,)):
    """Compare the container's `family` of a few experts against the slow reference path
    recomputed from the source tensor; returns the mismatching (expert, rank) list."""
    bad = []
    for e in experts:
        tensor = read_meta(meta, slice(e, e + 1))[0]
        for rank in (0, tp - 1):
            want = nvfp4_expert_rank_arrays_slow(cfg, tp, rank, **{FP4_TASK_ARGS[family]: tensor})
            got = read_rank_array(dst, rank, family, (layer, e), doc, unpack=False)
            if not np.array_equal(got, np.asarray(want[family]).view(got.dtype)):
                bad.append(f"{family}[{layer},{e}]@rank{rank}")
    return bad


def convert_presharded_nvfp4(src, dst_dir, tp=8, workers=None, experts_per_task=4,
                             max_shards=3, min_free_bytes=8 << 30, shards=None, self_check=True,
                             config=None, log=print):
    """Stream the NVFP4 checkpoint `src` (a Hub repo id such as `NVFP4_REPO`, or a local
    directory holding every shard) into the format-v2 container `dst_dir`.

    Shards are processed in index order; in Hub mode each is downloaded into the HF cache
    (`HF_HOME`), converted and deleted, with at most `max_shards` shards on disk (a shard is
    kept until every unit that needs one of its tensors has run, e.g. a layer whose dense
    tensors straddle two shards) and a free-space guard (`min_free_bytes` beyond the
    container's unwritten bytes) before every download. `shards` limits the run to the first
    n shards (tests); `config` overrides the checkpoint's `Config` (tests). Progress is per
    unit in `progress.json`; re-running resumes. Returns the progress doc. Uses a spawned
    process pool (call from a guarded `__main__`).
    """
    started = time.perf_counter()
    source = ShardSource(src, log=log)
    if config is None:
        raw = json.loads((source.dir / "config.json").read_text())
        vocab = raw.get("text_config", raw)["vocab_size"]
        config = Config.from_checkpoint(source.dir, vocab_used=min(Config().vocab_used, vocab))
    cfg = config
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    layers_src = {str(l): ("vendor" if _is_nvfp4_layer(source.index, l) else "rtn")
                  for l in range(cfg.layers)}
    doc = make_layout(cfg, tp, revision=source.revision, expert_format="nvfp4",
                      expert_source=layers_src)
    if is_presharded(dst):
        existing = read_layout(dst)
        for key in ("format", "config", "tp", "arrays"):
            if existing.get(key) != doc[key]:
                raise ValueError(f"{dst}: existing container differs in {key}; use a new dst")
        if existing.get("revision") != doc["revision"]:
            raise ValueError(f"{dst}: existing container was converted from another revision")
    else:
        _write_json(dst / "layout.json", doc)
    for name in ShardSource.METADATA:
        if (source.dir / name).exists() and not (dst / name).exists():
            (dst / name).write_bytes((source.dir / name).read_bytes())
    _allocate_files(dst, doc)
    progress = read_progress(dst)
    progress.setdefault("units", [])
    progress.setdefault("bytes_written", 0)
    progress.setdefault("bytes_downloaded", 0)
    progress.setdefault("elapsed", 0.0)
    done = set(progress["units"])
    units = nvfp4_units(cfg, source.index)
    order = sorted({source.index[k] for u in units.values() for k in u[2]})
    if shards is not None:
        order = order[:shards]
    users = {f: [uid for uid, u in units.items() if any(source.index[k] == f for k in u[2])]
             for f in order}
    todo = [f for f in order if any(uid not in done for uid in users[f])]
    total_bytes = doc["total_bytes"]
    log(f"convert nvfp4 {src} -> {dst}: tp={tp}, {len(units)} units ({len(done)} done), "
        f"{len(todo)}/{len(order)} shards to process, container {total_bytes / 1e9:.1f} GB, "
        f"free {_free_bytes(dst) / 1e9:.1f} GB")

    def save_progress():
        progress["units"] = sorted(done)
        progress["layers"] = sorted(
            l for l in range(cfg.layers)
            if all(uid in done for uid, u in units.items() if u[0] != "global" and u[1] == l))
        progress["globals"] = all(f"global:{n}" in done for n in GLOBAL_NAMES)
        progress["complete"] = len(done) == len(units)
        _write_json(dst / "progress.json", progress)

    if not todo:
        save_progress()
        return progress

    # Prefetcher: downloads shards ahead of the converter, bounded by the on-disk count and
    # the free-space guard; the converter's `need` event bypasses the count bound.
    lock = threading.Condition()
    on_disk, fetched, failure = {}, {}, []
    need = {"shard": None}

    def fetch_all():
        try:
            for f in todo:
                with lock:
                    while (len(on_disk) + 1 > max_shards and need["shard"] != f):
                        lock.wait(1.0)
                    while True:
                        free = _free_bytes(dst) - (total_bytes - progress["bytes_written"])
                        if source.present(f) or free >= source.size(f) + min_free_bytes:
                            break
                        log(f"waiting for disk space: {free / 1e9:.1f} GB free beyond the "
                            f"container, need {(source.size(f) + min_free_bytes) / 1e9:.1f} GB")
                        lock.wait(30.0)
                t0 = time.perf_counter()
                path = source.fetch(f)
                dt = time.perf_counter() - t0
                with lock:
                    on_disk[f] = path
                    fetched[f] = (dt, source.size(f))
                    lock.notify_all()
        except BaseException as error:  # noqa: BLE001
            failure.append(error)
            with lock:
                lock.notify_all()

    workers = workers or max(1, min(os.cpu_count() - 8, 128))
    ctx = multiprocessing.get_context("spawn")
    init_args = (cfg, tp, dst, doc)
    e_chunks = [(e, min(e + experts_per_task, cfg.experts))
                for e in range(0, cfg.experts, experts_per_task)]
    headers = {}
    shard_times, written_total = [], 0

    def meta(key):
        f = source.index[key]
        if f not in headers:
            headers[f] = safetensors_header(on_disk[f])
        header, base = headers[f]
        return _meta_from_header(on_disk[f], header, base, key)

    threading.Thread(target=fetch_all, daemon=True).start()
    try:
        with ctx.Pool(workers, initializer=_worker_init_v2, initargs=init_args) as pool:
            _worker_init_v2(*init_args)
            for i, f in enumerate(todo):
                with lock:
                    need["shard"] = f
                    lock.notify_all()
                    while f not in on_disk and not failure:
                        lock.wait(1.0)
                    if failure:
                        raise failure[0]
                    need["shard"] = None
                t0 = time.perf_counter()
                runnable = [uid for uid in units if uid not in done
                            and all(source.index[k] in on_disk for k in units[uid][2])]
                results, written = [], 0
                for uid in runnable:
                    kind, layer, keys, *rest = units[uid]
                    if kind == "fp4":
                        family = rest[0]
                        m = meta(keys[0])
                        results += [pool.apply_async(_fp4_task, ((family, layer, e0, e1, m),))
                                    for e0, e1 in e_chunks]
                    elif kind == "bf16":
                        m_gu, m_dn = meta(keys[0]), meta(keys[1])
                        results += [pool.apply_async(_bf16_expert_task, ((layer, e0, e1, m_gu, m_dn),))
                                    for e0, e1 in e_chunks]
                    elif kind == "dense":
                        metas = {s: meta(layer_key(layer, s)) for s in DENSE_SUFFIXES}
                        results += [pool.apply_async(_dense_task_v2, ((layer, names, metas),))
                                    for names in DENSE_TASKS]
                        written += _dense_task_v2((layer, VECTOR_NAMES, metas))
                    elif kind == "global":
                        results.append(pool.apply_async(_global_task, ((layer, meta(keys[0])),)))
                    elif kind == "gs":
                        gu_gs = read_meta(meta(keys[0])).astype(np.float32).reshape(cfg.experts, 2)
                        dn_gs = read_meta(meta(keys[1])).astype(np.float32).reshape(cfg.experts)
                        tiles = np.stack([expert_gs_tile(gu_gs[e, 0], gu_gs[e, 1], dn_gs[e])
                                          for e in range(cfg.experts)])
                        for rank in range(tp):
                            written += _write_rank_array(rank, "expert_gs", layer, tiles)
                written += sum(r.get() for r in results)
                for fd in _SHARED.fds.values():
                    os.fsync(fd)
                if self_check:
                    checked = []
                    for uid in runnable:
                        kind, layer, keys, *rest = units[uid]
                        if kind == "fp4":
                            bad = _fp4_self_check(cfg, tp, dst, doc, layer, rest[0], meta(keys[0]))
                            if bad:
                                raise RuntimeError(f"self-check failed: {bad}")
                            checked.append(f"{layer}:{rest[0]}")
                    if checked:
                        log(f"fp4 self-check ok (expert 0, ranks 0/{tp - 1}): {' '.join(checked)}")
                done.update(runnable)
                written_total += written
                progress["bytes_written"] += written
                now = time.perf_counter()
                dl_time, size = fetched.get(f, (0.0, 0))
                progress["bytes_downloaded"] += size
                shard_times.append(now - t0 + dl_time)
                progress["elapsed"] += now - t0 + dl_time
                save_progress()
                with lock:
                    for g in list(on_disk):
                        if all(uid in done for uid in users.get(g, [])):
                            source.release(g)
                            del on_disk[g]
                    lock.notify_all()
                remaining = sum(source.size(g) for g in todo[i + 1:])
                rate = progress["bytes_downloaded"] / max(progress["elapsed"], 1e-9)
                log(f"shard {f} ({size / 1e9:.1f} GB, download {dl_time:.0f} s = "
                    f"{size / max(dl_time, 1e-9) / 1e6:.0f} MB/s): {len(runnable)} units, "
                    f"wrote {written / 1e9:.1f} GB in {now - t0:.0f} s; {i + 1}/{len(todo)} shards, "
                    f"{len(done)}/{len(units)} units, {progress['bytes_written'] / 1e9:.0f}/"
                    f"{total_bytes / 1e9:.0f} GB written, {len(on_disk)} shards on disk, "
                    f"avg {rate / 1e6:.0f} MB/s, ETA {remaining / max(rate, 1e-9) / 60:.0f} min, "
                    f"free {_free_bytes(dst) / 1e9:.0f} GB")
    finally:
        for fd in _SHARED.fds.values():
            os.close(fd)
        _SHARED.fds = {}
    save_progress()
    log(f"done: {len(done)}/{len(units)} units, complete={progress['complete']}, "
        f"{progress['elapsed'] / 60:.1f} min total, {time.perf_counter() - started:.0f} s this run")
    return progress


# ---------------------------------------------------------------------------------------
# Verification helpers and CLI
# ---------------------------------------------------------------------------------------


def verify_layer(src_dir, dst_dir, layer=0, experts=(0,), log=print):
    """Compare the container's layer against the checkpoint: dense spot checks (exact) and
    the relative RMS error of the dequantized experts (int4 or nvfp4 container; `src_dir` may
    be the bf16 checkpoint, whose dense tensors and layer-0/61 experts the NVFP4 repo shares,
    so for a vendor-quantized layer the error is the vendor's own NVFP4 quantization error).
    Returns a dict of numbers."""
    checkpoint = Checkpoint(src_dir)
    doc = read_layout(dst_dir)
    cfg, tp = config_from_layout(doc), doc["tp"]
    qw, i_s, i_full = cfg.q_width // tp, cfg.expert_hidden // tp, cfg.expert_hidden
    results = {}
    q_proj = checkpoint.read(layer_key(layer, "self_attn.q_proj.weight"))
    o_proj = checkpoint.read(layer_key(layer, "self_attn.o_proj.weight"))
    bits = np.uint16
    for rank in range(tp):
        q = read_rank_array(dst_dir, rank, "q", layer, doc)
        o = read_rank_array(dst_dir, rank, "o", layer, doc)
        ok_q = np.array_equal(q.view(bits), q_proj[rank * qw:(rank + 1) * qw].T.view(bits))
        ok_o = np.array_equal(o.view(bits), o_proj[:, rank * qw:(rank + 1) * qw].T.view(bits))
        results[f"rank{rank}_q_o_exact"] = bool(ok_q and ok_o)
    row0 = read_rank_array(dst_dir, 0, "q", layer, doc)[:, 0]
    results["q_row0_equals_q_proj_row0"] = bool(
        np.array_equal(row0.view(bits), q_proj[0].view(bits)))

    fp4 = layout_expert_format(doc) == "nvfp4"

    def dequant(rank, kind, e):
        if not fp4:
            q = read_rank_array(dst_dir, rank, kind + "_q", (layer, e), doc)
            s = read_rank_array(dst_dir, rank, kind + "_s", (layer, e), doc)
            return quant.dequantize_int4(q, s)
        packed = read_rank_array(dst_dir, rank, kind + "_fp4", (layer, e), doc)
        bs = read_rank_array(dst_dir, rank, kind + "_bs", (layer, e), doc)
        gs = read_rank_array(dst_dir, rank, "expert_gs", (layer, e), doc)
        if kind == "gate_up":
            scale = np.concatenate([np.full(i_s, gs[0, 0]), np.full(i_s, gs[1, 0])])[None]
        else:
            scale = gs[2, 0]
        return quant.dequant_fp4_np(packed, bs, scale.astype(np.float32))

    errors = []
    for e in experts:
        gu = checkpoint.read(layer_key(layer, "mlp.experts.gate_up_proj"), slice(e, e + 1))[0]
        dn = checkpoint.read(layer_key(layer, "mlp.experts.down_proj"), slice(e, e + 1))[0]
        parts = [dequant(r, "gate_up", e) for r in range(tp)]  # [Hm, 2*Is] each
        gu_full = np.concatenate([p[:, :i_s] for p in parts] + [p[:, i_s:] for p in parts], axis=1)
        dn_full = np.concatenate([dequant(r, "down", e) for r in range(tp)], axis=0)  # [I, Hm]
        for name, ref, got in (("gate_up", _f32(gu).T, gu_full), ("down", _f32(dn).T, dn_full)):
            err = float(np.sqrt(np.mean((ref - got) ** 2)) / np.sqrt(np.mean(ref ** 2)))
            results[f"expert{e}_{name}_rel_rms_error"] = err
            errors.append(err)
            log(f"layer {layer} expert {e} {name}: relative RMS error {err:.4%}")
    results["max_rel_rms_error"] = max(errors) if errors else None
    log(f"layer {layer}: q row 0 == q_proj[0]: {results['q_row0_equals_q_proj_row0']}, "
        f"dense q/o exact on all ranks: {all(results[f'rank{r}_q_o_exact'] for r in range(tp))}")
    return results


def _cli(argv=None):
    parser = argparse.ArgumentParser(prog="python -m musespark.load")
    sub = parser.add_subparsers(dest="command", required=True)
    conv = sub.add_parser("convert", help="convert an HF checkpoint into the pre-sharded container")
    conv.add_argument("--src", required=True)
    conv.add_argument("--dst", required=True)
    conv.add_argument("--tp", type=int, default=8)
    conv.add_argument("--group", type=int, default=128)
    conv.add_argument("--layers", type=str, default=None, help="e.g. '0' or '0,1,5'")
    conv.add_argument("--workers", type=int, default=None)
    conv.add_argument("--read-threads", type=int, default=READ_THREADS)
    nv = sub.add_parser("convert-nvfp4", help="stream the NVFP4 Hub checkpoint into a v2 container")
    nv.add_argument("--src", default=NVFP4_REPO, help="Hub repo id or a local checkpoint directory")
    nv.add_argument("--dst", required=True)
    nv.add_argument("--tp", type=int, default=8)
    nv.add_argument("--workers", type=int, default=None)
    nv.add_argument("--max-shards", type=int, default=3, help="shards kept on disk at once")
    nv.add_argument("--shards", type=int, default=None, help="only the first n shards (tests)")
    nv.add_argument("--min-free-gb", type=float, default=8.0)
    qd = sub.add_parser("quantize-dense",
                        help="add int8 per-output-channel dense projections to a container")
    qd.add_argument("--dir", required=True)
    qd.add_argument("--workers", type=int, default=None)
    qd.add_argument("--layers", type=str, default=None, help="e.g. '0' or '0,1,5'")
    qd.add_argument("--no-publish", action="store_true",
                    help="write the files but leave layout.json on bf16 (publish with a later run)")
    ver = sub.add_parser("verify", help="spot-check a converted layer against the checkpoint")
    ver.add_argument("--src", required=True)
    ver.add_argument("--dst", required=True)
    ver.add_argument("--layer", type=int, default=0)
    ver.add_argument("--experts", type=str, default="0")
    args = parser.parse_args(argv)

    def log(message):
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    if args.command == "convert":
        layers = None if args.layers is None else [int(x) for x in args.layers.split(",")]
        progress = convert_presharded(args.src, args.dst, tp=args.tp, group=args.group,
                                      layers=layers, workers=args.workers,
                                      read_threads=args.read_threads, log=log)
        return 0 if progress["layers"] else 1
    if args.command == "convert-nvfp4":
        progress = convert_presharded_nvfp4(
            args.src, args.dst, tp=args.tp, workers=args.workers, max_shards=args.max_shards,
            shards=args.shards, min_free_bytes=int(args.min_free_gb * 2**30), log=log)
        return 0 if progress["complete"] else 1
    if args.command == "quantize-dense":
        layers = None if args.layers is None else [int(x) for x in args.layers.split(",")]
        progress = quantize_dense(args.dir, workers=args.workers, layers=layers, log=log,
                                  publish=not args.no_publish)
        return 0 if progress["dense_int8"]["complete"] else 1
    results = verify_layer(args.src, args.dst, args.layer,
                           [int(x) for x in args.experts.split(",")], log)
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
