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
PREFIX = "model.language_model."
FORMAT = "musespark-presharded-v1"
INT4_ENCODING = (
    "uint8, two int4 values per byte along the last axis, low nibble = even index; "
    "value = ((byte >> (4 * (i % 2))) & 0xF) sign-extended from 4 bits"
)

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
GLOBAL_KEYS = {
    "embed": PREFIX + "embed_tokens.weight",
    "lm_head": "lm_head.weight",
    "final_norm": PREFIX + "norm.weight",
}
EXPERT_NAMES = ("gate_up_q", "gate_up_s", "down_q", "down_s")
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
              "U8": np.dtype(np.uint8), "I8": np.dtype(np.int8)}

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


def _shapes(cfg, tp):
    """`{name: (per-rank shape, np dtype)}` from `layout.rank_shapes`."""
    out = {}
    for name, (shape, dtype) in layout.rank_shapes(cfg, tp).items():
        out[name] = (tuple(int(s) for s in shape), np.dtype(dtype))
    return out


def disk_spec(shape, dtype):
    """`(disk shape, disk dtype)`: int4 is nibble-packed along the last axis."""
    if np.dtype(dtype) == INT4:
        if shape[-1] % 2:
            raise ValueError("int4 arrays need an even last axis to pack")
        return (*shape[:-1], shape[-1] // 2), np.dtype(np.uint8)
    return tuple(shape), np.dtype(dtype)


def _dtype_name(dtype):
    dtype = np.dtype(dtype)
    return {BF16: "bfloat16", INT4: "int4"}.get(dtype, dtype.name)


def _np_dtype(name):
    return {"bfloat16": BF16, "int4": INT4}.get(name, np.dtype(name))


def make_layout(cfg, tp, revision=None):
    """The `layout.json` document of a container for `cfg`/`tp`."""
    arrays = {}
    for name, (shape, dtype) in _shapes(cfg, tp).items():
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
    return {
        "format": FORMAT,
        "config": cfg_dict,
        "tp": tp,
        "group": cfg.group_size,
        "revision": revision,
        "int4": INT4_ENCODING,
        "ranks": [f"rank{r}" for r in range(tp)],
        "arrays": arrays,
        "total_bytes": tp * sum(a["nbytes"] for a in arrays.values()),
    }


def read_layout(directory):
    directory = Path(directory)
    doc = json.loads((directory / "layout.json").read_text())
    if doc.get("format") != FORMAT:
        raise ValueError(f"{directory}: unexpected container format {doc.get('format')!r}")
    return doc


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


def _load_rank(directory, doc, rank, device, layer_chunk_bytes, log):
    """All arrays of one rank as single-device arrays on `device`."""
    placement = SingleDeviceSharding(device)
    arrays = {}
    for name, spec in doc["arrays"].items():
        path = rank_file(directory, rank, name)
        shape, dtype = tuple(spec["shape"]), _np_dtype(spec["dtype"])
        dshape, ddtype = tuple(spec["disk_shape"]), np.dtype(spec["disk_dtype"])
        per_layer = int(np.prod(dshape[1:])) * ddtype.itemsize if spec["layer_axis"] == 0 else 0
        if dtype != INT4:
            value = _read_file(path, dtype=ddtype).reshape(dshape)
            arrays[name] = jax.device_put(value[None], device)
            del value
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
                    log=None):
    """The container at `directory` as `{name: jax.Array[tp, ...]}` sharded `P("tp")` over
    `mesh` (rank r's arrays live on `mesh.devices.flat[r]`). int4 arrays are `jnp.int4`."""
    directory = Path(directory)
    doc = read_layout(directory)
    if not read_progress(directory).get("complete"):
        raise ValueError(f"{directory}: conversion is not complete (see progress.json)")
    tp = doc["tp"]
    if cfg is not None and make_layout(cfg, tp)["arrays"] != doc["arrays"]:
        raise ValueError(f"{directory}: container layout does not match the requested config")
    devices = list(mesh.devices.flat)
    if len(devices) != tp:
        raise ValueError(f"mesh has {len(devices)} devices, container has tp={tp}")
    log = log or (lambda _: None)
    started = time.perf_counter()
    per_rank = {}
    with ThreadPoolExecutor(max(1, ranks_in_flight)) as pool:
        futures = {r: pool.submit(_load_rank, directory, doc, r, devices[r], layer_chunk_bytes, log)
                   for r in range(tp)}
        for r, future in futures.items():
            per_rank[r] = future.result()
            log(f"rank {r} resident after {time.perf_counter() - started:.0f} s")
    sharding = NamedSharding(mesh, P("tp"))
    weights = {}
    for name, spec in doc["arrays"].items():
        weights[name] = jax.make_array_from_single_device_arrays(
            (tp, *spec["shape"]), sharding, [per_rank[r][name] for r in range(tp)]
        )
    log(f"{doc['total_bytes'] / 1e9:.0f} GB resident after {time.perf_counter() - started:.0f} s "
        f"({doc['total_bytes'] / 1e9 / (time.perf_counter() - started):.2f} GB/s)")
    return weights


def abstract_weights(mesh, cfg, tp=None):
    """`jax.ShapeDtypeStruct` tree matching `load_presharded`, for AOT compilation."""
    tp = tp or mesh.size
    sharding = NamedSharding(mesh, P("tp"))
    out = {}
    for name, (shape, dtype) in _shapes(cfg, tp).items():
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


def checkpoint_tensor_specs(cfg):
    """`{name: (shape, np dtype)}` of every text-model tensor of an HF checkpoint for `cfg`."""
    h, hm, i, e = cfg.hidden, cfg.moe_hidden, cfg.expert_hidden, cfg.experts
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


def random_checkpoint_tensors(cfg, seed=0):
    """Deterministic random HF-named tensors for `cfg` (bf16 weights ~N(0, 0.05); the
    router f32 with a non-bf16-representable fraction; norms/gates around zero)."""
    rng = np.random.default_rng(seed)
    tensors = {}
    for name, (shape, dtype) in checkpoint_tensor_specs(cfg).items():
        if dtype == np.float32:
            value = rng.standard_normal(shape, np.float32) * np.float32(0.05)
        elif len(shape) == 1:
            value = rng.standard_normal(shape, np.float32) * np.float32(0.5)
        else:
            value = rng.standard_normal(shape, np.float32) * np.float32(0.05)
        tensors[name] = value.astype(dtype)
    return tensors


def write_checkpoint(directory, cfg, tensors, shards=3):
    """Write `tensors` as an HF safetensors checkpoint (`model-0000k-of-0000n.safetensors` +
    `model.safetensors.index.json` + a `config.json` that `Config.from_checkpoint` accepts)."""
    from safetensors.numpy import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = sorted(tensors)
    weight_map, total = {}, 0
    for k in range(shards):
        filename = f"model-{k + 1:05d}-of-{shards:05d}.safetensors"
        part = {n: tensors[n] for n in names[k::shards]}
        save_file(part, str(directory / filename))
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


# ---------------------------------------------------------------------------------------
# Verification helpers and CLI
# ---------------------------------------------------------------------------------------


def verify_layer(src_dir, dst_dir, layer=0, experts=(0,), log=print):
    """Compare the container's layer against the checkpoint: dense spot checks (exact) and
    the relative RMS error of the dequantized int4 experts. Returns a dict of numbers."""
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

    def dequant(rank, kind, e):
        q = read_rank_array(dst_dir, rank, kind + "_q", (layer, e), doc)
        s = read_rank_array(dst_dir, rank, kind + "_s", (layer, e), doc)
        return quant.dequantize_int4(q, s)

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
    results = verify_layer(args.src, args.dst, args.layer,
                           [int(x) for x in args.experts.split(",")], log)
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
