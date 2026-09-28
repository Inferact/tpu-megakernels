"""Fast-start pre-sharded loader and round-trip reconstruction for DeepSeek-V4.1-Flash.

Loads pre-sharded `rank{00..N-1}/dense.bin` + `index.json` + `engram/layer_*.bin`
into JAX device arrays on a 1-D `"tp"` mesh (`world_size in {8, 32}`) using zero-copy
`mmap(MAP_POPULATE)` views and parallel per-device transfers.

Also provides `reconstruct_unsharded_weights` to reassemble full unsharded checkpoint
tensors from either `tp8` or `tp32` pre-sharded arrays for bit-exact verification.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import mmap
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from deepseek_v41.checkpoint import (
    BF16,
    F8_E4M3,
    SAFETENSORS_TO_NP_DTYPE,
)
from deepseek_v41.collectives import create_v7x_mesh
from deepseek_v41.config import DSV41Config
from deepseek_v41.preshard import PAGE_ALIGN, preshard_checkpoint
from deepseek_v41.quant import (
    dequant_fp8_block32x32,
    dequant_mxfp4_block32,
    unpack_mxfp4_v7x_bitcast,
)
from deepseek_v41.rope import precompute_layer_rope_tables


PRESHARD_DTYPE_TO_NP: dict[str, np.dtype] = {
    **SAFETENSORS_TO_NP_DTYPE,
    "U32": np.dtype(np.uint32),
}


class PreshardedWeightsDict(dict[str, Any]):
    """Dictionary of sharded `jax.Array` weights carrying `.config`, `.layout`, and `.engram_host_tables`."""

    config: DSV41Config
    layout: dict[str, Any]
    engram_host_tables: dict[int, dict[str, np.ndarray]]
    _mmaps: list[mmap.mmap]
    _files: list[Any]


def _flatten_presharded_dict(d: PreshardedWeightsDict) -> tuple[list[Any], tuple[str, ...]]:
    keys = tuple(sorted(d.keys()))
    return [d[k] for k in keys], keys


def _unflatten_presharded_dict(keys: tuple[str, ...], vals: list[Any]) -> dict[str, Any]:
    return dict(zip(keys, vals))


jax.tree_util.register_pytree_node(
    PreshardedWeightsDict,
    _flatten_presharded_dict,
    _unflatten_presharded_dict,
)


def is_presharded(path: str | os.PathLike[str]) -> bool:
    """Returns True if `path` (local directory or `gs://` prefix) contains `layout.json`."""
    path_str = str(path).rstrip("/")
    if path_str.startswith("gs://"):
        try:
            from google.cloud import storage  # type: ignore[import-untyped]

            without = path_str[len("gs://") :]
            bucket_name, _, prefix = without.partition("/")
            blob_name = f"{prefix}/layout.json" if prefix else "layout.json"
            return bool(storage.Client().bucket(bucket_name).blob(blob_name).exists())
        except Exception:
            gcloud = shutil.which("gcloud")
            if gcloud is not None:
                res = subprocess.run(
                    [gcloud, "storage", "ls", f"{path_str}/layout.json"],
                    check=False,
                    capture_output=True,
                )
                return res.returncode == 0
            return False
    return (Path(path_str) / "layout.json").is_file()


def ensure_presharded(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    *,
    world_size: int = 32,
    workers: int = 8,
    cfg: DSV41Config | None = None,
) -> str:
    """Checks `dst/layout.json`; if absent, runs `preshard_checkpoint(src, dst, ...)`."""
    if not is_presharded(dst):
        preshard_checkpoint(
            src,
            dst,
            world_size=world_size,
            workers=workers,
            cfg=cfg,
        )
    return str(dst)


_SUBPROC_RANGE_DL_CODE = """
import os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from google.cloud import storage
from requests.adapters import HTTPAdapter

bucket_name, blob_name, dst_path = sys.argv[1], sys.argv[2], sys.argv[3]
chunk_bytes = int(sys.argv[4])
workers = int(sys.argv[5])
start_offset = int(sys.argv[6]) if len(sys.argv) > 6 else 0
valid_bytes = int(sys.argv[7]) if len(sys.argv) > 7 else -1
total_bytes = int(sys.argv[8]) if len(sys.argv) > 8 else -1
pad_byte = int(sys.argv[9]) if len(sys.argv) > 9 else 0

client = storage.Client()
adapter = HTTPAdapter(pool_connections=max(32, workers * 2), pool_maxsize=max(32, workers * 2))
client._http.mount("https://", adapter)
client._http.mount("http://", adapter)
bucket = client.bucket(bucket_name)
if valid_bytes < 0:
    blob = bucket.get_blob(blob_name)
    if blob is None:
        raise FileNotFoundError(f"Missing GCS blob gs://{bucket_name}/{blob_name}")
    valid_bytes = int(blob.size or 0) - start_offset
if total_bytes < 0:
    total_bytes = valid_bytes

Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
part_path = dst_path + ".part"
if start_offset == 0 and total_bytes == valid_bytes and valid_bytes <= chunk_bytes:
    bucket.blob(blob_name).download_to_filename(part_path)
    os.replace(part_path, dst_path)
    sys.exit(0)
fd = os.open(part_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
try:
    os.ftruncate(fd, total_bytes)
    if total_bytes > valid_bytes and pad_byte != 0:
        os.pwrite(fd, bytes([pad_byte]) * (total_bytes - valid_bytes), valid_bytes)
    spans = [(s, min(valid_bytes, s + chunk_bytes)) for s in range(0, valid_bytes, chunk_bytes)]
    def _fetch(span):
        s, e = span
        data = bucket.blob(blob_name).download_as_bytes(
            start=start_offset + s, end=start_offset + e - 1, raw_download=True
        )
        os.pwrite(fd, data, s)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(_fetch, spans))
finally:
    os.close(fd)
os.replace(part_path, dst_path)
"""


def _download_blob_parallel_ranges(
    bucket_name: str,
    blob_name: str,
    dst_path: str,
    *,
    chunk_bytes: int = 64 * 1024 * 1024,
    workers: int = 24,
) -> None:
    """Downloads a GCS blob into `dst_path` using parallel byte-range requests and atomic rename."""
    from google.cloud import storage  # type: ignore[import-untyped]
    from requests.adapters import HTTPAdapter

    client = storage.Client()
    adapter = HTTPAdapter(pool_connections=max(32, workers * 2), pool_maxsize=max(32, workers * 2))
    client._http.mount("https://", adapter)
    client._http.mount("http://", adapter)
    bucket = client.bucket(bucket_name)
    blob = bucket.get_blob(blob_name)
    if blob is None:
        raise FileNotFoundError(f"Missing GCS blob gs://{bucket_name}/{blob_name}")
    size = int(blob.size or 0)
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    part_path = f"{dst_path}.part"
    if size <= chunk_bytes:
        blob.download_to_filename(part_path)
        os.replace(part_path, dst_path)
        return
    fd = os.open(part_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, size)
        spans = [(s, min(size, s + chunk_bytes)) for s in range(0, size, chunk_bytes)]

        def _fetch(span: tuple[int, int]) -> None:
            s, e = span
            data = bucket.blob(blob_name).download_as_bytes(start=s, end=e - 1, raw_download=True)
            os.pwrite(fd, data, s)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(_fetch, spans))
    finally:
        os.close(fd)
    os.replace(part_path, dst_path)


class _GcsRowTable:
    """Array-like wrapper over a GCS row-major binary table (`[N, cols]`) with row-level caching."""

    def __init__(
        self,
        bucket_name: str,
        blob_name: str,
        shape: tuple[int, ...],
        dtype: np.dtype,
        *,
        base_offset: int = 0,
    ) -> None:
        from google.cloud import storage  # type: ignore[import-untyped]
        from requests.adapters import HTTPAdapter

        client = storage.Client()
        adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32)
        client._http.mount("https://", adapter)
        client._http.mount("http://", adapter)
        self.bucket = client.bucket(bucket_name)
        self.blob_name = blob_name
        self.base_offset = int(base_offset)
        self.shape = tuple(int(x) for x in shape)
        self.dtype = np.dtype(dtype)
        self._row_elems = int(np.prod(self.shape[1:])) if len(self.shape) > 1 else 1
        self._row_bytes = self._row_elems * self.dtype.itemsize
        self._cache: dict[int, np.ndarray] = {}

    def __getitem__(self, idx: Any) -> np.ndarray:
        ids = np.asarray(idx, dtype=np.int64)
        flat = [int(x) for x in np.unique(ids.reshape(-1))]
        missing = [r for r in flat if r not in self._cache]
        if missing:
            def _get_row(r: int) -> tuple[int, np.ndarray]:
                s = self.base_offset + r * self._row_bytes
                e = s + self._row_bytes - 1
                raw = self.bucket.blob(self.blob_name).download_as_bytes(start=s, end=e, raw_download=True)
                arr = np.frombuffer(raw, dtype=self.dtype).copy().reshape(self.shape[1:])
                return r, arr

            with ThreadPoolExecutor(max_workers=min(24, len(missing))) as pool:
                for r, arr in pool.map(_get_row, missing):
                    self._cache[r] = arr
        out = np.empty((*ids.shape, *self.shape[1:]), dtype=self.dtype)
        flat_out = out.reshape(-1, *self.shape[1:])
        for i, r in enumerate(ids.reshape(-1)):
            flat_out[i] = self._cache[int(r)]
        return out

    def read_prefix_rows(self, max_rows: int = 65536) -> np.ndarray:
        n_rows = min(int(max_rows), int(self.shape[0]))
        n_bytes = n_rows * self._row_bytes
        raw = self.bucket.blob(self.blob_name).download_as_bytes(
            start=self.base_offset,
            end=self.base_offset + n_bytes - 1,
            raw_download=True,
        )
        return np.frombuffer(raw, dtype=self.dtype).copy().reshape(n_rows, *self.shape[1:])


def stage_from_gcs(
    gcs_uri: str,
    stage_dir: str | os.PathLike[str] = "/dev/shm/preshard",
    *,
    local_ranks: list[int] | None = None,
    stage_engram: bool = False,
) -> Path:
    """Stages `layout.json` and `rank{r:02d}/*` (for `local_ranks`) from GCS into `stage_dir`."""
    import sys

    gcs_uri = gcs_uri.rstrip("/")
    dst_root = Path(stage_dir)
    dst_root.mkdir(parents=True, exist_ok=True)

    from google.cloud import storage  # type: ignore[import-untyped]

    without = gcs_uri[len("gs://") :]
    bucket_name, _, prefix = without.partition("/")
    prefix_slash = f"{prefix.rstrip('/')}/" if prefix else ""
    bucket = storage.Client().bucket(bucket_name)

    layout_local = dst_root / "layout.json"
    bucket.blob(prefix_slash + "layout.json").download_to_filename(str(layout_local))
    layout = json.loads(layout_local.read_text())
    ranks_to_fetch = (
        local_ranks if local_ranks is not None else list(range(int(layout["world_size"])))
    )
    rels: list[str] = []
    for r in ranks_to_fetch:
        rels.append(f"rank{r:02d}/index.json")
        rels.append(f"rank{r:02d}/dense.bin")
    if stage_engram:
        for _, info in layout.get("engram", {}).items():
            if "weight_gcs_uri" not in info:
                rels.append(info["weight_file"])
            if "scale_gcs_uri" not in info:
                rels.append(info["scale_file"])

    def _dl_one(rel_path: str) -> None:
        target = dst_root / rel_path
        if target.is_file() and target.stat().st_size > 0 and not Path(f"{target}.part").exists():
            return
        if rel_path.endswith(".bin"):
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _SUBPROC_RANGE_DL_CODE,
                    bucket_name,
                    prefix_slash + rel_path,
                    str(target),
                    str(64 * 1024 * 1024),
                    "24",
                ],
                check=True,
            )
        else:
            _download_blob_parallel_ranges(
                bucket_name,
                prefix_slash + rel_path,
                str(target),
                workers=8,
            )

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(rels)))) as pool:
        list(pool.map(_dl_one, rels))
    return dst_root


def _mmap_file_ro(path: Path, *, pin_memory: bool = False) -> tuple[Any, mmap.mmap]:
    """Opens `path` and returns `(file_obj, mmap_obj)` mapped with `MAP_POPULATE` if supported."""
    f = open(path, "rb")
    size = os.fstat(f.fileno()).st_size
    flags = mmap.MAP_SHARED | getattr(mmap, "MAP_POPULATE", 0)
    mm = mmap.mmap(f.fileno(), size, flags=flags, prot=mmap.PROT_READ)
    if pin_memory:
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            arr_u8 = np.ndarray(shape=(size,), dtype=np.uint8, buffer=mm, offset=0)
            libc.mlock(ctypes.c_void_p(arr_u8.ctypes.data), ctypes.c_size_t(size))
        except Exception:
            pass
    return f, mm


def _parse_gcs_blob_uri(uri: str) -> tuple[str, str]:
    without = uri.rstrip("/")[len("gs://") :]
    b_name, _, b_path = without.partition("/")
    return b_name, b_path


def load_engram_host_tables(
    path: str | os.PathLike[str],
    *,
    pin_memory: bool = True,
    gcs_uri: str | None = None,
) -> tuple[dict[int, dict[str, Any]], list[Any], list[mmap.mmap]]:
    """Memory-maps local `engram/layer_{id:02d}.{weight,scale}.bin` or wraps GCS blobs lazily."""
    root = Path(path)
    layout_path = root / "layout.json"
    if not layout_path.is_file():
        return {}, [], []
    layout = json.loads(layout_path.read_text())
    engram_meta = layout.get("engram", {})
    tables: dict[int, dict[str, Any]] = {}
    files: list[Any] = []
    mmaps: list[mmap.mmap] = []

    bucket_name = ""
    prefix_slash = ""
    if gcs_uri and gcs_uri.startswith("gs://"):
        without = gcs_uri.rstrip("/")[len("gs://") :]
        bucket_name, _, prefix = without.partition("/")
        prefix_slash = f"{prefix.rstrip('/')}/" if prefix else ""

    for layer_str, info in sorted(engram_meta.items(), key=lambda kv: int(kv[0])):
        layer_id = int(layer_str)
        w_path = root / info["weight_file"]
        s_path = root / info["scale_file"]
        w_dt = PRESHARD_DTYPE_TO_NP[info["weight_dtype"]]
        s_dt = PRESHARD_DTYPE_TO_NP[info["scale_dtype"]]
        if w_path.is_file() and s_path.is_file():
            w_f, w_mm = _mmap_file_ro(w_path, pin_memory=pin_memory)
            s_f, s_mm = _mmap_file_ro(s_path, pin_memory=pin_memory)
            files.extend([w_f, s_f])
            mmaps.extend([w_mm, s_mm])
            w_arr: Any = np.ndarray(
                shape=tuple(info["weight_shape"]),
                dtype=w_dt,
                buffer=w_mm,
                offset=0,
            )
            s_arr: Any = np.ndarray(
                shape=tuple(info["scale_shape"]),
                dtype=s_dt,
                buffer=s_mm,
                offset=0,
            )
        elif "weight_gcs_uri" in info or bucket_name:
            if "weight_gcs_uri" in info:
                w_bkt, w_blob = _parse_gcs_blob_uri(info["weight_gcs_uri"])
            else:
                w_bkt, w_blob = bucket_name, prefix_slash + info["weight_file"]
            if "scale_gcs_uri" in info:
                s_bkt, s_blob = _parse_gcs_blob_uri(info["scale_gcs_uri"])
            else:
                s_bkt, s_blob = bucket_name, prefix_slash + info["scale_file"]
            w_arr = _GcsRowTable(
                w_bkt,
                w_blob,
                tuple(info["weight_shape"]),
                w_dt,
                base_offset=int(info.get("weight_data_start", 0)),
            )
            s_arr = _GcsRowTable(
                s_bkt,
                s_blob,
                tuple(info["scale_shape"]),
                s_dt,
                base_offset=int(info.get("scale_data_start", 0)),
            )
        else:
            continue
        tables[layer_id] = {"weight": w_arr, "scale": s_arr}
    return tables, files, mmaps


def _config_from_dict(cfg_dict: Mapping[str, Any]) -> DSV41Config:
    d = dict(cfg_dict)
    for tuple_key in (
        "compress_ratios",
        "kv_source_layer_ids",
        "index_source_layer_ids",
        "dspark_target_layer_ids",
        "engram_layer_ids",
        "engram_num_embeddings",
    ):
        if tuple_key in d and isinstance(d[tuple_key], list):
            d[tuple_key] = tuple(d[tuple_key])
    return DSV41Config(**d)


def load_presharded(
    path: str | os.PathLike[str],
    mesh: Mesh | None = None,
    *,
    world_size: int | None = None,
    device_order_bits: list[int] | None = None,
    stage_dir: str | os.PathLike[str] = "/dev/shm/preshard",
    workers: int = 8,
    max_seq_len: int = 256,
    load_engram_host: bool = True,
    replicate_engram_to_device: bool | None = None,
    return_metadata: bool = False,
) -> PreshardedWeightsDict | tuple[PreshardedWeightsDict, DSV41Config, dict[int, dict[str, np.ndarray]]]:
    """Loads pre-sharded weights from `path` directly onto the 1-D `"tp"` JAX mesh."""
    path_str = str(path)
    is_gcs = path_str.startswith("gs://")
    if is_gcs:
        # Fetch layout.json first so we know world_size and only stage this host's local_ranks.
        stage_from_gcs(path_str, stage_dir=stage_dir, local_ranks=[], stage_engram=False)
        local_root = Path(stage_dir)
    else:
        local_root = Path(path_str)

    layout_path = local_root / "layout.json"
    if not layout_path.is_file():
        raise FileNotFoundError(f"Missing layout.json in pre-sharded directory: {local_root}")
    layout = json.loads(layout_path.read_text())
    layout_world_size = int(layout["world_size"])
    if world_size is not None and world_size != layout_world_size:
        raise ValueError(
            f"Requested world_size={world_size} does not match layout.json world_size={layout_world_size}"
        )
    world_size = layout_world_size
    cfg = _config_from_dict(layout["config"])

    if mesh is None:
        mesh = create_v7x_mesh(
            world_size,
            device_order_bits=device_order_bits,
            axis_name="tp",
        )
    mesh_devices = list(mesh.devices.flat)
    if len(mesh_devices) != world_size:
        raise ValueError(
            f"Mesh has {len(mesh_devices)} devices, but pre-sharded layout requires {world_size}"
        )

    # Identify local ranks and stage only this host's ranks when reading from gs://.
    rank_for_dev: dict[jax.Device, int] = {dev: r for r, dev in enumerate(mesh_devices)}
    local_devices = list(mesh.local_devices)
    local_ranks = [rank_for_dev[d] for d in local_devices]
    if is_gcs:
        max_emb = max(cfg.engram_num_embeddings) if cfg.engram_num_embeddings else 0
        stage_from_gcs(
            path_str,
            stage_dir=stage_dir,
            local_ranks=local_ranks,
            stage_engram=(max_emb <= 65536),
        )

    rank_indices: dict[int, dict[str, Any]] = {}
    rank_mmaps: dict[int, mmap.mmap] = {}
    open_files: list[Any] = []
    open_mmaps: list[mmap.mmap] = []

    for r in local_ranks:
        r_dir = local_root / f"rank{r:02d}"
        idx_meta = json.loads((r_dir / "index.json").read_text())
        f_obj, mm_obj = _mmap_file_ro(r_dir / "dense.bin", pin_memory=False)
        rank_indices[r] = idx_meta["tensors"]
        rank_mmaps[r] = mm_obj
        open_files.append(f_obj)
        open_mmaps.append(mm_obj)

    first_rank_tensors = rank_indices[local_ranks[0]]
    tensor_names = list(first_rank_tensors.keys())
    tp_sharding = NamedSharding(mesh, P("tp"))

    def _slice_rank_view(r: int, name: str) -> np.ndarray:
        meta = rank_indices[r][name]
        offset = int(meta["offset"])
        assert offset % PAGE_ALIGN == 0, f"Unaligned tensor offset {offset} for {name} on rank {r}"
        shape = tuple(int(x) for x in meta["shape"])
        dt = PRESHARD_DTYPE_TO_NP[meta["dtype"]]
        return np.ndarray(shape=shape, dtype=dt, buffer=rank_mmaps[r], offset=offset)

    def _put_rank_shard(dev: jax.Device, name: str) -> jax.Array:
        r = rank_for_dev[dev]
        view = _slice_rank_view(r, name)
        return jax.device_put(view[None], dev)

    out = PreshardedWeightsDict()
    out.config = cfg
    out.layout = layout
    out._files = open_files
    out._mmaps = open_mmaps

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for name in tensor_names:
            meta0 = first_rank_tensors[name]
            per_rank_shape = tuple(int(x) for x in meta0["shape"])
            global_shape = (world_size, *per_rank_shape)
            shards = list(pool.map(lambda d: _put_rank_shard(d, name), local_devices))
            arr = jax.make_array_from_single_device_arrays(global_shape, tp_sharding, shards)
            out[name] = arr
            # Alias `.u32` routed expert weights to `.weight` so both `xla_decode.py` and
            # megakernel callers find their expected key without copying HBM buffers.
            if name.endswith(".u32"):
                alias = name[: -len(".u32")] + ".weight"
                out[alias] = arr

    if is_gcs and len(local_ranks) >= 8:
        jax.block_until_ready(list(out.values()))
        for mm_obj in open_mmaps:
            try:
                mm_obj.close()
            except Exception:
                pass
        for f_obj in open_files:
            try:
                f_obj.close()
            except Exception:
                pass
        open_mmaps.clear()
        open_files.clear()
        rank_mmaps.clear()
        default_unlink = "1" if world_size <= 8 else "0"
        if os.environ.get("DSV41_UNLINK_RANK_SHM", default_unlink) == "1":
            for r in local_ranks:
                try:
                    (local_root / f"rank{r:02d}" / "dense.bin").unlink(missing_ok=True)
                except Exception:
                    pass

    if load_engram_host:
        engram_tables, eng_files, eng_mmaps = load_engram_host_tables(
            local_root, pin_memory=True, gcs_uri=path_str if is_gcs else None
        )
        out.engram_host_tables = engram_tables
        out._files.extend(eng_files)
        out._mmaps.extend(eng_mmaps)
    else:
        out.engram_host_tables = {}

    if replicate_engram_to_device is None:
        max_emb = max(cfg.engram_num_embeddings) if cfg.engram_num_embeddings else 0
        replicate_engram_to_device = max_emb <= 65536

    def _replicate_tp(arr_np: np.ndarray) -> jax.Array:
        dt = jnp.float8_e4m3fn if arr_np.dtype == F8_E4M3 else arr_np.dtype
        j_arr = jnp.asarray(arr_np, dtype=dt)[None]
        global_shape = (world_size, *arr_np.shape)
        shards = [jax.device_put(j_arr, d) for d in local_devices]
        return jax.make_array_from_single_device_arrays(global_shape, tp_sharding, shards)

    if out.engram_host_tables:
        import sys
        from deepseek_v41.engram_hash import build_compressed_token_map

        eng_stage_dir = local_root / f"engram_tp{world_size}"
        fast_prefix = os.environ.get("DSV41_SKIP_ENGRAM_HBM", "0") == "1"
        for l_id, tbl in out.engram_host_tables.items():
            for sub_key in ("weight", "scale"):
                src = tbl[sub_key]
                if replicate_engram_to_device or (fast_prefix and isinstance(src, _GcsRowTable)):
                    arr_np = src.read_prefix_rows(65536) if isinstance(src, _GcsRowTable) else np.asarray(src)
                    out[f"layers.{l_id}.engram.embed.{sub_key}"] = _replicate_tp(arr_np)
                else:
                    n_total = int(src.shape[0])
                    part_n = (n_total + world_size - 1) // world_size
                    pad_val = 127 if sub_key == "scale" else 0
                    global_shape = (world_size, part_n, *src.shape[1:])
                    if isinstance(src, _GcsRowTable):
                        eng_stage_dir.mkdir(parents=True, exist_ok=True)
                        row_bytes = src._row_bytes
                        total_bytes = part_n * row_bytes

                        def _stage_eng_rank(r: int) -> Path:
                            dst_bin = eng_stage_dir / f"l{l_id}_{sub_key}_r{r:02d}.bin"
                            if dst_bin.is_file() and dst_bin.stat().st_size == total_bytes and not Path(f"{dst_bin}.part").exists():
                                return dst_bin
                            r_start = r * part_n
                            r_end = min(n_total, (r + 1) * part_n)
                            valid_rows = max(0, r_end - r_start)
                            valid_bytes = valid_rows * row_bytes
                            start_offset = src.base_offset + r_start * row_bytes
                            subprocess.run(
                                [
                                    sys.executable,
                                    "-c",
                                    _SUBPROC_RANGE_DL_CODE,
                                    src.bucket.name,
                                    src.blob_name,
                                    str(dst_bin),
                                    str(64 * 1024 * 1024),
                                    "16",
                                    str(start_offset),
                                    str(valid_bytes),
                                    str(total_bytes),
                                    str(pad_val),
                                ],
                                check=True,
                            )
                            return dst_bin

                        with ThreadPoolExecutor(max_workers=min(8, max(1, len(local_ranks)))) as eng_pool:
                            list(eng_pool.map(_stage_eng_rank, local_ranks))
                        eng_shards: list[jax.Array] = []
                        for dev in local_devices:
                            r = rank_for_dev[dev]
                            dst_bin = _stage_eng_rank(r)
                            f_obj, mm_obj = _mmap_file_ro(dst_bin, pin_memory=False)
                            view = np.ndarray(shape=(part_n, *src.shape[1:]), dtype=src.dtype, buffer=mm_obj, offset=0)
                            shard_arr = jax.device_put(view[None], dev)
                            jax.block_until_ready(shard_arr)
                            eng_shards.append(shard_arr)
                            try:
                                mm_obj.close()
                                f_obj.close()
                                if is_gcs:
                                    dst_bin.unlink(missing_ok=True)
                            except Exception:
                                pass
                        eng_arr = jax.make_array_from_single_device_arrays(
                            global_shape, tp_sharding, eng_shards
                        )
                        jax.block_until_ready(eng_arr)
                        out[f"layers.{l_id}.engram.embed.{sub_key}"] = eng_arr
                    else:
                        eng_shards = []
                        for dev in local_devices:
                            r = rank_for_dev[dev]
                            r_start = r * part_n
                            r_end = min(n_total, (r + 1) * part_n)
                            slc = np.asarray(src[r_start:r_end])
                            if slc.shape[0] < part_n:
                                pad = np.full((part_n - slc.shape[0], *src.shape[1:]), pad_val, dtype=slc.dtype)
                                slc = np.concatenate([slc, pad], axis=0)
                            eng_shards.append(jax.device_put(slc[None], dev))
                        out[f"layers.{l_id}.engram.embed.{sub_key}"] = jax.make_array_from_single_device_arrays(
                            global_shape, tp_sharding, eng_shards
                        )

        if cfg.vocab_size == 129280:
            tmap_cache = local_root / "token_map.npy"
            if tmap_cache.is_file():
                tmap_np = np.load(tmap_cache)
            else:
                tok_candidates = [
                    local_root / "tokenizer.json",
                    Path(os.environ.get("TOKENIZER_PATH", "")) if os.environ.get("TOKENIZER_PATH") else None,
                    Path("/app/deepseek_v41/tpu-v7x/tasks/tokenizer.json"),
                    Path(__file__).resolve().parent / "tpu-v7x" / "tasks" / "tokenizer.json",
                ]
                tok_found = next((p for p in tok_candidates if p is not None and p.is_file()), None)
                tmap_np, _ = build_compressed_token_map(
                    tok_found,
                    vocab_size=cfg.vocab_size,
                    compressed_vocab_size=getattr(cfg, "engram_compressed_vocab_size", 99092),
                )
                if tok_found is not None:
                    try:
                        np.save(tmap_cache, tmap_np)
                    except Exception:
                        pass
            out["engram.token_map"] = _replicate_tp(tmap_np.astype(np.int32))

    # Populate RoPE tables matching xla_decode.shard_weights
    rope_tables = precompute_layer_rope_tables(cfg, max_seq_len=max_seq_len)
    for r_name, (cos_arr, sin_arr) in rope_tables.items():
        for suffix, t_arr in (("cos", cos_arr), ("sin", sin_arr)):
            out[f"rope.{r_name}.{suffix}"] = _replicate_tp(np.asarray(t_arr))

    if return_metadata:
        return out, cfg, out.engram_host_tables
    return out


def unpack_checkpoint_mxfp4_from_v7x_u32(packed_u32: np.ndarray) -> np.ndarray:
    """Inverts `pack_checkpoint_mxfp4_for_v7x_bitcast(w_i8, transpose=True)` back to `[N, K // 2]` `int8`."""
    codes_kn = unpack_mxfp4_v7x_bitcast(packed_u32, axis=0, return_codes=True)
    codes_nk = np.swapaxes(np.asarray(codes_kn, dtype=np.uint8), -2, -1)
    low = codes_nk[..., 0::2] & np.uint8(0x0F)
    high = (codes_nk[..., 1::2] & np.uint8(0x0F)) << np.uint8(4)
    return np.ascontiguousarray((low | high).view(np.int8))


def dequantize_checkpoint_weights(
    raw_weights: Mapping[str, np.ndarray],
    cfg: DSV41Config,
) -> dict[str, np.ndarray]:
    """Dequantizes a raw unsharded checkpoint dictionary into `float32`/`bfloat16` reference tensors."""
    deq: dict[str, np.ndarray] = {}
    for k, v in raw_weights.items():
        if k.endswith(".scale"):
            continue
        scale_key = k[: -len(".weight")] + ".scale" if k.endswith(".weight") else None
        if scale_key is not None and scale_key in raw_weights:
            scale = raw_weights[scale_key]
            if v.dtype == np.dtype(np.int8):
                deq[k] = dequant_mxfp4_block32(v, scale, block_size=32)
            elif v.dtype == F8_E4M3:
                if ".engram.embed.weight" in k:
                    w_f32 = v.astype(np.float32).reshape(v.shape[0], v.shape[1] // 32, 32)
                    s_f32 = np.ldexp(
                        np.ones_like(scale, dtype=np.float32),
                        scale.astype(np.int32) - 127,
                    )[:, :, None]
                    deq[k] = (w_f32 * s_f32).reshape(v.shape)
                else:
                    deq[k] = dequant_fp8_block32x32(v, scale, block_size=32)
            else:
                deq[k] = np.asarray(v)
        else:
            deq[k] = np.asarray(v)
    return deq


def _reconstruct_block(
    sw: Mapping[str, np.ndarray],
    out: dict[str, np.ndarray],
    cfg: DSV41Config,
    *,
    prefix: str,
    layer_id: int,
    is_backbone: bool,
    world_size: int,
    dequantize: bool,
    engram_host_tables: Mapping[int, Mapping[str, np.ndarray]] | None,
) -> None:
    ep_lanes = 8
    tp_hosts = world_size // ep_lanes
    shared_tp = min(8, cfg.moe_inter_dim // 32)
    n_exp = cfg.n_routed_experts if is_backbone else cfg.dspark_n_routed_experts

    # 1. Replicated mHC & norms
    for rep_key in (
        f"{prefix}.hc_attn_fn",
        f"{prefix}.hc_ffn_fn",
        f"{prefix}.hc_attn_base",
        f"{prefix}.hc_ffn_base",
        f"{prefix}.hc_attn_scale",
        f"{prefix}.hc_ffn_scale",
        f"{prefix}.attn_norm.weight",
        f"{prefix}.ffn_norm.weight",
        f"{prefix}.attn.q_norm.weight",
        f"{prefix}.attn.kv_norm.weight",
        f"{prefix}.ffn.gate.weight",
        f"{prefix}.ffn.gate.bias",
        f"{prefix}.ffn.gate.bias_vl",
    ):
        out[rep_key] = np.asarray(sw[rep_key][0])

    # 2. Replicated FP8 wq_a and wkv
    for fp8_rep in (f"{prefix}.attn.wq_a", f"{prefix}.attn.wkv"):
        w = np.asarray(sw[f"{fp8_rep}.weight"][0])
        s = np.asarray(sw[f"{fp8_rep}.scale"][0])
        if dequantize:
            out[f"{fp8_rep}.weight"] = dequant_fp8_block32x32(w, s, block_size=32)
        else:
            out[f"{fp8_rep}.weight"] = w
            out[f"{fp8_rep}.scale"] = s

    # 3. Head-sharded wq_b and attn_sink
    out[f"{prefix}.attn.attn_sink"] = np.concatenate(
        [np.asarray(sw[f"{prefix}.attn.attn_sink"][r]) for r in range(world_size)],
        axis=0,
    )
    wqb_w = np.concatenate(
        [np.asarray(sw[f"{prefix}.attn.wq_b.weight"][r]) for r in range(world_size)],
        axis=0,
    )
    wqb_s = np.concatenate(
        [np.asarray(sw[f"{prefix}.attn.wq_b.scale"][r]) for r in range(world_size)],
        axis=0,
    )
    if dequantize:
        out[f"{prefix}.attn.wq_b.weight"] = dequant_fp8_block32x32(wqb_w, wqb_s, block_size=32)
    else:
        out[f"{prefix}.attn.wq_b.weight"] = wqb_w
        out[f"{prefix}.attn.wq_b.scale"] = wqb_s

    # 4. Group-diagonal wo_a
    ranks_per_group = world_size // cfg.o_groups
    if dequantize:
        woa_bf16_arr = np.asarray(sw[f"{prefix}.attn.wo_a.weight"])
        groups_bf16 = []
        for g in range(cfg.o_groups):
            r_start = g * ranks_per_group
            groups_bf16.append(
                np.concatenate(
                    [woa_bf16_arr[r_start + sub] for sub in range(ranks_per_group)],
                    axis=1,
                )
            )
        out[f"{prefix}.attn.wo_a.weight"] = np.concatenate(groups_bf16, axis=0).astype(
            np.float32
        )
    else:
        woa_fp8_arr = np.asarray(sw[f"{prefix}.attn.wo_a.fp8_weight"])
        woa_s_arr = np.asarray(sw[f"{prefix}.attn.wo_a.scale"])
        groups_w = []
        groups_s = []
        for g in range(cfg.o_groups):
            r_start = g * ranks_per_group
            groups_w.append(
                np.concatenate(
                    [woa_fp8_arr[r_start + sub] for sub in range(ranks_per_group)],
                    axis=1,
                )
            )
            groups_s.append(
                np.concatenate(
                    [woa_s_arr[r_start + sub] for sub in range(ranks_per_group)],
                    axis=1,
                )
            )
        out[f"{prefix}.attn.wo_a.weight"] = np.concatenate(groups_w, axis=0)
        out[f"{prefix}.attn.wo_a.scale"] = np.concatenate(groups_s, axis=0)

    # 5. Column-sharded wo_b
    wob_w = np.concatenate(
        [np.asarray(sw[f"{prefix}.attn.wo_b.weight"][r]) for r in range(world_size)],
        axis=1,
    )
    wob_s = np.concatenate(
        [np.asarray(sw[f"{prefix}.attn.wo_b.scale"][r]) for r in range(world_size)],
        axis=1,
    )
    if dequantize:
        out[f"{prefix}.attn.wo_b.weight"] = dequant_fp8_block32x32(wob_w, wob_s, block_size=32)
    else:
        out[f"{prefix}.attn.wo_b.weight"] = wob_w
        out[f"{prefix}.attn.wo_b.scale"] = wob_s

    # 6. Compressor & Indexer
    if is_backbone and layer_id in cfg.kv_source_layer_ids:
        out[f"{prefix}.attn.compressor.wkv.weight"] = np.asarray(
            sw[f"{prefix}.attn.compressor.wkv.weight"][0]
        )
        out[f"{prefix}.attn.compressor.norm.weight"] = np.asarray(
            sw[f"{prefix}.attn.compressor.norm.weight"][0]
        )
        if cfg.compress_ratios[layer_id] == 2:
            out[f"{prefix}.attn.compressor.wgate.weight"] = np.asarray(
                sw[f"{prefix}.attn.compressor.wgate.weight"][0]
            )

    if is_backbone and layer_id in cfg.index_source_layer_ids:
        idx_wqb_w = np.concatenate(
            [np.asarray(sw[f"{prefix}.attn.indexer.wq_b.weight"][r]) for r in range(world_size)],
            axis=0,
        )
        idx_wqb_s = np.concatenate(
            [np.asarray(sw[f"{prefix}.attn.indexer.wq_b.scale"][r]) for r in range(world_size)],
            axis=0,
        )
        if dequantize:
            out[f"{prefix}.attn.indexer.wq_b.weight"] = dequant_fp8_block32x32(
                idx_wqb_w, idx_wqb_s, block_size=32
            )
        else:
            out[f"{prefix}.attn.indexer.wq_b.weight"] = idx_wqb_w
            out[f"{prefix}.attn.indexer.wq_b.scale"] = idx_wqb_s

        out[f"{prefix}.attn.indexer.weights_proj.weight"] = np.concatenate(
            [
                np.asarray(sw[f"{prefix}.attn.indexer.weights_proj.weight"][r])
                for r in range(world_size)
            ],
            axis=0,
        )
        if layer_id in cfg.kv_source_layer_ids:
            out[f"{prefix}.attn.indexer.wk.weight"] = np.asarray(
                sw[f"{prefix}.attn.indexer.wk.weight"][0]
            )
            out[f"{prefix}.attn.indexer.k_norm.weight"] = np.asarray(
                sw[f"{prefix}.attn.indexer.k_norm.weight"][0]
            )

    # 7. Shared experts (sharded across `shared_tp` intra-host lanes)
    for proj, sh_axis in (("w1", 0), ("w2", 1), ("w3", 0)):
        sw_w = np.asarray(sw[f"{prefix}.ffn.shared_experts.{proj}.weight"])
        sw_s = np.asarray(sw[f"{prefix}.ffn.shared_experts.{proj}.scale"])
        w_cat = np.concatenate([sw_w[s_idx] for s_idx in range(shared_tp)], axis=sh_axis)
        s_cat = np.concatenate([sw_s[s_idx] for s_idx in range(shared_tp)], axis=sh_axis)
        if dequantize:
            out[f"{prefix}.ffn.shared_experts.{proj}.weight"] = dequant_fp8_block32x32(
                w_cat, s_cat, block_size=32
            )
        else:
            out[f"{prefix}.ffn.shared_experts.{proj}.weight"] = w_cat
            out[f"{prefix}.ffn.shared_experts.{proj}.scale"] = s_cat

    # 8. Routed experts (`EP=8` across lanes `r % 8`, `TP=tp_hosts` across hosts `r // 8`)
    exp_per_lane = n_exp // ep_lanes
    for proj, is_w2 in (("w1", False), ("w2", True), ("w3", False)):
        u32_all = np.asarray(sw[f"{prefix}.ffn.experts.{proj}.u32"])
        scale_all = np.asarray(sw[f"{prefix}.ffn.experts.{proj}.scale"])
        cat_axis = 1 if is_w2 else 0
        for e in range(n_exp):
            lane = e // exp_per_lane
            local_e = e % exp_per_lane
            w_slices = []
            s_slices = []
            for h in range(tp_hosts):
                r = h * ep_lanes + lane
                w_i8_slice = unpack_checkpoint_mxfp4_from_v7x_u32(u32_all[r, local_e])
                s_u8_slice = np.ascontiguousarray(scale_all[r, local_e].T)
                w_slices.append(w_i8_slice)
                s_slices.append(s_u8_slice)
            w_i8_full = np.concatenate(w_slices, axis=cat_axis)
            s_u8_full = np.concatenate(s_slices, axis=cat_axis)
            if dequantize:
                out[f"{prefix}.ffn.experts.{e}.{proj}.weight"] = dequant_mxfp4_block32(
                    w_i8_full, s_u8_full, block_size=32
                )
            else:
                out[f"{prefix}.ffn.experts.{e}.{proj}.weight"] = w_i8_full
                out[f"{prefix}.ffn.experts.{e}.{proj}.scale"] = s_u8_full

    # 9. Engram
    if is_backbone and layer_id in cfg.engram_layer_ids:
        if engram_host_tables is not None and layer_id in engram_host_tables:
            emb_w = np.asarray(engram_host_tables[layer_id]["weight"])
            emb_s = np.asarray(engram_host_tables[layer_id]["scale"])
        else:
            emb_w = np.asarray(sw[f"{prefix}.engram.embed.weight"][0])
            emb_s = np.asarray(sw[f"{prefix}.engram.embed.scale"][0])

        if dequantize:
            w_f32 = emb_w.astype(np.float32).reshape(emb_w.shape[0], emb_w.shape[1] // 32, 32)
            s_f32 = np.ldexp(
                np.ones_like(emb_s, dtype=np.float32),
                emb_s.astype(np.int32) - 127,
            )[:, :, None]
            out[f"{prefix}.engram.embed.weight"] = (w_f32 * s_f32).reshape(emb_w.shape)
        else:
            out[f"{prefix}.engram.embed.weight"] = emb_w
            out[f"{prefix}.engram.embed.scale"] = emb_s

        out[f"{prefix}.engram.q_weight"] = np.asarray(sw[f"{prefix}.engram.q_weight"][0])
        out[f"{prefix}.engram.k_weight"] = np.asarray(sw[f"{prefix}.engram.k_weight"][0])

        wkv_arr = np.asarray(sw[f"{prefix}.engram.wkv.weight"])
        wkv_s_arr = np.asarray(sw[f"{prefix}.engram.wkv.scale"])
        expected_wkv_rows = (cfg.hc_mult + 1) * cfg.dim
        if wkv_arr.shape[1] == expected_wkv_rows:
            wkv_w = wkv_arr[0]
            wkv_s = wkv_s_arr[0]
        else:
            wkv_w = np.concatenate([wkv_arr[r] for r in range(world_size)], axis=0)
            wkv_s = np.concatenate([wkv_s_arr[r] for r in range(world_size)], axis=0)
        if dequantize:
            out[f"{prefix}.engram.wkv.weight"] = dequant_fp8_block32x32(
                wkv_w, wkv_s, block_size=32
            )
        else:
            out[f"{prefix}.engram.wkv.weight"] = wkv_w
            out[f"{prefix}.engram.wkv.scale"] = wkv_s

    # 10. DSpark (`mtp.{k}`) extra heads
    if not is_backbone and layer_id == 0:
        out[f"{prefix}.main_norm.weight"] = np.asarray(sw[f"{prefix}.main_norm.weight"][0])
        mp_w = np.asarray(sw[f"{prefix}.main_proj.weight"][0])
        mp_s = np.asarray(sw[f"{prefix}.main_proj.scale"][0])
        if dequantize:
            out[f"{prefix}.main_proj.weight"] = dequant_fp8_block32x32(
                mp_w, mp_s, block_size=32
            )
        else:
            out[f"{prefix}.main_proj.weight"] = mp_w
            out[f"{prefix}.main_proj.scale"] = mp_s

    if not is_backbone and layer_id == cfg.n_mtp_layers - 1:
        out[f"{prefix}.norm.weight"] = np.asarray(sw[f"{prefix}.norm.weight"][0])
        out[f"{prefix}.confidence_head.proj.weight"] = np.asarray(
            sw[f"{prefix}.confidence_head.proj.weight"][0]
        )
        m_emb = np.asarray(sw[f"{prefix}.markov_head.embed.weight"][0])
        out[f"{prefix}.markov_head.embed.weight"] = m_emb[: cfg.vocab_size, :]

        m_head_cat = np.concatenate(
            [np.asarray(sw[f"{prefix}.markov_head.head.weight"][r]) for r in range(world_size)],
            axis=0,
        )
        out[f"{prefix}.markov_head.head.weight"] = m_head_cat[: cfg.vocab_size, :]


def reconstruct_unsharded_weights(
    sharded_weights: Mapping[str, Any],
    cfg: DSV41Config | None = None,
    *,
    dequantize: bool = False,
    engram_host_tables: Mapping[int, Mapping[str, np.ndarray]] | None = None,
) -> dict[str, np.ndarray]:
    """Reconstructs the full unsharded checkpoint dictionary from `tp8` or `tp32` sharded weights."""
    if cfg is None:
        cfg = getattr(sharded_weights, "config", None)
    if cfg is None:
        raise ValueError("cfg must be provided when sharded_weights has no .config attribute")

    if engram_host_tables is None:
        engram_host_tables = getattr(sharded_weights, "engram_host_tables", None)

    world_size = int(sharded_weights["embed.weight"].shape[0])
    out: dict[str, np.ndarray] = {}

    for v_key in ("embed.weight", "head.weight"):
        v_cat = np.concatenate(
            [np.asarray(sharded_weights[v_key][r]) for r in range(world_size)],
            axis=0,
        )
        out[v_key] = v_cat[: cfg.vocab_size, :]

    out["norm.weight"] = np.asarray(sharded_weights["norm.weight"][0])

    for l_id in range(cfg.n_layers):
        _reconstruct_block(
            sharded_weights,
            out,
            cfg,
            prefix=f"layers.{l_id}",
            layer_id=l_id,
            is_backbone=True,
            world_size=world_size,
            dequantize=dequantize,
            engram_host_tables=engram_host_tables,
        )

    for k in range(cfg.n_mtp_layers):
        _reconstruct_block(
            sharded_weights,
            out,
            cfg,
            prefix=f"mtp.{k}",
            layer_id=k,
            is_backbone=False,
            world_size=world_size,
            dequantize=dequantize,
            engram_host_tables=engram_host_tables,
        )

    return out
