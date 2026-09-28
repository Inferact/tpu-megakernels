"""Safetensors checkpoint index/header reader (local + GCS) and synthetic checkpoint writer.

Handles the exact tensor naming, shapes, and quantization dtypes of DeepSeek-V4.1-Flash
(`INVENTORY.md` §0, §6–§7 and `headers_summary.txt`):
- `BF16` -> `ml_dtypes.bfloat16`
- `F32` -> `np.float32`
- `I8` -> `np.int8` (packed 2x E2M1 nibbles along K for routed experts)
- `F8_E4M3` -> `ml_dtypes.float8_e4m3fn`
- `F8_E8M0` / `U8` -> `np.uint8` (biased E8M0 exponent bytes)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

import ml_dtypes
import numpy as np

from deepseek_v41.config import DSV41Config


BF16 = np.dtype(ml_dtypes.bfloat16)
F8_E4M3 = np.dtype(ml_dtypes.float8_e4m3fn)

SAFETENSORS_TO_NP_DTYPE: dict[str, np.dtype] = {
    "BF16": BF16,
    "F32": np.dtype(np.float32),
    "F16": np.dtype(np.float16),
    "I8": np.dtype(np.int8),
    "U8": np.dtype(np.uint8),
    "I32": np.dtype(np.int32),
    "I64": np.dtype(np.int64),
    "F8_E4M3": F8_E4M3,
    "F8_E4M3FN": F8_E4M3,
    "F8_E8M0": np.dtype(np.uint8),
    "F8_E8M0FNU": np.dtype(np.uint8),
}


def np_to_safetensors_dtype(dtype: Any, *, is_scale: bool = False) -> str:
    """Maps a NumPy dtype back to the canonical DeepSeek-V4.1-Flash safetensors header dtype string."""
    dt = np.dtype(dtype)
    if is_scale and dt == np.dtype(np.uint8):
        return "F8_E8M0"
    if hasattr(ml_dtypes, "float8_e8m0fnu") and dt == np.dtype(ml_dtypes.float8_e8m0fnu):
        return "F8_E8M0"
    if dt == BF16:
        return "BF16"
    if dt == F8_E4M3:
        return "F8_E4M3"
    if dt == np.dtype(np.float32):
        return "F32"
    if dt == np.dtype(np.float16):
        return "F16"
    if dt == np.dtype(np.int8):
        return "I8"
    if dt == np.dtype(np.uint8):
        return "U8"
    if dt == np.dtype(np.int32):
        return "I32"
    if dt == np.dtype(np.int64):
        return "I64"
    raise ValueError(f"Unsupported dtype for safetensors: {dt}")


def _layer_block_specs(
    cfg: DSV41Config,
    prefix: str,
    *,
    is_backbone: bool,
    layer_id: int,
) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Returns `(safetensors_dtype, shape)` specs for one backbone or DSpark (`mtp`) layer."""
    specs: dict[str, tuple[str, tuple[int, ...]]] = {}
    d = cfg.dim
    hc_rows = 2 * cfg.hc_mult + cfg.hc_mult * cfg.hc_mult  # 24 for hc_mult=4
    hc_in = cfg.hc_mult * d
    q_out = cfg.n_heads * cfg.head_dim
    o_inter = cfg.o_groups * cfg.o_lora_rank
    o_group_in = (cfg.n_heads // cfg.o_groups) * cfg.head_dim

    # Common attention weights
    specs[f"{prefix}.attn.attn_sink"] = ("F32", (cfg.n_heads,))
    specs[f"{prefix}.attn.kv_norm.weight"] = ("BF16", (cfg.head_dim,))
    specs[f"{prefix}.attn.q_norm.weight"] = ("BF16", (cfg.q_lora_rank,))
    specs[f"{prefix}.attn.wkv.scale"] = ("F8_E8M0", (cfg.head_dim // 32, d // 32))
    specs[f"{prefix}.attn.wkv.weight"] = ("F8_E4M3", (cfg.head_dim, d))
    specs[f"{prefix}.attn.wo_a.scale"] = ("F8_E8M0", (o_inter // 32, o_group_in // 32))
    specs[f"{prefix}.attn.wo_a.weight"] = ("F8_E4M3", (o_inter, o_group_in))
    specs[f"{prefix}.attn.wo_b.scale"] = ("F8_E8M0", (d // 32, o_inter // 32))
    specs[f"{prefix}.attn.wo_b.weight"] = ("F8_E4M3", (d, o_inter))
    specs[f"{prefix}.attn.wq_a.scale"] = ("F8_E8M0", (cfg.q_lora_rank // 32, d // 32))
    specs[f"{prefix}.attn.wq_a.weight"] = ("F8_E4M3", (cfg.q_lora_rank, d))
    specs[f"{prefix}.attn.wq_b.scale"] = ("F8_E8M0", (q_out // 32, cfg.q_lora_rank // 32))
    specs[f"{prefix}.attn.wq_b.weight"] = ("F8_E4M3", (q_out, cfg.q_lora_rank))

    # Compressor & Indexer on backbone layers
    if is_backbone and layer_id in cfg.kv_source_layer_ids:
        specs[f"{prefix}.attn.compressor.norm.weight"] = ("BF16", (cfg.head_dim,))
        if cfg.compress_ratios[layer_id] == 2:
            specs[f"{prefix}.attn.compressor.wgate.weight"] = ("BF16", (cfg.head_dim, d))
        specs[f"{prefix}.attn.compressor.wkv.weight"] = ("BF16", (cfg.head_dim, d))

    if is_backbone and layer_id in cfg.index_source_layer_ids:
        idx_q_out = cfg.index_n_heads * cfg.index_head_dim
        if layer_id in cfg.kv_source_layer_ids:
            specs[f"{prefix}.attn.indexer.k_norm.weight"] = ("BF16", (cfg.index_head_dim,))
        specs[f"{prefix}.attn.indexer.weights_proj.weight"] = ("BF16", (cfg.index_n_heads, d))
        if layer_id in cfg.kv_source_layer_ids:
            specs[f"{prefix}.attn.indexer.wk.weight"] = ("BF16", (cfg.index_head_dim, cfg.head_dim))
        specs[f"{prefix}.attn.indexer.wq_b.scale"] = ("F8_E8M0", (idx_q_out // 32, cfg.q_lora_rank // 32))
        specs[f"{prefix}.attn.indexer.wq_b.weight"] = ("F8_E4M3", (idx_q_out, cfg.q_lora_rank))

    # Norms & mHC
    specs[f"{prefix}.attn_norm.weight"] = ("BF16", (d,))
    specs[f"{prefix}.ffn_norm.weight"] = ("BF16", (d,))
    specs[f"{prefix}.hc_attn_base"] = ("F32", (hc_rows,))
    specs[f"{prefix}.hc_attn_fn"] = ("F32", (hc_rows, hc_in))
    specs[f"{prefix}.hc_attn_scale"] = ("F32", (3,))
    specs[f"{prefix}.hc_ffn_base"] = ("F32", (hc_rows,))
    specs[f"{prefix}.hc_ffn_fn"] = ("F32", (hc_rows, hc_in))
    specs[f"{prefix}.hc_ffn_scale"] = ("F32", (3,))

    # MoE Router, Shared Experts, and Routed Experts
    n_experts = cfg.n_routed_experts if is_backbone else cfg.dspark_n_routed_experts
    inter = cfg.moe_inter_dim
    shared_inter = cfg.n_shared_experts * inter

    for e in range(n_experts):
        specs[f"{prefix}.ffn.experts.{e}.w1.scale"] = ("F8_E8M0", (inter, d // 32))
        specs[f"{prefix}.ffn.experts.{e}.w1.weight"] = ("I8", (inter, d // 2))
        specs[f"{prefix}.ffn.experts.{e}.w2.scale"] = ("F8_E8M0", (d, inter // 32))
        specs[f"{prefix}.ffn.experts.{e}.w2.weight"] = ("I8", (d, inter // 2))
        specs[f"{prefix}.ffn.experts.{e}.w3.scale"] = ("F8_E8M0", (inter, d // 32))
        specs[f"{prefix}.ffn.experts.{e}.w3.weight"] = ("I8", (inter, d // 2))

    specs[f"{prefix}.ffn.gate.bias"] = ("F32", (n_experts,))
    specs[f"{prefix}.ffn.gate.bias_vl"] = ("F32", (n_experts,))
    specs[f"{prefix}.ffn.gate.weight"] = ("BF16", (n_experts, d))

    specs[f"{prefix}.ffn.shared_experts.w1.scale"] = ("F8_E8M0", (shared_inter // 32, d // 32))
    specs[f"{prefix}.ffn.shared_experts.w1.weight"] = ("F8_E4M3", (shared_inter, d))
    specs[f"{prefix}.ffn.shared_experts.w2.scale"] = ("F8_E8M0", (d // 32, shared_inter // 32))
    specs[f"{prefix}.ffn.shared_experts.w2.weight"] = ("F8_E4M3", (d, shared_inter))
    specs[f"{prefix}.ffn.shared_experts.w3.scale"] = ("F8_E8M0", (shared_inter // 32, d // 32))
    specs[f"{prefix}.ffn.shared_experts.w3.weight"] = ("F8_E4M3", (shared_inter, d))

    # Engram on configured backbone layers
    if is_backbone and layer_id in cfg.engram_layer_ids:
        eng_idx = cfg.engram_layer_ids.index(layer_id)
        n_emb = cfg.engram_num_embeddings[eng_idx]
        n_orders = cfg.engram_max_ngram_size - 1
        eng_in = n_orders * cfg.engram_n_heads * cfg.engram_head_dim
        eng_out = (cfg.hc_mult + 1) * d
        specs[f"{prefix}.engram.embed.scale"] = ("F8_E8M0", (n_emb, cfg.engram_head_dim // 32))
        specs[f"{prefix}.engram.embed.weight"] = ("F8_E4M3", (n_emb, cfg.engram_head_dim))
        specs[f"{prefix}.engram.k_weight"] = ("BF16", (cfg.hc_mult, d))
        specs[f"{prefix}.engram.q_weight"] = ("BF16", (cfg.hc_mult, d))
        specs[f"{prefix}.engram.wkv.scale"] = ("F8_E8M0", (eng_out // 32, eng_in // 32))
        specs[f"{prefix}.engram.wkv.weight"] = ("F8_E4M3", (eng_out, eng_in))

    return specs


def expected_checkpoint_specs(
    cfg: DSV41Config,
    *,
    include_mtp: bool = True,
) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Returns the complete `{tensor_name: (safetensors_dtype, shape)}` map for `cfg`."""
    specs: dict[str, tuple[str, tuple[int, ...]]] = {}
    d = cfg.dim

    specs["embed.weight"] = ("BF16", (cfg.vocab_size, d))
    for layer_id in range(cfg.n_layers):
        specs.update(_layer_block_specs(cfg, f"layers.{layer_id}", is_backbone=True, layer_id=layer_id))
    specs["head.weight"] = ("BF16", (cfg.vocab_size, d))
    specs["norm.weight"] = ("BF16", (d,))

    if include_mtp and cfg.n_mtp_layers > 0:
        for k in range(cfg.n_mtp_layers):
            prefix = f"mtp.{k}"
            specs.update(_layer_block_specs(cfg, prefix, is_backbone=False, layer_id=cfg.n_layers + k))
            if k == 0:
                main_in = len(cfg.dspark_target_layer_ids) * d
                specs[f"{prefix}.main_norm.weight"] = ("BF16", (d,))
                specs[f"{prefix}.main_proj.scale"] = ("F8_E8M0", (d // 32, main_in // 32))
                specs[f"{prefix}.main_proj.weight"] = ("F8_E4M3", (d, main_in))
            if k == cfg.n_mtp_layers - 1:
                specs[f"{prefix}.confidence_head.proj.weight"] = ("BF16", (1, d + cfg.markov_rank))
                specs[f"{prefix}.markov_head.embed.weight"] = ("BF16", (cfg.vocab_size, cfg.markov_rank))
                specs[f"{prefix}.markov_head.head.weight"] = ("BF16", (cfg.vocab_size, cfg.markov_rank))
                specs[f"{prefix}.norm.weight"] = ("BF16", (d,))

    return specs


def save_safetensors_file(
    tensors: Mapping[str, np.ndarray],
    path: Path | str,
    *,
    dtype_overrides: Mapping[str, str] | None = None,
    metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Writes a standard `.safetensors` file supporting `BF16`, `F32`, `I8`, `U8`, `F8_E4M3`, `F8_E8M0`."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    offset = 0
    ordered_arrays: list[np.ndarray] = []
    for name in sorted(tensors.keys()):
        arr = np.ascontiguousarray(tensors[name])
        if dtype_overrides and name in dtype_overrides:
            st_dtype = dtype_overrides[name]
        else:
            st_dtype = np_to_safetensors_dtype(arr.dtype, is_scale=name.endswith(".scale"))
        nbytes = int(arr.nbytes)
        header[name] = {
            "dtype": st_dtype,
            "shape": [int(s) for s in arr.shape],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
        ordered_arrays.append(arr)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(header_bytes) % 8)) % 8
    if pad:
        header_bytes += b" " * pad

    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for arr in ordered_arrays:
            f.write(arr.view(np.uint8).tobytes())

    return header


def generate_synthetic_weights(cfg: DSV41Config, seed: int = 0) -> dict[str, np.ndarray]:
    """Generates deterministic, well-conditioned synthetic checkpoint tensors matching `cfg`."""
    specs = expected_checkpoint_specs(cfg)
    weights: dict[str, np.ndarray] = {}

    for idx, (name, (st_dtype, shape)) in enumerate(sorted(specs.items())):
        rng = np.random.default_rng((seed, idx))
        if st_dtype == "F8_E8M0":
            # Normal finite E8M0 exponents around 2^-9 .. 2^-5 (biased 118..122) for unit-variance activations
            weights[name] = rng.integers(118, 123, size=shape, dtype=np.uint8)
        elif st_dtype == "I8":
            # Packed 2x E2M1 nibbles in each byte
            weights[name] = rng.integers(-128, 128, size=shape, dtype=np.int8)
        elif st_dtype == "F8_E4M3":
            vals = rng.standard_normal(size=shape, dtype=np.float32) * np.float32(0.5)
            f8 = vals.astype(F8_E4M3)
            u8 = f8.view(np.uint8).copy()
            u8[u8 == 0x80] = 0x00
            weights[name] = u8.view(F8_E4M3)
        elif st_dtype == "BF16":
            if "norm.weight" in name:
                vals = np.float32(1.0) + rng.standard_normal(size=shape, dtype=np.float32) * np.float32(0.05)
            else:
                fan_in = shape[-1] if len(shape) >= 2 else shape[0]
                scale = np.float32(1.0 / math.sqrt(max(1, fan_in)))
                vals = rng.standard_normal(size=shape, dtype=np.float32) * scale
            weights[name] = vals.astype(BF16)
        elif st_dtype == "F32":
            if name.endswith("_scale"):
                vals = np.ones(shape, dtype=np.float32) * np.float32(0.25)
            elif name.endswith("_base"):
                vals = rng.standard_normal(size=shape, dtype=np.float32) * np.float32(0.05)
            elif name.endswith("_fn"):
                vals = rng.standard_normal(size=shape, dtype=np.float32) * np.float32(0.02)
            elif name.endswith("attn_sink"):
                vals = rng.standard_normal(size=shape, dtype=np.float32) * np.float32(0.1) - np.float32(2.0)
            else:
                vals = rng.standard_normal(size=shape, dtype=np.float32) * np.float32(0.02)
            weights[name] = vals.astype(np.float32)
        else:
            raise ValueError(f"Unhandled synthetic safetensors dtype {st_dtype} for {name}")

    return weights


def write_synthetic_checkpoint(
    cfg: DSV41Config,
    out_dir: Path | str,
    seed: int = 0,
    *,
    num_shards: int = 2,
) -> dict[str, np.ndarray]:
    """Writes a complete valid DeepSeek-V4.1-Flash safetensors checkpoint + index + config.json."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    specs = expected_checkpoint_specs(cfg)
    weights = generate_synthetic_weights(cfg, seed=seed)

    # Partition keys across shard files (putting engram tables in the last shard like shard 47-48).
    engram_keys = sorted(k for k in weights if ".engram.embed." in k)
    other_keys = sorted(k for k in weights if ".engram.embed." not in k)

    shard_lists: list[list[str]] = []
    if num_shards <= 1 or not engram_keys:
        n_sh = max(1, num_shards)
        all_keys = sorted(weights.keys())
        chunk = max(1, (len(all_keys) + n_sh - 1) // n_sh)
        for i in range(0, len(all_keys), chunk):
            shard_lists.append(all_keys[i : i + chunk])
    else:
        n_dense = max(1, num_shards - 1)
        chunk = max(1, (len(other_keys) + n_dense - 1) // n_dense)
        for i in range(0, len(other_keys), chunk):
            shard_lists.append(other_keys[i : i + chunk])
        shard_lists.append(engram_keys)

    total_shards = len(shard_lists)
    weight_map: dict[str, str] = {}
    total_size = 0

    for shard_idx, keys in enumerate(shard_lists, start=1):
        shard_name = f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"
        shard_tensors = {k: weights[k] for k in keys}
        dtype_overrides = {k: specs[k][0] for k in keys}
        save_safetensors_file(
            shard_tensors,
            out_path / shard_name,
            dtype_overrides=dtype_overrides,
            metadata={"format": "pt"},
        )
        for k in keys:
            weight_map[k] = shard_name
            total_size += int(weights[k].nbytes)

    index_payload = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    (out_path / "model.safetensors.index.json").write_text(json.dumps(index_payload, indent=2) + "\n")
    (out_path / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    return weights


def _split_gcs_uri(uri: str) -> tuple[str, str]:
    rest = uri[len("gs://") :]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.rstrip("/")


class SafetensorsCheckpoint:
    """Range-reading safetensors checkpoint reader supporting local directories and `gs://` prefixes."""

    COALESCE_GAP = 16 << 20
    MAX_RANGE_BYTES = 64 << 20
    CHUNK_SIZE = 32 << 20
    READ_THREADS = 16

    def __init__(self, path_or_uri: Path | str):
        self.raw_path = str(path_or_uri)
        self.is_gcs = self.raw_path.startswith("gs://")
        self._layouts: dict[str, tuple[dict[str, Any], int]] = {}
        self._gcs_bucket = None
        self._gcs_prefix = ""

        if self.is_gcs:
            from google.cloud import storage
            import requests.adapters

            bucket_name, self._gcs_prefix = _split_gcs_uri(self.raw_path)
            client = storage.Client()
            adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64)
            client._http.mount("https://", adapter)
            client._http.mount("http://", adapter)
            self._gcs_bucket = client.bucket(bucket_name)
            index_blob = self._gcs_bucket.blob(f"{self._gcs_prefix}/model.safetensors.index.json")
            if index_blob.exists():
                index_data = json.loads(index_blob.download_as_bytes(raw_download=True).decode("utf-8"))
                self.weight_map: dict[str, str] = index_data["weight_map"]
                self.metadata: dict[str, Any] = index_data.get("metadata", {})
            else:
                raise FileNotFoundError(f"Missing model.safetensors.index.json at {self.raw_path}")
        else:
            self.path = Path(path_or_uri)
            index_file = self.path / "model.safetensors.index.json"
            if index_file.is_file():
                index_data = json.loads(index_file.read_text())
                self.weight_map = index_data["weight_map"]
                self.metadata = index_data.get("metadata", {})
            else:
                shard_files = sorted(p.name for p in self.path.glob("*.safetensors"))
                if not shard_files:
                    raise FileNotFoundError(f"No .safetensors files or index found in {self.path}")
                self.weight_map = {}
                self.metadata = {}
                for sf in shard_files:
                    hdr, _ = self._layout(sf)
                    for k in hdr:
                        if not k.startswith("__"):
                            self.weight_map[k] = sf

    def load_config(self, **overrides) -> DSV41Config:
        """Loads `DSV41Config` from `config.json` in the checkpoint directory if present, else `full_config`."""
        if self.is_gcs:
            blob = self._gcs_bucket.blob(f"{self._gcs_prefix}/config.json")
            if blob.exists():
                data = json.loads(blob.download_as_bytes(raw_download=True).decode("utf-8"))
                fields = {f: data[f] for f in DSV41Config.__dataclass_fields__ if f in data}
                for k, v in list(fields.items()):
                    if isinstance(getattr(DSV41Config, k), tuple) and isinstance(v, list):
                        fields[k] = tuple(v)
                fields.update(overrides)
                return DSV41Config(**fields)
            return DSV41Config(**overrides)
        cfg_file = self.path / "config.json"
        if cfg_file.is_file():
            return DSV41Config.from_json(cfg_file, **overrides)
        return DSV41Config(**overrides)

    def keys(self) -> list[str]:
        return sorted(self.weight_map.keys())

    def tensor_names(self) -> list[str]:
        return self.keys()

    def _read_range(self, filename: str, offset: int, length: int) -> bytes:
        if length <= 0:
            return b""
        if self.is_gcs:
            blob = self._gcs_bucket.blob(f"{self._gcs_prefix}/{filename}")
            if length <= self.MAX_RANGE_BYTES:
                return blob.download_as_bytes(start=offset, end=offset + length - 1, raw_download=True)
            spans = [
                (offset + pos, min(length - pos, self.CHUNK_SIZE))
                for pos in range(0, length, self.CHUNK_SIZE)
            ]
            with ThreadPoolExecutor(max_workers=min(self.READ_THREADS, len(spans))) as pool:
                parts = list(
                    pool.map(
                        lambda sp: blob.download_as_bytes(
                            start=sp[0], end=sp[0] + sp[1] - 1, raw_download=True
                        ),
                        spans,
                    )
                )
            return b"".join(parts)
        file_path = self.path / filename
        fd = os.open(file_path, os.O_RDONLY)
        try:
            pieces = []
            for pos in range(0, length, self.CHUNK_SIZE):
                pieces.append(os.pread(fd, min(self.CHUNK_SIZE, length - pos), offset + pos))
            return b"".join(pieces)
        finally:
            os.close(fd)

    def _layout(self, filename: str) -> tuple[dict[str, Any], int]:
        """Returns `(header_dict, data_start_offset)` for `filename`, cached."""
        if filename not in self._layouts:
            raw_size = self._read_range(filename, 0, 8)
            header_len = struct.unpack("<Q", raw_size)[0]
            header_json = self._read_range(filename, 8, header_len).decode("utf-8")
            self._layouts[filename] = (json.loads(header_json), 8 + int(header_len))
        return self._layouts[filename]

    def tensor_meta(self, key: str) -> dict[str, Any]:
        """Returns header metadata (`dtype`, `shape`, `data_offsets`, `shard_file`, `nbytes`) for `key`."""
        filename = self.weight_map[key]
        header, base = self._layout(filename)
        meta = header[key]
        a, b = meta["data_offsets"]
        return {
            "dtype": meta["dtype"],
            "shape": tuple(int(x) for x in meta["shape"]),
            "data_offsets": (int(a), int(b)),
            "data_start": int(base) + int(a),
            "nbytes": int(b) - int(a),
            "shard_file": filename,
        }

    def iter_shards(self) -> list[tuple[str, list[str]]]:
        """Returns `[(shard_filename, [keys_sorted_by_offset]), ...]` for streaming shard-by-shard."""
        by_shard: dict[str, list[str]] = {}
        for key, sf in self.weight_map.items():
            by_shard.setdefault(sf, []).append(key)
        result: list[tuple[str, list[str]]] = []
        for sf in sorted(by_shard.keys()):
            header, _ = self._layout(sf)
            ordered = sorted(by_shard[sf], key=lambda k: header[k]["data_offsets"][0])
            result.append((sf, ordered))
        return result

    def read_tensor(self, key: str, row_slice: slice | None = None) -> np.ndarray:
        """Reads a single tensor (or a contiguous leading-axis `row_slice`) as a NumPy array."""
        filename = self.weight_map[key]
        header, base = self._layout(filename)
        meta = header[key]
        shape = [int(x) for x in meta["shape"]]
        np_dtype = SAFETENSORS_TO_NP_DTYPE[meta["dtype"]]
        start, end = int(header[key]["data_offsets"][0]), int(header[key]["data_offsets"][1])

        if row_slice is not None and shape:
            r0, r1, step = row_slice.indices(shape[0])
            if step != 1:
                raise ValueError(f"Only unit-step row_slice supported, got {row_slice}")
            row_bytes = (end - start) // shape[0]
            byte_offset = base + start + r0 * row_bytes
            byte_length = (r1 - r0) * row_bytes
            out_shape = [r1 - r0, *shape[1:]]
        else:
            byte_offset = base + start
            byte_length = end - start
            out_shape = shape

        raw = self._read_range(filename, byte_offset, byte_length)
        return np.frombuffer(raw, dtype=np_dtype).reshape(out_shape).copy()

    def read_many(
        self,
        keys: Sequence[str],
        *,
        max_workers: int | None = None,
        coalesce_gap: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Reads multiple tensors via coalesced range streams in parallel."""
        if not keys:
            return {}
        workers = max_workers or self.READ_THREADS
        gap = self.COALESCE_GAP if coalesce_gap is None else coalesce_gap

        by_file: dict[str, list[str]] = {}
        for k in keys:
            by_file.setdefault(self.weight_map[k], []).append(k)

        out: dict[str, np.ndarray] = {}
        for filename, file_keys in by_file.items():
            header, base = self._layout(filename)
            spans = sorted((int(header[k]["data_offsets"][0]), int(header[k]["data_offsets"][1]), k) for k in file_keys)
            ranges: list[list[Any]] = []
            for s, e, k in spans:
                if (
                    ranges
                    and s - ranges[-1][1] <= gap
                    and max(ranges[-1][1], e) - ranges[-1][0] <= self.MAX_RANGE_BYTES
                ):
                    ranges[-1][1] = max(ranges[-1][1], e)
                    ranges[-1][2].append(k)
                else:
                    ranges.append([s, e, [k]])

            def _fetch_and_slice(bounds: list[Any]) -> list[tuple[str, np.ndarray]]:
                s, e, range_keys = bounds
                raw_buf = self._read_range(filename, base + s, e - s)
                u8_buf = np.frombuffer(raw_buf, dtype=np.uint8)
                res: list[tuple[str, np.ndarray]] = []
                for k in range_keys:
                    meta = header[k]
                    a, b = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
                    np_dtype = SAFETENSORS_TO_NP_DTYPE[meta["dtype"]]
                    shape = [int(x) for x in meta["shape"]]
                    res.append((k, u8_buf[a - s : b - s].view(np_dtype).reshape(shape).copy()))
                return res

            with ThreadPoolExecutor(max(1, min(workers, len(ranges)))) as pool:
                for batch in pool.map(_fetch_and_slice, ranges):
                    for k, arr in batch:
                        out[k] = arr

        return out
