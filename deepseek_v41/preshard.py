"""Offline / first-boot pre-sharder for DeepSeek-V4.1-Flash (`tp32` and `tp8`).

Streams raw `.safetensors` files from a local directory or `gs://` prefix and writes:
- `layout.json`: global metadata (`world_size`, `ep_lanes=8`, `tp_hosts=world_size//8`,
  config dict, Engram files, per-rank file sizes).
- `rank{00..N-1}/dense.bin`: 4 KiB-aligned binary blob of all TPU-resident weights for
  that rank (`O_DIRECT` / `mmap(MAP_POPULATE)` ready).
- `rank{00..N-1}/index.json`: `{tensor_name: {"offset", "nbytes", "shape", "dtype"}}`.
- `engram/layer_{id:02d}.weight.bin` and `engram/layer_{id:02d}.scale.bin`: contiguous
  row-major binary files for host DRAM `mmap(MAP_POPULATE) + mlock()`.

Matches `spec.md` §1 & §3 and `deepseek_v41/xla_decode.py` (`shard_weights`).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, BinaryIO

import numpy as np

from deepseek_v41.checkpoint import (
    BF16,
    SafetensorsCheckpoint,
    np_to_safetensors_dtype,
)
from deepseek_v41.config import DSV41Config, full_config
from deepseek_v41.quant import (
    dequant_fp8_block32x32,
    pack_checkpoint_mxfp4_for_v7x_bitcast,
)


PAGE_ALIGN = 4096


def align_up(offset: int, align: int = PAGE_ALIGN) -> int:
    """Rounds `offset` up to the next multiple of `align` (default 4096 bytes)."""
    return (offset + align - 1) // align * align


def dtype_to_preshard_str(dtype: Any, *, is_scale: bool = False) -> str:
    """Maps a NumPy dtype to its canonical string name in `rank{r}/index.json`."""
    dt = np.dtype(dtype)
    if dt == np.dtype(np.uint32):
        return "U32"
    return np_to_safetensors_dtype(dt, is_scale=is_scale)


class _RankBlobWriter:
    """Appends 4 KiB-aligned tensor slices into `rank{r:02d}/dense.bin` (local file or streaming GCS blob)."""

    def __init__(
        self,
        rank: int,
        rank_dir: Path | None,
        *,
        gcs_bucket: Any = None,
        gcs_prefix: str = "",
        active: bool = True,
    ) -> None:
        self.rank = rank
        self.rank_dir = rank_dir
        self.active = active
        self._gcs_bucket = gcs_bucket
        self._gcs_prefix = f"{gcs_prefix.rstrip('/')}/" if gcs_prefix else ""
        if not self.active:
            self._fh = None
            self.bin_path = None
            self.index_path = None
        elif self._gcs_bucket is not None:
            blob = self._gcs_bucket.blob(f"{self._gcs_prefix}rank{rank:02d}/dense.bin")
            self._fh: Any = blob.open("wb", chunk_size=64 * 1024 * 1024, ignore_flush=True)
            self.bin_path = None
            self.index_path = None
        else:
            assert rank_dir is not None
            rank_dir.mkdir(parents=True, exist_ok=True)
            self.bin_path = rank_dir / "dense.bin"
            self.index_path = rank_dir / "index.json"
            self._fh = open(self.bin_path, "wb")
        self.offset: int = 0
        self.tensors: dict[str, dict[str, Any]] = {}

    def append(self, name: str, arr: np.ndarray, *, is_scale: bool = False) -> None:
        aligned_offset = align_up(self.offset, PAGE_ALIGN)
        nbytes = int(arr.nbytes)
        if self.active and self._fh is not None:
            contig = np.ascontiguousarray(arr)
            pad_bytes = aligned_offset - self.offset
            if pad_bytes > 0:
                self._fh.write(b"\x00" * pad_bytes)
            raw = contig.view(np.uint8).ravel().tobytes()
            self._fh.write(raw)
        self.offset = aligned_offset + nbytes
        self.tensors[name] = {
            "offset": aligned_offset,
            "nbytes": nbytes,
            "shape": [int(s) for s in arr.shape],
            "dtype": dtype_to_preshard_str(arr.dtype, is_scale=is_scale),
        }

    def close(self, *, world_size: int) -> dict[str, Any]:
        # Pad total file size to a 4 KiB boundary so O_DIRECT reads of the whole file succeed.
        aligned_end = align_up(self.offset, PAGE_ALIGN)
        if aligned_end > self.offset:
            if self.active and self._fh is not None:
                self._fh.write(b"\x00" * (aligned_end - self.offset))
            self.offset = aligned_end
        if self.active and self._fh is not None:
            if self._gcs_bucket is None:
                self._fh.flush()
            self._fh.close()
        meta = {
            "rank": self.rank,
            "world_size": world_size,
            "page_align": PAGE_ALIGN,
            "total_bytes": self.offset,
            "tensors": self.tensors,
        }
        if self.active:
            payload = json.dumps(meta, indent=2, sort_keys=True)
            if self._gcs_bucket is not None:
                idx_blob = self._gcs_bucket.blob(f"{self._gcs_prefix}rank{self.rank:02d}/index.json")
                idx_blob.upload_from_string(payload, content_type="application/json")
            elif self.index_path is not None:
                self.index_path.write_text(payload)
        return meta


_WRITE_POOL = ThreadPoolExecutor(max_workers=32)


def _append_replicated(
    writers: list[_RankBlobWriter],
    name: str,
    arr: np.ndarray,
    *,
    is_scale: bool = False,
) -> None:
    contig = np.ascontiguousarray(arr)
    if len(writers) > 1 and any(w.active and w._gcs_bucket is not None for w in writers):
        list(_WRITE_POOL.map(lambda w: w.append(name, contig, is_scale=is_scale), writers))
    else:
        for w in writers:
            w.append(name, contig, is_scale=is_scale)


def _append_sharded_axis(
    writers: list[_RankBlobWriter],
    name: str,
    arr: np.ndarray,
    *,
    axis: int,
    is_scale: bool = False,
) -> None:
    world_size = len(writers)
    assert arr.shape[axis] % world_size == 0, (
        f"Tensor {name} shape {arr.shape} not divisible by world_size={world_size} along axis={axis}"
    )
    chunks = np.split(arr, world_size, axis=axis)
    if world_size > 1 and any(w.active and w._gcs_bucket is not None for w in writers):
        list(
            _WRITE_POOL.map(
                lambda rw: rw[1].append(name, chunks[rw[0]], is_scale=is_scale),
                enumerate(writers),
            )
        )
    else:
        for r, w in enumerate(writers):
            w.append(name, chunks[r], is_scale=is_scale)


def _pack_expert_for_hosts(
    w_i8: np.ndarray,
    s_u8: np.ndarray,
    *,
    is_w2: bool,
    tp_hosts: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Slices one expert's `(weight_i8, scale_u8)` across `tp_hosts` and packs for `pltpu.bitcast`."""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    if not is_w2:
        # w1 / w3: w_i8 [moe_inter_dim, dim // 2], s_u8 [moe_inter_dim, dim // 32]
        inter_per_host = w_i8.shape[0] // tp_hosts
        for h in range(tp_hosts):
            r0 = h * inter_per_host
            r1 = (h + 1) * inter_per_host
            w_slice = np.ascontiguousarray(w_i8[r0:r1, :])
            s_slice = np.ascontiguousarray(s_u8[r0:r1, :].T)  # [dim // 32, inter_per_host]
            u32 = pack_checkpoint_mxfp4_for_v7x_bitcast(w_slice, transpose=True)
            out.append((u32, s_slice))
    else:
        # w2: w_i8 [dim, moe_inter_dim // 2], s_u8 [dim, moe_inter_dim // 32]
        inter_per_host = (w_i8.shape[1] * 2) // tp_hosts
        half_inter = inter_per_host // 2
        scale_cols = inter_per_host // 32
        for h in range(tp_hosts):
            w_slice = np.ascontiguousarray(w_i8[:, h * half_inter : (h + 1) * half_inter])
            s_slice = np.ascontiguousarray(
                s_u8[:, h * scale_cols : (h + 1) * scale_cols].T
            )  # [inter_per_host // 32, dim]
            u32 = pack_checkpoint_mxfp4_for_v7x_bitcast(w_slice, transpose=True)
            out.append((u32, s_slice))
    return out


def _shard_block(
    ckpt: SafetensorsCheckpoint,
    writers: list[_RankBlobWriter],
    cfg: DSV41Config,
    *,
    prefix: str,
    layer_id: int,
    is_backbone: bool,
    engram_dir: Path,
    engram_manifest: dict[str, Any],
    executor: ThreadPoolExecutor,
) -> None:
    """Reads and shards one backbone layer (`layers.{L}`) or DSpark layer (`mtp.{K}`)."""
    world_size = len(writers)
    ep_lanes = 8
    tp_hosts = world_size // ep_lanes
    shared_tp = min(8, cfg.moe_inter_dim // 32)
    n_exp = cfg.n_routed_experts if is_backbone else cfg.dspark_n_routed_experts

    # 1. Gather all non-routed-expert keys for this block and read them in one coalesced batch
    dense_keys = [
        f"{prefix}.hc_attn_fn",
        f"{prefix}.hc_ffn_fn",
        f"{prefix}.hc_attn_base",
        f"{prefix}.hc_ffn_base",
        f"{prefix}.hc_attn_scale",
        f"{prefix}.hc_ffn_scale",
        f"{prefix}.attn_norm.weight",
        f"{prefix}.ffn_norm.weight",
        f"{prefix}.attn.wq_a.weight",
        f"{prefix}.attn.wq_a.scale",
        f"{prefix}.attn.q_norm.weight",
        f"{prefix}.attn.wq_b.weight",
        f"{prefix}.attn.wq_b.scale",
        f"{prefix}.attn.wkv.weight",
        f"{prefix}.attn.wkv.scale",
        f"{prefix}.attn.kv_norm.weight",
        f"{prefix}.attn.attn_sink",
        f"{prefix}.attn.wo_a.weight",
        f"{prefix}.attn.wo_a.scale",
        f"{prefix}.attn.wo_b.weight",
        f"{prefix}.attn.wo_b.scale",
        f"{prefix}.ffn.gate.weight",
        f"{prefix}.ffn.gate.bias",
        f"{prefix}.ffn.gate.bias_vl",
        f"{prefix}.ffn.shared_experts.w1.weight",
        f"{prefix}.ffn.shared_experts.w1.scale",
        f"{prefix}.ffn.shared_experts.w2.weight",
        f"{prefix}.ffn.shared_experts.w2.scale",
        f"{prefix}.ffn.shared_experts.w3.weight",
        f"{prefix}.ffn.shared_experts.w3.scale",
    ]
    if is_backbone and layer_id in cfg.kv_source_layer_ids:
        dense_keys.extend(
            [
                f"{prefix}.attn.compressor.wkv.weight",
                f"{prefix}.attn.compressor.norm.weight",
            ]
        )
        if cfg.compress_ratios[layer_id] == 2:
            dense_keys.append(f"{prefix}.attn.compressor.wgate.weight")

    if is_backbone and layer_id in cfg.index_source_layer_ids:
        dense_keys.extend(
            [
                f"{prefix}.attn.indexer.wq_b.weight",
                f"{prefix}.attn.indexer.wq_b.scale",
                f"{prefix}.attn.indexer.weights_proj.weight",
            ]
        )
        if layer_id in cfg.kv_source_layer_ids:
            dense_keys.extend(
                [
                    f"{prefix}.attn.indexer.wk.weight",
                    f"{prefix}.attn.indexer.k_norm.weight",
                ]
            )

    engram_in_place_gcs = False
    if is_backbone and layer_id in cfg.engram_layer_ids:
        w_meta = ckpt.tensor_meta(f"{prefix}.engram.embed.weight")
        engram_in_place_gcs = bool(ckpt.is_gcs and int(w_meta["nbytes"]) > 64 * 1024 * 1024)
        if not engram_in_place_gcs:
            dense_keys.extend(
                [
                    f"{prefix}.engram.embed.weight",
                    f"{prefix}.engram.embed.scale",
                ]
            )
        dense_keys.extend(
            [
                f"{prefix}.engram.q_weight",
                f"{prefix}.engram.k_weight",
                f"{prefix}.engram.wkv.weight",
                f"{prefix}.engram.wkv.scale",
            ]
        )

    if not is_backbone and layer_id == 0:
        dense_keys.extend(
            [
                f"{prefix}.main_norm.weight",
                f"{prefix}.main_proj.weight",
                f"{prefix}.main_proj.scale",
            ]
        )
    if not is_backbone and layer_id == cfg.n_mtp_layers - 1:
        dense_keys.extend(
            [
                f"{prefix}.norm.weight",
                f"{prefix}.confidence_head.proj.weight",
                f"{prefix}.markov_head.embed.weight",
                f"{prefix}.markov_head.head.weight",
            ]
        )

    tensors = ckpt.read_many(dense_keys, max_workers=min(8, executor._max_workers))

    # 2. Replicated mHC & layer norms & MLA input projections
    for rep_key in (
        f"{prefix}.hc_attn_fn",
        f"{prefix}.hc_ffn_fn",
        f"{prefix}.hc_attn_base",
        f"{prefix}.hc_ffn_base",
        f"{prefix}.hc_attn_scale",
        f"{prefix}.hc_ffn_scale",
        f"{prefix}.attn_norm.weight",
        f"{prefix}.ffn_norm.weight",
        f"{prefix}.attn.wq_a.weight",
        f"{prefix}.attn.q_norm.weight",
        f"{prefix}.attn.wkv.weight",
        f"{prefix}.attn.kv_norm.weight",
    ):
        _append_replicated(writers, rep_key, tensors[rep_key], is_scale=False)

    for rep_scale_key in (
        f"{prefix}.attn.wq_a.scale",
        f"{prefix}.attn.wkv.scale",
    ):
        _append_replicated(writers, rep_scale_key, tensors[rep_scale_key], is_scale=True)

    # 3. Head-sharded wq_b and attn_sink (TP=N along axis 0)
    _append_sharded_axis(
        writers, f"{prefix}.attn.wq_b.weight", tensors[f"{prefix}.attn.wq_b.weight"], axis=0
    )
    _append_sharded_axis(
        writers,
        f"{prefix}.attn.wq_b.scale",
        tensors[f"{prefix}.attn.wq_b.scale"],
        axis=0,
        is_scale=True,
    )
    _append_sharded_axis(
        writers, f"{prefix}.attn.attn_sink", tensors[f"{prefix}.attn.attn_sink"], axis=0
    )

    # 4. Group-diagonal wo_a (`o_groups=8`, `ranks_per_group = world_size // o_groups`)
    wo_a_fp8 = tensors[f"{prefix}.attn.wo_a.weight"]
    wo_a_scale = tensors[f"{prefix}.attn.wo_a.scale"]
    wo_a_bf16 = np.asarray(
        dequant_fp8_block32x32(wo_a_fp8, wo_a_scale, block_size=32, out_dtype=np.float32),
        dtype=BF16,
    )
    ranks_per_group = world_size // cfg.o_groups
    group_in_dim = (cfg.n_heads // cfg.o_groups) * cfg.head_dim
    cols_per_rank = group_in_dim // ranks_per_group
    scale_rows_per_group = cfg.o_lora_rank // 32
    scale_cols_per_rank = cols_per_rank // 32
    for r, w in enumerate(writers):
        g_id = r // ranks_per_group
        sub_id = r % ranks_per_group
        r0 = g_id * cfg.o_lora_rank
        r1 = (g_id + 1) * cfg.o_lora_rank
        c0 = sub_id * cols_per_rank
        c1 = (sub_id + 1) * cols_per_rank
        w.append(f"{prefix}.attn.wo_a.weight", wo_a_bf16[r0:r1, c0:c1])
        w.append(f"{prefix}.attn.wo_a.fp8_weight", wo_a_fp8[r0:r1, c0:c1])
        sr0 = g_id * scale_rows_per_group
        sr1 = (g_id + 1) * scale_rows_per_group
        sc0 = sub_id * scale_cols_per_rank
        sc1 = (sub_id + 1) * scale_cols_per_rank
        w.append(f"{prefix}.attn.wo_a.scale", wo_a_scale[sr0:sr1, sc0:sc1], is_scale=True)

    # 5. Column-sharded wo_b (TP=N along axis 1)
    _append_sharded_axis(
        writers, f"{prefix}.attn.wo_b.weight", tensors[f"{prefix}.attn.wo_b.weight"], axis=1
    )
    _append_sharded_axis(
        writers,
        f"{prefix}.attn.wo_b.scale",
        tensors[f"{prefix}.attn.wo_b.scale"],
        axis=1,
        is_scale=True,
    )

    # 6. Compressor & Indexer
    if is_backbone and layer_id in cfg.kv_source_layer_ids:
        _append_replicated(
            writers,
            f"{prefix}.attn.compressor.wkv.weight",
            tensors[f"{prefix}.attn.compressor.wkv.weight"],
        )
        _append_replicated(
            writers,
            f"{prefix}.attn.compressor.norm.weight",
            tensors[f"{prefix}.attn.compressor.norm.weight"],
        )
        if cfg.compress_ratios[layer_id] == 2:
            _append_replicated(
                writers,
                f"{prefix}.attn.compressor.wgate.weight",
                tensors[f"{prefix}.attn.compressor.wgate.weight"],
            )

    if is_backbone and layer_id in cfg.index_source_layer_ids:
        _append_sharded_axis(
            writers,
            f"{prefix}.attn.indexer.wq_b.weight",
            tensors[f"{prefix}.attn.indexer.wq_b.weight"],
            axis=0,
        )
        _append_sharded_axis(
            writers,
            f"{prefix}.attn.indexer.wq_b.scale",
            tensors[f"{prefix}.attn.indexer.wq_b.scale"],
            axis=0,
            is_scale=True,
        )
        _append_sharded_axis(
            writers,
            f"{prefix}.attn.indexer.weights_proj.weight",
            tensors[f"{prefix}.attn.indexer.weights_proj.weight"],
            axis=0,
        )
        if layer_id in cfg.kv_source_layer_ids:
            _append_replicated(
                writers,
                f"{prefix}.attn.indexer.wk.weight",
                tensors[f"{prefix}.attn.indexer.wk.weight"],
            )
            _append_replicated(
                writers,
                f"{prefix}.attn.indexer.k_norm.weight",
                tensors[f"{prefix}.attn.indexer.k_norm.weight"],
            )

    # 7. Router (replicated)
    _append_replicated(writers, f"{prefix}.ffn.gate.weight", tensors[f"{prefix}.ffn.gate.weight"])
    _append_replicated(writers, f"{prefix}.ffn.gate.bias", tensors[f"{prefix}.ffn.gate.bias"])
    _append_replicated(writers, f"{prefix}.ffn.gate.bias_vl", tensors[f"{prefix}.ffn.gate.bias_vl"])

    # 8. Shared expert (intra-host TP across `shared_tp = min(8, moe_inter_dim // 32)` lanes)
    s_inter_total = cfg.n_shared_experts * cfg.moe_inter_dim
    s_inter_per_lane = s_inter_total // shared_tp
    s_scale_per_lane = s_inter_per_lane // 32
    for proj in ("w1", "w3"):
        w_full = tensors[f"{prefix}.ffn.shared_experts.{proj}.weight"]
        s_full = tensors[f"{prefix}.ffn.shared_experts.{proj}.scale"]

        def _write_shared_proj(rw: tuple[int, _RankBlobWriter]) -> None:
            r, w = rw
            s_idx = r % shared_tp
            w.append(
                f"{prefix}.ffn.shared_experts.{proj}.weight",
                w_full[s_idx * s_inter_per_lane : (s_idx + 1) * s_inter_per_lane, :],
            )
            w.append(
                f"{prefix}.ffn.shared_experts.{proj}.scale",
                s_full[s_idx * s_scale_per_lane : (s_idx + 1) * s_scale_per_lane, :],
                is_scale=True,
            )

        list(_WRITE_POOL.map(_write_shared_proj, enumerate(writers)))

    w2_full = tensors[f"{prefix}.ffn.shared_experts.w2.weight"]
    s2_full = tensors[f"{prefix}.ffn.shared_experts.w2.scale"]

    def _write_shared_w2(rw: tuple[int, _RankBlobWriter]) -> None:
        r, w = rw
        s_idx = r % shared_tp
        w.append(
            f"{prefix}.ffn.shared_experts.w2.weight",
            w2_full[:, s_idx * s_inter_per_lane : (s_idx + 1) * s_inter_per_lane],
        )
        w.append(
            f"{prefix}.ffn.shared_experts.w2.scale",
            s2_full[:, s_idx * s_scale_per_lane : (s_idx + 1) * s_scale_per_lane],
            is_scale=True,
        )

    list(_WRITE_POOL.map(_write_shared_w2, enumerate(writers)))

    # 9. Routed experts (`EP=8` across intra-host lanes `r % 8`, `TP=tp_hosts` across hosts `r // 8`)
    exp_per_lane = n_exp // ep_lanes
    expert_keys: list[str] = []
    for e in range(n_exp):
        for proj in ("w1", "w2", "w3"):
            expert_keys.append(f"{prefix}.ffn.experts.{e}.{proj}.weight")
            expert_keys.append(f"{prefix}.ffn.experts.{e}.{proj}.scale")
    exp_tensors = ckpt.read_many(expert_keys, max_workers=max(8, executor._max_workers))

    for proj in ("w1", "w2", "w3"):
        def _pack_lane(lane: int) -> list[tuple[np.ndarray, np.ndarray]]:
            e_start = lane * exp_per_lane
            e_end = (lane + 1) * exp_per_lane
            w_stack = np.stack(
                [exp_tensors.pop(f"{prefix}.ffn.experts.{e}.{proj}.weight") for e in range(e_start, e_end)],
                axis=0,
            )
            s_stack = np.stack(
                [exp_tensors.pop(f"{prefix}.ffn.experts.{e}.{proj}.scale") for e in range(e_start, e_end)],
                axis=0,
            )
            host_slices: list[tuple[np.ndarray, np.ndarray]] = []
            if proj != "w2":
                inter_per_host = w_stack.shape[1] // tp_hosts
                k_u32 = w_stack.shape[2] // 4
                k_sc = s_stack.shape[2]
                for host in range(tp_hosts):
                    if not writers[host * ep_lanes + lane].active:
                        u32_s = np.broadcast_to(np.uint32(0), (exp_per_lane, k_u32, inter_per_host))
                        sc_s = np.broadcast_to(np.uint8(0), (exp_per_lane, k_sc, inter_per_host))
                        host_slices.append((u32_s, sc_s))
                        continue
                    r0 = host * inter_per_host
                    r1 = (host + 1) * inter_per_host
                    u32_s = pack_checkpoint_mxfp4_for_v7x_bitcast(w_stack[:, r0:r1, :], transpose=True)
                    sc_s = np.ascontiguousarray(np.swapaxes(s_stack[:, r0:r1, :], -2, -1))
                    host_slices.append((u32_s, sc_s))
            else:
                inter_per_host = (w_stack.shape[2] * 2) // tp_hosts
                half_inter = inter_per_host // 2
                scale_cols = inter_per_host // 32
                dim_out = w_stack.shape[1]
                for host in range(tp_hosts):
                    if not writers[host * ep_lanes + lane].active:
                        u32_s = np.broadcast_to(np.uint32(0), (exp_per_lane, half_inter // 4, dim_out))
                        sc_s = np.broadcast_to(np.uint8(0), (exp_per_lane, scale_cols, dim_out))
                        host_slices.append((u32_s, sc_s))
                        continue
                    u32_s = pack_checkpoint_mxfp4_for_v7x_bitcast(
                        w_stack[:, :, host * half_inter : (host + 1) * half_inter],
                        transpose=True,
                    )
                    sc_s = np.ascontiguousarray(
                        np.swapaxes(s_stack[:, :, host * scale_cols : (host + 1) * scale_cols], -2, -1)
                    )
                    host_slices.append((u32_s, sc_s))
            return host_slices

        lane_results = list(executor.map(_pack_lane, range(ep_lanes)))

        def _write_rank_proj(r: int) -> None:
            lane = r % ep_lanes
            host = r // ep_lanes
            u32_s, sc_s = lane_results[lane][host]
            writers[r].append(f"{prefix}.ffn.experts.{proj}.u32", u32_s)
            writers[r].append(f"{prefix}.ffn.experts.{proj}.scale", sc_s, is_scale=True)

        list(executor.map(_write_rank_proj, range(world_size)))
        del lane_results
    del exp_tensors

    # 10. Engram (host DRAM binary files + TPU HBM projections)
    if is_backbone and layer_id in cfg.engram_layer_ids:
        w_bin_rel = f"engram/layer_{layer_id:02d}.weight.bin"
        s_bin_rel = f"engram/layer_{layer_id:02d}.scale.bin"
        if engram_in_place_gcs:
            w_meta = ckpt.tensor_meta(f"{prefix}.engram.embed.weight")
            s_meta = ckpt.tensor_meta(f"{prefix}.engram.embed.scale")
            src_prefix = ckpt.raw_path.rstrip("/")
            engram_manifest[str(layer_id)] = {
                "weight_file": w_bin_rel,
                "weight_gcs_uri": f"{src_prefix}/{w_meta['shard_file']}",
                "weight_data_start": int(w_meta["data_start"]),
                "weight_shape": list(w_meta["shape"]),
                "weight_dtype": "F8_E4M3",
                "scale_file": s_bin_rel,
                "scale_gcs_uri": f"{src_prefix}/{s_meta['shard_file']}",
                "scale_data_start": int(s_meta["data_start"]),
                "scale_shape": list(s_meta["shape"]),
                "scale_dtype": "F8_E8M0",
            }
        else:
            emb_w = np.ascontiguousarray(tensors[f"{prefix}.engram.embed.weight"])
            emb_s = np.ascontiguousarray(tensors[f"{prefix}.engram.embed.scale"])
            engram_dir.mkdir(parents=True, exist_ok=True)
            (engram_dir / f"layer_{layer_id:02d}.weight.bin").write_bytes(
                emb_w.view(np.uint8).ravel().tobytes()
            )
            (engram_dir / f"layer_{layer_id:02d}.scale.bin").write_bytes(
                emb_s.view(np.uint8).ravel().tobytes()
            )
            engram_manifest[str(layer_id)] = {
                "weight_file": w_bin_rel,
                "weight_shape": list(emb_w.shape),
                "weight_dtype": "F8_E4M3",
                "scale_file": s_bin_rel,
                "scale_shape": list(emb_s.shape),
                "scale_dtype": "F8_E8M0",
            }

        _append_replicated(writers, f"{prefix}.engram.q_weight", tensors[f"{prefix}.engram.q_weight"])
        _append_replicated(writers, f"{prefix}.engram.k_weight", tensors[f"{prefix}.engram.k_weight"])

        wkv_w = tensors[f"{prefix}.engram.wkv.weight"]
        wkv_s = tensors[f"{prefix}.engram.wkv.scale"]
        if (wkv_w.shape[0] // world_size) % 32 == 0 and wkv_w.shape[0] % world_size == 0:
            _append_sharded_axis(writers, f"{prefix}.engram.wkv.weight", wkv_w, axis=0)
            _append_sharded_axis(
                writers, f"{prefix}.engram.wkv.scale", wkv_s, axis=0, is_scale=True
            )
        else:
            # Matches xla_decode.py line 778 when rows_per_rank is not a multiple of 32 (e.g. tiny_config on tp32)
            _append_replicated(writers, f"{prefix}.engram.wkv.weight", wkv_w)
            _append_replicated(
                writers, f"{prefix}.engram.wkv.scale", wkv_s, is_scale=True
            )

    # 11. DSpark (`mtp.{k}`) extra heads
    if not is_backbone and layer_id == 0:
        _append_replicated(
            writers, f"{prefix}.main_norm.weight", tensors[f"{prefix}.main_norm.weight"]
        )
        _append_replicated(
            writers, f"{prefix}.main_proj.weight", tensors[f"{prefix}.main_proj.weight"]
        )
        _append_replicated(
            writers,
            f"{prefix}.main_proj.scale",
            tensors[f"{prefix}.main_proj.scale"],
            is_scale=True,
        )

    if not is_backbone and layer_id == cfg.n_mtp_layers - 1:
        _append_replicated(writers, f"{prefix}.norm.weight", tensors[f"{prefix}.norm.weight"])
        _append_replicated(
            writers,
            f"{prefix}.confidence_head.proj.weight",
            tensors[f"{prefix}.confidence_head.proj.weight"],
        )
        # Matches xla_decode.py lines 531-536 & 791-795: pad both to padded_vocab_size;
        # markov_head.embed.weight is replicated, markov_head.head.weight is vocab-sharded.
        m_emb = tensors[f"{prefix}.markov_head.embed.weight"]
        m_emb_pad = np.zeros((cfg.padded_vocab_size, m_emb.shape[1]), dtype=m_emb.dtype)
        m_emb_pad[: m_emb.shape[0]] = m_emb
        _append_replicated(writers, f"{prefix}.markov_head.embed.weight", m_emb_pad)

        m_head = tensors[f"{prefix}.markov_head.head.weight"]
        m_head_pad = np.zeros((cfg.padded_vocab_size, m_head.shape[1]), dtype=m_head.dtype)
        m_head_pad[: m_head.shape[0]] = m_head
        _append_sharded_axis(writers, f"{prefix}.markov_head.head.weight", m_head_pad, axis=0)


def _upload_dir_to_gcs(local_dir: Path, gcs_uri: str, *, workers: int = 16, skip_layout: bool = False) -> None:
    """Uploads a local pre-sharded directory to a `gs://` prefix using `google.cloud.storage`."""
    gcs_uri = gcs_uri.rstrip("/")
    without = gcs_uri[len("gs://") :]
    bucket_name, _, prefix = without.partition("/")
    prefix_slash = f"{prefix.rstrip('/')}/" if prefix else ""

    files = [
        p
        for p in sorted(local_dir.rglob("*"))
        if p.is_file() and not (skip_layout and p.name == "layout.json")
    ]
    from google.cloud import storage  # type: ignore[import-untyped]
    import requests.adapters

    client = storage.Client()
    adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64)
    client._http.mount("https://", adapter)
    client._http.mount("http://", adapter)
    bucket = client.bucket(bucket_name)

    def _up(p: Path) -> None:
        rel = p.relative_to(local_dir).as_posix()
        blob = bucket.blob(prefix_slash + rel)
        blob.chunk_size = 64 * 1024 * 1024
        blob.upload_from_filename(str(p), timeout=600)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(_up, files))


def preshard_checkpoint(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    *,
    world_size: int = 32,
    workers: int = 8,
    cfg: DSV41Config | None = None,
    rank_start: int = 0,
    rank_end: int | None = None,
    upload_gcs: str | None = None,
) -> dict[str, Any]:
    """Streams a raw `.safetensors` checkpoint from `src` and writes pre-sharded artifacts to `dst`."""
    if world_size not in (8, 32):
        raise ValueError(f"world_size must be 8 or 32, got {world_size}")
    r_start = max(0, int(rank_start))
    r_end = world_size if rank_end is None else min(world_size, int(rank_end))

    ckpt = SafetensorsCheckpoint(src)
    if cfg is None:
        cfg = ckpt.load_config() or full_config()

    dst_str = str(dst)
    is_gcs_dst = dst_str.startswith("gs://")
    dst_bucket = None
    dst_prefix = ""
    if is_gcs_dst:
        from google.cloud import storage  # type: ignore[import-untyped]
        import requests.adapters

        without = dst_str.rstrip("/")[len("gs://") :]
        bucket_name, _, dst_prefix = without.partition("/")
        client = storage.Client()
        adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64)
        client._http.mount("https://", adapter)
        client._http.mount("http://", adapter)
        dst_bucket = client.bucket(bucket_name)
    shm_base = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmp_ctx = tempfile.TemporaryDirectory(prefix="dsv41_preshard_", dir=shm_base) if is_gcs_dst else None
    out_root = Path(tmp_ctx.name) if tmp_ctx is not None else Path(dst_str)
    out_root.mkdir(parents=True, exist_ok=True)

    writers = [
        _RankBlobWriter(
            r,
            None if is_gcs_dst else out_root / f"rank{r:02d}",
            gcs_bucket=dst_bucket,
            gcs_prefix=dst_prefix,
            active=(r_start <= r < r_end),
        )
        for r in range(world_size)
    ]
    engram_dir = out_root / "engram"
    engram_manifest: dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        # 1. Top-level embedding, final norm, and LM head
        top_tensors = ckpt.read_many(
            ["embed.weight", "norm.weight", "head.weight"],
            max_workers=min(4, max(1, workers)),
        )
        emb_arr = top_tensors["embed.weight"]
        emb_pad = np.zeros((cfg.padded_vocab_size, emb_arr.shape[1]), dtype=emb_arr.dtype)
        emb_pad[: emb_arr.shape[0]] = emb_arr
        _append_sharded_axis(writers, "embed.weight", emb_pad, axis=0)

        head_arr = top_tensors["head.weight"]
        head_pad = np.full(
            (cfg.padded_vocab_size, head_arr.shape[1]),
            -1e4,
            dtype=np.float32,
        ).astype(head_arr.dtype)
        head_pad[: head_arr.shape[0]] = head_arr
        _append_sharded_axis(writers, "head.weight", head_pad, axis=0)

        _append_replicated(writers, "norm.weight", top_tensors["norm.weight"])
        del top_tensors, emb_arr, emb_pad, head_arr, head_pad
        import gc as _gc, time as _time
        _gc.collect()

        # 2. Backbone layers 0..n_layers-1
        for l_id in range(cfg.n_layers):
            _t0 = _time.time()
            _shard_block(
                ckpt,
                writers,
                cfg,
                prefix=f"layers.{l_id}",
                layer_id=l_id,
                is_backbone=True,
                engram_dir=engram_dir,
                engram_manifest=engram_manifest,
                executor=executor,
            )
            _gc.collect()
            print(f"[preshard ws={world_size} ranks={r_start}..{r_end-1}] layers.{l_id} done in {_time.time() - _t0:.1f}s (rank0={writers[0].offset / 1e9:.2f} GB)", flush=True)

        # 3. DSpark (`mtp`) layers 0..n_mtp_layers-1
        for k in range(cfg.n_mtp_layers):
            _t0 = _time.time()
            _shard_block(
                ckpt,
                writers,
                cfg,
                prefix=f"mtp.{k}",
                layer_id=k,
                is_backbone=False,
                engram_dir=engram_dir,
                engram_manifest=engram_manifest,
                executor=executor,
            )
            _gc.collect()
            print(f"[preshard ws={world_size} ranks={r_start}..{r_end-1}] mtp.{k} done in {_time.time() - _t0:.1f}s (rank0={writers[0].offset / 1e9:.2f} GB)", flush=True)

        metas = list(executor.map(lambda w: w.close(world_size=world_size), writers))

    rank_summaries: dict[str, Any] = {}
    for w, meta in zip(writers, metas):
        rank_summaries[f"rank{w.rank:02d}"] = {
            "dense_bin": f"rank{w.rank:02d}/dense.bin",
            "index_json": f"rank{w.rank:02d}/index.json",
            "total_bytes": meta["total_bytes"],
            "num_tensors": len(meta["tensors"]),
        }

    layout = {
        "format": "dsv41_presharded_v1",
        "world_size": world_size,
        "ep_lanes": 8,
        "tp_hosts": world_size // 8,
        "page_align": PAGE_ALIGN,
        "config": asdict(cfg),
        "ranks": rank_summaries,
        "engram": engram_manifest,
    }
    layout_json = json.dumps(layout, indent=2, sort_keys=True)
    if is_gcs_dst and dst_bucket is not None:
        prefix_slash = f"{dst_prefix.rstrip('/')}/" if dst_prefix else ""
        if engram_dir.is_dir() and any(engram_dir.iterdir()):
            _upload_dir_to_gcs(out_root, dst_str)
        if r_start == 0:
            dst_bucket.blob(f"{prefix_slash}layout.json").upload_from_string(
                layout_json, content_type="application/json"
            )
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
    else:
        (out_root / "layout.json").write_text(layout_json)
        if upload_gcs:
            _t_up = _time.time()
            _upload_dir_to_gcs(
                out_root,
                upload_gcs,
                workers=min(16, max(4, workers)),
                skip_layout=(r_start != 0),
            )
            print(f"[preshard ws={world_size} ranks={r_start}..{r_end-1}] uploaded to {upload_gcs} in {_time.time() - _t_up:.1f}s", flush=True)

    return layout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-shard DeepSeek-V4.1-Flash safetensors checkpoint for TPU v7x (tp8 / tp32)."
    )
    parser.add_argument("--src", required=True, help="Source checkpoint directory or gs:// prefix")
    parser.add_argument("--dst", required=True, help="Destination pre-sharded directory or gs:// prefix")
    parser.add_argument(
        "--world-size",
        type=int,
        choices=(8, 32),
        default=32,
        help="Target TPU world size (8 for single-host tp8, 32 for 4-host tp32)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of worker threads for parallel range reads and MXFP4 packing",
    )
    parser.add_argument("--rank-start", type=int, default=0, help="First rank index to write (inclusive)")
    parser.add_argument("--rank-end", type=int, default=None, help="Last rank index to write (exclusive)")
    parser.add_argument("--upload-gcs", default=None, help="Optional gs:// prefix to upload local --dst to after completion")
    args = parser.parse_args(argv)
    preshard_checkpoint(
        args.src,
        args.dst,
        world_size=args.world_size,
        workers=args.workers,
        rank_start=args.rank_start,
        rank_end=args.rank_end,
        upload_gcs=args.upload_gcs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
