"""Suite T3: Pre-sharder & fast-start loader round-trip tests (`tp32` and `tp8`).

Verifies:
1. Synthetic `.safetensors` checkpoint generation (`BF16`, `F32`, `F8_E4M3`, `F8_E8M0`, `I8` MXFP4)
   and `expected_checkpoint_specs` parity with `INVENTORY.md` / `headers_summary.txt`.
2. `deepseek_v41.preshard` for both `world_size=32` (`tp32`) and `world_size=8` (`tp8`).
3. Presence of `layout.json`, `rank{00..N-1}/dense.bin`, `rank{00..N-1}/index.json`, and
   `engram/layer_{id:02d}.{weight,scale}.bin`.
4. 4 KiB (`offset % 4096 == 0`) alignment of every tensor offset and `dense.bin` total size.
5. Exact bit-for-bit round-trip (`max_abs_err == 0.0`) from `load_presharded` on 32-device and
   8-device virtual meshes back to the unsharded checkpoint tensors (both raw and dequantized).
6. Routed expert MXFP4 `uint32` packed slices unpacked via `pallas_bitcast_mxfp4_to_fp8`
   (`pltpu.bitcast`) + `E8M0` scale match reference nibble unpacking (`max_abs_err == 0.0`).
7. Exact tensor-by-tensor parity between `load_presharded` and `xla_decode.shard_weights`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=32")

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from deepseek_v41.checkpoint import (
    SafetensorsCheckpoint,
    expected_checkpoint_specs,
    write_synthetic_checkpoint,
)
from deepseek_v41.collectives import create_v7x_mesh
from deepseek_v41.config import full_config, tiny_config
from deepseek_v41.load import (
    dequantize_checkpoint_weights,
    ensure_presharded,
    is_presharded,
    load_presharded,
    reconstruct_unsharded_weights,
)
from deepseek_v41.preshard import PAGE_ALIGN, main as preshard_cli_main
from deepseek_v41.quant import (
    dequant_mxfp4_block32,
    pallas_bitcast_mxfp4_to_fp8,
)
from deepseek_v41.xla_decode import shard_weights


@pytest.fixture(scope="module")
def synthetic_ckpt(tmp_path_factory: pytest.TempPathFactory):
    cfg = tiny_config(engram_num_embeddings=(1024, 1024))
    ckpt_dir = tmp_path_factory.mktemp("dsv41_tiny_safetensors")
    raw_weights = write_synthetic_checkpoint(cfg, ckpt_dir, seed=42, num_shards=2)
    return cfg, ckpt_dir, raw_weights


def test_full_config_checkpoint_specs_match_inventory() -> None:
    """Verifies `expected_checkpoint_specs(full_config())` matches `INVENTORY.md` & `headers_summary.txt`."""
    full_cfg = full_config()
    specs = expected_checkpoint_specs(full_cfg)

    # Top-level tensors
    assert specs["embed.weight"] == ("BF16", (129280, 5120))
    assert specs["norm.weight"] == ("BF16", (5120,))
    assert specs["head.weight"] == ("BF16", (129280, 5120))

    # Layer 0 (SWA, non-source)
    assert "layers.0.attn.compressor.wkv.weight" not in specs
    assert "layers.0.attn.indexer.wq_b.weight" not in specs
    assert specs["layers.0.attn.wq_a.weight"] == ("F8_E4M3", (1280, 5120))
    assert specs["layers.0.attn.wq_a.scale"] == ("F8_E8M0", (40, 160))
    assert specs["layers.0.attn.wq_b.weight"] == ("F8_E4M3", (32768, 1280))
    assert specs["layers.0.attn.wq_b.scale"] == ("F8_E8M0", (1024, 40))
    assert specs["layers.0.attn.wkv.weight"] == ("F8_E4M3", (512, 5120))
    assert specs["layers.0.attn.wkv.scale"] == ("F8_E8M0", (16, 160))
    assert specs["layers.0.attn.wo_a.weight"] == ("F8_E4M3", (8192, 4096))
    assert specs["layers.0.attn.wo_a.scale"] == ("F8_E8M0", (256, 128))
    assert specs["layers.0.attn.wo_b.weight"] == ("F8_E4M3", (5120, 8192))
    assert specs["layers.0.attn.wo_b.scale"] == ("F8_E8M0", (160, 256))
    assert specs["layers.0.ffn.experts.0.w1.weight"] == ("I8", (2304, 2560))
    assert specs["layers.0.ffn.experts.0.w1.scale"] == ("F8_E8M0", (2304, 160))
    assert specs["layers.0.ffn.experts.0.w2.weight"] == ("I8", (5120, 1152))
    assert specs["layers.0.ffn.experts.0.w2.scale"] == ("F8_E8M0", (5120, 72))

    # Layer 1 (Engram layer)
    assert specs["layers.1.engram.embed.weight"] == ("F8_E4M3", (384006168, 256))
    assert specs["layers.1.engram.embed.scale"] == ("F8_E8M0", (384006168, 8))
    assert specs["layers.1.engram.wkv.weight"] == ("F8_E4M3", (25600, 6144))
    assert specs["layers.1.engram.wkv.scale"] == ("F8_E8M0", (800, 192))

    # Layer 2 (shared KV+Index source, ratio=2 -> has compressor.wgate + indexer.wk/k_norm)
    assert specs["layers.2.attn.compressor.wgate.weight"] == ("BF16", (512, 5120))
    assert specs["layers.2.attn.compressor.wkv.weight"] == ("BF16", (512, 5120))
    assert specs["layers.2.attn.indexer.wq_b.weight"] == ("F8_E4M3", (4096, 1280))
    assert specs["layers.2.attn.indexer.wk.weight"] == ("BF16", (128, 512))

    # Layer 14 (shared KV+Index source, ratio=2 + Engram)
    assert specs["layers.14.engram.embed.weight"] == ("F8_E4M3", (384016682, 256))
    assert specs["layers.14.attn.compressor.wgate.weight"] == ("BF16", (512, 5120))

    # Layer 20 (shared KV+Index source, ratio=1 -> no compressor.wgate, has wk/k_norm)
    assert "layers.20.attn.compressor.wgate.weight" not in specs
    assert specs["layers.20.attn.compressor.wkv.weight"] == ("BF16", (512, 5120))
    assert specs["layers.20.attn.indexer.wk.weight"] == ("BF16", (128, 512))

    # Layer 24 (index-only source -> indexer.wq_b + weights_proj, no wk/k_norm)
    assert specs["layers.24.attn.indexer.wq_b.weight"] == ("F8_E4M3", (4096, 1280))
    assert "layers.24.attn.indexer.wk.weight" not in specs

    # DSpark mtp.0 .. mtp.2
    assert specs["mtp.0.main_proj.weight"] == ("F8_E4M3", (5120, 15360))
    assert specs["mtp.0.ffn.experts.127.w1.weight"] == ("I8", (2304, 2560))
    assert "mtp.0.ffn.experts.128.w1.weight" not in specs
    assert specs["mtp.2.markov_head.embed.weight"] == ("BF16", (129280, 256))
    assert specs["mtp.2.confidence_head.proj.weight"] == ("BF16", (1, 5376))


def test_safetensors_checkpoint_reader(synthetic_ckpt) -> None:
    cfg, ckpt_dir, raw_weights = synthetic_ckpt
    ckpt = SafetensorsCheckpoint(ckpt_dir)
    assert ckpt.load_config() == cfg
    assert set(ckpt.tensor_names()) == set(raw_weights.keys())

    # Coalesced read_many must return bit-identical arrays for all tensors
    loaded = ckpt.read_many(ckpt.tensor_names(), max_workers=4)
    for k, expected in raw_weights.items():
        actual = loaded[k]
        assert actual.shape == expected.shape, f"Shape mismatch on {k}"
        assert actual.dtype == expected.dtype, f"Dtype mismatch on {k}"
        np.testing.assert_array_equal(
            actual.view(np.uint8),
            expected.view(np.uint8),
            err_msg=f"Bit mismatch reading {k} from safetensors",
        )


@pytest.mark.parametrize("world_size", [8, 32])
def test_preshard_and_load_roundtrip(
    synthetic_ckpt,
    tmp_path: Path,
    world_size: int,
) -> None:
    """Tests `preshard_checkpoint` + `load_presharded` for both `tp8` and `tp32`."""
    cfg, ckpt_dir, raw_weights = synthetic_ckpt
    out_dir = tmp_path / f"presharded_tp{world_size}"

    if world_size == 8:
        # Exercise CLI entrypoint for tp8
        rc = preshard_cli_main(
            [
                "--src",
                str(ckpt_dir),
                "--dst",
                str(out_dir),
                "--world-size",
                "8",
                "--workers",
                "4",
            ]
        )
        assert rc == 0
    else:
        # Exercise ensure_presharded / preshard_checkpoint API for tp32
        ensure_presharded(ckpt_dir, out_dir, world_size=32, workers=4, cfg=cfg)

    assert is_presharded(out_dir)

    # 1. Verify directory structure and 4 KiB page alignment
    layout = json.loads((out_dir / "layout.json").read_text())
    assert layout["world_size"] == world_size
    assert layout["ep_lanes"] == 8
    assert layout["tp_hosts"] == world_size // 8
    assert layout["page_align"] == PAGE_ALIGN

    for r in range(world_size):
        rank_dir = out_dir / f"rank{r:02d}"
        dense_bin = rank_dir / "dense.bin"
        index_json = rank_dir / "index.json"
        assert dense_bin.is_file(), f"Missing {dense_bin}"
        assert index_json.is_file(), f"Missing {index_json}"

        file_size = dense_bin.stat().st_size
        assert file_size > 0
        assert file_size % PAGE_ALIGN == 0, (
            f"rank{r:02d}/dense.bin size {file_size} is not a multiple of {PAGE_ALIGN}"
        )

        idx = json.loads(index_json.read_text())
        assert idx["rank"] == r
        assert idx["world_size"] == world_size
        assert idx["total_bytes"] == file_size
        for t_name, t_meta in idx["tensors"].items():
            offset = int(t_meta["offset"])
            nbytes = int(t_meta["nbytes"])
            assert offset % PAGE_ALIGN == 0, (
                f"Tensor {t_name} on rank {r} has unaligned offset {offset}"
            )
            assert offset + nbytes <= file_size

    # Verify Engram host binary files exist and match exact byte size
    for eng_idx, l_id in enumerate(cfg.engram_layer_ids):
        n_emb = cfg.engram_num_embeddings[eng_idx]
        w_bin = out_dir / f"engram/layer_{l_id:02d}.weight.bin"
        s_bin = out_dir / f"engram/layer_{l_id:02d}.scale.bin"
        assert w_bin.is_file()
        assert s_bin.is_file()
        assert w_bin.stat().st_size == n_emb * cfg.engram_head_dim
        assert s_bin.stat().st_size == n_emb * (cfg.engram_head_dim // 32)

    # 2. Load onto virtual mesh and verify sharding + metadata
    mesh = create_v7x_mesh(world_size, axis_name="tp")
    sharded_weights, loaded_cfg, engram_tables = load_presharded(
        out_dir,
        mesh,
        workers=4,
        return_metadata=True,
    )
    assert loaded_cfg == cfg
    assert set(engram_tables.keys()) == set(cfg.engram_layer_ids)

    # 3. Verify exact raw checkpoint reconstruction (max_abs_err == 0.0 across all tensors)
    reconstructed_raw = reconstruct_unsharded_weights(
        sharded_weights,
        cfg,
        dequantize=False,
        engram_host_tables=engram_tables,
    )
    assert set(reconstructed_raw.keys()) == set(raw_weights.keys())
    for k, expected in raw_weights.items():
        actual = reconstructed_raw[k]
        assert actual.shape == expected.shape, f"Raw shape mismatch on {k} (tp{world_size})"
        assert actual.dtype == expected.dtype, f"Raw dtype mismatch on {k} (tp{world_size})"
        np.testing.assert_array_equal(
            actual.view(np.uint8),
            expected.view(np.uint8),
            err_msg=f"Bit mismatch on raw tensor {k} (tp{world_size})",
        )

    # 4. Verify dequantized reconstruction (max_abs_err == 0.0, including BF16 pre-dequantized wo_a)
    expected_deq = dequantize_checkpoint_weights(raw_weights, cfg)
    reconstructed_deq = reconstruct_unsharded_weights(
        sharded_weights,
        cfg,
        dequantize=True,
        engram_host_tables=engram_tables,
    )
    assert set(reconstructed_deq.keys()) == set(expected_deq.keys())
    for k, expected_val in expected_deq.items():
        actual_val = reconstructed_deq[k]
        if k.endswith(".attn.wo_a.weight"):
            # wo_a.weight is stored in dense.bin as BF16 per spec.md §1
            expected_bf16_f32 = expected_val.astype(ml_dtypes.bfloat16).astype(np.float32)
            max_err = float(np.max(np.abs(actual_val.astype(np.float32) - expected_bf16_f32)))
        else:
            max_err = float(
                np.max(np.abs(actual_val.astype(np.float32) - expected_val.astype(np.float32)))
            )
        assert max_err == 0.0, f"Non-zero dequant error ({max_err}) on {k} (tp{world_size})"

    # 5. Verify routed expert uint32 MXFP4 slices unpack via Pallas `pltpu.bitcast` with 0.0 error
    tp_hosts = world_size // 8
    inter_per_host = cfg.moe_inter_dim // tp_hosts
    u32_w1 = np.asarray(sharded_weights["layers.0.ffn.experts.w1.u32"][0, 0])
    scale_w1 = np.asarray(sharded_weights["layers.0.ffn.experts.w1.scale"][0, 0])
    fp8_from_bitcast = np.asarray(pallas_bitcast_mxfp4_to_fp8(jnp.asarray(u32_w1)))
    scale_f32 = np.ldexp(
        np.ones_like(scale_w1, dtype=np.float32),
        scale_w1.astype(np.int32) - 127,
    )
    deq_from_bitcast = (
        fp8_from_bitcast.astype(np.float32).reshape(cfg.dim // 32, 32, inter_per_host)
        * scale_f32[:, None, :]
    ).reshape(cfg.dim, inter_per_host).T
    ref_deq_w1 = dequant_mxfp4_block32(
        raw_weights["layers.0.ffn.experts.0.w1.weight"][:inter_per_host, :],
        raw_weights["layers.0.ffn.experts.0.w1.scale"][:inter_per_host, :],
        block_size=32,
    )
    assert float(np.max(np.abs(deq_from_bitcast - ref_deq_w1))) == 0.0

    # 6. Verify parity with xla_decode.shard_weights(cfg, raw_weights, mesh)
    ref_sharded = shard_weights(cfg, raw_weights, mesh)
    for k, ref_arr in ref_sharded.items():
        assert k in sharded_weights, f"Missing key {k} required by xla_decode.shard_weights"
        loaded_arr = sharded_weights[k]
        assert loaded_arr.shape == ref_arr.shape, (
            f"Shape mismatch vs xla_decode.shard_weights on {k}: {loaded_arr.shape} vs {ref_arr.shape}"
        )
        assert loaded_arr.dtype == ref_arr.dtype, (
            f"Dtype mismatch vs xla_decode.shard_weights on {k}: {loaded_arr.dtype} vs {ref_arr.dtype}"
        )
        np.testing.assert_array_equal(
            np.asarray(loaded_arr).view(np.uint8),
            np.asarray(ref_arr).view(np.uint8),
            err_msg=f"Value mismatch vs xla_decode.shard_weights on {k} (tp{world_size})",
        )
