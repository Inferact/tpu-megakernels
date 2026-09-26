"""NVFP4 expert path of Muse Spark: `musespark.quant` fp4 helpers, the format-v2 container
(`load.convert_presharded_nvfp4`), `fp4.fp4_block_dot` and the prefill/reference dequant path.

CPU (`JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8`): every test,
Pallas bodies in interpret mode. TPU (single chip): the same dot checks on real per-rank slices
(`tpu_run.sh 3 python -m pytest tests/test_musespark_fp4.py`).
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

import musespark
import musespark.prefill as prefill_mod
from musespark import MINI, Config, fp4, layout, load, quant

ON_TPU = jax.devices()[0].platform == "tpu"
INTERPRET = not ON_TPU
TP = 8
VM = pl.BlockSpec(memory_space=pltpu.VMEM)
FIXTURES = Path(
    "/filestore/tmp/claude-0/-filestore-weights/bc41d5f3-5fda-4ab7-9ca1-48e0c99553cc/scratchpad/nvfp4"
)
E4M3 = ml_dtypes.float8_e4m3fn


def _rng(seed):
    return np.random.default_rng(seed)


def random_e4m3(rng, shape, lo=8, hi=16):
    """Random e4m3 scales with biased exponents `lo..hi-1` and random mantissas (no NaN)."""
    bits = (rng.integers(lo, hi, size=shape) << 3) | rng.integers(0, 7, size=shape)
    return bits.astype(np.uint8).view(E4M3)


# ---------------------------------------------------------------------------------------------
# quant helpers
# ---------------------------------------------------------------------------------------------


def test_pack_unpack_and_checkpoint_transpose():
    rng = _rng(0)
    codes = rng.integers(0, 16, size=(3, 64, 128), dtype=np.uint8)
    packed = quant.pack_fp4_rows(codes)
    assert packed.dtype == np.int32 and packed.shape == (3, 8, 128)
    assert np.array_equal(quant.unpack_fp4_rows(packed), codes)
    # nibble j of word k' is row 8k'+j (low nibble first)
    word = packed.view(np.uint32)[0, 0, 5]
    assert [(int(word) >> (4 * j)) & 0xF for j in range(8)] == codes[0, :8, 5].tolist()
    # checkpoint rows [N, K/2] (low nibble = even k) -> [K/8, N] is a plain uint32 transpose
    rows = rng.integers(0, 256, size=(5, 128, 32), dtype=np.uint8)
    slow = quant.pack_fp4_rows(np.swapaxes(quant.unpack_nvfp4_codes(rows), -1, -2))
    assert np.array_equal(quant.nvfp4_rows_to_packed(rows), slow)
    assert np.array_equal(quant.e2m1_values(np.arange(16)), quant.E2M1_TABLE)
    assert np.array_equal(np.asarray(quant.e2m1_values_jnp(jnp.arange(16))), quant.E2M1_TABLE)


def test_e2m1_encode_modelopt_rounding():
    v = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, -0.75, -0.3, 0.0, 2.9, -6.0])
    assert quant.e2m1_encode(v).tolist() == [0, 2, 2, 4, 4, 6, 6, 7, 10, 9, 0, 5, 15]


def test_dequant_np_equals_jnp_and_bits():
    rng = _rng(1)
    packed = quant.pack_fp4_rows(rng.integers(0, 16, size=(5, 64, 128), dtype=np.uint8))
    bs = random_e4m3(rng, (5, 4, 128))
    gs = rng.uniform(1e-5, 1e-4, size=(5, 1, 1)).astype(np.float32)
    d = quant.dequant_fp4_np(packed, bs, gs)
    assert d.shape == (5, 64, 128) and d.dtype == np.float32
    assert np.array_equal(np.asarray(quant.dequant_fp4_jnp(jnp.asarray(packed), jnp.asarray(bs), gs)), d)
    assert np.array_equal(np.asarray(quant.dequant_fp4(jnp.asarray(packed), jnp.asarray(bs.view(np.uint8)), gs)), d)
    chunked = quant.fp4_scales_to_chunked(bs, 64)
    assert chunked.shape == (5, 1, 4, 128) and chunked.dtype == E4M3
    assert np.array_equal(quant.dequant_fp4_np(packed, chunked, gs), d)
    # e2m1 * e4m3 is exact in bf16; only the global multiply rounds
    exact = quant.dequant_fp4_np(packed, bs)
    assert np.array_equal(exact, exact.astype(ml_dtypes.bfloat16).astype(np.float32))


def test_quantize_nvfp4_round_trip_error():
    rng = _rng(2)
    w = rng.standard_normal((4096, 1024)).astype(np.float32) / 64
    packed, bs, gs = quant.quantize_nvfp4_np(w)
    assert packed.shape == (512, 1024) and bs.shape == (256, 1024) and bs.dtype == E4M3
    assert gs.shape == () and gs == np.float32(np.abs(w).max() / (6 * 448))
    assert bs.astype(np.float32).max() == 448.0  # the amax block scales to exactly 448
    d = quant.dequant_fp4_np(packed, bs, gs)
    rel = np.sqrt(np.mean((w - d) ** 2)) / np.sqrt(np.mean(w**2))
    assert 0.08 < rel < 0.11  # ~9.5% RMS for Gaussian weights (int4 g128 RTN: ~11.7%)
    q4, s4 = quant.quantize_int4_np(w, 128)
    rel4 = np.sqrt(np.mean((w - quant.dequantize_int4_np(q4, s4)) ** 2)) / np.sqrt(np.mean(w**2))
    assert rel < rel4
    # every block uses the full e2m1 range (absmax RTN), like the vendor's tensors
    blocks = np.abs(quant.e2m1_values(quant.unpack_fp4_rows(packed))).reshape(256, 16, 1024)
    assert np.all(blocks.max(axis=1) == 6.0)
    # idempotent: re-quantizing the dequantized values with the same gs reproduces everything
    packed2, bs2, _ = quant.quantize_nvfp4_np(d, gs=gs)
    assert np.array_equal(bs2.view(np.uint8), bs.view(np.uint8))
    assert np.array_equal(quant.unpack_fp4_rows(packed2) & 7, quant.unpack_fp4_rows(packed) & 7)
    # batched leading axes and zero tensors
    packed, bs, gs = quant.quantize_nvfp4_np(np.zeros((2, 32, 128), np.float32))
    assert np.all(gs == 1) and not packed.any()


@pytest.mark.skipif(not (FIXTURES / "gu_w_l1_e0_rows0_256.bin").is_file(), reason="no fixtures")
def test_vendor_fixture_layer1_expert0():
    """Real bytes of the vendor checkpoint (layer 1, expert 0, gate rows 0..256): our decode
    reproduces the vendor's scales exactly and every code up to the sign of zero."""
    gu = np.fromfile(FIXTURES / "gu_w_l1_e0_rows0_256.bin", np.uint8).reshape(256, 2048)
    ws = np.fromfile(FIXTURES / "gu_ws_l1_e0.bin", np.uint8).reshape(8192, 256)[:256]
    ws2 = np.fromfile(FIXTURES / "gu_ws2_l1.bin", np.float32).reshape(256, 2)[0]
    packed = quant.nvfp4_rows_to_packed(gu)  # [512, 256] = [Hm/8, rows]
    bs = np.ascontiguousarray(ws.T).view(E4M3)  # [Hm/16, rows]
    w = quant.dequant_fp4_np(packed, bs, ws2[0])
    assert w.shape == (4096, 256) and 0.02 < np.abs(w).max() < 0.03
    assert quant.nvfp4_global_scale(w) == ws2[0]  # the gate half's amax lies in these rows
    packed2, bs2, _ = quant.quantize_nvfp4_np(w, gs=ws2[0])
    assert np.array_equal(bs2.view(np.uint8), bs.view(np.uint8))
    a, b = quant.unpack_fp4_rows(packed), quant.unpack_fp4_rows(packed2)
    mismatch = a != b
    assert np.all((a[mismatch] == 8) & (b[mismatch] == 0))  # only -0 vs +0
    blocks = np.abs(quant.e2m1_values(a)).reshape(256, 16, 256).max(axis=1)
    assert np.all(blocks == 6.0)


# ---------------------------------------------------------------------------------------------
# layout / container
# ---------------------------------------------------------------------------------------------


def test_layout_nvfp4_families():
    shapes = layout.rank_shapes(MINI, TP, "nvfp4")
    assert set(shapes) == set(layout.rank_shapes(MINI, TP)) - set(layout.EXPERT_FAMILIES) | set(
        layout.FP4_EXPERT_FAMILIES
    )
    assert shapes["gate_up_fp4"] == ((4, 16, 64, 128), jnp.int32)
    assert shapes["gate_up_bs"] == ((4, 16, 1, 32, 128), jnp.float8_e4m3fn)
    assert shapes["down_fp4"] == ((4, 16, 8, 512), jnp.int32)
    assert shapes["down_bs"] == ((4, 16, 1, 4, 512), jnp.float8_e4m3fn)
    assert shapes["expert_gs"] == ((4, 16, 8, 128), jnp.float32)
    real = layout.rank_shapes(Config(), TP, "nvfp4")
    assert real["gate_up_fp4"][0] == (62, 256, 512, 1024)
    assert real["gate_up_bs"][0] == (62, 256, 8, 32, 1024)
    assert real["down_fp4"][0] == (62, 256, 64, 4096)
    assert real["down_bs"][0] == (62, 256, 1, 32, 4096)
    b = layout.bytes_per_rank(Config(), TP, "nvfp4")
    assert b["experts"] == 62 * 256 * (4096 * 1024 + 512 * 4096) // 2
    assert 57 * 2**30 < b["total"] < 58 * 2**30
    assert layout.expert_format_of(shapes) == "nvfp4"
    assert layout.expert_format_of(layout.rank_shapes(MINI, TP)) == "int4"
    with pytest.raises(ValueError):
        layout.rank_shapes(MINI, TP, "fp8")


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("musespark-nvfp4")
    return load.synthetic_presharded_nvfp4(tmp, MINI, TP, seed=3, workers=4, experts_per_task=4)


def _same_bits(name, got, want):
    assert got.shape == np.shape(want), (name, got.shape, np.shape(want))
    assert np.array_equal(got.view(np.uint8), np.asarray(want).view(np.uint8)), name


def test_container_v2_layout_and_experts(converted):
    src, dst, tensors = converted
    doc = load.read_layout(dst)
    assert doc["format"] == load.FORMAT_V2 and load.layout_expert_format(doc) == "nvfp4"
    assert load.container_expert_format(dst) == "nvfp4"
    assert doc["expert_source"] == {"0": "rtn", "1": "vendor", "2": "vendor", "3": "rtn"}
    assert doc["layer_kinds"] == ["nvfp4"] * MINI.layers and doc["fp4"] == load.FP4_ENCODING
    assert list(doc["arrays"]) == list(layout.rank_shapes(MINI, TP, "nvfp4"))
    assert doc["arrays"]["gate_up_bs"]["disk_dtype"] == "uint8"
    assert doc["arrays"]["gate_up_bs"]["dtype"] == "float8_e4m3fn"
    assert load.make_layout(MINI, TP, doc["revision"], "nvfp4", doc["expert_source"]) == doc
    assert load.is_complete(dst) and (dst / "hf_quant_config.json").is_file()
    arrays = load.read_presharded(dst)
    assert set(arrays) == set(doc["arrays"])
    assert arrays["gate_up_bs"].dtype == E4M3 and arrays["gate_up_fp4"].dtype == np.int32
    p = load.PREFIX + "layers.{}.mlp.experts."
    for l in load.nvfp4_layers(MINI):  # vendor layers: pure byte re-layout, bit-exact
        for e in range(MINI.experts):
            for r in range(TP):
                want = load.nvfp4_expert_rank_arrays_slow(
                    MINI, TP, r,
                    tensors[p.format(l) + "gate_up_proj"][e],
                    tensors[p.format(l) + "gate_up_proj_weight_scale"][e],
                    tensors[p.format(l) + "down_proj"][e],
                    tensors[p.format(l) + "down_proj_weight_scale"][e],
                )
                for name, value in want.items():
                    _same_bits(name, arrays[name][r, l, e], value)
            gs = arrays["expert_gs"][:, l, e]
            ws2 = tensors[p.format(l) + "gate_up_proj_weight_scale_2"][e]
            assert np.all(gs[:, 0] == ws2[0]) and np.all(gs[:, 1] == ws2[1])
            assert np.all(gs[:, 2] == tensors[p.format(l) + "down_proj_weight_scale_2"][e])
            assert not gs[:, 3:].any()
    for l in (0, MINI.layers - 1):  # bf16 layers: RTN-quantized to the same format
        for e in range(MINI.experts):
            want = load.bf16_expert_to_nvfp4(
                MINI, TP, tensors[p.format(l) + "gate_up_proj"][e], tensors[p.format(l) + "down_proj"][e]
            )
            for r in range(TP):
                for name, value in want[r].items():
                    _same_bits(name, arrays[name][r, l, e], value)
    # dense/global families: identical to the reference's shard_canonical
    get = tensors.__getitem__
    canonical = musespark.canonical_global_from_checkpoint(MINI, get)
    per_layer = [musespark.canonical_layer_from_checkpoint(MINI, l, get, experts=False)
                 for l in range(MINI.layers)]
    for name in per_layer[0]:
        canonical[name] = np.stack([lw[name] for lw in per_layer])
    L, E, Hm, I, G = MINI.layers, MINI.experts, MINI.moe_hidden, MINI.expert_hidden, MINI.group_size
    canonical.update({
        "gate_up_q": np.zeros((L, E, Hm, 2 * I), np.int8), "gate_up_s": np.ones((L, E, Hm // G, 2 * I), np.float32),
        "down_q": np.zeros((L, E, I, Hm), np.int8), "down_s": np.ones((L, E, I // G, Hm), np.float32),
    })
    expected = musespark.shard_canonical(MINI, canonical, TP)
    for name, want in expected.items():
        if name not in layout.EXPERT_FAMILIES:
            assert arrays[name].dtype == want.dtype
            _same_bits(name, arrays[name], want)


def test_dequantized_vendor_expert_matches_checkpoint_side_reference(converted):
    _, dst, tensors = converted
    l, e, r = 1, 2, 3
    I, Is = MINI.expert_hidden, MINI.expert_hidden // TP
    p = load.PREFIX + f"layers.{l}.mlp.experts."
    gs = load.read_rank_array(dst, r, "expert_gs", (l, e))
    col = np.concatenate([np.full(Is, gs[0, 0]), np.full(Is, gs[1, 0])]).astype(np.float32)[None]
    w = quant.dequant_fp4_np(load.read_rank_array(dst, r, "gate_up_fp4", (l, e)),
                             load.read_rank_array(dst, r, "gate_up_bs", (l, e)), col)
    rows = np.r_[r * Is:(r + 1) * Is, I + r * Is:I + (r + 1) * Is]
    codes = quant.unpack_nvfp4_codes(tensors[p + "gate_up_proj"][e])[rows]  # [2Is, Hm]
    scales = tensors[p + "gate_up_proj_weight_scale"][e][rows].astype(np.float32)  # [2Is, Hm/16]
    ref = quant.e2m1_values(codes).T * np.repeat(scales, 16, axis=1).T * col
    assert np.array_equal(w, ref)
    d = quant.dequant_fp4_np(load.read_rank_array(dst, r, "down_fp4", (l, e)),
                             load.read_rank_array(dst, r, "down_bs", (l, e)), gs[2, 0])
    codes = quant.unpack_nvfp4_codes(tensors[p + "down_proj"][e])[:, r * Is:(r + 1) * Is]  # [Hm, Is]
    scales = tensors[p + "down_proj_weight_scale"][e][:, r * (Is // 16):(r + 1) * (Is // 16)]
    ref = quant.e2m1_values(codes).T * np.repeat(scales.astype(np.float32), 16, axis=1).T * gs[2, 0]
    assert np.array_equal(d, ref)


def test_resume_and_idempotence(converted, tmp_path):
    src, dst, _ = converted
    arrays = load.read_presharded(dst)
    out = tmp_path / "partial"
    quiet = dict(log=lambda _: None, workers=4)
    progress = load.convert_presharded_nvfp4(src, out, tp=TP, shards=2, config=MINI, **quiet)
    assert not progress["complete"] and 0 < len(progress["units"]) < 19
    partial = json.loads((out / "progress.json").read_text())["units"]
    with pytest.raises(ValueError, match="not complete"):
        load.load_presharded(None, out)
    progress = load.convert_presharded_nvfp4(src, out, tp=TP, config=MINI, **quiet)
    assert progress["complete"] and progress["layers"] == list(range(MINI.layers))
    assert set(partial) < set(progress["units"])
    for name, want in arrays.items():
        _same_bits(name, load.read_presharded(out, [name])[name], want)
    with pytest.raises(ValueError, match="differs"):
        load.convert_presharded_nvfp4(src, out, tp=4, config=MINI, **quiet)


def test_convert_nvfp4_cli(tmp_path, monkeypatch):
    src = load.write_checkpoint(tmp_path / "ckpt", MINI, load.random_checkpoint_tensors(MINI, 5, nvfp4=True),
                                shards=4, nvfp4=True)
    dst = tmp_path / "out"
    code = load._cli(["convert-nvfp4", "--src", str(src), "--dst", str(dst), "--tp", "2",
                      "--workers", "2", "--shards", "2"])
    assert code == 1 and not load.read_progress(dst)["complete"]
    doc = load.read_layout(dst)
    assert doc["tp"] == 2 and doc["format"] == load.FORMAT_V2
    assert tuple(doc["arrays"]["gate_up_fp4"]["shape"]) == (MINI.layers, MINI.experts, 64, 512)


# ---------------------------------------------------------------------------------------------
# fp4_block_dot (interpret on CPU, real on TPU)
# ---------------------------------------------------------------------------------------------


def run_fp4_block_dot(x_bd, packed, bs_chunked, rows):
    def kernel(x_ref, w_ref, s_ref, o_ref):
        o_ref[...] = fp4.fp4_block_dot(x_ref, w_ref, s_ref, rows)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, packed.shape[1]), jnp.float32),
        in_specs=[VM, VM, VM],
        out_specs=VM,
        interpret=INTERPRET,
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=48 << 20),
    )(x_bd, packed, bs_chunked)


@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("K,N", [(4096, 1024), (512, 4096), (512, 128), (64, 512)])
def test_fp4_block_dot_matches_dequant_reference(K, N, rows):
    rng = _rng(K * 7 + N + rows)
    packed = quant.pack_fp4_rows(rng.integers(0, 16, size=(K, N), dtype=np.uint8))
    bs = random_e4m3(rng, (K // 16, N))
    x = rng.standard_normal((rows, K)).astype(ml_dtypes.bfloat16).astype(np.float32)
    ref = x @ quant.dequant_fp4_np(packed, bs)
    x_bd = fp4.block_diag16(jnp.asarray(x))
    assert x_bd.shape == fp4.block_diag16_shape(K, rows)
    out = np.asarray(run_fp4_block_dot(x_bd, jnp.asarray(packed), jnp.asarray(quant.fp4_scales_to_chunked(bs, K)), rows))
    assert np.abs(out - ref).max() <= 1e-3 * np.abs(ref).max()
    # block-diagonal expansion: row block g of chunk i holds x masked to group g
    chunk = np.asarray(x_bd[0]).astype(np.float32)
    kc = x_bd.shape[2]
    for g in (0, kc // 16 - 1):
        block = chunk[g * rows:(g + 1) * rows]
        assert np.array_equal(block[:, g * 16:(g + 1) * 16], x[:, g * 16:(g + 1) * 16])
        assert not np.delete(block, np.s_[g * 16:(g + 1) * 16], axis=1).any()


def test_global_scales_and_scratch():
    rng = _rng(9)
    gs = load.expert_gs_tile(2.0, 3.0, 5.0)
    gu = rng.standard_normal((4, 128)).astype(np.float32)
    got = np.asarray(fp4.apply_gate_up_gs(jnp.asarray(gu), jnp.asarray(gs), 64))
    assert np.array_equal(got[:, :64], gu[:, :64] * 2) and np.array_equal(got[:, 64:], gu[:, 64:] * 3)
    assert np.array_equal(np.asarray(fp4.apply_down_gs(jnp.asarray(gu), jnp.asarray(gs))), gu * 5)
    shapes = fp4.slot_shapes(Config(), TP)
    assert shapes["gate_up_fp4"] == ((4, 512, 1024), jnp.int32)
    assert shapes["gate_up_bs"] == ((4, 8, 32, 1024), jnp.float8_e4m3fn)
    assert shapes["down_fp4"] == ((4, 64, 4096), jnp.int32)
    assert shapes["down_bs"] == ((4, 1, 32, 4096), jnp.float8_e4m3fn)
    assert 16 * 2**20 < fp4.scratch_bytes(Config(), 8) < 17 * 2**20
    assert 13 * 2**20 < fp4.scratch_bytes(Config(), 1) < 14 * 2**20
    assert len(fp4.scratch_shapes(MINI, 4)) == len(fp4.Fp4Scratch.__dataclass_fields__)


@pytest.mark.parametrize("batch", [1, 4])
def test_expert_slot_dots_in_kernel(converted, batch):
    """`gate_up_dot` / `down_dot` on a real container slot (MINI shapes, one expert) against the
    prefill-style bf16 dequant (`prefill.dequantized_experts`)."""
    _, dst, _ = converted
    l, e, r = 2, 5, 1
    Is = MINI.expert_hidden // TP
    names = ("gate_up_fp4", "gate_up_bs", "down_fp4", "down_bs", "expert_gs")
    slots = {n: jnp.asarray(load.read_rank_array(dst, r, n, (l, e)))[None] for n in names}
    rng = _rng(batch)
    h1 = rng.standard_normal((batch, MINI.moe_hidden)).astype(ml_dtypes.bfloat16).astype(np.float32)

    def kernel(h_ref, gu_ref, gbs_ref, dn_ref, dbs_ref, gs_ref, gu_out, y_out, h_bd):
        sc = fp4.Fp4Scratch(gu_ref, gbs_ref, dn_ref, dbs_ref, gs_ref, h_bd, *([None] * 6))
        fp4.block_diag16_to_ref(h_bd, h_ref[...])
        gu = fp4.gate_up_dot(sc, 0, batch)
        gu_out[...] = gu
        a = (jax.nn.silu(gu[:, :Is]) * gu[:, Is:]).astype(jnp.bfloat16).astype(jnp.float32)
        y_out[...] = fp4.down_dot(sc, 0, a)

    gu, y = pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((batch, 2 * Is), jnp.float32),
                   jax.ShapeDtypeStruct((batch, MINI.moe_hidden), jnp.float32)),
        in_specs=[VM] * 6,
        out_specs=(VM, VM),
        scratch_shapes=[pltpu.VMEM(fp4.block_diag16_shape(MINI.moe_hidden, batch), jnp.bfloat16)],
        interpret=INTERPRET,
    )(jnp.asarray(h1), *(slots[n] for n in names))
    w = {n: jnp.asarray(load.read_rank_array(dst, r, n))[:, e:e + 1] for n in names}
    gate_up, down = prefill_mod.dequantized_experts(w, l, Is)
    gu_ref = h1 @ np.asarray(gate_up[0]).astype(np.float32)
    assert np.abs(np.asarray(gu) - gu_ref).max() <= 2e-2 * np.abs(gu_ref).max()  # bf16 weights
    a = np.asarray(musespark.r16(jax.nn.silu(jnp.asarray(gu)[:, :Is]) * jnp.asarray(gu)[:, Is:]))
    y_ref = a @ np.asarray(down[0]).astype(np.float32)
    assert np.abs(np.asarray(y) - y_ref).max() <= 2e-2 * np.abs(y_ref).max()


# ---------------------------------------------------------------------------------------------
# loader + prefill + reference on a v2 container (8 host devices)
# ---------------------------------------------------------------------------------------------


def _mesh():
    if jax.device_count() < TP:
        pytest.skip("needs XLA_FLAGS=--xla_force_host_platform_device_count=8")
    return jax.sharding.Mesh(np.array(jax.devices()[:TP]), ("tp",))


def _dense_canonical(cfg, tensors, container_arrays):
    """Reference canonical dict: dense entries from the HF tensors, dense bf16 experts equal to
    the exact dequantization of the container's fp4 families (reassembled over the ranks)."""
    get = tensors.__getitem__
    canonical = musespark.canonical_global_from_checkpoint(cfg, get)
    per_layer = [musespark.canonical_layer_from_checkpoint(cfg, l, get, experts=False)
                 for l in range(cfg.layers)]
    for name in per_layer[0]:
        canonical[name] = np.stack([lw[name] for lw in per_layer])
    a = container_arrays
    Is = cfg.expert_hidden // TP
    gate_ups, downs = [], []
    for r in range(TP):
        gs = a["expert_gs"][r]  # [L, E, 8, 128]
        col = np.concatenate([np.repeat(gs[..., 0:1, 0:1], Is, axis=-1),
                              np.repeat(gs[..., 1:2, 0:1], Is, axis=-1)], axis=-1)  # [L, E, 1, 2Is]
        gate_ups.append(quant.dequant_fp4_np(a["gate_up_fp4"][r], a["gate_up_bs"][r], col))
        downs.append(quant.dequant_fp4_np(a["down_fp4"][r], a["down_bs"][r], gs[..., 2:3, 0:1]))
    gate_up = np.concatenate([g[..., :Is] for g in gate_ups] + [g[..., Is:] for g in gate_ups], axis=-1)
    canonical["gate_up"] = gate_up.astype(ml_dtypes.bfloat16)
    canonical["down"] = np.concatenate(downs, axis=-2).astype(ml_dtypes.bfloat16)
    return canonical


def test_load_presharded_v2_and_prefill_matches_reference(converted):
    _, dst, tensors = converted
    mesh = _mesh()
    weights = load.load_presharded(mesh, dst, MINI, layer_chunk_bytes=1 << 12)
    abstract = load.abstract_weights(mesh, MINI, TP, "nvfp4")
    assert set(weights) == set(abstract)
    arrays = load.read_presharded(dst)
    for name, value in weights.items():
        assert value.shape == abstract[name].shape and value.dtype == abstract[name].dtype, name
        host = np.asarray(value)
        want = arrays[name]
        assert host.dtype == want.dtype, name
        assert np.array_equal(host.view(np.uint8), want.view(np.uint8)), name
    assert weights["gate_up_bs"].dtype == jnp.float8_e4m3fn and weights["gate_up_fp4"].dtype == jnp.int32
    with pytest.raises(ValueError, match="does not match"):
        load.load_presharded(mesh, dst, Config(**{**MINI.__dict__, "experts": 8}))

    # prefill over the container vs the pure-JAX reference on the dequantized bf16 experts
    cfg = MINI
    canonical = _dense_canonical(cfg, tensors, arrays)
    rng = np.random.default_rng(0)
    for name in ("attn_gate", "ffn_gate"):  # damped residual gates (see test_musespark_prefill)
        g = (-0.3 + 0.05 * rng.standard_normal((cfg.layers, cfg.hidden))).astype(np.float32)
        alpha, beta = musespark.gate_coeffs(jnp.asarray(g), cfg.gate_temperature)
        canonical[name + "_alpha"], canonical[name + "_beta"] = np.asarray(alpha), np.asarray(beta)
    bias = np.asarray(canonical["router_bias"], np.float32).copy()
    for layer in range(cfg.layers):
        chosen = (layer * 5 + np.arange(cfg.top_k) * (cfg.experts // cfg.top_k)) % cfg.experts
        bias[layer, chosen] += 2.0
    canonical["router_bias"] = bias
    # push the modified gates/bias into the container weights too
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("tp"))
    for name in ("attn_gate_alpha", "attn_gate_beta", "ffn_gate_alpha", "ffn_gate_beta", "router_bias"):
        value = np.broadcast_to(np.asarray(canonical[name])[None, :, None], (TP, cfg.layers, 1, cfg.hidden if "gate" in name else cfg.experts))
        weights[name] = jax.device_put(jnp.asarray(np.ascontiguousarray(value)), sharding)
    T, context = 37, 512
    tokens = np.random.default_rng(1).integers(0, cfg.vocab_used, T).astype(np.int32)
    caches = musespark.init_caches(cfg, 1, context)
    run = jax.jit(lambda w, t, c: musespark.forward(cfg, w, t, jnp.zeros(1, jnp.int32), c, return_hidden=True))
    ref = run(musespark.to_device(canonical), jnp.asarray(tokens)[None], caches)
    ref_logits, ref_hidden = np.asarray(ref[0][0]), np.asarray(ref[2])[:, 0]
    prefill = prefill_mod.make_prefill(mesh, cfg, context, TP, taps=True)
    padded, length = prefill_mod.pad_prompt(cfg, tokens)
    logits, _, hidden = prefill(weights, load.zero_caches(mesh, cfg, 2, context), padded, length, 0)
    logits, hidden = np.asarray(logits), np.asarray(hidden)[:, :T]
    per_layer = np.abs(hidden - ref_hidden).reshape(cfg.layers + 1, -1).max(axis=1)
    print("residual max |diff| per layer", np.round(per_layer, 3))
    assert per_layer[0] == 0 and np.all(per_layer <= 0.1)
    diff = np.abs(logits - ref_logits[T - 1]).max()
    print("max |logit diff|", diff)
    assert diff <= 5e-2 and int(np.argmax(logits)) == int(np.argmax(ref_logits[T - 1]))


def test_reference_expert_weights_fp4_branch(converted):
    _, dst, _ = converted
    arrays = load.read_presharded(dst)
    r, l = 2, 1
    lw = {n: jnp.asarray(arrays[n][r, l]) for n in layout.FP4_EXPERT_FAMILIES}
    gate_up, down = musespark.expert_weights(lw)
    Is = MINI.expert_hidden // TP
    gs = arrays["expert_gs"][r, l]
    col = np.concatenate([np.repeat(gs[:, 0:1, 0:1], Is, -1), np.repeat(gs[:, 1:2, 0:1], Is, -1)], -1)
    want = quant.dequant_fp4_np(arrays["gate_up_fp4"][r, l], arrays["gate_up_bs"][r, l], col)
    assert gate_up.dtype == jnp.bfloat16 and np.array_equal(np.asarray(gate_up), want.astype(ml_dtypes.bfloat16))
    want = quant.dequant_fp4_np(arrays["down_fp4"][r, l], arrays["down_bs"][r, l], gs[:, 2:3, 0:1])
    assert np.array_equal(np.asarray(down), want.astype(ml_dtypes.bfloat16))
    assert set(musespark.layer_weights({**{n: arrays[n][r] for n in layout.FP4_EXPERT_FAMILIES}}, 0)) == set(layout.FP4_EXPERT_FAMILIES)
