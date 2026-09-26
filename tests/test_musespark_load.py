"""CPU tests of the Muse Spark checkpoint conversion and the pre-sharded container.

Synthetic HF checkpoint (real tensor names/dtypes/shapes for the MINI config) ->
`convert_presharded` -> container read back with numpy and compared bit-exactly against the
reference's `shard_canonical`. The device-placement test needs eight host devices:
`JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8`.
"""

import json
import time

import jax
import numpy as np
import pytest

import musespark
from musespark import MINI, Config, layout, load, quant


TP = 8
CONVERT = dict(workers=4, experts_per_task=3)


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("musespark")
    src, dst, tensors = load.synthetic_presharded(tmp, MINI, TP, seed=7, **CONVERT)
    canonical = load.canonical_from_tensors(MINI, tensors)
    expected = musespark.shard_canonical(MINI, canonical, TP)
    return src, dst, tensors, expected


def assert_same(name, got, want):
    assert got.shape == want.shape, (name, got.shape, want.shape)
    assert got.dtype == want.dtype, (name, got.dtype, want.dtype)
    assert np.array_equal(got.view(np.uint8), want.view(np.uint8)), name


def test_container_matches_shard_canonical(converted):
    _, dst, _, expected = converted
    arrays = load.read_presharded(dst)
    assert set(arrays) == set(expected) == set(layout.rank_shapes(MINI, TP))
    for name, want in expected.items():
        assert_same(name, arrays[name], want)


def test_layout_json_round_trip(converted):
    _, dst, _, expected = converted
    doc = load.read_layout(dst)
    assert doc["format"] == load.FORMAT and doc["tp"] == TP and doc["group"] == MINI.group_size
    assert load.config_from_layout(doc) == MINI
    assert doc["revision"] == load.Checkpoint(converted[0]).revision
    shapes = layout.rank_shapes(MINI, TP)
    assert list(doc["arrays"]) == list(shapes)
    total = 0
    for name, (shape, dtype) in shapes.items():
        spec = doc["arrays"][name]
        assert tuple(spec["shape"]) == shape and load._np_dtype(spec["dtype"]) == np.dtype(dtype)
        assert spec["nbytes"] == layout.nbytes(shape, dtype)
        assert spec["layer_axis"] == (None if name in load.GLOBAL_NAMES else 0)
        for rank in range(TP):
            assert load.rank_file(dst, rank, name).stat().st_size == spec["nbytes"]
        total += TP * spec["nbytes"]
    assert doc["total_bytes"] == total == TP * layout.bytes_per_rank(MINI, TP)["total"]
    assert doc["int4"] == load.INT4_ENCODING
    assert load.make_layout(MINI, TP, doc["revision"]) == doc
    assert load.is_complete(dst)


def test_int4_disk_packing(converted):
    _, dst, _, expected = converted
    packed = load.read_rank_array(dst, 1, "gate_up_q", (2, 3), unpack=False)
    assert packed.dtype == np.uint8
    want = expected["gate_up_q"][1, 2, 3]
    assert np.array_equal(packed, quant.pack_int4(want))
    assert np.array_equal(quant.unpack_int4(packed), want.astype(np.int8))
    # low nibble first: byte 0 holds elements 0 (low) and 1 (high) of the last axis
    q = want.astype(np.int8)
    assert int(packed[0, 0]) == ((int(q[0, 0]) & 0xF) | ((int(q[0, 1]) & 0xF) << 4))


def test_checkpoint_streaming_reads(converted):
    src, _, tensors, _ = converted
    checkpoint = load.Checkpoint(src)
    assert checkpoint.config == Config(**{**MINI.__dict__, "group_size": Config().group_size})
    key = load.layer_key(1, "mlp.experts.gate_up_proj")
    full = checkpoint.read(key)
    assert full.dtype == np.dtype(load.BF16) and np.array_equal(
        full.view(np.uint16), tensors[key].view(np.uint16)
    )
    part = checkpoint.read(key, slice(3, 7), threads=2)
    assert np.array_equal(part.view(np.uint16), tensors[key][3:7].view(np.uint16))
    many = checkpoint.read_many([load.GLOBAL_KEYS["embed"], load.layer_key(0, "mlp.gate.weight")])
    for k, v in many.items():
        assert np.array_equal(v.view(np.uint8), tensors[k].view(np.uint8))
    assert set(checkpoint.read_layer(2)) == set(load.LAYER_SUFFIXES)
    assert checkpoint.bytes_read > 0 and checkpoint.gbps() > 0


def test_resume_and_idempotence(tmp_path):
    src = load.write_checkpoint(tmp_path / "ckpt", MINI, load.random_checkpoint_tensors(MINI, 3))
    tensors = load.random_checkpoint_tensors(MINI, 3)
    expected = musespark.shard_canonical(MINI, load.canonical_from_tensors(MINI, tensors), TP)
    dst = tmp_path / "out"
    quiet = dict(log=lambda _: None, **CONVERT)
    progress = load.convert_presharded(
        src, dst, tp=TP, group=MINI.group_size, layers=[2, 0], **quiet
    )
    assert progress["layers"] == [0, 2] and progress["globals"] and not progress["complete"]
    assert not load.is_complete(dst)
    with pytest.raises(ValueError, match="not complete"):
        load.load_presharded(None, dst)
    for name in ("q", "gate_up_q", "down_s", "attn_gate_alpha"):
        for layer in (0, 2):
            got = np.stack([load.read_rank_array(dst, r, name, layer) for r in range(TP)])
            assert_same(name, got, expected[name][:, layer])
    partial = json.loads((dst / "progress.json").read_text())
    progress = load.convert_presharded(src, dst, tp=TP, group=MINI.group_size, **quiet)
    assert progress["complete"] and progress["layers"] == list(range(MINI.layers))
    assert partial["layers"] == [0, 2]
    arrays = load.read_presharded(dst)
    for name, want in expected.items():
        assert_same(name, arrays[name], want)
    # A finished container is never rewritten, and a different config is refused.
    stamp = load.rank_file(dst, 0, "q").stat().st_mtime_ns
    started = time.perf_counter()
    again = load.convert_presharded(src, dst, tp=TP, group=MINI.group_size, **quiet)
    assert again["complete"] and time.perf_counter() - started < 5
    assert load.rank_file(dst, 0, "q").stat().st_mtime_ns == stamp
    with pytest.raises(ValueError, match="differs"):
        load.convert_presharded(src, dst, tp=4, group=MINI.group_size, **quiet)


def test_verify_layer(converted):
    src, dst, _, _ = converted
    results = load.verify_layer(src, dst, layer=1, experts=(0, 5), log=lambda _: None)
    assert results["q_row0_equals_q_proj_row0"]
    assert all(results[f"rank{r}_q_o_exact"] for r in range(TP))
    # int4 absmax/7 rounding of Gaussian weights: ~11% relative RMS error (step/sqrt(12))
    assert 0 < results["max_rel_rms_error"] < 0.15


def test_tokenizer_adapter(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({"hello": 0, "world": 1, "<|x|>": 2}, unk_token="hello"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(["<|x|>"])
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    adapter = load.load_tokenizer(tmp_path)
    assert adapter.encode("hello <|x|> world") == [0, 2, 1]
    assert adapter.token_to_id("<|x|>") == 2
    assert "hello" in adapter.decode([0, 1])


def _mesh():
    if jax.device_count() < TP:
        pytest.skip("needs XLA_FLAGS=--xla_force_host_platform_device_count=8")
    return jax.sharding.Mesh(np.array(jax.devices()[:TP]), ("tp",))


def test_load_presharded_places_ranks(converted):
    _, dst, _, expected = converted
    mesh = _mesh()
    weights = load.load_presharded(mesh, dst, MINI, layer_chunk_bytes=1 << 14)
    abstract = load.abstract_weights(mesh, MINI, TP)
    assert set(weights) == set(abstract)
    for name, value in weights.items():
        assert value.shape == abstract[name].shape and value.dtype == abstract[name].dtype
        assert value.sharding.is_equivalent_to(abstract[name].sharding, value.ndim)
        shards = {s.device: s for s in value.addressable_shards}
        for rank, device in enumerate(mesh.devices.flat):
            assert shards[device].index[0] == slice(rank, rank + 1)
        host = np.asarray(value)
        want = expected[name]
        if want.dtype == np.dtype(load.INT4):
            assert host.dtype == np.dtype(load.INT4)
            assert np.array_equal(host.astype(np.int8), want.astype(np.int8)), name
        else:
            assert_same(name, host, want)
    caches = load.zero_caches(mesh, MINI, batch=4, context=256)
    for name, (shape, dtype) in layout.kv_cache_shapes(MINI, 4, 256, TP).items():
        assert caches[name].shape == (TP, *shape) and caches[name].dtype == dtype
        assert not np.any(np.asarray(caches[name]))


def test_convert_cli_layers_argument(tmp_path, monkeypatch):
    src = load.write_checkpoint(tmp_path / "ckpt", MINI, load.random_checkpoint_tensors(MINI, 5))
    dst = tmp_path / "out"
    monkeypatch.setattr(load, "READ_THREADS", 2)
    code = load._cli(["convert", "--src", str(src), "--dst", str(dst), "--tp", "2",
                      "--group", str(MINI.group_size), "--layers", "1", "--workers", "2"])
    assert code == 0
    progress = load.read_progress(dst)
    assert progress["layers"] == [1] and not progress["complete"]
    doc = load.read_layout(dst)
    assert doc["tp"] == 2 and tuple(doc["arrays"]["q"]["shape"]) == (MINI.layers, MINI.hidden, 512)
    assert load.config_from_layout(doc) == MINI


# ---------------------------------------------------------------------------------------
# int8 dense families added in place (`quantize-dense`)
# ---------------------------------------------------------------------------------------
def _check_int8_container(dst, expected_bf16, cfg=MINI):
    """Files bit-exact vs `quant.quantize_int8_np` of the bf16 families, layout.json updated."""
    doc = load.read_layout(dst)
    assert load.layout_dense_formats(doc) == ("bf16", "int8")
    assert load.layout_dense_format(doc) == "int8" and load.container_dense_format(dst) == "int8"
    assert doc["int8"] == load.INT8_ENCODING
    both = layout.rank_shapes(cfg, TP, load.layout_expert_format(doc), "both")
    assert list(doc["arrays"]) == list(both)
    assert doc["total_bytes"] == TP * sum(a["nbytes"] for a in doc["arrays"].values())
    for name, (shape, dtype) in both.items():
        spec = doc["arrays"][name]
        assert tuple(spec["shape"]) == shape and load._np_dtype(spec["dtype"]) == np.dtype(dtype)
        for rank in range(TP):
            assert load.rank_file(dst, rank, name).stat().st_size == spec["nbytes"]
    full = load.make_layout(cfg, TP, doc["revision"], load.layout_expert_format(doc),
                            doc.get("expert_source"), dense_formats=("bf16", "int8"))
    assert full["arrays"] == doc["arrays"] and full["total_bytes"] == doc["total_bytes"]
    assert full["dense_formats"] == doc["dense_formats"] and full["dense_format"] == "int8"
    arrays = load.read_presharded(dst)
    for name in layout.INT8_DENSE + ("lm_head",):
        bf16 = expected_bf16[name]
        assert_same(name, arrays[name], bf16)  # the bf16 family is untouched
        if name == "o":  # one scale per column over all ranks' rows
            q, s = quant.quantize_int8_np(np.concatenate(list(bf16), axis=1))  # [L, tp*qw, H]
            rows = bf16.shape[2]
            q = np.stack([q[:, r * rows:(r + 1) * rows] for r in range(TP)])
            s = np.broadcast_to(s, (TP,) + s.shape)
        else:
            q, s = quant.quantize_int8_np(bf16)
        assert_same(name + "_i8", arrays[name + "_i8"], np.ascontiguousarray(q))
        assert_same(name + "_s", arrays[name + "_s"], np.ascontiguousarray(s))
    assert (arrays["o_s"] == arrays["o_s"][:1]).all()
    assert layout.dense_format_of(arrays) == "int8"
    progress = load.read_progress(dst)
    assert progress["dense_int8"]["complete"] and progress["dense_int8"]["lm_head"]
    assert progress["dense_int8"]["layers"] == list(range(cfg.layers))
    return arrays


def test_quantize_dense_v1(converted, tmp_path):
    import shutil

    _, src_dst, _, expected = converted
    dst = tmp_path / "int8"
    shutil.copytree(src_dst, dst)
    quiet = lambda _: None
    # two partial runs (resumable) then the rest
    progress = load.quantize_dense(dst, workers=2, layers=[1, 3], log=quiet)
    assert progress["dense_int8"]["layers"] == [1, 3] and not progress["dense_int8"]["complete"]
    assert load.layout_dense_formats(load.read_layout(dst)) == ("bf16",)  # not yet published
    progress = load.quantize_dense(dst, workers=2, log=quiet, publish=False)
    assert progress["dense_int8"]["complete"]
    assert load.layout_dense_formats(load.read_layout(dst)) == ("bf16",)  # held back
    eff = load.effective_layout(dst)  # ... but an explicit int8 load already works
    assert load.layout_dense_formats(eff) == ("bf16", "int8") and load.layout_dense_format(eff) == "bf16"
    assert set(load.weight_array_names(eff, "int8")) == set(layout.rank_shapes(MINI, TP, dense_format="int8"))
    load.quantize_dense(dst, workers=1, log=quiet)  # nothing to do: publishes
    assert load.effective_layout(dst) == load.read_layout(dst)
    arrays = _check_int8_container(dst, expected)
    # idempotent
    stamp = load.rank_file(dst, 0, "q_i8").stat().st_mtime_ns
    load.quantize_dense(dst, workers=2, log=quiet)
    assert load.rank_file(dst, 0, "q_i8").stat().st_mtime_ns == stamp
    # the CLI entry point
    assert load._cli(["quantize-dense", "--dir", str(dst), "--workers", "1"]) == 0
    # loading: the preferred (int8) tree, or the bf16 one on request
    mesh = _mesh()
    doc = load.read_layout(dst)
    assert set(load.weight_array_names(doc)) == set(layout.rank_shapes(MINI, TP, dense_format="int8"))
    assert set(load.weight_array_names(doc, "bf16")) == set(layout.rank_shapes(MINI, TP))
    with pytest.raises(ValueError):
        load.weight_array_names(load.read_layout(src_dst), "int8")
    weights = load.load_presharded(mesh, dst, MINI, layer_chunk_bytes=1 << 14)
    abstract = load.abstract_weights(mesh, MINI, TP, dense_format="int8")
    assert set(weights) == set(abstract) and layout.dense_format_of(weights) == "int8"
    for name, value in weights.items():
        assert value.shape == abstract[name].shape and value.dtype == abstract[name].dtype
        if name.endswith(("_i8", "_s")):
            assert_same(name, np.asarray(value), arrays[name])
    bf16 = load.load_presharded(mesh, dst, MINI, dense_format="bf16", layer_chunk_bytes=1 << 14)
    assert set(bf16) == set(load.abstract_weights(mesh, MINI, TP))
    assert layout.dense_format_of(bf16) == "bf16"
    assert_same("q", np.asarray(bf16["q"]), expected["q"])
    with pytest.raises(ValueError):
        load.load_presharded(mesh, dst, MINI, dense_format="fp8")


def test_quantize_dense_v2(tmp_path):
    src, dst, tensors = load.synthetic_presharded_nvfp4(tmp_path, MINI, TP, seed=9, workers=2)
    doc = load.read_layout(dst)
    assert load.layout_expert_format(doc) == "nvfp4" and load.layout_dense_format(doc) == "bf16"
    expected = load.read_presharded(dst)
    progress = load.quantize_dense(dst, workers=3, log=lambda _: None)
    assert progress["dense_int8"]["complete"] and progress["complete"]
    _check_int8_container(dst, expected)
    assert load.layout_expert_format(load.read_layout(dst)) == "nvfp4"
