"""CPU tests for the Muse Spark per-rank layout: shapes, byte totals, tile schedule."""

import jax.numpy as jnp
import numpy as np
import pytest

from musespark import MINI, Config, layout

MIB, GIB = 2**20, 2**30


def test_real_config_shapes_match_design_table():
    c = Config()
    s = layout.rank_shapes(c, tp=8)
    L, H, E = 62, 8192, 256
    want = {
        "embed": ((25600, H), jnp.bfloat16),
        "lm_head": ((H, 25600), jnp.bfloat16),
        "final_norm": ((1, H), jnp.bfloat16),
        "attn_norm": ((L, 1, H), jnp.bfloat16),
        "attn_gate_alpha": ((L, 1, H), jnp.float32),
        "pre_expert_norm": ((L, 1, 4096), jnp.bfloat16),
        "router_bias": ((L, 1, E), jnp.float32),
        "q": ((L, H, 1024), jnp.bfloat16),
        "kv": ((L, H, 256), jnp.bfloat16),
        "gate": ((L, H, 1024), jnp.bfloat16),
        "o": ((L, 1024, H), jnp.bfloat16),
        "pre": ((L, H, 512), jnp.bfloat16),
        "router_hi": ((L, H, E), jnp.bfloat16),
        "router_lo": ((L, H, E), jnp.bfloat16),
        "post": ((L, 4096, 1024), jnp.bfloat16),
        "gate_up_q": ((L, E, 4096, 1024), jnp.int4),
        "gate_up_s": ((L, E, 8, 4, 1024), jnp.float32),
        "down_q": ((L, E, 512, 4096), jnp.int4),
        "down_s": ((L, E, 1, 4, 4096), jnp.float32),
    }
    for name, (shape, dtype) in want.items():
        assert s[name] == (shape, dtype), name
    assert len(s) == 25
    assert layout.vocab_pad(c, 8) == 25600 and 8 * 25600 >= c.vocab
    assert layout.cache_lanes(c, 8) == 128


def test_real_config_bytes_match_spec_totals():
    c = Config()
    b = layout.bytes_per_rank(c, tp=8)
    assert b["experts"] == 62 * 256 * (4096 * 1024 + 512 * 4096) // 2  # 46.5 GiB
    assert b["experts"] == 46.5 * GIB
    assert b["expert_scales"] == 62 * 256 * (8 * 4 * 1024 + 1 * 4 * 4096) * 4
    assert b["dense"] == 62 * (16 + 4 + 16 + 16 + 8 + 4 + 4 + 8) * MIB  # 4.6 GiB
    assert b["vocab"] == 2 * 25600 * 8192 * 2
    assert b["total"] == sum(b[n] for n in layout.rank_shapes(c))
    assert 53 * GIB < b["total"] < 56 * GIB


def test_mini_config_shapes():
    s = layout.rank_shapes(MINI, tp=8)
    assert s["q"] == ((4, 1024, 128), jnp.bfloat16)  # 2 q heads x 64
    assert s["kv"] == ((4, 1024, 128), jnp.bfloat16)  # 1 kv head: k lanes 0:64, v lanes 64:128
    assert s["o"] == ((4, 128, 1024), jnp.bfloat16)
    assert s["pre"] == ((4, 1024, 64), jnp.bfloat16)
    assert s["post"] == ((4, 512, 128), jnp.bfloat16)
    assert s["router_hi"] == ((4, 1024, 16), jnp.bfloat16)
    assert s["gate_up_q"] == ((4, 16, 512, 128), jnp.int4)
    assert s["gate_up_s"] == ((4, 16, 1, 8, 128), jnp.float32)
    assert s["down_q"] == ((4, 16, 64, 512), jnp.int4)
    assert s["down_s"] == ((4, 16, 1, 1, 512), jnp.float32)
    assert s["embed"] == ((1024, 1024), jnp.bfloat16)
    assert layout.cache_lanes(MINI, 8) == 128
    assert layout.kv_cache_shapes(MINI, 4, 512, 8) == {
        "k_cache": ((4, 4, 512, 128), jnp.bfloat16),
        "v_cache": ((4, 4, 512, 128), jnp.bfloat16),
    }


def test_invalid_tp_and_context_are_rejected():
    with pytest.raises(ValueError):
        layout.rank_shapes(MINI, tp=16)  # 8 kv heads over 16 ranks
    with pytest.raises(ValueError):
        layout.check_tp(MINI.__class__(**{**MINI.__dict__, "group_size": 128}), 8)  # 64 % 128
    with pytest.raises(ValueError):
        layout.kv_cache_shapes(MINI, 1, 100)


def test_tile_schedule_real_config():
    c = Config()
    sch = layout.tile_schedule(c, tp=8)
    geo = sch.geometry
    assert (geo.bank_k, geo.bank_n, geo.banks) == (1024, 1024, 12)
    assert geo.bank_bytes == 2 * MIB
    families = [t.family for t in sch.layer]
    # q(8 K-tiles) kv(8) gate(8) o(8 N-tiles) pre(8) router_hi(8) router_lo(8) post(4 K-tiles)
    counts = {f: families.count(f) for f in layout.STREAMED_FAMILIES}
    assert counts == {
        "q": 8,
        "kv": 8,
        "gate": 8,
        "o": 8,
        "pre": 8,
        "router_hi": 8,
        "router_lo": 8,
        "post": 4,
    }
    assert families == sorted(families, key=layout.STREAMED_FAMILIES.index)
    assert sch.tiles_per_layer == 60
    assert sch.layer[0] == ("q", 0, 0, 1024, 1024)
    assert sch.layer[8] == ("kv", 0, 0, 1024, 256)
    assert [t.n0 for t in sch.layer if t.family == "o"] == list(range(0, 8192, 1024))
    assert all(t.k0 == 0 for t in sch.layer if t.family == "o")
    assert len(sch.lm_head) == 8 * 25
    assert sch.lm_head[0] == ("lm_head", 0, 0, 1024, 1024)
    assert sch.lm_head[8] == ("lm_head", 0, 1024, 1024, 1024)  # N-tiles outer, K-tiles inner
    assert sch.total(c.layers) == 62 * 60 + 200
    for t in sch.layer + sch.lm_head:
        assert t.bk * t.bn <= geo.bank_k * geo.bank_n
    assert sch.tile(61, c.layers) == (1, sch.layer[1])
    assert sch.tile(62 * 60 + 3, c.layers) == (62, sch.lm_head[3])


def test_tile_schedule_mini():
    sch = layout.tile_schedule(MINI, tp=8)
    assert sch.tiles_per_layer == 8  # every MINI matrix is a single tile
    assert sch.layer[1] == ("kv", 0, 0, 1024, 128)
    assert sch.layer[3] == ("o", 0, 0, 128, 1024)
    assert sch.layer[7] == ("post", 0, 0, 512, 128)
    assert len(sch.lm_head) == 1


def test_nbytes_int4_is_half_a_byte():
    assert layout.nbytes((4, 8), jnp.int4) == 16
    assert layout.nbytes((4, 8), jnp.bfloat16) == 64
    assert layout.nbytes((4, 8), np.float32) == 128
