"""End-to-end test of the decode megakernel on the NVFP4 (container format v2) expert path.

A synthetic v2 MINI container (`load.synthetic_presharded_nvfp4`, as in
`tests/test_musespark_fp4.py`) is loaded with `load.load_presharded`; `make_decode` infers the
expert format from the weight tree (`layout.expert_format_of`) and dispatches to `musespark.fp4`.
The kernel is compared with the pure-JAX reference on the *exactly dequantized* bf16 experts
(`_dense_canonical`), with the checks of `tests/test_musespark_decode.py` (softcapped logits,
greedy tokens, per-layer residual streams, written KV-cache slots) over several decode steps.

CPU (8 host devices, `JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8`):
interpret mode, B in {1, 4}, 6 steps. TPU (`tpu_run.sh all python -m pytest
tests/test_musespark_decode_fp4.py -s`): B in {1, 2, 4, 8}, 16 steps.
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

if "xla_allow_excess_precision" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_allow_excess_precision=false"
    ).strip()

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import musespark
from musespark import decode_megakernel as dk
from musespark import layout, load
from musespark.config import MINI, Config

TESTS = Path(__file__).resolve().parent


def _module(name):
    spec = importlib.util.spec_from_file_location(name, TESTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


D = _module("test_musespark_decode")  # step helpers / comparisons
F = _module("test_musespark_fp4")  # _dense_canonical

TP = 8
CONTEXT = 512
ON_TPU = jax.default_backend() == "tpu"


@pytest.fixture(scope="module")
def mesh():
    devices = sorted(jax.devices(), key=lambda d: d.id)
    if len(devices) < TP:
        pytest.skip("needs 8 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)")
    return Mesh(np.array(devices[:TP]), ("tp",))


@pytest.fixture(scope="module")
def fp4_case(mesh, tmp_path_factory):
    """(canonical device weights with dense dequantized experts, sharded v2 container weights)
    with the damped residual gates and decisive routing biases of the int4 tests."""
    cfg = MINI
    tmp = tmp_path_factory.mktemp("musespark-nvfp4-decode")
    _, dst, tensors = load.synthetic_presharded_nvfp4(
        tmp, cfg, TP, seed=3, workers=4, experts_per_task=4
    )
    assert load.container_expert_format(dst) == "nvfp4"
    weights = load.load_presharded(mesh, dst, cfg, layer_chunk_bytes=1 << 12)
    assert layout.expert_format_of(weights) == "nvfp4"
    canonical = F._dense_canonical(cfg, tensors, load.read_presharded(dst))
    rng = np.random.default_rng(0)
    for name in ("attn_gate", "ffn_gate"):
        g = (-0.3 + 0.05 * rng.standard_normal((cfg.layers, cfg.hidden))).astype(np.float32)
        alpha, beta = musespark.gate_coeffs(jnp.asarray(g), cfg.gate_temperature)
        canonical[name + "_alpha"], canonical[name + "_beta"] = np.asarray(alpha), np.asarray(beta)
    bias = np.asarray(canonical["router_bias"], np.float32).copy()
    for layer in range(cfg.layers):
        chosen = (layer * 5 + np.arange(cfg.top_k) * (cfg.experts // cfg.top_k)) % cfg.experts
        bias[layer, chosen] += 2.0
    canonical["router_bias"] = bias
    sharding = NamedSharding(mesh, P("tp"))
    for name in ("attn_gate_alpha", "attn_gate_beta", "ffn_gate_alpha", "ffn_gate_beta",
                 "router_bias"):
        width = cfg.hidden if "gate" in name else cfg.experts
        value = np.broadcast_to(
            np.asarray(canonical[name])[None, :, None], (TP, cfg.layers, 1, width)
        )
        weights[name] = jax.device_put(jnp.asarray(np.ascontiguousarray(value)), sharding)
    return musespark.to_device(canonical), weights


def _run(mesh, fp4_case, batch, steps, interpret, name):
    cfg = MINI
    canonical_dev, weights = fp4_case
    rng = np.random.default_rng(batch * 11 + steps)
    tokens = rng.integers(0, cfg.vocab_used, (steps, batch)).astype(np.int32)
    positions = np.broadcast_to(np.arange(steps, dtype=np.int32)[:, None], (steps, batch)).copy()
    ref, ref_caches = D._reference_steps(
        cfg, canonical_dev, tokens, positions, musespark.init_caches(cfg, batch, CONTEXT)
    )
    options = {"aux_hidden"} | ({"interpret"} if interpret else set())
    decode = dk.make_decode(
        mesh, cfg, CONTEXT, batch, greedy=True, return_logits=True, options=frozenset(options)
    )
    started = time.perf_counter()
    got, caches = D._kernel_steps(
        decode, weights, load.zero_caches(mesh, cfg, batch, CONTEXT), tokens, positions
    )
    print(f"[{name}] {steps} kernel steps (incl. compile): {time.perf_counter() - started:.1f} s")
    D._compare_steps(name, cfg, ref, got, steps)
    D._compare_caches(name, cfg, caches, ref_caches, batch, steps)


@pytest.mark.skipif(ON_TPU, reason="interpret-mode test; see test_mini_decode_nvfp4_tpu")
@pytest.mark.parametrize("batch", [1, 4])
def test_mini_decode_nvfp4_interpret(mesh, fp4_case, batch):
    _run(mesh, fp4_case, batch, steps=6, interpret=True, name=f"nvfp4 interpret B={batch}")


@pytest.mark.skipif(not ON_TPU, reason="needs 8 TPU cores")
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_mini_decode_nvfp4_tpu(mesh, fp4_case, batch):
    _run(mesh, fp4_case, batch, steps=16, interpret=False, name=f"nvfp4 tpu B={batch}")


def test_vmem_budget_real_config_nvfp4():
    cfg = Config()
    for batch in (1, 8):
        slots, banks = dk.default_geometry(cfg, batch, expert_format="nvfp4")
        assert (slots, banks) == (4, 12)
        budget = dk.vmem_budget(cfg, batch, moe_slots=slots, banks=banks, expert_format="nvfp4")
        assert budget["total"] <= dk.VMEM_EXPLICIT_LIMIT, budget
    assert len(dk.weight_names("nvfp4")) == len(dk.WEIGHT_NAMES) + 1
    with pytest.raises(ValueError):
        dk.weight_names("fp8")
