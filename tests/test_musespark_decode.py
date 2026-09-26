"""End-to-end tests of the Muse Spark decode megakernel (`musespark.decode_megakernel`).

CPU (8 host devices, `JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8`):
the whole kernel runs under `pltpu.InterpretParams` on the MINI config against the pure-JAX
reference (`musespark.decode_step` / `musespark.forward`): 6 decode steps from position 0,
max |softcapped logit diff| <= 5e-2, greedy-token agreement every step (a disagreement passes
only as a near-tie, i.e. the kernel's token is within the tolerance of the reference maximum;
the achieved agreement is printed), per-layer residual streams (`aux_hidden` option) and the
written KV-cache slots.

TPU (`tpu_run.sh all python -m pytest tests/test_musespark_decode.py -s`): the same MINI
checks on hardware for B in {1, 2, 4, 8} over 16 steps, prefill (T=37, `musespark.prefill`)
followed by kernel decode against the sequential reference, and a real-width smoke test
(`Config(layers=4)`: hidden 8192, 256 experts, per-rank random weights generated on device)
reporting the VMEM budget and the step / per-layer timings for B in {1, 8}.

Weights: `random_canonical_weights` with the damped residual gates and decisive routing biases
of `tests/test_musespark_prefill.py` (the plain random MINI net is chaotic: bf16 summation-order
noise alone grows to O(1) after four layers).

The TPU runs set `XLA_FLAGS=--xla_allow_excess_precision=false` (module import, before jax
initialises) so the XLA glue's `lax.reduce_precision` roundings match the reference exactly.
"""

import os
import time

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
import musespark.prefill as prefill_mod
from musespark import layout
from musespark import decode_megakernel as dk
from musespark.config import MINI, Config
from musespark.load import zero_caches

TP = 8
CONTEXT = 512
ON_TPU = jax.default_backend() == "tpu"
LOGIT_TOL = 5e-2  # design.md section 7: max |softcapped logit diff|
RESIDUAL_TOL = 0.1  # f32 residual stream (unit RMS): bf16 summation-order noise
CACHE_TOL = 0.1  # bf16 K/V values (up to ~4): one or two ulps
T_PREFILL = 37


@pytest.fixture(scope="module")
def mesh():
    devices = sorted(jax.devices(), key=lambda d: d.id)
    if len(devices) < TP:
        pytest.skip("needs 8 devices (XLA_FLAGS=--xla_force_host_platform_device_count=8)")
    return Mesh(np.array(devices[:TP]), ("tp",))


def make_weights(cfg, key=1, decisive_routing=True):
    """Random canonical weights with damped residual gates and decisive routing biases
    (copied from tests/test_musespark_prefill.py::_make_weights)."""
    canonical = musespark.random_canonical_weights(cfg, key=key, quantized=True)
    rng = np.random.default_rng(key + 100)
    for name in ("attn_gate", "ffn_gate"):
        g = (-0.3 + 0.05 * rng.standard_normal((cfg.layers, cfg.hidden))).astype(np.float32)
        alpha, beta = musespark.gate_coeffs(jnp.asarray(g), cfg.gate_temperature)
        canonical[name + "_alpha"], canonical[name + "_beta"] = np.asarray(alpha), np.asarray(beta)
    if decisive_routing:
        bias = np.asarray(canonical["router_bias"], np.float32).copy()
        for layer in range(cfg.layers):
            chosen = (layer * 5 + np.arange(cfg.top_k) * (cfg.experts // cfg.top_k)) % cfg.experts
            bias[layer, chosen] += 2.0  # sigmoid scores are in (0, 1): always selected
        canonical["router_bias"] = bias
    return canonical


@pytest.fixture(scope="module")
def mini_case(mesh):
    """(canonical numpy weights, canonical device weights, sharded device weights)."""
    cfg = MINI
    canonical = make_weights(cfg)
    sharded = musespark.shard_canonical(cfg, canonical, TP)
    sharding = NamedSharding(mesh, P("tp"))
    weights = {
        n: jax.device_put(a, sharding) for n, a in musespark.to_device(sharded).items()
    }
    return canonical, musespark.to_device(canonical), weights


@pytest.fixture(scope="module")
def mini_case_int8(mesh):
    """`mini_case` with int8 per-output-channel dense projections + lm_head
    (`quantize_dense_canonical`): the reference runs `dot(x, q) * s`, the kernel streams the
    `_i8` / `_s` families (dense_format inferred from the weight tree)."""
    cfg = MINI
    canonical = musespark.quantize_dense_canonical(cfg, make_weights(cfg))
    sharded = musespark.shard_canonical(cfg, canonical, TP)
    assert layout.dense_format_of(sharded) == "int8"
    # the round trip through the per-rank layout is exact
    back = musespark.unshard(cfg, sharded, TP)
    for name in ("q_i8", "q_s", "k_i8", "v_s", "o_i8", "o_s", "pre_s", "post_i8", "lm_head_i8",
                 "lm_head_s"):
        assert np.array_equal(back[name], canonical[name]), name
    sharding = NamedSharding(mesh, P("tp"))
    weights = {
        n: jax.device_put(a, sharding) for n, a in musespark.to_device(sharded).items()
    }
    return canonical, musespark.to_device(canonical), weights


def _reference_steps(cfg, canonical_dev, tokens, positions, caches):
    """Sequential reference decode: lists of (logits [B, V], hidden [L+1, B, H]) per step."""
    run = jax.jit(
        lambda w, t, p, c: musespark.forward(cfg, w, t[:, None], p, c, return_hidden=True)
    )
    out = []
    for step in range(tokens.shape[0]):
        logits, caches, hidden = run(
            canonical_dev, jnp.asarray(tokens[step]), jnp.asarray(positions[step]), caches
        )
        out.append((np.asarray(logits[:, 0]), np.asarray(hidden[:, :, 0])))
    return out, caches


def _kernel_steps(decode, weights, caches, tokens, positions, aux=True):
    out = []
    for step in range(tokens.shape[0]):
        result = decode(weights, caches, jnp.asarray(tokens[step]), jnp.asarray(positions[step]))
        next_tokens, logits, caches = result[:3]
        hidden = np.asarray(result[3]) if aux else None
        out.append((np.asarray(next_tokens), np.asarray(logits), hidden))
    return out, caches


def _compare_steps(name, cfg, ref, got, steps):
    """Per step: max |softcapped logit diff| <= LOGIT_TOL, the kernel's greedy token equals the
    reference argmax (a disagreement is tolerated only on a near-tie: the kernel's token must
    lie within LOGIT_TOL of the reference maximum), residual streams within RESIDUAL_TOL."""
    worst_logit, worst_hidden, agreed, rows = 0.0, np.zeros(cfg.layers + 1), 0, 0
    for step in range(steps):
        ref_logits, ref_hidden = ref[step]
        tokens, logits, hidden = got[step]
        assert logits.shape == ref_logits.shape
        assert np.all(np.isfinite(logits[:, : cfg.vocab_used]))
        diff = np.abs(logits - ref_logits)[:, : cfg.vocab_used].max()
        worst_logit = max(worst_logit, diff)
        ref_top = np.argmax(ref_logits, axis=-1)
        assert np.array_equal(tokens, np.argmax(logits, axis=-1)), (step, tokens)
        agree = tokens == ref_top
        agreed += int(agree.sum())
        rows += len(agree)
        ref_max = ref_logits.max(axis=-1)
        chosen = np.take_along_axis(ref_logits, tokens[:, None], axis=-1)[:, 0]
        line = f"[{name}] step {step}: max |logit diff| {diff:.4f}, top-1 agree {agree.all()}"
        if not agree.all():
            line += f" (near-tie: reference margin {(ref_max - chosen)[~agree].max():.4f})"
        if hidden is not None:
            per_layer = np.abs(hidden - ref_hidden).reshape(cfg.layers + 1, -1).max(axis=1)
            worst_hidden = np.maximum(worst_hidden, per_layer)
            line += f", residual max |diff| per layer {np.round(per_layer, 3)}"
            assert per_layer[0] == 0, "embedding + embed norm must be bit-exact"
        print(line)
        assert np.all(chosen >= ref_max - LOGIT_TOL), (step, tokens, ref_top)
        assert diff <= LOGIT_TOL, (step, diff)
    if hidden is not None:
        assert np.all(worst_hidden <= RESIDUAL_TOL), worst_hidden
    print(
        f"[{name}] worst |logit diff| {worst_logit:.4f}, worst residual "
        f"{worst_hidden.max():.4f}, top-1 agreement {agreed}/{rows}"
    )
    return worst_logit


def _compare_caches(name, cfg, kernel_caches, ref_caches, batch, steps):
    got = musespark.unshard_caches(cfg, {k: np.asarray(v) for k, v in kernel_caches.items()}, TP)
    lanes = kernel_caches["k_cache"].shape[-1]
    kvw = (cfg.kv_heads // TP) * cfg.head_dim
    for kind, mine, ref in zip("kv", got, ref_caches):
        mine, ref = np.asarray(mine), np.asarray(ref)
        assert mine.shape == ref.shape == (cfg.layers, batch, CONTEXT, cfg.kv_heads, cfg.head_dim)
        mine_w, ref_w = mine[:, :, :steps].astype(np.float32), ref[:, :, :steps].astype(np.float32)
        written = np.abs(mine_w - ref_w)
        per_layer = written.reshape(cfg.layers, -1).max(axis=1)
        exact = (mine[:, :, :steps] == ref[:, :, :steps]).reshape(cfg.layers, -1).mean(axis=1)
        print(
            f"[{name}] {kind}-cache written slots: max |diff| per layer "
            f"{np.round(per_layer, 3)}, bf16-exact fraction per layer {np.round(exact, 3)}"
        )
        # bf16 K/V of the written slots agree to summation-order noise (the projections are
        # r16'd dots whose K-sum order differs between the ring gemv and the reference).
        assert np.all(per_layer <= CACHE_TOL)
        assert not np.any(mine[:, :, steps:])  # untouched slots stay zero
    if lanes > kvw:
        assert not np.any(np.asarray(kernel_caches["k_cache"])[..., kvw:])


def _run_mini(mesh, mini_case, batch, steps, interpret, name):
    cfg = MINI
    canonical, canonical_dev, weights = mini_case
    rng = np.random.default_rng(batch * 7 + steps)
    tokens = rng.integers(0, cfg.vocab_used, (steps, batch)).astype(np.int32)
    positions = np.broadcast_to(np.arange(steps, dtype=np.int32)[:, None], (steps, batch)).copy()
    ref, ref_caches = _reference_steps(
        cfg, canonical_dev, tokens, positions, musespark.init_caches(cfg, batch, CONTEXT)
    )
    options = {"aux_hidden"} | ({"interpret"} if interpret else set())
    decode = dk.make_decode(
        mesh, cfg, CONTEXT, batch, greedy=True, return_logits=True, options=frozenset(options)
    )
    started = time.perf_counter()
    got, caches = _kernel_steps(decode, weights, zero_caches(mesh, cfg, batch, CONTEXT), tokens,
                                positions)
    print(f"[{name}] {steps} kernel steps (incl. compile): {time.perf_counter() - started:.1f} s")
    _compare_steps(name, cfg, ref, got, steps)
    _compare_caches(name, cfg, caches, ref_caches, batch, steps)


# ---------------------------------------------------------------------------------------
# CPU interpret
# ---------------------------------------------------------------------------------------
@pytest.mark.skipif(ON_TPU, reason="interpret-mode test; see test_mini_decode_tpu")
@pytest.mark.parametrize("batch", [1, 4])
def test_mini_decode_interpret(mesh, mini_case, batch):
    _run_mini(mesh, mini_case, batch, steps=6, interpret=True, name=f"interpret B={batch}")


@pytest.mark.skipif(ON_TPU, reason="interpret-mode test; see test_mini_decode_int8_tpu")
def test_mini_decode_int8_interpret(mesh, mini_case_int8):
    _run_mini(mesh, mini_case_int8, 2, steps=6, interpret=True, name="interpret int8 B=2")


def test_vmem_budget_real_config():
    cfg = Config()
    # packed int4 slots (3.4 MiB): the 4 default slots fit at every batch with the full ring
    assert dk.default_geometry(cfg, 1) == (4, 12)
    assert dk.default_geometry(cfg, 8) == (4, 12)
    for batch in (1, 8):
        slots, banks = dk.default_geometry(cfg, batch)
        budget = dk.vmem_budget(cfg, batch, moe_slots=slots, banks=banks)
        assert budget["total"] <= dk.VMEM_EXPLICIT_LIMIT, budget
    full = dk.vmem_budget(cfg, 8, moe_slots=8)
    assert full["total"] > dk.VMEM_EXPLICIT_LIMIT
    slot = full["moe"] - dk.vmem_budget(cfg, 8, moe_slots=7)["moe"]
    assert slot == pytest.approx(3.375 * dk.MIB, rel=0.02)  # 2 + 1 MiB packed + 0.375 scales
    assert dk.default_geometry(MINI, 8) == (4, 12)
    assert dk.vmem_budget(MINI, 8)["total"] <= dk.VMEM_EXPLICIT_LIMIT
    # the int8 ring is the same 24 MiB; scale vectors + lm_head scales add < 0.5 MiB
    i8 = dk.vmem_budget(cfg, 1, dense_format="int8", expert_format="nvfp4", log=None)
    b16 = dk.vmem_budget(cfg, 1, dense_format="bf16", expert_format="nvfp4", log=None)
    assert i8["ring"] == b16["ring"] and 0 < i8["total"] - b16["total"] < dk.MIB // 2
    assert i8["total"] <= dk.VMEM_EXPLICIT_LIMIT


def test_parse_options():
    opts = dk.parse_options(
        frozenset({"interpret", "aux_hidden", "moe_slots=2", "wire=f32", "banks=10", "defer=none"})
    )
    assert opts.interpret and opts.aux_hidden and opts.moe_slots == 2
    assert opts.wire == jnp.float32 and opts.banks == 10 and opts.defer == "none"
    assert dk.parse_options(frozenset()).wire == jnp.bfloat16
    with pytest.raises(ValueError):
        dk.parse_options(frozenset({"bogus"}))


# ---------------------------------------------------------------------------------------
# TPU: MINI on hardware, prefill + decode, real-width smoke
# ---------------------------------------------------------------------------------------
@pytest.mark.skipif(not ON_TPU, reason="needs 8 TPU cores")
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_mini_decode_tpu(mesh, mini_case, batch):
    _run_mini(mesh, mini_case, batch, steps=16, interpret=False, name=f"tpu B={batch}")


@pytest.mark.skipif(not ON_TPU, reason="needs 8 TPU cores")
@pytest.mark.parametrize("batch", [1, 8])
def test_mini_decode_int8_tpu(mesh, mini_case_int8, batch):
    _run_mini(mesh, mini_case_int8, batch, steps=16, interpret=False, name=f"tpu int8 B={batch}")


@pytest.mark.skipif(not ON_TPU, reason="needs 8 TPU cores")
def test_prefill_then_kernel_decode_int8_tpu(mesh, mini_case_int8):
    """XLA prefill (`prefill._dense`: dot(x, q) * s) + kernel decode on the int8 MINI case."""
    _prefill_then_decode(mesh, mini_case_int8, "prefill+decode int8")


@pytest.mark.skipif(not ON_TPU, reason="needs 8 TPU cores")
def test_prefill_then_kernel_decode_tpu(mesh, mini_case):
    _prefill_then_decode(mesh, mini_case, "prefill+decode")


def _prefill_then_decode(mesh, mini_case, name):
    cfg = MINI
    canonical, canonical_dev, weights = mini_case
    steps, batch = 8, 2
    rng = np.random.default_rng(11)
    prompt = rng.integers(0, cfg.vocab_used, T_PREFILL).astype(np.int32)
    extra = rng.integers(0, cfg.vocab_used, (steps, batch)).astype(np.int32)
    # reference: the prompt in every row, then `steps` sequential decode steps
    prompts = np.broadcast_to(prompt, (batch, T_PREFILL))
    ref_run = jax.jit(lambda w, t, c: musespark.prefill_reference(cfg, w, t, c))
    _, ref_caches = ref_run(
        canonical_dev, jnp.asarray(prompts), musespark.init_caches(cfg, batch, CONTEXT)
    )
    positions = T_PREFILL + np.arange(steps, dtype=np.int32)[:, None]
    positions = np.broadcast_to(positions, (steps, batch)).copy()
    ref, ref_caches = _reference_steps(cfg, canonical_dev, extra, positions, ref_caches)
    # kernel: XLA prefill into every cache row, then the megakernel
    prefill = prefill_mod.make_prefill(mesh, cfg, CONTEXT, TP)
    caches = zero_caches(mesh, cfg, batch, CONTEXT)
    padded, length = prefill_mod.pad_prompt(cfg, prompt)
    for row in range(batch):
        _, caches = prefill(weights, caches, padded, length, row)
    decode = dk.make_decode(
        mesh, cfg, CONTEXT, batch, return_logits=True, options=frozenset({"aux_hidden"})
    )
    got, caches = _kernel_steps(decode, weights, caches, extra, positions)
    _compare_steps(name, cfg, ref, got, steps)
    _compare_caches(name, cfg, caches, ref_caches, batch, T_PREFILL + steps)


def random_rank_weights(mesh, cfg, tp=TP, seed=0):
    """Random per-rank weights in `layout.rank_shapes` layout generated on the devices
    (`{name: [tp, ...]}` sharded `P("tp")`): unit-RMS bf16 matrices scaled by 1/sqrt(fan_in),
    norm gains near 1, damped residual gates, int4 experts with bf16-valued group scales."""
    sharding = NamedSharding(mesh, P("tp"))
    shapes = layout.rank_shapes(cfg, tp)
    out = {}
    for i, (name, (shape, dtype)) in enumerate(shapes.items()):
        full = (tp, *shape)

        def make(name=name, shape=full, dtype=dtype, i=i):
            key = jax.random.fold_in(jax.random.PRNGKey(seed), i)
            if dtype == jnp.int4:
                return jax.random.randint(key, shape, -8, 8, jnp.int8).astype(jnp.int4)
            if name in ("gate_up_s", "down_s"):
                return (jax.random.uniform(key, shape, jnp.float32, 0.5, 1.5) * 0.004).astype(
                    jnp.bfloat16).astype(jnp.float32)
            if name.endswith("_norm"):
                return (1.0 + 0.05 * jax.random.normal(key, shape, jnp.float32)).astype(dtype)
            if name.endswith("_alpha") or name.endswith("_beta"):
                g = -0.3 + 0.05 * jax.random.normal(key, shape, jnp.float32)
                alpha, beta = musespark.gate_coeffs(g, cfg.gate_temperature)
                return alpha if name.endswith("_alpha") else beta
            if name == "router_bias":
                return 0.1 * jax.random.normal(key, shape, jnp.float32)
            if name == "router_lo":
                return jnp.zeros(shape, dtype)
            fan_in = shape[-2] if name != "embed" else 1
            x = jax.random.normal(key, shape, jnp.float32) / np.sqrt(fan_in)
            return x.astype(dtype)

        out[name] = jax.jit(make, out_shardings=sharding)()
    return out


STEPS_PER_CALL = 8  # decode steps per device call when timing (amortises the ~0.5 ms dispatch)


def _make_chunk(decode, cfg):
    """`demo_musespark.make_chunk` (the production call site) when importable, else a local
    equivalent: `steps` greedy decode steps per device call inside a `lax.fori_loop`."""
    try:
        from demo_musespark import make_chunk

        return make_chunk(decode, STEPS_PER_CALL, cfg, greedy=True, top_k=None)
    except ImportError:  # the demo pulls in the server module; fall back to the same loop

        def fn(weights, caches, tokens, pos, key, temperature, top_p):
            def step(i, carry):
                cur, caches, out = carry
                nxt, _, caches = decode(weights, caches, cur, pos + i)
                return nxt, caches, out.at[i].set(nxt)

            out = jnp.zeros((STEPS_PER_CALL, tokens.shape[0]), jnp.int32)
            return jax.lax.fori_loop(0, STEPS_PER_CALL, step, (tokens, caches, out))

        return jax.jit(fn, donate_argnums=(1,))


def _time_steps(decode, cfg, weights, caches, tokens, pos, iters, rounds=3):
    """Best-of-`rounds` seconds per decode step, `iters` chunks of STEPS_PER_CALL steps each."""
    chunk = _make_chunk(decode, cfg)
    key = jax.random.key(0)
    args = (tokens, pos, key, jnp.float32(1.0), jnp.float32(1.0))
    best = float("inf")
    for _ in range(rounds):
        cur, caches, *_ = chunk(weights, caches, *args)
        jax.block_until_ready(cur)
        t0 = time.perf_counter()
        for _ in range(iters):
            cur, caches, *_ = chunk(weights, caches, *args)
        jax.block_until_ready(cur)
        best = min(best, (time.perf_counter() - t0) / (iters * STEPS_PER_CALL))
    return best, caches


@pytest.mark.skipif(not ON_TPU, reason="needs 8 TPU cores")
def test_real_widths_smoke_tpu(mesh):
    """Real per-rank widths (hidden 8192, 256 experts top-8, Hm 4096, V 202048) with 4 and 8
    layers: compiles within the VMEM limit, runs without NaNs, reports us per step / layer."""
    context = 4096
    results = {}
    for layers in (4, 8):
        cfg = Config(layers=layers)
        weights = random_rank_weights(mesh, cfg)
        for batch in (1, 8):
            slots, banks = dk.default_geometry(cfg, batch)
            budget = dk.vmem_budget(cfg, batch, moe_slots=slots, banks=banks)
            assert budget["total"] <= 58 * dk.MIB
            decode = dk.make_decode(mesh, cfg, context, batch, return_logits=True)
            caches = zero_caches(mesh, cfg, batch, context)
            rng = np.random.default_rng(batch)
            tokens = jnp.asarray(rng.integers(0, cfg.vocab_used, batch).astype(np.int32))
            pos = jnp.asarray(rng.integers(0, 3000, batch).astype(np.int32))
            started = time.perf_counter()
            next_tokens, logits, caches = decode(weights, caches, tokens, pos)
            jax.block_until_ready(next_tokens)
            compile_s = time.perf_counter() - started
            logits = np.asarray(logits)
            assert np.all(np.isfinite(logits[:, : cfg.vocab_used]))
            assert np.all(np.abs(logits[:, : cfg.vocab_used]) <= cfg.softcap)
            assert np.all(np.asarray(next_tokens) < cfg.vocab_used)
            step_s, caches = _time_steps(decode, cfg, weights, caches, tokens, pos, iters=5)
            results[(layers, batch)] = step_s
            print(
                f"[real L={layers} B={batch}] first call {compile_s:.1f} s, "
                f"{step_s * 1e6:.0f} us per step "
                f"({step_s * 1e6 / layers:.1f} us / layer incl. lm_head+glue)"
            )
        del weights
    for batch in (1, 8):
        per_layer = (results[(8, batch)] - results[(4, batch)]) / 4
        fixed = results[(4, batch)] - 4 * per_layer
        print(f"[real B={batch}] marginal {per_layer * 1e6:.1f} us per layer, "
              f"{fixed * 1e6:.0f} us fixed (lm_head + glue + per-step overhead); "
              f"projected 62-layer step {(fixed + 62 * per_layer) * 1e3:.2f} ms")


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-s", "-q"])
