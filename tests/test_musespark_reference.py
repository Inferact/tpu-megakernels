"""CPU tests for the Muse Spark reference model (MINI config, design.md section 7 item 1).

`spec_decoder_layer` below is an INDEPENDENT transcription of the spec's section 6 pseudo-code
(nn.Linear `[out, in]` weights, `lax.top_k`, eager bf16 casts); it shares no helper with
`musespark` and is the oracle the package reference is checked against.
"""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax import lax

import musespark as ms
from musespark import MINI, Config, layout

CHECKPOINT = Path("/filestore/weights/Muse-Spark-1.2-816B-A42B-open")
F32, BF16 = jnp.float32, jnp.bfloat16


# ---- independent re-implementation of one layer (spec section 6) --------------------------------
def spec_r16(x):
    return x.astype(BF16).astype(F32)


def spec_rms(x, w=None, eps=1e-5):
    y = x * lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if w is None else y * w


def spec_mm(a, w_out_in):
    return spec_r16(jnp.dot(a.astype(BF16), w_out_in.T, preferred_element_type=F32))


def spec_gate_coeffs(g_bf16, temperature=0.3):
    u = g_bf16.astype(F32) / temperature
    beta = jax.nn.sigmoid(u)
    alpha = jnp.sqrt(jnp.maximum(jax.nn.sigmoid(-u) * (1.0 + beta), 1e-3))
    return alpha, beta


def spec_eff(w_bf16, center=1.0):
    return (w_bf16 + jnp.bfloat16(center)).astype(F32)


def spec_rope(x, pos, D, theta):
    inv = theta ** (-(jnp.arange(0, D, 2, dtype=F32)) / D)
    ang = pos.astype(F32)[:, None] * inv[None, :]
    c, s = jnp.cos(ang)[:, None, :], jnp.sin(ang)[:, None, :]
    a, b = x[..., : D // 2], x[..., D // 2 :]
    return spec_r16(jnp.concatenate([a * c - b * s, a * s + b * c], axis=-1))


def spec_decoder_layer(P, s, x_attn, pos, is_sliding, use_rope, c):
    """Spec section 6 for a fresh sequence of T tokens (the cache holds exactly these keys)."""
    H, NH, NKV, D = c.hidden, c.heads, c.kv_heads, c.head_dim
    HM, I, K = c.moe_hidden, c.expert_hidden, c.top_k
    EPS_PRE, EPS_POST = c.rms_eps, c.post_eps
    SCALE = c.qk_scale_factor / D**0.5
    ROUTER_MUL, ROUTE_EPS, WINDOW = c.output_multiplier, c.route_eps, c.sliding_window
    T = s.shape[0]
    q = spec_mm(x_attn, P.q_proj).reshape(T, NH, D)
    k = spec_mm(x_attn, P.k_proj).reshape(T, NKV, D)
    v = spec_mm(x_attn, P.v_proj).reshape(T, NKV, D)
    g = spec_mm(x_attn, P.gate_proj).reshape(T, NH, D)
    q = spec_r16(spec_rms(q, None, EPS_PRE))
    k = spec_r16(spec_rms(k, None, EPS_PRE))
    if use_rope:
        q = spec_rope(q, pos, D, c.rope_theta)
        k = spec_rope(k, pos, D, c.rope_theta)
    K_all, V_all, kpos = k.astype(BF16).astype(F32), v.astype(BF16).astype(F32), pos
    kk = jnp.repeat(K_all, NH // NKV, axis=1)
    vv = jnp.repeat(V_all, NH // NKV, axis=1)
    scores = SCALE * jnp.einsum("thd,shd->hts", q, kk, precision=lax.Precision.HIGHEST)
    allowed = kpos[None, :] <= pos[:, None]
    if is_sliding:
        allowed &= kpos[None, :] >= pos[:, None] - (WINDOW - 1)
    scores = jnp.where(allowed[None], scores, -jnp.inf)
    p = jax.nn.softmax(scores, axis=-1)
    o = jnp.einsum("hts,shd->thd", p, vv, precision=lax.Precision.HIGHEST)
    o = spec_rms(o, None, EPS_PRE) * jax.nn.sigmoid(g)
    attn_out = spec_mm(spec_r16(o).reshape(T, H), P.o_proj)
    nb = spec_r16(spec_rms(attn_out, None, EPS_POST))
    a1, b1 = spec_gate_coeffs(P.post_attention_residual_gate, c.gate_temperature)
    s = a1 * s + b1 * nb
    x_ffn = spec_r16(spec_rms(s, spec_eff(P.pre_feedforward_layernorm)))
    logits = jnp.dot(x_ffn, P.router_w.T, precision=lax.Precision.HIGHEST)
    sc = jax.nn.sigmoid(logits * ROUTER_MUL)
    idx = lax.top_k(sc + P.router_bias, K)[1]
    w = jnp.take_along_axis(sc, idx, axis=-1)
    w = w / (w.sum(-1, keepdims=True) + ROUTE_EPS) * 1.0
    h1 = spec_r16(spec_rms(spec_mm(x_ffn, P.pre_expert_proj), spec_eff(P.pre_expert_norm)))
    w_post = spec_eff(P.post_expert_norm)
    m = jnp.zeros((T, HM), F32)
    for kslot in range(K):
        e = idx[:, kslot]
        gu = spec_r16(
            jnp.einsum(
                "th,toh->to", h1.astype(BF16), P.gate_up[e].astype(BF16), preferred_element_type=F32
            )
        )
        act = spec_r16(jax.nn.silu(gu[:, :I]) * gu[:, I:])
        y = spec_r16(
            jnp.einsum(
                "ti,toi->to", act.astype(BF16), P.down[e].astype(BF16), preferred_element_type=F32
            )
        )
        t = spec_r16(y * w_post)
        yn = t * lax.rsqrt(jnp.mean(t * t, -1, keepdims=True) + EPS_POST)
        m = m + w[:, kslot : kslot + 1] * yn
    ffn_out = spec_mm(spec_r16(m), P.post_expert_proj)
    t = spec_r16(ffn_out * spec_eff(P.post_feedforward_layernorm))
    nb = spec_r16(t * lax.rsqrt(jnp.mean(t * t, -1, keepdims=True) + EPS_POST))
    a2, b2 = spec_gate_coeffs(P.post_feedforward_residual_gate, c.gate_temperature)
    s = a2 * s + b2 * nb
    x_next = spec_r16(spec_rms(s, spec_eff(P.next_input_layernorm)))
    return s, x_next


def raw_layer_params(c, rng, layer, quantized_experts):
    """Random RAW checkpoint-style parameters of one layer (nn.Linear `[out, in]`, raw gammas)."""
    bf = ml_dtypes.bfloat16

    def lin(out, inp):
        return jnp.asarray((rng.standard_normal((out, inp), np.float32) / np.sqrt(inp)).astype(bf))

    H, HM, I, E = c.hidden, c.moe_hidden, c.expert_hidden, c.experts
    P = SimpleNamespace(
        q_proj=lin(c.q_width, H),
        k_proj=lin(c.kv_width, H),
        v_proj=lin(c.kv_width, H),
        gate_proj=lin(c.q_width, H),
        o_proj=lin(H, c.q_width),
        pre_expert_proj=lin(HM, H),
        post_expert_proj=lin(H, HM),
        router_w=jnp.asarray(rng.standard_normal((E, H), np.float32) / np.sqrt(H)),
        router_bias=jnp.asarray(0.1 * rng.standard_normal(E, np.float32)),
        input_layernorm=jnp.asarray((0.1 * rng.standard_normal(H, np.float32)).astype(bf)),
        next_input_layernorm=jnp.asarray((0.1 * rng.standard_normal(H, np.float32)).astype(bf)),
        pre_feedforward_layernorm=jnp.asarray(
            (0.1 * rng.standard_normal(H, np.float32)).astype(bf)
        ),
        post_feedforward_layernorm=jnp.asarray(
            (0.1 * rng.standard_normal(H, np.float32)).astype(bf)
        ),
        pre_expert_norm=jnp.asarray((0.1 * rng.standard_normal(HM, np.float32)).astype(bf)),
        post_expert_norm=jnp.asarray((0.1 * rng.standard_normal(HM, np.float32)).astype(bf)),
        post_attention_residual_gate=jnp.asarray(
            (0.5 * rng.standard_normal(H, np.float32)).astype(bf)
        ),
        post_feedforward_residual_gate=jnp.asarray(
            (0.5 * rng.standard_normal(H, np.float32)).astype(bf)
        ),
    )
    gate_up = rng.standard_normal((E, 2 * I, HM), np.float32) / np.sqrt(HM)  # [E, out, in]
    down = rng.standard_normal((E, HM, I), np.float32) / np.sqrt(I)
    if quantized_experts:
        # the test's own dequantization of the canonical int4 form (K = in axis of x @ W)
        canon = ms.canonical_experts(c, gate_up, down)
        G = c.group_size
        gu = canon["gate_up_q"].astype(np.float32) * np.repeat(canon["gate_up_s"], G, axis=1)
        dn = canon["down_q"].astype(np.float32) * np.repeat(canon["down_s"], G, axis=1)
        gate_up, down = gu.transpose(0, 2, 1), dn.transpose(0, 2, 1)
        P.canonical_experts = canon
    P.gate_up = jnp.asarray(gate_up.astype(bf))
    P.down = jnp.asarray(down.astype(bf))
    return P


def canonical_from_raw(c, P, layer, quantized_experts):
    """Build the package's canonical dict (single layer, stacked to L=1) from the raw params."""
    pre = f"model.language_model.layers.{layer}."
    tensors = {
        f"{pre}input_layernorm.weight": P.input_layernorm,
        f"{pre}pre_feedforward_layernorm.weight": P.pre_feedforward_layernorm,
        f"{pre}post_feedforward_layernorm.weight": P.post_feedforward_layernorm,
        f"{pre}mlp.pre_expert_norm.weight": P.pre_expert_norm,
        f"{pre}mlp.experts.post_expert_norm.weight": P.post_expert_norm,
        f"{pre}mlp.gate.weight": P.router_w,
        f"{pre}mlp.gate.e_score_correction_bias": P.router_bias,
        f"{pre}self_attn.q_proj.weight": P.q_proj,
        f"{pre}self_attn.k_proj.weight": P.k_proj,
        f"{pre}self_attn.v_proj.weight": P.v_proj,
        f"{pre}self_attn.gate_proj.weight": P.gate_proj,
        f"{pre}self_attn.o_proj.weight": P.o_proj,
        f"{pre}mlp.pre_expert_proj.weight": P.pre_expert_proj,
        f"{pre}mlp.post_expert_proj.weight": P.post_expert_proj,
        f"{pre}post_attention_residual_gate.gate": P.post_attention_residual_gate,
        f"{pre}post_feedforward_residual_gate.gate": P.post_feedforward_residual_gate,
    }
    lw = ms.canonical_layer_from_checkpoint(
        c, layer, lambda n: np.asarray(tensors[n]), experts=False
    )
    if quantized_experts:
        lw.update(P.canonical_experts)
    else:
        lw["gate_up"] = np.asarray(P.gate_up).transpose(0, 2, 1)
        lw["down"] = np.asarray(P.down).transpose(0, 2, 1)
    return ms.to_device({k: v[None] for k, v in lw.items()})


@pytest.mark.parametrize("layer", [0, 1])  # 0: sliding + RoPE, 1: full attention + NoPE
@pytest.mark.parametrize("quantized", [False, True])
def test_layer_matches_spec_pseudocode(layer, quantized):
    c = MINI
    rng = np.random.default_rng(10 + layer)
    P = raw_layer_params(c, rng, layer, quantized)
    T = 6
    s0 = jnp.asarray(rng.standard_normal((T, c.hidden), np.float32))
    x0 = spec_r16(spec_rms(s0, spec_eff(P.input_layernorm)))
    pos = jnp.arange(T, dtype=jnp.int32)
    s_spec, x_spec = spec_decoder_layer(
        P, s0, x0, pos, is_sliding=(layer % 4 != 1), use_rope=(layer % 4 != 1), c=c
    )

    weights = canonical_from_raw(c, P, layer, quantized)
    lw = ms.layer_weights(weights, 0)
    k_cache = jnp.zeros((1, 128, c.kv_heads, c.head_dim), BF16)
    s_ref, _, _ = ms.decoder_layer(c, lw, s0[None], x0[None], pos[None], k_cache, k_cache, layer)
    next_norm = ms.eff_weight(P.next_input_layernorm)
    x_ref = ms.r16(ms.rms(s_ref, next_norm, c.rms_eps))

    s_spec, x_spec = np.asarray(s_spec), np.asarray(x_spec)
    s_ref, x_ref = np.asarray(s_ref[0]), np.asarray(x_ref[0])
    print(
        "layer",
        layer,
        "max|ds|",
        np.abs(s_spec - s_ref).max(),
        "exact frac",
        np.mean(s_spec == s_ref),
        "max|dx|",
        np.abs(x_spec - x_ref).max(),
    )
    # Same rounding points => almost everything is bit-identical; the rest are 1-ulp bf16 flips.
    assert np.mean(s_spec == s_ref) > 0.98
    assert np.abs(s_spec - s_ref).max() <= 1.6e-2 * np.abs(s_spec).max()
    assert np.mean(x_spec == x_ref) > 0.98
    assert np.abs(x_spec - x_ref).max() <= 2.0**-6  # <= 1 bf16 ulp at |x| ~ 4


# ---- component semantics -----------------------------------------------------------------------
def test_route_iterative_argmax_ties_to_lowest_index():
    E, k = 16, 4
    logits = jnp.zeros((2, E))  # all sigmoid scores 0.5: a 16-way tie
    bias = jnp.zeros(E)
    idx, w = ms.route(logits, bias, k)
    assert idx.tolist() == [[0, 1, 2, 3]] * 2
    np.testing.assert_allclose(np.asarray(w), 0.25, rtol=1e-6)
    # bias breaks ties only for the selection; weights come from the unbiased scores
    bias = jnp.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], F32)
    idx, w = ms.route(logits, bias, k)
    assert idx.tolist() == [[2, 5, 12, 15]] * 2
    logits = jnp.array([[3.0, 3.0, -1.0, 3.0] + [-9.0] * 12])
    idx, w = ms.route(logits, jnp.zeros(E), 2)
    assert idx.tolist() == [[0, 1]]
    # random logits agree with a numpy stable sort (lowest index first among equal keys)
    rng = np.random.default_rng(0)
    lg = rng.standard_normal((5, E)).astype(np.float32)
    lg[:, 3] = lg[:, 7]  # exact ties
    b = rng.standard_normal(E).astype(np.float32) * 0.1
    idx, w = ms.route(jnp.asarray(lg), jnp.asarray(b), k, eps=1e-15)
    sc = 1 / (1 + np.exp(-lg.astype(np.float64)))
    sel = (sc + b).astype(np.float32)
    want = np.argsort(-sel, axis=-1, kind="stable")[:, :k]
    assert np.array_equal(np.asarray(idx), want)
    ws = np.take_along_axis(sc.astype(np.float32), want, axis=-1)
    np.testing.assert_allclose(np.asarray(w), ws / (ws.sum(-1, keepdims=True) + 1e-15), rtol=1e-5)


def test_gate_coeffs_formula_and_clamp():
    g = np.array([-4, -0.3, 0, 0.3, 4, 30], ml_dtypes.bfloat16)
    alpha, beta = ms.gate_coeffs(jnp.asarray(g), 0.3)
    u = g.astype(np.float64) / 0.3
    beta64 = 1 / (1 + np.exp(-u))
    alpha64 = np.sqrt(np.maximum(1 / (1 + np.exp(u)) * (1 + beta64), 1e-3))
    np.testing.assert_allclose(np.asarray(beta), beta64, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(alpha), alpha64, rtol=1e-6)
    # norm-preserving (alpha^2 + beta^2 == 1) until the clamp engages (|g| >= ~2.3 -> u >= 7.6)
    norm = np.asarray(alpha) ** 2 + np.asarray(beta) ** 2
    np.testing.assert_allclose(norm[:4], 1.0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(alpha[4:]) ** 2, 1e-3, rtol=1e-5)
    np.testing.assert_allclose(norm[4:], 1.001, atol=1e-5)


def test_eff_weight_rounds_in_bf16():
    w = np.array([0.0, 2.0**-9, 3 * 2.0**-9, -(2.0**-9), 0.1, -1.5], ml_dtypes.bfloat16)
    got = np.asarray(ms.eff_weight(jnp.asarray(w)))
    want = (w.astype(np.float32) + 1).astype(ml_dtypes.bfloat16).astype(np.float32)
    assert np.array_equal(got, want)
    assert got[1] == 1.0  # 1 + 2^-9 is a bf16 tie -> even (1.0)
    assert got[2] == 1 + 2.0**-7  # 1 + 3*2^-9 rounds up
    assert got[4] == np.float32(np.float32(1.1).astype(ml_dtypes.bfloat16))
    # the final norm uses center 0
    assert np.array_equal(np.asarray(ms.eff_weight(jnp.asarray(w), 0.0)), w.astype(np.float32))


def test_rope_rotate_half_equals_interleaved_on_permuted_weights():
    D, T, heads = 64, 5, 3
    rng = np.random.default_rng(1)
    x = jnp.asarray(
        rng.standard_normal((T, heads, D)).astype(ml_dtypes.bfloat16).astype(np.float32)
    )
    pos = jnp.array([0, 1, 7, 2047, 100000], jnp.int32)
    theta = 500000.0
    out = np.asarray(ms.rope_rotate_half(x, pos, theta))
    # sglang: permute the head dims (new 2j <- old j, new 2j+1 <- old 32+j) and rotate the
    # interleaved pairs (2j, 2j+1)
    perm = np.stack([np.arange(D // 2), np.arange(D // 2) + D // 2], axis=1).reshape(-1)
    xp = np.asarray(x)[..., perm]
    # identical angle/trig maths (XLA's cos/sin differ from numpy's by an ulp); only the
    # pairing differs
    inv = theta ** (-jnp.arange(0, D, 2, dtype=F32) / D)
    ang = pos.astype(F32)[:, None] * inv[None]  # [T, 32]
    c, s = np.asarray(jnp.cos(ang))[:, None, :], np.asarray(jnp.sin(ang))[:, None, :]
    a, b = xp[..., 0::2], xp[..., 1::2]
    inter = np.empty_like(xp)
    inter[..., 0::2] = np.asarray(jnp.asarray(a) * c - jnp.asarray(b) * s)
    inter[..., 1::2] = np.asarray(jnp.asarray(a) * s + jnp.asarray(b) * c)
    inter = inter.astype(ml_dtypes.bfloat16).astype(np.float32)
    assert np.array_equal(out[..., perm], inter)
    # frequency convention: theta ** (-2j / D), position 0 is the identity
    assert np.array_equal(out[0], np.asarray(x)[0])
    assert np.allclose(
        np.asarray(ms.rope_rotate_half(x, pos, theta))[1, 0, 32],
        np.asarray(x)[1, 0, 0] * np.sin(1.0) + np.asarray(x)[1, 0, 32] * np.cos(1.0),
        atol=1e-2,
    )


def _layer_setup(seed=3, batch=1, context=512):
    c = MINI
    w = ms.to_device(ms.random_canonical_weights(c, seed))
    rng = np.random.default_rng(seed)
    k = jnp.asarray(
        rng.standard_normal((batch, context, c.kv_heads, c.head_dim)).astype(ml_dtypes.bfloat16)
    )
    v = jnp.asarray(
        rng.standard_normal((batch, context, c.kv_heads, c.head_dim)).astype(ml_dtypes.bfloat16)
    )
    s = jnp.asarray(rng.standard_normal((batch, 1, c.hidden), np.float32))
    x = ms.r16(ms.rms(s, w["attn_norm"][0], c.rms_eps))
    return c, w, k, v, s, x


def test_sliding_window_boundary():
    """MINI window 256: position 255 still sees slot 0, position 256 does not (layer 0 slides)."""
    c, w, k, v, s, x = _layer_setup()
    lw = ms.layer_weights(w, 0)
    k_zero = k.at[:, 0].set(0)
    v_zero = v.at[:, 0].set(0)
    for pos, same in ((255, False), (256, True)):
        p = jnp.array([[pos]], jnp.int32)
        s1, *_ = ms.decoder_layer(c, lw, s, x, p, k, v, 0)
        s2, *_ = ms.decoder_layer(c, lw, s, x, p, k_zero, v_zero, 0)
        assert np.array_equal(np.asarray(s1), np.asarray(s2)) == same, pos
    # a full-attention layer at position 256 still attends to slot 0
    lw1 = ms.layer_weights(w, 1)
    p = jnp.array([[256]], jnp.int32)
    s1, *_ = ms.decoder_layer(c, lw1, s, x, p, k, v, 1)
    s2, *_ = ms.decoder_layer(c, lw1, s, x, p, k_zero, v_zero, 1)
    assert not np.array_equal(np.asarray(s1), np.asarray(s2))
    # keys at slots beyond the position are never read
    k_future = k.at[:, 300].set(1.0)
    s3, *_ = ms.decoder_layer(c, lw1, s, x, p, k_future, v, 1)
    assert np.array_equal(np.asarray(s1), np.asarray(s3))


def test_nope_on_full_layers_and_rope_on_sliding_layers():
    c, w, k, v, s, x = _layer_setup(seed=4)
    for layer in (0, 1, 2, 3):
        lw = ms.layer_weights(w, layer)
        p = jnp.array([[300]], jnp.int32)
        _, k_out, v_out = ms.decoder_layer(c, lw, s, x, p, k, v, layer)
        k_plain = ms.r16(
            ms.rms(ms.mm(x, lw["k"]).reshape(1, 1, c.kv_heads, c.head_dim), None, c.rms_eps)
        )
        v_plain = ms.mm(x, lw["v"]).reshape(1, 1, c.kv_heads, c.head_dim)
        written = np.asarray(k_out[:, 300].astype(F32))
        if c.is_full_attention(layer):
            assert layer == 1
            assert np.array_equal(written, np.asarray(k_plain[:, 0]))  # NoPE: keys stored as-is
        else:
            assert not np.array_equal(written, np.asarray(k_plain[:, 0]))
            rotated = ms.rope_rotate_half(k_plain, p, c.rope_theta)
            assert np.array_equal(written, np.asarray(rotated[:, 0]))
        assert np.array_equal(
            np.asarray(v_out[:, 300].astype(F32)), np.asarray(v_plain[:, 0])
        )  # raw v
    # (Shift invariance cannot be tested with slot == position caches: a row starting at p > 0
    # attends to the p empty slots before it. Query-side NoPE is covered by the spec comparison.)


def test_prefill_equals_sequential_decode():
    c = MINI
    w = ms.to_device(ms.random_canonical_weights(c, 5))
    rng = np.random.default_rng(5)
    B, T, context = 2, 8, 256
    tokens = jnp.asarray(rng.integers(0, c.vocab, (B, T)), jnp.int32)
    start = jnp.array([0, 3], jnp.int32)  # rows are independent, with their own positions
    caches = ms.init_caches(c, B, context)
    logits_pre, caches_pre = jax.jit(lambda w, t, s, cc: ms.forward(c, w, t, s, cc))(
        w, tokens, start, caches
    )
    step = jax.jit(lambda w, t, p, cc: ms.decode_step(c, w, t, p, cc))
    caches_seq = ms.init_caches(c, B, context)
    worst = 0.0
    for t in range(T):
        logits_t, caches_seq = step(w, tokens[:, t], start + t, caches_seq)
        diff = np.abs(np.asarray(logits_t) - np.asarray(logits_pre[:, t])).max()
        worst = max(worst, float(diff))
        assert np.array_equal(
            np.argmax(np.asarray(logits_t), -1), np.argmax(np.asarray(logits_pre[:, t]), -1)
        )
    scale = float(np.abs(np.asarray(logits_pre)).max())
    print("prefill vs decode max |logit diff|", worst, "max |logit|", scale)
    # Identical maths, but T=8 vs T=1 matmuls accumulate in a different order: rare 1-ulp bf16
    # flips (layer 0 is bit-exact) amplified by the random model's gated residuals. Measured
    # ~3.4e-2 on softcapped logits of magnitude <= 0.75; the kernel tests use the same 5e-2.
    assert worst < 5e-2
    for a, b in zip(caches_pre, caches_seq):
        a, b = np.asarray(a.astype(F32)), np.asarray(b.astype(F32))
        assert np.abs(a - b).max() < 3e-2 * max(float(np.abs(b).max()), 1.0)
        assert np.mean(a == b) > 0.98  # the rest are 1-ulp bf16 flips
    # the untouched cache slots stay zero
    assert not np.asarray(caches_pre[0][:, 0, T:]).astype(np.float32).any()
    assert not np.asarray(caches_pre[0][:, 1, :3]).astype(np.float32).any()


def test_batch_rows_are_independent():
    c = MINI
    w = ms.to_device(ms.random_canonical_weights(c, 6))
    tokens = jnp.array([[5, 6, 7], [900, 1, 2]], jnp.int32)
    start = jnp.array([0, 10], jnp.int32)
    both, _ = ms.forward(c, w, tokens, start, ms.init_caches(c, 2, 128))
    for b in range(2):
        one, _ = ms.forward(c, w, tokens[b : b + 1], start[b : b + 1], ms.init_caches(c, 1, 128))
        assert np.abs(np.asarray(one[0]) - np.asarray(both[b])).max() < 1e-2


def test_quantized_reference_equals_dense_dequantized():
    c = MINI
    wq = ms.random_canonical_weights(c, 7)
    G = c.group_size
    wd = {k: v for k, v in wq.items() if not k.startswith(("gate_up_", "down_"))}
    wd["gate_up"] = (
        wq["gate_up_q"].astype(np.float32) * np.repeat(wq["gate_up_s"], G, axis=2)
    ).astype(ml_dtypes.bfloat16)
    wd["down"] = (wq["down_q"].astype(np.float32) * np.repeat(wq["down_s"], G, axis=2)).astype(
        ml_dtypes.bfloat16
    )
    tokens = jnp.array([[1, 2, 3, 4]], jnp.int32)
    lq, _ = ms.forward(c, ms.to_device(wq), tokens, jnp.array([0]), ms.init_caches(c, 1, 128))
    ld, _ = ms.forward(c, ms.to_device(wd), tokens, jnp.array([0]), ms.init_caches(c, 1, 128))
    assert np.array_equal(np.asarray(lq), np.asarray(ld))
    assert np.abs(np.asarray(lq)).max() <= c.softcap  # softcapped
    assert bool(np.isfinite(np.asarray(lq)).all())


def test_generate_greedy_and_vocab_mask():
    c = MINI
    w = ms.to_device(ms.random_canonical_weights(c, 8))
    out = ms.generate_greedy(c, w, [1, 2, 3], 4, 64)
    assert len(out) == 4 and all(0 <= t < c.vocab for t in out)
    small = Config(**{**c.__dict__, "vocab_used": 100})
    masked = ms.mask_unused_vocab(small, jnp.zeros((2, c.vocab)))
    assert bool(jnp.isfinite(masked[:, :100]).all()) and bool(jnp.isneginf(masked[:, 100:]).all())


# ---- canonical <-> per-rank layout ---------------------------------------------------------------
def test_shard_unshard_roundtrip_and_layout_shapes():
    c = MINI
    w = ms.random_canonical_weights(c, 9)
    sharded = ms.shard_canonical(c, w, tp=8)
    shapes = layout.rank_shapes(c, 8)
    assert set(sharded) == set(shapes)
    for name, (shape, dtype) in shapes.items():
        assert sharded[name].shape == (8,) + shape, name
        assert sharded[name].dtype == np.dtype(dtype), name
    back = ms.unshard(c, sharded, tp=8)
    assert set(back) == set(w)
    for name in w:
        a, b = np.asarray(w[name]), np.asarray(back[name])
        assert a.shape == b.shape, name
        assert np.array_equal(a.astype(np.float32), b.astype(np.float32)), name
    # replicated families are identical on every rank
    for name in layout.VECTOR_FAMILIES + ("final_norm", "router_hi", "router_lo"):
        assert all(np.array_equal(sharded[name][0], sharded[name][r]) for r in range(8)), name


def test_shard_matches_design_table_slices():
    c = MINI
    w = ms.random_canonical_weights(c, 11)
    r = 3
    rank = ms.shard_rank(c, w, r, tp=8)
    qw, kvw, isl = 128, 64, 64
    assert np.array_equal(rank["q"], np.asarray(w["q"])[:, :, r * qw : (r + 1) * qw])
    assert np.array_equal(rank["kv"][:, :, :kvw], np.asarray(w["k"])[:, :, r * kvw : (r + 1) * kvw])
    assert np.array_equal(rank["kv"][:, :, kvw:], np.asarray(w["v"])[:, :, r * kvw : (r + 1) * kvw])
    assert np.array_equal(rank["o"], np.asarray(w["o"])[:, r * qw : (r + 1) * qw, :])
    assert np.array_equal(rank["pre"], np.asarray(w["pre"])[:, :, r * 64 : (r + 1) * 64])
    assert np.array_equal(rank["post"], np.asarray(w["post"])[:, :, r * 128 : (r + 1) * 128])
    hi, lo = rank["router_hi"].astype(np.float32), rank["router_lo"].astype(np.float32)
    assert np.array_equal(hi, np.asarray(w["router"]).astype(ml_dtypes.bfloat16).astype(np.float32))
    assert (
        np.abs(hi + lo - np.asarray(w["router"])).max() == 0
    )  # exactly representable by construction
    I = c.expert_hidden
    gq = np.asarray(w["gate_up_q"])
    assert np.array_equal(
        rank["gate_up_q"][:, :, :, :isl].astype(np.int8), gq[:, :, :, r * isl : (r + 1) * isl]
    )
    assert np.array_equal(
        rank["gate_up_q"][:, :, :, isl:].astype(np.int8),
        gq[:, :, :, I + r * isl : I + (r + 1) * isl],
    )
    assert np.array_equal(
        rank["down_q"].astype(np.int8), np.asarray(w["down_q"])[:, :, r * isl : (r + 1) * isl, :]
    )
    G = c.group_size
    assert np.array_equal(
        rank["down_s"][:, :, 0, 0], np.asarray(w["down_s"])[:, :, r * isl // G, :]
    )
    assert np.array_equal(
        rank["gate_up_s"].reshape(4, 16, -1, 128)[..., :isl],
        np.asarray(w["gate_up_s"])[..., r * isl : (r + 1) * isl],
    )
    assert np.array_equal(rank["attn_norm"][:, 0], np.asarray(w["attn_norm"]))
    assert np.array_equal(rank["attn_gate_beta"][:, 0], np.asarray(w["attn_gate_beta"]))
    # vocab shards: Vp = ceil(2048 / 8 / 1024) * 1024 = 1024 rows per rank -> ranks 0 and 1 hold
    # ids 0:1024 / 1024:2048 and ranks 2..7 are all zero padding
    assert layout.vocab_pad(c, 8) == 1024
    assert np.array_equal(ms.shard_rank(c, w, 0, 8)["embed"], np.asarray(w["embed"])[:1024])
    assert np.array_equal(ms.shard_rank(c, w, 1, 8)["embed"], np.asarray(w["embed"])[1024:])
    assert np.array_equal(ms.shard_rank(c, w, 1, 8)["lm_head"], np.asarray(w["lm_head"])[:, 1024:])
    assert not rank["embed"].astype(np.float32).any()
    assert not rank["lm_head"].astype(np.float32).any()
    # dense canonical experts are quantized on the way to the rank layout
    dense = ms.random_canonical_weights(c, 12, quantized=False)
    rank_d = ms.shard_rank(c, dense, r, tp=8)
    canon = ms.unshard(c, ms.shard_canonical(c, dense, 8), 8)
    deq = ms.quant.dequantize_int4(canon["gate_up_q"], canon["gate_up_s"])
    assert (
        np.abs(deq - np.asarray(dense["gate_up"], np.float32)).max()
        <= np.repeat(canon["gate_up_s"], G, axis=2).max() / 2
    )
    assert rank_d["gate_up_q"].shape == layout.rank_shapes(c, 8)["gate_up_q"][0]


def test_cache_shard_roundtrip():
    c = MINI
    rng = np.random.default_rng(13)
    caches = tuple(
        jnp.asarray(
            rng.standard_normal((c.layers, 2, 128, c.kv_heads, c.head_dim)).astype(
                ml_dtypes.bfloat16
            )
        )
        for _ in range(2)
    )
    sharded = ms.shard_caches(c, caches, 8)
    shapes = layout.kv_cache_shapes(c, 2, 128, 8)
    for name, (shape, dtype) in shapes.items():
        assert sharded[name].shape == (8,) + shape and sharded[name].dtype == np.dtype(dtype)
    assert not sharded["k_cache"][..., 64:].astype(np.float32).any()  # MINI: one kv head per rank
    assert np.array_equal(sharded["k_cache"][5, :, :, :, :64], np.asarray(caches[0])[:, :, :, 5])
    back = ms.unshard_caches(c, sharded, 8)
    for a, b in zip(caches, back):
        assert np.array_equal(np.asarray(a), b)


def test_config_from_checkpoint():
    if not (CHECKPOINT / "config.json").exists():
        pytest.skip("checkpoint missing")
    c = Config.from_checkpoint(CHECKPOINT)
    assert c == Config()
    assert c.full_layers == tuple(range(1, 62, 4)) and len(c.sliding_layers) == 46
    assert c.eos == (200001, 200008) and c.bos == 200000 and c.pad == 200018
    assert c.softmax_scale == 0.6841258107984375


def test_config_validation():
    with pytest.raises(ValueError):
        Config(heads=100)
    assert MINI.is_full_attention(1) and not MINI.is_full_attention(0)
    assert MINI.full_layers == (1,)
