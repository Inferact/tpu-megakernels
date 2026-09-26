"""Muse Spark 1.2 text decoder: configuration and pure-JAX reference model.

The reference follows `muse_spark_spec.md` section 3/6 to the letter: fp32 residual stream,
bf16 GEMM inputs with fp32 accumulation, and an explicit bf16 rounding (`r16`) at every point
where the production model rounds. Activations are carried as f32 arrays holding
bf16-representable values so that every rounding point is visible in the code.

`r16` is implemented with `lax.reduce_precision`, which XLA honours even under its default
`--xla_allow_excess_precision=true` (a `f32 -> bf16 -> f32` convert pair may be elided).

Canonical (unsharded) weight dict, all matrices `[in, out]` (`y = x @ W`), layer axis first;
`V` vocab, `H` hidden, `Hm` moe_hidden, `I` expert_hidden, `E` experts, `G` group_size:

    embed            [V, H]        bf16   embed_tokens
    lm_head          [H, V]        bf16   lm_head.weight.T
    final_norm       [H]           bf16   model.norm.weight as-is (gain center 0: NO +1)
    attn_norm        [L, H]        bf16   EFFECTIVE r16(1 + input_layernorm)
    ffn_norm         [L, H]        bf16   EFFECTIVE r16(1 + pre_feedforward_layernorm)
    post_ffn_norm    [L, H]        bf16   EFFECTIVE r16(1 + post_feedforward_layernorm)
    attn_gate_alpha  [L, H]        f32    gate_coeffs(post_attention_residual_gate.gate)
    attn_gate_beta   [L, H]        f32
    ffn_gate_alpha   [L, H]        f32    gate_coeffs(post_feedforward_residual_gate.gate)
    ffn_gate_beta    [L, H]        f32
    pre_expert_norm  [L, Hm]       bf16   EFFECTIVE r16(1 + mlp.pre_expert_norm)
    post_expert_norm [L, Hm]       bf16   EFFECTIVE r16(1 + mlp.experts.post_expert_norm)
    router           [L, H, E]     f32    mlp.gate.weight.T (fp32 in the checkpoint)
    router_bias      [L, E]        f32    mlp.gate.e_score_correction_bias
    q, gate          [L, H, nH*D]  bf16   q_proj.T / self_attn.gate_proj.T, head-major columns
    k, v             [L, H, nKV*D] bf16   k_proj.T / v_proj.T
    o                [L, nH*D, H]  bf16   o_proj.T
    pre              [L, H, Hm]    bf16   mlp.pre_expert_proj.T
    post             [L, Hm, H]    bf16   mlp.post_expert_proj.T
    experts, EITHER dense
      gate_up        [L, E, Hm, 2I] bf16  gate_up_proj.transpose(0,2,1): cols [0,I) gate, [I,2I) up
      down           [L, E, I, Hm]  bf16  down_proj.transpose(0, 2, 1)
    OR int4 group-quantized (`musespark.quant`, unsharded, flat group-major scales)
      gate_up_q      [L, E, Hm, 2I] int4 (or int8 values in [-8, 7])
      gate_up_s      [L, E, Hm/G, 2I] f32
      down_q         [L, E, I, Hm]  int4
      down_s         [L, E, I/G, Hm] f32

The reference dequantizes on the fly (`dequantize_int4(...).astype(bf16)`), so one model serves
both "exact vs PyTorch" (dense bf16) and "same weights as the kernel" (quantized) comparisons.
KV caches are `(k, v)` with shape `[L, B, context, nKV, D]` bf16, token at position `p` in
slot `p`; K holds post-QK-norm, post-RoPE keys, V raw values.
`shard_canonical` produces the per-rank kernel layout of `musespark.layout` from this dict.
"""

from functools import partial

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax import lax

from musespark import layout, quant  # noqa: F401  (re-exported as musespark.quant)
from musespark.config import MINI, Config
from musespark.quant import (
    dequant_fp4_jnp,
    dequantize_int4,
    quantize_int4,
    scales_from_chunked,
    scales_to_chunked,
)

__all__ = [
    "MINI",
    "Config",
    "decode_step",
    "decoder_layer",
    "eff_weight",
    "embed",
    "forward",
    "gate_coeffs",
    "generate_greedy",
    "init_caches",
    "mask_unused_vocab",
    "mm",
    "prefill_reference",
    "r16",
    "random_canonical_weights",
    "rms",
    "rope_rotate_half",
    "route",
    "shard_canonical",
    "softcap_logits",
    "unshard",
]

F32 = jnp.float32
BF16 = jnp.bfloat16
NP_BF16 = ml_dtypes.bfloat16
HIGHEST = lax.Precision.HIGHEST

CHECKPOINT_PREFIX = "model.language_model."
LAYER_NAMES = (
    "attn_norm",
    "ffn_norm",
    "post_ffn_norm",
    "attn_gate_alpha",
    "attn_gate_beta",
    "ffn_gate_alpha",
    "ffn_gate_beta",
    "pre_expert_norm",
    "post_expert_norm",
    "router",
    "router_bias",
    "q",
    "k",
    "v",
    "gate",
    "o",
    "pre",
    "post",
    "gate_up",
    "down",
    "gate_up_q",
    "gate_up_s",
    "down_q",
    "down_s",
    "gate_up_fp4",
    "gate_up_bs",
    "down_fp4",
    "down_bs",
    "expert_gs",
)


# ---- rounding / numeric helpers ---------------------------------------------------------------
def r16(x):
    """Round to bfloat16 and back to f32 (the spec's `r16`); robust to XLA excess precision."""
    return lax.reduce_precision(jnp.asarray(x, F32), exponent_bits=8, mantissa_bits=7)


def rms(x, weight=None, eps=1e-5):
    """torch `F.rms_norm` semantics in f32: `x * rsqrt(mean(x^2) + eps) [* weight]`."""
    x = jnp.asarray(x, F32)
    y = x * lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return y if weight is None else y * jnp.asarray(weight, F32)


def mm(x, w):
    """bf16 x bf16 GEMM, f32 accumulation, bf16-rounded output (returned as f32)."""
    return r16(jnp.dot(x.astype(BF16), w.astype(BF16), preferred_element_type=F32))


def gate_coeffs(gate, temperature=0.3):
    """Residual gate `[H]` bf16 -> (alpha, beta) f32: `s = alpha * s + beta * branch`."""
    u = jnp.asarray(gate, F32) / temperature
    beta = jax.nn.sigmoid(u)
    alpha = jnp.sqrt(jnp.maximum(jax.nn.sigmoid(-u) * (1.0 + beta), 1e-3))
    return alpha, beta


def eff_weight(w, center=1.0):
    """Zero-centered gamma: `r16(w + center)` (bf16 maths); returns f32 holding bf16 values."""
    return r16(jnp.asarray(w, F32) + center)


def rope_rotate_half(x, positions, theta=500000.0):
    """Rotate-half RoPE on `[..., T, heads, D]` with `positions [..., T]`; pairs (j, j + D/2).

    `inv_freq[j] = theta ** (-2j / D)`; input values are bf16-rounded, the result is `r16`'d.
    """
    D = x.shape[-1]
    inv = theta ** (-jnp.arange(0, D, 2, dtype=F32) / D)  # [D/2]
    ang = jnp.asarray(positions, F32)[..., None, None] * inv  # [..., T, 1, D/2]
    c, s = jnp.cos(ang), jnp.sin(ang)
    a, b = x[..., : D // 2], x[..., D // 2 :]
    return r16(jnp.concatenate([a * c - b * s, a * s + b * c], axis=-1))


def route(logits, bias, k, eps=1e-15):
    """Top-k routing on sigmoid scores; `logits [..., E]` already carry the output multiplier.

    Selection uses `sigmoid(logits) + bias` with iterative argmax (ties -> lowest index); the
    mixing weights are the UNBIASED scores renormalised by `sum + eps`. Returns (idx, w) `[..., k]`.
    """
    scores = jax.nn.sigmoid(jnp.asarray(logits, F32))
    sel = scores + jnp.asarray(bias, F32)
    lanes = jnp.arange(sel.shape[-1])
    idx = []
    for _ in range(k):
        i = jnp.argmax(sel, axis=-1)  # first maximum -> lowest expert index on ties
        idx.append(i)
        sel = jnp.where(lanes == i[..., None], -jnp.inf, sel)
    idx = jnp.stack(idx, axis=-1).astype(jnp.int32)
    w = jnp.take_along_axis(scores, idx, axis=-1)
    return idx, w / (jnp.sum(w, axis=-1, keepdims=True) + eps)


def softcap_logits(cfg, raw):
    """`softcap * tanh(raw * output_multiplier / softcap)` on the f32 lm_head output."""
    return cfg.softcap * jnp.tanh(jnp.asarray(raw, F32) * (cfg.output_multiplier / cfg.softcap))


def mask_unused_vocab(cfg, logits):
    """-inf on the untrained padding ids `>= cfg.vocab_used` (last axis is the vocab)."""
    return jnp.where(jnp.arange(logits.shape[-1]) < cfg.vocab_used, logits, -jnp.inf)


def allowed_keys(cfg, positions, slots, layer):
    """Causal (+ sliding window on non-full layers) mask `[..., T, S]` for `positions [..., T]`."""
    p = jnp.asarray(positions)[..., None]
    ok = slots <= p
    if not cfg.is_full_attention(layer):
        ok &= slots > p - cfg.sliding_window  # sliding_window - 1 previous keys + self
    return ok


# ---- model -------------------------------------------------------------------------------------
def embed(weights, tokens, eps=1e-5):
    """`r16(rms(embed[tokens]))`: parameter-free embedding norm; f32 `[..., H]` (bf16 values)."""
    e0 = jnp.take(jnp.asarray(weights["embed"]), jnp.asarray(tokens), axis=0)
    return r16(rms(e0.astype(F32), None, eps))


def layer_weights(weights, layer):
    """The `layer`-th slice of every per-layer entry of a canonical dict."""
    return {name: weights[name][layer] for name in LAYER_NAMES if name in weights}


def expert_weights(lw):
    """(gate_up `[E, Hm, 2I]`, down `[E, I, Hm]`) bf16, dequantizing int4 experts on the fly.

    NVFP4 entries (`gate_up_fp4 [E, Hm/8, 2I]` int32, `gate_up_bs [E, Hm/16, 2I]` e4m3,
    `down_fp4`, `down_bs`, `expert_gs [E, 8, 128]` f32 with rows 0/1/2 = gate/up/down global
    scale; see `musespark.quant` / `musespark.layout`) are dequantized exactly the same way.
    """
    if "gate_up_fp4" in lw:
        gs = jnp.asarray(lw["expert_gs"], F32)  # [E, 8, 128]
        two_i = lw["gate_up_fp4"].shape[-1]
        col = jnp.arange(two_i, dtype=jnp.int32)[None, :] < two_i // 2
        gu_gs = jnp.where(col, gs[:, 0:1, 0:1], gs[:, 1:2, 0:1])  # [E, 1, 2I]
        gate_up = dequant_fp4_jnp(jnp.asarray(lw["gate_up_fp4"]), lw["gate_up_bs"], gu_gs)
        down = dequant_fp4_jnp(jnp.asarray(lw["down_fp4"]), lw["down_bs"], gs[:, 2:3, 0:1])
        return gate_up.astype(BF16), down.astype(BF16)
    if "gate_up_q" in lw:
        gate_up = dequantize_int4(jnp.asarray(lw["gate_up_q"]), lw["gate_up_s"]).astype(BF16)
        down = dequantize_int4(jnp.asarray(lw["down_q"]), lw["down_s"]).astype(BF16)
        return gate_up, down
    return jnp.asarray(lw["gate_up"]), jnp.asarray(lw["down"])


def attention_branch(cfg, lw, x_attn, positions, k_cache, v_cache, layer):
    """Attention branch for `x_attn [B, T, H]`; returns (attn_out `[B, T, H]`, k_cache, v_cache).

    Keys/values of the T tokens are written to cache slots `positions` first, then every query
    attends over the whole cache row under the causal / sliding-window mask.
    """
    B, T, _ = x_attn.shape
    nH, nKV, D = cfg.heads, cfg.kv_heads, cfg.head_dim
    q = mm(x_attn, lw["q"]).reshape(B, T, nH, D)
    k = mm(x_attn, lw["k"]).reshape(B, T, nKV, D)
    v = mm(x_attn, lw["v"]).reshape(B, T, nKV, D)
    g = mm(x_attn, lw["gate"]).reshape(B, T, nH, D)
    q = r16(rms(q, None, cfg.rms_eps))  # parameter-free per-head QK norm, eps = rms_eps
    k = r16(rms(k, None, cfg.rms_eps))
    if not cfg.is_full_attention(layer):  # full-attention layers are NoPE
        q = rope_rotate_half(q, positions, cfg.rope_theta)
        k = rope_rotate_half(k, positions, cfg.rope_theta)
    rows = jnp.arange(B)[:, None]
    k_cache = k_cache.at[rows, positions].set(k.astype(BF16))
    v_cache = v_cache.at[rows, positions].set(v.astype(BF16))

    group = nH // nKV  # query head h uses kv head h // group
    qg = q.reshape(B, T, nKV, group, D)
    scores = (
        jnp.einsum("btkgd,bskd->bkgts", qg, k_cache.astype(F32), precision=HIGHEST)
        * cfg.softmax_scale
    )
    slots = jnp.arange(k_cache.shape[1])
    ok = allowed_keys(cfg, positions, slots, layer)  # [B, T, S]
    scores = jnp.where(ok[:, None, None], scores, -jnp.inf)
    p = jax.nn.softmax(scores, axis=-1)
    o = jnp.einsum("bkgts,bskd->btkgd", p, v_cache.astype(F32), precision=HIGHEST)
    o = o.reshape(B, T, nH, D)
    o = rms(o, None, cfg.rms_eps) * jax.nn.sigmoid(g)  # per-head norm, per-channel sigmoid gate
    attn_out = mm(r16(o).reshape(B, T, nH * D), lw["o"])
    return attn_out, k_cache, v_cache


def moe_branch(cfg, lw, x_ffn):
    """MoE branch for `x_ffn [B, T, H]` (bf16 values) -> ffn_out `[B, T, H]` (bf16 values)."""
    I = cfg.expert_hidden
    # Router: fp32 x fp32 IEEE GEMM on the 8192-wide pre-FFN norm output.
    logits = jnp.dot(x_ffn, jnp.asarray(lw["router"], F32), precision=HIGHEST)
    idx, w = route(logits * cfg.output_multiplier, lw["router_bias"], cfg.top_k, cfg.route_eps)
    h1 = r16(rms(mm(x_ffn, lw["pre"]), lw["pre_expert_norm"], cfg.rms_eps))  # [B, T, Hm]
    gate_up, down = expert_weights(lw)
    w_post = jnp.asarray(lw["post_expert_norm"], F32)
    m = jnp.zeros(h1.shape, F32)
    for slot in range(cfg.top_k):  # dense gather form; kernels group tokens by expert
        e = idx[..., slot]  # [B, T]
        gu = r16(
            jnp.einsum("bth,btho->bto", h1.astype(BF16), gate_up[e], preferred_element_type=F32)
        )
        act = r16(jax.nn.silu(gu[..., :I]) * gu[..., I:])
        y = r16(jnp.einsum("bti,btio->bto", act.astype(BF16), down[e], preferred_element_type=F32))
        t = r16(y * w_post)  # post_expert_norm: weight BEFORE the norm, no weight after
        yn = t * lax.rsqrt(jnp.mean(t * t, axis=-1, keepdims=True) + cfg.post_eps)
        m = m + w[..., slot : slot + 1] * yn
    return mm(r16(m), lw["post"])


def decoder_layer(cfg, lw, s, x_attn, positions, k_cache, v_cache, layer):
    """One decoder layer on `B` independent rows of `T` tokens each.

    s: f32 residual `[B, T, H]`; x_attn: `r16(rms(s, attn_norm))` of this layer `[B, T, H]`;
    positions: int32 `[B, T]` absolute positions; k_cache/v_cache: this layer's `[B, S, nKV, D]`.
    Returns (s, k_cache, v_cache); the caller derives the next layer's `x_attn`.
    """
    attn_out, k_cache, v_cache = attention_branch(cfg, lw, x_attn, positions, k_cache, v_cache, layer)
    # post-attention boundary: weightless post-norm, gated residual, pre-FFN norm
    nb = r16(rms(attn_out, None, cfg.post_eps))
    s = lw["attn_gate_alpha"] * s + lw["attn_gate_beta"] * nb
    x_ffn = r16(rms(s, lw["ffn_norm"], cfg.rms_eps))
    ffn_out = moe_branch(cfg, lw, x_ffn)
    # post-FFN boundary: WithInputScale post-norm, gated residual
    t = r16(ffn_out * jnp.asarray(lw["post_ffn_norm"], F32))
    nb = r16(t * lax.rsqrt(jnp.mean(t * t, axis=-1, keepdims=True) + cfg.post_eps))
    s = lw["ffn_gate_alpha"] * s + lw["ffn_gate_beta"] * nb
    return s, k_cache, v_cache


def init_caches(cfg, batch, context):
    """Zero canonical KV caches `(k, v)`, each `[L, B, context, nKV, D]` bf16."""
    shape = (cfg.layers, batch, context, cfg.kv_heads, cfg.head_dim)
    return jnp.zeros(shape, BF16), jnp.zeros(shape, BF16)


def forward(cfg, weights, tokens, start_positions, caches, return_hidden=False):
    """`tokens [B, T]`, `start_positions [B]` -> (softcapped logits `[B, T, V]` f32, caches).

    Row `b`'s token `t` sits at absolute position `start_positions[b] + t` (its cache slot).
    With `return_hidden` also returns the per-layer f32 residual streams `[L + 1, B, T, H]`.
    """
    tokens = jnp.asarray(tokens, jnp.int32)
    T = tokens.shape[1]
    positions = jnp.asarray(start_positions, jnp.int32)[:, None] + jnp.arange(T, dtype=jnp.int32)
    k_cache, v_cache = caches
    s = embed(weights, tokens, cfg.rms_eps)
    x = r16(rms(s, weights["attn_norm"][0], cfg.rms_eps))
    hidden = [s]
    for layer in range(cfg.layers):
        lw = layer_weights(weights, layer)
        s, k_l, v_l = decoder_layer(cfg, lw, s, x, positions, k_cache[layer], v_cache[layer], layer)
        k_cache, v_cache = k_cache.at[layer].set(k_l), v_cache.at[layer].set(v_l)
        hidden.append(s)
        if layer + 1 < cfg.layers:
            x = r16(rms(s, weights["attn_norm"][layer + 1], cfg.rms_eps))
    hN = r16(rms(r16(s), weights["final_norm"], cfg.rms_eps))  # final norm weight as-is (no +1)
    raw = jnp.dot(hN.astype(BF16), jnp.asarray(weights["lm_head"]), preferred_element_type=F32)
    logits = softcap_logits(cfg, raw)
    if return_hidden:
        return logits, (k_cache, v_cache), jnp.stack(hidden)
    return logits, (k_cache, v_cache)


def decode_step(cfg, weights, tokens, positions, caches):
    """One token per row: `tokens [B]` at `positions [B]` -> (logits `[B, V]`, caches)."""
    logits, caches = forward(cfg, weights, jnp.asarray(tokens)[:, None], positions, caches)
    return logits[:, 0], caches


def prefill_reference(cfg, weights, tokens, caches, start_positions=None):
    """`tokens [B, T]` from `start_positions` (default 0) -> (logits `[B, T, V]`, caches)."""
    if start_positions is None:
        start_positions = jnp.zeros(tokens.shape[0], jnp.int32)
    return forward(cfg, weights, tokens, start_positions, caches)


def generate_greedy(cfg, weights, prompt, steps, context, stop_on_eos=False):
    """Prefill `prompt` (list of ids), then `steps` greedy tokens; returns the generated ids."""
    prompt = jnp.asarray(prompt, jnp.int32)[None]
    caches = init_caches(cfg, 1, context)
    step = jax.jit(partial(decode_step, cfg))
    logits, caches = jax.jit(partial(prefill_reference, cfg))(weights, prompt, caches)
    logits = logits[:, -1]
    out = []
    for i in range(steps):
        token = int(jnp.argmax(mask_unused_vocab(cfg, logits), axis=-1)[0])
        out.append(token)
        if stop_on_eos and token in cfg.eos:
            break
        pos = jnp.array([prompt.shape[1] + i], jnp.int32)
        logits, caches = step(weights, jnp.array([token], jnp.int32), pos, caches)
    return out


# ---- canonical weights: construction ------------------------------------------------------------
def _seed(key):
    if isinstance(key, (int, np.integer)):
        return int(key)
    return int(np.asarray(jax.random.key_data(key)).ravel()[-1])


def random_canonical_weights(cfg, key=0, quantized=True):
    """Deterministic random canonical dict (numpy) with realistic magnitudes.

    Norm weights are effective values near 1, gates come from random bf16 gate parameters via
    `gate_coeffs`, the f32 router is exactly `hi + lo` representable so `unshard(shard(w))` is
    exact. Experts are int4-quantized (`quantized=True`) or dense bf16.
    """
    rng = np.random.default_rng(_seed(key))
    L, H, Hm, I, E = cfg.layers, cfg.hidden, cfg.moe_hidden, cfg.expert_hidden, cfg.experts

    def mat(*shape):
        fan_in = shape[-2]
        return (rng.standard_normal(shape, np.float32) / np.sqrt(fan_in)).astype(NP_BF16)

    def norm(*shape):
        return (1.0 + 0.1 * rng.standard_normal(shape, np.float32)).astype(NP_BF16)

    def gate(*shape):
        g = (0.5 * rng.standard_normal(shape, np.float32)).astype(NP_BF16)
        alpha, beta = gate_coeffs(jnp.asarray(g), cfg.gate_temperature)
        return np.asarray(alpha), np.asarray(beta)

    w = {
        "embed": mat(cfg.vocab, H).astype(np.float32).__mul__(np.sqrt(H)).astype(NP_BF16),
        "lm_head": mat(H, cfg.vocab),
        "final_norm": norm(H),
        "attn_norm": norm(L, H),
        "ffn_norm": norm(L, H),
        "post_ffn_norm": norm(L, H),
        "pre_expert_norm": norm(L, Hm),
        "post_expert_norm": norm(L, Hm),
        "router_bias": (0.1 * rng.standard_normal((L, E), np.float32)).astype(np.float32),
        "q": mat(L, H, cfg.q_width),
        "k": mat(L, H, cfg.kv_width),
        "v": mat(L, H, cfg.kv_width),
        "gate": mat(L, H, cfg.q_width),
        "o": mat(L, cfg.q_width, H),
        "pre": mat(L, H, Hm),
        "post": mat(L, Hm, H),
    }
    w["attn_gate_alpha"], w["attn_gate_beta"] = gate(L, H)
    w["ffn_gate_alpha"], w["ffn_gate_beta"] = gate(L, H)
    router = rng.standard_normal((L, H, E), np.float32) / np.sqrt(H)
    hi, lo = split_hi_lo(router)
    w["router"] = hi.astype(np.float32) + lo.astype(np.float32)
    gate_up = rng.standard_normal((L, E, Hm, 2 * I), np.float32) / np.sqrt(Hm)
    down = rng.standard_normal((L, E, I, Hm), np.float32) / np.sqrt(I)
    if quantized:
        w["gate_up_q"], w["gate_up_s"] = quantize_int4(gate_up, cfg.group_size)
        w["down_q"], w["down_s"] = quantize_int4(down, cfg.group_size)
    else:
        w["gate_up"], w["down"] = gate_up.astype(NP_BF16), down.astype(NP_BF16)
    return w


def to_device(tree):
    """numpy canonical dict -> jax arrays (int8 expert values become int4)."""

    def put(a):
        a = np.asarray(a)
        if a.dtype == np.int8:
            a = a.astype(jnp.int4)
        return jnp.asarray(a)

    return jax.tree.map(put, tree)


def split_hi_lo(w):
    """f32 -> (hi = bf16(w), lo = bf16(w - f32(hi))) numpy bf16 pair (`hi + lo ~= w`)."""
    w = np.asarray(w, np.float32)
    hi = w.astype(NP_BF16)
    lo = (w - hi.astype(np.float32)).astype(NP_BF16)
    return hi, lo


def canonical_experts(cfg, gate_up_proj, down_proj):
    """Checkpoint experts `[e, 2I, Hm]` / `[e, Hm, I]` -> quantized canonical entries (no L axis).

    Accepts any leading expert count (stream a few experts at a time). Gate rows `0:I` and up
    rows `I:2I` of `gate_up_proj` become columns `0:I` / `I:2I` after the transpose.
    """
    gate_up = np.ascontiguousarray(np.asarray(gate_up_proj).transpose(0, 2, 1), np.float32)
    down = np.ascontiguousarray(np.asarray(down_proj).transpose(0, 2, 1), np.float32)
    gq, gs = quantize_int4(gate_up, cfg.group_size)
    dq, ds = quantize_int4(down, cfg.group_size)
    return {"gate_up_q": gq, "gate_up_s": gs, "down_q": dq, "down_s": ds}


def canonical_layer_from_checkpoint(cfg, layer, get, experts=True):
    """Canonical entries of one layer (no L axis) from checkpoint tensors.

    `get(name)` returns the numpy array of a checkpoint tensor (nn.Linear `[out, in]` layout,
    names as in `model.safetensors.index.json`). With `experts=False` the (huge) expert tensors
    are left to `canonical_experts`.
    """
    p = f"{CHECKPOINT_PREFIX}layers.{layer}."

    def eff(name):
        return np.asarray(eff_weight(jnp.asarray(get(p + name))).astype(BF16))

    def gates(name):
        alpha, beta = gate_coeffs(jnp.asarray(get(p + name)), cfg.gate_temperature)
        return np.asarray(alpha), np.asarray(beta)

    def t(name):
        return np.ascontiguousarray(np.asarray(get(p + name)).T)

    out = {
        "attn_norm": eff("input_layernorm.weight"),
        "ffn_norm": eff("pre_feedforward_layernorm.weight"),
        "post_ffn_norm": eff("post_feedforward_layernorm.weight"),
        "pre_expert_norm": eff("mlp.pre_expert_norm.weight"),
        "post_expert_norm": eff("mlp.experts.post_expert_norm.weight"),
        "router": t("mlp.gate.weight").astype(np.float32),
        "router_bias": np.asarray(get(p + "mlp.gate.e_score_correction_bias"), np.float32),
        "q": t("self_attn.q_proj.weight"),
        "k": t("self_attn.k_proj.weight"),
        "v": t("self_attn.v_proj.weight"),
        "gate": t("self_attn.gate_proj.weight"),
        "o": t("self_attn.o_proj.weight"),
        "pre": t("mlp.pre_expert_proj.weight"),
        "post": t("mlp.post_expert_proj.weight"),
    }
    out["attn_gate_alpha"], out["attn_gate_beta"] = gates("post_attention_residual_gate.gate")
    out["ffn_gate_alpha"], out["ffn_gate_beta"] = gates("post_feedforward_residual_gate.gate")
    if experts:
        out.update(
            canonical_experts(
                cfg, get(p + "mlp.experts.gate_up_proj"), get(p + "mlp.experts.down_proj")
            )
        )
    return out


def canonical_global_from_checkpoint(cfg, get):
    """`embed`, `lm_head`, `final_norm` from checkpoint tensors (see the module docstring)."""
    return {
        "embed": np.asarray(get(CHECKPOINT_PREFIX + "embed_tokens.weight")),
        "lm_head": np.ascontiguousarray(np.asarray(get("lm_head.weight")).T),
        "final_norm": np.asarray(get(CHECKPOINT_PREFIX + "norm.weight")),
    }


# ---- canonical <-> per-rank kernel layout -------------------------------------------------------
def _np(a):
    a = np.asarray(a)
    return a.astype(jnp.int4) if a.dtype == np.int8 else a


def shard_layer(cfg, lw, rank, tp=8):
    """One layer's canonical entries (no L axis, numpy) -> that rank's arrays (no L axis).

    Dense experts are quantized here if the dict holds `gate_up`/`down` instead of the
    quantized form. Output names/shapes are `layout.rank_shapes` without the leading L.
    """
    layout.check_tp(cfg, tp)
    H, Hm, I = cfg.hidden, cfg.moe_hidden, cfg.expert_hidden
    qw, kvw, isl = cfg.q_width // tp, cfg.kv_width // tp, I // tp
    r = rank
    if "gate_up_q" not in lw:
        gq, gs = quantize_int4(np.asarray(lw["gate_up"], np.float32), cfg.group_size)
        dq, ds = quantize_int4(np.asarray(lw["down"], np.float32), cfg.group_size)
        lw = {**lw, "gate_up_q": gq, "gate_up_s": gs, "down_q": dq, "down_s": ds}
    hi, lo = split_hi_lo(lw["router"])
    gate_up_q = np.asarray(lw["gate_up_q"])
    gate_up_s = np.asarray(lw["gate_up_s"], np.float32)
    gu_cols = np.r_[r * isl : (r + 1) * isl, I + r * isl : I + (r + 1) * isl]
    G = cfg.group_size
    out = {
        "attn_norm": np.asarray(lw["attn_norm"], NP_BF16)[None],
        "ffn_norm": np.asarray(lw["ffn_norm"], NP_BF16)[None],
        "post_ffn_norm": np.asarray(lw["post_ffn_norm"], NP_BF16)[None],
        "attn_gate_alpha": np.asarray(lw["attn_gate_alpha"], np.float32)[None],
        "attn_gate_beta": np.asarray(lw["attn_gate_beta"], np.float32)[None],
        "ffn_gate_alpha": np.asarray(lw["ffn_gate_alpha"], np.float32)[None],
        "ffn_gate_beta": np.asarray(lw["ffn_gate_beta"], np.float32)[None],
        "pre_expert_norm": np.asarray(lw["pre_expert_norm"], NP_BF16)[None],
        "post_expert_norm": np.asarray(lw["post_expert_norm"], NP_BF16)[None],
        "router_bias": np.asarray(lw["router_bias"], np.float32)[None],
        "q": np.asarray(lw["q"], NP_BF16)[:, r * qw : (r + 1) * qw],
        "kv": np.concatenate(
            (
                np.asarray(lw["k"], NP_BF16)[:, r * kvw : (r + 1) * kvw],
                np.asarray(lw["v"], NP_BF16)[:, r * kvw : (r + 1) * kvw],
            ),
            axis=1,
        ),
        "gate": np.asarray(lw["gate"], NP_BF16)[:, r * qw : (r + 1) * qw],
        "o": np.asarray(lw["o"], NP_BF16)[r * qw : (r + 1) * qw, :],
        "pre": np.asarray(lw["pre"], NP_BF16)[:, r * (Hm // tp) : (r + 1) * (Hm // tp)],
        "router_hi": hi,
        "router_lo": lo,
        "post": np.asarray(lw["post"], NP_BF16)[:, r * (H // tp) : (r + 1) * (H // tp)],
        "gate_up_q": _np(gate_up_q[:, :, gu_cols]),
        "gate_up_s": scales_to_chunked(gate_up_s[:, :, gu_cols], Hm),
        "down_q": _np(np.asarray(lw["down_q"])[:, r * isl : (r + 1) * isl, :]),
        "down_s": scales_to_chunked(
            np.asarray(lw["down_s"], np.float32)[:, r * (isl // G) : (r + 1) * (isl // G), :], isl
        ),
    }
    return {k: np.ascontiguousarray(v) for k, v in out.items()}


def shard_global(cfg, weights, rank, tp=8):
    """`embed [Vp, H]`, `lm_head [H, Vp]` (zero-padded vocab shard) and `final_norm [1, H]`."""
    vp = layout.vocab_pad(cfg, tp)
    lo, hi = rank * vp, min((rank + 1) * vp, cfg.vocab)
    n = max(hi - lo, 0)
    embed_ = np.zeros((vp, cfg.hidden), NP_BF16)
    lm = np.zeros((cfg.hidden, vp), NP_BF16)
    if n:
        embed_[:n] = np.asarray(weights["embed"], NP_BF16)[lo:hi]
        lm[:, :n] = np.asarray(weights["lm_head"], NP_BF16)[:, lo:hi]
    return {
        "embed": embed_,
        "lm_head": lm,
        "final_norm": np.asarray(weights["final_norm"], NP_BF16)[None],
    }


def shard_rank(cfg, weights, rank, tp=8):
    """Full canonical dict -> one rank's dict in `layout.rank_shapes` layout (numpy)."""
    out = shard_global(cfg, weights, rank, tp)
    per_layer = [
        shard_layer(cfg, layer_weights(weights, layer), rank, tp) for layer in range(cfg.layers)
    ]
    for name in per_layer[0]:
        out[name] = np.stack([lw[name] for lw in per_layer])
    shapes = layout.rank_shapes(cfg, tp)
    for name, (shape, dtype) in shapes.items():
        if out[name].shape != shape or out[name].dtype != np.dtype(dtype):
            raise ValueError(
                f"{name}: got {out[name].shape} {out[name].dtype}, want {shape} {dtype}"
            )
    return {name: out[name] for name in shapes}


def shard_canonical(cfg, weights, tp=8):
    """Canonical dict -> `{name: [tp, *rank_shape]}` numpy, exactly `layout.rank_shapes`."""
    ranks = [shard_rank(cfg, weights, rank, tp) for rank in range(tp)]
    return {name: np.stack([r[name] for r in ranks]) for name in ranks[0]}


def unshard(cfg, sharded, tp=8):
    """Inverse of `shard_canonical` (router = f32(hi) + f32(lo), experts stay quantized)."""
    s = {k: np.asarray(v) for k, v in sharded.items()}
    H = cfg.hidden
    isl = cfg.expert_hidden // tp
    kvw = cfg.kv_width // tp
    out = {
        "embed": s["embed"].reshape(-1, H)[: cfg.vocab],
        "lm_head": np.concatenate(list(s["lm_head"]), axis=1)[:, : cfg.vocab],
        "final_norm": s["final_norm"][0, 0],
    }
    for name in layout.VECTOR_FAMILIES:
        out[name] = s[name][0, :, 0]
    out["router"] = s["router_hi"][0].astype(np.float32) + s["router_lo"][0].astype(np.float32)
    out["q"] = np.concatenate(list(s["q"]), axis=2)
    out["gate"] = np.concatenate(list(s["gate"]), axis=2)
    out["k"] = np.concatenate([r[:, :, :kvw] for r in s["kv"]], axis=2)
    out["v"] = np.concatenate([r[:, :, kvw:] for r in s["kv"]], axis=2)
    out["o"] = np.concatenate(list(s["o"]), axis=1)
    out["pre"] = np.concatenate(list(s["pre"]), axis=2)
    out["post"] = np.concatenate(list(s["post"]), axis=2)
    gus = [scales_from_chunked(r) for r in s["gate_up_s"]]
    out["gate_up_q"] = np.concatenate(
        [r[..., :isl] for r in s["gate_up_q"]] + [r[..., isl:] for r in s["gate_up_q"]], axis=-1
    )
    out["gate_up_s"] = np.concatenate(
        [r[..., :isl] for r in gus] + [r[..., isl:] for r in gus], axis=-1
    )
    out["down_q"] = np.concatenate(list(s["down_q"]), axis=2)
    out["down_s"] = np.concatenate([scales_from_chunked(r) for r in s["down_s"]], axis=2)
    return {k: np.ascontiguousarray(v) for k, v in out.items()}


def shard_caches(cfg, caches, tp=8):
    """Canonical `(k, v) [L, B, S, nKV, D]` -> `{"k_cache", "v_cache": [tp, L, B, S, lanes]}`."""
    kvh = cfg.kv_heads // tp
    lanes = layout.cache_lanes(cfg, tp)
    out = {}
    for name, c in zip(("k_cache", "v_cache"), caches):
        c = np.asarray(c)
        L, B, S = c.shape[:3]
        r = c.reshape(L, B, S, tp, kvh * cfg.head_dim).transpose(3, 0, 1, 2, 4)
        out[name] = np.ascontiguousarray(
            np.pad(r, ((0, 0),) * 4 + ((0, lanes - kvh * cfg.head_dim),))
        )
    return out


def unshard_caches(cfg, sharded, tp=8):
    """Inverse of `shard_caches` -> canonical `(k, v)` numpy bf16."""
    kvh = cfg.kv_heads // tp
    out = []
    for name in ("k_cache", "v_cache"):
        c = np.asarray(sharded[name])[..., : kvh * cfg.head_dim]  # [tp, L, B, S, kvh*D]
        _, L, B, S, _ = c.shape
        out.append(
            np.ascontiguousarray(
                c.transpose(1, 2, 3, 0, 4).reshape(L, B, S, cfg.kv_heads, cfg.head_dim)
            )
        )
    return tuple(out)
