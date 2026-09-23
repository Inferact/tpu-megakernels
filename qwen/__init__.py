"""Qwen3.8 configuration and reference model equations."""

from dataclasses import dataclass
import json
from pathlib import Path

import jax
import jax.numpy as jnp


F32 = jnp.float32
HIGHEST = jax.lax.Precision.HIGHEST


@dataclass(frozen=True)
class Config:
    dim: int = 5120
    intermediate: int = 17408
    layers: int = 64
    heads: int = 24
    kv_heads: int = 4
    head_dim: int = 256
    rotary_dim: int = 64
    rope_theta: float = 10000000.0
    k_heads: int = 16
    k_dim: int = 128
    v_heads: int = 48
    v_dim: int = 128
    conv_size: int = 4
    vocab: int = 248320
    eps: float = 1e-6
    full_attention: tuple[int, ...] = tuple(range(3, 64, 4))

    @classmethod
    def from_checkpoint(cls, directory):
        raw = json.loads((Path(directory) / "config.json").read_text())
        c = raw["text_config"]
        required = {
            "attention_bias": False,
            "attn_output_gate": True,
            "hidden_act": "silu",
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_dtype": "float32",
            "output_gate_type": "swish",
            "tie_word_embeddings": False,
        }
        for name, expected in required.items():
            if c[name] != expected:
                raise ValueError(f"Unsupported Qwen3.8 setting {name}={c[name]!r}")
        rope = c["rope_parameters"]
        if rope["rope_type"] != "default" or not rope["mrope_interleaved"]:
            raise ValueError(f"Unsupported Qwen3.8 rope {rope!r}")
        expected_types = tuple(
            "full_attention" if i % 4 == 3 else "linear_attention" for i in range(c["num_hidden_layers"])
        )
        if tuple(c["layer_types"]) != expected_types:
            raise ValueError("Qwen3.8 layer pattern must be LLLF repeating")
        return cls(
            dim=c["hidden_size"],
            intermediate=c["intermediate_size"],
            layers=c["num_hidden_layers"],
            heads=c["num_attention_heads"],
            kv_heads=c["num_key_value_heads"],
            head_dim=c["head_dim"],
            rotary_dim=int(c["head_dim"] * rope["partial_rotary_factor"]),
            rope_theta=rope["rope_theta"],
            k_heads=c["linear_num_key_heads"],
            k_dim=c["linear_key_head_dim"],
            v_heads=c["linear_num_value_heads"],
            v_dim=c["linear_value_head_dim"],
            conv_size=c["linear_conv_kernel_dim"],
            vocab=c["vocab_size"],
            eps=c["rms_norm_eps"],
            full_attention=tuple(i for i in range(c["num_hidden_layers"]) if i % 4 == 3),
        )

    @property
    def key_dim(self):
        return self.k_heads * self.k_dim

    @property
    def value_dim(self):
        return self.v_heads * self.v_dim

    @property
    def conv_dim(self):
        return self.key_dim * 2 + self.value_dim


def linear(x, weight):
    return jnp.dot(x, weight, preferred_element_type=F32).astype(x.dtype)


def rms(x, weight, eps):
    """Zero-centered RMSNorm: FP32 normalize, scale by (1 + w), cast back."""
    value = x.astype(F32)
    value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + eps)
    return (value * (1.0 + weight.astype(F32))).astype(x.dtype)


def rms_gated(x, gate, weight, eps):
    """Qwen3.8 RMSNorm-Gated: normalize in FP32, scale in BF16, gate in FP32."""
    value = x.astype(F32)
    value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + eps)
    value = (weight * value.astype(x.dtype)).astype(F32)
    return (value * jax.nn.silu(gate.astype(F32))).astype(x.dtype)


def silu(x):
    return jax.nn.silu(x.astype(F32)).astype(x.dtype)


def sigmoid(x):
    """Match a BF16 sigmoid operator: compute in FP32, then round once."""
    return jax.nn.sigmoid(x.astype(F32)).astype(x.dtype)


def l2norm(x, eps=1e-6):
    value = x.astype(F32)
    return value * jax.lax.rsqrt(jnp.sum(value * value, axis=-1, keepdims=True) + eps)


def rope_cos_sin(c, position):
    """Text decode collapses mrope to standard partial RoPE (all grids equal)."""
    half = c.rotary_dim // 2
    inv = 1.0 / (c.rope_theta ** (jnp.arange(half, dtype=F32) * 2.0 / c.rotary_dim))
    angles = position.astype(F32) * inv
    return tuple(jnp.concatenate((f(angles), f(angles))).astype(jnp.bfloat16) for f in (jnp.cos, jnp.sin))


def apply_rope(x, cos, sin):
    """Split-half RoPE over the leading rotary_dim channels, in BF16."""
    rot, passthrough = x[..., : cos.shape[-1]], x[..., cos.shape[-1] :]
    first, second = rot[..., : rot.shape[-1] // 2], rot[..., rot.shape[-1] // 2 :]
    rotated = jnp.concatenate((-second, first), axis=-1)
    return jnp.concatenate((rot * cos + rotated * sin, passthrough), axis=-1)


def mlp(x, w):
    return linear(silu(linear(x, w["gate"])) * linear(x, w["up"]), w["down"])


def delta_step(x, w, state, c):
    """One gated-deltanet token; state = (conv [conv_dim, 4] BF16, S [v_heads, V, K] FP32).

    The recurrent state is kept transposed ([value, key]) so both contractions
    reduce over the minor axis.
    """
    conv, rec = state
    qkv = linear(x, w["qkv"])
    z = linear(x, w["z"])
    b = linear(x, w["b"])
    a = linear(x, w["a"])
    history = jnp.concatenate((conv, qkv[:, None]), axis=1)
    filtered = silu(
        jnp.sum(history[:, 1:].astype(F32) * w["conv"].astype(F32), axis=-1)
    )
    q, k, v = jnp.split(filtered, (c.key_dim, c.key_dim * 2))
    q = l2norm(q.reshape(c.k_heads, c.k_dim)) * c.k_dim**-0.5
    k = l2norm(k.reshape(c.k_heads, c.k_dim))
    v = v.reshape(c.v_heads, c.v_dim).astype(F32)
    expansion = c.v_heads // c.k_heads
    q = jnp.repeat(q, expansion, axis=0)
    k = jnp.repeat(k, expansion, axis=0)
    beta = sigmoid(b).astype(F32)
    gate = -jnp.exp(w["a_log"].astype(F32)) * jax.nn.softplus(
        a.astype(F32) + w["dt_bias"].astype(F32)
    )
    rec = rec * jnp.exp(gate)[:, None, None]
    prediction = jnp.einsum("hk,hvk->hv", k, rec, precision=HIGHEST)
    rec = rec + ((v - prediction) * beta[:, None])[..., None] * k[:, None, :]
    out = jnp.einsum("hk,hvk->hv", q, rec, precision=HIGHEST).astype(x.dtype)
    out = rms_gated(out, z.reshape(c.v_heads, c.v_dim), w["norm"], c.eps)
    return linear(out.reshape(-1), w["out"]), (history[:, 1:], rec)


def attention_step(x, w, cache, position, c):
    """One GQA token; cache = (keys, values) [kv_heads, context, head_dim] BF16."""
    query, gate = jnp.split(linear(x, w["q"]).reshape(c.heads, 2 * c.head_dim), 2, axis=-1)
    query = rms(query, w["q_norm"], c.eps)
    key = rms(linear(x, w["k"]).reshape(c.kv_heads, c.head_dim), w["k_norm"], c.eps)
    value = linear(x, w["v"]).reshape(c.kv_heads, c.head_dim)
    cos, sin = rope_cos_sin(c, position)
    query = apply_rope(query, cos, sin)
    key = apply_rope(key, cos, sin)
    keys, values = cache
    keys, values = keys.at[:, position].set(key), values.at[:, position].set(value)
    grouped = query.reshape(c.kv_heads, c.heads // c.kv_heads, c.head_dim)
    scores = jnp.einsum("gjd,gtd->gjt", grouped.astype(F32), keys.astype(F32), precision=HIGHEST)
    scores *= c.head_dim**-0.5
    scores = jnp.where(jnp.arange(keys.shape[1]) <= position, scores, -jnp.inf)
    probs = jax.nn.softmax(scores, axis=-1).astype(x.dtype)
    out = jnp.einsum("gjt,gtd->gjd", probs.astype(F32), values.astype(F32), precision=HIGHEST)
    out = out.reshape(c.heads, c.head_dim).astype(x.dtype) * sigmoid(gate)
    return linear(out.reshape(-1), w["o"]), (keys, values)


def initial_states(c, context, dtype=jnp.bfloat16):
    states = []
    for index in range(c.layers):
        if index in c.full_attention:
            states.append(
                (
                    jnp.zeros((c.kv_heads, context, c.head_dim), dtype),
                    jnp.zeros((c.kv_heads, context, c.head_dim), dtype),
                )
            )
        else:
            states.append(
                (
                    jnp.zeros((c.conv_dim, c.conv_size), dtype),
                    jnp.zeros((c.v_heads, c.v_dim, c.k_dim), F32),
                )
            )
    return tuple(states)


def forward(token, weights, states, position, c, return_hidden=False):
    """Full 64-layer decoder oracle with explicit hybrid state and BF16 residuals."""
    hidden = weights["embedding"][token]
    new_states = []
    for index, (w, state) in enumerate(zip(weights["layers"], states, strict=True)):
        mixed = rms(hidden, w["input_norm"], c.eps)
        if index in c.full_attention:
            output, next_state = attention_step(mixed, w["attention"], state, position, c)
        else:
            output, next_state = delta_step(mixed, w["linear"], state, c)
        hidden = hidden + output
        hidden = hidden + mlp(rms(hidden, w["post_norm"], c.eps), w["mlp"])
        new_states.append(next_state)
    if len(new_states) != c.layers:
        raise ValueError("The full forward requires every configured layer")
    hidden = rms(hidden, weights["norm"], c.eps)
    logits = jnp.dot(hidden, weights["lm_head"], preferred_element_type=F32)
    if return_hidden:
        return logits, tuple(new_states), hidden
    return logits, tuple(new_states)


def generate(weights, prompt, steps, c, context):
    """Greedy token-by-token generation; returns the produced token ids."""
    states = initial_states(c, context)
    tokens = []
    todo = list(prompt) + [None] * steps
    for position, token in enumerate(todo):
        if token is None:
            token = int(jnp.argmax(logits))
            tokens.append(token)
        logits, states = forward(token, weights, states, jnp.array(position, jnp.int32), c)
    return tokens


def linear_seq(x, w):
    """[L, in] @ [in, out] with the reference's fp32 accumulate + bf16 round."""
    return jnp.dot(x, w, preferred_element_type=F32).astype(x.dtype)


def causal_conv(x, w, conv_state):
    """[L, C] x [C, K] x conv_state [C, K] -> [L, C]: causal depthwise conv.

    The conv state holds the last K raw inputs including the previous token;
    the tap window for output t is history[t+1 : t+K+1].
    """
    L = x.shape[0]
    xp = jnp.concatenate((conv_state.T, x), axis=0)  # [K + L, C]
    acc = jnp.zeros(x.shape, F32)
    for tap in range(w.shape[1]):
        acc = acc + xp[tap + 1 : tap + 1 + L].astype(F32) * w[:, tap].astype(F32)
    return acc


def chunk_gated_delta_rule(q, k, v, gate, beta, initial, chunk_size=64):
    """Port of HF torch_chunk_gated_delta_rule (fp32 throughout).

    q, k: [L, kh, dk]; v: [L, vh, dv]; gate/beta: [L, vh]; initial state in the
    kernel's [vh, dv, dk] layout. Returns (out [L, vh, dv], state [vh, dv, dk]).
    """
    q = jnp.transpose(q, (1, 0, 2)).astype(F32)  # [kh, L, dk]
    k = jnp.transpose(k, (1, 0, 2)).astype(F32)
    v = jnp.transpose(v, (1, 0, 2)).astype(F32)  # [vh, L, dv]
    beta = jnp.transpose(beta, (1, 0)).astype(F32)  # [vh, L]
    decay = jnp.transpose(gate, (1, 0)).astype(F32)  # [vh, L]
    initial = initial.astype(F32).transpose(0, 2, 1)  # -> [vh, dk, dv]

    q = q * jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    q = q * q.shape[-1] ** -0.5
    k = k * jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)

    L = q.shape[1]
    C = chunk_size
    pad = (-L) % C
    q = jnp.pad(q, ((0, 0), (0, pad), (0, 0)))
    k = jnp.pad(k, ((0, 0), (0, pad), (0, 0)))
    v = jnp.pad(v, ((0, 0), (0, pad), (0, 0)))
    beta = jnp.pad(beta, ((0, 0), (0, pad)))
    decay = jnp.pad(decay, ((0, 0), (0, pad)))

    v_beta = v * beta[..., None]
    k_beta = k * beta[..., None]

    nc = (L + pad) // C
    q = q.reshape(q.shape[0], nc, C, q.shape[-1])
    k = k.reshape(k.shape[0], nc, C, k.shape[-1])
    k_beta = k_beta.reshape(k_beta.shape[0], nc, C, k_beta.shape[-1])
    v_beta = v_beta.reshape(v_beta.shape[0], nc, C, v_beta.shape[-1])
    decay = decay.reshape(decay.shape[0], nc, C)

    strictly_upper = jnp.triu(jnp.ones((C, C), bool), 1)
    cum_decay = jnp.cumsum(decay, axis=2)  # [vh, nc, C]
    pairwise = cum_decay[..., :, None] - cum_decay[..., None, :]
    pairwise = jnp.exp(jnp.where(strictly_upper, -jnp.inf, pairwise))

    ut_system = jnp.matmul(k_beta, k.transpose(0, 1, 3, 2)) * pairwise
    intra_attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * pairwise
    decayed_k_beta = k_beta * jnp.exp(cum_decay)[..., None]

    solve = jax.vmap(
        jax.vmap(
            lambda A, B: jax.lax.linalg.triangular_solve(
                A, B, left_side=True, lower=True, unit_diagonal=True
            )
        )
    )
    new_values = solve(ut_system, v_beta)
    k_cumdecay = solve(ut_system, decayed_k_beta)

    q = q * jnp.exp(cum_decay)[..., None]
    k = k * jnp.exp(cum_decay[..., -1:] - cum_decay)[..., None]
    chunk_decay = jnp.exp(cum_decay[..., -1])[..., None, None]

    state = initial
    outs = []
    for i in range(nc):
        v_new = new_values[:, i] - k_cumdecay[:, i] @ state
        inter = q[:, i] @ state
        outs.append(inter + intra_attn[:, i] @ v_new)
        state = state * chunk_decay[:, i] + k[:, i].transpose(0, 2, 1) @ v_new

    out = jnp.concatenate(outs, axis=1)[:, :L]  # [vh, L, dv]
    out = jnp.transpose(out, (1, 0, 2))  # [L, vh, dv]
    return out, state.transpose(0, 2, 1)  # back to [vh, dv, dk]


def linear_prefill(x, w, state, c, npad):
    """Batched gated-deltanet layer. x [L, dim] -> (out [L, dim], (conv, rec))."""
    conv, rec = state
    qkv_raw = linear_seq(x, w["qkv"])  # [L, conv_dim]
    qkv = jax.nn.silu(causal_conv(qkv_raw, w["conv"], conv).astype(x.dtype))
    z = linear_seq(x, w["z"])
    b = linear_seq(x, w["b"])
    a = linear_seq(x, w["a"])
    L = x.shape[0]
    q, k, v = jnp.split(qkv, (c.key_dim, c.key_dim * 2), axis=-1)
    q = q.reshape(L, c.k_heads, c.k_dim)
    k = k.reshape(L, c.k_heads, c.k_dim)
    v = v.reshape(L, c.v_heads, c.v_dim)
    real = jnp.arange(L)[:, None] >= npad  # [L, 1]: False on left-pad rows
    beta = sigmoid(jnp.where(real, b, -1e30)).astype(F32)
    a = jnp.where(real, a, -1e30)
    gate = -jnp.exp(w["a_log"].astype(F32)) * jax.nn.softplus(
        a.astype(F32) + w["dt_bias"].astype(F32)
    )
    expand = c.v_heads // c.k_heads
    q = jnp.repeat(q, expand, axis=1)
    k = jnp.repeat(k, expand, axis=1)
    out, new_rec = chunk_gated_delta_rule(q, k, v, gate, beta, rec)
    out = rms_gated(out, z.reshape(L, c.v_heads, c.v_dim), w["norm"], c.eps)
    out = linear_seq(out.reshape(L, -1), w["out"])
    combined = jnp.concatenate((conv.T, qkv_raw), axis=0)
    new_conv = combined[-c.conv_size :].T
    return out, (new_conv, new_rec)


def rope_tables(c, positions):
    """[P] positions -> (cos, sin) [P, rotary_dim] bf16 (text = standard RoPE)."""
    half = c.rotary_dim // 2
    inv = 1.0 / (c.rope_theta ** (jnp.arange(half, dtype=F32) * 2.0 / c.rotary_dim))
    angles = positions.astype(F32)[:, None] * inv
    return tuple(
        jnp.concatenate((f(angles), f(angles)), axis=-1).astype(jnp.bfloat16)
        for f in (jnp.cos, jnp.sin)
    )


def apply_rope_batched(vec, cos, sin):
    """vec [L, h, d]; cos/sin [L, rotary]; split-half rotate on the leading part."""
    half = cos.shape[-1] // 2
    rot, rest = vec[..., : cos.shape[-1]], vec[..., cos.shape[-1] :]
    a, b = rot[..., :half], rot[..., half:]
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    out = jnp.concatenate((a * cos[..., :half] - b * sin[..., :half],
                           b * cos[..., half:] + a * sin[..., half:],), axis=-1)
    # cos/sin carry duplicated halves; first half uses cv[:half], second cv[half:]
    return jnp.concatenate((out[..., :half], out[..., half:], rest), axis=-1).astype(vec.dtype)


def attention_prefill(x, w, npad, c):
    """Batched causal GQA over the B padded rows; pad rows never serve as keys.

    Rope positions are absolute (padded index - npad). The cache write happens
    in the caller (host-side shift of the real tail into slots [0, L_real)).
    """
    L = x.shape[0]
    qg = linear_seq(x, w["q"]).reshape(L, c.heads, 2 * c.head_dim)
    query, gate = qg[..., : c.head_dim], qg[..., c.head_dim :]
    query = rms(query, w["q_norm"], c.eps)
    key = rms(linear_seq(x, w["k"]).reshape(L, c.kv_heads, c.head_dim), w["k_norm"], c.eps)
    value = linear_seq(x, w["v"]).reshape(L, c.kv_heads, c.head_dim)

    positions = jnp.arange(L, dtype=jnp.int32) - npad
    cos, sin = rope_tables(c, positions)
    query = apply_rope_batched(query, cos, sin)
    key = apply_rope_batched(key, cos, sin)

    grouped = query.transpose(1, 0, 2).reshape(
        c.kv_heads, c.heads // c.kv_heads, L, c.head_dim
    )
    keys_t = key.transpose(1, 0, 2)  # [kv, L, hd] — the prompt's own keys
    scores = jnp.einsum("gjld,gtd->gjlt", grouped.astype(F32), keys_t.astype(F32))
    scores *= c.head_dim**-0.5
    real = jnp.arange(L) >= npad
    allowed = (jnp.arange(L)[None, :] <= (jnp.arange(L) - npad)[:, None]) & real[None, :]
    scores = jnp.where(allowed[None, None], scores, -jnp.inf)
    # pad rows must not become NaN (never read, but keep them finite anyway)
    scores = jnp.where((jnp.arange(L) >= npad)[None, None, :, None], scores, 0.0)
    probs = jax.nn.softmax(scores, axis=-1).astype(x.dtype)
    out = jnp.einsum("gjlt,gtd->gjld", probs.astype(F32), value.transpose(1, 0, 2).astype(F32))
    out = out.reshape(c.heads, L, c.head_dim).transpose(1, 0, 2)
    out = out * jax.nn.sigmoid(gate.astype(F32)).astype(x.dtype)
    return linear_seq(out.reshape(L, -1), w["o"]), (key, value)


def prefill(weights, states, tokens, c, npad=0):
    """Run the padded prompt through all layers; returns (new_states, last logits).

    `tokens` is the bucket-padded id array [B] with `npad` left-pad rows; pad rows
    are exact no-ops (masked hidden, gate/beta forced to no-op). Full-attention
    layers return their cache in the PADDED slot layout (pads at [0, npad));
    the caller shifts the real tail to slots [0, L_real) host-side.
    `states` holds prior context (zeros for a fresh conversation).
    """
    L = len(tokens)
    real = jnp.arange(L) >= npad
    hidden = weights["embedding"][jnp.asarray(tokens)]  # [L, dim] bf16
    hidden = jnp.where(real[:, None], hidden, jnp.zeros((), hidden.dtype))
    new_states = []
    new_caches = {}
    for index, (w, state) in enumerate(zip(weights["layers"], states, strict=True)):
        hidden = jnp.where(real[:, None], hidden, jnp.zeros((), hidden.dtype))
        mixed = rms(hidden, w["input_norm"], c.eps)
        if index in c.full_attention:
            output, (key, value) = attention_prefill(mixed, w["attention"], npad, c)
            kc, vc = state
            kc = kc.at[:, :L].set(key.transpose(1, 0, 2).astype(kc.dtype))
            vc = vc.at[:, :L].set(value.transpose(1, 0, 2).astype(vc.dtype))
            next_state = (kc, vc)
        else:
            output, next_state = linear_prefill(mixed, w["linear"], state, c, npad)
        hidden = hidden + output
        hidden = hidden + mlp(rms(hidden, w["post_norm"], c.eps), w["mlp"])
        new_states.append(next_state)
    last = rms(hidden[-1], weights["norm"], c.eps)
    logits = jnp.dot(last, weights["lm_head"], preferred_element_type=F32)
    return tuple(new_states), logits


def delta_rule_block(q, k, v, gate, beta, initial):
    """Sequential gated delta rule over L positions, snapshotting the state.

    Matches delta_step per token. q, k: [L, vh, kd] (unnormalized — l2norm
    is applied here); v: [L, vh, vd]; gate, beta: [L, vh]; initial state
    [vh, vd, kd] fp32. Returns (out [L, vh, vd] fp32, states [L, vh, vd, kd]).
    """
    L = q.shape[0]
    vh, kd = q.shape[1], q.shape[2]
    q = q.astype(F32).reshape(L, vh, kd)
    q = q * jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
    q = q * kd**-0.5
    k = k.astype(F32)
    k = k * jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)

    def step(rec, xs):
        g, b, kk, vv, qq = xs
        rec = rec * jnp.exp(g)[:, None, None]
        pred = jnp.einsum("hk,hvk->hv", kk, rec)
        rec = rec + ((vv - pred) * b[:, None])[..., None] * kk[:, None, :]
        out = jnp.einsum("hk,hvk->hv", qq, rec)
        return rec, (out, rec)

    _, (outs, recs) = jax.lax.scan(
        step, initial.astype(F32), (gate, beta, k, v.astype(F32), q)
    )
    return outs, recs
