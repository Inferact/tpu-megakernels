"""Kimi K3 checkpoint configuration and component reference helpers.

Equations follow the Kimi K3 model and KDA reference. All matrix weights use
[input, output] layout. This reference does not call fused kernels.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Config:
    dim: int = 7168
    latent: int = 3584
    expert_hidden: int = 3072
    shared_hidden: int = 6144
    dense_hidden: int = 33792
    experts: int = 896
    top_k: int = 16
    layers: int = 93
    heads: int = 96
    head_dim: int = 128
    q_rank: int = 1536
    kv_rank: int = 512
    qk_extra: int = 64
    vocab: int = 163840
    conv_size: int = 4
    residual_block: int = 12
    eps: float = 1e-5
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0
    gate_lower_bound: float = -5.0
    full_attention: tuple[int, ...] = tuple(range(3, 92, 4)) + (92,)

    @classmethod
    def from_checkpoint(cls, directory):
        raw = json.loads((Path(directory) / "config.json").read_text())
        c = raw.get("text_config", raw)
        a = c["linear_attn_config"]
        required = {
            "mla_use_nope": True,
            "mla_use_output_gate": True,
            "latent_moe_use_norm": True,
            "moe_renormalize": True,
            "moe_router_activation_func": "sigmoid",
            "hidden_act": "situ",
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "num_expert_group": 1,
            "routed_scaling_factor": 1.0,
        }
        for name, expected in required.items():
            if c[name] != expected:
                raise ValueError(f"Unsupported K3 setting {name}={c[name]!r}")
        if not a["use_full_rank_gate"]:
            raise ValueError("K3 requires the full-rank output gate")
        return cls(
            dim=c["hidden_size"],
            latent=c["routed_expert_hidden_size"],
            expert_hidden=c["moe_intermediate_size"],
            shared_hidden=c["moe_intermediate_size"] * c["num_shared_experts"],
            dense_hidden=c["intermediate_size"],
            experts=c["num_experts"],
            top_k=c["num_experts_per_token"],
            layers=c["num_hidden_layers"],
            heads=c["num_attention_heads"],
            head_dim=a["head_dim"],
            q_rank=c["q_lora_rank"],
            kv_rank=c["kv_lora_rank"],
            qk_extra=c["qk_rope_head_dim"],
            vocab=c["vocab_size"],
            conv_size=a["short_conv_kernel_size"],
            residual_block=c["attn_res_block_size"],
            eps=c["rms_norm_eps"],
            situ_beta=c["activation_situ_beta"],
            situ_linear_beta=c["activation_situ_linear_beta"],
            gate_lower_bound=a["gate_lower_bound"],
            full_attention=tuple(i - 1 for i in a["full_attn_layers"]),
        )


def rms(x, weight, eps=1e-5):
    value = x.astype(jnp.float32)
    value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + eps)
    return (value * weight.astype(jnp.float32)).astype(x.dtype)


def sigmoid(x):
    """Match a BF16 sigmoid operator: compute in FP32, then round once."""
    return jax.nn.sigmoid(x.astype(jnp.float32)).astype(x.dtype)


def route(logits, bias, top_k):
    """Correction bias selects IDs; uncorrected sigmoid scores set their weights."""
    scores = jax.nn.sigmoid(logits.astype(jnp.float32))
    _, ids = jax.lax.top_k(scores + bias, top_k)
    weights = scores[ids]
    return ids, weights / jnp.sum(weights)


def attention_residual(prefix, blocks, folded_weight, eps=1e-5):
    values = jnp.concatenate((blocks, prefix[None]), axis=0).astype(jnp.float32)
    scores = jnp.sum(values * folded_weight, axis=-1)
    scores *= jax.lax.rsqrt(jnp.mean(values * values, axis=-1) + eps)
    mixed = jnp.sum(jax.nn.softmax(scores)[:, None] * values, axis=0)
    return jax.lax.optimization_barrier(mixed).astype(prefix.dtype)


def kda_step(qkv, raw_gate, beta, output_gate, conv, state, w, c):
    """One token, [3, heads, D] QKV; recurrent state uses [heads, K, V]."""
    history = jnp.concatenate((conv, qkv[None]), axis=0)
    filtered = jax.nn.silu(
        jnp.sum(history.astype(jnp.float32) * w["conv"].astype(jnp.float32), axis=0)
    ).astype(qkv.dtype)
    q, k, v = filtered.astype(jnp.float32)
    q *= jax.lax.rsqrt(jnp.sum(q * q, -1, keepdims=True) + 1e-6) * c.head_dim**-0.5
    k *= jax.lax.rsqrt(jnp.sum(k * k, -1, keepdims=True) + 1e-6)
    gate = c.gate_lower_bound * jax.nn.sigmoid(
        jnp.exp(w["a_log"])[:, None] * (raw_gate.astype(jnp.float32) + w["dt_bias"])
    )
    state = state.astype(jnp.float32) * jnp.exp(gate)[..., None]
    prediction = jnp.einsum("hk,hkv->hv", k, state, precision=jax.lax.Precision.HIGHEST)
    delta = jax.nn.sigmoid(beta.astype(jnp.float32))[:, None] * (v - prediction)
    state += k[..., None] * delta[:, None, :]
    out = jnp.einsum("hk,hkv->hv", q, state, precision=jax.lax.Precision.HIGHEST).astype(qkv.dtype)
    out = rms(out, w["out_norm"], c.eps) * sigmoid(output_gate)
    return out, history[1:], state


def unpack_mxfp4(packed, scales):
    """[K/8,N] u32 low-nibble-first storage, [K/32,N] E8M0 scales."""
    shifts = jnp.arange(8, dtype=jnp.uint32) * 4
    codes = ((packed[:, None, :] >> shifts[None, :, None]) & 15).reshape(-1, packed.shape[-1])
    magnitudes = jnp.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], jnp.float32)
    values = magnitudes[codes & 7] * jnp.where(codes & 8, -1, 1)
    scale = jax.lax.bitcast_convert_type(scales.astype(jnp.uint32) << 23, jnp.float32)
    return (values * jnp.repeat(scale, 32, axis=0)).astype(jnp.bfloat16)


def tp_sum(value, axis_name="tp"):
    """Keep the reference collective in FP32 before any BF16 output cast."""
    return jax.lax.optimization_barrier(jax.lax.psum(value, axis_name))
