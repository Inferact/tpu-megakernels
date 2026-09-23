"""DSpark draft model for Kimi K3 (``RedHatAI/Kimi-K3-speculator.dspark``).

The draft is a DFlash backbone: a small Qwen3-style transformer whose keys
and values come from two sources, projected target hidden states (the
context, one row per token already run through the target) and the query
block itself (an anchor token followed by mask tokens). Every block slot
predicts the token one position after itself. DSpark adds a low-rank Markov
bias, ``markov_w2 @ markov_w1[previous token]``, applied while sampling the
block left to right.

Two implementations share one weight naming scheme (the checkpoint's):

* ``reference_*``: unsharded JAX on a plain dict, used for CPU tests against
  the PyTorch definitions and as the oracle for the sharded version;
* ``make_context_update`` / ``make_draft_step``: TP32 ``shard_map`` programs
  on the megakernel mesh. Weights lead with the rank, like the megakernel's.

The draft context cache is a ring over ``window`` positions per layer; slot
``position % window`` stores that position's key/value and ``cache_positions``
records which position a slot holds (``-1`` when empty). Rows written for
positions that later turn out to be rejected drafts are hidden by the
``position < anchor_position`` mask until the real token overwrites them.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

import collectives32
from .decode_megakernel import _attention_scores, _dot


@dataclasses.dataclass(frozen=True)
class DraftConfig:
    hidden: int = 7168
    layers: int = 5
    heads: int = 96
    kv_heads: int = 16
    head_dim: int = 64
    intermediate: int = 14336
    vocab: int = 163840
    aux_layers: tuple[int, ...] = (24, 48, 72, 88, 92)
    mask_token: int = 163837
    window: int = 2048
    rope_theta: float = 10000.0
    eps: float = 1e-5
    markov_rank: int = 256
    # Candidates per vocabulary shard kept for the sequential Markov step. The
    # biased argmax is searched among the top ``markov_top_m`` base logits of
    # every shard (all-gathered once) instead of the full vocabulary. Zero
    # means exact full-vocabulary sampling with a collective per slot.
    markov_top_m: int = 16

    @property
    def q_width(self) -> int:
        return self.heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def context_width(self) -> int:
        return len(self.aux_layers) * self.hidden

    @classmethod
    def from_checkpoint(cls, directory: Path) -> "DraftConfig":
        raw = json.loads((Path(directory) / "config.json").read_text())
        layer = raw["transformer_layer_config"]
        return cls(
            hidden=layer["hidden_size"],
            layers=layer["num_hidden_layers"],
            heads=layer["num_attention_heads"],
            kv_heads=layer["num_key_value_heads"],
            head_dim=layer["head_dim"],
            intermediate=layer["intermediate_size"],
            vocab=layer["vocab_size"],
            aux_layers=tuple(raw["aux_hidden_state_layer_ids"]),
            mask_token=raw["mask_token_id"],
            window=layer["sliding_window"],
            rope_theta=float(layer["rope_parameters"]["rope_theta"]),
            eps=layer["rms_norm_eps"],
            markov_rank=raw["markov_rank"],
        )


KIMI_K3_DRAFT = DraftConfig()

# Checkpoint tensor names (PyTorch ``[out, in]`` layouts) and their shapes.
def weight_shapes(c: DraftConfig) -> dict[str, tuple[int, ...]]:
    shapes = {
        "embed_tokens.weight": (c.vocab, c.hidden),
        "lm_head.weight": (c.vocab, c.hidden),
        "fc.weight": (c.hidden, c.context_width),
        "hidden_norm.weight": (c.hidden,),
        "norm.weight": (c.hidden,),
        "markov_head.markov_w1.weight": (c.vocab, c.markov_rank),
        "markov_head.markov_w2.weight": (c.vocab, c.markov_rank),
    }
    for layer in range(c.layers):
        prefix = f"layers.{layer}."
        shapes.update(
            {
                prefix + "input_layernorm.weight": (c.hidden,),
                prefix + "post_attention_layernorm.weight": (c.hidden,),
                prefix + "self_attn.q_proj.weight": (c.q_width, c.hidden),
                prefix + "self_attn.k_proj.weight": (c.kv_width, c.hidden),
                prefix + "self_attn.v_proj.weight": (c.kv_width, c.hidden),
                prefix + "self_attn.o_proj.weight": (c.hidden, c.q_width),
                prefix + "self_attn.q_norm.weight": (c.head_dim,),
                prefix + "self_attn.k_norm.weight": (c.head_dim,),
                prefix + "mlp.gate_proj.weight": (c.intermediate, c.hidden),
                prefix + "mlp.up_proj.weight": (c.intermediate, c.hidden),
                prefix + "mlp.down_proj.weight": (c.hidden, c.intermediate),
            }
        )
    return shapes


def random_weights(c: DraftConfig, seed: int = 0) -> dict[str, np.ndarray]:
    """Deterministic bf16 weights with the checkpoint's names and shapes."""
    rng = np.random.default_rng(seed)
    weights = {}
    for name, shape in weight_shapes(c).items():
        if name.endswith("norm.weight"):
            value = 1 + 0.05 * rng.standard_normal(shape)
        elif "markov" in name:
            value = 0.01 * rng.standard_normal(shape)
        else:
            fan_in = shape[-1] if len(shape) == 2 else shape[0]
            value = rng.standard_normal(shape) / np.sqrt(fan_in)
        weights[name] = jnp.asarray(value, jnp.float32).astype(jnp.bfloat16)
    return weights


# --------------------------------------------------------------------------
# Shared arithmetic (mirrors transformers' Qwen3 bf16 modules)


def rms_norm(x, weight, eps):
    """Qwen3RMSNorm: FP32 statistics, cast back, then scale by the weight."""
    dtype = x.dtype
    x32 = x.astype(jnp.float32)
    normed = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return weight * normed.astype(dtype)


def linear(x, weight_out_in):
    """``x @ W^T`` with FP32 accumulation, rounded to the activation dtype."""
    return jnp.einsum(
        "...i,oi->...o", x, weight_out_in, preferred_element_type=jnp.float32
    ).astype(x.dtype)


def rope_cos_sin(positions, head_dim, theta, dtype):
    inv_freq = 1.0 / theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    freqs = positions.astype(jnp.float32)[:, None] * jnp.asarray(inv_freq)[None, :]
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def apply_rope(x, cos, sin):
    """``x`` is ``[tokens, heads, head_dim]``; cos/sin are ``[tokens, head_dim]``."""
    half = x.shape[-1] // 2
    rotated = jnp.concatenate((-x[..., half:], x[..., :half]), axis=-1)
    return x * cos[:, None, :] + rotated * sin[:, None, :]


def silu(x):
    x32 = x.astype(jnp.float32)
    return (x32 * jax.nn.sigmoid(x32)).astype(x.dtype)


# --------------------------------------------------------------------------
# Unsharded reference


def reference_context_features(w, c: DraftConfig, aux):
    """``aux`` is ``[tokens, len(aux_layers) * hidden]`` bf16 -> ``[tokens, hidden]``."""
    return rms_norm(linear(aux, w["fc.weight"]), w["hidden_norm.weight"], c.eps)


def reference_context_kv(w, c: DraftConfig, context, positions):
    """Per-layer normalized, rotated keys and values of context rows.

    Returns ``keys, values`` shaped ``[layers, tokens, kv_heads, head_dim]``.
    """
    cos, sin = rope_cos_sin(positions, c.head_dim, c.rope_theta, context.dtype)
    keys, values = [], []
    for layer in range(c.layers):
        prefix = f"layers.{layer}.self_attn."
        k = linear(context, w[prefix + "k_proj.weight"]).reshape(-1, c.kv_heads, c.head_dim)
        k = rms_norm(k, w[prefix + "k_norm.weight"], c.eps)
        keys.append(apply_rope(k, cos, sin))
        values.append(linear(context, w[prefix + "v_proj.weight"]).reshape(-1, c.kv_heads, c.head_dim))
    return jnp.stack(keys), jnp.stack(values)


def _attend(q, k, v, allowed, groups):
    """Eager GQA attention: ``q [Q, H, d]``, ``k/v [K, Hkv, d]``, ``allowed [Q, K]``."""
    k = jnp.repeat(k, groups, axis=1)
    v = jnp.repeat(v, groups, axis=1)
    scores = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
    scores = scores * q.shape[-1] ** -0.5
    scores = jnp.where(allowed[None], scores, -jnp.inf)
    probabilities = jax.nn.softmax(scores, axis=-1).astype(q.dtype)
    return jnp.einsum("hqk,khd->qhd", probabilities, v, preferred_element_type=jnp.float32).astype(q.dtype)


def reference_block_hidden(
    w,
    c: DraftConfig,
    context_keys,  # [layers, tokens, kv_heads, head_dim]
    context_values,
    context_positions,  # [tokens] int32, -1 for empty slots
    anchor_token,  # scalar int32
    anchor_position,  # scalar int32
    block_size: int,
):
    """Run the block of ``anchor + (block_size - 1)`` mask tokens; return the
    final-normed hidden rows ``[block_size, hidden]``."""
    tokens = jnp.concatenate(
        (jnp.asarray([anchor_token], jnp.int32), jnp.full((block_size - 1,), c.mask_token, jnp.int32))
    )
    x = w["embed_tokens.weight"][tokens]
    block_positions = anchor_position + jnp.arange(block_size, dtype=jnp.int32)
    cos, sin = rope_cos_sin(block_positions, c.head_dim, c.rope_theta, x.dtype)
    context_allowed = (
        (context_positions >= 0)
        & (context_positions < anchor_position)
        & (context_positions >= anchor_position - c.window)
    )
    block_allowed = jnp.arange(block_size)[None, :] <= jnp.arange(block_size)[:, None]
    allowed = jnp.concatenate(
        (jnp.broadcast_to(context_allowed[None], (block_size, context_positions.shape[0])), block_allowed),
        axis=1,
    )
    groups = c.heads // c.kv_heads
    for layer in range(c.layers):
        prefix = f"layers.{layer}."
        h = rms_norm(x, w[prefix + "input_layernorm.weight"], c.eps)
        q = linear(h, w[prefix + "self_attn.q_proj.weight"]).reshape(block_size, c.heads, c.head_dim)
        q = apply_rope(rms_norm(q, w[prefix + "self_attn.q_norm.weight"], c.eps), cos, sin)
        k = linear(h, w[prefix + "self_attn.k_proj.weight"]).reshape(block_size, c.kv_heads, c.head_dim)
        k = apply_rope(rms_norm(k, w[prefix + "self_attn.k_norm.weight"], c.eps), cos, sin)
        v = linear(h, w[prefix + "self_attn.v_proj.weight"]).reshape(block_size, c.kv_heads, c.head_dim)
        keys = jnp.concatenate((context_keys[layer], k), axis=0)
        values = jnp.concatenate((context_values[layer], v), axis=0)
        attended = _attend(q, keys, values, allowed, groups).reshape(block_size, c.q_width)
        x = x + linear(attended, w[prefix + "self_attn.o_proj.weight"])
        h = rms_norm(x, w[prefix + "post_attention_layernorm.weight"], c.eps)
        gate = linear(h, w[prefix + "mlp.gate_proj.weight"])
        up = linear(h, w[prefix + "mlp.up_proj.weight"])
        x = x + linear(silu(gate) * up, w[prefix + "mlp.down_proj.weight"])
    return rms_norm(x, w["norm.weight"], c.eps)


def reference_markov_greedy(w, c: DraftConfig, hidden, anchor_token):
    """Left-to-right greedy sampling with the Markov bias. Returns
    ``[block_size]`` draft tokens where slot ``k`` predicts position ``p+k+1``."""
    base = linear(hidden, w["lm_head.weight"]).astype(jnp.float32)
    drafts = []
    previous = anchor_token
    for slot in range(hidden.shape[0]):
        embedding = w["markov_head.markov_w1.weight"][previous]
        bias = linear(embedding[None], w["markov_head.markov_w2.weight"])[0].astype(jnp.float32)
        previous = jnp.argmax(base[slot] + bias).astype(jnp.int32)
        drafts.append(previous)
    return jnp.stack(drafts)


# --------------------------------------------------------------------------
# TP32 sharded programs


def _shard_weights(c: DraftConfig, weights, ranks=32):
    """Split checkpoint-named arrays into ``[ranks, ...]`` per-rank shards (host side).

    Layout: q heads and the matching kv head by rank (``heads // ranks`` q
    heads per rank; kv head ``rank // (ranks // kv_heads)``), MLP channels
    and ``fc`` input features by rank, vocabulary rows by rank for the
    embedding, LM head and both Markov matrices, small norms replicated.
    """
    if c.heads % ranks or ranks % c.kv_heads or c.intermediate % ranks:
        raise ValueError("Draft dimensions must divide across the ranks")
    if c.context_width % ranks or c.vocab % ranks:
        raise ValueError("Context width and vocabulary must divide across the ranks")
    q_per_rank = c.heads // ranks
    kv_share = ranks // c.kv_heads
    shards: dict[str, Any] = {}

    def by_rows(value, rows_per_rank):
        return value.reshape(ranks, rows_per_rank, *value.shape[1:])

    def replicate(value):
        return np.broadcast_to(value[None], (ranks, *value.shape))

    for name, value in weights.items():
        value = np.asarray(value)
        if name == "markov_head.markov_w1.weight":
            shards[name] = replicate(value)  # local previous-token lookup
        elif name in ("embed_tokens.weight", "lm_head.weight") or "markov" in name:
            shards[name] = by_rows(value, c.vocab // ranks)
        elif name == "fc.weight":
            shards[name] = value.reshape(c.hidden, ranks, c.context_width // ranks).transpose(1, 0, 2)
        elif name.endswith("q_proj.weight"):
            shards[name] = by_rows(value, q_per_rank * c.head_dim)
        elif name.endswith("k_proj.weight") or name.endswith("v_proj.weight"):
            per_head = value.reshape(c.kv_heads, c.head_dim, c.hidden)
            shards[name] = np.repeat(per_head, kv_share, axis=0)
        elif name.endswith("o_proj.weight"):
            shards[name] = value.reshape(c.hidden, ranks, q_per_rank * c.head_dim).transpose(1, 0, 2)
        elif name.endswith("gate_proj.weight") or name.endswith("up_proj.weight"):
            shards[name] = by_rows(value, c.intermediate // ranks)
        elif name.endswith("down_proj.weight"):
            shards[name] = value.reshape(c.hidden, ranks, c.intermediate // ranks).transpose(1, 0, 2)
        else:
            shards[name] = replicate(value)
    return shards


def local_weight_shapes(c: DraftConfig, ranks=32) -> dict[str, tuple[int, ...]]:
    """Per-rank shard shapes produced by ``_shard_weights``."""
    q_width = c.heads // ranks * c.head_dim
    shapes = {}
    for name, shape in weight_shapes(c).items():
        if name == "markov_head.markov_w1.weight":
            shapes[name] = shape
        elif name in ("embed_tokens.weight", "lm_head.weight") or "markov" in name:
            shapes[name] = (c.vocab // ranks, shape[1])
        elif name == "fc.weight":
            shapes[name] = (c.hidden, c.context_width // ranks)
        elif name.endswith("q_proj.weight"):
            shapes[name] = (q_width, c.hidden)
        elif name.endswith("k_proj.weight") or name.endswith("v_proj.weight"):
            shapes[name] = (c.head_dim, c.hidden)
        elif name.endswith("o_proj.weight"):
            shapes[name] = (c.hidden, q_width)
        elif name.endswith("gate_proj.weight") or name.endswith("up_proj.weight"):
            shapes[name] = (c.intermediate // ranks, c.hidden)
        elif name.endswith("down_proj.weight"):
            shapes[name] = (c.hidden, c.intermediate // ranks)
        else:
            shapes[name] = shape
    return shapes


def random_sharded_weights(c: DraftConfig, mesh, seed: int = 0):
    """Random bf16 shards generated per rank on the host that owns them.

    Replicated tensors (norm scales) are identical on every rank; sharded
    tensors differ per rank. Suitable for timing, not for a coherent model.
    """
    from jax.sharding import NamedSharding, PartitionSpec as P

    ranks = mesh.size
    sharding = NamedSharding(mesh, P("tp"))
    weights = {}
    for index, (name, shape) in enumerate(sorted(local_weight_shapes(c, ranks).items())):
        replicated = name.endswith("norm.weight") or name == "markov_head.markov_w1.weight"

        def make(block_index, name=name, shape=shape, index=index, replicated=replicated):
            rank = 0 if replicated else (block_index[0].start or 0)
            rng = np.random.default_rng((seed * 1000 + index) * 64 + rank)
            if replicated:
                value = 1 + 0.05 * rng.standard_normal(shape, dtype=np.float32)
            elif "markov" in name:
                value = 0.01 * rng.standard_normal(shape, dtype=np.float32)
            else:
                value = rng.standard_normal(shape, dtype=np.float32) / np.sqrt(shape[-1])
            return jnp.asarray(value[None]).astype(jnp.bfloat16)

        weights[name] = jax.make_array_from_callback((ranks, *shape), sharding, make)
    return weights


def shard_weights_to_mesh(c: DraftConfig, weights, mesh):
    """Place ``[32, ...]`` shards on the mesh with the rank on ``tp``.

    Shards are cut on the host; each process uploads only the slices that
    live on its own devices.
    """
    from jax.sharding import NamedSharding, PartitionSpec as P

    sharding = NamedSharding(mesh, P("tp"))
    placed = {}
    for name, value in _shard_weights(c, weights, mesh.size).items():
        placed[name] = jax.make_array_from_callback(
            value.shape, sharding, lambda index, value=value: np.ascontiguousarray(value[index])
        )
    return placed


def empty_cache(c: DraftConfig, mesh):
    """Per-rank context key/value ring buffers and the replicated slot positions."""
    from jax.sharding import NamedSharding, PartitionSpec as P

    ranks = mesh.size
    keys = jnp.zeros((ranks, c.layers, c.window, c.head_dim), jnp.bfloat16)
    values = jnp.zeros((ranks, c.layers, c.window, c.head_dim), jnp.bfloat16)
    positions = jnp.full((c.window,), -1, jnp.int32)
    sharded = NamedSharding(mesh, P("tp"))
    return (
        jax.device_put(keys, sharded),
        jax.device_put(values, sharded),
        jax.device_put(positions, NamedSharding(mesh, P())),
    )


def _vocab_gather(table, token_ids, rank, rows_per_rank):
    """Replicated lookup of vocabulary-sharded rows via a masked TP32 sum."""
    local = token_ids - rank * rows_per_rank
    valid = (local >= 0) & (local < rows_per_rank)
    rows = table[jnp.clip(local, 0, rows_per_rank - 1)]
    return jax.lax.psum(
        jnp.where(valid[:, None], rows.astype(jnp.float32), 0), "tp"
    ).astype(table.dtype)


def make_context_update(c: DraftConfig, mesh, rows: int):
    """Project ``rows`` aux rows into the context cache at their positions.

    ``aux`` is ``[len(aux_layers), rows, hidden]`` (the megakernel's aux
    output), ``positions`` is ``[rows]``. Every row is written; the draft step
    masks slots whose position is not below the anchor position.
    """
    from jax.sharding import PartitionSpec as P

    ranks = mesh.size
    aux_width = c.context_width // ranks

    def local(weights, cache, cache_positions, aux, positions):
        weights = jax.tree.map(lambda x: x[0], weights)
        keys, values = (x[0] for x in cache[:2])
        rank = jax.lax.axis_index("tp")
        # fc contracts over the concatenated aux features; each rank holds a
        # slice of the input features and the partial products are summed.
        aux_rows = aux.transpose(1, 0, 2).reshape(rows, c.context_width)
        aux_slice = jax.lax.dynamic_slice_in_dim(aux_rows, rank * aux_width, aux_width, axis=1)
        partial = jnp.einsum(
            "ri,oi->ro", aux_slice, weights["fc.weight"], preferred_element_type=jnp.float32
        )
        context = jax.lax.psum(partial, "tp").astype(jnp.bfloat16)
        context = rms_norm(context, weights["hidden_norm.weight"], c.eps)
        cos, sin = rope_cos_sin(positions, c.head_dim, c.rope_theta, jnp.bfloat16)
        slots = positions % c.window
        for layer in range(c.layers):
            prefix = f"layers.{layer}.self_attn."
            k = linear(context, weights[prefix + "k_proj.weight"])[:, None, :]
            k = apply_rope(rms_norm(k, weights[prefix + "k_norm.weight"], c.eps), cos, sin)[:, 0]
            v = linear(context, weights[prefix + "v_proj.weight"])
            keys = keys.at[layer, slots].set(k)
            values = values.at[layer, slots].set(v)
        cache_positions = cache_positions.at[slots].set(positions)
        return keys[None], values[None], cache_positions

    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P("tp"), (P("tp"), P("tp"), P()), P(), P(), P()),
            out_specs=(P("tp"), P("tp"), P()),
            check_vma=False,
        ),
        donate_argnums=(1,),
    )


def _draft_backbone(c, weights, context_keys, context_values, cache_positions,
                    anchor_token, anchor_position, block_size, rank):
    """Per-rank DFlash backbone: returns the final-normed block hidden rows."""
    ranks_q = weights["layers.0.self_attn.q_proj.weight"].shape[0] // c.head_dim
    vocab_per_rank = weights["embed_tokens.weight"].shape[0]
    tokens = jnp.concatenate(
        (anchor_token[None], jnp.full((block_size - 1,), c.mask_token, jnp.int32))
    )
    x = _vocab_gather(weights["embed_tokens.weight"], tokens, rank, vocab_per_rank)
    block_positions = anchor_position + jnp.arange(block_size, dtype=jnp.int32)
    cos, sin = rope_cos_sin(block_positions, c.head_dim, c.rope_theta, x.dtype)
    context_allowed = (
        (cache_positions >= 0)
        & (cache_positions < anchor_position)
        & (cache_positions >= anchor_position - c.window)
    )
    block_allowed = jnp.arange(block_size)[None, :] <= jnp.arange(block_size)[:, None]
    allowed = jnp.concatenate(
        (jnp.broadcast_to(context_allowed[None], (block_size, c.window)), block_allowed),
        axis=1,
    )
    for layer in range(c.layers):
        prefix = f"layers.{layer}."
        h = rms_norm(x, weights[prefix + "input_layernorm.weight"], c.eps)
        q = linear(h, weights[prefix + "self_attn.q_proj.weight"]).reshape(
            block_size, ranks_q, c.head_dim
        )
        q = apply_rope(rms_norm(q, weights[prefix + "self_attn.q_norm.weight"], c.eps), cos, sin)
        k = linear(h, weights[prefix + "self_attn.k_proj.weight"])[:, None, :]
        k = apply_rope(rms_norm(k, weights[prefix + "self_attn.k_norm.weight"], c.eps), cos, sin)
        v = linear(h, weights[prefix + "self_attn.v_proj.weight"])[:, None, :]
        keys = jnp.concatenate((context_keys[layer][:, None, :], k), axis=0)
        values = jnp.concatenate((context_values[layer][:, None, :], v), axis=0)
        attended = _attend(q, keys, values, allowed, ranks_q).reshape(
            block_size, ranks_q * c.head_dim
        )
        partial = jnp.einsum(
            "ri,oi->ro",
            attended,
            weights[prefix + "self_attn.o_proj.weight"],
            preferred_element_type=jnp.float32,
        )
        x = x + jax.lax.psum(partial, "tp").astype(x.dtype)
        h = rms_norm(x, weights[prefix + "post_attention_layernorm.weight"], c.eps)
        gate = linear(h, weights[prefix + "mlp.gate_proj.weight"])
        up = linear(h, weights[prefix + "mlp.up_proj.weight"])
        partial = jnp.einsum(
            "ri,oi->ro",
            silu(gate) * up,
            weights[prefix + "mlp.down_proj.weight"],
            preferred_element_type=jnp.float32,
        )
        x = x + jax.lax.psum(partial, "tp").astype(x.dtype)
    return rms_norm(x, weights["norm.weight"], c.eps)


def _pack_candidates(values, ids, w2_rows):
    """``[block, M]`` FP32 values, int32 ids and ``[block, M, r]`` bf16 rows -> int32 ``[block, M, r/2 + 2]``."""
    block, count, width = w2_rows.shape
    return jnp.concatenate(
        (
            jax.lax.bitcast_convert_type(w2_rows.reshape(block, count, width // 2, 2), jnp.int32),
            jax.lax.bitcast_convert_type(values, jnp.int32)[..., None],
            ids[..., None],
        ),
        axis=2,
    )


def _unpack_candidates(packed):
    half = packed.shape[-1] - 2
    block = packed.shape[0]
    w2_rows = jax.lax.bitcast_convert_type(packed[..., :half], jnp.bfloat16).reshape(block, -1, 2 * half)
    values = jax.lax.bitcast_convert_type(packed[..., half], jnp.float32)
    return values, packed[..., half + 1], w2_rows


def _markov_sample(c, weights, hidden, anchor_token, rank):
    """Greedy sampling with the Markov bias, left to right.

    With ``markov_top_m > 0`` every shard keeps its top-M base logits per slot
    together with their ``markov_w2`` rows; one all-gather then lets each
    rank run the whole sequential chain locally (``markov_w1`` is replicated).
    Otherwise the exact chain runs with a lookup and a global argmax per slot.
    """
    vocab_per_rank = weights["lm_head.weight"].shape[0]
    base = linear(hidden, weights["lm_head.weight"]).astype(jnp.float32)
    block_size = hidden.shape[0]
    if c.markov_top_m:
        top_m = min(c.markov_top_m, vocab_per_rank)
        values, local_ids = jax.lax.top_k(base, top_m)  # [block, M]
        global_ids = local_ids + rank * vocab_per_rank
        w2_rows = weights["markov_head.markov_w2.weight"][local_ids]  # [block, M, rank]
        # One all-gather of an int32 payload. Ints are never flushed: token ids
        # reinterpreted as FP32 are denormals and TPUs flush those to zero.
        packed = jax.lax.all_gather(
            _pack_candidates(values, global_ids, w2_rows), "tp", axis=1, tiled=True
        )  # [block, ranks*M, half + 2]
        values, global_ids, w2_rows = _unpack_candidates(packed)
        previous = anchor_token
        drafts = []
        for slot in range(block_size):
            embedding = weights["markov_head.markov_w1.weight"][previous]  # [rank]
            bias = jnp.einsum(
                "mr,r->m", w2_rows[slot], embedding, preferred_element_type=jnp.float32
            )
            pick = jnp.argmax(values[slot] + bias)
            previous = global_ids[slot, pick].astype(jnp.int32)
            drafts.append(previous)
        return jnp.stack(drafts)
    # Exact path: the bias is computed for the whole local vocabulary shard
    # (markov_w1 is replicated), then one all-gather of (max, id) per slot.
    previous = anchor_token
    drafts = []
    for slot in range(block_size):
        embedding = weights["markov_head.markov_w1.weight"][previous]  # [rank]
        bias = jnp.einsum(
            "vr,r->v", weights["markov_head.markov_w2.weight"], embedding,
            preferred_element_type=jnp.float32,
        )
        logits = base[slot] + bias
        local_id = jnp.argmax(logits).astype(jnp.int32)
        local_max = logits[local_id]
        maxima, ids = jax.lax.all_gather(
            (local_max, local_id + rank * vocab_per_rank), "tp", axis=0
        )
        previous = ids[jnp.argmax(maxima)].astype(jnp.int32)
        drafts.append(previous)
    return jnp.stack(drafts)


def make_draft_step(c: DraftConfig, mesh, block_size: int, phase: str = "full"):
    """Draft ``block_size`` tokens for one sequence: returns ``[block_size]``
    ids where slot ``k`` predicts position ``anchor_position + k + 1``.

    ``phase`` selects the program for profiling: ``"full"`` (default),
    ``"backbone"`` (returns the replicated block hidden rows instead of ids)
    or ``"sampling"`` (takes those hidden rows and the anchor, returns ids).
    """
    from jax.sharding import PartitionSpec as P

    if phase not in ("full", "backbone", "sampling"):
        raise ValueError("phase must be 'full', 'backbone' or 'sampling'")

    def local(weights, cache, cache_positions, anchor_token, anchor_position):
        weights = jax.tree.map(lambda x: x[0], weights)
        context_keys, context_values = (x[0] for x in cache[:2])
        rank = jax.lax.axis_index("tp")
        hidden = _draft_backbone(
            c, weights, context_keys, context_values, cache_positions,
            anchor_token, anchor_position, block_size, rank,
        )
        if phase == "backbone":
            return hidden
        return _markov_sample(c, weights, hidden, anchor_token, rank)

    def local_sampling(weights, hidden, anchor_token):
        weights = jax.tree.map(lambda x: x[0], weights)
        return _markov_sample(c, weights, hidden, anchor_token, jax.lax.axis_index("tp"))

    if phase == "sampling":
        return jax.jit(
            jax.shard_map(
                local_sampling, mesh=mesh, in_specs=(P("tp"), P(), P()), out_specs=P(),
                check_vma=False,
            )
        )
    return jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P("tp"), (P("tp"), P("tp"), P()), P(), P(), P()),
            out_specs=P(),
            check_vma=False,
        )
    )


def make_sharded_argmax(mesh, rows: int, vocab: int, margin: bool = False):
    """Greedy ids from ``[32, rows, vocab / 32]`` logits with one all-gather.

    With ``margin`` the program also returns the gap between the best and the
    second-best logit per row (a near-tie diagnostic; still one all-gather).
    """
    from jax.sharding import PartitionSpec as P

    ranks = mesh.size
    vocab_per_rank = vocab // ranks

    def local(logits):
        local_logits = logits[0]
        rank = jax.lax.axis_index("tp")
        if margin:
            local_top, local_ids = jax.lax.top_k(local_logits, 2)  # [rows, 2]
            values, ids = jax.lax.all_gather(
                (local_top.T, (local_ids.astype(jnp.int32) + rank * vocab_per_rank).T), "tp", axis=0, tiled=True
            )  # [2 * ranks, rows]
            best, order = jax.lax.top_k(values.T, 2)  # [rows, 2], ties resolve to the lower index
            winner = jnp.take_along_axis(ids.T, order[:, :1], axis=1)[:, 0]
            return winner, best[:, 0] - best[:, 1]
        local_id = jnp.argmax(local_logits, axis=1).astype(jnp.int32)
        local_max = jnp.take_along_axis(local_logits, local_id[:, None], axis=1)[:, 0]
        maxima, ids = jax.lax.all_gather(
            (local_max, local_id + rank * vocab_per_rank), "tp", axis=0
        )  # [ranks, rows]
        winner = jnp.argmax(maxima, axis=0)  # first shard holding the maximum
        return jnp.take_along_axis(ids, winner[None], axis=0)[0]

    out_specs = (P(), P()) if margin else P()
    return jax.jit(
        jax.shard_map(local, mesh=mesh, in_specs=(P("tp"),), out_specs=out_specs, check_vma=False)
    )


def nucleus_accept(values, ids, lse, drafts, forced_count, top_p, uniforms):
    """Speculative top-p sampling over gathered candidates; returns ``(count, bonus)``.

    ``values`` ``[rows, n]`` are candidate logits (temperature applied) with
    token ``ids``, ``lse`` ``[rows]`` the log-normalizer of each row over the
    full vocabulary, so ``exp(values - lse)`` are exact probabilities. Row k's
    nucleus p' keeps the largest candidates whose cumulative mass reaches
    ``top_p`` (at least one) and renormalizes. ``drafts[k]`` is the token the
    draft proposed for row k (-1: none; the last row never has one). The
    drafts are deterministic, so their distribution is one-hot and exact
    rejection sampling reduces to: accept draft k with probability p'_k(d_k)
    (``uniforms[k, 0] < p'``); ``count`` is the number of leading accepted
    drafts; the bonus token is drawn (``uniforms[count, 1]``) from p'_count
    with the rejected draft removed, or from the full p'_count when every
    draft was accepted. With ``forced_count >= 0`` the count is given and the
    bonus is drawn from the full p'_count (prefill). The output is
    distributed exactly as p' (up to the candidate truncation).
    """
    values, ids, lse, drafts, uniforms = (jnp.asarray(x) for x in (values, ids, lse, drafts, uniforms))
    rows, n = values.shape
    order = jnp.argsort(-values, axis=1)
    sorted_values = jnp.take_along_axis(values, order, axis=1)
    sorted_ids = jnp.take_along_axis(ids, order, axis=1)
    probs = jnp.exp(sorted_values - lse[:, None])  # [rows, n], descending
    cumulative = jnp.cumsum(probs, axis=1)
    keep = ((cumulative - probs) < top_p) | (jnp.arange(n)[None, :] == 0)
    kept = jnp.where(keep, probs, 0.0)
    nucleus = kept / jnp.sum(kept, axis=1, keepdims=True)  # p'
    draft_mask = sorted_ids == drafts[:, None]
    draft_prob = jnp.sum(jnp.where(draft_mask, nucleus, 0.0), axis=1)  # [rows]
    k = jnp.arange(rows, dtype=jnp.int32)
    drafted = drafts >= 0
    accept = drafted & (uniforms[:, 0] < draft_prob)  # a row without a draft is a rejection
    first_reject = jnp.min(jnp.where(~accept & (k < rows - 1), k, rows - 1))
    count = jnp.where(forced_count >= 0, forced_count, first_reject).astype(jnp.int32)
    row = nucleus[count]
    remove = (forced_count < 0) & drafted[count]  # the rejected draft leaves the residual
    residual = jnp.where(remove & draft_mask[count], 0.0, row)
    cumulative = jnp.cumsum(residual)
    total = cumulative[-1]
    pick = jnp.argmax(cumulative > uniforms[count, 1] * total)  # first index past the threshold
    bonus = jnp.where(total > 0, sorted_ids[count, pick], sorted_ids[count, 0])
    return count, bonus.astype(jnp.int32)


def make_row_sampler(mesh, rows: int, vocab: int, candidates: int = 128):
    """``(logits [32, rows, vocab / 32], drafts [rows], forced_count, temperature, top_p, uniforms [rows, 2])``
    -> ``(count, bonus)``: :func:`nucleus_accept` on the verify's vocabulary-sharded logits.

    Every rank contributes its ``candidates`` largest logits (and the exact
    log-normalizer partials) through one all-gather, so the nucleus is exact
    unless the top-p mass needs more than ``candidates`` tokens of one rank.
    The gathered part runs replicated on every rank; all ranks (and hosts)
    reach the same ``count`` and ``bonus``. ``temperature`` must be > 0.
    """
    from jax.sharding import PartitionSpec as P

    ranks = mesh.size
    width = vocab // ranks

    def local(logits, drafts, forced_count, temperature, top_p, uniforms):
        local_logits = logits[0].astype(jnp.float32) / temperature  # [rows, vocab / 32]
        rank = jax.lax.axis_index("tp")
        local_max = jnp.max(local_logits, axis=1)
        local_lse = jnp.log(jnp.sum(jnp.exp(local_logits - local_max[:, None]), axis=1))  # relative to local_max
        values, ids = jax.lax.top_k(local_logits, candidates)  # [rows, candidates]
        ids = ids.astype(jnp.int32) + rank * width
        maxima, lses, values, ids = jax.lax.all_gather((local_max, local_lse, values, ids), "tp", axis=0)
        best = jnp.max(maxima, axis=0)  # [rows]
        lse = best + jnp.log(jnp.sum(jnp.exp(maxima - best[None] + lses), axis=0))
        values = jnp.transpose(values, (1, 0, 2)).reshape(rows, ranks * candidates)
        ids = jnp.transpose(ids, (1, 0, 2)).reshape(rows, ranks * candidates)
        return nucleus_accept(values, ids, lse, drafts, forced_count, top_p, uniforms)

    return jax.jit(jax.shard_map(
        local, mesh=mesh, in_specs=(P("tp"), P(), P(), P(), P(), P()), out_specs=(P(), P()), check_vma=False,
    ))


def load_checkpoint_weights(directory: Path, c: DraftConfig | None = None):
    """Read the safetensors file into checkpoint-named bf16 arrays."""
    from safetensors.numpy import load_file

    directory = Path(directory)
    c = c or DraftConfig.from_checkpoint(directory)
    import ml_dtypes

    raw = load_file(str(directory / "model.safetensors"))
    weights = {}
    for name, shape in weight_shapes(c).items():
        value = raw[name]
        if tuple(value.shape) != shape:
            raise ValueError(f"{name}: expected {shape}, found {value.shape}")
        # safetensors' numpy loader hands bf16 back as uint16 bit patterns.
        if value.dtype == np.uint16:
            value = value.view(ml_dtypes.bfloat16)
        weights[name] = np.asarray(value).astype(ml_dtypes.bfloat16)
    return weights
ROWS = 8  # rows the kernel computes (anchor + 7); smaller blocks are masked
RANKS = 32
LM_CHUNK = 512  # LM head columns per chunk (the gate/up buffer width)
MLP_PADDED = 512  # 448 MLP channels per rank padded to whole lane tiles
LM_CHUNKS = 10  # 10 * 512 = 5120 vocabulary rows per rank
FC_IN = 35840 // 4  # aux features contracted per host (8960)
CHUNK = 7168 // 8  # 896-column chunk of the hidden vector per host lane
PACKED = ROWS * CHUNK // 256  # 28 FP32 rows hold one BF16 [8, 896] chunk
ROW_PACKED = 7168 // 256  # 28 FP32 rows hold one BF16 [1, 7168] row
NEG = -3.0e38

# Output vector layout (int32 [1, 128]): drafts, this step's verify targets,
# accepted count, bonus token (the next anchor), top-2 margins as FP32 bits.
OUT_DRAFTS, OUT_TARGETS, OUT_COUNT, OUT_BONUS, OUT_MARGINS = 0, 8, 16, 17, 24

# Semaphore index ranges (per slot) so a fast peer's next phase can never
# satisfy a wait of the previous phase.
SEM_RS, SEM_HOST, SEM_GATHER, SEM_TINY, SEM_BCAST, SEMS_PER_SLOT = 0, 8, 16, 32, 64, 96

VMEM_NAMES = (
    "ln1", "ln2", "qnorm", "kvnorm", "hidden_norm", "final_norm", "mask_embed",
    "rot_q", "rot_kv", "head_ones", "kv_ones", "head_select", "head_place",
)
HBM_NAMES = ("q", "kv", "o", "gate", "up", "down", "fc", "lm_head", "w2t", "w1", "embed", "rope_q", "rope_kv")
WEIGHT_NAMES = VMEM_NAMES + HBM_NAMES


# --------------------------------------------------------------------------
# Host-side weight preparation


def _rotation(head_dim: int) -> np.ndarray:
    """``x @ R`` = ``concat(-x[half:], x[:half])`` (neox rotate-half)."""
    half = head_dim // 2
    rotation = np.zeros((head_dim, head_dim), np.float32)
    for i in range(half):
        rotation[i + half, i] = -1.0
        rotation[i, i + half] = 1.0
    return rotation


def rope_tables(c: DraftConfig, max_position: int):
    """``[P, 384]`` (cos x3 | sin x3) for the 3 local q heads and ``[P, 256]``
    (cos | ones | sin | zeros) for the packed K/V row, FP32."""
    positions = np.arange(max_position, dtype=np.float32)
    inv_freq = 1.0 / c.rope_theta ** (np.arange(0, c.head_dim, 2, dtype=np.float32) / c.head_dim)
    freqs = positions[:, None] * inv_freq[None, :]
    emb = np.concatenate((freqs, freqs), axis=-1)
    cos, sin = np.cos(emb), np.sin(emb)
    heads = 3
    rope_q = np.concatenate([cos] * heads + [sin] * heads, axis=1).astype(np.float32)
    ones, zeros = np.ones_like(cos), np.zeros_like(sin)
    rope_kv = np.concatenate((cos, ones, sin, zeros), axis=1).astype(np.float32)
    return rope_q, rope_kv


def kernel_weight_shards(c: DraftConfig, weights: dict[str, Any], *, max_position: int = 8192,
                         ranks: int = RANKS) -> dict[str, np.ndarray]:
    """Per-rank ``[ranks, ...]`` arrays in the kernel's layout from checkpoint-named arrays."""
    import ml_dtypes

    bf16 = ml_dtypes.bfloat16
    w = {name: np.asarray(value) for name, value in weights.items()}
    q_width = c.heads // ranks * c.head_dim  # 192
    kv_share = ranks // c.kv_heads  # 2
    mlp = c.intermediate // ranks  # 448
    vocab = c.vocab // ranks  # 5120
    shards: dict[str, np.ndarray] = {}

    def stack_layers(make):
        return np.stack([np.stack([make(layer, rank) for rank in range(ranks)]) for layer in range(c.layers)], axis=1)

    shards["q"] = stack_layers(
        lambda l, r: w[f"layers.{l}.self_attn.q_proj.weight"][r * q_width:(r + 1) * q_width].T
    ).astype(bf16)

    def kv(l, r):
        head = r // kv_share
        k = w[f"layers.{l}.self_attn.k_proj.weight"][head * c.head_dim:(head + 1) * c.head_dim].T
        v = w[f"layers.{l}.self_attn.v_proj.weight"][head * c.head_dim:(head + 1) * c.head_dim].T
        return np.concatenate((k, v), axis=1)

    shards["kv"] = stack_layers(kv).astype(bf16)
    # o and down produce the partial that is reduce-scattered: memory block k of
    # their 896-column chunks holds logical chunk (lane ^ k), so chunk k can be
    # sent to peer rank ^ k as soon as the MXU produces it.
    def permute_chunks(value, rank):
        lane = rank % 8
        return np.concatenate([value[:, (lane ^ k) * CHUNK:((lane ^ k) + 1) * CHUNK] for k in range(8)], axis=1)

    shards["o"] = stack_layers(
        lambda l, r: permute_chunks(w[f"layers.{l}.self_attn.o_proj.weight"][:, r * q_width:(r + 1) * q_width].T, r)
    ).astype(bf16)
    # MLP channels padded from 448 to 512 per rank: lane windows in VMEM must be
    # whole 128-lane tiles. The padding columns/rows are zero.
    def pad_columns(value):
        return np.pad(value, ((0, 0), (0, MLP_PADDED - value.shape[1])))

    def pad_rows(value):
        return np.pad(value, ((0, MLP_PADDED - value.shape[0]), (0, 0)))

    shards["gate"] = stack_layers(
        lambda l, r: pad_columns(w[f"layers.{l}.mlp.gate_proj.weight"][r * mlp:(r + 1) * mlp].T)
    ).astype(bf16)
    shards["up"] = stack_layers(
        lambda l, r: pad_columns(w[f"layers.{l}.mlp.up_proj.weight"][r * mlp:(r + 1) * mlp].T)
    ).astype(bf16)
    shards["down"] = stack_layers(
        lambda l, r: permute_chunks(pad_rows(w[f"layers.{l}.mlp.down_proj.weight"][:, r * mlp:(r + 1) * mlp].T), r)
    ).astype(bf16)

    def replicated(value):
        return np.broadcast_to(np.asarray(value)[None], (ranks, *np.asarray(value).shape))

    shards["ln1"] = replicated(np.stack([w[f"layers.{l}.input_layernorm.weight"][None] for l in range(c.layers)])).astype(bf16)
    shards["ln2"] = replicated(np.stack([w[f"layers.{l}.post_attention_layernorm.weight"][None] for l in range(c.layers)])).astype(bf16)
    shards["qnorm"] = replicated(np.stack([np.tile(w[f"layers.{l}.self_attn.q_norm.weight"], 3)[None] for l in range(c.layers)])).astype(bf16)
    shards["kvnorm"] = replicated(np.stack([
        np.concatenate((w[f"layers.{l}.self_attn.k_norm.weight"], np.ones((c.head_dim,), np.float32)))[None]
        for l in range(c.layers)
    ])).astype(bf16)
    shards["hidden_norm"] = replicated(w["hidden_norm.weight"][None]).astype(bf16)
    shards["final_norm"] = replicated(w["norm.weight"][None]).astype(bf16)
    shards["mask_embed"] = replicated(w["embed_tokens.weight"][c.mask_token][None]).astype(bf16)

    # fc: rank (host h, lane j) contracts host h's quarter of the concatenated
    # aux features for output columns [896 j, 896 (j + 1)).
    fc = w["fc.weight"]  # [7168 out, 35840 in]
    shards["fc"] = np.stack([
        fc[(r % 8) * CHUNK:(r % 8 + 1) * CHUNK, (r // 8) * FC_IN:(r // 8 + 1) * FC_IN].T for r in range(ranks)
    ]).astype(bf16)

    assert LM_CHUNKS * LM_CHUNK == vocab
    lm = w["lm_head.weight"].reshape(ranks, LM_CHUNKS, LM_CHUNK, c.hidden)  # [ranks, 10, 512, 7168]
    shards["lm_head"] = np.ascontiguousarray(np.transpose(lm, (0, 1, 3, 2))).astype(bf16)  # [ranks, 10, 7168, 512]
    w2 = w["markov_head.markov_w2.weight"].reshape(ranks, vocab, c.markov_rank)
    shards["w2t"] = np.ascontiguousarray(np.transpose(w2, (0, 2, 1))).astype(bf16)  # [ranks, 256, 5120]
    shards["w1"] = replicated(w["markov_head.markov_w1.weight"]).astype(bf16)
    shards["embed"] = w["embed_tokens.weight"].reshape(ranks, vocab, c.hidden).astype(bf16)

    rotation = _rotation(c.head_dim)
    shards["rot_q"] = replicated(np.kron(np.eye(3, dtype=np.float32), rotation)).astype(bf16)
    rot_kv = np.zeros((2 * c.head_dim, 2 * c.head_dim), np.float32)
    rot_kv[:c.head_dim, :c.head_dim] = rotation
    shards["rot_kv"] = replicated(rot_kv).astype(bf16)
    shards["head_ones"] = replicated(np.kron(np.eye(3, dtype=np.float32), np.ones((c.head_dim, c.head_dim), np.float32))).astype(bf16)
    kv_ones = np.zeros((2 * c.head_dim, 2 * c.head_dim), np.float32)
    kv_ones[:c.head_dim, :c.head_dim] = 1.0
    shards["kv_ones"] = replicated(kv_ones).astype(bf16)
    select = np.zeros((3, 3 * c.head_dim, 2 * c.head_dim), np.float32)
    place = np.zeros((3, 2 * c.head_dim, 3 * c.head_dim), np.float32)
    for head in range(3):
        for i in range(c.head_dim):
            select[head, head * c.head_dim + i, i] = 1.0
            place[head, c.head_dim + i, head * c.head_dim + i] = 1.0
    shards["head_select"] = replicated(select).astype(bf16)
    shards["head_place"] = replicated(place).astype(bf16)
    rope_q, rope_kv = rope_tables(c, max_position + 32)
    shards["rope_q"] = replicated(rope_q)
    shards["rope_kv"] = replicated(rope_kv)
    return shards


def kernel_weight_shapes(c: DraftConfig, *, max_position: int = 8192, ranks: int = RANKS) -> dict[str, tuple]:
    """``{name: (per-rank shape, dtype)}`` of :func:`kernel_weight_shards` without building them."""
    layers, hidden, head = c.layers, c.hidden, c.head_dim
    q_width, kv_width = 3 * head, 2 * head
    vocab = c.vocab // ranks
    bf16, f32 = jnp.bfloat16, jnp.float32
    return {
        "q": ((layers, hidden, q_width), bf16),
        "kv": ((layers, hidden, kv_width), bf16),
        "o": ((layers, q_width, hidden), bf16),
        "gate": ((layers, hidden, MLP_PADDED), bf16),
        "up": ((layers, hidden, MLP_PADDED), bf16),
        "down": ((layers, MLP_PADDED, hidden), bf16),
        "ln1": ((layers, 1, hidden), bf16),
        "ln2": ((layers, 1, hidden), bf16),
        "qnorm": ((layers, 1, q_width), bf16),
        "kvnorm": ((layers, 1, kv_width), bf16),
        "hidden_norm": ((1, hidden), bf16),
        "final_norm": ((1, hidden), bf16),
        "mask_embed": ((1, hidden), bf16),
        "fc": ((FC_IN, CHUNK), bf16),
        "lm_head": ((LM_CHUNKS, hidden, LM_CHUNK), bf16),
        "w2t": ((c.markov_rank, vocab), bf16),
        "w1": ((c.vocab, c.markov_rank), bf16),
        "embed": ((vocab, hidden), bf16),
        "rot_q": ((q_width, q_width), bf16),
        "rot_kv": ((kv_width, kv_width), bf16),
        "head_ones": ((q_width, q_width), bf16),
        "kv_ones": ((kv_width, kv_width), bf16),
        "head_select": ((3, q_width, kv_width), bf16),
        "head_place": ((3, kv_width, q_width), bf16),
        "rope_q": ((max_position + 32, 6 * head), f32),
        "rope_kv": ((max_position + 32, 4 * head), f32),
    }


KERNEL_SHARD_FORMAT = 1


def save_kernel_shards(shards: dict[str, np.ndarray], directory, *, ranks: int = RANKS) -> None:
    """Write ``kernel_weight_shards`` output as one raw file per rank plus a shared index."""
    import json

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = list(shards)
    index = {"format": KERNEL_SHARD_FORMAT, "ranks": ranks, "arrays": {}}
    offset = 0
    for name in names:
        value = shards[name]
        index["arrays"][name] = {"offset": offset, "shape": list(value.shape[1:]), "dtype": str(np.dtype(value.dtype))}
        offset += int(np.prod(value.shape[1:])) * np.dtype(value.dtype).itemsize
    index["bytes_per_rank"] = offset
    for rank in range(ranks):
        with open(directory / f"rank{rank:02d}.bin", "wb") as handle:
            for name in names:
                handle.write(np.ascontiguousarray(shards[name][rank]).tobytes())
    (directory / "index.json").write_text(json.dumps(index, indent=1))


def load_kernel_shards(directory, mesh, *, max_position: int | None = None) -> dict[str, jax.Array]:
    """Load :func:`save_kernel_shards` files: each process reads its ranks' files.

    ``max_position`` trims the RoPE tables to ``max_position + 32`` rows so the
    arrays match :func:`kernel_weight_shapes` for the caller's context length.
    """
    import json
    from concurrent.futures import ThreadPoolExecutor
    from jax.sharding import NamedSharding, PartitionSpec as P

    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text())
    if index.get("format") != KERNEL_SHARD_FORMAT:
        raise ValueError(f"{directory}: unexpected kernel shard format {index.get('format')}")
    devices = list(mesh.devices.flat)
    local_ranks = [rank for rank, device in enumerate(devices) if device.process_index == jax.process_index()]
    rope_rows = None if max_position is None else max_position + 32

    def load_rank(rank):
        raw = np.fromfile(directory / f"rank{rank:02d}.bin", dtype=np.uint8)
        arrays = {}
        for name, spec in index["arrays"].items():
            dtype = np.dtype(spec["dtype"])
            count = int(np.prod(spec["shape"]))
            value = raw[spec["offset"]:spec["offset"] + count * dtype.itemsize].view(dtype).reshape(spec["shape"])
            if name in ("rope_q", "rope_kv") and rope_rows is not None:
                if value.shape[0] < rope_rows:
                    raise ValueError(f"{directory}: RoPE table has {value.shape[0]} rows, need {rope_rows}")
                value = value[:rope_rows]
            arrays[name] = jax.device_put(np.ascontiguousarray(value)[None], devices[rank])
        return rank, arrays

    with ThreadPoolExecutor(len(local_ranks)) as pool:
        per_rank = dict(pool.map(load_rank, local_ranks))
    sharding = NamedSharding(mesh, P("tp"))
    return {
        name: jax.make_array_from_single_device_arrays(
            (RANKS, *per_rank[local_ranks[0]][name].shape[1:]), sharding, [per_rank[rank][name] for rank in local_ranks]
        )
        for name in index["arrays"]
    }


def place_kernel_weights(shards: dict[str, np.ndarray], mesh) -> dict[str, jax.Array]:
    """Upload ``[32, ...]`` shards with the rank on ``tp``; each process copies its own slices."""
    from jax.sharding import NamedSharding, PartitionSpec as P

    sharding = NamedSharding(mesh, P("tp"))
    return {
        name: jax.make_array_from_callback(
            value.shape, sharding, lambda index, value=value: np.ascontiguousarray(value[index])
        )
        for name, value in shards.items()
    }


def empty_kernel_cache(c: DraftConfig, mesh):
    """Per-rank ``[layers, window, 128]`` K/V ring (K lanes 0..63, V lanes 64..127) and positions."""
    from jax.sharding import NamedSharding, PartitionSpec as P

    cache = jax.jit(
        lambda: jnp.zeros((mesh.size, c.layers, c.window, 2 * c.head_dim), jnp.bfloat16),
        out_shardings=NamedSharding(mesh, P("tp")),
    )()
    positions = jax.device_put(jnp.full((1, c.window), -1, jnp.int32), NamedSharding(mesh, P()))
    return cache, positions


def kernel_cache_from_draft_cache(keys, values):
    """Convert the XLA draft's ``[32, layers, window, 64]`` K and V into the packed kernel cache."""
    return jnp.concatenate((keys, values), axis=-1)


# --------------------------------------------------------------------------
# In-kernel helpers


def pack(value):
    """``[rows, width]`` -> FP32 ``[rows * width / 256, 128]`` words holding BF16 pairs."""
    rows, width = value.shape
    lanes = value.astype(jnp.float32).reshape(rows * width // 128, 128).astype(jnp.bfloat16)
    return jax.lax.bitcast_convert_type(tpu.bitcast(lanes, jnp.uint32), jnp.float32)


def unpack(words, rows, width):
    """Inverse of :func:`pack`, returned as FP32 ``[rows, width]``."""
    values = tpu.bitcast(jax.lax.bitcast_convert_type(words, jnp.uint32), jnp.bfloat16)
    return values.astype(jnp.float32).reshape(rows, width)


def _bf16(x):
    return x.astype(jnp.bfloat16)


def _f32(x):
    return x.astype(jnp.float32)


def _rms_norm(x_bf16, weight_bf16, eps):
    """Qwen3RMSNorm as the reference computes it: FP32 statistics, BF16 cast, BF16 scale."""
    x32 = _f32(x_bf16)
    normed = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return _bf16(_f32(_bf16(normed)) * _f32(weight_bf16))


def _grouped_sumsq(x32, ones_bf16):
    """Per-group sum of squares broadcast to the group's lanes: ``x^2 @ blockdiag(ones)``
    with the FP32 square split into BF16 high and low parts (relative error ~2^-16)."""
    squares = x32 * x32
    high = _bf16(squares)
    low = _bf16(squares - _f32(high))
    return _dot(high, ones_bf16) + _dot(low, ones_bf16)


def _rope(x_bf16, rotation_bf16, cos32, sin32):
    """Reference ``x * cos + rotate_half(x) * sin`` in BF16 arithmetic; cos/sin are BF16-valued."""
    rotated = _bf16(_dot(x_bf16, rotation_bf16))
    first = _bf16(_f32(x_bf16) * cos32)
    second = _bf16(_f32(rotated) * sin32)
    return _bf16(_f32(first) + _f32(second))


def _select_rows(block32, offset, rows):
    """Rows ``offset .. offset + rows`` of a ``[16, W]`` value as ``[rows, W]`` (masked sums)."""
    row_index = jax.lax.broadcasted_iota(jnp.int32, block32.shape, 0)
    return jnp.concatenate(
        [jnp.sum(jnp.where(row_index == offset + k, block32, 0.0), axis=0, keepdims=True) for k in range(rows)],
        axis=0,
    )


# --------------------------------------------------------------------------
# The kernel


STOP_POINTS = ("barrier", "fc_wait", "fc_matmul", "fc", "layers:1", "layers:2", "layers:3", "layers:4", "layers:5",
               "lm_head", "full")
SMALL_NAMES = ("aux", "positions", "logits", "previous") + VMEM_NAMES  # HBM operands copied into VMEM at entry


def _make_body(c: DraftConfig, block: int, eps: float, stop_after: str = "full", host_bf16: bool = True):
    """``stop_after`` (profiling only) ends the kernel after a phase: ``"fc"``,
    ``"layers:k"`` (k layers done), ``"lm_head"`` or ``"full"``. ``host_bf16``
    transports the cross-host partial sums as BF16 pairs (the megakernel's
    choice) instead of FP32."""
    if stop_after not in STOP_POINTS:
        raise ValueError(f"stop_after must be one of {STOP_POINTS}")
    stop_layers = int(stop_after.split(":")[1]) if stop_after.startswith("layers:") else None
    layers = c.layers
    window = c.window
    kv_width = 2 * c.head_dim  # 128
    q_width = 3 * c.head_dim  # 192
    vocab = c.vocab // RANKS  # 5120
    hidden = c.hidden
    scale = c.head_dim ** -0.5

    def body(scalars, aux_hbm, positions_hbm, logits_hbm, previous_hbm, *refs):
        weights = dict(zip(WEIGHT_NAMES, refs[: len(WEIGHT_NAMES)], strict=True))
        cache_in = refs[len(WEIGHT_NAMES)]
        tokens_out, positions_out, cache_out, hidden_out = refs[len(WEIGHT_NAMES) + 1: len(WEIGHT_NAMES) + 5]
        scratch = refs[len(WEIGHT_NAMES) + 5:]
        small = dict(zip(SMALL_NAMES, scratch[: len(SMALL_NAMES)], strict=True))
        (
            a_buf, q_buf, kv_buf, o_buf, gate_buf, up_buf, down_buf, w2t_buf, w1_blk, embed_blk, bcast_buf,
            rope_q_blk, rope_kv_blk, cache_vmem, logits_vmem, fc_part, tokens_v, hidden_v, stage, local_recv,
            host_buf, host_packed, gather_recv, tiny_recv, argmax_recv, dma_sems, small_sems, sends, recvs,
        ) = scratch[len(SMALL_NAMES):]

        rank = jax.lax.axis_index("tp")
        lane = rank % 8
        host = rank // 8
        new_position = scalars[0]  # position of the verify step's first row
        forced_count = scalars[1]  # >= 0: use this accepted count (prefill); else from the matches
        forced_anchor = scalars[2]  # >= 0: use this bonus token (sampled by the caller); else the target at ``count``

        # ---------------- DMA helpers ----------------
        def copy(src, dst, sem):
            return tpu.make_async_copy(src, dst, dma_sems.at[sem])

        def remote(src, dst, sem_index, peer):
            return tpu.make_async_remote_copy(
                src, dst, sends.at[sem_index], recvs.at[sem_index],
                device_id=(peer,), device_id_type=pl.DeviceIdType.MESH,
            )

        def exchange_start(src_of, dst_of, offsets, base, slot):
            copies = [
                remote(src_of(offset), dst_of(offset), slot * SEMS_PER_SLOT + base + offset, rank ^ offset)
                for offset in offsets
            ]
            for item in copies:
                item.start()
            return copies

        def exchange_wait(copies):
            for item in copies:
                item.wait()

        def exchange(src_of, dst_of, offsets, base, slot):
            exchange_wait(exchange_start(src_of, dst_of, offsets, base, slot))

        # ---------------- collectives ----------------
        def host_reduce_gather(chunk32, slot):
            """FP32 chunk partial of this host lane -> full [8, 7168] sum over hosts,
            BF16-rounded (the reference casts the reduced value to BF16)."""
            if host_bf16:
                host_packed[slot, host] = pack(_bf16(chunk32))
                exchange(lambda o: host_packed.at[slot, host], lambda o: host_packed.at[slot, host], (8, 16, 24), SEM_HOST, slot)
                reduced = sum(unpack(host_packed[slot, h], ROWS, CHUNK) for h in range(4))
            else:
                host_buf[slot, host] = chunk32
                exchange(lambda o: host_buf.at[slot, host], lambda o: host_buf.at[slot, host], (8, 16, 24), SEM_HOST, slot)
                reduced = jnp.sum(host_buf[slot], axis=0)
            gather_recv[slot, lane] = pack(_bf16(reduced))
            exchange(lambda o: gather_recv.at[slot, lane], lambda o: gather_recv.at[slot, lane], range(1, 8), SEM_GATHER, slot)
            return jnp.concatenate([unpack(gather_recv[slot, j], ROWS, CHUNK) for j in range(8)], axis=1)

        def all_reduce_start(lhs_bf16, weight_ref, slot):
            """Project ``lhs`` through a chunk-permuted ``[K, 7168]`` weight and issue the
            host-local reduce-scatter chunk by chunk: memory block k holds logical
            chunk ``lane ^ k`` and goes to peer ``rank ^ k`` right after its matmul."""
            copies = []
            own = None
            for k in range(8):
                partial = _dot(lhs_bf16, weight_ref[:, pl.ds(k * CHUNK, CHUNK)])  # [8, 896] FP32
                stage[slot, k] = pack(_bf16(partial))
                if k == 0:
                    own = copy(stage.at[slot, 0], local_recv.at[slot, lane], 13)
                    own.start()
                else:
                    item = remote(stage.at[slot, k], local_recv.at[slot, lane], slot * SEMS_PER_SLOT + SEM_RS + k, rank ^ k)
                    item.start()
                    copies.append(item)
            return own, copies

        def all_reduce_finish(handles, slot):
            own, copies = handles
            exchange_wait(copies)
            own.wait()
            chunk = sum(unpack(local_recv[slot, k], ROWS, CHUNK) for k in range(8))
            return host_reduce_gather(chunk, slot)

        def tiny_exchange(row_i32, slot):
            """One [1, 128] int32 row from every rank -> [32, 1, 128] int32 (row = rank)."""
            tiny_recv[slot, rank] = row_i32
            exchange(lambda o: tiny_recv.at[slot, rank], lambda o: tiny_recv.at[slot, rank], range(1, 32), SEM_TINY, slot)
            return tiny_recv[slot]

        # ---------------- prologue: barrier and DMA issue ----------------
        collectives32.barrier()
        small_sources = {
            "aux": aux_hbm, "positions": positions_hbm, "logits": logits_hbm, "previous": previous_hbm,
            **{name: weights[name] for name in VMEM_NAMES},
        }
        small_copies = [
            tpu.make_async_copy(small_sources[name], small[name], small_sems.at[index])
            for index, name in enumerate(SMALL_NAMES)
        ]
        for item in small_copies:
            item.start()
        fc_copy = copy(weights["fc"], a_buf.at[:, pl.ds(0, CHUNK)], 1)
        fc_copy.start()
        cache_copies = [copy(cache_in.at[layer], cache_vmem.at[layer], 16 + layer) for layer in range(layers)]
        for item in cache_copies:
            item.start()

        # ---------------- verify argmax, acceptance, anchor ----------------
        for item in small_copies:
            item.wait()
        verify_logits = small["logits"][...]  # [8, 5120] FP32 (this rank's vocabulary shard)
        vocab_iota = jax.lax.broadcasted_iota(jnp.int32, (ROWS, vocab), 1)
        local_best = jnp.max(verify_logits, axis=1, keepdims=True)  # [8, 1]
        local_id = jnp.min(jnp.where(verify_logits == local_best, vocab_iota, vocab), axis=1, keepdims=True)
        local_second = jnp.max(jnp.where(vocab_iota == local_id, NEG, verify_logits), axis=1, keepdims=True)
        out_lane = jax.lax.broadcasted_iota(jnp.int32, (ROWS, 128), 1)
        packet = jnp.where(
            out_lane == 0, jax.lax.bitcast_convert_type(local_best, jnp.int32),
            jnp.where(out_lane == 1, local_id + rank * vocab,
                      jnp.where(out_lane == 2, jax.lax.bitcast_convert_type(local_second, jnp.int32), 0)),
        )  # [8, 128] int32
        argmax_recv[rank] = packet
        exchange(lambda o: argmax_recv.at[rank], lambda o: argmax_recv.at[rank], range(1, 32), SEM_TINY, 1)
        gathered = argmax_recv[...]  # [32, 8, 128] int32
        rank_iota3 = jax.lax.broadcasted_iota(jnp.int32, (RANKS, ROWS, 1), 0)
        bests = jax.lax.bitcast_convert_type(gathered[:, :, 0:1], jnp.float32)  # [32, 8, 1]
        seconds = jax.lax.bitcast_convert_type(gathered[:, :, 2:3], jnp.float32)
        best = jnp.max(bests, axis=0)  # [8, 1]
        winner = jnp.min(jnp.where(bests == best[None], rank_iota3, RANKS), axis=0)  # [8, 1], lowest rank on ties
        targets = jnp.sum(jnp.where(rank_iota3 == winner[None], gathered[:, :, 1:2], 0), axis=0)  # [8, 1]
        runner_up = jnp.maximum(
            jnp.max(seconds, axis=0), jnp.max(jnp.where(rank_iota3 == winner[None], NEG, bests), axis=0)
        )
        margins = best - runner_up  # [8, 1] FP32
        previous_vector = small["previous"][...]  # [1, 128] int32: last step's output vector
        row_iota_col = jax.lax.broadcasted_iota(jnp.int32, (ROWS, 1), 0)
        proposals = jnp.sum(
            jnp.where(jax.lax.broadcasted_iota(jnp.int32, (ROWS, 128), 1) == row_iota_col + OUT_DRAFTS,
                      jnp.broadcast_to(previous_vector, (ROWS, 128)), 0),
            axis=1, keepdims=True,
        )  # [8, 1]: proposal k was verify row k + 1
        drafts_count = block - 1
        # matches[k] = proposals[k] == targets[k] for k < drafts_count
        match_rows = (row_iota_col < drafts_count) & (proposals == targets)  # no select on i1 vectors
        first_miss = jnp.min(jnp.where(match_rows | (row_iota_col >= drafts_count), block, row_iota_col))  # scalar
        count = jnp.where(forced_count >= 0, forced_count, jnp.minimum(first_miss, drafts_count))
        anchor_token = jnp.sum(jnp.where(row_iota_col == count, targets, 0))
        anchor_token = jnp.where(forced_anchor >= 0, forced_anchor, anchor_token)
        anchor_position = new_position + count + 1
        result_vector = jnp.zeros((1, 128), jnp.int32) - 1
        result_lane = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)
        targets_row = jnp.sum(jnp.where(result_lane == row_iota_col + OUT_TARGETS, targets, 0), axis=0, keepdims=True)
        margins_row = jnp.sum(
            jnp.where(result_lane == row_iota_col + OUT_MARGINS, jax.lax.bitcast_convert_type(margins, jnp.int32), 0),
            axis=0, keepdims=True,
        )
        result_vector = jnp.where((result_lane >= OUT_TARGETS) & (result_lane < OUT_TARGETS + ROWS), targets_row, result_vector)
        result_vector = jnp.where((result_lane >= OUT_MARGINS) & (result_lane < OUT_MARGINS + ROWS), margins_row, result_vector)
        result_vector = jnp.where(result_lane == OUT_COUNT, count, result_vector)
        result_vector = jnp.where(result_lane == OUT_BONUS, anchor_token, result_vector)
        tokens_v[...] = result_vector
        w1_copies = [None, None]

        def start_w1(token, slot):
            start = pl.multiple_of((token // 16) * 16, 16)
            item = copy(weights["w1"].at[pl.ds(start, 16)], w1_blk.at[slot], 14 + slot)
            item.start()
            return item

        w1_copies[0] = start_w1(anchor_token, 0)
        rope_q_start = pl.multiple_of((anchor_position // 8) * 8, 8)
        rope_kv_start = pl.multiple_of((new_position // 8) * 8, 8)
        rope_copies = (
            copy(weights["rope_q"].at[pl.ds(rope_q_start, 16)], rope_q_blk, 10),
            copy(weights["rope_kv"].at[pl.ds(rope_q_start, 16)], rope_kv_blk.at[0], 11),
            copy(weights["rope_kv"].at[pl.ds(rope_kv_start, 16)], rope_kv_blk.at[1], 12),
        )
        for item in rope_copies:
            item.start()
        layer_buffers = {"q": q_buf, "kv": kv_buf, "o": o_buf, "gate": gate_buf, "up": up_buf, "down": down_buf}
        layer_sems = {"q": 2, "kv": 3, "o": 4, "gate": 5, "up": 6, "down": 7}

        def layer_copy(name, layer):
            return copy(weights[name].at[layer], layer_buffers[name], layer_sems[name])

        # Anchor embedding: the owner fetches its aligned 16-row block, picks
        # the row and sends it (packed) to the 31 peers.
        owner = anchor_token // vocab
        local_id = anchor_token - owner * vocab
        embed_start = pl.multiple_of((local_id // 16) * 16, 16)

        @pl.when(rank == owner)
        def send_anchor_row():
            item = copy(weights["embed"].at[pl.ds(embed_start, 16)], embed_blk, 9)
            item.start()
            item.wait()
            row = _select_rows(_f32(embed_blk[...]), local_id % 16, 1)  # [1, 7168]
            bcast_buf[...] = pack(_bf16(row))
            copies = [remote(bcast_buf, bcast_buf, SEM_BCAST, rank ^ offset) for offset in range(1, 32)]
            for item in copies:
                item.start()
            for item in copies:
                item.wait_send()

        def early_exit(drain):
            """Exit before the cache is resident: drain, then write outputs and copy the cache through."""
            for item in drain:
                item.wait()
            for item in cache_copies:
                item.wait()
            for item in rope_copies:
                item.wait()
            hidden_v[...] = jnp.zeros((ROWS, hidden), jnp.bfloat16)
            copies = [
                tpu.make_async_copy(tokens_v, tokens_out, small_sems.at[0]),
                tpu.make_async_copy(small["positions"], positions_out, small_sems.at[1]),
                tpu.make_async_copy(hidden_v, hidden_out, small_sems.at[2]),
            ] + [copy(cache_vmem.at[layer], cache_out.at[layer], 16 + layer) for layer in range(layers)]
            for item in copies:
                item.start()
            for item in copies:
                item.wait()

            @pl.when(rank != owner)
            def receive_anchor_row_early():
                remote(bcast_buf, bcast_buf, SEM_BCAST, rank).wait_recv()

        # ---------------- fc: context features for the new rows ----------------
        if stop_after == "barrier":
            early_exit([fc_copy, w1_copies[0]])
            return
        fc_copy.wait()
        if stop_after == "fc_wait":
            early_exit([w1_copies[0]])
            return
        # Layer-0 weights and the Markov pieces stream in behind the fc phase.
        pending = {name: layer_copy(name, 0) for name in layer_buffers}
        for name in ("kv", "q", "o", "gate", "up", "down"):
            pending[name].start()
        w2t_copy = copy(weights["w2t"], w2t_buf, 8)
        w2t_copy.start()

        fc_weight = a_buf.at[:, pl.ds(0, CHUNK)]
        aux_ref = small["aux"]
        spans = []
        for h in range(4):
            start, stop = h * FC_IN, (h + 1) * FC_IN
            pieces = []
            for layer_index in range(len(c.aux_layers)):
                lo, hi = layer_index * hidden, (layer_index + 1) * hidden
                a, b = max(start, lo), min(stop, hi)
                if a < b:
                    pieces.append((layer_index, a - lo, b - lo, a - start))
            spans.append(pieces)
        for h in range(4):
            @pl.when(host == h)
            def fc_branch(h=h):
                total = None
                for layer_index, lo, hi, offset in spans[h]:
                    part = _dot(aux_ref[layer_index, :, lo:hi], fc_weight[pl.ds(offset, hi - lo), :])
                    total = part if total is None else total + part
                fc_part[...] = total

        if stop_after == "fc_matmul":
            early_exit(list(pending.values()) + [w1_copies[0], w2t_copy])
            return
        context = _rms_norm(_bf16(host_reduce_gather(fc_part[...], 0)), small["hidden_norm"][...], eps)

        # ---------------- positions, masks, RoPE ----------------
        positions = small["positions"][...]  # [1, window] int32
        slot_lane = jax.lax.broadcasted_iota(jnp.int32, (1, window), 1)
        new_slots = [(new_position + k) % window for k in range(block)]
        for k in range(block):
            positions = jnp.where(slot_lane == new_slots[k], new_position + k, positions)
        small["positions"][...] = positions
        positions_copy = tpu.make_async_copy(small["positions"], positions_out, small_sems.at[1])
        positions_copy.start()
        context_allowed = (
            (positions >= 0) & (positions < anchor_position) & (positions >= anchor_position - window)
        )  # [1, window]
        row_iota = jax.lax.broadcasted_iota(jnp.int32, (ROWS, 16), 0)
        col_iota = jax.lax.broadcasted_iota(jnp.int32, (ROWS, 16), 1)
        block_allowed = (col_iota <= row_iota) & (col_iota < block)

        for item in rope_copies:
            item.wait()
        rope_q_rows = _select_rows(rope_q_blk[...], anchor_position % 8, ROWS)  # [8, 384]
        cos_q = _f32(_bf16(rope_q_rows[:, :q_width]))
        sin_q = _f32(_bf16(rope_q_rows[:, q_width:]))
        rope_kv_rows = _select_rows(rope_kv_blk[0], anchor_position % 8, ROWS)  # [8, 256]
        cos_kv = _f32(_bf16(rope_kv_rows[:, :kv_width]))
        sin_kv = _f32(_bf16(rope_kv_rows[:, kv_width:]))
        rope_new_rows = _select_rows(rope_kv_blk[1], new_position % 8, ROWS)
        cos_new = _f32(_bf16(rope_new_rows[:, :kv_width]))
        sin_new = _f32(_bf16(rope_new_rows[:, kv_width:]))

        # ---------------- block input rows ----------------
        @pl.when(rank != owner)
        def receive_anchor_row():
            remote(bcast_buf, bcast_buf, SEM_BCAST, rank).wait_recv()

        anchor_row = unpack(bcast_buf[...], 1, hidden)  # [1, 7168] FP32 (BF16-valued)
        mask_row = _f32(small["mask_embed"][...])
        x_row = jax.lax.broadcasted_iota(jnp.int32, (ROWS, hidden), 0)
        x = _bf16(jnp.where(x_row == 0, anchor_row, mask_row))  # [8, 7168]

        for item in cache_copies:
            item.wait()

        def finish(hidden_value, drain):
            """Write outputs (early exits drain their outstanding DMAs first)."""
            for item in drain:
                if item is not None:
                    item.wait()
            tokens_copy = tpu.make_async_copy(tokens_v, tokens_out, small_sems.at[0])
            tokens_copy.start()
            hidden_v[...] = hidden_value
            hidden_copy = tpu.make_async_copy(hidden_v, hidden_out, small_sems.at[2])
            hidden_copy.start()
            write_backs = [copy(cache_vmem.at[layer], cache_out.at[layer], 16 + layer) for layer in range(layers)]
            for item in write_backs:
                item.start()
            for item in write_backs:
                item.wait()
            tokens_copy.wait()
            hidden_copy.wait()
            positions_copy.wait()

        if stop_after == "fc":
            finish(x, list(pending.values()) + [w1_copies[0], w2t_copy])
            return

        def kv_rows(features_bf16, layer, cos32, sin32):
            """Packed [8, 128] K (normed, RoPE) | V rows for eight feature rows."""
            kv = _bf16(_dot(features_bf16, kv_buf[...]))  # reference linear: BF16 output
            sumsq = _grouped_sumsq(_f32(kv), small["kv_ones"][...])  # K lanes only
            lane_index = jax.lax.broadcasted_iota(jnp.int32, kv.shape, 1)
            normed = _f32(kv) * jax.lax.rsqrt(sumsq * (1.0 / c.head_dim) + eps)
            k_scaled = _bf16(_f32(_bf16(normed)) * _f32(small["kvnorm"][layer]))
            kv = jnp.where(lane_index < c.head_dim, k_scaled, kv)
            return _rope(kv, small["rot_kv"][...], cos32, sin32)  # sin is 0 on V lanes

        def write_context_rows(layer, new_kv):
            """New rows into the ring at their slots: two aligned 128-row slabs
            cover the consecutive slots even across the wrap."""
            for slab_start in (new_slots[0] // 128 * 128, new_slots[block - 1] // 128 * 128):
                slab_start = pl.multiple_of(slab_start, 128)
                slab_ref = cache_vmem.at[layer, pl.ds(slab_start, 128)]
                slab = slab_ref[...]
                slab_rows = jax.lax.broadcasted_iota(jnp.int32, slab.shape, 0) + slab_start
                for k in range(block):
                    slab = jnp.where(slab_rows == new_slots[k], new_kv[k:k + 1, :], slab)
                slab_ref[...] = slab

        # Slot-0 Markov bias needs only the anchor's w1 row: computed during a
        # collective wait below and consumed by the sampler.
        slot0_bias = [None]

        def compute_slot0_bias():
            w1_copies[0].wait()
            w2t_copy.wait()
            w1_row = _bf16(_select_rows(_f32(w1_blk[0]), anchor_token % 16, 1))  # [1, 256]
            slot0_bias[0] = _dot(w1_row, w2t_buf[...])  # [1, 5120] FP32

        # Layer 0's context rows; later layers compute theirs inside the
        # previous layer's down-projection reduction.
        pending["kv"].wait()
        pending["kv"] = None
        write_context_rows(0, kv_rows(context, 0, cos_new, sin_new))

        # ---------------- layers ----------------
        for layer in range(layers):
            h = _rms_norm(x, small["ln1"][layer], eps)
            block_kv = kv_rows(h, layer, cos_kv, sin_kv)  # the block's own K/V (kv weights of this layer)
            if layer + 1 < layers:
                pending["kv"] = layer_copy("kv", layer + 1)
                pending["kv"].start()

            pending["q"].wait()
            q = _bf16(_dot(h, q_buf[...]))  # [8, 192]
            if layer + 1 < layers:
                pending["q"] = layer_copy("q", layer + 1)
                pending["q"].start()
            sumsq = _grouped_sumsq(_f32(q), small["head_ones"][...])
            normed = _f32(q) * jax.lax.rsqrt(sumsq * (1.0 / c.head_dim) + eps)
            q = _bf16(_f32(_bf16(normed)) * _f32(small["qnorm"][layer]))
            q = _rope(q, small["rot_q"][...], cos_q, sin_q)

            block_tile = jnp.concatenate((block_kv, jnp.zeros((16 - ROWS, kv_width), jnp.bfloat16)), axis=0)
            cache_tile = cache_vmem[layer]  # [window, 128] bf16
            attended = jnp.zeros((ROWS, q_width), jnp.float32)
            for head in range(3):
                q_head = _bf16(_dot(q, small["head_select"][head]))  # [8, 128], zeros on V lanes
                s_ctx = _attention_scores(q_head, cache_tile) * scale  # [8, window]
                s_blk = _attention_scores(q_head, block_tile) * scale  # [8, 16]
                s_ctx = jnp.where(context_allowed, s_ctx, NEG)
                s_blk = jnp.where(block_allowed, s_blk, NEG)
                maximum = jnp.maximum(jnp.max(s_ctx, axis=1, keepdims=True), jnp.max(s_blk, axis=1, keepdims=True))
                p_ctx = jnp.exp(s_ctx - maximum)
                p_blk = jnp.exp(s_blk - maximum)
                denominator = jnp.sum(p_ctx, axis=1, keepdims=True) + jnp.sum(p_blk, axis=1, keepdims=True)
                inverse = 1.0 / denominator
                out = _dot(_bf16(p_ctx * inverse), cache_tile) + _dot(_bf16(p_blk * inverse), block_tile)
                attended = attended + _dot(_bf16(out), small["head_place"][head])  # V lanes -> head lanes
            attended = _bf16(attended)

            pending["o"].wait()
            handles = all_reduce_start(attended, o_buf, 1)  # [8, 7168] partial, chunked sends
            if layer + 1 < layers:
                pending["o"] = layer_copy("o", layer + 1)
                pending["o"].start()
            if layer == 0:
                compute_slot0_bias()  # overlaps the reduction's local phase
            x = _bf16(_f32(x) + all_reduce_finish(handles, 1))

            h2 = _rms_norm(x, small["ln2"][layer], eps)
            pending["gate"].wait()
            gate = _bf16(_dot(h2, gate_buf[...]))  # [8, 512], zero beyond 448
            pending["up"].wait()
            up = _bf16(_dot(h2, up_buf[...]))
            if layer + 1 < layers:
                pending["gate"] = layer_copy("gate", layer + 1)
                pending["gate"].start()
                pending["up"] = layer_copy("up", layer + 1)
                pending["up"].start()
            gate32 = _f32(gate)
            activation = _bf16(_f32(_bf16(gate32 * jax.nn.sigmoid(gate32))) * _f32(up))
            pending["down"].wait()
            handles = all_reduce_start(activation, down_buf, 0)
            if layer + 1 < layers:
                pending["down"] = layer_copy("down", layer + 1)
                pending["down"].start()
            if layer + 1 < layers:
                # Next layer's context rows depend only on the fc features.
                pending["kv"].wait()
                pending["kv"] = None  # consumed; the next layer's block rows reuse the buffer
                write_context_rows(layer + 1, kv_rows(context, layer + 1, cos_new, sin_new))
            x = _bf16(_f32(x) + all_reduce_finish(handles, 0))
            if stop_layers == layer + 1:
                # w1/w2t were already waited on by compute_slot0_bias in layer 0.
                finish(x, list(pending.values()) if layer + 1 < layers else [])
                return

        # ---------------- LM head ----------------
        hidden_rows = _rms_norm(x, small["final_norm"][...], eps)
        chunk_buffers = (
            a_buf.at[pl.ds(0, hidden), pl.ds(0, LM_CHUNK)],
            a_buf.at[pl.ds(0, hidden), pl.ds(LM_CHUNK, LM_CHUNK)],
            gate_buf,
            up_buf,
        )
        chunk_sems = (0, 1, 5, 6)

        def chunk_copy(chunk):
            return copy(weights["lm_head"].at[chunk], chunk_buffers[chunk % 4], chunk_sems[chunk % 4])

        chunk_copies = [chunk_copy(chunk) for chunk in range(4)]
        for item in chunk_copies:
            item.start()
        vocab_lane = jax.lax.broadcasted_iota(jnp.int32, (1, vocab), 1)
        for chunk in range(LM_CHUNKS):
            chunk_copies[chunk % 4].wait()
            logits = _f32(_bf16(_dot(hidden_rows, chunk_buffers[chunk % 4][...])))  # BF16-rounded like the reference
            if chunk + 4 < LM_CHUNKS:
                chunk_copies[chunk % 4] = chunk_copy(chunk + 4)
                chunk_copies[chunk % 4].start()
            logits_vmem[:, pl.ds(chunk * LM_CHUNK, LM_CHUNK)] = logits
        if stop_after == "lm_head":
            finish(hidden_rows, [])
            return

        # ---------------- exact Markov sampling ----------------
        tokens = tokens_v[...]
        token_lane = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)
        rank_iota = jax.lax.broadcasted_iota(jnp.int32, (RANKS, 1, 128), 0)
        previous = anchor_token
        for slot in range(block):
            if slot == 0:
                bias = slot0_bias[0]
            else:
                w1_copies[slot % 2].wait()
                w1_row = _bf16(_select_rows(_f32(w1_blk[slot % 2]), previous % 16, 1))  # [1, 256]
                bias = _dot(w1_row, w2t_buf[...])  # [1, 5120] FP32
            scores = logits_vmem[slot:slot + 1, :] + bias  # [1, 5120]
            best_value = jnp.max(scores, axis=1, keepdims=True)
            best_id = jnp.min(jnp.where(scores == best_value, vocab_lane, vocab), axis=1, keepdims=True)
            local_max_bits = jax.lax.bitcast_convert_type(best_value, jnp.int32)  # [1, 1]
            global_id = best_id + rank * vocab
            row = jnp.where(token_lane == 0, local_max_bits, jnp.where(token_lane == 1, global_id, 0))
            gathered = tiny_exchange(row, slot % 2)  # [32, 1, 128] int32
            maxima = jax.lax.bitcast_convert_type(gathered[:, :, 0:1], jnp.float32)  # [32, 1, 1]
            best = jnp.max(maxima, axis=0, keepdims=True)  # [1, 1, 1]
            winner = jnp.min(jnp.where(maxima == best, rank_iota[:, :, 0:1], RANKS), axis=0, keepdims=True)
            token_row = jnp.sum(jnp.where(rank_iota == winner, gathered, 0), axis=0)  # [1, 128]; lane 1 = id
            token = token_row[:, 1:2]  # [1, 1]
            tokens = jnp.where(token_lane == OUT_DRAFTS + slot, token, tokens)
            previous = token[0, 0]
            if slot + 1 < block:
                w1_copies[(slot + 1) % 2] = start_w1(previous, (slot + 1) % 2)
        tokens_v[...] = tokens
        finish(hidden_rows, [])

    return body


def make_fused_draft(c: DraftConfig, mesh, block: int, *, eps: float | None = None, stop_after: str = "full",
                     host_bf16: bool = True):
    """``(weights, cache, positions, aux, logits, previous, position, forced_count[, forced_anchor])`` ->
    ``(vector [1, 128] int32, cache, positions, hidden [8, 7168])``.

    The kernel consumes the verify step's outputs directly: ``logits``
    ``[32, 8, vocab / 32]`` (vocabulary sharded) give the per-row targets and
    top-2 margins through one 31-peer exchange, ``previous`` is the last
    step's output vector whose drafts were the verify rows 1.., ``position``
    the verify step's first row position. The accepted count is the number
    of leading drafts equal to their targets unless ``forced_count >= 0``
    (prefill, or sampling decided outside the kernel); the bonus token (the
    next anchor) is the target at ``count`` unless ``forced_anchor >= 0`` (a
    token sampled outside the kernel). The output vector holds the new drafts (lanes 0..7), the
    targets (8..15), the count (16), the bonus token (17) and the margins as
    FP32 bits (24..31); see ``OUT_*``.
    """
    from jax.sharding import PartitionSpec as P

    if not 2 <= block <= ROWS:
        raise ValueError("block must be between 2 and 8")
    if mesh.size != RANKS:
        raise ValueError("the fused draft needs the 32-rank mesh")
    eps = c.eps if eps is None else eps
    layers, window, hidden = c.layers, c.window, c.hidden
    body = _make_body(c, block, eps, stop_after, host_bf16)

    def local(weights, cache, positions, aux, logits, previous, position, forced_count, forced_anchor):
        weights = {name: weights[name][0] for name in WEIGHT_NAMES}
        cache = cache[0]
        logits = logits[0].astype(jnp.float32)  # [8, vocab / 32]
        aux = aux.astype(jnp.bfloat16)
        scalars = jnp.stack((position, forced_count, forced_anchor) + (jnp.int32(0),) * 5).astype(jnp.int32)
        small_shapes = {"aux": aux, "positions": positions, "logits": logits, "previous": previous, **weights}
        in_specs = [pl.BlockSpec(memory_space=tpu.HBM) for _ in range(4 + len(WEIGHT_NAMES) + 1)]
        out_specs = [pl.BlockSpec(memory_space=tpu.HBM) for _ in range(4)]
        scratch = (
            *(tpu.VMEM(small_shapes[name].shape, small_shapes[name].dtype) for name in SMALL_NAMES),
            tpu.VMEM((FC_IN, 1024), jnp.bfloat16),  # a_buf: fc [8960, 896], then LM chunks 0/1
            tpu.VMEM((hidden, 3 * c.head_dim), jnp.bfloat16),  # q_buf
            tpu.VMEM((hidden, 2 * c.head_dim), jnp.bfloat16),  # kv_buf
            tpu.VMEM((3 * c.head_dim, hidden), jnp.bfloat16),  # o_buf
            tpu.VMEM((hidden, LM_CHUNK), jnp.bfloat16),  # gate_buf: MLP lanes 0..447, or an LM chunk
            tpu.VMEM((hidden, LM_CHUNK), jnp.bfloat16),  # up_buf
            tpu.VMEM((MLP_PADDED, hidden), jnp.bfloat16),  # down_buf
            tpu.VMEM((c.markov_rank, c.vocab // RANKS), jnp.bfloat16),  # w2t_buf [256, 5120]
            tpu.VMEM((2, 16, c.markov_rank), jnp.bfloat16),  # w1_blk
            tpu.VMEM((16, hidden), jnp.bfloat16),  # embed_blk
            tpu.VMEM((ROW_PACKED, 128), jnp.float32),  # bcast_buf
            tpu.VMEM((16, 6 * c.head_dim), jnp.float32),  # rope_q_blk
            tpu.VMEM((2, 16, 4 * c.head_dim), jnp.float32),  # rope_kv_blk
            tpu.VMEM((layers, window, 2 * c.head_dim), jnp.bfloat16),  # cache_vmem
            tpu.VMEM((ROWS, c.vocab // RANKS), jnp.float32),  # logits_vmem [8, 5120]
            tpu.VMEM((ROWS, CHUNK), jnp.float32),  # fc_part
            tpu.VMEM((1, 128), jnp.int32),  # tokens_v
            tpu.VMEM((ROWS, hidden), jnp.bfloat16),  # hidden_v
            tpu.VMEM((2, 8, PACKED, 128), jnp.float32),  # stage
            tpu.VMEM((2, 8, PACKED, 128), jnp.float32),  # local_recv
            tpu.VMEM((2, 4, ROWS, CHUNK), jnp.float32),  # host_buf
            tpu.VMEM((2, 4, PACKED, 128), jnp.float32),  # host_packed
            tpu.VMEM((2, 8, PACKED, 128), jnp.float32),  # gather_recv
            tpu.VMEM((2, RANKS, 1, 128), jnp.int32),  # tiny_recv
            tpu.VMEM((RANKS, ROWS, 128), jnp.int32),  # argmax_recv
            tpu.SemaphoreType.DMA((24,)),
            tpu.SemaphoreType.DMA((len(SMALL_NAMES),)),
            tpu.SemaphoreType.DMA((2 * SEMS_PER_SLOT,)),
            tpu.SemaphoreType.DMA((2 * SEMS_PER_SLOT,)),
        )
        cache_index = 1 + 4 + len(WEIGHT_NAMES)  # scalars, aux, positions, logits, previous, weights
        tokens, positions_out, cache_out, hidden_out = pl.pallas_call(
            body,
            name="dspark_fused_draft",
            out_shape=(
                jax.ShapeDtypeStruct((1, 128), jnp.int32),
                jax.ShapeDtypeStruct((1, window), jnp.int32),
                jax.ShapeDtypeStruct(cache.shape, cache.dtype),
                jax.ShapeDtypeStruct((ROWS, hidden), jnp.bfloat16),
            ),
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1, in_specs=in_specs, out_specs=out_specs, scratch_shapes=scratch,
            ),
            input_output_aliases={cache_index: 2},
            compiler_params=tpu.CompilerParams(collective_id=25, vmem_limit_bytes=64 * 1024**2),
        )(scalars, aux, positions, logits, previous, *[weights[name] for name in WEIGHT_NAMES], cache)
        return tokens, cache_out[None], positions_out, hidden_out

    program = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P("tp"), P("tp"), P(), P(), P("tp"), P(), P(), P(), P()),
            out_specs=(P(), P("tp"), P(), P()),
            check_vma=False,
        ),
        donate_argnums=(1,),
    )

    def fused_draft(weights, cache, positions, aux, logits, previous, position, forced_count, forced_anchor=None):
        if forced_anchor is None:
            forced_anchor = jnp.int32(-1)
        return program(weights, cache, positions, aux, logits, previous, position, forced_count, forced_anchor)

    return fused_draft
