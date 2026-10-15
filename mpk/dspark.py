"""Muse Spark DSpark with BF16 draft compute and exact greedy verification."""

import collections
from dataclasses import dataclass
import json
from pathlib import Path
import struct
import time

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import NamedSharding, PartitionSpec as P

import pool_alias


@dataclass(frozen=True)
class DraftConfig:
    hidden: int = 8192
    layers: int = 5
    heads: int = 128
    kv_heads: int = 16
    head_dim: int = 64
    intermediate: int = 16384
    vocab: int = 202048
    aux_layers: tuple[int, ...] = (13, 29, 45, 57, 61)
    mask_token: int = 200018
    window: int = 2048
    rope_theta: float = 500000.0
    eps: float = 1e-5
    post_eps: float = 1e-8
    markov_rank: int = 512
    draft_tokens: int = 7
    output_multiplier: float = 0.17677669529663687
    soft_cap: float = 20.0

    @classmethod
    def from_checkpoint(cls, directory):
        raw = json.loads((Path(directory) / "config.json").read_text())
        required = {
            "model_type": "muse_spark_dspark", "sample_from_anchor": True,
            "norm_zero_centered_gamma": True, "use_residual_gate": True,
            "use_qk_norm": True, "use_attn_o_norm": True, "use_attn_o_gate": True,
            "use_ffn_learnable_scales": True, "residual_dtype": "fp32",
        }
        for key, value in required.items():
            if raw.get(key) != value:
                raise ValueError(f"unsupported DSpark setting {key}={raw.get(key)!r}")
        if raw["target_hidden_size"] != raw["hidden_size"] or raw["draft_vocab_size"] != raw["vocab_size"]:
            raise ValueError("DSpark requires matching target width and full draft vocabulary")
        return cls(
            hidden=raw["hidden_size"], layers=raw["num_hidden_layers"],
            heads=raw["num_attention_heads"], kv_heads=raw["num_key_value_heads"],
            head_dim=raw["head_dim"], intermediate=raw["intermediate_size_mlp"],
            vocab=raw["vocab_size"], aux_layers=tuple(raw["target_layer_ids"]),
            mask_token=raw["mask_token_id"], window=raw["sliding_window"],
            rope_theta=raw["rope_theta"], eps=raw["rms_norm_eps"],
            post_eps=raw["rms_post_norm_eps"], markov_rank=raw["markov_rank"],
            draft_tokens=raw["block_size"],
            output_multiplier=(raw["hidden_size"] / raw["metap"]["base_width"]) ** -0.5,
            soft_cap=raw["output_soft_cap_temp"],
        )


def norm(x, eps, gain=None):
    x = x.astype(jnp.float32)
    value = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return value if gain is None else value * gain.astype(jnp.float32)


def rope(x, positions, theta):
    width = x.shape[-1]
    angle = positions.astype(jnp.float32)[:, None] / theta ** (jnp.arange(0, width, 2, dtype=jnp.float32) / width)
    cosine, sine = jnp.cos(angle)[:, None], jnp.sin(angle)[:, None]
    even, odd = x[..., ::2].astype(jnp.float32), x[..., 1::2].astype(jnp.float32)
    return jnp.stack((even * cosine - odd * sine, odd * cosine + even * sine), axis=-1).reshape(x.shape).astype(jnp.bfloat16)


def acceptance(proposals, targets, fixed_length=-1):
    """AL includes the bonus token; fixed lengths are synthetic benchmarks only."""
    count = jnp.sum(jnp.cumprod((proposals == targets[:-1]).astype(jnp.int32)))
    count = jnp.where(fixed_length > 0, fixed_length - 1, count)
    bonus = targets[count]
    slots = jnp.arange(targets.shape[0])
    emitted = jnp.where(slots < count, jnp.pad(proposals, (0, 1)), jnp.where(slots == count, bonus, -1))
    return count, bonus, emitted


def load_weights(directory, config, mesh):
    """Read the existing checkpoint on compute hosts, placing only local shards."""
    path = Path(directory) / "model.safetensors"
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))

    def read(name):
        entry = header[name]
        dtype = {"BF16": np.uint16, "I64": np.int64, "F32": np.float32}[entry["dtype"]]
        start, end = entry["data_offsets"]
        array = np.fromfile(path, dtype=dtype, count=(end - start) // np.dtype(dtype).itemsize, offset=8 + size + start)
        if entry["dtype"] == "BF16":
            array = array.view(ml_dtypes.bfloat16)
        return array.reshape(entry["shape"])

    c, tp = config, mesh.size
    devices = list(mesh.devices.flat)
    local_ranks = [r for r, d in enumerate(devices) if d.process_index == jax.process_index()]
    sharding = NamedSharding(mesh, P("tp"))

    def place(function):
        arrays = [jax.device_put(np.ascontiguousarray(function(r)[None]), jax.sharding.SingleDeviceSharding(devices[r])) for r in local_ranks]
        shape = (tp, *arrays[0].shape[1:])
        return jax.make_array_from_single_device_arrays(shape, sharding, arrays)

    def columns(value, width):
        return place(lambda r: value[..., r * width:(r + 1) * width, :].swapaxes(-1, -2))

    weights = {}
    for name in ("q", "k", "v", "o_gate", "o"):
        value = np.stack([read(f"layers.{i}.self_attn.{name}_proj.weight") for i in range(c.layers)])
        if name in ("k", "v"):
            kvh = max(c.kv_heads // tp, 1)
            weights[name] = place(lambda r: value[:, (r * c.kv_heads // tp) * c.head_dim:(r * c.kv_heads // tp + kvh) * c.head_dim, :].swapaxes(-1, -2))
        elif name == "o":
            weights[name] = place(lambda r: value[:, :, r * c.hidden // tp:(r + 1) * c.hidden // tp].swapaxes(-1, -2))
        else:
            weights[name] = columns(value, c.heads * c.head_dim // tp)
    for name in ("gate", "up", "down"):
        value = np.stack([read(f"layers.{i}.feed_forward.{name}_proj.weight") for i in range(c.layers)])
        if name == "down":
            weights[name] = place(lambda r: value[:, :, r * c.intermediate // tp:(r + 1) * c.intermediate // tp].swapaxes(-1, -2))
        else:
            weights[name] = columns(value, c.intermediate // tp)
    for name, source in (("pre_attn", "self_attn.pre_qkv_proj_norm.weight"), ("pre_ffn", "pre_ffn_norm.weight"), ("post_ffn", "post_ffn_norm.weight")):
        value = np.stack([read(f"layers.{i}.{source}").astype(np.float32) + 1 for i in range(c.layers)])
        weights[name] = place(lambda r: value)
    for branch in ("attn", "ffn"):
        value = np.stack([read(f"layers.{i}.add_{branch}.gate").astype(np.float32) / 0.3 for i in range(c.layers)])
        beta = 1 / (1 + np.exp(-value))
        alpha = np.sqrt(np.maximum(1e-3, (1 / (1 + np.exp(value))) * (1 + beta)))
        weights[f"alpha_{branch}"] = place(lambda r: alpha)
        weights[f"beta_{branch}"] = place(lambda r: beta)
    for name in ("hidden_norm", "norm"):
        value = read(f"{name}.weight").astype(np.float32) + 1
        weights[name] = place(lambda r: value)
    value = read("fc.weight")
    weights["fc"] = columns(value, c.hidden // tp)
    value = read("lm_head.weight")
    weights["lm"] = columns(value, c.vocab // tp)
    value = read("markov_head.markov_w2.weight")
    weights["markov_w2"] = columns(value, c.vocab // tp)
    value = read("markov_head.markov_w1.weight")
    weights["markov_w1"] = place(lambda r: value)
    value = read("d2t").astype(np.int32)
    mapped = value + np.arange(c.vocab, dtype=np.int32)
    if np.any(mapped < 0) or np.any(mapped >= c.vocab) or len(np.unique(mapped)) != c.vocab:
        raise ValueError("invalid draft-to-target vocabulary mapping")
    weights["d2t"] = place(lambda r: value)
    jax.block_until_ready(weights)
    return weights


def _make_context_gather(c, mesh):
    tp = mesh.size
    rows, width = c.draft_tokens + 1, c.hidden // mesh.size

    def gather_body(value, out, combined, sends, recvs):
        rank = jax.lax.axis_index('tp')
        host = rank // 8
        barrier = tpu.get_barrier_semaphore()
        for offset in range(1, tp):
            pl.semaphore_signal(barrier, 1, device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(barrier, tp - 1)
        combined[rank] = value[...].reshape(rows * width // 128, 128)

        def remote(source, target, channel, peer):
            return tpu.make_async_remote_copy(source, target, sends.at[channel], recvs.at[channel], device_id=(peer,), device_id_type=pl.DeviceIdType.MESH)

        transfers = [remote(combined.at[rank], combined.at[rank], i, rank ^ offset) for i, offset in enumerate(range(7, 0, -1))]
        for transfer in transfers:
            transfer.start()
        for transfer in transfers:
            transfer.wait()
        transfers = [remote(combined.at[pl.ds(host * 8, 8)], combined.at[pl.ds(host * 8, 8)], 7 + i, rank ^ offset) for i, offset in enumerate(range(tp - 8, 0, -8))]
        for transfer in transfers:
            transfer.start()
        for transfer in transfers:
            transfer.wait()
        out[...] = jnp.concatenate([combined[i].reshape(rows, width) for i in range(tp)], axis=1)

    gather = pl.pallas_call(
        gather_body, out_shape=jax.ShapeDtypeStruct((rows, c.hidden), jnp.bfloat16),
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0, in_specs=[pl.BlockSpec()], out_specs=pl.BlockSpec(),
            scratch_shapes=(tpu.VMEM((tp, rows * width // 128, 128), jnp.bfloat16), tpu.SemaphoreType.DMA((10,)), tpu.SemaphoreType.DMA((10,))),
        ), compiler_params=tpu.CompilerParams(collective_id=55, shape_invariant_numerics=True),
        name='muse_dspark_context_gather',
    )

    return gather


def _make_cache_update(c, kvh):
    def cache_write(control, previous, delta, old_positions, updated, new_positions, block, semaphores):
        start = pl.multiple_of(control[0] // 16 * 16 % c.window, 16)
        second = pl.multiple_of((start + 16) % c.window, 16)
        transfers = [tpu.make_async_copy(previous.at[:, pl.ds(offset, 16)], block.at[:, pl.ds(i * 16, 16)], semaphores.at[i]) for i, offset in enumerate((start, second))]
        for transfer in transfers:
            transfer.start()
        for transfer in transfers:
            transfer.wait()
        values = block[...]
        slots = (start + jnp.arange(32, dtype=jnp.int32))[None, :, None] % c.window
        cached_positions = old_positions[...]
        for row in range(c.draft_tokens + 1):
            valid = (control[c.draft_tokens + 1 + row] != 0) & (control[row] >= 0)
            selected = valid & (slots == control[row] % c.window)
            values = jnp.where(selected, delta[:, row:row + 1], values)
            cached_positions = jnp.where(valid & (jnp.arange(c.window)[None] == control[row] % c.window), control[row], cached_positions)
        block[...] = values
        new_positions[...] = cached_positions
        transfers = [tpu.make_async_copy(block.at[:, pl.ds(i * 16, 16)], updated.at[:, pl.ds(offset, 16)], semaphores.at[i]) for i, offset in enumerate((start, second))]
        for transfer in transfers:
            transfer.start()
        for transfer in transfers:
            transfer.wait()

    def write_delta(previous, delta, positions, valid, cached_positions):
        result, cached_positions = pl.pallas_call(
            cache_write,
            out_shape=(jax.ShapeDtypeStruct(previous.shape, previous.dtype), jax.ShapeDtypeStruct((1, c.window), jnp.int32)),
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[pl.BlockSpec(memory_space=tpu.HBM), pl.BlockSpec(), pl.BlockSpec()],
                out_specs=[pl.BlockSpec(memory_space=tpu.HBM), pl.BlockSpec()],
                scratch_shapes=(tpu.VMEM((c.layers, 32, kvh * 128), jnp.bfloat16), tpu.SemaphoreType.DMA((2,))),
            ),
            input_output_aliases={1: 0, 3: 1}, compiler_params=tpu.CompilerParams(shape_invariant_numerics=True),
            name="muse_dspark_cache_update",
        )(jnp.concatenate((positions, valid.astype(jnp.int32))), previous, delta, cached_positions[None])
        return result, cached_positions[0]

    return write_delta


def make_programs(c, mesh, *, fused_update=False, packed_cache=False):
    tp = mesh.size
    context_gather = _make_context_gather(c, mesh) if fused_update else None
    qh, kvh = c.heads // tp, max(c.kv_heads // tp, 1)
    if c.heads % tp or c.hidden % tp or c.intermediate % tp or c.vocab % tp or qh % kvh:
        raise ValueError("draft dimensions must partition across the mesh")
    sharding = NamedSharding(mesh, P("tp"))

    cache_spec = (P("tp"), P()) if packed_cache else (P("tp"), P("tp"), P())
    cache_shardings = (sharding, NamedSharding(mesh, P())) if packed_cache else (sharding, sharding, NamedSharding(mesh, P()))
    permutation = jnp.concatenate((jnp.arange(0, c.head_dim, 2), jnp.arange(1, c.head_dim, 2)))

    @jax.jit(out_shardings=cache_shardings)
    def empty_cache():
        if packed_cache:
            return jnp.zeros((tp, c.layers, c.window, kvh * 128), jnp.bfloat16), jnp.full((c.window,), -1, jnp.int32)
        shape = (tp, c.layers, c.window, kvh, c.head_dim)
        return jnp.zeros(shape, jnp.bfloat16), jnp.zeros(shape, jnp.bfloat16), jnp.full((c.window,), -1, jnp.int32)

    write_delta = _make_cache_update(c, kvh) if packed_cache else None

    def matmul(x, w):
        return jnp.dot(x, w, preferred_element_type=jnp.float32).astype(jnp.bfloat16)

    def update_local(w, cache, aux, positions, valid):
        w = jax.tree.map(lambda a: a[0], w)
        if packed_cache:
            kv, cached_positions = cache
            kv = kv[0]
        else:
            kc, vc, cached_positions = cache
            kc, vc = kc[0], vc[0]
        context = matmul(aux.reshape(aux.shape[0], -1).astype(jnp.bfloat16), w["fc"])
        context = context_gather(context) if fused_update and aux.shape[0] == c.draft_tokens + 1 else jax.lax.all_gather(context, "tp", axis=-1, tiled=True)
        context = norm(context, c.eps, w["hidden_norm"]).astype(jnp.bfloat16)
        slots = jnp.where(valid & (positions >= 0), positions % c.window, c.window)

        def project(i):
            x = norm(context, c.eps, w["pre_attn"][i]).astype(jnp.bfloat16)
            k = matmul(x, w["k"][i]).reshape(-1, kvh, c.head_dim)
            k = rope(norm(k, c.eps).astype(jnp.bfloat16), positions, c.rope_theta)
            v = matmul(x, w["v"][i]).reshape(-1, kvh, c.head_dim)
            return k, v

        if packed_cache:
            def packed_rows(i):
                k, v = project(i)
                return jnp.concatenate((k[..., permutation], v), axis=-1).reshape(-1, kvh * 128)
            if fused_update and aux.shape[0] == c.draft_tokens + 1:
                delta = jnp.stack([packed_rows(i) for i in range(c.layers)])
                kv, cached_positions = write_delta(kv, delta, positions, valid, cached_positions)
            else:
                kv = jax.lax.fori_loop(0, c.layers, lambda i, kv: kv.at[i, slots].set(packed_rows(i), mode="drop"), kv)
                cached_positions = cached_positions.at[slots].set(positions, mode="drop")
            return kv[None], cached_positions

        def layer(i, state):
            kc, vc = state
            k, v = project(i)
            return kc.at[i, slots].set(k, mode="drop"), vc.at[i, slots].set(v, mode="drop")

        kc, vc = jax.lax.fori_loop(0, c.layers, layer, (kc, vc), unroll=fused_update and aux.shape[0] == c.draft_tokens + 1)
        cached_positions = cached_positions.at[slots].set(positions, mode="drop")
        return kc[None], vc[None], cached_positions

    update = jax.jit(jax.shard_map(
        update_local, mesh=mesh,
        in_specs=(P("tp"), cache_spec, P(), P(), P()),
        out_specs=cache_spec, check_vma=False,
    ), donate_argnums=(1,))

    def draft_local(w, embed, cache, anchor, position):
        w = jax.tree.map(lambda a: a[0], w)
        embed = embed[0]
        if packed_cache:
            kv, cached_positions = cache
            kv = kv[0].reshape(c.layers, c.window, kvh, 128)
            kc, vc = kv[..., :64][..., jnp.argsort(permutation)], kv[..., 64:]
        else:
            kc, vc, cached_positions = cache
            kc, vc = kc[0], vc[0]
        rank = jax.lax.axis_index("tp")
        n = c.draft_tokens
        tokens = jnp.full((n,), c.mask_token, jnp.int32).at[0].set(anchor)
        if embed.shape[0] == c.vocab:
            stream = embed[tokens].astype(jnp.float32)
        else:
            offsets = tokens - rank * (c.vocab // tp)
            stream = embed[jnp.clip(offsets, 0, c.vocab // tp - 1)]
            stream = jax.lax.psum(jnp.where(((offsets >= 0) & (offsets < c.vocab // tp))[:, None], stream.astype(jnp.float32), 0), "tp")
        positions = position + jnp.arange(n, dtype=jnp.int32)
        all_positions = jnp.concatenate((cached_positions, positions))
        keep = (all_positions[None] >= 0) & (all_positions[None] <= positions[:, None]) & (all_positions[None] > positions[:, None] - c.window)
        keep &= jnp.concatenate((cached_positions < position, jnp.ones((n,), jnp.bool_)))[None]

        def layer(i, stream):
            x = norm(stream.astype(jnp.bfloat16), c.eps, w["pre_attn"][i]).astype(jnp.bfloat16)
            q = matmul(x, w["q"][i]).reshape(n, qh, c.head_dim)
            k = matmul(x, w["k"][i]).reshape(n, kvh, c.head_dim)
            v = matmul(x, w["v"][i]).reshape(n, kvh, c.head_dim)
            gate = matmul(x, w["o_gate"][i]).reshape(n, qh, c.head_dim)
            q, k = rope(norm(q, c.eps), positions, c.rope_theta), rope(norm(k, c.eps), positions, c.rope_theta)
            keys = jnp.concatenate((kc[i], k))
            values = jnp.concatenate((vc[i], v))
            q = q.reshape(n, kvh, qh // kvh, c.head_dim)
            scores = jnp.einsum("qghd,kgd->ghqk", q, keys, preferred_element_type=jnp.float32) * (43.7840518911 / c.head_dim)
            probs = jax.nn.softmax(jnp.where(keep[None, None], scores, -jnp.inf), axis=-1).astype(jnp.bfloat16)
            attended = jnp.einsum("ghqk,kgd->qghd", probs, values, preferred_element_type=jnp.float32).reshape(n, qh, c.head_dim).astype(jnp.bfloat16)
            gated = (norm(attended, c.eps) * jax.nn.sigmoid(gate.astype(jnp.float32))).astype(jnp.bfloat16).reshape(n, -1)
            branch = jax.lax.psum(jnp.dot(gated, w["o"][i], preferred_element_type=jnp.float32), "tp").astype(jnp.bfloat16)
            branch = norm(branch, c.post_eps).astype(jnp.bfloat16)
            stream = w["alpha_attn"][i] * stream + w["beta_attn"][i] * branch.astype(jnp.float32)
            x = norm(stream.astype(jnp.bfloat16), c.eps, w["pre_ffn"][i]).astype(jnp.bfloat16)
            gate, up = matmul(x, w["gate"][i]), matmul(x, w["up"][i])
            act = (jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)).astype(jnp.bfloat16)
            branch = jax.lax.psum(jnp.dot(act, w["down"][i], preferred_element_type=jnp.float32), "tp").astype(jnp.bfloat16)
            branch = norm(branch.astype(jnp.float32) * w["post_ffn"][i], c.post_eps).astype(jnp.bfloat16)
            return w["alpha_ffn"][i] * stream + w["beta_ffn"][i] * branch.astype(jnp.float32)

        stream = jax.lax.fori_loop(0, c.layers, layer, stream)
        hidden = norm(stream.astype(jnp.bfloat16), c.eps, w["norm"]).astype(jnp.bfloat16)
        base = matmul(hidden, w["lm"])

        def sample(previous, logits):
            bias = matmul(w["markov_w1"][previous][None], w["markov_w2"])[0]
            values = (logits + bias).astype(jnp.float32) * c.output_multiplier
            values = c.soft_cap * jnp.tanh(values / c.soft_cap)
            local_top = jnp.max(values)
            top = jax.lax.pmax(local_top, "tp")
            index = jnp.argmax(values).astype(jnp.int32) + rank * (c.vocab // tp)
            index = jax.lax.pmin(jnp.where(local_top == top, index, c.vocab), "tp")
            token = index + w["d2t"][index]
            return token, token

        _, proposals = jax.lax.scan(sample, anchor, base)
        return proposals

    draft = jax.jit(jax.shard_map(
        draft_local, mesh=mesh,
        in_specs=(P("tp"), P("tp"), cache_spec, P(), P()),
        out_specs=P(), check_vma=False,
    ))
    return empty_cache, update, draft


SMALL = ("pre_attn", "pre_ffn", "post_ffn", "alpha_attn", "beta_attn", "alpha_ffn", "beta_ffn", "norm")
LARGE = ("qgkv", "o", "gu", "down", "lm", "markov_w1", "markov_w2")


def prepare_weights(c, mesh, weights):
    shard = NamedSharding(mesh, P("tp"))
    padded = (c.vocab // mesh.size + 511) // 512 * 512
    permutation = jnp.concatenate((jnp.arange(0, c.head_dim, 2), jnp.arange(1, c.head_dim, 2)))

    @jax.jit(out_shardings=shard)
    def prepare(w):
        def reorder(a):
            shape = a.shape
            return a.reshape(*shape[:-1], -1, c.head_dim)[..., permutation].reshape(shape)
        return {
            **{key: w[key] for key in SMALL},
            "qgkv": jnp.concatenate((reorder(w["q"]), w["o_gate"], reorder(w["k"]), w["v"]), axis=-1),
            "o": w["o"], "gu": jnp.concatenate((w["gate"], w["up"]), axis=-1),
            "down": w["down"], "markov_w1": w["markov_w1"],
            "lm": jnp.pad(w["lm"], ((0, 0), (0, 0), (0, padded - c.vocab // mesh.size))),
            "markov_w2": jnp.pad(w["markov_w2"], ((0, 0), (0, 0), (0, padded - c.vocab // mesh.size))),
        }

    if int(jax.device_get(jnp.max(jnp.abs(weights["d2t"])))) != 0:
        raise ValueError("the fused Muse draft requires the checkpoint's identity vocabulary mapping")
    return prepare(weights)


def make_draft(c, mesh, *, packed_cache=False):
    tp, hosts = mesh.size, mesh.size // 8
    if tp not in (8, 16, 32) or c.head_dim != 64 or c.kv_heads != 16 or c.hidden != 8192:
        raise ValueError("fused draft requires the Muse DSpark TP8, TP16, or TP32 geometry")
    rows, hidden, qwidth = 8, c.hidden, c.heads // tp * c.head_dim
    width, vocab = c.intermediate // tp, c.vocab // tp
    row_shape = (hidden // 128, 128) if tp <= 16 else (1, hidden)
    chunk = 512
    padded = (vocab + chunk - 1) // chunk * chunk
    kvh = max(c.kv_heads // tp, 1)
    tile_width = min(width, 512) if tp == 8 else width
    gu_shape = (hidden, 2 * tile_width)
    lm_shape = (2, hidden, chunk)
    q_shape, o_shape = (hidden, 2 * qwidth + kvh * 128), (qwidth, hidden)
    projection_parts = 2 if tp == 8 else 1
    o_shape = (o_shape[0], o_shape[1] // projection_parts)
    projection_tiles = max(pool_alias.storage_size(shape, jnp.dtype(jnp.bfloat16)) for shape in ((o_shape, (tile_width, hidden)) if tp == 8 else (o_shape, (width, hidden), (c.markov_rank, padded)))) // 4096
    arena_tiles = max(pool_alias.storage_size(shape, jnp.dtype(jnp.bfloat16)) for shape in ((gu_shape, lm_shape, q_shape, (c.markov_rank, padded)) if tp == 8 else (gu_shape, lm_shape, q_shape) if tp == 16 else (gu_shape, lm_shape))) // 4096

    def body(meta, *refs):
        small = dict(zip(SMALL, refs[:len(SMALL)]))
        w = dict(zip(LARGE, refs[len(SMALL):len(SMALL) + len(LARGE)]))
        (cache, positions, embed, out, qbuf, obuf, arena, dbuf, w2buf, embedbuf, maskbuf,
         w1buf, logitsbuf, tokensbuf, stage, local_recv, host_stage, host_recv, reduced, tiny,
         dsem, sends, recvs) = refs[len(SMALL) + len(LARGE):]
        gu_tiles = pool_alias.storage_size(gu_shape, jnp.dtype(jnp.bfloat16)) // 4096
        gubuf = pool_alias.view(arena.at[:gu_tiles] if tp <= 16 else arena, gu_shape, jnp.bfloat16)
        lm_tiles = pool_alias.storage_size(lm_shape, jnp.dtype(jnp.bfloat16)) // 4096
        lmbuf = pool_alias.view(arena.at[:lm_tiles] if tp <= 16 else arena, lm_shape, jnp.bfloat16)
        if tp <= 16:
            q_tiles = pool_alias.storage_size(q_shape, jnp.dtype(jnp.bfloat16)) // 4096
            qbuf = pool_alias.view(arena.at[:q_tiles], q_shape, jnp.bfloat16)
            projection = obuf
            o_tiles = pool_alias.storage_size(o_shape, jnp.dtype(jnp.bfloat16)) // 4096
            down_tiles = pool_alias.storage_size((tile_width, hidden), jnp.dtype(jnp.bfloat16)) // 4096
            markov_tiles = pool_alias.storage_size((c.markov_rank, padded), jnp.dtype(jnp.bfloat16)) // 4096
            obuf = pool_alias.view(projection.at[:o_tiles], o_shape, jnp.bfloat16)
            dbuf = pool_alias.view(projection.at[:down_tiles], (tile_width, hidden), jnp.bfloat16)
            w2buf = pool_alias.view((arena if tp == 8 else projection).at[:markov_tiles], (c.markov_rank, padded), jnp.bfloat16)
        rank = jax.lax.axis_index("tp")
        lane, host = rank % 8, rank // 8
        anchor, position = meta[0], meta[1]

        def copy(src, dst, channel):
            return tpu.make_async_copy(src, dst, dsem.at[channel])

        def remote(src, dst, channel, peer):
            return tpu.make_async_remote_copy(src, dst, sends.at[channel], recvs.at[channel], device_id=(peer,), device_id_type=pl.DeviceIdType.MESH)

        def allreduce(value, slot):
            stage[slot] = value.reshape(rows, *row_shape)
            local_recv[slot, lane] = stage[slot, lane]
            transfers = [remote(stage.at[slot, lane ^ offset], local_recv.at[slot, lane], slot * 10 + offset - 1, rank ^ offset) for offset in range(7, 0, -1)]
            for transfer in transfers:
                transfer.start()
            for transfer in transfers:
                transfer.wait()
            partial = jnp.sum(local_recv[slot], axis=0)
            host_stage[slot] = partial
            host_recv[slot, host] = partial
            transfers = [remote(host_stage.at[slot], host_recv.at[slot, host], slot * 10 + 6 + offset, rank ^ (offset * 8)) for offset in range(hosts - 1, 0, -1)]
            for transfer in transfers:
                transfer.start()
            for transfer in transfers:
                transfer.wait()
            reduced[slot, lane] = jnp.sum(host_recv[slot], axis=0).astype(jnp.bfloat16)
            transfers = [remote(reduced.at[slot, lane], reduced.at[slot, lane], slot * 10 + offset - 1, rank ^ offset) for offset in range(7, 0, -1)]
            for transfer in transfers:
                transfer.start()
            for transfer in transfers:
                transfer.wait()
            return reduced[slot].reshape(rows, hidden)

        def norm(value, gain=None, eps=c.eps):
            value = value.astype(jnp.float32)
            value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + eps)
            return value if gain is None else value * gain

        def head_norm(value):
            value = value.astype(jnp.float32)
            return value * jax.lax.rsqrt(jnp.sum(value * value, axis=-1, keepdims=True) / c.head_dim + c.eps)

        def dot(x, y):
            return jnp.dot(x, y, preferred_element_type=jnp.float32)

        def gu_copy(i, part, wait=False):
            if wait:
                copy(w["gu"].at[i, :, :2 * tile_width], gubuf, 2).wait()
            else:
                for half in range(2):
                    copy(w["gu"].at[i, :, pl.ds(half * width + part * tile_width, tile_width)],
                         gubuf.at[:, pl.ds(half * tile_width, tile_width)], 2).start()

        def down_copy(i, part):
            return copy(w["down"].at[i, pl.ds(part * tile_width, tile_width)], dbuf, 3)

        def layer_copy(name, i, part=0):
            buf, channel = {"qgkv": (qbuf, 0), "o": (obuf, 1), "gu": (gubuf, 2), "down": (dbuf, 3)}[name]
            source = w[name].at[i, :, pl.ds(part * buf.shape[1], buf.shape[1])] if tp == 8 and name in ("qgkv", "o") else w[name].at[i]
            return copy(source, buf, channel)

        barrier = tpu.get_barrier_semaphore()
        for offset in range(1, tp):
            pl.semaphore_signal(barrier, 1, device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(barrier, tp - 1)
        for name in (("qgkv", "o") if tp <= 16 else ("qgkv", "o", "gu", "down")):
            layer_copy(name, 0).start()
        if tp == 32:
            copy(w["markov_w2"], w2buf, 4).start()
        rindex = jnp.arange(rows, dtype=jnp.int32)[:, None]
        if tp <= 16:
            copy(w["markov_w1"].at[pl.ds(anchor // 8 * 8, 8), :], w1buf, 7).start()
            joined_embed = allreduce(embed[...].astype(jnp.float32), 0)
            anchor_row, mask_row = joined_embed[:1].astype(jnp.float32), joined_embed[1:2].astype(jnp.float32)
        else:
            copy(embed.at[pl.ds(anchor // 8 * 8, 8), :], embedbuf, 5).start()
            copy(embed.at[pl.ds(c.mask_token // 8 * 8, 8), :], maskbuf, 6).start()
            copy(w["markov_w1"].at[pl.ds(anchor // 8 * 8, 8), :], w1buf, 7).start()
            copy(embed.at[pl.ds(anchor // 8 * 8, 8), :], embedbuf, 5).wait()
            copy(embed.at[pl.ds(c.mask_token // 8 * 8, 8), :], maskbuf, 6).wait()
            anchor_row = jnp.sum(jnp.where(rindex == anchor % 8, embedbuf[...].astype(jnp.float32), 0), axis=0, keepdims=True)
            mask_row = jnp.sum(jnp.where(rindex == c.mask_token % 8, maskbuf[...].astype(jnp.float32), 0), axis=0, keepdims=True)
        stream = jnp.where(jnp.broadcast_to(rindex, (rows, hidden)) == 0, anchor_row, mask_row)
        columns = jnp.arange(128, dtype=jnp.int32)[None].astype(jnp.float32)
        inv = c.rope_theta ** (-(columns % 32) / 32)
        angle = (position.astype(jnp.float32) + rindex.astype(jnp.float32)) * inv
        cosine, sine = jnp.cos(angle), jnp.sin(angle)
        context_positions = positions[...]
        context_keep = (context_positions >= 0) & (context_positions < position)
        query_positions = position + rindex
        context_keep = context_keep & (context_positions > query_positions - c.window)
        block_keep = (jnp.arange(16)[None] <= rindex) & (jnp.arange(16)[None] < c.draft_tokens)

        def rotate(value):
            rotated = jnp.concatenate((-value[:, 32:64], value[:, :32], jnp.zeros((rows, 64), jnp.float32)), axis=1)
            return (value * cosine + rotated * sine).astype(jnp.bfloat16)

        def layer(li, stream):
            x = norm(stream.astype(jnp.bfloat16), small["pre_attn"][pl.ds(li, 1)]).astype(jnp.bfloat16)
            layer_copy("qgkv", li).wait()
            qgkv = dot(x, qbuf[...]).astype(jnp.bfloat16)
            if tp == 8:
                gu_copy(li, 0)
            elif tp == 16:
                layer_copy("gu", li).start()
            elif li + 1 < c.layers:
                layer_copy("qgkv", li + 1).start()
            head_groups = []
            for group in range(kvh):
                k = jnp.pad(qgkv[:, 2 * qwidth + group * 64:2 * qwidth + (group + 1) * 64], ((0, 0), (0, 64)))
                k = rotate(head_norm(k))
                kv = jnp.concatenate((k[:, :64], qgkv[:, 2 * qwidth + (kvh + group) * 64:2 * qwidth + (kvh + group + 1) * 64]), axis=1)
                block = jnp.pad(kv, ((0, 16 - rows), (0, 0)))
                context = cache[li, :, group * 128:(group + 1) * 128]
                queries = []
                for h in range(group * (c.heads // tp // kvh), (group + 1) * (c.heads // tp // kvh)):
                    q = jnp.pad(qgkv[:, h * 64:(h + 1) * 64], ((0, 0), (0, 64)))
                    queries.append(rotate(head_norm(q)))
                query = jnp.concatenate(queries, axis=0)
                sctx = dot(query, context.T) * (43.7840518911 / c.head_dim)
                sblk = dot(query, block.T) * (43.7840518911 / c.head_dim)
                sctx = jnp.where(jnp.tile(context_keep, (c.heads // tp // kvh, 1)), sctx, -1e30)
                sblk = jnp.where(jnp.tile(block_keep, (c.heads // tp // kvh, 1)), sblk, -1e30)
                maximum = jnp.maximum(jnp.max(sctx, axis=1, keepdims=True), jnp.max(sblk, axis=1, keepdims=True))
                pctx, pblk = jnp.exp(sctx - maximum), jnp.exp(sblk - maximum)
                denominator = jnp.sum(pctx, axis=1, keepdims=True) + jnp.sum(pblk, axis=1, keepdims=True)
                attended = (dot((pctx / denominator).astype(jnp.bfloat16), context) + dot((pblk / denominator).astype(jnp.bfloat16), block)).astype(jnp.bfloat16)
                values = jnp.pad(attended[:, 64:], ((0, 0), (0, 64)))
                normalized = head_norm(values)
                head_groups.append(normalized)
            normalized = jnp.concatenate(head_groups, axis=0)
            parts = []
            for h in range(c.heads // tp):
                gate = qgkv[:, qwidth + h * 64:qwidth + (h + 1) * 64].astype(jnp.float32)
                parts.append((normalized[h * rows:(h + 1) * rows, :64] * jax.nn.sigmoid(gate)).astype(jnp.bfloat16))
            attended = jnp.concatenate(parts, axis=1)
            projected = []
            for part in range(projection_parts):
                layer_copy("o", li, part).wait()
                projected.append(dot(attended, obuf[...]))
                if part + 1 < projection_parts:
                    layer_copy("o", li, part + 1).start()
            partial = jnp.concatenate(projected, axis=1)
            if tp == 8:
                down_copy(li, 0).start()
            elif tp == 16:
                layer_copy("down", li).start()
            elif li + 1 < c.layers:
                layer_copy("o", li + 1).start()
            branch = norm(allreduce(partial, 0), eps=c.post_eps).astype(jnp.bfloat16)
            stream = small["alpha_attn"][pl.ds(li, 1)] * stream + small["beta_attn"][pl.ds(li, 1)] * branch.astype(jnp.float32)
            x = norm(stream.astype(jnp.bfloat16), small["pre_ffn"][pl.ds(li, 1)]).astype(jnp.bfloat16)
            if tp == 8:
                partial = jnp.zeros((rows, hidden), jnp.float32)
                for part in range(width // tile_width):
                    gu_copy(li, part, wait=True)
                    gu = dot(x, gubuf[...]).astype(jnp.bfloat16).astype(jnp.float32)
                    if part + 1 < width // tile_width:
                        gu_copy(li, part + 1)
                    else:
                        @pl.when(li + 1 < c.layers)
                        def prefetch_q():
                            layer_copy("qgkv", li + 1).start()
                    act = (jax.nn.silu(gu[:, :tile_width]) * gu[:, tile_width:]).astype(jnp.bfloat16)
                    down_copy(li, part).wait()
                    partial += dot(act, dbuf[...])
                    if part + 1 < width // tile_width:
                        down_copy(li, part + 1).start()
                @pl.when(li + 1 < c.layers)
                def prefetch_o():
                    layer_copy("o", li + 1).start()
            else:
                layer_copy("gu", li).wait()
                gu = dot(x, gubuf[...]).astype(jnp.bfloat16).astype(jnp.float32)
                if tp <= 16:
                    @pl.when(li + 1 < c.layers)
                    def prefetch_q():
                        layer_copy("qgkv", li + 1).start()
                elif li + 1 < c.layers:
                    layer_copy("gu", li + 1).start()
                act = (jax.nn.silu(gu[:, :width]) * gu[:, width:]).astype(jnp.bfloat16)
                layer_copy("down", li).wait()
                partial = dot(act, dbuf[...])
                if tp <= 16:
                    @pl.when(li + 1 < c.layers)
                    def prefetch_o():
                        layer_copy("o", li + 1).start()
                elif li + 1 < c.layers:
                    layer_copy("down", li + 1).start()
            branch = allreduce(partial, 1).astype(jnp.float32) * small["post_ffn"][pl.ds(li, 1)]
            branch = norm(branch, eps=c.post_eps).astype(jnp.bfloat16)
            stream = small["alpha_ffn"][pl.ds(li, 1)] * stream + small["beta_ffn"][pl.ds(li, 1)] * branch.astype(jnp.float32)

            return stream

        if tp <= 16:
            stream = jax.lax.fori_loop(0, c.layers, layer, stream)
        else:
            for li in range(c.layers):
                stream = layer(li, stream)

        if tp == 16:
            copy(w["markov_w2"], w2buf, 4).start()
        x = norm(stream.astype(jnp.bfloat16), small["norm"][...]).astype(jnp.bfloat16)
        for i in range(2):
            copy(w["lm"].at[:, pl.ds(i * chunk, chunk)], lmbuf.at[i], i).start()
        for i in range(padded // chunk):
            slot = i % 2
            copy(w["lm"].at[:, pl.ds(i * chunk, chunk)], lmbuf.at[slot], slot).wait()
            logitsbuf[:, pl.ds(i * chunk, chunk)] = dot(x, lmbuf[slot]).astype(jnp.bfloat16)
            if i + 2 < padded // chunk:
                copy(w["lm"].at[:, pl.ds((i + 2) * chunk, chunk)], lmbuf.at[slot], slot).start()
        if tp == 8:
            copy(w["markov_w2"], w2buf, 4).start()
        copy(w["markov_w2"], w2buf, 4).wait()
        previous = anchor
        token_columns = jnp.arange(128)[None]
        tokens = jnp.zeros((1, 128), jnp.int32)
        vocab_columns = jnp.arange(padded)[None]
        for i in range(c.draft_tokens):
            copy(w["markov_w1"].at[pl.ds(previous // 8 * 8, 8), :], w1buf, 7).wait()
            selected = jnp.sum(jnp.where(rindex == previous % 8, w1buf[...].astype(jnp.float32), 0), axis=0, keepdims=True).astype(jnp.bfloat16)
            bias = dot(jnp.broadcast_to(selected, (rows, c.markov_rank)), w2buf[...])[:1].astype(jnp.bfloat16)
            scores = (logitsbuf[i:i + 1] + bias).astype(jnp.bfloat16).astype(jnp.float32) * c.output_multiplier
            scores = c.soft_cap * jnp.tanh(scores / c.soft_cap)
            scores = jnp.where(vocab_columns < vocab, scores, -jnp.inf)
            maximum = jnp.max(scores, axis=1, keepdims=True)
            index = jnp.min(jnp.where(scores == maximum, vocab_columns, padded), axis=1, keepdims=True) + rank * vocab
            packet = jnp.where(token_columns == 0, maximum, jnp.where(token_columns == 1, index.astype(jnp.float32), 0))
            slot = i % 2
            tiny[slot, rank] = packet
            transfers = [remote(tiny.at[slot, rank], tiny.at[slot, rank], 20 + slot * (tp - 1) + offset - 1, rank ^ offset) for offset in range(1, tp)]
            for transfer in transfers:
                transfer.start()
            for transfer in transfers:
                transfer.wait()
            maxima = tiny[slot, :, :, :1]
            winner = jnp.min(jnp.where(maxima == jnp.max(maxima, axis=0, keepdims=True), tiny[slot, :, :, 1:2], c.vocab), axis=0)
            tokens = jnp.where(token_columns == i, winner.astype(jnp.int32), tokens)
            previous = winner[0, 0].astype(jnp.int32)
            if i + 1 < c.draft_tokens:
                copy(w["markov_w1"].at[pl.ds(previous // 8 * 8, 8), :], w1buf, 7).start()
        out[...] = tokens

    def local(weights, embed, cache, anchor, position):
        weights = jax.tree.map(lambda a: a[0], weights)
        if packed_cache:
            kv, positions = cache
            kv = kv[0]
        else:
            kc, vc, positions = cache
            permutation = jnp.concatenate((jnp.arange(0, 64, 2), jnp.arange(1, 64, 2)))
            kv = jnp.concatenate((kc[0][..., permutation], vc[0]), axis=-1).reshape(c.layers, c.window, kvh * 128)
        embedding = embed[0]
        if tp <= 16:
            offsets = jnp.stack((anchor, jnp.int32(c.mask_token))) - jax.lax.axis_index("tp") * vocab
            selected = embedding[jnp.clip(offsets, 0, vocab - 1)]
            embedding = jnp.where(((offsets >= 0) & (offsets < vocab))[:, None], selected, 0)
            embedding = jnp.pad(embedding, ((0, rows - 2), (0, 0)))
        inputs = (*[weights[k] for k in SMALL], *[weights[k] for k in LARGE], kv, positions[None], embedding)
        v = lambda shape, dtype=jnp.bfloat16: tpu.VMEM(shape, dtype)
        result = pl.pallas_call(
            body, out_shape=jax.ShapeDtypeStruct((1, 128), jnp.int32),
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[pl.BlockSpec()] * len(SMALL) + [pl.BlockSpec(memory_space=tpu.HBM)] * len(LARGE) + [pl.BlockSpec(), pl.BlockSpec(), pl.BlockSpec() if tp <= 16 else pl.BlockSpec(memory_space=tpu.HBM)],
                out_specs=pl.BlockSpec(),
                scratch_shapes=(
                    v((1, 128)) if tp <= 16 else v(q_shape), v((projection_tiles, 32, 128), jnp.uint8) if tp <= 16 else v(o_shape),
                    v((arena_tiles, 32, 128), jnp.uint8), v((1, 128)) if tp <= 16 else v((width, hidden)),
                    v((1, 128)) if tp <= 16 else v((c.markov_rank, padded)), v((8, hidden)), v((8, hidden)),
                    v((8, c.markov_rank)), v((rows, padded)), v((1, 128), jnp.int32),
                    v((2, rows, *row_shape), jnp.float32), v((2, 8, *row_shape), jnp.float32),
                    v((2, *row_shape), jnp.float32), v((2, hosts, *row_shape), jnp.float32), v((2, rows, *row_shape)),
                    v((2, tp, 1, 128), jnp.float32),
                    tpu.SemaphoreType.DMA((8,)), tpu.SemaphoreType.DMA((20 + 2 * (tp - 1),)), tpu.SemaphoreType.DMA((20 + 2 * (tp - 1),)),
                ),
            ),
            compiler_params=tpu.CompilerParams(collective_id=49, vmem_limit_bytes=64 * 1024**2, shape_invariant_numerics=True),
            name="muse_dspark_persistent_draft",
        )(jnp.stack((anchor, position)), *inputs)
        return result[0, :c.draft_tokens]

    cache_spec = (P("tp"), P()) if packed_cache else (P("tp"), P("tp"), P())
    return jax.jit(jax.shard_map(local, mesh=mesh, in_specs=(P("tp"), P("tp"), cache_spec, P(), P()), out_specs=P(), check_vma=False))


def host(value):
    return np.asarray(value.addressable_shards[0].data)


class Decoder:
    def __init__(self, mesh, config, context, weights, prefill_weights, draft_path, zero_states, *, fixed_length=None, backend="fused", draft_backend="fused", stats_path=None, target_backend=None):
        self.config, self.context = config, context
        self.tp = mesh.size
        if target_backend is not None:
            self.target = target_backend
        elif self.tp in (8, 16):
            from . import decode_megakernel
            self.target = decode_megakernel
        elif self.tp == 32:
            from .decode_megakernel import get_tp32_backend
            self.target = get_tp32_backend()
        else:
            raise ValueError("DSpark requires TP8, TP16, or TP32")
        self.weights, self.prefill_weights = weights, prefill_weights
        self.c = DraftConfig.from_checkpoint(draft_path)
        if self.c.hidden != config.dim or self.c.vocab != config.vocab:
            raise ValueError("draft and target geometry do not match")
        self.rows = self.c.draft_tokens + 1
        self.draft_weights = load_weights(draft_path, self.c, mesh)
        self.packed_io = self.tp == 8 and draft_backend == "fused"
        self.empty_draft, self.update_draft, self.draft = make_programs(self.c, mesh, fused_update=draft_backend == "fused", packed_cache=self.packed_io)
        self.draft_backend, self.verify_backend = draft_backend, backend
        self.embed = prefill_weights["emb"]
        if draft_backend == "fused":
            prepared = prepare_weights(self.c, mesh, self.draft_weights)
            context_keys = ("fc", "hidden_norm", "pre_attn", "k", "v")
            self.draft_weights = {**{key: self.draft_weights[key] for key in context_keys}, **prepared}
            self.draft = make_draft(self.c, mesh, packed_cache=self.packed_io)
            self.embed = weights["emb"]
        self.zero_states = zero_states
        self.fixed_length = -1 if fixed_length is None else fixed_length
        self.stats_path = stats_path
        self.last_stats = {}
        self.serial = self.target.make_verify(mesh, config, context, self.rows, self.c.aux_layers)
        if backend not in ("fused", "sequential"):
            raise ValueError("verification backend must be fused or sequential")
        if backend == "fused":
            self.weights = self.target.prepare_verify_weights(mesh, weights)
        self.verify = self.target.make_verify_block(mesh, config, context, self.rows, self.c.aux_layers) if backend == "fused" else (
            lambda w, s, t, p: self.serial(w, s, t, jnp.array([p, self.rows], jnp.int32))
        )
        prefill = self.target.make_prefill(mesh, config, context, self.tp, aux_layers=self.c.aux_layers)
        if self.tp == 32 and self.target.CACHE_KEYS == ("kvcache",):
            sharding = NamedSharding(mesh, P("tp"))
            self.prefill = jax.jit(prefill, out_shardings=({"kvcache": sharding}, sharding, NamedSharding(mesh, P())))
        else:
            self.prefill = jax.jit(prefill)
        self.call = jax.jit(self._call, donate_argnums=(2, 3))

    def prepare(self, ids):
        if not ids or len(ids) >= self.context:
            raise ValueError("a nonempty prompt must fit the target context")
        states = self.zero_states()
        cache = self.empty_draft()
        if len(ids) <= 64:
            features = []
            for p in range(0, len(ids), self.rows):
                count = min(self.rows, len(ids) - p)
                tokens = np.full((self.rows,), self.config.pad, np.int32)
                tokens[:count] = ids[p:p + count]
                targets, aux, states = self.serial(self.weights, states, jnp.asarray(tokens), jnp.array([p, count], jnp.int32))
                features.append(aux[:count])
            anchor = targets[count - 1]
            aux = jnp.concatenate(features)
            positions = jnp.arange(len(ids), dtype=jnp.int32)
        else:
            width = (len(ids) + 63) // 64 * 64
            pad = width - len(ids)
            tokens = jnp.asarray([self.config.pad] * pad + ids, jnp.int32)
            states, logits, aux = self.prefill(self.prefill_weights, states, tokens, jnp.int32(pad))
            logits = logits.reshape(self.tp, -1)[:, :self.config.vocab // self.tp]
            anchor = jnp.argmax(logits.reshape(-1)).astype(jnp.int32)
            positions = jnp.arange(width, dtype=jnp.int32) - pad
        aux, positions = aux[-self.c.window:], positions[-self.c.window:]
        cache = self.update_draft(self.draft_weights, cache, aux, positions, positions >= 0)
        if self.verify_backend == "fused":
            states = self.target.prepare_verify_cache(states)
        return states, cache, anchor, jnp.int32(len(ids))

    def _call(self, weights, draft_weights, states, cache, anchor, position, remaining, ignore_eos, embed):
        emitted = jnp.full((4, self.rows), -1, jnp.int32)
        lengths = jnp.zeros((4,), jnp.int32)

        def one(i, carry):
            states, cache, anchor, position, remaining, emitted, lengths = carry
            proposals = self.draft(draft_weights, embed, cache, anchor, position)
            row_tokens = jnp.concatenate((anchor[None], proposals))
            targets, aux, states = self.verify(weights, states, row_tokens, position)
            count, _, _ = acceptance(proposals, targets, self.fixed_length)
            raw_length = count + 1
            count = jnp.minimum(count, jnp.minimum(remaining, self.context - position) - 1)
            bonus = targets[count]
            slots = jnp.arange(self.rows)
            output = jnp.where(slots < count, jnp.pad(proposals, (0, 1)), jnp.where(slots == count, bonus, -1))
            is_eos = jnp.zeros((self.rows,), jnp.bool_)
            for token in self.config.eos:
                is_eos |= output == token
            first_eos = jnp.min(jnp.where(is_eos & ~ignore_eos, slots, self.rows))
            output = jnp.where(slots <= first_eos, output, -1)
            cache = self.update_draft(draft_weights, cache, aux, position + slots, slots <= count)
            emitted = emitted.at[i].set(output)
            lengths = lengths.at[i].set(raw_length)
            remaining = jnp.where(first_eos < self.rows, 0, remaining - count - 1)
            return states, cache, bonus, position + count + 1, remaining, emitted, lengths

        initial = (states, cache, anchor, position, remaining, emitted, lengths)

        def more(carry):
            i, state = carry
            return (i < 4) & (state[4] > 0) & (state[3] < self.context)

        def advance(carry):
            i, state = carry
            return i + 1, one(i, state)

        result = jax.lax.while_loop(more, advance, (jnp.int32(0), initial))[1]
        if self.packed_io:
            report = jnp.pad(result[5], ((0, 0), (0, 128 - self.rows)))
            report = report.at[:, self.rows].set(result[6])
            report = report.at[0, self.rows + 1].set(result[3])
            report = report.at[0, self.rows + 2].set(result[4])
            return (*result[:5], report)
        return result

    def generate(self, ids, max_tokens, stop, ignore_eos, emit, tokenizer):
        t0 = time.perf_counter()
        states, cache, anchor, position = self.prepare(ids)
        anchor_host = int(host(anchor))
        jax.block_until_ready((states, cache))
        prefill_seconds = time.perf_counter() - t0
        generated = [anchor_host] if max_tokens > 0 else []
        finish = "stop" if generated and anchor_host in self.config.eos and not ignore_eos else "length"
        emit(generated, 0.0)
        lengths = []
        remaining = jnp.int32(max_tokens - len(generated))
        inflight = collections.deque()
        t0 = last = time.perf_counter()
        calls = 0
        done = finish == "stop" or len(generated) >= max_tokens
        while not done:
            while len(inflight) < 2:
                out = self.call(self.weights, self.draft_weights, states, cache, anchor, position, remaining, jnp.bool_(ignore_eos), self.embed)
                states, cache, anchor, position, remaining = out[:5]
                if self.packed_io:
                    out[5].addressable_shards[0].data.copy_to_host_async()
                inflight.append(out)
            out = inflight.popleft()
            if self.packed_io:
                report = host(out[5])
                tokens, accepted = report[:, :self.rows], report[:, self.rows]
            else:
                tokens, accepted = host(out[5]), host(out[6])
            calls += 1
            now = time.perf_counter()
            dt, last = now - last, now
            fresh = []
            for row, length in zip(tokens, accepted):
                if not length:
                    continue
                lengths.append(int(length))
                for token in row:
                    if token < 0:
                        continue
                    generated.append(int(token))
                    fresh.append(int(token))
                    if token in self.config.eos and not ignore_eos:
                        finish, done = "stop", True
                        break
                    if len(generated) >= max_tokens:
                        done = True
                        break
                if stop and any(s in tokenizer.decode(generated) for s in stop):
                    finish, done = "stop", True
                if done:
                    break
            emit(fresh, dt)
            if self.packed_io:
                done |= int(report[0, self.rows + 2]) <= 0 or int(report[0, self.rows + 1]) >= self.context
            else:
                done |= int(host(out[4])) <= 0 or int(host(out[3])) >= self.context
        for out in inflight:
            jax.block_until_ready(out)
        decode_seconds = time.perf_counter() - t0
        self.last_stats = {
            "prompt_tokens": len(ids), "generated_tokens": len(generated),
            "speculative_rounds": len(lengths), "acceptance_lengths": lengths,
            "mean_acceptance_length": float(np.mean(lengths)) if lengths else 0.0,
            "acceptance_histogram": {str(i): lengths.count(i) for i in range(1, self.rows + 1)},
            "fixed_acceptance_length": self.fixed_length if self.fixed_length > 0 else None,
            "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
            "steps_per_call": 4, "calls": calls,
            "draft_backend": self.draft_backend, "verify_backend": self.verify_backend,
        }
        if jax.process_index() == 0:
            print("DSPARK " + json.dumps(self.last_stats), flush=True)
            if self.stats_path:
                path = Path(self.stats_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as handle:
                    handle.write(json.dumps(self.last_stats) + "\n")
        return generated, finish, prefill_seconds, decode_seconds
