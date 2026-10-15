"""Muse Spark on 32 devices: replicated TP8 experts, or ENV-selected TP8 x EP4."""

import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pool_alias
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import NamedSharding, PartitionSpec as P

from . import F32, NVFP4_LUT
from .load import lm_width


CACHE_KEYS = ("kvcache",)
REPLICATED_EXPERTS = False


def rms_gain_shard(x, gain, eps):
    y = x.astype(F32)
    y = y * jax.lax.rsqrt(jnp.mean(y * y, axis=-1, keepdims=True) + eps)
    return (y * gain.astype(F32)).astype(jnp.bfloat16)


def rms_plain_shard(x, eps):
    y = x.astype(F32)
    y = y * jax.lax.rsqrt(jnp.mean(y * y, axis=-1, keepdims=True) + eps)
    return y.astype(jnp.bfloat16)


def rms_plain_f32(x, eps):
    y = x.astype(F32)
    return y * jax.lax.rsqrt(jnp.mean(y * y, axis=-1, keepdims=True) + eps)


def rms_gain_first_shard(x, gain, eps):
    y = x.astype(F32) * gain.astype(F32)
    y = y * jax.lax.rsqrt(jnp.mean(y * y, axis=-1, keepdims=True) + eps)
    return y  # FP32 out (post-norm contract)


def rms_gain_first_f32(x, gain, eps):
    return rms_gain_first_shard(x, gain, eps)


def rope_tables_shard(c, positions):
    inv = 1.0 / (c.rope_theta ** (jnp.arange(0, c.head_dim, 2, dtype=F32) / c.head_dim))
    angles = positions.astype(F32)[:, None] * inv
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope_shard(x, cos, sin):
    """x [L, heads, head_dim]; cos/sin [L, head_dim//2]; split-half rotation."""
    real, imag = jnp.split(x.astype(F32), 2, axis=-1)
    cv = cos[:, None, :]
    sv = sin[:, None, :]
    out = jnp.concatenate((real * cv - imag * sv, real * sv + imag * cv), axis=-1)
    return out.astype(jnp.bfloat16)


def _dequant_experts_tiled(packed, scales, globals_, count, k, n):
    """Expand packed experts in VMEM with the original FP32 scale products."""
    if jax.default_backend() == "cpu":
        codes = jnp.stack((packed & 15, packed >> 4), axis=2).reshape(count, k, n)
        values = jnp.asarray(NVFP4_LUT, F32)[codes]
        scale = jax.lax.bitcast_convert_type(scales, jnp.float8_e4m3fn).astype(F32)
        gains = jnp.repeat(globals_.reshape(count, -1), n // globals_.reshape(count, -1).shape[-1], axis=1)
        scale *= gains[:, None]
        return (values.reshape(count, k // 16, 16, n) * scale[:, :, None]).reshape(count, k, n).astype(jnp.bfloat16)
    bk, bn = min(k, 2048), min(n, 512)
    globals_ = globals_.reshape(count, -1)
    gains = jnp.repeat(globals_, n // globals_.shape[-1], axis=1)[:, None, :]

    def kernel(p, s, g, out):
        values = tpu.bitcast(p[0], jnp.float4_e2m1fn).astype(F32)
        scale = jax.lax.bitcast_convert_type(s[0], jnp.float8_e4m3fn).astype(F32)
        scale = scale * g[0, 0][None, :]
        out[0] = (values.reshape(bk // 16, 16, bn) * scale[:, None, :]).reshape(bk, bn).astype(jnp.bfloat16)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((count, k, n), jnp.bfloat16),
        grid=(count, k // bk, n // bn),
        in_specs=(
            pl.BlockSpec((1, bk // 2, bn), lambda e, i, j: (e, i, j)),
            pl.BlockSpec((1, bk // 16, bn), lambda e, i, j: (e, i, j)),
            pl.BlockSpec((1, 1, bn), lambda e, i, j: (e, 0, j)),
        ),
        out_specs=pl.BlockSpec((1, bk, bn), lambda e, i, j: (e, i, j)),
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel", "parallel", "parallel")),
    )(packed, scales, gains)


_VERIFY_WEIGHTS = ("qwkv", "ow", "pool", "unpool", "router", "router_bias", "scalars", "expert_norms", "gu_pack", "dw_pack", "gu_dense", "dw_dense", "gu_global", "dw_global", "lmw", "fnorm")


def _verify_router_order(experts):
    """Place ascending expert IDs in Mosaic's native argmax tie order."""
    columns = jnp.arange(experts, dtype=jnp.int32)
    if experts not in (128, 256):
        return columns
    return (127 - columns % 128) * (experts // 128) + columns // 128


def _verify_router_argmax(ranked):
    experts = ranked.shape[-1]
    if experts in (128, 256):
        chosen = jnp.argmax(ranked, axis=1).astype(jnp.int32)[:, None]
        return (127 - chosen % 128) * (experts // 128) + chosen // 128
    maximum = jnp.max(ranked, axis=1, keepdims=True)
    return jnp.min(jnp.where(ranked == maximum, jnp.arange(experts)[None], experts), axis=1, keepdims=True)


def _make_block(mesh, c, context, rows, aux_layers, *, batch_size=None):
    """One expert partition for independent requests and causal verification."""
    if mesh.size != 32 or rows not in (1, 2, 4, 8) or c.head_dim != 64:
        raise ValueError("the target requires 32 devices and 64-wide heads")
    if batch_size is not None and batch_size not in (1, 2, 4, 8):
        raise ValueError("batch size must be 1, 2, 4, or 8")
    if len(set(aux_layers)) != len(aux_layers) or any(i < 0 or i >= c.layers for i in aux_layers):
        raise ValueError("auxiliary layer indices must be unique target layers")
    h, mh, nh = c.dim, c.moe_hidden, c.heads // 32
    iw, di = c.moe_intermediate // 8, c.moe_intermediate // 8
    qwidth, pw, es = nh * 64, mh // 8, c.experts // 8
    router_rows, pool_rows = 2 * rows, rows * pw // 128
    mixed_stride = ((mh // 128 + 15) // 16) * 16
    attention_rows = ((rows * (h // 8) // 128 + 7) // 8) * 8
    qshape, oshape = (h, 2 * qwidth + 128), (8 * qwidth, h // 8)
    pshape, ushape = (h, pw), (mh, h // 8)
    gpshape, dpshape = (4, mh * 9 // 16, 2 * iw), (4, iw * 9 // 16, mh)
    dense_slots = 2 if rows > 1 else 4
    gdshape, ddshape = (dense_slots, mh, 2 * di), (dense_slots, di, mh)

    def tiles(shape, dtype):
        return pool_alias.storage_size(shape, jnp.dtype(dtype)) // 4096

    gt = max(tiles(gpshape, jnp.uint8), tiles(gdshape, jnp.bfloat16))
    dt = max(tiles(dpshape, jnp.uint8), tiles(ddshape, jnp.bfloat16))
    head_width = lm_width(c, 32)
    lm_chunk = min(640 if rows == 1 else 1280, head_width // 2)
    lm_shape = (2, h, lm_chunk)
    q_tiles = tiles(qshape, jnp.bfloat16)
    attention_group = batch_size or 1
    while attention_group > 1 and attention_group * nh > 16:
        attention_group //= 2
    cache_slots = attention_group if batch_size is not None else 1
    cache_tiles = tiles((cache_slots, context, 128), jnp.bfloat16)
    qt = max(q_tiles + cache_tiles, gt + dt, tiles(lm_shape, jnp.bfloat16))
    pt = max(tiles(shape, jnp.bfloat16) for shape in (oshape, pshape, ushape))

    def kernel(meta, *args):
        w = dict(zip(_VERIFY_WEIGHTS, args[:len(_VERIFY_WEIGHTS)]))
        (initial, kvc_in, final, aux, kvc, qa, pa, router, scalars, norms, bias,
         gg, dg, pair_ref, ahost, apaired, adirect, udirect, comb_ref, block_values, estage, epipe, pipe_rows, mixed_rows, unhost,
         ungather, ids_v, ids_s, candidates, sem, sends, recvs) = args[len(_VERIFY_WEIGHTS):]
        qbuf = pool_alias.view(qa.at[:q_tiles], qshape, jnp.bfloat16)
        kvbuf = pool_alias.view(qa.at[q_tiles:q_tiles + cache_tiles], (cache_slots, context, 128), jnp.bfloat16)
        gp = pool_alias.view(qa.at[:tiles(gpshape, jnp.uint8)], gpshape, jnp.uint8)
        dp = pool_alias.view(qa.at[gt:gt + tiles(dpshape, jnp.uint8)], dpshape, jnp.uint8)
        gd = pool_alias.view(qa.at[:tiles(gdshape, jnp.bfloat16)], gdshape, jnp.bfloat16)
        dd = pool_alias.view(qa.at[gt:gt + tiles(ddshape, jnp.bfloat16)], ddshape, jnp.bfloat16)
        obuf = pool_alias.view(pa.at[:tiles(oshape, jnp.bfloat16)], oshape, jnp.bfloat16)
        pbuf = pool_alias.view(pa.at[:tiles(pshape, jnp.bfloat16)], pshape, jnp.bfloat16)
        ubuf = pool_alias.view(pa.at[:tiles(ushape, jnp.bfloat16)], ushape, jnp.bfloat16)
        lm = pool_alias.view(qa.at[:tiles(lm_shape, jnp.bfloat16)], lm_shape, jnp.bfloat16)
        rank = jax.lax.axis_index("tp")
        lane, host = rank % 8, rank // 8
        owner = lane % rows
        r = jnp.arange(rows, dtype=jnp.int32)[:, None]
        if batch_size is None:
            pos = meta[0] + jnp.minimum(r, meta[1] - 1)
            live = r < meta[1]
        else:
            pos = jnp.concatenate([jnp.full((1, 1), meta[i], jnp.int32) for i in range(rows)])
            live = jnp.concatenate([jnp.full((1, 1), meta[rows + i], jnp.int32) for i in range(rows)]) != 0
        columns = jnp.arange(128, dtype=jnp.int32)[None].astype(jnp.float32)
        angle = pos.astype(jnp.float32) / c.rope_theta ** ((columns % 32) / 32)
        cosine, sine = jnp.cos(angle), jnp.sin(angle)
        experts = jnp.arange(c.experts)[None]
        router_ids = _verify_router_order(c.experts)[None]

        def copy(src, dst, channel):
            return tpu.make_async_copy(src, dst, sem.at[channel])

        def remote(src, dst, channel, peer):
            return tpu.make_async_remote_copy(src, dst, sends.at[channel], recvs.at[channel], device_id=(peer,), device_id_type=pl.DeviceIdType.MESH)

        def gather(buf, index, offsets, channel):
            transfers = [remote(buf.at[index], buf.at[index], channel + i, rank ^ offset) for i, offset in enumerate(offsets)]
            for transfer in transfers:
                transfer.start()
            for transfer in transfers:
                transfer.wait()

        def flat_gather(buffer, value, index, groups, offsets, channel):
            width = value.size
            stride = (width + 255) // 256 * 256
            region = pl.ds(pl.multiple_of(index * stride, 256), stride)
            buffer[region] = jnp.pad(jnp.concatenate([value[row] for row in range(value.shape[0])]), (0, stride - width))
            transfers = [remote(buffer.at[region], buffer.at[region], channel + i, rank ^ offset) for i, offset in enumerate(offsets)]
            for transfer in transfers:
                transfer.start()
            for transfer in transfers:
                transfer.wait()
            return jnp.stack([jnp.concatenate([buffer[group * stride + row * value.shape[1]:group * stride + (row + 1) * value.shape[1]][None, :] for row in range(value.shape[0])], axis=0) for group in range(groups)])

        def direct(buffer, value, layer, tag):
            slot = layer % 2
            channel = 69 + 2 * slot + tag
            if tag == 1:
                stride = (value.size + 255) // 256 * 256
                region = pl.ds(pl.multiple_of((slot * 32 + rank) * stride, 128), stride)
                buffer[region] = jnp.pad(jnp.concatenate([value[row] for row in range(value.shape[0])]), (0, stride - value.size))
                for offset in tuple(sorted(range(1, 32), key=lambda offset: (offset % 8, offset // 8), reverse=True)):
                    remote(buffer.at[region], buffer.at[region], channel, rank ^ offset).start()
                wait_region = pl.ds(pl.multiple_of(slot * 32 * stride, 128), 31 * stride)
                remote(buffer.at[wait_region], buffer.at[wait_region], channel, rank).wait()
                packets = []
                for peer in range(32):
                    base = (slot * 32 + peer) * stride
                    if (h // 8) % 256:
                        packet = buffer[pl.ds(pl.multiple_of(base, 256), stride)]
                        values = [packet[row * (h // 8):(row + 1) * (h // 8)][None, :] for row in range(rows)]
                    else:
                        values = [buffer[pl.ds(pl.multiple_of(base + row * (h // 8), 128), h // 8)][None, :] for row in range(rows)]
                    packets.append(jnp.concatenate(values, axis=0))
                values = jnp.stack(packets).reshape(4, 8, rows, h // 8)
                return jnp.sum(values.astype(jnp.float32), axis=0).transpose(1, 0, 2).reshape(rows, h)
            flat = value.reshape(-1)
            stride = buffer.shape[1] // 32
            alignment = 16 if buffer.dtype == jnp.bfloat16 else 8
            packet = pl.ds(pl.multiple_of(rank * stride, alignment), stride)
            buffer[slot, packet] = jnp.pad(flat, (0, stride * 128 - flat.size)).reshape(stride, 128).astype(buffer.dtype)
            offsets = tuple(sorted(range(1, 32), key=lambda offset: (offset % 8, offset // 8), reverse=True))
            for offset in offsets:
                remote(buffer.at[slot, packet], buffer.at[slot, packet], channel, rank ^ offset).start()
            remote(buffer.at[slot, :31 * stride], buffer.at[slot, :31 * stride], channel, rank).wait()
            values = buffer[slot].reshape(32, stride * 128)[:, :rows * (h // 8)].reshape(4, 8, rows, h // 8)
            return jnp.sum(values.astype(jnp.float32), axis=0).transpose(1, 0, 2).reshape(rows, h)

        def dot(x, y):
            count = x.shape[0]
            if count == 1:
                x = jnp.broadcast_to(x, (16, x.shape[1]))
            elif count < 16:
                x = jnp.pad(x, ((0, 16 - count), (0, 0)))
            return jnp.dot(x, y, preferred_element_type=jnp.float32)[:count]

        def norm(x, eps, gain=None):
            x = x.astype(jnp.float32)
            x *= jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
            return x if gain is None else x * gain

        def hn(x):
            x = x.astype(jnp.float32)
            square = (x * x).astype(jnp.bfloat16).astype(jnp.float32)
            return x * jax.lax.rsqrt(jnp.sum(square, axis=-1, keepdims=True) / 64 + c.eps)

        def rotate(x, sliding):
            x = x.astype(jnp.float32)
            shifted = jnp.concatenate((-x[:, 32:64], x[:, :32], jnp.zeros((x.shape[0], 64), jnp.float32)), axis=1)
            repeats = x.shape[0] // rows
            return jnp.where(sliding, x * jnp.tile(cosine, (repeats, 1)) + shifted * jnp.tile(sine, (repeats, 1)), x).astype(jnp.bfloat16)

        barrier = tpu.get_barrier_semaphore()
        for offset in range(1, 32):
            pl.semaphore_signal(barrier, 1, device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(barrier, 31)
        aux[...] = jnp.zeros(aux.shape, jnp.bfloat16)
        copy(w["qwkv"].at[0], qbuf, 0).start()
        copy(w["ow"].at[0], obuf, 1).start()
        copy(w["scalars"].at[0], scalars, 3).start()

        def layer(li, stream, dense, below):
            pi = jnp.clip(li - below, 0, c.layers - len(c.bf16_expert_layers) - 1)
            bi = jnp.clip(below, 0, len(c.bf16_expert_layers) - 1)
            sliding = li % 4 != 1
            def cache_region(anchor, count, action):
                write_rows = rows if batch_size is None else 1
                widths = [min(512, context)]
                while widths[-1] < context:
                    widths.append(min(2 * widths[-1], context))

                def prefix(index):
                    if index == len(widths) - 1:
                        return action(widths[index], jnp.int32(0))
                    return jax.lax.cond(jnp.maximum(anchor + count, jnp.minimum(anchor // 16 * 16, context - 32) + 32) <= widths[index], lambda: action(widths[index], jnp.int32(0)), lambda: prefix(index + 1))

                window = min(context, ((c.sliding_window + 126 + write_rows) // 128 + 1) * 128)
                base = jnp.clip((anchor - c.sliding_window + 1) // 128 * 128, 0, context - window)
                return jax.lax.cond(sliding & (anchor + count > window), lambda: action(window, base), lambda: prefix(0))

            loads = [(w["router"].at[li], router),
                     (w["scalars"].at[li], scalars), (w["expert_norms"].at[li], norms), (w["router_bias"].at[li], bias),
                     (w["gu_global"].at[pi], gg), (w["dw_global"].at[pi], dg)]
            for channel, (src, dst) in enumerate(loads, 2):
                if channel != 3 and (not dense or channel < 6):
                    copy(src, dst, channel).start()
            if batch_size in (None, 1):
                def prefetch_cache(length, base):
                    base = pl.multiple_of(base, 128)
                    copy(kvc.at[li, 0, pl.ds(base, length)], kvbuf.at[0, :length], 8).start()
                cache_region(meta[0], meta[1], prefetch_cache)
            copy(w["scalars"].at[li], scalars, 3).wait()
            sc = scalars[...]
            x = norm(stream, c.eps, sc[0:1]).astype(jnp.bfloat16)
            copy(w["qwkv"].at[li], qbuf, 0).wait()
            qgkv = dot(x, qbuf[...]).astype(jnp.bfloat16)
            k = rotate(hn(jnp.pad(qgkv[:, 2 * qwidth:2 * qwidth + 64], ((0, 0), (0, 64)))).astype(jnp.bfloat16), sliding)
            v = jnp.pad(qgkv[:, 2 * qwidth + 64:], ((0, 0), (0, 64)))
            values = jnp.concatenate((k[:, :64], v[:, :64]), axis=1)
            raw_queries = jnp.concatenate([jnp.pad(qgkv[:, head * 64:(head + 1) * 64], ((0, 0), (0, 64))) for head in range(nh)])
            normalized_queries = rotate(hn(raw_queries).astype(jnp.bfloat16), sliding)
            queries = [normalized_queries[head * rows:(head + 1) * rows] for head in range(nh)]

            def cached_attention(sequence, anchor, query, query_positions, new_values, count):
                write_rows = rows if batch_size is None else 1
                slot = sequence % cache_slots
                buffer = kvbuf.at[slot]
                absolute_write = pl.multiple_of(jnp.minimum(anchor // 16 * 16, context - 32), 16)

                def attend(length, base):
                    base = pl.multiple_of(base, 128)
                    copy(kvc.at[li, sequence, pl.ds(base, length)], buffer.at[:length], 8 + slot).wait()
                    local_write = pl.multiple_of(absolute_write - base, 16)
                    block = buffer[pl.ds(local_write, 32)]
                    for row in range(write_rows):
                        mask = (absolute_write + jnp.arange(32)[:, None] == anchor + row) & (row < count)
                        block = jnp.where(mask, new_values[row:row + 1], block)
                    buffer[pl.ds(local_write, 32)] = block
                    copy(buffer.at[pl.ds(local_write, 32)], kvc.at[li, sequence, pl.ds(absolute_write, 32)], 10).start()
                    key_positions = base + jnp.arange(length)[None]
                    keep = (key_positions <= query_positions) & ((key_positions > query_positions - c.sliding_window) | ~sliding)
                    scores = jnp.where(jnp.tile(keep.astype(jnp.int32), (nh, 1)) != 0, dot(query, buffer[:length].T) * c.scale, -1e30)
                    exponent = jnp.exp(scores - jnp.max(scores, axis=1, keepdims=True)).astype(jnp.bfloat16)
                    value = (dot(exponent, buffer[:length]) / jnp.sum(exponent.astype(jnp.float32), axis=1, keepdims=True)).astype(jnp.bfloat16)
                    copy(buffer.at[pl.ds(local_write, 32)], kvc.at[li, sequence, pl.ds(absolute_write, 32)], 10).wait()
                    return jnp.pad(value[:, 64:], ((0, 0), (0, 64)))

                return cache_region(anchor, count, attend)

            if batch_size in (None, 1):
                attended = cached_attention(jnp.int32(0), meta[0], jnp.concatenate(queries), pos, values, meta[1])
            else:
                query_blocks = jnp.stack(queries, axis=1)
                results = []
                for first in range(0, batch_size, attention_group):
                    anchor = meta[first]
                    for seq in range(first + 1, first + attention_group):
                        anchor = jnp.maximum(anchor, meta[seq])

                    def group_attention(length, unused_base):
                        bases = [pl.multiple_of(jnp.where(sliding, jnp.clip((meta[first + seq] - c.sliding_window + 1) // 128 * 128, 0, context - length), 0), 128) for seq in range(attention_group)]
                        writes = [pl.multiple_of(jnp.minimum(meta[first + seq] // 16 * 16, context - 32), 16) for seq in range(attention_group)]
                        for seq in range(attention_group):
                            copy(kvc.at[li, first + seq, pl.ds(bases[seq], length)], kvbuf.at[seq, :length], 8 + seq).start()
                        for seq in range(attention_group):
                            copy(kvc.at[li, first + seq, pl.ds(bases[seq], length)], kvbuf.at[seq, :length], 8 + seq).wait()
                            local_write = pl.multiple_of(writes[seq] - bases[seq], 16)
                            block = kvbuf[seq, pl.ds(local_write, 32)]
                            mask = (writes[seq] + jnp.arange(32)[:, None] == meta[first + seq]) & (meta[rows + first + seq] != 0)
                            kvbuf[seq, pl.ds(local_write, 32)] = jnp.where(mask, values[first + seq:first + seq + 1], block)
                            copy(kvbuf.at[seq, pl.ds(local_write, 32)], kvc.at[li, first + seq, pl.ds(writes[seq], 32)], 16 + seq).start()
                        query = query_blocks[first:first + attention_group].reshape(attention_group * nh, 128)
                        keys = kvbuf[:, :length].reshape(attention_group * length, 128)
                        key_positions = jnp.concatenate([base + jnp.arange(length)[None] for base in bases], axis=1)
                        query_positions = jnp.repeat(pos[first:first + attention_group], nh, axis=0)
                        own_sequence = jnp.arange(attention_group * nh)[:, None] // nh == jnp.arange(attention_group * length)[None] // length
                        keep = own_sequence & (key_positions <= query_positions) & ((key_positions > query_positions - c.sliding_window) | ~sliding)
                        scores = jnp.where(keep.astype(jnp.int32) != 0, dot(query, keys.T) * c.scale, -1e30)
                        exponent = jnp.exp(scores - jnp.max(scores, axis=1, keepdims=True)).astype(jnp.bfloat16)
                        value = (dot(exponent, keys) / jnp.sum(exponent.astype(jnp.float32), axis=1, keepdims=True)).astype(jnp.bfloat16)
                        for seq in range(attention_group):
                            local_write = pl.multiple_of(writes[seq] - bases[seq], 16)
                            copy(kvbuf.at[seq, pl.ds(local_write, 32)], kvc.at[li, first + seq, pl.ds(writes[seq], 32)], 16 + seq).wait()
                        return jnp.pad(value[:, 64:], ((0, 0), (0, 64))).reshape(attention_group, nh, 128)

                    results.append(cache_region(anchor, jnp.int32(1), group_attention))
                attended = jnp.concatenate(results, axis=0).transpose(1, 0, 2).reshape(nh * rows, 128)
            attended = hn(attended)
            gates = jnp.concatenate([jnp.pad(qgkv[:, qwidth + head * 64:qwidth + (head + 1) * 64], ((0, 0), (0, 64))) for head in range(nh)]).astype(jnp.float32)
            gated = (attended * jax.nn.sigmoid(gates)).astype(jnp.bfloat16)
            pair_value = jnp.concatenate([gated[head * rows:(head + 1) * rows, :64] for head in range(nh)], axis=1)
            pair = flat_gather(pair_ref, pair_value, lane, 8, tuple(range(7, 0, -1)), 0)
            joined = jnp.concatenate([pair[i] for i in range(8)], axis=1)
            copy(w["ow"].at[li], obuf, 1).wait()
            if rows == 1:
                attention_partial = dot(joined, obuf[...])
                copy(w["pool"].at[li], pbuf, 1).start()
                branch = direct(adirect, attention_partial, li, 0).astype(jnp.bfloat16)
            else:
                ahost[host] = dot(joined, obuf[...])
                copy(w["pool"].at[li], pbuf, 1).start()
                gather(ahost, host, (8, 16, 24), 7)
                apaired[lane] = jnp.sum(ahost[...], axis=0)
                gather(apaired, lane, tuple(range(7, 0, -1)), 10)
                branch = jnp.concatenate([apaired[i] for i in range(8)], axis=1).astype(jnp.bfloat16)
            stream = sc[3:4] * stream + sc[4:5] * norm(branch, c.post_eps)
            x = norm(stream, c.eps, sc[1:2]).astype(jnp.bfloat16)
            copy(w["pool"].at[li], pbuf, 1).wait()
            pooled = dot(x, pbuf[...]).astype(jnp.bfloat16)
            copy(w["router"].at[li], router, 2).wait()
            logits = jnp.concatenate([jnp.sum(x[i:i + 1].astype(jnp.float32) * router[...], axis=1)[None] for i in range(rows)], axis=0)
            comb_value = jnp.concatenate((tpu.bitcast(jnp.pad(logits, ((0, 0), (0, 128 - es))), jnp.bfloat16), *[pooled[:, i * 128:(i + 1) * 128] for i in range(pw // 128)]), axis=0)
            copy(w["unpool"].at[li], ubuf, 1).start()
            comb = flat_gather(comb_ref, comb_value, lane, 8, tuple(range(7, 0, -1)), 17)
            scores = jax.nn.sigmoid(jnp.concatenate([tpu.bitcast(comb[i, :router_rows], jnp.float32)[:, :es] for i in range(8)], axis=1) * c.output_multiplier)
            copy(w["router_bias"].at[li], bias, 5).wait()
            ranked = scores + bias[...][None]
            route_ids, route_weights = [], []
            requested = jnp.zeros((1, c.experts), jnp.float32)
            local_count = jnp.int32(0)
            local_ids = jnp.full((1, 128), c.experts, jnp.int32)
            for _ in range(c.top_k):
                index = _verify_router_argmax(ranked)
                selected = experts == index
                if rows == 1:
                    selected_id = index[0, 0]
                    owned = (selected_id % 4 == host) & (live.astype(jnp.int32)[0, 0] != 0)
                    local_ids = jnp.where(owned & (jnp.arange(128)[None] == local_count), selected_id, local_ids)

                    @pl.when(owned & (local_count < 4))
                    def early_expert():
                        gu, dw, gb, db, weight_index = (w["gu_dense"], w["dw_dense"], gd, dd, bi) if dense else (w["gu_pack"], w["dw_pack"], gp, dp, pi)
                        copy(gu.at[weight_index, selected_id // 4], gb.at[local_count], 12 + 2 * local_count).start()
                        copy(dw.at[weight_index, selected_id // 4], db.at[local_count], 13 + 2 * local_count).start()

                    local_count += owned.astype(jnp.int32)
                route_ids.append(index)
                route_weights.append(jnp.sum(jnp.where(router_ids == index, scores, 0), axis=1, keepdims=True))
                requested += jnp.sum((selected & live).astype(jnp.float32), axis=0, keepdims=True)
                ranked = jnp.where(router_ids == index, -jnp.inf, ranked)
            rw = jnp.concatenate(route_weights, axis=1)
            rw /= jnp.sum(rw, axis=1, keepdims=True) + c.router_eps
            requested = (requested > 0).astype(jnp.bfloat16)
            triangular = (jnp.arange(c.experts)[:, None] <= experts).astype(jnp.bfloat16)
            active = ((requested > 0) & (experts % 4 == host)).astype(jnp.int32)
            if rows == 1:
                ids_v[:128] = local_ids[0]
            else:
                prefix = dot(jnp.broadcast_to(active.astype(jnp.bfloat16), (8, c.experts)), triangular)[:1].astype(jnp.int32)
                ordinal = prefix - 1
                ids_v[:128] = jnp.min(jnp.where((active > 0) & (ordinal == jnp.arange(128)[:, None]), experts, c.experts), axis=1)
            route_slots = jnp.full((1, c.experts), c.top_k, jnp.int32)
            for k in range(c.top_k):
                selected = jnp.sum(jnp.where(r == owner, (experts == route_ids[k]).astype(jnp.int32), 0), axis=0, keepdims=True) > 0
                route_slots = jnp.where(selected, k, route_slots)
            ids_v[128:128 + c.experts] = route_slots[0]
            copy(ids_v, ids_s, 20).start()

            def run_experts(packed):
                gu, dw, gb, db, index, width = (w["gu_pack"], w["dw_pack"], gp, dp, pi, iw) if packed else (w["gu_dense"], w["dw_dense"], gd, dd, bi, di)

                def issue(e, slot):
                    local_e = e // 4
                    copy(gu.at[index, local_e], gb.at[slot], 12 + 2 * slot).start()
                    copy(dw.at[index, local_e], db.at[slot], 13 + 2 * slot).start()

                def dequant(blob, k, n):
                    values = tpu.bitcast(blob[:k // 2], jnp.float4_e2m1fn).astype(jnp.bfloat16)
                    scales = jax.lax.bitcast_convert_type(blob[k // 2:], jnp.float8_e4m3fn).astype(jnp.bfloat16)
                    return (values.reshape(k // 16, 16, n) * scales[:, None]).reshape(k, n)

                count = jnp.sum(active.astype(jnp.int32))
                copy(ids_v, ids_s, 20).wait()
                prefetch = 4 if packed else dense_slots
                if rows != 1:
                    for first_id in range(prefetch):
                        @pl.when(count > first_id)
                        def first():
                            issue(ids_s[first_id], jnp.int32(first_id))

                copy(w["expert_norms"].at[li], norms, 4).wait()
                pooled = norm(jnp.concatenate([comb[i, router_rows + j * rows:router_rows + (j + 1) * rows] for i in range(8) for j in range(pw // 128)], axis=1), c.eps, norms[0:1]).astype(jnp.bfloat16)
                if packed:
                    copy(w["gu_global"].at[pi], gg, 6).wait()
                    copy(w["dw_global"].at[pi], dg, 7).wait()

                def row_transfers(group):
                    # Four packets prevent reuse before every peer consumes a row.
                    slot = group % 4
                    return [remote(estage.at[slot, lane ^ offset], epipe.at[slot, lane], 24 + slot * 7 + offset - 1, rank ^ offset) for offset in range(7, 0, -1)]

                def finish_rows(group):
                    for transfer in row_transfers(group):
                        transfer.wait()
                    full = jnp.sum(epipe[group % 4].astype(jnp.float32), axis=0).reshape(2, mh)
                    for part in range(2):
                        @pl.when(group * 2 + part < count)
                        def store_row():
                            pipe_rows[ids_s[128 + ids_s[group * 2 + part]]] = full[part:part + 1]

                def step(i, unused):
                    e, slot = ids_s[i], i % prefetch
                    local_e = e // 4
                    copy(gu.at[index, local_e], gb.at[slot], 12 + 2 * slot).wait()
                    projected = dot(pooled, dequant(gb[slot], mh, 2 * width) if packed else gb[slot])
                    if packed:
                        gate_scale = jax.lax.bitcast_convert_type(jnp.full((1, 1), gg[2 * e], jnp.int32), jnp.float32)
                        up_scale = jax.lax.bitcast_convert_type(jnp.full((1, 1), gg[2 * e + 1], jnp.int32), jnp.float32)
                        gate = (projected[:, :width] * gate_scale).astype(jnp.bfloat16).astype(jnp.float32)
                        up = (projected[:, width:] * up_scale).astype(jnp.bfloat16).astype(jnp.float32)
                    else:
                        gate, up = projected[:, :width], projected[:, width:]
                    act = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
                    copy(dw.at[index, local_e], db.at[slot], 13 + 2 * slot).wait()
                    down_weights = dequant(db[slot], width, mh) if packed else db[slot]
                    value = dot(act, down_weights)
                    if packed:
                        value *= jax.lax.bitcast_convert_type(jnp.full((1, 1), dg[e], jnp.int32), jnp.float32)
                    block_values[i % 2] = value.astype(jnp.bfloat16)
                    @pl.when((i % 2 == 1) | (i + 1 == count))
                    def send_pair():
                        group = i // 2
                        estage[group % 4] = jnp.tile(block_values[...].transpose(1, 0, 2).reshape(rows, mh // 128, 256), (8 // rows, 1, 1))
                        epipe[group % 4, lane] = estage[group % 4, owner]
                        for transfer in row_transfers(group):
                            transfer.start()
                        @pl.when(group > 0)
                        def finish_previous():
                            finish_rows(group - 1)
                    @pl.when(i + prefetch < count)
                    def following():
                        issue(ids_s[i + prefetch], slot)
                    return unused

                pipe_rows[...] = jnp.zeros(pipe_rows.shape, jnp.float32)
                block_values[...] = jnp.zeros(block_values.shape, jnp.bfloat16)
                jax.lax.fori_loop(0, count, step, None)
                @pl.when(count > 0)
                def finish_last():
                    finish_rows((count - 1) // 2)

            if dense:
                run_experts(False)
            else:
                run_experts(True)
            @pl.when(li + 1 < c.layers)
            def prefetch_q():
                copy(w["qwkv"].at[li + 1], qbuf, 0).start()
                copy(w["scalars"].at[li + 1], scalars, 3).start()
            @pl.when(li + 1 == c.layers)
            def prefetch_head():
                copy(w["fnorm"].at[0], scalars.at[7], 3).start()
                for slot in range(2):
                    copy(w["lmw"].at[:, pl.ds(slot * lm_chunk, lm_chunk)], lm.at[slot], slot * 16).start()
            full = pipe_rows[:c.top_k, 0]
            weights = jnp.sum(jnp.where(r == owner, rw, 0), axis=0)[:, None]
            mixed = jnp.sum(norm(full * norms[1:2], c.post_eps) * weights, axis=0, keepdims=True)
            mixed_slice = pl.ds(pl.multiple_of(owner * mixed_stride, 16), mixed_stride)
            mixed_rows[mixed_slice] = jnp.pad(mixed.astype(jnp.bfloat16), ((0, 0), (0, mixed_stride * 128 - mh))).reshape(mixed_stride, 128)
            transfers = [remote(mixed_rows.at[mixed_slice], mixed_rows.at[mixed_slice], 52 + offset - 1, rank ^ offset) for offset in range(rows - 1, 0, -1)]
            for transfer in transfers:
                transfer.start()
            for transfer in transfers:
                transfer.wait()
            copy(w["unpool"].at[li], ubuf, 1).wait()
            projected = dot(mixed_rows[...].reshape(rows, mixed_stride * 128)[:, :mh], ubuf[...]).astype(jnp.bfloat16)
            @pl.when(li + 1 < c.layers)
            def prefetch_o():
                copy(w["ow"].at[li + 1], obuf, 1).start()
            if rows <= 2:
                branch = direct(udirect, projected, li, 1)
            else:
                unhost[host] = projected
                gather(unhost, host, (8, 16, 24), 59)
                ungather[lane] = jnp.sum(unhost[...].astype(jnp.float32), axis=0)
                gather(ungather, lane, tuple(range(7, 0, -1)), 62)
                branch = jnp.concatenate([ungather[i] for i in range(8)], axis=1)
            stream = sc[5:6] * stream + sc[6:7] * norm(branch * sc[2:3], c.post_eps)
            for i, layer_id in enumerate(aux_layers):
                @pl.when(li == layer_id)
                def capture():
                    aux[i] = stream.astype(jnp.bfloat16)
            return stream

        stream = initial[...]
        begin = 0
        for count, dense_index in enumerate(c.bf16_expert_layers):
            if begin < dense_index:
                stream = jax.lax.fori_loop(begin, dense_index, lambda li, stream: layer(li, stream, False, count), stream)
            stream = layer(dense_index, stream, True, count)
            begin = dense_index + 1
        if begin < c.layers:
            stream = jax.lax.fori_loop(begin, c.layers, lambda li, stream: layer(li, stream, False, len(c.bf16_expert_layers)), stream)
        copy(w["fnorm"].at[0], scalars.at[7], 3).wait()
        hidden = (norm(stream, c.eps) * scalars[7:8]).astype(jnp.bfloat16)

        def head_chunk(index, carry):
            best, best_index = carry
            slot = index % 2
            offset = index * lm_chunk
            copy(w["lmw"].at[:, pl.ds(offset, lm_chunk)], lm.at[slot], slot * 16).wait()
            logits = dot(hidden, lm[slot])
            columns = offset + jnp.arange(lm_chunk)[None]
            logits = jnp.where(columns < c.vocab // 32, logits, -jnp.inf)
            local_best = jnp.max(logits, axis=1, keepdims=True)
            local_index = jnp.min(jnp.where(logits == local_best, columns + rank * (c.vocab // 32), c.vocab), axis=1, keepdims=True)
            best_index = jnp.where(local_best > best, local_index, jnp.where(local_best == best, jnp.minimum(best_index, local_index), best_index))
            best = jnp.maximum(best, local_best)
            @pl.when(index + 2 < head_width // lm_chunk)
            def prefetch():
                copy(w["lmw"].at[:, pl.ds((index + 2) * lm_chunk, lm_chunk)], lm.at[slot], slot * 16).start()
            return best, best_index

        best, best_index = jax.lax.fori_loop(0, head_width // lm_chunk, head_chunk,
            (jnp.full((rows, 1), -jnp.inf, jnp.float32), jnp.full((rows, 1), c.vocab, jnp.int32)))

        def pack(best, index):
            return jnp.concatenate((jnp.broadcast_to(best, (rows, 64)), jnp.broadcast_to(index.astype(jnp.float32), (rows, 64))), axis=1)

        def winner(start, count):
            best = jnp.max(candidates[start:start + count, :, :1], axis=0)
            index = jnp.min(jnp.where(candidates[start:start + count, :, :1] == best[None], candidates[start:start + count, :, 64:65], c.vocab), axis=0)
            return best, index

        candidates[lane] = pack(best, best_index)
        gather(candidates, lane, tuple(range(7, 0, -1)), 73)
        best, best_index = winner(0, 8)
        candidates[8 + host] = pack(best, best_index)
        gather(candidates, 8 + host, (8, 16, 24), 80)
        best, best_index = winner(8, 4)
        final[...] = jnp.broadcast_to(best_index.astype(jnp.int32), (rows, 128))

    def local(global_weights, states, tokens, position):
        w = jax.tree.map(lambda a: a[0], global_weights)
        if w["gu_pack"].shape[1] != c.experts // 4:
            raise ValueError("decode requires the TP8 x EP4 expert partition")
        if w["qwkv"].shape[1:] != qshape:
            raise ValueError("weights were prepared for a different attention layout")
        w["gu_global"] = jax.lax.bitcast_convert_type(w["gu_global"], jnp.int32).reshape(-1, 2 * c.experts)
        w["dw_global"] = jax.lax.bitcast_convert_type(w["dw_global"], jnp.int32)
        kvc = states["kvcache"][0]
        stream = rms_plain_f32(w["emb"][tokens], c.eps)
        v = lambda shape, dtype=jnp.bfloat16: tpu.VMEM(shape, dtype)
        shapes = (jax.ShapeDtypeStruct((rows, 128), jnp.int32), jax.ShapeDtypeStruct((max(1, len(aux_layers)), rows, h), jnp.bfloat16), jax.ShapeDtypeStruct(kvc.shape, kvc.dtype))
        n = len(_VERIFY_WEIGHTS)
        stream, aux, kvc = pl.pallas_call(
            kernel, out_shape=shapes,
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[pl.BlockSpec(memory_space=tpu.HBM)] * n + [pl.BlockSpec(), pl.BlockSpec(memory_space=tpu.HBM)],
                out_specs=[pl.BlockSpec(), pl.BlockSpec(), pl.BlockSpec(memory_space=tpu.HBM)],
                scratch_shapes=(v((qt, 32, 128), jnp.uint8), v((pt, 32, 128), jnp.uint8),
                    v((es, h), jnp.float32), v((8, h), jnp.float32), v((2, mh), jnp.float32), v((c.experts,), jnp.float32),
                    tpu.SMEM((2 * c.experts,), jnp.int32), tpu.SMEM((c.experts,), jnp.int32),
                    v((8 * ((rows * qwidth + 255) // 256 * 256),)), v((4, rows, h // 8), jnp.float32), v((8, rows, h // 8), jnp.float32), v((2, 32 * attention_rows, 128), jnp.float32), v((2 * 32 * ((rows * (h // 8) + 255) // 256 * 256),)), v((8 * (((router_rows + pool_rows) * 128 + 255) // 256 * 256),)),
                    v((2, rows, mh)), v((4, 8, mh // 128, 256)), v((4, 8, mh // 128, 256)), v(((c.top_k + 8) // 8 * 8, 1, mh), jnp.float32), v((rows * mixed_stride, 128)),
                    v((4, rows, h // 8)), v((8, rows, h // 8), jnp.float32),
                    v((128 + c.experts,), jnp.int32), tpu.SMEM((128 + c.experts,), jnp.int32),
                    v((12, rows, 128), jnp.float32), tpu.SemaphoreType.DMA((max(21, 16 + attention_group),)), tpu.SemaphoreType.DMA((83,)), tpu.SemaphoreType.DMA((83,))),
            ),
            input_output_aliases={n + 2: 2},
            compiler_params=tpu.CompilerParams(collective_id=52, vmem_limit_bytes=64 * 1024**2, shape_invariant_numerics=True),
            name="mpk_verify" if batch_size is None else "mpk_decode",
        )(position, *[w[key] for key in _VERIFY_WEIGHTS], stream, kvc)
        targets = stream[:, 0]
        return targets, aux.transpose(1, 0, 2), {"kvcache": kvc[None]}

    return jax.jit(jax.shard_map(local, mesh=mesh, in_specs=(P("tp"), P("tp"), P(), P()), out_specs=(P(), P(), P("tp")), check_vma=False), donate_argnums=(1,))


def logical_mesh32():
    devices = sorted(jax.devices(), key=lambda device: (device.process_index, device.id))
    if len(devices) != 32:
        raise ValueError(f"the target requires 32 logical devices, found {len(devices)}")
    return jax.sharding.Mesh(np.asarray(devices), ("tp",))


def prepare_decode_weights(mesh, weights):
    from .load import prepare_expert_parallel_weights
    return prepare_expert_parallel_weights(mesh, weights)


def prepare_verify_weights(mesh, weights):
    return prepare_decode_weights(mesh, weights)


def prepare_verify_cache(states):
    return states


def make_verify_block(mesh, c, context, rows, aux_layers):
    block = _make_block(mesh, c, context, rows, aux_layers)

    def run(weights, states, tokens, position):
        return block(weights, states, tokens, jnp.stack((position, jnp.int32(rows))))

    return jax.jit(run, donate_argnums=(1,))


def make_verify(mesh, c, context, rows, aux_layers):
    block = _make_block(mesh, c, context, 1, aux_layers)

    def run(weights, states, tokens, control):
        def step(i, carry):
            states, targets, aux = carry
            predicted, captured, states = block(weights, states, jax.lax.dynamic_slice(tokens, (i,), (1,)), jnp.stack((control[0] + i, jnp.int32(1))))
            return states, targets.at[i].set(predicted[0]), aux.at[i].set(captured[0])

        states, targets, aux = jax.lax.fori_loop(0, control[1], step,
            (states, jnp.zeros((rows,), jnp.int32), jnp.zeros((rows, len(aux_layers), c.dim), jnp.bfloat16)))
        return targets, aux, states

    return jax.jit(run, donate_argnums=(1,))


def make_decode(mesh, c, context, tp=32, ablate=()):
    if tp != 32 or ablate:
        raise ValueError("decode requires the 32-device layout without ablations")
    block = _make_block(mesh, c, context, 1, ())

    def run(weights, states, token, position):
        predicted, _, states = block(weights, states, token[:1], jnp.stack((position[0], jnp.int32(1))))
        return predicted, jnp.zeros((32, 1), jnp.float32), states

    return jax.jit(run, donate_argnums=(1,))


def make_decode_batch(mesh, c, context, batch_size):
    if batch_size not in (1, 2, 4, 8):
        raise ValueError("batch size must be 1, 2, 4, or 8")
    rows = batch_size
    block = _make_block(mesh, c, context, rows, (), batch_size=batch_size)

    def run(weights, states, tokens, positions):
        live = (positions >= 0) & (positions < context)
        metadata = jnp.concatenate((jnp.pad(jnp.maximum(positions, 0), (0, rows - batch_size)), jnp.pad(live.astype(jnp.int32), (0, rows - batch_size))))
        predicted, _, states = block(weights, states, jnp.pad(tokens, (0, rows - batch_size)), metadata)
        return jnp.where(live, predicted[:batch_size], tokens), states

    return jax.jit(run, donate_argnums=(1,))


def make_decode_batch_chunk(mesh, c, context, batch_size, chunk_size=4):
    if chunk_size != 4:
        raise ValueError("steps_per_call must remain four")
    decode = make_decode_batch(mesh, c, context, batch_size)

    def run(weights, states, tokens, control):
        positions, counts = control[0], jnp.clip(control[1], 0, 4)
        counts = jnp.minimum(counts, jnp.maximum(context - positions, 0))

        def step(i, carry):
            tokens, states, emitted = carry
            active = (i < counts) & (positions >= 0)
            tokens, states = decode(weights, states, tokens, jnp.where(active, positions + i, -1))
            return tokens, states, emitted.at[i].set(jnp.where(active, tokens, -1))

        return jax.lax.fori_loop(0, jnp.max(counts), step,
            (tokens, states, jnp.full((4, batch_size), -1, jnp.int32)))

    return jax.jit(run, donate_argnums=(1,))


def make_decode_chunk(mesh, c, context, chunk_size, tp=32, ablate=()):
    if chunk_size != 4:
        raise ValueError("steps_per_call must remain four")
    decode = make_decode(mesh, c, context, tp, ablate)

    def run(weights, states, token, control):
        def step(i, carry):
            token, states, emitted = carry
            token, _, states = decode(weights, states, token, control[:1] + i)
            return token, states, emitted.at[i].set(token[0])

        return jax.lax.fori_loop(0, control[1], step, (token, states, jnp.zeros((4,), jnp.int32)))

    return jax.jit(run, donate_argnums=(1,))


def _prefill_router_topk(ranked, top_k):
    if jax.default_backend() == "cpu":
        return jax.lax.top_k(ranked, top_k)[1]
    length, experts = ranked.shape
    block = min(128, length)

    def kernel(x, out):
        values = x[...]
        indices = jnp.arange(experts, dtype=jnp.int32)[None, :]
        result = jnp.zeros((block, top_k), jnp.int32)
        for i in range(top_k):
            maximum = jnp.max(values, axis=1, keepdims=True)
            chosen = jnp.min(jnp.where(values == maximum, indices, experts), axis=1, keepdims=True)
            result = jnp.where(jnp.arange(top_k)[None, :] == i, chosen, result)
            values = jnp.where(indices == chosen, -jnp.inf, values)
        out[...] = result

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((length, top_k), jnp.int32),
        grid=(pl.cdiv(length, block),),
        in_specs=(pl.BlockSpec((block, experts), lambda i: (i, 0)),),
        out_specs=pl.BlockSpec((block, top_k), lambda i: (i, 0)),
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel",)),
    )(ranked)


def _prefill_exchange_down(x):
    """Exchange one down-projection layer from input to output shards."""
    experts, depth, width = x.shape
    columns = width // 8
    if jax.default_backend() == "cpu":
        value = x.reshape(experts, depth, 8, columns).transpose(2, 0, 1, 3)
        groups = tuple(tuple(range(8 * host, 8 * (host + 1))) for host in range(4))
        return jax.lax.all_to_all(value, "tp", split_axis=0, concat_axis=0, axis_index_groups=groups, tiled=True)

    def kernel(src, out, sends, recvs):
        rank = jax.lax.axis_index("tp")
        barrier = tpu.get_barrier_semaphore()
        for offset in range(1, 8):
            pl.semaphore_signal(barrier, 1, device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(barrier, 7)
        local = tpu.make_async_copy(src.at[:, :, pl.ds((rank % 8) * columns, columns)], out.at[rank % 8], recvs.at[0])
        local.start()
        transfers = []
        for offset in range(1, 8):
            peer = rank ^ offset
            transfer = tpu.make_async_remote_copy(
                src.at[:, :, pl.ds((peer % 8) * columns, columns)], out.at[rank % 8],
                sends.at[offset], recvs.at[offset],
                device_id=(peer,), device_id_type=pl.DeviceIdType.MESH,
            )
            transfer.start()
            transfers.append(transfer)
        local.wait()
        for transfer in transfers:
            transfer.wait()

    out = pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((8, experts, depth, columns), x.dtype),
        in_specs=(pl.BlockSpec(memory_space=tpu.HBM),),
        out_specs=pl.BlockSpec(memory_space=tpu.HBM),
        scratch_shapes=(tpu.SemaphoreType.DMA((8,)), tpu.SemaphoreType.DMA((8,))),
        compiler_params=tpu.CompilerParams(collective_id=0),
    )(x)
    return out


def _prefill_expand_down(packets, globals_):
    shards, count, depth, n = packets.shape
    k = depth // 9 * 16
    if jax.default_backend() == "cpu":
        return jnp.concatenate([_dequant_experts_tiled(p[:, :k // 2], p[:, k // 2:], globals_, count, k, n) for p in packets], axis=1)
    be = min(8, count)
    gains = jnp.broadcast_to(globals_.reshape(count, 1, 1), (count, 1, n))

    def kernel(p, g, out):
        values = tpu.bitcast(p[0, :, :k // 2], jnp.float4_e2m1fn).astype(F32)
        scale = jax.lax.bitcast_convert_type(p[0, :, k // 2:], jnp.float8_e4m3fn).astype(F32)
        scale = scale * g[:, 0, :][:, None, :]
        out[...] = (values.reshape(be, k // 16, 16, n) * scale[:, :, None, :]).reshape(be, k, n).astype(jnp.bfloat16)

    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((count, shards * k, n), jnp.bfloat16),
        grid=(count // be, shards),
        in_specs=(pl.BlockSpec((1, be, depth, n), lambda e, s: (s, e, 0, 0)),
                  pl.BlockSpec((be, 1, n), lambda e, s: (e, 0, 0))),
        out_specs=pl.BlockSpec((be, k, n), lambda e, s: (e, s, 0)),
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )(packets, gains)


def _prefill_flash(query, key, value, padding, scale, window):
    length, heads, dim = query.shape
    bq, bk = 1024, 2048
    length_padded = (length + bk - 1) // bk * bk
    query = jnp.pad(query, ((0, length_padded - length), (0, 0), (0, 0))).transpose(1, 0, 2)
    key = jnp.pad(key[:, 0], ((0, length_padded - length), (0, 0)))
    value = jnp.pad(value[:, 0], ((0, length_padded - length), (0, 0)))
    blocks = length_padded // bk if window is None else min(length_padded // bk, (window + bq + bk - 2) // bk + 1)

    def offset(q, k):
        if window is None:
            return k
        return jnp.minimum(jnp.maximum(0, (q * bq - window + 1) // bk), length_padded // bk - blocks) + k

    def kernel(q, k, v, pad, out, maximum, total, accumulator):
        qi, ki = pl.program_id(1), pl.program_id(2)
        start = offset(qi, ki) * bk
        @pl.when(ki == 0)
        def initialize():
            maximum[...] = jnp.full(maximum.shape, -jnp.inf, jnp.float32)
            total[...] = jnp.zeros(total.shape, jnp.float32)
            accumulator[...] = jnp.zeros(accumulator.shape, jnp.float32)
        @pl.when(start < (qi + 1) * bq)
        def attend():
            scores = jnp.dot(q[0], k[...].T, preferred_element_type=jnp.float32) * scale
            qr = qi * bq + jnp.arange(bq)[:, None]
            kr = start + jnp.arange(bk)[None, :]
            keep = (kr <= qr) & (kr >= pad[0, :1]) & (kr < length)
            if window is not None:
                keep &= kr > qr - window
            scores = jnp.where(keep, scores, jnp.finfo(jnp.float32).min)
            next_max = jnp.maximum(maximum[...], jnp.max(scores, axis=-1, keepdims=True))
            alpha = jnp.exp(maximum[...] - next_max)
            probabilities = jnp.exp(scores - next_max)
            next_total = total[...] * alpha + jnp.sum(probabilities, axis=-1, keepdims=True)
            product = jnp.dot(probabilities.astype(jnp.bfloat16), v[...], preferred_element_type=jnp.float32)
            accumulator[...] = accumulator[...] * alpha + product
            maximum[...], total[...] = next_max, next_total
        @pl.when(ki == blocks - 1)
        def finish():
            out[0] = accumulator[...] / total[...]

    result = pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct(query.shape, jnp.float32),
        grid=(heads, length_padded // bq, blocks),
        in_specs=(pl.BlockSpec((1, bq, dim), lambda h, q, k: (h, q, 0)),
                  pl.BlockSpec((bk, dim), lambda h, q, k: (offset(q, k), 0)),
                  pl.BlockSpec((bk, dim), lambda h, q, k: (offset(q, k), 0)),
                  pl.BlockSpec((1, 128), lambda h, q, k: (0, 0))),
        out_specs=pl.BlockSpec((1, bq, dim), lambda h, q, k: (h, q, 0)),
        scratch_shapes=(tpu.VMEM((bq, 1), jnp.float32), tpu.VMEM((bq, 1), jnp.float32), tpu.VMEM((bq, dim), jnp.float32)),
        compiler_params=tpu.CompilerParams(dimension_semantics=('parallel', 'parallel', 'arbitrary')),
    )(query, key, value, jnp.full((1, 128), padding, jnp.int32))
    return result.transpose(1, 0, 2)[:length]


def make_prefill(mesh, c, context, tp, *, aux_layers=()):
    """Routed prefill using the decode weights and combined BF16 cache."""
    if mesh.size != 32 or tp != 32:
        raise ValueError("prefill requires the 32-device expert-parallel layout")
    qh, kvh = c.heads // 32, 1
    mh_r, expert_count = c.moe_hidden // 8, c.experts // 4
    hosts = tuple(tuple(range(h * 8, (h + 1) * 8)) for h in range(4))
    owners = tuple(tuple(h * 8 + lane for h in range(4)) for lane in range(8))
    attention_pairs = owners

    def local(w, states, tokens, npad):
        w = jax.tree.map(lambda a: a[0], w)
        states = jax.tree.map(lambda a: a[0], states)
        rank = jax.lax.axis_index("tp")
        batch = 1 if tokens.ndim == 1 else tokens.shape[0]
        length = tokens.shape[-1]
        L = batch * length
        padding = jnp.broadcast_to(npad, (batch,))
        sequence = jnp.arange(L, dtype=jnp.int32) // length
        tokens = tokens.reshape(L)
        positions = jnp.tile(jnp.arange(length, dtype=jnp.int32), batch) - padding[sequence]
        real = positions >= 0
        stream = w["emb"][tokens].astype(F32)
        value = stream
        value = value * jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + c.eps)
        stream = jnp.where(real[:, None], value, jnp.zeros((), F32))
        slots = jnp.clip(positions, -context, context - 1)
        cos, sin = rope_tables_shard(c, positions)

        def one_layer(carry, li):
            stream, cache_all, *aux = carry
            li = li.astype(jnp.int32)
            sliding = (li % 4) != 1
            is_dense = jnp.bool_(False)
            for d in c.bf16_expert_layers:
                is_dense = is_dense | (li == d)
            dense_below = jnp.zeros((), jnp.int32)
            for d in c.bf16_expert_layers:
                dense_below = dense_below + (li > d).astype(jnp.int32)
            o_packed = jnp.clip(li - dense_below, 0, max(c.layers - len(c.bf16_expert_layers) - 1, 0))
            o_dense = jnp.clip(dense_below, 0, len(c.bf16_expert_layers) - 1)
            sc = w["scalars"][li]

            x = rms_gain_shard(stream, sc[0], c.eps)
            qwkv = jnp.dot(x, w["qwkv"][li], preferred_element_type=F32).astype(jnp.bfloat16)
            query = qwkv[:, : qh * c.head_dim].reshape(L, qh, c.head_dim)
            gate = qwkv[:, qh * c.head_dim : 2 * qh * c.head_dim].reshape(L, qh, c.head_dim)
            kv_off = 2 * qh * c.head_dim
            key = qwkv[:, kv_off : kv_off + kvh * c.head_dim].reshape(L, kvh, c.head_dim)
            value = qwkv[:, kv_off + kvh * c.head_dim :].reshape(L, kvh, c.head_dim)
            query = rms_plain_shard(query, c.eps)
            key = rms_plain_shard(key, c.eps)
            query = jnp.where(sliding, apply_rope_shard(query, cos, sin).astype(F32), query.astype(F32)).astype(jnp.bfloat16)
            key = jnp.where(sliding, apply_rope_shard(key, cos, sin).astype(F32), key.astype(F32)).astype(jnp.bfloat16)

            cache = cache_all[li]
            key_value = jnp.concatenate((key, value), axis=-1).reshape(L, 128)
            if batch == 1:
                cache = cache.at[0, slots].set(key_value, mode="drop")
            else:
                cache = cache.at[sequence, slots].set(key_value, mode="drop")
            cache_all = cache_all.at[li].set(cache)

            def sequence_attention(query, key, value, npad):
                if length > 2048 and jax.default_backend() != "cpu":
                    return jax.lax.cond(
                        sliding,
                        lambda: _prefill_flash(query, key, value, npad, c.scale, c.sliding_window),
                        lambda: _prefill_flash(query, key, value, npad, c.scale, None),
                    )
                L = length
                positions = jnp.arange(L, dtype=jnp.int32) - npad
                real = positions >= 0
                def window_attention():
                    block_q = 512
                    padded = (L + block_q - 1) // block_q * block_q
                    block_k = min(padded, (c.sliding_window + block_q + 126) // 128 * 128)
                    q = jnp.pad(query, ((0, padded - L), (0, 0), (0, 0)))
                    k = jnp.pad(key, ((0, padded - L), (0, 0), (0, 0)))
                    v = jnp.pad(value, ((0, padded - L), (0, 0), (0, 0)))
                    starts = jnp.clip(jnp.arange(padded // block_q) * block_q - c.sliding_window + 1, 0, padded - block_k)
                    kb = jax.vmap(lambda i: jax.lax.dynamic_slice(k, (i, 0, 0), (block_k, kvh, c.head_dim)))(starts).transpose(0, 2, 1, 3)
                    vb = jax.vmap(lambda i: jax.lax.dynamic_slice(v, (i, 0, 0), (block_k, kvh, c.head_dim)))(starts).transpose(0, 2, 1, 3)
                    qb = q.reshape(padded // block_q, block_q, kvh, qh // kvh, c.head_dim).transpose(0, 2, 3, 1, 4)
                    scores = jnp.einsum("bgqld,bgtd->bgqlt", qb.astype(F32), kb.astype(F32)) * c.scale
                    query_positions = jnp.arange(padded).reshape(-1, block_q)
                    key_positions = starts[:, None] + jnp.arange(block_k)[None, :]
                    keep = (key_positions[:, None, :] <= query_positions[:, :, None]) & (key_positions[:, None, :] > query_positions[:, :, None] - c.sliding_window) & (key_positions[:, None, :] >= npad)
                    scores = jnp.where(keep[:, None, None], scores, -jnp.inf)
                    scores = jnp.where((query_positions >= npad)[:, None, None, :, None], scores, 0.0)
                    probs = jax.nn.softmax(scores, axis=-1).astype(jnp.bfloat16)
                    out = jnp.einsum("bgqlt,bgtd->bgqld", probs.astype(F32), vb.astype(F32))
                    return out.transpose(0, 3, 1, 2, 4).reshape(padded, qh, c.head_dim)[:L]

                def full_attention():
                    grouped = query.transpose(1, 0, 2).reshape(kvh, qh // kvh, L, c.head_dim)
                    keys_t = key.transpose(1, 0, 2)  # padded-row order; mask by absolute position
                    values_t = value.transpose(1, 0, 2)
                    scores = jnp.einsum("gqld,gtd->gqlt", grouped.astype(F32), keys_t.astype(F32))
                    scores = scores * c.scale
                    key_abs = jnp.arange(L) - npad
                    lo_bound = jnp.where(sliding, positions - c.sliding_window, jnp.full_like(positions, -2147483648))
                    keep = (key_abs[None, :] <= positions[:, None]) & (key_abs[None, :] > lo_bound[:, None]) & (key_abs[None, :] >= 0)
                    scores = jnp.where(keep[None, None], scores, -jnp.inf)
                    scores = jnp.where(real[None, None, :, None], scores, 0.0)
                    probs = jax.nn.softmax(scores, axis=-1).astype(jnp.bfloat16)
                    out = jnp.einsum("gqlt,gtd->gqld", probs.astype(F32), values_t.astype(F32))
                    out = out.reshape(kvh, qh // kvh, L, c.head_dim).transpose(2, 0, 1, 3)
                    out = out.reshape(L, qh, c.head_dim)
                    return out

                if L > c.sliding_window:
                    out = jax.lax.cond(sliding, window_attention, full_attention)
                else:
                    out = full_attention()
                return out

            if batch == 1:
                out = sequence_attention(query, key, value, padding[0])
            else:
                out = jax.vmap(sequence_attention)(
                    query.reshape(batch, length, qh, c.head_dim),
                    key.reshape(batch, length, kvh, c.head_dim),
                    value.reshape(batch, length, kvh, c.head_dim), padding,
                ).reshape(L, qh, c.head_dim)
            out = rms_plain_shard(out, c.eps) * jax.nn.sigmoid(gate.astype(F32))
            out = out.astype(jnp.bfloat16).reshape(L, qh * c.head_dim)
            out = jax.lax.all_gather(out, "tp", axis=-1, tiled=True, axis_index_groups=hosts)
            partial = jnp.dot(out, w["ow"][li], preferred_element_type=F32)
            branch = jax.lax.psum(partial, "tp", axis_index_groups=attention_pairs).astype(jnp.bfloat16)
            branch = jax.lax.all_gather(branch, "tp", axis=-1, tiled=True, axis_index_groups=hosts)
            stream = sc[3] * stream + sc[4] * rms_plain_f32(branch, c.post_eps)
            stream = jnp.where(real[:, None], stream, jnp.zeros((), F32))

            x = rms_gain_shard(stream, sc[1], c.eps)
            rlogits = jnp.dot(x.astype(F32), w["router"][li].T, precision=jax.lax.Precision.HIGHEST)
            rfull = jax.lax.all_gather(rlogits, "tp", axis=-1, tiled=True, axis_index_groups=hosts)
            scores_r = jax.nn.sigmoid(rfull * c.output_multiplier)
            ranked = scores_r + w["router_bias"][li].astype(F32)
            inverse_order = jnp.argsort(_verify_router_order(c.experts))
            ranked, scores_r = ranked[:, inverse_order], scores_r[:, inverse_order]
            ids = _prefill_router_topk(ranked, c.top_k)
            rw = jnp.take_along_axis(scores_r, ids, axis=-1)
            rw = rw / (jnp.sum(rw, axis=-1, keepdims=True) + c.router_eps)

            pooled_part = jnp.dot(x, w["pool"][li], preferred_element_type=F32).astype(jnp.bfloat16)
            pooled = jax.lax.all_gather(pooled_part, "tp", axis=-1, tiled=True, axis_index_groups=hosts)
            pooled = rms_gain_shard(pooled, w["expert_norms"][li, 0], c.eps)

            def dense_experts():
                down = _prefill_exchange_down(w["dw_dense"][o_dense])
                down = down.transpose(1, 0, 2, 3).reshape(expert_count, c.moe_intermediate, mh_r)
                return w["gu_dense"][o_dense], down

            def packed_experts():
                local_experts = 4 * jnp.arange(expert_count) + rank // 8
                gp = w["gu_pack"][o_packed]  # [E, K/2 + K/16, N] merged blob
                ks = c.moe_hidden // 2
                w13 = _dequant_experts_tiled(
                    gp[:, :ks], gp[:, ks:], w["gu_global"][o_packed, local_experts],
                    expert_count, c.moe_hidden, 2 * (c.moe_intermediate // 8),
                )
                dp = _prefill_exchange_down(w["dw_pack"][o_packed])
                w2 = _prefill_expand_down(dp, w["dw_global"][o_packed, local_experts])
                return w13, w2

            w13, w2 = jax.lax.cond(is_dense, dense_experts, packed_experts)
            local_ids = jnp.where(ids % 4 == rank // 8, ids // 4, expert_count)
            order = jnp.argsort(local_ids.reshape(-1), stable=True)
            group_sizes = jnp.bincount(local_ids.reshape(-1), length=expert_count + 1)[:expert_count].astype(jnp.int32)
            maximum_rows = jax.lax.pmax(jnp.sum(group_sizes), "tp")
            capacity = min(order.size, ((order.size * 3 // 8 + 127) // 128) * 128)
            capacity_small = min(capacity, ((order.size * 5 // 16 + 127) // 128) * 128)
            def gather_inputs(capacity):
                values = pooled[order[:capacity] // c.top_k]
                return jnp.pad(values, ((0, order.size - capacity), (0, 0)))
            sorted_inputs = jax.lax.cond(maximum_rows <= capacity_small, lambda: gather_inputs(capacity_small), lambda: jax.lax.cond(maximum_rows <= capacity, lambda: gather_inputs(capacity), lambda: gather_inputs(order.size))) if capacity_small < order.size else gather_inputs(order.size)
            gu = jax.lax.ragged_dot(sorted_inputs, w13, group_sizes, preferred_element_type=F32).astype(jnp.bfloat16)
            gate_e, up_e = jnp.split(gu, 2, axis=-1)
            act = (jax.nn.silu(gate_e.astype(F32)) * up_e.astype(F32)).astype(jnp.bfloat16)
            def exchange(capacity):
                joined = jax.lax.all_gather(act[:capacity], "tp", axis=-1, tiled=True, axis_index_groups=hosts)
                return jnp.pad(joined, ((0, order.size - capacity), (0, 0)))
            act = jax.lax.cond(maximum_rows <= capacity_small, lambda: exchange(capacity_small), lambda: jax.lax.cond(maximum_rows <= capacity, lambda: exchange(capacity), lambda: exchange(order.size))) if capacity_small < order.size else exchange(order.size)
            sorted_rows = jax.lax.ragged_dot(act, w2, group_sizes, preferred_element_type=F32).astype(jnp.bfloat16)
            inverse = jnp.zeros_like(order).at[order].set(jnp.arange(order.size, dtype=jnp.int32), unique_indices=True)
            pair_rows = sorted_rows[inverse].reshape(L, c.top_k, mh_r)
            gain = jax.lax.dynamic_slice(w["expert_norms"][li, 1], ((rank % 8) * mh_r,), (mh_r,))
            rows = pair_rows.astype(F32) * gain.astype(F32)
            squares = jax.lax.psum(jnp.sum(rows * rows, axis=-1, keepdims=True), "tp", axis_index_groups=hosts)
            rows = rows * jax.lax.rsqrt(squares / c.moe_hidden + c.post_eps)
            rows = jnp.where((ids % 4 == rank // 8)[..., None], rows, 0)
            mixed = jnp.sum(rows * rw.astype(F32)[..., None], axis=-2).astype(jnp.bfloat16)
            if length > 2048:
                mixed = jax.lax.psum(mixed.astype(F32), "tp", axis_index_groups=owners)
                mixed = jax.lax.all_gather(mixed, "tp", axis=-1, tiled=True, axis_index_groups=hosts)
                branch = jnp.dot(mixed, w["unpool"][li].astype(F32), precision=jax.lax.Precision.HIGHEST).astype(jnp.bfloat16)
            else:
                mixed = jax.lax.all_gather(mixed, "tp", axis=-1, tiled=True, axis_index_groups=hosts)
                upartial = jnp.dot(mixed, w["unpool"][li], preferred_element_type=F32)
                branch = jax.lax.psum(upartial, "tp", axis_index_groups=owners).astype(jnp.bfloat16)
            branch = jax.lax.all_gather(branch, "tp", axis=-1, tiled=True, axis_index_groups=hosts)
            stream = sc[5] * stream + sc[6] * rms_gain_first_f32(branch, sc[2], c.post_eps)
            stream = jnp.where(real[:, None], stream, jnp.zeros((), F32))
            if aux_layers:
                captured = aux[0]
                for index, layer_id in enumerate(aux_layers):
                    captured = jax.lax.cond(
                        li == layer_id,
                        lambda a: a.at[:, index].set(stream.astype(jnp.bfloat16)),
                        lambda a: a,
                        captured,
                    )
                return (stream, cache_all, captured), None
            return (stream, cache_all), None

        carry = (stream, states["kvcache"])
        if aux_layers:
            carry += (jnp.zeros((L, len(aux_layers), c.dim), jnp.bfloat16),)
        (stream, cache_all, *aux), _ = jax.lax.scan(one_layer, carry, jnp.arange(c.layers))
        last = stream.reshape(batch, length, c.dim)[:, -1]
        hidden = rms_gain_shard(last, w["fnorm"][0], c.eps)
        logits = jnp.dot(hidden, w["lmw"], preferred_element_type=F32)  # [1, lm_width]
        logits = logits * c.output_multiplier
        logits = c.soft_cap * jnp.tanh(logits / c.soft_cap)
        new_states = {"kvcache": cache_all}
        result = (jax.tree.map(lambda a: a[None], new_states), logits)
        return result + (aux[0].reshape(batch, length, len(aux_layers), c.dim)[0] if batch == 1 else aux[0].reshape(batch, length, len(aux_layers), c.dim),) if aux_layers else result

    def prefill(w, states, tokens, npad):
        ws = P("tp")
        ss = jax.tree.map(lambda _: P("tp"), states)
        result = jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(ws, ss, P(), P()),
            out_specs=(ss, P("tp"), P()) if aux_layers else (ss, P("tp")),
            check_vma=False,
        )(w, states, tokens, npad)
        return result

    return prefill


def _replicated_tp8_backend():
    """Build the default BS=1 backend with replicated TP8 experts."""
    CACHE_KEYS = ("kcache", "vcache")
    REPLICATED_EXPERTS = True
    EXPERT_TP = 8
    _EP_DEFAULT = EXPERT_TP == 8
    EXPERT_SLOTS = int(os.environ.get("MPK_EXPERT_SLOTS", 1 if _EP_DEFAULT else 4))
    ROWS_XCHG = os.environ.get("MPK_ROWS_XCHG", "transpose")
    XCHG_PRIO = int(os.environ.get("MPK_XCHG_PRIO", "0"))
    EXPERT_PRIO = int(os.environ.get("MPK_EXPERT_PRIO", "0"))
    X3_PIPE = int(os.environ.get("MPK_X3_PIPE", "1")) and ROWS_XCHG == "transpose"
    MXU_SUM = int(os.environ.get("MPK_MXU_SUM", "0"))
    UNPOOL_COLUMNS = int(os.environ.get("MPK_UNPOOL_COLUMNS", _EP_DEFAULT))
    ATTN_GROUP = int(os.environ.get("MPK_ATTN_GROUP", 2 if _EP_DEFAULT else 1))
    LOCAL_UNPOOL = int(os.environ.get("MPK_LOCAL_UNPOOL", _EP_DEFAULT))
    EXPERT_ARENA = 1
    DEFER_DENSE_PREFETCH = 1
    DENSE_PREFETCH_STAGE = 3
    KV_UPDATE_COMPACT = 1
    ROUTER_TRANSPOSE = 1
    LOCAL_POOL = int(os.environ.get("MPK_LOCAL_POOL", _EP_DEFAULT))
    EP_UNPOOL_FUSED = int(os.environ.get("MPK_EP_UNPOOL_FUSED", _EP_DEFAULT))
    ROUTER_PREFETCH = int(os.environ.get("MPK_ROUTER_PREFETCH", _EP_DEFAULT))
    LM_CHUNK = int(os.environ.get("MPK_LM_CHUNK", 640 if _EP_DEFAULT else 256))
    COMPACT_EP_ROWS = int(os.environ.get("MPK_COMPACT_EP_ROWS", _EP_DEFAULT))
    COMPACT_UNPOOL = int(os.environ.get("MPK_COMPACT_UNPOOL", _EP_DEFAULT))
    COMPACT_ATTN = int(os.environ.get("MPK_COMPACT_ATTN", _EP_DEFAULT))
    EP_PIPE_ROWS = int(os.environ.get("MPK_EP_PIPE_ROWS", _EP_DEFAULT))
    LM_ALIAS = int(os.environ.get("MPK_LM_ALIAS", _EP_DEFAULT))
    EP_SERIAL = int(os.environ.get("MPK_EP_SERIAL", _EP_DEFAULT))
    REPLICATE_EMBEDDING = int(os.environ.get("MPK_REPLICATE_EMBEDDING", _EP_DEFAULT))
    HIERARCHICAL_ARGMAX = int(os.environ.get("MPK_HIERARCHICAL_ARGMAX", _EP_DEFAULT))
    KV_WRITE_ROWS = int(os.environ.get("MPK_KV_WRITE_ROWS", 16 if _EP_DEFAULT else 0))
    LM_PREFETCH = int(os.environ.get("MPK_LM_PREFETCH", _EP_DEFAULT))
    DENSE_PIPE = int(os.environ.get("MPK_DENSE_PIPE", 2 if _EP_DEFAULT else 0))
    UNPOOL_DIRECT = int(os.environ.get("MPK_UNPOOL_DIRECT", 2 if _EP_DEFAULT else 0))
    ATTN_DIRECT = int(os.environ.get("MPK_ATTN_DIRECT", _EP_DEFAULT))
    DIRECT_AGGREGATE = int(os.environ.get("MPK_DIRECT_AGGREGATE", _EP_DEFAULT))
    DIRECT_ORDER = int(os.environ.get("MPK_DIRECT_ORDER", _EP_DEFAULT))
    COMPACT_MATH = int(os.environ.get("MPK_COMPACT_MATH", _EP_DEFAULT))
    KV_PREFETCH = int(os.environ.get("MPK_KV_PREFETCH", 2 if _EP_DEFAULT else 0))
    ATTENTION_TP = int(os.environ.get("MPK_ATTENTION_TP", 16 if _EP_DEFAULT else 32))
    ATTN_INPUT_COMPACT = int(os.environ.get("MPK_ATTN_INPUT_COMPACT", _EP_DEFAULT))


    def prepare_decode_weights(mesh, weights):
        """Build decode layouts from the TP32 bundle, preserving prefill weights."""
        weights = dict(weights)

        @jax.jit(out_shardings=NamedSharding(mesh, P("tp")))
        def unpool_columns(value):
            tp, layers, rows, hidden = value.shape
            full = value.transpose(1, 0, 2, 3).reshape(layers, tp * rows, hidden)
            groups = 8 if LOCAL_UNPOOL else tp
            result = full.reshape(layers, tp * rows, groups, hidden // groups).transpose(2, 0, 1, 3)
            return jnp.tile(result, (tp // groups, 1, 1, 1))

        if UNPOOL_COLUMNS:
            weights["unpool"] = unpool_columns(weights["unpool"])
        if LOCAL_POOL:
            @jax.jit(out_shardings=NamedSharding(mesh, P("tp")))
            def local_columns(value):
                tp, layers, rows, cols = value.shape
                value = value.reshape(8, tp // 8, layers, rows, cols)
                value = value.transpose(0, 2, 3, 1, 4).reshape(8, layers, rows, cols * (tp // 8))
                return jnp.tile(value, (tp // 8, 1, 1, 1))

            weights["pool"] = local_columns(weights["pool"])
            weights["router"] = local_columns(weights["router"])
        if ATTENTION_TP != 32:
            @jax.jit(out_shardings=NamedSharding(mesh, P("tp")))
            def attention_columns(value):
                replicas = 32 // ATTENTION_TP
                _, layers, hidden, width = value.shape
                source = value.reshape(ATTENTION_TP, replicas, layers, hidden, width)
                qwidth = (width - 128) // 2
                pieces = [source[:, j, :, :, start:start + qwidth] for start in (0, qwidth) for j in range(replicas)]
                pieces += [source[:, j, :, :, start:start + 64] for start in (2 * qwidth, 2 * qwidth + 64) for j in range(0, replicas, 2)]
                return jnp.tile(jnp.concatenate(pieces, axis=-1), (replicas, 1, 1, 1))

            @jax.jit(out_shardings=NamedSharding(mesh, P("tp")))
            def attention_rows(value):
                replicas = 32 // ATTENTION_TP
                _, layers, rows, hidden = value.shape
                source = value.reshape(ATTENTION_TP, replicas, layers, rows, hidden)
                source = source.transpose(0, 2, 1, 3, 4).reshape(ATTENTION_TP, layers, replicas * rows, hidden)
                return jnp.tile(source, (replicas, 1, 1, 1))

            weights["qwkv"] = attention_columns(weights["qwkv"])
            weights["ow"] = attention_rows(weights["ow"])
        if ATTN_GROUP != 1:
            @jax.jit(out_shardings=NamedSharding(mesh, P("tp")))
            def attention_output(value):
                tp, layers, rows, hidden = value.shape
                value = value.reshape(tp // ATTN_GROUP, ATTN_GROUP, layers, rows, ATTN_GROUP, hidden // ATTN_GROUP)
                return value.transpose(0, 4, 2, 1, 3, 5).reshape(tp, layers, rows * ATTN_GROUP, hidden // ATTN_GROUP)

            weights["ow"] = attention_output(weights["ow"])
        if EXPERT_TP == 8:
            def regroup(value, contraction):
                _, layers, experts, rows, cols = value.shape
                value = value.reshape(8, 4, layers, experts, rows, cols)
                if contraction:
                    value = value.transpose(0, 2, 3, 1, 4, 5).reshape(8, layers, experts, 4 * rows, cols)
                else:
                    value = value.transpose(0, 2, 3, 4, 1, 5).reshape(8, layers, experts, rows, 4 * cols)
                return jnp.tile(value, (4, 1, 1, 1, 1))

            @jax.jit(out_shardings=NamedSharding(mesh, P("tp")))
            def gate_up(value):
                gate, up = jnp.split(value, 2, axis=-1)
                return jnp.concatenate((regroup(gate, False), regroup(up, False)), axis=-1)

            @jax.jit(out_shardings=NamedSharding(mesh, P("tp")))
            def down(value):
                packed_rows = value.shape[-2] * 8 // 9
                return jnp.concatenate(
                    (regroup(value[..., :packed_rows, :], True), regroup(value[..., packed_rows:, :], True)), axis=-2
                )

            def convert_layers(value, transform):
                chunk_shape = (value.shape[0], 1, *value.shape[2:])
                transformed = jax.eval_shape(transform, jax.ShapeDtypeStruct(chunk_shape, value.dtype))
                output_shape = (transformed.shape[0], value.shape[1], *transformed.shape[2:])
                sharding = NamedSharding(mesh, P("tp"))
                result = jax.jit(lambda: jnp.zeros(output_shape, value.dtype), out_shardings=sharding)()

                @jax.jit(donate_argnums=(0,), out_shardings=sharding)
                def convert_layer(destination, source, index):
                    start = (0, index, 0, 0, 0)
                    chunk = jax.lax.dynamic_slice(source, start, chunk_shape)
                    return jax.lax.dynamic_update_slice(destination, transform(chunk), start)

                for index in range(value.shape[1]):
                    result = convert_layer(result, value, jnp.int32(index)).block_until_ready()
                return result

            jax.block_until_ready(weights)
            weights["gu_pack"] = convert_layers(weights["gu_pack"], gate_up)
            weights["dw_pack"] = convert_layers(weights["dw_pack"], down)
        if REPLICATE_EMBEDDING:
            @jax.jit(out_shardings=NamedSharding(mesh, P("tp")))
            def full_embedding(value):
                tp, rows, hidden = value.shape
                return jnp.tile(value.reshape(1, tp * rows, hidden), (tp, 1, 1))

            weights["emb"] = full_embedding(weights["emb"]).block_until_ready()
        if ROUTER_TRANSPOSE:
            @jax.jit(out_shardings=NamedSharding(mesh, P("tp")))
            def transpose_router(value):
                return value.swapaxes(-2, -1)

            weights["router"] = transpose_router(weights["router"])
        jax.block_until_ready(weights)
        return weights


    def _dequant_experts(packed, scales, globals_, count, k, n):
        """K-major NVFP4 [E, K/2, N] + E4M3 [E, K/16, N] + FP32 globals -> BF16 [E, K, N]."""
        lut = jnp.array(NVFP4_LUT, F32)
        lo = lut[jnp.asarray(packed, jnp.uint8) & 15]
        hi = lut[jnp.asarray(packed, jnp.uint8) >> 4]
        values = jnp.stack((lo, hi), axis=2).reshape(count, k, n)  # [E, K/2, 2, N] -> [E, K, N]
        sf = jax.lax.bitcast_convert_type(scales, jnp.float8_e4m3fn).astype(F32)
        g = globals_.reshape(count, -1)  # [E] or [E, 2]
        if g.shape[-1] == 2:
            half = n // 2
            g = jnp.concatenate(
                (
                    jnp.broadcast_to(g[:, 0:1, None], (count, k // 16, half)),
                    jnp.broadcast_to(g[:, 1:2, None], (count, k // 16, n - half)),
                ),
                axis=2,
            )
        else:
            g = jnp.broadcast_to(g[:, None, :], (count, k // 16, n))
        sf = sf * g  # [E, K/16, N] fp32
        w = values.reshape(count, k // 16, 16, n) * sf[:, :, None, :]
        return w.reshape(count, k, n).astype(jnp.bfloat16)


    def _tp_specs(w, tp):
        def spec(a):
            return P("tp") if a.ndim and a.shape[0] == tp else P()

        return jax.tree.map(spec, w)


    def make_prefill(mesh, c, context, tp, *, aux_layers=()):
        """TP-sharded prefill over packed (untiled) weights, plain XLA.

        Prompt rows are LEFT-padded to the bucket length; each row's KV is written
        to absolute slot (row - npad) with out-of-range writes dropped, so caches
        come back in the compacted [0, L_real) layout the decode kernel expects.
        Returns per-rank logits shards for the last real position.
        """
        qh = c.heads // tp
        kvh = max(c.kv_heads // tp, 1)  # rank pairs share one KV head at tp=32
        mh_r = c.moe_hidden // tp
        vshard = c.vocab // tp

        def local(w, states, tokens, npad):
            w = jax.tree.map(lambda a: a[0], w)
            states = jax.tree.map(lambda a: a[0], states)
            rank = jax.lax.axis_index("tp")
            L = tokens.shape[0]
            real = jnp.arange(L) >= npad
            positions = jnp.arange(L, dtype=jnp.int32) - npad
            lo = rank * vshard
            in_range = (tokens >= lo) & (tokens < lo + vshard)
            rows = w["emb"][jnp.clip(tokens - lo, 0, vshard - 1)]
            stream = jax.lax.psum(jnp.where(in_range[:, None], rows.astype(F32), 0.0), "tp")
            value = stream
            value = value * jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + c.eps)
            stream = jnp.where(real[:, None], value, jnp.zeros((), F32))
            slots = jnp.clip(positions, -context, context - 1)
            cos, sin = rope_tables_shard(c, positions)

            def one_layer(carry, li):
                stream, kc_all, vc_all, *aux = carry
                li = li.astype(jnp.int32)
                sliding = (li % 4) != 1
                is_dense = jnp.bool_(False)
                for d in c.bf16_expert_layers:
                    is_dense = is_dense | (li == d)
                dense_below = jnp.zeros((), jnp.int32)
                for d in c.bf16_expert_layers:
                    dense_below = dense_below + (li > d).astype(jnp.int32)
                o_packed = jnp.clip(li - dense_below, 0, max(c.layers - len(c.bf16_expert_layers) - 1, 0))
                o_dense = jnp.clip(dense_below, 0, len(c.bf16_expert_layers) - 1)
                sc = w["scalars"][li]

                x = rms_gain_shard(stream, sc[0], c.eps)
                qwkv = jnp.dot(x, w["qwkv"][li], preferred_element_type=F32).astype(jnp.bfloat16)
                query = qwkv[:, : qh * c.head_dim].reshape(L, qh, c.head_dim)
                gate = qwkv[:, qh * c.head_dim : 2 * qh * c.head_dim].reshape(L, qh, c.head_dim)
                kv_off = 2 * qh * c.head_dim
                key = qwkv[:, kv_off : kv_off + kvh * c.head_dim].reshape(L, kvh, c.head_dim)
                value = qwkv[:, kv_off + kvh * c.head_dim :].reshape(L, kvh, c.head_dim)
                query = rms_plain_shard(query, c.eps)
                key = rms_plain_shard(key, c.eps)
                query = jnp.where(sliding, apply_rope_shard(query, cos, sin).astype(F32), query.astype(F32)).astype(jnp.bfloat16)
                key = jnp.where(sliding, apply_rope_shard(key, cos, sin).astype(F32), key.astype(F32)).astype(jnp.bfloat16)

                kc = kc_all[li]
                vc = vc_all[li]
                key_p = jnp.pad(key, ((0, 0), (0, 0), (0, kc.shape[-1] - c.head_dim)))
                value_p = jnp.pad(value, ((0, 0), (0, 0), (0, kc.shape[-1] - c.head_dim)))
                kc = kc.at[:, slots].set(key_p.transpose(1, 0, 2).astype(kc.dtype), mode="drop")
                vc = vc.at[:, slots].set(value_p.transpose(1, 0, 2).astype(vc.dtype), mode="drop")
                kc_all = kc_all.at[li].set(kc)
                vc_all = vc_all.at[li].set(vc)

                grouped = query.transpose(1, 0, 2).reshape(kvh, qh // kvh, L, c.head_dim)
                keys_t = key.transpose(1, 0, 2)  # padded-row order; mask by absolute position
                values_t = value.transpose(1, 0, 2)
                scores = jnp.einsum("gqld,gtd->gqlt", grouped.astype(F32), keys_t.astype(F32))
                scores = scores * c.scale
                key_abs = jnp.arange(L) - npad
                lo_bound = jnp.where(sliding, positions - c.sliding_window, jnp.full_like(positions, -2147483648))
                keep = (key_abs[None, :] <= positions[:, None]) & (key_abs[None, :] > lo_bound[:, None]) & (key_abs[None, :] >= 0)
                scores = jnp.where(keep[None, None], scores, -jnp.inf)
                scores = jnp.where(real[None, None, :, None], scores, 0.0)
                probs = jax.nn.softmax(scores, axis=-1).astype(jnp.bfloat16)
                out = jnp.einsum("gqlt,gtd->gqld", probs.astype(F32), values_t.astype(F32))
                out = out.reshape(kvh, qh // kvh, L, c.head_dim).transpose(2, 0, 1, 3)
                out = out.reshape(L, qh, c.head_dim)
                out = rms_plain_shard(out, c.eps) * jax.nn.sigmoid(gate.astype(F32))
                out = out.astype(jnp.bfloat16).reshape(L, qh * c.head_dim)
                partial = jnp.dot(out, w["ow"][li], preferred_element_type=F32)
                branch = jax.lax.psum(partial, "tp").astype(jnp.bfloat16)
                stream = sc[3] * stream + sc[4] * rms_plain_f32(branch, c.post_eps)
                stream = jnp.where(real[:, None], stream, jnp.zeros((), F32))

                x = rms_gain_shard(stream, sc[1], c.eps)
                rlogits = jnp.dot(x.astype(F32), w["router"][li], precision=jax.lax.Precision.HIGHEST)
                rfull = jax.lax.all_gather(rlogits, "tp", axis=-1, tiled=True)
                scores_r = jax.nn.sigmoid(rfull * c.output_multiplier)
                ranked = scores_r + w["router_bias"][li].astype(F32)
                ids = jax.lax.top_k(ranked, c.top_k)[1]
                rw = jnp.take_along_axis(scores_r, ids, axis=-1)
                rw = rw / (jnp.sum(rw, axis=-1, keepdims=True) + c.router_eps)

                pooled_part = jnp.dot(x, w["pool"][li], preferred_element_type=F32).astype(jnp.bfloat16)
                pooled = jax.lax.all_gather(pooled_part, "tp", axis=-1, tiled=True)
                pooled = rms_gain_shard(pooled, w["expert_norms"][li, 0], c.eps)

                def dense_experts():
                    return w["gu_dense"][o_dense], w["dw_dense"][o_dense]

                def packed_experts():
                    gp = w["gu_pack"][o_packed]  # [E, K/2 + K/16, N] merged blob
                    ks = c.moe_hidden // 2
                    w13 = _dequant_experts(
                        gp[:, :ks], gp[:, ks:], w["gu_global"][o_packed],
                        c.experts, c.moe_hidden, 2 * (c.moe_intermediate // tp),
                    )
                    dp = w["dw_pack"][o_packed]  # [E, I/2 + I/16, K] merged blob
                    ki = (c.moe_intermediate // tp) // 2
                    w2 = _dequant_experts(
                        dp[:, :ki], dp[:, ki:], w["dw_global"][o_packed],
                        c.experts, c.moe_intermediate // tp, c.moe_hidden,
                    )
                    return w13, w2

                w13, w2 = jax.lax.cond(is_dense, dense_experts, packed_experts)
                gu = jnp.einsum("lk,ekn->eln", pooled, w13, preferred_element_type=F32).astype(jnp.bfloat16)
                gate_e, up_e = jnp.split(gu, 2, axis=-1)
                act = (jax.nn.silu(gate_e.astype(F32)) * up_e.astype(F32)).astype(jnp.bfloat16)
                rows_all = jnp.einsum("eli,eik->elk", act, w2, preferred_element_type=F32).astype(jnp.bfloat16)
                pair_rows = rows_all[ids.reshape(-1), jnp.arange(L).repeat(c.top_k)].reshape(L, c.top_k, c.moe_hidden)
                rows = jax.lax.psum(pair_rows, "tp")
                rows = rms_gain_first_shard(rows, w["expert_norms"][li, 1], c.post_eps)
                mixed = jnp.sum(rows * rw.astype(F32)[..., None], axis=-2).astype(jnp.bfloat16)
                upartial = jnp.dot(
                    jax.lax.dynamic_slice(mixed, (0, rank * mh_r), (L, mh_r)),
                    w["unpool"][li],
                    preferred_element_type=F32,
                )
                branch = jax.lax.psum(upartial, "tp").astype(jnp.bfloat16)
                stream = sc[5] * stream + sc[6] * rms_gain_first_f32(branch, sc[2], c.post_eps)
                stream = jnp.where(real[:, None], stream, jnp.zeros((), F32))
                if aux_layers:
                    captured = aux[0]
                    for index, layer_id in enumerate(aux_layers):
                        captured = jax.lax.cond(
                            li == layer_id,
                            lambda a: a.at[:, index].set(stream.astype(jnp.bfloat16)),
                            lambda a: a,
                            captured,
                        )
                    return (stream, kc_all, vc_all, captured), None
                return (stream, kc_all, vc_all), None

            carry = (stream, states["kcache"], states["vcache"])
            if aux_layers:
                carry += (jnp.zeros((L, len(aux_layers), c.dim), jnp.bfloat16),)
            (stream, kc_all, vc_all, *aux), _ = jax.lax.scan(one_layer, carry, jnp.arange(c.layers))
            last = stream[L - 1]
            hidden = rms_gain_shard(last[None], w["fnorm"][0], c.eps)
            logits = jnp.dot(hidden, w["lmw"], preferred_element_type=F32)  # [1, lm_width]
            logits = logits * c.output_multiplier
            logits = c.soft_cap * jnp.tanh(logits / c.soft_cap)
            new_states = {"kcache": kc_all, "vcache": vc_all}
            result = (jax.tree.map(lambda a: a[None], new_states), logits)
            return result + (aux[0],) if aux_layers else result

        def prefill(w, states, tokens, npad):
            ws = _tp_specs(w, tp)
            ss = jax.tree.map(lambda _: P("tp"), states)
            result = jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(ws, ss, P(), P()),
                out_specs=(ss, P("tp"), P()) if aux_layers else (ss, P("tp")),
                check_vma=False,
            )(w, states, tokens, npad)
            if ATTENTION_TP != 32:
                new_states, logits, *aux = result
                heads = max(c.kv_heads // ATTENTION_TP, 1)
                indices = (jnp.arange(tp) % ATTENTION_TP)[:, None] * (tp // ATTENTION_TP) + 2 * jnp.arange(heads)[None, :]
                def replicate_cache(value):
                    value = value[indices, :, 0, :, :]
                    return value.transpose(0, 2, 1, 3, 4)
                return (jax.tree.map(replicate_cache, new_states), logits, *aux)
            return result

        return prefill


    def rms_gain_shard(x, gain, eps):
        y = x.astype(F32)
        y = y * jax.lax.rsqrt(jnp.mean(y * y, axis=-1, keepdims=True) + eps)
        return (y * gain.astype(F32)).astype(jnp.bfloat16)


    def rms_plain_shard(x, eps):
        y = x.astype(F32)
        y = y * jax.lax.rsqrt(jnp.mean(y * y, axis=-1, keepdims=True) + eps)
        return y.astype(jnp.bfloat16)


    def rms_plain_f32(x, eps):
        y = x.astype(F32)
        return y * jax.lax.rsqrt(jnp.mean(y * y, axis=-1, keepdims=True) + eps)


    def rms_gain_first_shard(x, gain, eps):
        y = x.astype(F32) * gain.astype(F32)
        y = y * jax.lax.rsqrt(jnp.mean(y * y, axis=-1, keepdims=True) + eps)
        return y  # FP32 out (post-norm contract)


    def rms_gain_first_f32(x, gain, eps):
        return rms_gain_first_shard(x, gain, eps)


    def rope_tables_shard(c, positions):
        inv = 1.0 / (c.rope_theta ** (jnp.arange(0, c.head_dim, 2, dtype=F32) / c.head_dim))
        angles = positions.astype(F32)[:, None] * inv
        return jnp.cos(angles), jnp.sin(angles)


    def apply_rope_shard(x, cos, sin):
        """x [L, heads, head_dim]; cos/sin [L, head_dim//2]; split-half rotation."""
        real, imag = jnp.split(x.astype(F32), 2, axis=-1)
        cv = cos[:, None, :]
        sv = sin[:, None, :]
        out = jnp.concatenate((real * cv - imag * sv, real * sv + imag * cv), axis=-1)
        return out.astype(jnp.bfloat16)


    # ---------------------------------------------------------------------------
    # Fused decode megakernel (Pallas)
    # ---------------------------------------------------------------------------


    def transformer_stack(c, token, w, states, position, context, ablate=frozenset(), tp=32, *, chunk_size=0, teacher_forcing=False, aux_layers=()):
        """Run one decode step, or a sequential chunk when chunk_size is positive.

        Dense slabs and selected experts stream from HBM to VMEM. TP8 experts
        reduce their rows within each host; the projected mixtures are summed
        across hosts before the residual update.
        """
        if tp != 32:
            raise ValueError("the TP32 kernel requires exactly 32 devices")
        if context % 128 or context < 256:
            raise ValueError("context must be a multiple of 128 and at least 256")
        control = position if chunk_size else jnp.concatenate((position, jnp.ones((1,), jnp.int32)))
        if teacher_forcing:
            control = jnp.concatenate((control, token))
            token = token[:1]
        history_size = max(chunk_size, 1)
        h = c.dim
        if ATTENTION_TP not in (16, 32):
            raise ValueError("attention tensor parallelism must be 16 or 32")
        nh = c.heads // ATTENTION_TP
        compact_unpool = COMPACT_UNPOOL and h >= 1024
        nkv = max(c.kv_heads // ATTENTION_TP, 1)
        hd = c.head_dim
        hdp = 128  # padded cache head dim
        mh = c.moe_hidden
        mh_r = mh // tp
        if EXPERT_TP not in (8, 32):
            raise ValueError("expert TP must be 8 or 32")
        if ROWS_XCHG not in ("transpose", "allreduce") or (EXPERT_TP == 8 and ROWS_XCHG != "transpose"):
            raise ValueError("TP8 experts require row-transpose exchange; allreduce is available with MPK_EXPERT_TP=32")
        if DENSE_PIPE not in (0, 1, 2):
            raise ValueError("dense pipeline mode must be 0, 1, or 2")
        inter_r = c.moe_intermediate // EXPERT_TP
        dense_inter_r = c.moe_intermediate // tp
        eshard = c.experts // (8 if LOCAL_POOL else tp)
        pool_width = mh // (8 if LOCAL_POOL else tp)
        ep_row_overlap = EXPERT_TP == 8 and EP_UNPOOL_FUSED and COMPACT_EP_ROWS and EP_PIPE_ROWS
        lm_w = lm_width(c, tp)
        vshard = c.vocab // tp
        # The TP8 dense-overlay (reusing the packed slot buffers for the bf16 expert
        # layers) needs an exact byte-count match that the merged expert blobs break;
        # at TP32 the dedicated buffers are cheap, so take them instead (and the
        # dedicated dense path uses fewer, whole-expert copies).
        dense_overlay = False
        sbw = min(int(os.environ.get("MPK_SBW", 1024 if EXPERT_TP == 8 else 512)), context)
        kv_write_rows = min(KV_WRITE_ROWS or sbw, sbw)
        if context % sbw or kv_write_rows % 16 or sbw % kv_write_rows:
            raise ValueError("attention blocks must divide the context and KV writes must use aligned 16-row tiles")
        groups = nh // nkv
        if nh % nkv or groups > 8 or groups % 2:
            raise ValueError("GQA groups must be even pairs of heads fitting one 8-row MXU tile")
        if c.experts % tp or c.moe_hidden % tp or c.moe_intermediate % tp:
            raise ValueError("MoE dims must split evenly across ranks")
        if c.top_k != 8:
            raise ValueError("the row-transpose exchange maps one expert row per lane")
        len_packed = c.layers - len(c.bf16_expert_layers)

        # Dense weights stream as whole per-rank slabs (one big copy per family per
        # layer, prefetched in-layer) instead of the TP8 bank ring: at TP32 the
        # families are narrow, so per-tile loop overhead dominates a tiled ring.
        qwkv_w = (h, 2 * nh * hd + 2 * nkv * hd)
        if ATTN_GROUP not in (1, 2, 4, 8):
            raise ValueError("attention group must be 1, 2, 4, or 8")
        ow_w = (nh * hd * (ATTN_GROUP), h // ATTN_GROUP)
        pool_w = (h, pool_width)
        if LOCAL_UNPOOL and not UNPOOL_COLUMNS:
            raise ValueError("local unpool requires output column sharding")
        if EP_UNPOOL_FUSED and (EXPERT_TP != 8 or not LOCAL_UNPOOL):
            raise ValueError("fused expert unpool requires TP8 experts and local unpool")
        if EP_PIPE_ROWS and (EXPERT_TP != 8 or not COMPACT_EP_ROWS):
            raise ValueError("expert row pipelining requires compact TP8 rows")
        if EP_SERIAL and EXPERT_TP != 8:
            raise ValueError("serial experts require TP8 expert shards")
        unpool_output_width = h // (8 if LOCAL_UNPOOL else tp)
        unpool_w = (mh, unpool_output_width) if UNPOOL_COLUMNS else (mh_r, h)
        unpool_input = mh if UNPOOL_COLUMNS else mh_r
        lm_chunk_w = min(LM_CHUNK, lm_w // 2)
        if lm_chunk_w <= 0 or lm_chunk_w % 128 or lm_w % lm_chunk_w:
            raise ValueError("lm_width must be a multiple of the lm chunk width")
        alias_lm = LM_ALIAS
        qwkv_tiles = pool_alias.storage_size(qwkv_w, w["qwkv"].dtype) // 4096
        lm_tiles = pool_alias.storage_size((2, h, lm_chunk_w), w["lmw"].dtype) // 4096
        gu_shape = (EXPERT_SLOTS, mh // 2 + mh // 16, 2 * inter_r)
        dw_shape = (EXPERT_SLOTS, inter_r // 2 + inter_r // 16, mh)
        gu_dense_shape = (2, mh, 2 * dense_inter_r) if DENSE_PIPE else (mh, 2 * dense_inter_r)
        dw_dense_shape = (2, dense_inter_r, mh) if DENSE_PIPE else (dense_inter_r, mh)
        gu_tiles = pool_alias.storage_size(gu_shape, jnp.dtype(jnp.uint8)) // 4096
        dw_tiles = pool_alias.storage_size(dw_shape, jnp.dtype(jnp.uint8)) // 4096
        gu_dense_tiles = pool_alias.storage_size(gu_dense_shape, jnp.dtype(jnp.bfloat16)) // 4096
        dw_dense_tiles = pool_alias.storage_size(dw_dense_shape, jnp.dtype(jnp.bfloat16)) // 4096
        projection_tiles = tuple(pool_alias.storage_size(shape, jnp.dtype(jnp.bfloat16)) // 4096 for shape in (ow_w, pool_w, unpool_w))
        projection_arena_tiles = max(projection_tiles)
        gu_arena_tiles = max(2 * gu_tiles, gu_dense_tiles)
        dw_arena_tiles = max(2 * dw_tiles, dw_dense_tiles)
        shared_arenas = bool(
            chunk_size and EXPERT_ARENA and LM_ALIAS and DEFER_DENSE_PREFETCH & 1
            and DENSE_PREFETCH_STAGE == 3 and gu_arena_tiles + dw_arena_tiles <= qwkv_tiles
        )
        direct_semaphore_width = 1 if DIRECT_AGGREGATE and COMPACT_ATTN and min(unpool_output_width, h // ATTN_GROUP) >= 256 else tp - 1
        def body(
            pr,
            tokenr,
            qwkv_r,
            ow_r,
            pool_r,
            unpool_r,
            lmw_r,
            router_r,
            scalars_r,
            expert_norms_r,
            router_bias_r,
            gu_pack_r,
            gu_global_r,
            dw_pack_r,
            dw_global_r,
            gu_dense_r,
            dw_dense_r,
            fnorm_r,
            emb_r,
            kc_r,
            vc_r,
            logits_out,
            token_out,
            tokens_out,
            kc_out,
            vc_out,
            *refs,
        ):
            if aux_layers:
                aux_out, *refs = refs
                aux_buffer, aux_sem = refs[-2:]
                refs = refs[:-2]
            (
                qwkv_slab,
                ow_slab,
                pool_slab,
                unpool_slab,
                routerbuf,
                lm_slab,
                dsem,
                scalarsbuf,
                sc_sem,
                biasbuf,
                xnormsbuf,
                residual,
                normed,
                proj,
                accum,
                lm_accum,
                qwkvbuf,
                attbuf,
                poolbuf,
                unpinbuf,
                pooledbuf,
                rowsbuf,
                combuf,
                recv,
                recv_h,
                recv_comb,
                recv_comb_h,
                recv_r1,
                recv_r2,
                rowstage,
                nrow,
                recv_r3,
                recv_rows,
                recv_rows_h,
                mixedbuf,
                xstage_h,
                xstage_rows,
                sends,
                recvs,
                qbg,
                kb,
                vb,
                amg,
                alg,
                aag,
                ks,
                vs,
                io,
                attended,
                gu_p2,
                guq_p2,
                dw_p2,
                dwq_p2,
                ggu,
                ggw,
                g_sem,
                eg_sem,
                gu_d,
                dw_d,
                ed_sem,
                rowbuf,
                e_sem,
                pairbuf,
                pairs,
                fnormbuf,
                ep_recv,
                attn_inputs,
                attn_outputs,
                local_unpool_recv,
                ep_rows_stage,
                ep_rows_recv,
                unpool_stage,
                unpool_hosts,
                attn_stage,
                attn_recv,
                attn_hosts,
                attn_output_stage,
                direct_unpool_recv,
                direct_unpool_stage,
                direct_attn_recv,
                all_attn_stage,
                direct_sends,
                direct_recvs,
                token_history,
                history_sem,
            ) = refs
            if shared_arenas:
                projection_storage = ow_slab
                ow_slab = pool_alias.view(projection_storage.at[:projection_tiles[0]], ow_w, w["ow"].dtype)
                pool_slab = pool_alias.view(projection_storage.at[:projection_tiles[1]], pool_w, w["pool"].dtype)
                unpool_slab = pool_alias.view(projection_storage.at[:projection_tiles[2]], unpool_w, w["unpool"].dtype)
            if alias_lm:
                storage = qwkv_slab
                qwkv_slab = pool_alias.view(storage.at[:qwkv_tiles], qwkv_w, w["qwkv"].dtype)
                if not shared_arenas:
                    lm_slab = pool_alias.view(storage.at[:lm_tiles], (2, h, lm_chunk_w), w["lmw"].dtype)
            if EXPERT_ARENA:
                if shared_arenas:
                    gu_storage = storage.at[:gu_arena_tiles]
                    dw_storage = storage.at[gu_arena_tiles:gu_arena_tiles + dw_arena_tiles]
                else:
                    gu_storage, dw_storage = gu_p2, dw_p2
                gu_p2 = pool_alias.view(gu_storage.at[:gu_tiles], gu_shape, jnp.uint8)
                guq_p2 = pool_alias.view(gu_storage.at[gu_tiles:2 * gu_tiles], gu_shape, jnp.uint8)
                dw_p2 = pool_alias.view(dw_storage.at[:dw_tiles], dw_shape, jnp.uint8)
                dwq_p2 = pool_alias.view(dw_storage.at[dw_tiles:2 * dw_tiles], dw_shape, jnp.uint8)
                gu_d = pool_alias.view(gu_storage.at[:gu_dense_tiles], gu_dense_shape, jnp.bfloat16)
                dw_d = pool_alias.view(dw_storage.at[:dw_dense_tiles], dw_dense_shape, jnp.bfloat16)
            rank = jax.lax.axis_index("tp")

            def one_token(token_index, _):
                pos = pr[0] + token_index
                window_start = pos - c.sliding_window + 1
                window_first = jnp.maximum(window_start, 0) // sbw
                cache_block = pos // sbw
                write_source = (pos % sbw) // kv_write_rows * kv_write_rows
                write_target = pos // kv_write_rows * kv_write_rows

                def compact_rms(value, eps, gain=None, gain_first=False):
                    shape = value.shape
                    value = value.astype(jnp.float32).reshape(-1, shape[-1] // 128, 128)
                    if gain_first:
                        value *= gain.reshape(1, shape[-1] // 128, 128)
                    value *= jax.lax.rsqrt(jnp.mean(value * value, axis=(-2, -1), keepdims=True) + eps)
                    if gain is not None and not gain_first:
                        value *= gain.reshape(1, shape[-1] // 128, 128)
                    return value.reshape(shape)

                def update_residual(branch, skip, scale, gain=None):
                    if COMPACT_MATH:
                        branch = compact_rms(branch, c.post_eps, gain, gain is not None)
                        value = skip.reshape(-1, 128) * residual[...].reshape(-1, 128)
                        value += scale.reshape(-1, 128) * branch.reshape(-1, 128)
                        residual[...] = value.reshape(1, h)
                    else:
                        if gain is not None:
                            branch *= gain
                        branch *= jax.lax.rsqrt(jnp.mean(branch * branch, axis=-1, keepdims=True) + c.post_eps)
                        residual[...] = skip * residual[...] + scale * branch

                def slab_dot(src, dst, buf, n, out_f32=False):
                    """[1, K] x resident [K, N] -> dst [1, n]: one MXU call, no tile loop."""
                    if "nogemv" not in ablate:
                        k = buf.shape[0]
                        a = jnp.broadcast_to(src[...][:, :k], (8, k))
                        r = jnp.dot(a, buf[...], preferred_element_type=jnp.float32)
                        dst[...] = r[:1, :n].astype(jnp.float32 if out_f32 else dst.dtype)

                def prefetch_slab(w_r, buf, sem_ix, li, force=False):
                    if shared_arenas and sem_ix != 0:
                        if sem_ix == 1:
                            tpu.make_async_copy(pool_r.at[li], pool_slab, dsem.at[2]).start()
                        elif sem_ix == 2:
                            tpu.make_async_copy(unpool_r.at[li], unpool_slab, dsem.at[3]).start()
                        elif sem_ix == 3:
                            @pl.when((li + 1 < c.layers) | ((token_index + 1 < pr[1]) & bool(chunk_size)))
                            def _():
                                tpu.make_async_copy(ow_r.at[(li + 1) % c.layers], ow_slab, dsem.at[1]).start()
                        return
                    if DEFER_DENSE_PREFETCH & (1 << sem_ix) and not force:
                        return
                    wrap = bool(chunk_size) and (sem_ix in (1, 2, 3) or (sem_ix == 0 and shared_arenas))
                    available = (li + 1 < c.layers) | ((token_index + 1 < pr[1]) & wrap)
                    @pl.when(available)
                    def _():
                        index = (li + 1) % c.layers if wrap else li + 1
                        tpu.make_async_copy(w_r.at[index], buf, dsem.at[sem_ix]).start()

                lane = rank % 8  # rank = host * 8 + lane
                host = rank // 8

                def phase_local(stage, recv_buf, slot, tag="", prewait=None):
                    """Intra-host all-to-all: recv_buf[slot, j] ends up holding lane j's stage."""
                    recv_buf[slot, lane, ...] = stage[...]
                    if "noxchg" in ablate or "nolocal" in ablate or (tag and tag in ablate):
                        if prewait is not None:
                            prewait()
                        return
                    for offset in range(7, 0, -1):
                        tpu.make_async_remote_copy(
                            stage,
                            recv_buf.at[slot, lane],
                            sends.at[offset - 1],
                            recvs.at[offset - 1],
                            device_id=(rank ^ offset,),
                            device_id_type=pl.DeviceIdType.MESH,
                        ).start(priority=XCHG_PRIO)
                    if prewait is not None:
                        prewait()
                    for offset in range(7, 0, -1):
                        tpu.make_async_remote_copy(
                            stage,
                            recv_buf.at[slot, lane],
                            sends.at[offset - 1],
                            recvs.at[offset - 1],
                            device_id=(rank ^ offset,),
                            device_id_type=pl.DeviceIdType.MESH,
                        ).wait()

                def phase_hosts(stage, recv_buf, slot, tag="", prewait=None):
                    """Inter-host all-to-all: recv_buf[slot, j] ends up holding host j's stage."""
                    recv_buf[slot, host, ...] = stage[...]
                    if "noxchg" in ablate or "nohost" in ablate or (tag and tag in ablate):
                        if prewait is not None:
                            prewait()
                        return
                    for offset in (8, 16, 24):
                        tpu.make_async_remote_copy(
                            stage,
                            recv_buf.at[slot, host],
                            sends.at[6 + offset // 8],
                            recvs.at[6 + offset // 8],
                            device_id=(rank ^ offset,),
                            device_id_type=pl.DeviceIdType.MESH,
                        ).start(priority=XCHG_PRIO)
                    if prewait is not None:
                        prewait()
                    for offset in (8, 16, 24):
                        tpu.make_async_remote_copy(
                            stage,
                            recv_buf.at[slot, host],
                            sends.at[6 + offset // 8],
                            recvs.at[6 + offset // 8],
                            device_id=(rank ^ offset,),
                            device_id_type=pl.DeviceIdType.MESH,
                        ).wait()

                def direct_gather(source, destination, tag, prewait, group_size=tp, phase=0):
                    """Gather equal-sized slices using completion counters reserved for this phase."""
                    index = rank % group_size
                    destination[index] = source[...]
                    skip = "noxchg" in ablate or tag in ablate
                    aggregate = DIRECT_AGGREGATE and source.shape[-2] >= 2
                    offsets = tuple(range(1, group_size))
                    if DIRECT_ORDER == 1:
                        offsets = tuple(sorted(offsets, key=lambda offset: (offset % 8, offset // 8), reverse=True))
                    if not skip:
                        for offset in offsets:
                            semaphore = 0 if aggregate else offset - 1
                            tpu.make_async_remote_copy(
                                source, destination.at[index], direct_sends.at[phase, semaphore], direct_recvs.at[phase, semaphore],
                                device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH,
                            ).start()
                    prewait()
                    if not skip:
                        if aggregate:
                            transfer = tpu.make_async_remote_copy(
                                destination.at[:group_size - 1], destination.at[:group_size - 1], direct_sends.at[phase, 0], direct_recvs.at[phase, 0],
                                device_id=(rank,), device_id_type=pl.DeviceIdType.MESH,
                            )
                            transfer.wait()
                        else:
                            for offset in range(1, group_size):
                                tpu.make_async_remote_copy(
                                    source, destination.at[index], direct_sends.at[phase, offset - 1], direct_recvs.at[phase, offset - 1],
                                    device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH,
                                ).wait()

                def rsum(buf):
                    """[G, 1, W] bf16 -> [1, W] f32: one MXU dot instead of a vector reduce."""
                    if not MXU_SUM:
                        return jnp.sum(buf.astype(jnp.float32), axis=0)
                    g = buf.shape[0]
                    ones = jnp.ones((8, g), jnp.bfloat16)
                    return jnp.dot(ones, buf[:, 0, :], preferred_element_type=jnp.float32)[0:1]

                def allreduce(iteration, tag="", prestaged=False):
                    """All-reduce proj ([1, h] f32) — or xstage_h directly when prestaged;
                    returns the fp32 total [1, h]."""
                    slot = iteration % 2
                    if not prestaged:
                        xstage_h[...] = proj[...].astype(xstage_h.dtype)
                    phase_local(xstage_h, recv, slot, tag)
                    part = rsum(recv[slot])
                    xstage_h[...] = part.astype(xstage_h.dtype)
                    phase_hosts(xstage_h, recv_h, slot, tag)
                    return rsum(recv_h[slot])

                def gemv_allreduce(src, w_slab, iteration, tag="", prewait=None):
                    """[1,K] x [K,h] GEMV fused with the all-reduce of its output, in two
                    column chunks: chunk 0's intra-host copies issue while chunk 1's GEMV
                    computes, and one full-region wait per offset consumes both chunks'
                    completions (the collectives32 schedule). `prewait` runs between the
                    sends and the waits so bulk slab streams hide under the exchange flight.
                    Returns the fp32 total [1, h]."""
                    slot = iteration % 2
                    kw = w_slab.shape[0]
                    skip_x = "noxchg" in ablate or "nolocal" in ablate or (tag and tag in ablate)
                    for cchunk in range(2):
                        off = cchunk * (h // 2)
                        if "nogemv" not in ablate:
                            a = jnp.broadcast_to(src[...][:, :kw], (8, kw))
                            r = jnp.dot(a, w_slab[...][:, off : off + h // 2], preferred_element_type=jnp.float32)
                            xstage_h[0, pl.ds(off, h // 2)] = r[0].astype(xstage_h.dtype)
                        if skip_x:
                            continue
                        recv[slot, lane, 0, pl.ds(off, h // 2)] = xstage_h[0, pl.ds(off, h // 2)]
                        for offset in range(7, 0, -1):
                            tpu.make_async_remote_copy(
                                xstage_h.at[:, pl.ds(off, h // 2)],
                                recv.at[slot, lane, :, pl.ds(off, h // 2)],
                                sends.at[offset - 1],
                                recvs.at[offset - 1],
                                device_id=(rank ^ offset,),
                                device_id_type=pl.DeviceIdType.MESH,
                            ).start(priority=XCHG_PRIO)
                    if prewait is not None:
                        prewait()
                    if not skip_x:
                        for offset in range(7, 0, -1):
                            # one full-region wait consumes both chunks' completions
                            tpu.make_async_remote_copy(
                                xstage_h,
                                recv.at[slot, lane],
                                sends.at[offset - 1],
                                recvs.at[offset - 1],
                                device_id=(rank ^ offset,),
                                device_id_type=pl.DeviceIdType.MESH,
                            ).wait()
                    part = rsum(recv[slot])
                    xstage_h[...] = part.astype(xstage_h.dtype)
                    phase_hosts(xstage_h, recv_h, slot, tag)
                    return rsum(recv_h[slot])

                def attention_projection(li, prewait):
                    group = ATTN_GROUP
                    output_width = h // group
                    slot = (li * 4) % 2
                    skip = "noxchg" in ablate or "noxattn" in ablate
                    if ATTN_INPUT_COMPACT:
                        all_attn_stage[...] = attbuf[...].reshape(all_attn_stage.shape)
                        att_source = all_attn_stage
                    else:
                        att_source = attbuf
                    attn_inputs[rank % group] = att_source[...]
                    if not skip:
                        for offset in range(1, group):
                            tpu.make_async_remote_copy(
                                att_source, attn_inputs.at[rank % group], sends.at[offset - 1], recvs.at[offset - 1],
                                device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH,
                            ).start()
                    tpu.make_async_copy(ow_r.at[li], ow_slab, dsem.at[1]).wait()
                    if group == 2:
                        partials = []
                        for piece in range(2):
                            if piece and not skip:
                                tpu.make_async_remote_copy(
                                    att_source, attn_inputs.at[rank % group], sends.at[0], recvs.at[0],
                                    device_id=(rank ^ 1,), device_id_type=pl.DeviceIdType.MESH,
                                ).wait()
                            part = (rank % 2) ^ piece
                            source = jnp.broadcast_to(attn_inputs[part].reshape(1, nh * hd), (8, nh * hd))
                            partials.append(jnp.dot(source, ow_slab[pl.ds(part * nh * hd, nh * hd)], preferred_element_type=jnp.float32)[:1])
                        partial = partials[0] + partials[1]
                    else:
                        if not skip:
                            for offset in range(1, group):
                                tpu.make_async_remote_copy(
                                    att_source, attn_inputs.at[rank % group], sends.at[offset - 1], recvs.at[offset - 1],
                                    device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH,
                                ).wait()
                        source = jnp.broadcast_to(attn_inputs[...].reshape(1, group * nh * hd), (8, group * nh * hd))
                        partial = jnp.dot(source, ow_slab[...], preferred_element_type=jnp.float32)[:1]
                    stage = attn_stage if COMPACT_ATTN else xstage_h.at[:, :output_width]
                    local_recv = attn_recv if COMPACT_ATTN else recv.at[:, :, :, :output_width]
                    host_recv = attn_hosts if COMPACT_ATTN else recv_h.at[:, :, :, :output_width]
                    stage[...] = partial.reshape(stage.shape).astype(jnp.bfloat16)
                    if ATTN_DIRECT:
                        direct_gather(stage, direct_attn_recv, "noxattn", prewait, ATTENTION_TP)
                        parts = direct_attn_recv[:ATTENTION_TP].reshape(ATTENTION_TP // 8, 8 // group, group, output_width // 128, 128)
                        local_sum = jnp.sum(parts.astype(jnp.float32), axis=1).astype(jnp.bfloat16)
                        return jnp.sum(local_sum.astype(jnp.float32), axis=0).reshape(1, h)
                    if group < 8:
                        local_recv[slot, lane // group] = stage[...]
                        if not skip:
                            for offset in range(group, 8, group):
                                tpu.make_async_remote_copy(
                                    stage, local_recv.at[slot, lane // group],
                                    sends.at[offset - 1],
                                    recvs.at[offset - 1],
                                    device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH,
                                ).start()
                        prewait()
                        if not skip:
                            for offset in range(group, 8, group):
                                tpu.make_async_remote_copy(
                                    stage, local_recv.at[slot, lane // group],
                                    sends.at[offset - 1], recvs.at[offset - 1],
                                    device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH,
                                ).wait()
                        stage[...] = rsum(local_recv[slot, : 8 // group]).astype(jnp.bfloat16)
                    if ATTENTION_TP == 32:
                        phase_hosts(stage, host_recv, slot, "noxattn", prewait if group == 8 else None)
                        total = rsum(host_recv[slot])
                    else:
                        host_recv[slot, host % 2] = stage[...]
                        if not skip:
                            tpu.make_async_remote_copy(
                                stage, host_recv.at[slot, host % 2], sends.at[7], recvs.at[7],
                                device_id=(rank ^ 8,), device_id_type=pl.DeviceIdType.MESH,
                            ).start()
                        if group == 8:
                            prewait()
                        if not skip:
                            tpu.make_async_remote_copy(
                                stage, host_recv.at[slot, host % 2], sends.at[7], recvs.at[7],
                                device_id=(rank ^ 8,), device_id_type=pl.DeviceIdType.MESH,
                            ).wait()
                        total = rsum(host_recv[slot, :2])
                    output_stage = attn_output_stage if COMPACT_ATTN else proj.at[:, :output_width]
                    output_stage[...] = total
                    attn_outputs[rank % group] = total
                    if not skip:
                        for offset in range(1, group):
                            tpu.make_async_remote_copy(
                                output_stage, attn_outputs.at[rank % group],
                                sends.at[offset - 1], recvs.at[offset - 1],
                                device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH,
                            ).start()
                    if not skip:
                        for offset in range(1, group):
                            tpu.make_async_remote_copy(
                                output_stage, attn_outputs.at[rank % group],
                                sends.at[offset - 1], recvs.at[offset - 1],
                                device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH,
                            ).wait()
                    return attn_outputs[...].reshape(1, h)

                def allreduce_rows(iteration, tag=""):
                    """All-reduce rowsbuf ([top_k, mh] expert partials); fp32 [top_k, mh] out.
                    Reference path kept for A/B bisection (MPK_ROWS_XCHG=allreduce)."""
                    slot = iteration % 2
                    xstage_rows[...] = rowsbuf[...].reshape(xstage_rows.shape).astype(xstage_rows.dtype)
                    phase_local(xstage_rows, recv_rows, slot, tag)
                    part = jnp.sum(recv_rows[slot].astype(jnp.float32), axis=0)
                    xstage_rows[...] = part.astype(xstage_rows.dtype)
                    phase_hosts(xstage_rows, recv_rows_h, slot, tag)
                    return jnp.sum(recv_rows_h[slot].astype(jnp.float32), axis=0).reshape(c.top_k, mh)

                def gather_rows(slot, rw_arr, wsum, xgain):
                    """Row-transpose MoE exchange: rowsbuf [top_k, mh] f32 TP partials ->
                    unpinbuf [1, mh_r] bf16 = this rank's slice of the normed, router-weighted
                    mix. Lane j collects row j's host partial (p1: intra all-to-all by row),
                    exchanges it with the same lane on the other hosts (p2) giving the complete
                    global row so the per-row norm is local, then pushes its normed+weighted
                    row slice to every local lane (p3); summing received slices folds the mix
                    into the collective. Same bf16 two-level transport and f32 sum order as a
                    full allreduce + norm + mix + slice."""
                    if X3_PIPE:
                        # p1 copies were issued per-row inside run_experts (semaphore 10);
                        # one wait per destination consumes that row's bytes
                        if (
                            "noxchg" not in ablate
                            and "noxrows" not in ablate
                            and "nolocal" not in ablate
                            and "noexpertcopy" not in ablate
                        ):
                            for j in range(8):
                                tpu.make_async_remote_copy(
                                    xstage_rows.at[j],
                                    recv_r1.at[slot, lane],
                                    sends.at[10],
                                    recvs.at[10],
                                    device_id=(rank - lane + j,),
                                    device_id_type=pl.DeviceIdType.MESH,
                                ).wait()
                    else:
                        xstage_rows[...] = rowsbuf[...].reshape(xstage_rows.shape).astype(xstage_rows.dtype)
                        recv_r1[slot, lane, ...] = xstage_rows[lane, ...]
                        if "noxchg" not in ablate and "noxrows" not in ablate and "nolocal" not in ablate:
                            for offset in range(7, 0, -1):
                                tpu.make_async_remote_copy(
                                    xstage_rows.at[lane ^ offset],  # row lane^offset goes to that lane
                                    recv_r1.at[slot, lane],
                                    sends.at[offset - 1],
                                    recvs.at[offset - 1],
                                    device_id=(rank ^ offset,),
                                    device_id_type=pl.DeviceIdType.MESH,
                                ).start(priority=XCHG_PRIO)
                            for offset in range(7, 0, -1):
                                tpu.make_async_remote_copy(
                                    xstage_rows.at[lane ^ offset],
                                    recv_r1.at[slot, lane],
                                    sends.at[offset - 1],
                                    recvs.at[offset - 1],
                                    device_id=(rank ^ offset,),
                                    device_id_type=pl.DeviceIdType.MESH,
                                ).wait()
                    hostrow = rsum(recv_r1[slot])  # [1, mh] row `lane`
                    rowstage[...] = hostrow.astype(rowstage.dtype)
                    recv_r2[slot, host, ...] = rowstage[...]
                    if "noxchg" not in ablate and "noxrows" not in ablate and "nohost" not in ablate:
                        for offset in (8, 16, 24):
                            tpu.make_async_remote_copy(
                                rowstage,
                                recv_r2.at[slot, host],
                                sends.at[6 + offset // 8],
                                recvs.at[6 + offset // 8],
                                device_id=(rank ^ offset,),
                                device_id_type=pl.DeviceIdType.MESH,
                            ).start(priority=XCHG_PRIO)
                        for offset in (8, 16, 24):
                            tpu.make_async_remote_copy(
                                rowstage,
                                recv_r2.at[slot, host],
                                sends.at[6 + offset // 8],
                                recvs.at[6 + offset // 8],
                                device_id=(rank ^ offset,),
                                device_id_type=pl.DeviceIdType.MESH,
                            ).wait()
                    grow = rsum(recv_r2[slot])  # [1, mh] global row `lane`
                    grow = grow * xgain
                    grow = grow * jax.lax.rsqrt(jnp.mean(grow * grow, axis=-1, keepdims=True) + c.post_eps)
                    iota8 = jax.lax.iota(jnp.int32, 8)
                    w_j = jnp.sum(jnp.where(iota8 == lane, rw_arr, 0.0))
                    nrow[...] = (grow * (w_j / (wsum + c.router_eps))).reshape(nrow.shape)
                    if UNPOOL_COLUMNS:
                        recv_r3[slot, lane, ...] = nrow[...].reshape(1, mh)
                    else:
                        recv_r3[slot, lane, ...] = nrow[rank, ...]
                    if "noxchg" not in ablate and "noxrows" not in ablate and "nolocal" not in ablate:
                        for offset in range(7, 0, -1):
                            tpu.make_async_remote_copy(
                                nrow if UNPOOL_COLUMNS else nrow.at[rank ^ offset],
                                recv_r3.at[slot, lane],
                                sends.at[offset - 1],
                                recvs.at[offset - 1],
                                device_id=(rank ^ offset,),
                                device_id_type=pl.DeviceIdType.MESH,
                            ).start(priority=XCHG_PRIO)
                        for offset in range(7, 0, -1):
                            tpu.make_async_remote_copy(
                                nrow if UNPOOL_COLUMNS else nrow.at[rank ^ offset],
                                recv_r3.at[slot, lane],
                                sends.at[offset - 1],
                                recvs.at[offset - 1],
                                device_id=(rank ^ offset,),
                                device_id_type=pl.DeviceIdType.MESH,
                            ).wait()
                    unpinbuf[...] = jnp.sum(recv_r3[slot], axis=0).astype(unpinbuf.dtype)

                def gather_ep_rows(li, slot, rw_arr, wsum, xgain):
                    if ep_row_overlap and "noexpertcopy" not in ablate:
                        ep_rows_recv[slot, lane] = ep_rows_stage[...]
                        iota8 = jax.lax.iota(jnp.int32, 8)
                        weights = jnp.stack([
                            jnp.sum(jnp.where(iota8 == host + 4 * j, rw_arr, 0.0)) for j in range(2)
                        ]) / (wsum + c.router_eps)
                        weighted = []
                        for j in range(2):
                            if not ablate.intersection(("noxchg", "nolocal", "noxrows")):
                                region = ep_rows_recv.at[slot, :7, j]
                                tpu.make_async_remote_copy(
                                    region, region, sends.at[17 + j], recvs.at[17 + j],
                                    device_id=(rank,), device_id_type=pl.DeviceIdType.MESH,
                                ).wait()
                            row = jnp.sum(ep_rows_recv[slot, :, j].astype(jnp.float32), axis=0).reshape(1, mh)
                            if COMPACT_MATH:
                                row = compact_rms(row, c.post_eps, xgain, True)
                            else:
                                row *= xgain
                                row *= jax.lax.rsqrt(jnp.mean(row * row, axis=-1, keepdims=True) + c.post_eps)
                            weighted.append(row * weights[j])
                        unpinbuf[...] = (weighted[0] + weighted[1]).astype(jnp.bfloat16)
                        return
                    if COMPACT_EP_ROWS:
                        if EP_PIPE_ROWS and "noexpertcopy" not in ablate:
                            ep_rows_recv[slot, lane] = ep_rows_stage[...]
                            if not ablate.intersection(("noxchg", "nolocal", "noxrows")):
                                for offset in range(7, 0, -1):
                                    tpu.make_async_remote_copy(
                                        ep_rows_stage, ep_rows_recv.at[slot, lane],
                                        sends.at[offset - 1], recvs.at[offset - 1],
                                        device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH,
                                    ).wait()
                        else:
                            ep_rows_stage[...] = rowsbuf[:2].reshape(2, mh // 128, 128).astype(jnp.bfloat16)
                            phase_local(ep_rows_stage, ep_rows_recv, slot, "noxrows")
                        rows = jnp.sum(ep_rows_recv[slot].astype(jnp.float32), axis=0).reshape(2, mh)
                    else:
                        xstage_rows[:2] = rowsbuf[:2].reshape(2, 1, mh).astype(jnp.bfloat16)
                        phase_local(xstage_rows.at[:2], recv_rows.at[:, :, :2], slot, "noxrows")
                        rows = jnp.sum(recv_rows[slot, :, :2].astype(jnp.float32), axis=0).reshape(2, mh)
                    if COMPACT_MATH:
                        rows = compact_rms(rows, c.post_eps, xgain, True)
                    else:
                        rows *= xgain
                        rows *= jax.lax.rsqrt(jnp.mean(rows * rows, axis=-1, keepdims=True) + c.post_eps)
                    iota8 = jax.lax.iota(jnp.int32, 8)
                    weights = jnp.stack([
                        jnp.sum(jnp.where(iota8 == host + 4 * j, rw_arr, 0.0)) for j in range(2)
                    ]) / (wsum + c.router_eps)
                    mixed = jnp.sum(rows * weights[:, None], axis=0, keepdims=True)
                    if EP_UNPOOL_FUSED:
                        unpinbuf[...] = mixed.astype(jnp.bfloat16)
                        return
                    if UNPOOL_COLUMNS:
                        nrow[...] = mixed
                    else:
                        for part in range(tp):
                            nrow[part, 0] = mixed[0, part * mh_r : (part + 1) * mh_r]
                    ep_recv[slot, host] = nrow[...] if UNPOOL_COLUMNS else nrow[rank]
                    if "noxchg" not in ablate and "noxrows" not in ablate and "nohost" not in ablate:
                        for offset in (8, 16, 24):
                            tpu.make_async_remote_copy(
                                nrow if UNPOOL_COLUMNS else nrow.at[rank ^ offset],
                                ep_recv.at[slot, host],
                                sends.at[6 + offset // 8], recvs.at[6 + offset // 8],
                                device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH,
                            ).start()
                        for offset in (8, 16, 24):
                            tpu.make_async_remote_copy(
                                nrow if UNPOOL_COLUMNS else nrow.at[rank ^ offset],
                                ep_recv.at[slot, host],
                                sends.at[6 + offset // 8], recvs.at[6 + offset // 8],
                                device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH,
                            ).wait()
                    unpinbuf[...] = jnp.sum(ep_recv[slot], axis=0).astype(jnp.bfloat16)

                def gather_flat(src, slot, width, tag="", prewait=None):
                    """All-gather [1, width] f32 per-rank slices into recv_comb_h[slot]
                    ([4 hosts, 8 lanes, 1, width], rank-major). `prewait` runs between the
                    intra-host sends and waits so slab streams hide under the flight."""
                    recv_comb[slot, lane, 0, :] = src[0, :width]
                    if "noxchg" not in ablate and not (tag and tag in ablate):
                        for offset in range(7, 0, -1):
                            tpu.make_async_remote_copy(
                                src.at[0],
                                recv_comb.at[slot, lane, 0, :],
                                sends.at[offset - 1],
                                recvs.at[offset - 1],
                                device_id=(rank ^ offset,),
                                device_id_type=pl.DeviceIdType.MESH,
                            ).start(priority=XCHG_PRIO)
                    if prewait is not None:
                        prewait()
                    if "noxchg" not in ablate and not (tag and tag in ablate):
                        for offset in range(7, 0, -1):
                            tpu.make_async_remote_copy(
                                src.at[0],
                                recv_comb.at[slot, lane, 0, :],
                                sends.at[offset - 1],
                                recvs.at[offset - 1],
                                device_id=(rank ^ offset,),
                                device_id_type=pl.DeviceIdType.MESH,
                            ).wait()
                        recv_comb_h[slot, host] = recv_comb[slot]
                        for offset in (8, 16, 24):
                            tpu.make_async_remote_copy(
                                recv_comb.at[slot],
                                recv_comb_h.at[slot, host],
                                sends.at[6 + offset // 8],
                                recvs.at[6 + offset // 8],
                                device_id=(rank ^ offset,),
                                device_id_type=pl.DeviceIdType.MESH,
                            ).start(priority=XCHG_PRIO)
                        for offset in (8, 16, 24):
                            tpu.make_async_remote_copy(
                                recv_comb.at[slot],
                                recv_comb_h.at[slot, host],
                                sends.at[6 + offset // 8],
                                recvs.at[6 + offset // 8],
                                device_id=(rank ^ offset,),
                                device_id_type=pl.DeviceIdType.MESH,
                            ).wait()

                def rms_pair(x, eps):
                    """Weightless per-64-head RMSNorm on [.., 128] two-heads-per-row values."""
                    xf = x.astype(jnp.float32)
                    means = jnp.dot(xf * xf, sel128, preferred_element_type=jnp.float32)  # [.., 2]
                    inv = jax.lax.rsqrt(means + eps)
                    factor = jnp.concatenate(
                        (
                            jnp.broadcast_to(inv[..., 0:1], x.shape[:-1] + (64,)),
                            jnp.broadcast_to(inv[..., 1:2], x.shape[:-1] + (64,)),
                        ),
                        axis=-1,
                    )
                    return xf * factor  # fp32

                def rope(vec):
                    """[..., 128] two-heads-per-row split-half rotation (fp32 out)."""
                    xf = vec.astype(jnp.float32)
                    rA, iA = xf[..., :32], xf[..., 32:64]
                    rB, iB = xf[..., 64:96], xf[..., 96:]
                    return jnp.concatenate(
                        (
                            rA * cos_v - iA * sin_v,
                            rA * sin_v + iA * cos_v,
                            rB * cos_v - iB * sin_v,
                            rB * sin_v + iB * cos_v,
                        ),
                        axis=-1,
                    )

                def flash_block(qb_, kbg, vbg, amg_, alg_, aag_, gi, t, sliding):
                    """Online-softmax update for (group, parity) gi on the resident block."""
                    scores = (
                        jnp.dot(
                            qb_[...],
                            kbg.reshape(sbw, hdp).T,
                            preferred_element_type=jnp.float32,
                        )
                        * c.scale
                    )
                    slots_t = jnp.arange(sbw)[None, :] + t * sbw
                    lo = jnp.where(sliding, window_start, 0)
                    keep = (slots_t <= pos) & (slots_t >= lo)
                    scores = jnp.where(keep, scores, -jnp.inf)
                    m = jnp.maximum(amg_[gi, ...], jnp.max(scores, axis=-1, keepdims=True))
                    alpha = jnp.exp(amg_[gi, ...] - m)
                    p = jnp.exp(scores - m).astype(jnp.bfloat16)
                    aag_[gi, ...] = aag_[gi, ...] * alpha + jnp.dot(
                        p.astype(jnp.float32),
                        vbg.astype(jnp.float32),
                        preferred_element_type=jnp.float32,
                    )
                    alg_[gi, ...] = alg_[gi, ...] * alpha + jnp.sum(
                        p.astype(jnp.float32), axis=-1, keepdims=True
                    )
                    amg_[gi, ...] = m

                def attend_streaming(li, q8, k2, v2, sliding):
                    """q8 [nh//2, 128] (two heads per row), k2/v2 [nkv//2, 128] (same packing).

                    One streaming loop over the cache blocks; each K/V copy carries all
                    of this rank's kv heads (nkv) at once.
                    """
                    rows_per_group = groups // 2
                    count = cache_block + 1
                    first = jnp.where(sliding, window_first, 0)

                    def cache_write(cache, buffer, channel):
                        source_offset = write_source
                        target_offset = write_target
                        return tpu.make_async_copy(
                            buffer.at[(cache_block + li) % 2 if KV_PREFETCH == 2 else cache_block % 2, :, pl.ds(source_offset, kv_write_rows), :],
                            cache.at[li, :, pl.ds(target_offset, kv_write_rows), :], io.at[channel],
                        )
                    for g in range(nkv):
                        q4 = q8[g * rows_per_group : (g + 1) * rows_per_group]  # [rpg, 128]
                        # one pass per group: even heads (cols 0:64) in rows 0:rpg, odd
                        # heads shifted into cols 0:64 in rows rpg:2*rpg
                        q_odd = jnp.pad(q4[:, hd:], ((0, 0), (0, hd)))
                        qbg[g, ...] = jnp.pad(
                            jnp.concatenate((q4, q_odd), axis=0),
                            ((0, 8 - 2 * rows_per_group), (0, 0)),
                        )
                    amg[...] = jnp.full((nkv, 8, 1), -jnp.inf, jnp.float32)
                    alg[...] = jnp.zeros((nkv, 8, 1), jnp.float32)
                    aag[...] = jnp.zeros((nkv, 8, hdp), jnp.float32)
                    if not KV_PREFETCH:
                        prefetch_kv(li, first)

                    def step(t, _):
                        tpu.make_async_copy(
                            kc_r.at[li, :, pl.ds(t * sbw, sbw), :], kb.at[(t + li) % 2 if KV_PREFETCH == 2 else t % 2], ks.at[(t + li) % 2 if KV_PREFETCH == 2 else t % 2]
                        ).wait()
                        tpu.make_async_copy(
                            vc_r.at[li, :, pl.ds(t * sbw, sbw), :], vb.at[(t + li) % 2 if KV_PREFETCH == 2 else t % 2], vs.at[(t + li) % 2 if KV_PREFETCH == 2 else t % 2]
                        ).wait()

                        @pl.when(t == cache_block)
                        def update_current():
                            if KV_UPDATE_COMPACT:
                                mask = jnp.arange(kv_write_rows)[:, None] == pos % kv_write_rows
                                region = pl.ds(write_source, kv_write_rows)
                            else:
                                mask = jnp.arange(sbw)[:, None] == pos % sbw
                                region = slice(None)
                            for g in range(nkv):
                                key_g = k2[g // 2, (g % 2) * hd : (g % 2 + 1) * hd]
                                value_g = v2[g // 2, (g % 2) * hd : (g % 2 + 1) * hd]
                                slot = (t + li) % 2 if KV_PREFETCH == 2 else t % 2
                                kb[slot, g, region, :hd] = jnp.where(mask, key_g[None, :], kb[slot, g, region, :hd])
                                vb[slot, g, region, :hd] = jnp.where(mask, value_g[None, :], vb[slot, g, region, :hd])
                            cache_write(kc_r, kb, 0).start()
                            cache_write(vc_r, vb, 1).start()

                        if "noattn" not in ablate and "nocompute" not in ablate:
                            for g in range(nkv):
                                flash_block(qbg[g], kb[(t + li) % 2 if KV_PREFETCH == 2 else t % 2, g], vb[(t + li) % 2 if KV_PREFETCH == 2 else t % 2, g], amg, alg, aag, g, t, sliding)

                        @pl.when(t + 2 < count)
                        def refill(t=t):
                            tpu.make_async_copy(
                                kc_r.at[li, :, pl.ds((t + 2) * sbw, sbw), :], kb.at[(t + li) % 2 if KV_PREFETCH == 2 else t % 2], ks.at[(t + li) % 2 if KV_PREFETCH == 2 else t % 2]
                            ).start()
                            tpu.make_async_copy(
                                vc_r.at[li, :, pl.ds((t + 2) * sbw, sbw), :], vb.at[(t + li) % 2 if KV_PREFETCH == 2 else t % 2], vs.at[(t + li) % 2 if KV_PREFETCH == 2 else t % 2]
                            ).start()

                    jax.lax.fori_loop(first, count, step, None)
                    if KV_PREFETCH == 2:
                        @pl.when(count > 1)
                        def _():
                            cache_write(kc_r, kb, 0).wait()
                            cache_write(vc_r, vb, 1).wait()
                    else:
                        cache_write(kc_r, kb, 0).wait()
                        cache_write(vc_r, vb, 1).wait()
                    for g in range(nkv):
                        ae = (aag[g] / alg[g])[:rows_per_group, :hd].astype(jnp.bfloat16)
                        ao = (aag[g] / alg[g])[rows_per_group : 2 * rows_per_group, :hd].astype(jnp.bfloat16)
                        attended[pl.ds(g * rows_per_group, rows_per_group), :] = jnp.concatenate(
                            (ae, ao), axis=1
                        )
                    if KV_PREFETCH:
                        @pl.when(li + 1 < c.layers)
                        def _():
                            next_first = jnp.where((li + 1) % 4 != 1, window_first, 0)
                            prefetch_kv(li + 1, next_first)

                def attend_single(li, q8, k2, v2, sliding):
                    slot = li % 2 if KV_PREFETCH == 2 else 0
                    if not KV_PREFETCH:
                        prefetch_kv(li, 0)
                    for cache, buffer, semaphore in ((kc_r, kb, ks), (vc_r, vb, vs)):
                        tpu.make_async_copy(cache.at[li, :, :sbw, :], buffer.at[slot], semaphore.at[slot]).wait()
                    mask = jnp.arange(kv_write_rows)[:, None] == pos % kv_write_rows
                    region = pl.ds(write_source, kv_write_rows)
                    for g in range(nkv):
                        key_g = k2[g // 2, (g % 2) * hd:(g % 2 + 1) * hd]
                        value_g = v2[g // 2, (g % 2) * hd:(g % 2 + 1) * hd]
                        kb[slot, g, region, :hd] = jnp.where(mask, key_g[None], kb[slot, g, region, :hd])
                        vb[slot, g, region, :hd] = jnp.where(mask, value_g[None], vb[slot, g, region, :hd])
                    for cache, buffer, channel in ((kc_r, kb, 0), (vc_r, vb, 1)):
                        tpu.make_async_copy(
                            buffer.at[slot, :, region, :], cache.at[li, :, pl.ds(write_target, kv_write_rows), :], io.at[channel]
                        ).start()
                    rows_per_group = groups // 2
                    for g in range(nkv):
                        q4 = q8[g * rows_per_group:(g + 1) * rows_per_group]
                        q_odd = jnp.pad(q4[:, hd:], ((0, 0), (0, hd)))
                        queries = jnp.pad(jnp.concatenate((q4, q_odd), axis=0), ((0, 8 - 2 * rows_per_group), (0, 0)))
                        if 'noattn' not in ablate and 'nocompute' not in ablate:
                            scores = jnp.dot(queries, kb[slot, g].reshape(sbw, hdp).T, preferred_element_type=jnp.float32) * c.scale
                            positions = jnp.arange(sbw)[None]
                            scores = jnp.where((positions <= pos) & (positions >= jnp.where(sliding, window_start, 0)), scores, -jnp.inf)
                            probability = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True)).astype(jnp.bfloat16)
                            denominator = jnp.sum(probability.astype(jnp.float32), axis=-1, keepdims=True)
                            numerator = jnp.dot(probability.astype(jnp.float32), vb[slot, g].astype(jnp.float32), preferred_element_type=jnp.float32)
                            result = numerator / denominator
                        else:
                            result = jnp.full((8, hdp), jnp.nan, jnp.float32)
                        even = result[:rows_per_group, :hd].astype(jnp.bfloat16)
                        odd = result[rows_per_group:2 * rows_per_group, :hd].astype(jnp.bfloat16)
                        attended[pl.ds(g * rows_per_group, rows_per_group)] = jnp.concatenate((even, odd), axis=1)
                    if KV_PREFETCH != 2:
                        for cache, buffer, channel in ((kc_r, kb, 0), (vc_r, vb, 1)):
                            tpu.make_async_copy(
                                buffer.at[slot, :, region, :], cache.at[li, :, pl.ds(write_target, kv_write_rows), :], io.at[channel]
                            ).wait()
                    if KV_PREFETCH:
                        @pl.when(li + 1 < c.layers)
                        def prefetch_next_cache():
                            prefetch_kv(li + 1, 0)

                def attend(li, q8, k2, v2, sliding):
                    if KV_UPDATE_COMPACT:
                        @pl.when(cache_block == 0)
                        def single_block():
                            attend_single(li, q8, k2, v2, sliding)
                        @pl.when(cache_block != 0)
                        def multiple_blocks():
                            attend_streaming(li, q8, k2, v2, sliding)
                    else:
                        attend_streaming(li, q8, k2, v2, sliding)

                def prefetch_kv(li, first):
                    for offset in range(2):
                        t = first + offset
                        @pl.when(t < cache_block + 1)
                        def issue(t=t):
                            t0 = jnp.minimum(t, context // sbw - 1)
                            tpu.make_async_copy(kc_r.at[li, :, pl.ds(t0 * sbw, sbw), :], kb.at[(t + li) % 2 if KV_PREFETCH == 2 else t % 2], ks.at[(t + li) % 2 if KV_PREFETCH == 2 else t % 2]).start()
                            tpu.make_async_copy(vc_r.at[li, :, pl.ds(t0 * sbw, sbw), :], vb.at[(t + li) % 2 if KV_PREFETCH == 2 else t % 2], vs.at[(t + li) % 2 if KV_PREFETCH == 2 else t % 2]).start()

                def expert_gemv_ng(w_kn, x_k, n, k):
                    """w [K, N] bf16 x [K] -> [N] f32 (K-major matvec on the MXU)."""
                    a = jnp.broadcast_to(x_k.astype(jnp.bfloat16)[None, :], (8, k))
                    return jnp.dot(a, w_kn, preferred_element_type=jnp.float32)[0]

                def dequant_k_major(packed_buf, scales_buf, k, n):
                    """[K/2, N] u8 + E4M3 [K/16, N] -> bf16 [K, N] (pre-global-scale).

                    fp4 value (<=2 mantissa bits) x e4m3 scale (4) needs <=6 mantissa
                    bits, so the product is exact in bf16 and matches the reference's
                    bf16 GEMM operands.
                    """
                    w = tpu.bitcast(packed_buf, jnp.float4_e2m1fn).astype(jnp.bfloat16)
                    s = jax.lax.bitcast_convert_type(scales_buf, jnp.float8_e4m3fn).astype(jnp.bfloat16)
                    w = w.reshape(k // 16, 16, n) * s[:, None, :]
                    return w.reshape(k, n)

                def push_row(slot3, j, row_val):
                    """Stage expert row j and push this rank's partial to its owner lane
                    (row j) immediately, so the p1 exchange traffic overlaps the remaining
                    expert compute (MPK_X3_PIPE)."""
                    xstage_rows[j, 0, :] = row_val.astype(xstage_rows.dtype)
                    if "noxchg" in ablate or "noxrows" in ablate or "nolocal" in ablate:
                        return
                    tpu.make_async_remote_copy(
                        xstage_rows.at[j],
                        recv_r1.at[slot3, lane],
                        sends.at[10],
                        recvs.at[10],
                        device_id=(rank - lane + j,),
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start(priority=XCHG_PRIO)

                def early_experts(o_packed, ids, is_dense):
                    """Issue the first expert pairs the moment the router has decided them:
                    their HBM flight then overlaps the pooled norm and pair 0's wait finds
                    data already moving. packed_all's loop waits consume these."""
                    if "noexpertcopy" in ablate:
                        return

                    @pl.when(jnp.logical_not(is_dense))
                    def _():
                        for sp in range(min(max(1, EXPERT_SLOTS - 1), len(ids) // 2)):
                            e0, e1 = ids[2 * sp], ids[2 * sp + 1]
                            slot_ = sp % EXPERT_SLOTS
                            tpu.make_async_copy(gu_pack_r.at[o_packed, e0], gu_p2.at[slot_], eg_sem.at[2 * slot_]).start(priority=EXPERT_PRIO)
                            tpu.make_async_copy(gu_pack_r.at[o_packed, e1], guq_p2.at[slot_], eg_sem.at[2 * slot_ + (2 if EP_SERIAL else 0)]).start(priority=EXPERT_PRIO)
                            tpu.make_async_copy(dw_pack_r.at[o_packed, e0], dw_p2.at[slot_], eg_sem.at[2 * slot_ + 1]).start(priority=EXPERT_PRIO)
                            tpu.make_async_copy(dw_pack_r.at[o_packed, e1], dwq_p2.at[slot_], eg_sem.at[2 * slot_ + (3 if EP_SERIAL else 1)]).start(priority=EXPERT_PRIO)

                def push_ep_row(li, j, value):
                    ep_rows_stage[j] = value.reshape(mh // 128, 128).astype(jnp.bfloat16)
                    if ablate.intersection(("noxchg", "nolocal", "noxrows")):
                        return
                    for offset in range(7, 0, -1):
                        tpu.make_async_remote_copy(
                            ep_rows_stage.at[j], ep_rows_recv.at[(li * 4 + 2) % 2, lane, j],
                            sends.at[17 + j if ep_row_overlap else offset - 1],
                            recvs.at[17 + j if ep_row_overlap else offset - 1],
                            device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH,
                        ).start()

                def prefetch_deferred(li):
                    if DEFER_DENSE_PREFETCH & 1:
                        prefetch_slab(qwkv_r, qwkv_slab, 0, li, force=True)
                    if DEFER_DENSE_PREFETCH & 2:
                        prefetch_slab(ow_r, ow_slab, 1, li, force=True)
                    if DEFER_DENSE_PREFETCH & 4:
                        prefetch_slab(pool_r, pool_slab, 2, li, force=True)

                def run_experts(li, pooled, ids, packed_ids, is_dense, o_packed, o_dense, ggu, ggw):
                    """Rows for the 8 routed experts into rowsbuf (fp32 TP partials)."""
                    if "noexpertcopy" in ablate:
                        if DENSE_PREFETCH_STAGE == 2:
                            prefetch_deferred(li)
                        rowsbuf[...] = jnp.zeros((c.top_k, mh), jnp.float32)
                        return

                    def packed_all():
                        rows = []
                        iota_e = jax.lax.iota(jnp.int32, c.experts)
                        n_pairs = len(packed_ids) // 2
                        if EP_SERIAL:
                            for j in range(2):
                                expert = packed_ids[j]
                                gu_buf = gu_p2.at[0] if j == 0 else guq_p2.at[0]
                                dw_buf = dw_p2.at[0] if j == 0 else dwq_p2.at[0]
                                tpu.make_async_copy(gu_pack_r.at[o_packed, expert], gu_buf, eg_sem.at[2 * j]).wait()
                                if "noexpert" in ablate or "nocompute" in ablate:
                                    tpu.make_async_copy(dw_pack_r.at[o_packed, expert], dw_buf, eg_sem.at[2 * j + 1]).wait()
                                    rows.append(jnp.zeros((mh,), jnp.float32))
                                    if EP_PIPE_ROWS:
                                        push_ep_row(li, j, rows[-1])
                                    if DENSE_PREFETCH_STAGE == 2 and j == 0:
                                        prefetch_deferred(li)
                                    continue
                                mask = (iota_e == expert).astype(jnp.float32)
                                g13 = jnp.sum(ggu * jnp.stack((mask, mask), axis=1), axis=0)
                                g2 = jnp.sum(ggw * mask, axis=0)
                                gu_weights = dequant_k_major(gu_buf[:mh // 2], gu_buf[mh // 2:], mh, 2 * inter_r)
                                source = jnp.broadcast_to(pooled.astype(jnp.bfloat16)[None, :], (8, mh))
                                gu = jnp.dot(source, gu_weights, preferred_element_type=jnp.float32)[0]
                                gate = (gu[:inter_r] * g13[0]).astype(jnp.bfloat16)
                                up = (gu[inter_r:] * g13[1]).astype(jnp.bfloat16)
                                act = (jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)).astype(jnp.bfloat16)
                                tpu.make_async_copy(dw_pack_r.at[o_packed, expert], dw_buf, eg_sem.at[2 * j + 1]).wait()
                                down_weights = dequant_k_major(dw_buf[:inter_r // 2], dw_buf[inter_r // 2:], inter_r, mh)
                                down = expert_gemv_ng(down_weights, act, mh, inter_r)
                                rows.append(down * g2)
                                if EP_PIPE_ROWS:
                                    push_ep_row(li, j, rows[-1])
                                if DENSE_PREFETCH_STAGE == 2 and j == 0:
                                    prefetch_deferred(li)
                            rowsbuf[...] = jnp.pad(jnp.stack(rows), ((0, c.top_k - 2), (0, 0)))
                            return

                        def issue_pair(sp, slot_):
                            e0, e1 = packed_ids[2 * sp], packed_ids[2 * sp + 1]
                            # merged blobs: one copy per (family, expert); per-slot sems so
                            # outstanding pairs can't alias each other's completions
                            tpu.make_async_copy(gu_pack_r.at[o_packed, e0], gu_p2.at[slot_], eg_sem.at[2 * slot_]).start(priority=EXPERT_PRIO)
                            tpu.make_async_copy(gu_pack_r.at[o_packed, e1], guq_p2.at[slot_], eg_sem.at[2 * slot_]).start(priority=EXPERT_PRIO)
                            tpu.make_async_copy(dw_pack_r.at[o_packed, e0], dw_p2.at[slot_], eg_sem.at[2 * slot_ + 1]).start(priority=EXPERT_PRIO)
                            tpu.make_async_copy(dw_pack_r.at[o_packed, e1], dwq_p2.at[slot_], eg_sem.at[2 * slot_ + 1]).start(priority=EXPERT_PRIO)

                        # the first S-1 pairs were issued by early_experts right after the
                        # router decision; in-loop prefetch keeps S-1 in flight from there
                        for sp in range(n_pairs):
                            slot_ix = sp % EXPERT_SLOTS
                            e0, e1 = packed_ids[2 * sp], packed_ids[2 * sp + 1]
                            tpu.make_async_copy(gu_pack_r.at[o_packed, e0], gu_p2.at[slot_ix], eg_sem.at[2 * slot_ix]).wait()
                            tpu.make_async_copy(gu_pack_r.at[o_packed, e1], guq_p2.at[slot_ix], eg_sem.at[2 * slot_ix]).wait()
                            fm0 = (iota_e == e0).astype(jnp.float32)
                            fm1 = (iota_e == e1).astype(jnp.float32)
                            g13a = jnp.sum(ggu * jnp.stack((fm0, fm0), axis=1), axis=0)
                            g13b = jnp.sum(ggu * jnp.stack((fm1, fm1), axis=1), axis=0)
                            g2a = jnp.sum(ggw * fm0, axis=0)
                            g2b = jnp.sum(ggw * fm1, axis=0)

                            @pl.when(sp + max(1, EXPERT_SLOTS - 1) < n_pairs)
                            def prefetch(sp=sp):
                                issue_pair(sp + max(1, EXPERT_SLOTS - 1), (sp + max(1, EXPERT_SLOTS - 1)) % EXPERT_SLOTS)

                            if "noexpert" in ablate or "nocompute" in ablate:
                                tpu.make_async_copy(dw_pack_r.at[o_packed, e0], dw_p2.at[slot_ix], eg_sem.at[2 * slot_ix + 1]).wait()
                                tpu.make_async_copy(dw_pack_r.at[o_packed, e1], dwq_p2.at[slot_ix], eg_sem.at[2 * slot_ix + 1]).wait()
                                rows.append(jnp.zeros((mh,), jnp.float32))
                                rows.append(jnp.zeros((mh,), jnp.float32))
                                if EP_PIPE_ROWS:
                                    push_ep_row(li, 0, rows[-2])
                                    push_ep_row(li, 1, rows[-1])
                                if X3_PIPE and EXPERT_TP == 32:
                                    push_row((li * 4 + 2) % 2, 2 * sp, rows[-2])
                                    push_row((li * 4 + 2) % 2, 2 * sp + 1, rows[-1])
                                continue
                            # paired gate/up: two dots on the shared input (no weight concat)
                            a = jnp.broadcast_to(pooled.astype(jnp.bfloat16)[None, :], (8, mh))
                            w13a = dequant_k_major(gu_p2[slot_ix, : mh // 2], gu_p2[slot_ix, mh // 2 :], mh, 2 * inter_r)
                            gu_a = jnp.dot(a, w13a, preferred_element_type=jnp.float32)[0]
                            w13b = dequant_k_major(guq_p2[slot_ix, : mh // 2], guq_p2[slot_ix, mh // 2 :], mh, 2 * inter_r)
                            gu_b = jnp.dot(a, w13b, preferred_element_type=jnp.float32)[0]
                            # the down weights are only needed now; their copies have been
                            # streaming since issue_pair while the gate/up GEMVs ran
                            tpu.make_async_copy(dw_pack_r.at[o_packed, e0], dw_p2.at[slot_ix], eg_sem.at[2 * slot_ix + 1]).wait()
                            tpu.make_async_copy(dw_pack_r.at[o_packed, e1], dwq_p2.at[slot_ix], eg_sem.at[2 * slot_ix + 1]).wait()
                            gate_a = (gu_a[:inter_r] * g13a[0]).astype(jnp.bfloat16)
                            up_a = (gu_a[inter_r:] * g13a[1]).astype(jnp.bfloat16)
                            gate_b = (gu_b[:inter_r] * g13b[0]).astype(jnp.bfloat16)
                            up_b = (gu_b[inter_r:] * g13b[1]).astype(jnp.bfloat16)
                            act_a = (
                                jax.nn.silu(gate_a.astype(jnp.float32)) * up_a.astype(jnp.float32)
                            ).astype(jnp.bfloat16)
                            act_b = (
                                jax.nn.silu(gate_b.astype(jnp.float32)) * up_b.astype(jnp.float32)
                            ).astype(jnp.bfloat16)
                            w2a = dequant_k_major(dw_p2[slot_ix, : inter_r // 2], dw_p2[slot_ix, inter_r // 2 :], inter_r, mh)
                            rows.append(expert_gemv_ng(w2a, act_a, mh, inter_r) * g2a)
                            if EP_PIPE_ROWS:
                                push_ep_row(li, 0, rows[-1])
                            w2b = dequant_k_major(dwq_p2[slot_ix, : inter_r // 2], dwq_p2[slot_ix, inter_r // 2 :], inter_r, mh)
                            rows.append(expert_gemv_ng(w2b, act_b, mh, inter_r) * g2b)
                            if EP_PIPE_ROWS:
                                push_ep_row(li, 1, rows[-1])
                            if X3_PIPE and EXPERT_TP == 32:
                                push_row((li * 4 + 2) % 2, 2 * sp, rows[-2])
                                push_row((li * 4 + 2) % 2, 2 * sp + 1, rows[-1])
                        rowsbuf[...] = jnp.pad(jnp.stack(rows), ((0, c.top_k - 2 * n_pairs), (0, 0)))

                    def dense_all():
                        if dense_overlay:
                            # The packed-slot buffers are idle on dense layers: overlay
                            # the dense expert tiles on them (frees VMEM for the ring).
                            gu_gate = pool_alias.view(gu_p2, (mh, dense_inter_r), jnp.bfloat16)
                            gu_up = pool_alias.view(guq_p2, (mh, dense_inter_r), jnp.bfloat16)
                            dw_0 = pool_alias.view(dw_p2, (dense_inter_r, mh // 2), jnp.bfloat16)
                            dw_1 = pool_alias.view(dwq_p2, (dense_inter_r, mh // 2), jnp.bfloat16)
                        def issue_dense(index):
                            slot = index % 2
                            tpu.make_async_copy(gu_dense_r.at[o_dense, ids[index]], gu_d.at[slot], ed_sem.at[2 * slot]).start()
                            tpu.make_async_copy(dw_dense_r.at[o_dense, ids[index]], dw_d.at[slot], ed_sem.at[2 * slot + 1]).start()

                        if DENSE_PIPE == 1:
                            issue_dense(0)
                            issue_dense(1)
                        rows = []
                        for s in range(c.top_k):
                            e = ids[s]
                            gu_current = gu_d.at[s % 2] if DENSE_PIPE else gu_d
                            dw_current = dw_d.at[s % 2] if DENSE_PIPE else dw_d
                            if dense_overlay:
                                tpu.make_async_copy(
                                    gu_dense_r.at[o_dense, e, :, pl.ds(0, dense_inter_r)], gu_gate, ed_sem.at[0]
                                ).start()
                                tpu.make_async_copy(
                                    gu_dense_r.at[o_dense, e, :, pl.ds(dense_inter_r, dense_inter_r)], gu_up, ed_sem.at[1]
                                ).start()
                                tpu.make_async_copy(
                                    dw_dense_r.at[o_dense, e, :, pl.ds(0, mh // 2)], dw_0, ed_sem.at[2]
                                ).start()
                                tpu.make_async_copy(
                                    dw_dense_r.at[o_dense, e, :, pl.ds(mh // 2, mh // 2)], dw_1, ed_sem.at[3]
                                ).start()
                                tpu.make_async_copy(
                                    gu_dense_r.at[o_dense, e, :, pl.ds(0, dense_inter_r)], gu_gate, ed_sem.at[0]
                                ).wait()
                                tpu.make_async_copy(
                                    gu_dense_r.at[o_dense, e, :, pl.ds(dense_inter_r, dense_inter_r)], gu_up, ed_sem.at[1]
                                ).wait()
                                tpu.make_async_copy(
                                    dw_dense_r.at[o_dense, e, :, pl.ds(0, mh // 2)], dw_0, ed_sem.at[2]
                                ).wait()
                                tpu.make_async_copy(
                                    dw_dense_r.at[o_dense, e, :, pl.ds(mh // 2, mh // 2)], dw_1, ed_sem.at[3]
                                ).wait()
                            elif DENSE_PIPE:
                                tpu.make_async_copy(gu_dense_r.at[o_dense, e], gu_current, ed_sem.at[2 * (s % 2)]).wait()
                                tpu.make_async_copy(dw_dense_r.at[o_dense, e], dw_current, ed_sem.at[2 * (s % 2) + 1]).wait()
                            else:
                                tpu.make_async_copy(gu_dense_r.at[o_dense, e], gu_d, ed_sem.at[0]).start()
                                tpu.make_async_copy(dw_dense_r.at[o_dense, e], dw_d, ed_sem.at[1]).start()
                                tpu.make_async_copy(gu_dense_r.at[o_dense, e], gu_d, ed_sem.at[0]).wait()
                                tpu.make_async_copy(dw_dense_r.at[o_dense, e], dw_d, ed_sem.at[1]).wait()
                            if "noexpert" in ablate or "nocompute" in ablate:
                                rows.append(jnp.zeros((mh,), jnp.float32))
                            elif dense_overlay:
                                gate = expert_gemv_ng(gu_gate[...], pooled, dense_inter_r, mh)
                                up = expert_gemv_ng(gu_up[...], pooled, dense_inter_r, mh)
                                act = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
                                rows.append(
                                    jnp.concat(
                                        (
                                            expert_gemv_ng(dw_0[...], act, mh // 2, dense_inter_r),
                                            expert_gemv_ng(dw_1[...], act, mh // 2, dense_inter_r),
                                        )
                                    )
                                )
                            else:
                                gu = expert_gemv_ng(gu_current[...], pooled, 2 * dense_inter_r, mh)
                                gate, up = gu[:dense_inter_r], gu[dense_inter_r:]
                                act = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
                                rows.append(expert_gemv_ng(dw_current[...], act, mh, dense_inter_r))
                            if X3_PIPE:
                                push_row((li * 4 + 2) % 2, s, rows[-1])
                            if DENSE_PIPE and s + 2 < c.top_k:
                                issue_dense(s + 2)
                            if DENSE_PREFETCH_STAGE == 2 and s == 0:
                                prefetch_deferred(li)
                        rowsbuf[...] = jnp.stack(rows)

                    @pl.when(is_dense)
                    def _():
                        dense_all()

                    @pl.when(jnp.logical_not(is_dense))
                    def _():
                        packed_all()

                def layer(li, _):
                    sliding = (li % 4) != 1
                    is_dense = jnp.bool_(False)
                    for d in c.bf16_expert_layers:
                        is_dense = is_dense | (li == d)
                    dense_below = jnp.zeros((), jnp.int32)
                    for d in c.bf16_expert_layers:
                        dense_below = dense_below + (li > d).astype(jnp.int32)
                    o_packed = jnp.clip(li - dense_below, 0, max(len_packed - 1, 0))
                    o_dense = jnp.clip(dense_below, 0, len(c.bf16_expert_layers) - 1)
                    tpu.make_async_copy(scalars_r.at[li], scalarsbuf.at[li % 2], sc_sem.at[0]).wait()
                    tpu.make_async_copy(gu_global_r.at[o_packed], ggu.at[li % 2], g_sem.at[0]).wait()
                    tpu.make_async_copy(dw_global_r.at[o_packed], ggw.at[li % 2], g_sem.at[1]).wait()
                    sv = scalarsbuf[li % 2]

                    @pl.when(li + 1 < c.layers)
                    def prefetch_next():
                        dense_below_next = jnp.zeros((), jnp.int32)
                        for d in c.bf16_expert_layers:
                            dense_below_next = dense_below_next + (li + 1 > d).astype(jnp.int32)
                        nxt_packed = jnp.clip(li + 1 - dense_below_next, 0, max(len_packed - 1, 0))
                        tpu.make_async_copy(
                            scalars_r.at[li + 1], scalarsbuf.at[(li + 1) % 2], sc_sem.at[0]
                        ).start()
                        tpu.make_async_copy(gu_global_r.at[nxt_packed], ggu.at[(li + 1) % 2], g_sem.at[0]).start()
                        tpu.make_async_copy(dw_global_r.at[nxt_packed], ggw.at[(li + 1) % 2], g_sem.at[1]).start()

                    # --- attention branch ---
                    if COMPACT_MATH:
                        normed[...] = compact_rms(residual[...], c.eps, sv[0]).astype(jnp.bfloat16)
                    else:
                        value = residual[...].astype(jnp.float32)
                        value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + c.eps)
                        normed[...] = (value * sv[0].astype(jnp.float32)).astype(jnp.bfloat16)
                    tpu.make_async_copy(qwkv_r.at[li], qwkv_slab, dsem.at[0]).wait()
                    slab_dot(
                        normed, qwkvbuf, qwkv_slab, 2 * nh * hd + 2 * nkv * hd,
                    )
                    qg = qwkvbuf[0, : 2 * nh * hd].reshape(nh, hdp)  # [q pairs | gate pairs]
                    q8 = qg[: nh // 2]
                    g8 = qg[nh // 2 :]
                    kv_off = 2 * nh * hd
                    if nkv == 1:
                        k2 = jnp.pad(qwkvbuf[0, kv_off:kv_off + hd][None, :], ((0, 0), (0, hdp - hd)))
                        v2 = jnp.pad(qwkvbuf[0, kv_off + hd:kv_off + 2 * hd][None, :], ((0, 0), (0, hdp - hd)))
                    else:
                        k2 = qwkvbuf[0, kv_off:kv_off + nkv * hd].reshape(nkv // 2, hdp)
                        v2 = qwkvbuf[0, kv_off + nkv * hd:kv_off + 2 * nkv * hd].reshape(nkv // 2, hdp)
                    q8 = rms_pair(q8, c.eps).astype(jnp.bfloat16)
                    k2 = rms_pair(k2, c.eps).astype(jnp.bfloat16)
                    q8 = jnp.where(sliding, rope(q8), q8.astype(jnp.float32)).astype(jnp.bfloat16)
                    k2 = jnp.where(sliding, rope(k2), k2.astype(jnp.float32)).astype(jnp.bfloat16)
                    attend(li, q8, k2, v2, sliding)
                    # o_norm (weightless, per head) then the output gate
                    ov = attended[: nh // 2].astype(jnp.float32)
                    ov = rms_pair(ov, c.eps)
                    gated = (ov * jax.nn.sigmoid(g8.astype(jnp.float32))).astype(jnp.bfloat16)
                    attbuf[0, pl.ds(0, nh * hd)] = gated.reshape(nh * hd)
                    def prewait_attn():
                        prefetch_slab(qwkv_r, qwkv_slab, 0, li)
                        prefetch_slab(ow_r, ow_slab, 1, li)
                        if LM_PREFETCH and not shared_arenas and "nolm" not in ablate:
                            @pl.when(li == c.layers - 1)
                            def _():
                                for slot in range(2):
                                    tpu.make_async_copy(
                                        lmw_r.at[:, pl.ds(slot * lm_chunk_w, lm_chunk_w)], lm_slab.at[slot], dsem.at[5 + slot]
                                    ).start()
                    if ATTN_GROUP == 1:
                        tpu.make_async_copy(ow_r.at[li], ow_slab, dsem.at[1]).wait()
                        total = gemv_allreduce(attbuf, ow_slab, li * 4, "noxattn", prewait=prewait_attn)
                    else:
                        total = attention_projection(li, prewait_attn)
                    if KV_PREFETCH == 2:
                        @pl.when(cache_block == 0)
                        def wait_deferred_cache_writes():
                            for cache, buffer, channel in ((kc_r, kb, 0), (vc_r, vb, 1)):
                                tpu.make_async_copy(
                                    buffer.at[li % 2, :, pl.ds(write_source, kv_write_rows), :],
                                    cache.at[li, :, pl.ds(write_target, kv_write_rows), :], io.at[channel],
                                ).wait()
                    update_residual(total, sv[3], sv[4])

                    # --- MoE branch ---
                    if COMPACT_MATH:
                        normed[...] = compact_rms(residual[...], c.eps, sv[1]).astype(jnp.bfloat16)
                    else:
                        value = residual[...].astype(jnp.float32)
                        value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + c.eps)
                        normed[...] = (value * sv[1].astype(jnp.float32)).astype(jnp.bfloat16)
                    tpu.make_async_copy(router_r.at[li], routerbuf, dsem.at[4]).wait()
                    if ROUTER_TRANSPOSE:
                        rpartial = jnp.sum(
                            normed[0].astype(jnp.float32)[None, :] * routerbuf[...], axis=1
                        )
                    else:
                        rpartial = jnp.sum(
                            normed[0].astype(jnp.float32)[:, None] * routerbuf[...], axis=0
                        )
                    tpu.make_async_copy(pool_r.at[li], pool_slab, dsem.at[2]).wait()
                    slab_dot(normed, poolbuf, pool_slab, pool_width)
                    # one fused gather: router partial in [0:128], pool shard in [128:128+mh_r]
                    combuf[0, :eshard] = rpartial
                    combuf[0, 128 : 128 + pool_width] = poolbuf[0].astype(jnp.float32)
                    slot = (li * 4 + 1) % 2
                    def _prewait_x2():
                        prefetch_slab(pool_r, pool_slab, 2, li)

                        @pl.when(li + 1 < c.layers)
                        def _():
                            tpu.make_async_copy(router_r.at[li + 1], routerbuf, dsem.at[4]).start()

                    if LOCAL_POOL:
                        phase_local(combuf, recv_comb, slot, "noxrouter", _prewait_x2)
                        local_combined = recv_comb[slot, :, 0, :]
                        logits_all = jnp.concatenate([local_combined[r, :eshard] for r in range(8)], axis=0)
                    else:
                        gather_flat(combuf, slot, 128 + pool_width, "noxrouter", prewait=_prewait_x2)
                        logits_all = jnp.concatenate(
                            [recv_comb_h[slot, r // 8, r % 8, 0, :eshard] for r in range(32)], axis=0
                        )
                    scores_all = jax.nn.sigmoid(logits_all * c.output_multiplier)
                    ranked = scores_all + biasbuf[li].astype(jnp.float32)
                    iota_e = jax.lax.iota(jnp.int32, c.experts)
                    ids_l, rw_l = [], []
                    wsum = jnp.zeros((), jnp.float32)
                    for j in range(c.top_k):
                        idx = jnp.argmax(ranked).astype(jnp.int32)
                        if DENSE_PIPE == 2 and j < 2 and "noexpertcopy" not in ablate:
                            @pl.when(is_dense)
                            def issue_dense_expert():
                                tpu.make_async_copy(gu_dense_r.at[o_dense, idx], gu_d.at[j], ed_sem.at[2 * j]).start()
                                tpu.make_async_copy(dw_dense_r.at[o_dense, idx], dw_d.at[j], ed_sem.at[2 * j + 1]).start()
                        if ROUTER_PREFETCH and EXPERT_TP == 8 and "noexpertcopy" not in ablate:
                            @pl.when(jnp.logical_not(is_dense) & (host == j % 4))
                            def issue_routed_expert():
                                gu_dst = gu_p2.at[0] if j < 4 else guq_p2.at[0]
                                dw_dst = dw_p2.at[0] if j < 4 else dwq_p2.at[0]
                                semaphore_offset = 2 if EP_SERIAL and j >= 4 else 0
                                tpu.make_async_copy(gu_pack_r.at[o_packed, idx], gu_dst, eg_sem.at[semaphore_offset]).start(priority=EXPERT_PRIO)
                                tpu.make_async_copy(dw_pack_r.at[o_packed, idx], dw_dst, eg_sem.at[semaphore_offset + 1]).start(priority=EXPERT_PRIO)
                        ids_l.append(idx)
                        wj = jnp.sum(jnp.where(iota_e == idx, scores_all, 0.0))
                        rw_l.append(wj)
                        wsum = wsum + wj
                        ranked = jnp.where(iota_e == idx, -jnp.inf, ranked)
                    if DENSE_PREFETCH_STAGE == 1:
                        prefetch_deferred(li)
                    rw_arr = jnp.stack(rw_l)
                    if EXPERT_TP == 8:
                        packed_ids = [
                            jnp.sum(jnp.where(jax.lax.iota(jnp.int32, 8) == host + 4 * j, jnp.stack(ids_l), 0))
                            for j in range(2)
                        ]
                    else:
                        packed_ids = ids_l
                    if not (ROUTER_PREFETCH and EXPERT_TP == 8):
                        early_experts(o_packed, packed_ids, is_dense)
                    if LOCAL_POOL:
                        pooledbuf[0] = jnp.concatenate(
                            [local_combined[r, 128 : 128 + pool_width] for r in range(8)], axis=0,
                        ).astype(jnp.bfloat16)
                    else:
                        pooledbuf[0] = jnp.concatenate(
                            [recv_comb_h[slot, r // 8, r % 8, 0, 128 : 128 + pool_width] for r in range(32)], axis=0,
                        ).astype(jnp.bfloat16)
                    if COMPACT_MATH:
                        pooledbuf[0] = compact_rms(pooledbuf[0], c.eps, xnormsbuf[li, 0]).astype(jnp.bfloat16)
                    else:
                        pv = pooledbuf[0].astype(jnp.float32)
                        pv *= jax.lax.rsqrt(jnp.mean(pv * pv, axis=-1, keepdims=True) + c.eps)
                        pooledbuf[0] = (pv * xnormsbuf[li, 0].astype(jnp.float32)).astype(jnp.bfloat16)

                    run_experts(
                        li, pooledbuf[0], ids_l, packed_ids, is_dense, o_packed, o_dense,
                        ggu[li % 2],
                        ggw[li % 2],
                    )

                    if shared_arenas and LM_PREFETCH and "nolm" not in ablate:
                        @pl.when(li == c.layers - 1)
                        def prefetch_final_head():
                            for slot in range(2):
                                tpu.make_async_copy(
                                    lmw_r.at[:, pl.ds(slot * lm_chunk_w, lm_chunk_w)], lm_slab.at[slot], dsem.at[5 + slot]
                                ).start()
                    if DENSE_PREFETCH_STAGE == 3:
                        prefetch_deferred(li)

                    if ROWS_XCHG == "transpose":
                        if EXPERT_TP == 8:
                            @pl.when(is_dense)
                            def _():
                                gather_rows((li * 4 + 2) % 2, rw_arr, wsum, xnormsbuf[li, 1].astype(jnp.float32))

                            @pl.when(jnp.logical_not(is_dense))
                            def _():
                                gather_ep_rows(li, (li * 4 + 2) % 2, rw_arr, wsum, xnormsbuf[li, 1].astype(jnp.float32))
                        else:
                            gather_rows((li * 4 + 2) % 2, rw_arr, wsum, xnormsbuf[li, 1].astype(jnp.float32))
                    else:
                        rows = allreduce_rows(li * 4 + 2, "noxrows")  # [top_k, mh] f32
                        rows = rows * xnormsbuf[li, 1].astype(jnp.float32)
                        rows = rows * jax.lax.rsqrt(jnp.mean(rows * rows, axis=-1, keepdims=True) + c.post_eps)
                        mixed = jnp.sum(rows * (rw_arr / (wsum + c.router_eps))[:, None], axis=0)
                        mixedbuf[0] = mixed.astype(jnp.bfloat16)
                        unpinbuf[0] = mixedbuf[0] if UNPOOL_COLUMNS else mixedbuf[0, pl.ds(rank * mh_r, mh_r)]
                    tpu.make_async_copy(unpool_r.at[li], unpool_slab, dsem.at[3]).wait()
                    def prewait_unpool():
                        prefetch_slab(unpool_r, unpool_slab, 3, li)
                    if LOCAL_UNPOOL:
                        output_width = unpool_output_width
                        source = proj.at[:, :output_width]
                        slab_dot(unpinbuf, source, unpool_slab, output_width, out_f32=True)
                        if compact_unpool:
                            unpool_stage[...] = source[...].reshape(unpool_stage.shape)
                            source = unpool_stage
                        if UNPOOL_DIRECT:
                            transport = source[...].reshape(-1)
                            transport = jnp.pad(transport, (0, max(output_width, 128) - output_width))
                            if UNPOOL_DIRECT >= 2:
                                direct_unpool_stage[...] = transport.reshape(direct_unpool_stage.shape).astype(direct_unpool_stage.dtype)
                                source = direct_unpool_stage
                            else:
                                unpool_stage[...] = transport.reshape(unpool_stage.shape)
                                source = unpool_stage
                            # A TP16 replica can advance while another still consumes the previous unpool.
                            unpool_slot = li % 2
                            direct_gather(source, direct_unpool_recv.at[unpool_slot], "noxunpool", prewait_unpool, phase=1 + unpool_slot)
                            parts = direct_unpool_recv[unpool_slot].reshape(4, 8, -1)[..., :output_width].astype(jnp.float32)
                            host_part = jnp.sum(jnp.where(jnp.arange(4)[:, None, None] == host, parts, 0.0), axis=0)
                            total = jnp.where(is_dense, host_part, jnp.sum(parts, axis=0))
                            if output_width < 128:
                                total = jnp.concatenate([total[r:r + 1] for r in range(8)], axis=-1)
                            else:
                                total = total.reshape(1, h)
                        else:
                            total = local_unpool_exchange(li, source, is_dense, prewait_unpool)
                    elif UNPOOL_COLUMNS:
                        slab_dot(unpinbuf, combuf.at[:, : h // tp], unpool_slab, h // tp, out_f32=True)
                        slot = (li * 4 + 3) % 2
                        gather_flat(
                            combuf, slot, 128 + pool_width, "noxunpool",
                            prewait=prewait_unpool,
                        )
                        total = jnp.concatenate(
                            [recv_comb_h[slot, r // 8, r % 8, :, : h // tp] for r in range(tp)], axis=-1
                        )
                    else:
                        total = gemv_allreduce(
                            unpinbuf, unpool_slab, li * 4 + 3, "noxunpool",
                            prewait=prewait_unpool,
                        )
                    update_residual(total, sv[5], sv[6], sv[2])
                    for index, layer_id in enumerate(aux_layers):
                        @pl.when(li == layer_id)
                        def capture_aux(index=index):
                            aux_buffer[index] = residual[...].astype(jnp.bfloat16)

                def local_unpool_exchange(li, source, is_dense, prewait):
                    output_width = h // 8
                    if EP_UNPOOL_FUSED:
                        @pl.when(jnp.logical_not(is_dense))
                        def reduce_projected_experts():
                            slot = (li * 4 + 3) % 2
                            host_recv = unpool_hosts if compact_unpool else ep_recv.at[:, :, :, :output_width]
                            phase_hosts(source, host_recv, slot, "noxunpool")
                            source[...] = jnp.sum(host_recv[slot], axis=0)
                    local_unpool_recv[lane] = source[...]
                    if "noxchg" not in ablate and "noxunpool" not in ablate:
                        for offset in range(7, 0, -1):
                            tpu.make_async_remote_copy(
                                source, local_unpool_recv.at[lane],
                                sends.at[offset - 1],
                                recvs.at[offset - 1],
                                device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH,
                            ).start()
                    prewait()
                    if "noxchg" not in ablate and "noxunpool" not in ablate:
                        for offset in range(7, 0, -1):
                            tpu.make_async_remote_copy(
                                source, local_unpool_recv.at[lane], sends.at[offset - 1], recvs.at[offset - 1],
                                device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH,
                            ).wait()
                    return local_unpool_recv[...].reshape(1, h) if compact_unpool else jnp.concatenate([local_unpool_recv[r] for r in range(8)], axis=-1)
                # Selector for per-64-half means on the two-heads-per-row [.., 128] views.
                i128 = jax.lax.iota(jnp.int32, 128)
                sel128 = jnp.stack(
                    (jnp.where(i128 < 64, 1.0 / 64, 0.0), jnp.where(i128 >= 64, 1.0 / 64, 0.0)), axis=1
                )

                # RoPE tables for this position (fp32).
                half = hd // 2
                inv = 1.0 / (c.rope_theta ** (jax.lax.iota(jnp.int32, half).astype(jnp.float32) / half))
                ang = pos.astype(jnp.float32) * inv
                cos_v = jnp.cos(ang)
                sin_v = jnp.sin(ang)

                # Embedding: fetch this rank's candidate row, all-reduce, weightless norm.
                tok = token_out[0]
                lo = jnp.int32(0) if REPLICATE_EMBEDDING else rank * vshard
                tokl = tok if REPLICATE_EMBEDDING else jnp.clip(tok - lo, 0, vshard - 1)
                tpu.make_async_copy(emb_r.at[pl.ds(tokl // 8 * 8, 8), :], rowbuf, e_sem.at[0]).start()
                tpu.make_async_copy(scalars_r.at[0], scalarsbuf.at[0], sc_sem.at[0]).start()
                # layer-0 dense slabs (later layers prefetch in-layer)
                @pl.when((token_index == 0) | (not shared_arenas))
                def initialize_qwkv():
                    tpu.make_async_copy(qwkv_r.at[0], qwkv_slab, dsem.at[0]).start()
                @pl.when((token_index == 0) | (not bool(chunk_size)))
                def initialize_ow():
                    tpu.make_async_copy(ow_r.at[0], ow_slab, dsem.at[1]).start()
                if not shared_arenas:
                    @pl.when((token_index == 0) | (not bool(chunk_size)))
                    def initialize_pool():
                        tpu.make_async_copy(pool_r.at[0], pool_slab, dsem.at[2]).start()
                if not shared_arenas:
                    @pl.when((token_index == 0) | (not bool(chunk_size)))
                    def initialize_unpool():
                        tpu.make_async_copy(unpool_r.at[0], unpool_slab, dsem.at[3]).start()
                tpu.make_async_copy(router_r.at[0], routerbuf, dsem.at[4]).start()
                if KV_PREFETCH:
                    prefetch_kv(jnp.int32(0), window_first)
                g0_packed = jnp.int32(0 if 0 in c.bf16_expert_layers else 0)
                tpu.make_async_copy(gu_global_r.at[g0_packed], ggu.at[0], g_sem.at[0]).start()
                tpu.make_async_copy(dw_global_r.at[g0_packed], ggw.at[0], g_sem.at[1]).start()
                barrier = tpu.get_barrier_semaphore()
                if tp > 1:
                    @pl.when(token_index == 0)
                    def synchronize_entry():
                        for offset in range(1, tp):
                            pl.semaphore_signal(
                                barrier, 1, device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH
                            )
                        pl.semaphore_wait(barrier, tp - 1)
                tpu.make_async_copy(emb_r.at[pl.ds(tokl // 8 * 8, 8), :], rowbuf, e_sem.at[0]).wait()
                sel = jax.lax.iota(jnp.int32, 8)
                row = jnp.sum(jnp.where(sel[:, None] == tokl % 8, rowbuf[...], 0), axis=0).astype(
                    jnp.float32
                )
                if REPLICATE_EMBEDDING:
                    total = row[None, :]
                else:
                    row = jnp.where((tok >= lo) & (tok < lo + vshard), row, 0.0)
                    proj[...] = row[None, :]
                    total = allreduce(1)
                value = total * jax.lax.rsqrt(jnp.mean(total * total, axis=-1, keepdims=True) + c.eps)
                residual[...] = value

                jax.lax.fori_loop(0, c.layers, layer, None)
                if aux_layers:
                    tpu.make_async_copy(aux_buffer, aux_out.at[token_index], aux_sem.at[0]).start()
                    tpu.make_async_copy(aux_buffer, aux_out.at[token_index], aux_sem.at[0]).wait()

                if "nolm" in ablate:
                    logits_out[...] = jnp.zeros(logits_out.shape, jnp.float32)
                    token_out[...] = jnp.full((1,), 200000, jnp.int32)
                    return

                # Final norm (plain gain, center 0) + LM head + greedy argmax.
                tpu.make_async_copy(fnorm_r, fnormbuf, sc_sem.at[1]).start()
                if not LM_PREFETCH:
                    for slot in range(2):
                        tpu.make_async_copy(
                            lmw_r.at[:, pl.ds(slot * lm_chunk_w, lm_chunk_w)], lm_slab.at[slot], dsem.at[5 + slot]
                        ).start()
                tpu.make_async_copy(fnorm_r, fnormbuf, sc_sem.at[1]).wait()
                value = residual[...].astype(jnp.float32)
                value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + c.eps)
                normed[...] = (value * fnormbuf[0].astype(jnp.float32)).astype(jnp.bfloat16)
                nchunks = lm_w // lm_chunk_w
                for cix in range(nchunks):
                    slot = cix % 2
                    tpu.make_async_copy(
                        lmw_r.at[:, pl.ds(cix * lm_chunk_w, lm_chunk_w)], lm_slab.at[slot], dsem.at[5 + slot]
                    ).wait()
                    if "nogemv" not in ablate:
                        a = jnp.broadcast_to(normed[...], (8, h))
                        values = jnp.dot(a, lm_slab[slot], preferred_element_type=jnp.float32)[0]
                        lm_accum[0, pl.ds(cix * lm_chunk_w, lm_chunk_w)] = values
                    if cix + 2 < nchunks:  # static python condition, no region
                        tpu.make_async_copy(
                            lmw_r.at[:, pl.ds((cix + 2) * lm_chunk_w, lm_chunk_w)],
                            lm_slab.at[slot],
                            dsem.at[5 + slot],
                        ).start()
                logits_out[...] = lm_accum[:1, :lm_w].astype(logits_out.dtype)
                if "dumpres" in ablate:
                    # debug path (tests only): overwrite the padded logits tail with the stream
                    if vshard + h <= lm_w:
                        logits_out[0, vshard : vshard + h] = residual[0]

                vals = lm_accum[0, :vshard]
                top = jnp.max(vals)
                iota = jax.lax.iota(jnp.int32, vshard)
                best = jnp.min(jnp.where(vals == top, iota, vshard)).astype(jnp.int32)
                pairbuf[...] = jnp.concatenate((top[None], best.astype(jnp.float32)[None]))
                if HIERARCHICAL_ARGMAX:
                    phase_local(pairbuf, pairs.at[:, :8], 0)
                    values = pairs[0, :8, 0]
                    top_value = jnp.max(values)
                    lanes = jax.lax.iota(jnp.int32, 8)
                    winner = jnp.min(jnp.where(values == top_value, lanes, 8))
                    index = jnp.sum(jnp.where(lanes == winner, pairs[0, :8, 1], 0.0))
                    token_id = (rank - lane + winner) * vshard + index
                    pairbuf[...] = jnp.concatenate((top_value[None], token_id.astype(jnp.float32)[None]))
                    phase_hosts(pairbuf, pairs.at[:, :4], 1)
                    values = pairs[1, :4, 0]
                    token_id = jnp.min(jnp.where(values == jnp.max(values), pairs[1, :4, 1], c.vocab))
                    token_out[...] = token_id.astype(jnp.int32).reshape(1)
                    return
                pairs[0, rank, ...] = pairbuf[...]
                if "noxchg" not in ablate:
                    for offset in range(1, tp):
                        tpu.make_async_remote_copy(
                            pairbuf, pairs.at[0, rank], sends.at[offset - 1], recvs.at[offset - 1],
                            device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH,
                        ).start()
                    for offset in range(1, tp):
                        tpu.make_async_remote_copy(
                            pairbuf, pairs.at[0, rank], sends.at[offset - 1], recvs.at[offset - 1],
                            device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH,
                        ).wait()
                allvals = pairs[0, :, 0]
                topv = jnp.max(allvals)
                wiota = jax.lax.iota(jnp.int32, tp)
                winner = jnp.min(jnp.where(allvals == topv, wiota, tp)).astype(jnp.int32)
                token_id = winner * vshard + jnp.sum(jnp.where(wiota == winner, pairs[0, :, 1], 0.0))
                token_out[...] = token_id.astype(jnp.int32).reshape(1)

            tpu.make_async_copy(router_bias_r, biasbuf, eg_sem.at[0]).start()
            tpu.make_async_copy(expert_norms_r, xnormsbuf, eg_sem.at[1]).start()
            tpu.make_async_copy(router_bias_r, biasbuf, eg_sem.at[0]).wait()
            tpu.make_async_copy(expert_norms_r, xnormsbuf, eg_sem.at[1]).wait()
            token_out[...] = tokenr[...]

            if chunk_size:
                def initialize_history(index, _):
                    token_history[index] = jnp.int32(0)
                jax.lax.fori_loop(0, chunk_size, initialize_history, None)

                def step(index, _):
                    if teacher_forcing:
                        token_out[...] = jnp.full((1,), pr[2 + index], jnp.int32)
                    one_token(index, None)
                    token_history[index] = token_out[0]
                jax.lax.fori_loop(0, pr[1], step, None)
                tpu.make_async_copy(token_history, tokens_out, history_sem.at[0]).start()
                tpu.make_async_copy(token_history, tokens_out, history_sem.at[0]).wait()
            else:
                one_token(0, None)
                tokens_out[...] = token_out[...]

        shape = lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype)
        v = lambda shp, dtype=jnp.bfloat16: tpu.VMEM(shp, dtype)
        dm = lambda n: tpu.SemaphoreType.DMA((n,))
        states_in = (states["kcache"], states["vcache"])
        weights_in = (
            w["qwkv"],
            w["ow"],
            w["pool"],
            w["unpool"],
            w["lmw"],
            w["router"],
            w["scalars"],
            w["expert_norms"],
            w["router_bias"],
            w["gu_pack"],
            w["gu_global"],
            w["dw_pack"],
            w["dw_global"],
            w["gu_dense"],
            w["dw_dense"],
            w["fnorm"],
            w["emb"],
        )
        args = (token, *weights_in, *states_in)
        logits_shape = jax.ShapeDtypeStruct((1, lm_w), jnp.float32)
        scratch = (
            v((qwkv_tiles if shared_arenas else max(qwkv_tiles, lm_tiles), 32, 128), jnp.uint8) if alias_lm else v(qwkv_w, w["qwkv"].dtype),  # qwkv_slab
            v((projection_arena_tiles, 32, 128), jnp.uint8) if shared_arenas else v(ow_w, w["ow"].dtype),
            v((1, 128)) if shared_arenas else v(pool_w, w["pool"].dtype),
            v((1, 128)) if shared_arenas else v(unpool_w, w["unpool"].dtype),
            v((eshard, h) if ROUTER_TRANSPOSE else (h, eshard), jnp.float32),
            v((1, 128) if alias_lm and not shared_arenas else (2, h, lm_chunk_w), w["lmw"].dtype),  # lm_slab
            dm(8),  # dsem: 0=qwkv 1=ow 2=pool 3=unpool 4=router 5/6=lm chunks
            v((2, w["scalars"].shape[-2], h), jnp.float32),  # scalarsbuf
            dm(2),  # sc_sem
            v((c.layers, c.experts), jnp.float32),  # biasbuf
            v((c.layers, 2, mh), jnp.float32),  # xnormsbuf
            v((1, h), jnp.float32),  # residual
            v((1, h)),  # normed
            v((1, h), jnp.float32),  # proj
            v((8, max(h, 2 * nh * hd + 2 * nkv * hd)), jnp.float32),  # accum
            v((1, lm_w), jnp.float32),  # lm_accum
            v((1, 2 * nh * hd + 2 * nkv * hd)),  # qwkvbuf
            v((1, nh * hd)),  # attbuf
            v((1, pool_width)),  # poolbuf
            v((1, unpool_input)),  # unpinbuf
            v((1, mh)),  # pooledbuf
            v((c.top_k, mh), jnp.float32),  # rowsbuf
            v((1, 128 + pool_width), jnp.float32),  # combuf (router partial | pool shard)
            v((2, 8, 1, h)),  # recv (bf16, intra-host exchange)
            v((2, 4, 1, h)),  # recv_h (bf16, inter-host exchange)
            v((2, 8, 1, 128 + pool_width), jnp.float32),  # recv_comb (intra-host gather)
            v((2, 4, 8, 1, 128 + pool_width), jnp.float32),  # recv_comb_h (inter-host gather)
            v((2, 8, 1, mh)),  # recv_r1 (bf16, intra-host row partials)
            v((2, 4, 1, mh)),  # recv_r2 (bf16, inter-host row partials)
            v((1, mh)),  # rowstage (bf16)
            v((1, mh) if UNPOOL_COLUMNS else (tp, 1, mh_r), jnp.float32),  # nrow
            v((2, 8, 1, unpool_input), jnp.float32),  # recv_r3 (f32, weighted row slices)
            v((2, 8, c.top_k, 1, mh)),  # recv_rows (bf16, allreduce-path intra)
            v((2, 4, c.top_k, 1, mh)),  # recv_rows_h (bf16, allreduce-path inter)
            v((1, mh)),  # mixedbuf (allreduce path)
            v((1, h)),  # xstage_h
            v((c.top_k, 1, mh)),  # xstage_rows
            dm(tp - 1),  # sends
            dm(tp - 1),  # recvs
            v((nkv, 8, hdp)),  # qbg (rows 0:rpg even heads, rpg:2rpg odd heads)
            v((2, nkv, sbw, hdp)),  # kb (sbw-position super-blocks)
            v((2, nkv, sbw, hdp)),  # vb
            v((nkv, 8, 1), jnp.float32),  # amg
            v((nkv, 8, 1), jnp.float32),  # alg
            v((nkv, 8, hdp), jnp.float32),  # aag
            dm(2),  # ks
            dm(2),  # vs
            dm(2),  # io
            v((nh, hdp)),  # attended
            v((1, 128)) if shared_arenas else v((max(2 * gu_tiles, gu_dense_tiles), 32, 128), jnp.uint8) if EXPERT_ARENA else v(gu_shape, jnp.uint8),
            v((1, 128)) if EXPERT_ARENA else v(gu_shape, jnp.uint8),
            v((1, 128)) if shared_arenas else v((max(2 * dw_tiles, dw_dense_tiles), 32, 128), jnp.uint8) if EXPERT_ARENA else v(dw_shape, jnp.uint8),
            v((1, 128)) if EXPERT_ARENA else v(dw_shape, jnp.uint8),
            v((2, c.experts, 2), jnp.float32),  # ggu
            v((2, c.experts), jnp.float32),  # ggw
            dm(2),  # g_sem
            dm(max(2 * EXPERT_SLOTS, 4 if EP_SERIAL else 0)),  # eg_sem
            v((1, 128)) if EXPERT_ARENA else v(gu_dense_shape),
            v((1, 128)) if EXPERT_ARENA else v(dw_dense_shape),
            dm(4),  # ed_sem
            v((8, h)),  # rowbuf
            dm(1),  # e_sem
            v((2,), jnp.float32),  # pairbuf
            v((2, tp, 2), jnp.float32),  # pairs: separate local and inter-host receive slots
            v(w["fnorm"].shape, jnp.float32),  # fnormbuf
            v((2, 4, 1, unpool_input), jnp.float32),  # ep_recv
            v((ATTN_GROUP, nh * hd // 128, 128) if ATTN_INPUT_COMPACT else (ATTN_GROUP, 1, nh * hd)),
            v((ATTN_GROUP, h // ATTN_GROUP // 128, 128) if COMPACT_ATTN else (ATTN_GROUP, 1, h // ATTN_GROUP), jnp.float32),
            v((8, h // 1024, 128) if compact_unpool else (8, 1, h // 8), jnp.float32),
            v((2, mh // 128, 128)),
            v((2, 8, 2, mh // 128, 128)),
            v((max(unpool_output_width // 128, 1), 128), jnp.float32),
            v((2, 4, max(h // 1024, 1), 128), jnp.float32),
            v((h // ATTN_GROUP // 128, 128)),
            v((2, 8, h // ATTN_GROUP // 128, 128)),
            v((2, 4, h // ATTN_GROUP // 128, 128)),
            v((h // ATTN_GROUP // 128, 128), jnp.float32),
            v((2, tp, max(unpool_output_width // 128, 1), 128), jnp.bfloat16 if UNPOOL_DIRECT >= 2 else jnp.float32),
            v((max(unpool_output_width // 128, 1), 128), jnp.bfloat16),
            v((tp, h // ATTN_GROUP // 128, 128)),
            v((nh * hd // 128, 128)),
            tpu.SemaphoreType.DMA((3, direct_semaphore_width)),
            tpu.SemaphoreType.DMA((3, direct_semaphore_width)),
            tpu.SMEM((history_size,), jnp.int32),
            dm(1),
        )
        state_offset = 1 + 1 + len(weights_in)
        if aux_layers:
            scratch += (v((len(aux_layers), 1, h)), dm(1))
        result = pl.pallas_call(
            body,
            out_shape=(
                logits_shape,
                jax.ShapeDtypeStruct((1,), jnp.int32),
                jax.ShapeDtypeStruct((history_size,), jnp.int32),
                *(shape(s) for s in states_in),
                *((jax.ShapeDtypeStruct((history_size, len(aux_layers), 1, h), jnp.bfloat16),) if aux_layers else ()),
            ),
            input_output_aliases={state_offset + i: i + 3 for i in range(2)},
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[pl.BlockSpec()] + [pl.BlockSpec(memory_space=tpu.HBM)] * (len(args) - 1),
                out_specs=[pl.BlockSpec()] * 2 + [pl.BlockSpec(memory_space=tpu.HBM) if chunk_size else pl.BlockSpec()] + [pl.BlockSpec(memory_space=tpu.HBM)] * (2 + bool(aux_layers)),
                scratch_shapes=scratch,
            ),
            compiler_params=tpu.CompilerParams(
                collective_id=47,
                vmem_limit_bytes=64 * 1024**2,
                disable_bounds_checks=True,
                shape_invariant_numerics=True,
            ),
            name="mpk_persistent_stack_tp32" if chunk_size else "mpk_transformer_stack_tp32",
        )(control, *args)
        new_states = dict(zip(("kcache", "vcache"), result[3:5]))
        if aux_layers:
            return result[2], result[5].reshape(history_size, len(aux_layers), h), new_states
        if chunk_size:
            return result[0], result[1], result[2], new_states
        return result[0], result[1], new_states


    def logical_mesh32():
        """Flat 32-device "tp" mesh across 4 hosts (rank = host * 8 + lane)."""
        devices = sorted(jax.devices(), key=lambda d: (d.process_index, d.id))
        if len(devices) != 32:
            raise RuntimeError(f"Expected 32 TPU devices, found {len(devices)}")
        return jax.sharding.Mesh(np.asarray(devices), ("tp",))


    def make_decode(mesh, c, context, tp=32, ablate=()):
        """jit(shard_map(transformer_stack)); (weights, states, token, position)."""
        ablate = frozenset(ablate)
        if mesh.size != 32:
            raise ValueError(f"the TP32 kernel needs a 32-device mesh, got {mesh.size}")

        def local(global_weights, global_states, token, position):
            weights = jax.tree.map(lambda a: a[0], global_weights)
            states = jax.tree.map(lambda a: a[0], global_states)
            logits, nxt, new_states = transformer_stack(
                c, token, weights, states, position, context, ablate, tp
            )
            return nxt, logits, jax.tree.map(lambda a: a[None], new_states)

        def run(w, states, token, position):
            ws = _tp_specs(w, tp)
            ss = jax.tree.map(lambda _: P("tp"), states)
            return jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(ws, ss, P(), P()),
                out_specs=(P(), P("tp"), ss),
                check_vma=False,
            )(w, states, token, position)

        return jax.jit(run, donate_argnums=(1,))


    def make_decode_chunk(mesh, c, context, chunk_size, tp=32, ablate=()):
        """Return (next_token, states, history) for control=[position, step_count].

        Each step consumes the preceding greedy token. The caller must keep
        step_count within [0, chunk_size] and position + step_count within context.
        Unused history entries are zero.
        """
        if mesh.size != tp or tp != 32:
            raise ValueError("the TP32 kernel requires a 32-device mesh")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        ablate = frozenset(ablate)

        def local(global_weights, global_states, token, control):
            weights = jax.tree.map(lambda a: a[0], global_weights)
            states = jax.tree.map(lambda a: a[0], global_states)
            _, nxt, history, new_states = transformer_stack(
                c, token, weights, states, control, context, ablate, tp, chunk_size=chunk_size
            )
            return nxt, jax.tree.map(lambda a: a[None], new_states), history

        def run(weights, states, token, control):
            specs = jax.tree.map(lambda _: P("tp"), states)
            return jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(_tp_specs(weights, tp), specs, P(), P()),
                out_specs=(P(), specs, P()),
                check_vma=False,
            )(weights, states, token, control)

        return jax.jit(run, donate_argnums=(1,))


    def make_verify(mesh, c, context, rows, aux_layers):
        """Teacher-forced persistent verification with post-layer draft features."""
        aux_layers = tuple(aux_layers)
        if mesh.size != 32 or rows < 1 or not aux_layers:
            raise ValueError("verification requires 32 devices, positive rows, and auxiliary layers")
        if len(set(aux_layers)) != len(aux_layers) or any(i < 0 or i >= c.layers for i in aux_layers):
            raise ValueError("auxiliary layer indices must be unique target layers")

        def local(global_weights, global_states, tokens, control):
            weights = jax.tree.map(lambda a: a[0], global_weights)
            states = jax.tree.map(lambda a: a[0], global_states)
            targets, aux, new_states = transformer_stack(
                c, tokens, weights, states, control, context,
                chunk_size=rows, teacher_forcing=True, aux_layers=aux_layers,
            )
            return targets, aux, jax.tree.map(lambda a: a[None], new_states)

        def run(weights, states, tokens, control):
            specs = jax.tree.map(lambda _: P("tp"), states)
            return jax.shard_map(
                local, mesh=mesh,
                in_specs=(_tp_specs(weights, 32), specs, P(), P()),
                out_specs=(P(), P(), specs), check_vma=False,
            )(weights, states, tokens, control)

        return jax.jit(run, donate_argnums=(1,))


    _VERIFY_WEIGHTS = ("qwkv", "ow", "pool", "unpool", "router", "router_bias", "scalars", "expert_norms", "gu_pack", "dw_pack", "gu_dense", "dw_dense", "gu_global", "dw_global", "lmw", "fnorm")


    def _verify_router_order(experts):
        """Place ascending expert IDs in Mosaic's native argmax tie order."""
        columns = jnp.arange(experts, dtype=jnp.int32)
        if experts not in (128, 256):
            return columns
        return (127 - columns % 128) * (experts // 128) + columns // 128


    def _verify_router_argmax(ranked):
        experts = ranked.shape[-1]
        if experts in (128, 256):
            chosen = jnp.argmax(ranked, axis=1).astype(jnp.int32)[:, None]
            return (127 - chosen % 128) * (experts // 128) + chosen // 128
        maximum = jnp.max(ranked, axis=1, keepdims=True)
        return jnp.min(jnp.where(ranked == maximum, jnp.arange(experts)[None], experts), axis=1, keepdims=True)


    def make_verify_block(mesh, c, context, rows, aux_layers):
        """Verify eight positions with TP8 experts and a combined BF16 KV cache."""
        if mesh.size != 32 or rows != 8 or c.head_dim != 64:
            raise ValueError("persistent verification requires TP32 and eight rows with 64-wide heads")
        if not aux_layers or len(set(aux_layers)) != len(aux_layers) or any(i < 0 or i >= c.layers for i in aux_layers):
            raise ValueError("auxiliary layer indices must be unique target layers")
        h, mh, nh = c.dim, c.moe_hidden, c.heads // 16
        iw, di = c.moe_intermediate // 8, c.moe_intermediate // 8
        qwidth, pw, es = nh * 64, mh // 8, c.experts // 8
        router_rows, pool_rows = 2 * rows, rows * pw // 128
        qshape, oshape = (h, 2 * qwidth + 128), (8 * qwidth, h // 8)
        pshape, ushape = (h, pw), (mh, h // 8)
        gpshape, dpshape = (4, mh * 9 // 16, 2 * iw), (4, iw * 9 // 16, mh)
        gdshape, ddshape = (4, mh, 2 * di), (4, di, mh)

        def tiles(shape, dtype):
            return pool_alias.storage_size(shape, jnp.dtype(dtype)) // 4096

        gt = max(tiles(gpshape, jnp.uint8), tiles(gdshape, jnp.bfloat16))
        dt = max(tiles(dpshape, jnp.uint8), tiles(ddshape, jnp.bfloat16))
        head_width = lm_width(c, 32)
        lm_chunk = min(1280, head_width // 2)
        lm_shape = (2, h, lm_chunk)
        q_tiles = tiles(qshape, jnp.bfloat16)
        cache_tiles = tiles((context, 128), jnp.bfloat16)
        qt = max(q_tiles + cache_tiles, gt + dt, tiles(lm_shape, jnp.bfloat16))
        pt = max(tiles(shape, jnp.bfloat16) for shape in (oshape, pshape, ushape))

        def kernel(meta, *args):
            w = dict(zip(_VERIFY_WEIGHTS, args[:len(_VERIFY_WEIGHTS)]))
            (initial, kvc_in, final, aux, kvc, qa, pa, router, scalars, norms, bias,
             gg, dg, pair, ahost, apaired, comb, block_values, estage, epipe, pipe_rows, mixed_rows, unhost,
             ungather, ids_v, ids_s, candidates, sem, sends, recvs) = args[len(_VERIFY_WEIGHTS):]
            qbuf = pool_alias.view(qa.at[:q_tiles], qshape, jnp.bfloat16)
            kvbuf = pool_alias.view(qa.at[q_tiles:q_tiles + cache_tiles], (context, 128), jnp.bfloat16)
            gp = pool_alias.view(qa.at[:tiles(gpshape, jnp.uint8)], gpshape, jnp.uint8)
            dp = pool_alias.view(qa.at[gt:gt + tiles(dpshape, jnp.uint8)], dpshape, jnp.uint8)
            gd = pool_alias.view(qa.at[:tiles(gdshape, jnp.bfloat16)], gdshape, jnp.bfloat16)
            dd = pool_alias.view(qa.at[gt:gt + tiles(ddshape, jnp.bfloat16)], ddshape, jnp.bfloat16)
            obuf = pool_alias.view(pa.at[:tiles(oshape, jnp.bfloat16)], oshape, jnp.bfloat16)
            pbuf = pool_alias.view(pa.at[:tiles(pshape, jnp.bfloat16)], pshape, jnp.bfloat16)
            ubuf = pool_alias.view(pa.at[:tiles(ushape, jnp.bfloat16)], ushape, jnp.bfloat16)
            lm = pool_alias.view(qa.at[:tiles(lm_shape, jnp.bfloat16)], lm_shape, jnp.bfloat16)
            rank = jax.lax.axis_index("tp")
            lane, host = rank % 8, rank // 8
            owner = lane % rows
            r = jnp.arange(rows, dtype=jnp.int32)[:, None]
            pos = meta[0] + r
            write_start = jnp.minimum(meta[0] // 16 * 16, context - 32)
            wr = write_start + jnp.arange(32)[:, None]
            cache_positions = jnp.arange(context)[None]
            columns = jnp.arange(128, dtype=jnp.int32)[None].astype(jnp.float32)
            angle = pos.astype(jnp.float32) / c.rope_theta ** ((columns % 32) / 32)
            cosine, sine = jnp.cos(angle), jnp.sin(angle)
            experts = jnp.arange(c.experts)[None]
            router_ids = _verify_router_order(c.experts)[None]

            def copy(src, dst, channel):
                return tpu.make_async_copy(src, dst, sem.at[channel])

            def remote(src, dst, channel, peer):
                return tpu.make_async_remote_copy(src, dst, sends.at[channel], recvs.at[channel], device_id=(peer,), device_id_type=pl.DeviceIdType.MESH)

            def gather(buf, index, offsets, channel):
                transfers = [remote(buf.at[index], buf.at[index], channel + i, rank ^ offset) for i, offset in enumerate(offsets)]
                for transfer in transfers:
                    transfer.start()
                for transfer in transfers:
                    transfer.wait()

            def dot(x, y):
                return jnp.dot(x, y, preferred_element_type=jnp.float32)

            def norm(x, eps, gain=None):
                x = x.astype(jnp.float32)
                x *= jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
                return x if gain is None else x * gain

            def hn(x):
                x = x.astype(jnp.float32)
                square = (x * x).astype(jnp.bfloat16).astype(jnp.float32)
                return x * jax.lax.rsqrt(jnp.sum(square, axis=-1, keepdims=True) / 64 + c.eps)

            def rotate(x, sliding):
                x = x.astype(jnp.float32)
                shifted = jnp.concatenate((-x[:, 32:64], x[:, :32], jnp.zeros((rows, 64), jnp.float32)), axis=1)
                return jnp.where(sliding, x * cosine + shifted * sine, x).astype(jnp.bfloat16)

            barrier = tpu.get_barrier_semaphore()
            for offset in range(1, 32):
                pl.semaphore_signal(barrier, 1, device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH)
            pl.semaphore_wait(barrier, 31)
            aux[...] = jnp.zeros(aux.shape, jnp.bfloat16)
            copy(w["qwkv"].at[0], qbuf, 0).start()
            copy(w["ow"].at[0], obuf, 1).start()
            copy(w["scalars"].at[0], scalars, 3).start()

            def layer(li, stream):
                dense = jnp.bool_(False)
                below = jnp.int32(0)
                for d in c.bf16_expert_layers:
                    dense |= li == d
                    below += (li > d).astype(jnp.int32)
                pi = jnp.clip(li - below, 0, c.layers - len(c.bf16_expert_layers) - 1)
                bi = jnp.clip(below, 0, len(c.bf16_expert_layers) - 1)
                sliding = li % 4 != 1
                loads = [(w["router"].at[li], router),
                         (w["scalars"].at[li], scalars), (w["expert_norms"].at[li], norms), (w["router_bias"].at[li], bias),
                         (w["gu_global"].at[pi], gg), (w["dw_global"].at[pi], dg),
                         (kvc.at[li, 0], kvbuf)]
                for channel, (src, dst) in enumerate(loads, 2):
                    if channel != 3:
                        copy(src, dst, channel).start()
                copy(w["scalars"].at[li], scalars, 3).wait()
                sc = scalars[...]
                x = norm(stream, c.eps, sc[0:1]).astype(jnp.bfloat16)
                copy(w["qwkv"].at[li], qbuf, 0).wait()
                qgkv = dot(x, qbuf[...]).astype(jnp.bfloat16)
                k = rotate(hn(jnp.pad(qgkv[:, 2 * qwidth:2 * qwidth + 64], ((0, 0), (0, 64)))).astype(jnp.bfloat16), sliding)
                v = jnp.pad(qgkv[:, 2 * qwidth + 64:], ((0, 0), (0, 64)))
                copy(kvc.at[li, 0], kvbuf, 8).wait()
                block = kvbuf[pl.ds(write_start, 32)]
                values = jnp.concatenate((k[:, :64], v[:, :64]), axis=1)
                for row in range(rows):
                    mask = jnp.broadcast_to(wr == meta[0] + row, (32, 128))
                    block = jnp.where(mask, values[row:row + 1], block)
                kvbuf[pl.ds(write_start, 32)] = block
                copy(kvbuf.at[pl.ds(write_start, 32)], kvc.at[li, 0, pl.ds(write_start, 32)], 10).start()
                keep = (cache_positions <= pos) & ((cache_positions > pos - c.sliding_window) | ~sliding)
                queries = []
                for head in range(nh):
                    q = rotate(hn(jnp.pad(qgkv[:, head * 64:(head + 1) * 64], ((0, 0), (0, 64)))).astype(jnp.bfloat16), sliding)
                    queries.append(q)
                query = jnp.concatenate(queries, axis=0)

                def attention(width):
                    scores = jnp.where(jnp.tile(keep[:, :width], (nh, 1)), dot(query, kvbuf[:width].T) * c.scale, -1e30)
                    exponent = jnp.exp(scores - jnp.max(scores, axis=1, keepdims=True)).astype(jnp.bfloat16)
                    value = (dot(exponent, kvbuf[:width]) / jnp.sum(exponent.astype(jnp.float32), axis=1, keepdims=True)).astype(jnp.bfloat16)
                    return jnp.pad(value[:, 64:], ((0, 0), (0, 64)))

                width = min(1024, context)
                attended = jax.lax.cond(meta[0] + rows <= width, lambda: attention(width), lambda: attention(context))
                attended = hn(attended)
                parts = []
                for head in range(nh):
                    gate = qgkv[:, qwidth + head * 64:qwidth + (head + 1) * 64].astype(jnp.float32)
                    parts.append((attended[head * rows:(head + 1) * rows, :64] * jax.nn.sigmoid(gate)).astype(jnp.bfloat16))
                pair[lane] = jnp.concatenate(parts, axis=1)
                gather(pair, lane, tuple(range(7, 0, -1)), 0)
                joined = jnp.concatenate([pair[i] for i in range(8)], axis=1)
                copy(w["ow"].at[li], obuf, 1).wait()
                ahost[host % 2] = sum(
                    dot(joined[:, i * 2 * qwidth:(i + 1) * 2 * qwidth],
                        obuf[i * 2 * qwidth:(i + 1) * 2 * qwidth])
                    .astype(jnp.bfloat16).astype(jnp.float32)
                    for i in range(4)
                )
                copy(w["pool"].at[li], pbuf, 1).start()
                gather(ahost, host % 2, (8,), 8)
                apaired[lane] = jnp.sum(ahost[...], axis=0)
                gather(apaired, lane, tuple(range(7, 0, -1)), 1)
                branch = jnp.concatenate([apaired[i] for i in range(8)], axis=1)
                stream = sc[3:4] * stream + sc[4:5] * norm(branch, c.post_eps)
                x = norm(stream, c.eps, sc[1:2]).astype(jnp.bfloat16)
                copy(w["router"].at[li], router, 2).wait()
                logits = jnp.concatenate([jnp.sum(x[i:i + 1].astype(jnp.float32) * router[...], axis=1)[None] for i in range(rows)], axis=0)
                copy(w["pool"].at[li], pbuf, 1).wait()
                pooled = dot(x, pbuf[...]).astype(jnp.bfloat16)
                comb[lane] = jnp.concatenate((tpu.bitcast(jnp.pad(logits, ((0, 0), (0, 128 - es))), jnp.bfloat16), *[pooled[:, i * 128:(i + 1) * 128] for i in range(pw // 128)]), axis=0)
                copy(w["unpool"].at[li], ubuf, 1).start()
                gather(comb, lane, tuple(range(7, 0, -1)), 9)
                scores = jax.nn.sigmoid(jnp.concatenate([tpu.bitcast(comb[i, :router_rows], jnp.float32)[:, :es] for i in range(8)], axis=1) * c.output_multiplier)
                copy(w["router_bias"].at[li], bias, 5).wait()
                ranked = scores + bias[...][None]
                route_ids, route_weights = [], []
                requested = jnp.zeros((1, c.experts), jnp.float32)
                for _ in range(c.top_k):
                    index = _verify_router_argmax(ranked)
                    selected = experts == index
                    route_ids.append(index)
                    route_weights.append(jnp.sum(jnp.where(router_ids == index, scores, 0), axis=1, keepdims=True))
                    requested += jnp.sum(selected.astype(jnp.float32), axis=0, keepdims=True)
                    ranked = jnp.where(router_ids == index, -jnp.inf, ranked)
                rw = jnp.concatenate(route_weights, axis=1)
                rw /= jnp.sum(rw, axis=1, keepdims=True) + c.router_eps
                requested = (requested > 0).astype(jnp.bfloat16)
                triangular = (jnp.arange(c.experts)[:, None] <= experts).astype(jnp.bfloat16)
                prefix = dot(jnp.broadcast_to(requested, (8, c.experts)), triangular)[:1].astype(jnp.int32)
                active = ((requested > 0) & ((prefix - 1) % 4 == host)).astype(jnp.int32)
                ordinal = (prefix - 1) // 4
                ids_v[:128] = jnp.min(jnp.where((active > 0) & (ordinal == jnp.arange(128)[:, None]), experts, c.experts), axis=1)
                route_slots = jnp.full((1, c.experts), c.top_k, jnp.int32)
                for k in range(c.top_k):
                    selected = jnp.sum(jnp.where(r == owner, (experts == route_ids[k]).astype(jnp.int32), 0), axis=0, keepdims=True) > 0
                    route_slots = jnp.where(selected, k, route_slots)
                ids_v[128:128 + c.experts] = route_slots[0]
                copy(ids_v, ids_s, 20).start()
                copy(w["expert_norms"].at[li], norms, 4).wait()
                pooled = norm(jnp.concatenate([comb[i, router_rows + j * rows:router_rows + (j + 1) * rows] for i in range(8) for j in range(pw // 128)], axis=1), c.eps, norms[0:1]).astype(jnp.bfloat16)
                copy(w["gu_global"].at[pi], gg, 6).wait()
                copy(w["dw_global"].at[pi], dg, 7).wait()

                copy(kvbuf.at[pl.ds(write_start, 32)], kvc.at[li, 0, pl.ds(write_start, 32)], 10).wait()

                def run_experts(packed):
                    gu, dw, gb, db, index, width = (w["gu_pack"], w["dw_pack"], gp, dp, pi, iw) if packed else (w["gu_dense"], w["dw_dense"], gd, dd, bi, di)

                    def issue(e, slot):
                        copy(gu.at[index, e], gb.at[slot], 12 + 2 * slot).start()
                        copy(dw.at[index, e], db.at[slot], 13 + 2 * slot).start()

                    def dequant(blob, k, n):
                        values = tpu.bitcast(blob[:k // 2], jnp.float4_e2m1fn).astype(jnp.bfloat16)
                        scales = jax.lax.bitcast_convert_type(blob[k // 2:], jnp.float8_e4m3fn).astype(jnp.bfloat16)
                        return (values.reshape(k // 16, 16, n) * scales[:, None]).reshape(k, n)

                    count = jnp.sum(active.astype(jnp.int32))
                    copy(ids_v, ids_s, 20).wait()
                    prefetch = 4
                    for first_id in range(prefetch):
                        @pl.when(count > first_id)
                        def first():
                            issue(ids_s[first_id], jnp.int32(first_id))

                    def row_transfers(group):
                        # Four packets prevent reuse before every peer consumes a row.
                        slot = group % 4
                        return [remote(estage.at[slot, lane ^ offset], epipe.at[slot, lane], 16 + slot * 7 + offset - 1, rank ^ offset) for offset in range(7, 0, -1)]

                    def finish_rows(group):
                        for transfer in row_transfers(group):
                            transfer.wait()
                        full = jnp.sum(epipe[group % 4].astype(jnp.float32), axis=0).reshape(2, mh)
                        for part in range(2):
                            @pl.when(group * 2 + part < count)
                            def store_row():
                                pipe_rows[ids_s[128 + ids_s[group * 2 + part]]] = full[part:part + 1]

                    def step(i, unused):
                        e, slot = ids_s[i], i % prefetch
                        copy(gu.at[index, e], gb.at[slot], 12 + 2 * slot).wait()
                        copy(dw.at[index, e], db.at[slot], 13 + 2 * slot).wait()
                        down_weights = dequant(db[slot], width, mh) if packed else db[slot]
                        projected = dot(pooled, dequant(gb[slot], mh, 2 * width) if packed else gb[slot])
                        if packed:
                            gate_scale = jax.lax.bitcast_convert_type(jnp.full((1, 1), gg[2 * e], jnp.int32), jnp.float32)
                            up_scale = jax.lax.bitcast_convert_type(jnp.full((1, 1), gg[2 * e + 1], jnp.int32), jnp.float32)
                            gate = (projected[:, :width] * gate_scale).astype(jnp.bfloat16).astype(jnp.float32)
                            up = (projected[:, width:] * up_scale).astype(jnp.bfloat16).astype(jnp.float32)
                        else:
                            gate, up = projected[:, :width], projected[:, width:]
                        act = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
                        value = dot(act, down_weights)
                        if packed:
                            value *= jax.lax.bitcast_convert_type(jnp.full((1, 1), dg[e], jnp.int32), jnp.float32)
                        block_values[i % 2] = value.astype(jnp.bfloat16)
                        @pl.when((i % 2 == 1) | (i + 1 == count))
                        def send_pair():
                            group = i // 2
                            estage[group % 4] = block_values[...].transpose(1, 0, 2).reshape(rows, mh // 128, 256)
                            epipe[group % 4, lane] = estage[group % 4, owner]
                            for transfer in row_transfers(group):
                                transfer.start()
                            @pl.when(group > 0)
                            def finish_previous():
                                finish_rows(group - 1)
                        @pl.when(i + prefetch < count)
                        def following():
                            issue(ids_s[i + prefetch], slot)
                        return unused

                    pipe_rows[...] = jnp.zeros(pipe_rows.shape, jnp.float32)
                    block_values[...] = jnp.zeros(block_values.shape, jnp.bfloat16)
                    jax.lax.fori_loop(0, count, step, None)
                    @pl.when(count > 0)
                    def finish_last():
                        finish_rows((count - 1) // 2)

                jax.lax.cond(dense, lambda: run_experts(False), lambda: run_experts(True))
                @pl.when(li + 1 < c.layers)
                def prefetch_q():
                    copy(w["qwkv"].at[li + 1], qbuf, 0).start()
                    copy(w["scalars"].at[li + 1], scalars, 3).start()
                @pl.when(li + 1 == c.layers)
                def prefetch_head():
                    copy(w["fnorm"].at[0], scalars.at[7], 3).start()
                    for slot in range(2):
                        copy(w["lmw"].at[:, pl.ds(slot * lm_chunk, lm_chunk)], lm.at[slot], slot * 16).start()
                full = pipe_rows[:c.top_k, 0]
                mixed = jnp.zeros((1, mh), jnp.float32)
                for k in range(c.top_k):
                    weight = jnp.sum(jnp.where(r == owner, rw[:, k:k + 1], 0), axis=0, keepdims=True)
                    mixed += norm(full[k:k + 1] * norms[1:2], c.post_eps) * weight
                mixed_slice = pl.ds(pl.multiple_of(owner * (mh // 128), 8), mh // 128)
                mixed_rows[mixed_slice] = mixed.astype(jnp.bfloat16).reshape(mh // 128, 128)
                transfers = [remote(mixed_rows.at[mixed_slice], mixed_rows.at[mixed_slice], 16 + offset - 1, rank ^ offset) for offset in range(rows - 1, 0, -1)]
                for transfer in transfers:
                    transfer.start()
                for transfer in transfers:
                    transfer.wait()
                copy(w["unpool"].at[li], ubuf, 1).wait()
                projected = dot(mixed_rows[...].reshape(rows, mh), ubuf[...]).astype(jnp.bfloat16)
                unhost[host] = projected
                @pl.when(li + 1 < c.layers)
                def prefetch_o():
                    copy(w["ow"].at[li + 1], obuf, 1).start()
                gather(unhost, host, (8, 16, 24), 26)
                ungather[lane] = jnp.sum(unhost[...].astype(jnp.float32), axis=0)
                gather(ungather, lane, tuple(range(7, 0, -1)), 29)
                branch = jnp.concatenate([ungather[i] for i in range(8)], axis=1)
                stream = sc[5:6] * stream + sc[6:7] * norm(branch * sc[2:3], c.post_eps)
                for i, layer_id in enumerate(aux_layers):
                    @pl.when(li == layer_id)
                    def capture():
                        aux[i] = stream.astype(jnp.bfloat16)
                return stream

            stream = jax.lax.fori_loop(0, c.layers, layer, initial[...])
            copy(w["fnorm"].at[0], scalars.at[7], 3).wait()
            hidden = (norm(stream, c.eps) * scalars[7:8]).astype(jnp.bfloat16)

            def head_chunk(index, carry):
                best, best_index = carry
                slot = index % 2
                offset = index * lm_chunk
                copy(w["lmw"].at[:, pl.ds(offset, lm_chunk)], lm.at[slot], slot * 16).wait()
                logits = dot(hidden, lm[slot])
                columns = offset + jnp.arange(lm_chunk)[None]
                logits = jnp.where(columns < c.vocab // 32, logits, -jnp.inf)
                local_best = jnp.max(logits, axis=1, keepdims=True)
                local_index = jnp.min(jnp.where(logits == local_best, columns + rank * (c.vocab // 32), c.vocab), axis=1, keepdims=True)
                best_index = jnp.where(local_best > best, local_index, jnp.where(local_best == best, jnp.minimum(best_index, local_index), best_index))
                best = jnp.maximum(best, local_best)
                @pl.when(index + 2 < head_width // lm_chunk)
                def prefetch():
                    copy(w["lmw"].at[:, pl.ds((index + 2) * lm_chunk, lm_chunk)], lm.at[slot], slot * 16).start()
                return best, best_index

            best, best_index = jax.lax.fori_loop(0, head_width // lm_chunk, head_chunk,
                (jnp.full((rows, 1), -jnp.inf, jnp.float32), jnp.full((rows, 1), c.vocab, jnp.int32)))

            def pack(best, index):
                return jnp.concatenate((jnp.broadcast_to(best, (rows, 64)), jnp.broadcast_to(index.astype(jnp.float32), (rows, 64))), axis=1)

            def winner(start, count):
                best = jnp.max(candidates[start:start + count, :, :1], axis=0)
                index = jnp.min(jnp.where(candidates[start:start + count, :, :1] == best[None], candidates[start:start + count, :, 64:65], c.vocab), axis=0)
                return best, index

            candidates[lane] = pack(best, best_index)
            gather(candidates, lane, tuple(range(7, 0, -1)), 0)
            best, best_index = winner(0, 8)
            candidates[8 + host] = pack(best, best_index)
            gather(candidates, 8 + host, (8, 16, 24), 8)
            best, best_index = winner(8, 4)
            final[...] = jnp.broadcast_to(best_index.astype(jnp.int32), (rows, 128))

        def local(global_weights, states, tokens, position):
            w = jax.tree.map(lambda a: a[0], global_weights)
            w["ow"] = w["ow_group8"]
            w["gu_dense"] = w["gu_dense_ep"]
            w["dw_dense"] = w["dw_dense_ep"]
            w["router"] = w["router_ordered"]
            w["router_bias"] = w["router_bias_ordered"]
            w["gu_global"] = jax.lax.bitcast_convert_type(w["gu_global"], jnp.int32).reshape(-1, 2 * c.experts)
            w["dw_global"] = jax.lax.bitcast_convert_type(w["dw_global"], jnp.int32)
            kvc = states["kvcache"][0]
            stream = rms_plain_f32(w["emb"][tokens], c.eps)
            v = lambda shape, dtype=jnp.bfloat16: tpu.VMEM(shape, dtype)
            shapes = (jax.ShapeDtypeStruct((rows, 128), jnp.int32), jax.ShapeDtypeStruct((len(aux_layers), rows, h), jnp.bfloat16), jax.ShapeDtypeStruct(kvc.shape, kvc.dtype))
            n = len(_VERIFY_WEIGHTS)
            stream, aux, kvc = pl.pallas_call(
                kernel, out_shape=shapes,
                grid_spec=tpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=1,
                    in_specs=[pl.BlockSpec(memory_space=tpu.HBM)] * n + [pl.BlockSpec(), pl.BlockSpec(memory_space=tpu.HBM)],
                    out_specs=[pl.BlockSpec(), pl.BlockSpec(), pl.BlockSpec(memory_space=tpu.HBM)],
                    scratch_shapes=(v((qt, 32, 128), jnp.uint8), v((pt, 32, 128), jnp.uint8),
                        v((es, h), jnp.float32), v((8, h), jnp.float32), v((2, mh), jnp.float32), v((c.experts,), jnp.float32),
                        tpu.SMEM((2 * c.experts,), jnp.int32), tpu.SMEM((c.experts,), jnp.int32),
                        v((8, rows, qwidth)), v((2, rows, h // 8), jnp.float32), v((8, rows, h // 8), jnp.float32), v((8, router_rows + pool_rows, 128)),
                        v((2, rows, mh)), v((4, rows, mh // 128, 256)), v((4, 8, mh // 128, 256)), v(((c.top_k + 8) // 8 * 8, 1, mh), jnp.float32), v((rows * mh // 128, 128)),
                        v((4, rows, h // 8)), v((8, rows, h // 8), jnp.float32),
                        v((128 + c.experts,), jnp.int32), tpu.SMEM((128 + c.experts,), jnp.int32),
                        v((12, rows, 128), jnp.float32), tpu.SemaphoreType.DMA((21,)), tpu.SemaphoreType.DMA((44,)), tpu.SemaphoreType.DMA((44,))),
                ),
                input_output_aliases={n + 2: 2},
                compiler_params=tpu.CompilerParams(collective_id=51, vmem_limit_bytes=64 * 1024**2, shape_invariant_numerics=True),
                name="mpk_dspark_persistent_verify",
            )(jnp.array([position], jnp.int32), *[w[key] for key in _VERIFY_WEIGHTS], stream, kvc)
            targets = stream[:, 0]
            return targets, aux.transpose(1, 0, 2), {"kvcache": kvc[None]}

        return jax.jit(jax.shard_map(local, mesh=mesh, in_specs=(P("tp"), P("tp"), P(), P()), out_specs=(P(), P(), P("tp")), check_vma=False), donate_argnums=(1,))


    def _prepare_dense_verify_weights(mesh, weights):
        if "gu_dense_ep" in weights:
            return weights
        sharding = NamedSharding(mesh, P("tp"))
        def regroup(value, contraction):
            _, layers, experts, rows, cols = value.shape
            value = value.reshape(8, 4, layers, experts, rows, cols)
            if contraction:
                value = value.transpose(0, 2, 3, 1, 4, 5).reshape(8, layers, experts, 4 * rows, cols)
            else:
                value = value.transpose(0, 2, 3, 4, 1, 5).reshape(8, layers, experts, rows, 4 * cols)
            return jnp.tile(value, (4, 1, 1, 1, 1))
        @jax.jit(out_shardings=sharding)
        def gu(value):
            gate, up = jnp.split(value, 2, axis=-1)
            return jnp.concatenate((regroup(gate, False), regroup(up, False)), axis=-1)
        @jax.jit(out_shardings=sharding)
        def dw(value):
            return regroup(value, True)
        return {**weights, "gu_dense_ep": gu(weights["gu_dense"]), "dw_dense_ep": dw(weights["dw_dense"])}


    def prepare_verify_weights(mesh, weights):
        weights = _prepare_dense_verify_weights(mesh, weights)
        @jax.jit(out_shardings=NamedSharding(mesh, P("tp")))
        def convert(value):
            _, layers, width, half_hidden = value.shape
            value = value.reshape(4, 4, 2, layers, width, half_hidden)
            value = value.transpose(0, 3, 1, 4, 2, 5).reshape(4, layers, 4 * width, 2 * half_hidden)
            value = value.reshape(4, layers, 4 * width, 8, half_hidden // 4)
            return value.transpose(0, 3, 1, 2, 4).reshape(32, layers, 4 * width, half_hidden // 4)

        @jax.jit(out_shardings=(NamedSharding(mesh, P("tp")), NamedSharding(mesh, P("tp"))))
        def reorder_router(router, bias):
            _, layers, per_lane, hidden = router.shape
            experts = 8 * per_lane
            if experts not in (128, 256):
                return router, bias
            order = _verify_router_order(experts)
            value = router.reshape(4, 8, layers, per_lane, hidden).transpose(0, 2, 1, 3, 4)
            value = value.reshape(4, layers, experts, hidden)[:, :, order, :]
            value = value.reshape(4, layers, 8, per_lane, hidden).transpose(0, 2, 1, 3, 4)
            return value.reshape(router.shape), bias[:, :, order]

        router, bias = jax.block_until_ready(reorder_router(weights["router"], weights["router_bias"]))
        return {**weights, "ow_group8": convert(weights["ow"]).block_until_ready(), "router_ordered": router, "router_bias_ordered": bias}


    @jax.jit
    def prepare_verify_cache(states):
        return {"kvcache": jnp.concatenate((states["kcache"][..., :64], states["vcache"][..., :64]), axis=-1)}

    return SimpleNamespace(
        CACHE_KEYS=CACHE_KEYS,
        REPLICATED_EXPERTS=REPLICATED_EXPERTS,
        EXPERT_TP=EXPERT_TP,
        ATTENTION_TP=ATTENTION_TP,
        logical_mesh32=logical_mesh32,
        prepare_decode_weights=prepare_decode_weights,
        make_prefill=make_prefill,
        make_decode=make_decode,
        make_decode_chunk=make_decode_chunk,
        make_verify=make_verify,
        make_verify_block=make_verify_block,
        prepare_verify_weights=prepare_verify_weights,
        prepare_verify_cache=prepare_verify_cache,
    )
