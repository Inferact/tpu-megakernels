"""Muse-Spark TP8/TP16 prefill and fused decode (Pallas)."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pool_alias
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import PartitionSpec as P

from . import F32, NVFP4_LUT, residual_coefficients  # noqa: F401  (re-exported for tests)
from .load import lm_width, tile_hw


BANK_COUNT = int(os.environ.get("MPK_NB", "13"))
MAX_BANK_K = int(os.environ.get("MPK_MAXK", "1024"))
MAX_BANK_N = int(os.environ.get("MPK_MAXN", "1280"))
EXPERT_SLOTS = int(os.environ.get("MPK_EXPERT_SLOTS", "2"))


def get_tp32_backend():
    value = os.environ.get("MPK_TP32_EXPERT_PARALLEL", "0")
    if value == "0":
        if os.environ.get("MPK_EXPERT_TP", "8") != "8":
            raise ValueError("The replicated TP32 backend requires MPK_EXPERT_TP=8")
        from . import decode_megakernel_32
        return decode_megakernel_32._replicated_tp8_backend()
    if value == "1":
        from . import decode_megakernel_32
        return decode_megakernel_32
    raise ValueError("MPK_TP32_EXPERT_PARALLEL must be 0 or 1")


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


def _dequant_experts_tiled(packed, scales, globals_, count, k, n):
    """Expand packed experts in VMEM with the original FP32 scale products."""
    bk, bn = min(k, 4096), min(n, 512)
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


def make_prefill(mesh, c, context, tp, *, aux_layers=()):
    """TP-sharded prefill over packed (untiled) weights.

    Prompt rows are LEFT-padded to the bucket length; each row's KV is written
    to absolute slot (row - npad) with out-of-range writes dropped, so caches
    come back in the compacted [0, L_real) layout the decode kernel expects.
    Returns per-rank logits shards for the last real position.
    """
    if tp == 16:
        return _make_prefill_persistent(mesh, c, context, tp, aux_layers=aux_layers)
    qh = c.heads // tp
    kvh = c.kv_heads // tp
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
            stream, kc_all, vc_all = carry
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
                w13 = _dequant_experts(
                    w["gu_packed"][o_packed], w["gu_scales"][o_packed], w["gu_global"][o_packed],
                    c.experts, c.moe_hidden, 2 * (c.moe_intermediate // tp),
                )
                w2 = _dequant_experts(
                    w["dw_packed"][o_packed], w["dw_scales"][o_packed], w["dw_global"][o_packed],
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
            return (stream, kc_all, vc_all), None

        (stream, kc_all, vc_all), _ = jax.lax.scan(
            one_layer, (stream, states["kcache"], states["vcache"]), jnp.arange(c.layers)
        )
        last = stream[L - 1]
        hidden = rms_gain_shard(last[None], w["fnorm"][0], c.eps)
        logits = jnp.dot(hidden, w["lmw"], preferred_element_type=F32)  # [1, lm_width]
        logits = logits * c.output_multiplier
        logits = c.soft_cap * jnp.tanh(logits / c.soft_cap)
        new_states = {"kcache": kc_all, "vcache": vc_all}
        return jax.tree.map(lambda a: a[None], new_states), logits

    def prefill(w, states, tokens, npad):
        if tp == 8 and "kvcache" in states:
            return _make_prefill_persistent(mesh, c, context, tp, aux_layers=aux_layers)(w, states, tokens, npad)
        ws = _tp_specs(w, tp)
        ss = jax.tree.map(lambda _: P("tp"), states)
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(ws, ss, P(), P()),
            out_specs=(ss, P("tp")),
            check_vma=False,
        )(w, states, tokens, npad)

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


def transformer_stack(c, token, w, states, position, context, ablate=frozenset(), tp=8):
    """One fused decode step; returns (logits [1, lm_width] f32, next_token, states).

    Dense weights stream HBM->VMEM through a ring of bank tiles shared by all
    families; the 8 routed experts per layer stream through a separate ring
    with in-VMEM NVFP4 dequantization. TP collectives use XOR-butterfly remote
    copies (single host).
    """
    if context % 128 or context < 256:
        raise ValueError("context must be a multiple of 128 and at least 256")
    h = c.dim
    nh = c.heads // tp
    nkv = c.kv_heads // tp
    hd = c.head_dim
    hdp = 128  # padded cache head dim
    mh = c.moe_hidden
    mh_r = mh // tp
    inter_r = c.moe_intermediate // tp
    eshard = c.experts // tp
    lm_w = lm_width(c, tp)
    vshard = c.vocab // tp
    # Dense expert tiles overlay the (idle) packed slot buffers when dims allow;
    # else they get their own VMEM buffers (MINI test configs).
    dense_overlay = mh % 2048 == 0 and inter_r % 128 == 0
    # Flash attention cache-block width (positions per read/compute step). 512-wide
    # super-blocks measured faster but shifted gsm8k strict-match to 0.9500; the
    # 128-block schedule scored 0.9515 on both metrics, so we stay there.
    sbw = min(128, context)
    groups = nh // nkv
    if nh % nkv or groups > 8 or groups % 2 or nkv % 2:
        raise ValueError("GQA groups must be even pairs of heads fitting one 8-row MXU tile")
    if c.experts % tp or c.moe_hidden % tp or c.moe_intermediate % tp:
        raise ValueError("MoE dims must split evenly across ranks")
    len_packed = c.layers - len(c.bf16_expert_layers)

    def slot(k, n):
        bk, bn = tile_hw(k, n)
        return bk, bn, k // bk, n // bn, (k // bk) * (n // bn)

    qwkv_s = slot(h, 2 * nh * hd + 2 * nkv * hd)
    ow_s = slot(nh * hd, h)
    pool_s = slot(h, mh_r)
    unpool_s = slot(mh_r, h)
    lm_s = slot(h, lm_w)
    off = (0,)
    for s in (qwkv_s, ow_s, pool_s):
        off = off + (off[-1] + s[-1],)
    t_layer = off[-1] + unpool_s[-1]
    t_layers = c.layers * t_layer
    t_total = t_layers + lm_s[-1]
    fams = (("qwkv", qwkv_s), ("ow", ow_s), ("pool", pool_s), ("unpool", unpool_s))
    wrefs = {}

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
        gu_packed_r,
        gu_scales_r,
        gu_global_r,
        dw_packed_r,
        dw_scales_r,
        dw_global_r,
        gu_dense_r,
        dw_dense_r,
        fnorm_r,
        emb_r,
        kc_r,
        vc_r,
        logits_out,
        token_out,
        kc_out,
        vc_out,
        *refs,
    ):
        (
            banks,
            ws,
            routerbuf,
            r_sem,
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
            mixedbuf,
            pooledbuf,
            rowsbuf,
            combuf,
            recv,
            recv_comb,
            recv_rows,
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
            gu_p,
            gus_p,
            dw_p,
            dws_p,
            guq_p,
            gusq_p,
            dwq_p,
            dwsq_p,
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
        ) = refs
        rank = jax.lax.axis_index("tp")
        pos = pr[0]
        wrefs.update(qwkv=qwkv_r, ow=ow_r, pool=pool_r, unpool=unpool_r)

        def copy_tile(weight, li, bank, ti, bk, bn):
            src = weight.at[ti] if li is None else weight.at[li, ti]
            return tpu.make_async_copy(
                src, banks.at[bank, pl.ds(0, bk), pl.ds(0, bn)], ws.at[bank]
            )

        def fetch(g):
            @pl.when(g < t_total)
            def valid():
                @pl.when(g < t_layers)
                def layer_tile():
                    li = g // t_layer
                    sub = g % t_layer
                    for i, (name, s) in enumerate(fams):

                        @pl.when((sub >= off[i]) & (sub < off[i] + s[-1]))
                        def load(name=name, s=s, i=i):
                            copy_tile(
                                wrefs[name], li, g % BANK_COUNT, sub - off[i], s[0], s[1]
                            ).start()

                @pl.when(g >= t_layers)
                def lm_tile():
                    copy_tile(lmw_r, None, g % BANK_COUNT, g - t_layers, lm_s[0], lm_s[1]).start()

        def gemv(src, dst, weight, li, base, s, out_f32=False, acc=None):
            """Matvec [1,K] x [K,N] streamed through the bank ring."""
            bk, bn, nk, nb, nt = s
            n = nb * bn
            ar = (accum if acc is None else acc).at[:, pl.ds(0, n)]
            ar[...] = jnp.zeros((8, n), jnp.float32)

            def step(ti, bank):
                ki, ni = ti % nk, ti // nk
                copy_tile(weight, li, bank, ti, bk, bn).wait()
                if "nogemv" not in ablate:
                    a = jnp.broadcast_to(src[:, pl.ds(ki * bk, bk)], (8, bk))
                    ar[:, pl.ds(ni * bn, bn)] += jnp.dot(
                        a, banks[bank, pl.ds(0, bk), pl.ds(0, bn)], preferred_element_type=jnp.float32
                    )
                fetch(base + ti + BANK_COUNT)
                return jnp.where(bank + 1 == BANK_COUNT, 0, bank + 1)

            jax.lax.fori_loop(0, nt, step, base % BANK_COUNT)
            dst[...] = ar[:1, :].astype(jnp.float32 if out_f32 else dst.dtype)

        def exchange(src, stage, recv_buf, slot):
            """XOR-butterfly (bf16 transport, like the reference's allreduce):
            recv_buf[slot, j] ends up holding rank j's src."""
            stage[...] = src[...].astype(stage.dtype)
            recv_buf[slot, rank, ...] = stage[...]
            if "noxchg" in ablate:
                return
            for offset in range(1, tp):
                tpu.make_async_remote_copy(
                    stage,
                    recv_buf.at[slot, rank],
                    sends.at[offset - 1],
                    recvs.at[offset - 1],
                    device_id=(rank ^ offset,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()
            for offset in range(1, tp):
                tpu.make_async_remote_copy(
                    stage,
                    recv_buf.at[slot, rank],
                    sends.at[offset - 1],
                    recvs.at[offset - 1],
                    device_id=(rank ^ offset,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).wait()

        def allreduce(iteration):
            """All-reduce proj ([1, h] f32); returns the fp32 total [1, h]."""
            slot = iteration % 2
            exchange(proj, xstage_h, recv, slot)
            return jnp.sum(recv[slot, ...].astype(jnp.float32), axis=0)

        def exchange_flat(src, recv_buf, slot, width):
            """Gather per-rank slices into recv_buf[slot, r] (per-rank rows)."""
            recv_buf[slot, rank, 0, :] = src[0, :width]
            if "noxchg" in ablate:
                return
            for offset in range(1, tp):
                tpu.make_async_remote_copy(
                    src.at[0],
                    recv_buf.at[slot, rank, 0, :],
                    sends.at[offset - 1],
                    recvs.at[offset - 1],
                    device_id=(rank ^ offset,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()
            for offset in range(1, tp):
                tpu.make_async_remote_copy(
                    src.at[0],
                    recv_buf.at[slot, rank, 0, :],
                    sends.at[offset - 1],
                    recvs.at[offset - 1],
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
            lo = jnp.where(sliding, pos - c.sliding_window + 1, 0)
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

        def attend(li, q8, k2, v2, sliding):
            """q8 [nh//2, 128] (two heads per row), k2/v2 [nkv//2, 128] (same packing).

            One streaming loop over the cache blocks; each K/V copy carries all
            of this rank's kv heads (nkv) at once.
            """
            rows_per_group = groups // 2
            count = pos // sbw + 1
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
            for t in range(2):

                @pl.when(t <= pos // sbw)
                def issue(t=t):
                    t0 = min(t, context // sbw - 1)  # in-bounds for the trace; guard gates execution
                    tpu.make_async_copy(
                        kc_r.at[li, :, pl.ds(t0 * sbw, sbw), :], kb.at[t], ks.at[t]
                    ).start()
                    tpu.make_async_copy(
                        vc_r.at[li, :, pl.ds(t0 * sbw, sbw), :], vb.at[t], vs.at[t]
                    ).start()

            def step(t, _):
                tpu.make_async_copy(
                    kc_r.at[li, :, pl.ds(t * sbw, sbw), :], kb.at[t % 2], ks.at[t % 2]
                ).wait()
                tpu.make_async_copy(
                    vc_r.at[li, :, pl.ds(t * sbw, sbw), :], vb.at[t % 2], vs.at[t % 2]
                ).wait()

                @pl.when(t == pos // sbw)
                def update_current():
                    mask = jnp.arange(sbw)[:, None] == pos % sbw
                    for g in range(nkv):
                        key_g = k2[g // 2, (g % 2) * hd : (g % 2 + 1) * hd]
                        value_g = v2[g // 2, (g % 2) * hd : (g % 2 + 1) * hd]
                        kb[t % 2, g, :, :hd] = jnp.where(mask, key_g[None, :], kb[t % 2, g, :, :hd])
                        vb[t % 2, g, :, :hd] = jnp.where(mask, value_g[None, :], vb[t % 2, g, :, :hd])
                    tpu.make_async_copy(
                        kb.at[t % 2], kc_r.at[li, :, pl.ds(pos // sbw * sbw, sbw), :], io.at[0]
                    ).start()
                    tpu.make_async_copy(
                        vb.at[t % 2], vc_r.at[li, :, pl.ds(pos // sbw * sbw, sbw), :], io.at[1]
                    ).start()

                if "noattn" not in ablate and "nocompute" not in ablate:
                    for g in range(nkv):
                        flash_block(
                            qbg[g], kb[t % 2, g], vb[t % 2, g],
                            amg, alg, aag, g, t, sliding,
                        )

                @pl.when(t + 2 < count)
                def refill(t=t):
                    tpu.make_async_copy(
                        kc_r.at[li, :, pl.ds((t + 2) * sbw, sbw), :], kb.at[t % 2], ks.at[t % 2]
                    ).start()
                    tpu.make_async_copy(
                        vc_r.at[li, :, pl.ds((t + 2) * sbw, sbw), :], vb.at[t % 2], vs.at[t % 2]
                    ).start()

            jax.lax.fori_loop(0, count, step, None)
            tpu.make_async_copy(
                kb.at[pos // sbw % 2], kc_r.at[li, :, pl.ds(pos // sbw * sbw, sbw), :], io.at[0]
            ).wait()
            tpu.make_async_copy(
                vb.at[pos // sbw % 2], vc_r.at[li, :, pl.ds(pos // sbw * sbw, sbw), :], io.at[1]
            ).wait()
            for g in range(nkv):
                ae = (aag[g] / alg[g])[:rows_per_group, :hd].astype(jnp.bfloat16)
                ao = (aag[g] / alg[g])[rows_per_group : 2 * rows_per_group, :hd].astype(jnp.bfloat16)
                attended[pl.ds(g * rows_per_group, rows_per_group), :] = jnp.concatenate(
                    (ae, ao), axis=1
                )

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

        def run_experts(li, pooled, ids, is_dense, o_packed, o_dense, ggu, ggw):
            """Rows for the 8 routed experts into rowsbuf (fp32 TP partials)."""

            def packed_all():
                rows = []
                iota_e = jax.lax.iota(jnp.int32, c.experts)
                n_pairs = c.top_k // 2

                def issue_pair(sp, slot_):
                    e0, e1 = ids[2 * sp], ids[2 * sp + 1]
                    tpu.make_async_copy(gu_packed_r.at[o_packed, e0], gu_p.at[slot_], eg_sem.at[0]).start()
                    tpu.make_async_copy(gu_packed_r.at[o_packed, e1], guq_p.at[slot_], eg_sem.at[0]).start()
                    tpu.make_async_copy(gu_scales_r.at[o_packed, e0], gus_p.at[slot_], eg_sem.at[1]).start()
                    tpu.make_async_copy(gu_scales_r.at[o_packed, e1], gusq_p.at[slot_], eg_sem.at[1]).start()
                    tpu.make_async_copy(dw_packed_r.at[o_packed, e0], dw_p.at[slot_], eg_sem.at[2]).start()
                    tpu.make_async_copy(dw_packed_r.at[o_packed, e1], dwq_p.at[slot_], eg_sem.at[2]).start()
                    tpu.make_async_copy(dw_scales_r.at[o_packed, e0], dws_p.at[slot_], eg_sem.at[3]).start()
                    tpu.make_async_copy(dw_scales_r.at[o_packed, e1], dwsq_p.at[slot_], eg_sem.at[3]).start()

                issue_pair(0, 0)
                for sp in range(n_pairs):
                    slot_ix = sp % EXPERT_SLOTS
                    e0, e1 = ids[2 * sp], ids[2 * sp + 1]
                    tpu.make_async_copy(gu_packed_r.at[o_packed, e0], gu_p.at[slot_ix], eg_sem.at[0]).wait()
                    tpu.make_async_copy(gu_packed_r.at[o_packed, e1], guq_p.at[slot_ix], eg_sem.at[0]).wait()
                    tpu.make_async_copy(gu_scales_r.at[o_packed, e0], gus_p.at[slot_ix], eg_sem.at[1]).wait()
                    tpu.make_async_copy(gu_scales_r.at[o_packed, e1], gusq_p.at[slot_ix], eg_sem.at[1]).wait()
                    tpu.make_async_copy(dw_packed_r.at[o_packed, e0], dw_p.at[slot_ix], eg_sem.at[2]).wait()
                    tpu.make_async_copy(dw_packed_r.at[o_packed, e1], dwq_p.at[slot_ix], eg_sem.at[2]).wait()
                    tpu.make_async_copy(dw_scales_r.at[o_packed, e0], dws_p.at[slot_ix], eg_sem.at[3]).wait()
                    tpu.make_async_copy(dw_scales_r.at[o_packed, e1], dwsq_p.at[slot_ix], eg_sem.at[3]).wait()
                    fm0 = (iota_e == e0).astype(jnp.float32)
                    fm1 = (iota_e == e1).astype(jnp.float32)
                    g13a = jnp.sum(ggu * jnp.stack((fm0, fm0), axis=1), axis=0)
                    g13b = jnp.sum(ggu * jnp.stack((fm1, fm1), axis=1), axis=0)
                    g2a = jnp.sum(ggw * fm0, axis=0)
                    g2b = jnp.sum(ggw * fm1, axis=0)

                    @pl.when(sp + 1 < n_pairs)
                    def prefetch(sp=sp):
                        issue_pair(sp + 1, (sp + 1) % EXPERT_SLOTS)

                    if "noexpert" in ablate or "nocompute" in ablate:
                        rows.append(jnp.zeros((mh,), jnp.float32))
                        rows.append(jnp.zeros((mh,), jnp.float32))
                        continue
                    # paired gate/up: two dots on the shared input (no weight concat)
                    a = jnp.broadcast_to(pooled.astype(jnp.bfloat16)[None, :], (8, mh))
                    w13a = dequant_k_major(gu_p[slot_ix], gus_p[slot_ix], mh, 2 * inter_r)
                    gu_a = jnp.dot(a, w13a, preferred_element_type=jnp.float32)[0]  # [2*inter_r]
                    w13b = dequant_k_major(guq_p[slot_ix], gusq_p[slot_ix], mh, 2 * inter_r)
                    gu_b = jnp.dot(a, w13b, preferred_element_type=jnp.float32)[0]
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
                    w2a = dequant_k_major(dw_p[slot_ix], dws_p[slot_ix], inter_r, mh)
                    rows.append(expert_gemv_ng(w2a, act_a, mh, inter_r) * g2a)
                    w2b = dequant_k_major(dwq_p[slot_ix], dwsq_p[slot_ix], inter_r, mh)
                    rows.append(expert_gemv_ng(w2b, act_b, mh, inter_r) * g2b)
                rowsbuf[...] = jnp.stack(rows)

            def dense_all():
                if dense_overlay:
                    # The packed-slot buffers are idle on dense layers: overlay
                    # the dense expert tiles on them (frees VMEM for the ring).
                    gu_gate = pool_alias.view(gu_p, (mh, inter_r), jnp.bfloat16)
                    gu_up = pool_alias.view(guq_p, (mh, inter_r), jnp.bfloat16)
                    dw_0 = pool_alias.view(dw_p, (inter_r, mh // 2), jnp.bfloat16)
                    dw_1 = pool_alias.view(dwq_p, (inter_r, mh // 2), jnp.bfloat16)
                rows = []
                for s in range(c.top_k):
                    e = ids[s]
                    if dense_overlay:
                        tpu.make_async_copy(
                            gu_dense_r.at[o_dense, e, :, pl.ds(0, inter_r)], gu_gate, ed_sem.at[0]
                        ).start()
                        tpu.make_async_copy(
                            gu_dense_r.at[o_dense, e, :, pl.ds(inter_r, inter_r)], gu_up, ed_sem.at[1]
                        ).start()
                        tpu.make_async_copy(
                            dw_dense_r.at[o_dense, e, :, pl.ds(0, mh // 2)], dw_0, ed_sem.at[2]
                        ).start()
                        tpu.make_async_copy(
                            dw_dense_r.at[o_dense, e, :, pl.ds(mh // 2, mh // 2)], dw_1, ed_sem.at[3]
                        ).start()
                        tpu.make_async_copy(
                            gu_dense_r.at[o_dense, e, :, pl.ds(0, inter_r)], gu_gate, ed_sem.at[0]
                        ).wait()
                        tpu.make_async_copy(
                            gu_dense_r.at[o_dense, e, :, pl.ds(inter_r, inter_r)], gu_up, ed_sem.at[1]
                        ).wait()
                        tpu.make_async_copy(
                            dw_dense_r.at[o_dense, e, :, pl.ds(0, mh // 2)], dw_0, ed_sem.at[2]
                        ).wait()
                        tpu.make_async_copy(
                            dw_dense_r.at[o_dense, e, :, pl.ds(mh // 2, mh // 2)], dw_1, ed_sem.at[3]
                        ).wait()
                    else:
                        tpu.make_async_copy(gu_dense_r.at[o_dense, e], gu_d, ed_sem.at[0]).start()
                        tpu.make_async_copy(dw_dense_r.at[o_dense, e], dw_d, ed_sem.at[1]).start()
                        tpu.make_async_copy(gu_dense_r.at[o_dense, e], gu_d, ed_sem.at[0]).wait()
                        tpu.make_async_copy(dw_dense_r.at[o_dense, e], dw_d, ed_sem.at[1]).wait()
                    if "noexpert" in ablate or "nocompute" in ablate:
                        rows.append(jnp.zeros((mh,), jnp.float32))
                    elif dense_overlay:
                        gate = expert_gemv_ng(gu_gate[...], pooled, inter_r, mh)
                        up = expert_gemv_ng(gu_up[...], pooled, inter_r, mh)
                        act = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
                        rows.append(
                            jnp.concat(
                                (
                                    expert_gemv_ng(dw_0[...], act, mh // 2, inter_r),
                                    expert_gemv_ng(dw_1[...], act, mh // 2, inter_r),
                                )
                            )
                        )
                    else:
                        gu = expert_gemv_ng(gu_d[...], pooled, 2 * inter_r, mh)
                        gate, up = gu[:inter_r], gu[inter_r:]
                        act = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
                        rows.append(expert_gemv_ng(dw_d[...], act, mh, inter_r))
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
            tpu.make_async_copy(router_r.at[li], routerbuf.at[li % 2], r_sem.at[0]).wait()
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
                tpu.make_async_copy(router_r.at[li + 1], routerbuf.at[(li + 1) % 2], r_sem.at[0]).start()
                tpu.make_async_copy(gu_global_r.at[nxt_packed], ggu.at[(li + 1) % 2], g_sem.at[0]).start()
                tpu.make_async_copy(dw_global_r.at[nxt_packed], ggw.at[(li + 1) % 2], g_sem.at[1]).start()

            # --- attention branch ---
            value = residual[...].astype(jnp.float32)
            value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + c.eps)
            normed[...] = (value * sv[0].astype(jnp.float32)).astype(jnp.bfloat16)
            base = li * t_layer
            gemv(normed, qwkvbuf, qwkv_r, li, base + off[0], qwkv_s)
            qg = qwkvbuf[0, : 2 * nh * hd].reshape(nh, hdp)  # [q pairs | gate pairs]
            q8 = qg[: nh // 2]
            g8 = qg[nh // 2 :]
            kv_off = 2 * nh * hd
            k2 = qwkvbuf[0, kv_off : kv_off + nkv * hd].reshape(nkv // 2, hdp)
            v2 = qwkvbuf[0, kv_off + nkv * hd : kv_off + 2 * nkv * hd].reshape(nkv // 2, hdp)
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
            gemv(attbuf, proj, ow_r, li, base + off[1], ow_s, out_f32=True)
            total = allreduce(li * 4)
            bvalue = total * jax.lax.rsqrt(jnp.mean(total * total, axis=-1, keepdims=True) + c.post_eps)
            residual[...] = sv[3] * residual[...] + sv[4] * bvalue

            # --- MoE branch ---
            value = residual[...].astype(jnp.float32)
            value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + c.eps)
            normed[...] = (value * sv[1].astype(jnp.float32)).astype(jnp.bfloat16)
            rpartial = jnp.sum(
                normed[0].astype(jnp.float32)[:, None] * routerbuf[li % 2][...], axis=0
            )
            gemv(normed, poolbuf, pool_r, li, base + off[2], pool_s)
            # one fused gather: router partial in [0:128], pool shard in [128:128+mh_r]
            combuf[0, :eshard] = rpartial
            combuf[0, 128 : 128 + mh_r] = poolbuf[0].astype(jnp.float32)
            slot = (li * 4 + 1) % 2
            exchange_flat(combuf, recv_comb, slot, 128 + mh_r)
            logits_all = jnp.concatenate(
                [recv_comb[slot, r, 0, :eshard] for r in range(tp)], axis=0
            )  # [experts], rank-major
            scores_all = jax.nn.sigmoid(logits_all * c.output_multiplier)
            ranked = scores_all + biasbuf[li].astype(jnp.float32)
            iota_e = jax.lax.iota(jnp.int32, c.experts)
            ids_l, rw_l = [], []
            wsum = jnp.zeros((), jnp.float32)
            for j in range(c.top_k):
                idx = jnp.argmax(ranked).astype(jnp.int32)  # first max, as min-index tie-break
                ids_l.append(idx)
                wj = jnp.sum(jnp.where(iota_e == idx, scores_all, 0.0))
                rw_l.append(wj)
                wsum = wsum + wj
                ranked = jnp.where(iota_e == idx, -jnp.inf, ranked)
            rw_arr = jnp.stack(rw_l)
            pooledbuf[0] = jnp.concatenate(
                [recv_comb[slot, r, 0, 128 : 128 + mh_r] for r in range(tp)], axis=0
            ).astype(jnp.bfloat16)
            pv = pooledbuf[0].astype(jnp.float32)
            pv *= jax.lax.rsqrt(jnp.mean(pv * pv, axis=-1, keepdims=True) + c.eps)
            pooledbuf[0] = (pv * xnormsbuf[li, 0].astype(jnp.float32)).astype(jnp.bfloat16)

            run_experts(li, pooledbuf[0], ids_l, is_dense, o_packed, o_dense, ggu[li % 2], ggw[li % 2])

            exchange(rowsbuf, xstage_rows, recv_rows, (li * 4 + 2) % 2)
            rows = jnp.sum(recv_rows[(li * 4 + 2) % 2].astype(jnp.float32), axis=0)  # [top_k, mh] f32
            rows = rows * xnormsbuf[li, 1].astype(jnp.float32)
            rows = rows * jax.lax.rsqrt(jnp.mean(rows * rows, axis=-1, keepdims=True) + c.post_eps)
            mixed = jnp.sum(rows * (rw_arr / (wsum + c.router_eps))[:, None], axis=0)
            mixedbuf[0] = mixed.astype(jnp.bfloat16)
            unpinbuf[0] = mixedbuf[0, pl.ds(rank * mh_r, mh_r)]
            gemv(unpinbuf, proj, unpool_r, li, base + off[3], unpool_s, out_f32=True)
            total = allreduce(li * 4 + 3)
            bvalue = total * sv[2].astype(jnp.float32)
            bvalue = bvalue * jax.lax.rsqrt(jnp.mean(bvalue * bvalue, axis=-1, keepdims=True) + c.post_eps)
            residual[...] = sv[5] * residual[...] + sv[6] * bvalue

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

        # Resident small vectors: router bias + pre/post expert norms.
        tpu.make_async_copy(router_bias_r, biasbuf, eg_sem.at[0]).start()
        tpu.make_async_copy(expert_norms_r, xnormsbuf, eg_sem.at[1]).start()

        # Embedding: fetch this rank's candidate row, all-reduce, weightless norm.
        tok = tokenr[...][0]
        lo = rank * vshard
        tokl = jnp.clip(tok - lo, 0, vshard - 1)
        tpu.make_async_copy(emb_r.at[pl.ds(tokl // 8 * 8, 8), :], rowbuf, e_sem.at[0]).start()
        tpu.make_async_copy(scalars_r.at[0], scalarsbuf.at[0], sc_sem.at[0]).start()
        tpu.make_async_copy(router_r.at[0], routerbuf.at[0], r_sem.at[0]).start()
        g0_packed = jnp.int32(0 if 0 in c.bf16_expert_layers else 0)
        tpu.make_async_copy(gu_global_r.at[g0_packed], ggu.at[0], g_sem.at[0]).start()
        tpu.make_async_copy(dw_global_r.at[g0_packed], ggw.at[0], g_sem.at[1]).start()
        for ti in range(BANK_COUNT):
            fetch(jnp.int32(ti))
        barrier = tpu.get_barrier_semaphore()
        if tp > 1:
            for offset in range(1, tp):
                pl.semaphore_signal(
                    barrier, 1, device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH
                )
            pl.semaphore_wait(barrier, tp - 1)
        tpu.make_async_copy(router_bias_r, biasbuf, eg_sem.at[0]).wait()
        tpu.make_async_copy(expert_norms_r, xnormsbuf, eg_sem.at[1]).wait()
        tpu.make_async_copy(emb_r.at[pl.ds(tokl // 8 * 8, 8), :], rowbuf, e_sem.at[0]).wait()
        sel = jax.lax.iota(jnp.int32, 8)
        row = jnp.sum(jnp.where(sel[:, None] == tokl % 8, rowbuf[...], 0), axis=0).astype(
            jnp.float32
        )
        row = jnp.where((tok >= lo) & (tok < lo + vshard), row, 0.0)
        proj[...] = row[None, :]
        total = allreduce(1)
        value = total * jax.lax.rsqrt(jnp.mean(total * total, axis=-1, keepdims=True) + c.eps)
        residual[...] = value

        jax.lax.fori_loop(0, c.layers, layer, None)

        # Final norm (plain gain, center 0) + LM head + greedy argmax.
        tpu.make_async_copy(fnorm_r, fnormbuf, sc_sem.at[1]).start()
        tpu.make_async_copy(fnorm_r, fnormbuf, sc_sem.at[1]).wait()
        value = residual[...].astype(jnp.float32)
        value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + c.eps)
        normed[...] = (value * fnormbuf[0].astype(jnp.float32)).astype(jnp.bfloat16)
        gemv(normed, logits_out, lmw_r, None, t_layers, lm_s, out_f32=True, acc=lm_accum)
        if "dumpres" in ablate:
            # debug path (tests only): overwrite the padded logits tail with the stream
            if vshard + h <= lm_w:
                logits_out[0, vshard : vshard + h] = residual[0]

        vals = lm_accum[0, :vshard]
        top = jnp.max(vals)
        iota = jax.lax.iota(jnp.int32, vshard)
        best = jnp.min(jnp.where(vals == top, iota, vshard)).astype(jnp.int32)
        pairbuf[...] = jnp.concatenate((top[None], best.astype(jnp.float32)[None]))
        pairs[0, rank, ...] = pairbuf[...]
        if "noxchg" not in ablate:
            for offset in range(1, tp):
                tpu.make_async_remote_copy(
                    pairbuf,
                    pairs.at[0, rank],
                    sends.at[offset - 1],
                    recvs.at[offset - 1],
                    device_id=(rank ^ offset,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()
            for offset in range(1, tp):
                tpu.make_async_remote_copy(
                    pairbuf,
                    pairs.at[0, rank],
                    sends.at[offset - 1],
                    recvs.at[offset - 1],
                    device_id=(rank ^ offset,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).wait()
        allvals = pairs[0, :, 0]
        topv = jnp.max(allvals)
        wiota = jax.lax.iota(jnp.int32, tp)
        winner = jnp.min(jnp.where(allvals == topv, wiota, tp)).astype(jnp.int32)
        token_id = winner * vshard + jnp.sum(jnp.where(wiota == winner, pairs[0, :, 1], 0.0))
        token_out[...] = token_id.astype(jnp.int32).reshape(1)

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
        w["gu_packed"],
        w["gu_scales"],
        w["gu_global"],
        w["dw_packed"],
        w["dw_scales"],
        w["dw_global"],
        w["gu_dense"],
        w["dw_dense"],
        w["fnorm"],
        w["emb"],
    )
    args = (token, *weights_in, *states_in)
    logits_shape = jax.ShapeDtypeStruct((1, lm_w), jnp.float32)
    scratch = (
        v((BANK_COUNT, MAX_BANK_K, MAX_BANK_N)),
        dm(BANK_COUNT),
        v((2, h, eshard), jnp.float32),  # routerbuf
        dm(1),  # r_sem
        v((2, 8, h), jnp.float32),  # scalarsbuf
        dm(2),  # sc_sem
        v((c.layers, c.experts), jnp.float32),  # biasbuf
        v((c.layers, 2, mh), jnp.float32),  # xnormsbuf
        v((1, h), jnp.float32),  # residual
        v((1, h)),  # normed
        v((1, h), jnp.float32),  # proj
        v((8, max(h, 2 * nh * hd + 2 * nkv * hd)), jnp.float32),  # accum
        v((8, lm_w), jnp.float32),  # lm_accum
        v((1, 2 * nh * hd + 2 * nkv * hd)),  # qwkvbuf
        v((1, nh * hd)),  # attbuf
        v((1, mh_r)),  # poolbuf
        v((1, mh_r)),  # unpinbuf
        v((1, mh)),  # mixedbuf
        v((1, mh)),  # pooledbuf
        v((c.top_k, mh), jnp.float32),  # rowsbuf
        v((1, 128 + mh_r), jnp.float32),  # combuf (router partial | pool shard)
        v((2, tp, 1, h)),  # recv (bf16 exchange transport)
        v((2, tp, 1, 128 + mh_r), jnp.float32),  # recv_comb
        v((2, tp, c.top_k, mh)),  # recv_rows (bf16 exchange transport)
        v((1, h)),  # xstage_h
        v((c.top_k, mh)),  # xstage_rows
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
        v((EXPERT_SLOTS, mh // 2, 2 * inter_r), jnp.uint8),  # gu_p
        v((EXPERT_SLOTS, mh // 16, 2 * inter_r), jnp.uint8),  # gus_p
        v((EXPERT_SLOTS, inter_r // 2, mh), jnp.uint8),  # dw_p
        v((EXPERT_SLOTS, inter_r // 16, mh), jnp.uint8),  # dws_p
        v((EXPERT_SLOTS, mh // 2, 2 * inter_r), jnp.uint8),  # guq_p
        v((EXPERT_SLOTS, mh // 16, 2 * inter_r), jnp.uint8),  # gusq_p
        v((EXPERT_SLOTS, inter_r // 2, mh), jnp.uint8),  # dwq_p
        v((EXPERT_SLOTS, inter_r // 16, mh), jnp.uint8),  # dwsq_p
        v((2, c.experts, 2), jnp.float32),  # ggu (per-layer gate/up globals)
        v((2, c.experts), jnp.float32),  # ggw (per-layer down globals)
        dm(2),  # g_sem
        dm(4),  # eg_sem
        *((v((1, 128)),) if dense_overlay else (v((mh, 2 * inter_r)),)),  # gu_d
        *((v((1, 128)),) if dense_overlay else (v((inter_r, mh)),)),  # dw_d
        dm(4),  # ed_sem
        v((8, h)),  # rowbuf
        dm(1),  # e_sem
        v((2,), jnp.float32),  # pairbuf
        v((2, tp, 2), jnp.float32),  # pairs
        v((1, h), jnp.float32),  # fnormbuf
    )
    state_offset = 1 + 1 + len(weights_in)
    result = pl.pallas_call(
        body,
        out_shape=(
            logits_shape,
            jax.ShapeDtypeStruct((1,), jnp.int32),
            *(shape(s) for s in states_in),
        ),
        input_output_aliases={state_offset + i: i + 2 for i in range(2)},
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[pl.BlockSpec()] + [pl.BlockSpec(memory_space=tpu.HBM)] * (len(args) - 1),
            out_specs=[pl.BlockSpec()] * 2 + [pl.BlockSpec(memory_space=tpu.HBM)] * 2,
            scratch_shapes=scratch,
        ),
        compiler_params=tpu.CompilerParams(
            collective_id=46 if tp > 1 else None,
            vmem_limit_bytes=64 * 1024**2,
            disable_bounds_checks=True,
            shape_invariant_numerics=True,
        ),
        name="mpk_transformer_stack",
    )(position, *args)
    return result[0], result[1], dict(zip(("kcache", "vcache"), result[2:]))


def make_decode(mesh, c, context, tp, ablate=()):
    """jit(shard_map(transformer_stack)); (weights, states, token, position)."""
    if tp == 16:
        return _make_decode_persistent(mesh, c, context, ablate)
    ablate = frozenset(ablate)

    def local(global_weights, global_states, token, position):
        weights = jax.tree.map(lambda a: a[0], global_weights)
        states = jax.tree.map(lambda a: a[0], global_states)
        logits, nxt, new_states = transformer_stack(
            c, token, weights, states, position, context, ablate, tp
        )
        return nxt, logits, jax.tree.map(lambda a: a[None], new_states)

    def run(w, states, token, position):
        if tp == 8 and "kvcache" in states:
            return _make_decode_persistent(mesh, c, context, ablate)(w, states, token, position)
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


def _prefill_router_topk(ranked, top_k):
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


def _prefill_exchange_down(x, tp=16):
    """Exchange one down-projection layer from input to output shards."""
    experts, depth, width = x.shape
    columns = width // tp

    def kernel(src, out, sends, recvs):
        rank = jax.lax.axis_index("tp")
        barrier = tpu.get_barrier_semaphore()
        for offset in range(1, tp):
            pl.semaphore_signal(barrier, 1, device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(barrier, tp - 1)
        local = tpu.make_async_copy(src.at[:, :, pl.ds(rank * columns, columns)], out.at[rank], recvs.at[0])
        local.start()
        transfers = []
        for offset in range(1, tp):
            peer = rank ^ offset
            transfer = tpu.make_async_remote_copy(
                src.at[:, :, pl.ds(peer * columns, columns)], out.at[rank],
                sends.at[offset], recvs.at[offset],
                device_id=(peer,), device_id_type=pl.DeviceIdType.MESH,
            )
            transfer.start()
            transfers.append(transfer)
        local.wait()
        for transfer in transfers:
            transfer.wait()

    out = pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((tp, experts, depth, columns), x.dtype),
        in_specs=(pl.BlockSpec(memory_space=tpu.HBM),),
        out_specs=pl.BlockSpec(memory_space=tpu.HBM),
        scratch_shapes=(tpu.SemaphoreType.DMA((tp,)), tpu.SemaphoreType.DMA((tp,))),
        compiler_params=tpu.CompilerParams(collective_id=0),
    )(x)
    return out


def _prefill_expand_down(packets, globals_):
    shards, count, depth, n = packets.shape
    k = depth // 9 * 16
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


def _prefill_packed_activation(
    inputs, packed, gains, group_sizes, *, shard_width, tile_rows, tile_columns
):
    """Expand each expert once per output tile and fuse the gate activation."""
    from jax.experimental.pallas.ops.tpu.megablox.gmm import _get_store_mask, make_group_metadata

    original_rows, k = inputs.shape
    experts, _, n = packed.shape
    tm, tn = tile_rows, tile_columns
    m = pl.cdiv(original_rows, tm) * tm
    inputs = jnp.pad(inputs, ((0, m - original_rows), (0, 0)))
    metadata, active_tiles = make_group_metadata(
        group_sizes=group_sizes,
        m=m,
        tm=tm,
        start_group=jnp.int32(0),
        num_nonzero_groups=experts,
        visit_empty_groups=False,
    )
    metadata = (*metadata, active_tiles[None])
    gains = gains.reshape(experts, 1, 2)

    def kernel(metadata, x, weights, gains, out, accumulator, values, scales, cached, sem):
        column, tile, reduction = pl.program_id(0), pl.program_id(1), pl.program_id(2)
        expert = metadata[1][tile]
        previous = metadata[1][jnp.maximum(tile - 1, 0)]
        columns = pl.ds(pl.multiple_of(column * tn, 128), tn)

        def transfers(expert):
            return (
                tpu.make_async_copy(
                    weights.at[
                        expert, pl.ds(pl.multiple_of(reduction * (k // 2), 32), k // 2), columns
                    ],
                    values,
                    sem.at[0],
                ),
                tpu.make_async_copy(
                    weights.at[
                        expert,
                        pl.ds(pl.multiple_of(k // 2 + reduction * (k // 16), 32), k // 16),
                        columns,
                    ],
                    scales,
                    sem.at[1],
                ),
            )

        @pl.when((tile == 0) | (previous != expert))
        def expand():
            value_copy, scale_copy = transfers(expert)

            @pl.when(tile == 0)
            def first():
                value_copy.start()
                scale_copy.start()

            value_copy.wait()
            scale_copy.wait()
            if n == 2 * shard_width:
                gain = jnp.where(
                    column * tn + jnp.arange(tn)[None] < n // 2, gains[0, 0], gains[0, 1]
                )
            else:
                gain = jnp.where(
                    (jnp.arange(tn)[None] // shard_width) % 2 == 0, gains[0, 0], gains[0, 1]
                )
            if tn <= 1024:
                quantized = tpu.bitcast(values[...], jnp.float4_e2m1fn).astype(F32)
                local_scale = jax.lax.bitcast_convert_type(scales[...], jnp.float8_e4m3fn).astype(
                    F32
                )
                local_scale *= gain
                cached[...] = (
                    (quantized.reshape(k // 16, 16, tn) * local_scale[:, None])
                    .reshape(k, tn)
                    .astype(jnp.bfloat16)
                )

            else:
                expand_rows = min(k, 512 if tn >= 4096 else 1024)

                def expand_chunk(chunk, unused):
                    start = pl.multiple_of(chunk * expand_rows, 128)
                    quantized = tpu.bitcast(
                        values[pl.ds(pl.multiple_of(start // 2, 32), expand_rows // 2)],
                        jnp.float4_e2m1fn,
                    ).astype(F32)
                    scale = (
                        jax.lax.bitcast_convert_type(
                            scales[pl.ds(pl.multiple_of(start // 16, 32), expand_rows // 16)],
                            jnp.float8_e4m3fn,
                        ).astype(F32)
                        * gain
                    )
                    cached[pl.ds(start, expand_rows)] = (
                        (quantized.reshape(expand_rows // 16, 16, tn) * scale[:, None])
                        .reshape(expand_rows, tn)
                        .astype(jnp.bfloat16)
                    )
                    return unused

                jax.lax.fori_loop(0, k // expand_rows, expand_chunk, None)

        following = metadata[1][jnp.minimum(tile + 1, metadata[1].shape[0] - 1)]

        @pl.when((tile + 1 < metadata[3][0]) & (following != expert))
        def prefetch():
            for transfer in transfers(following):
                transfer.start()

        accumulator[...] = jnp.zeros((tm, tn), F32)
        accumulator[...] += jnp.dot(x[...], cached[...], preferred_element_type=F32)
        projected = accumulator[...].astype(jnp.bfloat16).astype(F32)
        if tn == 2 * shard_width:
            gate, up = jnp.split(projected, 2, axis=1)
            activated = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
        else:
            projected = projected.reshape(tm, tn // (2 * shard_width), 2, shard_width)
            activated = (
                (jax.nn.silu(projected[:, :, 0]) * projected[:, :, 1])
                .reshape(tm, tn // 2)
                .astype(jnp.bfloat16)
            )
        mask = _get_store_mask(grid_id=tile, group_metadata=metadata[:3], tm=tm, tn=tn // 2)
        out[...] = jnp.where(mask, activated, out[...])

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n // 2), jnp.bfloat16),
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(
                pl.BlockSpec((tm, k), lambda n, m, k, meta: (meta[2][m], k)),
                pl.BlockSpec(memory_space=tpu.HBM),
                pl.BlockSpec((None, 1, 2), lambda n, m, k, meta: (meta[1][m], 0, 0)),
            ),
            out_specs=pl.BlockSpec((tm, tn // 2), lambda n, m, k, meta: (meta[2][m], n)),
            grid=(n // tn, active_tiles, 1),
            scratch_shapes=(
                tpu.VMEM((tm, tn), F32),
                tpu.VMEM((k // 2, tn), jnp.uint8),
                tpu.VMEM((k // 16, tn), jnp.uint8),
                tpu.VMEM((k, tn), jnp.bfloat16),
                tpu.SemaphoreType.DMA((2,)),
            ),
        ),
        compiler_params=tpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            vmem_limit_bytes=(64 if tm >= 512 or tn >= 1024 else 32) * 1024**2,
        ),
        name="mpk_prefill_packed_activation",
    )(metadata, inputs, packed, gains)[:original_rows]

def _make_prefill_persistent_sharded(mesh, c, context, tp, *, aux_layers=()):
    """TP-sharded prefill with tiled expert-weight expansion.

    Prompt rows are LEFT-padded to the bucket length; each row's KV is written
    to absolute slot (row - npad) with out-of-range writes dropped, so caches
    come back in the compacted [0, L_real) layout the decode kernel expects.
    Tokens may be [length] or [batch, length], with matching scalar or
    per-sequence padding. Batched logits are [tp, batch, vocab_shard].
    """
    qh = c.heads // tp
    kvh = c.kv_heads // tp
    mh_r = c.moe_hidden // tp
    vshard = c.vocab // tp
    hosts = tuple(tuple(range(i, i + 8)) for i in range(0, tp, 8))
    pairs = tuple(tuple(range(i, tp, 8)) for i in range(8))

    def local(w, states, tokens, npad):
        w = jax.tree.map(lambda a: a[0], w)
        states = jax.tree.map(lambda a: a[0], states)
        combined = "kvcache" in states
        rank = jax.lax.axis_index("tp")
        batched = tokens.ndim == 2
        batch = tokens.shape[0] if batched else 1
        length = tokens.shape[-1]
        L = batch * length
        tokens = tokens.reshape(L)
        pads = jnp.broadcast_to(npad, (batch,))
        positions = (jnp.arange(length, dtype=jnp.int32)[None] - pads[:, None]).reshape(L)
        real = positions >= 0
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

            if combined:
                def store(cache, values, positions):
                    return cache.at[positions].set(values.astype(cache.dtype), mode="drop")
                values = jnp.concatenate((key, value), axis=-1).reshape(batch, length, kvh * 128)
                cache = jax.vmap(store)(kc_all[li], values, slots.reshape(batch, length))
                kc_all = kc_all.at[li].set(cache)
            else:
                kc = kc_all[li]
                vc = vc_all[li]
                key_p = jnp.pad(key, ((0, 0), (0, 0), (0, kc.shape[-1] - c.head_dim)))
                value_p = jnp.pad(value, ((0, 0), (0, 0), (0, kc.shape[-1] - c.head_dim)))
                def store(cache, values, positions):
                    return cache.at[positions].set(values[:, 0].astype(cache.dtype), mode="drop")
                kc = jax.vmap(store)(kc, key_p.reshape(batch, length, kvh, -1), slots.reshape(batch, length))
                vc = jax.vmap(store)(vc, value_p.reshape(batch, length, kvh, -1), slots.reshape(batch, length))
                kc_all = kc_all.at[li].set(kc)
                vc_all = vc_all.at[li].set(vc)

            def attend_one(inputs):
                query, key, value, npad = inputs
                L = length
                real = jnp.arange(L) >= npad
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
                    blocks = []
                    block_q = min(1024, L)
                    for start in range(0, L, block_q):
                        end = min(start + block_q, L)
                        grouped = query[start:end].transpose(1, 0, 2).reshape(kvh, qh // kvh, end - start, c.head_dim)
                        keys_t = key[:end].transpose(1, 0, 2)
                        values_t = value[:end].transpose(1, 0, 2)
                        scores = jnp.einsum("gqld,gtd->gqlt", grouped.astype(F32), keys_t.astype(F32)) * c.scale
                        keep = (jnp.arange(end)[None, :] <= jnp.arange(start, end)[:, None]) & (jnp.arange(end)[None, :] >= npad)
                        scores = jnp.where(keep[None, None], scores, -jnp.inf)
                        scores = jnp.where(real[None, None, start:end, None], scores, 0.0)
                        probs = jax.nn.softmax(scores, axis=-1).astype(jnp.bfloat16)
                        out = jnp.einsum("gqlt,gtd->gqld", probs.astype(F32), values_t.astype(F32))
                        blocks.append(out.reshape(qh, end - start, c.head_dim).transpose(1, 0, 2))
                    return jnp.concatenate(blocks, axis=0)

                if L > c.sliding_window:
                    out = jax.lax.cond(sliding, window_attention, full_attention)
                else:
                    out = full_attention()
                return out
            if batched:
                out = jax.lax.map(attend_one, (query.reshape(batch, length, qh, c.head_dim),
                    key.reshape(batch, length, kvh, c.head_dim), value.reshape(batch, length, kvh, c.head_dim), pads)).reshape(L, qh, c.head_dim)
            else:
                out = attend_one((query, key, value, pads[0]))
            out = rms_plain_shard(out, c.eps) * jax.nn.sigmoid(gate.astype(F32))
            out = out.astype(jnp.bfloat16).reshape(L, qh * c.head_dim)
            out = jax.lax.all_gather(out, "tp", axis=-1, tiled=True, axis_index_groups=hosts)
            partial = jnp.dot(out, w["ow_group8"][li], preferred_element_type=F32)
            branch = jax.lax.psum(partial, "tp", axis_index_groups=pairs).astype(jnp.bfloat16)
            branch = jax.lax.all_gather(branch, "tp", axis=-1, tiled=True, axis_index_groups=hosts)
            stream = sc[3] * stream + sc[4] * rms_plain_f32(branch, c.post_eps)
            stream = jnp.where(real[:, None], stream, jnp.zeros((), F32))

            x = rms_gain_shard(stream, sc[1], c.eps)
            rlogits = jnp.dot(x.astype(F32), w["router"][li], precision=jax.lax.Precision.HIGHEST)
            rfull = jax.lax.all_gather(rlogits, "tp", axis=-1, tiled=True)
            scores_r = jax.nn.sigmoid(rfull * c.output_multiplier)
            ranked = scores_r + w["router_bias"][li].astype(F32)
            ids = _prefill_router_topk(ranked, c.top_k)
            rw = jnp.take_along_axis(scores_r, ids, axis=-1)
            rw = rw / (jnp.sum(rw, axis=-1, keepdims=True) + c.router_eps)

            pooled_part = jnp.dot(x, w["pool_local"][li], preferred_element_type=F32).astype(jnp.bfloat16)
            pooled = jax.lax.all_gather(pooled_part, "tp", axis=-1, tiled=True, axis_index_groups=hosts)
            pooled = rms_gain_shard(pooled, w["expert_norms"][li, 0], c.eps)

            order = jnp.argsort(ids.reshape(-1), stable=True)
            group_sizes = jnp.bincount(ids.reshape(-1), length=c.experts).astype(jnp.int32)
            sorted_inputs = pooled[order // c.top_k]

            def activation(gu):
                gate_e, up_e = jnp.split(gu, 2, axis=-1)
                act = (jax.nn.silu(gate_e.astype(F32)) * up_e.astype(F32)).astype(jnp.bfloat16)
                return jax.lax.all_gather(act, "tp", axis=-1, tiled=True)

            def dense_experts():
                down = _prefill_exchange_down(w["dw_dense"][o_dense], tp)
                down = down.transpose(1, 0, 2, 3).reshape(c.experts, c.moe_intermediate, mh_r)
                gu = jax.lax.ragged_dot(sorted_inputs, w["gu_dense"][o_dense], group_sizes, preferred_element_type=F32).astype(jnp.bfloat16)
                return jax.lax.ragged_dot(activation(gu), down, group_sizes, preferred_element_type=F32).astype(jnp.bfloat16)

            def packed_experts():
                act = _prefill_packed_activation(sorted_inputs, w["gu_pack"][o_packed], w["gu_global"][o_packed], group_sizes,
                    shard_width=c.moe_intermediate // tp, tile_rows=256, tile_columns=2 * (c.moe_intermediate // tp))
                act = jax.lax.all_gather(act, "tp", axis=-1, tiled=True)
                dp = _prefill_exchange_down(w["dw_pack"][o_packed], tp)
                down = _prefill_expand_down(dp, w["dw_global"][o_packed])
                return jax.lax.ragged_dot(act, down, group_sizes, preferred_element_type=F32).astype(jnp.bfloat16)

            sorted_rows = jax.lax.cond(is_dense, dense_experts, packed_experts)
            inverse = jnp.zeros_like(order).at[order].set(jnp.arange(order.size, dtype=jnp.int32), unique_indices=True)
            pair_rows = sorted_rows[inverse].reshape(L, c.top_k, mh_r)
            gain = jax.lax.dynamic_slice(w["expert_norms"][li, 1], (rank * mh_r,), (mh_r,))
            rows = pair_rows.astype(F32) * gain.astype(F32)
            squares = jax.lax.psum(jnp.sum(rows * rows, axis=-1, keepdims=True), "tp")
            rows = rows * jax.lax.rsqrt(squares / c.moe_hidden + c.post_eps)
            mixed = jnp.sum(rows * rw.astype(F32)[..., None], axis=-2).astype(jnp.bfloat16)
            mixed = jax.lax.all_gather(mixed, "tp", axis=-1, tiled=True)
            branch = jnp.dot(mixed, w["unpool_group8"][li], preferred_element_type=F32).astype(jnp.bfloat16)
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
                return (stream, kc_all, vc_all, captured), None
            return (stream, kc_all, vc_all), None

        carry = (stream, states["kvcache"], jnp.zeros((), jnp.int32)) if combined else (stream, states["kcache"], states["vcache"])
        if aux_layers:
            carry += (jnp.zeros((L, len(aux_layers), c.dim), jnp.bfloat16),)
        (stream, kc_all, vc_all, *aux), _ = jax.lax.scan(one_layer, carry, jnp.arange(c.layers))
        last = stream[(jnp.arange(batch) + 1) * length - 1]
        hidden = rms_gain_shard(last, w["fnorm"][0], c.eps)
        logits = jnp.dot(hidden, w["lmw"], preferred_element_type=F32)
        logits = logits * c.output_multiplier
        logits = c.soft_cap * jnp.tanh(logits / c.soft_cap)
        new_states = {"kvcache": kc_all} if combined else {"kcache": kc_all, "vcache": vc_all}
        result = (jax.tree.map(lambda a: a[None], new_states), logits[None] if batched else logits)
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
        return result

    return prefill


def _prefill_flash(
    query, kv, offset, padding, sliding, *, scale, window, query_block=512, key_block=512
):
    """Tiled GQA with FP32 score scaling and softmax accumulation."""
    heads, length, depth = query.shape
    kv_heads, key_length, _ = kv.shape
    group = heads // kv_heads
    qlength = pl.cdiv(length, query_block) * query_block
    klength = pl.cdiv(key_length, key_block) * key_block
    query = jnp.pad(query, ((0, 0), (0, qlength - length), (0, 0)))
    kv = jnp.pad(kv, ((0, 0), (0, klength - key_length), (0, 0)))
    control = jnp.stack((offset, padding, sliding.astype(jnp.int32)))

    def kernel(control, q, kv, out, buffers, sem):
        head, block = pl.program_id(0), pl.program_id(1)
        qstart = control[0] + block * query_block
        qpositions = qstart + jnp.arange(query_block)
        qpositions = jnp.tile(qpositions, group)[:, None]
        begin = jnp.where(control[2] != 0, jnp.maximum((qstart - window + 1) // key_block, 0), 0)
        end = jnp.minimum(pl.cdiv(qstart + query_block, key_block), klength // key_block)
        queries = q[...].reshape(group * query_block, depth)

        def transfer(index):
            return tpu.make_async_copy(
                kv.at[head, pl.ds(index * key_block, key_block)],
                buffers.at[index % 2],
                sem.at[index % 2],
            )

        transfer(begin).start()

        @pl.when(begin + 1 < end)
        def second():
            transfer(begin + 1).start()

        def step(index, carry):
            maximum, total, accumulator = carry
            transfer(index).wait()
            data = buffers[index % 2]
            scores = jnp.dot(queries, data[:, :depth].T, preferred_element_type=F32) * scale
            positions = index * key_block + jnp.arange(key_block)[None]
            keep = (positions <= qpositions) & (positions >= control[1]) & (positions < key_length)
            keep &= (control[2] == 0) | (positions > qpositions - window)
            scores = jnp.where(keep, scores, -1e30)
            current = jnp.maximum(maximum, jnp.max(scores, axis=-1, keepdims=True))
            correction = jnp.exp(maximum - current)
            probability = jnp.exp(scores - current)
            total = correction * total + jnp.sum(probability, axis=-1, keepdims=True)
            accumulator = correction * accumulator + jnp.dot(
                probability.astype(jnp.bfloat16), data[:, depth:], preferred_element_type=F32
            )

            @pl.when(index + 2 < end)
            def following():
                transfer(index + 2).start()

            return current, total, accumulator

        shape = (group * query_block, 1)
        _, total, accumulator = jax.lax.fori_loop(
            begin,
            end,
            step,
            (
                jnp.full(shape, -1e30, F32),
                jnp.zeros(shape, F32),
                jnp.zeros((group * query_block, depth), F32),
            ),
        )
        value = jnp.where(
            (qpositions >= control[1]) & (qpositions < control[0] + length), accumulator / total, 0
        )
        out[...] = value.reshape(group, query_block, depth)

    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((heads, qlength, depth), F32),
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(
                pl.BlockSpec((group, query_block, depth), lambda h, q, s: (h, q, 0)),
                pl.BlockSpec(memory_space=tpu.HBM),
            ),
            out_specs=pl.BlockSpec((group, query_block, depth), lambda h, q, s: (h, q, 0)),
            grid=(kv_heads, qlength // query_block),
            scratch_shapes=(
                tpu.VMEM((2, key_block, 2 * depth), jnp.bfloat16),
                tpu.SemaphoreType.DMA((2,)),
            ),
        ),
        compiler_params=tpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"), vmem_limit_bytes=64 * 1024**2
        ),
        name="mpk_prefill_flash",
    )(control, query, kv)
    return result[:, :length]


def _make_prefill_persistent_parallel(mesh, c, context, tp, *, aux_layers=()):
    """Shard prompt rows and redistribute one expert layer at a time."""
    vshard = c.vocab // tp
    expert_count = c.experts // tp
    hosts = tuple(tuple(range(i, i + 8)) for i in range(0, tp, 8))
    pairs = tuple(tuple(range(i, tp, 8)) for i in range(8))

    def gather_tokens(value):
        value = jax.lax.all_gather(value, "tp", axis=0, tiled=True, axis_index_groups=pairs)
        return jax.lax.all_gather(value, "tp", axis=0, tiled=True, axis_index_groups=hosts)

    def local(w, states, tokens, npad):
        w = jax.tree.map(lambda a: a[0], w)
        states = jax.tree.map(lambda a: a[0], states)
        rank = jax.lax.axis_index("tp")
        batch = tokens.shape[0] if tokens.ndim == 2 else 1
        length = tokens.shape[-1]
        sequence_parts = tp // batch
        local_length = length // sequence_parts
        total_length = batch * length
        prefetch_down = total_length >= 30720 and len(c.bf16_expert_layers) < c.layers
        sequence = rank // sequence_parts
        offset = (rank % sequence_parts) * local_length
        groups = tuple(
            tuple(range(i * sequence_parts, (i + 1) * sequence_parts)) for i in range(batch)
        )
        pads = jnp.broadcast_to(npad, (batch,))
        all_positions = jnp.arange(length, dtype=jnp.int32)[None] - pads[:, None]
        positions = offset + jnp.arange(local_length, dtype=jnp.int32) - pads[sequence]
        real = positions >= 0
        flat_tokens = tokens.reshape(-1)
        lo = rank * vshard
        emb = w["emb"][jnp.clip(flat_tokens - lo, 0, vshard - 1)]
        if batch == 8:
            emb = jnp.where(((flat_tokens >= lo) & (flat_tokens < lo + vshard))[:, None], emb, 0)
            stream = jax.lax.psum_scatter(emb, "tp", scatter_dimension=0, tiled=True).astype(F32)
        else:
            emb = jnp.where(
                ((flat_tokens >= lo) & (flat_tokens < lo + vshard))[:, None], emb.astype(F32), 0
            )
            emb = jax.lax.psum(emb, "tp")
            stream = jax.lax.dynamic_slice(emb, (rank * local_length, 0), (local_length, c.dim))
        stream = rms_plain_f32(stream, c.eps)
        stream = jnp.where(real[:, None], stream, 0)
        cos, sin = rope_tables_shard(c, positions)

        def one_layer(carry, li):
            if prefetch_down:
                stream, kv_all, exchanged_down, *captured = carry
            else:
                stream, kv_all, *captured = carry
            dense = jnp.bool_(False)
            below = jnp.int32(0)
            for d in c.bf16_expert_layers:
                dense |= li == d
                below += (li > d).astype(jnp.int32)
            packed_index = jnp.clip(li - below, 0, c.layers - len(c.bf16_expert_layers) - 1)
            dense_index = jnp.clip(below, 0, len(c.bf16_expert_layers) - 1)
            if prefetch_down:
                following = li + 1
                prior_dense = sum((following > d).astype(jnp.int32) for d in c.bf16_expert_layers)
                next_packed = jnp.clip(
                    following - prior_dense, 0, c.layers - len(c.bf16_expert_layers) - 1
                )
                next_down = jax.lax.all_to_all(
                    w["dw_pack"][next_packed], "tp", split_axis=0, concat_axis=1, tiled=True
                )
            sliding = li % 4 != 1
            sc = w["scalars"][li]
            qweight = jax.lax.all_gather(w["qwkv"][li], "tp", axis=-1, tiled=True)
            x = rms_gain_shard(stream, sc[0], c.eps)
            qkv = (
                jnp.dot(x, qweight, preferred_element_type=F32)
                .astype(jnp.bfloat16)
                .reshape(local_length, tp, -1)
            )
            qwidth = c.heads // tp * c.head_dim
            query = qkv[..., :qwidth].reshape(local_length, c.heads, c.head_dim)
            gate = qkv[..., qwidth : 2 * qwidth].reshape(local_length, c.heads, c.head_dim)
            kvwidth = c.kv_heads // tp * c.head_dim
            key = qkv[..., 2 * qwidth : 2 * qwidth + kvwidth].reshape(
                local_length, c.kv_heads, c.head_dim
            )
            value = qkv[..., 2 * qwidth + kvwidth :].reshape(
                local_length, c.kv_heads, c.head_dim
            )
            query = rms_plain_shard(query, c.eps)
            key = rms_plain_shard(key, c.eps)
            query = jnp.where(
                sliding, apply_rope_shard(query, cos, sin).astype(F32), query.astype(F32)
            ).astype(jnp.bfloat16)
            key = jnp.where(
                sliding, apply_rope_shard(key, cos, sin).astype(F32), key.astype(F32)
            ).astype(jnp.bfloat16)
            kv = jnp.concatenate((key, value), axis=-1)
            full_kv = jax.lax.all_gather(kv, "tp", axis=0, tiled=True, axis_index_groups=groups)
            if tp == 16:
                cache_kv = jax.lax.all_to_all(
                    kv.transpose(1, 0, 2), "tp", split_axis=0, concat_axis=0, tiled=False
                ).reshape(batch, length, 128)
            else:
                cache_kv = jax.lax.all_to_all(
                    kv.transpose(1, 0, 2).reshape(tp, c.kv_heads // tp, local_length, 128),
                    "tp", split_axis=0, concat_axis=0, tiled=False
                ).transpose(0, 2, 1, 3).reshape(batch, length, c.kv_heads // tp * 128)

            def store(cache, values, positions):
                return cache.at[jnp.clip(positions, -context, context - 1)].set(
                    values.astype(cache.dtype), mode="drop"
                )

            kv_all = kv_all.at[li].set(jax.vmap(store)(kv_all[li], cache_kv, all_positions))
            out = _prefill_flash(
                query.transpose(1, 0, 2),
                full_kv.transpose(1, 0, 2),
                offset,
                pads[sequence],
                sliding,
                scale=c.scale,
                window=c.sliding_window,
            ).transpose(1, 0, 2)
            out = (
                (rms_plain_shard(out, c.eps) * jax.nn.sigmoid(gate.astype(F32)))
                .astype(jnp.bfloat16)
                .reshape(local_length, c.dim)
            )
            oweight = jax.lax.all_gather(w["ow"][li], "tp", axis=0, tiled=True)
            branch = jnp.dot(out, oweight, preferred_element_type=F32).astype(jnp.bfloat16)
            stream = sc[3] * stream + sc[4] * rms_plain_f32(branch, c.post_eps)
            stream = jnp.where(real[:, None], stream, 0)
            x = rms_gain_shard(stream, sc[1], c.eps)
            router = jax.lax.all_gather(w["router"][li], "tp", axis=-1, tiled=True)
            rlogits = jnp.dot(x.astype(F32), router, precision=jax.lax.Precision.HIGHEST)
            scores = jax.nn.sigmoid(rlogits * c.output_multiplier)
            ids = _prefill_router_topk(scores + w["router_bias"][li].astype(F32), c.top_k)
            rw = jnp.take_along_axis(scores, ids, axis=-1)
            rw /= jnp.sum(rw, axis=-1, keepdims=True) + c.router_eps
            pool = jax.lax.all_gather(w["pool"][li], "tp", axis=-1, tiled=True)
            pooled = jnp.dot(x, pool, preferred_element_type=F32).astype(jnp.bfloat16)
            pooled = rms_gain_shard(pooled, w["expert_norms"][li, 0], c.eps)
            all_pooled = gather_tokens(pooled)
            all_ids = gather_tokens(ids)
            all_rw = gather_tokens(rw)

            def gu_exchange(weights):
                weights = jax.lax.all_to_all(weights, "tp", split_axis=0, concat_axis=2, tiled=True)
                return (
                    weights.reshape(expert_count, weights.shape[1], tp, 2, c.moe_intermediate // tp)
                    .transpose(0, 1, 3, 2, 4)
                    .reshape(expert_count, weights.shape[1], 2 * c.moe_intermediate)
                )

            def dense_weights():
                return gu_exchange(w["gu_dense"][dense_index]), jax.lax.all_to_all(
                    w["dw_dense"][dense_index], "tp", split_axis=0, concat_axis=1, tiled=True
                )

            def packed_weights():
                gp = jax.lax.all_to_all(
                    w["gu_pack"][packed_index], "tp", split_axis=0, concat_axis=2, tiled=True
                )
                dp = (
                    exchanged_down
                    if prefetch_down
                    else jax.lax.all_to_all(
                        w["dw_pack"][packed_index], "tp", split_axis=0, concat_axis=1, tiled=True
                    )
                )
                depth = c.moe_intermediate // tp
                dp = dp.reshape(expert_count, tp, depth * 9 // 16, c.moe_hidden)
                dv = dp[:, :, : depth // 2].reshape(
                    expert_count, c.moe_intermediate // 2, c.moe_hidden
                )
                ds = dp[:, :, depth // 2 :].reshape(
                    expert_count, c.moe_intermediate // 16, c.moe_hidden
                )
                dg = jax.lax.dynamic_slice(
                    w["dw_global"][packed_index], (rank * expert_count,), (expert_count,)
                )
                down = _dequant_experts_tiled(
                    dv, ds, dg, expert_count, c.moe_intermediate, c.moe_hidden
                )
                return gp, down

            def routed_experts(packed):
                gu, down = packed_weights() if packed else dense_weights()
                gains = jax.lax.dynamic_slice(
                    w["gu_global"][packed_index], (rank * expert_count, 0), (expert_count, 2)
                )
                order = jnp.argsort(all_ids.reshape(-1), stable=True)
                sizes = jnp.bincount(all_ids.reshape(-1), length=c.experts).astype(jnp.int32)
                totals = jnp.sum(sizes.reshape(tp, expert_count), axis=1)
                base = jnp.sum(jnp.where(jnp.arange(tp) < rank, totals, 0))
                local_sizes = jax.lax.dynamic_slice(sizes, (rank * expert_count,), (expert_count,))
                ends = jnp.cumsum(local_sizes)
                starts = ends - local_sizes
                capacity = pl.cdiv(total_length, 512) * 512
                from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm

                def expert_chunk(chunk, mixed):
                    begin = chunk * capacity
                    counts = jnp.maximum(
                        jnp.minimum(ends - begin, capacity) - jnp.maximum(starts - begin, 0), 0
                    )
                    live = jnp.sum(counts)
                    counts = jnp.concatenate((counts, (capacity - live)[None]))
                    indices = order[
                        jnp.minimum(base + begin + jnp.arange(capacity), order.size - 1)
                    ]
                    inputs = all_pooled[indices // c.top_k]
                    if packed:
                        act = _prefill_packed_activation(
                            inputs,
                            gu,
                            gains,
                            counts,
                            shard_width=c.moe_intermediate // tp,
                            tile_rows=512,
                            tile_columns=4096,
                        )
                    else:
                        projected = gmm(
                            inputs,
                            gu,
                            counts,
                            preferred_element_type=jnp.bfloat16,
                            tiling=(512, 4096, 1024),
                        )
                        gate_e, up_e = jnp.split(projected, 2, axis=-1)
                        act = (jax.nn.silu(gate_e.astype(F32)) * up_e.astype(F32)).astype(
                            jnp.bfloat16
                        )
                    output = gmm(
                        act,
                        down,
                        counts,
                        preferred_element_type=jnp.bfloat16,
                        tiling=(512, 4096, 1024),
                    )
                    output = rms_gain_first_f32(output, w["expert_norms"][li, 1], c.post_eps)
                    output = jnp.where(
                        (jnp.arange(capacity) < live)[:, None],
                        output * all_rw.reshape(-1)[indices, None],
                        0,
                    )
                    return mixed.at[indices // c.top_k].add(output)

                mixed = jax.lax.fori_loop(
                    0,
                    pl.cdiv(jnp.max(totals), capacity),
                    expert_chunk,
                    jnp.zeros((total_length, c.moe_hidden), F32),
                )
                mixed = jax.lax.psum_scatter(
                    mixed, "tp", scatter_dimension=0, tiled=True, axis_index_groups=hosts
                )
                mixed = jax.lax.psum_scatter(
                    mixed, "tp", scatter_dimension=0, tiled=True, axis_index_groups=pairs
                ).astype(jnp.bfloat16)
                return mixed

            mixed = jax.lax.cond(dense, lambda: routed_experts(False), lambda: routed_experts(True))
            unpool = jax.lax.all_gather(w["unpool"][li], "tp", axis=0, tiled=True)
            branch = jnp.dot(mixed, unpool, preferred_element_type=F32).astype(jnp.bfloat16)
            stream = sc[5] * stream + sc[6] * rms_gain_first_f32(branch, sc[2], c.post_eps)
            stream = jnp.where(real[:, None], stream, 0)
            next_carry = (stream, kv_all, next_down) if prefetch_down else (stream, kv_all)
            if aux_layers:
                saved = captured[0]
                for index, layer_id in enumerate(aux_layers):
                    saved = jax.lax.cond(
                        li == layer_id,
                        lambda a: a.at[:, index].set(stream.astype(jnp.bfloat16)),
                        lambda a: a,
                        saved,
                    )
                return (*next_carry, saved), None
            return next_carry, None

        combined = "kvcache" in states
        cache = (
            states["kvcache"]
            if combined
            else jnp.concatenate((states["kcache"][..., :64], states["vcache"][..., :64]), axis=-1)
        )
        carry = (stream, cache)
        if prefetch_down:
            initial_down = (
                jnp.zeros((expert_count, c.moe_intermediate * 9 // 16, c.moe_hidden), jnp.uint8)
                if 0 in c.bf16_expert_layers
                else jax.lax.all_to_all(
                    w["dw_pack"][0], "tp", split_axis=0, concat_axis=1, tiled=True
                )
            )
            carry += (initial_down,)
        if aux_layers:
            carry += (jnp.zeros((local_length, len(aux_layers), c.dim), jnp.bfloat16),)
        carry, _ = jax.lax.scan(one_layer, carry, jnp.arange(c.layers))
        stream, cache = carry[:2]
        captured = carry[3:] if prefetch_down else carry[2:]
        local_last = jnp.where((rank % sequence_parts == sequence_parts - 1), stream[-1], 0)
        all_last = (
            jax.lax.all_gather(local_last, "tp", axis=0, tiled=False)
            .reshape(batch, sequence_parts, c.dim)
            .sum(axis=1)
        )
        hidden = rms_gain_shard(all_last, w["fnorm"][0], c.eps)
        logits = jnp.dot(hidden, w["lmw"], preferred_element_type=F32) * c.output_multiplier
        logits = c.soft_cap * jnp.tanh(logits / c.soft_cap)
        new_states = (
            {"kvcache": cache}
            if combined
            else {
                "kcache": jnp.pad(cache[..., :64], [(0, 0)] * 3 + [(0, 64)]),
                "vcache": jnp.pad(cache[..., 64:], [(0, 0)] * 3 + [(0, 64)]),
            }
        )
        result = (
            jax.tree.map(lambda x: x[None], new_states),
            logits[None] if tokens.ndim == 2 else logits,
        )
        if aux_layers:
            aux = jax.lax.all_gather(captured[0], "tp", axis=0, tiled=True)
            return (*result, aux)
        return result

    def prefill(w, states, tokens, npad):
        ss = jax.tree.map(lambda _: P("tp"), states)
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(_tp_specs(w, tp), ss, P(), P()),
            out_specs=(ss, P("tp"), P()) if aux_layers else (ss, P("tp")),
            check_vma=False,
        )(w, states, tokens, npad)

    return prefill

def _make_prefill_persistent(mesh, c, context, tp, *, aux_layers=()):
    sharded = _make_prefill_persistent_sharded(mesh, c, context, tp, aux_layers=aux_layers)
    parallel = _make_prefill_persistent_parallel(mesh, c, context, tp, aux_layers=aux_layers)

    def prefill(weights, states, tokens, padding):
        batch = tokens.shape[0] if tokens.ndim == 2 else 1
        if batch in (2, 4, 8) and tokens.size >= 15360 and tokens.shape[-1] % (tp // batch) == 0:
            return parallel(weights, states, tokens, padding)
        return sharded(weights, states, tokens, padding)

    return prefill


_BLOCK_WEIGHTS = ("qwkv", "ow", "pool", "unpool", "router", "router_bias", "scalars", "expert_norms", "gu_pack", "dw_pack", "gu_dense", "dw_dense", "gu_global", "dw_global", "lmw", "fnorm")


def _router_order(experts):
    """Place ascending expert IDs in Mosaic's native argmax tie order."""
    columns = jnp.arange(experts, dtype=jnp.int32)
    if experts not in (128, 256):
        return columns
    return (127 - columns % 128) * (experts // 128) + columns // 128


def _router_argmax(ranked):
    experts = ranked.shape[-1]
    if experts in (128, 256):
        chosen = jnp.argmax(ranked, axis=1).astype(jnp.int32)[:, None]
        return (127 - chosen % 128) * (experts // 128) + chosen // 128
    maximum = jnp.max(ranked, axis=1, keepdims=True)
    return jnp.min(jnp.where(ranked == maximum, jnp.arange(experts)[None], experts), axis=1, keepdims=True)


def _make_decode_block(mesh, c, context, rows, aux_layers, *, batch_size=0, windowed=False):
    """Decode or verify with expert weights sharded across all TP8/TP16 ranks.

    Isolate DMA counters for phases with different peer maps or message sizes.
    """
    tp = mesh.size
    if tp not in (8, 16) or rows not in (1, 8) or c.head_dim != 64:
        raise ValueError("persistent verification requires TP8/TP16 and one or eight rows with 64-wide heads")
    if rows == 1 and c.top_k != 8:
        raise ValueError("single-token TP8/TP16 decoding requires eight routed experts")
    if batch_size and (batch_size not in (2, 4, 8) or rows != 8):
        raise ValueError("batched TP8/TP16 decoding requires two, four, or eight sequences and eight rows")
    aux_layers = aux_layers or (c.layers - 1,)
    if len(set(aux_layers)) != len(aux_layers) or any(i < 0 or i >= c.layers for i in aux_layers):
        raise ValueError("auxiliary layer indices must be unique target layers")
    h, mh, nh = c.dim, c.moe_hidden, c.heads // tp
    iw, di = c.moe_intermediate // tp, c.moe_intermediate // tp
    nkv = c.kv_heads // tp
    kvwidth = 128 * nkv
    tp8_verify = tp == 8 and rows == 8 and not batch_size
    prefetch = 2 if tp == 8 else 4
    qwidth, pw, es = nh * 64, mh // 8, c.experts // 8
    output_in_q = tp8_verify and context * kvwidth <= h * (qwidth + kvwidth)
    overlap_cache = tp8_verify and context <= 4096
    active_rows = batch_size or rows
    combine_expert_waits = rows == 8 and not batch_size and not tp8_verify and mh % 256 == 0 and iw % 128 == 0 and c.experts >= 3
    exchange_chunks = c.top_k if rows == 8 and not batch_size and mh % 2048 == 0 else 1
    remote_channels = 61 if rows == 1 else 53 + 8 * (exchange_chunks - 1)
    router_rows, pool_rows = 2 * active_rows, active_rows * pw // 128
    id_count = 128 if rows == 1 else 256 + rows * 128
    qshape, oshape = (h, 2 * qwidth + kvwidth), ((qwidth, h) if rows == 1 else (8 * qwidth, h // 8))
    o_chunks = 2 if tp == 8 and rows == 8 and not output_in_q else 1
    oshape = (oshape[0], oshape[1] // o_chunks)
    output_chunks = 4 if rows == 1 and h % 4096 == 0 else 1
    pshape, ushape = (h, pw), (mh, h // 8)
    gu_batch = 1 if rows == 1 or tp8_verify else 2
    packed_prefetch = 4
    gpshape, dpshape = (packed_prefetch // gu_batch, mh * 9 // 16, 2 * gu_batch * iw), (packed_prefetch, iw * 9 // 16, mh)
    gdshape, ddshape = (prefetch // gu_batch, mh, 2 * gu_batch * di), (prefetch, di, mh)

    def tiles(shape, dtype):
        return pool_alias.storage_size(shape, jnp.dtype(dtype)) // 4096

    gt = max(tiles(gpshape, jnp.uint8), tiles(gdshape, jnp.bfloat16))
    dt = max(tiles(dpshape, jnp.uint8), tiles(ddshape, jnp.bfloat16))
    head_width = lm_width(c, tp)
    lm_chunk = min(640, head_width // 2)
    lm_shape = (2, h, lm_chunk)
    q_tiles = tiles(qshape, jnp.bfloat16)
    cache_shape = (batch_size, context, kvwidth) if batch_size else (context, kvwidth)
    cache_tiles = tiles(cache_shape, jnp.bfloat16)
    cache_offset = 0 if tp == 8 and not overlap_cache else q_tiles
    qt = max(q_tiles, cache_offset + cache_tiles, gt + dt, tiles(lm_shape, jnp.bfloat16),
             q_tiles + tiles((3 * rows + 2 if rows == 8 else 10, h // 128, 128), jnp.float32))
    pt = max(tiles(shape, jnp.bfloat16) for shape in ((pshape, ushape) if output_in_q else (oshape, pshape, ushape)))

    def kernel(meta, *args):
        w = dict(zip(_BLOCK_WEIGHTS, args[:len(_BLOCK_WEIGHTS)]))
        (initial, kvc_in, final, aux, kvc, qa, pa, router, scalars, norms, bias,
         gg, dg, pair, ahost, apaired, comb, estage, epipe, rowhost, mixed_rows,
         ungather, ids_v, ids_s, candidates, expert_results, sem, sends, recvs) = args[len(_BLOCK_WEIGHTS):]
        kv_delta = kvc
        if tp == 8:
            kvc = kvc_in
        qbuf = pool_alias.view(qa.at[:q_tiles], qshape, jnp.bfloat16)
        kvbuf = pool_alias.view(qa.at[cache_offset:cache_offset + cache_tiles], cache_shape, jnp.bfloat16)
        gp = pool_alias.view(qa.at[:tiles(gpshape, jnp.uint8)], gpshape, jnp.uint8)
        dp_storage = (dpshape[0], (dpshape[1] + 31) // 32 * 32, dpshape[2])
        dp = pool_alias.view(qa.at[gt:gt + tiles(dpshape, jnp.uint8)], dp_storage, jnp.uint8).at[:, :dpshape[1]]
        gd = pool_alias.view(qa.at[:tiles(gdshape, jnp.bfloat16)], gdshape, jnp.bfloat16)
        dd = pool_alias.view(qa.at[gt:gt + tiles(ddshape, jnp.bfloat16)], ddshape, jnp.bfloat16)
        obuf = pool_alias.view(qa.at[cache_tiles:cache_tiles + tiles(oshape, jnp.bfloat16)] if output_in_q else pa.at[:tiles(oshape, jnp.bfloat16)], oshape, jnp.bfloat16)
        pbuf = pool_alias.view(pa.at[:tiles(pshape, jnp.bfloat16)], pshape, jnp.bfloat16)
        ubuf = pool_alias.view(pa.at[:tiles(ushape, jnp.bfloat16)], ushape, jnp.bfloat16)
        lm = pool_alias.view(qa.at[:tiles(lm_shape, jnp.bfloat16)], lm_shape, jnp.bfloat16)
        rank = jax.lax.axis_index("tp")
        lane, host = rank % 8, rank // 8
        owner = lane % rows
        r = jnp.arange(rows, dtype=jnp.int32)[:, None]
        if batch_size:
            pos = jnp.maximum(jnp.stack([meta[i % batch_size] for i in range(rows)])[:, None], 0)
        else:
            pos = meta[0] + jnp.minimum(r, meta[1] - 1)
            write_start = jnp.minimum(meta[0] // 16 * 16, context - 32)
            wr = write_start + jnp.arange(32)[:, None]
        cache_positions = jnp.arange(context)[None]
        columns = jnp.arange(128, dtype=jnp.int32)[None].astype(jnp.float32)
        angle = pos.astype(jnp.float32) / c.rope_theta ** ((columns % 32) / 32)
        cosine, sine = jnp.cos(angle), jnp.sin(angle)
        experts = jnp.arange(c.experts)[None]
        router_ids = _router_order(c.experts)[None]

        def copy(src, dst, channel):
            return tpu.make_async_copy(src, dst, sem.at[channel])

        def output_copy(li, chunk=0):
            source = w["ow"].at[li] if o_chunks == 1 else w["ow"].at[li, :, pl.ds(chunk * oshape[1], oshape[1])]
            return copy(source, obuf, 1)

        def remote(src, dst, channel, peer):
            return tpu.make_async_remote_copy(src, dst, sends.at[channel], recvs.at[channel], device_id=(peer,), device_id_type=pl.DeviceIdType.MESH)

        def gather(buf, index, offsets, channel):
            if rows == 1 and len(offsets) == 7:
                for offset in offsets:
                    remote(buf.at[index], buf.at[index], channel, rank ^ offset).start()
                remote(buf.at[:7], buf.at[:7], channel, rank ^ offsets[0]).wait()
            else:
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

        copy(w["qwkv"].at[0], qbuf, 0).start()
        if not output_in_q:
            output_copy(0).start()
        copy(w["scalars"].at[0], scalars, 3).start()
        barrier = tpu.get_barrier_semaphore()
        for offset in range(1, tp):
            pl.semaphore_signal(barrier, 1, device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH)
        pl.semaphore_wait(barrier, tp - 1)
        if rows == 8:
            local_tiles = tiles((8, h // 128, 128), jnp.float32)
            host_tiles = tiles((2, h // 128, 128), jnp.float32)
            parts = pool_alias.view(qa.at[q_tiles:q_tiles + local_tiles], (8, h // 128, 128), jnp.float32)
            hosts = pool_alias.view(qa.at[q_tiles + local_tiles:q_tiles + local_tiles + host_tiles], (2, h // 128, 128), jnp.float32)
            joined = pool_alias.view(qa.at[q_tiles + local_tiles + host_tiles:q_tiles + 2 * local_tiles + host_tiles], (8, h // 128, 128), jnp.float32)
            embed_tiles = tiles((rows, h // 128, 128), jnp.float32)
            source = pool_alias.view(qa.at[q_tiles + 2 * local_tiles + host_tiles:q_tiles + 2 * local_tiles + host_tiles + embed_tiles], (rows, h // 128, 128), jnp.float32)
            source[...] = initial[...].reshape(rows, h // 128, 128)
            parts[lane] = source[lane]
            transfers = [remote(source.at[lane ^ offset], parts.at[lane], offset - 1, rank ^ offset) for offset in range(7, 0, -1)]
            for transfer in transfers:
                transfer.start()
            for transfer in transfers:
                transfer.wait()
            hosts[host] = jnp.sum(parts[...], axis=0)
            if tp == 16:
                gather(hosts, host, (8,), 8)
            joined[lane] = jnp.sum(hosts[:tp // 8], axis=0)
            gather(joined, lane, tuple(range(7, 0, -1)), 36)
            initial_stream = joined[...].reshape(rows, h)
        else:
            local_tiles = tiles((8, h // 128, 128), jnp.float32)
            host_tiles = tiles((2, h // 128, 128), jnp.float32)
            parts = pool_alias.view(qa.at[q_tiles:q_tiles + local_tiles], (8, h // 128, 128), jnp.float32)
            hosts = pool_alias.view(qa.at[q_tiles + local_tiles:q_tiles + local_tiles + host_tiles], (2, h // 128, 128), jnp.float32)
            parts[lane] = initial[...].reshape(h // 128, 128)
            gather(parts, lane, tuple(range(7, 0, -1)), 53)
            hosts[host] = jnp.sum(parts[...], axis=0)
            if tp == 16:
                gather(hosts, host, (8,), 60)
            initial_stream = jnp.sum(hosts[:tp // 8], axis=0).reshape(1, h)
        aux[...] = jnp.zeros(aux.shape, jnp.bfloat16)

        def layer(li, stream):
            dense = jnp.bool_(False)
            below = jnp.int32(0)
            for d in c.bf16_expert_layers:
                dense |= li == d
                below += (li > d).astype(jnp.int32)
            pi = jnp.clip(li - below, 0, c.layers - len(c.bf16_expert_layers) - 1)
            bi = jnp.clip(below, 0, len(c.bf16_expert_layers) - 1)
            sliding = li % 4 != 1
            needed = jnp.int32(0)
            if batch_size:
                for row in range(batch_size):
                    needed = jnp.maximum(needed, jnp.clip(meta[row] // 16 * 16, 0, context - 32) + 32)
            else:
                needed = write_start + 32
            cache_widths = [min(512, context)]
            while cache_widths[-1] < context:
                cache_widths.append(min(cache_widths[-1] * 2, context))
            window_width = min(context, pl.cdiv(c.sliding_window + 256 + 128 + 32, 128) * 128)
            first_position = jnp.min(pos)
            cache_start = pl.multiple_of(jnp.clip((first_position - c.sliding_window + 1) // 128 * 128, 0, context - window_width), 128)
            use_window = sliding & (cache_start > 0) & (needed <= cache_start + window_width)
            def cache_copy(wait):
                def transfer(width):
                    src = kvc.at[li, :, pl.ds(0, width)] if batch_size else kvc.at[li, 0, pl.ds(0, width)]
                    dst = kvbuf.at[:, pl.ds(0, width)] if batch_size else kvbuf.at[pl.ds(0, width)]
                    dma = copy(src, dst, 8)
                    if wait:
                        dma.wait()
                    else:
                        dma.start()
                def choose(index):
                    if index == len(cache_widths) - 1:
                        return transfer(cache_widths[index])
                    return jax.lax.cond(needed <= cache_widths[index], lambda: transfer(cache_widths[index]), lambda: choose(index + 1))
                def window_transfer():
                    src = kvc.at[li, :, pl.ds(cache_start, window_width)] if batch_size else kvc.at[li, 0, pl.ds(cache_start, window_width)]
                    dst = kvbuf.at[:, pl.ds(0, window_width)] if batch_size else kvbuf.at[pl.ds(cache_start, window_width)]
                    dma = copy(src, dst, 8)
                    if wait:
                        dma.wait()
                    else:
                        dma.start()
                if rows == 8 and not batch_size:
                    transfer(context)
                elif windowed:
                    jax.lax.cond(use_window, window_transfer, lambda: choose(0))
                else:
                    choose(0)
            loads = [(w["router"].at[li], router),
                     (w["scalars"].at[li], scalars), (w["expert_norms"].at[li], norms), (w["router_bias"].at[li], bias),
                     (w["gu_global"].at[pi], gg), (w["dw_global"].at[pi], dg)]
            for channel, (src, dst) in enumerate(loads, 2):
                if channel != 3:
                    copy(src, dst, channel).start()
            if tp == 16 or overlap_cache:
                cache_copy(False)
            copy(w["scalars"].at[li], scalars, 3).wait()
            sc = scalars[...]
            x = norm(stream, c.eps, sc[0:1]).astype(jnp.bfloat16)
            copy(w["qwkv"].at[li], qbuf, 0).wait()
            qgkv = dot(x, qbuf[...]).astype(jnp.bfloat16)
            kv_parts = []
            for kv_head in range(nkv):
                key_start = 2 * qwidth + kv_head * 64
                value_start = 2 * qwidth + (nkv + kv_head) * 64
                key = rotate(hn(jnp.pad(qgkv[:, key_start:key_start + 64], ((0, 0), (0, 64)))).astype(jnp.bfloat16), sliding)
                kv_parts.extend((key[:, :64], qgkv[:, value_start:value_start + 64]))
            values = jnp.concatenate(kv_parts, axis=1)
            if tp == 8 and not overlap_cache:
                cache_copy(False)
            if output_in_q:
                output_copy(li).start()
            cache_copy(True)
            if batch_size:
                for row in range(batch_size):
                    start = pl.multiple_of(jnp.clip(meta[row] // 16 * 16, 0, context - 32), 16)
                    buffer_start = pl.multiple_of(start - jnp.where(use_window, cache_start, 0), 16) if windowed else start
                    mask = (start + jnp.arange(32)[:, None] == meta[row]) & (meta[row] >= 0)
                    block = jnp.where(mask, values[row:row + 1], kvbuf[row, pl.ds(buffer_start, 32)])
                    kvbuf[row, pl.ds(buffer_start, 32)] = block
                    if tp == 16:
                        copy(kvbuf.at[row, pl.ds(buffer_start, 32)], kvc.at[li, row, pl.ds(start, 32)], 21 + row).start()
            else:
                block = kvbuf[pl.ds(write_start, 32)]
                for row in range(rows):
                    mask = jnp.broadcast_to((wr == meta[0] + row) & (row < meta[1]), (32, kvwidth))
                    block = jnp.where(mask, values[row:row + 1], block)
                kvbuf[pl.ds(write_start, 32)] = block
                if tp == 16:
                    copy(kvbuf.at[pl.ds(write_start, 32)], kvc.at[li, 0, pl.ds(write_start, 32)], 10).start()
            if tp == 8:
                kv_delta[li] = values
            keep = (cache_positions <= pos) & ((cache_positions > pos - c.sliding_window) | ~sliding)
            queries = []
            for head in range(nh):
                q = rotate(hn(jnp.pad(qgkv[:, head * 64:(head + 1) * 64], ((0, 0), (0, 64)))).astype(jnp.bfloat16), sliding)
                queries.append(q)
            query = jnp.concatenate(queries, axis=0)

            def attention(width, query, cache, keep):
                scores = jnp.where((jnp.tile(keep[:, :width].astype(jnp.int32), (nh // nkv, 1)) != 0), dot(query, cache[:width].T) * c.scale, -1e30)
                exponent = jnp.exp(scores - jnp.max(scores, axis=1, keepdims=True)).astype(jnp.bfloat16)
                value = (dot(exponent, cache[:width]) / jnp.sum(exponent.astype(jnp.float32), axis=1, keepdims=True)).astype(jnp.bfloat16)
                return jnp.pad(value[:, 64:], ((0, 0), (0, 64)))

            widths = [min(512, context)]
            while widths[-1] < context:
                widths.append(min(widths[-1] * 2, context))
            def select(index, query, cache, keep, end):
                if index == len(widths) - 1:
                    return attention(widths[index], query, cache, keep)
                return jax.lax.cond(end <= widths[index], lambda: attention(widths[index], query, cache, keep), lambda: select(index + 1, query, cache, keep, end))
            def attend(query, cache, keep, query_pos, end):
                def window():
                    slots = cache_start + jnp.arange(window_width)[None]
                    mask = (slots <= query_pos) & (slots > query_pos - c.sliding_window)
                    return attention(window_width, query, cache.at[pl.ds(cache_start, window_width)], mask)
                if windowed:
                    return jax.lax.cond(use_window, window, lambda: select(0, query, cache, keep, end))
                return select(0, query, cache, keep, end)
            attended_kv = []
            for kv_group in range(nkv):
                kv_cache = kvbuf if nkv == 1 else kvbuf.at[:, :, kv_group * 128:(kv_group + 1) * 128] if batch_size else kvbuf.at[:, kv_group * 128:(kv_group + 1) * 128]
                if batch_size and windowed:
                    attention_batch = min(8, batch_size)
                    packed_queries = jnp.stack(queries[kv_group * (nh // nkv):(kv_group + 1) * (nh // nkv)], axis=1)[:batch_size]
                    attended_groups = []
                    for begin in range(0, batch_size, attention_batch):
                        query_group = packed_queries[begin:begin + attention_batch]
                        positions = pos[begin:begin + attention_batch]
                        def batched_attention(width, offset):
                            data = jnp.stack([kv_cache.at[begin + i][:width] for i in range(attention_batch)])
                            scores = jax.lax.dot_general(query_group, data, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.float32) * c.scale
                            slots = offset + jnp.arange(width)[None]
                            keep_group = (slots <= positions) & ((slots > positions - c.sliding_window) | ~sliding)
                            scores = jnp.where(keep_group[:, None, :], scores, -1e30)
                            exponent = jnp.exp(scores - jnp.max(scores, axis=2, keepdims=True)).astype(jnp.bfloat16)
                            result = jax.lax.dot_general(exponent, data, (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32)
                            result = (result / jnp.sum(exponent.astype(jnp.float32), axis=2, keepdims=True)).astype(jnp.bfloat16)
                            return jnp.pad(result[:, :, 64:], ((0, 0), (0, 0), (0, 64)))
                        end = jnp.max(positions) + 1
                        def choose_width(index):
                            if index == len(widths) - 1:
                                return batched_attention(widths[index], jnp.int32(0))
                            return jax.lax.cond(end <= widths[index], lambda: batched_attention(widths[index], jnp.int32(0)), lambda: choose_width(index + 1))
                        result = jax.lax.cond(use_window, lambda: batched_attention(window_width, cache_start), lambda: choose_width(0))
                        attended_groups.append(result)
                    result = jnp.concatenate(attended_groups, axis=0).transpose(1, 0, 2)
                    attended = jnp.tile(result, (1, rows // batch_size, 1)).reshape(nh // nkv * rows, 128)
                elif batch_size:
                    attended_rows = []
                    for row in range(batch_size):
                        row_query = jnp.concatenate([q[row:row + 1] for q in queries[kv_group * (nh // nkv):(kv_group + 1) * (nh // nkv)]], axis=0)
                        attended_rows.append(attend(row_query, kv_cache.at[row], keep[row:row + 1], pos[row:row + 1], meta[row] + 1))
                    attended = jnp.stack([attended_rows[i % batch_size] for i in range(rows)], axis=1).reshape(nh // nkv * rows, 128)
                else:
                    attended = attend(query[kv_group * (nh // nkv) * rows:(kv_group + 1) * (nh // nkv) * rows], kv_cache, keep, pos, meta[0] + rows)
                attended_kv.append(attended)
            attended = jnp.concatenate(attended_kv, axis=0)
            attended = hn(attended)
            parts = []
            for head in range(nh):
                gate = qgkv[:, qwidth + head * 64:qwidth + (head + 1) * 64].astype(jnp.float32)
                parts.append((attended[head * rows:(head + 1) * rows, :64] * jax.nn.sigmoid(gate)).astype(jnp.bfloat16))
            if rows == 1:
                output_copy(li).wait()
                attention_vector = jnp.concatenate(parts, axis=1)
                section = h // output_chunks // 128
                for chunk in range(output_chunks):
                    region = slice(chunk * section, (chunk + 1) * section)
                    pair[lane, region] = dot(attention_vector, obuf[:, chunk * h // output_chunks:(chunk + 1) * h // output_chunks]).reshape(section, 128)
                    for offset in range(7, 0, -1):
                        remote(pair.at[lane, region], pair.at[lane, region], chunk, rank ^ offset).start()
                copy(w["pool"].at[li], pbuf, 1).start()
                for chunk in range(output_chunks):
                    region = slice(chunk * section, (chunk + 1) * section)
                    remote(pair.at[:7, region], pair.at[:7, region], chunk, rank ^ 1).wait()
                    ahost[host, region] = jnp.sum(pair[:, region], axis=0)
                    if tp == 16:
                        remote(ahost.at[host, region], ahost.at[host, region], 8, rank ^ 8).start()
                if tp == 16:
                    remote(ahost.at[host], ahost.at[host], 8, rank ^ 8).wait()
                branch = jnp.sum(ahost[:tp // 8], axis=0).reshape(1, h).astype(jnp.bfloat16)
            else:
                pair[lane] = jnp.concatenate(parts, axis=1)[:active_rows]
                gather(pair, lane, tuple(range(7, 0, -1)), 0)
                joined = jnp.concatenate([pair[i] for i in range(8)], axis=1)
                if o_chunks == 1:
                    output_copy(li).wait()
                    ahost[host] = dot(joined, obuf[...])
                else:
                    for chunk in range(o_chunks):
                        output_copy(li, chunk).wait()
                        ahost[host, :, chunk * oshape[1]:(chunk + 1) * oshape[1]] = dot(joined, obuf[...])
                        if chunk + 1 < o_chunks:
                            output_copy(li, chunk + 1).start()
                copy(w["pool"].at[li], pbuf, 1).start()
                if tp == 16:
                    gather(ahost, host % 2, (8,), 8)
                apaired[lane] = jnp.sum(ahost[:tp // 8], axis=0).astype(jnp.bfloat16)
                gather(apaired, lane, tuple(range(7, 0, -1)), 46)
                branch = jnp.tile(jnp.concatenate([apaired[i] for i in range(8)], axis=1), (rows // active_rows, 1)).astype(jnp.bfloat16)
            stream = sc[3:4] * stream + sc[4:5] * norm(branch, c.post_eps)
            x = norm(stream, c.eps, sc[1:2]).astype(jnp.bfloat16)
            copy(w["router"].at[li], router, 2).wait()
            logits = jnp.concatenate([jnp.sum(x[i:i + 1].astype(jnp.float32) * router[...], axis=1)[None] for i in range(active_rows)], axis=0)
            copy(w["pool"].at[li], pbuf, 1).wait()
            pooled = dot(x[:active_rows], pbuf[...]).astype(jnp.bfloat16)
            comb[lane] = jnp.concatenate((tpu.bitcast(jnp.pad(logits, ((0, 0), (0, 128 - es))), jnp.bfloat16), *[pooled[:, i * 128:(i + 1) * 128] for i in range(pw // 128)]), axis=0)
            copy(w["unpool"].at[li], ubuf, 1).start()
            gather(comb, lane, tuple(range(7, 0, -1)), 9)
            scores = jax.nn.sigmoid(jnp.concatenate([tpu.bitcast(comb[i, :router_rows], jnp.float32)[:, :es] for i in range(8)], axis=1) * c.output_multiplier)
            scores = jnp.tile(scores, (rows // active_rows, 1))
            copy(w["router_bias"].at[li], bias, 5).wait()
            ranked = scores + bias[...][None]
            route_ids, route_weights = [], []
            if rows != 1:
                requested = jnp.zeros((rows, c.experts), jnp.float32)
            for _ in range(c.top_k):
                index = _router_argmax(ranked)
                selected = experts == index
                route_ids.append(index)
                route_weights.append(jnp.sum(jnp.where(router_ids == index, scores, 0), axis=1, keepdims=True))
                if rows != 1:
                    requested += selected.astype(jnp.float32)
                ranked = jnp.where(router_ids == index, -jnp.inf, ranked)
            rw = jnp.concatenate(route_weights, axis=1)
            rw /= jnp.sum(rw, axis=1, keepdims=True) + c.router_eps
            if rows == 1:
                ids_v[:128] = jnp.pad(jnp.concatenate(route_ids, axis=1).reshape(c.top_k), (0, 128 - c.top_k), constant_values=c.experts)
            else:
                first_row = jnp.min(jnp.where(requested > 0, r, rows), axis=0, keepdims=True)
                first = (first_row == r).astype(jnp.bfloat16)
                triangular = (jnp.arange(c.experts)[:, None] <= experts).astype(jnp.bfloat16)
                row_counts = jnp.sum(first.astype(jnp.int32), axis=1)
                prior = jnp.sum(jnp.where(jnp.arange(rows)[None] < r, row_counts[None], 0), axis=1, keepdims=True)
                prefix = dot(first, triangular).astype(jnp.int32) + prior
                active = (first_row < rows).astype(jnp.int32)
                ordinal = jnp.sum(jnp.where(first > 0, prefix - 1, 0), axis=0, keepdims=True)
                ids_v[:128] = jnp.min(jnp.where((active > 0) & (ordinal == jnp.arange(128)[:, None]), experts, c.experts), axis=1)
                positions = jnp.concatenate([jnp.sum(jnp.where(experts == route_ids[k], ordinal, 0), axis=1, keepdims=True) for k in range(c.top_k)], axis=1)
                ids_v[128:128 + rows * 128] = jnp.pad(positions, ((0, 0), (0, 128 - c.top_k))).reshape(-1)
                last_expert = jnp.max(positions, axis=1)
                ids_v[128 + rows * 128:] = jnp.sum(jnp.where(jnp.arange(128)[None] == last_expert[:, None], (1 << jnp.arange(rows))[:, None], 0), axis=0)
            copy(ids_v, ids_s, 20).start()
            copy(w["expert_norms"].at[li], norms, 4).wait()
            pooled = norm(jnp.concatenate([comb[i, router_rows + j * active_rows:router_rows + (j + 1) * active_rows] for i in range(8) for j in range(pw // 128)], axis=1), c.eps, norms[0:1]).astype(jnp.bfloat16)
            pooled = jnp.tile(pooled, (rows // active_rows, 1))
            copy(w["gu_global"].at[pi], gg, 6).wait()
            copy(w["dw_global"].at[pi], dg, 7).wait()

            if tp == 16:
                if batch_size:
                    for row in range(batch_size):
                        start = pl.multiple_of(jnp.clip(meta[row] // 16 * 16, 0, context - 32), 16)
                        buffer_start = pl.multiple_of(start - jnp.where(use_window, cache_start, 0), 16) if windowed else start
                        copy(kvbuf.at[row, pl.ds(buffer_start, 32)], kvc.at[li, row, pl.ds(start, 32)], 21 + row).wait()
                else:
                    copy(kvbuf.at[pl.ds(write_start, 32)], kvc.at[li, 0, pl.ds(write_start, 32)], 10).wait()

            def run_experts(packed):
                prefetch = packed_prefetch if packed else (2 if tp == 8 else 4)
                gu, dw, gb, db, index, width = (w["gu_pack"], w["dw_pack"], gp, dp, pi, iw) if packed else (w["gu_dense"], w["dw_dense"], gd, dd, bi, di)

                def issue(e, slot):
                    if gu_batch == 1:
                        copy(gu.at[index, e], gb.at[slot], 12 + 2 * slot).start()
                    else:
                        channel = 12 + 4 * (slot // 2) if combine_expert_waits else 12 + 2 * slot
                        copy(gu.at[index, e], gb.at[slot // 2, :, pl.ds((slot % 2) * 2 * width, 2 * width)], channel).start()
                    channel = 12 + 4 * (slot // 2) if combine_expert_waits else 13 + 2 * slot
                    copy(dw.at[index, e], db.at[slot], channel).start()

                def dequant(blob, k, n):
                    values = tpu.bitcast(blob[:k // 2], jnp.float4_e2m1fn).astype(jnp.bfloat16)
                    scales = jax.lax.bitcast_convert_type(blob[k // 2:], jnp.float8_e4m3fn).astype(jnp.bfloat16)
                    return (values.reshape(k // 16, 16, n) * scales[:, None]).reshape(k, n)

                count = c.top_k if rows == 1 else jnp.sum(active.astype(jnp.int32))
                copy(ids_v, ids_s, 20).wait()
                for first_id in range(prefetch):
                    @pl.when(count > first_id)
                    def first():
                        issue(ids_s[first_id], jnp.int32(first_id))

                if rows == 1:
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
                        start = pl.multiple_of(i * (mh // 128), 8)
                        estage[0, pl.ds(start, mh // 128)] = value[0].astype(jnp.bfloat16).reshape(mh // 128, 128)
                        @pl.when(i == lane)
                        def local_row():
                            epipe[rank] = estage[0, pl.ds(pl.multiple_of(i * (mh // 128), 8), mh // 128)]
                        for destination_host in range(tp // 8):
                            destination = destination_host * 8 + i
                            @pl.when(destination != rank)
                            def send_row():
                                remote(estage.at[0, pl.ds(pl.multiple_of(i * (mh // 128), 8), mh // 128)], epipe.at[rank], 16, destination).start()
                        @pl.when(i + prefetch < count)
                        def following():
                            issue(ids_s[i + prefetch], slot)
                        return unused

                    jax.lax.fori_loop(0, count, step, None)
                else:
                    def step(group, unused):
                        begin = group * gu_batch
                        if combine_expert_waits:
                            # Two gate/up and two down transfers total three gate/up blobs.
                            aggregate_shape = (3, mh * 9 // 16, 2 * width) if packed else (3, mh, 2 * width)
                            aggregate_dtype = jnp.uint8 if packed else jnp.bfloat16
                            aggregate = pool_alias.view(qa.at[:tiles(aggregate_shape, aggregate_dtype)], aggregate_shape, aggregate_dtype)
                            copy(gu.at[index, :3], aggregate, 12 + 4 * (group % (prefetch // 2))).wait()
                        else:
                            for part in range(gu_batch):
                                slot = (begin + part) % prefetch
                                e = ids_s[jnp.minimum(begin + part, count - 1)]
                                dest = gb.at[slot] if gu_batch == 1 else gb.at[slot // 2, :, pl.ds((slot % 2) * 2 * width, 2 * width)]
                                copy(gu.at[index, e], dest, 12 + 2 * slot).wait()
                                copy(dw.at[index, e], db.at[slot], 13 + 2 * slot).wait()
                        gu_weights = dequant(gb[group % (prefetch // gu_batch)], mh, 2 * gu_batch * width) if packed else gb[group % (prefetch // gu_batch)]
                        projected_pair = dot(pooled, gu_weights)
                        acts, downs = [], []
                        for part in range(gu_batch):
                            i = begin + part
                            slot = i % prefetch
                            e = ids_s[jnp.minimum(i, count - 1)]
                            projected = projected_pair[:, part * 2 * width:(part + 1) * 2 * width]
                            if gu_batch == 2:
                                down_weights = dequant(db[slot], width, mh) if packed else db[slot]
                            if packed:
                                gate_scale = jax.lax.bitcast_convert_type(jnp.full((1, 1), gg[2 * e], jnp.int32), jnp.float32)
                                up_scale = jax.lax.bitcast_convert_type(jnp.full((1, 1), gg[2 * e + 1], jnp.int32), jnp.float32)
                                gate = (projected[:, :width] * gate_scale).astype(jnp.bfloat16).astype(jnp.float32)
                                up = (projected[:, width:] * up_scale).astype(jnp.bfloat16).astype(jnp.float32)
                            else:
                                gate, up = projected[:, :width], projected[:, width:]
                            act = (jax.nn.silu(gate) * up).astype(jnp.bfloat16)
                            acts.append(act)
                            if gu_batch == 2:
                                downs.append(down_weights)
                        if gu_batch == 1:
                            slot = begin % prefetch
                            pair_value = dot(acts[0], dequant(db[slot], width, mh) if packed else db[slot])
                        else:
                            joined_act = jnp.concatenate((jnp.pad(acts[0], ((0, 0), (0, width))), jnp.pad(acts[1], ((0, 0), (width, 0)))), axis=0)
                            pair_value = dot(joined_act, jnp.concatenate(downs, axis=0))
                        for part in range(gu_batch):
                            i = begin + part
                            slot = i % prefetch
                            e = ids_s[jnp.minimum(i, count - 1)]
                            value = pair_value[part * rows:(part + 1) * rows]
                            if packed:
                                value *= jax.lax.bitcast_convert_type(jnp.full((1, 1), dg[e], jnp.int32), jnp.float32)
                            expert_results[i] = value.astype(jnp.bfloat16)
                            @pl.when(i + prefetch < pl.cdiv(count, gu_batch) * gu_batch)
                            def following():
                                issue(ids_s[jnp.minimum(i + prefetch, count - 1)], slot)
                            ready_mask = ids_s[128 + rows * 128 + i]
                            @pl.when(ready_mask != 0)
                            def finish_rows():
                                for row in range(batch_size or rows):
                                    @pl.when((ready_mask & (1 << row)) != 0)
                                    def finish_row():
                                        for k in range(c.top_k):
                                            selected = ids_s[128 + row * 128 + k]
                                            estage[row, pl.ds(k * (mh // 128), mh // 128)] = expert_results[selected, row].reshape(mh // 128, 128)
                                        @pl.when(row == lane)
                                        def local_ready():
                                            epipe[lane] = estage[row, :c.top_k * mh // 128]
                                        @pl.when(row != lane)
                                        def send_ready():
                                            if exchange_chunks == 1:
                                                remote(estage.at[row, :c.top_k * mh // 128], epipe.at[lane], 16 + (lane ^ row) - 1, host * 8 + row).start()
                                            else:
                                                chunk_width = c.top_k * mh // 128 // exchange_chunks
                                                for chunk in range(exchange_chunks):
                                                    region = pl.ds(chunk * chunk_width, chunk_width)
                                                    channel = 16 if chunk == 0 else 53 + (chunk - 1) * 7
                                                    remote(estage.at[row, region], epipe.at[lane, region], channel + (lane ^ row) - 1, host * 8 + row).start()
                        return unused

                    jax.lax.fori_loop(0, pl.cdiv(count, gu_batch), step, None)

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
            if rows == 1:
                remote(epipe.at[:tp - 1], epipe.at[:tp - 1], 16, rank ^ 1).wait()
                full = jnp.sum(jnp.sum(epipe[...].astype(jnp.float32).reshape(tp // 8, 8, mh // 128, 128), axis=1), axis=0).reshape(1, mh)
                mixed_rows[lane] = norm(full * norms[1:2], c.post_eps).reshape(mh // 128, 128)
                gather(mixed_rows, lane, tuple(range(7, 0, -1)), 36)
                mixed = jnp.zeros((1, mh), jnp.float32)
                for k in range(c.top_k):
                    mixed += mixed_rows[k].reshape(1, mh) * rw[:, k:k + 1]
            elif batch_size in (2, 4):
                for row in range(batch_size):
                    @pl.when(row != lane)
                    def sent():
                        remote(estage.at[row, :c.top_k * mh // 128], epipe.at[lane], 16 + (lane ^ row) - 1, host * 8 + row).wait_send()
                @pl.when(lane < batch_size)
                def combine():
                    for peer in range(8):
                        @pl.when(peer != lane)
                        def received():
                            remote(estage.at[0, :c.top_k * mh // 128], epipe.at[peer], 16 + (lane ^ peer) - 1, host * 8 + peer).wait_recv()
                    rowhost[host] = jnp.sum(epipe[...].astype(jnp.float32), axis=0).reshape(c.top_k, mh)
                    if tp == 16:
                        gather(rowhost, host, (8,), 44)
                    full = jnp.sum(rowhost[:tp // 8], axis=0)
                    mixed = jnp.zeros((1, mh), jnp.float32)
                    for k in range(c.top_k):
                        weight = jnp.sum(jnp.where(r == owner, rw[:, k:k + 1], 0), axis=0, keepdims=True)
                        mixed += norm(full[k:k + 1] * norms[1:2], c.post_eps) * weight
                    mixed_slice = pl.ds(pl.multiple_of(owner * (mh // 128), 8), mh // 128)
                    mixed_rows[mixed_slice] = mixed.astype(jnp.bfloat16).reshape(mh // 128, 128)
                    transfers = [remote(mixed_rows.at[mixed_slice], mixed_rows.at[mixed_slice], 36 + offset - 1, rank ^ offset) for offset in range(7, 0, -1)]
                    for transfer in transfers:
                        transfer.start()
                    for transfer in transfers:
                        transfer.wait_send()
                for source in range(batch_size):
                    @pl.when(source != lane)
                    def received_mixed():
                        source_slice = pl.ds(source * (mh // 128), mh // 128)
                        remote(mixed_rows.at[source_slice], mixed_rows.at[source_slice], 36 + (lane ^ source) - 1, host * 8 + source).wait_recv()
                mixed_rows[...] = jnp.tile(mixed_rows[:batch_size * mh // 128], (rows // batch_size, 1))
            else:
                if exchange_chunks == 1:
                    transfers = [remote(estage.at[lane ^ offset, :c.top_k * mh // 128], epipe.at[lane], 16 + offset - 1, rank ^ offset) for offset in range(7, 0, -1)]
                    for transfer in transfers:
                        transfer.wait()
                    rowhost[host] = jnp.sum(epipe[...].astype(jnp.float32), axis=0).reshape(c.top_k, mh)
                    if tp == 16:
                        gather(rowhost, host, (8,), 44)
                    full = jnp.sum(rowhost[:tp // 8], axis=0)
                    mixed = jnp.zeros((1, mh), jnp.float32)
                    for k in range(c.top_k):
                        weight = jnp.sum(jnp.where(r == owner, rw[:, k:k + 1], 0), axis=0, keepdims=True)
                        mixed += norm(full[k:k + 1] * norms[1:2], c.post_eps) * weight
                else:
                    chunk_experts = c.top_k // exchange_chunks
                    chunk_width = chunk_experts * mh // 128
                    for chunk in range(exchange_chunks):
                        region = pl.ds(chunk * chunk_width, chunk_width)
                        channel = 16 if chunk == 0 else 53 + (chunk - 1) * 7
                        transfers = [remote(estage.at[lane ^ offset, region], epipe.at[lane, region], channel + offset - 1, rank ^ offset) for offset in range(7, 0, -1)]
                        for transfer in transfers:
                            transfer.wait()
                        expert_region = pl.ds(chunk * chunk_experts, chunk_experts)
                        rowhost[host, expert_region] = jnp.sum(epipe[:, region].astype(jnp.float32), axis=0).reshape(chunk_experts, mh)
                        inter_channel = 44 if chunk == 0 else 53 + (exchange_chunks - 1) * 7 + chunk - 1
                        if tp == 16:
                            remote(rowhost.at[host, expert_region], rowhost.at[host, expert_region], inter_channel, rank ^ 8).start()
                    mixed = jnp.zeros((1, mh), jnp.float32)
                    for chunk in range(exchange_chunks):
                        expert_region = pl.ds(chunk * chunk_experts, chunk_experts)
                        inter_channel = 44 if chunk == 0 else 53 + (exchange_chunks - 1) * 7 + chunk - 1
                        if tp == 16:
                            remote(rowhost.at[host, expert_region], rowhost.at[host, expert_region], inter_channel, rank ^ 8).wait()
                        full = jnp.sum(rowhost[:tp // 8, expert_region], axis=0)
                        for k in range(chunk_experts):
                            weight = jnp.sum(jnp.where(r == owner, rw[:, chunk * chunk_experts + k:chunk * chunk_experts + k + 1], 0), axis=0, keepdims=True)
                            mixed += norm(full[k:k + 1] * norms[1:2], c.post_eps) * weight
                mixed_slice = pl.ds(pl.multiple_of(owner * (mh // 128), 8), mh // 128)
                mixed_rows[mixed_slice] = mixed.astype(jnp.bfloat16).reshape(mh // 128, 128)
                transfers = [remote(mixed_rows.at[mixed_slice], mixed_rows.at[mixed_slice], 36 + offset - 1, rank ^ offset) for offset in range(rows - 1, 0, -1)]
                for transfer in transfers:
                    transfer.start()
                for transfer in transfers:
                    transfer.wait()
            copy(w["unpool"].at[li], ubuf, 1).wait()
            projected = dot(mixed.astype(jnp.bfloat16) if rows == 1 else mixed_rows[...].reshape(rows, mh)[:active_rows], ubuf[...]).astype(jnp.bfloat16)
            @pl.when(li + 1 < c.layers)
            def prefetch_o():
                if not output_in_q:
                    output_copy(li + 1).start()
            if rows == 1:
                ungather[lane] = projected.astype(jnp.float32).reshape(h // 8 // 128, 128)
                gather(ungather, lane, tuple(range(7, 0, -1)), 29)
                branch = ungather[...].reshape(1, h)
            else:
                unpool_rows = active_rows * h // 8 // 128
                ungather[lane] = jnp.pad(projected.reshape(unpool_rows, 128), ((0, max(16, unpool_rows) - unpool_rows), (0, 0)))
                gather(ungather, lane, tuple(range(7, 0, -1)), 29)
                branch = jnp.tile(jnp.concatenate([ungather[i, :unpool_rows].reshape(active_rows, h // 8).astype(jnp.float32) for i in range(8)], axis=1), (rows // active_rows, 1))
            stream = sc[5:6] * stream + sc[6:7] * norm(branch * sc[2:3], c.post_eps)
            for i, layer_id in enumerate(aux_layers):
                @pl.when(li == layer_id)
                def capture():
                    aux[i] = stream.astype(jnp.bfloat16)
            return stream

        stream = jax.lax.fori_loop(0, c.layers, layer, initial_stream)
        copy(w["fnorm"].at[0], scalars.at[7], 3).wait()
        hidden = (norm(stream, c.eps) * scalars[7:8]).astype(jnp.bfloat16)

        def head_chunk(index, carry):
            best, best_index = carry
            slot = index % 2
            offset = index * lm_chunk
            copy(w["lmw"].at[:, pl.ds(offset, lm_chunk)], lm.at[slot], slot * 16).wait()
            logits = dot(hidden, lm[slot])
            columns = offset + jnp.arange(lm_chunk)[None]
            logits = jnp.where(columns < c.vocab // tp, logits, -jnp.inf)
            local_best = jnp.max(logits, axis=1, keepdims=True)
            local_index = jnp.min(jnp.where(logits == local_best, columns + rank * (c.vocab // tp), c.vocab), axis=1, keepdims=True)
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
        if tp == 16:
            candidates[8 + host] = pack(best, best_index)
            gather(candidates, 8 + host, (8,), 8)
            best, best_index = winner(8, 2)
        final[...] = jnp.broadcast_to(best_index.astype(jnp.int32), (rows, 128))

    def cache_write(control, previous, delta, updated, block, semaphore):
        start = pl.multiple_of(jnp.minimum(control[0] // 16 * 16, context - 32), 16)
        source = previous.at[:, 0, pl.ds(start, 32)]
        target = updated.at[:, 0, pl.ds(start, 32)]
        transfer = tpu.make_async_copy(source, block, semaphore)
        transfer.start()
        transfer.wait()
        values = block[...]
        positions = start + jnp.arange(32, dtype=jnp.int32)[None, :, None]
        for row in range(rows):
            values = jnp.where((positions == control[0] + row) & (row < control[1]), delta[:, row:row + 1], values)
        block[...] = values
        transfer = tpu.make_async_copy(block, target, semaphore)
        transfer.start()
        transfer.wait()

    def write_delta(previous, delta, control):
        return pl.pallas_call(
            cache_write, out_shape=jax.ShapeDtypeStruct(previous.shape, previous.dtype),
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[pl.BlockSpec(memory_space=tpu.HBM), pl.BlockSpec()],
                out_specs=pl.BlockSpec(memory_space=tpu.HBM),
                scratch_shapes=(tpu.VMEM((c.layers, 32, kvwidth), jnp.bfloat16), tpu.SemaphoreType.DMA)),
            input_output_aliases={1: 0}, compiler_params=tpu.CompilerParams(shape_invariant_numerics=True),
            name="mpk_tp8_verifier_cache_update")(control, previous, delta)

    def local(global_weights, states, tokens, control):
        w = jax.tree.map(lambda a: a[0], global_weights)
        if rows != 1:
            w["ow"] = w["ow_group8"]
        w["pool"] = w["pool_local"]
        w["unpool"] = w["unpool_group8"]
        w["router"] = w["router_ordered"]
        w["router_bias"] = w["router_bias_ordered"]
        w["gu_global"] = jax.lax.bitcast_convert_type(w["gu_global"], jnp.int32).reshape(-1, 2 * c.experts)
        w["dw_global"] = jax.lax.bitcast_convert_type(w["dw_global"], jnp.int32)
        kvc = states["kvcache"][0]
        offsets = tokens - jax.lax.axis_index("tp") * (c.vocab // tp)
        embeddings = w["emb"][jnp.clip(offsets, 0, c.vocab // tp - 1)]
        selected = jnp.where(((offsets >= 0) & (offsets < c.vocab // tp))[:, None], embeddings, 0)
        stream = (jnp.concatenate([rms_plain_f32(selected[i:i + 1], c.eps) for i in range(rows)], axis=0)
                  if batch_size else rms_plain_f32(selected, c.eps))
        v = lambda shape, dtype=jnp.bfloat16: tpu.VMEM(shape, dtype)
        shapes = (jax.ShapeDtypeStruct((rows, 128), jnp.int32), jax.ShapeDtypeStruct((len(aux_layers), rows, h), jnp.bfloat16), jax.ShapeDtypeStruct((c.layers, rows, kvwidth) if tp == 8 else kvc.shape, kvc.dtype))
        n = len(_BLOCK_WEIGHTS)
        stream, aux, kvc = pl.pallas_call(
            kernel, out_shape=shapes,
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[pl.BlockSpec(memory_space=tpu.HBM)] * n + [pl.BlockSpec(), pl.BlockSpec(memory_space=tpu.HBM)],
                out_specs=[pl.BlockSpec(), pl.BlockSpec(), pl.BlockSpec() if tp == 8 else pl.BlockSpec(memory_space=tpu.HBM)],
                scratch_shapes=(v((qt, 32, 128), jnp.uint8), v((pt, 32, 128), jnp.uint8),
                    v((es, h), jnp.float32), v((8, h), jnp.float32), v((2, mh), jnp.float32), v((c.experts,), jnp.float32),
                    tpu.SMEM((2 * c.experts,), jnp.int32), tpu.SMEM((c.experts,), jnp.int32),
                    v((8, h // 128, 128), jnp.float32) if rows == 1 else v((8, active_rows, qwidth)),
                    v((2, h // 128, 128) if rows == 1 else (2, active_rows, h // 8), jnp.float32),
                    v((1, 128)) if rows == 1 else v((8, active_rows, h // 8)), v((8, router_rows + pool_rows, 128)),
                    v((rows, (c.top_k + 1) * mh // 128, 128)), v((tp if rows == 1 else 8, (1 if rows == 1 else c.top_k) * mh // 128, 128)),
                    v((2, mh // 128, 128) if rows == 1 else (2, c.top_k, mh), jnp.float32),
                    v((8, mh // 128, 128), jnp.float32) if rows == 1 else v((rows * mh // 128, 128)),
                    v((8, h // 8 // 128, 128), jnp.float32) if rows == 1 else v((8, active_rows * h // 8 // 128, 128)),
                    v((id_count,), jnp.int32), tpu.SMEM((id_count,), jnp.int32),
                    v((10, rows, 128), jnp.float32), v((1, 128)) if rows == 1 else v((rows * c.top_k, rows, mh)), tpu.SemaphoreType.DMA((21 + batch_size,)), tpu.SemaphoreType.DMA((remote_channels,)), tpu.SemaphoreType.DMA((remote_channels,))),
            ),
            input_output_aliases={} if tp == 8 else {n + 2: 2},
            compiler_params=tpu.CompilerParams(collective_id=51, vmem_limit_bytes=64 * 1024**2, shape_invariant_numerics=True, disable_bounds_checks=combine_expert_waits or rows == 1),
            name="mpk_tp16_batch" if batch_size else "mpk_tp16_single" if rows == 1 else "mpk_tp16_persistent_verify",
        )(control, *[w[key] for key in _BLOCK_WEIGHTS], stream, kvc)
        if tp == 8:
            previous = states["kvcache"][0]
            layers = jnp.arange(c.layers, dtype=jnp.int32)[:, None]
            if batch_size:
                positions = jnp.where(control >= 0, control, context)
                kvc = previous.at[layers, jnp.arange(batch_size)[None], positions[None], :].set(kvc[:, :batch_size], mode="drop")
            elif tp8_verify and context >= 32 and context % 16 == 0:
                kvc = write_delta(previous, kvc, control)
            else:
                positions = jnp.where(jnp.arange(rows) < control[1], control[0] + jnp.arange(rows), context)
                kvc = previous.at[layers, 0, positions[None], :].set(kvc, mode="drop")
        targets = stream[:, 0]
        return targets, aux.transpose(1, 0, 2), {"kvcache": kvc[None]}

    return jax.jit(jax.shard_map(local, mesh=mesh, in_specs=(P("tp"), P("tp"), P(), P()), out_specs=(P(), P(), P("tp")), check_vma=False), donate_argnums=(1,))

def logical_mesh(tp):
    if tp not in (8, 16):
        raise ValueError("this kernel supports TP8 or TP16")
    devices = sorted(jax.devices(), key=lambda device: (device.process_index, device.id))
    if len(devices) != tp:
        raise ValueError(f"TP{tp} requires {tp} devices, found {len(devices)}")
    return jax.sharding.Mesh(np.asarray(devices), ("tp",))


def prepare_decode_weights(mesh, weights):
    from .load import prepare_tp8_weights, prepare_tp16_weights
    if mesh.size not in (8, 16):
        raise ValueError("this kernel supports TP8 or TP16")
    return (prepare_tp8_weights if mesh.size == 8 else prepare_tp16_weights)(mesh, weights)


def prepare_verify_weights(mesh, weights):
    return prepare_decode_weights(mesh, weights)


@jax.jit
def prepare_verify_cache(states):
    if "kvcache" in states:
        return states
    packed = jnp.concatenate((states["kcache"][..., :64], states["vcache"][..., :64]), axis=-1)
    tp, layers, heads, context, _ = packed.shape
    return {"kvcache": packed.transpose(0, 1, 3, 2, 4).reshape(tp, layers, 1, context, heads * 128)}


def make_verify_block(mesh, c, context, rows, aux_layers):
    block = _make_decode_block(mesh, c, context, rows, aux_layers)

    def run(weights, states, tokens, position):
        return block(weights, states, tokens, jnp.stack((position, jnp.int32(rows))))

    return jax.jit(run, donate_argnums=(1,))


def make_verify(mesh, c, context, rows, aux_layers):
    block = _make_decode_block(mesh, c, context, 8, aux_layers)

    def run(weights, states, tokens, control):
        initial = (states, jnp.zeros((rows,), jnp.int32), jnp.zeros((rows, len(aux_layers), c.dim), jnp.bfloat16))

        def step(i, carry):
            states, targets, aux = carry
            predicted, captured, states = block(weights, states, jnp.full((8,), tokens[i], jnp.int32), jnp.stack((control[0] + i, jnp.int32(1))))
            return states, targets.at[i].set(predicted[0]), aux.at[i].set(captured[0])

        states, targets, aux = jax.lax.fori_loop(0, control[1], step, initial)
        return targets, aux, states

    return jax.jit(run, donate_argnums=(1,))


def _make_decode_persistent(mesh, c, context, ablate=()):
    if ablate:
        raise ValueError("the persistent TP8/TP16 decoder does not support ablations")
    block = _make_decode_block(mesh, c, context, 1, (), windowed=True)

    def run(weights, states, token, position):
        predicted, _, states = block(weights, states, token[:1], jnp.stack((position[0], jnp.int32(1))))
        return predicted[:1], jnp.zeros((mesh.size, 1), jnp.float32), states

    return jax.jit(run, donate_argnums=(1,))


def make_decode_batch(mesh, c, context, batch_size):
    """TP8/TP16 greedy decode with one combined KV cache per sequence.

    Tokens and positions are [batch]. A negative position leaves that
    sequence's KV and input token unchanged. BS2/4 pad the compute rows by
    repeating sequences; each real sequence has exactly one KV cache.
    """
    if batch_size not in (2, 4, 8):
        raise ValueError("TP8/TP16 batch size must be two, four, or eight")
    block = _make_decode_block(mesh, c, context, 8, (), batch_size=batch_size)
    window_block = (_make_decode_block(mesh, c, context, 8, (), batch_size=batch_size, windowed=True)
                    if (context >= 128 if batch_size == 8 else context > 2 * c.sliding_window) else None)

    def run(weights, states, tokens, positions):
        if tokens.shape != (batch_size,) or positions.shape != (batch_size,):
            raise ValueError("tokens and positions must have shape [batch_size]")
        if states["kvcache"].shape != (mesh.size, c.layers, batch_size, context, c.kv_heads // mesh.size * 128):
            raise ValueError("batched KV must have shape [tp, layers, batch_size, context, kv_heads_per_rank * 128]")
        rows = jnp.tile(tokens, 8 // batch_size)
        if window_block is None:
            predicted, _, states = block(weights, states, rows, positions)
        else:
            first, last = jnp.min(positions), jnp.max(positions)
            use_window = (first >= (0 if batch_size == 8 else 2 * c.sliding_window)) & (last - first <= 256)
            predicted, _, states = jax.lax.cond(
                use_window, lambda: window_block(weights, states, rows, positions),
                lambda: block(weights, states, rows, positions))
        return jnp.where(positions >= 0, predicted[:batch_size], tokens), states

    return jax.jit(run, donate_argnums=(1,))


def make_decode_batch_chunk(mesh, c, context, batch_size, chunk_size=4):
    """Four-step batched decode; control holds [positions, step counts]."""
    if chunk_size != 4:
        raise ValueError("steps_per_call must remain four")
    decode = make_decode_batch(mesh, c, context, batch_size)

    def run(weights, states, tokens, control):
        if control.shape != (2, batch_size):
            raise ValueError("control must have shape [2, batch_size]")
        positions, counts = control[0], jnp.clip(control[1], 0, chunk_size)
        counts = jnp.minimum(counts, jnp.maximum(context - positions, 0))

        def step(i, carry):
            tokens, states, emitted = carry
            current = jnp.where((i < counts) & (positions >= 0), positions + i, -1)
            tokens, states = decode(weights, states, tokens, current)
            return tokens, states, emitted.at[i].set(tokens)

        return jax.lax.fori_loop(0, jnp.max(counts), step,
            (tokens, states, jnp.zeros((chunk_size, batch_size), jnp.int32)))

    return jax.jit(run, donate_argnums=(1,))


def make_decode_chunk(mesh, c, context, chunk_size, tp=16, ablate=()):
    decode = make_decode(mesh, c, context, tp, ablate)

    def run(weights, states, token, control):
        def step(i, carry):
            token, states, emitted = carry
            token, _, states = decode(weights, states, token, control[:1] + i)
            return token, states, emitted.at[i].set(token[0])

        return jax.lax.fori_loop(0, control[1], step, (token, states, jnp.zeros((chunk_size,), jnp.int32)))

    return jax.jit(run, donate_argnums=(1,))
