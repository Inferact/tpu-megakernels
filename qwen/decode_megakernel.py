"""Qwen3.8 TP-sharded prefill, fused decode, and speculative verification."""

import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import PartitionSpec as P

from . import (
    F32,
    apply_rope_batched,
    causal_conv,
    chunk_gated_delta_rule,
    delta_rule_block,
    linear_seq,
    rms,
    rms_gated,
    rope_tables,
    sigmoid,
)
from .load import dw_k, gu_width, lm_width, schedule, zba_width


_lm_width_fn = lm_width
BANK_COUNT = int(os.environ.get("QWEN_NB", "12"))
MAX_BANK_K = int(os.environ.get("QWEN_MAXK", "1024"))
MAX_BANK_N = int(os.environ.get("QWEN_MAXN", "1280"))


def make_prefill(mesh, c, context, tp, tap_layers=None):
    """TP-sharded prefill over packed (untiled) weights.

    Each rank computes its head/channel shard; row-parallel projections reduce
    with lax.psum, mirroring the decode kernel's TP split. Returns states in the
    kernel's packed per-rank layout (no host round-trip) plus per-rank
    last-position logits shards.
    """
    kh_r = c.k_heads // tp
    vh_r = c.v_heads // tp
    kl_r = kh_r * c.k_dim
    vl_r = vh_r * c.v_dim
    qkvw_r = 2 * kl_r + vl_r
    vshard = c.vocab // tp
    nh_r = c.heads // tp
    nkv_r = max(1, c.kv_heads // tp)
    inter_r = c.intermediate // tp
    tap_set = frozenset(tap_layers or ())

    def local(w, states, tokens, npad):
        w = jax.tree.map(lambda a: a[0], w)
        states = jax.tree.map(lambda a: a[0], states)
        rank = jax.lax.axis_index("tp")
        lo = rank * vshard
        in_range = (tokens >= lo) & (tokens < lo + vshard)
        rows = w["emb"][jnp.clip(tokens - lo, 0, vshard - 1)]
        hidden = jax.lax.psum(jnp.where(in_range[:, None], rows.astype(F32), 0.0), "tp").astype(
            jnp.bfloat16
        )
        L = tokens.shape[0]
        real = jnp.arange(L) >= npad
        hidden = jnp.where(real[:, None], hidden, jnp.zeros((), hidden.dtype))

        new_conv, new_rec, new_k, new_v = [], [], [], []
        taps = []
        for li in range(c.layers):
            hidden = jnp.where(real[:, None], hidden, jnp.zeros((), hidden.dtype))
            mixed = rms(hidden, w["norm_in"][li], c.eps)
            if li in c.full_attention:
                ord2 = (li + 1) // 4 - 1
                qwkv = linear_seq(mixed, w["qwkv"][ord2])
                packed_qg = qwkv[:, : nh_r * 2 * c.head_dim].reshape(L, nh_r, 2 * c.head_dim)
                query, gate = packed_qg[..., : c.head_dim], packed_qg[..., c.head_dim :]
                query = rms(query, w["qn"][ord2], c.eps)
                kv = qwkv[:, nh_r * 2 * c.head_dim : nh_r * 2 * c.head_dim + 2 * nkv_r * c.head_dim]
                key = rms(
                    kv[:, : nkv_r * c.head_dim].reshape(L, nkv_r, c.head_dim),
                    w["kn"][ord2],
                    c.eps,
                )
                value = kv[:, nkv_r * c.head_dim :].reshape(L, nkv_r, c.head_dim)
                cos, sin = rope_tables(c, jnp.arange(L, dtype=jnp.int32))
                query = apply_rope_batched(query, cos, sin)
                key = apply_rope_batched(key, cos, sin)
                grouped = query.transpose(1, 0, 2).reshape(nkv_r, nh_r // nkv_r, L, c.head_dim)
                keys_t = key.transpose(1, 0, 2)
                scores = jnp.einsum("gjld,gtd->gjlt", grouped.astype(F32), keys_t.astype(F32))
                scores *= c.head_dim**-0.5
                causal = jnp.arange(L)[None, :] <= jnp.arange(L)[:, None]
                keep = causal & real[None, :]
                scores = jnp.where(keep[None, None], scores, -jnp.inf)
                scores = jnp.where(real[None, None, :, None], scores, 0.0)
                probs = jax.nn.softmax(scores, axis=-1).astype(jnp.bfloat16)
                out = jnp.einsum(
                    "gjlt,gtd->gjld", probs.astype(F32), value.transpose(1, 0, 2).astype(F32)
                )
                out = out.reshape(nh_r, L, c.head_dim).transpose(1, 0, 2)
                out = out * jax.nn.sigmoid(gate.astype(F32)).astype(jnp.bfloat16)
                partial = jnp.dot(out.reshape(L, -1), w["ow"][ord2], preferred_element_type=F32)
                out = jax.lax.psum(partial, "tp").astype(jnp.bfloat16)
                kc, vc = states["kcache"][ord2], states["vcache"][ord2]
                kc = kc.at[:, :L].set(key.transpose(1, 0, 2).astype(kc.dtype))
                vc = vc.at[:, :L].set(value.transpose(1, 0, 2).astype(vc.dtype))
                new_k.append(kc)
                new_v.append(vc)
            else:
                ord_ = li - (li + 1) // 4
                qkvz = linear_seq(mixed, w["qkvz"][ord_])
                qkv = qkvz[:, :qkvw_r]
                conv = states["conv"][ord_]
                qkv_f = jax.nn.silu(causal_conv(qkv, w["convw"][ord_], conv).astype(jnp.bfloat16))
                q, k, v = jnp.split(qkv_f, (kl_r, 2 * kl_r), axis=-1)
                q = q.reshape(L, kh_r, c.k_dim)
                k = k.reshape(L, kh_r, c.k_dim)
                v = v.reshape(L, vh_r, c.v_dim)
                zb = qkvz[:, qkvw_r:]
                # Packed zba tail is tile-padded; slice a explicitly so the pad
                # columns don't leak into it.
                z = zb[:, :vl_r]
                b = zb[:, vl_r : vl_r + vh_r]
                a = zb[:, vl_r + vh_r : vl_r + 2 * vh_r]
                real_ = real[:, None]
                beta = sigmoid(jnp.where(real_, b, -1e30)).astype(F32)
                a = jnp.where(real_, a, -1e30)
                gate = -jnp.exp(w["alog"][ord_].astype(F32)) * jax.nn.softplus(
                    a.astype(F32) + w["dtb"][ord_].astype(F32)
                )
                expand = vh_r // kh_r
                q = jnp.repeat(q, expand, axis=1)
                k = jnp.repeat(k, expand, axis=1)
                out, new_rec_l = chunk_gated_delta_rule(q, k, v, gate, beta, states["rec"][ord_])
                out = rms_gated(out, z.reshape(L, vh_r, c.v_dim), w["onorm"][ord_], c.eps)
                partial = jnp.dot(out.reshape(L, -1), w["outw"][ord_], preferred_element_type=F32)
                out = jax.lax.psum(partial, "tp").astype(jnp.bfloat16)
                combined = jnp.concatenate((conv.T, qkv), axis=0)
                new_conv.append(combined[-c.conv_size :].T)
                new_rec.append(new_rec_l)
            hidden = hidden + out
            mixed = rms(hidden, w["norm_post"][li], c.eps)
            gu = linear_seq(mixed, w["gu"][li])
            gate = jax.nn.silu(gu[:, :inter_r].astype(F32)).astype(jnp.bfloat16)
            up = gu[:, inter_r : 2 * inter_r]
            hh = (gate * up).astype(jnp.bfloat16)
            dwr = w["dw"][li]
            if hh.shape[-1] != dwr.shape[0]:
                # Packed dw rows are zero-padded to the tile width; pad the
                # activation tail with zeros to match (numerically exact).
                hh = jnp.pad(hh, ((0, 0), (0, dwr.shape[0] - hh.shape[-1])))
            partial = jnp.dot(hh, dwr, preferred_element_type=F32)
            hidden = hidden + jax.lax.psum(partial, "tp").astype(jnp.bfloat16)
            if li in tap_set:
                taps.append(hidden)

        hidden = rms(hidden, w["fnorm"], c.eps)
        logits_local = jnp.dot(hidden[-1], w["lmw"], preferred_element_type=F32)[None]
        new_states = {
            "conv": jnp.stack(new_conv),
            "rec": jnp.stack(new_rec),
            "kcache": jnp.stack(new_k),
            "vcache": jnp.stack(new_v),
        }
        out = [jax.tree.map(lambda a: a[None], new_states), logits_local]
        if tap_set:
            out.append(jnp.concatenate(taps, axis=-1))  # [L, n_taps*dim], replicated
        return tuple(out)

    def prefill(w, states, tokens, npad):
        ws = jax.tree.map(lambda _: jax.sharding.PartitionSpec("tp"), w)
        ss = jax.tree.map(lambda _: jax.sharding.PartitionSpec("tp"), states)
        out_specs = (ss, jax.sharding.PartitionSpec("tp")) + (
            (jax.sharding.PartitionSpec(),) if tap_set else ()
        )
        return jax.shard_map(
            local, mesh=mesh, in_specs=(ws, ss, PNone(), PNone()), out_specs=out_specs,
            check_vma=False,
        )(w, states, tokens, npad)

    return prefill


def PNone():
    return jax.sharding.PartitionSpec()


def make_verify(mesh, c, context, tp, tap_layers=(5, 19, 33, 47, 61), block=8):
    """TP-sharded batched verify for speculative decoding.

    Processes `block` draft-block tokens (anchor + drafts) at cache slots
    [pos0, pos0+block) against the committed states. Returns per-position
    logits, hidden taps (for the drafter), and per-position recurrent-state
    snapshots so the driver can roll back to the accepted prefix. KV cache
    slots beyond the accepted prefix are simply overwritten next round.
    """
    kh_r = c.k_heads // tp
    vh_r = c.v_heads // tp
    kl_r = kh_r * c.k_dim
    vl_r = vh_r * c.v_dim
    qkvw_r = 2 * kl_r + vl_r
    vshard = c.vocab // tp
    nh_r = c.heads // tp
    nkv_r = max(1, c.kv_heads // tp)
    grp = nh_r // nkv_r
    inter_r = c.intermediate // tp
    tap_set = frozenset(tap_layers)
    B = block

    def local(w, states, tokens, pos0, key_lo):
        w = jax.tree.map(lambda a: a[0], w)
        states = jax.tree.map(lambda a: a[0], states)
        rank = jax.lax.axis_index("tp")
        lo = rank * vshard
        in_range = (tokens >= lo) & (tokens < lo + vshard)
        rows = w["emb"][jnp.clip(tokens - lo, 0, vshard - 1)]
        hidden = jax.lax.psum(jnp.where(in_range[:, None], rows.astype(F32), 0.0), "tp").astype(
            jnp.bfloat16
        )
        positions = pos0 + jnp.arange(B, dtype=jnp.int32)
        slots = jnp.arange(context, dtype=jnp.int32)

        new_conv, new_rec, new_k, new_v = [], [], [], []
        conv_snaps, rec_snaps, taps = [], [], []
        for li in range(c.layers):
            mixed = rms(hidden, w["norm_in"][li], c.eps)
            if li in c.full_attention:
                ord2 = (li + 1) // 4 - 1
                qwkv = linear_seq(mixed, w["qwkv"][ord2])
                packed_qg = qwkv[:, : nh_r * 2 * c.head_dim].reshape(B, nh_r, 2 * c.head_dim)
                query, gate = packed_qg[..., : c.head_dim], packed_qg[..., c.head_dim :]
                query = rms(query, w["qn"][ord2], c.eps)
                kv = qwkv[:, nh_r * 2 * c.head_dim : nh_r * 2 * c.head_dim + 2 * nkv_r * c.head_dim]
                key = rms(
                    kv[:, : nkv_r * c.head_dim].reshape(B, nkv_r, c.head_dim),
                    w["kn"][ord2],
                    c.eps,
                )
                value = kv[:, nkv_r * c.head_dim :].reshape(B, nkv_r, c.head_dim)
                cos, sin = rope_tables(c, positions)
                query = apply_rope_batched(query, cos, sin)
                key = apply_rope_batched(key, cos, sin)
                kc, vc = states["kcache"][ord2], states["vcache"][ord2]
                kc = jax.lax.dynamic_update_slice(
                    kc, key.transpose(1, 0, 2).astype(kc.dtype), (0, pos0, 0)
                )
                vc = jax.lax.dynamic_update_slice(
                    vc, value.transpose(1, 0, 2).astype(vc.dtype), (0, pos0, 0)
                )
                new_k.append(kc)
                new_v.append(vc)
                grouped = query.transpose(1, 0, 2).reshape(nkv_r, grp, B, c.head_dim)
                scores = jnp.einsum("gqbd,gtd->gqbt", grouped.astype(F32), kc.astype(F32))
                scores *= c.head_dim**-0.5
                keep = (slots[None, :] <= positions[:, None]) & (slots[None, :] >= key_lo)
                scores = jnp.where(keep[None, None], scores, -jnp.inf)
                probs = jax.nn.softmax(scores, axis=-1).astype(jnp.bfloat16)
                out = jnp.einsum("gqbt,gtd->gqbd", probs.astype(F32), vc.astype(F32))
                out = out.transpose(2, 0, 1, 3).reshape(B, nh_r, c.head_dim).astype(jnp.bfloat16)
                out = (out * sigmoid(gate)).reshape(B, nh_r * c.head_dim)
                partial = jnp.dot(out, w["ow"][ord2], preferred_element_type=F32)
                hidden = hidden + jax.lax.psum(partial, "tp").astype(jnp.bfloat16)
            else:
                ord_ = li - (li + 1) // 4
                qkvz = linear_seq(mixed, w["qkvz"][ord_])
                qkv = qkvz[:, :qkvw_r]
                conv = states["conv"][ord_]
                qkv_f = jax.nn.silu(causal_conv(qkv, w["convw"][ord_], conv).astype(jnp.bfloat16))
                q, k, v = jnp.split(qkv_f, (kl_r, 2 * kl_r), axis=-1)
                q = q.reshape(B, kh_r, c.k_dim)
                k = k.reshape(B, kh_r, c.k_dim)
                v = v.reshape(B, vh_r, c.v_dim)
                zb = qkvz[:, qkvw_r:]
                z = zb[:, :vl_r]
                b = zb[:, vl_r : vl_r + vh_r]
                a = zb[:, vl_r + vh_r : vl_r + 2 * vh_r]
                beta = sigmoid(b).astype(F32)
                gate = -jnp.exp(w["alog"][ord_].astype(F32)) * jax.nn.softplus(
                    a.astype(F32) + w["dtb"][ord_].astype(F32)
                )
                expand = vh_r // kh_r
                q = jnp.repeat(q, expand, axis=1)
                k = jnp.repeat(k, expand, axis=1)
                out, recs = delta_rule_block(q, k, v, gate, beta, states["rec"][ord_])
                out = rms_gated(out, z.reshape(B, vh_r, c.v_dim), w["onorm"][ord_], c.eps)
                partial = jnp.dot(out.reshape(B, -1), w["outw"][ord_], preferred_element_type=F32)
                hidden = hidden + jax.lax.psum(partial, "tp").astype(jnp.bfloat16)
                combined = jnp.concatenate((conv.T, qkv), axis=0)
                new_conv.append(combined[-c.conv_size :].T)
                new_rec.append(recs[-1])
                conv_snaps.append(
                    jnp.stack([combined[t + 1 : t + 1 + c.conv_size].T for t in range(B)])
                )
                rec_snaps.append(recs)
            mixed = rms(hidden, w["norm_post"][li], c.eps)
            gu = linear_seq(mixed, w["gu"][li])
            gate_u = jax.nn.silu(gu[:, :inter_r].astype(F32)).astype(jnp.bfloat16)
            up = gu[:, inter_r : 2 * inter_r]
            hh = (gate_u * up).astype(jnp.bfloat16)
            dwr = w["dw"][li]
            if hh.shape[-1] != dwr.shape[0]:
                hh = jnp.pad(hh, ((0, 0), (0, dwr.shape[0] - hh.shape[-1])))
            partial = jnp.dot(hh, dwr, preferred_element_type=F32)
            hidden = hidden + jax.lax.psum(partial, "tp").astype(jnp.bfloat16)
            if li in tap_set:
                taps.append(hidden)

        feats = jnp.concatenate(taps, axis=-1)  # [B, 5*dim] bf16, replicated
        normed = rms(hidden, w["fnorm"], c.eps)
        logits = jnp.dot(normed, w["lmw"], preferred_element_type=F32)  # [B, vshard]
        new_states = {
            "conv": jnp.stack(new_conv),
            "rec": jnp.stack(new_rec),
            "kcache": jnp.stack(new_k),
            "vcache": jnp.stack(new_v),
        }
        snaps = {
            "conv": jnp.stack(conv_snaps),  # [nl, B, C, 4]
            "rec": jnp.stack(rec_snaps),  # [nl, B, vh, vd, kd]
        }
        return (
            logits[None],
            feats,
            jax.tree.map(lambda a: a[None], new_states),
            jax.tree.map(lambda a: a[None], snaps),
        )

    def verify(w, states, tokens, pos0, key_lo):
        ws = jax.tree.map(lambda _: jax.sharding.PartitionSpec("tp"), w)
        ss = jax.tree.map(lambda _: jax.sharding.PartitionSpec("tp"), states)
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(ws, ss, PNone(), PNone(), PNone()),
            out_specs=(
                jax.sharding.PartitionSpec("tp"),
                jax.sharding.PartitionSpec(),
                ss,
                {"conv": jax.sharding.PartitionSpec("tp"), "rec": jax.sharding.PartitionSpec("tp")},
            ),
            check_vma=False,
        )(w, states, tokens, pos0, key_lo)

    return verify


def transformer_stack(c, token, w, states, position, context, ablate=frozenset(), tp=2):
    """Run the fused stack; returns (logits [1, lm_width] fp32, updated states)."""
    if c.layers % 4 or c.head_dim != 256 or context % 128:
        raise ValueError("layers%4, head_dim=256, context%128 are required")
    if c.heads % tp or (c.kv_heads % tp and tp % c.kv_heads):
        raise ValueError("heads must split evenly; KV heads split or pair-replicate")
    h = c.dim
    nh = c.heads // tp
    nkv = max(1, c.kv_heads // tp)
    hd = c.head_dim
    vh = c.v_heads // tp
    kh = c.k_heads // tp
    kl = kh * c.k_dim
    vl = vh * c.v_dim
    qkvw = kl * 2 + vl
    inter = c.intermediate // tp
    lm_width = _lm_width_fn(c, tp)
    groups = nh // nkv
    if nh % nkv or groups > 8:
        raise ValueError("GQA group must fit one 8-row MXU tile")
    att_w = max(nh * hd, vl)
    lin, full, lm, off_lin, off_full, t_lin, t_full, t_block = schedule(
        c, tp, lm_width
    )
    t_layers = c.layers // 4 * t_block
    t_total = t_layers + lm[-1]
    wrefs = {}

    def body(
        pr,
        tokenr,
        qkvz_r,
        outw_r,
        qwkv_r,
        ow_r,
        gu_r,
        dw_r,
        lmw_r,
        norm_in_r,
        norm_post_r,
        convw_r,
        alog_r,
        dtb_r,
        onorm_r,
        qn_r,
        kn_r,
        fnorm_r,
        emb_r,
        conv_r,
        rec_r,
        kc_r,
        vc_r,
        logits_out,
        token_out,
        conv_out,
        rec_out,
        kc_out,
        vc_out,
        *refs,
    ):
        (
            banks,
            ws,
            scale,
            norm_sems,
            residual,
            normed,
            proj,
            accum,
            lm_accum,
            qkvzbuf,
            qwkvbuf,
            attbuf,
            gubuf,
            hbuf,
            recv,
            sends,
            recvs,
            sbuf,
            s_sem,
            s_io,
            cbuf,
            cwb,
            abuf,
            bbuf,
            onbuf,
            c_sem,
            cw_sem,
            a_sem,
            on_sem,
            c_io,
            attended,
            qb,
            kb,
            vb,
            am,
            al,
            aa,
            ks,
            vs,
            io,
            qnbuf,
            knbuf,
            fnbuf,
            qn_sem,
            kn_sem,
            fn_sem,
            rowbuf,
            e_sem,
            pairbuf,
            pairs,
        ) = refs
        rank = jax.lax.axis_index("tp")
        pos = pr[0]
        key_lo = pr[1]
        wrefs.update(qkvz=qkvz_r, outw=outw_r, qwkv=qwkv_r, ow=ow_r, gu=gu_r, dw=dw_r)

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
                    blk = g // t_block
                    sub = g % t_block

                    @pl.when(sub < 3 * t_lin)
                    def linear_tile():
                        j = sub // t_lin
                        rem = sub - j * t_lin
                        ord_ = blk * 3 + j
                        for i, (name, _, bk, bn, nk, nb, nt) in enumerate(lin):

                            @pl.when((rem >= off_lin[i]) & (rem < off_lin[i] + nt))
                            def load():
                                ref = wrefs[name]
                                copy_tile(
                                    ref,
                                    blk * 4 + j if name in ("gu", "dw") else ord_,
                                    g % BANK_COUNT,
                                    rem - off_lin[i],
                                    bk,
                                    bn,
                                ).start()

                    @pl.when(sub >= 3 * t_lin)
                    def full_tile():
                        rem = sub - 3 * t_lin
                        for i, (name, _, bk, bn, nk, nb, nt) in enumerate(full):

                            @pl.when((rem >= off_full[i]) & (rem < off_full[i] + nt))
                            def load():
                                ref = wrefs[name]
                                copy_tile(
                                    ref,
                                    blk * 4 + 3 if name in ("gu", "dw") else blk,
                                    g % BANK_COUNT,
                                    rem - off_full[i],
                                    bk,
                                    bn,
                                ).start()

                @pl.when(g >= t_layers)
                def lm_tile():
                    _, _, bk, bn, nk, nb, nt = lm
                    copy_tile(lmw_r, None, g % BANK_COUNT, g - t_layers, bk, bn).start()

        def gemv(src, dst, weight, li, base, slot, out_f32=False, acc=None):
            """Matvec [1,K] x [K,N] streamed through the bank ring."""
            _, _, bk, bn, nk, nb, nt = slot
            n = nb * bn
            ar = (accum if acc is None else acc).at[:, pl.ds(0, n)]
            ar[...] = jnp.zeros((8, n), jnp.float32)

            def step(ti, bank):
                ki, ni = ti % nk, ti // nk
                copy_tile(weight, li, bank, ti, bk, bn).wait()
                a = jnp.broadcast_to(src[:, pl.ds(ki * bk, bk)], (8, bk))
                ar[:, pl.ds(ni * bn, bn)] += jnp.dot(
                    a,
                    banks[bank, pl.ds(0, bk), pl.ds(0, bn)],
                    preferred_element_type=jnp.float32,
                )
                fetch(base + ti + BANK_COUNT)
                return jnp.where(bank + 1 == BANK_COUNT, 0, bank + 1)

            jax.lax.fori_loop(0, nt, step, base % BANK_COUNT)
            dst[...] = ar[:1, :].astype(jnp.float32 if out_f32 else dst.dtype)

        def allreduce(iteration):
            """All-reduce proj over all ranks; returns the fp32 total (1, h)."""
            slot = iteration % 2
            recv[slot, rank, ...] = proj[...]
            for offset in range(1, tp):
                tpu.make_async_remote_copy(
                    proj,
                    recv.at[slot, rank],
                    sends.at[offset - 1],
                    recvs.at[offset - 1],
                    device_id=(rank ^ offset,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()
            for offset in range(1, tp):
                tpu.make_async_remote_copy(
                    proj,
                    recv.at[slot, rank],
                    sends.at[offset - 1],
                    recvs.at[offset - 1],
                    device_id=(rank ^ offset,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).wait()
            return jnp.sum(recv[slot, ...].astype(jnp.float32), axis=0)

        def exchange_add(iteration):
            if "noexchange" in ablate or "nocompute" in ablate:
                residual[...] = residual[...] + proj[...].astype(residual.dtype)
                return
            residual[...] = residual[...] + allreduce(iteration).astype(residual.dtype)

        def norm(weight, li, slot):
            """Zero-centered RMSNorm; prefetches row li+1 into the same slot."""
            sr = scale.at[slot]
            sem = norm_sems.at[slot]
            tpu.make_async_copy(weight.at[li, :, :], sr, sem).wait()
            value = residual[...].astype(jnp.float32)
            value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + c.eps)
            normed[...] = (value * (1.0 + sr[...].astype(jnp.float32))).astype(normed.dtype)

            @pl.when(li + 1 < c.layers)
            def next_norm():
                tpu.make_async_copy(weight.at[li + 1, :, :], sr, sem).start()

        def rms_zc(vec, w1p):
            value = vec.astype(jnp.float32)
            value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + c.eps)
            return (value * w1p).astype(jnp.bfloat16)

        def rope(vec):
            half = c.rotary_dim // 2
            rot, rest = vec[..., : c.rotary_dim], vec[..., c.rotary_dim :]
            a, b = rot[..., :half], rot[..., half:]
            cv, sv = cos_v[:half], sin_v[:half]
            return jnp.concatenate((a * cv - b * sv, b * cv + a * sv, rest), axis=-1).astype(
                jnp.bfloat16
            )

        def linear_attention(li):
            ord_ = li - (li + 1) // 4

            @pl.when(ord_ > 0)
            def wait_prior_writebacks():
                tpu.make_async_copy(cbuf, conv_r.at[0], c_io.at[0]).wait()
                tpu.make_async_copy(sbuf, rec_r.at[0], s_io.at[0]).wait()

            noaux = "noaux" in ablate
            tpu.make_async_copy(conv_r.at[ord_], cbuf, c_sem.at[0]).start()
            tpu.make_async_copy(rec_r.at[ord_], sbuf, s_sem.at[0]).start()
            if not noaux:
                tpu.make_async_copy(convw_r.at[ord_], cwb, cw_sem.at[0]).start()
                tpu.make_async_copy(alog_r.at[ord_], abuf, a_sem.at[0]).start()
                tpu.make_async_copy(dtb_r.at[ord_], bbuf, a_sem.at[0]).start()
                tpu.make_async_copy(onorm_r.at[ord_], onbuf, on_sem.at[0]).start()
            base = (li // 4) * t_block + (li % 4) * t_lin
            gemv(normed, qkvzbuf, qkvz_r, ord_, base + off_lin[0], lin[0])

            skip_delta = "nodelta" in ablate or "nocompute" in ablate
            tpu.make_async_copy(conv_r.at[ord_], cbuf, c_sem.at[0]).wait()
            if not noaux:
                tpu.make_async_copy(convw_r.at[ord_], cwb, cw_sem.at[0]).wait()
            if not skip_delta:
                hist = jnp.concatenate((cbuf[...], qkvzbuf[0, :qkvw].reshape(qkvw, 1)), axis=1)
                filtered = jax.nn.silu(
                    sum(
                        hist[:, 1 + tap].astype(jnp.float32) * cwb[...][:, tap].astype(jnp.float32)
                        for tap in range(c.conv_size)
                    )
                ).astype(jnp.bfloat16)
                cbuf[...] = hist[:, 1:]
                q = filtered[:kl].reshape(kh, c.k_dim).astype(jnp.float32)
                k = filtered[kl : 2 * kl].reshape(kh, c.k_dim).astype(jnp.float32)
                v = filtered[2 * kl :].reshape(vh, c.v_dim).astype(jnp.float32)
                q = q * jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
                q = q * c.k_dim**-0.5
                k = k * jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
                expand = c.v_heads // c.k_heads
                q = jnp.broadcast_to(q.reshape(kh, 1, c.k_dim), (kh, expand, c.k_dim)).reshape(
                    vh, -1
                )
                k = jnp.broadcast_to(k.reshape(kh, 1, c.k_dim), (kh, expand, c.k_dim)).reshape(
                    vh, -1
                )
            tpu.make_async_copy(cbuf, conv_r.at[ord_], c_io.at[0]).start()

            if not skip_delta:
                beta = jax.nn.sigmoid(qkvzbuf[0, qkvw + vl : qkvw + vl + vh].astype(jnp.float32))
                beta = beta.astype(jnp.bfloat16).astype(jnp.float32)
                if not noaux:
                    tpu.make_async_copy(alog_r.at[ord_], abuf, a_sem.at[0]).wait()
                    tpu.make_async_copy(dtb_r.at[ord_], bbuf, a_sem.at[0]).wait()
                gate = -jnp.exp(abuf[0].astype(jnp.float32)) * jax.nn.softplus(
                    qkvzbuf[0, qkvw + vl + vh : qkvw + vl + 2 * vh].astype(jnp.float32)
                    + bbuf[0].astype(jnp.float32)
                )
            tpu.make_async_copy(rec_r.at[ord_], sbuf, s_sem.at[0]).wait()
            if not skip_delta:
                sbuf[...] = sbuf[...] * jnp.exp(gate)[:, None, None]
                prediction = jnp.sum(sbuf[...] * k[:, None, :], axis=-1)
                delta = (v - prediction) * beta[:, None]
                sbuf[...] = sbuf[...] + delta[:, :, None] * k[:, None, :]
                out = jnp.sum(sbuf[...] * q[:, None, :], axis=-1).astype(jnp.bfloat16)
            tpu.make_async_copy(sbuf, rec_r.at[ord_], s_io.at[0]).start()

            if not skip_delta:
                value = out.astype(jnp.float32)
                value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + c.eps)
                if not noaux:
                    tpu.make_async_copy(onorm_r.at[ord_], onbuf, on_sem.at[0]).wait()
                value = (onbuf[...] * value.astype(jnp.bfloat16)).astype(jnp.float32)
                z = qkvzbuf[0, qkvw : qkvw + vl].reshape(vh, c.v_dim).astype(jnp.float32)
                attbuf[0, pl.ds(0, vl)] = (
                    (value * jax.nn.silu(z)).astype(jnp.bfloat16).reshape(vl)
                )
            gemv(attbuf, proj, outw_r, ord_, base + off_lin[1], lin[1], out_f32=True)
            exchange_add(li * 2)

        def attend(ord2, query, key, value):
            for g in range(nkv):
                qb[...] = jnp.zeros((8, hd), jnp.bfloat16)
                qb[pl.ds(0, groups), :] = query[g * groups : (g + 1) * groups]

                count = pos // 128 + 1
                for t in range(2):

                    @pl.when(t <= pos // 128)
                    def issue():
                        tpu.make_async_copy(
                            kc_r.at[ord2, g, pl.ds(t * 128, 128), :], kb.at[t], ks.at[t]
                        ).start()
                        tpu.make_async_copy(
                            vc_r.at[ord2, g, pl.ds(t * 128, 128), :], vb.at[t], vs.at[t]
                        ).start()

                am[...] = jnp.full((8, 1), -jnp.inf, jnp.float32)
                al[...] = jnp.zeros((8, 1), jnp.float32)
                aa[...] = jnp.zeros((8, hd), jnp.float32)

                def step(t, _):
                    tpu.make_async_copy(
                        kc_r.at[ord2, g, pl.ds(t * 128, 128), :], kb.at[t % 2], ks.at[t % 2]
                    ).wait()
                    tpu.make_async_copy(
                        vc_r.at[ord2, g, pl.ds(t * 128, 128), :], vb.at[t % 2], vs.at[t % 2]
                    ).wait()

                    @pl.when(t == pos // 128)
                    def update_current():
                        mask = jnp.arange(128)[:, None] == pos % 128
                        kb[t % 2, ...] = jnp.where(mask, key[g][None, :], kb[t % 2, ...])
                        vb[t % 2, ...] = jnp.where(mask, value[g][None, :], vb[t % 2, ...])
                        tpu.make_async_copy(
                            kb.at[t % 2], kc_r.at[ord2, g, pl.ds(pos // 128 * 128, 128), :], io.at[0]
                        ).start()
                        tpu.make_async_copy(
                            vb.at[t % 2], vc_r.at[ord2, g, pl.ds(pos // 128 * 128, 128), :], io.at[1]
                        ).start()

                    if "noattn" not in ablate and "nocompute" not in ablate:
                        scores = (
                            jnp.dot(
                                qb[...],
                                kb[t % 2, ...].reshape(128, hd).T,
                                preferred_element_type=jnp.float32,
                            )
                            * hd**-0.5
                        )
                        scores = jnp.where(
                            (jnp.arange(128)[None, :] + t * 128 <= pos)
                            & (jnp.arange(128)[None, :] + t * 128 >= key_lo),
                            scores,
                            -jnp.inf,
                        )
                        m = jnp.maximum(am[...], jnp.max(scores, axis=-1, keepdims=True))
                        alpha = jnp.exp(am[...] - m)
                        p = jnp.exp(scores - m).astype(jnp.bfloat16)
                        aa[...] = aa[...] * alpha + jnp.dot(
                            p.astype(jnp.float32),
                            vb[t % 2, ...].astype(jnp.float32),
                            preferred_element_type=jnp.float32,
                        )
                        al[...] = al[...] * alpha + jnp.sum(
                            p.astype(jnp.float32), axis=-1, keepdims=True
                        )
                        am[...] = m

                    @pl.when(t + 2 < count)
                    def refill():
                        tpu.make_async_copy(
                            kc_r.at[ord2, g, pl.ds((t + 2) * 128, 128), :],
                            kb.at[t % 2],
                            ks.at[t % 2],
                        ).start()
                        tpu.make_async_copy(
                            vc_r.at[ord2, g, pl.ds((t + 2) * 128, 128), :],
                            vb.at[t % 2],
                            vs.at[t % 2],
                        ).start()

                jax.lax.fori_loop(0, count, step, None)
                attended[pl.ds(g * groups, groups), :] = (aa[...] / al[...])[:groups].astype(
                    jnp.bfloat16
                )
                tpu.make_async_copy(
                    kb.at[pos // 128 % 2], kc_r.at[ord2, g, pl.ds(pos // 128 * 128, 128), :], io.at[0]
                ).wait()
                tpu.make_async_copy(
                    vb.at[pos // 128 % 2], vc_r.at[ord2, g, pl.ds(pos // 128 * 128, 128), :], io.at[1]
                ).wait()

        def full_attention(li):
            ord2 = (li + 1) // 4 - 1

            base = (li // 4) * t_block + 3 * t_lin
            if "noaux" not in ablate:
                tpu.make_async_copy(qn_r.at[ord2], qnbuf, qn_sem.at[0]).start()
                tpu.make_async_copy(kn_r.at[ord2], knbuf, kn_sem.at[0]).start()
            gemv(normed, qwkvbuf, qwkv_r, ord2, base + off_full[0], full[0])
            packed = qwkvbuf[0, : nh * 2 * hd].reshape(nh, 2 * hd)
            query, gate = packed[:, :hd], packed[:, hd:]
            if "noaux" not in ablate:
                tpu.make_async_copy(qn_r.at[ord2], qnbuf, qn_sem.at[0]).wait()
                tpu.make_async_copy(kn_r.at[ord2], knbuf, kn_sem.at[0]).wait()
            query = rms_zc(query, 1.0 + qnbuf[...].astype(jnp.float32))
            kv_off = nh * 2 * hd
            key = rms_zc(
                qwkvbuf[0, kv_off : kv_off + nkv * hd].reshape(nkv, hd),
                1.0 + knbuf[...].astype(jnp.float32),
            )
            value = qwkvbuf[0, kv_off + nkv * hd : kv_off + 2 * nkv * hd].reshape(nkv, hd)
            query = rope(query)
            key = rope(key)
            attend(ord2, query, key, value)
            gated = attended[...] * jax.nn.sigmoid(gate.astype(jnp.float32)).astype(jnp.bfloat16)
            attbuf[0, pl.ds(0, nh * hd)] = gated.reshape(nh * hd)
            gemv(attbuf, proj, ow_r, ord2, base + off_full[1], full[1], out_f32=True)
            exchange_add(li * 2)

        def mlp(li):
            base = (li // 4) * t_block + jnp.where(
                li % 4 == 3, 3 * t_lin, (li % 4) * t_lin
            )
            off3 = jnp.where(li % 4 == 3, off_full[2], off_lin[2])
            off4 = jnp.where(li % 4 == 3, off_full[3], off_lin[3])
            gemv(normed, gubuf, gu_r, li, base + off3, lin[2])
            gate = jax.nn.silu(gubuf[0, :inter].astype(jnp.float32)).astype(jnp.bfloat16)
            hbuf[0, pl.ds(0, inter)] = gate * gubuf[0, inter : 2 * inter]
            gemv(hbuf, proj, dw_r, li, base + off4, lin[3], out_f32=True)
            exchange_add(li * 2 + 1)

        def layer(li, _):
            norm(norm_in_r, li, 0)
            jax.lax.cond(li % 4 == 3, full_attention, linear_attention, li)
            norm(norm_post_r, li, 1)
            mlp(li)

        # RoPE tables computed in-kernel (text mrope == standard partial RoPE).
        half = c.rotary_dim // 2
        inv = 1.0 / (c.rope_theta ** (jax.lax.iota(jnp.int32, half).astype(jnp.float32) * 2.0 / c.rotary_dim))
        ang = pos.astype(jnp.float32) * inv
        cos_v = jnp.concatenate((jnp.cos(ang), jnp.cos(ang))).astype(jnp.bfloat16)
        sin_v = jnp.concatenate((jnp.sin(ang), jnp.sin(ang))).astype(jnp.bfloat16)

        # Embedding: fetch this rank's candidate row, keep it only if in-range.
        tok = tokenr[...][0]
        vshard = c.vocab // tp
        lo = rank * vshard
        tokl = jnp.clip(tok - lo, 0, vshard - 1)
        tpu.make_async_copy(
            emb_r.at[pl.ds(tokl // 8 * 8, 8), :], rowbuf, e_sem.at[0]
        ).start()
        tpu.make_async_copy(norm_in_r.at[0, :, :], scale.at[0], norm_sems.at[0]).start()
        tpu.make_async_copy(norm_post_r.at[0, :, :], scale.at[1], norm_sems.at[1]).start()
        for ti in range(BANK_COUNT):
            fetch(jnp.int32(ti))
        barrier = tpu.get_barrier_semaphore()
        for offset in range(1, tp):
            pl.semaphore_signal(
                barrier, 1, device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(barrier, tp - 1)
        hbuf[...] = jnp.zeros_like(hbuf[...])
        tpu.make_async_copy(
            emb_r.at[pl.ds(tokl // 8 * 8, 8), :], rowbuf, e_sem.at[0]
        ).wait()
        sel = jax.lax.iota(jnp.int32, 8)
        row = jnp.sum(
            jnp.where(sel[:, None] == tokl % 8, rowbuf[...], 0), axis=0
        ).astype(jnp.float32)
        row = jnp.where((tok >= lo) & (tok < lo + vshard), row, 0.0)
        proj[...] = row[None, :]
        residual[...] = allreduce(1).astype(residual.dtype)
        jax.lax.fori_loop(0, c.layers, layer, None)
        # Drain the trailing gated-deltanet state writebacks.
        tpu.make_async_copy(cbuf, conv_r.at[0], c_io.at[0]).wait()
        tpu.make_async_copy(sbuf, rec_r.at[0], s_io.at[0]).wait()
        tpu.make_async_copy(fnorm_r, fnbuf, fn_sem.at[0]).start()
        tpu.make_async_copy(fnorm_r, fnbuf, fn_sem.at[0]).wait()
        value = residual[...].astype(jnp.float32)
        value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + c.eps)
        normed[...] = (value * (1.0 + fnbuf[...].astype(jnp.float32))).astype(normed.dtype)
        gemv(normed, logits_out, lmw_r, None, t_layers, lm, out_f32=True, acc=lm_accum)

        # Greedy next-token selection: per-rank argmax, then a (value, index)
        # pair exchange picks the global winner on every rank.
        vals = lm_accum[0, :vshard]
        top = jnp.max(vals)
        iota = jax.lax.iota(jnp.int32, vshard)
        best = jnp.min(jnp.where(vals == top, iota, vshard)).astype(jnp.int32)
        pairbuf[...] = jnp.concatenate((top[None], best.astype(jnp.float32)[None]))
        pairs[0, rank, ...] = pairbuf[...]
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
        # pairs hold per-rank LOCAL argmax indices; offset by the winner's shard.
        token_id = winner * vshard + jnp.sum(jnp.where(wiota == winner, pairs[0, :, 1], 0.0))
        token_out[...] = token_id.astype(jnp.int32).reshape(1)
    nl = c.layers - len(c.full_attention)
    nf = len(c.full_attention)
    shape = lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype)
    v = lambda shp, dtype=jnp.bfloat16: tpu.VMEM(shp, dtype)
    dm = lambda n: tpu.SemaphoreType.DMA((n,))
    states_in = (states["conv"], states["rec"], states["kcache"], states["vcache"])
    weights_in = (
        w["qkvz"],
        w["outw"],
        w["qwkv"],
        w["ow"],
        w["gu"],
        w["dw"],
        w["lmw"],
        w["norm_in"][:, None, :],
        w["norm_post"][:, None, :],
        w["convw"],
        w["alog"][:, None, :],
        w["dtb"][:, None, :],
        w["onorm"][:, None, :],
        w["qn"][:, None, :],
        w["kn"][:, None, :],
        w["fnorm"][None, :],
        w["emb"],
    )
    args = (token, *weights_in, *states_in)
    logits_shape = jax.ShapeDtypeStruct((1, lm_width), jnp.float32)
    scratch = (
        v((BANK_COUNT, MAX_BANK_K, MAX_BANK_N)),
        dm(BANK_COUNT),
        v((2, 1, h)),
        dm(2),
        v((1, h)),
        v((1, h)),
        v((1, h), jnp.float32),
        v((8, gu_width(c, tp)), jnp.float32),
        v((8, lm_width), jnp.float32),
        v((1, qkvw + zba_width(c, tp))),
        v((1, nh * 2 * hd + 2 * nkv * hd)),
        v((1, att_w)),
        v((1, gu_width(c, tp))),
        v((1, dw_k(c, tp))),
        v((2, tp, 1, h), jnp.float32),
        dm(tp - 1),
        dm(tp - 1),
        v((vh, c.v_dim, c.k_dim), jnp.float32),
        dm(1),
        dm(1),
        v((qkvw, c.conv_size)),
        v((qkvw, c.conv_size)),
        v((1, vh)),
        v((1, vh)),
        v((1, c.v_dim)),
        dm(1),
        dm(1),
        dm(1),
        dm(1),
        dm(1),
        v((nh, hd)),
        v((8, hd)),
        v((2, 128, hd)),
        v((2, 128, hd)),
        v((8, 1), jnp.float32),
        v((8, 1), jnp.float32),
        v((8, hd), jnp.float32),
        dm(2),
        dm(2),
        dm(2),
        v((1, hd)),
        v((1, hd)),
        v((1, h)),
        dm(1),
        dm(1),
        dm(1),
        v((8, h)),
        dm(1),
        v((2,), jnp.float32),
        v((2, tp, 2), jnp.float32),
    )
    state_offset = 1 + 1 + len(weights_in)
    result = pl.pallas_call(
        body,
        out_shape=(logits_shape, jax.ShapeDtypeStruct((1,), jnp.int32), *(shape(s) for s in states_in)),
        input_output_aliases={state_offset + i: i + 2 for i in range(4)},
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[pl.BlockSpec()] + [pl.BlockSpec(memory_space=tpu.HBM)] * (len(args) - 1),
            out_specs=[pl.BlockSpec()] * 2 + [pl.BlockSpec(memory_space=tpu.HBM)] * 4,
            scratch_shapes=scratch,
        ),
        compiler_params=tpu.CompilerParams(
            collective_id=45,
            vmem_limit_bytes=64 * 1024**2,
            disable_bounds_checks=True,
            shape_invariant_numerics=True,
        ),
        name="qwen_transformer_stack",
    )(position, *args)
    return result[0], result[1], dict(zip(("conv", "rec", "kcache", "vcache"), result[2:]))


def transformer_stack_block(c, tokens, w, states, pospair, snaps, context, tp, tap_layers, block=8, ablate=frozenset(), bank_count=None):
    """8-token verify variant of transformer_stack for DFlash speculative decoding.

    One weight pass processes `block` tokens (anchor + drafts) at cache slots
    [pos0, pos0+block). Returns per-position argmax tokens (host computes
    acceptance) and per-position conv/rec snapshots for rollback. Hidden taps
    (layers tap_layers) are DMA'd into featbuf slots [pos0, pos0+block) for the
    drafter. The aliased conv/rec states receive the post-block state (host
    replaces them from snapshots); kcache/vcache are written positionally.
    """
    if c.layers % 4 or c.head_dim != 256 or context % 128:
        raise ValueError("layers%4, head_dim=256, context%128 are required")
    if c.heads % tp or (c.kv_heads % tp and tp % c.kv_heads):
        raise ValueError("heads must split evenly; KV heads split or pair-replicate")
    B = block
    h = c.dim
    nh = c.heads // tp
    nkv = max(1, c.kv_heads // tp)
    hd = c.head_dim
    vh = c.v_heads // tp
    kh = c.k_heads // tp
    kl = kh * c.k_dim
    vl = vh * c.v_dim
    qkvw = kl * 2 + vl
    inter = c.intermediate // tp
    lm_width = _lm_width_fn(c, tp)
    groups = nh // nkv
    if nh % nkv or groups > 8:
        raise ValueError("GQA group must fit one 8-row MXU tile")
    att_w = max(nh * hd, vl)
    lin, full, lm, off_lin, off_full, t_lin, t_full, t_block = schedule(
        c, tp, lm_width
    )
    t_layers = c.layers // 4 * t_block
    t_total = t_layers + lm[-1]
    taps = tuple(tap_layers)
    nl = c.layers - len(c.full_attention)
    bc = bank_count or BANK_COUNT
    wrefs = {}

    def body(
        pr,
        tokenr,
        qkvz_r,
        outw_r,
        qwkv_r,
        ow_r,
        gu_r,
        dw_r,
        lmw_r,
        norm_in_r,
        norm_post_r,
        convw_r,
        alog_r,
        dtb_r,
        onorm_r,
        qn_r,
        kn_r,
        fnorm_r,
        emb_r,
        conv_r,
        rec_r,
        kc_r,
        vc_r,
        convsnap_r,
        recsnap_r,
        post_out,
        taps_out,
        conv_snap_out,
        rec_snap_out,
        conv_out,
        rec_out,
        kc_out,
        vc_out,
        *refs,
    ):
        (
            banks,
            ws,
            scale,
            norm_sems,
            residual,
            normed,
            proj,
            accum,
            lm_accum,
            qkvzbuf,
            qwkvbuf,
            attbuf,
            gubuf,
            hbuf,
            rs_recv,
            rounded,
            ag_recv,
            sends,
            recvs,
            rsw_sends,
            rsw_recvs,
            ag_sends,
            ag_recvs,
            sbuf,
            s_sem,
            s_io,
            cbuf,
            cwb,
            abuf,
            bbuf,
            onbuf,
            c_sem,
            cw_sem,
            a_sem,
            on_sem,
            c_io,
            attended,
            qb,
            kb,
            vb,
            am,
            al,
            aa,
            ks,
            vs,
            io,
            qnbuf,
            knbuf,
            fnbuf,
            qn_sem,
            kn_sem,
            fn_sem,
            rowbuf,
            e_sem,
            pairbuf,
            pairs,
            recsnaps,
            convsnaps,
            snap_sem,
            kb8,
            vb8,
        ) = refs
        rank = jax.lax.axis_index("tp")
        pos0 = pr[0]
        key_lo = pr[1]
        wrefs.update(qkvz=qkvz_r, outw=outw_r, qwkv=qwkv_r, ow=ow_r, gu=gu_r, dw=dw_r)
        iotaB = jax.lax.iota(jnp.int32, B)
        posv = pos0 + iotaB  # [B] absolute slot positions
        iota128 = jax.lax.iota(jnp.int32, 128)

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
                    blk = g // t_block
                    sub = g % t_block

                    @pl.when(sub < 3 * t_lin)
                    def linear_tile():
                        j = sub // t_lin
                        rem = sub - j * t_lin
                        ord_ = blk * 3 + j
                        for i, (name, _, bk, bn, nk, nb, nt) in enumerate(lin):

                            @pl.when((rem >= off_lin[i]) & (rem < off_lin[i] + nt))
                            def load():
                                ref = wrefs[name]
                                copy_tile(
                                    ref,
                                    blk * 4 + j if name in ("gu", "dw") else ord_,
                                    g % bc,
                                    rem - off_lin[i],
                                    bk,
                                    bn,
                                ).start()

                    @pl.when(sub >= 3 * t_lin)
                    def full_tile():
                        rem = sub - 3 * t_lin
                        for i, (name, _, bk, bn, nk, nb, nt) in enumerate(full):

                            @pl.when((rem >= off_full[i]) & (rem < off_full[i] + nt))
                            def load():
                                ref = wrefs[name]
                                copy_tile(
                                    ref,
                                    blk * 4 + 3 if name in ("gu", "dw") else blk,
                                    g % bc,
                                    rem - off_full[i],
                                    bk,
                                    bn,
                                ).start()

                @pl.when(g >= t_layers)
                def lm_tile():
                    _, _, bk, bn, nk, nb, nt = lm
                    copy_tile(lmw_r, None, g % bc, g - t_layers, bk, bn).start()

        def gemv(src, dst, weight, li, base, slot, out_f32=False, acc=None):
            """[B,K] x [K,N] streamed through the bank ring."""
            _, _, bk, bn, nk, nb, nt = slot
            n = nb * bn
            ar = (accum if acc is None else acc).at[:, pl.ds(0, n)]
            ar[...] = jnp.zeros((B, n), jnp.float32)

            def step(ti, bank):
                ki, ni = ti % nk, ti // nk
                copy_tile(weight, li, bank, ti, bk, bn).wait()
                a = src[:, pl.ds(ki * bk, bk)]
                ar[:, pl.ds(ni * bn, bn)] += jnp.dot(
                    a,
                    banks[bank, pl.ds(0, bk), pl.ds(0, bn)],
                    preferred_element_type=jnp.float32,
                )
                fetch(base + ti + bc)
                return jnp.where(bank + 1 == bc, 0, bank + 1)

            jax.lax.fori_loop(0, nt, step, base % bc)
            dst[...] = ar[...].astype(jnp.float32 if out_f32 else dst.dtype)

        def allreduce(iteration):
            """Reduce-scatter in f32, round once, then all-gather in bf16."""
            slot = iteration % 2
            sw = h // tp
            own = jnp.zeros((B, sw), jnp.float32)
            for j in range(tp):
                own = jnp.where(rank == j, proj[...][:, j * sw : (j + 1) * sw], own)
            rs_recv[slot, rank, ...] = own
            for offset in range(1, tp):
                tgt = rank ^ offset
                tpu.make_async_remote_copy(
                    proj.at[:, pl.ds(tgt * sw, sw)],
                    rs_recv.at[slot, rank],
                    rsw_sends.at[offset - 1],
                    rsw_recvs.at[offset - 1],
                    device_id=(tgt,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()
            for offset in range(1, tp):
                tgt = rank ^ offset
                tpu.make_async_remote_copy(
                    proj.at[:, pl.ds(tgt * sw, sw)],
                    rs_recv.at[slot, rank],
                    rsw_sends.at[offset - 1],
                    rsw_recvs.at[offset - 1],
                    device_id=(tgt,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).wait()
            rounded[...] = jnp.sum(rs_recv[slot, ...], axis=0).astype(jnp.bfloat16)
            ag_recv[slot, rank, ...] = rounded[...]
            for offset in range(1, tp):
                tpu.make_async_remote_copy(
                    rounded,
                    ag_recv.at[slot, rank],
                    ag_sends.at[offset - 1],
                    ag_recvs.at[offset - 1],
                    device_id=(rank ^ offset,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()
            for offset in range(1, tp):
                tpu.make_async_remote_copy(
                    rounded,
                    ag_recv.at[slot, rank],
                    ag_sends.at[offset - 1],
                    ag_recvs.at[offset - 1],
                    device_id=(rank ^ offset,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).wait()
            return ag_recv[slot, ...].transpose(1, 0, 2).reshape(B, h)

        def exchange_add(iteration):
            if "noexchange" in ablate or "nocompute" in ablate:
                residual[...] = residual[...] + proj[...].astype(residual.dtype)
                return
            residual[...] = residual[...] + allreduce(iteration).astype(residual.dtype)

        def norm(weight, li, slot):
            sr = scale.at[slot]
            sem = norm_sems.at[slot]
            tpu.make_async_copy(weight.at[li, :, :], sr, sem).wait()
            value = residual[...].astype(jnp.float32)
            value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + c.eps)
            normed[...] = (value * (1.0 + sr[...].astype(jnp.float32))).astype(normed.dtype)

            @pl.when(li + 1 < c.layers)
            def next_norm():
                tpu.make_async_copy(weight.at[li + 1, :, :], sr, sem).start()

        def rms_zc(vec, w1p):
            value = vec.astype(jnp.float32)
            value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + c.eps)
            return (value * w1p).astype(jnp.bfloat16)

        def ropeb(vec):
            """vec [B, heads, hd] with per-position tables cos_tb/sin_tb [B, rotary]."""
            half = c.rotary_dim // 2
            rot, rest = vec[..., : c.rotary_dim], vec[..., c.rotary_dim :]
            a, b = rot[..., :half], rot[..., half:]
            cv = cos_tb[:, None, :half]
            sv = sin_tb[:, None, :half]
            return jnp.concatenate((a * cv - b * sv, b * cv + a * sv, rest), axis=-1).astype(
                jnp.bfloat16
            )

        def linear_attention(li):
            ord_ = li - (li + 1) // 4

            @pl.when(ord_ > 0)
            def wait_prior_writebacks():
                tpu.make_async_copy(cbuf, conv_r.at[0], c_io.at[0]).wait()
                tpu.make_async_copy(sbuf, rec_r.at[0], s_io.at[0]).wait()
                tpu.make_async_copy(convsnaps, convsnap_r.at[0], snap_sem.at[0]).wait()
                tpu.make_async_copy(recsnaps.at[0], recsnap_r.at[0, 0], snap_sem.at[1]).wait()
                tpu.make_async_copy(recsnaps.at[0], recsnap_r.at[0, 0], snap_sem.at[2]).wait()

            tpu.make_async_copy(conv_r.at[ord_], cbuf, c_sem.at[0]).start()
            tpu.make_async_copy(rec_r.at[ord_], sbuf, s_sem.at[0]).start()
            tpu.make_async_copy(convw_r.at[ord_], cwb, cw_sem.at[0]).start()
            tpu.make_async_copy(alog_r.at[ord_], abuf, a_sem.at[0]).start()
            tpu.make_async_copy(dtb_r.at[ord_], bbuf, a_sem.at[0]).start()
            tpu.make_async_copy(onorm_r.at[ord_], onbuf, on_sem.at[0]).start()
            base = (li // 4) * t_block + (li % 4) * t_lin
            gemv(normed, qkvzbuf, qkvz_r, ord_, base + off_lin[0], lin[0])

            tpu.make_async_copy(conv_r.at[ord_], cbuf, c_sem.at[0]).wait()
            tpu.make_async_copy(convw_r.at[ord_], cwb, cw_sem.at[0]).wait()
            # batched causal conv over the block; state = last 4 raw qkv inputs
            xp = jnp.concatenate((cbuf[...].T, qkvzbuf[:, :qkvw]), axis=0)  # [4+B, qkvw]
            accv = jnp.zeros((B, qkvw), jnp.float32)
            for tap in range(c.conv_size):
                accv = accv + xp[1 + tap : 1 + tap + B].astype(jnp.float32) * cwb[...][
                    :, tap
                ].astype(jnp.float32)[None, :]
            filtered = jax.nn.silu(accv).astype(jnp.bfloat16)
            for t in range(B):
                for tap in range(c.conv_size):
                    convsnaps[t * c.conv_size + tap] = xp[t + 1 + tap]
            cbuf[...] = xp[B : B + c.conv_size].T
            tpu.make_async_copy(cbuf, conv_r.at[ord_], c_io.at[0]).start()

            q = filtered[:, :kl].reshape(B, kh, c.k_dim).astype(jnp.float32)
            k = filtered[:, kl : 2 * kl].reshape(B, kh, c.k_dim).astype(jnp.float32)
            v = filtered[:, 2 * kl :].reshape(B, vh, c.v_dim).astype(jnp.float32)
            q = q * jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6)
            q = q * c.k_dim**-0.5
            k = k * jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
            expand = c.v_heads // c.k_heads
            q = jnp.broadcast_to(q[:, :, None, :], (B, kh, expand, c.k_dim)).reshape(B, vh, -1)
            k = jnp.broadcast_to(k[:, :, None, :], (B, kh, expand, c.k_dim)).reshape(B, vh, -1)
            beta = jax.nn.sigmoid(qkvzbuf[:, qkvw + vl : qkvw + vl + vh].astype(jnp.float32))
            beta = beta.astype(jnp.bfloat16).astype(jnp.float32)
            tpu.make_async_copy(alog_r.at[ord_], abuf, a_sem.at[0]).wait()
            tpu.make_async_copy(dtb_r.at[ord_], bbuf, a_sem.at[0]).wait()
            gate = -jnp.exp(abuf[0].astype(jnp.float32))[None, :] * jax.nn.softplus(
                qkvzbuf[:, qkvw + vl + vh : qkvw + vl + 2 * vh].astype(jnp.float32)
                + bbuf[0].astype(jnp.float32)[None, :]
            )
            tpu.make_async_copy(rec_r.at[ord_], sbuf, s_sem.at[0]).wait()
            out8rows = []
            skip_delta = "nodelta" in ablate or "nocompute" in ablate
            for t in range(B):
                if t >= 2:
                    # the snapshot buffer slot t%2 is reusable once its DMA landed
                    tpu.make_async_copy(
                        recsnaps, recsnap_r.at[ord_, 0], snap_sem.at[1 + t % 2]
                    ).wait()
                if skip_delta:
                    out8rows.append(jnp.zeros((vh, c.v_dim), jnp.float32))
                    recsnaps[t % 2] = sbuf[...]
                else:
                    sbuf[...] = sbuf[...] * jnp.exp(gate[t])[:, None, None]
                    prediction = jnp.sum(sbuf[...] * k[t][:, None, :], axis=-1)
                    delta = (v[t] - prediction) * beta[t][:, None]
                    sbuf[...] = sbuf[...] + delta[:, :, None] * k[t][:, None, :]
                    out8rows.append(jnp.sum(sbuf[...] * q[t][:, None, :], axis=-1))
                    recsnaps[t % 2] = sbuf[...]
                tpu.make_async_copy(
                    recsnaps.at[t % 2], recsnap_r.at[ord_, t], snap_sem.at[1 + t % 2]
                ).start()
            out8 = jnp.stack(out8rows, axis=0)
            tpu.make_async_copy(sbuf, rec_r.at[ord_], s_io.at[0]).start()
            tpu.make_async_copy(convsnaps, convsnap_r.at[ord_], snap_sem.at[0]).start()

            value = out8.astype(jnp.bfloat16).astype(jnp.float32)
            value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + c.eps)
            tpu.make_async_copy(onorm_r.at[ord_], onbuf, on_sem.at[0]).wait()
            value = (onbuf[...] * value.astype(jnp.bfloat16)).astype(jnp.float32)
            z = qkvzbuf[:, qkvw : qkvw + vl].reshape(B, vh, c.v_dim).astype(jnp.float32)
            attbuf[:, pl.ds(0, vl)] = (
                (value * jax.nn.silu(z)).astype(jnp.bfloat16).reshape(B, vl)
            )
            gemv(attbuf, proj, outw_r, ord_, base + off_lin[1], lin[1], out_f32=True)
            exchange_add(li * 2)

        def attend(ord2, query, key, value):
            kb8[...] = key  # [B, nkv, hd] bf16
            vb8[...] = value
            for g in range(nkv):
                tile0 = pos0 // 128
                tile1 = (pos0 + B - 1) // 128
                qb[...] = query[:, g * groups : (g + 1) * groups, :].reshape(B * groups, hd)

                count = (pos0 + B - 1) // 128 + 1
                am[...] = jnp.full((B * groups, 1), -jnp.inf, jnp.float32)
                al[...] = jnp.zeros((B * groups, 1), jnp.float32)
                aa[...] = jnp.zeros((B * groups, hd), jnp.float32)

                for t in range(2):

                    @pl.when(t < count)
                    def issue():
                        tpu.make_async_copy(
                            kc_r.at[ord2, g, pl.ds(t * 128, 128), :], kb.at[t], ks.at[t]
                        ).start()
                        tpu.make_async_copy(
                            vc_r.at[ord2, g, pl.ds(t * 128, 128), :], vb.at[t], vs.at[t]
                        ).start()

                def step(t, _):
                    tpu.make_async_copy(
                        kc_r.at[ord2, g, pl.ds(t * 128, 128), :], kb.at[t % 2], ks.at[t % 2]
                    ).wait()
                    tpu.make_async_copy(
                        vc_r.at[ord2, g, pl.ds(t * 128, 128), :], vb.at[t % 2], vs.at[t % 2]
                    ).wait()

                    @pl.when(t == tile0)
                    def patch0():
                        rowloc = pos0 % 128
                        for tt in range(B):
                            m = iota128[:, None] == (rowloc + tt)
                            kb[t % 2, ...] = jnp.where(
                                m, kb8[tt, g][None, :].astype(jnp.bfloat16), kb[t % 2, ...]
                            )
                            vb[t % 2, ...] = jnp.where(
                                m, vb8[tt, g][None, :].astype(jnp.bfloat16), vb[t % 2, ...]
                            )
                        tpu.make_async_copy(
                            kb.at[t % 2], kc_r.at[ord2, g, pl.ds(tile0 * 128, 128), :], io.at[0]
                        ).start()
                        tpu.make_async_copy(
                            vb.at[t % 2], vc_r.at[ord2, g, pl.ds(tile0 * 128, 128), :], io.at[1]
                        ).start()

                    @pl.when((tile1 > tile0) & (t == tile1))
                    def patch1():
                        for tt in range(B):
                            row = pos0 + tt - tile1 * 128

                            @pl.when(row >= 0)
                            def patch_row():
                                m = iota128[:, None] == row
                                kb[t % 2, ...] = jnp.where(
                                    m, kb8[tt, g][None, :].astype(jnp.bfloat16), kb[t % 2, ...]
                                )
                                vb[t % 2, ...] = jnp.where(
                                    m, vb8[tt, g][None, :].astype(jnp.bfloat16), vb[t % 2, ...]
                                )

                        tpu.make_async_copy(
                            kb.at[t % 2], kc_r.at[ord2, g, pl.ds(tile1 * 128, 128), :], io.at[2]
                        ).start()
                        tpu.make_async_copy(
                            vb.at[t % 2], vc_r.at[ord2, g, pl.ds(tile1 * 128, 128), :], io.at[3]
                        ).start()

                    slot = t * 128 + iota128
                    rowqi = jax.lax.iota(jnp.int32, B * groups) // groups  # (qi,gi)-major
                    keep_g = (slot[None, :] <= (pos0 + rowqi[:, None])) & (slot[None, :] >= key_lo)

                    if "noattn" in ablate or "nocompute" in ablate:
                        continue_tile = None
                    else:
                        pass
                    if "noattn" not in ablate and "nocompute" not in ablate:
                        scores = (
                            jnp.dot(
                                qb[...],
                                kb[t % 2, ...].reshape(128, hd).T,
                                preferred_element_type=jnp.float32,
                            )
                            * hd**-0.5
                        )
                        scores = jnp.where(keep_g, scores, -jnp.inf)
                        m = jnp.maximum(am[...], jnp.max(scores, axis=-1, keepdims=True))
                        alpha = jnp.exp(am[...] - m)
                        p = jnp.exp(scores - m).astype(jnp.bfloat16)
                        aa[...] = aa[...] * alpha + jnp.dot(
                            p.astype(jnp.float32),
                            vb[t % 2, ...].astype(jnp.float32),
                            preferred_element_type=jnp.float32,
                        )
                        al[...] = al[...] * alpha + jnp.sum(
                            p.astype(jnp.float32), axis=-1, keepdims=True
                        )
                        am[...] = m

                    @pl.when(t + 2 < count)
                    def refill():
                        tpu.make_async_copy(
                            kc_r.at[ord2, g, pl.ds((t + 2) * 128, 128), :],
                            kb.at[t % 2],
                            ks.at[t % 2],
                        ).start()
                        tpu.make_async_copy(
                            vc_r.at[ord2, g, pl.ds((t + 2) * 128, 128), :],
                            vb.at[t % 2],
                            vs.at[t % 2],
                        ).start()

                jax.lax.fori_loop(0, count, step, None)
                tpu.make_async_copy(
                    kb.at[pos0 // 128 % 2], kc_r.at[ord2, g, pl.ds(pos0 // 128 * 128, 128), :], io.at[0]
                ).wait()
                tpu.make_async_copy(
                    vb.at[pos0 // 128 % 2], vc_r.at[ord2, g, pl.ds(pos0 // 128 * 128, 128), :], io.at[1]
                ).wait()
                # tile1's writeback only ran when the block straddles two tiles
                @pl.when((pos0 + B - 1) // 128 > pos0 // 128)
                def wait_straddle():
                    tpu.make_async_copy(
                        kb.at[(pos0 + B - 1) // 128 % 2], kc_r.at[ord2, g, pl.ds((pos0 + B - 1) // 128 * 128, 128), :], io.at[2]
                    ).wait()
                    tpu.make_async_copy(
                        vb.at[(pos0 + B - 1) // 128 % 2], vc_r.at[ord2, g, pl.ds((pos0 + B - 1) // 128 * 128, 128), :], io.at[3]
                    ).wait()
                val = (aa[...] / al[...]).reshape(B, groups, hd)
                attended[:, pl.ds(g * groups * hd, groups * hd)] = val.astype(
                    jnp.bfloat16
                ).reshape(B, groups * hd)

        def full_attention(li):
            ord2 = (li + 1) // 4 - 1

            base = (li // 4) * t_block + 3 * t_lin
            tpu.make_async_copy(qn_r.at[ord2], qnbuf, qn_sem.at[0]).start()
            tpu.make_async_copy(kn_r.at[ord2], knbuf, kn_sem.at[0]).start()
            gemv(normed, qwkvbuf, qwkv_r, ord2, base + off_full[0], full[0])
            packed = qwkvbuf[:, : nh * 2 * hd].reshape(B, nh, 2 * hd)
            query, gate = packed[..., :hd], packed[..., hd:]
            tpu.make_async_copy(qn_r.at[ord2], qnbuf, qn_sem.at[0]).wait()
            tpu.make_async_copy(kn_r.at[ord2], knbuf, kn_sem.at[0]).wait()
            query = rms_zc(query, 1.0 + qnbuf[...].astype(jnp.float32))
            kv_off = nh * 2 * hd
            key = rms_zc(
                qwkvbuf[:, kv_off : kv_off + nkv * hd].reshape(B, nkv, hd),
                1.0 + knbuf[...].astype(jnp.float32),
            )
            value = qwkvbuf[:, kv_off + nkv * hd : kv_off + 2 * nkv * hd].reshape(B, nkv, hd)
            query = ropeb(query)
            key = ropeb(key)
            attend(ord2, query, key, value)
            gated = attended[...].reshape(B, nh, hd) * jax.nn.sigmoid(gate.astype(jnp.float32)).astype(
                jnp.bfloat16
            )
            attbuf[:, pl.ds(0, nh * hd)] = gated.reshape(B, nh * hd)
            gemv(attbuf, proj, ow_r, ord2, base + off_full[1], full[1], out_f32=True)
            exchange_add(li * 2)

        def mlp(li):
            base = (li // 4) * t_block + jnp.where(li % 4 == 3, 3 * t_lin, (li % 4) * t_lin)
            off3 = jnp.where(li % 4 == 3, off_full[2], off_lin[2])
            off4 = jnp.where(li % 4 == 3, off_full[3], off_lin[3])
            gemv(normed, gubuf, gu_r, li, base + off3, lin[2])
            gate = jax.nn.silu(gubuf[:, :inter].astype(jnp.float32)).astype(jnp.bfloat16)
            hbuf[:, pl.ds(0, inter)] = gate * gubuf[:, inter : 2 * inter]
            gemv(hbuf, proj, dw_r, li, base + off4, lin[3], out_f32=True)
            exchange_add(li * 2 + 1)

        def layer(li, _):
            norm(norm_in_r, li, 0)
            jax.lax.cond(li % 4 == 3, full_attention, linear_attention, li)
            norm(norm_post_r, li, 1)
            mlp(li)
            is_tap = jnp.zeros((), bool)
            for t_l in taps:
                is_tap = is_tap | (li == t_l)

            @pl.when(is_tap)
            def write_tap():
                tidx = jnp.zeros((), jnp.int32)
                for i, t_l in enumerate(taps):
                    tidx = jnp.where(li == t_l, i, tidx)
                taps_out[tidx] = residual[...]

        # RoPE tables for the B positions (same convention as the decode kernel).
        half = c.rotary_dim // 2
        inv = 1.0 / (c.rope_theta ** (jax.lax.iota(jnp.int32, half).astype(jnp.float32) * 2.0 / c.rotary_dim))
        ang = (pos0 + iotaB).astype(jnp.float32)[:, None] * inv[None, :]  # [B, half]
        cos_tb = jnp.concatenate((jnp.cos(ang), jnp.cos(ang)), axis=-1).astype(jnp.bfloat16)
        sin_tb = jnp.concatenate((jnp.sin(ang), jnp.sin(ang)), axis=-1).astype(jnp.bfloat16)

        # Embedding rows for the B tokens: aligned 8-row fetch + select per token.
        vshard = c.vocab // tp
        lo = rank * vshard
        rowbuf8_rows = []
        toks = tokenr[...]
        sel = jax.lax.iota(jnp.int32, 8)
        tpu.make_async_copy(norm_in_r.at[0, :, :], scale.at[0], norm_sems.at[0]).start()
        tpu.make_async_copy(norm_post_r.at[0, :, :], scale.at[1], norm_sems.at[1]).start()
        for t in range(B):
            tokl = jnp.clip(toks[t] - lo, 0, vshard - 1)
            tpu.make_async_copy(
                emb_r.at[pl.ds(tokl // 8 * 8, 8), :], rowbuf.at[t], e_sem.at[t]
            ).start()
        for t in range(B):
            tok = toks[t]
            tokl = jnp.clip(tok - lo, 0, vshard - 1)
            tpu.make_async_copy(
                emb_r.at[pl.ds(tokl // 8 * 8, 8), :], rowbuf.at[t], e_sem.at[t]
            ).wait()
            row = jnp.sum(
                jnp.where(sel[:, None] == tokl % 8, rowbuf[t], 0), axis=0
            ).astype(jnp.float32)
            row = jnp.where((tok >= lo) & (tok < lo + vshard), row, 0.0)
            rowbuf8_rows.append(row)
        for ti in range(bc):
            fetch(jnp.int32(ti))
        barrier = tpu.get_barrier_semaphore()
        for offset in range(1, tp):
            pl.semaphore_signal(
                barrier, 1, device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(barrier, tp - 1)
        hbuf[...] = jnp.zeros_like(hbuf[...])
        proj[...] = jnp.stack(rowbuf8_rows, axis=0)
        residual[...] = allreduce(1).astype(residual.dtype)
        jax.lax.fori_loop(0, c.layers, layer, None)
        # Drain trailing writebacks.
        tpu.make_async_copy(cbuf, conv_r.at[0], c_io.at[0]).wait()
        tpu.make_async_copy(sbuf, rec_r.at[0], s_io.at[0]).wait()
        tpu.make_async_copy(convsnaps, convsnap_r.at[0], snap_sem.at[0]).wait()
        tpu.make_async_copy(recsnaps.at[0], recsnap_r.at[0, 0], snap_sem.at[1]).wait()
        tpu.make_async_copy(recsnaps.at[0], recsnap_r.at[0, 0], snap_sem.at[2]).wait()
        tpu.make_async_copy(fnorm_r, fnbuf, fn_sem.at[0]).start()
        tpu.make_async_copy(fnorm_r, fnbuf, fn_sem.at[0]).wait()
        value = residual[...].astype(jnp.float32)
        value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + c.eps)
        normed[...] = (value * (1.0 + fnbuf[...].astype(jnp.float32))).astype(normed.dtype)
        gemv(normed, lm_accum, lmw_r, None, t_layers, lm, out_f32=True, acc=lm_accum)

        # Per-position greedy selection: per-rank argmax + (val, idx) pair exchange.
        vals = lm_accum[:, :vshard]  # [B, vshard]
        topv_r = jnp.max(vals, axis=1)  # [B]
        iotav = jax.lax.iota(jnp.int32, vshard)
        best = jnp.min(
            jnp.where(vals == topv_r[:, None], iotav[None, :], vshard), axis=1
        ).astype(jnp.int32)
        pairbuf[...] = jnp.stack((topv_r, best.astype(jnp.float32)), axis=1)  # [B, 2]
        pairs[0, rank, ...] = pairbuf[...]
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
        allvals = pairs[0, :, :, 0]  # [tp, B]
        topv = jnp.max(allvals, axis=0)  # [B]
        wiota = jax.lax.iota(jnp.int32, tp)
        winner = jnp.min(jnp.where(allvals == topv[None, :], wiota[:, None], tp), axis=0).astype(
            jnp.int32
        )
        local_idx = jnp.sum(
            jnp.where(wiota[:, None] == winner[None, :], pairs[0, :, :, 1], 0.0), axis=0
        )
        post_out[...] = (winner * vshard + local_idx.astype(jnp.int32)).astype(jnp.int32)

    shape = lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype)
    v = lambda shp, dtype=jnp.bfloat16: tpu.VMEM(shp, dtype)
    dm = lambda n: tpu.SemaphoreType.DMA((n,))
    states_in = (states["conv"], states["rec"], states["kcache"], states["vcache"])
    weights_in = (
        w["qkvz"],
        w["outw"],
        w["qwkv"],
        w["ow"],
        w["gu"],
        w["dw"],
        w["lmw"],
        w["norm_in"][:, None, :],
        w["norm_post"][:, None, :],
        w["convw"],
        w["alog"][:, None, :],
        w["dtb"][:, None, :],
        w["onorm"][:, None, :],
        w["qn"][:, None, :],
        w["kn"][:, None, :],
        w["fnorm"][None, :],
        w["emb"],
    )
    # args: (tokens, weights..., states..., conv_snaps, rec_snaps)
    args = (tokens, *weights_in, *states_in, snaps["conv"], snaps["rec"])
    grps = B * groups
    scratch = (
        v((bc, MAX_BANK_K, MAX_BANK_N)),
        dm(bc),
        v((2, 1, h)),
        dm(2),
        v((B, h)),
        v((B, h)),
        v((B, h), jnp.float32),
        v((B, gu_width(c, tp)), jnp.float32),
        v((B, lm_width), jnp.float32),
        v((B, qkvw + zba_width(c, tp))),
        v((B, nh * 2 * hd + 2 * nkv * hd)),
        v((B, att_w)),
        v((B, gu_width(c, tp))),
        v((B, dw_k(c, tp))),
        v((2, tp, B, h // tp), jnp.float32),
        v((B, h // tp)),
        v((2, tp, B, h // tp)),
        dm(tp - 1),
        dm(tp - 1),
        dm(tp - 1),
        dm(tp - 1),
        dm(tp - 1),
        dm(tp - 1),
        v((vh, c.v_dim, c.k_dim), jnp.float32),
        dm(1),
        dm(1),
        v((qkvw, c.conv_size)),
        v((qkvw, c.conv_size)),
        v((1, vh)),
        v((1, vh)),
        v((1, c.v_dim)),
        dm(1),
        dm(1),
        dm(1),
        dm(1),
        dm(1),
        v((B, nh * hd)),
        v((grps, hd)),
        v((2, 128, hd)),
        v((2, 128, hd)),
        v((grps, 1), jnp.float32),
        v((grps, 1), jnp.float32),
        v((grps, hd), jnp.float32),
        dm(2),
        dm(2),
        dm(4),
        v((1, hd)),
        v((1, hd)),
        v((1, h)),
        dm(1),
        dm(1),
        dm(1),
        v((B, 8, h)),
        dm(B),
        v((B, 2), jnp.float32),
        v((2, tp, B, 2), jnp.float32),
        v((2, vh, c.v_dim, c.k_dim), jnp.float32),
        v((B * c.conv_size, qkvw)),
        dm(3),
        v((B, nkv, hd)),
        v((B, nkv, hd)),
    )
    state_offset = 1 + 1 + len(weights_in)
    # inputs: 0 pospair | 1 tokens | 2..18 weights | 19..22 states | 23 convsnap | 24 recsnap
    # outputs: 0 post | 1 taps | 2 conv_snap | 3 rec_snap | 4 conv | 5 rec | 6 kc | 7 vc
    result = pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((B,), jnp.int32),
            jax.ShapeDtypeStruct((len(taps), B, h), jnp.bfloat16),
            shape(snaps["conv"]),
            shape(snaps["rec"]),
            *(shape(s) for s in states_in),
        ),
        input_output_aliases={
            state_offset + 0: 4,
            state_offset + 1: 5,
            state_offset + 2: 6,
            state_offset + 3: 7,
            state_offset + 4: 2,
            state_offset + 5: 3,
        },
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[pl.BlockSpec()] + [pl.BlockSpec(memory_space=tpu.HBM)] * (len(args) - 1),
            out_specs=[pl.BlockSpec(), pl.BlockSpec()] + [pl.BlockSpec(memory_space=tpu.HBM)] * 6,
            scratch_shapes=scratch,
        ),
        compiler_params=tpu.CompilerParams(
            collective_id=45,
            vmem_limit_bytes=64 * 1024**2,
            disable_bounds_checks=True,
            shape_invariant_numerics=True,
        ),
        name="qwen_transformer_stack_block",
    )(pospair, *args)
    return (
        result[0],  # post [B] int32
        result[1],  # taps [ntap, B, dim] bf16
        result[2],  # conv snapshots [nl, B, ...]
        result[3],  # rec snapshots [nl, B, ...]
        dict(zip(("conv", "rec", "kcache", "vcache"), result[4:8])),
    )


def make_decode(mesh, c, context, donate=True, ablate=(), tp=2):
    ablate = frozenset(ablate)
    """jit(shard_map(...)) decode: (weights, states, token, position) -> (next, logits, states)."""
    vocab_half = c.vocab // tp

    def local(global_weights, global_states, token, position):
        weights = jax.tree.map(lambda a: a[0], global_weights)
        states = jax.tree.map(lambda a: a[0], global_states)
        # `position` is the (2,) int32 pair [content_position, key_lo].
        logits, nxt, new_states = transformer_stack(
            c, token, weights, states, position, context, ablate, tp
        )
        new_states = jax.tree.map(lambda a: a[None], new_states)
        return nxt, logits, new_states

    def decode(weights, states, token, position):
        ws = jax.tree.map(lambda _: P("tp"), weights)
        ss = jax.tree.map(lambda _: P("tp"), states)
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(ws, ss, P(), P()),
            out_specs=(P(), P("tp"), ss),
            check_vma=False,
        )(weights, states, token, position)

    return jax.jit(decode, donate_argnums=(1,) if donate else ())


def make_verify_block(mesh, c, context, tp, tap_layers, block=8, ablate=(), bank_count=None):
    """jit(shard_map) wrapper for transformer_stack_block.

    (weights, states, block_tokens[B], snaps, pospair[2]) ->
    (post[B] replicated, taps [ntap, B, dim] replicated, snaps, states).
    states/snaps are donated through aliasing.
    """

    def local(gw, gs, gtok, gsnaps, gpos):
        w = jax.tree.map(lambda a: a[0], gw)
        s = jax.tree.map(lambda a: a[0], gs)
        sn = jax.tree.map(lambda a: a[0], gsnaps)
        post, taps, csnap, rsnap, new_states = transformer_stack_block(
            c, gtok, w, s, gpos, sn, context, tp, tap_layers, block=block, ablate=ablate,
            bank_count=bank_count,
        )
        new_states = jax.tree.map(lambda a: a[None], new_states)
        return post, taps, {"conv": csnap[None], "rec": rsnap[None]}, new_states

    def verify(w, states, tokens, snaps, pospair):
        ws = jax.tree.map(lambda _: P("tp"), w)
        ss = jax.tree.map(lambda _: P("tp"), states)
        ns = jax.tree.map(lambda _: P("tp"), snaps)
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(ws, ss, P(), ns, P()),
            out_specs=(P(), P(), ns, ss),
            check_vma=False,
        )(w, states, tokens, snaps, pospair)

    return jax.jit(verify)
