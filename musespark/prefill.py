"""Pure-XLA TP prefill of Muse Spark 1.2 over the decode megakernel's per-rank weights.

`make_prefill(mesh, cfg, context, tp)` returns a jitted

    prefill(weights, caches, tokens [Tp] int32, length int32, row int32)
        -> (logits_last [V] f32, caches)

that runs the whole prompt through the model inside one `jax.shard_map` over the SAME
`{name: [tp, ...]}` arrays the kernel consumes (`musespark.layout.rank_shapes`) and writes the
keys/values of the prompt into `caches["k_cache" | "v_cache"][:, l, row, 0:Tp, :]` in the
kernel's lane layout (design.md section 3: lanes `[h*D, (h+1)*D)` hold local kv head `h`, K is
post-QK-norm and post-RoPE, V raw). Token `i` of the prompt sits at absolute position `i`, so
the decode step continues at `pos = length` (slot `length` is written by decode itself).

The prompt is RIGHT-padded to a length bucket (`prefill_length_bucket`, multiples of 64) with
pad tokens; `length` is the number of real tokens. Pad rows never influence real rows (their
keys are masked, their K/V cache slots are written as zeros, their residual stream is zeroed
after every layer) and are only there so one executable serves every prompt of the bucket.

Numerics mirror the decode kernel (design.md section 4), i.e. the spec's rounding points:
f32 residual stream, `r16` (bf16 round trip) at every norm output and branch output, bf16 x bf16
GEMMs with f32 accumulation, the f32 router as `dot(x, hi) + dot(x, lo)`, quantized experts
(int4 g128 or NVFP4 families, `dequantized_experts`) dequantized on the fly one layer at a
time inside the layer loop and rounded to bf16, `gate_up` outputs kept in f32 until the SwiGLU rounding, the per-expert
`post_expert_norm` applied to the cross-rank sum of the partial `down` outputs. Attention scores
are bf16 q/k products accumulated in f32 and the probabilities stay f32 (`HIGHEST`, the kernel's
hi/lo MXU mode). Experts are evaluated with `jax.lax.ragged_dot` after sorting the `Tp * top_k`
routes by expert.

Communication per layer: `psum` of the o-proj partial sums, `all_gather` of the `pre` and
`post` output shards, `psum_scatter` of the expert partial sums (each rank normalises its own
`Hm / tp` slice, the row sums of squares are `psum`'d) and an `all_gather` of the mixed expert
output. Every rank ends up with a bit-identical residual stream (XLA all-reduces return the same
value on every participant), which the routing relies on.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

from musespark import layout, quant, r16, rms, rope_rotate_half, route, softcap_logits
from musespark.config import Config

BF16 = jnp.bfloat16
F32 = jnp.float32
CHUNK = 64  # prompt lengths are padded to a multiple of this
QUERY_BLOCK = 512  # query rows per attention block (scores are [heads, QUERY_BLOCK, Tp] f32)
NEG = -1e30  # masked score (finite: a fully masked pad row stays NaN-free)


def prefill_length_bucket(length, chunk=CHUNK):
    """Padded prompt length compiled for a prompt of `length` tokens (a positive multiple of 64)."""
    if length < 1:
        raise ValueError("prefill needs at least one token")
    return -(-length // chunk) * chunk


def pad_prompt(cfg: Config, ids, chunk=CHUNK):
    """`(tokens [bucket] int32, length)` for a prompt (ids), right-padded with pad tokens."""
    ids = [int(t) for t in ids]
    width = prefill_length_bucket(len(ids), chunk)
    return jnp.asarray(ids + [cfg.pad] * (width - len(ids)), jnp.int32), len(ids)


def _dot(x, w):
    """bf16 x bf16 -> f32 accumulate (the MXU path of the kernel)."""
    return jnp.dot(x.astype(BF16), w.astype(BF16), preferred_element_type=F32)


def _attend_block(cfg, q, qpos, k, v, kpos, real, window):
    """Causal / sliding-window GQA for one block of queries.

    q `[BQ, kvh, group, D]` and k/v `[S, kvh, D]` hold bf16 values (f32 arrays); `window` is the
    traced key window (`sliding_window`, or a huge value on full-attention layers); returns the
    f32 attention output `[BQ, kvh, group, D]`.
    """
    scores = (
        jnp.einsum("tkgd,skd->kgts", q.astype(BF16), k.astype(BF16), preferred_element_type=F32)
        * cfg.softmax_scale
    )
    qp = qpos[:, None]
    ok = (kpos[None, :] <= qp) & (kpos[None, :] > qp - window) & real[None, :]  # [BQ, S]
    scores = jnp.where(ok[None, None], scores, NEG)
    p = jax.nn.softmax(scores, axis=-1)  # f32
    return jnp.einsum("kgts,skd->tkgd", p, v, precision=lax.Precision.HIGHEST)  # f32 p


def _attention(cfg, q, k, v, positions, real, window, tp):
    """Blocked attention over the prompt: q `[Tp, qh, D]`, k/v `[Tp, kvh, D]` -> o `[Tp, qh, D]`."""
    T, qh, D = q.shape
    kvh = cfg.kv_heads // tp
    group = qh // kvh
    qg = q.reshape(T, kvh, group, D)
    attend = partial(_attend_block, cfg, k=k, v=v, kpos=positions, real=real, window=window)
    bq = min(QUERY_BLOCK, T)
    if bq == T:
        o = attend(qg, positions)
    else:
        blocks = T // bq
        o = lax.map(
            lambda args: attend(*args),
            (qg.reshape(blocks, bq, kvh, group, D), positions.reshape(blocks, bq)),
        ).reshape(T, kvh, group, D)
    return o.reshape(T, qh, D)


def dequantized_experts(w, l, isl):
    """This rank's experts of layer `l` as bf16 `(gate_up [E, Hm, 2*Is], down [E, Is, Hm])`
    from either expert format: int4 g128 (`quant.dequantize_int4`, container v1) or NVFP4
    (`quant.dequant_fp4_jnp`: `e2m1 * e4m3` exact, times the per-expert global scales of
    `expert_gs` -- row 0 for the gate columns, row 1 for the up columns, row 2 for down)."""
    if "gate_up_fp4" in w:
        gs = w["expert_gs"][l]  # [E, 8, 128]
        col = _iota_cols(2 * isl) < isl  # [1, 2*Is]
        gu_gs = jnp.where(col, gs[:, 0:1, 0:1], gs[:, 1:2, 0:1])  # [E, 1, 2*Is]
        gate_up = quant.dequant_fp4_jnp(w["gate_up_fp4"][l], w["gate_up_bs"][l], gu_gs)
        down = quant.dequant_fp4_jnp(w["down_fp4"][l], w["down_bs"][l], gs[:, 2:3, 0:1])
        return gate_up.astype(BF16), down.astype(BF16)
    gate_up = quant.dequantize_int4(w["gate_up_q"][l], w["gate_up_s"][l]).astype(BF16)
    down = quant.dequantize_int4(w["down_q"][l], w["down_s"][l]).astype(BF16)
    return gate_up, down


def _iota_cols(n):
    return jnp.arange(n, dtype=jnp.int32)[None, :]


def _experts(cfg, w, l, h1, idx, weights, rank, tp):
    """Routed experts of one layer on `h1 [Tp, Hm]` (bf16 values) -> `m [Tp, Hm/tp]` f32.

    `idx [Tp, K]` / `weights [Tp, K]` are the routes; the `Tp * K` (token, slot) rows are sorted
    by expert so `lax.ragged_dot` sees contiguous groups. The quantized experts of this layer
    are dequantized here (`dequantized_experts`: bf16 `[E, Hm, 2*Is]` and `[E, Is, Hm]`, int4
    or NVFP4 families); the `down` outputs are partial
    sums over this rank's `Is` slice and are reduce-scattered over `Hm`; the post-expert norm
    (weight before the norm, no weight after) is applied per (token, slot) on the full sum
    and the top-k mixture is formed on this rank's `Hm / tp` slice.
    """
    T, K = idx.shape
    E, Hm = cfg.experts, cfg.moe_hidden
    isl, hm_r = cfg.expert_hidden // tp, Hm // tp
    eid = idx.reshape(T * K)
    order = jnp.argsort(eid, stable=True)
    group_sizes = jnp.zeros((E,), jnp.int32).at[eid].add(1)
    lhs = jnp.take(h1.astype(BF16), order // K, axis=0)  # [R, Hm]
    gate_up, down = dequantized_experts(w, l, isl)
    gu = lax.ragged_dot(lhs, gate_up, group_sizes, preferred_element_type=F32)  # [R, 2*Is]
    act = r16(jax.nn.silu(gu[:, :isl]) * gu[:, isl:]).astype(BF16)
    y = lax.ragged_dot(act, down, group_sizes, preferred_element_type=F32)  # [R, Hm] partial
    y = lax.psum_scatter(y, "tp", scatter_dimension=1, tiled=True)  # [R, Hm/tp] full sum
    w_post = lax.dynamic_slice_in_dim(w["post_expert_norm"][l], rank * hm_r, hm_r, axis=1)
    t = r16(y * w_post.astype(F32))
    sumsq = lax.psum(jnp.sum(t * t, axis=-1, keepdims=True), "tp")  # [R, 1] over the full row
    yn = t * lax.rsqrt(sumsq / Hm + cfg.post_eps)
    yn = jnp.zeros_like(yn).at[order].set(yn).reshape(T, K, hm_r)  # back to (token, slot)
    m = jnp.zeros((T, hm_r), F32)
    for slot in range(K):  # slot order, like the kernel
        m = m + weights[:, slot : slot + 1] * yn[:, slot]
    return m


def make_prefill(mesh, cfg: Config, context, tp=8, taps=False):
    """Jitted `prefill(weights, caches, tokens [Tp], length, row) -> (logits_last [V], caches)`.

    `weights` is the `{name: [tp, ...]}` tree of `musespark.load.load_presharded` /
    `musespark.shard_canonical`, `caches` the `{"k_cache", "v_cache": [tp, L, B, context, lanes]}`
    tree of `musespark.load.zero_caches` (donated and returned updated). `tokens` must be a
    `prefill_length_bucket` wide, `length <= Tp <= context` real tokens are the prompt, and
    the K/V of positions `0..Tp-1` are written to cache row `row`. `logits_last` are the
    softcapped, full-vocabulary logits of position `length - 1` (replicated on every device);
    ids `>= cfg.vocab_used` are NOT masked (see `musespark.sampling`). One executable per
    bucket: `jax.jit` retraces when `tokens.shape` changes. With `taps` the program also returns
    the f32 residual stream after the embedding and after every layer, `[L + 1, Tp, H]`
    (replicated), for activation comparisons against the reference / the kernel.
    """
    layout.check_tp(cfg, tp)
    if mesh.size != tp:
        raise ValueError(f"mesh has {mesh.size} devices, tp={tp}")
    vp = layout.vocab_pad(cfg, tp)
    lanes = layout.cache_lanes(cfg, tp)
    qh, kvh, D = cfg.heads // tp, cfg.kv_heads // tp, cfg.head_dim
    kvw = kvh * D
    L, H, V = cfg.layers, cfg.hidden, cfg.vocab

    def local(w, caches, tokens, length, row):
        w = {name: value[0] for name, value in w.items()}
        kc, vc = caches["k_cache"][0], caches["v_cache"][0]  # [L, B, context, lanes]
        rank = lax.axis_index("tp")
        T = tokens.shape[0]
        positions = jnp.arange(T, dtype=jnp.int32)
        real = positions < length
        real_col = real[:, None]

        # --- embedding: vocab-sharded row lookup, one rank holds the row -> psum -------------
        lo = rank * vp
        rows = jnp.take(w["embed"], jnp.clip(tokens - lo, 0, vp - 1), axis=0).astype(F32)
        in_range = (tokens >= lo) & (tokens < lo + vp)
        e0 = lax.psum(jnp.where(in_range[:, None], rows, 0.0), "tp")
        s = jnp.where(real_col, r16(rms(e0, None, cfg.rms_eps)), 0.0)  # f32 stream [T, H]
        x = r16(rms(s, w["attn_norm"][0].astype(F32), cfg.rms_eps))

        def layer(l, carry):
            s, x, kc, vc, hidden = carry
            is_full = (l % cfg.full_attention_every) == cfg.full_attention_offset
            window = jnp.where(is_full, jnp.int32(2**30), jnp.int32(cfg.sliding_window))
            xb = x.astype(BF16)
            # --- attention --------------------------------------------------------------
            q = r16(_dot(xb, w["q"][l])).reshape(T, qh, D)
            kv = r16(_dot(xb, w["kv"][l]))
            k = kv[:, :kvw].reshape(T, kvh, D)
            v = kv[:, kvw:].reshape(T, kvh, D)
            g = r16(_dot(xb, w["gate"][l])).reshape(T, qh, D)
            q = r16(rms(q, None, cfg.rms_eps))
            k = r16(rms(k, None, cfg.rms_eps))
            q = jnp.where(is_full, q, rope_rotate_half(q, positions, cfg.rope_theta))
            k = jnp.where(is_full, k, rope_rotate_half(k, positions, cfg.rope_theta))
            k = jnp.where(real_col[:, None], k, 0.0)
            v = jnp.where(real_col[:, None], v, 0.0)
            pad = ((0, 0), (0, lanes - kvw))
            k_slab = jnp.pad(k.reshape(T, kvw), pad).astype(BF16)[None, None]
            v_slab = jnp.pad(v.reshape(T, kvw), pad).astype(BF16)[None, None]
            kc = lax.dynamic_update_slice(kc, k_slab, (l, row, 0, 0))
            vc = lax.dynamic_update_slice(vc, v_slab, (l, row, 0, 0))
            o = _attention(cfg, q, k, v, positions, real, window, tp)
            o = r16(rms(o, None, cfg.rms_eps) * jax.nn.sigmoid(g)).reshape(T, qh * D)
            attn_out = r16(lax.psum(_dot(o, w["o"][l]), "tp"))  # [T, H]
            # --- post-attention boundary --------------------------------------------------
            nb = r16(rms(attn_out, None, cfg.post_eps))
            s = w["attn_gate_alpha"][l] * s + w["attn_gate_beta"][l] * nb
            x_ffn = r16(rms(s, w["ffn_norm"][l].astype(F32), cfg.rms_eps))
            xf = x_ffn.astype(BF16)
            # --- MoE --------------------------------------------------------------------------
            logits = _dot(xf, w["router_hi"][l]) + _dot(xf, w["router_lo"][l])  # f32 [T, E]
            idx, weights = route(
                logits * cfg.output_multiplier, w["router_bias"][l], cfg.top_k, cfg.route_eps
            )
            h0 = r16(_dot(xf, w["pre"][l])).astype(BF16)  # [T, Hm/tp]
            h0 = lax.all_gather(h0, "tp", axis=1, tiled=True).astype(F32)  # [T, Hm]
            h1 = r16(rms(h0, w["pre_expert_norm"][l].astype(F32), cfg.rms_eps))
            m = _experts(cfg, w, l, h1, idx, weights, rank, tp)  # [T, Hm/tp]
            m = lax.all_gather(r16(m).astype(BF16), "tp", axis=1, tiled=True)  # [T, Hm]
            out = r16(_dot(m, w["post"][l])).astype(BF16)  # [T, H/tp]
            ffn_out = lax.all_gather(out, "tp", axis=1, tiled=True).astype(F32)  # [T, H]
            # --- post-FFN boundary ------------------------------------------------------------
            t = r16(ffn_out * w["post_ffn_norm"][l].astype(F32))
            nb = r16(t * lax.rsqrt(jnp.mean(t * t, axis=-1, keepdims=True) + cfg.post_eps))
            s = w["ffn_gate_alpha"][l] * s + w["ffn_gate_beta"][l] * nb
            s = jnp.where(real_col, s, 0.0)
            x = r16(rms(s, w["attn_norm"][jnp.minimum(l + 1, L - 1)].astype(F32), cfg.rms_eps))
            if taps:
                hidden = lax.dynamic_update_index_in_dim(hidden, s, l + 1, axis=0)
            return s, x, kc, vc, hidden

        hidden = jnp.zeros((L + 1, T, H), F32).at[0].set(s) if taps else None
        s, _, kc, vc, hidden = lax.fori_loop(0, L, layer, (s, x, kc, vc, hidden))
        h_last = lax.dynamic_slice_in_dim(r16(s), length - 1, 1, axis=0)  # [1, H]
        h_last = r16(rms(h_last, w["final_norm"].astype(F32), cfg.rms_eps))
        shard = _dot(h_last, w["lm_head"])  # [1, Vp]
        full = lax.all_gather(shard, "tp", axis=1, tiled=True)[0, :V]
        logits = softcap_logits(cfg, full)
        out = (logits, {"k_cache": kc[None], "v_cache": vc[None]})
        return out + (hidden,) if taps else out

    def prefill(weights, caches, tokens, length, row):
        T = tokens.shape[0]
        if T % CHUNK or T > context:
            raise ValueError(f"tokens must be a multiple of {CHUNK} and at most context={context}")
        w_specs = {name: P("tp") for name in weights}
        c_specs = {name: P("tp") for name in caches}
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(w_specs, c_specs, P(), P(), P()),
            out_specs=(P(), c_specs) + ((P(),) if taps else ()),
            check_vma=False,
        )(weights, caches, tokens, jnp.asarray(length, jnp.int32), jnp.asarray(row, jnp.int32))

    return jax.jit(prefill, donate_argnums=(1,))
