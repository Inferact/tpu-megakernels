"""Muse Spark 1.2 TP8 decode megakernel: one grid-less `pallas_call` per decode step.

`make_decode(mesh, cfg, context, batch, ...)` returns a jitted

    decode(weights, caches, tokens [B] int32, pos [B] int32)
        -> (next_tokens [B] int32, logits [B, V] f32 or None, caches)

over the per-rank weight tree of `musespark.load.load_presharded` / `musespark.shard_canonical`
(`{name: [tp, ...]}` sharded `P("tp")`) and the `{"k_cache", "v_cache": [tp, L, B, context,
lanes]}` caches of `musespark.load.zero_caches` (donated, updated in place through
`input_output_aliases`). The program is `jax.jit(jax.shard_map(local))`; inside `local` the
leading rank axis is stripped and (design.md section 4):

1. XLA glue: vocab-sharded embedding row lookup + `psum`, parameter-free embed norm
   `s0 = r16(rms(row))`, the RoPE table of the step (`attention.rope_table`).
2. The kernel (below): every layer, the final norm and the lm_head, writing the raw logits
   shard `[8, Vp]` f32 (rows `>= B` are padding) and the caches.
3. Greedy tokens in-kernel (`_greedy_tokens`: per-rank masked `(max, argmax)`, one all-gather
   of the 8 pairs, lowest id among the maxima -- the semantics of `sampling.greedy_local`)
   written as `[8, 128]` int32 (lane 0); with `return_logits` the XLA glue also gathers
   `sampling.gather_logits_local` (softcapped full logits).

Kernel body (per rank; `l` is a `lax.fori_loop` index, the layer kind is arithmetic):

    barrier; prime the dense ring; prefetch the layer-0 vectors; x_attn = r16(rms(s0, attn_norm[0]))
    for l in 0..L-1:
        q, kv, g = r16(gemv(x_attn, ...))                       ring order q, kv, gate
        o = attention_layer(...)                                 cache read/write, online softmax
        attn_out = r16(all_reduce(gemv(o, o[l])))                ring: o; collective 4l
        s = alpha * s + beta * r16(rms(attn_out, 1, post_eps)); x_ffn = r16(rms(s, ffn_norm[l]))
        h0 = gemv(x_ffn, pre[l]); logits = gemv(x_ffn, router_hi[l]) + gemv(x_ffn, router_lo[l])
        route; start_expert_stream (first wave of expert DMAs)
        h1 = r16(rms(all_gather(r16(h0)), pre_expert_norm[l]))  collective 4l+1
        expert_stream -> y_out; Y = all_reduce(y_out)            collective 4l+2
        m = finalize(Y, w, post_expert_norm[l])
        ffn_out = all_gather(r16(gemv(m, post[l])))              ring: post; collective 4l+3
        t = r16(ffn_out * post_ffn_norm[l]); s = alpha * s + beta * r16(t * rsqrt(mean(t^2) + eps))
        x_attn = r16(rms(s, attn_norm[l + 1]))
    hN = r16(rms(r16(s), final_norm)); logits = gemv(hN, lm_head)

Per-layer vectors (`layout.VECTOR_FAMILIES`) are double-buffered by `l % 2` and prefetched one
layer ahead; the dense ring keeps `layout.BANKS` tiles in flight across layer boundaries and
into the lm_head; expert DMAs start right after routing. All vector maths is replicated on
every rank and bit-identical (deterministic collectives), so every rank routes identically.

Payload rows of the collectives are padded to 8 (`stream.mxu_rows`); payloads narrower than
`tp * 128` lanes (MINI's `[K*B, 512]` expert outputs) are folded into `tp * 128`-wide rows
(`_all_reduce`). The MINI config runs the identical code.

`options` (frozenset of strings):
    "interpret"        run under `pltpu.InterpretParams` (CPU tests with 8 host devices)
    "aux_hidden"       also return the f32 residual stream after the embedding and after every
                       layer as `[L + 1, B, H]` (replicated): `decode(...) -> (tokens, logits,
                       caches, aux)`
    "moe_slots=N"      expert DMA slots (default: `default_moe_slots`, the largest count <= 4
                       that keeps the explicit VMEM total <= 58 MiB; one packed slot is 3.4 MiB
                       at real widths; measured: 4 slots beat 3, 5, 6 and 8 at B=1 and B=8)
    "hier=on|off"      chip-hierarchical expert-output all-reduce (pair + 4-chip phases,
                       collectives.py); default on at B >= 8 where the [64, 4096] payload
                       gains 1 us (bf16 wire), off below (latency-bound payloads lose)
    "banks=N"          dense ring depth in 2 MiB loads (default: `default_geometry`, 12 unless a
                       shallower ring (>= 8; the depth is not measurable above 8) is needed to
                       fit 8 expert slots)
    "wire=bf16|f32"    payload dtype of the two per-layer all-reduces (default bf16: every
                       rank's partial is rounded to bf16 before the fixed-order f32 summation
                       and the reduced blocks travel as bf16 -- the result is r16'd by the
                       caller anyway; f32 keeps the exact f32 partials on the wire at ~2x the
                       collective time)
    "flush=first|last" when the held-back refills are issued: after the first wave's body (the
                       second wave's slabs are queued) or after the last wave's (default)
    "kv_late"          issue the attention KV tile DMAs inside the attention phase (the
                       original order) instead of at the start of the layer
    "defer=none|next|post"
                       which ring refills issued during the pre/router gemvs are held back
                       until the expert slabs of the first two waves are in the (FIFO) DMA
                       queue: none (original order; the default at B > 4, where the many
                       expert waves need the ring tiles for DMA queue depth), next (the next
                       layer's tiles; the default at B <= 4), post (also this layer's post tiles)
    "skip=a,b,..."     PROFILING ONLY (wrong results): leave out phases to attribute time.
                       `collectives` (all-reduce -> identity, all-gather -> local tile),
                       `attention` (o := q, no cache traffic), `experts` (no expert DMAs/dots),
                       `route` (static experts 0..K-1), `lm_head` (ring ends after the last
                       layer, logits unwritten), `dense_dots` (ring DMAs only, no MXU work),
                       `expert_dots` (expert DMAs only)

Run TPU programs with `XLA_FLAGS=--xla_allow_excess_precision=false` so the glue's `r16`
(`lax.reduce_precision`) and the reference agree bit for bit.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

import musespark
from musespark import attention, collectives, fp4, layout, moe, sampling, stream
from musespark.config import Config

BF16 = jnp.bfloat16
F32 = jnp.float32
I32 = jnp.int32
MIB = 1 << 20
COLLECTIVE_ID = 37
COLLECTIVES_PER_LAYER = 4  # o-proj all-reduce, pre all-gather, expert all-reduce, post all-gather
TOKEN_LANES = 128  # tokens output tile [LOGIT_ROWS, TOKEN_LANES] int32, token of row b in lane 0
VMEM_LIMIT = 64 << 20
LOGIT_ROWS = moe.MR  # 8: rows of the lm_head accumulator (the MXU row block)

WEIGHT_NAMES = (
    layout.STREAMED_FAMILIES + layout.VECTOR_FAMILIES + layout.EXPERT_FAMILIES + ("lm_head",)
)


def weight_names(expert_format="int4"):
    """Kernel operand order: streamed, vector, expert (`layout.expert_families`), lm_head."""
    return (
        layout.STREAMED_FAMILIES
        + layout.VECTOR_FAMILIES
        + layout.expert_families(expert_format)
        + ("lm_head",)
    )


def _expert_module(expert_format):
    """`moe` (int4 g128) or `fp4` (NVFP4) -- drop-in twins (`ExpertWeights`, `Scratch`,
    `scratch_shapes`, `start_expert_stream`, `expert_stream`)."""
    if expert_format == "int4":
        return SimpleNamespace(
            scratch_shapes=moe.scratch_shapes,
            bind=moe.MoeScratch.bind,
            weights=lambda w: moe.ExpertWeights(*(w[n] for n in layout.EXPERT_FAMILIES)),
            start=moe.start_expert_stream,
            stream=moe.expert_stream,
        )
    if expert_format == "nvfp4":
        return SimpleNamespace(
            scratch_shapes=lambda cfg, batch, tp, slots, packed=True: fp4.scratch_shapes(
                cfg, batch, tp, slots
            ),
            bind=fp4.Fp4Scratch.bind,
            weights=fp4.Fp4ExpertWeights.from_dict,
            start=fp4.start_expert_stream,
            stream=fp4.expert_stream,
        )
    raise ValueError(f"unknown expert_format {expert_format!r}")


# --------------------------------------------------------------------------------------
# options / budget
# --------------------------------------------------------------------------------------
def parse_options(options):
    """`frozenset` of option strings -> namespace (see the module docstring)."""
    opts = SimpleNamespace(
        interpret=False, aux_hidden=False, moe_slots=None, skip=frozenset(), wire=BF16,
        kv_late=False, defer=None, banks=None, flush="last", hier=None,
    )
    for opt in options:
        if opt == "interpret":
            opts.interpret = True
        elif opt == "aux_hidden":
            opts.aux_hidden = True
        elif opt.startswith("moe_slots="):
            opts.moe_slots = int(opt.split("=", 1)[1])
        elif opt == "kv_late":
            opts.kv_late = True
        elif opt.startswith("flush="):
            opts.flush = opt.split("=", 1)[1]
            if opts.flush not in ("first", "last"):
                raise ValueError(f"unknown flush mode {opts.flush!r}")
        elif opt.startswith("defer="):
            opts.defer = opt.split("=", 1)[1]
            if opts.defer not in ("none", "next", "post"):
                raise ValueError(f"unknown defer mode {opts.defer!r}")
        elif opt.startswith("hier="):
            opts.hier = {"on": True, "off": False}[opt.split("=", 1)[1]]
        elif opt.startswith("banks="):
            opts.banks = int(opt.split("=", 1)[1])
        elif opt.startswith("wire="):
            opts.wire = {"bf16": BF16, "f32": F32}[opt.split("=", 1)[1]]
        elif opt.startswith("skip="):
            skip = frozenset(x for x in opt.split("=", 1)[1].split(",") if x)
            if skip - SKIPPABLE:
                raise ValueError(f"unknown skip phases {sorted(skip - SKIPPABLE)}")
            opts.skip = opts.skip | skip
        else:
            raise ValueError(f"unknown decode option {opt!r}")
    return opts


SKIPPABLE = frozenset(
    {"collectives", "attention", "experts", "route", "lm_head", "dense_dots", "expert_dots"}
)


def _padded_bytes(shape, dtype):
    """VMEM bytes of a scratch array as Mosaic allocates it (measured on TPU7x, jax 0.11.1):
    the minor two dims are padded to the native tile (`[8, 128]` f32, `[16, 128]` bf16,
    `[64, 128]` int4), single-row arrays (`[.., 1, W]`) use a compact `[1, 128]` tiling, and
    int4 occupies ONE BYTE per element (`s4[4,4096,1024]` is reported as 16 MiB)."""
    dtype = jnp.dtype(dtype)
    if dtype == jnp.dtype(jnp.int4):
        sub, item = 64, 1
    elif dtype.itemsize == 2:
        sub, item = 16, 2
    else:
        sub, item = 8, dtype.itemsize
    shape = tuple(shape)
    if len(shape) == 1:
        shape = (1,) + shape
    minor = -(-shape[-1] // 128) * 128
    second = shape[-2] if shape[-2] == 1 else -(-shape[-2] // sub) * sub
    return int(np.prod(shape[:-2], dtype=np.int64) * second * minor * item)


def _is_vmem(s):
    return str(getattr(s, "memory_space", "")).lower().endswith("vmem")


def _scratch_bytes(shapes):
    return sum(_padded_bytes(s.shape, s.dtype) for s in shapes if _is_vmem(s))


def _collective_geometry(cfg: Config, batch, tp):
    """`(rows, width, gather_width)` for `collectives.scratch_shapes` covering both all-reduce
    payloads (`[8, H]` and `[roundup(K*B, 8), Hm]`, folded to `tp*128` lanes) and both
    all-gathers (`[8, Hm/tp]` and `[8, H/tp]`, bf16)."""
    unit = tp * 128
    kb = max(8, -(-cfg.top_k * batch // 8) * 8)
    candidates = [(8, cfg.hidden)]
    if cfg.moe_hidden >= unit:
        candidates.append((kb, cfg.moe_hidden))
    else:
        f = unit // cfg.moe_hidden
        candidates.append((-(-kb // (8 * f)) * 8, unit))
    rows, width = max(candidates, key=lambda rw: collectives.wire_rows(rw[0], rw[1], tp))
    return rows, width, max(cfg.moe_hidden // tp, cfg.hidden // tp)


def default_hier(batch):
    """The hierarchical expert-output all-reduce pays only for the B=8 `[64, 4096]` payload."""
    return batch >= 8


def scratch_shapes(
    cfg: Config, batch, tp=8, moe_slots=moe.SLOTS, aux_hidden=False, wire=BF16,
    banks=layout.BANKS, packed=True, hier=None, expert_format="int4",
):
    """The kernel's scratch, grouped: `{group: tuple of pltpu.VMEM / semaphores}` in order.
    `packed=False` (interpret mode) uses plain int4 expert slots; `expert_format` selects the
    int4 (`moe`) or NVFP4 (`fp4`) expert slots."""
    rows, width, gather_width = _collective_geometry(cfg, batch, tp)
    hq, hkv, D = cfg.heads // tp, cfg.kv_heads // tp, cfg.head_dim
    vectors = []
    shapes = layout.rank_shapes(cfg, tp)
    for name in layout.VECTOR_FAMILIES:
        (_, _, w), dtype = shapes[name]
        vectors += list(stream.vector_scratch(w, dtype))
    groups = {
        "ring": stream.scratch_shapes(cfg, tp, banks),
        "collectives": collectives.scratch_shapes(
            rows, width, tp, gather_rows=8, gather_width=gather_width,
            bf16_wire=wire == BF16, f32_wire=wire == F32,
            hierarchical=default_hier(batch) if hier is None else hier,
        ),
        "attention": attention.scratch_shapes(cfg, batch, tp),
        "attention_io": (
            pltpu.VMEM((batch, hq * D), F32),
            pltpu.VMEM((batch, 2 * hkv * D), F32),
            pltpu.VMEM((batch, hq * D), F32),
        ),
        "moe": _expert_module(expert_format).scratch_shapes(cfg, batch, tp, moe_slots, packed),
        "vectors": tuple(vectors),
        "activations": (
            pltpu.VMEM((batch, cfg.hidden), F32),  # residual stream s
            pltpu.VMEM((batch, cfg.hidden), BF16),  # x_attn
        ),
    }
    if aux_hidden:
        groups["aux"] = (pltpu.VMEM((batch, cfg.hidden), F32), pltpu.SemaphoreType.DMA((1,)))
    return groups


VMEM_EXPLICIT_LIMIT = 58 * MIB  # hw report: leave >= ~6 MiB for spills / internal scratch


def default_moe_slots(
    cfg: Config, batch, tp=8, aux_hidden=False, wire=BF16, banks=layout.BANKS,
    expert_format="int4",
):
    """Largest expert slot count `<= moe.SLOTS` (at least 2) whose explicit VMEM total fits
    `VMEM_EXPLICIT_LIMIT` (real config: 4 for every B <= 8 with the packed int4 slots and
    with the NVFP4 slots)."""
    for slots in range(moe.SLOTS, 2, -1):
        total = vmem_budget(
            cfg, batch, tp, slots, aux_hidden, log=None, wire=wire, banks=banks,
            expert_format=expert_format,
        )
        total = total["total"]
        if total <= VMEM_EXPLICIT_LIMIT:
            return slots
    return 2


def default_geometry(cfg: Config, batch, tp=8, aux_hidden=False, wire=BF16, expert_format="int4"):
    """`(moe_slots, banks)`: the deepest ring in (12, 10, 8) that still fits `moe.SLOTS` expert
    slots, else 12 banks with as many slots as fit (`default_moe_slots`)."""
    for banks in (layout.BANKS, 10, 8):
        slots = default_moe_slots(cfg, batch, tp, aux_hidden, wire, banks, expert_format)
        if slots == moe.SLOTS:
            return slots, banks
    slots = default_moe_slots(cfg, batch, tp, aux_hidden, wire, layout.BANKS, expert_format)
    return slots, layout.BANKS


def vmem_budget(
    cfg: Config, batch, tp=8, moe_slots=None, aux_hidden=False, log=print, wire=BF16,
    banks=layout.BANKS, hier=None, expert_format="int4",
):
    """Explicit VMEM allocations of the kernel in bytes per group (+ `total`), printed via `log`.

    Counts the scratch groups of `scratch_shapes` (tile-padded, see `_padded_bytes`), the VMEM
    input windows (`x0`, `rope`, `final_norm`) and the `[8, Vp]` f32 logits output window. The
    compiler adds ~2.7 MiB of register spill slots on top (measured at real widths, B=8).
    """
    if moe_slots is None:
        moe_slots = default_moe_slots(cfg, batch, tp, aux_hidden, wire, banks, expert_format)
    groups = scratch_shapes(
        cfg, batch, tp, moe_slots, aux_hidden, wire, banks, hier=hier, expert_format=expert_format
    )
    out = {name: _scratch_bytes(shapes) for name, shapes in groups.items()}
    vp = layout.vocab_pad(cfg, tp)
    out["logits_acc"] = _padded_bytes((LOGIT_ROWS, vp), F32)
    out["inputs"] = (
        _padded_bytes((batch, cfg.hidden), F32)
        + _padded_bytes((batch, 2 * cfg.head_dim), F32)
        + _padded_bytes((1, cfg.hidden), BF16)
    )
    out["total"] = sum(out.values())
    if log is not None:
        parts = ", ".join(f"{k} {v / MIB:.2f}" for k, v in out.items() if k != "total")
        log(
            f"decode megakernel VMEM budget (B={batch}, {expert_format} experts, "
            f"moe_slots={moe_slots}, banks={banks}): "
            f"{out['total'] / MIB:.2f} MiB of {VMEM_LIMIT / MIB:.0f} [{parts}]"
        )
    return out


# --------------------------------------------------------------------------------------
# in-kernel helpers
# --------------------------------------------------------------------------------------
def _pad_rows(x, rows):
    b = x.shape[0]
    if b == rows:
        return x
    if b == 1:
        return jnp.broadcast_to(x, (rows,) + x.shape[1:])
    return jnp.pad(x, ((0, rows - b), (0, 0)))


def _all_reduce(x, ws, phase, tp, wire=BF16, hierarchical=False):
    """`all_reduce_rows` of any `[R, W]` f32: rows padded to 8, `W < tp*128` folded into
    `tp*128`-wide rows (row group `i` in lanes `i*W:(i+1)*W`), result sliced back to `[R, W]`."""
    rows, width = x.shape
    unit = tp * 128
    if width % unit == 0:
        rp = -(-rows // 8) * 8
        return collectives.all_reduce_rows(
            _pad_rows(x, rp), ws, phase, wire=wire, hierarchical=hierarchical
        )[:rows]
    if unit % width:
        raise ValueError(f"all-reduce width {width} must divide or be a multiple of {unit}")
    f = unit // width
    rp = -(-rows // (8 * f)) * (8 * f)
    x = _pad_rows(x, rp)
    g = rp // f
    folded = jnp.concatenate([x[i * g : (i + 1) * g] for i in range(f)], axis=1)
    y = collectives.all_reduce_rows(folded, ws, phase, wire=wire)
    return jnp.concatenate([y[:, i * width : (i + 1) * width] for i in range(f)], axis=0)[:rows]


def _all_gather_bf16(x, ws, phase):
    """`all_gather_rows` of a bf16-valued `[B, w]` shard (rows padded to 8) -> f32 `[B, tp*w]`."""
    rows = x.shape[0]
    y = collectives.all_gather_rows(_pad_rows(x.astype(BF16), 8), ws, phase)
    return y[:rows].astype(F32)


def _greedy_tokens(cfg: Config, logits_ref, ws, phase, tp):
    """Greedy token per row of the raw logits shard `[8, Vp]` (ids `>= vocab_used` masked):
    every rank gathers the 8 `(max, argmax)` pairs and takes the lowest id holding the global
    maximum (identical on every rank, ties -> lowest id). Returns `[8, TOKEN_LANES]` int32."""
    rank = lax.axis_index(collectives.AXIS)
    rows, vp = logits_ref.shape
    col = lax.broadcasted_iota(I32, (rows, vp), 1)
    gid = col + rank * vp
    masked = jnp.where(gid < cfg.vocab_used, logits_ref[...], -jnp.inf)
    local_max = jnp.max(masked, axis=1, keepdims=True)  # [rows, 1]
    local_id = jnp.min(jnp.where(masked == local_max, gid, I32(2**30)), axis=1, keepdims=True)
    lane = lax.broadcasted_iota(I32, (rows, TOKEN_LANES), 1)
    pair = jnp.where(lane == 0, local_max, jnp.where(lane == 1, local_id.astype(F32), 0.0))
    gathered = collectives.all_gather_rows(pair, ws, phase)  # [rows, tp*128]
    maxima = jnp.concatenate([gathered[:, r * TOKEN_LANES : r * TOKEN_LANES + 1] for r in range(tp)], 1)
    ids = jnp.concatenate(
        [gathered[:, r * TOKEN_LANES + 1 : r * TOKEN_LANES + 2] for r in range(tp)], 1
    ).astype(I32)
    best = jnp.max(maxima, axis=1, keepdims=True)
    token = jnp.min(jnp.where(maxima == best, ids, I32(2**30)), axis=1, keepdims=True)
    return jnp.broadcast_to(token, (rows, TOKEN_LANES))


def _kernel_body(cfg: Config, batch, tp, opts):
    """Build the `pallas_call` body closure for `cfg`/`batch`."""
    L, H, Hm = cfg.layers, cfg.hidden, cfg.moe_hidden
    B = batch
    r16 = stream.r16
    names = weight_names(opts.expert_format)
    n_weights = len(names)
    xp = _expert_module(opts.expert_format)
    groups = scratch_shapes(
        cfg, B, tp, opts.moe_slots, opts.aux_hidden, opts.wire, opts.banks,
        packed=not opts.interpret, hier=opts.hier, expert_format=opts.expert_format,
    )
    sizes = {name: len(shapes) for name, shapes in groups.items()}

    def body(pos_ref, x0_ref, rope_ref, final_norm_ref, *refs):
        weights = dict(zip(names, refs[:n_weights]))
        refs = refs[n_weights:]
        k_in, v_in, logits_ref, tokens_ref, k_cache, v_cache = refs[:6]
        del k_in, v_in  # aliased to k_cache / v_cache
        refs = refs[6:]
        aux_ref = None
        if opts.aux_hidden:
            aux_ref, refs = refs[0], refs[1:]
        scratch = {}
        for name in groups:
            scratch[name], refs = refs[: sizes[name]], refs[sizes[name] :]
        assert not refs, len(refs)

        skip = opts.skip
        lm_head = None if "lm_head" in skip else weights["lm_head"]
        ring = stream.make_ring(cfg, scratch["ring"], weights, lm_head, tp=tp)
        ws = collectives.workspace(
            *scratch["collectives"], f32_wire=opts.wire == F32, hierarchical=opts.hier
        )
        dots = "dense_dots" not in skip

        def gemv(x, family, l, **kw):
            return stream.gemv(ring, x, family, l, compute=dots, **kw)

        def all_reduce(x, phase, hierarchical=False):
            if "collectives" in skip:
                return x
            return _all_reduce(x, ws, phase, tp, opts.wire, hierarchical and opts.hier)

        def all_gather(x, phase):
            if "collectives" in skip:
                return jnp.tile(r16(x), (1, tp))
            return _all_gather_bf16(x, ws, phase)

        attn_ws = scratch["attention"]
        q_ref, kv_ref, g_ref = scratch["attention_io"]
        sc = xp.bind(scratch["moe"])
        experts = xp.weights(weights)
        vec = {}
        for i, name in enumerate(layout.VECTOR_FAMILIES):
            vec[name] = (weights[name], scratch["vectors"][2 * i], scratch["vectors"][2 * i + 1])
        s_ref, x_ref = scratch["activations"]

        def prefetch_vectors(l):
            for hbm, buf, sems in vec.values():
                stream.prefetch_vector(hbm, buf, sems, l)

        def wait_vector(name, l):
            hbm, buf, sems = vec[name]
            stream.wait_vector(hbm, buf, sems, l)

        def v(name, l):
            return stream.vector(vec[name][1], l)  # [1, W]

        def write_aux(index, value):
            if aux_ref is None:
                return
            stage, sem = scratch["aux"]
            stage[...] = value
            copy = pltpu.make_async_copy(stage, aux_ref.at[index], sem.at[0])
            copy.start()
            copy.wait()

        # ---- prologue ---------------------------------------------------------------------
        collectives.barrier(tp)
        stream.prime(ring)
        prefetch_vectors(0)
        s0 = x0_ref[...]
        s_ref[...] = s0
        wait_vector("attn_norm", 0)
        x_ref[...] = stream.norm_to_bf16(s0, v("attn_norm", 0), cfg.rms_eps)
        write_aux(0, s0)

        # ---- layer loop -------------------------------------------------------------------
        def layer(l, carry):
            phase = l * COLLECTIVES_PER_LAYER
            for name in layout.VECTOR_FAMILIES:
                if name != "attn_norm":  # waited at the end of the previous layer
                    wait_vector(name, l)

            @pl.when(l + 1 < L)
            def _prefetch_next():
                prefetch_vectors(l + 1)

            # -- attention ------------------------------------------------------------------
            kv_early = "attention" not in skip and not opts.kv_late
            if kv_early:  # KV tiles ahead of this layer's dense refills in the DMA queue
                attention.prefetch_kv(cfg, B, l, pos_ref, k_cache, v_cache, attn_ws)
            q_ref[...] = r16(gemv(x_ref, "q", l))
            kv_ref[...] = r16(gemv(x_ref, "kv", l))
            g_ref[...] = r16(gemv(x_ref, "gate", l))
            if "attention" in skip:
                o = q_ref[...]
            else:
                o = attention.attention_layer(
                    cfg, B, l, pos_ref, q_ref, kv_ref, g_ref, rope_ref, k_cache, v_cache,
                    attn_ws, tp=tp, wait_writes=False, primed=kv_early,
                )
            partial_o = gemv(o, "o", l)  # [B, H] f32 partial sums
            if "attention" not in skip:
                attention.wait_cache_writes(cfg, B, l, pos_ref, k_cache, v_cache, attn_ws)
            attn_out = r16(all_reduce(partial_o, phase))
            # -- post-attention boundary ----------------------------------------------------
            nb = r16(stream.rms(attn_out, None, cfg.post_eps))
            s = stream.gated_residual(s_ref, nb, v("attn_gate_alpha", l), v("attn_gate_beta", l))
            s_ref[...] = s
            x_ffn = stream.norm_to_bf16(s, v("ffn_norm", l), cfg.rms_eps)  # [B, H] bf16
            # -- MoE ------------------------------------------------------------------------
            # Refills targeting the post / next-layer tiles are held back until the expert
            # slabs of the first two waves are in the (FIFO) DMA queue.
            defer = {"none": None, "post": stream.post_offset(ring), "next": ring.per}[opts.defer]
            h0 = gemv(x_ffn, "pre", l, defer_from=defer)  # [B, Hm/tp] f32
            logits = gemv(x_ffn, "router_hi", l, defer_from=defer) + gemv(
                x_ffn, "router_lo", l, defer_from=defer
            )
            if "route" in skip:
                lane = lax.broadcasted_iota(I32, (B, moe.LANES), 1)
                idx = jnp.where(lane < cfg.top_k, lane, cfg.experts)
                w = jnp.where(lane < cfg.top_k, F32(1.0 / cfg.top_k), F32(0)) + logits[:, :1] * 0
            else:
                idx, w = moe.route_from_logits(cfg, logits, v("router_bias", l))
            moe.route_to_scratch(cfg, sc, idx, w)
            if "experts" not in skip:
                xp.start(cfg, l, experts, sc)
            h0 = all_gather(r16(h0), phase + 1)  # [B, Hm]
            h1 = stream.norm_to_bf16(h0, v("pre_expert_norm", l), cfg.rms_eps)
            if "experts" not in skip:

                def after_wave(w, n_waves=None):
                    # n_waves is None for streams that only report the wave index (fp4):
                    # flush after the first wave then.
                    target = 0 if opts.flush == "first" or n_waves is None else n_waves - 1
                    if isinstance(w, int):  # static waves (B=1)
                        if w == target:
                            stream.flush_deferred(ring)
                        return

                    @pl.when(w == target)
                    def _flush():
                        stream.flush_deferred(ring)

                extra = {} if "expert_dots" not in skip else {"compute": False}
                xp.stream(cfg, l, h1, experts, sc, started=True, after_wave=after_wave, **extra)
                assert not ring.deferred  # flushed inside the wave loop (>= 1 wave per layer)
            else:
                sc.y_out[...] = jnp.broadcast_to(h1[:1].astype(F32), sc.y_out.shape) * 0.01
                stream.flush_deferred(ring)
            Y = all_reduce(sc.y_out[...], phase + 2, hierarchical=True)  # [K*B, Hm]
            m = moe.finalize(cfg, Y, sc.w, v("post_expert_norm", l), B)  # [B, Hm]
            out = gemv(m, "post", l)  # [B, H/tp] f32
            ffn_out = all_gather(r16(out), phase + 3)  # [B, H]
            # -- post-FFN boundary ----------------------------------------------------------
            t = r16(ffn_out * v("post_ffn_norm", l).astype(F32))
            nb = r16(t * lax.rsqrt(jnp.mean(t * t, axis=-1, keepdims=True) + cfg.post_eps))
            s = stream.gated_residual(s_ref, nb, v("ffn_gate_alpha", l), v("ffn_gate_beta", l))
            s_ref[...] = s
            write_aux(l + 1, s)

            @pl.when(l + 1 < L)
            def _next_input():
                wait_vector("attn_norm", l + 1)
                x_ref[...] = stream.norm_to_bf16(s, v("attn_norm", l + 1), cfg.rms_eps)

            return carry

        lax.fori_loop(0, L, layer, 0)

        # ---- final norm + lm_head ---------------------------------------------------------
        hN = stream.norm_to_bf16(r16(s_ref[...]), final_norm_ref[...], cfg.rms_eps)
        if "lm_head" in skip:
            logits_ref[:, pl.ds(0, H)] = jnp.broadcast_to(hN.astype(F32)[:1], (LOGIT_ROWS, H))
        else:
            gemv(hN, "lm_head", L, acc=logits_ref)
        if "collectives" in skip:
            tokens_ref[...] = jnp.zeros((LOGIT_ROWS, TOKEN_LANES), I32)
        else:
            tokens_ref[...] = _greedy_tokens(cfg, logits_ref, ws, L * COLLECTIVES_PER_LAYER, tp)

    return body, groups


def make_kernel(cfg: Config, context, batch, *, tp=8, options=frozenset(), expert_format="int4"):
    """Per-rank `kernel(pos, x0, rope, final_norm, weights, k_cache, v_cache) -> (logits [8, Vp]
    f32, tokens [8, 128] i32, k_cache, v_cache[, aux])` around the `pallas_call` (no
    shard_map; `weights` is the per-rank dict without the leading rank axis). Used by
    `make_decode`."""
    opts = parse_options(options)
    opts.expert_format = expert_format
    layout.check_tp(cfg, tp, expert_format)
    if opts.moe_slots is None and opts.banks is None:
        opts.moe_slots, opts.banks = default_geometry(
            cfg, batch, tp, opts.aux_hidden, opts.wire, expert_format
        )
    elif opts.banks is None:
        opts.banks = layout.BANKS
    elif opts.moe_slots is None:
        opts.moe_slots = default_moe_slots(
            cfg, batch, tp, opts.aux_hidden, opts.wire, opts.banks, expert_format
        )
    if opts.defer is None:
        opts.defer = "next" if batch <= 4 else "none"
    if opts.hier is None:
        opts.hier = default_hier(batch)
    if context % attention.TOKENS:
        raise ValueError(f"context must be a multiple of {attention.TOKENS}")
    if not 1 <= batch <= moe.MR:
        raise ValueError(f"batch must be in 1..{moe.MR}")
    vp = layout.vocab_pad(cfg, tp)
    lanes = layout.cache_lanes(cfg, tp)
    cache_shape = (cfg.layers, batch, context, lanes)
    body, groups = _kernel_body(cfg, batch, tp, opts)
    scratch = [s for shapes in groups.values() for s in shapes]
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    names = weight_names(expert_format)
    n_weights = len(names)
    k_index = 4 + n_weights
    out_shape = [
        jax.ShapeDtypeStruct((LOGIT_ROWS, vp), F32),
        jax.ShapeDtypeStruct((LOGIT_ROWS, TOKEN_LANES), I32),
        jax.ShapeDtypeStruct(cache_shape, BF16),
        jax.ShapeDtypeStruct(cache_shape, BF16),
    ]
    out_specs = [vmem, vmem, hbm, hbm]
    if opts.aux_hidden:
        out_shape.append(jax.ShapeDtypeStruct((cfg.layers + 1, batch, cfg.hidden), F32))
        out_specs.append(hbm)
    call = pl.pallas_call(
        body,
        out_shape=tuple(out_shape),
        in_specs=[smem, vmem, vmem, vmem] + [hbm] * (n_weights + 2),
        out_specs=tuple(out_specs),
        scratch_shapes=scratch,
        input_output_aliases={k_index: 2, k_index + 1: 3},
        compiler_params=collectives.compiler_params(COLLECTIVE_ID, vmem_limit_bytes=VMEM_LIMIT),
        interpret=pltpu.InterpretParams(dma_execution_mode="eager") if opts.interpret else False,
        name="musespark_decode_step",
    )

    def kernel(pos, x0, rope, final_norm, weights, k_cache, v_cache):
        if tuple(k_cache.shape) != cache_shape:
            raise ValueError(f"k_cache {k_cache.shape} != {cache_shape}")
        args = [weights[name] for name in names]
        return call(pos, x0, rope, final_norm, *args, k_cache, v_cache)

    return kernel, opts


def make_decode(
    mesh, cfg: Config, context, batch, *, greedy=True, return_logits=False, tp=8,
    options=frozenset(), expert_format=None,
):
    """Jitted `decode(weights, caches, tokens [B] i32, pos [B] i32) -> (next_tokens [B] i32,
    logits [B, V] f32 or None, caches)` (design.md 5.6); `caches` are donated.

    `next_tokens` are always the greedy (argmax) tokens, replicated; `logits` (with
    `return_logits`, or when `greedy=False`) are the softcapped full-vocabulary logits with
    ids `>= vocab_used` masked to -inf, for `sampling.sample`. With the `"aux_hidden"` option
    a fourth output `[L + 1, B, H]` (the f32 residual stream per layer) is appended.
    `expert_format` ("int4" / "nvfp4", `layout.EXPERT_FORMATS`) selects the expert kernel;
    None (default) infers it from the family names of `weights` at the first call
    (`layout.expert_format_of`), so the same program serves both container formats.
    """
    if mesh.size != tp:
        raise ValueError(f"mesh has {mesh.size} devices, tp={tp}")
    options = frozenset(options)
    return_logits = return_logits or not greedy
    vp = layout.vocab_pad(cfg, tp)
    built = {}

    def kernel_for(fmt):
        if fmt not in built:
            kernel, opts = make_kernel(cfg, context, batch, tp=tp, options=options, expert_format=fmt)
            vmem_budget(
                cfg, batch, tp, opts.moe_slots, opts.aux_hidden, wire=opts.wire,
                banks=opts.banks, hier=opts.hier, expert_format=fmt,
            )
            built[fmt] = (kernel, opts)
        return built[fmt]

    if expert_format is not None:
        kernel_for(expert_format)
    aux_hidden = parse_options(options).aux_hidden

    def local(weights, caches, tokens, pos):
        fmt = expert_format or layout.expert_format_of(weights)
        kernel, opts = kernel_for(fmt)
        w = {name: value[0] for name, value in weights.items()}
        kc, vc = caches["k_cache"][0], caches["v_cache"][0]
        rank = lax.axis_index("tp")
        tokens = jnp.asarray(tokens, I32)
        pos = jnp.asarray(pos, I32)
        # embedding: vocab-sharded lookup, exactly one rank holds the row -> psum is exact
        lo = rank * vp
        rows = jnp.take(w["embed"], jnp.clip(tokens - lo, 0, vp - 1), axis=0).astype(F32)
        in_range = (tokens >= lo) & (tokens < lo + vp)
        e0 = lax.psum(jnp.where(in_range[:, None], rows, 0.0), "tp")
        x0 = musespark.r16(musespark.rms(e0, None, cfg.rms_eps))  # [B, H] f32
        rope = attention.rope_table(cfg, pos)
        out = kernel(pos, x0, rope, w["final_norm"], w, kc, vc)
        shard = out[0][:batch]  # [B, Vp] raw lm_head output
        next_tokens = out[1][:batch, 0]  # greedy tokens, identical on every rank
        logits = sampling.gather_logits_local(shard, rank, cfg, tp) if return_logits else None
        caches = {"k_cache": out[2][None], "v_cache": out[3][None]}
        result = (next_tokens, logits, caches)
        if opts.aux_hidden:
            result += (out[4],)
        return result

    def decode(weights, caches, tokens, pos):
        w_specs = {name: P("tp") for name in weights}
        c_specs = {name: P("tp") for name in caches}
        out_specs = (P(), P() if return_logits else None, c_specs)
        if aux_hidden:
            out_specs += (P(),)
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(w_specs, c_specs, P(), P()),
            out_specs=out_specs,
            check_vma=False,
        )(weights, caches, tokens, pos)

    return jax.jit(decode, donate_argnums=(1,))


__all__ = [
    "COLLECTIVE_ID",
    "WEIGHT_NAMES",
    "weight_names",
    "make_decode",
    "make_kernel",
    "parse_options",
    "scratch_shapes",
    "vmem_budget",
]
