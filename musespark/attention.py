"""Muse Spark decode attention mini-kernel: one layer, one rank, B independent rows.

The integrator projects the normed input (`q`, `kv`, `g` via the dense ring) and does the
o-projection + all-reduce; this module does everything in between (spec section 3.3):

    q_n = r16(rms_head(q, eps))      k_n = r16(rms_head(k, eps))          parameter-free, per head
    RoPE (rotate-half, theta 5e5) on q_n/k_n on sliding layers only; NoPE on `l % 4 == 1`
    cache[layer, b, pos[b]] <- k_n, v                                       (bf16 tile write-back)
    scores = scale * q_n . k_n over slots lo[b] <= s <= pos[b]              scale = qk_scale / sqrt(D)
    lo[b] = 0 (full layers) or max(0, pos[b] - (window - 1))                (sliding layers)
    o = r16(rms_head(softmax(scores) . v, eps) * sigmoid(g))                element-wise gate

Per-rank layouts (tp ranks, `hq = heads / tp` query heads, `hkv = kv_heads / tp` kv heads,
`gs = heads / kv_heads` queries per kv head, `D = head_dim`, `lanes = cache_lanes(cfg, tp)`):

    pos_ref   SMEM [B] int32     absolute 0-based position of the token decoded in row b
    q_ref     VMEM [B, hq*D] f32 bf16-valued projections, head-major (head h = lanes h*D:(h+1)*D)
    kv_ref    VMEM [B, 2*hkv*D]  k heads in lanes [0, hkv*D), v heads in [hkv*D, 2*hkv*D)
    g_ref     VMEM [B, hq*D]     attention output-gate logits, same layout as q
    rope_ref  VMEM [B, 2*D] f32  `rope_table(cfg, pos)`: [cos_j | cos_j | -sin_j | sin_j], j < D/2
    k_cache   HBM  [L, B, context, lanes] bf16  lanes h*D:(h+1)*D = local kv head h (post-norm,
    v_cache   HBM  [L, B, context, lanes] bf16  post-RoPE keys; raw values); slot s = position s
    returns   [B, hq*D] f32 (bf16-valued) = gated, per-head-normalised attention output

The KV cache is streamed in `TOKENS`-token tiles `[TOKENS, lanes]` per row, `SLOTS`-deep
buffered, over blocks `[lo // TOKENS, pos // TOKENS]`; all B rows advance together (one block step
per loop iteration, rows with fewer blocks re-read their last block fully masked). The current
token is patched into the resident tile with a slot mask, and the patched tile of the last block
is written back to HBM from a staging buffer (`wait_cache_writes`).

Scores use a block-diagonal `Q2 [hq, lanes]` (query head h in lanes of its kv head) so one
`dot_general` (contracting lanes with lanes) scores every local kv head per row; `P . V` then
gives `[hq, lanes]` from which each head's `D` lanes are extracted.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from musespark.config import Config
from musespark.layout import cache_lanes

BF16 = jnp.bfloat16
F32 = jnp.float32
TOKENS = 256  # tokens per KV tile; context must be a multiple (measured: 256 beats 128 by ~1.5x)
SLOTS = 3  # KV tile buffers per cache: 2 in flight while 1 is being consumed (3 beats 2)


class Workspace(NamedTuple):
    """Scoped VMEM + semaphores of the attention phase, in `scratch_shapes` order."""

    kbuf: object  # [SLOTS, B, TOKENS, lanes] bf16   K tiles, one per row per slot
    vbuf: object  # [SLOTS, B, TOKENS, lanes] bf16   V tiles
    kstage: object  # [B, TOKENS, lanes] bf16        patched last tile, source of the write-back
    vstage: object  # [B, TOKENS, lanes] bf16
    q2: object  # [B, hq, lanes] bf16             block-diagonal queries (post-norm, post-rope)
    knew: object  # [B, 1, lanes] bf16            new key row in cache-lane layout
    vnew: object  # [B, 1, lanes] bf16            new value row in cache-lane layout
    m: object  # [B, hq, 1] f32                   online-softmax running max
    l: object  # [B, hq, 1] f32                   running denominator
    acc: object  # [B, hq, lanes] f32             running P.V
    ksem: object  # DMA [SLOTS]                   one per slot, shared by the B row tiles
    vsem: object  # DMA [SLOTS]
    wsem: object  # DMA [2]                       write-back semaphores (K, V)


def scratch_shapes(cfg: Config, batch: int, tp: int = 8) -> tuple:
    """VMEM/semaphore allocations for `attention_layer` (see `Workspace`)."""
    lanes = cache_lanes(cfg, tp)
    hq = cfg.heads // tp
    return (
        pltpu.VMEM((SLOTS, batch, TOKENS, lanes), BF16),
        pltpu.VMEM((SLOTS, batch, TOKENS, lanes), BF16),
        pltpu.VMEM((batch, TOKENS, lanes), BF16),
        pltpu.VMEM((batch, TOKENS, lanes), BF16),
        pltpu.VMEM((batch, hq, lanes), BF16),
        pltpu.VMEM((batch, 1, lanes), BF16),
        pltpu.VMEM((batch, 1, lanes), BF16),
        pltpu.VMEM((batch, hq, 1), F32),
        pltpu.VMEM((batch, hq, 1), F32),
        pltpu.VMEM((batch, hq, lanes), F32),
        pltpu.SemaphoreType.DMA((SLOTS,)),
        pltpu.SemaphoreType.DMA((SLOTS,)),
        pltpu.SemaphoreType.DMA((2,)),
    )


def scratch_bytes(cfg: Config, batch: int, tp: int = 8) -> int:
    """VMEM bytes of `scratch_shapes` (tiles padded to [16, 128] bf16 / [8, 128] f32)."""
    lanes = cache_lanes(cfg, tp)
    hq = cfg.heads // tp
    tiles = (2 * SLOTS + 2) * batch * TOKENS * lanes * 2
    q2 = batch * max(hq, 16) * lanes * 2 + 2 * batch * 16 * lanes * 2
    stats = 2 * batch * max(hq, 8) * 128 * 4 + batch * max(hq, 8) * lanes * 4
    return tiles + q2 + stats


def rope_table(cfg: Config, pos) -> jax.Array:
    """`[B, 2*D] f32` rotate-half factors per row: `[cos | cos | -sin | sin]` of `pos * inv_freq`.

    `inv_freq_j = theta ** (-2j / D)` and `ang = pos * inv_freq` in f32, evaluated with the same
    jnp expressions as `musespark.rope_rotate_half` (bit-identical on one backend). Computed by
    XLA glue once per decode step; the kernel disables the rotation on NoPE layers itself.
    """
    D = cfg.head_dim
    inv = cfg.rope_theta ** (-jnp.arange(0, D, 2, dtype=F32) / D)
    ang = jnp.asarray(pos, jnp.int32).astype(F32)[:, None] * inv[None, :]
    c, s = jnp.cos(ang), jnp.sin(ang)
    return jnp.concatenate([c, c, -s, s], axis=-1)


def rope_table_np(cfg: Config, pos) -> np.ndarray:
    """`rope_table` evaluated in float64 and rounded to f32 (the most accurate table)."""
    D = cfg.head_dim
    inv = cfg.rope_theta ** (-np.arange(0, D, 2, dtype=np.float64) / D)
    ang = np.asarray(pos, np.int64).astype(np.float64)[:, None] * inv[None, :]
    c, s = np.cos(ang), np.sin(ang)
    return np.concatenate([c, c, -s, s], axis=-1).astype(np.float32)


def _r16(x):
    return x.astype(BF16).astype(F32)


def _head_norm(x, eps):
    """Parameter-free RMSNorm over the minor (head_dim) axis, in f32, rounded to bf16."""
    return _r16(x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps))


def _rotate_half(x, cos, sin):
    """`x [B, h, D]` (bf16-valued f32) rotated by `cos`, `sin` `[B, 1, D]` (`sin` sign-folded)."""
    half = x.shape[-1] // 2
    partner = jnp.concatenate([x[..., half:], x[..., :half]], axis=-1)
    return _r16(x * cos + partner * sin)


def _head_iota(shape, gs):
    """kv-head index of each query row of a `[B, hq, *]` array."""
    return jax.lax.broadcasted_iota(jnp.int32, shape, 1) // gs


def _block_diag(x, gs, hkv, lanes):
    """`[B, hq, D]` -> `[B, hq, lanes]`: query head h moved into the lanes of kv head h // gs."""
    d = x.shape[-1]
    group = _head_iota(x.shape, gs)
    parts = [jnp.where(group == g, x, 0.0) for g in range(hkv)]
    if hkv * d < lanes:
        parts.append(jnp.zeros(x.shape[:-1] + (lanes - hkv * d,), x.dtype))
    return jnp.concatenate(parts, axis=-1)


def _row_layout(x, lanes):
    """`[B, hkv, D]` -> `[B, 1, lanes]`: kv heads side by side in cache-lane order, zero padded."""
    b, hkv, d = x.shape
    parts = [x[:, g : g + 1, :] for g in range(hkv)]
    if hkv * d < lanes:
        parts.append(jnp.zeros((b, 1, lanes - hkv * d), x.dtype))
    return jnp.concatenate(parts, axis=-1)


def _extract_heads(o2, gs, hkv, d):
    """`[B, hq, lanes]` -> `[B, hq, D]`: take each query head's lanes (those of its kv head)."""
    group = _head_iota(o2.shape[:-1] + (d,), gs)
    out = o2[..., :d]
    for g in range(1, hkv):
        out = jnp.where(group == g, o2[..., g * d : (g + 1) * d], out)
    return out


def _tile_copy(cache, buf, sem, layer, row, block, slot):
    start = pl.multiple_of(block * TOKENS, TOKENS)
    return pltpu.make_async_copy(
        cache.at[layer, row, pl.ds(start, TOKENS), :], buf.at[slot, row], sem.at[slot]
    )


def _write_copy(stage, cache, sem, layer, row, block):
    start = pl.multiple_of(block * TOKENS, TOKENS)
    return pltpu.make_async_copy(stage.at[row], cache.at[layer, row, pl.ds(start, TOKENS), :], sem)


def _block_plan(cfg: Config, batch, layer, pos_ref):
    """Per-row scalars: `(pos, lo, first block, last block, block count, max count)`."""
    layer = jnp.asarray(layer, jnp.int32)
    is_full = (layer % cfg.full_attention_every) == cfg.full_attention_offset
    not_full = 1 - is_full.astype(jnp.int32)
    pos = [pos_ref[b] for b in range(batch)]
    lo = [jnp.maximum(p - (cfg.sliding_window - 1), 0) * not_full for p in pos]
    first = [x // TOKENS for x in lo]
    last = [p // TOKENS for p in pos]
    count = [la - fi + 1 for fi, la in zip(first, last)]
    nsteps = functools.reduce(jnp.maximum, count)
    return pos, lo, first, last, count, nsteps


def _issue_tiles(batch, layer, first, last, k_cache, v_cache, ws, j, slot):
    """Start the K and V tile DMAs of block step `j` of every row into `slot`."""
    for b in range(batch):
        blk = jnp.minimum(first[b] + j, last[b])  # rows past their last block re-read it
        _tile_copy(k_cache, ws.kbuf, ws.ksem, layer, b, blk, slot).start()
        _tile_copy(v_cache, ws.vbuf, ws.vsem, layer, b, blk, slot).start()


def _prime_tiles(batch, layer, first, last, nsteps, k_cache, v_cache, ws):
    for j in range(SLOTS):

        @pl.when(j < nsteps)
        def prime(j=j):
            _issue_tiles(batch, layer, first, last, k_cache, v_cache, ws, j, j)


def prefetch_kv(cfg: Config, batch: int, layer, pos_ref, k_cache, v_cache, ws):
    """Issue the first `SLOTS` KV tile blocks of `attention_layer(..., primed=True)` early.

    Legal as soon as the previous layer's attention loop has finished (the tile buffers are
    free; the write-backs use the separate staging buffers)."""
    ws = Workspace(*ws)
    _, _, first, last, _, nsteps = _block_plan(cfg, batch, layer, pos_ref)
    _prime_tiles(batch, layer, first, last, nsteps, k_cache, v_cache, ws)


def wait_cache_writes(cfg: Config, batch: int, layer, pos_ref, k_cache, v_cache, ws):
    """Wait for the K/V tile write-backs issued by `attention_layer(..., wait_writes=False)`.

    Must run before the next `attention_layer` call reuses the staging buffers (any time after
    the o-projection is fine). `pos_ref` and `layer` must be the ones passed to that call.
    """
    ws = Workspace(*ws)
    for b in range(batch):
        block = pos_ref[b] // TOKENS
        _write_copy(ws.kstage, k_cache, ws.wsem.at[0], layer, b, block).wait()
        _write_copy(ws.vstage, v_cache, ws.wsem.at[1], layer, b, block).wait()


def attention_layer(
    cfg: Config,
    batch: int,
    layer,
    pos_ref,
    q_ref,
    kv_ref,
    g_ref,
    rope_ref,
    k_cache,
    v_cache,
    ws,
    *,
    tp: int = 8,
    wait_writes: bool = True,
    primed: bool = False,
):
    """One layer of decode attention for `batch` rows (layouts in the module docstring).

    `layer` may be a traced int32 (the layer loop index): the layer kind is derived
    arithmetically (`cfg.full_attention_every/offset`), no `lax.cond`. Returns the
    `[B, hq*D]` f32 (bf16-valued) gated attention output; the new K/V rows are written to
    `k_cache/v_cache[layer, b, pos[b]]` (asynchronously unless `wait_writes`). With
    `primed=True` the first `SLOTS` KV tiles were already issued by `prefetch_kv` (same
    `layer`, `pos_ref`), e.g. at the start of the layer, ahead of the dense-weight DMAs.
    """
    ws = Workspace(*ws)
    if k_cache.shape[2] % TOKENS or v_cache.shape[2] % TOKENS:
        raise ValueError(f"context {k_cache.shape[2]} must be a multiple of TOKENS={TOKENS}")
    B = batch
    hq, hkv, gs, D = cfg.heads // tp, cfg.kv_heads // tp, cfg.group_size_q, cfg.head_dim
    lanes = cache_lanes(cfg, tp)
    scale = cfg.softmax_scale
    layer = jnp.asarray(layer, jnp.int32)
    is_full = (layer % cfg.full_attention_every) == cfg.full_attention_offset
    not_full = 1 - is_full.astype(jnp.int32)

    # ---- per-row block range (scalars) --------------------------------------------------
    pos, lo, first, last, count, nsteps = _block_plan(cfg, B, layer, pos_ref)

    # ---- q / k / v preparation (all rows at once) --------------------------------------
    q = _head_norm(q_ref[...].reshape(B, hq, D), cfg.rms_eps)
    k = _head_norm(kv_ref[:, : hkv * D].reshape(B, hkv, D), cfg.rms_eps)
    v = kv_ref[:, hkv * D : 2 * hkv * D].reshape(B, hkv, D)
    rope = rope_ref[...].reshape(B, 1, 2 * D)
    cos = jnp.where(is_full, 1.0, rope[..., :D])  # NoPE layers: identity rotation, exact
    sin = jnp.where(is_full, 0.0, rope[..., D:])
    q = _rotate_half(q, cos, sin)
    k = _rotate_half(k, cos, sin)
    ws.q2[...] = _block_diag(q, gs, hkv, lanes).astype(BF16)
    ws.knew[...] = _row_layout(k, lanes).astype(BF16)
    ws.vnew[...] = _row_layout(v, lanes).astype(BF16)
    ws.m[...] = jnp.full((B, hq, 1), -jnp.inf, F32)
    ws.l[...] = jnp.zeros((B, hq, 1), F32)
    ws.acc[...] = jnp.zeros((B, hq, lanes), F32)

    # ---- KV tile stream -------------------------------------------------------------------
    def issue(j, slot):
        _issue_tiles(B, layer, first, last, k_cache, v_cache, ws, j, slot)

    def wait(j, slot):
        # DMA semaphores count bytes: one wait sized like the B row tiles of the slot.
        rows = pl.ds(0, B)
        src_k = k_cache.at[layer, rows, pl.ds(0, TOKENS), :]
        src_v = v_cache.at[layer, rows, pl.ds(0, TOKENS), :]
        pltpu.make_async_copy(src_k, ws.kbuf.at[slot], ws.ksem.at[slot]).wait()
        pltpu.make_async_copy(src_v, ws.vbuf.at[slot], ws.vsem.at[slot]).wait()

    if not primed:
        _prime_tiles(B, layer, first, last, nsteps, k_cache, v_cache, ws)

    tok = jax.lax.broadcasted_iota(jnp.int32, (TOKENS, lanes), 0)  # token within the tile
    col = jax.lax.broadcasted_iota(jnp.int32, (1, TOKENS), 1)

    def step(j, carry):
        slot = j % SLOTS
        wait(j, slot)
        # Write-backs first (conditional regions), so the B rows' compute below shares one
        # basic block and the scheduler can overlap their MXU chains.
        for b in range(B):

            @pl.when(j == count[b] - 1)
            def write_back(b=b):
                is_cur = tok + last[b] * TOKENS == pos[b]
                ws.kstage[b] = jnp.where(is_cur, ws.knew[b], ws.kbuf[slot, b])
                ws.vstage[b] = jnp.where(is_cur, ws.vnew[b], ws.vbuf[slot, b])
                _write_copy(ws.kstage, k_cache, ws.wsem.at[0], layer, b, last[b]).start()
                _write_copy(ws.vstage, v_cache, ws.wsem.at[1], layer, b, last[b]).start()

        for b in range(B):
            base = (first[b] + j) * TOKENS  # unclamped: past the last block every slot is masked
            # Patch the current token into the resident tile (its cache row is stale).
            is_cur = tok + base == pos[b]
            kt = jnp.where(is_cur, ws.knew[b], ws.kbuf[slot, b])
            vt = jnp.where(is_cur, ws.vnew[b], ws.vbuf[slot, b])
            s = jax.lax.dot_general(
                ws.q2[b], kt, (((1,), (1,)), ((), ())), preferred_element_type=F32
            )
            valid = (col + base <= pos[b]) & (col + base >= lo[b])
            s = jnp.where(valid, s * scale, -jnp.inf)
            m_old = ws.m[b]
            m_new = jnp.maximum(m_old, jnp.max(s, axis=-1, keepdims=True))
            alpha = jnp.exp(m_old - m_new)
            p = jnp.exp(s - m_new)
            p_hi = p.astype(BF16)
            p_lo = (p - p_hi.astype(F32)).astype(BF16)
            pv = jnp.dot(p_hi, vt, preferred_element_type=F32) + jnp.dot(
                p_lo, vt, preferred_element_type=F32
            )
            ws.acc[b] = ws.acc[b] * alpha + pv
            ws.l[b] = ws.l[b] * alpha + jnp.sum(p, axis=-1, keepdims=True)
            ws.m[b] = m_new

        @pl.when(j + SLOTS < nsteps)
        def refill():
            issue(j + SLOTS, slot)

        return carry

    jax.lax.fori_loop(0, nsteps, step, 0)

    # ---- output: per-head norm, sigmoid gate ------------------------------------------------
    o2 = ws.acc[...] / ws.l[...]
    o = _extract_heads(o2, gs, hkv, D)
    o = o * jax.lax.rsqrt(jnp.mean(o * o, axis=-1, keepdims=True) + cfg.rms_eps)
    o = _r16(o * jax.nn.sigmoid(g_ref[...].reshape(B, hq, D)))
    if wait_writes:
        wait_cache_writes(cfg, B, layer, pos_ref, k_cache, v_cache, ws)
    return o.reshape(B, hq * D)


# ---- pure-jnp reference (same rounding points) -------------------------------------------------


def reference_attention(cfg: Config, layer, q, kv, g, k_cache, v_cache, pos, rope=None, tp=8):
    """Per-rank reference of `attention_layer` on the same layouts (pure jnp, f32 maths).

    `q [B, hq*D]`, `kv [B, 2*hkv*D]`, `g [B, hq*D]` f32; caches `[L, B, context, lanes]` bf16;
    `pos [B]` int32; `rope` = `rope_table(cfg, pos)` (default: computed here). Returns
    `(o [B, hq*D] f32 bf16-valued, k_cache, v_cache)` with slot `pos[b]` of row b updated.
    """
    B = q.shape[0]
    hq, hkv, gs, D = cfg.heads // tp, cfg.kv_heads // tp, cfg.group_size_q, cfg.head_dim
    pos = jnp.asarray(pos, jnp.int32)
    rope = rope_table(cfg, pos) if rope is None else jnp.asarray(rope, F32)
    full = cfg.is_full_attention(layer)
    q = _head_norm(jnp.asarray(q, F32).reshape(B, hq, D), cfg.rms_eps)
    k = _head_norm(jnp.asarray(kv, F32)[:, : hkv * D].reshape(B, hkv, D), cfg.rms_eps)
    v = jnp.asarray(kv, F32)[:, hkv * D :].reshape(B, hkv, D)
    if not full:
        cos, sin = rope[:, None, :D], rope[:, None, D:]
        q, k = _rotate_half(q, cos, sin), _rotate_half(k, cos, sin)
    rows = jnp.arange(B)
    k_cache = k_cache.at[layer, rows, pos, : hkv * D].set(k.reshape(B, hkv * D).astype(BF16))
    v_cache = v_cache.at[layer, rows, pos, : hkv * D].set(v.reshape(B, hkv * D).astype(BF16))
    context = k_cache.shape[2]
    keys = k_cache[layer, :, :, : hkv * D].astype(F32).reshape(B, context, hkv, D)
    vals = v_cache[layer, :, :, : hkv * D].astype(F32).reshape(B, context, hkv, D)
    keys = jnp.repeat(keys, gs, axis=2)
    vals = jnp.repeat(vals, gs, axis=2)
    scores = cfg.softmax_scale * jnp.einsum("bhd,bshd->bhs", q, keys)
    slot = jnp.arange(context)[None, :]
    allowed = slot <= pos[:, None]
    if not full:
        allowed &= slot >= pos[:, None] - (cfg.sliding_window - 1)
    scores = jnp.where(allowed[:, None, :], scores, -jnp.inf)
    p = jax.nn.softmax(scores, axis=-1)
    o = jnp.einsum("bhs,bshd->bhd", p, vals)
    o = o * jax.lax.rsqrt(jnp.mean(o * o, axis=-1, keepdims=True) + cfg.rms_eps)
    o = _r16(o * jax.nn.sigmoid(jnp.asarray(g, F32).reshape(B, hq, D)))
    return o.reshape(B, hq * D), k_cache, v_cache


# ---- standalone pallas_call (tests, benchmarks) --------------------------------------------


def make_attention_call(cfg: Config, batch: int, context: int, *, tp=8, reps=1, interpret=False):
    """Grid-less `pallas_call` around `attention_layer` for one rank (no collectives).

    Returns `f(layer [1] i32, pos [B] i32, q, kv, g, rope, k_cache, v_cache) -> (o, k_cache,
    v_cache)`; the caches are donated (`input_output_aliases`). `reps > 1` repeats the layer
    in-kernel for timing (idempotent: the same rows are re-written).
    """
    lanes = cache_lanes(cfg, tp)
    hq, D = cfg.heads // tp, cfg.head_dim
    cache_shape = (None, batch, context, lanes)

    def kernel(
        layer_ref, pos_ref, q_ref, kv_ref, g_ref, rope_ref, k_in, v_in, o_ref, k_hbm, v_hbm, *ws
    ):
        del k_in, v_in  # aliased to the k_hbm / v_hbm outputs (separate copies in interpret mode)

        def body(_, carry):
            o_ref[...] = attention_layer(
                cfg,
                batch,
                layer_ref[0],
                pos_ref,
                q_ref,
                kv_ref,
                g_ref,
                rope_ref,
                k_hbm,
                v_hbm,
                ws,
                tp=tp,
            )
            return carry

        jax.lax.fori_loop(0, reps, body, 0)

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pl.ANY)

    def call(layer, pos, q, kv, g, rope, k_cache, v_cache):
        assert k_cache.shape[1:] == cache_shape[1:], (k_cache.shape, cache_shape)
        return pl.pallas_call(
            kernel,
            out_shape=(
                jax.ShapeDtypeStruct((batch, hq * D), F32),
                jax.ShapeDtypeStruct(k_cache.shape, BF16),
                jax.ShapeDtypeStruct(v_cache.shape, BF16),
            ),
            in_specs=[smem, smem, vmem, vmem, vmem, vmem, hbm, hbm],
            out_specs=(vmem, hbm, hbm),
            scratch_shapes=list(scratch_shapes(cfg, batch, tp)),
            input_output_aliases={6: 1, 7: 2},
            interpret=interpret,
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=32 << 20),
        )(layer, pos, q, kv, g, rope, k_cache, v_cache)

    return jax.jit(call, donate_argnums=(6, 7))
