"""DSpark (`mtp.0..2`) 3-layer sharded speculative drafter and target block verification.

Implements the exact DeepSeek-V4.1-Flash DSpark architecture (`REF/model.py:1013-1283`)
on sharded TPU v7x / CPU meshes:
- 3 MoE layers (`mtp.0`, `mtp.1`, `mtp.2`) with 64 routed experts (`EP=8` across lanes) + 1 shared expert
- Shared `embed.weight` and `head.weight` from the target model's `sharded_weights`
- `mtp.0.main_proj` + `mtp.0.main_norm` projecting concatenated target hidden states `[L0, L13, L26, L39]` (`[B, S, 4*dim] -> [B, S, dim]`)
- `DSparkAttention` writing `main_kv` into `dspark_window_kv` (`sliding_window=512`) and attending over `[window_kv, draft_kv]`
- `mtp.2.markov_head` sequential autoregressive chain over `dspark_block_size=5` draft slots + `mtp.2.confidence_head`
- Target block verification (`q = 1 + k <= 6` tokens) with O(1) rollback of `comp_kv_state` / `comp_score_state` via `comp_kv_ring` / `comp_score_ring`, guaranteeing greedy token-for-token identity (`T_dspark == T_base`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np

from deepseek_v41.config import DSV41Config
from deepseek_v41.xla_decode import (
    act_quant,
    apply_rotary_emb,
    fp8_linear,
    hc_mixes,
    hc_post,
    hc_pre,
    moe_gate,
    replicate_to_mesh,
    rms_norm,
    run_shared_expert_fp8,
    run_single_expert_fp4,
    sparse_attn,
)


@dataclass
class AcceptanceStats:
    """Tracks speculative decoding acceptance counts across `0..dspark_block_size`."""

    max_draft_tokens: int = 5
    histogram: list[int] = field(init=False)
    total_verify_steps: int = 0
    total_accepted_draft_tokens: int = 0
    total_emitted_tokens: int = 0

    def __post_init__(self) -> None:
        self.histogram = [0 for _ in range(self.max_draft_tokens + 1)]

    def record(self, num_accepted: int) -> None:
        idx = max(0, min(int(num_accepted), self.max_draft_tokens))
        self.histogram[idx] += 1
        self.total_verify_steps += 1
        self.total_accepted_draft_tokens += int(num_accepted)
        self.total_emitted_tokens += int(num_accepted) + 1

    @property
    def mean_accepted_length(self) -> float:
        if self.total_verify_steps == 0:
            return 0.0
        return float(self.total_accepted_draft_tokens) / float(self.total_verify_steps)

    @property
    def mean_tokens_per_step(self) -> float:
        if self.total_verify_steps == 0:
            return 0.0
        return float(self.total_emitted_tokens) / float(self.total_verify_steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_draft_tokens": self.max_draft_tokens,
            "histogram": list(self.histogram),
            "total_verify_steps": self.total_verify_steps,
            "total_accepted_draft_tokens": self.total_accepted_draft_tokens,
            "total_emitted_tokens": self.total_emitted_tokens,
            "mean_accepted_length": self.mean_accepted_length,
            "mean_tokens_per_step": self.mean_tokens_per_step,
        }


@jax.jit
def _jitted_rollback_with_rings(
    comp_kv_state: jax.Array,
    comp_score_state: jax.Array,
    comp_kv_ring: jax.Array,
    comp_score_ring: jax.Array,
    tok_hist: jax.Array,
    p_commit_arr: jax.Array,
    end_pos_arr: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    ring_len = comp_kv_ring.shape[2]
    is_even = (p_commit_arr % 2) == 0
    r0 = jnp.where(is_even, p_commit_arr % ring_len, (p_commit_arr - 1) % ring_len)
    r1 = jnp.where(is_even, jnp.maximum(p_commit_arr - 1, 0) % ring_len, p_commit_arr % ring_len)
    kv0 = jnp.take(comp_kv_ring, r0, axis=2)
    sc0 = jnp.take(comp_score_ring, r0, axis=2)
    kv1 = jnp.take(comp_kv_ring, r1, axis=2)
    sc1 = jnp.take(comp_score_ring, r1, axis=2)
    comp_kv_state = comp_kv_state.at[:, :, 0, :].set(kv0)
    comp_score_state = comp_score_state.at[:, :, 0, :].set(sc0)
    has_odd_history = p_commit_arr >= 1
    comp_kv_state = jnp.where(has_odd_history, comp_kv_state.at[:, :, 1, :].set(kv1), comp_kv_state)
    comp_score_state = jnp.where(has_odd_history, comp_score_state.at[:, :, 1, :].set(sc1), comp_score_state)
    seq_iota = jnp.arange(tok_hist.shape[1], dtype=jnp.int32)[None, :]
    clear_mask = (seq_iota > p_commit_arr) & (seq_iota < end_pos_arr)
    tok_hist = jnp.where(clear_mask, jnp.int32(2), tok_hist)
    return comp_kv_state, comp_score_state, tok_hist


@partial(jax.jit, static_argnames=("q",))
def _jitted_restore_pre_window_kv(
    pre_window_kv: jax.Array,
    post_window_kv: jax.Array,
    start_pos_arr: jax.Array,
    n_acc_arr: jax.Array,
    q: int,
) -> jax.Array:
    win = post_window_kv.shape[2]
    w_kv = post_window_kv
    for k in range(1, q):
        is_rej = jnp.int32(k) > n_acc_arr
        pos_k = start_pos_arr + k
        slot_w = pos_k % win
        old_w = jnp.take(pre_window_kv, slot_w, axis=2)
        cur_w = jnp.take(w_kv, slot_w, axis=2)
        w_kv = w_kv.at[:, :, slot_w, :].set(jnp.where(is_rej, old_w, cur_w))
    return w_kv


@jax.jit
def _jitted_rollback_tok_hist_only(
    tok_hist: jax.Array,
    p_commit_arr: jax.Array,
    end_pos_arr: jax.Array,
) -> jax.Array:
    seq_iota = jnp.arange(tok_hist.shape[1], dtype=jnp.int32)[None, :]
    clear_mask = (seq_iota > p_commit_arr) & (seq_iota < end_pos_arr)
    return jnp.where(clear_mask, jnp.int32(2), tok_hist)


def rollback_target_cache(
    cfg: DSV41Config,
    cache: dict[str, jax.Array],
    start_pos: int,
    num_draft_tokens: int,
    num_accepted: int,
    *,
    pre_cache: dict[str, jax.Array] | None = None,
) -> dict[str, jax.Array]:
    """Roll back target KV/compressor state to `p_commit = start_pos + num_accepted` in O(1).

    When verifying a block of `q = 1 + num_draft_tokens` tokens at `start_pos .. start_pos + q - 1`,
    if `num_accepted < num_draft_tokens`, positions `p_commit + 1 .. start_pos + q - 1` were rejected.
    This function restores circular `window_kv` (when `pre_cache` is supplied), `comp_kv_state`,
    and `comp_score_state` from the 8-slot ring buffers `comp_kv_ring` and `comp_score_ring`,
    and clears rejected `token_history` slots so subsequent steps from `p_commit + 1` are
    bit-identical to single-token greedy execution even past `sliding_window`.
    """
    q = 1 + int(num_draft_tokens)
    n_acc = int(num_accepted)
    if n_acc >= num_draft_tokens:
        return cache

    p_commit = int(start_pos) + n_acc
    end_pos = int(start_pos) + q
    start_pos_arr = jnp.asarray(int(start_pos), dtype=jnp.int32)
    n_acc_arr = jnp.asarray(n_acc, dtype=jnp.int32)
    p_commit_arr = jnp.asarray(p_commit, dtype=jnp.int32)
    end_pos_arr = jnp.asarray(end_pos, dtype=jnp.int32)
    new_cache = dict(cache)

    if pre_cache is not None and "window_kv" in pre_cache and "window_kv" in cache:
        new_cache["window_kv"] = _jitted_restore_pre_window_kv(
            pre_cache["window_kv"],
            cache["window_kv"],
            start_pos_arr,
            n_acc_arr,
            q,
        )

    comp_kv_ring = cache.get("comp_kv_ring")
    comp_score_ring = cache.get("comp_score_ring")
    if comp_kv_ring is not None and comp_score_ring is not None:
        comp_kv_state, comp_score_state, tok_hist = _jitted_rollback_with_rings(
            cache["comp_kv_state"],
            cache["comp_score_state"],
            comp_kv_ring,
            comp_score_ring,
            cache["token_history"],
            p_commit_arr,
            end_pos_arr,
        )
        new_cache["comp_kv_state"] = comp_kv_state
        new_cache["comp_score_state"] = comp_score_state
        new_cache["token_history"] = tok_hist
    elif p_commit + 1 < end_pos:
        new_cache["token_history"] = _jitted_rollback_tok_hist_only(
            cache["token_history"],
            p_commit_arr,
            end_pos_arr,
        )

    return new_cache


def _embed_lookup_sharded(
    tokens: jax.Array,
    embed_w: jax.Array,
    cfg: DSV41Config,
    num_devices: int,
    axis_name: str,
) -> jax.Array:
    """Lookup token embeddings `[B, S, dim]` from vocab-sharded or replicated `embed.weight`."""
    if embed_w.shape[0] == cfg.padded_vocab_size // num_devices:
        rank = jax.lax.axis_index(axis_name)
        v_per_rank = embed_w.shape[0]
        v_start = rank * v_per_rank
        local_idx = tokens - v_start
        valid = (local_idx >= 0) & (local_idx < v_per_rank)
        safe_idx = jnp.clip(local_idx, 0, v_per_rank - 1)
        h0_local = jnp.where(valid[..., None], embed_w[safe_idx], jnp.bfloat16(0.0)).astype(jnp.float32)
        return jax.lax.psum(h0_local, axis_name).astype(jnp.bfloat16)
    return embed_w[tokens].astype(jnp.bfloat16)


def _dspark_attn_update_main_kv(
    cfg: DSV41Config,
    w_dict: Mapping[str, jax.Array],
    dspark_window_kv_all: jax.Array,
    main_x: jax.Array,
    start_pos: jax.Array,
    actual_len: jax.Array | None = None,
) -> jax.Array:
    """Write `main_kv` for `main_x` (`[B, s_acc, dim]`) at `start_pos .. start_pos + actual_len - 1` across `mtp.0..2`."""
    bsz, seqlen, _ = main_x.shape
    win = cfg.sliding_window
    cos_swa_all = w_dict["rope.swa.cos"]
    sin_swa_all = w_dict["rope.swa.sin"]
    valid_len = jnp.int32(seqlen) if actual_len is None else actual_len.astype(jnp.int32)

    pos_vec = start_pos + jnp.arange(seqlen, dtype=jnp.int32)
    cos_m = cos_swa_all[pos_vec][None, :, :]
    sin_m = sin_swa_all[pos_vec][None, :, :]

    for stage_id in range(cfg.n_mtp_layers):
        prefix = f"mtp.{stage_id}."
        ap = prefix + "attn."
        main_kv = rms_norm(
            fp8_linear(main_x, w_dict[ap + "wkv.weight"], w_dict[ap + "wkv.scale"]),
            w_dict[ap + "kv_norm.weight"],
            cfg.rms_norm_eps,
        )
        main_kv = apply_rotary_emb(main_kv, cos_m, sin_m, cfg.qk_rope_head_dim, inverse=False)
        main_kv = act_quant(main_kv, 32, inplace=True)

        win_stage = dspark_window_kv_all[stage_id]
        for s_i in range(seqlen):
            slot = (start_pos + s_i) % win
            cur_val = win_stage[:, slot]
            new_val = jnp.where(jnp.int32(s_i) < valid_len, main_kv[:, s_i], cur_val)
            win_stage = win_stage.at[:, slot].set(new_val)
        dspark_window_kv_all = dspark_window_kv_all.at[stage_id].set(win_stage)

    return dspark_window_kv_all


def _dspark_draft_local(
    cfg: DSV41Config,
    w_dict: Mapping[str, jax.Array],
    dspark_window_kv_all: jax.Array,
    last_token_ids: jax.Array,
    accepted_main_hidden: jax.Array,
    start_pos: jax.Array,
    actual_s_acc: jax.Array | None = None,
    *,
    num_devices: int,
    axis_name: str,
    sinkhorn_iters: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Per-rank `shard_map` body for DSpark `mtp.0..2` drafting (`REF/model.py:1128-1283`)."""
    bsz, s_acc, _ = accepted_main_hidden.shape
    valid_s_acc = jnp.int32(s_acc) if actual_s_acc is None else actual_s_acc.astype(jnp.int32)
    block_size = cfg.dspark_block_size
    win = cfg.sliding_window
    ep_lanes = min(8, num_devices)
    shared_tp = min(8, num_devices, cfg.moe_inter_dim // 32)
    rank = jax.lax.axis_index(axis_name)

    # 1. Project target main_hidden [B, s_acc, 4*dim] -> main_x [B, s_acc, dim] and update dspark_window_kv
    main_x = rms_norm(
        fp8_linear(
            accepted_main_hidden,
            w_dict["mtp.0.main_proj.weight"],
            w_dict["mtp.0.main_proj.scale"],
        ),
        w_dict["mtp.0.main_norm.weight"],
        cfg.rms_norm_eps,
    )
    dspark_window_kv_all = _dspark_attn_update_main_kv(
        cfg, w_dict, dspark_window_kv_all, main_x, start_pos, valid_s_acc
    )

    # 2. Construct draft input tokens [B, block_size]: slot 0 is last_token_ids, slots 1..k-1 are noise_token_id
    draft_ids = jnp.full((bsz, block_size), cfg.dspark_noise_token_id, dtype=jnp.int32)
    draft_ids = draft_ids.at[:, 0].set(last_token_ids.reshape(bsz))
    h0 = _embed_lookup_sharded(draft_ids, w_dict["embed.weight"], cfg, num_devices, axis_name)
    h = jnp.repeat(h0[:, :, None, :], cfg.hc_mult, axis=2)
    pre_mix = jnp.zeros((bsz, block_size, cfg.hc_mult), dtype=jnp.float32).at[:, :, 0].set(1.0)

    draft_start = start_pos + valid_s_acc
    draft_pos_vec = draft_start + jnp.arange(block_size, dtype=jnp.int32)
    cos_swa_all = w_dict["rope.swa.cos"]
    sin_swa_all = w_dict["rope.swa.sin"]
    cos_b = cos_swa_all[draft_pos_vec][None, :, :]
    sin_b = sin_swa_all[draft_pos_vec][None, :, :]

    valid_ring = jnp.minimum(jnp.int32(win), draft_start)
    ring_iota = jnp.arange(win, dtype=jnp.int32)
    ring_idxs = jnp.where(ring_iota < valid_ring, ring_iota, jnp.int32(-1))
    draft_idxs = win + jnp.arange(block_size, dtype=jnp.int32)
    idxs_1d = jnp.concatenate([ring_idxs, draft_idxs], axis=0)
    topk_idxs = jnp.broadcast_to(idxs_1d[None, None, :], (bsz, block_size, idxs_1d.shape[0]))

    # 3. Run 3 DSpark layers (mtp.0 .. mtp.2)
    for stage_id in range(cfg.n_mtp_layers):
        prefix = f"mtp.{stage_id}."
        ap = prefix + "attn."
        fp = prefix + "ffn."

        residual = h
        a_pre, a_post, a_comb = hc_mixes(
            h,
            w_dict[prefix + "hc_attn_fn"],
            w_dict[prefix + "hc_attn_scale"],
            w_dict[prefix + "hc_attn_base"],
            hc_mult=cfg.hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            hc_eps=cfg.hc_eps,
            norm_eps=cfg.rms_norm_eps,
        )
        u = rms_norm(hc_pre(h, pre_mix), w_dict[prefix + "attn_norm.weight"], cfg.rms_norm_eps)

        qr = rms_norm(
            fp8_linear(u, w_dict[ap + "wq_a.weight"], w_dict[ap + "wq_a.scale"]),
            w_dict[ap + "q_norm.weight"],
            cfg.rms_norm_eps,
        )
        q_raw = fp8_linear(qr, w_dict[ap + "wq_b.weight"], w_dict[ap + "wq_b.scale"])
        n_local_heads = cfg.n_heads // num_devices
        q = q_raw.reshape(bsz, block_size, n_local_heads, cfg.head_dim)
        q = apply_rotary_emb(q, cos_b, sin_b, cfg.qk_rope_head_dim, inverse=False)

        kv_raw = rms_norm(
            fp8_linear(u, w_dict[ap + "wkv.weight"], w_dict[ap + "wkv.scale"]),
            w_dict[ap + "kv_norm.weight"],
            cfg.rms_norm_eps,
        )
        kv_rot = apply_rotary_emb(kv_raw, cos_b, sin_b, cfg.qk_rope_head_dim, inverse=False)
        kv_quant = act_quant(kv_rot, 32, inplace=True)

        kv_all = jnp.concatenate([dspark_window_kv_all[stage_id], kv_quant], axis=1)
        o = sparse_attn(q, kv_all, w_dict[ap + "attn_sink"], topk_idxs, cfg.head_dim**-0.5)
        o = apply_rotary_emb(o, cos_b, sin_b, cfg.qk_rope_head_dim, inverse=True)

        wo_a = w_dict[ap + "wo_a.weight"]
        if num_devices >= cfg.o_groups:
            ranks_per_group = num_devices // cfg.o_groups
            o_flat = o.reshape(bsz, block_size, -1).astype(jnp.float32)
            o_group_partial = jnp.einsum(
                "bsd,rd->bsr", o_flat, wo_a.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST
            )
            if ranks_per_group > 1:
                gathered_partials = jax.lax.all_gather(o_group_partial, axis_name, axis=0)
                grp_base = (rank // ranks_per_group) * ranks_per_group
                grp_partials = jax.lax.dynamic_slice_in_dim(gathered_partials, grp_base, ranks_per_group, axis=0)
                o_group_full = jnp.sum(grp_partials, axis=0).astype(jnp.bfloat16)
                sub_idx = rank % ranks_per_group
                sub_width = cfg.o_lora_rank // ranks_per_group
                o_mid = jax.lax.dynamic_slice_in_dim(o_group_full, sub_idx * sub_width, sub_width, axis=-1)
            else:
                o_mid = o_group_partial.astype(jnp.bfloat16)
        else:
            groups_per_rank = cfg.o_groups // num_devices
            o_grp = o.reshape(bsz, block_size, groups_per_rank, -1).astype(jnp.float32)
            o_mid = jnp.einsum(
                "bsgd,grd->bsgr", o_grp, wo_a.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST
            ).reshape(bsz, block_size, -1).astype(jnp.bfloat16)

        attn_out_f32 = fp8_linear(
            o_mid, w_dict[ap + "wo_b.weight"], w_dict[ap + "wo_b.scale"], out_dtype=jnp.float32
        )
        attn_out = jax.lax.psum(attn_out_f32, axis_name).astype(jnp.bfloat16)
        h = hc_post(attn_out, residual, a_post, a_comb)

        residual = h
        f_pre, f_post, f_comb = hc_mixes(
            h,
            w_dict[prefix + "hc_ffn_fn"],
            w_dict[prefix + "hc_ffn_scale"],
            w_dict[prefix + "hc_ffn_base"],
            hc_mult=cfg.hc_mult,
            sinkhorn_iters=sinkhorn_iters,
            hc_eps=cfg.hc_eps,
            norm_eps=cfg.rms_norm_eps,
        )
        v = rms_norm(hc_pre(h, a_pre), w_dict[prefix + "ffn_norm.weight"], cfg.rms_norm_eps)
        v_flat = v.reshape(bsz * block_size, cfg.dim)

        top_w, top_idx = moe_gate(
            v_flat,
            w_dict[fp + "gate.weight"],
            w_dict[fp + "gate.bias"],
            cfg.dspark_n_activated_experts,
            cfg.route_scale,
        )
        sort_perm = jnp.argsort(top_idx, axis=-1)
        top_idx = jnp.take_along_axis(top_idx, sort_perm, axis=-1)
        top_w = jnp.take_along_axis(top_w, sort_perm, axis=-1)

        w1_u32 = w_dict[fp + "experts.w1.u32"]
        w1_s = w_dict[fp + "experts.w1.scale"]
        w3_u32 = w_dict[fp + "experts.w3.u32"]
        w3_s = w_dict[fp + "experts.w3.scale"]
        w2_u32 = w_dict[fp + "experts.w2.u32"]
        w2_s = w_dict[fp + "experts.w2.scale"]
        n_local_exp = w1_u32.shape[0]
        lane = rank % ep_lanes
        exp_start = lane * n_local_exp

        routed_out_dtype = jnp.bfloat16 if num_devices <= 8 else jnp.float32
        e_global_all = top_idx - exp_start
        active_mask_all = (e_global_all >= 0) & (e_global_all < n_local_exp)
        safe_local_all = jnp.clip(e_global_all, 0, n_local_exp - 1)

        def _scan_dspark_slot(
            routed_acc: jax.Array, slot_inputs: tuple[jax.Array, jax.Array, jax.Array]
        ) -> tuple[jax.Array, None]:
            safe_local, active_mask, w_sel = slot_inputs
            for t_i in range(bsz * block_size):
                et = safe_local[t_i]

                def _run_active(et_val: jax.Array) -> jax.Array:
                    return run_single_expert_fp4(
                        v_flat[t_i : t_i + 1],
                        w_sel[t_i : t_i + 1],
                        w1_u32[et_val],
                        w1_s[et_val],
                        w3_u32[et_val],
                        w3_s[et_val],
                        w2_u32[et_val],
                        w2_s[et_val],
                        cfg.swiglu_limit,
                        out_dtype=routed_out_dtype,
                    )[0].astype(jnp.float32)

                def _skip_inactive(_: jax.Array) -> jax.Array:
                    return jnp.zeros((cfg.dim,), dtype=jnp.float32)

                out_t = jax.lax.cond(active_mask[t_i], _run_active, _skip_inactive, et)
                routed_acc = routed_acc.at[t_i].add(out_t)
            return routed_acc, None

        routed_part, _ = jax.lax.scan(
            _scan_dspark_slot,
            jnp.zeros((bsz * block_size, cfg.dim), dtype=jnp.float32),
            (safe_local_all.T, active_mask_all.T, top_w.T),
        )
        routed_full = jax.lax.psum(routed_part, axis_name)

        sp = fp + "shared_experts."
        shared_part = run_shared_expert_fp8(
            v_flat,
            w_dict[sp + "w1.weight"],
            w_dict[sp + "w1.scale"],
            w_dict[sp + "w3.weight"],
            w_dict[sp + "w3.scale"],
            w_dict[sp + "w2.weight"],
            w_dict[sp + "w2.scale"],
            cfg.swiglu_limit,
            out_dtype=jnp.float32,
        ).astype(jnp.float32)
        shared_rep = num_devices // shared_tp
        shared_full = (jax.lax.psum(shared_part, axis_name) / jnp.float32(shared_rep)).astype(jnp.bfloat16).astype(jnp.float32)

        ffn_out = (routed_full + shared_full).astype(jnp.bfloat16).reshape(bsz, block_size, cfg.dim)
        h = hc_post(ffn_out, residual, f_post, f_comb)
        pre_mix = f_pre

    # 4. Output head + sequential Markov chain + confidence head (`REF/model.py:1137-1156`)
    last_prefix = f"mtp.{cfg.n_mtp_layers - 1}."
    x_collapsed = hc_pre(h, pre_mix)
    normed = rms_norm(x_collapsed, w_dict[last_prefix + "norm.weight"], cfg.rms_norm_eps)

    head_w = w_dict["head.weight"].astype(jnp.float32)
    logits_local = jnp.einsum(
        "bsd,vd->bsv", normed.astype(jnp.float32), head_w, precision=jax.lax.Precision.HIGHEST
    )
    if head_w.shape[0] == cfg.padded_vocab_size // num_devices:
        logits_full = jax.lax.all_gather(logits_local, axis_name, axis=-1, tiled=True)
    else:
        logits_full = logits_local
    base_logits = logits_full[:, :, : cfg.vocab_size]

    markov_emb_w = w_dict[last_prefix + "markov_head.embed.weight"].astype(jnp.bfloat16)
    markov_head_w = w_dict[last_prefix + "markov_head.head.weight"].astype(jnp.float32)
    conf_proj_w = w_dict[last_prefix + "confidence_head.proj.weight"].astype(jnp.float32)

    out_tokens = [last_token_ids.reshape(bsz)]
    updated_logits = []
    markov_embeds = []
    for i in range(block_size):
        cur_tok = out_tokens[i]
        m_emb = markov_emb_w[cur_tok]
        bias_local = jnp.einsum(
            "br,vr->bv", m_emb.astype(jnp.float32), markov_head_w, precision=jax.lax.Precision.HIGHEST
        )
        if markov_head_w.shape[0] == cfg.padded_vocab_size // num_devices:
            bias_full = jax.lax.all_gather(bias_local, axis_name, axis=-1, tiled=True)
        else:
            bias_full = bias_local
        logits_bias = bias_full[:, : cfg.vocab_size]
        l_i = base_logits[:, i, :] + logits_bias
        next_tok = jnp.argmax(l_i, axis=-1).astype(jnp.int32)
        out_tokens.append(next_tok)
        updated_logits.append(l_i)
        markov_embeds.append(m_emb)

    output_ids = jnp.stack(out_tokens, axis=1)  # [B, block_size + 1]
    logits_out = jnp.stack(updated_logits, axis=1)  # [B, block_size, vocab_size]
    markov_embed_all = jnp.stack(markov_embeds, axis=1)  # [B, block_size, markov_rank]
    conf_in = jnp.concatenate([x_collapsed, markov_embed_all], axis=-1).astype(jnp.float32)
    confidence = jnp.einsum(
        "bsd,od->bso", conf_in, conf_proj_w, precision=jax.lax.Precision.HIGHEST
    ).squeeze(-1)

    return output_ids, logits_out, confidence, dspark_window_kv_all


class DSparkDrafter:
    """Compiled 3-layer DSpark speculative drafter sharing `embed`/`head` with the target model."""

    def __init__(
        self,
        cfg: DSV41Config,
        mesh: Mesh,
        *,
        sinkhorn_iters: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.mesh = mesh
        self.num_devices = mesh.size
        self.sinkhorn_iters = cfg.hc_sinkhorn_iters if sinkhorn_iters is None else int(sinkhorn_iters)
        self._compiled_prefill_fns: dict[int, Callable] = {}
        self._compiled_draft_fns: dict[int, Callable] = {}

    def _get_prefill_fn(self, seqlen: int) -> Callable:
        if seqlen not in self._compiled_prefill_fns:
            cfg = self.cfg

            def _local_prefill(sharded_weights, dspark_window_kv_all, main_hidden, start_pos):
                w_dict = jax.tree.map(lambda x: x[0], sharded_weights)
                main_x = rms_norm(
                    fp8_linear(
                        main_hidden,
                        w_dict["mtp.0.main_proj.weight"],
                        w_dict["mtp.0.main_proj.scale"],
                    ),
                    w_dict["mtp.0.main_norm.weight"],
                    cfg.rms_norm_eps,
                )
                return _dspark_attn_update_main_kv(cfg, w_dict, dspark_window_kv_all, main_x, start_pos)

            self._compiled_prefill_fns[seqlen] = jax.jit(
                jax.shard_map(
                    _local_prefill,
                    mesh=self.mesh,
                    in_specs=(P("tp"), P(), P(), P()),
                    out_specs=P(),
                    check_vma=False,
                )
            )
        return self._compiled_prefill_fns[seqlen]

    def _get_draft_fn(self, max_s_acc: int) -> Callable:
        if max_s_acc not in self._compiled_draft_fns:
            cfg = self.cfg
            num_devices = self.num_devices
            s_iters = self.sinkhorn_iters

            def _local_draft(
                sharded_weights,
                dspark_window_kv_all,
                last_token_ids,
                accepted_main_hidden,
                start_pos,
                actual_s_acc,
            ):
                w_dict = jax.tree.map(lambda x: x[0], sharded_weights)
                return _dspark_draft_local(
                    cfg,
                    w_dict,
                    dspark_window_kv_all,
                    last_token_ids,
                    accepted_main_hidden,
                    start_pos,
                    actual_s_acc,
                    num_devices=num_devices,
                    axis_name="tp",
                    sinkhorn_iters=s_iters,
                )

            self._compiled_draft_fns[max_s_acc] = jax.jit(
                jax.shard_map(
                    _local_draft,
                    mesh=self.mesh,
                    in_specs=(P("tp"), P(), P(), P(), P(), P()),
                    out_specs=(P(), P(), P(), P()),
                    check_vma=False,
                )
            )
        return self._compiled_draft_fns[max_s_acc]

    def prefill_main_hidden(
        self,
        sharded_weights: Mapping[str, jax.Array],
        cache: dict[str, jax.Array],
        main_hidden: jax.Array,
        start_pos: int | jax.Array,
    ) -> dict[str, jax.Array]:
        """Update `cache['dspark_window_kv']` from prefill `main_hidden` (`[B, S, 4*dim]`)."""
        if self.cfg.n_mtp_layers <= 0 or self.cfg.dspark_block_size <= 0:
            return cache
        mh = jnp.asarray(main_hidden, dtype=jnp.bfloat16)
        if mh.ndim == 2:
            mh = mh[:, None, :]
        seqlen = mh.shape[1]
        fn = self._get_prefill_fn(seqlen)
        new_dspark_kv = fn(
            sharded_weights,
            cache["dspark_window_kv"],
            replicate_to_mesh(mh, self.mesh),
            replicate_to_mesh(jnp.asarray(start_pos, dtype=jnp.int32), self.mesh),
        )
        return {**cache, "dspark_window_kv": new_dspark_kv}

    def draft(
        self,
        sharded_weights: Mapping[str, jax.Array],
        cache: dict[str, jax.Array],
        last_token_ids: jax.Array,
        accepted_main_hidden: jax.Array,
        start_pos: int | jax.Array,
        *,
        confidence_threshold: float | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        """Run one DSpark step (`mtp.0..2`) and return `(candidate_block, logits, confidence, new_cache)`."""
        mh = jnp.asarray(accepted_main_hidden, dtype=jnp.bfloat16)
        if mh.ndim == 2:
            mh = mh[:, None, :]
        s_acc = int(mh.shape[1])
        max_s_acc = max(s_acc, 1 + int(self.cfg.dspark_block_size))
        if s_acc < max_s_acc:
            mh_padded = jnp.pad(mh, ((0, 0), (0, max_s_acc - s_acc), (0, 0)))
        else:
            mh_padded = mh
        last_ids = jnp.asarray(last_token_ids, dtype=jnp.int32).reshape(mh.shape[0])
        fn = self._get_draft_fn(max_s_acc)
        output_ids, logits_out, confidence, new_dspark_kv = fn(
            sharded_weights,
            cache["dspark_window_kv"],
            replicate_to_mesh(last_ids, self.mesh),
            replicate_to_mesh(mh_padded, self.mesh),
            replicate_to_mesh(jnp.asarray(start_pos, dtype=jnp.int32), self.mesh),
            replicate_to_mesh(jnp.asarray(s_acc, dtype=jnp.int32), self.mesh),
        )
        new_cache = {**cache, "dspark_window_kv": new_dspark_kv}

        if confidence_threshold is not None and output_ids.shape[0] == 1:
            conf_np = np.asarray(jax.nn.sigmoid(confidence[0]))
            k_keep = self.cfg.dspark_block_size
            for i in range(self.cfg.dspark_block_size):
                if float(conf_np[i]) < float(confidence_threshold):
                    k_keep = max(1, i + 1)
                    break
            output_ids = output_ids[:, : 1 + k_keep]
            logits_out = logits_out[:, :k_keep, :]
            confidence = confidence[:, :k_keep]

        return output_ids, logits_out, confidence, new_cache


def verify_and_commit_block(
    target_step_fn: Callable,
    sharded_weights: Mapping[str, jax.Array],
    cache: dict[str, jax.Array],
    candidate_block: jax.Array,
    start_pos: int,
    cfg: DSV41Config,
    *,
    force_num_accepted: int | None = None,
    stats: AcceptanceStats | None = None,
) -> dict[str, Any]:
    """Verify `candidate_block` (`[1, 1 + k]`) with `target_step_fn` and roll back rejected KV/compressor state.

    Guarantees greedy token-for-token equivalence (`T_dspark == T_base`):
    - Runs `target_step_fn(sharded_weights, cache, candidate_block, start_pos)` in a single `q = 1 + k` block call.
    - Compares greedy target predictions `argmax(logits[:, i, :])` against `candidate_block[:, i + 1]`.
    - Accepts the longest matching prefix `n_accepted` (`0 <= n_accepted <= k`) plus 1 target bonus token
      `bonus_token = argmax(logits[:, n_accepted, :])`.
    - Rolls back `comp_kv_state`, `comp_score_state`, and `token_history` to `start_pos + n_accepted` via
      `rollback_target_cache`.
    """
    block = jnp.asarray(candidate_block, dtype=jnp.int32)
    if block.ndim == 1:
        block = block[None, :]
    bsz, q = block.shape
    if bsz != 1:
        raise ValueError(f"DSpark verify_and_commit_block requires batch_size == 1, got {bsz}")
    num_draft = q - 1

    logits_q, mh_q, post_cache = target_step_fn(
        sharded_weights,
        cache,
        block,
        jnp.asarray(start_pos, dtype=jnp.int32),
    )
    preds = np.asarray(jnp.argmax(logits_q, axis=-1))[0]  # [q]
    draft_np = np.asarray(block)[0]  # [q]

    n_accepted = 0
    for i in range(num_draft):
        if int(preds[i]) == int(draft_np[i + 1]):
            n_accepted += 1
        else:
            break
    if force_num_accepted is not None:
        n_accepted = max(0, min(n_accepted, int(force_num_accepted)))

    committed_cache = rollback_target_cache(
        cfg,
        post_cache,
        start_pos=int(start_pos),
        num_draft_tokens=num_draft,
        num_accepted=n_accepted,
        pre_cache=cache,
    )
    bonus_token = int(preds[n_accepted])
    emitted_tokens = [int(draft_np[i]) for i in range(1, n_accepted + 1)] + [bonus_token]
    accepted_main_hidden = mh_q[:, : n_accepted + 1, :]

    if stats is not None:
        stats.record(n_accepted)

    return {
        "num_draft_tokens": num_draft,
        "num_accepted": n_accepted,
        "bonus_token": bonus_token,
        "emitted_tokens": emitted_tokens,
        "next_start_pos": int(start_pos) + n_accepted + 1,
        "accepted_main_hidden": accepted_main_hidden,
        "logits": logits_q,
        "cache": committed_cache,
    }
