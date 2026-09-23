"""DFlash2 block-diffusion speculative decoding for Qwen3.8-27B (TP8).

The draft predicts a block of eight tokens in one pass, conditioned on the
target model's hidden-state taps for the committed context, reusing the
target's token embedding and LM head. Verification is the target model's block
kernel (``qwen.decode_megakernel.make_verify_block``), so greedy speculative
decoding commits exactly the target's greedy tokens (lossless). Included: the
draft reference equations, checkpoint loading, the packed per-rank kernel
layout (with the channel permutation the fused kernel convolves in), the fused
Pallas draft kernel, and the fused speculation round / device-side decode loop
harness (``make_scan_fn``).
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu



# --------------------------------------------------------------------------
# draft model equations and checkpoint loading


F32 = jnp.float32


class DraftConfig:
    def __init__(self, cfg):
        d = cfg["dflash_config"]
        self.hidden = cfg["hidden_size"]
        self.layers = cfg["num_hidden_layers"]
        self.heads = cfg["num_attention_heads"]
        self.kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg["head_dim"]
        self.intermediate = cfg["intermediate_size"]
        self.vocab = cfg["vocab_size"]
        self.eps = cfg["rms_norm_eps"]
        self.theta = cfg["rope_parameters"]["rope_theta"]
        self.sliding_window = cfg["sliding_window"]
        self.block_size = d["block_size"]
        self.conv_kernel = d["conv_kernel_size"]
        self.conv_group = d["conv_group_size"]
        self.mask_token_id = d["mask_token_id"]
        self.selector_rank = d["selector_rank"]
        self.selector_top_k = d["selector_top_k"]
        self.target_layer_ids = d["target_layer_ids"]


def rms_norm(x, w, eps):
    """Qwen3RMSNorm: fp32 normalize, cast back, then scale (matches torch order)."""
    v = jnp.mean(x.astype(F32) ** 2, axis=-1, keepdims=True)
    return (x.astype(F32) * jax.lax.rsqrt(v + eps)).astype(x.dtype) * w


def _convolve(hidden, dynamic, base, group_size):
    """hidden [L, H]; dynamic [L, K, G]; base [K, H] -> [L, H]."""
    L, H = hidden.shape
    G = H // group_size
    blocks = hidden.reshape(L, G, group_size)
    out = jnp.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = blocks if offset == 0 else jnp.pad(blocks[:-offset], ((offset, 0), (0, 0), (0, 0)))
        kernel = base[offset].reshape(1, G, group_size)
        out = out + kernel * values + dynamic[:, offset, :, None] * values
    return out.reshape(L, H)


def conv_prepare(hidden, cw, cfg):
    """Pre-attention/MLP conv; returns (convolved, dynamic_for_finish)."""
    groups = hidden.shape[-1] // cfg.conv_group
    K = cfg.conv_kernel
    dynamic = jnp.dot(hidden, cw["kernel_projection"].T, preferred_element_type=F32)
    dynamic = dynamic.reshape(*hidden.shape[:-1], 2, K, groups).astype(hidden.dtype)
    return (
        _convolve(hidden, dynamic[:, 0], cw["base_kernel"][0], cfg.conv_group),
        dynamic[:, 1],
    )


def conv_finish(hidden, dynamic, cw, cfg):
    return _convolve(hidden, dynamic, cw["base_kernel"][1], cfg.conv_group)


def rope_tables(theta, head_dim, positions):
    """positions [P] -> (cos, sin) [P, head_dim] bf16 (split-half convention).

    inv_freq exponent is 2i/head_dim (HF Qwen3RotaryEmbedding default).
    """
    half = head_dim // 2
    inv = 1.0 / (theta ** (jnp.arange(half, dtype=F32) * 2.0 / head_dim))
    angles = positions.astype(F32)[:, None] * inv
    return tuple(
        jnp.concatenate((f(angles), f(angles)), axis=-1).astype(jnp.bfloat16)
        for f in (jnp.cos, jnp.sin)
    )


def apply_rope(vec, cos, sin):
    """vec [L, h, d] (full-dim rotary); cos/sin [L, d] -> rotate_half convention."""
    half = cos.shape[-1] // 2
    a, b = vec[..., :half], vec[..., half:]
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return jnp.concatenate(
        (a * cos[..., :half] - b * sin[..., :half], b * cos[..., half:] + a * sin[..., half:]),
        axis=-1,
    ).astype(vec.dtype)


def draft_layer(h, ctx, layer, cfg, cos_blk, sin_blk, cos_all, sin_all, kpos, qpos, kvalid):
    """One decoder layer. h [B, H] block stream; ctx [C, H] context features."""
    B = h.shape[0]
    residual = h
    hn = rms_norm(h, layer["input_layernorm"], cfg.eps)
    hn, dyn = conv_prepare(hn, layer["attention_conv"], cfg)
    q = jnp.dot(hn, layer["q_proj"].T, preferred_element_type=F32).astype(h.dtype)
    q = rms_norm(q.reshape(B, cfg.heads, cfg.head_dim), layer["q_norm"], cfg.eps)
    k_ctx = jnp.dot(ctx, layer["k_proj"].T, preferred_element_type=F32).astype(h.dtype)
    k_noise = jnp.dot(hn, layer["k_proj"].T, preferred_element_type=F32).astype(h.dtype)
    k = jnp.concatenate([k_ctx, k_noise], axis=0)
    k = rms_norm(k.reshape(-1, cfg.kv_heads, cfg.head_dim), layer["k_norm"], cfg.eps)
    v = jnp.concatenate(
        [
            jnp.dot(ctx, layer["v_proj"].T, preferred_element_type=F32).astype(h.dtype),
            jnp.dot(hn, layer["v_proj"].T, preferred_element_type=F32).astype(h.dtype),
        ],
        axis=0,
    ).reshape(-1, cfg.kv_heads, cfg.head_dim)
    q = apply_rope(q, cos_blk, sin_blk)
    k = apply_rope(k, cos_all, sin_all)
    # bidirectional sliding-window attention; keys before key_lo are pads
    grp = cfg.heads // cfg.kv_heads
    qg = q.transpose(1, 0, 2).reshape(cfg.kv_heads, grp, B, cfg.head_dim)
    kt = k.transpose(1, 0, 2).astype(F32)
    scores = jnp.einsum("gibd,gtd->gibt", qg.astype(F32), kt) * cfg.head_dim**-0.5
    dist = jnp.abs(qpos[:, None] - kpos[None, :])
    keep = (dist < cfg.sliding_window) & kvalid[None, :]
    scores = jnp.where(keep[None, None], scores, -jnp.inf)
    probs = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("gibt,gtd->gibd", probs, v.transpose(1, 0, 2).astype(F32))
    out = out.reshape(cfg.heads, B, cfg.head_dim).transpose(1, 0, 2).reshape(B, -1)
    out = jnp.dot(out, layer["o_proj"].T, preferred_element_type=F32).astype(h.dtype)
    out = conv_finish(out, dyn, layer["attention_conv"], cfg)
    h = residual + out
    residual = h
    hn = rms_norm(h, layer["post_attention_layernorm"], cfg.eps)
    hn, dyn = conv_prepare(hn, layer["mlp_conv"], cfg)
    gate = jax.nn.silu(jnp.dot(hn, layer["gate_proj"].T, preferred_element_type=F32))
    up = jnp.dot(hn, layer["up_proj"].T, preferred_element_type=F32)
    out = jnp.dot((gate * up).astype(h.dtype), layer["down_proj"].T, preferred_element_type=F32)
    out = conv_finish(out.astype(h.dtype), dyn, layer["mlp_conv"], cfg)
    return h + out


def draft_forward(params, tgt, anchor_and_masks, feats, ctx_pos, ctx_valid, blk_pos, cfg):
    """One draft round.

    anchor_and_masks: [B] token ids (anchor + MASK tokens); feats: [C, 5H] target
    features at absolute positions ctx_pos [C]; ctx_valid: [C] bool (committed,
    non-pad slots only); blk_pos [B] absolute block positions. Returns the draft
    hidden states at the B block positions [B, H] (post final norm, pre LM head).
    """
    noise = tgt["emb"][anchor_and_masks]  # [B, H]
    ctx = jnp.dot(feats, params["fc"].T, preferred_element_type=F32)
    ctx = rms_norm(ctx.astype(noise.dtype), params["hidden_norm"], cfg.eps)
    all_pos = jnp.concatenate([ctx_pos, blk_pos])
    kvalid = jnp.concatenate([ctx_valid, jnp.ones(len(blk_pos), bool)])
    cos_all, sin_all = rope_tables(cfg.theta, cfg.head_dim, all_pos)
    cos_blk, sin_blk = cos_all[-len(blk_pos) :], sin_all[-len(blk_pos) :]
    h = noise
    for layer in params["layers"]:  # list from pack_draft_for_kernel
        h = draft_layer(
            h, ctx, layer, cfg, cos_blk, sin_blk, cos_all, sin_all,
            all_pos.astype(F32), blk_pos.astype(F32), kvalid,
        )
    return rms_norm(h, params["norm"], cfg.eps)


def draft_logits(hidden, tgt):
    """LM head of the TARGET model applied to draft hidden states."""
    return jnp.dot(hidden, tgt["lmw"], preferred_element_type=F32)


def select_path(hidden, logits, anchor_id, selector, key_lo_unused=None):
    """Greedy candidate-path selection (torch CandidateSelector.select, T=0).

    hidden [B, H]; logits [B, V]; anchor_id int. Returns (path [B], cand [B, K],
    unary [B, K]).
    """
    K = selector["top_k"]
    unary, cand = jax.lax.top_k(logits, K)  # sorted desc; argmax below is order-invariant
    hproj = jnp.dot(hidden, selector["hidden_projection"].T, preferred_element_type=F32)
    pred = anchor_id
    path = []
    for i in range(hidden.shape[0]):
        cond = selector["predecessor_codebook"][pred] * hproj[i]  # [R]
        scores = unary[i] + selector["successor_codebook"][cand[i]] @ cond  # [K]
        idx = jnp.argmax(scores)
        pred = cand[i, idx]
        path.append(pred)
    return jnp.stack(path).astype(jnp.int32), cand, unary


def load_draft(path):
    """Load the DFlash2 draft checkpoint into stacked numpy arrays."""
    import json
    from safetensors import safe_open

    path = str(path)
    with open(f"{path}/config.json") as f:
        cfg = DraftConfig(json.load(f))
    raw = {}
    with safe_open(f"{path}/model.safetensors", framework="numpy") as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k)
    params = {
        "fc": raw["fc.weight"],
        "hidden_norm": raw["hidden_norm.weight"],
        "norm": raw["norm.weight"],
        "selector": {
            "hidden_projection": raw["candidate_selector.hidden_projection.weight"],
            "predecessor_codebook": raw["candidate_selector.predecessor_codebook"],
            "successor_codebook": raw["candidate_selector.successor_codebook"],
            "top_k": cfg.selector_top_k,
        },
    }
    layers = {}
    for name in (
        "input_layernorm", "post_attention_layernorm",
    ):
        layers[name] = np.stack([raw[f"layers.{i}.{name}.weight"] for i in range(cfg.layers)])
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        layers[name] = np.stack([raw[f"layers.{i}.self_attn.{name}.weight"] for i in range(cfg.layers)])
    for name in ("q_norm", "k_norm"):
        layers[name] = np.stack([raw[f"layers.{i}.self_attn.{name}.weight"] for i in range(cfg.layers)])
    for name in ("gate_proj", "up_proj", "down_proj"):
        layers[name] = np.stack([raw[f"layers.{i}.mlp.{name}.weight"] for i in range(cfg.layers)])
    for conv in ("attention_conv", "mlp_conv"):
        layers[f"{conv}.base_kernel"] = np.stack(
            [raw[f"layers.{i}.{conv}.base_kernel"] for i in range(cfg.layers)]
        )
        layers[f"{conv}.kernel_projection"] = np.stack(
            [raw[f"layers.{i}.{conv}.kernel_projection.weight"] for i in range(cfg.layers)]
        )
    # nest conv params as expected by conv_prepare/conv_finish
    params["layers"] = layers
    return params, cfg


def pack_draft_for_kernel(params, cfg):
    """Restructure stacked layer arrays into the per-layer dict form used by
    draft_layer, with conv metadata attached."""
    layers = []
    for i in range(cfg.layers):
        layer = {}
        for k, v in params["layers"].items():
            layer[k] = v[i]
        layer["attention_conv"] = {
            "base_kernel": layer.pop("attention_conv.base_kernel"),
            "kernel_projection": layer.pop("attention_conv.kernel_projection"),
        }
        layer["mlp_conv"] = {
            "base_kernel": layer.pop("mlp_conv.base_kernel"),
            "kernel_projection": layer.pop("mlp_conv.kernel_projection"),
        }
        layers.append(layer)
    return layers




# ---------------------------------------------------------------------------
# TP-sharded draft (block diffusion over the mesh, psum reductions)
# ---------------------------------------------------------------------------


def pack_draft_tp(params, cfg, tp):
    """Per-rank shards of the draft weights, stacked with a leading tp axis.

    q/k/v split by heads; o_proj/down_proj split by input dim; gate|up fused and
    row-split; fc splits its input dim; convs/norms/selector.hidden_projection
    replicated. Codebooks and the target's emb/lmw shards come from the target
    pack (emb: [tp, vocab/tp, dim]; lmw: [tp, dim, lm_width]).
    """
    import numpy as np

    H = cfg.hidden
    I8 = cfg.intermediate // tp
    V8 = cfg.vocab // tp
    nh_r = cfg.heads // tp
    nkv_r = max(1, cfg.kv_heads // tp)
    qrows = nh_r * cfg.head_dim
    krows = nkv_r * cfg.head_dim

    def sh(name, kind):
        arr = params["layers"][name]  # [L, out, in]
        out = []
        for r in range(tp):
            if kind == "q":
                part = arr[:, r * qrows : (r + 1) * qrows, :]
            elif kind == "kv":
                part = arr[:, r * krows : (r + 1) * krows, :]
            elif kind == "o":  # [H, nh*hd] -> split input (heads) dim
                part = arr[:, :, r * qrows : (r + 1) * qrows]
            elif kind == "down":
                part = arr[:, :, r * I8 : (r + 1) * I8]
            out.append(part)
        return np.stack(out)

    gate = params["layers"]["gate_proj"]  # [L, I, H]
    up = params["layers"]["up_proj"]
    gu = np.concatenate([gate, up], axis=1)  # [L, 2I, H]
    gus = np.stack(
        [
            np.concatenate(
                [gu[:, r * I8 : (r + 1) * I8, :], gu[:, cfg.intermediate + r * I8 : cfg.intermediate + (r + 1) * I8, :]],
                axis=1,
            )
            for r in range(tp)
        ]
    )

    layers = {
        "q_proj": sh("q_proj", "q"),
        "k_proj": sh("k_proj", "kv"),
        "v_proj": sh("v_proj", "kv"),
        "o_proj": sh("o_proj", "o"),
        "gate_up": gus,
        "down_proj": sh("down_proj", "down"),
    }
    for key in ("input_layernorm", "post_attention_layernorm", "q_norm", "k_norm"):
        layers[key] = np.stack([params["layers"][key]] * tp)
    for key in ("attention_conv.base_kernel", "attention_conv.kernel_projection",
                "mlp_conv.base_kernel", "mlp_conv.kernel_projection"):
        layers[key] = np.stack([params["layers"][key]] * tp)

    fc = params["fc"]  # [H, 5H]
    return {
        "fc": np.stack([fc[:, r * (5 * H // tp) : (r + 1) * (5 * H // tp)] for r in range(tp)]),
        "hidden_norm": np.stack([params["hidden_norm"]] * tp),
        "norm": np.stack([params["norm"]] * tp),
        "layers": layers,
        "selector": {
            "hidden_projection": np.stack([params["selector"]["hidden_projection"]] * tp),
            "predecessor_codebook": np.stack(
                [params["selector"]["predecessor_codebook"][r * V8 : (r + 1) * V8] for r in range(tp)]
            ),
            "successor_codebook": np.stack(
                [params["selector"]["successor_codebook"][r * V8 : (r + 1) * V8] for r in range(tp)]
            ),
        },
    }


def _embed_shard(table, ids, rank, vshard, axis="tp"):
    """Row lookup over a vocab-sharded table; psum combines (ids outside the
    shard contribute zeros). table [vshard, ...], ids [...]."""
    lo = rank * vshard
    in_range = (ids >= lo) & (ids < lo + vshard)
    rows = table[jnp.clip(ids - lo, 0, vshard - 1)]
    rows = jnp.where(in_range[..., None], rows, jnp.zeros((), rows.dtype))
    return jax.lax.psum(rows, axis)


def draft_layer_tp(h, ctx, layer, cfg, cos_blk, sin_blk, cos_all, sin_all, kpos, qpos, kvalid):
    """TP-sharded draft layer. h [B, H] replicated; per-rank head/kv shards."""
    B = h.shape[0]
    nh_r = layer["q_proj"].shape[0] // cfg.head_dim
    nkv_r = max(1, layer["k_proj"].shape[0] // cfg.head_dim)
    grp = nh_r // nkv_r
    residual = h
    hn = rms_norm(h, layer["input_layernorm"], cfg.eps)
    hn, dyn = conv_prepare(hn, layer["attention_conv"], cfg)
    q = jnp.dot(hn, layer["q_proj"].T, preferred_element_type=F32).astype(h.dtype)
    q = rms_norm(q.reshape(B, nh_r, cfg.head_dim), layer["q_norm"], cfg.eps)
    k = jnp.concatenate(
        [
            jnp.dot(ctx, layer["k_proj"].T, preferred_element_type=F32).astype(h.dtype),
            jnp.dot(hn, layer["k_proj"].T, preferred_element_type=F32).astype(h.dtype),
        ],
        axis=0,
    )
    k = rms_norm(k.reshape(-1, nkv_r, cfg.head_dim), layer["k_norm"], cfg.eps)
    v = jnp.concatenate(
        [
            jnp.dot(ctx, layer["v_proj"].T, preferred_element_type=F32).astype(h.dtype),
            jnp.dot(hn, layer["v_proj"].T, preferred_element_type=F32).astype(h.dtype),
        ],
        axis=0,
    ).reshape(-1, nkv_r, cfg.head_dim)
    q = apply_rope(q, cos_blk, sin_blk)
    k = apply_rope(k, cos_all, sin_all)
    qg = q.transpose(1, 0, 2).reshape(nkv_r, grp, B, cfg.head_dim)
    kt = k.transpose(1, 0, 2).astype(F32)
    scores = jnp.einsum("gibd,gtd->gibt", qg.astype(F32), kt) * cfg.head_dim**-0.5
    dist = jnp.abs(qpos[:, None] - kpos[None, :])
    keep = (dist < cfg.sliding_window) & kvalid[None, :]
    scores = jnp.where(keep[None, None], scores, -jnp.inf)
    probs = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("gibt,gtd->gibd", probs, v.transpose(1, 0, 2).astype(F32))
    out = out.reshape(nh_r, B, cfg.head_dim).transpose(1, 0, 2).reshape(B, -1)
    out = jnp.dot(out, layer["o_proj"].T, preferred_element_type=F32)
    out = jax.lax.psum(out, "tp").astype(h.dtype)
    out = conv_finish(out, dyn, layer["attention_conv"], cfg)
    h = residual + out
    residual = h
    hn = rms_norm(h, layer["post_attention_layernorm"], cfg.eps)
    hn, dyn = conv_prepare(hn, layer["mlp_conv"], cfg)
    gu = jnp.dot(hn, layer["gate_up"].T, preferred_element_type=F32)
    I8 = layer["gate_up"].shape[0] // 2
    gate = jax.nn.silu(gu[:, :I8].astype(F32)).astype(h.dtype)
    hh = (gate * gu[:, I8 : 2 * I8]).astype(h.dtype)
    out = jnp.dot(hh, layer["down_proj"].T, preferred_element_type=F32)
    out = jax.lax.psum(out, "tp").astype(h.dtype)
    out = conv_finish(out, dyn, layer["mlp_conv"], cfg)
    return h + out


def make_draft(mesh, cfg, tp):
    """TP-sharded DFlash2 draft round. Returns jitted
    (params, lmw, emb, feats, fpos, fvalid, anchor, blk_pos) -> path [B-1] int32.

    lmw: target pack "lmw" [tp, H, lm_width]; emb: "emb" [tp, V8, H]. All
    weights arrive as arguments (never closure constants).
    """
    vshard = cfg.vocab // tp
    nh_r = cfg.heads // tp
    nkv_r = max(1, cfg.kv_heads // tp)

    def local(params, lmw, emb, feats, fpos, fvalid, anchor, blk_pos):
        params = jax.tree.map(lambda a: a[0], params)
        lmw = lmw[0]
        emb = emb[0]
        rank = jax.lax.axis_index("tp")
        B = blk_pos.shape[0]
        # noise stream: [anchor, MASK x (B-1)]
        ids = jnp.concatenate(
            [jnp.reshape(anchor, (1,)), jnp.full((B - 1,), cfg.mask_token_id, jnp.int32)]
        )
        noise = _embed_shard(emb, ids, rank, vshard)  # [B, H]
        # context features: fc over the rank's input slice, then psum
        C5 = feats.shape[1]
        sub = C5 // tp
        fr = jax.lax.dynamic_slice_in_dim(feats, rank * sub, sub, axis=1)
        ctx = jnp.dot(fr, params["fc"].T, preferred_element_type=F32)
        ctx = jax.lax.psum(ctx, "tp")
        ctx = rms_norm(ctx.astype(noise.dtype), params["hidden_norm"], cfg.eps)
        all_pos = jnp.concatenate([fpos, blk_pos])
        kvalid = jnp.concatenate([fvalid, jnp.ones(B, bool)])
        cos_all, sin_all = rope_tables(cfg.theta, cfg.head_dim, all_pos)
        cos_blk, sin_blk = cos_all[-B:], sin_all[-B:]
        h = noise
        for li in range(cfg.layers):
            layer = {k: v[li] for k, v in params["layers"].items()}
            layer = dict(layer)
            layer["attention_conv"] = {
                "base_kernel": layer.pop("attention_conv.base_kernel"),
                "kernel_projection": layer.pop("attention_conv.kernel_projection"),
            }
            layer["mlp_conv"] = {
                "base_kernel": layer.pop("mlp_conv.base_kernel"),
                "kernel_projection": layer.pop("mlp_conv.kernel_projection"),
            }
            h = draft_layer_tp(
                h, ctx, layer, cfg, cos_blk, sin_blk, cos_all, sin_all,
                all_pos.astype(F32), blk_pos.astype(F32), kvalid,
            )
        h = rms_norm(h, params["norm"], cfg.eps)
        hidden = h[-(B - 1):]
        # LM head shard -> per-rank top-k -> global merge
        logits = jnp.dot(hidden, lmw, preferred_element_type=F32)  # [B-1, lm_width]
        logits = logits.at[:, vshard:].set(-jnp.inf)  # exclude the lm pad
        unary_r, cand_r = jax.lax.top_k(logits, cfg.selector_top_k)
        unary = jax.lax.all_gather(unary_r, "tp")  # [tp, B-1, K]
        cand = jax.lax.all_gather(cand_r, "tp")  # local indices
        tpax = jax.lax.iota(jnp.int32, tp)[:, None, None]
        cand_global = cand + (tpax * vshard).astype(cand.dtype)
        # merge: flatten ranks then re-topk
        flat_v = unary.transpose(1, 0, 2).reshape(B - 1, tp * cfg.selector_top_k)
        flat_c = cand_global.transpose(1, 0, 2).reshape(B - 1, tp * cfg.selector_top_k)
        order = jnp.argsort(-flat_v, axis=1)[:, : cfg.selector_top_k]
        unary_g = jnp.take_along_axis(flat_v, order, axis=1)
        cand_g = jnp.take_along_axis(flat_c, order, axis=1)
        # selector: codebooks are vocab-sharded; gather rows via psum trick
        hproj = jnp.dot(hidden, params["selector"]["hidden_projection"].T, preferred_element_type=F32)
        pred = anchor
        path = []
        for i in range(B - 1):
            cond = _embed_shard(params["selector"]["predecessor_codebook"], pred, rank, vshard) * hproj[i]
            succ = _embed_shard(params["selector"]["successor_codebook"], cand_g[i], rank, vshard)
            scores = unary_g[i] + succ @ cond
            idx = jnp.argmax(scores)
            pred = cand_g[i, idx]
            path.append(pred)
        return jnp.stack(path).astype(jnp.int32)

    def draft(params, lmw, emb, feats, fpos, fvalid, anchor, blk_pos):
        P = jax.sharding.PartitionSpec
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(
                jax.tree.map(lambda _: P("tp"), params),
                P("tp"),
                P("tp"),
                P(),
                P(),
                P(),
                P(),
                P(),
            ),
            out_specs=P(),
            check_vma=False,
        )(params, lmw, emb, feats, fpos, fvalid, anchor, blk_pos)

    return jax.jit(draft)



# --------------------------------------------------------------------------
# the fused draft kernel




def pack_draft_kernel(params, cfg, tp):
    """Per-rank kernel weights, [in, out]-oriented, stacked [tp, ...].

    Channel permutation: the residual stream's hidden channels are permuted so
    that group g's conv_group channels sit at permuted positions {g, g+G, ...}
    (p % groups selects the group) — then the dynamic conv's group->channel
    expansion is a free concat, not a matmul. All H-axes of the residual stream
    are permuted consistently; q/k/v/o's head dims and the selector codebooks
    are untouched. The draft only *proposes* tokens (the verify is the lossless
    ground truth), so this is correctness-safe.
    """
    H = cfg.hidden
    I8 = cfg.intermediate // tp
    nh_r = cfg.heads // tp
    nkv_r = max(1, cfg.kv_heads // tp)
    qrows = nh_r * cfg.head_dim
    krows = nkv_r * cfg.head_dim
    groups = H // cfg.conv_group
    gs = cfg.conv_group
    perm = (np.arange(H) % groups) * gs + (np.arange(H) // groups)  # inv: perm[p] = natural channel at p

    L = params["layers"]  # stacked [L, out, in] torch orientation

    def per_rank(build):
        return np.stack([build(r) for r in range(tp)])

    q = per_rank(lambda r: L["q_proj"][:, r * qrows : (r + 1) * qrows, :].transpose(0, 2, 1))
    kv = per_rank(
        lambda r: np.concatenate(
            [L["k_proj"][:, r * krows : (r + 1) * krows, :],
             L["v_proj"][:, r * krows : (r + 1) * krows, :]],
            axis=1,
        ).transpose(0, 2, 1)
    )
    o = per_rank(lambda r: L["o_proj"][:, :, r * qrows : (r + 1) * qrows].transpose(0, 2, 1))
    gate = L["gate_proj"]
    up = L["up_proj"]
    CW = 128  # mlp chunk width — must match make_draft_kernel's CW
    I8P = -(-I8 // CW) * CW  # zero-padded to whole chunks (silu(0)*0 contributes exactly 0)
    NCH = I8P // CW

    def gu_pack(r):
        gs_ = gate[:, r * I8 : (r + 1) * I8, :].transpose(0, 2, 1)  # [L, H, I8]
        us = up[:, r * I8 : (r + 1) * I8, :].transpose(0, 2, 1)
        if I8P != I8:
            gs_ = np.pad(gs_, ((0, 0), (0, 0), (0, I8P - I8)))
            us = np.pad(us, ((0, 0), (0, 0), (0, I8P - I8)))
        gs_ = gs_.reshape(gs_.shape[0], gs_.shape[1], NCH, CW)
        us = us.reshape(us.shape[0], us.shape[1], NCH, CW)
        # interleave per chunk: cols [cc*2*CW, cc*2*CW+CW) = gate, next CW = up
        return np.stack([gs_, us], axis=3).reshape(gs_.shape[0], gs_.shape[1], 2 * I8P)

    gu = per_rank(gu_pack)

    def dw_pack(r):
        d = L["down_proj"][:, :, r * I8 : (r + 1) * I8].transpose(0, 2, 1)  # [L, I8, H]
        if I8P != I8:
            d = np.pad(d, ((0, 0), (0, I8P - I8), (0, 0)))
        return d

    dw = per_rank(dw_pack)

    rep1 = lambda name: np.stack([params["layers"][name]] * tp)  # [tp, nl, w]
    rep2 = lambda name: np.stack([params["layers"][name]] * tp)[:, :, None, :]  # [tp, nl, 1, w]
    # permute the residual-stream H axis of each packed weight
    q = np.take(q, perm, axis=2)      # [tp, nl, H, qrows]: H is the input axis
    kv = np.take(kv, perm, axis=2)
    o = np.take(o, perm, axis=3)      # [tp, nl, qrows, H]: H is the output axis
    gu = np.take(gu, perm, axis=2)
    dw = np.take(dw, perm, axis=3)    # [tp, nl, I8, H]: H is the output axis
    abk = np.take(rep1("attention_conv.base_kernel"), perm, axis=-1)  # per-channel H
    akp = np.stack([params["layers"]["attention_conv.kernel_projection"].transpose(0, 2, 1)] * tp)
    akp = np.take(akp, perm, axis=2)  # [tp, nl, H, ...]: H is the input axis
    mbk = np.take(rep1("mlp_conv.base_kernel"), perm, axis=-1)
    mkp = np.stack([params["layers"]["mlp_conv.kernel_projection"].transpose(0, 2, 1)] * tp)
    mkp = np.take(mkp, perm, axis=2)
    # ctx fold: fc projects the target's feature taps into the (permuted) ctx
    # space. Pre-transpose to [sub, H] per rank so the kernel needs no value
    # transpose; permute the output (H) axis.
    fc = params["fc"]  # [H, feat_dim] natural
    sub = fc.shape[1] // tp
    fcw = np.stack(
        [np.take(fc[:, r * sub : (r + 1) * sub].transpose(1, 0), perm, axis=1) for r in range(tp)]
    )  # [tp, sub, H]
    hnorm = np.take(np.stack([params["hidden_norm"]] * tp)[:, None, :], perm, axis=-1)  # [tp, 1, H]
    return {
        "q": q, "kv": kv, "o": o, "gu": gu, "dw": dw,
        "abk": abk, "akp": akp, "mbk": mbk, "mkp": mkp,
        "ln1": np.take(rep2("input_layernorm"), perm, axis=-1),
        "ln2": np.take(rep2("post_attention_layernorm"), perm, axis=-1),
        "qn": rep2("q_norm"), "kn": rep2("k_norm"),  # head-dim space: not permuted
        "fnorm": np.take(np.stack([params["norm"]] * tp)[:, None, :], perm, axis=-1),
        "selh": np.take(np.stack([params["selector"]["hidden_projection"].transpose(1, 0)] * tp), perm, axis=1),
        "fcw": fcw, "hnorm": hnorm,
        "perm": perm,
    }


def make_draft_kernel(mesh, cfg, tp, block=8, context=1536, fold_ablate=frozenset()):
    """jit(shard_map) draft round: (params, lmw, emb, predcb, succcb, ctxbuf,
    feats_prev, anchor, pospair[3]) -> (path [B-1], ctxbuf).

    The kernel folds the ctx update into its prologue: the previous round's B
    verified feature rows are projected with fc+hidden_norm (permuted copies in
    params) into ctxbuf rows [prev_start, prev_start+B). prev_start == context
    writes dummy padding rows (round 0). The kernel's folded ctx write RMWs an
    8-aligned 16-row block, so ctxbuf must have 16 trailing dummy rows."""
    H = cfg.hidden
    B = block
    K = cfg.selector_top_k
    R = cfg.selector_rank
    nh_r = cfg.heads // tp
    nkv_r = max(1, cfg.kv_heads // tp)
    grp = nh_r // nkv_r
    vshard = cfg.vocab // tp
    hd = cfg.head_dim
    i8 = cfg.intermediate // tp
    C = context
    nl = cfg.layers
    qrows = nh_r * hd
    kvw = 2 * nkv_r * hd
    half = hd // 2
    groups = H // cfg.conv_group
    KC = cfg.conv_kernel
    KPW = 2 * KC * groups  # conv kernel_projection width (narrow slice of bigtile)
    CW = 128  # mlp stream chunk width (128-tile-aligned)
    NCH = -(-i8 // CW)  # pack zero-pads i8 to a whole number of chunks
    LMCW = 1280  # lm head tile width
    TWC = 128  # ctx tile rows (wide tiles, double-buffered)
    WIN = C > cfg.sliding_window
    WINW = C if not WIN else (cfg.sliding_window // TWC + 2) * TWC
    CB = WINW + B
    lm_width = vshard + (-vshard % LMCW)
    feat_dim = len(cfg.target_layer_ids) * H
    sub = feat_dim // tp

    def body(
        pr,
        anchorr,
        q_r,
        kv_r,
        o_r,
        gu_r,
        dw_r,
        abk_r,
        akp_r,
        mbk_r,
        mkp_r,
        ln1_r,
        ln2_r,
        qn_r,
        kn_r,
        fnorm_r,
        selh_r,
        emb_r,
        lmw_r,
        predcb_r,
        succcb_r,
        ctxbuf_r,
        fcw_r,
        hnorm_r,
        feats_r,
        path_out,
        ctxbuf_out,
        *scratch,
    ):
        (
            hidden,
            part,
            recv,
            sends,
            recvs,
            qbuf,
            kvbuf,
            obuf,
            bigtile,
            basebuf,
            ln1buf,
            ln2buf,
            qnbuf,
            knbuf,
            selhbuf,
            gtile,
            dtile,
            ctxtile,
            kc,
            vc,
            cosp,
            sinp,
            topp,
            allp,
            ap_ss,
            ap_sr,
            rowblk,
            e_sem,
            cbblk,
            cb_sem,
            succstage,
            succrows,
            wtsem,
            ctxsem,
            fbuf,
            fctile,
            hnbuf,
            ctxstage,
            fcsem,
            ctxwsem,
        ) = scratch
        rank = jax.lax.axis_index("tp")
        start = pr[0]
        npad = pr[1]
        prev_start = pr[2]
        anchor = anchorr[0]
        iotaCB = jax.lax.iota(jnp.int32, CB)
        w0 = (jnp.maximum(jnp.int32(0), start - cfg.sliding_window) // TWC * TWC) if WIN else jnp.int32(0)
        pos_k = jnp.where(iotaCB < WINW, w0 + iotaCB, start + (iotaCB - WINW))  # [CB]
        valid_k = (iotaCB >= WINW) | ((pos_k >= npad) & (pos_k < start))
        iota_lm = jax.lax.iota(jnp.int32, lm_width)
        iotaB = jax.lax.iota(jnp.int32, B)

        def psum(val):
            """All-reduce a [B, H] value across ranks (bf16 wire); returns f32 sum."""
            part[...] = val.astype(jnp.bfloat16)
            recv[0, rank, ...] = part[...]
            for offset in range(1, tp):
                tpu.make_async_remote_copy(
                    part,
                    recv.at[0, rank],
                    sends.at[offset - 1],
                    recvs.at[offset - 1],
                    device_id=(rank ^ offset,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).start()
            for offset in range(1, tp):
                tpu.make_async_remote_copy(
                    part,
                    recv.at[0, rank],
                    sends.at[offset - 1],
                    recvs.at[offset - 1],
                    device_id=(rank ^ offset,),
                    device_id_type=pl.DeviceIdType.MESH,
                ).wait()
            return jnp.sum(recv[0, ...].astype(F32), axis=0)

        def rms(x, w):
            v = x.astype(F32)
            v *= jax.lax.rsqrt(jnp.mean(v * v, axis=-1, keepdims=True) + cfg.eps)
            return v.astype(jnp.bfloat16) * w  # match reference: cast, then scale

        def rope_rows(vec, cos, sin):
            """vec [rows, hd]; cos/sin [rows, hd] per-row tables."""
            a, b = vec[..., :half], vec[..., half:]
            return jnp.concatenate(
                (a * cos[..., :half] - b * sin[..., :half],
                 b * cos[..., half:] + a * sin[..., half:]),
                axis=-1,
            ).astype(jnp.bfloat16)

        def conv_dyn(x, kp, base):
            """Dynamic grouped 2-tap causal conv on [B, H]; returns (conv0, dyn1).

            base [KC, H] per-channel; kp [H, 2*KC*groups] produces the per-group
            dynamic kernel. The channel permutation makes the group->channel
            expansion a free concat (d01c[:, p] = d01[:, p % groups]).
            """
            dyn = jnp.dot(x, kp, preferred_element_type=F32).astype(jnp.bfloat16)  # [B, 2*KC*groups]
            d0 = jnp.concatenate(
                [dyn[:, tap * groups : (tap + 1) * groups] for tap in range(KC)], axis=0
            )  # [KC*B, groups] -- variant 0 taps stacked
            d1 = jnp.concatenate(
                [dyn[:, KC * groups + tap * groups : KC * groups + (tap + 1) * groups] for tap in range(KC)],
                axis=0,
            )  # [KC*B, groups] -- variant 1 taps (finish conv)
            d01 = jnp.concatenate([d0, d1], axis=0)  # [2*KC*B, groups]
            # permuted channel p holds natural channel (p % groups)*gs + p//groups,
            # whose group is p % groups — so the expansion is a free concat
            d01c = jnp.concatenate([d01] * cfg.conv_group, axis=1)  # [2*KC*B, H]
            d0c = d01c[: KC * B]   # [KC*B, H] per-channel
            d1c = d01c[KC * B :]
            blocks = x  # [B, H]
            out = jnp.zeros_like(blocks)
            for tap in range(KC):
                vals = blocks if tap == 0 else jnp.concatenate(
                    [jnp.zeros((tap, H), blocks.dtype), blocks[: B - tap]], axis=0
                )
                dch = d0c[tap * B : (tap + 1) * B, :]
                out = out + base[tap][None, :] * vals + dch * vals  # bf16 like the reference
            return out, d1c

        def conv_apply(x, d1c, base):
            out = jnp.zeros_like(x)
            for tap in range(KC):
                vals = x if tap == 0 else jnp.concatenate(
                    [jnp.zeros((tap, H), x.dtype), x[: B - tap]], axis=0
                )
                dch = d1c[tap * B : (tap + 1) * B, :]
                out = out + base[tap][None, :] * vals + dch * vals
            return out

        def dma_w(ref, dst, slot, li=None, narrow=None):
            src = ref.at[li] if li is not None else ref
            d = dst.at[:, pl.ds(0, narrow)] if narrow is not None else dst
            return tpu.make_async_copy(src, d, wtsem.at[slot])

        # ---------------- rope tables [CB, hd] ----------------
        inv = 1.0 / (cfg.theta ** (jax.lax.iota(jnp.int32, half).astype(F32) * 2.0 / hd))
        ang = pos_k.astype(F32)[:, None] * inv[None, :]
        cosp[...] = jnp.concatenate((jnp.cos(ang), jnp.cos(ang)), axis=-1).astype(jnp.bfloat16)
        sinp[...] = jnp.concatenate((jnp.sin(ang), jnp.sin(ang)), axis=-1).astype(jnp.bfloat16)

        # -------- folded ctx update: project the previous round's B verified
        # feature rows into ctxbuf rows [prev_start, prev_start+B) --------
        if "nofold" not in fold_ablate:
            tpu.make_async_copy(feats_r.at[:, pl.ds(rank * sub, sub)], fbuf, wtsem.at[14]).start()
            tpu.make_async_copy(hnorm_r, hnbuf, wtsem.at[15]).start()
            tpu.make_async_copy(feats_r.at[:, pl.ds(rank * sub, sub)], fbuf, wtsem.at[14]).wait()
            # fcw streams in SCB-row chunks over the feature axis (contiguous HBM
            # rows), double-buffered; acc over chunks (draft-side: acceptance-only)
            SCB = 64  # fc stream chunk rows (VMEM-capped at large context)
            NSC = sub // SCB
            ctxp = jnp.zeros((B, H), F32)
            if "nodot" not in fold_ablate:
                if "nodma" not in fold_ablate:
                    tpu.make_async_copy(fcw_r.at[pl.ds(0, SCB), :], fctile.at[0], fcsem.at[0]).start()
                for t in range(NSC):
                    buf = t % 2
                    if t + 1 < NSC and "nodma" not in fold_ablate:
                        tpu.make_async_copy(
                            fcw_r.at[pl.ds((t + 1) * SCB, SCB), :], fctile.at[(t + 1) % 2], fcsem.at[(t + 1) % 2]
                        ).start()
                    if "nodma" not in fold_ablate:
                        tpu.make_async_copy(
                            fcw_r.at[pl.ds(t * SCB, SCB), :], fctile.at[buf], fcsem.at[buf]
                        ).wait()
                    ctxp = ctxp + jnp.dot(
                        fbuf[:, t * SCB : (t + 1) * SCB], fctile[buf], preferred_element_type=F32
                    )
            ctxf = ctxp if "nopsum" in fold_ablate else psum(ctxp)  # [B, H] f32 full sum
            tpu.make_async_copy(hnorm_r, hnbuf, wtsem.at[15]).wait()
            ctxn = rms(ctxf, hnbuf[...])  # [B, H] bf16
            if "normw" not in fold_ablate:
                # the HBM ref is tiled 8x128, so dynamic row offsets must be 8-aligned:
                # read-modify-write the aligned 16-row block covering [prev_start, +B).
                # The shifted placement inside the block uses a one-hot matmul (no
                # dynamic gathers/scatters on values).
                al = prev_start // 8 * 8
                off = prev_start - al  # in [0, 8)
                tpu.make_async_copy(ctxbuf_r.at[pl.ds(al, 16), :], ctxstage, ctxwsem.at[0]).start()
                tpu.make_async_copy(ctxbuf_r.at[pl.ds(al, 16), :], ctxstage, ctxwsem.at[0]).wait()
                i16 = jax.lax.iota(jnp.int32, 16)
                j8 = jax.lax.iota(jnp.int32, B)
                shot = (i16[:, None] == (off + j8)[None, :]).astype(jnp.bfloat16)  # [16, B]
                keep = ((i16 < off) | (i16 >= off + B)).astype(jnp.bfloat16)[:, None]  # [16, 1]
                ctxstage[...] = (ctxstage[...].astype(F32) * keep
                                 + jnp.dot(shot, ctxn, preferred_element_type=F32)).astype(jnp.bfloat16)
                tpu.make_async_copy(ctxstage, ctxbuf_r.at[pl.ds(al, 16), :], ctxwsem.at[0]).start()
                # the write-back wait is deferred to just before the layer loop
                # (overlapping the HBM write latency with the emb fetches + barrier + psum)

        # ---------------- embedding rows ----------------
        lo = rank * vshard
        sel8 = jax.lax.iota(jnp.int32, 8)

        def emb_start(slot, tok):
            tokl = jnp.clip(tok - lo, 0, vshard - 1)
            tpu.make_async_copy(
                emb_r.at[pl.ds(tokl // 8 * 8, 8), :], rowblk.at[slot], e_sem.at[slot]
            ).start()
            return tokl

        def emb_wait(slot, tok, tokl):
            tpu.make_async_copy(
                emb_r.at[pl.ds(tokl // 8 * 8, 8), :], rowblk.at[slot], e_sem.at[slot]
            ).wait()
            row = jnp.sum(jnp.where(sel8[:, None] == tokl % 8, rowblk[slot], 0), axis=0).astype(F32)
            return jnp.where((tok >= lo) & (tok < lo + vshard), row, 0.0)

        tokl_a = emb_start(0, anchor)
        tokl_m = emb_start(1, jnp.int32(cfg.mask_token_id))
        # layer-0 weight DMAs: started here so they stream through the barrier+psum
        dma_w(q_r, qbuf, 0, 0).start()
        dma_w(kv_r, kvbuf, 1, 0).start()
        dma_w(o_r, obuf, 2, 0).start()
        dma_w(akp_r, bigtile, 3, 0, narrow=KPW).start()
        dma_w(abk_r, basebuf, 4, 0).start()
        dma_w(ln1_r, ln1buf, 5, 0).start()
        dma_w(qn_r, qnbuf, 6, 0).start()
        dma_w(kn_r, knbuf, 7, 0).start()
        ra = emb_wait(0, anchor, tokl_a)
        rm = emb_wait(1, jnp.int32(cfg.mask_token_id), tokl_m)
        rows = jnp.stack([ra] + [rm] * (B - 1)).astype(jnp.bfloat16)
        barrier = tpu.get_barrier_semaphore()
        for offset in range(1, tp):
            pl.semaphore_signal(
                barrier, 1, device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(barrier, tp - 1)
        hidden[...] = psum(rows.astype(F32)).astype(jnp.bfloat16)
        if "nofold" not in fold_ablate and "normw" not in fold_ablate:
            # deferred: the folded ctx write-back only has to land before the
            # layer loop's ctx streaming reads those rows
            tpu.make_async_copy(ctxstage, ctxbuf_r.at[pl.ds(al, 16), :], ctxwsem.at[0]).wait()

        # Zero the ctx K/V scratch once per round: the position-bounded ctx
        # streaming skips tiles past the anchor, and masked positions MUST read
        # finite values — the attention output einsum evaluates 0 * stale, and
        # stale scratch can be NaN/Inf (0 * NaN = NaN poisons the whole draft).
        kc[...] = jnp.zeros_like(kc[...])
        vc[...] = jnp.zeros_like(vc[...])

        # ---------------- the 5 draft layers ----------------
        for li in range(nl):
            if li > 0:  # li==0 DMAs were started pre-barrier
                dma_w(q_r, qbuf, 0, li).start()
                dma_w(kv_r, kvbuf, 1, li).start()
                dma_w(o_r, obuf, 2, li).start()
                dma_w(akp_r, bigtile, 3, li, narrow=KPW).start()
                dma_w(abk_r, basebuf, 4, li).start()
                dma_w(ln1_r, ln1buf, 5, li).start()
                dma_w(qn_r, qnbuf, 6, li).start()
                dma_w(kn_r, knbuf, 7, li).start()
            dma_w(q_r, qbuf, 0, li).wait()
            dma_w(kv_r, kvbuf, 1, li).wait()
            dma_w(o_r, obuf, 2, li).wait()
            dma_w(akp_r, bigtile, 3, li, narrow=KPW).wait()
            dma_w(abk_r, basebuf, 4, li).wait()
            dma_w(ln1_r, ln1buf, 5, li).wait()
            dma_w(qn_r, qnbuf, 6, li).wait()
            dma_w(kn_r, knbuf, 7, li).wait()

            hn = rms(hidden[...], ln1buf[...])
            hnc, dyn_a = conv_dyn(hn, bigtile[:, pl.ds(0, KPW)], basebuf[0])

            # q/k/v for the block noise rows. Queries for all heads are batched
            # into ONE [nh_r*B, hd] dot chain (rows = (head, block)): the rms is
            # row-wise over hd and rope tables tile per head, so values are
            # identical to the per-head loop.
            qf = jnp.dot(hnc, qbuf[...], preferred_element_type=F32).astype(jnp.bfloat16)  # [B, qrows]
            q4 = jnp.concatenate(
                [qf[:, h_i * hd : (h_i + 1) * hd] for h_i in range(nh_r)], axis=0
            )  # [nh_r*B, hd]
            q4 = rms(q4, qnbuf[...])
            cosb = jnp.concatenate([cosp[WINW:, :]] * nh_r, axis=0)
            sinb = jnp.concatenate([sinp[WINW:, :]] * nh_r, axis=0)
            q4 = rope_rows(q4, cosb, sinb)
            kvn = jnp.dot(hnc, kvbuf[...], preferred_element_type=F32).astype(jnp.bfloat16)  # [B, kvw]
            for g in range(nkv_r):
                kg = kvn[:, g * hd : (g + 1) * hd]
                kg = rms(kg, knbuf[...])
                kg = rope_rows(kg, cosp[WINW:, :], sinp[WINW:, :])
                kc[g, WINW:, :] = kg
                vc[g, WINW:, :] = kvn[:, kvw // 2 + g * hd : kvw // 2 + (g + 1) * hd]

            # context k/v: stream ctxbuf tiles through kvbuf (double-buffered);
            # skip tiles fully past the anchor (their rows are masked by valid_k)
            NT = -(-WINW // TWC)
            tpu.make_async_copy(ctxbuf_r.at[pl.ds(w0, TWC), :], ctxtile.at[0], ctxsem.at[0]).start()
            for t in range(NT):
                o = t * TWC
                if t + 1 < NT:
                    @pl.when(w0 + o + TWC < start)
                    def _prefetch():
                        tpu.make_async_copy(
                            ctxbuf_r.at[pl.ds(w0 + o + TWC, TWC), :], ctxtile.at[(t + 1) % 2], ctxsem.at[(t + 1) % 2]
                        ).start()

                @pl.when(w0 + o < start)
                def _consume():
                    tpu.make_async_copy(ctxbuf_r.at[pl.ds(w0 + o, TWC), :], ctxtile.at[t % 2], ctxsem.at[t % 2]).wait()
                    kvt = jnp.dot(ctxtile[t % 2, ...], kvbuf[...], preferred_element_type=F32).astype(jnp.bfloat16)
                    for g in range(nkv_r):
                        kt = kvt[:, g * hd : (g + 1) * hd]
                        kt = rms(kt, knbuf[...])
                        kt = rope_rows(kt, cosp[o : o + TWC, :], sinp[o : o + TWC, :])
                        kc[g, o : o + TWC, :] = kt
                        vc[g, o : o + TWC, :] = kvt[:, kvw // 2 + g * hd : kvw // 2 + (g + 1) * hd]

            # attention: one dot pair per kv group covering its grp heads
            keep = (jnp.abs(pos_k[None, :] - pos_k[WINW:, None]) < cfg.sliding_window) & valid_k[None, :]
            outs = []
            for g in range(nkv_r):
                rows = q4[g * grp * B : (g + 1) * grp * B]  # [grp*B, hd]
                sc = jnp.einsum("rd,td->rt", rows.astype(F32), kc[g].astype(F32)) * hd**-0.5
                keep_g = jnp.concatenate([keep] * grp, axis=0)  # [grp*B, CB]
                sc = jnp.where(keep_g, sc, -jnp.inf)
                m = jnp.max(sc, axis=-1, keepdims=True)
                p = jnp.exp(sc - m)
                pr_ = p / jnp.sum(p, axis=-1, keepdims=True)  # f32 probs (reference keeps f32)
                outg = jnp.einsum("rt,td->rd", pr_, vc[g].astype(F32))  # [grp*B, hd]
                for hh in range(grp):
                    outs.append(outg[hh * B : (hh + 1) * B])
            attnout = jnp.concatenate(outs, axis=1)  # [B, qrows] f32, head-major
            o_part = jnp.dot(attnout, obuf[...], preferred_element_type=F32)  # [B, H]
            out = psum(o_part).astype(jnp.bfloat16)
            out = conv_apply(out, dyn_a, basebuf[1])
            hidden[...] = (hidden[...] + out).astype(jnp.bfloat16)

            # ---- MLP phase ----
            dma_w(mkp_r, bigtile, 3, li, narrow=KPW).start()
            dma_w(mbk_r, basebuf, 4, li).start()
            dma_w(ln2_r, ln2buf, 5, li).start()
            dma_w(mkp_r, bigtile, 3, li, narrow=KPW).wait()
            dma_w(mbk_r, basebuf, 4, li).wait()
            dma_w(ln2_r, ln2buf, 5, li).wait()
            hn = rms(hidden[...], ln2buf[...])
            hnm, dyn_m = conv_dyn(hn, bigtile[:, pl.ds(0, KPW)], basebuf[0])

            def mlp_dma(cc, buf):
                co = cc * (2 * CW)
                tpu.make_async_copy(gu_r.at[li, :, pl.ds(co, 2 * CW)], gtile.at[buf], wtsem.at[8 + buf]).start()
                tpu.make_async_copy(dw_r.at[li, pl.ds(cc * CW, CW), :], dtile.at[buf], wtsem.at[12 + buf]).start()

            acc = jnp.zeros((B, H), F32)
            mlp_dma(0, 0)
            for cc in range(NCH):
                buf = cc % 2
                if cc + 1 < NCH:
                    mlp_dma(cc + 1, 1 - buf)
                tpu.make_async_copy(gu_r.at[li, :, pl.ds(cc * 2 * CW, 2 * CW)], gtile.at[buf], wtsem.at[8 + buf]).wait()
                guc = jnp.dot(hnm, gtile[buf], preferred_element_type=F32)  # [B, 2*CW] gate|up
                gate_c = jax.nn.silu(guc[:, :CW]).astype(jnp.bfloat16)  # reference rounds gate first
                hh = (gate_c.astype(F32) * guc[:, CW : 2 * CW]).astype(jnp.bfloat16)
                tpu.make_async_copy(dw_r.at[li, pl.ds(cc * CW, CW), :], dtile.at[buf], wtsem.at[12 + buf]).wait()
                acc = acc + jnp.dot(hh, dtile[buf], preferred_element_type=F32)
            out = psum(acc).astype(jnp.bfloat16)
            out = conv_apply(out, dyn_m, basebuf[1])
            hidden[...] = (hidden[...] + out).astype(jnp.bfloat16)

        # ---------------- final norm + LM head + top-k merge ----------------
        dma_w(fnorm_r, ln1buf, 5).start()
        dma_w(selh_r, selhbuf, 0).start()
        dma_w(fnorm_r, ln1buf, 5).wait()
        dma_w(selh_r, selhbuf, 0).wait()
        hidden7 = rms(hidden[...], ln1buf[...])[1:, :]  # [B-1, H]

        lm_parts = []
        NTL = lm_width // LMCW  # 25 single-width tiles, double-buffered (2x1280 halves of bigtile)

        def lm_dma(t, buf):
            tpu.make_async_copy(
                lmw_r.at[:, pl.ds(t * LMCW, LMCW)],
                bigtile.at[:, pl.ds(buf * LMCW, LMCW)],
                wtsem.at[8 + buf],
            ).start()

        lm_dma(0, 0)
        for t in range(NTL):
            buf = t % 2
            if t + 1 < NTL:
                lm_dma(t + 1, 1 - buf)
            tpu.make_async_copy(
                lmw_r.at[:, pl.ds(t * LMCW, LMCW)],
                bigtile.at[:, pl.ds(buf * LMCW, LMCW)],
                wtsem.at[8 + buf],
            ).wait()
            lm_parts.append(
                jnp.dot(hidden7, bigtile[:, pl.ds(buf * LMCW, LMCW)], preferred_element_type=F32)
            )
        lm = jnp.concatenate(lm_parts, axis=1)  # [B-1, lm_width] f32
        lm = jnp.where(iota_lm[None, :] >= vshard, -jnp.inf, lm)

        # per-rank top-K (argmax iterations)
        vals = lm
        cand_l = []
        unary_l = []
        for _ in range(K):
            m = jnp.max(vals, axis=1)  # [B-1]
            idx = jnp.min(
                jnp.where(vals == m[:, None], iota_lm[None, :], lm_width), axis=1
            ).astype(jnp.int32)
            cand_l.append(idx)
            unary_l.append(m)
            vals = jnp.where(iota_lm[None, :] == idx[:, None], -jnp.inf, vals)
        unary_l = jnp.stack(unary_l, axis=1)  # [B-1, K]
        cand_l = jnp.stack(cand_l, axis=1)  # LOCAL ids; the merge adds the rank offset
        topp[...] = jnp.concatenate(
            (jnp.stack((unary_l, cand_l.astype(F32)), axis=-1), jnp.zeros((1, K, 2), F32)), axis=0
        )  # [B, K, 2]
        allp[rank, ...] = topp[...]
        for offset in range(1, tp):
            tpu.make_async_remote_copy(
                topp,
                allp.at[rank],
                ap_ss.at[offset - 1],
                ap_sr.at[offset - 1],
                device_id=(rank ^ offset,),
                device_id_type=pl.DeviceIdType.MESH,
            ).start()
        for offset in range(1, tp):
            tpu.make_async_remote_copy(
                topp,
                allp.at[rank],
                ap_ss.at[offset - 1],
                ap_sr.at[offset - 1],
                device_id=(rank ^ offset,),
                device_id_type=pl.DeviceIdType.MESH,
            ).wait()
        # global merge: concat over ranks (rank-major; order irrelevant to argmax)
        gv = jnp.concatenate([allp[r, : B - 1, :, 0] for r in range(tp)], axis=1)  # [B-1, tp*K]
        gi = jnp.concatenate(
            [allp[r, : B - 1, :, 1].astype(jnp.int32) + r * vshard for r in range(tp)], axis=1
        )
        iota_g = jax.lax.iota(jnp.int32, tp * K)
        unary_rows = []
        cand_rows = []
        for _ in range(K):
            m = jnp.max(gv, axis=1)
            idx = jnp.min(jnp.where(gv == m[:, None], iota_g[None, :], tp * K), axis=1).astype(jnp.int32)
            unary_rows.append(m)
            cand_rows.append(jnp.sum(jnp.where(iota_g[None, :] == idx[:, None], gi, 0), axis=1))
            gv = jnp.where(iota_g[None, :] == idx[:, None], -jnp.inf, gv)
        unary = jnp.stack(unary_rows, axis=1)  # [B-1, K]
        cand = jnp.stack(cand_rows, axis=1).astype(jnp.int32)  # [B-1, K]

        # selector hidden projection
        hproj = jnp.dot(hidden7, selhbuf[...], preferred_element_type=F32)  # [B-1, R]

        # bulk-fetch successor codebook rows for all candidates: start all, wait
        # all, THEN read — the m-th wait only guarantees m total completions, so
        # reading row m before the last wait can race an in-flight copy.
        for i in range(B - 1):
            for k in range(K):
                cid = cand[i, k]
                tpu.make_async_copy(
                    succcb_r.at[pl.ds(cid // 8 * 8, 8), :], succstage.at[i * K + k], cb_sem.at[0]
                ).start()
        for i in range(B - 1):
            for k in range(K):
                cid = cand[i, k]
                tpu.make_async_copy(
                    succcb_r.at[pl.ds(cid // 8 * 8, 8), :], succstage.at[i * K + k], cb_sem.at[0]
                ).wait()
        sel8 = jax.lax.iota(jnp.int32, 8)
        for i in range(B - 1):
            for k in range(K):
                cid = cand[i, k]
                succrows[i, k] = jnp.sum(
                    jnp.where(sel8[:, None] == cid % 8, succstage[i * K + k], 0), axis=0
                ).astype(jnp.bfloat16)

        # 7-step greedy path trace
        pred = anchor
        path = []
        for i in range(B - 1):
            tpu.make_async_copy(predcb_r.at[pl.ds(pred // 8 * 8, 8), :], cbblk, cb_sem.at[0]).start()
            tpu.make_async_copy(predcb_r.at[pl.ds(pred // 8 * 8, 8), :], cbblk, cb_sem.at[0]).wait()
            pred_row = jnp.sum(
                jnp.where(sel8[:, None] == pred % 8, cbblk[...], 0), axis=0
            ).astype(F32)
            cond = pred_row * hproj[i]  # [R]
            scores = unary[i].astype(F32) + succrows[i].astype(F32) @ cond  # [K]
            idx = jnp.argmax(scores)
            iotaK = jax.lax.iota(jnp.int32, K)
            pred = jnp.sum(jnp.where(iotaK == idx, cand[i, :], 0), axis=0)
            path.append(pred)
        path_out[...] = jnp.stack(path).astype(jnp.int32)


    # ---------------- pallas_call + shard_map wrapper ----------------
    v = lambda shp, dtype=jnp.bfloat16: tpu.VMEM(shp, dtype)
    dm = lambda n: tpu.SemaphoreType.DMA((n,))
    scratch = (
        v((B, H)),  # hidden
        v((B, H)),  # part (psum staging, bf16 wire)
        v((1, tp, B, H)),  # recv (bf16 wire)
        dm(tp - 1),  # sends
        dm(tp - 1),  # recvs
        v((H, qrows)),  # qbuf
        v((H, kvw)),  # kvbuf
        v((qrows, H)),  # obuf
        v((H, 2 * LMCW)),  # bigtile: conv kp ([H, 2*KC*groups] slice) + LM double-wide tiles
        v((2, KC, H)),  # basebuf
        v((1, H)),  # ln1buf
        v((1, H)),  # ln2buf
        v((1, hd)),  # qnbuf
        v((1, hd)),  # knbuf
        v((H, R)),  # selhbuf
        v((2, H, 2 * CW)),  # gtile (double-buffered, fused gate|up chunks)
        v((2, CW, H)),  # dtile (double-buffered)
        v((2, TWC, H)),  # ctxtile
        v((nkv_r, CB, hd)),  # kc
        v((nkv_r, CB, hd)),  # vc
        v((CB, hd)),  # cosp
        v((CB, hd)),  # sinp
        v((B, K, 2), jnp.float32),  # topp
        v((tp, B, K, 2), jnp.float32),  # allp
        dm(tp - 1),  # ap_ss
        dm(tp - 1),  # ap_sr
        v((2, 8, H)),  # rowblk (anchor + mask emb blocks, batched)
        dm(2),  # e_sem
        v((8, R)),  # cbblk
        dm(1),  # cb_sem
        v((B * K, 8, R)),  # succstage
        v((B - 1, K, R)),  # succrows
        dm(16),  # wtsem
        dm(2),  # ctxsem
        v((B, sub)),  # fbuf (this rank's slice of the prev round's features)
        v((2, 64, H)),  # fctile (fc feature-axis chunks, double-buffered)
        v((1, H)),  # hnbuf
        v((16, H)),  # ctxstage (aligned RMW block for the folded ctx write)
        dm(2),  # fcsem
        dm(1),  # ctxwsem
    )
    P = jax.sharding.PartitionSpec

    def local(params, lmw, emb, predcb, succcb, ctxbuf, feats, anchor, pospair):
        w = jax.tree.map(lambda a: a[0], params)
        lmw = lmw[0]
        emb = emb[0]
        anchor = jnp.reshape(anchor, (1,))
        return pl.pallas_call(
            body,
            out_shape=(
                jax.ShapeDtypeStruct((B - 1,), jnp.int32),
                jax.ShapeDtypeStruct(ctxbuf.shape, ctxbuf.dtype),
            ),
            input_output_aliases={21: 1},  # ctxbuf threads through in place
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=[pl.BlockSpec()] + [pl.BlockSpec(memory_space=tpu.HBM)] * 23,
                out_specs=[pl.BlockSpec(), pl.BlockSpec(memory_space=tpu.HBM)],
                scratch_shapes=scratch,
            ),
            compiler_params=tpu.CompilerParams(
                collective_id=46,
                vmem_limit_bytes=64 * 1024**2,
                disable_bounds_checks=True,
                shape_invariant_numerics=True,
            ),
            name="dflash_draft_kernel",
        )(
            pospair,
            anchor,
            w["q"], w["kv"], w["o"], w["gu"], w["dw"],
            w["abk"], w["akp"], w["mbk"], w["mkp"],
            w["ln1"], w["ln2"], w["qn"], w["kn"], w["fnorm"], w["selh"],
            emb, lmw, predcb, succcb, ctxbuf, w["fcw"], w["hnorm"], feats,
        )

    def draft(params, lmw, emb, predcb, succcb, ctxbuf, feats, anchor, pospair):
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(
                jax.tree.map(lambda _: P("tp"), params),
                P("tp"), P("tp"), P(), P(), P(), P(), P(), P(),
            ),
            out_specs=(P(), P()),
            check_vma=False,
        )(params, lmw, emb, predcb, succcb, ctxbuf, feats, anchor, pospair)

    return jax.jit(draft)



# --------------------------------------------------------------------------
# the fused speculative round and decode loop




def make_round_fn(draft_fn, verify_fn, c, dcfg, context):
    """jax.jit'd fused speculation round: draft -> verify -> accept -> select.

    (draft_params, lmw, emb, packed_t, states, snaps, featbuf, cur, start, npad)
    -> (bonus, produced, block, states', snaps', featbuf')  — all on device.
    The host slices block[1:produced] + [bonus] for the text stream.
    """
    B = dcfg.block_size

    def _round(draft_params, lmw, emb, packed_t, states, snaps, featbuf, cur, start, npad):
        fpos = jnp.arange(context, dtype=jnp.int32)
        fvalid = (fpos >= npad) & (fpos < start)
        blk_pos = start + jnp.arange(B, dtype=jnp.int32)
        path = draft_fn(draft_params, lmw, emb, featbuf, fpos, fvalid, cur, blk_pos)
        block = jnp.concatenate([cur[None], path]).astype(jnp.int32)
        post, taps, snaps_new, new_states = verify_fn(
            packed_t, states, block, snaps, jnp.array([start, npad], jnp.int32)
        )
        match = block[1:] == post[:-1]
        acc = jnp.sum(jnp.cumprod(match.astype(jnp.int32))).astype(jnp.int32)
        bonus = post[acc]
        # conv snaps are stored as [nl, B*4, qkvw] (row t*4+tap = xp[t+1+tap]);
        # select the accepted window and transpose back to the [nl, qkvw, 4] state layout.
        conv_sel = jax.lax.dynamic_slice_in_dim(
            snaps_new["conv"], acc * c.conv_size, c.conv_size, axis=2
        )
        states_out = {
            "conv": jnp.swapaxes(conv_sel, -2, -1),
            "rec": snaps_new["rec"][:, :, acc],
            "kcache": new_states["kcache"],
            "vcache": new_states["vcache"],
        }
        feats8 = taps.transpose(1, 0, 2).reshape(B, -1)
        featbuf_out = jax.lax.dynamic_update_slice(featbuf, feats8, (start, 0))
        return bonus, acc + 1, block, states_out, snaps_new, featbuf_out

    return jax.jit(_round, donate_argnums=(4, 5, 6))


def make_ctx_update(mesh, dcfg, feat_dim, tp):
    """fc + hidden_norm projection of feature rows into the kernel's ctxbuf.

    fc is input-sharded [tp, H, feat_dim/tp]; feats [N, feat_dim] replicated;
    writes rows [start : start + N) of ctxbuf [context, H] (replicated).
    """
    sub = feat_dim // tp
    P = jax.sharding.PartitionSpec

    def local(fc, hnorm, feats, ctxbuf, start):
        fc, hnorm = fc[0], hnorm[0]
        rank = jax.lax.axis_index("tp")
        fr = jax.lax.dynamic_slice_in_dim(feats, rank * sub, sub, axis=1)
        ctx = jnp.dot(fr, fc.T, preferred_element_type=jnp.float32)
        ctx = jax.lax.psum(ctx, "tp")
        ctx = rms_norm(ctx.astype(jnp.bfloat16), hnorm, dcfg.eps)
        return jax.lax.dynamic_update_slice(ctxbuf, ctx, (jnp.reshape(start, ()), 0))

    def update(fc, hnorm, feats, ctxbuf, start):
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(P("tp"), P("tp"), P(), P(), P()),
            out_specs=P(),
            check_vma=False,
        )(fc, hnorm, feats, ctxbuf, start)

    return jax.jit(update)


def make_round_fn_kernel(draft_k, verify_fn, c, dcfg, context):
    """Fused round with the pallas draft kernel threading a projected ctxbuf.

    (dk, lmw, emb, predcb, succcb, packed_t, states, snaps, featbuf, ctxbuf,
     feats_prev, cur, start, npad, prev_start)
    -> (bonus, produced, block, states', snaps', featbuf', ctxbuf', feats8).
    ctxbuf rows [npad:start) must hold fc+norm-projected target features; the
    draft kernel folds the projection of the previous round's feats8 (at rows
    [prev_start, prev_start+B)) into its prologue. Round 0 passes
    prev_start == context (dummy rows).
    """
    B = dcfg.block_size

    def _round(dk, lmw, emb, predcb, succcb, packed_t, states, snaps, featbuf, ctxbuf,
               feats_prev, cur, start, npad, prev_start):
        pospair = jnp.stack([start, npad, prev_start]).astype(jnp.int32)
        path, ctxbuf_out = draft_k(
            dk, lmw, emb, predcb, succcb, ctxbuf, feats_prev, cur, pospair
        )
        block = jnp.concatenate([cur[None], path]).astype(jnp.int32)
        post, taps, snaps_new, new_states = verify_fn(
            packed_t, states, block, snaps, jnp.array([start, npad], jnp.int32)
        )
        match = block[1:] == post[:-1]
        acc = jnp.sum(jnp.cumprod(match.astype(jnp.int32))).astype(jnp.int32)
        bonus = post[acc]
        conv_sel = jax.lax.dynamic_slice_in_dim(
            snaps_new["conv"], acc * c.conv_size, c.conv_size, axis=2
        )
        states_out = {
            "conv": jnp.swapaxes(conv_sel, -2, -1),
            "rec": snaps_new["rec"][:, :, acc],
            "kcache": new_states["kcache"],
            "vcache": new_states["vcache"],
        }
        feats8 = taps.transpose(1, 0, 2).reshape(B, -1)
        featbuf_out = jax.lax.dynamic_update_slice(featbuf, feats8, (start, 0))
        return bonus, acc + 1, block, states_out, snaps_new, featbuf_out, ctxbuf_out, feats8

    return jax.jit(_round, donate_argnums=(6, 7, 8, 9))


def make_scan_fn(draft_k, verify_fn, c, dcfg, context,
                 stop_ids=(248044, 248046), force_acc=None):
    """Whole speculative decode in one device-side while_loop (no per-round host
    sync). Runs rounds until a stop token is committed, the context is full, or
    `limit` (absolute position bound) is reached.

    (dk, lmw, emb, predcb, succcb, packed_t, states, snaps, featbuf, ctxbuf,
     cur, start, npad, limit)
    -> (tokens [context] int32 — token at absolute position p —, start_final,
        rounds, produced_sum, states', snaps', featbuf', ctxbuf').
    The host reads tokens[width:start_final] for the generated stream.
    ctxbuf must already hold the prompt rows; each draft call
    folds the previous round's ctx projection in via its prologue.
    force_acc (int or None): if set, accept exactly that many drafts per round
    (synthetic acceptance for perf benches; correctness-irrelevant).
    """
    B = dcfg.block_size
    s0, s1 = stop_ids
    feat_dim = len(dcfg.target_layer_ids) * c.dim

    def scan(dk, lmw, emb, predcb, succcb, packed_t, states, snaps, featbuf, ctxbuf,
             cur, start, npad, limit):
        tokens = jnp.zeros((context,), jnp.int32)

        def _round(states, snaps, featbuf, ctxbuf, feats_prev, cur, start, npad, prev_start):
            pospair = jnp.stack([start, npad, prev_start]).astype(jnp.int32)
            path, ctxbuf_out = draft_k(
                dk, lmw, emb, predcb, succcb, ctxbuf, feats_prev, cur, pospair
            )
            block = jnp.concatenate([cur[None], path]).astype(jnp.int32)
            post, taps, snaps_new, new_states = verify_fn(
                packed_t, states, block, snaps, jnp.array([start, npad], jnp.int32)
            )
            match = block[1:] == post[:-1]
            if force_acc is None:
                acc = jnp.sum(jnp.cumprod(match.astype(jnp.int32))).astype(jnp.int32)
            else:
                acc = jnp.minimum(jnp.array(force_acc, jnp.int32), B - 1)
            bonus = post[acc]
            conv_sel = jax.lax.dynamic_slice_in_dim(
                snaps_new["conv"], acc * c.conv_size, c.conv_size, axis=2
            )
            states_out = {
                "conv": jnp.swapaxes(conv_sel, -2, -1),
                "rec": snaps_new["rec"][:, :, acc],
                "kcache": new_states["kcache"],
                "vcache": new_states["vcache"],
            }
            feats8 = taps.transpose(1, 0, 2).reshape(B, -1)
            featbuf_out = jax.lax.dynamic_update_slice(featbuf, feats8, (start, 0))
            return bonus, acc + 1, block, states_out, snaps_new, featbuf_out, ctxbuf_out, feats8

        def cond(carry):
            (_, _, _, _, _, _, start, _, _, _, stopped, _, _) = carry
            return (~stopped) & (start + B <= context) & (start < limit)

        def body(carry):
            (states, snaps, featbuf, ctxbuf, feats_prev, cur, start, prev_start, tokens, npad,
             stopped, rounds, produced_sum) = carry
            (bonus, produced, block, states_out, snaps_new, featbuf_out, ctxbuf_out,
             feats8) = _round(
                states, snaps, featbuf, ctxbuf, feats_prev, cur, start, npad, prev_start
            )
            # token at absolute position p: block[1:] -> [start+1, start+8),
            # bonus -> start+produced (later rounds overwrite unaccepted slots)
            tokens = jax.lax.dynamic_update_slice(tokens, block[1:], (start + 1,))
            tokens = jax.lax.dynamic_update_slice(tokens, bonus[None], (start + produced,))
            idx = jnp.arange(B)
            committed = (idx >= 1) & (idx < produced)
            hit = jnp.any((block == s0) & committed) | jnp.any((block == s1) & committed)
            hit = hit | (bonus == s0) | (bonus == s1)
            return (states_out, snaps_new, featbuf_out, ctxbuf_out, feats8, bonus,
                    start + produced, start, tokens, npad, stopped | hit, rounds + 1,
                    produced_sum + produced)

        init = (states, snaps, featbuf, ctxbuf, jnp.zeros((B, feat_dim), jnp.bfloat16),
                cur, start, jnp.array(context, jnp.int32), tokens, npad,
                jnp.array(False), jnp.array(0, jnp.int32), jnp.array(0, jnp.int32))
        (states, snaps, featbuf, ctxbuf, feats_prev, cur, start, prev_start, tokens, npad,
         stopped, rounds, produced_sum) = jax.lax.while_loop(cond, body, init)
        return tokens, start, rounds, produced_sum, states, snaps, featbuf, ctxbuf

    return jax.jit(scan, donate_argnums=(6, 7, 8, 9))
