"""Golden token and logit generator for DeepSeek-V4.1-Flash (`REF/model.py` + `torch_cpu_shim`).

Runs prefill + 128 greedy (`temperature=0.0`) continuation steps across the 8 evaluation prompts
in `deepseek_v41/golden/prompts.json` (including at least one prompt >= 4,096 tokens) under both
`index_k_mode="reference"` and `index_k_mode="intended"`, saving top-k / full logits and tokens
to JSON and `.npz` (local directory or `gs://` prefix).
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from deepseek_v41.config import DSV41Config, full_config, tiny_config
from deepseek_v41.golden.torch_cpu_shim import (
    build_torch_reference_model,
    make_random_reference_weights,
    set_torch_model_modes,
)

DEFAULT_PROMPTS_PATH = Path(__file__).with_name("prompts.json")


def load_golden_prompts(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Loads the 8 evaluation prompts from `prompts.json`."""
    p = Path(path) if path is not None else DEFAULT_PROMPTS_PATH
    data = json.loads(p.read_text())
    return list(data["prompts"])


def encode_prompt_tokens(
    prompt_entry: dict[str, Any],
    tokenizer: Any = None,
    *,
    vocab_size: int = 129280,
    scale_min_tokens: float = 1.0,
) -> list[int]:
    """Encodes a prompt entry into token IDs, repeating if needed to satisfy `min_tokens`."""
    text = str(prompt_entry["text"])
    min_tokens = max(4, int(round(int(prompt_entry.get("min_tokens", 16)) * scale_min_tokens)))
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        base_ids = list(tokenizer.encode(text))
    elif tokenizer is not None and hasattr(tokenizer, "backend_tokenizer") and hasattr(tokenizer.backend_tokenizer, "encode"):
        base_ids = list(tokenizer.backend_tokenizer.encode(text).ids)
    else:
        # Deterministic byte/word hash tokenization when running without a HF tokenizer file.
        raw_bytes = text.encode("utf-8")
        base_ids = [
            (3 + (b * 257 + idx * 131)) % max(vocab_size - 4, 8)
            for idx, b in enumerate(raw_bytes)
        ]
    if not base_ids:
        base_ids = [3, 4, 5, 6]
    tokens = list(base_ids)
    while len(tokens) < min_tokens:
        tokens.extend(base_ids)
    if prompt_entry.get("repeat_to_min_tokens", False) or len(tokens) > min_tokens:
        tokens = tokens[: max(min_tokens, len(base_ids))]
    return [int(t) % vocab_size for t in tokens]


def _save_bytes(out_prefix: str, filename: str, payload: bytes, content_type: str = "application/octet-stream") -> str:
    if out_prefix.startswith("gs://"):
        from google.cloud import storage

        bucket_name, _, prefix = out_prefix[5:].partition("/")
        blob_path = f"{prefix.rstrip('/')}/{filename}" if prefix else filename
        storage.Client().bucket(bucket_name).blob(blob_path).upload_from_string(payload, content_type=content_type)
        return f"gs://{bucket_name}/{blob_path}"
    os.makedirs(out_prefix, exist_ok=True)
    full_path = os.path.join(out_prefix, filename)
    with open(full_path, "wb") as f:
        f.write(payload)
    return full_path


@torch.inference_mode()
def generate_golden_for_mode(
    model: Any,
    prompt_token_lists: list[list[int]],
    prompt_ids: list[str],
    *,
    continuation_steps: int = 128,
    index_k_mode: str = "reference",
    sinkhorn_iters: int | None = None,
    topk_save: int = 16,
    save_full_logits: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Runs prefill + `continuation_steps` greedy steps for each prompt under `index_k_mode`."""
    set_torch_model_modes(model, index_k_mode=index_k_mode, sinkhorn_iters=sinkhorn_iters)
    records: list[dict[str, Any]] = []
    npz_arrays: dict[str, np.ndarray] = {}

    for p_idx, (pid, p_tokens) in enumerate(zip(prompt_ids, prompt_token_lists)):
        t0 = time.perf_counter()
        if getattr(model, "engram_hash", None) is not None:
            model.engram_hash.cache.zero_()
        for layer in model.layers:
            layer.attn.window_kv_cache.zero_()
            if layer.attn.compressor is not None and hasattr(layer.attn.compressor, "kv_state"):
                layer.attn.compressor.kv_state.zero_()
                layer.attn.compressor.score_state.fill_(-torch.inf)
            if layer.attn.indexer is not None and hasattr(layer.attn.indexer, "k_cache"):
                layer.attn.indexer.k_cache.zero_()

        inp = torch.tensor([p_tokens], dtype=torch.long)
        next_tok_t, logits_t, _ = model.forward(inp, start_pos=0)

        gen_tokens: list[int] = []
        step_topk_ids: list[np.ndarray] = []
        step_topk_logits: list[np.ndarray] = []
        step_full_logits: list[np.ndarray] = []

        k_eff = min(topk_save, logits_t.size(-1))
        cur_pos = len(p_tokens)
        for step in range(continuation_steps):
            if step > 0:
                tok_in = torch.tensor([[gen_tokens[-1]]], dtype=torch.long)
                next_tok_t, logits_t, _ = model.forward(tok_in, start_pos=cur_pos)
                cur_pos += 1
            logits_np = logits_t[0].float().cpu().numpy()
            vals_t, idxs_t = torch.topk(logits_t[0].float(), k=k_eff, dim=-1)
            tok_id = int(idxs_t[0].item())
            gen_tokens.append(tok_id)
            step_topk_ids.append(idxs_t.cpu().numpy().astype(np.int32))
            step_topk_logits.append(vals_t.cpu().numpy().astype(np.float32))
            if save_full_logits:
                step_full_logits.append(logits_np.astype(np.float16))

        elapsed = time.perf_counter() - t0
        rec = {
            "prompt_id": pid,
            "prompt_index": p_idx,
            "prompt_len": len(p_tokens),
            "continuation_steps": continuation_steps,
            "index_k_mode": index_k_mode,
            "generated_tokens": gen_tokens,
            "top1_tokens": [int(x[0]) for x in step_topk_ids],
            "seconds": round(elapsed, 4),
        }
        records.append(rec)
        prefix = f"p{p_idx:02d}"
        npz_arrays[f"{prefix}_prompt_tokens"] = np.asarray(p_tokens, dtype=np.int32)
        npz_arrays[f"{prefix}_generated_tokens"] = np.asarray(gen_tokens, dtype=np.int32)
        npz_arrays[f"{prefix}_topk_ids"] = np.stack(step_topk_ids, axis=0)
        npz_arrays[f"{prefix}_topk_logits"] = np.stack(step_topk_logits, axis=0)
        if save_full_logits:
            npz_arrays[f"{prefix}_full_logits"] = np.stack(step_full_logits, axis=0)

    summary = {
        "index_k_mode": index_k_mode,
        "sinkhorn_iters": sinkhorn_iters,
        "num_prompts": len(records),
        "continuation_steps": continuation_steps,
        "max_prompt_len": max(len(x) for x in prompt_token_lists),
        "prompts": records,
    }
    return summary, npz_arrays


def run_make_golden(
    out_dir: str,
    *,
    cfg: DSV41Config | None = None,
    ckpt_path: str | None = None,
    prompts_file: str | Path | None = None,
    continuation_steps: int = 128,
    modes: tuple[str, ...] = ("reference", "intended"),
    seed: int = 0,
    topk_save: int = 16,
    save_full_logits: bool = False,
    scale_min_tokens: float = 1.0,
) -> dict[str, Any]:
    """Runs golden generation across `modes` and writes JSON + `.npz` artifacts to `out_dir`."""
    if cfg is None:
        cfg = full_config()
    prompts = load_golden_prompts(prompts_file)
    tokenizer = None
    if ckpt_path and Path(ckpt_path).exists():
        tok_json = Path(ckpt_path) / "tokenizer.json"
        if tok_json.exists():
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(tok_json))

    prompt_ids = [str(p["id"]) for p in prompts]
    prompt_token_lists = [
        encode_prompt_tokens(p, tokenizer=tokenizer, vocab_size=cfg.vocab_size, scale_min_tokens=scale_min_tokens)
        for p in prompts
    ]
    max_len = max(len(t) for t in prompt_token_lists) + continuation_steps + 16

    if ckpt_path and Path(ckpt_path).exists():
        from safetensors.torch import load_model

        model = build_torch_reference_model(
            cfg,
            torch_state_dict={},
            max_batch_size=1,
            max_seq_len=max(max_len, 4352),
            temperature=0.0,
        )
        shard_file = Path(ckpt_path) / "model0-mp1.safetensors"
        if shard_file.exists():
            load_model(model, str(shard_file))
    else:
        torch_sd, _ = make_random_reference_weights(cfg, seed=seed)
        model = build_torch_reference_model(
            cfg,
            torch_state_dict=torch_sd,
            max_batch_size=1,
            max_seq_len=max(max_len, 256),
            temperature=0.0,
        )

    all_summaries: dict[str, Any] = {}
    for mode in modes:
        summary, npz_dict = generate_golden_for_mode(
            model,
            prompt_token_lists,
            prompt_ids,
            continuation_steps=continuation_steps,
            index_k_mode=mode,
            topk_save=topk_save,
            save_full_logits=save_full_logits,
        )
        json_bytes = json.dumps(summary, indent=2).encode("utf-8")
        _save_bytes(out_dir, f"golden_{mode}.json", json_bytes, content_type="application/json")

        buf = io.BytesIO()
        np.savez_compressed(buf, **npz_dict)
        _save_bytes(out_dir, f"golden_{mode}.npz", buf.getvalue())
        all_summaries[mode] = summary

    manifest = {
        "modes": list(modes),
        "continuation_steps": continuation_steps,
        "num_prompts": len(prompts),
        "max_prompt_len": max(len(t) for t in prompt_token_lists),
        "summaries": all_summaries,
    }
    _save_bytes(
        out_dir,
        "golden_manifest.json",
        json.dumps(manifest, indent=2).encode("utf-8"),
        content_type="application/json",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="Output directory or gs:// prefix")
    parser.add_argument("--ckpt-path", default=None, help="Path to converted safetensors checkpoint")
    parser.add_argument("--prompts-file", default=None, help="Path to prompts.json")
    parser.add_argument("--steps", type=int, default=128, help="Number of greedy continuation steps")
    parser.add_argument("--modes", default="reference,intended", help="Comma-separated index_k_mode values")
    parser.add_argument("--tiny", action="store_true", help="Run with tiny_config and synthetic weights")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic weights")
    parser.add_argument("--topk", type=int, default=16, help="Number of top-k logits to record per step")
    parser.add_argument("--save-full-logits", action="store_true", help="Also store fp16 full vocab logits")
    parser.add_argument("--scale-min-tokens", type=float, default=1.0, help="Scale factor for prompt min_tokens")
    args = parser.parse_args()

    cfg = tiny_config() if args.tiny else full_config()
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    run_make_golden(
        args.out,
        cfg=cfg,
        ckpt_path=args.ckpt_path,
        prompts_file=args.prompts_file,
        continuation_steps=args.steps,
        modes=modes,
        seed=args.seed,
        topk_save=args.topk,
        save_full_logits=args.save_full_logits,
        scale_min_tokens=args.scale_min_tokens,
    )


if __name__ == "__main__":
    main()
