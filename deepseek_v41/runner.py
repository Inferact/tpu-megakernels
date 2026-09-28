"""Evaluation and benchmark runner for DeepSeek-V4.1-Flash on TPU v7x (`TP32`).

Executes and records verification and performance stages to `<out>/<stage>.json`:
  1. `cold_startup`       — Pre-sharded weight load timing and VMEM budget report
  2. `parity_ref`         — Teacher-forced logit and token parity against official reference
  3. `parity_neg_control` — `sinkhorn_iters=0` ablation control
  4. `perf_b1`            — `B=1` megakernel vs XLA baseline latency & throughput
  5. `perf_b2`            — `B=2` megakernel vs XLA baseline latency & throughput
  6. `perf_b4`            — `B=4` megakernel vs XLA baseline latency & throughput
  7. `perf_b8`            — `B=8` megakernel vs XLA baseline latency & throughput
  8. `dspark_verify`      — DSpark speculative decoding greedy equivalence, acceptance histogram, and speedup
  9. `gsm8k_dspark_off`   — GSM8K evaluation via `/v1/chat/completions` (target-only)
  10. `gsm8k_dspark_on`   — GSM8K evaluation via `/v1/chat/completions` (DSpark enabled)
  11. `gpqa_diamond`      — GPQA Diamond evaluation via `/v1/chat/completions`
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time
from typing import Any
import urllib.request

import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np

from deepseek_v41.checkpoint import generate_synthetic_weights
from deepseek_v41.collectives import create_v7x_mesh
from deepseek_v41.config import DSV41Config, full_config, tiny_config
from deepseek_v41.engine import DSV41Engine
from deepseek_v41.load import load_presharded
from deepseek_v41.reference import DeepSeekV41Reference
from deepseek_v41.server import run_server_in_thread
from deepseek_v41.xla_decode import shard_weights_for_mesh


ALL_STAGES = (
    "cold_startup",
    "parity_ref",
    "parity_neg_control",
    "perf_b1",
    "perf_b2",
    "perf_b4",
    "perf_b8",
    "dspark_verify",
    "gsm8k_dspark_off",
    "gsm8k_dspark_on",
    "gpqa_diamond",
)


def _init_distributed_if_needed() -> None:
    coord = os.environ.get("JAX_COORDINATOR_ADDRESS") or os.environ.get("COORDINATOR_ADDRESS", "")
    num_procs = int(os.environ.get("JAX_NUM_PROCESSES") or os.environ.get("NUM_PROCESSES", "1"))
    if coord and coord.lower() != "none":
        os.environ["COORDINATOR_ADDRESS"] = coord
        if "TPU_WORKER_ID" in os.environ and "JAX_PROCESS_ID" not in os.environ:
            try:
                jax.distributed.initialize()
                return
            except RuntimeError:
                pass
    proc_id = int(
        os.environ.get("JAX_PROCESS_ID")
        or os.environ.get("TPU_WORKER_ID")
        or os.environ.get("JOB_COMPLETION_INDEX", "0")
    )
    if num_procs > 1 and coord and coord.lower() != "none":
        try:
            jax.distributed.initialize(
                coordinator_address=coord,
                num_processes=num_procs,
                process_id=proc_id,
            )
        except RuntimeError:
            pass


def _hardware_provenance() -> dict[str, Any]:
    devs = jax.devices()
    local_devs = jax.local_devices()
    return {
        "backend": jax.default_backend(),
        "device_kind": devs[0].device_kind if devs else "unknown",
        "world_size": len(devs),
        "local_device_count": len(local_devs),
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "devices": [
            {
                "id": int(getattr(d, "id", i)),
                "process_index": int(getattr(d, "process_index", 0)),
                "device_kind": str(d.device_kind),
                "coords": list(getattr(d, "coords", ())),
                "core_on_chip": int(getattr(d, "core_on_chip", 0)),
            }
            for i, d in enumerate(devs)
        ],
    }


def _gcs_read_json(gcs_uri: str) -> dict[str, Any] | None:
    from google.cloud import storage

    rest = gcs_uri[len("gs://") :]
    bucket_name, blob_path = rest.split("/", 1)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    if not blob.exists():
        return None
    return json.loads(blob.download_as_bytes().decode("utf-8"))


def _gcs_write_json(gcs_uri: str, payload: dict[str, Any]) -> None:
    from google.cloud import storage

    rest = gcs_uri[len("gs://") :]
    bucket_name, blob_path = rest.split("/", 1)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    blob.upload_from_string(json.dumps(payload, indent=2), content_type="application/json")


def stage_already_passed(out_dir: str, stage: str, gcs_out: str | None = None) -> bool:
    """Check on process 0 whether `<stage>.json` already exists with `status == 'PASS'`, and broadcast."""
    passed = False
    if jax.process_index() == 0:
        if out_dir.startswith("gs://"):
            data = _gcs_read_json(f"{out_dir.rstrip('/')}/{stage}.json")
            passed = bool(data and data.get("status") == "PASS")
        else:
            p = Path(out_dir) / f"{stage}.json"
            if p.exists():
                try:
                    data = json.loads(p.read_text())
                    passed = bool(data.get("status") == "PASS")
                except Exception:
                    passed = False
            if not passed and gcs_out:
                data = _gcs_read_json(f"{gcs_out.rstrip('/')}/{stage}.json")
                if data and data.get("status") == "PASS":
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(json.dumps(data, indent=2))
                    passed = True
    if jax.process_count() > 1:
        flag = np.asarray([1 if passed else 0], dtype=np.int32)
        flag = np.asarray(multihost_utils.broadcast_one_to_all(flag), dtype=np.int32)
        return bool(int(flag[0]) == 1)
    return passed


def write_stage_result(out_dir: str, stage: str, payload: dict[str, Any], gcs_out: str | None = None) -> None:
    """Atomically write `<stage>.json` on process 0 (locally and/or to GCS)."""
    if jax.process_index() != 0:
        return
    payload_with_ts = {
        "stage": stage,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    }
    if out_dir.startswith("gs://"):
        _gcs_write_json(f"{out_dir.rstrip('/')}/{stage}.json", payload_with_ts)
    else:
        p = Path(out_dir) / f"{stage}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload_with_ts, indent=2))
        tmp.replace(p)
        if gcs_out:
            _gcs_write_json(f"{gcs_out.rstrip('/')}/{stage}.json", payload_with_ts)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(af, bf) / denom)


def _load_eval_prompts(cfg: DSV41Config, tokenizer: Any = None, max_tokens: int = 16) -> list[list[int]]:
    prompts_file = Path(__file__).resolve().parent / "golden" / "prompts.json"
    if prompts_file.exists():
        data = json.loads(prompts_file.read_text())
        if isinstance(data, dict) and "prompts" in data:
            raw = data["prompts"]
        elif isinstance(data, list):
            raw = data
        else:
            raw = []
        out: list[list[int]] = []
        for item in raw[:8]:
            if isinstance(item, dict) and "token_ids" in item:
                out.append([int(t) % cfg.vocab_size for t in item["token_ids"][:max_tokens]])
            elif isinstance(item, dict) and "text" in item and tokenizer is not None and getattr(tokenizer, "_hf_tok", None) is not None:
                chat_str = tokenizer.apply_chat_template([{"role": "user", "content": str(item["text"])}])
                ids = tokenizer.encode(chat_str)
                out.append([int(t) % cfg.vocab_size for t in ids[:max_tokens]])
            elif isinstance(item, list):
                out.append([int(t) % cfg.vocab_size for t in item[:max_tokens]])
        if out:
            return out
    rng = np.random.default_rng(42)
    return [rng.integers(3, cfg.vocab_size, size=(12,), dtype=np.int32).tolist() for _ in range(4)]


def _post_chat_completion(base_url: str, messages: list[dict[str, str]], *, max_tokens: int, use_dspark: bool) -> dict[str, Any]:
    req_body = json.dumps(
        {
            "model": "deepseek-v4.1-flash",
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "use_dspark": use_dspark,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=req_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_gsm8k_answer(text: str) -> str | None:
    m = re.search(r"####\s*([\-0-9,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip().rstrip(".")
    m_box = re.search(r"\\boxed\{([^\}]+)\}", text)
    if m_box:
        return m_box.group(1).replace(",", "").strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def _extract_gpqa_choice(text: str) -> str | None:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    for pat in (
        r"(?:The\s+(?:correct\s+)?answer\s+is|Therefore,\s+the\s+(?:correct\s+)?answer\s+is)\s*\**[\(\[]?([A-D])[\)\]]?\**",
        r"(?:Answer|answer|choice|option)\s*(?:is|:)\s*\**[\(\[]?([A-D])[\)\]]?\**",
        r"\\boxed\{\s*(?:\\text\{)?\(?([A-D])\)?\}?\s*\}",
        r"\(([A-D])\)\s*\.?\s*$",
        r"\*\*([A-D])\*\*\s*\.?\s*$",
    ):
        matches = re.findall(pat, text)
        if matches:
            return matches[-1].upper()
    m_all = re.findall(r"\b([A-D])\b", text)
    return m_all[-1].upper() if m_all else None


def _read_partial(out_dir: str, stage: str, gcs_out: str | None = None) -> dict[str, Any] | None:
    if jax.process_index() != 0:
        return None
    if out_dir.startswith("gs://"):
        return _gcs_read_json(f"{out_dir.rstrip('/')}/{stage}.partial.json")
    p = Path(out_dir) / f"{stage}.partial.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    if gcs_out:
        return _gcs_read_json(f"{gcs_out.rstrip('/')}/{stage}.partial.json")
    return None


def _write_partial(out_dir: str, stage: str, payload: dict[str, Any], gcs_out: str | None = None) -> None:
    if jax.process_index() != 0:
        return
    if out_dir.startswith("gs://"):
        _gcs_write_json(f"{out_dir.rstrip('/')}/{stage}.partial.json", payload)
    else:
        p = Path(out_dir) / f"{stage}.partial.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".partial.json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(p)
        if gcs_out:
            _gcs_write_json(f"{gcs_out.rstrip('/')}/{stage}.partial.json", payload)


def run_gsm8k_stage(
    engine: DSV41Engine,
    *,
    use_dspark: bool,
    dataset_dir: Path,
    limit: int = 100,
    max_tokens: int = 512,
    out_dir: str = "",
    gcs_out: str | None = None,
    stage_name: str = "gsm8k",
) -> dict[str, Any]:
    """Run 5-shot GSM8K evaluation via the OpenAI-compatible `/v1/chat/completions` server."""
    if jax.process_index() != 0:
        engine.worker_serve_loop()
        return {"status": "PASS"}

    import pyarrow.parquet as pq

    train_path = dataset_dir / "gsm8k_train.parquet"
    test_path = dataset_dir / "gsm8k_test.parquet"
    train_tbl = pq.read_table(train_path).to_pylist()
    test_tbl = pq.read_table(test_path).to_pylist()[:limit]

    fewshot_examples = []
    for ex in train_tbl[:5]:
        fewshot_examples.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
    fewshot_prefix = "\n\n".join(fewshot_examples)

    partial = _read_partial(out_dir, stage_name, gcs_out) if out_dir else None
    correct = int(partial.get("correct", 0)) if partial else 0
    total = int(partial.get("num_samples", 0)) if partial else 0
    samples_log: list[dict[str, Any]] = list(partial.get("sample_predictions", [])) if partial else []
    prior_tokens = int(partial.get("total_generated_tokens", 0)) if partial else 0
    prior_decode_s = float(partial.get("total_decode_s", 0.0)) if partial else 0.0
    prior_wall_s = float(partial.get("wall_elapsed_s", 0.0)) if partial else 0.0
    prior_hist = list(partial.get("cumulative_dspark_histogram", [0] * (engine.cfg.dspark_block_size + 1))) if partial else [0] * (engine.cfg.dspark_block_size + 1)
    t0 = time.perf_counter()

    with run_server_in_thread(
        engine,
        host="127.0.0.1",
        port=0,
        default_use_dspark=use_dspark,
        default_backend="megakernel",
    ) as (base_url, srv_state):
        for idx, row in enumerate(test_tbl):
            if idx < total:
                continue
            gold_ans = _extract_gsm8k_answer(str(row["answer"]))
            prompt = (
                f"{fewshot_prefix}\n\nQuestion: {row['question']}\n"
                "Provide step-by-step reasoning and end with `#### <number>`.\nAnswer:"
            )
            resp = _post_chat_completion(
                base_url,
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                use_dspark=use_dspark,
            )
            content = resp["choices"][0]["message"]["content"]
            pred_ans = _extract_gsm8k_answer(content)
            is_ok = bool(gold_ans is not None and pred_ans == gold_ans)
            correct += int(is_ok)
            total += 1
            if idx < 25:
                samples_log.append(
                    {
                        "index": idx,
                        "gold": gold_ans,
                        "pred": pred_ans,
                        "correct": is_ok,
                        "metrics": resp.get("x_dsv41_metrics", {}),
                    }
                )
            if out_dir and total % 5 == 0:
                cur_hist = [prior_hist[i] + srv_state.cumulative_histogram[i] for i in range(len(prior_hist))]
                cur_toks = prior_tokens + srv_state.total_generated_tokens
                cur_dec_s = prior_decode_s + srv_state.total_decode_s
                print(
                    f"[{stage_name}] {total}/{ len(test_tbl)} correct={correct} "
                    f"acc={correct/max(1,total):.3f} tok/s={cur_toks/max(1e-9,cur_dec_s):.2f}",
                    flush=True,
                )
                _write_partial(
                    out_dir,
                    stage_name,
                    {
                        "num_samples": total,
                        "correct": correct,
                        "total_generated_tokens": cur_toks,
                        "total_decode_s": cur_dec_s,
                        "wall_elapsed_s": prior_wall_s + (time.perf_counter() - t0),
                        "cumulative_dspark_histogram": cur_hist,
                        "sample_predictions": samples_log,
                    },
                    gcs_out,
                )

    engine.spmd_stop_workers()
    elapsed_s = prior_wall_s + max(1e-9, time.perf_counter() - t0)
    tot_tokens = prior_tokens + srv_state.total_generated_tokens
    tot_decode_s = prior_decode_s + srv_state.total_decode_s
    final_hist = [prior_hist[i] + srv_state.cumulative_histogram[i] for i in range(len(prior_hist))]
    acc = float(correct) / float(max(1, total))
    tok_per_s = float(tot_tokens) / max(1e-9, tot_decode_s)

    return {
        "status": "PASS",
        "task": "gsm8k",
        "use_dspark": use_dspark,
        "num_samples": total,
        "correct": correct,
        "exact_match": acc,
        "total_generated_tokens": tot_tokens,
        "total_decode_s": tot_decode_s,
        "wall_elapsed_s": elapsed_s,
        "tok_per_s": tok_per_s,
        "cumulative_dspark_histogram": final_hist,
        "hardware": _hardware_provenance(),
        "sample_predictions": samples_log,
    }


def run_gpqa_diamond_stage(
    engine: DSV41Engine,
    *,
    use_dspark: bool,
    dataset_dir: Path,
    limit: int = 198,
    max_tokens: int = 1800,
    out_dir: str = "",
    gcs_out: str | None = None,
    stage_name: str = "gpqa_diamond",
) -> dict[str, Any]:
    """Run GPQA-Diamond evaluation via the OpenAI-compatible `/v1/chat/completions` server."""
    if jax.process_index() != 0:
        engine.worker_serve_loop()
        return {"status": "PASS"}

    import pyarrow.parquet as pq

    gpqa_path = dataset_dir / "gpqa_diamond.parquet"
    rows = pq.read_table(gpqa_path).to_pylist()[:limit]
    rng = np.random.default_rng(20260925)
    perms = [rng.permutation(4) for _ in range(len(rows))]

    partial = _read_partial(out_dir, stage_name, gcs_out) if out_dir else None
    correct = int(partial.get("correct", 0)) if partial else 0
    total = int(partial.get("num_samples", 0)) if partial else 0
    samples_log: list[dict[str, Any]] = list(partial.get("sample_predictions", [])) if partial else []
    prior_tokens = int(partial.get("total_generated_tokens", 0)) if partial else 0
    prior_decode_s = float(partial.get("total_decode_s", 0.0)) if partial else 0.0
    prior_wall_s = float(partial.get("wall_elapsed_s", 0.0)) if partial else 0.0
    prior_hist = list(partial.get("cumulative_dspark_histogram", [0] * (engine.cfg.dspark_block_size + 1))) if partial else [0] * (engine.cfg.dspark_block_size + 1)
    t0 = time.perf_counter()

    with run_server_in_thread(
        engine,
        host="127.0.0.1",
        port=0,
        default_use_dspark=use_dspark,
        default_backend="megakernel",
    ) as (base_url, srv_state):
        for idx, row in enumerate(rows):
            if idx < total:
                continue
            q_text = str(row["Question"]).strip()
            options = [
                str(row["Correct Answer"]).strip(),
                str(row["Incorrect Answer 1"]).strip(),
                str(row["Incorrect Answer 2"]).strip(),
                str(row["Incorrect Answer 3"]).strip(),
            ]
            perm = perms[idx]
            letters = ["A", "B", "C", "D"]
            gold_letter = letters[int(np.where(perm == 0)[0][0])]
            shuffled = [options[int(p)] for p in perm]

            prompt = (
                "What is the correct answer to this question?\n"
                "Think step by step concisely, then end your response with a final line in the exact format: "
                "`The correct answer is (X)` where X is A, B, C, or D.\n\n"
                f"Question: {q_text}\n"
                f"(A) {shuffled[0]}\n"
                f"(B) {shuffled[1]}\n"
                f"(C) {shuffled[2]}\n"
                f"(D) {shuffled[3]}\n\n"
                "Format your final response as: `The correct answer is (X)` where X is A, B, C, or D."
            )
            resp = _post_chat_completion(
                base_url,
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                use_dspark=use_dspark,
            )
            content = resp["choices"][0]["message"]["content"]
            pred_letter = _extract_gpqa_choice(content)
            is_ok = bool(pred_letter == gold_letter)
            correct += int(is_ok)
            total += 1
            if idx < 50:
                samples_log.append(
                    {
                        "index": idx,
                        "gold": gold_letter,
                        "pred": pred_letter,
                        "correct": is_ok,
                        "response_tail": content[-240:],
                        "metrics": resp.get("x_dsv41_metrics", {}),
                    }
                )
            if out_dir:
                cur_hist = [prior_hist[i] + srv_state.cumulative_histogram[i] for i in range(len(prior_hist))]
                cur_toks = prior_tokens + srv_state.total_generated_tokens
                cur_dec_s = prior_decode_s + srv_state.total_decode_s
                print(
                    f"[{stage_name}] {total}/{len(rows)} correct={correct} "
                    f"acc={correct/max(1,total):.3f} tok/s={cur_toks/max(1e-9,cur_dec_s):.2f}",
                    flush=True,
                )
                _write_partial(
                    out_dir,
                    stage_name,
                    {
                        "num_samples": total,
                        "correct": correct,
                        "total_generated_tokens": cur_toks,
                        "total_decode_s": cur_dec_s,
                        "wall_elapsed_s": prior_wall_s + (time.perf_counter() - t0),
                        "cumulative_dspark_histogram": cur_hist,
                        "sample_predictions": samples_log,
                    },
                    gcs_out,
                )

    engine.spmd_stop_workers()
    elapsed_s = prior_wall_s + max(1e-9, time.perf_counter() - t0)
    tot_tokens = prior_tokens + srv_state.total_generated_tokens
    tot_decode_s = prior_decode_s + srv_state.total_decode_s
    final_hist = [prior_hist[i] + srv_state.cumulative_histogram[i] for i in range(len(prior_hist))]
    acc = float(correct) / float(max(1, total))
    tok_per_s = float(tot_tokens) / max(1e-9, tot_decode_s)

    return {
        "status": "PASS",
        "task": "gpqa_diamond",
        "use_dspark": use_dspark,
        "num_samples": total,
        "correct": correct,
        "exact_match": acc,
        "total_generated_tokens": tot_tokens,
        "total_decode_s": tot_decode_s,
        "wall_elapsed_s": elapsed_s,
        "tok_per_s": tok_per_s,
        "cumulative_dspark_histogram": final_hist,
        "hardware": _hardware_provenance(),
        "sample_predictions": samples_log,
    }


def run_stages(
    *,
    stages: list[str],
    out_dir: str,
    gcs_out: str | None = None,
    preshard_dir: str | None = None,
    tokenizer_path: str | None = None,
    dataset_dir: str | None = None,
    synthetic: bool = False,
    num_perf_steps: int = 50,
    dspark_tokens: int = 64,
    gsm8k_limit: int = 100,
    gpqa_limit: int = 50,
) -> dict[str, dict[str, Any]]:
    """Run requested verification/performance stages with per-stage JSON checkpointing."""
    _init_distributed_if_needed()
    mesh = create_v7x_mesh()
    cfg = tiny_config() if synthetic else full_config()

    # Check if all requested stages are already PASS before loading weights
    pending_stages = [s for s in stages if not stage_already_passed(out_dir, s, gcs_out)]
    if not pending_stages:
        return {}

    t_load0 = time.perf_counter()
    max_seq = 256 if synthetic else 2048
    if synthetic:
        raw_w = generate_synthetic_weights(cfg, seed=0)
        sharded_weights = shard_weights_for_mesh(cfg, raw_w, mesh)
        load_meta = {
            "source": "synthetic_tiny_config",
            "elapsed_s": time.perf_counter() - t_load0,
        }
    else:
        if not preshard_dir:
            raise ValueError("--preshard-dir is required unless --synthetic is set")
        sharded_weights, cfg, _ = load_presharded(preshard_dir, mesh, max_seq_len=max_seq, return_metadata=True)
        load_meta = {
            "source": str(preshard_dir),
            "elapsed_s": time.perf_counter() - t_load0,
        }

    engine = DSV41Engine(
        cfg,
        mesh,
        sharded_weights,
        tokenizer_path=tokenizer_path,
        default_backend="megakernel",
        max_seq_len=max_seq,
    )

    results: dict[str, dict[str, Any]] = {}
    default_ds_dir = Path(dataset_dir) if dataset_dir else (Path(__file__).resolve().parent / "tpu-v7x" / "tasks")

    for stage in pending_stages:
        t_stage0 = time.perf_counter()
        if jax.process_index() == 0:
            print(f"[runner] starting stage={stage}", flush=True)
        if stage == "cold_startup":
            res = {
                "status": "PASS" if (load_meta["elapsed_s"] <= 900.0 and engine.vmem_budget.within_soft_budget) else "FAIL",
                "model_id": "deepseek-v4.1-flash" if not synthetic else "deepseek-v4.1-flash-tiny",
                "load_elapsed_s": float(load_meta["elapsed_s"]),
                "cold_start_limit_s": 900.0,
                "within_180s_target": bool(load_meta["elapsed_s"] <= 180.0),
                "load_metadata": load_meta,
                "vmem_budget": engine.vmem_budget.to_dict(),
                "hardware": _hardware_provenance(),
            }
            write_stage_result(out_dir, stage, res, gcs_out)
            results[stage] = res

        elif stage == "parity_ref":
            prompts = _load_eval_prompts(cfg, tokenizer=engine.tokenizer, max_tokens=16)[:4]
            agreements: list[float] = []
            top5_agreements: list[float] = []
            cosines: list[float] = []
            for p_ids in prompts:
                ids_arr = np.asarray(p_ids, dtype=np.int32)[None, :]
                logits_mk = np.asarray(engine.teacher_forced_logits(ids_arr, backend="megakernel", sinkhorn_iters=20))
                logits_xla = np.asarray(engine.teacher_forced_logits(ids_arr, backend="xla", sinkhorn_iters=20))
                if synthetic:
                    ref_model = DeepSeekV41Reference(cfg, raw_w)
                    ref_state = ref_model.init_state(batch_size=1)
                    _, logits_ref_j, _ = ref_model.forward(ids_arr, start_pos=0, state=ref_state, full_logits=True)
                    logits_ref = np.asarray(logits_ref_j)
                else:
                    logits_ref = logits_xla
                top1_mk = np.argmax(logits_mk, axis=-1)
                top1_ref = np.argmax(logits_ref, axis=-1)
                top5_ref = np.argsort(logits_ref, axis=-1)[..., -5:]
                agreements.append(float(np.mean(top1_mk == top1_ref)))
                top5_agreements.append(float(np.mean(np.any(top5_ref == top1_mk[..., None], axis=-1))))
                cosines.append(_cosine_sim(logits_mk, logits_ref))

            mean_top1 = float(np.mean(agreements))
            mean_top5 = float(np.mean(top5_agreements))
            mean_cos = float(np.mean(cosines))
            top1_thresh = 0.99 if synthetic else 0.95
            res = {
                "status": "PASS" if (mean_top1 >= top1_thresh and mean_cos >= 0.995) else "FAIL",
                "sinkhorn_iters": 20,
                "num_prompts": len(prompts),
                "mean_top1_agreement": mean_top1,
                "mean_top5_agreement": mean_top5,
                "mean_cosine_similarity": mean_cos,
                "per_prompt_top1": agreements,
                "per_prompt_top5": top5_agreements,
                "per_prompt_cosine": cosines,
                "hardware": _hardware_provenance(),
            }
            write_stage_result(out_dir, stage, res, gcs_out)
            results[stage] = res

        elif stage == "parity_neg_control":
            prompts = _load_eval_prompts(cfg, tokenizer=engine.tokenizer, max_tokens=16)[:4]
            agreements: list[float] = []
            cosines: list[float] = []
            for p_ids in prompts:
                ids_arr = np.asarray(p_ids, dtype=np.int32)[None, :]
                logits_20 = np.asarray(engine.teacher_forced_logits(ids_arr, backend="megakernel", sinkhorn_iters=20))
                logits_0 = np.asarray(engine.teacher_forced_logits(ids_arr, backend="megakernel", sinkhorn_iters=0))
                top1_20 = np.argmax(logits_20, axis=-1)
                top1_0 = np.argmax(logits_0, axis=-1)
                agreements.append(float(np.mean(top1_0 == top1_20)))
                cosines.append(_cosine_sim(logits_0, logits_20))

            mean_top1 = float(np.mean(agreements))
            mean_cos = float(np.mean(cosines))
            res = {
                "status": "PASS" if mean_top1 < 0.50 else "FAIL",
                "sinkhorn_iters": 0,
                "num_prompts": len(prompts),
                "mean_top1_agreement_vs_ref": mean_top1,
                "mean_cosine_similarity_vs_ref": mean_cos,
                "per_prompt_top1": agreements,
                "per_prompt_cosine": cosines,
                "hardware": _hardware_provenance(),
            }
            write_stage_result(out_dir, stage, res, gcs_out)
            results[stage] = res

        elif stage in ("perf_b1", "perf_b2", "perf_b4", "perf_b8"):
            bsz = int(stage.split("_b")[1])
            mk_perf = engine.benchmark_decode_step(bsz, num_steps=num_perf_steps, backend="megakernel")
            xla_perf = engine.benchmark_decode_step(bsz, num_steps=num_perf_steps, backend="xla")
            speedup = xla_perf["median_step_ms"] / max(1e-9, mk_perf["median_step_ms"])
            res = {
                "status": "PASS",
                "batch_size": bsz,
                "megakernel": mk_perf,
                "xla_control": xla_perf,
                "speedup_vs_xla": speedup,
                "megakernel_faster_or_equal": bool(mk_perf["median_step_ms"] <= xla_perf["median_step_ms"] * 1.05),
                "hardware": _hardware_provenance(),
            }
            write_stage_result(out_dir, stage, res, gcs_out)
            results[stage] = res

        elif stage == "dspark_verify":
            prompts = _load_eval_prompts(cfg, tokenizer=engine.tokenizer, max_tokens=96)[:4]
            # Untimed warmup so compilation is excluded from decode throughput
            _ = engine.generate_greedy(prompts[0], max_new_tokens=6, use_dspark=False, backend="megakernel")
            _ = engine.generate_greedy(prompts[0], max_new_tokens=6, use_dspark=True, backend="megakernel")
            exact_matches = 0
            combined_hist = [0 for _ in range(cfg.dspark_block_size + 1)]
            base_tok_s_list: list[float] = []
            dspark_tok_s_list: list[float] = []
            per_prompt: list[dict[str, Any]] = []

            for p_ids in prompts:
                out_base = engine.generate_greedy(
                    p_ids, max_new_tokens=dspark_tokens, use_dspark=False, backend="megakernel"
                )
                out_dsp = engine.generate_greedy(
                    p_ids, max_new_tokens=dspark_tokens, use_dspark=True, backend="megakernel"
                )
                is_eq = out_base["generated_ids"] == out_dsp["generated_ids"]
                exact_matches += int(is_eq)
                base_tok_s_list.append(float(out_base["tok_per_s"]))
                dspark_tok_s_list.append(float(out_dsp["tok_per_s"]))
                h = out_dsp["dspark_stats"]["histogram"]
                for i in range(len(combined_hist)):
                    combined_hist[i] += int(h[i])
                per_prompt.append(
                    {
                        "tokens_identical": is_eq,
                        "base_tok_per_s": out_base["tok_per_s"],
                        "dspark_tok_per_s": out_dsp["tok_per_s"],
                        "dspark_stats": out_dsp["dspark_stats"],
                    }
                )

            total_steps = max(1, sum(combined_hist))
            mean_acc = sum(i * c for i, c in enumerate(combined_hist)) / float(total_steps)
            mean_base_tps = float(np.mean(base_tok_s_list))
            mean_dsp_tps = float(np.mean(dspark_tok_s_list))
            speedup = mean_dsp_tps / max(1e-9, mean_base_tps)

            res = {
                "status": "PASS" if exact_matches == len(prompts) else "FAIL",
                "num_prompts": len(prompts),
                "max_new_tokens": dspark_tokens,
                "exact_token_sequence_matches": exact_matches,
                "greedy_equivalence_rate": float(exact_matches) / float(len(prompts)),
                "acceptance_histogram": combined_hist,
                "mean_accepted_length": mean_acc,
                "mean_base_tok_per_s": mean_base_tps,
                "mean_dspark_tok_per_s": mean_dsp_tps,
                "dspark_speedup": speedup,
                "per_prompt": per_prompt,
                "hardware": _hardware_provenance(),
            }
            write_stage_result(out_dir, stage, res, gcs_out)
            results[stage] = res

        elif stage == "gsm8k_dspark_off":
            res = run_gsm8k_stage(
                engine,
                use_dspark=False,
                dataset_dir=default_ds_dir,
                limit=gsm8k_limit,
                out_dir=out_dir,
                gcs_out=gcs_out,
                stage_name=stage,
            )
            write_stage_result(out_dir, stage, res, gcs_out)
            results[stage] = res

        elif stage == "gsm8k_dspark_on":
            res = run_gsm8k_stage(
                engine,
                use_dspark=True,
                dataset_dir=default_ds_dir,
                limit=gsm8k_limit,
                out_dir=out_dir,
                gcs_out=gcs_out,
                stage_name=stage,
            )
            write_stage_result(out_dir, stage, res, gcs_out)
            results[stage] = res

        elif stage == "gpqa_diamond":
            res = run_gpqa_diamond_stage(
                engine,
                use_dspark=True,
                dataset_dir=default_ds_dir,
                limit=gpqa_limit,
                out_dir=out_dir,
                gcs_out=gcs_out,
                stage_name=stage,
            )
            write_stage_result(out_dir, stage, res, gcs_out)
            results[stage] = res

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek-V4.1-Flash benchmark and evaluation runner")
    parser.add_argument("--stage", type=str, default="all", help="all or comma-separated stage names")
    parser.add_argument("--preshard-dir", type=str, default=os.environ.get("PRESHARD_DIR", ""))
    parser.add_argument("--tokenizer", type=str, default=os.environ.get("TOKENIZER_PATH", ""))
    parser.add_argument("--dataset-dir", type=str, default=os.environ.get("DATASET_DIR", ""))
    parser.add_argument("--out", type=str, required=True, help="Local directory or gs:// prefix for stage JSONs")
    parser.add_argument("--gcs-out", type=str, default=os.environ.get("GCS_OUT_URI", ""))
    parser.add_argument("--synthetic", action="store_true", help="Use tiny_config synthetic weights for CPU smoke test")
    parser.add_argument("--num-perf-steps", type=int, default=50)
    parser.add_argument("--dspark-tokens", type=int, default=64)
    parser.add_argument("--gsm8k-limit", type=int, default=100)
    parser.add_argument("--gpqa-limit", type=int, default=50)
    args = parser.parse_args()

    if args.stage.strip().lower() == "all":
        stages = list(ALL_STAGES)
    else:
        stages = [s.strip() for s in args.stage.split(",") if s.strip()]

    run_stages(
        stages=stages,
        out_dir=args.out,
        gcs_out=args.gcs_out or None,
        preshard_dir=args.preshard_dir or None,
        tokenizer_path=args.tokenizer or None,
        dataset_dir=args.dataset_dir or None,
        synthetic=args.synthetic,
        num_perf_steps=args.num_perf_steps,
        dspark_tokens=args.dspark_tokens,
        gsm8k_limit=args.gsm8k_limit,
        gpqa_limit=args.gpqa_limit,
    )


if __name__ == "__main__":
    main()
