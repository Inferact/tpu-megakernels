"""Interactive DeepSeek-V4.1-Flash + DSpark speculative decoding demo on the TP32 megakernel.

Loads the DeepSeek-V4.1-Flash checkpoint (pre-sharded TP32 rank directory or HF snapshot)
including the 40-layer backbone (35 MLA-CSA + 5 full MLA, 4-stream mHC Sinkhorn,
32-head Hadamard Indexer, mla_fp4 KV cache, Engram lookup, FP8 shared expert,
and 192 MXFP4 routed experts) and 3-layer DSpark speculator (layers 40..42),
then reads prompts from the terminal on process 0 (or serves OpenAI-compatible
HTTP endpoints) while all processes execute the single-pallas_call megakernel in lockstep.

Run under the four-host launcher (``scripts/demo_deepseek_v41_dspark.sh``);
pass ``--prompt`` one or more times for a non-interactive session, or
``--serve HOST:PORT`` for the OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import jax

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from deepseek_v41.checkpoint import generate_synthetic_weights
from deepseek_v41.collectives import create_v7x_mesh
from deepseek_v41.config import full_config, tiny_config
from deepseek_v41.engine import DSV41Engine
from deepseek_v41.load import load_presharded
from deepseek_v41.runner import _init_distributed_if_needed
from deepseek_v41.server import run_server_in_thread
from deepseek_v41.xla_decode import shard_weights_for_mesh
from model_paths import resolve_model_source

DEFAULT_CHECKPOINT = "deepseek-ai/DeepSeek-V4.1-Flash"
CHAT_MODES = ("none", "response", "think")


def _log(msg: str) -> None:
    if jax.process_index() == 0:
        print(msg, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("DSV41_CHECKPOINT", DEFAULT_CHECKPOINT),
        help="HF snapshot or local checkpoint directory (for tokenizer & config)",
    )
    parser.add_argument(
        "--weights",
        default=os.environ.get("DSV41_PRESHARDED_WEIGHTS"),
        help="Pre-sharded rank directory (tp32/ or tp8/) produced by deepseek_v41.preshard",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="Prompt(s) for a non-interactive session; repeatable",
    )
    parser.add_argument(
        "--chat",
        choices=CHAT_MODES,
        default="response",
        help="Chat format: 'response', 'think', or 'none' (raw completion)",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=4096,
        help="Maximum attention cache context length",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Generation token budget per prompt",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also decode greedily one token at a time (q=1) for speed/equivalence comparison",
    )
    parser.add_argument(
        "--target-only",
        "--no-spec",
        dest="target_only",
        action="store_true",
        help="Serve or decode with the plain single-token target megakernel without DSpark speculation",
    )
    parser.add_argument(
        "--backend",
        choices=("megakernel", "xla"),
        default="megakernel",
        help="Decode backend: single-pallas_call VMEM megakernel (default) or XLA shard_map control",
    )
    parser.add_argument(
        "--serve",
        default=None,
        metavar="HOST:PORT",
        help="Start OpenAI-compatible /v1/completions and /v1/chat/completions HTTP server",
    )
    parser.add_argument(
        "--model-name",
        default="deepseek-v4.1-flash",
        help="Model ID reported by /v1/models and accepted by /v1/chat/completions",
    )
    parser.add_argument(
        "--progress",
        choices=("tty", "log", "none"),
        default="tty",
        help="Startup progress output format",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use tiny synthetic config for fast CPU smoke testing",
    )
    args = parser.parse_args()

    _init_distributed_if_needed()
    started = time.perf_counter()
    world_size = jax.device_count()
    mesh = create_v7x_mesh()

    cfg = tiny_config() if args.tiny else full_config(max_seq_len=args.context)
    tokenizer_path: str | Path | None = None
    if not args.tiny:
        try:
            ckpt_dir = resolve_model_source(
                args.checkpoint,
                revision=os.environ.get("DSV41_CHECKPOINT_REVISION"),
                local_files_only=bool(os.environ.get("HF_HUB_OFFLINE")),
            )
            tokenizer_path = ckpt_dir
        except Exception:
            if args.weights and (Path(args.weights) / "tokenizer.json").exists():
                tokenizer_path = Path(args.weights)

    if args.weights:
        _log(f"[{time.perf_counter() - started:6.1f}s] loading pre-sharded weights from {args.weights} (world_size={world_size})")
        sharded_weights, load_stats = load_presharded(args.weights, mesh=mesh, cfg=cfg)
        if tokenizer_path is None and (Path(args.weights) / "tokenizer.json").exists():
            tokenizer_path = Path(args.weights)
        _log(f"[{time.perf_counter() - started:6.1f}s] weights resident in {load_stats['total_load_s']:.2f}s")
    else:
        _log(f"[{time.perf_counter() - started:6.1f}s] initializing synthetic weights for config (world_size={world_size})")
        raw = generate_synthetic_weights(cfg, seed=0)
        sharded_weights = shard_weights_for_mesh(raw, cfg, mesh)

    engine = DSV41Engine(
        cfg,
        mesh,
        sharded_weights,
        tokenizer_path=tokenizer_path,
        default_backend=args.backend,
    )

    if args.serve:
        host, _, port_str = args.serve.rpartition(":")
        host = host or "0.0.0.0"
        port = int(port_str) if port_str else 8000
        if jax.process_index() == 0:
            with run_server_in_thread(
                engine,
                host=host,
                port=port,
                model_name=args.model_name,
                default_use_dspark=not args.target_only,
                default_backend=args.backend,
            ) as (base_url, _):
                _log(f"[{time.perf_counter() - started:6.1f}s] serving {args.model_name} on {base_url} (DSpark={'OFF' if args.target_only else 'ON'})")
                try:
                    while True:
                        time.sleep(1.0)
                except KeyboardInterrupt:
                    _log("shutting down server")
                finally:
                    engine.spmd_stop_workers()
        else:
            engine.worker_serve_loop()
        return

    # Interactive / --prompt mode
    if jax.process_index() == 0:
        prompts = list(args.prompt) if args.prompt else None
        try:
            while True:
                if prompts is not None:
                    if not prompts:
                        break
                    user_prompt = prompts.pop(0)
                    print(f"prompt> {user_prompt}", flush=True)
                else:
                    sys.stdout.write("prompt> ")
                    sys.stdout.flush()
                    line = sys.stdin.readline()
                    if not line:
                        break
                    user_prompt = line.strip()
                    if user_prompt.lower() in ("quit", "exit"):
                        break
                    if not user_prompt:
                        continue

                if args.chat == "none":
                    prompt_ids = engine.tokenizer.encode(user_prompt, add_bos=True)
                else:
                    formatted = engine.tokenizer.apply_chat_template(
                        [{"role": "user", "content": user_prompt}]
                    )
                    prompt_ids = engine.tokenizer.encode(formatted, add_bos=True)

                res = engine.spmd_generate_greedy(
                    prompt_ids,
                    max_new_tokens=args.max_tokens,
                    use_dspark=not args.target_only,
                    backend=args.backend,
                    eos_token_id=engine.tokenizer.eos_token_id,
                )
                text = engine.tokenizer.decode(res["generated_ids"])
                print(text, flush=True)
                dspark_stats = res["dspark_stats"]
                print(
                    f"[{len(res['generated_ids'])} tok in {res['decode_s']:.2f}s "
                    f"({res['tok_per_s']:.1f} tok/s, median step {res['median_step_ms']:.2f} ms, "
                    f"DSpark={'OFF' if args.target_only else 'ON'}, "
                    f"emitted/step={dspark_stats.get('mean_emitted_tokens_per_step', 1.0):.2f})]",
                    flush=True,
                )

                if args.baseline and not args.target_only:
                    base = engine.spmd_generate_greedy(
                        prompt_ids,
                        max_new_tokens=args.max_tokens,
                        use_dspark=False,
                        backend=args.backend,
                        eos_token_id=engine.tokenizer.eos_token_id,
                    )
                    match = base["generated_ids"] == res["generated_ids"]
                    speedup = res["tok_per_s"] / max(base["tok_per_s"], 1e-9)
                    print(
                        f"[baseline q=1: {base['tok_per_s']:.1f} tok/s "
                        f"({base['median_step_ms']:.2f} ms/step), "
                        f"DSpark speedup={speedup:.2f}x, greedy_exact_match={match}]",
                        flush=True,
                    )
        finally:
            engine.spmd_stop_workers()
    else:
        engine.worker_serve_loop()


if __name__ == "__main__":
    main()
