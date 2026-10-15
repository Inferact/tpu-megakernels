# TPU megakernels for Kimi K3, Qwen3.8-27B, and Muse-Spark-Open

<p align="center">
  <a href="https://inferact.ai/blog/tpu-megakernels">
    <img src="assets/figures/hero.png" alt="TPU megakernel: 709 tokens per second on Kimi K3, 16x TPU v7 vs 452 on 16x GB200" width="100%">
  </a>
</p>

Companion code for the blog post
[700 TPS on Kimi K3: A Case for TPU Megakernels](https://inferact.ai/blog/tpu-megakernels).

This repository contains fused decode implementations for three models:

- **Kimi K3 + DSpark** runs on 32 TPU devices across four hosts. One megakernel
  executes the KDA, MLA, MoE, residual, normalization, and collective work for
  the target model; DSpark provides fused speculative decoding with one anchor
  and seven draft rows.
- **Qwen3.8-27B + DFlash2** runs on eight TPU devices on one host. Its dense
  BF16 megakernel covers the hybrid gated-delta-net and grouped-query-attention
  stack, with an optional fused DFlash2 speculative-decoding loop.
- **Muse-Spark-Open + DSpark** runs the NVFP4 model on 8, 16, or 32 TPU devices across one, two, or four hosts, with optional speculative decoding and non-speculative batching.

The demos support terminal interaction and an OpenAI-compatible HTTP server.

## Results

With speculative decoding at an acceptance length of 6, the megakernels reach
709 accepted tokens per second on Kimi K3 (16× TPU v7 vs 16× GB200, 1.57×) and
1,515 on Qwen3.8-27B (4× TPU v7 vs 4× GB200, 2.18×):

<p align="center">
  <img src="assets/figures/spec-decode-throughput.svg" alt="Speculative decode throughput at acceptance length 6: Kimi K3 709 vs 452 tokens/s; Qwen3.8-27B 1,515 vs 695 tokens/s" width="720">
</p>

DSpark keeps a large lead at shorter acceptance lengths as well:

<p align="center">
  <img src="assets/figures/dspark-acceptance-lengths.svg" alt="Accepted tokens per second with DSpark speculative decoding at acceptance lengths 3 and 6" width="720">
</p>

Without speculative decoding, the megakernels deliver roughly 1.4–2× the
decode throughput of the GB200 baseline at batch sizes 1 through 8:

<p align="center">
  <img src="assets/figures/batch-decode-throughput.svg" alt="Aggregate decode throughput at batch sizes 1, 2, 4 and 8, TPU megakernel vs GB200 baseline" width="720">
</p>

See the [blog post](https://inferact.ai/blog/tpu-megakernels) for the full
analysis and measurement setup.

## Setup

Create the environment from the repository root:

```bash
uv sync
```

The launchers run Hugging Face in offline mode, so the model snapshots must
already be present in the cache selected by `HF_HOME`, or supplied as explicit
local paths. All scripts automatically load an optional, untracked `.env` file
from the repository root. A typical configuration is:

```bash
HF_HOME=/path/to/huggingface
KIMI_PRESHARDED_WEIGHTS=/path/to/kimi-k3-tp32
KIMI_DRAFT_PRESHARDED_WEIGHTS=/path/to/kimi-k3-tp32/dspark
MPK_PACKED_WEIGHTS_TP8=/path/to/muse-spark-tp8-pack
MPK_PACKED_WEIGHTS_TP16=/path/to/muse-spark-tp16-pack
MPK_PACKED_WEIGHTS_TP32=/path/to/muse-spark-tp32-pack
```

The Kimi pre-sharded variables are optional. Without them, the demo prepares
its kernel layouts from `moonshotai/Kimi-K3` and
`RedHatAI/Kimi-K3-speculator.dspark`. The Qwen demo loads
`Qwen/Qwen3.8-27B` and `z-lab/Qwen3.8-27B-DFlash2`; packed weight containers
under `checkpoints/qwen38-tp8` (or `$QWEN_PACKED_WEIGHTS` when set) are used
instead when present. The optional `KIMI_CHECKPOINT_REVISION`,
`KIMI_DRAFT_REVISION`, `QWEN_CHECKPOINT_REVISION`, and
`QWEN_DRAFT_REVISION` variables pin the commits resolved from the `HF_HOME`
cache for the default repositories. Explicit `--checkpoint`,
`--draft`, `--weights`, and `--draft-weights` arguments override these
defaults.

Muse-Spark-Open loads `meta-models/Muse-Spark-Open-816B-A42B-NVFP4` and `Inferact/Muse-Spark-Open-DSpark`. Set the matching `MPK_PACKED_WEIGHTS_TP8`, `MPK_PACKED_WEIGHTS_TP16`, or `MPK_PACKED_WEIGHTS_TP32` to an existing target pack, or pass `--weights`. To create a pack from an existing checkpoint on a high-memory compute node:

```bash
JOBID=<slurm-job> bash scripts/pack_mpk.sh \
  --checkpoint /path/to/Muse-Spark-Open-816B-A42B-NVFP4 \
  --tp 16 --output /path/to/model-storage/muse-spark-tp16-pack
```

Use `--tp 8`, `16`, or `32` and a separate output directory on model-weight storage for each. The DSpark draft loads directly from its checkpoint. `MPK_CHECKPOINT_REVISION` and `MPK_DRAFT_REVISION` optionally pin cached revisions.

For parallel GCS packing, install `uv sync --extra gcs`, set `GCS_BUCKET` and `GCS_MOUNT_ROOT` in `.env`, and run `.venv/bin/python gcs_parallel_pack.py pack_mpk.py` with the same packing arguments on a compute node. Output must be under the bucket's root mount; authentication uses Application Default Credentials.

Kimi requires a four-node Slurm allocation with eight TPU devices per host.
Qwen uses one host with eight TPU devices and can run on the first node of any
allocation containing at least one host.

## CPU tests

On an allocated compute node, run the reference and CPU-interpreter tests with:

```bash
JAX_PLATFORMS=cpu uv run pytest -m "not cpu32"
```

Run the complete suite, including 32-device integration tests, with 32 virtual CPU
devices:

```bash
JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=32 \
uv run pytest
```

These tests validate numerical behavior, not TPU-specific Mosaic lowering,
layouts, VMEM capacity, DMA scheduling, collectives, or performance.

## OpenAI-compatible servers

Start the Kimi K3 + DSpark server on a four-node allocation:

```bash
JOBID=<four-node-job> bash scripts/demo_kimi_dspark.sh \
  --serve 0.0.0.0:8000 --context 4096 --max-tokens 512
```

Start the Qwen3.8-27B + DFlash2 server on the first host of an allocation:

```bash
JOBID=<slurm-job> bash scripts/demo_qwen_dflash.sh \
  --serve 0.0.0.0:8000 --context 2048 --max-tokens 512
```

Start Muse-Spark-Open + DSpark on a two-host allocation:

```bash
JOBID=<two-host-job> bash scripts/demo_mpk_dspark.sh --tp 16 \
  --serve 0.0.0.0:8000 --context 4096 --max-tokens 512
```

For TP8, use `--tp 8` and optionally `HOST=<allocated-host>`. For TP32, use `scripts/demo_mpk_tp32.sh` on four hosts and add `--draft` to enable DSpark. All Muse-Spark-Open paths use `steps_per_call=4`; DSpark supports one sequence at a time.

For non-speculative TP8/TP16 serving, add `--no-spec`; `--batch-size 1`, `2`, `4`, or `8` sets the maximum batch. TP32 defaults to non-speculative batch size one; set `MPK_TP32_EXPERT_PARALLEL=1` to select TP8×EP4 and enable larger batches. Batched serving waits up to 10 ms to collect requests; `--batch-wait-ms` overrides this.

`scripts/eval_mpk_gsm8k.sh` and `scripts/benchmark_mpk.sh` support all three layouts through `--tp 8`, `16`, or `32`. Both default to non-speculative decoding; `--spec` enables DSpark. The evaluator supports `--reuse-server`. Run each script with `--help` for options; both run clients on compute nodes and stop only servers they launch.

The servers expose `GET /health`, `GET /v1/models`, `POST /tokenize`,
`POST /detokenize`, `POST /v1/completions`, and
`POST /v1/chat/completions`. Kimi uses DSpark by default and supports greedy
or temperature/top-p generation; `--target-only` serves the target megakernel
without speculation. Qwen uses DFlash2 by default with greedy generation;
`--no-spec` selects its plain single-token decoder. Omit `--serve` to use
the launchers interactively.

## Source layout

- `kimi/decode_megakernel.py`: fused Kimi K3 target decoder, including KDA,
  MLA, MoE, residual/normalization, and collective communication.
- `kimi/dspark.py`: DSpark draft model, fused speculation step, cache
  management, and draft-weight loading.
- `kimi/load.py`: Kimi tokenizer, checkpoint streaming, and TP32 weight-layout
  preparation.
- `qwen/__init__.py`: Qwen model configuration and reference equations.
- `qwen/load.py`: Qwen tokenizer, checkpoint loading, TP weight packing,
  packed containers, and device placement.
- `qwen/decode_megakernel.py`: Qwen TP-sharded prefill, fused decode, and
  block verification.
- `qwen/dflash.py`: DFlash2 reference and fused draft/speculation kernels.
- `mpk/`: Muse-Spark-Open reference equations, weight packing, TP8/TP16/TP32 kernels, DSpark, and chat formatting.
- `demo_kimi_dspark.py`, `demo_qwen_dflash.py`, `demo_mpk_dspark.py`, and `demo_mpk_tp32.py`: model loading, generation, and serving.
- `pack_mpk.py` and `scripts/pack_mpk.sh`: Muse-Spark-Open target weight packing.
- `gcs_parallel_pack.py`: parallel GCS uploads for sequential packing writes.
- `collectives32.py` and `pool_alias.py`: TPU collective primitives and Pallas
  VMEM aliasing support used by Kimi.
- `openai_server.py` and `distributed_entry.py`: shared HTTP protocol support
  and multi-host process startup.
- `model_paths.py`: local-path and Hugging Face repository resolution through
  `HF_HOME`.
- `scripts/`: Slurm launchers, correctness evaluations, and performance
  benchmarks for the supported models.
- `tests/`: CPU reference, packing, Pallas-interpreter, and integration tests.

Generated weights, checkpoints, compilation caches, evaluation output,
benchmark output, and Slurm logs are excluded from version control.

## Citation

If you use this codebase or build on our results, please cite the companion
blog post:

```bibtex
@misc{novack2026tpumegakernels,
  author       = {Novack, George and Liu, Xuting and Ma, Jeff and Kwon, Woosuk},
  title        = {700 TPS on Kimi K3: A Case for TPU Megakernels},
  howpublished = {\url{https://inferact.ai/blog/tpu-megakernels}},
  year         = {2026},
  month        = sep,
  note         = {Inferact blog post, September 23, 2026},
}
```
