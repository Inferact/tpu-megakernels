# TPU megakernels for Kimi K3, Qwen3.8-27B, and Muse-Spark

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
- **Muse-Spark-Open + DSpark** runs on 8, 16, or 32 TPU devices across one, two, or four hosts. TP8, TP16, and the optional 32-device TP8×EP4 layout support non-speculative batches of 1/2/4/8 requests. The default 32-device layout uses replicated TP8 experts and supports BS=1. Single-sequence DSpark uses a separate eight-position verifier with shared expert weight fetches, using the existing NVFP4 checkpoint without additional quantization.

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

Muse Spark requires a target pack selected by `MPK_PACKED_WEIGHTS_TP8`, `MPK_PACKED_WEIGHTS_TP16`, or `MPK_PACKED_WEIGHTS_TP32`, or by `--weights`. `MPK_CHECKPOINT_REVISION` and `MPK_DRAFT_REVISION` optionally select cached revisions of the default target and draft repositories. Without a revision override, Muse Spark uses the cache's current repository revision. It does not download checkpoints during serving.

Kimi requires a four-node Slurm allocation with eight TPU devices per host.
Qwen uses one host with eight TPU devices and can run on the first node of any
allocation containing at least one host.

### Parallel GCS packing

For parallel uploads while packing, install `uv sync --extra gcs` and add these settings to the untracked `.env` in the repository root:

```bash
GCS_BUCKET=your-model-bucket
GCS_MOUNT_ROOT=/mnt/models
```

`GCS_MOUNT_ROOT` must be the mount of that bucket's root. Authentication uses Google Application Default Credentials, including the compute node's service account. On an allocated compute node, wrap the packing script:

```bash
.venv/bin/python gcs_parallel_pack.py pack_mpk.py \
  --checkpoint /path/to/Muse-Spark-Open-816B-A42B-NVFP4 \
  --tp 16 --output /mnt/models/muse-spark-tp16
```

The wrapper streams `pack.bin` writes with 16 upload threads and chunks of up to 128 MiB, verifies CRC32C and size, composes the parts in order, and removes temporary objects. It preserves the packer's layout and manifests and refuses to overwrite existing payloads. Put uploader options such as `--workers 8 --chunk-mib 64` before the packing script; arguments after it belong to the packer.

The default filename pattern covers both MPK and Qwen `save_pack` containers. Other packers can import `GCSParallelWriter` or use repeated `--pattern` options, such as `--pattern '*.bin'` for Kimi-style shard names. Writers must use sequential Python `write()` calls; seeking, file-descriptor writes such as NumPy `tofile`, and child-process writes are not redirected. Kimi's model-specific shard conversion remains the responsibility of its packer.

## CPU correctness tests

Run the fast reference and CPU-interpreter tests with:

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

The servers expose `GET /health`, `GET /v1/models`, `POST /tokenize`,
`POST /detokenize`, `POST /v1/completions`, and
`POST /v1/chat/completions`. Kimi uses DSpark by default and supports greedy
or temperature/top-p generation; `--target-only` serves the target megakernel
without speculation. Qwen uses DFlash2 by default with greedy generation;
`--no-spec` selects its plain single-token decoder. Omit `--serve` to use
either launcher interactively.

## Muse Spark

Muse Spark supports TP8 on one host, TP16 on two hosts, and TP32 on four hosts, with eight logical devices per host. Non-speculative decoding and DSpark use `steps_per_call=4` by default; no command-line argument is needed. Use existing target packs and the `Inferact/Muse-Spark-Open-DSpark` checkpoint.

TP8 and TP16 share `demo_mpk_dspark.py` and `mpk/decode_megakernel.py`. DSpark is enabled by default; add `--no-spec` for non-speculative decoding and batching. Select `--tp 8` for one host or `--tp 16` for two hosts. The 32-device entrypoint is `demo_mpk_tp32.py`, with both of its expert layouts in `mpk/decode_megakernel_32.py`; add `--draft` to enable DSpark there.

DSpark uses greedy acceptance, seven draft proposals, eight verification rows, and four speculative rounds per call. Synthetic acceptance benchmarks use the same draft and verifier for every acceptance length; they measure performance separately from accuracy and observed acceptance. Run model, evaluation, and benchmark processes on allocated compute hosts. The serving host runs JAX process 0, which can differ from the first allocated host; the multi-host benchmark scripts locate it through `/health`.

Use `scripts/benchmark_mpk.sh` for streaming benchmarks and `scripts/eval_mpk_gsm8k.sh` for GSM8K across all three TP sizes. Both accept `--tp 8`, `16`, or `32`; otherwise they use `MPK_TP` or the allocation's device count. Both default to non-speculative decoding. `--spec` enables the cached DSpark draft, and `--draft /path/to/checkpoint` selects another existing draft. `--fixed-acceptance-length` enables synthetic DSpark acceptance in the benchmark only. The evaluator accepts `--reuse-server` for an already running server and passes arguments after `--` to `lm_eval`. Both scripts run their clients on compute nodes and stop only servers they launch. Run `--help` for environment overrides and defaults.

<details>
<summary>Create packed weights</summary>

Run the packer once for each required TP size on a high-memory compute node. The full checkpoint requires several hundred GB of host RAM; use a node with approximately 1 TB. Choose a new output directory on shared model-weight storage, outside the repository.

```bash
JOBID=<slurm-job> bash scripts/pack_mpk.sh \
  --checkpoint /path/to/Muse-Spark-Open-816B-A42B-NVFP4 \
  --tp 32 --output /path/to/shared-models/muse-spark-tp32
```

Use `--tp 8`, `16`, or `32` and a separate output directory for each. The script creates `base/`, `prefill/`, and `tiled/` containers while preserving the checkpoint's existing quantization. Pass the output directory as `--weights` when serving. The checkpoint must already be local or cached; the packer does not download weights. `--workers` controls concurrent layer reads. `HOST` selects a compute node within the allocation. On an allocated compute node without the launcher, use `.venv/bin/python pack_mpk.py` with the same arguments.

Serving expects the target pack to exist. DSpark loads its draft checkpoint directly through `--draft` and needs no separate pack.

</details>

### TP8

Start the Muse-Spark-Open-816B-A42B-NVFP4 server on one host of an allocation:

```bash
JOBID=<slurm-job> bash scripts/demo_mpk_dspark.sh --tp 8 --no-spec \
  --serve 0.0.0.0:8000 --context 4096 --max-tokens 512
```

TP8 uses eight logical devices on one host with four chips. A two-host 2×2×2 slice can run two independent TP8 servers; select the host with `HOST`. The target and draft embeddings and expert weights remain sharded. Prefill and decode use the same resident checkpoint pack, with two KV heads per rank in a combined BF16 cache. Output-column attention and unpool projections retain their original layouts for prefill. Packed expert bytes and scales are joined on the CPU before placement to avoid a second expert-weight allocation in HBM.

The persistent target decoder overlaps weight loads with compute, pipelines attention-output transfers, and exchanges expert rows within the host. Packed experts use four prefetch slots; the wider BF16 expert shards use two. The communicating kernel returns only the new KV rows, which are scattered into the donated cache outside that kernel. This removes compiler-inserted full-cache copies. The draft loads its full Q projection and tiles the O and feed-forward projections, using shared VMEM buffers. These paths add no quantization.

TP8 DSpark verification processes one expert per iteration, reducing temporary matrices and avoiding block-diagonal down-projection inputs. After Q projection, its VMEM storage holds the full attention-output matrix, replacing two sequential loads. At context capacities up to 4096, KV loading overlaps Q projection. A separate cache-update kernel uses strided DMA to update an aligned 32-row region per layer in place, avoiding a full-cache scatter. Weight layout, precision, seven proposals, eight verification rows, and four rounds per call remain the same. These specializations apply only to TP8 verification. The TP8 draft keeps K and V together in the fused kernel's native BF16 layout and updates two aligned 16-row ring-cache regions in place. It packs emitted tokens, acceptance lengths, and terminal state into one buffer and starts its host copy asynchronously. FC/K/V projections, normalization, and RoPE retain their arithmetic; no weights are replicated.

<details>
<summary>Launch TP8 and reproduce the benchmarks</summary>

Repeat the sweep with maximum server batch sizes 1, 2, 4, and 8. Restart the server for each size and use a separate result directory; every server configuration uses the same concurrency sweep and request counts below.

```bash
B=8
JOBID=<slurm-job> HOST=<one-allocated-host> bash scripts/demo_mpk_dspark.sh --tp 8 --no-spec \
  --weights /path/to/muse-spark-tp8-pack \
  --serve 0.0.0.0:8000 --context 9216 --max-tokens 1000 \
  --batch-size "$B" --model-name muse-spark-tp8

for C in 1 2 4 8; do
  srun --jobid=<same-job> --overlap --nodes=1 --ntasks=1 \
    /path/to/vllm-bench \
    --backend openai --base-url http://<server-host>:8000 --model muse-spark-tp8 \
    --tokenizer <existing-target-checkpoint> \
    --dataset-name random --random-input-len 8000 --random-output-len 1000 \
    --random-range-ratio 0 --prompt-token-ids --seed 0 \
    --max-concurrency "$C" --num-warmups "$((2 * C))" --num-prompts "$((4 * C))" \
    --ignore-eos --temperature 0 --percentile-metrics ttft,tpot --no-steady-state \
    --save-result --save-detailed --result-dir "benchmark-results/tp8/isl8000-osl1000/bs$B/c$C" \
    --result-filename result.json
done

# Run sequentially on an idle host for BS=1, ISL=1/OSL=1000.
JOBID=<slurm-job> HOST=<one-allocated-host> bash scripts/benchmark_mpk.sh --tp 8
JOBID=<slurm-job> HOST=<one-allocated-host> bash scripts/benchmark_mpk.sh --tp 8 \
  --fixed-acceptance-length 3
JOBID=<slurm-job> HOST=<one-allocated-host> bash scripts/benchmark_mpk.sh --tp 8 \
  --fixed-acceptance-length 6

JOBID=<slurm-job> HOST=<one-allocated-host> bash scripts/eval_mpk_gsm8k.sh --tp 8
JOBID=<slurm-job> HOST=<one-allocated-host> bash scripts/eval_mpk_gsm8k.sh --tp 8 --batch-size 8
JOBID=<slurm-job> HOST=<one-allocated-host> bash scripts/eval_mpk_gsm8k.sh --tp 8 --spec
```

Omit `--fixed-acceptance-length` for actual greedy acceptance and accuracy evaluation. `scripts/demo_mpk.sh` remains a non-speculative TP8 convenience launcher for the same Python entrypoint. `MPK_PACKED_WEIGHTS_TP8` overrides the pack directory; the single-host launcher also accepts the existing `MPK_PACKED_WEIGHTS` override.

</details>

### TP16

The two-host launcher selects TP16 from the allocation; process 0 serves:

```bash
JOBID=<two-node-job> bash scripts/demo_mpk_dspark.sh --no-spec \
  --serve 0.0.0.0:8000 --context 4096 --max-tokens 512
```

TP16 uses sixteen logical devices on an eight-chip, two-host slice. Expert weights are partitioned once across all sixteen devices. Attention uses TP16 Q/K/V, with attention output and unpool projections arranged for host-local gathers. Small router/pool weights and the unpool projection are replicated across the two hosts. The implementation is in `mpk/decode_megakernel.py`, `mpk/dspark.py`, and `mpk/load.py`; the existing TP8 and TP32 kernel paths are retained.

#### Prefill

TP16 prefill fuses packed gate/up expansion, grouped GEMM, and the expert activation. Expanded weights are reused across the routed token tiles. Shorter prompts use tensor-sharded projections. At BS=2/4/8 and at least 15,360 bucketed prompt tokens, prefill shards prompt rows across the sixteen devices and temporarily redistributes one expert layer at a time. It gathers token activations across host pairs and then within each host, with the reverse hierarchy for the expert-output reduction. Larger batches prefetch the next packed down-projection layer. Resident weights and the decode KV layout remain unchanged.

Attention remains independent per sequence, including left padding and sliding-window masks. Flash attention scales FP32 scores after the BF16 QK product. Packed expansion retains the original FP32 scale products and BF16 weight values. Wider gate/up tiles expand weights in smaller chunks to fit VMEM. Embeddings remain sharded; BS=8 directly reduce-scatters their existing BF16 values into prompt shards. There is no added quantization or permanent weight copy. Grouped projections and collective reductions use different FP32 accumulation orders.

#### Non-speculative batching

`demo_mpk_dspark.py --tp 16 --no-spec --batch-size 8` enables static batches of up to eight independent requests. BS=2/4/8 share target weights and have separate KV caches, positions, EOS handling, and generation budgets. The default is BS=1; a single queued request uses the existing single-token kernel. Batched decoding keeps `steps-per-call=4` and the asynchronous two-chunk dispatch pipeline. No additional quantization or embedding replication is used.

Concurrent completion/chat requests and a non-streaming completion's list of prompts can form a batch. `--batch-wait-ms` defaults to 10 ms and applies only to non-speculative serving with `--batch-size > 1`. After the first request, the server waits up to this limit for more requests and starts immediately when the batch fills. BS=1 and speculative decoding have no batching delay. Setting `--batch-wait-ms 0` collects only requests already queued. Batches are not refilled while decoding. Long prompts prefill together before batched decode. Streaming accepts one prompt per HTTP request, with concurrent requests streamed independently.

The kernel entry points are `make_decode_batch` and `make_decode_batch_chunk` in `mpk/decode_megakernel.py`. KV layout is `[16, layers, batch_size, context, 128]`; token IDs and positions have shape `[batch_size]`. Negative positions preserve inactive cache slots. BS=2/4 use eight compute rows with repeated inputs, while allocating only two/four sequence caches. Embedding normalization preserves the BS=1 arithmetic order. Grouped attention-output reductions use FP32 sums in a different order. DMA counters are separated for embedding, attention output, and expert-row phases so outstanding transfers cannot satisfy a different phase's waits.

<details>
<summary>Launch non-speculative TP16 batching and reproduce the sweep</summary>

Repeat with maximum server batch sizes 1, 2, 4, and 8, restarting the server for each size. Keep the concurrency sweep and request counts unchanged.

```bash
B=8
JOBID=<two-node-job> bash scripts/demo_mpk_dspark.sh --no-spec \
  --tp 16 --serve 0.0.0.0:8000 --context 9216 --max-tokens 1000 \
  --batch-size "$B" --model-name muse-spark-tp16-batch

curl http://<server-host>:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"muse-spark-tp16-batch","prompt":["One plus one is","The capital of France is"],"max_tokens":32,"temperature":0}'

for C in 1 2 4 8; do
  srun --jobid=<same-job> --overlap --nodes=1 --ntasks=1 \
    /path/to/vllm-bench \
    --backend openai --base-url http://<server-host>:8000 --model muse-spark-tp16-batch \
    --tokenizer <existing-target-checkpoint> \
    --dataset-name random --random-input-len 8000 --random-output-len 1000 \
    --random-range-ratio 0 --prompt-token-ids --seed 0 \
    --max-concurrency "$C" --num-warmups "$((2 * C))" --num-prompts "$((4 * C))" \
    --ignore-eos --temperature 0 --percentile-metrics ttft,tpot --no-steady-state \
    --save-result --save-detailed --result-dir "benchmark-results/tp16/isl8000-osl1000/bs$B/c$C" \
    --result-filename result.json
done
```

</details>

#### Single-sequence decoding and DSpark

Single-token decoding uses a dedicated one-row kernel, direct routed-expert indices, compact communication buffers, and a two-phase attention-output reduction. It normalizes the requested embedding row on its owning shard and assembles that row inside the kernel while the first Q/O weights load. The embedding table remains sharded. Expert results are sent directly to their owners on both hosts while later experts compute, preserving the FP32 sum order.

For single-token TP16 decoding, each gather shares completion counters across its seven intra-host transfers, and the fifteen remote expert partials use separate shared counters. The attention output projection sends four column tiles as they become available and starts each inter-host reduction after its intra-host sum completes. This preserves the FP32 sum order and uses the existing weight and VMEM layouts. The validated specialization omits runtime bounds checks while retaining semaphore checks. TP16 single-request serving passes host control arrays directly into the compiled four-token chunk, avoiding a separate device-array dispatch for each chunk. The token dependency and two-chunk asynchronous pipeline remain on device.

The verifier orders distinct experts by the first row that requests them and evaluates them in pairs. Paired gate/up projections share an input; block-diagonal activations keep the two down-projection results separate. Each row starts its exchange as soon as its routed results are ready, overlapping communication with later experts. Attention-output and unpool gathers transport values at their existing BF16 rounding points; reductions retain FP32 arithmetic. Attention uses width buckets, and smaller LM-head buffers leave more VMEM available for longer contexts.

The verifier normalizes embedding rows on their owning shards, assembles the FP32 rows inside the kernel, and overlaps that exchange with the first Q/O weight loads. Embedding weights are sharded across TP16.

For single-sequence TP16 verification, each completed row is exchanged in one chunk per routed expert. Intra-host reduction, inter-host transfer, and normalization overlap while preserving the original FP32 sum and weighted-mix order. Each chunk has separate DMA counters. Each prefetched expert pair shares one weight-transfer counter: a single wait consumes the combined byte count of its two gate/up and two down copies. Runtime bounds checks are omitted for this validated speculative specialization; semaphore checks remain enabled. These changes add no weight storage or quantization.

The TP16 draft uses a layer loop with the complete gate/up projection in VMEM and gathers the two selected embedding rows inside that kernel. The embedding table remains sharded. Weight, activation, cache, and reduction arithmetic retain their existing precision; BF16 transports preserve the existing rounding points. Every acceptance length uses seven proposals and eight verification rows, with `steps-per-call=4`.

<details>
<summary>TP16 launch and evaluation</summary>

```bash
JOBID=<two-node-job> bash scripts/eval_mpk_gsm8k.sh --tp 16 --spec
JOBID=<two-node-job> bash scripts/eval_mpk_gsm8k.sh --tp 16
JOBID=<two-node-job> bash scripts/eval_mpk_gsm8k.sh --tp 16 --batch-size 8
JOBID=<server-job> bash scripts/eval_mpk_gsm8k.sh --tp 16 --reuse-server

JOBID=<idle-two-node-job> bash scripts/benchmark_mpk.sh --tp 16
JOBID=<idle-two-node-job> bash scripts/benchmark_mpk.sh --tp 16 --fixed-acceptance-length 3
JOBID=<idle-two-node-job> bash scripts/benchmark_mpk.sh --tp 16 --fixed-acceptance-length 6

JOBID=<idle-two-node-job> bash scripts/demo_mpk_dspark.sh --no-spec \
  --serve 0.0.0.0:8000 --context 9216 --max-tokens 1000 \
  --model-name muse-spark-tp16
# For each synthetic AL, use this server command instead:
JOBID=<idle-two-node-job> bash scripts/demo_mpk_dspark.sh \
  --draft /path/to/Muse-Spark-Open-DSpark \
  --serve 0.0.0.0:8000 --context 9216 --max-tokens 1000 \
  --model-name muse-spark-tp16 \
  --fixed-acceptance-length <3-or-6>

srun --jobid=<same-job> --overlap --nodes=1 --ntasks=1 \
  /path/to/vllm-bench \
  --backend openai --base-url http://<server-host>:8000 --model muse-spark-tp16 \
  --tokenizer <existing-target-checkpoint> \
  --dataset-name random --random-input-len 8000 --random-output-len 1000 \
  --random-range-ratio 0 --prompt-token-ids --seed 0 \
  --num-prompts 24 --num-warmups 2 --max-concurrency 1 \
  --ignore-eos --temperature 0 --percentile-metrics ttft,tpot --no-steady-state \
  --save-result --save-detailed --result-dir benchmark-results/<mode> \
  --result-filename result.json
```

Set `MPK_PACKED_WEIGHTS_TP16` to your existing TP16 pack, or pass its path with `--weights`. GSM8K defaults to all 1,319 questions; `START` and `END` select a subset. `MODEL_NAME` overrides the served model ID discovered by the evaluator. DSpark evaluations save per-request acceptance statistics in their output directory.

</details>

### TP32

TP32 uses 32 logical devices on 16 chips across four hosts. Its default layout replicates TP8 experts across hosts and supports BS=1 with or without DSpark. Decode attention uses two TP16 replicas, and each host holds a TP8 partition of every expert. The default context length is 4096.

Set `MPK_PACKED_WEIGHTS_TP32` to an existing canonical TP32 pack, or pass its path with `--weights`. Packs created by `scripts/pack_mpk.sh --tp 32` work with both layouts; startup prepares the selected resident layout. The four-host launcher selects TP32 from the allocation; process 0 serves. Run each server separately on an idle allocation.

Set `MPK_TP32_EXPERT_PARALLEL=1` to select TP8×EP4: each host owns one quarter of the experts, sharded across that host's eight devices. Attention uses TP32 query heads with shared KV heads. Prefill, non-speculative decoding, and DSpark verification share the same resident expert partition and combined BF16 KV cache. This layout supports non-speculative maximum BS=1/2/4/8 and 8,000-input/1,000-output requests with `--context 9216`; DSpark supports BS=1. The default batching wait is 10 ms and applies only to non-speculative BS>1. Both layouts preserve the checkpoint's quantization and BF16 caches. The ENV selects only the 32-device backend; standalone TP8 and TP16 use their own entrypoint.

<details>
<summary>Launch non-speculative or DSpark TP32</summary>

```bash
JOBID=<four-node-job> bash scripts/demo_mpk_tp32.sh \
  --serve 0.0.0.0:8000 --max-tokens 1000 \
  --model-name muse-spark-tp32

JOBID=<four-node-job> bash scripts/demo_mpk_tp32.sh \
  --draft /path/to/Muse-Spark-Open-DSpark \
  --serve 0.0.0.0:8000 --max-tokens 1000 \
  --model-name muse-spark-tp32

MPK_TP32_EXPERT_PARALLEL=1 JOBID=<four-node-job> bash scripts/demo_mpk_tp32.sh \
  --serve 0.0.0.0:8000 --context 9216 --max-tokens 1000 --batch-size 8 \
  --model-name muse-spark-tp32
```

</details>

<details>
<summary>Reproduce the non-speculative 8k/1k concurrency sweep</summary>

Launch the TP8×EP4 non-speculative server above with `MPK_TP32_EXPERT_PARALLEL=1` on a four-host allocation. Run the installed `vllm-bench` on a compute host that can reach process 0. For each chosen maximum server batch size, sweep concurrency 1/2/4/8 with 2×C warmups and 4×C measured requests.

```bash
for C in 1 2 4 8; do
  vllm-bench --backend openai --base-url http://<serving-host>:8000 \
    --model muse-spark-tp32 --tokenizer /path/to/Muse-Spark-Open-816B-A42B-NVFP4 \
    --dataset-name random --random-input-len 8000 --random-output-len 1000 \
    --random-range-ratio 0 --prompt-token-ids --seed 0 \
    --max-concurrency "${C}" --num-warmups "$((2 * C))" --num-prompts "$((4 * C))" \
    --ignore-eos --temperature 0 --percentile-metrics ttft,tpot --no-steady-state \
    --save-result --save-detailed --result-dir benchmark-results/tp32 \
    --result-filename "c${C}.json"
done
```

</details>

<details>
<summary>Reproduce BS=1, ISL=1/OSL=1000 for non-speculative decoding and synthetic AL=3/6</summary>

Each command starts its server, runs two warmups and 24 measured requests at concurrency one, and shuts down the server. Run sequentially on one allocation or concurrently on separate four-host allocations.

```bash
JOBID=<four-node-job> bash scripts/benchmark_mpk.sh --tp 32
JOBID=<four-node-job> bash scripts/benchmark_mpk.sh --tp 32 --fixed-acceptance-length 3
JOBID=<four-node-job> bash scripts/benchmark_mpk.sh --tp 32 --fixed-acceptance-length 6

JOBID=<four-node-job> bash scripts/eval_mpk_gsm8k.sh --tp 32
JOBID=<four-node-job> bash scripts/eval_mpk_gsm8k.sh --tp 32 --spec
MPK_TP32_EXPERT_PARALLEL=1 JOBID=<four-node-job> \
  bash scripts/eval_mpk_gsm8k.sh --tp 32 --batch-size 8
```

</details>

## Source layout

- `pack_mpk.py` and `scripts/pack_mpk.sh`: CPU weight packing for Muse Spark TP8/TP16/TP32 and its Slurm launcher.
- `gcs_parallel_pack.py`: reusable parallel GCS writer and wrapper for packing scripts; configured through `.env`.
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
- `mpk/__init__.py`: Muse-Spark model configuration and reference equations.
- `mpk/load.py`: Muse-Spark tokenizer, raw-byte safetensors streaming (bit-exact
  NVFP4/F8/BF16), TP8/TP16/TP32 weight packing, packed containers, and device
  placement.
- `mpk/decode_megakernel.py`: Muse-Spark TP8/TP16 prefill, persistent single-token and batched decode, and eight-position verification.
- `mpk/decode_megakernel_32.py`: both four-host layouts: default replicated TP8 experts with BS=1 decode and DSpark verification, plus ENV-selected TP8×EP4 experts with prefill, BS=1/2/4/8 decode and DSpark verification.
- `mpk/dspark.py`: DSpark checkpoint loading, reference and persistent draft
  kernels, acceptance and cache updates, and the four-round serving pipeline.
- `demo_kimi_dspark.py` and `demo_qwen_dflash.py`: Kimi and Qwen loading and serving.
- `demo_mpk_dspark.py`: TP8/TP16 loading, prefill, non-speculative batching, DSpark, and serving.
- `demo_mpk_tp32.py`: TP32 loading, non-speculative decoding, DSpark, and serving.
- `mpk/chat.py`: shared Muse Spark chat formatting and token streaming.
- `collectives32.py` and `pool_alias.py`: TPU collective primitives and Pallas
  VMEM aliasing support used by Kimi.
- `openai_server.py` and `distributed_entry.py`: shared HTTP protocol support
  and Kimi multi-host process startup.
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
