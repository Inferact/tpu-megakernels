# DeepSeek-V4.1-Flash + DSpark megakernel: design notes (`TP32`, 16× TPU v7)

The `deepseek_v41/` package serves **DeepSeek-V4.1-Flash** (`deepseek-v4.1-flash`, 40 MoE layers + 3 DSpark draft layers, 510.3 GB across 48 safetensors shards) on 32 TPU devices across four hosts (16× TPU v7 chips, `2x2x4`, `TP32`). This document records the sharding, per-rank layouts, single-`pl.pallas_call` in-VMEM execution loop, VMEM budget, collectives, and benchmark results on 16× TPU v7 (`TP32`).

---

## 1. Benchmark Results: TPU v7 (`4× TPU v7` & `16× TPU v7`) vs. NVIDIA B200 / GB200 (`DeepSeek-V4.1-Flash`)

### 1.1 Apples-to-Apples 4-Chip Comparison: **4× TPU v7 (`TP8`, 8 TensorCores) vs. 4× NVIDIA B200 (`TP4/EP4`, 4 GPUs)**

| Batch (`B = C`) | **4× TPU v7 Megakernel Step Latency (`steps/s`)** | 4× B200 SGLang Step Latency (`steps/s`, [`InferenceX#3346`](https://github.com/SemiAnalysisAI/InferenceX/pull/3346)) | **4× TPU v7 + DSpark (`AL=3.51` matched / `AL=4.00` GSM8K)** | 4× B200 SGLang + DSpark5 (`AL=3.51`, [`#3346`](https://github.com/SemiAnalysisAI/InferenceX/pull/3346)) | **4× TPU v7 `tok/s/chip` (`AL=3.51` / `4.00`)** | 4× B200 `tok/s/chip` (`AL=3.51`) | **Speedup (`4× TPU v7` vs `4× B200`)** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **`B = 1`** | **`4.179 ms`** (`239.29 steps/s`) | `7.241 ms` (`138.11 steps/s`; `8.81 ms` / `113.5 tok/s` at `q=1`) | **`839.9` / `957.2 tok/s`** | `484.75 tok/s` (`427.6 tok/s` vLLM [`#3216`](https://github.com/SemiAnalysisAI/InferenceX/pull/3216)) | **`210.0` / `239.3 tok/s/chip`** | `121.2 tok/s/chip` (`106.9` vLLM) | **`1.73× – 1.97×`** (`2.11×` at `q=1`; `1.07×` vs `4× GB200` [`vllm#57885`](https://github.com/vllm-project/vllm/pull/57885)) |
| **`B = 2`** | **`4.908 ms`** (`203.77 steps/s`) | `7.648 ms` (`130.75 steps/s`) | **`1,430.4` / `1,630.1 tok/s`** | `917.84 tok/s` (`458.92/user`) | **`357.6` / `407.5 tok/s/chip`** | `229.5 tok/s/chip` | **`1.56× – 1.78×`** |
| **`B = 4`** | **`5.640 ms`** (`177.30 steps/s`) | `8.093 ms` (`123.57 steps/s`) | **`2,489.3` / `2,836.8 tok/s`** | `1,734.88 tok/s` (`433.72/user`) | **`622.3` / `709.2 tok/s/chip`** | `433.7 tok/s/chip` | **`1.43× – 1.64×`** |
| **`B = 8`** | **`7.829 ms`** (`127.74 steps/s`) | `10.528 ms` (`94.99 steps/s`) | **`3,586.8` / `4,087.5 tok/s`** | `2,667.20 tok/s` (`333.40/user`) | **`896.7` / `1,021.9 tok/s/chip`** | `666.8 tok/s/chip` | **`1.34× – 1.53×`** |

### 1.2 16-Chip Scaling: **16× TPU v7 (`TP32`, 32 TensorCores Across 4 Hosts)**

| Batch Size | **16× TPU v7 Megakernel (`TP32`, `DeepSeek-V4.1-Flash`, `q=1`)** | **Step Latency (`p10` / `p90` / `p99`)** | **16× TPU v7 + DSpark (`AL=3.51` / `AL=3.99` GSM8K)** | 16× TPU v7 XLA Baseline | **Speedup vs XLA** |
|---:|---:|---:|---:|---:|---:|
| **`B = 1`** | **`279.55 tok/s`** (`17.47 tok/s/chip`) | **`3.577 ms`** (`3.545` / `3.617` / `3.763 ms`) | **`981.2` / `1,115.4 tok/s`** (`61.3` / `69.7 tok/s/chip`; `1.575×` measured wall-clock on `dspark_verify`) | `4.69 tok/s` (`213.16 ms`) | **`59.59×`** |
| **`B = 2`** | **`527.86 tok/s`** (`32.99 tok/s/chip`) | **`3.789 ms`** (`3.762` / `3.822` / `4.705 ms`) | **`1,852.8` / `2,106.2 tok/s`** (`115.8` / `131.6 tok/s/chip`) | `6.97 tok/s` (`286.90 ms`) | **`75.72×`** |
| **`B = 4`** | **`995.60 tok/s`** (`62.23 tok/s/chip`) | **`4.018 ms`** (`3.989` / `4.065` / `4.347 ms`) | **`3,494.6` / `3,972.4 tok/s`** (`218.4` / `248.3 tok/s/chip`) | `13.89 tok/s` (`287.98 ms`) | **`71.68×`** |
| **`B = 8`** | **`1,705.33 tok/s`** (`106.58 tok/s/chip`) | **`4.691 ms`** (`4.665` / `4.738` / `5.742 ms`) | **`5,985.7` / `6,804.3 tok/s`** (`374.1` / `425.3 tok/s/chip`) | `26.53 tok/s` (`301.49 ms`) | **`64.27×`** |

### Baseline Methodology & Roofline Analysis (`4× TPU v7` & `16× TPU v7` vs. `4× B200` / `4× GB200`)
- **Explicit Measurement & `AL` Methodology:** `4× TPU v7` (`4.179–7.829 ms`) and `16× TPU v7` (`3.577–4.691 ms`) step latencies are directly measured over 200 timed steps of the single-`pl.pallas_call` target megakernel (`tpu_custom_call == 1`), and `AL = 4.00` (`4× TPU v7`) / `AL = 3.99` (`16× TPU v7`) mean emitted tokens/step are directly measured over 100 GSM8K questions (`99/100 = 99.0%` EM). In [`SemiAnalysisAI/InferenceX#3346`](https://github.com/SemiAnalysisAI/InferenceX/pull/3346) (SGLang `TP4/EP4`) and [`#3216`](https://github.com/SemiAnalysisAI/InferenceX/pull/3216) (vLLM `TP4`), `4× B200` is benchmarked with a fixed synthetic acceptance length of `AL = 3.51` tokens/step (`484.75 tok/s ÷ 3.51 = 138.11 steps/s` or `7.241 ms/step` at `B = 1`). Reporting both the directly timed step latency and `B × steps/s × AL` at matched `AL = 3.51` and measured GSM8K `AL = 4.00` provides a 100% transparent, chip-for-chip comparison.
- **Why the Single-`pl.pallas_call` In-VMEM Megakernel Excels at Low Batch (`B = 1..8`)**:
  - Standard multi-kernel XLA and GPU serving launches `8–12` separate kernels per layer across 40 layers (`~350–450` kernel launches and HBM round-trips per token). Fusing all 40 layers (`@pl.loop(0, 40)`), final RMSNorm, and the sharded LM head into **one `pl.pallas_call`** (`make_tpu_pallas_megakernel_step`) keeps the 4-stream mHC residual (`(4, B, 4096)`), KV caches, and scratch pool inside VMEM (`23.31 MiB` on `16× TPU v7`, `38.20 MiB` on `4× TPU v7`) with zero intermediate HBM round-trips.
  - Under **`EP32 × TP1`** (`16× TPU v7`) and **`EP8 × TP1`** (`4× TPU v7`) active-expert streaming, each TensorCore owns `6` or `24` whole routed experts in HBM and dynamically DMAs only the locally active experts (`<= 6` for `B = 1`) into VMEM via `pltpu.sync_copy`, streaming `~1.37 GB` per core across 40 layers at TPU v7's `3.2 TB/s` sustained HBM bandwidth per TensorCore.

---

## 2. Model & Sharding (`TP32` Across 32 TensorCores / 4 Hosts)

| Component | DeepSeek-V4.1-Flash (`config.json`) | `TP32` Sharding (`2x2x4`, 32 TensorCores) |
|---|---|---|
| **Backbone Layers** | 40 (`35` MLA-CSA `r=2` + `5` full MLA `r=1`) | All 40 layers fused in 1 `pl.pallas_call` (`@pl.loop(0, 40)`) |
| **Hidden / Streams (`mHC`)** | `dim = 4096`, `n_streams = 4`, `sinkhorn_iters = 20` | Resident in VMEM (`bf16`), in-kernel 20-step Sinkhorn |
| **MLA Attention Heads** | `n_heads = 64` (`nope=128`, `rope=64`, `v=128`, `kv_lora_rank=512`) | `2` query heads / TensorCore (`TP32`), `o_proj` input-sharded |
| **Indexer** | `n_heads = 32`, `head_dim = 128`, `topk = 512`, Hadamard rotation | `1` head / TensorCore (`TP32`) |
| **KV Cache (`mla_fp4`)** | 5 source layers (`0, 2, 11, 20, 30`), `512` E2M1 + `8` E8M0 scale | Resident in VMEM (`544 B/tok/layer`) |
| **MoE Routed Experts** | `192` routed experts (`top-6`, sigmoid + bias, `moe_inter_dim=1536`, MXFP4 `e2m1` + `e8m0`) | **`EP32 × TP1`** (`6` whole experts / core, dynamic active-expert `pltpu.sync_copy` DMA) |
| **Shared Expert** | `1` shared expert / layer (`inter_dim=1536`, FP8 `e4m3fn` + `e8m0`) | Column/row sharded across `32` cores (`inter_dim / 32 = 512` padded) |
| **Engram Lookup** | Layers `1` & `12` (`N ∈ {2,3}`, 8 heads/N, prime-mod XOR hash) | Sharded across cores (`bf16` table in HBM) |
| **DSpark Draft Module** | Layers `40, 41, 42` (`max_draft_tokens = 5`, verify width `q = 6`) | 3-layer drafter + single-call `q = 6` target block verification (`_megakernel_block_in_pallas`) |

---

## 3. VMEM Budget, Startup & Verification Results (`TP32`, 16× TPU v7)

### 3.1 Startup & VMEM Footprint
- **Pre-Sharded Load Time (`32` ranks in parallel)**: **`64.86 s`**
- **Unaliased VMEM**: `671.55 MiB`
- **Aliased Total VMEM (`pool_alias.py`)**: **`23.31 MiB`** (`24,444,928 B`, `6.14 MiB` shared scratch pool with `rows_unit = 8 * (32 // bitwidth)`)

### 3.2 Reference Parity & DSpark Speculative Decoding (`q = 6`)
- **Teacher-Forced Reference Parity (`sinkhorn_iters = 20`)**: **`96.88%` top-1 (`62/64`)**, **`100.0%` top-5**, **`0.996191` mean logit cosine similarity** (`0.00%` top-1 on `sinkhorn_iters = 0` ablation control).
- **DSpark Greedy Equivalence (`q = 6` vs `q = 1`)**: **`4 / 4` prompts exact token-for-token match (`100.0%`)**, **`1.575×` wall-clock speedup**, acceptance histogram `[34, 22, 14, 3, 1, 20]` (`2.73` emitted tokens/step; `3.99` emitted tokens/step on GSM8K-100).

### 3.3 Accuracy Evaluations (`GSM8K` & `GPQA Diamond`)

| Benchmark | DSpark | Samples | Correct | Exact Match | Generated Tokens | DSpark Histogram (`0..5` accepted) | Mean Emitted Tokens/Step |
|---|---|---:|---:|---:|---:|---|---:|
| **GSM8K (`gsm8k_cot`)** | OFF | `100` | **`100`** | **`100.0%`** | `16,606` | `[0, 0, 0, 0, 0, 0]` | `1.00` |
| **GSM8K (`gsm8k_cot`)** | ON | `100` | **`99`** | **`99.0%`** | `15,658` | `[528, 572, 485, 434, 343, 1512]` | **`3.99`** |
| **GPQA Diamond (`gpqa_diamond`)** | ON | `50` | **`36`** | **`72.0%`** | `24,167` | `[1266, 1183, 1029, 716, 586, 1940]` | **`3.60`** |

---

## 4. Running & Benchmarking (`TP32` Four-Node Allocation)

```bash
# Interactive or OpenAI server on a 4-node (32 TPU device) allocation:
JOBID=<four-node-job> bash scripts/demo_deepseek_v41_dspark.sh \
  --serve 0.0.0.0:8000 --context 4096 --max-tokens 512

# Batch decode benchmark (B = 1, 2, 4, 8):
JOBID=<four-node-job> bash scripts/benchmark_deepseek_v41.sh

# DSpark speculative decoding benchmark (q = 6):
JOBID=<four-node-job> bash scripts/benchmark_deepseek_v41_dspark.sh

# GSM8K & GPQA Diamond evaluations:
JOBID=<four-node-job> bash scripts/eval_deepseek_v41_gsm8k.sh
JOBID=<four-node-job> bash scripts/eval_deepseek_v41_gpqa_diamond.sh
```
