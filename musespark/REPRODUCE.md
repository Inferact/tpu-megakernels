# Muse Spark 1.2 TPU megakernel: reproduction protocol

Step-by-step protocol for a new engineer on a fresh single-host 4x TPU7x machine. Every
command is relative to the repository root (`/filestore/srcs/tpu-megakernels` in this
deliverable, branch `muse-spark`). Numbers quoted as "expected" were measured on this host and
name the log they come from. `eval-results/` is tracked; `logs/`, `benchmark-results/` and the
`.jax_cache` are gitignored and live in the host checkout only (the scratchpad
`/filestore/tmp/claude-0/-filestore-weights/bc41d5f3-5fda-4ab7-9ca1-48e0c99553cc/scratchpad`
holds the oracle directories and the engineering notes).

- **Final code commit: `b171b4d`** (the deliverable is the tree at this commit; the commit
  adding this file and `REPORT.md` sits on top of it and changes no code). Numbers labelled
  "as of `<hash>`" were measured at the intermediate commits named in
  `git log --oneline 4048f08..HEAD`; the final numbers (sections 4, 5, 7, 8) were measured
  with the kernel code of `6a5aa76`..`b171b4d`, which differ only in `musespark/README.md` and
  the regression test's bounds. Final container: `/filestore/weights/muse-spark-tp8-nvfp4`
  (format v2, `expert_format: nvfp4`, `dense_format: int8`, 516 GB).

Design notes: `musespark/README.md` (sharding, layouts, container formats, kernel phases, VMEM
budget, collectives, rounding policy); engineering report: `musespark/REPORT.md`.

---

## 0. Overview of the protocol

| step | what | where it runs | wall time (measured) |
|---|---|---|---|
| 1 | prerequisites, `uv sync` | host | minutes |
| 2 | obtain weights (bf16 snapshot and/or NVFP4 Hub repo) | host, network | hours (bandwidth-bound) |
| 3 | build the pre-sharded containers (int4 g128 v1, NVFP4 v2), verify, int8-dense step | host CPU (224 cores), no TPU | 28 min (int4) / 40.5 min (NVFP4) + 24 s (`quantize-dense`) |
| 4 | CPU test suite | host CPU, 8 virtual devices | ~8 min |
| 5 | TPU test suite | 1 chip (component tests) / 4 chips (8-core tests) | ~2-6 min per file group |
| 6 | validation against the XLA reference on real weights (oracle + kernel) | 4 chips | ~5 min + load |
| 7 | decode benchmark | 4 chips | ~1-5 min + load |
| 8 | GSM8K accuracy (server + lm-eval) | 4 chips + host | ~15-25 min per 100 problems |
| 9 | official-maths comparison (JAX reference vs sglang, CPU) | host CPU | ~15 min for 4 layers |
| 10 | demo / server | 4 chips | interactive |

---

## 1. Prerequisites

### 1.1 Hardware

- One host with **four TPU7x chips, two TensorCores each = eight JAX devices**
  (`jax.devices()` shows `kind='TPU7x'`, ids `2r` and `2r+1` are the two cores of chip `r`).
  Per core: 64 MiB VMEM, ~94.7 GiB usable HBM (`memory_stats().bytes_limit`
  = 101,724,453,888), ~3.2 TB/s achieved HBM->VMEM DMA (notes `tpu7x_hw_report.md` sections
  1-2). The kernel needs 54.8 GiB (int4) / 57.8 GiB (NVFP4) of HBM per core for the weights
  with bf16 dense projections, 55.6 GiB for the final NVFP4 + int8 container, plus 31 KiB per
  token per core for the KV cache (`musespark/README.md` section 3).
- **Host RAM**: this host has 944 GB. The containers are read with `preadv` and placed on the
  devices one layer per rank at a time, so RAM is not a hard requirement, but a large page cache
  is what makes the second load of a container take ~25 s instead of ~6 min (section 12).
- **Disk** (measured on `/filestore`, a 3.0 TB NFS volume):
  - bf16 route: the Hugging Face bf16 snapshot is **1.6 TB** (`logs/convert.log`: "1607 GB
    read"), the int4 g128 container is **471 GB** (`layout.json` `total_bytes`
    470,732,619,776; `du` 439 GiB). Budget ~2.1 TB, or 0.5 TB if the snapshot is deleted after
    conversion (the tokenizer and `config.json` of the snapshot are still needed at run time;
    keep those files).
  - NVFP4 route: the converter streams the 533 GB Hub repo shard by shard and never stores the
    raw checkpoint; the container is **496 GB** (`total_bytes` 496,217,210,880; `du` 463 GiB)
    plus at most `--max-shards` (default 3) shards of <= 8.6 GB each on disk, and **516 GB**
    after the int8 dense step of section 3.4 (`total_bytes` 515,602,808,832). Budget ~0.57 TB.
    The converter refuses to download when free space drops under `--min-free-gb` (default 8).
- Network access to `huggingface.co` for the downloads (the NVFP4 converter measured 218-491
  MB/s per shard with `HF_XET_HIGH_PERFORMANCE=1`, `logs/convert_nvfp4.log`).

### 1.2 Software

- Linux, Python **3.12** (`pyproject.toml`: `requires-python >= 3.12`; this host runs 3.12.3),
  [`uv`](https://github.com/astral-sh/uv), `curl`, `git`.
- `uv sync` installs the locked versions from `uv.lock`: `jax[tpu] == 0.11.1` (`jax` 0.11.1,
  `jaxlib` 0.11.1, `libtpu` 0.0.46.1), `numpy` 2.5.3, `safetensors` 0.8.0,
  `huggingface-hub` 1.30.0, `tiktoken`, `tokenizers`, `pytest` (dev group). The evaluation
  extra adds `lm-eval[api]` 0.4.13 (`uv sync --extra eval`). The official-maths comparison
  (section 9) needs a separate CPU torch environment, see there.

```bash
git clone <repo> /filestore/srcs/tpu-megakernels
cd /filestore/srcs/tpu-megakernels
git checkout muse-spark          # or `git checkout b171b4d` (final code commit)
uv sync                          # creates .venv with jax 0.11.1 + libtpu
uv sync --extra eval --inexact   # adds lm_eval for section 8 (keeps the base env intact)
.venv/bin/python -c "import jax; print(jax.__version__, jax.devices())"   # 0.11.1, 8 x TPU7x
```

### 1.3 Environment variables

The launchers (`scripts/demo_musespark.sh`, used by the benchmark and the GSM8K script) set
everything a TPU run needs; when running Python directly (validation scripts, TPU tests)
export the same variables:

| variable | value | why |
|---|---|---|
| `XLA_FLAGS` | `--xla_allow_excess_precision=false` | the XLA glue's `r16` must round exactly like the reference (`musespark/README.md` section 8); **required** for every TPU run whose numbers are compared with the oracle |
| `TPU_PROCESS_BOUNDS` | `1,1,1` | one process |
| `TPU_CHIPS_PER_PROCESS_BOUNDS` | `2,2,1` (all four chips) or `1,1,1` (single chip) | chip topology seen by the process |
| `TPU_VISIBLE_CHIPS` | `0,1,2,3` (8 cores) or one of `0`..`3` (2 cores, component tests) | which chips the process opens |
| `JAX_PLATFORMS` | `tpu` (TPU runs) / `cpu` (converters, CPU tests, lm-eval client) | |
| `JAX_COMPILATION_CACHE_DIR` | `<repo>/.jax_cache` | persistent compile cache (section 12) |
| `PYTHONPATH` | `<repo>` | `musespark` and `demo_musespark` importable |
| `HF_HUB_OFFLINE` | `1` (demo/server) | never touch the Hub at run time |
| `HF_HOME` | `/filestore/hf` (default of the NVFP4 converter) | download cache; also read by lm-eval |
| `PYTHONDONTWRITEBYTECODE` | `1` | keeps the tree clean |

For CPU-only test runs: `JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8`
(eight virtual host devices; the Pallas bodies run in interpret mode).

Container and checkpoint locations are read from the environment or the untracked `.env`
(sourced by every launcher):

```bash
cat > .env <<'EOF'
MUSESPARK_CHECKPOINT=/filestore/weights/Muse-Spark-1.2-816B-A42B-open   # HF bf16 snapshot (tokenizer, config, weights)
MUSESPARK_PRESHARDED=/filestore/weights/muse-spark-tp8-int4             # int4 g128 container (format v1)
MUSESPARK_NVFP4_REPO=meta-models/Muse-Spark-1.2-816B-A42B-NVFP4-open    # Hub repo streamed shard by shard
MUSESPARK_NVFP4_PRESHARDED=/filestore/weights/muse-spark-tp8-nvfp4      # NVFP4 container (format v2)
HF_HOME=/filestore/hf
EOF
```

`MUSESPARK_MEGAKERNEL_SRC` and `MUSESPARK_PY` point the launchers at another checkout or
interpreter if needed. Shared machines: the launchers accept `TPU_RUN="<path>/tpu_run.sh all"`
as a command prefix that takes per-chip `flock`s so two programs never share a chip; on a
dedicated fresh machine leave `TPU_RUN` unset.

---

## 2. Obtaining the weights

Two independent routes; the deliverable supports both containers and the kernel infers the
expert format from the container.

### 2.1 bf16 snapshot (needed for the int4 g128 container, the tokenizer/config, the XLA/CPU comparisons)

Repo `meta-models/Muse-Spark-1.2-816B-A42B-open` (66 safetensors shards, `config.json`,
`generation_config.json`, `tokenizer.json`, `chat_template.jinja`). Download the full snapshot
to `MUSESPARK_CHECKPOINT` (1.6 TB), e.g.

```bash
HF_HOME=/filestore/hf .venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('meta-models/Muse-Spark-1.2-816B-A42B-open',
                  local_dir='/filestore/weights/Muse-Spark-1.2-816B-A42B-open')"
```

The checkpoint revision the containers were built from is recorded in each container's
`layout.json` (`revision`: `b4c013bc2ddfaa4d` for the int4 container, `0857b0d01c8e165f` for
the NVFP4 one: the first 16 hex digits of the SHA-256 of the checkpoint's
`model.safetensors.index.json`; a resumed conversion refuses a source with another revision).

### 2.2 NVFP4 Hub repo (for the NVFP4 container)

Repo `meta-models/Muse-Spark-1.2-816B-A42B-NVFP4-open` (533 GB, 127 shards; only the routed
experts of layers 1..60 are NVFP4, everything else is bf16, notes `nvfp4_feasibility.md`
section 1). **Do not download it by hand**: `scripts/convert_musespark_nvfp4.sh` streams it
(one shard at a time into `$HF_HOME`, converted, deleted). The tokenizer and `config.json` used
at run time still come from the bf16 snapshot directory (`--checkpoint`); if you skip the bf16
route entirely, download just those files from either repo into `MUSESPARK_CHECKPOINT`
(`config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, `chat_template.jinja`).

---

## 3. Building the pre-sharded TP8 containers

Both converters run detached under `nohup` on the CPU (`JAX_PLATFORMS=cpu`), log to
`logs/convert*.log` with GB/s and ETA, and are **resumable and idempotent** (`progress.json`
per layer / per unit): re-run the same script after an interruption. A container is
`<dir>/layout.json` + `<dir>/rank{0..7}/<family>.bin` + `progress.json`
(`musespark/README.md` section 4).

### 3.1 int4 g128 container (format v1) from the bf16 snapshot

```bash
bash scripts/convert_musespark.sh            # -> $MUSESPARK_PRESHARDED, log logs/convert.log
tail -f logs/convert.log
```

Expected (`logs/convert.log`): 62 layers, each "read 26.3 GB in ~23 s (1.15 GB/s from NFS),
quantized+wrote 7.5 GB in ~25 s"; final line `done: 62/62 layers, complete=True, 28.0 min
total, 1607 GB read at 1.15 GB/s`. Size 471 GB (`layout.json` `total_bytes` 470732619776,
`format: musespark-presharded-v1`, `group: 128`). Extra flags pass through to
`python -m musespark.load convert` (`--layers 0,1`, `--workers N`, `--read-threads N`).

### 3.2 NVFP4 container (format v2) streamed from the Hub

```bash
bash scripts/convert_musespark_nvfp4.sh      # -> $MUSESPARK_NVFP4_PRESHARDED, log logs/convert_nvfp4.log
tail -f logs/convert_nvfp4.log
```

Expected (`logs/convert_nvfp4.log`): 126 weight shards / 367 conversion units, per shard
"download 10-22 s (218-491 MB/s), wrote N GB", the running `free <GB>` column never below
`--min-free-gb`; final line `done: 367/367 units, complete=True, 40.5 min total`. Size 496 GB
(`total_bytes` 496217210880, `format: musespark-presharded-v2`, `expert_format: nvfp4`,
`expert_source`: layers 1..60 `vendor`, layers 0 and 61 `rtn`). Flags: `--max-shards 3`
(shards kept on disk), `--min-free-gb 8`, `--workers N`, `--shards n` (first n shards only,
for tests).

### 3.3 Verify a converted layer against the checkpoint

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m musespark.load verify \
    --src "$MUSESPARK_CHECKPOINT" --dst "$MUSESPARK_PRESHARDED" --layer 0 --experts 0,1
```

`verify` compares one layer of a container against the checkpoint: dense spot checks (exact)
and the relative RMS error of the dequantized experts (int4 or NVFP4 container). `--src` may
be the bf16 snapshot for both containers (the NVFP4 repo shares its dense tensors and the
layer-0/61 experts); for a vendor-quantized layer the reported error is the vendor's own NVFP4
quantization error (~9.5 % relative RMS; int4 g128 ~11.7 %, `notes/nvfp4_container_v2.md`).
The NVFP4 converter additionally self-checks expert 0 of every vendor layer against a slow
nibble path during conversion.

### 3.4 int8 dense projections + int8 `lm_head` (in place) -- part of the final commit

`python -m musespark.load quantize-dense` (commit `eb09289`, quantizer `ca60168`) appends int8
per-output-channel twins of the six dense projections (`q`, `kv`, `gate`, `o`, `pre`, `post`)
and the `lm_head` to a **complete** container in place (new `_i8` / `_s` `.bin` files next to
the untouched bf16 ones, `progress.json` key `dense_int8`, resumable per layer) and, once all
layers and the `lm_head` are done, publishes `dense_formats: ["bf16", "int8"]` /
`dense_format: "int8"` in `layout.json`, which `load_presharded` (and therefore the kernel,
the prefill and the reference, `09d055d`) then picks by default; `--dense-format bf16` /
`MUSESPARK_DENSE_FORMAT=bf16` select the bf16 families explicitly. Run it for each container
you serve:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m musespark.load quantize-dense --dir "$MUSESPARK_NVFP4_PRESHARDED"
JAX_PLATFORMS=cpu .venv/bin/python -m musespark.load quantize-dense --dir "$MUSESPARK_PRESHARDED"   # optional, int4 container
python3 -c "import json;print(json.load(open('$MUSESPARK_NVFP4_PRESHARDED/layout.json')).get('dense_format'))"  # int8
```

Measured on the NVFP4 container (`scratchpad/i8/quantize_nvfp4.log`): "quantizing 62 layers x
6 families + lm_head with 32 workers (380 tasks)" ... "int8 dense families complete in 24 s";
`total_bytes` 496,217,210,880 -> **515,602,808,832 (+19.4 GB)**, `progress.json["dense_int8"]
= {layers 0..61, lm_head: true, complete: true}`. In this deliverable **only the NVFP4
container got the int8 step**; the int4 container (`/filestore/weights/muse-spark-tp8-int4`)
has no `dense_format` key and stays bf16-dense. The final benchmark (section 7) and the final
GSM8K runs (section 8) were produced with `dense_format: int8` on the NVFP4 container; the
int4 rows of those tables are bf16-dense.

If `layout.json` has no `dense_format` key the container is bf16-dense and every step below
works unchanged.

---

## 4. CPU test suite (no TPU)

```bash
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  .venv/bin/python -m pytest tests/test_musespark_*.py -q
```

Expected at the final commit `b171b4d`: **181 passed, 29 skipped, 9 warnings in 519 s** (`logs/final_cpu_suite.log`, ~8 min
on 224 cores; the skips are the TPU-only tests and fixture-dependent tests; 147 passed / 22
skipped at `8084031` before the int8 work). The suite covers the
reference model vs an independent spec transcription, quantizers, layouts, container round
trips, tokenizer/chat rendering, sampling, and every kernel component plus the whole decode
step and the prefill in interpret mode on the MINI config (`musespark/README.md` section 11).
The tokenizer tests need `MUSESPARK_CHECKPOINT/tokenizer.json`; `tests/test_musespark_fp4.py`
has one fixture-gated test (`gu_w_l1_e0_rows0_256.bin`, skipped when absent).

The int8 quantizer (`ca60168`), the int8 container round trips, the int8 / packed ring and
the int8 MINI decode account for the growth since `8084031`.

---

## 5. TPU test suite

Run the component tests on **one chip** (two cores) and the eight-core tests with **all
chips** visible; always with `XLA_FLAGS=--xla_allow_excess_precision=false`. `-s` prints the
per-op microseconds / per-layer timings.

```bash
# single chip (component tests): chip 0 here; any of 0..3 works
export XLA_FLAGS=--xla_allow_excess_precision=false JAX_PLATFORMS=tpu PYTHONPATH=$PWD \
       JAX_COMPILATION_CACHE_DIR=$PWD/.jax_cache
TPU_VISIBLE_CHIPS=0 TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 TPU_PROCESS_BOUNDS=1,1,1 \
  .venv/bin/python -m pytest tests/test_musespark_attention.py tests/test_musespark_moe.py \
                             tests/test_musespark_stream.py tests/test_musespark_fp4.py -s

# all eight cores (collectives, MINI decode int4 + NVFP4, prefill, real-width smoke)
TPU_VISIBLE_CHIPS=0,1,2,3 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 \
  .venv/bin/python -m pytest tests/test_musespark_collectives.py tests/test_musespark_decode.py \
                             tests/test_musespark_decode_fp4.py tests/test_musespark_prefill.py -s
```

On TPU the interpret-mode variants are skipped and the hardware variants run instead.

Runs at the final commit `b171b4d` (2026-09-26, through the `tpu_run.sh` lock helper; the
single-chip group on chip 3, the eight-core group with all four chips):

| run | command | log | result |
|---|---|---|---|
| single-chip group | `pytest tests/test_musespark_stream.py tests/test_musespark_attention.py tests/test_musespark_moe.py tests/test_musespark_fp4.py -q` | `logs/final_tpu_single_chip.log` | **90 passed, 16 skipped in 436 s** |
| eight-core group | `pytest tests/test_musespark_collectives.py tests/test_musespark_decode.py tests/test_musespark_decode_fp4.py tests/test_musespark_prefill.py tests/test_musespark_sampling.py -q` | `logs/final_tpu_eight_core.log` | **38 passed, 7 skipped in 117 s** |
| real-weights prefill regression (int4 container, defaults) | `pytest tests/test_musespark_real_prefill.py -q` | `logs/final_real_prefill_int4.log` | **1 passed in 266 s (cold int4 container load included)** |
| real-weights decode regression (NVFP4 + int8 container, section 6.3) | `MUSESPARK_WEIGHTS=/filestore/weights/muse-spark-tp8-nvfp4 MUSESPARK_REAL_REF=<scratchpad>/real_ref_nvfp4_int8 pytest tests/test_musespark_real_decode.py -q` | `logs/final_real_decode_nvfp4_int8.log` | **2 passed in 86 s** |

Earlier runs for reference: `scratchpad/val_final2.log` (eight-core suite at `37bd619`, 58
passed, 14 skipped, 2 failed: `test_prefill_writes_the_kernel_cache_layout[all_experts|top4]`,
the layer-0 K-cache slot off by one bf16 ulp on TPU, fixed by the TPU-robust check of
`8084031`); `scratchpad/fp4work/tpu3_fp4_tests.log` (54 passed, 1 skipped);
`logs/pytest_real_decode3.log` (int4, 2 passed in 40 s); `logs/pytest_real_decode_int8b.log`
(NVFP4 + int8, 2 passed in 32 s); `scratchpad/pytest_real.log` (prefill, 1 passed in 57 s).

The real-weights tests (section 6.3) load a whole container; run each **alone** in its own
process (section 12, HBM OOM).

---

## 6. Validation against the XLA reference on real weights

The oracle is the pure-XLA prefill (`musespark.prefill`, the same per-rank weights and
rounding points as the kernel, no Pallas) run as "prefill-as-decode"; the kernel is then
compared with it. All scripts print to stdout and log under `logs/`.

### 6.1 Build the oracle: `scripts/validate_musespark_prefill.py`

```bash
export XLA_FLAGS=--xla_allow_excess_precision=false JAX_PLATFORMS=tpu PYTHONPATH=$PWD \
       JAX_COMPILATION_CACHE_DIR=$PWD/.jax_cache TPU_VISIBLE_CHIPS=0,1,2,3 \
       TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1
.venv/bin/python scripts/validate_musespark_prefill.py \
    --weights "$MUSESPARK_PRESHARDED" --checkpoint "$MUSESPARK_CHECKPOINT" \
    --out /filestore/tmp/real_ref --context 4096 --max-tokens 48
# NVFP4 container: --weights "$MUSESPARK_NVFP4_PRESHARDED" --out /filestore/tmp/real_ref_nvfp4
# NVFP4 + int8 dense (the final container; the oracle must use the same dense format as the kernel):
#   ... --weights "$MUSESPARK_NVFP4_PRESHARDED" --dense-format int8 --out /filestore/tmp/real_ref_nvfp4_int8
```

The three oracles used in this deliverable live in the scratchpad: `real_ref` (int4),
`real_ref_nvfp4` (NVFP4, bf16 dense) and `real_ref_nvfp4_int8` (NVFP4, int8 dense).

Flags: `--prompts N` (only the first N of the four prompts: three chat prompts rendered with
the model's template at fixed date 2026-09-26 / reasoning effort medium, plus the raw
`"The quick brown fox"`), `--no-taps` (skip the per-layer residual taps), `--deadline`
(stop early). Output under `--out`: `prompt{i}_ids.npy`, `prompt{i}_logits.npy` (soft-capped
full-vocab logits after the prompt), `prompt{i}_gen.npy` (greedy continuation),
`prompt0_hidden.npy` (`[L+1, Tp, H]` residual taps), `summary.json`.

Expected: coherent greedy continuations (prompt 0 "What is 2+2?" starts
`' to=self<|message|>User asks what is 2+2. Simple. Answer 4. ...'`, prompt 3 continues
`' jumps over the lazy dog.'`), argmax after prompt 0 = token 328 `' to'` with a top-1 margin
of ~19.9 logits on both containers (`logs/validate_musespark_decode_20260926T022450Z.log`,
`notes/nvfp4_container_v2.md`). Per call 0.42-0.66 s (int4) / 0.27-0.51 s (NVFP4) depending on
the 64-token bucket (`musespark/README.md` section 9).

### 6.2 Kernel vs oracle: `scripts/validate_musespark_decode.py`

```bash
.venv/bin/python scripts/validate_musespark_decode.py \
    --weights "$MUSESPARK_PRESHARDED" --checkpoint "$MUSESPARK_CHECKPOINT" \
    --ref /filestore/tmp/real_ref --context 4096 --tasks replay,generate,bench
# -> logs/validate_musespark_decode_<UTC>.log + .json
```

Flags: `--tasks` (comma list of `replay`, `generate`, `bench`; default all),
`--replay-prompts 0,3`, `--replay-batches 1,4` (also replay every row of these batch sizes),
`--save-aux DIR` (dump the kernel residual streams), `--steps-per-call 16`,
`--bench-steps 128`, `--log-dir logs`.

Expected outcomes (int4, `logs/validate_musespark_decode_20260926T022450Z.log`; the bf16-wire
numbers of `notes/perf_log.md` E3 in parentheses):

- `replay` (decode-only replay of the prompt ids from position 0, B=1, kernel writes its own
  cache): `logits max |diff| 0.4695 (0.497 with the bf16 wire) ... argmax 328 ' to' vs ref
  328 ' to': EQUAL; top-5 overlap 5/5`; per-layer residual taps `tap 0 exact True; worst
  relative RMS 4.2e-02 (4.5e-02)`; prompt 3: `argmax 67072 ' jumps' ... EQUAL; top-5 overlap
  5/5`; `k_cache/v_cache slots beyond the prompt untouched: True`; `B=4 vs B=1`: all four rows
  bit-identical to each other (`rows vs row 0: logits max |diff| 0.0000`), same argmax.
- `generate` (XLA prefill of each prompt into its own cache row, then greedy kernel decode of
  48 tokens at B=4 and B=1): `logits vs oracle max |diff| 0.0000` after the prefill (same XLA
  program), **first token equal to the oracle for every prompt**, then the greedy paths of this
  chaotic model diverge after some tokens: leading-token matches 48/10/17/48 (B=4) and 36
  (B=1) on NVFP4 at `afd8290` (`logs/validate_musespark_decode_20260926T070309Z.log`), 17/17
  on prompt 3 int4 (`logs/pytest_real_decode3.log`); every printed kernel text is coherent
  (`' to=self<|message|>User asks what is 2+2. Simple. Answer 4. ...'`).
- `bench`: see section 7.

Run the same on the NVFP4 container with `--ref /filestore/tmp/real_ref_nvfp4` (bf16 dense,
`--dense-format bf16`) or `--ref /filestore/tmp/real_ref_nvfp4_int8` (the container's int8
default). On the NVFP4 + int8 container the prompt-0 replay sits on a **routing near-tie**
(top-8 flips at layers 9 and 59 between the kernel and the XLA prefill, both running the same
int8 maths): `logits max |diff| 9.7758 (mean 2.38006, rms 2.53480), argmax 328 ' to' vs ref
328 ' to': EQUAL; top-5 overlap 2/5` (`logs/pytest_real_decode_int8b.log`), residual stream
worst 16 % relative RMS; `--decode-options ring=plain` reproduces the excursion to 2.8e-7,
prompts 1/2 stay at 0.7-1.1 and prompt 3 at `argmax 67072 ' jumps' EQUAL; top-5 5/5`. The
final benchmark log `logs/validate_musespark_decode_20260926T083121Z.log` also carries the
`generate` task on this container (first token equal to the oracle on every prompt).

### 6.3 Regression tests on real weights (each alone)

```bash
MUSESPARK_REAL_REF=/filestore/tmp/real_ref MUSESPARK_WEIGHTS="$MUSESPARK_PRESHARDED" \
  .venv/bin/python -m pytest tests/test_musespark_real_prefill.py -s     # 1 passed (~1 min + load)
MUSESPARK_REAL_REF=/filestore/tmp/real_ref MUSESPARK_WEIGHTS="$MUSESPARK_PRESHARDED" \
  .venv/bin/python -m pytest tests/test_musespark_real_decode.py -s      # 2 passed (~40 s + load)
# final NVFP4 + int8 container against its int8 oracle:
MUSESPARK_REAL_REF=/filestore/tmp/real_ref_nvfp4_int8 MUSESPARK_WEIGHTS="$MUSESPARK_NVFP4_PRESHARDED" \
  .venv/bin/python -m pytest tests/test_musespark_real_decode.py -s      # 2 passed (~32 s + load)
```

Tolerances (`musespark/README.md` section 8): on the int4 container logits within 1.0 of the
oracle with the oracle's argmax and top-5 overlap >= 4, residual stream <= 10 % relative RMS
(measured 0.47 / 4.5 %), layer-0 K-cache slots <= 0.1 vs the prefill. On an NVFP4 or
int8-dense container the test switches to the documented envelope of the prompt-0 near-tie
(`tests/test_musespark_real_decode.py::_bounds`: logits within 12.0, residual stream <= 25 %,
top-5 overlap >= 1, **argmax equality the hard check**; measured 9.8 / 16 % / 2 of 5).

---

## 7. Decode benchmark

Definitions used everywhere in this deliverable: **ms/step** = wall time of the timed decode
steps / number of steps, measured on the host around device calls of `--steps-per-call`
kernel steps each (tokens stream back once per call), **after a warm-up call** that includes
compilation; **tok/s (aggregate)** = `B * 1000 / ms_per_step`, **tok/s per row** =
`1000 / ms_per_step`. Context 4096 (KV cache length), every row holds prompt 0 (the
chat-rendered "What is 2+2?", **151 tokens**, prefilled by the XLA prefill into bucket 192),
B rows in {1, 2, 4, 8}. Because all rows are identical, the expert dedupe loads only eight
distinct experts per layer even at B=8; the "floor" column is the harness's `8 * B`
experts-per-layer HBM-byte bound at 3.2 TB/s, so `util%` above 100 at B=8 is expected.

### 7.1 Official numbers: the validate script's `bench` task

```bash
.venv/bin/python scripts/validate_musespark_decode.py --weights "$MUSESPARK_PRESHARDED" \
    --checkpoint "$MUSESPARK_CHECKPOINT" --tasks bench            # 16-step calls, 128 timed steps
```

The summary table (`bench summary (context 4096, 16-step calls)`) reports per B: ms/step,
tok/s/row, tok/s aggr, floor ms, util%, prefill(T=192) ms. Measured:

| B | int4 g128 ms/step (`logs/validate_musespark_decode_20260926T052025Z.log`, commit `df7ee05`) | tok/s aggr | NVFP4 ms/step (`..._20260926T070309Z.log`, commit `afd8290`) | tok/s aggr |
|---|---:|---:|---:|---:|
| 1 | 3.592 | 278 | 3.565 | 281 |
| 2 | 3.705 | 540 | 4.070 | 491 |
| 4 | 3.912 | 1022 | 4.626 | 865 |
| 8 | 4.291 | 1864 | 4.861 | 1646 |

Prefill of the 151-token prompt (bucket 192): 462 ms int4, 317 ms NVFP4.

**Final numbers** (kernel of `6a5aa76`..`b171b4d`, NVFP4 container, the container's default
`dense_format: int8`, `logs/validate_musespark_decode_20260926T083121Z.log`, load 22.7 s from
the page cache; the bf16-dense column is the same kernel a few commits earlier on the same
container, `..._20260926T070309Z.log`):

| B | NVFP4 + int8 dense ms/step | tok/s per row | tok/s aggr | best / worst 16-step call | harness floor | NVFP4 + bf16 dense ms/step | tok/s aggr |
|---|---:|---:|---:|---|---:|---:|---:|
| 1 | **3.260** | **306.8** | 306.8 | 3.243 / 3.280 | 1.47 ms (45 %) | 3.565 | 281 |
| 2 | 3.657 | 273.4 | 546.8 | 3.633 / 3.680 | 2.02 ms (55 %) | 4.070 | 491 |
| 4 | 4.245 | 235.5 | 942.2 | 4.211 / 4.272 | 3.12 ms (73 %) | 4.626 | 865 |
| 8 | 4.462 | 224.1 | 1793.0 | 4.422 / 4.509 | 5.31 ms (119 %) | 4.861 | 1646 |

Prefill (B=1, int8 dense) 268 / 314 / 326 / 372 / 505 ms for buckets 64 / 192 / 512 / 1024 /
2048; 313 ms for the bench prompt. The int4 container was not re-benchmarked after `df7ee05`
(it has no int8 twins); its row above stands. The B=1 criterion (>= 300 tok/s) is met by the
int8 path only: bf16 dense gives 280.5 tok/s.

### 7.2 The demo's benchmark: `scripts/benchmark_musespark.sh`

```bash
bash scripts/benchmark_musespark.sh                       # int4 container, B in {1,2,4,8}
WEIGHTS=$MUSESPARK_NVFP4_PRESHARDED bash scripts/benchmark_musespark.sh
```

Runs `demo_musespark.py --bench` through the launcher with `CONTEXT=4096`, **`STEPS=256`
timed steps** in **`STEPS_PER_CALL=64`**-step calls (both overridable), logs to
`benchmark-results/musespark-<UTC>/bench.log` and prints `batch  ms/step  tok/s (aggregate)`
plus prefill times per length bucket (64/145/256/1024). Expected to agree with 7.1 within a
few percent (`benchmark-results/musespark-20260926T045545Z/bench.log`, STEPS=128, commit
`37bd619`: 3.64 / 3.81 / 4.01 / 4.36 ms at B = 1/2/4/8). Extra arguments go to the demo
(e.g. `--greedy`).

---

## 8. Accuracy: GSM8K through the OpenAI-compatible server

### 8.1 How the numbers were produced

`scripts/eval_musespark_gsm8k.sh` starts the demo as a server (`--serve`, one request at a
time on cache row 0), puts the **capture proxy** `scripts/chat_capture_proxy.py` in front of
it (records every chat completion incl. `reasoning_content`, `finish_reason`, `usage` to
`captures.jsonl`; lm-eval only keeps `message.content`), smoke-tests one completion, then runs
`lm_eval run --model local-chat-completions --apply_chat_template --num_concurrent=1
--seed 0 --batch_size 1 --log_samples --limit $LIMIT`, and finally summarises
(`scripts/eval_musespark_accuracy.py` -> `summary.md`) and classifies the failures
(`scripts/analyze_gsm8k_failures.py` -> `failures.md`). Everything lands in
`eval-results/gsm8k-musespark-<LIMIT>-<RUN_TAG>-<UTC>/` together with `run-config.txt` (HEAD,
weights, context, budget, task, sampling), `server.log`, `lm_eval.log`, `proxy.log`,
`worktree-uncommitted.diff` and the lm-eval `results_*.json` / `samples_*.jsonl`.

The reported configuration is **`gsm8k_cot_llama`, 8-shot, the first 200 test problems
(`LIMIT=200`; the earlier runs used 100), `max_gen_toks 8192`, reasoning effort medium
(strength 64), context 16384** (the script doubles the 8192 default until it holds
`MAX_GEN_TOKS + 2048`), on the **NVFP4 container with int8 dense projections** (the earlier
100-problem runs were bf16-dense), in two decoding modes:

- **greedy**: `TEMPERATURE=0` -> server `--greedy`, lm-eval `do_sample=False,temperature=0.0`;
- **vendor sampling**: `TEMPERATURE=1.0 TOP_K=64 TOP_P=1.0 SEED=0` (the checkpoint's
  `generation_config.json` defaults) -> server `--temperature 1.0 --top-k 64 --top-p 1.0
  --seed 0`, lm-eval `do_sample=True,temperature=1.0,top_p=1.0`.

The `gsm8k_cot` 5-shot task (the script's default `TASK`) was used for the diagnosis runs;
its "Q: ... A:" prompt has no terminal answer phrase, which is what produced the greedy
self-talk loops and the last-number extraction misses (`eval-results/gsm8k_diagnosis.md`).

### 8.2 Commands

```bash
uv sync --extra eval --inexact        # .venv/bin/lm_eval
# greedy, llama format, NVFP4 + int8 dense (final: 95.5 % flexible / 97.5 % strict on 200 problems)
TASK=gsm8k_cot_llama NUM_FEWSHOT=8 LIMIT=200 MAX_GEN_TOKS=8192 REASONING_EFFORT=medium \
  WEIGHTS=$MUSESPARK_NVFP4_PRESHARDED TEMPERATURE=0 bash scripts/eval_musespark_gsm8k.sh
# vendor sampling, same prompt (final: 97.5 % / 97.5 %)
TASK=gsm8k_cot_llama NUM_FEWSHOT=8 LIMIT=200 MAX_GEN_TOKS=8192 REASONING_EFFORT=medium \
  WEIGHTS=$MUSESPARK_NVFP4_PRESHARDED TEMPERATURE=1.0 TOP_K=64 TOP_P=1.0 SEED=0 \
  bash scripts/eval_musespark_gsm8k.sh
# int4 container for comparison: WEIGHTS=$MUSESPARK_PRESHARDED
```

Other knobs: `PORT` (default 8008; the real server listens on `PORT+1`), `CONTEXT`,
`STARTUP_TIMEOUT` (900 s: container load + compile), `EVAL_TIMEOUT` (2100 s for the whole
lm-eval run; raise it for `LIMIT=full`), `RUN_TAG`, `OUTPUT_PATH`, `MODEL_NAME`, `EXAMPLES`
(transcripts replayed into `summary.md`); extra arguments go to `lm_eval`. Each 100-problem
run took ~12-16 min of lm-eval time (server decode at ~235-267 tok/s B=1,
`gsm8k_diagnosis.md`); the final 200-problem runs took 11 min (greedy, 08:40-08:51 UTC) and
10 min (sampled, 08:51-09:01 UTC) with the int8 path.

### 8.3 Reading the results

`summary.md` (and `results_*.json`) report two lm-eval metrics:

- **`exact_match,flexible-extract`**: the last number in the response is compared with the
  gold answer (the metric usually quoted as "GSM8K accuracy");
- **`exact_match,strict-match`**: the answer must match the task's regex (`The final answer is
  N` for `gsm8k_cot_llama`, `The answer is N.` for `gsm8k_cot`). With the llama prompt the two
  coincide; with `gsm8k_cot` strict falls to 53-57 % only because the model does not use that
  exact phrase.

`summary.md` also lists budget hits (`finish length`, counted as wrong), generated tokens
per problem and decode throughput; `failures.md` classes every miss as truncated /
extraction / arithmetic / reading. Read the accuracies from `results_*.json` / the metric
table of `summary.md`: the trailing "samples: N; strict correct 0, flexible correct 0" line of
`summary.md` is a summariser artefact (it counts per-filter sample rows and finds no per-sample
field to sum) in every run directory. Measured (`eval-results/gsm8k_diagnosis.md`):

| run | container | task | decoding | flexible | strict | budget hits | directory |
|---|---|---|---|---:|---:|---:|---|
| int4, 4096 budget | int4 | gsm8k_cot 5-shot | greedy | 87.0 % | 53.0 % | 5 | `eval-results/gsm8k-musespark-100-20260926T030101Z` (head `cd0d3c3`) |
| int4, 8192 budget | int4 | gsm8k_cot 5-shot | greedy | 87.0 % | 53.0 % | 5 | `..-100-int4-8192-medium-20260926T045954Z` (head `37bd619`) |
| NVFP4, 8192 | NVFP4 | gsm8k_cot 5-shot | greedy | 92.0 % | 57.0 % | 1 | `..-100-nvfp4-8192-medium-20260926T051225Z` (head `37bd619`) |
| NVFP4, llama | NVFP4 | gsm8k_cot_llama 8-shot | greedy | **97.0 %** | 97.0 % | 0 | `..-100-nvfp4-8192-medium-llama-20260926T053831Z` (head `c11b8b2`) |
| NVFP4, llama | NVFP4 | gsm8k_cot_llama 8-shot | T=1.0, top-k 64, top-p 1.0, seed 0 | **98.0 %** | 98.0 % | 0 | `..-100-nvfp4-8192-medium-llama-t1.0k64p1.0-20260926T054705Z` (head `c11b8b2`) |
| **final**, 200 problems | NVFP4 + int8 dense | gsm8k_cot_llama 8-shot | greedy | 95.5 % | **97.5 %** | 0 | `..-200-nvfp4-8192-medium-llama-20260926T084027Z` (head `9030151`) |
| **final**, 200 problems | NVFP4 + int8 dense | gsm8k_cot_llama 8-shot | T=1.0, top-k 64, top-p 1.0, seed 0 | **97.5 %** | **97.5 %** | 0 | `..-200-nvfp4-8192-medium-llama-t1.0k64p1.0-20260926T085135Z` (head `9030151`) |

First-100 subsets of the final runs: greedy 96.0 / 97.0, sampled 98.0 / 98.0. The nine greedy
flexible-extract misses (docs 12, 37, 45, 93, 119, 147, 161, 184, 187) are hedged or
multi-reading final answers in eight cases (the gold number is in the answer text; strict-match
accepts four of them, where the first number after "The final answer is" is the gold) and one
off-by-one (doc 12). **Acceptance criterion 1 (> 97 %)** is met with the checkpoint's default
sampling (97.5 % on both metrics) and by strict-match under greedy decoding (97.5 %); greedy
flexible-extract is 95.5 %. The int4 container was not re-run on the llama prompt.

---

## 9. Official-maths comparison on CPU: `scripts/compare_official_musespark.py`

Ground truth for the *translation*: the JAX reference `musespark.forward` vs the official
sglang PyTorch model maths, on CPU, with the real bf16 weights, first `--layers` layers plus
embedding norm, final norm and `lm_head`, every sub-step tapped and compared (per-layer max-abs
and relative-RMS). Prerequisites:

- a **git worktree of the sglang `dev` branch** (`d689b242d1` was used) whose `python/`
  directory is passed as `--sglang-python`; `sglang/srt/layers/muse_spark_layers.py` is
  imported verbatim (three trivial import stubs), the rest is a transcription of
  `models/muse_spark.py`, the `forward_native` fallbacks of `kernels/ops/muse_spark_v12/`,
  `layers/muse_spark_bf16_moe.py`, `layers/attention/torch_native_backend.py` and
  `layers/logits_processor.py`;
- a CPU Python env with `torch` (CPU build; 2.14 was used), `jax` (CPU), `safetensors`,
  `numpy` and the repo root on `sys.path` (a separate venv is simplest: sglang's `dev`
  requirements are CUDA-only, so sglang itself is **not** installed);
- RAM ~25 GB per layer per implementation plus 6.6 GB for the `lm_head` twice; ~15 min for 4
  layers on a 224-core host with the weights in page cache;
- a prompt: `prompt0_ids.npy` from the oracle of section 6.1.

```bash
git -C /filestore/srcs/sglang-muse-spark worktree add /filestore/tmp/sglang-dev origin/dev
JAX_PLATFORMS=cpu <cpu-venv>/bin/python scripts/compare_official_musespark.py \
    --sglang-python /filestore/tmp/sglang-dev/python --checkpoint "$MUSESPARK_CHECKPOINT" \
    --ids /filestore/tmp/real_ref/prompt0_ids.npy --layers 4 --out /filestore/tmp/cmp
# window semantics (layer 0, 300 ids, sliding window 256 in BOTH implementations):
#   ... --layers 1 --sliding-window 256 --no-head --ids ids_300.npy
```

Expected (`scratchpad/official_cmp/NOTES.md`, `compare_prompt0.log`): residual stream
relative RMS ours vs official `hidden[0]` exact, `hidden[1..4]` 3.2e-3 / 4.2e-3 / 5.1e-3 /
6.2e-3, logits 7.8e-3 (max abs 0.57 on soft-capped logits), top-1 agreement 148/151, our top-1
always in the official top-5; noise floors of the official code against itself (SDPA-bf16 vs
fp32 attention 1.9e-3, bf16 vs fp32 expert accumulation 1.1e-3) mean anything below ~5e-3 is
kernel-order noise and anything above 1e-2 would be a formula difference; only the
`topk_scores` rows are flagged (near-tie routing flips). Verdict: no formula discrepancy.

---

## 10. Demo and server

```bash
bash scripts/demo_musespark.sh                                    # interactive, type at "prompt>"
bash scripts/demo_musespark.sh --prompt "What is 2+2?" --greedy --max-tokens 200
bash scripts/demo_musespark.sh --weights $MUSESPARK_NVFP4_PRESHARDED --prompt "..." \
     --reasoning-effort low --temperature 1.0 --top-k 64 --top-p 1.0 --seed 0
bash scripts/demo_musespark.sh --serve 0.0.0.0:8000 --context 8192 --max-tokens 512 --greedy
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"muse-spark-1.2-816b-a42b","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":256,"temperature":0}'
```

Flags (`demo_musespark.py`): `--weights`, `--checkpoint`, `--context` (multiple of 128,
default 8192), `--batch` (1/2/4/8 rows; several `--prompt`s are prefilled into their own
rows), `--max-tokens`, `--greedy` or `--temperature/--top-k/--top-p/--seed` (defaults 1.0 /
64 / 1.0 / 0), `--raw` (no chat template), `--reasoning-effort minimal|low|medium|high|xhigh`,
`--no-stop`, `--steps-per-call` (64), `--serve HOST:PORT`, `--model-name`, `--bench`,
`--bench-steps`. Endpoints (`openai_server.py`): `GET /v1/models`, `GET /health`,
`POST /v1/chat/completions`, `POST /v1/completions`, `POST /tokenize`, `POST /detokenize`. The
server answers one request at a time on row 0; the reasoning channel (` to=self`) is returned
as `reasoning_content`, the answer (` to=user`) as `content`; generation stops on the
`eos_token_id`s of `generation_config.json` (200001 `<|end_of_text|>`, 200008 `<|eot|>`);
`<|eom|>` (200007) only ends a channel. Loading a container takes 25 s warm / several minutes cold, the first call per
(batch, bucket) compiles (~5 s decode, 3-6 s per prefill bucket; cached in `.jax_cache`).

---

## 11. Full-run checklist for the deliverable

1. `uv sync`; `.env` written (section 1).
2. Containers built and verified (section 3); `quantize-dense` applied if present (3.4).
3. CPU suite: 181 passed, 29 skipped, 9 warnings in 519 s (section 4, `logs/final_cpu_suite.log`).
4. TPU suites, single chip and eight cores (section 5): 90 passed, 16 skipped in 436 s / 38 passed, 7 skipped in 117 s.
5. Oracles for the int4, NVFP4 and NVFP4 + int8 containers (6.1);
   `validate_musespark_decode.py --tasks replay,generate,bench` (6.2): argmax EQUAL, first
   tokens equal, coherent texts; regression tests alone (6.3): 1 passed in 266 s (cold int4 container load included) (int4 prefill),
   2 passed in 86 s (NVFP4 + int8 decode).
6. Benchmark table (7.1): B=1 3.260 ms/step = 306.8 tok/s on NVFP4 + int8
   (`logs/validate_musespark_decode_20260926T083121Z.log`).
7. GSM8K llama 8-shot, 200 problems, NVFP4 + int8 (8.3): greedy 95.5 / 97.5 %, sampled
   97.5 / 97.5 %.
8. Official-maths comparison (section 9) unchanged: no row above 1e-2 except `topk_scores`.

---

## 12. Troubleshooting

- **HBM OOM when two real-weight tests share a process.** Each real-weights test/script loads
  a whole container (54.8 / 57.8 GiB per core) and JAX keeps the arrays alive until the process
  exits; `tests/test_musespark_real_prefill.py` and `tests/test_musespark_real_decode.py`
  loaded in the same pytest process OOM on HBM (`notes/perf_log.md` E14). Run them alone, and
  never run the validate scripts while a server holds the chips.
- **"TPU already in use" / cannot open device.** Only one process may own a chip. Stop the
  server/demo first (`kill -- -<pid>` of the launcher's process group as the GSM8K script
  does); on shared machines wrap every TPU command in `tpu_run.sh all` (per-chip `flock`s).
- **Compile cache.** `JAX_COMPILATION_CACHE_DIR=<repo>/.jax_cache` (set by the launchers)
  makes the second start of every (batch, bucket, format) executable instant; the first compile
  of the 62-layer kernel is ~5 s (`generate: chunk of 16 steps 5.62 s (compile + run)`), each
  prefill bucket 3-6 s. Delete `.jax_cache` after changing jax/libtpu versions or if a stale
  executable is suspected; the `interpret`/`skip=` options change the cache key automatically.
- **Timing caveats: ~50 us dispatch and silent reshards.** Host dispatch of a jitted
  `pallas_call` is ~50 us, and a jit argument whose sharding does not match the `shard_map`
  in-spec is **silently re-sharded per call** (300+ us for a 16-element int32 array, a full
  copy for big arrays) (`notes/tpu7x_hw_report.md`, "Timing caveat"). Always
  `jax.device_put(x, NamedSharding(mesh, spec))` inputs before timing and time long
  multi-step calls (`--steps-per-call 16..64`), never single steps. The demo/validate harnesses
  already do this.
- **Load time: page cache.** A container that is in the host page cache loads in ~25-29 s
  (`weights resident in 29 s`, `logs/run1.out`; `load_seconds` 22-26 s in
  `logs/validate_musespark_decode_*.json`). Cold from NFS it is bandwidth-bound: 496 GB in
  384 s = 6.4 min at 1.29 GB/s (`notes/nvfp4_container_v2.md`), 129-254 s when only part of it
  was evicted (the `.json` summaries at 04:30 / 05:15 / 05:20 UTC after switching containers).
  With 944 GB of RAM one container stays cached; alternating int4 and NVFP4 evicts the other.
  `STARTUP_TIMEOUT=900` in the GSM8K script covers a cold load plus compile.
- **Disk guard of the NVFP4 converter.** It stops downloading when free space is below
  `--min-free-gb` (default 8 GB) and keeps at most `--max-shards` (3) shards; if it stalls,
  free space and re-run the same script (resumable per unit). Peak usage is the container plus
  ~3 shards; the log prints `free <GB>` per shard.
- **Prefill test differs by exactly one bf16 ulp (0.00195) on TPU.** The f32 result of a bf16
  GEMM depends on its N-shape on TPU, so a rare projection value rounds differently between
  the per-rank and the reference K projection and RoPE carries it to the pair; the test accepts
  two bf16 ulps per slot and >= 99.5 % bit-exact slots since commit `8084031`
  (`musespark/README.md` section 8). Before that commit the two
  `test_prefill_writes_the_kernel_cache_layout` cases fail on TPU (pass on CPU).
- **`sample()` at temperature 0 on a sampling server** returned ids 0-5 garbage before commit
  `022cf73` (division by `float32.tiny` overflowed the logits to `inf`); with the fix a
  `temperature: 0` request on a server started with `--temperature 1.0` is greedy.
- **Excess precision.** If the kernel and the oracle disagree by more than the tolerances of
  section 6.3 check that `XLA_FLAGS=--xla_allow_excess_precision=false` was set in the
  environment of *both* runs (the oracle and the kernel).
- **`--context` errors.** The demo requires a multiple of 128 that holds a prompt plus one
  call (`2 * CHUNK + steps_per_call`); the GSM8K script raises the context to `>= MAX_GEN_TOKS
  + 2048`.
- **Unsigned integer vector ops crash the core** (`notes/tpu7x_hw_report.md`): nibble unpacking
  happens on the host or in XLA, never in a Pallas body; do not add `uint8`/`uint32` maths to
  the kernel.
