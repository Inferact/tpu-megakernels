#!/usr/bin/env bash
# Muse Spark 1.2 decode throughput on the TP8 megakernel: ms/step and aggregate tok/s for
# B in {1, 2, 4, 8} plus prefill timings (the demo's --bench mode), logged under
# benchmark-results/.
#
# Usage:
#   bash scripts/benchmark_musespark.sh
#   TPU_RUN="<scratchpad>/tpu_run.sh all" bash scripts/benchmark_musespark.sh   # shared chips
#
# CONTEXT (default 4096), STEPS (timed decode steps per batch size, default 256),
# STEPS_PER_CALL (default 64), WEIGHTS (int4 or NVFP4 container, default the int4 one),
# CHECKPOINT, OUTPUT_PATH may be overridden through the environment; extra arguments are
# passed to the demo (e.g. --greedy). The official numbers of README.md come from
# scripts/validate_musespark_decode.py --tasks bench (16 steps per call).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTEXT="${CONTEXT:-4096}"
STEPS="${STEPS:-256}"
STEPS_PER_CALL="${STEPS_PER_CALL:-64}"
WEIGHTS="${WEIGHTS:-/filestore/weights/muse-spark-tp8-int4}"
CHECKPOINT="${CHECKPOINT:-/filestore/weights/Muse-Spark-1.2-816B-A42B-open}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/benchmark-results/musespark-${RUN_ID}}"
mkdir -p "${OUTPUT_PATH}"

echo "Muse Spark decode benchmark: context ${CONTEXT}, ${STEPS} steps per batch size (log: ${OUTPUT_PATH}/bench.log)"
MUSESPARK_MEGAKERNEL_SRC="${REPO_ROOT}" bash "${REPO_ROOT}/scripts/demo_musespark.sh" \
  --bench --bench-steps "${STEPS}" --steps-per-call "${STEPS_PER_CALL}" \
  --context "${CONTEXT}" --weights "${WEIGHTS}" --checkpoint "${CHECKPOINT}" "$@" \
  2>&1 | tee "${OUTPUT_PATH}/bench.log"
echo "results in ${OUTPUT_PATH}/bench.log"
