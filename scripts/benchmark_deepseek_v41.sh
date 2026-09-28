#!/usr/bin/env bash
# Benchmark the DeepSeek-V4.1-Flash single-pallas_call decode megakernel across batch sizes B = 1, 2, 4, 8
# against the same-run XLA control.
#
# Usage:
#   JOBID=<slurm-job> bash scripts/benchmark_deepseek_v41.sh
#   # Or directly on GKE / TPU v7x host:
#   bash scripts/benchmark_deepseek_v41.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi

PY="${DSV41_PY:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python3 || command -v python)"
fi
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/benchmark-results/deepseek-v41-target-${RUN_ID}}"
STAGES="${STAGES:-perf_b1,perf_b2,perf_b4,perf_b8}"
mkdir -p "${OUTPUT_PATH}"

ARGS=(
  -m deepseek_v41.runner
  --stage "${STAGES}"
  --out "${OUTPUT_PATH}"
)
if [[ -n "${DSV41_PRESHARDED_WEIGHTS:-}" ]]; then
  ARGS+=(--presharded "${DSV41_PRESHARDED_WEIGHTS}")
fi
if [[ -n "${DSV41_CHECKPOINT:-}" ]]; then
  ARGS+=(--checkpoint "${DSV41_CHECKPOINT}")
fi

PYTHONPATH="${REPO_ROOT}" "${PY}" "${ARGS[@]}" "$@"
