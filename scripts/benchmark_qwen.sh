#!/usr/bin/env bash
# Reproduce the Qwen3.8-27B plain TP8 decode throughput number: start the
# demo's OpenAI-compatible server in single-token mode, wait for it to become
# healthy, then measure the client-visible TTFT / TPOT / end-to-end tok/s on
# ISL=1 / OSL=1000 streaming requests (2 warmup + 12 timed).
#
# Usage:
#   JOBID=<slurm-job> bash scripts/benchmark_qwen.sh
#
# PORT, CONTEXT, MAX_TOKENS, STEPS_PER_CALL (steps per device call, default 256),
# OUTPUT_PATH may be overridden through the environment.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi
JOBID="${JOBID:-${SLURM_JOB_ID:-}}"
: "${JOBID:?Set JOBID to a running Slurm allocation}"

PORT="${PORT:-8000}"
CONTEXT="${CONTEXT:-2048}"
MAX_TOKENS="${MAX_TOKENS:-1000}"
STEPS_PER_CALL="${STEPS_PER_CALL:-256}"
ISL="${ISL:-1}"
OSL="${OSL:-1000}"
N_WARMUP="${N_WARMUP:-2}"
N_BENCH="${N_BENCH:-12}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/benchmark-results/qwen-plain-${RUN_ID}}"
SERVER_LOG="${OUTPUT_PATH}/server.log"

for command in curl scontrol squeue; do
  command -v "${command}" >/dev/null || {
    echo "required command not found: ${command}" >&2
    exit 1
  }
done

NODELIST="$(squeue -j "${JOBID}" -h -o %N)"
: "${NODELIST:?Slurm job ${JOBID} is not running}"
mapfile -t HOSTS < <(scontrol show hostnames "${NODELIST}")
SERVER_HOST="${SERVER_HOST:-${HOSTS[0]}}"
BASE_URL="http://${SERVER_HOST}:${PORT}"

mkdir -p "${OUTPUT_PATH}"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting Qwen3.8 plain server on ${BASE_URL} (log: ${SERVER_LOG})"
QWEN_MEGAKERNEL_SRC="${REPO_ROOT}" JOBID="${JOBID}" HOST="${SERVER_HOST}" \
  bash "${REPO_ROOT}/scripts/demo_qwen_dflash.sh" \
    --serve "0.0.0.0:${PORT}" \
    --context "${CONTEXT}" \
    --max-tokens "${MAX_TOKENS}" \
    --no-spec \
    --steps-per-call "${STEPS_PER_CALL}" \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl --fail --silent --show-error "${BASE_URL}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Qwen3.8 server exited during startup:" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "server did not become ready within ${STARTUP_TIMEOUT} seconds" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 5
done

echo "Server ready; benching ${N_BENCH} timed requests (ISL=${ISL} OSL=${OSL}, plain decode)"
ISL="${ISL}" OSL="${OSL}" N_WARMUP="${N_WARMUP}" N_BENCH="${N_BENCH}" \
  "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/bench_openai_stream.py" \
  --base-url "${BASE_URL}" | tee "${OUTPUT_PATH}/bench.log"
