#!/usr/bin/env bash
# Benchmark the true-batched Kimi K3 megakernel without DSpark. Requests contain
# exactly one random input token and generate exactly 1,024 tokens each.
#
# Usage:
#   JOBID=<four-node-job> bash scripts/benchmark_kimi.sh
#
# BATCH_SIZE, NUM_PROMPTS, STEPS_PER_CALL, PORT, CONTEXT, MODEL_NAME,
# TOKENIZER, VLLM_BENCH, STARTUP_TIMEOUT, and OUTPUT_PATH may be overridden
# through the environment. Additional arguments are passed to vllm-bench.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi
JOBID="${JOBID:-${SLURM_JOB_ID:-}}"
: "${JOBID:?Set JOBID to a running four-node Slurm allocation}"

PORT="${PORT:-8000}"
CONTEXT="${CONTEXT:-1152}"
MODEL_NAME="${MODEL_NAME:-kimi-k3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
if [[ ! "${BATCH_SIZE}" =~ ^[1-8]$ ]]; then
  echo "BATCH_SIZE must be an integer from 1 through 8" >&2
  exit 2
fi
NUM_PROMPTS="${NUM_PROMPTS:-$((4 * BATCH_SIZE))}"
NUM_WARMUPS=0
if [[ ! "${NUM_PROMPTS}" =~ ^[1-9][0-9]*$ ]] || (( NUM_PROMPTS % BATCH_SIZE != 0 )); then
  echo "NUM_PROMPTS must be a positive multiple of BATCH_SIZE" >&2
  exit 2
fi
STEPS_PER_CALL="${STEPS_PER_CALL:-64}"
if [[ ! "${STEPS_PER_CALL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "STEPS_PER_CALL must be a positive integer" >&2
  exit 2
fi
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
VLLM_BENCH="${VLLM_BENCH:-vllm-bench}"
TOKENIZER="${TOKENIZER:-moonshotai/Kimi-K3}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/benchmark-results/kimi-target-${RUN_ID}}"
SERVER_LOG="${OUTPUT_PATH}/server.log"
RESULT_FILE="benchmark.json"
CLIENT_LOG="${OUTPUT_PATH}/vllm-bench.log"
CLIENT_RESULT_FILE="vllm-bench.json"

for command in curl scontrol squeue; do
  command -v "${command}" >/dev/null || {
    echo "required command not found: ${command}" >&2
    exit 1
  }
done
if [[ "${VLLM_BENCH}" == */* ]]; then
  [[ -x "${VLLM_BENCH}" ]] || {
    echo "vllm-bench is not executable: ${VLLM_BENCH}" >&2
    exit 1
  }
else
  command -v "${VLLM_BENCH}" >/dev/null || {
    echo "vllm-bench not found; set VLLM_BENCH to its executable path" >&2
    exit 1
  }
fi
NODELIST="$(squeue -j "${JOBID}" -h -o %N)"
: "${NODELIST:?Slurm job ${JOBID} is not running}"
mapfile -t HOSTS < <(scontrol show hostnames "${NODELIST}")
[[ ${#HOSTS[@]} -eq 4 ]] || {
  echo "job ${JOBID} holds ${#HOSTS[@]} nodes; the demo requires 4" >&2
  exit 1
}
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

echo "Starting target-only Kimi K3 server on ${BASE_URL} with batch ${BATCH_SIZE} and ${STEPS_PER_CALL} steps/call"
echo "Server log: ${SERVER_LOG}"
K3_MEGAKERNEL_SRC="${REPO_ROOT}" JOBID="${JOBID}" \
  bash "${REPO_ROOT}/scripts/demo_kimi_dspark.sh" \
    --serve "0.0.0.0:${PORT}" \
    --model-name "${MODEL_NAME}" \
    --context "${CONTEXT}" \
    --max-tokens 1024 \
    --chat none \
    --target-only \
    --target-batch-size "${BATCH_SIZE}" \
    --steps-per-call "${STEPS_PER_CALL}" \
    --progress log \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl --fail --silent --show-error "${BASE_URL}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Kimi K3 server exited during startup:" >&2
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

echo "Server ready; benchmarking ${NUM_PROMPTS} requests (1 input token, 1024 output tokens, batch ${BATCH_SIZE})"
if ! HF_HUB_OFFLINE=1 "${VLLM_BENCH}" \
  --backend openai \
  --base-url "${BASE_URL}" \
  --model "${MODEL_NAME}" \
  --tokenizer "${TOKENIZER}" \
  --trust-remote-code \
  --dataset-name random \
  --random-input-len 1 \
  --random-output-len 1024 \
  --random-range-ratio 1.0 \
  --prompt-token-ids \
  --num-prompts "${NUM_PROMPTS}" \
  --num-warmups "${NUM_WARMUPS}" \
  --max-concurrency "${BATCH_SIZE}" \
  --ignore-eos \
  --no-steady-state \
  --save-result \
  --save-detailed \
  --result-dir "${OUTPUT_PATH}" \
  --result-filename "${CLIENT_RESULT_FILE}" \
  --label "kimi-target-b${BATCH_SIZE}-steps-${STEPS_PER_CALL}" \
  "$@" >"${CLIENT_LOG}" 2>&1; then
  echo "vllm-bench failed:" >&2
  tail -n 80 "${CLIENT_LOG}" >&2 || true
  exit 1
fi

# The first generated token in each request is produced by prefill. The server
# logs one shared device duration per true batch; summing per-request durations
# would overcount concurrent device work.
EXPECTED_BATCHES=$((NUM_PROMPTS / BATCH_SIZE))
deadline=$((SECONDS + 10))
while (( $(grep -c '^target_batch:' "${SERVER_LOG}" || true) < EXPECTED_BATCHES )); do
  if (( SECONDS >= deadline )); then
    echo "server log did not contain all batch timings" >&2
    exit 1
  fi
  sleep 0.1
done

BATCH_ROWS="$(
  sed -n -E 's/^target_batch: ([0-9]+) requests, ([0-9]+) generated in ([0-9.]+) s \(([0-9]+) steps\)$/\1 \2 \3 \4/p' "${SERVER_LOG}" \
    | tail -n "${EXPECTED_BATCHES}"
)"
ROW_COUNT="$(printf '%s\n' "${BATCH_ROWS}" | awk 'NF == 4 { count += 1 } END { print count + 0 }')"
if (( ROW_COUNT != EXPECTED_BATCHES )); then
  echo "could not parse ${EXPECTED_BATCHES} measured batch timings from ${SERVER_LOG}" >&2
  exit 1
fi

read -r DECODE_TOKENS DECODE_SECONDS DECODE_TOKENS_PER_SECOND <<<"$(
  printf '%s\n' "${BATCH_ROWS}" \
    | awk '{ tokens += $2 - $1; seconds += $3 } END { printf "%d %.6f %.6f", tokens, seconds, tokens / seconds }'
)"

printf '{\n' >"${OUTPUT_PATH}/${RESULT_FILE}"
printf '  "mode": "target_only",\n' >>"${OUTPUT_PATH}/${RESULT_FILE}"
printf '  "batch_size": %d,\n' "${BATCH_SIZE}" >>"${OUTPUT_PATH}/${RESULT_FILE}"
printf '  "steps_per_call": %d,\n' "${STEPS_PER_CALL}" >>"${OUTPUT_PATH}/${RESULT_FILE}"
printf '  "requests": %d,\n' "${NUM_PROMPTS}" >>"${OUTPUT_PATH}/${RESULT_FILE}"
printf '  "decode_tokens": %d,\n' "${DECODE_TOKENS}" >>"${OUTPUT_PATH}/${RESULT_FILE}"
printf '  "decode_seconds": %.6f,\n' "${DECODE_SECONDS}" >>"${OUTPUT_PATH}/${RESULT_FILE}"
printf '  "decode_tokens_per_second": %.6f\n' "${DECODE_TOKENS_PER_SECOND}" >>"${OUTPUT_PATH}/${RESULT_FILE}"
printf '}\n' >>"${OUTPUT_PATH}/${RESULT_FILE}"

printf 'Decode-only aggregate throughput (B%d): %.2f tokens/s\n' "${BATCH_SIZE}" "${DECODE_TOKENS_PER_SECOND}"
