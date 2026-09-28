#!/usr/bin/env bash
# Launch the DeepSeek-V4.1-Flash OpenAI-compatible server and evaluate GPQA Diamond with
# lm-evaluation-harness.
#
# Usage:
#   JOBID=<slurm-job> bash scripts/eval_deepseek_v41_gpqa_diamond.sh
#   # Or on GKE / single-host TPU v7x:
#   bash scripts/eval_deepseek_v41_gpqa_diamond.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi

LIMIT="${LIMIT:-}"
PORT="${PORT:-8000}"
# DeepSeek-V4.1-Flash reasoning on GPQA Diamond has median length ~3,000 tokens (up to ~15,800).
CONTEXT="${CONTEXT:-16384}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-15800}"
MODEL_NAME="${MODEL_NAME:-deepseek-v4.1-flash}"
CHAT_MODE="${CHAT_MODE:-think}"
TASK="${TASK:-gpqa_diamond_reasoning}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
SAMPLE_LABEL="${LIMIT:-full}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/eval-results/deepseek-v41-gpqa-diamond-${SAMPLE_LABEL}-${RUN_ID}}"
SERVER_LOG="${OUTPUT_PATH}/server.log"

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
if [[ -n "${JOBID:-${SLURM_JOB_ID:-}}" ]] && command -v squeue >/dev/null 2>&1; then
  JOBID="${JOBID:-${SLURM_JOB_ID}}"
  NODELIST="$(squeue -j "${JOBID}" -h -o %N)"
  mapfile -t HOSTS < <(scontrol show hostnames "${NODELIST}")
  SERVER_HOST="${HOSTS[0]}"
fi
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

echo "Starting DeepSeek-V4.1-Flash server on ${BASE_URL} (log: ${SERVER_LOG})"
DSV41_MEGAKERNEL_SRC="${REPO_ROOT}" \
  bash "${REPO_ROOT}/scripts/demo_deepseek_v41_dspark.sh" \
    --serve "0.0.0.0:${PORT}" \
    --model-name "${MODEL_NAME}" \
    --context "${CONTEXT}" \
    --max-tokens "${MAX_GEN_TOKS}" \
    --chat "${CHAT_MODE}" \
    --progress log \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl --fail --silent --show-error "${BASE_URL}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "DeepSeek-V4.1-Flash server exited during startup:" >&2
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

EVAL_COMMAND=(
  uv run --project "${REPO_ROOT}" --extra eval lm_eval run
  --model local-chat-completions
  --tasks "${TASK}"
  --include_path "${REPO_ROOT}/scripts"
  --apply_chat_template
  --model_args "model=${MODEL_NAME},base_url=${BASE_URL}/v1/chat/completions,num_concurrent=1,max_retries=3,tokenized_requests=False"
  --gen_kwargs "max_gen_toks=${MAX_GEN_TOKS}"
  --batch_size 1
  --log_samples
  --output_path "${OUTPUT_PATH}"
)
if [[ -n "${LIMIT}" ]]; then
  EVAL_COMMAND+=(--limit "${LIMIT}")
fi

OPENAI_API_KEY="${OPENAI_API_KEY:-local-server}" JAX_PLATFORMS=cpu \
  "${EVAL_COMMAND[@]}" "$@"

echo "GPQA Diamond evaluation complete: ${OUTPUT_PATH}"
