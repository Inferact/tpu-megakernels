#!/usr/bin/env bash
# Launch the Qwen3.8 demo's OpenAI-compatible server and evaluate GSM8K with
# lm-evaluation-harness (raw-text completions, five in-context examples).
#
# Usage:
#   JOBID=<slurm-job> bash scripts/eval_qwen_gsm8k.sh
#
# LIMIT, PORT, CONTEXT, MAX_GEN_TOKS, MODEL_NAME, STARTUP_TIMEOUT, and
# OUTPUT_PATH may be overridden through the environment. Additional arguments
# are passed directly to lm_eval.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi
JOBID="${JOBID:-${SLURM_JOB_ID:-}}"
: "${JOBID:?Set JOBID to a running Slurm allocation}"

LIMIT="${LIMIT:-100}"
PORT="${PORT:-8000}"
CONTEXT="${CONTEXT:-2048}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-1024}"
MODEL_NAME="${MODEL_NAME:-qwen3.8-27b-dflash2}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/eval-results/gsm8k-qwen-${LIMIT}-${RUN_ID}}"
SERVER_LOG="${OUTPUT_PATH}/server.log"

for command in curl scontrol squeue uv; do
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

echo "Starting Qwen3.8 server on ${BASE_URL} (log: ${SERVER_LOG})"
QWEN_MEGAKERNEL_SRC="${REPO_ROOT}" JOBID="${JOBID}" HOST="${SERVER_HOST}" \
  bash "${REPO_ROOT}/scripts/demo_qwen_dflash.sh" \
    --serve "0.0.0.0:${PORT}" \
    --model-name "${MODEL_NAME}" \
    --context "${CONTEXT}" \
    --max-tokens "${MAX_GEN_TOKS}" \
    --chat response \
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

echo "Server ready; evaluating ${LIMIT} GSM8K prompts"
API_TYPE="${API_TYPE:-chat}"
if [[ "${API_TYPE}" == "chat" ]]; then
  # the chat endpoint renders the model's think-off assistant scaffold, which
  # the raw completions endpoint lacks (thinking is on by default)
  LM_ARGS=(--model local-chat-completions
    --apply_chat_template
    --model_args "model=${MODEL_NAME},base_url=${BASE_URL}/v1/chat/completions,tokenized_requests=False,num_concurrent=1,max_retries=3,timeout=3600")
else
  LM_ARGS=(--model local-completions
    --model_args "model=${MODEL_NAME},base_url=${BASE_URL}/v1/completions,tokenized_requests=False,tokenizer_backend=None,num_concurrent=1,max_retries=3,timeout=3600")
fi

OPENAI_API_KEY="${OPENAI_API_KEY:-local-server}" JAX_PLATFORMS=cpu \
  uv run --project "${REPO_ROOT}" --extra eval lm_eval run \
    --tasks gsm8k \
    --num_fewshot 5 \
    --gen_kwargs "do_sample=False,temperature=0.0,top_p=1.0,max_gen_toks=${MAX_GEN_TOKS}" \
    --seed 0 \
    --batch_size 1 \
    --limit "${LIMIT}" \
    --log_samples \
    --output_path "${OUTPUT_PATH}" \
    ${LM_ARGS[@]+"${LM_ARGS[@]}"} \
    "$@"

echo "GSM8K evaluation complete: ${OUTPUT_PATH}"
