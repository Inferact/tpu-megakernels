#!/usr/bin/env bash
# Launch the Kimi K3 OpenAI-compatible server and evaluate GPQA Diamond with
# lm-evaluation-harness using an explicit reasoning and final-answer format.
#
# Usage:
#   JOBID=<four-node-slurm-job> bash scripts/eval_kimi_gpqa_diamond.sh
#
# LIMIT (unset means the full split), PORT, CONTEXT, MAX_GEN_TOKS, MODEL_NAME,
# CHAT_MODE, TASK, TEMPERATURE, TOP_P, SYSTEM_INSTRUCTION, STARTUP_TIMEOUT, and OUTPUT_PATH may be
# overridden through the environment. Additional arguments are passed directly
# to lm_eval.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi
JOBID="${JOBID:-${SLURM_JOB_ID:-}}"
: "${JOBID:?Set JOBID to a running four-node Slurm allocation}"

LIMIT="${LIMIT:-}"
PORT="${PORT:-8000}"
# K3's max-effort reasoning commonly exceeds 2K tokens on GPQA.
CONTEXT="${CONTEXT:-65536}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-64000}"
MODEL_NAME="${MODEL_NAME:-kimi-k3}"
CHAT_MODE="${CHAT_MODE:-think}"
TASK="${TASK:-gpqa_diamond_reasoning}"
TEMPERATURE="${TEMPERATURE:-}"
TOP_P="${TOP_P:-}"
SYSTEM_INSTRUCTION="${SYSTEM_INSTRUCTION:-}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
SAMPLE_LABEL="${LIMIT:-full}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/eval-results/gpqa-diamond-${SAMPLE_LABEL}-${RUN_ID}}"
SERVER_LOG="${OUTPUT_PATH}/server.log"

for command in curl scontrol squeue uv; do
  command -v "${command}" >/dev/null || {
    echo "required command not found: ${command}" >&2
    exit 1
  }
done

case "${CHAT_MODE}" in
  response|think) ;;
  *)
    echo "CHAT_MODE must be 'response' or 'think'" >&2
    exit 1
    ;;
esac

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

echo "Starting Kimi K3 server on ${BASE_URL} (log: ${SERVER_LOG})"
K3_MEGAKERNEL_SRC="${REPO_ROOT}" JOBID="${JOBID}" \
  bash "${REPO_ROOT}/scripts/demo_kimi_dspark.sh" \
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

if [[ -n "${LIMIT}" ]]; then
  echo "Server ready; evaluating ${LIMIT} GPQA Diamond prompts with ${TASK}"
else
  echo "Server ready; evaluating the full GPQA Diamond split with ${TASK}"
fi

GEN_KWARGS="max_gen_toks=${MAX_GEN_TOKS}"
if [[ -n "${TEMPERATURE}" ]]; then GEN_KWARGS+=",temperature=${TEMPERATURE}"; fi
if [[ -n "${TOP_P}" ]]; then GEN_KWARGS+=",top_p=${TOP_P}"; fi

EVAL_COMMAND=(
  uv run --project "${REPO_ROOT}" --extra eval lm_eval run
  --model local-chat-completions
  --tasks "${TASK}"
  --include_path "${REPO_ROOT}/scripts"
  --apply_chat_template
  --model_args "model=${MODEL_NAME},base_url=${BASE_URL}/v1/chat/completions,num_concurrent=1,max_retries=3,tokenized_requests=False"
  --gen_kwargs "${GEN_KWARGS}"
  --batch_size 1
  --log_samples
  --output_path "${OUTPUT_PATH}"
)
if [[ -n "${SYSTEM_INSTRUCTION}" ]]; then
  EVAL_COMMAND+=(--system_instruction "${SYSTEM_INSTRUCTION}")
fi
if [[ -n "${LIMIT}" ]]; then
  EVAL_COMMAND+=(--limit "${LIMIT}")
fi

OPENAI_API_KEY="${OPENAI_API_KEY:-local-server}" JAX_PLATFORMS=cpu \
  "${EVAL_COMMAND[@]}" "$@"

echo "GPQA Diamond evaluation complete: ${OUTPUT_PATH}"
