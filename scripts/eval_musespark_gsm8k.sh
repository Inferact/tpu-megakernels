#!/usr/bin/env bash
# Launch the Muse Spark 1.2 demo's OpenAI-compatible server (TP8 decode megakernel, single
# host; int4 or NVFP4 container) and evaluate GSM8K chain-of-thought with lm-evaluation-harness, then summarise the run
# (accuracy, tokens per problem, truncations, server throughput, example transcripts) and
# classify the failures (scripts/analyze_gsm8k_failures.py).
#
# Usage:
#   [TPU_RUN="<scratchpad>/tpu_run.sh all"] bash scripts/eval_musespark_gsm8k.sh
#
# LIMIT (default 100), PORT, WEIGHTS (container directory; default the demo's int4 container),
# TASK (lm-eval task: gsm8k_cot (default, "The answer is N."), gsm8k_cot_llama (8-shot, "The
# final answer is N") or gsm8k_cot_zeroshot), CONTEXT (default 8192, or 16384 when MAX_GEN_TOKS
# > 4096), MAX_GEN_TOKS, NUM_FEWSHOT (default 8 for gsm8k_cot_llama, else 5), REASONING_EFFORT
# (minimal/low/medium/high/xhigh: the chat template's "Reasoning strength"), TEMPERATURE
# (default 0 = greedy; > 0 starts the server in sampling mode with TOP_K (64) / TOP_P (1.0) /
# SEED (0), the checkpoint's generation_config defaults, and asks lm-eval for do_sample), MODEL_NAME,
# STARTUP_TIMEOUT, EVAL_TIMEOUT (whole lm_eval run, seconds), RUN_TAG (suffix of the results
# directory) and OUTPUT_PATH may be overridden through the environment. Additional arguments
# are passed directly to lm_eval.
#
# The harness talks to a capture proxy (scripts/chat_capture_proxy.py) in front of the server
# so every chat completion (reasoning_content, finish_reason, usage) lands in
# <OUTPUT_PATH>/captures.jsonl; lm-eval itself only keeps message.content.
#
# The server holds the TPU lock (TPU_RUN) for the whole session, so keep LIMIT / EVAL_TIMEOUT
# bounded when the chips are shared. The server answers one request at a time (row 0), so the
# harness runs with num_concurrent=1.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi

LIMIT="${LIMIT:-100}"
PORT="${PORT:-8008}"
HOST="${HOST:-127.0.0.1}"
WEIGHTS="${WEIGHTS:-/filestore/weights/muse-spark-tp8-int4}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-4096}"
if [[ -z "${CONTEXT:-}" ]]; then
  # prompt (~700 tokens with 5 shots) + generation budget + prefill/decode slack must fit
  CONTEXT=8192
  while (( CONTEXT < MAX_GEN_TOKS + 2048 )); do CONTEXT=$((CONTEXT * 2)); done
fi
TASK="${TASK:-gsm8k_cot}"
if [[ -z "${NUM_FEWSHOT:-}" ]]; then
  case "${TASK}" in
    gsm8k_cot_llama) NUM_FEWSHOT=8 ;;
    gsm8k_cot_zeroshot) NUM_FEWSHOT=0 ;;
    *) NUM_FEWSHOT=5 ;;
  esac
fi
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_K="${TOP_K:-64}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-0}"
if [[ "${TEMPERATURE}" == "0" || "${TEMPERATURE}" == "0.0" ]]; then
  SERVER_SAMPLING=(--greedy)
  GEN_SAMPLING="do_sample=False,temperature=0.0"
  SAMPLING_TAG="greedy"
else
  # lm-eval forwards temperature / top_p / seed per request; top_k is the server's --top-k
  SERVER_SAMPLING=(--temperature "${TEMPERATURE}" --top-k "${TOP_K}" --top-p "${TOP_P}" --seed "${SEED}")
  GEN_SAMPLING="do_sample=True,temperature=${TEMPERATURE},top_p=${TOP_P}"
  SAMPLING_TAG="t${TEMPERATURE}k${TOP_K}p${TOP_P}"
fi
MODEL_NAME="${MODEL_NAME:-muse-spark-1.2-816b-a42b}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-2100}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
TASK_TAG="${TASK#gsm8k_cot}"; TASK_TAG="${TASK_TAG#_}"
RUN_TAG="${RUN_TAG:-$(basename "${WEIGHTS}" | sed 's/^muse-spark-tp8-//')-${MAX_GEN_TOKS}-${REASONING_EFFORT}${TASK_TAG:+-${TASK_TAG}}$([[ "${SAMPLING_TAG}" == greedy ]] || echo "-${SAMPLING_TAG}")}"
OUTPUT_PATH="${OUTPUT_PATH:-${REPO_ROOT}/eval-results/gsm8k-musespark-${LIMIT}-${RUN_TAG}-${RUN_ID}}"
SERVER_LOG="${OUTPUT_PATH}/server.log"
SERVER_PORT=$((PORT + 1))            # the real server; lm-eval talks to the capture proxy on PORT
SERVER_URL="http://${HOST}:${SERVER_PORT}"
BASE_URL="http://${HOST}:${PORT}"
CAPTURES="${OUTPUT_PATH}/captures.jsonl"
PYTHON="${REPO_ROOT}/.venv/bin/python"

for command in curl; do
  command -v "${command}" >/dev/null || {
    echo "required command not found: ${command}" >&2
    exit 1
  }
done
LM_EVAL="${REPO_ROOT}/.venv/bin/lm_eval"
if [[ ! -x "${LM_EVAL}" ]]; then
  echo "lm_eval not found at ${LM_EVAL}; run 'uv sync --extra eval --inexact' in ${REPO_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_PATH}"
SERVER_PID=""
PROXY_PID=""

cleanup() {
  if [[ -n "${PROXY_PID}" ]] && kill -0 "${PROXY_PID}" 2>/dev/null; then
    kill "${PROXY_PID}" 2>/dev/null || true
    wait "${PROXY_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    # the launcher exec's into `flock ... python`; end the whole process group
    kill -- -"${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# provenance: the kernel under test is the working tree, not just HEAD
{
  echo "head: $(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "weights: ${WEIGHTS}"
  echo "context: ${CONTEXT} max_gen_toks: ${MAX_GEN_TOKS} reasoning_effort: ${REASONING_EFFORT} num_fewshot: ${NUM_FEWSHOT} limit: ${LIMIT}"
  echo "task: ${TASK} sampling: ${SAMPLING_TAG} temperature: ${TEMPERATURE} top_k: ${TOP_K} top_p: ${TOP_P} seed: ${SEED}"
} >"${OUTPUT_PATH}/run-config.txt"
git -C "${REPO_ROOT}" diff -- musespark demo_musespark.py openai_server.py >"${OUTPUT_PATH}/worktree-uncommitted.diff" 2>/dev/null || true

echo "Starting Muse Spark server on ${SERVER_URL} (weights ${WEIGHTS}, context ${CONTEXT}, log: ${SERVER_LOG})"
MUSESPARK_MEGAKERNEL_SRC="${REPO_ROOT}" \
  setsid bash "${REPO_ROOT}/scripts/demo_musespark.sh" \
    --serve "${HOST}:${SERVER_PORT}" \
    --weights "${WEIGHTS}" \
    --model-name "${MODEL_NAME}" \
    --context "${CONTEXT}" \
    --max-tokens "${MAX_GEN_TOKS}" \
    --reasoning-effort "${REASONING_EFFORT}" \
    "${SERVER_SAMPLING[@]}" \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

"${PYTHON}" "${REPO_ROOT}/scripts/chat_capture_proxy.py" \
  --listen "${HOST}:${PORT}" --upstream "${SERVER_URL}" --out "${CAPTURES}" \
  >"${OUTPUT_PATH}/proxy.log" 2>&1 &
PROXY_PID=$!

deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl --fail --silent --show-error "${SERVER_URL}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Muse Spark server exited during startup:" >&2
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

# one manual chat completion first (through the proxy): confirms the server survives a real
# request and that the channel splitter puts the reasoning in reasoning_content and the
# answer in content
echo "Server ready after ${SECONDS} s; smoke-testing one chat completion"
SMOKE="${OUTPUT_PATH}/smoke.json"
curl --fail --silent --show-error --max-time 600 "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"'"${MODEL_NAME}"'","messages":[{"role":"user","content":"Q: A baker makes 12 loaves an hour for 3 hours and sells 20. How many loaves are left?\nA:"}],"max_tokens":1024,"temperature":0.0}' \
  >"${SMOKE}" || { echo "smoke chat completion failed:" >&2; tail -n 40 "${SERVER_LOG}" >&2; exit 1; }
"${PYTHON}" - "${SMOKE}" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
m = r["choices"][0]["message"]
print(f"smoke: finish={r['choices'][0]['finish_reason']} usage={r.get('usage')}")
print(f"smoke reasoning_content ({len(m.get('reasoning_content') or '')} chars): {(m.get('reasoning_content') or '')[:300]!r}")
print(f"smoke content ({len(m.get('content') or '')} chars): {(m.get('content') or '')[:300]!r}")
if not (m.get("content") or "").strip():
    print("smoke: empty content channel (reasoning only?)", file=sys.stderr)
PY
# the smoke request is not part of the evaluation
: >"${CAPTURES}"

echo "Evaluating ${LIMIT} GSM8K problems (${TASK}, ${NUM_FEWSHOT}-shot, chat template, ${SAMPLING_TAG}, max ${MAX_GEN_TOKS} tokens, reasoning ${REASONING_EFFORT})"
EVAL_COMMAND=(
  "${LM_EVAL}" run
    --model local-chat-completions
    --tasks "${TASK}"
    --num_fewshot "${NUM_FEWSHOT}"
    --apply_chat_template
    --model_args "model=${MODEL_NAME},base_url=${BASE_URL}/v1/chat/completions,num_concurrent=1,max_retries=3,timeout=3600,tokenized_requests=False"
    --gen_kwargs "${GEN_SAMPLING},max_gen_toks=${MAX_GEN_TOKS}"
    --seed 0
    --batch_size 1
    --log_samples
    --output_path "${OUTPUT_PATH}"
)
if [[ -n "${LIMIT}" && "${LIMIT}" != "full" ]]; then
  EVAL_COMMAND+=(--limit "${LIMIT}")
fi

set +e
OPENAI_API_KEY="${OPENAI_API_KEY:-local-server}" JAX_PLATFORMS=cpu HF_HOME="${HF_HOME:-/filestore/hf}" \
  timeout --signal=INT --kill-after=30 "${EVAL_TIMEOUT}" "${EVAL_COMMAND[@]}" "$@" 2>&1 | tee "${OUTPUT_PATH}/lm_eval.log"
status=${PIPESTATUS[0]}
set -e
if (( status == 124 )); then
  echo "lm_eval timed out after ${EVAL_TIMEOUT} s (results are incomplete; see ${SERVER_LOG})" >&2
elif (( status != 0 )); then
  echo "lm_eval exited with status ${status}" >&2
fi

echo "GSM8K evaluation complete: ${OUTPUT_PATH}; summarising (replays ${EXAMPLES:-3} problems for reasoning transcripts)"
"${PYTHON}" "${REPO_ROOT}/scripts/eval_musespark_accuracy.py" "${OUTPUT_PATH}" \
  --base-url "${BASE_URL}" --model "${MODEL_NAME}" --max-tokens "${MAX_GEN_TOKS}" --examples "${EXAMPLES:-3}" \
  | tee "${OUTPUT_PATH}/summary.md" || true

echo "Stopping the server"
cleanup
SERVER_PID=""
PROXY_PID=""

echo "Classifying failures"
"${PYTHON}" "${REPO_ROOT}/scripts/analyze_gsm8k_failures.py" "${OUTPUT_PATH}" \
  | tee "${OUTPUT_PATH}/failures.md" || true
exit "${status}"
