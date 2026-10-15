#!/usr/bin/env bash
set -euo pipefail

SRC="${MPK_MEGAKERNEL_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -f "${SRC}/.env" ]]; then
  set -a
  source "${SRC}/.env"
  set +a
fi
cd "${SRC}"
TP="${MPK_TP:-}"
SPEC=0
DRAFT=""
REUSE=0
BATCH_SIZE="${BATCH_SIZE:-1}"
SERVER_ARGS=()
EVAL_ARGS=()
while (( $# )); do
  case "$1" in
    --help|-h)
      cat <<'USAGE'
Usage: JOBID=<job> bash scripts/eval_mpk_gsm8k.sh [options] [server arguments] [-- lm_eval arguments]
  --tp 8|16|32         Default: MPK_TP or devices in the allocation
  --spec               Enable normal DSpark acceptance
  --draft [checkpoint] Enable DSpark, optionally using a local checkpoint
  --no-spec            Non-speculative decoding (default)
  --batch-size 1|2|4|8  Maximum server batch size (default: 1)
  --reuse-server       Evaluate a running server without starting or stopping it

Environment: HOST (TP8 or reuse), PORT=8000, CONTEXT=4096, START=0,
END=1319 (or LIMIT), MAX_GEN_TOKS=384, NUM_CONCURRENT=BATCH_SIZE,
MODEL_NAME (otherwise discovered), OUTPUT_PATH, STARTUP_TIMEOUT=3600.
TP32 defaults to replicated experts; MPK_TP32_EXPERT_PARALLEL=1 enables batches.
GSM8K runs on compute with five-shot greedy decoding and seed 0.
USAGE
      exit 0 ;;
    --tp) TP="${2:?--tp requires 8, 16, or 32}"; shift 2 ;;
    --tp=*) TP="${1#*=}"; shift ;;
    --spec) SPEC=1; shift ;;
    --no-spec) SPEC=0; DRAFT=""; shift ;;
    --draft)
      SPEC=1
      DRAFT=""
      shift
      if (( $# )) && [[ "$1" != --* ]]; then DRAFT="$1"; shift; fi
      ;;
    --draft=*) SPEC=1; DRAFT="${1#*=}"; shift ;;
    --batch-size) BATCH_SIZE="${2:?--batch-size requires 1, 2, 4, or 8}"; shift 2 ;;
    --batch-size=*) BATCH_SIZE="${1#*=}"; shift ;;
    --reuse-server) REUSE=1; shift ;;
    --fixed-acceptance-length|--fixed-acceptance-length=*)
      echo "GSM8K evaluation requires normal acceptance" >&2; exit 2 ;;
    --) shift; EVAL_ARGS=("$@"); break ;;
    *) SERVER_ARGS+=("$1"); shift ;;
  esac
done
if [[ -n "${TP}" && ! "${TP}" =~ ^(8|16|32)$ ]]; then echo "--tp must be 8, 16, or 32" >&2; exit 2; fi
[[ "${BATCH_SIZE}" =~ ^(1|2|4|8)$ ]] || { echo "--batch-size must be 1, 2, 4, or 8" >&2; exit 2; }
if (( SPEC )) && [[ "${BATCH_SIZE}" != 1 ]]; then echo "DSpark requires --batch-size 1" >&2; exit 2; fi
if (( REUSE )) && (( ${#SERVER_ARGS[@]} )); then echo "Server arguments require launching a server" >&2; exit 2; fi
JOBID="${JOBID:-${SLURM_JOB_ID:-}}"
: "${JOBID:?Set JOBID to a running Slurm allocation}"
PORT="${PORT:-8000}"
CONTEXT="${CONTEXT:-4096}"
START="${START:-0}"
END="${END:-${LIMIT:-1319}}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-384}"
NUM_CONCURRENT="${NUM_CONCURRENT:-${BATCH_SIZE}}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}"
NODELIST="$(squeue -j "${JOBID}" -h -o %N)"
: "${NODELIST:?Slurm job ${JOBID} is not running}"
mapfile -t HOSTS < <(scontrol show hostnames "${NODELIST}")
TP="${TP:-$((8 * ${#HOSTS[@]}))}"
case "${TP}" in
  8)
    HOST="${HOST:-${SERVER_HOST:-${HOSTS[0]}}}"
    [[ " ${HOSTS[*]} " == *" ${HOST} "* ]] || { echo "HOST must belong to job ${JOBID}" >&2; exit 2; }
    HOSTS=("${HOST}")
    export HOST
    ;;
  16|32)
    if (( ! REUSE )); then
      [[ ${#HOSTS[@]} -eq $((TP / 8)) ]] || { echo "TP${TP} needs $((TP / 8)) allocated hosts" >&2; exit 2; }
    fi
    ;;
  *) echo "Use --tp 8, 16, or 32" >&2; exit 2 ;;
esac
if (( REUSE )) && [[ -n "${HOST:-}" ]]; then HOSTS=("${HOST}"); fi
MODE=non-spec
if (( SPEC )); then MODE=dspark; fi
if (( REUSE )); then MODE=server; fi
OUTPUT_PATH="${OUTPUT_PATH:-${SRC}/eval-results/gsm8k-mpk-tp${TP}-${MODE}-bs${BATCH_SIZE}-${START}-${END}-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUTPUT_PATH}"
SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if (( ! REUSE )); then
  LAUNCHER="${SRC}/scripts/demo_mpk_dspark.sh"
  MODE_ARGS=()
  if [[ "${TP}" == 32 ]]; then
    LAUNCHER="${SRC}/scripts/demo_mpk_tp32.sh"
    if (( SPEC )); then MODE_ARGS=(--draft); fi
  elif (( ! SPEC )); then
    MODE_ARGS=(--no-spec)
  fi
  if (( SPEC )) && [[ -n "${DRAFT}" ]]; then MODE_ARGS=(--draft "${DRAFT}"); fi
  if (( SPEC )); then MODE_ARGS+=(--dspark-stats "${OUTPUT_PATH}/dspark.jsonl"); fi
  MPK_MEGAKERNEL_SRC="${SRC}" JOBID="${JOBID}" MPK_TP="${TP}" \
    bash "${LAUNCHER}" --tp "${TP}" "${MODE_ARGS[@]}" \
      --serve "0.0.0.0:${PORT}" --context "${CONTEXT}" --max-tokens "${MAX_GEN_TOKS}" \
      --batch-size "${BATCH_SIZE}" "${SERVER_ARGS[@]}" > "${OUTPUT_PATH}/server.log" 2>&1 &
  SERVER_PID=$!
fi
SERVER_HOST=""
deadline=$((SECONDS + STARTUP_TIMEOUT))
while [[ -z "${SERVER_HOST}" ]]; do
  if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -n 80 "${OUTPUT_PATH}/server.log" >&2
    exit 1
  fi
  for candidate in "${HOSTS[@]}"; do
    if curl --noproxy '*' --connect-timeout 1 --max-time 2 --fail --silent \
        "http://${candidate}:${PORT}/health" >/dev/null 2>&1; then
      SERVER_HOST="${candidate}"
      break
    fi
  done
  [[ -n "${SERVER_HOST}" ]] && break
  (( SECONDS < deadline )) || { echo "server startup timed out" >&2; exit 1; }
  sleep 2
done
BASE_URL="http://${SERVER_HOST}:${PORT}"
MODEL_NAME="${MODEL_NAME:-$(curl --noproxy '*' --fail --silent "${BASE_URL}/v1/models" |
  "${SRC}/.venv/bin/python" -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')}"
srun --jobid="${JOBID}" --overlap --nodes=1 --ntasks=1 --nodelist="${SERVER_HOST}" --time=03:00:00 \
  env OPENAI_API_KEY="${OPENAI_API_KEY:-local-server}" JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 \
  bash -c '
    samples=$(.venv/bin/python -c "import json,sys; print(json.dumps({\"gsm8k\": list(range(int(sys.argv[1]), int(sys.argv[2])))}))" "$2" "$3")
    exec uv run --extra eval lm_eval run --tasks gsm8k --num_fewshot 5 \
      --gen_kwargs "do_sample=False,temperature=0.0,top_p=1.0,max_gen_toks=$6" \
      --seed 0 --batch_size 1 --samples "$samples" --log_samples --output_path "$4" \
      --model local-completions \
      --model_args "model=$5,base_url=$1/v1/completions,tokenized_requests=False,tokenizer_backend=None,num_concurrent=$7,max_retries=3,timeout=3600" "${@:8}"
  ' bash "${BASE_URL}" "${START}" "${END}" "${OUTPUT_PATH}" "${MODEL_NAME}" "${MAX_GEN_TOKS}" "${NUM_CONCURRENT}" "${EVAL_ARGS[@]}"
