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
AL=""
SERVER_ARGS=()
while (( $# )); do
  case "$1" in
    --help|-h)
      cat <<'USAGE'
Usage: JOBID=<job> bash scripts/benchmark_mpk.sh [options] [server arguments]
  --tp 8|16|32                  Default: MPK_TP or devices in the allocation
  --spec                        Enable normal DSpark acceptance
  --draft [checkpoint]          Enable DSpark, optionally using a local checkpoint
  --no-spec                     Non-speculative decoding (default)
  --fixed-acceptance-length 1..8 Enable DSpark with synthetic acceptance

Environment: HOST (TP8), PORT=8000, CONTEXT=4096, ISL=1, OSL=1000,
N_WARMUP=2, N_BENCH=24, OUTPUT_PATH, STARTUP_TIMEOUT=3600.
TP32 defaults to replicated experts; MPK_TP32_EXPERT_PARALLEL=1 selects TP8xEP4.
The script starts the server, runs the streaming client on compute, then stops it.
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
    --fixed-acceptance-length) AL="${2:?--fixed-acceptance-length requires 1..8}"; SPEC=1; shift 2 ;;
    --fixed-acceptance-length=*) AL="${1#*=}"; SPEC=1; shift ;;
    --) shift; SERVER_ARGS+=("$@"); break ;;
    *) SERVER_ARGS+=("$1"); shift ;;
  esac
done
if [[ -n "${TP}" && ! "${TP}" =~ ^(8|16|32)$ ]]; then echo "--tp must be 8, 16, or 32" >&2; exit 2; fi
if [[ -n "${AL}" ]]; then
  [[ "${AL}" =~ ^[1-8]$ ]] || { echo "acceptance length must be from 1 to 8" >&2; exit 2; }
  (( SPEC )) || { echo "synthetic acceptance requires DSpark" >&2; exit 2; }
fi
JOBID="${JOBID:-${SLURM_JOB_ID:-}}"
: "${JOBID:?Set JOBID to a running Slurm allocation}"
PORT="${PORT:-8000}"
CONTEXT="${CONTEXT:-4096}"
ISL="${ISL:-1}"
OSL="${OSL:-1000}"
MAX_TOKENS="${MAX_TOKENS:-${OSL}}"
N_WARMUP="${N_WARMUP:-2}"
N_BENCH="${N_BENCH:-24}"
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
    [[ ${#HOSTS[@]} -eq $((TP / 8)) ]] || { echo "TP${TP} needs $((TP / 8)) allocated hosts" >&2; exit 2; }
    ;;
  *) echo "Use --tp 8, 16, or 32" >&2; exit 2 ;;
esac
LAUNCHER="${SRC}/scripts/demo_mpk_dspark.sh"
MODE_ARGS=()
if [[ "${TP}" == 32 ]]; then
  LAUNCHER="${SRC}/scripts/demo_mpk_tp32.sh"
  if (( SPEC )); then MODE_ARGS=(--draft); fi
elif (( ! SPEC )); then
  MODE_ARGS=(--no-spec)
fi
if (( SPEC )) && [[ -n "${DRAFT}" ]]; then MODE_ARGS=(--draft "${DRAFT}"); fi
if [[ -n "${AL}" ]]; then MODE_ARGS+=(--fixed-acceptance-length "${AL}"); fi
MODE=non-spec
if (( SPEC )); then MODE=dspark; fi
if [[ -n "${AL}" ]]; then MODE="dspark-al${AL}"; fi
OUTPUT_PATH="${OUTPUT_PATH:-${SRC}/benchmark-results/mpk-tp${TP}-${MODE}-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUTPUT_PATH}"
SERVER_LOG="${OUTPUT_PATH}/server.log"
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
if (( SPEC )); then MODE_ARGS+=(--dspark-stats "${OUTPUT_PATH}/dspark.jsonl"); fi
echo "Starting Muse-Spark TP${TP} ${MODE} server (log: ${SERVER_LOG})"
MPK_MEGAKERNEL_SRC="${SRC}" JOBID="${JOBID}" MPK_TP="${TP}" \
  bash "${LAUNCHER}" --tp "${TP}" "${MODE_ARGS[@]}" \
    --serve "0.0.0.0:${PORT}" --context "${CONTEXT}" --max-tokens "${MAX_TOKENS}" \
    "${SERVER_ARGS[@]}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
SERVER_HOST=""
deadline=$((SECONDS + STARTUP_TIMEOUT))
while [[ -z "${SERVER_HOST}" ]]; do
  kill -0 "${SERVER_PID}" 2>/dev/null || { tail -n 80 "${SERVER_LOG}" >&2; exit 1; }
  for candidate in "${HOSTS[@]}"; do
    if curl --noproxy '*' --connect-timeout 1 --max-time 2 --fail --silent \
        "http://${candidate}:${PORT}/health" >/dev/null 2>&1; then
      SERVER_HOST="${candidate}"
      break
    fi
  done
  [[ -n "${SERVER_HOST}" ]] && break
  if (( SECONDS >= deadline )); then
    echo "server did not become ready within ${STARTUP_TIMEOUT} seconds" >&2
    tail -n 80 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 2
done
echo "Server ready; ${N_WARMUP} warmups and ${N_BENCH} measured requests (ISL=${ISL} OSL=${OSL})"
srun --jobid="${JOBID}" --overlap --nodes=1 --ntasks=1 --nodelist="${SERVER_HOST}" \
  env ISL="${ISL}" OSL="${OSL}" N_WARMUP="${N_WARMUP}" N_BENCH="${N_BENCH}" \
  "${SRC}/.venv/bin/python" "${SRC}/bench_openai_stream.py" \
  --base-url "http://${SERVER_HOST}:${PORT}" | tee "${OUTPUT_PATH}/bench.log"
