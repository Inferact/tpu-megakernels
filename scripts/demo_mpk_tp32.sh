#!/usr/bin/env bash
# Muse-Spark-Open TP32 demo on a four-host Slurm allocation.
# Usage: JOBID=<job> bash scripts/demo_mpk_tp32.sh [demo arguments]
# Set MPK_PACKED_WEIGHTS_TP32 or pass --weights; run one demo per allocation.
set -uo pipefail
JOBID="${JOBID:-${SLURM_JOB_ID:-}}"
: "${JOBID:?Set JOBID=<slurm job id> of a 4-node allocation (or run inside salloc)}"
SRC="${MPK_MEGAKERNEL_SRC:-}"
for candidate in "${SRC}" "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" "${SLURM_SUBMIT_DIR:-}"; do
  if [[ -n "${candidate}" && -f "${candidate}/demo_mpk_tp32.py" ]]; then SRC="${candidate}"; break; fi
done
: "${SRC:?cannot locate demo_mpk_tp32.py; set MPK_MEGAKERNEL_SRC}"
if [[ -f "${SRC}/.env" ]]; then
  set -a
  source "${SRC}/.env"
  set +a
fi
PY="${MPK_PY:-${SRC}/.venv/bin/python}"
[[ -x "${PY}" ]] || { echo "Python environment not found at ${PY}; run 'uv sync' or set MPK_PY" >&2; exit 1; }
requested_tp="${MPK_TP:-32}"
arguments=("$@")
for ((i=0; i<${#arguments[@]}; i++)); do
  if [[ "${arguments[i]}" == --tp && $((i + 1)) -lt ${#arguments[@]} ]]; then requested_tp="${arguments[i+1]}"; fi
  if [[ "${arguments[i]}" == --tp=* ]]; then requested_tp="${arguments[i]#--tp=}"; fi
done
[[ "${requested_tp}" == 32 ]] || { echo "TP8/TP16 use scripts/demo_mpk_dspark.sh" >&2; exit 1; }
NODELIST="$(squeue -j "${JOBID}" -h -o %N)"
: "${NODELIST:?Job ${JOBID} is not running}"
mapfile -t HOSTS < <(scontrol show hostnames "${NODELIST}")
[[ ${#HOSTS[@]} -eq 4 ]] || { echo "job ${JOBID} holds ${#HOSTS[@]} nodes, need 4" >&2; exit 1; }
export MPK_TP=32
worker_hosts=$(IFS=,; echo "${HOSTS[*]}")
cd "${SRC}"
# stdin goes to every task (--input=<taskid> does not reliably pick SLURM_PROCID 0);
# only JAX process 0 reads it, through the descriptor distributed_entry.py saves.
exec srun --jobid="${JOBID}" --overlap --nodes="${#HOSTS[@]}" --ntasks="${#HOSTS[@]}" --ntasks-per-node=1 --kill-on-bad-exit=1 --unbuffered --input=all --export=ALL \
  env PYTHONDONTWRITEBYTECODE=1 NUMPY_MADVISE_HUGEPAGE="${NUMPY_MADVISE_HUGEPAGE:-0}" \
  XLA_FLAGS="${XLA_FLAGS:---xla_allow_excess_precision=false}" \
  TPU_PROCESS_BOUNDS="1,1,${#HOSTS[@]}" TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_CHIPS=0,1,2,3 \
  TPU_WORKER_HOSTNAMES="${worker_hosts}" TP32_COORDINATOR="${HOSTS[0]}:12378" JAX_PLATFORMS=tpu \
  HF_HUB_OFFLINE=1 JAX_COMPILATION_CACHE_DIR="${SRC}/.jax_cache" PYTHONPATH="${SRC}" \
  "${PY}" distributed_entry.py "${MPK_ENTRY:-demo_mpk_tp32.py}" "$@"
