#!/usr/bin/env bash
set -euo pipefail
SRC="${MPK_MEGAKERNEL_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -f "${SRC}/.env" ]]; then
  set -a
  source "${SRC}/.env"
  set +a
fi
JOBID="${JOBID:-${SLURM_JOB_ID:-}}"
: "${JOBID:?Set JOBID to a running Slurm allocation}"
PY="${MPK_PY:-${SRC}/.venv/bin/python}"
[[ -x "${PY}" ]] || { echo "Run 'uv sync' in ${SRC} first" >&2; exit 1; }
NODELIST="$(squeue -j "${JOBID}" -h -o %N)"
: "${NODELIST:?Job ${JOBID} is not running}"
mapfile -t HOSTS < <(scontrol show hostnames "${NODELIST}")
tp="${MPK_TP:-$((8 * ${#HOSTS[@]}))}"
arguments=("$@")
for ((i=0; i<${#arguments[@]}; i++)); do
  if [[ "${arguments[i]}" == --tp && $((i + 1)) -lt ${#arguments[@]} ]]; then tp="${arguments[i+1]}"; fi
  if [[ "${arguments[i]}" == --tp=* ]]; then tp="${arguments[i]#--tp=}"; fi
done
command=("${PY}")
distributed_env=()
case "${tp}" in
  8) HOSTS=("${HOST:-${HOSTS[0]}}") ;;
  16)
    [[ ${#HOSTS[@]} -eq 2 ]] || { echo "TP16 needs a two-host allocation" >&2; exit 1; }
    worker_hosts=$(IFS=,; echo "${HOSTS[*]}")
    distributed_env=(TPU_WORKER_HOSTNAMES="${worker_hosts}" TP32_COORDINATOR="${HOSTS[0]}:12378")
    command+=(distributed_entry.py)
    ;;
  *) echo "Use --tp 8 or --tp 16; TP32 uses scripts/demo_mpk_tp32.sh" >&2; exit 1 ;;
esac
worker_hosts=$(IFS=,; echo "${HOSTS[*]}")
cd "${SRC}"
exec srun --jobid="${JOBID}" --overlap --nodes="${#HOSTS[@]}" --ntasks="${#HOSTS[@]}" --ntasks-per-node=1 \
  --nodelist="${worker_hosts}" --kill-on-bad-exit=1 --unbuffered --input=all --export=ALL \
  env PYTHONDONTWRITEBYTECODE=1 NUMPY_MADVISE_HUGEPAGE="${NUMPY_MADVISE_HUGEPAGE:-0}" \
  XLA_FLAGS="${XLA_FLAGS:---xla_allow_excess_precision=false}" \
  TPU_PROCESS_BOUNDS="1,1,${#HOSTS[@]}" TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_CHIPS=0,1,2,3 \
  "${distributed_env[@]}" MPK_TP="${tp}" JAX_PLATFORMS=tpu HF_HUB_OFFLINE=1 \
  JAX_COMPILATION_CACHE_DIR="${SRC}/.jax_cache" PYTHONPATH="${SRC}" \
  "${command[@]}" "${MPK_ENTRY:-demo_mpk_dspark.py}" "$@"
