#!/usr/bin/env bash
# Interactive DeepSeek-V4.1-Flash + DSpark demo on the TP32 megakernel.
#
# Usage: with a 4-node allocation (salloc, or a batch job holding the nodes,
# e.g. `sbatch --wrap="sleep 12h"`), run this script from the login node or
# from any node, naming the job when SLURM_JOB_ID is not already set:
#   JOBID=<job id> bash scripts/demo_deepseek_v41_dspark.sh [--chat response|think|none] [--max-tokens N] [--baseline] [--prompt "..."]
set -uo pipefail

SRC="${DSV41_MEGAKERNEL_SRC:-}"
for candidate in "${SRC}" "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" "${SLURM_SUBMIT_DIR:-}"; do
  if [[ -n "${candidate}" && -f "${candidate}/demo_deepseek_v41_dspark.py" ]]; then SRC="${candidate}"; break; fi
done
: "${SRC:?cannot locate demo_deepseek_v41_dspark.py; set DSV41_MEGAKERNEL_SRC}"

if [[ -f "${SRC}/.env" ]]; then
  set -a
  source "${SRC}/.env"
  set +a
fi

PY="${DSV41_PY:-${SRC}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python3 || command -v python)"
fi
[[ -x "${PY}" ]] || { echo "Python environment not found; run 'uv sync' or set DSV41_PY" >&2; exit 1; }

cd "${SRC}"
JOBID="${JOBID:-${SLURM_JOB_ID:-}}"

if [[ -n "${JOBID}" ]] && command -v squeue >/dev/null 2>&1; then
  NODELIST="$(squeue -j "${JOBID}" -h -o %N)"
  : "${NODELIST:?Job ${JOBID} is not running}"
  mapfile -t HOSTS < <(scontrol show hostnames "${NODELIST}")
  NNODES="${#HOSTS[@]}"
  if [[ "${NNODES}" -eq 4 ]]; then
    worker_hosts=$(IFS=,; echo "${HOSTS[*]}")
    exec srun --jobid="${JOBID}" --overlap --nodes=4 --ntasks=4 --ntasks-per-node=1 --kill-on-bad-exit=1 --unbuffered --input=all --export=ALL \
      env PYTHONDONTWRITEBYTECODE=1 XLA_FLAGS="${XLA_FLAGS:---xla_allow_excess_precision=false}" \
      TPU_PROCESS_BOUNDS=1,1,4 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_CHIPS=0,1,2,3 \
      TPU_WORKER_HOSTNAMES="${worker_hosts}" TP32_COORDINATOR="${HOSTS[0]}:12377" JAX_PLATFORMS=tpu \
      HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" JAX_COMPILATION_CACHE_DIR="${SRC}/.jax_cache" PYTHONPATH="${SRC}" \
      "${PY}" distributed_entry.py demo_deepseek_v41_dspark.py "$@"
  else
    HOST="${HOST:-${HOSTS[0]}}"
    exec srun --jobid="${JOBID}" --overlap --nodes=1 --ntasks=1 --nodelist="${HOST}" --kill-on-bad-exit=1 --unbuffered --input=all --export=ALL \
      env PYTHONDONTWRITEBYTECODE=1 XLA_FLAGS="${XLA_FLAGS:---xla_allow_excess_precision=false}" \
      TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_CHIPS=0,1,2,3 \
      JAX_PLATFORMS=tpu HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" JAX_COMPILATION_CACHE_DIR="${SRC}/.jax_cache" PYTHONPATH="${SRC}" \
      "${PY}" demo_deepseek_v41_dspark.py "$@"
  fi
else
  exec env PYTHONDONTWRITEBYTECODE=1 XLA_FLAGS="${XLA_FLAGS:---xla_allow_excess_precision=false}" \
    JAX_PLATFORMS="${JAX_PLATFORMS:-tpu}" JAX_COMPILATION_CACHE_DIR="${SRC}/.jax_cache" PYTHONPATH="${SRC}" \
    "${PY}" demo_deepseek_v41_dspark.py "$@"
fi
