#!/usr/bin/env bash
# Interactive Kimi K3 + DSpark demo on the TP32 megakernel.
#
# Usage: with a 4-node allocation (salloc, or a batch job holding the nodes,
# e.g. `sbatch --wrap="sleep 12h"`), run this script from the login node or
# from any node, naming the job when SLURM_JOB_ID is not already set:
#   JOBID=<job id> bash scripts/demo_kimi_dspark.sh [--chat response|think|none] [--max-tokens N] [--baseline] [--prompt "..."]
# Loading the weights takes about 3 minutes warm (up to 20 cold); then type
# prompts at "prompt>" (empty line or 'quit' ends). Only one demo per allocation.
set -uo pipefail
JOBID="${JOBID:-${SLURM_JOB_ID:-}}"
: "${JOBID:?Set JOBID=<slurm job id> of the 4-node allocation (or run inside salloc)}"
# The repository: K3_MEGAKERNEL_SRC, else the script's parent directory (sbatch
# copies the script to the spool directory, so fall back to the submit
# directory. K3_MEGAKERNEL_SRC is the explicit override when launching a
# copied script from somewhere else.
DSPARK="${K3_MEGAKERNEL_SRC:-}"
for candidate in "${DSPARK}" "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" "${SLURM_SUBMIT_DIR:-}"; do
  if [[ -n "${candidate}" && -f "${candidate}/demo_kimi_dspark.py" ]]; then DSPARK="${candidate}"; break; fi
done
: "${DSPARK:?cannot locate demo_kimi_dspark.py; set K3_MEGAKERNEL_SRC}"
if [[ -f "${DSPARK}/.env" ]]; then
  set -a
  source "${DSPARK}/.env"
  set +a
fi
PY="${K3_PY:-${DSPARK}/.venv/bin/python}"
[[ -x "${PY}" ]] || { echo "Python environment not found at ${PY}; run 'uv sync' or set K3_PY" >&2; exit 1; }
NODELIST="$(squeue -j "${JOBID}" -h -o %N)"
: "${NODELIST:?Job ${JOBID} is not running}"
mapfile -t HOSTS < <(scontrol show hostnames "${NODELIST}")
[[ ${#HOSTS[@]} -eq 4 ]] || { echo "job ${JOBID} holds ${#HOSTS[@]} nodes, need 4" >&2; exit 1; }
worker_hosts=$(IFS=,; echo "${HOSTS[*]}")
cd "${DSPARK}"
# stdin goes to every task (--input=<taskid> does not reliably pick SLURM_PROCID 0);
# only JAX process 0 reads it, through the descriptor distributed_entry.py saves.
exec srun --jobid="${JOBID}" --overlap --nodes=4 --ntasks=4 --ntasks-per-node=1 --kill-on-bad-exit=1 --unbuffered --input=all --export=ALL \
  env PYTHONDONTWRITEBYTECODE=1 XLA_FLAGS="${XLA_FLAGS:---xla_allow_excess_precision=false}" \
  TPU_PROCESS_BOUNDS=1,1,4 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_CHIPS=0,1,2,3 \
  TPU_WORKER_HOSTNAMES="${worker_hosts}" TP32_COORDINATOR="${HOSTS[0]}:12377" JAX_PLATFORMS=tpu \
  HF_HUB_OFFLINE=1 JAX_COMPILATION_CACHE_DIR="${DSPARK}/.jax_cache" PYTHONPATH="${DSPARK}" \
  "${PY}" distributed_entry.py demo_kimi_dspark.py "$@"
