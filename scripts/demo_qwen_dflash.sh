#!/usr/bin/env bash
# Interactive Qwen3.8-27B + DFlash2 demo on the TP8 megakernel.
#
# Usage: with a Slurm allocation holding at least one node (salloc, or a batch
# job holding the nodes, e.g. `sbatch --wrap="sleep 12h"`), run this script
# from the login node or from any node, naming the job when SLURM_JOB_ID is
# not already set:
#   JOBID=<job id> bash scripts/demo_qwen_dflash.sh [--chat response|think|none] [--max-tokens N] [--baseline] [--prompt "..."]
# The demo runs on one node (the first of the allocation): four TPU chips,
# eight devices, one process. Loading the weights takes about 2 minutes with
# packed containers (up to 10 from the raw checkpoint); then type prompts at
# "prompt>" (empty line or 'quit' ends). Only one demo per node.
set -uo pipefail
JOBID="${JOBID:-${SLURM_JOB_ID:-}}"
: "${JOBID:?Set JOBID=<slurm job id> of an allocation (or run inside salloc)}"
# The repository: QWEN_MEGAKERNEL_SRC, else the script's parent directory
# (sbatch copies the script to the spool directory, so fall back to the
# submit directory).
SRC="${QWEN_MEGAKERNEL_SRC:-}"
for candidate in "${SRC}" "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" "${SLURM_SUBMIT_DIR:-}"; do
  if [[ -n "${candidate}" && -f "${candidate}/demo_qwen_dflash.py" ]]; then SRC="${candidate}"; break; fi
done
: "${SRC:?cannot locate demo_qwen_dflash.py; set QWEN_MEGAKERNEL_SRC}"
if [[ -f "${SRC}/.env" ]]; then
  set -a
  source "${SRC}/.env"
  set +a
fi
PY="${QWEN_PY:-${SRC}/.venv/bin/python}"
[[ -x "${PY}" ]] || { echo "Python environment not found at ${PY}; run 'uv sync' or set QWEN_PY" >&2; exit 1; }
NODELIST="$(squeue -j "${JOBID}" -h -o %N)"
: "${NODELIST:?Job ${JOBID} is not running}"
mapfile -t HOSTS < <(scontrol show hostnames "${NODELIST}")
HOST="${HOST:-${HOSTS[0]}}"
cd "${SRC}"
exec srun --jobid="${JOBID}" --overlap --nodes=1 --ntasks=1 --nodelist="${HOST}" --kill-on-bad-exit=1 --unbuffered --input=all --export=ALL \
  env PYTHONDONTWRITEBYTECODE=1 XLA_FLAGS="${XLA_FLAGS:---xla_allow_excess_precision=false}" \
  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_CHIPS=0,1,2,3 \
  JAX_PLATFORMS=tpu JAX_COMPILATION_CACHE_DIR="${SRC}/.jax_cache" PYTHONPATH="${SRC}" \
  HF_HUB_OFFLINE=1 \
  "${PY}" demo_qwen_dflash.py "$@"
