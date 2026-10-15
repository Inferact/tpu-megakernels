#!/usr/bin/env bash
set -euo pipefail
SRC="${MPK_MEGAKERNEL_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -f "${SRC}/.env" ]]; then
  set -a
  source "${SRC}/.env"
  set +a
fi
PY="${MPK_PY:-${SRC}/.venv/bin/python}"
[[ -x "${PY}" ]] || { echo "Run 'uv sync' in ${SRC} first" >&2; exit 1; }
for argument in "$@"; do
  if [[ "${argument}" == --help || "${argument}" == -h ]]; then
    exec "${PY}" "${SRC}/pack_mpk.py" "$@"
  fi
done
JOBID="${JOBID:-${SLURM_JOB_ID:-}}"
: "${JOBID:?Set JOBID to a running allocation with a high-memory compute node}"
node_args=()
if [[ -n "${HOST:-}" ]]; then node_args+=(--nodelist="${HOST}"); fi
cd "${SRC}"
exec srun --jobid="${JOBID}" --overlap --nodes=1 --ntasks=1 --cpus-per-task="${PACK_CPUS:-8}" \
  --kill-on-bad-exit=1 --unbuffered --export=ALL "${node_args[@]}" \
  env JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  "${PY}" "${SRC}/pack_mpk.py" "$@"
