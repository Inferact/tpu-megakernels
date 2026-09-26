#!/usr/bin/env bash
# Muse Spark 1.2 demo / server / benchmark on the TP8 decode megakernel (single host, no Slurm).
#
# Usage:
#   bash scripts/demo_musespark.sh [--prompt "..."] [--greedy] [--max-tokens N] [--serve 0.0.0.0:8000] [--bench]
#
# One process drives the eight devices of four TPU chips (TPU_VISIBLE_CHIPS=0,1,2,3). Weights
# come from the pre-sharded int4 container (--weights, default /filestore/weights/muse-spark-tp8-int4)
# and the tokenizer from the HF snapshot (--checkpoint). Then type prompts at "prompt>" (empty
# line or 'quit' ends).
#
# During development the chips are shared through the lock helper: run the launcher via
#   TPU_RUN="<scratchpad>/tpu_run.sh all" bash scripts/demo_musespark.sh ...
# (TPU_RUN is prefixed to the python command; `tpu_run.sh all` takes the four chip locks so no
# other agent's program runs on the chips at the same time.)
set -uo pipefail
SRC="${MUSESPARK_MEGAKERNEL_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[[ -f "${SRC}/demo_musespark.py" ]] || { echo "cannot locate demo_musespark.py; set MUSESPARK_MEGAKERNEL_SRC" >&2; exit 1; }
if [[ -f "${SRC}/.env" ]]; then
  set -a
  source "${SRC}/.env"
  set +a
fi
PY="${MUSESPARK_PY:-${SRC}/.venv/bin/python}"
[[ -x "${PY}" ]] || { echo "Python environment not found at ${PY}; run 'uv sync' or set MUSESPARK_PY" >&2; exit 1; }
cd "${SRC}"
# shellcheck disable=SC2086  # TPU_RUN is an optional command prefix ("<path>/tpu_run.sh all")
exec ${TPU_RUN:-} env PYTHONDONTWRITEBYTECODE=1 XLA_FLAGS="${XLA_FLAGS:---xla_allow_excess_precision=false}" \
  TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_VISIBLE_CHIPS=0,1,2,3 \
  JAX_PLATFORMS=tpu JAX_COMPILATION_CACHE_DIR="${SRC}/.jax_cache" PYTHONPATH="${SRC}" \
  HF_HUB_OFFLINE=1 \
  "${PY}" demo_musespark.py "$@"
