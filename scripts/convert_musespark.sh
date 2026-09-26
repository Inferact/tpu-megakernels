#!/bin/bash
# Convert the Muse Spark 1.2 HF checkpoint into the pre-sharded TP8 int4 container.
#
#   scripts/convert_musespark.sh [--layers 0] [--workers N] ...   (extra args go to the CLI)
#
# Runs detached under nohup; progress (GB/s, ETA) is appended to logs/convert.log and the
# conversion is resumable: re-run the script after an interruption to continue.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && source .env
SRC=${MUSESPARK_CHECKPOINT:-/filestore/weights/Muse-Spark-1.2-816B-A42B-open}
DST=${MUSESPARK_PRESHARDED:-/filestore/weights/muse-spark-tp8-int4}
mkdir -p logs
echo "converting $SRC -> $DST (log: logs/convert.log)"
nohup env JAX_PLATFORMS=cpu .venv/bin/python -m musespark.load convert \
    --src "$SRC" --dst "$DST" --tp 8 --group 128 "$@" >> logs/convert.log 2>&1 &
echo "pid $!"
