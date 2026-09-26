#!/bin/bash
# Stream the vendor NVFP4 Muse Spark 1.2 checkpoint (meta-models/Muse-Spark-1.2-816B-A42B-NVFP4-open,
# 533 GB, 127 shards) from the Hub into the pre-sharded TP8 format-v2 container.
#
#   scripts/convert_musespark_nvfp4.sh [--workers N] [--max-shards 3] ...   (extra args go to the CLI)
#
# Shards are downloaded one at a time into $HF_HOME (xet, high performance), converted and
# deleted; at most --max-shards shards live on disk. Runs detached under nohup; progress
# (MB/s, ETA, units, free space) is appended to logs/convert_nvfp4.log and the conversion is
# resumable: re-run the script after an interruption to continue (progress.json per unit).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && source .env
SRC=${MUSESPARK_NVFP4_REPO:-meta-models/Muse-Spark-1.2-816B-A42B-NVFP4-open}
DST=${MUSESPARK_NVFP4_PRESHARDED:-/filestore/weights/muse-spark-tp8-nvfp4}
export HF_HOME=${HF_HOME:-/filestore/hf}
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
mkdir -p logs
echo "converting $SRC -> $DST (log: logs/convert_nvfp4.log)"
nohup env JAX_PLATFORMS=cpu .venv/bin/python -m musespark.load convert-nvfp4 \
    --src "$SRC" --dst "$DST" --tp 8 "$@" >> logs/convert_nvfp4.log 2>&1 &
echo "pid $!"
