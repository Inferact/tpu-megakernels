#!/usr/bin/env bash
set -euo pipefail
SRC="${MPK_MEGAKERNEL_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mode=(--no-spec)
for argument in "$@"; do
  case "${argument}" in --draft|--draft=*|--no-spec) mode=() ;; esac
done
exec bash "${SRC}/scripts/demo_mpk_dspark.sh" "${mode[@]}" --tp 8 "$@"
