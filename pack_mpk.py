"""Create Muse Spark TP8/TP16/TP32 weight bundles from a local checkpoint."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Local checkpoint directory or repository ID already in the Hugging Face cache")
    parser.add_argument("--revision", default=None, help="Cached checkpoint revision for a repository ID")
    parser.add_argument("--tp", type=int, choices=(8, 16, 32), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory on model-weight storage")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent layer reads (default: 4)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"output already exists: {output}; choose a new directory")

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
    from model_paths import resolve_model_source
    from mpk import Config, load

    checkpoint = resolve_model_source(args.checkpoint, revision=args.revision, local_files_only=True)
    config = Config.from_checkpoint(checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    print(f"Packing {checkpoint} for TP{args.tp} into {output}", flush=True)
    base, dense = load.pack_from_checkpoint(checkpoint, config, tp=args.tp, workers=args.workers)
    load.save_pack(output / "base", base, progress=True)
    del base
    load.save_pack(output / "prefill", dense, progress=True)
    load.save_pack(output / "tiled", load.tile_dense(dense, config, args.tp), progress=True)
    (output / "pack.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "tp": args.tp, "format_version": 1,
    }, indent=2) + "\n")
    print(f"Ready: --weights {output}", flush=True)


if __name__ == "__main__":
    main()
