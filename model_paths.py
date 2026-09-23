"""Resolve model repository IDs through the local Hugging Face cache."""

import os
from pathlib import Path

from huggingface_hub import snapshot_download


def resolve_model_source(
    source: str | Path, *, revision: str | None = None, local_files_only: bool = False
) -> Path:
    """Return ``source`` as a local directory, downloading/caching a repo ID if needed."""
    path = Path(source).expanduser()
    if path.is_dir():
        return path
    if path.is_absolute() or str(source).startswith("."):
        raise FileNotFoundError(f"model directory not found: {path}")
    hf_home = os.environ.get("HF_HOME")
    cache_dir = Path(hf_home).expanduser() / "hub" if hf_home else None
    if cache_dir is not None:
        repository_cache = cache_dir / f"models--{str(source).replace('/', '--')}"
        commit = revision
        ref = repository_cache / "refs" / (revision or "main")
        if ref.is_file():
            commit = ref.read_text().strip()
        snapshot = repository_cache / "snapshots" / commit if commit else None
        if snapshot is not None and snapshot.is_dir():
            return snapshot
    return Path(
        snapshot_download(
            repo_id=str(source),
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )
