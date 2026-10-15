"""Run a packer with parallel GCS uploads for its sequential binary writes."""

import argparse
import base64
import builtins
import fnmatch
import io
import logging
import os
import runpy
import sys
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import google_crc32c
from dotenv import load_dotenv
from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage

logger = logging.getLogger(__name__)


class GCSParallelWriter:
    """Sequential binary writer; closing publishes a new, checksum-verified object."""

    def __init__(self, bucket, name, *, workers=16, chunk_size=128 << 20):
        if workers < 1 or chunk_size < 1:
            raise ValueError("workers and chunk_size must be positive")
        if bucket.blob(name).exists():
            raise FileExistsError(f"gs://{bucket.name}/{name}")
        self.bucket = bucket
        self.name = name
        self.workers = workers
        self.chunk_size = chunk_size
        self.upload_id = uuid.uuid4().hex
        self.prefix = f"{name}.parts-{self.upload_id}/"
        self.pool = ThreadPoolExecutor(workers)
        self.pending = deque()
        self.parts = []
        self.checksum = google_crc32c.Checksum()
        self.total = 0
        self.count = 0
        self.closed = False
        self.failed = False
        self.published = False
        self.start = time.monotonic()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.failed |= exc_type is not None
        self.close()

    def writable(self):
        return not self.closed

    def seekable(self):
        return False

    def fileno(self):
        raise io.UnsupportedOperation("GCSParallelWriter supports sequential write() only")

    def tell(self):
        return self.total

    def _upload(self, name, data):
        blob = self.bucket.blob(name)
        blob.upload_from_string(
            data, content_type="application/octet-stream", checksum="crc32c",
            if_generation_match=0, timeout=(30, 300),
        )
        if blob.size != len(data):
            raise OSError(f"Uploaded size mismatch: {name}")
        return blob

    def _compose(self, name, sources, *, final=False):
        blob = self.bucket.blob(name)
        if final:
            blob.metadata = {"gcs_parallel_pack": self.upload_id}
        try:
            blob.compose(
                sources, if_generation_match=0,
                if_source_generation_match=[part.generation for part in sources],
                timeout=(30, 300),
            )
        except PreconditionFailed:
            blob.reload()
            if not final or (blob.metadata or {}).get("gcs_parallel_pack") != self.upload_id:
                raise
        return blob

    def _finish_one(self):
        self.parts.append(self.pending.popleft().result())

    def write(self, data):
        if self.closed or self.failed:
            raise ValueError("write to a closed or failed GCS upload")
        view = memoryview(data).cast("B")
        try:
            for offset in range(0, len(view), self.chunk_size):
                if len(self.pending) >= 2 * self.workers:
                    self._finish_one()
                chunk = bytes(view[offset:offset + self.chunk_size])
                self.checksum.update(chunk)
                name = f"{self.prefix}{self.count:08d}"
                self.pending.append(self.pool.submit(self._upload, name, chunk))
                self.count += 1
                self.total += len(chunk)
        except BaseException:
            self.failed = True
            raise
        return len(view)

    def flush(self):
        if self.closed or self.failed:
            raise ValueError("flush of a closed or failed GCS upload")
        try:
            while self.pending:
                self._finish_one()
        except BaseException:
            self.failed = True
            raise

    @staticmethod
    def _delete(blob):
        try:
            blob.delete(if_generation_match=blob.generation)
        except NotFound:
            pass

    def _cleanup(self):
        try:
            if not self.published:
                blob = self.bucket.get_blob(self.name)
                if blob and (blob.metadata or {}).get("gcs_parallel_pack") == self.upload_id:
                    self._delete(blob)
            with ThreadPoolExecutor(self.workers) as pool:
                list(pool.map(self._delete, self.bucket.list_blobs(prefix=self.prefix)))
        except Exception:
            logger.exception("Could not clean up upload gs://%s/%s", self.bucket.name, self.prefix)

    def close(self):
        if self.closed:
            return
        try:
            if self.failed:
                return
            self.flush()
            parts = self.parts or [self._upload(f"{self.prefix}empty", b"")]
            level = 0
            while len(parts) > 32:
                futures = [
                    self.pool.submit(self._compose, f"{self.prefix}level{level}-{i}", parts[i:i+32])
                    for i in range(0, len(parts), 32)
                ]
                parts = [future.result() for future in futures]
                level += 1
            blob = self._compose(self.name, parts, final=True)
            expected = base64.b64encode(self.checksum.digest()).decode("ascii")
            if blob.size != self.total or blob.crc32c != expected:
                raise OSError(f"Composed size or CRC32C mismatch: gs://{self.bucket.name}/{self.name}")
            self.published = True
            print(
                f"Uploaded gs://{self.bucket.name}/{self.name}: "
                f"{self.total / (1 << 30):.2f} GiB in {time.monotonic() - self.start:.1f}s",
                flush=True,
            )
        finally:
            self.closed = True
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pending.clear()
            self._cleanup()


@contextmanager
def parallel_pack_writes(bucket, mount_root, *, patterns=("pack.bin",), workers=16,
                         chunk_size=128 << 20):
    """Redirect matching Python binary writes under a bucket mount to GCS."""
    mount_root = Path(mount_root).expanduser().resolve()
    original_open, original_io_open = builtins.open, io.open
    writers = []

    def parallel_open(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            path = Path(file).expanduser().resolve()
            if path.is_relative_to(mount_root) and any(flag in mode for flag in "wax+") and any(
                fnmatch.fnmatchcase(path.name, pattern) for pattern in patterns
            ):
                if mode not in ("wb", "bw", "xb", "bx"):
                    raise ValueError(f"Parallel GCS writes require wb or xb mode: {path}")
                writer = GCSParallelWriter(
                    bucket, path.relative_to(mount_root).as_posix(),
                    workers=workers, chunk_size=chunk_size,
                )
                writers.append(writer)
                return writer
        return original_open(file, mode, *args, **kwargs)

    builtins.open = io.open = parallel_open
    try:
        yield
        for writer in writers:
            writer.close()
    finally:
        builtins.open, io.open = original_open, original_io_open
        for writer in writers:
            if not writer.closed:
                writer.failed = True
                writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(__file__).with_name(".env"))
    parser.add_argument("--mount-root", type=Path, help="GCS bucket mount (default: GCS_MOUNT_ROOT or /mnt/models)")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent uploads (default: 16)")
    parser.add_argument("--chunk-mib", type=int, default=128, help="Upload chunk size (default: 128 MiB)")
    parser.add_argument("--pattern", action="append", help="Binary filename glob; repeatable (default: pack.bin)")
    parser.add_argument("script", type=Path, help="Python packing script to run on this compute node")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    bucket_name = os.environ.get("GCS_BUCKET", "").strip()
    if not bucket_name or "/" in bucket_name:
        parser.error("Set GCS_BUCKET to the bucket name (without gs://) in .env or the environment")
    if args.workers < 1 or args.chunk_mib < 1:
        parser.error("--workers and --chunk-mib must be positive")
    script = args.script.expanduser().resolve()
    if not script.is_file():
        parser.error(f"Packing script not found: {script}")
    mount_root = (args.mount_root or Path(os.environ.get("GCS_MOUNT_ROOT", "/mnt/models"))).expanduser().resolve()
    if not mount_root.is_dir():
        parser.error(f"GCS mount directory not found: {mount_root}")
    bucket = storage.Client().bucket(bucket_name)
    original_argv, original_path = sys.argv, sys.path[:]
    try:
        sys.argv = [str(script), *args.script_args]
        sys.path.insert(0, str(script.parent))
        with parallel_pack_writes(
            bucket, mount_root, patterns=args.pattern or ("pack.bin",),
            workers=args.workers, chunk_size=args.chunk_mib << 20,
        ):
            runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv, sys.path[:] = original_argv, original_path


if __name__ == "__main__":
    main()
