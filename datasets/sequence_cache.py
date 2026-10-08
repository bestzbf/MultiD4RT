from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None


_READY_MARKER = ".d4rt_cache_ready"


@contextlib.contextmanager
def _cache_lock(lock_path: Path):
    """Serialize materialization of the same cache key across processes."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _safe_rmtree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    if path.exists():
        shutil.rmtree(path)


def _tree_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        root_path = Path(root)
        for name in files:
            file_path = root_path / name
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total


@dataclass(frozen=True)
class MaterializeResult:
    path: Path
    reused: bool
    elapsed_s: float
    size_bytes: int


class DiskSequenceCache:
    """Disk-backed directory materializer with cross-process locking.

    This is a prototype helper for feasibility testing. It materializes a
    directory tree under ``cache_dir/<key>`` using ``rsync`` and publishes it
    atomically via ``os.replace`` after writing a ready marker.
    """

    def __init__(
        self,
        cache_dir: str | os.PathLike[str],
        *,
        rsync_bin: str = "rsync",
    ) -> None:
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.rsync_bin = rsync_bin
        self._locks_dir = self.root / ".locks"
        self._tmp_dir = self.root / ".tmp"
        self._locks_dir.mkdir(parents=True, exist_ok=True)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

    def clear(self, key: str) -> None:
        dst = self.root / key
        lock_path = self._lock_path(key)
        with _cache_lock(lock_path):
            _safe_rmtree(dst)

    def materialize_tree(
        self,
        key: str,
        source_dir: str | os.PathLike[str],
        *,
        force: bool = False,
    ) -> MaterializeResult:
        source = Path(source_dir)
        if not source.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source}")

        dst = self.root / key
        ready = dst / _READY_MARKER
        start = time.perf_counter()

        if not force and dst.is_dir() and ready.exists():
            return MaterializeResult(
                path=dst,
                reused=True,
                elapsed_s=time.perf_counter() - start,
                size_bytes=_tree_size_bytes(dst),
            )

        lock_path = self._lock_path(key)
        with _cache_lock(lock_path):
            if force:
                _safe_rmtree(dst)

            if dst.exists() and not ready.exists():
                _safe_rmtree(dst)

            if dst.is_dir() and ready.exists():
                return MaterializeResult(
                    path=dst,
                    reused=True,
                    elapsed_s=time.perf_counter() - start,
                    size_bytes=_tree_size_bytes(dst),
                )

            digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
            tmp = self._tmp_dir / f"{digest}.{os.getpid()}.{time.time_ns()}"
            _safe_rmtree(tmp)
            tmp.mkdir(parents=True, exist_ok=True)

            try:
                self._rsync_dir(source, tmp)
                (tmp / _READY_MARKER).write_text("ready\n", encoding="utf-8")
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(tmp, dst)
            except Exception:
                _safe_rmtree(tmp)
                raise

            return MaterializeResult(
                path=dst,
                reused=False,
                elapsed_s=time.perf_counter() - start,
                size_bytes=_tree_size_bytes(dst),
            )

    def _lock_path(self, key: str) -> Path:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
        return self._locks_dir / f"{digest}.lock"

    def _rsync_dir(self, source: Path, dest: Path) -> None:
        source_arg = str(source)
        if not source_arg.endswith(os.sep):
            source_arg = source_arg + os.sep

        proc = subprocess.run(
            [self.rsync_bin, "-a", source_arg, str(dest)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            raise RuntimeError(
                f"rsync failed for {source} -> {dest} "
                f"(exit={proc.returncode}): {stderr}"
            )
