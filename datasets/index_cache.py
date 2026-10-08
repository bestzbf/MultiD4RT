"""Pickle-based disk cache for adapter index building."""

from __future__ import annotations

import contextlib
import os
import pickle
import sys
import types
import pathlib
from pathlib import Path
from typing import Callable, TypeVar

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None

T = TypeVar('T')


def _load_pickle_compat(cache_path: Path):
    """Load a pickle with small compatibility shims for older Python outputs.

    Some historical cache files were written under a Python version that
    pickled pathlib classes from ``pathlib._local``.  Python 3.10 cannot
    import that module, so plain ``pickle.load`` fails and would otherwise
    trigger an expensive cold rebuild on remote storage.
    """
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except ModuleNotFoundError as exc:
        if exc.name != "pathlib._local":
            raise

    shim = types.ModuleType("pathlib._local")
    shim.Path = pathlib.Path
    shim.PosixPath = pathlib.PosixPath
    shim.WindowsPath = pathlib.WindowsPath
    previous = sys.modules.get("pathlib._local")
    sys.modules["pathlib._local"] = shim
    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    finally:
        if previous is None:
            sys.modules.pop("pathlib._local", None)
        else:
            sys.modules["pathlib._local"] = previous


@contextlib.contextmanager
def _cache_build_lock(lock_path: Path):
    """Serialize expensive cold builds across DDP ranks/processes."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _is_empty_index(result: object) -> bool:
    """True for a sized-but-empty build result (e.g. ``[]``).

    An empty index is treated as a transient failure rather than a valid
    cache entry: it almost always means the dataset root was unreadable or
    incompletely materialized when the build ran (e.g. a busy NFS mount).
    Persisting it would poison every later run, and the cache key does not
    always include ``root`` so a stale empty entry can shadow a healthy tree.
    """
    try:
        return len(result) == 0  # type: ignore[arg-type]
    except TypeError:
        return False  # non-sized results are always considered valid


def load_or_build(build_fn: Callable[[], T], cache_path: Path) -> T:
    """Load pickled index from cache_path, or call build_fn and save result.

    Safe for concurrent DDP use: cold builds are serialized by a lock file, and
    writes use a temp file + os.replace (atomic). To force rebuild, delete the
    cache file.

    An empty index is never persisted and an empty cached index is treated as a
    miss, so a build that transiently saw zero records cannot poison later runs.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        try:
            cached = _load_pickle_compat(cache_path)
            if not _is_empty_index(cached):
                return cached
            # Empty cache entry — ignore and rebuild (self-heal poison).
            print(
                f"[index_cache] ignoring empty cached index, rebuilding: {cache_path}",
                file=sys.stderr,
            )
        except Exception:
            pass  # Corrupt cache — fall through and rebuild

    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with _cache_build_lock(lock_path):
        if cache_path.exists():
            try:
                cached = _load_pickle_compat(cache_path)
                if not _is_empty_index(cached):
                    return cached
            except Exception:
                pass  # Corrupt cache — rebuild under lock.

        result = build_fn()
        if _is_empty_index(result):
            # Do not cache an empty index: let the caller raise its own
            # "no sequences found" error this run, and allow a healthy rebuild
            # next run instead of returning poison from disk.
            print(
                f"[index_cache] build returned an empty index, not caching: {cache_path}",
                file=sys.stderr,
            )
            return result
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Use a per-process temp file to avoid clobbering partially-written
        # output if the builder crashes.
        tmp = cache_path.with_suffix(f'.tmp{os.getpid()}')
        with open(tmp, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            os.replace(tmp, cache_path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return result
