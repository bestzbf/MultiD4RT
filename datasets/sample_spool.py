"""
Sample spool for managing local QuerySample bundle storage.

This module handles the lifecycle of pre-built QuerySample bundles:
- Writing bundles atomically (.building -> .ready)
- Waiting for bundles to become ready
- Cleaning up consumed bundles
- Error marker handling
- Generation-based isolation to prevent epoch cross-pollution
- Disk watermark control to bound spool size
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path
from typing import Optional

from .query_builder import QuerySample

logger = logging.getLogger(__name__)


class SampleSpool:
    """Manages local storage of pre-built QuerySample bundles.

    Each bundle is stored as a pickle file with atomic rename:
    - g{generation:04d}_{index:08d}.building (temp file during construction)
    - g{generation:04d}_{index:08d}.ready   (final file, ready for consumption)
    - g{generation:04d}_{index:08d}.error   (error marker with exception info)

    Bundles are deleted immediately after loading to save disk space.
    A generation counter isolates epochs so stale builders never pollute
    the current epoch's data.  A disk watermark (max_spool_bytes) limits growth
    when prefetch_depth is large.  The limit is a soft cap: concurrent builders
    may each pass the check before any of them writes, so actual usage can
    overshoot by up to ~sample_size × builder_workers.  This is by design —
    adding a lock would serialize writes with no practical benefit since the
    overshoot is small relative to the limit.
    """

    def __init__(
        self,
        spool_dir: str | Path,
        rank: int = 0,
        cleanup_on_init: bool = True,
        max_spool_bytes: int = 2 * 1024**3,
    ):
        """
        Args:
            spool_dir: Directory for spool files (e.g., /tmp/d4rt_spool_rank0/)
            rank: Rank ID (for multi-GPU training)
            cleanup_on_init: Whether to clean up existing spool files on init
            max_spool_bytes: Maximum total spool size in bytes (default 2 GB)
        """
        self.spool_dir = Path(spool_dir)
        self.rank = rank
        self.max_spool_bytes = max_spool_bytes
        self._generation: int = 0
        self._profile = os.getenv("D4RT_PROFILE_SPOOL", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        self._profile_interval = max(
            1, int(os.getenv("D4RT_PROFILE_SPOOL_INTERVAL", "20"))
        )
        self._profile_wait_threshold_s = float(
            os.getenv("D4RT_PROFILE_SPOOL_WAIT_THRESHOLD_S", "2.0")
        )
        self._profile_count = 0
        self.spool_dir.mkdir(parents=True, exist_ok=True)

        if cleanup_on_init:
            self._cleanup_all()

    # ------------------------------------------------------------------
    # Generation management
    # ------------------------------------------------------------------

    def set_generation(self, gen: int) -> None:
        """Update generation counter and purge all files from older generations.

        Args:
            gen: New generation number
        """
        old_gen = self._generation
        self._generation = gen
        for p in self.spool_dir.iterdir():
            if not p.is_file():
                continue
            file_gen = self._parse_generation(p.name)
            if file_gen is not None and file_gen != gen:
                try:
                    p.unlink()
                except Exception:
                    pass
        logger.debug(
            "Spool generation %d -> %d, purged old files", old_gen, gen
        )

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _get_building_path(self, index: int, generation: int) -> Path:
        return self.spool_dir / f"g{generation:04d}_{index:08d}.building"

    def _get_ready_path(self, index: int, generation: int) -> Path:
        return self.spool_dir / f"g{generation:04d}_{index:08d}.ready"

    def _get_error_path(self, index: int, generation: int) -> Path:
        return self.spool_dir / f"g{generation:04d}_{index:08d}.error"

    @staticmethod
    def _parse_generation(filename: str) -> Optional[int]:
        """Extract generation number from a spool filename, or None."""
        if not filename.startswith("g"):
            return None
        try:
            return int(filename[1:5])
        except (ValueError, IndexError):
            return None

    # ------------------------------------------------------------------
    # Disk watermark
    # ------------------------------------------------------------------

    def get_spool_size(self) -> int:
        """Get total size of current-generation spool files in bytes."""
        total = 0
        gen_prefix = f"g{self._generation:04d}_"
        for p in self.spool_dir.iterdir():
            if p.is_file() and p.name.startswith(gen_prefix):
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total

    def wait_for_space(
        self, timeout: float = 600.0, poll_interval: float = 0.5
    ) -> None:
        """Block until spool size drops below max_spool_bytes.

        Note: this is a soft check — concurrent callers may all pass before any
        writes, so transient overshoot up to ~sample_size × num_callers is
        expected.

        The consumer side (wait_for_bundle) deletes files after loading,
        so space is freed naturally as training progresses.

        Args:
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds

        Raises:
            TimeoutError: If space is not freed within timeout
        """
        start = time.time()
        while True:
            if self.get_spool_size() < self.max_spool_bytes:
                return
            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(
                    f"Spool size did not drop below "
                    f"{self.max_spool_bytes / 1024**2:.0f} MB within {timeout}s. "
                    f"Current size: {self.get_spool_size() / 1024**2:.0f} MB"
                )
            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def write_bundle(
        self, index: int, sample: QuerySample, generation: int
    ) -> None:
        """Write a QuerySample bundle atomically.

        Blocks if spool is over the disk watermark before writing.

        Args:
            index: Sample index
            sample: QuerySample to write
            generation: Generation number for this bundle
        """
        self.wait_for_space()

        building_path = self._get_building_path(index, generation)
        ready_path = self._get_ready_path(index, generation)

        data = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)
        for _attempt in range(2):
            with open(building_path, "wb") as f:
                f.write(data)
            try:
                os.replace(building_path, ready_path)
                return
            except FileNotFoundError:
                # Rare race: file vanished between write and rename (e.g. NFS).
                # Re-write and retry once.
                if _attempt == 0:
                    continue
                raise

    def write_error(
        self, index: int, exception: Exception, generation: int
    ) -> None:
        """Write an error marker for a failed sample.

        Args:
            index: Sample index
            exception: Exception that occurred
            generation: Generation number for this bundle
        """
        error_path = self._get_error_path(index, generation)
        error_info = {
            "index": index,
            "generation": generation,
            "exception_type": type(exception).__name__,
            "exception_str": str(exception),
        }
        with open(error_path, "wb") as f:
            pickle.dump(error_info, f)

    # ------------------------------------------------------------------
    # Read / wait operations
    # ------------------------------------------------------------------

    def wait_for_bundle(
        self,
        index: int,
        generation: int,
        timeout: float = 300.0,
        poll_interval: float = 0.1,
    ) -> QuerySample:
        """Wait for a bundle to become ready and load it.

        Only considers files matching the specified generation.

        Args:
            index: Sample index to wait for
            generation: Generation number to look for
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds

        Returns:
            QuerySample loaded from disk

        Raises:
            TimeoutError: If bundle doesn't become ready within timeout
            RuntimeError: If bundle has an error marker
        """
        ready_path = self._get_ready_path(index, generation)
        error_path = self._get_error_path(index, generation)

        start_time = time.time()
        while True:
            if error_path.exists():
                with open(error_path, "rb") as f:
                    error_info = pickle.load(f)
                raise RuntimeError(
                    f"Sample g{generation:04d}_{index} failed: "
                    f"{error_info['exception_type']}: "
                    f"{error_info['exception_str']}"
                )

            if ready_path.exists():
                wait_s = time.time() - start_time
                if self._profile:
                    size_bytes = ready_path.stat().st_size
                    t_read0 = time.perf_counter()
                    with open(ready_path, "rb") as f:
                        payload = f.read()
                    t_read = time.perf_counter() - t_read0
                    t_pickle0 = time.perf_counter()
                    sample = pickle.loads(payload)
                    t_pickle = time.perf_counter() - t_pickle0
                    self._profile_count += 1
                    if (
                        self._profile_count <= 3
                        or self._profile_count % self._profile_interval == 0
                        or wait_s >= self._profile_wait_threshold_s
                    ):
                        print(
                            f"[SpoolProfile rank{self.rank}] "
                            f"sample={self._profile_count} "
                            f"g{generation:04d}_{index:08d} "
                            f"size={size_bytes / 1024**2:.1f}MB "
                            f"wait={wait_s * 1000:.1f}ms "
                            f"read={t_read * 1000:.1f}ms "
                            f"unpickle={t_pickle * 1000:.1f}ms",
                            flush=True,
                        )
                    del payload
                else:
                    with open(ready_path, "rb") as f:
                        sample = pickle.load(f)
                try:
                    ready_path.unlink()
                except Exception:
                    pass
                return sample

            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(
                    f"Sample g{generation:04d}_{index} did not become ready "
                    f"within {timeout}s. Spool dir: {self.spool_dir}"
                )

            time.sleep(poll_interval)

    def is_ready(self, index: int, generation: int) -> bool:
        """Check if a bundle is ready (non-blocking).

        Args:
            index: Sample index
            generation: Generation number

        Returns:
            True if bundle is ready
        """
        return self._get_ready_path(index, generation).exists()

    def list_ready_indices(self, generation: int) -> list[int]:
        """Return sorted ready bundle indices for a generation."""
        prefix = f"g{generation:04d}_"
        indices: list[int] = []
        for path in self.spool_dir.iterdir():
            name = path.name
            if not (name.startswith(prefix) and name.endswith(".ready")):
                continue
            try:
                indices.append(int(name[len(prefix):len(prefix) + 8]))
            except ValueError:
                continue
        indices.sort()
        return indices

    def has_error(self, index: int, generation: int) -> bool:
        """Check if a bundle has an error marker.

        Args:
            index: Sample index
            generation: Generation number

        Returns:
            True if bundle has error
        """
        return self._get_error_path(index, generation).exists()

    def list_error_indices(self, generation: int) -> list[int]:
        """Return sorted error marker indices for a generation."""
        prefix = f"g{generation:04d}_"
        indices: list[int] = []
        for path in self.spool_dir.iterdir():
            name = path.name
            if not (name.startswith(prefix) and name.endswith(".error")):
                continue
            try:
                indices.append(int(name[len(prefix):len(prefix) + 8]))
            except ValueError:
                continue
        indices.sort()
        return indices

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def get_ready_count(self) -> int:
        """Get number of ready bundles in the current generation."""
        gen_prefix = f"g{self._generation:04d}_"
        count = 0
        for p in self.spool_dir.iterdir():
            if (
                p.is_file()
                and p.name.startswith(gen_prefix)
                and p.name.endswith(".ready")
            ):
                count += 1
        return count

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Clean up all spool files across all generations."""
        self._cleanup_all()

    def cleanup_generation(self, gen: int) -> None:
        """Clean up all spool files for a specific generation.

        Args:
            gen: Generation number to purge
        """
        gen_prefix = f"g{gen:04d}_"
        for p in self.spool_dir.iterdir():
            if p.is_file() and p.name.startswith(gen_prefix):
                try:
                    p.unlink()
                except Exception:
                    pass

    def _cleanup_all(self) -> None:
        """Remove all existing spool files regardless of generation."""
        for pattern in ("*.building", "*.ready", "*.error"):
            for p in self.spool_dir.glob(pattern):
                try:
                    p.unlink()
                except Exception:
                    pass
