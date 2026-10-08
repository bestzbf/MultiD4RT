#!/usr/bin/env python3
"""External janitor for SampleLocalStager raw-file cache."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.sample_stage import SampleLocalStager, SampleStageConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SampleLocalStager cache eviction outside training builders."
    )
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--cache-max-gb", type=float, required=True)
    parser.add_argument("--low-watermark", type=float, default=0.95)
    parser.add_argument("--scan-interval-s", type=float, default=60.0)
    parser.add_argument("--sleep-s", type=float, default=5.0)
    parser.add_argument("--work-stale-min", type=float, default=30.0)
    parser.add_argument("--work-clean-interval-s", type=float, default=300.0)
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--pinned-manifest-root", default="")
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--force-low-watermark",
        action="store_true",
        help=(
            "Trim to the low watermark even if the cache is still below the "
            "hard cap. Intended for startup pre-cleaning, not the continuous "
            "janitor loop."
        ),
    )
    return parser.parse_args()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _cleanup_stale_work_dirs(
    stage_root: Path,
    stale_min: float,
    *,
    emit_log: bool = True,
) -> tuple[int, float]:
    if stale_min <= 0:
        return 0, 0.0
    work_root = stage_root / "work"
    if not work_root.is_dir():
        return 0, 0.0

    cutoff = time.time() - stale_min * 60.0
    removed = 0
    started = time.perf_counter()
    for path in work_root.iterdir():
        if not path.is_dir() or not path.name.startswith("sample_stage_"):
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime >= cutoff:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed += 1

    elapsed = time.perf_counter() - started
    if emit_log and (removed > 0 or elapsed >= 2.0):
        print(
            f"[SampleStageWorkCleanup] removed={removed} "
            f"stale_min={stale_min:.1f} elapsed={elapsed:.3f}s",
            flush=True,
        )
    return removed, elapsed


def main() -> None:
    args = parse_args()
    max_bytes = int(args.cache_max_gb * 1024**3)
    if max_bytes <= 0:
        raise SystemExit("cache-max-gb must be positive")

    stager = SampleLocalStager(
        SampleStageConfig(
            backend="cos_sdk",
            stage_root=args.stage_root,
            sdk_workers=1,
            cache_max_bytes=max_bytes,
            cache_low_watermark_ratio=args.low_watermark,
            cache_scan_interval_s=args.scan_interval_s,
            eviction_mode="disabled",
            pinned_manifest_root=args.pinned_manifest_root,
        )
    )

    print(
        "[SampleStageJanitor] start "
        f"stage_root={args.stage_root} "
        f"max_bytes={max_bytes} "
        f"low_watermark={args.low_watermark:.3f} "
        f"scan_interval={args.scan_interval_s:.1f}s "
        f"sleep={args.sleep_s:.1f}s "
        f"work_stale_min={args.work_stale_min:.1f} "
        f"work_clean_interval={args.work_clean_interval_s:.1f}s "
        f"parent_pid={args.parent_pid} "
        f"pinned_manifest_root={args.pinned_manifest_root or '<none>'}",
        flush=True,
    )

    last_work_cleanup_s = 0.0
    cleanup_log = os.getenv(
        "D4RT_SAMPLE_STAGE_JANITOR_CLEANUP_LOG", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    while True:
        if not _pid_alive(args.parent_pid):
            print(
                f"[SampleStageJanitor] parent exited parent_pid={args.parent_pid}",
                flush=True,
            )
            return
        try:
            stager.evict_cache_once(
                emit_log=True,
                force_low_watermark=args.force_low_watermark,
            )
        except Exception as exc:
            print(
                f"[SampleStageJanitorError] {type(exc).__name__}: {exc}",
                flush=True,
            )
            if args.once:
                raise
        now = time.time()
        if (
            args.once
            or now - last_work_cleanup_s >= max(1.0, args.work_clean_interval_s)
        ):
            last_work_cleanup_s = now
            try:
                _cleanup_stale_work_dirs(
                    Path(args.stage_root),
                    args.work_stale_min,
                    emit_log=cleanup_log,
                )
            except Exception as exc:
                print(
                    f"[SampleStageWorkCleanupError] {type(exc).__name__}: {exc}",
                    flush=True,
                )
                if args.once:
                    raise
        if args.once:
            return
        time.sleep(max(1.0, args.sleep_s))


if __name__ == "__main__":
    main()
