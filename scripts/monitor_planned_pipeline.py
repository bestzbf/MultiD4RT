#!/usr/bin/env python3
"""Low-impact planned-mode pipeline monitor.

This script only stats local spool/stage files. It does not load samples and
does not touch COS, so it is safe to run next to an active training job.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import re


def _fmt_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{n}B"


def _summarize_spool(root: Path, ranks: set[int] | None = None) -> list[str]:
    lines: list[str] = []
    for spool_dir in sorted(root.glob("d4rt_spool_rank*")):
        if not spool_dir.is_dir():
            continue
        match = re.fullmatch(r"d4rt_spool_rank(\d+)", spool_dir.name)
        if ranks is not None:
            if match is None or int(match.group(1)) not in ranks:
                continue
        ready = list(spool_dir.glob("*.ready"))
        building = list(spool_dir.glob("*.building"))
        error = list(spool_dir.glob("*.error"))
        sizes: list[int] = []
        for path in ready:
            try:
                sizes.append(path.stat().st_size)
            except OSError:
                pass
        total = sum(sizes)
        avg = int(total / len(sizes)) if sizes else 0
        max_size = max(sizes) if sizes else 0
        lines.append(
            f"{spool_dir.name}: ready={len(ready)} building={len(building)} "
            f"error={len(error)} total={_fmt_bytes(total)} "
            f"avg={_fmt_bytes(avg)} max={_fmt_bytes(max_size)}"
        )
    return lines


def _summarize_stage(stage_root: Path, scan_parts: bool = False) -> list[str]:
    work_root = stage_root / "work"
    cache_root = stage_root / "shared_raw_cache" / "data"
    work_dirs = 0
    if work_root.is_dir():
        work_dirs = sum(1 for p in work_root.iterdir() if p.is_dir())

    part_files: int | str = "skipped"
    if scan_parts and cache_root.is_dir():
        part_files = sum(1 for _ in cache_root.rglob("*.part.*"))

    return [
        f"stage: work_dirs={work_dirs} part_files={part_files}",
    ]


def _classify(spool_lines: list[str], min_ready: int) -> str:
    if not spool_lines:
        return "classification: no spool dirs found"
    ready_counts: list[int] = []
    avg_sizes: list[float] = []
    for line in spool_lines:
        fields = dict(
            token.split("=", 1)
            for token in line.replace(":", "").split()
            if "=" in token
        )
        try:
            ready_counts.append(int(fields.get("ready", "0")))
        except ValueError:
            pass
        avg = fields.get("avg", "0B")
        if avg.endswith("MB"):
            avg_sizes.append(float(avg[:-2]))
        elif avg.endswith("GB"):
            avg_sizes.append(float(avg[:-2]) * 1024.0)

    min_count = min(ready_counts) if ready_counts else None
    max_avg = max(avg_sizes) if avg_sizes else 0.0
    if min_count == 0:
        return "classification: builder/COS path is not keeping spool filled"
    if min_count is not None and min_count < min_ready:
        if max_avg > 100.0:
            return (
                "classification: low spool cushion and large samples; "
                "expect read/unpickle/collate stalls"
            )
        return "classification: low spool cushion; builders are barely ahead"
    if max_avg > 100.0:
        return "classification: spool is filled; data time is likely pickle/read/collate of huge samples"
    return "classification: spool has headroom; inspect train loop or GPU transfer next"


def _parse_ranks(raw: str) -> set[int] | None:
    raw = raw.strip()
    if not raw:
        return None
    ranks: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        ranks.add(int(item))
    return ranks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmpdir", default="/data1/zbf/d4rt_tmp")
    parser.add_argument("--stage-root", default="/data1/zbf/d4rt_sample_stage")
    parser.add_argument("--interval", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--ranks", default="", help="Comma-separated rank ids to include, e.g. 0,1,2.")
    parser.add_argument("--min-ready", type=int, default=16)
    parser.add_argument(
        "--scan-stage-parts",
        action="store_true",
        help="Recursively scan stage cache for partial downloads. This can be slow on large caches.",
    )
    args = parser.parse_args()
    ranks = _parse_ranks(args.ranks)

    for i in range(max(1, args.samples)):
        if i:
            time.sleep(max(0.1, args.interval))
        print(time.strftime("[%H:%M:%S]"))
        spool_lines = _summarize_spool(Path(args.tmpdir), ranks=ranks)
        for line in spool_lines:
            print("  " + line)
        for line in _summarize_stage(Path(args.stage_root), scan_parts=args.scan_stage_parts):
            print("  " + line)
        print("  " + _classify(spool_lines, min_ready=args.min_ready))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
