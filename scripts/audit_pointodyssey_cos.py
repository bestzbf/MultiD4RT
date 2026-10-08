#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit PointOdyssey files on Tencent COS using coscli and optionally "
            "repair only the scenes with missing files."
        )
    )
    parser.add_argument(
        "--local-root",
        default="/data2/d4rt/datasets/PointOdyssey",
        help="Local PointOdyssey root.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to audit.",
    )
    parser.add_argument(
        "--bucket",
        default="hd-ai-data-1251882982",
        help="COS bucket name (bucket-appid).",
    )
    parser.add_argument(
        "--endpoint",
        default="cos.ap-beijing.myqcloud.com",
        help="COS endpoint.",
    )
    parser.add_argument(
        "--remote-prefix",
        default="hdu_datasets/PointOdyssey",
        help="Remote COS prefix for the dataset root.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Specific scene name to audit. Can be passed multiple times.",
    )
    parser.add_argument(
        "--report-json",
        default="pointodyssey_cos_audit.json",
        help="Path to write JSON audit report.",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Run coscli sync for scenes that have missing files.",
    )
    parser.add_argument(
        "--thread-num",
        type=int,
        default=32,
        help="coscli sync thread count when --repair is enabled.",
    )
    parser.add_argument(
        "--part-size",
        type=int,
        default=64,
        help="coscli sync part size in MB when --repair is enabled.",
    )
    parser.add_argument(
        "--rate-limiting",
        type=float,
        default=0.0,
        help="Optional coscli sync rate limit in MB/s. 0 means unlimited.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit only; print repair commands without executing them.",
    )
    return parser.parse_args()


def parse_coscli_ls_table(stdout: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip("\n")
        if "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 2:
            continue
        key = parts[0]
        obj_type = parts[1]
        if not key or key == "KEY" or "TOTAL OBJECTS" in key:
            continue
        if set(key) <= {"-", "+"}:
            continue
        entries.append((key, obj_type))
    return entries


def run_coscli_ls(endpoint: str, cos_path: str, recursive: bool) -> list[tuple[str, str]]:
    cmd = ["coscli", "-e", endpoint, "ls", cos_path]
    if recursive:
        cmd.append("-r")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"coscli ls failed for {cos_path}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return parse_coscli_ls_table(result.stdout)


def list_local_files(scene_dir: Path) -> set[str]:
    rel_files: set[str] = set()
    for path in scene_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(scene_dir).as_posix()
        parts = rel.split("/")
        if any(part.startswith(".") for part in parts):
            continue
        rel_files.add(rel)
    return rel_files


def list_remote_files(endpoint: str, bucket: str, remote_scene_prefix: str) -> set[str]:
    cos_path = f"cos://{bucket}/{remote_scene_prefix}/"
    entries = run_coscli_ls(endpoint=endpoint, cos_path=cos_path, recursive=True)
    base = f"{remote_scene_prefix.rstrip('/')}/"
    rel_files: set[str] = set()
    for key, obj_type in entries:
        if obj_type == "DIR":
            continue
        rel = key[len(base):] if key.startswith(base) else key
        rel = rel.lstrip("/")
        if rel:
            rel_files.add(rel)
    return rel_files


def summarize_missing(missing_files: set[str]) -> dict[str, int]:
    counter = Counter()
    for rel_path in missing_files:
        head = rel_path.split("/", 1)[0]
        counter[head] += 1
    return dict(sorted(counter.items()))


def repair_scene(
    endpoint: str,
    bucket: str,
    remote_scene_prefix: str,
    local_scene_dir: Path,
    thread_num: int,
    part_size: int,
    rate_limiting: float,
    dry_run: bool,
) -> int:
    cmd = [
        "coscli",
        "-e",
        endpoint,
        "sync",
        str(local_scene_dir),
        f"cos://{bucket}/{remote_scene_prefix}/",
        "-r",
        "--thread-num",
        str(thread_num),
        "--part-size",
        str(part_size),
    ]
    if rate_limiting > 0:
        cmd.extend(["--rate-limiting", str(rate_limiting)])

    print("REPAIR", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd).returncode


def main() -> int:
    args = parse_args()
    local_split_root = Path(args.local_root) / args.split
    if not local_split_root.is_dir():
        raise SystemExit(f"Local split root not found: {local_split_root}")

    scene_names = sorted(args.scene) if args.scene else sorted(
        path.name for path in local_split_root.iterdir() if path.is_dir()
    )
    if not scene_names:
        raise SystemExit(f"No scenes found under {local_split_root}")

    report: list[dict[str, object]] = []
    bad_scenes: list[str] = []

    for idx, scene_name in enumerate(scene_names, start=1):
        local_scene_dir = local_split_root / scene_name
        remote_scene_prefix = f"{args.remote_prefix.rstrip('/')}/{args.split}/{scene_name}"
        print(f"[{idx}/{len(scene_names)}] audit {scene_name}", flush=True)

        local_files = list_local_files(local_scene_dir)
        remote_files = list_remote_files(
            endpoint=args.endpoint,
            bucket=args.bucket,
            remote_scene_prefix=remote_scene_prefix,
        )

        missing_remote = sorted(local_files - remote_files)
        extra_remote = sorted(remote_files - local_files)
        item = {
            "scene": scene_name,
            "local_file_count": len(local_files),
            "remote_file_count": len(remote_files),
            "missing_remote_count": len(missing_remote),
            "extra_remote_count": len(extra_remote),
            "missing_remote_summary": summarize_missing(set(missing_remote)),
            "extra_remote_summary": summarize_missing(set(extra_remote)),
            "missing_remote_examples": missing_remote[:50],
            "extra_remote_examples": extra_remote[:50],
        }
        report.append(item)

        if missing_remote:
            bad_scenes.append(scene_name)
            print(
                f"  missing={len(missing_remote)} extra={len(extra_remote)} "
                f"summary={item['missing_remote_summary']}",
                flush=True,
            )
        else:
            print(f"  ok files={len(local_files)}", flush=True)

    report_path = Path(args.report_json)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"report written to {report_path}", flush=True)

    if bad_scenes:
        print(f"scenes with missing remote files: {len(bad_scenes)}", flush=True)
        for scene_name in bad_scenes:
            print(f"  - {scene_name}", flush=True)
    else:
        print("no missing remote files detected", flush=True)

    if args.repair and bad_scenes:
        failures = 0
        for scene_name in bad_scenes:
            local_scene_dir = local_split_root / scene_name
            remote_scene_prefix = f"{args.remote_prefix.rstrip('/')}/{args.split}/{scene_name}"
            ret = repair_scene(
                endpoint=args.endpoint,
                bucket=args.bucket,
                remote_scene_prefix=remote_scene_prefix,
                local_scene_dir=local_scene_dir,
                thread_num=args.thread_num,
                part_size=args.part_size,
                rate_limiting=args.rate_limiting,
                dry_run=args.dry_run,
            )
            if ret != 0:
                failures += 1
        if failures:
            print(f"repair finished with {failures} failed scene syncs", flush=True)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
