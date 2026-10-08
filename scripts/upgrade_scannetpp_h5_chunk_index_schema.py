#!/usr/bin/env python3
"""Upgrade legacy ScanNet++ precomputed.h5 chunk-index pickles.

Older ``precomputed.h5_chunk_index.pkl`` files stored each dataset entry as a
plain ``[(byte_offset, size), ...]`` list.  The newer schema stores metadata too:
``offsets``, ``dtype``, ``chunk_shape``, ``shape``, and ``compression``, plus
top-level scalar ``num_frames`` / ``num_points`` entries.

This script rewrites only the small chunk-index pickle files.  It does not read
or rewrite the large ``precomputed.h5`` payloads.
"""

from __future__ import annotations

import argparse
import json
import pickle
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from qcloud_cos import CosConfig, CosS3Client


ARRAY_KEYS = (
    "extrinsics",
    "intrinsics",
    "normals",
    "trajs_2d",
    "trajs_3d_world",
    "valids",
    "visibs",
)
TRACK_KEYS = ("trajs_2d", "trajs_3d_world", "valids", "visibs")

_TLS = threading.local()


def _read_cos_credentials(path: Path) -> tuple[str, str]:
    parts = path.read_text().strip().split(":")
    if len(parts) == 2:
        return parts[0], parts[1]
    if len(parts) == 3:
        return parts[1], parts[2]
    raise ValueError(f"Unsupported COS passwd file format: {path}")


def _client(args: argparse.Namespace) -> CosS3Client:
    client = getattr(_TLS, "cos_client", None)
    if client is None:
        secret_id, secret_key = _read_cos_credentials(Path(args.passwd_file))
        client = CosS3Client(
            CosConfig(
                Region=args.region,
                SecretId=secret_id,
                SecretKey=secret_key,
                Scheme="https",
                Timeout=args.timeout,
            )
        )
        _TLS.cos_client = client
    return client


def _load_scenes(args: argparse.Namespace) -> list[str]:
    scenes: list[str] = []
    if args.scenes:
        scenes.extend(args.scenes)
    if args.scenes_file:
        for path in args.scenes_file:
            scenes.extend(
                line.strip()
                for line in Path(path).read_text().splitlines()
                if line.strip()
            )
    if args.scenes_record:
        for raw in Path(args.scenes_record).read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            if item.get("status", "ok") == "ok" and item.get("scene_name"):
                scenes.append(str(item["scene_name"]))
    if not scenes and args.splits_dir:
        for split_file in sorted(Path(args.splits_dir).glob("*.txt")):
            scenes.extend(
                line.strip()
                for line in split_file.read_text().splitlines()
                if line.strip()
            )

    deduped: list[str] = []
    seen: set[str] = set()
    for scene in scenes:
        if scene not in seen:
            seen.add(scene)
            deduped.append(scene)
    return deduped


def _infer_legacy_entry(key: str, offsets: Any) -> dict[str, Any]:
    if not isinstance(offsets, list) or not offsets:
        raise TypeError(f"{key}: legacy entry must be a non-empty list")
    clean_offsets: list[tuple[int, int]] = []
    sizes: set[int] = set()
    for item in offsets:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError(f"{key}: bad offset entry {item!r}")
        offset, size = int(item[0]), int(item[1])
        clean_offsets.append((offset, size))
        sizes.add(size)
    if len(sizes) != 1:
        raise ValueError(f"{key}: varying chunk sizes suggest compressed chunks: {sorted(sizes)[:5]}")
    chunk_size = next(iter(sizes))

    if key == "trajs_2d":
        dtype = "float32"
        num_points = chunk_size // (np.dtype(dtype).itemsize * 2)
        chunk_shape = (1, num_points, 2)
    elif key == "trajs_3d_world":
        dtype = "float32"
        num_points = chunk_size // (np.dtype(dtype).itemsize * 3)
        chunk_shape = (1, num_points, 3)
    elif key in {"valids", "visibs"}:
        dtype = "bool"
        num_points = chunk_size // np.dtype(dtype).itemsize
        chunk_shape = (1, num_points)
    elif key == "intrinsics":
        dtype = "float32"
        chunk_shape = (1, 3, 3)
    elif key == "extrinsics":
        dtype = "float32"
        chunk_shape = (1, 4, 4)
    elif key == "normals":
        dtype = "float16"
        chunk_shape = (1, 192, 256, 3)
    else:
        raise KeyError(f"Unsupported legacy key: {key}")

    expected_size = int(np.prod(chunk_shape)) * np.dtype(dtype).itemsize
    if expected_size != chunk_size:
        raise ValueError(
            f"{key}: chunk size mismatch, got {chunk_size}, inferred {expected_size}"
        )

    return {
        "offsets": clean_offsets,
        "dtype": dtype,
        "chunk_shape": chunk_shape,
        "shape": (len(clean_offsets),) + tuple(chunk_shape[1:]),
        "compression": None,
    }


def _entry_num_points(entry: dict[str, Any], key: str) -> int | None:
    shape = tuple(entry["shape"])
    if key in {"trajs_2d", "trajs_3d_world", "valids", "visibs"} and len(shape) >= 2:
        return int(shape[1])
    return None


def _upgrade_index(obj: Any) -> tuple[dict[str, Any], bool, str]:
    if not isinstance(obj, dict):
        raise TypeError(f"top-level pickle is {type(obj).__name__}, expected dict")

    upgraded: dict[str, Any] = {}
    changed = False
    legacy_entries = 0
    frame_counts: set[int] = set()
    point_counts: set[int] = set()

    for key in ARRAY_KEYS:
        if key not in obj:
            raise KeyError(f"missing key: {key}")
        entry = obj[key]
        if isinstance(entry, dict):
            new_entry = entry
        else:
            new_entry = _infer_legacy_entry(key, entry)
            changed = True
            legacy_entries += 1
        upgraded[key] = new_entry
        frame_counts.add(len(new_entry["offsets"]))
        n_points = _entry_num_points(new_entry, key)
        if n_points is not None:
            point_counts.add(n_points)

    if len(frame_counts) != 1:
        raise ValueError(f"frame-count mismatch across keys: {sorted(frame_counts)}")
    if len(point_counts) != 1:
        raise ValueError(f"point-count mismatch across track keys: {sorted(point_counts)}")

    for key, value in obj.items():
        if key not in upgraded and key not in {"num_frames", "num_points"}:
            upgraded[key] = value

    num_frames = next(iter(frame_counts))
    num_points = next(iter(point_counts))
    expected_frames = {"scalar": True, "value": int(num_frames)}
    expected_points = {"scalar": True, "value": int(num_points)}
    if obj.get("num_frames") != expected_frames:
        upgraded["num_frames"] = expected_frames
        changed = True
    else:
        upgraded["num_frames"] = obj["num_frames"]
    if obj.get("num_points") != expected_points:
        upgraded["num_points"] = expected_points
        changed = True
    else:
        upgraded["num_points"] = obj["num_points"]

    schema = "legacy_offsets_only" if legacy_entries else "new_with_metadata"
    return upgraded, changed, schema


def _object_key(args: argparse.Namespace, scene: str) -> str:
    return f"{args.data_prefix.rstrip('/')}/{scene}/precomputed.h5_chunk_index.pkl"


def _process_scene(args: argparse.Namespace, scene: str) -> dict[str, Any]:
    key = _object_key(args, scene)
    response = _client(args).get_object(Bucket=args.bucket, Key=key)
    raw = response["Body"].get_raw_stream().read()
    obj = pickle.loads(raw)
    upgraded, changed, schema = _upgrade_index(obj)
    result = {
        "scene": scene,
        "schema": schema,
        "changed": changed,
        "status": "dry_run" if not args.apply else "skipped",
    }
    if not changed:
        return result

    backup_dir = Path(args.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    (backup_dir / f"{scene}.precomputed.h5_chunk_index.pkl").write_bytes(raw)
    result["backup"] = str(backup_dir / f"{scene}.precomputed.h5_chunk_index.pkl")

    if args.apply:
        payload = pickle.dumps(upgraded, protocol=pickle.HIGHEST_PROTOCOL)
        _client(args).put_object(Bucket=args.bucket, Key=key, Body=payload)
        result["status"] = "updated"
        result["bytes_before"] = len(raw)
        result["bytes_after"] = len(payload)
    return result


def main() -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default="hd-ai-data-1251882982")
    parser.add_argument("--region", default="ap-beijing")
    parser.add_argument("--passwd-file", default="/etc/passwd-cosfs")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--data-prefix", default="hdu_datasets/scannetpp/data")
    parser.add_argument("--scenes-record", default="/data_cos/hdu_datasets/scannetpp/scenes_record.json")
    parser.add_argument("--splits-dir", default="/data_cos/hdu_datasets/scannetpp/splits")
    parser.add_argument("--scenes-file", action="append")
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--backup-dir",
        default=f"/tmp/d4rt_scannetpp_h5_chunk_index_backup_{timestamp}",
    )
    args = parser.parse_args()

    if not Path(args.passwd_file).exists() and Path("/etc/passwd-s3fs-data_cos").exists():
        args.passwd_file = "/etc/passwd-s3fs-data_cos"

    scenes = _load_scenes(args)
    if args.limit > 0:
        scenes = scenes[: args.limit]
    if not scenes:
        raise SystemExit("No scenes selected")

    print(f"scenes={len(scenes)} apply={args.apply} backup_dir={args.backup_dir}", flush=True)
    counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    report_path = Path(args.backup_dir) / "upgrade_report.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor, report_path.open("w") as report:
        future_to_scene = {executor.submit(_process_scene, args, scene): scene for scene in scenes}
        for idx, future in enumerate(as_completed(future_to_scene), 1):
            scene = future_to_scene[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "scene": scene,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(result)
            counts[result["status"]] += 1
            counts[f"schema:{result.get('schema', 'unknown')}"] += 1
            report.write(json.dumps(result, sort_keys=True) + "\n")
            if idx % 100 == 0 or idx == len(scenes):
                print(f"progress {idx}/{len(scenes)} {dict(counts)}", flush=True)

    print(f"done counts={dict(counts)} failures={len(failures)} report={report_path}", flush=True)
    if failures:
        print("first_failures:")
        for failure in failures[:20]:
            print(json.dumps(failure, sort_keys=True), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
