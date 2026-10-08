#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from qcloud_cos import CosConfig, CosS3Client


REQUIRED_FILES = [
    "precomputed.h5",
    "iphone/depth.bin",
    "iphone/depth_chunk_index.pkl",
    "iphone/colmap/images.txt",
    "iphone/colmap/cameras.txt",
    "iphone/pose_intrinsic_imu.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit uploaded ScanNet++ data on Tencent COS.")
    parser.add_argument("--local-root", default="/data3/dataset/scannetpp/data")
    parser.add_argument("--splits-dir", default="/data3/dataset/scannetpp/splits")
    parser.add_argument("--scenes-record", default="/data_cos/hdu_datasets/scannetpp/scenes_record.json")
    parser.add_argument("--bucket", default="hd-ai-data-1251882982")
    parser.add_argument("--region", default="ap-beijing")
    parser.add_argument("--remote-prefix", default="hdu_datasets/scannetpp/data")
    parser.add_argument("--passwd-file", default="/etc/passwd-s3fs-data_cos")
    parser.add_argument("--report-json", default="tmp/test_results_scannetpp_cos_upload/audit_scannetpp_cos.json")
    parser.add_argument("--report-md", default="tmp/test_results_scannetpp_cos_upload/SCANNETPP_COS_AUDIT.md")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit-scenes", type=int, default=0, help="Audit only the first N selected scenes; 0 means all.")
    parser.add_argument("--scene", action="append", default=[], help="Specific scene to audit; can be passed multiple times.")
    parser.add_argument("--range-check-scene", action="append", default=[], help="Scene whose large files should be range-compared.")
    parser.add_argument("--range-bytes", type=int, default=1024 * 1024)
    return parser.parse_args()


def read_split_names(splits_dir: Path) -> list[str]:
    names: list[str] = []
    for split in ["nvs_sem_train.txt", "nvs_sem_val.txt"]:
        path = splits_dir / split
        if path.exists():
            names.extend(line.strip() for line in path.read_text().splitlines() if line.strip())
    return names


def read_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        scene = item.get("scene_name")
        if scene:
            records[str(scene)] = item
    return records


def make_client(args: argparse.Namespace) -> CosS3Client:
    secret_id, secret_key = Path(args.passwd_file).read_text().strip().split(":", 1)
    return CosS3Client(CosConfig(Region=args.region, SecretId=secret_id, SecretKey=secret_key, Scheme="https"))


def head_object(client: CosS3Client, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except Exception:
        return None


def object_size(head: dict[str, Any] | None) -> int | None:
    if not head:
        return None
    value = head.get("Content-Length") or head.get("content-length")
    return int(value) if value is not None else None


def list_frame_count(client: CosS3Client, bucket: str, prefix: str) -> tuple[int, int]:
    marker = ""
    count = 0
    total_size = 0
    while True:
        resp = client.list_objects(Bucket=bucket, Prefix=prefix, Marker=marker, MaxKeys=1000)
        contents = resp.get("Contents") or []
        if isinstance(contents, dict):
            contents = [contents]
        for item in contents:
            key = item.get("Key", "")
            if key.endswith(".jpg"):
                count += 1
                total_size += int(item.get("Size", 0))
        if resp.get("IsTruncated") == "true":
            marker = resp.get("NextMarker") or (contents[-1].get("Key") if contents else "")
            if not marker:
                break
        else:
            break
    return count, total_size


def cos_range_bytes(client: CosS3Client, bucket: str, key: str, start: int, end: int) -> bytes:
    resp = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")
    body = resp["Body"]
    try:
        return body.get_raw_stream().read()
    finally:
        close = getattr(body, "close", None)
        if close is not None:
            close()


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def range_compare_file(
    client: CosS3Client,
    bucket: str,
    key: str,
    local_path: Path,
    remote_size: int,
    chunk_size: int,
) -> dict[str, Any]:
    local_size = local_path.stat().st_size
    result: dict[str, Any] = {
        "key": key,
        "local_size": local_size,
        "remote_size": remote_size,
        "ok": local_size == remote_size,
        "chunks": [],
    }
    if local_size != remote_size or local_size <= 0:
        return result

    spans = [(0, min(chunk_size, local_size) - 1)]
    if local_size > chunk_size:
        spans.append((max(0, local_size - chunk_size), local_size - 1))

    with local_path.open("rb") as f:
        for start, end in spans:
            f.seek(start)
            local_bytes = f.read(end - start + 1)
            remote_bytes = cos_range_bytes(client, bucket, key, start, end)
            local_md5 = md5_bytes(local_bytes)
            remote_md5 = md5_bytes(remote_bytes)
            ok = local_md5 == remote_md5
            result["chunks"].append(
                {
                    "range": [start, end],
                    "bytes": end - start + 1,
                    "local_md5": local_md5,
                    "remote_md5": remote_md5,
                    "ok": ok,
                }
            )
            result["ok"] = bool(result["ok"] and ok)
    return result


def audit_scene(
    args: argparse.Namespace,
    scene: str,
    expected_frames: int | None,
    do_range_check: bool,
) -> dict[str, Any]:
    client = make_client(args)
    local_scene = Path(args.local_root) / scene
    remote_scene = f"{args.remote_prefix.rstrip('/')}/{scene}"
    item: dict[str, Any] = {
        "scene": scene,
        "expected_frames": expected_frames,
        "ok": True,
        "files": {},
        "frames": {},
        "range_checks": [],
        "errors": [],
    }

    for rel in REQUIRED_FILES:
        local_path = local_scene / rel
        key = f"{remote_scene}/{rel}"
        head = head_object(client, args.bucket, key)
        remote_size = object_size(head)
        local_size = local_path.stat().st_size if local_path.exists() else None
        ok = local_size is not None and remote_size == local_size
        item["files"][rel] = {
            "local_size": local_size,
            "remote_size": remote_size,
            "etag": (head or {}).get("ETag"),
            "ok": ok,
        }
        if not ok:
            item["ok"] = False
            item["errors"].append(f"{rel}: local_size={local_size}, remote_size={remote_size}")

    local_frames_dir = local_scene / "iphone/frames"
    local_frame_count = len(list(local_frames_dir.glob("*.jpg"))) if local_frames_dir.exists() else 0
    remote_frame_count, remote_frame_bytes = list_frame_count(client, args.bucket, f"{remote_scene}/iphone/frames/")
    frame_ok = local_frame_count == remote_frame_count and (expected_frames is None or remote_frame_count == expected_frames)
    item["frames"] = {
        "local_count": local_frame_count,
        "remote_count": remote_frame_count,
        "remote_total_bytes": remote_frame_bytes,
        "ok": frame_ok,
    }
    if not frame_ok:
        item["ok"] = False
        item["errors"].append(
            f"frames: local_count={local_frame_count}, remote_count={remote_frame_count}, expected={expected_frames}"
        )

    if do_range_check:
        for rel in ["precomputed.h5", "iphone/depth.bin"]:
            file_info = item["files"].get(rel, {})
            remote_size = file_info.get("remote_size")
            local_path = local_scene / rel
            if isinstance(remote_size, int) and local_path.exists():
                try:
                    rc = range_compare_file(
                        client,
                        args.bucket,
                        f"{remote_scene}/{rel}",
                        local_path,
                        remote_size,
                        args.range_bytes,
                    )
                except Exception as exc:
                    rc = {"key": f"{remote_scene}/{rel}", "ok": False, "error": str(exc)}
                item["range_checks"].append(rc)
                if not rc.get("ok", False):
                    item["ok"] = False
                    item["errors"].append(f"range check failed: {rel}")

    return item


def write_markdown(path: Path, args: argparse.Namespace, scenes: list[str], records: dict[str, dict[str, Any]], results: list[dict[str, Any]], elapsed: float) -> None:
    ok_count = sum(1 for item in results if item.get("ok"))
    bad = [item for item in results if not item.get("ok")]
    range_results = [rc for item in results for rc in item.get("range_checks", [])]
    lines = [
        "# ScanNet++ COS 上传审计报告",
        "",
        f"- COS prefix: `cos://{args.bucket}/{args.remote_prefix.rstrip('/')}`",
        f"- 本地基准: `{args.local_root}`",
        f"- split 场景数: `{len(scenes)}`",
        f"- scenes_record 场景数: `{len(records)}`",
        f"- 本次审计场景数: `{len(results)}`",
        f"- 通过: `{ok_count}` / `{len(results)}`",
        f"- 耗时: `{elapsed:.1f}s`",
        "",
        "## 结论",
        "",
    ]
    if bad:
        lines.append(f"发现 `{len(bad)}` 个 scene 有缺失或大小不一致，不能认为 COS 数据完整。")
    else:
        lines.append("本次审计的 scene 中，关键对象均存在且 COS 精确字节大小与本地基准一致；帧数也与本地 frames 和 scenes_record 中的 `num_frames` 对齐。")
    if range_results:
        ok_range = sum(1 for rc in range_results if rc.get("ok"))
        lines.append(f"额外 Range 内容校验：`{ok_range}` / `{len(range_results)}` 个大文件首尾字节块与本地一致。")
    lines.extend(
        [
            "",
            "## 检查项",
            "",
            "- 必需对象：`precomputed.h5`, `iphone/depth.bin`, `iphone/depth_chunk_index.pkl`, `iphone/colmap/images.txt`, `iphone/colmap/cameras.txt`, `iphone/pose_intrinsic_imu.json`。",
            "- 帧目录：`iphone/frames/*.jpg` 的远端数量必须等于本地数量，并且等于 `scenes_record.num_frames`。",
            "- 对 `--range-check-scene` 指定的 scene，额外检查 `precomputed.h5` 和 `depth.bin` 的首尾字节块 MD5。",
            "",
            "## Scene 结果",
            "",
            "| scene | 结果 | frames local/remote/expected | 错误 |",
            "|---|---|---:|---|",
        ]
    )
    for item in results:
        frames = item.get("frames", {})
        frame_text = f"{frames.get('local_count')}/{frames.get('remote_count')}/{item.get('expected_frames')}"
        errors = "; ".join(item.get("errors", []))
        lines.append(f"| `{item['scene']}` | {'通过' if item.get('ok') else '失败'} | `{frame_text}` | `{errors}` |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    out_json = Path(args.report_json)
    out_md = Path(args.report_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    split_names = read_split_names(Path(args.splits_dir))
    records = read_records(Path(args.scenes_record))
    selected = list(dict.fromkeys(args.scene or split_names))
    if args.limit_scenes > 0:
        selected = selected[: args.limit_scenes]

    range_check = set(args.range_check_scene)
    start = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                audit_scene,
                args,
                scene,
                int(records[scene]["num_frames"]) if scene in records and "num_frames" in records[scene] else None,
                scene in range_check,
            ): scene
            for scene in selected
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            item = future.result()
            results.append(item)
            status = "OK" if item.get("ok") else "FAIL"
            print(f"[{idx}/{len(futures)}] {status} {item['scene']}", flush=True)

    results.sort(key=lambda item: selected.index(item["scene"]) if item["scene"] in selected else 10**9)
    elapsed = time.perf_counter() - start
    payload = {
        "status": "PASS" if all(item.get("ok") for item in results) else "FAIL",
        "args": vars(args),
        "split_scene_count": len(split_names),
        "record_scene_count": len(records),
        "audited_scene_count": len(results),
        "elapsed_sec": elapsed,
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    write_markdown(out_md, args, split_names, records, results, elapsed)
    print(out_json)
    print(out_md)
    print("status", payload["status"])
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
