#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import yaml
from qcloud_cos import CosConfig, CosS3Client

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets.registry import DATASET_REGISTRY
from datasets.sampling import DatasetSampler


BUCKET = "hd-ai-data-1251882982"
REGION = "ap-beijing"
MOUNT_ROOT = Path("/data_cos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark exact-sample COS SDK prefetch versus direct /data_cos "
            "load_clip() for COS-backed datasets."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/mixture_5datasets_cos_planned.yaml",
        help="Mixture config YAML.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["kubric", "dynamic_replica", "blendedmvs"],
        help="Subset of dataset names to benchmark.",
    )
    parser.add_argument(
        "--sdk-workers",
        type=int,
        default=16,
        help="Concurrent COS SDK download workers.",
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=42,
        help="Seed for DatasetSampler.sample().",
    )
    parser.add_argument(
        "--stage-root",
        default="/data1/zbf/d4rt_sdk_sample_bench",
        help="Temporary local staging root for exact-file prefetch.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--keep-stage",
        action="store_true",
        help="Keep staged files after benchmark.",
    )
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _dataset_entries(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in config["datasets"]}


def _common_factory_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if config.get("index_cache_dir"):
        out["cache_dir"] = config["index_cache_dir"]
    if config.get("index_workers") is not None:
        out["index_workers"] = config["index_workers"]
    return out


def _make_adapter(config: dict[str, Any], dataset_name: str):
    entry = _dataset_entries(config)[dataset_name]
    adapter_cls = DATASET_REGISTRY[dataset_name]
    kwargs = dict(entry.get("adapter_kwargs", {}))
    kwargs.update(_common_factory_kwargs(config))
    return adapter_cls(
        root=entry["root"],
        split=entry.get("split", "train"),
        **kwargs,
    )


def _make_sampler(config: dict[str, Any], adapter, dataset_name: str) -> DatasetSampler:
    entry = _dataset_entries(config)[dataset_name]
    return DatasetSampler(
        adapter=adapter,
        clip_len=config.get("clip_len", 48),
        sampling_mode=config.get("sampling_mode", "stride"),
        min_frames=config.get("clip_len", 48),
        allowed_sequences=entry.get("train_sequences") or entry.get("sequences"),
        custom_stride_range=tuple(config["stride_range"]) if "stride_range" in config else None,
        sequence_locality_size=config.get("sequence_locality_size", 3),
        frame_locality_radius=config.get("frame_locality_radius", config.get("clip_len", 48)),
    )


def _sample_once(config: dict[str, Any], adapter, dataset_name: str, seed: int) -> tuple[str, list[int]]:
    sampler = _make_sampler(config, adapter, dataset_name)
    rng = random.Random(seed)
    sequence_name, frame_indices = sampler.sample(rng)
    return sequence_name, frame_indices


def _manifest_kubric(adapter, sequence_name: str, frame_indices: list[int]) -> list[Path]:
    record = adapter._get_record(sequence_name)
    paths: list[Path] = [record.rank_path]
    if record.h5_path is not None:
        paths.append(record.h5_path)
        if record.depth_names is not None or record.depth_dir.exists():
            paths.extend(adapter._depth_path_for_index(record, i) for i in frame_indices)
    else:
        paths.append(record.ann_path)
    paths.extend(adapter._frame_path_for_index(record, i) for i in frame_indices)
    return paths


def _manifest_dynamic_replica(adapter, sequence_name: str, frame_indices: list[int]) -> list[Path]:
    record = adapter._get_record(sequence_name)
    paths: list[Path] = [adapter._annotation_file()]
    for idx in frame_indices:
        paths.append(adapter.split_root / record.image_rel_paths[idx])
        paths.append(adapter.split_root / record.depth_rel_paths[idx])
        traj_rel = record.traj_rel_paths[idx]
        if traj_rel is not None:
            paths.append(adapter.split_root / traj_rel)
    return paths


def _manifest_blendedmvs(adapter, sequence_name: str, frame_indices: list[int]) -> list[Path]:
    record = adapter._get_record(sequence_name)
    paths: list[Path] = []
    for idx in frame_indices:
        fid = record.frame_ids[idx]
        paths.append(record.rgb_path(fid, adapter.use_masked))
        paths.append(record.depth_path(fid))
        paths.append(record.cam_path(fid))
    return paths


MANIFEST_BUILDERS: dict[str, Callable[..., list[Path]]] = {
    "kubric": _manifest_kubric,
    "dynamic_replica": _manifest_dynamic_replica,
    "blendedmvs": _manifest_blendedmvs,
}


def _dedupe_keep_order(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            out.append(path)
            seen.add(path)
    return out


def _to_cos_key(path: Path) -> str:
    # Keep this lexical.  Calling Path.resolve() on /data_cos paths goes
    # through the mounted filesystem and can turn a COS-key conversion into a
    # slow metadata operation, badly skewing SDK prefetch benchmarks.
    path = Path(path)
    try:
        rel_path = path.relative_to(MOUNT_ROOT)
    except ValueError:
        path_s = str(path)
        prefix = str(MOUNT_ROOT).rstrip("/") + "/"
        if not path_s.startswith(prefix):
            raise
        rel_path = Path(path_s[len(prefix):])
    return str(rel_path).replace(os.sep, "/")


def _stage_target(stage_root: Path, key: str) -> Path:
    return stage_root / Path(key)


def _stat_manifest(paths: list[Path]) -> dict[str, Any]:
    sizes = [p.stat().st_size for p in paths]
    return {
        "file_count": len(paths),
        "total_bytes": sum(sizes),
        "total_mb": sum(sizes) / 1024**2,
        "largest_file_mb": (max(sizes) / 1024**2) if sizes else 0.0,
    }


def _make_cos_client() -> CosS3Client:
    secret_id, secret_key = Path("/etc/passwd-s3fs-data_cos").read_text().strip().split(":", 1)
    config = CosConfig(
        Region=REGION,
        SecretId=secret_id,
        SecretKey=secret_key,
        Scheme="https",
    )
    return CosS3Client(config)


def _download_manifest_sdk(paths: list[Path], stage_root: Path, workers: int) -> dict[str, Any]:
    tls = threading.local()

    def get_client() -> CosS3Client:
        client = getattr(tls, "client", None)
        if client is None:
            client = _make_cos_client()
            tls.client = client
        return client

    def download_one(src_path: Path) -> dict[str, Any]:
        key = _to_cos_key(src_path)
        dst = _stage_target(stage_root, key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        client = get_client()
        t0 = time.perf_counter()
        resp = client.get_object(Bucket=BUCKET, Key=key)
        body = resp["Body"].get_raw_stream().read()
        with open(dst, "wb") as f:
            f.write(body)
        dt = time.perf_counter() - t0
        return {
            "key": key,
            "bytes": len(body),
            "seconds": dt,
        }

    start = time.perf_counter()
    items: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(download_one, p) for p in paths]
        for fut in as_completed(futures):
            items.append(fut.result())
    total_s = time.perf_counter() - start
    total_bytes = sum(item["bytes"] for item in items)
    return {
        "seconds": total_s,
        "total_bytes": total_bytes,
        "mb_per_s": (total_bytes / 1024**2 / total_s) if total_s > 0 else 0.0,
        "workers": workers,
        "items": items,
    }


@contextlib.contextmanager
def _rebase_adapter_paths(adapter, dataset_name: str, sequence_name: str, staged_dataset_root: Path):
    if dataset_name == "kubric":
        old_record = adapter._name_to_record[sequence_name]
        new_record = dataclasses.replace(
            old_record,
            scene_dir=staged_dataset_root / old_record.scene_dir.relative_to(adapter.root),
            ann_path=staged_dataset_root / old_record.ann_path.relative_to(adapter.root),
            rank_path=staged_dataset_root / old_record.rank_path.relative_to(adapter.root),
            h5_path=(
                staged_dataset_root / old_record.h5_path.relative_to(adapter.root)
                if old_record.h5_path is not None
                else None
            ),
            trajs_2d_path=(
                staged_dataset_root / old_record.trajs_2d_path.relative_to(adapter.root)
                if old_record.trajs_2d_path is not None
                else None
            ),
        )
        old_root = adapter.root
        adapter.root = staged_dataset_root
        adapter._name_to_record[sequence_name] = new_record
        try:
            yield
        finally:
            adapter.root = old_root
            adapter._name_to_record[sequence_name] = old_record
        return

    if dataset_name == "dynamic_replica":
        old_root = adapter.root
        old_split_root = adapter.split_root
        old_record = adapter._name_to_record[sequence_name]
        new_record = dataclasses.replace(
            old_record,
            sequence_dir=staged_dataset_root / old_record.sequence_dir.relative_to(adapter.root),
        )
        adapter.root = staged_dataset_root
        adapter.split_root = staged_dataset_root / adapter.split
        adapter._name_to_record[sequence_name] = new_record
        try:
            yield
        finally:
            adapter.root = old_root
            adapter.split_root = old_split_root
            adapter._name_to_record[sequence_name] = old_record
        return

    if dataset_name == "blendedmvs":
        old_record = adapter._name_to_record[sequence_name]
        new_record = old_record.__class__(
            scene_id=old_record.scene_id,
            scene_dir=staged_dataset_root / old_record.scene_dir.relative_to(adapter.root),
            frame_ids=old_record.frame_ids,
        )
        old_root = adapter.root
        adapter.root = staged_dataset_root
        adapter._name_to_record[sequence_name] = new_record
        try:
            yield
        finally:
            adapter.root = old_root
            adapter._name_to_record[sequence_name] = old_record
        return

    raise ValueError(f"Unsupported dataset for rebase: {dataset_name}")


def _time_load_clip(adapter, sequence_name: str, frame_indices: list[int]) -> dict[str, Any]:
    t0 = time.perf_counter()
    clip = adapter.load_clip(sequence_name, frame_indices)
    dt = time.perf_counter() - t0
    return {
        "seconds": dt,
        "video_frames": len(clip.images),
        "num_paths": len(clip.frame_paths),
        "height": int(clip.images[0].shape[0]) if clip.images else None,
        "width": int(clip.images[0].shape[1]) if clip.images else None,
    }


def _dataset_root_from_config(config: dict[str, Any], dataset_name: str) -> Path:
    return Path(_dataset_entries(config)[dataset_name]["root"]).resolve()


def _run_one_dataset(
    config: dict[str, Any],
    dataset_name: str,
    sampling_seed: int,
    sdk_workers: int,
    stage_root: Path,
) -> dict[str, Any]:
    adapter = _make_adapter(config, dataset_name)
    sequence_name, frame_indices = _sample_once(config, adapter, dataset_name, sampling_seed)
    manifest = _dedupe_keep_order(MANIFEST_BUILDERS[dataset_name](adapter, sequence_name, frame_indices))

    manifest_stats = _stat_manifest(manifest)
    mounted = _time_load_clip(adapter, sequence_name, frame_indices)

    stage_root.mkdir(parents=True, exist_ok=True)
    sdk = _download_manifest_sdk(manifest, stage_root, workers=sdk_workers)

    staged_dataset_root = stage_root / _to_cos_key(_dataset_root_from_config(config, dataset_name))
    with _rebase_adapter_paths(adapter, dataset_name, sequence_name, staged_dataset_root):
        local = _time_load_clip(adapter, sequence_name, frame_indices)

    return {
        "dataset": dataset_name,
        "sequence_name": sequence_name,
        "frame_indices": frame_indices,
        "manifest": manifest_stats,
        "mounted_load": mounted,
        "sdk_prefetch": sdk,
        "local_load_after_prefetch": local,
        "sdk_plus_local_total_s": sdk["seconds"] + local["seconds"],
        "speedup_vs_mounted": (
            mounted["seconds"] / (sdk["seconds"] + local["seconds"])
            if (sdk["seconds"] + local["seconds"]) > 0
            else None
        ),
    }


def main() -> int:
    args = parse_args()
    config = _load_config((REPO_ROOT / args.config).resolve())

    ts = time.strftime("%Y%m%d_%H%M%S")
    bench_root = Path(args.stage_root) / f"bench_{ts}_pid{os.getpid()}"
    reports: list[dict[str, Any]] = []

    try:
        for dataset_name in args.datasets:
            if dataset_name not in MANIFEST_BUILDERS:
                raise ValueError(
                    f"Unsupported dataset {dataset_name!r}. "
                    f"Supported: {sorted(MANIFEST_BUILDERS)}"
                )
            print(f"[sdk-bench] dataset={dataset_name}", flush=True)
            dataset_stage_root = bench_root / dataset_name
            result = _run_one_dataset(
                config=config,
                dataset_name=dataset_name,
                sampling_seed=args.sampling_seed,
                sdk_workers=args.sdk_workers,
                stage_root=dataset_stage_root,
            )
            reports.append(result)
            print(
                json.dumps(
                    {
                        "dataset": result["dataset"],
                        "sequence_name": result["sequence_name"],
                        "manifest": result["manifest"],
                        "mounted_load_s": round(result["mounted_load"]["seconds"], 3),
                        "sdk_prefetch_s": round(result["sdk_prefetch"]["seconds"], 3),
                        "local_load_after_prefetch_s": round(result["local_load_after_prefetch"]["seconds"], 3),
                        "sdk_plus_local_total_s": round(result["sdk_plus_local_total_s"], 3),
                        "speedup_vs_mounted": round(result["speedup_vs_mounted"], 3)
                        if result["speedup_vs_mounted"] is not None
                        else None,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        if not args.keep_stage:
            shutil.rmtree(bench_root, ignore_errors=True)

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n")
        print(f"[sdk-bench] report_json={report_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
