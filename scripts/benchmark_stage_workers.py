#!/usr/bin/env python3
"""Benchmark SampleLocalStager materialization with different SDK worker counts."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.registry import create_adapter
from datasets.sample_stage import SampleLocalStager, SampleStageConfig
from datasets.sampling import DatasetSampler


def _latest_config(default: str) -> str:
    tmpdir = Path("/data1/zbf/d4rt_tmp")
    candidates = sorted(
        tmpdir.glob("mixture_5datasets_cos_planned.*.yaml"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else default


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dataset_cfg(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    for item in config.get("datasets", []):
        if item.get("name") == dataset:
            return item
    raise KeyError(f"dataset {dataset!r} not found in config")


def _build_adapter(config: dict[str, Any], dataset: str):
    cfg = _dataset_cfg(config, dataset)
    kwargs = dict(cfg.get("adapter_kwargs", {}))
    if config.get("index_cache_dir"):
        kwargs["cache_dir"] = config["index_cache_dir"]
    if config.get("index_workers") is not None:
        kwargs["index_workers"] = config["index_workers"]
    return create_adapter(
        name=dataset,
        root=cfg["root"],
        split=cfg.get("split", "train"),
        **kwargs,
    )


def _sample_clip(config: dict[str, Any], adapter, dataset: str, seed: int) -> tuple[str, list[int]]:
    cfg = _dataset_cfg(config, dataset)
    sampler = DatasetSampler(
        adapter=adapter,
        clip_len=int(config.get("clip_len", 48)),
        sampling_mode=str(config.get("sampling_mode", "stride")),
        min_frames=int(config.get("clip_len", 48)),
        allowed_sequences=cfg.get("train_sequences") or cfg.get("sequences"),
        custom_stride_range=tuple(config["stride_range"]) if "stride_range" in config else None,
        sequence_locality_size=int(config.get("sequence_locality_size", 1)),
        frame_locality_radius=config.get("frame_locality_radius"),
    )
    return sampler.sample(random.Random(seed))


def _parse_workers(raw: str) -> list[int]:
    return [max(1, int(part)) for part in raw.replace(" ", "").split(",") if part]


def _cache_bytes(stage_root: Path) -> int:
    root = stage_root / "shared_raw_cache" / "data"
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if path.is_file() and ".part." not in path.name:
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


def _run_stage_once(stager: SampleLocalStager, adapter, dataset: str, seq: str, frames: list[int], tag: str) -> float:
    t0 = time.perf_counter()
    with stager.stage_sample(adapter, seq, frames, sample_tag=tag):
        pass
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="latest")
    parser.add_argument("--datasets", default="kubric,dynamic_replica,scannetpp")
    parser.add_argument("--workers", default="2,4,8,16")
    parser.add_argument("--stage-root", default="/data1/zbf/d4rt_stage_worker_probe")
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--window-radius", type=int, default=0)
    parser.add_argument("--cache-max-gb", type=float, default=20.0)
    parser.add_argument("--request-timeout-s", type=float, default=15.0)
    parser.add_argument("--request-retries", type=int, default=1)
    parser.add_argument("--warm-run", action="store_true")
    parser.add_argument("--clear", action="store_true", default=True)
    args = parser.parse_args()

    config_path = _latest_config("configs/mixture_5datasets_cos_planned.yaml") if args.config == "latest" else args.config
    config = _load_config(config_path)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    workers = _parse_workers(args.workers)
    base_stage_root = Path(args.stage_root)

    print(f"[stage-workers] config={config_path}", flush=True)
    print(f"[stage-workers] datasets={datasets} workers={workers} stage_root={base_stage_root}", flush=True)

    for dataset in datasets:
        print(f"\n[stage-workers] init dataset={dataset}", flush=True)
        adapter = _build_adapter(config, dataset)
        seq, frames = _sample_clip(config, adapter, dataset, args.seed)
        print(
            f"[stage-workers] sample dataset={dataset} seq={seq} "
            f"frames={len(frames)}[{min(frames)}..{max(frames)}]",
            flush=True,
        )

        for sdk_workers in workers:
            run_root = base_stage_root / dataset / f"w{sdk_workers}"
            if args.clear:
                shutil.rmtree(run_root, ignore_errors=True)
            stager = SampleLocalStager(
                SampleStageConfig(
                    backend="cos_sdk",
                    stage_root=str(run_root),
                    sdk_workers=sdk_workers,
                    request_timeout_s=args.request_timeout_s,
                    request_retries=args.request_retries,
                    cache_max_bytes=int(args.cache_max_gb * 1024**3),
                    cache_low_watermark_ratio=config.get("sample_stage_cache_low_watermark_ratio", 0.9),
                    cache_touch_interval_s=config.get("sample_stage_cache_touch_interval_s", 30.0),
                    cache_scan_interval_s=config.get("sample_stage_cache_scan_interval_s", 30.0),
                    window_radius=args.window_radius,
                    mount_root=config.get("sample_stage_mount_root", "/data_cos"),
                    bucket=config.get("sample_stage_bucket", "hd-ai-data-1251882982"),
                    region=config.get("sample_stage_region", "ap-beijing"),
                    passwd_file=config.get("sample_stage_passwd_file", "/etc/passwd-s3fs-data_cos"),
                    enabled_datasets=(dataset,),
                    scene_prefetch_datasets=(),
                )
            )
            manifest = stager._build_manifest(adapter, seq, frames)
            try:
                cold_s = _run_stage_once(
                    stager, adapter, dataset, seq, frames, f"probe_{dataset}_w{sdk_workers}_cold"
                )
                warm_s = None
                if args.warm_run:
                    warm_s = _run_stage_once(
                        stager, adapter, dataset, seq, frames, f"probe_{dataset}_w{sdk_workers}_warm"
                    )
                cache_gb = _cache_bytes(run_root) / 1024**3
                msg = (
                    f"[StageWorkersResult] dataset={dataset} workers={sdk_workers} "
                    f"files={len(manifest)} cold={cold_s:.3f}s cache={cache_gb:.2f}GB"
                )
                if warm_s is not None:
                    msg += f" warm={warm_s:.3f}s"
                print(msg, flush=True)
            except Exception as exc:
                print(
                    f"[StageWorkersResult] dataset={dataset} workers={sdk_workers} "
                    f"files={len(manifest)} error={type(exc).__name__}: {exc}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
