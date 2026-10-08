#!/usr/bin/env python3
"""Benchmark concurrent SampleLocalStager manifest materialization."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _make_sampler(config: dict[str, Any], adapter, dataset: str) -> DatasetSampler:
    cfg = _dataset_cfg(config, dataset)
    return DatasetSampler(
        adapter=adapter,
        clip_len=int(config.get("clip_len", 48)),
        sampling_mode=str(config.get("sampling_mode", "stride")),
        min_frames=int(config.get("clip_len", 48)),
        allowed_sequences=cfg.get("train_sequences") or cfg.get("sequences"),
        custom_stride_range=tuple(config["stride_range"]) if "stride_range" in config else None,
        sequence_locality_size=1,
        frame_locality_radius=config.get("frame_locality_radius"),
    )


def _parse_ints(raw: str) -> list[int]:
    return [max(1, int(part)) for part in raw.replace(" ", "").split(",") if part]


def _pct(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * pct / 100.0))))
    return values[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="latest")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--concurrency", default="4,8,16")
    parser.add_argument("--sdk-workers", default="2,4,8,16")
    parser.add_argument("--stage-root", default="/data1/zbf/d4rt_stage_concurrency_probe")
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--request-timeout-s", type=float, default=15.0)
    parser.add_argument("--request-retries", type=int, default=1)
    parser.add_argument("--clear", action="store_true", default=True)
    args = parser.parse_args()

    config_path = _latest_config("configs/mixture_5datasets_cos_planned.yaml") if args.config == "latest" else args.config
    config = _load_config(config_path)
    conc_values = _parse_ints(args.concurrency)
    sdk_values = _parse_ints(args.sdk_workers)
    base_root = Path(args.stage_root)

    print(
        f"[stage-concurrency] config={config_path} dataset={args.dataset} "
        f"samples={args.samples} concurrency={conc_values} sdk_workers={sdk_values}",
        flush=True,
    )

    adapter = _build_adapter(config, args.dataset)
    sampler = _make_sampler(config, adapter, args.dataset)
    rng = random.Random(args.seed)
    plans: list[tuple[str, list[int], list[Path]]] = []
    for i in range(args.samples):
        seq, frames = sampler.sample(rng)
        temp_stager = SampleLocalStager(
            SampleStageConfig(
                backend="cos_sdk",
                stage_root=str(base_root / "_manifest_only"),
                window_radius=0,
                enabled_datasets=(args.dataset,),
                mount_root=config.get("sample_stage_mount_root", "/data_cos"),
                bucket=config.get("sample_stage_bucket", "hd-ai-data-1251882982"),
                region=config.get("sample_stage_region", "ap-beijing"),
                passwd_file=config.get("sample_stage_passwd_file", "/etc/passwd-s3fs-data_cos"),
            )
        )
        manifest = temp_stager._build_manifest(adapter, seq, frames)
        plans.append((seq, frames, manifest))
        print(
            f"[stage-concurrency] plan[{i}] seq={seq} "
            f"frames={len(frames)}[{min(frames)}..{max(frames)}] files={len(manifest)}",
            flush=True,
        )

    for sdk_workers in sdk_values:
        for concurrency in conc_values:
            run_root = base_root / args.dataset / f"sw{sdk_workers}_c{concurrency}"
            if args.clear:
                shutil.rmtree(run_root, ignore_errors=True)
            stager = SampleLocalStager(
                SampleStageConfig(
                    backend="cos_sdk",
                    stage_root=str(run_root),
                    sdk_workers=sdk_workers,
                    request_timeout_s=args.request_timeout_s,
                    request_retries=args.request_retries,
                    cache_max_bytes=50 * 1024**3,
                    cache_low_watermark_ratio=config.get("sample_stage_cache_low_watermark_ratio", 0.9),
                    cache_touch_interval_s=config.get("sample_stage_cache_touch_interval_s", 30.0),
                    cache_scan_interval_s=config.get("sample_stage_cache_scan_interval_s", 30.0),
                    window_radius=0,
                    mount_root=config.get("sample_stage_mount_root", "/data_cos"),
                    bucket=config.get("sample_stage_bucket", "hd-ai-data-1251882982"),
                    region=config.get("sample_stage_region", "ap-beijing"),
                    passwd_file=config.get("sample_stage_passwd_file", "/etc/passwd-s3fs-data_cos"),
                    enabled_datasets=(args.dataset,),
                    scene_prefetch_datasets=(),
                )
            )

            def materialize_one(item: tuple[int, tuple[str, list[int], list[Path]]]) -> dict[str, Any]:
                idx, (seq, frames, manifest) = item
                temp_dir = Path(
                    tempfile.mkdtemp(prefix=f"conc_{idx:03d}_", dir=stager.work_root)
                )
                try:
                    t0 = time.perf_counter()
                    stats = stager._materialize_manifest(manifest, temp_dir)
                    elapsed = time.perf_counter() - t0
                    return {
                        "idx": idx,
                        "seq": seq,
                        "seconds": elapsed,
                        "files": len(manifest),
                        "cold": stats.get("cold_files", 0),
                        "file_max_s": stats.get("file_max_s", 0.0),
                        "file_sum_s": stats.get("file_sum_s", 0.0),
                    }
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)

            started = time.perf_counter()
            results: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = [
                    ex.submit(materialize_one, item)
                    for item in enumerate(plans)
                ]
                for fut in as_completed(futures):
                    result = fut.result()
                    results.append(result)
                    print(
                        f"[StageConcurrencySample] dataset={args.dataset} "
                        f"sdk_workers={sdk_workers} concurrency={concurrency} "
                        f"idx={result['idx']} seconds={result['seconds']:.3f} "
                        f"cold={result['cold']} file_max={result['file_max_s']:.3f}",
                        flush=True,
                    )
            wall_s = time.perf_counter() - started
            times = [float(r["seconds"]) for r in results]
            file_max = [float(r["file_max_s"]) for r in results]
            total_files = sum(int(r["files"]) for r in results)
            print(
                f"[StageConcurrencyResult] dataset={args.dataset} "
                f"sdk_workers={sdk_workers} concurrency={concurrency} "
                f"samples={len(results)} files={total_files} wall={wall_s:.3f}s "
                f"sample_mean={sum(times)/len(times):.3f}s "
                f"sample_p50={_pct(times, 50):.3f}s sample_p90={_pct(times, 90):.3f}s "
                f"sample_max={max(times):.3f}s file_max_max={max(file_max):.3f}s "
                f"throughput={len(results)/wall_s:.2f}samples/s",
                flush=True,
            )


if __name__ == "__main__":
    main()
