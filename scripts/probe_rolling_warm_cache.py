#!/usr/bin/env python3
"""Probe planned rolling-warm raw cache feasibility.

This is intentionally not wired into training. It uses the real planned sampler
and SampleLocalStager manifest logic to answer one question:

  Can a separate high-concurrency warm phase materialize the raw COS files for a
  future block faster than training would consume that block?

The probe does not build QuerySample bundles, does not create a model, and does
not use CUDA. Use a separate stage root so it cannot disturb a running training
cache.
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from datasets.factory import create_training_dataset
from datasets.sample_stage import SampleLocalStager, SampleStageConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--block-batches", type=int, default=1)
    parser.add_argument("--sdk-workers", type=int, default=64)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--compute-s-per-batch", type=float, default=2.5)
    parser.add_argument("--ranks", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clean-stage-root", action="store_true")
    parser.add_argument("--second-pass", action="store_true")
    return parser.parse_args()


def _safe_clean_stage_root(path: Path) -> None:
    text = str(path)
    if "rolling_warm_probe" not in text:
        raise SystemExit(
            f"Refusing to clean stage root without 'rolling_warm_probe' in path: {path}"
        )
    if path.exists():
        shutil.rmtree(path)


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    config = dict(config)
    config["planned_mode"] = True
    config["planned_start_immediately"] = False
    config["planned_initial_epoch"] = int(args.epoch)
    config["sample_stage_root"] = str(args.stage_root)
    config["sample_stage_sdk_workers"] = int(args.sdk_workers)
    config["sample_stage_request_timeout_s"] = float(args.timeout_s)
    config["sample_stage_request_retries"] = int(args.retries)
    config["sample_stage_eviction_mode"] = "disabled"
    config.setdefault(
        "sample_stage_datasets",
        ["pointodyssey", "kubric", "dynamic_replica", "co3dv2", "scannetpp"],
    )
    return config


def _stager_from_config(config: dict[str, Any]) -> SampleLocalStager:
    return SampleLocalStager(
        SampleStageConfig.from_dict(
            {
                "backend": config.get("sample_stage_backend", "cos_sdk"),
                "stage_root": config["sample_stage_root"],
                "sdk_workers": config.get("sample_stage_sdk_workers", 64),
                "request_timeout_s": config.get("sample_stage_request_timeout_s", 20.0),
                "request_retries": config.get("sample_stage_request_retries", 2),
                "cache_max_bytes": config.get("sample_stage_cache_max_bytes", 400 * 1024**3),
                "cache_low_watermark_ratio": config.get(
                    "sample_stage_cache_low_watermark_ratio", 0.85
                ),
                "cache_touch_interval_s": 0.0,
                "cache_scan_interval_s": 3600.0,
                "eviction_mode": "disabled",
                "window_radius": config.get("sample_stage_window_radius", 0),
                "mount_root": config.get("sample_stage_mount_root", "/data_cos"),
                "bucket": config.get("sample_stage_bucket", "hd-ai-data-1251882982"),
                "region": config.get("sample_stage_region", "ap-beijing"),
                "passwd_file": config.get(
                    "sample_stage_passwd_file", "/etc/passwd-s3fs-data_cos"
                ),
                "enabled_datasets": config.get("sample_stage_datasets", ()),
                "scene_prefetch_datasets": (),
            }
        )
    )


def _rank_list(raw: str, world_size: int) -> list[int]:
    if not raw.strip():
        return list(range(world_size))
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0 or value >= world_size:
            raise SystemExit(f"rank out of range: {value}")
        out.append(value)
    return sorted(set(out))


def _collect_manifests(
    config: dict[str, Any],
    stager: SampleLocalStager,
    *,
    ranks: list[int],
    world_size: int,
    batch_size: int,
    start_batch: int,
    block_batches: int,
) -> tuple[list[Path], Counter[str], Counter[str], int]:
    unique: dict[str, Path] = {}
    dataset_counts: Counter[str] = Counter()
    file_counts_by_dataset: Counter[str] = Counter()
    total_specs = 0
    start = int(start_batch) * int(batch_size)
    end = start + int(block_batches) * int(batch_size)

    for rank in ranks:
        t0 = time.perf_counter()
        dataset = create_training_dataset(
            config,
            split="train",
            rank=rank,
            world_size=world_size,
        )
        init_s = time.perf_counter() - t0
        plan = dataset.current_plan
        print(
            f"[RollingWarmProbe] rank={rank} dataset_init={init_s:.2f}s "
            f"plan_len={len(plan)} local_range=[{start},{min(end, len(plan))})",
            flush=True,
        )
        for spec in plan[start:end]:
            adapter = dataset.adapters[spec.dataset_idx]
            dataset_name = str(getattr(adapter, "dataset_name", type(adapter).__name__))
            dataset_counts[dataset_name] += 1
            total_specs += 1
            if not stager.supports(adapter):
                continue
            manifest = stager._build_manifest(
                adapter,
                spec.sequence_name,
                list(spec.frame_indices),
            )
            file_counts_by_dataset[dataset_name] += len(manifest)
            for path in manifest:
                rel_key = stager._to_cos_key(Path(path))
                unique.setdefault(rel_key, Path(path))
        try:
            dataset.cleanup()
        except Exception:
            pass
    return list(unique.values()), dataset_counts, file_counts_by_dataset, total_specs


def _cache_bytes(stager: SampleLocalStager, paths: list[Path]) -> tuple[int, int]:
    total = 0
    present = 0
    for path in paths:
        cache_path = stager.cache_data_root / Path(stager._to_cos_key(path))
        try:
            stat = cache_path.stat()
        except OSError:
            continue
        present += 1
        total += stat.st_size
    return present, total


def _summarize_file_sizes(stager: SampleLocalStager, paths: list[Path]) -> str:
    sizes = []
    for path in paths:
        cache_path = stager.cache_data_root / Path(stager._to_cos_key(path))
        try:
            sizes.append(cache_path.stat().st_size)
        except OSError:
            pass
    if not sizes:
        return "sizes=<none>"
    sizes.sort()
    return (
        f"bytes={sum(sizes)} mb={sum(sizes) / 1024**2:.1f} "
        f"p50_kb={sizes[len(sizes)//2] / 1024:.1f} "
        f"max_mb={max(sizes) / 1024**2:.1f}"
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("D4RT_SERIALIZE_ADAPTER_INIT", "1")

    stage_root = Path(args.stage_root)
    if args.clean_stage_root:
        _safe_clean_stage_root(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    config = _load_config(args)
    stager = _stager_from_config(config)
    ranks = _rank_list(args.ranks, args.world_size)

    print(
        "[RollingWarmProbe] start "
        f"config={args.config} stage_root={stage_root} epoch={args.epoch} "
        f"world_size={args.world_size} ranks={ranks} batch_size={args.batch_size} "
        f"start_batch={args.start_batch} block_batches={args.block_batches} "
        f"sdk_workers={args.sdk_workers} timeout={args.timeout_s}s retries={args.retries}",
        flush=True,
    )

    t_collect0 = time.perf_counter()
    paths, dataset_counts, manifest_counts, total_specs = _collect_manifests(
        config,
        stager,
        ranks=ranks,
        world_size=args.world_size,
        batch_size=args.batch_size,
        start_batch=args.start_batch,
        block_batches=args.block_batches,
    )
    collect_s = time.perf_counter() - t_collect0
    existing, existing_bytes = _cache_bytes(stager, paths)
    consume_s = float(args.block_batches) * float(args.compute_s_per_batch)
    print(
        "[RollingWarmProbe] manifest "
        f"specs={total_specs} unique_files={len(paths)} "
        f"existing={existing} existing_mb={existing_bytes / 1024**2:.1f} "
        f"collect={collect_s:.2f}s consume_budget={consume_s:.2f}s "
        f"datasets={dict(dataset_counts)} manifest_files={dict(manifest_counts)}",
        flush=True,
    )

    if args.dry_run:
        return

    t_warm0 = time.perf_counter()
    stats = stager._materialize_manifest_cache_only(paths)
    warm_s = time.perf_counter() - t_warm0
    present, bytes_total = _cache_bytes(stager, paths)
    keep_up = warm_s <= consume_s
    print(
        "[RollingWarmProbe] warm "
        f"time={warm_s:.3f}s keep_up={keep_up} "
        f"budget={consume_s:.3f}s ratio={warm_s / max(1e-6, consume_s):.2f} "
        f"present={present}/{len(paths)} {_summarize_file_sizes(stager, paths)} "
        f"cold_files={stats.get('cold_files')} "
        f"file_max={stats.get('file_max_s'):.3f}s "
        f"file_sum={stats.get('file_sum_s'):.3f}s",
        flush=True,
    )

    if args.second_pass:
        t_hot0 = time.perf_counter()
        hot_stats = stager._materialize_manifest_cache_only(paths)
        hot_s = time.perf_counter() - t_hot0
        print(
            "[RollingWarmProbe] hot_second_pass "
            f"time={hot_s:.3f}s budget={consume_s:.3f}s "
            f"ratio={hot_s / max(1e-6, consume_s):.2f} "
            f"cold_files={hot_stats.get('cold_files')} "
            f"file_max={hot_stats.get('file_max_s'):.3f}s "
            f"file_sum={hot_stats.get('file_sum_s'):.3f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
