#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets.factory import create_training_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark planned-mode sample spool throughput with an isolated spool "
            "directory so it does not interfere with active training runs."
        )
    )
    parser.add_argument(
        "--config",
        default="latest-effective",
        help=(
            "Config path to benchmark. Use 'latest-effective' to auto-pick the newest "
            "mixture_5datasets_cos_planned.*.yaml under --tmpdir."
        ),
    )
    parser.add_argument(
        "--tmpdir",
        default="/data1/zbf/d4rt_tmp",
        help="Directory to search for latest effective config when --config=latest-effective.",
    )
    parser.add_argument(
        "--spool-root",
        default="/data1/zbf/d4rt_spool_bench",
        help="Parent directory for the isolated benchmark spool.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="How many sequential samples to consume per benchmark batch.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=8,
        help="How many benchmark batches to consume.",
    )
    parser.add_argument(
        "--compute-ms",
        type=float,
        default=0.0,
        help=(
            "Optional sleep after each benchmark batch to simulate GPU compute time "
            "while builders keep prefetching."
        ),
    )
    parser.add_argument(
        "--builder-workers",
        type=int,
        default=None,
        help="Override planned-mode builder_workers.",
    )
    parser.add_argument(
        "--prefetch-depth",
        type=int,
        default=None,
        help="Override planned-mode prefetch_depth.",
    )
    parser.add_argument(
        "--epoch-size",
        type=int,
        default=None,
        help="Override epoch_size. Defaults to batch_size * num_batches.",
    )
    parser.add_argument(
        "--index-cache-dir",
        default=None,
        help="Optional override for index_cache_dir.",
    )
    parser.add_argument(
        "--keep-spool",
        action="store_true",
        help="Do not delete benchmark spool files on exit.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to write a JSON benchmark report.",
    )
    return parser.parse_args()


def resolve_config_path(config_arg: str, tmpdir: str) -> Path:
    if config_arg != "latest-effective":
        path = Path(config_arg)
        if not path.is_file():
            raise FileNotFoundError(f"Config not found: {path}")
        return path

    candidates = sorted(
        Path(tmpdir).glob("mixture_5datasets_cos_planned.*.yaml"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No effective config matching mixture_5datasets_cos_planned.*.yaml under {tmpdir}"
        )
    return candidates[0]


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def spool_state(dataset: Any) -> tuple[int, int]:
    spool_dir = Path(dataset.spool.spool_dir)
    ready_files = list(spool_dir.glob("*.ready"))
    total_bytes = 0
    for path in ready_files:
        try:
            total_bytes += path.stat().st_size
        except OSError:
            pass
    return len(ready_files), total_bytes


def percentile_ms(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = min(len(values_sorted) - 1, max(0, int(round((q / 100.0) * (len(values_sorted) - 1)))))
    return values_sorted[idx] * 1000.0


def summarize_waits_ms(values_s: list[float], prefix: str) -> dict[str, float]:
    if not values_s:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": statistics.mean(values_s) * 1000.0,
        f"{prefix}_p50": percentile_ms(values_s, 50.0),
        f"{prefix}_p95": percentile_ms(values_s, 95.0),
        f"{prefix}_max": max(values_s) * 1000.0,
    }


def make_benchmark_config(base_config: dict[str, Any], args: argparse.Namespace, spool_dir: Path) -> dict[str, Any]:
    config = dict(base_config)
    config["planned_mode"] = True
    config["spool_dir"] = str(spool_dir)
    config["builder_workers"] = (
        args.builder_workers
        if args.builder_workers is not None
        else int(config.get("builder_workers", 2))
    )
    config["prefetch_depth"] = (
        args.prefetch_depth
        if args.prefetch_depth is not None
        else int(config.get("prefetch_depth", 32))
    )
    config["epoch_size"] = (
        args.epoch_size
        if args.epoch_size is not None
        else args.batch_size * args.num_batches
    )
    if args.index_cache_dir is not None:
        config["index_cache_dir"] = args.index_cache_dir
    return config


def main() -> int:
    args = parse_args()
    config_path = resolve_config_path(args.config, args.tmpdir)
    base_config = yaml.safe_load(config_path.read_text())

    ts = time.strftime("%Y%m%d_%H%M%S")
    spool_dir = Path(args.spool_root) / f"bench_{ts}_pid{os.getpid()}"
    spool_dir.parent.mkdir(parents=True, exist_ok=True)

    config = make_benchmark_config(base_config, args, spool_dir)
    num_samples = args.batch_size * args.num_batches

    print("[spool-bench] config=", config_path, flush=True)
    print("[spool-bench] spool_dir=", spool_dir, flush=True)
    print("[spool-bench] index_cache_dir=", config.get("index_cache_dir"), flush=True)
    print("[spool-bench] builder_workers=", config["builder_workers"], flush=True)
    print("[spool-bench] prefetch_depth=", config["prefetch_depth"], flush=True)
    print("[spool-bench] batch_size=", args.batch_size, flush=True)
    print("[spool-bench] num_batches=", args.num_batches, flush=True)
    print("[spool-bench] compute_ms=", args.compute_ms, flush=True)
    print("[spool-bench] num_samples=", num_samples, flush=True)

    t0 = time.perf_counter()
    dataset = create_training_dataset(config, split="train", rank=0, world_size=1)
    init_s = time.perf_counter() - t0
    print(f"[spool-bench] dataset_init={init_s:.3f}s", flush=True)

    sample_waits_s: list[float] = []
    batch_waits_s: list[float] = []
    dataset_counts: Counter[str] = Counter()
    sequence_examples: list[str] = []
    compute_sleep_total_s = 0.0

    overall_start = time.perf_counter()
    try:
        for batch_idx in range(args.num_batches):
            batch_sample_waits_s: list[float] = []
            batch_start = time.perf_counter()

            for sample_offset in range(args.batch_size):
                sample_index = batch_idx * args.batch_size + sample_offset
                t_sample = time.perf_counter()
                sample = dataset[sample_index]
                wait_s = time.perf_counter() - t_sample

                sample_waits_s.append(wait_s)
                batch_sample_waits_s.append(wait_s)
                dataset_counts[sample.dataset_name] += 1
                if len(sequence_examples) < 10:
                    sequence_examples.append(f"{sample.dataset_name}:{sample.sequence_name}")

            batch_wait_s = time.perf_counter() - batch_start
            batch_waits_s.append(batch_wait_s)

            ready_count, ready_bytes = spool_state(dataset)
            mean_sample_ms = statistics.mean(batch_sample_waits_s) * 1000.0
            max_sample_ms = max(batch_sample_waits_s) * 1000.0
            print(
                f"[spool-bench] batch={batch_idx:02d} "
                f"wait={batch_wait_s:.3f}s "
                f"sample_mean={mean_sample_ms:.0f}ms "
                f"sample_max={max_sample_ms:.0f}ms "
                f"ready_files={ready_count} "
                f"ready_bytes={format_bytes(ready_bytes)}",
                flush=True,
            )

            if args.compute_ms > 0:
                sleep_s = args.compute_ms / 1000.0
                time.sleep(sleep_s)
                compute_sleep_total_s += sleep_s
                ready_count, ready_bytes = spool_state(dataset)
                print(
                    f"[spool-bench] batch={batch_idx:02d} post_compute "
                    f"ready_files={ready_count} ready_bytes={format_bytes(ready_bytes)}",
                    flush=True,
                )
    finally:
        if hasattr(dataset, "cleanup"):
            dataset.cleanup()
        if not args.keep_spool:
            shutil.rmtree(spool_dir, ignore_errors=True)

    total_wall_s = time.perf_counter() - overall_start
    active_consume_s = max(total_wall_s - compute_sleep_total_s, 1e-9)
    summary = {
        "config": str(config_path),
        "spool_dir": str(spool_dir),
        "builder_workers": int(config["builder_workers"]),
        "prefetch_depth": int(config["prefetch_depth"]),
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "num_samples": num_samples,
        "compute_ms": args.compute_ms,
        "dataset_init_s": init_s,
        "total_wall_s": total_wall_s,
        "active_consume_s": active_consume_s,
        "samples_per_s_active": num_samples / active_consume_s,
        "batches_per_s_active": args.num_batches / active_consume_s,
        "dataset_counts": dict(dataset_counts),
        "sequence_examples": sequence_examples,
    }
    summary.update(summarize_waits_ms(sample_waits_s, "sample_wait_ms"))
    summary.update(summarize_waits_ms(batch_waits_s, "batch_wait_ms"))
    if len(batch_waits_s) > 1:
        steady_batch_waits_s = batch_waits_s[1:]
        steady_sample_waits_s = sample_waits_s[args.batch_size:]
        summary["steady_batches"] = len(steady_batch_waits_s)
        summary.update(summarize_waits_ms(steady_batch_waits_s, "steady_batch_wait_ms"))
        summary.update(summarize_waits_ms(steady_sample_waits_s, "steady_sample_wait_ms"))
    else:
        summary["steady_batches"] = 0
        summary.update(summarize_waits_ms([], "steady_batch_wait_ms"))
        summary.update(summarize_waits_ms([], "steady_sample_wait_ms"))

    print("\n[spool-bench] summary", flush=True)
    for key in (
        "dataset_init_s",
        "total_wall_s",
        "active_consume_s",
        "samples_per_s_active",
        "batches_per_s_active",
        "sample_wait_ms_mean",
        "sample_wait_ms_p50",
        "sample_wait_ms_p95",
        "sample_wait_ms_max",
        "batch_wait_ms_mean",
        "batch_wait_ms_p50",
        "batch_wait_ms_p95",
        "batch_wait_ms_max",
        "steady_batches",
        "steady_sample_wait_ms_mean",
        "steady_sample_wait_ms_p50",
        "steady_sample_wait_ms_p95",
        "steady_sample_wait_ms_max",
        "steady_batch_wait_ms_mean",
        "steady_batch_wait_ms_p50",
        "steady_batch_wait_ms_p95",
        "steady_batch_wait_ms_max",
    ):
        value = summary[key]
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}", flush=True)
        else:
            print(f"  {key}: {value}", flush=True)
    print(f"  dataset_counts: {summary['dataset_counts']}", flush=True)
    print(f"  sequence_examples: {summary['sequence_examples']}", flush=True)

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(f"[spool-bench] report_json={report_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
