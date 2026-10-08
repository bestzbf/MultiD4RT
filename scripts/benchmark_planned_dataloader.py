#!/usr/bin/env python3
"""CPU-only planned data loading benchmark.

This mirrors the training data path:

  PlannedMixtureDataset -> SequentialSampler -> DataLoader(num_workers=0)
  -> d4rt_collate_fn -> optional BatchPrefetchIterator

It intentionally does not create a model, initialize CUDA, move tensors to a
device, run loss/backward, or save checkpoints.  When launched with torchrun it
uses RANK/WORLD_SIZE from the environment so each process consumes the same rank
partition that training would consume.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import Sampler

from datasets.collate import d4rt_collate_fn
from datasets.factory import create_training_dataset
from train_mixture import (
    BatchPrefetchIterator,
    format_batch_sample_details,
    format_dataset_counts,
    infer_next_planned_index,
    set_dataset_epoch,
    summarize_spool_ready,
)


class OffsetSequentialSampler(Sampler[int]):
    def __init__(self, data_source: Any, start_index: int) -> None:
        self.data_source = data_source
        self.start_index = max(0, int(start_index))

    def __iter__(self):
        return iter(range(self.start_index, len(self.data_source)))

    def __len__(self) -> int:
        return max(0, len(self.data_source) - self.start_index)


class TimingCollateFn:
    def __init__(self, rank: int, interval: int) -> None:
        self.rank = rank
        self.interval = max(1, int(interval))
        self.count = 0
        self.times_s: list[float] = []

    def __call__(self, batch):
        t0 = time.perf_counter()
        result = d4rt_collate_fn(batch)
        elapsed = time.perf_counter() - t0
        self.count += 1
        self.times_s.append(elapsed)
        if self.count <= 3 or self.count % self.interval == 0:
            print(
                f"[DataloadProbe rank{self.rank}] collate "
                f"batch={self.count} samples={len(batch)} "
                f"time={elapsed * 1000:.1f}ms",
                flush=True,
            )
        return result


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * (pct / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def compact_spool_stats(stats: dict[str, Any]) -> str:
    if not stats:
        return "spool=<none>"
    return (
        f"spool_ready={stats.get('ready_count')} "
        f"spool_min={stats.get('ready_min')} "
        f"spool_max={stats.get('ready_max')} "
        f"focus_next={stats.get('focus_index')} "
        f"contiguous={stats.get('contiguous_from_focus')} "
        f"next_ready={stats.get('next_ready')}"
    )


def summarize_waits(values_s: list[float], threshold_s: float) -> dict[str, Any]:
    slow = [v for v in values_s if v >= threshold_s]
    return {
        "count": len(values_s),
        "avg_ms": statistics.mean(values_s) * 1000.0 if values_s else 0.0,
        "p50_ms": percentile(values_s, 50.0) * 1000.0,
        "p90_ms": percentile(values_s, 90.0) * 1000.0,
        "p95_ms": percentile(values_s, 95.0) * 1000.0,
        "p99_ms": percentile(values_s, 99.0) * 1000.0,
        "max_ms": max(values_s) * 1000.0 if values_s else 0.0,
        "slow_batches": len(slow),
        "slow_ratio": len(slow) / max(1, len(values_s)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--planned-mode", action="store_true")
    parser.add_argument("--builder-workers", type=int, default=8)
    parser.add_argument("--prefetch-depth", type=int, default=256)
    parser.add_argument("--batch-prefetch-depth", type=int, default=4)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--simulate-compute-ms", type=float, default=0.0)
    parser.add_argument("--startup-sleep-s", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--data-wait-threshold-s", type=float, default=2.0)
    parser.add_argument("--detail-max-samples", type=int, default=8)
    parser.add_argument("--profile-collate", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--report-dir", type=str, default="")
    parser.add_argument("--spool-root", type=str, default="")
    parser.add_argument("--aggregate-timeout-s", type=float, default=120.0)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if args.start_batch > 0:
        start_index = max(0, int(args.start_batch) * int(args.batch_size))
        os.environ["D4RT_PLANNED_START_INDEX"] = str(start_index)

    if args.planned_mode:
        config["planned_mode"] = True
        config["builder_workers"] = args.builder_workers
        config["prefetch_depth"] = args.prefetch_depth
        config["planned_initial_epoch"] = int(args.start_epoch)
        config["planned_start_immediately"] = True
    if args.spool_root:
        config["spool_dir"] = str(Path(args.spool_root) / f"rank{rank}")

    if rank == 0:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        print(
            "[DataloadProbe] CPU-only data path benchmark\n"
            f"  config={args.config}\n"
            f"  world_size={world_size} batch_size={args.batch_size} "
            f"planned={bool(config.get('planned_mode'))}\n"
            f"  builder_workers={config.get('builder_workers')} "
            f"prefetch_depth={config.get('prefetch_depth')} "
            f"batch_prefetch_depth={args.batch_prefetch_depth}\n"
            f"  spool_root={args.spool_root or '<default>'}\n"
            f"  batches={args.batches} warmup={args.warmup_batches} "
            f"start_epoch={args.start_epoch} start_batch={args.start_batch} "
            f"simulate_compute_ms={args.simulate_compute_ms:.0f}\n"
            f"  CUDA_VISIBLE_DEVICES={cuda_visible!r}",
            flush=True,
        )

    dataset = None
    train_iter = None
    summary: dict[str, Any] | None = None
    try:
        t_init0 = time.perf_counter()
        dataset = create_training_dataset(
            config,
            split="train",
            rank=rank,
            world_size=world_size,
        )
        init_s = time.perf_counter() - t_init0

        start_index = max(0, int(args.start_batch) * int(args.batch_size))
        if args.planned_mode or config.get("planned_mode"):
            sampler = (
                OffsetSequentialSampler(dataset, start_index)
                if start_index > 0
                else SequentialSampler(dataset)
            )
            num_workers = 0
        else:
            sampler = RandomSampler(dataset)
            num_workers = args.num_workers

        collate_fn = (
            TimingCollateFn(rank=local_rank, interval=args.log_interval)
            if args.profile_collate
            else d4rt_collate_fn
        )
        drop_last = len(dataset) >= args.batch_size
        loader_kwargs: dict[str, Any] = {
            "batch_size": args.batch_size,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "sampler": sampler,
            "pin_memory": args.pin_memory,
            "persistent_workers": num_workers > 0,
            "drop_last": drop_last,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader = DataLoader(dataset, **loader_kwargs)

        print(
            f"[DataloadProbe rank{local_rank}] dataset_init={init_s:.2f}s "
            f"len={len(dataset)} loader_batches={len(loader)} "
            f"num_workers={num_workers} pin_memory={args.pin_memory}",
            flush=True,
        )

        set_dataset_epoch(dataset, args.start_epoch)
        if args.startup_sleep_s > 0:
            if local_rank == 0:
                print(
                    f"[DataloadProbe] startup_sleep={args.startup_sleep_s:.1f}s "
                    "to let builders prefill spool",
                    flush=True,
                )
            time.sleep(args.startup_sleep_s)

        waits_s: list[float] = []
        measured_waits_s: list[float] = []
        batch_sizes: list[int] = []
        total_batches = 0
        total_samples = 0
        first_batch_time_s: float | None = None
        bench_start = time.perf_counter()
        stop = False

        for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
            set_dataset_epoch(dataset, epoch)
            base_iter = iter(loader)
            train_iter = (
                BatchPrefetchIterator(loader, depth=args.batch_prefetch_depth)
                if args.batch_prefetch_depth > 0
                else base_iter
            )
            t_data_start = time.perf_counter()
            for batch_idx, batch in enumerate(train_iter):
                absolute_batch_idx = batch_idx + max(0, int(args.start_batch))
                t_now = time.perf_counter()
                data_s = t_now - t_data_start
                if first_batch_time_s is None:
                    first_batch_time_s = t_now - bench_start

                waits_s.append(data_s)
                if total_batches >= args.warmup_batches:
                    measured_waits_s.append(data_s)

                sample_count = len(batch.get("dataset_names") or [])
                total_samples += sample_count
                batch_sizes.append(sample_count)
                next_planned_index = infer_next_planned_index(batch)
                spool_stats = summarize_spool_ready(dataset, focus_index=next_planned_index)
                prefetch_stats = (
                    train_iter.last_stats()
                    if isinstance(train_iter, BatchPrefetchIterator)
                    else {}
                )
                dataset_counts = format_dataset_counts(batch.get("dataset_names"))

                should_log = (
                    total_batches < 3
                    or total_batches % max(1, args.log_interval) == 0
                    or data_s >= args.data_wait_threshold_s
                )
                if should_log:
                    print(
                        f"[DataloadProbe rank{local_rank}] "
                        f"epoch={epoch} batch={absolute_batch_idx} global_batch={total_batches} "
                        f"data_wait={data_s * 1000:.0f}ms "
                        f"datasets={dataset_counts} "
                        f"prefetch_wait={prefetch_stats.get('get_wait_s', 0.0) * 1000:.0f}ms "
                        f"q_before={prefetch_stats.get('qsize_before')} "
                        f"q_after={prefetch_stats.get('qsize_after')} "
                        f"{compact_spool_stats(spool_stats)}",
                        flush=True,
                    )
                if data_s >= args.data_wait_threshold_s:
                    for detail_line in format_batch_sample_details(
                        batch,
                        max_samples=args.detail_max_samples,
                    ):
                        print(
                            f"[DataloadProbe rank{local_rank}] {detail_line}",
                            flush=True,
                        )

                total_batches += 1
                if total_batches >= args.batches:
                    stop = True
                    break

                if args.simulate_compute_ms > 0:
                    time.sleep(args.simulate_compute_ms / 1000.0)
                t_data_start = time.perf_counter()

            if isinstance(train_iter, BatchPrefetchIterator):
                train_iter.close()
            train_iter = None
            if stop:
                break

        elapsed_s = time.perf_counter() - bench_start
        measured = summarize_waits(measured_waits_s, args.data_wait_threshold_s)
        all_waits = summarize_waits(waits_s, args.data_wait_threshold_s)
        keep_up = measured["slow_batches"] == 0
        summary = {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "total_batches": total_batches,
            "total_samples": total_samples,
            "avg_batch_size": statistics.mean(batch_sizes) if batch_sizes else 0.0,
            "elapsed_s": elapsed_s,
            "first_batch_s": first_batch_time_s,
            "simulate_compute_ms": args.simulate_compute_ms,
            "data_wait_threshold_s": args.data_wait_threshold_s,
            "warmup_batches": args.warmup_batches,
            "measured": measured,
            "all": all_waits,
            "keep_up": keep_up,
            "throughput_batches_per_s": total_batches / elapsed_s if elapsed_s > 0 else 0.0,
            "throughput_samples_per_s": total_samples / elapsed_s if elapsed_s > 0 else 0.0,
        }
        print(
            f"[DataloadProbe rank{local_rank}] SUMMARY "
            f"batches={total_batches} samples={total_samples} "
            f"elapsed={elapsed_s:.1f}s first_batch={first_batch_time_s or 0:.1f}s "
            f"measured_avg={measured['avg_ms']:.0f}ms "
            f"p95={measured['p95_ms']:.0f}ms max={measured['max_ms']:.0f}ms "
            f"slow={measured['slow_batches']}/{measured['count']} "
            f"keep_up={keep_up}",
            flush=True,
        )

    finally:
        if isinstance(train_iter, BatchPrefetchIterator):
            train_iter.close()
        if dataset is not None and hasattr(dataset, "cleanup"):
            dataset.cleanup()

    if summary is not None and args.report_dir:
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"rank{rank}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

        if rank == 0:
            deadline = time.time() + args.aggregate_timeout_s
            expected = [report_dir / f"rank{i}.json" for i in range(world_size)]
            while time.time() < deadline and not all(p.exists() for p in expected):
                time.sleep(1.0)
            summaries = []
            for p in expected:
                if p.exists():
                    summaries.append(json.loads(p.read_text(encoding="utf-8")))
            if summaries:
                worst_p95 = max(item["measured"]["p95_ms"] for item in summaries)
                worst_max = max(item["measured"]["max_ms"] for item in summaries)
                total_slow = sum(item["measured"]["slow_batches"] for item in summaries)
                total_count = sum(item["measured"]["count"] for item in summaries)
                keep_up_all = all(item["keep_up"] for item in summaries)
                print(
                    "[DataloadProbe] AGGREGATE "
                    f"ranks={len(summaries)}/{world_size} "
                    f"worst_p95={worst_p95:.0f}ms worst_max={worst_max:.0f}ms "
                    f"slow={total_slow}/{total_count} "
                    f"keep_up_all={keep_up_all} "
                    f"report_dir={report_dir}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
