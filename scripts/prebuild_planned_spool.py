#!/usr/bin/env python3
"""Prebuild planned QuerySample spool files without consuming them.

This is a CPU-only companion to planned-mode training.  Launch it with
torchrun using the same config/rank count/batch size as training.  Each rank
builds its own local planned sample indices into ``spool_root/rank{rank}``.
Training can then start with ``D4RT_SPOOL_CLEANUP_ON_INIT=0`` and consume the
ready spool files before falling back to live builders.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from datasets.factory import create_training_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--spool-root", required=True)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--builder-workers", type=int, default=18)
    parser.add_argument("--prefetch-depth", type=int, default=0)
    parser.add_argument("--max-spool-gb", type=float, default=100.0)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--poll-s", type=float, default=2.0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--report-dir", default="")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def _rank_env() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def _count_ready(spool: Any, generation: int, start: int, end: int) -> tuple[int, int, int | None, int | None]:
    ready = [idx for idx in spool.list_ready_indices(generation) if start <= idx < end]
    errors = [idx for idx in spool.list_error_indices(generation) if start <= idx < end]
    return (
        len(ready),
        len(errors),
        min(ready) if ready else None,
        max(ready) if ready else None,
    )


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    rank, world_size = _rank_env()
    start_index = max(0, int(args.start_batch) * int(args.batch_size))
    target_samples = max(0, int(args.batches) * int(args.batch_size))
    prefetch_depth = int(args.prefetch_depth) if args.prefetch_depth > 0 else target_samples

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["D4RT_PLANNED_START_INDEX"] = str(start_index)
    os.environ.setdefault("D4RT_BUILDER_START_METHOD", "fork")
    os.environ.setdefault("D4RT_SKIP_READY_ENQUEUE", "1")
    os.environ.setdefault("D4RT_PRESERVE_SPOOL_ON_CLEANUP", "1")
    os.environ["D4RT_SPOOL_CLEANUP_ON_INIT"] = "0" if args.reuse_existing else "1"

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    spool_dir = Path(args.spool_root) / f"rank{rank}"
    config["planned_mode"] = True
    config["planned_start_immediately"] = True
    config["planned_initial_epoch"] = int(args.start_epoch)
    config["builder_workers"] = int(args.builder_workers)
    config["prefetch_depth"] = max(1, prefetch_depth)
    config["max_spool_bytes"] = int(float(args.max_spool_gb) * 1024**3)
    config["spool_dir"] = str(spool_dir)

    if rank == 0:
        print(
            "[PrebuildSpool] start "
            f"world_size={world_size} batch_size={args.batch_size} "
            f"batches={args.batches} start_batch={args.start_batch} "
            f"builder_workers={args.builder_workers} prefetch_depth={config['prefetch_depth']} "
            f"spool_root={args.spool_root}",
            flush=True,
        )

    dataset = create_training_dataset(
        config,
        split="train",
        rank=rank,
        world_size=world_size,
    )
    end_index = min(len(dataset), start_index + target_samples)
    target_samples = max(0, end_index - start_index)
    deadline = time.time() + float(args.timeout_s)
    last_log = 0.0
    requeues = 0

    try:
        while True:
            ready_count, error_count, ready_min, ready_max = _count_ready(
                dataset.spool,
                dataset._generation,
                start_index,
                end_index,
            )
            if ready_count >= target_samples:
                elapsed = float(args.timeout_s) - max(0.0, deadline - time.time())
                print(
                    f"[PrebuildSpool rank{rank}] ready={ready_count}/{target_samples} "
                    f"errors={error_count} elapsed={elapsed:.1f}s "
                    f"range=[{start_index},{end_index}) spool={spool_dir}",
                    flush=True,
                )
                break

            for error_idx in dataset.spool.list_error_indices(dataset._generation):
                if start_index <= error_idx < end_index:
                    dataset._handle_failed_index(error_idx, max_requeue=3)
                    requeues += 1

            now = time.time()
            if now >= deadline:
                raise TimeoutError(
                    f"rank{rank} prebuild timed out: ready={ready_count}/{target_samples} "
                    f"errors={error_count} range=[{start_index},{end_index}) spool={spool_dir}"
                )
            if now - last_log >= max(1.0, float(args.poll_s) * 5.0):
                print(
                    f"[PrebuildSpool rank{rank}] ready={ready_count}/{target_samples} "
                    f"errors={error_count} min={ready_min} max={ready_max} "
                    f"requeues={requeues}",
                    flush=True,
                )
                last_log = now
            time.sleep(max(0.1, float(args.poll_s)))
    finally:
        dataset._stop_pipeline()

    report = {
        "rank": rank,
        "world_size": world_size,
        "spool_dir": str(spool_dir),
        "start_index": start_index,
        "end_index": end_index,
        "target_samples": target_samples,
        "ready_samples": _count_ready(dataset.spool, dataset._generation, start_index, end_index)[0],
        "requeues": requeues,
    }
    if args.report_dir:
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"rank{rank}.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
