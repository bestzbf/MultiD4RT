#!/usr/bin/env python3
"""Benchmark QuerySample spool consumption without touching active training data.

The script creates synthetic QuerySample pickle files in an isolated directory
and measures:
  - serial sample pickle.load + collate
  - parallel sample pickle.load within one batch + collate
  - batch prefetch overlap with simulated compute

Use /dev/shm as --root to avoid competing with NVMe/COS training I/O.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import statistics
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Any

import sys
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets.collate import d4rt_collate_fn
from datasets.query_builder import QuerySample, _build_empty_targets


def _fmt_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{n}B"


def make_sample(args: argparse.Namespace, idx: int) -> QuerySample:
    T = args.num_frames
    S = args.img_size
    Q = args.num_queries
    H = args.highres_h
    W = args.highres_w

    if args.video_dtype == "float32":
        video = torch.empty((T, 3, S, S), dtype=torch.float32)
    elif args.video_dtype == "uint8":
        video = torch.empty((T, 3, S, S), dtype=torch.uint8)
    else:
        raise ValueError(args.video_dtype)
    depths = torch.empty((T, 1, S, S), dtype=torch.float32) if args.include_depths else None
    normals = torch.empty((T, 3, S, S), dtype=torch.float32) if args.include_normals else None

    if args.highres_dtype == "none":
        highres_video = None
    elif args.highres_dtype == "float32":
        highres_video = torch.empty((T, 3, H, W), dtype=torch.float32)
    elif args.highres_dtype == "uint8":
        highres_video = torch.empty((T, 3, H, W), dtype=torch.uint8)
    else:
        raise ValueError(args.highres_dtype)

    local_patches = (
        torch.empty((Q, 3, args.patch_size, args.patch_size), dtype=torch.float32)
        if args.include_local_patches
        else None
    )

    return QuerySample(
        video=video,
        highres_video=highres_video,
        depths=depths,
        normals=normals,
        coords=torch.empty((Q, 2), dtype=torch.float32),
        t_src=torch.empty((Q,), dtype=torch.long),
        t_tgt=torch.empty((Q,), dtype=torch.long),
        t_cam=torch.empty((Q,), dtype=torch.long),
        intrinsics=torch.empty((T, 3, 3), dtype=torch.float32),
        extrinsics=torch.empty((T, 4, 4), dtype=torch.float32),
        targets=_build_empty_targets(Q),
        local_patches=local_patches,
        transform_metadata={
            "canonical_space": torch.tensor(0, dtype=torch.long),
            "original_hw": torch.empty((2,), dtype=torch.float32),
            "crop_offset_xy": torch.empty((2,), dtype=torch.float32),
            "crop_size_hw": torch.tensor([float(H), float(W)], dtype=torch.float32),
            "resized_hw": torch.tensor([float(S), float(S)], dtype=torch.float32),
        },
        aspect_ratio=torch.tensor([float(W) / max(float(H), 1.0)], dtype=torch.float32),
        dataset_name="synthetic",
        sequence_name=f"seq_{idx}",
        metadata={},
    )


def write_samples(args: argparse.Namespace, root: Path) -> list[Path]:
    total = args.batch_size * args.num_batches
    paths: list[Path] = []
    for i in range(total):
        sample = make_sample(args, i)
        path = root / f"sample_{i:05d}.pkl"
        t0 = time.perf_counter()
        with open(path, "wb") as f:
            pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
        dt = time.perf_counter() - t0
        if args.verbose:
            print(f"[write] {path.name} {_fmt_bytes(path.stat().st_size)} {dt:.3f}s", flush=True)
        paths.append(path)
        del sample
    return paths


def load_one(path: Path) -> QuerySample:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_one_split(path: Path) -> tuple[QuerySample, float, float]:
    t_read0 = time.perf_counter()
    with open(path, "rb") as f:
        payload = f.read()
    read_s = time.perf_counter() - t_read0

    t_pickle0 = time.perf_counter()
    sample = pickle.loads(payload)
    unpickle_s = time.perf_counter() - t_pickle0
    return sample, read_s, unpickle_s


def bench_split_serial(paths: list[Path], batch_size: int) -> dict[str, Any]:
    batch_times: list[float] = []
    read_times: list[float] = []
    unpickle_times: list[float] = []
    collate_times: list[float] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        t0 = time.perf_counter()
        samples = []
        read_s = 0.0
        unpickle_s = 0.0
        for path in batch_paths:
            sample, one_read_s, one_unpickle_s = load_one_split(path)
            samples.append(sample)
            read_s += one_read_s
            unpickle_s += one_unpickle_s
        t_collate0 = time.perf_counter()
        batch = d4rt_collate_fn(samples)
        t_collate = time.perf_counter() - t_collate0
        batch_times.append(time.perf_counter() - t0)
        read_times.append(read_s)
        unpickle_times.append(unpickle_s)
        collate_times.append(t_collate)
        del samples, batch
    return {
        "name": "split_serial_read_pickle_collate",
        "batches": len(batch_times),
        "batch_mean_s": statistics.mean(batch_times),
        "read_mean_s": statistics.mean(read_times),
        "unpickle_mean_s": statistics.mean(unpickle_times),
        "collate_mean_s": statistics.mean(collate_times),
        "batch_values_s": batch_times,
        "read_values_s": read_times,
        "unpickle_values_s": unpickle_times,
        "collate_values_s": collate_times,
    }


def bench_serial(paths: list[Path], batch_size: int) -> dict[str, Any]:
    batch_times: list[float] = []
    load_times: list[float] = []
    collate_times: list[float] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        t0 = time.perf_counter()
        samples = []
        t_load0 = time.perf_counter()
        for path in batch_paths:
            samples.append(load_one(path))
        t_load = time.perf_counter() - t_load0
        t_collate0 = time.perf_counter()
        batch = d4rt_collate_fn(samples)
        t_collate = time.perf_counter() - t_collate0
        batch_times.append(time.perf_counter() - t0)
        load_times.append(t_load)
        collate_times.append(t_collate)
        del samples, batch
    return summarize("serial", batch_times, load_times, collate_times)


def bench_parallel(paths: list[Path], batch_size: int, workers: int) -> dict[str, Any]:
    batch_times: list[float] = []
    load_times: list[float] = []
    collate_times: list[float] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            t0 = time.perf_counter()
            t_load0 = time.perf_counter()
            samples = list(executor.map(load_one, batch_paths))
            t_load = time.perf_counter() - t_load0
            t_collate0 = time.perf_counter()
            batch = d4rt_collate_fn(samples)
            t_collate = time.perf_counter() - t_collate0
            batch_times.append(time.perf_counter() - t0)
            load_times.append(t_load)
            collate_times.append(t_collate)
            del samples, batch
    return summarize(f"parallel_load_{workers}", batch_times, load_times, collate_times)


def bench_prefetch(
    paths: list[Path],
    batch_size: int,
    workers: int,
    queue_depth: int,
    compute_ms: float,
) -> dict[str, Any]:
    batches = [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]
    q: Queue[Any] = Queue(maxsize=queue_depth)
    stop = object()

    def producer() -> None:
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for batch_paths in batches:
                    samples = list(executor.map(load_one, batch_paths))
                    batch = d4rt_collate_fn(samples)
                    del samples
                    q.put(batch)
        finally:
            q.put(stop)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    wait_times: list[float] = []
    count = 0
    while True:
        t0 = time.perf_counter()
        item = q.get()
        wait_times.append(time.perf_counter() - t0)
        if item is stop:
            break
        count += 1
        del item
        if compute_ms > 0:
            time.sleep(compute_ms / 1000.0)

    thread.join(timeout=10.0)
    return {
        "name": f"prefetch_parallel_{workers}_q{queue_depth}_compute{compute_ms:g}ms",
        "batches": count,
        "wait_mean_s": statistics.mean(wait_times[:-1]) if count else 0.0,
        "wait_max_s": max(wait_times[:-1]) if count else 0.0,
        "wait_values_s": wait_times[:-1],
    }


def summarize(name: str, batch_times: list[float], load_times: list[float], collate_times: list[float]) -> dict[str, Any]:
    return {
        "name": name,
        "batches": len(batch_times),
        "batch_mean_s": statistics.mean(batch_times),
        "batch_min_s": min(batch_times),
        "batch_max_s": max(batch_times),
        "load_mean_s": statistics.mean(load_times),
        "collate_mean_s": statistics.mean(collate_times),
        "batch_values_s": batch_times,
        "load_values_s": load_times,
        "collate_values_s": collate_times,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/dev/shm/d4rt_spool_consume_bench")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--num-frames", type=int, default=48)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--num-queries", type=int, default=2048)
    parser.add_argument("--video-dtype", choices=["float32", "uint8"], default="float32")
    parser.add_argument("--highres-h", type=int, default=576)
    parser.add_argument("--highres-w", type=int, default=576)
    parser.add_argument("--highres-dtype", choices=["float32", "uint8", "none"], default="uint8")
    parser.add_argument("--include-depths", action="store_true", default=True)
    parser.add_argument("--no-depths", dest="include_depths", action="store_false")
    parser.add_argument("--include-normals", action="store_true")
    parser.add_argument("--include-local-patches", action="store_true")
    parser.add_argument("--patch-size", type=int, default=9)
    parser.add_argument("--parallel-workers", type=int, default=5)
    parser.add_argument("--queue-depth", type=int, default=2)
    parser.add_argument("--compute-ms", type=float, default=1200.0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--report-json", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    root_parent = Path(args.root)
    root_parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="run_", dir=root_parent))

    try:
        print(f"[bench] work_dir={work_dir}", flush=True)
        print(
            f"[bench] batch_size={args.batch_size} batches={args.num_batches} "
            f"video={args.video_dtype} highres={args.highres_dtype}:{args.highres_h}x{args.highres_w} "
            f"depths={args.include_depths} normals={args.include_normals} "
            f"local_patches={args.include_local_patches}",
            flush=True,
        )
        t0 = time.perf_counter()
        paths = write_samples(args, work_dir)
        write_s = time.perf_counter() - t0
        sizes = [p.stat().st_size for p in paths]
        print(
            f"[bench] wrote {len(paths)} files total={_fmt_bytes(sum(sizes))} "
            f"avg={_fmt_bytes(int(sum(sizes)/len(sizes)))} write_s={write_s:.3f}",
            flush=True,
        )

        results = {
            "config": vars(args),
            "file_count": len(paths),
            "file_total_bytes": sum(sizes),
            "file_avg_bytes": int(sum(sizes) / len(sizes)),
            "write_s": write_s,
            "split_serial": bench_split_serial(paths, args.batch_size),
            "serial": bench_serial(paths, args.batch_size),
            "parallel": bench_parallel(paths, args.batch_size, args.parallel_workers),
            "prefetch": bench_prefetch(
                paths,
                args.batch_size,
                args.parallel_workers,
                args.queue_depth,
                args.compute_ms,
            ),
        }

        for key in ("split_serial", "serial", "parallel", "prefetch"):
            item = results[key]
            print(f"[bench] {item['name']}", flush=True)
            for stat_key, value in item.items():
                if stat_key in {
                    "name",
                    "batch_values_s",
                    "load_values_s",
                    "read_values_s",
                    "unpickle_values_s",
                    "collate_values_s",
                    "wait_values_s",
                }:
                    continue
                if isinstance(value, float):
                    print(f"  {stat_key}: {value:.3f}", flush=True)
                else:
                    print(f"  {stat_key}: {value}", flush=True)

        if args.report_json:
            report_path = Path(args.report_json)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(results, indent=2) + "\n")
            print(f"[bench] report_json={report_path}", flush=True)
    finally:
        if not args.keep:
            shutil.rmtree(work_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
