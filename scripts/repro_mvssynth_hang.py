#!/usr/bin/env python3
"""Reproduce mvssynth builder hang using real forkserver workers, no GPU/model."""
from __future__ import annotations
import ctypes
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from datasets.registry import create_adapter
from datasets.sample_builder import SampleBuilder, start_builder_process
from datasets.sample_spool import SampleSpool
from datasets.planning import SampleSpec
from datasets.query_builder import D4RTQueryBuilder
from datasets.transforms import GeometryTransformPipeline

EFFECTIVE_CONFIG = sorted(
    Path("/data1/zbf/d4rt_tmp").glob("mixture_5datasets_cos_planned.*.yaml"),
    key=os.path.getmtime, reverse=True,
)
CONFIG_PATH = str(EFFECTIVE_CONFIG[0]) if EFFECTIVE_CONFIG else "configs/mixture_5datasets_cos_planned.yaml"

MVSSYNTH_ROOT = "/data2/d4rt/datasets/MVS-Synth/GTAV_1080"
SPOOL_DIR = "/tmp/test_mvssynth_repro_spool"
NUM_WORKERS = 4
NUM_SAMPLES = 8
CLIP_LEN = 48
TIMEOUT_PER_SAMPLE = 120


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    shutil.rmtree(SPOOL_DIR, ignore_errors=True)

    print(f"config: {CONFIG_PATH}")
    print(f"mvssynth root: {MVSSYNTH_ROOT}")
    print(f"workers: {NUM_WORKERS}, samples: {NUM_SAMPLES}, clip_len: {CLIP_LEN}")

    # Build adapter
    adapter = create_adapter(
        name="mvssynth", root=MVSSYNTH_ROOT, split="train",
        load_precomputed=True,
        cache_dir=cfg.get("index_cache_dir"),
    )
    print(f"sequences: {len(adapter.list_sequences())}")

    # Build sample_stage_config exactly like training
    sample_stage_config = None
    if cfg.get("sample_stage_backend"):
        sample_stage_config = {
            "backend": cfg["sample_stage_backend"],
            "stage_root": cfg.get("sample_stage_root", "/data1/zbf/d4rt_sample_stage"),
            "sdk_workers": cfg.get("sample_stage_sdk_workers", 32),
            "cache_max_bytes": cfg.get("sample_stage_cache_max_bytes", 100 * 1024**3),
            "passwd_file": cfg.get("sample_stage_passwd_file", "/etc/passwd-s3fs-data_cos"),
            "enabled_datasets": cfg.get("sample_stage_datasets", []),
            "scene_prefetch_datasets": cfg.get("sample_stage_scene_prefetch_datasets", []),
            "mount_root": cfg.get("sample_stage_mount_root", "/data_cos"),
            "bucket": cfg.get("sample_stage_bucket", "hd-ai-data-1251882982"),
            "region": cfg.get("sample_stage_region", "ap-beijing"),
        }
    print(f"sample_stage enabled_datasets: {sample_stage_config.get('enabled_datasets') if sample_stage_config else None}")

    transform = GeometryTransformPipeline(
        img_size=cfg.get("img_size", 256), use_augs=True,
    )
    qb = D4RTQueryBuilder(num_queries=cfg.get("num_queries", 2048))
    spool = SampleSpool(spool_dir=SPOOL_DIR, rank=0)

    # Queues + generation counter
    ctx = mp.get_context("forkserver")
    input_queue = ctx.Queue()
    output_queue = ctx.Queue()
    current_gen = ctx.Value(ctypes.c_int64, 0)

    # Start workers
    processes = []
    for i in range(NUM_WORKERS):
        p = start_builder_process(
            builder_id=i,
            adapters=[adapter],
            transform=transform,
            query_builder=qb,
            spool=SampleSpool(spool_dir=SPOOL_DIR, rank=0, cleanup_on_init=False),
            input_queue=input_queue,
            output_queue=output_queue,
            clip_len=CLIP_LEN,
            current_generation=current_gen,
            sample_stage_config=sample_stage_config,
        )
        processes.append(p)
    print(f"started {len(processes)} builder workers")

    # Generate specs
    rng = random.Random(42)
    seqs = adapter.list_sequences()
    for i in range(NUM_SAMPLES):
        seq = rng.choice(seqs)
        n = adapter.get_num_frames(seq)
        start = rng.randint(0, max(0, n - CLIP_LEN))
        fi = list(range(start, min(start + CLIP_LEN, n)))
        spec = SampleSpec(
            dataset_idx=0,
            sequence_name=seq,
            frame_indices=fi,
            local_index=i,
            global_index=i,
            generation=0,
            rng_state=rng.getstate(),
        )
        input_queue.put(spec)
        print(f"  enqueued sample {i}: seq={seq} frames={len(fi)}[{fi[0]}..{fi[-1]}]")

    # Wait for results
    done = 0
    t0 = time.perf_counter()
    while done < NUM_SAMPLES:
        elapsed = time.perf_counter() - t0
        if elapsed > TIMEOUT_PER_SAMPLE * NUM_SAMPLES / NUM_WORKERS + 60:
            print(f"GLOBAL TIMEOUT after {elapsed:.0f}s, got {done}/{NUM_SAMPLES}")
            break
        try:
            idx, ok = output_queue.get(timeout=10)
            done += 1
            print(f"  sample {idx} {'OK' if ok else 'FAIL'} ({done}/{NUM_SAMPLES}) elapsed={time.perf_counter()-t0:.1f}s")
        except Exception:
            alive = sum(1 for p in processes if p.is_alive())
            print(f"  waiting... done={done}/{NUM_SAMPLES} alive={alive}/{NUM_WORKERS} elapsed={elapsed:.0f}s")

    total = time.perf_counter() - t0
    print(f"\nDone: {done}/{NUM_SAMPLES} in {total:.1f}s ({total/max(done,1):.1f}s/sample)")

    # Cleanup
    for _ in processes:
        input_queue.put(None)
    for p in processes:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    shutil.rmtree(SPOOL_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
