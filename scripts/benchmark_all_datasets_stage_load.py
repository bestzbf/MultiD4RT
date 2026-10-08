#!/usr/bin/env python3
"""Benchmark stage+load time for each dataset by random sampling."""
from __future__ import annotations

import contextlib
import random
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets.registry import create_adapter
from datasets.sample_stage import SampleLocalStager, SampleStageConfig

import glob, os
configs = sorted(glob.glob('/data1/zbf/d4rt_tmp/mixture_5datasets_cos_planned.*.yaml'), key=os.path.getmtime, reverse=True)
CONFIG = configs[0] if configs else 'configs/mixture_5datasets_cos_planned.yaml'
print(f'[bench] using config: {CONFIG}')
STAGE_ROOT = "/data1/zbf/d4rt_stage_load_bench"
PASSWD_FILE = "/etc/passwd-s3fs-data_cos"
CLIP_LEN = 8  # small clip for speed
SAMPLES_PER_DATASET = 3
SDK_WORKERS = 16
SEED = 42


def main() -> None:
    config_path = CONFIG or 'configs/mixture_5datasets_cos_planned.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    cache_dir = cfg.get("index_cache_dir")
    datasets = {d["name"]: d for d in cfg.get("datasets", [])}

    results = {}
    for name, ds_cfg in datasets.items():
        print(f"\n{'='*50}")
        print(f"[{name}] initializing adapter...")
        t0 = time.perf_counter()
        try:
            kwargs = dict(ds_cfg.get("adapter_kwargs", {}))
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            adapter = create_adapter(
                name=name,
                root=ds_cfg["root"],
                split=ds_cfg.get("split", "train"),
                **kwargs,
            )
        except Exception as e:
            print(f"[{name}] adapter init failed: {e}")
            continue
        init_s = time.perf_counter() - t0
        print(f"[{name}] adapter init={init_s:.2f}s, sequences={len(adapter.list_sequences())}")

        stager = SampleLocalStager(SampleStageConfig(
            backend="cos_sdk",
            stage_root=STAGE_ROOT,
            sdk_workers=SDK_WORKERS,
            passwd_file=PASSWD_FILE,
            enabled_datasets=(name,),
        ))

        rng = random.Random(SEED)
        seqs = adapter.list_sequences()
        sample_seqs = rng.sample(seqs, min(SAMPLES_PER_DATASET, len(seqs)))

        stage_times, load_times = [], []
        for seq in sample_seqs:
            n_frames = adapter.get_num_frames(seq)
            if n_frames < CLIP_LEN:
                continue
            start = rng.randint(0, n_frames - CLIP_LEN)
            frame_indices = list(range(start, start + CLIP_LEN))

            # stage
            t_stage0 = time.perf_counter()
            try:
                ctx = stager.stage_sample(adapter, seq, frame_indices, sample_tag="bench")
                with ctx as staged_adapter:
                    t_stage = time.perf_counter() - t_stage0
                    # load
                    t_load0 = time.perf_counter()
                    staged_adapter.load_clip(seq, frame_indices)
                    t_load = time.perf_counter() - t_load0
                stage_times.append(t_stage)
                load_times.append(t_load)
                print(f"  seq={seq[:40]} stage={t_stage*1000:.0f}ms load={t_load*1000:.0f}ms")
            except Exception as e:
                print(f"  seq={seq[:40]} ERROR: {e}")

        if stage_times:
            results[name] = {
                "stage_ms_mean": sum(stage_times) / len(stage_times) * 1000,
                "load_ms_mean": sum(load_times) / len(load_times) * 1000,
            }

    print(f"\n{'='*50}")
    print("SUMMARY (mean over samples):")
    print(f"{'dataset':<20} {'stage_ms':>10} {'load_ms':>10} {'total_ms':>10}")
    for name, r in results.items():
        total = r['stage_ms_mean'] + r['load_ms_mean']
        print(f"{name:<20} {r['stage_ms_mean']:>10.0f} {r['load_ms_mean']:>10.0f} {total:>10.0f}")


if __name__ == "__main__":
    main()
