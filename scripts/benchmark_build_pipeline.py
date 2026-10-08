#!/usr/bin/env python3
"""Benchmark full _build_sample pipeline (stage+load+transform+query) per dataset."""
from __future__ import annotations
import argparse, glob, os, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from datasets.factory import create_training_dataset
from datasets.transforms import GeometryTransformPipeline
from datasets.query_builder import D4RTQueryBuilder
from datasets.sample_stage import SampleLocalStager, SampleStageConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--samples-per-dataset", type=int, default=2)
    p.add_argument("--clip-len", type=int, default=48)
    p.add_argument("--stage-root", default="/data1/zbf/d4rt_stage_bench_pipeline")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    if args.config is None:
        configs = sorted(glob.glob("/data1/zbf/d4rt_tmp/mixture_5datasets_cos_planned.*.yaml"),
                         key=os.path.getmtime, reverse=True)
        args.config = configs[0] if configs else "configs/mixture_5datasets_cos_planned.yaml"
    print(f"config: {args.config}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from datasets.registry import create_adapter
    cache_dir = cfg.get("index_cache_dir")
    stage_cfg = cfg.get("sample_stage_datasets", [])
    passwd = cfg.get("sample_stage_passwd_file", "/etc/passwd-s3fs-data_cos")

    stager = SampleLocalStager(SampleStageConfig(
        backend="cos_sdk",
        stage_root=args.stage_root,
        sdk_workers=cfg.get("sample_stage_sdk_workers", 16),
        passwd_file=passwd,
        enabled_datasets=tuple(stage_cfg),
        mount_root=cfg.get("sample_stage_mount_root", "/data_cos"),
        bucket=cfg.get("sample_stage_bucket", "hd-ai-data-1251882982"),
        region=cfg.get("sample_stage_region", "ap-beijing"),
    ))

    transform = GeometryTransformPipeline(img_size=cfg.get("img_size", 256), use_augs=True)
    qb = D4RTQueryBuilder(num_queries=cfg.get("num_queries", 2048))

    print(f"\n{'dataset':<20} {'stage_ms':>9} {'load_ms':>9} {'xfm_ms':>9} {'qb_ms':>9} {'total_ms':>9}")
    print("-" * 70)

    for ds in cfg["datasets"]:
        name = ds["name"]
        kwargs = dict(ds.get("adapter_kwargs", {}))
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        try:
            adapter = create_adapter(name=name, root=ds["root"],
                                     split=ds.get("split", "train"), **kwargs)
        except Exception as e:
            print(f"{name:<20} ERROR init: {e}")
            continue

        rng = random.Random(args.seed)
        seqs = adapter.list_sequences()
        samples = rng.sample(seqs, min(args.samples_per_dataset, len(seqs)))

        ts_list, tl_list, tx_list, tq_list = [], [], [], []
        for seq in samples:
            n = adapter.get_num_frames(seq)
            fi = list(range(0, min(args.clip_len, n)))
            py_rng = random.Random(args.seed)
            try:
                t0 = time.perf_counter()
                with stager.stage_sample(adapter, seq, fi, "bench") as a:
                    ts = time.perf_counter() - t0
                    t1 = time.perf_counter()
                    clip = a.load_clip(seq, fi)
                    tl = time.perf_counter() - t1
                t2 = time.perf_counter()
                result = transform(clip, rng=py_rng)
                tx = time.perf_counter() - t2
                t3 = time.perf_counter()
                qb(result, py_rng=py_rng)
                tq = time.perf_counter() - t3
                ts_list.append(ts); tl_list.append(tl)
                tx_list.append(tx); tq_list.append(tq)
            except Exception as e:
                print(f"{name:<20} ERROR seq={seq}: {e}")

        if ts_list:
            def ms(lst): return sum(lst) / len(lst) * 1000
            total = ms(ts_list) + ms(tl_list) + ms(tx_list) + ms(tq_list)
            print(f"{name:<20} {ms(ts_list):>9.0f} {ms(tl_list):>9.0f} "
                  f"{ms(tx_list):>9.0f} {ms(tq_list):>9.0f} {total:>9.0f}")


if __name__ == "__main__":
    main()
