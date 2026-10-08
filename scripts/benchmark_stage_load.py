#!/usr/bin/env python3
"""Benchmark COS sample staging and adapter load time for one D4RT sample.

Example:
    python scripts/benchmark_stage_load.py \
      --dataset co3dv2 \
      --seq stopsign/427_59942_115928 \
      --frames '9..56' \
      --root /data_cos/hdu_datasets/Co3Dv2 \
      --repeat 2 --clear-cache
"""

from __future__ import annotations

import argparse
import os
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

from datasets.query_builder import D4RTQueryBuilder
from datasets.registry import create_adapter
from datasets.sample_stage import SampleLocalStager, SampleStageConfig
from datasets.transforms import GeometryTransformPipeline


def _parse_frames(raw: str) -> list[int]:
    text = raw.strip()
    if "[" in text and "]" in text:
        text = text[text.index("[") + 1 : text.rindex("]")]
    text = text.replace(" ", "")
    for sep in ("..", ":", "-"):
        if sep in text and "," not in text:
            lo_s, hi_s = text.split(sep, 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"Bad frame range: {raw!r}")
            return list(range(lo, hi + 1))
    frames = [int(part) for part in text.split(",") if part]
    if not frames:
        raise ValueError(f"No frames parsed from {raw!r}")
    return frames


def _load_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _dataset_config(config: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    if config.get("mode") == "mixture":
        for item in config["datasets"]:
            if item["name"] == dataset_name:
                return item
        raise KeyError(f"Dataset {dataset_name!r} not found in mixture config")
    if config.get("name") == dataset_name:
        return config
    raise KeyError(f"Config dataset is {config.get('name')!r}, not {dataset_name!r}")


def _build_adapter(config: dict[str, Any], dataset_name: str, root: str | None, split: str):
    try:
        ds_cfg = _dataset_config(config, dataset_name)
    except KeyError:
        if root is None:
            raise
        ds_cfg = {"name": dataset_name, "root": root, "adapter_kwargs": {}}
    adapter_kwargs = dict(ds_cfg.get("adapter_kwargs", {}))
    if dataset_name == "dynamic_replica":
        if "DYNAMIC_REPLICA_SKIP_DEPTH_WHEN_TRACKS" in os.environ:
            adapter_kwargs["skip_depth_when_tracks"] = os.environ[
                "DYNAMIC_REPLICA_SKIP_DEPTH_WHEN_TRACKS"
            ].strip().lower() in {"1", "true", "yes", "on"}
        if "D4RT_DYNAMIC_REPLICA_IO_WORKERS" in os.environ:
            adapter_kwargs["io_workers"] = int(os.environ["D4RT_DYNAMIC_REPLICA_IO_WORKERS"])
    if dataset_name == "scannetpp":
        env_map = {
            "SCANNETPP_H5_CHUNK_CACHE_DIR": "precomputed_h5_chunk_cache_dir",
            "SCANNETPP_H5_CHUNK_CACHE_MIN_BYTES": "precomputed_h5_chunk_cache_min_bytes",
            "SCANNETPP_PRECOMPUTED_COS_TIMEOUT_S": "precomputed_cos_timeout_s",
            "SCANNETPP_PRECOMPUTED_COS_RANGE_RETRIES": "precomputed_cos_range_retries",
            "SCANNETPP_PRECOMPUTED_COS_RANGE_WORKERS": "precomputed_cos_range_workers",
            "SCANNETPP_PRECOMPUTED_COS_RANGE_MERGE_GAP_BYTES": "precomputed_cos_range_merge_gap_bytes",
            "SCANNETPP_PRECOMPUTED_COS_RANGE_MAX_SPAN_BYTES": "precomputed_cos_range_max_span_bytes",
        }
        for env_name, key in env_map.items():
            raw = os.environ.get(env_name, "").strip()
            if raw:
                adapter_kwargs[key] = raw if key.endswith("_dir") else int(float(raw))
        raw = os.environ.get("SCANNETPP_H5_CHUNK_CACHE_MAX_GB", "").strip()
        if raw:
            adapter_kwargs["precomputed_h5_chunk_cache_max_bytes"] = int(float(raw) * 1024**3)
    cache_dir = config.get("index_cache_dir")
    index_workers = config.get("index_workers")
    kwargs: dict[str, Any] = {
        "name": dataset_name,
        "root": root or ds_cfg["root"],
        "split": ds_cfg.get("split", split),
        **adapter_kwargs,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if index_workers is not None:
        kwargs["index_workers"] = index_workers
    return create_adapter(**kwargs)


def _build_stager(config: dict[str, Any], args: argparse.Namespace) -> SampleLocalStager:
    return SampleLocalStager(
        SampleStageConfig(
            backend="cos_sdk",
            stage_root=args.stage_root,
            sdk_workers=args.sdk_workers,
            cache_max_bytes=int(args.cache_max_gb * 1024**3),
            cache_low_watermark_ratio=config.get("sample_stage_cache_low_watermark_ratio", 0.9),
            cache_touch_interval_s=config.get("sample_stage_cache_touch_interval_s", 30.0),
            cache_scan_interval_s=config.get("sample_stage_cache_scan_interval_s", 30.0),
            window_radius=args.window_radius,
            mount_root=args.mount_root or config.get("sample_stage_mount_root", "/data_cos"),
            bucket=args.bucket or config.get("sample_stage_bucket", "hd-ai-data-1251882982"),
            region=args.region or config.get("sample_stage_region", "ap-beijing"),
            passwd_file=args.passwd_file
            or config.get("sample_stage_passwd_file", "/etc/passwd-s3fs-data_cos"),
            enabled_datasets=(args.dataset,),
            scene_prefetch_datasets=(),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark COS stage/load/transform/query for one sample."
    )
    parser.add_argument("--config", default="configs/mixture_5datasets_cos_planned.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--frames", required=True, help="Examples: 9..56, 9:56, 48[9..56], or comma list")
    parser.add_argument("--root", default=None, help="Override dataset root, e.g. /data_cos/hdu_datasets/Co3Dv2")
    parser.add_argument("--split", default="train")
    parser.add_argument("--stage-root", default="/data1/zbf/d4rt_stage_load_bench")
    parser.add_argument("--sdk-workers", type=int, default=16)
    parser.add_argument("--window-radius", type=int, default=0)
    parser.add_argument("--cache-max-gb", type=float, default=20.0)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--no-transform-query", action="store_true")
    parser.add_argument("--num-queries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mount-root", default=None)
    parser.add_argument("--bucket", default=None)
    parser.add_argument("--region", default=None)
    parser.add_argument("--passwd-file", default=None)
    args = parser.parse_args()

    frames = _parse_frames(args.frames)
    if args.clear_cache:
        shutil.rmtree(args.stage_root, ignore_errors=True)

    config = _load_config(args.config)
    adapter = _build_adapter(config, args.dataset, args.root, args.split)
    stager = _build_stager(config, args)

    transform = None
    query_builder = None
    if not args.no_transform_query:
        transform = GeometryTransformPipeline(
            img_size=config.get("img_size", 256),
            use_augs=config.get("use_augs", True),
            keep_cropped_images=config.get("keep_cropped_images", True),
            color_aug_after_resize=config.get("color_aug_after_resize", False),
            max_track_points=config.get("max_track_points"),
        )
        query_builder = D4RTQueryBuilder(
            num_queries=args.num_queries or config.get("num_queries", 2048),
            boundary_ratio=config.get("boundary_ratio", 0.3),
            t_tgt_eq_t_cam_ratio=config.get("t_tgt_eq_t_cam_ratio", 0.4),
            use_motion_boundaries=config.get("use_motion_boundaries", True),
            precompute_patches=config.get("precompute_patches", False),
            precompute_from_highres=config.get("precompute_from_highres", False),
            return_highres_video=config.get("return_highres_video"),
            allow_track_fallback=config.get("allow_track_fallback", True),
            store_video_uint8=config.get("store_video_uint8", False),
            store_auxiliary_tensors=config.get("store_auxiliary_tensors", True),
            motion_boundary_on_resized=config.get("motion_boundary_on_resized", True),
        )

    manifest = stager._build_manifest(adapter, args.seq, frames) if stager.supports(adapter) else []
    print(
        f"[bench] dataset={args.dataset} root={adapter.root} seq={args.seq} "
        f"frames={len(frames)}[{min(frames)}..{max(frames)}] "
        f"stage_root={args.stage_root} sdk_workers={args.sdk_workers} "
        f"manifest_files={len(manifest)} supports_stage={stager.supports(adapter)}",
        flush=True,
    )

    for run_idx in range(max(1, args.repeat)):
        rng = random.Random(args.seed + run_idx)
        sample_tag = f"bench_{args.dataset}_{run_idx}".replace("/", "_")
        original_root = Path(adapter.root)
        t0 = time.perf_counter()
        with stager.stage_sample(adapter, args.seq, frames, sample_tag=sample_tag) as staged_adapter:
            stage_s = time.perf_counter() - t0
            staged = Path(staged_adapter.root) != original_root

            t_load0 = time.perf_counter()
            clip = staged_adapter.load_clip(args.seq, frames)
            load_s = time.perf_counter() - t_load0

            transform_s = 0.0
            query_s = 0.0
            has_valid_3d = None
            sample_shape = None
            if transform is not None and query_builder is not None:
                t_transform0 = time.perf_counter()
                result = transform(clip, rng=rng)
                transform_s = time.perf_counter() - t_transform0
                t_query0 = time.perf_counter()
                sample = query_builder(result, py_rng=rng)
                query_s = time.perf_counter() - t_query0
                has_valid_3d = float(sample.targets["mask_3d"].float().mean())
                sample_shape = tuple(sample.video.shape)

        total_s = time.perf_counter() - t0
        print(
            f"[StageLoadBench] run={run_idx} total={total_s:.3f}s "
            f"stage={stage_s:.3f}s load={load_s:.3f}s "
            f"transform={transform_s:.3f}s query={query_s:.3f}s "
            f"dataset={args.dataset} seq={args.seq} "
            f"frames={len(frames)}[{min(frames)}..{max(frames)}] "
            f"staged={staged} has_tracks={clip.metadata.get('has_tracks')} "
            f"mask3d={has_valid_3d if has_valid_3d is not None else 'NA'} "
            f"video={sample_shape if sample_shape is not None else 'NA'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
