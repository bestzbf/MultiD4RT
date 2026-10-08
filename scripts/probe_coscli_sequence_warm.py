#!/usr/bin/env python3
"""Probe planned block warming with coscli sequence/prefix sync.

This is an offline feasibility probe for a rolling warm cache design:

1. Generate the same planned samples as training for one or more ranks.
2. Group raw COS inputs by dataset/sequence prefix.
3. Use `coscli sync` to materialize those prefixes into the SampleLocalStager
   shared raw cache layout.
4. Verify the exact per-sample manifests are present.

It does not build QuerySample bundles, does not import CUDA, and should use a
separate stage root.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.factory import create_training_dataset
from datasets.planning import SamplePlanner
from datasets.adapters.scannetpp import (
    _extract_video_frames_by_timestamps,
    _load_frame_index,
    _read_rgb_jpg,
    rgb_cache_frame_path,
    write_rgb_cache_frame,
)
from datasets.sample_stage import SampleLocalStager, SampleStageConfig


@dataclass(frozen=True)
class PrefixSync:
    dataset: str
    sequence: str
    cos_prefix: str
    local_dir: Path
    include_pattern: str = ""
    expected_paths: tuple[Path, ...] = ()


@dataclass
class ScanNetPPRgbDecodeSpec:
    sequence: str
    scene_dir: Path
    target_hw: tuple[int, int]
    frame_indices: set[int]
    source: str = "video"


@dataclass
class ScanNetPPDepthSpec:
    sequence: str
    scene_dir: Path
    full_indices: set[int]


@dataclass
class ScanNetPPH5Spec:
    adapter: Any
    sequence: str
    frame_indices: set[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage-root", required=True)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--block-batches", type=int, default=10)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--compute-s-per-batch", type=float, default=2.0)
    parser.add_argument("--ranks", default="")
    parser.add_argument("--datasets", default="dynamic_replica,co3dv2,scannetpp")
    parser.add_argument(
        "--exact-warm-datasets",
        default=os.getenv("D4RT_ROLLING_WARM_EXACT_DATASETS", "dynamic_replica"),
    )
    parser.add_argument(
        "--sdk-exact-datasets",
        default=os.getenv("D4RT_ROLLING_WARM_SDK_EXACT_DATASETS", ""),
        help=(
            "Warm planned files for these datasets via the COS SDK exact "
            "manifest instead of coscli prefix/include sync."
        ),
    )
    parser.add_argument("--coscli-routines", type=int, default=64)
    parser.add_argument("--coscli-thread-num", type=int, default=4)
    parser.add_argument("--prefix-workers", type=int, default=4)
    parser.add_argument("--stage-sdk-workers", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--clean-stage-root", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-update", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--include-scannetpp-h5", action="store_true")
    parser.add_argument("--no-overlap-decode", action="store_true")
    parser.add_argument(
        "--manifest-mode",
        choices=("prefix", "exact"),
        default=os.getenv("D4RT_ROLLING_WARM_MANIFEST_MODE", "prefix"),
        help=(
            "'prefix' warms sequence-level COS prefixes plus planned ScanNet++ "
            "RGB cache frames. 'exact' additionally builds every per-sample "
            "file manifest, which is useful for debugging but much slower."
        ),
    )
    parser.add_argument(
        "--scannetpp-rgb-mode",
        choices=("auto", "frames", "video", "cache", "frame_cache"),
        default=(
            os.getenv("SCANNETPP_RGB_READ_MODE", "cache").strip().lower()
            if os.getenv("SCANNETPP_RGB_READ_MODE", "cache").strip().lower()
            in {"auto", "frames", "video", "cache", "frame_cache"}
            else "cache"
        ),
        help=(
            "ScanNet++ RGB source to warm. 'cache' syncs rgb.mkv, then decodes "
            "planned frames into local target-size uint8 frame cache. "
            "'frame_cache' syncs bucketed JPEG frames, then decodes those JPEGs "
            "into the same target-size uint8 frame cache."
        ),
    )
    parser.add_argument("--scannetpp-decode-workers", type=int, default=4)
    parser.add_argument(
        "--scannetpp-frame-workers",
        type=int,
        default=int(os.getenv("ROLLING_WARM_SCANNETPP_FRAME_WORKERS", "4")),
        help=(
            "Per-scene worker count for decoding ScanNet++ frame-cache JPEGs "
            "into target-size RGB cache files. Only used with "
            "--scannetpp-rgb-mode frame_cache."
        ),
    )
    parser.add_argument("--no-scannetpp-rgb-decode", action="store_true")
    parser.add_argument("--scannetpp-depth-workers", type=int, default=4)
    parser.add_argument("--no-scannetpp-depth-warm", action="store_true")
    parser.add_argument("--scannetpp-h5-workers", type=int, default=4)
    parser.add_argument("--no-scannetpp-h5-warm", action="store_true")
    parser.add_argument(
        "--scannetpp-h5-cache-dir",
        default=os.getenv("SCANNETPP_H5_CHUNK_CACHE_DIR", ""),
    )
    parser.add_argument(
        "--scannetpp-h5-cache-max-gb",
        type=float,
        default=float(os.getenv("SCANNETPP_H5_CHUNK_CACHE_MAX_GB", "80")),
    )
    parser.add_argument(
        "--scannetpp-h5-cache-min-bytes",
        type=int,
        default=int(os.getenv("SCANNETPP_H5_CHUNK_CACHE_MIN_BYTES", "4096")),
    )
    parser.add_argument(
        "--scannetpp-h5-cache-low-watermark",
        type=float,
        default=float(os.getenv("SCANNETPP_H5_CHUNK_CACHE_LOW_WATERMARK_RATIO", "0.9")),
    )
    parser.add_argument("--ready-dir", default="")
    parser.add_argument("--progress-dir", default="")
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--lookahead-blocks", type=int, default=2)
    parser.add_argument(
        "--block-workers",
        type=int,
        default=int(os.getenv("ROLLING_WARM_BLOCK_WORKERS", "1")),
        help="Number of future rolling-warm blocks to materialize concurrently.",
    )
    parser.add_argument("--poll-s", type=float, default=1.0)
    parser.add_argument("--max-ready-blocks", type=int, default=0)
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--min-progress-ranks", type=int, default=0)
    return parser.parse_args()


def _safe_clean_stage_root(path: Path) -> None:
    text = str(path)
    if "warm" not in text and "probe" not in text:
        raise SystemExit(f"Refusing to clean non-probe stage root: {path}")
    if path.exists():
        shutil.rmtree(path)


def _rank_list(raw: str, world_size: int) -> list[int]:
    if not raw.strip():
        return list(range(world_size))
    ranks: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        rank = int(part)
        if rank < 0 or rank >= world_size:
            raise SystemExit(f"rank out of range: {rank}")
        ranks.append(rank)
    return sorted(set(ranks))


def _load_config(
    path: str,
    stage_root: Path,
    *,
    scannetpp_h5_cache_dir: str = "",
    scannetpp_h5_cache_min_bytes: int = 4096,
    scannetpp_h5_cache_max_gb: float = 80.0,
    scannetpp_h5_cache_low_watermark: float = 0.9,
) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    config = dict(config)
    config["planned_mode"] = True
    config["planned_start_immediately"] = False
    config["sample_stage_root"] = str(stage_root)
    config["sample_stage_eviction_mode"] = "disabled"
    for env_name, config_key in (
        ("DATASET_LOCALITY_SIZE", "dataset_locality_size"),
        ("SEQUENCE_LOCALITY_SIZE", "sequence_locality_size"),
        ("FRAME_LOCALITY_RADIUS", "frame_locality_radius"),
    ):
        value = os.getenv(env_name, "").strip()
        if value:
            config[config_key] = int(value)
    config.setdefault(
        "sample_stage_datasets",
        ["pointodyssey", "kubric", "dynamic_replica", "co3dv2", "scannetpp"],
    )
    if scannetpp_h5_cache_dir:
        for item in config.get("datasets", []):
            if item.get("name") != "scannetpp":
                continue
            kwargs = item.setdefault("adapter_kwargs", {})
            kwargs["precomputed_h5_chunk_cache_dir"] = str(scannetpp_h5_cache_dir)
            kwargs["precomputed_h5_chunk_cache_min_bytes"] = int(
                scannetpp_h5_cache_min_bytes
            )
            kwargs["precomputed_h5_chunk_cache_max_bytes"] = int(
                float(scannetpp_h5_cache_max_gb) * 1024**3
            )
            kwargs["precomputed_h5_chunk_cache_low_watermark_ratio"] = float(
                scannetpp_h5_cache_low_watermark
            )
            range_workers = os.getenv("SCANNETPP_PRECOMPUTED_COS_RANGE_WORKERS", "").strip()
            if range_workers:
                kwargs["precomputed_cos_range_workers"] = int(range_workers)
            range_retries = os.getenv("SCANNETPP_PRECOMPUTED_COS_RANGE_RETRIES", "").strip()
            if range_retries:
                kwargs["precomputed_cos_range_retries"] = int(range_retries)
            range_timeout = os.getenv("SCANNETPP_PRECOMPUTED_COS_TIMEOUT_S", "").strip()
            if range_timeout:
                kwargs["precomputed_cos_timeout_s"] = int(float(range_timeout))
            merge_gap = os.getenv("SCANNETPP_PRECOMPUTED_COS_RANGE_MERGE_GAP_BYTES", "").strip()
            if merge_gap:
                kwargs["precomputed_cos_range_merge_gap_bytes"] = int(merge_gap)
            max_span = os.getenv("SCANNETPP_PRECOMPUTED_COS_RANGE_MAX_SPAN_BYTES", "").strip()
            if max_span:
                kwargs["precomputed_cos_range_max_span_bytes"] = int(max_span)
    return config


def _stager_from_config(config: dict[str, Any]) -> SampleLocalStager:
    return SampleLocalStager(
        SampleStageConfig.from_dict(
            {
                "backend": config.get("sample_stage_backend", "cos_sdk"),
                "stage_root": config["sample_stage_root"],
                "sdk_workers": config.get("sample_stage_sdk_workers", 8),
                "request_timeout_s": config.get("sample_stage_request_timeout_s", 20.0),
                "request_retries": config.get("sample_stage_request_retries", 1),
                "cache_max_bytes": config.get("sample_stage_cache_max_bytes", 400 * 1024**3),
                "cache_low_watermark_ratio": config.get("sample_stage_cache_low_watermark_ratio", 0.9),
                "cache_touch_interval_s": 0.0,
                "cache_scan_interval_s": 3600.0,
                "eviction_mode": "disabled",
                "window_radius": config.get("sample_stage_window_radius", 0),
                "mount_root": config.get("sample_stage_mount_root", "/data_cos"),
                "extra_mount_roots": config.get("sample_stage_extra_mount_roots", ()),
                "bucket": config.get("sample_stage_bucket", "hd-ai-data-1251882982"),
                "region": config.get("sample_stage_region", "ap-beijing"),
                "passwd_file": config.get("sample_stage_passwd_file", "/etc/passwd-s3fs-data_cos"),
                "enabled_datasets": config.get("sample_stage_datasets", ()),
                "scene_prefetch_datasets": (),
            }
        )
    )


def _cache_path(stager: SampleLocalStager, path: Path) -> Path:
    return stager.cache_data_root / Path(stager._to_cos_key(path))


def _cache_stats(stager: SampleLocalStager, paths: list[Path]) -> tuple[int, int]:
    present = 0
    bytes_total = 0
    for path in paths:
        cache_path = _cache_path(stager, path)
        try:
            stat = cache_path.stat()
        except OSError:
            continue
        present += 1
        bytes_total += stat.st_size
    return present, bytes_total


def _load_scannetpp_frame_index_for_collect(
    stager: SampleLocalStager,
    scene_dir: Path,
) -> dict[str, Any]:
    rel_scene = Path(stager._to_cos_key(scene_dir))
    local_scene = stager.cache_data_root / rel_scene
    local_index = local_scene / "iphone" / "frame_index.pkl"
    if not local_index.is_file():
        try:
            stager._ensure_cached(scene_dir / "iphone" / "frame_index.pkl")
        except Exception:
            pass
    if local_index.is_file():
        return _load_frame_index(local_scene)
    return _load_frame_index(scene_dir)


def _prefix_for_spec(
    stager: SampleLocalStager,
    adapter: Any,
    dataset: str,
    sequence: str,
    frame_indices: list[int] | None = None,
    *,
    include_scannetpp_h5: bool,
    scannetpp_rgb_mode: str,
) -> list[PrefixSync]:
    root = Path(getattr(adapter, "root"))
    mount = stager.mount_root
    syncs: list[PrefixSync] = []
    if dataset == "co3dv2":
        record = adapter._get_record(sequence)
        seq_dir = root / record.category / record.sequence_name
        rel = Path(stager._to_cos_key(seq_dir))
        syncs.append(PrefixSync(dataset, sequence, rel.as_posix() + "/", stager.cache_data_root / rel))
    elif dataset == "dynamic_replica":
        record = adapter._get_record(sequence)
        seq_dir = adapter.split_root / record.sequence_name
        if (
            bool(getattr(adapter, "load_trajectories", False))
            and bool(getattr(record, "has_trajectories", False))
        ):
            if bool(getattr(adapter, "prefer_trajectory_npz", False)):
                image_dir = seq_dir / "images"
                image_rel = Path(stager._to_cos_key(image_dir))
                syncs.append(
                    PrefixSync(
                        dataset,
                        sequence,
                        image_rel.as_posix() + "/",
                        stager.cache_data_root / image_rel,
                        "*.png",
                    )
                )
                traj_dir = seq_dir / "trajectories"
                traj_rel = Path(stager._to_cos_key(traj_dir))
                syncs.append(
                    PrefixSync(
                        dataset,
                        sequence,
                        traj_rel.as_posix() + "/",
                        stager.cache_data_root / traj_rel,
                        "*.npz",
                    )
                )
                return syncs
            seq_dir = seq_dir / "trajectories"
        rel = Path(stager._to_cos_key(seq_dir))
        syncs.append(PrefixSync(dataset, sequence, rel.as_posix() + "/", stager.cache_data_root / rel))
    elif dataset == "scannetpp":
        scene_dir = adapter.data_root / sequence / "iphone"
        if scannetpp_rgb_mode in {"video", "cache"}:
            rel_file = Path(stager._to_cos_key(scene_dir / "rgb.mkv"))
            syncs.append(
                PrefixSync(
                    dataset,
                    sequence,
                    rel_file.as_posix(),
                    stager.cache_data_root / rel_file,
                )
            )
        else:
            rel = Path(stager._to_cos_key(scene_dir / "frames"))
            local_dir = stager.cache_data_root / rel
            patterns = _frame_bucket_patterns(frame_indices or [], "jpg")
            if patterns:
                for pattern, expected_indices in patterns:
                    syncs.append(
                        PrefixSync(
                            dataset,
                            sequence,
                            rel.as_posix() + "/",
                            local_dir,
                            pattern,
                            tuple(local_dir / f"{idx:06d}.jpg" for idx in expected_indices),
                        )
                    )
            else:
                syncs.append(
                    PrefixSync(
                        dataset,
                        sequence,
                        rel.as_posix() + "/",
                        stager.cache_data_root / rel,
                    )
                )
        # frame_index.pkl is materialized during block collection; depth warm
        # materializes depth_chunk_index.pkl. Avoid a concurrent coscli writer
        # racing those readers on the same cache path.
        # precomputed.h5 is warmed by _warm_scannetpp_h5_chunks via COS Range
        # into the dedicated H5 chunk cache. Syncing the full scene H5 here
        # costs hundreds of MB per scene and is not used by staged training.
        _ = include_scannetpp_h5
    else:
        return []
    # Keep mypy/linters happy when mount is unused; stager._to_cos_key validates.
    _ = mount
    return syncs


def _frame_bucket_patterns(
    frame_indices: list[int],
    suffix: str,
) -> list[tuple[str, tuple[int, ...]]]:
    bucket_indices: dict[int, set[int]] = defaultdict(set)
    for idx in frame_indices:
        idx_int = max(0, int(idx))
        bucket_indices[idx_int // 100].add(idx_int)
    suffix = suffix.lstrip(".")
    return [
        (
            f"^.*{bucket:04d}[0-9][0-9]\\.{suffix}$",
            tuple(sorted(indices)),
        )
        for bucket, indices in sorted(bucket_indices.items())
    ]


def _exact_name_pattern_chunks(
    names: list[str],
    *,
    max_chars: int = 12000,
) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_chars = len("^.*()$")
    for name in sorted(set(names)):
        escaped = re.escape(name)
        added = len(escaped) + (1 if current else 0)
        if current and current_chars + added > max_chars:
            chunks.append(current)
            current = []
            current_chars = len("^.*()$")
        current.append(name)
        current_chars += added
    if current:
        chunks.append(current)
    return chunks


def _exact_name_include_pattern(names: list[str]) -> str:
    escaped = [re.escape(name) for name in sorted(set(names))]
    if not escaped:
        return r"$.^"
    if len(escaped) == 1:
        return f"^.*{escaped[0]}$"
    return f"^.*({'|'.join(escaped)})$"


def _prefix_syncs_for_exact_paths(
    stager: SampleLocalStager,
    *,
    dataset: str,
    sequence: str,
    paths: list[Path],
) -> list[PrefixSync]:
    """Warm exact planned files with directory-level coscli include filters."""
    grouped: dict[Path, dict[str, Path]] = defaultdict(dict)
    for path in paths:
        rel = Path(stager._to_cos_key(Path(path)))
        grouped[rel.parent][rel.name] = stager.cache_data_root / rel

    syncs: list[PrefixSync] = []
    for parent_rel, by_name in sorted(grouped.items(), key=lambda item: item[0].as_posix()):
        names = sorted(by_name)
        for chunk in _exact_name_pattern_chunks(names):
            syncs.append(
                PrefixSync(
                    dataset=dataset,
                    sequence=sequence,
                    cos_prefix=parent_rel.as_posix().rstrip("/") + "/",
                    local_dir=stager.cache_data_root / parent_rel,
                    include_pattern=_exact_name_include_pattern(chunk),
                    expected_paths=tuple(by_name[name] for name in chunk),
                )
            )
    return syncs


def _expected_path_ready(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _missing_expected_paths(paths: tuple[Path, ...], *, limit: int = 20) -> list[Path]:
    missing: list[Path] = []
    for path in paths:
        if not _expected_path_ready(path):
            missing.append(path)
            if len(missing) >= limit:
                break
    return missing


def _sync_marker_name(sync: PrefixSync) -> str:
    marker_name = ".d4rt_coscli_sync_complete"
    if sync.include_pattern:
        import hashlib

        digest = hashlib.sha1(sync.include_pattern.encode("utf-8")).hexdigest()[:12]
        marker_name = f".d4rt_coscli_sync_complete.{digest}"
    return marker_name


def _prefix_sync_key(sync: PrefixSync) -> tuple[str, str, str, str, str]:
    return (
        sync.dataset,
        sync.sequence,
        sync.cos_prefix,
        sync.local_dir.as_posix(),
        sync.include_pattern,
    )


def _add_prefix_sync(
    unique_prefixes: dict[tuple[str, str, str, str, str], PrefixSync],
    sync: PrefixSync,
) -> None:
    key = _prefix_sync_key(sync)
    prev = unique_prefixes.get(key)
    if prev is None:
        unique_prefixes[key] = sync
        return
    if not sync.expected_paths:
        return
    if not prev.expected_paths:
        unique_prefixes[key] = sync
        return
    merged: dict[str, Path] = {path.as_posix(): path for path in prev.expected_paths}
    for path in sync.expected_paths:
        merged.setdefault(path.as_posix(), path)
    if len(merged) != len(prev.expected_paths):
        unique_prefixes[key] = PrefixSync(
            dataset=prev.dataset,
            sequence=prev.sequence,
            cos_prefix=prev.cos_prefix,
            local_dir=prev.local_dir,
            include_pattern=prev.include_pattern,
            expected_paths=tuple(merged.values()),
        )


def _init_rank_datasets(
    config: dict[str, Any],
    *,
    ranks: list[int],
    world_size: int,
) -> list[tuple[int, Any]]:
    if not ranks:
        return []
    t0 = time.perf_counter()
    base_rank = ranks[0]
    base = create_training_dataset(
        config,
        split="train",
        rank=base_rank,
        world_size=world_size,
    )
    epoch = int(config.get("planned_initial_epoch", 0))
    count_per_rank = math.ceil(int(base.epoch_size) / int(world_size))
    datasets: list[tuple[int, Any]] = [(base_rank, base)]
    for rank in ranks[1:]:
        planner = SamplePlanner(
            mixture_sampler=base.mixture_sampler,
            seed=base.seed,
            rank=rank,
            world_size=world_size,
            reshuffle_each_epoch=base.reshuffle_each_epoch,
        )
        plan = planner.generate_plan(
            epoch=epoch,
            count_per_rank=count_per_rank,
            epoch_size=base.epoch_size,
            generation=0,
        )
        datasets.append(
            (
                rank,
                SimpleNamespace(
                    current_plan=plan,
                    adapters=base.adapters,
                    cleanup=lambda: None,
                ),
            )
        )
    datasets.sort(key=lambda item: item[0])
    print(
        f"[CoscliWarmProbe] rank_init_shared ranks={ranks} "
        f"time={time.perf_counter() - t0:.2f}s "
        f"plan_len={len(base.current_plan)} adapters={len(base.adapters)}",
        flush=True,
    )
    return datasets


def _replan_rank_datasets(
    rank_datasets: list[tuple[int, Any]],
    *,
    epoch: int,
    world_size: int,
) -> None:
    if not rank_datasets:
        return
    base = rank_datasets[0][1]
    count_per_rank = math.ceil(int(base.epoch_size) / int(world_size))
    for rank, dataset in rank_datasets:
        planner = SamplePlanner(
            mixture_sampler=base.mixture_sampler,
            seed=base.seed,
            rank=rank,
            world_size=world_size,
            reshuffle_each_epoch=base.reshuffle_each_epoch,
        )
        dataset.current_plan = planner.generate_plan(
            epoch=int(epoch),
            count_per_rank=count_per_rank,
            epoch_size=base.epoch_size,
            generation=0,
        )


def _collect_block(
    rank_datasets: list[tuple[int, Any]],
    stager: SampleLocalStager,
    *,
    batch_size: int,
    start_batch: int,
    block_batches: int,
    enabled_datasets: set[str],
    include_scannetpp_h5: bool,
    scannetpp_rgb_mode: str,
    decode_scannetpp_rgb: bool,
    exact_warm_datasets: set[str],
    sdk_exact_datasets: set[str],
    manifest_mode: str,
) -> tuple[
    list[Path],
    dict[tuple[str, str, str, str, str], PrefixSync],
    list[Path],
    Counter[str],
    Counter[str],
    dict[tuple[str, str, tuple[int, int]], ScanNetPPRgbDecodeSpec],
    dict[tuple[str, str], ScanNetPPDepthSpec],
    dict[tuple[int, str], ScanNetPPH5Spec],
]:
    unique_paths: dict[str, Path] = {}
    unique_prefixes: dict[tuple[str, str, str, str, str], PrefixSync] = {}
    exact_paths: dict[str, Path] = {}
    scannetpp_decode: dict[tuple[str, str, tuple[int, int]], ScanNetPPRgbDecodeSpec] = {}
    scannetpp_depth: dict[tuple[str, str], ScanNetPPDepthSpec] = {}
    scannetpp_h5: dict[tuple[int, str], ScanNetPPH5Spec] = {}
    scannetpp_frame_index_cache: dict[str, Any] = {}
    planned_prefix_frames: dict[tuple[str, str], tuple[Any, set[int]]] = {}
    planned_sdk_exact_frames: dict[tuple[str, str], tuple[Any, set[int]]] = {}
    planned_sdk_exact_paths: dict[str, Path] = {}
    sample_counts: Counter[str] = Counter()
    manifest_counts: Counter[str] = Counter()
    start = start_batch * batch_size
    end = start + block_batches * batch_size

    for _rank, dataset in rank_datasets:
        for spec in dataset.current_plan[start:min(end, len(dataset.current_plan))]:
            adapter = dataset.adapters[spec.dataset_idx]
            dataset_name = str(getattr(adapter, "dataset_name", type(adapter).__name__))
            sample_counts[dataset_name] += 1
            if dataset_name not in enabled_datasets:
                continue
            if not stager.supports(adapter):
                continue
            if dataset_name in exact_warm_datasets:
                manifest = stager._build_manifest(
                    adapter,
                    spec.sequence_name,
                    list(spec.frame_indices),
                )
                manifest_counts[dataset_name] += len(manifest)
                for path in manifest:
                    key = stager._to_cos_key(Path(path))
                    unique_paths.setdefault(key, Path(path))
                    exact_paths.setdefault(key, Path(path))
                continue
            if dataset_name in {"co3dv2", "dynamic_replica"}:
                planned_frames = (
                    planned_sdk_exact_frames
                    if dataset_name in sdk_exact_datasets
                    else planned_prefix_frames
                )
                planned_key = (dataset_name, spec.sequence_name)
                _adapter, frame_set = planned_frames.setdefault(
                    planned_key,
                    (adapter, set()),
                )
                frame_set.update(int(i) for i in list(spec.frame_indices))
                continue
            if dataset_name == "scannetpp" and dataset_name in sdk_exact_datasets:
                scene_dir = adapter.data_root / spec.sequence_name
                if scannetpp_rgb_mode == "frame_cache":
                    for frame_idx in list(spec.frame_indices):
                        path = scene_dir / "iphone" / "frames" / f"{int(frame_idx):06d}.jpg"
                        key = stager._to_cos_key(path)
                        planned_sdk_exact_paths.setdefault(key, path)
                else:
                    path = scene_dir / "iphone" / "rgb.mkv"
                    key = stager._to_cos_key(path)
                    planned_sdk_exact_paths.setdefault(key, path)
                syncs = []
            else:
                syncs = _prefix_for_spec(
                    stager,
                    adapter,
                    dataset_name,
                    spec.sequence_name,
                    list(spec.frame_indices),
                    include_scannetpp_h5=include_scannetpp_h5,
                    scannetpp_rgb_mode=scannetpp_rgb_mode,
                )
                for sync in syncs:
                    _add_prefix_sync(unique_prefixes, sync)
                    source = stager.mount_root / sync.cos_prefix.rstrip("/")
                    unique_paths.setdefault(f"prefix:{sync.cos_prefix}", source)
            if manifest_mode == "exact":
                manifest = stager._build_manifest(
                    adapter,
                    spec.sequence_name,
                    list(spec.frame_indices),
                )
                manifest_counts[dataset_name] += len(manifest)
                for path in manifest:
                    key = stager._to_cos_key(Path(path))
                    unique_paths.setdefault(key, Path(path))
                if dataset_name == "scannetpp" and include_scannetpp_h5:
                    scene_dir = adapter.data_root / spec.sequence_name
                    for name in ("precomputed.h5", "precomputed.h5_chunk_index.pkl"):
                        path = scene_dir / name
                        key = stager._to_cos_key(path)
                        unique_paths.setdefault(key, path)
            else:
                manifest_counts[dataset_name] += len(syncs)
            if (
                dataset_name == "scannetpp"
                and scannetpp_rgb_mode in {"cache", "frame_cache"}
                and decode_scannetpp_rgb
                and getattr(adapter, "target_hw", None) is not None
            ):
                target_hw = tuple(int(v) for v in adapter.target_hw)
                scene_dir = adapter.data_root / spec.sequence_name
                decode_key = (scene_dir.as_posix(), spec.sequence_name, target_hw)
                decode = scannetpp_decode.setdefault(
                    decode_key,
                    ScanNetPPRgbDecodeSpec(
                        sequence=spec.sequence_name,
                        scene_dir=scene_dir,
                        target_hw=target_hw,
                        frame_indices=set(),
                        source=(
                            "frames"
                            if scannetpp_rgb_mode == "frame_cache"
                            else "video"
                        ),
                    ),
                )
                for frame_idx in list(spec.frame_indices):
                    idx = int(frame_idx)
                    decode.frame_indices.add(idx)
                    cache_path = rgb_cache_frame_path(scene_dir, target_hw, idx)
                    key = stager._to_cos_key(cache_path)
                    unique_paths.setdefault(key, cache_path)
            if dataset_name == "scannetpp":
                h5_key = (id(adapter), spec.sequence_name)
                h5_spec = scannetpp_h5.setdefault(
                    h5_key,
                    ScanNetPPH5Spec(
                        adapter=adapter,
                        sequence=spec.sequence_name,
                        frame_indices=set(),
                    ),
                )
                for frame_idx in list(spec.frame_indices):
                    h5_spec.frame_indices.add(int(frame_idx))

                scene_dir = adapter.data_root / spec.sequence_name
                depth_key = (scene_dir.as_posix(), spec.sequence_name)
                depth_spec = scannetpp_depth.setdefault(
                    depth_key,
                    ScanNetPPDepthSpec(
                        sequence=spec.sequence_name,
                        scene_dir=scene_dir,
                        full_indices=set(),
                    ),
                )
                try:
                    scene_key = scene_dir.as_posix()
                    scene_data = scannetpp_frame_index_cache.get(scene_key)
                    if scene_data is None:
                        scene_data = _load_scannetpp_frame_index_for_collect(
                            stager,
                            scene_dir,
                        )
                        scannetpp_frame_index_cache[scene_key] = scene_data
                    full_indices = scene_data["full_indices"]
                    for frame_idx in list(spec.frame_indices):
                        depth_spec.full_indices.add(int(full_indices[int(frame_idx)]))
                except Exception:
                    for frame_idx in list(spec.frame_indices):
                        depth_spec.full_indices.add(int(frame_idx))
    for (dataset_name, sequence_name), (adapter, frame_set) in planned_prefix_frames.items():
        manifest = stager._build_manifest(
            adapter,
            sequence_name,
            sorted(frame_set),
        )
        manifest_counts[dataset_name] += len(manifest)
        by_key: dict[str, Path] = {}
        for path in manifest:
            key = stager._to_cos_key(Path(path))
            unique_paths.setdefault(key, Path(path))
            by_key.setdefault(key, Path(path))
        syncs = _prefix_syncs_for_exact_paths(
            stager,
            dataset=dataset_name,
            sequence=sequence_name,
            paths=list(by_key.values()),
        )
        for sync in syncs:
            _add_prefix_sync(unique_prefixes, sync)
    for (dataset_name, sequence_name), (adapter, frame_set) in planned_sdk_exact_frames.items():
        manifest = stager._build_manifest(
            adapter,
            sequence_name,
            sorted(frame_set),
        )
        manifest_counts[dataset_name] += len(manifest)
        for path in manifest:
            key = stager._to_cos_key(Path(path))
            unique_paths.setdefault(key, Path(path))
            exact_paths.setdefault(key, Path(path))
    for key, path in planned_sdk_exact_paths.items():
        manifest_counts["scannetpp"] += 1
        unique_paths.setdefault(key, path)
        exact_paths.setdefault(key, path)
    return (
        list(unique_paths.values()),
        unique_prefixes,
        list(exact_paths.values()),
        sample_counts,
        manifest_counts,
        scannetpp_decode,
        scannetpp_depth,
        scannetpp_h5,
    )


def _sync_one(
    sync: PrefixSync,
    *,
    bucket: str,
    routines: int,
    thread_num: int,
    timeout_s: float,
    update: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    if skip_existing:
        if sync.expected_paths:
            if not _missing_expected_paths(sync.expected_paths, limit=1):
                return {"ok": True, "skipped": True, "elapsed_s": 0.0, "sync": sync}
        else:
            marker_name = _sync_marker_name(sync)
            complete_marker = sync.local_dir / marker_name
            if sync.local_dir.is_dir() and complete_marker.is_file():
                return {"ok": True, "skipped": True, "elapsed_s": 0.0, "sync": sync}
            if sync.local_dir.is_file():
                return {"ok": True, "skipped": True, "elapsed_s": 0.0, "sync": sync}

    src = f"cos://{bucket}/{sync.cos_prefix}"
    dst = sync.local_dir.as_posix()
    if sync.cos_prefix.endswith("/"):
        dst = dst.rstrip("/") + "/"
        Path(dst).mkdir(parents=True, exist_ok=True)
    else:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "coscli",
        "sync",
        src,
        dst,
        "--routines",
        str(routines),
        "--thread-num",
        str(thread_num),
        "--disable-log",
        "--process-log=false",
        "--fail-output=false",
    ]
    if sync.cos_prefix.endswith("/"):
        cmd.append("-r")
    if sync.include_pattern:
        cmd.extend(["--include", sync.include_pattern])
    if update:
        cmd.append("--update")
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    missing_expected: list[Path] = []
    if proc.returncode == 0 and sync.expected_paths:
        missing_expected = _missing_expected_paths(sync.expected_paths)
    ok = proc.returncode == 0 and not missing_expected
    if ok and sync.cos_prefix.endswith("/"):
        try:
            marker_name = _sync_marker_name(sync)
            (sync.local_dir / marker_name).write_text(
                (
                    f"prefix={sync.cos_prefix}\n"
                    f"include={sync.include_pattern}\n"
                    f"expected={len(sync.expected_paths)}\n"
                    f"time={time.time():.6f}\n"
                ),
                encoding="utf-8",
            )
        except OSError:
            pass
    output_tail = "\n".join(proc.stdout.splitlines()[-8:])
    if missing_expected:
        output_tail = (
            output_tail
            + "\nmissing_expected="
            + ",".join(path.name for path in missing_expected[:20])
        ).strip()
    return {
        "ok": ok,
        "skipped": False,
        "elapsed_s": elapsed,
        "returncode": proc.returncode,
        "sync": sync,
        "missing_expected": len(missing_expected),
        "output_tail": output_tail,
    }


def _warm_prefixes(
    prefixes: list[PrefixSync],
    *,
    bucket: str,
    routines: int,
    thread_num: int,
    timeout_s: float,
    prefix_workers: int,
    update: bool,
    skip_existing: bool,
) -> tuple[float, list[dict[str, Any]]]:
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, prefix_workers)) as ex:
        futs = [
            ex.submit(
                _sync_one,
                sync,
                bucket=bucket,
                routines=routines,
                thread_num=thread_num,
                timeout_s=timeout_s,
                update=update,
                skip_existing=skip_existing,
            )
            for sync in prefixes
        ]
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())
    return time.perf_counter() - t0, results


def _warm_prefixes_and_decode(
    *,
    prefixes: list[PrefixSync],
    stager: SampleLocalStager,
    scannetpp_decode: dict[tuple[str, str, tuple[int, int]], ScanNetPPRgbDecodeSpec],
    bucket: str,
    routines: int,
    thread_num: int,
    timeout_s: float,
    prefix_workers: int,
    update: bool,
    skip_existing: bool,
    decode_workers: int,
    frame_workers: int,
    overlap_decode: bool,
) -> tuple[float, float, float, list[dict[str, Any]], list[dict[str, Any]]]:
    if not overlap_decode or not scannetpp_decode:
        prefix_s, results = _warm_prefixes(
            prefixes,
            bucket=bucket,
            routines=routines,
            thread_num=thread_num,
            timeout_s=timeout_s,
            prefix_workers=prefix_workers,
            update=update,
            skip_existing=skip_existing,
        )
        decode_s, decode_results = _decode_scannetpp_rgb_caches(
            stager,
            scannetpp_decode,
            workers=decode_workers,
            frame_workers=frame_workers,
        )
        return prefix_s + decode_s, prefix_s, decode_s, results, decode_results

    scannetpp_prefixes = [sync for sync in prefixes if sync.dataset == "scannetpp"]
    other_prefixes = [sync for sync in prefixes if sync.dataset != "scannetpp"]
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        scan_future = ex.submit(
            _warm_prefixes,
            scannetpp_prefixes,
            bucket=bucket,
            routines=routines,
            thread_num=thread_num,
            timeout_s=timeout_s,
            prefix_workers=prefix_workers,
            update=update,
            skip_existing=skip_existing,
        )
        other_future = ex.submit(
            _warm_prefixes,
            other_prefixes,
            bucket=bucket,
            routines=routines,
            thread_num=thread_num,
            timeout_s=timeout_s,
            prefix_workers=prefix_workers,
            update=update,
            skip_existing=skip_existing,
        )
        scan_s, scan_results = scan_future.result()
        decode_future = ex.submit(
            _decode_scannetpp_rgb_caches,
            stager,
            scannetpp_decode,
            workers=decode_workers,
            frame_workers=frame_workers,
        )
        other_s, other_results = other_future.result()
        decode_s, decode_results = decode_future.result()
    elapsed = time.perf_counter() - t0
    return elapsed, max(scan_s, other_s), decode_s, scan_results + other_results, decode_results


def _warm_exact_paths(
    stager: SampleLocalStager,
    paths: list[Path],
) -> tuple[float, dict[str, Any]]:
    if not paths:
        return 0.0, {"files": 0, "cold_files": 0, "file_sum_s": 0.0, "file_max_s": 0.0}
    t0 = time.perf_counter()
    stats = stager._materialize_manifest_cache_only(paths)
    return time.perf_counter() - t0, stats


def _fmt_counter(counter: Counter[str]) -> str:
    return ",".join(f"{k}:{v}" for k, v in sorted(counter.items())) or "-"


def _write_ready_marker(
    ready_dir: str,
    *,
    generation: int,
    block_id: int,
    start_batch: int,
    block_batches: int,
    warm_s: float,
    bytes_total: int,
    files: int,
) -> None:
    if not ready_dir:
        return
    root = Path(ready_dir) / f"g{int(generation):04d}"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"block_{int(block_id):08d}.ready"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        "\n".join(
            [
                f"generation={int(generation)}",
                f"block={int(block_id)}",
                f"start_batch={int(start_batch)}",
                f"block_batches={int(block_batches)}",
                f"warm_s={float(warm_s):.6f}",
                f"bytes={int(bytes_total)}",
                f"files={int(files)}",
                f"time={time.time():.6f}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _write_block_manifest(
    ready_dir: str,
    *,
    generation: int,
    block_id: int,
    cache_paths: list[Path],
) -> None:
    if not ready_dir:
        return
    root = Path(ready_dir) / f"g{int(generation):04d}"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"block_{int(block_id):08d}.manifest"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    lines_set: set[str] = set()
    for cache_path in cache_paths:
        text = Path(cache_path).as_posix()
        try:
            if Path(cache_path).is_dir():
                text = text.rstrip("/") + "/"
        except OSError:
            pass
        lines_set.add(text)
    lines = sorted(lines_set)
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    os.replace(tmp, path)


def _cleanup_old_ready_markers(
    ready_dir: str,
    *,
    generation: int,
    keep_from_block: int,
) -> None:
    if not ready_dir:
        return
    root = Path(ready_dir) / f"g{int(generation):04d}"
    if not root.is_dir():
        return
    for path in root.glob("block_*.ready"):
        try:
            block_id = int(path.stem.split("_", 1)[1])
        except Exception:
            continue
        if block_id < keep_from_block:
            path.unlink(missing_ok=True)
            (root / f"block_{block_id:08d}.manifest").unlink(missing_ok=True)


def _cleanup_old_generations(ready_dir: str, *, keep_generation: int) -> None:
    if not ready_dir:
        return
    root = Path(ready_dir)
    if not root.is_dir():
        return
    for gen_dir in root.glob("g*"):
        if not gen_dir.is_dir():
            continue
        try:
            generation = int(gen_dir.name[1:])
        except ValueError:
            continue
        if generation < keep_generation:
            shutil.rmtree(gen_dir, ignore_errors=True)


def _ready_marker_path(
    ready_dir: str,
    *,
    generation: int,
    block_id: int,
) -> Path | None:
    if not ready_dir:
        return None
    return (
        Path(ready_dir)
        / f"g{int(generation):04d}"
        / f"block_{int(block_id):08d}.ready"
    )


def _ready_exists(ready_dir: str, *, generation: int, block_id: int) -> bool:
    path = _ready_marker_path(ready_dir, generation=generation, block_id=block_id)
    return bool(path and path.is_file())


def _parse_progress_file(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for raw in lines:
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in {"rank", "local_index", "local_batch", "block", "generation", "epoch"}:
            try:
                out[key] = int(value)
            except ValueError:
                continue
    return out


def _progress_state(
    progress_dir: str,
    *,
    generation: int,
    epoch: int,
    fallback_block: int,
    world_size: int,
    min_progress_ranks: int,
) -> tuple[int, int, int, int]:
    if not progress_dir:
        return generation, epoch, fallback_block, 0
    progress_root = Path(progress_dir)
    roots: list[tuple[int, Path]] = []
    for root in progress_root.glob("g*"):
        if not root.is_dir():
            continue
        try:
            gen = int(root.name[1:])
        except ValueError:
            continue
        roots.append((gen, root))
    if not roots:
        return generation, epoch, fallback_block, 0
    roots.sort(reverse=True)
    selected_generation = generation
    selected_epoch = epoch
    selected_blocks: list[int] = []
    progress_ranks = 0
    for gen, root in roots:
        blocks: list[int] = []
        epochs: list[int] = []
        ranks_seen: set[int] = set()
        for path in root.glob("rank*.progress"):
            parsed = _parse_progress_file(path)
            if parsed.get("generation", gen) != gen:
                continue
            if "rank" in parsed:
                ranks_seen.add(int(parsed["rank"]))
            if "block" in parsed:
                blocks.append(int(parsed["block"]))
            if "epoch" in parsed:
                epochs.append(int(parsed["epoch"]))
        if blocks:
            selected_generation = gen
            selected_epoch = min(epochs) if epochs else epoch
            selected_blocks = blocks
            progress_ranks = len(ranks_seen) or len(blocks)
            break
    if not selected_blocks:
        return generation, epoch, fallback_block, 0
    required = min_progress_ranks if min_progress_ranks > 0 else max(1, world_size)
    if progress_ranks < required:
        return selected_generation, selected_epoch, fallback_block, progress_ranks
    return (
        selected_generation,
        selected_epoch,
        max(fallback_block, min(selected_blocks)),
        progress_ranks,
    )


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _cache_paths_for_manifest(stager: SampleLocalStager, paths: list[Path]) -> list[Path]:
    return [_cache_path(stager, path) for path in paths]


def _valid_rgb_cache_file(path: Path, target_hw: tuple[int, int]) -> bool:
    try:
        return path.is_file() and path.stat().st_size == int(target_hw[0]) * int(target_hw[1]) * 3
    except OSError:
        return False


def _decode_scannetpp_rgb_one(
    stager: SampleLocalStager,
    spec: ScanNetPPRgbDecodeSpec,
    *,
    frame_workers: int = 1,
) -> dict[str, Any]:
    rel_scene = Path(stager._to_cos_key(spec.scene_dir))
    local_scene = stager.cache_data_root / rel_scene
    local_video = local_scene / "iphone" / "rgb.mkv"
    source = str(getattr(spec, "source", "video") or "video")
    if source == "video" and not local_video.is_file():
        raise FileNotFoundError(local_video)

    target_hw = spec.target_hw
    frame_indices = sorted(int(i) for i in spec.frame_indices)
    missing: list[int] = []
    output_paths: dict[int, Path] = {}
    for idx in frame_indices:
        source_cache_path = rgb_cache_frame_path(spec.scene_dir, target_hw, idx)
        rel_key = Path(stager._to_cos_key(source_cache_path))
        cache_path = stager.cache_data_root / rel_key
        output_paths[idx] = cache_path
        if not _valid_rgb_cache_file(cache_path, target_hw):
            missing.append(idx)
        else:
            stager._touch_cache_entry(cache_path)

    if not missing:
        return {
            "sequence": spec.sequence,
            "frames": len(frame_indices),
            "decoded": 0,
            "bytes": 0,
            "elapsed_s": 0.0,
            "source": source,
        }

    t0 = time.perf_counter()
    if source == "frames":
        frames_dir = local_scene / "iphone" / "frames"
        def _decode_one_frame(idx: int) -> tuple[int, int, bool]:
            source_cache_path = rgb_cache_frame_path(spec.scene_dir, target_hw, idx)
            rel_key = Path(stager._to_cos_key(source_cache_path))
            cache_path = output_paths[idx]
            with stager._path_lock(rel_key):
                if _valid_rgb_cache_file(cache_path, target_hw):
                    stager._touch_cache_entry(cache_path)
                    return idx, 0, False
                image = _read_rgb_jpg(frames_dir / f"{int(idx):06d}.jpg")
                bytes_one = write_rgb_cache_frame(cache_path, image, target_hw)
                stager._touch_cache_entry(cache_path, force=True)
                return idx, bytes_one, True

        bytes_written = 0
        decoded = 0
        workers = max(1, min(int(frame_workers), len(missing)))
        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_decode_one_frame, idx) for idx in missing]
                for fut in concurrent.futures.as_completed(futures):
                    _idx, bytes_one, did_decode = fut.result()
                    bytes_written += int(bytes_one)
                    decoded += int(bool(did_decode))
        else:
            for idx in missing:
                _idx, bytes_one, did_decode = _decode_one_frame(idx)
                bytes_written += int(bytes_one)
                decoded += int(bool(did_decode))

        if bytes_written:
            stager._adjust_cache_usage(bytes_written)

        return {
            "sequence": spec.sequence,
            "frames": len(frame_indices),
            "decoded": decoded,
            "bytes": bytes_written,
            "elapsed_s": time.perf_counter() - t0,
            "source": source,
        }
    else:
        scene_data = _load_frame_index(local_scene)
        timestamps_all = [float(scene_data["timestamps"][idx]) for idx in missing]
        fallback_indices_all = scene_data["full_indices"][missing].tolist()
        max_gap = max(1, int(os.getenv("SCANNETPP_RGB_DECODE_MAX_GAP", "16") or "16"))
        ordered_positions = sorted(
            range(len(missing)),
            key=lambda pos: int(fallback_indices_all[pos]),
        )
        groups: list[list[int]] = []
        current: list[int] = []
        prev_frame: int | None = None
        for pos in ordered_positions:
            frame = int(fallback_indices_all[pos])
            if current and prev_frame is not None and frame - prev_frame > max_gap:
                groups.append(current)
                current = []
            current.append(pos)
            prev_frame = frame
        if current:
            groups.append(current)

        images_by_pos: list[np.ndarray | None] = [None] * len(missing)
        for group in groups:
            group_timestamps = [timestamps_all[pos] for pos in group]
            group_fallback = [int(fallback_indices_all[pos]) for pos in group]
            group_images = _extract_video_frames_by_timestamps(
                local_video,
                group_timestamps,
                group_fallback,
            )
            for pos, image in zip(group, group_images):
                images_by_pos[pos] = image
        images = [image for image in images_by_pos if image is not None]
    if len(images) != len(missing):
        raise IOError(
            f"Decoded {len(images)}/{len(missing)} RGB cache frames for {spec.sequence}"
        )

    bytes_written = 0
    decoded = 0
    for idx, image in zip(missing, images):
        source_cache_path = rgb_cache_frame_path(spec.scene_dir, target_hw, idx)
        rel_key = Path(stager._to_cos_key(source_cache_path))
        cache_path = output_paths[idx]
        with stager._path_lock(rel_key):
            if _valid_rgb_cache_file(cache_path, target_hw):
                stager._touch_cache_entry(cache_path)
                continue
            bytes_written += write_rgb_cache_frame(cache_path, image, target_hw)
            stager._touch_cache_entry(cache_path, force=True)
            decoded += 1

    if bytes_written:
        stager._adjust_cache_usage(bytes_written)

    return {
        "sequence": spec.sequence,
        "frames": len(frame_indices),
        "decoded": decoded,
        "bytes": bytes_written,
        "elapsed_s": time.perf_counter() - t0,
        "source": source,
    }


def _decode_scannetpp_rgb_caches(
    stager: SampleLocalStager,
    decode_specs: dict[tuple[str, str, tuple[int, int]], ScanNetPPRgbDecodeSpec],
    *,
    workers: int,
    frame_workers: int = 1,
) -> tuple[float, list[dict[str, Any]]]:
    if not decode_specs:
        return 0.0, []
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [
            ex.submit(
                _decode_scannetpp_rgb_one,
                stager,
                spec,
                frame_workers=frame_workers,
            )
            for spec in decode_specs.values()
        ]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
    return time.perf_counter() - t0, results


def _scannetpp_depth_cache_paths(
    stager: SampleLocalStager,
    depth_specs: dict[tuple[str, str], ScanNetPPDepthSpec],
) -> list[Path]:
    paths: list[Path] = []
    for spec in depth_specs.values():
        rel_scene = Path(stager._to_cos_key(spec.scene_dir))
        paths.append(stager.cache_data_root / rel_scene / "iphone" / "depth_chunk_index.pkl")
        depth_key = f"{rel_scene.as_posix()}/iphone/depth.bin"
        chunk_dir = stager.cache_data_root / Path(f"{depth_key}.chunks")
        for full_idx in sorted(int(i) for i in spec.full_indices):
            paths.append(chunk_dir / f"{full_idx:08d}.bin")
    return paths


def _warm_scannetpp_depth_one(
    stager: SampleLocalStager,
    spec: ScanNetPPDepthSpec,
) -> dict[str, Any]:
    frame_indices = sorted(int(i) for i in spec.full_indices)
    if not frame_indices:
        return {
            "sequence": spec.sequence,
            "chunks": 0,
            "present": 0,
            "bytes": 0,
            "elapsed_s": 0.0,
        }
    local_scene = stager.cache_data_root / Path(stager._to_cos_key(spec.scene_dir))
    before_paths = _scannetpp_depth_cache_paths(stager, {("one", spec.sequence): spec})
    before_present = sum(1 for path in before_paths if path.is_file())
    t0 = time.perf_counter()
    stager._stage_scannetpp_depth(spec.scene_dir, local_scene, frame_indices)
    elapsed = time.perf_counter() - t0
    after_present = 0
    bytes_total = 0
    for path in before_paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        after_present += 1
        bytes_total += stat.st_size
    return {
        "sequence": spec.sequence,
        "chunks": len(frame_indices),
        "present": after_present,
        "new": max(0, after_present - before_present),
        "bytes": bytes_total,
        "elapsed_s": elapsed,
    }


def _warm_scannetpp_depth_chunks(
    stager: SampleLocalStager,
    depth_specs: dict[tuple[str, str], ScanNetPPDepthSpec],
    *,
    workers: int,
) -> tuple[float, list[dict[str, Any]]]:
    if not depth_specs:
        return 0.0, []
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [
            ex.submit(_warm_scannetpp_depth_one, stager, spec)
            for spec in depth_specs.values()
        ]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
    return time.perf_counter() - t0, results


def _warm_scannetpp_h5_one(spec: ScanNetPPH5Spec) -> dict[str, Any]:
    cache_dir = getattr(spec.adapter, "precomputed_h5_chunk_cache_dir", None)
    if cache_dir is None:
        return {
            "sequence": spec.sequence,
            "frames": len(spec.frame_indices),
            "warmed": False,
            "elapsed_s": 0.0,
        }
    frame_indices = sorted(int(i) for i in spec.frame_indices)
    if not frame_indices:
        return {
            "sequence": spec.sequence,
            "frames": 0,
            "warmed": False,
            "elapsed_s": 0.0,
        }
    t0 = time.perf_counter()
    spec.adapter._load_precomputed(spec.sequence, frame_indices)
    return {
        "sequence": spec.sequence,
        "frames": len(frame_indices),
        "warmed": True,
        "elapsed_s": time.perf_counter() - t0,
    }


def _warm_scannetpp_h5_chunks(
    h5_specs: dict[tuple[int, str], ScanNetPPH5Spec],
    *,
    workers: int,
) -> tuple[float, list[dict[str, Any]]]:
    specs = [
        spec
        for spec in h5_specs.values()
        if getattr(spec.adapter, "precomputed_h5_chunk_cache_dir", None) is not None
    ]
    if not specs:
        return 0.0, []
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(_warm_scannetpp_h5_one, spec) for spec in specs]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
    return time.perf_counter() - t0, results


def _warm_block(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    stager: SampleLocalStager,
    rank_datasets: list[tuple[int, Any]],
    enabled_datasets: set[str],
    generation: int,
    start_batch: int,
    block_id: int | None = None,
) -> bool:
    block_start = start_batch
    if block_id is None:
        block_id = block_start // args.block_batches
    collect_t0 = time.perf_counter()
    (
        paths,
        prefix_map,
        exact_paths,
        sample_counts,
        manifest_counts,
        scannetpp_decode,
        scannetpp_depth,
        scannetpp_h5,
    ) = _collect_block(
        rank_datasets,
        stager,
        batch_size=args.batch_size,
        start_batch=block_start,
        block_batches=args.block_batches,
        enabled_datasets=enabled_datasets,
        include_scannetpp_h5=args.include_scannetpp_h5,
        scannetpp_rgb_mode=args.scannetpp_rgb_mode,
        decode_scannetpp_rgb=not args.no_scannetpp_rgb_decode,
        exact_warm_datasets={
            item.strip()
            for item in str(args.exact_warm_datasets).split(",")
            if item.strip()
        },
        sdk_exact_datasets={
            item.strip()
            for item in str(args.sdk_exact_datasets).split(",")
            if item.strip()
        },
        manifest_mode=args.manifest_mode,
    )
    collect_s = time.perf_counter() - collect_t0
    prefixes = sorted(prefix_map.values(), key=lambda s: (s.dataset, s.sequence, s.cos_prefix))
    depth_cache_paths = (
        []
        if args.no_scannetpp_depth_warm
        else _scannetpp_depth_cache_paths(stager, scannetpp_depth)
    )
    existing, existing_bytes = _cache_stats(stager, paths)
    depth_existing = sum(1 for path in depth_cache_paths if path.is_file())
    depth_existing_bytes = sum(
        path.stat().st_size for path in depth_cache_paths if path.is_file()
    )
    budget_s = args.block_batches * args.compute_s_per_batch
    prefix_counts = Counter(sync.dataset for sync in prefixes)
    print(
        "[CoscliWarmProbe] block_manifest "
        f"block={block_id} start_batch={block_start} "
        f"samples={_fmt_counter(sample_counts)} manifest_files={len(paths)} "
        f"existing={existing}/{len(paths)} existing_mb={existing_bytes / 1024**2:.1f} "
        f"exact_files={len(exact_paths)} "
        f"depth_chunks={depth_existing}/{len(depth_cache_paths)} "
        f"depth_mb={depth_existing_bytes / 1024**2:.1f} "
        f"prefixes={len(prefixes)} prefix_counts={_fmt_counter(prefix_counts)} "
        f"collect={collect_s:.2f}s budget={budget_s:.2f}s",
        flush=True,
    )
    if args.dry_run:
        return True

    sdk_exact_dataset_set = {
        item.strip()
        for item in str(args.sdk_exact_datasets).split(",")
        if item.strip()
    }
    decode_depends_on_exact = bool(scannetpp_decode) and "scannetpp" in sdk_exact_dataset_set

    warm_t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        exact_future = ex.submit(_warm_exact_paths, stager, exact_paths)
        depth_future = None
        if not args.no_scannetpp_depth_warm:
            depth_future = ex.submit(
                _warm_scannetpp_depth_chunks,
                stager,
                scannetpp_depth,
                workers=args.scannetpp_depth_workers,
            )
        h5_future = None
        if not args.no_scannetpp_h5_warm:
            h5_future = ex.submit(
                _warm_scannetpp_h5_chunks,
                scannetpp_h5,
                workers=args.scannetpp_h5_workers,
            )
        if decode_depends_on_exact:
            # ScanNet++ SDK-exact mode downloads the source JPEGs in
            # exact_future; decoding the resized RGB cache must wait for those
            # files to exist.
            exact_s, exact_stats = exact_future.result()
            prefix_future = ex.submit(
                _warm_prefixes_and_decode,
                prefixes=prefixes,
                stager=stager,
                scannetpp_decode=scannetpp_decode,
                bucket=str(config.get("sample_stage_bucket", "hd-ai-data-1251882982")),
                routines=args.coscli_routines,
                thread_num=args.coscli_thread_num,
                timeout_s=args.timeout_s,
                prefix_workers=args.prefix_workers,
                update=not args.no_update,
                skip_existing=args.skip_existing,
                decode_workers=args.scannetpp_decode_workers,
                frame_workers=args.scannetpp_frame_workers,
                overlap_decode=False,
            )
        else:
            prefix_future = ex.submit(
                _warm_prefixes_and_decode,
                prefixes=prefixes,
                stager=stager,
                scannetpp_decode=scannetpp_decode,
                bucket=str(config.get("sample_stage_bucket", "hd-ai-data-1251882982")),
                routines=args.coscli_routines,
                thread_num=args.coscli_thread_num,
                timeout_s=args.timeout_s,
                prefix_workers=args.prefix_workers,
                update=not args.no_update,
                skip_existing=args.skip_existing,
                decode_workers=args.scannetpp_decode_workers,
                frame_workers=args.scannetpp_frame_workers,
                overlap_decode=not args.no_overlap_decode,
            )
            exact_s, exact_stats = exact_future.result()
        prefix_elapsed_s, prefix_warm_s, decode_s, results, decode_results = prefix_future.result()
        depth_s = 0.0
        depth_results: list[dict[str, Any]] = []
        if depth_future is not None:
            depth_s, depth_results = depth_future.result()
        h5_s = 0.0
        h5_results: list[dict[str, Any]] = []
        if h5_future is not None:
            h5_s, h5_results = h5_future.result()
    warm_s = time.perf_counter() - warm_t0
    failed = [r for r in results if not r.get("ok")]
    present, bytes_total = _cache_stats(stager, paths)
    depth_present = sum(1 for path in depth_cache_paths if path.is_file())
    depth_bytes = sum(path.stat().st_size for path in depth_cache_paths if path.is_file())
    total_warm_s = collect_s + warm_s
    keep_up = total_warm_s <= budget_s
    by_dataset_elapsed: defaultdict[str, list[float]] = defaultdict(list)
    skipped = 0
    for result in results:
        sync = result["sync"]
        by_dataset_elapsed[sync.dataset].append(float(result.get("elapsed_s", 0.0)))
        skipped += int(bool(result.get("skipped")))
    by_dataset_msg = ",".join(
        f"{dataset}:n{len(vals)}/max{max(vals):.2f}s/sum{sum(vals):.2f}s"
        for dataset, vals in sorted(by_dataset_elapsed.items())
        if vals
    )
    decoded_frames = sum(int(item.get("decoded", 0)) for item in decode_results)
    decode_bytes = sum(int(item.get("bytes", 0)) for item in decode_results)
    decode_max_s = max((float(item.get("elapsed_s", 0.0)) for item in decode_results), default=0.0)
    depth_chunks = sum(int(item.get("chunks", 0)) for item in depth_results)
    depth_new = sum(int(item.get("new", 0)) for item in depth_results)
    depth_max_s = max((float(item.get("elapsed_s", 0.0)) for item in depth_results), default=0.0)
    h5_sequences = sum(1 for item in h5_results if item.get("warmed"))
    h5_frames = sum(int(item.get("frames", 0)) for item in h5_results if item.get("warmed"))
    h5_max_s = max((float(item.get("elapsed_s", 0.0)) for item in h5_results), default=0.0)
    ok = (
        len(failed) == 0
        and present == len(paths)
        and (args.no_scannetpp_depth_warm or depth_present == len(depth_cache_paths))
    )
    print(
        "[CoscliWarmProbe] block_warm "
        f"block={block_id} total={total_warm_s:.3f}s collect={collect_s:.3f}s "
        f"warm={warm_s:.3f}s exact={exact_s:.3f}s "
        f"prefix={prefix_warm_s:.3f}s "
        f"decode={decode_s:.3f}s depth={depth_s:.3f}s h5={h5_s:.3f}s "
        f"budget={budget_s:.3f}s "
        f"ratio={total_warm_s / max(1e-6, budget_s):.2f} keep_up={keep_up} "
        f"present={present}/{len(paths)} bytes_mb={bytes_total / 1024**2:.1f} "
        f"depth_present={depth_present}/{len(depth_cache_paths)} "
        f"depth_mb={depth_bytes / 1024**2:.1f} depth_chunks={depth_chunks} "
        f"depth_new={depth_new} depth_max={depth_max_s:.2f}s "
        f"h5_seq={h5_sequences} h5_frames={h5_frames} h5_max={h5_max_s:.2f}s "
        f"exact_cold={int(exact_stats.get('cold_files', 0))} "
        f"exact_file_max={float(exact_stats.get('file_max_s', 0.0)):.2f}s "
        f"failed={len(failed)} skipped={skipped} "
        f"decoded_frames={decoded_frames} decode_mb={decode_bytes / 1024**2:.1f} "
        f"decode_max={decode_max_s:.2f}s prefix_elapsed={by_dataset_msg}",
        flush=True,
    )
    for result in failed[:10]:
        sync = result["sync"]
        print(
            "[CoscliWarmProbe] sync_failed "
            f"dataset={sync.dataset} seq={sync.sequence} prefix={sync.cos_prefix} "
            f"rc={result.get('returncode')} tail={result.get('output_tail', '')}",
            flush=True,
        )
    if ok:
        _write_block_manifest(
            args.ready_dir,
            generation=generation,
            block_id=block_id,
            cache_paths=_cache_paths_for_manifest(stager, paths) + depth_cache_paths,
        )
        _write_ready_marker(
            args.ready_dir,
            generation=generation,
            block_id=block_id,
            start_batch=block_start,
            block_batches=args.block_batches,
            warm_s=total_warm_s,
            bytes_total=bytes_total,
            files=len(paths),
        )
    return ok


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("D4RT_SERIALIZE_ADAPTER_INIT", "1")
    os.environ.setdefault("D4RT_PLANNED_BATCH_BALANCE", "1")
    os.environ.setdefault("D4RT_PLANNED_BATCH_SIZE", str(args.batch_size))
    os.environ["SCANNETPP_RGB_READ_MODE"] = (
        "cache" if str(args.scannetpp_rgb_mode) == "frame_cache" else str(args.scannetpp_rgb_mode)
    )

    stage_root = Path(args.stage_root)
    if args.clean_stage_root:
        _safe_clean_stage_root(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    config = _load_config(
        args.config,
        stage_root,
        scannetpp_h5_cache_dir=args.scannetpp_h5_cache_dir,
        scannetpp_h5_cache_min_bytes=args.scannetpp_h5_cache_min_bytes,
        scannetpp_h5_cache_max_gb=args.scannetpp_h5_cache_max_gb,
        scannetpp_h5_cache_low_watermark=args.scannetpp_h5_cache_low_watermark,
    )
    config["planned_initial_epoch"] = int(args.epoch)
    if int(args.stage_sdk_workers) > 0:
        config["sample_stage_sdk_workers"] = int(args.stage_sdk_workers)
    stager = _stager_from_config(config)
    ranks = _rank_list(args.ranks, args.world_size)
    enabled_datasets = {x.strip() for x in args.datasets.split(",") if x.strip()}

    print(
        "[CoscliWarmProbe] start "
        f"config={args.config} stage_root={stage_root} epoch={args.epoch} "
        f"world_size={args.world_size} ranks={ranks} batch_size={args.batch_size} "
        f"start_batch={args.start_batch} block_batches={args.block_batches} "
        f"blocks={args.blocks} datasets={sorted(enabled_datasets)} "
        f"exact_warm_datasets={args.exact_warm_datasets} "
        f"sdk_exact_datasets={args.sdk_exact_datasets} "
        f"scannetpp_rgb_mode={args.scannetpp_rgb_mode} "
        f"scannetpp_rgb_decode={not args.no_scannetpp_rgb_decode} "
        f"manifest_mode={args.manifest_mode} "
        f"scannetpp_depth_warm={not args.no_scannetpp_depth_warm} "
        f"scannetpp_depth_workers={args.scannetpp_depth_workers} "
        f"scannetpp_h5_warm={not args.no_scannetpp_h5_warm} "
        f"scannetpp_h5_workers={args.scannetpp_h5_workers} "
        f"scannetpp_h5_cache={args.scannetpp_h5_cache_dir or '<disabled>'} "
        f"scannetpp_h5_min_bytes={args.scannetpp_h5_cache_min_bytes} "
        f"prefix_workers={args.prefix_workers} sdk_workers={config.get('sample_stage_sdk_workers')} "
        f"routines={args.coscli_routines} "
        f"thread_num={args.coscli_thread_num}",
        flush=True,
    )

    rank_datasets = _init_rank_datasets(
        config,
        ranks=ranks,
        world_size=args.world_size,
    )
    try:
        if args.daemon:
            warmed: set[tuple[int, int]] = set()
            inflight: dict[tuple[int, int], concurrent.futures.Future] = {}
            active_generation = int(args.generation)
            active_epoch = int(args.epoch)
            base_block = max(0, int(args.start_batch) // max(1, int(args.block_batches)))
            progress_dir = args.progress_dir or args.ready_dir
            block_workers = max(1, int(args.block_workers))
            with concurrent.futures.ThreadPoolExecutor(max_workers=block_workers) as block_ex:
                while _pid_alive(args.parent_pid):
                    for warm_key, fut in list(inflight.items()):
                        if not fut.done():
                            continue
                        try:
                            ok = bool(fut.result())
                        except Exception as exc:
                            ok = False
                            print(
                                "[CoscliWarmProbe] daemon_block_failed "
                                f"generation={warm_key[0]} block={warm_key[1]} "
                                f"error={type(exc).__name__}: {exc}",
                                flush=True,
                            )
                        if ok:
                            warmed.add(warm_key)
                        inflight.pop(warm_key, None)

                    generation, epoch, current_block, progress_ranks = _progress_state(
                        progress_dir,
                        generation=active_generation,
                        epoch=active_epoch,
                        fallback_block=base_block,
                        world_size=args.world_size,
                        min_progress_ranks=args.min_progress_ranks,
                    )
                    if generation != active_generation or epoch != active_epoch:
                        if inflight:
                            for warm_key, fut in list(inflight.items()):
                                try:
                                    if fut.result():
                                        warmed.add(warm_key)
                                except Exception as exc:
                                    print(
                                        "[CoscliWarmProbe] daemon_block_failed "
                                        f"generation={warm_key[0]} block={warm_key[1]} "
                                        f"error={type(exc).__name__}: {exc}",
                                        flush=True,
                                    )
                            inflight.clear()
                        active_generation = int(generation)
                        active_epoch = int(epoch)
                        base_block = 0
                        _replan_rank_datasets(
                            rank_datasets,
                            epoch=active_epoch,
                            world_size=args.world_size,
                        )
                        print(
                            "[CoscliWarmProbe] daemon_epoch_switch "
                            f"generation={active_generation} epoch={active_epoch}",
                            flush=True,
                        )
                        _cleanup_old_generations(
                            args.ready_dir,
                            keep_generation=active_generation,
                        )
                    max_block_exclusive = base_block + args.blocks if args.blocks > 0 else 10**9
                    if current_block >= max_block_exclusive:
                        print(
                            "[CoscliWarmProbe] daemon_done "
                            f"generation={active_generation} epoch={active_epoch} "
                            f"current_block={current_block} max_block={max_block_exclusive}",
                            flush=True,
                        )
                        return
                    target = min(
                        max_block_exclusive,
                        current_block + max(1, args.lookahead_blocks),
                    )
                    submitted = 0
                    for block_id in range(current_block, target):
                        warm_key = (active_generation, block_id)
                        if warm_key in warmed or _ready_exists(
                            args.ready_dir,
                            generation=active_generation,
                            block_id=block_id,
                        ):
                            warmed.add(warm_key)
                            continue
                        if warm_key in inflight:
                            continue
                        if len(inflight) >= block_workers:
                            break
                        inflight[warm_key] = block_ex.submit(
                            _warm_block,
                            args=args,
                            config=config,
                            stager=stager,
                            rank_datasets=rank_datasets,
                            enabled_datasets=enabled_datasets,
                            generation=active_generation,
                            start_batch=block_id * args.block_batches,
                            block_id=block_id,
                        )
                        submitted += 1
                        print(
                            "[CoscliWarmProbe] daemon_submit "
                            f"generation={active_generation} block={block_id} "
                            f"inflight={len(inflight)}/{block_workers}",
                            flush=True,
                        )
                    if args.max_ready_blocks > 0:
                        keep_from = max(
                            base_block,
                            current_block - max(0, args.max_ready_blocks - 1),
                        )
                        _cleanup_old_ready_markers(
                            args.ready_dir,
                            generation=active_generation,
                            keep_from_block=keep_from,
                        )
                    print(
                        "[CoscliWarmProbe] daemon_poll "
                        f"generation={active_generation} epoch={active_epoch} "
                        f"current_block={current_block} progress_ranks={progress_ranks} "
                        f"target=[{current_block},{target}) warmed={len(warmed)} "
                        f"inflight={len(inflight)} submitted={submitted}",
                        flush=True,
                    )
                    time.sleep(max(0.1, args.poll_s))
        else:
            for block_idx in range(args.blocks):
                block_start = int(args.start_batch) + block_idx * args.block_batches
                _warm_block(
                    args=args,
                    config=config,
                    stager=stager,
                    rank_datasets=rank_datasets,
                    enabled_datasets=enabled_datasets,
                    generation=args.generation,
                    start_batch=block_start,
                    block_id=block_start // args.block_batches,
                )
    finally:
        for _rank, dataset in rank_datasets:
            try:
                dataset.cleanup()
            except Exception:
                pass


if __name__ == "__main__":
    main()
