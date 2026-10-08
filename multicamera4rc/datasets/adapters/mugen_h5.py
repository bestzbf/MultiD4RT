"""Generic single-camera adapter for MUGEN packed HDF5 datasets.

Expected schema::

    /train/<sequence>/{images,depths,intrinsics,extrinsics}
    /test/<sequence>/{images,depths,intrinsics,extrinsics}

Optional authoritative track schema (used by the offline PointOdyssey pack):

    /<split>/<sequence>/{trajs_2d,trajs_3d_world,valids,visibs}
    /<split>/<sequence>/visibility_valids
    /<split>/<sequence>/track_is_dynamic

Images and depths are frame-chunked. Sparse HDF5 fancy indexing is
intentionally avoided because it scans/decompresses intervening chunks on
these multi-terabyte files.
"""

from __future__ import annotations

import os
import random
import threading
import time
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import (
    BaseAdapter,
    UnifiedClip,
    canonical_anchor_frame_index,
    canonical_scene_anchor_metadata,
)


class _WorkerFrameCache:
    """Process-local cache shared by every MUGEN adapter in one worker."""

    def __init__(self) -> None:
        self._pid = os.getpid()
        self._capacity = 0
        self._bytes = 0
        self._items: OrderedDict[tuple[str, str, str, int], np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def configure(self, capacity_bytes: int) -> None:
        self._ensure_pid()
        self._capacity = max(self._capacity, max(0, int(capacity_bytes)))

    def _ensure_pid(self) -> None:
        pid = os.getpid()
        if pid == self._pid:
            return
        self._pid = pid
        self._bytes = 0
        self._items.clear()
        self.hits = self.misses = self.evictions = 0

    def get(self, key: tuple[str, str, str, int]) -> np.ndarray | None:
        with self._lock:
            self._ensure_pid()
            value = self._items.pop(key, None)
            if value is None:
                self.misses += 1
                return None
            self._items[key] = value
            self.hits += 1
            return value

    def put(self, key: tuple[str, str, str, int], value: np.ndarray) -> None:
        size = int(value.nbytes)
        with self._lock:
            self._ensure_pid()
            if self._capacity <= 0 or size > self._capacity:
                return
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= int(old.nbytes)
            self._items[key] = value
            self._bytes += size
            while self._bytes > self._capacity and self._items:
                _, evicted = self._items.popitem(last=False)
                self._bytes -= int(evicted.nbytes)
                self.evictions += 1

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            self._ensure_pid()
            requests = self.hits + self.misses
            return {
                "capacity_bytes": self._capacity,
                "resident_bytes": self._bytes,
                "resident_items": len(self._items),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": self.hits / requests if requests else 0.0,
            }


_WORKER_FRAME_CACHE = _WorkerFrameCache()
_DIRECT_EXECUTOR_PID: int | None = None
_DIRECT_EXECUTOR_WORKERS = 0
_DIRECT_EXECUTOR: ThreadPoolExecutor | None = None
_DIRECT_EXECUTOR_LOCK = threading.Lock()


def _get_direct_executor(workers: int) -> ThreadPoolExecutor:
    """Lazily create one fork-safe decompression pool per DataLoader worker."""
    global _DIRECT_EXECUTOR_PID, _DIRECT_EXECUTOR_WORKERS, _DIRECT_EXECUTOR
    pid = os.getpid()
    with _DIRECT_EXECUTOR_LOCK:
        if (
            _DIRECT_EXECUTOR is None
            or _DIRECT_EXECUTOR_PID != pid
            or _DIRECT_EXECUTOR_WORKERS != workers
        ):
            if _DIRECT_EXECUTOR is not None and _DIRECT_EXECUTOR_PID == pid:
                _DIRECT_EXECUTOR.shutdown(wait=True)
            _DIRECT_EXECUTOR = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="mugen-gzip",
            )
            _DIRECT_EXECUTOR_PID = pid
            _DIRECT_EXECUTOR_WORKERS = workers
        return _DIRECT_EXECUTOR


@dataclass(frozen=True)
class _MugenH5Record:
    sequence_name: str
    group_path: str
    num_frames: int
    height: int
    width: int
    depth_range: tuple[float, float] | None
    pad_pixel: int
    raw_num_frames: int
    pair_count: int
    has_precomputed_tracks: bool
    depth_unit: str


class MugenH5Adapter(BaseAdapter):
    """Read one packed MUGEN H5 as a monocular training dataset.

    Optional static tracks are generated online from depth, intrinsics and
    w2c poses. A track observation is considered valid only when its projected
    depth agrees with that frame's depth map, which suppresses moving objects,
    occlusions and geometry outliers.
    """

    dataset_name = "mugen_h5"

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        dataset_name: str | None = None,
        generate_static_tracks: bool = True,
        prefer_precomputed_tracks: bool = False,
        require_precomputed_tracks: bool = False,
        max_track_points: int = 4096,
        track_seed_frames: int = 8,
        min_track_observations: int = 2,
        track_depth_tolerance_abs: float = 0.01,
        track_depth_tolerance_scene_ratio: float = 0.002,
        track_depth_tolerance_rel: float = 0.01,
        track_seed_near_ratio: float = 0.0,
        track_seed_near_quantile: float = 0.6,
        visibility_radius: int = 1,
        visibility_min_support: int = 2,
        occlusion_min_support: int = 7,
        require_generated_tracks: bool = True,
        min_generated_tracks: int = 256,
        train_only_val_ratio: float = 0.05,
        h5_chunk_cache_mb: int = 128,
        direct_chunk_workers: int = 0,
        frame_cache_mb: int = 0,
        use_pairs_covisibility: bool = True,
        frame_interleave: int = 1,
        camera_stream_index: int = 0,
        strip_padding: bool = False,
        depth_max: float | None = None,
        depth_max_percentile: float | None = None,
        assume_static_scene: bool = True,
        track_query_target_known_ratio: float = 0.0,
        track_query_target_local_ratio: float = 0.0,
        track_query_target_local_radius: int = 0,
        track_query_continuous_visible_ratio: float = 0.0,
        strict: bool = True,
        verbose: bool = True,
        **_: Any,
    ) -> None:
        self.root = Path(root)
        self.split = str(split).lower()
        self.dataset_name = str(dataset_name or f"mugen_{self.root.stem}").lower()
        self.generate_static_tracks = bool(generate_static_tracks)
        self.prefer_precomputed_tracks = bool(
            prefer_precomputed_tracks or require_precomputed_tracks
        )
        self.require_precomputed_tracks = bool(require_precomputed_tracks)
        self.max_track_points = max(1, int(max_track_points))
        self.track_seed_frames = max(1, int(track_seed_frames))
        self.min_track_observations = max(1, int(min_track_observations))
        self.track_depth_tolerance_abs = max(0.0, float(track_depth_tolerance_abs))
        self.track_depth_tolerance_scene_ratio = max(
            0.0, float(track_depth_tolerance_scene_ratio)
        )
        self.track_depth_tolerance_rel = max(0.0, float(track_depth_tolerance_rel))
        self.track_seed_near_ratio = min(max(float(track_seed_near_ratio), 0.0), 1.0)
        self.track_seed_near_quantile = min(
            max(float(track_seed_near_quantile), 0.0), 1.0
        )
        self.visibility_radius = max(0, int(visibility_radius))
        self.visibility_min_support = max(1, int(visibility_min_support))
        self.occlusion_min_support = max(1, int(occlusion_min_support))
        self.require_generated_tracks = bool(require_generated_tracks)
        self.min_generated_tracks = max(1, int(min_generated_tracks))
        self.train_only_val_ratio = min(max(float(train_only_val_ratio), 0.0), 0.5)
        self.h5_chunk_cache_bytes = max(1, int(h5_chunk_cache_mb)) * 1024**2
        self.direct_chunk_workers = max(0, int(direct_chunk_workers))
        self.frame_cache_bytes = max(0, int(frame_cache_mb)) * 1024**2
        _WORKER_FRAME_CACHE.configure(self.frame_cache_bytes)
        self.use_pairs_covisibility = bool(use_pairs_covisibility)
        self.frame_interleave = max(1, int(frame_interleave))
        self.camera_stream_index = int(camera_stream_index)
        if not (0 <= self.camera_stream_index < self.frame_interleave):
            raise ValueError("camera_stream_index must be in [0, frame_interleave)")
        self.strip_padding = bool(strip_padding)
        self.depth_max = self._optional_positive_float(depth_max, "depth_max")
        self.depth_max_percentile = self._optional_percentile(
            depth_max_percentile, "depth_max_percentile"
        )
        self.assume_static_scene = bool(assume_static_scene)
        self.track_query_target_known_ratio = min(
            max(float(track_query_target_known_ratio), 0.0), 1.0
        )
        self.track_query_target_local_ratio = min(
            max(float(track_query_target_local_ratio), 0.0), 1.0
        )
        self.track_query_target_local_radius = max(
            0, int(track_query_target_local_radius)
        )
        self.track_query_continuous_visible_ratio = min(
            max(float(track_query_continuous_visible_ratio), 0.0), 1.0
        )
        self._h5_handle = None
        self._h5_pid: int | None = None
        # Packed multi-camera files may expose only one group-level
        # ``pad_pixel`` even though every interleaved camera has a different
        # letterbox.  Cache the padding inferred for this adapter's selected
        # stream per sequence; the cache contains only small integers.
        self._stream_padding_cache: dict[str, tuple[int, int]] = {}

        if not self.root.is_file():
            if strict:
                raise FileNotFoundError(f"MUGEN H5 file not found: {self.root}")
            self._records = []
            self._name_to_record = {}
            return
        self._records = self._build_index()
        self._name_to_record = {record.sequence_name: record for record in self._records}
        if not self._records and strict:
            raise RuntimeError(f"No compatible sequences found in {self.root} for split={split}")
        if verbose:
            print(
                f"[MugenH5Adapter] name={self.dataset_name} split={self.split} "
                f"sequences={len(self._records)} path={self.root} "
                f"static_tracks={self.generate_static_tracks} "
                f"precomputed_tracks={self.prefer_precomputed_tracks} "
                f"depth_max={self.depth_max} "
                f"depth_max_percentile={self.depth_max_percentile}"
            )

    def __len__(self) -> int:
        return len(self._records)

    def list_sequences(self) -> list[str]:
        return [record.sequence_name for record in self._records]

    def get_sequence_name(self, index: int) -> str:
        return self._records[index].sequence_name

    def get_num_frames(self, sequence_name: str) -> int:
        return self._get_record(sequence_name).num_frames

    def get_canonical_anchor_frame_index(self, sequence_name: str) -> int:
        return canonical_anchor_frame_index(
            self._get_record(sequence_name).num_frames
        )

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        record = self._get_record(sequence_name)
        source_pad_top, source_pad_bottom = self._source_vertical_crop(record)
        crop_top, crop_bottom = (
            (source_pad_top, source_pad_bottom) if self.strip_padding else (0, 0)
        )
        return {
            "dataset_name": self.dataset_name,
            "sequence_name": sequence_name,
            "h5_path": str(self.root),
            "h5_group": record.group_path,
            "num_frames": record.num_frames,
            "raw_num_frames": record.raw_num_frames,
            "frame_interleave": self.frame_interleave,
            "camera_stream_index": self.camera_stream_index,
            "pair_count": record.pair_count,
            "height": record.height - crop_top - crop_bottom,
            "width": record.width,
            "source_height": record.height,
            "source_pad_pixel": min(source_pad_top, source_pad_bottom),
            "source_pad_top": source_pad_top,
            "source_pad_bottom": source_pad_bottom,
            "packed_source_pad_pixel": record.pad_pixel,
            "source_padding_policy": self._source_padding_policy(),
            "padding_stripped": bool(crop_top or crop_bottom),
            "has_depth": True,
            "has_precomputed_tracks": bool(record.has_precomputed_tracks),
            "has_tracks": bool(
                record.has_precomputed_tracks and self.prefer_precomputed_tracks
            ) or self.generate_static_tracks,
            "has_visibility": bool(
                record.has_precomputed_tracks and self.prefer_precomputed_tracks
            ) or self.generate_static_tracks,
            "extrinsics_convention": "w2c",
            "depth_unit": record.depth_unit,
            "depth_max": self.depth_max,
            "depth_max_percentile": self.depth_max_percentile,
        }

    def load_clip(
        self,
        sequence_name: str,
        frame_indices: list[int],
        *,
        canonical_anchor_required: bool = False,
    ) -> UnifiedClip:
        load_started = time.perf_counter()
        if not frame_indices:
            raise ValueError("MugenH5Adapter.load_clip requires non-empty frame_indices")
        record = self._get_record(sequence_name)
        clipped_indices = [min(max(int(i), 0), record.num_frames - 1) for i in frame_indices]
        raw_indices = [
            index * self.frame_interleave + self.camera_stream_index
            for index in clipped_indices
        ]
        group = self._get_h5()[record.group_path]

        # Read each unique frame chunk directly. h5py ds[sparse_indices] is up
        # to an order of magnitude slower for these files.
        unique_indices = list(dict.fromkeys(raw_indices))
        cache_before = _WORKER_FRAME_CACHE.snapshot()
        payload_started = time.perf_counter()
        image_values, image_direct = self._read_frame_dataset(
            group["images"], record.group_path, "images", unique_indices, np.uint8
        )
        depth_values, depth_direct = self._read_frame_dataset(
            group["depths"], record.group_path, "depths", unique_indices, np.float32
        )
        source_pad_top, source_pad_bottom = self._source_vertical_crop(
            record,
            images=[image_values[index] for index in unique_indices],
            depths=[depth_values[index] for index in unique_indices],
        )
        crop_top, crop_bottom = (
            (source_pad_top, source_pad_bottom) if self.strip_padding else (0, 0)
        )
        # When letterboxing is retained, depth-derived points must still stay
        # inside the selected camera stream's physical image.  When it is
        # stripped, coordinates already live in the cropped view.
        track_vertical_padding = (
            (0, 0)
            if crop_top or crop_bottom
            else (source_pad_top, source_pad_bottom)
        )
        payload_seconds = time.perf_counter() - payload_started

        frame_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        depth_filter_thresholds: dict[int, float | None] = {}
        depth_filter_removed_pixels = 0
        depth_sanitized_pixels = 0
        for frame_index in unique_indices:
            image = image_values[frame_index]
            depth = depth_values[frame_index]
            intrinsic = np.asarray(group["intrinsics"][frame_index], dtype=np.float32)
            if crop_top or crop_bottom:
                stop = image.shape[0] - crop_bottom if crop_bottom else image.shape[0]
                image = np.ascontiguousarray(image[crop_top:stop])
                depth = np.ascontiguousarray(depth[crop_top:stop])
                intrinsic = intrinsic.copy()
                intrinsic[1, 2] -= float(crop_top)
            depth, filter_stats = self._filter_depth(depth)
            depth_filter_thresholds[frame_index] = filter_stats["percentile_threshold"]
            depth_filter_removed_pixels += int(filter_stats["filtered_positive_pixels"])
            depth_sanitized_pixels += int(filter_stats["sanitized_pixels"])
            frame_cache[frame_index] = (
                image,
                depth,
                intrinsic,
                np.asarray(group["extrinsics"][frame_index], dtype=np.float32),
            )

        images = [frame_cache[index][0] for index in raw_indices]
        depths = [frame_cache[index][1] for index in raw_indices]
        intrinsics = np.stack([frame_cache[index][2] for index in raw_indices]).astype(
            np.float32, copy=False
        )
        extrinsics = np.stack([frame_cache[index][3] for index in raw_indices]).astype(
            np.float32, copy=False
        )

        trajs_2d = trajs_3d_world = valids = visibs = visibility_valids = None
        track_is_dynamic = None
        track_geometry_support_policy = None
        track_stats: dict[str, Any] = {}
        use_precomputed_tracks = (
            self.prefer_precomputed_tracks and record.has_precomputed_tracks
        )
        if self.require_precomputed_tracks and not use_precomputed_tracks:
            raise RuntimeError(
                f"[{self.dataset_name}/{sequence_name}] packed H5 has no "
                "authoritative precomputed tracks"
            )
        if use_precomputed_tracks:
            # PointOdyssey and other offline converters store authoritative
            # trajectories in the same frame-chunked H5 group.  Read only the
            # requested chunks and never regenerate tracks from depth.
            track_arrays = {
                key: [
                    np.asarray(group[key][frame_index])
                    for frame_index in dict.fromkeys(raw_indices)
                ]
                for key in ("trajs_2d", "trajs_3d_world", "valids", "visibs")
            }
            unique_positions = list(dict.fromkeys(raw_indices))
            position_to_value = {
                key: {
                    frame_index: value
                    for frame_index, value in zip(unique_positions, values)
                }
                for key, values in track_arrays.items()
            }
            trajs_2d = np.stack(
                [position_to_value["trajs_2d"][frame_index] for frame_index in raw_indices]
            ).astype(np.float32, copy=False)
            if crop_top or crop_bottom:
                # Precomputed tracks use packed-image pixel coordinates just
                # like K. Keep them aligned with the cropped RGB/depth view.
                trajs_2d = trajs_2d.copy()
                trajs_2d[..., 1] -= float(crop_top)
            trajs_3d_world = np.stack(
                [
                    position_to_value["trajs_3d_world"][frame_index]
                    for frame_index in raw_indices
                ]
            ).astype(np.float32, copy=False)
            valids = np.stack(
                [position_to_value["valids"][frame_index] for frame_index in raw_indices]
            ).astype(bool, copy=False)
            visibs = np.stack(
                [position_to_value["visibs"][frame_index] for frame_index in raw_indices]
            ).astype(bool, copy=False)
            if "visibility_valids" in group:
                visibility_values = [
                    np.asarray(group["visibility_valids"][frame_index])
                    for frame_index in dict.fromkeys(raw_indices)
                ]
                visibility_by_frame = {
                    frame_index: value
                    for frame_index, value in zip(unique_positions, visibility_values)
                }
                visibility_valids = np.stack(
                    [visibility_by_frame[frame_index] for frame_index in raw_indices]
                ).astype(bool, copy=False)
            if "track_is_dynamic" in group:
                track_is_dynamic = np.asarray(group["track_is_dynamic"][()]).astype(
                    bool, copy=False
                )
            track_stats = {
                "precomputed_track_points": int(trajs_2d.shape[1]),
                "track_source": "precomputed_h5",
            }
            track_geometry_support_policy = "authoritative_world_trajectory_v1"
        elif self.generate_static_tracks:
            rng = random.Random(self._sample_seed(sequence_name, clipped_indices))
            generated = self._generate_tracks(
                depths,
                intrinsics,
                extrinsics,
                rng,
                record,
                vertical_padding=track_vertical_padding,
            )
            if generated is not None:
                (
                    trajs_2d,
                    trajs_3d_world,
                    valids,
                    visibs,
                    visibility_valids,
                    track_stats,
                ) = generated
                if not self.assume_static_scene:
                    # Depth/pose generation creates one fixed world point; it
                    # does not create an authoritative object trajectory.  On
                    # a non-static source, a closer target surface proves only
                    # an occlusion label and must never legitimise that fixed
                    # world point as target-frame geometry.  Keep geometry
                    # only where target depth positively matches the point.
                    # ``visibility_valids`` deliberately remains broader so
                    # confirmed occlusions are still useful BCE negatives.
                    valids = visibs.copy()
                    track_geometry_support_policy = (
                        "target_depth_visible_exact_observation_v1"
                    )
                else:
                    track_geometry_support_policy = (
                        "assumed_static_world_projection_v1"
                    )
                track_stats.update(
                    {
                        "generated_track_geometry_support_ratio": float(
                            valids.mean()
                        ),
                        "generated_track_geometry_support_count": int(
                            valids.sum()
                        ),
                        "generated_track_depth_confirmed_geometry_count": int(
                            (valids & visibs & visibility_valids).sum()
                        ),
                        "generated_track_visibility_negative_count": int(
                            (visibility_valids & ~visibs).sum()
                        ),
                        "generated_track_visibility_negative_geometry_count": int(
                            (valids & visibility_valids & ~visibs).sum()
                        ),
                    }
                )

        has_tracks = trajs_2d is not None
        generated_track_count = int(track_stats.get("generated_track_points", 0))
        if (
            not use_precomputed_tracks
            and self.generate_static_tracks
            and self.require_generated_tracks
            and generated_track_count < self.min_generated_tracks
        ):
            raise RuntimeError(
                f"[{self.dataset_name}/{sequence_name}] only {generated_track_count} "
                f"reliable static tracks, require >= {self.min_generated_tracks}"
            )
        metadata = {
            "dataset_name": self.dataset_name,
            "sequence_root": str(self.root),
            "h5_path": str(self.root),
            "h5_group": record.group_path,
            "frame_indices": clipped_indices,
            "raw_frame_indices": raw_indices,
            "frame_interleave": int(self.frame_interleave),
            "camera_stream_index": int(self.camera_stream_index),
            "assume_static_scene": bool(self.assume_static_scene),
            "track_query_target_known_ratio": self.track_query_target_known_ratio,
            "track_query_target_local_ratio": self.track_query_target_local_ratio,
            "track_query_target_local_radius": self.track_query_target_local_radius,
            "track_query_continuous_visible_ratio": self.track_query_continuous_visible_ratio,
            "num_frames_in_sequence": record.num_frames,
            "raw_image_hw": [record.height, record.width],
            "rgb_hw": [record.height - crop_top - crop_bottom, record.width],
            "target_hw": [record.height - crop_top - crop_bottom, record.width],
            "source_pad_pixel": int(min(source_pad_top, source_pad_bottom)),
            "source_pad_top": int(source_pad_top),
            "source_pad_bottom": int(source_pad_bottom),
            "packed_source_pad_pixel": int(record.pad_pixel),
            "source_padding_policy": self._source_padding_policy(),
            "pad_pixel": (
                0
                if crop_top or crop_bottom
                else int(min(source_pad_top, source_pad_bottom))
            ),
            "valid_image_y_bounds": [
                int(track_vertical_padding[0]),
                int(record.height - track_vertical_padding[1])
                if not (crop_top or crop_bottom)
                else int(record.height - crop_top - crop_bottom),
            ],
            "padding_stripped": bool(crop_top or crop_bottom),
            "depth_range": list(record.depth_range) if record.depth_range else None,
            # These filters are applied online after optional padding removal and
            # before depth-derived track generation; the packed H5 is unchanged.
            "depth_filter": {
                "max_depth": self.depth_max,
                "max_percentile": self.depth_max_percentile,
                "percentile_population": "all_pixels_including_zero",
                "application_stage": "after_padding_before_track_generation",
                "per_raw_frame_percentile_threshold": {
                    str(index): depth_filter_thresholds[index]
                    for index in unique_indices
                },
                "filtered_positive_pixels": depth_filter_removed_pixels,
                "sanitized_pixels": depth_sanitized_pixels,
            },
            "has_depth": True,
            "has_normals": False,
            "has_tracks": has_tracks,
            "has_visibility": has_tracks,
            "has_trajs_3d_world": has_tracks,
            # Explicitly separate the mask that authorises 3-D/trajectory
            # losses from the mask that authorises visible/invisible BCE.
            # Query construction and the strict GT audit fail closed on this
            # versioned contract for generated tracks in non-static sources.
            "track_geometry_support_policy": track_geometry_support_policy,
            "track_geometry_support_requires_exact_observation": bool(
                track_geometry_support_policy
                == "target_depth_visible_exact_observation_v1"
            ),
            "track_geometry_support_mask": (
                "visible_and_target_depth_confirmed"
                if track_geometry_support_policy
                == "target_depth_visible_exact_observation_v1"
                else "adapter_valids"
                if has_tracks
                else None
            ),
            "pose_convention": "w2c",
            "extrinsics_convention": "w2c",
            "intrinsics_convention": "pinhole_pixel",
            "depth_unit": record.depth_unit,
            "track_source": (
                "precomputed_h5"
                if use_precomputed_tracks
                else "online_depth_pose_static"
                if has_tracks
                else None
            ),
            "precomputed_tracks": bool(use_precomputed_tracks),
            "infer_dynamic_from_authoritative_world_trajectories": bool(
                use_precomputed_tracks and track_is_dynamic is None
            ),
            "dynamic_motion_label_schema": (
                "adapter_explicit_binary_with_known_mask_v1"
                if track_is_dynamic is not None
                else (
                    "world_displacement_three_state_strict_static_unknown_band_"
                    "distinct_time_v2"
                    if use_precomputed_tracks
                    else None
                )
            ),
            "_load_timing": {
                "frame_payload_s": payload_seconds,
                "direct_chunk_images": image_direct,
                "direct_chunk_depths": depth_direct,
                "total_s": time.perf_counter() - load_started,
            },
            "_frame_cache": self._cache_delta(cache_before),
            **track_stats,
        }
        if canonical_anchor_required:
            anchor_frame_index = self.get_canonical_anchor_frame_index(sequence_name)
            anchor_raw_index = (
                anchor_frame_index * self.frame_interleave
                + self.camera_stream_index
            )
            if not 0 <= anchor_raw_index < record.raw_num_frames:
                raise RuntimeError(
                    f"[{self.dataset_name}/{sequence_name}] canonical anchor raw "
                    f"index {anchor_raw_index} is outside "
                    f"{record.raw_num_frames} frames"
                )
            anchor_matches = [
                slot
                for slot, logical_index in enumerate(clipped_indices)
                if logical_index == anchor_frame_index
            ]
            if len(anchor_matches) != 1:
                raise RuntimeError(
                    f"[{self.dataset_name}/{sequence_name}] canonical anchor "
                    "must occur exactly once in the clip; got "
                    f"{len(anchor_matches)} occurrences"
                )
            # Read the authoritative row from the full sequence only for the
            # Stage-1 path.  Ordinary V9 loading keeps its previous I/O and
            # cannot be blocked by an unused off-clip pose.
            anchor_extrinsic = np.asarray(
                group["extrinsics"][anchor_raw_index], dtype=np.float32
            )
            metadata.update(
                canonical_scene_anchor_metadata(
                    extrinsic=anchor_extrinsic,
                    time=float(anchor_frame_index),
                    camera_id=0,
                    frame_id=(
                        f"{self.dataset_name}/{record.group_path}/"
                        f"stream={self.camera_stream_index}/"
                        f"logical={anchor_frame_index}/raw={anchor_raw_index}"
                    ),
                    anchor_slot=anchor_matches[0],
                    num_slots=len(clipped_indices),
                    require_in_clip=True,
                )
            )
        return UnifiedClip(
            dataset_name=self.dataset_name,
            sequence_name=sequence_name,
            frame_paths=[f"{self.root}#{record.group_path}/frame={i}" for i in raw_indices],
            images=images,
            depths=depths,
            normals=None,
            trajs_2d=trajs_2d,
            trajs_3d_world=trajs_3d_world,
            valids=valids,
            visibs=visibs,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            metadata=metadata,
            visibility_valids=visibility_valids,
            track_is_dynamic=track_is_dynamic,
            track_dynamic_known=(
                np.ones_like(track_is_dynamic, dtype=bool)
                if track_is_dynamic is not None
                else None
            ),
        )

    def _read_frame_dataset(
        self,
        dataset: Any,
        group_path: str,
        dataset_name: str,
        frame_indices: list[int],
        dtype: Any,
    ) -> tuple[dict[int, np.ndarray], bool]:
        values: dict[int, np.ndarray] = {}
        missing: list[int] = []
        root_key = str(self.root)
        for frame_index in frame_indices:
            key = (root_key, group_path, dataset_name, int(frame_index))
            value = _WORKER_FRAME_CACHE.get(key) if self.frame_cache_bytes else None
            if value is None:
                missing.append(frame_index)
            else:
                values[frame_index] = value

        used_direct = False
        if missing and self._supports_direct_gzip(dataset):
            try:
                raw_chunks = [
                    dataset.id.read_direct_chunk(
                        (int(frame_index),) + (0,) * (dataset.ndim - 1)
                    )
                    for frame_index in missing
                ]
                decode_args = [
                    (filter_mask, raw, dataset.dtype, dataset.shape[1:])
                    for filter_mask, raw in raw_chunks
                ]
                if self.direct_chunk_workers > 1 and len(decode_args) > 1:
                    executor = _get_direct_executor(self.direct_chunk_workers)
                    decoded = list(executor.map(self._decode_gzip_chunk, decode_args))
                else:
                    decoded = [self._decode_gzip_chunk(args) for args in decode_args]
                values.update(zip(missing, decoded))
                used_direct = True
            except (OSError, RuntimeError, ValueError, zlib.error):
                # Unknown filters or malformed chunks retain the proven h5py path.
                used_direct = False

        for frame_index in missing:
            if frame_index not in values:
                values[frame_index] = np.ascontiguousarray(
                    dataset[frame_index], dtype=dtype
                )
            else:
                values[frame_index] = np.ascontiguousarray(
                    values[frame_index], dtype=dtype
                )
            if self.frame_cache_bytes:
                key = (root_key, group_path, dataset_name, int(frame_index))
                _WORKER_FRAME_CACHE.put(key, values[frame_index])
        return values, used_direct

    @staticmethod
    def _optional_positive_float(value: Any, name: str) -> float | None:
        if value is None:
            return None
        parsed = float(value)
        if not np.isfinite(parsed) or parsed <= 0:
            raise ValueError(f"{name} must be a finite positive number or None")
        return parsed

    @staticmethod
    def _optional_percentile(value: Any, name: str) -> float | None:
        if value is None:
            return None
        parsed = float(value)
        if not np.isfinite(parsed) or not (0 < parsed <= 100):
            raise ValueError(f"{name} must be in (0, 100] or None")
        return parsed

    def _filter_depth(self, depth: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply VGGT-style upper-depth filtering without mutating H5 cache data.

        VGGT computes Co3D's percentile over the complete masked depth image,
        including zero-valued invalid pixels. It first applies an absolute cap
        when configured (Virtual KITTI uses 80 m), then applies the percentile
        cap. The output stays float32 and invalid samples use the repository's
        canonical zero representation.
        """
        source = np.asarray(depth, dtype=np.float32)
        if self.depth_max is None and self.depth_max_percentile is None:
            return source, {
                "percentile_threshold": None,
                "filtered_positive_pixels": 0,
                "sanitized_pixels": 0,
            }

        filtered = source.copy()
        valid_before = np.isfinite(filtered) & (filtered > 0)
        invalid = ~valid_before
        sanitized_pixels = int(np.count_nonzero(invalid & (filtered != 0)))
        filtered[invalid] = 0.0

        if self.depth_max is not None:
            filtered[filtered > self.depth_max] = 0.0

        percentile_threshold: float | None = None
        if self.depth_max_percentile is not None:
            # Match VGGT's threshold_depth_map semantics: zeros participate in
            # nanpercentile, and a zero threshold is deliberately not applied.
            threshold = float(np.nanpercentile(filtered, self.depth_max_percentile))
            if np.isfinite(threshold):
                percentile_threshold = threshold
                if threshold > 0:
                    filtered[filtered > threshold] = 0.0

        valid_after = filtered > 0
        return filtered, {
            "percentile_threshold": percentile_threshold,
            "filtered_positive_pixels": int(
                np.count_nonzero(valid_before & ~valid_after)
            ),
            "sanitized_pixels": sanitized_pixels,
        }

    def _supports_direct_gzip(self, dataset: Any) -> bool:
        creation = dataset.id.get_create_plist()
        if creation.get_nfilters() != 1:
            return False
        filter_id, _, _, _ = creation.get_filter(0)
        return bool(
            self.direct_chunk_workers > 0
            and dataset.chunks == (1,) + tuple(dataset.shape[1:])
            and dataset.compression == "gzip"
            and int(filter_id) == 1  # H5Z_FILTER_DEFLATE
            and dataset.dtype.kind not in {"O", "V"}
        )

    @staticmethod
    def _decode_gzip_chunk(args: tuple[int, bytes, np.dtype, tuple[int, ...]]) -> np.ndarray:
        filter_mask, raw, dtype, shape = args
        if filter_mask == 0:
            payload = zlib.decompress(raw)
        elif filter_mask == 1:
            payload = raw
        else:
            raise ValueError(f"Unsupported HDF5 gzip filter mask: {filter_mask}")
        expected = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
        if len(payload) != expected:
            raise ValueError(
                f"Decoded HDF5 chunk has {len(payload)} bytes, expected {expected}"
            )
        return np.frombuffer(payload, dtype=dtype).reshape(shape).copy()

    @staticmethod
    def _cache_delta(before: dict[str, int | float]) -> dict[str, int | float]:
        after = _WORKER_FRAME_CACHE.snapshot()
        return {
            **after,
            "sample_hits": int(after["hits"]) - int(before["hits"]),
            "sample_misses": int(after["misses"]) - int(before["misses"]),
            "sample_evictions": int(after["evictions"]) - int(before["evictions"]),
        }

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        record = self._get_record(sequence_name)
        count = min(4, record.num_frames)
        clip = self.load_clip(sequence_name, list(range(count)))
        return {
            "ok": len(clip.images) == count,
            "sequence_name": sequence_name,
            "num_frames": record.num_frames,
            "image_shape": list(clip.images[0].shape),
            "valid_depth_ratio": float(np.mean(np.asarray(clip.depths[0]) > 0)),
            "has_tracks": bool(clip.metadata["has_tracks"]),
            "track_points": int(
                clip.metadata.get(
                    "generated_track_points",
                    clip.metadata.get("precomputed_track_points", 0),
                )
            ),
        }

    def _build_index(self) -> list[_MugenH5Record]:
        import h5py

        with h5py.File(self.root, "r") as handle:
            available = set(handle.keys())
            if self.split == "train":
                group_name = "train" if "train" in available else None
            else:
                group_name = "test" if "test" in available else "train" if "train" in available else None
            if group_name is None:
                return []
            names = sorted(handle[group_name].keys())
            if "test" not in available and group_name == "train" and self.train_only_val_ratio > 0:
                cut = max(1, int(round(len(names) * (1.0 - self.train_only_val_ratio))))
                names = names[:cut] if self.split == "train" else names[cut:]
            records: list[_MugenH5Record] = []
            for name in names:
                group = handle[f"{group_name}/{name}"]
                required = {"images", "depths", "intrinsics", "extrinsics"}
                if not required.issubset(group.keys()):
                    continue
                shape = group["images"].shape
                if len(shape) != 4 or shape[-1] != 3 or shape[0] < 1:
                    continue
                depth_range_raw = group.attrs.get("depth_range")
                depth_range = None
                if depth_range_raw is not None and len(depth_range_raw) >= 2:
                    depth_range = (float(depth_range_raw[0]), float(depth_range_raw[1]))
                raw_num_frames = int(shape[0])
                logical_num_frames = (
                    max(0, (raw_num_frames - 1 - self.camera_stream_index) // self.frame_interleave + 1)
                )
                pair_count = int(group["pairs"].shape[0]) if "pairs" in group else 0
                precomputed_track_keys = {
                    "trajs_2d",
                    "trajs_3d_world",
                    "valids",
                    "visibs",
                }
                has_precomputed_tracks = precomputed_track_keys.issubset(group.keys())
                sequence_name = group.attrs.get("sequence_name", name)
                if isinstance(sequence_name, bytes):
                    sequence_name = sequence_name.decode("utf-8")
                depth_unit = group.attrs.get("depth_unit", "scene_units")
                if isinstance(depth_unit, bytes):
                    depth_unit = depth_unit.decode("utf-8")
                if logical_num_frames < 1:
                    continue
                records.append(
                    _MugenH5Record(
                        sequence_name=str(sequence_name),
                        group_path=f"{group_name}/{name}",
                        num_frames=logical_num_frames,
                        height=int(shape[1]),
                        width=int(shape[2]),
                        depth_range=depth_range,
                        pad_pixel=int(group.attrs.get("pad_pixel", 0)),
                        raw_num_frames=raw_num_frames,
                        pair_count=pair_count,
                        has_precomputed_tracks=has_precomputed_tracks,
                        depth_unit=str(depth_unit),
                    )
                )
        return records

    def get_covisibility_neighbors(
        self, sequence_name: str, frame_index: int
    ) -> Optional[list[int]]:
        """Return packed H5 ``pairs`` neighbors when available."""
        if not self.use_pairs_covisibility or self.frame_interleave != 1:
            return None
        record = self._get_record(sequence_name)
        if record.pair_count <= 0 or not (0 <= int(frame_index) < record.pair_count):
            return None
        raw = np.asarray(
            self._get_h5()[record.group_path]["pairs"][int(frame_index)],
            dtype=np.int64,
        ).reshape(-1)
        valid = raw[(raw >= 0) & (raw < record.num_frames) & (raw != int(frame_index))]
        return list(dict.fromkeys(int(value) for value in valid.tolist()))

    def get_covisibility_anchor_count(self, sequence_name: str) -> Optional[int]:
        if not self.use_pairs_covisibility or self.frame_interleave != 1:
            return None
        record = self._get_record(sequence_name)
        return record.pair_count if record.pair_count > 0 else None

    def get_static_covisibility_neighbors(self, sequence_name, frame_index, admitted_frames):
        """Verify packed neighbors against measured depth in a common world.

        Some Taskonomy pair rows include opposite views with little overlap.
        A small diagnostic grid ranks/rejects those views; it does not sample
        or replace the dense training labels.
        """
        neighbors = self.get_covisibility_neighbors(sequence_name, frame_index)
        if neighbors is None:
            return None
        allowed = set(admitted_frames)
        neighbors = [i for i in neighbors if i in allowed]
        group = self._get_h5()[self._get_record(sequence_name).group_path]
        depth, _ = self._filter_depth(np.asarray(group["depths"][frame_index], np.float32))
        K = np.asarray(group["intrinsics"][frame_index], np.float64).copy()
        K[:2, 2] -= .5  # packed legacy_h5 pixel centers -> array indices
        E = np.asarray(group["extrinsics"][frame_index], np.float64)
        yy, xx = np.mgrid[0:depth.shape[0]:16, 0:depth.shape[1]:16]
        z = depth[yy.ravel(), xx.ravel()]
        valid = np.isfinite(z) & (z > 0)
        pixels = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)], -1)[valid]
        camera = (pixels @ np.linalg.inv(K).T)*z[valid, None]
        world = (camera-E[:3, 3]) @ E[:3, :3]
        center = -E[:3, :3].T @ E[:3, 3]
        baseline_floor = .01*float(np.median(z[valid])) if len(world) else 0.
        minimum = max(1, int(np.ceil(.03*len(world))))
        if not len(world):
            return []
        ranked = []
        for order, i in enumerate(neighbors):
            target_depth, _ = self._filter_depth(np.asarray(group["depths"][i], np.float32))
            target_K = np.asarray(group["intrinsics"][i], np.float64).copy()
            target_K[:2, 2] -= .5
            target_E = np.asarray(group["extrinsics"][i], np.float64)
            xyz = world @ target_E[:3, :3].T+target_E[:3, 3]
            projected = xyz @ target_K.T
            uv = projected[:, :2]/np.maximum(projected[:, 2:], 1e-10)
            h, w = target_depth.shape
            inside = ((xyz[:, 2] > 0) & np.isfinite(uv).all(-1)
                      & (uv[:, 0] >= 0) & (uv[:, 0] < w-.5)
                      & (uv[:, 1] >= 0) & (uv[:, 1] < h-.5))
            x, y = np.rint(uv[inside]).astype(np.int64).T
            measured = target_depth[y, x]
            matches = int(((measured > 0) & np.isfinite(measured)
                           & (abs(xyz[inside, 2]-measured) <= .05*measured)).sum())
            if matches >= minimum:
                target_center = -target_E[:3, :3].T @ target_E[:3, 3]
                translated = np.linalg.norm(target_center-center) >= baseline_floor
                # Prefer useful parallax when it exists; retain valid pure
                # rotation views when the packed neighborhood lacks baseline.
                ranked.append((-int(translated), -matches, order, i))
        return [i for _, _, _, i in sorted(ranked)]

    def _get_h5(self):
        import h5py

        pid = os.getpid()
        if self._h5_handle is None or self._h5_pid != pid:
            self._close_h5()
            self._h5_handle = h5py.File(
                self.root,
                "r",
                rdcc_nbytes=self.h5_chunk_cache_bytes,
                rdcc_nslots=10007,
            )
            self._h5_pid = pid
        return self._h5_handle

    def _close_h5(self) -> None:
        # Construction can fail before the lazy HDF5 fields are initialized;
        # make destructor/cleanup safe on partially initialized adapters.
        handle = getattr(self, "_h5_handle", None)
        self._h5_handle = None
        self._h5_pid = None
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_h5_handle"] = None
        state["_h5_pid"] = None
        return state

    def __del__(self):
        self._close_h5()

    def _get_record(self, sequence_name: str) -> _MugenH5Record:
        try:
            return self._name_to_record[str(sequence_name)]
        except KeyError as exc:
            raise KeyError(f"Unknown sequence {sequence_name!r} in {self.root}") from exc

    def _source_padding_policy(self) -> str:
        return "conservative_rgb_letterbox" if self.strip_padding else "retained"

    def _source_vertical_crop(
        self,
        record: _MugenH5Record,
        *,
        images: list[np.ndarray] | None = None,
        depths: list[np.ndarray] | None = None,
    ) -> tuple[int, int]:
        """Return top/bottom letterbox rows for the selected camera stream.

        Some nominally single-camera packs contain a one-row discrepancy
        between ``pad_pixel`` and actual content, while some multi-camera packs
        (notably five-stream Waymo) retained only the value from the last
        camera written into the group. Reusing the attribute blindly can crop
        real RGB/depth content and shift ``cy`` by the wrong amount.

        For an interleaved stream, infer the constant-white letterbox from an
        actual selected-stream RGB frame.  RGB is the authority here because
        sparse LiDAR depth legitimately has empty rows inside the real view.
        Multiple loaded frames are intersected conservatively: a row is
        considered padding only when it is white in every candidate.  This
        avoids treating a genuine all-white scene row in one frame as padding.
        The result is cached per sequence and never reads another camera.
        Cached crops are conservative upper bounds: later loaded RGB/depth
        evidence may only reduce either side.  In particular, the group-level
        value returned by single-camera ``get_sequence_info`` is provisional
        and must not prevent a later clip from correcting an off-by-one packer
        boundary.
        """
        cached = self._stream_padding_cache.get(record.sequence_name)
        if cached is not None and images is None:
            return cached

        def remember(result: tuple[int, int]) -> tuple[int, int]:
            normalized = (max(0, int(result[0])), max(0, int(result[1])))
            if cached is not None:
                normalized = (
                    min(int(cached[0]), normalized[0]),
                    min(int(cached[1]), normalized[1]),
                )
            self._stream_padding_cache[record.sequence_name] = normalized
            return normalized

        packed = min(max(int(record.pad_pixel), 0), record.height // 2)
        packed_fallback = (
            (packed, packed) if record.height - 2 * packed > 0 else (0, 0)
        )
        # Padding inference requires the packer's exact-white border. A
        # multi-camera group attribute may describe a different stream.
        # Keeping the full image is the only fail-closed fallback there.
        # For single-camera data the attribute remains useful when an RGB frame
        # itself provides no evidence (for example an all-white capture).
        stream_fallback = (0, 0)
        fallback = packed_fallback if self.frame_interleave <= 1 else stream_fallback

        # Most historical single-camera fixtures/packs rely on the explicit
        # attribute and do not store exact-white borders. Read RGB only when a
        # loaded clip provides direct evidence that the declared crop contains
        # non-white content or positive depth; otherwise preserve the contract.
        if self.frame_interleave <= 1 and images is None:
            # This metadata-only answer is deliberately not cached.  Dataset
            # indexing calls get_sequence_info before any payload is decoded;
            # caching here used to lock Co3D's declared 123-row crop and stop a
            # later RGB/depth-backed clip from safely tightening it to 122.
            return fallback
        if self.frame_interleave <= 1 and images:
            rgb_border_evidence = False
            for image in images:
                array = np.asarray(image)
                if array.ndim != 3 or array.shape[:2] != (record.height, record.width):
                    continue
                white_rows = np.all(array == 255, axis=(1, 2))
                if packed and (
                    white_rows[: max(1, packed - 1)].all()
                    and white_rows[-max(1, packed - 1) :].all()
                ):
                    rgb_border_evidence = True
                    break
            if not rgb_border_evidence:
                return remember(fallback)

        candidates = images
        if candidates is None:
            group = self._get_h5()[record.group_path]
            raw_index = self.camera_stream_index
            if not (0 <= raw_index < record.raw_num_frames):
                return remember(fallback)
            candidates = [np.asarray(group["images"][raw_index])]

        row_is_padding: np.ndarray | None = None
        for image in candidates:
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[:2] != (record.height, record.width):
                continue
            # The packer uses exact uint8 white letterboxing. Requiring every
            # pixel and channel to equal 255 is deliberately fail-closed.
            current = np.all(array == 255, axis=(1, 2))
            row_is_padding = (
                current
                if row_is_padding is None
                else row_is_padding & current
            )

        if row_is_padding is None:
            return remember(fallback)
        non_padding = np.flatnonzero(~row_is_padding)
        if not non_padding.size:
            # An all-white selected-stream frame provides no evidence. Never
            # substitute another stream's group attribute and risk deleting
            # content; retain the complete image instead.
            return remember(fallback)
        top = int(non_padding[0])
        bottom = int(record.height - non_padding[-1] - 1)
        if top + bottom >= record.height:
            top, bottom = fallback

        # The packer vertically centers each resized camera image. A real
        # scene can itself contain one or more complete white rows at one edge
        # (observed in a Waymo first frame: 113 white rows above versus the
        # true 93-row letterbox below).  Only the padding supported by both
        # sides is therefore safe to remove.  This also deliberately retains
        # one harmless row for odd-height 93/94 letterboxes instead of ever
        # cropping a content row.
        symmetric_pad = min(top, bottom)
        top = bottom = symmetric_pad

        # Positive depth is strong evidence for physical content. If it exists
        # outside the RGB-inferred crop, widen the crop rather than discarding
        # supervision. This also handles an unusual white image border safely.
        if depths:
            first_positive = record.height
            last_positive = -1
            for depth in depths:
                array = np.asarray(depth)
                if array.ndim != 2 or array.shape != (record.height, record.width):
                    continue
                rows = np.flatnonzero(
                    np.any(np.isfinite(array) & (array > 1e-4), axis=1)
                )
                if rows.size:
                    first_positive = min(first_positive, int(rows[0]))
                    last_positive = max(last_positive, int(rows[-1]))
            if last_positive >= 0:
                top = min(top, first_positive)
                bottom = min(bottom, record.height - last_positive - 1)

        return remember((top, bottom))

    @staticmethod
    def _sample_seed(sequence_name: str, frame_indices: list[int]) -> int:
        import hashlib

        payload = f"{sequence_name}:{','.join(map(str, frame_indices))}".encode()
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")

    def _generate_tracks(
        self,
        depths: list[np.ndarray],
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        rng: random.Random,
        record: _MugenH5Record,
        *,
        vertical_padding: tuple[int, int] = (0, 0),
    ) -> Optional[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            dict[str, Any],
        ]
    ]:
        """Generate static world tracks from depth and calibrated poses.

        Depth maps in the packed datasets are not guaranteed to be fully
        finite.  In particular, invalid pixels are represented by NaN/Inf in
        some sequences.  Never use those values in the pinhole equations:
        besides producing unusable labels, expressions such as ``0 * NaN``
        emit a ``RuntimeWarning`` and can make it difficult to distinguish a
        real calibration problem from an expected missing-depth pixel.

        The implementation therefore uses a fail-closed policy.  Invalid
        depth/K/pose rows remain unknown, their output coordinates are kept
        finite sentinels, and only rows passing all validity checks are
        back-projected or compared against a depth sample.
        """
        num_frames = len(depths)
        seed_count = min(self.track_seed_frames, num_frames)
        seed_frames = np.linspace(0, num_frames - 1, seed_count, dtype=np.int64)
        points_per_seed = max(1, int(np.ceil(self.max_track_points / seed_count)))
        point_sets: list[np.ndarray] = []
        np_rng = np.random.default_rng(rng.getrandbits(64))

        # A pinhole matrix with a non-finite or zero focal length cannot
        # produce a trustworthy 3-D point.  Checking the complete matrix also
        # catches corrupted principal points before they enter a multiply.
        def valid_intrinsic(matrix: np.ndarray) -> bool:
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                return False
            return (
                abs(float(matrix[0, 0])) > 1e-8
                and abs(float(matrix[1, 1])) > 1e-8
            )

        def safe_inverse(matrix: np.ndarray) -> np.ndarray | None:
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                return None
            try:
                with np.errstate(all="ignore"):
                    inverse = np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                return None
            return inverse if np.isfinite(inverse).all() else None

        invalid_intrinsic_frames = 0
        invalid_extrinsic_frames = 0
        invalid_seed_frames = 0
        intrinsic_valid = np.zeros(num_frames, dtype=bool)
        extrinsic_valid = np.zeros(num_frames, dtype=bool)
        extrinsic_inverses: list[np.ndarray | None] = [None] * num_frames
        for frame_index in range(num_frames):
            intrinsic = np.asarray(intrinsics[frame_index], dtype=np.float64)
            if valid_intrinsic(intrinsic):
                intrinsic_valid[frame_index] = True
            else:
                invalid_intrinsic_frames += 1
            extrinsic = np.asarray(extrinsics[frame_index], dtype=np.float64)
            inverse = safe_inverse(extrinsic)
            if inverse is not None:
                extrinsic_valid[frame_index] = True
                extrinsic_inverses[frame_index] = inverse
            else:
                invalid_extrinsic_frames += 1

        for frame_index in seed_frames.tolist():
            depth = np.asarray(depths[frame_index], dtype=np.float32)
            if depth.ndim != 2 or not intrinsic_valid[frame_index]:
                invalid_seed_frames += 1
                continue
            inverse_extrinsic = extrinsic_inverses[frame_index]
            if inverse_extrinsic is None:
                invalid_seed_frames += 1
                continue
            valid_flat = np.flatnonzero(np.isfinite(depth) & (depth > 1e-4))
            # Do not seed world points from padded/invalid image bands. A point
            # seeded outside the valid crop can later project into the crop as
            # the camera moves and contaminate the reconstructed scene.
            pad_top = min(max(int(vertical_padding[0]), 0), depth.shape[0])
            pad_bottom = min(max(int(vertical_padding[1]), 0), depth.shape[0] - pad_top)
            if pad_top or pad_bottom:
                valid_rows = np.zeros(depth.shape[0], dtype=bool)
                stop = depth.shape[0] - pad_bottom if pad_bottom else depth.shape[0]
                valid_rows[pad_top:stop] = True
                valid_flat = valid_flat[valid_rows[valid_flat // depth.shape[1]]]
            if valid_flat.size == 0:
                continue
            if valid_flat.size > points_per_seed:
                if self.track_seed_near_ratio > 0.0:
                    depth_values = depth.reshape(-1)[valid_flat]
                    near_threshold = float(
                        np.quantile(depth_values, self.track_seed_near_quantile)
                    )
                    near = valid_flat[depth_values <= near_threshold]
                    far = valid_flat[depth_values > near_threshold]
                    near_count = min(
                        len(near), int(round(points_per_seed * self.track_seed_near_ratio))
                    )
                    far_count = points_per_seed - near_count
                    selected_parts = []
                    if near_count:
                        selected_parts.append(
                            np_rng.choice(near, size=near_count, replace=False)
                        )
                    if far_count:
                        if len(far) >= far_count:
                            selected_parts.append(
                                np_rng.choice(far, size=far_count, replace=False)
                            )
                        else:
                            remainder = points_per_seed - sum(
                                len(part) for part in selected_parts
                            )
                            selected_parts.append(
                                np_rng.choice(
                                    valid_flat,
                                    size=remainder,
                                    replace=False,
                                )
                            )
                    valid_flat = np.concatenate(selected_parts)
                else:
                    valid_flat = np_rng.choice(
                        valid_flat, size=points_per_seed, replace=False
                    )
                valid_flat = valid_flat.astype(np.int64, copy=False)
            ys, xs = np.divmod(valid_flat, depth.shape[1])
            z = depth[ys, xs].astype(np.float64)
            intrinsic = intrinsics[frame_index].astype(np.float64)
            # Every row here has finite positive depth and a validated K.  The
            # errstate guard additionally handles pathological but finite
            # scene values which overflow float64; those rows are discarded
            # below instead of leaking non-finite labels.
            with np.errstate(all="ignore"):
                camera = np.stack(
                    [
                        (xs.astype(np.float64) - intrinsic[0, 2]) * z
                        / intrinsic[0, 0],
                        (ys.astype(np.float64) - intrinsic[1, 2]) * z
                        / intrinsic[1, 1],
                        z,
                        np.ones_like(z),
                    ],
                    axis=1,
                )
                world = camera @ inverse_extrinsic.T
            finite = np.isfinite(world[:, :3]).all(axis=1)
            point_sets.append(world[finite, :3].astype(np.float32))

        if not point_sets:
            return None
        points_world = np.concatenate(point_sets, axis=0)[: self.max_track_points]
        num_points = len(points_world)
        trajectories = np.broadcast_to(points_world[None], (num_frames, num_points, 3)).copy()
        trajs_2d = np.zeros((num_frames, num_points, 2), dtype=np.float32)
        visible = np.zeros((num_frames, num_points), dtype=bool)
        visibility_known = np.zeros((num_frames, num_points), dtype=bool)
        defined_3d = np.zeros((num_frames, num_points), dtype=bool)
        homogeneous = np.concatenate(
            [points_world.astype(np.float64), np.ones((num_points, 1), dtype=np.float64)], axis=1
        )
        scene_depth = (
            float(np.mean(record.depth_range)) if record.depth_range is not None else 1.0
        )
        absolute_tolerance = max(
            self.track_depth_tolerance_abs,
            self.track_depth_tolerance_scene_ratio * scene_depth,
        )
        if not np.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
            # Configuration values are normally validated at construction;
            # keep GT generation fail-closed if a legacy/pickled adapter has a
            # corrupted tolerance field.
            absolute_tolerance = 0.01
        float32_limit = float(np.finfo(np.float32).max)
        for frame_index in range(num_frames):
            depth = np.asarray(depths[frame_index], dtype=np.float32)
            if depth.ndim != 2:
                continue
            if not intrinsic_valid[frame_index] or not extrinsic_valid[frame_index]:
                # Keep all outputs finite and leave this frame unknown.
                continue
            with np.errstate(all="ignore"):
                camera = homogeneous @ extrinsics[frame_index].astype(np.float64).T
            z = camera[:, 2]
            intrinsic = intrinsics[frame_index].astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                u = intrinsic[0, 0] * camera[:, 0] / z + intrinsic[0, 2]
                v = intrinsic[1, 1] * camera[:, 1] / z + intrinsic[1, 2]
            # ``trajs_2d`` is part of the public clip contract.  Store finite
            # sentinels even for points behind the camera or outside the image;
            # validity masks below retain the distinction between an unknown
            # point and a genuine pixel at (0, 0).  Clipping avoids an overflow
            # warning when a finite but extreme projection is cast to float32.
            u_safe = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
            v_safe = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            u_safe = np.clip(u_safe, -float32_limit, float32_limit)
            v_safe = np.clip(v_safe, -float32_limit, float32_limit)
            trajs_2d[frame_index] = np.stack([u_safe, v_safe], axis=1).astype(
                np.float32
            )
            h, w = depth.shape
            pad_top = min(max(int(vertical_padding[0]), 0), h)
            pad_bottom = min(max(int(vertical_padding[1]), 0), h - pad_top)
            valid_y_min = pad_top
            valid_y_max = h - pad_bottom if pad_bottom else h
            finite_uv = np.isfinite(u) & np.isfinite(v)
            ui = np.zeros(num_points, dtype=np.int64)
            vi = np.zeros(num_points, dtype=np.int64)
            # Only cast values that could possibly land in this image.  A
            # finite projection of magnitude 1e300 is valid as a number but
            # cannot be represented by int64 and must be treated as out of
            # bounds without invoking an unsafe cast.
            castable = finite_uv & (u >= -1.0) & (u <= float(w)) & (
                v >= -1.0
            ) & (v <= float(h))
            ui[castable] = np.rint(u[castable]).astype(np.int64)
            vi[castable] = np.rint(v[castable]).astype(np.int64)
            in_bounds = (
                castable & np.isfinite(z) & (z > 1e-4)
                & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
                & (vi >= valid_y_min) & (vi < valid_y_max)
            )
            # Round-trip projection of a float32 depth/K/pose sample can land
            # a few ulps outside the physical image boundary (for example
            # -1.6e-7 instead of 0).  The validity decision uses the rounded
            # integer pixel, so clamp only valid public coordinates to that
            # same closed pixel domain while retaining diagnostic coordinates
            # for invalid/out-of-view rows.
            if in_bounds.any():
                trajs_2d[frame_index, in_bounds, 0] = np.clip(
                    trajs_2d[frame_index, in_bounds, 0], 0.0, float(w - 1)
                )
                trajs_2d[frame_index, in_bounds, 1] = np.clip(
                    trajs_2d[frame_index, in_bounds, 1],
                    float(valid_y_min),
                    float(valid_y_max - 1),
                )
            defined_3d[frame_index] = in_bounds
            z_safe = np.where(np.isfinite(z) & (z > 1e-4), z, 0.0)
            tolerance = np.maximum(
                absolute_tolerance, self.track_depth_tolerance_rel * z_safe
            )
            focal_min = max(
                min(abs(float(intrinsic[0, 0])), abs(float(intrinsic[1, 1]))),
                1e-6,
            )
            lateral_tolerance = 1.5 * z_safe / focal_min
            with np.errstate(all="ignore"):
                roundtrip_tolerance = np.sqrt(
                    tolerance * tolerance + lateral_tolerance * lateral_tolerance
                )
            roundtrip_tolerance = np.nan_to_num(
                roundtrip_tolerance,
                nan=np.inf,
                posinf=np.inf,
                neginf=np.inf,
            )
            visible_support = np.zeros(num_points, dtype=np.int16)
            occluding_support = np.zeros(num_points, dtype=np.int16)
            center_depth = np.zeros(num_points, dtype=np.float64)
            center_depth[in_bounds] = depth[vi[in_bounds], ui[in_bounds]]
            center_depth_valid = (
                in_bounds & np.isfinite(center_depth) & (center_depth > 1e-4)
            )
            radius = self.visibility_radius
            for offset_y in range(-radius, radius + 1):
                for offset_x in range(-radius, radius + 1):
                    sample_u = ui + offset_x
                    sample_v = vi + offset_y
                    sample_ok = (
                        in_bounds
                        & (sample_u >= 0) & (sample_u < w)
                        & (sample_v >= 0) & (sample_v < h)
                    )
                    sampled_depth = np.zeros(num_points, dtype=np.float64)
                    sampled_depth[sample_ok] = depth[sample_v[sample_ok], sample_u[sample_ok]]
                    depth_ok = sample_ok & np.isfinite(sampled_depth) & (sampled_depth > 1e-4)
                    # Build the camera-space sample only for finite positive
                    # depths.  The old all-row stack evaluated NaN/Inf values
                    # before applying ``depth_ok`` (e.g. ``0 * NaN``), which
                    # caused the RuntimeWarning seen during training.
                    sampled_camera = np.zeros((num_points, 3), dtype=np.float64)
                    if depth_ok.any():
                        valid_rows = np.flatnonzero(depth_ok)
                        with np.errstate(all="ignore"):
                            sampled_camera[valid_rows, 0] = (
                                sample_u[valid_rows].astype(np.float64)
                                - intrinsic[0, 2]
                            ) * sampled_depth[valid_rows] / intrinsic[0, 0]
                            sampled_camera[valid_rows, 1] = (
                                sample_v[valid_rows].astype(np.float64)
                                - intrinsic[1, 2]
                            ) * sampled_depth[valid_rows] / intrinsic[1, 1]
                            sampled_camera[valid_rows, 2] = sampled_depth[valid_rows]
                        finite_sampled_camera = np.isfinite(sampled_camera).all(axis=1)
                        depth_ok &= finite_sampled_camera
                    roundtrip_error = np.full(num_points, np.inf, dtype=np.float64)
                    if depth_ok.any():
                        valid_rows = np.flatnonzero(depth_ok)
                        with np.errstate(all="ignore"):
                            deltas = (
                                sampled_camera[valid_rows]
                                - camera[valid_rows, :3]
                            )
                            errors = np.linalg.norm(deltas, axis=1)
                        finite_errors = np.isfinite(errors)
                        if finite_errors.any():
                            roundtrip_error[valid_rows[finite_errors]] = errors[
                                finite_errors
                            ]
                        # Keep the comparison fail-closed.  In particular,
                        # ``inf <= inf`` is true in NumPy and must not turn an
                        # overflowed round trip into visible support.
                        depth_ok[valid_rows] &= finite_errors
                    visible_support += (
                        depth_ok & (roundtrip_error <= roundtrip_tolerance)
                    ).astype(np.int16)
                    # A reliably closer surface is positive evidence that the
                    # queried point is occluded. A farther surface is not a
                    # reliable negative: it can indicate a hole/dynamic point.
                    occluding_support += (
                        depth_ok & (sampled_depth < z - tolerance)
                    ).astype(np.int16)

            center_camera = np.zeros((num_points, 3), dtype=np.float64)
            if center_depth_valid.any():
                valid_rows = np.flatnonzero(center_depth_valid)
                with np.errstate(all="ignore"):
                    center_camera[valid_rows, 0] = (
                        ui[valid_rows].astype(np.float64) - intrinsic[0, 2]
                    ) * center_depth[valid_rows] / intrinsic[0, 0]
                    center_camera[valid_rows, 1] = (
                        vi[valid_rows].astype(np.float64) - intrinsic[1, 2]
                    ) * center_depth[valid_rows] / intrinsic[1, 1]
                    center_camera[valid_rows, 2] = center_depth[valid_rows]
                center_depth_valid &= np.isfinite(center_camera).all(axis=1)
            center_roundtrip_error = np.full(num_points, np.inf, dtype=np.float64)
            if center_depth_valid.any():
                valid_rows = np.flatnonzero(center_depth_valid)
                with np.errstate(all="ignore"):
                    deltas = center_camera[valid_rows] - camera[valid_rows, :3]
                    errors = np.linalg.norm(deltas, axis=1)
                finite_errors = np.isfinite(errors)
                if finite_errors.any():
                    center_roundtrip_error[valid_rows[finite_errors]] = errors[
                        finite_errors
                    ]
                center_depth_valid[valid_rows] &= finite_errors
            center_agrees = center_depth_valid & (
                center_roundtrip_error <= roundtrip_tolerance
            )
            # Sub-pixel reprojection and pose/depth rasterization can move a
            # true surface observation by one pixel at discontinuities. A
            # matching center is sufficient; otherwise >=2 matching neighbors
            # recover that small registration error.
            is_visible = in_bounds & (
                center_agrees
                | (visible_support >= self.visibility_min_support)
            )
            is_occluded = (
                in_bounds
                & ~is_visible
                & center_depth_valid
                & (center_depth < z - tolerance)
                # Occlusion needs a coherent foreground patch, not merely two
                # boundary pixels. Any matching-depth support makes the case
                # ambiguous and therefore Unknown rather than a negative.
                & (visible_support == 0)
                & (occluding_support >= self.occlusion_min_support)
            )
            visible[frame_index] = is_visible
            visibility_known[frame_index] = is_visible | is_occluded

        observation_count = visible.sum(axis=0)
        keep = observation_count >= self.min_track_observations
        if not keep.any():
            return None
        trajs_2d = trajs_2d[:, keep]
        trajectories = trajectories[:, keep]
        defined_3d = defined_3d[:, keep]
        visibility_known = visibility_known[:, keep]
        visible = visible[:, keep]
        return (
            trajs_2d,
            trajectories,
            defined_3d,
            visible.copy(),
            visibility_known,
            {
                "generated_track_points": int(keep.sum()),
                "generated_track_candidate_points": int(num_points),
                "generated_track_mean_observations": float(observation_count[keep].mean()),
                "generated_track_min_observations": int(self.min_track_observations),
                "generated_track_depth_tolerance_abs": float(absolute_tolerance),
                "generated_track_depth_tolerance_rel": float(self.track_depth_tolerance_rel),
                "generated_track_visibility_known_ratio": float(
                    visibility_known.mean()
                ),
                "generated_track_visible_ratio": float(visible.mean()),
                "generated_track_occluded_ratio": float(
                    (visibility_known & ~visible).mean()
                ),
                "generated_track_unknown_ratio": float((~visibility_known).mean()),
                "generated_track_visibility_radius": int(self.visibility_radius),
                "generated_track_visibility_min_support": int(
                    self.visibility_min_support
                ),
                "generated_track_occlusion_min_support": int(
                    self.occlusion_min_support
                ),
                "generated_track_invalid_intrinsic_frames": int(
                    invalid_intrinsic_frames
                ),
                "generated_track_invalid_extrinsic_frames": int(
                    invalid_extrinsic_frames
                ),
                "generated_track_invalid_seed_frames": int(invalid_seed_frames),
            },
        )
