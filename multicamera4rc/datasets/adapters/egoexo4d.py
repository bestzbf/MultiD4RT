from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import random
import tempfile
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import fcntl
import numpy as np

from .base import BaseAdapter, UnifiedClip
from ..multicam_slots import (
    aligned_times_per_camera,
    assert_no_cross_camera_tubelets,
    build_camera_major_slot_plan,
    valid_time_choices,
)


_EGOPOSE_JOINT_NAMES = (
    "nose", "left-eye", "right-eye", "left-ear", "right-ear",
    "left-shoulder", "right-shoulder", "left-elbow", "right-elbow",
    "left-wrist", "right-wrist", "left-hip", "right-hip",
    "left-knee", "right-knee", "left-ankle", "right-ankle",
)


def _configure_cv2_runtime() -> None:
    raw_threads = os.getenv("D4RT_OPENCV_THREADS", "1").strip()
    try:
        threads = max(1, int(raw_threads)) if raw_threads else 1
        cv2.setNumThreads(threads)
    except Exception:
        pass
    try:
        if hasattr(cv2, "ocl"):
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


_configure_cv2_runtime()


@dataclass(frozen=True)
class _CameraCalib:
    camera_id: str
    intrinsic: np.ndarray
    distortion: np.ndarray
    extrinsic: np.ndarray
    image_width: int
    image_height: int


@dataclass(frozen=True)
class _TakeRecord:
    take_name: str
    take_uid: str
    split: str
    take_dir: Path
    video_dir: Path
    trajectory_dir: Path
    cameras: tuple[str, ...]
    num_frames: int


class EgoExo4DAdapter(BaseAdapter):
    """Adapter for Ego-Exo4D synchronized exo-camera training clips.

    The adapter exposes one take as one sequence. Each sample selects a random
    subset of fixed exo cameras, lays them out as camera-major slots, and uses
    the take's semi-dense SLAM point cloud projected into each camera as sparse
    depth supervision.
    """

    dataset_name: str = "egoexo4d"

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        sequence_names: Optional[list[str]] = None,
        cameras: Optional[list[str]] = None,
        min_cameras: int = 2,
        max_cameras: int = 5,
        min_times: int = 1,
        max_times: Optional[int] = None,
        random_num_times: bool = False,
        random_camera_order: bool = False,
        enforce_tubelet_aligned_layout: bool = True,
        encoder_tubelet_size: int = 2,
        fps: float = 30.0,
        downscaled_size: int = 448,
        use_camera_pose_annotations: bool = True,
        require_depth: bool = True,
        require_camera_pose: bool = False,
        undistort: bool = True,
        undistort_balance: float = 0.0,
        cache_dir: str | Path | None = None,
        max_points: int = 300_000,
        point_scan_limit: int = 6_000_000,
        max_inv_dist_std: Optional[float] = 0.005,
        min_inv_dist_std: Optional[float] = None,
        max_dist_std: float = 0.01,
        depth_max_m: float = 80.0,
        depth_dilation_radius: int = 2,
        use_direct_semidense_queries: bool = False,
        semidense_filter_aria_wearer_occlusion: bool = False,
        semidense_aria_wearer_radius_m: float = 0.65,
        semidense_aria_wearer_height_m: float = 1.8,
        semidense_aria_wearer_head_margin_m: float = 0.2,
        semidense_multiview_balance_camera_pairs: bool = False,
        semidense_projection_point_cap: int = 80_000,
        semidense_query_grid_size: int = 16,
        semidense_query_depth_bins: int = 8,
        semidense_depth_tolerance_m: float = 0.10,
        semidense_depth_tolerance_rel: float = 0.03,
        semidense_visibility_radius: int = 0,
        semidense_depth_support_radius: int = 0,
        semidense_depth_min_support: int = 0,
        semidense_use_depth_visibility: bool = True,
        semidense_allow_sparse_depth_fallback: bool = True,
        semidense_multiview_group_ratio: float = 0.0,
        semidense_multiview_group_size: int = 2,
        semidense_multiview_group_min_cameras: int = 2,
        semidense_multiview_nonvisible_ref_ratio: float = 0.0,
        semidense_multiview_nonvisible_ref_strict: bool = False,
        semidense_multiview_nonvisible_ref_mask_visibility: bool = False,
        egopose_query_ratio: float = 0.0,
        egopose_manual_only: bool = True,
        egopose_manual_group_ratio: float = 0.0,
        egopose_min_views_for_3d: int = 2,
        egopose_group_size: int = 4,
        egopose_cross_time_ratio: float = 0.0,
        egopose_3d_weight: float = 1.0,
        egopose_cache_items: int = 2,
        frame_cache_items: Optional[int] = None,
        depth_cache_items: Optional[int] = None,
        frame_disk_cache: Optional[bool] = None,
        frame_disk_cache_dir: str | Path | None = None,
        frame_disk_cache_jpeg_quality: Optional[int] = None,
        video_read_strategy: Optional[str] = None,
        sequential_decode_factor: Optional[int] = None,
        sequential_decode_max_extra: Optional[int] = None,
        camera_read_workers: Optional[int] = None,
        strict: bool = True,
        verbose: bool = False,
        **_: Any,
    ) -> None:
        self.root = Path(root)
        self.split = str(split)
        self.sequence_names = (
            None if sequence_names is None else frozenset(str(name) for name in sequence_names)
        )
        self.cameras = tuple(cameras or ["cam01", "cam02", "cam03", "cam04", "cam05"])
        self.min_cameras = max(1, int(min_cameras))
        self.max_cameras = max(self.min_cameras, int(max_cameras))
        self.min_times = max(1, int(min_times))
        self.max_times = None if max_times is None else max(1, int(max_times))
        self.random_num_times = bool(random_num_times)
        self.random_camera_order = bool(random_camera_order)
        self.enforce_tubelet_aligned_layout = bool(enforce_tubelet_aligned_layout)
        self.encoder_tubelet_size = max(1, int(encoder_tubelet_size))
        self.fps = float(fps)
        self.downscaled_size = int(downscaled_size)
        self.use_camera_pose_annotations = bool(use_camera_pose_annotations)
        self.require_depth = bool(require_depth)
        self.require_camera_pose = bool(require_camera_pose)
        self.undistort = bool(undistort)
        self.undistort_balance = float(undistort_balance)
        self.max_points = max(1, int(max_points))
        self.point_scan_limit = max(0, int(point_scan_limit))
        if max_inv_dist_std is None:
            if min_inv_dist_std is None:
                max_inv_dist_std = 0.005
            else:
                max_inv_dist_std = min_inv_dist_std
        self.max_inv_dist_std = float(max_inv_dist_std)
        self.min_inv_dist_std = None if min_inv_dist_std is None else float(min_inv_dist_std)
        self.max_dist_std = float(max_dist_std)
        self.depth_max_m = float(depth_max_m)
        self.depth_dilation_radius = max(0, int(depth_dilation_radius))
        self.use_direct_semidense_queries = bool(use_direct_semidense_queries)
        self.semidense_filter_aria_wearer_occlusion = bool(
            semidense_filter_aria_wearer_occlusion
        )
        self.semidense_aria_wearer_radius_m = float(semidense_aria_wearer_radius_m)
        self.semidense_aria_wearer_height_m = float(semidense_aria_wearer_height_m)
        self.semidense_aria_wearer_head_margin_m = float(semidense_aria_wearer_head_margin_m)
        self.semidense_multiview_balance_camera_pairs = bool(
            semidense_multiview_balance_camera_pairs
        )
        self.semidense_projection_point_cap = max(0, int(semidense_projection_point_cap))
        self.semidense_query_grid_size = max(2, int(semidense_query_grid_size))
        self.semidense_query_depth_bins = max(1, int(semidense_query_depth_bins))
        self.semidense_depth_tolerance_m = max(0.0, float(semidense_depth_tolerance_m))
        self.semidense_depth_tolerance_rel = max(0.0, float(semidense_depth_tolerance_rel))
        self.semidense_visibility_radius = max(0, int(semidense_visibility_radius))
        self.semidense_depth_support_radius = max(0, int(semidense_depth_support_radius))
        self.semidense_depth_min_support = max(0, int(semidense_depth_min_support))
        self.semidense_use_depth_visibility = bool(semidense_use_depth_visibility)
        self.semidense_allow_sparse_depth_fallback = bool(semidense_allow_sparse_depth_fallback)
        self.semidense_multiview_group_ratio = min(max(float(semidense_multiview_group_ratio), 0.0), 1.0)
        self.semidense_multiview_group_size = max(2, int(semidense_multiview_group_size))
        self.semidense_multiview_group_min_cameras = max(2, int(semidense_multiview_group_min_cameras))
        env_nonvisible_ref_ratio = os.getenv(
            "D4RT_EGOEXO4D_SEMIDENSE_MULTIVIEW_NONVISIBLE_REF_RATIO",
            "",
        ).strip()
        nonvisible_ref_ratio = (
            float(env_nonvisible_ref_ratio)
            if env_nonvisible_ref_ratio
            else float(semidense_multiview_nonvisible_ref_ratio)
        )
        self.semidense_multiview_nonvisible_ref_ratio = min(
            max(nonvisible_ref_ratio, 0.0),
            1.0,
        )
        env_nonvisible_ref_strict = os.getenv(
            "D4RT_EGOEXO4D_SEMIDENSE_MULTIVIEW_NONVISIBLE_REF_STRICT",
            "",
        ).strip().lower()
        self.semidense_multiview_nonvisible_ref_strict = (
            env_nonvisible_ref_strict in {"1", "true", "yes", "on"}
            if env_nonvisible_ref_strict
            else bool(semidense_multiview_nonvisible_ref_strict)
        )
        self.semidense_multiview_nonvisible_ref_mask_visibility = bool(
            semidense_multiview_nonvisible_ref_mask_visibility
        )
        self.egopose_query_ratio = min(max(float(egopose_query_ratio), 0.0), 1.0)
        self.egopose_manual_only = bool(egopose_manual_only)
        self.egopose_manual_group_ratio = min(
            max(float(egopose_manual_group_ratio), 0.0), 1.0
        )
        self.egopose_min_views_for_3d = max(1, int(egopose_min_views_for_3d))
        self.egopose_group_size = max(2, int(egopose_group_size))
        self.egopose_cross_time_ratio = min(
            max(float(egopose_cross_time_ratio), 0.0), 1.0
        )
        self.egopose_3d_weight = max(0.0, float(egopose_3d_weight))
        self.egopose_cache_items = max(0, int(egopose_cache_items))
        if frame_cache_items is None:
            raw_frame_cache_items = os.getenv("D4RT_EGOEXO4D_FRAME_CACHE_ITEMS", "512").strip()
            frame_cache_items = int(raw_frame_cache_items) if raw_frame_cache_items else 0
        if depth_cache_items is None:
            raw_depth_cache_items = os.getenv("D4RT_EGOEXO4D_DEPTH_CACHE_ITEMS", "128").strip()
            depth_cache_items = int(raw_depth_cache_items) if raw_depth_cache_items else 0
        self.frame_cache_items = max(0, int(frame_cache_items))
        self.depth_cache_items = max(0, int(depth_cache_items))
        if video_read_strategy is None:
            video_read_strategy = os.getenv("D4RT_EGOEXO4D_VIDEO_READ_STRATEGY", "auto")
        video_read_strategy = str(video_read_strategy).strip().lower() or "auto"
        if video_read_strategy not in {"auto", "seek", "sequential"}:
            video_read_strategy = "auto"
        if sequential_decode_factor is None:
            raw_factor = os.getenv("D4RT_EGOEXO4D_SEQUENTIAL_DECODE_FACTOR", "16").strip()
            sequential_decode_factor = int(raw_factor) if raw_factor else 16
        if sequential_decode_max_extra is None:
            raw_max_extra = os.getenv("D4RT_EGOEXO4D_SEQUENTIAL_DECODE_MAX_EXTRA", "512").strip()
            sequential_decode_max_extra = int(raw_max_extra) if raw_max_extra else 512
        self.video_read_strategy = video_read_strategy
        self.sequential_decode_factor = max(1, int(sequential_decode_factor))
        self.sequential_decode_max_extra = max(0, int(sequential_decode_max_extra))
        if camera_read_workers is None:
            raw_camera_workers = os.getenv("D4RT_EGOEXO4D_CAMERA_READ_WORKERS", "1").strip()
            camera_read_workers = int(raw_camera_workers) if raw_camera_workers else 1
        self.camera_read_workers = max(1, int(camera_read_workers))
        self.strict = bool(strict)
        self.verbose = bool(verbose)

        if cache_dir is None:
            cache_dir = os.getenv(
                "D4RT_EGOEXO4D_CACHE_DIR",
                str(self.root / ".d4rt_cache"),
            )
        self.cache_dir = Path(cache_dir)
        self.depth_cache_dir = self.cache_dir / "depth_v1"
        self.points_cache_dir = self.cache_dir / "points_v1"
        self.depth_cache_dir.mkdir(parents=True, exist_ok=True)
        self.points_cache_dir.mkdir(parents=True, exist_ok=True)
        if frame_disk_cache is None:
            raw_frame_disk_cache = os.getenv("D4RT_EGOEXO4D_FRAME_DISK_CACHE", "0")
            frame_disk_cache = raw_frame_disk_cache.strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        if frame_disk_cache_jpeg_quality is None:
            raw_jpeg_quality = os.getenv(
                "D4RT_EGOEXO4D_FRAME_DISK_CACHE_JPEG_QUALITY",
                "95",
            ).strip()
            frame_disk_cache_jpeg_quality = int(raw_jpeg_quality) if raw_jpeg_quality else 95
        if frame_disk_cache_dir is None:
            frame_disk_cache_dir = os.getenv(
                "D4RT_EGOEXO4D_FRAME_DISK_CACHE_DIR",
                str(self.cache_dir / "rgb_jpeg_v1"),
            )
        self.frame_disk_cache = bool(frame_disk_cache)
        self.frame_disk_cache_dir = Path(frame_disk_cache_dir)
        self.frame_disk_cache_jpeg_quality = min(
            100,
            max(50, int(frame_disk_cache_jpeg_quality)),
        )
        if self.frame_disk_cache:
            self.frame_disk_cache_dir.mkdir(parents=True, exist_ok=True)

        self._records: list[_TakeRecord] = []
        self._record_by_name: dict[str, _TakeRecord] = {}
        self._calib_cache: dict[tuple[str, int, int], dict[str, _CameraCalib]] = {}
        self._rectify_cache: dict[tuple[str, str, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._points_cache: dict[str, np.ndarray] = {}
        self._aria_wearer_centers_cache: dict[str, Optional[dict[int, np.ndarray]]] = {}
        self._egopose_cache: OrderedDict[str, tuple[dict, dict]] = OrderedDict()
        self._frame_cache: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        self._depth_array_cache: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

        self._build_index()
        if not self._records and self.strict:
            raise RuntimeError(
                f"EgoExo4DAdapter found no usable {self.split!r} takes under {self.root}"
            )

        if self.verbose:
            print(
                f"[EgoExo4DAdapter] split={self.split}, takes={len(self._records)}, "
                f"root={self.root}, cache={self.cache_dir}",
                flush=True,
            )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_lock"] = None
        state["_points_cache"] = {}
        state["_aria_wearer_centers_cache"] = {}
        state["_egopose_cache"] = OrderedDict()
        state["_frame_cache"] = OrderedDict()
        state["_depth_array_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
        if self.__dict__.get("_points_cache") is None:
            self._points_cache = {}
        if self.__dict__.get("_aria_wearer_centers_cache") is None:
            self._aria_wearer_centers_cache = {}
        if self.__dict__.get("_egopose_cache") is None:
            self._egopose_cache = OrderedDict()
        if self.__dict__.get("_frame_cache") is None:
            self._frame_cache = OrderedDict()
        if self.__dict__.get("_depth_array_cache") is None:
            self._depth_array_cache = OrderedDict()

    def __len__(self) -> int:
        return len(self._records)

    def list_sequences(self) -> list[str]:
        return [record.take_name for record in self._records]

    def get_sequence_name(self, index: int) -> str:
        return self._records[index].take_name

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        record = self._get_record(sequence_name)
        return {
            "num_frames": record.num_frames,
            "take_uid": record.take_uid,
            "take_dir": str(record.take_dir),
            "cameras": list(record.cameras),
            "has_depth": self.require_depth,
            "has_tracks": False,
        }

    def get_num_frames(self, sequence_name: str) -> int:
        return self._get_record(sequence_name).num_frames

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        record = self._get_record(sequence_name)
        frame_indices = list(range(min(8, record.num_frames)))
        clip = self.load_clip(sequence_name, frame_indices)
        return {
            "ok": True,
            "sequence_name": sequence_name,
            "num_frames": record.num_frames,
            "num_images": len(clip.images),
            "image_shape": list(clip.images[0].shape),
            "has_depth": clip.depths is not None,
            "selected_cameras": clip.metadata.get("selected_cameras"),
            "layout_num_cameras": clip.metadata.get("layout_num_cameras"),
        }

    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        load_t0 = time.perf_counter()
        record = self._get_record(sequence_name)
        if not frame_indices:
            raise ValueError("EgoExo4DAdapter.load_clip requires non-empty frame_indices")

        clip_len = len(frame_indices)
        rng = random.Random(self._sample_seed(record.take_uid, frame_indices))
        available_cameras = list(record.cameras)
        max_cameras = min(
            self.max_cameras,
            len(available_cameras),
            self._max_cameras_for_clip(clip_len),
        )
        min_cameras = min(self.min_cameras, max_cameras)
        num_cameras = rng.randint(min_cameras, max_cameras)
        selected_cameras = sorted(rng.sample(available_cameras, k=num_cameras))
        num_times = self._choose_num_times(rng, clip_len, num_cameras)

        time_indices = self._sample_time_indices(frame_indices, num_times, record.num_frames)
        slot_plan = self._build_slot_plan(clip_len, num_cameras, num_times)
        used_slots = int(slot_plan.valid_slot.sum())
        if used_slots <= 0:
            raise RuntimeError(f"[{sequence_name}] no valid Ego-Exo4D slots")

        probe_camera = selected_cameras[0]
        probe_path = self._video_path(record, probe_camera)
        height, width = self._video_hw(probe_path)
        calibs = self._load_calibrations(record, width=width, height=height)
        calibrated_cameras = [camera_id for camera_id in available_cameras if camera_id in calibs]
        if len(calibrated_cameras) < self.min_cameras:
            raise RuntimeError(
                f"[{sequence_name}] only {len(calibrated_cameras)} calibrated exo cameras available"
            )
        if any(camera_id not in calibs for camera_id in selected_cameras):
            max_cameras = min(
                self.max_cameras,
                len(calibrated_cameras),
                self._max_cameras_for_clip(clip_len),
            )
            min_cameras = min(self.min_cameras, max_cameras)
            num_cameras = rng.randint(min_cameras, max_cameras)
            selected_cameras = sorted(rng.sample(calibrated_cameras, k=num_cameras))
            num_times = self._choose_num_times(rng, clip_len, num_cameras)
            time_indices = self._sample_time_indices(frame_indices, num_times, record.num_frames)
            slot_plan = self._build_slot_plan(clip_len, num_cameras, num_times)
            used_slots = int(slot_plan.valid_slot.sum())
        if self.random_camera_order:
            rng.shuffle(selected_cameras)
        t_depth_ensure0 = time.perf_counter()
        self._ensure_depth_cache(record, selected_cameras, width, height, calibs)
        depth_ensure_s = time.perf_counter() - t_depth_ensure0
        direct_points_world: Optional[np.ndarray] = None
        semidense_aria_wearer_centers_world: Optional[np.ndarray] = None
        egopose_metadata: Optional[dict[str, Any]] = None
        points_load_s = 0.0
        if self.use_direct_semidense_queries:
            t_points0 = time.perf_counter()
            direct_points_world = self._load_filtered_points(record)
            points_load_s = time.perf_counter() - t_points0
            if self.semidense_filter_aria_wearer_occlusion:
                semidense_aria_wearer_centers_world = self._load_aria_wearer_centers(
                    record, time_indices
                )
        if self.egopose_query_ratio > 0.0:
            egopose_metadata = self._load_egopose_clip(
                record,
                time_indices,
                selected_cameras,
            )

        per_camera_frames: dict[str, dict[int, np.ndarray]] = {}
        t_rgb0 = time.perf_counter()
        if self.camera_read_workers > 1 and len(selected_cameras) > 1:
            max_workers = min(self.camera_read_workers, len(selected_cameras))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._read_video_frames,
                        self._video_path(record, camera_id),
                        time_indices,
                        record,
                        camera_id,
                        width,
                        height,
                        calibs[camera_id],
                    ): camera_id
                    for camera_id in selected_cameras
                }
                for future in as_completed(futures):
                    camera_id = futures[future]
                    per_camera_frames[camera_id] = future.result()
        else:
            for camera_id in selected_cameras:
                per_camera_frames[camera_id] = self._read_video_frames(
                    self._video_path(record, camera_id),
                    time_indices,
                    record,
                    camera_id,
                    width,
                    height,
                    calibs[camera_id],
                )
        rgb_load_s = time.perf_counter() - t_rgb0

        t_depth_load0 = time.perf_counter()
        per_camera_depth: dict[str, np.ndarray] = {
            camera_id: self._load_depth(record, camera_id, width, height)
            for camera_id in selected_cameras
        }
        depth_load_s = time.perf_counter() - t_depth_load0

        images: list[np.ndarray] = []
        depths: list[np.ndarray] = []
        intrinsics: list[np.ndarray] = []
        extrinsics: list[np.ndarray] = []
        frame_paths: list[str] = []
        slot_camera: list[int] = []
        slot_time: list[int] = []
        valid_slot: list[bool] = []

        physical_times = int(slot_plan.physical_times_per_camera)
        zero_image = np.zeros_like(next(iter(per_camera_frames[selected_cameras[0]].values())))
        zero_depth = np.zeros_like(per_camera_depth[selected_cameras[0]])
        for camera_index, camera_id in enumerate(selected_cameras):
            calib = calibs[camera_id]
            rect_intrinsic = self._rectified_intrinsic(record, camera_id, width, height, calib)
            depth = per_camera_depth[camera_id]
            for physical_time in range(physical_times):
                is_valid = physical_time < num_times
                time_index = min(physical_time, num_times - 1)
                frame_index = time_indices[time_index]
                images.append(
                    per_camera_frames[camera_id][frame_index] if is_valid else zero_image.copy()
                )
                depths.append(depth if is_valid else zero_depth.copy())
                intrinsics.append(rect_intrinsic.astype(np.float32, copy=True))
                extrinsics.append(calib.extrinsic.astype(np.float32, copy=True))
                frame_paths.append(
                    f"{self._video_path(record, camera_id)}#{frame_index}"
                    if is_valid else ""
                )
                slot_camera.append(camera_index)
                slot_time.append(time_index)
                valid_slot.append(is_valid)

        if not images:
            raise RuntimeError(f"[{sequence_name}] no Ego-Exo4D frames loaded")

        pad_image = np.zeros_like(images[-1])
        pad_depth = np.zeros_like(depths[-1])
        pad_intrinsic = intrinsics[-1]
        pad_extrinsic = extrinsics[-1]
        while len(images) < clip_len:
            images.append(pad_image)
            depths.append(pad_depth)
            intrinsics.append(pad_intrinsic)
            extrinsics.append(pad_extrinsic)
            frame_paths.append("")
            slot_camera.append(0)
            slot_time.append(0)
            valid_slot.append(False)

        if self.enforce_tubelet_aligned_layout:
            assert_no_cross_camera_tubelets(
                np.asarray(slot_camera),
                np.asarray(valid_slot),
                self.encoder_tubelet_size,
            )

        if len(images) > clip_len:
            images = images[:clip_len]
            depths = depths[:clip_len]
            intrinsics = intrinsics[:clip_len]
            extrinsics = extrinsics[:clip_len]
            frame_paths = frame_paths[:clip_len]
            slot_camera = slot_camera[:clip_len]
            slot_time = slot_time[:clip_len]
            valid_slot = valid_slot[:clip_len]

        metadata = {
            "dataset_name": self.dataset_name,
            "split": record.split,
            "take_name": record.take_name,
            "take_uid": record.take_uid,
            "sequence_root": str(record.take_dir),
            "frame_indices": list(frame_indices),
            "time_indices": list(time_indices),
            "selected_cameras": list(selected_cameras),
            "has_depth": True,
            "has_normals": False,
            "has_tracks": False,
            "has_visibility": False,
            "pose_convention": "w2c",
            "extrinsics_convention": "w2c",
            "intrinsics_convention": "undistorted_pinhole",
            "depth_unit": "meters",
            "layout_mode": "provided_multicam",
            "provided_slot_layout": True,
            "slot_camera": slot_camera,
            "slot_time": slot_time,
            "valid_slot": valid_slot,
            "layout_num_cameras": int(num_cameras),
            "layout_num_times": int(num_times),
            "layout_physical_times_per_camera": int(physical_times),
            "layout_tubelet_size": int(self.encoder_tubelet_size),
            "layout_tubelet_aligned": bool(self.enforce_tubelet_aligned_layout),
            "layout_valid_slots": int(sum(valid_slot)),
            "slot_source_indices": list(range(clip_len)),
            "query_semantics": "egoexo4d_static_map_multicam",
            "has_temporal_supervision": False,
            "raw_image_hw": [height, width],
            "rgb_hw": [height, width],
            "target_hw": [height, width],
            "_load_timing": {
                "total_s": time.perf_counter() - load_t0,
                "scene_data_s": depth_ensure_s + points_load_s,
                "rgb_load_s": rgb_load_s,
                "rgb_resize_s": 0.0,
                "depth_load_s": depth_load_s,
                "depth_resize_s": 0.0,
                "precomputed_s": 0.0,
                "process_s": 0.0,
                "depth_ensure_s": depth_ensure_s,
                "points_load_s": points_load_s,
                "rgb_cameras": len(selected_cameras),
                "rgb_unique_frames": len(set(time_indices)),
                "rgb_read_workers": int(min(self.camera_read_workers, len(selected_cameras))),
                "layout_num_cameras": int(num_cameras),
                "layout_num_times": int(num_times),
            },
        }
        if direct_points_world is not None:
            semidense_filter_enabled = semidense_aria_wearer_centers_world is not None
            metadata.update(
                {
                    "semidense_direct_query": True,
                    "semidense_points_world": direct_points_world,
                    "semidense_filter_aria_wearer_occlusion": semidense_filter_enabled,
                    "semidense_aria_wearer_radius_m": float(self.semidense_aria_wearer_radius_m),
                    "semidense_aria_wearer_height_m": float(self.semidense_aria_wearer_height_m),
                    "semidense_aria_wearer_head_margin_m": float(
                        self.semidense_aria_wearer_head_margin_m
                    ),
                    "semidense_multiview_balance_camera_pairs": bool(
                        self.semidense_multiview_balance_camera_pairs
                    ),
                    "semidense_projection_point_cap": int(self.semidense_projection_point_cap),
                    "semidense_query_grid_size": int(self.semidense_query_grid_size),
                    "semidense_query_depth_bins": int(self.semidense_query_depth_bins),
                    "semidense_depth_tolerance_m": float(self.semidense_depth_tolerance_m),
                    "semidense_depth_tolerance_rel": float(self.semidense_depth_tolerance_rel),
                    "semidense_visibility_radius": int(self.semidense_visibility_radius),
                    "semidense_depth_support_radius": int(self.semidense_depth_support_radius),
                    "semidense_depth_min_support": int(self.semidense_depth_min_support),
                    "semidense_use_depth_visibility": bool(self.semidense_use_depth_visibility),
                    "semidense_allow_sparse_depth_fallback": bool(
                        self.semidense_allow_sparse_depth_fallback
                    ),
                    "semidense_share_camera_projections": True,
                    "semidense_multiview_group_ratio": float(self.semidense_multiview_group_ratio),
                    "semidense_multiview_group_size": int(self.semidense_multiview_group_size),
                    "semidense_multiview_group_min_cameras": int(
                        self.semidense_multiview_group_min_cameras
                    ),
                    "semidense_multiview_nonvisible_ref_ratio": float(
                        self.semidense_multiview_nonvisible_ref_ratio
                    ),
                    "semidense_multiview_nonvisible_ref_strict": bool(
                        self.semidense_multiview_nonvisible_ref_strict
                    ),
                    "semidense_multiview_nonvisible_ref_mask_visibility": bool(
                        self.semidense_multiview_nonvisible_ref_mask_visibility
                    ),
                }
            )
            if semidense_filter_enabled:
                metadata["semidense_aria_wearer_centers_world"] = (
                    semidense_aria_wearer_centers_world
                )
        if egopose_metadata is not None:
            metadata.update(egopose_metadata)

        return UnifiedClip(
            dataset_name=self.dataset_name,
            sequence_name=sequence_name,
            frame_paths=frame_paths,
            images=images,
            depths=depths,
            normals=None,
            trajs_2d=None,
            trajs_3d_world=None,
            valids=None,
            visibs=None,
            intrinsics=np.stack(intrinsics, axis=0).astype(np.float32, copy=False),
            extrinsics=np.stack(extrinsics, axis=0).astype(np.float32, copy=False),
            metadata=metadata,
        )

    def _build_index(self) -> None:
        if not self.root.is_dir():
            if self.strict:
                raise FileNotFoundError(f"Ego-Exo4D root not found: {self.root}")
            return

        takes_path = self.root / "takes.json"
        splits_path = self.root / "annotations" / "splits.json"
        if not takes_path.exists() or not splits_path.exists():
            if self.strict:
                raise FileNotFoundError("Ego-Exo4D requires takes.json and annotations/splits.json")
            return

        takes = json.loads(takes_path.read_text())
        splits = json.loads(splits_path.read_text()).get("take_uid_to_split", {})

        records: list[_TakeRecord] = []
        for take in takes:
            if take.get("is_dropped"):
                continue
            take_uid = str(take.get("take_uid", ""))
            take_name = str(take.get("take_name") or Path(take.get("root_dir", "")).name)
            if not take_uid or not take_name:
                continue
            if (
                self.sequence_names is not None
                and take_name not in self.sequence_names
                and take_uid not in self.sequence_names
            ):
                continue
            split = str(splits.get(take_uid, "unknown"))
            if self.split not in {"all", "*"} and split != self.split:
                continue

            take_dir = self.root / str(take.get("root_dir", f"takes/{take_name}"))
            video_dir = take_dir / "frame_aligned_videos" / "downscaled" / str(self.downscaled_size)
            trajectory_dir = take_dir / "trajectory"
            if not video_dir.is_dir() or not trajectory_dir.is_dir():
                continue
            if self.require_depth and not (trajectory_dir / "semidense_points.csv.gz").exists():
                continue

            cameras = []
            for camera_id in self.cameras:
                if not (video_dir / f"{camera_id}.mp4").exists():
                    continue
                cameras.append(camera_id)
            if len(cameras) < self.min_cameras:
                continue

            pose_path = self._camera_pose_path_for_uid(take_uid, split)
            has_pose = pose_path is not None and pose_path.exists()
            if self.require_camera_pose and not has_pose:
                continue

            calibrated_cameras = self._calibrated_camera_ids_for_index(
                trajectory_dir,
                cameras,
                pose_path,
            )
            if len(calibrated_cameras) < self.min_cameras:
                continue

            duration_sec = float(take.get("duration_sec") or 0.0)
            num_frames = int(round(duration_sec * self.fps)) if duration_sec > 0 else 0
            if num_frames <= 0:
                first_video = video_dir / f"{calibrated_cameras[0]}.mp4"
                cap = cv2.VideoCapture(str(first_video))
                try:
                    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                finally:
                    cap.release()
            if num_frames <= 0:
                continue

            records.append(
                _TakeRecord(
                    take_name=take_name,
                    take_uid=take_uid,
                    split=split,
                    take_dir=take_dir,
                    video_dir=video_dir,
                    trajectory_dir=trajectory_dir,
                    cameras=tuple(calibrated_cameras),
                    num_frames=num_frames,
                )
            )

        records.sort(key=lambda item: item.take_name)
        self._records = records
        self._record_by_name = {record.take_name: record for record in records}

    def _get_record(self, sequence_name: str) -> _TakeRecord:
        try:
            return self._record_by_name[sequence_name]
        except KeyError as exc:
            raise KeyError(f"Unknown Ego-Exo4D take: {sequence_name}") from exc

    def _calibrated_camera_ids_for_index(
        self,
        trajectory_dir: Path,
        cameras: list[str],
        pose_path: Optional[Path],
    ) -> list[str]:
        wanted = set(cameras)
        calibrated: set[str] = set()
        gopro_path = trajectory_dir / "gopro_calibs.csv"
        if gopro_path.exists():
            with gopro_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    camera_id = str(row.get("cam_uid", "")).strip()
                    if camera_id not in wanted:
                        continue
                    if self._valid_gopro_calibration_row(row):
                        calibrated.add(camera_id)

        missing_pose_cameras = wanted - calibrated
        if missing_pose_cameras and pose_path is not None and pose_path.exists():
            pose = json.loads(pose_path.read_text())
            if isinstance(pose, dict):
                for camera_id in missing_pose_cameras:
                    item = pose.get(camera_id)
                    if (
                        isinstance(item, dict)
                        and "camera_intrinsics" in item
                        and "camera_extrinsics" in item
                    ):
                        calibrated.add(camera_id)

        return [camera_id for camera_id in cameras if camera_id in calibrated]

    @staticmethod
    def _valid_gopro_calibration_row(row: dict[str, str]) -> bool:
        required = [
            "tx_world_cam",
            "ty_world_cam",
            "tz_world_cam",
            "qx_world_cam",
            "qy_world_cam",
            "qz_world_cam",
            "qw_world_cam",
            "image_width",
            "image_height",
            "intrinsics_0",
            "intrinsics_1",
            "intrinsics_2",
            "intrinsics_3",
            "intrinsics_4",
            "intrinsics_5",
            "intrinsics_6",
            "intrinsics_7",
        ]
        try:
            for key in required:
                float(row[key])
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def _camera_pose_path_for_uid(self, take_uid: str, split: str) -> Optional[Path]:
        if not self.use_camera_pose_annotations:
            return None
        candidates = [
            self.root / "annotations" / "ego_pose" / split / "camera_pose" / f"{take_uid}.json",
            self.root / "annotations" / "ego_pose" / "train" / "camera_pose" / f"{take_uid}.json",
            self.root / "annotations" / "ego_pose" / "val" / "camera_pose" / f"{take_uid}.json",
            self.root / "annotations" / "ego_pose" / "test" / "camera_pose" / f"{take_uid}.json",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _load_egopose_jsons(self, record: _TakeRecord) -> tuple[dict, dict]:
        """Load manual and automatic EgoPose files with a small per-worker LRU."""
        cache_key = f"{record.split}:{record.take_uid}"
        with self._lock:
            cached = self._egopose_cache.get(cache_key)
            if cached is not None:
                self._egopose_cache.move_to_end(cache_key)
                return cached

        body_root = self.root / "annotations" / "ego_pose" / record.split / "body"

        def _read(kind: str) -> dict:
            path = body_root / kind / f"{record.take_uid}.json"
            if not path.is_file():
                return {}
            try:
                value = json.loads(path.read_text())
                return value if isinstance(value, dict) else {}
            except (OSError, json.JSONDecodeError):
                return {}

        value = (_read("annotation"), _read("automatic"))
        if self.egopose_cache_items > 0:
            with self._lock:
                self._egopose_cache[cache_key] = value
                self._egopose_cache.move_to_end(cache_key)
                while len(self._egopose_cache) > self.egopose_cache_items:
                    self._egopose_cache.popitem(last=False)
        return value

    def _load_egopose_clip(
        self,
        record: _TakeRecord,
        time_indices: list[int],
        selected_cameras: list[str],
    ) -> Optional[dict[str, Any]]:
        """Build compact dynamic body GT for just the sampled times/cameras.

        2D annotation coordinates are deliberately used only as observation-quality
        flags. Query pixels are reprojected from the 3D world GT with the rectified,
        cropped intrinsics later in QueryBuilder, avoiding distorted/raw-image
        coordinate mismatches.
        """
        manual, automatic = self._load_egopose_jsons(record)
        if not manual and not automatic:
            return None

        frame_people: list[list[dict]] = []
        frame_is_manual: list[bool] = []
        max_people = 0
        for frame_index in time_indices:
            key = str(int(frame_index))
            people = manual.get(key)
            from_manual = isinstance(people, list) and bool(people)
            if not from_manual:
                people = automatic.get(key)
            if not isinstance(people, list):
                people = []
            frame_people.append(people)
            frame_is_manual.append(from_manual)
            max_people = max(max_people, len(people))
        if max_people == 0:
            return None

        T = len(time_indices)
        C = len(selected_cameras)
        J = len(_EGOPOSE_JOINT_NAMES)
        points_world = np.full((T, max_people, J, 3), np.nan, dtype=np.float32)
        valid_3d = np.zeros((T, max_people, J), dtype=bool)
        observed_2d = np.zeros((C, T, max_people, J), dtype=bool)
        manual_2d = np.zeros_like(observed_2d)

        for ti, people in enumerate(frame_people):
            for pi, person in enumerate(people):
                if not isinstance(person, dict):
                    continue
                ann3d = person.get("annotation3D") or {}
                ann2d = person.get("annotation2D") or {}
                for ji, joint_name in enumerate(_EGOPOSE_JOINT_NAMES):
                    joint3d = ann3d.get(joint_name)
                    if isinstance(joint3d, dict):
                        xyz = np.asarray(
                            [joint3d.get("x"), joint3d.get("y"), joint3d.get("z")],
                            dtype=np.float32,
                        )
                        try:
                            views = int(joint3d.get("num_views_for_3d", 0))
                        except (TypeError, ValueError):
                            views = 0
                        if np.isfinite(xyz).all() and views >= self.egopose_min_views_for_3d:
                            points_world[ti, pi, ji] = xyz
                            valid_3d[ti, pi, ji] = True
                    for ci, camera_id in enumerate(selected_cameras):
                        camera_ann = ann2d.get(camera_id) or {}
                        joint2d = camera_ann.get(joint_name)
                        if not isinstance(joint2d, dict):
                            continue
                        try:
                            xy = np.asarray([joint2d.get("x"), joint2d.get("y")], dtype=np.float32)
                        except (TypeError, ValueError):
                            continue
                        if not np.isfinite(xy).all():
                            continue
                        observed_2d[ci, ti, pi, ji] = True
                        manual_2d[ci, ti, pi, ji] = (
                            str(joint2d.get("placement", "")).lower() == "manual"
                        )

        if not valid_3d.any():
            return None
        return {
            "egopose_direct_query": True,
            "egopose_query_ratio": float(self.egopose_query_ratio),
            "egopose_manual_only": bool(self.egopose_manual_only),
            "egopose_manual_group_ratio": float(self.egopose_manual_group_ratio),
            "egopose_group_size": int(self.egopose_group_size),
            "egopose_cross_time_ratio": float(self.egopose_cross_time_ratio),
            "egopose_3d_weight": float(self.egopose_3d_weight),
            "egopose_joint_names": list(_EGOPOSE_JOINT_NAMES),
            "egopose_points_world": points_world,
            "egopose_valid_3d": valid_3d,
            "egopose_observed_2d": observed_2d,
            "egopose_manual_2d": manual_2d,
            "egopose_annotation_frame_manual": np.asarray(frame_is_manual, dtype=bool),
        }

    def _load_aria_wearer_centers(
        self,
        record: _TakeRecord,
        time_indices: list[int],
    ) -> Optional[np.ndarray]:
        with self._lock:
            cached = self._aria_wearer_centers_cache.get(record.take_uid)
            cached_present = record.take_uid in self._aria_wearer_centers_cache
        if not cached_present:
            pose_path = (
                self.root
                / "annotations"
                / "ego_pose"
                / record.split
                / "camera_pose"
                / f"{record.take_uid}.json"
            )
            centers: Optional[dict[int, np.ndarray]] = {}
            try:
                pose = json.loads(pose_path.read_text())
                extrinsics = pose["aria01"]["camera_extrinsics"]
                if not isinstance(extrinsics, dict):
                    raise ValueError("invalid aria01 camera extrinsics")
                for frame_key, matrix in extrinsics.items():
                    matrix_array = np.asarray(matrix, dtype=np.float64)
                    if matrix_array.shape != (3, 4) or not np.all(np.isfinite(matrix_array)):
                        raise ValueError("invalid aria01 camera extrinsic")
                    rotation = matrix_array[:, :3]
                    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
                        raise ValueError("invalid aria01 camera rotation")
                    if np.linalg.det(rotation) <= 0.0:
                        raise ValueError("invalid aria01 camera rotation")
                    centers[int(frame_key)] = (-rotation.T @ matrix_array[:, 3]).astype(
                        np.float32
                    )
            except (
                OSError,
                UnicodeError,
                TypeError,
                ValueError,
                OverflowError,
                KeyError,
                np.linalg.LinAlgError,
                json.JSONDecodeError,
            ):
                centers = None
            with self._lock:
                self._aria_wearer_centers_cache[record.take_uid] = centers
            cached = centers

        if cached is None or any(int(index) not in cached for index in time_indices):
            return None
        return np.stack([cached[int(index)] for index in time_indices], axis=0).astype(
            np.float32, copy=False
        )

    @staticmethod
    def _sample_seed(take_uid: str, frame_indices: list[int]) -> int:
        payload = f"{take_uid}:{','.join(str(int(i)) for i in frame_indices)}".encode()
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")

    @staticmethod
    def _sample_time_indices(frame_indices: list[int], num_times: int, num_frames: int) -> list[int]:
        clipped = [min(max(int(i), 0), max(num_frames - 1, 0)) for i in frame_indices]
        if num_times <= 1:
            return [clipped[0]]
        if len(clipped) <= num_times:
            return clipped + [clipped[-1]] * (num_times - len(clipped))
        positions = np.linspace(0, len(clipped) - 1, num_times)
        return [clipped[int(round(pos))] for pos in positions]

    def _choose_num_times(
        self,
        rng: random.Random,
        clip_len: int,
        num_cameras: int,
    ) -> int:
        if not self.enforce_tubelet_aligned_layout:
            max_by_slots = max(1, int(clip_len) // max(1, int(num_cameras)))
            upper = min(max_by_slots, self.max_times if self.max_times is not None else max_by_slots)
            lower = min(self.min_times, upper)
            return rng.randint(lower, upper) if self.random_num_times and upper > lower else upper

        choices = valid_time_choices(
            num_slots=clip_len,
            num_cameras=num_cameras,
            min_times=self.min_times,
            max_times=self.max_times,
            tubelet_size=self.encoder_tubelet_size,
            even_multi_time=True,
        )
        if not choices:
            raise ValueError(
                f"No tubelet-aligned time count fits clip_len={clip_len}, "
                f"num_cameras={num_cameras}, min_times={self.min_times}, "
                f"max_times={self.max_times}"
            )
        return rng.choice(choices) if self.random_num_times else choices[-1]

    def _max_cameras_for_clip(self, clip_len: int) -> int:
        if not self.enforce_tubelet_aligned_layout:
            return max(1, int(clip_len) // max(1, self.min_times))
        minimum_logical_times = self.min_times
        if minimum_logical_times > 1 and minimum_logical_times % self.encoder_tubelet_size:
            minimum_logical_times += self.encoder_tubelet_size - (
                minimum_logical_times % self.encoder_tubelet_size
            )
        block = aligned_times_per_camera(
            minimum_logical_times,
            self.encoder_tubelet_size,
        )
        return max(1, int(clip_len) // block)

    def _build_slot_plan(self, clip_len: int, num_cameras: int, num_times: int):
        if self.enforce_tubelet_aligned_layout:
            return build_camera_major_slot_plan(
                num_slots=clip_len,
                num_cameras=num_cameras,
                num_times=num_times,
                tubelet_size=self.encoder_tubelet_size,
            )
        return build_camera_major_slot_plan(
            num_slots=clip_len,
            num_cameras=num_cameras,
            num_times=num_times,
            tubelet_size=1,
        )

    def _video_path(self, record: _TakeRecord, camera_id: str) -> Path:
        return record.video_dir / f"{camera_id}.mp4"

    @staticmethod
    def _video_hw(path: Path) -> tuple[int, int]:
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise OSError(f"Failed to open video: {path}")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise OSError(f"Failed to read first video frame: {path}")
                height, width = frame.shape[:2]
            return height, width
        finally:
            cap.release()

    def _load_calibrations(
        self,
        record: _TakeRecord,
        *,
        width: int,
        height: int,
    ) -> dict[str, _CameraCalib]:
        key = (record.take_uid, width, height)
        with self._lock:
            cached = self._calib_cache.get(key)
        if cached is not None:
            return cached

        calibs: dict[str, _CameraCalib] = {}
        calibs.update(self._load_gopro_calibrations(record, list(record.cameras)))

        missing = [camera_id for camera_id in record.cameras if camera_id not in calibs]
        pose_path = self._camera_pose_path_for_uid(record.take_uid, record.split)
        if missing and pose_path is not None and pose_path.exists():
            pose = json.loads(pose_path.read_text())
            for camera_id in missing:
                item = pose.get(camera_id)
                if not item:
                    continue
                intrinsic = np.asarray(item["camera_intrinsics"], dtype=np.float64)
                distortion = np.asarray(item.get("distortion_coeffs", [0, 0, 0, 0]), dtype=np.float64)
                extrinsic_3x4 = np.asarray(item["camera_extrinsics"], dtype=np.float64)
                extrinsic = np.eye(4, dtype=np.float64)
                extrinsic[:3, :4] = extrinsic_3x4
                orig_width = int(round(float(intrinsic[0, 2]) * 2.0)) or 3840
                orig_height = int(round(float(intrinsic[1, 2]) * 2.0)) or 2160
                calibs[camera_id] = _CameraCalib(
                    camera_id=camera_id,
                    intrinsic=intrinsic,
                    distortion=distortion[:4],
                    extrinsic=extrinsic,
                    image_width=orig_width,
                    image_height=orig_height,
                )

        scaled = {
            camera_id: self._scale_calibration(calib, width=width, height=height)
            for camera_id, calib in calibs.items()
            if camera_id in record.cameras
        }

        with self._lock:
            self._calib_cache[key] = scaled
        return scaled

    def _load_gopro_calibrations(
        self,
        record: _TakeRecord,
        camera_ids: list[str],
    ) -> dict[str, _CameraCalib]:
        path = record.trajectory_dir / "gopro_calibs.csv"
        if not path.exists():
            return {}
        wanted = set(camera_ids)
        out: dict[str, _CameraCalib] = {}
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                camera_id = row.get("cam_uid", "")
                if camera_id not in wanted:
                    continue
                intrinsic = np.array(
                    [
                        [float(row["intrinsics_0"]), 0.0, float(row["intrinsics_2"])],
                        [0.0, float(row["intrinsics_1"]), float(row["intrinsics_3"])],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                distortion = np.array(
                    [float(row[f"intrinsics_{idx}"]) for idx in range(4, 8)],
                    dtype=np.float64,
                )
                q = np.array(
                    [
                        float(row["qx_world_cam"]),
                        float(row["qy_world_cam"]),
                        float(row["qz_world_cam"]),
                        float(row["qw_world_cam"]),
                    ],
                    dtype=np.float64,
                )
                t = np.array(
                    [
                        float(row["tx_world_cam"]),
                        float(row["ty_world_cam"]),
                        float(row["tz_world_cam"]),
                    ],
                    dtype=np.float64,
                )
                rotation_c2w = self._quat_xyzw_to_matrix(q)
                rotation_w2c = rotation_c2w.T
                extrinsic = np.eye(4, dtype=np.float64)
                extrinsic[:3, :3] = rotation_w2c
                extrinsic[:3, 3] = -rotation_w2c @ t
                out[camera_id] = _CameraCalib(
                    camera_id=camera_id,
                    intrinsic=intrinsic,
                    distortion=distortion,
                    extrinsic=extrinsic,
                    image_width=int(float(row["image_width"])),
                    image_height=int(float(row["image_height"])),
                )
        return out

    @staticmethod
    def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
        x, y, z, w = q.astype(np.float64)
        norm = float(np.linalg.norm(q))
        if norm <= 0.0:
            return np.eye(3, dtype=np.float64)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _scale_calibration(calib: _CameraCalib, *, width: int, height: int) -> _CameraCalib:
        sx = float(width) / max(float(calib.image_width), 1.0)
        sy = float(height) / max(float(calib.image_height), 1.0)
        intrinsic = calib.intrinsic.copy()
        intrinsic[0, 0] *= sx
        intrinsic[0, 2] *= sx
        intrinsic[1, 1] *= sy
        intrinsic[1, 2] *= sy
        return _CameraCalib(
            camera_id=calib.camera_id,
            intrinsic=intrinsic,
            distortion=calib.distortion.copy(),
            extrinsic=calib.extrinsic.copy(),
            image_width=width,
            image_height=height,
        )

    def _rectified_intrinsic(
        self,
        record: _TakeRecord,
        camera_id: str,
        width: int,
        height: int,
        calib: _CameraCalib,
    ) -> np.ndarray:
        if not self.undistort:
            return calib.intrinsic.astype(np.float32, copy=True)
        _, _, rectified = self._rectify_maps(record, camera_id, width, height, calib)
        return rectified.astype(np.float32, copy=True)

    def _rectify_maps(
        self,
        record: _TakeRecord,
        camera_id: str,
        width: int,
        height: int,
        calib: _CameraCalib,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (record.take_uid, camera_id, width, height)
        with self._lock:
            cached = self._rectify_cache.get(key)
        if cached is not None:
            return cached

        K = calib.intrinsic.astype(np.float64)
        D = calib.distortion.astype(np.float64).reshape(4, 1)
        dim = (int(width), int(height))
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K,
            D,
            dim,
            np.eye(3),
            balance=self.undistort_balance,
            new_size=dim,
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K,
            D,
            np.eye(3),
            new_K,
            dim,
            cv2.CV_16SC2,
        )
        cached = (map1, map2, new_K.astype(np.float32))
        with self._lock:
            self._rectify_cache[key] = cached
        return cached

    def _read_video_frames(
        self,
        path: Path,
        indices: list[int],
        record: _TakeRecord,
        camera_id: str,
        width: int,
        height: int,
        calib: _CameraCalib,
    ) -> dict[int, np.ndarray]:
        unique_indices = sorted(set(int(i) for i in indices))
        out: dict[int, np.ndarray] = {}
        missing: list[int] = []
        for frame_index in unique_indices:
            cached = self._get_frame_cache(
                path,
                record,
                camera_id,
                frame_index,
                width,
                height,
            )
            if cached is None:
                cached = self._get_frame_disk_cache(
                    path,
                    record,
                    camera_id,
                    frame_index,
                    width,
                    height,
                )
            if cached is None:
                missing.append(frame_index)
            else:
                out[frame_index] = cached

        if not missing:
            return out

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise OSError(f"Failed to open Ego-Exo4D video: {path}")
        try:
            map_pair: Optional[tuple[np.ndarray, np.ndarray]] = None
            if self.undistort:
                map1, map2, _ = self._rectify_maps(record, camera_id, width, height, calib)
                map_pair = (map1, map2)

            if self._should_decode_sequential(missing):
                self._read_video_frames_sequential(
                    cap,
                    path,
                    missing,
                    record,
                    camera_id,
                    width,
                    height,
                    map_pair,
                    out,
                )
                missing = [frame_index for frame_index in missing if frame_index not in out]

            for frame_index in missing:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    raise OSError(f"Failed to read {path} frame {frame_index}")
                out[frame_index] = self._prepare_video_frame(
                    frame_bgr,
                    path,
                    record,
                    camera_id,
                    frame_index,
                    width,
                    height,
                    map_pair,
                )
            return out
        finally:
            cap.release()

    def _should_decode_sequential(self, missing: list[int]) -> bool:
        if self.video_read_strategy == "seek":
            return False
        if len(missing) < 4:
            return self.video_read_strategy == "sequential" and len(missing) > 1
        span = int(missing[-1]) - int(missing[0]) + 1
        if self.video_read_strategy == "sequential":
            return span > 1
        extra_frames = max(0, span - len(missing))
        return (
            extra_frames <= self.sequential_decode_max_extra
            and span <= len(missing) * self.sequential_decode_factor
        )

    def _read_video_frames_sequential(
        self,
        cap: cv2.VideoCapture,
        path: Path,
        missing: list[int],
        record: _TakeRecord,
        camera_id: str,
        width: int,
        height: int,
        map_pair: Optional[tuple[np.ndarray, np.ndarray]],
        out: dict[int, np.ndarray],
    ) -> None:
        wanted = set(missing)
        start = int(missing[0])
        end = int(missing[-1])
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for frame_index in range(start, end + 1):
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                if frame_index in wanted:
                    raise OSError(f"Failed to read {path} frame {frame_index}")
                break
            if frame_index not in wanted:
                continue
            out[frame_index] = self._prepare_video_frame(
                frame_bgr,
                path,
                record,
                camera_id,
                frame_index,
                width,
                height,
                map_pair,
            )

    def _prepare_video_frame(
        self,
        frame_bgr: np.ndarray,
        path: Path,
        record: _TakeRecord,
        camera_id: str,
        frame_index: int,
        width: int,
        height: int,
        map_pair: Optional[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
            frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
        if map_pair is not None:
            map1, map2 = map_pair
            frame_bgr = cv2.remap(frame_bgr, map1, map2, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._put_frame_cache(
            path,
            record,
            camera_id,
            frame_index,
            width,
            height,
            rgb,
        )
        self._put_frame_disk_cache(
            path,
            record,
            camera_id,
            frame_index,
            width,
            height,
            rgb,
        )
        return rgb.copy()

    def _frame_cache_key(
        self,
        path: Path,
        record: _TakeRecord,
        camera_id: str,
        frame_index: int,
        width: int,
        height: int,
    ) -> tuple[Any, ...]:
        return (
            "rgb",
            record.take_uid,
            str(path),
            camera_id,
            int(frame_index),
            int(width),
            int(height),
            bool(self.undistort),
            round(float(self.undistort_balance), 4),
        )

    def _get_frame_cache(
        self,
        path: Path,
        record: _TakeRecord,
        camera_id: str,
        frame_index: int,
        width: int,
        height: int,
    ) -> Optional[np.ndarray]:
        if self.frame_cache_items <= 0:
            return None
        key = self._frame_cache_key(path, record, camera_id, frame_index, width, height)
        with self._lock:
            cached = self._frame_cache.get(key)
            if cached is None:
                return None
            self._frame_cache.move_to_end(key)
            return cached.copy()

    def _put_frame_cache(
        self,
        path: Path,
        record: _TakeRecord,
        camera_id: str,
        frame_index: int,
        width: int,
        height: int,
        frame_rgb: np.ndarray,
    ) -> None:
        if self.frame_cache_items <= 0:
            return
        key = self._frame_cache_key(path, record, camera_id, frame_index, width, height)
        with self._lock:
            self._frame_cache[key] = frame_rgb
            self._frame_cache.move_to_end(key)
            while len(self._frame_cache) > self.frame_cache_items:
                self._frame_cache.popitem(last=False)

    def _frame_disk_cache_path(
        self,
        path: Path,
        record: _TakeRecord,
        camera_id: str,
        frame_index: int,
        width: int,
        height: int,
    ) -> Path:
        payload = {
            "v": 1,
            "take": record.take_uid,
            "video": str(path),
            "camera": camera_id,
            "w": int(width),
            "h": int(height),
            "undistort": bool(self.undistort),
            "balance": round(float(self.undistort_balance), 4),
            "quality": int(self.frame_disk_cache_jpeg_quality),
        }
        key = hashlib.blake2b(
            json.dumps(payload, sort_keys=True).encode(),
            digest_size=8,
        ).hexdigest()
        safe_uid = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_"
            for ch in record.take_uid
        )
        return (
            self.frame_disk_cache_dir
            / safe_uid
            / camera_id
            / f"{int(frame_index):08d}_{key}.jpg"
        )

    def _get_frame_disk_cache(
        self,
        path: Path,
        record: _TakeRecord,
        camera_id: str,
        frame_index: int,
        width: int,
        height: int,
    ) -> Optional[np.ndarray]:
        if not self.frame_disk_cache:
            return None
        cache_path = self._frame_disk_cache_path(
            path,
            record,
            camera_id,
            frame_index,
            width,
            height,
        )
        if not cache_path.exists():
            return None
        frame_bgr = cv2.imread(str(cache_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return None
        if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
            return None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._put_frame_cache(
            path,
            record,
            camera_id,
            frame_index,
            width,
            height,
            rgb,
        )
        return rgb

    def _put_frame_disk_cache(
        self,
        path: Path,
        record: _TakeRecord,
        camera_id: str,
        frame_index: int,
        width: int,
        height: int,
        frame_rgb: np.ndarray,
    ) -> None:
        if not self.frame_disk_cache:
            return
        cache_path = self._frame_disk_cache_path(
            path,
            record,
            camera_id,
            frame_index,
            width,
            height,
        )
        if cache_path.exists():
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_name(
                f".{cache_path.stem}.{os.getpid()}.{threading.get_ident()}.tmp.jpg"
            )
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(
                str(tmp_path),
                frame_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, self.frame_disk_cache_jpeg_quality],
            )
            if ok:
                os.replace(tmp_path, cache_path)
            elif tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _depth_cache_key(self, *, width: int, height: int) -> str:
        payload = {
            "w": int(width),
            "h": int(height),
            "quality_v": 2,
            "undistort": self.undistort,
            "balance": round(float(self.undistort_balance), 4),
            "max_points": int(self.max_points),
            "scan_limit": int(self.point_scan_limit),
            "max_inv": float(self.max_inv_dist_std),
            "max_dist": float(self.max_dist_std),
            "depth_max": float(self.depth_max_m),
            "dilate": int(self.depth_dilation_radius),
        }
        raw = json.dumps(payload, sort_keys=True).encode()
        return hashlib.blake2b(raw, digest_size=8).hexdigest()

    def _depth_cache_path(self, record: _TakeRecord, camera_id: str, width: int, height: int) -> Path:
        key = self._depth_cache_key(width=width, height=height)
        return self.depth_cache_dir / f"{record.take_uid}_{camera_id}_{key}.npz"

    def _points_cache_key(self) -> str:
        payload = {
            "quality_v": 2,
            "max_points": int(self.max_points),
            "scan_limit": int(self.point_scan_limit),
            "max_inv": float(self.max_inv_dist_std),
            "max_dist": float(self.max_dist_std),
        }
        raw = json.dumps(payload, sort_keys=True).encode()
        return hashlib.blake2b(raw, digest_size=8).hexdigest()

    def _points_cache_path(self, record: _TakeRecord) -> Path:
        safe_uid = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in record.take_uid)
        return self.points_cache_dir / f"{safe_uid}_{self._points_cache_key()}.npy"

    def _ensure_depth_cache(
        self,
        record: _TakeRecord,
        camera_ids: list[str],
        width: int,
        height: int,
        calibs: dict[str, _CameraCalib],
    ) -> None:
        missing = [
            camera_id
            for camera_id in camera_ids
            if not self._depth_cache_path(record, camera_id, width, height).exists()
        ]
        if not missing:
            return

        lock_path = self.depth_cache_dir / f"{record.take_uid}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            try:
                missing = [
                    camera_id
                    for camera_id in camera_ids
                    if not self._depth_cache_path(record, camera_id, width, height).exists()
                ]
                if not missing:
                    return
                points = self._load_filtered_points(record)
                for camera_id in missing:
                    calib = calibs[camera_id]
                    K = self._rectified_intrinsic(record, camera_id, width, height, calib)
                    depth = self._project_points_to_depth(points, K, calib.extrinsic, width, height)
                    self._save_depth(record, camera_id, width, height, depth)
            finally:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)

    def _load_filtered_points(self, record: _TakeRecord) -> np.ndarray:
        with self._lock:
            cached = self._points_cache.get(record.take_uid)
        if cached is not None:
            return cached

        cache_path = self._points_cache_path(record)
        if cache_path.exists():
            arr = np.load(cache_path).astype(np.float32, copy=False)
            with self._lock:
                self._points_cache[record.take_uid] = arr
            return arr

        lock_path = self.points_cache_dir / f"{record.take_uid}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            try:
                if cache_path.exists():
                    arr = np.load(cache_path).astype(np.float32, copy=False)
                    with self._lock:
                        self._points_cache[record.take_uid] = arr
                    return arr

                arr = self._read_filtered_points_csv(record)
                fd, tmp_name = tempfile.mkstemp(
                    prefix=f"{cache_path.stem}.",
                    suffix=".tmp",
                    dir=str(cache_path.parent),
                )
                os.close(fd)
                tmp_path = Path(tmp_name)
                try:
                    with tmp_path.open("wb") as handle:
                        np.save(handle, arr)
                    os.replace(tmp_path, cache_path)
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)
                with self._lock:
                    self._points_cache[record.take_uid] = arr
                return arr
            finally:
                fcntl.flock(lock_handle, fcntl.LOCK_UN)

    def _read_filtered_points_csv(self, record: _TakeRecord) -> np.ndarray:
        path = record.trajectory_dir / "semidense_points.csv.gz"
        points: list[list[float]] = []
        scanned = 0
        with gzip.open(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                scanned += 1
                try:
                    inv_std = float(row.get("inv_dist_std", 0.0))
                    dist_std = float(row.get("dist_std", 0.0))
                except ValueError:
                    continue
                if (
                    np.isfinite(inv_std)
                    and np.isfinite(dist_std)
                    and inv_std <= self.max_inv_dist_std
                    and dist_std <= self.max_dist_std
                ):
                    try:
                        points.append(
                            [
                                float(row["px_world"]),
                                float(row["py_world"]),
                                float(row["pz_world"]),
                            ]
                        )
                    except ValueError:
                        continue
                    if len(points) >= self.max_points:
                        break
                if self.point_scan_limit and scanned >= self.point_scan_limit:
                    break
        if not points:
            raise RuntimeError(f"[{record.take_name}] no usable Ego-Exo4D semidense points")
        return np.asarray(points, dtype=np.float32)

    def _project_points_to_depth(
        self,
        points_world: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        R = extrinsic[:3, :3].astype(np.float32, copy=False)
        t = extrinsic[:3, 3].astype(np.float32, copy=False)
        pts_cam = points_world @ R.T + t[None, :]
        z = pts_cam[:, 2]
        valid = (
            np.isfinite(pts_cam).all(axis=1)
            & (z > 1e-3)
            & (z < self.depth_max_m)
        )
        if not valid.any():
            return np.zeros((height, width), dtype=np.float32)

        pts = pts_cam[valid]
        z = pts[:, 2]
        K = intrinsic.astype(np.float32, copy=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * pts[:, 0] / z + K[0, 2]
            v = K[1, 1] * pts[:, 1] / z + K[1, 2]
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        in_bounds = (
            np.isfinite(u)
            & np.isfinite(v)
            & (ui >= 0)
            & (ui < width)
            & (vi >= 0)
            & (vi < height)
        )
        if not in_bounds.any():
            return np.zeros((height, width), dtype=np.float32)

        flat = (vi[in_bounds] * width + ui[in_bounds]).astype(np.int64, copy=False)
        depth_values = z[in_bounds].astype(np.float32, copy=False)
        depth_flat = np.full(height * width, np.inf, dtype=np.float32)
        np.minimum.at(depth_flat, flat, depth_values)
        depth = depth_flat.reshape(height, width)
        depth[~np.isfinite(depth)] = 0.0

        if self.depth_dilation_radius > 0 and np.any(depth > 0):
            radius = self.depth_dilation_radius
            kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
            valid_mask = depth > 0
            high = np.where(valid_mask, depth, self.depth_max_m + 1.0).astype(np.float32)
            min_depth = cv2.erode(high, kernel, iterations=1)
            dilated_valid = cv2.dilate(valid_mask.astype(np.uint8), kernel, iterations=1) > 0
            depth = np.where(dilated_valid, min_depth, 0.0).astype(np.float32, copy=False)
            depth[depth > self.depth_max_m] = 0.0
        return depth.astype(np.float32, copy=False)

    def _save_depth(
        self,
        record: _TakeRecord,
        camera_id: str,
        width: int,
        height: int,
        depth: np.ndarray,
    ) -> None:
        path = self._depth_cache_path(record, camera_id, width, height)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.stem}.",
            suffix=".tmp.npz",
            dir=str(path.parent),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            np.savez_compressed(tmp_path, depth=depth.astype(np.float32, copy=False))
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _load_depth(
        self,
        record: _TakeRecord,
        camera_id: str,
        width: int,
        height: int,
    ) -> np.ndarray:
        path = self._depth_cache_path(record, camera_id, width, height)
        key = ("depth", str(path))
        if self.depth_cache_items > 0:
            with self._lock:
                cached = self._depth_array_cache.get(key)
                if cached is not None:
                    self._depth_array_cache.move_to_end(key)
                    return cached.copy()
        with np.load(path) as data:
            depth = data["depth"].astype(np.float32, copy=False)
        if self.depth_cache_items > 0:
            with self._lock:
                cached = self._depth_array_cache.get(key)
                if cached is not None:
                    self._depth_array_cache.move_to_end(key)
                    return cached.copy()
                self._depth_array_cache[key] = depth
                self._depth_array_cache.move_to_end(key)
                while len(self._depth_array_cache) > self.depth_cache_items:
                    self._depth_array_cache.popitem(last=False)
        return depth.copy()
