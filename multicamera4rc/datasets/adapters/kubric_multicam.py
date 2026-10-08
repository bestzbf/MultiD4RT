from __future__ import annotations

import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .base import (
    BaseAdapter,
    UnifiedClip,
    canonical_anchor_frame_index,
    canonical_scene_anchor_metadata,
)
from ..multicam_slots import (
    aligned_times_per_camera,
    assert_no_cross_camera_tubelets,
    build_camera_major_slot_plan,
    valid_time_choices,
)


@dataclass
class _KubricMultiCamRecord:
    sequence_name: str
    h5_path: Path
    num_frames: int
    num_cameras: int
    height: int
    width: int
    camera_ids: list[str]
    fps: float | None = None
    has_depth: bool = False
    has_segmentation: bool = False
    has_tracks: bool = False
    num_tracks: int = 0
    schema_version: str = ""
    variant: str = ""
    has_dynamic_labels: bool = False


class KubricMultiCamAdapter(BaseAdapter):
    """Adapter for packed multi-camera Kubric H5 scenes.

    Expected H5 schema:
        frames:      [T, C, H, W, 3] uint8
        intrinsics:  [C, 3, 3]
        extrinsics:  [C, 4, 4] camera-to-world
        camera_ids:  [C]

    Optional native tracking schema:
        tracks_3d:   [N, T, 3] world coordinates
        tracks_2d:   [N, T, C, 2] pixel coordinates
        track_depth: [N, T, C] radial distance
        visibility:  [N, T, C]

    The adapter expands selected cameras and selected times into camera-major
    slots: slot = camera_index * num_times + time_index.  It pads invalid slots
    up to the requested clip length, matching the Ego-Exo4D multi-camera path.
    """

    dataset_name = "kubric_multicam"

    def __init__(
        self,
        root: str,
        split: str = "train",
        cache_dir: str | None = None,
        verbose: bool = True,
        index_workers: int = 8,
        split_strategy: str = "tail",
        val_ratio: float = 0.05,
        require_native_tracks: bool = False,
        require_depth: bool = False,
        required_schema_version: str | None = None,
        validation_stamp_dir: str | None = None,
        require_validation_stamp: bool = False,
        min_validation_revision: int = 0,
        min_cameras: int = 10,
        max_cameras: int = 10,
        min_times: int = 1,
        max_times: int | None = None,
        random_num_cameras: bool = False,
        random_num_times: bool = False,
        random_cameras: bool = False,
        random_camera_order: bool = False,
        camera_count_choices: list[int] | None = None,
        camera_count_weights: list[float] | None = None,
        single_time_probability: float = 0.0,
        use_direct_depth_queries: bool = True,
        load_depth: bool = True,
        depth_is_ray_distance: bool = True,
        extrinsics_are_opengl: bool = True,
        pixel_coordinates: str = "legacy_h5",
        direct_point_stride: int = 4,
        direct_point_cap: int = 60_000,
        semidense_projection_point_cap: int = 20_000,
        semidense_query_grid_size: int = 16,
        semidense_query_depth_bins: int = 8,
        semidense_depth_tolerance_m: float = 0.05,
        semidense_depth_tolerance_rel: float = 0.02,
        semidense_multiview_group_ratio: float = 0.75,
        semidense_multiview_group_size: int = 4,
        semidense_multiview_group_min_cameras: int = 2,
        semidense_multiview_balance_camera_pairs: bool = True,
        semidense_multiview_dynamic_group_ratio: float | None = None,
        dynamic_point_3d_weight: float = 1.0,
        ground_static_tracks: bool = False,
        ground_static_track_cap: int = 4096,
        ground_static_source_slots: int = 6,
        ground_static_source_distance_max_m: float | None = None,
        background_static_tracks: bool = False,
        background_static_track_cap: int = 4096,
        background_static_source_slots: int = 6,
        background_static_source_distance_max_m: float | None = None,
        ground_plane_z: float = 0.0,
        ground_plane_tolerance_m: float = 0.025,
        sampler_units: str = "time",
        encoder_tubelet_size: int = 2,
        strict: bool = True,
        **kwargs,
    ) -> None:
        self.root = Path(root)
        self.split = str(split)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.verbose = bool(verbose)
        self.index_workers = max(1, int(index_workers))
        self.split_strategy = str(split_strategy).strip().lower()
        if self.split_strategy not in {"tail", "variant_balanced"}:
            raise ValueError("split_strategy must be 'tail' or 'variant_balanced'")
        self.val_ratio = min(max(float(val_ratio), 0.0), 0.5)
        self.require_native_tracks = bool(require_native_tracks)
        self.require_depth = bool(require_depth)
        self.required_schema_version = (
            None
            if required_schema_version is None
            else str(required_schema_version).strip()
        )
        self.validation_stamp_dir = (
            Path(validation_stamp_dir) if validation_stamp_dir else None
        )
        self.require_validation_stamp = bool(require_validation_stamp)
        self.min_validation_revision = max(0, int(min_validation_revision))
        self.min_cameras = max(1, int(min_cameras))
        self.max_cameras = max(self.min_cameras, int(max_cameras))
        self.min_times = max(1, int(min_times))
        self.max_times = None if max_times is None else max(1, int(max_times))
        self.random_num_cameras = bool(random_num_cameras)
        self.random_num_times = bool(random_num_times)
        self.random_cameras = bool(random_cameras)
        self.random_camera_order = bool(random_camera_order)
        self.camera_count_choices = tuple(int(value) for value in (camera_count_choices or []))
        self.camera_count_weights = tuple(float(value) for value in (camera_count_weights or []))
        if self.camera_count_weights and len(self.camera_count_weights) != len(self.camera_count_choices):
            raise ValueError("camera_count_weights must match camera_count_choices")
        self.single_time_probability = min(max(float(single_time_probability), 0.0), 1.0)
        self.use_direct_depth_queries = bool(use_direct_depth_queries)
        self.load_depth = bool(load_depth)
        if self.use_direct_depth_queries and not self.load_depth:
            raise ValueError("direct depth queries require load_depth=True")
        self.depth_is_ray_distance = bool(depth_is_ray_distance)
        self.extrinsics_are_opengl = bool(extrinsics_are_opengl)
        if pixel_coordinates not in {"legacy_h5", "array_index_v1"}:
            raise ValueError("pixel_coordinates must be legacy_h5 or array_index_v1")
        self.pixel_coordinates = pixel_coordinates
        self.direct_point_stride = max(1, int(direct_point_stride))
        self.direct_point_cap = max(0, int(direct_point_cap))
        self.semidense_projection_point_cap = max(0, int(semidense_projection_point_cap))
        self.semidense_query_grid_size = max(2, int(semidense_query_grid_size))
        self.semidense_query_depth_bins = max(1, int(semidense_query_depth_bins))
        self.semidense_depth_tolerance_m = max(0.0, float(semidense_depth_tolerance_m))
        self.semidense_depth_tolerance_rel = max(0.0, float(semidense_depth_tolerance_rel))
        self.semidense_multiview_group_ratio = min(
            max(float(semidense_multiview_group_ratio), 0.0), 1.0
        )
        self.semidense_multiview_group_size = max(2, int(semidense_multiview_group_size))
        self.semidense_multiview_group_min_cameras = max(
            2, int(semidense_multiview_group_min_cameras)
        )
        self.semidense_multiview_balance_camera_pairs = bool(
            semidense_multiview_balance_camera_pairs
        )
        self.semidense_multiview_dynamic_group_ratio = (
            None
            if semidense_multiview_dynamic_group_ratio is None
            else min(max(float(semidense_multiview_dynamic_group_ratio), 0.0), 1.0)
        )
        self.dynamic_point_3d_weight = max(0.0, float(dynamic_point_3d_weight))
        self.ground_static_tracks = bool(ground_static_tracks)
        self.ground_static_track_cap = int(ground_static_track_cap)
        self.ground_static_source_slots = int(ground_static_source_slots)
        self.ground_static_source_distance_max_m = (
            None
            if ground_static_source_distance_max_m is None
            else float(ground_static_source_distance_max_m)
        )
        self.background_static_tracks = bool(background_static_tracks)
        self.background_static_track_cap = int(background_static_track_cap)
        self.background_static_source_slots = int(background_static_source_slots)
        self.background_static_source_distance_max_m = (
            None
            if background_static_source_distance_max_m is None
            else float(background_static_source_distance_max_m)
        )
        self.ground_plane_z = float(ground_plane_z)
        self.ground_plane_tolerance_m = float(ground_plane_tolerance_m)
        if self.ground_static_track_cap < 1:
            raise ValueError("ground_static_track_cap must be positive")
        if self.ground_static_source_slots < 1:
            raise ValueError("ground_static_source_slots must be positive")
        if self.background_static_track_cap < 1:
            raise ValueError("background_static_track_cap must be positive")
        if self.background_static_source_slots < 1:
            raise ValueError("background_static_source_slots must be positive")
        if (
            self.ground_static_source_distance_max_m is not None
            and (
                not np.isfinite(self.ground_static_source_distance_max_m)
                or self.ground_static_source_distance_max_m <= 0
            )
        ):
            raise ValueError(
                "ground_static_source_distance_max_m must be finite and positive"
            )
        if (
            self.background_static_source_distance_max_m is not None
            and (
                not np.isfinite(self.background_static_source_distance_max_m)
                or self.background_static_source_distance_max_m <= 0
            )
        ):
            raise ValueError(
                "background_static_source_distance_max_m must be finite and positive"
            )
        if not np.isfinite(self.ground_plane_z):
            raise ValueError("ground_plane_z must be finite")
        if (not np.isfinite(self.ground_plane_tolerance_m)
                or self.ground_plane_tolerance_m <= 0):
            raise ValueError("ground_plane_tolerance_m must be finite and positive")
        if self.ground_static_tracks and not self.require_depth:
            raise ValueError("ground static tracks require require_depth=True")
        if self.background_static_tracks and not self.require_depth:
            raise ValueError("background static tracks require require_depth=True")
        self.encoder_tubelet_size = max(1, int(encoder_tubelet_size))
        self.sampler_units = str(sampler_units).strip().lower()
        if self.sampler_units not in {"time", "slot"}:
            raise ValueError("sampler_units must be 'time' or 'slot'")

        if not self.root.exists():
            if strict:
                raise FileNotFoundError(f"KubricMultiCam root not found: {self.root}")
            self._records = []
        else:
            if self.require_validation_stamp and (
                self.validation_stamp_dir is None
                or not self.validation_stamp_dir.is_dir()
            ):
                raise FileNotFoundError(
                    "KubricMultiCam validation stamp directory not found: "
                    f"{self.validation_stamp_dir}"
                )
            if self.cache_dir is not None:
                from multicamera4rc.datasets.index_cache import load_or_build

                cache_key = {
                    "dataset": self.dataset_name,
                    "root": str(self.root.resolve()),
                    "cache_schema": (6 if self.background_static_tracks
                                     else 5 if self.ground_static_tracks else 4),
                    "require_native_tracks": self.require_native_tracks,
                    "require_depth": self.require_depth,
                    "required_schema_version": self.required_schema_version,
                    "validation_stamp_dir": (
                        str(self.validation_stamp_dir.resolve())
                        if self.validation_stamp_dir is not None
                        else None
                    ),
                    "require_validation_stamp": self.require_validation_stamp,
                    "min_validation_revision": self.min_validation_revision,
                }
                # Keep the legacy cache key byte-identical when the opt-in
                # ground path is disabled, including for a running V10 job.
                if self.ground_static_tracks:
                    cache_key["ground_static_tracks"] = True
                if self.background_static_tracks:
                    cache_key["background_static_tracks"] = True
                cache_suffix = hashlib.sha1(
                    json.dumps(cache_key, sort_keys=True).encode("utf-8")
                ).hexdigest()[:12]
                cache_path = (
                    self.cache_dir
                    / f"{self.dataset_name}_all_{cache_suffix}.pkl"
                )
                self._records = load_or_build(self._build_index, cache_path)
            else:
                self._records = self._build_index()

        all_records = self._records
        if self.split not in {"all", "*"} and len(all_records) > 1:
            self._records = self._split_records(all_records)

        self.sequence_names = [record.sequence_name for record in self._records]
        self._name_to_record = {record.sequence_name: record for record in self._records}

        if self.verbose:
            print(
                f"[KubricMultiCamAdapter] split={self.split!r}, "
                f"num_sequences={len(self.sequence_names)} "
                f"(total_indexed={len(all_records)}) root={self.root} "
                f"cameras={self.min_cameras}-{self.max_cameras} "
                f"times={self.min_times}-{self.max_times or 'auto'} "
                f"split_strategy={self.split_strategy}"
            )

    def __len__(self) -> int:
        return len(self.sequence_names)

    def list_sequences(self) -> list[str]:
        return self.sequence_names

    def get_sequence_name(self, index: int) -> str:
        return self._records[index].sequence_name

    def get_num_frames(self, sequence_name: str) -> int:
        record = self._get_record(sequence_name)
        if self.sampler_units == "slot":
            sampler_cameras = min(self.max_cameras, record.num_cameras)
            return int(record.num_frames * max(1, sampler_cameras))
        return int(record.num_frames)

    def get_canonical_anchor_frame_index(self, sequence_name: str) -> int:
        """Return the fixed source-time anchor (independent of clip/layout)."""

        return canonical_anchor_frame_index(
            self._get_record(sequence_name).num_frames
        )

    def get_canonical_anchor_sampler_index(self, sequence_name: str) -> int:
        """Express the fixed source time in the configured sampler units."""

        record = self._get_record(sequence_name)
        anchor_time = self.get_canonical_anchor_frame_index(sequence_name)
        if self.sampler_units == "slot":
            sampler_cameras = max(1, min(self.max_cameras, record.num_cameras))
            # Choose physical camera zero in the flattened sampler interval.
            # ``_sample_time_indices`` divides this by the same camera count.
            return int(anchor_time * sampler_cameras)
        return int(anchor_time)

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        record = self._get_record(sequence_name)
        return {
            "dataset_name": self.dataset_name,
            "sequence_name": sequence_name,
            "path": str(record.h5_path),
            "num_frames": self.get_num_frames(sequence_name),
            "num_source_frames": record.num_frames,
            "num_cameras": record.num_cameras,
            "fps": record.fps,
            "physical_time_unit": "seconds" if record.fps is not None else "frames",
            "height": record.height,
            "width": record.width,
            "has_depth": record.has_depth,
            "has_segmentation": record.has_segmentation,
            "has_normals": False,
            "has_tracks": record.has_tracks,
            "has_visibility": record.has_tracks,
            "has_trajs_3d_world": record.has_tracks,
            "num_tracks": record.num_tracks,
            "has_dynamic_labels": record.has_dynamic_labels,
            "schema_version": record.schema_version,
            "variant": record.variant,
            "intrinsics_shape": (record.num_cameras, 3, 3),
            "extrinsics_shape": (record.num_cameras, 4, 4),
            "extrinsics_convention": "w2c",
            "source_extrinsics_convention": (
                "opengl_c2w" if self.extrinsics_are_opengl else "opencv_c2w"
            ),
        }

    def load_clip(
        self,
        sequence_name: str,
        frame_indices: list[int],
        *,
        canonical_anchor_required: bool = False,
        layout_num_cameras: int | None = None,
        layout_num_times: int | None = None,
        sampling_seed: int | None = None,
        first_slot_min_visible_tracks: int = 0,
    ) -> UnifiedClip:
        if not frame_indices:
            raise ValueError("KubricMultiCamAdapter.load_clip requires non-empty frame_indices")

        import h5py

        record = self._get_record(sequence_name)
        clip_len = len(frame_indices)
        if sampling_seed is not None and isinstance(sampling_seed, bool):
            raise TypeError("sampling_seed must be an integer")
        if not isinstance(first_slot_min_visible_tracks, (int, np.integer)) or isinstance(
            first_slot_min_visible_tracks, (bool, np.bool_)
        ):
            raise TypeError("first_slot_min_visible_tracks must be an integer")
        first_slot_min_visible_tracks = int(first_slot_min_visible_tracks)
        if first_slot_min_visible_tracks < 0:
            raise ValueError("first_slot_min_visible_tracks must be nonnegative")
        if canonical_anchor_required and first_slot_min_visible_tracks:
            raise ValueError(
                "canonical_anchor_required and first_slot_min_visible_tracks are mutually exclusive"
            )
        rng = random.Random(
            self._sample_seed(sequence_name, frame_indices)
            if sampling_seed is None else int(sampling_seed)
        )

        available_cameras = list(range(record.num_cameras))
        sampler_cameras = min(self.max_cameras, record.num_cameras)
        max_cameras = min(
            self.max_cameras,
            record.num_cameras,
            clip_len // aligned_times_per_camera(self.min_times, self.encoder_tubelet_size),
        )
        min_cameras = min(self.min_cameras, max_cameras)
        if max_cameras <= 0:
            raise RuntimeError(f"[{sequence_name}] no usable camera slots")
        valid_camera_choices = [
            value
            for value in self.camera_count_choices
            if min_cameras <= value <= max_cameras
        ]
        if (layout_num_cameras is None) != (layout_num_times is None):
            raise ValueError(
                "layout_num_cameras and layout_num_times must be provided together"
            )
        if layout_num_cameras is not None:
            if isinstance(layout_num_cameras, bool) or isinstance(layout_num_times, bool):
                raise TypeError("explicit layout camera/time counts must be integers")
            num_cameras = int(layout_num_cameras)
            if num_cameras != layout_num_cameras or not min_cameras <= num_cameras <= max_cameras:
                raise ValueError(
                    f"explicit layout cameras={layout_num_cameras} do not fit "
                    f"the configured/available range {min_cameras}-{max_cameras}"
                )
        elif self.random_num_cameras and valid_camera_choices:
            if self.camera_count_weights:
                valid_indices = [
                    index
                    for index, value in enumerate(self.camera_count_choices)
                    if min_cameras <= value <= max_cameras
                ]
                valid_weights = [self.camera_count_weights[index] for index in valid_indices]
            else:
                valid_weights = None
            num_cameras = rng.choices(valid_camera_choices, weights=valid_weights, k=1)[0]
        elif self.random_num_cameras and max_cameras > min_cameras:
            num_cameras = rng.randint(min_cameras, max_cameras)
        else:
            num_cameras = max_cameras

        if self.random_cameras:
            selected_camera_indices = sorted(rng.sample(available_cameras, k=num_cameras))
        else:
            selected_camera_indices = available_cameras[:num_cameras]
        if self.random_camera_order:
            rng.shuffle(selected_camera_indices)

        # Canonical Stage-1 always uses the first physical camera.  Camera IDs
        # in ``slot_camera`` are clip-local layout IDs, so force the physical
        # camera into the selected subset before building that local layout.
        if canonical_anchor_required and 0 not in selected_camera_indices:
            selected_camera_indices[-1] = 0
            if self.random_camera_order:
                # Preserve random order apart from the mandatory replacement.
                if len(set(selected_camera_indices)) != len(selected_camera_indices):
                    raise RuntimeError(
                        f"[{self.dataset_name}/{sequence_name}] failed to select "
                        "a unique physical canonical camera"
                    )
            else:
                selected_camera_indices.sort()

        if layout_num_times is not None:
            num_times = int(layout_num_times)
            if num_times != layout_num_times or num_times < 1:
                raise ValueError("explicit layout time count must be a positive integer")
            if num_times > record.num_frames:
                raise ValueError(
                    f"explicit layout times={num_times} exceed scene frames={record.num_frames}"
                )
            if record.has_dynamic_labels and num_times < 2:
                raise ValueError("dynamic trajectories require at least two physical times")
            required_slots = num_cameras * aligned_times_per_camera(
                num_times, self.encoder_tubelet_size
            )
            if required_slots > clip_len:
                raise ValueError(
                    f"explicit layout {num_cameras}x{num_times} needs "
                    f"{required_slots} tubelet-safe slots, got {clip_len}"
                )
        else:
            num_times = self._choose_num_times(
                rng,
                clip_len,
                num_cameras,
                require_multiple_times=(
                    record.has_dynamic_labels
                    or (canonical_anchor_required and record.has_tracks)
                ),
            )
        time_indices = self._sample_time_indices(
            frame_indices,
            num_times,
            record.num_frames,
            slots_per_time=sampler_cameras,
        )
        if canonical_anchor_required:
            time_indices = self._inject_unique_canonical_time(
                time_indices,
                anchor_time=self.get_canonical_anchor_frame_index(sequence_name),
                num_frames=record.num_frames,
            )

        slot_plan = build_camera_major_slot_plan(
            num_slots=clip_len,
            num_cameras=num_cameras,
            num_times=num_times,
            tubelet_size=self.encoder_tubelet_size,
        )

        images: list[np.ndarray] = []
        intrinsics: list[np.ndarray] = []
        extrinsics: list[np.ndarray] = []
        frame_paths: list[str] = []
        slot_camera: list[int] = []
        slot_physical_camera_id: list[int] = []
        slot_time: list[float] = []
        slot_time_id: list[int] = []
        valid_slot: list[bool] = []
        depths: list[np.ndarray] | None = [] if record.has_depth and self.load_depth else None
        trajs_2d: list[np.ndarray] | None = [] if record.has_tracks else None
        trajs_3d_world: list[np.ndarray] | None = [] if record.has_tracks else None
        valids: list[np.ndarray] | None = [] if record.has_tracks else None
        visibs: list[np.ndarray] | None = [] if record.has_tracks else None
        visibility_valids: list[np.ndarray] | None = [] if record.has_tracks else None
        track_is_ground = np.zeros(record.num_tracks, dtype=bool)
        track_is_background = np.zeros(record.num_tracks, dtype=bool)
        ground_source_slots: list[int] = []
        background_source_slots: list[int] = []
        background_source_distance_min_m = None
        background_source_distance_mean_m = None
        background_source_distance_max_m = None
        first_slot_anchor_visible_tracks = 0
        first_slot_anchor_reordered = False
        first_slot_anchor_replaced_camera_id: int | None = None
        first_slot_anchor_requirement_met = first_slot_min_visible_tracks == 0

        with h5py.File(record.h5_path, "r") as hf:
            intrinsics_all = np.asarray(hf["intrinsics"], dtype=np.float32)
            c2w_all = np.asarray(hf["extrinsics"], dtype=np.float32)
            if self.extrinsics_are_opengl:
                camera_axis_flip = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
                c2w_all = np.matmul(c2w_all, camera_axis_flip)
            w2c_all = np.linalg.inv(c2w_all).astype(np.float32, copy=False)
            frames = hf["frames"]
            depth_ds = hf["depth"] if record.has_depth else None
            tracks_2d_ds = hf["tracks_2d"] if record.has_tracks else None
            tracks_3d_ds = hf["tracks_3d"] if record.has_tracks else None
            track_depth_ds = hf.get("track_depth") if record.has_tracks else None
            visibility_ds = hf["visibility"] if record.has_tracks else None
            track_is_dynamic = None
            track_object_ids = None
            if record.has_dynamic_labels:
                track_object_ids = np.asarray(hf["track_object_ids"], dtype=np.int64)
                object_is_dynamic = np.asarray(hf["object_is_dynamic"], dtype=bool)
                object_indices = track_object_ids - 1
                if (
                    object_indices.shape != (record.num_tracks,)
                    or np.any(object_indices < 0)
                    or np.any(object_indices >= len(object_is_dynamic))
                ):
                    raise RuntimeError(
                        f"[{sequence_name}] invalid track_object_ids/object_is_dynamic mapping"
                    )
                track_is_dynamic = np.ascontiguousarray(
                    object_is_dynamic[object_indices], dtype=bool
                )

            # V10 defines its shared coordinate gauge from slot 0. A random
            # camera order must not put an empty view first while another
            # selected camera has ample visible GT. Prefer an already selected
            # camera; only replace the weakest selected view if every selected
            # view is inadequate. Camera/time cardinality stays unchanged.
            if first_slot_min_visible_tracks:
                if (
                    visibility_ds is None
                    or tracks_2d_ds is None
                    or tracks_3d_ds is None
                ):
                    raise ValueError(
                        "first-slot visibility anchoring requires native tracks"
                    )
                first_time = int(time_indices[0])
                first_world = np.asarray(
                    tracks_3d_ds[:, first_time], dtype=np.float32
                )
                world_defined = np.isfinite(first_world).all(axis=1)
                camera_scores: dict[int, int] = {}

                def visible_track_count(camera_index: int) -> int:
                    camera_index = int(camera_index)
                    cached = camera_scores.get(camera_index)
                    if cached is not None:
                        return cached
                    xy = np.asarray(
                        tracks_2d_ds[:, first_time, camera_index],
                        dtype=np.float32,
                    )
                    visible = np.asarray(
                        visibility_ds[:, first_time, camera_index], dtype=bool
                    )
                    defined = world_defined & np.isfinite(xy).all(axis=1)
                    if track_depth_ds is not None:
                        point_depth = np.asarray(
                            track_depth_ds[:, first_time, camera_index],
                            dtype=np.float32,
                        )
                        defined &= np.isfinite(point_depth) & (point_depth > 0)
                    else:
                        camera_xyz = (
                            first_world @ w2c_all[camera_index, :3, :3].T
                            + w2c_all[camera_index, :3, 3]
                        )
                        defined &= (
                            np.isfinite(camera_xyz).all(axis=1)
                            & (camera_xyz[:, 2] > 1.0e-4)
                        )
                    count = int(np.count_nonzero(visible & defined))
                    camera_scores[camera_index] = count
                    return count

                selected_scores = [
                    visible_track_count(camera_index)
                    for camera_index in selected_camera_indices
                ]
                anchor_position = 0
                if selected_scores[0] < first_slot_min_visible_tracks:
                    anchor_position = max(
                        range(len(selected_camera_indices)),
                        key=lambda position: (selected_scores[position], -position),
                    )
                    if selected_scores[anchor_position] < first_slot_min_visible_tracks:
                        best_available_camera = max(
                            available_cameras,
                            key=lambda camera_index: (
                                visible_track_count(camera_index), -camera_index
                            ),
                        )
                        if (
                            visible_track_count(best_available_camera)
                            >= first_slot_min_visible_tracks
                            and best_available_camera not in selected_camera_indices
                        ):
                            replace_position = min(
                                range(len(selected_camera_indices)),
                                key=lambda position: (
                                    selected_scores[position], position
                                ),
                            )
                            first_slot_anchor_replaced_camera_id = int(
                                selected_camera_indices[replace_position]
                            )
                            selected_camera_indices[replace_position] = int(
                                best_available_camera
                            )
                            selected_scores[replace_position] = visible_track_count(
                                best_available_camera
                            )
                            anchor_position = replace_position
                if anchor_position:
                    selected_camera_indices.insert(
                        0, selected_camera_indices.pop(anchor_position)
                    )
                    selected_scores.insert(0, selected_scores.pop(anchor_position))
                    first_slot_anchor_reordered = True
                first_slot_anchor_visible_tracks = int(selected_scores[0])
                first_slot_anchor_requirement_met = (
                    first_slot_anchor_visible_tracks
                    >= first_slot_min_visible_tracks
                )
            world_track_cache: dict[int, np.ndarray] = {}

            zero_image = np.zeros(
                (record.height, record.width, 3), dtype=np.uint8
            )
            zero_depth = (
                np.zeros((record.height, record.width), dtype=np.float32)
                if record.has_depth
                else None
            )
            for camera_slot, camera_index in enumerate(selected_camera_indices):
                K = intrinsics_all[camera_index].astype(np.float32, copy=True)
                E = w2c_all[camera_index].astype(np.float32, copy=True)
                physical_times = slot_plan.physical_times_per_camera
                for physical_time in range(physical_times):
                    slot = camera_slot * physical_times + physical_time
                    is_valid = bool(slot_plan.valid_slot[slot])
                    time_slot = int(slot_plan.slot_time[slot])
                    frame_index = time_indices[time_slot]
                    image = (
                        np.asarray(frames[int(frame_index), int(camera_index)], dtype=np.uint8)
                        if is_valid
                        else zero_image
                    )
                    images.append(np.ascontiguousarray(image))
                    intrinsics.append(K)
                    extrinsics.append(E)
                    frame_paths.append(
                        f"{record.h5_path}#t={frame_index},cam={camera_index}"
                        if is_valid else ""
                    )
                    slot_camera.append(int(slot_plan.slot_camera[slot]))
                    slot_physical_camera_id.append(int(camera_index))
                    slot_time_id.append(time_slot)
                    slot_time.append(
                        float(frame_index) / float(record.fps)
                        if record.fps is not None
                        else float(frame_index)
                    )
                    valid_slot.append(is_valid)
                    if depths is not None and depth_ds is not None:
                        d = (
                            np.asarray(
                                depth_ds[int(frame_index), int(camera_index)],
                                dtype=np.float32,
                            )
                            if is_valid
                            else zero_depth
                        )
                        if is_valid and self.depth_is_ray_distance:
                            d = self._ray_distance_to_z_depth(d, K)
                        depths.append(d)
                    if (
                        trajs_2d is not None
                        and trajs_3d_world is not None
                        and valids is not None
                        and visibs is not None
                        and visibility_valids is not None
                        and tracks_2d_ds is not None
                        and tracks_3d_ds is not None
                        and visibility_ds is not None
                    ):
                        if is_valid:
                            xy = np.asarray(
                                tracks_2d_ds[:, int(frame_index), int(camera_index)],
                                dtype=np.float32,
                            )
                            world = world_track_cache.get(int(frame_index))
                            if world is None:
                                world = np.asarray(
                                    tracks_3d_ds[:, int(frame_index)], dtype=np.float32
                                )
                                world_track_cache[int(frame_index)] = world
                            visible = np.asarray(
                                visibility_ds[:, int(frame_index), int(camera_index)],
                                dtype=bool,
                            )
                            if track_depth_ds is not None:
                                point_depth = np.asarray(
                                    track_depth_ds[
                                        :, int(frame_index), int(camera_index)
                                    ],
                                    dtype=np.float32,
                                )
                                defined = (
                                    np.isfinite(point_depth) & (point_depth > 0)
                                )
                            else:
                                camera_xyz = (
                                    world @ E[:3, :3].T + E[:3, 3]
                                )
                                defined = (
                                    np.isfinite(camera_xyz).all(axis=1)
                                    & (camera_xyz[:, 2] > 1e-4)
                                )
                            defined &= np.isfinite(xy).all(axis=1)
                            defined &= np.isfinite(world).all(axis=1)
                        else:
                            xy = np.full((record.num_tracks, 2), np.nan, np.float32)
                            world = np.full((record.num_tracks, 3), np.nan, np.float32)
                            visible = np.zeros(record.num_tracks, dtype=bool)
                            defined = np.zeros(record.num_tracks, dtype=bool)
                        trajs_2d.append(np.ascontiguousarray(xy))
                        trajs_3d_world.append(np.ascontiguousarray(world))
                        valids.append(np.ascontiguousarray(defined))
                        visibs.append(np.ascontiguousarray(visible & defined))
                        visibility_valids.append(np.ascontiguousarray(defined))

            while len(images) < clip_len:
                slot = len(images)
                camera_slot = int(slot_plan.slot_camera[slot])
                camera_index = selected_camera_indices[camera_slot]
                images.append(zero_image.copy())
                intrinsics.append(intrinsics_all[camera_index].astype(np.float32, copy=True))
                extrinsics.append(w2c_all[camera_index].astype(np.float32, copy=True))
                frame_paths.append("")
                slot_camera.append(int(slot_plan.slot_camera[slot]))
                slot_physical_camera_id.append(int(camera_index))
                time_slot = int(slot_plan.slot_time[slot])
                frame_index = time_indices[time_slot]
                slot_time_id.append(time_slot)
                slot_time.append(
                    float(frame_index) / float(record.fps)
                    if record.fps is not None
                    else float(frame_index)
                )
                valid_slot.append(False)
                if depths is not None:
                    depths.append(zero_depth.copy())
                if trajs_2d is not None:
                    trajs_2d.append(
                        np.full((record.num_tracks, 2), np.nan, np.float32)
                    )
                    trajs_3d_world.append(
                        np.full((record.num_tracks, 3), np.nan, np.float32)
                    )
                    valids.append(np.zeros(record.num_tracks, dtype=bool))
                    visibs.append(np.zeros(record.num_tracks, dtype=bool))
                    visibility_valids.append(
                        np.zeros(record.num_tracks, dtype=bool)
                    )

        if not images:
            raise RuntimeError(f"[{sequence_name}] no images loaded")

        if len(images) > clip_len:
            images = images[:clip_len]
            intrinsics = intrinsics[:clip_len]
            extrinsics = extrinsics[:clip_len]
            frame_paths = frame_paths[:clip_len]
            slot_camera = slot_camera[:clip_len]
            slot_physical_camera_id = slot_physical_camera_id[:clip_len]
            slot_time = slot_time[:clip_len]
            slot_time_id = slot_time_id[:clip_len]
            valid_slot = valid_slot[:clip_len]
            if depths is not None:
                depths = depths[:clip_len]
            if trajs_2d is not None:
                trajs_2d = trajs_2d[:clip_len]
                trajs_3d_world = trajs_3d_world[:clip_len]
                valids = valids[:clip_len]
                visibs = visibs[:clip_len]
                visibility_valids = visibility_valids[:clip_len]

        if self.ground_static_tracks:
            if not record.has_tracks or trajs_2d is None:
                raise RuntimeError(
                    f"[{sequence_name}] ground supervision requires native tracks"
                )
            ground = self._sample_ground_static_tracks(
                h5_path=record.h5_path,
                frame_indices=time_indices,
                slot_time_id=slot_time_id,
                slot_physical_camera_id=slot_physical_camera_id,
                slot_camera=slot_camera,
                valid_slot=valid_slot,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                rng=rng,
            )
            ground_count = int(ground["world"].shape[1])
            if ground_count < 1:
                distance_clause = (
                    ""
                    if self.ground_static_source_distance_max_m is None
                    else (
                        ", source_distance <= "
                        f"{self.ground_static_source_distance_max_m:g}m"
                    )
                )
                raise RuntimeError(
                    f"[{sequence_name}] no depth/segmentation ground points satisfy "
                    f"|world_z-{self.ground_plane_z:g}| <= "
                    f"{self.ground_plane_tolerance_m:g}m{distance_clause}"
                )
            for slot in range(clip_len):
                trajs_2d[slot] = np.concatenate(
                    (trajs_2d[slot], ground["uv"][slot]), axis=0
                )
                trajs_3d_world[slot] = np.concatenate(
                    (trajs_3d_world[slot], ground["world"][slot]), axis=0
                )
                valids[slot] = np.concatenate(
                    (valids[slot], ground["valid"][slot]), axis=0
                )
                visibs[slot] = np.concatenate(
                    (visibs[slot], ground["visible"][slot]), axis=0
                )
                visibility_valids[slot] = np.concatenate(
                    (visibility_valids[slot], ground["valid"][slot]), axis=0
                )
            track_is_ground = np.concatenate(
                (track_is_ground, np.ones(ground_count, dtype=bool))
            )
            track_is_background = np.concatenate(
                (track_is_background, np.zeros(ground_count, dtype=bool))
            )
            ground_source_slots = [int(value) for value in ground["source_slots"]]
            ground_source_distance_min_m = float(
                np.min(ground["source_distances_m"])
            )
            ground_source_distance_mean_m = float(
                np.mean(ground["source_distances_m"])
            )
            ground_source_distance_max_m = float(
                np.max(ground["source_distances_m"])
            )
            if track_is_dynamic is not None:
                track_is_dynamic = np.concatenate(
                    (track_is_dynamic, np.zeros(ground_count, dtype=bool))
                )

        if self.background_static_tracks:
            if not record.has_tracks or trajs_2d is None:
                raise RuntimeError(
                    f"[{sequence_name}] background supervision requires native tracks"
                )
            background = self._sample_ground_static_tracks(
                h5_path=record.h5_path,
                frame_indices=time_indices,
                slot_time_id=slot_time_id,
                slot_physical_camera_id=slot_physical_camera_id,
                slot_camera=slot_camera,
                valid_slot=valid_slot,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                rng=rng,
                select_ground=False,
                track_cap=self.background_static_track_cap,
                source_slots=self.background_static_source_slots,
                source_distance_max_m=self.background_static_source_distance_max_m,
            )
            background_count = int(background["world"].shape[1])
            if background_count < 1:
                raise RuntimeError(
                    f"[{sequence_name}] no depth/segmentation non-ground background "
                    "points satisfy the configured distance limit"
                )
            for slot in range(clip_len):
                trajs_2d[slot] = np.concatenate(
                    (trajs_2d[slot], background["uv"][slot]), axis=0
                )
                trajs_3d_world[slot] = np.concatenate(
                    (trajs_3d_world[slot], background["world"][slot]), axis=0
                )
                valids[slot] = np.concatenate(
                    (valids[slot], background["valid"][slot]), axis=0
                )
                visibs[slot] = np.concatenate(
                    (visibs[slot], background["visible"][slot]), axis=0
                )
                visibility_valids[slot] = np.concatenate(
                    (visibility_valids[slot], background["valid"][slot]), axis=0
                )
            track_is_ground = np.concatenate(
                (track_is_ground, np.zeros(background_count, dtype=bool))
            )
            track_is_background = np.concatenate(
                (track_is_background, np.ones(background_count, dtype=bool))
            )
            background_source_slots = [
                int(value) for value in background["source_slots"]
            ]
            background_source_distance_min_m = float(
                np.min(background["source_distances_m"])
            )
            background_source_distance_mean_m = float(
                np.mean(background["source_distances_m"])
            )
            background_source_distance_max_m = float(
                np.max(background["source_distances_m"])
            )
            if track_is_dynamic is not None:
                track_is_dynamic = np.concatenate(
                    (track_is_dynamic, np.zeros(background_count, dtype=bool))
                )

        assert_no_cross_camera_tubelets(
            np.asarray(slot_camera),
            np.asarray(valid_slot),
            tubelet_size=self.encoder_tubelet_size,
        )

        direct_points_world = None
        if self.use_direct_depth_queries and depths is not None and not record.has_tracks:
            direct_points_world = self._depths_to_world_points(
                depths,
                intrinsics,
                extrinsics,
                valid_slot,
                rng,
            )

        selected_camera_ids = [record.camera_ids[i] for i in selected_camera_indices]
        rigid_object_ids = np.full(len(track_is_ground), -1, dtype=np.int64)
        if track_object_ids is not None:
            rigid_object_ids[:record.num_tracks] = track_object_ids
        metadata = {
            "track_rigid_object_id": rigid_object_ids,
            "dataset_name": self.dataset_name,
            "sequence_root": str(record.h5_path.parent),
            "h5_path": str(record.h5_path),
            "frame_indices": list(frame_indices),
            "time_indices": list(time_indices),
            "sampler_cameras_per_time": int(sampler_cameras),
            "selected_camera_indices": list(selected_camera_indices),
            "selected_cameras": list(selected_camera_ids),
            "has_depth": record.has_depth,
            "has_normals": False,
            "has_tracks": record.has_tracks,
            "has_visibility": record.has_tracks,
            "has_trajs_3d_world": record.has_tracks,
            "schema_version": record.schema_version,
            "variant": record.variant,
            "pose_convention": "w2c",
            "extrinsics_convention": "w2c",
            "source_extrinsics_convention": (
                "opengl_c2w" if self.extrinsics_are_opengl else "opencv_c2w"
            ),
            "intrinsics_convention": "pinhole",
            "layout_mode": "provided_multicam",
            "provided_slot_layout": True,
            "allow_rgb_only_multicam_layout": True,
            "slot_camera": slot_camera,
            "slot_physical_camera_id": slot_physical_camera_id,
            "slot_time": slot_time,
            "slot_time_id": slot_time_id,
            "valid_slot": valid_slot,
            "fps": record.fps,
            "physical_time_unit": "seconds" if record.fps is not None else "frames",
            "layout_num_cameras": int(num_cameras),
            "layout_num_times": int(num_times),
            "layout_valid_slots": int(sum(valid_slot)),
            "layout_physical_times_per_camera": int(
                slot_plan.physical_times_per_camera
            ),
            "layout_tubelet_size": int(self.encoder_tubelet_size),
            "layout_tubelet_aligned": True,
            "first_slot_anchor_min_visible_tracks": int(
                first_slot_min_visible_tracks
            ),
            "first_slot_anchor_visible_tracks": int(
                first_slot_anchor_visible_tracks
            ),
            "first_slot_anchor_reordered": bool(first_slot_anchor_reordered),
            "first_slot_anchor_replaced": bool(
                first_slot_anchor_replaced_camera_id is not None
            ),
            "first_slot_anchor_replaced_camera_id": (
                None
                if first_slot_anchor_replaced_camera_id is None
                else int(first_slot_anchor_replaced_camera_id)
            ),
            "first_slot_anchor_requirement_met": bool(
                first_slot_anchor_requirement_met
            ),
            "slot_source_indices": list(range(clip_len)),
            "query_semantics": (
                "kubric_multicam_tracks"
                if record.has_tracks
                else (
                    "kubric_multicam_depth_direct"
                    if direct_points_world is not None
                    else "kubric_multicam_rgb_only"
                )
            ),
            "has_temporal_supervision": record.has_tracks,
            "multicam_track_query": record.has_tracks,
            "has_dynamic_point_labels": bool(track_is_dynamic is not None),
            "infer_dynamic_from_authoritative_world_trajectories": bool(
                record.has_tracks and track_is_dynamic is None
            ),
            "dynamic_motion_label_schema": (
                "adapter_explicit_binary_with_known_mask_v1"
                if track_is_dynamic is not None
                else (
                    "world_displacement_three_state_strict_static_unknown_band_"
                    "distinct_time_v2"
                    if record.has_tracks
                    else None
                )
            ),
            "track_multiview_group_ratio": self.semidense_multiview_group_ratio,
            "track_multiview_group_size": self.semidense_multiview_group_size,
            "track_multiview_group_min_cameras": self.semidense_multiview_group_min_cameras,
            "track_multiview_dynamic_group_ratio": self.semidense_multiview_dynamic_group_ratio,
            "track_dynamic_point_3d_weight": self.dynamic_point_3d_weight,
            "track_is_ground": track_is_ground,
            "track_is_background": track_is_background,
            "ground_static_track_count": int(track_is_ground.sum()),
            "ground_source_slots": ground_source_slots,
            "ground_static_source_distance_limit_m": (
                self.ground_static_source_distance_max_m
            ),
            "ground_source_distance_min_m": (
                ground_source_distance_min_m
                if self.ground_static_tracks else None
            ),
            "ground_source_distance_mean_m": (
                ground_source_distance_mean_m
                if self.ground_static_tracks else None
            ),
            "ground_source_distance_max_m": (
                ground_source_distance_max_m
                if self.ground_static_tracks else None
            ),
            "ground_label_schema": (
                "segmentation_zero_depth_world_plane_source_distance_v2"
                if self.ground_static_tracks else None
            ),
            "background_static_track_count": int(track_is_background.sum()),
            "background_source_slots": background_source_slots,
            "background_static_source_distance_limit_m": (
                self.background_static_source_distance_max_m
            ),
            "background_source_distance_min_m": background_source_distance_min_m,
            "background_source_distance_mean_m": background_source_distance_mean_m,
            "background_source_distance_max_m": background_source_distance_max_m,
            "background_label_schema": (
                "segmentation_zero_depth_non_ground_source_distance_v1"
                if self.background_static_tracks else None
            ),
            "raw_image_hw": [record.height, record.width],
            "rgb_hw": [record.height, record.width],
            "target_hw": [record.height, record.width],
        }
        if canonical_anchor_required:
            anchor_time_index = self.get_canonical_anchor_frame_index(sequence_name)
            anchor_physical_time = (
                float(anchor_time_index) / float(record.fps)
                if record.fps is not None
                else float(anchor_time_index)
            )
            anchor_matches = [
                slot
                for slot, (physical_camera, physical_time, is_valid) in enumerate(
                    zip(slot_physical_camera_id, slot_time, valid_slot)
                )
                if is_valid
                and physical_camera == 0
                and np.isclose(
                    physical_time,
                    anchor_physical_time,
                    rtol=0.0,
                    atol=1.0e-12 * max(1.0, abs(anchor_physical_time)),
                )
            ]
            if len(anchor_matches) != 1:
                raise RuntimeError(
                    f"[{self.dataset_name}/{sequence_name}] canonical physical "
                    "camera/time must map to exactly one valid layout slot; got "
                    f"{len(anchor_matches)} matches"
                )
            metadata.update(
                canonical_scene_anchor_metadata(
                    extrinsic=w2c_all[0],
                    time=anchor_physical_time,
                    camera_id=0,
                    frame_id=(
                        f"{self.dataset_name}/{sequence_name}/"
                        f"physical_camera=0/frame={anchor_time_index}"
                    ),
                    anchor_slot=anchor_matches[0],
                    num_slots=clip_len,
                    require_in_clip=True,
                )
            )
        if direct_points_world is not None:
            metadata.update(
                {
                    "semidense_direct_query": True,
                    "semidense_points_world": direct_points_world,
                    "semidense_projection_point_cap": self.semidense_projection_point_cap,
                    "semidense_query_grid_size": self.semidense_query_grid_size,
                    "semidense_query_depth_bins": self.semidense_query_depth_bins,
                    "semidense_depth_tolerance_m": self.semidense_depth_tolerance_m,
                    "semidense_depth_tolerance_rel": self.semidense_depth_tolerance_rel,
                    "semidense_visibility_radius": 0,
                    "semidense_depth_support_radius": 0,
                    "semidense_depth_min_support": 0,
                    "semidense_use_depth_visibility": True,
                    "semidense_allow_sparse_depth_fallback": False,
                    "semidense_multiview_group_ratio": self.semidense_multiview_group_ratio,
                    "semidense_multiview_group_size": self.semidense_multiview_group_size,
                    "semidense_multiview_group_min_cameras": self.semidense_multiview_group_min_cameras,
                    "semidense_multiview_balance_camera_pairs": self.semidense_multiview_balance_camera_pairs,
                    "semidense_multiview_nonvisible_ref_ratio": 0.0,
                }
            )

        # All H5/depth/ground construction above uses renderer pixel centres
        # (x+.5,y+.5). Convert ONCE, at the image-array interface. Applying
        # this after ground augmentation also corrects its generated queries.
        # Also tolerate adapters pickled by already-running legacy parents.
        pixel_coordinates = getattr(self, "pixel_coordinates", "legacy_h5")
        metadata["pixel_coordinates"] = pixel_coordinates
        if pixel_coordinates == "array_index_v1":
            shift = np.array([[1., 0., -.5], [0., 1., -.5], [0., 0., 1.]])
            intrinsics = [shift @ value for value in intrinsics]
            if trajs_2d is not None:
                trajs_2d = [value - .5 for value in trajs_2d]

        return UnifiedClip(
            dataset_name=self.dataset_name,
            sequence_name=sequence_name,
            frame_paths=frame_paths,
            images=images,
            depths=depths if depths else None,
            normals=None,
            trajs_2d=(
                np.stack(trajs_2d, axis=0).astype(np.float32, copy=False)
                if trajs_2d is not None else None
            ),
            trajs_3d_world=(
                np.stack(trajs_3d_world, axis=0).astype(np.float32, copy=False)
                if trajs_3d_world is not None else None
            ),
            valids=np.stack(valids, axis=0) if valids is not None else None,
            visibs=np.stack(visibs, axis=0) if visibs is not None else None,
            intrinsics=np.stack(intrinsics, axis=0).astype(np.float32, copy=False),
            extrinsics=np.stack(extrinsics, axis=0).astype(np.float32, copy=False),
            metadata=metadata,
            visibility_valids=(
                np.stack(visibility_valids, axis=0)
                if visibility_valids is not None else None
            ),
            track_is_dynamic=track_is_dynamic,
            track_dynamic_known=(
                np.ones_like(track_is_dynamic, dtype=bool)
                if track_is_dynamic is not None
                else None
            ),
        )

    def _sample_ground_static_tracks(
        self,
        *,
        h5_path: Path,
        frame_indices: list[int],
        slot_time_id: list[int],
        slot_physical_camera_id: list[int],
        slot_camera: list[int],
        valid_slot: list[bool],
        intrinsics: list[np.ndarray],
        extrinsics: list[np.ndarray],
        rng: random.Random,
        select_ground: bool = True,
        track_cap: int | None = None,
        source_slots: int | None = None,
        source_distance_max_m: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Lift verified floor pixels into static world-space trajectories.

        Kubric's native object tracks intentionally omit background label zero.
        We therefore sample the rendered floor directly from segmentation and
        depth.  Each added identity has one depth-verified visible source view;
        its constant world point is then valid in every camera where it lies in
        front of the camera.  We never invent visibility in another view.
        """
        import h5py

        num_slots = len(valid_slot)
        aligned = (
            len(slot_time_id), len(slot_physical_camera_id), len(slot_camera),
            len(intrinsics), len(extrinsics),
        )
        if any(length != num_slots for length in aligned):
            raise RuntimeError("ground supervision slot metadata is not aligned")
        available = [index for index, value in enumerate(valid_slot) if value]
        if not available:
            raise RuntimeError("ground supervision has no valid input slots")

        # Select a small, deterministic set of views that spans both physical
        # time and cameras. Randomness is reserved for spatial pixel sampling.
        track_cap = self.ground_static_track_cap if track_cap is None else int(track_cap)
        source_slots = (
            self.ground_static_source_slots if source_slots is None else int(source_slots)
        )
        if source_distance_max_m is None and select_ground:
            source_distance_max_m = self.ground_static_source_distance_max_m
        source_count = min(source_slots, len(available))
        time_values = sorted({int(slot_time_id[index]) for index in available})
        camera_values = sorted({int(slot_camera[index]) for index in available})
        time_rank = {value: index for index, value in enumerate(time_values)}
        desired_times = np.linspace(0, len(time_values) - 1, source_count)
        unused = set(available)
        proposed_sources: list[int] = []
        for source_index, desired_time in enumerate(desired_times):
            desired_camera = camera_values[source_index % len(camera_values)]
            selected = min(
                unused,
                key=lambda slot: (
                    int(slot_camera[slot]) != desired_camera,
                    abs(time_rank[int(slot_time_id[slot])] - float(desired_time)),
                    int(slot_time_id[slot]),
                    slot,
                ),
            )
            proposed_sources.append(int(selected))
            unused.remove(selected)

        candidate_world: list[np.ndarray] = []
        candidate_source: list[int] = []
        candidate_source_distances: list[np.ndarray] = []
        with h5py.File(h5_path, "r") as hf:
            if "depth" not in hf or "segmentation" not in hf:
                raise RuntimeError(
                    f"[{h5_path.stem}] ground supervision requires depth and segmentation"
                )
            expected = tuple(hf["frames"].shape[:4])
            if hf["depth"].shape != expected or hf["segmentation"].shape != expected:
                raise RuntimeError(
                    f"[{h5_path.stem}] depth/segmentation must match frames [T,C,H,W]"
                )
            quotas = np.full(source_count, track_cap // source_count)
            quotas[: track_cap % source_count] += 1
            for source_slot, quota in zip(proposed_sources, quotas, strict=True):
                time_id = int(slot_time_id[source_slot])
                frame_index = int(frame_indices[time_id])
                camera_index = int(slot_physical_camera_id[source_slot])
                depth = np.asarray(
                    hf["depth"][frame_index, camera_index], dtype=np.float32
                )
                segmentation = np.asarray(
                    hf["segmentation"][frame_index, camera_index]
                )
                intrinsic = np.asarray(intrinsics[source_slot], dtype=np.float32)
                if self.depth_is_ray_distance:
                    depth = self._ray_distance_to_z_depth(depth, intrinsic)
                pixels = (
                    (segmentation == 0)
                    & np.isfinite(depth)
                    & (depth > 1.0e-3)
                    & (depth < 100.0)
                )
                ys, xs = np.nonzero(pixels)
                if not len(xs):
                    continue
                z = depth[ys, xs]
                u = xs.astype(np.float32) + 0.5
                v = ys.astype(np.float32) + 0.5
                points_camera = np.stack(
                    (
                        (u - float(intrinsic[0, 2])) * z / float(intrinsic[0, 0]),
                        (v - float(intrinsic[1, 2])) * z / float(intrinsic[1, 1]),
                        z,
                    ),
                    axis=1,
                )
                source_distances = np.linalg.norm(points_camera, axis=1)
                w2c = np.asarray(extrinsics[source_slot], dtype=np.float32)
                c2w = np.linalg.inv(w2c).astype(np.float32, copy=False)
                world = points_camera @ c2w[:3, :3].T + c2w[:3, 3]
                geometrically_valid = (
                    np.isfinite(world).all(axis=1)
                    & np.isfinite(source_distances)
                )
                floor = (
                    np.abs(world[:, 2] - self.ground_plane_z)
                    <= self.ground_plane_tolerance_m
                )
                cohort = geometrically_valid & (floor if select_ground else ~floor)
                if source_distance_max_m is not None:
                    cohort &= source_distances <= source_distance_max_m
                world = world[cohort]
                source_distances = source_distances[cohort]
                if not len(world):
                    continue
                take = min(int(quota), len(world))
                indices = np.asarray(
                    rng.sample(range(len(world)), k=take), dtype=np.int64
                )
                candidate_world.append(world[indices])
                candidate_source_distances.append(source_distances[indices])
                candidate_source.extend([source_slot] * take)

        if not candidate_world:
            empty_world = np.empty((num_slots, 0, 3), dtype=np.float32)
            return {
                "world": empty_world,
                "uv": np.empty((num_slots, 0, 2), dtype=np.float32),
                "valid": np.empty((num_slots, 0), dtype=bool),
                "visible": np.empty((num_slots, 0), dtype=bool),
                "source_slots": np.empty(0, dtype=np.int64),
                "source_distances_m": np.empty(0, dtype=np.float32),
            }

        points_world = np.ascontiguousarray(
            np.concatenate(candidate_world, axis=0), dtype=np.float32
        )
        point_sources = np.asarray(candidate_source, dtype=np.int64)
        point_source_distances = np.ascontiguousarray(
            np.concatenate(candidate_source_distances), dtype=np.float32
        )
        count = len(points_world)
        world = np.broadcast_to(points_world[None], (num_slots, count, 3)).copy()
        uv = np.full((num_slots, count, 2), np.nan, dtype=np.float32)
        valid = np.zeros((num_slots, count), dtype=bool)
        for slot, (intrinsic, w2c, is_valid) in enumerate(
            zip(intrinsics, extrinsics, valid_slot, strict=True)
        ):
            if not is_valid:
                continue
            intrinsic = np.asarray(intrinsic, dtype=np.float32)
            w2c = np.asarray(w2c, dtype=np.float32)
            camera = points_world @ w2c[:3, :3].T + w2c[:3, 3]
            defined = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1.0e-4)
            projected = np.empty((count, 2), dtype=np.float32)
            projected[:, 0] = (
                float(intrinsic[0, 0]) * camera[:, 0] / camera[:, 2]
                + float(intrinsic[0, 2])
            )
            projected[:, 1] = (
                float(intrinsic[1, 1]) * camera[:, 1] / camera[:, 2]
                + float(intrinsic[1, 2])
            )
            defined &= np.isfinite(projected).all(axis=1)
            uv[slot] = projected
            valid[slot] = defined

        visible = np.zeros_like(valid)
        visible[point_sources, np.arange(count)] = True
        if not valid[point_sources, np.arange(count)].all():
            raise RuntimeError("depth-derived ground source projection is invalid")
        return {
            "world": np.ascontiguousarray(world),
            "uv": np.ascontiguousarray(uv),
            "valid": np.ascontiguousarray(valid),
            "visible": np.ascontiguousarray(visible),
            "source_slots": np.unique(point_sources),
            "source_distances_m": point_source_distances,
        }

    def _depths_to_world_points(
        self,
        depths: list[np.ndarray],
        intrinsics: list[np.ndarray],
        extrinsics: list[np.ndarray],
        valid_slot: list[bool],
        rng: random.Random,
    ) -> np.ndarray | None:
        point_sets: list[np.ndarray] = []
        stride = self.direct_point_stride
        for depth, intrinsic, w2c, is_valid in zip(
            depths, intrinsics, extrinsics, valid_slot
        ):
            if not is_valid:
                continue
            height, width = depth.shape
            ys, xs = np.mgrid[0:height:stride, 0:width:stride]
            z = depth[ys, xs].astype(np.float32, copy=False)
            valid = np.isfinite(z) & (z > 1e-3) & (z < 100.0)
            if not valid.any():
                continue
            xs_valid = xs[valid].astype(np.float32, copy=False)
            ys_valid = ys[valid].astype(np.float32, copy=False)
            xs_valid += 0.5
            ys_valid += 0.5
            z_valid = z[valid]
            fx = float(intrinsic[0, 0])
            fy = float(intrinsic[1, 1])
            if abs(fx) <= 1e-8 or abs(fy) <= 1e-8:
                continue
            points_camera = np.stack(
                [
                    (xs_valid - float(intrinsic[0, 2])) * z_valid / fx,
                    (ys_valid - float(intrinsic[1, 2])) * z_valid / fy,
                    z_valid,
                    np.ones_like(z_valid),
                ],
                axis=1,
            )
            c2w = np.linalg.inv(np.asarray(w2c, dtype=np.float32)).astype(
                np.float32, copy=False
            )
            points_world = points_camera @ c2w.T
            finite = np.isfinite(points_world[:, :3]).all(axis=1)
            if finite.any():
                point_sets.append(points_world[finite, :3].astype(np.float32, copy=False))
        if not point_sets:
            return None
        points = np.concatenate(point_sets, axis=0)
        if self.direct_point_cap > 0 and len(points) > self.direct_point_cap:
            indices = rng.sample(range(len(points)), k=self.direct_point_cap)
            points = points[np.asarray(sorted(indices), dtype=np.int64)]
        return np.ascontiguousarray(points, dtype=np.float32)

    @staticmethod
    def _ray_distance_to_z_depth(
        depth: np.ndarray,
        intrinsic: np.ndarray,
    ) -> np.ndarray:
        height, width = depth.shape
        ys, xs = np.mgrid[0:height, 0:width]
        ray_x = (xs.astype(np.float32) + 0.5 - float(intrinsic[0, 2])) / float(intrinsic[0, 0])
        ray_y = (ys.astype(np.float32) + 0.5 - float(intrinsic[1, 2])) / float(intrinsic[1, 1])
        ray_norm = np.sqrt(ray_x * ray_x + ray_y * ray_y + 1.0)
        return np.ascontiguousarray(depth / ray_norm, dtype=np.float32)

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        import h5py

        record = self._get_record(sequence_name)
        messages: list[str] = []
        ok = True
        with h5py.File(record.h5_path, "r") as hf:
            frames_shape = tuple(hf["frames"].shape)
            intr_shape = tuple(hf["intrinsics"].shape)
            ext_shape = tuple(hf["extrinsics"].shape)
            ok = ok and frames_shape == (
                record.num_frames,
                record.num_cameras,
                record.height,
                record.width,
                3,
            )
            ok = ok and intr_shape == (record.num_cameras, 3, 3)
            ok = ok and ext_shape == (record.num_cameras, 4, 4)
            c2w = np.asarray(hf["extrinsics"][0], dtype=np.float32)
            w2c = np.linalg.inv(c2w).astype(np.float32)
            eye_err = float(np.abs(w2c @ c2w - np.eye(4, dtype=np.float32)).max())
            messages.append(f"c2w_to_w2c_identity_max_err={eye_err:.3e}")

        return {
            "dataset_name": self.dataset_name,
            "sequence_name": sequence_name,
            "ok": bool(ok),
            "messages": messages,
        }

    def _split_records(
        self,
        records: list[_KubricMultiCamRecord],
    ) -> list[_KubricMultiCamRecord]:
        if self.split not in {"train", "val"} or self.val_ratio <= 0.0:
            return records

        if self.split_strategy == "tail":
            n_val = max(1, int(round(len(records) * self.val_ratio)))
            return records[-n_val:] if self.split == "val" else records[:-n_val]

        by_variant: dict[str, list[_KubricMultiCamRecord]] = {}
        for record in records:
            key = record.variant or "unknown"
            by_variant.setdefault(key, []).append(record)
        train_records: list[_KubricMultiCamRecord] = []
        val_records: list[_KubricMultiCamRecord] = []
        for key in sorted(by_variant):
            group = by_variant[key]
            n_val = max(1, int(round(len(group) * self.val_ratio)))
            n_val = min(n_val, len(group) - 1) if len(group) > 1 else 1
            train_records.extend(group[:-n_val])
            val_records.extend(group[-n_val:])
        return val_records if self.split == "val" else train_records

    def _build_index(self) -> list[_KubricMultiCamRecord]:
        h5_paths = sorted(self.root.rglob("*.h5"))
        if self.index_workers > 1 and len(h5_paths) > 1:
            indexed: list[_KubricMultiCamRecord | None] = [None] * len(h5_paths)
            worker_count = min(self.index_workers, len(h5_paths))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(self._probe_h5, path): index
                    for index, path in enumerate(h5_paths)
                }
                for future in as_completed(futures):
                    indexed[futures[future]] = future.result()
            records = [record for record in indexed if record is not None]
        else:
            records = [
                record
                for path in h5_paths
                if (record := self._probe_h5(path)) is not None
            ]
        if not records:
            raise RuntimeError(f"No valid Kubric multi-camera H5 scenes found under: {self.root}")
        return records

    def _has_valid_stamp(self, h5_path: Path) -> bool:
        if not self.require_validation_stamp:
            return True
        assert self.validation_stamp_dir is not None
        stamp_name = h5_path.stem
        if stamp_name.startswith("scene_"):
            stamp_name = stamp_name[len("scene_") :]
        stamp_path = self.validation_stamp_dir / f"{stamp_name}.json"
        try:
            stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
            stat = h5_path.stat()
            return (
                int(stamp.get("validation_revision", 0))
                >= self.min_validation_revision
                and int(stamp.get("size", -1)) == int(stat.st_size)
                and int(stamp.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def _probe_h5(self, h5_path: Path) -> _KubricMultiCamRecord | None:
        import h5py

        if not self._has_valid_stamp(h5_path):
            return None
        try:
            with h5py.File(h5_path, "r") as hf:
                required = {"frames", "intrinsics", "extrinsics", "camera_ids"}
                if not required.issubset(set(hf.keys())):
                    return None
                raw_schema = hf.attrs.get("schema_version", "")
                schema_version = (
                    raw_schema.decode("utf-8")
                    if isinstance(raw_schema, bytes)
                    else str(raw_schema)
                )
                if (
                    self.required_schema_version is not None
                    and schema_version != self.required_schema_version
                ):
                    return None
                raw_variant = hf.attrs.get("variant", "")
                variant = (
                    raw_variant.decode("utf-8")
                    if isinstance(raw_variant, bytes)
                    else str(raw_variant)
                )
                raw_fps = hf.attrs.get("fps")
                fps: float | None = None
                if raw_fps is not None:
                    try:
                        fps = float(raw_fps)
                    except (TypeError, ValueError, OverflowError):
                        return None
                    if not np.isfinite(fps) or fps <= 0.0:
                        return None
                frames = hf["frames"]
                if frames.ndim != 5 or frames.shape[-1] != 3:
                    return None
                num_frames, num_cameras, height, width, _ = frames.shape
                intrinsics = hf["intrinsics"]
                extrinsics = hf["extrinsics"]
                if intrinsics.shape != (num_cameras, 3, 3):
                    return None
                if extrinsics.shape != (num_cameras, 4, 4):
                    return None
                camera_ids = self._decode_camera_ids(hf["camera_ids"][()])
                if len(camera_ids) != num_cameras:
                    return None
                has_depth = "depth" in hf and hf["depth"].shape == (num_frames, num_cameras, height, width)
                if self.require_depth and not has_depth:
                    return None
                has_segmentation = (
                    "segmentation" in hf
                    and hf["segmentation"].shape
                    == (num_frames, num_cameras, height, width)
                )
                if self.ground_static_tracks and not has_segmentation:
                    return None
                has_tracks = False
                num_tracks = 0
                track_keys = {"tracks_3d", "tracks_2d", "visibility"}
                if track_keys.issubset(set(hf.keys())):
                    num_tracks = int(hf["tracks_3d"].shape[0])
                    has_tracks = (
                        hf["tracks_3d"].shape == (num_tracks, num_frames, 3)
                        and hf["tracks_2d"].shape
                        == (num_tracks, num_frames, num_cameras, 2)
                        and hf["visibility"].shape
                        == (num_tracks, num_frames, num_cameras)
                    )
                    if "track_depth" in hf:
                        has_tracks = has_tracks and hf["track_depth"].shape == (
                            num_tracks,
                            num_frames,
                            num_cameras,
                        )
                if self.require_native_tracks and (
                    not has_tracks or "track_depth" not in hf
                ):
                    return None
                if self.ground_static_tracks and not has_tracks:
                    return None
                has_dynamic_labels = False
                if has_tracks and {
                    "track_object_ids", "object_is_dynamic"
                }.issubset(set(hf.keys())):
                    track_object_ids = hf["track_object_ids"]
                    object_is_dynamic = hf["object_is_dynamic"]
                    has_dynamic_labels = (
                        track_object_ids.shape == (num_tracks,)
                        and object_is_dynamic.ndim == 1
                        and object_is_dynamic.shape[0] > 0
                    )
        except Exception:
            return None

        return _KubricMultiCamRecord(
            sequence_name=h5_path.stem,
            h5_path=h5_path,
            num_frames=int(num_frames),
            num_cameras=int(num_cameras),
            height=int(height),
            width=int(width),
            camera_ids=camera_ids,
            fps=fps,
            has_depth=has_depth,
            has_segmentation=has_segmentation,
            has_tracks=has_tracks,
            num_tracks=num_tracks if has_tracks else 0,
            schema_version=schema_version,
            variant=variant,
            has_dynamic_labels=has_dynamic_labels,
        )

    @staticmethod
    def _decode_camera_ids(raw: Any) -> list[str]:
        out: list[str] = []
        for item in np.asarray(raw).tolist():
            if isinstance(item, bytes):
                out.append(item.decode("utf-8"))
            else:
                out.append(str(item))
        return out

    def _get_record(self, sequence_name: str) -> _KubricMultiCamRecord:
        try:
            return self._name_to_record[sequence_name]
        except KeyError as exc:
            raise KeyError(f"Unknown Kubric multi-camera sequence: {sequence_name}") from exc

    def _choose_num_times(
        self,
        rng: random.Random,
        clip_len: int,
        num_cameras: int,
        *,
        require_multiple_times: bool = False,
    ) -> int:
        choices = valid_time_choices(
            num_slots=clip_len,
            num_cameras=num_cameras,
            min_times=self.min_times,
            max_times=self.max_times,
            tubelet_size=self.encoder_tubelet_size,
            even_multi_time=False,
        )
        if require_multiple_times:
            choices = [value for value in choices if value >= 2]
        if not choices:
            dynamic_context = " for dynamic trajectories" if require_multiple_times else ""
            raise ValueError(
                f"No tubelet-safe time count fits clip_len={clip_len}, "
                f"num_cameras={num_cameras}, min_times={self.min_times}, "
                f"max_times={self.max_times}{dynamic_context}"
            )
        if self.random_num_times:
            multi_time_choices = [value for value in choices if value >= 2]
            if 1 in choices and (
                not multi_time_choices
                or rng.random() < self.single_time_probability
            ):
                return 1
            return rng.choice(multi_time_choices or choices)
        return choices[-1]

    def _sample_time_indices(
        self,
        frame_indices: list[int],
        num_times: int,
        num_frames: int,
        *,
        slots_per_time: int,
    ) -> list[int]:
        if self.sampler_units == "slot":
            denom = max(1, int(slots_per_time))
            clipped = [
                min(max(int(i) // denom, 0), max(num_frames - 1, 0))
                for i in frame_indices
            ]
        else:
            clipped = [min(max(int(i), 0), max(num_frames - 1, 0)) for i in frame_indices]

        if num_times <= 1:
            return [clipped[0]]
        unique = list(dict.fromkeys(clipped))
        if len(unique) < num_times and num_times <= num_frames:
            low = min(unique)
            high = max(unique)
            if high - low + 1 < num_times:
                midpoint = (low + high) // 2
                low = max(0, midpoint - (num_times - 1) // 2)
                high = min(num_frames - 1, low + num_times - 1)
                low = max(0, high - num_times + 1)
            return [
                int(round(value))
                for value in np.linspace(low, high, num_times)
            ]
        if len(unique) <= num_times:
            return unique + [unique[-1]] * (num_times - len(unique))
        positions = np.linspace(0, len(unique) - 1, num_times)
        return [unique[int(round(pos))] for pos in positions]

    @staticmethod
    def _inject_unique_canonical_time(
        time_indices: list[int],
        *,
        anchor_time: int,
        num_frames: int,
    ) -> list[int]:
        """Return the same-size timeline containing the fixed time once.

        Multi-camera sampler indices may collapse to duplicate source times.
        Canonical Stage-1 needs one unambiguous physical camera/time view, so
        duplicates are filled with other real sequence times.  Missing source
        times are never synthesized by repeating the anchor.
        """

        if not time_indices:
            raise ValueError("canonical timeline cannot be empty")
        count = len(time_indices)
        if count > int(num_frames):
            raise ValueError(
                "canonical timeline cannot contain unique source times: "
                f"requested={count}, available={num_frames}"
            )
        anchor = int(anchor_time)
        if not 0 <= anchor < int(num_frames):
            raise ValueError("canonical source time lies outside the sequence")
        values = [int(value) for value in time_indices]
        if any(value < 0 or value >= int(num_frames) for value in values):
            raise ValueError("sampled source time lies outside the sequence")
        ordered = all(left <= right for left, right in zip(values, values[1:]))

        # Preserve sampled evidence in first-occurrence order, then add the
        # immutable anchor and fill any holes using farthest-point temporal
        # coverage over the real sequence timeline.
        unique = list(dict.fromkeys(values))
        if anchor not in unique:
            if len(unique) == count:
                remove_index = min(
                    range(len(unique)),
                    key=lambda index: (abs(unique[index] - anchor), index),
                )
                unique.pop(remove_index)
            unique.append(anchor)
        while len(unique) < count:
            available = [
                value for value in range(int(num_frames)) if value not in unique
            ]
            if not available:
                raise RuntimeError("failed to construct a unique canonical timeline")
            value = max(
                available,
                key=lambda candidate: (
                    min(abs(candidate - selected) for selected in unique),
                    -candidate,
                ),
            )
            unique.append(value)
        if ordered:
            unique.sort()
        if len(unique) != count or unique.count(anchor) != 1:
            raise RuntimeError("failed to retain exactly one canonical source time")
        return unique

    @staticmethod
    def _sample_seed(sequence_name: str, frame_indices: list[int]) -> int:
        payload = f"{sequence_name}:{','.join(str(int(i)) for i in frame_indices)}".encode()
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
