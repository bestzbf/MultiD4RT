from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .base import BaseAdapter, UnifiedClip


class VKITTI2Adapter(BaseAdapter):
    dataset_name: str = "vkitti2"

    def __init__(
        self,
        root: str,
        split: str = "train",
        camera: str = "Camera_0",
        precompute_root: Optional[str] = None,
        load_precomputed: bool = True,
        load_normals: bool = True,
        load_flow: bool = True,
        verbose: bool = True,
        cache_dir: Optional[str] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.camera = camera
        self.load_precomputed = bool(load_precomputed)
        self.load_normals = bool(load_normals)
        self.load_flow = bool(load_flow)
        self.precompute_root = (
            Path(precompute_root) if precompute_root and self.load_precomputed
            else (self.root if self.load_precomputed else None)
        )
        self.verbose = verbose

        # VKITTI 2 default intrinsics.
        self.default_K = np.array([
            [725.0, 0.0,   620.5],
            [0.0,   725.0, 187.0],
            [0.0,   0.0,   1.0],
        ], dtype=np.float32)

        self.sequences: list[str] = []
        self.seq_to_dir: dict[str, Path] = {}
        self._num_frames_map: dict[str, int] = {}
        self._metadata_cache: dict[
            str,
            tuple[dict[int, np.ndarray], dict[int, np.ndarray]],
        ] = {}

        if cache_dir is not None:
            from datasets.index_cache import load_or_build

            cache_key = {
                "dataset": self.dataset_name,
                "split": self.split,
                "camera": self.camera,
                "root": str(self.root),
                "cache_schema": 2,
            }
            suffix = hashlib.sha1(
                json.dumps(cache_key, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            cache_path = (
                Path(cache_dir) / f"{self.dataset_name}_{self.split}_{suffix}.pkl"
            )
            index: list[tuple[str, int]] = load_or_build(self._build_index, cache_path)
        else:
            index = self._build_index()

        for seq_name, num_frames in index:
            scene, variation = self._parse_sequence_name(seq_name)
            self.sequences.append(seq_name)
            self.seq_to_dir[seq_name] = self.root / scene / variation
            self._num_frames_map[seq_name] = int(num_frames)

        if len(self.sequences) == 0:
            raise RuntimeError(f"No valid VKITTI2 sequences found under {self.root}")

        if self.verbose:
            print(
                f"[VKITTI2Adapter] split={self.split}, camera={self.camera}, "
                f"num_sequences={len(self.sequences)}, "
                f"precompute={'yes' if self.precompute_root else 'no'} "
                f"load_normals={self.load_normals} load_flow={self.load_flow}"
            )

    def __len__(self) -> int:
        return len(self.sequences)

    def list_sequences(self) -> list[str]:
        return self.sequences.copy()

    def get_sequence_name(self, index: int) -> str:
        return self.sequences[index]

    def get_num_frames(self, sequence_name: str) -> int:
        return self._num_frames_map.get(sequence_name, 0)

    def _build_index(self) -> list[tuple[str, int]]:
        index: list[tuple[str, int]] = []
        scene_dirs = [
            path
            for path in self.root.iterdir()
            if path.is_dir() and re.match(r"^Scene\d+$", path.name)
        ]
        for scene_dir in sorted(scene_dirs):
            scene_name = scene_dir.name
            variation_dirs = [path for path in scene_dir.iterdir() if path.is_dir()]
            for variation_dir in sorted(variation_dirs):
                variation_name = variation_dir.name
                num_frames = self._count_rgb_frames(variation_dir)
                if num_frames <= 0:
                    continue
                index.append((f"{scene_name}_{variation_name}", num_frames))
        return index

    @staticmethod
    def _parse_sequence_name(sequence_name: str) -> tuple[str, str]:
        match = re.match(r"^(Scene\d+)_(.+)$", sequence_name)
        if match is None:
            raise ValueError(f"Invalid VKITTI2 sequence name: {sequence_name!r}")
        return match.group(1), match.group(2)

    def _count_rgb_frames(self, sequence_dir: Path) -> int:
        img_dir = self._get_actual_dir(sequence_dir / "frames", "rgb")
        return len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0

    def _get_actual_dir(self, base_dir: Path, sub_folder: str) -> Path:
        target_dir = base_dir / sub_folder / self.camera
        if not target_dir.exists():
            target_dir = base_dir / sub_folder
        return target_dir

    @staticmethod
    def _frame_path(directory: Path, prefix: str, index: int, suffix: str) -> Path:
        return directory / f"{prefix}_{index:05d}{suffix}"

    def _load_precomputed(
        self,
        sequence_name: str,
        frame_indices: list[int],
    ) -> Optional[dict]:
        """Load precomputed normals/tracks for frame_indices. Prefers .h5."""
        if not self.load_precomputed or self.precompute_root is None:
            return None
        from datasets.adapters.base import load_precomputed_fast

        cache_path = self.precompute_root / sequence_name / "precomputed.npz"
        h5_path = cache_path.with_suffix(".h5")
        if not cache_path.exists() and not h5_path.exists():
            return None
        try:
            skip_keys = {"normals"} if not self.load_normals else set()
            return load_precomputed_fast(
                cache_path,
                frame_indices,
                skip_keys=skip_keys,
            )
        except Exception as e:
            if self.verbose:
                print(
                    f"[VKITTI2Adapter] Warning: failed to load cache "
                    f"for {sequence_name}: {e}"
                )
            return None

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        seq_dir = self.seq_to_dir[sequence_name]
        frames_dir = seq_dir / "frames"

        has_precomputed = (
            self.precompute_root is not None
            and (
                (self.precompute_root / sequence_name / "precomputed.npz").exists()
                or (self.precompute_root / sequence_name / "precomputed.h5").exists()
            )
        )

        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "sequence_name": sequence_name,
            "num_frames": self.get_num_frames(sequence_name),
            "has_depth": (self._get_actual_dir(frames_dir, "depth")).exists(),
            "has_flow": self.load_flow
            and (self._get_actual_dir(frames_dir, "forwardFlow")).exists(),
            "has_normals": has_precomputed and self.load_normals,
            "has_tracks": has_precomputed,
            "has_visibility": has_precomputed,
        }

    def _read_vkitti_flow(self, flow_path: Path) -> Optional[np.ndarray]:
        flow_img = cv2.imread(str(flow_path), cv2.IMREAD_UNCHANGED)
        if flow_img is None:
            return None

        b, g, r = cv2.split(flow_img)
        u = (r.astype(np.float32) - 32768.0) / 64.0
        v = (g.astype(np.float32) - 32768.0) / 64.0
        flow_data = np.stack([u, v], axis=-1)

        invalid_mask = b == 0
        flow_data[invalid_mask] = 0.0
        return flow_data

    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        seq_dir = self.seq_to_dir[sequence_name]
        frames_dir = seq_dir / "frames"
        num_frames = self.get_num_frames(sequence_name)
        self._check_indices(frame_indices, num_frames, sequence_name)

        img_dir = self._get_actual_dir(frames_dir, "rgb")
        depth_dir = self._get_actual_dir(frames_dir, "depth")
        flow_dir = self._get_actual_dir(frames_dir, "forwardFlow")
        pose_dict, K_dict = self._load_sequence_metadata(sequence_name)

        images, depths, extrinsics, intrinsics, flows = [], [], [], [], []
        frame_paths = []

        for idx in frame_indices:
            img_path = self._frame_path(img_dir, "rgb", idx, ".jpg")
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"Missing VKITTI2 RGB frame: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img.astype(np.float32) / 255.0)
            frame_paths.append(str(img_path))

            depth_path = self._frame_path(depth_dir, "depth", idx, ".png")
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(f"Missing VKITTI2 depth frame: {depth_path}")
            # VKITTI2: sky/invalid pixels are stored as 65535 (max uint16).
            sky_mask = depth == 65535
            depth = depth.astype(np.float32) / 100.0
            depth[sky_mask] = 0.0
            depths.append(depth)

            extrinsics.append(pose_dict.get(idx, np.eye(4, dtype=np.float32)))
            intrinsics.append(K_dict.get(idx, self.default_K.copy()))

            if self.load_flow:
                flow_path = self._frame_path(flow_dir, "flow", idx, ".png")
                flows.append(
                    self._read_vkitti_flow(flow_path) if flow_path.exists() else None
                )

        normals_out, trajs_2d_out, trajs_3d_out, valids_out, visibs_out = (
            None,
            None,
            None,
            None,
            None,
        )
        track_types_out, track_object_ids_out, track_ref_frames_out = None, None, None
        has_precomputed = False
        track_source = "none"
        cache = self._load_precomputed(sequence_name, frame_indices)
        if cache is not None:
            try:
                if self.load_normals and "normals" in cache:
                    normals_out = [n.astype(np.float32) for n in cache["normals"]]
                trajs_2d_out = cache["trajs_2d"]
                trajs_3d_out = cache["trajs_3d_world"]
                valids_out = cache["valids"].copy()
                visibs_out = cache["visibs"].copy()
                track_types_out = cache.get("track_types")
                track_object_ids_out = cache.get("track_object_ids")
                track_ref_frames_out = cache.get("track_ref_frames")
                has_precomputed = True
                track_source = "precomputed"

                # Invalidate tracks whose 2D position falls on sky/invalid pixels.
                if len(depths) > 0:
                    for ti, depth_t in enumerate(depths):
                        if depth_t is None:
                            continue
                        H_d, W_d = depth_t.shape[:2]
                        uv = trajs_2d_out[ti]
                        finite = np.isfinite(uv).all(axis=-1)
                        in_bounds = (
                            finite
                            & (uv[:, 0] >= 0.0)
                            & (uv[:, 0] < W_d)
                            & (uv[:, 1] >= 0.0)
                            & (uv[:, 1] < H_d)
                        )
                        if valids_out is not None:
                            in_bounds &= valids_out[ti].astype(bool)
                        valid_idx = np.flatnonzero(in_bounds)
                        if len(valid_idx) == 0:
                            continue
                        xs = np.clip(np.round(uv[valid_idx, 0]).astype(int), 0, W_d - 1)
                        ys = np.clip(np.round(uv[valid_idx, 1]).astype(int), 0, H_d - 1)
                        sky = depth_t[ys, xs] == 0.0
                        if np.any(sky):
                            bad_idx = valid_idx[sky]
                            valids_out[ti][bad_idx] = False
                            if visibs_out is not None:
                                visibs_out[ti][bad_idx] = False

            except Exception as e:
                if self.verbose:
                    print(
                        f"[VKITTI2Adapter] Warning: precomputed indexing failed "
                        f"for {sequence_name}: {e}"
                    )

        metadata = {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "has_depth": len(depths) > 0,
            "has_flow": any(f is not None for f in flows),
            "has_normals": normals_out is not None,
            "has_tracks": has_precomputed,
            "has_visibility": has_precomputed,
            "pose_convention": "world_to_camera",
            "intrinsics_convention": "pinhole",
            "extrinsics_convention": "w2c",
            "vkitti_extrinsics_source": "extrinsic.txt_raw_w2c",
            "track_source": track_source,
            "depth_unit": "meters",
        }
        if track_types_out is not None:
            metadata["track_types"] = np.asarray(track_types_out, dtype=np.int8)
        if track_object_ids_out is not None:
            metadata["track_object_ids"] = np.asarray(track_object_ids_out, dtype=np.int32)
        if track_ref_frames_out is not None:
            metadata["track_ref_frames"] = np.asarray(track_ref_frames_out, dtype=np.int32)

        return UnifiedClip(
            dataset_name=self.dataset_name,
            sequence_name=sequence_name,
            frame_paths=frame_paths,
            images=images,
            depths=depths if len(depths) > 0 else None,
            normals=normals_out,
            trajs_2d=trajs_2d_out,
            trajs_3d_world=trajs_3d_out,
            valids=valids_out,
            visibs=visibs_out,
            intrinsics=np.stack(intrinsics, axis=0),
            extrinsics=np.stack(extrinsics, axis=0),
            flows=flows if any(f is not None for f in flows) else None,
            metadata=metadata,
        )

    def _load_sequence_metadata(
        self,
        sequence_name: str,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        cached = self._metadata_cache.get(sequence_name)
        if cached is not None:
            return cached

        seq_dir = self.seq_to_dir[sequence_name]
        extrinsic_file = seq_dir / "extrinsic.txt"
        intrinsic_file = seq_dir / "intrinsic.txt"

        pose_dict: dict[int, np.ndarray] = {}
        # VKITTI2 extrinsic.txt stores world-to-camera (w2c) transforms in the
        # same camera convention as its RGB/depth/flow files. Do not invert.
        if extrinsic_file.exists():
            with open(extrinsic_file, "r") as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split()
                    if len(parts) > 2 and parts[1] == self.camera[-1]:
                        frame_id = int(parts[0])
                        w2c = np.eye(4, dtype=np.float32)
                        w2c[:3, :4] = np.array(
                            list(map(float, parts[2:14])),
                            dtype=np.float32,
                        ).reshape(3, 4)
                        pose_dict[frame_id] = w2c

        K_dict: dict[int, np.ndarray] = {}
        if intrinsic_file.exists():
            with open(intrinsic_file, "r") as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split()
                    if len(parts) >= 6 and parts[1] == self.camera[-1]:
                        frame_id = int(parts[0])
                        fx, fy, cx, cy = map(float, parts[2:6])
                        K_dict[frame_id] = np.array([
                            [fx, 0.0, cx],
                            [0.0, fy, cy],
                            [0.0, 0.0, 1.0],
                        ], dtype=np.float32)

        self._metadata_cache[sequence_name] = (pose_dict, K_dict)
        return pose_dict, K_dict

    @staticmethod
    def _check_indices(
        frame_indices: list[int],
        num_frames: int,
        sequence_name: str,
    ) -> None:
        if not frame_indices:
            raise ValueError("frame_indices is empty")
        lo, hi = min(frame_indices), max(frame_indices)
        if lo < 0 or hi >= num_frames:
            raise IndexError(
                f"[{sequence_name}] frame_indices out of range: "
                f"min={lo}, max={hi}, num_frames={num_frames}"
            )

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        info = self.get_sequence_info(sequence_name)
        info["ok"] = info["num_frames"] > 0
        return info
