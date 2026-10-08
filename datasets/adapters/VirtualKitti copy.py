from __future__ import annotations

import os
import glob
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
        verbose: bool = True,
        cache_dir: Optional[str] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.camera = camera
        self.precompute_root = Path(precompute_root) if precompute_root else self.root
        self.verbose = verbose

        # VKITTI 2 default intrinsics
        self.default_K = np.array([
            [725.0, 0.0,   620.5],
            [0.0,   725.0, 187.0],
            [0.0,   0.0,   1.0]
        ], dtype=np.float32)

        self.sequences: list[str] = []
        self.seq_to_dir: dict[str, Path] = {}
        if cache_dir is not None:
            from datasets.index_cache import load_or_build
            _cache_path = Path(cache_dir) / f"{self.dataset_name}_{split}.pkl"
            def _build_and_return():
                self._build_index()
                return (list(self.sequences), dict(self.seq_to_dir))
            self.sequences, self.seq_to_dir = load_or_build(_build_and_return, _cache_path)
        else:
            self._build_index()

        if len(self.sequences) == 0:
            raise RuntimeError(f"No valid VKITTI2 sequences found under {self.root}")

        if self.verbose:
            print(f"[VKITTI2Adapter] split={self.split}, camera={self.camera}, "
                  f"num_sequences={len(self.sequences)}, "
                  f"precompute={'yes' if self.precompute_root else 'no'}")

    def __len__(self) -> int:
        return len(self.sequences)

    def list_sequences(self) -> list[str]:
        return self.sequences.copy()

    def get_sequence_name(self, index: int) -> str:
        return self.sequences[index]

    def get_num_frames(self, sequence_name: str) -> int:
        """Skip per-sequence glob; assume sequences have sufficient frames."""
        return 10_000

    def _build_index(self):
        scene_dirs = [f for f in self.root.iterdir() if f.is_dir() and "Scene" in f.name]
        for scene_dir in sorted(scene_dirs):
            scene_name = scene_dir.name
            variation_dirs = [f for f in scene_dir.iterdir() if f.is_dir()]
            for var_dir in sorted(variation_dirs):
                var_name = var_dir.name
                seq_name = f"{scene_name}_{var_name}"
                self.sequences.append(seq_name)
                self.seq_to_dir[seq_name] = var_dir

    def _get_actual_dir(self, base_dir: Path, sub_folder: str) -> Path:
        target_dir = base_dir / sub_folder / self.camera
        if not target_dir.exists():
            target_dir = base_dir / sub_folder
        return target_dir

    def _load_precomputed(self, sequence_name: str, frame_indices: list[int]) -> Optional[dict]:
        """Load precomputed normals/tracks for frame_indices. Prefers .h5 over .npz."""
        if self.precompute_root is None:
            return None
        from datasets.adapters.base import load_precomputed_fast
        cache_path = self.precompute_root / sequence_name / "precomputed.npz"
        h5_path = cache_path.with_suffix('.h5')
        if not cache_path.exists() and not h5_path.exists():
            return None
        try:
            return load_precomputed_fast(cache_path, frame_indices)
        except Exception as e:
            if self.verbose:
                print(f"[VKITTI2Adapter] Warning: failed to load cache for {sequence_name}: {e}")
            return None

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        seq_dir = self.seq_to_dir[sequence_name]
        frames_dir = seq_dir / "frames"
        img_dir = self._get_actual_dir(frames_dir, "rgb")

        num_frames = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0

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
            "num_frames": num_frames,
            "has_depth": (self._get_actual_dir(frames_dir, "depth")).exists(),
            "has_flow": (self._get_actual_dir(frames_dir, "forwardFlow")).exists(),
            "has_normals": has_precomputed,
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

        img_dir = self._get_actual_dir(frames_dir, "rgb")
        depth_dir = self._get_actual_dir(frames_dir, "depth")
        flow_dir = self._get_actual_dir(frames_dir, "forwardFlow")

        extrinsic_file = seq_dir / "extrinsic.txt"
        intrinsic_file = seq_dir / "intrinsic.txt"

        all_img_paths = sorted(img_dir.glob("*.jpg"))
        all_depth_paths = sorted(depth_dir.glob("*.png")) if depth_dir.exists() else []
        all_flow_paths = sorted(flow_dir.glob("*.png")) if flow_dir.exists() else []

        # Parse extrinsics
        # VKITTI2 extrinsic.txt stores camera-to-world (c2w) transforms.
        # We invert analytically to get world-to-camera (w2c).
        pose_dict = {}
        if extrinsic_file.exists():
            with open(extrinsic_file, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    parts = line.strip().split()
                    if len(parts) > 2 and parts[1] == self.camera[-1]:
                        frame_id = int(parts[0])
                        c2w_mat = np.array(list(map(float, parts[2:14]))).reshape(3, 4)
                        c2w = np.eye(4, dtype=np.float32)
                        c2w[:3, :4] = c2w_mat

                        # VKITTI2 extrinsic.txt stores camera-to-world (c2w) transforms.
                        # Invert analytically to get world-to-camera (w2c).
                        # No coordinate system conversion needed: the precomputed tracks
                        # were generated using the same raw c2w convention.
                        R = c2w[:3, :3]
                        t = c2w[:3, 3]
                        w2c = np.eye(4, dtype=np.float32)
                        w2c[:3, :3] = R.T
                        w2c[:3, 3] = -R.T @ t
                        pose_dict[frame_id] = w2c

        # Parse intrinsics
        K_dict = {}
        if intrinsic_file.exists():
            with open(intrinsic_file, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    parts = line.strip().split()
                    if len(parts) >= 6 and parts[1] == self.camera[-1]:
                        frame_id = int(parts[0])
                        fx, fy, cx, cy = map(float, parts[2:6])
                        K_mat = np.array([
                            [fx,  0.0, cx],
                            [0.0, fy,  cy],
                            [0.0, 0.0, 1.0]
                        ], dtype=np.float32)
                        K_dict[frame_id] = K_mat

        images, depths, extrinsics, intrinsics, flows = [], [], [], [], []
        frame_paths = []

        for idx in frame_indices:
            if idx < len(all_img_paths):
                img_path = all_img_paths[idx]
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img.astype(np.float32) / 255.0)
                frame_paths.append(str(img_path))

            if idx < len(all_depth_paths):
                depth = cv2.imread(str(all_depth_paths[idx]), cv2.IMREAD_UNCHANGED)
                if depth is not None:
                    depth = depth.astype(np.float32) / 100.0
                depths.append(depth)

            extrinsics.append(pose_dict.get(idx, np.eye(4, dtype=np.float32)))
            intrinsics.append(K_dict.get(idx, self.default_K.copy()))

            if idx < len(all_flow_paths):
                flow_data = self._read_vkitti_flow(all_flow_paths[idx])
                flows.append(flow_data)
            else:
                flows.append(None)

        # Load precomputed normals/tracks if available
        normals_out, trajs_2d_out, trajs_3d_out, valids_out, visibs_out = None, None, None, None, None
        has_precomputed = False
        cache = self._load_precomputed(sequence_name, frame_indices)
        if cache is not None:
            try:
                normals_out  = [n.astype(np.float32) for n in cache["normals"]]
                trajs_2d_out = cache["trajs_2d"]
                trajs_3d_out = cache["trajs_3d_world"]
                valids_out   = cache["valids"]
                visibs_out   = cache["visibs"]
                has_precomputed = True
            except Exception as e:
                if self.verbose:
                    print(f"[VKITTI2Adapter] Warning: precomputed indexing failed for {sequence_name}: {e}")

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
            metadata={
                "dataset_name": self.dataset_name,
                "split": self.split,
                "has_depth": len(depths) > 0,
                "has_flow": any(f is not None for f in flows),
                "has_normals": has_precomputed,
                "has_tracks": has_precomputed,
                "has_visibility": has_precomputed,
                "pose_convention": "world_to_camera",
                "intrinsics_convention": "pinhole",
                "depth_unit": "meters",
            }
        )

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        info = self.get_sequence_info(sequence_name)
        info["ok"] = info["num_frames"] > 0
        return info
