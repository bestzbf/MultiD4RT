# from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from .base import BaseAdapter, UnifiedClip


@dataclass
class _KubricSceneRecord:
    sequence_name: str
    scene_dir: Path
    ann_path: Path
    rank_path: Path
    h5_path: Path | None
    trajs_2d_path: Path | None
    frame_names: list[str] | None
    depth_names: list[str] | None
    num_frames: int
    num_tracks: int
    height: int
    width: int
    frame_file_count: int
    frame_name_width: int = 3
    frame_name_suffix: str = ".png"
    depth_name_width: int = 3
    depth_name_suffix: str = ".npy"
    has_depth_dir: bool = False

    @property
    def frame_dir(self) -> Path:
        return self.scene_dir / "frames"

    @property
    def depth_dir(self) -> Path:
        return self.scene_dir / "depths"


class KubricAdapter(BaseAdapter):
    """
    Kubric adapter for extracted scene format.

    Expected layout:
        root/
          0000/
            frames/
            0000.npy
            0000_with_rank.npz
          0001/
            frames/
            0001.npy
            0001_with_rank.npz
          ...

    Notes:
    - ann["coords"]         : [N, T, 2]
    - ann["coords_depth"]   : [N, T]
    - ann["visibility"]     : [N, T]
    - ann["depth"]          : [T, H, W, 1]
    - rank["shared_intrinsics"] : [3, 3]
    - rank["extrinsics"]        : [T, 3, 4], confirmed world-to-camera (w2c)

    Adapter output unified schema:
    - trajs_2d         : [T, N, 2]
    - trajs_3d_world   : [T, N, 3]
    - visibs           : [T, N]
    - intrinsics       : [T, 3, 3]
    - extrinsics       : [T, 4, 4]   (w2c)
    """

    dataset_name = "kubric"

    def __init__(
        self,
        root: str,
        split: str = "train",
        cache_dir: str | None = None,
        verbose: bool = True,
        index_workers: int = 8,
        **kwargs,
    ):
        """
        Args:
            root: Root directory containing Kubric scenes
            split: Split name (ignored, Kubric doesn't have splits)
            cache_dir: Optional directory for a pickled scene index cache
            verbose: Print a short summary after initialization

        Extra kwargs:
            visibility_mode: "raycasting" (default) | "depth_test" | "union"
                "raycasting" keeps Kubric's segmentation-based visibility.
                "depth_test" rebuilds visibility from a per-point z-buffer
                test (|coords_depth - frame_depth| / frame_depth < tol).
                "union" sets visibility True if either source agrees.
            visibility_depth_rel_tol: float, default 0.05.
                Relative-error threshold for the depth z-buffer test.
        """
        self.root = Path(root)
        self.split = split
        self.verbose = verbose
        self.index_workers = index_workers
        self.visibility_mode = str(kwargs.pop("visibility_mode", "union"))
        self.visibility_depth_rel_tol = float(
            kwargs.pop("visibility_depth_rel_tol", 0.05)
        )
        if self.visibility_mode not in ("raycasting", "depth_test", "union"):
            raise ValueError(
                f"[KubricAdapter] visibility_mode must be one of "
                f"'raycasting' | 'depth_test' | 'union', got {self.visibility_mode!r}"
            )
        self._h5_visibility_warned = False
        # Check cache first to skip slow root.exists() on remote storage
        _cache_hit = False
        if cache_dir is not None:
            from datasets.index_cache import load_or_build

            cache_key = {
                "dataset": self.dataset_name,
                "root": str(self.root.resolve()),
                "cache_schema": 5,
            }
            cache_suffix = hashlib.sha1(
                json.dumps(cache_key, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            cache_path = Path(cache_dir) / f"{self.dataset_name}_all_{cache_suffix}.pkl"
            if cache_path.exists():
                _cache_hit = True
        if not _cache_hit and not self.root.exists():
            raise FileNotFoundError(f"Kubric root not found: {self.root}")
        if cache_dir is not None:
            from datasets.index_cache import load_or_build

            cache_key = {
                "dataset": self.dataset_name,
                "root": str(self.root.resolve()),
                "cache_schema": 5,
            }
            cache_suffix = hashlib.sha1(
                json.dumps(cache_key, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            cache_path = Path(cache_dir) / f"{self.dataset_name}_all_{cache_suffix}.pkl"
            all_records = load_or_build(self._build_index, cache_path)
        else:
            all_records = self._build_index()
        # Deterministic train/val split: last 5% as val
        n_val = max(1, len(all_records) // 20)
        if split == 'val':
            self._records = all_records[-n_val:]
        else:
            self._records = all_records[:-n_val]
        self._name_to_record = {r.sequence_name: r for r in self._records}
        self.sequence_names = [r.sequence_name for r in self._records]
        if self.verbose:
            print(
                f"[KubricAdapter] split={self.split!r}, "
                f"num_sequences={len(self.sequence_names)} "
                f"(total_indexed={len(all_records)}) "
                f"visibility_mode={self.visibility_mode!r} "
                f"depth_tol={self.visibility_depth_rel_tol}"
            )

    def __len__(self) -> int:
        return len(self.sequence_names)

    def list_sequences(self) -> list[str]:
        return self.sequence_names

    def get_sequence_name(self, index: int) -> str:
        return self._records[index].sequence_name

    def get_num_frames(self, sequence_name: str) -> int:
        """Fast path: read from cached record, no annotation I/O."""
        return self._get_record(sequence_name).num_frames

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        record = self._get_record(sequence_name)
        if record.height <= 0 or record.width <= 0:
            height, width = self._resolve_record_image_size(record)
            record.height = height
            record.width = width

        info = {
            "dataset_name": self.dataset_name,
            "sequence_name": sequence_name,
            "path": str(record.scene_dir),
            "num_frames": record.num_frames,
            "num_tracks": record.num_tracks,
            "height": record.height,
            "width": record.width,
            "frame_file_count": record.frame_file_count,
            "has_depth": True,
            "has_normals": False,
            "has_tracks": True,
            "has_visibility": True,
            "has_trajs_3d_world": True,
            "intrinsics_shape": (3, 3),
            "extrinsics_shape": (record.num_frames, 3, 4),
            "extrinsics_convention": "w2c",
        }
        return info

    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        record = self._get_record(sequence_name)
        ann_path = record.ann_path
        rank_path = record.rank_path
        h5_path = record.h5_path
        if h5_path is not None:
            import h5py
            with h5py.File(h5_path, 'r') as hf:
                T_total = hf['trajs_2d'].shape[0]
                sorted_idx = sorted(set(frame_indices))
                idx_map = {v: pos for pos, v in enumerate(sorted_idx)}
                reorder = [idx_map[i] for i in frame_indices]
                s = np.array(sorted_idx)
                trajs_2d     = hf['trajs_2d'][s][reorder]       # [T,N,2]
                coords_depth = hf['coords_depth'][s][reorder]   # [T,N]
                visibs       = hf['visibility'][s][reorder]     # [T,N]

            # depth still comes from per-frame .npy files in depths/
            if record.depth_names is not None or getattr(record, 'has_depth_dir', False):
                depth_paths = [self._depth_path_for_index(record, i) for i in frame_indices]
                depths = [
                    np.load(path).astype(np.float32).squeeze()
                    for path in depth_paths
                ]
            else:
                depths = None

            ann = None
            seg_thw1 = None
            camera_raw = None
            metadata_raw = None
        else:
            ann = np.load(ann_path, allow_pickle=True).item()

            coords_nt2 = np.asarray(ann["coords"], dtype=np.float32)              # [N,T,2]
            coords_depth_nt = np.asarray(ann["coords_depth"], dtype=np.float32)   # [N,T]
            visibility_nt = np.asarray(ann["visibility"], dtype=bool)             # [N,T]

            trajs_2d     = np.transpose(coords_nt2[:, frame_indices, :], (1, 0, 2))
            coords_depth = np.transpose(coords_depth_nt[:, frame_indices], (1, 0))
            visibs       = np.transpose(visibility_nt[:, frame_indices], (1, 0))

            dense_depth_thw1 = np.asarray(ann["depth"], dtype=np.float32)
            dense_depth = dense_depth_thw1[frame_indices, ..., 0]
            depths = [dense_depth[i] for i in range(len(frame_indices))]

            seg_thw1 = np.asarray(ann["segmentations"]) if "segmentations" in ann else None
            camera_raw   = ann.get("camera", None)
            metadata_raw = ann.get("metadata", None)

        rank = np.load(rank_path, allow_pickle=True)

        K_shared = np.asarray(rank["shared_intrinsics"], dtype=np.float32)    # [3,3]
        extrinsics_t34 = np.asarray(rank["extrinsics"], dtype=np.float32)     # [T,3,4], w2c
        ranking_tt = np.asarray(rank["ranking"])

        if ann is not None:
            T_total = np.asarray(ann["coords"]).shape[1]
        # else T_total was already set from h5 header
        self._check_indices(frame_indices, T_total, sequence_name)

        if record.frame_file_count != T_total:
            raise ValueError(
                f"[{sequence_name}] frame file count mismatch: "
                f"frames/ has {record.frame_file_count}, annotations have {T_total}"
            )

        frame_paths = [self._frame_path_for_index(record, i) for i in frame_indices]
        selected_frame_paths = [str(path) for path in frame_paths]
        images = [self._read_image(path) for path in frame_paths]

        # trajs_2d, coords_depth, visibs already set (both paths produce [T,N,*])

        intrinsics = np.repeat(K_shared[None, :, :], len(frame_indices), axis=0)   # [T,3,3]
        extrinsics = self._to_4x4(extrinsics_t34[frame_indices])                    # [T,4,4], w2c

        trajs_3d_world = self._backproject_tracks_to_world(
            trajs_2d=trajs_2d,
            coords_depth=coords_depth,
            intrinsics=intrinsics,
            extrinsics_w2c=extrinsics,
        )  # [T,N,3]

        valids = (
            np.isfinite(trajs_2d[..., 0])
            & np.isfinite(trajs_2d[..., 1])
            & np.isfinite(coords_depth)
            & (coords_depth > 0)
        )

        # Kubric's raw `visibility` only encodes occlusion via raycasting, with
        # out-of-frame projections defaulting to True. Combine with an explicit
        # in-frame check so visibs at the adapter boundary already means
        # "occluded-free AND in-frame".
        H_img, W_img = images[0].shape[:2]
        in_bounds = (
            (trajs_2d[..., 0] >= 0) & (trajs_2d[..., 0] < W_img) &
            (trajs_2d[..., 1] >= 0) & (trajs_2d[..., 1] < H_img)
        )
        visibs = visibs.astype(bool) & in_bounds

        # Optional: rebuild visibility from a depth z-buffer test. Some Kubric
        # tracks have segmentation-based visibility=False even though the point
        # is depth-consistent and visually unoccluded (e.g. surface points that
        # land on background pixels, or sub-pixel raycasting misses). The depth
        # test ("|coords_depth - frame_depth| / frame_depth < tol") gives a
        # geometric notion of visibility that matches PointOdyssey / Dynamic
        # Replica more closely.
        if self.visibility_mode != "raycasting":
            depth_visible = self._compute_depth_visibility(
                trajs_2d=trajs_2d,
                coords_depth=coords_depth,
                depths=depths,
                metadata_raw=metadata_raw,
                ann=ann,
                frame_indices=frame_indices,
                H_img=H_img,
                W_img=W_img,
            )
            if depth_visible is not None:
                depth_visible = depth_visible & in_bounds
                if self.visibility_mode == "depth_test":
                    visibs = depth_visible
                else:  # "union"
                    visibs = visibs | depth_visible

        metadata = {
            "backend": "kubric_extracted",
            "has_depth": True,
            "has_normals": False,
            "has_tracks": True,
            "has_visibility": True,
            "has_trajs_3d_world": True,
            "normal_convention": None,
            "normal_supervision_compatible": False,
            "num_frames_total": int(T_total),
            "num_frames_clip": int(len(frame_indices)),
            "coords_depth": coords_depth,  # [T,N], useful for debug / reprojection checks
            "segmentations": seg_thw1[frame_indices, ..., 0] if seg_thw1 is not None else None,
            "ranking": ranking_tt[np.ix_(frame_indices, frame_indices)],
            "camera_raw": camera_raw,
            "metadata_raw": metadata_raw,
            "extrinsics_convention": "w2c",
        }

        return UnifiedClip(
            dataset_name=self.dataset_name,
            sequence_name=sequence_name,
            frame_paths=selected_frame_paths,
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
        )

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        record = self._get_record(sequence_name)
        ann = np.load(record.ann_path, allow_pickle=True).item()
        rank = np.load(record.rank_path, allow_pickle=True)

        msgs: list[str] = []
        ok = True

        coords = np.asarray(ann["coords"])
        coords_depth = np.asarray(ann["coords_depth"])
        visibility = np.asarray(ann["visibility"])
        depth = np.asarray(ann["depth"])
        extrinsics = np.asarray(rank["extrinsics"])
        K = np.asarray(rank["shared_intrinsics"])

        if coords.ndim != 3 or coords.shape[-1] != 2:
            ok = False
            msgs.append(f"coords shape invalid: {coords.shape}")

        if coords_depth.shape != coords.shape[:2]:
            ok = False
            msgs.append(
                f"coords_depth shape {coords_depth.shape} != coords[:2] {coords.shape[:2]}"
            )

        if visibility.shape != coords.shape[:2]:
            ok = False
            msgs.append(
                f"visibility shape {visibility.shape} != coords[:2] {coords.shape[:2]}"
            )

        if depth.ndim != 4 or depth.shape[-1] != 1:
            ok = False
            msgs.append(f"depth shape invalid: {depth.shape}")

        if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (3, 4):
            ok = False
            msgs.append(f"extrinsics shape invalid: {extrinsics.shape}")

        if K.shape != (3, 3):
            ok = False
            msgs.append(f"shared_intrinsics shape invalid: {K.shape}")

        if record.frame_file_count != coords.shape[1]:
            ok = False
            msgs.append(
                f"frames/ count {record.frame_file_count} != annotation num_frames {coords.shape[1]}"
            )

        # Confirm extrinsics convention using camera positions.
        # For w2c, camera center in world coordinates is: C = -R^T t
        if "camera" in ann and isinstance(ann["camera"], dict) and "positions" in ann["camera"]:
            pos = np.asarray(ann["camera"]["positions"], dtype=np.float32)  # [T,3]
            E = extrinsics.astype(np.float32)                               # [T,3,4]
            R = E[:, :, :3]
            t = E[:, :, 3]
            pos_from_w2c = -np.einsum("tji,tj->ti", R, t)
            err_w2c = np.linalg.norm(pos_from_w2c - pos, axis=1).mean()
            msgs.append(f"mean position error assuming w2c: {err_w2c:.6e}")

        return {
            "dataset_name": self.dataset_name,
            "sequence_name": sequence_name,
            "ok": ok,
            "messages": msgs,
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_index(self) -> list[_KubricSceneRecord]:
        scene_dirs = sorted(
            [Path(e.path) for e in os.scandir(self.root) if e.is_dir()],
            key=lambda p: p.name,
        )

        def _index_one(p: Path) -> Optional[_KubricSceneRecord]:
            seq = p.name
            # os.scandir() returns DirEntry with cached type — no extra stat per entry.
            try:
                top_entries = {e.name: e for e in os.scandir(p)}
            except Exception:
                return None

            nested_entry = top_entries.get(seq)
            if nested_entry is not None and nested_entry.is_dir():
                seq_dir = Path(nested_entry.path)
                try:
                    present = {e.name: Path(e.path) for e in os.scandir(seq_dir)}
                except Exception:
                    return None
            else:
                seq_dir = p
                present = {name: Path(e.path) for name, e in top_entries.items()}

            ann_name = f"{seq}.npy"
            rank_name = f"{seq}_with_rank.npz"
            if ann_name not in present or rank_name not in present or "frames" not in present:
                return None

            ann_path = present[ann_name]
            rank_path = present[rank_name]
            frame_dir = present["frames"]

            h5_name = f"{seq}.h5"
            h5_path = present[h5_name] if h5_name in present else None

            trajs_2d_name = f"{seq}_trajs_2d.npy"
            trajs_2d_path = present[trajs_2d_name] if trajs_2d_name in present else None

            num_frames, num_tracks, height, width = self._probe_scene_metadata(
                scene_dir=seq_dir,
                sequence_name=seq,
                ann_path=ann_path,
                h5_path=h5_path,
                trajs_2d_path=trajs_2d_path,
            )
            frame_names, frame_name_width, frame_name_suffix = self._maybe_build_sequential_names(
                suffix_candidates=(".png", ".jpg", ".jpeg"),
            )
            depth_dir = present.get("depths")
            if depth_dir is not None:
                depth_names, depth_name_width, depth_name_suffix = self._maybe_build_sequential_names(
                    suffix_candidates=(".npy",),
                )
            else:
                depth_names = None
                depth_name_width = 3
                depth_name_suffix = ".npy"
            return _KubricSceneRecord(
                sequence_name=seq,
                scene_dir=seq_dir,
                ann_path=ann_path,
                rank_path=rank_path,
                h5_path=h5_path,
                trajs_2d_path=trajs_2d_path,
                frame_names=frame_names,
                depth_names=depth_names,
                num_frames=num_frames,
                num_tracks=num_tracks,
                height=height,
                width=width,
                frame_file_count=num_frames,
                frame_name_width=frame_name_width,
                frame_name_suffix=frame_name_suffix,
                depth_name_width=depth_name_width,
                depth_name_suffix=depth_name_suffix,
                has_depth_dir=depth_dir is not None,
            )

        n_workers = min(self.index_workers, len(scene_dirs))
        if n_workers > 1:
            results: list[Optional[_KubricSceneRecord]] = [None] * len(scene_dirs)
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                future_to_idx = {executor.submit(_index_one, p): i for i, p in enumerate(scene_dirs)}
                for fut in as_completed(future_to_idx):
                    results[future_to_idx[fut]] = fut.result()
            records = [r for r in results if r is not None]
        else:
            records = [r for p in scene_dirs if (r := _index_one(p)) is not None]

        if len(records) == 0:
            raise RuntimeError(f"No valid Kubric scenes found under: {self.root}")
        return records

    def _get_record(self, sequence_name: str) -> _KubricSceneRecord:
        if sequence_name not in self._name_to_record:
            raise KeyError(f"Unknown sequence_name: {sequence_name}")
        return self._name_to_record[sequence_name]

    def _resolve_scene_dir(self, scene_root: Path, sequence_name: str) -> Path:
        nested_dir = scene_root / sequence_name
        return nested_dir if nested_dir.is_dir() else scene_root

    def _probe_scene_metadata(
        self,
        scene_dir: Path,
        sequence_name: str,
        ann_path: Path,
        h5_path: Optional[Path],
        trajs_2d_path: Optional[Path],
    ) -> tuple[int, int, int, int]:
        if trajs_2d_path is not None:
            try:
                num_frames, num_tracks = self._probe_track_array_metadata(trajs_2d_path)
                return num_frames, num_tracks, -1, -1
            except Exception:
                pass

        if h5_path is not None:
            try:
                num_frames, num_tracks = self._probe_h5_metadata(h5_path)
                return num_frames, num_tracks, -1, -1
            except Exception:
                pass

        ann = np.load(ann_path, allow_pickle=True).item()
        coords = np.asarray(ann["coords"])
        depth = np.asarray(ann["depth"])
        return (
            int(coords.shape[1]),
            int(coords.shape[0]),
            int(depth.shape[1]),
            int(depth.shape[2]),
        )

    def _probe_track_array_metadata(self, trajs_2d_path: Path) -> tuple[int, int]:
        # Read only numpy header (first ~128 bytes) instead of mmap entire file.
        import struct
        with open(trajs_2d_path, "rb") as f:
            magic = f.read(6)
            if magic[:6] != b"\x93NUMPY":
                raise ValueError(f"Not a numpy file: {trajs_2d_path}")
            major, minor = struct.unpack("BB", f.read(2))
            if major == 1:
                header_len = struct.unpack("<H", f.read(2))[0]
            elif major in (2, 3):
                header_len = struct.unpack("<I", f.read(4))[0]
            else:
                raise ValueError(f"Unsupported numpy version {major}.{minor}")
            header_bytes = f.read(header_len)
            header_str = header_bytes.decode("latin1")
            # Parse shape from header dict string like "{'descr': '<f4', 'fortran_order': False, 'shape': (T, N, 2)}"
            import ast
            header_dict = ast.literal_eval(header_str)
            shape = header_dict["shape"]
        if len(shape) != 3 or shape[-1] != 2:
            raise ValueError(f"Invalid trajs_2d shape in {trajs_2d_path}: {shape}")
        dim0, dim1 = int(shape[0]), int(shape[1])
        if dim0 > dim1:
            # Kubric *_trajs_2d.npy stores [N, T, 2], while .h5 stores [T, N, 2].
            return dim1, dim0
        return dim0, dim1

    def _probe_h5_metadata(self, h5_path: Path) -> tuple[int, int]:
        import h5py

        with h5py.File(h5_path, "r") as hf:
            trajs_2d = hf["trajs_2d"]
            num_frames = int(hf["num_frames"][()]) if "num_frames" in hf else int(trajs_2d.shape[0])
            return num_frames, int(trajs_2d.shape[1])

    def _maybe_build_sequential_names(
        self,
        suffix_candidates: tuple[str, ...],
    ) -> tuple[list[str] | None, int, str]:
        return None, 3, suffix_candidates[0]

    def _frame_name_for_index(self, record: _KubricSceneRecord, index: int) -> str:
        if record.frame_names is not None:
            return record.frame_names[index]
        return f"{index:0{record.frame_name_width}d}{record.frame_name_suffix}"

    def _depth_name_for_index(self, record: _KubricSceneRecord, index: int) -> str:
        if record.depth_names is not None:
            return record.depth_names[index]
        return f"{index:0{record.depth_name_width}d}{record.depth_name_suffix}"

    def _frame_path_for_index(self, record: _KubricSceneRecord, index: int) -> Path:
        path = record.frame_dir / self._frame_name_for_index(record, index)
        if path.exists():
            return path
        if record.frame_names is None:
            record.frame_names = self._list_matching_names(record.frame_dir, {".png", ".jpg", ".jpeg"})
            record.frame_file_count = len(record.frame_names)
        return record.frame_dir / record.frame_names[index]

    def _depth_path_for_index(self, record: _KubricSceneRecord, index: int) -> Path:
        path = record.depth_dir / self._depth_name_for_index(record, index)
        if path.exists():
            return path
        if record.depth_names is None:
            record.depth_names = self._list_matching_names(record.depth_dir, {".npy"})
        return record.depth_dir / record.depth_names[index]

    def _resolve_record_image_size(self, record: _KubricSceneRecord) -> tuple[int, int]:
        try:
            return self._read_image_size(self._frame_path_for_index(record, 0))
        except Exception:
            ann = np.load(record.ann_path, allow_pickle=True).item()
            depth = np.asarray(ann["depth"])
            return int(depth.shape[1]), int(depth.shape[2])

    def _check_scene_exists(self, scene_dir: Path, sequence_name: str) -> None:
        if not scene_dir.exists():
            raise FileNotFoundError(f"[{sequence_name}] scene dir not found: {scene_dir}")
        if not (scene_dir / f"{sequence_name}.npy").exists():
            raise FileNotFoundError(f"[{sequence_name}] missing file: {scene_dir / f'{sequence_name}.npy'}")
        if not (scene_dir / f"{sequence_name}_with_rank.npz").exists():
            raise FileNotFoundError(
                f"[{sequence_name}] missing file: {scene_dir / f'{sequence_name}_with_rank.npz'}"
            )
        if not (scene_dir / "frames").exists():
            raise FileNotFoundError(f"[{sequence_name}] missing frames dir: {scene_dir / 'frames'}")

    def _list_matching_names(self, directory: Path, suffixes: set[str]) -> list[str]:
        names: list[str] = []
        for entry in os.scandir(directory):
            if os.path.splitext(entry.name)[1].lower() in suffixes:
                names.append(entry.name)
        return sorted(names)

    def _read_image(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))

    def _read_image_size(self, path: Path) -> tuple[int, int]:
        # Read only image header to get dimensions, not the full image data.
        with open(path, "rb") as f:
            header = f.read(24)

        # PNG: starts with \x89PNG, size at bytes 16-23
        if header[:8] == b'\x89PNG\r\n\x1a\n':
            import struct
            width, height = struct.unpack(">II", header[16:24])
            return height, width

        # JPEG: scan for SOF0/SOF2 marker
        if header[:2] == b'\xff\xd8':
            with open(path, "rb") as f:
                f.seek(2)
                while True:
                    marker = f.read(2)
                    if len(marker) != 2:
                        break
                    if marker[0] != 0xff:
                        break
                    marker_type = marker[1]
                    if marker_type in (0xc0, 0xc2):  # SOF0 or SOF2
                        f.read(3)  # skip length + precision
                        import struct
                        height, width = struct.unpack(">HH", f.read(4))
                        return int(height), int(width)
                    elif marker_type in (0xd8, 0xd9, 0x01):  # SOI, EOI, TEM
                        continue
                    else:
                        import struct
                        length = struct.unpack(">H", f.read(2))[0]
                        f.seek(length - 2, 1)

        # Fallback to PIL for other formats
        with Image.open(path) as image:
            width, height = image.size
        return height, width

    def _check_indices(self, frame_indices: list[int], num_frames: int, sequence_name: str) -> None:
        if len(frame_indices) == 0:
            raise ValueError("frame_indices is empty")
        if min(frame_indices) < 0 or max(frame_indices) >= num_frames:
            raise IndexError(
                f"[{sequence_name}] frame_indices out of range: "
                f"min={min(frame_indices)}, max={max(frame_indices)}, num_frames={num_frames}"
            )

    def _to_4x4(self, extrinsics_t34: np.ndarray) -> np.ndarray:
        """
        Convert [T,3,4] -> [T,4,4].
        Input extrinsics are world-to-camera (w2c).
        """
        T = extrinsics_t34.shape[0]
        out = np.zeros((T, 4, 4), dtype=np.float32)
        out[:, :3, :4] = extrinsics_t34
        out[:, 3, 3] = 1.0
        return out

    def _compute_depth_visibility(
        self,
        trajs_2d: np.ndarray,           # [T,N,2]
        coords_depth: np.ndarray,       # [T,N]
        depths: list | None,            # list of [H,W] frame depth
        metadata_raw: dict | None,      # ann["metadata"] (npy path) or None (h5 path)
        ann: dict | None,
        frame_indices: list[int],
        H_img: int,
        W_img: int,
    ) -> np.ndarray | None:
        """
        Compute per-track per-frame visibility from a depth z-buffer test:
            visible[n,t] = |coords_depth[n,t] - frame_depth[t, v, u]| / frame_depth < tol

        Returns a [T,N] bool array, or None if frame depth is unavailable.

        - npy path: ``ann["depth"]`` is uint16 quantized, ``depths`` is the same
          values as float32. Need to dequantize using ``ann["metadata"]["depth_range"]``.
        - h5 path: ``depths/<idx>.npy`` already stores depth in meters, so the
          values can be used directly.
        """
        if depths is None or len(depths) == 0:
            return None
        fd_q = np.stack([np.asarray(d, dtype=np.float32) for d in depths], axis=0)  # [T,H,W]
        # Dequantize if values look like raw uint16 codes (>= 0..65535).
        if metadata_raw is not None and isinstance(metadata_raw, dict):
            depth_range = metadata_raw.get("depth_range")
            if depth_range is not None:
                dmin = float(np.asarray(depth_range).reshape(-1)[0])
                dmax = float(np.asarray(depth_range).reshape(-1)[1])
                # Heuristic: if values exceed plausible meter range, treat as quantized.
                if fd_q.max() > max(dmax * 2.0, 1024.0):
                    fd_m = dmin + (fd_q / 65535.0) * (dmax - dmin)
                else:
                    fd_m = fd_q
            else:
                fd_m = fd_q
        else:
            # h5 path: depths are already in meters
            fd_m = fd_q
        T_clip, H_d, W_d = fd_m.shape
        xy_finite = np.isfinite(trajs_2d[..., 0]) & np.isfinite(trajs_2d[..., 1])
        in_bounds = (
            xy_finite
            & (trajs_2d[..., 0] >= 0)
            & (trajs_2d[..., 0] < W_d)
            & (trajs_2d[..., 1] >= 0)
            & (trajs_2d[..., 1] < H_d)
        )
        u = np.zeros(trajs_2d.shape[:2], dtype=np.int32)
        v = np.zeros(trajs_2d.shape[:2], dtype=np.int32)
        if np.any(in_bounds):
            u[in_bounds] = np.clip(
                np.round(trajs_2d[..., 0][in_bounds]).astype(np.int32),
                0,
                W_d - 1,
            )
            v[in_bounds] = np.clip(
                np.round(trajs_2d[..., 1][in_bounds]).astype(np.int32),
                0,
                H_d - 1,
            )
        idx_flat = (v * W_d + u).astype(np.int64)                # [T,N]
        fd_at = np.take_along_axis(fd_m.reshape(T_clip, -1), idx_flat, axis=1)
        denom = np.maximum(fd_at, 1e-6)
        rel = np.abs(coords_depth - fd_at) / denom
        depth_visible = (
            in_bounds
            & (rel < self.visibility_depth_rel_tol)
            & np.isfinite(coords_depth)
            & np.isfinite(fd_at)
            & (coords_depth > 0)
        )
        return depth_visible

    def _backproject_tracks_to_world(
        self,
        trajs_2d: np.ndarray,        # [T,N,2]
        coords_depth: np.ndarray,    # [T,N]
        intrinsics: np.ndarray,      # [T,3,3]
        extrinsics_w2c: np.ndarray,  # [T,4,4]
    ) -> np.ndarray:
        """
        Reconstruct world-space track points from:
        - 2D pixel coords
        - per-track depth in camera z
        - K
        - w2c extrinsics

        world point = inv(w2c) @ (z * inv(K) @ [u,v,1])
        """
        T, N, _ = trajs_2d.shape
        trajs_3d_world = np.full((T, N, 3), np.nan, dtype=np.float32)

        for t in range(T):
            uv = trajs_2d[t]            # [N,2]
            z = coords_depth[t]         # [N]
            K = intrinsics[t]           # [3,3]
            w2c = extrinsics_w2c[t]     # [4,4]

            valid = (
                np.isfinite(uv[:, 0])
                & np.isfinite(uv[:, 1])
                & np.isfinite(z)
                & (z > 0)
            )
            if not np.any(valid):
                continue

            uv_valid = uv[valid]
            z_valid = z[valid]  # ray distance (Euclidean), not Z depth

            ones = np.ones((uv_valid.shape[0], 1), dtype=np.float32)
            pix = np.concatenate([uv_valid, ones], axis=-1)  # [M,3]

            K_inv = np.linalg.inv(K).astype(np.float32)
            rays = (K_inv @ pix.T).T                          # [M,3], z-component == 1
            ray_len = np.linalg.norm(rays, axis=1)            # [M], ||[xn,yn,1]||
            z_cam = z_valid / ray_len                         # convert ray_dist -> Z depth
            pts_cam = rays * z_cam[:, None]                   # [M,3]

            pts_cam_h = np.concatenate(
                [pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float32)],
                axis=-1,
            )  # [M,4]

            c2w = np.linalg.inv(w2c).astype(np.float32)
            pts_world_h = (c2w @ pts_cam_h.T).T               # [M,4]
            trajs_3d_world[t, valid] = pts_world_h[:, :3]

        return trajs_3d_world
