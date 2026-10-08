"""
ScanNet dataset adapter for D4RT.

This adapter reads scene data from the ``processed/`` exports. Runtime loading
does not decode ``.sens`` streams. Instead it reads:

- ``processed/images/*.jpg``
- ``processed/depth/*.png``
- ``processed/pose/*.txt``
- ``processed/precomputed.h5`` or ``processed/precomputed.npz``

Expected layout::

    <root>/
        [scans/]
            scene0000_00/
                processed/
                    images/00000.jpg
                    depth/00000.png
                    pose/00000.txt
                    precomputed.h5

``load_clip(sequence_name, frame_indices)`` expects indices in the filtered
processed-frame order ``0 .. num_frames - 1``. These are not raw ``.sens``
frame numbers.

Depth PNGs store raw ScanNet ``uint16`` depth values. The adapter divides by
``depth_scale`` (default ``1000.0``) and resizes depth to the colour image
resolution with nearest-neighbour interpolation.

All returned extrinsics are **world-to-camera** (w2c) matrices in the
axis-aligned world frame: ``p_cam = E @ p_world_aligned``.

Usage example::

    from multicamera4rc.datasets.adapters.scannet import ScanNetAdapter

    adapter = ScanNetAdapter(
        root="/mnt/ccw_1/d4rt/datas/Scannet",
        precompute_root="/mnt/ccw_1/d4rt/datas/Scannet",
        verbose=True,
    )
    clip = adapter.load_clip("scene0000_00", [0, 1, 2, 3])

Constructor parameters
----------------------
- ``root``: ScanNet root. May contain scene directories directly or under
  ``scans/``.
- ``depth_scale``: divisor that converts processed ``uint16`` depth to metres.
- ``default_pose_convention``: must be ``"w2c"``.
- ``precompute_root``: optional root containing ``<scene_id>/processed``
  exports and ``precomputed.h5`` / ``precomputed.npz``.
- ``verbose``: print indexing summary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .base import BaseAdapter, UnifiedClip


# ---------------------------------------------------------------------------
# Normal resizing
# ---------------------------------------------------------------------------

def _resize_normals(normal: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    tgt_h, tgt_w = target_hw
    if normal.shape[:2] == (tgt_h, tgt_w):
        out = normal.astype(np.float32)
    else:
        out = cv2.resize(
            normal.astype(np.float32),
            (tgt_w, tgt_h),
            interpolation=cv2.INTER_LINEAR,
        )
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    safe = norm > 1e-6
    return np.where(safe, out / np.where(safe, norm, 1.0), 0.0).astype(np.float32)


def _read_processed_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing processed image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_processed_depth(
    path: Path,
    depth_scale: float,
    target_hw: tuple[int, int],
) -> np.ndarray:
    depth_u16 = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_u16 is None:
        raise FileNotFoundError(f"Missing processed depth: {path}")
    if depth_u16.ndim != 2:
        raise ValueError(f"Processed depth must be single-channel: {path}")
    depth_m = depth_u16.astype(np.float32) / depth_scale
    depth_m[depth_u16 == 0] = 0.0
    target_h, target_w = target_hw
    if depth_m.shape != (target_h, target_w):
        depth_m = cv2.resize(
            depth_m,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
    return depth_m


def _read_processed_pose(path: Path) -> np.ndarray:
    pose = np.asarray(np.loadtxt(path, dtype=np.float32), dtype=np.float32)
    if pose.size != 16:
        raise ValueError(f"Processed pose must contain 16 values: {path}")
    pose = pose.reshape(4, 4)
    if not np.isfinite(pose).all():
        raise ValueError(f"Processed pose contains non-finite values: {path}")
    return pose


def _read_precomputed_scene_info(
    npz_path: Path,
) -> tuple[int, tuple[int, int], tuple[int, int], bool]:
    h5_path = npz_path.with_suffix(".h5")
    if h5_path.exists():
        import h5py

        with h5py.File(h5_path, "r") as f:
            if "num_frames" not in f or "rgb_hw" not in f or "depth_hw" not in f:
                raise KeyError(
                    f"Missing scene metadata keys in {h5_path.name}: num_frames/rgb_hw/depth_hw"
                )
            num_frames = int(np.asarray(f["num_frames"][()]))
            rgb_hw = np.asarray(f["rgb_hw"][()], dtype=np.int32).reshape(-1)
            depth_hw = np.asarray(f["depth_hw"][()], dtype=np.int32).reshape(-1)
            has_axis_alignment = bool(
                np.asarray(f["has_axis_alignment"][()])
            ) if "has_axis_alignment" in f else False
    elif npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as raw:
            required = {"num_frames", "rgb_hw", "depth_hw"}
            missing = sorted(required.difference(raw.files))
            if missing:
                raise KeyError(
                    f"Missing scene metadata keys in {npz_path.name}: {missing}"
                )
            num_frames = int(np.asarray(raw["num_frames"]))
            rgb_hw = np.asarray(raw["rgb_hw"], dtype=np.int32).reshape(-1)
            depth_hw = np.asarray(raw["depth_hw"], dtype=np.int32).reshape(-1)
            has_axis_alignment = bool(
                np.asarray(raw["has_axis_alignment"])
            ) if "has_axis_alignment" in raw.files else False
    else:
        raise FileNotFoundError(f"Missing precomputed cache: {h5_path} or {npz_path}")

    if rgb_hw.size != 2 or depth_hw.size != 2:
        raise ValueError("rgb_hw/depth_hw metadata must have shape (2,)")
    return (
        num_frames,
        (int(rgb_hw[0]), int(rgb_hw[1])),
        (int(depth_hw[0]), int(depth_hw[1])),
        has_axis_alignment,
    )


# ---------------------------------------------------------------------------
# Sequence record
# ---------------------------------------------------------------------------

class _SequenceRecord:
    """Index entry for a single ScanNet scene."""

    __slots__ = (
        "scene_id",
        "scene_dir",
        "processed_dir",
        "image_dir",
        "depth_dir",
        "pose_dir",
        "precomputed_path",
        "mesh_path",
        "num_frames",
        "color_height",
        "color_width",
        "depth_height",
        "depth_width",
        "has_axis_alignment",
    )

    def __init__(
        self,
        scene_id: str,
        scene_dir: Path,
        processed_dir: Path,
        image_dir: Path,
        depth_dir: Path,
        pose_dir: Path,
        precomputed_path: Path,
        mesh_path: Optional[Path],
        num_frames: int,
        color_hw: tuple[int, int],
        depth_hw: tuple[int, int],
        has_axis_alignment: bool,
    ) -> None:
        self.scene_id = scene_id
        self.scene_dir = scene_dir
        self.processed_dir = processed_dir
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.pose_dir = pose_dir
        self.precomputed_path = precomputed_path
        self.mesh_path = mesh_path
        self.num_frames = int(num_frames)
        self.color_height = int(color_hw[0])
        self.color_width = int(color_hw[1])
        self.depth_height = int(depth_hw[0])
        self.depth_width = int(depth_hw[1])
        self.has_axis_alignment = bool(has_axis_alignment)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ScanNetAdapter(BaseAdapter):
    """Dataset adapter for ScanNet.

    See module docstring for full details on data layout, alignment, and
    supervision availability.
    """

    dataset_name: str = "scannet"

    def __init__(
        self,
        root: str,
        split: str = "train",
        depth_scale: float = 1000.0,
        default_pose_convention: str = "w2c",
        precompute_root: Optional[str] = None,
        verbose: bool = True,
        cache_dir: Optional[str] = None,
        recompute_legacy_visibility: bool = True,
        load_normals: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        root :
            Root directory of the ScanNet dataset.  May contain scene
            directories directly, or a ``scans/`` subdirectory.
        depth_scale :
            Divisor for raw uint16 depth values to obtain metres.
            Default 1000.0 (ScanNet stores depth in millimetres).
        default_pose_convention :
            Pose convention for the returned extrinsics.
            Must be ``"w2c"`` (the only supported value).
        precompute_root :
            Optional root containing ``<scene_id>/processed`` exports,
            including ``precomputed.h5`` / ``precomputed.npz``.
            Defaults to the resolved scene root under *root*.
        verbose :
            Print summary during construction.
        """
        self.root = Path(root)
        # Check cache first to skip slow root.exists() on remote storage
        _cache_hit = False
        if cache_dir is not None:
            from multicamera4rc.datasets.index_cache import load_or_build
            import hashlib, json as _json
            cache_key = {
                "dataset": "scannet",
                "cache_schema": 1,
            }
            suffix = hashlib.sha1(
                _json.dumps(cache_key, sort_keys=True).encode()
            ).hexdigest()[:12]
            cache_path = Path(cache_dir) / f"scannet_train_{suffix}.pkl"
            if cache_path.exists():
                _cache_hit = True
        if not _cache_hit and not self.root.exists():
            raise FileNotFoundError(f"ScanNet root not found: {self.root}")
        if default_pose_convention != "w2c":
            raise ValueError(
                f"ScanNetAdapter only supports default_pose_convention='w2c', got {default_pose_convention!r}"
            )

        self.depth_scale = depth_scale
        self.default_pose_convention = default_pose_convention
        self.verbose = verbose
        self.recompute_legacy_visibility = bool(recompute_legacy_visibility)
        # When False, skip reading the per-pixel `normals` array from the
        # precomputed h5/npz.  loss_normal is metric-only and disabled by
        # default (LOSS_W_NORMAL=0), so reading normals is wasted IO.
        self.load_normals = bool(load_normals)

        # Locate scene directories
        scans_sub = self.root / "scans"
        self._scans_dir = scans_sub if scans_sub.is_dir() else self.root
        if precompute_root:
            pr = Path(precompute_root)
            pr_scans = pr / "scans"
            self.precompute_root = pr_scans if pr_scans.is_dir() else pr
        else:
            self.precompute_root = self._scans_dir

        # Build the sequence index once; workers then read only sampled clips.
        _from_cache = False
        if cache_dir is not None:
            from multicamera4rc.datasets.index_cache import load_or_build
            import hashlib, json as _json
            cache_key = {
                "dataset": self.dataset_name,
                "root": str(self.root),
                "scans_dir": str(self._scans_dir),
                "precompute_root": str(self.precompute_root),
                "cache_schema": 2,
            }
            suffix = hashlib.sha1(
                _json.dumps(cache_key, sort_keys=True).encode()
            ).hexdigest()[:12]
            cache_path = Path(cache_dir) / f"{self.dataset_name}_train_{suffix}.pkl"
            index: list[tuple[str, int]] = load_or_build(self._build_index_tuples, cache_path)
            _from_cache = cache_path.exists()
        else:
            index = self._build_index_tuples()

        self._records: list[_SequenceRecord] = []
        self._name_to_record: dict[str, _SequenceRecord] = {}
        for scene_id, num_frames in index:
            try:
                rec = self._make_record(
                    scene_id, num_frames, skip_validation=_from_cache
                )
                self._records.append(rec)
                self._name_to_record[scene_id] = rec
            except Exception:
                pass

        if not self._records:
            raise RuntimeError(
                f"No valid ScanNet scenes found under {self.root}"
            )

        if self.verbose:
            print(
                f"[ScanNetAdapter] root={self.root}, "
                f"num_sequences={len(self._records)}"
            )

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def list_sequences(self) -> list[str]:
        return [r.scene_id for r in self._records]

    def get_sequence_name(self, index: int) -> str:
        return self._records[index].scene_id

    def get_num_frames(self, sequence_name: str) -> int:
        return self._get_record(sequence_name).num_frames

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        r = self._get_record(sequence_name)
        return {
            "dataset_name": self.dataset_name,
            "sequence_name": r.scene_id,
            "sequence_root": str(r.scene_dir),
            "num_frames": r.num_frames,
            "height": r.color_height,
            "width": r.color_width,
            "depth_height": r.depth_height,
            "depth_width": r.depth_width,
            "has_depth": True,
            "has_normals": True,
            "has_tracks": True,
            "has_visibility": True,
            "has_trajs_3d_world": True,
            "extrinsics_convention": "w2c",
            "depth_unit": "meters",
            "has_mesh": r.mesh_path is not None,
        }

    def load_clip(
        self, sequence_name: str, frame_indices: list[int]
    ) -> UnifiedClip:
        """Load a clip from a ScanNet scene.

        Parameters
        ----------
        sequence_name :
            Scene ID (e.g. ``"scene0000_00"``).
        frame_indices :
            Frame indices within ``[0, num_frames)``.

        Returns
        -------
        UnifiedClip
            Images at colour resolution, depth resized to colour resolution,
            intrinsics for the colour camera, extrinsics as w2c in the
            axis-aligned world frame.
        """
        r = self._get_record(sequence_name)
        self._check_indices(frame_indices, r.num_frames, sequence_name)

        T = len(frame_indices)
        color_h, color_w = r.color_height, r.color_width
        valid_pose = np.ones(T, dtype=bool)
        frame_paths = [str(r.image_dir / f"{fi:05d}.jpg") for fi in frame_indices]
        color_images = [
            _read_processed_image(r.image_dir / f"{fi:05d}.jpg")
            for fi in frame_indices
        ]
        depths = [
            _read_processed_depth(
                r.depth_dir / f"{fi:05d}.png",
                self.depth_scale,
                (color_h, color_w),
            )
            for fi in frame_indices
        ]
        extrinsics = np.stack(
            [
                _read_processed_pose(r.pose_dir / f"{fi:05d}.txt")
                for fi in frame_indices
            ],
            axis=0,
        ).astype(np.float32)

        # ---- Load precomputed cache --------------------------------------
        cache = self._load_precomputed(sequence_name, frame_indices)
        intrinsics = cache["intrinsics"].astype(np.float32)
        normals_out: Optional[list[np.ndarray]] = None
        trajs_2d_out = cache["trajs_2d"].astype(np.float32)
        trajs_3d_out = cache["trajs_3d_world"].astype(np.float32)
        valids_out = cache["valids"].astype(bool)
        visibs_out = cache["visibs"].astype(bool)
        has_tracks_out = True
        normals_source = None
        legacy_visibility_recomputed = False
        if "normals" in cache:
            normals_out = [
                _resize_normals(normal, (color_h, color_w))
                for normal in cache["normals"]
            ]
            normals_source = "precomputed"

        # Older ScanNet precomputed caches do not record track semantics
        # metadata. Their cached visibs can disagree with the current
        # depth/pose-projection rule, so refresh the projected 2D tracks and
        # defined/visible masks for the loaded clip on the fly.
        if self.recompute_legacy_visibility and "track_semantics_version" not in cache:
            from multicamera4rc.datasets.computer.depth_to_tracks import recompute_track_projection_masks

            track_depth_max = float(np.asarray(cache.get("track_depth_max", 100.0)).reshape(-1)[0])
            refreshed = recompute_track_projection_masks(
                depths=depths,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                trajs_3d_world=trajs_3d_out,
                depth_consistency_thresh=0.05,
                depth_max=track_depth_max,
            )
            trajs_2d_out = refreshed["trajs_2d"].astype(np.float32)
            valids_out = refreshed["valids"].astype(bool)
            visibs_out = refreshed["visibs"].astype(bool)
            legacy_visibility_recomputed = True

        has_normals_out = normals_out is not None

        return UnifiedClip(
            dataset_name=self.dataset_name,
            sequence_name=sequence_name,
            frame_paths=frame_paths,
            images=color_images,
            depths=depths,
            normals=normals_out,
            trajs_2d=trajs_2d_out,
            trajs_3d_world=trajs_3d_out,
            valids=valids_out,
            visibs=visibs_out,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            metadata={
                "dataset_name": self.dataset_name,
                "sequence_root": str(r.scene_dir),
                "frame_indices": list(frame_indices),
                "num_frames_in_sequence": r.num_frames,
                "raw_image_hw": [color_h, color_w],
                "has_depth": True,
                "has_normals": has_normals_out,
                "has_tracks": has_tracks_out,
                "has_visibility": has_tracks_out,
                "has_trajs_3d_world": has_tracks_out,
                "normals_source": normals_source,
                "pose_convention": "w2c",
                "extrinsics_convention": "w2c",
                "intrinsics_convention": "pinhole",
                "depth_unit": "meters",
                "depth_scale": self.depth_scale,
                "has_axis_alignment": r.has_axis_alignment,
                "valid_pose_mask": valid_pose.tolist(),
                "track_semantics_version": (
                    int(np.asarray(cache["track_semantics_version"]).reshape(-1)[0])
                    if "track_semantics_version" in cache
                    else None
                ),
                "legacy_visibility_recomputed": legacy_visibility_recomputed,
            },
        )

    def _load_precomputed(
        self, sequence_name: str, frame_indices: list[int]
    ) -> dict:
        path = self._get_precomputed_npz_path(sequence_name)
        h5_path = path.with_suffix(".h5")
        if not path.exists() and not h5_path.exists():
            raise FileNotFoundError(
                f"Missing precomputed cache: {h5_path} or {path}"
            )

        required = [
            "trajs_2d",
            "trajs_3d_world",
            "valids",
            "visibs",
            "intrinsics",
        ]
        optional = [
            "normals",
            "extrinsics",
            "track_semantics_version",
            "ref_frame",
            "ref_frames",
            "num_ref_segments",
            "track_depth_max",
            "source_depth_sampling_mode",
        ]
        if not self.load_normals:
            # loss_normal disabled — skip the per-pixel 3-channel normals read.
            optional = [k for k in optional if k != "normals"]
        selected_keys = required + optional

        if h5_path.exists():
            import h5py

            sorted_idx = sorted(set(frame_indices))
            idx_map = {v: i for i, v in enumerate(sorted_idx)}
            reorder = [idx_map[i] for i in frame_indices]
            needs_reorder = reorder != list(range(len(frame_indices)))

            cache: dict[str, Any] = {}
            with h5py.File(h5_path, "r") as f:
                missing = [key for key in required if key not in f]
                if missing:
                    raise KeyError(f"Missing keys in {h5_path.name}: {missing}")
                for key in selected_keys:
                    if key not in f:
                        continue
                    data = f[key][sorted_idx]
                    if needs_reorder:
                        data = data[reorder]
                    cache[key] = data
            return cache

        cache = {}
        with np.load(path, allow_pickle=False) as raw:
            missing = [key for key in required if key not in raw.files]
            if missing:
                raise KeyError(f"Missing keys in {path.name}: {missing}")
            index_array = np.asarray(frame_indices)
            for key in selected_keys:
                if key not in raw.files:
                    continue
                cache[key] = raw[key][index_array]
        return cache

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        """Run consistency checks on a sequence.

        Checks
        ------
        1. Required files (.sens, .txt) exist.
        2. ``load_clip`` succeeds on first 5 frames.
        3. Intrinsics/extrinsics shapes and finite-ness.
        4. Depth values finite with some positive values per frame.
        5. Round-trip reprojection error < 1 px at image centre.
        """
        r = self._get_record(sequence_name)
        msgs: list[str] = []
        ok = True

        for label, p in [
            ("processed_dir", r.processed_dir),
            ("images_dir", r.image_dir),
            ("depth_dir", r.depth_dir),
            ("pose_dir", r.pose_dir),
            ("precomputed", r.precomputed_path),
        ]:
            if not p.exists():
                ok = False
                msgs.append(f"missing {label}: {p}")

        if not ok:
            return {
                "dataset_name": self.dataset_name,
                "sequence_name": sequence_name,
                "ok": False,
                "messages": msgs,
            }

        probe_count = min(5, r.num_frames)
        try:
            clip = self.load_clip(sequence_name, list(range(probe_count)))
        except Exception as exc:
            return {
                "dataset_name": self.dataset_name,
                "sequence_name": sequence_name,
                "ok": False,
                "messages": msgs + [f"load_clip failed: {repr(exc)}"],
            }

        T = probe_count

        # Shape checks
        if clip.intrinsics.shape != (T, 3, 3):
            ok = False
            msgs.append(
                f"intrinsics shape {clip.intrinsics.shape} != ({T}, 3, 3)"
            )
        if clip.extrinsics.shape != (T, 4, 4):
            ok = False
            msgs.append(
                f"extrinsics shape {clip.extrinsics.shape} != ({T}, 4, 4)"
            )

        # Finite checks
        if not np.isfinite(clip.intrinsics).all():
            ok = False
            msgs.append("intrinsics contains non-finite values")
        if not np.isfinite(clip.extrinsics).all():
            ok = False
            msgs.append("extrinsics contains non-finite values")

        # Rotation non-singularity
        for t in range(T):
            det = float(np.linalg.det(clip.extrinsics[t, :3, :3]))
            if abs(det) < 1e-6:
                ok = False
                msgs.append(
                    f"extrinsics[{t}] rotation det ≈ {det:.2e} (singular)"
                )

        # Depth sanity
        for t in range(T):
            dep = clip.depths[t]
            if not np.isfinite(dep).all():
                ok = False
                msgs.append(f"depth[{t}] contains non-finite values")
            if (dep > 0).sum() == 0:
                ok = False
                msgs.append(f"depth[{t}] has no positive values")

        # Round-trip reprojection at image centre
        valid_pose = clip.metadata.get("valid_pose_mask", [True] * T)
        for t in range(T):
            if not valid_pose[t]:
                continue
            K = clip.intrinsics[t].astype(np.float64)
            E = clip.extrinsics[t].astype(np.float64)
            dep = clip.depths[t]
            H, W = dep.shape
            cy_px, cx_px = H // 2, W // 2
            d = float(dep[cy_px, cx_px])

            if not (np.isfinite(d) and d > 0):
                continue

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            x_c = (cx_px - cx) / fx * d
            y_c = (cy_px - cy) / fy * d
            p_cam = np.array([x_c, y_c, d, 1.0], dtype=np.float64)

            try:
                E_inv = np.linalg.inv(E)
            except np.linalg.LinAlgError:
                ok = False
                msgs.append(f"extrinsics[{t}] singular")
                continue

            p_world = E_inv @ p_cam
            p_cam2 = E @ p_world
            p_proj = K @ p_cam2[:3]
            u = p_proj[0] / p_proj[2]
            v = p_proj[1] / p_proj[2]
            reproj_err = max(abs(u - cx_px), abs(v - cy_px))
            if reproj_err > 1.0:
                ok = False
                msgs.append(
                    f"reprojection error frame {t}: "
                    f"({u:.2f}, {v:.2f}) vs ({cx_px}, {cy_px}), "
                    f"err={reproj_err:.4f} px"
                )

        if ok and not msgs:
            msgs.append("all checks passed")

        return {
            "dataset_name": self.dataset_name,
            "sequence_name": sequence_name,
            "ok": ok,
            "messages": msgs,
            "num_frames": r.num_frames,
            "probe_frames": probe_count,
        }

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    @staticmethod
    def _find_scene_root(root: Path) -> Path:
        """Locate the directory containing ``scene*`` subdirectories."""
        # Check if root directly contains scene dirs
        if any(
            d.name.startswith("scene") for d in root.iterdir()
            if d.is_dir()
        ):
            return root
        # Check for scans/ subdirectory
        scans = root / "scans"
        if scans.is_dir():
            return scans
        # Fallback
        return root

    def _build_index_tuples(self) -> list[tuple[str, int]]:
        """Return (scene_id, num_frames) for all valid scenes. Cacheable."""
        records = self._build_index()
        return [(r.scene_id, r.num_frames) for r in records]

    def _make_record(
        self, scene_id: str, num_frames: int, *, skip_validation: bool = False
    ) -> _SequenceRecord:
        """Reconstruct a _SequenceRecord from cached (scene_id, num_frames).

        When *skip_validation* is True, skip NFS stat calls for directories and
        files.  Safe when the record comes from a validated index cache.
        """
        scene_dir = self._scans_dir / scene_id
        processed_scene_dir = self.precompute_root / scene_id
        processed_dir = processed_scene_dir / "processed"
        image_dir = processed_dir / "images"
        depth_dir = processed_dir / "depth"
        pose_dir = processed_dir / "pose"
        precomputed_h5 = processed_dir / "precomputed.h5"
        precomputed_npz = processed_dir / "precomputed.npz"

        if skip_validation:
            precomputed_path = precomputed_h5
        else:
            precomputed_path = precomputed_h5 if precomputed_h5.exists() else precomputed_npz
            if not scene_dir.is_dir():
                raise FileNotFoundError(f"Missing scene_dir: {scene_dir}")
            for label, path in (
                ("processed_dir", processed_dir),
                ("images_dir", image_dir),
                ("depth_dir", depth_dir),
                ("pose_dir", pose_dir),
            ):
                if not path.exists():
                    raise FileNotFoundError(f"Missing {label}: {path}")
            if not precomputed_path.exists():
                raise FileNotFoundError(
                    f"Missing precomputed cache: {precomputed_h5} or {precomputed_npz}"
                )

        return _SequenceRecord(
            scene_id=scene_id,
            scene_dir=scene_dir,
            processed_dir=processed_dir,
            image_dir=image_dir,
            depth_dir=depth_dir,
            pose_dir=pose_dir,
            precomputed_path=precomputed_path,
            mesh_path=None,
            num_frames=num_frames,
            color_hw=(968, 1296),
            depth_hw=(480, 640),
            has_axis_alignment=True,
        )

    def _build_index(self) -> list[_SequenceRecord]:
        records: list[_SequenceRecord] = []
        skipped: list[str] = []

        scene_dirs = sorted(
            d
            for d in self._scans_dir.iterdir()
            if d.is_dir() and d.name.startswith("scene")
        )

        for scene_dir in scene_dirs:
            scene_id = scene_dir.name
            try:
                rec = self._index_scene(scene_id, scene_dir)
                if rec is not None:
                    records.append(rec)
            except Exception as exc:
                skipped.append(f"{scene_id}: {exc}")
                if self.verbose:
                    print(
                        f"[ScanNetAdapter][WARN] skip {scene_id}: {exc}"
                    )

        if self.verbose and skipped:
            print(
                f"[ScanNetAdapter] skipped {len(skipped)} scenes"
            )

        return records

    def _index_scene(
        self, scene_id: str, scene_dir: Path
    ) -> Optional[_SequenceRecord]:
        # Mesh file (optional — used only for precomputation)
        mesh_path = scene_dir / f"{scene_id}_vh_clean.ply"
        if not mesh_path.exists():
            mesh_path = None

        processed_scene_dir = self.precompute_root / scene_id
        processed_dir = processed_scene_dir / "processed"
        image_dir = processed_dir / "images"
        depth_dir = processed_dir / "depth"
        pose_dir = processed_dir / "pose"
        precomputed_path = self._get_precomputed_npz_path(scene_id)
        precomputed_h5_path = precomputed_path.with_suffix(".h5")
        if not precomputed_path.exists() and not precomputed_h5_path.exists():
            raise FileNotFoundError(
                "Missing precomputed cache: "
                f"{precomputed_h5_path} or {precomputed_path}"
            )
        cache_path = precomputed_h5_path if precomputed_h5_path.exists() else precomputed_path
        num_frames, color_hw, depth_hw, has_axis_alignment = _read_precomputed_scene_info(
            precomputed_path
        )

        for label, p in [
            ("processed_dir", processed_dir),
            ("images_dir", image_dir),
            ("depth_dir", depth_dir),
            ("pose_dir", pose_dir),
        ]:
            if not p.exists():
                raise FileNotFoundError(f"Missing {label}: {p}")

        return _SequenceRecord(
            scene_id=scene_id,
            scene_dir=scene_dir,
            processed_dir=processed_dir,
            image_dir=image_dir,
            depth_dir=depth_dir,
            pose_dir=pose_dir,
            precomputed_path=cache_path,
            mesh_path=mesh_path,
            num_frames=num_frames,
            color_hw=color_hw,
            depth_hw=depth_hw,
            has_axis_alignment=has_axis_alignment,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_record(self, sequence_name: str) -> _SequenceRecord:
        if sequence_name not in self._name_to_record:
            raise KeyError(
                f"Unknown sequence_name: '{sequence_name}'. "
                f"Use list_sequences() to enumerate valid names."
            )
        return self._name_to_record[sequence_name]

    def _get_precomputed_npz_path(self, sequence_name: str) -> Path:
        return self.precompute_root / sequence_name / "processed" / "precomputed.npz"

    @staticmethod
    def _check_indices(
        frame_indices: list[int], num_frames: int, sequence_name: str
    ) -> None:
        if not frame_indices:
            raise ValueError("frame_indices is empty")
        lo, hi = min(frame_indices), max(frame_indices)
        if lo < 0 or hi >= num_frames:
            raise IndexError(
                f"[{sequence_name}] frame_indices out of range: "
                f"min={lo}, max={hi}, num_frames={num_frames}"
            )
