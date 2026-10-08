from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from .base import BaseAdapter, UnifiedClip, load_precomputed_fast

try:
    import cv2
except Exception:  # pragma: no cover - optional fast path
    cv2 = None


_MIN_TEMPORAL_TRACK_RATIO = 0.05
_MIN_MEAN_VISIBLE_FRAMES = 1.5
_TRACK_FAILURE_MARKER_NAME = "precomputed.failed.json"
_FRAME_ANNO_CACHE_MAXSIZE = 4   # categories kept per DataLoader worker (~200 MB each)
_TRACK_ARRAY_KEYS = ("trajs_2d", "trajs_3d_world", "valids", "visibs")
_LOCAL_TRACK_OVERLAY_NAME = "precomputed.overlay.npz"
_REAL_VISIBILITY_KEYS = ("has_real_visibility", "co3d_has_real_visibility")

# Co3D stores IEEE-754 float16 bit patterns inside uint16 PNG pixels.  Keep
# this contract in one place so adapter metadata, documentation, and any
# downstream materialisation code cannot drift back to treating the pixels as
# linearly quantised uint16 values.
CO3D_DEPTH_ENCODING = "float16_bits_in_uint16_png"
CO3D_DEPTH_NOTE = (
    "depth = raw_uint16.view(float16).astype(float32) * scale_adjustment; "
    "non-finite values are set to 0; the depth mask is applied separately"
)


# ---------------------------------------------------------------------------
# Sequence index record
# ---------------------------------------------------------------------------

@dataclass
class _Co3DSequenceRecord:
    """Index entry for one Co3Dv2 sequence."""

    category: str
    sequence_name: str
    sequence_dir: Path
    # Frame numbers included in the chosen subset/split (sorted ascending).
    frame_numbers: list[int]
    viewpoint_quality_score: Optional[float] = None
    pointcloud_quality_score: Optional[float] = None
    pointcloud_n_points: Optional[int] = None
    valid_depth_ratio: Optional[float] = None
    foreground_ratio: Optional[float] = None
    has_pointcloud: bool = False
    has_precomputed: bool = False

    @property
    def num_frames(self) -> int:
        return len(self.frame_numbers)

    @property
    def uid(self) -> str:
        """Globally unique sequence ID: '<category>/<sequence_name>'."""
        return f"{self.category}/{self.sequence_name}"


def _overlay_npz_and_label(
    *,
    precompute_root: Path | None,
    track_precompute_root: Path | None,
    sequence_name: str,
) -> tuple[Path | None, str | None]:
    if track_precompute_root is not None:
        return track_precompute_root / sequence_name / "precomputed.npz", "track_precompute_root"
    if precompute_root is not None:
        return precompute_root / sequence_name / _LOCAL_TRACK_OVERLAY_NAME, "local_track_overlay"
    return None, None


def _summarize_precomputed_track_validity(valids: Optional[np.ndarray]) -> dict[str, float]:
    """Summarize how temporal a precomputed track set really is."""
    if valids is None:
        return {
            "mean_visible_frames": 0.0,
            "tracks_ge_2_ratio": 0.0,
            "tracks_ge_4_ratio": 0.0,
        }

    valids_np = np.asarray(valids, dtype=bool)
    if valids_np.ndim != 2 or valids_np.shape[1] == 0:
        return {
            "mean_visible_frames": 0.0,
            "tracks_ge_2_ratio": 0.0,
            "tracks_ge_4_ratio": 0.0,
        }

    per_track_visible = valids_np.sum(axis=0)
    return {
        "mean_visible_frames": float(per_track_visible.mean()),
        "tracks_ge_2_ratio": float((per_track_visible >= 2).mean()),
        "tracks_ge_4_ratio": float((per_track_visible >= 4).mean()),
    }


def _precomputed_tracks_look_temporal(
    valids: Optional[np.ndarray],
    *,
    min_track_ge_2_ratio: float = _MIN_TEMPORAL_TRACK_RATIO,
    min_mean_visible_frames: float = _MIN_MEAN_VISIBLE_FRAMES,
) -> bool:
    """Reject degenerate precomputed caches that only contain single-frame points."""
    stats = _summarize_precomputed_track_validity(valids)
    return (
        stats["tracks_ge_2_ratio"] >= float(min_track_ge_2_ratio)
        and stats["mean_visible_frames"] >= float(min_mean_visible_frames)
    )


def _scalar_from_cache(value: Any) -> Any:
    """Return a Python scalar from npz/h5 scalar payloads."""
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
    return value


def _cache_bool(cache: Optional[dict], key: str, default: bool = False) -> bool:
    if not cache or key not in cache:
        return default
    value = _scalar_from_cache(cache[key])
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    try:
        return bool(value)
    except Exception:
        return default


def _cache_str(cache: Optional[dict], key: str, default: Optional[str] = None) -> Optional[str]:
    if not cache or key not in cache:
        return default
    value = _scalar_from_cache(cache[key])
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if value is None:
        return default
    return str(value)


def _cache_has_real_visibility(cache: Optional[dict]) -> bool:
    return any(_cache_bool(cache, key, default=False) for key in _REAL_VISIBILITY_KEYS)


def _precomputed_path_has_real_visibility(npz_path: Path) -> bool:
    """Check lightweight metadata without loading large track arrays when possible."""
    h5_path = npz_path.with_suffix(".h5")
    if h5_path.exists():
        try:
            import h5py

            with h5py.File(h5_path, "r") as handle:
                for key in _REAL_VISIBILITY_KEYS:
                    if key in handle:
                        return _cache_bool({key: handle[key][()]}, key, default=False)
        except Exception:
            return False
        return False

    if npz_path.exists():
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                for key in _REAL_VISIBILITY_KEYS:
                    if key in data:
                        return _cache_bool({key: data[key]}, key, default=False)
        except Exception:
            return False
    return False


# ---------------------------------------------------------------------------
# Camera conversion helpers
# ---------------------------------------------------------------------------

def _ndc_to_pinhole(
    focal_length: list[float],
    principal_point: list[float],
    image_wh: list[int],
) -> np.ndarray:
    """Convert Co3D NDC camera intrinsics to a standard 3×3 pinhole matrix.

    Co3D stores intrinsics in PyTorch3D NDC (normalised device coordinates)
    with ``intrinsics_format="ndc_isotropic"``:

    - ``focal_length`` is normalised by ``min(W, H) / 2``
    - ``principal_point`` has (0, 0) at the image centre, **x pointing left,
      y pointing up** (opposite to the OpenCV pixel convention)

    We convert to OpenCV pixel-space intrinsics (origin at top-left corner,
    x pointing right, y pointing down).

    Args:
        focal_length:     [fx_ndc, fy_ndc]
        principal_point:  [px_ndc, py_ndc]   (NDC, (0,0) = centre)
        image_wh:         [width, height]

    Returns:
        K: (3, 3) float32 pinhole intrinsics in pixel coordinates.
    """
    # Co3D annotation stores image.size as [H, W] (height first).
    H, W = float(image_wh[0]), float(image_wh[1])
    half_s = min(H, W) / 2.0

    fx = focal_length[0] * half_s
    fy = focal_length[1] * half_s

    # NDC pp sign convention: pp_ndc_x = (W/2 - cx_px) / half_s
    #  → cx_px = W/2 - pp_ndc_x * half_s
    cx = W / 2.0 - principal_point[0] * half_s
    cy = H / 2.0 - principal_point[1] * half_s

    return np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _p3d_to_opencv_extrinsics(
    R_p3d: list[list[float]],
    T_p3d: list[float],
) -> np.ndarray:
    """Convert PyTorch3D camera pose to a standard 4×4 OpenCV-style w2c matrix.

    PyTorch3D uses **row-vector** convention:
        p_cam_p3d = p_world @ R + T

    and its camera space has **+x left, +y up, +z into scene**, whereas OpenCV
    uses **+x right, +y down, +z into scene**.

    Conversion:
        - Transpose R to switch to column-vector convention.
        - Apply axis flip D = diag(−1, −1, 1) to correct +x and +y directions.

        R_cv = D @ R_p3d.T
        T_cv = D @ T_p3d

    The resulting 4×4 matrix satisfies
        p_cam_cv = E_w2c @ p_world_h   (homogeneous column vectors, OpenCV).

    Args:
        R_p3d:  3×3 rotation as nested list (row-vector convention).
        T_p3d:  3-element translation as list.

    Returns:
        E: (4, 4) float32 world-to-camera matrix (OpenCV convention).
    """
    R = np.array(R_p3d, dtype=np.float64)   # (3, 3)
    T = np.array(T_p3d, dtype=np.float64)   # (3,)

    D = np.diag([-1.0, -1.0, 1.0])
    R_cv = (D @ R.T).astype(np.float32)
    T_cv = (D @ T).astype(np.float32)

    E = np.eye(4, dtype=np.float32)
    E[:3, :3] = R_cv
    E[:3, 3] = T_cv
    return E


# ---------------------------------------------------------------------------
# Low-level file I/O
# ---------------------------------------------------------------------------

def _load_jgz(path: Path) -> Any:
    """Load a gzip-compressed JSON file."""
    with gzip.open(path, "rb") as f:
        return json.load(f)


def _load_depth(path: Path, scale_adjustment: float) -> np.ndarray:
    """Load a Co3D 16-bit PNG depth map.

    Co3D depth maps store float16 values packed into uint16 PNG pixels
    (i.e. the raw uint16 bits are reinterpreted as float16, NOT divided by
    65535).  The official loading procedure (pytorch3d load_16big_png_depth):

        raw_uint16 = load_png_as_uint16(path)
        depth = raw_uint16.view(float16).astype(float32) * scale_adjustment
        depth[~isfinite(depth)] = 0

    Args:
        path:              Path to the 16-bit PNG depth file.
        scale_adjustment:  Per-frame scale factor from frame annotations.

    Returns:
        depth: (H, W) float32 array.  Zero indicates invalid / missing depth.
    """
    raw_uint16 = None
    if cv2 is not None:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is not None:
            raw_uint16 = np.asarray(raw, dtype=np.uint16)
    if raw_uint16 is None:
        raw_uint16 = np.array(Image.open(path), dtype=np.uint16)
    depth = raw_uint16.view(np.float16).astype(np.float32) * float(scale_adjustment)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def _estimate_depth_scale_from_ply(
    ply_path: Path,
    frame_annos: list[dict],
    root: Path,
    n_frames: int = 5,
) -> float:
    """Estimate a residual PLY-to-depth scale from point-cloud projections.

    Co3D depth PNG pixels are float16 bit patterns stored in a uint16 image,
    not linearly quantised values.  Decode them through :func:`_load_depth`
    before comparing them with projected PLY camera-Z values.  A geometrically
    consistent sequence should therefore produce a residual scale near 1.

    Args:
        ply_path:    Path to the sequence's pointcloud.ply.
        frame_annos: List of frame annotation dicts for frames with the most
                     depth coverage.  Needs viewpoint R/T/focal_length/
                     principal_point/intrinsics_format and depth path.
        root:        Dataset root (prefix for relative depth paths).
        n_frames:    Max number of frames to use for scale estimation.

    Returns:
        Residual multiplicative scale. Falls back to 1.0 on failure.
    """
    if not ply_path.exists():
        return 1.0

    try:
        # Read PLY point cloud (binary little-endian, float xyz + uchar rgb)
        with open(ply_path, "rb") as f:
            header_lines = []
            while True:
                line = f.readline()
                header_lines.append(line.decode("latin-1").strip())
                if header_lines[-1] == "end_header":
                    break

        # Parse header for vertex count and properties
        n_verts = 0
        properties = []
        for hline in header_lines:
            if hline.startswith("element vertex"):
                n_verts = int(hline.split()[-1])
            elif hline.startswith("property"):
                parts = hline.split()
                properties.append((parts[1], parts[2]))

        if n_verts == 0:
            return 1.0

        # Compute vertex byte size and xyz byte offsets
        _type_sizes = {
            "float": 4, "float32": 4, "double": 8, "float64": 8,
            "int": 4, "int32": 4, "uint": 4, "uint32": 4,
            "short": 2, "int16": 2, "ushort": 2, "uint16": 2,
            "uchar": 1, "uint8": 1, "char": 1, "int8": 1,
        }
        vertex_size = 0
        xyz_offsets: dict[str, int] = {}
        for ptype, pname in properties:
            if pname in ("x", "y", "z"):
                xyz_offsets[pname] = vertex_size
            vertex_size += _type_sizes.get(ptype, 4)

        if "x" not in xyz_offsets or vertex_size == 0:
            return 1.0

        import struct
        with open(ply_path, "rb") as f:
            while True:
                if f.readline().strip() == b"end_header":
                    break
            raw_bytes = f.read(n_verts * vertex_size)

        # Unpack xyz float32 values using struct
        pts = np.zeros((n_verts, 3), dtype=np.float32)
        for i, axis in enumerate(("x", "y", "z")):
            off = xyz_offsets[axis]
            pts[:, i] = [
                struct.unpack_from("<f", raw_bytes, vi * vertex_size + off)[0]
                for vi in range(n_verts)
            ]

    except Exception:
        return 1.0

    scales = []
    for anno in frame_annos[:n_frames]:
        try:
            vp = anno["viewpoint"]
            H, W = anno["image"]["size"]
            R_p3d = np.array(vp["R"], dtype=np.float64)
            T_p3d = np.array(vp["T"], dtype=np.float64)

            # Convert PyTorch3D camera to OpenCV (same as load_clip does)
            D = np.diag([-1.0, -1.0, 1.0])
            R_cv = D @ R_p3d.T  # [3,3]
            T_cv = D @ T_p3d    # [3]
            E_cv = np.eye(4, dtype=np.float64)
            E_cv[:3, :3] = R_cv
            E_cv[:3, 3] = T_cv

            # Convert NDC intrinsics to pixel intrinsics (same as load_clip)
            K_cv = _ndc_to_pinhole(vp["focal_length"], vp["principal_point"], anno["image"]["size"])

            # Project PLY points using OpenCV convention
            pts_h = np.concatenate([pts.astype(np.float64), np.ones((len(pts), 1))], axis=1)  # [N,4]
            pts_cam_cv = (E_cv @ pts_h.T).T[:, :3]  # [N,3]
            z_cv = pts_cam_cv[:, 2]

            # Project to 2D using OpenCV pinhole
            fx, fy = float(K_cv[0, 0]), float(K_cv[1, 1])
            cx, cy = float(K_cv[0, 2]), float(K_cv[1, 2])
            u = pts_cam_cv[:, 0] / z_cv * fx + cx
            v = pts_cam_cv[:, 1] / z_cv * fy + cy

            in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z_cv > 0)
            if in_bounds.sum() < 5:
                continue

            xs = np.clip(np.round(u[in_bounds]).astype(int), 0, W - 1)
            ys = np.clip(np.round(v[in_bounds]).astype(int), 0, H - 1)

            depth_path = root / anno["depth"]["path"]
            depth_decoded = _load_depth(
                depth_path,
                float(anno["depth"]["scale_adjustment"]),
            )
            depth_sampled = depth_decoded[ys, xs]
            z_in = z_cv[in_bounds]

            valid = (depth_sampled > 0) & np.isfinite(depth_sampled)
            if valid.sum() < 3:
                continue

            ratio = z_in[valid] / depth_sampled[valid]
            scales.append(float(np.median(ratio)))
        except Exception:
            continue

    if not scales:
        return 1.0

    return float(np.median(scales))


def _render_depth_from_ply(
    pts: np.ndarray,
    E: np.ndarray,
    K: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """Render a depth map from a PLY point cloud using z-buffering.

    Args:
        pts: [N, 3] world-space points (OpenCV world = PyTorch3D world).
        E:   [4, 4] world-to-camera extrinsics (OpenCV convention).
        K:   [3, 3] camera intrinsics.
        H, W: image dimensions.

    Returns:
        depth: [H, W] float32 depth map in camera-space z (metres).
               Zero where no PLY point projects.
    """
    pts_h = np.concatenate([pts.astype(np.float64), np.ones((len(pts), 1))], axis=1)
    pts_cam = (E.astype(np.float64) @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = pts_cam[:, 0] / z * fx + cx
    v = pts_cam[:, 1] / z * fy + cy

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
    if not in_bounds.any():
        return np.zeros((H, W), dtype=np.float32)

    xs = np.clip(np.round(u[in_bounds]).astype(np.int32), 0, W - 1)
    ys = np.clip(np.round(v[in_bounds]).astype(np.int32), 0, H - 1)
    z_in = z[in_bounds].astype(np.float32)

    # Z-buffer: far-to-near so closer points overwrite farther ones
    order = np.argsort(z_in)[::-1]
    depth = np.zeros((H, W), dtype=np.float32)
    depth[ys[order], xs[order]] = z_in[order]
    return depth


def _load_depth_mask(path: Path) -> np.ndarray:
    """Load a Co3D depth validity mask.

    Returns:
        mask: (H, W) bool array.  True = valid depth.
    """
    if cv2 is not None:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is not None:
            return np.asarray(mask).astype(bool, copy=False)
    return np.array(Image.open(path), dtype=bool)


def _load_rgb(path: Path) -> np.ndarray:
    """Load an RGB image as uint8 (H, W, 3)."""
    if cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.asarray(Image.open(path).convert("RGB"))


def _load_soft_mask(path: Path) -> np.ndarray:
    """Load a soft foreground mask to [0, 1]."""
    if cv2 is not None:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is not None:
            arr = np.asarray(raw, dtype=np.float32)
        else:
            arr = np.asarray(Image.open(path), dtype=np.float32)
    else:
        arr = np.asarray(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return np.clip(arr / 255.0, 0.0, 1.0)


def _flatten_sequence_filter_items(raw: Any) -> list[str]:
    """Flatten a JSON/list/text-loaded allow/deny list into strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, dict):
        items: list[str] = []
        for value in raw.values():
            items.extend(_flatten_sequence_filter_items(value))
        return items
    if isinstance(raw, (list, tuple, set)):
        items: list[str] = []
        for value in raw:
            items.extend(_flatten_sequence_filter_items(value))
        return items
    raise TypeError(f"Unsupported sequence filter value type: {type(raw)!r}")


def _parse_sequence_filter(source: Any) -> tuple[set[str], set[str]]:
    """Parse a sequence allow/deny source into (uids, bare_names)."""
    if source is None:
        return set(), set()

    raw_items: list[str]
    if isinstance(source, str):
        maybe_path = Path(source)
        if maybe_path.exists():
            if maybe_path.suffix.lower() == ".json":
                with open(maybe_path, "r") as f:
                    raw_items = _flatten_sequence_filter_items(json.load(f))
            else:
                raw_items = [
                    line.strip()
                    for line in maybe_path.read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                ]
        else:
            raw_items = [source]
    else:
        raw_items = _flatten_sequence_filter_items(source)

    uids: set[str] = set()
    bare_names: set[str] = set()
    for raw in raw_items:
        item = str(raw).strip().replace("\\", "/").strip("/")
        if not item:
            continue
        if "/" in item:
            uids.add(item)
        else:
            bare_names.add(item)
    return uids, bare_names


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

# All 51 Co3Dv2 categories.
ALL_CATEGORIES: list[str] = [
    "apple", "backpack", "ball", "banana", "baseballbat", "baseballglove",
    "bench", "bicycle", "book", "bottle", "bowl", "broccoli", "cake", "car",
    "carrot", "cellphone", "chair", "couch", "cup", "donut", "frisbee",
    "hairdryer", "handbag", "hotdog", "hydrant", "keyboard", "kite", "laptop",
    "microwave", "motorcycle", "mouse", "orange", "parkingmeter", "pizza",
    "plant", "remote", "sandwich", "skateboard", "stopsign", "suitcase",
    "teddybear", "toaster", "toilet", "toybus", "toyplane", "toytrain",
    "toytruck", "tv", "umbrella", "vase", "wineglass",
]

# Maps subset_name → (set_lists filename suffix, split key inside that file).
_SUBSET_MAP: dict[str, tuple[str, str]] = {
    "fewview_train":    ("fewview_train",    "train"),
    "fewview_dev":      ("fewview_dev",      "val"),
    "fewview_test":     ("fewview_test",     "test"),
    "manyview_dev_0":   ("manyview_dev_0",   "val"),
    "manyview_dev_1":   ("manyview_dev_1",   "val"),
    "manyview_test_0":  ("manyview_test_0",  "test"),
}


class Co3Dv2Adapter(BaseAdapter):
    """Dataset adapter for Co3D version 2.

    Dataset layout (per category)::

        <root>/
            <category>/
                frame_annotations.jgz
                sequence_annotations.jgz
                set_lists/
                    set_lists_fewview_train.json   # {train/val/test: [[seq,frame,path], ...]}
                    set_lists_fewview_dev.json
                    ...
                <sequence_name>/
                    images/
                        frameXXXXXX.jpg
                    depths/
                        frameXXXXXX.jpg.geometric.png   (uint16, depth in scene units)
                    depth_masks/
                        frameXXXXXX.png                 (bool, valid depth)
                    masks/
                        frameXXXXXX.png                 (bool, object foreground)
                    pointcloud.ply

    Sequence ID
    -----------
    A sequence in Co3D corresponds to a video of a single object category.
    The unique ID used by this adapter is ``"<category>/<sequence_name>"``.

    Camera convention
    -----------------
    PyTorch3D cameras (row-vector convention, +x left, +y up) are converted to
    standard OpenCV extrinsics (column-vector convention, +x right, +y down,
    world-to-camera).

    Intrinsics
    ----------
    NDC intrinsics (``ndc_isotropic`` format) are converted to pixel-space
    pinhole intrinsics.

    Depth
    -----
    ``depth = raw_uint16.view(float16).astype(float32) * scale_adjustment``

    The PNG stores float16 IEEE-754 bit patterns inside uint16 pixels.  It is
    not a linearly quantised uint16 depth image and must not be divided by
    65535.

    The depth mask is applied in-place (invalid pixels set to 0.0).
    Depth is in the same units as the camera extrinsics (Co3D scene units).

    Supervision availability
    ------------------------
    When a ``precomputed.npz`` or ``precomputed.h5`` file is present inside
    the sequence directory (i.e. ``<root>/<category>/<sequence>/precomputed.*``),
    track-based 2D/3D supervision is enabled:

    - ``has_depth``      : True  (always)
    - ``has_normals``    : True  (from precomputed file)
    - ``has_tracks``     : True  (from precomputed file)
    - ``has_visibility`` : False (Co3D visibility loss is disabled; sparse
      point-cloud tracks do not provide reliable occlusion supervision)

    If no precomputed file is found:

    - ``has_depth``      : True
    - ``has_normals``    : False
    - ``has_tracks``     : False
    - ``has_visibility`` : False
    """

    dataset_name: str = "co3dv2"

    def __init__(
        self,
        root: str,
        categories: Optional[list[str]] = None,
        subset_name: str = "fewview_train",
        split: str = "train",
        min_frames: int = 2,
        strict: bool = False,
        verbose: bool = True,
        precompute_root: Optional[str] = None,
        track_precompute_root: Optional[str] = None,
        cache_dir: Optional[str] = None,
        min_viewpoint_quality: Optional[float] = None,
        min_pointcloud_quality: Optional[float] = None,
        min_pointcloud_n_points: Optional[int] = None,
        min_valid_depth_ratio: Optional[float] = None,
        min_foreground_ratio: Optional[float] = None,
        quality_probe_frames: int = 3,
        require_pointcloud: bool = False,
        require_precomputed: bool = False,
        max_sequences_per_category: Optional[int] = None,
        sequence_allowlist: Optional[Any] = None,
        sequence_denylist: Optional[Any] = None,
        index_workers: int = 8,
        io_workers: int = 2,
        load_normals: bool = True,
        frame_cache_items: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        root :
            Root directory of the Co3Dv2 dataset.
        categories :
            List of category names to include.  ``None`` → all 51 categories.
        subset_name :
            Which subset to use.  One of:
            ``"fewview_train"``, ``"fewview_dev"``, ``"fewview_test"``,
            ``"manyview_dev_0"``, ``"manyview_dev_1"``, ``"manyview_test_0"``.
        split :
            Split key inside the set_lists file: ``"train"``, ``"val"``,
            or ``"test"``.  Defaults to ``"train"``.
        min_frames :
            Sequences with fewer than this many frames in the chosen split are
            skipped.
        strict :
            If *True*, raise on any error during index construction.
            If *False*, skip broken categories/sequences with a warning.
        verbose :
            Print a summary line after index construction.
        min_viewpoint_quality :
            Drop sequences whose Co3D ``viewpoint_quality_score`` is below this
            threshold.
        min_pointcloud_quality :
            Drop sequences whose sequence-level point-cloud quality score is
            below this threshold.
        min_pointcloud_n_points :
            Require at least this many points in the sequence point cloud.
        min_valid_depth_ratio :
            Require the average valid-depth-mask ratio over sampled probe frames
            to be above this threshold.
        min_foreground_ratio :
            Require the average soft foreground-mask occupancy over sampled
            probe frames to be above this threshold.
        quality_probe_frames :
            Number of frames per sequence used when probing mask quality
            metrics. Must be >= 1.
        require_pointcloud :
            Keep only sequences with an on-disk point cloud.
        require_precomputed :
            Keep only sequences with ``precomputed.npz`` / ``precomputed.h5``.
        max_sequences_per_category :
            If set, keep at most this many sequences per category after
            filtering, preferring higher-quality sequences.
        sequence_allowlist / sequence_denylist :
            Optional list / JSON / text file of sequence names. Entries may be
            bare sequence names (``seq123``) or full UIDs
            (``category/seq123``).
        index_workers :
            Number of threads for parallel category indexing in
            ``_build_index``.  Each category is processed independently.
            Set to 1 to disable threading.  Defaults to 8.
        io_workers :
            Number of per-sample threads used to read RGB/depth/mask frames.
            Keep this small when many DataLoader workers run in parallel.
        load_normals :
            Load normals from Co3D precomputed files. Disable when normal loss
            is not used; track supervision remains available.
        frame_cache_items :
            Per-worker LRU size for decoded RGB/depth/mask frame arrays.
            Local sampling often requests overlapping windows from the same
            sequence; caching removes repeated jpeg/png decode without changing
            sample semantics.  Can also be set with D4RT_CO3D_FRAME_CACHE_ITEMS.
        """
        self.root = Path(root)
        # Check cache first to skip slow root.exists() on remote storage
        _cache_hit = False
        if cache_dir is not None:
            # We need to set basic attributes for _build_index_cache_path
            self.subset_name = subset_name
            self.split = split
            self.categories = list(categories) if categories is not None else list(ALL_CATEGORIES)
            self.min_frames = min_frames
            self.min_viewpoint_quality = _safe_float(min_viewpoint_quality)
            self.min_pointcloud_quality = _safe_float(min_pointcloud_quality)
            self.min_pointcloud_n_points = _safe_int(min_pointcloud_n_points)
            self.min_valid_depth_ratio = _safe_float(min_valid_depth_ratio)
            self.min_foreground_ratio = _safe_float(min_foreground_ratio)
            self.quality_probe_frames = int(quality_probe_frames)
            self.require_pointcloud = bool(require_pointcloud)
            self.require_precomputed = bool(require_precomputed)
            self.max_sequences_per_category = _safe_int(max_sequences_per_category)
            self._sequence_allow_uids, self._sequence_allow_names = _parse_sequence_filter(sequence_allowlist)
            self._sequence_deny_uids, self._sequence_deny_names = _parse_sequence_filter(sequence_denylist)
            _cache_path = self._build_index_cache_path(Path(cache_dir))
            if _cache_path.exists():
                _cache_hit = True
        if not _cache_hit and not self.root.exists():
            raise FileNotFoundError(f"Co3Dv2 root not found: {self.root}")

        if subset_name not in _SUBSET_MAP:
            raise ValueError(
                f"Unknown subset_name '{subset_name}'. "
                f"Valid options: {sorted(_SUBSET_MAP.keys())}"
            )

        self.categories: list[str] = (
            list(categories) if categories is not None else list(ALL_CATEGORIES)
        )
        self.subset_name = subset_name
        self.split = split
        self.min_frames = min_frames
        self.strict = strict
        self.verbose = verbose
        self.precompute_root = Path(precompute_root) if precompute_root else self.root
        self.track_precompute_root = (
            Path(track_precompute_root) if track_precompute_root else None
        )
        self.min_viewpoint_quality = _safe_float(min_viewpoint_quality)
        self.min_pointcloud_quality = _safe_float(min_pointcloud_quality)
        self.min_pointcloud_n_points = _safe_int(min_pointcloud_n_points)
        self.min_valid_depth_ratio = _safe_float(min_valid_depth_ratio)
        self.min_foreground_ratio = _safe_float(min_foreground_ratio)
        self.quality_probe_frames = int(quality_probe_frames)
        self.require_pointcloud = bool(require_pointcloud)
        self.require_precomputed = bool(require_precomputed)
        self.max_sequences_per_category = _safe_int(max_sequences_per_category)
        self.io_workers = max(1, int(io_workers))
        self.load_normals = bool(load_normals)
        if frame_cache_items is None:
            raw_cache_items = os.getenv("D4RT_CO3D_FRAME_CACHE_ITEMS", "").strip()
            frame_cache_items = int(raw_cache_items) if raw_cache_items else 0
        self.frame_cache_items = max(0, int(frame_cache_items))
        self._frame_array_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._frame_array_cache_lock = threading.RLock()
        if self.quality_probe_frames < 1:
            raise ValueError("quality_probe_frames must be >= 1")
        if self.max_sequences_per_category is not None and self.max_sequences_per_category < 1:
            raise ValueError("max_sequences_per_category must be >= 1")
        self._sequence_allow_uids, self._sequence_allow_names = _parse_sequence_filter(sequence_allowlist)
        self._sequence_deny_uids, self._sequence_deny_names = _parse_sequence_filter(sequence_denylist)
        self.index_workers = int(index_workers)
        self._filter_summary: dict[str, Any] = {}

        # category → {(seq_name, frame_number): frame_annotation_dict}
        # LRU-evicted OrderedDict, capped at _FRAME_ANNO_CACHE_MAXSIZE categories.
        # Each decompressed jgz index is ~150-250 MB; keeping a few categories per
        # worker avoids reloading 5-6 s on every category miss.
        self._frame_anno_cache: OrderedDict[str, dict[tuple[str, int], dict]] = OrderedDict()
        # category → {sequence_name: sequence_annotation_dict}
        # sequence_annotations.jgz is tiny (~50 KB), safe to keep all 51 categories.
        self._sequence_anno_cache: dict[str, dict[str, dict]] = {}

        # Optional fast frame-annotation npz cache directory.
        # When available, _get_frame_anno uses these ~0.2 ms npz files instead of
        # the 14 s jgz → JSON → dict pipeline, eliminating the biggest data spike.
        self._frame_anno_npz_dir: Optional[Path] = (
            Path(cache_dir) / "co3dv2_frame_anno" if cache_dir is not None else None
        )
        self._frame_anno_npz_paths: dict[str, Path] = {}
        if self._frame_anno_npz_dir is not None and self._frame_anno_npz_dir.exists():
            for npz_path in self._frame_anno_npz_dir.glob("frame_anno_*.npz"):
                stem = npz_path.stem
                prefix = "frame_anno_"
                if not stem.startswith(prefix):
                    continue
                suffix = stem[len(prefix):]
                category = suffix.rsplit("_", 1)[0]
                # Newer index writers append a content hash
                # (frame_anno_<category>_<hash>.npz), while older code looked
                # for frame_anno_<category>_v1.npz.  Keep one deterministic
                # path per category and let the exact v1 name override below.
                self._frame_anno_npz_paths.setdefault(category, npz_path)
            for category in self.categories:
                exact = self._frame_anno_npz_dir / f"frame_anno_{category}_v1.npz"
                if exact.exists():
                    self._frame_anno_npz_paths[category] = exact
        # category → loaded NpzFile (lazy, kept open; OS page-cache means hot reads ~0.2 ms)
        self._frame_anno_npz: dict[str, Any] = {}

        self._degenerate_precomputed_warn_count = 0
        self._track_failure_marker_warn_count = 0


        if cache_dir is not None:
            from multicamera4rc.datasets.index_cache import load_or_build
            _cache_path = self._build_index_cache_path(Path(cache_dir))
            self._records: list[_Co3DSequenceRecord] = load_or_build(self._build_index, _cache_path)
        else:
            self._records: list[_Co3DSequenceRecord] = self._build_index()
        self._uid_to_record: dict[str, _Co3DSequenceRecord] = {
            r.uid: r for r in self._records
        }

        if not self._records:
            raise RuntimeError(
                f"No valid Co3Dv2 sequences found. "
                f"categories={self.categories}, subset={subset_name}, split={split}"
            )

        if self.verbose:
            print(
                f"[Co3Dv2Adapter] subset={subset_name!r}, split={split!r}, "
                f"categories={len(self.categories)}, "
                f"sequences={len(self._records)}"
            )
            active_filters = self._describe_active_filters()
            if active_filters:
                print(f"[Co3Dv2Adapter] active filters: {active_filters}")

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Builder processes receive adapters through multiprocessing pickle when
        # forkserver/spawn is used.  Locks are not picklable, and decoded frame
        # arrays should start empty in each child process.
        state["_frame_array_cache"] = OrderedDict()
        state["_frame_array_cache_lock"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._frame_array_cache = OrderedDict()
        self._frame_array_cache_lock = threading.RLock()

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def list_sequences(self) -> list[str]:
        return [r.uid for r in self._records]

    def get_sequence_name(self, index: int) -> str:
        return self._records[index].uid

    def get_num_frames(self, sequence_name: str) -> int:
        """Fast path: read directly from in-memory record, no .jgz loading."""
        return self._get_record(sequence_name).num_frames

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        r = self._get_record(sequence_name)
        # Retrieve image size from the first frame's annotation.
        frame_anno = self._get_frame_anno(r.category, r.sequence_name, r.frame_numbers[0])
        H, W = frame_anno["image"]["size"]   # annotation stores [H, W]

        # Check whether precomputed tracks are available for this sequence.
        has_precomputed = False
        base_has_precomputed = False
        if self.precompute_root is not None:
            npz_path = self.precompute_root / sequence_name / "precomputed.npz"
            h5_path  = npz_path.with_suffix(".h5")
            base_has_precomputed = h5_path.exists() or npz_path.exists()
            has_precomputed = base_has_precomputed
        track_has_precomputed = False
        track_failure_marker = False
        npz_path, _track_label = _overlay_npz_and_label(
            precompute_root=self.precompute_root,
            track_precompute_root=self.track_precompute_root,
            sequence_name=sequence_name,
        )
        if npz_path is not None:
            h5_path = npz_path.with_suffix(".h5")
            track_has_precomputed = h5_path.exists() or npz_path.exists()
            track_failure_marker = (
                not track_has_precomputed
                and npz_path.with_name(_TRACK_FAILURE_MARKER_NAME).exists()
            )
            has_precomputed = has_precomputed or track_has_precomputed

        has_tracks = track_has_precomputed or (base_has_precomputed and not track_failure_marker)
        has_normals = has_precomputed and self.load_normals

        return {
            "dataset_name": self.dataset_name,
            "split": self.split,
            "subset_name": self.subset_name,
            "sequence_name": r.uid,
            "category": r.category,
            "sequence_root": str(r.sequence_dir),
            "num_frames": r.num_frames,
            "viewpoint_quality_score": r.viewpoint_quality_score,
            "pointcloud_quality_score": r.pointcloud_quality_score,
            "pointcloud_n_points": r.pointcloud_n_points,
            "valid_depth_ratio": r.valid_depth_ratio,
            "foreground_ratio": r.foreground_ratio,
            "has_pointcloud": r.has_pointcloud,
            "has_precomputed": r.has_precomputed,
            "height": H,
            "width": W,
            "has_depth": True,
            "has_normals": has_normals,
            "has_tracks": has_tracks,
            "has_visibility": False,
            "has_trajs_3d_world": has_tracks,
            "visibility_loss_disabled_reason": "co3dv2_sparse_pointcloud_tracks",
            "extrinsics_convention": "w2c",
            "depth_unit": "co3d_scene_units",
            "depth_encoding": CO3D_DEPTH_ENCODING,
            "depth_scale_adjustment_applied": True,
            "depth_note": CO3D_DEPTH_NOTE,
        }

    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        """Load a clip from a Co3Dv2 sequence.

        Parameters
        ----------
        sequence_name :
            Unique sequence ID in ``"<category>/<sequence_name>"`` format.
        frame_indices :
            Positions into the sequence's sorted frame list.  Must be within
            ``[0, num_frames)``.

        Returns
        -------
        UnifiedClip
            Unified intermediate representation.
            If a ``precomputed.npz`` / ``.h5`` file is found next to the
            sequence directory, ``normals``, ``trajs_2d``, ``trajs_3d_world``,
            ``valids``, and ``visibs`` are populated from it and
            ``metadata["has_tracks"]`` is set to ``True``.
            Otherwise these fields are ``None`` and ``has_tracks`` is ``False``.
        """
        r = self._get_record(sequence_name)
        # Clamp frame_indices to valid range for co3dv2 (precomputed data may have fewer frames)
        frame_indices = [max(0, min(i, r.num_frames - 1)) for i in frame_indices]
        self._check_indices(frame_indices, r.num_frames, sequence_name)

        def _load_one(idx):
            fn   = r.frame_numbers[idx]
            anno = self._get_frame_anno(r.category, r.sequence_name, fn)
            img  = self._load_rgb_cached(self.root / anno["image"]["path"])
            dep  = self._load_depth_cached(
                self.root / anno["depth"]["path"],
                anno["depth"]["scale_adjustment"],
            )
            # Apply depth_mask (cross-view consistent pixels per official Co3D docs)
            dep_mask_path = self.root / anno["depth"]["mask_path"]
            if dep_mask_path.exists():
                dep[~self._load_depth_mask_cached(dep_mask_path)] = 0.0
            H, W = anno["image"]["size"]
            K = _ndc_to_pinhole(
                anno["viewpoint"]["focal_length"],
                anno["viewpoint"]["principal_point"],
                anno["image"]["size"],
            )
            E = _p3d_to_opencv_extrinsics(
                anno["viewpoint"]["R"],
                anno["viewpoint"]["T"],
            )
            return img, dep, (H, W), K, E, str(self.root / anno["image"]["path"]), fn

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(frame_indices), self.io_workers)) as ex:
            rows = list(ex.map(_load_one, frame_indices))
        (images, depths, image_sizes,
         intrinsics_list, extrinsics_list,
         frame_paths, frame_numbers_clip) = map(list, zip(*rows))

        intrinsics = np.stack(intrinsics_list, axis=0)   # (T, 3, 3) float32
        extrinsics = np.stack(extrinsics_list, axis=0)   # (T, 4, 4) float32

        # Load precomputed normals / tracks if available
        normals_out, trajs_2d_out, trajs_3d_out, valids_out, visibs_out = \
            None, None, None, None, None
        has_normals_out, has_tracks_out = False, False
        precomputed_ref_frame = None
        precomputed_num_points = None
        precomputed_track_source = None
        precomputed_visibility_policy = None
        precomputed_has_real_visibility = False
        precomputed_track_stats = _summarize_precomputed_track_validity(None)
        precomputed_tracks_degenerate = False
        base_cache = None
        if self.precompute_root is not None:
            base_npz_path = self.precompute_root / sequence_name / "precomputed.npz"
            # Skip "normals" on the first pass — it is the largest array in the
            # npz (~2 s zlib decompress for a 50 MB file) and only needed if
            # base_cache is actually selected for normal supervision.  We reload
            # it below only when we confirm base_cache will supply normals.
            base_cache = load_precomputed_fast(
                base_npz_path, frame_indices, skip_keys={"normals"}
            )

        track_cache = None
        track_failure_marker = False
        track_npz_path, track_cache_label = _overlay_npz_and_label(
            precompute_root=self.precompute_root,
            track_precompute_root=self.track_precompute_root,
            sequence_name=sequence_name,
        )
        if track_npz_path is not None:
            track_cache = load_precomputed_fast(track_npz_path, frame_indices)
            track_failure_marker = (
                track_cache is None
                and track_npz_path.with_name(_TRACK_FAILURE_MARKER_NAME).exists()
            )

        selected_track_cache = None
        selected_track_cache_label = None
        if track_cache is not None:
            selected_track_cache = track_cache
            selected_track_cache_label = track_cache_label
        elif not track_failure_marker:
            selected_track_cache = base_cache
            selected_track_cache_label = "precompute_root"

        if track_failure_marker and self.verbose and self._track_failure_marker_warn_count < 8:
            self._track_failure_marker_warn_count += 1
            print(
                "[Co3Dv2Adapter] overlay failure marker blocks base precomputed tracks "
                f"for {sequence_name}. Falling back to depth-only supervision until "
                "a valid overlay precomputed.npz exists."
            )

        if base_cache is not None and self.load_normals:
            # Load normals lazily: only when the base precomputed file is
            # actually the selected source (i.e. no track_cache overlay or
            # track_failure_marker).  Normals are the largest array in the npz
            # (~6 s of zlib decompress for a 50 MB file) so we skip them when
            # they won't be used.
            if selected_track_cache is base_cache or (
                selected_track_cache is None and not track_failure_marker
            ):
                base_npz_path = self.precompute_root / sequence_name / "precomputed.npz"
                normals_npz = load_precomputed_fast(
                    base_npz_path, frame_indices, skip_keys={
                        k for k in ("trajs_2d", "trajs_3d_world", "valids", "visibs",
                                    "ref_frame", "num_points", "num_frames", "track_source")
                    }
                )
                if normals_npz is not None and "normals" in normals_npz:
                    T_clip = len(frame_indices)
                    normals_raw = normals_npz["normals"]
                    normals_out = [
                        normals_raw[i].astype(np.float32) for i in range(T_clip)
                    ]
                    has_normals_out = True
                elif base_npz_path.exists():
                    # A generated tracks-only precomputed.h5 shadows the original
                    # npz for fast track reads. If a caller explicitly asks for
                    # normals, fall back to the npz payload rather than silently
                    # losing normal supervision.
                    with np.load(base_npz_path, allow_pickle=True) as raw_npz:
                        if "normals" in raw_npz:
                            normals_raw = raw_npz["normals"][np.array(frame_indices)]
                            normals_out = [
                                normals_raw[i].astype(np.float32)
                                for i in range(len(frame_indices))
                            ]
                            has_normals_out = True

        if selected_track_cache is not None:
            has_track_arrays = all(
                key in selected_track_cache for key in _TRACK_ARRAY_KEYS
            )
            precomputed_track_stats = _summarize_precomputed_track_validity(
                selected_track_cache.get("valids") if has_track_arrays else None
            )
            if has_track_arrays and _precomputed_tracks_look_temporal(
                selected_track_cache.get("valids")
            ):
                trajs_2d_out = selected_track_cache["trajs_2d"]
                trajs_3d_out = selected_track_cache["trajs_3d_world"]
                valids_out = selected_track_cache["valids"]
                visibs_out = selected_track_cache["visibs"]
                has_tracks_out = True
                precomputed_has_real_visibility = _cache_has_real_visibility(selected_track_cache)
                precomputed_visibility_policy = _cache_str(
                    selected_track_cache,
                    "visibility_policy",
                    default=None,
                )
            elif has_track_arrays:
                precomputed_tracks_degenerate = True
                if self.verbose and self._degenerate_precomputed_warn_count < 8:
                    self._degenerate_precomputed_warn_count += 1
                    print(
                        "[Co3Dv2Adapter] ignoring degenerate precomputed tracks "
                        f"for {sequence_name} from {selected_track_cache_label}: "
                        f"mean_visible_frames={precomputed_track_stats['mean_visible_frames']:.3f}, "
                        f"tracks_ge_2_ratio={precomputed_track_stats['tracks_ge_2_ratio']:.3f}. "
                        "Falling back to depth-only supervision for this clip."
                    )

            if "ref_frame" in selected_track_cache:
                precomputed_ref_frame = int(selected_track_cache["ref_frame"])
            if "num_points" in selected_track_cache:
                precomputed_num_points = int(selected_track_cache["num_points"])
            if "track_source" in selected_track_cache:
                src = selected_track_cache["track_source"]
                if isinstance(src, bytes):
                    precomputed_track_source = src.decode("utf-8", errors="replace")
                elif isinstance(src, np.ndarray) and src.ndim == 0:
                    precomputed_track_source = str(src.item())
                else:
                    precomputed_track_source = str(src)


        return UnifiedClip(
            dataset_name=self.dataset_name,
            sequence_name=sequence_name,
            frame_paths=frame_paths,
            images=images,
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
                "category": r.category,
                "split": self.split,
                "subset_name": self.subset_name,
                "sequence_root": str(r.sequence_dir),
                "frame_indices": list(frame_indices),
                "frame_numbers": frame_numbers_clip,
                "num_frames_in_sequence": r.num_frames,
                "viewpoint_quality_score": r.viewpoint_quality_score,
                "pointcloud_quality_score": r.pointcloud_quality_score,
                "pointcloud_n_points": r.pointcloud_n_points,
                "valid_depth_ratio": r.valid_depth_ratio,
                "foreground_ratio": r.foreground_ratio,
                "has_pointcloud": r.has_pointcloud,
                "has_precomputed": r.has_precomputed,
                "raw_image_hw": list(image_sizes[0]) if image_sizes else None,
                "has_depth": True,
                "has_normals": has_normals_out,
                "has_tracks": has_tracks_out,
                "has_visibility": False,
                "has_trajs_3d_world": has_tracks_out,
                "visibility_loss_disabled_reason": "co3dv2_sparse_pointcloud_tracks",
                "normal_convention": "camera_space_opencv_towards_camera" if has_normals_out else None,
                "normal_supervision_compatible": has_normals_out,
                "precomputed_mean_visible_frames": precomputed_track_stats["mean_visible_frames"],
                "precomputed_tracks_ge_2_ratio": precomputed_track_stats["tracks_ge_2_ratio"],
                "precomputed_tracks_ge_4_ratio": precomputed_track_stats["tracks_ge_4_ratio"],
                "precomputed_tracks_degenerate": precomputed_tracks_degenerate,
                "precomputed_track_failure_marker": track_failure_marker,
                "precomputed_track_cache_layer": selected_track_cache_label if selected_track_cache is not None else None,
                "precomputed_has_real_visibility": bool(precomputed_has_real_visibility),
                "precomputed_visibility_policy": precomputed_visibility_policy,
                "pose_convention": "w2c",
                "extrinsics_convention": "w2c",
                "intrinsics_convention": "pinhole",
                "precomputed_ref_frame": precomputed_ref_frame,
                "precomputed_num_points": precomputed_num_points,
                "precomputed_track_source": precomputed_track_source,
                "depth_unit": "co3d_scene_units",
                "depth_encoding": CO3D_DEPTH_ENCODING,
                "depth_scale_adjustment_applied": True,
                "depth_note": CO3D_DEPTH_NOTE,
            },
        )

    def _cache_array(self, key: tuple, loader, *, copy_on_return: bool) -> np.ndarray:
        """Return decoded frame data from this adapter process' bounded LRU."""
        if self.frame_cache_items <= 0:
            arr = loader()
            return arr.copy() if copy_on_return else arr

        with self._frame_array_cache_lock:
            cached = self._frame_array_cache.get(key)
            if cached is not None:
                self._frame_array_cache.move_to_end(key)
                return cached.copy() if copy_on_return else cached

        arr = loader()
        with self._frame_array_cache_lock:
            cached = self._frame_array_cache.get(key)
            if cached is not None:
                self._frame_array_cache.move_to_end(key)
                return cached.copy() if copy_on_return else cached
            self._frame_array_cache[key] = arr
            while len(self._frame_array_cache) > self.frame_cache_items:
                self._frame_array_cache.popitem(last=False)
            return arr.copy() if copy_on_return else arr

    def _load_rgb_cached(self, path: Path) -> np.ndarray:
        # RGB frames are not mutated downstream; returning the cached array is safe.
        return self._cache_array(("rgb", str(path)), lambda: _load_rgb(path), copy_on_return=False)

    def _load_depth_cached(self, path: Path, scale_adjustment: float) -> np.ndarray:
        # Depth is masked in-place per sample, so each caller must get a copy.
        key = ("depth", str(path), float(scale_adjustment))
        return self._cache_array(
            key,
            lambda: _load_depth(path, scale_adjustment),
            copy_on_return=True,
        )

    def _load_depth_mask_cached(self, path: Path) -> np.ndarray:
        return self._cache_array(
            ("depth_mask", str(path)),
            lambda: _load_depth_mask(path),
            copy_on_return=False,
        )

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        """Run consistency checks on a sequence.

        Checks
        ------
        1. Image, depth, and depth-mask files exist for the first 5 frames.
        2. Intrinsics shape (T, 3, 3), finite, positive focal lengths.
        3. Extrinsics shape (T, 4, 4), finite, rotation near-orthonormal,
           last row == [0, 0, 0, 1].
        4. Depth values finite; at least some positive values per frame.
        5. Round-trip reprojection: unproject a valid depth pixel, re-project
           back, check error < 1 px.

        Returns
        -------
        dict with ``dataset_name``, ``sequence_name``, ``ok`` (bool),
        ``messages`` (list[str]).
        """
        r = self._get_record(sequence_name)
        msgs: list[str] = []
        ok = True

        # ── 1. File existence (first 5 frames) ──────────────────────────
        probe_count = min(5, r.num_frames)
        for i in range(probe_count):
            fn = r.frame_numbers[i]
            try:
                anno = self._get_frame_anno(r.category, r.sequence_name, fn)
            except KeyError as exc:
                ok = False
                msgs.append(f"frame {fn}: annotation lookup failed: {exc}")
                continue

            for key, subpath in [
                ("image", anno["image"]["path"]),
                ("depth", anno["depth"]["path"]),
                ("depth_mask", anno["depth"]["mask_path"]),
            ]:
                p = self.root / subpath
                if not p.exists():
                    ok = False
                    msgs.append(f"frame {fn}: missing {key} file: {p}")

        if not ok:
            return {"dataset_name": self.dataset_name, "sequence_name": sequence_name,
                    "ok": False, "messages": msgs}

        # ── Load probe clip ──────────────────────────────────────────────
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

        # ── 2. Intrinsics ───────────────────────────────────────────────
        if clip.intrinsics.shape != (T, 3, 3):
            ok = False
            msgs.append(f"intrinsics shape {clip.intrinsics.shape} != ({T},3,3)")
        elif not np.isfinite(clip.intrinsics).all():
            ok = False
            msgs.append("intrinsics contains non-finite values")
        else:
            fx = clip.intrinsics[:, 0, 0]
            fy = clip.intrinsics[:, 1, 1]
            if not (np.all(fx > 0) and np.all(fy > 0)):
                ok = False
                msgs.append(f"non-positive focal lengths: fx={fx}, fy={fy}")

        # ── 3. Extrinsics ───────────────────────────────────────────────
        if clip.extrinsics.shape != (T, 4, 4):
            ok = False
            msgs.append(f"extrinsics shape {clip.extrinsics.shape} != ({T},4,4)")
        elif not np.isfinite(clip.extrinsics).all():
            ok = False
            msgs.append("extrinsics contains non-finite values")
        else:
            # Last row == [0,0,0,1]
            bottom = clip.extrinsics[:, 3, :]
            if not np.allclose(bottom, [0, 0, 0, 1], atol=1e-5):
                ok = False
                msgs.append("extrinsics last row != [0,0,0,1]")
            # Rotation orthonormality
            R = clip.extrinsics[:, :3, :3].astype(np.float64)
            RRt = R @ np.transpose(R, (0, 2, 1))
            err = float(np.linalg.norm(RRt - np.eye(3)[None], axis=(1, 2)).max())
            if err > 1e-4:
                ok = False
                msgs.append(f"rotation matrices non-orthonormal: max ||RR^T-I||_F={err:.2e}")

        # ── 4. Depth ────────────────────────────────────────────────────
        for t in range(T):
            dep = clip.depths[t]
            if not np.isfinite(dep).all():
                ok = False
                msgs.append(f"depth[{t}] contains non-finite values")

        # ── 5. Reprojection ─────────────────────────────────────────────
        for t in range(T):
            dep = clip.depths[t]
            K = clip.intrinsics[t].astype(np.float64)
            E = clip.extrinsics[t].astype(np.float64)

            ys, xs = np.where(dep > 0)
            if ys.size == 0:
                msgs.append(f"reproj check skipped frame {t}: no valid depth")
                continue

            # Sample centre-ish pixel
            mid = ys.size // 2
            u, v = int(xs[mid]), int(ys[mid])
            d = float(dep[v, u])

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            xc = (u - cx) / fx * d
            yc = (v - cy) / fy * d
            p_cam = np.array([xc, yc, d, 1.0])

            try:
                p_world = np.linalg.inv(E) @ p_cam
            except np.linalg.LinAlgError:
                ok = False
                msgs.append(f"extrinsics[{t}] singular")
                continue

            p_cam2 = E @ p_world
            p_2d = K @ p_cam2[:3]
            u2 = p_2d[0] / p_2d[2]
            v2 = p_2d[1] / p_2d[2]
            reproj_err = float(np.hypot(u2 - u, v2 - v))
            if reproj_err > 1.0:
                ok = False
                msgs.append(
                    f"reproj error frame {t}: ({u2:.2f},{v2:.2f}) vs ({u},{v}), "
                    f"err={reproj_err:.4f}px"
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

    def get_filter_summary(self) -> dict[str, Any]:
        """Return a shallow copy of the latest index filter summary."""
        return json.loads(json.dumps(self._filter_summary))

    # ------------------------------------------------------------------
    # Filter helpers
    # ------------------------------------------------------------------

    def _describe_active_filters(self) -> str:
        parts: list[str] = []
        if self.min_viewpoint_quality is not None:
            parts.append(f"viewpoint>={self.min_viewpoint_quality:g}")
        if self.min_pointcloud_quality is not None:
            parts.append(f"pointcloud_quality>={self.min_pointcloud_quality:g}")
        if self.min_pointcloud_n_points is not None:
            parts.append(f"pointcloud_n_points>={self.min_pointcloud_n_points}")
        if self.min_valid_depth_ratio is not None:
            parts.append(f"valid_depth_ratio>={self.min_valid_depth_ratio:g}")
        if self.min_foreground_ratio is not None:
            parts.append(f"foreground_ratio>={self.min_foreground_ratio:g}")
        if self.require_pointcloud:
            parts.append("require_pointcloud")
        if self.require_precomputed:
            parts.append("require_precomputed")
        if self.max_sequences_per_category is not None:
            parts.append(f"max_per_category={self.max_sequences_per_category}")
        if self._sequence_allow_uids or self._sequence_allow_names:
            parts.append("allowlist")
        if self._sequence_deny_uids or self._sequence_deny_names:
            parts.append("denylist")
        return ", ".join(parts)

    def _build_index_cache_path(self, cache_dir: Path) -> Path:
        key_payload = {
            "dataset_name": self.dataset_name,
            # Cached records embed absolute sequence_dir paths under ``root``,
            # so the key must include ``root``.  Without it, a cache built
            # against one mount (e.g. /data2) is silently served to a run
            # pointing at another (e.g. /data4), yielding dead paths.
            "root": str(Path(self.root).resolve()),
            "split": self.split,
            "subset_name": self.subset_name,
            "categories": self.categories,
            "min_frames": self.min_frames,
            "min_viewpoint_quality": self.min_viewpoint_quality,
            "min_pointcloud_quality": self.min_pointcloud_quality,
            "min_pointcloud_n_points": self.min_pointcloud_n_points,
            "min_valid_depth_ratio": self.min_valid_depth_ratio,
            "min_foreground_ratio": self.min_foreground_ratio,
            "quality_probe_frames": self.quality_probe_frames,
            "require_pointcloud": self.require_pointcloud,
            "require_precomputed": self.require_precomputed,
            "max_sequences_per_category": self.max_sequences_per_category,
            "allow_uids": sorted(self._sequence_allow_uids),
            "allow_names": sorted(self._sequence_allow_names),
            "deny_uids": sorted(self._sequence_deny_uids),
            "deny_names": sorted(self._sequence_deny_names),
        }
        digest = hashlib.sha1(
            json.dumps(key_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        filename = f"{self.dataset_name}_{self.split}_{self.subset_name}_{digest}.pkl"
        return cache_dir / filename

    def _has_precomputed_for_sequence(self, sequence_uid: str) -> bool:
        npz_path = self.precompute_root / sequence_uid / "precomputed.npz"
        h5_path = npz_path.with_suffix(".h5")
        return npz_path.exists() or h5_path.exists()

    def _sequence_is_allowed(self, sequence_uid: str, sequence_name: str) -> bool:
        if self._sequence_allow_uids or self._sequence_allow_names:
            if sequence_uid not in self._sequence_allow_uids and sequence_name not in self._sequence_allow_names:
                return False
        if sequence_uid in self._sequence_deny_uids or sequence_name in self._sequence_deny_names:
            return False
        return True

    def _probe_frame_numbers(self, frame_numbers: list[int]) -> list[int]:
        if len(frame_numbers) <= self.quality_probe_frames:
            return list(frame_numbers)
        idxs = np.linspace(
            0,
            len(frame_numbers) - 1,
            num=self.quality_probe_frames,
            dtype=np.int64,
        )
        unique_idxs = sorted(set(int(i) for i in idxs.tolist()))
        return [frame_numbers[i] for i in unique_idxs]

    def _compute_sequence_mask_ratios(
        self,
        category: str,
        sequence_name: str,
        frame_numbers: list[int],
    ) -> tuple[Optional[float], Optional[float]]:
        if self.min_valid_depth_ratio is None and self.min_foreground_ratio is None:
            return None, None

        depth_ratios: list[float] = []
        fg_ratios: list[float] = []
        for frame_number in self._probe_frame_numbers(frame_numbers):
            anno = self._get_frame_anno(category, sequence_name, frame_number)

            if self.min_valid_depth_ratio is not None:
                depth_info = anno.get("depth")
                depth_mask_rel = depth_info.get("mask_path") if depth_info else None
                if depth_mask_rel:
                    depth_mask_path = self.root / depth_mask_rel
                    depth_mask = _load_depth_mask(depth_mask_path) if depth_mask_path.exists() else None
                    depth_ratios.append(float(depth_mask.mean()) if depth_mask is not None else 0.0)
                else:
                    depth_ratios.append(0.0)

            if self.min_foreground_ratio is not None:
                mask_info = anno.get("mask")
                mask_rel = mask_info.get("path") if mask_info else None
                if mask_rel:
                    mask_path = self.root / mask_rel
                    fg_mask = _load_soft_mask(mask_path) if mask_path.exists() else None
                    fg_ratios.append(float(fg_mask.mean()) if fg_mask is not None else 0.0)
                else:
                    fg_ratios.append(0.0)

        depth_ratio = float(np.mean(depth_ratios)) if depth_ratios else None
        fg_ratio = float(np.mean(fg_ratios)) if fg_ratios else None
        return depth_ratio, fg_ratio

    def _sequence_sort_key(self, record: _Co3DSequenceRecord) -> tuple:
        neg_inf = float("-inf")
        return (
            record.viewpoint_quality_score if record.viewpoint_quality_score is not None else neg_inf,
            record.pointcloud_quality_score if record.pointcloud_quality_score is not None else neg_inf,
            record.valid_depth_ratio if record.valid_depth_ratio is not None else neg_inf,
            record.foreground_ratio if record.foreground_ratio is not None else neg_inf,
            float(record.pointcloud_n_points) if record.pointcloud_n_points is not None else neg_inf,
            float(record.num_frames),
            record.uid,
        )

    def _extract_sequence_metrics(
        self,
        category: str,
        sequence_name: str,
        sequence_dir: Path,
        frame_numbers: list[int],
    ) -> dict[str, Any]:
        seq_anno = self._get_sequence_anno(category, sequence_name)
        pointcloud_anno = seq_anno.get("point_cloud") if seq_anno else None
        pointcloud_path_rel = pointcloud_anno.get("path") if pointcloud_anno else None
        pointcloud_path = (
            self.root / pointcloud_path_rel
            if pointcloud_path_rel
            else sequence_dir / "pointcloud.ply"
        )

        valid_depth_ratio, foreground_ratio = self._compute_sequence_mask_ratios(
            category=category,
            sequence_name=sequence_name,
            frame_numbers=frame_numbers,
        )
        sequence_uid = f"{category}/{sequence_name}"
        return {
            "viewpoint_quality_score": _safe_float(
                seq_anno.get("viewpoint_quality_score") if seq_anno else None
            ),
            "pointcloud_quality_score": _safe_float(
                pointcloud_anno.get("quality_score") if pointcloud_anno else None
            ),
            "pointcloud_n_points": _safe_int(
                pointcloud_anno.get("n_points") if pointcloud_anno else None
            ),
            "valid_depth_ratio": valid_depth_ratio,
            "foreground_ratio": foreground_ratio,
            "has_pointcloud": bool(pointcloud_path.exists()),
            "has_precomputed": self._has_precomputed_for_sequence(sequence_uid),
        }

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> list[_Co3DSequenceRecord]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        file_suffix, split_key = _SUBSET_MAP[self.subset_name]
        records: list[_Co3DSequenceRecord] = []
        skipped_cats: list[str] = []
        dropped_by_reason: dict[str, int] = {}
        per_category: dict[str, dict[str, Any]] = {}
        total_seen = 0

        # Phase 1: collect valid (cat, cat_dir, set_lists_path) tuples.
        # Path validation is cheap: only two stat() calls per category.
        category_tasks: list[tuple[str, Path, Path, str]] = []
        for cat in self.categories:
            cat_dir = self.root / cat
            if not cat_dir.is_dir():
                msg = f"category directory not found: {cat_dir}"
                if self.strict:
                    raise FileNotFoundError(msg)
                skipped_cats.append(f"{cat}: {msg}")
                continue

            set_lists_path = cat_dir / "set_lists" / f"set_lists_{file_suffix}.json"
            if not set_lists_path.exists():
                msg = f"set_lists file not found: {set_lists_path}"
                if self.strict:
                    raise FileNotFoundError(msg)
                skipped_cats.append(f"{cat}: {msg}")
                continue

            category_tasks.append((cat, cat_dir, set_lists_path, split_key))

        # Phase 2: index each category (heavy I/O — parallelise).
        def _process_category(
            args: tuple[str, Path, Path, str],
        ) -> tuple[str, list[_Co3DSequenceRecord], dict[str, Any], Optional[Exception]]:
            cat, cat_dir, slp, sk = args
            try:
                cat_records, cat_stats = self._index_category(cat, cat_dir, slp, sk)
                return cat, cat_records, cat_stats, None
            except Exception as exc:
                return cat, [], {}, exc

        n_workers = min(self.index_workers, len(category_tasks))
        # Collect results into a list first, then aggregate — avoids
        # mutating shared locals (total_seen, dropped_by_reason) from threads.
        category_results: list[tuple[str, list[_Co3DSequenceRecord], dict[str, Any], Optional[Exception]]] = []

        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(_process_category, task): task[0]
                    for task in category_tasks
                }
                for fut in as_completed(futures):
                    category_results.append(fut.result())
        else:
            for task in category_tasks:
                category_results.append(_process_category(task))

        # Phase 3: aggregate results (single-threaded, safe).
        for cat, cat_records, cat_stats, exc in category_results:
            if exc is not None:
                if self.strict:
                    raise exc
                skipped_cats.append(f"{cat}: {exc}")
                if self.verbose:
                    print(f"[Co3Dv2Adapter][WARN] skip category {cat}: {exc}")
            else:
                records.extend(cat_records)
                per_category[cat] = cat_stats
                total_seen += int(cat_stats["seen"])
                for reason, count in cat_stats["dropped"].items():
                    dropped_by_reason[reason] = dropped_by_reason.get(reason, 0) + int(count)

        if self.verbose and skipped_cats:
            print(
                f"[Co3Dv2Adapter] skipped {len(skipped_cats)} categories "
                "(non-strict mode)"
            )

        self._filter_summary = {
            "subset_name": self.subset_name,
            "split": self.split,
            "total_sequences_seen": total_seen,
            "total_sequences_kept": len(records),
            "dropped_by_reason": dropped_by_reason,
            "per_category": per_category,
            "skipped_categories": skipped_cats,
        }

        return records

    def _index_category(
        self,
        cat: str,
        cat_dir: Path,
        set_lists_path: Path,
        split_key: str,
    ) -> tuple[list[_Co3DSequenceRecord], dict[str, Any]]:
        with open(set_lists_path, "r") as f:
            set_lists = json.load(f)

        if split_key not in set_lists:
            raise KeyError(
                f"split key '{split_key}' not found in {set_lists_path}. "
                f"Available keys: {list(set_lists.keys())}"
            )

        # Group (sequence_name, frame_number) entries by sequence.
        seq_to_frames: dict[str, list[int]] = {}
        for seq_name, frame_number, _img_path in set_lists[split_key]:
            seq_to_frames.setdefault(seq_name, []).append(int(frame_number))

        records: list[_Co3DSequenceRecord] = []
        dropped: dict[str, int] = {}

        def _drop(reason: str) -> None:
            dropped[reason] = dropped.get(reason, 0) + 1

        for seq_name, frame_numbers in seq_to_frames.items():
            frame_numbers_sorted = sorted(set(frame_numbers))
            sequence_uid = f"{cat}/{seq_name}"
            if len(frame_numbers_sorted) < self.min_frames:
                _drop("too_few_frames")
                continue

            if not self._sequence_is_allowed(sequence_uid, seq_name):
                _drop("blocked_by_name_filter")
                continue

            seq_dir = cat_dir / seq_name
            if not seq_dir.is_dir():
                if self.strict:
                    raise FileNotFoundError(
                        f"sequence directory not found: {seq_dir}"
                    )
                _drop("missing_sequence_dir")
                continue

            metrics = self._extract_sequence_metrics(
                category=cat,
                sequence_name=seq_name,
                sequence_dir=seq_dir,
                frame_numbers=frame_numbers_sorted,
            )

            if (
                self.min_viewpoint_quality is not None
                and (
                    metrics["viewpoint_quality_score"] is None
                    or metrics["viewpoint_quality_score"] < self.min_viewpoint_quality
                )
            ):
                _drop("low_viewpoint_quality")
                continue

            if (
                self.min_pointcloud_quality is not None
                and (
                    metrics["pointcloud_quality_score"] is None
                    or metrics["pointcloud_quality_score"] < self.min_pointcloud_quality
                )
            ):
                _drop("low_pointcloud_quality")
                continue

            if (
                self.min_pointcloud_n_points is not None
                and (
                    metrics["pointcloud_n_points"] is None
                    or metrics["pointcloud_n_points"] < self.min_pointcloud_n_points
                )
            ):
                _drop("insufficient_pointcloud_points")
                continue

            if (
                self.min_valid_depth_ratio is not None
                and (
                    metrics["valid_depth_ratio"] is None
                    or metrics["valid_depth_ratio"] < self.min_valid_depth_ratio
                )
            ):
                _drop("low_valid_depth_ratio")
                continue

            if (
                self.min_foreground_ratio is not None
                and (
                    metrics["foreground_ratio"] is None
                    or metrics["foreground_ratio"] < self.min_foreground_ratio
                )
            ):
                _drop("low_foreground_ratio")
                continue

            if self.require_pointcloud and not metrics["has_pointcloud"]:
                _drop("missing_pointcloud")
                continue

            if self.require_precomputed and not metrics["has_precomputed"]:
                _drop("missing_precomputed")
                continue

            records.append(
                _Co3DSequenceRecord(
                    category=cat,
                    sequence_name=seq_name,
                    sequence_dir=seq_dir,
                    frame_numbers=frame_numbers_sorted,
                    viewpoint_quality_score=metrics["viewpoint_quality_score"],
                    pointcloud_quality_score=metrics["pointcloud_quality_score"],
                    pointcloud_n_points=metrics["pointcloud_n_points"],
                    valid_depth_ratio=metrics["valid_depth_ratio"],
                    foreground_ratio=metrics["foreground_ratio"],
                    has_pointcloud=metrics["has_pointcloud"],
                    has_precomputed=metrics["has_precomputed"],
                )
            )

        if self.max_sequences_per_category is not None and len(records) > self.max_sequences_per_category:
            records = sorted(records, key=self._sequence_sort_key, reverse=True)
            dropped["category_cap"] = dropped.get("category_cap", 0) + (
                len(records) - self.max_sequences_per_category
            )
            records = records[: self.max_sequences_per_category]

        stats = {
            "seen": len(seq_to_frames),
            "kept": len(records),
            "dropped": dropped,
        }
        return records, stats

    # ------------------------------------------------------------------
    # Frame annotation lazy loader
    # ------------------------------------------------------------------

    def _ensure_sequence_anno_loaded(self, category: str) -> None:
        """Load and cache sequence_annotations.jgz for *category* if needed."""
        if category in self._sequence_anno_cache:
            return

        path = self.root / category / "sequence_annotations.jgz"
        if not path.exists():
            raise FileNotFoundError(f"sequence_annotations.jgz not found: {path}")

        annotations = _load_jgz(path)
        index: dict[str, dict] = {}
        for entry in annotations:
            seq_name = str(entry["sequence_name"])
            index[seq_name] = entry
        self._sequence_anno_cache[category] = index

    def _get_sequence_anno(self, category: str, sequence_name: str) -> Optional[dict]:
        self._ensure_sequence_anno_loaded(category)
        return self._sequence_anno_cache[category].get(sequence_name)

    def _ensure_frame_anno_loaded(self, category: str) -> None:
        """Load and cache frame_annotations for *category* if not already done.

        Fast path (≈23 ms one-time, ~0.01 ms per query): loads a pre-built
        stacked .npz file from ``_frame_anno_npz_dir`` into RAM.  All viewpoint
        and depth-scale arrays land in contiguous numpy memory, so per-frame
        lookups use searchsorted rather than a 163 K-entry Python dict.
        Falls back to the LRU-evicted jgz path (~14 s) if no pre-built npz exists.
        """
        if category in self._frame_anno_cache:
            self._frame_anno_cache.move_to_end(category)
            return

        # ── Fast path: pre-built stacked npz ──────────────────────────────
        if self._frame_anno_npz_dir is not None:
            npz_path = self._frame_anno_npz_paths.get(category)
            if npz_path is not None and npz_path.exists():
                if category not in self._frame_anno_npz:
                    # np.load without mmap: all arrays land in RAM (~23 ms for
                    # apple with 163 K frames, 13 MB).  Random index then costs
                    # ~0.01 ms vs ~15 ms with mmap random access.
                    z = np.load(npz_path)
                    seq_names = z["seq_names"]
                    offsets   = z["offsets"]
                    self._frame_anno_npz[category] = {
                        "seq_to_range": {
                            str(name): (int(offsets[i]), int(offsets[i + 1]))
                            for i, name in enumerate(seq_names)
                        },
                        "fns":       z["fns"],
                        "R":         z["R"],
                        "T":         z["T"],
                        "focal":     z["focal"],
                        "pp":        z["pp"],
                        "imgsz":     z["imgsz"],
                        "dscale":    z["dscale"],
                        "img_stems": z["img_stems"] if "img_stems" in z else None,
                    }
                # Sentinel in LRU cache so move_to_end / eviction still works.
                self._frame_anno_cache[category] = {"__npz__": True}
                if len(self._frame_anno_cache) > _FRAME_ANNO_CACHE_MAXSIZE:
                    evicted_cat = next(iter(self._frame_anno_cache))
                    self._frame_anno_cache.popitem(last=False)
                    self._frame_anno_npz.pop(evicted_cat, None)
                return

        # ── Slow path: jgz → JSON → dict (LRU-evicted) ────────────────────
        path = self.root / category / "frame_annotations.jgz"
        if not path.exists():
            raise FileNotFoundError(f"frame_annotations.jgz not found: {path}")

        annotations = _load_jgz(path)
        index: dict[tuple[str, int], dict] = {}
        for entry in annotations:
            key = (entry["sequence_name"], int(entry["frame_number"]))
            index[key] = entry

        self._frame_anno_cache[category] = index
        if len(self._frame_anno_cache) > _FRAME_ANNO_CACHE_MAXSIZE:
            self._frame_anno_cache.popitem(last=False)

    def _get_frame_anno(
        self, category: str, sequence_name: str, frame_number: int
    ) -> dict:
        self._ensure_frame_anno_loaded(category)
        self._frame_anno_cache.move_to_end(category)
        entry = self._frame_anno_cache[category]

        # ── Fast path: npz-backed ──────────────────────────────────────────
        if entry.get("__npz__"):
            return self._lookup_frame_anno_from_npz(category, sequence_name, frame_number)

        # ── Slow path: plain dict from jgz ─────────────────────────────────
        key = (sequence_name, int(frame_number))
        if key not in entry:
            raise KeyError(
                f"No annotation for ({category}/{sequence_name}, frame {frame_number})"
            )
        return entry[key]

    def _lookup_frame_anno_from_npz(
        self, category: str, sequence_name: str, frame_number: int
    ) -> dict:
        """Look up a single frame's annotation from the in-memory npz arrays.

        Cost: ~0.01 ms (dict lookup + searchsorted + 6 scalar index ops).
        """
        z = self._frame_anno_npz[category]
        start, end = z["seq_to_range"][sequence_name]
        fn_slice = z["fns"][start:end]
        pos = int(np.searchsorted(fn_slice, frame_number))
        fi  = start + pos
        if fi >= end or int(z["fns"][fi]) != frame_number:
            raise KeyError(
                f"No annotation for ({category}/{sequence_name}, frame {frame_number})"
            )

        fn     = int(z["fns"][fi])
        focal  = z["focal"][fi].tolist()
        pp     = z["pp"][fi].tolist()
        R_list = z["R"][fi].tolist()
        T_list = z["T"][fi].tolist()
        H, W   = int(z["imgsz"][fi, 0]), int(z["imgsz"][fi, 1])
        dscale = float(z["dscale"][fi])

        # Use stored image stem (e.g. "frame000025") to reconstruct paths.
        # Co3D's frame_number is NOT always equal to the numeric part of the
        # filename — for ~18 % of frames they diverge — so we must use the
        # pre-stored stem rather than formatting frame_number directly.
        # Fall back to formatting if img_stems is absent (old v1 npz).
        img_stems = z.get("img_stems")
        stem = str(img_stems[fi]) if img_stems is not None else f"frame{fn:06d}"
        return {
            "sequence_name": sequence_name,
            "frame_number":  fn,
            "image": {
                "path": f"{category}/{sequence_name}/images/{stem}.jpg",
                "size": [H, W],
            },
            "depth": {
                "path": f"{category}/{sequence_name}/depths/{stem}.jpg.geometric.png",
                "scale_adjustment": dscale,
                "mask_path": f"{category}/{sequence_name}/depth_masks/{stem}.png",
            },
            "mask": {"path": f"{category}/{sequence_name}/masks/{stem}.png"},
            "viewpoint": {
                "focal_length":    focal,
                "principal_point": pp,
                "R": R_list,
                "T": T_list,
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_record(self, sequence_name: str) -> _Co3DSequenceRecord:
        if sequence_name not in self._uid_to_record:
            raise KeyError(
                f"Unknown sequence_name: '{sequence_name}'. "
                f"Use list_sequences() to enumerate valid names."
            )
        return self._uid_to_record[sequence_name]

    def _check_indices(
        self, frame_indices: list[int], num_frames: int, sequence_name: str
    ) -> None:
        if not frame_indices:
            raise ValueError("frame_indices is empty")
        lo, hi = min(frame_indices), max(frame_indices)
        if lo < 0 or hi >= num_frames:
            raise IndexError(
                f"[{sequence_name}] frame_indices out of range: "
                f"min={lo}, max={hi}, num_frames={num_frames}"
            )
