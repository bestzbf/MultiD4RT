"""
scannetpp.py

功能说明
--------
该文件定义 ScanNet++ 版适配器 `ScanNetPPAdapter`，用于从原始
ScanNet++ scene 目录与 `precomputed.npz` 中读取训练/验证所需的数据，
并输出统一的 `UnifiedClip` 结构。

该适配器严格遵循数据契约：

1. 最终帧子集由 `iphone/colmap/images.txt` 决定。
2. RGB 的取帧依据 `iphone/pose_intrinsic_imu.json` 中的帧名-时间戳映射。
3. Depth 的取帧依据 JSON 完整序列中的 `full_index`。
4. 相机几何来自 `iphone/colmap/cameras.txt` 与 `images.txt`。
5. 内部固定采用单一 `undistorted pinhole plane`。
6. `trajs_2d`、`intrinsics`、`images` 在同一 RGB native 平面或其缩放平面上解释。
7. `normals` 原始定义在 depth native 平面，输出时根据 `target_hw` 同步 resize。

期望目录结构
------------
适配器默认读取如下原始场景布局：

```text
<root>/<scene_id>/
  iphone/
    colmap/
      images.txt
      cameras.txt
    pose_intrinsic_imu.json
    rgb.mkv
    depth.bin
  scans/
    mesh_aligned_0.05.ply
  precomputed.npz
```

其中：

- `precomputed.npz` 必须存在。
- 当前版本按“有 npz”的情况处理，不在 adapter 内回退重建 tracks。

坐标与缩放约定
--------------
适配器内部显式区分 native 平面与输出平面：

- `camera_native_hw`
  - 指 undistorted RGB 图像平面尺寸。
  - `npz` 中的 `trajs_2d` 与 `intrinsics` 默认定义在该平面。

- `depth_native_hw`
  - 固定为 ScanNet++ iPhone depth 的 `192 x 256`。
  - `npz` 中的 `normals` 默认定义在该平面。

- `target_hw`
  - `UnifiedClip` 输出给下游时采用的目标分辨率。
  - 若未指定，则默认等于 `camera_native_hw`。

当 `target_hw != camera_native_hw` 时，适配器会同步处理：

- `images` 按双线性插值 resize。
- `depths` 按最近邻 resize。
- `trajs_2d` 按 `(sx, sy)` 缩放。
- `intrinsics` 按 `(sx, sy)` 缩放 `fx, fy, cx, cy`。
- `normals` resize 后重新归一化。

因此该 adapter 不是“裸读 npz 后原样返回”，而是会根据输出空间对 2D 几何量做一致变换。

主要接口说明
------------
该文件不是独立命令行脚本，通常在训练、验证或数据检查脚本中被导入使用。

典型用法：

```python
from multicamera4rc.datasets.adapters.scannetpp import ScanNetPPAdapter

adapter = ScanNetPPAdapter(
    root="/mnt/ccw_1/d4rt/datas/scannetpp_ccw/data",
    target_hw=None,
)

clip = adapter.load_clip("0b031f3119", [0, 1, 2, 3])
```

若希望输出固定分辨率：

```python
adapter = ScanNetPPAdapter(
    root="/mnt/ccw_1/d4rt/datas/scannetpp_ccw/data",
    target_hw=(384, 512),
)
```

构造参数说明
------------
- `root`
  - 数据根目录。
  - 既可以直接指向包含 scene 子目录的目录，也可以让适配器自动尝试 `root.parent / "data"`。

- `split`
  - 数据划分名称。
  - 当前版本中主要作为接口保留字段，不直接参与场景筛选。

- `target_hw`
  - 输出目标分辨率，格式为 `(target_h, target_w)`。
  - 若为 `None`，则默认使用 undistorted RGB native 分辨率。

- `verbose`
  - 是否启用更详细的日志或调试输出。
  - 当前实现中仅保留接口位。

- `strict`
  - 是否启用严格模式。
  - 严格模式下若数据根目录、场景或必要文件缺失，会直接报错。

- `precomputed_name`
  - 预计算文件名。
  - 默认值为 `precomputed.npz`。

`load_clip()` 参数说明
----------------------
- `sequence_name`
  - 要加载的场景名，例如 `0b031f3119`。

- `frame_indices`
  - 以 COLMAP 子序列为索引的帧号列表。
  - 这些索引不是 JSON 完整序列索引，也不是视频原始帧号。
  - 适配器内部会把它们映射到：
    - JSON 时间戳
    - JSON 完整序列索引
    - 对应相机参数

`load_clip()` 返回内容
----------------------
返回一个 `UnifiedClip`，其中主要字段含义如下：

- `images`
  - 已按时间戳从 `rgb.mkv` 取帧、再做 undistort、再按 `target_hw` resize 的 RGB 图像列表。

- `depths`
  - 已按 JSON 完整序列索引从 `depth.bin` 取帧、再做 undistort、再按 `target_hw` resize 的深度图列表。

- `normals`
  - 从 `npz` 读取，并在需要时 resize + 归一化后的法线图列表。

- `trajs_2d`
  - 从 `npz` 读取后，根据 `target_hw` 做尺度变换的 2D 轨迹。

- `trajs_3d_world`
  - 从 `npz` 读取的世界坐标轨迹。

- `valids` / `visibs`
  - 从 `npz` 读取的有效性与可见性掩码。

- `intrinsics`
  - 从 `npz` 读取并按 `target_hw` 同步缩放后的内参。

- `extrinsics`
  - 从 `npz` 读取的 `w2c` 外参。

- `metadata`
  - 提供 clip 基本信息，例如总帧数、clip 帧数、原始分辨率、目标分辨率等。
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .base import BaseAdapter, UnifiedClip, load_precomputed_fast

_DEPTH_H = 192
_DEPTH_W = 256
_FRAME_INDEX_NAME = "frame_index.pkl"
_RGB_CACHE_DTYPE = np.dtype(np.uint8)


def _resolve_load_max_track_points() -> Optional[int]:
    raw = os.getenv(
        "D4RT_SCANNETPP_LOAD_MAX_TRACK_POINTS",
        os.getenv("D4RT_MAX_TRACK_POINTS", ""),
    ).strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


@dataclass(frozen=True)
class _ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass(frozen=True)
class _ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


@dataclass(frozen=True)
class _JsonFrame:
    stem: str
    full_index: int
    timestamp: float
    relative_timestamp: float


@dataclass(frozen=True)
class _FrameRecord:
    stem: str
    image_name: str
    full_index: int
    relative_timestamp: float
    camera_id: int
    w2c: np.ndarray


@dataclass(frozen=True)
class _UndistortSpec:
    undistorted_K: np.ndarray
    width: int
    height: int


def _read_cameras_text(path: Path) -> dict[int, _ColmapCamera]:
    cameras: dict[int, _ColmapCamera] = {}
    with open(path, "r") as fid:
        for raw_line in fid:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            cameras[camera_id] = _ColmapCamera(
                camera_id=camera_id,
                model=parts[1],
                width=int(parts[2]),
                height=int(parts[3]),
                params=np.array([float(x) for x in parts[4:]], dtype=np.float64),
            )
    return cameras


def _read_images_text(path: Path) -> dict[int, _ColmapImage]:
    images: dict[int, _ColmapImage] = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            image_id = int(parts[0])
            qvec = np.array([float(x) for x in parts[1:5]], dtype=np.float64)
            tvec = np.array([float(x) for x in parts[5:8]], dtype=np.float64)
            camera_id = int(parts[8])
            name = parts[9]
            fid.readline()
            images[image_id] = _ColmapImage(
                image_id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
            )
    return images


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
            [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
            [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2],
        ],
        dtype=np.float64,
    )


def _world_to_camera_matrix(image: _ColmapImage) -> np.ndarray:
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = _qvec_to_rotmat(image.qvec)
    w2c[:3, 3] = image.tvec
    return w2c


def _camera_intrinsic(camera: _ColmapCamera) -> np.ndarray:
    K = np.eye(3, dtype=np.float64)
    p = camera.params
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        K[0, 0] = p[0]
        K[1, 1] = p[0]
        K[0, 2] = p[1]
        K[1, 2] = p[2]
        return K
    if camera.model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"}:
        K[0, 0] = p[0]
        K[1, 1] = p[1]
        K[0, 2] = p[2]
        K[1, 2] = p[3]
        return K
    raise NotImplementedError(f"Unsupported COLMAP camera model: {camera.model}")


def _scale_intrinsic(K: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int) -> np.ndarray:
    out = K.copy().astype(np.float64)
    out[0, 0] *= dst_w / src_w
    out[0, 2] *= dst_w / src_w
    out[1, 1] *= dst_h / src_h
    out[1, 2] *= dst_h / src_h
    return out


def _build_undistort_maps(camera: _ColmapCamera) -> _UndistortSpec:
    raw_K = _camera_intrinsic(camera)
    return _UndistortSpec(
        undistorted_K=raw_K.copy(),
        width=camera.width,
        height=camera.height,
    )


def _load_json_frames(json_path: Path) -> dict[str, _JsonFrame]:
    with open(json_path, "r") as fh:
        payload = json.load(fh)
    stems = sorted(payload.keys())
    if not stems:
        raise RuntimeError(f"No frames in {json_path}")
    base_ts = float(payload[stems[0]]["timestamp"])
    frame_map: dict[str, _JsonFrame] = {}
    for full_index, stem in enumerate(stems):
        timestamp = float(payload[stem]["timestamp"])
        frame_map[stem] = _JsonFrame(
            stem=stem,
            full_index=full_index,
            timestamp=timestamp,
            relative_timestamp=timestamp - base_ts,
        )
    return frame_map


def _frame_index_path(scene_dir: Path) -> Path:
    return scene_dir / "iphone" / _FRAME_INDEX_NAME


def _scene_data_from_frame_index(scene_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    version = int(payload.get("version", 0))
    if version != 1:
        raise ValueError(f"Unsupported ScanNet++ frame index version: {version}")

    frame_stems = [str(v) for v in payload["frame_stems"]]
    full_indices = np.asarray(payload["full_indices"], dtype=np.int32)
    timestamps = [float(v) for v in payload["timestamps"]]
    intrinsics = np.asarray(payload["intrinsics"], dtype=np.float64)
    w2c = np.asarray(payload["w2c"], dtype=np.float64)
    camera_ids = [int(v) for v in payload.get("camera_ids", [0] * len(frame_stems))]
    image_names = payload.get("image_names")
    if image_names is None:
        image_names = [f"video/{stem}.jpg" for stem in frame_stems]
    image_names = [str(v) for v in image_names]

    n = len(frame_stems)
    if not (
        len(full_indices) == n
        and len(timestamps) == n
        and len(camera_ids) == n
        and len(image_names) == n
        and intrinsics.shape == (n, 3, 3)
        and w2c.shape == (n, 4, 4)
    ):
        raise ValueError(
            f"Invalid ScanNet++ frame index shapes for {scene_dir}: "
            f"n={n}, full_indices={full_indices.shape}, timestamps={len(timestamps)}, "
            f"intrinsics={intrinsics.shape}, w2c={w2c.shape}"
        )

    records = [
        _FrameRecord(
            stem=frame_stems[i],
            image_name=image_names[i],
            full_index=int(full_indices[i]),
            relative_timestamp=float(timestamps[i]),
            camera_id=int(camera_ids[i]),
            w2c=w2c[i],
        )
        for i in range(n)
    ]
    return {
        "scene_dir": scene_dir,
        "records": records,
        "camera_specs": {},
        "frame_stems": frame_stems,
        "full_indices": full_indices,
        "timestamps": timestamps,
        "camera_ids": camera_ids,
        "intrinsics": intrinsics,
        "w2c": w2c,
        "rgb_width": int(payload["rgb_width"]),
        "rgb_height": int(payload["rgb_height"]),
    }


def frame_index_payload_from_scene_data(scene_data: dict[str, Any]) -> dict[str, Any]:
    records = scene_data.get("records") or []
    image_names = [str(getattr(r, "image_name", f"video/{stem}.jpg")) for r, stem in zip(records, scene_data["frame_stems"])]
    if len(image_names) != len(scene_data["frame_stems"]):
        image_names = [f"video/{stem}.jpg" for stem in scene_data["frame_stems"]]
    return {
        "version": 1,
        "frame_stems": list(scene_data["frame_stems"]),
        "image_names": image_names,
        "full_indices": np.asarray(scene_data["full_indices"], dtype=np.int32),
        "timestamps": np.asarray(scene_data["timestamps"], dtype=np.float64),
        "camera_ids": np.asarray(scene_data["camera_ids"], dtype=np.int32),
        "intrinsics": np.asarray(scene_data["intrinsics"], dtype=np.float32),
        "w2c": np.asarray(scene_data["w2c"], dtype=np.float32),
        "rgb_width": int(scene_data["rgb_width"]),
        "rgb_height": int(scene_data["rgb_height"]),
    }


def _load_frame_index(scene_dir: Path) -> dict[str, Any]:
    with open(_frame_index_path(scene_dir), "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid ScanNet++ frame index payload: {type(payload).__name__}")
    return _scene_data_from_frame_index(scene_dir, payload)


def _join_frames_from_raw_metadata(scene_dir: Path) -> dict[str, Any]:
    colmap_dir = scene_dir / "iphone" / "colmap"
    json_path = scene_dir / "iphone" / "pose_intrinsic_imu.json"
    cameras = _read_cameras_text(colmap_dir / "cameras.txt")
    images = _read_images_text(colmap_dir / "images.txt")
    json_frames = _load_json_frames(json_path)
    records: list[_FrameRecord] = []
    for image in images.values():
        stem = Path(image.name).stem
        jf = json_frames.get(stem)
        if jf is None:
            raise KeyError(f"Frame '{stem}' in images.txt is missing from {json_path.name}")
        records.append(
            _FrameRecord(
                stem=stem,
                image_name=image.name,
                full_index=jf.full_index,
                relative_timestamp=jf.relative_timestamp,
                camera_id=image.camera_id,
                w2c=_world_to_camera_matrix(image),
            )
        )
    records.sort(key=lambda item: item.full_index)
    if not records:
        raise RuntimeError(f"No valid COLMAP frames found in {colmap_dir}")
    specs = {camera_id: _build_undistort_maps(camera) for camera_id, camera in cameras.items()}
    rgb_sizes = {(specs[r.camera_id].width, specs[r.camera_id].height) for r in records}
    if len(rgb_sizes) != 1:
        raise ValueError(f"Multiple RGB sizes found: {sorted(rgb_sizes)}")
    rgb_w, rgb_h = next(iter(rgb_sizes))
    intrinsics = np.stack([specs[r.camera_id].undistorted_K for r in records], axis=0).astype(np.float64)
    w2c = np.stack([r.w2c for r in records], axis=0).astype(np.float64)
    return {
        "scene_dir": scene_dir,
        "records": records,
        "camera_specs": specs,
        "frame_stems": [r.stem for r in records],
        "full_indices": np.array([r.full_index for r in records], dtype=np.int32),
        "timestamps": [r.relative_timestamp for r in records],
        "camera_ids": [r.camera_id for r in records],
        "intrinsics": intrinsics,
        "w2c": w2c,
        "rgb_width": rgb_w,
        "rgb_height": rgb_h,
    }


def _join_frames(scene_dir: Path) -> dict[str, Any]:
    index_path = _frame_index_path(scene_dir)
    if index_path.exists():
        return _load_frame_index(scene_dir)
    return _join_frames_from_raw_metadata(scene_dir)


def build_frame_index_payload(scene_dir: Path) -> dict[str, Any]:
    return frame_index_payload_from_scene_data(_join_frames_from_raw_metadata(scene_dir))


def _load_depth_bin_all(depth_path: Path) -> np.ndarray:
    try:
        with open(depth_path, "rb") as f:
            raw = f.read()
        data = zlib.decompress(raw, wbits=-zlib.MAX_WBITS)
        return np.frombuffer(data, dtype=np.float32).reshape(-1, _DEPTH_H, _DEPTH_W).copy()
    except Exception:
        pass
    import lz4.block

    frames: list[np.ndarray] = []
    with open(depth_path, "rb") as f:
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            chunk_size = int.from_bytes(hdr, byteorder="little")
            chunk = f.read(chunk_size)
            if len(chunk) < chunk_size:
                break
            try:
                dec = lz4.block.decompress(chunk, uncompressed_size=_DEPTH_H * _DEPTH_W * 2)
                depth = np.frombuffer(dec, dtype=np.uint16).reshape(_DEPTH_H, _DEPTH_W).astype(np.float32) / 1000.0
            except Exception:
                dec = zlib.decompress(chunk, wbits=-zlib.MAX_WBITS)
                depth = np.frombuffer(dec, dtype=np.float32).reshape(_DEPTH_H, _DEPTH_W)
            frames.append(depth.copy())
    return np.stack(frames, axis=0) if frames else np.zeros((0, _DEPTH_H, _DEPTH_W), dtype=np.float32)


def _index_depth_bin_chunks(depth_path: Path) -> list[tuple[int, int]]:
    """Return (offset, chunk_size) for every chunk in an lz4-chunked depth.bin."""
    offsets: list[tuple[int, int]] = []
    with open(depth_path, "rb") as f:
        while True:
            hdr_offset = f.tell()
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            chunk_size = int.from_bytes(hdr, byteorder="little")
            offsets.append((hdr_offset + 4, chunk_size))
            f.seek(chunk_size, 1)
    return offsets


def _load_depth_frames_indexed(depth_path: Path, frame_indices: list[int], chunk_offsets: list[tuple[int, int]]) -> list[np.ndarray]:
    """Decode only the requested frames using pre-built chunk offset index."""
    import lz4.block

    needed = sorted(set(frame_indices))
    idx_map: dict[int, np.ndarray] = {}
    with open(depth_path, "rb") as f:
        for fi in needed:
            if fi >= len(chunk_offsets):
                raise IndexError(f"Frame {fi} out of range ({len(chunk_offsets)} chunks)")
            offset, chunk_size = chunk_offsets[fi]
            f.seek(offset)
            chunk = f.read(chunk_size)
            idx_map[fi] = _decode_depth_chunk(chunk)
    return [idx_map[fi] for fi in frame_indices]


def _decode_depth_chunk(chunk: bytes) -> np.ndarray:
    import lz4.block

    try:
        dec = lz4.block.decompress(chunk, uncompressed_size=_DEPTH_H * _DEPTH_W * 2)
        depth = (
            np.frombuffer(dec, dtype=np.uint16)
            .reshape(_DEPTH_H, _DEPTH_W)
            .astype(np.float32)
            / 1000.0
        )
    except Exception:
        dec = zlib.decompress(chunk, wbits=-zlib.MAX_WBITS)
        depth = np.frombuffer(dec, dtype=np.float32).reshape(_DEPTH_H, _DEPTH_W)
    return depth.copy()


def _extract_video_frames_by_timestamps(video_path: Path, relative_timestamps: list[float], fallback_indices: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    results: list[np.ndarray | None] = [None] * len(relative_timestamps)
    sorted_pairs = sorted(enumerate(relative_timestamps), key=lambda item: item[1])

    # Seek once to the first target timestamp, then read sequentially.
    # This avoids N random seeks (each requiring keyframe decode) and replaces
    # them with 1 seek + sequential forward reads — much faster for H.264/MKV.
    first_ts = max(sorted_pairs[0][1], 0.0)
    cap.set(cv2.CAP_PROP_POS_MSEC, first_ts * 1000.0)

    target_idx = 0
    last_frame: np.ndarray | None = None

    while target_idx < len(sorted_pairs):
        out_idx, target_ts = sorted_pairs[target_idx]
        ret, frame = cap.read()
        if not ret:
            break
        got_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        last_frame = frame

        if got_seconds >= target_ts - 0.025:
            # Close enough: accept this frame for the current target.
            results[out_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            target_idx += 1
        # else: frame is still before target — keep reading forward.

    cap.release()

    # Fallback: any target not filled (e.g. video ended early) gets a direct seek.
    missing = [i for i, r in enumerate(results) if r is None]
    if missing:
        cap2 = cv2.VideoCapture(str(video_path))
        for out_idx in missing:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, int(fallback_indices[out_idx]))
            ret, frame = cap2.read()
            if not ret:
                raise IOError(f"Failed to read fallback frame {fallback_indices[out_idx]} from {video_path}")
            results[out_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap2.release()

    return [r for r in results]  # type: ignore[return-value]


def _read_rgb_jpg(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Failed to read RGB frame: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_cache_subdir_name(target_hw: tuple[int, int]) -> str:
    h, w = int(target_hw[0]), int(target_hw[1])
    return f"frames_{h}x{w}_rgb_uint8"


def rgb_cache_frame_path(
    scene_dir: Path,
    target_hw: tuple[int, int],
    frame_idx: int,
) -> Path:
    return (
        scene_dir
        / "iphone"
        / rgb_cache_subdir_name(target_hw)
        / f"{int(frame_idx):06d}.rgb"
    )


def _read_rgb_cache(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(target_hw[0]), int(target_hw[1])
    expected = h * w * 3
    data = np.fromfile(path, dtype=_RGB_CACHE_DTYPE)
    if data.size != expected:
        raise IOError(
            f"Invalid ScanNet++ RGB cache frame: {path} "
            f"got={data.size} expected={expected}"
        )
    return data.reshape(h, w, 3)


def write_rgb_cache_frame(
    path: Path,
    image: np.ndarray,
    target_hw: tuple[int, int],
) -> int:
    h, w = int(target_hw[0]), int(target_hw[1])
    if image.shape[:2] != (h, w):
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    image = np.ascontiguousarray(image.astype(np.uint8, copy=False))
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(tmp, "wb") as f:
            f.write(image.tobytes(order="C"))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return int(image.size * image.dtype.itemsize)


def _resize_normals(normal: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    tgt_h, tgt_w = target_hw
    if normal.shape[:2] == (tgt_h, tgt_w):
        out = normal.astype(np.float32)
    else:
        out = cv2.resize(normal.astype(np.float32), (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    safe = norm > 1e-6
    return np.where(safe, out / np.where(safe, norm, 1.0), 0.0).astype(np.float32)


class ScanNetPPAdapter(BaseAdapter):
    dataset_name = "scannetpp"

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_hw: Optional[tuple[int, int]] = None,
        verbose: bool = False,
        strict: bool = True,
        precomputed_name: str = "precomputed.npz",
        use_precomputed_tracks: bool = True,
        splits_dir: Optional[str] = None,
        split_file: Optional[str] = None,
        scenes_record: Optional[str] = None,
        cache_dir: Optional[str] = None,
        index_workers: int = 8,
        **kwargs,
    ):
        self.split = split
        self.root = Path(root)
        self.target_hw = target_hw
        self.verbose = verbose
        self.strict = strict
        self.precomputed_name = precomputed_name
        if isinstance(use_precomputed_tracks, str):
            self.use_precomputed_tracks = use_precomputed_tracks.strip().lower() in {
                "1", "true", "yes", "on"
            }
        else:
            self.use_precomputed_tracks = bool(use_precomputed_tracks)
        self.rgb_read_mode = os.getenv("SCANNETPP_RGB_READ_MODE", "auto").strip().lower() or "auto"
        if self.rgb_read_mode in {"cache256", "decoded_cache", "rgb_cache"}:
            self.rgb_read_mode = "cache"
        if self.rgb_read_mode not in {"auto", "frames", "video", "cache"}:
            self.rgb_read_mode = "auto"
        try:
            self.rgb_load_workers = max(
                1, int(os.getenv("SCANNETPP_RGB_LOAD_WORKERS", "4"))
            )
        except ValueError:
            self.rgb_load_workers = 4
        self.load_max_track_points = _resolve_load_max_track_points()
        self.index_workers = index_workers
        self.splits_dir = Path(splits_dir) if splits_dir is not None else self.root / "splits"
        self.split_file = split_file
        self.scenes_record_path = (
            Path(scenes_record)
            if scenes_record is not None
            else self.root.parent / "scenes_record.json"
        )

        # Check cache first to skip slow root.exists() on remote storage
        _cache_hit = False
        if cache_dir is not None:
            from multicamera4rc.datasets.index_cache import load_or_build
            # Assume data_root is root for cache key check
            cache_key = {
                "dataset": "scannetpp",
                "data_root": str(self.root),
                "split": split,
                "split_file": split_file,
                "splits_dir": str(self.splits_dir),
                "precomputed_name": precomputed_name,
                "scenes_record": str(self.scenes_record_path),
                "cache_schema": 4,
            }
            cache_suffix = hashlib.sha1(
                json.dumps(cache_key, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            _cache_path = Path(cache_dir) / f"scannetpp_{split}_{cache_suffix}.pkl"
            if _cache_path.exists():
                _cache_hit = True

        if _cache_hit:
            self.data_root = self.root
        else:
            self.data_root = self._resolve_data_root()

        if cache_dir is not None:
            from multicamera4rc.datasets.index_cache import load_or_build
            cache_key = {
                "dataset": self.dataset_name,
                "data_root": str(self.data_root),
                "split": self.split,
                "split_file": self.split_file,
                "splits_dir": str(self.splits_dir),
                "precomputed_name": self.precomputed_name,
                "scenes_record": str(self.scenes_record_path),
                "cache_schema": 4,
            }
            cache_suffix = hashlib.sha1(
                json.dumps(cache_key, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            _cache_path = Path(cache_dir) / f"{self.dataset_name}_{self.split}_{cache_suffix}.pkl"
            index: list[tuple[str, int]] = load_or_build(self._build_index, _cache_path)
        else:
            index = self._build_index()

        self.sequence_names: list[str] = [name for name, _ in index]
        self._num_frames_map: dict[str, int] = {name: nf for name, nf in index}
        self._scene_cache: dict[str, dict[str, Any]] = {}
        self._depth_chunk_cache: dict[str, list[tuple[int, int]]] = {}
        self._precomputed_info_cache: dict[str, dict[str, Any]] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if not hasattr(self, "load_max_track_points"):
            self.load_max_track_points = _resolve_load_max_track_points()

    def _precomputed_npz_path(self, scene_dir: Path) -> Path:
        return scene_dir / self.precomputed_name

    def _precomputed_h5_path(self, scene_dir: Path) -> Path:
        return self._precomputed_npz_path(scene_dir).with_suffix(".h5")

    def _has_precomputed(self, scene_dir: Path) -> bool:
        return self._precomputed_npz_path(scene_dir).exists() or self._precomputed_h5_path(scene_dir).exists()

    def _get_precomputed_info(self, scene_name: str) -> dict[str, Any]:
        if scene_name in self._precomputed_info_cache:
            return self._precomputed_info_cache[scene_name]

        if not self.use_precomputed_tracks:
            info = {
                "backend": None,
                "has_precomputed": False,
                "has_normals": False,
                "has_tracks": False,
                "has_visibility": False,
                "has_trajs_3d_world": False,
            }
            self._precomputed_info_cache[scene_name] = info
            return info

        scene_dir = self.data_root / scene_name
        npz_path = self._precomputed_npz_path(scene_dir)
        h5_path = self._precomputed_h5_path(scene_dir)

        keys: set[str] = set()
        backend = None
        if not keys and h5_path.exists():
            import h5py

            with h5py.File(h5_path, "r") as handle:
                keys = set(handle.keys())
            backend = "h5"
        if not keys and npz_path.exists():
            with np.load(npz_path, allow_pickle=False) as handle:
                keys = set(handle.files)
            backend = "npz"

        info = {
            "backend": backend,
            "has_precomputed": bool(keys),
            "has_normals": "normals" in keys,
            "has_tracks": {"trajs_2d", "trajs_3d_world", "valids", "visibs"}.issubset(keys),
            "has_visibility": "visibs" in keys,
            "has_trajs_3d_world": "trajs_3d_world" in keys,
        }
        self._precomputed_info_cache[scene_name] = info
        return info

    def _resolve_data_root(self) -> Path:
        # Fast path: if root itself looks like a data dir (or strict=False), skip iterdir scan
        if not self.strict and self.root.exists():
            return self.root
        candidates = [self.root]
        sibling = self.root.parent / "data"
        if sibling != self.root:
            candidates.append(sibling)
        for cand in candidates:
            if not cand.exists():
                continue
            for p in cand.iterdir():
                if p.is_dir() and (p / "iphone" / "colmap" / "images.txt").exists():
                    return cand
        if not self.strict:
            return self.root
        raise FileNotFoundError(f"No raw ScanNet++ scenes found under: {candidates}")

    def _load_split_allowlist(self) -> Optional[set[str]]:
        # Explicit split_file takes priority.
        if self.split_file is not None:
            p = Path(self.split_file)
            if not p.is_absolute():
                p = self.splits_dir / p
            if not p.exists():
                raise FileNotFoundError(f"split_file not found: {p}")
            return {line.strip() for line in p.read_text().splitlines() if line.strip()}

        # Auto-resolve from splits_dir using the standard nvs_sem_<split>.txt naming.
        if self.splits_dir.exists():
            candidate = self.splits_dir / f"nvs_sem_{self.split}.txt"
            if candidate.exists():
                return {line.strip() for line in candidate.read_text().splitlines() if line.strip()}

        return None

    def _load_scenes_record_index(self) -> dict[str, int]:
        """Load optional ScanNet++ preprocessing manifest as scene -> frame count."""
        path = self.scenes_record_path
        if not path.exists():
            return {}
        records: dict[str, int] = {}
        try:
            for raw_line in path.read_text().splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if str(item.get("status", "ok")) != "ok":
                    continue
                scene = item.get("scene_name")
                num_frames = int(item.get("num_frames", 0))
                if scene and num_frames > 0:
                    records[str(scene)] = num_frames
        except Exception:
            if self.strict:
                raise
            return {}
        return records

    def _build_index(self) -> list[tuple[str, int]]:
        """Return list of (scene_name, num_frames) for all valid scenes in this split."""
        results: list[tuple[str, int]] = []
        if not self.data_root.exists():
            if self.strict:
                raise RuntimeError(f"Data root does not exist: {self.data_root}")
            return results
        allowlist = self._load_split_allowlist()

        scene_dirs: list[Path] = []
        if allowlist is not None:
            # Use the allowlist directly to avoid a full directory walk.
            for name in sorted(allowlist):
                p = self.data_root / name
                scene_dirs.append(p)
        else:
            for p in sorted(self.data_root.iterdir()):
                if p.is_dir():
                    scene_dirs.append(p)

        record_index = self._load_scenes_record_index() if allowlist is not None else {}
        if record_index:
            for p in scene_dirs:
                num_frames = record_index.get(p.name)
                if num_frames is not None:
                    results.append((p.name, num_frames))
            if results:
                return results

        def _index_one(p: Path):
            import os as _os
            trust_allowlist = _os.getenv("D4RT_SCANNETPP_TRUST_ALLOWLIST", "").strip().lower() in {"1", "true", "yes"}
            if trust_allowlist and allowlist is not None:
                try:
                    sd = _join_frames(p)
                    return (p.name, len(sd["frame_stems"]))
                except Exception:
                    return None
            # Normal path: check files exist
            if (p / "iphone" / "colmap" / "images.txt").exists() and self._has_precomputed(p):
                try:
                    sd = _join_frames(p)
                    return (p.name, len(sd["frame_stems"]))
                except Exception:
                    return None
            return None

        n_workers = min(self.index_workers, len(scene_dirs))
        if n_workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(_index_one, p) for p in scene_dirs]
                for fut in as_completed(futures):
                    r = fut.result()
                    if r is not None:
                        results.append(r)
        else:
            for p in scene_dirs:
                r = _index_one(p)
                if r is not None:
                    results.append(r)

        if len(results) == 0 and self.strict:
            raise RuntimeError(f"No valid ScanNet++ scenes under {self.data_root} (split={self.split})")
        return results

    def get_num_frames(self, sequence_name: str) -> int:
        """O(1) lookup from the pre-built index — avoids re-reading scene files."""
        if sequence_name in self._num_frames_map:
            return self._num_frames_map[sequence_name]
        # Fallback for sequences loaded outside the index.
        return len(self._get_scene_data(sequence_name)["frame_stems"])

    def _get_scene_data(self, scene_name: str) -> dict[str, Any]:
        if scene_name in self._scene_cache:
            return self._scene_cache[scene_name]
        scene_dir = self.data_root / scene_name
        data = _join_frames(scene_dir)
        self._scene_cache[scene_name] = data
        return data

    def _get_depth_chunks(self, scene_name: str) -> list[tuple[int, int]]:
        if scene_name in self._depth_chunk_cache:
            return self._depth_chunk_cache[scene_name]
        sd = self._get_scene_data(scene_name)
        depth_path = sd["scene_dir"] / "iphone" / "depth.bin"
        chunks = _index_depth_bin_chunks(depth_path)
        self._depth_chunk_cache[scene_name] = chunks
        return chunks

    def _get_depths(self, scene_name: str, frame_indices: list[int]) -> list[np.ndarray]:
        sd = self._get_scene_data(scene_name)
        depth_path = sd["scene_dir"] / "iphone" / "depth.bin"
        # h5py fancy indexing requires sorted unique indices
        sorted_idx = sorted(set(frame_indices))
        idx_map = {v: pos for pos, v in enumerate(sorted_idx)}
        reorder = [idx_map[i] for i in frame_indices]
        full_indices_sorted = sd["full_indices"][sorted_idx].tolist()
        full_indices = [full_indices_sorted[r] for r in reorder]
        try:
            chunks = self._get_depth_chunks(scene_name)
            return _load_depth_frames_indexed(depth_path, full_indices, chunks)
        except Exception:
            all_depths = _load_depth_bin_all(depth_path)
            return [all_depths[fi] for fi in full_indices]

    def _load_precomputed(self, scene_name: str, frame_indices: list[int]) -> dict[str, Any]:
        sd = self._get_scene_data(scene_name)
        precomputed_dir = sd["scene_dir"]
        npz_path = precomputed_dir / self.precomputed_name
        # Read only the sampled frames directly from the scene's H5/NPZ file.
        cache = load_precomputed_fast(npz_path, frame_indices, skip_keys={"normals"})
        if cache is not None:
            cache = self._cap_precomputed_tracks(cache, scene_name, frame_indices)
        if cache is None:
            raise FileNotFoundError(npz_path)
        required = ["trajs_2d", "trajs_3d_world", "valids", "visibs", "intrinsics", "extrinsics"]
        missing = [key for key in required if key not in cache]
        if missing:
            raise KeyError(f"Missing keys in {npz_path.name}: {missing}")
        return cache

    def _cap_precomputed_tracks(
        self,
        cache: dict[str, Any],
        sequence_name: str,
        frame_indices: list[int],
    ) -> dict[str, Any]:
        trajs = cache.get("trajs_2d")
        if trajs is None or getattr(trajs, "ndim", 0) < 2:
            return cache
        keep_idx = self._select_track_points(
            sequence_name=sequence_name,
            frame_indices=frame_indices,
            num_points=int(trajs.shape[1]),
        )
        if keep_idx is None:
            return cache
        out = dict(cache)
        for key in ("trajs_2d", "trajs_3d_world", "valids", "visibs"):
            value = out.get(key)
            if value is not None and getattr(value, "ndim", 0) >= 2:
                out[key] = value[:, keep_idx]
        return out

    def _select_track_points(
        self,
        *,
        sequence_name: str,
        frame_indices: list[int],
        num_points: int,
    ) -> Optional[np.ndarray]:
        max_points = self.load_max_track_points
        if max_points is None or num_points <= max_points:
            return None
        payload = (
            f"scannetpp|{sequence_name}|{max_points}|"
            + ",".join(str(int(v)) for v in frame_indices)
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
        rng = np.random.default_rng(seed)
        return np.sort(
            rng.choice(num_points, size=max_points, replace=False).astype(np.int64)
        )

    def __len__(self) -> int:
        return len(self.sequence_names)

    def list_sequences(self) -> list[str]:
        return list(self.sequence_names)

    def get_sequence_name(self, index: int) -> str:
        return self.sequence_names[index]

    def get_sequence_info(self, sequence_name: str) -> dict[str, Any]:
        sd = self._get_scene_data(sequence_name)
        T = len(sd["frame_stems"])
        precomputed = self._get_precomputed_info(sequence_name)
        return {
            "dataset_name": self.dataset_name,
            "sequence_name": sequence_name,
            "path": str(sd["scene_dir"]),
            "num_frames": T,
            "rgb_width": sd["rgb_width"],
            "rgb_height": sd["rgb_height"],
            "depth_width": _DEPTH_W,
            "depth_height": _DEPTH_H,
            "target_hw": self.target_hw,
            "has_depth": True,
            "has_normals": precomputed["has_normals"],
            "has_tracks": precomputed["has_tracks"],
            "has_visibility": precomputed["has_visibility"],
            "has_trajs_3d_world": precomputed["has_trajs_3d_world"],
            "precomputed_backend": precomputed["backend"],
        }

    def load_clip(self, sequence_name: str, frame_indices: list[int]) -> UnifiedClip:
        import time as _time

        # Timing profile for this clip
        timing: dict[str, float] = {}
        t_total_start = _time.perf_counter()

        # Scene data lookup (includes cache)
        t0 = _time.perf_counter()
        sd = self._get_scene_data(sequence_name)
        timing["scene_data_s"] = _time.perf_counter() - t0

        T_total = len(sd["frame_stems"])
        if len(frame_indices) == 0:
            raise ValueError("frame_indices is empty")
        if min(frame_indices) < 0 or max(frame_indices) >= T_total:
            raise IndexError(f"[{sequence_name}] frame_indices out of range")

        native_h = sd["rgb_height"]
        native_w = sd["rgb_width"]
        if self.target_hw is None:
            tgt_h, tgt_w = native_h, native_w
        else:
            tgt_h, tgt_w = self.target_hw
        sx = tgt_w / float(native_w)
        sy = tgt_h / float(native_h)

        # Load RGB frames
        # h5py fancy indexing requires sorted unique indices
        sorted_idx = sorted(set(frame_indices))
        idx_map = {v: pos for pos, v in enumerate(sorted_idx)}
        reorder = [idx_map[i] for i in frame_indices]
        fallback_indices_sorted = sd["full_indices"][sorted_idx].tolist()
        fallback_indices = [fallback_indices_sorted[r] for r in reorder]
        frames_dir = sd["scene_dir"] / "iphone" / "frames"
        video_path = sd["scene_dir"] / "iphone" / "rgb.mkv"
        t0 = _time.perf_counter()
        cache_paths = [
            rgb_cache_frame_path(sd["scene_dir"], (tgt_h, tgt_w), i)
            for i in frame_indices
        ]
        missing_cache_paths: list[Path] = []
        if self.rgb_read_mode == "cache":
            missing_cache_paths = [path for path in cache_paths if not path.is_file()]
        use_cache = self.rgb_read_mode == "cache" and not missing_cache_paths
        use_video = self.rgb_read_mode == "video" and video_path.is_file()
        if use_cache:
            raw_images = [_read_rgb_cache(path, (tgt_h, tgt_w)) for path in cache_paths]
        elif self.rgb_read_mode == "cache":
            examples = ", ".join(str(path) for path in missing_cache_paths[:5])
            more = "" if len(missing_cache_paths) <= 5 else f", ... (+{len(missing_cache_paths) - 5} more)"
            raise FileNotFoundError(
                "ScanNet++ RGB cache mode requires predecoded frame cache; "
                f"refusing to read full iphone/rgb.mkv for sample "
                f"sequence={sequence_name!r} target_hw={(tgt_h, tgt_w)}. "
                f"Missing cache frames: {examples}{more}. "
                "Pre-generate the ScanNet++ RGB frame cache or use "
                "SCANNETPP_RGB_READ_MODE=video/frames/auto."
            )
        elif frames_dir.is_dir() and not use_video and self.rgb_read_mode != "cache":
            frame_paths = [frames_dir / f"{i:06d}.jpg" for i in frame_indices]
            if self.rgb_load_workers > 1 and len(frame_paths) > 1:
                with ThreadPoolExecutor(
                    max_workers=min(self.rgb_load_workers, len(frame_paths))
                ) as ex:
                    raw_images = list(ex.map(_read_rgb_jpg, frame_paths))
            else:
                raw_images = [_read_rgb_jpg(path) for path in frame_paths]
        else:
            timestamps = [sd["timestamps"][i] for i in frame_indices]
            raw_images = _extract_video_frames_by_timestamps(
                video_path,
                timestamps,
                fallback_indices,
            )
        timing["rgb_load_s"] = _time.perf_counter() - t0

        t0 = _time.perf_counter()
        images: list[np.ndarray] = []
        for image in raw_images:
            if image.shape[:2] != (tgt_h, tgt_w):
                image = cv2.resize(image, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
            images.append(image)
        timing["rgb_resize_s"] = _time.perf_counter() - t0

        # Load depth frames
        t0 = _time.perf_counter()
        raw_depths = self._get_depths(sequence_name, frame_indices)
        timing["depth_load_s"] = _time.perf_counter() - t0

        t0 = _time.perf_counter()
        depths: list[np.ndarray] = []
        for depth in raw_depths:
            if depth.shape[:2] != (tgt_h, tgt_w):
                depth = cv2.resize(depth, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST)
            depths.append(depth.astype(np.float32))
        timing["depth_resize_s"] = _time.perf_counter() - t0

        normals_out: Optional[list[np.ndarray]] = None
        trajs_2d: Optional[np.ndarray] = None
        trajs_3d_world: Optional[np.ndarray] = None
        valids: Optional[np.ndarray] = None
        visibs: Optional[np.ndarray] = None
        extrinsics: np.ndarray

        if self.use_precomputed_tracks:
            # Load precomputed data (tracks, intrinsics, etc.)
            t0 = _time.perf_counter()
            cache = self._load_precomputed(sequence_name, frame_indices)
            timing["precomputed_s"] = _time.perf_counter() - t0

            # Process intrinsics
            t0 = _time.perf_counter()
            intrinsics = cache["intrinsics"].astype(np.float32).copy()
            intrinsics[:, 0, 0] *= sx
            intrinsics[:, 0, 2] *= sx
            intrinsics[:, 1, 1] *= sy
            intrinsics[:, 1, 2] *= sy

            trajs_2d = cache["trajs_2d"].astype(np.float32).copy()
            trajs_2d[..., 0] *= sx
            trajs_2d[..., 1] *= sy
            trajs_3d_world = cache["trajs_3d_world"].astype(np.float32)
            valids = cache["valids"].astype(bool)
            visibs = cache["visibs"].astype(bool)
            extrinsics = cache["extrinsics"].astype(np.float32)

            if "normals" in cache:
                normals_out = [_resize_normals(normal, (tgt_h, tgt_w)) for normal in cache["normals"]]
            timing["process_s"] = _time.perf_counter() - t0
        else:
            timing["precomputed_s"] = 0.0
            t0 = _time.perf_counter()
            # h5py fancy indexing requires sorted unique indices
            sorted_idx = sorted(set(frame_indices))
            idx_map = {v: pos for pos, v in enumerate(sorted_idx)}
            reorder = [idx_map[i] for i in frame_indices]
            intrinsics_sorted = sd["intrinsics"][sorted_idx].astype(np.float32).copy()
            extrinsics_sorted = sd["w2c"][sorted_idx].astype(np.float32)
            intrinsics = intrinsics_sorted[reorder]
            extrinsics = extrinsics_sorted[reorder]
            intrinsics[:, 0, 0] *= sx
            intrinsics[:, 0, 2] *= sx
            intrinsics[:, 1, 1] *= sy
            intrinsics[:, 1, 2] *= sy
            timing["process_s"] = _time.perf_counter() - t0

        timing["total_s"] = _time.perf_counter() - t_total_start

        frame_paths = [f"{sd['scene_dir']}/iphone/rgb.mkv@t={sd['timestamps'][i]:.6f}s" for i in frame_indices]
        has_tracks = trajs_3d_world is not None
        track_points_loaded = int(trajs_2d.shape[1]) if trajs_2d is not None else None
        metadata = {
            "dataset_name": self.dataset_name,
            "sequence_name": sequence_name,
            "num_frames_total": T_total,
            "num_frames_clip": len(frame_indices),
            "rgb_hw": (native_h, native_w),
            "depth_hw": (_DEPTH_H, _DEPTH_W),
            "target_hw": (tgt_h, tgt_w),
            "extrinsics_convention": "w2c",
            "has_depth": True,
            "has_normals": normals_out is not None,
            "has_tracks": has_tracks,
            "has_visibility": visibs is not None,
            "has_trajs_3d_world": trajs_3d_world is not None,
            "track_points_loaded": track_points_loaded,
            "timestamps": [float(sd["timestamps"][i]) for i in frame_indices],
            "_load_timing": timing,
        }
        return UnifiedClip(
            dataset_name=self.dataset_name,
            sequence_name=sequence_name,
            frame_paths=frame_paths,
            images=images,
            depths=depths,
            normals=normals_out,
            trajs_2d=trajs_2d,
            trajs_3d_world=trajs_3d_world,
            valids=valids,
            visibs=visibs,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            metadata=metadata,
        )

    def sanity_check(self, sequence_name: str) -> dict[str, Any]:
        msgs: list[str] = []
        ok = True
        try:
            sd = self._get_scene_data(sequence_name)
            precomputed = self._get_precomputed_info(sequence_name)
        except Exception as e:
            return {
                "dataset_name": self.dataset_name,
                "sequence_name": sequence_name,
                "ok": False,
                "messages": [str(e)],
            }
        msgs.append(f"num_frames={len(sd['frame_stems'])} (COLMAP subset ordered by JSON full sequence)")
        msgs.append(f"rgb_size={sd['rgb_width']}x{sd['rgb_height']}")
        for rel_path in [Path("iphone/rgb.mkv"), Path("iphone/depth.bin")]:
            full = sd["scene_dir"] / rel_path
            if full.exists():
                msgs.append(f"{rel_path} exists")
            else:
                ok = False
                msgs.append(f"{rel_path} NOT found")
        if precomputed["has_precomputed"]:
            msgs.append(f"precomputed backend={precomputed['backend']}")
            msgs.append(f"has_normals={precomputed['has_normals']}")
            msgs.append(f"has_tracks={precomputed['has_tracks']}")
        else:
            ok = False
            msgs.append(f"{self.precomputed_name} / {Path(self.precomputed_name).with_suffix('.h5').name} NOT found")
        return {
            "dataset_name": self.dataset_name,
            "sequence_name": sequence_name,
            "ok": ok,
            "messages": msgs,
        }
