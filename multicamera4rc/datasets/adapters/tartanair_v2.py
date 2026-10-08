"""Native TartanAir V2 90-degree pinhole stereo archives (not fisheye)."""
from __future__ import annotations

import io
import zipfile
import cv2
import numpy as np

from .base import UnifiedClip
from .dense_geometry import _ArchiveGeometry, depth_tail_support
from .stereo import stereo_times
from .TartanAir import quat_to_rotation_matrix


def decode_depth_png(payload):
    rgba = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.dtype != np.uint8 or rgba.shape != (640, 640, 4):
        raise ValueError("TartanAir V2 depth must be native 640x640 lossless float32 PNG")
    # The official format is the four bytes returned by OpenCV, not RGB order.
    return np.ascontiguousarray(rgba).view("<f4")[..., 0].copy()


def pose_w2c(p):
    if p.shape != (7,) or not np.isfinite(p).all() or abs(np.linalg.norm(p[3:])-1) > 5e-3:
        raise ValueError("invalid native NED quaternion pose")
    cv_from_ned = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    E = np.eye(4)
    E[:3, :3] = cv_from_ned @ np.asarray(quat_to_rotation_matrix(p[3:])).T
    E[:3, 3] = -E[:3, :3] @ p[:3]
    return E


class TartanAirV2ZipAdapter(_ArchiveGeometry):
    dataset_name = "tartanair_v2_zip"

    def __init__(self, root, split="train", cache_dir=None, direction="front",
                 stereo=True, camera="lcam", depth_max=200., depth_percentile=98.,
                 verbose=False):
        if direction not in ("front", "back", "left", "right", "top", "bottom"):
            raise ValueError("unsupported native V2 pinhole direction")
        if not isinstance(stereo, bool) or camera not in ("lcam", "rcam"):
            raise ValueError("invalid V2 stereo/camera selection")
        depth_tail_support(np.ones((1, 1)), depth_max=depth_max, percentile=depth_percentile)
        self.direction, self.stereo = direction, stereo
        self.cameras = ["lcam_"+direction, "rcam_"+direction] if stereo else [camera+"_"+direction]
        self.depth_max, self.depth_percentile = depth_max, depth_percentile
        self._initialize(root, split, cache_dir, ("native_pinhole_v2", tuple(self.cameras)))

    def _build_index(self):
        records = {}
        for env in sorted(self.root.iterdir()):
            if not env.is_dir(): continue
            for difficulty in ("Data_easy", "Data_hard"):
                directory = env/difficulty
                archives = [(directory/f"depth_{c}.zip", directory/f"image_{c}.zip") for c in self.cameras]
                if not all(d.is_file() and i.is_file() for d, i in archives): continue
                stream_records = []
                for camera, (dp, ip) in zip(self.cameras, archives):
                    stream = {}
                    with zipfile.ZipFile(dp) as dz, zipfile.ZipFile(ip) as iz:
                        names, images = set(dz.namelist()), set(iz.namelist())
                        for pn in sorted(n for n in names if n.endswith(f"/pose_{camera}.txt")):
                            base = pn.rsplit("/", 1)[0]
                            p = np.atleast_2d(np.loadtxt(io.BytesIO(dz.read(pn))))
                            if p.shape[1] != 7: continue
                            frames = [f for f in range(len(p))
                                if f"{base}/depth_{camera}/{f:06d}_{camera}_depth.png" in names
                                and f"{base}/image_{camera}/{f:06d}_{camera}.png" in images
                                and np.isfinite(p[f]).all() and abs(np.linalg.norm(p[f, 3:])-1) < 5e-3]
                            if frames:
                                stream[base] = dict(poses=p, frames=frames, depth_zip=str(dp), image_zip=str(ip))
                    stream_records.append(stream)
                common = set(stream_records[0])
                for s in stream_records[1:]: common &= set(s)
                for base in sorted(common):
                    frames = set(stream_records[0][base]["frames"])
                    for s in stream_records[1:]: frames &= set(s[base]["frames"])
                    if frames:
                        records[base] = dict(frames=sorted(frames), base=base, physical_scene=env.name,
                                             streams=[s[base] for s in stream_records])
        return records

    def get_sequence_info(self, sequence_name):
        return dict(super().get_sequence_info(sequence_name), num_cameras=len(self.cameras))

    def sanity_check(self, sequence_name):
        clip = self.load_clip(sequence_name, [0, 0] if self.stereo else [0])
        return dict(ok=bool(any(np.any(d > 0) for d in clip.depths)),
                    num_frames=self.get_num_frames(sequence_name))

    def load_clip(self, sequence_name, frame_indices):
        times = stereo_times(frame_indices) if self.stereo else frame_indices
        if not times or any(i < 0 or i >= self.get_num_frames(sequence_name) for i in times):
            raise IndexError("V2 frame outside admitted native timeline")
        r = self.records[sequence_name]
        frames = [r["frames"][i] for i in times]
        images, depths, E, support, stats, paths = [], [], [], [], [], []
        for camera, stream in zip(self.cameras, r["streams"]):
            dz, iz = self._zip(stream["depth_zip"]), self._zip(stream["image_zip"])
            for f in frames:
                base = r["base"]
                ip = f"{base}/image_{camera}/{f:06d}_{camera}.png"
                raw = cv2.imdecode(np.frombuffer(iz.read(ip), np.uint8), cv2.IMREAD_COLOR)
                if raw is None or raw.shape != (640, 640, 3):
                    raise ValueError("V2 requires native 640x640 pinhole RGB")
                images.append(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
                d = decode_depth_png(dz.read(f"{base}/depth_{camera}/{f:06d}_{camera}_depth.png"))
                depths.append(d)
                mask, stat = depth_tail_support(d, depth_max=self.depth_max, percentile=self.depth_percentile)
                support.append(mask); stats.append(stat)
                E.append(pose_w2c(stream["poses"][f])); paths.append(stream["image_zip"]+"#"+ip)
        C, T = len(self.cameras), len(frames)
        K = np.array([[320., 0., 320.], [0., 320., 320.], [0., 0., 1.]], np.float32)
        return UnifiedClip(dataset_name=self.dataset_name, sequence_name=sequence_name,
            normals=None, trajs_2d=None, trajs_3d_world=None, valids=None, visibs=None,
            frame_paths=paths, images=images, depths=depths,
            intrinsics=np.repeat(K[None], C*T, 0), extrinsics=np.asarray(E, np.float32),
            metadata=dict(frame_numbers=frames*C, frame_indices=frames*C, time_idx=frames*C,
                slot_camera=np.repeat(np.arange(C), T), slot_physical_camera_id=np.repeat(np.arange(C), T),
                camera_names=np.repeat(self.cameras, T).tolist(), physical_scene=r["physical_scene"],
                num_cameras=C, num_times=T, extrinsics_convention="w2c", pixel_coordinates="array_index_v1",
                gt_kind="geometry_only", pixel_support=support, depth_filter=stats,
                depth_encoding="opencv_four_bytes_little_endian_float32", synchronized_stereo=self.stereo))
