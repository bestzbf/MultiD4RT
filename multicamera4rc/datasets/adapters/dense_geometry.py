"""Read-only geometry adapters for native archives; no generated tracks."""
from __future__ import annotations

from collections import OrderedDict
import csv
import hashlib
import io
from pathlib import Path
import zipfile

import cv2
import numpy as np

from .base import BaseAdapter, UnifiedClip, h5_read_frame_slice
from .pointodyssey import PointOdysseyAdapter


def canonicalize_world_basis(extrinsics):
    """Rebase a consistently reflected world, preserving camera coordinates.

    E' = E F with F=diag(-1,1,1,1) makes det(R') positive; scene world points
    would become F X. E'_0 inv(E'_t) equals E_0 inv(E_t) exactly. This is a
    world-basis convention conversion, never a fitted/repaired camera pose.
    """
    E = np.asarray(extrinsics).copy()
    det = np.linalg.det(E[:, :3, :3])
    negative = det < 0
    if negative.any() and not negative.all():
        raise ValueError("world handedness changes within one geometry clip")
    if negative.all():
        E[:, :3, 0] *= -1
        return E, "left"
    return E, "right"


class PointOdysseyGeometryAdapter(PointOdysseyAdapter):
    """Load K/E and dense depth without decoding the large particle arrays."""
    def __init__(self, root, split="train", cache_dir=None, io_workers=1,
                 index_workers=2, verbose=False):
        super().__init__(root, split, cache_dir=cache_dir, io_workers=io_workers,
                         index_workers=index_workers, load_normals=False,
                         require_tracks=False, verbose=verbose)

    def load_clip(self, sequence_name, frame_indices):
        import h5py
        r = self.get_record(sequence_name)
        if not frame_indices or any(i < 0 or i >= r.num_frames for i in frame_indices):
            raise IndexError("PointOdyssey geometry frame outside sequence")
        if r.fast_anno_paths:
            anno = {k: np.load(r.fast_anno_paths[k], mmap_mode="r")[frame_indices]
                    for k in ("intrinsics", "extrinsics")}
        else:
            if r.anno_path is None:
                raise FileNotFoundError("PointOdyssey geometry requires native calibration")
            hp = r.anno_path.with_suffix(".h5")
            if hp.is_file():
                with h5py.File(hp, "r") as f:
                    anno = h5_read_frame_slice(f, frame_indices, keys=["intrinsics", "extrinsics"])
            else:
                with np.load(r.anno_path, allow_pickle=False) as f:
                    anno = {k: f[k][frame_indices] for k in ("intrinsics", "extrinsics")}
        images, depths, _ = self._load_frame_payload(sequence_name, r, np.asarray(frame_indices))
        if depths is None or any(d is None for d in depths):
            raise ValueError("missing native PointOdyssey depth")
        depths = [np.asarray(d, np.float32)*(1000./65535.) for d in depths]
        extrinsics, handedness = canonicalize_world_basis(np.asarray(anno["extrinsics"], np.float32))
        return UnifiedClip(dataset_name="pointodyssey_geometry", sequence_name=sequence_name,
            normals=None, trajs_2d=None, trajs_3d_world=None, valids=None, visibs=None,
            frame_paths=[str(r.rgb_paths[i]) for i in frame_indices], images=images,
            depths=depths, intrinsics=np.asarray(anno["intrinsics"], np.float32),
            extrinsics=extrinsics,
            metadata={"frame_indices": list(frame_indices), "extrinsics_convention": "w2c",
                      "pixel_coordinates": "array_index_v1", "has_tracks": False,
                      "depth_unit": "pointodyssey_world_units", "gt_kind": "geometry_only",
                      "native_world_handedness": handedness,
                      "world_basis_conversion": "x_reflection" if handedness == "left" else "identity"})


class PointOdysseyTracksAdapter(PointOdysseyGeometryAdapter):
    """Dense native depth plus all released persistent particle trajectories."""
    def load_clip(self, sequence_name, frame_indices):
        from dataclasses import replace
        clip = super().load_clip(sequence_name, frame_indices)
        anno = self._load_anno(sequence_name, np.asarray(frame_indices, np.int64))
        uv = np.asarray(anno["trajs_2d"], np.float64)
        for key in ("valids", "visibs"):
            if not np.isfinite(anno[key]).all() or not np.isin(anno[key], [0, 1]).all():
                raise ValueError(f"invalid native PointOdyssey {key} labels")
        native_valid = np.asarray(anno["valids"], bool).copy()
        valid = native_valid.copy(); visible = np.asarray(anno["visibs"], bool).copy()
        S = len(frame_indices)
        if uv.ndim != 3 or uv.shape[0] != S or uv.shape[-1] != 2:
            raise ValueError("PointOdyssey requires native persistent image trajectories")
        N = uv.shape[1]
        raw_world = anno.get("trajs_3d")
        lifted = raw_world is None or np.asarray(raw_world).ndim == 0 or np.asarray(raw_world).size == 0
        if lifted:
            # Some released sequences only annotate 2D persistent identities.
            # Native calibrated depth establishes visible 3D positions, never
            # a hidden endpoint. Require all bilinear contributors to belong
            # to a smooth valid surface so a snapped UV cannot bridge objects.
            world = np.zeros((S, N, 3), np.float64); valid[:] = False
            for s in range(S):
                d = clip.depths[s]; h, w = d.shape
                lo = cv2.erode(d, np.ones((3, 3), np.uint8))
                hi = cv2.dilate(d, np.ones((3, 3), np.uint8))
                safe = np.isfinite(d) & (lo > 0) & ((hi-lo) <= .02+.01*d)
                coords = np.nan_to_num(uv[s], nan=-1e6, posinf=1e6, neginf=-1e6).astype(np.float32)
                z = cv2.remap(d, coords[:, 0, None], coords[:, 1, None], cv2.INTER_LINEAR)[:, 0]
                support = cv2.remap(safe.astype(np.float32), coords[:, 0, None], coords[:, 1, None], cv2.INTER_LINEAR)[:, 0] > .999
                okay = native_valid[s] & visible[s] & support & np.isfinite(uv[s]).all(-1)
                okay &= (coords[:, 0] >= 0) & (coords[:, 0] <= w-1) & (coords[:, 1] >= 0) & (coords[:, 1] <= h-1)
                okay &= np.isfinite(z) & (z > 0)
                xyz = np.c_[coords, np.ones(N)] @ np.linalg.inv(clip.intrinsics[s]).T*z[:, None]
                C = np.linalg.inv(clip.extrinsics[s].astype(np.float64))
                world[s] = np.where(okay[:, None], xyz @ C[:3, :3].T+C[:3, 3], 0)
                valid[s] = okay
        else:
            world = np.asarray(raw_world, np.float64).copy()
            if world.shape != (S, N, 3):
                raise ValueError("PointOdyssey native world trajectory dimensions disagree")
            if clip.metadata["native_world_handedness"] == "left":
                world[..., 0] *= -1
        if uv.shape != (S, N, 2) or valid.shape != (S, N) or visible.shape != valid.shape:
            raise ValueError("PointOdyssey native trajectory shapes disagree")
        valid &= np.isfinite(world).all(-1)
        source_support = np.zeros_like(valid)
        diagnostics = []
        # Native release precision is finite (half-quantized XYZ/UV, and
        # snapped boundary UV). Check source agreement without fitting XYZ.
        for s in range(S):
            d = clip.depths[s]; h, w = d.shape
            camera = world[s] @ clip.extrinsics[s, :3, :3].T+clip.extrinsics[s, :3, 3]
            p = camera @ clip.intrinsics[s].T
            projected = p[:, :2]/np.maximum(p[:, 2:], 1e-8)
            inside = np.isfinite(uv[s]).all(-1) & (uv[s, :, 0] >= 0) & (uv[s, :, 0] <= w-1)
            inside &= (uv[s, :, 1] >= 0) & (uv[s, :, 1] <= h-1) & (camera[:, 2] > 0)
            reprojection = np.linalg.norm(projected-uv[s], axis=-1)
            xy = np.rint(np.nan_to_num(uv[s], nan=0, posinf=0, neginf=0)).astype(np.int64)
            xy[:, 0] = np.clip(xy[:, 0], 0, w-1); xy[:, 1] = np.clip(xy[:, 1], 0, h-1)
            dz = np.full(N, np.inf)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    observed = d[np.clip(xy[:, 1]+dy, 0, h-1), np.clip(xy[:, 0]+dx, 0, w-1)]
                    dz = np.minimum(dz, np.where(np.isfinite(observed) & (observed > 0),
                                                abs(observed-camera[:, 2]), np.inf))
            okay = valid[s] & visible[s] & inside & (reprojection <= 2.)
            okay &= dz <= .03*np.maximum(camera[:, 2], 0)+2*1000./65535.
            source_support[s] = okay
            diagnostics.append(dict(native_visible=int((valid[s] & visible[s]).sum()),
                                    source_supported=int(okay.sum())))
            # A visibly inconsistent XYZ must also be excluded when queried
            # from a different source frame. Native occluded XYZ remains valid;
            # 2D-only annotations never establish an occluded 3D endpoint.
            valid[s] &= ~visible[s] | okay
        metadata = dict(clip.metadata, has_tracks=True, has_visibility=True,
            gt_kind="dense_geometry_native_tracks", track_source_valid=source_support,
            native_track_checks=diagnostics, track_precision_policy="native_fp16_2px_3pct_depth_neighborhood",
            trajectory_source="depth_lifted_native_2d_visible_only" if lifted else "native_persistent_particle_ids_no_rasterization",
            has_native_world_trajectories=not lifted, hidden_3d_supervision=not lifted)
        return replace(clip, trajs_2d=uv.astype(np.float32), trajs_3d_world=world.astype(np.float32),
                       valids=valid, visibs=visible, visibility_valids=native_valid, metadata=metadata)


class _ArchiveGeometry(BaseAdapter):
    def _initialize(self, root, split, cache_dir, key):
        self.root, self.split = Path(root), split
        self._archives = OrderedDict()
        if cache_dir:
            from ..index_cache import load_or_build
            stamp = hashlib.sha256(str((str(self.root.resolve()), split, key)).encode()).hexdigest()[:16]
            self.records = load_or_build(self._build_index, Path(cache_dir)/f"{self.dataset_name}_{stamp}.pkl")
        else:
            self.records = self._build_index()
        if not self.records:
            raise ValueError(f"no valid {self.dataset_name} geometry sequences in {root}")

    def __getstate__(self):
        state = self.__dict__.copy(); state["_archives"] = OrderedDict(); return state

    def _zip(self, path):
        key = str(path)
        if key not in self._archives:
            self._archives[key] = zipfile.ZipFile(path)
        self._archives.move_to_end(key)
        while len(self._archives) > 4:
            _, z = self._archives.popitem(last=False); z.close()
        return self._archives[key]

    @staticmethod
    def _array(z, name):
        import h5py
        with h5py.File(io.BytesIO(z.read(name)), "r") as f:
            return np.asarray(f["dataset"])

    def __len__(self): return len(self.records)
    def list_sequences(self): return sorted(self.records)
    def get_sequence_name(self, index): return self.list_sequences()[index]
    def get_num_frames(self, sequence_name): return len(self.records[sequence_name]["frames"])
    def get_sequence_info(self, sequence_name):
        r = self.records[sequence_name]
        return {"num_frames": len(r["frames"]), "physical_scene": r["physical_scene"],
                "has_depth": True, "has_tracks": False}
    def sanity_check(self, sequence_name):
        c = self.load_clip(sequence_name, [0])
        return {"ok": bool(np.any(c.depths[0] > 0)), "num_frames": self.get_num_frames(sequence_name)}


def depth_tail_support(depth, *, depth_max=None, percentile=None):
    """Mask range/tail labels without clamping or rewriting native depth.

    Apply the physical range before the percentile: a sky occupying more than
    two percent of the image otherwise survives a plain p98 threshold.
    """
    if depth_max is not None and (not np.isfinite(depth_max) or depth_max <= 0):
        raise ValueError("depth_max must be finite and positive, or None")
    if percentile is not None and (not np.isfinite(percentile) or not 0 < percentile <= 100):
        raise ValueError("depth percentile must be in (0,100], or None")
    d = np.asarray(depth)
    if d.ndim != 2:
        raise ValueError("depth tail filtering requires a two-dimensional map")
    finite = np.isfinite(d) & (d > 0)
    support = finite.copy()
    if depth_max is not None:
        support &= d <= depth_max
    ranged = int(support.sum())
    cutoff = float(np.percentile(d[support], percentile)) if percentile is not None and ranged else None
    if cutoff is not None:
        # Ties (including a constant depth plane) remain valid.
        support &= d <= cutoff
    stats = dict(depth_max=depth_max, percentile=percentile, cutoff=cutoff,
                 finite_pixels=int(finite.sum()), range_valid_pixels=ranged,
                 retained_pixels=int(support.sum()))
    return support, stats


class TartanAirV1ZipAdapter(_ArchiveGeometry):
    """Both V1 camera streams; calibration is OpenCV w2c, depth is camera Z."""
    dataset_name = "tartanair_v1_zip"

    def __init__(self, root, split="train", cache_dir=None, camera="left", verbose=False,
                 depth_max=None, depth_percentile=None):
        if camera not in ("left", "right"):
            raise ValueError("TartanAir V1 camera must be left or right")
        self.camera = camera
        depth_tail_support(np.ones((1, 1)), depth_max=depth_max, percentile=depth_percentile)
        self.depth_max, self.depth_percentile = depth_max, depth_percentile
        self._initialize(root, split, cache_dir, ("v1", camera, 1))

    def _build_index(self):
        records = {}
        for env in sorted(self.root.iterdir()):
            if not env.is_dir(): continue
            for difficulty in ("Easy", "Hard"):
                directory = env/difficulty
                dp, ip = directory/f"depth_{self.camera}.zip", directory/f"image_{self.camera}.zip"
                if not dp.is_file() or not ip.is_file(): continue
                with zipfile.ZipFile(dp) as z, zipfile.ZipFile(ip) as rgb:
                    names, rgb_names = set(z.namelist()), set(rgb.namelist())
                    for pn in sorted(n for n in names if n.endswith(f"/pose_{self.camera}.txt")):
                        base = pn.rsplit("/", 1)[0]
                        poses = np.atleast_2d(np.loadtxt(io.BytesIO(z.read(pn))))
                        frames = [f for f in range(len(poses))
                            if f"{base}/depth_{self.camera}/{f:06d}_{self.camera}_depth.npy" in names
                            and f"{base}/image_{self.camera}/{f:06d}_{self.camera}.png" in rgb_names]
                        if not frames or poses.shape[1] != 7 or not np.isfinite(poses).all(): continue
                        sequence = f"{base}/{self.camera}"
                        records[sequence] = dict(frames=frames, poses=poses, base=base,
                            depth_zip=str(dp), image_zip=str(ip), physical_scene=env.name)
        return records

    def load_clip(self, sequence_name, frame_indices):
        from .TartanAir import quat_to_rotation_matrix
        r = self.records[sequence_name]
        dz, iz = self._zip(r["depth_zip"]), self._zip(r["image_zip"])
        images, depths, E, frames, support, filter_stats = [], [], [], [], [], []
        for i in frame_indices:
            f = r["frames"][i]; frames.append(f)
            dn = f"{r['base']}/depth_{self.camera}/{f:06d}_{self.camera}_depth.npy"
            im = f"{r['base']}/image_{self.camera}/{f:06d}_{self.camera}.png"
            raw = cv2.imdecode(np.frombuffer(iz.read(im), np.uint8), cv2.IMREAD_COLOR)
            if raw is None or raw.shape != (480, 640, 3):
                raise ValueError("native TartanAir V1 image must be 480x640; do not reuse K for V2")
            images.append(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
            d = np.load(io.BytesIO(dz.read(dn)), allow_pickle=False).astype(np.float32)
            if d.shape != (480, 640): raise ValueError("TartanAir depth shape mismatch")
            depths.append(d)
            mask, stats = depth_tail_support(d, depth_max=self.depth_max,
                                             percentile=self.depth_percentile)
            support.append(mask); filter_stats.append(stats)
            p = r["poses"][f]; R = np.asarray(quat_to_rotation_matrix(p[3:]))
            cv_from_ned = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
            pose = np.eye(4); pose[:3, :3] = cv_from_ned @ R.T
            pose[:3, 3] = -pose[:3, :3] @ p[:3]; E.append(pose)
        K = np.array([[320., 0, 320.], [0, 320., 240.], [0, 0, 1.]])
        return UnifiedClip(dataset_name=self.dataset_name, sequence_name=sequence_name,
            normals=None, trajs_2d=None, trajs_3d_world=None, valids=None, visibs=None,
            frame_paths=[f"{r['image_zip']}#{f}" for f in frames], images=images, depths=depths,
            intrinsics=np.repeat(K[None], len(frames), 0).astype(np.float32), extrinsics=np.asarray(E, np.float32),
            metadata={"frame_numbers": frames, "physical_scene": r["physical_scene"],
                      "extrinsics_convention": "w2c", "pixel_coordinates": "array_index_v1",
                      "gt_kind": "geometry_only", "pixel_support": support,
                      "depth_filter": filter_stats})


def factor_projection(P, height, width):
    """Official projection -> virtual OpenCV K and proper camera-axis rotation."""
    A = np.stack((width/2*P[0, :3]+(width/2-.5)*P[3, :3],
                  -height/2*P[1, :3]+(height/2-.5)*P[3, :3], P[3, :3])) @ np.diag([1., -1., -1.])
    Q, B = np.linalg.qr(np.flipud(A).T)
    K, R = np.fliplr(np.flipud(B.T)), np.flipud(Q.T)
    sign = np.diag(np.sign(np.diag(K))); K, R = K @ sign, sign @ R
    scale = K[2, 2]; K = K/scale
    if not np.isfinite(K).all() or np.linalg.det(R) < .99999 or not np.allclose(A, scale*K @ R, atol=1e-8):
        raise ValueError("official Hypersim projection cannot be represented by a proper central camera")
    if np.abs(P[[0, 1, 3], 3]).max() > 1e-10:
        raise ValueError("unsupported non-central Hypersim projection")
    return K, R


def rectify_skew(image, depth, K):
    """Remove intrinsic skew by a known homography; keep the entire image."""
    if abs(K[0, 1]) < 1e-5:
        return image, depth, K, np.isfinite(depth) & (depth > 0)
    h, w = depth.shape
    newK = K.copy(); newK[0, 1] = 0
    H = newK @ np.linalg.inv(K)
    corners = np.array([[0, 0, 1], [w-1, 0, 1], [0, h-1, 1], [w-1, h-1, 1]]) @ H.T
    minx, maxx = np.floor(corners[:, 0].min()), np.ceil(corners[:, 0].max())
    shift = np.eye(3); shift[0, 2] = -minx
    newK, H = shift @ newK, shift @ H
    size = (int(maxx-minx)+1, h)
    rgb = cv2.warpPerspective(image, H, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = np.isfinite(depth) & (depth > 0)
    # Normalized bilinear interpolation with full support, then a depth-edge
    # mask, avoids interpolating invalid holes or across surface boundaries.
    support = cv2.warpPerspective(valid.astype(np.float32), H, size, flags=cv2.INTER_LINEAR)
    z = cv2.warpPerspective(np.where(valid, depth, 0), H, size, flags=cv2.INTER_LINEAR)
    lo, hi = cv2.erode(depth, np.ones((3, 3), np.uint8)), cv2.dilate(depth, np.ones((3, 3), np.uint8))
    interior = valid & (lo > 0) & ((hi-lo) < .02+.01*depth)
    safe = cv2.warpPerspective(interior.astype(np.uint8), H, size, flags=cv2.INTER_NEAREST) > 0
    mask = (support > .999) & safe & np.isfinite(z) & (z > 0)
    return rgb, np.where(mask, z, 0), newK, mask


class HypersimZipAdapter(_ArchiveGeometry):
    dataset_name = "hypersim_zip"

    def __init__(self, root, camera_metadata, split_metadata, split="train", cache_dir=None,
                 verbose=False):
        self.camera_metadata, self.split_metadata = Path(camera_metadata), Path(split_metadata)
        self._initialize(root, split, cache_dir, ("official_rq_rectify_v1", hashlib.sha256(self.camera_metadata.read_bytes()).hexdigest(), hashlib.sha256(self.split_metadata.read_bytes()).hexdigest()))

    def _build_index(self):
        official = {r["scene_name"]: r for r in csv.DictReader(self.camera_metadata.open())}
        allowed = {}
        for row in csv.DictReader(self.split_metadata.open()):
            if row["included_in_public_release"] != "True": continue
            if self.split != "all" and row["split_partition_name"] != self.split: continue
            allowed.setdefault((row["scene_name"], row["camera_name"]), set()).add(int(row["frame_id"]))
        records = {}
        for path in sorted(self.root.glob("*.zip")):
            scene = path.stem
            if scene not in official or not any(s == scene for s, _ in allowed): continue
            row = official[scene]
            P = np.array([[float(row[f"M_proj_{i}{j}"]) for j in range(4)] for i in range(4)])
            h, w = int(float(row["settings_output_img_height"])), int(float(row["settings_output_img_width"]))
            K, axis = factor_projection(P, h, w)
            with zipfile.ZipFile(path) as z:
                names = set(z.namelist())
                units = {r["parameter_name"]: float(r["parameter_value"]) for r in csv.DictReader(io.StringIO(z.read(f"{scene}/_detail/metadata_scene.csv").decode()))}
                scale = units["meters_per_asset_unit"]
                if not np.isfinite(scale) or scale <= 0: raise ValueError("invalid Hypersim asset units")
                for s, cam in sorted(allowed):
                    if s != scene: continue
                    base = f"{scene}/_detail/{cam}"
                    pn = base+"/camera_keyframe_positions.hdf5"
                    if pn not in names: continue
                    ids = self._array(z, base+"/camera_keyframe_frame_indices.hdf5").astype(int)
                    positions = self._array(z, pn).astype(np.float64)*scale
                    rotations = self._array(z, base+"/camera_keyframe_orientations.hdf5").astype(np.float64)
                    selection = []
                    for i, f in enumerate(ids):
                        geo = f"{scene}/images/scene_{cam}_geometry_hdf5/frame.{f:04d}"
                        rgb = f"{scene}/images/scene_{cam}_final_preview/frame.{f:04d}.tonemap.jpg"
                        if int(f) in allowed[(scene, cam)] and rgb in names and geo+".depth_meters.hdf5" in names and geo+".position.hdf5" in names:
                            selection.append(i)
                    if not selection: continue
                    frames = ids[selection].tolist()
                    E = []
                    for i in selection:
                        R = axis @ np.diag([1., -1., -1.]) @ rotations[i].T
                        pose = np.eye(4); pose[:3, :3] = R; pose[:3, 3] = -R @ positions[i]; E.append(pose)
                    records[f"{scene}/{cam}"] = dict(frames=frames, physical_scene=scene,
                        archive=str(path), cam=cam, K=K, E=np.asarray(E), scale=scale,
                        camera_centers=positions[selection], native_hw=(h, w))
        return records

    def load_clip(self, sequence_name, frame_indices):
        r = self.records[sequence_name]; z = self._zip(r["archive"]); scene = r["physical_scene"]
        images, depths, intrinsics, support, frames, paths = [], [], [], [], [], []
        h, w = r["native_hw"]; y, x = np.indices((h, w))
        rays = np.stack((x, y, np.ones_like(x)), -1) @ np.linalg.inv(r["K"]).T
        length = np.linalg.norm(rays, axis=-1)
        for i in frame_indices:
            frame = r["frames"][i]; frames.append(frame)
            rgb = f"{scene}/images/scene_{r['cam']}_final_preview/frame.{frame:04d}.tonemap.jpg"
            geo = f"{scene}/images/scene_{r['cam']}_geometry_hdf5/frame.{frame:04d}"
            raw = cv2.imdecode(np.frombuffer(z.read(rgb), np.uint8), cv2.IMREAD_COLOR)
            if raw is None or raw.shape != (h, w, 3): raise ValueError("Hypersim RGB/calibration dimensions mismatch")
            image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            distance = self._array(z, geo+".depth_meters.hdf5").astype(np.float32)
            world = self._array(z, geo+".position.hdf5").astype(np.float64)*r["scale"]
            if distance.shape != (h, w) or world.shape != (h, w, 3): raise ValueError("Hypersim GT shape mismatch")
            expected = np.linalg.norm(world-r["camera_centers"][i], axis=-1)
            valid = np.isfinite(world).all(-1) & np.isfinite(distance) & (distance > 0)
            # Independent renderer world-position/range fields screen every
            # loaded pixel, including the original float16 quantization floor.
            valid &= abs(expected-distance) <= .01+.01*distance
            camera = world @ r["E"][i, :3, :3].T+r["E"][i, :3, 3]
            projected = camera @ r["K"].T
            uv = projected[..., :2]/np.maximum(projected[..., 2:], 1e-8)
            valid &= (camera[..., 2] > 0) & (np.linalg.norm(uv-np.stack((x, y), -1), axis=-1) <= 2.)
            depth = np.where(valid, distance/length, 0).astype(np.float32)
            image, depth, K, mask = rectify_skew(image, depth, r["K"])
            images.append(image); depths.append(depth); intrinsics.append(K); support.append(mask); paths.append(f"{r['archive']}#{rgb}")
        return UnifiedClip(dataset_name=self.dataset_name, sequence_name=sequence_name,
            normals=None, trajs_2d=None, trajs_3d_world=None, valids=None, visibs=None,
            frame_paths=paths, images=images, depths=depths, intrinsics=np.asarray(intrinsics, np.float32),
            extrinsics=r["E"][frame_indices].astype(np.float32), metadata={"frame_numbers": frames,
                "extrinsics_convention": "w2c", "pixel_coordinates": "array_index_v1",
                "pixel_support": support, "physical_scene": scene, "gt_kind": "geometry_only",
                "official_projection": True, "skew_rectified": bool(abs(r["K"][0, 1]) >= 1e-5)})
