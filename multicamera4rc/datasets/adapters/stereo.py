"""Synchronized native stereo streams, sharing the released world gauge."""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from .base import BaseAdapter
from .VirtualKitti import VKITTI2Adapter
from .dense_geometry import TartanAirV1ZipAdapter


def join_stereo(left, right, camera_names):
    """Camera-major C2/T observations; independent track pools cannot be joined."""
    times = [list(c.metadata.get("frame_numbers", c.metadata.get("frame_indices", [])))
             for c in (left, right)]
    if not times[0] or times[0] != times[1]:
        raise ValueError("stereo streams do not share native frame numbers")
    if any(c.trajs_3d_world is not None for c in (left, right)):
        raise ValueError("stereo geometry cannot merge independent particle identities")
    T = len(times[0])
    metadata = dict(left.metadata)
    metadata.update(frame_indices=times[0]*2, frame_numbers=times[0]*2,
                    time_idx=times[0]*2, slot_camera=[0]*T+[1]*T,
                    slot_physical_camera_id=[0]*T+[1]*T,
                    camera_names=[camera_names[0]]*T+[camera_names[1]]*T,
                    num_cameras=2, num_times=T, synchronized_stereo=True)
    for key in ("pixel_support", "depth_filter"):
        if key in left.metadata or key in right.metadata:
            metadata[key] = list(left.metadata[key])+list(right.metadata[key])
    return replace(left, images=left.images+right.images, depths=left.depths+right.depths,
        intrinsics=np.concatenate((left.intrinsics, right.intrinsics)),
        extrinsics=np.concatenate((left.extrinsics, right.extrinsics)),
        frame_paths=(left.frame_paths or [])+(right.frame_paths or []), metadata=metadata)


def stereo_times(frame_indices):
    if len(frame_indices) % 2 or not frame_indices:
        raise ValueError("native stereo requests must contain two identical time blocks")
    T = len(frame_indices)//2
    if frame_indices[:T] != frame_indices[T:] or len(set(frame_indices[:T])) != T:
        raise ValueError("native stereo requests must contain two identical unique time blocks")
    return frame_indices[:T]


class VKITTI2StereoAdapter(BaseAdapter):
    dataset_name = "vkitti2_stereo"

    def __init__(self, root, split="train", cache_dir=None, verbose=False):
        self.root, self.split = root, split
        self.streams = [VKITTI2Adapter(root, split, camera=cam, cache_dir=cache_dir,
            verbose=verbose, strict_calibration=True, load_precomputed=False,
            load_normals=False, load_flow=False) for cam in ("Camera_0", "Camera_1")]
        self.seq_to_dir = self.streams[0].seq_to_dir
        self.sequences = sorted(set(self.streams[0].list_sequences()) & set(self.streams[1].list_sequences()))
        for seq in self.sequences:
            if self.streams[0].get_num_frames(seq) != self.streams[1].get_num_frames(seq):
                raise ValueError(f"{seq}: unequal stereo timelines")

    def list_sequences(self): return self.sequences.copy()
    def __len__(self): return len(self.sequences)
    def get_sequence_name(self, index): return self.sequences[index]
    def get_num_frames(self, sequence_name): return self.streams[0].get_num_frames(sequence_name)
    def get_sequence_info(self, sequence_name):
        return dict(self.streams[0].get_sequence_info(sequence_name), num_cameras=2)
    def load_clip(self, sequence_name, frame_indices):
        times = stereo_times(frame_indices)
        clips = [a.load_clip(sequence_name, times) for a in self.streams]
        return join_stereo(*clips, ("Camera_0", "Camera_1"))
    def sanity_check(self, sequence_name): return self.get_sequence_info(sequence_name)


class TartanAirV1StereoAdapter(BaseAdapter):
    dataset_name = "tartanair_v1_stereo"

    def __init__(self, root, split="train", cache_dir=None, verbose=False,
                 depth_max=200., depth_percentile=98.):
        self.root, self.split = root, split
        self.streams = [TartanAirV1ZipAdapter(root, split, cache_dir=cache_dir,
            camera=cam, verbose=verbose, depth_max=depth_max,
            depth_percentile=depth_percentile) for cam in ("left", "right")]
        left = {s.rsplit("/", 1)[0] for s in self.streams[0].list_sequences()}
        right = {s.rsplit("/", 1)[0] for s in self.streams[1].list_sequences()}
        self.sequences = sorted(left & right)
        self.frames = {}
        for seq in self.sequences:
            records = [a.records[seq+"/"+cam] for a, cam in zip(self.streams, ("left", "right"))]
            self.frames[seq] = sorted(set(records[0]["frames"]) & set(records[1]["frames"]))
        self.sequences = [s for s in self.sequences if self.frames[s]]

    def list_sequences(self): return self.sequences.copy()
    def __len__(self): return len(self.sequences)
    def get_sequence_name(self, index): return self.sequences[index]
    def get_num_frames(self, sequence_name): return len(self.frames[sequence_name])
    def get_sequence_info(self, sequence_name):
        return dict(num_frames=self.get_num_frames(sequence_name), num_cameras=2,
                    physical_scene=sequence_name.split("/")[0], has_depth=True, has_tracks=False)
    def load_clip(self, sequence_name, frame_indices):
        times = stereo_times(frame_indices)
        physical = [self.frames[sequence_name][i] for i in times]
        clips = []
        for a, cam in zip(self.streams, ("left", "right")):
            seq = sequence_name+"/"+cam
            mapping = {f: i for i, f in enumerate(a.records[seq]["frames"])}
            clips.append(a.load_clip(seq, [mapping[f] for f in physical]))
        return replace(join_stereo(*clips, ("left", "right")), sequence_name=sequence_name)
    def sanity_check(self, sequence_name): return self.get_sequence_info(sequence_name)
