"""Exercise real migrated HDF5 readers through the 4RC supervision bridge."""

from __future__ import annotations

import multiprocessing

import h5py
import numpy as np
import pytest
import torch

from multicamera4rc import MultiCamera4RC, MultiCamera4RCConfig
from multicamera4rc.data import collate_multicamera
from multicamera4rc.datasets.adapters.base import h5_read_frame_slice
from multicamera4rc.datasets.bridge import prepare_clip
from multicamera4rc.datasets.registry import create_adapter, list_datasets


def _camera_fixture():
    height, width = 28, 42
    intrinsic = np.array([[30., 0., 21.], [0., 30., 14.], [0., 0., 1.]], np.float32)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], np.float32)
    reference = np.eye(4, dtype=np.float32)
    reference[:3, :3] = rotation
    reference[:3, 3] = [4., 2., -1.]
    cameras = np.repeat(reference[None], 2, axis=0)
    cameras[1, :3, 3] += rotation @ np.array([.2, 0., 0.], np.float32)
    world = np.empty((3, 11, 3), np.float32)
    for time in range(11):
        u = np.array([10.5, 18.5, 25.5], np.float32) + .15 * time
        v = np.array([8.5, 14.5, 20.5], np.float32)
        camera = np.stack(((u-21.)*2./30., (v-14.)*2./30., np.full(3, 2.)), axis=-1)
        world[:, time] = camera @ rotation.T + reference[:3, 3]
    return height, width, intrinsic, cameras, world


def _write_kubric(path):
    height, width, intrinsic, cameras, world = _camera_fixture()
    uv = np.empty((3, 11, 2, 2), np.float32)
    radial = np.empty((3, 11, 2), np.float32)
    for time in range(11):
        for camera_index in range(2):
            inverse = np.linalg.inv(cameras[camera_index])
            xyz = world[:, time] @ inverse[:3, :3].T + inverse[:3, 3]
            uv[:, time, camera_index] = xyz[:, :2] / xyz[:, 2:] * 30. + [21., 14.]
            radial[:, time, camera_index] = np.linalg.norm(xyz, axis=-1)
    ys, xs = np.mgrid[:height, :width]
    depth = 2. * np.sqrt(1. + ((xs+.5-21.)/30.)**2 + ((ys+.5-14.)/30.)**2)
    visibility = np.ones((3, 11, 2), bool)
    visibility[0, 5] = False
    with h5py.File(path, "w") as handle:
        frames = np.empty((11, 2, height, width, 3), np.uint8)
        for time in range(11):
            frames[time] = time * 10
        handle.create_dataset("frames", data=frames, chunks=(1, 1, height, width, 3))
        handle.create_dataset("depth", data=np.broadcast_to(depth, (11, 2, height, width)).astype(np.float32))
        handle.create_dataset("intrinsics", data=np.repeat(intrinsic[None], 2, axis=0))
        # Native Blender camera-to-world; the reader must convert it once.
        handle.create_dataset("extrinsics", data=cameras @ np.diag([1., -1., -1., 1.]))
        handle.create_dataset("camera_ids", data=np.array([b"cam0", b"cam1"]))
        handle.create_dataset("tracks_3d", data=world)
        handle.create_dataset("tracks_2d", data=uv)
        handle.create_dataset("track_depth", data=radial)
        handle.create_dataset("visibility", data=visibility)
        handle.attrs["schema_version"] = "movi_tracking_v3"
        handle.attrs["fps"] = 30.
    return cameras, world


def _kubric_adapter(root):
    return create_adapter(
        "kubric_multicam", root=str(root), split="all", verbose=False,
        min_cameras=2, max_cameras=2, min_times=3, max_times=3,
        encoder_tubelet_size=1, sampler_units="time", require_native_tracks=True,
        pixel_coordinates="array_index_v1", index_workers=1,
    )


def _write_mugen(path, *, tracks=True):
    height, width, intrinsic, cameras, world = _camera_fixture()
    intrinsic = intrinsic.copy()
    intrinsic[:2, 2] -= .5  # MUGEN already uses array-index coordinates.
    w2c = np.linalg.inv(cameras[0]).astype(np.float32)
    world = world.transpose(1, 0, 2)
    xyz = world @ w2c[:3, :3].T + w2c[:3, 3]
    uv = xyz[..., :2] / xyz[..., 2:] * 30. + intrinsic[:2, 2]
    valids, visible, known = np.ones((11, 3), bool), np.ones((11, 3), bool), np.ones((11, 3), bool)
    visible[5, 0] = False
    known[5, 1] = False
    with h5py.File(path, "w") as handle:
        group = handle.create_group("train/seq_000")
        group.attrs["sequence_name"] = "moving"
        group.create_dataset("images", data=np.full((11, height, width, 3), 100, np.uint8), chunks=(1, height, width, 3), compression="gzip")
        group.create_dataset("depths", data=np.full((11, height, width), 2., np.float32), chunks=(1, height, width), compression="gzip")
        group.create_dataset("intrinsics", data=np.repeat(intrinsic[None], 11, axis=0))
        group.create_dataset("extrinsics", data=np.repeat(w2c[None], 11, axis=0))
        if tracks:
            for name, array in {"trajs_2d": uv, "trajs_3d_world": world, "valids": valids, "visibs": visible, "visibility_valids": known}.items():
                group.create_dataset(name, data=array, chunks=(1,) + array.shape[1:])
    return cameras, world


def _mugen_adapter(path):
    return create_adapter("mugen_h5", root=str(path), split="train", dataset_name="pointodyssey", generate_static_tracks=False, prefer_precomputed_tracks=True, require_precomputed_tracks=True, train_only_val_ratio=0., verbose=False)


def test_kubric_real_h5_preserves_camera_depth_and_frame_gaps(tmp_path):
    cameras, world = _write_kubric(tmp_path / "scene.h5")
    clip = _kubric_adapter(tmp_path).load_clip("scene", [0, 5, 10, 0, 5, 10])
    assert clip.metadata["time_indices"] == [0, 5, 10]
    assert clip.metadata["pixel_coordinates"] == "array_index_v1"
    np.testing.assert_allclose(clip.depths, 2., atol=2e-6)
    np.testing.assert_allclose(clip.extrinsics[0], np.linalg.inv(cameras[0]), atol=1e-6)
    np.testing.assert_allclose(clip.intrinsics[0, :2, 2], [20.5, 13.5])
    np.testing.assert_allclose(clip.trajs_2d[0, 0], [10., 8.], atol=1e-5)
    sample = prepare_clip(clip, image_size=(28, 42), num_queries=40)
    assert sample["images"].shape == (6, 3, 28, 42)
    assert sample["time_idx"].tolist() == [0, 5, 10, 0, 5, 10]
    assert sample["query_time_idx"].tolist() == [0, 5, 10]
    target = sample["targets"]
    torch.testing.assert_close(target["c2w"][0], torch.eye(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(target["c2w"][3, :3, 3], torch.tensor([.2, 0., 0.]), atol=1e-6, rtol=0)
    np.testing.assert_allclose(target["K"][0, :2, 2], [20.5, 13.5])
    valid = target["sparse_query_valid"]
    ids = target["sparse_track_id"][valid].numpy()
    inverse = np.linalg.inv(cameras[0])
    expected = world[ids][:, [0, 5, 10]] @ inverse[:3, :3].T + inverse[:3, 3]
    np.testing.assert_allclose(target["sparse_trajectory"][valid], expected, atol=2e-6)
    occluded = valid & (target["sparse_track_id"] == 0)
    assert occluded.any()
    assert target["sparse_trajectory_valid"][occluded, 1].all()
    assert not target["sparse_visibility"][occluded, 1].any()
    assert target["sparse_visibility_valid"][occluded, 1].all()
    assert not target["sparse_trajectory_valid"][~valid].any()
    assert not target["sparse_visibility_valid"][~valid].any()


def test_mugen_uses_dynamic_authoritative_tracks_and_unknown_visibility(tmp_path):
    path = tmp_path / "pointodyssey.h5"
    cameras, world = _write_mugen(path)
    adapter = _mugen_adapter(path)
    repeated = adapter.load_clip("moving", [10, 0, 10])
    np.testing.assert_array_equal(repeated.trajs_3d_world[0], repeated.trajs_3d_world[2])
    np.testing.assert_allclose(repeated.trajs_3d_world[1], world[0])
    clip = adapter.load_clip("moving", [0, 5, 10])
    assert clip.metadata["track_source"] == "precomputed_h5"
    sample = prepare_clip(clip, image_size=(28, 42), num_queries=32)
    target = sample["targets"]
    valid = target["sparse_query_valid"]
    ids = target["sparse_track_id"][valid].numpy()
    inverse = np.linalg.inv(cameras[0])
    expected = world[[0, 5, 10]][:, ids].transpose(1, 0, 2) @ inverse[:3, :3].T + inverse[:3, 3]
    np.testing.assert_allclose(target["sparse_trajectory"][valid], expected, atol=2e-6)
    assert torch.any(target["sparse_displacement"][valid].abs() > .01)
    unknown = valid & (target["sparse_track_id"] == 1)
    assert unknown.any()
    assert target["sparse_trajectory_valid"][unknown, 1].all()
    assert not target["sparse_visibility_valid"][unknown, 1].any()


def test_required_authoritative_tracks_do_not_fall_back_to_static(tmp_path):
    path = tmp_path / "no_tracks.h5"
    _write_mugen(path, tracks=False)
    with pytest.raises(RuntimeError, match="precomputed tracks"):
        _mugen_adapter(path).load_clip("moving", [0, 5, 10])


def _spawn_read_mugen(adapter, result):
    try:
        clip = adapter.load_clip("moving", [0, 5, 10])
        result.put((clip.num_frames, float(clip.trajs_3d_world[2, 0, 0]), None))
    except Exception as exc:  # Propagate child failure without hanging parent.
        result.put((0, 0., repr(exc)))


def test_mugen_open_h5_adapter_is_spawn_picklable(tmp_path):
    path = tmp_path / "moving.h5"
    _, world = _write_mugen(path)
    adapter = _mugen_adapter(path)
    adapter.load_clip("moving", [0])  # The parent owns an already-open H5 handle.
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_spawn_read_mugen, args=(adapter, queue))
    process.start()
    try:
        frames, x, error = queue.get(timeout=25)
        process.join(timeout=10)
        assert process.exitcode == 0
        assert error is None
        assert frames == 3
        assert x == pytest.approx(float(world[10, 0, 0]))
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        queue.close()
        queue.join_thread()


def test_h5_slice_reads_only_requested_frames_and_rejects_bounds(tmp_path):
    with h5py.File(tmp_path / "frames.h5", "w") as handle:
        array = np.arange(24).reshape(6, 4)
        handle.create_dataset("frames", data=array, chunks=(1, 4), compression="gzip")
        got = h5_read_frame_slice(handle, [5, 1, 5])
        np.testing.assert_array_equal(got["frames"], array[[5, 1, 5]])
        with pytest.raises(IndexError):
            h5_read_frame_slice(handle, [6])


def test_native_registry_rejects_silently_misspelled_options(tmp_path):
    assert {"kubric_multicam", "mugen_h5", "pointodyssey", "waymo"}.issubset(list_datasets())
    with pytest.raises(TypeError, match="(?i)unknown|unsupported"):
        create_adapter("kubric_multicam", root=str(tmp_path), pixel_coordinate="array_index_v1")
