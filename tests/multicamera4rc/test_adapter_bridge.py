"""Coordinate, time, point-identity and visibility contracts of the bridge."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from multicamera4rc.datasets.adapters.base import UnifiedClip
from multicamera4rc.datasets.bridge import prepare_clip


def _clip():
    height, width = 28, 42
    camera_ids = [9, 4, 9, 4, 9]
    times = [0, 0, 5, 5, 10]
    intrinsic = np.array([[30., 0., 20.5], [0., 30., 13.5], [0., 0., 1.]], np.float32)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], np.float32)
    ref = np.eye(4, dtype=np.float32)
    ref[:3, :3], ref[:3, 3] = rotation, [4., 2., -1.]
    other = ref.copy()
    other[:3, 3] += rotation @ np.array([.2, 0., 0.], np.float32)
    world, uv, extrinsics = [], [], []
    for camera_id, time in zip(camera_ids, times):
        u = np.array([10., 18., 25.], np.float32) + .15*time
        if time == 10:
            u[2] = 100.  # A defined point leaves every source image.
        v = np.array([8., 14., 20.], np.float32)
        reference_xyz = np.stack(((u-20.5)*2./30., (v-13.5)*2./30., np.full(3, 2.)), axis=-1)
        xyz = reference_xyz @ rotation.T + ref[:3, 3]
        w2c = np.linalg.inv(ref if camera_id == 9 else other).astype(np.float32)
        camera_xyz = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        world.append(xyz)
        uv.append(camera_xyz[:, :2] / camera_xyz[:, 2:] * 30. + intrinsic[:2, 2])
        extrinsics.append(w2c)
    visible = np.ones((5, 3), bool)
    visible[2:4, 0] = False  # Ground truth remains defined through occlusion.
    visible[4, 2] = False
    known = np.ones((5, 3), bool)
    known[3, 1] = False
    return UnifiedClip(
        dataset_name="fixture", sequence_name="scene", frame_paths=None,
        images=[np.full((height, width, 3), 100, np.uint8) for _ in times],
        depths=[np.full((height, width), 2., np.float32) for _ in times],
        normals=None, trajs_2d=np.array(uv, np.float32),
        trajs_3d_world=np.array(world, np.float32), valids=np.ones((5, 3), bool),
        visibs=visible, visibility_valids=known,
        intrinsics=np.repeat(intrinsic[None], 5, axis=0),
        extrinsics=np.array(extrinsics, np.float32),
        metadata={"extrinsics_convention": "w2c", "pixel_coordinates": "array_index_v1",
                  "provided_slot_layout": True, "slot_physical_camera_id": camera_ids,
                  "slot_camera": [1, 0, 1, 0, 1], "slot_time_id": [0, 0, 1, 1, 2],
                  "slot_time": [0., 0., 5/30., 5/30., 10/30.], "fps": 30.,
                  "time_indices": [0, 5, 10], "valid_slot": [True]*5},
    )


def test_bridge_references_all_world_geometry_to_first_camera():
    clip = _clip()
    sample = prepare_clip(clip, image_size=(28, 42), num_queries=32)
    target = sample["targets"]
    assert sample["camera_id"].tolist() == [9, 4, 9, 4, 9]
    assert sample["time_idx"].tolist() == [0, 0, 5, 5, 10]
    torch.testing.assert_close(target["c2w"][0], torch.eye(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(target["c2w"][1, :3, 3], torch.tensor([.2, 0., 0.]), atol=1e-6, rtol=0)
    valid = target["sparse_query_valid"]
    ids = target["sparse_track_id"][valid].numpy()
    raw = clip.trajs_3d_world[[0, 2, 4]][:, ids].transpose(1, 0, 2)
    ref = clip.extrinsics[0]
    expected = raw @ ref[:3, :3].T + ref[:3, 3]
    np.testing.assert_allclose(target["sparse_trajectory"][valid], expected, atol=2e-6)
    queries = target["sparse_queries"][valid]
    source_slots = queries[:, 0].long().numpy()
    np.testing.assert_allclose(queries[:, 1:], clip.trajs_2d[source_slots, ids], atol=2e-6)
    expected_base = clip.trajs_3d_world[source_slots, ids] @ ref[:3, :3].T + ref[:3, 3]
    np.testing.assert_allclose(target["sparse_displacement"][valid], expected - expected_base[:, None], atol=2e-6)


def test_rectangular_resize_preserves_half_pixel_projection_convention():
    clip = _clip()
    sample = prepare_clip(clip, image_size=(56, 84), num_queries=32)
    target = sample["targets"]
    torch.testing.assert_close(target["K"][0], torch.tensor([[60., 0., 41.5], [0., 60., 27.5], [0., 0., 1.]]))
    valid = target["sparse_query_valid"]
    ids = target["sparse_track_id"][valid].numpy()
    slots = target["sparse_queries"][valid, 0].long().numpy()
    expected = (clip.trajs_2d[slots, ids] + .5) * 2. - .5
    np.testing.assert_allclose(target["sparse_queries"][valid, 1:], expected, atol=3e-6)
    assert target["depth"].shape == (5, 56, 84)


def test_native_padding_preserves_pixels_calibration_depth_and_track_coordinates():
    clip = _clip()
    h, w = clip.images[0].shape[:2]
    patterned = (np.arange(h*w*3).reshape(h, w, 3) % 251).astype(np.uint8)
    clip.images = [patterned.copy() for _ in clip.images]
    native = prepare_clip(clip, image_size=(h, w), num_queries=32, seed=7)
    padded = prepare_clip(clip, image_size=(h+14, w+14), image_mode="pad", num_queries=32, seed=7)
    torch.testing.assert_close(padded["images"][..., :h, :w], native["images"], atol=0, rtol=0)
    assert not padded["images"][..., h:, :].any()
    assert not padded["images"][..., :, w:].any()
    for name in ("K", "c2w", "sparse_queries", "sparse_track_id",
                 "sparse_query_valid", "sparse_trajectory", "sparse_displacement",
                 "sparse_trajectory_valid", "sparse_visibility", "sparse_visibility_valid"):
        torch.testing.assert_close(padded["targets"][name], native["targets"][name], atol=0, rtol=0)
    target = padded["targets"]
    torch.testing.assert_close(target["depth"][..., :h, :w], native["targets"]["depth"], atol=0, rtol=0)
    assert target["depth_valid"][..., :h, :w].all()
    assert not target["depth_valid"][..., h:, :].any()
    assert not target["depth_valid"][..., :, w:].any()


def test_native_padding_cannot_create_a_source_query_outside_the_original_image():
    clip = _clip()
    # Place this track in the padded strip but outside the actual native image,
    # even while its supplied visibility claims it is visible.
    clip.trajs_2d[:, 2] = [45., 20.]
    clip.visibs[:, 2] = True
    sample = prepare_clip(clip, image_size=(42, 56), image_mode="pad", num_queries=32)
    valid = sample["targets"]["sparse_query_valid"]
    assert valid.any()
    assert not (sample["targets"]["sparse_track_id"][valid] == 2).any()


def test_native_padding_rejects_a_canvas_smaller_than_the_source():
    with pytest.raises(ValueError, match="smaller than native"):
        prepare_clip(_clip(), image_size=(28, 28), image_mode="pad")


def test_explicit_legacy_pixel_origin_is_converted_exactly_once():
    array = _clip()
    legacy = copy.deepcopy(array)
    legacy.intrinsics[:, :2, 2] += .5
    legacy.trajs_2d += .5
    legacy.metadata["pixel_coordinates"] = "legacy_h5"
    expected = prepare_clip(array, image_size=(56, 84), num_queries=32, seed=7)
    actual = prepare_clip(legacy, image_size=(56, 84), num_queries=32, seed=7)
    for name in ("K", "c2w", "sparse_queries", "sparse_trajectory", "sparse_displacement"):
        torch.testing.assert_close(actual["targets"][name], expected["targets"][name], atol=3e-6, rtol=1e-6)


def test_occlusion_out_of_view_and_unknown_visibility_are_distinct_masks():
    sample = prepare_clip(_clip(), image_size=(28, 42), num_queries=32)
    target = sample["targets"]
    valid = target["sparse_query_valid"]
    occluded = valid & (target["sparse_track_id"] == 0)
    assert occluded.any()
    assert target["sparse_trajectory_valid"][occluded, 1].all()
    assert target["sparse_visibility_valid"][occluded, 1].all()
    assert not target["sparse_visibility"][occluded, 1].any()
    unknown = valid & (target["sparse_track_id"] == 1)
    # Visibility C axis is sorted physical cameras: [4,9].
    assert not target["sparse_visibility_valid"][unknown, 1, 0].any()
    assert target["sparse_visibility_valid"][unknown, 1, 1].all()
    missing = target["sparse_visibility_valid"][valid, 2, 0]
    assert not missing.any()  # No camera 4 observation at frame 10.
    out_of_view = valid & (target["sparse_track_id"] == 2)
    assert target["sparse_trajectory_valid"][out_of_view, 2].all()
    assert not target["sparse_visibility"][out_of_view, 2, 1].any()


def test_same_point_disagreement_across_synchronized_cameras_is_rejected():
    clip = _clip()
    clip.trajs_3d_world[1, 1, 0] += .2
    with pytest.raises(ValueError, match="(?i)world|inconsistent|agree"):
        prepare_clip(clip, image_size=(28, 42), num_queries=8)


def test_padding_has_no_fake_queries_or_supervision():
    sample = prepare_clip(_clip(), image_size=(28, 42), num_queries=64)
    target = sample["targets"]
    valid = target["sparse_query_valid"]
    assert 0 < valid.sum() < 64
    assert (target["sparse_track_id"][~valid] == -1).all()
    assert not target["sparse_trajectory_valid"][~valid].any()
    assert not target["sparse_displacement_valid"][~valid].any()
    assert not target["sparse_visibility_valid"][~valid].any()
    active = target["sparse_queries"][valid]
    assert torch.unique(active, dim=0).shape[0] == active.shape[0]


def test_query_time_subset_is_reordered_without_changing_source_anchor():
    full = prepare_clip(_clip(), image_size=(28, 42), num_queries=32, seed=9)
    subset = prepare_clip(_clip(), image_size=(28, 42), num_queries=32, seed=9, query_time_idx=[10, 5])
    assert subset["query_time_idx"].tolist() == [10, 5]
    torch.testing.assert_close(subset["targets"]["sparse_queries"], full["targets"]["sparse_queries"])
    for name in ("sparse_trajectory", "sparse_displacement", "sparse_trajectory_valid", "sparse_visibility", "sparse_visibility_valid"):
        torch.testing.assert_close(subset["targets"][name], full["targets"][name][:, [2, 1]])


def test_invalid_first_slot_is_removed_before_reference_selection():
    clip = _clip()
    clip.metadata["valid_slot"][0] = False
    sample = prepare_clip(clip, image_size=(28, 42), num_queries=16)
    assert sample["images"].shape[0] == 4
    assert sample["camera_id"][0].item() == 4
    assert sample["observation_valid"].all()
    torch.testing.assert_close(sample["targets"]["c2w"][0], torch.eye(4), atol=1e-6, rtol=0)


def test_unobserved_target_frame_is_rejected():
    with pytest.raises(ValueError, match="(?i)observed|present|time"):
        prepare_clip(_clip(), image_size=(28, 42), num_queries=16, query_time_idx=[7])


def test_missing_tracks_can_train_geometry_without_fake_dynamic_labels():
    clip = _clip()
    clip.trajs_2d = clip.trajs_3d_world = clip.valids = clip.visibs = clip.visibility_valids = None
    sample = prepare_clip(clip, image_size=(28, 42), num_queries=16)
    target = sample["targets"]
    assert target["depth_valid"].all() and target["camera_valid"].all()
    # Either omit sparse targets, or return masked padding; never invent tracks.
    if "sparse_queries" in target:
        assert not target["sparse_query_valid"].any()
        assert not target["sparse_trajectory_valid"].any()


@pytest.mark.parametrize("pose_convention", ["world_to_camera", "camera_to_world"])
def test_upstream_long_pose_convention_names_are_supported(pose_convention):
    # Waymo and TartanAir publish only pose_convention='world_to_camera'.
    clip = _clip()
    expected = prepare_clip(clip, image_size=(28, 42), num_queries=16, seed=1)
    del clip.metadata["extrinsics_convention"]
    clip.metadata["pose_convention"] = pose_convention
    if pose_convention == "camera_to_world":
        clip.extrinsics = np.linalg.inv(clip.extrinsics).astype(np.float32)
    actual = prepare_clip(clip, image_size=(28, 42), num_queries=16, seed=1)
    torch.testing.assert_close(actual["targets"]["c2w"], expected["targets"]["c2w"], atol=1e-6, rtol=0)
    torch.testing.assert_close(actual["targets"]["sparse_trajectory"], expected["targets"]["sparse_trajectory"], atol=2e-6, rtol=0)


def test_unknown_motion_static_map_has_only_source_time_supervision():
    clip = _clip()
    clip.metadata["has_temporal_supervision"] = False
    sample = prepare_clip(clip, image_size=(28, 42), num_queries=32)
    target = sample["targets"]
    active = target["sparse_query_valid"]
    sources = target["sparse_queries"][active, 0].long()
    source_time = sample["time_idx"][sources]
    expected = source_time[:, None] == sample["query_time_idx"][None]
    torch.testing.assert_close(target["sparse_trajectory_valid"][active], expected)
    assert not target["sparse_displacement"][active][expected].any()


def test_explicit_known_static_tracks_can_extend_static_map_supervision():
    clip = _clip()
    clip.metadata["has_temporal_supervision"] = False
    clip.track_is_dynamic = np.array([False, True, False])
    clip.track_dynamic_known = np.array([True, True, False])
    sample = prepare_clip(clip, image_size=(28, 42), num_queries=32)
    target = sample["targets"]
    static = target["sparse_query_valid"] & (target["sparse_track_id"] == 0)
    assert static.any()
    assert target["sparse_trajectory_valid"][static].all()
    assert not target["sparse_displacement"][static].any()
    other = target["sparse_query_valid"] & (target["sparse_track_id"] != 0)
    slots = target["sparse_queries"][other, 0].long()
    expected = sample["time_idx"][slots, None] == sample["query_time_idx"][None]
    torch.testing.assert_close(target["sparse_trajectory_valid"][other], expected)
