"""Native stereo identity and particle supervision, with analytic motion GT."""
import io
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from test_partial_dense import clip
from multicamera4rc import MultiCamera4RC, MultiCamera4RCConfig
from multicamera4rc.datasets.adapters.stereo import join_stereo, stereo_times
from multicamera4rc.datasets.adapters.tartanair_v2 import decode_depth_png, pose_w2c
from multicamera4rc.datasets.adapters.dense_geometry import PointOdysseyGeometryAdapter, PointOdysseyTracksAdapter
from multicamera4rc.datasets.bridge import prepare_clip
from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.datasets.mixture import collate_adapter_samples
from multicamera4rc.datasets.partial_dense import attach_partial_dense_recipe, vkitti_pairs


def particles():
    c = clip(T=3, shape=(28, 42)); S, N = 3, 7
    c.metadata["frame_indices"] = [1, 3, 7]
    uv = np.repeat(np.array([[[10., 10.], [12., 12.], [14., 14.], [16., 16.],
                              [18., 18.], [20., 20.], [22., 22.]]]), S, 0)
    uv[:, :, 0] += np.array([0., .4, 1.2])[:, None]
    xyz = np.concatenate((uv, np.ones((S, N, 1))), -1) @ np.linalg.inv(c.intrinsics[0]).T*2.
    visible = np.ones((S, N), bool); visible[1, 0] = False
    return replace(c, trajs_2d=uv, trajs_3d_world=xyz, valids=np.ones((S, N), bool), visibs=visible)


def test_all_native_tracks_retain_occluded_xyz_and_shared_scale():
    c = particles()
    s = prepare_clip(c, image_mode="native", num_queries=0, all_track_queries=True)
    assert len(s["targets"]["sparse_queries"]) == 20  # Every visible source; no quota.
    attach_partial_dense_recipe(c, s, kind="pointodyssey_tracks", image_mode="native")
    b = collate_adapter_samples([s]); target = materialize_dense_targets(b, torch.tensor([[0, 1, 2]]))
    assert target["depth_valid"].sum() == 3*28*42
    assert not target["displacement_valid"].any()  # No dense motion fabricated.
    ids = target["sparse_track_id"][0]
    src = target["sparse_queries"][0, :, 0].long()
    expected = torch.tensor(c.trajs_3d_world.transpose(1, 0, 2)[ids], dtype=torch.float32)
    torch.testing.assert_close(target["sparse_trajectory"][0], expected/target["scene_scale"][0])
    occluded = (ids == 0) & (src == 0)
    assert target["sparse_trajectory_valid"][0, occluded, 1].all()
    assert not target["sparse_visibility"][0, occluded, 1].any()
    assert target["sparse_visibility_valid"][0, occluded, 1].all()


def test_variable_native_track_counts_collate_without_dropping_labels():
    samples = []
    for n in (7, 4):
        c = particles()
        for k in ("trajs_2d", "trajs_3d_world", "valids", "visibs"):
            setattr(c, k, getattr(c, k)[:, :n])
        s = prepare_clip(c, image_mode="native", num_queries=0, all_track_queries=True)
        attach_partial_dense_recipe(c, s, kind="pointodyssey_tracks", image_mode="native"); samples.append(s)
    b = collate_adapter_samples(samples)
    assert b["targets"]["sparse_queries"].shape == (2, 20, 3)
    assert b["targets"]["sparse_query_valid"].sum(1).tolist() == [20, 11]
    assert not b["targets"]["sparse_trajectory_valid"][1, 11:].any()


def test_stereo_same_frame_has_unique_camera_and_common_world_gauge():
    left = clip(); right = clip(); right.extrinsics[:, 0, 3] = -.25
    both = join_stereo(left, right, ("left", "right"))
    s = prepare_clip(both, image_mode="native", num_queries=0)
    assert s["camera_id"].tolist() == [0, 0, 0, 1, 1, 1]
    assert s["time_idx"].tolist() == [0, 1, 2, 0, 1, 2]
    assert s["query_time_idx"].tolist() == [0, 1, 2]
    torch.testing.assert_close(s["targets"]["c2w"][3, :3, 3], torch.tensor([.25, 0., 0.]))
    assert stereo_times([1, 3, 8, 1, 3, 8]) == [1, 3, 8]
    for bad in ([0], [0, 1], [0, 0, 0, 0]):
        with pytest.raises(ValueError): stereo_times(bad)


def test_vkitti_pair_matching_never_crosses_camera_streams(tmp_path):
    left = clip(T=2); right = clip(T=2); right.extrinsics[:, 0, 3] = -.25
    both = join_stereo(left, right, ("Camera_0", "Camera_1"))
    h, w = left.depths[0].shape
    for camera, velocity in (("Camera_0", .1), ("Camera_1", .2)):
        for f, step, d, prefix, fd, fp in ((0, 1, "forwardSceneFlow", "sceneFlow", "forwardFlow", "flow"),
                (1, -1, "backwardSceneFlow", "backwardSceneFlow", "backwardFlow", "backwardFlow")):
            sf = np.zeros((h, w, 3)); sf[..., 0] = step*velocity
            raw = np.rint((sf/10+1)*.5*65535).astype(np.uint16)[..., ::-1].copy()
            flow = np.zeros((h, w, 3), np.uint16); flow[..., 0] = 1
            flow[..., 2] = np.rint((step*velocity*10/(w-1)+1)*.5*65535)
            flow[..., 1] = 32768
            for directory, pr, a in ((d, prefix, raw), (fd, fp, flow)):
                p = tmp_path/"frames"/directory/camera; p.mkdir(parents=True, exist_ok=True)
                assert cv2.imwrite(str(p/f"{pr}_{f:05d}.png"), a)
    pairs = vkitti_pairs(both, SimpleNamespace(seq_to_dir={"scene": tmp_path}))
    assert pairs[0][0][0] == 1 and pairs[2][0][0] == 3
    np.testing.assert_allclose(pairs[0][0][1][..., 0], .1, atol=.001)
    np.testing.assert_allclose(pairs[2][0][1][..., 0], .2, atol=.001)


def test_v2_lossless_depth_byte_order_and_ned_camera_pose():
    depth = np.linspace(.1, 100., 640*640, dtype=np.float32).reshape(640, 640)
    rgba = depth[..., None].view(np.uint8)
    okay, png = cv2.imencode(".png", rgba); assert okay
    np.testing.assert_array_equal(decode_depth_png(png.tobytes()), depth)
    p = np.array([3., 4., 5., 0., 0., 0., 1.])
    E = pose_w2c(p)
    np.testing.assert_allclose(E[:3, :3] @ np.array([1., 2., 3.]), [2., 3., 1.])
    np.testing.assert_allclose(E @ np.r_[p[:3], 1.], [0., 0., 0., 1.])
    assert np.linalg.det(E[:3, :3]) == 1


@pytest.mark.parametrize("native3d", [True, False])
def test_particle_adapter_world_basis_and_missing_3d_visibility(monkeypatch, native3d):
    c = particles()
    c.metadata["native_world_handedness"] = "left"
    annotations = dict(trajs_2d=c.trajs_2d.copy(), trajs_3d=c.trajs_3d_world.copy() if native3d else np.array(None),
                       valids=c.valids.copy(), visibs=c.visibs.copy())
    if native3d: annotations["trajs_3d"][..., 0] *= -1
    base = replace(c, trajs_2d=None, trajs_3d_world=None, valids=None, visibs=None)
    monkeypatch.setattr(PointOdysseyGeometryAdapter, "load_clip", lambda *a, **k: base)
    adapter = PointOdysseyTracksAdapter.__new__(PointOdysseyTracksAdapter)
    adapter._load_anno = lambda *a, **k: annotations
    loaded = adapter.load_clip("scene", [0, 1, 2])
    if native3d:
        np.testing.assert_allclose(loaded.trajs_3d_world, c.trajs_3d_world, atol=1e-6)
        assert loaded.valids[1, 0] and loaded.metadata["hidden_3d_supervision"]
    else:
        np.testing.assert_allclose(loaded.trajs_3d_world[loaded.valids], c.trajs_3d_world[loaded.valids], atol=1e-6)
        assert not loaded.valids[1, 0] and not loaded.metadata["hidden_3d_supervision"]
    assert loaded.visibility_valids[1, 0] and not loaded.visibs[1, 0]


def test_depth_lifted_particles_reject_surface_boundaries(monkeypatch):
    c = particles(); c.metadata["native_world_handedness"] = "right"
    for d in c.depths: d[:, 11:] = 8.
    annotations = dict(trajs_2d=c.trajs_2d.copy(), trajs_3d=np.array(None), valids=c.valids, visibs=c.visibs)
    monkeypatch.setattr(PointOdysseyGeometryAdapter, "load_clip", lambda *a, **k: c)
    adapter = PointOdysseyTracksAdapter.__new__(PointOdysseyTracksAdapter)
    adapter._load_anno = lambda *a, **k: annotations
    loaded = adapter.load_clip("scene", [0, 1, 2])
    assert not loaded.valids[0, 0]  # u=10 touches a 2m/8m depth boundary.
    assert loaded.visibility_valids[0, 0]  # Visibility label is still known.


def test_inconsistent_visible_native_xyz_is_masked_as_source_and_target(monkeypatch):
    c = particles(); c.metadata["native_world_handedness"] = "right"
    xyz = c.trajs_3d_world.copy(); xyz[2, 0, 0] += 10.
    annotations = dict(trajs_2d=c.trajs_2d.copy(), trajs_3d=xyz, valids=c.valids, visibs=c.visibs)
    monkeypatch.setattr(PointOdysseyGeometryAdapter, "load_clip", lambda *a, **k: c)
    adapter = PointOdysseyTracksAdapter.__new__(PointOdysseyTracksAdapter)
    adapter._load_anno = lambda *a, **k: annotations
    loaded = adapter.load_clip("scene", [0, 1, 2])
    assert not loaded.valids[2, 0] and not loaded.metadata["track_source_valid"][2, 0]
    assert loaded.valids[1, 0] and not loaded.visibs[1, 0]  # Native hidden XYZ retained.
    assert loaded.visibility_valids[2, 0]  # Annotation validity is independent.
    s = prepare_clip(loaded, image_mode="native", num_queries=0, all_track_queries=True)
    ids = s["targets"]["sparse_track_id"]
    source = s["targets"]["sparse_queries"][:, 0].long()
    assert not s["targets"]["sparse_trajectory_valid"][(ids == 0) & (source == 0), 2].any()


@pytest.mark.parametrize("visibility", [True, False])
def test_dense_head_checkpoint_preserves_outputs_and_every_parameter_gradient(visibility):
    torch.set_num_threads(1); torch.manual_seed(123)
    a = MultiCamera4RC(MultiCamera4RCConfig.tiny(gradient_checkpointing=False))
    b = MultiCamera4RC(MultiCamera4RCConfig.tiny(gradient_checkpointing=True))
    b.load_state_dict(a.state_dict())
    images = torch.rand(1, 6, 3, 28, 42)
    cameras = torch.tensor([[0, 0, 0, 1, 1, 1]]); times = torch.tensor([[2, 5, 9, 2, 5, 9]])
    pa = a(images, cameras, times, source_obs_idx=[0, 4], return_visibility=visibility)
    pb = b(images, cameras, times, source_obs_idx=[0, 4], return_visibility=visibility)
    keys = ["depth", "depth_conf", "ray", "ray_conf", "camera_encoding",
            "trajectory", "displacement", "motion_confidence"]
    if visibility: keys.append("visibility_logits")
    for k in keys: torch.testing.assert_close(pa[k], pb[k], atol=1e-6, rtol=1e-5)
    la = sum(pa[k].square().mean() for k in keys); lb = sum(pb[k].square().mean() for k in keys)
    la.backward(); lb.backward()
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None: torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=1e-4, msg=name)
