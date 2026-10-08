"""Renderer-derived dense labels, scale/gauge consistency and full-pixel losses."""

import numpy as np
import json
import pytest
import torch

from multicamera4rc.datasets.adapters.base import UnifiedClip
from multicamera4rc.datasets.bridge import prepare_clip
from multicamera4rc.datasets.dense import attach_kubric_dense_recipe, materialize_dense_targets
from multicamera4rc.datasets.mixture import collate_adapter_samples
from multicamera4rc.losses import MultiCamera4RCLoss


def _scene(tmp_path, *, rotate=False, include_visibility=True):
    h5py = pytest.importorskip("h5py")
    h, w = 27, 39
    times = [0, 2, 5]
    K = np.array([[20., 0., 19.], [0., 20., 13.], [0., 0., 1.]], np.float32)
    R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], np.float32)
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    c2w[:, :3, :3] = R
    c2w[:, :3, 3] = [[1., 2., 0.], [1., 2.1, 0.]]
    poses = np.zeros((1, 6, 7), np.float32)
    poses[0, :, :3] = [[1., 2.+.2*t, 2.] for t in range(6)]
    poses[0, :, 5:] = np.sqrt(.5)
    if rotate:
        poses[0, 5, 5:] = [1., 0.]
    segmentation = np.zeros((6, 2, h, w), np.int32)
    for t in range(6):
        for c in range(2):
            segmentation[t, c, 4:10, 8+2*t-c:12+2*t-c] = 1
    path = tmp_path/"dense.h5"
    with h5py.File(path, "w") as f:
        f["frames"] = np.full((6, 2, h, w, 3), 100, np.uint8)
        renderer_K = np.repeat(K[None], 2, axis=0)
        renderer_K[:, :2, 2] += .5
        f["intrinsics"] = renderer_K
        f["extrinsics"] = c2w
        f["camera_ids"] = np.array(["camera0", "camera1"], dtype=h5py.string_dtype())
        f["segmentation"] = segmentation
        f["depth"] = np.full((6, 2, h, w), 2., np.float32)
        f["object_trajectories"] = poses
        f["object_is_dynamic"] = [True]
    slots = [(c, t) for c in range(2) for t in times]
    clip = UnifiedClip(
        dataset_name="kubric_multicam", sequence_name="analytic", frame_paths=None,
        images=[np.full((h, w, 3), 100, np.uint8) for _ in slots],
        depths=[np.full((h, w), 2., np.float32) for _ in slots], normals=None,
        trajs_2d=None, trajs_3d_world=None, valids=None, visibs=None,
        intrinsics=np.repeat(K[None], 6, axis=0),
        extrinsics=np.array([np.linalg.inv(c2w[c]) for c, t in slots]),
        metadata={"h5_path": str(path), "extrinsics_convention": "w2c",
                  "pixel_coordinates": "array_index_v1", "time_indices": times,
                  "slot_time_id": [0, 1, 2]*2, "slot_physical_camera_id": [c for c, t in slots],
                  "valid_slot": [True]*6})
    sample = prepare_clip(clip, num_queries=0, image_mode="native")
    attach_kubric_dense_recipe(clip, sample, image_mode="native", include_visibility=include_visibility)
    return clip, sample, collate_adapter_samples([sample])


def test_dense_native_pixels_survive_non_square_images_larger_than_256(tmp_path):
    clip, _, _ = _scene(tmp_path)
    h, w = 270, 480
    pixels = (np.arange(h*w*3).reshape(h, w, 3) % 251).astype(np.uint8)
    clip.images = [pixels.copy() for _ in clip.images]
    clip.depths = [np.full((h, w), 2., np.float32) for _ in clip.images]
    sample = prepare_clip(clip, num_queries=0, image_size=None, image_mode="native")
    assert sample["images"].shape[-2:] == (280, 490)
    torch.testing.assert_close(sample["images"][0, :, :h, :w], torch.from_numpy(pixels).permute(2, 0, 1).float()/255, rtol=0, atol=0)
    torch.testing.assert_close(sample["targets"]["K"], torch.from_numpy(clip.intrinsics))
    assert not sample["targets"]["depth_valid"][..., h:, :].any()
    assert not sample["targets"]["depth_valid"][..., :, w:].any()


def test_fit_keeps_content_aspect_ratio_and_masks_letterbox(tmp_path):
    clip, _, _ = _scene(tmp_path)
    sample = prepare_clip(clip, num_queries=0, image_size=(56, 56), image_mode="fit")
    valid = sample["targets"]["depth_valid"]
    assert valid[..., :39, :].all()
    assert not valid[..., 39:, :].any()
    torch.testing.assert_close(sample["targets"]["K"][0, 0, 0], torch.tensor(20*56/39))
    torch.testing.assert_close(sample["targets"]["K"][0, 1, 1], torch.tensor(20*39/27))


@pytest.mark.parametrize("rotate", [False, True])
def test_dense_world_motion_is_analytic_and_same_point_agrees_across_cameras(tmp_path, rotate):
    _, sample, batch = _scene(tmp_path, rotate=rotate, include_visibility=False)
    target = materialize_dense_targets(batch, torch.tensor([[0, 3]]), normalize_scene=False)
    assert not any(k.startswith("sparse_") for k in sample["targets"])
    assert target["trajectory"].shape == (1, 2, 3, 28, 42, 3)
    assert target["trajectory_valid"].sum() == 2*3*27*39
    # These are the same physical surface point in the two source cameras.
    first = target["trajectory"][0, 0, :, 6, 10]
    second = target["trajectory"][0, 1, :, 6, 9]
    expected = torch.tensor([[-.9, -.7, 2.], [-.5, -.7, 2.], [.1, -.7, 2.]])
    if rotate:
        expected[2] = torch.tensor([1.7, -.9, 2.])
    torch.testing.assert_close(first, expected, atol=2e-6, rtol=0)
    torch.testing.assert_close(second, expected, atol=2e-6, rtol=0)
    assert not target["displacement"][:, :, 0].any()
    assert not target["trajectory_valid"][..., 27:, :].any()
    assert not target["trajectory_valid"][..., :, 39:].any()
    assert not target["displacement"][0, 0, :, 16, 16].any()


def test_one_scene_scale_applies_to_depth_camera_rays_and_motion(tmp_path):
    _, _, batch = _scene(tmp_path, include_visibility=False)
    source = torch.tensor([[0, 3]])
    raw = materialize_dense_targets(batch, source, normalize_scene=False)
    scaled = materialize_dense_targets(batch, source)
    s = scaled["scene_scale"][0]
    for k in ("depth", "trajectory", "displacement"):
        torch.testing.assert_close(scaled[k]*s, raw[k], atol=3e-6, rtol=1e-6)
    torch.testing.assert_close(scaled["c2w"][..., :3, 3]*s, raw["c2w"][..., :3, 3])
    torch.testing.assert_close(scaled["ray"][..., 3:]*s, raw["ray"][..., 3:])
    torch.testing.assert_close(scaled["ray"][..., :3], raw["ray"][..., :3])
    torch.testing.assert_close(raw["ray"][0, 0, 6, 10, :3], torch.tensor([-.45, -.35, 1.]), atol=1e-6, rtol=0)
    torch.testing.assert_close(raw["ray"][0, 3, 6, 10, 3:], torch.tensor([.1, 0., 0.]), atol=1e-6, rtol=0)
    native_xyz = scaled["ray"][..., 3:]+scaled["depth"][..., None]*scaled["ray"][..., :3]
    mean_range = torch.linalg.vector_norm(native_xyz, dim=-1)[scaled["depth_valid"]].mean()
    torch.testing.assert_close(mean_range, torch.tensor(1.), atol=2e-6, rtol=0)


def test_dense_occlusion_does_not_remove_world_position_supervision(tmp_path):
    _, _, batch = _scene(tmp_path)
    target = materialize_dense_targets(batch, torch.tensor([[0]]))
    # A source-background point is covered by the moving instance at time 5.
    assert target["visibility"][0, 0, 0, 0, 6, 19]
    assert not target["visibility"][0, 0, 2, 0, 6, 19]
    assert target["visibility_valid"][0, 0, 2, 0, 6, 19]
    assert target["trajectory_valid"][0, 0, 2, 6, 19]
    assert target["visibility"][0, 0, :, :, 6, 10].all()


def test_dense_loss_updates_a_pixel_outside_the_old_sparse_pool_and_logs_no_sparse_terms(tmp_path):
    _, _, batch = _scene(tmp_path)
    target = materialize_dense_targets(batch, torch.tensor([[0]]))
    pred = {k: target[k].clone().requires_grad_() for k in ("depth", "ray", "K", "c2w", "trajectory", "displacement")}
    pred.update(depth_conf=torch.ones_like(target["depth"]), ray_conf=torch.ones_like(target["depth"]),
                motion_confidence=torch.ones_like(target["trajectory_valid"], dtype=torch.float32),
                query_time_idx=batch["query_time_idx"], source_time_idx=torch.tensor([[0]]),
                observation_valid=batch["observation_valid"], source_valid=torch.tensor([[True]]),
                visibility_logits=torch.zeros_like(target["visibility"], dtype=torch.float32, requires_grad=True),
                projection_valid=torch.zeros_like(target["visibility"]))
    with torch.no_grad():
        pred["trajectory"][0, 0, 2, 6, 10, 0] += .3
        pred["displacement"][0, 0, 2, 6, 10, 0] += .3
    terms = MultiCamera4RCLoss.dense()(pred, target)
    assert not any(k.startswith("sparse_") for k in terms)
    assert terms["trajectory"] > 0 and terms["velocity"] > 0
    terms["loss"].backward()
    assert pred["trajectory"].grad[0, 0, 2, 6, 10].abs().sum() > 0
    assert pred["visibility_logits"].grad.abs().sum() > 0  # GT masks determine supervision.
    assert not pred["trajectory"].grad[..., 27:, :, :].any()
    assert all(torch.isfinite(v.grad).all() for v in pred.values() if isinstance(v, torch.Tensor) and v.grad is not None)


def test_velocity_uses_frame_gaps_and_sorts_target_times():
    times = torch.tensor([[5, 0, 2]])
    truth = torch.zeros(1, 1, 3, 2, 2, 3)
    truth[..., 0] = times[:, None, :, None, None].float()
    estimate = (3*truth).requires_grad_()
    pred = {"displacement": estimate, "query_time_idx": times, "source_time_idx": torch.tensor([[0]])}
    target = {"displacement": truth, "displacement_valid": torch.ones(truth.shape[:-1], dtype=torch.bool)}
    loss = MultiCamera4RCLoss(depth_weight=0, camera_weight=0, trajectory_weight=0,
                             displacement_weight=0, velocity_weight=1, regression_norm="l2")(pred, target)
    torch.testing.assert_close(loss["velocity"], torch.tensor(2.))
    torch.testing.assert_close(loss["loss"], torch.tensor(2.))
    loss["loss"].backward()
    assert estimate.grad.abs().sum() > 0


def _ground_and_dome_scene(tmp_path, *, policy="auto"):
    import h5py
    clip, _, _ = _scene(tmp_path)
    h, w = clip.images[0].shape[:2]
    angle = np.deg2rad(150.)
    R = np.array([[0., -np.cos(angle), np.sin(angle)],
                  [1., 0., 0.], [0., np.sin(angle), np.cos(angle)]], np.float32)
    yy, xx = np.mgrid[:h, :w]
    rays = np.stack(((xx-19)/20, (yy-13)/20, np.ones((h,w))), -1)
    directions = rays @ R.T
    with h5py.File(clip.metadata['h5_path'], 'a') as f:
        f.attrs['schema_version'] = 'movi_tracking_v3'
        f.attrs['coordinate_system'] = 'opencv_camera__blender_world'
        for slot in range(6):
            camera, time = slot//3, [0,2,5][slot%3]
            E = np.eye(4, dtype=np.float32); E[:3,:3]=R; E[:3,3]=[1.,2.+camera*.1,3.]
            clip.extrinsics[slot] = np.linalg.inv(E)
            depth = (-3/directions[...,2]).astype(np.float32)  # exact world ground z=0
            ids = f['segmentation'][time,camera]
            depth[ids>0] *= .6
            depth[:3] = 40.  # non-ground background, distinct from the true ground
            if slot == 0:
                depth[6,10] = 80.  # a labelled object must survive even at extreme distance
            clip.depths[slot] = depth
    sample = prepare_clip(clip, num_queries=0, image_mode='native')
    rgb = sample['images'].clone()
    attach_kubric_dense_recipe(clip, sample, image_mode='native', background_policy=policy)
    torch.testing.assert_close(sample['images'], rgb, rtol=0, atol=0)
    return clip, sample, collate_adapter_samples([sample])


def test_dome_mask_uses_world_plane_preserves_distant_objects_and_all_target_masks(tmp_path):
    _, _, batch = _ground_and_dome_scene(tmp_path)
    source = torch.tensor([[0,3]])
    target = materialize_dense_targets(batch, source)
    excluded = target['excluded_background']
    assert excluded.sum() == 6*3*39
    assert excluded[..., :3, :39].all()
    assert not target['depth_valid'][excluded].any()
    assert not target['ray_valid'][excluded].any()
    assert target['depth_valid'][0,0,6,10]  # labelled object at 80 m
    assert target['depth_valid'][0,0,16,16]  # actual ground
    assert not target['source_pixel_valid'][..., :3, :].any()
    assert not target['trajectory_valid'][..., :3, :].any()
    assert not target['displacement_valid'][..., :3, :].any()
    assert not target['static_mask'][..., :3, :].any()
    assert not target['visibility_valid'][..., :3, :].any()
    assert not target['visibility'][..., :3, :].any()
    assert not target['depth_valid'][...,27:,:].any()
    # Filtering happens before normalization; one scale still governs all tasks.
    raw = materialize_dense_targets(batch, source, normalize_scene=False)
    base = raw['ray'][...,3:]+raw['depth'][...,None]*raw['ray'][...,:3]
    expected = torch.linalg.vector_norm(base,dim=-1)[raw['depth_valid']].mean()
    torch.testing.assert_close(target['scene_scale'][0], expected)
    for key in ('depth','trajectory','displacement'):
        torch.testing.assert_close(target[key]*expected,raw[key],atol=2e-5,rtol=1e-6)
    torch.testing.assert_close(target['c2w'][...,:3,3]*expected,raw['c2w'][...,:3,3])
    assert batch['targets']['depth_valid'][0,0,0,0]  # materialization does not mutate raw targets


def test_dome_pixels_receive_no_gradient_from_any_dense_task(tmp_path):
    _, _, batch = _ground_and_dome_scene(tmp_path)
    target = materialize_dense_targets(batch, torch.tensor([[0,3]]))
    pred = {k:(target[k]+.01).detach().requires_grad_() for k in
            ('depth','ray','K','c2w','trajectory','displacement')}
    pred.update(depth_conf=torch.full_like(target['depth'],2.,requires_grad=True),
                ray_conf=torch.full_like(target['depth'],2.,requires_grad=True),
                motion_confidence=torch.full_like(target['trajectory_valid'],2.,dtype=torch.float32,requires_grad=True),
                query_time_idx=batch['query_time_idx'],source_time_idx=torch.tensor([[0,0]]),
                observation_valid=batch['observation_valid'],source_valid=torch.tensor([[True,True]]),
                visibility_logits=torch.zeros_like(target['visibility'],dtype=torch.float32,requires_grad=True),
                projection_valid=torch.ones_like(target['visibility']))
    loss = MultiCamera4RCLoss.dense()(pred,target)['loss']
    loss.backward()
    masks={'depth':target['depth_valid'],'ray':target['ray_valid'],
           'trajectory':target['trajectory_valid'],'displacement':target['displacement_valid'],
           'depth_conf':target['depth_valid'],'ray_conf':target['ray_valid'],
           'motion_confidence':target['trajectory_valid'],'visibility_logits':target['visibility_valid']}
    for key,valid in masks.items():
        assert torch.isfinite(pred[key].grad).all(), key
        assert not pred[key].grad[~valid].any(), key
        assert pred[key].grad[valid].abs().sum()>0, key


def test_explicit_legacy_policy_and_unknown_schema_are_recordable(tmp_path):
    clip, _, batch = _ground_and_dome_scene(tmp_path,policy='all')
    target = materialize_dense_targets(batch,torch.tensor([[0]]))
    assert not target['excluded_background'].any()
    assert target['depth_valid'].sum() == 6*27*39
    import h5py
    with h5py.File(clip.metadata['h5_path'],'a') as f:
        f.attrs['coordinate_system']='unknown'
    sample = prepare_clip(clip,num_queries=0,image_mode='native')
    with pytest.raises(ValueError,match='audited'):
        attach_kubric_dense_recipe(clip,sample,image_mode='native',background_policy='objects_ground')


@pytest.mark.parametrize("profile", ["auto", "dense_balanced"])
def test_real_adapter_dense_training_forward_backward_and_checkpoint(tmp_path, capsys, profile):
    from multicamera4rc.train import main as train_main
    _scene(tmp_path)
    config = {"image_mode": "native", "supervision": "dense", "num_queries": 0,
              "num_times": 3, "epoch_size": 2,
              "sources": [{"adapter": "kubric_multicam", "root": str(tmp_path),
                           "split": "all", "cameras": 2,
                           "options": {"depth_is_ray_distance": False, "verbose": False}}]}
    path = tmp_path/"dense.json"
    path.write_text(json.dumps(config))
    checkpoint = train_main(["--data-config", str(path), "--output", str(tmp_path/"train"),
                             "--preset", "tiny", "--device", "cpu", "--steps", "1",
                              "--source-count", "1", "--no-query-chunking", "--cpu-threads", "1", "--loss-profile", profile])
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["step"] == 1
    assert payload["training_state"]["supervision"] == "dense"
    assert payload["training_state"]["scene_normalized"]
    assert payload["training_state"]["loss_profile"] == ("dense" if profile == "auto" else profile)
    assert payload["training_state"]["loss_config"]
    records = [json.loads(s) for s in capsys.readouterr().out.splitlines() if s.startswith('{"step"')]
    assert records[0]["supervised_source_pixels"] == 27*39
    assert records[0]["supervised_trajectory_pixels"] == 3*27*39
    assert not any(k.startswith("sparse_") for k in records[0])
    if profile == "dense_balanced":
        resumed_path = train_main(["--data-config", str(path), "--output", str(tmp_path/"resumed"),
            "--resume", str(checkpoint), "--device", "cpu", "--steps", "2", "--source-count", "1",
            "--cpu-threads", "1", "--override-lr", "--lr", "3e-5", "--backbone-lr", "3e-6",
            "--lr-schedule", "cosine"])
        resumed = torch.load(resumed_path, weights_only=True)
        assert resumed["training_state"]["loss_config"] == payload["training_state"]["loss_config"]
        assert resumed["training_state"]["loss_profile"] == profile
        assert resumed["optimizer"]["param_groups"][1]["lr"] == 3e-5
        assert resumed["training_state"]["optimizer_updates"] == 2
