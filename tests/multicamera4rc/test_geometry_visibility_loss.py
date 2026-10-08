"""Numerical semantics and gradient boundaries for multi-camera geometry/losses."""

import math

import pytest
import torch

from multicamera4rc.geometry import ExtendedCameraDec, decode_camera, unproject_depth
from multicamera4rc.losses import MultiCamera4RCLoss
from multicamera4rc.tasks import sample_tracks
from multicamera4rc.visibility import VisibilityHead


def _visibility_inputs(height=4, width=6):
    c2w = torch.eye(4)[None, None, None].expand(1, 1, 2, 4, 4).clone().requires_grad_()
    K = torch.tensor([[3., 0., 3.], [0., 2., 2.], [0., 0., 1.]])[None, None, None]
    K = K.expand(1, 1, 2, 3, 3).clone().requires_grad_()
    trajectory = unproject_depth(torch.ones(1, 1, height, width), K[:, :, 0].detach(),
                                 c2w[:, :, 0].detach()).detach().requires_grad_()
    confidence = torch.full((1, 1, 2, height, width), 2., requires_grad=True)
    return dict(trajectory=trajectory,
                motion_features=torch.randn(1, 1, 16, height, width, requires_grad=True),
                target_features=torch.randn(1, 1, 2, 6, 64, requires_grad=True),
                target_depth=torch.ones(1, 1, 2, height, width, requires_grad=True),
                target_depth_conf=confidence, target_K=K, target_c2w=c2w,
                target_valid=torch.tensor([[[True, False]]]), patch_grid=(2, 3))


def test_c2w_translation_known_quaternion_roundtrip_and_gradients():
    height, width = 4, 6
    pose = torch.tensor([[[1., -2., 3., 0., 0., math.sqrt(.5), math.sqrt(.5),
                          math.pi/2, math.pi/2, .1, -.15]]], requires_grad=True)
    K, c2w = decode_camera(pose, height, width)
    torch.testing.assert_close(c2w[0, 0, :3, 3], torch.tensor([1., -2., 3.]))
    torch.testing.assert_close(c2w[0, 0, :3, :3],
                               torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]),
                               atol=1e-6, rtol=1e-6)
    depth = torch.full((1, 1, height, width), 2., requires_grad=True)
    world = unproject_depth(depth, K, c2w)
    camera = (world-c2w[..., :3, 3][:, :, None, None]) @ c2w[..., :3, :3]
    projected = camera @ K.transpose(-1, -2)[:, :, None]
    uv = projected[..., :2] / projected[..., 2:]
    v, u = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    torch.testing.assert_close(uv[0, 0], torch.stack((u, v), -1).float(), atol=1e-5, rtol=1e-5)
    world.square().mean().backward()
    assert torch.isfinite(pose.grad).all() and pose.grad[..., 9:11].abs().sum() > 0
    assert torch.isfinite(depth.grad).all() and depth.grad.abs().sum() > 0


def test_camera_head_checkpoint_names_zero_principal_and_zero_quaternion():
    head = ExtendedCameraDec(128)
    assert "backbone.0.weight" in head.state_dict()
    assert "fc_t.weight" in head.state_dict() and "fc_qvec.weight" in head.state_dict()
    assert "fc_fov.0.weight" in head.state_dict()
    assert torch.count_nonzero(head.fc_principal.weight) == 0
    assert torch.count_nonzero(head.fc_principal.bias) == 0
    assert head(torch.randn(1, 2, 128)).shape == (1, 2, 11)
    pose = torch.zeros(1, 1, 11)
    pose[..., 7:9] = 1
    K, c2w = decode_camera(pose, 4, 6)
    torch.testing.assert_close(c2w[..., :3, :3], torch.eye(3)[None, None])
    assert K[0, 0, 0, 2] == 3 and K[0, 0, 1, 2] == 2


def test_visibility_missing_camera_and_gradient_boundaries():
    torch.manual_seed(7)
    inputs = _visibility_inputs()
    output = VisibilityHead(16, 64, query_chunk_size=5)(**inputs)
    assert output["projection_valid"].dtype == torch.bool
    assert output["projection_valid"][:, :, 0].all()
    assert not output["projection_valid"][:, :, 1].any()
    assert not output["visibility"][:, :, 1].any()
    assert torch.isfinite(output["visibility_logits"]).all()
    output["visibility_logits"][:, :, 0].sum().backward()
    assert inputs["motion_features"].grad.abs().sum() > 0
    assert inputs["target_features"].grad[:, :, 0].abs().sum() > 0
    for name in ("trajectory", "target_K", "target_c2w", "target_depth", "target_depth_conf"):
        assert inputs[name].grad is None


def test_visibility_chunks_preserve_predictions():
    torch.manual_seed(19)
    head = VisibilityHead(16, 64, query_chunk_size=5)
    inputs = _visibility_inputs()
    small = head(**inputs)
    head.query_chunk_size = 100
    large = head(**inputs)
    torch.testing.assert_close(small["visibility_logits"], large["visibility_logits"])
    assert torch.equal(small["projection_valid"], large["projection_valid"])


def test_rotated_camera_border_pixels_remain_valid():
    height, width = 8, 12
    pose = torch.tensor([[[.2, -.7, 1.4, .3, -.2, .6, .7, 1.1, 1.2, .07, -.03]]])
    K, c2w = decode_camera(pose, height, width)
    world = unproject_depth(torch.ones(1, 1, height, width), K, c2w)
    output = VisibilityHead(16, 64, query_chunk_size=13)(
        world, torch.randn(1, 1, 16, height, width), torch.randn(1, 1, 1, 6, 64),
        torch.ones(1, 1, 1, height, width), torch.ones(1, 1, 1, height, width),
        K[:, :, None], c2w[:, :, None], torch.ones(1, 1, 1, dtype=torch.bool), (2, 3))
    assert output["projection_valid"].all()


def _motion_predictions():
    shape = (1, 1, 2, 4, 6)
    logits = torch.zeros(*shape[:3], 1, *shape[-2:], requires_grad=True)
    return {"trajectory": torch.zeros(*shape, 3, requires_grad=True),
            "motion_confidence": torch.full(shape, 2., requires_grad=True),
            "visibility_logits": logits, "visibility": torch.zeros_like(logits),
            "projection_valid": torch.ones_like(logits, dtype=torch.bool),
            "source_valid": torch.tensor([[True]]), "source_time_idx": torch.tensor([[0]]),
            "query_time_idx": torch.tensor([[0, 1]])}


def test_loss_supervises_occluded_3d_and_excludes_identity_confidence():
    pred = _motion_predictions()
    truth = torch.ones_like(pred["trajectory"])
    truth[:, :, 0] = 0
    labels = {"trajectory": truth, "trajectory_valid": torch.ones(truth.shape[:-1], dtype=torch.bool),
              "visibility": torch.zeros_like(pred["visibility_logits"]),
              "visibility_valid": torch.ones_like(pred["visibility_logits"], dtype=torch.bool)}
    loss = MultiCamera4RCLoss()(pred, labels)
    loss["loss"].backward()
    assert pred["motion_confidence"].grad[:, :, 0].abs().sum() == 0
    assert pred["motion_confidence"].grad[:, :, 1].abs().sum() > 0
    assert pred["trajectory"].grad[:, :, 1].abs().sum() > 0
    assert pred["visibility_logits"].grad.abs().sum() > 0


def test_loss_rejects_missing_or_all_invalid_supervision():
    pred = _motion_predictions()
    labels = {"trajectory": torch.ones_like(pred["trajectory"]),
              "trajectory_valid": torch.zeros(pred["trajectory"].shape[:-1], dtype=torch.bool)}
    for target in ({}, labels):
        with pytest.raises(ValueError, match="no valid"):
            MultiCamera4RCLoss()(pred, target)


def test_invalid_projection_is_not_a_visibility_negative():
    pred = _motion_predictions()
    pred["projection_valid"] = torch.zeros_like(pred["projection_valid"])
    with pytest.raises(ValueError, match="no valid"):
        MultiCamera4RCLoss()(pred, {"visibility": torch.zeros_like(pred["visibility_logits"]),
                                  "visibility_valid": torch.ones_like(pred["projection_valid"])})


def test_native_depth_nan_padding_is_excluded_before_error_arithmetic():
    estimate = torch.ones(1, 1, 2, 3, requires_grad=True)
    target = torch.ones(1, 1, 4, 6)
    target[0, 0, 0, 0] = float("nan")
    loss = MultiCamera4RCLoss()({"depth": estimate, "depth_conf": torch.full_like(estimate, 2.),
                               "observation_valid": torch.tensor([[True]])},
                              {"depth": target, "depth_valid": torch.isfinite(target)})
    assert torch.isfinite(loss["loss"])
    loss["loss"].backward()
    assert torch.isfinite(estimate.grad).all()


def test_depth_camera_only_training_scales_native_intrinsics():
    pose = torch.tensor([[[.2, -.7, 1.4, .3, -.2, .6, .7, 1.1, 1.2, .07, -.03]]])
    K, c2w = decode_camera(pose, 8, 12)
    prediction_K, prediction_camera = K.clone().requires_grad_(), c2w.clone().requires_grad_()
    pred = {"depth": torch.ones(1, 1, 8, 12, requires_grad=True), "K": prediction_K,
            "c2w": prediction_camera, "observation_valid": torch.ones(1, 1, dtype=torch.bool)}
    target = {"depth": torch.ones(1, 1, 16, 24),
              "K": K * torch.tensor([2., 2., 1.])[None, None, :, None], "c2w": c2w}
    loss = MultiCamera4RCLoss()(pred, target)
    assert loss["camera"] == 0 and loss["loss"] == 0
    loss["loss"].backward()
    assert prediction_K.grad is not None and prediction_camera.grad is not None


def _sparse_predictions():
    pred = _motion_predictions()
    pred["displacement"] = torch.zeros_like(pred["trajectory"], requires_grad=True)
    pred["source_obs_idx"] = torch.tensor([[5]])
    return pred


def test_sparse_supervision_gradients_occlusion_and_unselected_sources():
    pred = _sparse_predictions()
    queries = torch.tensor([[[5., 2., 1.], [7., 2., 1.]]])
    truth = torch.zeros(1, 2, 2, 3)
    truth[:, 0, 1] = 1
    truth[:, 1] = 1000  # Unselected source must contribute neither error nor gradient.
    labels = {"sparse_queries": queries, "sparse_trajectory": truth,
              "sparse_trajectory_valid": torch.ones(1, 2, 2, dtype=torch.bool),
              "sparse_visibility": torch.tensor([[[[1.], [0.]], [[0.], [0.]]]]),
              "sparse_visibility_valid": torch.ones(1, 2, 2, 1, dtype=torch.bool)}
    loss = MultiCamera4RCLoss()(pred, labels)
    trimmed = {key: (value[:, :1] if key.startswith("sparse_") else value)
               for key, value in labels.items()}
    single = MultiCamera4RCLoss()(pred, trimmed)
    torch.testing.assert_close(loss["loss"], single["loss"])
    loss["loss"].backward()
    assert pred["trajectory"].grad[0, 0, 1, 1, 2].abs().sum() > 0
    assert pred["motion_confidence"].grad[:, :, 0].abs().sum() == 0
    assert pred["motion_confidence"].grad[0, 0, 1, 1, 2].abs() > 0
    assert pred["visibility_logits"].grad[0, 0, 1, 0, 1, 2].abs() > 0
    # These GT-occluded tracks receive 3D supervision although pred visibility=0.
    assert not pred["visibility"].any()
    assert pred["trajectory"].grad.abs().sum() > 0


def test_sparse_displacement_uses_trajectory_validity_and_confidence():
    pred = _sparse_predictions()
    truth = torch.zeros(1, 1, 2, 3)
    truth[:, :, 1] = 1
    result = MultiCamera4RCLoss()(pred, {
        "sparse_queries": torch.tensor([[[5., 2., 1.]]]), "sparse_displacement": truth,
        "sparse_trajectory_valid": torch.ones(1, 1, 2, dtype=torch.bool)})
    result["loss"].backward()
    assert result["sparse_displacement"] > 0
    assert pred["displacement"].grad[0, 0, 1, 1, 2].abs().sum() > 0
    assert pred["motion_confidence"].grad[:, :, 0].abs().sum() == 0


def test_sparse_all_unselected_or_invalid_projection_rejects_fake_supervision():
    pred = _sparse_predictions()
    with pytest.raises(ValueError, match="no valid"):
        MultiCamera4RCLoss()(pred, {
            "sparse_queries": torch.tensor([[[7., 2., 1.]]]),
            "sparse_trajectory": torch.ones(1, 1, 2, 3),
            "sparse_trajectory_valid": torch.ones(1, 1, 2, dtype=torch.bool)})
    pred["projection_valid"] = torch.zeros_like(pred["projection_valid"])
    with pytest.raises(ValueError, match="no valid"):
        MultiCamera4RCLoss()(pred, {
            "sparse_queries": torch.tensor([[[5., 2., 1.]]]),
            "sparse_visibility": torch.zeros(1, 1, 2, 1),
            "sparse_visibility_valid": torch.ones(1, 1, 2, 1, dtype=torch.bool)})


def test_sparse_visibility_bilinear_boundary_excludes_invalid_logits():
    pred = _sparse_predictions()
    pred["projection_valid"][..., 3:] = False
    pred["visibility"] = pred["visibility_logits"].sigmoid() * pred["projection_valid"]
    queries = torch.tensor([[[5., 2., 1.], [5., 2.49, 1.], [5., 3., 1.]]])
    sampled = sample_tracks(pred, queries)
    assert sampled["projection_valid"].dtype == torch.bool
    assert sampled["projection_valid"][:, 0].all()  # Integer valid pixel stays valid.
    assert not sampled["projection_valid"][:, 1:].any()
    torch.testing.assert_close(sampled["visibility"][:, 0], torch.full((1, 2, 1), .5))
    assert not sampled["visibility"][:, 1:].any()

    # Keep an integer valid query so BCE has supervision while boundary queries
    # neither train an invalid projection as a negative nor touch its neighbors.
    result = MultiCamera4RCLoss()(pred, {
        "sparse_queries": queries,
        "sparse_visibility": torch.zeros(1, 3, 2, 1),
        "sparse_visibility_valid": torch.ones(1, 3, 2, 1, dtype=torch.bool)})
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    gradient = pred["visibility_logits"].grad
    assert gradient[..., 2].abs().sum() > 0
    assert not gradient[~pred["projection_valid"]].any()
