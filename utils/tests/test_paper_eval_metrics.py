"""Regression checks for paper-aligned evaluation metrics in utils.metrics."""

from __future__ import annotations

import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.metrics import (
    compute_depth_metrics,
    compute_point_cloud_metrics,
    compute_pose_auc,
    compute_pose_metrics,
    compute_tracking_metrics,
)


def test_depth_metrics_scale_only_alignment_recovers_absrel() -> None:
    target = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32)
    pred = target * 3.5

    metrics = compute_depth_metrics(pred, target, scale_invariant=True, shift_invariant=False)

    assert metrics["abs_rel"].item() < 1e-6
    assert metrics["rmse"].item() < 1e-6
    assert metrics["a1"].item() > 1.0 - 1e-6
    assert metrics["a2"].item() > 1.0 - 1e-6
    assert metrics["a3"].item() > 1.0 - 1e-6


def test_depth_metrics_scale_and_shift_alignment_handles_affine_bias() -> None:
    target = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float32)
    pred = target * 2.0 + 3.0

    scale_only = compute_depth_metrics(pred, target, scale_invariant=True, shift_invariant=False)
    scale_shift = compute_depth_metrics(pred, target, scale_invariant=False, shift_invariant=True)

    assert scale_only["abs_rel"].item() > 0.2
    assert scale_shift["abs_rel"].item() < 1e-6
    assert scale_shift["rmse"].item() < 1e-6


def test_point_cloud_metrics_mean_shift_alignment_matches_paper_protocol() -> None:
    gt = torch.tensor(
        [[0.0, 0.0, 0.0],
         [1.0, 2.0, 3.0],
         [-1.0, 0.5, 2.5]],
        dtype=torch.float32,
    )
    offset = torch.tensor([10.0, -4.0, 1.5], dtype=torch.float32)
    pred = gt + offset

    no_align = compute_point_cloud_metrics(pred, gt, align=False)
    aligned = compute_point_cloud_metrics(pred, gt, align=True)

    assert no_align["l1"].item() > 1.0
    assert aligned["l1"].item() < 1e-6
    assert aligned["chamfer"].item() < 1e-6


def test_pose_metrics_sim3_alignment_recovers_global_transform() -> None:
    gt = torch.eye(4, dtype=torch.float32).repeat(3, 1, 1)
    gt[:, 0, 3] = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)

    theta = math.pi / 2.0
    global_rotation = torch.tensor(
        [[math.cos(theta), -math.sin(theta), 0.0],
         [math.sin(theta), math.cos(theta), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    global_translation = torch.tensor([1.0, -0.5, 0.25], dtype=torch.float32)
    global_scale = 2.0

    pred = gt.clone()
    pred[:, :3, :3] = global_rotation.T
    gt_centers = gt[:, :3, 3]
    pred[:, :3, 3] = ((gt_centers - global_translation) / global_scale) @ global_rotation

    unaligned = compute_pose_metrics(pred, gt, align=False)
    aligned = compute_pose_metrics(pred, gt, align=True)

    assert unaligned["ate"].item() > 0.5
    assert aligned["ate"].item() < 1e-5
    assert aligned["rpe_trans"].item() < 1e-5
    assert aligned["rpe_rot"].item() < 1e-4


def test_pose_auc_at_30_behaves_like_accuracy_area() -> None:
    rotation_errors = torch.tensor([0.0, 0.5], dtype=torch.float32)
    translation_errors = torch.tensor([0.0, 0.0], dtype=torch.float32)

    auc = compute_pose_auc(rotation_errors, translation_errors, threshold=1.0)

    assert 0.73 < auc.item() < 0.77


def test_tracking_metrics_follow_paper_named_semantics() -> None:
    pred_tracks = torch.tensor(
        [[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ]],
        dtype=torch.float32,
    )
    gt_tracks = torch.tensor(
        [[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
        ]],
        dtype=torch.float32,
    )
    visibility = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
    pred_visibility = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    metrics = compute_tracking_metrics(
        pred_tracks,
        gt_tracks,
        visibility,
        pred_visibility=pred_visibility,
        thresholds=[0.5, 1.5],
    )

    expected_l1 = torch.tensor(1.0 / 3.0, dtype=torch.float32)
    assert torch.allclose(metrics["l1"], expected_l1, atol=1e-6)
    assert torch.allclose(metrics["apd"], torch.tensor(0.75), atol=1e-6)
    assert torch.allclose(metrics["aj_approx"], torch.tensor(0.5), atol=1e-6)
    assert torch.allclose(metrics["oa"], torch.tensor(2.0 / 3.0), atol=1e-6)


if __name__ == "__main__":
    test_depth_metrics_scale_only_alignment_recovers_absrel()
    test_depth_metrics_scale_and_shift_alignment_handles_affine_bias()
    test_point_cloud_metrics_mean_shift_alignment_matches_paper_protocol()
    test_pose_metrics_sim3_alignment_recovers_global_transform()
    test_pose_auc_at_30_behaves_like_accuracy_area()
    test_tracking_metrics_follow_paper_named_semantics()
    print("All paper evaluation metric tests passed.")
