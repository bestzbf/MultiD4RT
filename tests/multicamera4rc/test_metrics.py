"""Diagnostics must measure physical errors independently of AMP/confidence."""

import math

import pytest
import torch

from multicamera4rc.evaluate import summarize_records
from multicamera4rc.metrics import dense_metrics, native_track_metrics, rotation_error_degrees
from multicamera4rc.train import _sparse_supervised_queries


@pytest.mark.parametrize("degrees", [0., .01, .1, .5, 1., 2., 5., 90., 179.9, 180.])
def test_camera_geodesic_preserves_small_angles_under_bfloat16_autocast(degrees):
    angle = math.radians(degrees)
    target = torch.tensor([[math.cos(angle), -math.sin(angle), 0.],
                           [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
    for enabled in (False, True):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=enabled):
            actual = rotation_error_degrees(torch.eye(3), target)
        assert actual.dtype == torch.float32
        assert float(actual) == pytest.approx(degrees, abs=2e-5)


def test_native_metrics_exclude_unselected_masked_nan_and_identity_point_times():
    times = torch.tensor([[5, 0, 2]])
    truth = torch.zeros(1, 4, 3, 3)
    truth[:, 3] = float("nan")
    estimate = torch.zeros_like(truth)
    estimate[:, 0, :, 0] = 2*times
    estimate[:, 1:3] = 1000
    estimate.requires_grad_()
    sampled = dict(trajectory=estimate, displacement=estimate,
                   query_valid=torch.tensor([[True, False, True, True]]),
                   query_time_idx=times, source_time_idx=torch.zeros(1, 4, dtype=torch.long),
                   motion_confidence=torch.full((1, 4, 3), 10000.))
    target = dict(sparse_trajectory=truth, sparse_displacement=truth,
                  sparse_query_valid=torch.tensor([[True, True, False, True]]),
                  sparse_trajectory_valid=torch.ones(1, 4, 3, dtype=torch.bool))
    metrics = native_track_metrics(sampled, target)
    for name in ("trajectory", "displacement"):
        assert float(metrics[f"native_{name}_epe_normalized"]) == pytest.approx(14/3)
        assert float(metrics[f"native_{name}_epe_temporal_normalized"]) == pytest.approx(7.)
    assert metrics["native_velocity_epe_normalized"] == 2
    assert metrics["native_track_queries"] == 1
    assert metrics["native_track_point_times"] == 3
    assert metrics["native_temporal_point_times"] == 2
    assert all(not value.requires_grad for value in metrics.values())


def test_native_metrics_omit_unknown_motion_instead_of_reporting_zero_error():
    sampled = dict(trajectory=torch.zeros(1, 1, 1, 3),
                   query_valid=torch.ones(1, 1, dtype=torch.bool),
                   query_time_idx=torch.tensor([[0]]), source_time_idx=torch.tensor([[0]]))
    target = dict(sparse_trajectory=torch.zeros(1, 1, 1, 3),
                  sparse_trajectory_valid=torch.zeros(1, 1, 1, dtype=torch.bool))
    metrics = native_track_metrics(sampled, target)
    assert not any("epe" in key for key in metrics)
    assert metrics["native_track_queries"] == 0


def test_source_selection_respects_explicit_query_validity():
    targets = dict(sparse_queries=torch.tensor([[[0., 1., 1.], [1., 1., 1.]]]),
                   sparse_trajectory=torch.ones(1, 2, 2, 3),
                   sparse_query_valid=torch.tensor([[False, True]]))
    assert _sparse_supervised_queries(targets, 0).tolist() == [False, True]


def test_dense_dynamic_velocity_includes_anchor_to_moving_endpoint():
    truth = torch.zeros(1, 1, 2, 2, 2, 3)
    truth[:, :, 1, ..., 0] = 1
    valid = torch.ones(truth.shape[:-1], dtype=torch.bool)
    foreground = valid.clone()
    foreground[:, :, 0] = False
    depth = torch.ones(1, 1, 2, 2)
    prediction = dict(trajectory=2*truth, displacement=2*truth, depth=depth,
                      query_time_idx=torch.tensor([[0, 2]]), source_time_idx=torch.tensor([[0]]))
    target = dict(trajectory=truth, displacement=truth, trajectory_valid=valid,
                  trajectory_foreground=foreground, depth=depth, depth_valid=depth.bool())
    metrics = dense_metrics(prediction, target)
    assert metrics["velocity_epe_dynamic_normalized"] == .5
    assert metrics["trajectory_epe_temporal_normalized"] == 1
    assert metrics["displacement_epe_normalized"] == .5


def test_evaluation_groups_repeated_layouts_as_one_physical_scene():
    records = [dict(sequence="scene_a", layout=[1, 4], metrics={"loss": 1.}),
               dict(sequence="scene_a", layout=[4, 1], metrics={"loss": 3.}),
               dict(sequence="scene_b", layout=[1, 4], metrics={"loss": 5.})]
    summary = summarize_records(records)
    assert summary["clips"] == 3 and summary["scenes"] == 2
    assert summary["by_sequence"]["scene_a"]["loss"] == 2.
    assert summary["mean"]["loss"] == 3.
