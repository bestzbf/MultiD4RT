"""
Lightweight regression checks for static/temporal observability metrics in D4RTLoss.

Run:
    python /workspace/openclaw/d4rt/losses/test_static_semantic_metrics.py
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from losses import D4RTLoss


def _make_batch() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    B, N = 2, 6
    g = torch.Generator().manual_seed(0)

    predictions = {
        "pos_3d": torch.rand(B, N, 3, generator=g),
        "pos_2d": torch.rand(B, N, 2, generator=g),
        "visibility": torch.randn(B, N, 1, generator=g),
        "displacement": torch.rand(B, N, 3, generator=g),
        "normal": torch.rand(B, N, 3, generator=g),
        # Kendall s = log(sigma^2)
        "uncertainty": torch.randn(B, N, 1, generator=g) * 0.1,
    }

    targets = {
        "pos_3d": torch.rand(B, N, 3, generator=g),
        "pos_2d": torch.rand(B, N, 2, generator=g),
        "visibility": torch.randint(0, 2, (B, N), generator=g).float(),
        "displacement": torch.rand(B, N, 3, generator=g),
        "normal": torch.rand(B, N, 3, generator=g),
        "mask_3d": torch.tensor(
            [[True, True, True, False, False, False],
             [True, True, False, True, False, False]]
        ),
        "mask_2d": torch.tensor(
            [[True, True, True, True, False, False],
             [True, True, True, True, False, False]]
        ),
        "mask_vis": torch.tensor(
            [[True, True, True, True, False, False],
             [True, True, True, True, False, False]]
        ),
        "mask_disp": torch.tensor(
            [[True, True, True, False, False, False],
             [True, True, False, True, False, False]]
        ),
        "mask_normal": torch.tensor(
            [[True, False, True, False, False, False],
             [True, False, False, True, False, False]]
        ),
        "is_static_reprojection": torch.tensor(
            [[True, True, False, False, False, False],
             [True, False, False, True, False, False]]
        ),
    }

    normalize_groups = torch.tensor(
        [[0, 0, 1, 1, 2, 2],
         [0, 1, 1, 2, 2, 2]],
        dtype=torch.long,
    )

    return predictions, targets, normalize_groups


def test_static_semantic_metrics_present_and_finite() -> None:
    predictions, targets, normalize_groups = _make_batch()
    loss_fn = D4RTLoss()

    out = loss_fn(predictions, targets, normalize_groups=normalize_groups)

    required_keys = [
        "metric_static_query_ratio",
        "metric_temporal_query_ratio",
        "metric_valid_3d_query_ratio",
        "metric_static_valid3d_ratio",
        "metric_temporal_valid3d_ratio",
        "metric_normal_query_ratio",
        "metric_normal_valid3d_ratio",
        "metric_loss_3d_static_unweighted",
        "metric_loss_3d_temporal_unweighted",
        "metric_raw_3d_l1_static",
        "metric_raw_3d_l1_temporal",
        "metric_raw_3d_euclidean_static",
        "metric_raw_3d_euclidean_temporal",
    ]

    for k in required_keys:
        assert k in out, f"missing metric key: {k}"
        v = out[k]
        assert torch.is_tensor(v), f"metric {k} is not a tensor"
        assert torch.isfinite(v), f"metric {k} is not finite: {v}"

    ratio_keys = [
        "metric_static_query_ratio",
        "metric_temporal_query_ratio",
        "metric_valid_3d_query_ratio",
        "metric_static_valid3d_ratio",
        "metric_temporal_valid3d_ratio",
        "metric_normal_query_ratio",
        "metric_normal_valid3d_ratio",
    ]
    for k in ratio_keys:
        v = out[k].item()
        assert -1e-6 <= v <= 1.0 + 1e-6, f"ratio {k} out of [0,1]: {v}"

    print("✓ test_static_semantic_metrics_present_and_finite passed")


def test_normal_loss_uses_effective_valid_3d_mask() -> None:
    predictions = {
        "pos_3d": torch.zeros(1, 2, 3),
        "normal": torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        "uncertainty": torch.zeros(1, 2, 1),
    }
    targets = {
        "pos_3d": torch.zeros(1, 2, 3),
        "normal": torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]]),
        "mask_3d": torch.tensor([[True, False]]),
        "mask_normal": torch.tensor([[True, True]]),
    }

    out = D4RTLoss()(predictions, targets)

    assert torch.isclose(out["loss_normal"], torch.tensor(0.0)), out["loss_normal"]
    assert torch.isclose(out["metric_normal_query_ratio"], torch.tensor(0.5)), out["metric_normal_query_ratio"]
    assert torch.isclose(out["metric_normal_valid3d_ratio"], torch.tensor(1.0)), out["metric_normal_valid3d_ratio"]

    print("✓ test_normal_loss_uses_effective_valid_3d_mask passed")


if __name__ == "__main__":
    test_static_semantic_metrics_present_and_finite()
    test_normal_loss_uses_effective_valid_3d_mask()
    print("All static semantic metric tests passed.")
