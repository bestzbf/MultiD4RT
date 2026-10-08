"""Minimal tests for static_reprojection_weight in D4RTLoss.

Verifies:
1. Default weight=1.0 produces identical loss to a loss_fn without the parameter.
2. weight=0.0 zeroes out static-query contribution to loss_3d.
3. weight=0.5 produces a loss strictly between the all-zero and all-one cases.
4. loss_3d_unweighted is never affected by static_reprojection_weight.

Run:
    python /workspace/openclaw/d4rt/losses/test_static_reprojection_weight.py
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from losses import D4RTLoss


def _make_batch(B=2, N=8, static_ratio=0.5, seed=42):
    """Build minimal predictions/targets dicts for testing."""
    torch.manual_seed(seed)
    n_static = int(N * static_ratio)

    is_static = torch.zeros(B, N, dtype=torch.bool)
    is_static[:, :n_static] = True

    predictions = {
        'pos_3d': torch.randn(B, N, 3),
        'pos_2d': torch.randn(B, N, 2),
        'visibility': torch.randn(B, N, 1),
        'displacement': torch.randn(B, N, 3),
        'normal': torch.randn(B, N, 3),
        'uncertainty': torch.zeros(B, N, 1),  # s=0 → exp(-s)=1, no conf effect
    }
    targets = {
        'pos_3d': torch.randn(B, N, 3),
        'pos_2d': torch.randn(B, N, 2),
        'visibility': torch.randint(0, 2, (B, N)).float(),
        'displacement': torch.randn(B, N, 3),
        'normal': torch.randn(B, N, 3),
        'mask_3d': torch.ones(B, N),
        'mask_disp': torch.ones(B, N),
        'mask_normal': torch.ones(B, N),
        'is_static_reprojection': is_static,
    }
    return predictions, targets


def test_default_weight_identical_to_baseline():
    """weight=1.0 must produce the same loss as a baseline without the param."""
    predictions, targets = _make_batch()

    baseline = D4RTLoss(lambda_conf=0.0)
    with_default = D4RTLoss(lambda_conf=0.0, static_reprojection_weight=1.0)

    out_base = baseline(predictions, targets)
    out_new = with_default(predictions, targets)

    assert torch.allclose(out_base['loss_3d'], out_new['loss_3d'], atol=1e-6), (
        f"loss_3d differs: {out_base['loss_3d']} vs {out_new['loss_3d']}"
    )
    assert torch.allclose(out_base['loss'], out_new['loss'], atol=1e-6), (
        f"total loss differs: {out_base['loss']} vs {out_new['loss']}"
    )


def test_zero_weight_reduces_loss():
    """weight=0.0 should reduce loss_3d compared to weight=1.0 when static queries exist."""
    predictions, targets = _make_batch(static_ratio=0.5)

    fn_full = D4RTLoss(lambda_conf=0.0, static_reprojection_weight=1.0)
    fn_zero = D4RTLoss(lambda_conf=0.0, static_reprojection_weight=0.0)

    out_full = fn_full(predictions, targets)
    out_zero = fn_zero(predictions, targets)

    assert out_zero['loss_3d'].item() < out_full['loss_3d'].item(), (
        "weight=0.0 should produce lower loss_3d than weight=1.0"
    )


def test_half_weight_is_between():
    """weight=0.5 loss_3d should be strictly between weight=0.0 and weight=1.0."""
    predictions, targets = _make_batch(static_ratio=0.5)

    fn_zero = D4RTLoss(lambda_conf=0.0, static_reprojection_weight=0.0)
    fn_half = D4RTLoss(lambda_conf=0.0, static_reprojection_weight=0.5)
    fn_full = D4RTLoss(lambda_conf=0.0, static_reprojection_weight=1.0)

    l_zero = fn_zero(predictions, targets)['loss_3d'].item()
    l_half = fn_half(predictions, targets)['loss_3d'].item()
    l_full = fn_full(predictions, targets)['loss_3d'].item()

    assert l_zero < l_half < l_full, (
        f"Expected {l_zero:.6f} < {l_half:.6f} < {l_full:.6f}"
    )


def test_unweighted_metric_unaffected():
    """loss_3d_unweighted must be identical regardless of static_reprojection_weight."""
    predictions, targets = _make_batch()

    fn_full = D4RTLoss(lambda_conf=0.0, static_reprojection_weight=1.0)
    fn_zero = D4RTLoss(lambda_conf=0.0, static_reprojection_weight=0.0)

    uw_full = fn_full(predictions, targets)['loss_3d_unweighted']
    uw_zero = fn_zero(predictions, targets)['loss_3d_unweighted']

    assert torch.allclose(uw_full, uw_zero, atol=1e-6), (
        f"loss_3d_unweighted should not be affected by static_reprojection_weight: "
        f"{uw_full} vs {uw_zero}"
    )


def test_no_static_mask_unaffected():
    """When is_static_reprojection is absent, any weight value must not change the loss."""
    predictions, targets = _make_batch()
    targets_no_static = {k: v for k, v in targets.items() if k != 'is_static_reprojection'}

    fn_full = D4RTLoss(lambda_conf=0.0, static_reprojection_weight=1.0)
    fn_zero = D4RTLoss(lambda_conf=0.0, static_reprojection_weight=0.0)

    out_full = fn_full(predictions, targets_no_static)
    out_zero = fn_zero(predictions, targets_no_static)

    assert torch.allclose(out_full['loss_3d'], out_zero['loss_3d'], atol=1e-6), (
        "Without is_static_reprojection, weight should have no effect"
    )


if __name__ == '__main__':
    test_default_weight_identical_to_baseline()
    test_zero_weight_reduces_loss()
    test_half_weight_is_between()
    test_unweighted_metric_unaffected()
    test_no_static_mask_unaffected()
    print("All tests passed.")
