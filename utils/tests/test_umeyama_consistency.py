"""Test Umeyama backend consistency: native vs PyTorch3D must produce identical results."""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.camera import _umeyama_native, umeyama_alignment, PYTORCH3D_AVAILABLE


def _make_test_case(n=20, seed=42):
    """Generate a random rigid transformation and apply it to source points."""
    torch.manual_seed(seed)
    source = torch.randn(n, 3)

    # Ground truth: random rotation, translation, scale
    from scipy.spatial.transform import Rotation
    import numpy as np
    R_gt = torch.from_numpy(
        Rotation.random(random_state=seed).as_matrix()
    ).float()
    t_gt = torch.randn(3)
    s_gt = float(torch.empty(1).uniform_(0.5, 2.0))

    target = s_gt * (source @ R_gt.T) + t_gt
    return source, target, R_gt, t_gt, s_gt


def _rotation_angle_diff(R1, R2):
    """Angle (degrees) between two rotation matrices."""
    R_diff = R1 @ R2.T
    trace = R_diff.trace().clamp(-1.0, 3.0)
    angle_rad = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))
    return torch.rad2deg(angle_rad).item()


class TestUmeyamaNative:
    """Verify native implementation recovers ground truth."""

    def test_with_scale(self):
        source, target, R_gt, t_gt, s_gt = _make_test_case()
        R, t, s = _umeyama_native(source, target, with_scale=True)

        assert _rotation_angle_diff(R, R_gt) < 0.5, f"Rotation error too large"
        assert torch.allclose(t, t_gt, atol=1e-4), f"Translation error: {(t - t_gt).abs().max()}"
        assert abs(s.item() - s_gt) < 1e-4, f"Scale error: {abs(s.item() - s_gt)}"

    def test_without_scale(self):
        source, target, R_gt, t_gt, _ = _make_test_case()
        # No scale: target = R @ source + t
        target_rigid = source @ R_gt.T + t_gt
        R, t, s = _umeyama_native(source, target_rigid, with_scale=False)

        assert _rotation_angle_diff(R, R_gt) < 0.5
        assert torch.allclose(t, t_gt, atol=1e-4)
        assert abs(s.item() - 1.0) < 1e-4, "Scale should be ~1.0 when with_scale=False"

    def test_weighted(self):
        source, target, R_gt, t_gt, s_gt = _make_test_case(n=50)
        weights = torch.ones(50)
        R, t, s = _umeyama_native(source, target, weights=weights, with_scale=True)

        assert _rotation_angle_diff(R, R_gt) < 0.5
        assert torch.allclose(t, t_gt, atol=1e-4)


@pytest.mark.skipif(not PYTORCH3D_AVAILABLE, reason="PyTorch3D not installed")
class TestUmeyamaBackendConsistency:
    """Verify PyTorch3D and native backends produce identical results."""

    def test_with_scale_consistency(self):
        from utils.camera import _umeyama_pytorch3d
        source, target, _, _, _ = _make_test_case(seed=0)

        R_native, t_native, s_native = _umeyama_native(source, target, with_scale=True)
        R_p3d, t_p3d, s_p3d = _umeyama_pytorch3d(source, target, with_scale=True)

        angle_diff = _rotation_angle_diff(R_native, R_p3d)
        assert angle_diff < 1.0, f"Rotation mismatch: {angle_diff:.3f} deg"
        assert torch.allclose(t_native, t_p3d, atol=1e-3), \
            f"Translation mismatch: {(t_native - t_p3d).abs().max():.6f}"
        assert abs(s_native.item() - s_p3d.item()) < 1e-3, \
            f"Scale mismatch: native={s_native.item():.6f}, p3d={s_p3d.item():.6f}"

    def test_without_scale_consistency(self):
        from utils.camera import _umeyama_pytorch3d
        source, target, R_gt, t_gt, _ = _make_test_case(seed=1)
        target_rigid = source @ R_gt.T + t_gt

        R_native, t_native, s_native = _umeyama_native(source, target_rigid, with_scale=False)
        R_p3d, t_p3d, s_p3d = _umeyama_pytorch3d(source, target_rigid, with_scale=False)

        angle_diff = _rotation_angle_diff(R_native, R_p3d)
        assert angle_diff < 1.0, f"Rotation mismatch: {angle_diff:.3f} deg"
        assert torch.allclose(t_native, t_p3d, atol=1e-3), \
            f"Translation mismatch: {(t_native - t_p3d).abs().max():.6f}"

    def test_multiple_seeds(self):
        """Run consistency check across multiple random seeds."""
        from utils.camera import _umeyama_pytorch3d
        for seed in range(5):
            source, target, _, _, _ = _make_test_case(seed=seed)
            R_n, t_n, s_n = _umeyama_native(source, target, with_scale=True)
            R_p, t_p, s_p = _umeyama_pytorch3d(source, target, with_scale=True)

            angle_diff = _rotation_angle_diff(R_n, R_p)
            assert angle_diff < 1.0, f"seed={seed}: rotation mismatch {angle_diff:.3f} deg"
            assert torch.allclose(t_n, t_p, atol=1e-3), \
                f"seed={seed}: translation mismatch {(t_n - t_p).abs().max():.6f}"


if __name__ == '__main__':
    # Run native tests always
    t = TestUmeyamaNative()
    t.test_with_scale()
    t.test_without_scale()
    t.test_weighted()
    print("Native Umeyama tests passed.")

    if PYTORCH3D_AVAILABLE:
        t2 = TestUmeyamaBackendConsistency()
        t2.test_with_scale_consistency()
        t2.test_without_scale_consistency()
        t2.test_multiple_seeds()
        print("PyTorch3D consistency tests passed.")
    else:
        print("PyTorch3D not available, skipping consistency tests.")
