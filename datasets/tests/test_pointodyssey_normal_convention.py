"""Compatibility wrapper for PointOdyssey normal convention regression tests."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.tests.test_pointodyssey_normals import (
    test_pointodyssey_released_normals_identity_camera_rotation,
    test_pointodyssey_released_normals_respect_saved_camera_rotation,
)


def test_pointodyssey_normal_conversion_basis_vectors() -> None:
    test_pointodyssey_released_normals_identity_camera_rotation()
    test_pointodyssey_released_normals_respect_saved_camera_rotation()
    print("✓ test_pointodyssey_normal_conversion_basis_vectors passed")


if __name__ == "__main__":
    test_pointodyssey_normal_conversion_basis_vectors()
    print("PointOdyssey normal convention tests passed.")
