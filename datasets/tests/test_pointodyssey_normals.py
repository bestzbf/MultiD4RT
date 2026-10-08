"""Regression checks for PointOdyssey released-normal camera conversion."""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.adapters.pointodyssey import PointOdysseyAdapter


def _adapter_without_init() -> PointOdysseyAdapter:
    return PointOdysseyAdapter.__new__(PointOdysseyAdapter)


def _extrinsic(rotation: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = rotation.astype(np.float32)
    return out


def test_pointodyssey_released_normals_identity_camera_rotation() -> None:
    adapter = _adapter_without_init()
    normals = [
        np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )
    ]
    extrinsics = np.stack([_extrinsic(np.eye(3, dtype=np.float32))], axis=0)

    converted = adapter._convert_released_normals_to_camera_space(normals, extrinsics)[0]
    expected = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    assert np.allclose(converted, expected), (converted, expected)
    assert np.allclose(np.linalg.norm(converted[0, 0]), 1.0)
    assert np.allclose(np.linalg.norm(converted[0, 1]), 1.0)
    assert np.allclose(np.linalg.norm(converted[1, 0]), 1.0)
    assert np.allclose(converted[1, 1], 0.0)


def test_pointodyssey_released_normals_respect_saved_camera_rotation() -> None:
    adapter = _adapter_without_init()
    normals = [
        np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )
    ]
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    extrinsics = np.stack([_extrinsic(rotation)], axis=0)

    converted = adapter._convert_released_normals_to_camera_space(normals, extrinsics)[0]
    expected = np.array(
        [
            [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    assert np.allclose(converted, expected), (converted, expected)
    assert np.allclose(converted[1, 1], 0.0)


if __name__ == "__main__":
    test_pointodyssey_released_normals_identity_camera_rotation()
    print("✓ test_pointodyssey_released_normals_identity_camera_rotation passed")
    test_pointodyssey_released_normals_respect_saved_camera_rotation()
    print("✓ test_pointodyssey_released_normals_respect_saved_camera_rotation passed")
    print("PointOdyssey released normal conversion tests passed.")
