#!/usr/bin/env python3
"""Regression tests for occlusion-aware depth-to-track visibility semantics."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets.computer.depth_to_tracks import (
    TRACK_SEMANTICS_VERSION,
    recompute_track_projection_masks,
)


def test_occluded_projection_is_defined_but_not_visible() -> None:
    depths = [
        np.full((3, 3), 2.0, dtype=np.float32),
        np.full((3, 3), 2.0, dtype=np.float32),
    ]
    depths[1][1, 1] = 1.0

    intrinsics = np.array(
        [
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    extrinsics = np.eye(4, dtype=np.float32)[None].repeat(2, axis=0)
    trajs_3d_world = np.array(
        [
            [[0.0, 0.0, 2.0]],
            [[0.0, 0.0, 2.0]],
        ],
        dtype=np.float32,
    )

    refreshed = recompute_track_projection_masks(
        depths=depths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        trajs_3d_world=trajs_3d_world,
    )

    assert TRACK_SEMANTICS_VERSION >= 2
    assert refreshed["trajs_2d"].shape == (2, 1, 2)
    assert np.allclose(refreshed["trajs_2d"][:, 0], np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert refreshed["valids"][0, 0]
    assert refreshed["visibs"][0, 0]
    assert refreshed["valids"][1, 0]
    assert not refreshed["visibs"][1, 0]


if __name__ == "__main__":
    test_occluded_projection_is_defined_but_not_visible()
    print("ok")
