#!/usr/bin/env python3
"""Regression test for clamped crop-coordinate normalization in QueryBuilder."""

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets.query_builder import D4RTQueryBuilder
from datasets.transforms import CropParams, TransformResult


def test_track_queries_are_clamped_to_unit_interval() -> None:
    T, H, W = 1, 10, 10
    images = [np.zeros((H, W, 3), dtype=np.float32)]
    depths = [np.ones((H, W), dtype=np.float32)]
    normals = [np.dstack([np.zeros((H, W), dtype=np.float32),
                          np.zeros((H, W), dtype=np.float32),
                          np.ones((H, W), dtype=np.float32)])]

    result = TransformResult(
        images=images,
        cropped_images=images,
        depths=depths,
        normals=normals,
        normal_valids=[True],
        trajs_2d=np.array([[[9.9, 5.0]]], dtype=np.float32),
        trajs_3d_world=np.array([[[9.9, 5.0, 1.0]]], dtype=np.float32),
        valids=np.array([[True]], dtype=bool),
        visibs=np.array([[True]], dtype=bool),
        intrinsics=np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float32),
        extrinsics=np.eye(4, dtype=np.float32)[None],
        crop=CropParams(x0=0, y0=0, crop_w=W, crop_h=H),
        img_size=H,
        original_h=H,
        original_w=W,
        dataset_name="blendedmvs",
        sequence_name="synthetic",
        metadata={"has_tracks": True, "has_visibility": True},
    )

    builder = D4RTQueryBuilder(
        num_queries=32,
        boundary_ratio=0.0,
        t_tgt_eq_t_cam_ratio=1.0,
        precompute_patches=False,
    )
    sample = builder(
        result,
        py_rng=random.Random(0),
        np_rng=np.random.default_rng(0),
    )

    coords = sample.coords.numpy()
    pos_2d = sample.targets["pos_2d"].numpy()
    mask_2d = sample.targets["mask_2d"].numpy().astype(bool)

    assert coords.min() >= 0.0
    assert coords.max() <= 1.0
    assert mask_2d.any()
    assert pos_2d[mask_2d].min() >= 0.0
    assert pos_2d[mask_2d].max() <= 1.0


if __name__ == "__main__":
    test_track_queries_are_clamped_to_unit_interval()
    print("ok")
