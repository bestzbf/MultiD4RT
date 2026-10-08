from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets.adapters.base import UnifiedClip
from datasets.query_builder import D4RTQueryBuilder
from datasets.transforms import CropParams, GeometryTransformPipeline, TransformResult


def _make_unified_clip(height: int = 480, width: int = 640, num_frames: int = 2) -> UnifiedClip:
    images = [
        np.full((height, width, 3), fill_value=idx * 20, dtype=np.uint8)
        for idx in range(num_frames)
    ]

    intrinsics = np.tile(np.eye(3, dtype=np.float32)[None], (num_frames, 1, 1))
    intrinsics[:, 0, 0] = 400.0
    intrinsics[:, 1, 1] = 300.0
    intrinsics[:, 0, 2] = width / 2.0
    intrinsics[:, 1, 2] = height / 2.0

    extrinsics = np.tile(np.eye(4, dtype=np.float32)[None], (num_frames, 1, 1))

    return UnifiedClip(
        dataset_name="synthetic",
        sequence_name="seq0",
        frame_paths=None,
        images=images,
        depths=None,
        normals=None,
        trajs_2d=None,
        trajs_3d_world=None,
        valids=None,
        visibs=None,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        metadata={"has_tracks": False},
    )


def _make_transform_result(
    *,
    original_h: int,
    original_w: int,
    crop_h: int,
    crop_w: int,
    img_size: int = 256,
    num_frames: int = 2,
) -> TransformResult:
    resized_images = [
        np.zeros((img_size, img_size, 3), dtype=np.float32)
        for _ in range(num_frames)
    ]
    cropped_images = [
        np.zeros((crop_h, crop_w, 3), dtype=np.float32)
        for _ in range(num_frames)
    ]
    intrinsics = np.tile(np.eye(3, dtype=np.float32)[None], (num_frames, 1, 1))
    extrinsics = np.tile(np.eye(4, dtype=np.float32)[None], (num_frames, 1, 1))

    return TransformResult(
        images=resized_images,
        cropped_images=cropped_images,
        depths=None,
        normals=None,
        normal_valids=None,
        trajs_2d=None,
        trajs_3d_world=None,
        valids=None,
        visibs=None,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        crop=CropParams(x0=11, y0=17, crop_w=crop_w, crop_h=crop_h),
        img_size=img_size,
        original_h=original_h,
        original_w=original_w,
        dataset_name="synthetic",
        sequence_name="seq0",
        metadata={"has_tracks": False},
    )


def test_noaug_keeps_full_frame_before_square_resize():
    clip = _make_unified_clip(height=480, width=640)
    pipeline = GeometryTransformPipeline(img_size=256, use_augs=False)

    result = pipeline(clip)

    assert result.crop == CropParams(x0=0, y0=0, crop_w=640, crop_h=480)
    assert result.cropped_images[0].shape == (480, 640, 3)
    assert result.images[0].shape == (256, 256, 3)

    expected = np.array(
        [
            [160.0, 0.0, 128.0],
            [0.0, 160.0, 128.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(result.intrinsics[0], expected)


def test_query_builder_aspect_ratio_tracks_actual_input_view():
    result = _make_transform_result(
        original_h=540,
        original_w=960,
        crop_h=280,
        crop_w=350,
    )
    builder = D4RTQueryBuilder(num_queries=16, precompute_patches=False)

    sample = builder(result)

    expected_ratio = torch.tensor([350.0 / 280.0], dtype=torch.float32)
    original_ratio = torch.tensor([960.0 / 540.0], dtype=torch.float32)

    assert torch.allclose(sample.aspect_ratio, expected_ratio)
    assert not torch.allclose(sample.aspect_ratio, original_ratio)


if __name__ == "__main__":
    test_noaug_keeps_full_frame_before_square_resize()
    print("✓ test_noaug_keeps_full_frame_before_square_resize passed")
    test_query_builder_aspect_ratio_tracks_actual_input_view()
    print("✓ test_query_builder_aspect_ratio_tracks_actual_input_view passed")
