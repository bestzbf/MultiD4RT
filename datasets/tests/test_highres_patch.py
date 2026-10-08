"""
Regression tests for highres patch collation and sampled_highres forward.

Run:
    python -m pytest datasets/tests/test_highres_patch.py -v
    # or
    python datasets/tests/test_highres_patch.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets.collate import d4rt_collate_fn
from datasets.query_builder import D4RTQueryBuilder, QuerySample
from datasets.transforms import CropParams, TransformResult
from utils.patches import extract_local_patches_from_frame_list, extract_local_patches_with_valid_hw


def _make_transform_result(
    img_size: int,
    crop_h: int,
    crop_w: int,
    T: int = 4,
) -> TransformResult:
    rng = np.random.default_rng(0)
    orig_h, orig_w = 480, 640

    resized = [rng.random((img_size, img_size, 3)).astype(np.float32) for _ in range(T)]
    cropped = [rng.random((crop_h, crop_w, 3)).astype(np.float32) for _ in range(T)]

    K = np.eye(3, dtype=np.float32)[None].repeat(T, axis=0)
    K[:, 0, 0] = 320.0
    K[:, 1, 1] = 320.0
    K[:, 0, 2] = img_size / 2
    K[:, 1, 2] = img_size / 2
    E = np.eye(4, dtype=np.float32)[None].repeat(T, axis=0)

    return TransformResult(
        images=resized,
        cropped_images=cropped,
        depths=None,
        normals=None,
        normal_valids=None,
        trajs_2d=None,
        trajs_3d_world=None,
        valids=None,
        visibs=None,
        intrinsics=K,
        extrinsics=E,
        crop=CropParams(x0=10, y0=10, crop_w=crop_w, crop_h=crop_h),
        img_size=img_size,
        original_h=orig_h,
        original_w=orig_w,
        dataset_name="synthetic",
        sequence_name="seq0",
        metadata={"has_tracks": False},
    )


def _build_sample(img_size: int, crop_h: int, crop_w: int, T: int = 4, Q: int = 64) -> QuerySample:
    result = _make_transform_result(img_size, crop_h, crop_w, T)
    builder = D4RTQueryBuilder(num_queries=Q, precompute_patches=False)
    return builder(result)


def test_collate_all_highres():
    samples = [_build_sample(img_size=256, crop_h=400, crop_w=350) for _ in range(3)]
    for s in samples:
        assert s.highres_video is not None

    batch = d4rt_collate_fn(samples)
    hv = batch["highres_video"]
    assert hv is not None
    assert isinstance(hv, list) and len(hv) == 3
    assert all(t is not None for t in hv)
    print("✓ test_collate_all_highres passed")


def test_collate_no_highres():
    samples = [_build_sample(img_size=256, crop_h=256, crop_w=256) for _ in range(3)]
    for s in samples:
        assert s.highres_video is None

    batch = d4rt_collate_fn(samples)
    assert batch["highres_video"] is None
    print("✓ test_collate_no_highres passed")


def test_collate_mixed_highres():
    s_highres = _build_sample(img_size=256, crop_h=400, crop_w=350)
    s_none = _build_sample(img_size=256, crop_h=256, crop_w=256)
    assert s_highres.highres_video is not None
    assert s_none.highres_video is None

    batch = d4rt_collate_fn([s_highres, s_none])
    hv = batch["highres_video"]
    assert hv is not None
    assert isinstance(hv, list) and len(hv) == 2
    assert hv[0] is not None
    assert hv[1] is None
    print("✓ test_collate_mixed_highres passed")


def test_sampled_highres_forward():
    from models.d4rt import create_d4rt

    device = torch.device("cpu")
    B, T, Q = 2, 4, 32
    img_size = 64

    crop_sizes = [(100, 90), (80, 120)]
    samples = [
        _build_sample(img_size=img_size, crop_h=ch, crop_w=cw, T=T, Q=Q)
        for ch, cw in crop_sizes
    ]
    batch = d4rt_collate_fn(samples)

    hv = batch["highres_video"]
    assert hv is not None and all(t is not None for t in hv)
    assert hv[0].shape[-2:] != hv[1].shape[-2:]

    model = create_d4rt(
        variant='base',
        img_size=img_size,
        temporal_size=T,
        patch_size=(1, 8, 8),
        decoder_depth=2,
        decoder_num_heads=4,
        query_patch_size=5,
        patch_provider='sampled_highres',
        encoder_pretrained=False,
    ).to(device).eval()

    video = batch["video"].to(device)
    coords = batch["coords"].to(device)
    t_src = batch["t_src"].to(device)
    t_tgt = batch["t_tgt"].to(device)
    t_cam = batch["t_cam"].to(device)
    transform_metadata = {k: v.to(device) for k, v in batch["transform_metadata"].items()}
    query_frames = batch["highres_video"]
    prepared_query_frames = model._prepare_query_frames(video, query_frames)
    assert isinstance(prepared_query_frames, list)
    assert [tuple(f.shape[-2:]) for f in prepared_query_frames] == crop_sizes

    with torch.no_grad():
        out = model(
            video, coords, t_src, t_tgt, t_cam,
            query_frames=query_frames,
            transform_metadata=transform_metadata,
        )

    assert out["pos_3d"].shape == (B, Q, 3)
    assert out["pos_2d"].shape == (B, Q, 2)
    assert torch.isfinite(out["pos_3d"]).all()
    print(f"✓ test_sampled_highres_forward passed  pos_3d={tuple(out['pos_3d'].shape)}")


def test_frame_list_patch_extraction_matches_padded_batch():
    torch.manual_seed(0)
    T, Q, patch_size = 4, 16, 5
    frames = [
        torch.rand(T, 3, 10, 8),
        torch.rand(T, 3, 6, 11),
    ]
    valid_hw = torch.tensor([[10.0, 8.0], [6.0, 11.0]])
    coords = torch.rand(2, Q, 2)
    t_src = torch.randint(0, T, (2, Q))

    max_h = max(f.shape[-2] for f in frames)
    max_w = max(f.shape[-1] for f in frames)
    padded = torch.stack(
        [
            torch.nn.functional.pad(
                f,
                (0, max_w - f.shape[-1], 0, max_h - f.shape[-2]),
            )
            for f in frames
        ],
        dim=0,
    )

    patches_from_list = extract_local_patches_from_frame_list(
        frames,
        coords,
        t_src,
        patch_size=patch_size,
        valid_hw=valid_hw,
    )
    patches_from_padded = extract_local_patches_with_valid_hw(
        padded,
        coords,
        t_src,
        patch_size=patch_size,
        valid_hw=valid_hw,
    )

    assert torch.allclose(patches_from_list, patches_from_padded, atol=1e-6)
    print("✓ test_frame_list_patch_extraction_matches_padded_batch passed")


def test_precomputed_highres_compact_sample():
    result = _make_transform_result(img_size=64, crop_h=100, crop_w=90, T=4)
    builder = D4RTQueryBuilder(
        num_queries=32,
        precompute_patches=True,
        precompute_from_highres=True,
        store_video_uint8=True,
        store_auxiliary_tensors=False,
    )
    sample = builder(result)

    assert sample.video.dtype == torch.uint8
    assert sample.highres_video is None
    assert sample.local_patches is not None
    assert sample.local_patches.dtype == torch.uint8
    assert int(sample.local_patches.min()) >= 0
    assert int(sample.local_patches.max()) <= 255
    assert sample.depths is None
    assert sample.normals is None
    print("✓ test_precomputed_highres_compact_sample passed")


def test_precomputed_resized_does_not_store_highres():
    result = _make_transform_result(img_size=64, crop_h=100, crop_w=90, T=4)
    builder = D4RTQueryBuilder(
        num_queries=32,
        precompute_patches=True,
        precompute_from_highres=False,
        store_video_uint8=True,
        store_auxiliary_tensors=False,
    )
    sample = builder(result)

    assert sample.video.dtype == torch.uint8
    assert sample.highres_video is None
    assert sample.local_patches is not None
    assert sample.local_patches.dtype == torch.uint8
    assert sample.depths is None
    assert sample.normals is None
    print("✓ test_precomputed_resized_does_not_store_highres passed")


def test_sampled_resized_does_not_store_highres():
    result = _make_transform_result(img_size=64, crop_h=100, crop_w=90, T=4)
    builder = D4RTQueryBuilder(
        num_queries=32,
        precompute_patches=False,
        return_highres_video=False,
        store_video_uint8=True,
        store_auxiliary_tensors=False,
    )
    sample = builder(result)

    assert sample.video.dtype == torch.uint8
    assert sample.highres_video is None
    assert sample.local_patches is None
    assert sample.depths is None
    assert sample.normals is None
    print("✓ test_sampled_resized_does_not_store_highres passed")


if __name__ == "__main__":
    test_collate_all_highres()
    test_collate_no_highres()
    test_collate_mixed_highres()
    test_sampled_highres_forward()
    test_frame_list_patch_extraction_matches_padded_batch()
    test_precomputed_highres_compact_sample()
    test_precomputed_resized_does_not_store_highres()
    test_sampled_resized_does_not_store_highres()
    print("\nAll tests passed.")
