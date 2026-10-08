#!/usr/bin/env python3
"""Regression tests for BlendedMVS visibility cache refresh semantics."""

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets.adapters.blendedmvs import BlendedMVSAdapter
from datasets.query_builder import D4RTQueryBuilder
from datasets.transforms import GeometryTransformPipeline


def _write_pfm(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected single-channel depth map, got shape {image.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"Pf\n")
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode("ascii"))
        f.write(b"-1.0\n")
        np.flipud(image).tofile(f)


def _write_cam(path: Path, intrinsic: np.ndarray, extrinsic: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["extrinsic"]
    lines.extend(" ".join(f"{float(v):.8f}" for v in row) for row in extrinsic)
    lines.append("")
    lines.append("intrinsic")
    lines.extend(" ".join(f"{float(v):.8f}" for v in row) for row in intrinsic)
    lines.append("")
    lines.append("0.10000000 0.01000000 128.00000000 10.00000000")
    path.write_text("\n".join(lines) + "\n")


def _build_minimal_blendedmvs_root(root: Path) -> str:
    scene_id = "synthetic_scene"
    root.mkdir(parents=True, exist_ok=True)
    scene_dir = root / scene_id
    (scene_dir / "blended_images").mkdir(parents=True, exist_ok=True)
    (scene_dir / "rendered_depth_maps").mkdir(parents=True, exist_ok=True)
    (scene_dir / "cams").mkdir(parents=True, exist_ok=True)
    (root / "BlendedMVS_training.txt").write_text(f"{scene_id}\n")
    (root / "validation_list.txt").write_text(f"{scene_id}\n")

    rgb = np.full((3, 3, 3), 127, dtype=np.uint8)
    Image.fromarray(rgb).save(scene_dir / "blended_images" / "00000000.jpg")
    Image.fromarray(rgb).save(scene_dir / "blended_images" / "00000001.jpg")

    depth0 = np.full((3, 3), 2.0, dtype=np.float32)
    depth1 = np.full((3, 3), 2.0, dtype=np.float32)
    depth1[1, 1] = 1.0
    _write_pfm(scene_dir / "rendered_depth_maps" / "00000000.pfm", depth0)
    _write_pfm(scene_dir / "rendered_depth_maps" / "00000001.pfm", depth1)

    intrinsic = np.array(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    extrinsic = np.eye(4, dtype=np.float32)
    _write_cam(scene_dir / "cams" / "00000000_cam.txt", intrinsic, extrinsic)
    _write_cam(scene_dir / "cams" / "00000001_cam.txt", intrinsic, extrinsic)

    trajs_3d_world = np.array(
        [
            [[0.0, 0.0, 2.0]],
            [[0.0, 0.0, 2.0]],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        scene_dir / "precomputed.npz",
        normals=np.zeros((2, 3, 3, 3), dtype=np.float16),
        trajs_2d=np.array([[[1.0, 1.0]], [[1.0, 1.0]]], dtype=np.float32),
        trajs_3d_world=trajs_3d_world,
        valids=np.array([[True], [True]]),
        visibs=np.array([[True], [True]]),
        ref_frame=np.array(0, dtype=np.int32),
        num_frames=np.array(2, dtype=np.int32),
        num_points=np.array(1, dtype=np.int32),
    )
    return scene_id


def test_blendedmvs_old_cache_is_refreshed_and_yields_visibility_negatives(tmp_path: Path) -> None:
    root = tmp_path / "blendedmvs"
    sequence_name = _build_minimal_blendedmvs_root(root)

    adapter = BlendedMVSAdapter(root=str(root), split="train", verbose=False)
    clip = adapter.load_clip(sequence_name, [0, 1])

    assert clip.metadata["has_tracks"] is True
    assert clip.metadata["precomputed_track_semantics_version"] == 0
    assert clip.metadata["track_semantics_version"] >= 2
    assert clip.metadata["precomputed_track_semantics_refreshed"] is True
    assert clip.valids.shape == (2, 1)
    assert clip.visibs.shape == (2, 1)
    assert clip.valids[1, 0]
    assert not clip.visibs[1, 0]

    pipeline = GeometryTransformPipeline(img_size=3, use_augs=False)
    transformed = pipeline(clip)
    builder = D4RTQueryBuilder(
        num_queries=128,
        boundary_ratio=0.0,
        t_tgt_eq_t_cam_ratio=1.0,
        precompute_patches=False,
    )
    sample = builder(
        transformed,
        py_rng=random.Random(0),
        np_rng=np.random.default_rng(0),
    )

    mask_vis = sample.targets["mask_vis"].bool()
    visibility = sample.targets["visibility"]
    assert sample.metadata["query_semantics"] == "full_temporal"
    assert mask_vis.any()
    assert (visibility[mask_vis] < 0.5).any()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_blendedmvs_old_cache_is_refreshed_and_yields_visibility_negatives(Path(tmpdir))
    print("ok")
