from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.adapters.co3dv2 import Co3Dv2Adapter


def _write_json_gz(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)


def _write_image(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _make_sequence(
    root: Path,
    category: str,
    sequence_name: str,
    viewpoint_quality: float,
    pointcloud_quality: float,
    valid_depth_ratio: float,
    foreground_ratio: float,
    has_precomputed: bool,
) -> tuple[list[list], list[dict], dict]:
    seq_dir = root / category / sequence_name
    seq_dir.mkdir(parents=True, exist_ok=True)
    (seq_dir / "pointcloud.ply").write_text("ply\n")
    if has_precomputed:
        np.savez(seq_dir / "precomputed.npz", dummy=np.array([1], dtype=np.int32))

    set_entries: list[list] = []
    frame_annos: list[dict] = []
    for frame_number in range(2):
        image_rel = f"{category}/{sequence_name}/images/frame{frame_number:06d}.jpg"
        depth_rel = f"{category}/{sequence_name}/depths/frame{frame_number:06d}.png"
        depth_mask_rel = f"{category}/{sequence_name}/depth_masks/frame{frame_number:06d}.png"
        mask_rel = f"{category}/{sequence_name}/masks/frame{frame_number:06d}.png"

        _write_image(root / image_rel, np.full((4, 4, 3), 127, dtype=np.uint8))
        _write_image(root / depth_rel, np.zeros((4, 4), dtype=np.uint16))

        depth_mask = np.zeros((4, 4), dtype=np.uint8)
        depth_mask.reshape(-1)[: int(round(valid_depth_ratio * 16))] = 255
        _write_image(root / depth_mask_rel, depth_mask)

        fg_mask = np.zeros((4, 4), dtype=np.uint8)
        fg_mask.reshape(-1)[: int(round(foreground_ratio * 16))] = 255
        _write_image(root / mask_rel, fg_mask)

        set_entries.append([sequence_name, frame_number, image_rel])
        frame_annos.append(
            {
                "sequence_name": sequence_name,
                "frame_number": frame_number,
                "frame_timestamp": float(frame_number),
                "image": {"path": image_rel, "size": [4, 4]},
                "depth": {
                    "path": depth_rel,
                    "scale_adjustment": 1.0,
                    "mask_path": depth_mask_rel,
                },
                "mask": {"path": mask_rel, "mass": float(fg_mask.sum() / 255.0)},
                "viewpoint": {
                    "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "T": [0.0, 0.0, 0.0],
                    "focal_length": [1.0, 1.0],
                    "principal_point": [0.0, 0.0],
                    "intrinsics_format": "ndc_isotropic",
                },
            }
        )

    seq_anno = {
        "sequence_name": sequence_name,
        "category": category,
        "point_cloud": {
            "path": f"{category}/{sequence_name}/pointcloud.ply",
            "quality_score": pointcloud_quality,
            "n_points": 128,
        },
        "viewpoint_quality_score": viewpoint_quality,
    }
    return set_entries, frame_annos, seq_anno


def test_co3dv2_quality_filters(tmp_path: Path) -> None:
    root = tmp_path / "co3d"
    category = "apple"

    set_entries: list[list] = []
    frame_annos: list[dict] = []
    seq_annos: list[dict] = []
    for sequence_name, viewpoint_quality, pointcloud_quality, depth_ratio, fg_ratio, has_precomputed in [
        ("good_seq", 0.9, 0.8, 1.0, 0.75, True),
        ("low_view_seq", 0.2, 0.8, 1.0, 0.75, True),
        ("low_depth_seq", 0.95, 0.8, 0.0, 0.75, False),
    ]:
        entries, frames, seq = _make_sequence(
            root=root,
            category=category,
            sequence_name=sequence_name,
            viewpoint_quality=viewpoint_quality,
            pointcloud_quality=pointcloud_quality,
            valid_depth_ratio=depth_ratio,
            foreground_ratio=fg_ratio,
            has_precomputed=has_precomputed,
        )
        set_entries.extend(entries)
        frame_annos.extend(frames)
        seq_annos.append(seq)

    _write_json_gz(root / category / "frame_annotations.jgz", frame_annos)
    _write_json_gz(root / category / "sequence_annotations.jgz", seq_annos)
    (root / category / "set_lists").mkdir(parents=True, exist_ok=True)
    (root / category / "set_lists" / "set_lists_fewview_train.json").write_text(
        json.dumps({"train": set_entries, "val": [], "test": []})
    )

    adapter = Co3Dv2Adapter(
        root=str(root),
        categories=[category],
        min_viewpoint_quality=0.5,
        min_pointcloud_quality=0.5,
        min_valid_depth_ratio=0.5,
        require_precomputed=True,
        verbose=False,
    )

    assert adapter.list_sequences() == [f"{category}/good_seq"]
    info = adapter.get_sequence_info(f"{category}/good_seq")
    assert info["viewpoint_quality_score"] == 0.9
    assert info["pointcloud_quality_score"] == 0.8
    assert info["valid_depth_ratio"] == 1.0
    assert info["has_precomputed"] is True

    summary = adapter.get_filter_summary()
    assert summary["total_sequences_seen"] == 3
    assert summary["total_sequences_kept"] == 1
    assert summary["dropped_by_reason"]["low_viewpoint_quality"] == 1
    assert summary["dropped_by_reason"]["low_valid_depth_ratio"] == 1
