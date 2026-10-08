"""
Regression tests for target-field preservation in d4rt_collate_fn.

Run:
    python -m pytest datasets/tests/test_collate_targets.py -v
    # or
    python datasets/tests/test_collate_targets.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
COLLATE_PATH = ROOT / "datasets" / "collate.py"
_spec = importlib.util.spec_from_file_location("d4rt_collate_module", COLLATE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Unable to load collate module from {COLLATE_PATH}")
_collate_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_collate_module)
d4rt_collate_fn = _collate_module.d4rt_collate_fn


Q = 4
T = 3
H = 8
W = 8


def _make_sample(dataset_name: str, seq_name: str, seed: int) -> SimpleNamespace:
    g = torch.Generator().manual_seed(seed)

    targets = {
        "pos_2d": torch.rand(Q, 2, generator=g),
        "pos_3d": torch.rand(Q, 3, generator=g),
        "visibility": torch.rand(Q, generator=g),
        "displacement": torch.rand(Q, 3, generator=g),
        "normal": torch.rand(Q, 3, generator=g),
        "mask_2d": torch.tensor([True, False, True, False]),
        "mask_3d": torch.tensor([True, True, False, False]),
        "mask_vis": torch.tensor([True, True, True, False]),
        "mask_disp": torch.tensor([False, True, False, True]),
        "mask_normal": torch.tensor([True, False, False, True]),
        "source_is_boundary": torch.tensor([True, False, False, True]),
        "source_is_depth_boundary": torch.tensor([False, True, False, True]),
        "source_is_motion_boundary": torch.tensor([False, False, True, True]),
        "point_indices": torch.tensor([1, 7, 11, 15], dtype=torch.long),
        "is_static_reprojection": torch.tensor([True, True, False, False]),
        "custom_marker": torch.tensor([seed, seed + 1, seed + 2, seed + 3], dtype=torch.long),
    }

    return SimpleNamespace(
        video=torch.rand(T, 3, H, W, generator=g),
        highres_video=None,
        depths=None,
        normals=None,
        coords=torch.rand(Q, 2, generator=g),
        t_src=torch.tensor([0, 0, 1, 2], dtype=torch.long),
        t_tgt=torch.tensor([0, 0, 1, 2], dtype=torch.long),
        t_cam=torch.tensor([0, 1, 1, 2], dtype=torch.long),
        intrinsics=torch.eye(3).repeat(T, 1, 1),
        extrinsics=torch.eye(4).repeat(T, 1, 1),
        targets=targets,
        local_patches=None,
        transform_metadata={
            "canonical_space": torch.tensor(0, dtype=torch.long),
            "original_hw": torch.tensor([480.0, 640.0]),
            "crop_offset_xy": torch.tensor([10.0, 20.0]),
            "crop_size_hw": torch.tensor([320.0, 480.0]),
            "resized_hw": torch.tensor([256.0, 256.0]),
        },
        aspect_ratio=torch.tensor([640.0 / 480.0]),
        dataset_name=dataset_name,
        sequence_name=seq_name,
        metadata={"seed": seed},
    )


def test_collate_preserves_all_target_keys_and_values():
    s0 = _make_sample("dataset_a", "seq0", seed=1)
    s1 = _make_sample("dataset_b", "seq1", seed=5)

    batch = d4rt_collate_fn([s0, s1])

    assert set(batch["targets"].keys()) == set(s0.targets.keys()) == set(s1.targets.keys())

    for key in s0.targets:
        expected = torch.stack([s0.targets[key], s1.targets[key]], dim=0)
        actual = batch["targets"][key]
        assert torch.equal(actual, expected), f"target field {key!r} was not stacked correctly"

    for key in [
        "source_is_boundary",
        "source_is_depth_boundary",
        "source_is_motion_boundary",
        "point_indices",
        "is_static_reprojection",
        "custom_marker",
    ]:
        assert key in batch["targets"], f"missing preserved target field: {key}"

    print("✓ test_collate_preserves_all_target_keys_and_values passed")


if __name__ == "__main__":
    test_collate_preserves_all_target_keys_and_values()
    print("All collate target preservation tests passed.")
