"""Native mixed-camera batches, spawn workers and public training/inference CLI."""

from __future__ import annotations

import copy
import json

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
import yaml

from multicamera4rc import MultiCamera4RC, MultiCamera4RCConfig
from multicamera4rc.datasets.bridge import prepare_clip
from multicamera4rc.datasets.mixture import (
    collate_adapter_samples,
    create_dataset,
    read_data_config,
)
from multicamera4rc.datasets.registry import create_adapter
from multicamera4rc.infer import main as infer_main

from test_adapter_bridge import _clip
from test_native_loaders import _write_kubric, _write_mugen


def _configuration(tmp_path):
    root = tmp_path / "kubric"
    root.mkdir()
    _write_kubric(root / "scene.h5")
    _write_mugen(tmp_path / "packed.h5")
    return {
        "image_size": [28, 42], "num_times": 3, "num_queries": 12,
        "epoch_size": 2, "seed": 0,
        "sources": [
            {"adapter": "kubric_multicam", "root": "kubric", "split": "all",
             "cameras": 2, "weight": 1., "strides": [5],
             "options": {"verbose": False, "index_workers": 1,
                         "require_native_tracks": True, "extrinsics_are_opengl": True}},
            {"adapter": "mugen_h5", "root": "packed.h5", "split": "train",
             "cameras": 1, "weight": 1., "strides": [5],
             "options": {"dataset_name": "pointodyssey", "verbose": False,
                         "require_precomputed_tracks": True, "train_only_val_ratio": 0.}},
        ],
    }


def _write_configuration(tmp_path, suffix=".json"):
    config = _configuration(tmp_path)
    path = tmp_path / ("mixture" + suffix)
    if suffix == ".json":
        path.write_text(json.dumps(config))
    else:
        path.write_text(yaml.safe_dump(config))
    return path


def test_native_mixture_retains_time_gaps_and_pads_only_missing_observations(tmp_path):
    dataset = create_dataset(_write_configuration(tmp_path))
    samples = [dataset[0], dataset[1]]
    assert [sample["dataset_name"] for sample in samples] == ["mugen_h5", "kubric_multicam"]
    assert [sample["images"].shape[0] for sample in samples] == [3, 6]
    assert set(samples[0]["targets"]) == set(samples[1]["targets"])
    for sample in samples:
        assert sample["query_time_idx"].tolist() == [0, 5, 10]
        assert sample["time_idx"].unique().tolist() == [0, 5, 10]
    batch = collate_adapter_samples(samples)
    assert batch["images"].shape == (2, 6, 3, 28, 42)
    assert batch["query_time_idx"].shape == (2, 3)
    assert batch["observation_valid"].tolist() == [[True]*3 + [False]*3, [True]*6]
    assert batch["camera_id"][0, 3:].eq(-1).all()
    assert batch["time_idx"][0, 3:].eq(-1).all()
    target = batch["targets"]
    assert target["sparse_trajectory"].shape == (2, 12, 3, 3)
    assert target["sparse_visibility"].shape == (2, 12, 3, 2)
    assert not target["depth_valid"][0, 3:].any()
    assert not target["camera_valid"][0, 3:].any()
    assert not target["sparse_visibility_valid"][0, :, :, 1].any()
    assert target["depth_valid"][1].all()
    assert target["camera_valid"][1].all()


def test_native_dataset_with_open_parent_h5_can_spawn_dataloader_worker(tmp_path):
    dataset = create_dataset(_write_configuration(tmp_path))
    expected = collate_adapter_samples([dataset[0], dataset[1]])
    # dataset[0] leaves the MUGEN H5 handle open in the parent. The worker must
    # unpickle the dataset, reopen its own handle and preserve sample identity.
    loader = DataLoader(dataset, batch_size=2, num_workers=1,
                        multiprocessing_context="spawn", timeout=25,
                        collate_fn=collate_adapter_samples)
    iterator = iter(loader)
    actual = next(iterator)
    assert list(iterator) == []
    torch.testing.assert_close(actual["images"], expected["images"])
    torch.testing.assert_close(actual["targets"]["sparse_queries"], expected["targets"]["sparse_queries"])
    assert actual["dataset_name"] == expected["dataset_name"]


def test_safe_yaml_roots_resolve_relative_to_configuration(tmp_path, monkeypatch):
    config = _write_configuration(tmp_path, ".yaml")
    monkeypatch.chdir(tmp_path.parent)
    loaded = read_data_config(config)
    assert loaded["sources"][0]["root"] == str((tmp_path / "kubric").resolve())
    assert loaded["sources"][1]["root"] == str((tmp_path / "packed.h5").resolve())
    dataset = create_dataset(config)
    assert len(dataset) == 2
    assert dataset[0]["images"].shape == (3, 3, 28, 42)


def test_native_padding_config_preserves_calibration_and_masks_padded_pixels(tmp_path):
    path = _write_configuration(tmp_path, ".yaml")
    config = read_data_config(path)
    native = create_dataset(config)[1]
    config.update(image_mode="pad", image_size=[42, 56])
    path.write_text(yaml.safe_dump(config))
    padded = create_dataset(path)[1]
    # This fixture stores native 28x42 images. The adapter bridge pads once.
    torch.testing.assert_close(padded["images"][..., :28, :42], native["images"], atol=0, rtol=0)
    for name in ("K", "sparse_queries", "sparse_trajectory", "sparse_visibility"):
        torch.testing.assert_close(padded["targets"][name], native["targets"][name], atol=0, rtol=0)
    assert not padded["targets"]["depth_valid"][..., 28:, :].any()
    assert not padded["targets"]["depth_valid"][..., :, 42:].any()


def test_weighted_sources_can_sample_different_camera_and_time_counts(tmp_path):
    path = _write_configuration(tmp_path)
    config = read_data_config(path)
    multi = config["sources"][0]
    multi.update(num_times=5, strides=[1, 2])
    single = copy.deepcopy(multi)
    single.update(cameras=1, num_times=3, strides=[1, 3])
    config.update(sources=[multi, single], epoch_size=10)
    dataset = create_dataset(config)
    layouts = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        cameras = sample["camera_id"].unique().numel()
        times = sample["query_time_idx"].numel()
        layouts.add((cameras, times))
        assert sample["images"].shape[0] == cameras*times
        assert sample["targets"]["sparse_trajectory"].shape[1] == times
        assert sample["targets"]["sparse_visibility"].shape[1:3] == (times, cameras)
        batch = collate_adapter_samples([sample])
        assert batch["query_time_idx"].shape == (1, times)
    assert layouts == {(2, 5), (1, 3)}


def test_index_and_epoch_control_sampling_without_worker_rng(tmp_path):
    dataset = create_dataset(_write_configuration(tmp_path))
    original = dataset[0]
    torch.manual_seed(234)
    np.random.seed(77)
    repeated = dataset[0]
    torch.testing.assert_close(repeated["targets"]["sparse_queries"], original["targets"]["sparse_queries"])
    dataset.set_epoch(1)
    changed = dataset[0]
    assert not torch.equal(changed["targets"]["sparse_queries"], original["targets"]["sparse_queries"])
    dataset.set_epoch(0)
    restored = dataset[0]
    torch.testing.assert_close(restored["targets"]["sparse_queries"], original["targets"]["sparse_queries"])


@pytest.mark.parametrize("invalid", ["mugen_cameras", "tubelet", "typo"])
def test_unsupported_layouts_and_unknown_adapter_options_fail_early(tmp_path, invalid):
    path = _write_configuration(tmp_path)
    config = read_data_config(path)
    if invalid == "mugen_cameras":
        config["sources"][1]["cameras"] = 2
        expected, match = ValueError, "one camera stream"
    elif invalid == "tubelet":
        config["sources"][0]["options"]["encoder_tubelet_size"] = 2
        expected, match = ValueError, "encoder_tubelet_size must be 1"
    else:
        config["sources"][0]["options"]["pixel_coordinate"] = "array_index_v1"
        expected, match = TypeError, "Unsupported option"
    with pytest.raises(expected, match=match):
        create_dataset(config)


def test_mugen_interleaved_storage_uses_logical_shared_frame_labels(tmp_path):
    path = tmp_path / "interleaved.h5"
    _write_mugen(path)
    with h5py.File(path, "r+") as handle:
        group = handle["train/seq_000"]
        for name in list(group):
            array = group[name][()]
            if array.shape[0] == 11:
                del group[name]
                group.create_dataset(name, data=np.repeat(array, 10, axis=0))
    adapter = create_adapter(
        "mugen_h5", root=str(path), split="train", dataset_name="pointodyssey",
        frame_interleave=10, camera_stream_index=0, generate_static_tracks=False,
        require_precomputed_tracks=True, train_only_val_ratio=0., verbose=False,
    )
    clip = adapter.load_clip("moving", [0, 5, 10])
    assert clip.metadata["raw_frame_indices"] == [0, 50, 100]
    sample = prepare_clip(clip, image_size=(28, 42), num_queries=12)
    assert sample["time_idx"].tolist() == [0, 5, 10]
    assert sample["query_time_idx"].tolist() == [0, 5, 10]


def test_negative_visibility_label_can_be_known_without_world_target(tmp_path):
    clip = _clip()
    clip.valids[2:4, 0] = False
    clip.visibility_valids[2:4, 0] = True
    clip.visibs[2:4, 0] = False
    sample = prepare_clip(clip, image_size=(28, 42), num_queries=32)
    target = sample["targets"]
    queries = target["sparse_query_valid"] & (target["sparse_track_id"] == 0)
    assert queries.any()
    assert not target["sparse_trajectory_valid"][queries, 1].any()
    assert target["sparse_visibility_valid"][queries, 1].all()
    assert not target["sparse_visibility"][queries, 1].any()


def test_all_behind_camera_tracks_report_wrong_camera_convention():
    clip = _clip()
    for observation in range(clip.num_frames):
        c2w = np.linalg.inv(clip.extrinsics[observation])
        clip.trajs_3d_world[observation] = np.array([0., 0., -2.]) @ c2w[:3, :3].T + c2w[:3, 3]
    # Single camera avoids a separate cross-camera identity mismatch diagnosis.
    clip.metadata["valid_slot"] = [True, False, True, False, True]
    with pytest.raises(ValueError, match="(?i)behind"):
        prepare_clip(clip, image_size=(28, 42), num_queries=8)


def test_geometry_resize_uses_the_same_pixel_centers_as_rgb_and_intrinsics():
    clip = copy.deepcopy(_clip())
    clip.metadata["valid_slot"] = [True, False, False, False, False]
    clip.images[0] = np.zeros((2, 4, 3), np.uint8)
    clip.depths[0] = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]], np.float32)
    clip.trajs_3d_world = clip.trajs_2d = clip.valids = clip.visibs = clip.visibility_valids = None
    sample = prepare_clip(clip, image_size=(1, 2), num_queries=0)
    torch.testing.assert_close(sample["targets"]["depth"][0], torch.tensor([[6., 8.]]))
