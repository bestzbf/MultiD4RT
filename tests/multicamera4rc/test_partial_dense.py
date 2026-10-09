"""Partial GT must not manufacture motion or feed calibration into the model."""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from multicamera4rc import MultiCamera4RC, MultiCamera4RCConfig
from multicamera4rc.attention import relation_attention
from multicamera4rc.datasets.adapters.base import UnifiedClip
from multicamera4rc.datasets.adapters.dense_geometry import factor_projection, rectify_skew, canonicalize_world_basis
from multicamera4rc.datasets.bridge import prepare_clip
from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.datasets.mixture import create_dataset, collate_adapter_samples
from multicamera4rc.datasets.partial_dense import attach_partial_dense_recipe, replica_pairs, vkitti_pairs


def clip(T=3, shape=(27, 39)):
    h, w = shape
    return UnifiedClip(dataset_name="analytic", sequence_name="scene", frame_paths=None,
        images=[np.full((h, w, 3), 100, np.uint8) for _ in range(T)],
        depths=[np.full((h, w), 2., np.float32) for _ in range(T)],
        normals=None, trajs_2d=None, trajs_3d_world=None, valids=None, visibs=None,
        intrinsics=np.repeat(np.array([[20., 0, (w-1)/2], [0, 20., (h-1)/2], [0, 0, 1.]])[None], T, 0),
        extrinsics=np.repeat(np.eye(4)[None], T, 0),
        metadata={"frame_indices": list(range(T)), "extrinsics_convention": "w2c", "pixel_coordinates": "array_index_v1"})


def test_native_vkitti_scene_flow_removes_camera_motion_and_keeps_hidden_endpoint(tmp_path):
    c = clip(T=2); c.metadata["frame_indices"] = [0, 1]
    c.extrinsics[1, 0, 3] = -1.  # Camera center moves +1m in world.
    base = tmp_path/"frames"
    adapter = SimpleNamespace(seq_to_dir={"scene": tmp_path}, camera="Camera_0")
    h, w = c.depths[0].shape
    for f, step, sfdir, prefix, flowdir, fp in ((0, -1., "forwardSceneFlow", "sceneFlow", "forwardFlow", "flow"),
            (1, 1., "backwardSceneFlow", "backwardSceneFlow", "backwardFlow", "backwardFlow")):
        sf = np.zeros((h, w, 3)); sf[..., 0] = step
        optical = np.zeros((h, w, 2)); optical[..., 0] = step*20./2.
        raw = np.rint((sf/10.+1)*.5*65535).astype(np.uint16)[..., ::-1].copy()
        flow = np.zeros((h, w, 3), np.uint16); flow[..., 0] = 1
        flow[..., 2] = np.rint((optical[..., 0]/(w-1)+1)*.5*65535)
        flow[..., 1] = np.rint((optical[..., 1]/(h-1)+1)*.5*65535)
        for directory, pr, array in ((sfdir, prefix, raw), (flowdir, fp, flow)):
            path = base/directory/"Camera_0"; path.mkdir(parents=True)
            assert cv2.imwrite(str(path/f"{pr}_{f:05d}.png"), array)
    # Target depth intentionally disagrees: occlusion must not delete native
    # scene-flow endpoints. A raw camera-space vector of -1m is world-static.
    c.depths[1][:] = .5
    pairs = vkitti_pairs(c, adapter)
    _, delta, valid = pairs[0][0]
    assert valid.all()
    assert np.linalg.norm(delta, axis=-1).max() < .001
    s = prepare_clip(c, num_queries=0, image_mode="native")
    attach_partial_dense_recipe(c, s, kind="vkitti_scene_flow", image_mode="native", adapter=adapter)
    target = materialize_dense_targets(collate_adapter_samples([s]), torch.tensor([[0]]), normalize_scene=False)
    assert target["displacement_valid"][0, 0, 1, :h, :w].all()
    assert not target["displacement_valid"][..., h:, :].any()
    assert not target["static_mask"].any()


def test_replica_flow_bit_decoder_uses_dense_depth_and_never_vertex_tracks(tmp_path):
    c = clip(T=3); c.metadata.update(sequence_dir=str(tmp_path/"scene"), frame_numbers=[0, 1, 2], camera_name="left")
    h, w = c.depths[0].shape
    for directory, frames, dx in (("forward", [0, 1], 1.), ("backward", [1, 2], -1.)):
        p = tmp_path/"scene"/f"flow_{directory}"; p.mkdir(parents=True)
        m = tmp_path/"scene"/f"flow_{directory}_mask"; m.mkdir(parents=True)
        raw = np.concatenate((np.full((h, w), dx, np.float16), np.zeros((h, w), np.float16)), 1).view(np.uint16)
        for f in frames:
            assert cv2.imwrite(str(p/f"scene_{f:04d}.png"), raw)
            mask = np.full((h, w), 255, np.uint8); mask[0, 0] = 0
            assert cv2.imwrite(str(m/f"scene_{f:04d}.png"), mask)
    pairs = replica_pairs(c)
    _, delta, valid = pairs[0][0]
    assert valid.sum() > .9*h*w and not valid[0, 0] and not valid[:, -1].any()
    np.testing.assert_allclose(delta[valid, 0], .1, atol=1e-6)
    np.testing.assert_allclose(delta[valid, 1:], 0, atol=1e-6)
    s = prepare_clip(c, num_queries=0, image_mode="native")
    attach_partial_dense_recipe(c, s, kind="replica_flow", image_mode="native")
    target = materialize_dense_targets(collate_adapter_samples([s]), torch.tensor([[0]]), normalize_scene=False)
    assert target["displacement_valid"][0, 0, 1].any()
    assert not target["trajectory_valid"][0, 0, 2].any()  # No invented long-term GT.


def _write_replica_pair(c, source, *, placeholder=False, empty_direction=None):
    base = Path(c.metadata["sequence_dir"])
    h, w = c.depths[0].shape
    for direction, frame in (("forward", source), ("backward", source + 1)):
        flow_dir = base / f"flow_{direction}"
        mask_dir = base / f"flow_{direction}_mask"
        flow_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        name = f"{base.name}_{frame:04d}.png"
        raw = np.zeros((h, 2*w), np.uint8 if placeholder else np.uint16)
        mask = np.full((h, w), 0 if direction == empty_direction else 255, np.uint8)
        assert cv2.imwrite(str(flow_dir / name), raw)
        assert cv2.imwrite(str(mask_dir / name), mask)


@pytest.mark.parametrize("empty_direction", ["forward", "backward"])
def test_replica_empty_mask_placeholder_keeps_geometry_without_motion(tmp_path, empty_direction):
    c = clip(T=2)
    c.metadata.update(sequence_dir=str(tmp_path / "scene"), frame_numbers=[0, 1], camera_name="right")
    _write_replica_pair(c, 0, placeholder=True, empty_direction=empty_direction)
    assert replica_pairs(c) == [[None, None], [None, None]]
    s = prepare_clip(c, num_queries=0, image_mode="native")
    attach_partial_dense_recipe(c, s, kind="replica_flow", image_mode="native")
    target = materialize_dense_targets(collate_adapter_samples([s]), torch.tensor([[0, 1]]))
    assert target["depth_valid"].sum() == 2*27*39
    assert target["trajectory_valid"].sum() == 2*27*39  # Identity geometry only.
    assert not target["displacement_valid"].any()
    assert not target["static_mask"].any()


def test_replica_empty_pair_preserves_valid_zero_motion_neighbor(tmp_path):
    c = clip(T=3)
    c.metadata.update(sequence_dir=str(tmp_path / "scene"), frame_numbers=[0, 1, 2], camera_name="right")
    _write_replica_pair(c, 0)
    _write_replica_pair(c, 1, placeholder=True, empty_direction="forward")
    s = prepare_clip(c, num_queries=0, image_mode="native")
    attach_partial_dense_recipe(c, s, kind="replica_flow", image_mode="native")
    target = materialize_dense_targets(collate_adapter_samples([s]), torch.tensor([[0, 1, 2]]))
    assert target["displacement_valid"][0, 0, 1, :27, :39].all()
    assert target["displacement_valid"][0, 1, 0, :27, :39].all()
    assert not target["displacement"].any()  # Valid stationary GT stays valid.
    assert not target["displacement_valid"][0, 1, 2].any()
    assert not target["displacement_valid"][0, 2].any()


def test_replica_nonempty_mask_still_rejects_invalid_flow_encoding(tmp_path):
    c = clip(T=2)
    c.metadata.update(sequence_dir=str(tmp_path / "scene"), frame_numbers=[0, 1], camera_name="right")
    _write_replica_pair(c, 0, placeholder=True)
    with pytest.raises(ValueError, match="invalid Replica float16 flow"):
        replica_pairs(c)


def test_coverage_includes_every_frame_not_a_random_sequence_subsample(tmp_path):
    from test_native_loaders import _write_mugen
    root = tmp_path/"packed.h5"; _write_mugen(root)
    config = dict(image_mode="native", supervision="dense", sampling="coverage", epoch_size=1,
        num_times=3, sources=[dict(adapter="mugen_h5", root=str(root), num_times=3,
            options=dict(verbose=False, train_only_val_ratio=0., strip_padding=True))])
    d = create_dataset(config)
    assert len(d) == d.coverage_size > 1
    frames = set()
    for i in range(len(d)):
        s = d[i]; frames.update(s["time_idx"].tolist())
        assert not any(k.startswith("sparse_") for k in s["targets"])
    assert frames == set(range(d.sources[0]["adapter"].get_num_frames("moving")))


def test_no_depth_frame_is_rejected_logged_and_replaced_without_static_gt(tmp_path):
    import h5py
    from test_native_loaders import _write_mugen
    root = tmp_path/"packed.h5"; _write_mugen(root)
    with h5py.File(root, "r+") as f:
        f["train/seq_000/depths"][0] = 0
    log = tmp_path/"invalid.jsonl"
    d = create_dataset(dict(image_mode="native", supervision="dense", sampling="coverage",
        num_times=1, epoch_size=1, skip_invalid=True, invalid_log=str(log),
        sources=[dict(adapter="mugen_h5", root=str(root), options=dict(verbose=False, train_only_val_ratio=0.))]))
    s = d[0]
    assert s["targets"]["depth_valid"].any() and s["time_idx"].item() != 0
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[0]["invalid_index"] == 0 and "no valid pixels" in records[0]["reason"]


def test_invalid_dense_retry_preserves_time_bucket_across_sources(tmp_path):
    import h5py
    from test_native_loaders import _write_mugen
    root = tmp_path / "packed.h5"; _write_mugen(root)
    with h5py.File(root, "r+") as f:
        f["train/seq_000/depths"][0] = 0
    log = tmp_path / "invalid.jsonl"
    source = dict(adapter="mugen_h5", root=str(root),
                  options=dict(verbose=False, train_only_val_ratio=0.))
    dataset = create_dataset(dict(image_mode="native", supervision="dense", sampling="coverage",
        num_times=1, epoch_size=1, seed=0, skip_invalid=True, invalid_log=str(log),
        sources=[dict(source, num_times=1), dict(source, num_times=3)]))
    assert dataset.sample_plan(0)["num_times"] == 1
    sample = dataset[0]
    assert sample["query_time_idx"].numel() == 1
    assert sample["targets"]["depth_valid"].any()
    assert dataset[0]["query_time_idx"].tolist() == sample["query_time_idx"].tolist()
    assert all(dataset.sample_plan(json.loads(line)["invalid_index"])["num_times"] == 1
               for line in log.read_text().splitlines())


def test_invalid_dense_retry_fails_when_time_bucket_has_no_valid_clip(tmp_path):
    import h5py
    from test_native_loaders import _write_mugen
    invalid = tmp_path / "invalid.h5"; _write_mugen(invalid)
    valid = tmp_path / "valid.h5"; _write_mugen(valid)
    with h5py.File(invalid, "r+") as f:
        f["train/seq_000/depths"][:] = 0
    dataset = create_dataset(dict(image_mode="native", supervision="dense", sampling="coverage",
        num_times=1, epoch_size=1, seed=0, skip_invalid=True,
        invalid_log=str(tmp_path / "invalid.jsonl"), sources=[
            dict(adapter="mugen_h5", root=str(invalid), num_times=1,
                 options=dict(verbose=False, train_only_val_ratio=0.)),
            dict(adapter="mugen_h5", root=str(valid), num_times=3,
                 options=dict(verbose=False, train_only_val_ratio=0.))]))
    with pytest.raises(RuntimeError, match="No replacement dense clip with num_times=1"):
        dataset[0]


def test_single_camera_attention_matches_bias_forward_and_gradients():
    generator = torch.Generator().manual_seed(7)
    values = [torch.randn(2, 3, n, 8, dtype=torch.float64, generator=generator) for n in (7, 9, 9)]
    weight = torch.tensor([.8, -.2, 1.2], dtype=torch.float64)
    actual_values = [x.clone().requires_grad_() for x in [*values, weight]]
    reference_values = [x.clone().requires_grad_() for x in [*values, weight]]
    qc = torch.tensor([[-1, 0, 0, 1, 1, 0, 0], [-1, 2, 2, 1, 1, 2, 2]])
    kc = torch.tensor([[0]*8+[-1], [2]*8+[-1]])
    valid = torch.tensor([[True]*8+[False]]*2)
    result = relation_attention(*actual_values[:3], qc, kc, actual_values[3], valid, query_chunk_size=None)
    q, k, v, w = reference_values
    same = (qc[:, :, None] == kc[:, None]) & (qc[:, :, None] >= 0) & (kc[:, None] >= 0)
    bias = same[:, None].double()*w[None, :, None, None]
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias.masked_fill(~valid[:, None, None], -float("inf")))
    torch.testing.assert_close(result, expected, atol=1e-12, rtol=1e-12)
    result.square().sum().backward(); expected.square().sum().backward()
    for a, b in zip(actual_values, reference_values): torch.testing.assert_close(a.grad, b.grad, atol=1e-10, rtol=1e-9)


def test_skew_rectification_keeps_camera_z_and_all_image_corners():
    image = np.full((30, 40, 3), 100, np.uint8); depth = np.full((30, 40), 2., np.float32)
    K = np.array([[40., 3., 19.5], [0., 40., 14.5], [0., 0., 1.]])
    rgb, z, newK, valid = rectify_skew(image, depth, K)
    assert rgb.shape[1] > 40 and newK[0, 1] == 0
    assert valid.sum() > 1000
    np.testing.assert_allclose(z[valid], 2., atol=1e-6)
    P = np.zeros((4, 4)); P[0, 0] = 2.; P[1, 1] = 2.; P[3, 2] = -1.; P[2, 2] = -1.
    k, axis = factor_projection(P, 30, 40)
    np.testing.assert_allclose(axis, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(k, [[40, 0, 19.5], [0, 30, 14.5], [0, 0, 1]], atol=1e-12)


def test_world_reflection_preserves_canonical_camera_gauge_and_projection():
    F = np.diag([-1., 1., 1., 1.])
    proper = np.repeat(np.eye(4)[None], 3, 0)
    proper[:, :3, 3] = [[1., 2., 3.], [2., 3., 4.], [3., 4., 5.]]
    native = proper @ F
    E, handedness = canonicalize_world_basis(native)
    assert handedness == "left"
    np.testing.assert_allclose(E, proper)
    np.testing.assert_allclose(E[0] @ np.linalg.inv(E), native[0] @ np.linalg.inv(native))
    world = np.array([[1., 2., 5., 1.], [3., 4., 6., 1.]])
    np.testing.assert_allclose(world @ native[1].T, (world @ F.T) @ E[1].T)
