"""Analytic release-format scenes: camera handedness, surface identity and masks."""
import csv
import json

import numpy as np
import pytest
import torch

cv2 = pytest.importorskip('cv2')
OpenEXR = pytest.importorskip('OpenEXR')
save_file = pytest.importorskip('safetensors.numpy').save_file

from multicamera4rc.datasets.adapters.syn4d import Syn4DAdapter, sequence_split
from multicamera4rc.datasets.adapters.syn4d_geometry import CAMERA_AXES, calibration, read_depth
from multicamera4rc.datasets.bridge import prepare_clip
from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.datasets.mixture import collate_adapter_samples, create_dataset, read_data_config
from multicamera4rc.datasets.syn4d_dense import attach_syn4d_dense_recipe, _depth_visibility


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def _fixture(root, *, camera_ids=(2, 7)):
    scene = root / 'data/syn4d_v1_stride_5/analytic'
    h, w, frames = 28, 42, 6
    # Triangle spans the whole image at Z=2; only a small mask belongs to it.
    cv_vertices = np.array([[-4., -4., 2.], [8., -4., 2.], [-4., 8., 2.]], np.float32)
    raw_vertices = (cv_vertices @ CAMERA_AXES.T) * [1., -1., 1.]
    asset = root / 'data/metadata/new_weight_bone/fixture/object/vertices_sequence.npz'
    asset.parent.mkdir(parents=True)
    np.savez(asset, vertices_seq=np.stack([raw_vertices, raw_vertices]).astype(np.float32),
             faces=np.array([[0, 1, 2]], np.int32))
    actor_rows, transforms = [], []
    for ci, cid in enumerate(camera_ids):
        name = f'seq_000000_{cid}'
        group = dict(Type='Group', Body='None', X=0, Y=0, Z=0, Yaw=0, Pitch=0, Roll=0,
                     Comment=f'sequence_name={name};frames={frames}')
        body = dict(Type='Body', Body='fixture_human_1', X=0, Y=0, Z=0, Yaw=0, Pitch=0, Roll=0,
                    Comment='start_frame=1')
        motion = [dict(frame=t, x=0, y=t * 10., z=0, yaw=0, pitch=0, roll=0, scale=1) for t in range(frames)]
        obj = dict(Type='Object', Body='fixture_object', X=0, Y=0, Z=0, Yaw=0, Pitch=0, Roll=0,
                   Comment='object_id=object;object_group=fixture', PerFrameMotion=json.dumps(motion))
        actor_rows.extend([group, body, obj])
        transforms.append(dict(sequence_id=name, sequence_body_index=0,
                               transform_channels=dict(scale_x=1, scale_y=1, scale_z=1)))
        rows, records, env_index = [], {}, {}
        offsets, flat_values, face_values, sizes = [0], [], [], []
        env_offsets, encoded_values = [0], []
        for t in range(frames):
            stem = f'{name}_{t:04d}'
            rows.append(dict(name=f'{stem}.png', x=0, y=ci * 20, z=0, yaw=0, pitch=0, roll=0,
                             focal_length=1, sensor_width=2, sensor_height=4/3))
            rgb_dir = scene / 'png' / name; rgb_dir.mkdir(parents=True, exist_ok=True)
            rgb = np.full((h, w, 3), [10 + t, 30, 90], np.uint8)
            cv2.imwrite(str(rgb_dir / f'{stem}.png'), rgb)
            depth_dir = scene / 'exr_layers/depth' / name; depth_dir.mkdir(parents=True, exist_ok=True)
            depth = np.full((h, w), 400., np.float32)
            depth[8:14, 12:20] = 200.
            depth[2:5, 2:5] = 200.  # Body geometry exists, mesh resources do not.
            depth[0, :3] = [np.nan, 0, 40000]
            OpenEXR.File({}, {'Depth': depth}).write(str(depth_dir / f'{stem}_depth.exr'))
            env = np.full((h, w), 255, np.uint8)
            env[8:14, 12:20] = 0; env[2:5, 2:5] = 0
            encoded = cv2.imencode('.png', env)[1].reshape(-1)
            env_index[stem] = t; env_offsets.append(env_offsets[-1] + len(encoded)); encoded_values.append(encoded)
            if t % 5 == 0:
                for mask, ys, xs in [('obj_01', slice(8, 14), slice(12, 20)), ('_00_body', slice(2, 5), slice(2, 5))]:
                    mask_array = np.zeros((h, w), bool); mask_array[ys, xs] = True
                    flat = np.flatnonzero(mask_array).astype(np.int32)
                    records[f'{stem}/{mask}'] = len(sizes)
                    sizes.append([h, w]); flat_values.append(flat); face_values.append(np.zeros(len(flat), np.int32))
                    offsets.append(offsets[-1] + len(flat))
        _csv(scene / 'ground_truth/meta_exr_csv' / f'{name}_camera.csv', rows)
        td = scene / 'tracking_safetensors'; td.mkdir(exist_ok=True)
        (td / f'{name}_index.json').write_text(json.dumps(dict(format='syn4d_compact_faceid_safetensors_v1', record_key_to_id=records)))
        save_file(dict(offsets=np.array(offsets, np.int64), record_hw=np.array(sizes, np.int32),
                       valid_flat_idx_values=np.concatenate(flat_values), face_ids_values=np.concatenate(face_values)), str(td / f'{name}.safetensors'))
        ed = scene / 'env_mask_safetensors'; ed.mkdir(exist_ok=True)
        (ed / f'{name}_index.json').write_text(json.dumps(dict(format='syn4d_env_mask_png_bytes_safetensors_v1', frame_to_id=env_index)))
        save_file(dict(offsets=np.array(env_offsets, np.int64), png_bytes_values=np.concatenate(encoded_values)), str(ed / f'{name}.safetensors'))
    _csv(scene / 'be_seq.csv', actor_rows)
    (scene / 'be_seq_transform_channels.json').write_text(json.dumps(transforms))
    return scene


def _batch(root, *, frame_policy='reference', C=2, include_visibility=True):
    adapter = Syn4DAdapter(root, split='all', frame_policy=frame_policy)
    seq = adapter.sequences[0]
    T = adapter.get_num_frames(seq)
    clip = adapter.load_clip(seq, list(range(T)) * C, layout_num_cameras=C, layout_num_times=T)
    sample = prepare_clip(clip, num_queries=0, image_mode='native')
    attach_syn4d_dense_recipe(clip, sample, image_mode='native', include_visibility=include_visibility)
    return clip, sample, collate_adapter_samples([sample])


def test_release_depth_units_camera_handedness_and_reference_frame_gaps(tmp_path):
    _fixture(tmp_path)
    clip, sample, batch = _batch(tmp_path)
    assert clip.metadata['time_indices'] == [0, 5]
    assert sample['query_time_idx'].tolist() == [0, 5]
    assert sample['camera_id'].tolist() == [2, 2, 7, 7]
    np.testing.assert_allclose(np.linalg.det(clip.extrinsics[:, :3, :3]), 1)
    assert clip.depths[0][10, 15] == 2.
    assert not clip.depths[0][0, :3].any()
    assert sample['images'].shape == (4, 3, 28, 42)
    np.testing.assert_allclose(sample['images'][0].permute(1, 2, 0).numpy(), clip.images[0] / 255., atol=1e-7)
    assert len(sample['syn4d']['missing_meshes']) == 2
    assert not any(k.startswith('sparse_') for k in batch['targets'])


def test_dense_motion_identity_static_environment_and_unknown_body(tmp_path):
    _fixture(tmp_path)
    _, _, batch = _batch(tmp_path)
    target = materialize_dense_targets(batch, torch.tensor([[0, 2]]), normalize_scene=False)
    assert target['trajectory'].shape == (1, 2, 2, 28, 42, 3)
    assert target['displacement_valid'][0, 0, :, 10, 15].all()
    torch.testing.assert_close(target['displacement'][0, :, 1, 10, 15], torch.tensor([[.5, 0, 0], [.5, 0, 0]]), atol=1e-6, rtol=0)
    assert target['trajectory_valid'][0, 0, 0, 3, 3]  # Self-time depth geometry.
    assert not target['trajectory_valid'][0, 0, 1, 3, 3]
    assert not target['displacement_valid'][0, 0, :, 3, 3].any()
    assert not target['static_mask'][0, 0, 3, 3]
    assert target['static_mask'][0, 0, 20, 30]
    assert not target['displacement'][0, 0, :, 20, 30].any()
    assert not target['trajectory_valid'][..., 0, :3].any()


def test_visibility_uses_content_boundary_and_preserves_unknown_depth_holes():
    # A 3x3 camera image is padded to 4x5. Points on the right/bottom pad
    # have known negative visibility, while a depth hole inside stays unknown.
    h, w = 4, 5
    trajectory = torch.zeros(1, 1, 1, h, w, 3)
    trajectory[0, 0, 0, 0, :5] = torch.tensor([
        [3., 1., 1.], [1., 3., 1.], [1., 1., 1.], [2., 2., 1.], [2.25, 2., 1.]])
    valid = torch.zeros(1, 1, 1, h, w, dtype=torch.bool)
    valid[0, 0, 0, 0, :5] = True
    depth = torch.zeros(1, 1, h, w)
    depth[0, 0, 2, 2] = 1.
    batch = dict(camera_id=torch.tensor([[7]]), observation_valid=torch.tensor([[True]]),
        time_idx=torch.tensor([[13]]), query_time_idx=torch.tensor([[13]]),
        targets=dict(dense_syn4d_content_hw=torch.tensor([[[3, 3]]])))
    target = dict(trajectory=trajectory, trajectory_valid=valid, camera_valid=torch.tensor([[True]]),
        c2w=torch.eye(4)[None, None], K=torch.eye(3)[None, None], depth=depth,
        depth_valid=depth > 0, scene_scale=torch.tensor([1.]))
    visible, labeled = _depth_visibility(batch, target)
    assert visible[0, 0, 0, 0, 0, :5].tolist() == [False, False, False, True, True]
    assert labeled[0, 0, 0, 0, 0, :5].tolist() == [True, True, False, True, True]


def test_all_frames_recovers_disjoint_camera_references_with_correct_motion(tmp_path):
    scene = _fixture(tmp_path)
    for camera, reference in ((2, 0), (7, 5)):
        path = scene/'tracking_safetensors'/f'seq_000000_{camera}_index.json'
        index = json.loads(path.read_text())
        index['record_key_to_id'] = {k: v for k, v in index['record_key_to_id'].items()
                                    if f'_{reference:04d}/' in k}
        path.write_text(json.dumps(index))
    strict = Syn4DAdapter(tmp_path, split='all', frame_policy='reference')
    assert not len(strict)
    _, _, batch = _batch(tmp_path, frame_policy='all', C=2)
    target = materialize_dense_targets(batch, torch.tensor([[0, 11]]), normalize_scene=False)
    assert batch['query_time_idx'].tolist() == [[0, 1, 2, 3, 4, 5]]
    assert target['displacement_valid'][0, :, :, 10, 15].all()
    torch.testing.assert_close(target['displacement'][0, 0, 5, 10, 15], torch.tensor([.5, 0, 0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(target['displacement'][0, 1, 0, 10, 15], torch.tensor([-.5, 0, 0]), atol=1e-6, rtol=0)


def test_scene_scale_is_shared_by_all_geometry_and_motion(tmp_path):
    _fixture(tmp_path)
    _, _, batch = _batch(tmp_path)
    source = torch.tensor([[0, 2]])
    raw = materialize_dense_targets(batch, source, normalize_scene=False)
    normalized = materialize_dense_targets(batch, source)
    scale = normalized['scene_scale'][0]
    for key in ('depth', 'trajectory', 'displacement'):
        torch.testing.assert_close(normalized[key] * scale, raw[key], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(normalized['c2w'][..., :3, 3] * scale, raw['c2w'][..., :3, 3])
    torch.testing.assert_close(normalized['ray'][..., :3], raw['ray'][..., :3])


def test_supplied_bedlam_mesh_enables_body_motion_with_correct_start_frame(tmp_path):
    _fixture(tmp_path)
    asset = tmp_path / 'data/metadata/new_weight_bone/fixture/object/vertices_sequence.npz'
    with np.load(asset) as data:
        v, faces = data['vertices_seq'][0], data['faces']
    vertices = np.repeat(v[None], 7, axis=0)
    # Body conversion maps these right-handed vertices to the same world frame.
    # start_frame=1 means mesh index 1 is rendered frame zero.
    vertices[:, :, 1] -= (np.arange(7)-1)[:, None] * .05
    body = tmp_path / 'data/metadata/bedlam2_smpl_npz/fixture_human/fixture_human_1.npz'
    body.parent.mkdir(parents=True)
    np.savez(body, vertices_seq=vertices, faces=faces)
    _, sample, batch = _batch(tmp_path)
    target = materialize_dense_targets(batch, torch.tensor([[0]]), normalize_scene=False)
    assert target['displacement_valid'][0,0,:,3,3].all()
    torch.testing.assert_close(target['displacement'][0,0,1,3,3], torch.tensor([.25,0.,0.]), atol=1e-6, rtol=0)
    assert [v['mask'] for v in sample['syn4d']['missing_meshes']] == ['_00_clothing']


def test_degenerate_or_distant_surface_is_unknown_motion(tmp_path):
    _fixture(tmp_path)
    _, _, batch = _batch(tmp_path, C=1)
    # Shift the full mesh one meter along the first camera's Z.
    batch['targets']['dense_syn4d_vertices'][..., 2] += 1.
    target = materialize_dense_targets(batch, torch.tensor([[0]]))
    assert not target['displacement_valid'][0, 0, :, 10, 15].any()
    assert target['depth_valid'][0, 0, 10, 15]


def test_visibility_keeps_hidden_3d_and_masks_missing_target_depth(tmp_path):
    _fixture(tmp_path)
    _, _, batch = _batch(tmp_path, C=1)
    batch['targets']['depth'][0, 1] = 1.  # All future geometry is behind a foreground occluder.
    target = materialize_dense_targets(batch, torch.tensor([[0]]), normalize_scene=False)
    assert target['trajectory_valid'][0, 0, 1, 10, 15]
    assert target['visibility_valid'][0, 0, 1, 0, 10, 15]
    assert not target['visibility'][0, 0, 1, 0, 10, 15]
    batch['targets']['depth_valid'][0, 1] = False
    target = materialize_dense_targets(batch, torch.tensor([[0]]))
    assert target['trajectory_valid'][0, 0, 1, 10, 15]
    assert not target['visibility_valid'][0, 0, 1, 0, 10, 15]


def test_variable_camera_counts_collate_and_mixture_plans(tmp_path):
    _fixture(tmp_path)
    _, mono, _ = _batch(tmp_path, C=1)
    _, stereo, _ = _batch(tmp_path, C=2)
    batch = collate_adapter_samples([mono, stereo])
    target = materialize_dense_targets(batch, torch.tensor([[0], [0]]))
    assert not target['depth_valid'][0, 2:].any()
    assert (batch['targets']['dense_syn4d_face_map'][0, 2:] == -1).all()
    config = dict(image_mode='native', supervision='dense', sampling='balanced_coverage', num_times=2,
                  epoch_size=12, sources=[dict(adapter='syn4d', root=str(tmp_path), split='all',
                    layouts=[dict(cameras=1, num_times=2), dict(cameras=2, num_times=2)])])
    dataset = create_dataset(config)
    plans = [dataset.sample_plan(i) for i in range(12)]
    assert {p['cameras'] for p in plans} == {1, 2}
    sample = dataset[0]
    assert sample['query_time_idx'].tolist() == [0, 5]
    assert 'dense_syn4d_vertices' in sample['targets']


def test_missing_assets_strict_mode_incomplete_download_and_split(tmp_path):
    scene = _fixture(tmp_path)
    adapter = Syn4DAdapter(tmp_path, split='all', missing_mesh='error')
    with pytest.raises(FileNotFoundError, match='mesh resources missing'):
        adapter.load_clip(adapter.sequences[0], [0])
    adapter = Syn4DAdapter(tmp_path, split='all', require_human_meshes=True)
    with pytest.raises(FileNotFoundError, match='required Syn4D human meshes'):
        adapter.load_clip(adapter.sequences[0], [0])
    (scene / 'exr_layers/depth/seq_000000_7/seq_000000_7_0001_depth.exr').unlink()
    adapter = Syn4DAdapter(tmp_path, split='all')
    assert adapter.get_sequence_info(adapter.sequences[0])['cameras'] == [2]
    assert any('incomplete RGB/depth' in r['reason'] for r in adapter.rejected)
    assigned = sequence_split('analytic', 'seq_000000')
    for split in ('train', 'val', 'test'):
        assert bool(len(Syn4DAdapter(tmp_path, split=split))) == (split == assigned)


def test_explicit_geometry_mode_does_not_require_partially_downloaded_track_shards(tmp_path):
    scene = _fixture(tmp_path)
    for path in (scene/'tracking_safetensors').glob('*.safetensors'):
        path.unlink()  # Indices arrived but payloads are not yet downloaded.
    assert not len(Syn4DAdapter(tmp_path, split='all'))
    adapter = Syn4DAdapter(tmp_path, split='all', load_tracking=False)
    assert adapter.get_num_frames(adapter.sequences[0]) == 6
    dataset = create_dataset(dict(image_mode='native', supervision='dense', num_times=2,
        sources=[dict(adapter='syn4d', root=str(tmp_path), split='all', cameras=2, dense_kind='geometry')]))
    target = materialize_dense_targets(collate_adapter_samples([dataset[0]]), torch.tensor([[0]]))
    assert target['depth_valid'].any()
    assert not target['displacement_valid'].any()


def test_metadata_paths_are_relative_to_config_and_camera_sequences_must_agree(tmp_path):
    scene = _fixture(tmp_path)
    path = tmp_path / 'data.yaml'
    path.write_text('sources:\n- adapter: syn4d\n  root: .\n  options:\n    metadata_root: data/metadata\n')
    config = read_data_config(path)
    assert config['sources'][0]['options']['metadata_root'] == str(tmp_path / 'data/metadata')
    transforms = json.loads((scene / 'be_seq_transform_channels.json').read_text())
    transforms[1]['transform_channels']['scale_x'] = 2
    (scene / 'be_seq_transform_channels.json').write_text(json.dumps(transforms))
    adapter = Syn4DAdapter(tmp_path, split='all')
    with pytest.raises(ValueError, match='synchronized actor geometry'):
        adapter.load_clip(adapter.sequences[0], [0, 0], layout_num_cameras=2)


def test_fixed_evaluation_case_uses_syn4d_recipe_and_raw_frame_ids(tmp_path):
    from multicamera4rc.evaluate import prepare_renderer_case
    _fixture(tmp_path)
    dataset = create_dataset(dict(image_mode='native', supervision='dense', num_times=2,
                     sources=[dict(adapter='syn4d', root=str(tmp_path), split='all', cameras=2)]))
    sequence = dataset.sources[0]['pool'][0][0]
    sample = prepare_renderer_case(dataset, dict(layout=0, sequence=sequence, times=[0,1], sampling_seed=0))
    assert sample['query_time_idx'].tolist() == [0,5]
    assert 'dense_syn4d_vertices' in sample['targets']
    assert 'dense_object_R' not in sample['targets']
