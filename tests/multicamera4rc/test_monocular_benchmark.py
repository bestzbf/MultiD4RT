import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import pytest
import torch

from multicamera4rc import MultiCamera4RC, MultiCamera4RCConfig
from multicamera4rc.checkpoint import save_checkpoint
from scripts.evaluate_monocular import main
from scripts.monocular_protocol import (BenchmarkClip, align_depth_per_frame, frame_indices,
    geometry_metrics, geometry_on_eval_grid, load_clip, read_sintel_camera, read_sintel_depth,
    resize_intrinsics)


def make_scannet(tmp_path, name='scene0707_00', frames=8):
    root = tmp_path/'scans_test'; p = root/name/'processed'
    for folder in ('images', 'depth', 'pose'): (p/folder).mkdir(parents=True)
    K = np.array([[40., 0, 29.5], [0, 40., 19.5], [0, 0, 1]], np.float32)
    cameras = []
    for i in range(frames):
        rgb = np.full((40, 60, 3), [i*10, 70, 100], np.uint8)
        depth = np.full((20, 30), 1000+i*20, np.uint16)
        cv2.imwrite(str(p/'images'/f'{i:05d}.jpg'), rgb)
        cv2.imwrite(str(p/'depth'/f'{i:05d}.png'), depth)
        E = np.eye(4, dtype=np.float32); E[0, 3] = -.1*i; E[1, 3] = -.02*i*i
        cameras.append(E); np.savetxt(p/'pose'/f'{i:05d}.txt', E)
    with h5py.File(p/'precomputed.h5', 'w') as f:
        f['num_frames'] = frames
        f['intrinsics'] = np.repeat(K[None], frames, axis=0)
        f['extrinsics'] = np.stack(cameras)
    return root


def test_scannet_reuses_metric_gt_and_preserves_aspect_without_gt_network_input(tmp_path):
    root = make_scannet(tmp_path)
    case = dict(dataset='scannet', root=str(root), scene='scene0707_00', frame_indices=[0, 2, 4, 6])
    clip = load_clip(case, eval_size=28, long_side=42)
    assert clip.content_hw == (28, 42) and clip.images.shape == (1, 4, 3, 28, 42)
    assert clip.raw_hw == (40, 60)
    torch.testing.assert_close(clip.depth[:, 0, 0], torch.tensor([1., 1.04, 1.08, 1.12]))
    assert clip.K[0, 0, 0] == pytest.approx(40*28/60)
    assert clip.c2w[1, 0, 3] == pytest.approx(.2)
    assert clip.frame_indices == [0, 2, 4, 6]
    with h5py.File(root/'scene0707_00/processed/precomputed.h5', 'r+') as f:
        f['extrinsics'][0, 0, 3] = 8
    with pytest.raises(ValueError, match='disagree'): load_clip(case)


def test_unique_short_windows_never_inflate_pose_pairs():
    assert frame_indices(20, 48, dataset='sintel') == list(range(20))
    assert frame_indices(50, 48, dataset='sintel') == list(range(48))
    assert frame_indices(784, 48, stride=4, dataset='scannet') == list(range(0, 192, 4))
    assert frame_indices(12, 48, stride=4, dataset='scannet') == list(range(12))
    with pytest.raises(ValueError): frame_indices(0, 48, dataset='sintel')


def test_padding_is_removed_before_depth_and_calibration_resampling():
    clip = SimpleNamespace(content_hw=(14, 21), depth=torch.ones(2, 7, 7))
    depth = torch.full((1, 2, 28, 28), 999.)
    depth[:, :, :14, :21] = 3.
    K = torch.eye(3)[None, None].repeat(1, 2, 1, 1); K[..., 0, 0] = 9.; K[..., 1, 1] = 8.
    pred, intr, _ = geometry_on_eval_grid(dict(depth=depth, K=K, c2w=torch.eye(4)[None, None].repeat(1, 2, 1, 1)), clip)
    assert torch.equal(pred, torch.full((2, 7, 7), 3.))
    assert intr[0, 0, 0] == pytest.approx(3.) and intr[0, 1, 1] == pytest.approx(4.)
    assert intr[0, 0, 2] == pytest.approx(-1/3)


def test_d4rt_depth_alignment_matches_legacy_per_frame_function():
    import ast
    path = Path(__file__).resolve().parents[2]/'tests/fixtures/d4rt_depth_alignment.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_per_frame_align_depth')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    gt = torch.arange(1., 33.).reshape(2, 4, 4); pred = gt/3.+.1
    mask = torch.ones_like(gt, dtype=torch.bool)
    for mode in ('scale', 'scale_shift'):
        expected = namespace['_per_frame_align_depth'](pred, gt, mask, mode, max_depth=70)
        torch.testing.assert_close(align_depth_per_frame(pred, gt, mask, mode, 70), expected)


def test_one_scene_gauge_recovers_geometry_without_per_frame_repair():
    n, h, w = 4, 8, 8
    depth = torch.arange(1., h*w+1).reshape(1, h, w).repeat(n, 1, 1)/32+1
    K = torch.tensor([[20., 0, 3.5], [0, 20., 3.5], [0, 0, 1.]])[None].repeat(n, 1, 1)
    c2w = torch.eye(4)[None].repeat(n, 1, 1)
    c2w[:, 0, 3] = torch.tensor([0, .1, .2, .4]); c2w[:, 1, 3] = torch.tensor([0, 0, .1, .2])
    clip = BenchmarkClip(torch.zeros(1, n, 3, h, w), depth, K, c2w, list(range(n)), (h, w), (h, w), 'test')
    scaled_pose = c2w.clone(); scaled_pose[:, :3, 3] /= 3.
    metrics = geometry_metrics(depth/3, K, scaled_pose, clip, dataset='scannet')
    assert metrics['depth_S_abs_rel'] < 1e-6
    assert metrics['pointcloud_l1_scale_shift_m'] < 1e-6
    assert metrics['pointcloud_l1_sim3_m'] < 1e-6
    assert metrics['pointcloud_l1_raw_gauge'] > .05
    assert metrics['pose_ate_m'] < 1e-6 and metrics['pose_rpe_trans_m'] < 1e-6
    assert metrics['pose_num_pairs'] == n-1
    wrong = depth/3; wrong[1] *= 1.5
    drift = geometry_metrics(wrong, K, scaled_pose, clip, dataset='scannet')
    assert drift['depth_S_abs_rel'] < 1e-6  # Framewise depth metric remains invariant.
    assert drift['pointcloud_l1_scale_shift_m'] > .01  # One 3D gauge cannot hide drift.


def test_sintel_sdk_readers_reject_corrupt_files_and_preserve_w2c(tmp_path):
    depth = tmp_path/'frame.dpt'; cam = tmp_path/'frame.cam'
    with depth.open('wb') as f:
        np.array([202021.25], '<f4').tofile(f); np.array([3, 2], '<i4').tofile(f)
        np.arange(6, dtype='<f4').tofile(f)
    np.testing.assert_array_equal(read_sintel_depth(depth), np.arange(6).reshape(2, 3))
    with cam.open('wb') as f:
        np.array([202021.25], '<f4').tofile(f); np.eye(3, dtype='<f8').tofile(f)
        E = np.eye(4, dtype='<f8'); E[0, 3] = 2.; E[:3].tofile(f)
    K, E = read_sintel_camera(cam)
    assert E[0, 3] == 2 and K[2, 2] == 1
    depth.write_bytes(depth.read_bytes()[:-4])
    with pytest.raises(ValueError, match='truncated'): read_sintel_depth(depth)


def test_real_tiny_geometry_inference_and_frozen_audit(tmp_path):
    root = make_scannet(tmp_path)
    output, plan = tmp_path/'output', tmp_path/'plan.json'
    args = ['--dataset', 'scannet', '--roots', str(root), '--output', str(output), '--plan', str(plan),
            '--num-frames', '4', '--frame-stride', '2', '--eval-size', '28', '--long-side', '42', '--device', 'cpu']
    audit = main(args+['--audit-only'])
    assert audit['passed'] and len(audit['cases']) == 1
    checkpoint = tmp_path/'checkpoint.pt'
    torch.manual_seed(4)
    model = MultiCamera4RC(MultiCamera4RCConfig.tiny(gradient_checkpointing=False))
    save_checkpoint(checkpoint, model, step=55000)
    result = main(args+['--checkpoint', str(checkpoint)])
    assert result['stage'] == 'passed' and result['checkpoint_step'] == 55000
    assert result['records'][0]['actual_times'] == 4
    assert result['protocol']['intrinsics_input'] is False
    assert result['protocol']['confidence_filter'] is False
    # Same result can be resumed; a changed GT audit contract cannot be used.
    resumed = main(args+['--checkpoint', str(checkpoint), '--resume-results'])
    assert resumed['records'] == result['records']
    data = json.loads((output/'gt_audit.json').read_text()); data['plan_sha256'] = 'wrong'
    (output/'gt_audit.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='exact evaluation plan'): main(args+['--checkpoint', str(checkpoint)])
