"""Input synchronization, pixel transforms and real inference/viewer exports."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import pytest
import torch

from multid4rt.tasks import sample_tracks
from multid4rt.visual_inputs import load_visual_input
from multid4rt.visualization import build_visualization, export_visualization
from multid4rt.visualize import main as visualize_main, make_visual_demo


def image(path, size=(39, 27), color=(120, 60, 180)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color).save(path)
    return path


def analytic_prediction(visual):
    """Known world geometry and linear movement, independent of viewer code."""
    _, s, _, h, w = visual.images.shape
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    base = torch.stack((x.float(), y.float(), torch.full_like(x, 4).float()), -1)
    base = base[None].repeat(s, 1, 1, 1)
    base[..., 0] += torch.arange(s)[:, None, None]*100
    times = visual.time_idx.unique(sorted=True)[None]
    sources = torch.tensor([[0, s-1]]) if s > 1 else torch.tensor([[0]])
    source_times = visual.time_idx[:, sources[0]]
    delta = (times[:, None]-source_times[..., None]).float()
    move = delta[..., None, None, None]*torch.tensor([.1, -.05, 0.])
    move = move.expand(-1, -1, -1, h, w, -1).clone()
    return dict(base_points=base[None], depth=base[None, ..., 2], depth_conf=torch.ones(1,s,h,w),
                trajectory=base[sources[0]][None, :, None]+move, displacement=move,
                motion_confidence=torch.ones(1, sources.shape[1], times.shape[1], h, w),
                K=torch.eye(3)[None, None].repeat(1,s,1,1),
                c2w=torch.eye(4)[None, None].repeat(1,s,1,1),
                camera_id=visual.camera_id, time_idx=visual.time_idx, query_time_idx=times,
                source_obs_idx=sources, source_time_idx=source_times,
                source_valid=torch.ones_like(sources, dtype=torch.bool))


def test_synchronized_folders_match_frame_labels_not_positions(tmp_path):
    for camera, frames in enumerate(([0,5,10], [5,10,15])):
        for frame in frames:
            image(tmp_path/f'camera_{camera}'/f'frame_{frame}.png')
    visual = load_visual_input(tmp_path)
    assert visual.mode == 'multicamera'
    assert visual.camera_id.tolist() == [[0,0,1,1]]
    assert visual.time_idx.tolist() == [[5,10,5,10]]


def test_aspect_ratio_padding_and_reuse_masks(tmp_path):
    image(tmp_path/'left.png', (320,160))
    image(tmp_path/'right.png', (200,300))
    visual = load_visual_input(tmp_path, 'multiview', long_side=140)
    assert visual.images.shape == (1,2,3,140,140)
    assert [i['content_hw'] for i in visual.observations] == [[70,140],[140,93]]
    assert visual.pixel_valid.sum().item() == 70*140+140*93
    assert not visual.pixel_valid[0,70:].any()
    manifest = visual.write(tmp_path/'prepared')
    restored = load_visual_input(manifest, long_side=0, max_frames=0)
    assert restored.observations == visual.observations
    assert torch.equal(restored.images, visual.images)
    assert torch.equal(restored.pixel_valid, visual.pixel_valid)


def test_manifest_keeps_reference_and_noncontiguous_camera_ids(tmp_path):
    image(tmp_path/'a.png')
    image(tmp_path/'b.png')
    path = tmp_path/'observations.json'
    data = dict(clips=[dict(id='scene', observations=[
        dict(image='a.png',camera_id=9,time_idx=10),
        dict(image='b.png',camera_id=2,time_idx=0)])])
    path.write_text(json.dumps(data))
    visual = load_visual_input(path)
    assert visual.camera_id.tolist() == [[9,2]]
    assert visual.time_idx.tolist() == [[10,0]]
    data['clips'][0]['observations'][0]['camera_id'] = 9.2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='integer'):
        load_visual_input(path)


def test_flat_folder_requires_mode_and_supports_ordinary_filenames(tmp_path):
    image(tmp_path/'apple.png')
    image(tmp_path/'banana.png')
    with pytest.raises(ValueError, match='ambiguous'):
        load_visual_input(tmp_path)
    assert load_visual_input(tmp_path, 'multiview').time_idx.tolist() == [[0,0]]
    assert load_visual_input(tmp_path, 'monocular').time_idx.tolist() == [[0,1]]


def test_disjoint_and_duplicate_camera_frame_labels_fail(tmp_path):
    image(tmp_path/'camera_0'/'frame_0.png')
    image(tmp_path/'camera_1'/'frame_5.png')
    with pytest.raises(ValueError, match='share no frame'):
        load_visual_input(tmp_path)
    image(tmp_path/'camera_0'/'other_0.png')
    with pytest.raises(ValueError, match='duplicate frame'):
        load_visual_input(tmp_path)


def test_video_stride_preserves_original_frame_indices_and_rgb(tmp_path):
    cv2 = pytest.importorskip('cv2')
    path = tmp_path/'sequence.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (42,28))
    assert writer.isOpened()
    try:
        for frame in range(6):
            pixels = np.full((28,42,3), [10,20,200-frame*10], np.uint8)
            writer.write(pixels)
    finally:
        writer.release()
    visual = load_visual_input(path, stride=2, max_frames=3)
    assert visual.mode == 'monocular'
    assert visual.time_idx.tolist() == [[0,2,4]]
    np.testing.assert_allclose(visual.images[0,0,:,14,20].numpy()*255, [200,20,10], atol=4)


def test_surface_identity_padding_and_raw_queries(tmp_path):
    image(tmp_path/'frame_0.png', (160,80))
    image(tmp_path/'frame_5.png', (160,80))
    visual = load_visual_input(tmp_path, 'monocular', long_side=70)
    pred = analytic_prediction(visual)
    queries = np.array([[0,80,40]], np.float32)
    original = queries.copy()
    payload = build_visualization(pred, visual, max_points=0, queries=queries)
    np.testing.assert_array_equal(queries, original)
    processed = [[0,(80+.5)*70/160-.5,(40+.5)*35/80-.5]]
    expected = sample_tracks(pred, torch.tensor([processed]))['trajectory'][0].numpy().transpose(1,0,2)
    np.testing.assert_allclose(payload['tracks']['points'], expected)
    assert len(payload['surface']['source']) == 2*35*70
    for t in range(2):
        surface = payload['surface']
        for source in (0,1):
            mask = surface['source'] == source
            x,y = surface['pixel'][mask].T
            np.testing.assert_allclose(surface['points'][t,mask], pred['trajectory'][0,source,t,y,x])
    assert all(len(frame['points']) == 35*70 for frame in payload['geometry'])
    with pytest.raises(ValueError, match='original source image'):
        build_visualization(pred, visual, queries=[[0,160,40]])
    with pytest.raises(ValueError, match='integer observation'):
        build_visualization(pred, visual, queries=[[.5,80,40]])


def test_exports_use_predictions_and_escape_html(tmp_path):
    pytest.importorskip('plotly')
    image(tmp_path/'frame_0.png')
    image(tmp_path/'frame_5.png')
    visual = load_visual_input(tmp_path, 'monocular')
    visual.clip_id = '</script><script>window.INJECTED=true</script>'
    pred = analytic_prediction(visual)
    pred['trajectory'][0,0,1,0,0] = float('nan')
    payload = build_visualization(pred, visual, max_points=0)
    path = export_visualization(payload, tmp_path/'export')
    html = path.read_text()
    body = re.search(r'<script id="visual-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert json.loads(body)['clip_id'] == visual.clip_id
    assert '<script>window.INJECTED' not in html
    assert 'cdn.plot.ly' not in html.split('</head>')[0] or '<script src=' not in html
    assert 'Plotly.newPlot' in html
    flow = np.load(tmp_path/'export/scene_flow.npz')
    valid = payload['surface']['valid'][0] & payload['surface']['valid'][1]
    np.testing.assert_array_equal(flow['valid'][0], valid)
    np.testing.assert_allclose(flow['displacement'][0,valid], np.tile([.5,-.25,0], (valid.sum(),1)), atol=1e-5)
    assert not flow['displacement'][0,~valid].any()
    ply = (tmp_path/'export/ply/frame_0000_t0_geometry.ply').read_bytes()
    header, binary = ply.split(b'end_header\n')
    assert f"element vertex {len(payload['geometry'][0]['points'])}".encode() in header
    vertices = np.frombuffer(binary, dtype=[('xyz','<f4',(3,)),('rgb','u1',(3,))])
    np.testing.assert_array_equal(vertices['xyz'], payload['geometry'][0]['points'])
    np.testing.assert_array_equal(vertices['rgb'], payload['geometry'][0]['rgb'])


@pytest.mark.parametrize('mode,cameras,times', [('multicamera',2,3),('multiview',3,1),('monocular',1,3)])
def test_real_tiny_inference_all_modes_and_prediction_reuse(tmp_path, mode, cameras, times):
    pytest.importorskip('plotly')
    output = tmp_path/mode
    path = visualize_main(['--demo','--mode',mode,'--output',str(output),
                           '--max-points','32','--num-tracks','4','--cpu-threads','1'])
    assert path.exists()
    pred_path = output/'prediction/clip_000000.pt'
    pred = torch.load(pred_path, weights_only=True)
    assert pred['trajectory'].shape == (1,cameras,times,28,42,3)
    assert pred['camera_id'].unique().numel() == cameras
    assert pred['time_idx'].shape[1] == cameras*times
    expected_times = [[0]] if times == 1 else [[0,5,10]]
    assert pred['query_time_idx'].tolist() == expected_times
    summary = json.loads((output/'visualization.json').read_text())
    assert summary['weights'] == 'random_tiny'
    assert summary['displayed_points'] == 32
    assert len(summary['source_observations']) == cameras
    before = hashlib.sha256(pred_path.read_bytes()).hexdigest()
    visualize_main(['--prediction',str(output),'--output',str(tmp_path/'reuse'),
                    '--max-points','32','--num-tracks','4'])
    assert json.loads((tmp_path/'reuse/visualization.json').read_text())['weights'] == 'random_tiny'
    assert hashlib.sha256(pred_path.read_bytes()).hexdigest() == before
    first, second = np.load(output/'tracks.npz'), np.load(tmp_path/'reuse/tracks.npz')
    for key in first.files:
        np.testing.assert_array_equal(first[key], second[key])
    old = load_visual_input(output/'input/observations.json', long_side=0, max_frames=0)
    new = load_visual_input(tmp_path/'reuse/input/observations.json', long_side=0, max_frames=0)
    assert torch.equal(old.pixel_valid, new.pixel_valid)


def test_one_selected_frame_of_multicamera_stream_is_static(tmp_path):
    directory, mode = make_visual_demo(tmp_path/'streams', 'multicamera')
    visual = load_visual_input(directory, mode, max_frames=1)
    assert visual.mode == 'multiview'
    assert visual.images.shape[1] == 2
