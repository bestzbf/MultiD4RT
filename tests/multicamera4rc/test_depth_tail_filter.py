"""Tail labels cannot dominate scale, ray/base GT, or produce fake motion."""
import io
import zipfile

import cv2
import numpy as np
import pytest
import torch

from multicamera4rc.datasets.adapters.dense_geometry import TartanAirV1ZipAdapter, depth_tail_support
from multicamera4rc.datasets.bridge import prepare_clip
from multicamera4rc.datasets.dense import materialize_dense_targets
from multicamera4rc.datasets.mixture import collate_adapter_samples
from multicamera4rc.datasets.partial_dense import attach_partial_dense_recipe


def test_p98_alone_does_not_remove_a_large_sky_but_range_then_p98_does():
    depth = np.full((100, 100), 10., np.float32)
    depth[:25] = 16000.
    original = depth.copy()
    plain, p = depth_tail_support(depth, percentile=98.)
    filtered, f = depth_tail_support(depth, depth_max=200., percentile=98.)
    assert plain.all() and p['cutoff'] == 16000.
    assert not filtered[:25].any() and filtered[25:].all()
    assert f['finite_pixels'] == 10000 and f['range_valid_pixels'] == f['retained_pixels'] == 7500
    np.testing.assert_array_equal(depth, original)


def test_percentile_is_computed_on_finite_positive_in_range_values_only():
    depth = np.linspace(1., 101., 10000, dtype=np.float32).reshape(100, 100)
    depth.flat[:5] = [np.nan, np.inf, -3., 0., 10000.]
    mask, stats = depth_tail_support(depth, depth_max=200., percentile=98.)
    population = depth[np.isfinite(depth) & (depth > 0) & (depth <= 200.)]
    threshold = np.percentile(population, 98.)
    assert stats['cutoff'] == threshold
    expected = np.isfinite(depth) & (depth > 0) & (depth <= threshold)
    np.testing.assert_array_equal(mask, expected)
    assert not mask.flat[:5].any()


def test_constant_depth_plane_and_disabled_filter_retain_valid_labels():
    depth = np.full((10, 10), 200., np.float32)
    mask, stats = depth_tail_support(depth, depth_max=200., percentile=98.)
    assert mask.all() and stats['cutoff'] == 200.
    depth[0, 0] = 10000.; depth[0, 1] = np.nan
    mask, stats = depth_tail_support(depth)
    assert mask[0, 0] and not mask[0, 1] and stats['cutoff'] is None


def test_all_far_depth_is_unknown_rather_than_clamped_to_a_valid_plane():
    depth = np.full((10, 10), 16000., np.float32)
    mask, stats = depth_tail_support(depth, depth_max=200., percentile=98.)
    assert not mask.any() and stats['cutoff'] is None and stats['retained_pixels'] == 0
    assert (depth == 16000.).all()


@pytest.mark.parametrize('options', [dict(depth_max=0), dict(depth_max=float('inf')),
    dict(percentile=0), dict(percentile=101), dict(percentile=float('nan'))])
def test_invalid_tail_policy_is_rejected(options):
    with pytest.raises(ValueError):
        depth_tail_support(np.ones((2, 2)), **options)


def test_native_zip_policy_changes_only_supervision_and_scale(tmp_path):
    torch.set_num_threads(1)
    directory = tmp_path/'synthetic'/'Easy'; directory.mkdir(parents=True)
    base = 'synthetic/Easy/P000'
    depth = np.full((480, 640), 10., np.float16); depth[:120] = 16000.
    depth[200, :3] = [float('nan'), 0., -1.]
    data = io.BytesIO(); np.save(data, depth)
    with zipfile.ZipFile(directory/'depth_left.zip', 'w') as z:
        z.writestr(base+'/pose_left.txt', '0 0 0 0 0 0 1\n')
        z.writestr(base+'/depth_left/000000_left_depth.npy', data.getvalue())
    image = np.full((480, 640, 3), 90, np.uint8)
    ok, encoded = cv2.imencode('.png', image); assert ok
    with zipfile.ZipFile(directory/'image_left.zip', 'w') as z:
        z.writestr(base+'/image_left/000000_left.png', encoded.tobytes())
    original_bytes = (directory/'depth_left.zip').read_bytes()
    raw = TartanAirV1ZipAdapter(tmp_path)
    filtered = TartanAirV1ZipAdapter(tmp_path, depth_max=200., depth_percentile=98.)
    name = base+'/left'
    a, b = raw.load_clip(name, [0]), filtered.load_clip(name, [0])
    np.testing.assert_array_equal(a.images[0], b.images[0])
    np.testing.assert_array_equal(a.depths[0], b.depths[0])
    np.testing.assert_array_equal(a.intrinsics, b.intrinsics)
    np.testing.assert_array_equal(a.extrinsics, b.extrinsics)
    assert (directory/'depth_left.zip').read_bytes() == original_bytes
    targets = []
    for clip in (a, b):
        sample = prepare_clip(clip, image_mode='native', num_queries=0)
        attach_partial_dense_recipe(clip, sample, kind='geometry', image_mode='native')
        batch = collate_adapter_samples([sample])
        targets.append(materialize_dense_targets(batch, torch.tensor([[0]])))
    before, after = targets
    mask = after['depth_valid'][0, 0]
    assert not mask[:120].any() and not mask[480:].any() and not mask[:, 640:].any()
    assert mask.sum() == 360*640-3
    assert after['ray_valid'].sum() == after['source_pixel_valid'].sum() == after['trajectory_valid'].sum() == mask.sum()
    assert not after['displacement_valid'].any() and not after['static_mask'].any()
    assert before['scene_scale'].item() > 100*after['scene_scale'].item()
    points = after['trajectory'][0, 0, 0][mask]
    torch.testing.assert_close(points.norm(dim=-1).mean(), torch.tensor(1.), atol=1e-6, rtol=1e-6)
