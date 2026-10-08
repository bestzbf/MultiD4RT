import json

import pytest

from scripts.audit_static_scene_regression import compare, read_evaluation


def test_scene_weighting_and_leave_one_out_are_not_clip_weighting():
    # Repeating layouts for one scene must not increase its scene-level weight.
    before = {('a', 4, (1, 2, 3, 4)): {'error': 1.},
              ('a', 8, tuple(range(8))): {'error': 1.},
              ('b', 4, (1, 2, 3, 4)): {'error': 3.}}
    after = {k: {'error': v['error'] + (1 if k[0] == 'a' else 0)}
             for k, v in before.items()}
    result = compare(before, after, 'error')
    assert result['scene_mean_before'] == 2.
    assert result['scene_mean_after'] == 2.5
    assert result['clip_mean_before'] == pytest.approx(5/3)
    assert result['scenes'][0]['signed_share_of_scene_mean_delta'] == 1.
    assert result['leave_one_scene_out'][0]['omitted_scene'] == 'a'
    assert result['leave_one_scene_out'][0]['ratio'] == 1.


def test_unmatched_views_and_nonfinite_metrics_are_rejected():
    before = {('a', 2, (1, 2)): {'error': 1.}}
    with pytest.raises(ValueError, match='different validation cases'):
        compare(before, {('a', 2, (2, 1)): {'error': 1.}}, 'error')
    with pytest.raises(ValueError, match='invalid error'):
        compare(before, {('a', 2, (1, 2)): {'error': float('nan')}}, 'error')


def test_duplicate_cases_are_not_silently_overwritten(tmp_path):
    row = dict(source='blended', sequence='a', actual_cameras=2,
               view_indices=[1, 2], metrics={'error': 1.})
    path = tmp_path/'evaluation.json'
    path.write_text(json.dumps(dict(stage='passed', checkpoints={'model': {'records': [row, row]}})))
    with pytest.raises(ValueError, match='duplicate validation case'):
        read_evaluation(path, 'model', 'blended')
