from copy import deepcopy

import pytest

from scripts.resume_audited_replay import verify_replay_config


def configuration():
    return dict(image_mode='native', invalid_log='previous.jsonl', sources=[
        dict(name='kubric', weight=.864, layouts=[dict(cameras=4, num_times=12)]),
        dict(name='blended', weight=.036, manifest='heldout_disjoint.json'),
        dict(name='syn4d', weight=.1, options=dict(require_human_meshes=True, scenes=['a', 'b']))])


def test_replay_preserves_all_data_while_rebalancing():
    before = configuration()
    after = deepcopy(before)
    after['sources'][0]['weight'] = .82
    after['sources'][1]['weight'] = .08
    after['invalid_log'] = 'new.jsonl'
    verify_replay_config(before, after, dict(kubric=.82, blended=.08))
    assert before == configuration()


@pytest.mark.parametrize('change', ['scene', 'layout', 'unlisted_weight', 'sum'])
def test_replay_rejects_changes_outside_recorded_weights(change):
    before = configuration(); after = deepcopy(before)
    weights = {}
    if change == 'scene':
        after['sources'][2]['options']['scenes'].pop()
    elif change == 'layout':
        after['sources'][0]['layouts'][0]['cameras'] = 1
    elif change == 'unlisted_weight':
        after['sources'][2]['weight'] = .02
    else:
        after['sources'][1]['weight'] = .1
        weights = dict(blended=.1)
    with pytest.raises(ValueError):
        verify_replay_config(before, after, weights)
