import copy
import json

import pytest
import torch

from multicamera4rc.train import _sources
from multicamera4rc.quality import compare_errors
from scripts.build_quality_curriculum import make_quality_config, replay_layouts, split_replica_manifest
from scripts.build_variable_curriculum import kubric_layouts


def test_anchor_replay_keeps_variable_layouts_and_has_expected_probability():
    original = kubric_layouts(0)
    replay = replay_layouts(original, .5)
    weights = {(r['cameras'], r['num_times']): r['weight'] for r in replay}
    assert sum(weights.values()) == pytest.approx(1.)
    assert {c for c, _ in weights} == set(range(1, 11))
    assert .35*weights[4, 6] == pytest.approx(.07723333333333334)
    before = {(r['cameras'], r['num_times']): r['weight'] for r in original}
    for key in before:
        if key not in ((2, 12), (3, 8), (4, 6)):
            assert weights[key] == pytest.approx(before[key]/2)
    unchanged = {(r['cameras'], r['num_times']): r['weight'] for r in replay_layouts(original, 0)}
    assert unchanged == pytest.approx(before)
    with pytest.raises(ValueError):
        replay_layouts(original, 1.)


def test_replica_split_retains_every_frame_and_every_label(tmp_path):
    manifest = dict(root='/native/replica', sequences={
        'a_left': dict(length=12, valid_frames=[0, 2, 4], physical_scene='a'),
        'a_right': dict(length=12, valid_frames=[0, 2, 4], physical_scene='a')},
        validation_sequences={'b_left': dict(length=6), 'b_right': dict(length=6)})
    path = tmp_path/'original.json'
    path.write_text(json.dumps(manifest))
    original = dict(image_mode='native', sampling='balanced_coverage', sources=[
        dict(name='kubric_variable', adapter='kubric_multicam', weight=35., layouts=kubric_layouts(0)),
        dict(name='dynamic_replica', adapter='dynamic_replica', weight=12., manifest=str(path), dense_kind='replica_flow'),
        dict(name='pointodyssey', adapter='pointodyssey_geometry', weight=12., manifest='/same/point.json')])
    saved = copy.deepcopy(original)
    config, proof = make_quality_config(original, tmp_path/'candidate')
    assert original == saved
    by_name = {s['name']: s for s in config['sources']}
    assert by_name['dynamic_replica_left']['weight'] == 9.
    assert by_name['dynamic_replica_right']['weight'] == 3.
    combined = {}
    for camera in ('left', 'right'):
        source = by_name['dynamic_replica_'+camera]
        assert source['dense_kind'] == 'replica_flow'
        with open(source['manifest']) as stream:
            part = json.load(stream)
        combined.update(part['sequences'])
        assert not set(part['sequences']) & set(part['validation_sequences'])
    assert combined == manifest['sequences']
    assert proof['all_admitted_sequences_retained']
    assert by_name['pointodyssey'] == original['sources'][-1]
    malformed = copy.deepcopy(manifest)
    malformed['sequences']['unknown'] = dict(length=3)
    with pytest.raises(ValueError, match='unrecognized'):
        split_replica_manifest(malformed)


@pytest.mark.parametrize('seed', [1, 9, 83])
def test_native_pair_source_preference_uses_valid_gt_without_adding_labels(seed):
    torch.manual_seed(seed)
    pair_valid = torch.zeros(1, 5, 2, 4, 4, dtype=torch.bool)
    pair_valid[:, 3, 0] = True
    pair_valid[:, 4] = True  # Invalid image cannot become a source.
    batch = dict(observation_valid=torch.tensor([[True, True, True, True, False]]),
                 targets={'dense_pair_valid': pair_valid})
    saved = pair_valid.clone()
    assert _sources(batch, 1, prefer_motion=True).tolist() == [[3]]
    assert torch.equal(saved, pair_valid)
    pair_valid.zero_()
    assert _sources(batch, 2, prefer_motion=True).shape == (1, 2)
    assert not pair_valid.any()


def test_quality_gate_rejects_task_regression_hidden_by_an_improving_average():
    reference = dict(depth=1., tracking=1., motion=1.)
    assert compare_errors(reference, dict(depth=.95, tracking=.95, motion=.95))['accepted']
    assert not compare_errors(reference, dict(depth=.7, tracking=.7, motion=1.2))['accepted']
    assert not compare_errors(reference, dict(depth=1., tracking=1., motion=1.))['accepted']
    with pytest.raises(ValueError, match='same'):
        compare_errors(reference, dict(depth=.5))


def test_time_migration_resets_only_absolute_projections_and_their_adam_state():
    from multicamera4rc import MultiCamera4RC, MultiCamera4RCConfig
    from multicamera4rc.train import _reset_absolute_time_projections
    model = MultiCamera4RC(MultiCamera4RCConfig.tiny(time_origin='clip_start'))
    params = dict(model.named_parameters())
    optimizer = torch.optim.AdamW(model.parameters())
    names = ('backbone.frame_projection.weight', 'target_projection.weight', 'delta_projection.weight')
    for name in names:
        with torch.no_grad():
            params[name].fill_(.2)
        optimizer.state[params[name]] = {'step': torch.tensor(200.), 'exp_avg': torch.ones_like(params[name]), 'exp_avg_sq': torch.ones_like(params[name])}
    reset = _reset_absolute_time_projections(model, optimizer)
    for name in reset:
        assert not params[name].any()
        assert params[name] not in optimizer.state
    assert params['delta_projection.weight'].eq(.2).all()
    assert optimizer.state[params['delta_projection.weight']]['step'] == 200
