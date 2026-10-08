import json
from pathlib import Path

import pytest

from scripts import continue_extended_training as extension


def test_replay_preserves_every_source_layout_and_family_ratios():
    base = {'sources': [
        {'adapter': 'kubric_multicam', 'weight': 35, 'layouts': [{'cameras': 4, 'num_times': 6}]},
        {'adapter': 'pointodyssey_tracks', 'weight': 12},
        {'adapter': 'vkitti2', 'weight': 4.4},
        {'adapter': 'vkitti2_stereo', 'weight': 6.6},
        {'adapter': 'dynamic_replica', 'weight': 12},
        {'adapter': 'mugen_h5', 'weight': 30},
    ]}
    output = extension.motion_replay_config(base)
    assert [s['weight'] for s in output['sources']] == pytest.approx([28, 16, 5.6, 8.4, 12, 30])
    assert sum(s['weight'] for s in output['sources']) == pytest.approx(100)
    assert base['sources'][0]['weight'] == 35
    for old, new in zip(base['sources'], output['sources']):
        assert {k: v for k, v in old.items() if k != 'weight'} == {k: v for k, v in new.items() if k != 'weight'}


def test_historical_migration_regression_triggers_replay_without_stopping_recovery():
    historical = {'native/pointodyssey/displacement': .001, 'kubric/depth': .3}
    repaired = {'native/pointodyssey/displacement': .0019, 'kubric/depth': .06}
    improved = {'native/pointodyssey/displacement': .0016, 'kubric/depth': .055}
    decision = extension.quality_decision(repaired, repaired, historical, improved)
    assert decision['action'] == 'continue'
    assert decision['unresolved_historical_metrics'] == {'native/pointodyssey/displacement': 1.6}


def test_one_task_regression_cannot_be_hidden_by_an_improving_average():
    before = {'native/pointodyssey/displacement': .001, 'kubric/depth': .3}
    candidate = {'native/pointodyssey/displacement': .0013, 'kubric/depth': .03}
    decision = extension.quality_decision(before, before, before, candidate)
    assert decision['versus_previous']['score_ratio'] < 1
    assert decision['action'] == 'rollback_lower_lr'


def test_best_anchor_prevents_accumulating_small_regressions():
    best = {'native/a': 1.}
    previous = {'native/a': 1.2}
    candidate = {'native/a': 1.3}
    assert extension.quality_decision(previous, best, best, candidate)['action'] == 'rollback_lower_lr'


def test_nonfinite_validation_is_an_error():
    with pytest.raises(ValueError, match='finite'):
        extension.quality_decision({'a': 1.}, {'a': 1.}, {'a': 1.}, {'a': float('nan')})


def test_schedule_continues_beyond_50k_and_stops_exactly_at_100k():
    assert extension.next_segment_end(45300, 100000, first=True) == 45800
    assert extension.next_segment_end(45800, 100000) == 46000
    assert extension.next_segment_end(50000, 100000) == 52000
    assert extension.next_segment_end(99500, 100000) == 100000
    with pytest.raises(ValueError):
        extension.next_segment_end(100000, 100000)


def test_continuation_preserves_optimizer_schedule_and_only_resets_changed_pool(tmp_path):
    record = {'checkpoint': str(tmp_path/'checkpoint.pt'), 'data_config': str(tmp_path/'base.yaml')}
    args = extension.continuation_argv({'python': 'python'}, record, record['data_config'], tmp_path/'run', 100000)
    assert args[args.index('--resume')+1] == record['checkpoint']
    assert '--pretrained' not in args and '--override-lr' not in args
    assert '--reset-data-cursor' not in args
    changed = extension.continuation_argv({'python': 'python'}, record, tmp_path/'replay.yaml', tmp_path/'other', 45800)
    assert '--reset-data-cursor' in changed
    assert '--no-query-chunking' in changed


def test_retry_halves_saved_current_rates_and_extends_decay(tmp_path):
    import torch
    path = tmp_path/'checkpoint.pt'
    torch.save({'optimizer': {'param_groups': [{'lr': 8e-8}, {'lr': 8e-7}]}}, path)
    record = {'checkpoint': str(path), 'data_config': str(tmp_path/'base.yaml')}
    args = extension.continuation_argv({'python': 'python', 'target_step': 100000}, record,
                                      record['data_config'], tmp_path/'run', 45800, retry=1)
    assert float(args[args.index('--backbone-lr')+1]) == pytest.approx(4e-8)
    assert float(args[args.index('--lr')+1]) == pytest.approx(4e-7)
    assert args[args.index('--lr-end-step')+1] == '100000'


def test_rejected_probe_fallback_never_restores_the_old_high_lr_or_50k_end(tmp_path):
    record = {'checkpoint': str(tmp_path/'old.pt'), 'data_config': 'base.yaml', 'restart_lr': True}
    for retry, expected in [(0, 1e-6), (1, 5e-7)]:
        args = extension.continuation_argv({'python': 'python', 'target_step': 100000}, record,
                                          'base.yaml', tmp_path/'run', 45700, retry=retry)
        assert float(args[args.index('--lr')+1]) == pytest.approx(expected)
        assert args[args.index('--lr-end-step')+1] == '100000'
        assert '--resume' in args


def make_controller(tmp_path):
    job = dict(repo=str(tmp_path), control_dir=str(tmp_path), output_root=str(tmp_path/'runs'),
               python='python', target_step=100000, max_quality_retries=2)
    path = tmp_path/'job.json'; path.write_text(json.dumps(job))
    c = extension.ExtendedTrainingController(path)
    c.output.mkdir()
    c.progress['restored_training_state'] = {'loss_config': {'unchanged': True}}
    c.publish = lambda stage: None
    return c


def test_rejected_segment_retries_same_accepted_state_then_progresses(tmp_path, monkeypatch):
    c = make_controller(tmp_path)
    state = dict(dense_target_version=2, kubric_background_policy='objects_ground',
                 loss_config={'unchanged': True}, lr_plan={'end_step': 100000})
    monkeypatch.setattr(extension, 'validate_checkpoint', lambda path, step, **kwargs: state)
    commands = []
    monkeypatch.setattr(extension, 'continuation_argv', lambda job, record, config, run, end, retry:
                        [record['checkpoint'], str(retry)])
    c.run_process = lambda name, argv: commands.append(argv)
    errors = iter([{'a': 1.5}, {'a': .95}])
    c.evaluate = lambda path, label: dict(checkpoint=str(path), label=label, errors=next(errors))
    reference = dict(step=45300, checkpoint='accepted.pt', data_config='base.yaml', errors={'a': 1.})
    result = c.segment(reference, reference, reference, 'base.yaml', 45800)
    assert commands == [['accepted.pt', '0'], ['accepted.pt', '1']]
    assert result['step'] == 45800
    assert c.progress['decisions'][0]['action'] == 'rollback_lower_lr'


def test_persistent_quality_failure_is_bounded_and_keeps_evidence(tmp_path, monkeypatch):
    c = make_controller(tmp_path)
    state = dict(dense_target_version=2, kubric_background_policy='objects_ground',
                 loss_config={'unchanged': True}, lr_plan={'end_step': 100000})
    monkeypatch.setattr(extension, 'validate_checkpoint', lambda path, step, **kwargs: state)
    monkeypatch.setattr(extension, 'continuation_argv', lambda *args, **kwargs: [])
    c.run_process = lambda *args: None
    c.evaluate = lambda path, label: dict(checkpoint=str(path), label=label, errors={'a': 2.})
    reference = dict(step=45300, checkpoint='accepted.pt', data_config='base.yaml', errors={'a': 1.})
    with pytest.raises(RuntimeError, match='retries exhausted'):
        c.segment(reference, reference, reference, 'base.yaml', 45800)
    assert len(c.progress['decisions']) == 3
    assert json.loads((c.output/'latest_evaluated_checkpoint.json').read_text())['errors'] == {'a': 2.}


def test_adopted_trial_retry_budget_does_not_repeat_zero_or_add_an_extra_retry(tmp_path, monkeypatch):
    c = make_controller(tmp_path)
    state = dict(dense_target_version=2, kubric_background_policy='objects_ground',
                 loss_config={'unchanged': True}, lr_plan={'end_step': 100000})
    monkeypatch.setattr(extension, 'validate_checkpoint', lambda path, step, **kwargs: state)
    calls = []
    monkeypatch.setattr(extension, 'continuation_argv', lambda *args, retry: [str(retry)])
    c.run_process = lambda name, argv: calls.append(argv)
    c.evaluate = lambda path, label: dict(checkpoint=str(path), label=label, errors={'a': 2.})
    reference = dict(step=55000, checkpoint='accepted.pt', data_config='base.yaml', errors={'a': 1.})
    with pytest.raises(RuntimeError, match='retries exhausted'):
        c.segment(reference, reference, reference, 'base.yaml', 55500, start_retry=1)
    assert calls == [['1'], ['2']]


def test_controller_adopts_probe_and_automatically_crosses_50k(tmp_path):
    c = make_controller(tmp_path)
    (tmp_path/'outputs/training_preflight').mkdir(parents=True)
    for key, value in [('historical_record', 1.), ('initial_record', 1.9), ('probe_record', 1.6)]:
        path = tmp_path/(key+'.json')
        path.write_text(json.dumps(dict(checkpoint=key+'.pt', errors={'native/a': value})))
        c.job[key] = str(path)
    c.job.update(initial_step=49900, probe_end=50000, base_data_config='base.yaml',
                 replay_data_config='replay.yaml', target_step=52000)
    c.wait_for_probe = lambda: None
    stages = []
    c.publish = stages.append
    calls = []
    def segment(reference, best, historical, config, end):
        calls.append((reference['step'], config, end))
        return dict(checkpoint=f'{end}.pt', errors={'native/a': 1.5}, step=end, data_config=config)
    c.segment = segment
    c.run()
    assert calls == [(50000, 'replay.yaml', 50500), (50500, 'replay.yaml', 52000)]
    assert stages[-1] == 'completed_52000'
    assert c.job['selected_checkpoint'] == '52000.pt'


def test_probe_adoption_cannot_replace_another_active_controller(tmp_path):
    c = make_controller(tmp_path)
    c.active.parent.mkdir(parents=True)
    c.active.write_text(json.dumps(dict(pid=20, control_dir='other')))
    c.job['previous_probe'] = dict(pid=10, control_dir='expected', start='1')
    with pytest.raises(RuntimeError, match='another training job'):
        c.wait_for_probe()
    assert not c.owns_training
