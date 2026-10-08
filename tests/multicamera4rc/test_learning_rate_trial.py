from copy import deepcopy
import json

import pytest
import torch

from multicamera4rc.schedule import LearningRatePlan
from scripts import continue_extended_training as extension
from scripts import migrate_static_validation as migration


POLICY = dict(start_step=55500, end_step=56000, multiplier=1.5, warmup_steps=100)


def test_trial_command_preserves_checkpoint_and_smoothly_increases_both_rates(tmp_path):
    checkpoint = tmp_path/'checkpoint.pt'
    torch.save(dict(optimizer={'param_groups': [{'lr': 4.7e-8}, {'lr': 4.7e-7}]}), checkpoint)
    record = dict(step=55500, checkpoint=str(checkpoint), data_config='same.json', learning_rate_trial=POLICY)
    job = dict(python='python', target_step=100000)
    argv = extension.continuation_argv(job, record, 'same.json', tmp_path/'run', 56000)
    assert argv[argv.index('--resume')+1] == str(checkpoint)
    assert '--reset-data-cursor' not in argv
    rates = [float(argv[argv.index(k)+1]) for k in ('--backbone-lr', '--lr')]
    assert rates == pytest.approx([7.05e-8, 7.05e-7])
    plan = LearningRatePlan(mode='cosine', start_step=55500, end_step=int(argv[argv.index('--lr-end-step')+1]),
        base_lrs=rates, warmup_steps=int(argv[argv.index('--warmup-steps')+1]),
        warmup_start_ratio=float(argv[argv.index('--warmup-start-ratio')+1]), min_ratio=.2)
    # The first update is 0.5% above the restored LR; the 100th reaches 1.5x.
    assert plan.factor(55500)*rates[1] == pytest.approx(4.7e-7*1.005)
    assert plan.factor(55599)*rates[1] == pytest.approx(4.7e-7*1.5)
    assert plan.end_step == 100000
    fallback = extension.continuation_argv(job, record, 'same.json', tmp_path/'fallback', 56000, retry=1)
    assert not any(flag in fallback for flag in ('--override-lr', '--lr-schedule', '--reset-data-cursor'))
    assert fallback[fallback.index('--resume')+1] == str(checkpoint)
    reduced = extension.continuation_argv(job, record, 'same.json', tmp_path/'reduced', 56000, retry=2)
    assert float(reduced[reduced.index('--lr')+1]) == pytest.approx(2.35e-7)


def test_trial_never_discards_adam_moments_when_lr_is_overridden():
    parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
    old = torch.optim.AdamW([parameter], lr=4.7e-7)
    parameter.sum().backward(); old.step()
    snapshot = deepcopy(old.state_dict())
    new_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = torch.optim.AdamW([new_parameter], lr=7.05e-7)
    resumed.load_state_dict(snapshot)
    resumed.param_groups[0]['lr'] = 7.05e-7
    for key in ('step', 'exp_avg', 'exp_avg_sq'):
        torch.testing.assert_close(resumed.state[new_parameter][key], old.state[parameter][key])


@pytest.mark.parametrize('fault', ['oversized', 'warmup', 'interval'])
def test_invalid_trial_parameters_are_rejected(fault):
    policy = dict(POLICY)
    if fault == 'oversized': policy['multiplier'] = 4.
    if fault == 'warmup': policy['warmup_steps'] = 500
    if fault == 'interval': policy['end_step'] = 55000
    with pytest.raises(ValueError): extension.validate_lr_trial(policy)


@pytest.mark.parametrize('fault', [None, 'retry', 'wrong_checkpoint', 'unstable', 'failed'])
def test_raise_requires_stable_new_panel_and_no_boundary_recovery(fault):
    reference = dict(step=55500, checkpoint='55500.pt')
    decision = dict(step=55500, checkpoint='55500.pt', retry=0, action='continue',
        versus_previous=dict(score_ratio=.99, worst_metric_ratio=1.02),
        versus_best=dict(score_ratio=1., worst_metric_ratio=1.05))
    if fault == 'retry': decision['retry'] = 1
    if fault == 'wrong_checkpoint': decision['checkpoint'] = 'old.pt'
    if fault == 'unstable': decision['versus_best']['worst_metric_ratio'] = 1.15
    if fault == 'failed': decision['action'] = 'rollback_lower_lr'
    enabled, reason = migration.lr_trial_eligibility(POLICY, reference, [decision], 56000)
    assert enabled == (fault is None)


def test_trial_requires_improvement_and_restores_original_lr_before_reducing():
    reference = dict(a=1., b=1.)
    improved = extension.trial_quality_decision(reference, reference, reference, dict(a=.98, b=.99), trial=True)
    assert improved['action'] == 'continue'
    slight_regression = extension.trial_quality_decision(reference, reference, reference, dict(a=1.01, b=1.), trial=True)
    assert slight_regression['action'] == 'rollback_restore_lr'
    mixed = extension.trial_quality_decision(reference, reference, reference, dict(a=.5, b=1.11), trial=True)
    assert mixed['action'] == 'rollback_restore_lr'
    ordinary = extension.trial_quality_decision(reference, reference, reference, dict(a=1.01, b=1.))
    assert ordinary['action'] == 'continue'


def test_failed_high_lr_trial_falls_back_to_the_original_schedule(tmp_path, monkeypatch):
    path = tmp_path/'job.json'
    path.write_text(json.dumps(dict(repo=str(tmp_path), control_dir=str(tmp_path), output_root=str(tmp_path/'runs'),
        python='python', target_step=100000, max_quality_retries=2)))
    c = extension.ExtendedTrainingController(path); c.output.mkdir()
    c.progress['restored_training_state'] = dict(loss_config={'same': True})
    c.publish = lambda _: None
    checkpoint = tmp_path/'reference.pt'
    torch.save(dict(optimizer={'param_groups': [{'lr': 4.7e-8}, {'lr': 4.7e-7}]}), checkpoint)
    state = dict(dense_target_version=2, kubric_background_policy='objects_ground',
                 loss_config={'same': True}, lr_plan={'end_step': 100000})
    monkeypatch.setattr(extension, 'validate_checkpoint', lambda *args, **kwargs: state)
    commands = []; c.run_process = lambda name, argv: commands.append(argv)
    errors = iter([{'depth': 1.01}, {'depth': .995}])
    c.evaluate = lambda path, label: dict(checkpoint=str(path), label=label, errors=next(errors))
    reference = dict(step=55500, checkpoint=str(checkpoint), data_config='same.json',
                     errors={'depth': 1.}, learning_rate_trial=POLICY)
    result = c.segment(reference, reference, reference, 'same.json', 56000)
    assert '--override-lr' in commands[0] and '--override-lr' not in commands[1]
    assert all(argv[argv.index('--resume')+1] == str(checkpoint) for argv in commands)
    assert c.progress['decisions'][0]['action'] == 'rollback_restore_lr'
    assert c.progress['decisions'][1]['retry'] == 1
    assert result['step'] == 56000


def test_only_the_eligible_controller_segment_receives_the_trial(tmp_path, monkeypatch):
    contract = dict(metrics_version=2, plan_sha256='same-plan')
    job = dict(repo=str(tmp_path), control_dir=str(tmp_path), output_root=str(tmp_path),
               static_contract=contract, learning_rate_trial=POLICY)
    path = tmp_path/'job.json'; path.write_text(json.dumps(job))
    c = migration.ExpandedStaticController(path)
    reference = dict(step=55500, checkpoint='55500.pt', static_contract=contract, errors={'a': 1.})
    c.progress['decisions'] = [dict(step=55500, checkpoint='55500.pt', retry=0, action='continue',
        versus_previous=dict(score_ratio=1., worst_metric_ratio=1.),
        versus_best=dict(score_ratio=1., worst_metric_ratio=1.))]
    seen = []
    def segment(self, record, best, historical, config, end, **options):
        seen.append(record)
        self.progress['decisions'].append(dict(step=end, retry=0))
        self.progress['resumed_training_state'] = dict(lr_plan={'base_lrs': [7.05e-8, 7.05e-7]})
        return dict(reference, step=end, checkpoint=f'{end}.pt')
    monkeypatch.setattr(extension.ExtendedTrainingController, 'segment', segment)
    result = c.segment(reference, reference, reference, 'same.json', 56000)
    assert seen[0]['learning_rate_trial'] == POLICY
    assert 'learning_rate_trial' not in reference
    assert c.progress['learning_rate_trial']['stage'] == 'completed'
    c.segment(result, reference, reference, 'same.json', 58000)
    assert 'learning_rate_trial' not in seen[1]
