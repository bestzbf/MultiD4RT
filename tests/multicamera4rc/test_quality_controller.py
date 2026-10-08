import os
from pathlib import Path

import pytest

from scripts import continue_quality_recovery as controller


def test_paired_trial_cannot_hide_a_regression_from_its_initial_checkpoint():
    base = dict(geometry=1., motion=1.)
    assert controller.choose_trial(base, base, dict(geometry=.99, motion=.97))['selected'] == 'candidate'
    assert controller.choose_trial(base, base, base)['selected'] == 'control'
    # Better than an already degraded control is insufficient for migration.
    assert controller.choose_trial(base, dict(geometry=1.5, motion=1.5),
                                   dict(geometry=1.2, motion=1.2))['selected'] == 'control'


def test_trial_commands_preserve_lr_optimizer_and_data_cursor():
    job = dict(python='/venv/bin/python')
    options = dict(resume='/same/step.pt', config='/same/data.yaml', run='/trial', end=32750)
    control = controller.training_argv(job, **options)
    candidate = controller.training_argv(job, **options, time_origin='clip_start')
    assert candidate[:-4] == control[:-2]
    assert candidate[-4:] == ['--time-origin', 'clip_start', '--loss-profile', 'auto']
    for flag in ('--override-lr', '--reset-data-cursor', '--reset-time-projections'):
        assert flag not in control and flag not in candidate
    assert '--prefer-motion-sources' in control and '--no-query-chunking' in control
    loss = controller.training_argv(job, **options, loss_config='/foreground.json')
    assert loss[:-2] == control[:-2]
    assert loss[-2:] == ['--loss-config', '/foreground.json']


def test_four_gpu_continuation_uses_requested_devices_without_changing_lr():
    job = dict(python='/venv/bin/python', physical_gpus=[4, 5, 2, 3], allow_shared=False)
    args = controller.training_argv(job, 'checkpoint.pt', 'data.json', 'out', 56000)
    assert args[args.index('--gpus')+1] == '4,5,2,3'
    assert '--allow-shared' not in args and '--override-lr' not in args
    assert '--reset-data-cursor' not in args


def test_capture_supervisor_requires_exact_pid_start_uid_and_control(monkeypatch):
    previous = dict(pid=999, control_dir='/recorded/job')
    identity = dict(pid=999, ppid=1, uid=os.getuid(), start='12', state='S',
                    cmd='python scripts/queue_variable_continuation.py --controller /recorded/job/job.json ')
    child = dict(pid=1000, ppid=999, uid=os.getuid(), start='13', state='S', cmd='python multicamera4rc.train')
    other = dict(pid=1001, ppid=1, uid=os.getuid(), start='14', state='S', cmd='other user job')
    records = {999: identity, 1000: child, 1001: other}
    monkeypatch.setattr(controller, 'proc', records.get)
    monkeypatch.setattr(Path, 'iterdir', lambda _: iter(map(lambda pid: Path('/proc')/str(pid), records)))
    assert set(controller.capture_recorded_tree(previous, '12')) == {999, 1000}
    with pytest.raises(RuntimeError, match='identity'):
        controller.capture_recorded_tree(previous, 'different')
    identity['cmd'] = 'unrelated controller'
    with pytest.raises(RuntimeError, match='PID'):
        controller.capture_recorded_tree(previous, '12')
    identity['cmd'] = '/recorded/job/supervise.py'
    child['uid'] = os.getuid()+1
    with pytest.raises(RuntimeError, match="another user's"):
        controller.capture_recorded_tree(previous, '12')


@pytest.mark.parametrize('failure', [None, 'train', 'evaluate'])
def test_failed_trial_preserves_control_and_only_selected_model_can_be_best(tmp_path, failure):
    # Exercise the orchestration without GPU work. Every paired branch must
    # resume the same parent, including the second trial after rejecting time.
    import json
    steps = (32000, 34000, 40000)
    job = dict(repo=str(tmp_path), control_dir=str(tmp_path/'control'), output_root=str(tmp_path/'run'),
               initial_checkpoint='/ckpt/32000.pt', initial_step=32000, target_step=34000, trial_steps=250,
               foreground_loss_config='/fg.json', curriculum={'stages': [dict(start_step=s, data_config=f'/data/{s}') for s in steps]})
    (tmp_path/'control').mkdir()
    (tmp_path/'run').mkdir()
    path = tmp_path/'job.json'
    path.write_text(json.dumps(job))
    instance = controller.RecoveryController(path)
    calls, best = [], []
    instance.cutover = lambda: None
    instance.publish = lambda stage: None
    def train(resume, config, begin, end, label, **options):
        calls.append((str(resume), begin, end, label, options))
        if label == 'relative_time_candidate' and failure == 'train':
            raise FloatingPointError('nonfinite candidate gradient')
        return Path(f'/ckpt/{label}_{end}.pt')
    def evaluate(checkpoint, label):
        if 'relative_time_candidate' in label and failure == 'evaluate':
            raise RuntimeError('candidate validation failed')
        value = 1.2 if 'candidate' in label else 1.
        return dict(checkpoint=str(checkpoint), errors={'motion': value}, label=label)
    def remember(record):
        best.append(record['checkpoint'])
        instance.progress['best'] = record
    instance.train, instance.evaluate, instance.remember_best = train, evaluate, remember
    instance.run()
    assert calls[1][0] == calls[2][0]
    assert calls[3][0] == calls[4][0] == '/ckpt/relative_time_control_32750.pt'
    assert calls[5][0] == '/ckpt/foreground_balance_control_33000.pt'
    assert calls[0][4]['reset_cursor']
    assert not any('candidate' in name for name in best)
    assert instance.job['final_checkpoint'] == '/ckpt/continuation_34000.pt'
    if failure:
        assert 'candidate_error' in instance.progress['decisions'][0]
