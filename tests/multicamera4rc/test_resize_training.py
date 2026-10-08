import json
from pathlib import Path

import pytest
import torch

from scripts.continue_quality_recovery import validate_checkpoint
from scripts import resize_training as resize


def test_immediate_resize_reuses_evaluated_parent_and_audited_replay(tmp_path):
    parent = tmp_path/'parent.json'
    replay = tmp_path/'replay.json'
    baseline = tmp_path/'quality.json'
    data = dict(epoch_size=100000, sources=[dict(name='a', weight=.8), dict(name='b', weight=.2)])
    parent.write_text(json.dumps(data))
    data['sources'][0]['weight'] = .7; data['sources'][1]['weight'] = .3
    replay.write_text(json.dumps(data))
    baseline.write_text(json.dumps(dict(checkpoint='accepted.pt', errors={'a': 1.})))
    previous = dict(initial_record=str(baseline), initial_checkpoint='accepted.pt', initial_step=55000,
                    replay_parent_config=str(parent), data_config=str(replay), replay_weights={'a': .7, 'b': .3})
    job = dict(previous=previous, data_config=str(replay), resume_existing_baseline=True)
    record = resize.recorded_baseline(previous)
    assert record['step'] == 55000 and record['data_config'] == str(parent)
    assert resize.verify_resize_mixture(job, record) is True
    assert resize.verify_resize_mixture(job, dict(record, data_config=str(replay))) is False
    job['resume_existing_baseline'] = False
    with pytest.raises(RuntimeError, match='implicit'): resize.verify_resize_mixture(job, record)
    job['resume_existing_baseline'] = True
    data['sources'][0]['root'] = 'unaudited'
    replay.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='source definitions'): resize.verify_resize_mixture(job, record)
    previous['initial_checkpoint'] = 'rejected.pt'
    with pytest.raises(ValueError, match='evaluated baseline'): resize.recorded_baseline(previous)


def test_four_rank_checkpoint_requires_all_optimizer_and_rng_state(tmp_path):
    path = tmp_path/'checkpoint.pt'
    payload = dict(step=55504, optimizer={'state': {0: {'step': torch.tensor(55504)}}},
        training_state=dict(world_size=4, batch_size=1, optimizer_updates=55504),
        rank_rng=[{}, {}, {}, {}])
    torch.save(payload, path)
    assert validate_checkpoint(path, 55504, world_size=4)['world_size'] == 4
    with pytest.raises(ValueError, match='rank count'):
        validate_checkpoint(path, 55504)
    payload['rank_rng'].pop(); torch.save(payload, path)
    with pytest.raises(ValueError, match='every rank'):
        validate_checkpoint(path, 55504, world_size=4)


@pytest.mark.parametrize('failure', [None, 'rank', 'skip', 'nonfinite', 'gt', 'step'])
def test_resize_health_rejects_partial_or_invalid_native_updates(tmp_path, failure):
    row = dict(step=55501, world_size=4, optimizer_step_skipped=False,
        loss=-1., grad_norm=100., lr=4.7e-7, backbone_lr=4.7e-8,
        rank_batches=[{'metrics': {'supervised_source_pixels': 100}} for _ in range(4)])
    if failure == 'rank': row['rank_batches'].pop()
    if failure == 'skip': row['optimizer_step_skipped'] = True
    if failure == 'nonfinite': row['grad_norm'] = float('inf')
    if failure == 'gt': row['rank_batches'][2]['metrics']['supervised_source_pixels'] = 0
    if failure == 'step': row['step'] = 55502
    path = tmp_path/'train.log'; path.write_text('[default0]:' + json.dumps(row) + '\n')
    if failure:
        with pytest.raises(ValueError): resize.distributed_health(path, 55500, 55501, 4)
    else:
        assert resize.distributed_health(path, 55500, 55501, 4)['samples'] == 4


@pytest.mark.parametrize('failed_probe', [False, True])
def test_resized_continuation_keeps_gates_and_falls_back_on_probe_failure(tmp_path, failed_probe):
    control = tmp_path/'control'; control.mkdir()
    output = tmp_path/'run'; output.mkdir()
    (tmp_path/'outputs/training_preflight').mkdir(parents=True)
    historical = tmp_path/'historical.json'; historical.write_text(json.dumps({'errors': {'a': 1.}}))
    job = dict(repo=str(tmp_path), control_dir=str(control), output_root=str(output),
        physical_gpus=[4, 5, 2, 3], global_batch_size=4, target_step=56500, data_config='same.json',
        environment={'CUDA_VISIBLE_DEVICES': 'old0,old1,new0,new1'}, historical_record=str(historical),
        previous=dict(physical_gpus=[4, 5], global_batch_size=2,
                      environment={'CUDA_VISIBLE_DEVICES': 'old0,old1'}))
    path = control/'job.json'; path.write_text(json.dumps(job))
    c = resize.ResizedTrainingController(path)
    record = dict(step=55500, checkpoint='accepted.pt', data_config='same.json', errors={'a': 1.})
    c.wait_for_accepted = lambda: record
    def cutover(current):
        c.owns_training = True
        c.progress.update(best=record, resize={})
    c.verify_and_cutover = cutover
    def probe(current):
        if failed_probe: raise RuntimeError('native DDP failed')
    c.admit_resize = probe
    stages = []; c.publish = stages.append
    calls = []
    def segment(current, best, legacy, config, end):
        calls.append((current['checkpoint'], c.job['physical_gpus'], config, end))
        assert legacy['errors'] == {'a': 1.} and best == record
        return dict(current, step=end, checkpoint=f'{end}.pt')
    c.segment = segment
    c.remember_best = lambda current: None
    c.run()
    expected = [4, 5] if failed_probe else [4, 5, 2, 3]
    assert calls == [('accepted.pt', expected, 'same.json', 56000), ('56000.pt', expected, 'same.json', 56500)]
    assert stages[-1] == 'completed_56500'
    assert ('admission_error' in c.progress['resize']) == failed_probe


def test_failed_wait_never_takes_ownership_of_the_existing_job(tmp_path):
    control = tmp_path/'control'; control.mkdir()
    (tmp_path/'outputs/training_preflight').mkdir(parents=True)
    active = tmp_path/'outputs/training_preflight/active_training.json'
    active.write_text('{"pid": 123}')
    path = control/'job.json'
    path.write_text(json.dumps(dict(repo=str(tmp_path), control_dir=str(control), output_root=str(tmp_path/'run'))))
    c = resize.ResizedTrainingController(path)
    def fail(): raise RuntimeError('a different controller became active')
    c.wait_for_accepted = fail
    with pytest.raises(RuntimeError, match='different controller'): c.run()
    assert json.loads(active.read_text()) == {'pid': 123}
    assert json.loads((control/'progress.json').read_text())['stage'] == 'verification_failed'
