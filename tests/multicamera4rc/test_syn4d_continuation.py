from copy import deepcopy
import json
import hashlib

import pytest

from scripts.continue_syn4d_training import (Syn4DTrainingController, distributed_proof,
                                            pool_admission_record, validate_expansion, verify_state)


def test_direct_full_pool_admission_requires_matching_passed_proof(tmp_path):
    config = tmp_path/'full.json'; config.write_text('{}')
    proof = tmp_path/'proof.json'
    payload = dict(passed=True, config=str(config),
                   config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(), sources=[dict(name='v1')])
    proof.write_text(json.dumps(payload))
    record = pool_admission_record(str(config), str(proof))
    assert record['sources'] == payload['sources']
    config.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='differs'):
        pool_admission_record(str(config), str(proof))
    proof.write_text(json.dumps(dict(passed=False)))
    with pytest.raises(ValueError, match='passed'):
        pool_admission_record(str(config), str(proof))


def test_expansion_only_adds_scenes_without_changing_training_contract():
    initial = dict(seed=49, sources=[dict(adapter='old', weight=.9),
        dict(adapter='syn4d', weight=.1, options=dict(scenes=['a'], subsets=['syn4d_sim']))])
    expanded = deepcopy(initial)
    expanded['sources'][1]['options']['scenes'].append('b')
    validate_expansion(initial, expanded)
    changed = deepcopy(expanded)
    changed['sources'][0]['weight'] = .8
    with pytest.raises(ValueError, match='weights'):
        validate_expansion(initial, changed)
    changed = deepcopy(expanded)
    changed['sources'][1]['options']['scenes'] = ['b']
    with pytest.raises(ValueError, match='removed'):
        validate_expansion(initial, changed)
    assert initial['sources'][1]['options']['scenes'] == ['a']


def test_distributed_proof_requires_updates_and_both_layouts(tmp_path):
    log = tmp_path/'training.log'
    def batch(name):
        return dict(dataset=[name], syn4d=[{}], metrics=dict(supervised_trajectory_pixels=123))
    record = dict(step=53001, world_size=2, optimizer_step_skipped=False,
                  rank_batches=[batch('v1'), batch('sim')])
    def write():
        log.write_text('initialization\n[default0]:'+json.dumps(record)+'\n')
    write()
    assert distributed_proof(log, ['v1','sim'], 53000, 53001)['passed']
    with pytest.raises(ValueError, match='coverage'):
        distributed_proof(log, ['v1','sim','missing'], 53000, 53001)
    record['optimizer_step_skipped'] = True
    write()
    with pytest.raises(ValueError, match='skipped'):
        distributed_proof(log, ['v1','sim'], 53000, 53001)


def test_restored_state_requires_corrected_gt_and_existing_schedule():
    state = dict(dense_target_version=2, kubric_background_policy='objects_ground',
                 lr_plan=dict(end_step=100000))
    verify_state(state, 100000)
    with pytest.raises(ValueError, match='corrected GT'):
        verify_state(dict(state, dense_target_version=1), 100000)
    with pytest.raises(ValueError, match='LR plan'):
        verify_state(state, 50000)


@pytest.mark.parametrize('failure', [False, True])
def test_admission_failure_resumes_old_mix_and_success_uses_new_mix(tmp_path, failure):
    root = tmp_path
    (root/'outputs/training_preflight').mkdir(parents=True)
    output = root/'run'; output.mkdir()
    historical = root/'historical.json'
    historical.write_text(json.dumps(dict(errors={'depth': 1.}, checkpoint='best.pt')))
    job = dict(repo=str(root), control_dir=str(root), output_root=str(output),
               initial_checkpoint='53000.pt', initial_step=53000, target_step=53600,
               base_data_config='old.json', mixed_data_config='new.json',
               historical_record=str(historical), anchor_record=str(historical))
    path = root/'job.json'; path.write_text(json.dumps(job))
    c = Syn4DTrainingController(path)
    c.cutover = lambda: None
    stages, calls = [], []
    c.publish = stages.append
    def admit():
        if failure:
            raise RuntimeError('memory proof failed')
        c.progress['syn4d_admitted'] = True
    c.admit = admit
    c.evaluate = lambda cp, label: dict(checkpoint=cp, errors={'depth': .99})
    def segment(reference, best, historical, config, end):
        calls.append((reference['step'], config, end))
        return dict(checkpoint=str(end)+'.pt', errors={'depth': .98}, step=end, data_config=config)
    c.segment = segment
    c.run()
    expected = 'old.json' if failure else 'new.json'
    assert calls == [(53000, expected, 53500), (53500, expected, 53600)]
    assert stages[-1] == 'completed_53600'
    assert bool(c.progress.get('admission_error')) == failure


def test_pool_expansion_waits_for_passed_proof_and_checks_config_hash(tmp_path):
    output = tmp_path/'out'; output.mkdir()
    proof = tmp_path/'proof.json'
    job = dict(repo=str(tmp_path), control_dir=str(tmp_path), output_root=str(output),
               full_data_proof=str(proof), full_data_config=str(tmp_path/'full.json'))
    path = tmp_path/'job.json'; path.write_text(json.dumps(job))
    c = Syn4DTrainingController(path)
    c.progress['syn4d_admitted'] = True
    assert c.expanded_config('initial.json') == 'initial.json'
    proof.write_text(json.dumps(dict(passed=False)))
    assert c.expanded_config('initial.json') == 'initial.json'
    proof.write_text(json.dumps(dict(passed=True, config='wrong.json')))
    with pytest.raises(ValueError, match='differs'):
        c.expanded_config('initial.json')
