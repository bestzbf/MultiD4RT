from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path

import pytest

from multicamera4rc.quality import error_panel
from scripts import migrate_static_validation as migration


def test_handover_lock_is_released_before_waiting_for_training(tmp_path):
    directory = tmp_path/'outputs/training_preflight'
    directory.mkdir(parents=True)
    path = tmp_path/'job.json'
    path.write_text(json.dumps(dict(repo=str(tmp_path), control_dir=str(tmp_path),
        output_root=str(tmp_path), target_step=55500)))
    controller = migration.ExpandedStaticController(path)
    record = dict(step=55500)
    def adopt(lock):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (directory/'controller_handover.lock').open('a') as handover:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handover, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return {'historical': record}
    def finish(_):
        with (directory/'controller_handover.lock').open('a') as handover:
            fcntl.flock(handover, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (directory/'extended_training.lock').open('a') as training:
            with pytest.raises(BlockingIOError):
                fcntl.flock(training, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return record
    controller.adopt = adopt
    controller.finish_adopted = finish
    controller.remember_best = lambda _: None
    controller.publish = lambda _: None
    # Simulate the legacy supervisor's lifetime-long migration lock.
    with (directory/'validation_migration.lock').open('a') as legacy:
        fcntl.flock(legacy, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller.run()


def case(scene, source=0):
    return dict(source=source, sequence=scene, index=0, requested_cameras=4,
                view_indices=[1, 2, 3, 4])


def test_plan_expansion_keeps_original_cases_and_other_sources():
    old = dict(sources=['blended', 'hypersim'], cases=[case('a'), case('h', 1)])
    new = dict(sources=old['sources'], cases=[case('a'), case('b'), case('h', 1)])
    assert migration.validate_expanded_plan(old, new, 2)['plan_sha256'] == migration.plan_digest(new)
    with pytest.raises(ValueError, match='removed'):
        migration.validate_expanded_plan(old, dict(new, cases=[case('b'), case('h', 1)]), 1)
    with pytest.raises(ValueError, match='another static source'):
        migration.validate_expanded_plan(old, dict(new, cases=new['cases']+[case('h2', 1)]), 2)
    with pytest.raises(ValueError, match='duplicate'):
        migration.validate_expanded_plan(old, dict(new, cases=new['cases']+[case('b')]), 2)


def reference_reports(tmp_path):
    metrics = dict(depth_abs_rel=1., base_epe_normalized=1.,
                   trajectory_epe_temporal_normalized=1., displacement_epe_dynamic_normalized=1.)
    geometry = {k: metrics[k] for k in ('depth_abs_rel', 'base_epe_normalized')}
    suites = dict(kubric=dict(mean=metrics), variable=dict(mean=metrics),
                  native_dynamic=dict(by_source={'dynamic_replica': metrics}),
                  static=dict(by_source={'blended': dict(geometry), 'hypersim': dict(geometry)}))
    record = dict(checkpoint=str(tmp_path/'checkpoint.pt'), label='baseline',
                  errors=error_panel(suites), reports={})
    for name, entry in suites.items():
        path = tmp_path/(name+'.json')
        entry.update(path=record['checkpoint'], step=55000)
        path.write_text(json.dumps(dict(stage='passed', metrics_version=migration.METRICS_VERSION,
                                       checkpoints={'baseline': entry})))
        record['reports'][name] = str(path)
    plan = dict(sources=['blended', 'hypersim'], cases=[case('a'), case('b'), case('h', 1)])
    expanded = deepcopy(suites['static'])
    expanded['by_source']['blended']['depth_abs_rel'] = 1.02
    expanded.update(clips=3, records=[dict(source=plan['sources'][c['source']], sequence=c['sequence'],
        actual_cameras=4, view_indices=c['view_indices']) for c in plan['cases']])
    report = dict(stage='passed', metrics_version=migration.METRICS_VERSION,
                  plan_sha256=migration.plan_digest(plan), checkpoints={'baseline': expanded})
    return record, plan, report


def test_rebase_uses_measured_same_checkpoint_and_preserves_other_metrics(tmp_path):
    record, plan, report = reference_reports(tmp_path)
    path = tmp_path/'expanded.json'; path.write_text(json.dumps(report))
    rebased = migration.rebase_record(record, path, plan)
    assert rebased['errors']['static/blended/depth_abs_rel'] == 1.02
    assert record['errors']['static/blended/depth_abs_rel'] == 1.
    assert rebased['reports']['static'] == str(path)
    migration.require_contract(rebased['static_contract'], rebased)
    with pytest.raises(ValueError, match='across static validation contracts'):
        migration.require_contract(rebased['static_contract'], rebased, record)
    record['errors']['static/blended/depth_abs_rel'] = .9
    with pytest.raises(ValueError, match='cached quality panel'):
        migration.rebase_record(record, path, plan)


@pytest.mark.parametrize('fault', ['checkpoint', 'plan', 'coverage', 'other_source'])
def test_rebase_rejects_incompatible_or_incomplete_references(tmp_path, fault):
    record, plan, report = reference_reports(tmp_path)
    entry = report['checkpoints']['baseline']
    if fault == 'checkpoint': entry['path'] = 'different.pt'
    if fault == 'plan': report['plan_sha256'] = 'different'
    if fault == 'coverage': entry['records'].pop()
    if fault == 'other_source': entry['by_source']['hypersim']['depth_abs_rel'] = 2.
    path = tmp_path/'expanded.json'; path.write_text(json.dumps(report))
    with pytest.raises(ValueError): migration.rebase_record(record, path, plan)


def test_handover_signals_only_controller_and_identifies_actual_training_ranks(monkeypatch):
    signals = []
    identity = dict(pid=123, uid=os.getuid(), start='1')
    live = iter([True, False])
    monkeypatch.setattr(migration, 'alive', lambda _: next(live))
    monkeypatch.setattr(migration.os, 'kill', lambda pid, sig: signals.append((pid, sig)))
    migration.stop_controller_only(identity)
    assert signals == [(123, migration.signal.SIGTERM)]
    assert migration.is_training_rank({'cmd': 'python -u -m multicamera4rc.train --steps 55500'})
    assert not migration.is_training_rank({'cmd': 'python -m torch.distributed.run --module multicamera4rc.train'})
    tree = {10: dict(pid=10, ppid=1, cmd='python -m torch.distributed.run --module multicamera4rc.train'),
            11: dict(pid=11, ppid=10, cmd='python -m multicamera4rc.train'),
            12: dict(pid=12, ppid=11, cmd='python -m multicamera4rc.train')}
    assert [r['pid'] for r in migration.training_ranks(tree)] == [11]
    expected = dict(start='1', run='/run', end=55500, data_config='/data.json')
    child = dict(pid=124, ppid=123, uid=os.getuid(), start='1', state='S',
                 cmd='python -m multicamera4rc.launch --output /run --steps 55500 --data-config /data.json')
    migration.require_training_child(child, expected, 123)
    with pytest.raises(RuntimeError, match='identity'):
        migration.require_training_child(dict(child, start='reused'), expected, 123)


def test_second_handover_requires_recorded_orphan_and_unchanged_training_ranks(tmp_path, monkeypatch):
    adopted = dict(pid=124, start='1', run='/run', end=55500, data_config='/data.json')
    previous = dict(pid=123, output_root=str(tmp_path), adopted_training=adopted)
    controller = dict(pid=123, ppid=1, uid=os.getuid(), cmd='python -m scripts.migrate_static_validation')
    child = dict(pid=124, ppid=1, uid=os.getuid(), start='1', state='S',
                 cmd='python -m multicamera4rc.launch --output /run --steps 55500 --data-config /data.json')
    torchrun = dict(pid=125, ppid=124, uid=os.getuid(), cmd='python -m torch.distributed.run --module multicamera4rc.train')
    rank = dict(pid=126, ppid=125, uid=os.getuid(), cmd='python -m multicamera4rc.train')
    records = {r['pid']: r for r in (controller, child, torchrun, rank)}
    proof = dict(passed=True, training_launcher=124, preserved_rank_pids=[126])
    (tmp_path/'handover_proof.json').write_text(json.dumps(proof))
    monkeypatch.setattr(migration, 'capture_recorded_tree', lambda *args: {123: controller})
    monkeypatch.setattr(migration, 'proc', records.get)
    monkeypatch.setattr(Path, 'iterdir', lambda _: iter(Path('/proc')/str(pid) for pid in records))
    assert set(migration.verified_training_tree(previous, '0', adopted)) == set(records)
    with pytest.raises(RuntimeError, match='identity'):
        migration.verified_training_tree(previous, '0', dict(adopted, start='reused'))
    proof['preserved_rank_pids'] = [999]
    (tmp_path/'handover_proof.json').write_text(json.dumps(proof))
    with pytest.raises(RuntimeError, match='rank identities'):
        migration.verified_training_tree(previous, '0', adopted)


@pytest.mark.parametrize('rejected', [False, True])
def test_adopted_completion_keeps_progress_and_consumes_only_remaining_retries(tmp_path, monkeypatch, rejected):
    contract = dict(metrics_version=2, plan_sha256='expanded')
    job = dict(repo=str(tmp_path), control_dir=str(tmp_path), output_root=str(tmp_path),
        target_step=100000, physical_gpus=[4, 5, 2, 3], static_contract=contract,
        adopted_training=dict(pid=123, start='1', run=str(tmp_path/'old_run'), begin=55000,
                              end=55500, log=str(tmp_path/'old.log'), data_config='same.json'))
    path = tmp_path/'job.json'; path.write_text(json.dumps(job))
    controller = migration.ExpandedStaticController(path)
    state = dict(dense_target_version=2, kubric_background_policy='objects_ground',
                 loss_config={'same': True}, lr_plan={'end_step': 100000})
    baseline = dict(checkpoint='baseline.pt', errors={'depth': 1.}, static_contract=contract, step=55000)
    controller.progress.update(best=baseline, restored_training_state=state)
    monkeypatch.setattr(migration, 'alive', lambda _: False)
    monkeypatch.setattr(migration, 'validate_checkpoint', lambda *args, **kwargs: state)
    monkeypatch.setattr(migration, 'distributed_health', lambda *args: {'passed': True})
    controller.publish = lambda _: None
    calls = []
    def evaluate(checkpoint, label):
        calls.append((checkpoint, label))
        return dict(checkpoint=str(checkpoint), label=label, static_contract=contract,
                    errors={'depth': 1.5 if rejected else .99})
    controller.evaluate = evaluate
    retry_calls = []
    def retry(*args, **kwargs):
        retry_calls.append((args, kwargs))
        return dict(baseline, step=55500)
    controller.segment = retry
    result = controller.finish_adopted(dict(initial=baseline, historical=baseline))
    assert result['step'] == 55500
    assert calls == [(tmp_path/'old_run/latest.pt', 'extended_55500_adopted')]
    assert controller.progress['decisions'][0]['retry'] == 0
    if rejected:
        assert retry_calls[0][1] == {'start_retry': 1}
    else:
        assert not retry_calls


def test_controller_continues_from_adopted_checkpoint_without_retraining_old_segment(tmp_path):
    (tmp_path/'outputs/training_preflight').mkdir(parents=True)
    contract = dict(metrics_version=2, plan_sha256='expanded')
    job = dict(repo=str(tmp_path), control_dir=str(tmp_path), output_root=str(tmp_path),
               static_contract=contract, target_step=58000, data_config='same.json')
    path = tmp_path/'job.json'; path.write_text(json.dumps(job))
    c = migration.ExpandedStaticController(path)
    record = dict(checkpoint='55500.pt', step=55500, errors={'a': 1.}, static_contract=contract)
    c.progress['best'] = record
    c.adopt = lambda _: {'historical': record}
    c.finish_adopted = lambda _: record
    c.remember_best = lambda _: None
    c.publish = lambda _: None
    calls = []
    def segment(current, best, historical, config, end):
        calls.append((current['step'], end, config))
        return dict(current, step=end, checkpoint=f'{end}.pt')
    c.segment = segment
    c.run()
    assert calls == [(55500, 56000, 'same.json'), (56000, 58000, 'same.json')]
