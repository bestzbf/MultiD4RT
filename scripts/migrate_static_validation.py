"""Migrate fixed static validation while preserving an active training segment.

All reference checkpoints must first be evaluated on the expanded fixed plan.
Only the identified controller process is replaced; its launcher and training
ranks continue to the original checkpoint boundary. No model, optimizer, data
cursor, learning rate or quality threshold is changed by this handover.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import time
import traceback

from multicamera4rc.metrics import METRICS_VERSION
from multicamera4rc.quality import error_panel
from scripts.continue_extended_training import next_segment_end, quality_decision, validate_lr_trial
from scripts.continue_quality_recovery import capture_recorded_tree, digest, validate_checkpoint
from scripts.continue_syn4d_training import Syn4DTrainingController, verify_state
from scripts.cutover_native_motion_stereo import alive, proc, write_json
from scripts.resize_training import distributed_health


def plan_digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def validate_expanded_plan(old, new, expected_scenes):
    if old['sources'] != new['sources']:
        raise ValueError('static source ordering changed')
    source = new['sources'].index('blended')
    if not all(case in new['cases'] for case in old['cases']):
        raise ValueError('expanded validation removed an original case')
    if [c for c in old['cases'] if c['source'] != source] != [
            c for c in new['cases'] if c['source'] != source]:
        raise ValueError('another static source changed during Blended expansion')
    cases = [c for c in new['cases'] if c['source'] == source]
    if len({c['sequence'] for c in cases}) != expected_scenes:
        raise ValueError('expanded plan does not cover the required Blended scenes')
    identities = [(c['source'], c['sequence'], tuple(c['view_indices'])) for c in new['cases']]
    if len(identities) != len(set(identities)):
        raise ValueError('duplicate static validation views')
    return dict(metrics_version=METRICS_VERSION, plan_sha256=plan_digest(new))


def checkpoint_entry(report, record):
    if report.get('stage') != 'passed' or report.get('metrics_version') != METRICS_VERSION:
        raise ValueError('validation report is incomplete or has a different metrics version')
    matches = [v for v in report['checkpoints'].values()
               if Path(v['path']).resolve() == Path(record['checkpoint']).resolve()]
    if len(matches) != 1:
        raise ValueError('report does not identify exactly the reference checkpoint')
    return matches[0]


def verify_static_cases(report, record, plan):
    entry = checkpoint_entry(report, record)
    if report['plan_sha256'] != plan_digest(plan):
        raise ValueError('static evaluation plan differs from the admitted plan')
    expected = sorted((plan['sources'][c['source']], c['sequence'], tuple(c['view_indices']))
                      for c in plan['cases'])
    actual = sorted((r['source'], r['sequence'], tuple(r['view_indices'])) for r in entry['records'])
    if actual != expected or entry['clips'] != len(expected):
        raise ValueError('static report has missing or extra cases')
    for row in entry['records']:
        if row['actual_cameras'] != len(row['view_indices']):
            raise ValueError('static report camera count differs from its views')
    return entry


def rebase_record(record, expanded_path, plan):
    """Recompute the panel from measured reports, with no cross-plan ratios."""
    original = {}
    for suite in ('kubric', 'native_dynamic', 'static', 'variable'):
        report = json.loads(Path(record['reports'][suite]).read_text())
        original[suite] = checkpoint_entry(report, record)
    if error_panel(original) != record['errors']:
        raise ValueError('cached quality panel differs from its source reports')
    expanded = json.loads(Path(expanded_path).read_text())
    replacement = verify_static_cases(expanded, record, plan)
    suites = dict(original, static=replacement)
    panel = error_panel(suites)
    if set(panel) != set(record['errors']):
        raise ValueError('validation migration changed the metric set')
    if any(panel[k] != v for k, v in record['errors'].items()
           if not k.startswith('static/blended/')):
        raise ValueError('validation migration changed metrics outside BlendedMVS')
    result = deepcopy(record)
    result.update(errors=panel, step=replacement['step'],
                  static_contract=dict(metrics_version=expanded['metrics_version'],
                                       plan_sha256=expanded['plan_sha256']))
    result['reports']['static'] = str(expanded_path)
    return result


def require_contract(contract, *records):
    if any(record.get('static_contract') != contract for record in records):
        raise ValueError('refusing quality comparison across static validation contracts')


def require_training_child(identity, expected, previous_pid, *, allow_recorded_orphan=False):
    if (not identity or identity['start'] != expected['start']
            or identity['uid'] != os.getuid() or identity['state'] == 'Z'
            or (identity['ppid'] != previous_pid and not (allow_recorded_orphan and identity['ppid'] == 1))):
        raise RuntimeError('recorded training launcher identity changed')
    args = shlex.split(identity['cmd'])
    if 'multicamera4rc.launch' not in args:
        raise RuntimeError('recorded child is not the training launcher')
    for flag, value in (('--output', expected['run']), ('--steps', str(expected['end'])),
                        ('--data-config', expected['data_config'])):
        if flag not in args or args[args.index(flag)+1] != value:
            raise RuntimeError('recorded training command changed: '+flag)


def stop_controller_only(identity):
    """Never signal the launcher's process group or any training child."""
    if identity['uid'] != os.getuid() or not alive(identity):
        raise RuntimeError('controller identity changed before handover')
    os.kill(identity['pid'], signal.SIGTERM)
    deadline = time.monotonic()+10
    while alive(identity):
        if time.monotonic() > deadline:
            raise RuntimeError('controller did not exit; training processes were not signalled')
        time.sleep(.1)


def is_training_rank(identity):
    args = shlex.split(identity['cmd'])
    return '-m' in args and args[args.index('-m')+1] == 'multicamera4rc.train'


def training_ranks(tree):
    # Forked DataLoader workers inherit the rank's command line. Their parent
    # is the rank; the rank itself is a direct child of the torchrun launcher.
    return [r for r in tree.values() if is_training_rank(r)
            and (r['ppid'] not in tree or not is_training_rank(tree[r['ppid']]))]


def verified_training_tree(previous, previous_start, adopted):
    tree = capture_recorded_tree(previous, previous_start)
    child = proc(adopted['pid'])
    if adopted['pid'] in tree:
        require_training_child(child, adopted, previous['pid'])
        return tree
    # A prior controller-only handover leaves the verified launcher parented
    # to init. Require that handover's proof before adopting it a second time.
    prior = previous.get('adopted_training', {})
    if any(prior.get(k) != adopted[k] for k in ('pid', 'start', 'run', 'end', 'data_config')):
        raise RuntimeError('orphan launcher lacks the prior adoption identity')
    proof = json.loads((Path(previous['output_root'])/'handover_proof.json').read_text())
    if not proof.get('passed') or proof.get('training_launcher') != adopted['pid']:
        raise RuntimeError('orphan launcher lacks a passed handover proof')
    require_training_child(child, adopted, previous['pid'], allow_recorded_orphan=True)
    descendants = {child['pid']: child}
    processes = [proc(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()]
    while True:
        added = [r for r in processes if r and r['ppid'] in descendants and r['pid'] not in descendants]
        if not added:
            break
        descendants.update({r['pid']: r for r in added})
    if any(r['uid'] != os.getuid() for r in descendants.values()):
        raise RuntimeError('orphan training tree contains another user')
    if {r['pid'] for r in training_ranks(descendants)} != set(proof['preserved_rank_pids']):
        raise RuntimeError('previously preserved training rank identities changed')
    tree.update(descendants)
    return tree


def lr_trial_eligibility(policy, reference, decisions, end):
    validate_lr_trial(policy)
    if (reference['step'], end) != (policy['start_step'], policy['end_step']):
        return False, 'outside_trial_interval'
    if not decisions:
        return False, 'no_completed_boundary_validation'
    last = decisions[-1]
    if (last.get('step') != reference['step'] or last.get('checkpoint') != reference['checkpoint']
            or last.get('retry') != 0 or last.get('action') != 'continue'):
        return False, 'boundary_required_recovery_or_did_not_pass'
    if any(last[key]['score_ratio'] > 1.02 or last[key]['worst_metric_ratio'] > 1.10
           for key in ('versus_previous', 'versus_best')):
        return False, 'boundary_validation_not_stable_enough'
    return True, 'passed_matched_validation_without_recovery'


class ExpandedStaticController(Syn4DTrainingController):
    def contract(self):
        return self.job['static_contract']

    def evaluate(self, checkpoint, label):
        # Check the frozen plan before inference and validate every returned case.
        plan = json.loads(Path(self.job['evaluation']['static_plan']).read_text())
        if plan_digest(plan) != self.contract()['plan_sha256']:
            raise ValueError('admitted static plan changed on disk')
        record = super().evaluate(checkpoint, label)
        report = json.loads(Path(record['reports']['static']).read_text())
        verify_static_cases(report, record, plan)
        record['static_contract'] = self.contract()
        write_json(self.output/'validation'/label/'quality.json', record)
        return record

    def segment(self, reference, best, historical, config, end, **options):
        require_contract(self.contract(), reference, best, historical)
        policy = self.job.get('learning_rate_trial')
        enabled = False
        if policy and reference['step'] == policy['start_step'] and not options.get('start_retry'):
            enabled, reason = lr_trial_eligibility(policy, reference, self.progress['decisions'], end)
            self.progress['learning_rate_trial'] = dict(policy=policy, enabled=enabled, reason=reason,
                                                       stage='running' if enabled else 'skipped')
            if enabled:
                reference = dict(reference, learning_rate_trial=policy)
            write_json(self.output/'learning_rate_trial.json', self.progress['learning_rate_trial'])
        result = super().segment(reference, best, historical, config, end, **options)
        if enabled:
            self.progress['learning_rate_trial'].update(stage='completed',
                selected_retry=self.progress['decisions'][-1]['retry'],
                selected_checkpoint=result['checkpoint'],
                accepted_lr_plan=self.progress['resumed_training_state']['lr_plan'])
            write_json(self.output/'learning_rate_trial.json', self.progress['learning_rate_trial'])
        return result

    def remember_best(self, record):
        require_contract(self.contract(), record, self.progress['best'])
        return super().remember_best(record)

    def verify_handover(self):
        previous, adopted = self.job['previous'], self.job['adopted_training']
        if self.job.get('learning_rate_trial'):
            policy = validate_lr_trial(self.job['learning_rate_trial'])
            if policy['start_step'] != adopted['end'] or policy['end_step'] > self.job['target_step']:
                raise ValueError('LR trial is not attached to the adopted checkpoint boundary')
            if self.job.get('max_quality_retries', 2) < 1:
                raise ValueError('LR trial requires a budgeted original-LR fallback')
        for path, expected in self.job['evidence_sha256'].items():
            if digest(path) != expected:
                raise RuntimeError('validation migration evidence changed: '+path)
        for path in self.job['required_proofs']:
            if not json.loads(Path(path).read_text()).get('passed'):
                raise RuntimeError('validation migration proof failed: '+path)
        import subprocess
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.root, text=True).strip() != self.job['commit']:
            raise RuntimeError('controller checkout differs from the tested commit')
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=self.root, text=True).strip():
            raise RuntimeError('controller checkout is dirty')
        old_plan = json.loads(Path(previous['evaluation']['static_plan']).read_text())
        plan = json.loads(Path(self.job['evaluation']['static_plan']).read_text())
        contract = validate_expanded_plan(old_plan, plan, self.job['blended_scene_count'])
        if contract != self.contract():
            raise ValueError('validation contract does not identify the frozen plan')
        refs = {name: json.loads(Path(self.job[name+'_record']).read_text())
                for name in ('initial', 'anchor', 'historical')}
        require_contract(contract, *refs.values())
        for record in refs.values():
            report = json.loads(Path(record['reports']['static']).read_text())
            verify_static_cases(report, record, plan)
        active = json.loads(self.active.read_text())
        if (active['pid'], active['control_dir']) != (previous['pid'], previous['control_dir']):
            raise RuntimeError('another training job became active')
        status = json.loads((Path(previous['control_dir'])/'progress.json').read_text())
        if status.get('child_pid') != adopted['pid'] or not status['stage'].startswith('train_'):
            raise RuntimeError('previous controller has moved beyond the recorded training segment')
        if (active['data_config'] != adopted['data_config'] or active['run'] != adopted['run']
                or active['physical_gpus'] != self.job['physical_gpus']):
            raise RuntimeError('active training settings changed during preparation')
        if status['best']['checkpoint'] != refs['anchor']['checkpoint']:
            raise RuntimeError('best checkpoint changed during preparation')
        tree = verified_training_tree(previous, self.job['previous_start'], adopted)
        ranks = training_ranks(tree)
        if len(ranks) != len(self.job['physical_gpus']):
            raise RuntimeError('not all expected training ranks are present')
        if refs['initial']['checkpoint'] != active['selected_checkpoint']:
            raise RuntimeError('current reference checkpoint changed during preparation')
        decision = quality_decision(refs['initial']['errors'], refs['anchor']['errors'],
                                    refs['historical']['errors'], refs['initial']['errors'])
        if decision['action'] != 'continue':
            raise RuntimeError('current accepted model failed the recalibrated unchanged gate')
        return refs, status, tree, decision

    def adopt(self, lock):
        refs, status, tree, decision = self.verify_handover()
        previous, adopted = self.job['previous'], self.job['adopted_training']
        write_json(self.control/'previous_progress.json', status)
        write_json(self.control/'preserved_processes.json', list(tree.values()))
        # Complete all preparation before replacing the supervisor. No rank is
        # restarted, so in-flight Adam/RNG/sampler state stays in memory.
        stop_controller_only(tree[previous['pid']])
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.owns_training = True
        self.progress = deepcopy(status)
        self.progress.update(best=refs['anchor'], baseline_decision=decision,
                             child_pid=adopted['pid'], child_log=adopted['log'])
        self.progress['validation_migration'] = dict(
            static_contract=self.contract(), blended_scenes=self.job['blended_scene_count'],
            preserved_training_launcher=adopted['pid'], old_controller=previous['pid'],
            training_restarted=False, learning_rate_changed=False)
        if self.job.get('learning_rate_trial'):
            self.progress['learning_rate_trial'] = dict(policy=self.job['learning_rate_trial'],
                stage='pending_boundary_validation', enabled=False)
            write_json(self.output/'learning_rate_trial.json', self.progress['learning_rate_trial'])
        self.remember_best(refs['initial'])
        write_json(self.output/'latest_accepted_checkpoint.json', refs['initial'])
        write_json(Path(previous['control_dir'])/'transition.json', dict(
            stage='superseded_by_expanded_static_validation', new_control_dir=str(self.control),
            stopped_processes=[previous['pid']], preserved_training_launcher=adopted['pid']))
        survivors = training_ranks(tree)
        if not alive(tree[adopted['pid']]) or not all(alive(r) for r in survivors):
            raise RuntimeError('training process exited during controller handover')
        write_json(self.output/'handover_proof.json', dict(passed=True,
            stopped_controller=previous['pid'], training_launcher=adopted['pid'],
            preserved_rank_pids=[r['pid'] for r in survivors], static_contract=self.contract(),
            training_restarted=False))
        self.publish('train_'+str(adopted['end'])+'_adopted')
        return refs

    def finish_adopted(self, refs):
        adopted = self.job['adopted_training']
        deadline = time.monotonic()+self.job.get('adopted_wait_hours', 24)*3600
        while alive(adopted):
            if time.monotonic() > deadline:
                raise TimeoutError('adopted segment exceeded its original completion window')
            time.sleep(5)
        checkpoint = Path(adopted['run'])/'latest.pt'
        state = validate_checkpoint(checkpoint, adopted['end'], world_size=len(self.job['physical_gpus']))
        verify_state(state, self.job['target_step'])
        restored = self.progress['restored_training_state']
        if any(state[k] != restored[k] for k in ('loss_config', 'lr_plan')):
            raise ValueError('adopted training changed optimizer schedule or loss semantics')
        proof = distributed_health(adopted['log'], adopted['begin'], adopted['end'], len(self.job['physical_gpus']))
        write_json(self.output/'adopted_training_proof.json', proof)
        self.progress.pop('child_pid', None)
        record = self.evaluate(checkpoint, 'extended_'+str(adopted['end'])+'_adopted')
        record.update(step=adopted['end'], data_config=adopted['data_config'])
        require_contract(self.contract(), refs['initial'], self.progress['best'], refs['historical'], record)
        decision = quality_decision(refs['initial']['errors'], self.progress['best']['errors'],
                                    refs['historical']['errors'], record['errors'])
        decision.update(step=adopted['end'], retry=0, checkpoint=str(checkpoint),
                        rollback_checkpoint=refs['initial']['checkpoint'], adopted=True)
        self.progress['decisions'].append(decision)
        write_json(self.output/'latest_evaluated_checkpoint.json', record)
        self.publish('quality_checked')
        if decision['action'] == 'continue':
            self.progress['resumed_training_state'] = state
            return record
        return self.segment(refs['initial'], self.progress['best'], refs['historical'],
                            adopted['data_config'], adopted['end'], start_retry=1)

    def run(self):
        # Legacy controllers retain validation_migration.lock for their whole
        # lifetime. Use a handover-only lock so a verified second adoption can
        # stop that controller without restarting its orphaned training ranks.
        directory = self.root/'outputs/training_preflight'
        with (directory/'controller_handover.lock').open('a') as migration, (
                directory/'extended_training.lock').open('a') as lock:
            fcntl.flock(migration, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                refs = self.adopt(lock)
                fcntl.flock(migration, fcntl.LOCK_UN)
                current = self.finish_adopted(refs)
                self.remember_best(current)
                write_json(self.output/'latest_accepted_checkpoint.json', current)
                while current['step'] < self.job['target_step']:
                    end = next_segment_end(current['step'], self.job['target_step'],
                                           interval=self.job.get('validation_interval', 2000))
                    current = self.segment(current, self.progress['best'], refs['historical'], self.job['data_config'], end)
                    self.remember_best(current)
                    write_json(self.output/'latest_accepted_checkpoint.json', current)
                self.publish('completed_'+str(self.job['target_step']))
            except Exception:
                self.progress['error'] = traceback.format_exc()
                if self.owns_training:
                    self.publish('failed')
                else:
                    write_json(self.control/'progress.json', dict(self.progress, stage='verification_failed'))
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    controller = ExpandedStaticController(args.controller)
    if args.verify_only:
        refs, status, tree, decision = controller.verify_handover()
        print(json.dumps(dict(passed=True, reference_labels=[r['label'] for r in refs.values()],
                              old_stage=status['stage'], processes=len(tree), baseline_decision=decision)))
    else:
        controller.run()
