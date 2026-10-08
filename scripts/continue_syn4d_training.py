"""Admit Syn4D at a recorded checkpoint and continue the existing 100k run.

Data preparation and CPU audits finish while the previous run keeps training.
Only its recorded process tree is stopped. Giant-layout and distributed
optimizer checks then run before admission; failure resumes the old mixture.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
import traceback

from scripts.continue_extended_training import ExtendedTrainingController, next_segment_end
from scripts.continue_quality_recovery import (capture_recorded_tree, digest,
                                              training_argv, validate_checkpoint)
from scripts.cutover_native_motion_stereo import alive, proc, stop_tree, write_json


def verify_state(state, target):
    if (state.get('dense_target_version') != 2
            or state.get('kubric_background_policy') != 'objects_ground'
            or state['lr_plan']['end_step'] != target):
        raise ValueError('resume must preserve corrected GT and the existing extended LR plan')


def distributed_proof(log, expected_sources, begin, end):
    records = []
    for line in Path(log).read_text().splitlines():
        if '{"step":' not in line:
            continue
        records.append(json.loads(line[line.index('{'):]))
    steps = [r['step'] for r in records]
    if steps != list(range(begin+1, end+1)):
        raise ValueError('distributed preflight did not execute every requested update')
    seen = set()
    for row in records:
        if row['world_size'] != 2 or row['optimizer_step_skipped']:
            raise ValueError('distributed preflight skipped an optimizer update')
        if len(row['rank_batches']) != 2:
            raise ValueError('both training ranks must report their samples')
        for batch in row['rank_batches']:
            seen.update(batch['dataset'])
            if not batch.get('syn4d') or batch['metrics']['supervised_trajectory_pixels'] <= 0:
                raise ValueError('a Syn4D training rank did not consume dense GT')
    if set(expected_sources) != seen:
        raise ValueError(f'distributed preflight layout coverage differs: {sorted(seen)}')
    return dict(passed=True, begin=begin, end=end, sources=sorted(seen),
                both_ranks=True, all_updates_applied=True, log=str(log))


def validate_expansion(initial, candidate):
    """Only an explicitly audited expansion of Syn4D scene lists is allowed."""
    from copy import deepcopy
    left, right = deepcopy(initial), deepcopy(candidate)
    left.pop('invalid_log', None); right.pop('invalid_log', None)
    if len(left['sources']) != len(right['sources']):
        raise ValueError('pool expansion must preserve all source families')
    for old, new in zip(left['sources'], right['sources']):
        if old['adapter'] == 'syn4d':
            before = old['options'].pop('scenes')
            after = new['options'].pop('scenes')
            if not set(before) <= set(after):
                raise ValueError('pool expansion removed an admitted scene')
    if left != right:
        raise ValueError('pool expansion changed weights, layouts or another training setting')


def pool_admission_record(config, proof_path):
    proof = json.loads(Path(proof_path).read_text())
    if (not proof.get('passed') or Path(proof['config']).resolve() != Path(config).resolve()
            or digest(config) != proof['config_sha256']):
        raise ValueError('complete pool differs from its passed admission audit')
    return dict(config=config, proof=proof_path, sources=proof['sources'])


class Syn4DTrainingController(ExtendedTrainingController):
    def expanded_config(self, config):
        path = self.job.get('full_data_proof')
        if (not self.progress.get('syn4d_admitted') or not path or not Path(path).is_file()
                or config == self.job['full_data_config']):
            return config
        proof = json.loads(Path(path).read_text())
        if not proof.get('passed'):
            return config
        full = self.job['full_data_config']
        record = pool_admission_record(full, path)
        from multicamera4rc.datasets.mixture import read_data_config
        validate_expansion(read_data_config(config), read_data_config(full))
        self.progress['full_pool_admitted'] = record
        self.publish('full_syn4d_pool_admitted')
        return full

    def cutover(self):
        previous = self.job['previous']
        checkpoint = Path(self.job['initial_checkpoint'])
        deadline = time.monotonic()+24*3600
        self.publish('waiting_atomic_checkpoint')
        while True:
            active = json.loads(self.active.read_text())
            identity = proc(previous['pid'])
            if (active['pid'] != previous['pid'] or not identity or not alive(identity)
                    or identity['start'] != self.job['previous_start']):
                raise RuntimeError('previous training identity changed; refusing cutover')
            if checkpoint.is_file():
                break
            if time.monotonic() > deadline:
                raise TimeoutError('checkpoint did not arrive within 24 hours')
            time.sleep(5)
        for path, expected in self.job['evidence_sha256'].items():
            if digest(path) != expected:
                raise RuntimeError('verified evidence changed: '+path)
        for path in self.job['required_proofs']:
            if not json.loads(Path(path).read_text()).get('passed'):
                raise RuntimeError('required data proof failed: '+path)
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.root, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=self.root, text=True).strip()
        if head != previous['commit'] or dirty:
            raise RuntimeError('live checkout changed; previous run remains running')
        subprocess.run(['git', 'merge-base', '--is-ancestor', head, self.job['commit']], cwd=self.root, check=True)
        state = validate_checkpoint(checkpoint, self.job['initial_step'])
        verify_state(state, self.job['target_step'])
        self.progress['restored_training_state'] = state
        tree = capture_recorded_tree(previous, self.job['previous_start'])
        write_json(self.control/'previous_active_training.json', active)
        self.publish('stopping_recorded_supervisor')
        stop_tree(tree)
        write_json(Path(previous['control_dir'])/'transition.json', dict(
            stage='superseded_by_syn4d', checkpoint=str(checkpoint),
            new_control_dir=str(self.control), stopped_processes=list(tree)))
        subprocess.run(['git', 'merge', '--ff-only', self.job['commit']], cwd=self.root, check=True)
        self.owns_training = True
        self.publish('syn4d_cutover_started')

    def admit(self):
        checkpoint, begin = self.job['initial_checkpoint'], self.job['initial_step']
        giant = self.output/'giant_layouts.json'
        self.run_process('giant_syn4d_layouts', [self.job['python'], '-u', '-m', 'scripts.preflight_syn4d',
            '--checkpoint', checkpoint, '--data-config', self.job['mixed_data_config'],
            '--output', str(giant), '--memory-fraction', '.82'])
        proof = json.loads(giant.read_text())
        if not proof.get('passed') or not proof.get('production_layouts'):
            raise ValueError('native giant layout proof did not pass')
        ddp_config = self.job['ddp_data_config']
        ddp = json.loads(Path(ddp_config).read_text())
        # Two samples of every layout, exactly one distributed epoch.
        end = begin+len(ddp['sources'])
        if ddp['epoch_size'] != 2*len(ddp['sources']):
            raise ValueError('distributed proof requires two samples per layout')
        run = self.output/'ddp_preflight'
        self.run_process('ddp_syn4d_preflight', training_argv(self.job, checkpoint, ddp_config,
                         run, end, reset_cursor=True))
        state = validate_checkpoint(run/'latest.pt', end)
        if (state['lr_plan'] != self.progress['restored_training_state']['lr_plan']
                or state['loss_config'] != self.progress['restored_training_state']['loss_config']):
            raise ValueError('distributed resume changed LR or loss semantics')
        proof = distributed_proof(self.control/'ddp_syn4d_preflight.log',
                                  [s['name'] for s in ddp['sources']], begin, end)
        write_json(self.output/'ddp_proof.json', proof)
        if self.job['mixed_data_config'] == self.job.get('full_data_config'):
            self.progress['full_pool_admitted'] = pool_admission_record(
                self.job['mixed_data_config'], self.job['full_data_proof'])
        self.progress['syn4d_admitted'] = True
        self.publish('syn4d_admitted')

    def evaluate(self, checkpoint, label):
        record = super().evaluate(checkpoint, label)
        if self.progress.get('syn4d_admitted'):
            directory = self.output/'validation'/label/'syn4d'
            self.run_process('validate_'+label+'_syn4d', [self.job['python'], '-u', '-m',
                'multicamera4rc.evaluate', '--data-config', self.job['syn4d_eval_config'],
                '--plan', self.job['syn4d_eval_plan'], '--output', str(directory),
                '--checkpoints', label+'='+str(checkpoint), '--cpu-threads', '4'], evaluation=True)
            path = directory/'evaluation.json'
            if json.loads(path.read_text())['stage'] != 'passed':
                raise RuntimeError('Syn4D held-out evaluation did not finish')
            record['reports']['syn4d'] = str(path)
            # Keep the old quality panel comparable; report the new domain separately.
            write_json(self.output/'validation'/label/'quality.json', record)
        return record

    def run(self):
        try:
            self.cutover()
            with (self.root/'outputs/training_preflight/extended_training.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    self.admit()
                    config = self.job['mixed_data_config']
                except Exception:
                    self.progress['admission_error'] = traceback.format_exc()
                    self.progress['syn4d_admitted'] = False
                    config = self.job['base_data_config']
                    self.publish('syn4d_preflight_failed_resuming_previous_mixture')
                current = self.evaluate(self.job['initial_checkpoint'], 'baseline_'+str(self.job['initial_step']))
                current.update(step=self.job['initial_step'], data_config=self.job['base_data_config'])
                historical = json.loads(Path(self.job['historical_record']).read_text())
                self.progress['best'] = json.loads(Path(self.job['anchor_record']).read_text())
                self.remember_best(current)
                first = True
                while current['step'] < self.job['target_step']:
                    config = self.expanded_config(config)
                    end = next_segment_end(current['step'], self.job['target_step'], first=first,
                                           interval=self.job.get('validation_interval', 2000))
                    current = self.segment(current, self.progress['best'], historical, config, end)
                    self.remember_best(current)
                    self.job['selected_checkpoint'] = current['checkpoint']
                    write_json(self.output/'latest_accepted_checkpoint.json', current)
                    first = False
                self.publish('completed_'+str(self.job['target_step']))
        except Exception:
            self.progress['error'] = traceback.format_exc()
            self.publish('failed')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', required=True)
    Syn4DTrainingController(parser.parse_args().controller).run()
