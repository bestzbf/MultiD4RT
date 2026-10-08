"""Continue corrected GPU4/5 training to a larger target with bounded recovery.

Adopts a completed short recovery probe. Historical pre-repair metrics remain
visible, while new regressions are measured against corrected, evaluated
checkpoints. A rejected segment rolls back model AND Adam/RNG/cursor, reduces
LR and retries. Data/runtime failures are never hidden or retried indefinitely.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import math
from pathlib import Path
import subprocess
import time
import traceback

from multicamera4rc.quality import compare_errors
from scripts.continue_quality_recovery import (RecoveryController, digest,
                                              training_argv, validate_checkpoint)
from scripts.cutover_native_motion_stereo import alive, proc, write_json


def motion_replay_config(base):
    """Keep all admitted sources/layouts; move 7% from Kubric to motion replay."""
    config = deepcopy(base)
    sources = config['sources']
    total = sum(float(s.get('weight', 1.)) for s in sources)
    groups = {
        'kubric': [s for s in sources if s['adapter'] == 'kubric_multicam'],
        'pointodyssey': [s for s in sources if s['adapter'] == 'pointodyssey_tracks'],
        'vkitti': [s for s in sources if s['adapter'] in ('vkitti2', 'vkitti2_stereo')],
    }
    if total <= 0 or any(not group for group in groups.values()):
        raise ValueError('recovery needs Kubric, PointOdyssey and vKITTI sources')
    for name, delta in [('kubric', -.07), ('pointodyssey', .04), ('vkitti', .03)]:
        group = groups[name]
        original = sum(float(s.get('weight', 1.)) for s in group)
        target = original + delta * total
        if original <= 0 or target <= 0:
            raise ValueError('recovery cannot remove a source family')
        for source in group:
            source['weight'] = float(source.get('weight', 1.)) * target / original
    return config


def quality_decision(reference, best, historical, candidate):
    recent = compare_errors(reference, candidate)
    anchored = compare_errors(best, candidate)
    legacy = compare_errors(historical, candidate)
    # Historical GT migration regressions trigger replay, not an unconditional
    # exit after every subsequent evaluation of the already corrected model.
    unresolved = {k: v for k, v in legacy['metric_ratios'].items()
                  if k.startswith(('native/', 'static/')) and v > 1.25}
    retry = any(v['worst_metric_ratio'] > 1.25 or v['score_ratio'] > 1.10
                for v in (recent, anchored))
    return dict(action='rollback_lower_lr' if retry else 'continue',
                versus_previous=recent, versus_best=anchored,
                versus_pre_repair=legacy, unresolved_historical_metrics=unresolved)


def next_segment_end(step, target, *, first=False, interval=2000):
    if target <= step or interval <= 0:
        raise ValueError('target must be ahead of the current checkpoint')
    return min(target, step + 500 if first else (step // interval + 1) * interval)


def continuation_argv(job, record, config, run, end, *, retry=0):
    reset = Path(record['data_config']).resolve() != Path(config).resolve()
    argv = training_argv(job, record['checkpoint'], config, run, end, reset_cursor=reset)
    trial = record.get('learning_rate_trial')
    if trial:
        if record.get('restart_lr'):
            raise ValueError('LR trial cannot also restart a historical LR phase')
        validate_lr_trial(trial)
        if (record['step'], end) != (trial['start_step'], trial['end_step']):
            raise ValueError('LR trial must use its recorded checkpoint interval')
        if retry == 1:
            # The first fallback restores the reference checkpoint's exact LR
            # schedule, alongside its Adam moments, RNG and sampling cursor.
            return argv
        if retry == 0:
            import torch
            payload = torch.load(record['checkpoint'], map_location='cpu', mmap=True, weights_only=True)
            rates = [float(group['lr'])*trial['multiplier'] for group in payload['optimizer']['param_groups']]
            if len(rates) != 2 or any(not math.isfinite(v) or v <= 0 for v in rates):
                raise ValueError('LR trial requires two finite positive optimizer learning rates')
            return argv + ['--override-lr', '--backbone-lr', str(rates[0]), '--lr', str(rates[1]),
                '--lr-schedule', 'cosine', '--warmup-steps', str(trial['warmup_steps']),
                '--warmup-start-ratio', str(1/trial['multiplier']), '--min-lr-ratio', '.2',
                '--lr-end-step', str(job['target_step'])]
        # Only a failed original-LR fallback proceeds to the usual reduction.
        retry -= 1
    if retry or record.get('restart_lr'):
        if record.get('restart_lr'):
            rates = list(job.get('recovery_learning_rates', [1e-7, 1e-6]))
        else:
            import torch
            payload = torch.load(record['checkpoint'], map_location='cpu', mmap=True, weights_only=True)
            rates = [float(group['lr']) for group in payload['optimizer']['param_groups']]
        rates = [rate * (.5 ** retry) for rate in rates]
        if len(rates) != 2 or any(not math.isfinite(v) or v <= 0 for v in rates):
            raise ValueError('invalid restored optimizer learning rates')
        argv += ['--override-lr', '--backbone-lr', str(rates[0]), '--lr', str(rates[1]),
                 '--lr-schedule', 'cosine', '--warmup-steps', '20',
                 '--warmup-start-ratio', '.5', '--min-lr-ratio', '.2',
                 '--lr-end-step', str(job['target_step'])]
    return argv


def validate_lr_trial(trial):
    if (not isinstance(trial['start_step'], int) or trial['start_step'] < 0 or not isinstance(trial['end_step'], int)
            or not isinstance(trial['warmup_steps'], int)
            or not 0 < trial['warmup_steps'] < trial['end_step']-trial['start_step']
            or not math.isfinite(trial['multiplier']) or not 1 < trial['multiplier'] <= 2):
        raise ValueError('invalid bounded learning-rate trial')
    return trial


def trial_quality_decision(reference, best, historical, candidate, *, trial=False):
    decision = quality_decision(reference, best, historical, candidate)
    if trial:
        recent = decision['versus_previous']
        keep = (decision['action'] == 'continue' and recent['score_ratio'] <= 1.
                and recent['worst_metric_ratio'] <= 1.10)
        decision['learning_rate_trial'] = dict(keep_higher_lr=keep,
            max_score_ratio=1., max_metric_ratio=1.10)
        if not keep:
            decision['action'] = 'rollback_restore_lr'
    return decision


class ExtendedTrainingController(RecoveryController):
    def wait_for_probe(self):
        previous = self.job['previous_probe']
        deadline = time.monotonic() + 12 * 3600
        self.publish('waiting_recovery_probe')
        while True:
            active = json.loads(self.active.read_text())
            if (active['pid'], active['control_dir']) != (previous['pid'], previous['control_dir']):
                raise RuntimeError('another training job became active; refusing overlap')
            identity = proc(previous['pid'])
            if not identity or not alive(identity) or identity['start'] != previous['start']:
                break
            if time.monotonic() > deadline:
                raise TimeoutError('recovery probe did not finish')
            time.sleep(5)
        progress = json.loads((Path(previous['control_dir']) / 'progress.json').read_text())
        if progress['stage'] != 'probe_complete_waiting_continuation':
            raise RuntimeError('recovery probe did not complete: ' + str(progress.get('error', progress['stage'])))
        if progress.get('child_pid'):
            child = proc(progress['child_pid'])
            if child and alive(child):
                raise RuntimeError('a probe child is still running')
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.root, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=self.root, text=True).strip()
        if head != self.job['commit'] or dirty:
            raise RuntimeError('continuation requires the clean verified commit')
        for path, sha in self.job['evidence_sha256'].items():
            if digest(path) != sha:
                raise RuntimeError('continuation evidence changed: ' + path)
        for path in self.job['required_proofs']:
            if not json.loads(Path(path).read_text()).get('passed'):
                raise RuntimeError('continuation proof did not pass: ' + path)
        state = validate_checkpoint(self.job['probe_checkpoint'], self.job['probe_end'])
        if (state.get('dense_target_version') != 2
                or state.get('kubric_background_policy') != 'objects_ground'
                or state['lr_plan']['end_step'] != self.job['target_step']):
            raise RuntimeError('probe did not preserve corrected GT or extend the LR schedule')
        self.progress['restored_training_state'] = state
        self.owns_training = True
        self.publish('extended_training_started')

    def segment(self, reference, best, historical, config, end, *, start_retry=0):
        maximum = self.job.get('max_quality_retries', 2)
        if not 0 <= start_retry <= maximum:
            raise ValueError('retry start is outside the bounded recovery budget')
        for retry in range(start_retry, maximum + 1):
            begin = reference['step']
            run = self.output / f'continuation_{begin}_{end}_try{retry}'
            if run.exists():
                raise RuntimeError('refusing to overwrite an existing training segment: ' + str(run))
            self.job.update(run=str(run), selected_checkpoint=reference['checkpoint'],
                            resume_step=begin, phase_target_step=end, data_config=str(config))
            argv = continuation_argv(self.job, reference, config, run, end, retry=retry)
            self.run_process(f'train_{end}_try{retry}', argv)
            checkpoint = run / 'latest.pt'
            state = validate_checkpoint(checkpoint, end,
                world_size=len(self.job.get('physical_gpus', [4, 5])),
                batch_size=self.job.get('batch_size_per_gpu', 1))
            restored = self.progress['restored_training_state']
            if (state.get('dense_target_version') != 2
                    or state.get('kubric_background_policy') != 'objects_ground'
                    or state['loss_config'] != restored['loss_config']
                    or state['lr_plan']['end_step'] != self.job['target_step']):
                raise RuntimeError('continuation changed the required training contract')
            record = self.evaluate(checkpoint, f'extended_{end}_try{retry}')
            record.update(step=end, data_config=str(config))
            decision = trial_quality_decision(reference['errors'], best['errors'], historical['errors'],
                record['errors'], trial=bool(reference.get('learning_rate_trial')) and retry == 0)
            decision.update(step=end, retry=retry, checkpoint=str(checkpoint),
                            rollback_checkpoint=reference['checkpoint'])
            self.progress['decisions'].append(decision)
            write_json(self.output / 'latest_evaluated_checkpoint.json', record)
            self.publish('quality_checked')
            if decision['action'] == 'continue':
                self.progress['resumed_training_state'] = state
                return record
            # Every retry starts at the same accepted checkpoint, including its
            # moments, source RNG and data cursor. Failed candidates stay on disk.
            self.publish('quality_recovery_restore_lr' if decision['action'] == 'rollback_restore_lr'
                         else 'quality_recovery_lower_lr')
        raise RuntimeError('bounded lower-LR retries exhausted; all checkpoints retained')

    def run(self):
        lock_path = self.root / 'outputs/training_preflight/extended_training.lock'
        with lock_path.open('a') as lock:
            # Do not publish or replace another controller's manifest on lock failure.
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                self.wait_for_probe()
                historical = json.loads(Path(self.job['historical_record']).read_text())
                initial = json.loads(Path(self.job['initial_record']).read_text())
                initial.update(step=self.job['initial_step'], data_config=self.job['base_data_config'], restart_lr=True)
                probe = json.loads(Path(self.job['probe_record']).read_text())
                probe.update(step=self.job['probe_end'], data_config=self.job['base_data_config'])
                decision = quality_decision(initial['errors'], initial['errors'], historical['errors'], probe['errors'])
                self.progress['decisions'].append(dict(phase='probe_adoption', **decision))
                current = probe if decision['action'] == 'continue' else initial
                best = current
                self.progress['best'] = best
                write_json(self.output / 'best_checkpoint.json', best)
                config = (self.job['replay_data_config'] if decision['unresolved_historical_metrics']
                          else self.job['base_data_config'])
                self.progress['using_motion_replay'] = config == self.job['replay_data_config']
                first = True
                while current['step'] < self.job['target_step']:
                    end = next_segment_end(current['step'], self.job['target_step'], first=first,
                                           interval=self.job.get('validation_interval', 2000))
                    current = self.segment(current, best, historical, config, end)
                    self.remember_best(current)
                    best = self.progress['best']
                    self.job['selected_checkpoint'] = current['checkpoint']
                    write_json(self.output / 'latest_accepted_checkpoint.json', current)
                    first = False
                self.publish('completed_' + str(self.job['target_step']))
            except Exception:
                self.progress['error'] = traceback.format_exc()
                self.publish('failed')
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', required=True)
    ExtendedTrainingController(parser.parse_args().controller).run()
