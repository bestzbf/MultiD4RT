"""Resize a recorded training job using a verified checkpoint.

The old job remains active while waiting. Only its verified process tree is
replaced. A short native DDP check precedes continuation with the same data,
optimizer, LR schedule and validation gates; failed DDP admission falls back
to the original devices and the same accepted checkpoint.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import subprocess
import time
import traceback

from multicamera4rc.launch import gpu_snapshot
from scripts.continue_extended_training import next_segment_end, quality_decision
from scripts.continue_quality_recovery import capture_recorded_tree, digest, training_argv, validate_checkpoint
from scripts.continue_syn4d_training import Syn4DTrainingController, verify_state
from scripts.cutover_native_motion_stereo import alive, proc, stop_tree, write_json
from scripts.resume_audited_replay import verify_replay_config


def recorded_baseline(previous):
    """Reuse the exact evaluated parent of the active recovery job."""
    record = json.loads(Path(previous['initial_record']).read_text())
    if record['checkpoint'] != previous['initial_checkpoint']:
        raise ValueError('evaluated baseline differs from the recorded recovery parent')
    record.update(step=previous['initial_step'], data_config=previous['replay_parent_config'])
    return record


def verify_resize_mixture(job, current):
    """Permit only the already audited replay when reverting to its parent."""
    if current['data_config'] == job['data_config']:
        return False
    previous = job['previous']
    if (not job.get('resume_existing_baseline')
            or current['data_config'] != previous['replay_parent_config']
            or job['data_config'] != previous['data_config']):
        raise RuntimeError('accepted mixture changed; refusing an implicit data migration')
    verify_replay_config(json.loads(Path(current['data_config']).read_text()),
                         json.loads(Path(job['data_config']).read_text()), previous['replay_weights'])
    return True


def distributed_health(path, begin, end, world_size):
    records = []
    for line in Path(path).read_text().splitlines():
        if '{"step":' in line:
            records.append(json.loads(line[line.index('{'):]))
    if [r['step'] for r in records] != list(range(begin + 1, end + 1)):
        raise ValueError('DDP check did not complete every requested update')
    for row in records:
        if (row['world_size'] != world_size or len(row['rank_batches']) != world_size
                or row['optimizer_step_skipped']):
            raise ValueError('DDP check has a missing rank or skipped update')
        for key in ('loss', 'grad_norm', 'lr', 'backbone_lr'):
            if not math.isfinite(row[key]):
                raise ValueError('nonfinite DDP metric: ' + key)
        if any(batch['metrics'].get('supervised_source_pixels', 0) <= 0 for batch in row['rank_batches']):
            raise ValueError('a DDP rank has no valid geometry supervision')
    return dict(passed=True, world_size=world_size, updates=len(records),
                samples=sum(len(r['rank_batches']) for r in records),
                begin=begin, end=end, log=str(path), all_updates_applied=True)


class ResizedTrainingController(Syn4DTrainingController):
    def wait_for_accepted(self):
        previous = self.job['previous']
        accepted = Path(previous['output_root']) / 'latest_accepted_checkpoint.json'
        deadline = time.monotonic() + self.job.get('wait_hours', 24) * 3600
        self.publish('waiting_accepted_checkpoint')
        while True:
            active = json.loads(self.active.read_text())
            if (active['pid'], active['control_dir']) != (previous['pid'], previous['control_dir']):
                raise RuntimeError('another job became active; previous training remains untouched')
            identity = proc(previous['pid'])
            if not identity or not alive(identity) or identity['start'] != self.job['previous_start']:
                raise RuntimeError('previous controller exited before resize; inspect its validation result')
            if self.job.get('resume_existing_baseline'):
                self.publish('resuming_recorded_baseline')
                return recorded_baseline(previous)
            if accepted.is_file():
                record = json.loads(accepted.read_text())
                if record['step'] >= self.job['minimum_step']:
                    return record
            if time.monotonic() >= deadline:
                raise TimeoutError('accepted checkpoint did not arrive before the resize deadline')
            time.sleep(2)

    def verify_and_cutover(self, current):
        previous = self.job['previous']
        for path, expected in self.job['evidence_sha256'].items():
            if digest(path) != expected:
                raise RuntimeError('resize evidence changed: ' + path)
        for path in self.job['required_proofs']:
            if not json.loads(Path(path).read_text()).get('passed'):
                raise RuntimeError('resize verification failed: ' + path)
        reset_cursor = verify_resize_mixture(self.job, current)
        state = validate_checkpoint(current['checkpoint'], current['step'],
            world_size=len(previous['physical_gpus']))
        verify_state(state, self.job['target_step'])
        # Check alignment before stopping the old controller. Mixture epoch_size
        # is explicit in this production job, so no dataset enumeration is needed.
        from multicamera4rc.train import _resume_data_cursor
        config = json.loads(Path(current['data_config']).read_text())
        cursor = _resume_data_cursor(state, current['step'], config['epoch_size'],
                                     len(self.job['physical_gpus']), 1, reset=reset_cursor)
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.root, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=self.root, text=True).strip()
        if head != previous['commit'] or dirty:
            raise RuntimeError('live checkout changed; previous training remains untouched')
        subprocess.run(['git', 'merge-base', '--is-ancestor', head, self.job['commit']], cwd=self.root, check=True)
        old_progress = json.loads((Path(previous['control_dir'])/'progress.json').read_text())
        if self.job.get('resume_existing_baseline'):
            historical = json.loads(Path(self.job['historical_record']).read_text())
            decision = quality_decision(current['errors'], old_progress['best']['errors'],
                                        historical['errors'], current['errors'])
            if decision['action'] != 'continue':
                raise RuntimeError('recorded baseline no longer passes the unchanged quality gate')
            self.progress['baseline_decision'] = decision
        tree = capture_recorded_tree(previous, self.job['previous_start'])
        devices, processes = gpu_snapshot()
        selected = [devices[i] for i in self.job['physical_gpus']]
        uuids = {gpu.uuid for gpu in selected}
        if any(p.gpu_uuid in uuids and p.pid not in tree for p in processes):
            raise RuntimeError('a requested GPU has another task; previous training remains untouched')
        self.job['environment']['CUDA_VISIBLE_DEVICES'] = ','.join(gpu.uuid for gpu in selected)
        self.progress.update(restored_training_state=state, best=old_progress['best'],
            syn4d_admitted=True, full_pool_admitted=old_progress.get('full_pool_admitted'),
            resize=dict(previous_gpus=previous['physical_gpus'], gpus=self.job['physical_gpus'],
                checkpoint=current['checkpoint'], step=current['step'],
                data_epoch=cursor[0], batches_consumed=cursor[1], reset_data_cursor=reset_cursor,
                learning_rate_changed=False))
        self.job.update(initial_step=current['step'], initial_checkpoint=current['checkpoint'],
                        resume_step=current['step'], selected_checkpoint=current['checkpoint'])
        active = json.loads(self.active.read_text())
        if (active['pid'], active['control_dir']) != (previous['pid'], previous['control_dir']):
            raise RuntimeError('active job changed during resize verification')
        write_json(self.control/'previous_active_training.json', active)
        write_json(self.output/'initial_record.json', current)
        self.publish('stopping_recorded_supervisor')
        stop_tree(tree)
        self.owns_training = True
        write_json(Path(previous['control_dir'])/'transition.json', dict(
            stage='superseded_by_gpu_resize', checkpoint=current['checkpoint'],
            new_control_dir=str(self.control), stopped_processes=list(tree)))
        subprocess.run(['git', 'merge', '--ff-only', self.job['commit']], cwd=self.root, check=True)
        self.publish('gpu_resize_started')

    def admit_resize(self, current):
        begin = current['step']
        end = min(begin + self.job.get('preflight_steps', 4), self.job['target_step'])
        run = self.output / 'ddp_preflight'
        self.run_process('ddp_resize_preflight', training_argv(self.job, current['checkpoint'],
            self.job['data_config'], run, end,
            reset_cursor=current['data_config'] != self.job['data_config']))
        world = len(self.job['physical_gpus'])
        state = validate_checkpoint(run/'latest.pt', end, world_size=world)
        for key in ('lr_plan', 'loss_config'):
            if state[key] != self.progress['restored_training_state'][key]:
                raise ValueError('DDP resize changed ' + key)
        proof = distributed_health(self.control/'ddp_resize_preflight.log', begin, end, world)
        write_json(self.output/'ddp_proof.json', proof)
        self.progress['resize']['admitted'] = True
        self.publish('gpu_resize_admitted')

    def run(self):
        try:
            current = self.wait_for_accepted()
            self.verify_and_cutover(current)
            with (self.root/'outputs/training_preflight/extended_training.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    self.admit_resize(current)
                except Exception:
                    self.progress['resize']['admission_error'] = traceback.format_exc()
                    previous = self.job['previous']
                    self.job.update(physical_gpus=previous['physical_gpus'], global_batch_size=previous['global_batch_size'])
                    self.job['environment']['CUDA_VISIBLE_DEVICES'] = previous['environment']['CUDA_VISIBLE_DEVICES']
                    self.publish('resize_preflight_failed_resuming_original_devices')
                # The short probe is diagnostic. Main training restarts at the
                # accepted parent, preserving the checkpoint comparison anchor.
                historical = json.loads(Path(self.job['historical_record']).read_text())
                write_json(self.output/'latest_accepted_checkpoint.json', current)
                first = True
                while current['step'] < self.job['target_step']:
                    end = next_segment_end(current['step'], self.job['target_step'], first=first,
                        interval=self.job.get('validation_interval', 2000))
                    current = self.segment(current, self.progress['best'], historical, self.job['data_config'], end)
                    self.remember_best(current)
                    write_json(self.output/'latest_accepted_checkpoint.json', current)
                    first = False
                self.publish('completed_' + str(self.job['target_step']))
        except Exception:
            self.progress['error'] = traceback.format_exc()
            self.publish('failed' if self.owns_training else 'verification_failed')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', required=True)
    ResizedTrainingController(parser.parse_args().controller).run()
