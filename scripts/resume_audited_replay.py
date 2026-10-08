"""Resume a stopped quality-gated run after a recorded GT audit and reweighting.

All source definitions, manifests, layouts and quality thresholds are preserved.
This entry point cannot take over a running job or accept an unreviewed model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import subprocess
import traceback

from scripts.continue_extended_training import next_segment_end
from scripts.continue_quality_recovery import digest, validate_checkpoint
from scripts.continue_syn4d_training import Syn4DTrainingController, verify_state
from scripts.cutover_native_motion_stereo import alive, proc, write_json


def verify_replay_config(parent, replay, weights):
    """Require an explicit weight-only change, with no scene/layout removal."""
    expected, actual = deepcopy(parent), deepcopy(replay)
    expected.pop('invalid_log', None); actual.pop('invalid_log', None)
    names = [source['name'] for source in expected['sources']]
    if len(names) != len(set(names)) or not set(weights) <= set(names):
        raise ValueError('replay weights must identify existing unique sources')
    for source in expected['sources']:
        if source['name'] in weights:
            value = float(weights[source['name']])
            if not 0 < value < 1:
                raise ValueError('replay weights must be positive fractions')
            source['weight'] = value
    if expected != actual:
        raise ValueError('replay changed source definitions, layouts or an unlisted weight')
    if abs(sum(float(s['weight']) for s in actual['sources'])-1.) > 1e-8:
        raise ValueError('replay weights must sum to one')


class AuditedReplayController(Syn4DTrainingController):
    def run(self):
        with (self.root/'outputs/training_preflight/extended_training.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                previous = json.loads(Path(self.job['stopped_job']).read_text())
                status = json.loads((Path(previous['control_dir'])/'progress.json').read_text())
                active = json.loads(self.active.read_text())
                if (active['pid'], active['control_dir']) != (previous['pid'], previous['control_dir']):
                    raise RuntimeError('another job became active; refusing overlap')
                for pid in (previous['pid'], status.get('child_pid')):
                    identity = proc(pid) if pid else None
                    if identity and alive(identity):
                        raise RuntimeError('previous controller or child is still running')
                if (status['stage'] != 'failed' or
                        'bounded lower-LR retries exhausted' not in status.get('error', '')):
                    raise RuntimeError('previous stop was not the audited quality-retry limit')
                for path, sha in self.job['evidence_sha256'].items():
                    if digest(path) != sha:
                        raise RuntimeError('verified evidence changed: '+path)
                for path in self.job['required_proofs']:
                    if not json.loads(Path(path).read_text()).get('passed'):
                        raise RuntimeError('required audit failed: '+path)
                parent = json.loads(Path(self.job['replay_parent_config']).read_text())
                config = self.job['data_config']
                verify_replay_config(parent, json.loads(Path(config).read_text()), self.job['replay_weights'])
                head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.root, text=True).strip()
                dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=self.root, text=True).strip()
                if head != self.job['commit'] or dirty:
                    raise RuntimeError('replay requires the clean verified commit')
                current = json.loads(Path(self.job['initial_record']).read_text())
                current.update(step=self.job['initial_step'], data_config=self.job['replay_parent_config'])
                if current['checkpoint'] != self.job['initial_checkpoint']:
                    raise ValueError('evaluated starting checkpoint differs from the resume checkpoint')
                state = validate_checkpoint(current['checkpoint'], current['step'])
                verify_state(state, self.job['target_step'])
                self.progress.update(restored_training_state=state,
                    syn4d_admitted=True, full_pool_admitted=dict(config=config,
                        parent=self.job['replay_parent_config'], weights=self.job['replay_weights']),
                    best=json.loads(Path(self.job['anchor_record']).read_text()))
                historical = json.loads(Path(self.job['historical_record']).read_text())
                self.owns_training = True
                self.publish('audited_replay_started')
                first = True
                while current['step'] < self.job['target_step']:
                    end = next_segment_end(current['step'], self.job['target_step'], first=first,
                        interval=self.job.get('validation_interval', 2000))
                    current = self.segment(current, self.progress['best'], historical, config, end)
                    self.remember_best(current)
                    write_json(self.output/'latest_accepted_checkpoint.json', current)
                    first = False
                self.publish('completed_'+str(self.job['target_step']))
            except Exception:
                self.progress['error'] = traceback.format_exc()
                # A failed pre-ownership verification must not replace an
                # unrelated active/queued training job's state.
                if self.owns_training:
                    self.publish('failed')
                else:
                    write_json(self.control/'progress.json', dict(self.progress, stage='verification_failed'))
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', required=True)
    AuditedReplayController(parser.parse_args().controller).run()
