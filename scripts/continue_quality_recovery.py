"""Continue a recorded GPU4/5 run with paired trials and fixed quality gates.

Preparation records the exact supervisor, code and evidence. The controller
waits for an atomic checkpoint before stopping only that supervisor's process
tree, fast-forwards the clean checkout, and restores AdamW, LR and rank RNG.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from multicamera4rc.quality import compare_errors, error_panel
from scripts.cutover_native_motion_stereo import alive, proc, stop_tree, write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture_recorded_tree(previous, expected_start):
    root = proc(previous['pid'])
    control = Path(previous['control_dir'])
    if not root or root['uid'] != os.getuid() or root['start'] != expected_start or root['state'] == 'Z':
        raise RuntimeError('recorded supervisor identity changed')
    argv = shlex.split(root['cmd'])
    accepted = (str(control/'supervise.py') in argv or any(
        left == '--controller' and right == str(control/'job.json') for left, right in zip(argv, argv[1:])))
    if not accepted:
        raise RuntimeError('PID does not identify the recorded training controller')
    records = [proc(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()]
    tree = {root['pid']: root}
    while True:
        children = [p for p in records if p and p['ppid'] in tree and p['pid'] not in tree]
        if not children:
            break
        tree.update({p['pid']: p for p in children})
    if any(p['uid'] != os.getuid() for p in tree.values()):
        raise RuntimeError("refusing to stop another user's process")
    return tree


def validate_checkpoint(path, expected_step, *, world_size=2, batch_size=1):
    import torch
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    state = payload['training_state']
    if payload['step'] != expected_step or not payload['optimizer']['state']:
        raise ValueError('expected an atomic checkpoint with optimizer moments')
    if (state['world_size'], state['batch_size'], state['optimizer_updates']) != (world_size, batch_size, expected_step):
        raise ValueError('checkpoint batch size, rank count or optimizer updates changed')
    if len(payload.get('rank_rng', [])) != world_size:
        raise ValueError('every rank RNG state must be available')
    return state


def training_argv(job, resume, config, run, end, *, reset_cursor=False, time_origin=None, loss_config=None):
    gpus = job.get('physical_gpus', [4, 5])
    if not gpus or len(set(gpus)) != len(gpus) or any(not isinstance(g, int) or g < 0 for g in gpus):
        raise ValueError('physical_gpus must contain unique nonnegative indices')
    command = [job['python'], '-u', '-m', 'multicamera4rc.launch', '--gpus', ','.join(map(str, gpus)),
        '--resume', str(resume), '--data-config', str(config), '--output', str(run), '--preset', 'giant',
        '--steps', str(end), '--save-every', '1000', '--batch-size', '1', '--source-count', '4',
        '--max-source-time-pairs', '64', '--prefer-motion-sources', '--workers', '2',
        '--amp-dtype', 'bfloat16', '--cpu-threads', '4', '--seed', '49', '--no-query-chunking',
        '--decoder-memory', 'target_time']
    if job.get('allow_shared', True):
        command.append('--allow-shared')
    if reset_cursor:
        command.append('--reset-data-cursor')
    if time_origin:
        command += ['--time-origin', time_origin]
    command += ['--loss-config', str(loss_config)] if loss_config else ['--loss-profile', 'auto']
    return command


def choose_trial(reference, control, candidate):
    """A candidate must beat its paired control and pass the migration guard."""
    paired = compare_errors(control, candidate)
    migration = compare_errors(reference, candidate)
    selected = 'candidate' if paired['accepted'] and migration['allow_longer_sequences'] else 'control'
    return dict(selected=selected, versus_control=paired, versus_reference=migration)


class RecoveryController:
    def __init__(self, path):
        self.path = Path(path)
        self.job = json.loads(self.path.read_text())
        self.root = Path(self.job['repo'])
        self.control = Path(self.job['control_dir'])
        self.output = Path(self.job['output_root'])
        self.active = self.root/'outputs/training_preflight/active_training.json'
        self.progress = dict(stage='starting', phases=[], decisions=[])
        self.owns_training = False
        self.job['pid'] = os.getpid()

    def publish(self, stage):
        self.job['stage'] = self.progress['stage'] = stage
        write_json(self.path, self.job)
        write_json(self.control/'progress.json', self.progress)
        write_json(self.root/'outputs/training_preflight/queued_continuation.json', self.job)
        if self.owns_training:
            write_json(self.active, self.job)

    def run_process(self, name, command, *, evaluation=False):
        self.publish(name)
        environment = os.environ | self.job['environment']
        if evaluation:
            environment['CUDA_VISIBLE_DEVICES'] = environment['CUDA_VISIBLE_DEVICES'].split(',')[1]
        began = time.monotonic()
        log_path = self.control/(name+'.log')
        with log_path.open('w') as log:
            child = subprocess.Popen(command, cwd=self.root, env=environment, stdout=log, stderr=subprocess.STDOUT)
            self.progress['child_pid'] = child.pid
            self.progress['child_log'] = str(log_path)
            self.publish(name)
            code = child.wait()
        self.progress['phases'].append(dict(name=name, exit_code=code, seconds=time.monotonic()-began,
                                            log=str(log_path), argv=command))
        self.progress.pop('child_pid', None)
        self.publish(name)
        if code:
            raise RuntimeError(f'{name} failed with exit code {code}; see {log_path}')

    def evaluate(self, checkpoint, label):
        settings = self.job['evaluation']
        reports, paths = {}, {}
        for suite in ('kubric', 'native_dynamic', 'static', 'variable'):
            directory = self.output/'validation'/label/suite
            directory.mkdir(parents=True, exist_ok=True)
            common = ['--output', str(directory), '--checkpoints', label+'='+str(checkpoint)]
            if suite in ('kubric', 'variable'):
                module = 'multicamera4rc.evaluate'
                args = ['--data-config', settings[suite+'_config'], '--plan', settings[suite+'_plan'], '--cpu-threads', '2']
            elif suite == 'native_dynamic':
                module = 'scripts.evaluate_native_dynamic'
                args = ['--data-config', settings['native_config'], '--plan', settings['native_plan'], '--memory-fraction', '.20']
            else:
                module = 'scripts.evaluate_static_multiview'
                # The static evaluator materializes manifests beside its plan.
                # Each run gets a copy; the original fixed plan stays immutable.
                plan = directory/'fixed_plan.json'
                shutil.copyfile(settings['static_plan'], plan)
                args = ['--train-config', settings['static_config'], '--plan', str(plan), '--memory-fraction', '.20']
            wrapper = 'import runpy,sys,torch; torch.cuda.set_per_process_memory_fraction(.20,0); runpy.run_module(sys.argv.pop(1),run_name="__main__")'
            self.run_process('validate_'+label+'_'+suite, [self.job['python'], '-u', '-c', wrapper, module, *args, *common], evaluation=True)
            path = directory/'evaluation.json'
            report = json.loads(path.read_text())
            if report['stage'] != 'passed':
                raise RuntimeError('incomplete fixed validation: '+str(path))
            reports[suite] = report['checkpoints'][label]
            paths[suite] = str(path)
        panel = error_panel(reports)
        record = dict(checkpoint=str(checkpoint), label=label, errors=panel, reports=paths)
        write_json(self.output/'validation'/label/'quality.json', record)
        return record

    def remember_best(self, record):
        best = self.progress.get('best')
        if best is None or compare_errors(best['errors'], record['errors'])['accepted']:
            self.progress['best'] = record
            write_json(self.output/'best_checkpoint.json', record)
        write_json(self.output/'latest_evaluated_checkpoint.json', record)
        self.publish('quality_recorded')

    def train(self, resume, config, begin, end, label, **options):
        import torch
        payload = torch.load(resume, map_location='cpu', mmap=True, weights_only=True)
        saved_origin = payload['config'].get('time_origin', 'absolute')
        saved_loss = payload['training_state']['loss_config']
        del payload
        run = self.output/f'{label}_{begin}_{end}'
        self.job.update(run=str(run), selected_checkpoint=str(resume), resume_step=begin,
                        data_config=str(config), phase_target_step=end,
                        time_origin=options.get('time_origin') or saved_origin,
                        loss_config=(json.loads(Path(options['loss_config']).read_text())
                                     if options.get('loss_config') else saved_loss))
        command = training_argv(self.job, resume, config, run, end, **options)
        self.run_process('train_'+label+'_'+str(end), command)
        result = run/'latest.pt'
        state = validate_checkpoint(result, end)
        if state['lr_plan'] != self.progress['restored_training_state']['lr_plan']:
            raise ValueError('continuation changed the saved learning-rate plan')
        return result

    def cutover(self):
        previous = self.job['previous']
        checkpoint = Path(self.job['initial_checkpoint'])
        self.publish('waiting_atomic_checkpoint')
        deadline = time.monotonic()+12*3600
        while not checkpoint.exists():
            active = json.loads(self.active.read_text())
            identity = proc(previous['pid'])
            if active['pid'] != previous['pid'] or not identity or not alive(identity):
                raise RuntimeError('previous run exited or changed before the checkpoint')
            if identity['start'] != self.job['previous_start']:
                raise RuntimeError('previous supervisor PID was reused')
            if time.monotonic() > deadline:
                raise TimeoutError('atomic checkpoint did not arrive within 12 hours')
            time.sleep(10)
        if json.loads(self.active.read_text())['pid'] != previous['pid']:
            raise RuntimeError('another run became active; refusing overlap')
        for path, expected in self.job['evidence_sha256'].items():
            if digest(path) != expected:
                raise RuntimeError('verified evidence changed: '+path)
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.root, text=True).strip()
        if head != previous['commit'] or subprocess.check_output(['git', 'status', '--porcelain'], cwd=self.root, text=True).strip():
            raise RuntimeError('live checkout changed; previous training remains running')
        subprocess.run(['git', 'merge-base', '--is-ancestor', head, self.job['commit']], cwd=self.root, check=True)
        self.progress['restored_training_state'] = validate_checkpoint(checkpoint, self.job['initial_step'])
        loss = dict(self.progress['restored_training_state']['loss_config'], foreground_fraction=.5)
        if loss != json.loads(Path(self.job['foreground_loss_config']).read_text()):
            raise ValueError('saved loss differs from the prepared foreground-only trial')
        tree = capture_recorded_tree(previous, self.job['previous_start'])
        write_json(self.control/'previous_active_training.json', json.loads(self.active.read_text()))
        self.publish('stopping_recorded_supervisor')
        stop_tree(tree)
        transition = dict(stage='superseded_by_quality_recovery', checkpoint=str(checkpoint),
                          new_control_dir=str(self.control), stopped_processes=list(tree))
        write_json(Path(previous['control_dir'])/'transition.json', transition)
        # Only now replace source files imported by the previous workers.
        subprocess.run(['git', 'merge', '--ff-only', self.job['commit']], cwd=self.root, check=True)
        self.owns_training = True
        self.publish('quality_recovery_started')

    def run(self):
        try:
            self.cutover()
            checkpoint = Path(self.job['initial_checkpoint'])
            step = self.job['initial_step']
            stages = self.job['curriculum']['stages']
            config = stages[0]['data_config']
            reference = self.evaluate(checkpoint, 'baseline_'+str(step))
            self.remember_best(reference)
            # Stabilize the sampling change before architecture or loss trials.
            end = step+500
            checkpoint = self.train(checkpoint, config, step, end, 'sampling_recovery', reset_cursor=True)
            step = end
            current = self.evaluate(checkpoint, 'sampling_'+str(step))
            self.remember_best(current)
            for trial, options in (('relative_time', dict(time_origin='clip_start')),
                                   ('foreground_balance', dict(loss_config=self.job['foreground_loss_config']))):
                start_checkpoint, begin = checkpoint, step
                end = begin+self.job['trial_steps']
                # Both branches restore exactly the same data cursor, rank RNG,
                # optimizer and LR schedule. Only the tested option changes.
                control = self.train(start_checkpoint, config, begin, end, trial+'_control')
                control_result = self.evaluate(control, trial+'_control_'+str(end))
                candidate = None
                try:
                    candidate = self.train(start_checkpoint, config, begin, end, trial+'_candidate', **options)
                    candidate_result = self.evaluate(candidate, trial+'_candidate_'+str(end))
                    decision = choose_trial(reference['errors'], control_result['errors'], candidate_result['errors'])
                except Exception as error:
                    # The matched control has already completed both training
                    # and validation. A failed experiment must not strand it.
                    decision = dict(selected='control', candidate_error=str(error))
                decision.update(trial=trial, start_checkpoint=str(start_checkpoint),
                                control=str(control), candidate=str(candidate) if candidate else None, end_step=end)
                current = candidate_result if decision['selected'] == 'candidate' else control_result
                checkpoint = Path(current['checkpoint'])
                step = end
                self.progress['decisions'].append(decision)
                write_json(self.output/(trial+'_decision.json'), decision)
                # A rejected trial must never become the recommended model.
                self.remember_best(current)
            level = 0
            while step < self.job['target_step']:
                desired = max(i for i, stage in enumerate(stages) if step >= stage['start_step'])
                gate = compare_errors(reference['errors'], current['errors'])
                next_level = min(desired, level+1) if gate['allow_longer_sequences'] else level
                next_config = stages[next_level]['data_config']
                end = min(self.job['target_step'], ((step//2000)+1)*2000)
                decision = dict(step=step, desired_curriculum=desired, selected_curriculum=next_level, **gate)
                self.progress['decisions'].append(decision)
                checkpoint = self.train(checkpoint, next_config, step, end, 'continuation', reset_cursor=next_config != config)
                config, level, step = next_config, next_level, end
                current = self.evaluate(checkpoint, 'continuation_'+str(step))
                self.remember_best(current)
            self.job['final_checkpoint'] = str(checkpoint)
            self.job['best_checkpoint'] = self.progress['best']['checkpoint']
            self.publish('passed')
        except Exception as error:
            self.progress['error'] = str(error)
            self.publish('failed')
            raise


def prepare(args):
    repo, audit = args.repo.resolve(), args.audit.resolve()
    previous = json.loads((repo/'outputs/training_preflight/active_training.json').read_text())
    if previous['physical_gpus'] != [4, 5]:
        raise ValueError('expected GPU4/5 training')
    identity = proc(previous['pid'])
    if not identity:
        raise ValueError('previous supervisor is not running')
    capture_recorded_tree(previous, identity['start'])
    plan = json.loads((audit/'curriculum.json').read_text())
    start = plan['stages'][0]['start_step']
    if start != 32000 or plan['target_step'] != 50000:
        raise ValueError('this recovery protocol expects the audited 32k to 50k curriculum')
    proofs = [audit/name for name in ('unit_tests.json', 'ddp_preflight.json', 'data_proof.json')]
    for path in proofs:
        if not json.loads(path.read_text()).get('passed'):
            raise ValueError('required preflight failed: '+str(path))
    code_root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code_root, text=True).strip()
    if commit != args.commit or subprocess.check_output(['git', 'status', '--porcelain'], cwd=code_root, text=True).strip():
        raise ValueError('controller checkout must be clean at the verified commit')
    if shutil.disk_usage(args.output_root).free < 650*2**30:
        raise ValueError('recovery trials and checkpoints require 650 GiB free space')
    tag = 'quality_recovery_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SUTC')
    control = repo/'outputs/training_preflight/jobs'/tag
    output = args.output_root/tag
    control.mkdir(parents=True)
    output.mkdir(parents=True)
    foreground = code_root/'multicamera4rc/audits/quality_20261003/loss_foreground_only.json'
    # Verify the candidate changes only foreground/background normalization.
    from multicamera4rc.losses import MultiCamera4RCLoss
    candidate = json.loads(foreground.read_text())
    expected = MultiCamera4RCLoss.dense().to_config()
    expected['foreground_fraction'] = .5
    if candidate != expected:
        raise ValueError('foreground trial must change only foreground_fraction')
    evaluation = json.loads(args.evaluation.read_text())
    paths = proofs+[audit/'curriculum.json', foreground, args.evaluation]+list(map(Path, evaluation.values()))
    import yaml
    for stage in plan['stages']:
        path = Path(stage['data_config'])
        paths.append(path)
        data = yaml.safe_load(path.read_text())
        paths.extend(Path(source['manifest']) for source in data['sources'] if source.get('manifest'))
    environment = dict(previous['environment'], PYTHONPATH=str(repo))
    job = dict(repo=str(repo), control_dir=str(control), output_root=str(output), run=str(output),
               python=str(repo/'.venv/bin/python'), commit=commit, stage='prepared',
               log=str(control/'supervisor.log'), environment=environment, previous=previous,
               previous_start=identity['start'], initial_step=start, resume_step=start,
               initial_checkpoint=str(Path(previous['run'])/f'step_{start:08d}.pt'),
               target_step=50000, trial_steps=250, physical_gpus=[4, 5], allow_shared=True,
               global_batch_size=2, batch_size_per_gpu=1, source_count=4, max_source_time_pairs=64,
               no_query_chunking=True, image_mode='native', supervision='dense',
               decoder_memory='target_time', time_origin='absolute', prefer_motion_sources=True,
               curriculum=plan, evaluation=evaluation, foreground_loss_config=str(foreground),
               evidence_sha256={str(path): digest(path) for path in paths},
               created_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(control/'job.json', job)
    dry = training_argv(job, args.preflight_checkpoint, plan['stages'][0]['data_config'], output/'dry_run', start+500,
                        reset_cursor=True)+['--dry-run']
    result = subprocess.run(dry, cwd=code_root, env=os.environ|dict(environment, PYTHONPATH=str(code_root)), capture_output=True, text=True)
    (control/'launcher_dry_run.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError('launcher dry run failed before queueing')
    with (control/'supervisor.log').open('a') as log:
        process = subprocess.Popen([job['python'], '-u', str(Path(__file__).resolve()), '--controller', str(control/'job.json')],
            cwd=code_root, env=os.environ|dict(environment, PYTHONPATH=str(code_root)), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    # The child owns job.json after launch; the parent does not race its state.
    print(json.dumps(dict(queued=True, pid=process.pid, control_dir=str(control), output_root=str(output))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller', type=Path)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--commit')
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--preflight-checkpoint', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path('/data8/zbf/multicamera4rc_runs'))
    args = parser.parse_args()
    if args.controller:
        RecoveryController(args.controller).run()
    else:
        if not all((args.repo, args.audit, args.commit, args.evaluation, args.preflight_checkpoint)):
            parser.error('preparation requires repo, audit, commit, evaluation and preflight-checkpoint')
        prepare(args)


if __name__ == '__main__':
    main()
