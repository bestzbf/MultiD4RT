"""Migrate a recorded GPU4/5 run to filtered Kubric targets at an atomic checkpoint.

Retains the model, Adam moments, sampler cursor and rank RNG. Re-evaluates the
baseline with the new masks/scale before a bounded LR warmup and continuation.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

from scripts.continue_quality_recovery import (RecoveryController, capture_recorded_tree,
                                              digest, training_argv, validate_checkpoint)
from scripts.cutover_native_motion_stereo import alive, proc, stop_tree, write_json
from multicamera4rc.quality import compare_errors


class BackgroundRecoveryController(RecoveryController):
    def cutover(self):
        previous=self.job['previous'];checkpoint=Path(self.job['initial_checkpoint'])
        self.publish('waiting_atomic_checkpoint')
        deadline=time.monotonic()+12*3600
        while not checkpoint.exists():
            identity=proc(previous['pid'])
            if (json.loads(self.active.read_text())['pid']!=previous['pid'] or not identity
                    or not alive(identity) or identity['start']!=self.job['previous_start']):
                raise RuntimeError('recorded training changed before the atomic checkpoint')
            if time.monotonic()>deadline:raise TimeoutError('atomic checkpoint did not arrive')
            time.sleep(10)
        if json.loads(self.active.read_text())['pid']!=previous['pid']:
            raise RuntimeError('another training job became active')
        for path,expected in self.job['evidence_sha256'].items():
            if digest(path)!=expected:raise RuntimeError('verified evidence changed: '+path)
        for path in self.job['required_proofs']:
            if not json.loads(Path(path).read_text()).get('passed'):
                raise RuntimeError('required proof failed: '+path)
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root,text=True).strip()
        if head!=previous['commit'] or subprocess.check_output(['git','status','--porcelain'],cwd=self.root,text=True).strip():
            raise RuntimeError('live code changed; existing training was not stopped')
        subprocess.run(['git','merge-base','--is-ancestor',head,self.job['commit']],cwd=self.root,check=True)
        state=validate_checkpoint(checkpoint,self.job['initial_step'])
        self.progress['restored_training_state']=state
        tree=capture_recorded_tree(previous,self.job['previous_start'])
        write_json(self.control/'previous_active_training.json',json.loads(self.active.read_text()))
        self.publish('stopping_recorded_supervisor')
        stop_tree(tree)
        write_json(Path(previous['control_dir'])/'transition.json',dict(
            stage='superseded_by_background_recovery',checkpoint=str(checkpoint),
            new_control_dir=str(self.control),stopped_processes=list(tree)))
        subprocess.run(['git','merge','--ff-only',self.job['commit']],cwd=self.root,check=True)
        self.owns_training=True
        self.publish('background_recovery_started')

    def train_filtered(self,checkpoint,begin,end,first):
        run=self.output/f'filtered_{begin}_{end}'
        self.job.update(run=str(run),selected_checkpoint=str(checkpoint),resume_step=begin,
                        phase_target_step=end,kubric_background_policy='objects_ground',dense_target_version=2)
        argv=training_argv(self.job,checkpoint,self.job['data_config'],run,end)
        # The target scale changed. Resume Adam/RNG/data, but give the geometry
        # heads a short, explicit warmup instead of inheriting the near-final LR.
        if first:
            argv+=['--override-lr','--lr','5e-6','--backbone-lr','5e-7',
                   '--lr-schedule','cosine','--warmup-steps','100','--warmup-start-ratio','.4',
                   '--min-lr-ratio','.2','--lr-end-step',str(self.job['target_step'])]
        self.run_process('train_filtered_'+str(end),argv)
        checkpoint=run/'latest.pt';state=validate_checkpoint(checkpoint,end)
        if state.get('dense_target_version')!=2 or state.get('kubric_background_policy')!='objects_ground':
            raise RuntimeError('checkpoint does not declare the corrected target protocol')
        if state['loss_config']!=self.progress['restored_training_state']['loss_config']:
            raise RuntimeError('loss hyperparameters changed during the target repair')
        self.progress['resumed_training_state']=state
        return checkpoint

    def run(self):
        try:
            self.cutover()
            checkpoint=Path(self.job['initial_checkpoint']);step=self.job['initial_step']
            reference=self.evaluate(checkpoint,'filtered_baseline_'+str(step))
            self.remember_best(reference)
            ends=sorted({min(step+200,self.job['target_step']),self.job['target_step'],
                         *range((step//1000+1)*1000,self.job['target_step']+1,1000)})
            for index,end in enumerate(ends):
                if end<=step:continue
                checkpoint=self.train_filtered(checkpoint,step,end,first=index==0)
                step=end
                current=self.evaluate(checkpoint,'filtered_'+str(step))
                comparison=compare_errors(reference['errors'],current['errors'])
                self.progress['decisions'].append(dict(step=step,versus_filtered_baseline=comparison))
                self.remember_best(current)
                # Other datasets have unchanged GT. A large regression there
                # must stop the migration, rather than silently rolling back to
                # the unfiltered targets or continuing to overwrite checkpoints.
                unaffected={k:v for k,v in comparison['metric_ratios'].items()
                            if k.startswith(('native/','static/'))}
                if max(unaffected.values(),default=1)>1.25:
                    raise RuntimeError('unchanged-dataset error grew >25%; checkpoints preserved for review')
            self.job['selected_checkpoint']=str(checkpoint)
            self.publish('completed_50000')
        except Exception:
            self.progress['error']=traceback.format_exc();self.publish('failed');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller',required=True)
    BackgroundRecoveryController(parser.parse_args().controller).run()
