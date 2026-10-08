"""Queue verified variable-C/T stages after the recorded 25k run and validation.

This controller never signals or terminates a process. It waits for successful
completion of the recorded supervisor, then restores its atomic checkpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

if __package__:
    from .cutover_native_motion_stereo import proc, write_json
else:  # Direct script invocation with the project installed or PYTHONPATH=.
    from cutover_native_motion_stereo import proc, write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_argv(python, repo, resume, stage, output, first):
    argv=[python,"-u","-m","multicamera4rc.launch","--gpus","4,5","--allow-shared",
        "--resume",str(resume),"--data-config",stage["data_config"],"--output",str(output),
        "--preset","giant","--steps",str(stage["end_step"]),"--save-every","2500",
        "--batch-size","1","--source-count","4","--max-source-time-pairs","64","--workers","2","--amp-dtype","bfloat16",
        "--cpu-threads","4","--seed","49","--no-query-chunking","--loss-profile","auto"]
    if first:
        argv += ["--reset-data-cursor","--override-lr","--lr","1e-5","--backbone-lr","1e-6","--lr-schedule","cosine",
                 "--warmup-steps","500","--warmup-start-ratio",".3","--min-lr-ratio",".1","--lr-end-step","50000"]
    return argv


def run_controller(job_file):
    job_file=Path(job_file); job=json.loads(job_file.read_text()); root=Path(job["repo"])
    control=Path(job["control_dir"]); active=root/"outputs/training_preflight/active_training.json"
    queued=root/"outputs/training_preflight/queued_continuation.json"
    job["pid"]=os.getpid();progress=dict(stage="waiting_previous_run",phases=[],target_step=50000)
    def publish(stage, activate=False):
        progress["stage"]=stage;job["stage"]=stage
        write_json(control/"progress.json",progress);write_json(job_file,job);write_json(queued,job)
        if activate:write_json(active,job)
    try:
        publish("waiting_previous_run")
        deadline=time.monotonic()+24*3600
        previous=job["previous_job"]
        while True:
            old=json.loads((Path(previous["control_dir"])/"job.json").read_text())
            current=proc(previous["pid"])
            if job.get("recovery_of"):
                if old["stage"] != "failed" or (current and current["state"] != "Z"):
                    raise RuntimeError("recovery requires a failed, stopped supervisor")
                child=proc(json.loads((Path(previous["control_dir"])/"progress.json").read_text()).get("child_pid",0))
                if child and child["state"] != "Z":
                    raise RuntimeError("previous training child is still running")
                break
            if current and current["state"] != "Z" and (current["uid"] != os.getuid() or current["start"] != job["previous_start"]):
                raise RuntimeError("previous supervisor identity changed")
            if old["stage"] == "passed":break
            if old["stage"] == "failed" or not current or current["state"] == "Z":
                raise RuntimeError("previous run exited before successful training/validation")
            if time.monotonic() > deadline:raise TimeoutError("previous run did not finish within 24h")
            time.sleep(15)
        if json.loads(active.read_text())["pid"] != previous["pid"]:
            raise RuntimeError("another run became active; refusing to overlap its training")
        if subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip() != job["commit"]:
            raise RuntimeError("training code changed after verified queue preparation")
        for path,sha in job["evidence_sha256"].items():
            if digest(path) != sha:raise RuntimeError(f"verified config/evidence changed: {path}")
        import torch
        checkpoint=torch.load(job["selected_checkpoint"],map_location="cpu",weights_only=True,mmap=True)
        state=checkpoint["training_state"]
        expected_step=job.get("resume_step",25000)
        if checkpoint["step"] != expected_step or not checkpoint["optimizer"]["state"]:
            raise RuntimeError(f"expected {expected_step} checkpoint with AdamW moments")
        if state["world_size"] != 2 or state["batch_size"] != 1 or state["optimizer_updates"] != expected_step:
            raise RuntimeError("checkpoint training shape/update count changed")
        progress["restored_training_state"]=state;del checkpoint
        environment=os.environ.copy();environment.update(job["environment"])
        for phase in job["phases"]:
            job["run"]=phase.get("run",job["run"])
            job["data_config"]=phase.get("data_config",job["data_config"])
            job["phase_target_step"]=phase.get("target_step",job.get("phase_target_step"))
            publish(phase["name"],True)
            print(json.dumps(dict(phase=phase["name"],argv=phase["argv"])),flush=True)
            began=time.monotonic();env=dict(environment);env.update(phase.get("environment",{}))
            process=subprocess.Popen(phase["argv"],cwd=root,env=env)
            progress["child_pid"]=process.pid;write_json(control/"progress.json",progress)
            code=process.wait();progress["phases"].append(dict(name=phase["name"],exit_code=code,elapsed_seconds=time.monotonic()-began))
            if code:raise RuntimeError(f"phase {phase['name']} failed with exit code {code}")
            if "evaluation" in phase:
                evaluation=json.loads(Path(phase["evaluation"]).read_text())
                progress.setdefault("evaluations",{})[phase["name"]]=evaluation
            write_json(control/"progress.json",progress)
        publish("passed",True)
    except Exception as error:
        progress["error"]=str(error);publish("failed",job.get("stage") != "waiting_previous_run")
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path)
    p.add_argument("--audit",type=Path)
    p.add_argument("--output-root",type=Path,default=Path("/data8/zbf/multicamera4rc_runs"))
    p.add_argument("--commit")
    p.add_argument("--controller",type=Path)
    args=p.parse_args()
    if args.controller:return run_controller(args.controller)
    if not args.repo or not args.audit or not args.commit:p.error("queue preparation requires repo, audit and commit")
    root=args.repo.resolve();audit=args.audit.resolve();active=root/"outputs/training_preflight/active_training.json"
    old=json.loads(active.read_text());identity=proc(old["pid"])
    if old["target_step"] != 25000 or old["physical_gpus"] != [4,5] or not identity or identity["uid"] != os.getuid():
        raise RuntimeError("expected recorded GPU4/5 25k supervisor")
    if str(Path(old["control_dir"])/"supervise.py") not in identity["cmd"]:
        raise RuntimeError("recorded PID does not identify expected supervisor")
    proof_paths=[audit/name for name in ("native_audit.json","point_long_gt.json","giant_audit.json","unit_tests.json","ddp_preflight.json")]
    proof_paths.append(audit/"variable_validation/proof.json")
    for path in proof_paths:
        if not json.loads(path.read_text()).get("passed"):raise RuntimeError(f"verification failed: {path}")
    plan=json.loads((audit/"curriculum.json").read_text())
    if plan["target_step"] != 50000 or [(s["start_step"],s["end_step"]) for s in plan["stages"]] != [(25000,32000),(32000,40000),(40000,50000)]:
        raise RuntimeError("curriculum stage targets changed")
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    if commit != args.commit or subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True).strip():
        raise RuntimeError("verified branch must be clean at requested commit")
    args.output_root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.output_root).free < 300*2**30:raise RuntimeError("50k continuation requires 300 GiB checkpoint space")
    existing=root/"outputs/training_preflight/queued_continuation.json"
    if existing.exists():
        q=json.loads(existing.read_text());running=proc(q.get("pid",0))
        if running and running["uid"] == os.getuid() and "queue_variable_continuation.py" in running["cmd"]:
            raise RuntimeError("a continuation controller is already queued")
    tag="variable_ct_gpu45_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")
    control=root/"outputs/training_preflight/jobs"/tag;control.mkdir()
    output=args.output_root/tag;output.mkdir();python=str(root/".venv/bin/python")
    resume=Path(old["run"])/"step_00025000.pt";baseline=resume;phases=[]
    env=dict(old["environment"]);env["PYTHONPATH"]=str(root)
    for i,stage in enumerate(plan["stages"]):
        run=output/f"stage_{i+1}_{stage['start_step']}_{stage['end_step']}"
        argv=training_argv(python,root,resume,stage,run,first=i==0)
        # Future checkpoints do not yet exist. Validate each stage's exact
        # CLI/resource setup against a verified, existing resume checkpoint.
        alternate=list(argv);alternate[alternate.index("--resume")+1]=str(Path(old["run"])/"step_00022500.pt")
        dry=subprocess.run(alternate+["--dry-run"],cwd=root,env=os.environ|env,capture_output=True,text=True)
        (control/f"launch_dry_run_stage_{i+1}.json").write_text(dry.stdout)
        if dry.returncode:raise RuntimeError("continuation CLI preflight failed: "+dry.stderr[-1500:])
        phases.append(dict(name=f"training_stage_{i+1}",argv=argv,run=str(run),data_config=stage["data_config"],target_step=stage["end_step"]))
        resume=run/"latest.pt"
        validation=output/f"fixed_validation_{stage['end_step']}"
        # Reuse the immutable C2/C3/C4 plan to detect multiview forgetting.
        evaluation_argv=[python,"-u","-c","import sys,torch; torch.cuda.set_per_process_memory_fraction(.35,0); from multicamera4rc.evaluate import main; main(sys.argv[1:])",
            "--data-config",str(root/"outputs/training_preflight/kubric_gpu45_dense_native.yaml"),
            "--plan",str(root/"outputs/training_preflight/fixed_validation_plan.json"),
            "--output",str(validation),"--checkpoints",f"native25000={baseline}",f"variable{stage['end_step']}={resume}"]
        phases.append(dict(name=f"fixed_validation_{stage['end_step']}",argv=evaluation_argv,evaluation=str(validation/"evaluation.json"),
            environment={"CUDA_VISIBLE_DEVICES":env["CUDA_VISIBLE_DEVICES"].split(",")[1]}))
        if i == 2:
            extra=output/"variable_validation_50000"
            new=list(evaluation_argv)
            for flag,value in [("--data-config",str(audit/"variable_validation/data.yaml")),
                               ("--plan",str(audit/"variable_validation/fixed_plan.json")),("--output",str(extra))]:
                new[new.index(flag)+1]=value
            phases.append(dict(name="variable_layout_validation_50000",argv=new,
                evaluation=str(extra/"evaluation.json"),environment={"CUDA_VISIBLE_DEVICES":env["CUDA_VISIBLE_DEVICES"].split(",")[1]}))
    evidence=proof_paths+[audit/"curriculum.json",audit/"variable_validation/data.yaml",audit/"variable_validation/fixed_plan.json"]+[Path(s["data_config"]) for s in plan["stages"]]
    job=dict(stage="queued",repo=str(root),run=str(output),control_dir=str(control),log=str(control/"supervisor.log"),
        commit=commit,environment=env,phases=phases,previous_job=old,previous_start=identity["start"],
        selected_checkpoint=str(baseline),resume_step=25000,target_step=50000,physical_gpus=[4,5],allow_shared=True,
        global_batch_size=2,batch_size_per_gpu=1,source_count=4,max_source_time_pairs=64,workers_per_gpu=2,no_query_chunking=True,
        image_mode="native",supervision="dense",data_config=plan["stages"][0]["data_config"],
        evidence_sha256={str(path):digest(path) for path in evidence},curriculum=plan,
        created_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(control/"job.json",job)
    with (control/"supervisor.log").open("a") as log:
        child=subprocess.Popen([python,"-u",str(root/"scripts/queue_variable_continuation.py"),"--controller",str(control/"job.json")],
            cwd=root,env=os.environ|env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
    job["pid"]=child.pid;write_json(control/"job.json",job);write_json(existing,job)
    print(json.dumps(dict(queued=True,pid=child.pid,control_dir=str(control),output=str(output),target_step=50000)),flush=True)


if __name__ == "__main__":main()
