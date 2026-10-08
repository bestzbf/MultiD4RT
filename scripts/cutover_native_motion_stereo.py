"""Wait for an atomic checkpoint, verify giant native data, then resume GPU4/5.

Stops only the explicitly identified supervisor's same-UID process tree. If
the giant verification fails, resumes the previous data configuration with
its original optimizer, RNG, learning-rate plan and sampler cursor.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time


def write_json(path, value):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def proc(pid):
    path = Path("/proc") / str(pid)
    try:
        stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
        return dict(pid=pid, ppid=int(stat[1]), state=stat[0], start=stat[19],
                    uid=path.stat().st_uid,
                    cmd=(path / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace"))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def capture_tree(job):
    supervisor = proc(job["pid"])
    if not supervisor or supervisor["uid"] != os.getuid():
        raise RuntimeError("expected same-UID supervisor is not running")
    if str(Path(job["control_dir"]) / "supervise.py") not in supervisor["cmd"]:
        raise RuntimeError("supervisor PID no longer identifies the recorded job")
    candidates = [proc(int(p.name)) for p in Path("/proc").iterdir() if p.name.isdigit()]
    tree = {supervisor["pid"]: supervisor}
    while True:
        children = [p for p in candidates if p and p["ppid"] in tree and p["pid"] not in tree]
        if not children:
            break
        tree.update({p["pid"]: p for p in children})
    if not any("multicamera4rc.train" in p["cmd"] for p in tree.values()):
        raise RuntimeError("expected training ranks are absent")
    if any(p["uid"] != os.getuid() for p in tree.values()):
        raise RuntimeError("refusing to stop another user's process")
    return tree


def alive(record):
    current = proc(record["pid"])
    return current and current["start"] == record["start"] and current["state"] != "Z"


def stop_tree(tree):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for record in tree.values():
            if alive(record):
                try:
                    os.kill(record["pid"], sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + (25 if sig == signal.SIGTERM else 5)
        while any(alive(p) for p in tree.values()) and time.monotonic() < deadline:
            time.sleep(.5)
    if any(alive(p) for p in tree.values()):
        raise RuntimeError("recorded training tree did not exit")


SUPERVISOR = '''from pathlib import Path
import json,os,subprocess,sys,time,traceback
c=Path(__file__).parent;j=json.loads((c/'job.json').read_text());j['pid']=os.getpid()
env=os.environ.copy();env.update(j['environment']);active=Path(j['repo'])/'outputs/training_preflight/active_training.json'
report={'stage':'starting','phases':[]}
def write(path,value):
    temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\\n');temp.replace(path)
def publish(stage):
    report['stage']=stage;write(c/'progress.json',report)
    j['stage']=stage;write(c/'job.json',j);write(active,j)
try:
    for phase in j['phases']:
        publish(phase['name']);began=time.monotonic();phase_env=dict(env);phase_env.update(phase.get('environment',{}))
        print(json.dumps({'phase':phase['name']}),flush=True)
        process=subprocess.Popen(phase['argv'],cwd=j['repo'],env=phase_env)
        report['child_pid']=process.pid;write(c/'progress.json',report);code=process.wait()
        report['phases'].append({'name':phase['name'],'exit_code':code,'elapsed_seconds':time.monotonic()-began})
        if not code and phase.get('proof'):
            try:
                proof=json.loads(Path(phase['proof']).read_text())
                if not proof.get('passed') or len(proof.get('samples',[]))!=15:code=1
            except (OSError,ValueError):code=1
        if code:
            if phase['name']=='native_motion_stereo_preflight':
                fallback=json.loads((c/'rollback.json').read_text())
                publish('preflight_failed_resuming_previous_data')
                with (Path(fallback['control_dir'])/'supervisor.log').open('a') as log:
                    p=subprocess.Popen([sys.executable,'-u',str(Path(fallback['control_dir'])/'supervise.py')],cwd=j['repo'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
                fallback['pid']=p.pid;write(Path(fallback['control_dir'])/'job.json',fallback);write(active,fallback)
                report['rollback_pid']=p.pid;write(c/'progress.json',report)
            else:publish('failed')
            sys.exit(code)
        if phase.get('evaluation'):report.setdefault('evaluations',{})[phase['name']]=json.loads(Path(phase['evaluation']).read_text())
        write(c/'progress.json',report)
    publish('passed')
except Exception:
    report['error']=traceback.format_exc();publish('failed');raise
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--supervisor-pid", type=int, required=True)
    p.add_argument("--resume-step", type=int, default=17500)
    p.add_argument("--wait-hours", type=float, default=12.)
    args = p.parse_args(); root = args.repo.resolve(); audit = args.audit.resolve()
    active = root / "outputs/training_preflight/active_training.json"
    old = json.loads(active.read_text())
    if old["pid"] != args.supervisor_pid or old["physical_gpus"] != [4, 5]:
        raise RuntimeError("active job changed; no process will be stopped")
    capture_tree(old)
    for name in ("native_gt_audit.json", "pointodyssey_all_sequence_gt.json", "native_motion_data_plan.json", "unit_tests.json"):
        if not json.loads((audit / name).read_text())["passed"]:
            raise RuntimeError(f"required evidence failed: {name}")
    state_file = audit / "cutover_state.json"
    resume = Path(old["run"]) / f"step_{args.resume_step:08d}.pt"
    write_json(state_file, dict(stage="waiting_checkpoint", old_pid=old["pid"], checkpoint=str(resume), commit=args.commit))
    deadline = time.monotonic() + args.wait_hours*3600
    while not resume.exists():
        current = json.loads(active.read_text())
        if current.get("pid") != old["pid"] or not proc(old["pid"]):
            raise RuntimeError("active job changed while waiting; no process will be stopped")
        if time.monotonic() > deadline:
            raise TimeoutError("atomic checkpoint did not arrive")
        time.sleep(10)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != args.commit:
        raise RuntimeError("code commit changed while waiting")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise RuntimeError("working tree changed while waiting")
    tests = json.loads((audit / "unit_tests.json").read_text())
    for name, digest in tests["code_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"tested source changed: {name}")
    if shutil.disk_usage(root).free < 5*resume.stat().st_size + 8*1024**3:
        raise RuntimeError("insufficient disk for remaining atomic checkpoints; previous job remains running")
    import torch
    checkpoint = torch.load(resume, map_location="cpu", mmap=True, weights_only=False)
    state = checkpoint["training_state"]
    if checkpoint["step"] != args.resume_step or "optimizer" not in checkpoint:
        raise RuntimeError("checkpoint cannot restore optimizer")
    if state["lr_plan"] != old["lr_plan"] or state["world_size"] != 2 or state["batch_size"] != 1:
        raise RuntimeError("checkpoint training state changed")
    saved_state = dict(state); del checkpoint
    plan = json.loads((audit / "native_motion_data_plan.json").read_text())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")
    name = "full_dense_native_motion_stereo_gpu45_" + stamp
    output = root / "outputs" / name; control = root / "outputs/training_preflight/jobs" / name
    control.mkdir(parents=True, exist_ok=False)
    new = dict(old); new.pop("midphase_validation", None)
    new.update(stage="native_motion_stereo_preflight", run=str(output), control_dir=str(control),
        log=str(control / "supervisor.log"), commit=args.commit, selected_checkpoint=str(resume),
        resume_step=args.resume_step, remaining_training_steps_at_launch=old["target_step"]-args.resume_step,
        data_config=plan["data_config"], coverage_windows=plan["coverage_windows"], epoch_size=plan["epoch_size"],
        dataset_sources=plan["sources"], sampler_cursor_preserved=False, restored_training_state=saved_state,
        previous_control_dir=old["control_dir"], started_at_utc=datetime.now(timezone.utc).isoformat(),
        notes="Dense native geometry plus all native PointOdyssey particles; native stereo C2T6. Restore AdamW/RNG/LR, reset changed coverage cursor. Giant real-data backward verification precedes training.")
    new["environment"] = dict(old["environment"], PYTORCH_ALLOC_CONF="expandable_segments:True")
    train = list(old["argv"])
    for flag, value in (("--resume", str(resume)), ("--data-config", plan["data_config"]), ("--output", str(output))):
        train[train.index(flag)+1] = value
    if "--reset-data-cursor" not in train: train.append("--reset-data-cursor")
    new["argv"] = train
    proof = audit / "giant_preflight.json"
    preflight = [str(root / ".venv/bin/python"), "-u", str(root / "scripts/preflight_native_motion_stereo.py"),
        "--data-config", plan["data_config"], "--checkpoint", str(resume), "--output", str(proof), "--memory-fraction", ".75"]
    new["phases"] = [dict(name="native_motion_stereo_preflight", argv=preflight, proof=str(proof),
        environment={"CUDA_VISIBLE_DEVICES": old["environment"]["CUDA_VISIBLE_DEVICES"].split(",")[0]}),
        dict(name="training", argv=train)]
    for phase in old["phases"][1:]:
        changed = dict(phase)
        changed["argv"] = [a.replace(old["run"], str(output)) for a in phase["argv"]]
        changed["argv"] = ["pre_native17500="+str(resume) if a.startswith("pre_static15000=")
                            else a.replace("static25000=", "native25000=") for a in changed["argv"]]
        changed["evaluation"] = phase["evaluation"].replace(old["run"], str(output))
        new["phases"].append(changed)
    new["gt_preflights"] = old["gt_preflights"] + [str(audit / n) for n in
        ("native_gt_audit.json", "pointodyssey_all_sequence_gt.json", "native_motion_data_plan.json", "unit_tests.json", "giant_preflight.json")]
    rollback_control = Path(str(control) + "_rollback"); rollback_control.mkdir(exist_ok=False)
    rollback = dict(old); rollback.pop("midphase_validation", None)
    rollback.update(run=str(output)+"_rollback", control_dir=str(rollback_control),
                    log=str(rollback_control / "supervisor.log"), selected_checkpoint=str(resume),
                    resume_step=args.resume_step, restored_training_state=saved_state, commit=args.commit)
    rollback["argv"] = list(old["argv"])
    for flag, value in (("--resume", str(resume)), ("--output", rollback["run"])):
        rollback["argv"][rollback["argv"].index(flag)+1] = value
    rollback["phases"] = [dict(name="training", argv=rollback["argv"])]
    compile(SUPERVISOR, "supervise.py", "exec")
    for c, j in ((control, new), (rollback_control, rollback)):
        (c / "supervise.py").write_text(SUPERVISOR); write_json(c / "job.json", j)
    write_json(control / "rollback.json", rollback); write_json(control / "previous_active_training.json", old)
    env = dict(os.environ); env.update(new["environment"])
    dry_run = subprocess.run(train + ["--dry-run"], cwd=root, env=env, capture_output=True, text=True)
    (control / "launcher_dry_run.log").write_text(dry_run.stdout + dry_run.stderr)
    if dry_run.returncode: raise RuntimeError("new launcher preflight failed; previous job remains running")
    if json.loads(active.read_text()).get("pid") != old["pid"]:
        raise RuntimeError("active job changed during preparation; no process will be stopped")
    tree = capture_tree(old)
    write_json(state_file, dict(stage="stopping_recorded_job", checkpoint=str(resume), control_dir=str(control)))
    stop_tree(tree)
    transition = dict(stage="stopped_for_native_motion_stereo", checkpoint=str(resume),
                      new_control_dir=str(control), stopped_processes=list(tree))
    write_json(Path(old["control_dir"]) / "transition.json", transition)
    old_progress = json.loads((Path(old["control_dir"]) / "progress.json").read_text())
    write_json(Path(old["control_dir"]) / "progress.json", dict(old_progress, **transition))
    with (control / "supervisor.log").open("a") as log:
        process = subprocess.Popen([str(root / ".venv/bin/python"), "-u", str(control / "supervise.py")],
            cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, stdin=subprocess.DEVNULL)
    new["pid"] = process.pid
    write_json(control / "job.json", new); write_json(active, new); write_json(audit / "training_job.json", new)
    write_json(state_file, dict(stage="preflight_started", pid=process.pid, checkpoint=str(resume), control_dir=str(control)))
    print(json.dumps(dict(started=True, pid=process.pid, control_dir=str(control))), flush=True)


if __name__ == "__main__":
    main()
