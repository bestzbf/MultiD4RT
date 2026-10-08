"""Resume a failed variable-C/T training phase from a verified checkpoint.

Preserves the immutable data plans, optimizer/LR/RNG/cursor and remaining
validation phases. Never stops a process or automatically retries bad data.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch

from multicamera4rc.checkpoint import config_from_checkpoint

if __package__:
    from .cutover_native_motion_stereo import proc, write_json
    from .queue_variable_continuation import digest
else:
    from cutover_native_motion_stereo import proc, write_json
    from queue_variable_continuation import digest


def set_argument(argv, flag, value):
    if flag in argv:
        argv[argv.index(flag)+1] = str(value)
    else:
        argv.extend((flag, str(value)))


def recovery_phases(phases, failed_index, checkpoint, decoder_memory, save_every):
    """An interrupted phase restores its checkpoint, not its initial LR reset."""
    remaining = deepcopy(phases[failed_index:])
    if not remaining or not remaining[0]["name"].startswith("training_stage_"):
        raise ValueError("recovery requires a failed training phase")
    argv = remaining[0]["argv"]
    set_argument(argv, "--resume", checkpoint)
    for flag in ("--reset-data-cursor", "--override-lr"):
        if flag in argv:
            argv.remove(flag)
    for flag in ("--lr", "--backbone-lr", "--lr-schedule", "--warmup-steps",
                 "--warmup-start-ratio", "--min-lr-ratio", "--lr-end-step"):
        if flag in argv:
            index = argv.index(flag)
            del argv[index:index+2]
    for phase in remaining:
        set_argument(phase["argv"], "--decoder-memory", decoder_memory)
        if phase["name"].startswith("training_stage_"):
            set_argument(phase["argv"], "--save-every", save_every)
    return remaining


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--failed-job", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--proof", required=True, type=Path,
                        help="JSON with passed=true from replaying the failed sample and regression tests")
    parser.add_argument("--save-every", type=int, default=1000)
    args = parser.parse_args(argv)
    root = args.repo.resolve()
    old = json.loads(args.failed_job.read_text())
    progress = json.loads((Path(old["control_dir"])/"progress.json").read_text())
    active = json.loads((root/"outputs/training_preflight/active_training.json").read_text())
    if Path(old["repo"]).resolve() != root or old["stage"] != "failed":
        raise RuntimeError("expected this repository's failed job")
    if active["pid"] != old["pid"] or active["control_dir"] != old["control_dir"]:
        raise RuntimeError("another job has become active")
    for pid in (old["pid"], progress.get("child_pid", 0)):
        identity = proc(pid)
        if identity and identity["state"] != "Z":
            raise RuntimeError(f"recorded process {pid} is still running")
    if old["physical_gpus"] != [4, 5] or old["target_step"] != 50000:
        raise RuntimeError("expected GPU4/5 variable-C/T 50k job")
    failed_index = len(progress["phases"])-1
    if failed_index < 0 or not progress["phases"][failed_index]["exit_code"]:
        raise RuntimeError("failed training phase was not recorded")
    if progress["phases"][failed_index]["name"] != old["phases"][failed_index]["name"]:
        raise RuntimeError("phase records disagree")
    if any(phase["exit_code"] for phase in progress["phases"][:failed_index]):
        raise RuntimeError("an earlier phase also failed")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if commit != args.commit or dirty:
        raise RuntimeError("recovery requires clean code at the verified commit")
    for path, sha in old["evidence_sha256"].items():
        if digest(path) != sha:
            raise RuntimeError(f"original data plan/evidence changed: {path}")
    proof = json.loads(args.proof.read_text())
    if not proof.get("passed") or proof.get("commit") != commit:
        raise RuntimeError("sample replay and regression proof must pass at this commit")
    if args.save_every <= 0:
        raise ValueError("save-every must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    step = checkpoint["step"]
    state = checkpoint["training_state"]
    failed_phase = old["phases"][failed_index]
    previous_end = max([old["resume_step"]] + [p["target_step"] for p in old["phases"][:failed_index] if "target_step" in p])
    if not previous_end <= step < failed_phase["target_step"]:
        raise RuntimeError("checkpoint must belong to the interrupted phase")
    if args.checkpoint.resolve().parent != Path(failed_phase["run"]).resolve():
        raise RuntimeError("checkpoint is outside the interrupted phase output directory")
    if (state["world_size"], state["batch_size"], state["optimizer_updates"]) != (2, 1, step):
        raise RuntimeError("checkpoint world size, batch size or optimizer updates disagree")
    if not checkpoint["optimizer"]["state"] or len(checkpoint.get("rank_rng", [])) != 2:
        raise RuntimeError("checkpoint must contain optimizer moments and both ranks' RNG states")
    if state["lr_plan"]["end_step"] != old["target_step"]:
        raise RuntimeError("checkpoint LR schedule does not cover the planned target")
    decoder_memory = config_from_checkpoint(checkpoint).decoder_memory
    del checkpoint
    # Budget remaining checkpoint writes before increasing save frequency.
    # Recovery does not prune checkpoints or delete the previous failure log.
    remaining_saves = (old["target_step"]-step)//args.save_every + 6
    if shutil.disk_usage(failed_phase["run"]).free < remaining_saves*args.checkpoint.stat().st_size:
        raise RuntimeError("insufficient space for the remaining checkpoints")
    phases = recovery_phases(old["phases"], failed_index, args.checkpoint.resolve(), decoder_memory, args.save_every)
    env = os.environ | old["environment"]
    dry = subprocess.run(phases[0]["argv"]+["--dry-run"], cwd=root, env=env, text=True, capture_output=True)
    if dry.returncode:
        raise RuntimeError("recovery launch preflight failed: "+dry.stderr[-2000:])
    tag = "variable_ct_recovery_"+datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")
    control = root/"outputs/training_preflight/jobs"/tag
    control.mkdir()
    write_json(control/"failed_job.json", old)
    (control/"launch_dry_run.json").write_text(dry.stdout)
    job = {key: value for key, value in old.items()
           if key not in ("pid", "previous_job", "previous_start", "phases", "recovery_of")}
    job.update(stage="queued", commit=commit, control_dir=str(control), log=str(control/"supervisor.log"),
               run=phases[0]["run"], data_config=phases[0]["data_config"], phases=phases,
               previous_job=old, recovery_of=str(args.failed_job.resolve()),
               selected_checkpoint=str(args.checkpoint.resolve()), resume_step=step,
               decoder_memory=decoder_memory, save_every=args.save_every,
               created_at_utc=datetime.now(timezone.utc).isoformat())
    job["evidence_sha256"] = old["evidence_sha256"] | {str(args.proof.resolve()): digest(args.proof)}
    write_json(control/"job.json", job)
    python = str(root/".venv/bin/python")
    with (control/"supervisor.log").open("a") as log:
        process = subprocess.Popen([python, "-u", str(root/"scripts/queue_variable_continuation.py"),
                                    "--controller", str(control/"job.json")],
                                   cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, stdin=subprocess.DEVNULL)
    # The controller publishes its own PID and status. Do not race its first
    # phase update by writing a stale queued record from this parent process.
    print(json.dumps(dict(pid=process.pid, control_dir=str(control), resume_step=step,
                          target_step=job["target_step"], decoder_memory=decoder_memory)), flush=True)


if __name__ == "__main__":
    main()
