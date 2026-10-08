"""Preflight and start one-host DDP training on explicit physical NVIDIA GPUs.

Run ``python -m multicamera4rc.launch --help``. This module never selects a GPU
through a CUDA ordinal: nvidia-smi indices are resolved to GPU UUIDs first.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    name: str
    memory_total_mib: int
    memory_used_mib: int


@dataclass(frozen=True)
class GPUProcess:
    gpu_uuid: str
    pid: int
    name: str
    memory_mib: str


def _gpu_indices(value):
    try:
        values = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("gpus must be comma-separated physical indices, e.g. 4,5") from error
    if not values or any(value < 0 for value in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("GPU indices must be nonnegative and unique")
    return values


def gpu_snapshot():
    """Query driver metadata only; do not initialize CUDA or create a context."""
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    devices = {}
    for row in csv.reader(gpu.stdout.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 5:
            raise RuntimeError("Unexpected nvidia-smi GPU inventory; cannot resolve physical indices safely")
        index, uuid, name, total, used = (part.strip() for part in row)
        item = GPU(int(index), uuid, name, int(total), int(used))
        if not item.uuid.startswith("GPU-") or item.index in devices:
            raise RuntimeError("Unexpected or duplicate GPU identity in nvidia-smi")
        devices[item.index] = item
    processes = []
    for row in csv.reader(apps.stdout.splitlines(), skipinitialspace=True):
        if not row or (len(row) == 1 and "No running processes" in row[0]):
            continue
        if len(row) != 4:
            raise RuntimeError("Unexpected nvidia-smi compute-process output; cannot check GPU occupancy safely")
        uuid, pid, name, memory = (part.strip() for part in row)
        processes.append(GPUProcess(uuid, int(pid), name, memory))
    return devices, processes


def _selected_snapshot(indices):
    devices, processes = gpu_snapshot()
    missing = set(indices) - set(devices)
    if missing:
        raise RuntimeError(f"Requested physical GPUs do not exist: {sorted(missing)}")
    selected = [devices[index] for index in indices]
    selected_uuids = {gpu.uuid for gpu in selected}
    active = [process for process in processes if process.gpu_uuid in selected_uuids]
    return selected, active


def _data_summary(path):
    from .datasets.mixture import read_data_config

    config = read_data_config(path)
    sources = []
    for source in config["sources"]:
        root = Path(source["root"])
        if not root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        options = source.get("options", {})
        if options.get("require_validation_stamp"):
            stamps = Path(options.get("validation_stamp_dir", ""))
            if not options.get("validation_stamp_dir") or not stamps.is_dir():
                raise FileNotFoundError(f"Required validation-stamp directory does not exist: {stamps}")
        sources.append({key: source.get(key) for key in ("adapter", "root", "split", "cameras", "num_times", "weight", "strides", "view_mode", "layouts")})
    return {key: config.get(key) for key in ("image_size", "image_mode", "supervision", "sampling", "num_times", "num_queries", "epoch_size")} | {"sources": sources}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python executable of the training environment")
    parser.add_argument("--gpus", type=_gpu_indices, default=(4, 5), help="nvidia-smi physical GPU indices; default 4,5")
    parser.add_argument("--data-config", default=str(Path(__file__).with_name("configs") / "kubric_multicam_native.yaml"))
    parser.add_argument("--output", help="Fresh run directory; default outputs/multicamera4rc_gpu4_5_TIMESTAMP")
    weights = parser.add_mutually_exclusive_group(required=True)
    weights.add_argument("--pretrained", help="Original 4RC .pth/.safetensors initialization")
    weights.add_argument("--resume", help="Native model+optimizer checkpoint; steps is the total target")
    parser.add_argument("--reset-data-cursor", action="store_true",
                        help="With --resume, restart the sampler for a changed data configuration")
    parser.add_argument("--preset", choices=("giant", "tiny"), default="giant")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU; default global batch size is 2")
    parser.add_argument("--source-count", type=int, default=2, help="Decoded sources per clip; 0 decodes all")
    parser.add_argument("--max-source-time-pairs", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers per GPU")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--override-lr", action="store_true")
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--warmup-start-ratio", type=float, default=None)
    parser.add_argument("--lr-end-step", type=int, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=None)
    loss_settings = parser.add_mutually_exclusive_group()
    loss_settings.add_argument("--loss-profile", choices=("auto", "dense", "dense_balanced"), default="auto")
    loss_settings.add_argument("--loss-config", help="JSON/YAML loss settings; also stored in checkpoints")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--cpu-threads", type=int, default=4)
    attention = parser.add_mutually_exclusive_group()
    attention.add_argument("--attention-chunk-size", type=int,
                        help="Attention query chunk size; larger chunks trade memory for fewer launches")
    attention.add_argument("--no-query-chunking", action="store_true",
                           help="Disable attention and visibility query splitting, including resume")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--freeze-pretrained-backbone", action="store_true")
    parser.add_argument("--freeze-geometry", action="store_true")
    parser.add_argument("--prefer-motion-sources", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--decoder-memory", choices=("all_observations", "target_time"),
                        help="Override decoder memory scope; resume defaults to saved semantics")
    parser.add_argument("--time-origin", choices=("clip_start", "absolute"),
                        help="Override time encoding origin; resume defaults to saved semantics")
    parser.add_argument("--reset-time-projections", action="store_true")
    parser.add_argument("--partial-pretrained", action="store_true")
    resource = parser.add_mutually_exclusive_group()
    resource.add_argument("--allow-shared", action="store_true", help="Explicitly permit existing compute processes on selected GPUs")
    resource.add_argument("--wait-free", type=float, default=0, metavar="SECONDS", help="Explicit foreground wait for selected GPUs to become free; timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Print preflight and exact command; never start torchrun, even if GPUs are busy")
    return parser


def _command(args, python, data_config, weights, output):
    # Use explicit loopback addresses for this one-host launcher. B300's
    # hostname does not resolve to a local interface; --standalone can stall
    # before workers start when it infers the rendezvous node address.
    command = [python, "-m", "torch.distributed.run", "--nnodes=1",
               "--rdzv-backend=c10d", "--rdzv-endpoint=127.0.0.1:0",
               "--local-addr=127.0.0.1", "--rdzv-id=multicamera4rc",
               f"--nproc_per_node={len(args.gpus)}", "--log-dir", str(output / "torchrun"),
               "--tee", "3", "--module", "multicamera4rc.train",
               "--data-config", str(data_config), "--output", str(output),
               "--resume" if args.resume else "--pretrained", str(weights),
               "--preset", args.preset, "--device", "cuda", "--amp", "--amp-dtype", args.amp_dtype]
    for name in ("steps", "batch_size", "source_count", "max_source_time_pairs", "workers", "save_every", "lr",
                 "backbone_lr", "weight_decay", "clip_grad", "seed", "cpu_threads"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    for name in ("freeze_pretrained_backbone", "freeze_geometry", "no_checkpointing", "partial_pretrained", "reset_data_cursor", "override_lr", "prefer_motion_sources", "reset_time_projections"):
        if getattr(args, name):
            command.append("--" + name.replace("_", "-"))
    if args.prefer_motion_sources is False:
        command.append("--no-prefer-motion-sources")
    for name in ("loss_profile", "loss_config", "lr_schedule", "warmup_steps", "min_lr_ratio", "warmup_start_ratio", "lr_end_step", "decoder_memory", "time_origin"):
        value = getattr(args, name)
        if name == "loss_profile" and args.loss_config:
            continue
        if value is not None:
            command.extend(["--"+name.replace("_", "-"), str(value)])
    if args.attention_chunk_size is not None:
        command.extend(["--attention-chunk-size", str(args.attention_chunk_size)])
    if args.no_query_chunking:
        command.append("--no-query-chunking")
    return command


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if min(args.steps, args.batch_size, args.save_every, args.cpu_threads) <= 0 or args.source_count < 0 or args.workers < 0 or args.max_source_time_pairs < 0:
        parser.error("steps, batch size, save interval and threads must be positive; source count/workers nonnegative")
    if not all(math.isfinite(value) and value > 0 for value in (args.lr, args.backbone_lr, args.clip_grad)):
        parser.error("learning rates and clip-grad must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0 or not math.isfinite(args.wait_free) or args.wait_free < 0:
        parser.error("weight-decay and wait-free must be finite and nonnegative")
    if args.attention_chunk_size is not None and args.attention_chunk_size <= 0:
        parser.error("attention-chunk-size must be positive")
    if args.reset_data_cursor and not args.resume:
        parser.error("reset-data-cursor requires --resume")
    if args.override_lr and not args.resume:
        parser.error("override-lr requires --resume")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        parser.error("warmup-steps must be nonnegative")
    if args.min_lr_ratio is not None and (not math.isfinite(args.min_lr_ratio) or not 0 < args.min_lr_ratio <= 1):
        parser.error("min-lr-ratio must lie in (0,1]")
    if args.loss_config:
        args.loss_config = str(Path(args.loss_config).expanduser().resolve())
        if not Path(args.loss_config).is_file():
            parser.error("loss-config does not exist")
    python = shutil.which(os.path.expanduser(args.python))
    if python is None:
        parser.error(f"Python executable does not exist or is not executable: {args.python}")
    # Resolving a venv's Python symlink would silently select the base Python
    # and lose the packages installed in the requested training environment.
    python = str(Path(python).absolute())
    data_config = Path(args.data_config).expanduser().resolve()
    weights = Path(args.resume or args.pretrained).expanduser().resolve()
    if not weights.is_file():
        parser.error(f"Initialization checkpoint does not exist: {weights}")
    try:
        data = _data_summary(data_config)
        selected, active = _selected_snapshot(args.gpus)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    default_output = Path.cwd() / "outputs" / ("multicamera4rc_gpu" + "_".join(map(str, args.gpus)) + "_" + timestamp)
    output = Path(args.output).expanduser().resolve() if args.output else default_output
    command = _command(args, python, data_config, weights, output)
    environment = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                   "CUDA_VISIBLE_DEVICES": ",".join(gpu.uuid for gpu in selected),
                   "OMP_NUM_THREADS": str(args.cpu_threads)}
    report = {"dry_run": args.dry_run, "physical_gpu_indices": list(args.gpus),
              "local_rank_to_physical_gpu": {str(rank): gpu.index for rank, gpu in enumerate(selected)},
              "gpus": [asdict(gpu) for gpu in selected],
              "existing_compute_processes": [asdict(process) for process in active],
              "global_batch_size": args.batch_size * len(args.gpus), "data": data,
              "resource_policy": {"allow_shared": args.allow_shared, "wait_free_seconds": args.wait_free},
              "output": str(output), "environment": environment, "argv": command,
              "command": " ".join(f"{key}={shlex.quote(value)}" for key, value in environment.items()) + " " + shlex.join(command)}
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    if not args.resume and output.exists() and any(output.iterdir()):
        print(f"Fresh-training output directory is not empty: {output}. Choose a new --output.", file=sys.stderr)
        return 2
    if active and args.wait_free:
        print(f"Waiting in the foreground for physical GPUs {','.join(map(str, args.gpus))}; timeout {args.wait_free:g}s.", flush=True)
        deadline = time.monotonic() + args.wait_free
        identities = tuple(gpu.uuid for gpu in selected)
        while active and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            time.sleep(min(5, max(0, remaining)))
            selected, active = _selected_snapshot(args.gpus)
            if tuple(gpu.uuid for gpu in selected) != identities:
                raise RuntimeError("Physical GPU identities changed while waiting; repeat preflight")
        report["existing_compute_processes"] = [asdict(process) for process in active]
        report["gpus"] = [asdict(gpu) for gpu in selected]
    if active and not args.allow_shared:
        detail = ", ".join(f"{process.gpu_uuid}: PID {process.pid} ({process.name})" for process in active)
        print(f"Selected GPUs are busy; training was not started. {detail}\n"
              "Use --wait-free SECONDS for an explicit foreground wait, or --allow-shared to explicitly share.",
              file=sys.stderr)
        return 2
    # Metadata and logging directories are created only after the resource gate.
    output.mkdir(parents=True, exist_ok=True)
    (output / "launch.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    child_environment = os.environ.copy()
    child_environment.update(environment)
    return subprocess.run(command, env=child_environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
