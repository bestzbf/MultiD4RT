#!/usr/bin/env python3
"""Mixed dataset training for D4RT."""

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from datasets.factory import create_training_dataset
from datasets.collate import d4rt_collate_fn
from models import create_d4rt
from losses import D4RTLoss
import json
import time
import os
import contextlib
import math
import queue
import signal
import threading
import re
from datetime import timedelta


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def align_config_to_patch_provider(config: dict, patch_provider: str) -> dict:
    """Keep dataset patch materialization consistent with explicit patch providers."""
    if patch_provider not in {
        "precomputed_resized",
        "precomputed_highres",
        "sampled_resized",
        "sampled_highres",
    }:
        return config

    config = dict(config)
    if patch_provider == "precomputed_highres":
        config["precompute_patches"] = True
        config["precompute_from_highres"] = True
        config["return_highres_video"] = False
    elif patch_provider == "precomputed_resized":
        config["precompute_patches"] = True
        config["precompute_from_highres"] = False
        config["return_highres_video"] = False
    elif patch_provider == "sampled_resized":
        config["precompute_patches"] = False
        config["precompute_from_highres"] = False
        config["return_highres_video"] = False
    elif patch_provider == "sampled_highres":
        config["precompute_patches"] = False
        config["precompute_from_highres"] = False
        config["return_highres_video"] = True
    return config


def infer_resume_start_epoch_from_path(resume_path: str | None) -> int | None:
    """Best-effort first training epoch hint before the checkpoint is loaded."""
    if not resume_path:
        return None
    name = Path(resume_path).name
    match = re.search(r"_(\d+)\.pth$", name)
    if match is None:
        return None
    # Checkpoints are named with the 1-based completed epoch
    # (checkpoint_latest_428.pth stores epoch=427), so the first post-resume
    # training epoch is the number in the filename.
    return int(match.group(1))


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    video = batch.get("video")
    local_patches = batch.get("local_patches")
    batch = {
        k: v.to(device, non_blocking=True)
        if isinstance(v, torch.Tensor) and k not in {"video", "local_patches"}
        else v
        for k, v in batch.items()
    }
    if isinstance(video, (list, tuple)):
        batch["video"] = torch.stack(
            [v.to(device, non_blocking=True) for v in video],
            dim=0,
        )
    elif isinstance(video, torch.Tensor):
        batch["video"] = video.to(device, non_blocking=True)
    if isinstance(local_patches, (list, tuple)):
        batch["local_patches"] = torch.stack(
            [v.to(device, non_blocking=True) for v in local_patches],
            dim=0,
        )
    elif isinstance(local_patches, torch.Tensor):
        batch["local_patches"] = local_patches.to(device, non_blocking=True)
    batch["targets"] = {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch["targets"].items()
    }
    transform_metadata = batch.get("transform_metadata")
    if isinstance(transform_metadata, dict):
        batch["transform_metadata"] = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in transform_metadata.items()
        }
    if batch["video"].dtype == torch.uint8:
        batch["video"] = batch["video"].float() / 255.0
    local_patches = batch.get("local_patches")
    if isinstance(local_patches, torch.Tensor) and local_patches.dtype == torch.uint8:
        batch["local_patches"] = local_patches.float() / 255.0
    return batch


class BatchPrefetchIterator:
    """Prefetch CPU batches on a background thread.

    This is intentionally a thread wrapper around the existing DataLoader
    iterator, not DataLoader multiprocessing.  Planned mode requires a single
    sequential consumer for its spool/window state, and this keeps that
    invariant while overlapping CPU batch loading with GPU compute.
    """

    _STOP = object()

    def __init__(self, iterable, depth: int):
        self._iterator = iter(iterable)
        self._depth = max(1, int(depth))
        self._queue: queue.Queue = queue.Queue(maxsize=self._depth)
        self._closed = threading.Event()
        self._last_get_wait_s = 0.0
        self._last_qsize_before = None
        self._last_qsize_after = None
        self._thread = threading.Thread(
            target=self._producer,
            name="D4RTBatchPrefetch",
            daemon=True,
        )
        self._thread.start()

    def _producer(self) -> None:
        try:
            for batch in self._iterator:
                if self._closed.is_set():
                    break
                self._queue.put((True, batch))
        except BaseException as exc:
            self._queue.put((False, exc))
        finally:
            self._queue.put((True, self._STOP))

    def __iter__(self):
        return self

    def __next__(self):
        self._last_qsize_before = self._safe_qsize()
        t0 = time.perf_counter()
        ok, item = self._queue.get()
        self._last_get_wait_s = time.perf_counter() - t0
        self._last_qsize_after = self._safe_qsize()
        if not ok:
            raise item
        if item is self._STOP:
            raise StopIteration
        return item

    def _safe_qsize(self):
        try:
            return self._queue.qsize()
        except NotImplementedError:
            return None

    def last_stats(self) -> dict:
        return {
            "depth": self._depth,
            "get_wait_s": self._last_get_wait_s,
            "qsize_before": self._last_qsize_before,
            "qsize_after": self._last_qsize_after,
        }

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=2.0)


class TrainPhaseWatchdog:
    """Emit a slow-path heartbeat when a rank stays in one train phase too long."""

    def __init__(self, rank: int):
        self.threshold_s = float(os.getenv("D4RT_TRAIN_WATCHDOG_S", "0") or "0")
        self.interval_s = max(
            1.0,
            float(os.getenv("D4RT_TRAIN_WATCHDOG_INTERVAL_S", "60") or "60"),
        )
        # After this many seconds past threshold, kill the process so NCCL
        # propagates the error to other ranks instead of hanging forever.
        self.kill_after_s = float(os.getenv("D4RT_TRAIN_WATCHDOG_KILL_AFTER_S", "0") or "0")
        self.kill_descendants = os.getenv(
            "D4RT_TRAIN_WATCHDOG_KILL_DESCENDANTS", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.enabled = self.threshold_s > 0
        self.rank = int(rank)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._epoch = None
        self._batch = None
        self._phase = "init"
        self._extra = ""
        self._updated = time.monotonic()
        self._last_report = 0.0
        self._thread = None
        if self.enabled:
            self._thread = threading.Thread(
                target=self._run,
                name="D4RTTrainWatchdog",
                daemon=True,
            )
            self._thread.start()

    def update(self, epoch=None, batch=None, phase: str = "", extra: str = "") -> None:
        if not self.enabled:
            return
        with self._lock:
            if epoch is not None:
                self._epoch = int(epoch)
            if batch is not None:
                self._batch = int(batch)
            if phase:
                self._phase = str(phase)
            self._extra = str(extra)
            self._updated = time.monotonic()
            self._last_report = 0.0

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            now = time.monotonic()
            with self._lock:
                elapsed = now - self._updated
                if elapsed < self.threshold_s:
                    continue
                if self._last_report and now - self._last_report < self.interval_s:
                    continue
                self._last_report = now
                epoch = self._epoch
                batch = self._batch
                phase = self._phase
                extra = self._extra
            suffix = f" {extra}" if extra else ""
            print(
                f"[TrainWatchdog rank{self.rank}] "
                f"epoch={epoch} batch={batch} phase={phase} "
                f"elapsed={elapsed:.1f}s threshold={self.threshold_s:.1f}s{suffix}",
                flush=True,
            )
            if self.kill_after_s > 0 and elapsed >= self.threshold_s + self.kill_after_s:
                print(
                    f"[TrainWatchdog rank{self.rank}] elapsed={elapsed:.1f}s exceeds "
                    f"kill_after={self.kill_after_s:.1f}s, sending SIGKILL to self"
                    f"{' and descendants' if self.kill_descendants else ''}",
                    flush=True,
                )
                self._kill_process_tree()

    def _kill_process_tree(self) -> None:
        pid = os.getpid()
        if self.kill_descendants:
            for child_pid in reversed(_collect_descendant_pids(pid)):
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    pass
        os.kill(pid, signal.SIGKILL)


def _collect_descendant_pids(root_pid: int) -> list[int]:
    """Return descendant PIDs using /proc without requiring psutil."""
    children: dict[int, list[int]] = {}
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            stat = (entry / "stat").read_text()
            close = stat.rfind(")")
            if close < 0:
                continue
            fields = stat[close + 2 :].split()
            # fields[0] is process state, fields[1] is PPID.
            ppid = int(fields[1])
        except Exception:
            continue
        children.setdefault(ppid, []).append(pid)

    ordered: list[int] = []
    stack = list(children.get(int(root_pid), ()))
    while stack:
        pid = stack.pop()
        ordered.append(pid)
        stack.extend(children.get(pid, ()))
    return ordered


def summarize_spool_ready(dataset, focus_index: int | None = None) -> dict:
    """Return a cheap ready-file summary for planned-mode diagnostics."""
    spool = getattr(dataset, "spool", None)
    if spool is None:
        return {}
    generation = getattr(dataset, "_generation", getattr(spool, "_generation", 0))
    prefix = f"g{int(generation):04d}_"
    indices = []
    try:
        for path in spool.spool_dir.iterdir():
            name = path.name
            if not name.startswith(prefix) or not name.endswith(".ready"):
                continue
            try:
                indices.append(int(name[len(prefix):len(prefix) + 8]))
            except ValueError:
                continue
    except OSError:
        return {"ready_count": 0}
    if not indices:
        return {"ready_count": 0}
    indices.sort()
    out = {
        "ready_count": len(indices),
        "ready_min": indices[0],
        "ready_max": indices[-1],
    }
    if focus_index is not None:
        ready_set = set(indices)
        contiguous = 0
        cursor = int(focus_index)
        while cursor in ready_set:
            contiguous += 1
            cursor += 1
        next_ready = next((idx for idx in indices if idx >= focus_index), None)
        near = [idx for idx in indices if focus_index - 8 <= idx <= focus_index + 16]
        out.update(
            focus_index=int(focus_index),
            contiguous_from_focus=contiguous,
            next_ready=next_ready,
            ready_near=near,
        )
    return out


def planned_startup_warmup(
    dataset,
    target_samples: int,
    timeout_s: float,
    local_rank: int,
    epoch: int,
) -> None:
    """Wait for an initial contiguous planned-mode window before timing data."""
    target_samples = max(0, int(target_samples))
    if target_samples <= 0:
        return

    current = dataset
    while current is not None and getattr(current, "spool", None) is None:
        current = getattr(current, "dataset", None)
    if current is None or getattr(current, "spool", None) is None:
        return

    try:
        target_samples = min(target_samples, len(current))
    except Exception:
        pass
    if target_samples <= 0:
        return

    start = time.perf_counter()
    deadline = start + max(0.0, float(timeout_s))
    stats = summarize_spool_ready(current, focus_index=0)
    timed_out = False
    while int(stats.get("contiguous_from_focus", 0) or 0) < target_samples:
        if time.perf_counter() >= deadline:
            timed_out = True
            break
        time.sleep(0.1)
        stats = summarize_spool_ready(current, focus_index=0)

    waited_s = time.perf_counter() - start
    if local_rank == 0 and (waited_s >= 0.5 or timed_out):
        status = "timeout" if timed_out else "ready"
        print(
            f"[PlannedWarmup rank{local_rank}] "
            f"epoch={epoch} status={status} waited={waited_s * 1000:.0f}ms "
            f"target_contiguous={target_samples} "
            f"contiguous={stats.get('contiguous_from_focus')} "
            f"spool_ready={stats.get('ready_count')} "
            f"spool_min={stats.get('ready_min')} "
            f"spool_max={stats.get('ready_max')}",
            flush=True,
        )


def format_dataset_counts(dataset_names) -> str:
    counts = {}
    for name in dataset_names or []:
        counts[name] = counts.get(name, 0) + 1
    return ",".join(f"{name}:{counts[name]}" for name in sorted(counts)) or "<none>"


def _slice_first_dim(value, indices: torch.Tensor, batch_size: int):
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
        return value.detach().index_select(0, indices)
    return value


@torch.no_grad()
def compute_per_dataset_loss_metrics(
    loss_fn: D4RTLoss,
    outputs: dict,
    targets: dict,
    batch: dict,
    num_frames: int,
    metric_dataset_set=None,
) -> dict:
    dataset_names = batch.get("dataset_names") or []
    if not dataset_names or "pos_3d" not in outputs:
        return {}
    batch_size = int(outputs["pos_3d"].shape[0])
    groups: dict[str, list[int]] = {}
    for idx, name in enumerate(dataset_names[:batch_size]):
        groups.setdefault(str(name), []).append(idx)
    if not groups:
        return {}

    metric_keys = {
        "loss": "loss",
        "loss_3d": "loss_3d",
        "loss_3d_nocon": "loss_3d_unweighted",
        "loss_raw_3d": "loss_raw_3d",
        "loss_2d": "loss_2d",
        "loss_vis": "loss_vis",
        "loss_disp": "loss_disp",
        "loss_conf": "loss_conf",
        "raw_3d_l1": "metric_raw_3d_l1",
        "valid_3d_ratio": "metric_valid_3d_query_ratio",
        "static_query_ratio": "metric_static_query_ratio",
        "temporal_query_ratio": "metric_temporal_query_ratio",
        "pred_abs_depth_mean": "metric_pred_abs_depth_mean",
        "target_abs_depth_mean": "metric_target_abs_depth_mean",
    }
    out = {}
    device = outputs["pos_3d"].device
    for dataset_name, sample_indices in sorted(groups.items()):
        index_tensor = torch.tensor(sample_indices, device=device, dtype=torch.long)
        sub_outputs = {
            key: _slice_first_dim(value, index_tensor, batch_size)
            for key, value in outputs.items()
        }
        sub_targets = {
            key: _slice_first_dim(value, index_tensor, batch_size)
            for key, value in targets.items()
        }
        sub_dataset_id = _slice_first_dim(batch["dataset_id"], index_tensor, batch_size)
        sub_t_cam = _slice_first_dim(batch["t_cam"], index_tensor, batch_size)
        if metric_dataset_set and getattr(loss_fn, "dust3r_metric_branch", False):
            sub_is_metric = torch.full(
                (sub_dataset_id.shape[0],),
                dataset_name in metric_dataset_set,
                dtype=torch.bool,
                device=sub_dataset_id.device,
            )
        else:
            sub_is_metric = None
        sub_loss = loss_fn(
            sub_outputs,
            sub_targets,
            normalize_groups=sub_dataset_id * num_frames + sub_t_cam,
            is_metric=sub_is_metric,
        )
        out[dataset_name] = {"count": len(sample_indices)}
        for log_key, loss_key in metric_keys.items():
            if loss_key in sub_loss:
                value = sub_loss[loss_key]
                out[dataset_name][log_key] = f"{float(value.detach().float().item()):.4f}"
    return out


def _as_int(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return int(value.detach().flatten()[0].item())
    try:
        return int(value)
    except Exception:
        return None


def _format_frame_indices(value) -> str:
    if value is None:
        return "<unknown>"
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().flatten().tolist()
    try:
        vals = [int(v) for v in value]
    except Exception:
        return str(value)
    if not vals:
        return "0[]"
    step = vals[1] - vals[0] if len(vals) > 1 else 0
    regular = len(vals) <= 2 or all(vals[i] - vals[i - 1] == step for i in range(1, len(vals)))
    suffix = f",step={step}" if regular and step not in (0, 1) else ""
    return f"{len(vals)}[{min(vals)}..{max(vals)}{suffix}]"


def _tensor_range(tensor, sample_idx: int) -> str:
    try:
        values = tensor[sample_idx].detach()
        if values.numel() == 0:
            return "empty"
        return f"{int(values.min().item())}..{int(values.max().item())}"
    except Exception:
        return "?"


def _target_ratio(batch: dict, key: str, sample_idx: int) -> str:
    try:
        target = batch["targets"][key][sample_idx].detach()
        if target.numel() == 0:
            return "nan"
        return f"{target.float().mean().item():.3f}"
    except Exception:
        return "?"


def infer_next_planned_index(batch: dict) -> int | None:
    indices = []
    for metadata in batch.get("metadata") or []:
        if isinstance(metadata, dict):
            idx = _as_int(metadata.get("planned_local_index"))
            if idx is not None:
                indices.append(idx)
    if not indices:
        return None
    return max(indices) + 1


def format_batch_sample_details(batch: dict, max_samples: int = 8) -> list[str]:
    """Compact per-sample diagnostics for slow data waits."""
    dataset_names = batch.get("dataset_names") or []
    sequence_names = batch.get("sequence_names") or []
    metadata_list = batch.get("metadata") or []
    batch_size = max(len(dataset_names), len(sequence_names), len(metadata_list))
    lines = []
    for i in range(min(batch_size, max_samples)):
        md = metadata_list[i] if i < len(metadata_list) and isinstance(metadata_list[i], dict) else {}
        dataset = dataset_names[i] if i < len(dataset_names) else md.get("dataset_name", "?")
        sequence = sequence_names[i] if i < len(sequence_names) else md.get("sequence_name", "?")
        frame_indices = (
            md.get("planned_frame_indices")
            or md.get("frame_indices")
            or md.get("logical_frame_indices")
        )
        local_idx = _as_int(md.get("planned_local_index"))
        global_idx = _as_int(md.get("planned_global_index"))
        planned = (
            f"planned={local_idx}"
            + (f"/global={global_idx}" if global_idx is not None else "")
            if local_idx is not None
            else "planned=?"
        )
        tracks = md.get("has_tracks", md.get("has_temporal_supervision", "?"))
        semantics = md.get("query_semantics", "?")
        num_frames = md.get("num_frames_in_sequence", md.get("num_frames_total", "?"))
        line = (
            f"  sample[{i}] {planned} dataset={dataset} seq={sequence} "
            f"frames={_format_frame_indices(frame_indices)} total_frames={num_frames} "
            f"tracks={tracks} sem={semantics} "
            f"t_src={_tensor_range(batch.get('t_src'), i)} "
            f"t_tgt={_tensor_range(batch.get('t_tgt'), i)} "
            f"t_cam={_tensor_range(batch.get('t_cam'), i)} "
            f"mask3d={_target_ratio(batch, 'mask_3d', i)} "
            f"mask2d={_target_ratio(batch, 'mask_2d', i)} "
            f"static={_target_ratio(batch, 'is_static_reprojection', i)}"
        )
        lines.append(line)

        # Add load timing info if available
        load_timing = md.get("_load_timing")
        if isinstance(load_timing, dict) and load_timing.get("total_s", 0) > 0:
            timing = load_timing
            lines.append(
                f"    load_time={timing.get('total_s', 0)*1000:.0f}ms: "
                f"scene_data={timing.get('scene_data_s', 0)*1000:.0f}ms "
                f"rgb={timing.get('rgb_load_s', 0)*1000:.0f}ms "
                f"depth={timing.get('depth_load_s', 0)*1000:.0f}ms "
                f"precomputed={timing.get('precomputed_s', 0)*1000:.0f}ms"
            )
            if "precomputed_range_cache_hits" in timing:
                local_h5 = int(timing.get("precomputed_range_local_h5", 0))
                lines.append(
                    f"    h5_range_cache: "
                    f"hit={int(timing.get('precomputed_range_cache_hits', 0))} "
                    f"miss={int(timing.get('precomputed_range_cache_misses', 0))} "
                    f"tasks={int(timing.get('precomputed_range_range_tasks', 0))} "
                    f"bytes={timing.get('precomputed_range_range_bytes', 0) / 1024**2:.1f}MB"
                    + (f" local_h5={local_h5}" if local_h5 else "")
                )
        builder_profile = md.get("_builder_profile")
        if isinstance(builder_profile, dict) and builder_profile.get("load_s", 0) > 0:
            lines.append(
                f"    build_time: "
                f"stage={builder_profile.get('stage_s', 0)*1000:.0f}ms "
                f"load={builder_profile.get('load_s', 0)*1000:.0f}ms "
                f"transform={builder_profile.get('transform_s', 0)*1000:.0f}ms "
                f"query={builder_profile.get('query_s', 0)*1000:.0f}ms"
            )
            stage_stats = builder_profile.get("stage_stats")
            if isinstance(stage_stats, dict):
                lines.append(
                    f"    stage_cache: files={stage_stats.get('manifest_files')} "
                    f"hit={stage_stats.get('cache_hits')} "
                    f"miss={stage_stats.get('cache_misses')} "
                    f"linked={stage_stats.get('linked')} "
                    f"copied={stage_stats.get('copied')} "
                    f"bytes={int(stage_stats.get('bytes', 0)) / 1024**2:.1f}MB "
                    f"materialize={stage_stats.get('materialize_s', 0)*1000:.0f}ms "
                    f"scannetpp_depth={stage_stats.get('scannetpp_depth_stage_s', 0)*1000:.0f}ms"
                )

    if batch_size > max_samples:
        lines.append(f"  ... {batch_size - max_samples} more samples omitted")
    return lines


class ProfilingCollateFn:
    """Small optional wrapper for timing CPU collate in planned mode."""

    def __init__(self, rank: int, interval: int):
        self.rank = rank
        self.interval = max(1, int(interval))
        self.count = 0
        self.print_periodic = os.getenv("D4RT_PROFILE_COLLATE_PERIODIC", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        self.slow_threshold_s = float(os.getenv("D4RT_COLLATE_SLOW_THRESHOLD_S", "2.0"))

    def __call__(self, batch):
        t0 = time.perf_counter()
        result = d4rt_collate_fn(batch)
        dt = time.perf_counter() - t0
        self.count += 1
        if (
            (self.print_periodic and (self.count <= 3 or self.count % self.interval == 0))
            or dt >= self.slow_threshold_s
        ):
            tag = "DataProfileSlow" if dt >= self.slow_threshold_s else "DataProfile"
            print(
                f"[{tag} rank{self.rank}] collate "
                f"batch={self.count} samples={len(batch)} time={dt * 1000:.1f}ms",
                flush=True,
            )
        return result


def maybe_fallback_patch_provider(
    model: nn.Module,
    batch: dict,
    configured_provider: str,
    local_rank: int,
    warned: bool,
) -> bool:
    if configured_provider != "sampled_highres":
        return warned
    if batch.get("transform_metadata") is not None:
        return warned

    decoder = unwrap_model(model).decoder
    if getattr(decoder, "patch_provider", None) == "sampled_highres":
        decoder.patch_provider = "auto"
    if local_rank == 0 and not warned:
        print(
            "[Patch Provider] WARNING: batch is missing transform_metadata/highres_video; "
            "falling back from sampled_highres to auto.",
            flush=True,
        )
    return True


def set_dataset_epoch(dataset, epoch: int, progress_cb=None) -> None:
    current = dataset
    while current is not None:
        if hasattr(current, "set_epoch"):
            try:
                current.set_epoch(epoch, progress_cb=progress_cb)
            except TypeError:
                # Datasets without progress_cb support (older signature).
                current.set_epoch(epoch)
            return
        current = getattr(current, "dataset", None)


def cleanup_dataset(dataset) -> None:
    current = dataset
    while current is not None:
        if hasattr(current, "cleanup"):
            current.cleanup()
            return
        current = getattr(current, "dataset", None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Dataset config YAML")
    parser.add_argument("--val-config", type=str, default=None, help="Separate val config YAML (optional, defaults to --config)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--compile", action="store_true", help="torch.compile the model (faster steady-state, slow first-batch)")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--lr-warmup-steps", type=int, default=2500)
    parser.add_argument("--encoder-lr-mult", type=float, default=1.0,
                        help="LR multiplier for encoder parameters. Default 1.0 keeps one LR for the whole model.")
    parser.add_argument("--decoder-lr-mult", type=float, default=1.0,
                        help="LR multiplier for decoder parameters. Default 1.0 keeps one LR for the whole model.")
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=48)
    parser.add_argument("--num-queries", type=int, default=2048)
    parser.add_argument("--loss-w-3d", type=float, default=1.0)
    parser.add_argument("--loss-w-raw-3d", type=float, default=0.0)
    parser.add_argument("--loss-w-2d", type=float, default=0.1)
    parser.add_argument("--loss-w-vis", type=float, default=0.01)
    parser.add_argument("--loss-w-disp", type=float, default=0.1)
    parser.add_argument("--loss-w-conf", type=float, default=0.2)
    parser.add_argument("--loss-conf-warmup-steps", type=int, default=0,
                        help="Linearly ramp confidence penalty and Kendall weighting from 0 to --loss-w-conf. "
                             "Default 0 keeps confidence fully enabled from the first step.")
    parser.add_argument("--loss-w-normal", type=float, default=0.5)
    parser.add_argument("--loss-w-static-reprojection", type=float, default=1.0,
                        help="Weight for static-reprojection (has_tracks=False) queries in 3D loss. "
                             "Default 1.0 = no change. Set <1.0 to down-weight static queries.")
    parser.add_argument("--shared-depth-norm", action="store_true", default=False,
                        help="Use target median depth to normalize both pred and target (scale-aware). "
                             "Default False: pred and target are normalized by their own median depth.")
    parser.add_argument("--no-shared-depth-norm", dest="shared_depth_norm", action="store_false",
                        help="Use independent normalization (paper default): pred and target each "
                             "divided by their own median depth. Scale-invariant but blind to depth-scale drift.")
    parser.add_argument("--loss-3d-mode", type=str, default="scale_invariant",
                        choices=["scale_invariant", "raw_l1", "log_space"],
                        help="3D loss mode. 'scale_invariant': paper default (median-norm + log1p). "
                             "'log_space': depth-invariant log/angular loss (no normalization needed). "
                             "'raw_l1': raw L1 for debugging.")
    parser.add_argument("--dust3r-metric-branch", action="store_true", default=False,
                        help="Enable DUSt3R-style two-branch normalization. For samples in "
                             "--metric-datasets, the 3D loss uses target normalizer for both "
                             "pred and target (z̄=z, scale-anchoring). Other samples use the "
                             "paper default (independent normalization, scale-invariant). "
                             "Off by default; behaviour identical to current paper-default path.")
    parser.add_argument("--metric-datasets", type=str,
                        default="scannet,scannetpp,dynamic_replica,kubric,blendedmvs,pointodyssey,"
                                "tartanair,vkitti2,mvssynth,waymo",
                        help="Comma-separated dataset names treated as metric for the DUSt3R "
                             "metric branch. Datasets not listed (e.g. co3dv2) keep "
                             "scale-invariant independent normalization. Only used when "
                             "--dust3r-metric-branch is enabled.")
    parser.add_argument("--resume-warmup-steps", type=int, default=0,
                        help="When >0 and --resume is given, apply a fresh-start-style linear "
                             "warmup 0 -> peak_lr over this many optimizer steps starting at "
                             "the resumed global_step. After the warmup window, the cosine "
                             "schedule resumes from the original global_step's cosine value. "
                             "Useful when resuming with a new loss configuration (e.g. enabling "
                             "the DUSt3R metric branch) where the loss surface suddenly "
                             "produces much larger gradients. 0 disables.")
    parser.add_argument("--output-dir", type=str, default="outputs/mixture")
    parser.add_argument("--pretrain", type=str, default=None, help="Path to pretrained checkpoint")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument(
        "--reset-confidence-head-on-pretrain",
        action="store_true",
        help=(
            "Zero-init decoder confidence head after loading --pretrain. "
            "Useful when transferring from a checkpoint whose uncertainty calibration "
            "does not match the new dataset mix."
        ),
    )
    parser.add_argument("--quick-test", action="store_true", help="Quick test mode with only 10 samples")
    parser.add_argument("--save-interval", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--val-interval", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--val-samples", type=int, default=200, help="Number of val samples per validation run")
    parser.add_argument("--keep-checkpoints", type=int, default=10, help="Keep last N checkpoints (except milestone)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--log-interval", type=int, default=50, help="Print training logs every N batches")
    parser.add_argument(
        "--variant",
        type=str,
        default="large",
        choices=["base", "large", "huge", "giant"],
        help="Model variant: base, large, huge, or giant",
    )
    parser.add_argument(
        "--encoder-patch-size",
        type=int,
        default=16,
        help=(
            "Spatial patch size for the custom VideoMAE-initialized encoder. "
            "Use 14 for VideoMAEv2-Giant/ViT-g to match its patch embedding."
        ),
    )
    parser.add_argument(
        "--dist-timeout-minutes",
        type=int,
        default=60,
        help="Distributed process group timeout in minutes.",
    )
    parser.add_argument(
        "--broadcast-buffers",
        action="store_true",
        help=(
            "Enable DDP buffer broadcasts before each forward pass. Disabled by default "
            "because this model has no batch-norm style running stats that need syncing."
        ),
    )
    parser.add_argument(
        "--use-videomae-v2-init",
        action="store_true",
        help="Initialize the encoder from VideoMAE V2 weights instead of VideoMAE V1.",
    )
    parser.add_argument(
        "--videomae-model",
        type=str,
        default=None,
        help="Optional HuggingFace model ID or local path for encoder VideoMAE initialization.",
    )
    parser.add_argument(
        "--patch-provider", type=str, default="auto",
        help=(
            "Patch provider: auto | precomputed_resized | precomputed_highres | sampled_resized | sampled_highres.\n"
            "  auto              → resolves to precomputed_resized when local_patches is precomputed.\n"
            "  precomputed_highres → uses local_patches extracted from highres_video during sample build.\n"
            "  sampled_resized   → samples patches on GPU from the resized encoder video.\n"
            "  sampled_highres   → samples from highres_video (original crop res).\n"
            "  NOTE: 'auto' does NOT enable high-res patches. For the paper's strongest setting\n"
            "  use --patch-provider sampled_highres with a config that sets precompute_patches: false."
        )
    )
    parser.add_argument(
        "--planned-mode",
        action="store_true",
        help="Use planned sample-bundle prefetch mode (for COS acceleration)",
    )
    parser.add_argument(
        "--builder-workers",
        type=int,
        default=2,
        help="Number of background sample builder processes in planned mode",
    )
    parser.add_argument(
        "--prefetch-depth",
        type=int,
        default=32,
        help="Number of samples to prefetch ahead in planned mode",
    )
    parser.add_argument(
        "--batch-prefetch-depth",
        type=int,
        default=0,
        help=(
            "CPU batch prefetch queue depth. In planned mode this overlaps "
            "spool pickle.load + collate with GPU compute without enabling "
            "DataLoader multiprocessing."
        ),
    )
    parser.add_argument(
        "--profile-data-loading",
        action="store_true",
        help=(
            "Print lightweight data-path timing. Use for short diagnostics only; "
            "it may add small overhead."
        ),
    )
    parser.add_argument(
        "--data-profile-interval",
        type=int,
        default=20,
        help="Print data-path profile every N batches/samples when profiling is enabled.",
    )
    args = parser.parse_args()

    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=args.dist_timeout_minutes),
        )
    torch.cuda.set_device(local_rank)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = align_config_to_patch_provider(config, args.patch_provider)

    # Inject planned mode settings into config
    if args.planned_mode:
        config['planned_mode'] = True
        config['builder_workers'] = args.builder_workers
        config['prefetch_depth'] = args.prefetch_depth
        resume_start_epoch = infer_resume_start_epoch_from_path(args.resume)
        if resume_start_epoch is not None:
            config['planned_initial_epoch'] = resume_start_epoch
            config['planned_start_immediately'] = True
        if local_rank == 0:
            print(f"[Planned Mode] Enabled with {args.builder_workers} builder workers, prefetch depth {args.prefetch_depth}")
            if resume_start_epoch is not None:
                print(
                    f"[Planned Mode] Initial prefetch epoch={resume_start_epoch} "
                    f"(from resume filename {Path(args.resume).name})",
                    flush=True,
                )

    # Load val config (if separate)
    val_config = config
    if args.val_config is not None:
        with open(args.val_config) as f:
            val_config = yaml.safe_load(f)
        val_config = align_config_to_patch_provider(val_config, args.patch_provider)

    # Planned mode is compatible with val only in non-distributed runs
    # (SequentialSampler + num_workers=0 required by PlannedDataset).
    if val_config.get('planned_mode') and distributed:
        val_config = {**val_config, 'planned_mode': False}

    # Create dataset
    rank = dist.get_rank() if distributed else 0
    train_dataset = create_training_dataset(config, split='train', rank=rank, world_size=world_size)
    val_loader = None
    if args.val_interval > 0 and args.val_samples > 0:
        val_dataset = create_training_dataset(val_config, split='val', rank=rank, world_size=world_size)
        from torch.utils.data import Subset
        val_dataset = Subset(val_dataset, range(min(args.val_samples, len(val_dataset))))
    else:
        val_dataset = None
    if local_rank == 0:
        val_len = len(val_dataset) if val_dataset is not None else 0
        print(f"Dataset: {train_dataset.get_dataset_names()}")
        print(f"Train Length: {len(train_dataset)}, Val Length: {val_len}")
        print(f"[Patch Provider] configured={args.patch_provider!r} "
              f"{'⚠ WARNING: high-res patches NOT active (use --patch-provider sampled_highres for paper setting)' if args.patch_provider == 'auto' else ''}")

    # Quick validation mode: use only first 10 samples
    if args.quick_test:
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(min(10, len(train_dataset))))
        if local_rank == 0:
            print(f"Quick test mode: using {len(train_dataset)} samples")

    # DataLoader sampler setup
    if args.planned_mode:
        # Planned mode: dataset 内部已处理 rank 分片和确定性顺序
        train_sampler = SequentialSampler(train_dataset)
    elif distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = RandomSampler(train_dataset)

    train_drop_last = len(train_dataset) >= args.batch_size
    if local_rank == 0 and not train_drop_last:
        print(
            "[DataLoader] train_dataset is smaller than batch_size; "
            "disabling drop_last so quick/small sanity runs still produce batches."
        )

    # Adjust num_workers for planned mode (samples are pre-built)
    train_num_workers = args.num_workers
    if args.planned_mode:
        train_num_workers = 0  # No workers needed, samples are pre-built
        if local_rank == 0:
            print(f"[Planned Mode] Setting DataLoader num_workers=0 (samples pre-built by {args.builder_workers} builder processes)")
            if args.batch_prefetch_depth > 0:
                print(
                    f"[Planned Mode] CPU batch prefetch enabled "
                    f"(depth={args.batch_prefetch_depth})"
                )

    train_collate_fn = d4rt_collate_fn
    if args.profile_data_loading and args.planned_mode:
        train_collate_fn = ProfilingCollateFn(
            rank=local_rank,
            interval=args.data_profile_interval,
        )

    train_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=train_num_workers,
        collate_fn=train_collate_fn,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=train_num_workers > 0,
        drop_last=train_drop_last,
    )
    if train_num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    if len(train_loader) == 0:
        raise RuntimeError(
            "Train loader is empty. Check dataset size, batch size, and drop_last "
            f"(samples={len(train_dataset)}, batch_size={args.batch_size}, drop_last={train_drop_last})."
        )
    if val_dataset is not None:
        val_planned = args.planned_mode and not distributed
        val_num_workers = 0 if val_planned else max(2, args.num_workers // 2)
        if distributed:
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        else:
            val_sampler = SequentialSampler(val_dataset)
        val_loader_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=val_num_workers,
            collate_fn=d4rt_collate_fn,
            sampler=val_sampler,
            pin_memory=True,
            persistent_workers=val_num_workers > 0,
            drop_last=False,
        )
        if val_num_workers > 0:
            val_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    # Setup device and model
    device = torch.device(f"cuda:{local_rank}")
    amp_enabled = (device.type == "cuda")
    torch.set_float32_matmul_precision("high")
    videomae_model = args.videomae_model
    if args.encoder_patch_size <= 0:
        raise ValueError(f"--encoder-patch-size must be positive, got {args.encoder_patch_size}")
    if videomae_model is None and not args.use_videomae_v2_init:
        _videomae_roots = {
            "large": "/data1/zbf/pretrained/videomae-large",
            "base": "/data1/zbf/pretrained/videomae-base",
            "huge": "MCG-NJU/videomae-huge",
        }
        videomae_model = _videomae_roots.get(args.variant)
    if local_rank == 0:
        init_family = "VideoMAE V2" if args.use_videomae_v2_init else "VideoMAE V1"
        print(f"Encoder init: {init_family} ({videomae_model or 'variant default'})")
        print(f"Encoder patch size: {args.encoder_patch_size}")
    model = create_d4rt(variant=args.variant, img_size=args.resolution,
                        temporal_size=args.num_frames,
                        patch_size=(2, args.encoder_patch_size, args.encoder_patch_size),
                        query_patch_size=9, videomae_model=videomae_model,
                        use_videomae_v2=args.use_videomae_v2_init,
                        patch_provider=args.patch_provider,).to(device)
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            broadcast_buffers=args.broadcast_buffers,
        )

    if args.compile:
        if local_rank == 0:
            print("torch.compile: compiling model (mode=default, dynamic=False)...")
        model = torch.compile(model, mode="default", dynamic=False)
    val_model = getattr(model, "_orig_mod", model)

    # Load pretrained weights
    if args.pretrain:
        if local_rank == 0:
            print(f"Loading pretrained weights from {args.pretrain}")
        checkpoint = torch.load(args.pretrain, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model'))
        if state_dict is None:
            raise KeyError(
                f"Checkpoint {args.pretrain} missing both 'model_state_dict' and 'model' keys"
            )
        unwrap_model(model).load_state_dict(state_dict, strict=True)
        if args.reset_confidence_head_on_pretrain:
            decoder = unwrap_model(model).decoder
            torch.nn.init.zeros_(decoder.head_conf.weight)
            torch.nn.init.zeros_(decoder.head_conf.bias)
            if local_rank == 0:
                print(
                    "Reset decoder.head_conf after loading pretrained weights "
                    "to avoid stale uncertainty calibration."
                )

    # Optimizer and loss - separate weight decay for bias/norm and optional
    # encoder/decoder LR multipliers.
    param_groups_by_key: dict[tuple[bool, float], dict] = {}

    def _lr_multiplier_for_name(param_name: str) -> float:
        local_name = param_name.removeprefix("module.")
        if local_name.startswith("encoder."):
            return float(args.encoder_lr_mult)
        if local_name.startswith("decoder."):
            return float(args.decoder_lr_mult)
        return 1.0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        use_decay = not ('bias' in name or 'norm' in name or 'embed' in name)
        lr_mult = _lr_multiplier_for_name(name)
        key = (use_decay, lr_mult)
        if key not in param_groups_by_key:
            param_groups_by_key[key] = {
                'params': [],
                'weight_decay': args.weight_decay if use_decay else 0.0,
                'lr': args.lr * lr_mult,
                'lr_mult': lr_mult,
                'decay': use_decay,
            }
        param_groups_by_key[key]['params'].append(param)

    optimizer_groups = [
        group for group in param_groups_by_key.values()
        if group['params']
    ]
    optimizer = torch.optim.AdamW(optimizer_groups, lr=args.lr, fused=True)
    loss_fn = D4RTLoss(lambda_3d=args.loss_w_3d, lambda_raw_3d=args.loss_w_raw_3d,
                       lambda_2d=args.loss_w_2d,
                       lambda_vis=args.loss_w_vis, lambda_disp=args.loss_w_disp,
                       lambda_conf=args.loss_w_conf, lambda_normal=args.loss_w_normal,
                       static_reprojection_weight=args.loss_w_static_reprojection,
                       shared_depth_normalization=args.shared_depth_norm,
                       dust3r_metric_branch=args.dust3r_metric_branch,
                       debug_3d_loss_mode=args.loss_3d_mode)

    # Resolve metric-dataset whitelist for DUSt3R two-branch normalization.
    metric_dataset_set: set[str] = (
        {n.strip() for n in args.metric_datasets.split(",") if n.strip()}
        if args.dust3r_metric_branch else set()
    )

    # LR scheduler: warmup + cosine annealing
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = args.epochs * optimizer_steps_per_epoch
    # Resume-warmup window is anchored at the resumed global_step. It is
    # populated below right after we know the resume step. Until then it is
    # disabled (resume_warmup_start = -1, length = 0 → no effect).
    resume_warmup_state = {"start": -1, "length": 0}
    def lr_lambda(step):
        # Resume warmup: linear 0 -> 1.0 over [start, start+length], anchored
        # at the resumed global_step. Active only when --resume-warmup-steps>0
        # and we are inside the window. Identical multiplicative shape as the
        # cold-start warmup (step / warmup_steps) so 0 -> peak_lr.
        rw_start = resume_warmup_state["start"]
        rw_len = resume_warmup_state["length"]
        if rw_len > 0 and rw_start >= 0 and rw_start <= step < rw_start + rw_len:
            return (step - rw_start) / rw_len
        if args.lr_warmup_steps > 0 and step < args.lr_warmup_steps:
            return step / args.lr_warmup_steps
        # Short sanity runs can legitimately have total_steps <= warmup_steps.
        # In that case keep a pure warmup schedule and skip the cosine phase.
        if total_steps <= args.lr_warmup_steps:
            return 1.0
        progress = (step - args.lr_warmup_steps) / (total_steps - args.lr_warmup_steps)
        # When resuming from a checkpoint trained with a different world size,
        # planned-mode setting, or epoch length, the restored global_step can be
        # beyond the newly computed total_steps.  The raw cosine formula is
        # periodic after progress > 1 and raises LR again, which is not intended
        # for a finished cosine decay.  Clamp to keep the schedule at lr_min.
        progress = min(max(float(progress), 0.0), 1.0)
        return args.lr_min / args.lr + (1 - args.lr_min / args.lr) * 0.5 * (1 + math.cos(progress * math.pi))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_list = []  # Track regular checkpoints for rotation

    start_epoch = 0
    global_step = 0
    warned_patch_provider_fallback = False

    # Resume from checkpoint
    if args.resume:
        if local_rank == 0:
            print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', start_epoch * optimizer_steps_per_epoch)
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            # Advance scheduler to match current step for older checkpoints.
            for _ in range(global_step):
                scheduler.step()
        # Optimizer state carries the LR stored in the checkpoint.  If the run
        # is resumed with a different total_steps geometry, align it immediately
        # to the current scheduler so the first post-resume optimizer step does
        # not use a stale or already-rebounded LR.
        # If --resume-warmup-steps is set, anchor a fresh-start-style linear
        # warmup at the resumed global_step. lr_lambda will return 0 at
        # global_step and ramp to 1.0 after `resume_warmup_steps` optimizer
        # steps, then hand control back to the cosine schedule.
        if args.resume_warmup_steps > 0:
            resume_warmup_state["start"] = global_step
            resume_warmup_state["length"] = args.resume_warmup_steps
            if local_rank == 0:
                print(
                    f"[LR Scheduler] Resume warmup armed: linear 0 -> {args.lr:g} "
                    f"over {args.resume_warmup_steps} optimizer steps starting at "
                    f"global_step={global_step}.",
                    flush=True,
                )
        scheduler.last_epoch = global_step
        resumed_lr_factor = float(lr_lambda(global_step))
        for param_group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
            param_group['lr'] = float(base_lr) * resumed_lr_factor
        scheduler._last_lr = [group['lr'] for group in optimizer.param_groups]
        if global_step > total_steps and local_rank == 0:
            print(
                "[LR Scheduler] WARNING: resumed global_step "
                f"{global_step} exceeds configured total_steps {total_steps}. "
                "Cosine progress will be clamped at lr_min; if you intended a "
                "new fine-tuning schedule, start with a fresh scheduler state "
                "or increase --epochs.",
                flush=True,
            )
        if local_rank == 0:
            print(f"Resumed at epoch {start_epoch}, global_step {global_step}")

    if local_rank == 0:
        print(f"Starting training for {args.epochs} epochs")
        effective_batch = args.batch_size * args.grad_accum * (world_size if distributed else 1)
        print(f"Distributed: {distributed}, world_size: {world_size}")
        if distributed:
            print(
                f"DDP settings: timeout={args.dist_timeout_minutes}min, "
                f"broadcast_buffers={args.broadcast_buffers}"
            )
        print(f"Grad accum steps: {args.grad_accum}, effective batch size: {effective_batch}")
        print(f"Steps per epoch: {len(train_loader)}, samples per epoch: {len(train_dataset)}")
        print(f"Optimizer steps per epoch: {optimizer_steps_per_epoch}")
        print(
            "LR multipliers: "
            f"encoder={args.encoder_lr_mult:g}, decoder={args.decoder_lr_mult:g}"
        )
        if len(optimizer.param_groups) > 2:
            group_summary = [
                f"lr={group['lr']:.2e}, wd={group['weight_decay']}, params={len(group['params'])}"
                for group in optimizer.param_groups
            ]
            print("Optimizer param groups: " + " | ".join(group_summary))
        if args.loss_conf_warmup_steps > 0:
            print(
                "Confidence warmup: "
                f"0 -> {args.loss_w_conf:g} over {args.loss_conf_warmup_steps} optimizer steps"
            )
    profile_data_wait = os.getenv("D4RT_PROFILE_DATA_WAIT", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    data_wait_threshold_s = float(os.getenv("D4RT_DATA_WAIT_THRESHOLD_S", "2.0"))
    data_wait_detail = os.getenv("D4RT_DATA_WAIT_DETAIL", "1").strip().lower() in {
        "1", "true", "yes", "on",
    }
    data_wait_compare_fwd = os.getenv("D4RT_DATA_WAIT_COMPARE_FWD", "1").strip().lower() in {
        "1", "true", "yes", "on",
    }
    data_wait_detail_max_samples = int(os.getenv("D4RT_DATA_WAIT_DETAIL_MAX_SAMPLES", "8"))
    # Auto-print slow data diagnostics when data time exceeds this threshold
    slow_data_threshold_s = float(os.getenv("D4RT_SLOW_DATA_THRESHOLD_S", "3.0"))
    train_watchdog = TrainPhaseWatchdog(local_rank)
    sync_step_timing = os.getenv("D4RT_SYNC_STEP_TIMING", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    planned_warmup_samples = int(os.getenv("D4RT_PLANNED_STARTUP_WARMUP_SAMPLES", "0"))
    planned_warmup_timeout_s = float(os.getenv("D4RT_PLANNED_STARTUP_WARMUP_TIMEOUT_S", "60.0"))
    for epoch in range(start_epoch, args.epochs):
        if distributed and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        # Heartbeat the watchdog through the (potentially slow) epoch
        # transition: tearing down + re-forking builder processes can take
        # minutes under NFS contention, and without these updates the phase
        # label stays frozen at "save_checkpoint" and the elapsed timer
        # accumulates the whole rebuild into one apparent stall -> false kill.
        def _epoch_transition_progress(step, detail=""):
            train_watchdog.update(
                epoch, 0, f"epoch_transition:{step}", extra=detail
            )
        train_watchdog.update(epoch, 0, "epoch_transition:begin")
        set_dataset_epoch(train_dataset, epoch, progress_cb=_epoch_transition_progress)
        if args.planned_mode:
            planned_startup_warmup(
                train_dataset,
                target_samples=planned_warmup_samples,
                timeout_s=planned_warmup_timeout_s,
                local_rank=local_rank,
                epoch=epoch,
            )
        model.train()
        epoch_loss = 0
        optimizer.zero_grad(set_to_none=True)
        train_iter = train_loader
        if args.batch_prefetch_depth > 0:
            train_iter = BatchPrefetchIterator(
                train_loader,
                depth=args.batch_prefetch_depth,
            )
        _startup_sleep_s = float(os.environ.get("D4RT_EPOCH_STARTUP_SLEEP_S", "0"))
        if _startup_sleep_s > 0:
            if local_rank == 0:
                print(
                    f"[TrainLoop] epoch={epoch} startup sleep {_startup_sleep_s:.0f}s "
                    f"to allow builder prefetch",
                    flush=True,
                )
            time.sleep(_startup_sleep_s)
        t_data_start = time.perf_counter()
        train_watchdog.update(epoch, 0, "data_wait")
        for batch_idx, batch in enumerate(train_iter):
            t_data_end = time.perf_counter()
            t_data = t_data_end - t_data_start
            prefetch_stats_for_batch = (
                train_iter.last_stats()
                if isinstance(train_iter, BatchPrefetchIterator)
                else {}
            )
            next_planned_index_for_batch = infer_next_planned_index(batch)

            train_watchdog.update(
                epoch,
                batch_idx,
                "move_batch_to_device",
                extra=f"data_ms={t_data * 1000:.0f}",
            )
            batch = move_batch_to_device(batch, device)
            warned_patch_provider_fallback = maybe_fallback_patch_provider(
                model, batch, args.patch_provider, local_rank, warned_patch_provider_fallback
            )

            query_frames_arg = batch.get('highres_video') if args.patch_provider == 'sampled_highres' else None
            if local_rank == 0 and epoch == start_epoch and batch_idx == 0:
                info = unwrap_model(model).decoder.get_patch_provider_info(
                    batch.get('local_patches'),
                    batch.get('transform_metadata'),
                )
                print("\n[Patch Provider Info]", flush=True)
                print(f"  Configured: {info['configured']}", flush=True)
                print(f"  Resolved: {info['resolved']}", flush=True)
                print(f"  Has local_patches: {info['has_local_patches']}", flush=True)
                print(f"  Has transform_metadata: {info['has_transform_metadata']}", flush=True)
                print(f"  Has highres_video: {batch.get('highres_video') is not None}", flush=True)

            is_last_accum = (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader)

            t_fwd_start = time.perf_counter()
            if args.loss_conf_warmup_steps > 0:
                confidence_ramp = min(global_step / max(1, args.loss_conf_warmup_steps), 1.0)
            else:
                confidence_ramp = 1.0
            effective_loss_w_conf = args.loss_w_conf * confidence_ramp
            loss_fn.set_confidence_schedule(
                effective_loss_w_conf,
                weighting_factor=confidence_ramp,
            )
            # 梯度累积：只在最后一步才同步梯度，减少NCCL通信频次
            use_no_sync = distributed and hasattr(model, "no_sync") and not is_last_accum
            ctx = model.no_sync() if use_no_sync else contextlib.nullcontext()
            with ctx:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                    t_model_start = time.perf_counter()
                    train_watchdog.update(epoch, batch_idx, "model_forward")
                    outputs = model(batch['video'], batch['coords'], batch['t_src'], batch['t_tgt'], batch['t_cam'],
                                    aspect_ratio=batch.get('aspect_ratio'),
                                    local_patches=batch.get('local_patches'),
                                    transform_metadata=batch.get('transform_metadata'),
                                    query_frames=query_frames_arg,)
                    if sync_step_timing:
                        torch.cuda.synchronize()
                    t_model = time.perf_counter() - t_model_start

                    # Per-(dataset, frame) depth normalization:
                    # dataset_id * num_frames + t_cam gives each (dataset, frame) pair a unique
                    # group id, so median depth is computed independently per frame within each
                    # dataset. This is correct for both single- and multi-dataset batches.
                    t_loss_start = time.perf_counter()
                    normalize_groups = batch['dataset_id'] * args.num_frames + batch['t_cam']
                    # Build per-sample is_metric mask from dataset_names. Cheap: B≤~16.
                    if args.dust3r_metric_branch:
                        is_metric_t = torch.tensor(
                            [name in metric_dataset_set for name in batch['dataset_names']],
                            dtype=torch.bool,
                            device=normalize_groups.device,
                        )
                    else:
                        is_metric_t = None
                    train_watchdog.update(epoch, batch_idx, "loss")
                    loss_dict = loss_fn(outputs, batch['targets'],
                                        normalize_groups=normalize_groups,
                                        is_metric=is_metric_t)
                    loss = loss_dict['loss'] / args.grad_accum
                    if sync_step_timing:
                        torch.cuda.synchronize()
                    t_loss = time.perf_counter() - t_loss_start

                t_bwd_start = time.perf_counter()
                train_watchdog.update(epoch, batch_idx, "backward")
                loss.backward()
                if sync_step_timing:
                    torch.cuda.synchronize()
                t_bwd = time.perf_counter() - t_bwd_start

            t_opt_start = time.perf_counter()
            train_watchdog.update(epoch, batch_idx, "optimizer")
            if is_last_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            torch.cuda.synchronize()
            t_opt = time.perf_counter() - t_opt_start

            t_fwd = time.perf_counter() - t_fwd_start
            reasons = []
            if profile_data_wait:
                if data_wait_threshold_s > 0 and t_data >= data_wait_threshold_s:
                    reasons.append(f"threshold>={data_wait_threshold_s:.3f}s")
                if data_wait_compare_fwd and t_data > t_fwd:
                    reasons.append("data_gt_fwd_bwd")
                if reasons:
                    spool_stats = summarize_spool_ready(
                        train_dataset, focus_index=next_planned_index_for_batch
                    )
                    print(
                        f"[DataWaitDetail rank{local_rank}] "
                        f"reason={','.join(reasons)} "
                        f"epoch={epoch} batch={batch_idx} "
                        f"data={t_data * 1000:.0f}ms | "
                        f"model={t_model*1000:.0f}ms loss={t_loss*1000:.0f}ms bwd={t_bwd*1000:.0f}ms opt={t_opt*1000:.0f}ms | "
                        f"total={t_fwd * 1000:.0f}ms "
                        f"datasets={format_dataset_counts(batch.get('dataset_names'))} "
                        f"batch_prefetch_wait={prefetch_stats_for_batch.get('get_wait_s', 0.0) * 1000:.0f}ms "
                        f"q_before={prefetch_stats_for_batch.get('qsize_before')} "
                        f"q_after={prefetch_stats_for_batch.get('qsize_after')} "
                        f"q_depth={prefetch_stats_for_batch.get('depth')} "
                        f"spool_ready={spool_stats.get('ready_count')} "
                        f"spool_min={spool_stats.get('ready_min')} "
                        f"spool_max={spool_stats.get('ready_max')} "
                        f"focus_next={spool_stats.get('focus_index')} "
                        f"contiguous_from_next={spool_stats.get('contiguous_from_focus')} "
                        f"next_ready={spool_stats.get('next_ready')} "
                        f"ready_near={spool_stats.get('ready_near')}",
                        flush=True,
                    )
                    if data_wait_detail:
                        for detail_line in format_batch_sample_details(
                            batch, max_samples=data_wait_detail_max_samples
                        ):
                            print(f"[DataWaitDetail rank{local_rank}] {detail_line}", flush=True)

            # Auto-print slow data diagnostics only for positive thresholds.
            is_slow_data = slow_data_threshold_s > 0 and t_data >= slow_data_threshold_s
            if is_slow_data and not reasons:
                spool_stats = summarize_spool_ready(
                    train_dataset, focus_index=next_planned_index_for_batch
                )
                dataset_counts = format_dataset_counts(batch.get('dataset_names'))
                print(
                    f"[SlowData rank{local_rank}] "
                    f"epoch={epoch} batch={batch_idx} "
                    f"data={t_data * 1000:.0f}ms "
                    f"datasets={dataset_counts} "
                    f"spool_ready={spool_stats.get('ready_count')} "
                    f"spool_min={spool_stats.get('ready_min')} "
                    f"spool_max={spool_stats.get('ready_max')}",
                    flush=True,
                )
                for detail_line in format_batch_sample_details(
                    batch, max_samples=data_wait_detail_max_samples
                ):
                    print(f"[SlowData rank{local_rank}] {detail_line}", flush=True)

            real_loss = loss.item() * args.grad_accum  # 还原除法，得到真实loss值
            epoch_loss += real_loss
            if batch_idx % args.log_interval == 0:
                current_lrs = [group['lr'] for group in optimizer.param_groups]
                current_lr = max(current_lrs)
                lr_info = f"LR: {current_lr:.2e} (warmup {global_step}/{args.lr_warmup_steps} → {args.lr:.2e})" \
                    if global_step < args.lr_warmup_steps else f"LR: {current_lr:.2e}"
                if min(current_lrs) != max(current_lrs):
                    lr_info += f" [min {min(current_lrs):.2e}]"
                # Print from ALL ranks so we can compare data/compute time per rank
                print(f"[{time.strftime('%H:%M:%S')}][rank{local_rank}] Epoch {epoch}, Batch {batch_idx}, "
                      f"data={t_data*1000:.0f}ms | "
                      f"model={t_model*1000:.0f}ms loss={t_loss*1000:.0f}ms bwd={t_bwd*1000:.0f}ms opt={t_opt*1000:.0f}ms | "
                      f"total={t_fwd*1000:.0f}ms Loss: {real_loss:.4f}, {lr_info}", flush=True)

                if local_rank == 0:
                    # Save loss log
                    log_entry = {
                        'epoch': epoch,
                        'step': global_step,
                        'batch': batch_idx,
                        'datasets': format_dataset_counts(batch.get('dataset_names')),
                        'loss': f"{real_loss:.4f}",
                        'loss_3d': f"{loss_dict.get('loss_3d', 0):.4f}",
                        'loss_3d_nocon': f"{loss_dict.get('loss_3d_unweighted', 0):.4f}",
                        'loss_raw_3d': f"{loss_dict.get('loss_raw_3d', 0):.4f}",
                        'loss_2d': f"{loss_dict.get('loss_2d', 0):.4f}",
                        'loss_vis': f"{loss_dict.get('loss_vis', 0):.4f}",
                        'loss_disp': f"{loss_dict.get('loss_disp', 0):.4f}",
                        'loss_conf': f"{loss_dict.get('loss_conf', 0):.4f}",
                        'loss_w_conf_effective': f"{effective_loss_w_conf:.6f}",
                        'loss_conf_ramp': f"{confidence_ramp:.6f}",
                        'loss_normal': f"{loss_dict.get('loss_normal', 0):.4f}",
                        'raw_3d_l1': f"{loss_dict.get('metric_raw_3d_l1', 0):.4f}",
                        'raw_3d_euc': f"{loss_dict.get('metric_raw_3d_euclidean', 0):.4f}",
                        'raw_3d_l1_static': f"{loss_dict.get('metric_raw_3d_l1_static', 0):.4f}",
                        'raw_3d_l1_temporal': f"{loss_dict.get('metric_raw_3d_l1_temporal', 0):.4f}",
                        'raw_3d_euc_static': f"{loss_dict.get('metric_raw_3d_euclidean_static', 0):.4f}",
                        'raw_3d_euc_temporal': f"{loss_dict.get('metric_raw_3d_euclidean_temporal', 0):.4f}",
                        'loss_3d_static_nocon': f"{loss_dict.get('metric_loss_3d_static_unweighted', 0):.4f}",
                        'loss_3d_temporal_nocon': f"{loss_dict.get('metric_loss_3d_temporal_unweighted', 0):.4f}",
                        'valid_3d_ratio': f"{loss_dict.get('metric_valid_3d_query_ratio', 0):.4f}",
                        'static_query_ratio': f"{loss_dict.get('metric_static_query_ratio', 0):.4f}",
                        'temporal_query_ratio': f"{loss_dict.get('metric_temporal_query_ratio', 0):.4f}",
                        'static_valid3d_ratio': f"{loss_dict.get('metric_static_valid3d_ratio', 0):.4f}",
                        'temporal_valid3d_ratio': f"{loss_dict.get('metric_temporal_valid3d_ratio', 0):.4f}",
                        'normal_query_ratio': f"{loss_dict.get('metric_normal_query_ratio', 0):.4f}",
                        'normal_valid3d_ratio': f"{loss_dict.get('metric_normal_valid3d_ratio', 0):.4f}",
                        'conf_mean': f"{loss_dict.get('metric_conf_mean', 0):.4f}",
                        'lr': f"{current_lr:.6f}"
                    }
                    log_entry['per_dataset'] = compute_per_dataset_loss_metrics(
                        loss_fn,
                        outputs,
                        batch['targets'],
                        batch,
                        args.num_frames,
                        metric_dataset_set=metric_dataset_set,
                    )
                    with open(output_dir / 'loss_log.jsonl', 'a') as f:
                        f.write(json.dumps(log_entry) + '\n')

            t_data_start = time.perf_counter()
            train_watchdog.update(epoch, batch_idx + 1, "data_wait")
        if isinstance(train_iter, BatchPrefetchIterator):
            train_iter.close()

        avg_loss = epoch_loss / len(train_loader)
        if local_rank == 0:
            print(f"Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}")

        # Validation
        if val_loader is not None and (epoch + 1) % args.val_interval == 0:
            train_watchdog.update(epoch, len(train_loader), "validation")
            model.eval()
            val_metrics = {'loss': 0, 'loss_3d': 0, 'loss_3d_nocon': 0, 'loss_raw_3d': 0, 'loss_2d': 0, 'loss_vis': 0,
                          'loss_disp': 0, 'loss_conf': 0, 'loss_normal': 0,
                          'raw_3d_l1': 0, 'raw_3d_euc': 0,
                          'raw_3d_l1_static': 0, 'raw_3d_l1_temporal': 0,
                          'raw_3d_euc_static': 0, 'raw_3d_euc_temporal': 0,
                          'loss_3d_static_nocon': 0, 'loss_3d_temporal_nocon': 0,
                          'valid_3d_ratio': 0,
                          'static_query_ratio': 0, 'temporal_query_ratio': 0,
                          'static_valid3d_ratio': 0, 'temporal_valid3d_ratio': 0,
                          'normal_query_ratio': 0, 'normal_valid3d_ratio': 0,
                          'conf_mean': 0}
            with torch.no_grad():
                for val_batch_idx, batch in enumerate(val_loader):
                    # 每个 val batch 刷新 watchdog：验证慢但在推进时不应被误杀，
                    # 只有真正卡住(单 batch 超阈值)才触发。
                    train_watchdog.update(epoch, len(train_loader), "validation")
                    batch = move_batch_to_device(batch, device)
                    warned_patch_provider_fallback = maybe_fallback_patch_provider(
                        model, batch, args.patch_provider, local_rank, warned_patch_provider_fallback
                    )
                    query_frames_arg = batch.get('highres_video') if args.patch_provider == 'sampled_highres' else None
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                        outputs = val_model(batch['video'], batch['coords'], batch['t_src'], batch['t_tgt'], batch['t_cam'],
                                       aspect_ratio=batch.get('aspect_ratio'),
                                       local_patches=batch.get('local_patches'),
                                       transform_metadata=batch.get('transform_metadata'),
                                       query_frames=query_frames_arg,)
                        if args.dust3r_metric_branch:
                            val_is_metric_t = torch.tensor(
                                [name in metric_dataset_set for name in batch['dataset_names']],
                                dtype=torch.bool,
                                device=batch['dataset_id'].device,
                            )
                        else:
                            val_is_metric_t = None
                        loss_dict = loss_fn(outputs, batch['targets'],
                                           normalize_groups=batch['dataset_id'] * args.num_frames + batch['t_cam'],
                                           is_metric=val_is_metric_t)
                    val_metrics['loss'] += loss_dict['loss'].item()
                    val_metrics['loss_3d'] += loss_dict.get('loss_3d', 0).item()
                    val_metrics['loss_3d_nocon'] += loss_dict.get('loss_3d_unweighted', 0).item()
                    val_metrics['loss_raw_3d'] += loss_dict.get('loss_raw_3d', 0).item()
                    val_metrics['loss_2d'] += loss_dict.get('loss_2d', 0).item()
                    val_metrics['loss_vis'] += loss_dict.get('loss_vis', 0).item()
                    val_metrics['loss_disp'] += loss_dict.get('loss_disp', 0).item()
                    val_metrics['loss_conf'] += loss_dict.get('loss_conf', 0).item()
                    val_metrics['loss_normal'] += loss_dict.get('loss_normal', 0).item()
                    val_metrics['raw_3d_l1'] += loss_dict.get('metric_raw_3d_l1', 0).item()
                    val_metrics['raw_3d_euc'] += loss_dict.get('metric_raw_3d_euclidean', 0).item()
                    val_metrics['raw_3d_l1_static'] += loss_dict.get('metric_raw_3d_l1_static', 0).item()
                    val_metrics['raw_3d_l1_temporal'] += loss_dict.get('metric_raw_3d_l1_temporal', 0).item()
                    val_metrics['raw_3d_euc_static'] += loss_dict.get('metric_raw_3d_euclidean_static', 0).item()
                    val_metrics['raw_3d_euc_temporal'] += loss_dict.get('metric_raw_3d_euclidean_temporal', 0).item()
                    val_metrics['loss_3d_static_nocon'] += loss_dict.get('metric_loss_3d_static_unweighted', 0).item()
                    val_metrics['loss_3d_temporal_nocon'] += loss_dict.get('metric_loss_3d_temporal_unweighted', 0).item()
                    val_metrics['valid_3d_ratio'] += loss_dict.get('metric_valid_3d_query_ratio', 0).item()
                    val_metrics['static_query_ratio'] += loss_dict.get('metric_static_query_ratio', 0).item()
                    val_metrics['temporal_query_ratio'] += loss_dict.get('metric_temporal_query_ratio', 0).item()
                    val_metrics['static_valid3d_ratio'] += loss_dict.get('metric_static_valid3d_ratio', 0).item()
                    val_metrics['temporal_valid3d_ratio'] += loss_dict.get('metric_temporal_valid3d_ratio', 0).item()
                    val_metrics['normal_query_ratio'] += loss_dict.get('metric_normal_query_ratio', 0).item()
                    val_metrics['normal_valid3d_ratio'] += loss_dict.get('metric_normal_valid3d_ratio', 0).item()
                    val_metrics['conf_mean'] += loss_dict.get('metric_conf_mean', 0).item()
            for k in val_metrics:
                val_metrics[k] /= len(val_loader)
                if distributed:
                    val_metrics[k] = torch.tensor(val_metrics[k], device=device)
                    dist.all_reduce(val_metrics[k], op=dist.ReduceOp.AVG)
                    val_metrics[k] = val_metrics[k].item()
            if distributed:
                dist.barrier()
            if local_rank == 0:
                print(f"Validation Loss: {val_metrics['loss']:.4f}")
                val_log = {'epoch': epoch + 1}
                val_log.update({k: f"{v:.4f}" for k, v in val_metrics.items()})
                with open(output_dir / 'val_log.jsonl', 'a') as f:
                    f.write(json.dumps(val_log) + '\n')

        # Save checkpoint
        # 所有 rank 在进入存盘阶段时都刷新 watchdog（不只 rank0）：非 rank0 会在
        # 下个 epoch 的首个 collective 处阻塞等 rank0 存完 ~14GB ckpt，若不刷新，
        # 它们的计时会从 validation 一路累加到"验证+等存盘"而被误杀。
        if (epoch + 1) % args.save_interval == 0:
            train_watchdog.update(epoch, len(train_loader), "save_checkpoint")
        if (epoch + 1) % args.save_interval == 0 and local_rank == 0:
            is_milestone = (epoch + 1) % 1000 == 0
            if is_milestone:
                checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            else:
                checkpoint_path = output_dir / f"checkpoint_latest_{epoch+1}.pth"
                checkpoint_list.append(checkpoint_path)
                # Remove old checkpoints if exceeds limit
                if len(checkpoint_list) > args.keep_checkpoints:
                    old_ckpt = checkpoint_list.pop(0)
                    if old_ckpt.exists():
                        old_ckpt.unlink()

            torch.save({
                'model': unwrap_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    train_watchdog.stop()
    cleanup_dataset(train_dataset)


    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
