"""
Sample builder for background QuerySample construction.

This module implements background worker processes that build QuerySample
bundles from SampleSpec objects, enabling prefetch from COS.

Key design decisions:
- Uses forkserver context to avoid inheriting CUDA/NCCL state from parent.
- Checks spec.generation against a shared counter so stale specs from
  previous epochs are silently discarded instead of built and spooled.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import random as _random_module
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from .adapters.base import BaseAdapter
from .planning import SampleSpec
from .query_builder import D4RTQueryBuilder, QuerySample
from .sampling import _is_sorted
from .sample_stage import build_sample_stager
from .sample_spool import SampleSpool
from .transforms import GeometryTransformPipeline


def _set_parent_death_signal() -> None:
    """Ask Linux to terminate this builder if the rank process dies."""
    if os.name != "posix":
        return
    try:
        import ctypes
        import signal

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        pr_set_pdeathsig = 1
        rc = libc.prctl(pr_set_pdeathsig, signal.SIGTERM, 0, 0, 0)
        if rc != 0:
            return
        if os.getppid() == 1:
            os._exit(0)
    except Exception:
        pass


def _limit_cpu_threads() -> None:
    """Keep builder processes single-threaded to avoid CPU oversubscription."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    try:
        import cv2
        cv2.setNumThreads(1)
        if hasattr(cv2, "ocl"):
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


class SampleBuilder:
    """Background process that builds QuerySample bundles from SampleSpecs.

    Each builder runs in a separate process and:
    1. Reads SampleSpec from input queue
    2. Checks generation -- discards stale specs
    3. Calls adapter.load_clip() -> transform() -> query_builder()
    4. Writes QuerySample to spool directory
    5. Signals completion via output queue

    Error handling:
    - Transient I/O errors: retry same spec (up to 3 attempts)
    - Structural errors: write error marker, fail fast
    """

    def __init__(
        self,
        builder_id: int,
        adapters: list[BaseAdapter],
        transform: GeometryTransformPipeline,
        query_builder: D4RTQueryBuilder,
        spool: SampleSpool,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        clip_len: int,
        current_generation: "mp.Value",
        sample_stage_config: Optional[dict[str, Any]] = None,
    ):
        """
        Args:
            builder_id: Unique ID for this builder process
            adapters: List of dataset adapters (one per dataset in mixture)
            transform: Transform pipeline
            query_builder: Query builder
            spool: Sample spool for writing bundles
            input_queue: Queue for receiving SampleSpec objects
            output_queue: Queue for signaling completion
            clip_len: Expected clip length
            current_generation: Shared integer updated by the main process on
                epoch transitions.  Specs whose generation is older than this
                value are silently discarded.
        """
        self.builder_id = builder_id
        self.adapters = adapters
        self.transform = transform
        self.query_builder = query_builder
        self.spool = spool
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.clip_len = clip_len
        self.current_generation = current_generation
        self.sample_stage_config = sample_stage_config
        self._sample_stager = None
        self._last_profile: dict[str, Any] = {}
        self._profile_builder = os.getenv("D4RT_PROFILE_BUILDER", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        self._profile_builder_all = os.getenv("D4RT_PROFILE_BUILDER_ALL", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        self._profile_builder_threshold_s = float(
            os.getenv("D4RT_PROFILE_BUILDER_THRESHOLD_S", "5.0")
        )
        self._verbose_builder = os.getenv("D4RT_VERBOSE_BUILDER", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        self._enable_faulthandler = os.getenv("D4RT_BUILDER_FAULTHANDLER", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        self._rolling_warm_ready_dir = os.getenv(
            "D4RT_ROLLING_WARM_READY_DIR", ""
        ).strip()
        self._rolling_warm_block_batches = max(
            1,
            int(os.getenv("D4RT_ROLLING_WARM_BLOCK_BATCHES", "10") or "10"),
        )
        self._rolling_warm_batch_size = max(
            1,
            int(
                os.getenv(
                    "D4RT_ROLLING_WARM_BATCH_SIZE",
                    os.getenv("D4RT_PLANNED_BATCH_SIZE", "1"),
                )
                or "1"
            ),
        )
        self._rolling_warm_timeout_s = float(
            os.getenv("D4RT_ROLLING_WARM_TIMEOUT_S", "0") or "0"
        )
        self._rolling_warm_log = os.getenv(
            "D4RT_ROLLING_WARM_LOG", ""
        ).strip().lower() in {"1", "true", "yes", "on"}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main loop for builder process."""
        _set_parent_death_signal()
        _limit_cpu_threads()
        enable_faulthandler = getattr(self, "_enable_faulthandler", False)
        verbose_builder = getattr(self, "_verbose_builder", False)
        if enable_faulthandler:
            import faulthandler, signal, sys
            faulthandler.enable(file=sys.stderr, all_threads=True)
            # Do not register SIGTERM: multiprocessing.terminate() uses it for
            # normal cleanup, and dumping a stack on expected shutdown is noisy.
            for sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGFPE):
                try:
                    faulthandler.register(sig, file=sys.stderr, chain=True)
                except Exception:
                    pass
        if verbose_builder:
            print(f"[Builder {self.builder_id}] STARTED pid={os.getpid()}", flush=True)
        try:
            while True:
                try:
                    # Get next spec from queue (blocking with short timeout so we
                    # can re-check for shutdown even if the queue is empty).
                    task = self.input_queue.get(timeout=1.0)

                    if task is None:
                        # Shutdown signal
                        if verbose_builder:
                            print(f"[Builder {self.builder_id}] received None, EXITING normally", flush=True)
                        break

                    specs = task if isinstance(task, list) else [task]
                    for spec in specs:
                        if spec is None:
                            continue

                        # --- generation gate ---
                        if spec.generation < self.current_generation.value:
                            # Stale spec from a previous epoch; drop it.
                            continue

                        # Build sample with retry logic
                        success = self._build_sample_with_retry(spec)

                        # Signal completion (use local_index so the coordinator knows
                        # which spool slot was written).
                        self.output_queue.put((spec.local_index, success))

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[Builder {self.builder_id}] Unexpected error: {e}", flush=True)
                    traceback.print_exc()
        except BaseException as e:
            print(f"[Builder {self.builder_id}] FATAL EXIT: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            raise
        finally:
            if verbose_builder:
                print(f"[Builder {self.builder_id}] EXITED pid={os.getpid()}", flush=True)

    # ------------------------------------------------------------------
    # Build with retry
    # ------------------------------------------------------------------

    def _build_sample_with_retry(self, spec: SampleSpec, max_attempts: int = 3) -> bool:
        """Build a sample with retry logic for transient errors.

        Args:
            spec: SampleSpec to build
            max_attempts: Maximum number of retry attempts

        Returns:
            True if successful, False if failed
        """
        build_timeout = float(os.environ.get("D4RT_BUILD_TIMEOUT", "120"))
        for attempt in range(max_attempts):
            if attempt == 0:
                self._wait_for_rolling_warm(spec)
            attempt_start = time.perf_counter()
            try:
                if build_timeout > 0:
                    result_box: list = []
                    exc_box: list = []
                    def _run():
                        try:
                            result_box.append(self._build_sample(spec))
                        except Exception as e:
                            exc_box.append(e)
                    t = threading.Thread(target=_run, daemon=True)
                    t.start()
                    t.join(timeout=build_timeout)
                    if t.is_alive():
                        timeout_grace = float(
                            os.environ.get("D4RT_BUILD_TIMEOUT_GRACE", "70")
                        )
                        if timeout_grace > 0:
                            t.join(timeout=timeout_grace)
                        if not t.is_alive():
                            if exc_box:
                                raise exc_box[0]
                            if not result_box:
                                raise RuntimeError(
                                    f"_build_sample thread finished without result "
                                    f"(spec={spec.local_index}, dataset_idx={spec.dataset_idx})"
                                )
                            sample = result_box[0]
                        else:
                            adapter = self.adapters[spec.dataset_idx]
                            dataset_name = getattr(
                                adapter, "dataset_name", type(adapter).__name__
                            )
                            self._reset_adapter_after_timeout(
                                adapter, spec.sequence_name
                            )

                            # Get timing info if available (from partial execution)
                            timing_info = ""
                            if hasattr(self, '_last_load_timing') and self._last_load_timing:
                                timing = self._last_load_timing
                                timing_info = (
                                    f" | partial_timing: "
                                    f"scene_data={timing.get('scene_data_s', 0)*1000:.0f}ms "
                                    f"rgb_load={timing.get('rgb_load_s', 0)*1000:.0f}ms "
                                    f"depth_load={timing.get('depth_load_s', 0)*1000:.0f}ms "
                                    f"precomputed={timing.get('precomputed_s', 0)*1000:.0f}ms"
                                )

                            err = TimeoutError(
                                f"_build_sample timed out after {build_timeout}s "
                                f"(spec={spec.local_index}, dataset_idx={spec.dataset_idx}, "
                                f"dataset={dataset_name}, seq={spec.sequence_name}, "
                                f"frames={len(spec.frame_indices)}){timing_info}"
                            )
                            total_s = time.perf_counter() - attempt_start

                            # Print detailed timeout info
                            print(
                                f"[Timeout] dataset={dataset_name} seq={spec.sequence_name} "
                                f"frames={len(spec.frame_indices)} timeout={build_timeout}s "
                                f"total_elapsed={total_s:.1f}s{timing_info}",
                                flush=True,
                            )

                            self._maybe_print_profile(spec=spec, attempt=attempt, success=False,
                                                       total_s=total_s, error=err)
                            # Don't retry on timeout — stale threads exhaust COS connections
                            self.spool.write_error(spec.local_index, err, spec.generation)
                            return False
                    else:
                        if exc_box:
                            raise exc_box[0]
                        sample = result_box[0]
                else:
                    sample = self._build_sample(spec)

                # Validate frame count
                if sample.video.shape[0] != self.clip_len:
                    raise RuntimeError(
                        f"Sample has {sample.video.shape[0]} frames, "
                        f"expected {self.clip_len}"
                    )

                # Wait for disk headroom, then write
                t_wait0 = time.perf_counter()
                self.spool.wait_for_space()
                t_wait_space = time.perf_counter() - t_wait0
                t_write0 = time.perf_counter()
                self.spool.write_bundle(
                    spec.local_index, sample, spec.generation
                )
                t_write = time.perf_counter() - t_write0
                total_s = time.perf_counter() - attempt_start

                # Print detailed timing for slow samples (>5s) or timeout cases
                slow_threshold = float(os.environ.get("D4RT_SLOW_SAMPLE_THRESHOLD_S", "5.0"))
                if total_s > slow_threshold and hasattr(self, '_last_load_timing') and self._last_load_timing:
                    timing = self._last_load_timing
                    adapter = self.adapters[spec.dataset_idx]
                    dataset_name = getattr(adapter, "dataset_name", type(adapter).__name__)
                    extra_parts = []
                    for key, label in (
                        ("traj_load_s", "traj"),
                        ("stack_s", "stack"),
                        ("query_prepare_s", "query_prep"),
                    ):
                        value = timing.get(key)
                        if value is not None:
                            extra_parts.append(f"{label}={value * 1000:.0f}ms")
                    extra_timing = ""
                    if extra_parts:
                        extra_timing = " " + " ".join(extra_parts)
                    print(
                        f"[SlowSample] dataset={dataset_name} seq={spec.sequence_name} "
                        f"frames={len(spec.frame_indices)} total_load={timing.get('total_s', 0)*1000:.0f}ms | "
                        f"scene_data={timing.get('scene_data_s', 0)*1000:.0f}ms "
                        f"rgb_load={timing.get('rgb_load_s', 0)*1000:.0f}ms "
                        f"rgb_resize={timing.get('rgb_resize_s', 0)*1000:.0f}ms "
                        f"depth_load={timing.get('depth_load_s', 0)*1000:.0f}ms "
                        f"depth_resize={timing.get('depth_resize_s', 0)*1000:.0f}ms "
                        f"precomputed={timing.get('precomputed_s', 0)*1000:.0f}ms "
                        f"process={timing.get('process_s', 0)*1000:.0f}ms{extra_timing} | "
                        f"stage={self._last_profile.get('stage_s', 0)*1000:.0f}ms "
                        f"transform={self._last_profile.get('transform_s', 0)*1000:.0f}ms "
                        f"query={self._last_profile.get('query_s', 0)*1000:.0f}ms | "
                        f"total_build={total_s*1000:.0f}ms",
                        flush=True,
                    )

                self._maybe_print_profile(
                    spec=spec,
                    attempt=attempt,
                    success=True,
                    total_s=total_s,
                    wait_space_s=t_wait_space,
                    write_s=t_write,
                )
                return True

            except Exception as e:
                total_s = time.perf_counter() - attempt_start
                self._maybe_print_profile(
                    spec=spec,
                    attempt=attempt,
                    success=False,
                    total_s=total_s,
                    error=e,
                )
                if attempt < max_attempts - 1:
                    print(
                        f"[Builder {self.builder_id}] Retry {attempt + 1}/{max_attempts} "
                        f"for sample {spec.local_index}: {e}"
                    )
                else:
                    print(
                        f"[Builder {self.builder_id}] Failed sample {spec.local_index} "
                        f"after {max_attempts} attempts: {e}"
                    )
                    self.spool.write_error(
                        spec.local_index, e, spec.generation
                    )
                    return False

        return False  # unreachable, but keeps mypy happy

    def _wait_for_rolling_warm(self, spec: SampleSpec) -> None:
        ready_root_raw = getattr(self, "_rolling_warm_ready_dir", "")
        if not ready_root_raw:
            return
        batch_size = max(1, int(getattr(self, "_rolling_warm_batch_size", 1)))
        block_batches = max(1, int(getattr(self, "_rolling_warm_block_batches", 1)))
        local_batch = int(spec.local_index) // batch_size
        block_id = local_batch // block_batches
        ready_path = (
            Path(ready_root_raw)
            / f"g{int(spec.generation):04d}"
            / f"block_{block_id:08d}.ready"
        )
        timeout_s = float(getattr(self, "_rolling_warm_timeout_s", 0.0))
        deadline = None if timeout_s <= 0 else time.time() + timeout_s
        logged = False
        while not ready_path.is_file():
            if deadline is not None and time.time() >= deadline:
                raise TimeoutError(
                    "rolling warm block did not become ready "
                    f"for spec={spec.local_index} generation={spec.generation} "
                    f"local_batch={local_batch} block={block_id} path={ready_path}"
                )
            if getattr(self, "_rolling_warm_log", False) and not logged:
                print(
                    f"[RollingWarmWait b{self.builder_id}] "
                    f"g{spec.generation:04d} local_index={spec.local_index} "
                    f"batch={local_batch} block={block_id} path={ready_path}",
                    flush=True,
                )
                logged = True
            time.sleep(0.1)

    def _reset_adapter_after_timeout(self, adapter: BaseAdapter, sequence_name: str) -> None:
        dataset_name = str(getattr(adapter, "dataset_name", ""))
        if dataset_name == "scannetpp" and hasattr(adapter, "root"):
            try:
                adapter.data_root = Path(adapter.root)
            except Exception:
                pass

        if hasattr(adapter, "_scene_cache"):
            try:
                adapter._scene_cache.pop(sequence_name, None)
                if dataset_name == "scannetpp" and hasattr(adapter, "root"):
                    root = Path(adapter.root)
                    stale_keys: list[str] = []
                    for key, scene_data in list(adapter._scene_cache.items()):
                        if not isinstance(scene_data, dict):
                            continue
                        scene_dir = scene_data.get("scene_dir")
                        if scene_dir is None:
                            continue
                        try:
                            Path(scene_dir).relative_to(root)
                        except ValueError:
                            stale_keys.append(key)
                    for key in stale_keys:
                        adapter._scene_cache.pop(key, None)
            except Exception:
                pass

        if hasattr(adapter, "_depth_chunk_cache"):
            try:
                adapter._depth_chunk_cache.pop(sequence_name, None)
            except Exception:
                pass

        if dataset_name == "scannetpp":
            adapter.__dict__.pop("_staged_depth_map_tmp", None)
            adapter.__dict__.pop("_staged_depth_chunks_tmp", None)

    # ------------------------------------------------------------------
    # Core build logic
    # ------------------------------------------------------------------

    def _build_sample(self, spec: SampleSpec) -> QuerySample:
        """Build a single QuerySample from a SampleSpec.

        Args:
            spec: SampleSpec with dataset_idx, sequence_name, frame_indices,
                  rng_state, and global_index (used as RNG seed).

        Returns:
            QuerySample ready for training
        """
        adapter = self.adapters[spec.dataset_idx]
        self._last_profile = {
            "dataset": getattr(adapter, "dataset_name", type(adapter).__name__),
            "sequence": spec.sequence_name,
            "frame_min": min(spec.frame_indices) if spec.frame_indices else None,
            "frame_max": max(spec.frame_indices) if spec.frame_indices else None,
            "frame_count": len(spec.frame_indices),
            "stage_s": 0.0,
            "load_s": 0.0,
            "transform_s": 0.0,
            "query_s": 0.0,
        }

        sample_stager = self._get_sample_stager()
        sample_tag = f"b{self.builder_id}_g{spec.generation}_i{spec.local_index}"
        stage_ctx = (
            sample_stager.stage_sample(
                adapter,
                spec.sequence_name,
                spec.frame_indices,
                sample_tag=sample_tag,
            )
            if sample_stager is not None
            else _nullcontext(adapter)
        )
        t_stage0 = time.perf_counter()
        with stage_ctx as staged_adapter:
            t_stage = time.perf_counter() - t_stage0
            self._last_profile["stage_s"] = t_stage
            t_load0 = time.perf_counter()
            clip = staged_adapter.load_clip(spec.sequence_name, spec.frame_indices)
            t_load = time.perf_counter() - t_load0
            self._last_profile["load_s"] = t_load

        # Extract detailed timing from clip metadata if available
        self._last_load_timing = clip.metadata.pop("_load_timing", None)

        # Restore RNG state (deterministic transforms + query building)
        rng = _random_module.Random()
        rng.setstate(spec.rng_state)

        # Apply geometry transforms
        t_transform0 = time.perf_counter()
        result = self.transform(clip, rng=rng)
        t_transform = time.perf_counter() - t_transform0
        self._last_profile["transform_s"] = t_transform

        # Build query sample
        t_query0 = time.perf_counter()
        # A clip whose planned frame_indices are not time-ascending was permuted
        # by shuffle_prob; signal the query builder to skip neighbour-frame motion
        # boundaries for this clip only.
        sample = self.query_builder(
            result,
            py_rng=rng,
            frames_time_ordered=_is_sorted(spec.frame_indices),
        )
        t_query = time.perf_counter() - t_query0
        self._last_profile["query_s"] = t_query
        sample.metadata.update(
            {
                "planned_generation": spec.generation,
                "planned_local_index": spec.local_index,
                "planned_global_index": spec.global_index,
                "planned_dataset_idx": spec.dataset_idx,
                "planned_frame_indices": list(spec.frame_indices),
                "planned_frame_count": len(spec.frame_indices),
                "planned_frame_min": min(spec.frame_indices) if spec.frame_indices else None,
                "planned_frame_max": max(spec.frame_indices) if spec.frame_indices else None,
                "builder_id": self.builder_id,
            }
        )

        self._last_profile.update(
            stage_s=t_stage,
            load_s=t_load,
            transform_s=t_transform,
            query_s=t_query,
        )

        return sample

    def _maybe_print_profile(
        self,
        spec: SampleSpec,
        attempt: int,
        success: bool,
        total_s: float,
        wait_space_s: float = 0.0,
        write_s: float = 0.0,
        error: Optional[Exception] = None,
    ) -> None:
        if not self._profile_builder:
            return
        if not self._profile_builder_all and total_s < self._profile_builder_threshold_s:
            return

        p = self._last_profile
        status = "ok" if success else "fail"
        msg = (
            f"[BuilderProfile b{self.builder_id}] "
            f"status={status} g{spec.generation:04d}_{spec.local_index:08d} "
            f"attempt={attempt + 1} total={total_s:.3f}s "
            f"dataset={p.get('dataset', '<unknown>')} "
            f"seq={p.get('sequence', spec.sequence_name)} "
            f"frames={p.get('frame_count', len(spec.frame_indices))}"
            f"[{p.get('frame_min', None)}..{p.get('frame_max', None)}] "
            f"stage={p.get('stage_s', 0.0):.3f}s "
            f"load={p.get('load_s', 0.0):.3f}s "
            f"transform={p.get('transform_s', 0.0):.3f}s "
            f"query={p.get('query_s', 0.0):.3f}s "
            f"wait_space={wait_space_s:.3f}s "
            f"write={write_s:.3f}s"
        )
        if error is not None:
            msg += f" error={type(error).__name__}: {error}"
        print(msg, flush=True)

    def _get_sample_stager(self):
        if self._sample_stager is None:
            self._sample_stager = build_sample_stager(self.sample_stage_config)
        return self._sample_stager


class _nullcontext:
    def __init__(self, value: Any):
        self.value = value

    def __enter__(self) -> Any:
        return self.value

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


# ======================================================================
# Process lifecycle helpers
# ======================================================================


def start_builder_process(
    builder_id: int,
    adapters: list[BaseAdapter],
    transform: GeometryTransformPipeline,
    query_builder: D4RTQueryBuilder,
    spool: SampleSpool,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    clip_len: int,
    current_generation: "mp.Value",
    sample_stage_config: Optional[dict[str, Any]] = None,
) -> mp.Process:
    """Start a builder process using the *forkserver* start method.

    Using forkserver (instead of the default fork) avoids inheriting
    CUDA/NCCL context from the parent process, which would otherwise
    cause hangs or crashes in the child.

    Args:
        builder_id: Unique ID for this builder
        adapters: List of dataset adapters
        transform: Transform pipeline
        query_builder: Query builder
        spool: Sample spool
        input_queue: Input queue for SampleSpecs
        output_queue: Output queue for completion signals
        clip_len: Expected clip length
        current_generation: Shared mp.Value(ctypes.c_int64) that the main
            process bumps on each epoch transition

    Returns:
        Started multiprocessing.Process
    """
    builder = SampleBuilder(
        builder_id=builder_id,
        adapters=adapters,
        transform=transform,
        query_builder=query_builder,
        spool=spool,
        input_queue=input_queue,
        output_queue=output_queue,
        clip_len=clip_len,
        current_generation=current_generation,
        sample_stage_config=sample_stage_config,
    )

    start_method = os.getenv("D4RT_BUILDER_START_METHOD", "forkserver").strip() or "forkserver"
    ctx = mp.get_context(start_method)
    process = ctx.Process(target=builder.run, name=f"SampleBuilder-{builder_id}")
    process.start()
    return process


def stop_builder_processes(
    processes: list[mp.Process],
    input_queue: mp.Queue | list[mp.Queue] | None,
    timeout: float = 5.0,
) -> None:
    """Stop all builder processes gracefully.

    Args:
        processes: List of builder processes
        input_queue: Input queue (for sending shutdown signals)
        timeout: Timeout for joining processes
    """
    input_queues = (
        list(input_queue)
        if isinstance(input_queue, list)
        else ([input_queue] if input_queue is not None else [])
    )

    # Send shutdown signals. The input queue may be full of prefetched specs;
    # blocking here would hang cleanup before the terminate fallback runs.
    for q in input_queues:
        try:
            q.put_nowait(None)
        except queue.Full:
            continue
        except (BrokenPipeError, EOFError, OSError, ValueError):
            continue

    # Wait for all processes as a group.  Joining with the full timeout per
    # process makes short probes spend minutes in cleanup when many builders
    # are busy in I/O.
    deadline = time.monotonic() + max(0.0, timeout)
    for p in processes:
        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0.0:
            break
        p.join(timeout=remaining)

    alive = [p for p in processes if p.is_alive()]
    if alive:
        names = ", ".join(p.name for p in alive[:8])
        suffix = "" if len(alive) <= 8 else f", ... +{len(alive) - 8}"
        print(
            f"Warning: {len(alive)} builder processes did not terminate "
            f"within {timeout:.1f}s, terminating: {names}{suffix}",
            flush=True,
        )
        for p in alive:
            try:
                p.terminate()
            except Exception:
                pass

        terminate_deadline = time.monotonic() + 1.0
        for p in alive:
            remaining = max(0.0, terminate_deadline - time.monotonic())
            if remaining <= 0.0:
                break
            p.join(timeout=remaining)

    alive = [p for p in processes if p.is_alive()]
    if alive:
        for p in alive:
            try:
                p.kill()
            except Exception:
                pass
        for p in alive:
            p.join(timeout=0.2)

    try:
        for q in input_queues:
            q.cancel_join_thread()
            q.close()
    except Exception:
        pass
