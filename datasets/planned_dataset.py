"""
Planned dataset for deterministic sample-bundle prefetch.

Design invariants
-----------------
- __getitem__(i) corresponds to plan[i]; external sampler MUST be
  SequentialSampler (enforced by train_mixture.py).
- Epoch transitions fully tear down and rebuild the prefetch pipeline
  (queues + builder processes + spool generation) so there is zero
  chance of cross-epoch pollution.
- All child processes are created via the "forkserver" mp context to
  avoid CUDA-fork issues.  Adapters / transforms / query_builder are
  pure-CPU objects and safe to pickle across the forkserver boundary.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import hashlib
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

import torch.utils.data

from .adapters.base import BaseAdapter
from .mixture import MixtureSampler
from .planning import SamplePlanner
from .query_builder import D4RTQueryBuilder, QuerySample
from .sample_builder import start_builder_process, stop_builder_processes
from .sample_spool import SampleSpool
from .transforms import GeometryTransformPipeline

logger = logging.getLogger(__name__)


def _get_mp_context() -> mp.context.BaseContext:
    method = os.getenv("D4RT_BUILDER_START_METHOD", "forkserver").strip() or "forkserver"
    return mp.get_context(method)


class PlannedMixtureDataset(torch.utils.data.Dataset):
    """Drop-in Dataset with planned prefetch.

    Epoch transitions destroy old queues / processes and create fresh
    ones rather than hot-switching, guaranteeing zero cross-epoch
    pollution.

    **epoch_size padding:** When ``world_size`` does not evenly divide
    ``epoch_size``, each rank processes ``math.ceil(epoch_size / world_size)``
    samples, so the effective global sample count is
    ``ceil(epoch_size / world_size) * world_size`` — potentially slightly more
    than ``epoch_size``.  This matches PyTorch ``DistributedSampler``'s padding
    behavior and ensures no rank runs out of work early.

    **Cleanup:** Call ``cleanup()`` or let the destructor run to stop builder
    processes.  If called mid-epoch (before all samples are consumed), builders
    may not exit gracefully within the 5-second timeout and will be terminated.
    This is safe but may log warnings.  Normal training (full epoch consumption
    before ``set_epoch()``) avoids this.
    """

    def __init__(
        self,
        adapters: list[BaseAdapter],
        dataset_weights: list[float],
        transform: GeometryTransformPipeline,
        query_builder: D4RTQueryBuilder,
        mixture_sampler: MixtureSampler,
        clip_len: int,
        seed: int,
        epoch_size: int,
        reshuffle_each_epoch: bool = True,
        builder_workers: int = 2,
        prefetch_depth: int = 32,
        spool_dir: Optional[str] = None,
        rank: int = 0,
        world_size: int = 1,
        max_spool_bytes: int = 2 * 1024**3,
        sample_stage_config: Optional[dict[str, Any]] = None,
        start_immediately: bool = True,
        initial_epoch: int = 0,
    ):
        # Store config for pipeline rebuilds.
        self.adapters = adapters
        self.dataset_weights = dataset_weights
        self.transform = transform
        self.query_builder = query_builder
        self.mixture_sampler = mixture_sampler
        self.clip_len = clip_len
        self.seed = seed
        self.epoch_size = epoch_size
        self.reshuffle_each_epoch = reshuffle_each_epoch
        self.builder_workers = builder_workers
        self.prefetch_depth = prefetch_depth
        self.rank = rank
        self.world_size = world_size
        self.max_spool_bytes = max_spool_bytes
        self.sample_stage_config = sample_stage_config
        self.start_immediately = bool(start_immediately)
        self.initial_epoch = int(initial_epoch)
        self._wait_log = os.getenv("D4RT_PLANNED_WAIT_LOG", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
        self._relaxed_order = os.getenv(
            "D4RT_PLANNED_RELAXED_ORDER", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._relaxed_lookahead = max(
            0, int(os.getenv("D4RT_PLANNED_RELAXED_LOOKAHEAD", "32"))
        )
        self._relaxed_grace_s = max(
            0.0, float(os.getenv("D4RT_PLANNED_RELAXED_GRACE_S", "0.25"))
        )
        self._relaxed_log = os.getenv(
            "D4RT_PLANNED_RELAXED_LOG", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._rolling_warm_progress_dir = os.getenv(
            "D4RT_ROLLING_WARM_PROGRESS_DIR",
            os.getenv("D4RT_ROLLING_WARM_READY_DIR", ""),
        ).strip()
        self._rolling_warm_ready_dir = os.getenv(
            "D4RT_ROLLING_WARM_READY_DIR", ""
        ).strip()
        self._rolling_warm_block_batches = max(
            1, int(os.getenv("D4RT_ROLLING_WARM_BLOCK_BATCHES", "10") or "10")
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
        self._rolling_warm_timeout_s = max(
            0.0, float(os.getenv("D4RT_ROLLING_WARM_TIMEOUT_S", "0") or "0")
        )
        self._rolling_warm_log = os.getenv(
            "D4RT_ROLLING_WARM_LOG", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._rolling_warm_last_block: Optional[int] = None
        self._rolling_warm_ready_blocks: set[int] = set()
        self._pipeline_start_index = max(
            0,
            int(os.getenv("D4RT_PLANNED_START_INDEX", "0") or "0"),
        )
        self._returned_indices: set[int] = set()
        self._requeue_counts: dict[int, int] = {}
        self._skip_ready_enqueue = os.getenv(
            "D4RT_SKIP_READY_ENQUEUE", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._preserve_spool_on_cleanup = os.getenv(
            "D4RT_PRESERVE_SPOOL_ON_CLEANUP", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._spool_cleanup_on_init = os.getenv(
            "D4RT_SPOOL_CLEANUP_ON_INIT", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        self._read_only_spool = os.getenv(
            "D4RT_PLANNED_READ_ONLY_SPOOL", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._read_only_spool_timeout_s = max(
            0.0,
            float(os.getenv("D4RT_PLANNED_READ_ONLY_SPOOL_TIMEOUT_S", "600") or "600"),
        )
        self._sequence_affinity = os.getenv(
            "D4RT_PLANNED_SEQUENCE_AFFINITY", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._block_enqueue = os.getenv(
            "D4RT_PLANNED_BLOCK_ENQUEUE", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Hot epoch-switch: keep builder processes alive across epochs and only
        # bump the shared generation + re-plan + drain queues, instead of the
        # default kill-all-64 + re-fork-64.  The re-fork path can deadlock under
        # NFS contention (a freshly forked builder wedges in init before reading
        # its spec pipe; the main process blocks writing the spec -> hang).
        # Builders are already generation-aware (sample_builder.py gates on
        # spec.generation), and the consumer reads only via the spool's
        # (index, generation) filter, so cross-epoch pollution is impossible.
        # Default ON; set =0 to fall back to the full teardown/refork.
        self._hot_epoch_switch = os.getenv(
            "D4RT_PLANNED_HOT_EPOCH_SWITCH", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._block_max_samples = max(
            1,
            int(os.getenv("D4RT_PLANNED_BLOCK_MAX_SAMPLES", "8") or "8"),
        )
        self._block_lookahead = max(
            1,
            int(os.getenv("D4RT_PLANNED_BLOCK_LOOKAHEAD", "240") or "240"),
        )
        self._pipeline_started = False
        self._enqueue_thread: Optional[threading.Thread] = None
        self._enqueue_stop = threading.Event()
        self._consumed_index: int = -1  # last index returned to caller
        self._consumed_lock = threading.Lock()

        self.current_epoch = self.initial_epoch
        self._generation: int = 0

        # Spool persists across epochs; generation tag isolates files.
        if spool_dir is None:
            spool_dir = str(
                Path(tempfile.gettempdir()) / f"d4rt_spool_rank{rank}"
            )
        self.spool = SampleSpool(
            spool_dir, rank=rank, cleanup_on_init=self._spool_cleanup_on_init,
            max_spool_bytes=max_spool_bytes,
        )
        self.spool.set_generation(self._generation)

        self.planner = SamplePlanner(
            mixture_sampler=mixture_sampler,
            seed=seed, rank=rank, world_size=world_size,
            reshuffle_each_epoch=reshuffle_each_epoch,
        )

        # Mutable pipeline state -- rebuilt every epoch.
        self.input_queue: Optional[mp.Queue] = None
        self.input_queues: list[mp.Queue] = []
        self.output_queue: Optional[mp.Queue] = None
        self.builder_processes: list[mp.Process] = []
        self._shared_generation: Optional[Any] = None
        self.current_plan: list = []
        self.next_enqueue_index: int = 0
        self._enqueued_indices: set[int] = set()

        # Initial plan + pipeline for epoch 0.
        self.current_plan = self.planner.generate_plan(
            epoch=self.current_epoch,
            count_per_rank=math.ceil(self.epoch_size / self.world_size),
            epoch_size=self.epoch_size, generation=self._generation,
        )
        if self.current_plan:
            self._write_rolling_warm_progress(0)
        if self.start_immediately:
            self._start_pipeline()

    # ------------------------------------------------------------------
    # Pipeline lifecycle (rebuilt each epoch)
    # ------------------------------------------------------------------

    def _start_pipeline(self, progress_cb=None) -> None:
        """Create fresh queues, builder processes, and seed prefetch window.

        ``progress_cb(step, detail)`` is heartbeated every few forked builders
        so an external watchdog survives a slow re-fork (e.g. under NFS
        contention from co-running jobs).  See ``set_epoch`` for rationale.
        """
        if self._read_only_spool:
            self.next_enqueue_index = min(self._pipeline_start_index, len(self.current_plan))
            self._pipeline_started = True
            return
        if self.builder_workers <= 0:
            raise ValueError(
                "builder_workers must be > 0 unless "
                "D4RT_PLANNED_READ_ONLY_SPOOL=1 is set"
            )
        ctx = _get_mp_context()
        self._shared_generation = ctx.Value("i", self._generation)
        self.input_queue = None
        self.input_queues = []
        if self._sequence_affinity:
            queue_max = max(1, self.prefetch_depth * 2)
            self.input_queues = [
                ctx.Queue(maxsize=queue_max)
                for _ in range(self.builder_workers)
            ]
        else:
            self.input_queue = ctx.Queue(maxsize=self.prefetch_depth * 2)
        self.output_queue = ctx.Queue()

        self.builder_processes = []
        for i in range(self.builder_workers):
            proc = start_builder_process(
                builder_id=i,
                adapters=self.adapters,
                transform=self.transform,
                query_builder=self.query_builder,
                spool=self.spool,
                input_queue=(
                    self.input_queues[i]
                    if self._sequence_affinity
                    else self.input_queue
                ),
                output_queue=self.output_queue,
                clip_len=self.clip_len,
                current_generation=self._shared_generation,
                sample_stage_config=self.sample_stage_config,
            )
            self.builder_processes.append(proc)
            # Heartbeat every 8 forks so a slow re-fork (NFS contention) does
            # not look like a stall to an external watchdog.
            if progress_cb is not None and (i + 1) % 8 == 0:
                try:
                    progress_cb("fork_builders", f"{i + 1}/{self.builder_workers}")
                except Exception:
                    pass

        # Reset consumed index and start background enqueue thread.
        with self._consumed_lock:
            self._consumed_index = self._pipeline_start_index - 1
        self.next_enqueue_index = min(self._pipeline_start_index, len(self.current_plan))
        self._enqueue_stop.clear()
        self._enqueue_thread = threading.Thread(
            target=self._enqueue_loop, daemon=True, name=f"enqueue-rank{self.rank}"
        )
        self._enqueue_thread.start()
        self._pipeline_started = True

    def _stop_pipeline(self) -> None:
        """Tear down builder processes and queues.  Safe to call repeatedly."""
        self._enqueue_stop.set()
        if self._enqueue_thread is not None:
            self._enqueue_thread.join(timeout=3.0)
            self._enqueue_thread = None
        if self.builder_processes:
            stop_builder_processes(
                self.builder_processes,
                self.input_queues if self._sequence_affinity else self.input_queue,
                timeout=5.0,
            )
            self.builder_processes = []
        if self.output_queue is not None:
            try:
                self.output_queue.cancel_join_thread()
                self.output_queue.close()
            except Exception:
                pass
        # Let old queues be GC'd; new ones are created in _start_pipeline.
        self.input_queue = None
        self.input_queues = []
        self.output_queue = None
        self._shared_generation = None
        self._pipeline_started = False

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        # Padded to ensure equal work across ranks (see class docstring).
        return math.ceil(self.epoch_size / self.world_size)

    def __getitem__(self, index: int) -> QuerySample:
        """Return the pre-built bundle for *index*.

        The index MUST arrive sequentially (0, 1, 2, ...) because we use
        SequentialSampler.  The spool blocks until the bundle with the
        matching (index, generation) pair is ready.
        """
        if not self._pipeline_started:
            self._start_pipeline()
        if self._read_only_spool:
            sample = self._wait_read_only_spool(index)
        elif self._relaxed_order:
            sample = self._wait_relaxed_order(index)
        else:
            sample = self._wait_with_worker_check(index)

        returned_index = int(index)
        try:
            returned_index = int(sample.metadata.get("planned_returned_index", index))
        except Exception:
            returned_index = int(index)
        consumed_index = max(int(index), returned_index)
        with self._consumed_lock:
            if consumed_index > self._consumed_index:
                self._consumed_index = consumed_index

        self._write_rolling_warm_progress(index)
        return sample

    def _wait_read_only_spool(self, index: int) -> QuerySample:
        """Wait for an externally produced spool sample.

        This mode is used by offline/daemon prebuilders.  The training process
        does not enqueue work or retry failed builds; a missing sample means the
        producer has not caught up yet.
        """
        deadline = time.time() + self._read_only_spool_timeout_s
        requested = int(index)
        first_loop = True
        last_wait_log = 0.0

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Read-only spool sample g{self._generation:04d}_{requested} "
                    f"did not become ready within "
                    f"{self._read_only_spool_timeout_s:.1f}s. "
                    f"{self._format_plan_spec(requested)} "
                    f"{self._format_spool_summary()} "
                    f"Spool dir: {self.spool.spool_dir}"
                )

            exact_timeout = min(
                self._relaxed_grace_s if self._relaxed_order else remaining,
                max(0.0, remaining),
            )
            try:
                sample = self.spool.wait_for_bundle(
                    requested,
                    self._generation,
                    timeout=exact_timeout,
                )
                self._mark_returned_sample(sample, requested, requested)
                return sample
            except TimeoutError:
                pass
            except RuntimeError:
                raise

            if self._relaxed_order:
                ready_index = self._pick_relaxed_ready_index(requested)
                if ready_index is not None:
                    try:
                        sample = self.spool.wait_for_bundle(
                            ready_index,
                            self._generation,
                            timeout=0.0,
                        )
                    except TimeoutError:
                        first_loop = False
                        continue
                    self._mark_returned_sample(sample, requested, ready_index)
                    if self._relaxed_log and ready_index != requested:
                        print(
                            f"[PlannedDataset] read_only relaxed_order "
                            f"request={requested} returned={ready_index} "
                            f"lookahead={self._relaxed_lookahead}",
                            flush=True,
                        )
                    return sample

            now = time.time()
            if self._wait_log and now - last_wait_log >= 30.0:
                print(
                    f"[PlannedDataset] read_only waiting "
                    f"g{self._generation:04d}_{requested} "
                    f"elapsed={self._read_only_spool_timeout_s - remaining:.0f}s "
                    f"{self._format_plan_spec(requested)} "
                    f"{self._format_spool_summary()}",
                    flush=True,
                )
                last_wait_log = now
            first_loop = False
            time.sleep(0.05)

    def _enqueue_plan_index(self, index: int) -> None:
        if self._block_enqueue:
            self._enqueue_plan_block(index)
            return
        if self._skip_ready_enqueue and self.spool.is_ready(index, self._generation):
            self._enqueued_indices.add(int(index))
            return
        self._wait_rolling_warm_ready(index)
        self._enqueued_indices.add(int(index))
        self._put_spec(self.current_plan[index])

    def _enqueue_plan_block(self, index: int) -> None:
        index = int(index)
        if index in self._enqueued_indices or index >= len(self.current_plan):
            return
        if self._skip_ready_enqueue and self.spool.is_ready(index, self._generation):
            self._enqueued_indices.add(index)
            return

        seed = self.current_plan[index]
        self._wait_rolling_warm_ready(index)
        block = [seed]
        self._enqueued_indices.add(index)

        limit = min(len(self.current_plan), index + self._block_lookahead)
        for j in range(index + 1, limit):
            if len(block) >= self._block_max_samples:
                break
            if j in self._enqueued_indices:
                continue
            spec = self.current_plan[j]
            if (
                int(spec.dataset_idx) != int(seed.dataset_idx)
                or str(spec.sequence_name) != str(seed.sequence_name)
            ):
                continue
            if self._skip_ready_enqueue and self.spool.is_ready(j, self._generation):
                self._enqueued_indices.add(j)
                continue
            self._wait_rolling_warm_ready(j)
            block.append(spec)
            self._enqueued_indices.add(j)

        self._put_spec(block)

    def _enqueue_loop(self) -> None:
        """Background thread: keep builders fed up to prefetch_depth ahead of consumption."""
        while not self._enqueue_stop.is_set():
            with self._consumed_lock:
                consumed = self._consumed_index
            target = min(consumed + 1 + self.prefetch_depth, len(self.current_plan))
            while self.next_enqueue_index < target and not self._enqueue_stop.is_set():
                self._enqueue_plan_index(self.next_enqueue_index)
                self.next_enqueue_index += 1
            self._enqueue_stop.wait(timeout=0.02)

    def _put_spec(self, spec: Any) -> None:
        if self._sequence_affinity:
            if not self.input_queues:
                raise RuntimeError("sequence affinity is enabled but input queues are not initialized")
            affinity_spec = spec[0] if isinstance(spec, list) else spec
            builder_idx = self._affinity_builder_index(affinity_spec)
            self.input_queues[builder_idx].put(spec)
            return
        assert self.input_queue is not None
        self.input_queue.put(spec)

    def _affinity_builder_index(self, spec: Any) -> int:
        workers = max(1, int(self.builder_workers))
        key = f"{int(spec.dataset_idx)}\0{spec.sequence_name}".encode("utf-8", "surrogatepass")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        return int.from_bytes(digest, "little") % workers

    def _rolling_warm_block_for_index(self, index: int) -> int:
        batch_size = max(1, int(self._rolling_warm_batch_size))
        block_batches = max(1, int(self._rolling_warm_block_batches))
        local_batch = int(index) // batch_size
        return local_batch // block_batches

    def _wait_rolling_warm_ready(self, index: int) -> None:
        if not self._rolling_warm_ready_dir or self._rolling_warm_timeout_s <= 0:
            return
        block_id = self._rolling_warm_block_for_index(index)
        if block_id in self._rolling_warm_ready_blocks:
            return
        ready_path = (
            Path(self._rolling_warm_ready_dir)
            / f"g{self._generation:04d}"
            / f"block_{block_id:08d}.ready"
        )
        start = time.time()
        deadline = start + self._rolling_warm_timeout_s
        logged_wait = False
        while True:
            if ready_path.is_file():
                self._rolling_warm_ready_blocks.add(block_id)
                waited = time.time() - start
                if self._rolling_warm_log and (logged_wait or waited >= 1.0):
                    print(
                        f"[RollingWarmGate rank{self.rank}] ready "
                        f"epoch={self.current_epoch} generation={self._generation} "
                        f"block={block_id} wait={waited:.2f}s path={ready_path}",
                        flush=True,
                    )
                return
            now = time.time()
            if now >= deadline:
                raise TimeoutError(
                    f"Rolling warm block not ready after "
                    f"{self._rolling_warm_timeout_s:.1f}s: "
                    f"epoch={self.current_epoch} generation={self._generation} "
                    f"rank={self.rank} index={index} block={block_id} "
                    f"ready_path={ready_path}"
                )
            if self._rolling_warm_log and not logged_wait and now - start >= 1.0:
                print(
                    f"[RollingWarmGate rank{self.rank}] waiting "
                    f"epoch={self.current_epoch} generation={self._generation} "
                    f"block={block_id} path={ready_path}",
                    flush=True,
                )
                logged_wait = True
            time.sleep(min(0.25, max(0.01, deadline - now)))

    def _write_rolling_warm_progress(self, index: int) -> None:
        if not self._rolling_warm_progress_dir:
            return
        batch_size = max(1, int(self._rolling_warm_batch_size))
        local_batch = int(index) // batch_size
        block_id = self._rolling_warm_block_for_index(index)
        if self._rolling_warm_last_block == block_id:
            return
        self._rolling_warm_last_block = block_id
        root = Path(self._rolling_warm_progress_dir) / f"g{self._generation:04d}"
        try:
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"rank{self.rank}.progress"
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            tmp.write_text(
                "\n".join(
                    [
                        f"generation={self._generation}",
                        f"epoch={self.current_epoch}",
                        f"rank={self.rank}",
                        f"local_index={int(index)}",
                        f"local_batch={local_batch}",
                        f"block={block_id}",
                        f"time={time.time():.6f}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except OSError:
            pass

    def _wait_relaxed_order(
        self,
        index: int,
        total_timeout: float = 600.0,
        max_requeue: int = 3,
    ) -> QuerySample:
        """Return the nearest unconsumed ready sample within a lookahead window.

        This keeps the planned sample set but allows a bounded amount of
        out-of-order consumption, avoiding head-of-line stalls from one slow
        remote or CPU-heavy sample.
        """
        import time

        max_requeue = max(
            1,
            int(os.getenv("D4RT_MAX_REQUEUE", str(max_requeue))),
        )
        # 当 requested 等了这么久还没拿到任何能用的 sample,就放弃 lookahead 窗口
        # 限制,从整个 plan 里找任意 ready sample,避免训练永久 hang。
        force_skip_s = float(os.getenv("D4RT_PLANNED_FORCE_SKIP_S", "60") or "60")
        deadline = time.time() + total_timeout
        request_start = time.time()
        requested = int(index)
        first_loop = True
        last_alive_log = 0.0
        force_skip_logged = False

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                alive = sum(1 for p in self.builder_processes if p.is_alive())
                spec_info = self._format_plan_spec(requested)
                spool_info = self._format_spool_summary()
                raise TimeoutError(
                    f"Relaxed sample request {requested} did not find an "
                    f"unconsumed ready bundle within {total_timeout}s. "
                    f"alive_workers={alive}/{len(self.builder_processes)} "
                    f"{spec_info} {spool_info} Spool dir: {self.spool.spool_dir}"
                )

            self._handle_relaxed_errors(requested, max_requeue=max_requeue)

            if requested not in self._returned_indices:
                exact_timeout = min(
                    self._relaxed_grace_s if first_loop else 0.0,
                    max(0.0, remaining),
                )
                try:
                    sample = self.spool.wait_for_bundle(
                        requested,
                        self._generation,
                        timeout=exact_timeout,
                    )
                    self._mark_returned_sample(sample, requested, requested)
                    return sample
                except TimeoutError:
                    pass
                except RuntimeError:
                    self._handle_failed_index(requested, max_requeue=max_requeue)
                    first_loop = False
                    continue

            ready_index = self._pick_relaxed_ready_index(requested)
            # 等太久了,扩大搜索到整个 plan,有任何 ready sample 就用。
            request_elapsed = time.time() - request_start
            if ready_index is None and force_skip_s > 0 and request_elapsed >= force_skip_s:
                ready_index = self._pick_any_ready_index()
                if ready_index is not None and not force_skip_logged:
                    print(
                        f"[PlannedDataset] force_skip rank{self.rank} "
                        f"request={requested} returned={ready_index} "
                        f"waited={request_elapsed:.1f}s "
                        f"(lookahead={self._relaxed_lookahead} window empty, "
                        f"falling back to whole plan)",
                        flush=True,
                    )
                    force_skip_logged = True
                # spool 里完全没有 ready sample(builder 全卡死),把 requested
                # 写成 error 触发 substitute 逻辑,从 plan 别处随机选一个 spec
                # 重新入队,让 builder 有新任务可做。
                if (
                    ready_index is None
                    and request_elapsed >= force_skip_s * 2
                    and len(self.current_plan) > 1
                ):
                    print(
                        f"[PlannedDataset] force_substitute rank{self.rank} "
                        f"request={requested} waited={request_elapsed:.1f}s "
                        f"spool empty, marking as failed to trigger substitute",
                        flush=True,
                    )
                    self._handle_failed_index(requested, max_requeue=1)
                    request_start = time.time()
                    force_skip_logged = False
                    first_loop = False
                    continue
            if ready_index is not None:
                try:
                    sample = self.spool.wait_for_bundle(
                        ready_index,
                        self._generation,
                        timeout=0.0,
                    )
                except TimeoutError:
                    first_loop = False
                    continue
                except RuntimeError:
                    self._handle_failed_index(ready_index, max_requeue=max_requeue)
                    first_loop = False
                    continue
                self._mark_returned_sample(sample, requested, ready_index)
                if self._relaxed_log and ready_index != requested:
                    print(
                        f"[PlannedDataset] relaxed_order request={requested} "
                        f"returned={ready_index} lookahead={self._relaxed_lookahead}",
                        flush=True,
                    )
                return sample

            alive_workers = [p for p in self.builder_processes if p.is_alive()]
            dead_workers = [p for p in self.builder_processes if not p.is_alive()]
            now = time.time()
            if self._wait_log and now - last_alive_log > 30.0:
                print(
                    f"[PlannedDataset] relaxed waiting request={requested} "
                    f"alive={len(alive_workers)}/{len(self.builder_processes)} "
                    f"elapsed={total_timeout - remaining:.0f}s "
                    f"{self._format_plan_spec(requested)} {self._format_spool_summary()}",
                    flush=True,
                )
                last_alive_log = now
            if dead_workers and not alive_workers:
                unreturned_ready = self._pick_relaxed_ready_index(requested)
                if unreturned_ready is None:
                    exitcodes = [p.exitcode for p in dead_workers]
                    raise RuntimeError(
                        f"All {len(dead_workers)} builder workers died "
                        f"(exitcodes={exitcodes}). Relaxed request {requested} "
                        "cannot be satisfied."
                    )
            first_loop = False
            time.sleep(0.05)

    def _pick_relaxed_ready_index(self, requested: int) -> Optional[int]:
        limit = min(
            len(self.current_plan) - 1,
            int(requested) + self._relaxed_lookahead,
        )
        for ready_index in self.spool.list_ready_indices(self._generation):
            if ready_index in self._returned_indices:
                continue
            if ready_index <= limit:
                return ready_index
        return None

    def _pick_any_ready_index(self) -> Optional[int]:
        """Force-skip fallback: pick any unconsumed ready sample, ignoring lookahead.

        Used when the lookahead window has been empty for too long and the
        normal relaxed_order would otherwise hang indefinitely.
        """
        for ready_index in self.spool.list_ready_indices(self._generation):
            if ready_index in self._returned_indices:
                continue
            return ready_index
        return None

    def _handle_relaxed_errors(self, requested: int, max_requeue: int) -> None:
        limit = min(
            len(self.current_plan) - 1,
            int(requested) + self._relaxed_lookahead,
        )
        for error_index in self.spool.list_error_indices(self._generation):
            if error_index in self._returned_indices:
                continue
            if error_index <= limit:
                self._handle_failed_index(error_index, max_requeue=max_requeue)
                return

    def _handle_failed_index(self, index: int, max_requeue: int) -> None:
        import dataclasses
        import random

        self.spool._get_error_path(index, self._generation).unlink(missing_ok=True)
        count = self._requeue_counts.get(index, 0) + 1
        self._requeue_counts[index] = count
        if count >= max_requeue and len(self.current_plan) > 1:
            alt_indices = [i for i in range(len(self.current_plan)) if i != index]
            alt_idx = random.choice(alt_indices)
            alt_spec = self.current_plan[alt_idx]
            sub_spec = dataclasses.replace(
                alt_spec,
                local_index=index,
                generation=self._generation,
            )
            self._put_spec(sub_spec)
            self._requeue_counts[index] = 0
            print(
                f"[PlannedDataset] sample g{self._generation:04d}_{index} failed "
                f"{max_requeue}x, substituting with plan[{alt_idx}] "
                f"(dataset_idx={sub_spec.dataset_idx})",
                flush=True,
            )
        elif index < len(self.current_plan):
            self._put_spec(self.current_plan[index])

    def _mark_returned_sample(
        self,
        sample: QuerySample,
        requested_index: int,
        returned_index: int,
    ) -> None:
        self._returned_indices.add(int(returned_index))
        try:
            sample.metadata["planned_requested_index"] = int(requested_index)
            sample.metadata["planned_returned_index"] = int(returned_index)
            sample.metadata["planned_relaxed_order"] = (
                int(requested_index) != int(returned_index)
            )
        except Exception:
            pass

    def _wait_with_worker_check(self, index: int, total_timeout: float = 600.0, check_interval: float = 5.0, max_requeue: int = 3) -> QuerySample:
        """Wait for spool bundle, failing fast if all workers have died.

        After ``max_requeue`` consecutive failures for the same index, the
        failing spec is substituted with a randomly chosen spec from a
        different position in the plan so training can continue.
        """
        import dataclasses
        import random
        import time
        max_requeue = max(
            1,
            int(os.getenv("D4RT_MAX_REQUEUE", str(max_requeue))),
        )
        deadline = time.time() + total_timeout
        last_alive_log = 0.0
        requeue_count = 0
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                alive = sum(1 for p in self.builder_processes if p.is_alive())
                spec_info = self._format_plan_spec(index)
                spool_info = self._format_spool_summary()
                raise TimeoutError(
                    f"Sample g{self._generation:04d}_{index} did not become ready "
                    f"within {total_timeout}s. alive_workers={alive}/{len(self.builder_processes)} "
                    f"{spec_info} {spool_info} Spool dir: {self.spool.spool_dir}"
                )
            try:
                return self.spool.wait_for_bundle(
                    index, self._generation, timeout=min(check_interval, remaining)
                )
            except TimeoutError:
                pass
            except RuntimeError as e:
                # Builder wrote an error marker (e.g. timeout, COS failure).
                # Delete the marker and re-enqueue the spec so a worker retries it.
                self.spool._get_error_path(index, self._generation).unlink(missing_ok=True)
                requeue_count += 1
                if requeue_count >= max_requeue and len(self.current_plan) > 1:
                    # Persistent failure: substitute with a random spec from a
                    # different plan position so this slot doesn't block forever.
                    alt_indices = [i for i in range(len(self.current_plan)) if i != index]
                    alt_idx = random.choice(alt_indices)
                    alt_spec = self.current_plan[alt_idx]
                    sub_spec = dataclasses.replace(alt_spec, local_index=index, generation=self._generation)
                    self._put_spec(sub_spec)
                    requeue_count = 0
                    print(
                        f"[PlannedDataset] sample g{self._generation:04d}_{index} failed "
                        f"{max_requeue}x, substituting with plan[{alt_idx}] "
                        f"(dataset_idx={sub_spec.dataset_idx})",
                        flush=True,
                    )
                elif index < len(self.current_plan):
                    self._put_spec(self.current_plan[index])
                    print(f"[PlannedDataset] sample g{self._generation:04d}_{index} failed ({e}), re-enqueued", flush=True)

            alive_workers = [p for p in self.builder_processes if p.is_alive()]
            dead_workers = [p for p in self.builder_processes if not p.is_alive()]

            # Periodically log worker status
            now = time.time()
            if self._wait_log and now - last_alive_log > 30.0:
                print(
                    f"[PlannedDataset] waiting g{self._generation:04d}_{index} "
                    f"alive={len(alive_workers)}/{len(self.builder_processes)} "
                    f"elapsed={total_timeout - remaining:.0f}s "
                    f"{self._format_plan_spec(index)} {self._format_spool_summary()}",
                    flush=True,
                )
                last_alive_log = now

            # Hard fail if ALL workers are dead — no one will ever write the bundle.
            if dead_workers and not alive_workers and not self.spool.is_ready(index, self._generation):
                exitcodes = [p.exitcode for p in dead_workers]
                raise RuntimeError(
                    f"All {len(dead_workers)} builder workers died (exitcodes={exitcodes}). "
                    f"Sample g{self._generation:04d}_{index} cannot be produced. "
                    f"Check stderr for builder crash traceback / faulthandler output."
                )

    # ------------------------------------------------------------------
    # Epoch management
    # ------------------------------------------------------------------

    def _format_plan_spec(self, index: int) -> str:
        """Compact description of the sample currently blocking consumption."""
        try:
            spec = self.current_plan[index]
            dataset = self.adapters[spec.dataset_idx].dataset_name
            frame_min = min(spec.frame_indices) if spec.frame_indices else None
            frame_max = max(spec.frame_indices) if spec.frame_indices else None
            return (
                f"spec=dataset:{dataset},seq:{spec.sequence_name},"
                f"frames:{len(spec.frame_indices)}[{frame_min}..{frame_max}],"
                f"global:{spec.global_index}"
            )
        except Exception as exc:
            return f"spec=<unavailable:{type(exc).__name__}>"

    def _format_spool_summary(self) -> str:
        """Summarize current-generation ready indices for head-of-line diagnosis."""
        try:
            prefix = f"g{self._generation:04d}_"
            ready_indices = []
            for path in self.spool.spool_dir.iterdir():
                name = path.name
                if not (name.startswith(prefix) and name.endswith(".ready")):
                    continue
                try:
                    ready_indices.append(int(name.split("_", 1)[1].split(".", 1)[0]))
                except Exception:
                    continue
            if not ready_indices:
                return "spool_ready=0"
            ready_indices.sort()
            return (
                f"spool_ready={len(ready_indices)},"
                f"min={ready_indices[0]},max={ready_indices[-1]}"
            )
        except Exception as exc:
            return f"spool_ready=<unavailable:{type(exc).__name__}>"

    def set_epoch(self, epoch: int, progress_cb=None) -> None:
        """Advance to *epoch*.

        Two paths:
          * **Hot switch** (default, when builders are already alive): keep the
            builder processes running and only bump the shared generation +
            re-plan + drain queues.  This avoids the kill-64/re-fork-64 cost and,
            critically, the re-fork deadlock that can occur under NFS contention.
            See ``_hot_switch_epoch``.
          * **Full rebuild** (first epoch, read-only spool, or
            ``D4RT_PLANNED_HOT_EPOCH_SWITCH=0``): tear down old queues +
            processes and create fresh ones.  Heavier but guarantees a clean
            slate.

        Both paths preserve zero cross-epoch pollution: builders gate on
        ``spec.generation`` and the consumer reads only via the spool's
        ``(index, generation)`` filter.

        ``progress_cb`` (optional) is invoked as ``progress_cb(step, detail)``
        at each sub-step of the transition.  It exists so an external watchdog
        can be heartbeated during this multi-second/-minute rebuild: without
        it, a caller-side watchdog whose phase label is frozen at the prior
        step (e.g. "save_checkpoint") accumulates the entire transition as a
        single stall and may false-kill a slow-but-progressing rebuild.  The
        callback must not raise; callers should swallow their own errors.
        """
        if (
            epoch == self.current_epoch
            and self._pipeline_started
            and self.current_plan
        ):
            return

        def _tick(step: str, detail: str = "") -> None:
            if progress_cb is not None:
                try:
                    progress_cb(step, detail)
                except Exception:
                    pass

        # Hot path: builders already alive -> bump generation + re-plan + drain
        # queues, no kill/refork.  Avoids the re-fork deadlock under contention.
        if (
            self._hot_epoch_switch
            and self._pipeline_started
            and not self._read_only_spool
            and self.builder_processes
            and self._shared_generation is not None
        ):
            self._hot_switch_epoch(epoch, _tick)
            return

        self.current_epoch = epoch
        t0 = time.monotonic()

        # 1. Kill old builders + queues.
        _tick("stop_pipeline", "")
        self._stop_pipeline()
        t_stop = time.monotonic()

        # 2. Bump generation.
        self._generation += 1

        # 3. Purge stale spool files.
        _tick("purge_spool", f"stop={t_stop - t0:.1f}s")
        self.spool.set_generation(self._generation)
        self._returned_indices.clear()
        self._requeue_counts.clear()
        self._enqueued_indices.clear()
        self._rolling_warm_ready_blocks.clear()
        self._rolling_warm_last_block = None

        # 4. Re-plan.
        _tick("re_plan", f"stop={t_stop - t0:.1f}s")
        self.current_plan = self.planner.generate_plan(
            epoch=epoch, count_per_rank=math.ceil(self.epoch_size / self.world_size),
            epoch_size=self.epoch_size, generation=self._generation,
        )
        t_plan = time.monotonic()
        if self.current_plan:
            self._write_rolling_warm_progress(0)

        # 5. Spin up fresh pipeline.
        _tick("start_pipeline", f"plan={t_plan - t_stop:.1f}s")
        self._start_pipeline(progress_cb=progress_cb)
        _tick("done", f"total={time.monotonic() - t0:.1f}s")

    def _hot_switch_epoch(self, epoch: int, tick) -> None:
        """Advance epoch WITHOUT killing/re-forking builders.

        Builders are kept running; correctness is preserved by:
          1. bumping ``_shared_generation.value`` so live builders drop any
             in-flight stale specs at their generation gate, and
          2. bumping the spool generation so the consumer's
             ``wait_for_bundle(index, generation)`` cannot return old samples.

        Both input and output queues are drained: input queues so builders are
        not fed stale specs (and to free queue capacity), output queues because
        nothing consumes them and they would otherwise grow unbounded once the
        builders stop being recreated each epoch.
        """
        self.current_epoch = epoch
        t0 = time.monotonic()

        # 1. Pause the enqueue thread so it stops feeding old-gen specs.
        tick("hot:pause_enqueue", "")
        self._enqueue_stop.set()
        if self._enqueue_thread is not None:
            self._enqueue_thread.join(timeout=5.0)
            self._enqueue_thread = None
        t_pause = time.monotonic()

        # 2. Bump generation: shared (builders see it) + local + spool.
        tick("hot:bump_generation", f"pause={t_pause - t0:.1f}s")
        self._generation += 1
        try:
            self._shared_generation.value = self._generation
        except Exception:
            pass
        self.spool.set_generation(self._generation)

        # 3. Drain queues (stale specs in / unread completions out).
        tick("hot:drain_queues", "")
        drained_in = self._drain_input_queues()
        drained_out = self._drain_output_queue()
        t_drain = time.monotonic()

        # 4. Reset per-epoch bookkeeping.
        self._returned_indices.clear()
        self._requeue_counts.clear()
        self._enqueued_indices.clear()
        self._rolling_warm_ready_blocks.clear()
        self._rolling_warm_last_block = None

        # 5. Re-plan.
        tick("hot:re_plan", f"drain_in={drained_in} drain_out={drained_out}")
        self.current_plan = self.planner.generate_plan(
            epoch=epoch, count_per_rank=math.ceil(self.epoch_size / self.world_size),
            epoch_size=self.epoch_size, generation=self._generation,
        )
        if self.current_plan:
            self._write_rolling_warm_progress(0)

        # 6. Reset consume/enqueue cursors and restart the enqueue thread
        #    feeding the SAME live builders.
        with self._consumed_lock:
            self._consumed_index = self._pipeline_start_index - 1
        self.next_enqueue_index = min(self._pipeline_start_index, len(self.current_plan))
        self._enqueue_stop.clear()
        self._enqueue_thread = threading.Thread(
            target=self._enqueue_loop, daemon=True, name=f"enqueue-rank{self.rank}"
        )
        self._enqueue_thread.start()
        tick("hot:done", f"total={time.monotonic() - t0:.1f}s plan={time.monotonic() - t_drain:.1f}s")

    def _drain_input_queues(self) -> int:
        """Discard any queued specs so live builders are not fed stale work."""
        queues = self.input_queues if self._sequence_affinity else (
            [self.input_queue] if self.input_queue is not None else []
        )
        drained = 0
        for q in queues:
            if q is None:
                continue
            while True:
                try:
                    q.get_nowait()
                    drained += 1
                except queue.Empty:
                    break
                except (OSError, ValueError):
                    break
        return drained

    def _drain_output_queue(self) -> int:
        """Discard unread completion signals (nothing consumes them)."""
        if self.output_queue is None:
            return 0
        drained = 0
        while True:
            try:
                self.output_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
            except (OSError, ValueError):
                break
        return drained
        _tick("done", f"total={time.monotonic() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def get_dataset_names(self) -> list[str]:
        return [adapter.dataset_name for adapter in self.adapters]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Stop builder processes and remove spool files."""
        self._stop_pipeline()
        if not self._preserve_spool_on_cleanup:
            self.spool.cleanup()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass


# ======================================================================
# Utility
# ======================================================================

def set_dataset_epoch(dataset: Any, epoch: int) -> None:
    """Set epoch on a dataset (works for both MixtureDataset and PlannedMixtureDataset)."""
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    elif hasattr(dataset, "current_epoch"):
        dataset.current_epoch = epoch
