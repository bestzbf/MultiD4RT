"""
Sample planning for deterministic prefetch in planned mode.

This module generates deterministic sample plans for each rank, enabling
predictable I/O patterns and efficient prefetching from COS.

Key design decisions:
- **Single-RNG-per-sample:** each sample creates one ``Random(seed + sample_index)``
  matching online mode's per-sample RNG structure (``MixtureDataset.__getitem__``).
  The same RNG drives ``mixture_sampler.sample()`` and is then captured as
  ``rng_state`` for downstream transform / query_builder.
- Each rank only *keeps* the items where ``global_index % world_size == rank``.
- ``local_index`` is the position within a rank's sub-plan (0..count_per_rank-1),
  used as the spool file name.
- ``global_index`` is the absolute sample ID, used for RNG seed computation.
- ``generation`` tracks epoch transitions so builders can discard stale tasks.
"""

from __future__ import annotations

import math
import os
import pickle
import random as _random_module
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

from .mixture import MixtureSampler


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class SampleSpec:
    """Specification for a single sample to be built by a background worker.

    Attributes:
        local_index: Position in this rank's plan (0..count-1), used for spool filename.
        global_index: Absolute sample ID across all ranks, used for RNG seed.
        dataset_idx: Index into the mixture's dataset list.
        sequence_name: Name of the sequence to load.
        frame_indices: List of frame indices to load.
        rng_state: Serialized RNG state after sampling (for transform + query_builder).
        generation: Epoch generation; builders discard tasks from older generations.
    """
    local_index: int
    global_index: int
    dataset_idx: int
    sequence_name: str
    frame_indices: list[int]
    rng_state: tuple[Any, ...]
    generation: int


class SamplePlanner:
    """Generates deterministic sample plans for a single rank.

    Uses a **single-RNG-per-sample** design that matches online mode's
    per-sample RNG structure (``MixtureDataset.__getitem__``): each sample
    creates one ``Random(seed + sample_index)`` which first drives
    ``mixture_sampler.sample(rng)`` and then continues into transform /
    query_builder via the captured ``rng_state``.

    **Known difference from online mode:** The inter-sample locality patterns
    differ because online mode's locality evolves through DataLoader worker
    scheduling (non-deterministic), while planned mode uses independent
    per-sample RNGs (deterministic but no implicit locality).

    The planner iterates ``total_samples = count_per_rank * world_size``
    and keeps only items where ``global_index % world_size == rank``.
    """

    def __init__(
        self,
        mixture_sampler: MixtureSampler,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        reshuffle_each_epoch: bool = True,
    ):
        self.mixture_sampler = mixture_sampler
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.reshuffle_each_epoch = reshuffle_each_epoch

    def generate_plan(
        self, epoch: int, count_per_rank: int, epoch_size: int, generation: int | None = None,
    ) -> list[SampleSpec]:
        """Generate a deterministic plan of SampleSpecs for this rank.

        Each sample uses a single RNG seeded from (seed, epoch, sample_index),
        matching the online mode's per-sample RNG structure (mixture.py).  The
        same RNG drives sampling, then its post-sampling state is captured as
        ``rng_state`` for downstream transform / query_builder.

        **Padding behavior (PyTorch DistributedSampler semantics):** When
        ``total_samples = count_per_rank * world_size`` exceeds ``epoch_size``,
        padding samples reuse early indices via ``sample_index = global_index % epoch_size``.
        This ensures all ranks process the same number of samples while maintaining
        deterministic RNG seeding that wraps around the epoch boundary.

        **Known difference from online mode:** in online mode, MixtureSampler's
        locality state (dataset blocks, sequence blocks) evolves implicitly
        through DataLoader worker scheduling, which is non-deterministic.  In
        planned mode, locality is not preserved because each sample creates an
        independent RNG — the sampling decisions are identical to what online
        mode would produce for the same (seed, epoch, index) tuple, but the
        inter-sample locality patterns differ.  This is an intentional trade-off:
        deterministic per-sample reproducibility over implicit locality.

        Args:
            epoch: Current epoch number.
            count_per_rank: Number of samples each rank should produce.
            epoch_size: Total number of unique samples in the epoch (before padding).
            generation: Generation tag stamped into every SampleSpec.
                If *None*, defaults to *epoch* for backward compat.

        Returns:
            List of ``SampleSpec`` with ``len == count_per_rank``.
        """
        if generation is None:
            generation = epoch

        total_samples = count_per_rank * self.world_size

        # Reset locality so every rank starts from the same clean state.
        self.reset_for_epoch()

        plan: list[SampleSpec] = []
        local_index = 0

        for global_index in range(total_samples):
            # Wrap sample_index to epoch_size (PyTorch DistributedSampler padding).
            # When total_samples > epoch_size, padding samples reuse early indices.
            sample_index = global_index % epoch_size

            # Single RNG per sample, matching online mode's seeding.
            if self.reshuffle_each_epoch:
                sample_seed = self.seed + epoch * epoch_size + sample_index
            else:
                sample_seed = self.seed + sample_index
            rng = _random_module.Random(sample_seed)

            # Sampling consumes the RNG (same structure as online mode).
            dataset_idx, sequence_name, frame_indices = (
                self.mixture_sampler.sample(rng)
            )

            # Capture post-sampling RNG state for transform / query_builder.
            rng_state = rng.getstate()

            if global_index % self.world_size == self.rank:
                plan.append(SampleSpec(
                    local_index=local_index,
                    global_index=global_index,
                    dataset_idx=dataset_idx,
                    sequence_name=sequence_name,
                    frame_indices=frame_indices,
                    rng_state=rng_state,
                    generation=generation,
                ))
                local_index += 1

        if _env_flag("D4RT_PLANNED_BATCH_BALANCE", False) and plan:
            batch_size = _env_int("D4RT_PLANNED_BATCH_SIZE", 0)
            if batch_size > 1:
                plan = self._balance_plan_batches(plan, batch_size)

        return plan

    def reset_for_epoch(self) -> None:
        """Reset the mixture sampler's locality state for a fresh epoch."""
        self.mixture_sampler.reset_locality_state()

    def _balance_plan_batches(
        self,
        plan: list[SampleSpec],
        batch_size: int,
    ) -> list[SampleSpec]:
        """Reorder each rank plan so local batches follow dataset weights.

        This does not change the sampled set for the epoch; it only avoids long
        local windows of the same expensive dataset draining the planned spool.
        Per-dataset order is preserved inside each bucket, so sequence/frame
        locality within a dataset is mostly retained.
        """
        probs = list(getattr(self.mixture_sampler, "dataset_probs", []))
        if not probs:
            return plan
        num_datasets = len(probs)
        buckets: list[deque[SampleSpec]] = [deque() for _ in range(num_datasets)]
        overflow: deque[SampleSpec] = deque()
        for spec in plan:
            if 0 <= int(spec.dataset_idx) < num_datasets:
                buckets[int(spec.dataset_idx)].append(spec)
            else:
                overflow.append(spec)

        targets = self._balanced_batch_targets(probs, batch_size)
        balanced: list[SampleSpec] = []

        while any(buckets) or overflow:
            batch: list[SampleSpec] = []
            counts = [0] * num_datasets

            for dataset_idx, target in enumerate(targets):
                take = min(target, len(buckets[dataset_idx]))
                for _ in range(take):
                    batch.append(buckets[dataset_idx].popleft())
                counts[dataset_idx] = take

            while len(batch) < batch_size and (any(buckets) or overflow):
                if overflow:
                    batch.append(overflow.popleft())
                    continue
                best_idx = max(
                    range(num_datasets),
                    key=lambda idx: len(buckets[idx]) - max(0, targets[idx] - counts[idx]),
                )
                if not buckets[best_idx]:
                    break
                batch.append(buckets[best_idx].popleft())
                counts[best_idx] += 1

            balanced.extend(self._interleave_batch_by_dataset(batch, num_datasets))

        return [replace(spec, local_index=i) for i, spec in enumerate(balanced)]

    @staticmethod
    def _balanced_batch_targets(probs: list[float], batch_size: int) -> list[int]:
        raw = [max(0.0, float(p)) * batch_size for p in probs]
        targets = [int(math.floor(v)) for v in raw]
        remainder = batch_size - sum(targets)
        order = sorted(range(len(raw)), key=lambda idx: raw[idx] - targets[idx], reverse=True)
        for idx in order[: max(0, remainder)]:
            targets[idx] += 1
        return targets

    @staticmethod
    def _interleave_batch_by_dataset(
        batch: list[SampleSpec],
        num_datasets: int,
    ) -> list[SampleSpec]:
        grouped: list[deque[SampleSpec]] = [deque() for _ in range(num_datasets)]
        other: deque[SampleSpec] = deque()
        for spec in batch:
            if 0 <= int(spec.dataset_idx) < num_datasets:
                grouped[int(spec.dataset_idx)].append(spec)
            else:
                other.append(spec)

        out: list[SampleSpec] = []
        while any(grouped) or other:
            for bucket in grouped:
                if bucket:
                    out.append(bucket.popleft())
            if other:
                out.append(other.popleft())
        return out


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def serialize_sample_spec(spec: SampleSpec) -> bytes:
    """Serialize a SampleSpec to bytes for IPC."""
    return pickle.dumps(spec)


def deserialize_sample_spec(data: bytes) -> SampleSpec:
    """Deserialize a SampleSpec from bytes."""
    return pickle.loads(data)
