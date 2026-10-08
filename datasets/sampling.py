"""
Single-dataset sampler for D4RT training.

Handles:
- Sequence sampling (uniform or weighted)
- Frame window sampling (random / sequential / dataset-adaptive stride)
- Clip length validation
- RNG state management

Usage:
    sampler = DatasetSampler(
        adapter=pointodyssey_adapter,
        clip_len=48,
        sampling_mode='stride',
    )
    sequence_name, frame_indices = sampler.sample(rng)
"""

from __future__ import annotations

import random as _random_module
from dataclasses import dataclass
from math import gcd
from typing import Literal, Optional

from datasets.adapters.base import BaseAdapter


@dataclass
class SamplerConfig:
    """Configuration for dataset sampling."""
    clip_len: int = 8
    sampling_mode: Literal['random', 'sequential', 'stride', 'stride_wrap', 'covisibility', 'covisibility_stride'] = 'random'
    min_frames: int = 8
    sequence_weights: Optional[dict[str, float]] = None
    sequence_locality_size: int = 1
    frame_locality_radius: Optional[int] = None


# Dataset-adaptive stride policy.
#
# Tuned from tools/probe_adjacent_stride.py: adjacent-frame cov(i, i+stride)
# across 25 scenes/dataset, 4 pairs/scene (2026-06-09). This is the correct
# metric: stride controls the gap between *consecutive* sampled frames
# (exactly `stride` raw frames apart), NOT clip endpoints which are
# (clip_len-1)*stride apart and get inflated by clip_len.
#
# Decision rule (p10 across scenes, i.e. worst-case scene):
#   p10 >= 0.60  -> learnable, include
#   p10  0.40-0.60 -> borderline, include with low weight
#   p10 <  0.40  -> too sparse, exclude
#
# Weights tilt toward larger strides for novel-view diversity while keeping
# small-stride samples for fine-grained tracking supervision.
_DATASET_STRIDE_POLICY: dict[str, tuple[list[int], list[float]]] = {
    # pointodyssey: very slow camera. Adjacent cov at s=8: p10=0.82, p50=0.984.
    # Sequences long (p50=1649). Rebalance weights toward s=4,8 to target pad ~10%.
    # Probed 2026-06-09, 25 seq, adjacent metric.
    'pointodyssey':    ([1, 2, 4, 8, 16, 24],     [0.20, 0.20, 0.25, 0.25, 0.07, 0.03]),

    # kubric: synthetic slow-camera scenes, fixed 120 frames. Collapse to stride=1
    # to target pad <5%. Stride diversity provided by mixture training with other
    # datasets. Per-step random stride with s=2 causes 23% pad due to short sequences.
    # Probed 2026-06-09, 25 seq, adjacent metric; collapsed 2026-06-09.
    'kubric':          ([1],                      [1.0]),

    # dynamic_replica: photorealistic dynamic scenes, fixed 300 frames/seq.
    # Rebalance heavily toward s=1,2,3 to target pad ~12%.
    # Probed 2026-06-09, 20 seq (valid split only), adjacent metric; re-tuned 2026-06-09.
    'dynamic_replica': ([1, 2, 3, 4],             [0.35, 0.35, 0.20, 0.10]),

    # scannet: 30Hz indoor RGBD. Adjacent cov at s=8: p10=0.814, p50=0.895.
    # Long sequences (p50=1350), rebalance toward s=3,4,6 to maintain diversity
    # while keeping pad ~8%.
    # Probed 2026-06-09, 25 seq, adjacent metric; extended 2026-06-09.
    'scannet':         ([1, 2, 3, 4, 6, 8],       [0.05, 0.20, 0.25, 0.30, 0.15, 0.05]),

    # scannetpp: indoor RGBD, similar motion profile to scannet. Inheriting
    # scannet optimized policy.
    'scannetpp':       ([1, 2, 3, 4, 6, 8],       [0.05, 0.20, 0.25, 0.30, 0.15, 0.05]),

    # co3dv2: object-centric orbit (~102 frames = one full 360 turn).
    # Used with 'stride_wrap' sampling mode (modular indices), NOT the per-step
    # random-stride walk. Adjacent cov saturated ~1.0 across all strides (object
    # fills FOV). Strides 3/6 excluded: gcd(3,102)=3 / gcd(6,102)=6 leave only
    # 34/17 distinct orbit positions < clip_len=48; _sample_wrap_stride
    # filters them. Kept strides {1,2,4,5,7,8} all have gcd<=2 -> >=51
    # distinct positions. Tilt toward mid/large strides for subset diversity.
    # Probed 2026-06-09, 25 seq, adjacent metric.
    'co3dv2':          ([1, 2, 4, 5, 7, 8],       [0.10, 0.15, 0.20, 0.20, 0.20, 0.15]),

    # blendedmvs: unordered multi-view captures (NOT a temporal video).
    # Adjacent cov p10~0.32-0.55, p50~0.78-0.82, FLAT across all strides --
    # consecutive indices are already wide-baseline views so stride doesn't
    # control overlap. These candidates only apply when falling back from
    # covisibility_stride mode; the covisibility-aware path is preferred.
    # Probed 2026-06-09, 25 seq, adjacent metric.
    'blendedmvs':      ([1, 2, 3, 4, 6],          [0.10, 0.15, 0.20, 0.25, 0.30]),

    # tartanair: simulated outdoor/indoor, moderate-to-fast motion.
    # Adjacent cov at s=4: p10=0.806, p50=0.885. At s=8: p10=0.640 (borderline).
    # Mid-length sequences (p50=599), rebalance toward s=2,3,4 for pad ~12%.
    # Probed 2026-06-09, 25 seq, adjacent metric; extended 2026-06-09.
    'tartanair':       ([1, 2, 3, 4, 6, 8],       [0.10, 0.25, 0.30, 0.25, 0.08, 0.02]),

    # mvssynth: GTA-rendered driving scenes, fixed 100 frames, fast motion.
    # Collapse to stride=1 to target pad <5%. Stride diversity provided by
    # mixture training with other datasets.
    # Probed 2026-06-09, 25 seq, adjacent metric; collapsed 2026-06-09.
    'mvssynth':        ([1],                      [1.0]),

    # waymo: real autonomous driving, ~200 frames, fast motion. Trim to [1,2,3]
    # with heavy s=1,2 bias to target pad ~12%.
    # No local copy at 2026-06-04 probe time; re-tuned 2026-06-09 per-step test.
    'waymo':           ([1, 2, 3],                [0.60, 0.30, 0.10]),

    # vkitti2: KITTI highway simulation, fastest motion in the mix, p50=339 frames.
    # Trim to [1,2,3,4], rebalance heavily toward s=1,2,3 for pad ~12%.
    # Probed 2026-06-09, 25 seq, adjacent metric; confirmed 2026-06-09 per-step.
    'vkitti2':         ([1, 2, 3, 4],             [0.40, 0.35, 0.20, 0.05]),
}

# Conservative fallback for unknown datasets.
_DEFAULT_STRIDE_CANDIDATES = [1, 2, 3, 4]
_DEFAULT_STRIDE_WEIGHTS = [0.50, 0.25, 0.15, 0.10]


def _is_sorted(xs) -> bool:
    """Return True if ``xs`` is non-decreasing.

    A clip whose frame_indices are non-decreasing has its array positions in
    time order, so neighbour-frame motion boundaries are meaningful. _pad_repeat
    appends duplicates of the last index, which keeps a sorted clip sorted. A
    clip permuted by shuffle_prob is (almost surely) not sorted, which is the
    signal the query builder uses to skip motion-boundary computation per clip.
    """
    return all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))


@dataclass
class _SequenceBlockState:
    """Worker-local reusable sequence block for short-term sampling locality."""
    sequence_name: str
    num_frames: int
    remaining_uses: int
    last_start_idx: Optional[int] = None


class DatasetSampler:
    """
    Single-dataset sampler.

    Args:
        adapter: Dataset adapter (PointOdyssey, ScanNet, etc.)
        clip_len: Number of frames per clip
        sampling_mode: 'random', 'sequential', or 'stride'
        min_frames: Minimum frames required in a sequence
        sequence_weights: Optional per-sequence sampling weights
        allowed_sequences: If set, only these sequence names will be sampled.
                           Useful for single-scene training mode.

    Notes:
        sampling_mode='stride' uses a dataset-adaptive temporal/view stride
        policy instead of a single global max_stride. This is more reliable for
        mixed training across dynamic video, RGB-D scan, and static multi-view
        datasets.
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        clip_len: int = 8,
        sampling_mode: Literal['random', 'sequential', 'stride', 'stride_wrap', 'covisibility', 'covisibility_stride'] = 'random',
        min_frames: int = 8,
        sequence_weights: Optional[dict[str, float]] = None,
        allowed_sequences: Optional[list[str]] = None,
        custom_stride_range: Optional[tuple[int, int]] = None,
        sequence_locality_size: int = 1,
        frame_locality_radius: Optional[int] = None,
        stride_policy: Optional[dict[str, tuple[list[int], list[float]]]] = None,
        shuffle_frame_order: bool = False,
        shuffle_prob: Optional[float] = None,
    ):
        self.adapter = adapter
        self.clip_len = clip_len
        self.sampling_mode = sampling_mode
        self.min_frames = min_frames
        self.sequence_weights = sequence_weights
        self.allowed_sequences_set = set(allowed_sequences) if allowed_sequences is not None else None
        self.custom_stride_range = custom_stride_range
        # Optional per-dataset stride policy override (from config). When None,
        # fall back to the module-level _DATASET_STRIDE_POLICY hardcoded default.
        self.stride_policy = stride_policy
        # When True, the final clip's frame_indices are randomly permuted so the
        # model sees an unordered multi-view set instead of a time-ascending clip.
        # All sampling modes otherwise return time-ascending (sorted) indices.
        #
        # shuffle_prob is the per-clip probability of applying that permutation,
        # enabling a *mix* of ordered and shuffled clips within a single dataset.
        # Back-compat: a bare shuffle_frame_order=True maps to prob=1.0 (always
        # shuffle), False maps to 0.0 (never). An explicit shuffle_prob overrides.
        self.shuffle_frame_order = shuffle_frame_order
        if shuffle_prob is None:
            self.shuffle_prob = 1.0 if shuffle_frame_order else 0.0
        else:
            self.shuffle_prob = max(0.0, min(1.0, float(shuffle_prob)))
        self.sequence_locality_size = max(1, int(sequence_locality_size))
        self.frame_locality_radius = (
            None if frame_locality_radius is None else max(0, int(frame_locality_radius))
        )
        self._active_sequence: Optional[_SequenceBlockState] = None

        # Filter sequences by minimum frame count (and whitelist if given)
        self.valid_sequences = self._filter_sequences()

        if len(self.valid_sequences) == 0:
            if self.allowed_sequences_set is not None:
                raise ValueError(
                    f"None of the requested sequences {sorted(self.allowed_sequences_set)} "
                    f"were found (or had enough frames) in {adapter.dataset_name}. "
                    f"Available sequences: {adapter.list_sequences()[:10]}"
                )
            raise ValueError(
                f"No valid sequences found with >= {min_frames} frames in {adapter.dataset_name}"
            )

        # Build sampling weights
        self.sampling_probs = self._build_sampling_probs()

    def _filter_sequences(self) -> list[str]:
        """Filter sequences that have enough frames, and optionally by whitelist."""
        valid = []
        for seq_name in self.adapter.list_sequences():
            if self.allowed_sequences_set is not None and seq_name not in self.allowed_sequences_set:
                continue
            try:
                num_frames = self.adapter.get_num_frames(seq_name)
                if num_frames >= self.min_frames:
                    valid.append(seq_name)
            except Exception:
                pass  # Skip sequences with missing or corrupt files
        return valid

    def _build_sampling_probs(self) -> list[float]:
        """Build sampling probabilities for each valid sequence."""
        if self.sequence_weights is None:
            return [1.0 / len(self.valid_sequences)] * len(self.valid_sequences)

        weights = [self.sequence_weights.get(seq_name, 1.0) for seq_name in self.valid_sequences]
        total = sum(weights)
        return [w / total for w in weights]

    def _get_stride_policy(self) -> tuple[list[int], list[float]]:
        """Return dataset-adaptive stride candidates and weights.

        Prefers an instance-level ``stride_policy`` (supplied from config) over
        the module-level ``_DATASET_STRIDE_POLICY`` hardcoded default. Unknown
        datasets fall back to the conservative default candidates/weights.
        """
        dataset_name = str(getattr(self.adapter, 'dataset_name', '')).lower()
        policy = self.stride_policy if self.stride_policy is not None else _DATASET_STRIDE_POLICY
        return policy.get(
            dataset_name,
            (_DEFAULT_STRIDE_CANDIDATES, _DEFAULT_STRIDE_WEIGHTS),
        )

    def _sample_stride(self, num_frames: int, rng: _random_module.Random) -> int:
        """Sample a single stride value from the dataset-adaptive policy or custom range.

        Only filters out strides that exceed the sequence length entirely (stride >=
        num_frames). The per-step random-walk in _sample_stride_sequence handles the
        actual boundary via _pad_repeat, so the old (clip_len-1)*stride budget check
        is intentionally absent here.
        """
        if self.custom_stride_range is not None:
            min_stride, max_stride = self.custom_stride_range
            valid_strides = [s for s in range(min_stride, max_stride + 1) if s < num_frames]
            if not valid_strides:
                return 1
            return rng.choice(valid_strides)

        candidates, weights = self._get_stride_policy()

        valid_candidates = []
        valid_weights = []
        for stride, weight in zip(candidates, weights):
            if stride < num_frames:
                valid_candidates.append(stride)
                valid_weights.append(weight)

        if not valid_candidates:
            return 1

        total = sum(valid_weights)
        valid_probs = [w / total for w in valid_weights]
        return rng.choices(valid_candidates, weights=valid_probs, k=1)[0]

    def _sample_stride_sequence(self, start_idx: int, num_frames: int, rng: _random_module.Random) -> list[int]:
        """Build a clip by stepping forward with per-step independently sampled strides.

        Each of the (clip_len-1) gaps between consecutive frames draws its own stride
        from the dataset policy. This produces variable inter-frame gaps within a single
        clip, giving richer motion-magnitude diversity than a fixed-stride clip.

        If the random walk runs off the end of the sequence before clip_len frames are
        collected, _pad_repeat fills the remainder with the last valid frame index.
        """
        indices = [start_idx]
        pos = start_idx
        while len(indices) < self.clip_len:
            stride = self._sample_stride(num_frames, rng)
            pos = pos + stride
            if pos >= num_frames:
                break
            indices.append(pos)
        return self._pad_repeat(indices, self.clip_len)

    def _sample_wrap_stride(self, num_frames: int, rng: _random_module.Random) -> int:
        """Sample a stride for wrap-around (modular) sampling.

        Unlike _sample_stride, this is NOT bounded by the linear length
        constraint (clip_len-1)*stride+1 <= num_frames, because indices wrap
        modulo num_frames. The only constraint is that clip_len distinct
        frames must exist on the orbit: with stride s the residues
        (start + i*s) % n cycle through n // gcd(s, n) distinct positions, so
        we require clip_len <= num_frames // gcd(stride, num_frames). Strides
        violating this (e.g. s=3,6 on a 102-frame orbit -> 34/17 positions)
        would force duplicate frames and are filtered out.
        """
        candidates, weights = self._get_stride_policy()

        valid_candidates = []
        valid_weights = []
        for stride, weight in zip(candidates, weights):
            distinct = num_frames // gcd(stride, num_frames)
            if distinct >= self.clip_len:
                valid_candidates.append(stride)
                valid_weights.append(weight)

        if not valid_candidates:
            return 1

        total = sum(valid_weights)
        valid_probs = [w / total for w in valid_weights]
        return rng.choices(valid_candidates, weights=valid_probs, k=1)[0]

    def reset_locality_state(self) -> None:
        """Drop any reusable sequence block for this worker-local sampler copy."""
        self._active_sequence = None

    def _sample_sequence_name(self, rng: _random_module.Random) -> str:
        return rng.choices(self.valid_sequences, weights=self.sampling_probs, k=1)[0]

    def _start_sequence_block(self, rng: _random_module.Random) -> _SequenceBlockState:
        sequence_name = self._sample_sequence_name(rng)
        num_frames = self.adapter.get_num_frames(sequence_name)
        state = _SequenceBlockState(
            sequence_name=sequence_name,
            num_frames=num_frames,
            remaining_uses=self.sequence_locality_size,
        )
        return state

    @staticmethod
    def _pad_repeat(indices: list[int], clip_len: int) -> list[int]:
        """Repeat the last frame index until the list reaches clip_len."""
        if len(indices) >= clip_len:
            return indices
        return indices + [indices[-1]] * (clip_len - len(indices))

    def _sample_start_idx(
        self,
        max_start: int,
        rng: _random_module.Random,
        previous_start: Optional[int] = None,
    ) -> int:
        if max_start <= 0:
            return 0
        if previous_start is None or self.frame_locality_radius is None or self.frame_locality_radius <= 0:
            return rng.randint(0, max_start)

        low = max(0, previous_start - self.frame_locality_radius)
        high = min(max_start, previous_start + self.frame_locality_radius)
        if low > high:
            low, high = 0, max_start
        return rng.randint(low, high)

    def sample(self, rng: Optional[_random_module.Random] = None) -> tuple[str, list[int]]:
        """
        Sample a sequence and frame indices.

        Returns:
            (sequence_name, frame_indices)
        """
        if rng is None:
            rng = _random_module.Random()

        block_state = self._active_sequence
        if block_state is None or block_state.remaining_uses <= 0:
            block_state = self._start_sequence_block(rng)

        frame_indices = self._sample_frame_indices(block_state.num_frames, rng, block_state=block_state)
        block_state.remaining_uses -= 1
        self._active_sequence = block_state if block_state.remaining_uses > 0 else None
        return block_state.sequence_name, frame_indices

    def _sample_frame_indices(
        self,
        num_frames: int,
        rng: _random_module.Random,
        block_state: Optional[_SequenceBlockState] = None,
    ) -> list[int]:
        """Sample frame indices, then optionally shuffle their temporal order.

        Single收口点 for all sampling modes: the per-mode logic in
        ``_sample_frame_indices_ordered`` always returns time-ascending indices;
        with probability ``shuffle_prob`` the clip is randomly permuted so the
        model sees an unordered multi-view set (no temporal cue). Drawing per
        clip means a single dataset can mix ordered and shuffled clips. The frame
        multiset is unchanged — only the order differs.
        """
        indices = self._sample_frame_indices_ordered(num_frames, rng, block_state=block_state)
        if self.shuffle_prob > 0.0 and len(indices) > 1 and rng.random() < self.shuffle_prob:
            indices = list(indices)
            rng.shuffle(indices)
        return indices

    def _sample_frame_indices_ordered(
        self,
        num_frames: int,
        rng: _random_module.Random,
        block_state: Optional[_SequenceBlockState] = None,
    ) -> list[int]:
        """Sample frame indices based on sampling mode (time-ascending)."""
        if self.sampling_mode == 'random':
            # Random sampling with replacement.
            # Useful for stress-testing, but usually not ideal for coherent video clips.
            return sorted(rng.choices(range(num_frames), k=self.clip_len))

        if self.sampling_mode == 'sequential':
            if num_frames <= self.clip_len:
                if block_state is not None:
                    block_state.last_start_idx = 0
                return self._pad_repeat(list(range(num_frames)), self.clip_len)
            start_idx = self._sample_start_idx(
                num_frames - self.clip_len,
                rng,
                previous_start=None if block_state is None else block_state.last_start_idx,
            )
            if block_state is not None:
                block_state.last_start_idx = start_idx
            return list(range(start_idx, start_idx + self.clip_len))

        if self.sampling_mode == 'stride':
            # start_idx: anywhere in [0, num_frames-1]; the random walk in
            # _sample_stride_sequence handles boundary via _pad_repeat.
            start_idx = self._sample_start_idx(
                num_frames - 1,
                rng,
                previous_start=None if block_state is None else block_state.last_start_idx,
            )
            if block_state is not None:
                block_state.last_start_idx = start_idx
            return self._sample_stride_sequence(start_idx, num_frames, rng)

        if self.sampling_mode == 'covisibility':
            # Covisibility-aware sampling for static multi-view datasets.
            # Pick an anchor frame, then sample from its co-visible neighbors.
            # Falls back to random if adapter doesn't support covisibility.
            get_neighbors = getattr(self.adapter, 'get_covisibility_neighbors', None)
            if get_neighbors is None:
                # Adapter doesn't support covisibility, fall back to random
                return sorted(rng.choices(range(num_frames), k=self.clip_len))

            # Sample anchor frame
            anchor = rng.randint(0, num_frames - 1)
            sequence_name = block_state.sequence_name if block_state is not None else None
            if sequence_name is None:
                # No sequence context, fall back to random
                return sorted(rng.choices(range(num_frames), k=self.clip_len))

            neighbors = get_neighbors(sequence_name, anchor)
            if neighbors is None or len(neighbors) == 0:
                # No neighbor info, fall back to random
                return sorted(rng.choices(range(num_frames), k=self.clip_len))

            # Sample clip_len-1 frames from neighbors (plus anchor)
            if len(neighbors) >= self.clip_len - 1:
                sampled_neighbors = rng.sample(neighbors, self.clip_len - 1)
            else:
                # Not enough neighbors, sample with replacement
                sampled_neighbors = rng.choices(neighbors, k=self.clip_len - 1)

            indices = [anchor] + sampled_neighbors
            return sorted(indices)

        if self.sampling_mode == 'covisibility_stride':
            # Hybrid mode: Mix covisibility-aware sampling (dense views) with
            # random stride subsampling (sparse views) to train both regimes.
            # This increases sparse-view reconstruction robustness by exposing
            # the model to variable inter-frame gaps within co-visible sets.
            get_neighbors = getattr(self.adapter, 'get_covisibility_neighbors', None)
            if get_neighbors is None:
                # Fall back to per-step random-stride walk (same as 'stride' mode).
                start_idx = self._sample_start_idx(
                    num_frames - 1, rng,
                    previous_start=None if block_state is None else block_state.last_start_idx,
                )
                if block_state is not None:
                    block_state.last_start_idx = start_idx
                return self._sample_stride_sequence(start_idx, num_frames, rng)

            # Sample anchor and get co-visible neighbors
            anchor = rng.randint(0, num_frames - 1)
            sequence_name = block_state.sequence_name if block_state is not None else None
            if sequence_name is None:
                return sorted(rng.choices(range(num_frames), k=self.clip_len))

            neighbors = get_neighbors(sequence_name, anchor)
            if neighbors is None or len(neighbors) == 0:
                return sorted(rng.choices(range(num_frames), k=self.clip_len))

            # Build candidate pool: anchor + all neighbors
            candidates = sorted([anchor] + neighbors)

            # If pool is smaller than clip_len, sample with replacement to reach clip_len
            if len(candidates) < self.clip_len:
                # Use all candidates + sample additional with replacement
                additional_needed = self.clip_len - len(candidates)
                additional = rng.choices(candidates, k=additional_needed)
                return sorted(candidates + additional)

            # If pool equals clip_len, use all
            if len(candidates) == self.clip_len:
                return candidates

            # Apply stride subsampling on the sorted candidate pool
            # This gives variable inter-frame gaps while staying within co-visible set
            max_stride = max(1, len(candidates) // self.clip_len)
            stride = rng.randint(1, max(2, max_stride))  # Random stride in [1, max_stride]

            max_start = len(candidates) - (self.clip_len - 1) * stride
            if max_start <= 0:
                # Not enough candidates for this stride, sample without stride
                return sorted(rng.sample(candidates, self.clip_len))

            start = rng.randint(0, max_start - 1)
            indices = [candidates[start + i * stride] for i in range(self.clip_len)]
            return sorted(indices)

        if self.sampling_mode == 'stride_wrap':
            # Wrap-around (modular) stride for orbit-style multi-view datasets
            # (e.g. co3dv2: ~102 frames = one full 360 orbit around a static
            # object). Frame index = (start + i*stride) % num_frames, so the
            # clip can use strides larger than the linear length budget allows
            # and still pick clip_len distinct frames spread around the orbit.
            #
            # Benefit is subset diversity, not wider baseline: linear stride=2
            # yields only ~8 distinct clips per sequence (all "every other
            # frame"), while wrap-around with stride coprime-ish to num_frames
            # yields one distinct full-orbit subset per start phase (~num_frames
            # of them). The max pairwise baseline (~180 antipodal) is already
            # reached at stride=2, so this trades no baseline for much richer
            # view-combination coverage across epochs.
            #
            # _sample_wrap_stride filters strides whose gcd with num_frames
            # leaves < clip_len distinct orbit positions (would force dups).
            stride = self._sample_wrap_stride(num_frames, rng)
            if num_frames <= self.clip_len:
                # Degenerate: not enough frames to subsample; use all in order.
                return list(range(num_frames))
            start_idx = rng.randint(0, num_frames - 1)
            if block_state is not None:
                block_state.last_start_idx = start_idx
            indices = [(start_idx + i * stride) % num_frames for i in range(self.clip_len)]
            return sorted(indices)

        raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")

    def __len__(self) -> int:
        """Number of valid sequences."""
        return len(self.valid_sequences)

    def get_dataset_name(self) -> str:
        """Get dataset name."""
        return self.adapter.dataset_name
