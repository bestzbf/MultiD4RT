"""
Multi-dataset mixture sampler and dataset for D4RT training.

Combines multiple datasets with configurable sampling weights.

Usage:
    mixture = MixtureDataset(
        adapters=[pointodyssey_adapter, scannet_adapter],
        dataset_weights=[0.7, 0.3],
        clip_len=8,
        img_size=256,
        use_augs=True,
    )
    sample = mixture[0]  # Returns QuerySample

Single-scene usage (only one sequence):
    mixture = MixtureDataset(
        adapters=[pointodyssey_adapter],
        allowed_sequences_per_adapter=[['ani']],
        clip_len=8,
        img_size=256,
    )
"""

from __future__ import annotations

import multiprocessing as _mp
import os
import random as _random_module
from typing import Optional

from datasets.adapters.base import BaseAdapter
from datasets.query_builder import D4RTQueryBuilder, QuerySample
from datasets.sampling import DatasetSampler, _is_sorted
from datasets.transforms import GeometryTransformPipeline


def _resolve_positive_int(
    explicit_value: Optional[int],
    env_name: str,
    default_value: int,
) -> int:
    raw_value = explicit_value
    if raw_value is None:
        raw_env = os.getenv(env_name)
        raw_value = default_value if raw_env is None else raw_env
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} must be an integer, got {raw_value!r}") from exc
    return max(1, value)


def _resolve_optional_nonnegative_int(
    explicit_value: Optional[int],
    env_name: str,
    default_value: Optional[int],
) -> Optional[int]:
    raw_value = explicit_value
    if raw_value is None:
        raw_value = os.getenv(env_name, default_value)
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} must be an integer, got {raw_value!r}") from exc
    return None if value <= 0 else value


class MixtureSampler:
    """
    Multi-dataset sampler with configurable weights.

    Args:
        samplers: List of DatasetSampler instances
        dataset_weights: Sampling weight for each dataset (will be normalized)
    """

    def __init__(
        self,
        samplers: list[DatasetSampler],
        dataset_weights: Optional[list[float]] = None,
        dataset_locality_size: int = 1,
    ):
        if len(samplers) == 0:
            raise ValueError("samplers cannot be empty")

        self.samplers = samplers
        self.dataset_locality_size = max(1, int(dataset_locality_size))
        self._active_dataset_idx: Optional[int] = None
        self._remaining_dataset_uses = 0

        # Normalize weights
        if dataset_weights is None:
            dataset_weights = [1.0] * len(samplers)

        if len(dataset_weights) != len(samplers):
            raise ValueError(
                f"dataset_weights length {len(dataset_weights)} != samplers length {len(samplers)}"
            )

        total = sum(dataset_weights)
        self.dataset_probs = [w / total for w in dataset_weights]

    def reset_locality_state(self, dataset_idx: Optional[int] = None) -> None:
        """Reset short-lived dataset / sequence reuse state."""
        if dataset_idx is None or dataset_idx == self._active_dataset_idx:
            self._active_dataset_idx = None
            self._remaining_dataset_uses = 0
        if dataset_idx is None:
            for sampler in self.samplers:
                sampler.reset_locality_state()
            return
        self.samplers[dataset_idx].reset_locality_state()

    def _sample_dataset_idx(self, rng: _random_module.Random) -> int:
        if self._active_dataset_idx is not None and self._remaining_dataset_uses > 0:
            dataset_idx = self._active_dataset_idx
            self._remaining_dataset_uses -= 1
            if self._remaining_dataset_uses <= 0:
                self._active_dataset_idx = None
            return dataset_idx

        dataset_idx = rng.choices(range(len(self.samplers)), weights=self.dataset_probs, k=1)[0]
        if self.dataset_locality_size > 1:
            self._active_dataset_idx = dataset_idx
            self._remaining_dataset_uses = self.dataset_locality_size - 1
        return dataset_idx

    def sample(self, rng: Optional[_random_module.Random] = None) -> tuple[int, str, list[int]]:
        """
        Sample a dataset, sequence, and frame indices.

        Returns:
            (dataset_idx, sequence_name, frame_indices)
        """
        if rng is None:
            rng = _random_module.Random()

        dataset_idx = self._sample_dataset_idx(rng)

        # Sample from that dataset
        sequence_name, frame_indices = self.samplers[dataset_idx].sample(rng)

        return dataset_idx, sequence_name, frame_indices

    def get_dataset_names(self) -> list[str]:
        """Get list of dataset names."""
        return [s.get_dataset_name() for s in self.samplers]


class MixtureDataset:
    """
    Multi-dataset PyTorch-style dataset.

    Combines multiple adapters with transforms and query builder.

    Args:
        adapters: List of dataset adapters
        dataset_weights: Sampling weight for each dataset
        clip_len: Number of frames per clip
        img_size: Target image size (square)
        use_augs: Whether to use data augmentation
        num_queries: Number of queries per sample
        boundary_ratio: Ratio of boundary samples
        t_tgt_eq_t_cam_ratio: Ratio of t_tgt == t_cam samples
        seed: Random seed for reproducibility
        allowed_sequences_per_adapter: Optional list (one entry per adapter) of
            sequence name whitelists. Pass None for an adapter to allow all its
            sequences. Example: [['ani', 'ani11_new_'], None] — restricts the
            first adapter to two sequences while the second is unrestricted.
    """

    def __init__(
        self,
        adapters: list[BaseAdapter],
        dataset_weights: Optional[list[float]] = None,
        clip_len: int = 8,
        img_size: int = 256,
        use_augs: bool = True,
        num_queries: int = 2048,
        boundary_ratio: float = 0.3,
        t_tgt_eq_t_cam_ratio: float = 0.4,
        use_motion_boundaries: bool = True,
        seed: int = 42,
        allowed_sequences_per_adapter: Optional[list[Optional[list[str]]]] = None,
        sampling_mode: str | dict[str, str] = 'stride',
        epoch_size: int = 10000,
        custom_stride_range: Optional[tuple[int, int]] = None,
        stride_policy: Optional[dict[str, tuple[list[int], list[float]]]] = None,
        shuffle_frame_order: bool = False,
        shuffle_prob: Optional[dict[str, float]] = None,
        precompute_patches: bool = True,
        precompute_from_highres: bool = False,
        return_highres_video: Optional[bool] = None,
        allow_track_fallback: bool = False,
        store_video_uint8: bool = False,
        store_auxiliary_tensors: bool = True,
        keep_cropped_images: bool = True,
        color_aug_after_resize: bool = False,
        motion_boundary_on_resized: bool = True,
        max_track_points: Optional[int] = None,
        reshuffle_each_epoch: bool = False,
        dataset_locality_size: Optional[int] = None,
        sequence_locality_size: Optional[int] = None,
        frame_locality_radius: Optional[int] = None,
    ):
        self.adapters = adapters
        self.clip_len = clip_len
        self.seed = seed
        self.epoch_size = epoch_size
        self.reshuffle_each_epoch = reshuffle_each_epoch
        default_dataset_locality = 2 if use_augs and len(adapters) > 1 else 1
        self.dataset_locality_size = _resolve_positive_int(
            dataset_locality_size,
            "D4RT_DATASET_LOCALITY_SIZE",
            default_dataset_locality,
        )
        default_sequence_locality = 3 if use_augs else 1
        self.sequence_locality_size = _resolve_positive_int(
            sequence_locality_size,
            "D4RT_SEQUENCE_LOCALITY_SIZE",
            default_sequence_locality,
        )
        default_frame_locality = clip_len if use_augs and sampling_mode != 'random' and self.sequence_locality_size > 1 else None
        self.frame_locality_radius = _resolve_optional_nonnegative_int(
            frame_locality_radius,
            "D4RT_FRAME_LOCALITY_RADIUS",
            default_frame_locality,
        )
        # Keep epoch visible across worker-local dataset copies when DataLoader
        # uses persistent workers.
        self._shared_epoch = _mp.Value("q", 0, lock=False)
        self._local_epoch = self.current_epoch

        # Build samplers for each dataset
        samplers = []
        for i, adapter in enumerate(adapters):
            allowed = None
            if allowed_sequences_per_adapter is not None:
                allowed = allowed_sequences_per_adapter[i]

            # Resolve sampling_mode for this adapter
            if isinstance(sampling_mode, dict):
                dataset_name = getattr(adapter, 'dataset_name', None)
                mode = sampling_mode.get(dataset_name, 'stride')
            else:
                mode = sampling_mode

            samplers.append(
                DatasetSampler(
                    adapter=adapter,
                    clip_len=clip_len,
                    allowed_sequences=allowed,
                    sampling_mode=mode,
                    custom_stride_range=custom_stride_range,
                    sequence_locality_size=self.sequence_locality_size,
                    frame_locality_radius=self.frame_locality_radius,
                    stride_policy=stride_policy,
                    shuffle_frame_order=shuffle_frame_order,
                    shuffle_prob=(
                        None if shuffle_prob is None
                        else shuffle_prob.get(str(getattr(adapter, 'dataset_name', '')).lower(), 0.0)
                    ),
                )
            )

        # Build mixture sampler
        self.mixture_sampler = MixtureSampler(
            samplers=samplers,
            dataset_weights=dataset_weights,
            dataset_locality_size=self.dataset_locality_size,
        )

        # Build transform pipeline
        self.transform = GeometryTransformPipeline(
            img_size=img_size,
            use_augs=use_augs,
            keep_cropped_images=keep_cropped_images,
            color_aug_after_resize=color_aug_after_resize,
            max_track_points=max_track_points,
        )

        # Build query builder
        self.query_builder = D4RTQueryBuilder(
            num_queries=num_queries,
            boundary_ratio=boundary_ratio,
            t_tgt_eq_t_cam_ratio=t_tgt_eq_t_cam_ratio,
            use_motion_boundaries=use_motion_boundaries,
            precompute_patches=precompute_patches,
            precompute_from_highres=precompute_from_highres,
            return_highres_video=return_highres_video,
            allow_track_fallback=allow_track_fallback,
            store_video_uint8=store_video_uint8,
            store_auxiliary_tensors=store_auxiliary_tensors,
            motion_boundary_on_resized=motion_boundary_on_resized,
        )

    @property
    def current_epoch(self) -> int:
        return int(self._shared_epoch.value)

    def set_epoch(self, epoch: int) -> None:
        self._shared_epoch.value = int(epoch)

    def _reset_worker_locality_state(self) -> None:
        self.mixture_sampler.reset_locality_state()
        self._local_epoch = self.current_epoch

    def _sync_epoch_state(self) -> None:
        if self._local_epoch != self.current_epoch:
            self._reset_worker_locality_state()

    def __getitem__(self, index: int) -> QuerySample:
        """
        Get a training sample.

        Args:
            index: Sample index (used as RNG seed)

        Returns:
            QuerySample with video, coords, targets, masks, etc.
        """
        self._sync_epoch_state()

        # Create RNG from index + seed
        sample_index = index
        if self.reshuffle_each_epoch:
            sample_index = self.current_epoch * self.epoch_size + index
        rng = _random_module.Random(self.seed + sample_index)

        # Sample dataset, sequence, and frames; retry on corrupt files or wrong shape
        for attempt in range(10):
            dataset_idx, sequence_name, frame_indices = self.mixture_sampler.sample(rng)
            adapter = self.adapters[dataset_idx]
            try:
                clip = adapter.load_clip(sequence_name, frame_indices)
                result = self.transform(clip, rng=rng)
                sample = self.query_builder(
                    result, py_rng=rng, frames_time_ordered=_is_sorted(frame_indices)
                )
                if sample.video.shape[0] != self.clip_len:
                    raise RuntimeError(f"Sample has {sample.video.shape[0]} frames, expected {self.clip_len}")
                break
            except Exception as e:
                self.mixture_sampler.reset_locality_state(dataset_idx)
                if attempt >= 2:
                    print(f"Warning: skipping bad sample (attempt {attempt+1}): {e}")
                rng = _random_module.Random(self.seed + sample_index + attempt + 1)
        else:
            raise RuntimeError(f"Failed to load a valid sample after 10 attempts at index {index}")

        return sample

    def __len__(self) -> int:
        """
        Dataset length (arbitrary, since we sample randomly).
        Set to a reasonable epoch size.
        """
        return self.epoch_size

    def get_dataset_names(self) -> list[str]:
        """Get list of dataset names in the mixture."""
        return self.mixture_sampler.get_dataset_names()
