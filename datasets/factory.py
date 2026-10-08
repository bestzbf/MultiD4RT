"""
Dataset factory for D4RT training.

Supports three training modes:
  - 'single'  : train on one entire dataset
  - 'scene'   : train on specific sequences from one dataset
  - 'mixture' : train on multiple datasets mixed by weight

Usage:
    # Single dataset
    dataset = create_training_dataset(config)

    # Single scene
    dataset = create_training_dataset(scene_config)

    # Mixture dataset
    dataset = create_training_dataset(mixture_config)
"""

from __future__ import annotations
import os
from typing import Union

from datasets.registry import create_adapter
from datasets.mixture import MixtureDataset
from datasets.planned_dataset import PlannedMixtureDataset


def _resolve_keep_cropped_images(config: dict) -> bool:
    explicit = config.get("keep_cropped_images", None)
    if explicit is not None:
        return bool(explicit)

    precompute_patches = bool(config.get("precompute_patches", True))
    precompute_from_highres = bool(config.get("precompute_from_highres", False))
    return_highres = config.get("return_highres_video", None)

    if precompute_patches and precompute_from_highres:
        return True
    if return_highres is not None:
        return bool(return_highres)
    return not precompute_patches


def _resolve_max_track_points(config: dict):
    explicit = config.get("max_track_points", None)
    if explicit is not None:
        value = int(explicit)
        return value if value > 0 else None
    raw = os.getenv("D4RT_MAX_TRACK_POINTS", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _resolve_sampling_mode(config: dict, adapter) -> str:
    """Resolve the sampling mode for a single adapter.

    ``sampling_mode`` in config may be either a plain string (applied to all
    datasets) or a dict mapping dataset_name -> mode (per-dataset override,
    falling back to 'stride' for datasets not listed).
    """
    sampling_mode = config.get('sampling_mode', 'stride')
    if isinstance(sampling_mode, dict):
        dataset_name = getattr(adapter, 'dataset_name', None)
        return sampling_mode.get(dataset_name, 'stride')
    return sampling_mode


def _resolve_stride_policy(config: dict):
    """Resolve an optional per-dataset stride policy from config.

    ``stride_policy`` in YAML is a dict mapping dataset_name -> either
    ``{candidates: [...], weights: [...]}`` or a 2-element ``[candidates,
    weights]`` list. Returns a normalized ``dict[str, (list[int], list[float])]``
    suitable for ``DatasetSampler(stride_policy=...)``, or ``None`` when absent
    (in which case the hardcoded module default _DATASET_STRIDE_POLICY is used).
    """
    raw = config.get('stride_policy', None)
    if not isinstance(raw, dict) or not raw:
        return None

    policy: dict = {}
    for dataset_name, spec in raw.items():
        if isinstance(spec, dict):
            candidates = spec.get('candidates')
            weights = spec.get('weights')
        elif isinstance(spec, (list, tuple)) and len(spec) == 2:
            candidates, weights = spec[0], spec[1]
        else:
            raise ValueError(
                f"stride_policy['{dataset_name}'] must be a dict with "
                f"'candidates'/'weights' or a [candidates, weights] pair, got: {spec!r}"
            )
        if not isinstance(candidates, (list, tuple)) or not isinstance(weights, (list, tuple)):
            raise ValueError(
                f"stride_policy['{dataset_name}'] candidates/weights must be lists, got "
                f"candidates={candidates!r}, weights={weights!r}"
            )
        if len(candidates) != len(weights):
            raise ValueError(
                f"stride_policy['{dataset_name}'] candidates/weights length mismatch: "
                f"{len(candidates)} vs {len(weights)}"
            )
        policy[str(dataset_name).lower()] = (
            [int(c) for c in candidates],
            [float(w) for w in weights],
        )
    return policy


def _resolve_shuffle_frame_order(config: dict) -> bool:
    """Resolve whether to randomly permute each clip's frame order.

    When True, all sampling modes still pick a time-ascending clip but the final
    frame_indices are shuffled, so the model sees an unordered multi-view set.
    """
    return bool(config.get('shuffle_frame_order', False))


def _resolve_shuffle_prob(config: dict):
    """Resolve an optional per-dataset shuffle probability map from config.

    ``shuffle_prob`` in YAML is a dict mapping dataset_name -> float in [0, 1],
    the per-clip probability of permuting that clip's frame order. This lets a
    single dataset mix ordered and shuffled clips (e.g. blendedmvs 0.9 vs
    pointodyssey 0.05). Returns a normalized ``dict[str, float]`` (names
    lowercased), or ``None`` when absent (in which case the bare
    ``shuffle_frame_order`` boolean governs, applying uniformly to all datasets).
    """
    raw = config.get('shuffle_prob', None)
    if not isinstance(raw, dict) or not raw:
        return None

    prob: dict = {}
    for dataset_name, value in raw.items():
        try:
            p = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"shuffle_prob['{dataset_name}'] must be a number in [0, 1], got: {value!r}"
            )
        if not (0.0 <= p <= 1.0):
            raise ValueError(
                f"shuffle_prob['{dataset_name}'] must be in [0, 1], got: {p}"
            )
        prob[str(dataset_name).lower()] = p
    return prob


def create_training_dataset(
    config: dict,
    split: str = 'train',
    rank: int = 0,
    world_size: int = 1,
) -> Union[MixtureDataset, PlannedMixtureDataset]:
    """
    Create dataset from config.

    Args:
        config: Dataset configuration dict
        split: 'train', 'val', or 'test'
        rank: DDP rank (for planned mode)
        world_size: DDP world size (for planned mode)

    Returns:
        MixtureDataset or PlannedMixtureDataset

    ---
    Config format for single dataset (all sequences):
        mode: single
        name: pointodyssey
        root: /path/to/data
        clip_len: 48
        img_size: 256
        num_queries: 2048

    ---
    Config format for single scene (one or more specific sequences):
        mode: scene
        name: pointodyssey
        root: /path/to/data
        sequences:
          - ani
        # or multiple:
        # sequences:
        #   - ani
        #   - ani11_new_
        clip_len: 48
        img_size: 256
        num_queries: 2048

    ---
    Config format for mixture:
        mode: mixture
        datasets:
          - name: pointodyssey
            root: /path
            weight: 0.5
          - name: scannet
            root: /path
            weight: 0.3
        clip_len: 48
        img_size: 256
        num_queries: 2048
    """
    mode = config.get('mode', 'single')

    if mode == 'single':
        return _create_single_dataset(config, split, rank, world_size)
    elif mode == 'scene':
        return _create_scene_dataset(config, split, rank, world_size)
    elif mode == 'mixture':
        return _create_mixture_dataset(config, split, rank, world_size)
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Use 'single', 'scene', or 'mixture'")


def _create_single_dataset(config: dict, split: str, rank: int = 0, world_size: int = 1) -> Union[MixtureDataset, PlannedMixtureDataset]:
    """Train on all sequences of one dataset."""
    index_cache_dir = config.get('index_cache_dir', None)
    index_workers = config.get('index_workers', None)
    extra = {}
    if index_cache_dir:
        extra['cache_dir'] = index_cache_dir
    if index_workers is not None:
        extra['index_workers'] = index_workers
    adapter = create_adapter(
        name=config['name'],
        root=config['root'],
        split=split,
        **config.get('adapter_kwargs', {}),
        **extra,
    )

    return MixtureDataset(
        adapters=[adapter],
        dataset_weights=[1.0],
        clip_len=config.get('clip_len', 48),
        img_size=config.get('img_size', 256),
        use_augs=config.get('use_augs', True),
        num_queries=config.get('num_queries', 2048),
        boundary_ratio=config.get('boundary_ratio', 0.3),
        t_tgt_eq_t_cam_ratio=config.get('t_tgt_eq_t_cam_ratio', 0.4),
        use_motion_boundaries=config.get('use_motion_boundaries', True),
        seed=config.get('seed', 42),
        sampling_mode=config.get('sampling_mode', 'stride'),
        epoch_size=config.get('epoch_size', 10000),
        precompute_patches=config.get('precompute_patches', True),
        precompute_from_highres=config.get('precompute_from_highres', False),
        return_highres_video=config.get('return_highres_video', None),
        allow_track_fallback=config.get('allow_track_fallback', False),
        store_video_uint8=config.get('store_video_uint8', False),
        store_auxiliary_tensors=config.get('store_auxiliary_tensors', True),
        keep_cropped_images=_resolve_keep_cropped_images(config),
        color_aug_after_resize=config.get('color_aug_after_resize', False),
        motion_boundary_on_resized=config.get('motion_boundary_on_resized', True),
        max_track_points=_resolve_max_track_points(config),
        reshuffle_each_epoch=config.get('reshuffle_each_epoch', split == 'train'),
    )


def _create_scene_dataset(config: dict, split: str, rank: int = 0, world_size: int = 1) -> Union[MixtureDataset, PlannedMixtureDataset]:
    """Train on specific sequences (scenes) from one dataset."""
    sequences = config.get('sequences')
    if not sequences:
        raise ValueError(
            "mode='scene' requires a non-empty 'sequences' list. "
            "Example:\n  sequences:\n    - ani"
        )

    index_cache_dir = config.get('index_cache_dir', None)
    index_workers = config.get('index_workers', None)
    extra = {}
    if index_cache_dir:
        extra['cache_dir'] = index_cache_dir
    if index_workers is not None:
        extra['index_workers'] = index_workers
    adapter = create_adapter(
        name=config['name'],
        root=config['root'],
        split=split,
        **config.get('adapter_kwargs', {}),
        **extra,
    )

    custom_stride_range = None
    if 'stride_range' in config:
        stride_range = config['stride_range']
        if isinstance(stride_range, list) and len(stride_range) == 2:
            custom_stride_range = tuple(stride_range)

    return MixtureDataset(
        adapters=[adapter],
        dataset_weights=[1.0],
        clip_len=config.get('clip_len', 48),
        img_size=config.get('img_size', 256),
        use_augs=config.get('use_augs', True),
        num_queries=config.get('num_queries', 2048),
        boundary_ratio=config.get('boundary_ratio', 0.3),
        t_tgt_eq_t_cam_ratio=config.get('t_tgt_eq_t_cam_ratio', 0.4),
        use_motion_boundaries=config.get('use_motion_boundaries', True),
        seed=config.get('seed', 42),
        allowed_sequences_per_adapter=[list(sequences)],
        sampling_mode=config.get('sampling_mode', 'stride'),
        epoch_size=config.get('epoch_size', 10000),
        custom_stride_range=custom_stride_range,
        precompute_patches=config.get('precompute_patches', True),
        precompute_from_highres=config.get('precompute_from_highres', False),
        return_highres_video=config.get('return_highres_video', None),
        allow_track_fallback=config.get('allow_track_fallback', False),
        store_video_uint8=config.get('store_video_uint8', False),
        store_auxiliary_tensors=config.get('store_auxiliary_tensors', True),
        keep_cropped_images=_resolve_keep_cropped_images(config),
        color_aug_after_resize=config.get('color_aug_after_resize', False),
        motion_boundary_on_resized=config.get('motion_boundary_on_resized', True),
        max_track_points=_resolve_max_track_points(config),
        reshuffle_each_epoch=config.get('reshuffle_each_epoch', split == 'train'),
    )


def _create_mixture_dataset(config: dict, split: str, rank: int = 0, world_size: int = 1) -> Union[MixtureDataset, PlannedMixtureDataset]:
    """Train on multiple datasets mixed by weight."""
    index_cache_dir = config.get('index_cache_dir', None)
    index_workers = config.get('index_workers', None)
    custom_stride_range = None
    if 'stride_range' in config:
        stride_range = config['stride_range']
        if isinstance(stride_range, list) and len(stride_range) == 2:
            custom_stride_range = tuple(stride_range)

    ds_configs = config['datasets']

    def _build_one(ds_config):
        actual_split = ds_config.get('val_split', split) if split == 'val' else ds_config.get('split', split)
        extra = {}
        if index_cache_dir:
            extra['cache_dir'] = index_cache_dir
        if index_workers is not None:
            extra['index_workers'] = index_workers
        adapter = create_adapter(
            name=ds_config['name'],
            root=ds_config['root'],
            split=actual_split,
            **ds_config.get('adapter_kwargs', {}),
            **extra,
        )
        if split == 'val' and 'val_sequences' in ds_config:
            allowed = ds_config['val_sequences']
        elif split == 'train' and 'train_sequences' in ds_config:
            allowed = ds_config['train_sequences']
        else:
            allowed = ds_config.get('sequences', None)
        return adapter, ds_config.get('weight', 1.0), allowed

    serialize_build = os.getenv("D4RT_SERIALIZE_ADAPTER_INIT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    results = [None] * len(ds_configs)
    if serialize_build or len(ds_configs) <= 1:
        for idx, ds_config in enumerate(ds_configs):
            results[idx] = _build_one(ds_config)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

        with ThreadPoolExecutor(max_workers=len(ds_configs)) as executor:
            future_to_idx = {executor.submit(_build_one, dc): i for i, dc in enumerate(ds_configs)}
            for fut in _as_completed(future_to_idx):
                results[future_to_idx[fut]] = fut.result()

    adapters = [r[0] for r in results]
    weights = [r[1] for r in results]
    allowed_per_adapter = [r[2] for r in results]

    # Check if planned mode is enabled
    planned_mode = config.get('planned_mode', False)

    if planned_mode:
        # Create PlannedMixtureDataset
        from datasets.mixture import MixtureSampler
        from datasets.sampling import DatasetSampler
        from datasets.query_builder import D4RTQueryBuilder
        from datasets.transforms import GeometryTransformPipeline

        stride_policy = _resolve_stride_policy(config)
        shuffle_frame_order = _resolve_shuffle_frame_order(config)
        shuffle_prob = _resolve_shuffle_prob(config)
        # Frame-order shuffling is now decided per clip (shuffle_prob), so motion
        # boundaries are gated per clip inside the query builder via the
        # frames_time_ordered predicate (ordered clips keep the supervision, only
        # permuted clips skip it). No global force-off needed here.
        use_motion_boundaries = config.get('use_motion_boundaries', True)
        if shuffle_prob is None and shuffle_frame_order:
            print(
                f"[factory] shuffle_frame_order={shuffle_frame_order} (uniform); "
                f"motion boundaries gated per clip via frames_time_ordered."
            )
        elif shuffle_prob is not None:
            print(f"[factory] per-dataset shuffle_prob={shuffle_prob}")

        # Build dataset samplers
        dataset_samplers = []
        for adapter, allowed_seqs in zip(adapters, allowed_per_adapter):
            ds_name = str(getattr(adapter, 'dataset_name', '')).lower()
            sampler = DatasetSampler(
                adapter=adapter,
                clip_len=config.get('clip_len', 48),
                sampling_mode=_resolve_sampling_mode(config, adapter),
                min_frames=config.get('clip_len', 48),
                allowed_sequences=allowed_seqs,
                custom_stride_range=custom_stride_range,
                sequence_locality_size=config.get('sequence_locality_size', 3),
                frame_locality_radius=config.get('frame_locality_radius', config.get('clip_len', 48)),
                stride_policy=stride_policy,
                shuffle_frame_order=shuffle_frame_order,
                shuffle_prob=(None if shuffle_prob is None else shuffle_prob.get(ds_name, 0.0)),
            )
            dataset_samplers.append(sampler)

        # Build mixture sampler
        mixture_sampler = MixtureSampler(
            samplers=dataset_samplers,
            dataset_weights=weights,
            dataset_locality_size=config.get('dataset_locality_size', 2),
        )

        # Build transform pipeline
        transform = GeometryTransformPipeline(
            img_size=config.get('img_size', 256),
            use_augs=config.get('use_augs', True) if split == 'train' else False,
            keep_cropped_images=_resolve_keep_cropped_images(config),
            color_aug_after_resize=config.get('color_aug_after_resize', False),
            max_track_points=_resolve_max_track_points(config),
        )

        # Build query builder
        query_builder = D4RTQueryBuilder(
            num_queries=config.get('num_queries', 2048),
            boundary_ratio=config.get('boundary_ratio', 0.3),
            t_tgt_eq_t_cam_ratio=config.get('t_tgt_eq_t_cam_ratio', 0.4),
            use_motion_boundaries=use_motion_boundaries,
            precompute_patches=config.get('precompute_patches', True),
            precompute_from_highres=config.get('precompute_from_highres', False),
            return_highres_video=config.get('return_highres_video', None),
            allow_track_fallback=config.get('allow_track_fallback', False),
            store_video_uint8=config.get('store_video_uint8', False),
            store_auxiliary_tensors=config.get('store_auxiliary_tensors', True),
            motion_boundary_on_resized=config.get('motion_boundary_on_resized', True),
        )

        spool_dir = config.get('spool_dir', None)
        if isinstance(spool_dir, str) and "{rank}" in spool_dir:
            spool_dir = spool_dir.format(rank=rank, world_size=world_size)

        return PlannedMixtureDataset(
            adapters=adapters,
            dataset_weights=weights,
            transform=transform,
            query_builder=query_builder,
            mixture_sampler=mixture_sampler,
            clip_len=config.get('clip_len', 48),
            seed=config.get('seed', 42),
            epoch_size=config.get('epoch_size', 10000),
            reshuffle_each_epoch=config.get('reshuffle_each_epoch', split == 'train'),
            builder_workers=config.get('builder_workers', 2),
            prefetch_depth=config.get('prefetch_depth', 32),
            spool_dir=spool_dir,
            rank=rank,
            world_size=world_size,
            max_spool_bytes=config.get('max_spool_bytes', 2 * 1024**3),
            start_immediately=config.get('planned_start_immediately', True),
            initial_epoch=config.get('planned_initial_epoch', 0),
            sample_stage_config={
                'backend': config.get('sample_stage_backend', ''),
                'stage_root': config.get('sample_stage_root', ''),
                'sdk_workers': config.get('sample_stage_sdk_workers', 8),
                'request_timeout_s': config.get('sample_stage_request_timeout_s', 20.0),
                'request_retries': config.get('sample_stage_request_retries', 1),
                'cache_max_bytes': config.get('sample_stage_cache_max_bytes', 100 * 1024**3),
                'cache_low_watermark_ratio': config.get('sample_stage_cache_low_watermark_ratio', 0.9),
                'cache_touch_interval_s': config.get('sample_stage_cache_touch_interval_s', 30.0),
                'cache_scan_interval_s': config.get('sample_stage_cache_scan_interval_s', 30.0),
                'eviction_mode': config.get('sample_stage_eviction_mode', 'background'),
                'window_radius': config.get('sample_stage_window_radius', 0),
                'mount_root': config.get('sample_stage_mount_root', '/data_cos'),
                'extra_mount_roots': config.get('sample_stage_extra_mount_roots', []),
                'bucket': config.get('sample_stage_bucket', 'hd-ai-data-1251882982'),
                'region': config.get('sample_stage_region', 'ap-beijing'),
                'passwd_file': config.get('sample_stage_passwd_file', '/etc/passwd-s3fs-data_cos'),
                'enabled_datasets': config.get(
                    'sample_stage_datasets',
                    [
                        'pointodyssey',
                        'kubric',
                        'dynamic_replica',
                        'co3dv2',
                        'blendedmvs',
                        'mvssynth',
                        'tartanair',
                        'waymo',
                    ],
                ),
                'scene_prefetch_datasets': config.get(
                    'sample_stage_scene_prefetch_datasets',
                    [],
                ),
                'pinned_manifest_root': config.get(
                    'sample_stage_pinned_manifest_root',
                    '',
                ),
            },
        )
    else:
        # Create standard MixtureDataset
        stride_policy = _resolve_stride_policy(config)
        shuffle_frame_order = _resolve_shuffle_frame_order(config)
        shuffle_prob = _resolve_shuffle_prob(config)
        # Motion boundaries are gated per clip inside the query builder via the
        # frames_time_ordered predicate, so no global force-off is needed even
        # when shuffling is enabled.
        use_motion_boundaries = config.get('use_motion_boundaries', True)
        if shuffle_prob is None and shuffle_frame_order:
            print(
                f"[factory] shuffle_frame_order={shuffle_frame_order} (uniform); "
                f"motion boundaries gated per clip via frames_time_ordered."
            )
        elif shuffle_prob is not None:
            print(f"[factory] per-dataset shuffle_prob={shuffle_prob}")
        return MixtureDataset(
            adapters=adapters,
            dataset_weights=weights,
            clip_len=config.get('clip_len', 48),
            img_size=config.get('img_size', 256),
            use_augs=config.get('use_augs', True) if split == 'train' else False,
            num_queries=config.get('num_queries', 2048),
            boundary_ratio=config.get('boundary_ratio', 0.3),
            t_tgt_eq_t_cam_ratio=config.get('t_tgt_eq_t_cam_ratio', 0.4),
            use_motion_boundaries=use_motion_boundaries,
            seed=config.get('seed', 42),
            allowed_sequences_per_adapter=allowed_per_adapter,
            sampling_mode=config.get('sampling_mode', 'stride'),
            epoch_size=config.get('epoch_size', 10000),
            custom_stride_range=custom_stride_range,
            stride_policy=stride_policy,
            shuffle_frame_order=shuffle_frame_order,
            shuffle_prob=shuffle_prob,
            precompute_patches=config.get('precompute_patches', True),
            precompute_from_highres=config.get('precompute_from_highres', False),
            return_highres_video=config.get('return_highres_video', None),
            allow_track_fallback=config.get('allow_track_fallback', False),
            store_video_uint8=config.get('store_video_uint8', False),
            store_auxiliary_tensors=config.get('store_auxiliary_tensors', True),
            keep_cropped_images=_resolve_keep_cropped_images(config),
            color_aug_after_resize=config.get('color_aug_after_resize', False),
            motion_boundary_on_resized=config.get('motion_boundary_on_resized', True),
            max_track_points=_resolve_max_track_points(config),
            reshuffle_each_epoch=config.get('reshuffle_each_epoch', split == 'train'),
        )
