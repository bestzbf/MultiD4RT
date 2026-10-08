"""Benchmark whether per-sequence local materialization is viable.

This script builds a normal adapter on the source root, materializes a single
sequence/scene into a tiny local dataset root, builds a second adapter on that
mini root, and compares load_clip timings.

It is intended as a low-risk prototype for COS-backed training:
1. Keep the training code untouched.
2. Verify that "copy one unit locally, then read locally" works with the
   existing adapters.
3. Estimate break-even reuse count for a cache-backed design.

Example:
    python datasets/tests/benchmark_sequence_cache_feasibility.py \
      --config configs/mixture_5datasets_blendedmvs_hdu.yaml \
      --dataset blendedmvs \
      --cache-dir /tmp/d4rt_seq_cache_proto \
      --num-clips 4 \
      --clip-len 48 \
      --rebuild
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from datasets.registry import create_adapter
from datasets.sequence_cache import DiskSequenceCache

try:
    from datasets.adapters.co3dv2 import _SUBSET_MAP
except Exception:  # pragma: no cover - fallback if adapter import changes
    _SUBSET_MAP = {
        "fewview_train": ("fewview_train", "train"),
        "fewview_dev": ("fewview_dev", "val"),
        "fewview_test": ("fewview_test", "test"),
        "manyview_dev_0": ("manyview_dev_0", "val"),
        "manyview_dev_1": ("manyview_dev_1", "val"),
        "manyview_test_0": ("manyview_test_0", "test"),
    }


SUPPORTED_DATASETS = {
    "kubric",
    "blendedmvs",
    "pointodyssey",
    "dynamic_replica",
    "co3dv2",
}


def _copy_file(src: Path, dst: Path) -> None:
    import shutil

    if not src.exists():
        raise FileNotFoundError(f"Required file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_jgz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _write_jgz(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)


def _sanitize_name(value: str) -> str:
    return value.replace("/", "__")


def _resolve_dataset_config(config: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    for ds_config in config.get("datasets", []):
        if ds_config.get("name") == dataset_name:
            return ds_config
    raise KeyError(f"Dataset {dataset_name!r} not found in config")


def _resolve_split(ds_config: dict[str, Any], requested_split: str) -> str:
    if requested_split == "val":
        return ds_config.get("val_split", ds_config.get("split", "val"))
    return ds_config.get("split", requested_split)


def _build_source_adapter(
    *,
    config: dict[str, Any],
    ds_config: dict[str, Any],
    split: str,
) -> tuple[Any, float]:
    kwargs = {
        "name": ds_config["name"],
        "root": ds_config["root"],
        "split": split,
        **ds_config.get("adapter_kwargs", {}),
    }
    if config.get("index_cache_dir"):
        kwargs["cache_dir"] = config["index_cache_dir"]
    if config.get("index_workers") is not None:
        kwargs["index_workers"] = config["index_workers"]

    start = time.perf_counter()
    adapter = create_adapter(**kwargs)
    return adapter, time.perf_counter() - start


def _pick_sequence(
    adapter: Any,
    *,
    clip_len: int,
    requested_sequence: Optional[str] = None,
) -> str:
    if requested_sequence is not None:
        n = adapter.get_num_frames(requested_sequence)
        if n < clip_len:
            raise ValueError(
                f"Requested sequence {requested_sequence!r} has only {n} frames < clip_len={clip_len}"
            )
        return requested_sequence

    for sequence_name in adapter.list_sequences():
        try:
            if adapter.get_num_frames(sequence_name) >= clip_len:
                return sequence_name
        except Exception:
            continue
    raise RuntimeError(f"No sequence with >= {clip_len} frames was found")


def _build_clip_indices(num_frames: int, clip_len: int, num_clips: int) -> list[list[int]]:
    if num_frames < clip_len:
        raise ValueError(f"num_frames={num_frames} < clip_len={clip_len}")

    if num_frames == clip_len:
        return [list(range(clip_len))]

    max_start = num_frames - clip_len
    if num_clips <= 1:
        starts = [0]
    else:
        starts = []
        for i in range(num_clips):
            frac = i / max(1, num_clips - 1)
            starts.append(int(round(frac * max_start)))

    unique_starts: list[int] = []
    seen = set()
    for start in starts:
        if start not in seen:
            unique_starts.append(start)
            seen.add(start)

    return [list(range(start, start + clip_len)) for start in unique_starts]


def _benchmark_loads(adapter: Any, sequence_name: str, clips: list[list[int]]) -> list[float]:
    times: list[float] = []
    for frame_indices in clips:
        start = time.perf_counter()
        clip = adapter.load_clip(sequence_name, frame_indices)
        # Touch a couple of fields so lazy paths are forced.
        _ = clip.num_frames
        _ = clip.images[0].shape
        times.append(time.perf_counter() - start)
    return times


def _compare_optional_lists(
    lhs: Optional[list[np.ndarray]],
    rhs: Optional[list[np.ndarray]],
    *,
    atol: float = 1e-5,
) -> tuple[bool, str]:
    if lhs is None and rhs is None:
        return True, ""
    if (lhs is None) != (rhs is None):
        return False, "one side is None"
    assert lhs is not None and rhs is not None
    if len(lhs) != len(rhs):
        return False, f"length mismatch: {len(lhs)} != {len(rhs)}"
    for idx, (a, b) in enumerate(zip(lhs, rhs)):
        if a.shape != b.shape:
            return False, f"shape mismatch at index {idx}: {a.shape} != {b.shape}"
        if not np.allclose(a, b, atol=atol, equal_nan=True):
            return False, f"value mismatch at index {idx}"
    return True, ""


def _compare_optional_arrays(
    lhs: Optional[np.ndarray],
    rhs: Optional[np.ndarray],
    *,
    atol: float = 1e-5,
) -> tuple[bool, str]:
    if lhs is None and rhs is None:
        return True, ""
    if (lhs is None) != (rhs is None):
        return False, "one side is None"
    assert lhs is not None and rhs is not None
    if lhs.shape != rhs.shape:
        return False, f"shape mismatch: {lhs.shape} != {rhs.shape}"
    if not np.allclose(lhs, rhs, atol=atol, equal_nan=True):
        return False, "value mismatch"
    return True, ""


def _verify_same_clip(source_clip: Any, local_clip: Any) -> list[str]:
    issues: list[str] = []
    if source_clip.sequence_name != local_clip.sequence_name:
        issues.append(
            f"sequence_name mismatch: {source_clip.sequence_name!r} != {local_clip.sequence_name!r}"
        )

    checks = [
        ("images", _compare_optional_lists(source_clip.images, local_clip.images, atol=0.0)),
        ("depths", _compare_optional_lists(source_clip.depths, local_clip.depths)),
        ("normals", _compare_optional_lists(source_clip.normals, local_clip.normals)),
        ("trajs_2d", _compare_optional_arrays(source_clip.trajs_2d, local_clip.trajs_2d)),
        ("trajs_3d_world", _compare_optional_arrays(source_clip.trajs_3d_world, local_clip.trajs_3d_world)),
        ("valids", _compare_optional_arrays(source_clip.valids, local_clip.valids, atol=0.0)),
        ("visibs", _compare_optional_arrays(source_clip.visibs, local_clip.visibs, atol=0.0)),
        ("intrinsics", _compare_optional_arrays(source_clip.intrinsics, local_clip.intrinsics)),
        ("extrinsics", _compare_optional_arrays(source_clip.extrinsics, local_clip.extrinsics)),
    ]
    for name, (ok, reason) in checks:
        if not ok:
            issues.append(f"{name}: {reason}")
    return issues


def _materialize_optional_external_root(
    *,
    local_root: Path,
    key_name: str,
    source_external_root: Optional[str],
    relative_key: str,
    force: bool,
) -> Optional[str]:
    if source_external_root is None:
        return None

    source_root = Path(source_external_root)
    local_external_root = local_root / f".{key_name}"
    cache = DiskSequenceCache(local_external_root)
    cache.materialize_tree(relative_key, source_root / relative_key, force=force)
    return str(local_external_root)


def _prepare_local_root(
    *,
    dataset_name: str,
    source_root: Path,
    source_adapter: Any,
    sequence_name: str,
    split: str,
    adapter_kwargs: dict[str, Any],
    local_root: Path,
    force: bool,
) -> tuple[str, dict[str, Any], list[str], Any]:
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset {dataset_name!r}. Supported: {sorted(SUPPORTED_DATASETS)}"
        )

    if force and local_root.exists():
        import shutil

        shutil.rmtree(local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    cache = DiskSequenceCache(local_root)
    local_kwargs = dict(adapter_kwargs)
    notes: list[str] = []

    if dataset_name == "kubric":
        source_dir = source_root / sequence_name
        if not source_dir.exists():
            source_dir = Path(source_adapter.get_sequence_info(sequence_name)["path"])
        result = cache.materialize_tree(sequence_name, source_dir, force=force)
        notes.append(
            "Kubric mini-root is loaded with split='val' because the adapter always "
            "reserves at least one scene for validation."
        )
        return "val", local_kwargs, notes, result

    if dataset_name == "blendedmvs":
        result = cache.materialize_tree(sequence_name, source_root / sequence_name, force=force)
        split_key = split.lower()
        if split_key in {"val", "valid", "validation"}:
            list_name = "validation_list.txt"
        else:
            list_name = "BlendedMVS_training.txt"
        _write_text(local_root / list_name, f"{sequence_name}\n")

        source_precompute = local_kwargs.get("precompute_root")
        if source_precompute is not None:
            if Path(source_precompute).resolve() == source_root.resolve():
                local_kwargs["precompute_root"] = str(local_root)
            else:
                local_kwargs["precompute_root"] = _materialize_optional_external_root(
                    local_root=local_root,
                    key_name="precompute_root",
                    source_external_root=source_precompute,
                    relative_key=sequence_name,
                    force=force,
                )
        return split, local_kwargs, notes, result

    if dataset_name == "pointodyssey":
        relative_key = f"{split}/{sequence_name}"
        result = cache.materialize_tree(relative_key, source_root / relative_key, force=force)
        source_fast_root = local_kwargs.get("fast_root")
        if source_fast_root is not None:
            fast_root_path = Path(source_fast_root)
            if fast_root_path.resolve() == source_root.resolve():
                local_kwargs["fast_root"] = str(local_root)
            else:
                local_fast_root = local_root / ".fast_root"
                fast_cache = DiskSequenceCache(local_fast_root)
                fast_cache.materialize_tree(
                    relative_key,
                    fast_root_path / relative_key,
                    force=force,
                )
                local_kwargs["fast_root"] = str(local_fast_root)
        return split, local_kwargs, notes, result

    if dataset_name == "dynamic_replica":
        relative_key = f"{split}/{sequence_name}"
        result = cache.materialize_tree(relative_key, source_root / relative_key, force=force)
        anno_name = f"frame_annotations_{split}.jgz"
        _copy_file(source_root / split / anno_name, local_root / split / anno_name)
        notes.append(
            "Dynamic_Replica still needs split-level metadata local. The prototype "
            "copies frame_annotations_<split>.jgz alongside the sequence directory."
        )
        return split, local_kwargs, notes, result

    if dataset_name == "co3dv2":
        category, bare_sequence = sequence_name.split("/", 1)
        relative_key = f"{category}/{bare_sequence}"
        result = cache.materialize_tree(relative_key, source_root / relative_key, force=force)

        category_root = local_root / category
        original_category_root = source_root / category

        frame_ann_path = original_category_root / "frame_annotations.jgz"
        sequence_ann_path = original_category_root / "sequence_annotations.jgz"
        set_lists_dir = original_category_root / "set_lists"

        frame_ann = _read_jgz(frame_ann_path)
        sequence_ann = _read_jgz(sequence_ann_path)
        filtered_frame_ann = [
            entry for entry in frame_ann if str(entry.get("sequence_name")) == bare_sequence
        ]
        filtered_sequence_ann = [
            entry for entry in sequence_ann if str(entry.get("sequence_name")) == bare_sequence
        ]
        _write_jgz(category_root / "frame_annotations.jgz", filtered_frame_ann)
        _write_jgz(category_root / "sequence_annotations.jgz", filtered_sequence_ann)

        subset_name = str(local_kwargs.get("subset_name", "fewview_train"))
        if subset_name not in _SUBSET_MAP:
            raise KeyError(f"Unknown Co3Dv2 subset_name: {subset_name}")
        file_suffix, _split_key = _SUBSET_MAP[subset_name]
        original_set_lists_path = set_lists_dir / f"set_lists_{file_suffix}.json"
        with open(original_set_lists_path, "r", encoding="utf-8") as f:
            set_lists = json.load(f)
        filtered_set_lists = {
            key: [entry for entry in entries if str(entry[0]) == bare_sequence]
            for key, entries in set_lists.items()
        }
        filtered_set_path = category_root / "set_lists" / f"set_lists_{file_suffix}.json"
        filtered_set_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_set_path.write_text(
            json.dumps(filtered_set_lists),
            encoding="utf-8",
        )

        local_kwargs["categories"] = [category]
        local_kwargs["sequence_allowlist"] = [sequence_name]

        source_precompute = local_kwargs.get("precompute_root")
        if source_precompute is not None:
            if Path(source_precompute).resolve() == source_root.resolve():
                local_kwargs["precompute_root"] = str(local_root)
            else:
                local_kwargs["precompute_root"] = _materialize_optional_external_root(
                    local_root=local_root,
                    key_name="precompute_root",
                    source_external_root=source_precompute,
                    relative_key=sequence_name,
                    force=force,
                )

        source_track_precompute = local_kwargs.get("track_precompute_root")
        if source_track_precompute is not None:
            if Path(source_track_precompute).resolve() == source_root.resolve():
                local_kwargs["track_precompute_root"] = str(local_root)
            else:
                local_kwargs["track_precompute_root"] = _materialize_optional_external_root(
                    local_root=local_root,
                    key_name="track_precompute_root",
                    source_external_root=source_track_precompute,
                    relative_key=sequence_name,
                    force=force,
                )

        notes.append(
            "Co3Dv2 needs category-level metadata local. The prototype writes "
            "filtered frame_annotations/sequence_annotations/set_lists files."
        )
        return split, local_kwargs, notes, result

    raise AssertionError(f"Unhandled dataset: {dataset_name}")


def _build_local_adapter(
    *,
    dataset_name: str,
    local_root: Path,
    split: str,
    adapter_kwargs: dict[str, Any],
) -> tuple[Any, float]:
    kwargs = {
        "name": dataset_name,
        "root": str(local_root),
        "split": split,
        **adapter_kwargs,
        "cache_dir": str(local_root / ".index_cache"),
        "index_workers": 1,
    }
    start = time.perf_counter()
    adapter = create_adapter(**kwargs)
    return adapter, time.perf_counter() - start


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, required=True)
    parser.add_argument("--clip-len", type=int, default=None)
    parser.add_argument("--num-clips", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ds_config = _resolve_dataset_config(config, args.dataset)
    split = _resolve_split(ds_config, args.split)
    clip_len = int(args.clip_len or config.get("clip_len", 48))
    cache_dir = Path(args.cache_dir)

    print("=" * 80)
    print("Sequence Cache Feasibility Benchmark")
    print("=" * 80)
    print(f"config      : {args.config}")
    print(f"dataset     : {args.dataset}")
    print(f"source_root : {ds_config['root']}")
    print(f"split       : {split}")
    print(f"clip_len    : {clip_len}")
    print(f"num_clips   : {args.num_clips}")
    print(f"cache_dir   : {cache_dir}")
    print()

    print("Building source adapter...")
    source_adapter, source_init_s = _build_source_adapter(
        config=config,
        ds_config=ds_config,
        split=split,
    )
    print(f"  source adapter init: {source_init_s:.3f}s")
    print(f"  sequences          : {len(source_adapter)}")

    sequence_name = _pick_sequence(
        source_adapter,
        clip_len=clip_len,
        requested_sequence=args.sequence,
    )
    num_frames = source_adapter.get_num_frames(sequence_name)
    clips = _build_clip_indices(num_frames, clip_len, args.num_clips)

    print(f"Selected sequence: {sequence_name}")
    print(f"Sequence frames  : {num_frames}")
    print(f"Clip count       : {len(clips)}")
    if clips:
        print(f"First clip idx   : [{clips[0][0]}, ..., {clips[0][-1]}]")
    print()

    print("Benchmarking source root load_clip...")
    source_times = _benchmark_loads(source_adapter, sequence_name, clips)
    print("  source load times (s):", ", ".join(f"{t:.3f}" for t in source_times))
    print(f"  source mean          : {_mean(source_times):.3f}s/clip")
    print()

    source_root = Path(ds_config["root"])
    local_root = cache_dir / "mini_roots" / args.dataset / _sanitize_name(sequence_name)
    local_split, local_kwargs, notes, materialize_result = _prepare_local_root(
        dataset_name=args.dataset,
        source_root=source_root,
        source_adapter=source_adapter,
        sequence_name=sequence_name,
        split=split,
        adapter_kwargs=ds_config.get("adapter_kwargs", {}),
        local_root=local_root,
        force=args.rebuild,
    )

    print("Building local mini-root adapter...")
    local_adapter, local_init_s = _build_local_adapter(
        dataset_name=args.dataset,
        local_root=local_root,
        split=local_split,
        adapter_kwargs=local_kwargs,
    )
    print(f"  local adapter init: {local_init_s:.3f}s")
    print(f"  local sequences   : {len(local_adapter)}")
    print()

    print("Benchmarking local mini-root load_clip...")
    local_times = _benchmark_loads(local_adapter, sequence_name, clips)
    print("  local load times (s):", ", ".join(f"{t:.3f}" for t in local_times))
    print(f"  local mean          : {_mean(local_times):.3f}s/clip")
    if len(local_times) > 1:
        print(f"  local hot mean      : {_mean(local_times[1:]):.3f}s/clip")
    print()

    print("Verifying one clip matches byte-for-byte / numerically...")
    source_clip = source_adapter.load_clip(sequence_name, clips[0])
    local_clip = local_adapter.load_clip(sequence_name, clips[0])
    issues = _verify_same_clip(source_clip, local_clip)
    if issues:
        print("  verification: FAILED")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  verification: OK")
    print()

    source_mean = _mean(source_times)
    local_hot_mean = _mean(local_times[1:]) if len(local_times) > 1 else _mean(local_times)
    saved_per_clip = source_mean - local_hot_mean
    if saved_per_clip > 0:
        break_even_clips = materialize_result.elapsed_s / saved_per_clip
    else:
        break_even_clips = float("inf")

    summary = {
        "config": str(args.config),
        "dataset": args.dataset,
        "split": split,
        "sequence_name": sequence_name,
        "clip_len": clip_len,
        "num_frames": num_frames,
        "num_clips": len(clips),
        "source_root": str(source_root),
        "local_root": str(local_root),
        "source_adapter_init_s": source_init_s,
        "local_adapter_init_s": local_init_s,
        "materialize_elapsed_s": materialize_result.elapsed_s,
        "materialize_reused": materialize_result.reused,
        "materialized_size_bytes": materialize_result.size_bytes,
        "source_times_s": source_times,
        "local_times_s": local_times,
        "source_mean_s": source_mean,
        "local_hot_mean_s": local_hot_mean,
        "saved_per_clip_s": saved_per_clip,
        "break_even_clips": break_even_clips,
        "verification_ok": len(issues) == 0,
        "verification_issues": issues,
        "notes": notes,
    }

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.json_out is not None:
        json_out = Path(args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote summary to {json_out}")


if __name__ == "__main__":
    main()
