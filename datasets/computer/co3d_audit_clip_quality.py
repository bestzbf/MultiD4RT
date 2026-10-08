#!/usr/bin/env python3
"""Audit Co3D clip-level track quality under the training sampler."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from datasets.adapters.co3dv2 import (
    Co3Dv2Adapter,
    _precomputed_tracks_look_temporal,
    _summarize_precomputed_track_validity,
)
from datasets.sampling import _DATASET_STRIDE_POLICY


_FAILURE_MARKER_NAME = "precomputed.failed.json"


def _deterministic_seed(sequence: str, seed: int) -> int:
    digest = hashlib.md5(f"{sequence}|{seed}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _load_valids(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as z:
        if "valids" not in z.files:
            return None
        return z["valids"]


def _sample_frame_indices(
    num_frames: int,
    clip_len: int,
    rng: random.Random,
    candidates: list[int],
    weights: list[float],
) -> list[int]:
    valid_candidates = []
    valid_weights = []
    for stride, weight in zip(candidates, weights):
        if num_frames >= 1 + (clip_len - 1) * stride:
            valid_candidates.append(stride)
            valid_weights.append(weight)

    if valid_candidates:
        total = sum(valid_weights)
        valid_probs = [w / total for w in valid_weights]
        stride = rng.choices(valid_candidates, weights=valid_probs, k=1)[0]
    else:
        stride = 1

    max_start = num_frames - (clip_len - 1) * stride
    if max_start <= 0:
        return list(range(min(clip_len, num_frames)))

    start_idx = rng.randint(0, max_start - 1)
    return [start_idx + i * stride for i in range(clip_len)]


def _inspect_sequence(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        root_str,
        overlay_root_str,
        sequence,
        num_frames,
        clip_len,
        num_samples,
        seed,
        candidates,
        weights,
        min_bad_clip_ratio,
        min_zero_clip_ratio,
    ) = args
    root = Path(root_str)
    overlay_root = Path(overlay_root_str) if overlay_root_str else None
    base_npz = root / sequence / "precomputed.npz"

    selected_label = "none"
    overlay_npz_exists = False
    overlay_marker_exists = False
    valids = None
    if overlay_root is not None:
        overlay_npz = overlay_root / sequence / "precomputed.npz"
        overlay_marker = overlay_root / sequence / _FAILURE_MARKER_NAME
        overlay_npz_exists = overlay_npz.exists()
        overlay_marker_exists = overlay_marker.exists()
        if overlay_npz_exists:
            valids = _load_valids(overlay_npz)
            selected_label = "track_precompute_root"
        elif overlay_marker_exists:
            valids = None
            selected_label = "overlay_failure_marker"

    if valids is None and selected_label != "overlay_failure_marker":
        valids = _load_valids(base_npz)
        if valids is not None:
            selected_label = "precompute_root"

    seq_stats = _summarize_precomputed_track_validity(valids)
    effective_num_frames = int(num_frames)
    if valids is not None and getattr(valids, "ndim", 0) >= 1:
        effective_num_frames = min(effective_num_frames, int(valids.shape[0]))

    rng = random.Random(_deterministic_seed(sequence, seed))
    bad = 0
    zero = 0
    clip_means: list[float] = []
    clip_ratios: list[float] = []
    for _ in range(num_samples):
        frame_indices = _sample_frame_indices(
            effective_num_frames,
            clip_len,
            rng,
            candidates,
            weights,
        )

        if valids is None:
            stats = {
                "mean_visible_frames": 0.0,
                "tracks_ge_2_ratio": 0.0,
                "tracks_ge_4_ratio": 0.0,
            }
            looks_temporal = False
        else:
            clip_valids = valids[np.asarray(frame_indices)]
            stats = _summarize_precomputed_track_validity(clip_valids)
            looks_temporal = _precomputed_tracks_look_temporal(clip_valids)

        if not looks_temporal:
            bad += 1
        if stats["mean_visible_frames"] == 0.0:
            zero += 1
        clip_means.append(float(stats["mean_visible_frames"]))
        clip_ratios.append(float(stats["tracks_ge_2_ratio"]))

    bad_clip_ratio = bad / float(num_samples)
    zero_clip_ratio = zero / float(num_samples)
    drop_recommended = (
        bad_clip_ratio >= float(min_bad_clip_ratio)
        and zero_clip_ratio >= float(min_zero_clip_ratio)
    )

    return {
        "sequence": sequence,
        "category": sequence.split("/", 1)[0],
        "selected_track_cache": selected_label,
        "overlay_npz_exists": overlay_npz_exists,
        "overlay_marker_exists": overlay_marker_exists,
        "num_frames": int(num_frames),
        "effective_num_frames_for_stats": int(effective_num_frames),
        "sequence_mean_visible_frames": float(seq_stats["mean_visible_frames"]),
        "sequence_tracks_ge_2_ratio": float(seq_stats["tracks_ge_2_ratio"]),
        "sequence_tracks_ge_4_ratio": float(seq_stats["tracks_ge_4_ratio"]),
        "sequence_looks_temporal": bool(
            _precomputed_tracks_look_temporal(valids) if valids is not None else False
        ),
        "sampled_bad_clip_ratio": float(bad_clip_ratio),
        "sampled_zero_clip_ratio": float(zero_clip_ratio),
        "sampled_clip_mean_visible_frames_median": float(np.median(clip_means)),
        "sampled_clip_mean_visible_frames_p90": float(np.percentile(clip_means, 90)),
        "sampled_clip_tracks_ge_2_ratio_median": float(np.median(clip_ratios)),
        "sampled_clip_tracks_ge_2_ratio_p90": float(np.percentile(clip_ratios, 90)),
        "drop_recommended": bool(drop_recommended),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Co3D clip-level track quality")
    parser.add_argument("--root", required=True)
    parser.add_argument("--overlay-root", default=None)
    parser.add_argument("--subset-name", default="fewview_train")
    parser.add_argument("--split", default="train")
    parser.add_argument("--require-pointcloud", action="store_true")
    parser.add_argument("--clip-len", type=int, default=48)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--min-bad-clip-ratio", type=float, default=0.9)
    parser.add_argument("--min-zero-clip-ratio", type=float, default=0.3)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--denylist-txt", required=True)
    args = parser.parse_args()

    adapter = Co3Dv2Adapter(
        root=args.root,
        subset_name=args.subset_name,
        split=args.split,
        require_pointcloud=args.require_pointcloud,
        verbose=True,
    )

    records = list(adapter._records)
    candidates, weights = _DATASET_STRIDE_POLICY["co3dv2"]
    job_args = [
        (
            args.root,
            args.overlay_root,
            r.uid,
            r.num_frames,
            args.clip_len,
            args.num_samples,
            args.seed,
            candidates,
            weights,
            args.min_bad_clip_ratio,
            args.min_zero_clip_ratio,
        )
        for r in records
    ]

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_inspect_sequence, job) for job in job_args]
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda x: x["sequence"])
    denylist = sorted(r["sequence"] for r in results if r["drop_recommended"])

    bad_clip_thresholds = [0.5, 0.75, 0.9, 0.95]
    threshold_counts = {
        str(th): sum(1 for r in results if r["sampled_bad_clip_ratio"] >= th)
        for th in bad_clip_thresholds
    }

    per_category = defaultdict(lambda: {"total": 0, "drop": 0})
    for r in results:
        row = per_category[r["category"]]
        row["total"] += 1
        row["drop"] += int(r["drop_recommended"])

    category_summary = []
    for cat, stats in per_category.items():
        category_summary.append(
            {
                "category": cat,
                "total": stats["total"],
                "drop": stats["drop"],
                "drop_ratio": (stats["drop"] / stats["total"]) if stats["total"] else 0.0,
            }
        )
    category_summary.sort(key=lambda x: (-x["drop_ratio"], -x["drop"], x["category"]))

    summary = {
        "root": args.root,
        "overlay_root": args.overlay_root,
        "subset_name": args.subset_name,
        "split": args.split,
        "require_pointcloud": args.require_pointcloud,
        "clip_len": args.clip_len,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "workers": args.workers,
        "min_bad_clip_ratio": args.min_bad_clip_ratio,
        "min_zero_clip_ratio": args.min_zero_clip_ratio,
        "total_sequences": len(results),
        "denylist_count": len(denylist),
        "threshold_counts": threshold_counts,
        "selected_track_cache_counts": {
            label: sum(1 for r in results if r["selected_track_cache"] == label)
            for label in sorted({r["selected_track_cache"] for r in results})
        },
        "drop_by_category_top20": category_summary[:20],
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    out_txt = Path(args.denylist_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("".join(seq + "\n" for seq in denylist))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
