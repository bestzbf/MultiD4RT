#!/usr/bin/env python3
"""Audit Co3D precomputed track caches for temporal degeneracy."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm


def summarize_valids(valids: np.ndarray | None) -> dict[str, float]:
    if valids is None:
        return {
            "mean_visible_frames": 0.0,
            "tracks_ge_2_ratio": 0.0,
            "tracks_ge_4_ratio": 0.0,
        }

    valids_np = np.asarray(valids, dtype=bool)
    if valids_np.ndim != 2 or valids_np.shape[1] == 0:
        return {
            "mean_visible_frames": 0.0,
            "tracks_ge_2_ratio": 0.0,
            "tracks_ge_4_ratio": 0.0,
        }

    per_track_visible = valids_np.sum(axis=0)
    return {
        "mean_visible_frames": float(per_track_visible.mean()),
        "tracks_ge_2_ratio": float((per_track_visible >= 2).mean()),
        "tracks_ge_4_ratio": float((per_track_visible >= 4).mean()),
    }


def looks_temporal(
    stats: dict[str, float],
    *,
    min_track_ge_2_ratio: float,
    min_mean_visible_frames: float,
) -> bool:
    return (
        stats["tracks_ge_2_ratio"] >= float(min_track_ge_2_ratio)
        and stats["mean_visible_frames"] >= float(min_mean_visible_frames)
    )


def inspect_sequence(args: tuple[str, str, float, float]) -> dict:
    root, sequence, min_track_ge_2_ratio, min_mean_visible_frames = args
    npz_path = Path(root) / sequence / "precomputed.npz"
    result = {
        "sequence": sequence,
        "path": str(npz_path),
        "exists": npz_path.exists(),
    }
    if not npz_path.exists():
        result["error"] = "missing_precomputed"
        return result

    try:
        with np.load(npz_path) as z:
            valids = z["valids"] if "valids" in z.files else None
            stats = summarize_valids(valids)
            result.update(stats)
            result["has_track_source"] = "track_source" in z.files
            result["looks_temporal"] = looks_temporal(
                stats,
                min_track_ge_2_ratio=min_track_ge_2_ratio,
                min_mean_visible_frames=min_mean_visible_frames,
            )
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Co3D precomputed caches")
    parser.add_argument("--root", required=True, help="Co3D dataset root")
    parser.add_argument("--sequence-list", default=None,
                        help="Optional txt/json file with one sequence per line")
    parser.add_argument("--output-json", required=True, help="Where to write full audit JSON")
    parser.add_argument("--bad-list-txt", required=True, help="Where to write bad sequence list")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--min-track-ge2-ratio", type=float, default=0.05)
    parser.add_argument("--min-mean-visible-frames", type=float, default=1.5)
    args = parser.parse_args()

    from datasets.adapters.co3dv2 import Co3Dv2Adapter

    if args.sequence_list:
        seq_path = Path(args.sequence_list)
        if seq_path.suffix.lower() == ".json":
            raw = json.loads(seq_path.read_text())
            if isinstance(raw, dict):
                sequences = []
                for value in raw.values():
                    if isinstance(value, list):
                        sequences.extend(value)
            else:
                sequences = list(raw)
        else:
            sequences = [line.strip() for line in seq_path.read_text().splitlines() if line.strip()]
    else:
        adapter = Co3Dv2Adapter(root=args.root, subset_name="fewview_train", split="train", verbose=True)
        sequences = adapter.list_sequences()

    job_args = [
        (args.root, seq, args.min_track_ge2_ratio, args.min_mean_visible_frames)
        for seq in sequences
    ]

    results = []
    bad_sequences = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(inspect_sequence, job) for job in job_args]
        with tqdm(total=len(futures), desc="co3d_audit") as pbar:
            for fut in as_completed(futures):
                result = fut.result()
                results.append(result)
                if (not result.get("looks_temporal", False)) or result.get("error"):
                    bad_sequences.append(result["sequence"])
                pbar.update(1)

    results.sort(key=lambda x: x["sequence"])
    bad_sequences = sorted(set(bad_sequences))

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))

    out_txt = Path(args.bad_list_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("".join(seq + "\n" for seq in bad_sequences))

    good = [r for r in results if r.get("looks_temporal", False) and not r.get("error")]
    bad = [r for r in results if (not r.get("looks_temporal", False)) and not r.get("error")]
    print(
        json.dumps(
            {
                "total": len(results),
                "good": len(good),
                "bad": len(bad),
                "errors": sum(1 for r in results if r.get("error")),
                "bad_ratio": (len(bad) / len(results)) if results else 0.0,
                "output_json": str(out_json),
                "bad_list_txt": str(out_txt),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
