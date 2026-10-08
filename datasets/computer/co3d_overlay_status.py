#!/usr/bin/env python3
"""Summarize Co3D track overlay build progress and sample quality."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Co3D overlay build status")
    parser.add_argument("--overlay-root", required=True)
    parser.add_argument("--bad-list", required=True)
    parser.add_argument("--log", required=False, default=None)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    overlay_root = Path(args.overlay_root)
    bad_sequences = [
        line.strip() for line in Path(args.bad_list).read_text().splitlines() if line.strip()
    ]
    bad_set = set(bad_sequences)

    overlay_files = list(overlay_root.rglob("precomputed.npz"))
    overlay_sequences = sorted(
        str(p.relative_to(overlay_root).parent) for p in overlay_files
    )
    overlay_set = set(overlay_sequences)

    result: dict[str, object] = {
        "overlay_root": str(overlay_root),
        "bad_total": len(bad_sequences),
        "overlay_count": len(overlay_sequences),
        "coverage_ratio": (len(overlay_sequences) / len(bad_sequences)) if bad_sequences else 0.0,
        "unexpected_overlay_sequences": sorted(overlay_set - bad_set)[:20],
    }

    if args.log:
        log_text = Path(args.log).read_text(errors="ignore")
        result["log_fail_count"] = log_text.count("[FAIL]")

    sample = []
    rng = random.Random(args.seed)
    if overlay_sequences:
        chosen = rng.sample(overlay_sequences, min(args.sample_size, len(overlay_sequences)))
        for seq in chosen:
            with np.load(overlay_root / seq / "precomputed.npz") as z:
                stats = summarize_valids(z["valids"] if "valids" in z.files else None)
                stats["sequence"] = seq
                stats["track_source"] = str(z["track_source"].item()) if "track_source" in z.files else None
                sample.append(stats)
        result["sample"] = sample
        result["sample_mean_visible_frames_mean"] = float(
            np.mean([x["mean_visible_frames"] for x in sample])
        )
        result["sample_tracks_ge_2_ratio_mean"] = float(
            np.mean([x["tracks_ge_2_ratio"] for x in sample])
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
