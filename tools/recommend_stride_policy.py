"""Convert probe_clip_overlap.py results into a recommended _DATASET_STRIDE_POLICY.

Reads one or more probe JSON files (we run local + COS probes separately) and
emits a recommendation per dataset.

Recommendation rules (tunable in CLI flags):
  Target window: cov_geom p50 in [target_low, target_high].
  - Always include the largest stride whose p50 >= floor (so we get challenge).
  - Prefer strides whose p50 lies inside [target_low, target_high].
  - Avoid strides whose p50 < floor (those are essentially black holes).
  - When EVERY probed stride has p50 > target_high, use the largest probed
    stride (slow scenes) and let the user know to extend the probe.
  - Per-sequence speed IQR > 0.20 is flagged as "consider per-sequence
    speed bucketing"; we still emit a single policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_results(paths: list[Path]) -> dict:
    out: dict = {}
    for p in paths:
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        for entry in data:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not name or "cov_geom_by_stride" not in entry:
                continue
            if name in out:
                # Merge: union strides, prefer the entry with more probes.
                existing = out[name]
                merged = dict(existing)
                merged_strides = dict(existing.get("cov_geom_by_stride", {}))
                merged_strides.update(entry.get("cov_geom_by_stride", {}))
                merged["cov_geom_by_stride"] = merged_strides
                merged_depth = dict(existing.get("cov_depth_by_stride", {}))
                merged_depth.update(entry.get("cov_depth_by_stride", {}))
                merged["cov_depth_by_stride"] = merged_depth
                out[name] = merged
            else:
                out[name] = entry
    return out


def recommend_for_dataset(
    entry: dict,
    target_low: float,
    target_high: float,
    floor: float,
    min_n: int,
) -> tuple[list[int], list[float], list[str]]:
    cov_by_stride = entry.get("cov_geom_by_stride", {})
    notes: list[str] = []

    rows = []
    for s_str, stats in cov_by_stride.items():
        if not stats:
            continue
        n = stats.get("n", 0)
        if n < min_n:
            continue
        stride = int(s_str)
        rows.append((stride, stats.get("p50", 0.0), stats.get("p25", 0.0), n))
    rows.sort()

    if not rows:
        return [1], [1.0], ["no_data; falling back to stride=1"]

    in_window = [(s, p50) for s, p50, _, _ in rows if target_low <= p50 <= target_high]
    above_high = [(s, p50) for s, p50, _, _ in rows if p50 > target_high]
    below_floor = [(s, p50) for s, p50, _, _ in rows if p50 < floor]
    between_floor_and_low = [
        (s, p50) for s, p50, _, _ in rows if floor <= p50 < target_low
    ]

    candidates: list[int] = []
    weights: list[float] = []

    if in_window:
        # Use all in-window strides equally.
        for s, _ in in_window:
            candidates.append(s)
            weights.append(1.0)
    if above_high:
        # Append the largest above_high stride to bias slow scenes harder.
        s_top = max(s for s, _ in above_high)
        if s_top not in candidates:
            candidates.append(s_top)
            weights.append(0.6)
    # If still empty, pick the stride with the HIGHEST p50 (most learnable
    # signal), not the largest stride. A low-cov stride with no signal is
    # useless even if temporally diverse.
    if not candidates and between_floor_and_low:
        s_top, _ = max(between_floor_and_low, key=lambda x: x[1])
        candidates.append(s_top)
        weights.append(1.0)
    if not candidates:
        # Last resort: use the stride with the highest p50 even if below floor.
        s_top, p_top = max(rows, key=lambda r: r[1])[:2]
        candidates.append(s_top)
        weights.append(1.0)
        notes.append(
            f"no stride reaches floor={floor:.2f}; clipping to s={s_top} (p50={p_top:.2f})"
        )

    # If only one stride and it's in_window, no need for an additional anchor.
    # If we have in_window strides AND above_high strides, that's a healthy
    # speed range. If we have ONLY above_high (all p50 > target_high), suggest
    # extending the probe.
    if not in_window and above_high and not below_floor:
        s_top = max(s for s, _ in above_high)
        notes.append(
            f"all probed strides cov > {target_high}; max probed stride={s_top}"
            f" still cov={dict(above_high)[s_top]:.2f}; consider extending probe"
        )

    # Sort candidates ascending and renormalize.
    pairs = sorted(zip(candidates, weights))
    candidates = [s for s, _ in pairs]
    weights = [w for _, w in pairs]
    total = sum(weights)
    weights = [round(w / total, 3) for w in weights]
    drift = round(1.0 - sum(weights), 3)
    if drift != 0:
        weights[0] = round(weights[0] + drift, 3)

    speed = entry.get("per_sequence_speed", {})
    iqr = speed.get("iqr_p75_minus_p25", 0.0) or 0.0
    if iqr >= 0.20:
        notes.append(
            f"per-seq cov(0,1) IQR={iqr:.3f}: bimodal; consider speed-bucketed stride"
        )

    return candidates, weights, notes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="probe_results/*.json")
    parser.add_argument("--target-low", type=float, default=0.30)
    parser.add_argument("--target-high", type=float, default=0.70)
    parser.add_argument("--floor", type=float, default=0.10,
                        help="Below this cov_p50, the stride is treated as a black hole.")
    parser.add_argument("--min-n", type=int, default=20,
                        help="Skip strides with fewer than this many probes (low confidence).")
    args = parser.parse_args()

    merged = merge_results([Path(p) for p in args.inputs])
    if not merged:
        print("No data loaded.")
        return

    print(
        f"# Recommended _DATASET_STRIDE_POLICY  "
        f"target p50 in [{args.target_low}, {args.target_high}], floor={args.floor}"
    )
    print()
    print("_DATASET_STRIDE_POLICY: dict[str, tuple[list[int], list[float]]] = {")
    for name in sorted(merged):
        entry = merged[name]
        cands, ws, notes = recommend_for_dataset(
            entry,
            args.target_low,
            args.target_high,
            args.floor,
            args.min_n,
        )
        comment = ""
        if notes:
            comment = "    # " + "; ".join(notes)
        print(f"    {name!r:>22}: ({cands}, {ws}),{comment}")
    print("}")
    print()

    # All strides we probed across any dataset
    all_strides = sorted({
        int(s) for entry in merged.values()
        for s in entry.get("cov_geom_by_stride", {}).keys()
    })
    print("# --- per-dataset cov_geom p50 by stride ---")
    header_strides = " ".join(f"s{s:>2}" for s in all_strides)
    print(f"{'name':<22} {'iqr':>6} {header_strides}")
    for name in sorted(merged):
        entry = merged[name]
        speed = entry.get("per_sequence_speed", {})
        iqr = speed.get("iqr_p75_minus_p25", 0.0) or 0.0
        row = []
        for s in all_strides:
            stats = entry.get("cov_geom_by_stride", {}).get(str(s), {})
            v = stats.get("p50") if stats else None
            n = stats.get("n", 0) if stats else 0
            if v is None or n < args.min_n:
                row.append(" - ")
            else:
                row.append(f"{v:.2f}")
        print(f"{name:<22} {iqr:>6.3f} " + " ".join(f"{v:>4}" for v in row))


if __name__ == "__main__":
    main()
