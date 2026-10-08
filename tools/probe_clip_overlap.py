"""Probe per-dataset frame-to-frame overlap as a function of stride.

Outputs cov(0, 47*stride) distributions per dataset to retune
_DATASET_STRIDE_POLICY in datasets/sampling.py.

Two cov metrics:
  - cov_geom  (PRIMARY): fraction of valid pixels in frame 0 that, after
    cam0 -> world -> cam1 reprojection, land in frame 1's FOV with z1 > 0.
    Independent of frame_1 depth validity, so sparse LiDAR (waymo) and
    invalid-depth holes (scannet) do not deflate the score.
  - cov_depth (SECONDARY): same as above plus a 10% relative depth
    consistency check at the projected pixel. Lower bound from sparse
    or noisy depth; useful only when both frames have dense GT depth.

Per-sequence cov(0, 1) is stored so we can decide whether per-dataset
stride is sufficient or per-sequence speed bucketing is needed.

Usage:
    python tools/probe_clip_overlap.py \
        --config configs/mixture_5datasets_cos_planned.yaml \
        --output probe_results/full_run.json \
        --num-sequences 200 --starts-per-seq 5 \
        --stride-candidates 1,2,3,4,6,8,12,16,24
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets.registry import create_adapter  # noqa: E402

N_PROBE_PIXELS = 2048
CLIP_LEN = 48
DEPTH_TOL_REL = 0.10  # for cov_depth secondary metric


# Per-dataset adapter_kwargs override needed *only for the probe*. These keep
# probe behaviour faithful even when training defaults disable depth.
_PROBE_KWARGS_OVERRIDE: dict[str, dict] = {
    # DynamicReplica training disables depth IO when tracks are present, but
    # the probe needs depth for cov computation.
    "dynamic_replica": {"skip_depth_when_tracks": False},
}


def reproject_cov(
    depth0: np.ndarray,
    K0: np.ndarray,
    E0: np.ndarray,  # w2c
    depth1: np.ndarray,
    K1: np.ndarray,
    E1: np.ndarray,  # w2c
    n_pixels: int = N_PROBE_PIXELS,
    rng: np.random.Generator | None = None,
) -> tuple[float | None, float | None]:
    """Return (cov_geom, cov_depth).

    cov_geom : fraction of frame-0 valid pixels that re-project into
               frame 1's image rect with z1 > 0. Independent of frame-1 depth
               validity. This is the right signal for stride decisions because
               it measures "can frame 1 see this point" without being deflated
               by sparse/missing GT depth.
    cov_depth: subset of the above that *also* satisfies a 10% relative depth
               consistency check at the projected pixel. Sparse-LiDAR / black-
               border depth in frame 1 will NaN this out for many pixels even
               when the scene is genuinely co-visible -- treat it as a lower
               bound, not a primary signal.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    H, W = depth0.shape[:2]
    valid_mask = np.isfinite(depth0) & (depth0 > 0)
    if not valid_mask.any():
        return None, None

    ys, xs = np.where(valid_mask)
    if len(ys) > n_pixels:
        idx = rng.choice(len(ys), size=n_pixels, replace=False)
        ys, xs = ys[idx], xs[idx]
    z0 = depth0[ys, xs].astype(np.float64)

    K0_inv = np.linalg.inv(K0.astype(np.float64))
    pix0 = np.stack(
        [xs.astype(np.float64), ys.astype(np.float64), np.ones_like(z0)], axis=0
    )
    Xc0 = (K0_inv @ pix0) * z0[None, :]

    E0_inv = np.linalg.inv(E0.astype(np.float64))
    R0_c2w, t0_c2w = E0_inv[:3, :3], E0_inv[:3, 3:4]
    Xw = R0_c2w @ Xc0 + t0_c2w

    E1f = E1.astype(np.float64)
    R1, t1 = E1f[:3, :3], E1f[:3, 3:4]
    Xc1 = R1 @ Xw + t1
    z1 = Xc1[2]

    in_front = z1 > 1e-3
    K1f = K1.astype(np.float64)
    proj = K1f @ Xc1
    u1 = proj[0] / np.clip(proj[2], 1e-6, None)
    v1 = proj[1] / np.clip(proj[2], 1e-6, None)
    H1, W1 = depth1.shape[:2]
    in_fov = (u1 >= 0) & (u1 < W1) & (v1 >= 0) & (v1 < H1) & in_front

    n0 = max(1, len(z0))
    cov_geom = float(in_fov.sum()) / n0

    cov_depth = 0.0
    if in_fov.any():
        u_int = np.clip(np.round(u1[in_fov]).astype(int), 0, W1 - 1)
        v_int = np.clip(np.round(v1[in_fov]).astype(int), 0, H1 - 1)
        z_obs = depth1[v_int, u_int]
        z_pred = z1[in_fov]
        valid_obs = np.isfinite(z_obs) & (z_obs > 0)
        if valid_obs.any():
            rel = np.abs(z_pred[valid_obs] - z_obs[valid_obs]) / np.clip(
                z_obs[valid_obs], 1e-3, None
            )
            cov_depth = float((rel < DEPTH_TOL_REL).sum()) / n0

    return cov_geom, cov_depth


def two_frame_clip(adapter, sequence_name: str, frame_a: int, frame_b: int):
    try:
        clip = adapter.load_clip(sequence_name, [frame_a, frame_b])
    except Exception as exc:  # noqa: BLE001
        return None, f"load_clip_failed: {type(exc).__name__}: {exc}"
    if clip.depths is None:
        return None, "no_depth"
    if len(clip.depths) < 2 or clip.depths[0] is None or clip.depths[1] is None:
        return None, "depth_missing"
    return (
        clip.depths[0], clip.intrinsics[0], clip.extrinsics[0],
        clip.depths[1], clip.intrinsics[1], clip.extrinsics[1],
    ), None


def _percentiles(xs: list[float]) -> dict:
    if not xs:
        return {}
    a = np.asarray(xs)
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
    }


def probe_dataset(
    name: str,
    adapter,
    num_sequences: int,
    starts_per_seq: int,
    stride_candidates: list[int],
    rng: random.Random,
) -> dict:
    """Run cov probe over many sequences and starts."""
    seqs = adapter.list_sequences()
    if not seqs:
        return {"name": name, "error": "no_sequences"}

    rng.shuffle(seqs)
    sampled_seqs = seqs[: min(num_sequences, len(seqs))]
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))

    cov_geom_records: dict[int, list[float]] = {s: [] for s in stride_candidates}
    cov_depth_records: dict[int, list[float]] = {s: [] for s in stride_candidates}
    speed_geom: list[float] = []
    speed_depth: list[float] = []
    self_geom: list[float] = []
    per_seq_speed: dict[str, float] = {}

    fail_counters: dict[str, int] = {}
    skip_short = 0
    seq_lengths: list[int] = []

    t_start = time.time()
    n_probes = 0

    for seq_name in sampled_seqs:
        try:
            num_frames = adapter.get_num_frames(seq_name)
        except Exception as exc:  # noqa: BLE001
            key = f"num_frames:{type(exc).__name__}"
            fail_counters[key] = fail_counters.get(key, 0) + 1
            continue
        seq_lengths.append(num_frames)

        if num_frames < 1 + (CLIP_LEN - 1) * 1:
            skip_short += 1
            continue

        valid_strides = [
            s for s in stride_candidates if num_frames >= 1 + (CLIP_LEN - 1) * s
        ]

        seq_speed_samples: list[float] = []
        for _ in range(starts_per_seq):
            for stride in valid_strides:
                end_offset = (CLIP_LEN - 1) * stride
                start = rng.randint(0, num_frames - 1 - end_offset)
                pair, err = two_frame_clip(adapter, seq_name, start, start + end_offset)
                n_probes += 1
                if pair is None:
                    fail_counters[err] = fail_counters.get(err, 0) + 1
                    continue
                d0, K0, E0, d1, K1, E1 = pair
                cg, cd = reproject_cov(d0, K0, E0, d1, K1, E1, rng=np_rng)
                if cg is not None:
                    cov_geom_records[stride].append(cg)
                if cd is not None:
                    cov_depth_records[stride].append(cd)
                if cg is None:
                    fail_counters["empty_depth0"] = fail_counters.get(
                        "empty_depth0", 0) + 1

            # speed via stride=1 short hop
            if num_frames >= 2:
                start = rng.randint(0, num_frames - 2)
                pair, _ = two_frame_clip(adapter, seq_name, start, start + 1)
                if pair is not None:
                    d0, K0, E0, d1, K1, E1 = pair
                    cg, cd = reproject_cov(d0, K0, E0, d1, K1, E1, rng=np_rng)
                    if cg is not None:
                        speed_geom.append(cg)
                        seq_speed_samples.append(cg)
                    if cd is not None:
                        speed_depth.append(cd)
                    sg, _ = reproject_cov(d0, K0, E0, d0, K0, E0, rng=np_rng)
                    if sg is not None:
                        self_geom.append(sg)

        if seq_speed_samples:
            per_seq_speed[seq_name] = float(np.median(seq_speed_samples))

    elapsed = time.time() - t_start

    speed_vals = list(per_seq_speed.values())
    if speed_vals:
        speed_arr = np.asarray(speed_vals)
        speed_buckets = {
            "n_seq": int(len(speed_arr)),
            "p10": float(np.percentile(speed_arr, 10)),
            "p25": float(np.percentile(speed_arr, 25)),
            "p33": float(np.percentile(speed_arr, 33)),
            "p50": float(np.percentile(speed_arr, 50)),
            "p67": float(np.percentile(speed_arr, 67)),
            "p75": float(np.percentile(speed_arr, 75)),
            "p90": float(np.percentile(speed_arr, 90)),
            "iqr_p75_minus_p25": float(
                np.percentile(speed_arr, 75) - np.percentile(speed_arr, 25)),
        }
    else:
        speed_buckets = {}

    out = {
        "name": name,
        "num_sequences_total": len(seqs),
        "num_sequences_probed": len(sampled_seqs),
        "starts_per_seq": starts_per_seq,
        "n_probes_attempted": n_probes,
        "elapsed_s": round(elapsed, 1),
        "skip_short_seq": skip_short,
        "seq_length_stats": _percentiles([float(x) for x in seq_lengths]),
        "self_geom": _percentiles(self_geom),
        "speed_geom_stride1": _percentiles(speed_geom),
        "speed_depth_stride1": _percentiles(speed_depth),
        "per_sequence_speed": speed_buckets,
        "fail_counters": fail_counters,
        "cov_geom_by_stride": {
            str(s): _percentiles(cov_geom_records[s]) for s in stride_candidates
        },
        "cov_depth_by_stride": {
            str(s): _percentiles(cov_depth_records[s]) for s in stride_candidates
        },
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-sequences", type=int, default=200)
    parser.add_argument("--starts-per-seq", type=int, default=5)
    parser.add_argument("--stride-candidates", type=str,
                        default="1,2,3,4,6,8,12,16,24")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated whitelist. Default = all datasets in config.")
    args = parser.parse_args()

    stride_candidates = [int(x) for x in args.stride_candidates.split(",")]

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    selected = set(args.datasets.split(",")) if args.datasets else None
    rng = random.Random(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing partial results so multiple runs (with --datasets X) merge
    # rather than overwrite. Final overnight launch will run with no whitelist
    # and overwrite this file deliberately; partial sanity passes will accrete.
    if output_path.exists():
        try:
            with open(output_path) as f:
                results = json.load(f)
            if not isinstance(results, list):
                results = []
        except Exception:
            results = []
    else:
        results = []
    results_by_name = {r.get("name"): i for i, r in enumerate(results)
                       if isinstance(r, dict) and r.get("name")}

    for ds_cfg in cfg["datasets"]:
        name = ds_cfg["name"]
        if selected is not None and name not in selected:
            continue

        kwargs = dict(ds_cfg.get("adapter_kwargs", {}))
        kwargs.update(_PROBE_KWARGS_OVERRIDE.get(name, {}))
        # Reuse the training index cache so COS-mounted datasets (co3dv2,
        # scannetpp) skip the multi-minute first-time scan. Falls back to
        # whatever the adapter does if the cache is missing.
        cache_dir = cfg.get("index_cache_dir")
        if cache_dir and "cache_dir" not in kwargs:
            kwargs["cache_dir"] = cache_dir

        print(f"\n=== {name} ===", flush=True)
        try:
            adapter = create_adapter(
                name=name,
                root=ds_cfg["root"],
                split=ds_cfg.get("split", "train"),
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  adapter_create_failed: {exc}", flush=True)
            traceback.print_exc()
            entry = {"name": name, "error": f"create_adapter: {exc}"}
            if name in results_by_name:
                results[results_by_name[name]] = entry
            else:
                results_by_name[name] = len(results)
                results.append(entry)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            continue

        try:
            res = probe_dataset(
                name=name, adapter=adapter,
                num_sequences=args.num_sequences,
                starts_per_seq=args.starts_per_seq,
                stride_candidates=stride_candidates,
                rng=rng,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  probe_failed: {exc}", flush=True)
            traceback.print_exc()
            res = {"name": name, "error": f"probe: {exc}"}

        if name in results_by_name:
            results[results_by_name[name]] = res
        else:
            results_by_name[name] = len(results)
            results.append(res)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  wrote partial -> {output_path}", flush=True)

        if "cov_geom_by_stride" in res:
            print(
                f"  seq_probed={res['num_sequences_probed']}/{res['num_sequences_total']} "
                f"elapsed={res['elapsed_s']}s skip_short={res['skip_short_seq']} "
                f"n_probes={res['n_probes_attempted']}",
                flush=True,
            )
            sanity = res.get("self_geom", {}).get("p50", float("nan"))
            print(f"  self_geom p50={sanity:.3f} (expect >=0.95)", flush=True)
            speed_buckets = res.get("per_sequence_speed", {})
            if speed_buckets:
                print(
                    f"  per_seq_speed_geom n={speed_buckets.get('n_seq')} "
                    f"p25={speed_buckets.get('p25', 0):.3f} "
                    f"p50={speed_buckets.get('p50', 0):.3f} "
                    f"p75={speed_buckets.get('p75', 0):.3f} "
                    f"iqr={speed_buckets.get('iqr_p75_minus_p25', 0):.3f}",
                    flush=True,
                )
            for s_str, stats in res["cov_geom_by_stride"].items():
                if stats:
                    depth_stats = res.get("cov_depth_by_stride", {}).get(s_str, {})
                    print(
                        f"  stride={s_str:>3}  n={stats['n']:>5}  "
                        f"geom p25={stats['p25']:.3f} p50={stats['p50']:.3f} p75={stats['p75']:.3f} | "
                        f"depth p50={depth_stats.get('p50', float('nan')):.3f}",
                        flush=True,
                    )
            if res.get("fail_counters"):
                print(f"  fail_counters={res['fail_counters']}", flush=True)

    print(f"\nDone. results -> {output_path}")


if __name__ == "__main__":
    main()
