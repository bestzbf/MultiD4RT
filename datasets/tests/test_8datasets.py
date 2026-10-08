#!/usr/bin/env python3
"""
8-dataset mixed training validation + per-dataset loading speed benchmark.

Tests:
  1. Each adapter initializes correctly
  2. load_clip() works and returns valid UnifiedClip
  3. Full pipeline (transforms + query_builder) produces correct QuerySample
  4. MixtureDataset mixes all 8 datasets
  5. Per-dataset I/O speed breakdown (load_clip / transform / query_builder)

Run:
    cd /data2/d4rt/code
    conda activate d4rt
    python datasets/tests/test_8datasets.py
"""

import sys
import time
import traceback
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from datasets.adapters.pointodyssey   import PointOdysseyAdapter
from datasets.adapters.kubric         import KubricAdapter
from datasets.adapters.dynamic_replica import DynamicReplicaAdapter
from datasets.adapters.scannet        import ScanNetAdapter
from datasets.adapters.blendedmvs     import BlendedMVSAdapter
from datasets.adapters.mvssynth       import MVSSynthAdapter
from datasets.adapters.TartanAir      import TartanAirAdapter
from datasets.adapters.VirtualKitti   import VKITTI2Adapter
from datasets.transforms              import GeometryTransformPipeline
from datasets.query_builder           import D4RTQueryBuilder
from datasets.mixture                 import MixtureDataset

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CLIP_LEN        = 8
IMG_SIZE        = 256
NUM_QUERIES     = 512   # smaller for speed test; paper uses 2048
N_SAMPLES       = 5     # clips to time per dataset
SLOW_THRESHOLD  = 2.0   # seconds; datasets above this flagged for H5

DATASET_CONFIGS = [
    dict(
        name        = "pointodyssey",
        cls         = PointOdysseyAdapter,
        kwargs      = dict(root="/data2/d4rt/datasets/PointOdyssey", split="train", verbose=False),
        has_tracks  = True,
    ),
    dict(
        name        = "kubric",
        cls         = KubricAdapter,
        kwargs      = dict(root="/data2/d4rt/datasets/kubric"),
        has_tracks  = True,
    ),
    dict(
        name        = "dynamic_replica",
        cls         = DynamicReplicaAdapter,
        kwargs      = dict(root="/data1/d4rt/datasets/Dynamic_Replica", split="train",
                           load_trajectories=True, verbose=False),
        has_tracks  = True,   # left-camera only
    ),
    dict(
        name        = "scannet",
        cls         = ScanNetAdapter,
        kwargs      = dict(root="/data2/d4rt/datasets/scannet/scannet",
                           precompute_root="/data2/d4rt/datasets/scannet/scannet"),
        has_tracks  = False,
    ),
    dict(
        name        = "blendedmvs",
        cls         = BlendedMVSAdapter,
        kwargs      = dict(root="/data2/d4rt/datasets/BlendedMVS", split="train",
                           precompute_root="/data2/d4rt/datasets/BlendedMVS", verbose=False),
        has_tracks  = False,
    ),
    dict(
        name        = "mvssynth",
        cls         = MVSSynthAdapter,
        kwargs      = dict(root="/data2/d4rt/datasets/MVS-Synth/GTAV_1080",
                           precompute_root="/data2/d4rt/datasets/MVS-Synth/GTAV_1080", verbose=False),
        has_tracks  = False,
    ),
    dict(
        name        = "tartanair",
        cls         = TartanAirAdapter,
        kwargs      = dict(root="/data2/d4rt/datasets/TartanAir", camera="left",
                           precompute_root="/data2/d4rt/datasets/TartanAir", verbose=False),
        has_tracks  = False,
    ),
    dict(
        name        = "vkitti2",
        cls         = VKITTI2Adapter,
        kwargs      = dict(root="/data2/d4rt/datasets/VirtualKitti",
                           precompute_root="/data2/d4rt/datasets/VirtualKitti", verbose=False),
        has_tracks  = False,
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def fmt(v):
    return f"{v:.3f}s"

def check_unified_clip(clip, name):
    """Basic shape / content checks on a UnifiedClip."""
    issues = []
    T = clip.num_frames
    if T < CLIP_LEN:
        issues.append(f"only {T} frames (expected {CLIP_LEN})")
    for i, img in enumerate(clip.images):
        if img.ndim != 3 or img.shape[2] != 3:
            issues.append(f"images[{i}] shape {img.shape} bad")
            break
    if clip.intrinsics.shape != (T, 3, 3):
        issues.append(f"intrinsics shape {clip.intrinsics.shape}")
    if clip.extrinsics.shape != (T, 4, 4):
        issues.append(f"extrinsics shape {clip.extrinsics.shape}")
    has_tracks = clip.metadata.get("has_tracks", False)
    if has_tracks:
        if clip.trajs_2d is None:
            issues.append("has_tracks=True but trajs_2d is None")
        if clip.trajs_3d_world is None:
            issues.append("has_tracks=True but trajs_3d_world is None")
    else:
        if clip.depths is None:
            issues.append("has_tracks=False and depths is None (mask_3d will be all False)")
    return issues

def check_query_sample(sample, name):
    """Check QuerySample shapes and mask ratios."""
    issues = []
    Q = NUM_QUERIES
    S = CLIP_LEN
    checks = {
        "video":        (S, 3, IMG_SIZE, IMG_SIZE),
        "coords":       (Q, 2),
        "t_src":        (Q,),
        "t_tgt":        (Q,),
        "t_cam":        (Q,),
        "intrinsics":   (S, 3, 3),
        "extrinsics":   (S, 4, 4),
    }
    for attr, expected in checks.items():
        val = getattr(sample, attr)
        if tuple(val.shape) != expected:
            issues.append(f"{attr}: {tuple(val.shape)} != {expected}")

    tgt = sample.targets
    for k in ["pos_2d", "pos_3d", "visibility", "displacement", "normal",
              "mask_3d", "mask_2d", "mask_vis", "mask_disp", "mask_normal"]:
        if k not in tgt:
            issues.append(f"targets missing '{k}'")

    mask_stats = {
        "mask_3d":     float(tgt["mask_3d"].float().mean()),
        "mask_2d":     float(tgt["mask_2d"].float().mean()),
        "mask_vis":    float(tgt["mask_vis"].float().mean()),
        "mask_disp":   float(tgt["mask_disp"].float().mean()),
        "mask_normal": float(tgt["mask_normal"].float().mean()),
        "boundary":    float(tgt["source_is_boundary"].float().mean()),
    }
    return issues, mask_stats

# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset test
# ─────────────────────────────────────────────────────────────────────────────
transform = GeometryTransformPipeline(img_size=IMG_SIZE, use_augs=False)
qbuilder  = D4RTQueryBuilder(num_queries=NUM_QUERIES, precompute_patches=False)

results = {}

print("=" * 72)
print("D4RT  8-Dataset  Validation + Speed Benchmark")
print(f"  clip_len={CLIP_LEN}  img_size={IMG_SIZE}  num_queries={NUM_QUERIES}  N={N_SAMPLES}")
print("=" * 72)

adapters_ok = []

for cfg in DATASET_CONFIGS:
    name = cfg["name"]
    print(f"\n{'─'*60}")
    print(f"[{name}]")

    r = {
        "init_ok": False,
        "clip_ok": False,
        "pipeline_ok": False,
        "num_sequences": 0,
        "t_load":  [],   # load_clip seconds per clip
        "t_xform": [],   # transform seconds per clip
        "t_query": [],   # query_builder seconds per clip
        "mask_stats": {},
        "issues": [],
        "error": None,
    }
    results[name] = r

    # ── Init ──────────────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        adapter = cfg["cls"](**cfg["kwargs"])
        r["init_time"] = time.perf_counter() - t0
        r["init_ok"] = True
        r["num_sequences"] = len(adapter)
        print(f"  init  : {fmt(r['init_time'])}  sequences={r['num_sequences']}")
    except Exception as e:
        r["error"] = traceback.format_exc()
        print(f"  INIT FAILED: {e}")
        continue

    seqs = adapter.list_sequences()
    rng = random.Random(42)

    all_mask_stats = []

    for trial in range(N_SAMPLES):
        seq = rng.choice(seqs)
        info = adapter.get_sequence_info(seq)
        nf = info["num_frames"]
        if nf < CLIP_LEN:
            seq = next((s for s in seqs if adapter.get_sequence_info(s)["num_frames"] >= CLIP_LEN), seq)
            nf = adapter.get_sequence_info(seq)["num_frames"]

        start = rng.randint(0, max(0, nf - CLIP_LEN))
        frame_indices = list(range(start, start + CLIP_LEN))

        # ── load_clip ───────────────────────────────────────────────────
        try:
            t0 = time.perf_counter()
            clip = adapter.load_clip(seq, frame_indices)
            t_load = time.perf_counter() - t0
            r["t_load"].append(t_load)
        except Exception as e:
            r["issues"].append(f"load_clip failed on {seq}: {e}")
            print(f"  trial {trial+1}: load_clip FAILED: {e}")
            continue

        clip_issues = check_unified_clip(clip, name)
        if clip_issues and trial == 0:
            r["issues"].extend(clip_issues)
            print(f"  clip issues: {clip_issues}")

        # ── transform ───────────────────────────────────────────────────
        try:
            py_rng = random.Random(trial)
            t0 = time.perf_counter()
            result = transform(clip, rng=py_rng)
            t_xform = time.perf_counter() - t0
            r["t_xform"].append(t_xform)
        except Exception as e:
            r["issues"].append(f"transform failed on {seq}: {e}")
            print(f"  trial {trial+1}: transform FAILED: {e}")
            continue

        # ── query_builder ────────────────────────────────────────────────
        try:
            t0 = time.perf_counter()
            sample = qbuilder(result, py_rng=py_rng)
            t_query = time.perf_counter() - t0
            r["t_query"].append(t_query)
        except Exception as e:
            r["issues"].append(f"query_builder failed on {seq}: {e}")
            print(f"  trial {trial+1}: query_builder FAILED: {e}")
            continue

        sample_issues, mask_stats = check_query_sample(sample, name)
        if sample_issues and trial == 0:
            r["issues"].extend(sample_issues)
        all_mask_stats.append(mask_stats)

        if trial == 0:
            r["clip_ok"] = len(clip_issues) == 0
            r["pipeline_ok"] = len(sample_issues) == 0

        print(f"  trial {trial+1}: load={fmt(t_load)}  xform={fmt(t_xform)}  query={fmt(t_query)}"
              f"  mask_3d={mask_stats['mask_3d']:.2f}"
              f"  mask_2d={mask_stats['mask_2d']:.2f}")

    # ── Aggregate mask stats ────────────────────────────────────────────
    if all_mask_stats:
        avg = {}
        for k in all_mask_stats[0]:
            avg[k] = float(np.mean([m[k] for m in all_mask_stats]))
        r["mask_stats"] = avg

    # ── Timing summary ────────────────────────────────────────────────
    if r["t_load"]:
        avg_load  = np.mean(r["t_load"])
        avg_xform = np.mean(r["t_xform"]) if r["t_xform"] else 0
        avg_query = np.mean(r["t_query"]) if r["t_query"] else 0
        avg_total = avg_load + avg_xform + avg_query
        print(f"  avg   : load={fmt(avg_load)}  xform={fmt(avg_xform)}  query={fmt(avg_query)}  total={fmt(avg_total)}")
        if avg_load > SLOW_THRESHOLD:
            print(f"  ⚠️  SLOW I/O (>{SLOW_THRESHOLD}s) — candidate for H5 conversion")
        if r["pipeline_ok"]:
            adapters_ok.append(adapter)

# ─────────────────────────────────────────────────────────────────────────────
# MixtureDataset full-pipeline test
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"MixtureDataset test  ({len(adapters_ok)}/{len(DATASET_CONFIGS)} adapters available)")
print("=" * 72)

if len(adapters_ok) >= 2:
    try:
        mixture = MixtureDataset(
            adapters=adapters_ok,
            dataset_weights=None,  # uniform
            clip_len=CLIP_LEN,
            img_size=IMG_SIZE,
            use_augs=True,
            num_queries=NUM_QUERIES,
        )

        dataset_counts = {a.dataset_name: 0 for a in adapters_ok}
        N_MIX = 20
        t_mix_start = time.perf_counter()

        for i in range(N_MIX):
            sample = mixture[i]
            dataset_counts[sample.dataset_name] += 1

        t_mix_total = time.perf_counter() - t_mix_start
        print(f"  {N_MIX} samples in {t_mix_total:.2f}s ({t_mix_total/N_MIX:.2f}s/sample)")
        print(f"  Dataset distribution:")
        for ds, cnt in sorted(dataset_counts.items()):
            pct = 100.0 * cnt / N_MIX
            bar = "█" * int(pct / 5)
            print(f"    {ds:20s}  {cnt:3d} / {N_MIX}  ({pct:5.1f}%)  {bar}")
        print("  ✅ MixtureDataset OK")
    except Exception as e:
        print(f"  ❌ MixtureDataset FAILED: {e}")
        traceback.print_exc()
else:
    print("  SKIP: fewer than 2 adapters available")

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("SUMMARY")
print("=" * 72)
print(f"{'Dataset':20s}  {'Init':>5}  {'Seq':>5}  {'Load':>7}  {'Xform':>7}  {'Query':>7}  "
      f"{'Total':>7}  {'m3d':>5}  {'m2d':>5}  {'Status'}")
print("─" * 100)

for cfg in DATASET_CONFIGS:
    name = cfg["name"]
    r = results[name]
    if not r["init_ok"]:
        print(f"{name:20s}  {'':>5}  {'':>5}  {'':>7}  {'':>7}  {'':>7}  {'':>7}  {'':>5}  {'':>5}  ❌ INIT FAILED")
        continue

    avg_load  = np.mean(r["t_load"])  if r["t_load"]  else float("nan")
    avg_xform = np.mean(r["t_xform"]) if r["t_xform"] else float("nan")
    avg_query = np.mean(r["t_query"]) if r["t_query"] else float("nan")
    avg_total = avg_load + avg_xform + avg_query

    m3d = r["mask_stats"].get("mask_3d", float("nan"))
    m2d = r["mask_stats"].get("mask_2d", float("nan"))

    slow = " ⚠️ SLOW" if avg_load > SLOW_THRESHOLD else ""
    pipe = "✅" if r["pipeline_ok"] else ("⚠️" if r["clip_ok"] else "❌")
    issues = f"  [{', '.join(r['issues'][:2])}]" if r["issues"] else ""

    print(f"{name:20s}  {r['init_time']:5.2f}  {r['num_sequences']:5d}  "
          f"{avg_load:7.3f}  {avg_xform:7.3f}  {avg_query:7.3f}  {avg_total:7.3f}  "
          f"{m3d:5.2f}  {m2d:5.2f}  {pipe}{slow}{issues}")

print()
# H5 recommendation
slow_datasets = [
    name for name, r in results.items()
    if r["t_load"] and np.mean(r["t_load"]) > SLOW_THRESHOLD
]
if slow_datasets:
    print("H5 CONVERSION CANDIDATES:")
    for ds in slow_datasets:
        avg = np.mean(results[ds]["t_load"])
        print(f"  {ds}: avg load {avg:.2f}s — convert images/depths to HDF5")
else:
    print("All datasets within acceptable I/O speed.")
