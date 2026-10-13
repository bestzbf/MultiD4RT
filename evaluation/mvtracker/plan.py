"""Immutable scene/query manifests, strict prediction archives, and reports."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .official import sha256_file
from .protocols import PROTOCOLS


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def inspect_cache(path, protocol):
    with np.load(path, allow_pickle=False) as cache:
        t = cache["traj3d_world"].shape[0]
        n, v = protocol.queries, len(protocol.views)
        if t < 2 or (protocol.frames is not None and t != protocol.frames):
            raise ValueError(f"Unexpected official sequence length: {path}, {t}")
        expected = {"query_points": (n, 4), "traj3d_world": (t, n, 3),
                    "visibility": (v, t, n), "traj2d_w_z": (v, t, n, 3),
                    "valids": (t, n), "track_indices": (n,)}
        for name, shape in expected.items():
            if cache[name].shape != shape:
                raise ValueError(f"Invalid official cache {path}: {name} {cache[name].shape} != {shape}")
        queries = cache["query_points"]
        times = queries[:, 0]
        if not np.isfinite(queries).all() or not np.isfinite(cache["traj3d_world"]).all():
            raise ValueError(f"Non-finite query/trajectory in {path}")
        if not np.equal(times, times.astype(int)).all() or not ((times >= 0) & (times < t - 1)).all():
            raise ValueError(f"Invalid query timestamps in {path}")
        visible = cache["visibility"].astype(bool).any(axis=0)
        if not visible[times.astype(int), np.arange(n)].all():
            raise ValueError(f"Query invisible in all input cameras: {path}")
        after_query = np.arange(t)[:, None] >= times[None]
        if not ((visible & after_query).sum(axis=0) >= 2).all():
            raise ValueError(f"Query has no visible evaluation timestep: {path}")
        anchor_error = np.linalg.norm(cache["traj3d_world"][times.astype(int), np.arange(n)] - queries[:, 1:], axis=-1)
        if anchor_error.max() > 1e-4:
            raise ValueError(f"Cached query/track mismatch: {path}")
    return {"frames": t, "queries": n, "views": list(protocol.views),
            "cache_sha256": sha256_file(path), "max_query_anchor_error": float(anchor_error.max())}


def build_plan(root, official, profiles):
    root = Path(root).resolve()
    scenes = []
    for name in profiles:
        protocol = PROTOCOLS[name]
        folder = root / "datasets/mvtracker" / protocol.folder
        sequence_dirs = sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))
        if protocol.family == "kubric-multiview":
            sequence_dirs.sort(key=lambda path: int(path.name))  # Official reader uses numeric IDs.
        # A partial download must not silently become the published test set.
        if len(sequence_dirs) != protocol.sequences:
            raise ValueError(f"{name}: expected {protocol.sequences} sequences, found {len(sequence_dirs)}")
        for path in sequence_dirs:
            cache = path / "cache" / protocol.cache_name
            if not cache.is_file():
                raise FileNotFoundError(f"Released query cache missing; resampling is forbidden: {cache}")
            record = {"id": f"{name}/{path.name}", "profile": name, "sequence": path.name,
                      "cache": str(cache.relative_to(root)), **inspect_cache(cache, protocol)}
            scenes.append(record)
    plan = {"format": "multicamera4rc-official-multiview-v1", "benchmark_root": str(root),
            "profiles": {name: PROTOCOLS[name].to_dict() for name in profiles},
            "official": official.provenance, "scenes": scenes,
            "query_mode": "first", "visibility": "OR over the selected input cameras",
            "coordinates": "official loader normalized world; restore scale in official scorer",
            "aggregation": "official rounded per-sequence metrics, arithmetic mean within each profile"}
    verification = root / "metadata/verification.json"
    if verification.is_file():
        plan["download_verification_sha256"] = sha256_file(verification)
    plan["plan_sha256"] = digest_json(plan)
    return plan


def read_plan(path, root=None):
    plan = json.loads(Path(path).read_text())
    unsigned = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if digest_json(unsigned) != plan.get("plan_sha256"):
        raise ValueError("Frozen plan was modified")
    if root is not None and Path(root).resolve() != Path(plan["benchmark_root"]).resolve():
        raise ValueError("Plan belongs to another benchmark root")
    for name, spec in plan["profiles"].items():
        if name not in PROTOCOLS or spec != PROTOCOLS[name].to_dict():
            raise ValueError(f"Plan uses a different official profile: {name}")
    expected = sum(PROTOCOLS[name].sequences for name in plan["profiles"])
    if len(plan["scenes"]) != expected or len({s["id"] for s in plan["scenes"]}) != expected:
        raise ValueError("Frozen plan does not contain the entire released test set")
    return plan


def cache_for_scene(plan, scene):
    path = Path(plan["benchmark_root"]) / scene["cache"]
    if sha256_file(path) != scene["cache_sha256"]:
        raise ValueError(f"Frozen query cache changed: {path}")
    return path


def prediction_path(folder, scene):
    return Path(folder) / scene["profile"] / (scene["sequence"] + ".npz")


def save_prediction(path, plan, scene, tracks, visible, run):
    path = Path(path)
    shape = (scene["frames"], scene["queries"])
    tracks, visible = np.asarray(tracks), np.asarray(visible)
    if tracks.shape != shape + (3,) or visible.shape != shape or visible.dtype != np.bool_:
        raise ValueError("Prediction shape/boolean visibility does not match the complete frozen scene")
    if not np.isfinite(tracks).all():
        raise ValueError("Non-finite predictions cannot be scored")
    metadata = {"plan_sha256": plan["plan_sha256"], "scene_id": scene["id"],
                "cache_sha256": scene["cache_sha256"], "coordinates": plan["coordinates"],
                "run": run, "run_sha256": digest_json(run)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, tracks=tracks.astype(np.float32), visible=visible,
                            metadata=np.asarray(json.dumps(metadata, allow_nan=False)))
    temporary.replace(path)


def read_prediction(path, plan, scene, run=None):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        tracks, visible = archive["tracks"].copy(), archive["visible"].copy()
    for key, expected in (("plan_sha256", plan["plan_sha256"]), ("scene_id", scene["id"]),
                          ("cache_sha256", scene["cache_sha256"]), ("coordinates", plan["coordinates"])):
        if metadata.get(key) != expected:
            raise ValueError(f"Prediction metadata mismatch ({key}): {path}")
    if metadata.get("run_sha256") != digest_json(metadata["run"]):
        raise ValueError(f"Prediction run metadata was modified: {path}")
    if run is not None and metadata["run"] != run:
        raise ValueError(f"Refusing to resume predictions from a different checkpoint/configuration: {path}")
    expected = (scene["frames"], scene["queries"])
    if tracks.shape != expected + (3,) or visible.shape != expected or visible.dtype != np.bool_:
        raise ValueError(f"Incomplete or invalid prediction arrays: {path}")
    if not np.isfinite(tracks).all():
        raise ValueError(f"Non-finite predictions: {path}")
    return tracks, visible, metadata["run"]


def score_predictions(plan, official, folder, allow_partial=False):
    if plan["official"] != official.provenance:
        raise ValueError("The scorer is different from the frozen official source")
    records = []
    missing = []
    run = None
    for scene in plan["scenes"]:
        path = prediction_path(folder, scene)
        if not path.is_file():
            missing.append(scene["id"])
            continue
        tracks, visible, scene_run = read_prediction(path, plan, scene)
        if run is not None and scene_run != run:
            raise ValueError("A benchmark report cannot mix checkpoints or input conditions")
        run = scene_run
        protocol = PROTOCOLS[scene["profile"]]
        labels = official.cached_labels(cache_for_scene(plan, scene), protocol)
        metrics = official.score(labels, tracks, visible, protocol)
        records.append({"id": scene["id"], "profile": scene["profile"], "sequence": scene["sequence"],
                        "metrics": {k: float(v) if np.isfinite(v) else None for k, v in metrics.items()}})
    if missing and not allow_partial:
        raise ValueError(f"Missing {len(missing)} scenes; full benchmark scoring refused: {missing[:8]}")
    if not records:
        raise ValueError("No predictions to evaluate")
    summaries = {}
    for name in plan["profiles"]:
        subset = [r for r in records if r["profile"] == name]
        keys = sorted({k for r in subset for k in r["metrics"]})
        summary = {}
        for key in keys:
            values = [r["metrics"][key] for r in subset if r["metrics"].get(key) is not None]
            summary[key] = float(np.mean(values)) if values else None
        summaries[name] = {"completed": len(subset), "expected": PROTOCOLS[name].sequences, "metrics": summary}
    return {"complete": not missing, "result_kind": "full_benchmark" if not missing else "partial_validation",
            "plan_sha256": plan["plan_sha256"], "official": official.provenance,
            "run": run, "missing": missing, "summaries": summaries, "per_sequence": records}


def write_report(folder, report):
    folder = Path(folder)
    write_json(folder / "results.json", report)
    keys = sorted({k for r in report["per_sequence"] for k in r["metrics"]})
    with (folder / "per_sequence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "profile", "sequence"] + keys)
        writer.writeheader()
        for record in report["per_sequence"]:
            writer.writerow({k: record[k] for k in ("id", "profile", "sequence")} | record["metrics"])
