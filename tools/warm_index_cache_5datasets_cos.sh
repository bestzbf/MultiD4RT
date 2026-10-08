#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-configs/mixture_5datasets_cos_planned.yaml}"
INDEX_CACHE_DIR="${INDEX_CACHE_DIR:-/data1/zbf/my_dfrt/.index_cache_5datasets_blendedmvs_hdu}"

POINTODYSSEY_ROOT="${POINTODYSSEY_ROOT:-/data_cos/hdu_datasets/PointOdyssey}"
POINTODYSSEY_LOCAL_ROOT="${POINTODYSSEY_LOCAL_ROOT:-/data2/d4rt/datasets/PointOdyssey}"
POINTODYSSEY_LOCAL_CACHE_DIR="${POINTODYSSEY_LOCAL_CACHE_DIR:-/data/zbf/openclaw/d4rt/.index_cache_5datasets_local}"
POINTODYSSEY_REQUIRE_TRACKS="${POINTODYSSEY_REQUIRE_TRACKS:-1}"
POINTODYSSEY_ASSUME_TRACKS="${POINTODYSSEY_ASSUME_TRACKS:-0}"
POINTODYSSEY_TRACK_WORKERS="${POINTODYSSEY_TRACK_WORKERS:-8}"

KUBRIC_ROOT="${KUBRIC_ROOT:-/data_cos/hdu_datasets/Kubric}"
DYNAMIC_REPLICA_ROOT="${DYNAMIC_REPLICA_ROOT:-/data_cos/hdu_datasets/Dynamic_Replica}"
CO3DV2_ROOT="${CO3DV2_ROOT:-/data2/d4rt/datasets/Co3Dv2}"
BLENDEDMVS_ROOT="${BLENDEDMVS_ROOT:-/data_cos/hdu_datasets/BlendedMVS}"
MVSSYNTH_ROOT="${MVSSYNTH_ROOT:-/data2/d4rt/datasets/MVS-Synth/GTAV_1080}"
TARTANAIR_ROOT="${TARTANAIR_ROOT:-/data5/d4rt_dataset/tartanairv1}"
WAYMO_ROOT="${WAYMO_ROOT:-/data2/d4rt/datasets/waymo_pre}"
VKITTI_ROOT="${VKITTI_ROOT:-/data3/dataset/VirtualKitti}"
SCANNETPP_ROOT="${SCANNETPP_ROOT:-/data_cos/hdu_datasets/scannetpp/data}"
SCANNETPP_SPLITS_DIR="${SCANNETPP_SPLITS_DIR:-/data_cos/hdu_datasets/scannetpp/splits}"
SCANNETPP_SCENES_RECORD="${SCANNETPP_SCENES_RECORD:-/data_cos/hdu_datasets/scannetpp/scenes_record.json}"
CO3DV2_DENYLIST="${CO3DV2_DENYLIST:-/data/zbf/openclaw/d4rt/configs/co3dv2_denylist_degenerate_clips_20260422.txt}"

INDEX_WORKERS="${INDEX_WORKERS:-8}"
WARM_VAL="${WARM_VAL:-0}"
ONLY_DATASETS="${ONLY_DATASETS:-}"
D4RT_SUPPRESS_MKL_WARNING="${D4RT_SUPPRESS_MKL_WARNING:-1}"
CO3DV2_FRAME_ANNO_CACHE="${CO3DV2_FRAME_ANNO_CACHE:-1}"
CO3DV2_FRAME_ANNO_CACHE_WORKERS="${CO3DV2_FRAME_ANNO_CACHE_WORKERS:-2}"
CO3DV2_FRAME_ANNO_CACHE_FORCE="${CO3DV2_FRAME_ANNO_CACHE_FORCE:-0}"

mkdir -p "$INDEX_CACHE_DIR"

if [[ "$D4RT_SUPPRESS_MKL_WARNING" == "1" ]]; then
  mkl_warning_filter="ignore:mkl-service package failed to import:UserWarning"
  if [[ -n "${PYTHONWARNINGS:-}" ]]; then
    export PYTHONWARNINGS="${mkl_warning_filter},${PYTHONWARNINGS}"
  else
    export PYTHONWARNINGS="$mkl_warning_filter"
  fi
fi

echo "[warm_index_cache] CONFIG=$CONFIG"
echo "[warm_index_cache] INDEX_CACHE_DIR=$INDEX_CACHE_DIR"
echo "[warm_index_cache] POINTODYSSEY_ROOT=$POINTODYSSEY_ROOT"
echo "[warm_index_cache] POINTODYSSEY_LOCAL_ROOT=$POINTODYSSEY_LOCAL_ROOT"
echo "[warm_index_cache] POINTODYSSEY_LOCAL_CACHE_DIR=$POINTODYSSEY_LOCAL_CACHE_DIR"
echo "[warm_index_cache] POINTODYSSEY_ASSUME_TRACKS=$POINTODYSSEY_ASSUME_TRACKS"
echo "[warm_index_cache] POINTODYSSEY_TRACK_WORKERS=$POINTODYSSEY_TRACK_WORKERS"
echo "[warm_index_cache] KUBRIC_ROOT=$KUBRIC_ROOT"
echo "[warm_index_cache] DYNAMIC_REPLICA_ROOT=$DYNAMIC_REPLICA_ROOT"
echo "[warm_index_cache] CO3DV2_ROOT=$CO3DV2_ROOT"
echo "[warm_index_cache] BLENDEDMVS_ROOT=$BLENDEDMVS_ROOT"
echo "[warm_index_cache] MVSSYNTH_ROOT=$MVSSYNTH_ROOT"
echo "[warm_index_cache] TARTANAIR_ROOT=$TARTANAIR_ROOT"
echo "[warm_index_cache] WAYMO_ROOT=$WAYMO_ROOT"
echo "[warm_index_cache] VKITTI_ROOT=$VKITTI_ROOT"
echo "[warm_index_cache] SCANNETPP_ROOT=$SCANNETPP_ROOT"
echo "[warm_index_cache] SCANNETPP_SPLITS_DIR=$SCANNETPP_SPLITS_DIR"
echo "[warm_index_cache] SCANNETPP_SCENES_RECORD=$SCANNETPP_SCENES_RECORD"
echo "[warm_index_cache] INDEX_WORKERS=$INDEX_WORKERS"
echo "[warm_index_cache] WARM_VAL=$WARM_VAL"
echo "[warm_index_cache] CO3DV2_FRAME_ANNO_CACHE=$CO3DV2_FRAME_ANNO_CACHE"
echo "[warm_index_cache] CO3DV2_FRAME_ANNO_CACHE_WORKERS=$CO3DV2_FRAME_ANNO_CACHE_WORKERS"
if [[ -n "$ONLY_DATASETS" ]]; then
  echo "[warm_index_cache] ONLY_DATASETS=$ONLY_DATASETS"
fi

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/d4rt/bin/python}"
if [[ ! -f "$PYTHON_BIN" ]]; then
  # 尝试本地conda环境
  if [[ -f "$HOME/miniconda3/envs/d4rt/bin/python" ]]; then
    PYTHON_BIN="$HOME/miniconda3/envs/d4rt/bin/python"
  elif [[ -f "$HOME/anaconda3/envs/d4rt/bin/python" ]]; then
    PYTHON_BIN="$HOME/anaconda3/envs/d4rt/bin/python"
  else
    echo "[warm_index_cache] ERROR: Python not found at $PYTHON_BIN" >&2
    exit 1
  fi
fi
"$PYTHON_BIN" - "$CONFIG" "$INDEX_CACHE_DIR" "$POINTODYSSEY_ROOT" "$POINTODYSSEY_LOCAL_ROOT" "$POINTODYSSEY_LOCAL_CACHE_DIR" "$POINTODYSSEY_REQUIRE_TRACKS" "$POINTODYSSEY_ASSUME_TRACKS" "$POINTODYSSEY_TRACK_WORKERS" "$KUBRIC_ROOT" "$DYNAMIC_REPLICA_ROOT" "$CO3DV2_ROOT" "$BLENDEDMVS_ROOT" "$MVSSYNTH_ROOT" "$TARTANAIR_ROOT" "$WAYMO_ROOT" "$VKITTI_ROOT" "$SCANNETPP_ROOT" "$SCANNETPP_SPLITS_DIR" "$SCANNETPP_SCENES_RECORD" "$CO3DV2_DENYLIST" "$INDEX_WORKERS" "$WARM_VAL" "$ONLY_DATASETS" <<'PY'
from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from datasets.registry import create_adapter

config_path = Path(sys.argv[1])
index_cache_dir = Path(sys.argv[2])
point_root = Path(sys.argv[3]).resolve()
point_local_root = Path(sys.argv[4]).resolve()
point_local_cache_dir = Path(sys.argv[5])
point_require_tracks = sys.argv[6] == "1"
point_assume_tracks = sys.argv[7] == "1"
point_track_workers = int(sys.argv[8])
kubric_root = Path(sys.argv[9]).resolve()
dynamic_replica_root = Path(sys.argv[10]).resolve()
co3dv2_root = Path(sys.argv[11]).resolve()
blendedmvs_root = Path(sys.argv[12]).resolve()
mvssynth_root = Path(sys.argv[13]).resolve()
tartanair_root = Path(sys.argv[14]).resolve()
waymo_root = Path(sys.argv[15]).resolve()
vkitti_root = Path(sys.argv[16]).resolve()
scannetpp_root = Path(sys.argv[17]).resolve()
scannetpp_splits_dir = Path(sys.argv[18]).resolve()
scannetpp_scenes_record = Path(sys.argv[19]).resolve()
co3dv2_denylist = sys.argv[20]
index_workers = int(sys.argv[21])
warm_val = sys.argv[22] == "1"
only_datasets = {x.strip() for x in sys.argv[23].split(",") if x.strip()}
VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".exr", ".npy"}


def point_cache_path(
    root: Path,
    split: str,
    fast_root: Path | None,
    require_tracks: bool,
    cache_dir: Path,
    cache_schema: int,
) -> Path:
    payload = {
        "dataset": "pointodyssey",
        "split": split,
        "root": str(root.resolve()),
        "fast_root": str(fast_root.resolve()) if fast_root is not None else None,
        "require_tracks": require_tracks,
        "strict": True,
        "cache_schema": cache_schema,
    }
    suffix = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"pointodyssey_{split}_{suffix}.pkl"


def compute_has_tracks(record: Any) -> bool:
    cached = getattr(record, "has_tracks", None)
    if cached is not None:
        return bool(cached)

    try:
        fast_anno_paths = getattr(record, "fast_anno_paths", None)
        if fast_anno_paths is not None and "trajs_3d" in fast_anno_paths:
            trajs_3d = np.load(fast_anno_paths["trajs_3d"], mmap_mode="r")
            return not (trajs_3d.ndim == 0 or trajs_3d.shape[0] == 0)

        anno_path = getattr(record, "anno_path", None)
        if anno_path is not None:
            anno = np.load(anno_path, allow_pickle=True)
            if "trajs_3d" in anno:
                trajs_3d = anno["trajs_3d"]
                return not (trajs_3d.ndim == 0 or trajs_3d.shape[0] == 0)
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return False

    return False


def rewrite_prefix(obj: Any, old_prefix: str, new_prefix: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Path):
        value = str(obj)
        if value.startswith(old_prefix):
            return Path(new_prefix + value[len(old_prefix):])
        return obj
    if isinstance(obj, str):
        if obj.startswith(old_prefix):
            return new_prefix + obj[len(old_prefix):]
        return obj
    if isinstance(obj, list):
        return [rewrite_prefix(x, old_prefix, new_prefix) for x in obj]
    if isinstance(obj, tuple):
        return tuple(rewrite_prefix(x, old_prefix, new_prefix) for x in obj)
    if isinstance(obj, dict):
        return {k: rewrite_prefix(v, old_prefix, new_prefix) for k, v in obj.items()}
    return obj


def try_frame_id_from_path(path: Path) -> str | None:
    m = re.search(r"(\d+)$", path.stem)
    if m is None:
        return None
    return m.group(1)


def sorted_frame_files(directory: Path) -> list[Path]:
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in VALID_SUFFIXES]
    return sorted(
        files,
        key=lambda p: (int(fid), p.name) if (fid := try_frame_id_from_path(p)) is not None else (10**18, p.name),
    )


def build_frame_map(files: list[Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in files:
        fid = try_frame_id_from_path(path)
        if fid is None:
            continue
        out[fid] = path
    return out


def align_modal_paths(directory: Path, expected_frame_ids: list[str]) -> list[Path] | None:
    if not directory.is_dir():
        return None
    modal_map = build_frame_map(sorted_frame_files(directory))
    if any(fid not in modal_map for fid in expected_frame_ids):
        return None
    return [modal_map[fid] for fid in expected_frame_ids]


def sanitize_remote_point_record(record: Any) -> Any | None:
    seq_root = Path(record.sequence_root)
    if not seq_root.is_dir():
        return None

    rgb_dir = seq_root / "rgbs"
    if not rgb_dir.is_dir():
        return None

    rgb_map = build_frame_map(sorted_frame_files(rgb_dir))
    expected_frame_ids = [f"{i:05d}" for i in range(int(record.num_frames))]
    if len(rgb_map) != len(expected_frame_ids):
        return None
    if any(fid not in rgb_map for fid in expected_frame_ids):
        return None

    record.rgb_paths = [rgb_map[fid] for fid in expected_frame_ids]
    record.depth_paths = align_modal_paths(seq_root / "depths", expected_frame_ids)
    record.normal_paths = align_modal_paths(seq_root / "normals", expected_frame_ids)
    record.mask_paths = align_modal_paths(seq_root / "masks", expected_frame_ids)
    record.info_path = (seq_root / "info.npz") if (seq_root / "info.npz").exists() else None
    record.scene_info_path = (seq_root / "scene_info.json") if (seq_root / "scene_info.json").exists() else None

    npz_path = seq_root / "anno.npz"
    h5_path = seq_root / "anno.h5"
    if npz_path.exists():
        record.anno_path = npz_path
    elif h5_path.exists():
        # Keep the canonical npz-shaped path so adapter._load_anno can still
        # resolve anno.h5 via with_suffix(".h5").
        record.anno_path = npz_path
    else:
        return None

    return record


def prime_pointodyssey_cache(split: str) -> None:
    target_cache = point_cache_path(
        root=point_root,
        split=split,
        fast_root=None,
        require_tracks=point_require_tracks,
        cache_dir=index_cache_dir,
        cache_schema=3,
    )
    if target_cache.exists():
        print(f"[warm_index_cache] PointOdyssey {split}: target cache already exists: {target_cache}")
        return

    source_candidates = [
        point_cache_path(
            root=point_local_root,
            split=split,
            fast_root=None,
            require_tracks=point_require_tracks,
            cache_dir=point_local_cache_dir,
            cache_schema=3,
        )
    ]
    if point_require_tracks:
        source_candidates.append(
            point_cache_path(
                root=point_local_root,
                split=split,
                fast_root=None,
                require_tracks=False,
                cache_dir=point_local_cache_dir,
                cache_schema=3,
            )
        )
        source_candidates.append(
            point_cache_path(
                root=point_local_root,
                split=split,
                fast_root=None,
                require_tracks=point_require_tracks,
                cache_dir=point_local_cache_dir,
                cache_schema=2,
            )
        )
        source_candidates.append(
            point_cache_path(
                root=point_local_root,
                split=split,
                fast_root=None,
                require_tracks=False,
                cache_dir=point_local_cache_dir,
                cache_schema=2,
            )
        )
    else:
        source_candidates.append(
            point_cache_path(
                root=point_local_root,
                split=split,
                fast_root=None,
                require_tracks=point_require_tracks,
                cache_dir=point_local_cache_dir,
                cache_schema=2,
            )
        )

    source_cache = next((p for p in source_candidates if p.exists()), None)
    if source_cache is None:
        print(
            "[warm_index_cache] PointOdyssey "
            f"{split}: no local source cache found in {point_local_cache_dir}, "
            "will fall back to remote build"
        )
        return

    print(f"[warm_index_cache] PointOdyssey {split}: priming remote cache from local cache {source_cache}")
    with open(source_cache, "rb") as f:
        records = pickle.load(f)

    old_prefix = str(point_local_root)
    new_prefix = str(point_root)
    rewritten = []
    dropped = 0
    start = time.time()
    if point_assume_tracks:
        track_flags = [True] * len(records)
    else:
        n_workers = max(1, min(point_track_workers, len(records)))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            track_flags = list(executor.map(compute_has_tracks, records))

    for record, has_tracks in zip(records, track_flags):
        rec = copy.deepcopy(record)
        for attr in (
            "sequence_root",
            "rgb_paths",
            "depth_paths",
            "normal_paths",
            "mask_paths",
            "anno_path",
            "info_path",
            "scene_info_path",
            "fast_dir",
            "fast_anno_paths",
            "encoded_cache_paths",
        ):
            if hasattr(rec, attr):
                setattr(rec, attr, rewrite_prefix(getattr(rec, attr), old_prefix, new_prefix))
        rec.has_tracks = has_tracks
        rec = sanitize_remote_point_record(rec)
        if rec is None:
            dropped += 1
            continue
        rewritten.append(rec)

    index_cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = target_cache.with_suffix(f".tmp{os.getpid()}")
    with open(tmp_path, "wb") as f:
        pickle.dump(rewritten, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, target_cache)
    print(
        f"[warm_index_cache] PointOdyssey {split}: wrote {target_cache} "
        f"with {len(rewritten)} records (dropped {dropped}) in {time.time() - start:.1f}s"
    )


def warm_adapter(name: str, root: str, split: str, adapter_kwargs: dict[str, Any]) -> None:
    if only_datasets and name not in only_datasets:
        return

    kwargs = dict(adapter_kwargs)
    kwargs.update({"root": root, "split": split, "cache_dir": str(index_cache_dir), "verbose": True})
    if name == "pointodyssey":
        try:
            on_remote_mount = Path(root).resolve().is_relative_to(Path("/data_cos").resolve())
        except Exception:
            on_remote_mount = str(root).startswith("/data_cos/")
        if on_remote_mount:
            # The warmed cache already contains the canonical COS paths.  Letting
            # PointOdysseyAdapter sanitize here causes thousands of slow s3fs
            # metadata calls before training even starts.
            kwargs["runtime_sanitize"] = False
    if name in {"pointodyssey", "kubric", "blendedmvs", "mvssynth", "tartanair", "waymo", "vkitti2"}:
        kwargs["index_workers"] = index_workers
    if name == "co3dv2":
        kwargs["sequence_denylist"] = co3dv2_denylist
    if name == "scannetpp":
        kwargs["splits_dir"] = str(scannetpp_splits_dir)
        kwargs["scenes_record"] = str(scannetpp_scenes_record)
        kwargs["index_workers"] = index_workers
        kwargs["strict"] = False
    if name == "mvssynth":
        kwargs["precompute_root"] = root
        kwargs["load_precomputed"] = True
        kwargs["load_normals"] = False
    if name == "tartanair":
        kwargs["precompute_root"] = root
        kwargs["load_precomputed"] = True
        kwargs["load_normals"] = False
    if name == "waymo":
        kwargs["precompute_root"] = root
        kwargs["extract_flow"] = False
    if name == "vkitti2":
        kwargs["camera"] = kwargs.get("camera", "Camera_0")
        kwargs["precompute_root"] = root
        kwargs["load_precomputed"] = True
        kwargs["load_normals"] = False
        kwargs["load_flow"] = False

    print(f"[warm_index_cache] warming {name} split={split} root={root}")
    start = time.time()
    adapter = create_adapter(name, **kwargs)
    count = len(adapter)
    print(f"[warm_index_cache] ready {name} split={split}: {count} sequences in {time.time() - start:.1f}s")


cfg = yaml.safe_load(config_path.read_text())
config_dataset_names = {item["name"] for item in cfg.get("datasets", [])}
root_map = {
    "pointodyssey": str(point_root),
    "kubric": str(kubric_root),
    "dynamic_replica": str(dynamic_replica_root),
    "co3dv2": str(co3dv2_root),
    "blendedmvs": str(blendedmvs_root),
    "mvssynth": str(mvssynth_root),
    "tartanair": str(tartanair_root),
    "waymo": str(waymo_root),
    "vkitti2": str(vkitti_root),
    "scannetpp": str(scannetpp_root),
}

should_prime_point = (
    "pointodyssey" in config_dataset_names
    and ((not only_datasets) or ("pointodyssey" in only_datasets))
)
if should_prime_point:
    prime_pointodyssey_cache("train")
    if warm_val:
        prime_pointodyssey_cache("val")

for ds_cfg in cfg["datasets"]:
    name = ds_cfg["name"]
    if name in root_map:
        root = root_map[name]
    else:
        continue
    # pointodyssey is already handled by prime_pointodyssey_cache above;
    # re-creating the adapter here would trigger slow _record_has_tracks I/O
    # against COS-rewritten paths in the shared cache.
    if name == "pointodyssey":
        continue
    adapter_kwargs = dict(ds_cfg.get("adapter_kwargs", {}))
    adapter_kwargs.pop("fast_root", None)
    split = ds_cfg.get("split", "train")
    warm_adapter(name, root, split, adapter_kwargs)

if warm_val:
    for ds_cfg in cfg["datasets"]:
        name = ds_cfg["name"]
        if name in root_map:
            root = root_map[name]
        else:
            continue
        adapter_kwargs = dict(ds_cfg.get("adapter_kwargs", {}))
        adapter_kwargs.pop("fast_root", None)
        split = ds_cfg.get("val_split", "val")
        warm_adapter(name, root, split, adapter_kwargs)

print("[warm_index_cache] done")
PY

if [[ "$CO3DV2_FRAME_ANNO_CACHE" == "1" ]]; then
  should_warm_co3dv2=0
  if [[ -z "$ONLY_DATASETS" ]]; then
    should_warm_co3dv2=1
  elif [[ ",$ONLY_DATASETS," == *",co3dv2,"* ]]; then
    should_warm_co3dv2=1
  fi
  if [[ "$should_warm_co3dv2" == "1" ]]; then
    force_args=()
    if [[ "$CO3DV2_FRAME_ANNO_CACHE_FORCE" == "1" ]]; then
      force_args+=(--force)
    fi
    /root/miniconda3/envs/d4rt/bin/python "$ROOT_DIR/scripts/build_co3dv2_frame_cache.py" \
      --root "$CO3DV2_ROOT" \
      --cache-dir "$INDEX_CACHE_DIR" \
      --workers "$CO3DV2_FRAME_ANNO_CACHE_WORKERS" \
      "${force_args[@]}"
  fi
fi
