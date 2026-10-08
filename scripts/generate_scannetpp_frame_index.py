#!/usr/bin/env python3
"""Generate a compact ScanNet++ frame index sidecar.

The adapter only needs the COLMAP first-line image metadata plus timestamp and
depth full-frame mapping.  Full ``iphone/colmap/images.txt`` can be hundreds of
MB because it stores all 2D feature observations.  This script builds a small
``iphone/frame_index.pkl`` payload that lets staging skip those large text
files.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.adapters.scannetpp import build_frame_index_payload


def _stage_cache_output(stage_root: Path, mount_root: Path, scene_dir: Path) -> Path:
    rel = scene_dir.relative_to(mount_root)
    return stage_root / "shared_raw_cache" / "data" / rel / "iphone" / "frame_index.pkl"


def _list_scenes(root: Path, scenes_record: Path | None) -> list[str]:
    if scenes_record is not None and scenes_record.exists():
        scenes: list[str] = []
        with open(scenes_record, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if str(item.get("status", "ok")) != "ok":
                    continue
                scene = item.get("scene_name")
                if scene:
                    scenes.append(str(scene))
        return sorted(set(scenes))
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _output_path_for_scene(args: argparse.Namespace, scene_dir: Path) -> Path:
    if args.output:
        output = Path(args.output)
        if output.suffix:
            return output
        return output / scene_dir.name / "iphone" / "frame_index.pkl"
    if args.stage_root:
        return _stage_cache_output(Path(args.stage_root), Path(args.mount_root), scene_dir)
    return scene_dir / "iphone" / "frame_index.pkl"


def _generate_one(args: argparse.Namespace, scene: str) -> tuple[str, str, float, int]:
    scene_dir = Path(args.root) / scene
    output_path = _output_path_for_scene(args, scene_dir)
    if args.skip_existing and output_path.exists():
        return scene, "skip", 0.0, output_path.stat().st_size

    t0 = time.perf_counter()
    payload = build_frame_index_payload(scene_dir)
    build_s = time.perf_counter() - t0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.part.{scene}.{os.getpid()}")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(output_path)

    return scene, "ok", build_s, output_path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data_cos/hdu_datasets/scannetpp/data")
    parser.add_argument("--scene", default="")
    parser.add_argument("--all", action="store_true", help="Generate for all scenes.")
    parser.add_argument("--scenes-record", default="/data_cos/hdu_datasets/scannetpp/scenes_record.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--stage-root", default="")
    parser.add_argument("--mount-root", default="/data_cos")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not args.scene and not args.all:
        raise SystemExit("Specify --scene SCENE or --all")

    if args.scene:
        scenes = [args.scene]
    else:
        scenes_record = Path(args.scenes_record) if args.scenes_record else None
        scenes = _list_scenes(Path(args.root), scenes_record)
    if args.limit > 0:
        scenes = scenes[: args.limit]

    t_all0 = time.perf_counter()
    done = skipped = errors = 0
    workers = max(1, int(args.workers))

    def report(scene: str, status: str, build_s: float, size: int) -> None:
        nonlocal done, skipped, errors
        if status == "ok":
            done += 1
        elif status == "skip":
            skipped += 1
        else:
            errors += 1
        total = done + skipped + errors
        if status != "skip" or total % 50 == 0:
            print(
                f"[{total}/{len(scenes)}] {status} scene={scene} "
                f"size={size} build={build_s:.3f}s "
                f"done={done} skip={skipped} err={errors}",
                flush=True,
            )

    if workers == 1:
        for scene in scenes:
            try:
                report(*_generate_one(args, scene))
            except Exception as exc:
                report(scene, f"error:{type(exc).__name__}:{exc}", 0.0, 0)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_generate_one, args, scene): scene for scene in scenes}
            for future in as_completed(futures):
                scene = futures[future]
                try:
                    report(*future.result())
                except Exception as exc:
                    report(scene, f"error:{type(exc).__name__}:{exc}", 0.0, 0)

    elapsed = time.perf_counter() - t_all0
    print(
        f"finished scenes={len(scenes)} done={done} skip={skipped} "
        f"errors={errors} elapsed={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
