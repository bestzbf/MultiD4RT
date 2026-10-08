"""Extract Waymo TFRecord tar shards into the D4RT preprocessed layout.

The output schema matches the existing Waymo training cache:

    <output-root>/<split>/<segment>.tfrecord/
        calib.json
        labels.npz
        00000_1.jpg
        00000_1.npz
        ...

Each per-frame npz stores:
    pose:      vehicle/camera image pose transform, shape [4, 4]
    pixels:    projected LiDAR pixels for that camera, shape [N, 2], int16
    pts3d:     LiDAR points in vehicle frame, shape [N, 3], float32
    timestamp: camera image pose timestamp in seconds
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


DEFAULT_KEEP_LABEL_TYPES = (1, 2, 4)  # vehicle, pedestrian, cyclist


def _import_waymo():
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.utils import frame_utils

    # Older Waymo helpers still reference bytearray as a mutable type hook.
    frame_utils.bytearray = bytes
    return tf, dataset_pb2, frame_utils


def _camera_calibration_json(frame) -> list[list[object]]:
    out = []
    for calibration in frame.context.camera_calibrations:
        out.append(
            [
                int(calibration.name),
                {
                    "width": int(calibration.width),
                    "height": int(calibration.height),
                    "intrinsics": [float(v) for v in calibration.intrinsic],
                    "extrinsics": [float(v) for v in calibration.extrinsic.transform],
                },
            ]
        )
    return out


def _collect_laser_labels(frame, frame_idx: int, keep_types: set[int], label_store: dict[str, list]) -> None:
    if not label_store["context_name"]:
        label_store["context_name"] = frame.context.name
    for label in frame.laser_labels:
        label_type = int(label.type)
        if label_type not in keep_types or not label.id:
            continue
        box = label.box
        label_store["frame_indices"].append(int(frame_idx))
        label_store["object_ids"].append(str(label.id))
        label_store["label_types"].append(label_type)
        label_store["boxes_vehicle"].append(
            [
                float(box.center_x),
                float(box.center_y),
                float(box.center_z),
                float(box.length),
                float(box.width),
                float(box.height),
                float(box.heading),
            ]
        )


def _save_labels(path: Path, label_store: dict[str, list], num_frames: int) -> None:
    np.savez_compressed(
        path,
        frame_indices=np.asarray(label_store["frame_indices"], dtype=np.int32),
        object_ids=np.asarray(label_store["object_ids"], dtype=np.str_),
        label_types=np.asarray(label_store["label_types"], dtype=np.int32),
        boxes_vehicle=np.asarray(label_store["boxes_vehicle"], dtype=np.float32).reshape(-1, 7),
        context_name=np.asarray(label_store["context_name"]),
        num_frames=np.asarray(num_frames, dtype=np.int32),
    )


def _save_camera_jpeg(image_bytes: bytes, path: Path) -> None:
    # The existing training cache stores decoded images re-encoded with PIL's
    # default JPEG settings, not the original Waymo JPEG byte stream.
    with Image.open(BytesIO(image_bytes)) as image:
        image.convert("RGB").save(path)


def _frame_points_by_camera(frame, frame_utils) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    range_images, camera_projections, _, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose
    )
    if not points:
        return {}

    points_all = np.concatenate(points, axis=0).astype(np.float32, copy=False)
    cp_all = np.concatenate(cp_points, axis=0)
    by_camera: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for camera_name in np.unique(cp_all[:, 0].astype(np.int32)):
        mask = cp_all[:, 0] == camera_name
        pixels = cp_all[mask, 1:3].astype(np.int16, copy=False)
        by_camera[int(camera_name)] = (points_all[mask].astype(np.float32, copy=False), pixels)
    return by_camera


def extract_tfrecord(
    tfrecord_path: Path,
    out_dir: Path,
    *,
    keep_label_types: Iterable[int] = DEFAULT_KEEP_LABEL_TYPES,
    overwrite: bool = False,
) -> dict[str, object]:
    tf, dataset_pb2, frame_utils = _import_waymo()

    done_path = out_dir / "extraction_done.json"
    if done_path.exists() and not overwrite:
        return {"sequence": out_dir.name, "status": "skipped_done"}

    tmp_parent = out_dir.parent
    tmp_dir = tmp_parent / f".{out_dir.name}.tmp.{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    keep_types = {int(v) for v in keep_label_types}
    label_store: dict[str, list] = {
        "frame_indices": [],
        "object_ids": [],
        "label_types": [],
        "boxes_vehicle": [],
        "context_name": "",
    }

    num_frames = 0
    num_images = 0
    total_lidar_points = 0
    calibration_payload: list[list[object]] | None = None

    dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")
    for frame_idx, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytes(data.numpy()))

        if calibration_payload is None:
            calibration_payload = _camera_calibration_json(frame)

        _collect_laser_labels(frame, frame_idx, keep_types, label_store)
        points_by_camera = _frame_points_by_camera(frame, frame_utils)

        for camera_image in frame.images:
            camera_name = int(camera_image.name)
            prefix = f"{frame_idx:05d}_{camera_name}"
            _save_camera_jpeg(bytes(camera_image.image), tmp_dir / f"{prefix}.jpg")

            pts3d, pixels = points_by_camera.get(
                camera_name,
                (
                    np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0, 2), dtype=np.int16),
                ),
            )
            total_lidar_points += int(len(pts3d))
            pose = np.asarray(camera_image.pose.transform, dtype=np.float64).reshape(4, 4)
            timestamp = float(camera_image.pose_timestamp)
            np.savez_compressed(
                tmp_dir / f"{prefix}.npz",
                pose=pose,
                pixels=pixels,
                pts3d=pts3d,
                timestamp=np.asarray(timestamp, dtype=np.float64),
            )
            num_images += 1

        num_frames += 1

    if calibration_payload is None:
        raise RuntimeError(f"{tfrecord_path} contained no frames")
    (tmp_dir / "calib.json").write_text(
        json.dumps(calibration_payload, indent=2),
        encoding="utf-8",
    )
    _save_labels(tmp_dir / "labels.npz", label_store, num_frames)
    done = {
        "sequence": out_dir.name,
        "source_tfrecord": str(tfrecord_path),
        "num_frames": int(num_frames),
        "num_images": int(num_images),
        "num_labels": int(len(label_store["frame_indices"])),
        "total_lidar_camera_points": int(total_lidar_points),
    }
    (tmp_dir / "extraction_done.json").write_text(
        json.dumps(done, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if out_dir.exists():
        if overwrite:
            shutil.rmtree(out_dir)
        else:
            backup = out_dir.with_name(f".{out_dir.name}.partial.{os.getpid()}")
            out_dir.rename(backup)
    tmp_dir.rename(out_dir)
    return {"sequence": out_dir.name, "status": "done", **done}


def _copy_tar_member_to_temp(tar: tarfile.TarFile, member: tarfile.TarInfo, tmp_root: Path) -> Path:
    tmp_root.mkdir(parents=True, exist_ok=True)
    out_path = tmp_root / Path(member.name).name
    if out_path.exists():
        out_path.unlink()
    src = tar.extractfile(member)
    if src is None:
        raise RuntimeError(f"failed to read tar member {member.name}")
    with src, out_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    return out_path


def iter_tfrecord_members(tar_path: Path) -> list[tarfile.TarInfo]:
    with tarfile.open(tar_path, "r") as tar:
        return [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".tfrecord")]


def extract_tar(
    tar_path: Path,
    *,
    output_root: Path,
    split: str,
    tmp_root: Path,
    keep_label_types: Iterable[int],
    overwrite: bool,
    sequence_limit: int | None,
) -> list[dict[str, object]]:
    split_root = output_root / split
    split_root.mkdir(parents=True, exist_ok=True)
    tmp_root = tmp_root / f"{tar_path.stem}.{os.getpid()}"
    results: list[dict[str, object]] = []

    with tarfile.open(tar_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".tfrecord")]
        if sequence_limit is not None:
            members = members[: int(sequence_limit)]
        for member_idx, member in enumerate(members, start=1):
            seq_name = Path(member.name).name
            out_dir = split_root / seq_name
            done_path = out_dir / "extraction_done.json"
            if done_path.exists() and not overwrite:
                result = {"sequence": seq_name, "status": "skipped_done"}
                print(json.dumps(result, ensure_ascii=False), flush=True)
                results.append(result)
                continue

            tmp_tfrecord = _copy_tar_member_to_temp(tar, member, tmp_root)
            try:
                result = extract_tfrecord(
                    tmp_tfrecord,
                    out_dir,
                    keep_label_types=keep_label_types,
                    overwrite=overwrite,
                )
                result["tar"] = str(tar_path)
                result["member_index"] = member_idx
                result["members_total"] = len(members)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                results.append(result)
            finally:
                tmp_tfrecord.unlink(missing_ok=True)

    shutil.rmtree(tmp_root, ignore_errors=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="/data2/d4rt/datasets/waymo_raw")
    parser.add_argument("--output-root", default="/data2/d4rt/datasets/waymo_pre")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--tar", action="append", default=[])
    parser.add_argument("--tmp-root", default="/data2/d4rt/datasets/waymo_raw/.extract_tmp")
    parser.add_argument("--keep-label-types", default="1,2,4")
    parser.add_argument("--sequence-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    tar_paths = [Path(p) for p in args.tar]
    if not tar_paths:
        tar_paths = sorted((raw_root / args.split).glob("*.tar"))
    keep_label_types = [int(v) for v in str(args.keep_label_types).split(",") if v.strip()]

    all_results: list[dict[str, object]] = []
    for tar_path in tar_paths:
        all_results.extend(
            extract_tar(
                tar_path,
                output_root=Path(args.output_root),
                split=args.split,
                tmp_root=Path(args.tmp_root),
                keep_label_types=keep_label_types,
                overwrite=bool(args.overwrite),
                sequence_limit=args.sequence_limit,
            )
        )

    summary = {
        "split": args.split,
        "num_tars": len(tar_paths),
        "num_results": len(all_results),
        "done": sum(1 for item in all_results if item.get("status") == "done"),
        "skipped": sum(1 for item in all_results if str(item.get("status", "")).startswith("skipped")),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
