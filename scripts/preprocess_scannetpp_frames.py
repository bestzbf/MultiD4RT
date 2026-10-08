#!/usr/bin/env python3
"""
Preprocess ScanNetPP scenes:
1. Decode rgb.mkv -> iphone/frames/{:06d}.jpg (q=95)
2. Generate iphone/depth_chunk_index.pkl
3. Upload frames/ + depth_chunk_index.pkl + precomputed.h5 to COS

Usage:
    python scripts/preprocess_scannetpp_frames.py \
        --data-root /data3/dataset/scannetpp/data \
        --workers 8 \
        --upload
"""
import argparse, os, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.adapters.scannetpp import _join_frames, _extract_video_frames_by_timestamps, _index_depth_bin_chunks

QUALITY = 95


def _process_scene(args):
    scene_dir, upload, bucket, cos_creds = args
    scene = scene_dir.name
    frames_dir = scene_dir / "iphone" / "frames"
    index_path = scene_dir / "iphone" / "depth_chunk_index.pkl"
    h5_path = scene_dir / "precomputed.h5"

    results = {"scene": scene, "jpg": False, "index": False, "h5": False, "error": None}
    try:
        # 1. decode jpg frames
        if not frames_dir.exists() or len(list(frames_dir.glob("*.jpg"))) == 0:
            frames_dir.mkdir(parents=True, exist_ok=True)
            sd = _join_frames(scene_dir)
            n = len(sd["frame_stems"])
            indices = list(range(n))
            timestamps = [sd["timestamps"][i] for i in indices]
            fallback = sd["full_indices"][indices].tolist()
            frames = _extract_video_frames_by_timestamps(scene_dir / "iphone" / "rgb.mkv", timestamps, fallback)
            for i, frame in enumerate(frames):
                cv2.imwrite(
                    str(frames_dir / f"{i:06d}.jpg"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, QUALITY],
                )
        results["jpg"] = True

        # 2. generate depth chunk index
        if not index_path.exists():
            depth_path = scene_dir / "iphone" / "depth.bin"
            chunks = _index_depth_bin_chunks(depth_path)
            index_path.write_bytes(pickle.dumps(chunks))
        results["index"] = True

        # 3. upload to COS
        if upload:
            from qcloud_cos import CosConfig, CosS3Client
            secret_id, secret_key = cos_creds
            client = CosS3Client(CosConfig(Region="ap-beijing", SecretId=secret_id, SecretKey=secret_key, Scheme="https"))
            prefix = f"hdu_datasets/scannetpp/data/{scene}"

            # upload frames/
            jpg_files = sorted(frames_dir.glob("*.jpg"))
            for jpg in jpg_files:
                key = f"{prefix}/iphone/frames/{jpg.name}"
                client.upload_file(Bucket=bucket, Key=key, LocalFilePath=str(jpg))

            # upload depth_chunk_index.pkl
            client.put_object(Bucket=bucket, Key=f"{prefix}/iphone/depth_chunk_index.pkl", Body=index_path.read_bytes())

            # upload precomputed.h5
            if h5_path.exists():
                client.upload_file(Bucket=bucket, Key=f"{prefix}/precomputed.h5", LocalFilePath=str(h5_path))
                results["h5"] = True

    except Exception as e:
        results["error"] = str(e)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data3/dataset/scannetpp/data")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--bucket", default="hd-ai-data-1251882982")
    parser.add_argument("--passwd-file", default="/etc/passwd-s3fs-data_cos")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    scene_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    print(f"Total scenes: {len(scene_dirs)}")

    if args.skip_existing:
        scene_dirs = [
            p for p in scene_dirs
            if not (p / "iphone" / "frames").exists()
            or len(list((p / "iphone" / "frames").glob("*.jpg"))) == 0
        ]
        print(f"Scenes to process: {len(scene_dirs)}")

    cos_creds = None
    if args.upload:
        secret_id, secret_key = Path(args.passwd_file).read_text().strip().split(":", 1)
        cos_creds = (secret_id, secret_key)

    task_args = [(p, args.upload, args.bucket, cos_creds) for p in scene_dirs]

    t0 = time.perf_counter()
    done = ok = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_scene, a): a[0].name for a in task_args}
        for fut in as_completed(futures):
            r = fut.result()
            done += 1
            if r["error"]:
                failed += 1
                print(f"[{done}/{len(task_args)}] FAIL {r['scene']}: {r['error']}")
            else:
                ok += 1
                if done % 50 == 0 or done <= 5:
                    elapsed = time.perf_counter() - t0
                    eta = elapsed / done * (len(task_args) - done)
                    print(f"[{done}/{len(task_args)}] OK {r['scene']}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    print(f"\nDone: {ok} ok, {failed} failed, total={time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
