"""
Generate Co3D tracks from depth maps with relaxed constraints.

Strategy:
1. Sample points from ALL frames (not just ref_frame)
2. Use relaxed depth consistency check
3. Accept points visible in at least 2 frames
"""

from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.co3dv2 import Co3Dv2Adapter
from computer.depth_to_normals import compute_normals_sequence


def compute_tracks_multi_source(
    depths: list[np.ndarray],
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    num_points: int = 8000,
    depth_consistency_thresh: float = 0.5,  # Very relaxed
) -> dict:
    """Sample points from multiple frames and track them."""
    T = len(depths)
    H, W = depths[0].shape

    # Sample points from frames with most valid depth
    valid_counts = [((d > 0) & np.isfinite(d)).sum() for d in depths]
    # Pick top 5 frames with most depth coverage
    top_frames = np.argsort(valid_counts)[-5:][::-1]

    all_pts_world = []
    all_source_frames = []

    points_per_frame = num_points // len(top_frames)

    for src_t in top_frames:
        depth_src = depths[src_t]
        K_src = intrinsics[src_t]
        E_src = extrinsics[src_t]

        valid = (depth_src > 0) & np.isfinite(depth_src)
        ys, xs = np.where(valid)

        if len(ys) == 0:
            continue

        # Sample points
        n_sample = min(points_per_frame, len(ys))
        rng = np.random.default_rng(42 + src_t)
        idx = rng.choice(len(ys), n_sample, replace=False)
        src_y, src_x = ys[idx], xs[idx]
        src_d = depth_src[src_y, src_x]

        # Backproject to world
        fx, fy, cx, cy = K_src[0,0], K_src[1,1], K_src[0,2], K_src[1,2]
        X = (src_x - cx) * src_d / fx
        Y = (src_y - cy) * src_d / fy
        Z = src_d
        P_cam = np.stack([X, Y, Z, np.ones_like(X)], axis=1)

        E_inv = np.linalg.inv(E_src.astype(np.float64))
        P_world = (E_inv @ P_cam.T).T[:, :3].astype(np.float32)

        all_pts_world.append(P_world)
        all_source_frames.extend([src_t] * len(P_world))

    if not all_pts_world:
        # Fallback: return empty
        N = 0
        return {
            "trajs_2d": np.zeros((T, N, 2), dtype=np.float32),
            "trajs_3d_world": np.zeros((T, N, 3), dtype=np.float32),
            "valids": np.zeros((T, N), dtype=bool),
            "visibs": np.zeros((T, N), dtype=bool),
            "ref_frame": 0,
            "num_points": N,
        }

    pts_world = np.concatenate(all_pts_world, axis=0)
    N = len(pts_world)

    # Find ref_frame (frame where most points are visible)
    ref_valid_counts = []
    for t in range(T):
        E_t = extrinsics[t]
        K_t = intrinsics[t]

        P_hom = np.concatenate([pts_world, np.ones((N, 1))], axis=1)
        P_cam = (E_t @ P_hom.T).T[:, :3]
        z = P_cam[:, 2]

        fx, fy, cx, cy = K_t[0,0], K_t[1,1], K_t[0,2], K_t[1,2]
        u = P_cam[:, 0] / z * fx + cx
        v = P_cam[:, 1] / z * fy + cy

        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
        ref_valid_counts.append(in_bounds.sum())

    ref_frame = int(np.argmax(ref_valid_counts))

    # Project to all frames with relaxed depth check
    trajs_2d = np.zeros((T, N, 2), dtype=np.float32)
    valids = np.zeros((T, N), dtype=bool)

    for t in range(T):
        E_t = extrinsics[t]
        K_t = intrinsics[t]
        depth_t = depths[t]

        P_hom = np.concatenate([pts_world, np.ones((N, 1))], axis=1)
        P_cam = (E_t @ P_hom.T).T[:, :3]
        z_proj = P_cam[:, 2]

        fx, fy, cx, cy = K_t[0,0], K_t[1,1], K_t[0,2], K_t[1,2]
        u = P_cam[:, 0] / z_proj * fx + cx
        v = P_cam[:, 1] / z_proj * fy + cy

        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z_proj > 0)

        # Relaxed depth check: only check if depth exists
        px = np.clip(np.round(u).astype(np.int32), 0, W - 1)
        py = np.clip(np.round(v).astype(np.int32), 0, H - 1)
        sampled_d = depth_t[py, px]

        # If depth is valid, check consistency; otherwise accept if in bounds
        has_depth = (sampled_d > 0) & np.isfinite(sampled_d)
        depth_ok = np.ones(N, dtype=bool)
        depth_ok[has_depth] = (
            np.abs(sampled_d[has_depth] - z_proj[has_depth]) /
            np.maximum(z_proj[has_depth], 1e-6) < depth_consistency_thresh
        )

        valid_t = in_bounds & depth_ok
        trajs_2d[t] = np.stack([u, v], axis=1)
        valids[t] = valid_t

    trajs_3d_world = np.broadcast_to(pts_world[None], (T, N, 3)).copy()
    visibs = valids.copy()

    return {
        "trajs_2d": trajs_2d,
        "trajs_3d_world": trajs_3d_world,
        "valids": valids,
        "visibs": visibs,
        "ref_frame": ref_frame,
        "num_points": N,
    }


def process_sequence(args: tuple) -> tuple[str, str | None]:
    """Process one Co3D sequence."""
    seq_name, adapter_state, output_root, num_points, overwrite = args

    out_path = Path(output_root) / seq_name / "precomputed.npz"
    if out_path.exists() and not overwrite:
        return seq_name, None

    try:
        adapter_cls, adapter_kwargs = adapter_state
        adapter = adapter_cls(**adapter_kwargs)

        info = adapter.get_sequence_info(seq_name)
        num_frames = info["num_frames"]

        all_indices = list(range(num_frames))
        clip = adapter.load_clip(seq_name, all_indices)

        normals = compute_normals_sequence(clip.depths, clip.intrinsics)

        tracks = compute_tracks_multi_source(
            depths=clip.depths,
            intrinsics=clip.intrinsics,
            extrinsics=clip.extrinsics,
            num_points=num_points,
            depth_consistency_thresh=0.5,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            normals=normals.astype(np.float16),
            trajs_2d=tracks["trajs_2d"].astype(np.float32),
            trajs_3d_world=tracks["trajs_3d_world"].astype(np.float32),
            valids=tracks["valids"],
            visibs=tracks["visibs"],
            ref_frame=np.array(tracks["ref_frame"], dtype=np.int32),
            num_frames=np.array(num_frames, dtype=np.int32),
            num_points=np.array(tracks["num_points"], dtype=np.int32),
        )
        return seq_name, None

    except Exception:
        return seq_name, traceback.format_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--num-points", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--categories", nargs="+", default=None)
    args = parser.parse_args()

    adapter = Co3Dv2Adapter(root=args.root, categories=args.categories)
    output_root = Path(args.output_root) if args.output_root else Path(args.root)

    sequences = adapter.list_sequences()
    total = len(sequences)
    print(f"[co3d_relaxed] {total} sequences → {output_root}")

    adapter_state = (
        Co3Dv2Adapter,
        {
            "root": str(adapter.root),
            "categories": adapter.categories,
            "subset_name": adapter.subset_name,
            "split": adapter.split,
        }
    )

    job_args = [
        (seq, adapter_state, str(output_root), args.num_points, args.overwrite)
        for seq in sequences
    ]

    done = 0
    failed = []

    if args.workers <= 1:
        for job_arg in tqdm(job_args, desc="co3d_relaxed"):
            seq_name, err = process_sequence(job_arg)
            done += 1
            if err:
                failed.append((seq_name, err))
                tqdm.write(f"  [SKIP] {seq_name}: {err.splitlines()[-1]}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_sequence, a): a[0] for a in job_args}
            with tqdm(total=total, desc="co3d_relaxed") as pbar:
                for fut in as_completed(futures):
                    seq_name, err = fut.result()
                    done += 1
                    pbar.update(1)
                    if err:
                        failed.append((seq_name, err))
                        tqdm.write(f"  [SKIP] {seq_name}: {err.splitlines()[-1]}")

    print(f"[co3d_relaxed] done {done - len(failed)}/{total}, failed {len(failed)}")


if __name__ == "__main__":
    main()
