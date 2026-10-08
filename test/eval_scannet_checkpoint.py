#!/usr/bin/env python3
"""Evaluate a D4RT checkpoint on local ScanNet scenes."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.adapters.scannet import ScanNetAdapter
from datasets.transforms import GeometryTransformPipeline, TransformResult
from models import create_d4rt
from utils.camera import compute_relative_pose_error, umeyama_alignment
from utils.metrics import (
    compute_depth_metrics,
    compute_pose_auc,
    compute_pose_metrics,
    mean_shift_align_points,
    paired_coordinate_l1,
)


SCANNET_MAX_DEPTH = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate depth / point-cloud / pose metrics on local ScanNet scenes.")
    parser.add_argument("--root", type=str, default="/home/zbf/16t/f/d4rt/scannet/scannet", help="Local ScanNet root.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--output-dir", type=str, required=True, help="Where summary.json will be written.")
    parser.add_argument("--model-variant", type=str, default="base", choices=("base", "large", "huge", "giant"))
    parser.add_argument("--videomae-model", type=str, default=None, help="Optional local VideoMAE weights path.")
    parser.add_argument("--patch-provider", type=str, default="sampled_resized")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=48, help="Clip length used for evaluation.")
    parser.add_argument("--frame-stride", type=int, default=4, help="Preferred stride when selecting frames from a scene.")
    parser.add_argument("--start-frame", type=int, default=0, help="Preferred start frame inside each scene.")
    parser.add_argument("--scene-offset", type=int, default=0, help="Start from this scene index when --scene-names is not set.")
    parser.add_argument("--num-scenes", type=int, default=3, help="How many scenes to evaluate. <=0 means all scenes.")
    parser.add_argument("--scene-names", type=str, default=None, help="Comma-separated scene names to evaluate.")
    parser.add_argument("--depth-stride", type=int, default=1, help="Spatial stride for depth queries.")
    parser.add_argument("--pointcloud-stride", type=int, default=1, help="Spatial stride for point-cloud queries.")
    parser.add_argument("--max-pointcloud-points", type=int, default=0, help="Cap paired point-cloud points for L1; <=0 keeps all queried points.")
    parser.add_argument("--pose-grid-h", type=int, default=8)
    parser.add_argument("--pose-grid-w", type=int, default=8)
    parser.add_argument("--pose-confidence-threshold", type=float, default=0.0)
    parser.add_argument(
        "--depth-confidence-quantile",
        type=float,
        default=0.0,
        help="Optional bottom confidence quantile to drop for depth. 0 disables filtering for strict paper eval.",
    )
    parser.add_argument(
        "--pointcloud-confidence-quantile",
        type=float,
        default=0.0,
        help="Optional bottom confidence quantile to drop for point cloud. 0 disables filtering for strict paper eval.",
    )
    parser.add_argument("--reference-frame", type=int, default=0, help="Reference frame for point-cloud and pose eval.")
    parser.add_argument("--query-batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--pose-pair-mode",
        type=str,
        default="adjacent",
        choices=("reference", "adjacent"),
        help="adjacent solves (k-1,k) per pair (MonST3R/CUT3R protocol). "
             "reference solves (ref,k) and chains via Sim(3) align (legacy).",
    )
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _strip_compile_prefix(state: dict) -> dict:
    # Checkpoints saved from a torch.compile()-wrapped model keep an "_orig_mod." prefix
    # on every key. Strip it so the state_dict matches the eager model. No-op otherwise.
    if state and all(k.startswith("_orig_mod.") for k in state):
        return {k[len("_orig_mod."):]: v for k, v in state.items()}
    return state


def _infer_patch_size(checkpoint_path: str) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt
    state = _strip_compile_prefix(state)
    w = state.get("encoder.patch_embed.proj.weight")
    if w is not None:
        return tuple(w.shape[2:])
    return (2, 16, 16)


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, torch.device]:
    patch_size = _infer_patch_size(args.checkpoint)
    model_kwargs = dict(
        variant=args.model_variant,
        img_size=args.resolution,
        temporal_size=args.num_frames,
        patch_size=patch_size,
        query_patch_size=9,
        patch_provider=args.patch_provider,
        encoder_pretrained=False,
    )
    if args.videomae_model:
        model_kwargs["videomae_model"] = args.videomae_model
    model = create_d4rt(**model_kwargs)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model") or ckpt.get("model_state_dict")
    if state is None:
        raise KeyError(f"Unsupported checkpoint format: {args.checkpoint}")
    state = _strip_compile_prefix(state)
    model.load_state_dict(state, strict=True)
    try:
        model = model.to(device)
        actual_device = device
    except torch.cuda.OutOfMemoryError:
        if device.type != "cuda":
            raise
        print("[eval] CUDA OOM while moving model to GPU, falling back to CPU.", flush=True)
        model = model.to(torch.device("cpu"))
        actual_device = torch.device("cpu")
    model.eval()
    return model, actual_device


def select_scene_names(adapter: ScanNetAdapter, args: argparse.Namespace) -> list[str]:
    all_names = adapter.list_sequences()
    if args.scene_names:
        requested = [name.strip() for name in args.scene_names.split(",") if name.strip()]
        missing = [name for name in requested if name not in all_names]
        if missing:
            raise ValueError(f"Unknown scene names: {missing}")
        return requested

    if args.num_scenes <= 0:
        return all_names[args.scene_offset:]
    return all_names[args.scene_offset: args.scene_offset + args.num_scenes]


def build_frame_indices(total_frames: int, clip_len: int, frame_stride: int, start_frame: int) -> list[int]:
    """Select clip frames.

    Paper protocol (CUT3R/MonST3R/π³ ScanNet test100): take ``clip_len`` consecutive
    frames sampled with ``frame_stride`` from ``start_frame``. This keeps inter-frame
    motion small (stride=3 ≈ 0.3°/pair RPE-R) and matches reported numbers.

    The legacy ``np.linspace(0, total_frames-1, clip_len)`` mode collapsed the entire
    scene into 48 samples (stride 20-100), inflating ATE/RPE by ~30x.

    Falls back to linspace when ``frame_stride <= 0`` or when the requested span
    exceeds available frames.
    """
    if frame_stride <= 0:
        indices = np.linspace(0, total_frames - 1, clip_len)
        return np.round(indices).astype(np.int64).clip(0, total_frames - 1).tolist()
    end = start_frame + clip_len * frame_stride
    if end <= total_frames:
        return list(range(start_frame, end, frame_stride))
    available = max(total_frames - start_frame, 1)
    stride = max(available // clip_len, 1)
    indices = np.arange(start_frame, start_frame + clip_len * stride, stride)
    return indices.clip(0, total_frames - 1).astype(np.int64).tolist()


def load_transform_result(
    adapter: ScanNetAdapter,
    sequence_name: str,
    frame_indices: list[int],
    resolution: int,
) -> TransformResult:
    clip = adapter.load_clip(sequence_name, frame_indices)
    pipeline = GeometryTransformPipeline(img_size=resolution, use_augs=False)
    return pipeline(clip)


def to_video_tensor(result: TransformResult, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    video_np = np.stack(result.images, axis=0).astype(np.float32)
    video = torch.from_numpy(video_np).unsqueeze(0).to(device)
    crop_h = float(result.crop.crop_h)
    crop_w = float(result.crop.crop_w)
    aspect_ratio = torch.tensor([[crop_w / max(crop_h, 1.0)]], dtype=torch.float32, device=device)
    return video, aspect_ratio


def to_patch_frames_tensor(result: TransformResult, device: torch.device) -> torch.Tensor | None:
    cropped_images = getattr(result, "cropped_images", None)
    if not cropped_images:
        return None

    crop_h, crop_w = cropped_images[0].shape[:2]
    for frame_idx, image in enumerate(cropped_images):
        if image.shape[:2] != (crop_h, crop_w):
            raise ValueError(
                f"inconsistent cropped_images shape at frame {frame_idx}: {image.shape[:2]} vs {(crop_h, crop_w)}"
            )

    if crop_h == result.img_size and crop_w == result.img_size:
        return None

    patch_np = np.stack(cropped_images, axis=0).astype(np.float32)
    return torch.from_numpy(patch_np).permute(0, 3, 1, 2).unsqueeze(0).to(device)


def build_transform_metadata(result: TransformResult, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "canonical_space": torch.tensor([0], dtype=torch.long, device=device),
        "original_hw": torch.tensor(
            [[float(result.original_h), float(result.original_w)]],
            dtype=torch.float32,
            device=device,
        ),
        "crop_offset_xy": torch.tensor(
            [[float(result.crop.x0), float(result.crop.y0)]],
            dtype=torch.float32,
            device=device,
        ),
        "crop_size_hw": torch.tensor(
            [[float(result.crop.crop_h), float(result.crop.crop_w)]],
            dtype=torch.float32,
            device=device,
        ),
        "resized_hw": torch.tensor(
            [[float(result.img_size), float(result.img_size)]],
            dtype=torch.float32,
            device=device,
        ),
    }


def encode_video(
    model: torch.nn.Module,
    result: TransformResult,
    device: torch.device,
    patch_provider: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
    video, aspect_ratio = to_video_tensor(result, device)
    patch_frames = None
    transform_metadata = None
    if patch_provider == "sampled_highres":
        patch_frames = to_patch_frames_tensor(result, device)
        transform_metadata = build_transform_metadata(result, device)

    with torch.inference_mode():
        with autocast_context(device):
            encoder_features = model.encode(video, aspect_ratio)
    frames_bcthw = model._prepare_query_frames(video, patch_frames)
    return encoder_features, frames_bcthw, transform_metadata


def make_query_grid(size: int, stride: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    xs = np.arange(0, size, stride, dtype=np.int32)
    ys = np.arange(0, size, stride, dtype=np.int32)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    coords_norm = np.stack([gx, gy], axis=-1).reshape(-1, 2).astype(np.float32) / max(size - 1, 1)
    coords = torch.from_numpy(coords_norm).to(device)
    return xs, ys, coords


def decode_reference_points(
    model: torch.nn.Module,
    encoder_features: torch.Tensor,
    frames_bcthw: torch.Tensor,
    coords: torch.Tensor,
    frame_idx: int,
    camera_idx: int,
    query_batch_size: int,
    device: torch.device,
    transform_metadata: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_queries = coords.shape[0]
    points_list: list[torch.Tensor] = []
    conf_list: list[torch.Tensor] = []
    for start in range(0, num_queries, query_batch_size):
        end = min(start + query_batch_size, num_queries)
        chunk = coords[start:end].unsqueeze(0)
        sz = end - start
        t_src = torch.full((1, sz), frame_idx, device=device, dtype=torch.long)
        t_tgt = torch.full((1, sz), frame_idx, device=device, dtype=torch.long)
        t_cam = torch.full((1, sz), camera_idx, device=device, dtype=torch.long)
        with torch.inference_mode():
            with autocast_context(device):
                outputs = model.decode(
                    encoder_features,
                    frames_bcthw,
                    chunk,
                    t_src,
                    t_tgt,
                    t_cam,
                    transform_metadata=transform_metadata,
                )
        points_list.append(outputs["pos_3d"][0].detach().float().cpu())
        conf_list.append(outputs["confidence_weight"][0].squeeze(-1).detach().float().cpu())
    return torch.cat(points_list, dim=0), torch.cat(conf_list, dim=0)


def unproject_depth_grid(depth: torch.Tensor, xs: np.ndarray, ys: np.ndarray, intrinsics: torch.Tensor) -> torch.Tensor:
    """Back-project a sparse image grid into camera coordinates."""
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    u = torch.from_numpy(gx.reshape(-1)).float()
    v = torch.from_numpy(gy.reshape(-1)).float()
    d = depth.reshape(-1)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    return torch.stack([x, y, d], dim=-1)


def transform_camera_points(points_cam: torch.Tensor, src_w2c: torch.Tensor, dst_w2c: torch.Tensor) -> torch.Tensor:
    """Express camera-frame points from one camera coordinate system in another."""
    ones = torch.ones((points_cam.shape[0], 1), dtype=points_cam.dtype)
    points_h = torch.cat([points_cam, ones], dim=-1)
    src_c2w = torch.linalg.inv(src_w2c)
    world_h = points_h @ src_c2w.T
    dst_h = world_h @ dst_w2c.T
    return dst_h[:, :3]


def _subsample_paired_points(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    max_points: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsample paired points without breaking the one-to-one correspondence."""

    if max_points <= 0 or pred_points.shape[0] <= max_points:
        return pred_points, gt_points

    generator = torch.Generator().manual_seed(seed)
    keep = torch.randperm(pred_points.shape[0], generator=generator)[:max_points]
    return pred_points[keep], gt_points[keep]


def _prefix_metric_block(prefix: str, metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    """Convert a metric tensor block into JSON-friendly scalar values."""

    return {f"{prefix}_{name}": float(value.item()) for name, value in metrics.items()}


def _apply_confidence_quantile_filter(
    valid: torch.Tensor,
    confidence: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    if quantile <= 0.0 or not valid.any():
        return valid
    q = min(max(float(quantile), 0.0), 1.0)
    conf_thresh = torch.quantile(confidence[valid], q)
    return valid & (confidence >= conf_thresh)


def _per_frame_align_depth(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
    max_depth: float | None = None,
) -> torch.Tensor:
    """Per-frame depth alignment, MonST3R/DUSt3R/CUT3R protocol.

    mode="scale": scale_t = median(gt_t / pred_t) over valid pixels of frame t.
    mode="scale_shift": robust median-centered scale + median-residual shift,
    fitted independently per frame. Each frame gets its own (s, t).
    """
    out = pred.clone()
    if max_depth is not None:
        out = out.clamp(min=1e-6, max=float(max_depth))
    T = pred.shape[0]
    for t in range(T):
        valid_t = mask[t]
        if not valid_t.any():
            continue
        p_t = out[t][valid_t].clamp(min=1e-6)
        g_t = gt[t][valid_t]
        if mode == "scale":
            scale = (g_t / p_t).median()
            if torch.isfinite(scale) and scale > 0:
                out[t] = out[t] * scale
        elif mode == "scale_shift":
            p_med = p_t.median()
            g_med = g_t.median()
            p_centered = p_t - p_med + 1e-6
            g_centered = g_t - g_med + 1e-6
            finite = torch.isfinite(p_centered) & torch.isfinite(g_centered) & (p_centered.abs() > 1e-6)
            if not finite.any():
                scale = (g_t / p_t).median()
                if torch.isfinite(scale) and scale > 0:
                    out[t] = out[t] * scale
                continue
            scale = (g_centered[finite] / p_centered[finite]).median()
            if not torch.isfinite(scale) or scale <= 0:
                scale = (g_t / p_t).median()
                if torch.isfinite(scale) and scale > 0:
                    out[t] = out[t] * scale
                continue
            shift = (g_t - scale * p_t).median()
            out[t] = out[t] * scale + shift
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return out


def compute_depth_metrics_for_clip(
    model: torch.nn.Module,
    result: TransformResult,
    encoder_features: torch.Tensor,
    frames_bcthw: torch.Tensor,
    query_batch_size: int,
    stride: int,
    device: torch.device,
    transform_metadata: dict[str, torch.Tensor] | None,
    confidence_quantile: float = 0.0,
) -> dict[str, float]:
    """Evaluate depth on a clip using depth-style queries.

    For depth, the paper uses queries with ``t_src = t_tgt = t_cam = frame_idx``.
    The resulting 3D point's ``z`` coordinate is treated as depth and evaluated
    under both scale-only (S) and scale-and-shift (SS) alignment.
    """

    xs, ys, coords = make_query_grid(result.img_size, stride, device)
    pred_frames: list[torch.Tensor] = []
    gt_frames: list[torch.Tensor] = []
    mask_frames: list[torch.Tensor] = []

    for frame_idx in range(len(result.images)):
        pred_points, conf = decode_reference_points(
            model,
            encoder_features,
            frames_bcthw,
            coords,
            frame_idx=frame_idx,
            camera_idx=frame_idx,
            query_batch_size=query_batch_size,
            device=device,
            transform_metadata=transform_metadata,
        )
        pred_depth = pred_points[:, 2].reshape(len(ys), len(xs))
        conf_map = conf.reshape(len(ys), len(xs))
        gt_depth = torch.from_numpy(result.depths[frame_idx][ys][:, xs]).float()
        valid = torch.isfinite(gt_depth) & (gt_depth > 0.0) & (gt_depth < SCANNET_MAX_DEPTH) & torch.isfinite(pred_depth)
        valid = _apply_confidence_quantile_filter(valid, conf_map, confidence_quantile)
        pred_frames.append(pred_depth)
        gt_frames.append(gt_depth)
        mask_frames.append(valid)

    pred = torch.stack(pred_frames, dim=0)
    gt = torch.stack(gt_frames, dim=0)
    mask = torch.stack(mask_frames, dim=0)
    pred = torch.clamp(pred, max=SCANNET_MAX_DEPTH)
    # Per-frame median alignment (MonST3R/DUSt3R/CUT3R protocol). The previous
    # clip-level IRLS alignment (scale_invariant=True / shift_invariant=True)
    # over a 188-frame clip biased the global (s, t) toward dominant near
    # surfaces and inflated AbsRel.
    pred_s = _per_frame_align_depth(pred, gt, mask, mode="scale", max_depth=SCANNET_MAX_DEPTH)
    pred_ss = _per_frame_align_depth(pred, gt, mask, mode="scale_shift", max_depth=SCANNET_MAX_DEPTH)
    metrics_s = compute_depth_metrics(pred_s, gt, mask=mask, scale_invariant=False, shift_invariant=False, max_depth=SCANNET_MAX_DEPTH)
    metrics_ss = compute_depth_metrics(pred_ss, gt, mask=mask, scale_invariant=False, shift_invariant=False, max_depth=SCANNET_MAX_DEPTH)

    out = {
        "depth_valid_ratio": float(mask.float().mean().item()),
        "depth_num_valid_pixels": float(mask.sum().item()),
    }
    out.update(_prefix_metric_block("depth_S", metrics_s))
    out.update(_prefix_metric_block("depth_SS", metrics_ss))
    return out


def compute_pointcloud_l1_for_clip(
    model: torch.nn.Module,
    result: TransformResult,
    encoder_features: torch.Tensor,
    frames_bcthw: torch.Tensor,
    query_batch_size: int,
    stride: int,
    reference_frame: int,
    max_points: int,
    seed: int,
    device: torch.device,
    transform_metadata: dict[str, torch.Tensor] | None,
    confidence_quantile: float = 0.0,
) -> dict[str, float]:
    """Evaluate paper-style point-cloud L1 on a clip.

    Each frame is decoded into a shared reference camera coordinate system by
    fixing ``t_cam = reference_frame`` while keeping ``t_src = t_tgt = frame``.
    GT depths are back-projected and transformed into the same reference frame.
    The final score is the paper's mean-shift alignment followed by paired
    coordinate-wise L1.
    """

    xs, ys, coords = make_query_grid(result.img_size, stride, device)
    pred_points_all: list[torch.Tensor] = []
    gt_points_all: list[torch.Tensor] = []

    ref_w2c = torch.from_numpy(result.extrinsics[reference_frame]).float()

    for frame_idx in range(len(result.images)):
        pred_points, conf = decode_reference_points(
            model,
            encoder_features,
            frames_bcthw,
            coords,
            frame_idx=frame_idx,
            camera_idx=reference_frame,
            query_batch_size=query_batch_size,
            device=device,
            transform_metadata=transform_metadata,
        )

        gt_depth = torch.from_numpy(result.depths[frame_idx][ys][:, xs]).float()
        gt_cam = unproject_depth_grid(
            gt_depth,
            xs,
            ys,
            torch.from_numpy(result.intrinsics[frame_idx]).float(),
        )
        gt_ref = transform_camera_points(
            gt_cam,
            torch.from_numpy(result.extrinsics[frame_idx]).float(),
            ref_w2c,
        )

        valid = (gt_depth.reshape(-1) > 0.0) & (gt_depth.reshape(-1) < SCANNET_MAX_DEPTH) & torch.isfinite(gt_cam).all(dim=-1) & torch.isfinite(pred_points).all(dim=-1)
        valid = _apply_confidence_quantile_filter(valid, conf, confidence_quantile)
        if valid.any():
            pred_points_all.append(pred_points[valid])
            gt_points_all.append(gt_ref[valid])

    if not pred_points_all:
        return {
            "pointcloud_num_points": 0.0,
            "pointcloud_l1": float("nan"),
        }

    pred_points = torch.cat(pred_points_all, dim=0)
    gt_points = torch.cat(gt_points_all, dim=0)
    pred_points, gt_points = _subsample_paired_points(pred_points, gt_points, max_points=max_points, seed=seed)

    pred_aligned, gt_aligned = mean_shift_align_points(pred_points, gt_points)
    l1_meanshift = paired_coordinate_l1(pred_aligned, gt_aligned)

    R_sim3, t_sim3, s_sim3 = umeyama_alignment(pred_points, gt_points, with_scale=True)
    pred_sim3 = s_sim3 * (pred_points @ R_sim3.T) + t_sim3
    l1_sim3 = paired_coordinate_l1(pred_sim3, gt_points)

    return {
        "pointcloud_num_points": float(pred_points.shape[0]),
        "pointcloud_l1": float(l1_meanshift.item()),
        "pointcloud_l1_sim3": float(l1_sim3.item()),
        "pointcloud_sim3_scale": float(s_sim3.item()),
    }


def estimate_relative_pose(
    model: torch.nn.Module,
    encoder_features: torch.Tensor,
    frames_bcthw: torch.Tensor,
    frame_i: int,
    frame_j: int,
    grid_h: int,
    grid_w: int,
    confidence_threshold: float,
    device: torch.device,
    transform_metadata: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    """Estimate relative pose from query-decoded 3D correspondences.

    We decode the same reference-frame grid twice:
    - once in camera ``i`` coordinates
    - once in camera ``j`` coordinates

    A weighted rigid Umeyama solve then recovers the camera transform between
    the two coordinate systems.
    """

    u = torch.linspace(0.1, 0.9, grid_w, device=device)
    v = torch.linspace(0.1, 0.9, grid_h, device=device)
    grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")
    coords = torch.stack([grid_u, grid_v], dim=-1).reshape(1, -1, 2)
    num_points = coords.shape[1]

    t_i = torch.full((1, num_points), frame_i, device=device, dtype=torch.long)
    t_j = torch.full((1, num_points), frame_j, device=device, dtype=torch.long)

    with torch.inference_mode():
        with autocast_context(device):
            outputs_i = model.decode(
                encoder_features,
                frames_bcthw,
                coords,
                t_i,
                t_i,
                t_i,
                transform_metadata=transform_metadata,
            )
            outputs_j = model.decode(
                encoder_features,
                frames_bcthw,
                coords,
                t_i,
                t_i,
                t_j,
                transform_metadata=transform_metadata,
            )

    points_i = outputs_i["pos_3d"].squeeze(0).detach().float()
    points_j = outputs_j["pos_3d"].squeeze(0).detach().float()
    weights = (
        outputs_i["confidence_weight"].squeeze(0).squeeze(-1).detach().float()
        * outputs_j["confidence_weight"].squeeze(0).squeeze(-1).detach().float()
    )

    valid = (
        torch.isfinite(points_i).all(dim=-1)
        & torch.isfinite(points_j).all(dim=-1)
        & torch.isfinite(weights)
        & (weights > confidence_threshold)
    )
    if valid.sum() < 4:
        return torch.eye(4, dtype=torch.float32)

    R, t, _ = umeyama_alignment(points_j[valid], points_i[valid], weights[valid], with_scale=False)
    R_ij = R.T
    t_ij = -R_ij @ t

    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, :3] = R_ij.cpu()
    pose[:3, 3] = t_ij.cpu()
    return pose


def compute_pose_metrics_for_clip(
    model: torch.nn.Module,
    result: TransformResult,
    encoder_features: torch.Tensor,
    frames_bcthw: torch.Tensor,
    reference_frame: int,
    grid_h: int,
    grid_w: int,
    confidence_threshold: float,
    device: torch.device,
    transform_metadata: dict[str, torch.Tensor] | None,
    pair_mode: str = "reference",
) -> dict[str, float]:
    """Evaluate trajectory pose quality for one clip.

    pair_mode="reference" solves (ref, k) per frame and chains pred trajectory
    via Sim(3) align (legacy). RPE between consecutive aligned frames is biased
    toward zero because adjacent (ref, k) and (ref, k+1) solutions share the
    same source frame and the Sim(3) align renormalizes them.

    pair_mode="adjacent" solves (k-1, k) per frame, chains the steps directly,
    and matches the MonST3R/CUT3R Sintel/ScanNet protocol.
    """

    gt_w2c = torch.from_numpy(result.extrinsics).float()
    gt_c2w = torch.linalg.inv(gt_w2c)
    ref_w2c = gt_w2c[reference_frame]
    n = len(result.images)

    pred_w2c_list: list[torch.Tensor] = []
    rot_errors: list[torch.Tensor] = []
    trans_errors: list[torch.Tensor] = []

    if pair_mode == "reference":
        for frame_idx in range(n):
            if frame_idx == reference_frame:
                pred_w2c_rel = torch.eye(4, dtype=torch.float32)
            else:
                pred_w2c_rel = estimate_relative_pose(
                    model,
                    encoder_features,
                    frames_bcthw,
                    frame_i=reference_frame,
                    frame_j=frame_idx,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    confidence_threshold=confidence_threshold,
                    device=device,
                    transform_metadata=transform_metadata,
                )

                gt_w2c_rel = gt_w2c[frame_idx] @ torch.linalg.inv(ref_w2c)
                rel_errors = compute_relative_pose_error(
                    pred_w2c_rel[:3, :3],
                    pred_w2c_rel[:3, 3],
                    gt_w2c_rel[:3, :3],
                    gt_w2c_rel[:3, 3],
                )
                rot_errors.append(rel_errors["rotation_error"])
                trans_errors.append(rel_errors["translation_error"])

            pred_w2c_list.append(pred_w2c_rel)
    elif pair_mode == "adjacent":
        current = torch.eye(4, dtype=torch.float32)
        pred_w2c_list.append(current.clone())
        for frame_idx in range(1, n):
            step = estimate_relative_pose(
                model,
                encoder_features,
                frames_bcthw,
                frame_i=frame_idx - 1,
                frame_j=frame_idx,
                grid_h=grid_h,
                grid_w=grid_w,
                confidence_threshold=confidence_threshold,
                device=device,
                transform_metadata=transform_metadata,
            )
            current = step @ current
            gt_step = gt_w2c[frame_idx] @ torch.linalg.inv(gt_w2c[frame_idx - 1])
            rel_errors = compute_relative_pose_error(
                step[:3, :3], step[:3, 3],
                gt_step[:3, :3], gt_step[:3, 3],
            )
            rot_errors.append(rel_errors["rotation_error"])
            trans_errors.append(rel_errors["translation_error"])
            pred_w2c_list.append(current.clone())
    else:
        raise ValueError(f"Unknown pair_mode: {pair_mode}")

    pred_c2w = torch.linalg.inv(torch.stack(pred_w2c_list, dim=0))
    pose_metrics = compute_pose_metrics(pred_c2w, gt_c2w, align=True)

    if rot_errors:
        pose_auc_30 = compute_pose_auc(torch.stack(rot_errors), torch.stack(trans_errors), threshold=30.0)
    else:
        pose_auc_30 = torch.tensor(float("nan"))

    return {
        "pose_ate": float(pose_metrics["ate"].item()),
        "pose_rpe_trans": float(pose_metrics["rpe_trans"].item()),
        "pose_rpe_rot": float(pose_metrics["rpe_rot"].item()),
        "pose_auc_30": float(pose_auc_30.item() * 100.0),
        "pose_auc_30_frac": float(pose_auc_30.item()),
        "pose_num_pairs": float(len(rot_errors)),
    }


def evaluate_scene(
    adapter: ScanNetAdapter,
    model: torch.nn.Module,
    scene_name: str,
    args: argparse.Namespace,
    device: torch.device,
    scene_index: int,
) -> dict[str, Any]:
    info = adapter.get_sequence_info(scene_name)
    total_frames = int(info["num_frames"])
    frame_indices = build_frame_indices(total_frames, args.num_frames, args.frame_stride, args.start_frame)
    result = load_transform_result(adapter, scene_name, frame_indices, args.resolution)

    if not (0 <= args.reference_frame < len(result.images)):
        raise ValueError(f"reference_frame={args.reference_frame} out of range for clip length {len(result.images)}")

    encoder_features, frames_bcthw, transform_metadata = encode_video(model, result, device, args.patch_provider)
    depth_metrics = compute_depth_metrics_for_clip(
        model,
        result,
        encoder_features,
        frames_bcthw,
        query_batch_size=args.query_batch_size,
        stride=args.depth_stride,
        device=device,
        transform_metadata=transform_metadata,
        confidence_quantile=args.depth_confidence_quantile,
    )
    pointcloud_metrics = compute_pointcloud_l1_for_clip(
        model,
        result,
        encoder_features,
        frames_bcthw,
        query_batch_size=args.query_batch_size,
        stride=args.pointcloud_stride,
        reference_frame=args.reference_frame,
        max_points=args.max_pointcloud_points,
        seed=args.seed + scene_index,
        device=device,
        transform_metadata=transform_metadata,
        confidence_quantile=args.pointcloud_confidence_quantile,
    )
    pose_metrics = compute_pose_metrics_for_clip(
        model,
        result,
        encoder_features,
        frames_bcthw,
        reference_frame=args.reference_frame,
        grid_h=args.pose_grid_h,
        grid_w=args.pose_grid_w,
        confidence_threshold=args.pose_confidence_threshold,
        device=device,
        transform_metadata=transform_metadata,
        pair_mode=args.pose_pair_mode,
    )

    return {
        "scene_name": scene_name,
        "frame_indices": frame_indices,
        **depth_metrics,
        **pointcloud_metrics,
        **pose_metrics,
    }


def summarize_scene_metrics(scene_summaries: list[dict[str, Any]]) -> dict[str, float]:
    if not scene_summaries:
        return {}
    metric_keys = [
        key for key in scene_summaries[0].keys()
        if key not in {"scene_name", "frame_indices"}
    ]
    out: dict[str, float] = {}
    for key in metric_keys:
        values = [float(item[key]) for item in scene_summaries if math.isfinite(float(item[key]))]
        out[key] = float(sum(values) / len(values)) if values else float("nan")
    return out


def main() -> int:
    args = parse_args()
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = ScanNetAdapter(root=args.root, precompute_root=args.root)
    scene_names = select_scene_names(adapter, args)
    if not scene_names:
        raise RuntimeError("No ScanNet scenes selected for evaluation.")

    print(f"[eval] device={device}")
    print(f"[eval] checkpoint={args.checkpoint}")
    print(f"[eval] scenes={scene_names}")

    model, device = load_model(args, device)

    scene_summaries: list[dict[str, Any]] = []
    failed_scenes: list[dict[str, str]] = []
    for scene_index, scene_name in enumerate(scene_names):
        print(f"[eval] scene {scene_index + 1}/{len(scene_names)}: {scene_name}", flush=True)
        try:
            summary = evaluate_scene(adapter, model, scene_name, args, device, scene_index)
        except Exception as exc:  # noqa: BLE001
            failure = {"scene_name": scene_name, "error": repr(exc)}
            failed_scenes.append(failure)
            print(json.dumps({"scene_name": scene_name, "status": "failed", "error": repr(exc)}, ensure_ascii=False), flush=True)
            continue
        scene_summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    strict_metric_settings = (
        args.depth_confidence_quantile <= 0.0
        and args.pointcloud_confidence_quantile <= 0.0
        and args.pose_confidence_threshold <= 0.0
    )
    paper_density_settings = (
        args.depth_stride == 1
        and args.pointcloud_stride == 1
        and args.max_pointcloud_points <= 0
    )

    final_summary = {
        "root": args.root,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "num_requested_scenes": len(scene_names),
        "num_scenes": len(scene_summaries),
        "num_failed_scenes": len(failed_scenes),
        "protocol": {
            "paper_aligned": strict_metric_settings and paper_density_settings,
            "paper_metric_settings": strict_metric_settings,
            "paper_density_settings": paper_density_settings,
            "depth": "global sequence scale-only and scale-and-shift alignment, AbsRel reported",
            "pointcloud": "shared reference camera, mean-shift translation alignment, paired coordinate Mean L1",
            "pose": "rigid Umeyama relative poses, trajectory Sim(3) alignment, ATE/RPE/Pose AUC@30",
            "depth_confidence_quantile": args.depth_confidence_quantile,
            "pointcloud_confidence_quantile": args.pointcloud_confidence_quantile,
            "pose_confidence_threshold": args.pose_confidence_threshold,
            "depth_stride": args.depth_stride,
            "pointcloud_stride": args.pointcloud_stride,
            "max_pointcloud_points": args.max_pointcloud_points,
        },
        "scene_summaries": scene_summaries,
        "failed_scenes": failed_scenes,
        "mean_metrics": summarize_scene_metrics(scene_summaries),
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps({"summary_path": str(summary_path), "mean_metrics": final_summary["mean_metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
