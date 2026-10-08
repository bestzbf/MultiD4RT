#!/usr/bin/env python3
"""Evaluate a D4RT checkpoint on MPI Sintel (final pass)."""

from __future__ import annotations

import argparse
import json
import math
import struct
from contextlib import nullcontext
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

# Standard 14-sequence subset (final pass, following CUT3R / Geo4D / MonST3R)
SINTEL_14_SEQUENCES = [
    "alley_1", "alley_2", "ambush_2", "ambush_4", "ambush_5",
    "ambush_6", "ambush_7", "bamboo_1", "bamboo_2", "bandage_1",
    "bandage_2", "cave_2", "cave_4", "market_2",
]

# MonST3R/DUSt3R Sintel protocol: cap GT and clip pred at 70m to exclude sky.
# Sintel fills sky with a large constant (~2500m for ambush_2); the previous 1000m
# threshold let ~26% sky pixels through and dominated AbsRel(S).
SINTEL_MAX_DEPTH = 70.0
SINTEL_PRED_CLIP_MAX = 70.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate depth / point-cloud / pose metrics on MPI Sintel.")
    parser.add_argument("--root", type=str, default="/data3/dataset/sintel", help="Sintel dataset root.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--output-dir", type=str, required=True, help="Where summary.json will be written.")
    parser.add_argument("--model-variant", type=str, default="base", choices=("base", "large", "huge", "giant"))
    parser.add_argument("--videomae-model", type=str, default=None, help="Optional local VideoMAE weights path.")
    parser.add_argument("--patch-provider", type=str, default="sampled_resized")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=48, help="Clip length used for evaluation.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Preferred stride when selecting frames.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-scenes", type=int, default=0, help="<=0 means all scenes.")
    parser.add_argument("--scene-names", type=str, default=None, help="Comma-separated scene names.")
    parser.add_argument("--use-14-subset", action="store_true", default=True, help="Use the 14-sequence subset for eval.")
    parser.add_argument("--all-scenes", action="store_true", default=False, help="Use all 23 scenes.")
    parser.add_argument("--depth-stride", type=int, default=1)
    parser.add_argument("--pointcloud-stride", type=int, default=1)
    parser.add_argument("--max-pointcloud-points", type=int, default=0)
    parser.add_argument("--pose-grid-h", type=int, default=8)
    parser.add_argument("--pose-grid-w", type=int, default=8)
    parser.add_argument("--pose-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--pose-ransac-iters", type=int, default=256)
    parser.add_argument("--pose-inlier-thresh", type=float, default=0.1)
    parser.add_argument("--pose-gt-depth-mask", action="store_true",
                        help="Diagnostic: drop pose-grid points where Sintel GT depth is sky/invalid.")
    parser.add_argument("--pose-gt-depth-mask-threshold", type=float, default=70.0,
                        help="Upper-bound GT depth (m) for --pose-gt-depth-mask. 70 = strict; 1000 ≈ sky-only.")
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--query-batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sintel-pass", type=str, default="final", choices=("clean", "final"))
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
    parser.add_argument(
        "--pose-solver",
        type=str,
        default="umeyama",
        choices=("umeyama", "ransac"),
        help="Rigid Umeyama is the paper method; RANSAC is a diagnostic/tuned option.",
    )
    parser.add_argument(
        "--pose-pair-mode",
        type=str,
        default="adjacent",
        choices=("reference", "adjacent"),
        help="adjacent solves (k-1,k) per pair (MonST3R/CUT3R protocol). "
             "reference solves (ref,k) and chains via Sim(3) align (legacy).",
    )
    parser.add_argument(
        "--frame-mode",
        type=str,
        default="unique",
        choices=("linspace", "unique"),
        help="unique uses real frame ordering and pads to clip_len for encoder; "
             "linspace round-resamples short sequences and creates duplicate "
             "indices that artificially zero out RPE between adjacent duplicates.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Sintel data loading
# ---------------------------------------------------------------------------

def read_sintel_dpt(path: Path) -> np.ndarray:
    """Read a Sintel .dpt depth file (official SDK format)."""
    TAG_FLOAT = 202021.25
    with open(path, "rb") as f:
        magic = np.fromfile(f, dtype=np.float32, count=1)[0]
        if abs(magic - TAG_FLOAT) > 1e-4:
            raise ValueError(f"Invalid .dpt magic: {magic}")
        w = np.fromfile(f, dtype=np.int32, count=1)[0]
        h = np.fromfile(f, dtype=np.int32, count=1)[0]
        data = np.fromfile(f, dtype=np.float32, count=h * w)
    return data.reshape(h, w)


def read_sintel_cam(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a Sintel .cam camera file.

    Format (from official SDK):
      - float32 magic (202021.25)
      - 3x3 float64 intrinsic matrix M
      - 3x4 float64 extrinsic matrix N  (world-to-camera [R|t])

    Returns (extrinsic_4x4, intrinsic_3x3) both as float32.
    The extrinsic is world-to-camera (w2c) as a 4x4 matrix.
    """
    TAG_FLOAT = 202021.25
    with open(path, "rb") as f:
        magic = np.fromfile(f, dtype=np.float32, count=1)[0]
        if abs(magic - TAG_FLOAT) > 1e-4:
            raise ValueError(f"Invalid .cam magic: {magic}")
        intrinsic = np.fromfile(f, dtype=np.float64, count=9).reshape(3, 3)
        extrinsic_34 = np.fromfile(f, dtype=np.float64, count=12).reshape(3, 4)
    # Pad to 4x4
    extrinsic_44 = np.eye(4, dtype=np.float64)
    extrinsic_44[:3, :] = extrinsic_34
    return extrinsic_44.astype(np.float32), intrinsic.astype(np.float32)


class SintelScene:
    """Minimal scene loader for Sintel evaluation."""

    def __init__(self, root: Path, scene_name: str, sintel_pass: str = "final"):
        self.image_dir = root / "training" / sintel_pass / scene_name
        self.depth_dir = root / "training" / "depth" / scene_name
        self.cam_dir = root / "training" / "camdata_left" / scene_name

        self.image_files = sorted(self.image_dir.glob("frame_*.png"))
        if not self.image_files:
            raise FileNotFoundError(f"No images found in {self.image_dir}")
        self.num_frames = len(self.image_files)

    def load_clip(
        self, frame_indices: list[int], resolution: int
    ) -> TransformResult:
        from PIL import Image

        images = []
        depths = []
        intrinsics_list = []
        extrinsics_list = []

        for idx in frame_indices:
            img_path = self.image_files[idx]
            frame_name = img_path.stem

            img = np.array(Image.open(img_path))
            images.append(img)

            depth_path = self.depth_dir / f"{frame_name}.dpt"
            if depth_path.exists():
                depth = read_sintel_dpt(depth_path)
            else:
                depth = np.full(img.shape[:2], float("nan"), dtype=np.float32)
            depths.append(depth)

            cam_path = self.cam_dir / f"{frame_name}.cam"
            if cam_path.exists():
                ext, intr = read_sintel_cam(cam_path)
                extrinsics_list.append(ext.astype(np.float32))
                intr_scaled = self._scale_intrinsics(
                    intr.astype(np.float32), img.shape[:2], resolution
                )
                intrinsics_list.append(intr_scaled)
            else:
                extrinsics_list.append(np.eye(4, dtype=np.float32))
                intrinsics_list.append(np.eye(3, dtype=np.float32))

        pipeline = GeometryTransformPipeline(img_size=resolution, use_augs=False)
        from datasets.adapters.base import UnifiedClip
        clip = UnifiedClip(
            dataset_name="sintel",
            sequence_name="",
            frame_paths=None,
            images=images,
            depths=depths,
            normals=None,
            trajs_2d=None,
            trajs_3d_world=None,
            valids=None,
            visibs=None,
            intrinsics=np.array(intrinsics_list),
            extrinsics=np.array(extrinsics_list),
        )
        result = pipeline(clip)
        return result

    @staticmethod
    def _scale_intrinsics(
        K: np.ndarray, original_hw: tuple[int, int], target_size: int
    ) -> np.ndarray:
        """Intrinsics will be rescaled by the pipeline; return original."""
        return K


# ---------------------------------------------------------------------------
# Model loading (same as ScanNet eval)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Encoding / decoding helpers (shared with ScanNet eval)
# ---------------------------------------------------------------------------

def to_video_tensor(result: TransformResult, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    video_np = np.stack(result.images, axis=0).astype(np.float32)
    video = torch.from_numpy(video_np).unsqueeze(0).to(device)
    # Always 1.0: video is resized to img_size×img_size regardless of original aspect ratio
    aspect_ratio = torch.tensor([[1.0]], dtype=torch.float32, device=device)
    return video, aspect_ratio


def to_patch_frames_tensor(result: TransformResult, device: torch.device) -> torch.Tensor | None:
    cropped_images = getattr(result, "cropped_images", None)
    if not cropped_images:
        return None
    crop_h, crop_w = cropped_images[0].shape[:2]
    if crop_h == result.img_size and crop_w == result.img_size:
        return None
    patch_np = np.stack(cropped_images, axis=0).astype(np.float32)
    return torch.from_numpy(patch_np).permute(0, 3, 1, 2).unsqueeze(0).to(device)


def build_transform_metadata(result: TransformResult, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "canonical_space": torch.tensor([0], dtype=torch.long, device=device),
        "original_hw": torch.tensor(
            [[float(result.original_h), float(result.original_w)]],
            dtype=torch.float32, device=device,
        ),
        "crop_offset_xy": torch.tensor(
            [[float(result.crop.x0), float(result.crop.y0)]],
            dtype=torch.float32, device=device,
        ),
        "crop_size_hw": torch.tensor(
            [[float(result.crop.crop_h), float(result.crop.crop_w)]],
            dtype=torch.float32, device=device,
        ),
        "resized_hw": torch.tensor(
            [[float(result.img_size), float(result.img_size)]],
            dtype=torch.float32, device=device,
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
    model, encoder_features, frames_bcthw, coords, frame_idx, camera_idx,
    query_batch_size, device, transform_metadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_queries = coords.shape[0]
    points_list, conf_list = [], []
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
                    encoder_features, frames_bcthw, chunk,
                    t_src, t_tgt, t_cam,
                    transform_metadata=transform_metadata,
                )
        points_list.append(outputs["pos_3d"][0].detach().float().cpu())
        conf_list.append(outputs["confidence_weight"][0].squeeze(-1).detach().float().cpu())
    return torch.cat(points_list, dim=0), torch.cat(conf_list, dim=0)


def decode_frames_batched(
    model, encoder_features, frames_bcthw, coords,
    frame_indices, camera_indices,
    query_batch_size, device, transform_metadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode the SAME query grid for many frames in one fused query stream.

    Per-query outputs are independent given (coords, t_src, t_tgt, t_cam) and the
    (B=1) encoder_features row, so flattening all frames' queries into one large
    N dimension is numerically identical to looping ``decode_reference_points``
    frame-by-frame — it just collapses ~n_frames*ceil(G/qbs) tiny decode launches
    (each followed by a GPU->CPU sync) into ceil(n_frames*G/qbs) large ones,
    accumulating on-GPU and copying to CPU once.

    Args:
        coords: (G, 2) normalized query grid, shared across frames.
        frame_indices: length-n list/seq of t_src (= t_tgt) per frame.
        camera_indices: length-n list/seq of t_cam per frame.

    Returns:
        points: (n, G, 3) on CPU.
        conf:   (n, G)   on CPU.
    """
    n = len(frame_indices)
    G = coords.shape[0]
    if n == 0:
        return torch.empty(0, G, 3), torch.empty(0, G)

    # Flatten: frame f occupies rows [f*G : (f+1)*G]; coords tiled, t_* repeated.
    coords_all = coords.unsqueeze(0).expand(n, G, 2).reshape(n * G, 2)
    fi = torch.as_tensor(frame_indices, device=device, dtype=torch.long)
    ci = torch.as_tensor(camera_indices, device=device, dtype=torch.long)
    t_src_all = fi.repeat_interleave(G)
    t_cam_all = ci.repeat_interleave(G)

    total = n * G
    pts_chunks, conf_chunks = [], []
    for start in range(0, total, query_batch_size):
        end = min(start + query_batch_size, total)
        chunk = coords_all[start:end].unsqueeze(0)
        t_src = t_src_all[start:end].unsqueeze(0)
        t_tgt = t_src
        t_cam = t_cam_all[start:end].unsqueeze(0)
        with torch.inference_mode():
            with autocast_context(device):
                outputs = model.decode(
                    encoder_features, frames_bcthw, chunk,
                    t_src, t_tgt, t_cam,
                    transform_metadata=transform_metadata,
                )
        pts_chunks.append(outputs["pos_3d"][0].detach().float())
        conf_chunks.append(outputs["confidence_weight"][0].squeeze(-1).detach().float())
    points = torch.cat(pts_chunks, dim=0).reshape(n, G, 3).cpu()
    conf = torch.cat(conf_chunks, dim=0).reshape(n, G).cpu()
    return points, conf


# ---------------------------------------------------------------------------
# Metric computation (same structure as ScanNet eval)
# ---------------------------------------------------------------------------

def unproject_depth_grid(depth, xs, ys, intrinsics):
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    u = torch.from_numpy(gx.reshape(-1)).float()
    v = torch.from_numpy(gy.reshape(-1)).float()
    d = depth.reshape(-1)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    return torch.stack([x, y, d], dim=-1)


def transform_camera_points(points_cam, src_w2c, dst_w2c):
    ones = torch.ones((points_cam.shape[0], 1), dtype=points_cam.dtype)
    points_h = torch.cat([points_cam, ones], dim=-1)
    src_c2w = torch.linalg.inv(src_w2c)
    world_h = points_h @ src_c2w.T
    dst_h = world_h @ dst_w2c.T
    return dst_h[:, :3]


def _subsample_paired_points(pred_points, gt_points, max_points, seed):
    if max_points <= 0 or pred_points.shape[0] <= max_points:
        return pred_points, gt_points
    generator = torch.Generator().manual_seed(seed)
    keep = torch.randperm(pred_points.shape[0], generator=generator)[:max_points]
    return pred_points[keep], gt_points[keep]


def _prefix_metric_block(prefix, metrics):
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
    """Per-frame depth alignment, MonST3R/DUSt3R/CUT3R Sintel protocol.

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
    model, result, encoder_features, frames_bcthw,
    query_batch_size, stride, device, transform_metadata,
    num_real_frames: int = -1,
    confidence_quantile: float = 0.0,
):
    xs, ys, coords = make_query_grid(result.img_size, stride, device)
    pred_frames, gt_frames, mask_frames = [], [], []
    n = num_real_frames if num_real_frames > 0 else len(result.images)

    frame_ids = list(range(n))
    pred_points_nf, conf_nf = decode_frames_batched(
        model, encoder_features, frames_bcthw, coords,
        frame_indices=frame_ids, camera_indices=frame_ids,
        query_batch_size=query_batch_size, device=device,
        transform_metadata=transform_metadata,
    )
    for frame_idx in range(n):
        pred_points = pred_points_nf[frame_idx]
        conf = conf_nf[frame_idx]
        pred_depth = pred_points[:, 2].reshape(len(ys), len(xs))
        conf_map = conf.reshape(len(ys), len(xs))
        gt_depth = torch.from_numpy(result.depths[frame_idx][ys][:, xs]).float()
        valid = torch.isfinite(gt_depth) & (gt_depth > 0.0) & (gt_depth < SINTEL_MAX_DEPTH) & torch.isfinite(pred_depth)
        valid = _apply_confidence_quantile_filter(valid, conf_map, confidence_quantile)
        pred_frames.append(pred_depth)
        gt_frames.append(gt_depth)
        mask_frames.append(valid)

    pred = torch.stack(pred_frames, dim=0)
    gt = torch.stack(gt_frames, dim=0)
    mask = torch.stack(mask_frames, dim=0)
    pred = torch.clamp(pred, max=SINTEL_PRED_CLIP_MAX)
    # Per-frame median alignment per MonST3R/DUSt3R/CUT3R Sintel protocol.
    # _per_frame_align_depth already applies max_depth clamp pre-alignment;
    # post-alignment values are masked by GT < SINTEL_MAX_DEPTH so we do not
    # pass max_depth to compute_depth_metrics (root utils/metrics.py copy
    # may not accept it).
    pred_s = _per_frame_align_depth(pred, gt, mask, mode="scale", max_depth=SINTEL_MAX_DEPTH)
    pred_ss = _per_frame_align_depth(pred, gt, mask, mode="scale_shift", max_depth=SINTEL_MAX_DEPTH)
    metrics_s = compute_depth_metrics(pred_s, gt, mask=mask, scale_invariant=False, shift_invariant=False)
    metrics_ss = compute_depth_metrics(pred_ss, gt, mask=mask, scale_invariant=False, shift_invariant=False)

    out = {
        "depth_valid_ratio": float(mask.float().mean().item()),
        "depth_num_valid_pixels": float(mask.sum().item()),
    }
    out.update(_prefix_metric_block("depth_S", metrics_s))
    out.update(_prefix_metric_block("depth_SS", metrics_ss))
    return out


def compute_pointcloud_l1_for_clip(
    model, result, encoder_features, frames_bcthw,
    query_batch_size, stride, reference_frame, max_points, seed,
    device, transform_metadata, num_real_frames: int = -1,
    confidence_quantile: float = 0.0,
):
    """Evaluate paper-style point-cloud L1 on a clip.

    Each frame is decoded into a shared reference camera coordinate system by
    fixing ``t_cam = reference_frame`` while keeping ``t_src = t_tgt = frame``.
    GT depths are back-projected and transformed into the same reference frame.
    The final score is mean-shift alignment followed by paired coordinate-wise
    L1; Sintel sky/infinity depths are filtered out.
    """

    xs, ys, coords = make_query_grid(result.img_size, stride, device)
    pred_points_all, gt_points_all = [], []
    ref_w2c = torch.from_numpy(result.extrinsics[reference_frame]).float()
    n = num_real_frames if num_real_frames > 0 else len(result.images)

    frame_ids = list(range(n))
    pred_points_nf, conf_nf = decode_frames_batched(
        model, encoder_features, frames_bcthw, coords,
        frame_indices=frame_ids, camera_indices=[reference_frame] * n,
        query_batch_size=query_batch_size, device=device,
        transform_metadata=transform_metadata,
    )
    for frame_idx in range(n):
        pred_points = pred_points_nf[frame_idx]
        conf = conf_nf[frame_idx]
        gt_depth = torch.from_numpy(result.depths[frame_idx][ys][:, xs]).float()
        gt_cam = unproject_depth_grid(gt_depth, xs, ys, torch.from_numpy(result.intrinsics[frame_idx]).float())
        gt_ref = transform_camera_points(
            gt_cam,
            torch.from_numpy(result.extrinsics[frame_idx]).float(),
            ref_w2c,
        )
        valid = (gt_depth.reshape(-1) > 0.0) & (gt_depth.reshape(-1) < SINTEL_MAX_DEPTH) & torch.isfinite(gt_cam).all(dim=-1) & torch.isfinite(pred_points).all(dim=-1)
        valid = _apply_confidence_quantile_filter(valid, conf, confidence_quantile)
        if valid.any():
            pred_points_all.append(pred_points[valid])
            gt_points_all.append(gt_ref[valid])

    if not pred_points_all:
        return {"pointcloud_num_points": 0.0, "pointcloud_l1": float("nan")}

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
    model, encoder_features, frames_bcthw,
    frame_i, frame_j, grid_h, grid_w, confidence_threshold,
    device, transform_metadata,
    pose_solver="umeyama",
    ransac_iters=256, inlier_thresh=0.1,
    gt_depth_at_grid_i=None, gt_depth_at_grid_j=None,
    gt_depth_mask_threshold: float = 70.0,
):
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
                encoder_features, frames_bcthw, coords,
                t_i, t_i, t_i, transform_metadata=transform_metadata,
            )
            outputs_j = model.decode(
                encoder_features, frames_bcthw, coords,
                t_i, t_i, t_j, transform_metadata=transform_metadata,
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
    if gt_depth_at_grid_i is not None:
        d_i = gt_depth_at_grid_i.to(valid.device)
        valid = valid & torch.isfinite(d_i) & (d_i > 0.0) & (d_i < gt_depth_mask_threshold)
    if gt_depth_at_grid_j is not None:
        d_j = gt_depth_at_grid_j.to(valid.device)
        valid = valid & torch.isfinite(d_j) & (d_j > 0.0) & (d_j < gt_depth_mask_threshold)
    if valid.sum() < 4:
        return torch.eye(4, dtype=torch.float32)

    if pose_solver == "ransac":
        from utils.camera import umeyama_ransac
        R, t, _, _ = umeyama_ransac(
            points_j[valid], points_i[valid],
            weights=weights[valid],
            inlier_thresh=inlier_thresh,
            max_iters=ransac_iters,
            min_inliers=4,
        )
    else:
        R, t, _ = umeyama_alignment(points_j[valid], points_i[valid], weights[valid], with_scale=False)
    R_ij = R.T
    t_ij = -R_ij @ t

    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, :3] = R_ij.cpu()
    pose[:3, 3] = t_ij.cpu()
    return pose


def compute_pose_metrics_for_clip(
    model, result, encoder_features, frames_bcthw,
    reference_frame, grid_h, grid_w, confidence_threshold,
    device, transform_metadata, num_real_frames: int = -1,
    pose_solver: str = "umeyama",
    ransac_iters: int = 256, inlier_thresh: float = 0.1,
    use_gt_depth_mask: bool = False,
    gt_depth_mask_threshold: float = 70.0,
    pair_mode: str = "reference",
):
    gt_w2c = torch.from_numpy(result.extrinsics).float()
    gt_c2w = torch.linalg.inv(gt_w2c)
    ref_w2c = gt_w2c[reference_frame]
    n = num_real_frames if num_real_frames > 0 else len(result.images)

    # Optionally sample Sintel GT depth at the (margin=0.1, grid_h x grid_w) pose grid
    grid_pix_uv = None
    if use_gt_depth_mask:
        u_norm = np.linspace(0.1, 0.9, grid_w, dtype=np.float32)
        v_norm = np.linspace(0.1, 0.9, grid_h, dtype=np.float32)
        gu, gv = np.meshgrid(u_norm, v_norm, indexing="xy")
        us = np.round(gu.reshape(-1) * (result.img_size - 1)).astype(np.int64)
        vs = np.round(gv.reshape(-1) * (result.img_size - 1)).astype(np.int64)
        grid_pix_uv = (np.clip(us, 0, result.img_size - 1), np.clip(vs, 0, result.img_size - 1))

    def gt_depth_at(idx):
        if not use_gt_depth_mask:
            return None
        depth_full = result.depths[idx]
        us, vs = grid_pix_uv
        return torch.from_numpy(depth_full[vs, us]).float()

    pred_w2c_list: list[torch.Tensor] = []
    rot_errors, trans_errors = [], []

    if pair_mode == "reference":
        for frame_idx in range(n):
            if frame_idx == reference_frame:
                pred_w2c_rel = torch.eye(4, dtype=torch.float32)
            else:
                pred_w2c_rel = estimate_relative_pose(
                    model, encoder_features, frames_bcthw,
                    frame_i=reference_frame, frame_j=frame_idx,
                    grid_h=grid_h, grid_w=grid_w,
                    confidence_threshold=confidence_threshold,
                    device=device, transform_metadata=transform_metadata,
                    pose_solver=pose_solver,
                    ransac_iters=ransac_iters, inlier_thresh=inlier_thresh,
                    gt_depth_at_grid_i=gt_depth_at(reference_frame),
                    gt_depth_at_grid_j=gt_depth_at(frame_idx),
                    gt_depth_mask_threshold=gt_depth_mask_threshold,
                )
                gt_w2c_rel = gt_w2c[frame_idx] @ torch.linalg.inv(ref_w2c)
                rel_errors = compute_relative_pose_error(
                    pred_w2c_rel[:3, :3], pred_w2c_rel[:3, 3],
                    gt_w2c_rel[:3, :3], gt_w2c_rel[:3, 3],
                )
                rot_errors.append(rel_errors["rotation_error"])
                trans_errors.append(rel_errors["translation_error"])
            pred_w2c_list.append(pred_w2c_rel)
    elif pair_mode == "adjacent":
        # Chain consecutive (k-1, k) steps; pred_w2c[k] is pose relative to frame 0.
        current = torch.eye(4, dtype=torch.float32)
        pred_w2c_list.append(current.clone())
        for frame_idx in range(1, n):
            step = estimate_relative_pose(
                model, encoder_features, frames_bcthw,
                frame_i=frame_idx - 1, frame_j=frame_idx,
                grid_h=grid_h, grid_w=grid_w,
                confidence_threshold=confidence_threshold,
                device=device, transform_metadata=transform_metadata,
                pose_solver=pose_solver,
                ransac_iters=ransac_iters, inlier_thresh=inlier_thresh,
                gt_depth_at_grid_i=gt_depth_at(frame_idx - 1),
                gt_depth_at_grid_j=gt_depth_at(frame_idx),
                gt_depth_mask_threshold=gt_depth_mask_threshold,
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
    pose_metrics = compute_pose_metrics(pred_c2w, gt_c2w[:n], align=True)

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


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

def build_frame_indices(
    total_frames: int,
    clip_len: int,
    frame_stride: int,
    start_frame: int,
    frame_mode: str = "linspace",
) -> tuple[list[int], int]:
    """Return (frame_indices, num_real_frames).

    frame_mode="linspace" (legacy): always clip_len indices via round-linspace.
    Short sequences produce duplicate indices that inflate pair count with
    identity (zero-error) pairs. Long sequences (e.g. 50-frame Sintel) still
    introduce one stride=2 jump (idx 47 -> 49) that artificially inflates RPE
    on the affected pair.

    frame_mode="unique": stride-1 consecutive window. Long sequences use
    range(start_frame, start_frame + clip_len) - all stride=1, matching the
    paper-strict CUT3R/MonST3R protocol. Short sequences use [0..total-1]
    padded with the last frame; only num_real_frames=total are evaluated.
    """
    if frame_mode == "linspace":
        indices = np.round(np.linspace(0, total_frames - 1, clip_len)).astype(np.int64)
        indices = indices.clip(0, total_frames - 1).tolist()
        return indices, clip_len
    if total_frames >= clip_len:
        # Paper-strict consecutive window. probe_frame_protocol.py shows this
        # improves RPE-R by ~11% on hard scenes vs linspace; pose head trains
        # on stride=1 consecutive frames so eval should match.
        start = max(0, min(int(start_frame), total_frames - clip_len))
        indices = list(range(start, start + clip_len))
        return indices, clip_len
    real = list(range(total_frames))
    pad = [total_frames - 1] * (clip_len - total_frames)
    return real + pad, total_frames


# ---------------------------------------------------------------------------
# Per-scene evaluation
# ---------------------------------------------------------------------------

def evaluate_scene(
    root: Path,
    model: torch.nn.Module,
    scene_name: str,
    args: argparse.Namespace,
    device: torch.device,
    scene_index: int,
) -> dict[str, Any]:
    scene = SintelScene(root, scene_name, sintel_pass=args.sintel_pass)
    total_frames = scene.num_frames
    frame_indices, num_real_frames = build_frame_indices(
        total_frames, args.num_frames, args.frame_stride, args.start_frame,
        frame_mode=args.frame_mode,
    )

    result = scene.load_clip(frame_indices, args.resolution)

    if not (0 <= args.reference_frame < num_real_frames):
        raise ValueError(f"reference_frame={args.reference_frame} out of range for real frames {num_real_frames}")

    encoder_features, frames_bcthw, transform_metadata = encode_video(model, result, device, args.patch_provider)

    depth_metrics = compute_depth_metrics_for_clip(
        model, result, encoder_features, frames_bcthw,
        query_batch_size=args.query_batch_size, stride=args.depth_stride,
        device=device, transform_metadata=transform_metadata,
        num_real_frames=num_real_frames,
        confidence_quantile=args.depth_confidence_quantile,
    )
    pointcloud_metrics = compute_pointcloud_l1_for_clip(
        model, result, encoder_features, frames_bcthw,
        query_batch_size=args.query_batch_size, stride=args.pointcloud_stride,
        reference_frame=args.reference_frame, max_points=args.max_pointcloud_points,
        seed=args.seed + scene_index, device=device,
        transform_metadata=transform_metadata,
        num_real_frames=num_real_frames,
        confidence_quantile=args.pointcloud_confidence_quantile,
    )
    pose_metrics = compute_pose_metrics_for_clip(
        model, result, encoder_features, frames_bcthw,
        reference_frame=args.reference_frame,
        grid_h=args.pose_grid_h, grid_w=args.pose_grid_w,
        confidence_threshold=args.pose_confidence_threshold,
        device=device, transform_metadata=transform_metadata,
        num_real_frames=num_real_frames,
        pose_solver=args.pose_solver,
        ransac_iters=args.pose_ransac_iters,
        inlier_thresh=args.pose_inlier_thresh,
        use_gt_depth_mask=bool(getattr(args, "pose_gt_depth_mask", False)),
        gt_depth_mask_threshold=float(getattr(args, "pose_gt_depth_mask_threshold", 70.0)),
        pair_mode=args.pose_pair_mode,
    )

    return {
        "scene_name": scene_name,
        "num_frames": total_frames,
        "num_real_frames": num_real_frames,
        "clip_len": len(frame_indices),
        "frame_indices": frame_indices,
        **depth_metrics,
        **pointcloud_metrics,
        **pose_metrics,
    }


def summarize_scene_metrics(scene_summaries: list[dict[str, Any]]) -> dict[str, float]:
    if not scene_summaries:
        return {}
    skip_keys = {"scene_name", "frame_indices", "num_frames", "clip_len"}
    metric_keys = [key for key in scene_summaries[0].keys() if key not in skip_keys]
    out: dict[str, float] = {}
    for key in metric_keys:
        values = [float(item[key]) for item in scene_summaries if math.isfinite(float(item[key]))]
        out[key] = float(sum(values) / len(values)) if values else float("nan")
    return out


def main() -> int:
    import time as _time
    _t_main_start = _time.time()
    args = parse_args()
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)

    # Determine scene list
    if args.scene_names:
        scene_names = [s.strip() for s in args.scene_names.split(",") if s.strip()]
    elif args.all_scenes:
        scene_names = sorted([d.name for d in (root / "training" / args.sintel_pass).iterdir() if d.is_dir()])
    else:
        scene_names = SINTEL_14_SEQUENCES

    if args.num_scenes > 0:
        scene_names = scene_names[:args.num_scenes]

    if not scene_names:
        raise RuntimeError("No Sintel scenes selected for evaluation.")

    print(f"[eval] device={device}")
    print(f"[eval] checkpoint={args.checkpoint}")
    print(f"[eval] pass={args.sintel_pass}, scenes={len(scene_names)}")
    print(f"[eval] scenes: {scene_names}")

    model, device = load_model(args, device)

    import time as _time
    _t_loaded = _time.time()
    print(f"[eval] model loaded in {_t_loaded - _t_main_start:.1f}s", flush=True)

    scene_summaries: list[dict[str, Any]] = []
    failed_scenes: list[dict[str, str]] = []
    for scene_index, scene_name in enumerate(scene_names):
        print(f"\n[eval] scene {scene_index + 1}/{len(scene_names)}: {scene_name}", flush=True)
        try:
            summary = evaluate_scene(root, model, scene_name, args, device, scene_index)
        except Exception as exc:
            failure = {"scene_name": scene_name, "error": repr(exc)}
            failed_scenes.append(failure)
            print(json.dumps({"scene_name": scene_name, "status": "failed", "error": repr(exc)}, ensure_ascii=False), flush=True)
            continue
        scene_summaries.append(summary)
        print(json.dumps({
            k: v for k, v in summary.items() if k != "frame_indices"
        }, ensure_ascii=False), flush=True)

    _t_done = _time.time()
    print(f"[eval] compute (all scenes) {_t_done - _t_loaded:.1f}s | total {_t_done - _t_main_start:.1f}s", flush=True)

    strict_metric_settings = (
        args.depth_confidence_quantile <= 0.0
        and args.pointcloud_confidence_quantile <= 0.0
        and args.pose_confidence_threshold <= 0.0
        and args.pose_solver == "umeyama"
    )
    paper_density_settings = (
        args.depth_stride == 1
        and args.pointcloud_stride == 1
        and args.max_pointcloud_points <= 0
    )
    full_paper_scene_set = (not args.all_scenes) and (not args.scene_names) and args.num_scenes <= 0

    final_summary = {
        "root": str(root),
        "checkpoint": args.checkpoint,
        "device": str(device),
        "sintel_pass": args.sintel_pass,
        "num_requested_scenes": len(scene_names),
        "num_scenes": len(scene_summaries),
        "num_failed_scenes": len(failed_scenes),
        "protocol": {
            "paper_aligned": strict_metric_settings and paper_density_settings and full_paper_scene_set,
            "paper_metric_settings": strict_metric_settings,
            "paper_density_settings": paper_density_settings,
            "full_paper_scene_set": full_paper_scene_set,
            "sintel_scene_protocol": "14 final-pass sequences unless --all-scenes or --scene-names is set",
            "depth": "global sequence scale-only and scale-and-shift alignment, AbsRel reported",
            "pointcloud": "shared reference camera, mean-shift translation alignment, paired coordinate Mean L1",
            "pose": "rigid Umeyama relative poses, trajectory Sim(3) alignment, ATE/RPE/Pose AUC@30",
            "depth_confidence_quantile": args.depth_confidence_quantile,
            "pointcloud_confidence_quantile": args.pointcloud_confidence_quantile,
            "pose_confidence_threshold": args.pose_confidence_threshold,
            "pose_solver": args.pose_solver,
            "pose_pair_mode": args.pose_pair_mode,
            "frame_mode": args.frame_mode,
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

    def _fmt(v, fmt=".4f"):
        return f"{v:{fmt}}" if isinstance(v, (int, float)) else str(v)
    print(f"\n{'='*60}")
    print(f"[eval] DONE — {len(scene_summaries)} scenes, {len(failed_scenes)} failed")
    mean = final_summary["mean_metrics"]
    print(f"[eval] Depth: AbsRel(S)={_fmt(mean.get('depth_S_abs_rel', 'N/A'))}  AbsRel(SS)={_fmt(mean.get('depth_SS_abs_rel', 'N/A'))}")
    print(f"[eval] PointCloud L1 (mean-shift, paper)={_fmt(mean.get('pointcloud_l1', 'N/A'))}  L1 (Sim3 diagnostic)={_fmt(mean.get('pointcloud_l1_sim3', 'N/A'))}")
    print(f"[eval] Pose: ATE={_fmt(mean.get('pose_ate', 'N/A'))}  RPE-T={_fmt(mean.get('pose_rpe_trans', 'N/A'))}  RPE-R={_fmt(mean.get('pose_rpe_rot', 'N/A'), '.3f')}")
    print(f"[eval] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
