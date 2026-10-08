#!/usr/bin/env python3
"""Visualize D4RT checkpoint predictions from a plain video file."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import create_d4rt
from utils.visualization import save_point_cloud_ply


CAMERA_VIEW_ELEV = -90
CAMERA_VIEW_AZIM = -90

VARIANT_BY_EMBED_DIM = {
    768: "base",
    1024: "large",
    1280: "huge",
    1408: "giant",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从普通视频文件读取 clip，使用 D4RT checkpoint 预测 2D/3D 点轨迹，"
            "并输出无 GT 的可视化结果。"
        )
    )
    parser.add_argument(
        "--input-path",
        "--video",
        dest="input_path",
        type=str,
        required=True,
        help="输入路径。可以是视频文件，也可以是按时间排序的图片目录。",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint 路径。支持 'model' 或 'model_state_dict'。")
    parser.add_argument(
        "--model-variant",
        type=str,
        default="auto",
        choices=("auto", "base", "large", "huge", "giant"),
        help="模型尺寸。默认 auto 会从 checkpoint 张量形状自动推断。",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录。")
    parser.add_argument(
        "--patch-provider",
        type=str,
        default="sampled_resized",
        help="decoder patch provider。通常保持 sampled_resized。",
    )
    parser.add_argument("--resolution", type=int, default=256, help="模型输入空间分辨率。")
    parser.add_argument("--num-frames", type=int, default=48, help="模型输入时序长度。")
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="contiguous",
        choices=["contiguous", "uniform"],
        help="从输入序列抽取 clip 的方式。默认 contiguous，更接近训练分布。",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=-1,
        help="contiguous 模式下 clip 起始帧。默认 -1 表示自动取中间窗口。",
    )
    parser.add_argument(
        "--query-grid-size",
        type=int,
        default=32,
        help="在参考帧上按 grid_size x grid_size 规则网格采样 query 点。",
    )
    parser.add_argument(
        "--query-margin",
        type=float,
        default=0.06,
        help="query 点离边界的归一化留白比例，范围 [0, 0.45)。",
    )
    parser.add_argument(
        "--reference-frame",
        type=int,
        default=-1,
        help="clip 内参考帧索引。默认 -1 表示取中间帧。",
    )
    parser.add_argument("--num-display-points", type=int, default=180, help="2D 图中实际绘制多少个点。")
    parser.add_argument("--track-tail", type=int, default=12, help="2D 可视化里每个点保留多少帧历史轨迹。")
    parser.add_argument(
        "--dense-point-cloud-stride",
        type=int,
        default=4,
        help="稠密动态点云使用的规则网格步长。越小越密，越慢。",
    )
    parser.add_argument(
        "--dense-vis-threshold",
        type=float,
        default=0.3,
        help="稠密动态点云保留点的 visibility 阈值。",
    )
    parser.add_argument(
        "--dense-confidence-percentile",
        "--dense-pred-confidence-percentile",
        dest="dense_confidence_percentile",
        type=float,
        default=10.0,
        help="稠密动态点云在 visibility 过滤后丢弃最低 N%% confidence 的点。0 表示关闭。",
    )
    parser.add_argument(
        "--dense-query-batch-size",
        type=int,
        default=4096,
        help="稠密动态点云 decode 的 query batch 大小。",
    )
    parser.add_argument(
        "--dense-motion-percentile",
        type=float,
        default=85.0,
        help="额外输出一组“动态部分优先”的点云，可视化 3D 位移排名前百分位的点；0 表示关闭。",
    )
    parser.add_argument("--gif-fps", type=int, default=8, help="输出 GIF 帧率。")
    parser.add_argument("--seed", type=int, default=42, help="用于显示点子采样的随机种子。")
    parser.add_argument("--flip-y-axis", action="store_true", help="对 3D 图和导出的 PLY 翻转 Y 轴。")
    parser.add_argument(
        "--normalize-3d",
        action="store_true",
        help="归一化 3D 点云到单位尺度，改善相对结构可视化（推荐用于无相机内参的视频）。",
    )
    parser.add_argument("--device", type=str, default="cuda", help="期望设备：cuda 或 cpu。")
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _strip_compile_prefix(state: dict[str, Any]) -> dict[str, Any]:
    # Checkpoints saved from a torch.compile()-wrapped model keep an "_orig_mod." prefix
    # on every key. Strip it so the state_dict matches the eager model. No-op otherwise.
    if state and all(k.startswith("_orig_mod.") for k in state):
        return {k[len("_orig_mod."):]: v for k, v in state.items()}
    return state


def extract_model_state(ckpt: dict[str, Any], checkpoint_path: str) -> dict[str, torch.Tensor]:
    if "model" in ckpt:
        return _strip_compile_prefix(ckpt["model"])
    if "model_state_dict" in ckpt:
        return _strip_compile_prefix(ckpt["model_state_dict"])
    raise KeyError(f"Unsupported checkpoint format in {checkpoint_path}")


def infer_model_variant_from_state(state: dict[str, torch.Tensor]) -> str:
    preferred_suffixes = (
        "encoder.patch_embed.proj.weight",
        "encoder.norm.weight",
        "decoder.norm.weight",
    )
    for suffix in preferred_suffixes:
        for key, tensor in state.items():
            if not key.endswith(suffix) or not hasattr(tensor, "shape") or len(tensor.shape) == 0:
                continue
            embed_dim = int(tensor.shape[0])
            if embed_dim in VARIANT_BY_EMBED_DIM:
                return VARIANT_BY_EMBED_DIM[embed_dim]
            raise ValueError(
                f"Cannot infer model variant: {key} has embed_dim={embed_dim}, "
                f"expected one of {sorted(VARIANT_BY_EMBED_DIM)}"
            )
    raise ValueError("Cannot infer model variant: checkpoint has no recognizable encoder/decoder shape keys")


def load_checkpoint_state(checkpoint_path: str) -> dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return extract_model_state(ckpt, checkpoint_path)


def resolve_model_variant(requested_variant: str, state: dict[str, torch.Tensor]) -> str:
    if requested_variant != "auto":
        return requested_variant
    inferred_variant = infer_model_variant_from_state(state)
    print(f"[vis_video] Auto-detected model_variant={inferred_variant} from checkpoint tensor shapes", flush=True)
    return inferred_variant


def load_model_weights(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    state = load_checkpoint_state(args.checkpoint)
    model_variant = resolve_model_variant(args.model_variant, state)
    args.model_variant = model_variant
    model = create_d4rt(
        variant=model_variant,
        decoder_depth=6,
        img_size=args.resolution,
        temporal_size=args.num_frames,
        patch_size=(2, 16, 16),
        query_patch_size=9,
        videomae_model="/data1/zbf/pretrained/videomae-base",
        patch_provider=args.patch_provider,
        encoder_pretrained=False,
    ).to(device)
    load_model_weights(model, state)
    del state
    model.eval()
    return model


def _read_video_cv2(video_path: str) -> tuple[list[np.ndarray], float | None]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture failed to open: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = None

    frames: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(frame_rgb)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frames, fps


def _read_video_imageio(video_path: str) -> tuple[list[np.ndarray], float | None]:
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    fps = meta.get("fps")
    if fps is not None:
        fps = float(fps)
        if not np.isfinite(fps) or fps <= 0:
            fps = None

    frames: list[np.ndarray] = []
    for frame in reader:
        frame_np = np.asarray(frame)
        if frame_np.ndim != 3 or frame_np.shape[2] < 3:
            raise RuntimeError(f"Unsupported frame shape from imageio: {frame_np.shape}")
        frames.append(frame_np[..., :3].astype(np.float32) / 255.0)
    reader.close()

    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frames, fps


def read_image_directory(image_dir: str) -> tuple[list[np.ndarray], list[str]]:
    path = Path(image_dir)
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {image_dir}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    image_paths = sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if len(image_paths) == 0:
        raise RuntimeError(f"No supported images found in directory: {image_dir}")

    frames: list[np.ndarray] = []
    names: list[str] = []
    for image_path in image_paths:
        frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(frame_rgb)
        names.append(image_path.name)
    return frames, names


def read_input_frames(input_path: str) -> tuple[list[np.ndarray], float | None, str, list[str] | None]:
    path = Path(input_path)
    if path.is_dir():
        frames, names = read_image_directory(input_path)
        return frames, None, "image_dir", names

    if not path.is_file():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    try:
        frames, fps = _read_video_cv2(input_path)
    except Exception:
        frames, fps = _read_video_imageio(input_path)
    return frames, fps, "video_file", None


def sample_clip_frames(
    frames: list[np.ndarray],
    target_frames: int,
    sampling_mode: str,
    start_index: int,
) -> tuple[list[np.ndarray], list[int], int]:
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")

    total = len(frames)
    if total >= target_frames and sampling_mode == "uniform":
        indices = np.linspace(0, total - 1, num=target_frames, dtype=int).tolist()
    elif total >= target_frames:
        max_start = total - target_frames
        if start_index < 0:
            start = max_start // 2
        else:
            start = min(max(start_index, 0), max_start)
        indices = list(range(start, start + target_frames))
    else:
        indices = list(range(total)) + [total - 1] * (target_frames - total)

    sampled = [frames[idx] for idx in indices]
    padded_frames = max(0, target_frames - total)
    return sampled, indices, padded_frames


def resize_frames_square(frames: list[np.ndarray], size: int) -> list[np.ndarray]:
    resized: list[np.ndarray] = []
    for frame in frames:
        h, w = frame.shape[:2]
        interp = cv2.INTER_AREA if max(h, w) >= size else cv2.INTER_LINEAR
        resized.append(cv2.resize(frame, (size, size), interpolation=interp).astype(np.float32))
    return resized


def build_query_grid(
    width: int,
    height: int,
    grid_size: int,
    margin: float,
    reference_frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if grid_size <= 0:
        raise ValueError(f"query-grid-size must be positive, got {grid_size}")
    if not (0.0 <= margin < 0.45):
        raise ValueError(f"query-margin must be in [0, 0.45), got {margin}")

    if grid_size == 1:
        xs = np.array([0.5], dtype=np.float32)
        ys = np.array([0.5], dtype=np.float32)
    else:
        xs = np.linspace(margin, 1.0 - margin, num=grid_size, dtype=np.float32)
        ys = np.linspace(margin, 1.0 - margin, num=grid_size, dtype=np.float32)

    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    query_points = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2).astype(np.float32)
    query_frames = np.full((len(query_points),), reference_frame, dtype=np.int64)
    query_points_px = query_points * np.array(
        [max(width - 1, 1), max(height - 1, 1)],
        dtype=np.float32,
    )[None]
    return query_points, query_frames, query_points_px


def point_colors_from_frame(
    image: np.ndarray,
    points_px: np.ndarray,
) -> np.ndarray:
    h, w = image.shape[:2]
    xs = np.clip(np.round(points_px[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.round(points_px[:, 1]).astype(np.int32), 0, h - 1)
    return image[ys, xs].astype(np.float32)


def figure_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return rgba[..., :3].copy()


def maybe_flip_y(points: np.ndarray, flip_y: bool) -> np.ndarray:
    if not flip_y:
        return points
    out = points.copy()
    out[:, 1] *= -1.0
    return out


def to_video_tensor(frames: list[np.ndarray], device: torch.device) -> torch.Tensor:
    video_np = np.stack(frames, axis=0).astype(np.float32)
    return torch.from_numpy(video_np).unsqueeze(0).to(device)


def make_aspect_ratio(orig_width: int, orig_height: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[float(orig_width) / max(float(orig_height), 1.0)]],
        dtype=torch.float32,
        device=device,
    )


def compute_predictions(
    model: torch.nn.Module,
    frames_resized: list[np.ndarray],
    orig_width: int,
    orig_height: int,
    display_size: int,
    query_points_norm: np.ndarray,
    query_frames: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    video = to_video_tensor(frames_resized, device)
    aspect_ratio = make_aspect_ratio(orig_width, orig_height, device)
    qp = torch.from_numpy(query_points_norm).unsqueeze(0).to(device)
    qf = torch.from_numpy(query_frames).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model.predict_point_tracks(
                video=video,
                query_points=qp,
                query_frames=qf,
                aspect_ratio=aspect_ratio,
            )

    pred_2d_norm = outputs["tracks_2d"][0].detach().float().cpu().numpy()
    pred_3d = outputs["tracks_3d"][0].detach().float().cpu().numpy()
    pred_vis = outputs["visibility"][0].detach().float().cpu().numpy()
    pred_conf_tensor = outputs.get("confidence_weight")
    if pred_conf_tensor is None:
        pred_conf = np.ones_like(pred_vis, dtype=np.float32)
    else:
        pred_conf = pred_conf_tensor[0].detach().float().cpu().numpy()
        if pred_conf.ndim == 3 and pred_conf.shape[-1] == 1:
            pred_conf = pred_conf[..., 0]

    pred_2d_px = pred_2d_norm * np.array(
        [max(display_size - 1, 1), max(display_size - 1, 1)],
        dtype=np.float32,
    )[None, None]

    return {
        "pred_2d_norm": pred_2d_norm,
        "pred_2d_px": pred_2d_px,
        "pred_3d_cam": pred_3d,
        "pred_vis": pred_vis,
        "pred_conf": pred_conf,
    }


def compute_dense_reference_sequence(
    model: torch.nn.Module,
    frames_resized: list[np.ndarray],
    reference_frame: int,
    stride: int,
    vis_threshold: float,
    confidence_percentile: float,
    batch_size: int,
    orig_width: int,
    orig_height: int,
    device: torch.device,
    flip_y: bool,
) -> list[dict[str, Any]]:
    if stride <= 0:
        raise ValueError(f"dense-point-cloud-stride must be positive, got {stride}")
    if batch_size <= 0:
        raise ValueError(f"dense-query-batch-size must be positive, got {batch_size}")

    video = to_video_tensor(frames_resized, device)
    aspect_ratio = make_aspect_ratio(orig_width, orig_height, device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            encoder_features = model.encode(video, aspect_ratio)

    frames_bcthw = video.permute(0, 1, 4, 2, 3)
    T = len(frames_resized)
    H, W = frames_resized[0].shape[:2]

    x_pixels = np.arange(0, W, stride, dtype=np.int32)
    y_pixels = np.arange(0, H, stride, dtype=np.int32)
    if len(x_pixels) == 0 or len(y_pixels) == 0:
        raise RuntimeError(f"stride={stride} produced empty dense grid for resolution {W}x{H}")

    grid_x, grid_y = np.meshgrid(x_pixels, y_pixels, indexing="xy")
    coords_px = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
    coords_norm = coords_px.astype(np.float32) / np.array(
        [max(W - 1, 1), max(H - 1, 1)],
        dtype=np.float32,
    )[None]
    coords_torch = torch.from_numpy(coords_norm).to(device=device, dtype=torch.float32)
    reference_colors = frames_resized[reference_frame][coords_px[:, 1], coords_px[:, 0]].astype(np.float32)

    dense_frames: list[dict[str, Any]] = []
    for frame_id in range(T):
        points_list: list[np.ndarray] = []
        vis_list: list[np.ndarray] = []
        conf_list: list[np.ndarray] = []
        total_points = int(coords_torch.shape[0])

        for start in range(0, total_points, batch_size):
            end = min(start + batch_size, total_points)
            coords_chunk = coords_torch[start:end].unsqueeze(0)
            chunk_size = end - start
            t_src = torch.full((1, chunk_size), reference_frame, device=device, dtype=torch.long)
            t_tgt = torch.full((1, chunk_size), frame_id, device=device, dtype=torch.long)
            t_cam = torch.full((1, chunk_size), reference_frame, device=device, dtype=torch.long)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    outputs = model.decode(
                        encoder_features,
                        frames_bcthw,
                        coords_chunk,
                        t_src,
                        t_tgt,
                        t_cam,
                    )

            points_list.append(outputs["pos_3d"][0].detach().float().cpu().numpy())
            vis_list.append(torch.sigmoid(outputs["visibility"][0].squeeze(-1)).detach().float().cpu().numpy())
            conf_tensor = outputs.get("confidence_weight")
            if conf_tensor is None:
                conf_list.append(np.ones((chunk_size,), dtype=np.float32))
            else:
                conf_list.append(conf_tensor[0].squeeze(-1).detach().float().cpu().numpy())

        points_ref = np.concatenate(points_list, axis=0)
        visibility = np.concatenate(vis_list, axis=0)
        confidence = np.concatenate(conf_list, axis=0)
        frame_colors = reference_colors

        keep = visibility > vis_threshold
        keep &= np.isfinite(points_ref).all(axis=-1)
        visible_before_confidence = int(keep.sum())
        confidence_threshold = math.nan
        if confidence_percentile > 0.0 and keep.sum() > 0:
            confidence_threshold = float(np.percentile(confidence[keep], confidence_percentile))
            keep &= confidence >= confidence_threshold

        points_ref = maybe_flip_y(points_ref[keep], flip_y)
        frame_colors = frame_colors[keep]

        dense_frames.append(
            {
                "frame_id": frame_id,
                "points_ref": points_ref,
                "colors": frame_colors,
                "point_ids": np.flatnonzero(keep).astype(np.int32),
                "visible_points": int(keep.sum()),
                "visible_before_confidence": visible_before_confidence,
                "total_points": total_points,
                "confidence_threshold": confidence_threshold,
                "mean_confidence": float(np.mean(confidence[keep])) if keep.any() else math.nan,
            }
        )
    return dense_frames


def make_frame_ids(total_frames: int, num_frames: int) -> list[int]:
    frame_ids = np.linspace(0, total_frames - 1, num=min(num_frames, total_frames), dtype=int)
    return list(dict.fromkeys(frame_ids.tolist()))


def select_display_ids(
    pred: dict[str, np.ndarray],
    query_points_px: np.ndarray,
    num_display_points: int,
    seed: int,
) -> np.ndarray:
    num_points = int(len(query_points_px))
    if num_display_points <= 0 or num_display_points >= num_points:
        return np.arange(num_points, dtype=np.int64)

    src = query_points_px[:, None, :]
    displacement = np.linalg.norm(pred["pred_2d_px"] - src, axis=-1)
    displacement = np.where(pred["pred_vis"] > 0.5, displacement, np.nan)

    motion_scores = np.full((num_points,), -np.inf, dtype=np.float32)
    for point_id in range(num_points):
        finite_disp = displacement[point_id][np.isfinite(displacement[point_id])]
        if finite_disp.size > 0:
            motion_scores[point_id] = float(np.percentile(finite_disp, 90))

    ranked = np.flatnonzero(np.isfinite(motion_scores))
    rng = np.random.default_rng(seed)
    if len(ranked) > 0:
        jitter = rng.uniform(0.0, 1e-4, size=len(ranked)).astype(np.float32)
        ranked = ranked[np.argsort(motion_scores[ranked] + jitter)]
        selected = ranked[-num_display_points:]
        if len(selected) == num_display_points:
            return np.sort(selected.astype(np.int64))

    return np.sort(rng.choice(np.arange(num_points), size=num_display_points, replace=False))


def make_display_colors(num_points: int) -> np.ndarray:
    if num_points <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    cmap = plt.get_cmap("hsv")
    positions = np.linspace(0.0, 1.0, num_points, endpoint=False, dtype=np.float32)
    return np.asarray([cmap(float(pos))[:3] for pos in positions], dtype=np.float32)


def _draw_track_tail(
    ax,
    track_px: np.ndarray,
    visibility: np.ndarray,
    color: np.ndarray,
    track_tail: int,
) -> None:
    num_steps = track_px.shape[0]
    start_t = max(0, num_steps - track_tail)
    hist = track_px[start_t:]
    hist_vis = visibility[start_t:] > 0.5
    if not np.any(hist_vis):
        return

    for idx in range(len(hist) - 1):
        if hist_vis[idx] and hist_vis[idx + 1]:
            ax.plot(
                [hist[idx, 0], hist[idx + 1, 0]],
                [hist[idx, 1], hist[idx + 1, 1]],
                color=color,
                linewidth=1.4,
                alpha=0.85,
            )

    if hist_vis[-1]:
        ax.scatter(
            hist[-1, 0],
            hist[-1, 1],
            s=30,
            c=[color],
            marker="o",
            edgecolors="#ffffff",
            linewidths=0.6,
            alpha=0.98,
            zorder=10,
        )


def plot_reference_queries(
    frame: np.ndarray,
    query_points_px: np.ndarray,
    colors: np.ndarray,
    output_path: Path,
    source_frame: int,
    original_frame_index: int,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 7.0), constrained_layout=True)
    ax.imshow(frame)
    ax.scatter(
        query_points_px[:, 0],
        query_points_px[:, 1],
        s=10,
        c=colors,
        alpha=0.85,
    )
    ax.set_title(
        f"Reference queries\nclip frame={source_frame}  video frame={original_frame_index}  points={len(query_points_px)}",
        fontsize=12,
    )
    ax.axis("off")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_2d_tracks_static(
    frames: list[np.ndarray],
    pred: dict[str, np.ndarray],
    display_ids: np.ndarray,
    colors: np.ndarray,
    output_path: Path,
    sampled_indices: list[int],
    track_tail: int,
) -> None:
    T = len(frames)
    frame_ids = make_frame_ids(T, num_frames=5)

    fig, axes = plt.subplots(
        1,
        len(frame_ids),
        figsize=(4.6 * len(frame_ids), 4.8),
        constrained_layout=True,
    )
    if len(frame_ids) == 1:
        axes = [axes]

    for ax, frame_id in zip(axes, frame_ids, strict=True):
        ax.imshow(frames[frame_id])
        for display_rank, point_id in enumerate(display_ids):
            track_px = pred["pred_2d_px"][point_id, : frame_id + 1]
            track_vis = pred["pred_vis"][point_id, : frame_id + 1]
            _draw_track_tail(
                ax=ax,
                track_px=track_px,
                visibility=track_vis,
                color=colors[display_rank],
                track_tail=track_tail,
            )
        visible_count = int((pred["pred_vis"][display_ids, frame_id] > 0.5).sum())
        ax.set_title(
            f"clip frame {frame_id}\nvideo frame {sampled_indices[frame_id]}  vis={visible_count}",
            fontsize=10,
        )
        ax.axis("off")

    fig.suptitle(
        f"Predicted 2D tracks\nshown points={len(display_ids)}  selection=motion-topk  tail={track_tail}",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_2d_tracks_gif(
    frames: list[np.ndarray],
    pred: dict[str, np.ndarray],
    display_ids: np.ndarray,
    colors: np.ndarray,
    output_path: Path,
    sampled_indices: list[int],
    track_tail: int,
    fps: int,
) -> None:
    frames_rgb: list[np.ndarray] = []
    for frame_id, frame in enumerate(frames):
        fig, ax = plt.subplots(1, 1, figsize=(6.8, 6.0), constrained_layout=True)
        ax.imshow(frame)
        for display_rank, point_id in enumerate(display_ids):
            track_px = pred["pred_2d_px"][point_id, : frame_id + 1]
            track_vis = pred["pred_vis"][point_id, : frame_id + 1]
            _draw_track_tail(
                ax=ax,
                track_px=track_px,
                visibility=track_vis,
                color=colors[display_rank],
                track_tail=track_tail,
            )
        visible_count = int((pred["pred_vis"][display_ids, frame_id] > 0.5).sum())
        fig.suptitle(
            f"Predicted tracks  clip frame={frame_id}  video frame={sampled_indices[frame_id]}  "
            f"vis={visible_count}/{len(display_ids)}",
            fontsize=12,
        )
        ax.axis("off")
        frames_rgb.append(figure_to_rgb(fig))
        plt.close(fig)

    imageio.mimsave(output_path, frames_rgb, fps=fps, loop=0)


def normalize_3d_points(points: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """归一化 3D 点云到单位尺度，保留相对结构。

    Args:
        points: (N, 3) 3D 点云

    Returns:
        normalized_points: (N, 3) 归一化后的点云
        stats: 归一化统计信息（center, scale）
    """
    valid_mask = np.isfinite(points).all(axis=-1)
    if not valid_mask.any():
        return points, {"center": [0.0, 0.0, 0.0], "scale": 1.0}

    valid_points = points[valid_mask]
    center = valid_points.mean(axis=0)
    centered = valid_points - center
    scale = np.percentile(np.linalg.norm(centered, axis=-1), 95)
    scale = max(scale, 1e-6)

    normalized = points.copy()
    normalized[valid_mask] = centered / scale

    return normalized, {
        "center": center.tolist(),
        "scale": float(scale),
    }


def _set_equal_3d_axes(ax, points: np.ndarray) -> None:
    pts = points[np.isfinite(points).all(axis=-1)]
    if len(pts) == 0:
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def dense_axis_limits(dense_frames: list[dict[str, Any]]) -> tuple[np.ndarray, float]:
    valid_clouds = [frame["points_ref"] for frame in dense_frames if len(frame["points_ref"]) > 0]
    if not valid_clouds:
        return np.zeros(3, dtype=np.float32), 1.0
    pts = np.concatenate(valid_clouds, axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = ((mins + maxs) / 2.0).astype(np.float32)
    radius = max(float((maxs - mins).max()) / 2.0, 1e-3)
    return center, radius


def set_axes_limits(ax, center: np.ndarray, radius: float) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def set_camera_view(ax) -> None:
    ax.view_init(elev=CAMERA_VIEW_ELEV, azim=CAMERA_VIEW_AZIM)


def _extract_reference_3d_points(
    pred: dict[str, np.ndarray],
    colors: np.ndarray,
    reference_frame: int,
    flip_y: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    visible = pred["pred_vis"][:, reference_frame] > 0.5
    finite = np.isfinite(pred["pred_3d_cam"][:, reference_frame]).all(axis=-1)
    mask = visible & finite
    points = pred["pred_3d_cam"][mask, reference_frame]
    colors_vis = colors[mask]

    if len(points) == 0:
        raise RuntimeError(f"no visible finite 3D points at reference frame {reference_frame}")

    points_vis = maybe_flip_y(points, flip_y)
    depth = points[:, 2]
    depth = depth[np.isfinite(depth)]
    return points_vis, colors_vis, depth


def plot_reference_3d(
    pred: dict[str, np.ndarray],
    colors: np.ndarray,
    reference_frame: int,
    output_path: Path,
    ply_path: Path,
    flip_y: bool,
    normalize_3d: bool = False,
) -> dict[str, float]:
    points_vis, colors_vis, depth = _extract_reference_3d_points(
        pred=pred,
        colors=colors,
        reference_frame=reference_frame,
        flip_y=flip_y,
    )

    # 归一化处理
    norm_stats = {}
    if normalize_3d:
        points_vis, norm_stats = normalize_3d_points(points_vis)

    fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(
        points_vis[:, 0],
        points_vis[:, 1],
        points_vis[:, 2],
        c=colors_vis,
        s=4.0,
        alpha=0.9,
    )
    set_camera_view(ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_equal_3d_axes(ax, points_vis)
    title = f"Predicted 3D points @ reference frame {reference_frame}\npoints={len(points_vis)}"
    if normalize_3d:
        title += " (normalized)"
    fig.suptitle(title, fontsize=13)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    save_point_cloud_ply(str(ply_path), points_vis, colors=colors_vis)

    if depth.size == 0:
        return {
            "num_visible_ref_points": float(len(points_vis)),
            "ref_depth_mean": math.nan,
            "ref_depth_median": math.nan,
            "ref_depth_p10": math.nan,
            "ref_depth_p90": math.nan,
        }

    return {
        "num_visible_ref_points": float(len(points_vis)),
        "ref_depth_mean": float(np.mean(depth)),
        "ref_depth_median": float(np.median(depth)),
        "ref_depth_p10": float(np.percentile(depth, 10)),
        "ref_depth_p90": float(np.percentile(depth, 90)),
    }


def write_reference_3d_turntable_gif(
    pred: dict[str, np.ndarray],
    colors: np.ndarray,
    reference_frame: int,
    output_path: Path,
    flip_y: bool,
    fps: int,
    num_frames: int = 36,
    normalize_3d: bool = False,
) -> None:
    points_vis, colors_vis, _ = _extract_reference_3d_points(
        pred=pred,
        colors=colors,
        reference_frame=reference_frame,
        flip_y=flip_y,
    )

    # 归一化处理
    if normalize_3d:
        points_vis, _ = normalize_3d_points(points_vis)

    frames_rgb: list[np.ndarray] = []
    for frame_idx in range(num_frames):
        fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.scatter(
            points_vis[:, 0],
            points_vis[:, 1],
            points_vis[:, 2],
            c=colors_vis,
            s=4.0,
            alpha=0.9,
        )
        set_camera_view(ax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        _set_equal_3d_axes(ax, points_vis)
        title = f"Predicted 3D points @ reference frame {reference_frame}\ncamera view frame {frame_idx + 1}/{num_frames}"
        if normalize_3d:
            title = title.replace("\ncamera", " (normalized)\ncamera")
        fig.suptitle(title, fontsize=13)
        frames_rgb.append(figure_to_rgb(fig))
        plt.close(fig)

    imageio.mimsave(output_path, frames_rgb, fps=fps, loop=0)


def plot_dense_reference_static(
    dense_frames: list[dict[str, Any]],
    output_path: Path,
    title_prefix: str = "Predicted dense dynamic point cloud in reference frame",
) -> dict[str, float]:
    frame_ids = make_frame_ids(len(dense_frames), num_frames=6)
    center, radius = dense_axis_limits(dense_frames)

    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    for plot_idx, frame_id in enumerate(frame_ids, start=1):
        ax = fig.add_subplot(2, 3, plot_idx, projection="3d")
        frame = dense_frames[frame_id]
        ax.scatter(
            frame["points_ref"][:, 0],
            frame["points_ref"][:, 1],
            frame["points_ref"][:, 2],
            c=np.clip(frame["colors"], 0.0, 1.0),
            s=0.45,
            alpha=0.9,
        )
        set_camera_view(ax)
        ax.set_title(
            f"frame {frame_id}\nkept={frame['visible_points']} "
            f"pre_conf={frame.get('visible_before_confidence', frame['visible_points'])} "
            f"total={frame['total_points']}",
            fontsize=10,
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        set_axes_limits(ax, center, radius)

    fig.suptitle(title_prefix, fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    visible_counts = np.array([frame["visible_points"] for frame in dense_frames], dtype=np.float32)
    pre_conf_counts = np.array(
        [frame.get("visible_before_confidence", frame["visible_points"]) for frame in dense_frames],
        dtype=np.float32,
    )
    mean_conf = np.array([frame.get("mean_confidence", math.nan) for frame in dense_frames], dtype=np.float32)
    return {
        "dense_pred_mean_visible_points": float(np.mean(visible_counts)),
        "dense_pred_max_visible_points": float(np.max(visible_counts)),
        "dense_pred_mean_visible_before_confidence": float(np.mean(pre_conf_counts)),
        "dense_pred_mean_confidence": float(np.nanmean(mean_conf)) if np.isfinite(mean_conf).any() else math.nan,
    }


def write_dense_reference_gif(
    dense_frames: list[dict[str, Any]],
    output_path: Path,
    fps: int,
    title_prefix: str = "Predicted dense dynamic point cloud",
) -> None:
    center, radius = dense_axis_limits(dense_frames)
    frames_rgb: list[np.ndarray] = []

    for frame in dense_frames:
        fig = plt.figure(figsize=(6.4, 5.8), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.scatter(
            frame["points_ref"][:, 0],
            frame["points_ref"][:, 1],
            frame["points_ref"][:, 2],
            c=np.clip(frame["colors"], 0.0, 1.0),
            s=0.45,
            alpha=0.9,
        )
        set_camera_view(ax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        set_axes_limits(ax, center, radius)
        fig.suptitle(
            f"{title_prefix}\nframe={frame['frame_id']}  "
            f"kept={frame['visible_points']} "
            f"pre_conf={frame.get('visible_before_confidence', frame['visible_points'])} "
            f"total={frame['total_points']}",
            fontsize=12,
        )
        frames_rgb.append(figure_to_rgb(fig))
        plt.close(fig)

    imageio.mimsave(output_path, frames_rgb, fps=fps, loop=0)


def select_dense_dynamic_ids(
    dense_frames: list[dict[str, Any]],
    reference_frame: int,
    motion_percentile: float,
) -> tuple[set[int], dict[str, float]]:
    if motion_percentile <= 0.0:
        return set(), {
            "dense_motion_percentile": float(motion_percentile),
            "dense_dynamic_selected_points": 0.0,
            "dense_dynamic_motion_threshold": math.nan,
            "dense_dynamic_max_motion": math.nan,
        }

    reference = dense_frames[reference_frame]
    if len(reference["point_ids"]) == 0:
        return set(), {
            "dense_motion_percentile": float(motion_percentile),
            "dense_dynamic_selected_points": 0.0,
            "dense_dynamic_motion_threshold": math.nan,
            "dense_dynamic_max_motion": math.nan,
        }

    reference_points = {
        int(point_id): point
        for point_id, point in zip(reference["point_ids"], reference["points_ref"], strict=False)
    }

    motion_scores: dict[int, float] = {}
    for frame in dense_frames:
        for point_id, point in zip(frame["point_ids"], frame["points_ref"], strict=False):
            point_id_int = int(point_id)
            ref_point = reference_points.get(point_id_int)
            if ref_point is None:
                continue
            motion = float(np.linalg.norm(point - ref_point))
            prev = motion_scores.get(point_id_int)
            if prev is None or motion > prev:
                motion_scores[point_id_int] = motion

    if not motion_scores:
        return set(), {
            "dense_motion_percentile": float(motion_percentile),
            "dense_dynamic_selected_points": 0.0,
            "dense_dynamic_motion_threshold": math.nan,
            "dense_dynamic_max_motion": math.nan,
        }

    score_values = np.array(list(motion_scores.values()), dtype=np.float32)
    threshold = float(np.percentile(score_values, motion_percentile))
    selected_ids = {
        point_id
        for point_id, motion in motion_scores.items()
        if motion >= threshold and motion > 1e-4
    }
    if not selected_ids:
        selected_ids = {max(motion_scores, key=motion_scores.get)}

    return selected_ids, {
        "dense_motion_percentile": float(motion_percentile),
        "dense_dynamic_selected_points": float(len(selected_ids)),
        "dense_dynamic_motion_threshold": float(threshold),
        "dense_dynamic_max_motion": float(np.max(score_values)),
    }


def filter_dense_frames_by_ids(
    dense_frames: list[dict[str, Any]],
    selected_ids: set[int],
) -> list[dict[str, Any]]:
    filtered_frames: list[dict[str, Any]] = []
    selected_total = int(len(selected_ids))

    for frame in dense_frames:
        if selected_total == 0:
            mask = np.zeros((len(frame["point_ids"]),), dtype=bool)
        else:
            mask = np.array([int(point_id) in selected_ids for point_id in frame["point_ids"]], dtype=bool)

        filtered_frames.append(
            {
                "frame_id": frame["frame_id"],
                "points_ref": frame["points_ref"][mask],
                "colors": frame["colors"][mask],
                "point_ids": frame["point_ids"][mask],
                "visible_points": int(mask.sum()),
                "visible_before_confidence": int(mask.sum()),
                "total_points": selected_total,
                "confidence_threshold": frame.get("confidence_threshold", math.nan),
                "mean_confidence": frame.get("mean_confidence", math.nan),
            }
        )

    return filtered_frames


def summarize_predictions(
    pred: dict[str, np.ndarray],
    query_points_px: np.ndarray,
    reference_frame: int,
) -> dict[str, float]:
    vis_mask = pred["pred_vis"] > 0.5
    visible_ratio = float(vis_mask.mean())
    visible_ratio_ref = float(vis_mask[:, reference_frame].mean())
    pred_conf = pred.get("pred_conf")
    if pred_conf is not None and np.isfinite(pred_conf).any():
        confidence_stats = {
            "pred_mean_confidence": float(np.nanmean(pred_conf)),
            "pred_mean_visible_confidence": float(np.nanmean(pred_conf[vis_mask])) if vis_mask.any() else math.nan,
        }
    else:
        confidence_stats = {
            "pred_mean_confidence": math.nan,
            "pred_mean_visible_confidence": math.nan,
        }

    src = query_points_px[:, None, :]
    displacement = np.linalg.norm(pred["pred_2d_px"] - src, axis=-1)
    visible_disp = displacement[vis_mask]
    if visible_disp.size == 0:
        motion_stats = {
            "mean_visible_track_disp_px": math.nan,
            "median_visible_track_disp_px": math.nan,
            "p90_visible_track_disp_px": math.nan,
        }
    else:
        motion_stats = {
            "mean_visible_track_disp_px": float(np.mean(visible_disp)),
            "median_visible_track_disp_px": float(np.median(visible_disp)),
            "p90_visible_track_disp_px": float(np.percentile(visible_disp, 90)),
        }

    return {
        "pred_visible_ratio": visible_ratio,
        "pred_visible_ratio_ref": visible_ratio_ref,
        **confidence_stats,
        **motion_stats,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (0.0 <= args.dense_motion_percentile <= 100.0):
        raise ValueError(
            f"dense-motion-percentile must be in [0, 100], got {args.dense_motion_percentile}"
        )
    if not (0.0 <= args.dense_confidence_percentile <= 100.0):
        raise ValueError(
            f"dense-confidence-percentile must be in [0, 100], got {args.dense_confidence_percentile}"
        )

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = select_device(args.device)
    frames_all, fps, input_kind, source_names = read_input_frames(str(input_path))
    sampled_frames_orig, sampled_indices, padded_frames = sample_clip_frames(
        frames_all,
        args.num_frames,
        args.sampling_mode,
        args.start_index,
    )
    orig_h, orig_w = sampled_frames_orig[0].shape[:2]
    sampled_frames_resized = resize_frames_square(sampled_frames_orig, args.resolution)

    reference_frame = args.reference_frame
    if reference_frame < 0:
        reference_frame = args.num_frames // 2
    if not (0 <= reference_frame < args.num_frames):
        raise ValueError(
            f"reference-frame must be in [0, {args.num_frames - 1}], got {reference_frame}"
        )

    query_points_norm, query_frames, query_points_px = build_query_grid(
        width=args.resolution,
        height=args.resolution,
        grid_size=args.query_grid_size,
        margin=args.query_margin,
        reference_frame=reference_frame,
    )
    colors_all = point_colors_from_frame(sampled_frames_resized[reference_frame], query_points_px)

    model = load_model(args, device)
    pred = compute_predictions(
        model=model,
        frames_resized=sampled_frames_resized,
        orig_width=orig_w,
        orig_height=orig_h,
        display_size=args.resolution,
        query_points_norm=query_points_norm,
        query_frames=query_frames,
        device=device,
    )

    display_ids = select_display_ids(
        pred=pred,
        query_points_px=query_points_px,
        num_display_points=args.num_display_points,
        seed=args.seed,
    )
    display_colors = make_display_colors(len(display_ids))

    query_frame_path = out_dir / "reference_queries.png"
    overlay_path = out_dir / "tracks_2d_overlay.png"
    gif_path = out_dir / "tracks_2d_overlay.gif"
    pred_3d_path = out_dir / "pred_3d_reference.png"
    pred_3d_gif_path = out_dir / "pred_3d_reference.gif"
    pred_ply_path = out_dir / "pred_3d_reference.ply"
    dense_static_path = out_dir / "pred_dense_reference_static.png"
    dense_gif_path = out_dir / "pred_dense_reference.gif"
    dense_ref_ply_path = out_dir / f"pred_dense_reference_frame_{reference_frame:03d}.ply"
    dense_motion_static_path = out_dir / "pred_dense_motion_reference_static.png"
    dense_motion_gif_path = out_dir / "pred_dense_motion_reference.gif"
    dense_motion_ply_path = out_dir / f"pred_dense_motion_reference_frame_{reference_frame:03d}.ply"

    plot_reference_queries(
        frame=sampled_frames_resized[reference_frame],
        query_points_px=query_points_px,
        colors=colors_all,
        output_path=query_frame_path,
        source_frame=reference_frame,
        original_frame_index=sampled_indices[reference_frame],
    )
    plot_2d_tracks_static(
        frames=sampled_frames_resized,
        pred=pred,
        display_ids=display_ids,
        colors=display_colors,
        output_path=overlay_path,
        sampled_indices=sampled_indices,
        track_tail=args.track_tail,
    )
    write_2d_tracks_gif(
        frames=sampled_frames_resized,
        pred=pred,
        display_ids=display_ids,
        colors=display_colors,
        output_path=gif_path,
        sampled_indices=sampled_indices,
        track_tail=args.track_tail,
        fps=args.gif_fps,
    )
    pred_3d_metrics = plot_reference_3d(
        pred=pred,
        colors=colors_all,
        reference_frame=reference_frame,
        output_path=pred_3d_path,
        ply_path=pred_ply_path,
        flip_y=args.flip_y_axis,
        normalize_3d=args.normalize_3d,
    )
    write_reference_3d_turntable_gif(
        pred=pred,
        colors=colors_all,
        reference_frame=reference_frame,
        output_path=pred_3d_gif_path,
        flip_y=args.flip_y_axis,
        fps=args.gif_fps,
        normalize_3d=args.normalize_3d,
    )
    dense_frames = compute_dense_reference_sequence(
        model=model,
        frames_resized=sampled_frames_resized,
        reference_frame=reference_frame,
        stride=args.dense_point_cloud_stride,
        vis_threshold=args.dense_vis_threshold,
        confidence_percentile=args.dense_confidence_percentile,
        batch_size=args.dense_query_batch_size,
        orig_width=orig_w,
        orig_height=orig_h,
        device=device,
        flip_y=args.flip_y_axis,
    )
    dense_metrics = plot_dense_reference_static(
        dense_frames=dense_frames,
        output_path=dense_static_path,
    )
    write_dense_reference_gif(
        dense_frames=dense_frames,
        output_path=dense_gif_path,
        fps=args.gif_fps,
    )
    save_point_cloud_ply(
        str(dense_ref_ply_path),
        dense_frames[reference_frame]["points_ref"],
        colors=dense_frames[reference_frame]["colors"],
    )

    dense_dynamic_ids, dense_dynamic_metrics = select_dense_dynamic_ids(
        dense_frames=dense_frames,
        reference_frame=reference_frame,
        motion_percentile=args.dense_motion_percentile,
    )
    dense_motion_artifacts: dict[str, str | None] = {
        "pred_dense_motion_reference_static": None,
        "pred_dense_motion_reference_gif": None,
        "pred_dense_motion_reference_ply": None,
    }
    if dense_dynamic_ids:
        dense_motion_frames = filter_dense_frames_by_ids(dense_frames, dense_dynamic_ids)
        plot_dense_reference_static(
            dense_frames=dense_motion_frames,
            output_path=dense_motion_static_path,
            title_prefix="Predicted dynamic-motion subset in reference frame",
        )
        write_dense_reference_gif(
            dense_frames=dense_motion_frames,
            output_path=dense_motion_gif_path,
            fps=args.gif_fps,
            title_prefix="Predicted dynamic-motion subset",
        )
        save_point_cloud_ply(
            str(dense_motion_ply_path),
            dense_motion_frames[reference_frame]["points_ref"],
            colors=dense_motion_frames[reference_frame]["colors"],
        )
        dense_motion_artifacts = {
            "pred_dense_motion_reference_static": str(dense_motion_static_path),
            "pred_dense_motion_reference_gif": str(dense_motion_gif_path),
            "pred_dense_motion_reference_ply": str(dense_motion_ply_path),
        }

    summary: dict[str, Any] = {
        "input_path": str(input_path),
        "input_kind": input_kind,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "model_variant": args.model_variant,
        "patch_provider": args.patch_provider,
        "resolution": int(args.resolution),
        "num_frames": int(args.num_frames),
        "sampling_mode": args.sampling_mode,
        "start_index": int(args.start_index),
        "input_video_num_frames": int(len(frames_all)),
        "input_video_fps": float(fps) if fps is not None else None,
        "sampled_frame_indices": list(map(int, sampled_indices)),
        "sampled_source_names": [source_names[idx] for idx in sampled_indices] if source_names is not None else None,
        "padded_tail_frames": int(padded_frames),
        "original_frame_size_hw": [int(orig_h), int(orig_w)],
        "display_frame_size_hw": [int(args.resolution), int(args.resolution)],
        "reference_frame_clip_index": int(reference_frame),
        "reference_frame_video_index": int(sampled_indices[reference_frame]),
        "query_grid_size": int(args.query_grid_size),
        "query_margin": float(args.query_margin),
        "dense_point_cloud_stride": int(args.dense_point_cloud_stride),
        "dense_vis_threshold": float(args.dense_vis_threshold),
        "dense_confidence_percentile": float(args.dense_confidence_percentile),
        "dense_motion_percentile": float(args.dense_motion_percentile),
        "num_query_points": int(len(query_points_norm)),
        "num_display_points": int(len(display_ids)),
        **summarize_predictions(
            pred=pred,
            query_points_px=query_points_px,
            reference_frame=reference_frame,
        ),
        **pred_3d_metrics,
        **dense_metrics,
        **dense_dynamic_metrics,
        "artifacts": {
            "reference_queries": str(query_frame_path),
            "tracks_2d_overlay": str(overlay_path),
            "tracks_2d_overlay_gif": str(gif_path),
            "pred_3d_reference": str(pred_3d_path),
            "pred_3d_reference_gif": str(pred_3d_gif_path),
            "pred_3d_reference_ply": str(pred_ply_path),
            "pred_dense_reference_static": str(dense_static_path),
            "pred_dense_reference_gif": str(dense_gif_path),
            "pred_dense_reference_ply": str(dense_ref_ply_path),
            **dense_motion_artifacts,
        },
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps({"summary_path": str(summary_path), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
