"""
Visualize augmented video frames + 2D/3D query inputs from a kubric batch.
Output: /tmp/kubric_vis/
"""
import os, sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from torch.utils.data import DataLoader, SequentialSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)
from datasets.factory import create_training_dataset
from datasets.collate import d4rt_collate_fn

OUT_DIR = "/tmp/kubric_vis"
os.makedirs(OUT_DIR, exist_ok=True)

# ── load config & dataset ──────────────────────────────────────────────────
with open("configs/single_kubric.yaml") as f:
    config = yaml.safe_load(f)

ds = create_training_dataset(config, split="train")
loader = DataLoader(ds, batch_size=2, num_workers=0,
                    sampler=SequentialSampler(ds),
                    collate_fn=d4rt_collate_fn)

batch = next(iter(loader))

video   = batch["video"]        # (B, T, C, H, W)  float [0,1]
coords  = batch["coords"]       # (B, Q, 2)  normalized [0,1]
t_src   = batch["t_src"]        # (B, Q)
targets = batch["targets"]
# trajs_3d_world stored in targets if present
trajs_3d = targets.get("pos_3d", None)  # (B, Q, 3)

B, T, C, H, W = video.shape
print(f"video: {tuple(video.shape)}, coords: {tuple(coords.shape)}, t_src: {tuple(t_src.shape)}")
if trajs_3d is not None:
    print(f"trajs_3d_world: {tuple(trajs_3d.shape)}")

# ── 1. Augmented frames (first sample, every 8th frame) ───────────────────
b = 0
frames_to_show = list(range(0, T, max(1, T // 6)))[:6]
fig, axes = plt.subplots(1, len(frames_to_show), figsize=(3 * len(frames_to_show), 3))
for ax, fi in zip(axes, frames_to_show):
    img = video[b, fi].permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(f"t={fi}")
    ax.axis("off")
fig.suptitle(f"Augmented frames — {batch['sequence_names'][b]}")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/frames_b{b}.png", dpi=100)
plt.close(fig)
print(f"Saved {OUT_DIR}/frames_b{b}.png")

# ── 2. 2D query coords overlaid on source frame ───────────────────────────
for b in range(B):
    fig, axes = plt.subplots(1, min(4, T), figsize=(4 * min(4, T), 4))
    if min(4, T) == 1:
        axes = [axes]
    shown = set()
    ax_idx = 0
    for q in range(coords.shape[1]):
        ti = int(t_src[b, q].item())
        if ti in shown or ax_idx >= len(axes):
            continue
        shown.add(ti)
        ax = axes[ax_idx]; ax_idx += 1
        img = video[b, ti].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        # all queries whose t_src == ti
        mask = (t_src[b] == ti)
        xy = coords[b][mask].numpy()  # [M, 2], normalized [0,1]
        ax.scatter(xy[:, 0] * W, xy[:, 1] * H, s=4, c='red', alpha=0.5, linewidths=0)
        ax.set_title(f"t_src={ti}  n={mask.sum()}")
        ax.axis("off")
    fig.suptitle(f"2D queries — {batch['sequence_names'][b]}")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/coords2d_b{b}.png", dpi=100)
    plt.close(fig)
    print(f"Saved {OUT_DIR}/coords2d_b{b}.png")

# ── 3. 3D query points (scatter) ──────────────────────────────────────────
if trajs_3d is not None:
    for b in range(B):
        pts = trajs_3d[b].numpy()           # (Q, 3)
        valid = np.isfinite(pts).all(-1)
        pts = pts[valid]
        if len(pts) == 0:
            print(f"b={b}: no valid 3D points"); continue
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, alpha=0.3)
        ax.set_title(f"3D queries — {batch['sequence_names'][b]}\n{len(pts)} valid pts")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/coords3d_b{b}.png", dpi=100)
        plt.close(fig)
        print(f"Saved {OUT_DIR}/coords3d_b{b}.png")
else:
    print("trajs_3d_world not in targets, skipping 3D plot")

print("Done.")
