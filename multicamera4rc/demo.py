"""Reproducible CPU integration smoke with independent synthetic supervision."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .infer import main as infer_main
from .train import main as train_main


def make_demo_dataset(output):
    """Simple calibrated translating rig and constant-depth plane; labels are not model outputs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    h, w = 28, 42
    times = [0, 5, 10]
    cameras = [0, 0, 0, 1, 1, 1]
    frame_labels = times*2
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    observations = []
    K = torch.eye(3).repeat(6, 1, 1)
    K[:, 0, 0], K[:, 1, 1] = 30., 30.
    K[:, 0, 2], K[:, 1, 2] = w/2, h/2
    c2w = torch.eye(4).repeat(6, 1, 1)
    c2w[3:, 0, 3] = 0.25
    c2w[:, 1, 3] = torch.tensor(frame_labels)*0.002
    for s, (camera, time) in enumerate(zip(cameras, frame_labels)):
        rgb = torch.stack(((xx.float()/w+time/40).remainder(1), yy.float()/h,
                           (xx+yy).float().remainder(8)/8), -1)
        path = output/f"camera{camera}_frame{time}.png"
        Image.fromarray((rgb.numpy()*255).astype(np.uint8)).save(path)
        observations.append(dict(image=path.name, camera_id=camera, time_idx=time))
    queries = torch.tensor([[s, 10., 8.] for s in range(6)] + [[0, 20., 18.]])
    tracks = torch.zeros(len(queries), 3, 3)
    for n, (source, u, v) in enumerate(queries):
        s = int(source)
        base = torch.tensor([(u-w/2)/30*2, (v-h/2)/30*2, 2.])+c2w[s, :3, 3]
        for target, time in enumerate(times):
            tracks[n, target] = base+torch.tensor([0.01*(time-frame_labels[s]), 0., 0.])
    targets = dict(depth=torch.full((6, h, w), 2.), K=K, c2w=c2w,
                   sparse_queries=queries, sparse_trajectory=tracks,
                   sparse_trajectory_valid=torch.ones(len(queries), 3, dtype=torch.bool))
    torch.save(targets, output/"targets.pt")
    manifest = output/"manifest.json"
    manifest.write_text(json.dumps({"clips": [dict(id="synthetic_smoke", observations=observations,
                         query_time_idx=times, targets="targets.pt")]}, indent=2))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-steps", type=int, default=2)
    args = parser.parse_args(argv)
    if args.train_steps <= 0:
        parser.error("train-steps must be positive")
    output = Path(args.output)
    manifest = make_demo_dataset(output/"data")
    checkpoint = train_main(["--manifest", str(manifest), "--output", str(output/"training"),
                             "--preset", "tiny", "--device", "cpu", "--steps", str(args.train_steps),
                             "--source-count", "2", "--cpu-threads", "2", "--save-every", "1"])
    infer_main(["--manifest", str(manifest), "--checkpoint", str(checkpoint),
                "--output", str(output/"prediction"), "--device", "cpu", "--cpu-threads", "2"])
    prediction = torch.load(output/"prediction"/"clip_000000.pt", weights_only=True)
    assert prediction["trajectory"].shape == (1, 6, 3, 28, 42, 3)
    for source, slot in enumerate([0, 1, 2, 0, 1, 2]):
        assert torch.count_nonzero(prediction["displacement"][:, source, slot]) == 0
    assert torch.isfinite(prediction["trajectory"]).all()
    print(json.dumps({"smoke": "passed", "checkpoint": str(checkpoint)}))


if __name__ == "__main__":
    main()
