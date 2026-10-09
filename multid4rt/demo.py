"""CPU inference demo with random tiny weights and generated RGB inputs."""

import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from . import MultiD4RT, MultiD4RTConfig
from .checkpoint import save_checkpoint
from .infer import main as infer_main


def make_demo_checkpoint(path):
    torch.manual_seed(0)
    model = MultiD4RT(MultiD4RTConfig.tiny(gradient_checkpointing=False)).eval()
    save_checkpoint(path, model)
    return Path(path)


def make_demo_input(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    h, w = 28, 42
    times = [0, 5, 10]
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    observations = []
    for camera in range(2):
        for time in times:
            rgb = torch.stack(((xx.float()/w+time/40).remainder(1), yy.float()/h,
                               (xx+yy).float().remainder(8)/8), -1)
            path = output/f"camera{camera}_frame{time}.png"
            Image.fromarray((rgb.numpy()*255).astype("uint8")).save(path)
            observations.append(dict(image=path.name, camera_id=camera, time_idx=time))
    manifest = output/"observations.json"
    manifest.write_text(json.dumps(dict(clips=[dict(id="demo", observations=observations)]), indent=2))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    manifest = make_demo_input(output/"images")
    checkpoint = make_demo_checkpoint(output/"tiny_random.pt")
    infer_main(["--manifest", str(manifest), "--checkpoint", str(checkpoint),
                "--output", str(output/"prediction"), "--device", "cpu", "--cpu-threads", "2"])
    prediction = torch.load(output/"prediction"/"clip_000000.pt", weights_only=True)
    assert prediction["trajectory"].shape == (1, 6, 3, 28, 42, 3)
    for source, slot in enumerate([0, 1, 2, 0, 1, 2]):
        assert torch.count_nonzero(prediction["displacement"][:, source, slot]) == 0
    assert torch.isfinite(prediction["trajectory"]).all()
    print(json.dumps(dict(passed=True, checkpoint=str(checkpoint), weights="random_tiny")))


if __name__ == "__main__":
    main()
