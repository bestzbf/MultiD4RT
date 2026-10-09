"""Real H5 I/O smoke: mixed native Kubric multi-view and dynamic MUGEN data."""

import argparse
import json
from pathlib import Path

import numpy as np


def make_demo_data(output):
    """Generate independent synthetic GT in both migrated native H5 schemas."""
    import h5py

    output = Path(output).resolve()
    kubric = output/"kubric"
    kubric.mkdir(parents=True, exist_ok=True)
    t, c, h, w, n = 11, 2, 28, 42, 6
    K = np.array([[30., 0., 21.], [0., 30., 14.], [0., 0., 1.]], np.float32)
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], c, axis=0)
    c2w[:, 0, 3] = [.3, .5]
    world = np.empty((n, t, 3), np.float32)
    uv = np.empty((n, t, c, 2), np.float32)
    radial_track = np.empty((n, t, c), np.float32)
    for time in range(t):
        us = np.arange(n, dtype=np.float32)*2+10.5+time*.15
        vs = np.full(n, 14.5, np.float32)
        xyz = np.stack(((us-21.)*2./30., (vs-14.)*2./30., np.full(n, 2.)), -1)
        world[:, time] = xyz+c2w[0, :3, 3]
        for camera in range(c):
            local = world[:, time]-c2w[camera, :3, 3]
            uv[:, time, camera] = local[:, :2]/local[:, 2:]*30.+[21., 14.]
            radial_track[:, time, camera] = np.linalg.norm(local, axis=-1)
    yy, xx = np.mgrid[:h, :w]
    radial_depth = 2.*np.sqrt(1.+((xx+.5-21.)/30.)**2+((yy+.5-14.)/30.)**2)
    visible = np.ones((n, t, c), bool)
    visible[0, 5] = False
    with h5py.File(kubric/"scene.h5", "w") as handle:
        handle.create_dataset("frames", data=np.full((t, c, h, w, 3), 100, np.uint8), chunks=(1, 1, h, w, 3))
        handle.create_dataset("depth", data=np.broadcast_to(radial_depth, (t, c, h, w)).astype(np.float32))
        handle.create_dataset("intrinsics", data=np.repeat(K[None], c, axis=0))
        # Maintained revision-4 packs have OpenCV C2W, not Blender C2W.
        handle.create_dataset("extrinsics", data=c2w)
        handle.create_dataset("camera_ids", data=np.array([b"cam0", b"cam1"]))
        handle.create_dataset("tracks_3d", data=world)
        handle.create_dataset("tracks_2d", data=uv)
        handle.create_dataset("track_depth", data=radial_track)
        handle.create_dataset("visibility", data=visible)
        handle.attrs["schema_version"] = "movi_tracking_v3"
        handle.attrs["fps"] = 30.
    packed = output/"dynamic.h5"
    array_K = K.copy()
    array_K[:2, 2] -= .5
    with h5py.File(packed, "w") as handle:
        group = handle.create_group("train/scene")
        group.create_dataset("images", data=np.full((t, h, w, 3), 150, np.uint8), chunks=(1, h, w, 3))
        group.create_dataset("depths", data=np.full((t, h, w), 2., np.float32), chunks=(1, h, w))
        group.create_dataset("intrinsics", data=np.repeat(array_K[None], t, axis=0))
        group.create_dataset("extrinsics", data=np.repeat(np.linalg.inv(c2w[0])[None], t, axis=0))
        for name, value in {
            "trajs_3d_world": world.transpose(1, 0, 2),
            "trajs_2d": uv[:, :, 0].transpose(1, 0, 2)-.5,
            "valids": np.ones((t, n), bool),
            "visibs": visible[:, :, 0].T,
            "visibility_valids": np.ones((t, n), bool),
        }.items():
            group.create_dataset(name, data=value, chunks=(1,)+value.shape[1:])
    config = dict(image_size=[h, w], num_times=3, num_queries=16,
                  epoch_size=2, seed=0, sources=[
        dict(adapter="kubric_multicam", root="kubric", split="all", cameras=2,
             strides=[5], options=dict(verbose=False, index_workers=1,
                required_schema_version="movi_tracking_v3", require_native_tracks=True)),
        dict(adapter="mugen_h5", root="dynamic.h5", split="train", cameras=1,
             strides=[5], options=dict(verbose=False, dataset_name="pointodyssey_demo",
                train_only_val_ratio=0., require_precomputed_tracks=True)),
    ])
    path = output/"data.json"
    path.write_text(json.dumps(config, indent=2)+"\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/native_adapter_smoke")
    args = parser.parse_args(argv)
    from ..demo import make_demo_checkpoint
    from ..infer import main as infer_main
    from .mixture import create_dataset, collate_adapter_samples
    output = Path(args.output).resolve()
    config = make_demo_data(output/"data")
    dataset = create_dataset(config)
    batch = collate_adapter_samples([dataset[0], dataset[1]])
    assert set(batch["dataset_name"]) == {"kubric_multicam", "mugen_h5"}
    assert batch["images"].shape == (2, 6, 3, 28, 42)
    assert batch["targets"]["sparse_visibility"].shape == (2, 16, 3, 2)
    print(json.dumps({"mixed_sources": batch["dataset_name"],
                      "images": list(batch["images"].shape),
                      "query_time_idx": batch["query_time_idx"].tolist()}))
    checkpoint = make_demo_checkpoint(output/"tiny_random.pt")
    infer_main(["--data-config", str(config), "--checkpoint", str(checkpoint),
        "--output", str(output/"prediction"), "--device", "cpu", "--max-clips", "2",
        "--source-obs", "0", "--cpu-threads", "1"])
    return output


if __name__ == "__main__":
    main()
