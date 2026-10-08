"""Explicit multi-camera manifest I/O, independent of the existing D4RT loader."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .metadata import gather_observations


class MultiCameraManifestDataset(Dataset):
    """Read JSON/JSONL clips with explicit camera IDs and shared frame labels.

    Labels are prepared in a common world gauge. This adapter does not reinterpret
    D4RT's sparse camera-coordinate labels as world trajectories.
    """
    def __init__(self, manifest, load_targets=True, image_size=None):
        self.path = Path(manifest).resolve()
        text = self.path.read_text()
        if self.path.suffix == ".jsonl":
            self.records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            data = json.loads(text)
            self.records = data.get("clips", []) if isinstance(data, dict) else data
        if not isinstance(self.records, list) or not self.records:
            raise ValueError("manifest must contain a nonempty clip list")
        self.load_targets = load_targets
        self.image_size = tuple(image_size) if image_size is not None else None
        if self.image_size is not None and min(self.image_size) <= 0:
            raise ValueError("image_size must contain positive [height,width]")

    def __len__(self):
        return len(self.records)

    def _path(self, value):
        return (self.path.parent/str(value)).resolve()

    def __getitem__(self, index):
        record = self.records[index]
        observations = record.get("observations", [])
        if not observations:
            raise ValueError(f"clip {index} has no observations")
        if self.image_size is not None and self.load_targets and record.get("targets"):
            raise ValueError("resize training images and all pixel-coordinate labels together before creating a manifest")
        images, camera_id, time_idx, valid = [], [], [], []
        for observation in observations:
            with Image.open(self._path(observation["image"])) as image:
                image = image.convert("RGB")
                if self.image_size is not None:
                    image = image.resize((self.image_size[1], self.image_size[0]), Image.Resampling.BICUBIC)
                pixels = np.array(image, copy=True)
            images.append(torch.from_numpy(pixels).permute(2, 0, 1).float()/255)
            camera_id.append(observation["camera_id"])
            time_idx.append(observation["time_idx"])
            valid.append(observation.get("valid", True))
        if any(image.shape != images[0].shape for image in images):
            raise ValueError("all observations in a clip must have the same prepared image size")
        query = record.get("query_time_idx", sorted({t for t, v in zip(time_idx, valid) if v}))
        result = dict(images=torch.stack(images), camera_id=torch.tensor(camera_id),
                      time_idx=torch.tensor(time_idx), query_time_idx=torch.tensor(query),
                      observation_valid=torch.tensor(valid, dtype=torch.bool),
                      clip_id=str(record.get("id", index)), targets={})
        if self.load_targets and record.get("targets"):
            path = self._path(record["targets"])
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as data:
                    targets = {key: torch.from_numpy(np.array(value, copy=True))
                               for key, value in data.items()}
            else:
                data = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(data, dict):
                    raise ValueError("target files must contain a tensor dictionary")
                targets = {key: torch.as_tensor(value) for key, value in data.items()}
            # Model visibility cameras are returned in sorted camera-label order.
            if "target_camera_id" in record:
                labels = list(record["target_camera_id"])
                expected = sorted({c for c, v in zip(camera_id, valid) if v})
                if len(labels) != len(set(labels)) or set(labels) != set(expected):
                    raise ValueError("target_camera_id must label every visibility camera exactly once")
                order = torch.tensor([labels.index(c) for c in expected])
                for key in ("visibility", "visibility_valid", "sparse_visibility", "sparse_visibility_valid"):
                    if key in targets:
                        targets[key] = targets[key].index_select(2, order)
            result["targets"] = targets
        return result


def collate_multicamera(batch):
    """Stack prepared rectangular clips; missing views use observation_valid."""
    if not batch:
        raise ValueError("cannot collate an empty batch")
    output = {"clip_id": [sample["clip_id"] for sample in batch]}
    for key in ("images", "camera_id", "time_idx", "query_time_idx", "observation_valid"):
        if any(sample[key].shape != batch[0][key].shape for sample in batch):
            raise ValueError(f"batch {key} shapes differ; bucket or pad clips explicitly")
        output[key] = torch.stack([sample[key] for sample in batch])
    keys = set(batch[0]["targets"])
    if any(set(sample["targets"]) != keys for sample in batch):
        raise ValueError("batch target fields differ; use valid masks for absent supervision")
    output["targets"] = {}
    for key in keys:
        if any(sample["targets"][key].shape != batch[0]["targets"][key].shape for sample in batch):
            raise ValueError(f"batch target {key} shapes differ; bucket or pad targets explicitly")
        output["targets"][key] = torch.stack([sample["targets"][key] for sample in batch])
    return output


def select_source_targets(targets, source_indices, observation_count):
    """Dense manifest labels cover all S sources; sparse labels retain original IDs."""
    output = dict(targets)
    fields = ("trajectory", "displacement", "trajectory_valid", "displacement_valid",
              "visibility", "visibility_valid", "static_mask")
    for key in fields:
        if key in output:
            if output[key].shape[1] != observation_count:
                raise ValueError(f"dense {key} labels must cover all S observations before source selection")
            output[key] = gather_observations(output[key], source_indices)
    return output
