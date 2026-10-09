"""RGB observation manifests for model inference."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class ObservationManifest(Dataset):
    """Read prepared RGB images, camera identities and shared frame labels."""

    def __init__(self, manifest, image_size=None):
        self.path = Path(manifest).resolve()
        text = self.path.read_text()
        if self.path.suffix == '.jsonl':
            self.records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            data = json.loads(text)
            self.records = data.get('clips', []) if isinstance(data, dict) else data
        if not isinstance(self.records, list) or not self.records:
            raise ValueError('manifest must contain a nonempty clip list')
        self.image_size = tuple(image_size) if image_size is not None else None
        if self.image_size is not None and (len(self.image_size) != 2 or min(self.image_size) <= 0):
            raise ValueError('image_size must contain positive [height,width]')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        observations = record.get('observations', [])
        if not observations:
            raise ValueError(f'clip {index} has no observations')
        images, camera_id, time_idx, valid = [], [], [], []
        for observation in observations:
            path = (self.path.parent / str(observation['image'])).resolve()
            with Image.open(path) as image:
                image = image.convert('RGB')
                if self.image_size is not None:
                    image = image.resize((self.image_size[1], self.image_size[0]), Image.Resampling.BICUBIC)
                pixels = np.array(image, copy=True)
            images.append(torch.from_numpy(pixels).permute(2, 0, 1).float() / 255)
            camera_id.append(observation['camera_id'])
            time_idx.append(observation['time_idx'])
            valid.append(observation.get('valid', True))
        if any(image.shape != images[0].shape for image in images):
            raise ValueError('all observations in a clip must have the same prepared image size')
        query = record.get('query_time_idx', sorted({t for t, v in zip(time_idx, valid) if v}))
        return dict(images=torch.stack(images), camera_id=torch.tensor(camera_id),
                    time_idx=torch.tensor(time_idx), query_time_idx=torch.tensor(query),
                    observation_valid=torch.tensor(valid, dtype=torch.bool), clip_id=str(record.get('id', index)))


def collate_observations(samples):
    if not samples:
        raise ValueError('cannot collate an empty batch')
    output = {'clip_id': [sample['clip_id'] for sample in samples]}
    for key in ('images', 'camera_id', 'time_idx', 'query_time_idx', 'observation_valid'):
        if any(sample[key].shape != samples[0][key].shape for sample in samples):
            raise ValueError(f'batch {key} shapes differ; bucket or pad clips explicitly')
        output[key] = torch.stack([sample[key] for sample in samples])
    return output
