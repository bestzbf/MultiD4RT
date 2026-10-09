"""Ordinary images/videos to synchronized RGB observations for the demo."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch

from .metadata import validate_observations


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
VIDEO_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
MODES = ('auto', 'multicamera', 'multiview', 'monocular')


def natural_key(path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', str(path))]


def frame_label(path):
    """Use the trailing frame number, rather than each stream's list position."""
    match = re.search(r'(\d+)$', path.stem)
    if not match:
        raise ValueError(f'{path.name}: camera sequences need a trailing frame number or a JSON manifest')
    return int(match.group(1))


def read_video(path, stride, max_frames):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("video input requires: pip install 'multid4rt[viz]'") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f'cannot open video: {path}')
    frames, index = {}, 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                frames[index] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if max_frames and len(frames) >= max_frames:
                    break
            index += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError(f'video has no decodable frames: {path}')
    return frames


def mode_for(cameras, times):
    if len(set(cameras)) > 1:
        return 'multicamera' if len(set(times)) > 1 else 'multiview'
    return 'monocular'


@dataclass
class VisualInput:
    images: torch.Tensor                 # [1,S,3,H,W], padded RGB
    camera_id: torch.Tensor              # [1,S]
    time_idx: torch.Tensor               # [1,S], original frame labels
    pixel_valid: torch.Tensor            # [S,H,W], content, not padding
    observations: list                  # raw/content sizes and coordinate transforms
    mode: str
    clip_id: str

    def write(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, info in enumerate(self.observations):
            name = f'observation_{index:04d}.png'
            pixels = (self.images[0, index].permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
            Image.fromarray(pixels).save(directory/name)
            rows.append(dict(image=name, camera_id=info['camera_id'], time_idx=info['time_idx']))
        path = directory/'observations.json'
        path.write_text(json.dumps(dict(clips=[dict(id=self.clip_id, observations=rows)]), indent=2)+'\n')
        (directory/'preprocessing.json').write_text(json.dumps(dict(mode=self.mode, observations=self.observations), indent=2)+'\n')
        return path


def load_visual_input(path, mode='auto', *, stride=1, max_frames=16, long_side=518, clip_index=0):
    path = Path(path).resolve()
    if mode not in MODES or stride < 1 or max_frames < 0 or long_side < 0 or clip_index < 0:
        raise ValueError('invalid mode, stride, frame limit, image size or clip index')
    rows, clip_id, restored = [], path.stem, None
    if path.suffix.lower() in ('.json', '.jsonl'):
        data = (list(json.loads(x) for x in path.read_text().splitlines() if x.strip())
                if path.suffix.lower() == '.jsonl' else json.loads(path.read_text()))
        clips = data.get('clips', []) if isinstance(data, dict) else data
        if not isinstance(clips, list) or not 0 <= clip_index < len(clips):
            raise ValueError('manifest clip index is outside its clip list')
        clip = clips[clip_index]
        clip_id = str(clip.get('id', clip_index))
        observations = [o for o in clip['observations'] if o.get('valid', True)]
        for observation in observations:
            for key in ('camera_id', 'time_idx'):
                if type(observation[key]) is not int:
                    raise ValueError(f'{key} must contain integer indices')
        timeline = sorted({o['time_idx'] for o in observations})[::stride]
        if max_frames:
            timeline = timeline[:max_frames]
        for observation in observations:
            if observation['time_idx'] in timeline:
                rows.append((int(observation['camera_id']), int(observation['time_idx']),
                             path.parent/str(observation['image'])))
        metadata = path.parent/'preprocessing.json'
        if metadata.exists():
            restored = json.loads(metadata.read_text())
    elif path.suffix.lower() in VIDEO_SUFFIXES:
        if mode not in ('auto', 'monocular'):
            raise ValueError('one video is a monocular sequence')
        rows = [(0, frame, image) for frame, image in read_video(path, stride, max_frames).items()]
    elif path.suffix.lower() in IMAGE_SUFFIXES:
        rows = [(0, 0, path)]
    elif path.is_dir():
        images = sorted((p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)
        videos = sorted((p for p in path.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES), key=natural_key)
        folders = sorted((p for p in path.iterdir() if p.is_dir() and
                          any(q.suffix.lower() in IMAGE_SUFFIXES for q in p.iterdir())), key=natural_key)
        if images:
            if videos or folders:
                raise ValueError('mixed top-level images and camera streams need a manifest')
            if mode == 'auto' and len(images) > 1:
                raise ValueError('flat image folder is ambiguous; choose --mode multiview or --mode monocular')
            if mode == 'multicamera':
                raise ValueError('multicamera folders need one image subfolder or video per camera')
            if mode == 'multiview':
                rows = [(camera, 0, image) for camera, image in enumerate(images)]
            else:
                chosen = images[::stride][:max_frames or None]
                numbered = all(re.search(r'(\d+)$', p.stem) for p in images)
                labels = [frame_label(p) for p in chosen] if numbered else list(range(len(images)))[::stride][:max_frames or None]
                rows = [(0, label, image) for label, image in zip(labels, chosen)]
        elif videos or folders:
            if videos and folders:
                raise ValueError('mixed video and image camera streams need a manifest')
            if mode == 'multiview':
                raise ValueError('static multiview input expects images in one folder')
            streams = ([read_video(p, stride, max_frames) for p in videos] if videos else
                       [dict((frame_label(p), p) for p in sorted(folder.iterdir(), key=natural_key)
                             if p.suffix.lower() in IMAGE_SUFFIXES) for folder in folders])
            if folders:
                for folder, stream in zip(folders, streams):
                    count = sum(p.suffix.lower() in IMAGE_SUFFIXES for p in folder.iterdir())
                    if len(stream) != count:
                        raise ValueError(f'duplicate frame labels in {folder}')
            timeline = sorted(set.intersection(*(set(stream) for stream in streams)))
            if folders:
                timeline = timeline[::stride][:max_frames or None]
            if not timeline:
                raise ValueError('camera streams share no frame labels; use a synchronized manifest')
            rows = [(camera, frame, stream[frame]) for camera, stream in enumerate(streams) for frame in timeline]
        else:
            raise ValueError(f'no images or camera streams in {path}')
    else:
        raise ValueError(f'unsupported or missing input: {path}')
    if not rows:
        raise ValueError('input has no selected valid observations')
    # A manifest's first observation defines the model's world reference.
    # Keep its order; generated directory/video rows are already camera-major.
    cameras, times = [r[0] for r in rows], [r[1] for r in rows]
    actual = mode_for(cameras, times)
    if mode != 'auto' and actual != mode:
        # A one-frame monocular sequence or one-view static scene is valid.
        if mode == 'multicamera' and actual == 'multiview':
            pass  # A synchronized stream can contain just one selected frame.
        elif mode == 'multiview' and len(set(times)) == 1:
            actual = mode
        else:
            raise ValueError(f'input metadata describes {actual}, not {mode}')
    validate_observations(torch.tensor([cameras]), torch.tensor([times]))
    prepared, infos = [], []
    prior = {(o['camera_id'], o['time_idx']): o for o in restored['observations']} if restored else {}
    for camera, frame, item in rows:
        if isinstance(item, Image.Image):
            image = item.convert('RGB')
        else:
            with Image.open(item) as source:
                image = source.convert('RGB')
        width, height = image.size
        scale = min(1., long_side/max(height, width)) if long_side else 1.
        content = (max(1, round(height*scale)), max(1, round(width*scale)))
        if content != (height, width):
            image = image.resize((content[1], content[0]), Image.Resampling.LANCZOS)
        prepared.append(torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float()/255)
        info = dict(camera_id=camera, time_idx=frame, raw_hw=[height, width], content_hw=list(content),
                    scale_xy=[content[1]/width, content[0]/height])
        if prior and not long_side:
            info = prior.get((camera, frame), info)
            ch, cw = info['content_hw']
            if not (0 < ch <= height and 0 < cw <= width):
                raise ValueError('stored content dimensions exceed prepared image dimensions')
        infos.append(info)
    height = max(28, (max(x.shape[-2] for x in prepared)+13)//14*14)
    width = max(28, (max(x.shape[-1] for x in prepared)+13)//14*14)
    pixels = torch.zeros(len(rows), 3, height, width)
    masks = torch.zeros(len(rows), height, width, dtype=torch.bool)
    for index, (image, info) in enumerate(zip(prepared, infos)):
        h, w = image.shape[-2:]
        pixels[index, :, :h, :w] = image
        ch, cw = info['content_hw']
        masks[index, :ch, :cw] = True
    return VisualInput(pixels[None], torch.tensor([cameras]), torch.tensor([times]), masks, infos, actual, clip_id)
