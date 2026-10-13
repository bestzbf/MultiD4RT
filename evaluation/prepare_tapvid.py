"""Download frozen official TAPVid-3D release files and verify their contents."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
from PIL import Image

from .common import digest, write_json


def verify(path):
    with np.load(path, allow_pickle=True) as data:
        required = {'images_jpeg_bytes', 'tracks_XYZ', 'visibility', 'queries_xyt', 'fx_fy_cx_cy'}
        if not required.issubset(data.files):
            raise ValueError('Missing RGB/GT fields: ' + str(required - set(data.files)))
        tracks, visible = data['tracks_XYZ'], data['visibility']
        frames = len(data['images_jpeg_bytes'])
        if tracks.shape != (*visible.shape, 3) or visible.shape[0] != frames:
            raise ValueError('Inconsistent annotation shapes')
        if not np.isfinite(tracks[visible]).all() or not (tracks[visible, 2] > 0).all():
            raise ValueError('Visible GT contains invalid camera coordinates')
        image = np.asarray(Image.open(io.BytesIO(bytes(data['images_jpeg_bytes'][0]))).convert('RGB'))
        intrinsics = data['fx_fy_cx_cy']
        if intrinsics.shape != (4,) or not np.isfinite(intrinsics).all() or min(intrinsics[:2]) <= 0:
            raise ValueError('Invalid intrinsics')
        return dict(frames=frames, tracks=tracks.shape[1], rgb_hw=list(image.shape[:2]),
                    intrinsics=intrinsics.tolist(), sha256=digest(path), bytes=path.stat().st_size)


def download_one(root, base, subset, name):
    path = root / subset / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix('.download')
        url = base + subset + '/' + name
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=45) as source, temporary.open('wb') as target:
                    while block := source.read(1024 * 1024):
                        target.write(block)
                verify(temporary)
                temporary.replace(path)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)
    return dict(subset=subset, name=name, path=str(path), **verify(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--subsets', nargs='+', default=['pstudio'])
    parser.add_argument('--split', choices=['full_eval', 'minival'], default='full_eval')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text())
    tasks = [(subset, name) for subset in args.subsets
             for name in source['datasets'][subset][args.split]]
    args.output.mkdir(parents=True, exist_ok=True)
    status = dict(stage='running', split=args.split, source_commit=source['commit'],
                  requested_files=len(tasks), files=[], failures=[])
    with ThreadPoolExecutor(args.workers) as pool:
        pending = {pool.submit(download_one, args.output, source['release_base_url'], subset, name):
                   (subset, name) for subset, name in tasks}
        for future in as_completed(pending):
            try:
                status['files'].append(future.result())
            except Exception as error:
                status['failures'].append(dict(task=pending[future], error=repr(error)))
            write_json(args.output / ('download_' + args.split + '.json'), status)
            print(json.dumps(dict(completed=len(status['files']), total=len(tasks), failed=len(status['failures']))), flush=True)
    status['stage'] = 'passed' if len(status['files']) == len(tasks) else 'failed'
    write_json(args.output / ('download_' + args.split + '.json'), status)
    if status['stage'] != 'passed':
        raise RuntimeError('Some official release files could not be prepared')


if __name__ == '__main__':
    main()
