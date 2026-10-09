"""One inference/viewer entry for multiview, monocular and multi-camera RGB."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .demo import make_demo_checkpoint
from .infer import main as infer_main
from .visual_inputs import MODES, load_visual_input
from .visualization import build_visualization, export_visualization


def make_visual_demo(directory, mode):
    directory = Path(directory)
    mode = 'multicamera' if mode == 'auto' else mode
    cameras = range(3 if mode == 'multiview' else 2 if mode == 'multicamera' else 1)
    times = [0] if mode == 'multiview' else [0, 5, 10]
    for camera in cameras:
        height, width = (27, 39) if camera == 0 else (23, 37)
        y, x = np.mgrid[:height, :width]
        for time in times:
            image = np.stack(((x/width+time/40)%1, y/height,
                              ((x+y+camera*2)%8)/8), -1)
            path = (directory/f'view_{camera}.png' if mode == 'multiview' else
                    directory/f'frame_{time:05d}.png' if mode == 'monocular' else
                    directory/f'camera_{camera}'/f'frame_{time:05d}.png')
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((image*255).astype(np.uint8)).save(path)
    return directory, mode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--input', '--manifest', dest='input', type=Path,
                        help='Video, image folder, camera subfolders/videos, or JSON/JSONL observations')
    source.add_argument('--demo', action='store_true', help='Generate example RGB; use random tiny weights without a checkpoint')
    parser.add_argument('--mode', choices=MODES, default='auto')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--prediction', type=Path, help='Reuse a .pt prediction or a previous visualization output directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clip-index', type=int, default=0, help='Selected clip within a manifest')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--max-frames', type=int, help='Default: 16 for inference; 0 for reuse. Set 0 for all frames')
    parser.add_argument('--long-side', type=int, help='Aspect-preserving downscale: default 518 for inference, 0 for reuse')
    parser.add_argument('--source-obs', type=int, nargs='+', help='Default: first observation of each camera')
    parser.add_argument('--query-times', type=int, nargs='+')
    parser.add_argument('--queries', type=Path, help='JSON array of [source_observation,original_x,original_y]')
    parser.add_argument('--max-points', type=int, default=20000, help='Display/export point budget per frame; 0 keeps all pixels')
    parser.add_argument('--num-tracks', type=int, default=24)
    parser.add_argument('--max-depth', type=float, help='Optional display filter, in checkpoint depth units')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--amp-dtype', choices=('auto','bfloat16','float16'), default='auto')
    parser.add_argument('--query-chunking', action='store_true', help='Use checkpoint query chunking; default disables it')
    parser.add_argument('--no-visibility', action='store_true')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    if args.prediction and (args.demo or args.checkpoint or args.source_obs or args.query_times):
        parser.error('reusing predictions accepts input/display options, not a checkpoint or new source/time requests')
    if args.input is None and not args.demo and args.prediction is None:
        parser.error('provide --input, --demo, or a prior --prediction directory')
    if not args.demo and not args.prediction and args.checkpoint is None:
        parser.error('inference requires --checkpoint; --demo can use random tiny weights')
    if args.max_points < 0 or args.num_tracks < 1 or args.cpu_threads < 1:
        parser.error('point limits, track count and CPU thread count must be positive (point limit may be 0)')
    if args.max_depth is not None and (not np.isfinite(args.max_depth) or args.max_depth <= 0):
        parser.error('--max-depth must be finite and positive')
    try:
        from plotly.offline import get_plotlyjs  # Check the optional dependency before any expensive inference.
    except ImportError:
        parser.error("install the visualization extra: pip install 'multid4rt[viz]'")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_path, mode, checkpoint = args.input, args.mode, args.checkpoint
    random_weights = False
    if args.demo:
        input_path, mode = make_visual_demo(output/'demo_input', mode)
        if checkpoint is None:
            checkpoint = make_demo_checkpoint(output/'tiny_random.pt')
            random_weights = True
    prediction_path = args.prediction
    if prediction_path and prediction_path.is_dir():
        previous_summary = prediction_path/'visualization.json'
        if previous_summary.exists():
            random_weights = json.loads(previous_summary.read_text()).get('weights') == 'random_tiny'
        input_path = input_path or prediction_path/'input/observations.json'
        prediction_path = prediction_path/'prediction/clip_000000.pt'
    if input_path is None:
        parser.error('a prediction file also needs --input with matching RGB observations')
    visual = load_visual_input(input_path, mode, stride=args.stride,
        max_frames=args.max_frames if args.max_frames is not None else 0 if prediction_path else 16,
        long_side=args.long_side if args.long_side is not None else 0 if prediction_path else 518,
        clip_index=args.clip_index)
    manifest = visual.write(output/'input')
    if prediction_path is None:
        sources = args.source_obs
        if sources is None:
            sources = []
            for camera in sorted(set(visual.camera_id[0].tolist())):
                sources.append(torch.nonzero(visual.camera_id[0] == camera, as_tuple=False)[0].item())
        arguments = ['--manifest', str(manifest), '--checkpoint', str(checkpoint),
                     '--output', str(output/'prediction'), '--device', args.device,
                     '--source-obs', *map(str, sources), '--cpu-threads', str(args.cpu_threads)]
        if args.query_times:
            arguments += ['--query-times', *map(str, args.query_times)]
        if args.amp:
            arguments += ['--amp', '--amp-dtype', args.amp_dtype]
        if not args.query_chunking:
            arguments += ['--no-query-chunking']
        if args.no_visibility:
            arguments += ['--no-visibility']
        infer_main(arguments)
        prediction_path = output/'prediction/clip_000000.pt'
    prediction = torch.load(prediction_path, map_location='cpu', weights_only=True)
    queries = json.loads(args.queries.read_text()) if args.queries else None
    payload = build_visualization(prediction, visual, max_points=args.max_points,
                                  num_tracks=args.num_tracks, queries=queries, max_depth=args.max_depth)
    payload['settings']['weights'] = 'random_tiny' if random_weights else 'provided_prediction' if args.prediction else 'provided_checkpoint'
    result = export_visualization(payload, output)
    print(json.dumps(dict(viewer=str(result), mode=visual.mode, cameras=len(payload['cameras']),
                         times=payload['times'], sources=payload['settings']['source_observations']), ensure_ascii=False))
    return result


if __name__ == '__main__':
    main()
