"""Manifest inference: python -m multicamera4rc.infer --help."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from . import MultiCamera4RC, MultiCamera4RCConfig
from .data import MultiCameraManifestDataset, collate_multicamera
from .checkpoint import config_from_checkpoint, read_checkpoint
from .distributed import resolve_amp_dtype


def _cpu_payload(value):
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, dict):
        return {key: _cpu_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_payload(item) for item in value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    data_input = parser.add_mutually_exclusive_group(required=True)
    data_input.add_argument("--manifest", help="Prepared JSON/JSONL observations")
    data_input.add_argument("--data-config", help="Native adapter mixture JSON/YAML configuration")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preset", choices=("giant", "tiny"), default="giant",
                        help="Architecture fallback for original checkpoints without config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--source-obs", type=int, nargs="+", help="Default: every source observation")
    parser.add_argument("--query-times", type=int, nargs="+", help="Shared frame labels, not flattened observation slots")
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("H", "W"),
                        help="Resize input only, for inference; both dimensions must be multiples of 14")
    parser.add_argument("--no-visibility", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision; geometry remains float32")
    parser.add_argument("--amp-dtype", choices=("auto", "bfloat16", "float16"), default="auto")
    parser.add_argument("--no-query-chunking", action="store_true")
    parser.add_argument("--decoder-memory", choices=("all_observations", "target_time"),
                        help="Override saved memory scope for an explicit architecture ablation")
    parser.add_argument("--time-origin", choices=("clip_start", "absolute"),
                        help="Override saved time encoding for an explicit architecture ablation")
    parser.add_argument("--max-clips", type=int, help="Limit manifest or native-dataset predictions")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.cpu_threads <= 0:
        parser.error("cpu-threads must be positive")
    if args.max_clips is not None and args.max_clips <= 0:
        parser.error("max-clips must be positive")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    # Native checkpoints select their saved architecture; original 4RC uses giant.
    checkpoint = read_checkpoint(args.checkpoint)
    fallback = MultiCamera4RCConfig.tiny() if args.preset == "tiny" else MultiCamera4RCConfig()
    cfg = config_from_checkpoint(checkpoint, fallback=fallback)
    cfg = replace(cfg, gradient_checkpointing=False)
    if args.no_query_chunking:
        cfg = replace(cfg, attention_chunk_size=None, visibility_chunk_size=None)
    if args.decoder_memory is not None:
        cfg = replace(cfg, decoder_memory=args.decoder_memory)
    if args.time_origin is not None:
        cfg = replace(cfg, time_origin=args.time_origin)
    model = MultiCamera4RC.from_checkpoint(checkpoint, config=cfg).to(device).eval()
    dtype = resolve_amp_dtype(device, args.amp_dtype) if args.amp else torch.float32
    del checkpoint
    if args.data_config:
        from .datasets.mixture import create_dataset, collate_adapter_samples
        if args.image_size is not None:
            parser.error("set image_size in the native data config to transform calibration and pixels together")
        dataset = create_dataset(args.data_config)
        collate_fn = collate_adapter_samples
    else:
        dataset = MultiCameraManifestDataset(args.manifest, load_targets=False, image_size=args.image_size)
        collate_fn = collate_multicamera
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"inference.json").write_text(json.dumps({
        "checkpoint": str(Path(args.checkpoint).resolve()), "model_config": cfg.to_dict(),
        "amp": args.amp, "amp_dtype": str(dtype), "device": str(device),
    }, indent=2)+"\n")
    with torch.inference_mode():
        count = min(len(dataset), args.max_clips) if args.max_clips is not None else len(dataset)
        for index in range(count):
            batch = collate_fn([dataset[index]])
            query = args.query_times if args.query_times is not None else batch["query_time_idx"]
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=args.amp):
                prediction = model(batch["images"].to(device), batch["camera_id"].to(device),
                                   batch["time_idx"].to(device), query, args.source_obs,
                                   observation_valid=batch["observation_valid"].to(device),
                                   return_visibility=not args.no_visibility)
            # Fixed numeric file names do not interpret user clip IDs as paths.
            path = output/f"clip_{index:06d}.pt"
            payload = _cpu_payload(prediction)
            payload["clip_id"] = batch["clip_id"][0]
            torch.save(payload, path)
            print(json.dumps({"clip": payload["clip_id"], "output": str(path),
                              "trajectory": list(prediction["trajectory"].shape)}))
    return output


if __name__ == "__main__":
    main()
