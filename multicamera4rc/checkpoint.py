"""Load original 4RC weights or compatible native multi-camera checkpoints."""

from collections.abc import Mapping
from pathlib import Path

import torch

from .config import MultiCamera4RCConfig


def config_from_checkpoint(data, fallback=None):
    """Restore saved execution semantics, including checkpoints predating scopes.

    Historical native checkpoints used target-time memory and absolute frame
    encodings. New models use all observations and a shared clip time origin.
    Weight compatibility does not justify changing a resumed computation, so
    migration of either setting is explicit.
    """
    values = data.get("config") if isinstance(data, Mapping) else None
    if isinstance(values, Mapping) and "embed_dim" in values:
        values = dict(values)
        values.setdefault("decoder_memory", "target_time")
        values.setdefault("time_origin", "absolute")
        return MultiCamera4RCConfig.from_dict(values)
    return fallback if fallback is not None else MultiCamera4RCConfig()


def read_checkpoint(checkpoint):
    if isinstance(checkpoint, Mapping):
        return checkpoint
    path = Path(checkpoint)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("install the 'weights' extra to read safetensors") from exc
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def _is_extension_key(key):
    return (key.startswith(("backbone.frame_projection.", "target_projection.",
                            "delta_projection.", "visibility_head.", "cam_dec.fc_principal."))
            or key == "motion_decoder.camera_bias"
            or key.endswith(".attn.camera_relation_weight"))


def load_4rc_weights(model, checkpoint, strict_base=True):
    """Require original module coverage; only newly added parameters may be absent.

    A D4RT checkpoint is not a 4RC checkpoint. Shape mismatches always raise,
    including in partial mode; no incompatible weight is silently reshaped.
    """
    data = read_checkpoint(checkpoint)
    if not isinstance(data, Mapping):
        raise ValueError("checkpoint must contain a state dictionary")
    native = data.get("architecture") == "multicamera4rc"
    for wrapper in ("model", "state_dict", "model_state_dict"):
        if wrapper in data and isinstance(data[wrapper], Mapping):
            data = data[wrapper]
            break
    state = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            continue
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        if key in state:
            raise ValueError(f"duplicate checkpoint key after removing prefixes: {key}")
        state[key] = value
    expected = model.state_dict()
    matched = {key: value for key, value in state.items() if key in expected}
    if not matched:
        raise ValueError("no weights match this model; use 4RC weights, not existing D4RT weights")
    mismatches = {key: (tuple(value.shape), tuple(expected[key].shape))
                  for key, value in matched.items() if value.shape != expected[key].shape}
    if mismatches:
        raise ValueError(f"checkpoint architecture/shape mismatch: {mismatches}")
    missing = sorted(set(expected)-set(matched))
    missing_base = [key for key in missing if not _is_extension_key(key)]
    unexpected = sorted(set(state)-set(expected))
    if strict_base and native and (missing or unexpected):
        raise ValueError(f"native checkpoint must cover the complete model; missing={missing}, unexpected={unexpected}")
    if strict_base and missing_base:
        raise ValueError(f"checkpoint is missing original 4RC parameters: {missing_base}")
    # Validate the entire mapping before mutating any parameter.
    model.load_state_dict(matched, strict=False)
    return dict(matched=len(matched), missing=missing, missing_base=missing_base,
                unexpected=unexpected)


def save_checkpoint(path, model, step=0, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(architecture="multicamera4rc", format_version=2,
                model=model.state_dict(), config=model.config.to_dict(), step=int(step),
                **metadata)
    temporary = path.with_name(path.name+".tmp")
    torch.save(data, temporary)
    temporary.replace(path)
