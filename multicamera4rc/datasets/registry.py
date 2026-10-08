"""Lazy raw-data adapter registry, isolated from the original D4RT package.

Only the selected adapter is imported.  Unknown constructor options are an
error even when an upstream adapter accepts ``**kwargs`` for compatibility.
"""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .adapters.base import BaseAdapter


_ADAPTERS: dict[str, tuple[str, str]] = {
    "blendedmvs": ("blendedmvs", "BlendedMVSAdapter"),
    "co3dv2": ("co3dv2", "Co3Dv2Adapter"),
    "dynamic_replica": ("dynamic_replica", "DynamicReplicaAdapter"),
    "egoexo4d": ("egoexo4d", "EgoExo4DAdapter"),
    "kubric": ("kubric", "KubricAdapter"),
    "kubric_multicam": ("kubric_multicam", "KubricMultiCamAdapter"),
    "mugen_h5": ("mugen_h5", "MugenH5Adapter"),
    "mvssynth": ("mvssynth", "MVSSynthAdapter"),
    "pointodyssey": ("pointodyssey", "PointOdysseyAdapter"),
    "pointodyssey_geometry": ("dense_geometry", "PointOdysseyGeometryAdapter"),
    "pointodyssey_tracks": ("dense_geometry", "PointOdysseyTracksAdapter"),
    "tartanair_v1_zip": ("dense_geometry", "TartanAirV1ZipAdapter"),
    "tartanair_v1_stereo": ("stereo", "TartanAirV1StereoAdapter"),
    "tartanair_v2_zip": ("tartanair_v2", "TartanAirV2ZipAdapter"),
    "hypersim_zip": ("dense_geometry", "HypersimZipAdapter"),
    "scannet": ("scannet", "ScanNetAdapter"),
    "scannetpp": ("scannetpp", "ScanNetPPAdapter"),
    "syn4d": ("syn4d", "Syn4DAdapter"),
    "tartanair": ("TartanAir", "TartanAirAdapter"),
    "vkitti2": ("VirtualKitti", "VKITTI2Adapter"),
    "vkitti2_stereo": ("stereo", "VKITTI2StereoAdapter"),
    "waymo": ("Waymo", "WaymoAdapter"),
}

# Kubric consumes these real options from **kwargs.  Other **kwargs in the
# migrated adapters merely swallow unsupported legacy factory options.
_EXTRA_OPTIONS: dict[str, frozenset[str]] = {
    "kubric": frozenset({
        "canonical_cache_items", "io_workers", "visibility_depth_radius",
        "visibility_depth_rel_tol", "visibility_mode",
    }),
}


def list_datasets() -> list[str]:
    """Return supported dataset names without importing their dependencies."""
    return sorted(_ADAPTERS)


def get_adapter_class(name: str) -> type[BaseAdapter]:
    """Load one adapter class; imports of other dataset packages stay lazy."""
    if not isinstance(name, str):
        raise TypeError("dataset name must be a string")
    key = name.strip().lower()
    if key not in _ADAPTERS:
        raise ValueError(
            f"Unknown dataset {name!r}. Available: {', '.join(list_datasets())}"
        )
    module_name, class_name = _ADAPTERS[key]
    try:
        module = importlib.import_module(
            f"multicamera4rc.datasets.adapters.{module_name}"
        )
    except ModuleNotFoundError as error:
        raise ImportError(
            f"Dataset {key!r} requires the optional data dependencies. "
            "Install with pip install -e '.[data]'. "
            f"Missing module: {error.name}"
        ) from error
    return getattr(module, class_name)


def create_adapter(name: str, **options: Any) -> BaseAdapter:
    """Construct an adapter using its original keyword arguments.

    ``root`` and ``split`` retain each adapter's upstream meaning.  Options
    unsupported by that dataset raise rather than silently disappearing.
    """
    adapter_class = get_adapter_class(name)
    signature = inspect.signature(adapter_class.__init__)
    supported = {
        key for key, parameter in signature.parameters.items()
        if key != "self" and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    supported.update(_EXTRA_OPTIONS.get(name.strip().lower(), ()))
    unknown = sorted(set(options).difference(supported))
    if unknown:
        raise TypeError(
            f"Unsupported option(s) for dataset {name!r}: {', '.join(unknown)}. "
            f"Supported: {', '.join(sorted(supported))}"
        )
    return adapter_class(**options)


__all__ = ["create_adapter", "get_adapter_class", "list_datasets"]
