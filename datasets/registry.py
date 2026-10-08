"""
Dataset adapter registry for D4RT.

Maintains a mapping from dataset name to adapter class.
Allows instantiation of adapters from configuration.

Usage:
    adapter = create_adapter('pointodyssey', root='/path/to/data', split='train')

    # Or get all available datasets
    names = list_datasets()
"""

from __future__ import annotations

import inspect
from typing import Type

from datasets.adapters.base import BaseAdapter
from datasets.adapters.blendedmvs import BlendedMVSAdapter
from datasets.adapters.co3dv2 import Co3Dv2Adapter
from datasets.adapters.dynamic_replica import DynamicReplicaAdapter
from datasets.adapters.kubric import KubricAdapter
from datasets.adapters.mvssynth import MVSSynthAdapter
from datasets.adapters.pointodyssey import PointOdysseyAdapter
from datasets.adapters.scannet import ScanNetAdapter
from datasets.adapters.TartanAir import TartanAirAdapter
from datasets.adapters.VirtualKitti import VKITTI2Adapter
from datasets.adapters.scannetpp import ScanNetPPAdapter


# Dataset name -> Adapter class mapping
DATASET_REGISTRY: dict[str, Type[BaseAdapter]] = {
    "pointodyssey": PointOdysseyAdapter,
    "scannet": ScanNetAdapter,
    "co3dv2": Co3Dv2Adapter,
    "kubric": KubricAdapter,
    "blendedmvs": BlendedMVSAdapter,
    "mvssynth": MVSSynthAdapter,
    "dynamic_replica": DynamicReplicaAdapter,
    "tartanair": TartanAirAdapter,
    "vkitti2": VKITTI2Adapter,
    "scannetpp": ScanNetPPAdapter,

}

# Lazy-loaded adapters (requires special dependencies)
LAZY_ADAPTERS = {
    "waymo": ("datasets.adapters.Waymo", "WaymoAdapter"),
}

_OPTIONAL_COMMON_KWARGS = frozenset({"cache_dir", "index_workers"})


def _filter_optional_common_kwargs(
    adapter_class: Type[BaseAdapter],
    kwargs: dict,
) -> dict:
    """Drop optional factory/cache kwargs unsupported by a specific adapter."""
    if not kwargs:
        return {}

    try:
        signature = inspect.signature(adapter_class.__init__)
    except (TypeError, ValueError):
        return dict(kwargs)

    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return dict(kwargs)

    supported = {
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    unsupported_optional = _OPTIONAL_COMMON_KWARGS.difference(supported)
    if not unsupported_optional:
        return dict(kwargs)

    return {
        key: value
        for key, value in kwargs.items()
        if key not in unsupported_optional
    }


def register_dataset(name: str, adapter_class: Type[BaseAdapter]) -> None:
    """Register a new dataset adapter."""
    if name in DATASET_REGISTRY:
        raise ValueError(f"Dataset '{name}' already registered")
    DATASET_REGISTRY[name] = adapter_class


def _load_lazy_adapter(name: str) -> Type[BaseAdapter]:
    """Load a lazy adapter on demand."""
    if name not in LAZY_ADAPTERS:
        return None

    module_path, class_name = LAZY_ADAPTERS[name]
    try:
        import importlib
        module = importlib.import_module(module_path)
        adapter_class = getattr(module, class_name)
        # Cache it for future use
        DATASET_REGISTRY[name] = adapter_class
        return adapter_class
    except Exception as e:
        raise ImportError(
            f"Failed to load adapter '{name}' from {module_path}.{class_name}: {e}"
        )


def create_adapter(name: str, **kwargs) -> BaseAdapter:
    """
    Create an adapter instance by name.

    Args:
        name: Dataset name (e.g., 'pointodyssey', 'scannet')
        **kwargs: Arguments passed to adapter constructor

    Returns:
        Adapter instance

    Example:
        adapter = create_adapter('pointodyssey', root='/data/pointodyssey', split='train')
    """
    # Check if already loaded
    if name in DATASET_REGISTRY:
        adapter_class = DATASET_REGISTRY[name]
        return adapter_class(**_filter_optional_common_kwargs(adapter_class, kwargs))

    # Try lazy loading
    if name in LAZY_ADAPTERS:
        adapter_class = _load_lazy_adapter(name)
        return adapter_class(**_filter_optional_common_kwargs(adapter_class, kwargs))

    # Not found
    available = ", ".join(list(DATASET_REGISTRY.keys()) + list(LAZY_ADAPTERS.keys()))
    raise ValueError(f"Unknown dataset '{name}'. Available: {available}")


def list_datasets() -> list[str]:
    """Get list of registered dataset names."""
    return list(DATASET_REGISTRY.keys()) + list(LAZY_ADAPTERS.keys())


def get_adapter_class(name: str) -> Type[BaseAdapter]:
    """Get adapter class by name."""
    if name in DATASET_REGISTRY:
        return DATASET_REGISTRY[name]

    if name in LAZY_ADAPTERS:
        return _load_lazy_adapter(name)

    available = ", ".join(list(DATASET_REGISTRY.keys()) + list(LAZY_ADAPTERS.keys()))
    raise ValueError(f"Unknown dataset '{name}'. Available: {available}")
