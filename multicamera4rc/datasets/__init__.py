"""Dataset adapters and the MultiCamera4RC data bridge.

Importing this package does not eagerly import OpenCV, HDF5, or individual
dataset adapters.  Install the optional ``data`` extra to read raw datasets.
"""

from .registry import create_adapter, get_adapter_class, list_datasets

__all__ = ["create_adapter", "get_adapter_class", "list_datasets"]
