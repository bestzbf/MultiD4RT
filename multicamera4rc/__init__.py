"""Synchronous, uncalibrated multi-camera reconstruction and point tracking."""

from .config import MultiCamera4RCConfig
from .metadata import pack_synchronized_images
from .model import EncodedScene, MultiCamera4RC

__all__ = ["MultiCamera4RC", "MultiCamera4RCConfig", "EncodedScene", "pack_synchronized_images"]
