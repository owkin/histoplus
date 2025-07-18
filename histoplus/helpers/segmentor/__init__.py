"""Segmentor implementation."""

from .base import Segmentor
from .maskdino import MaskDINOSegmentor


__all__ = ["Segmentor", "MaskDINOSegmentor"]
