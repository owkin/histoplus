"""Extractor implementation."""

from .base import Extractor
from .timm import TimmExtractor


__all__ = ["Extractor", "TimmExtractor"]


def get_extractor(
    backbone_weights: str,
    encoder_keys: list[str],
    tile_size: int,
) -> Extractor:
    """Get extractor."""
    if backbone_weights == "aquavit_105k":
        return TimmExtractor(
            model="base_s14", output_layers=encoder_keys, tile_size=tile_size
        )
    raise ValueError(f"Unknown backbone weights. Got {backbone_weights}")
