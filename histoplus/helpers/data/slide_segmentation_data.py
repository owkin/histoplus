"""Data class for serializating and deserializing cell masks at the slide level."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from histoplus.helpers.types import TilePrediction


@dataclass
class SlideSegmentationData:
    """Main data structure for storing and manipulating slide-level data."""

    # Path to the slide
    slide_path: Union[str, Path]

    # MPP of the slide
    mpp: float

    # Cell masks organized in tiles
    cell_masks: list[TilePrediction]

    # Coordinates of the tiles
    coords: np.ndarray

    # DeepZoom level of the tiles
    level: int

    # Tile size
    tile_size: int

    # Model name
    model_name: str

    def save(self, path: Union[str, Path]) -> None:
        """Save the slide segmentation data to a file."""
        with open(path, "w") as f:
            json.dump(self, f)
