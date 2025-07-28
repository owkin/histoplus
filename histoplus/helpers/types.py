"""Type useful for data manipulation."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TilePrediction:
    """Standardized output for the prediction of a tile.

    Note that each attributes of the dataclass is a list. All the lists are expected to
    be aligned and of the same length.
    """

    """Coordinates of the segmentation masks for all the cells in the tile."""
    contours: list[np.ndarray]

    """Coordinates of the bounding boxes of all the segmented cells in the tile."""
    bounding_boxes: list[list[float]]

    """Coordinates of the centroids for the cells in the tile."""
    centroids: list[list[float]]

    """Cell types of the cells in the tile."""
    cell_types: list[str]

    """Classification probability of the cell type."""
    cell_type_probabilities: list[float]


@dataclass
class BorderInfo:
    """Structured container for border information."""

    tiles: Optional[list[list[int]]]


@dataclass
class GlobalSegmentedCell:
    """Contain all information about a segmented cell."""

    """Coordinates of the segmentation mask. Shape (n_points, 2)."""
    contour: np.ndarray

    """Coordinates of the centroid of the cell. Shape (2)."""
    centroid: np.ndarray

    """Coordinates of the bounding box of the cell. Shape (4)."""
    bounding_box: np.ndarray

    """Type of the cell."""
    cell_type: str

    """Probability of the type of the cell."""
    cell_type_probability: float

    """Whether the cell is inside a tile margin."""
    in_safe_area: bool

    """Whether the cell is on a border."""
    is_touching_border: bool

    """Offset global (used to project the cell from a local to a global system)."""
    offset_global: np.ndarray

    """Tile coordinates (as output by the TilingTool)."""
    tile_coordinates: np.ndarray

    """Border information."""
    border_information: BorderInfo
