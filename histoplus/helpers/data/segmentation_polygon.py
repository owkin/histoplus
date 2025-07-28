"""Data class for serializating and deserializing segmentation masks."""


from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True, kw_only=True)
class SegmentationPolygon:
    """Data class for serializating and deserializing segmentation masks.

    Attributes
    ----------
    cell_id : int
        Unique identifier of the cell

    cell_type : str
        Biological classification of the cell

    confidence : float
        Probability of the assigned classification

    coordinates : list[list[float]]
        Polygon coordinates of the mask

    centroid : list[float]
        Centroid coordinates
    """
    cell_id: int
    cell_type: str
    confidence: float
    coordinates: list[list[float]]
    centroid: list[float]

    @cached_property
    def points(self) -> np.ndarray:
        """Return the polygon coordinates as a numpy array.

        It ensures that the polygon is closed by repeating the first point at the end.

        Returns
        -------
        np.ndarray, shape [n_points, 2]
            Point coordinates.
        """
        coords = np.array(self.coordinates)
        if not np.array_equal(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])
        return coords
