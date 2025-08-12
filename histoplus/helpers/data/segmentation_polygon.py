"""Data class for serializating and deserializing segmentation masks."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentationPolygon:
        """Create a SegmentationPolygon object from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing the data to create the object.

        Returns
        -------
        SegmentationPolygon
        """
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert the object to a dictionary.

        Returns
        -------
        dict[str, Any]
        """
        coordinates_list = np.asarray(self.coordinates).tolist()
        centroid_list = np.asarray(self.centroid).tolist()

        return {
            "cell_id": float(self.cell_id),
            "cell_type": self.cell_type,
            "confidence": self.confidence,
            "coordinates": coordinates_list,
            "centroid": centroid_list,
        }
