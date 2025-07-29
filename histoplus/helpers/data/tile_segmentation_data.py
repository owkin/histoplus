"""Data class for serializating and deserializing cell masks at the tile level."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from histoplus.helpers.data.segmentation_polygon import SegmentationPolygon


@dataclass(frozen=True)
class TileSegmentationData:
    """Data class for storing and manipulating tile-level data."""

    # DeepZoom level
    level: int

    # Tile coordinates (can be float for Visium data)
    x: float

    # Tile coordinates (can be float for Visium data)
    y: float

    # Tile width
    width: int

    # Tile height
    height: int

    # List of segmentation masks
    masks: list[SegmentationPolygon]

    @classmethod
    def from_predictions(
        cls,
        mask_coordinates: list[np.ndarray],
        centroid_coordinates: list[list[float]],
        cell_types: list[str],
        probabilities: list[float],
        metadata: dict[str, Union[str, int, float]],
    ) -> TileSegmentationData:
        """Create a TileSegmentationData object from the model predictions.

        Parameters
        ----------
        mask_coordinates : list[np.ndarray]
            Segmentation masks. Each np.ndarray is of shape [polygon_length, 2].
        centroids : list[list[float]]
            Centroid coordinates of the segmentation masks.
        cell_types : list[str]
            Cell type identifiers.
        probabilities : list[float]
            Cell type confidence scores.
        metadata : dict[str, Union[str, int, float]]
            Metadata of the tile.

        Returns
        -------
        TileSegmentationData
        """
        segmentation_polygons = []

        for cell_id, (mask_coords, centroid, cell_type, probability) in enumerate(
            zip(
                mask_coordinates,
                centroid_coordinates,
                cell_types,
                probabilities,
                strict=True,
            )
        ):
            mask_coords_list: list[list[float]] = mask_coords.tolist()
            segmentation_polygons.append(
                SegmentationPolygon(
                    cell_id=cell_id,
                    cell_type=cell_type,
                    confidence=probability,
                    coordinates=mask_coords_list,
                    centroid=centroid,
                )
            )

        return cls(
            level=int(metadata["level"]),
            x=int(metadata["x"]),
            y=int(metadata["y"]),
            width=int(metadata["width"]),
            height=int(metadata["height"]),
            masks=segmentation_polygons,
        )
