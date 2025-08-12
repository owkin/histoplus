"""Data class for serializating and deserializing cell masks at the tile level."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TileSegmentationData:
        """Create a TileSegmentationData object from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary containing the data to create the object.

        Returns
        -------
        TileSegmentationData
        """
        masks = [SegmentationPolygon.from_dict(mask) for mask in data["masks"]]

        return cls(
            level=int(data["level"]),
            x=int(data["x"]),
            y=int(data["y"]),
            width=int(data["width"]),
            height=int(data["height"]),
            masks=masks,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the object to a dictionary.

        Parameters
        ----------
        include_masks : bool
            Include the `masks` key. Be careful this can be slow for large slides.

        Returns
        -------
        dict[str, Any]
        """
        masks = [mask.to_dict() for mask in self.masks]

        # Object of type int64 is not JSON-serializable, hence the conversion to float
        return {
            "level": float(self.level),
            "x": float(self.x),
            "y": float(self.y),
            "width": float(self.width),
            "height": float(self.height),
            "masks": masks,
        }
