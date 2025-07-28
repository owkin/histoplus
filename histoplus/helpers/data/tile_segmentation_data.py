"""Data class for serializating and deserializing cell masks at the tile level."""

from __future__ import annotations
from typing import Any, Union
import numpy as np
from histoplus.helpers.data.segmentation_polygon import SegmentationPolygon


def get_tile_id(
    level: Union[str, int, float],
    x: Union[str, int, float],
    y: Union[str, int, float],
    width: Union[str, int, float],
    height: Union[str, int, float],
) -> str:
    """Get the tile id."""
    return f"{level}__{x}__{y}__{width}__{height}"


class TileSegmentationData:
    """Class for serializating and deserializing cell masks at the tile level.

    Attributes
    ----------
    tile_id : str
        Unique identifier of the tile.

    level : int
        DeepZoom level.

    x : float
        Tile coordinates (can be float for Visium data).

    y : float
        Tile coordinates (can be float for Visium data).

    width : int
        Tile width.

    height : int
        Tile height.

    tissue_type : Optional[TissueType]
        Tissue type.

    tumor_confidence : float
        Tumor confidence.

    masks : list[SegmentationPolygon]
        List of segmentation masks.
    """

    def __init__(
        self,
        tile_id: str,
        level: int,
        x: float,
        y: float,
        width: int,
        height: int,
        masks: list[SegmentationPolygon],
    ):
        self._tile_id = tile_id
        self._level = level
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._masks = masks

    @classmethod
    def from_predictions(
        cls,
        mask_coordinates: list[np.ndarray],
        centroid_coordinates: list[list[float]],
        cell_types: np.ndarray,
        probabilities: np.ndarray,
        metadata: dict[str, Union[str, int, float]],
    ) -> TileSegmentationData:
        """Create a TileSegmentationData object from the model predictions.

        Parameters
        ----------
        mask_coordinates : list[np.ndarray]
            Segmentation masks. Each np.ndarray is of shape [polygon_length, 2].
        centroids : list[list[float]]
            Centroid coordinates of the segmentation masks.
        cell_types : np.ndarray, shape [n_instances]
            Cell type identifiers.
        probabilities : np.ndarray, shape [n_instances]
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

        tile_id = get_tile_id(
            level=metadata["level"],
            x=metadata["x"],
            y=metadata["y"],
            width=metadata["width"],
            height=metadata["height"],
        )

        return cls(
            tile_id=tile_id,
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
            tile_id=data["tile_id"],
            level=int(data["level"]),
            x=int(data["x"]),
            y=int(data["y"]),
            width=int(data["width"]),
            height=int(data["height"]),
            masks=masks,
        )

    def to_dict(self, include_masks: bool = True) -> dict[str, Any]:
        """Convert the object to a dictionary.

        Parameters
        ----------
        include_masks : bool
            Include the `masks` key. Be careful this can be slow for large slides.

        Returns
        -------
        dict[str, Any]
        """
        masks = []

        if include_masks:
            masks = [mask.to_dict() for mask in self.masks]

        # Object of type int64 is not JSON-serializable, hence the conversion to float
        return {
            "tile_id": self.tile_id,
            "level": float(self.level),
            "x": float(self.x),
            "y": float(self.y),
            "width": float(self.width),
            "height": float(self.height),
            "masks": masks,
        }

    def metadata_dict(self) -> dict[str, Any]:
        """Get the metadata dictionary."""
        return {
            "tile_id": self.tile_id,
            "level": self.level,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @property
    def tile_id(self) -> str:
        """Get the tile id."""
        return self._tile_id

    @property
    def level(self) -> int:
        """Get the level."""
        return self._level

    @property
    def x(self) -> float:
        """Get the x coordinate."""
        return self._x

    @property
    def y(self) -> float:
        """Get the y coordinate."""
        return self._y

    @property
    def width(self) -> int:
        """Get the width."""
        return self._width

    @property
    def height(self) -> int:
        """Get the height."""
        return self._height

    @property
    def masks(self) -> list[SegmentationPolygon]:
        """Get the masks."""
        return self._masks

    @property
    def n_cells(self) -> int:
        """Get the number of cells."""
        return len(self.masks)
