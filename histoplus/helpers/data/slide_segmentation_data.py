"""Data class for serializating and deserializing cell masks at the slide level."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import numpy as np

from histoplus.helpers.data.segmentation_polygon import SegmentationPolygon
from histoplus.helpers.data.tile_segmentation_data import TileSegmentationData
from histoplus.helpers.types import TilePrediction


@dataclass(frozen=True)
class SlideSegmentationData:
    """Main data structure for storing and manipulating slide-level data."""

    # Model name used for the inference
    model_name: str

    # MPP used for the inference
    inference_mpp: float

    # Cell masks organized in tiles
    cell_masks: list[TileSegmentationData]

    @classmethod
    def from_predictions(
        cls,
        model_name: str,
        inference_mpp: float,
        deepzoom_level: int,
        tile_size: int,
        tile_coordinates: np.ndarray,
        tile_predictions: list[TilePrediction],
    ) -> SlideSegmentationData:
        """Create a SlideSegmentationData object from the model predictions."""
        assert len(tile_coordinates) == len(tile_predictions)

        cell_masks = []

        for tile_idx, tile_prediction in enumerate(tile_predictions):
            cell_masks.append(
                TileSegmentationData.from_predictions(
                    mask_coordinates=tile_prediction.contours,
                    centroid_coordinates=tile_prediction.centroids,
                    cell_types=tile_prediction.cell_types,
                    probabilities=tile_prediction.cell_type_probabilities,
                    metadata={
                        "level": deepzoom_level,
                        "x": tile_coordinates[tile_idx, 0],
                        "y": tile_coordinates[tile_idx, 1],
                        "width": tile_size,
                        "height": tile_size,
                    },
                )
            )

        return cls(
            model_name=model_name,
            inference_mpp=inference_mpp,
            cell_masks=cell_masks,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> SlideSegmentationData:
        """Load a SlideSegmentationData object from a path."""
        with open(path, "r") as fin:
            raw_json = json.load(fin)

        # Reconstruct nested dataclasses
        raw_cell_masks = raw_json.get("cell_masks", [])
        cell_masks: list[TileSegmentationData] = []
        for tile_dict in raw_cell_masks:
            masks_dicts = tile_dict.get("masks", [])
            masks = [SegmentationPolygon(**mask_dict) for mask_dict in masks_dicts]
            cell_masks.append(
                TileSegmentationData(
                    level=int(tile_dict["level"]),
                    x=float(tile_dict["x"]),
                    y=float(tile_dict["y"]),
                    width=int(tile_dict["width"]),
                    height=int(tile_dict["height"]),
                    masks=masks,
                )
            )

        return cls(
            model_name=raw_json["model_name"],
            inference_mpp=float(raw_json["inference_mpp"]),
            cell_masks=cell_masks,
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save the slide segmentation data to a file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f)
