"""Data class for serializating and deserializing cell masks at the slide level."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from histoplus.helpers.serializers.base import BaseSerializer
from histoplus.helpers.types import TilePrediction


class SlideSegmentationData:
    """Main data structure for storing and manipulating slide-level data.

    This data structure has two purposes:

    1. Provide an interface for serializing and deserializing all data inferred by
    a cell segmentation model.

    2. Store histomics features as they are computed by a pipeline of histomics functons.

    Note that we do not force the user to store the histomics features in this data
    structure. All the histomics functions are designed to accept a SlideSegmentationData
    object and output a pd.DataFrame. Yet, when running a pipeline of histomics functions,
    like `compute_cell_shape_descriptors` followed by `aggregate_cell_level_features`,
    it is very convenient to store the intermediate results in the `cell_level_features`
    attribute of a SlideSegmentationData object.

    Parameters
    ----------
    serializer : BaseSerializer
        The serializer strategy to use for data persistence.
    data : Any
        The internal data representation (e.g., protobuf object).
    """

    def __init__(self, serializer: BaseSerializer, data: Any):
        """Initialize with a serializer strategy and data."""
        self._serializer = serializer
        self._data = data

        # Feature storage
        self._slide_level_features: Optional[pd.DataFrame] = None
        self._tile_level_features: Optional[pd.DataFrame] = None
        self._cell_level_features: Optional[pd.DataFrame] = None

    @classmethod
    def from_predictions(
        cls,
        slide_path: Union[str, Path],
        mpp: float,
        cell_masks: list[TilePrediction],
        tissue_types: list[TissueType],
        tumor_confidence: np.ndarray,
        coords: np.ndarray,
        level: int,
        max_level: int,
        tile_size: int,
        model_name: str,
        serializer: Optional[BaseSerializer] = None,
    ) -> SlideSegmentationData:
        """Create a SlideSegmentationData object from the model predictions."""
        if serializer is None:
            serializer = ProtobufSerializer()

        data = serializer.create_from_predictions(
            slide_path=slide_path,
            mpp=mpp,
            cell_masks=cell_masks,
            tissue_types=tissue_types,
            tumor_confidence=tumor_confidence,
            coords=coords,
            level=level,
            max_level=max_level,
            tile_size=tile_size,
            model_name=model_name,
        )

        return cls(serializer=serializer, data=data)

    @classmethod
    def from_file_path(
        cls, path: Union[str, Path], serializer: Optional[BaseSerializer] = None
    ) -> SlideSegmentationData:
        """Create a SlideSegmentationData object from a file."""
        if serializer is None:
            serializer = ProtobufSerializer()

        data = serializer.load(path)
        return cls(serializer=serializer, data=data)

    def save(self, path: Union[str, Path]) -> None:
        """Save the object to a file."""
        self._serializer.save(self._data, path)

    @property
    def slide_id(self) -> str:
        """Return the slide id."""
        return self._data.slide_id

    @property
    def slide_path(self) -> Path:
        """Return the slide path."""
        return Path(self._data.slide_path)

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._data.model_name

    @property
    def mpp(self) -> float:
        """Return the mpp."""
        return self._data.mpp

    @property
    def max_level(self) -> int:
        """Return the max level."""
        return self._data.max_level

    def metadata_dict(self) -> dict[str, Any]:
        """Get the metadata dictionary of the slide."""
        return {
            "slide_id": self.slide_id,
            "slide_path": str(self.slide_path),
            "model_name": self.model_name,
            "mpp": self.mpp,
            "max_level": self.max_level,
        }
