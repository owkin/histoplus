"""Base interface for a post-processor."""

from abc import ABC, abstractmethod
from typing import Any

from histoplus.helpers.types import TilePrediction


class Postprocessor(ABC):
    """Base class for a post-processor.

    The segmentor saves raw prediction maps. The postprocessor is in charge of
    assembling them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def postprocess(self, outputs: Any) -> list[TilePrediction]:
        """Postprocessing function."""
        raise NotImplementedError
