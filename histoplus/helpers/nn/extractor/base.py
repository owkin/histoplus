"""Base classes for feature extraction."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from typing import Optional

import torch
from torch import nn


class Extractor(nn.Module, ABC):
    """Base class for extractor."""

    def __init__(self):
        super().__init__()
        self._output_layers: Optional[Sequence[str]] = None
        self.out_channels: int
        self._patch_size: int = 16

    @abstractmethod
    def __call__(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """Return a dictionary with intermediate features.

        Parameters
        ----------
        images: torch.Tensor
            (BS, C, H, W)

        Returns
        -------
        features : OrderedDict[str, torch.Tensor]
            example shapes: {
                "layer4": (BS, 256, 56, 56),
                "layer5": (BS, 512, 28, 28),
                "layer6": (BS, 1024, 14, 14),
                "layer7": (BS, 2048, 7, 7),
            }
        """

    @property
    def output_layers(self) -> Sequence[str]:
        """Get output layers.

        The output layers are the layers of the encoder which we want to output
        in the extract_features. In the previous example, it would be ("layer4",...)

        Returns
        -------
        Sequence[str]
            A sequence containing strings referencing the output layers.

        Raises
        ------
        ValueError
            If the output layers are not instanced.
        """
        if self._output_layers is None:
            raise ValueError("Output layer not instanced yet.")
        return self._output_layers

    @output_layers.setter
    def output_layers(self, value: Sequence[str]):
        """Set output layers.

        The output layers are the layers of the encoder which we want to output
        in the extract_features. In the previous example, it would be ("layer4",...)

        Parameters
        ----------
        value: Sequence[str]
            An iterable containing strings referencing the output layers.
        """
        self._output_layers = value

    @property
    def patch_size(self) -> int:
        """Patch size of the encoder.

        Returns
        -------
        int
            Patch size of the encoder.
        """
        return self._patch_size

    @patch_size.setter
    def patch_size(self, value: int):
        """Set the patch size.

        Parameters
        ----------
        value : int
            Value of the patch size.
        """
        self._patch_size = value
