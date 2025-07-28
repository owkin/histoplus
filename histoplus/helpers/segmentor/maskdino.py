"""MaskDINO segmentor."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Union

import torch
from torchvision.transforms import Compose, Normalize, ToTensor

from histoplus.helpers.nn.maskdino import MaskDINO
from .base import Segmentor


class MaskDINOSegmentor(Segmentor):
    """Implementation of the MaskDINO segmentor."""

    def __init__(
        self,
        model: torch.nn.Module,
        segmentor_name: str,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        gpu: Union[None, int, list[int]] = None,
        mixed_precision: bool = False,
    ):
        super().__init__()

        self.class_mapping = model.cell_type_mapping

        self.transform = Compose([ToTensor(), Normalize(mean=mean, std=std)])
        self.segmentor_name = segmentor_name

        self.train_image_size = self.model.train_image_size
        self.target_mpp = self.model.mpp

        self.mean = mean
        self.std = std

        self.postprocessor = None  # TODO

    @property
    def mpp(self) -> float:
        """Get the mpp of the model."""
        return self.model.mpp

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Infers the segmentation masks from the input images.

        Parameters
        ----------
        images : torch.Tensor
            A transformed image.

        Returns
        -------
        dict[str, torch.Tensor]
            Raw prediction maps used during post-processing.
        """
        outputs = self.model(images)

        # TODO

        return outputs

    def get_postprocess_fn(self) -> Callable:
        """Get the postprocessing function."""
        return self.postprocessor.postprocess