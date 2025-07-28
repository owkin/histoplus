"""Base class for segmentation model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, TypeVar

import torch
from PIL import Image


Transformed = TypeVar("Transformed")


class Segmentor(ABC):
    """Base class for segmentation model.

    The segmentor handles the call to the model and the post-processing of the raw
    output of the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for segmentation.
    transform : Callable[[Image.Image], Transformed]
        The transform method to apply to the image before
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = torch.nn.Identity()
        self._transform = lambda x: x
        self._device = "cpu"
        self._class_mapping = {}
        self._target_mpp = 0.5
        self._segmentor_name = ""
        self._train_image_size = 224
        self._mean = (0.0, 0.0, 0.0)
        self._std = (1.0, 1.0, 1.0)

    @abstractmethod
    def get_postprocess_fn(self) -> Callable:
        """Get the postprocessing function."""
        raise NotImplementedError

    @property
    def segmentor_name(self) -> str:
        """Name of the segmentor."""
        return self._segmentor_name

    @segmentor_name.setter
    def segmentor_name(self, name: str) -> None:
        """Set the name of the segmentor."""
        self._segmentor_name = name

    @property
    def target_mpp(self) -> float:
        """The mpp for the model during training."""
        return self._target_mpp

    @target_mpp.setter
    def target_mpp(self, mpp: float) -> None:
        """Set the target mpp for the model.

        Parameters
        ----------
        mpp : float
            The MPP of training.
        """
        self._target_mpp = mpp

    @property
    def train_image_size(self) -> int:
        """Train image size during training."""
        return self._train_image_size

    @train_image_size.setter
    def train_image_size(self, value: int) -> None:
        """Set the train image size.

        Parameters
        ----------
        value : int
            Value to set.
        """
        self._train_image_size = value

    @property
    def device(self) -> str:
        """Device for computation.

        Returns
        -------
        str
            Device.
        """
        return self._device

    @device.setter
    def device(self, new_device: str) -> None:
        """Set a new device to the segmentor.

        Parameters
        ----------
        device : str
            Device.
        """
        self._device = new_device

    @property
    def model(self) -> torch.nn.Module:
        """Cell segmentation and classification model.

        Returns
        -------
        torch.nn.Module
            The cell segmentation model.
        """
        return self._model

    @model.setter
    def model(self, model_module: torch.nn.Module) -> None:
        """Set a new model to the segmentor.

        Parameters
        ----------
        model_module : torch.nn.Module
            The model to be used to segment and classify cells.
        """
        self._model = model_module

    @property
    def mean(self) -> tuple[float, float, float]:
        """Mean statistics used for input normalization.

        Returns
        -------
        tuple[float, float, float]
            Mean statistics.
        """
        return self._mean

    @mean.setter
    def mean(self, value: tuple[float, float, float]) -> None:
        """Set a new mean statistics.

        Parameters
        ----------
        value: tuple[float, float, float]
            Value to set.
        """
        self._mean = value

    @property
    def std(self) -> tuple[float, float, float]:
        """Standard deviation statistics used for input normalization.

        Returns
        -------
        tuple[float, float, float]
            Standard deviation statistics.
        """
        return self._std

    @std.setter
    def std(self, value: tuple[float, float, float]) -> None:
        """Set a new std statistics.

        Parameters
        ----------
        value: tuple[float, float, float]
            Value to set.
        """
        self._std = value

    @property
    def transform(self) -> Callable[[Image.Image], Transformed]:
        """Transform method to apply to the image before segmentation.

        Returns
        -------
        Callable[[Image.Image], Transformed]
            The transform method.
        """
        return self._transform

    @transform.setter
    def transform(
        self, transform_function: Callable[[Image.Image], Transformed]
    ) -> None:
        """Set a new transform function to the segmentor.

        Parameters
        ----------
        transform_function: Callable[[PIL.Image.Image], Transformed]
            The transform function to be set for the extractor.
        """
        self._transform = transform_function

    @property
    def class_mapping(self) -> dict[int, str]:
        """Mapping of class integer to their names.

        Returns
        -------
        dict[int, str]
            Mapping of class integers to names.
        """
        return self._class_mapping

    @class_mapping.setter
    def class_mapping(self, mapping: dict[int, str]) -> None:
        """Set the mapping of integer to their names.

        Parameters
        ----------
        mapping : dict[int, str]
            Mapping of cell types to their integer representation.
        """
        self._class_mapping = mapping

    @property
    def inference_image_size(self) -> int:
        """The spatial dimension of the image used during inference.

        Returns
        -------
        int
            Inference image size.
        """
        return self._model.inference_image_size

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
        return self.model(images.to(self.device))
