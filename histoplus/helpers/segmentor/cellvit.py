"""CellViT segmentor trained on the HIPE dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Union

import torch
from torchvision.transforms import Compose, Normalize, ToTensor

from histoplus.helpers.constants import DEFAULT_DEVICE
from histoplus.helpers.hub import (
    histoplus_cellvit_segmentor_20x,
    histoplus_cellvit_segmentor_40x,
)
from histoplus.helpers.mixed_precision import prepare_module
from histoplus.helpers.nn.cellvit.model import CellViT
from histoplus.helpers.postprocessor import CellViTPostprocessor
from histoplus.helpers.segmentor.base import Segmentor


CellViTModelFn = Union[
    type[histoplus_cellvit_segmentor_20x], type[histoplus_cellvit_segmentor_40x]
]


MODEL_FUNCS_FOR_VERSION: dict[float, CellViTModelFn] = {
    0.5: histoplus_cellvit_segmentor_20x,
    0.25: histoplus_cellvit_segmentor_40x,
}


class CellViTSegmentor(Segmentor):
    """CellViT segmentor implementation."""

    def __init__(
        self,
        model: CellViT,
        segmentor_name: str,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        gpu: Union[None, int, list[int]] = DEFAULT_DEVICE,
        mixed_precision: bool = False,
    ):
        super().__init__()

        self.class_mapping = model.cell_type_mapping

        self.transform = Compose([ToTensor(), Normalize(mean=mean, std=std)])
        self.model = prepare_module(model, gpu, mixed_precision)
        self.segmentor_name = segmentor_name

        self.train_image_size = model.train_image_size
        self.target_mpp = model.mpp

        self.mean = mean
        self.std = std

        self.postprocessor = CellViTPostprocessor(self.target_mpp, self.class_mapping)

    @classmethod
    def from_weights_path(
        cls,
        weights_path: Union[str, Path],
        cell_type_mapping: dict[int, str],
        mpp: float,
        segmentor_name: str,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        mixed_precision: bool = False,
        **model_kwargs,
    ) -> CellViTSegmentor:
        """Create a CellViT segmentor with a custom checkpoint."""
        model = CellViT(
            cell_type_mapping=cell_type_mapping,
            mpp=mpp,
            **model_kwargs,
        )

        model.load_state_dict(
            torch.load(weights_path, weights_only=False, map_location="cpu")
        )

        return cls(
            model=model,
            segmentor_name=segmentor_name,
            mean=mean,
            std=std,
            mixed_precision=mixed_precision,
        )

    @staticmethod
    def _get_model_fn(mpp: float) -> CellViTModelFn:
        try:
            return MODEL_FUNCS_FOR_VERSION[mpp]
        except KeyError as exc_mpp:
            raise ValueError(
                "The requested MPP is not available for version. "
            ) from exc_mpp

    @classmethod
    def from_histoplus(
        cls,
        mixed_precision: bool,
        mpp: float,
        inference_image_size: int,
    ) -> CellViTSegmentor:
        """Get the best checkpoint from HIPE iterations."""
        model_fn = cls._get_model_fn(mpp)
        return cls(
            model=model_fn(inference_image_size),  # type: ignore
            segmentor_name=model_fn.name,
            mean=model_fn.mean,
            std=model_fn.std,
            mixed_precision=mixed_precision,
        )

    @property
    def mpp(self) -> float:
        """Get the mpp of the model."""
        return self.target_mpp

    def get_postprocess_fn(self) -> Callable:
        """Get the postprocessing function."""
        return self.postprocessor.postprocess

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

        # Only select the nuclei probability channel
        outputs["np"] = torch.softmax(outputs["np"], dim=1)[:, 1]
        outputs["tp"] = torch.argmax(outputs["tp"], dim=1, keepdim=False)

        outputs["hv"] = outputs["hv"].to(torch.float16)
        outputs["np"] = outputs["np"].to(torch.float16)
        outputs["tp"] = outputs["tp"].to(torch.uint8)

        return outputs
