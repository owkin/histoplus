"""Implementation of CellViT."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

import torch
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys

from histoplus.helpers.nn.cellvit.decoder import (
    CellViTNeck,
    DecoderBranch,
)
from histoplus.helpers.nn.extractor import TimmExtractor, get_extractor
from histoplus.helpers.nn.utils import PretrainedBackboneWeight


class CellViTBase(nn.Module, ABC):
    """Abstract base class for CellViT models.

    Parameters
    ----------
    class_mapping : dict[int, str]
        Mapping from class identifier to class names.
    mpp : float
        MPP used for training.
    output_layers : list[int]
        List of layer indices used to extract the encoder layers.
    backbone_weights_pretraining: PretrainedBackboneWeight
        The original backbone weights used.
    train_image_size : int
        Image size used during training.
    inference_image_size : int
        Image size used during inference.
    """

    def __init__(
        self,
        class_mapping: dict[int, str],
        mpp: float,
        output_layers: list[int],
        backbone_weights_pretraining: PretrainedBackboneWeight,
        train_image_size: int,
        inference_image_size: int,
    ) -> None:
        super().__init__()

        self.mpp = mpp
        self.train_image_size = train_image_size
        self.inference_image_size = inference_image_size
        self.class_mapping = class_mapping

        self.backbone = get_extractor(
            backbone_weights_pretraining,
            encoder_keys=[f"encoder_layer_{i}" for i in output_layers],
            tile_size=train_image_size,
        )

        self.patch_size = self.backbone.patch_size

        self.embed_dim = 768
        self.skip_dim_1 = 512
        self.skip_dim_2 = 256
        self.bottleneck_dim = 512

        self.neck = CellViTNeck(
            embed_dim=self.embed_dim,
            skip_dim_1=self.skip_dim_1,
            skip_dim_2=self.skip_dim_2,
            bottleneck_dim=self.bottleneck_dim,
        )

    def _extract_features(self, x: torch.Tensor) -> tuple[list[torch.Tensor], Any]:
        """Extract features from the backbone and neck."""
        feature_maps = self.backbone(x)
        assert len(feature_maps) == 4
        z = [x, *list(feature_maps.values())]
        n = self.neck(z[:-1])
        return z, n

    @abstractmethod
    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """Do forward pass.

        feats is a dictionary of features at different depths with the spatial
        dimensions already restored.
        feats = {"encoder_layer_2": z1, "encoder_layer_5": z2, "encoder_layer_8": z3,
        "encoder_layer_11": z4, "output": output}

        output = output of the last layer of the backbone : [B, N+1, emb_dim]

        z = [z0, z1, z2, z3, z4]
        z0 : [B, C, H, W]
        zi : [B, emb_dim, hp, wp], i = 1, 2, 3, 4
        - Apply the neck to the features

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, C, H, W]

        Returns
        -------
        OrderedDict[str, torch.Tensor]
            Dictionary containing the output tensors for different branches.
        """
        raise NotImplementedError


class CellViT(CellViTBase):
    """CellViT model for cell detection and classification."""

    def __init__(
        self,
        cell_type_mapping: dict[int, str],
        mpp: float,
        output_layers: list[int],
        train_image_size: int,
        inference_image_size: int,
        backbone_weights_pretraining: PretrainedBackboneWeight = "aquavit_105k",
    ):
        super().__init__(
            cell_type_mapping,
            mpp,
            output_layers,
            backbone_weights_pretraining,
            train_image_size,
            inference_image_size,
        )

        self.cell_type_mapping = cell_type_mapping
        self.number_cell_types = len(cell_type_mapping)

        self.np_branch = DecoderBranch(
            num_classes=2,
            embed_dim=self.embed_dim,
            bottleneck_dim=self.bottleneck_dim,
            image_size=self.inference_image_size,
            patch_size=self.patch_size,
        )

        self.hv_branch = DecoderBranch(
            num_classes=2,
            embed_dim=self.embed_dim,
            bottleneck_dim=self.bottleneck_dim,
            image_size=self.inference_image_size,
            patch_size=self.patch_size,
        )

        self.tp_branch = DecoderBranch(
            num_classes=self.number_cell_types,
            embed_dim=self.embed_dim,
            bottleneck_dim=self.bottleneck_dim,
            image_size=self.inference_image_size,
            patch_size=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """Do forward pass for cell detection and classification.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, C, H, W].

        Returns
        -------
        OrderedDict[str, torch.Tensor]
            Dictionary containing the outputs of the different branches.
            - "np" : torch.Tensor : [B, 2, H, W]
            - "hv" : torch.Tensor : [B, 2, H, W]
            - "tp" : torch.Tensor : [B, number_cell_types, H, W]
        """
        z, n = self._extract_features(x)

        out_dict = OrderedDict()
        out_dict["np"] = self.np_branch(z[-1], n)
        out_dict["hv"] = self.hv_branch(z[-1], n)
        out_dict["tp"] = self.tp_branch(z[-1], n)

        return out_dict

    def load_state_dict(
        self, state_dict: Any, strict: bool = True, assign: bool = False
    ) -> _IncompatibleKeys:
        """Load state dict with backward compatibility for old TimmExtractor structure.

        Parameters
        ----------
        state_dict : Any
            The state dictionary to load.
        strict : bool, optional
            Whether to strictly enforce that the keys in `state_dict` match the keys
            returned by this module's `state_dict()` function, by default True.
        assign : bool, optional
            Whether to assign the state_dict to the module, by default False.

        Returns
        -------
        _IncompatibleKeys
            A named tuple with two attributes: `missing_keys` and `unexpected_keys`.
            `missing_keys` contains the keys that are in the module's state_dict but not in `state_dict`.
            `unexpected_keys` contains the keys that are in `state_dict` but not in the module's state_dict.
        """
        if not isinstance(self.backbone, TimmExtractor):
            return super().load_state_dict(state_dict, strict, assign)
        new_state_dict = TimmExtractor.remap_checkpoint_keys(state_dict)
        return super().load_state_dict(new_state_dict, strict, assign)
