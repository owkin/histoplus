"""Timm extractor."""

from collections import OrderedDict
from collections.abc import Sequence

import torch
from torch.nn.parameter import Parameter

from histoplus.helpers.constants import BIOPTIMUS_MEAN, BIOPTIMUS_STD
from histoplus.helpers.nn.utils import interpolate_positional_encoding
from histoplus.helpers.nn.vit import (
    module_fn_vit_base_s14,
)

from .base import Extractor


class TimmExtractor(Extractor):
    """Feature extractors models from timm library.

    Parameters
    ----------
    model_name: str
        Name of the model to load from timm.
        Default is Swin-Base'swin_base_patch4_window7_224.ms_in22k_ft_in1k'.
        ConvNeXt: 'convnextv2_base.fcmae_ft_in22k_in1k'.
    out_indices: Optional[Sequence[int]] = None
        Layers' indices to extract the features from.
        Default is model-dependent but typically is of size 4.
    mean: tuple[float, float, float]
        Mean for the transform operation.
        If not specified, taking the mean from the model config.
    std: tuple[float, float, float]
        Std for the transform operation.
        If not specified, taking the std from the model config.
    """

    def __init__(
        self,
        model: str = "base_s14",
        output_layers: Sequence[str] = (
            "encoder_layer_3",
            "encoder_layer_5",
            "encoder_layer_7",
            "encoder_layer_11",
        ),
        tile_size: int = 224,
        mean: tuple[float, float, float] = BIOPTIMUS_MEAN,
        std: tuple[float, float, float] = BIOPTIMUS_STD,
    ):
        super().__init__()

        assert model == "base_s14"

        self.mean = mean
        self.std = std

        # TODO (PAB): Since there is only one extractor supported for now, I hardcode
        # its hyperparameters.
        self.embed_dim = 768
        self.patch_size = 14
        self.has_cls_token = False

        self.output_layers = output_layers

        out_indices = [int(x.split("_")[-1]) for x in output_layers]
        self.feature_extractor = module_fn_vit_base_s14(out_indices)

        new_pos_embedding = interpolate_positional_encoding(
            pos_embedding=self.feature_extractor.model.pos_embed,
            embed_dim=self.embed_dim,
            old_dims=(224, 224),
            new_dims=(tile_size, tile_size),
            patch_size=self.patch_size,
            has_cls_token=self.has_cls_token,
        )

        assert tile_size % self.patch_size == 0
        self.feature_extractor.model.pos_embed = Parameter(new_pos_embedding)
        self.feature_extractor.model.patch_embed.grid_size = (
            tile_size // self.patch_size,
            tile_size // self.patch_size,
        )

    def __call__(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """Return features.

        Parameters
        ----------
        images: torch.Tensor
            input of size (n_tiles, n_channels, dim_x, dim_y, )
            Ideally `dim_x == dim_y == 224`.

        Returns
        -------
        features : torch.Tensor
            tensor of size (n_tiles, dim) where dim=384 or 768
            or 1024 if the model is respectively a ViT-S, a ViT-B or a ViT-L.
        """
        features_list = self.feature_extractor(images)
        return OrderedDict(dict(zip(self.output_layers, features_list, strict=False)))  # type: ignore

    @staticmethod
    def remap_checkpoint_keys(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Remap keys in the state_dict to match the expected model structure.

        The function handles the difference between:
        'backbone.feature_extractor.blocks.X.component' (current)
        'backbone.feature_extractor.model.blocks.X.component' (expected)

        Parameters
        ----------
        state_dict : dict[str, torch.Tensor]
            The original state dictionary with keys to be remapped

        Returns
        -------
        dict[str, torch.Tensor]
            A new state dictionary with remapped keys
        """
        new_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("backbone.feature_extractor."):
                # Special cases for top level components
                if key in [
                    "backbone.feature_extractor.cls_token",
                    "backbone.feature_extractor.reg_token",
                    "backbone.feature_extractor.pos_embed",
                ]:
                    new_key = key.replace(
                        "backbone.feature_extractor.",
                        "backbone.feature_extractor.model.",
                    )
                # Handle patch_embed component
                elif key.startswith("backbone.feature_extractor.patch_embed"):
                    new_key = key.replace(
                        "backbone.feature_extractor.patch_embed",
                        "backbone.feature_extractor.model.patch_embed",
                    )
                # Handle blocks components
                elif key.startswith("backbone.feature_extractor.blocks"):
                    new_key = key.replace(
                        "backbone.feature_extractor.blocks",
                        "backbone.feature_extractor.model.blocks",
                    )
                # Handle norm component
                elif key.startswith("backbone.feature_extractor.norm"):
                    # No need to handle them, since we are only extracting intermediate
                    # feature maps. This norm (weights and bias) is applied in the head
                    # of the network at the very end.
                    continue
                else:
                    new_key = key
            else:
                new_key = key
            new_state_dict[new_key] = value
        return new_state_dict
