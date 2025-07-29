"""Wrapper around classic-algos.

This is needed to extract features at different blocks of the Vision Transformer.

ruff: disable=N801
"""

from __future__ import annotations

import timm
from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer

from .ops import SwiGLUFFNFused


def module_fn_vit_base_s14(out_indices: list[int]) -> TimmVisionTransformer:
    """Load weights for a ViT Base features extractor and patch size 14.

    Using ``timm`` library which incorporates the latest
    developments on Vision Transformers.
    """
    _args = {
        "model_name": "vit_base_patch14_reg4_dinov2.lvd142m",
        "img_size": 224,
        "patch_size": 14,
        "init_values": 1e-5,
        "num_classes": 0,
        "dynamic_img_size": True,  # timm automatically interpolates pos encoding
        "mlp_ratio": 4,
        "mlp_layer": SwiGLUFFNFused,
        "features_only": True,  # A call returns all the features
        "out_indices": out_indices,
    }
    module = timm.create_model(**_args)  # type: ignore
    return module
