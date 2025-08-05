"""Utilities functions."""

from typing import Literal

import torch
from torch.nn import functional as F


PretrainedBackboneWeight = Literal["aquavit_105k"]


def interpolate_positional_encoding(
    pos_embedding: torch.Tensor | torch.nn.parameter.Parameter | None,
    embed_dim: int,
    old_dims: tuple[int, int],
    new_dims: tuple[int, int],
    patch_size: int,
    has_cls_token: bool,
    num_reg_token: int = 0,
    interpolate_offset: float = 0.1,
):
    """Interpolate positional encoding.

    Parameters
    ----------
    pos_embedding: torch.Tensor
        Original positional embedding.
    embed_dim: int
        Embedding dimension
    old_dims: tuple[int, int]
        Spatial dimensions of the input volume used during training.
    new_dims: tuple[int, int]
        Spatial dimensions of the input volume used during inference.
    patch_size: int
        Patch size.
    has_cls_token: bool
        Whether the positional encoding includes the [CLS] token.
    num_reg_token: int
        Number of register tokens in the positional encoding.
    interpolate_offset: float
        Offset used for interpolation.

    Returns
    -------
    torch.Tensor
        Interpolated positional encoding (with first position being the [CLS] token if
        it exists).
    """
    assert pos_embedding is not None

    if num_reg_token > 0 and not has_cls_token:
        raise Exception(
            "If register tokens are found in the positional encodings, the [CLS] token "
            "should also be there."
        )

    if old_dims == new_dims:
        return pos_embedding

    def _extract_position_embeddings(pos_embedding: torch.Tensor):
        # Extract CLS token position embedding if present
        class_pos_emb = None
        if has_cls_token:
            class_pos_emb = pos_embedding[:, 0]

        # Extract register token position embeddings if present
        reg_pos_emb = None
        if num_reg_token > 0:
            reg_pos_emb = pos_embedding[:, 1 : 1 + num_reg_token]

        # Extract patch position embeddings
        start_idx = 1 + num_reg_token if has_cls_token else 0
        patch_pos_emb = pos_embedding[:, start_idx:]

        return class_pos_emb, reg_pos_emb, patch_pos_emb

    class_pos_emb, reg_pos_emb, patch_pos_emb = _extract_position_embeddings(
        pos_embedding
    )

    old_w, old_h = old_dims
    new_w, new_h = new_dims

    old_wp, old_hp = old_w // patch_size, old_h // patch_size
    new_wp, new_hp = new_w // patch_size, new_h // patch_size

    # We add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    new_wp_off, new_hp_off = new_wp + interpolate_offset, new_hp + interpolate_offset

    interp_pos_emb = F.interpolate(
        patch_pos_emb.reshape(1, old_wp, old_hp, embed_dim).permute(0, 3, 1, 2),
        scale_factor=(new_wp_off / old_wp, new_hp_off / old_hp),
        mode="bicubic",
        antialias=False,
    )

    assert interp_pos_emb.shape[-2] == new_wp
    assert interp_pos_emb.shape[-1] == new_hp

    interp_pos_emb = interp_pos_emb.permute(0, 2, 3, 1).view(1, -1, embed_dim)

    new_pos_emb = interp_pos_emb

    if has_cls_token:
        assert class_pos_emb is not None  # for typing
        if num_reg_token > 0:
            new_pos_emb = torch.cat(
                (class_pos_emb[None], reg_pos_emb, interp_pos_emb), dim=1
            )
        else:
            new_pos_emb = torch.cat((class_pos_emb[None], interp_pos_emb), dim=1)

    new_pos_emb = new_pos_emb.to(pos_embedding.dtype)

    return new_pos_emb
