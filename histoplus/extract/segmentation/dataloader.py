"""Dataloader for tile extraction."""

import numpy as np
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from torch.utils.data import DataLoader

from histoplus.helpers.segmentor import Segmentor
from histoplus.helpers.tiling.tiles_map import TilesMap


def get_tile_dataloader(
    slide: OpenSlide,
    deepzoom: DeepZoomGenerator,
    segmentor: Segmentor,
    coords: np.ndarray,
    level: int,
    n_workers: int,
    batch_size: int,
) -> DataLoader:
    """Get tile dataloader."""
    tile_map = TilesMap(
        slide,
        deepzoom,
        tissue_coords=coords,
        tiles_level=level,
        tiles_size=segmentor.inference_image_size,
        n_samples=None,
        random_sampling=False,
        transform=segmentor.transform,
        metadata=False,
        float_coords=True,
    )

    dataloader = DataLoader(
        tile_map,
        num_workers=n_workers,
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,
    )

    return dataloader
