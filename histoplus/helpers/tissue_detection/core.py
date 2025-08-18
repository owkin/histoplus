"""Detect tissue using Otsu's threshold."""

import math

import numpy as np
from loguru import logger
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

from histoplus.helpers.tiling.optimal_mpp import get_tiling_slide_level


def _apply_otsu_threshold(
    slide: OpenSlide, mask_width: int, mask_height: int
) -> np.ndarray:
    thumbnail = slide.get_thumbnail((mask_width, mask_height))
    img_arr = np.array(rgb2gray(thumbnail))
    thresh = threshold_otsu(img_arr)
    return (img_arr > thresh).astype(np.uint8)


def detect_tissue_on_wsi(
    slide: OpenSlide,
    matter_threshold: float = 0.2,
    target_mpp: float = 8.0,
    base_mpp: float = 0.5,
    tile_size_at_base_mpp: int = 224,
    default_mpp_max: float = 0.23,
    mpp_rtol: float = 0.2,
) -> tuple[np.ndarray, int]:
    """Detect tissue on WSI using Otsu's thresholding."""
    downsample_factor = target_mpp / base_mpp
    tile_size_at_target_mpp = int(tile_size_at_base_mpp / downsample_factor)

    deepzoom = DeepZoomGenerator(slide, tile_size=tile_size_at_base_mpp, overlap=0)

    dz_level_at_target_mpp = get_tiling_slide_level(
        slide,
        deepzoom,
        mpp=target_mpp,
        default_mpp_max=default_mpp_max,
        mpp_rtol=mpp_rtol,
    )

    dz_level_at_base_mpp = get_tiling_slide_level(
        slide,
        deepzoom,
        mpp=base_mpp,
        default_mpp_max=default_mpp_max,
        mpp_rtol=mpp_rtol,
    )

    width_at_target_mpp, height_at_target_mpp = deepzoom.level_dimensions[
        dz_level_at_target_mpp
    ]

    mask = _apply_otsu_threshold(slide, width_at_target_mpp, height_at_target_mpp)

    num_cols = math.floor(width_at_target_mpp / tile_size_at_target_mpp)
    num_rows = math.floor(height_at_target_mpp / tile_size_at_target_mpp)

    # Crop mask to make it reshapable (fix the math.floor)
    mask_cropped = mask[
        : num_rows * tile_size_at_target_mpp, : num_cols * tile_size_at_target_mpp
    ]

    tiles = mask_cropped.reshape(
        num_rows, tile_size_at_target_mpp, num_cols, tile_size_at_target_mpp
    )
    tissue_scores = tiles.mean(axis=(1, 3))

    tissue_coords = np.array(np.nonzero(tissue_scores < matter_threshold))
    tissue_coords = np.transpose(tissue_coords).astype(int)

    # Swapping columns to respect Openslide's conventions
    tissue_coords[:, [0, 1]] = tissue_coords[:, [1, 0]]

    logger.info(f"Found {len(tissue_coords)} tiles with Otsu's threshold.")

    return tissue_coords, dz_level_at_base_mpp
