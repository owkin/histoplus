"""Check performed to validate the parameters passed to the extract function."""

import numpy as np
import pandas as pd
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator

from histoplus.helpers.segmentor.base import Segmentor
from histoplus.helpers.tiling.new_tiling import (
    get_new_tiling_for_target_tile_size_and_deepzoom_level,
)
from histoplus.helpers.tiling.optimal_mpp import get_tiling_slide_level


def get_tile_coordinates_and_deepzoom_for_segmentor(
    slide: OpenSlide,
    features: np.ndarray,
    segmentor: Segmentor,
    original_tile_size: int,
    inference_tile_overlap: int,
    verbose: int,
) -> tuple[np.ndarray, DeepZoomGenerator, int, int]:
    """Get tile coordinates and the DeepZoom object of the slide expected by the segmentor.

    This is used to create the tiling at the expected MPP and with the expected tile size
    of the segmentor.

    Parameters
    ----------
    slide : OpenSlide
        The slide object.

    features : np.ndarray
        Features from the TilingTool.

    segmentor : Segmentor
        Segmentor object.

    original_tile_size : int
        Original tile size.

    inference_tile_overlap : int
        Overlap (horizontal and vertical) between two consecutive tiles on the grid.

    Returns
    -------
    np.ndarray
        New tile coordinates.

    DeepZoom
        Associated DeepZoom object to the slide.

    int
        DeepZoom level of the features provided by the user. Most likely the level at
        MPP 0.5, used by the TilingTool.

    int
        DeepZoom level associated to the target MPP of the segmentor. If the segmentor
        predicts cells at MPP 0.25, this is the corresponding level.
    """
    coords = features[:, 1:3]
    original_deepzoom_level = int(features[0, 0])

    inference_tile_size_without_overlap = (
        segmentor.inference_image_size - 2 * inference_tile_overlap
    )

    deepzoom = DeepZoomGenerator(
        slide,
        inference_tile_size_without_overlap,
        overlap=inference_tile_overlap,
        limit_bounds=False,
    )

    target_deepzoom_level = get_tiling_slide_level(
        slide,
        deepzoom,
        mpp=segmentor.target_mpp,
        verbose=verbose,
    )

    new_coords = get_new_tiling_for_target_tile_size_and_deepzoom_level(
        coords=coords,
        original_tile_size=original_tile_size,
        original_deepzoom_level=original_deepzoom_level,
        target_tile_size=inference_tile_size_without_overlap,
        target_deepzoom_level=target_deepzoom_level,
    )

    return new_coords, deepzoom, original_deepzoom_level, target_deepzoom_level

