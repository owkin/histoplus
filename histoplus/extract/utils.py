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


def rescale_cell_mask_coordinates_to_original_resolution(
    cell_df: pd.DataFrame,
    original_dz_level: int,
    extraction_dz_level: int,
) -> pd.DataFrame:
    """Rescale cell mask coordinates to fit into a tile of original tile size.

    If the inference of cell masks is made on tiles of different size (e.g. 448) as the
    one given by the user (e.g. 224), then the cell mask coordinates are lying in
    different coordinate systems. Therefore, we need to rescale the cell mask
    coordinates.

    Parameters
    ----------
    cell_df : list[pd.DataFrame]
        Collection of cells.

    original_dz_level : int
        DeepZoom level used provided by the user.

    extraction_dz_level: int
        DeepZoom level used to extract cell masks.

    Returns
    -------
    list[TilePrediction]
        Rescaled mask coordinates.
    """
    if original_dz_level == extraction_dz_level:
        return cell_df

    scale_ratio = 2 ** (original_dz_level - extraction_dz_level)

    def _scale(x):
        return x * scale_ratio

    cell_df.loc[:, "contour"] = cell_df["contour"].apply(_scale)
    cell_df.loc[:, "centroid"] = cell_df["centroid"].apply(_scale)
    cell_df.loc[:, "bounding_box"] = cell_df["bounding_box"].apply(_scale)

    return cell_df
