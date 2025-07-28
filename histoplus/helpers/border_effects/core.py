"""Border Effect utilities."""

import numpy as np

from histoplus.helpers.types import TilePrediction

from .overlap import remove_overlapping_cells
from .project import (
    assign_cells_back_to_original_tiling,
    project_all_cells_to_global_coordinate_system,
)
from .utils import rescale_cell_mask_coordinates_to_original_resolution


def fix_border_effects(
    tile_predictions: list[TilePrediction],
    tile_coords: np.ndarray,
    tile_size: int,
    inference_tile_coords: np.ndarray,
    inference_tile_size: int,
    inference_tile_overlap: int,
    original_dz_level: int,
    extraction_dz_level: int,
    intersection_threshold: float = 0.01,
    verbose: int = 1,
) -> list[TilePrediction]:
    """Fix border effects. Entry point.

    During inference, cells are detected using overlapping image patches, which can
    result in duplicate detections of the same cell. This function eliminates these
    duplicates by identifying and removing overlapping cell instances.

    Note
    ----
    By convention, we name coordinates in the tile coordinate system local coordinates,
    while coordinates in the coordinate system of the slide are called global
    coordinates.

    Parameters
    ----------
    tile_predictions : list[TilePrediction]
        A list of all the tile predictions.

    tile_coords : np.ndarray, shape (n_tiles, 2)
        Tile coordinates provided by the user, and from the TilingTool.

    tile_size : int
        Tile size used for tiling in the TilingTool.

    inference_tile_coords : np.ndarray
        Tile coordinates used for inference. These are typically coordinates of larger
        tiles. Typically of size 896.

    inference_tile_size : int
        Tile size used for inference. Typically this is 1024.

    inference_tile_overlap : int
        Tile overlap. Typically this is 64.

    original_dz_level : int
        DeepZoom level of the features provided by the user. Most likely the level at
        MPP 0.5, used by the TilingTool.

    extraction_dz_level : int
        DeepZoom level associated to the target MPP of the segmentor. If the segmentor
        predicts cells at MPP 0.25, this is the corresponding level.

    intersection_threshold : float
        Intersection threshold.

    verbose : int
        Verbosity level.
    """
    projected_cells = project_all_cells_to_global_coordinate_system(
        tile_predictions=tile_predictions,
        inference_tile_coords=inference_tile_coords,
        inference_tile_size=inference_tile_size,
        inference_tile_overlap=inference_tile_overlap,
        verbose=verbose,
    )

    clean_cells = remove_overlapping_cells(
        cell_list=projected_cells,
        intersection_threshold=intersection_threshold,
        verbose=verbose,
    )

    # If the cell mask extraction is made at a MPP different from the MPP of the coords
    # provided by the user (this is the case when using a segmentor trained at MPP 0.25
    # with coordinates extracted at MPP 0.5), the cell masks are at MPP 0.25. The
    # contours, bounding boxes and centroids must be downscaled to MPP 0.5.
    clean_cells = rescale_cell_mask_coordinates_to_original_resolution(
        cell_df=clean_cells,
        original_dz_level=original_dz_level,
        extraction_dz_level=extraction_dz_level,
    )

    final_predictions = assign_cells_back_to_original_tiling(
        clean_cells,
        coords=tile_coords,
        tile_size=tile_size,
        verbose=verbose,
    )

    return final_predictions
