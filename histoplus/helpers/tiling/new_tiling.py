"""Utils to create a new tiling with larger tiles."""

import numpy as np


def get_new_tiling_for_target_tile_size_and_deepzoom_level(
    coords: np.ndarray,
    original_tile_size: int,
    original_deepzoom_level: int,
    target_tile_size: int,
    target_deepzoom_level: int,
) -> np.ndarray:
    """Generate new tile coordinates for a target tile size at a given DeepZoom level.

    This is useful to transform tile coordinates representing tiles of shape (224, 224)
    at MPP 0.5, into tile coordinates representing tiles of shape (448, 448) at MPP 0.25.

    Parameters
    ----------
    coords : np.ndarray
        Tile coordinates given by the TilingTool.

    original_tile_size : int
        Tile size used during TilingTool's extraction. Typically 224.

    original_deepzoom_level : int
        DeepZoom level used during TilingTool's extraction. Typically the one corresponding
        to the MPP 0.5.

    target_tile_size : int
        Tile size of the new tiling.

    target_deepzoom_level : int
        DeepZoom level of the new tiling.

    Returns
    -------
    np.ndarray
        New tile coordinates.
    """
    target_tile_size_in_original_level = target_tile_size * 2 ** (
        original_deepzoom_level - target_deepzoom_level
    )
    return _get_large_tile_coords(
        coords,
        original_tile_size,
        target_tile_size_in_original_level,
    )


def _get_large_tile_coords(
    coords: np.ndarray,
    original_tile_size: int,
    target_tile_size: int,
) -> np.ndarray:
    """Generate coordinates of large tiles for inference.

    This function takes as input the coordinates of the tiles given by the user.

    To adapt to the input size expected by some of the cell segmentation model, this
    function can generate a coarser tiling of the slide, given some tile coordinates.

    If the expected tile size is different from 224, we use the coordinates given by the
    user and creates new coordinates of non-overlapping tiles covering the same tissue
    area. Essentially, this function creates a coarser tiling of the WSI at the same
    resolution level.

    Parameters
    ----------
    coords : np.ndarray
        Tile coordinates.

    original_tile_size : int
        Tile size used during TilingTool's extraction.

    target_tile_size : int
        Tile size of the new tiling.

    Returns
    -------
    np.ndarray
        Large tile coordinates at the same level of magnification.
    """
    if original_tile_size == target_tile_size:
        return coords

    # Get top-left coords of the tiles in the slide coordinate system
    slide_level_coords = coords * original_tile_size

    # Compute the top-left coordinates of the tiles forming the coarser tiling
    coarse_coords = (slide_level_coords // target_tile_size) * target_tile_size

    # Extract unique coordinates to form the new tiling
    unique_coarse_coords = np.unique(coarse_coords, axis=0)

    unique_coarse_coords = unique_coarse_coords // target_tile_size

    return unique_coarse_coords.astype(int)
