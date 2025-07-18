"""Project cells from a local coordinate system to a global one."""

from typing import Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from histoplus.helpers.types import BorderInfo, GlobalSegmentedCell, TilePrediction


def project_all_cells_to_global_coordinate_system(
    tile_predictions: list[TilePrediction],
    inference_tile_coords: np.ndarray,
    inference_tile_size: int,
    inference_tile_overlap: int,
    verbose: int = 1,
) -> list[GlobalSegmentedCell]:
    """Project all cells from all tiles to global coordinate system."""
    projected_cells = []

    for tile_prediction, tile_coord in tqdm(
        zip(tile_predictions, inference_tile_coords, strict=False),
        total=len(tile_predictions),
        desc="Projecting cells to global coordinate system",
        leave=False,
        disable=bool(verbose == 0),
    ):
        global_cells = _project_cell_instances_to_global_coordinate_system(
            tile_prediction=tile_prediction,
            inference_tile_coords=tile_coord,
            inference_tile_size=inference_tile_size,
            inference_tile_overlap=inference_tile_overlap,
        )
        projected_cells.extend(global_cells)

    return projected_cells


def _project_cell_instances_to_global_coordinate_system(
    tile_prediction: TilePrediction,
    inference_tile_coords: np.ndarray,
    inference_tile_size: int,
    inference_tile_overlap: int,
) -> list[GlobalSegmentedCell]:
    """Project cell from a local coordinate system (tile-based) to a global (slide-based).

    Parameters
    ----------
    tile_predictions : TilePrediction
        Tile information.
    inference_tile_coords : np.ndarray, shape [2]
        Tile coordinates (without overlap). Typically these are tiles of shape 896.
    inference_tile_size : int
        Tile size used for inference. Typically 1024.
    inference_tile_overlap : float
        Amount of overlapping (in pixels) between patches. Typically 64.

    Returns
    -------
    list[GlobalSegmentedCell]
        Cell instances in the global coordinate system.
    """
    n_cells = len(tile_prediction.centroids)

    # Calculate global offset
    col, row = inference_tile_coords[0], inference_tile_coords[1]
    offset_global = _calculate_global_offset(
        col,
        row,
        inference_tile_size,
        inference_tile_overlap,
    )

    return [
        _project_single_cell(
            cell_idx=idx,
            tile_prediction=tile_prediction,
            tile_coords=inference_tile_coords,
            offset_global=offset_global,
            tile_size=inference_tile_size,
            tile_overlap=inference_tile_overlap,
        )
        for idx in range(n_cells)
    ]


def _project_single_cell(
    cell_idx: int,
    tile_prediction: TilePrediction,
    tile_coords: np.ndarray,
    offset_global: np.ndarray,
    tile_size: int,
    tile_overlap: int,
) -> GlobalSegmentedCell:
    """Project a single cell to global coordinates."""
    local_contour = tile_prediction.contours[cell_idx]
    local_centroid = tile_prediction.centroids[cell_idx]
    local_bounding_box = tile_prediction.bounding_boxes[cell_idx]

    # Project to global coordinate system
    global_centroid = np.rint(local_centroid + offset_global)
    global_contour = local_contour + offset_global

    # [top, down, left, right] -> [y, y, x, x] + [x, y]
    box_offset_global = np.array(
        [offset_global[1], offset_global[1], offset_global[0], offset_global[0]]
    )
    global_bounding_box = local_bounding_box + box_offset_global

    # Compute border information
    col, row = tile_coords[0], tile_coords[1]
    border_tiles = _get_border_tile(local_bounding_box, tile_size, row, col)
    border_information = BorderInfo(tiles=border_tiles)

    return GlobalSegmentedCell(
        contour=global_contour,
        centroid=global_centroid,
        bounding_box=global_bounding_box,
        cell_type=tile_prediction.cell_types[cell_idx],
        cell_type_probability=tile_prediction.cell_type_probabilities[cell_idx],
        in_safe_area=_is_safe_cell(local_bounding_box, tile_size, tile_overlap),
        is_touching_border=_is_touching_border(local_bounding_box, tile_size),
        offset_global=offset_global,
        tile_coordinates=tile_coords,
        border_information=border_information,
    )


def _calculate_global_offset(
    col: int,
    row: int,
    tile_size: int,
    tile_overlap: int,
) -> np.ndarray:
    """Calculate global offset for a tile."""
    tile_size_no_overlap = tile_size - 2 * tile_overlap

    offset_col = int(col * tile_size_no_overlap - tile_overlap)
    offset_row = int(row * tile_size_no_overlap - tile_overlap)

    return np.array([offset_col, offset_row])


def _is_safe_cell(
    bounding_box: Union[np.ndarray, list[float]], tile_size: int, tile_overlap: int
) -> bool:
    """Check if the cell is within the tile boundaries."""
    return bool(
        np.min(bounding_box) >= tile_overlap
        and np.max(bounding_box) <= tile_size - tile_overlap
    )


def _is_touching_border(
    bounding_box: Union[np.ndarray, list[float]], tile_size: int
) -> bool:
    """Check that the cell size is on the border."""
    return bool(np.max(bounding_box) == tile_size or np.min(bounding_box) == 0)


def _get_cell_position(bb: list[float], tile_size: int = 1024) -> list[int]:
    """Get cell position as a list.

    Notes
    -----
    Entry is 1, if cell touches the border: [top, down, left, right].
    """
    assert bb[0] <= bb[1]
    assert bb[2] <= bb[3]
    return [
        int(bb[0] == 0),  # Top
        int(bb[1] == tile_size),  # Down
        int(bb[2] == 0),  # Left
        int(bb[3] == tile_size),  # Right
    ]


def _get_border_tile(  # noqa: PLR0911
    bounding_box: list[float], tile_size: int, row: int, col: int
) -> Optional[list[list[int]]]:
    """Get the border tiles of a cell located at the border."""
    position = _get_cell_position(bounding_box, tile_size)

    if position == [1, 0, 0, 0]:  # top
        return [[col, row - 1]]
    if position == [1, 0, 0, 1]:  # top and right
        return [[col, row - 1], [col + 1, row - 1], [col + 1, row]]
    if position == [0, 0, 0, 1]:  # right
        return [[col + 1, row]]
    if position == [0, 1, 0, 1]:  # right and down
        return [[col + 1, row], [col + 1, row + 1], [col, row + 1]]
    if position == [0, 1, 0, 0]:  # down
        return [[col, row + 1]]
    if position == [0, 1, 1, 0]:  # down and left
        return [[col, row + 1], [col - 1, row + 1], [col - 1, row]]
    if position == [0, 0, 1, 0]:  # left
        return [[col - 1, row]]
    if position == [1, 0, 1, 0]:  # left and top
        return [[col - 1, row], [col - 1, row - 1], [col, row - 1]]
    return None


def assign_cells_back_to_original_tiling(
    cell_df: pd.DataFrame,
    tile_size: int,
    coords: np.ndarray,
    verbose: int = 1,
) -> list[TilePrediction]:
    """Assign cells back to the original tiling.

    The input dataframe contains information about cells that have been postprocessed
    (doublons and overlapping cells have been removed). These cells have been projected
    in a slide-wise coordinate system at some resolution level (usually MPP 0.5 or
    MPP 0.25). Now, they need to be assigned back to their original tiles, that is the
    tiling that has been originally provided by the user when launching an extraction.
    It is usually the tiling output by the TilingTool at MPP 0.5 with tiles of size 224.

    Parameters
    ----------
    cell_df : pd.DataFrame
        Cell predictions after cleaning.
    tile_size : int
        Tile size provided by the user and used to extract tiles in the TilingTool.
        Typically 224.
    coords : np.ndarray
        Tile coordinates as provided by the user.
    verbose : int
        Verbosity level.

    Returns
    -------
    final_predictions : list[TilePrediction]
        Predictions assigned to the user-provided tiles.
    """
    relevant_cols = [
        "contour",
        "centroid",
        "cell_type",
        "cell_type_probability",
        "bounding_box",
    ]
    cell_df = cell_df[relevant_cols].copy()

    # Parse the centroid coordinates
    cell_df.loc[:, "centroid_x"] = cell_df["centroid"].apply(lambda x: x[0])
    cell_df.loc[:, "centroid_y"] = cell_df["centroid"].apply(lambda x: x[1])

    # Re-compute the (col, row) tuple of the original tile
    cell_df.loc[:, "original_tile_col"] = (cell_df["centroid_x"] // tile_size).astype(
        int
    )
    cell_df.loc[:, "original_tile_row"] = (cell_df["centroid_y"] // tile_size).astype(
        int
    )

    # Compute offset for each original tile
    cell_df.loc[:, "tile_offset_x"] = (cell_df["original_tile_col"] * tile_size).astype(
        int
    )
    cell_df.loc[:, "tile_offset_y"] = (cell_df["original_tile_row"] * tile_size).astype(
        int
    )

    # Compute the cell contours in the tile coordinate system
    cell_df.loc[:, "contour_in_tile_coord_system"] = cell_df.apply(
        lambda row: np.array(
            [
                row["contour"][:, 0] - row["tile_offset_x"],  # x - x_offset
                row["contour"][:, 1] - row["tile_offset_y"],  # y - y_offset
            ]
        ).T,
        axis=1,
    )

    # Compute the cell centroids in the tile coordinate system
    cell_df.loc[:, "centroid_in_tile_coord_system"] = cell_df.apply(
        lambda row: np.array(
            [
                row["centroid"][0] - row["tile_offset_x"],  # x - x_offset
                row["centroid"][1] - row["tile_offset_y"],  # y - y_offset
            ]
        ),
        axis=1,
    )

    # Compute the bounding boxes in the tile coordinate system
    cell_df.loc[:, "bounding_box_in_tile_coord_system"] = cell_df.apply(
        lambda row: np.array(
            [
                row["bounding_box"][0] - row["tile_offset_y"],  # top - y_offset
                row["bounding_box"][1] - row["tile_offset_y"],  # down - y_offset
                row["bounding_box"][2] - row["tile_offset_x"],  # left - x_offset
                row["bounding_box"][3] - row["tile_offset_x"],  # right - x_offset
            ]
        ),
        axis=1,
    )

    result = []

    for tile_coord in tqdm(
        coords,
        desc="Assigning cells to original tiles",
        leave=False,
        disable=bool(verbose == 0),
    ):
        col, row = tile_coord[0], tile_coord[1]

        current_df = cell_df[
            (cell_df["original_tile_col"] == col)
            & (cell_df["original_tile_row"] == row)
        ]

        result.append(
            TilePrediction(
                contours=current_df["contour_in_tile_coord_system"].to_list(),
                bounding_boxes=current_df[
                    "bounding_box_in_tile_coord_system"
                ].to_list(),
                centroids=current_df["centroid_in_tile_coord_system"].to_list(),
                cell_types=current_df["cell_type"].to_list(),
                cell_type_probabilities=current_df["cell_type_probability"].to_list(),
            )
        )

    return result
