"""Test for the project function."""

import numpy as np
import pandas as pd
import pytest

from histoplus.helpers.border_effects.project import (
    _calculate_global_offset,
    _get_border_tile,
    _get_cell_position,
    _is_safe_cell,
    _is_touching_border,
    _project_cell_instances_to_global_coordinate_system,
    _project_single_cell,
    assign_cells_back_to_original_tiling,
    project_all_cells_to_global_coordinate_system,
)
from histoplus.helpers.types import BorderInfo, GlobalSegmentedCell, TilePrediction


@pytest.fixture
def sample_tile_prediction():
    """Create a sample tile prediction with a few cells."""
    return TilePrediction(
        contours=[
            # Cell 1 - in safe area
            np.array([[110, 110], [120, 110], [120, 120], [110, 120]]),
            # Cell 2 - touching top border
            np.array([[500, 0], [520, 0], [520, 20], [500, 20]]),
            # Cell 3 - touching right border
            np.array([[1000, 500], [1024, 500], [1024, 520], [1000, 520]]),
        ],
        bounding_boxes=[
            # Cell 1 - in safe area
            [110, 110, 120, 120],
            # Cell 2 - touching top border
            [0, 500, 20, 520],
            # Cell 3 - touching right border
            [500, 1000, 520, 1024],
        ],
        centroids=[
            [115, 115],  # Cell 1
            [510, 10],  # Cell 2
            [510, 1010],  # Cell 3
        ],
        cell_types=["type_A", "type_B", "type_C"],
        cell_type_probabilities=[0.9, 0.8, 0.7],
    )


@pytest.fixture
def sample_tile_coords():
    """Create sample tile coordinates (2x2 grid)."""
    return np.array(
        [
            [0, 0],  # Tile 1
            [1, 0],  # Tile 2
            [0, 1],  # Tile 3
            [1, 1],  # Tile 4
        ]
    )


def test_calculate_global_offset():
    """Test calculation of global offset."""
    # Test case 1: origin tile (0,0)
    offset = _calculate_global_offset(col=0, row=0, tile_size=1024, tile_overlap=64)
    assert np.array_equal(offset, np.array([-64, -64]))

    # Test case 2: tile at (1,0)
    offset = _calculate_global_offset(col=1, row=0, tile_size=1024, tile_overlap=64)
    assert np.array_equal(offset, np.array([896 - 64, -64]))

    # Test case 3: tile at (0,1)
    offset = _calculate_global_offset(col=0, row=1, tile_size=1024, tile_overlap=64)
    assert np.array_equal(offset, np.array([-64, 896 - 64]))

    # Test case 4: tile at (2,3)
    offset = _calculate_global_offset(col=2, row=3, tile_size=1024, tile_overlap=64)
    assert np.array_equal(offset, np.array([2 * 896 - 64, 3 * 896 - 64]))


def test_is_safe_cell():
    """Test if a cell is within the safe area of a tile."""
    # Cell fully in safe area
    assert _is_safe_cell([100, 100, 200, 200], tile_size=1024, tile_overlap=64) is True

    # Cell touching left border
    assert _is_safe_cell([0, 100, 100, 200], tile_size=1024, tile_overlap=64) is False

    # Cell touching right border
    assert (
        _is_safe_cell([900, 100, 1024, 200], tile_size=1024, tile_overlap=64) is False
    )

    # Cell in overlap area but not at border
    assert _is_safe_cell([50, 50, 100, 100], tile_size=1024, tile_overlap=64) is False


def test_is_touching_border():
    """Test if a cell is touching the border of a tile."""
    # Cell not at border
    assert _is_touching_border([100, 100, 200, 200], tile_size=1024) is False

    # Cell at top border
    assert _is_touching_border([0, 100, 100, 200], tile_size=1024) is True

    # Cell at right border
    assert _is_touching_border([900, 100, 1024, 200], tile_size=1024) is True


def test_get_cell_position():
    """Test getting the position of a cell relative to tile borders."""
    tile_size = 1024

    # Cell not at any border
    assert _get_cell_position([100, 100, 200, 200], tile_size) == [0, 0, 0, 0]

    # Cell at top border
    assert _get_cell_position([0, 100, 100, 200], tile_size) == [1, 0, 0, 0]

    # Cell at bottom border
    assert _get_cell_position([100, tile_size, 100, 200], tile_size) == [0, 1, 0, 0]

    # Cell at left border
    assert _get_cell_position([100, 100, 0, 100], tile_size) == [0, 0, 1, 0]

    # Cell at right border
    assert _get_cell_position([100, 300, 200, tile_size], tile_size) == [0, 0, 0, 1]

    # Cell at top-right corner
    assert _get_cell_position([0, 100, 100, tile_size], tile_size) == [1, 0, 0, 1]


def test_get_border_tile():
    """Test getting bordering tiles for cells at the border."""
    tile_size = 1024

    # Cell at top border (row=2, col=3)
    bb = [0, 100, 100, 200]
    assert _get_border_tile(bb, tile_size, row=2, col=3) == [[3, 1]]

    # Cell at right border (row=2, col=3)
    bb = [100, 100, 200, tile_size]
    assert _get_border_tile(bb, tile_size, row=2, col=3) == [[4, 2]]

    # Cell at bottom border (row=2, col=3)
    bb = [100, tile_size, 100, 200]
    assert _get_border_tile(bb, tile_size, row=2, col=3) == [[3, 3]]

    # Cell at left border (row=2, col=3)
    bb = [100, 200, 0, 100]
    assert _get_border_tile(bb, tile_size, row=2, col=3) == [[2, 2]]

    # Cell at top-right corner (row=2, col=3)
    bb = [0, 100, 100, tile_size]
    assert _get_border_tile(bb, tile_size, row=2, col=3) == [[3, 1], [4, 1], [4, 2]]

    # Cell not at border (should return None)
    bb = [100, 100, 200, 200]
    assert _get_border_tile(bb, tile_size, row=2, col=3) is None


def test_project_single_cell(sample_tile_prediction):
    """Test projecting a single cell to global coordinates."""
    # Test projecting a cell from a tile at position (1, 2)
    tile_coords = np.array([1, 2])
    tile_size = 1024
    tile_overlap = 64

    # Calculate expected global offset
    expected_offset = _calculate_global_offset(
        col=tile_coords[0],
        row=tile_coords[1],
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )

    # Project the first cell (safe area)
    cell_idx = 0
    global_cell = _project_single_cell(
        cell_idx=cell_idx,
        tile_prediction=sample_tile_prediction,
        tile_coords=tile_coords,
        offset_global=expected_offset,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )

    # Verify the projection
    assert isinstance(global_cell, GlobalSegmentedCell)
    assert np.array_equal(
        global_cell.contour, sample_tile_prediction.contours[cell_idx] + expected_offset
    )
    assert np.array_equal(
        global_cell.centroid,
        np.rint(sample_tile_prediction.centroids[cell_idx] + expected_offset),
    )
    assert global_cell.cell_type == sample_tile_prediction.cell_types[cell_idx]
    assert (
        global_cell.cell_type_probability
        == sample_tile_prediction.cell_type_probabilities[cell_idx]
    )
    assert global_cell.in_safe_area is True
    assert global_cell.is_touching_border is False
    assert np.array_equal(global_cell.offset_global, expected_offset)
    assert np.array_equal(global_cell.tile_coordinates, tile_coords)
    assert global_cell.border_information.tiles is None


def test_project_cell_instances(sample_tile_prediction):
    """Test projecting all cells from a tile to global coordinates."""
    # Test with a tile at (1, 2)
    tile_coords = np.array([1, 2])
    tile_size = 1024
    tile_overlap = 64

    # Project all cells from the tile
    global_cells = _project_cell_instances_to_global_coordinate_system(
        tile_prediction=sample_tile_prediction,
        inference_tile_coords=tile_coords,
        inference_tile_size=tile_size,
        inference_tile_overlap=tile_overlap,
    )

    # Verify the results
    assert len(global_cells) == len(sample_tile_prediction.centroids)
    assert all(isinstance(cell, GlobalSegmentedCell) for cell in global_cells)

    # Calculate expected global offset
    expected_offset = _calculate_global_offset(
        col=tile_coords[0],
        row=tile_coords[1],
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )

    # Check the first cell (detailed check)
    first_cell = global_cells[0]
    assert np.array_equal(
        first_cell.contour, sample_tile_prediction.contours[0] + expected_offset
    )
    assert np.array_equal(
        first_cell.centroid,
        np.rint(sample_tile_prediction.centroids[0] + expected_offset),
    )
    assert np.array_equal(first_cell.offset_global, expected_offset)

    # Check if border information is correctly set for all cells
    for i, cell in enumerate(global_cells):
        assert cell.cell_type == sample_tile_prediction.cell_types[i]
        assert (
            cell.cell_type_probability
            == sample_tile_prediction.cell_type_probabilities[i]
        )
        assert isinstance(cell.border_information, BorderInfo)


def test_project_all_cells(sample_tile_prediction, sample_tile_coords):
    """Test projecting cells from multiple tiles."""
    # Create a list of tile predictions for each tile coord
    tile_predictions = [sample_tile_prediction] * len(sample_tile_coords)
    tile_size = 1024
    tile_overlap = 64

    # Project all cells from all tiles
    all_global_cells = project_all_cells_to_global_coordinate_system(
        tile_predictions=tile_predictions,
        inference_tile_coords=sample_tile_coords,
        inference_tile_size=tile_size,
        inference_tile_overlap=tile_overlap,
    )

    # Verify the results
    # Each tile has 3 cells, we have 4 tiles
    expected_cell_count = len(sample_tile_prediction.centroids) * len(
        sample_tile_coords
    )
    assert len(all_global_cells) == expected_cell_count

    # Check that all cells are properly projected
    assert all(isinstance(cell, GlobalSegmentedCell) for cell in all_global_cells)

    # Spot check: verify cells from the first tile
    first_tile_cells = all_global_cells[: len(sample_tile_prediction.centroids)]
    first_tile_coords = sample_tile_coords[0]

    expected_offset = _calculate_global_offset(
        col=first_tile_coords[0],
        row=first_tile_coords[1],
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )

    # Check the first cell from the first tile
    first_cell = first_tile_cells[0]
    assert np.array_equal(
        first_cell.contour, sample_tile_prediction.contours[0] + expected_offset
    )
    assert np.array_equal(
        first_cell.centroid,
        np.rint(sample_tile_prediction.centroids[0] + expected_offset),
    )
    assert np.array_equal(first_cell.offset_global, expected_offset)
    assert np.array_equal(first_cell.tile_coordinates, first_tile_coords)


def test_assign_cells_back_to_original_tiling():
    """Test assigning cells back to their original tiles."""
    # Create a sample dataframe of cells
    cell_data = {
        "contour": [
            np.array([[230, 230], [250, 230], [250, 250], [230, 250]]),  # Tile (1, 1)
            np.array([[450, 230], [470, 230], [470, 250], [450, 250]]),  # Tile (2, 1)
            np.array([[230, 450], [250, 450], [250, 470], [230, 470]]),  # Tile (1, 2)
        ],
        "centroid": [
            np.array([240, 240]),  # Tile (1, 1)
            np.array([460, 240]),  # Tile (2, 1)
            np.array([240, 460]),  # Tile (1, 2)
        ],
        "bounding_box": [
            np.array([230, 250, 230, 250]),  # Tile (1, 1)
            np.array([230, 250, 450, 470]),  # Tile (2, 1)
            np.array([450, 470, 230, 250]),  # Tile (1, 2)
        ],
        "cell_type": ["type_A", "type_B", "type_C"],
        "cell_type_probability": [0.9, 0.8, 0.7],
    }
    cell_df = pd.DataFrame(cell_data)

    # Define tile size and tile coordinates
    tile_size = 224
    coords = np.array(
        [
            [1, 1],  # Tile (1, 1)
            [2, 1],  # Tile (2, 1)
            [1, 2],  # Tile (1, 2)
        ]
    )

    # Assign cells back to original tiles
    result = assign_cells_back_to_original_tiling(
        cell_df=cell_df,
        tile_size=tile_size,
        coords=coords,
    )

    # Verify the results
    assert len(result) == 3  # Should have 3 tile predictions

    # Check cells assigned to tile (1, 1)
    assert len(result[0].contours) == 1
    assert len(result[0].centroids) == 1
    assert result[0].cell_types == ["type_A"]
    assert result[0].cell_type_probabilities == [0.9]

    # Check that coordinates are properly transformed to local tile system
    expected_local_centroid = cell_data["centroid"][0] - np.array(
        [1 * tile_size, 1 * tile_size]
    )
    expected_local_contour = cell_data["contour"][0] - np.array(
        [1 * tile_size, 1 * tile_size]
    )
    assert np.array_equal(result[0].centroids[0], expected_local_centroid)
    assert np.array_equal(result[0].contours[0], expected_local_contour)

    # Check cells assigned to tile (2, 1)
    assert len(result[1].contours) == 1
    assert len(result[1].centroids) == 1
    assert result[1].cell_types == ["type_B"]
    assert result[1].cell_type_probabilities == [0.8]

    # Check cells assigned to tile (1, 2)
    assert len(result[2].contours) == 1
    assert len(result[2].centroids) == 1
    assert result[2].cell_types == ["type_C"]
    assert result[2].cell_type_probabilities == [0.7]


def test_assign_cells_back_to_original_tiling_with_multiple_cells():
    """Test assigning multiple cells back to original tiles."""
    # Create a sample dataframe of cells with multiple cells in the same tile
    cell_data = {
        "contour": [
            np.array(
                [[230, 230], [250, 230], [250, 250], [230, 250]]
            ),  # Tile (1, 1), Cell 1
            np.array(
                [[260, 260], [280, 260], [280, 280], [260, 280]]
            ),  # Tile (1, 1), Cell 2
            np.array([[450, 230], [470, 230], [470, 250], [450, 250]]),  # Tile (2, 1)
        ],
        "centroid": [
            np.array([240, 240]),  # Tile (1, 1), Cell 1
            np.array([270, 270]),  # Tile (1, 1), Cell 2
            np.array([460, 240]),  # Tile (2, 1)
        ],
        "bounding_box": [
            np.array([230, 250, 230, 250]),  # Tile (1, 1), Cell 1
            np.array([260, 280, 260, 280]),  # Tile (1, 1), Cell 2
            np.array([230, 250, 450, 470]),  # Tile (2, 1)
        ],
        "cell_type": ["type_A", "type_A", "type_B"],
        "cell_type_probability": [0.9, 0.85, 0.8],
    }
    cell_df = pd.DataFrame(cell_data)

    # Define tile size and tile coordinates
    tile_size = 224
    coords = np.array(
        [
            [1, 1],  # Tile (1, 1)
            [2, 1],  # Tile (2, 1)
        ]
    )

    # Assign cells back to original tiles
    result = assign_cells_back_to_original_tiling(
        cell_df=cell_df,
        tile_size=tile_size,
        coords=coords,
    )

    # Verify the results
    assert len(result) == 2  # Should have 2 tile predictions

    # Check cells assigned to tile (1, 1) - should have 2 cells
    assert len(result[0].contours) == 2
    assert len(result[0].centroids) == 2
    assert result[0].cell_types == ["type_A", "type_A"]
    assert result[0].cell_type_probabilities == [0.9, 0.85]

    # Check cells assigned to tile (2, 1) - should have 1 cell
    assert len(result[1].contours) == 1
    assert len(result[1].centroids) == 1
    assert result[1].cell_types == ["type_B"]
    assert result[1].cell_type_probabilities == [0.8]


def test_assign_cells_back_to_original_tiling_with_empty_tiles():
    """Test assigning cells with some empty tiles."""
    # Create a sample dataframe of cells
    cell_data = {
        "contour": [
            np.array([[230, 230], [250, 230], [250, 250], [230, 250]]),  # Tile (1, 1)
            np.array([[450, 230], [470, 230], [470, 250], [450, 250]]),  # Tile (2, 1)
        ],
        "centroid": [
            np.array([240, 240]),  # Tile (1, 1)
            np.array([460, 240]),  # Tile (2, 1)
        ],
        "bounding_box": [
            np.array([230, 250, 230, 250]),  # Tile (1, 1)
            np.array([230, 250, 450, 470]),  # Tile (2, 1)
        ],
        "cell_type": ["type_A", "type_B"],
        "cell_type_probability": [0.9, 0.8],
    }
    cell_df = pd.DataFrame(cell_data)

    # Define tile size and tile coordinates, including a tile with no cells
    tile_size = 224
    coords = np.array(
        [
            [1, 1],  # Tile (1, 1)
            [2, 1],  # Tile (2, 1)
            [1, 2],  # Tile (1, 2) - should be empty
        ]
    )

    # Assign cells back to original tiles
    result = assign_cells_back_to_original_tiling(
        cell_df=cell_df,
        tile_size=tile_size,
        coords=coords,
    )

    # Verify the results
    assert len(result) == 3  # Should have 3 tile predictions

    # Check cells assigned to tile (1, 1)
    assert len(result[0].contours) == 1
    assert len(result[0].centroids) == 1

    # Check cells assigned to tile (2, 1)
    assert len(result[1].contours) == 1
    assert len(result[1].centroids) == 1

    # Check cells assigned to tile (1, 2) - should be empty
    assert len(result[2].contours) == 0
    assert len(result[2].centroids) == 0
    assert result[2].cell_types == []
    assert result[2].cell_type_probabilities == []
