"""Tests for the overlap module."""

import numpy as np
import pytest

from histoplus.helpers.border_effects.overlap import remove_overlapping_cells
from histoplus.helpers.types import BorderInfo, GlobalSegmentedCell


@pytest.fixture
def cell_grid_fixture():
    """Create a 2x2 grid of tiles with cells, some overlapping at borders."""
    # Create cells in a 2x2 grid (4 tiles)
    # Each tile is 100x100 pixels
    # Tile coordinates are (0,0), (0,1), (1,0), (1,1)
    cells = []

    # Safe margin for each tile (20 pixels from the border)
    safe_margin = 20

    # Generate cell data
    for tile_row in range(2):
        for tile_col in range(2):
            tile_origin_x = tile_col * 100
            tile_origin_y = tile_row * 100

            # Generate cells within the safe area (not at borders)
            for i in range(3):
                # Cell in safe area
                x = tile_origin_x + safe_margin + i * 20
                y = tile_origin_y + safe_margin + i * 20

                # Create a simple circular contour
                theta = np.linspace(0, 2 * np.pi, 20)
                radius = 5
                contour = np.array(
                    [[x + radius * np.cos(t), y + radius * np.sin(t)] for t in theta]
                )

                # Bounding box
                bbox = np.array([y - radius, y + radius, x - radius, x + radius])

                cells.append(
                    GlobalSegmentedCell(
                        contour=contour,
                        centroid=np.array([x, y]),
                        bounding_box=bbox,
                        cell_type="safe_cell",
                        cell_type_probability=0.9,
                        in_safe_area=True,
                        is_touching_border=False,
                        offset_global=np.array([tile_origin_x, tile_origin_y]),
                        tile_coordinates=np.array([tile_col, tile_row]),
                        border_information=BorderInfo(tiles=None),
                    )
                )

            # Generate cells at horizontal borders
            if tile_row < 1:  # Bottom row tiles
                for i in range(2):
                    # Cell at bottom border
                    x = tile_origin_x + 30 + i * 40
                    y = tile_origin_y + 100 - 5  # At the border

                    # Create a simple circular contour
                    theta = np.linspace(0, 2 * np.pi, 20)
                    radius = 10
                    contour = np.array(
                        [
                            [x + radius * np.cos(t), y + radius * np.sin(t)]
                            for t in theta
                        ]
                    )
                    bbox = np.array([y - radius, y + radius, x - radius, x + radius])

                    cells.append(
                        GlobalSegmentedCell(
                            contour=contour,
                            centroid=np.array([x, y]),
                            bounding_box=bbox,
                            cell_type="border_cell",
                            cell_type_probability=0.8,
                            in_safe_area=False,
                            is_touching_border=True,
                            offset_global=np.array([tile_origin_x, tile_origin_y]),
                            tile_coordinates=np.array([tile_col, tile_row]),
                            border_information=BorderInfo(
                                tiles=[[tile_col, tile_row + 1]]
                            ),
                        )
                    )

                    # Create overlapping cell from the tile below (only for some cells to test overlap removal)
                    if i == 0:
                        cells.append(
                            GlobalSegmentedCell(
                                contour=contour,  # Same contour for overlap
                                bounding_box=bbox,
                                centroid=np.array([x, y]),
                                cell_type="border_cell_overlap",
                                cell_type_probability=0.7,
                                in_safe_area=False,
                                is_touching_border=True,
                                offset_global=np.array(
                                    [tile_origin_x, tile_origin_y + 100]
                                ),
                                tile_coordinates=np.array([tile_col, tile_row + 1]),
                                border_information=BorderInfo(
                                    tiles=[[tile_col, tile_row]]
                                ),
                            )
                        )

            # Generate cells at vertical borders
            if tile_col < 1:  # Left column tiles
                for i in range(2):
                    # Cell at right border
                    x = tile_origin_x + 100 - 5  # At the border
                    y = tile_origin_y + 30 + i * 40

                    # Create a simple circular contour
                    theta = np.linspace(0, 2 * np.pi, 20)
                    radius = 10
                    contour = np.array(
                        [
                            [x + radius * np.cos(t), y + radius * np.sin(t)]
                            for t in theta
                        ]
                    )
                    bbox = np.array([y - radius, y + radius, x - radius, x + radius])

                    cells.append(
                        GlobalSegmentedCell(
                            contour=contour,
                            centroid=np.array([x, y]),
                            bounding_box=bbox,
                            cell_type="border_cell",
                            cell_type_probability=0.8,
                            in_safe_area=False,
                            is_touching_border=True,
                            offset_global=np.array([tile_origin_x, tile_origin_y]),
                            tile_coordinates=np.array([tile_col, tile_row]),
                            border_information=BorderInfo(
                                tiles=[[tile_col + 1, tile_row]]
                            ),
                        )
                    )

                    # Create overlapping cell from the tile to the right (only for some cells)
                    if i == 0:
                        cells.append(
                            GlobalSegmentedCell(
                                contour=contour,  # Same contour for overlap
                                centroid=np.array([x, y]),
                                bounding_box=bbox,
                                cell_type="border_cell_overlap",
                                cell_type_probability=0.7,
                                in_safe_area=False,
                                is_touching_border=True,
                                offset_global=np.array(
                                    [tile_origin_x + 100, tile_origin_y]
                                ),
                                tile_coordinates=np.array([tile_col + 1, tile_row]),
                                border_information=BorderInfo(
                                    tiles=[[tile_col, tile_row]]
                                ),
                            )
                        )

    return cells


def test_remove_overlapping_cells(cell_grid_fixture):
    """Test that overlapping cells are properly removed."""
    # Start with our test fixture cells
    cells = cell_grid_fixture

    # Count original number of cells by type
    original_cell_count = len(cells)
    original_types = {
        cell_type: sum(1 for cell in cells if cell.cell_type == cell_type)
        for cell_type in {cell.cell_type for cell in cells}
    }

    # Apply the remove_overlapping_cells function
    result_df = remove_overlapping_cells(cells, intersection_threshold=0.01)

    # Verify the number of cells has decreased (overlaps removed)
    assert len(result_df) < original_cell_count

    # Specific checks
    border_overlap_count = original_types.get("border_cell_overlap", 0)
    assert border_overlap_count > 0, (
        "Test fixture should include overlapping border cells"
    )

    # Check that all overlapping cells were removed
    assert "border_cell_overlap" not in set(result_df["cell_type"]), (
        "Overlapping cells should be removed"
    )

    # Check that all safe cells are retained
    safe_cells_count = sum(1 for cell in cells if cell.cell_type == "safe_cell")
    assert sum(result_df["cell_type"] == "safe_cell") == safe_cells_count, (
        "Safe cells should not be removed"
    )


def test_identical_centroids(cell_grid_fixture):
    """Test handling of cells with identical centroids."""
    cells = cell_grid_fixture.copy()

    # Create cells with identical centroids
    duplicate_cell = cells[0]

    # Create a duplicate cell with identical centroid but different contour
    new_cell = GlobalSegmentedCell(
        contour=duplicate_cell.contour.copy() * 1.1,  # Slightly larger
        centroid=duplicate_cell.centroid.copy(),  # Same centroid
        bounding_box=duplicate_cell.bounding_box.copy(),
        cell_type="duplicate_cell",
        cell_type_probability=0.9,
        in_safe_area=duplicate_cell.in_safe_area,
        is_touching_border=duplicate_cell.is_touching_border,
        offset_global=duplicate_cell.offset_global.copy(),
        tile_coordinates=duplicate_cell.tile_coordinates.copy(),
        border_information=duplicate_cell.border_information,
    )

    cells.append(new_cell)

    # Process cells with duplicates
    result_df = remove_overlapping_cells(cells, intersection_threshold=0.01)

    # Verify duplicate handling - should keep only one of them
    centroid_str = (
        f"{int(duplicate_cell.centroid[0])}_{int(duplicate_cell.centroid[1])}"
    )
    matching_centroids = result_df[result_df["unique_id"] == centroid_str]

    assert len(matching_centroids) == 1, "Should only keep one cell per unique centroid"
