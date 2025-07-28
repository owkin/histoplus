"""Test for the tiling helper functions."""

import numpy as np
import pytest

from histoplus.helpers.tiling.new_tiling import (
    _get_large_tile_coords,
    get_new_tiling_for_target_tile_size_and_deepzoom_level,
)


def test_coords_are_identical_for_same_tile_size(slide_data):
    """Test that the coarser tiling has the same coordinates as the original tiling."""
    _, features = slide_data
    coords = features[:, 1:3].astype(int)
    original_tile_size = 224
    coarser_coords = _get_large_tile_coords(
        coords, original_tile_size, original_tile_size
    )
    assert np.all(coarser_coords == coords)


@pytest.mark.parametrize("target_tile_size", [256, 448, 896])
def test_coarser_tiling_has_fewer_tiles(slide_data, target_tile_size):
    """Test that the coarser tiling has fewer tiles than the original tiling."""
    _, features = slide_data
    coords = features[:, 1:3].astype(int)
    original_tile_size = 224

    coarser_coords = _get_large_tile_coords(
        coords, original_tile_size, target_tile_size
    )

    assert coarser_coords.shape[0] < coords.shape[0]


def test_coarser_tiling_covers_same_area(slide_data):
    """Test that the coarser tiling covers the same area as the original tiling."""
    _, features = slide_data
    coords = features[:, 1:3].astype(int)
    original_tile_size = 224
    target_tile_size = 448

    coarser_coords = _get_large_tile_coords(
        coords, original_tile_size, target_tile_size
    )

    coarser_set = {tuple(coord) for coord in coarser_coords}

    # Check coverage for each original tile
    for x, y in coords:
        # Calculate slide-level coordinates
        x_slide = x * original_tile_size
        y_slide = y * original_tile_size
        x_end = x_slide + original_tile_size
        y_end = y_slide + original_tile_size

        # Determine which coarse tiles are needed to cover this original tile
        cx_start = x_slide // target_tile_size
        cx_end = (x_end - 1) // target_tile_size
        cy_start = y_slide // target_tile_size
        cy_end = (y_end - 1) // target_tile_size

        # Verify all required coarse tiles exist
        for cx in range(cx_start, cx_end + 1):
            for cy in range(cy_start, cy_end + 1):
                assert (cx, cy) in coarser_set, (
                    f"Missing coarse tile ({cx}, {cy}) needed to cover original tile at ({x}, {y})"
                )


def test_coarser_tiling_of_square_area_has_n_times_fewer_tiles():
    """Test that for a square area and with a target tile size that is a divisor of the original tile size, the coarser tiling has fewer tiles."""
    original_tile_size = 224

    # Create a square area
    x = np.arange(10)
    y = np.arange(10)
    xx, yy = np.meshgrid(x, y)
    coords = np.vstack([xx.ravel(), yy.ravel()]).T

    # Get the coarser tiling
    target_tile_size = 448
    coarser_coords = _get_large_tile_coords(
        coords, original_tile_size, target_tile_size
    )

    assert (
        coarser_coords.shape[0]
        == coords.shape[0] / (target_tile_size / original_tile_size) ** 2
    )


def test_identical_tiling_for_same_tile_size_and_deepzoom_level(slide_data):
    """Test that for the same tissue area covered, the tiling is identical."""
    _, features = slide_data
    coords = features[:, 1:3].astype(int)

    new_coords = get_new_tiling_for_target_tile_size_and_deepzoom_level(
        coords,
        original_tile_size=224,
        original_deepzoom_level=16,
        target_tile_size=448,
        target_deepzoom_level=17,
    )

    assert np.all(new_coords == coords)
