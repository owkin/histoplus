"""Unit tests for the extract module."""

import numpy as np
import pytest

from histoplus.extract import extract


N_WORKERS = 4
TILE_SIZE = 224
BATCH_SIZE = 16
N_TILES = 3


@pytest.mark.parametrize("segmentor_fixture", [("maskdino_segmentor", 224, 0.5)])
def test_extract(request, slide_data, segmentor_fixture):
    """Test the extract endpoint."""
    slide, features = slide_data

    fixture_name, expected_train_image_size, expected_mpp = segmentor_fixture
    segmentor = request.getfixturevalue(fixture_name)

    assert expected_train_image_size == segmentor.train_image_size
    assert expected_mpp == segmentor.target_mpp

    cell_segmentation_data = extract(
        slide=slide,
        features=features,
        slide_path="",
        segmentor=segmentor,
        tile_size=TILE_SIZE,
        n_tiles=N_TILES,
        n_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        random_sampling=False,
    )

    assert cell_segmentation_data.mpp == expected_mpp

    original_coords = features[:N_TILES, 1:3]
    extracted_coords = np.array(
        [[tile["x"], tile["y"]] for tile in cell_segmentation_data.cell_masks]
    )

    np.testing.assert_equal(original_coords.astype(int), extracted_coords.astype(int))