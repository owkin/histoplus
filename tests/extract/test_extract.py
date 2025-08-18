"""Unit tests for the extract module."""

import numpy as np
import pytest

from histoplus.extract import extract
from histoplus.helpers.tissue_detection import detect_tissue_on_wsi


N_WORKERS = 4
TILE_SIZE = 224
BATCH_SIZE = 16
N_TILES = 3


@pytest.mark.parametrize("segmentor_fixture", [("cellvit_segmentor", 448, 0.25)])
def test_extract(request, slide_data, segmentor_fixture):
    """Test the extract endpoint."""
    slide, _ = slide_data

    coords, dz_level = detect_tissue_on_wsi(slide)

    fixture_name, expected_train_image_size, expected_mpp = segmentor_fixture
    segmentor = request.getfixturevalue(fixture_name)

    assert expected_train_image_size == segmentor.train_image_size
    assert expected_mpp == segmentor.target_mpp

    cell_segmentation_data = extract(
        slide=slide,
        coords=coords,
        deepzoom_level=dz_level,
        segmentor=segmentor,
        tile_size=TILE_SIZE,
        n_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        n_tiles=N_TILES,
    )

    assert cell_segmentation_data is not None
