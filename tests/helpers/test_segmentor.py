"""Unit tests for the segmentor."""

import pytest
import torch

from histoplus.helpers.constants import BIOPTIMUS_MEAN, BIOPTIMUS_STD

from ..conftest import HF_HUB_NOT_AVAILABLE


# Batch size
B_SIZE = 4

BIOPTIMUS_STATS = (BIOPTIMUS_MEAN, BIOPTIMUS_STD)


@pytest.mark.skipif(HF_HUB_NOT_AVAILABLE, reason="Need access to a HF token")
@pytest.mark.parametrize(
    "segmentor_fixture", [("cellvit_segmentor", 448, 0.25, *BIOPTIMUS_STATS)]
)
def test_extract(request, segmentor_fixture):
    """Test the extract endpoint."""
    (
        fixture_name,
        expected_train_image_size,
        expected_target_mpp,
        expected_mean,
        expected_std,
    ) = segmentor_fixture
    segmentor = request.getfixturevalue(fixture_name)

    assert segmentor.train_image_size == expected_train_image_size
    assert segmentor.target_mpp == expected_target_mpp
    assert segmentor.mean == expected_mean
    assert segmentor.std == expected_std

    img = torch.randn(B_SIZE, 3, expected_train_image_size, expected_train_image_size)

    out = segmentor.forward(img)

    assert out is not None
