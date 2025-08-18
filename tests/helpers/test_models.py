"""Unit and integration tests for :mod:`histoplus.utils.models`."""

import pytest
import torch

from histoplus.helpers.nn.cellvit import CellViT


NUMBER_CELL_TYPES = 15
OUT_LAYERS = [3, 5, 7, 11]
dummy_mapping = {i: str(i) for i in range(NUMBER_CELL_TYPES)}


@pytest.fixture(
    params=[
        CellViT(
            cell_type_mapping=dummy_mapping,
            mpp=1.0,
            output_layers=OUT_LAYERS,
            backbone_weights_pretraining="aquavit_105k",
            train_image_size=448,
            inference_image_size=448,
        ),
    ]
)
def cell_segmentation_model(request):
    """Create a cell segmentor model."""
    return request.param


def test_cell_segmentation_model_forward(cell_segmentation_model):
    """Test that the cell segmentation model forward pass works and outputs correct shapes."""
    input_size = cell_segmentation_model.train_image_size
    img = torch.randn(1, 3, input_size, input_size)
    cell_segmentation_model.eval()
    cell_segmentation_model.to("cpu")
    outputs = cell_segmentation_model.forward(img)
    assert outputs["np"].shape == (1, 2, input_size, input_size)
    assert outputs["tp"].shape == (1, NUMBER_CELL_TYPES, input_size, input_size)
    assert outputs["hv"].shape == (1, 2, input_size, input_size)
