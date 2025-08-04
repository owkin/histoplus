"""Constants for the histoplus package."""

from enum import Enum

import torch


class OutputFileType(Enum):
    """Output file types."""

    JSON_CELL_MASKS = "cell_masks.json"


"""Normalization statistics of Bioptimus extractor."""
BIOPTIMUS_MEAN = (0.707223, 0.578729, 0.703617)
BIOPTIMUS_STD = (0.211883, 0.230117, 0.177517)

"""Default batch size used to fill the buffer for the `extract` command."""
DEFAULT_BUFFER_BATCH_SIZE = 10

"""Overlap (in pixels) between two consecutive (horizontal and vertical) tiles."""
INFERENCE_TILE_OVERLAP = 64

"""Tile size (in pixels) for the inference. It should be a multiple of 14 and 16 to
adapt for ViT extractors trained with patch sizes of 14 and 16, and its square root
should be a whole integer. Only one candidate: 784."""
INFERENCE_TILE_SIZE = 784

"""Default device."""
DEFAULT_DEVICE = (
    None if (torch.cuda.is_available() or torch.backends.mps.is_available()) else -1
)
