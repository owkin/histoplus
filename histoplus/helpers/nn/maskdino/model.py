"""Implementation of the MaskDINO model."""
from torch import nn

class MaskDINO(nn.Module):
    """Implementation of MaskDINO."""

    def __init__(
        self,
        cell_type_mapping: dict[int, str],
        mpp: float,
        output_layers: list[int],
        train_image_size: int,
        inference_image_size: int,
    ):
        super().__init__()

        self.mpp = mpp
        self.train_image_size = train_image_size
        self.inference_image_size = inference_image_size
        self.cell_type_mapping = cell_type_mapping
        self.output_layers = output_layers
        # TODO