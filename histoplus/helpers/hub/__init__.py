"""Access model weights from the S3 bucket hub."""

from typing import Callable, Union

import torch

from histoplus.helpers.constants import BIOPTIMUS_MEAN, BIOPTIMUS_STD
from histoplus.helpers.hub.utils import load_weights_from_hub
from histoplus.helpers.nn.cellvit import CellViT


ModuleCallbackType = Callable[
    [dict[int, str], float, list[int], int, int], torch.nn.Module
]


HF_REPO_ID = "Owkin-Bioptimus/histoplus"


class PretrainedSegmentor(torch.nn.Module):
    """PretrainedSegmentor class to hold information on the weights."""

    # This function returns the torch.nn.Module associated to the Hub weights
    module_fn: ModuleCallbackType
    # Mapping from predicted class indices to their corresponding names
    class_mapping: dict[int, str]
    # The resolution (microns per pixel) of the expected model's input
    mpp: float
    # The layer indices used to extract the intermediate feature maps fed to the decoder
    output_layers: list[int]
    # The image size used during training
    train_image_size: int
    # Mean for normalizing the input
    mean: tuple[float, float, float]
    # Standard deviation for normalizing the input
    std: tuple[float, float, float]

    def __new__(cls, inference_image_size):
        """Create a new instance of the segmentor with the pretrained weights loaded."""
        module = cls.module_fn(
            cls.class_mapping,
            cls.mpp,
            cls.output_layers,
            cls.train_image_size,
            inference_image_size,
        )
        module.load_state_dict(
            load_weights_from_hub(HF_REPO_ID, cls.filename, map_location="cpu")
        )
        module.eval()
        return module


class histoplus_cellvit_segmentor_40x(PretrainedSegmentor):
    """CellViT cell segmentor trained at MPP 0.25.

    Trained on: LUAD, LUSC, MESO, BLCA, COAD, PAAD, OV, BRCA
    """

    name = "histoplus_cellvit_segmentor_40x"
    filename = "histoplus_cellvit_segmentor_40x.pt"
    mpp = 0.25

    output_layers = [3, 5, 7, 11]

    mean = BIOPTIMUS_MEAN
    std = BIOPTIMUS_STD

    train_image_size = 448

    module_fn: ModuleCallbackType = (
        lambda cell_type_mapping,
        mpp,
        out_layers,
        train_image_size,
        inference_image_size: CellViT(
            cell_type_mapping=cell_type_mapping,
            mpp=mpp,
            output_layers=out_layers,
            train_image_size=train_image_size,
            inference_image_size=inference_image_size,
        )
    )

    class_mapping = {
        0: "Background",
        1: "Cancer cell",
        2: "Lymphocytes",
        3: "Fibroblasts",
        4: "Plasmocytes",
        5: "Eosinophils",
        6: "Neutrophils",
        7: "Macrophages",
        8: "Muscle Cell",
        9: "Endothelial Cell",
        10: "Red blood cell",
        11: "Epithelial",
        12: "Apoptotic Body",
        13: "Mitotic Figures",
        14: "Minor Stromal Cell",
    }


class histoplus_cellvit_segmentor_20x(PretrainedSegmentor):
    """CellViT cell segmentor trained at MPP 0.5.

    Trained on: LUAD, LUSC, MESO, BLCA, COAD, PAAD, OV, BRCA.
    """

    name = "histoplus_cellvit_segmentor_20x"
    filename = "histoplus_cellvit_segmentor_20x.pt"
    mpp = 0.5

    output_layers = [3, 5, 7, 11]

    mean = BIOPTIMUS_MEAN
    std = BIOPTIMUS_STD

    train_image_size = 224

    module_fn: ModuleCallbackType = (
        lambda cell_type_mapping,
        mpp,
        out_layers,
        train_image_size,
        inference_image_size: CellViT(
            cell_type_mapping=cell_type_mapping,
            mpp=mpp,
            output_layers=out_layers,
            train_image_size=train_image_size,
            inference_image_size=inference_image_size,
        )
    )

    class_mapping = {
        0: "Background",
        1: "Cancer cell",
        2: "Lymphocytes",
        3: "Fibroblasts",
        4: "Plasmocytes",
        5: "Eosinophils",
        6: "Neutrophils",
        7: "Macrophages",
        8: "Muscle Cell",
        9: "Endothelial Cell",
        10: "Red blood cell",
        11: "Epithelial",
        12: "Apoptotic Body",
        13: "Mitotic Figures",
        14: "Minor Stromal Cell",
    }
