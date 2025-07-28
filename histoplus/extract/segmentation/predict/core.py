"""Predict module for segmentation."""

import os

import torch
from torch.utils.data import DataLoader

from histoplus.helpers.segmentor import Segmentor

from .concurrent_inference import ConcurrentModelInference


# Name of the temp subdirectory storing the post-processed segmentation maps
TMP_RAW_PREDICTIONS_SUBDIR = "raw_tile_predictions"


def predict_raw_maps(
    segmentor: Segmentor,
    dataloader: DataLoader,
    tmp_dir: str,
    verbose: int = 1,
):
    """Predict raw segmentation maps (e.g. HV, TP, NP for HoVerNet-based models)."""
    predictions_dir = os.path.join(tmp_dir, TMP_RAW_PREDICTIONS_SUBDIR)
    os.makedirs(predictions_dir, exist_ok=True)

    inference_module = ConcurrentModelInference(
        segmentor, predictions_dir, verbose=verbose
    )
    inference_module.run(dataloader)

    torch.cuda.empty_cache()
