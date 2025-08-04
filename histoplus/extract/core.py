"""Extract cell segmentation masks from a whole slide image."""

import tempfile

import numpy as np
from openslide import OpenSlide

from histoplus.extract.segmentation import extract_cell_segmentation_masks
from histoplus.extract.utils import get_tile_coordinates_and_deepzoom_for_segmentor
from histoplus.helpers.constants import (
    DEFAULT_BUFFER_BATCH_SIZE,
    INFERENCE_TILE_OVERLAP,
)
from histoplus.helpers.data.slide_segmentation_data import SlideSegmentationData
from histoplus.helpers.segmentor import Segmentor


def extract(
    slide: OpenSlide,
    features: np.ndarray,
    segmentor: Segmentor,
    tile_size: int = 224,
    n_workers: int = 4,
    batch_size: int = 16,
    buffer_batch_size: int = DEFAULT_BUFFER_BATCH_SIZE,
    inference_tile_overlap: int = INFERENCE_TILE_OVERLAP,
    verbose: int = 1,
) -> SlideSegmentationData:
    """Extract cell segmentation masks from a whole slide image.

    This function applies a cell segmentation and classification model to a whole
    slide image (or random subset of tiles) and outputs an object with the
    segmentation masks and cell classes.

    It also classifies ALL tissue tiles using a tile classification model.

    Note that it assumes that the slide has already been preprocessed and its features
    extracted using Owkin's `TilingTool`.

    Parameters
    ----------
    slide : OpenSlide
        The whole slide image to process.

    features : np.ndarray
        Feature matrix of shape (n_tiles, embedding_dim). By default, the tumor detector
        uses features extracted with a WideResNet50 model trained with MoCo on TCGA-COAD.
        Therefore, the feature dimension is 2048. If you are using a different feature
        extractor, make sure to specify your own tumor detector model.

    segmentor: torch.nn.Module, optional
        The cell segmentation and classification model to use. If not provided, the
        best model available (developed in the HIPE project) will be used.

    tile_size : int, optional
        The size of the tiles to use for the segmentation. Default is 224.

    n_workers : int, optional
        The number of workers to use for parallel processing. Default is 1.

    batch_size : int, optional
        Batch size for inference. Default is 16.

    buffer_batch_size : int, optional
        Amount of batches accumulated in memory before saving on disk intermediate maps.
        A low value can affect the performance, while a high value can lead to an OOM
        error.

    inference_tile_overlap : int
        Overlap (horizontal and vertical) between two consecutive tiles on the grid.

    verbose : int, optional
        If non null, displays message to stdout and tqdm.

    Returns
    -------
    SlideSegmentationData
        The segmentation masks and cell classes.
    """
    original_coords = features[:, 1:3]
    original_dz_level = int(features[0, 0])

    coarse_coords, deepzoom, extraction_dz_level = (
        get_tile_coordinates_and_deepzoom_for_segmentor(
            slide,
            original_coords,
            original_dz_level,
            segmentor,
            original_tile_size=tile_size,
            inference_tile_overlap=inference_tile_overlap,
            verbose=verbose,
        )
    )

    with tempfile.TemporaryDirectory(prefix="cell_segmentation_") as tmp_dir:
        tile_predictions = extract_cell_segmentation_masks(
            slide=slide,
            deepzoom=deepzoom,
            original_dz_level=original_dz_level,
            extraction_dz_level=extraction_dz_level,
            original_coords=original_coords,
            coarse_coords=coarse_coords,
            segmentor=segmentor,
            tmp_dir=tmp_dir,
            tile_size=tile_size,
            n_workers=n_workers,
            batch_size=batch_size,
            buffer_batch_size=buffer_batch_size,
            inference_tile_overlap=inference_tile_overlap,
            verbose=verbose,
        )

        slide_data = SlideSegmentationData.from_predictions(
            model_name=segmentor.segmentor_name,
            inference_mpp=segmentor.target_mpp,
            deepzoom_level=original_dz_level,
            tile_size=tile_size,
            tile_coordinates=original_coords,
            tile_predictions=tile_predictions,
        )

    return slide_data
