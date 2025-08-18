"""Extract cell segmentation masks from a whole slide image."""

import tempfile
from typing import Optional

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
    coords: np.ndarray,
    deepzoom_level: int,
    segmentor: Segmentor,
    tile_size: int = 224,
    n_tiles: Optional[int] = None,
    n_workers: int = 4,
    batch_size: int = 16,
    buffer_batch_size: int = DEFAULT_BUFFER_BATCH_SIZE,
    inference_tile_overlap: int = INFERENCE_TILE_OVERLAP,
    verbose: int = 1,
) -> SlideSegmentationData:
    """Extract cell segmentation masks from a whole slide image.

    This function applies a cell segmentation model to a whole slide image
    and outputs an object with the segmentation masks and cell classes.

    Parameters
    ----------
    slide : OpenSlide
        The whole slide image to process.

    coords : np.ndarray
        Tile coordinates matrix of shape (n_tiles, 2). These coordinates are given to
        the DeepZoomGenerator.

    deepzoom_level : int
        The DeepZoom level used for the extraction.

    segmentor: torch.nn.Module, optional
        The cell segmentation and classification model to use. If not provided, the
        best model available (developed in the HIPE project) will be used.

    tile_size : int, optional
        The size of the tiles to use for the segmentation. Default is 224.

    n_tiles : int, optional
        The number of tiles to infer. By default, all tiles are inferred.

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
    original_coords = coords[:n_tiles, :]

    coarse_coords, deepzoom, extraction_dz_level = (
        get_tile_coordinates_and_deepzoom_for_segmentor(
            slide,
            original_coords,
            deepzoom_level,
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
            original_dz_level=deepzoom_level,
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
            deepzoom_level=deepzoom_level,
            tile_size=tile_size,
            tile_coordinates=original_coords,
            tile_predictions=tile_predictions,
        )

    return slide_data
