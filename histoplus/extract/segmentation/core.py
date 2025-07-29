"""Cell segmentation logic."""

import numpy as np
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator

from histoplus.helpers.border_effects import fix_border_effects
from histoplus.helpers.constants import INFERENCE_TILE_OVERLAP
from histoplus.helpers.segmentor import Segmentor
from histoplus.helpers.types import TilePrediction

from .assemble import assemble_tile_predictions
from .dataloader import get_tile_dataloader
from .postprocess import postprocess_predictions
from .predict import predict_raw_maps


def extract_cell_segmentation_masks(
    slide: OpenSlide,
    deepzoom: DeepZoomGenerator,
    original_dz_level: int,
    extraction_dz_level: int,
    original_coords: np.ndarray,
    coarse_coords: np.ndarray,
    segmentor: Segmentor,
    tmp_dir: str,
    tile_size: int = 224,
    n_workers: int = 8,
    batch_size: int = 64,
    buffer_batch_size: int = 10,
    inference_tile_overlap: int = INFERENCE_TILE_OVERLAP,
    verbose: int = 1,
) -> list[TilePrediction]:
    """Use a cell segmentation model to extract cell segmentation masks from a WSI.

    Parameters
    ----------
    slide : OpenSlideType
        The whole slide image to process.

    deepzoom : DeepZoomType
        Deepzoom of the slide to extract. Either `openslide` deepzoom or client(image)
        deepzoom.

    original_dz_level : int
        DeepZoom level of the features provided by the user. Most likely the level at
        MPP 0.5, used by the TilingTool.

    extraction_dz_level : int
        DeepZoom level associated to the target MPP of the segmentor. If the segmentor
        predicts cells at MPP 0.25, this is the corresponding level.

    original_coords : np.ndarray
        The coordinates of the tissue tiles provided by the user. Typically these stem
        from the TilingTool.

    coarse_coords : np.ndarray
        The coordinates of the inference tissue tiles. These are tile coordinates of
        much larger tiles.

    segmentor : Segmentor
        The cell segmentation and classification model to use. If not provided, the
        best model available (developed in the HIPE project) will be used.

    tmp_dir : str
        Temporary directory to store raw maps.

    tile_size : int
        Tile size of the original tiling. This is typically 224, as it's the default in
        the TilingTool.

    n_workers : int
        The number of workers to use for parallel processing. Default is 1.

    batch_size : int
        Batch size for inference.

    buffer_batch_size : int
        Buffer batch size before dumping the raw intermediate maps.

    inference_tile_overlap : int
        Overlap (horizontal and vertical) between two consecutive tiles on the grid.

    verbose : int, optional
        If non null, displays message to stdout and tqdm.

    Returns
    -------
    list[TilePrediction]
        The segmentation masks and cell classes.
    """
    dataloader = get_tile_dataloader(
        slide=slide,
        deepzoom=deepzoom,
        segmentor=segmentor,
        coords=coarse_coords,
        level=extraction_dz_level,
        n_workers=n_workers,
        batch_size=batch_size,
    )

    # Run the forward pass of the segmentation / classification model and save the raw
    # predicted maps in a temporary file
    predict_raw_maps(segmentor, dataloader, tmp_dir, verbose)

    # Run the appropriate post-processing function to fuse the raw predicted maps into
    # a single instance segmentation map
    postprocess_predictions(segmentor, tmp_dir, n_workers)

    # Read from temporary file the batch-wise tile predictions and concatenate them to
    # create the final tile predictions object
    tile_predictions = assemble_tile_predictions(tmp_dir, verbose)

    # Remove border effects by projecting tile-wise predictions to a global coordinate
    # system, then discard cells at the border, and clean duplicates using the
    # union-find algorithm
    tile_predictions = fix_border_effects(
        tile_predictions=tile_predictions,
        tile_coords=original_coords,
        tile_size=tile_size,
        inference_tile_coords=coarse_coords,
        inference_tile_size=segmentor.inference_image_size,
        inference_tile_overlap=inference_tile_overlap,
        original_dz_level=original_dz_level,
        extraction_dz_level=extraction_dz_level,
        verbose=verbose,
    )

    return tile_predictions
