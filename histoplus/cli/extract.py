"""Extract CLI command."""

import time
from pathlib import Path
from typing import List

import numpy as np
import openslide
import typer
from loguru import logger
from openslide import OpenSlideError
from PIL import Image

from histoplus.cli.utils import (
    collect_slides_to_extract,
    dump_config,
    get_optimal_segmentor_for_slide,
)
from histoplus.extract import extract
from histoplus.helpers.constants import OutputFileType
from histoplus.helpers.exceptions import MPPNotAvailableError


Image.MAX_IMAGE_PIXELS = None


def _launch_extraction(
    slide_path: Path,
    features_path: Path,
    export_dir: Path,
    tile_size: int,
    batch_size: int,
    n_workers: int,
    verbose: int,
) -> None:
    """Extract cell masks from a slide using HistoPLUS segmentation."""
    slide_export_dir = export_dir / slide_path.name

    slide_cell_masks_path = slide_export_dir / OutputFileType.JSON_CELL_MASKS.value

    try:
        slide = openslide.OpenSlide(str(slide_path))

        segmentor = get_optimal_segmentor_for_slide(slide, verbose)

        slide_export_dir.mkdir(exist_ok=True, parents=True)

        features_arr = np.load(features_path)

        try:
            cell_segmentation_data = extract(
                slide=slide,
                features=features_arr,
                segmentor=segmentor,
                tile_size=tile_size,
                n_workers=n_workers,
                batch_size=batch_size,
                verbose=verbose,
            )
        # MPP max error
        except MPPNotAvailableError as e:
            logger.error(str(e))
        # Openslide errors
        except KeyError as e:
            logger.error(str(e))
        except OpenSlideError as e:
            logger.error(str(e))
        # Unexpected exception
        except Exception as e:
            logger.exception(e)
            raise e

        cell_segmentation_data.save(slide_cell_masks_path)

    except Exception as e:
        logger.exception(e)
        return


def extract_command(
    ctx: typer.Context,
    slides: List[str],
    features: List[str],
    export_dir: Path,
    tile_size: int,
    n_workers: int,
    batch_size: int,
    verbose: int,
):
    """Extract cell masks from slides using HistoPLUS segmentation."""
    dump_config(ctx, export_dir)

    logger.add(export_dir / "{time}.log")

    slide_paths, features_paths = collect_slides_to_extract(
        slides, features, export_dir
    )

    for slide_idx, (slide_path, features_path) in enumerate(
        zip(slide_paths, features_paths, strict=False)
    ):
        logger.info(
            f"{slide_idx + 1}/{len(slide_paths)} --- Starting processing of {slide_path.name}"
        )

        start = time.time()

        _launch_extraction(
            slide_path=slide_path,
            features_path=features_path,
            export_dir=export_dir,
            tile_size=tile_size,
            n_workers=n_workers,
            batch_size=batch_size,
            verbose=verbose,
        )

        logger.info(
            f"{slide_idx + 1}/{len(slide_paths)} --- Finished processing of {slide_path.name} in {time.time() - start:.1f} seconds"
        )
