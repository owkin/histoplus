"""Utils for the CLI."""

import glob
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import typer
import yaml
from loguru import logger
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator

from histoplus.helpers.constants import INFERENCE_TILE_SIZE
from histoplus.helpers.exceptions import MPPNotAvailableError
from histoplus.helpers.segmentor import CellViTSegmentor, Segmentor
from histoplus.helpers.tiling.optimal_mpp import get_tiling_slide_level


def is_url(url: str) -> bool:
    """Check if a string is a URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def dump_config(ctx: typer.Context, export_dir: Optional[Path] = None) -> None:
    """Save the command's configuration in the export directory.

    Parameters
    ----------
    ctx : typer.Context
        The Typer context containing the command parameters
    export_dir : Path
        Directory where to save the config
    """
    # If export_dir is None, try to find it in command arguments
    if export_dir is None:
        args = ctx.params.get("command", "").split()
        try:
            export_dir_index = args.index("--export_dir")
            if export_dir_index + 1 < len(args):
                export_dir = Path(args[export_dir_index + 1])
            else:
                raise ValueError("No value provided for --export_dir in command")
        except ValueError:
            logger.error("Could not find --export_dir in command arguments")
            raise

    config_file_export_path = export_dir / "config.yaml"
    config_file_export_path.parent.mkdir(exist_ok=True, parents=True)

    # Get the command's parameters from the context
    config = {
        key: value
        for key, value in ctx.params.items()
        if not key.startswith("_")  # Skip internal parameters
    }

    # Convert to YAML
    yaml_config = yaml.dump(
        config,
        default_flow_style=False,
        sort_keys=False,
    )

    logger.info(
        f"Configuration used for this pipeline (it will be saved in {config_file_export_path}):\n"
        f"{yaml_config}"
    )

    config_file_export_path.write_text(yaml_config)


def collect_paths(paths: list[str]) -> list[Path]:
    """Collect the paths from a list or a wildcard.

    Parameters
    ----------
    paths : list[str]
        The paths to collect.

    Returns
    -------
    paths : list[Path]
        The collected paths.
    """
    result = []
    for path in paths:
        # For URLs, we directly add the path
        if is_url(str(path)):
            result.append(path)
        # For local paths, we use the standard glob
        else:
            result += sorted(glob.glob(str(path)))
    return [Path(p) for p in result]


def _get_best_available_mpp(slide: OpenSlide, verbose: int) -> float:
    """Get the highest resolution possible on the slide.

    If neither 40x nor 20x are available, raise an error as we do not have a model
    suited for a resolution with MPP > 0.5.
    """
    best_mpp = 0.5

    # Create a dummy DeepZoomGenerator object to access zoom levels (we do not care
    # about tile size and overlap).
    deepzoom = DeepZoomGenerator(slide, tile_size=224, overlap=0, limit_bounds=False)

    try:
        dz_level = get_tiling_slide_level(slide, deepzoom, mpp=0.25, verbose=verbose)
        if verbose:
            logger.success(
                f"Found DeepZoom level corresponding to MPP 0.25 ({dz_level})!"
            )
        best_mpp = 0.25
    except MPPNotAvailableError:
        try:
            dz_level = get_tiling_slide_level(slide, deepzoom, mpp=0.5, verbose=verbose)
            if verbose:
                logger.warning(
                    "Could not find DeepZoom level for MPP 0.25. Falling back to MPP 0.5."
                )
        except MPPNotAvailableError as e:
            raise MPPNotAvailableError(
                "Could not find DeepZoom level for both MPP 0.25 and MPP 0.5"
            ) from e

    return best_mpp


def _get_best_available_segmentor(
    mpp: float,
    inference_image_size: int,
) -> Segmentor:
    """Get the best segmentor based on the required version and available MPP on the slide.

    Raises
    ------
    MPPNotAvailableError
        If neither MPP 0.25 nor MPP 0.5 resolution is available.
    ValueError
        If the specified version by the user is not available for the segmentor.
    """
    return CellViTSegmentor.from_histoplus(
        mpp=mpp,
        mixed_precision=True,
        inference_image_size=inference_image_size,
    )


def get_optimal_segmentor_for_slide(slide: OpenSlide, verbose: int = 1) -> Segmentor:
    """Get the segmentor that gives the best detection/classification performance.

    For a given segmentor version, two magnification levels may be available:
    20x (MPP 0.5) or 40x (MPP 0.25). Typically, models inferring at MPP 0.25
    provide better performance.

    This function first tries to use a segmentor at MPP 0.25 if both:
    1. The slide has this resolution available
    2. The requested segmentor version supports MPP 0.25 (e.g., V6)

    Otherwise, it falls back to MPP 0.5.

    Parameters
    ----------
    slide : OpenSlide
        Slide object.

    verbose : int
        Verbosity level.

    Returns
    -------
    Segmentor
        The segmentor that should yield the best performance.

    Raises
    ------
    ValueError
        If the requested segmentor version is not supported.
    """
    best_mpp = _get_best_available_mpp(slide, verbose)

    return _get_best_available_segmentor(
        mpp=best_mpp,
        inference_image_size=INFERENCE_TILE_SIZE,
    )
