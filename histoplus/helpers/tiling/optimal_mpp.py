"""Utilites to determine the optimal MPP to use for cell mask segmentation.

These functions are extracted from the TilingTool.
"""

from typing import Optional

import numpy as np
from loguru import logger
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator

from histoplus.helpers.exceptions import MPPNotAvailableError


micrometers_from = {
    "cm": 10000,
    "mm": 1000,
    "centimeter": 10000,
    "millimeter": 1000,
    "inch": 25400,
}


def get_slide_resolution(slide: OpenSlide) -> float:
    """Get the slide resolution in mpp.

    Parameters
    ----------
    slide: OpenSlideType
        slide object

    Returns
    -------
    mpp: float
        The mpp of the slide
    """
    if "openslide.mpp-x" in slide.properties:
        return float(slide.properties["openslide.mpp-x"])
    elif "tiff.XResolution" in slide.properties:
        x_res = float(slide.properties["tiff.XResolution"])
        if x_res < 1:
            return x_res
        else:
            resolution_unit = slide.properties["tiff.ResolutionUnit"]
            if resolution_unit not in micrometers_from:
                raise KeyError("%s is not a valid resolution unit" % resolution_unit)
            return micrometers_from[resolution_unit] / x_res
    raise KeyError("Slide doesn't contain resolution property")


def get_level_raw_mpp_mapping(
    slide: OpenSlide,
    deepzoom: DeepZoomGenerator,
    default_mpp_max: Optional[float] = 0.25,
) -> dict[int, float]:
    """Compute the mapping between Deep Zoom level and raw mpp.

    Parameters
    ----------
    slide: OpenSlideType
        Slide object to extract tiles from
    deepzoom : DeepZoomType
        DeepZoomGenerator associated with the slide
    default_mpp_max: float = 0.25
        If the mpp max cannot be retrieved from the slide metadata, default mpp max will
        be used instead.
        if None, a KeyError will be raised

    Returns
    -------
    dict[int, float]:
        the mpp at each level

    Notes
    -----
        The raw mpp are returned (eg 0.243, 0.486), not the rounded mpp like 0.25, 0.5.
    """
    try:
        mpp_max = get_slide_resolution(slide)
    except KeyError as e:
        if default_mpp_max is not None:
            logger.warning(
                "Slide doesn't contain resolution property. "
                f"Assuming mpp max is {default_mpp_max}.",
            )
            mpp_max = default_mpp_max
        else:
            raise e

    n_levels = deepzoom.level_count
    level_mpp_mapping = {
        i: mpp_max * (2 ** (n_levels - 1 - i)) for i in range(n_levels)
    }
    return level_mpp_mapping


def get_tiling_slide_level(
    slide: OpenSlide,
    deepzoom: DeepZoomGenerator,
    mpp: float,
    default_mpp_max: Optional[float] = 0.25,
    mpp_rtol: float = 0.2,
    verbose: int = 1,
) -> int:
    """Get the DeepZoomGenerator level (tiling level) at a specific mpp.

    Also checks that the given mpp is valid.

    Parameters
    ----------
    slide: openslide.OpenSlide
        Slide object to extract tiles from
    deepzoom : DeepZoomType
        DeepZoomGenerator associated with the slide
    mpp: float
        the wanted mpp
    default_mpp_max: float = 0.25
        If the mpp max cannot be retrieved from the slide metadata, default mpp max will
        be used instead.
        if None, a KeyError will be raised
    mpp_rtol : float
        Maximum relative tolerance between the requested MPP
        and the closest MPP found in the slide.
    verbose : int
        Verbosity level.

    Returns
    -------
    level: int
        the level corresponding to the mpp
    """
    level_mpp_mapping = get_level_raw_mpp_mapping(slide, deepzoom, default_mpp_max)
    levels, raw_mpps = zip(*level_mpp_mapping.items(), strict=False)

    # Get the level for the target mpp
    level = levels[np.abs(np.array(raw_mpps) - mpp).argmin()]
    chosen_raw_mpp = raw_mpps[level]
    mpp_relative_diff = abs(mpp - chosen_raw_mpp) / mpp

    if mpp_relative_diff > mpp_rtol:
        raise MPPNotAvailableError(
            f"Requested MPP ({mpp}) is not available in the slide. "
            f"Closest one is {chosen_raw_mpp} (relative difference {mpp_relative_diff} > {mpp_rtol})."
            "Double check the resolution of the slide, or increase the `mpp_rtol` parameter."
        )

    if verbose:
        logger.info(
            f"{chosen_raw_mpp} is the closest available MPP to the one requested ({mpp}). "
            f"Minimal MPP of the slide is {min(raw_mpps)}."
        )

    return level
