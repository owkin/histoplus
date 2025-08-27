"""Utilities to fix border effects."""

from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from tqdm import tqdm

from histoplus.helpers.types import BorderInfo


def _fix_invalid_polygon(polygon: Polygon) -> Polygon:
    multi = polygon.buffer(0)

    if isinstance(multi, MultiPolygon):
        polygons = list(multi.geoms)
        if len(polygons) > 1:
            poly_idx = np.argmax([p.area for p in polygons])
            polygon = polygons[poly_idx]
        else:
            polygon = multi[0]
    else:
        polygon = Polygon(multi)

    return polygon


def build_cell_polygons(
    cell_df: pd.DataFrame, verbose: int
) -> tuple[list[Polygon], dict[Polygon, Any]]:
    """Build cell polygons."""
    cell_polygons = []
    cell_uid = {}

    for cell in tqdm(
        cell_df.itertuples(),
        desc="Filter overlapping cells (building polygons)",
        leave=False,
        disable=bool(verbose == 0),
    ):
        try:
            polygon = Polygon(cell.contour)
        except ValueError:
            # Some polygons have a degenerate shape with only 2 points
            # (i.e. 4 coordinates) which prevents shapely from creating
            # a shell. We skip those cells.
            continue

        if not polygon.is_valid:
            polygon = _fix_invalid_polygon(polygon)

        cell_uid[polygon] = cell.Index
        cell_polygons.append(polygon)

    return cell_polygons, cell_uid


def find_non_overlapping_border_cells(
    border_cells: pd.DataFrame,
    existing_tiles: set[str],
    verbose: int,
) -> pd.DataFrame:
    """Find border cells that don't have overlapping tiles."""
    non_overlapping = []

    for cell in tqdm(
        border_cells.itertuples(),
        desc="Filter overlapping cells (remove border cells)",
        leave=False,
        disable=bool(verbose == 0),
    ):
        border_info = _parse_border_information(cell.border_information)  # type: ignore
        if border_info.tiles:
            border_tile = _format_patch_coordinate(border_info.tiles[0])  # type: ignore
            if border_tile not in existing_tiles:
                non_overlapping.append(cell)

    out = pd.DataFrame([cell._asdict() for cell in non_overlapping])  # type: ignore

    if "Index" in out.columns:
        out.drop(columns=["Index"], axis=1, inplace=True)

    return out


def _parse_border_information(border_info_dict: dict) -> BorderInfo:
    """Parse border information dictionary into structured format."""
    return BorderInfo(tiles=border_info_dict["tiles"])


def _format_patch_coordinate(patch: tuple) -> str:
    """Format patch coordinate tuple into string identifier."""
    return f"{patch[0]}_{patch[1]}"


def rescale_cell_mask_coordinates_to_original_resolution(
    cell_df: pd.DataFrame,
    original_dz_level: int,
    extraction_dz_level: int,
) -> pd.DataFrame:
    """Rescale cell mask coordinates to fit into a tile of original tile size.

    If the inference of cell masks is made on tiles of different size (e.g. 448) as the
    one given by the user (e.g. 224), then the cell mask coordinates are lying in
    different coordinate systems. Therefore, we need to rescale the cell mask
    coordinates.

    Parameters
    ----------
    cell_df : list[pd.DataFrame]
        Collection of cells.

    original_dz_level : int
        DeepZoom level used provided by the user.

    extraction_dz_level: int
        DeepZoom level used to extract cell masks.

    Returns
    -------
    list[TilePrediction]
        Rescaled mask coordinates.
    """
    if original_dz_level == extraction_dz_level:
        return cell_df

    scale_ratio = 2 ** (original_dz_level - extraction_dz_level)

    def _scale(x):
        return x * scale_ratio

    cell_df.loc[:, "contour"] = cell_df["contour"].apply(_scale)
    cell_df.loc[:, "centroid"] = cell_df["centroid"].apply(_scale)
    cell_df.loc[:, "bounding_box"] = cell_df["bounding_box"].apply(_scale)

    return cell_df
