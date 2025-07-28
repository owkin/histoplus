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
    cell_polygons = []
    cell_uid = {}

    for cell in tqdm(
        cell_df.itertuples(),
        desc="Filter overlapping cells (building polygons)",
        leave=False,
        disable=bool(verbose == 0),
    ):
        polygon = Polygon(cell.contour)

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
