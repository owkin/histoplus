"""Remove overlapping cells."""

from collections import defaultdict
from dataclasses import asdict

import pandas as pd
from shapely import strtree
from tqdm import tqdm

from histoplus.helpers.types import GlobalSegmentedCell

from .utils import build_cell_polygons, find_non_overlapping_border_cells


def remove_overlapping_cells(
    cell_list: list[GlobalSegmentedCell],
    intersection_threshold: float = 0.01,
    verbose: int = 1,
):
    """Fix border effects in 3 steps.

    Step 1: Distinguish between safe and unsafe cells. Safe cells are within the ROI of
    a tile. Unsafe cells are outside the ROI (but not necesarily on the border) or cut
    by a border.

    Step 2: Filter all cells that are lying on the border of a tile, but do not have an
    overlap (e.g. a cell on the border of a tile located on the border of the tissue
    mask of a slide).

    Step 3: Iteratively suppress overlapping cells from unsafe cells.

    Parameters
    ----------
    cell_list : list[GlobalSegmentedCell]
        List of detected cell instances.
    intersection_threshold : float
        Minimum intersection ratio to consider cells as overlapping.

    Returns
    -------
    pd.DataFrame
        Cells without overlap.
    """
    cell_df = pd.DataFrame([asdict(c) for c in cell_list])

    if intersection_threshold <= 0 or intersection_threshold > 1:
        raise ValueError("intersection_threshold shoud be strictly between 0 and 1.")

    # Generate DataFrame
    cell_df[["tile_col", "tile_row"]] = pd.DataFrame(
        cell_df["tile_coordinates"].tolist(), index=cell_df.index
    )
    cell_df["tile_coordinates"] = cell_df["tile_coordinates"].apply(
        lambda x: "_".join(map(str, x))
    )

    # Define cells that are on the border and cells in the middle of a tile
    unsafe_cells = cell_df[cell_df["in_safe_area"] == 0]
    safe_cells = cell_df[cell_df["in_safe_area"] == 1]

    # Discard cells that are either on a tile border or overlapping other cells
    filtered_cells = _filter_cells_on_edges(unsafe_cells, verbose)

    # Merge with safe cells to obtain the cleaned cells
    filtered_cells = pd.concat((safe_cells, filtered_cells))

    # Edge case: sometimes two cells have an identical centroid
    filtered_cells["unique_id"] = filtered_cells["centroid"].apply(
        lambda x: f"{int(x[0])}_{int(x[1])}"
    )
    filtered_cells = filtered_cells.drop_duplicates(subset="unique_id", keep="first")
    filtered_cells = filtered_cells.reset_index(drop=True)

    postprocessed_cells = _filter_overlapping_cells(
        filtered_cells,
        intersection_threshold=intersection_threshold,
        verbose=verbose,
    )

    return postprocessed_cells


def _filter_cells_on_edges(unsafe_cell_df: pd.DataFrame, verbose) -> pd.DataFrame:
    # Cells outside the ROI (at the margin), but not touching the border
    margin_cells = unsafe_cell_df[~unsafe_cell_df["is_touching_border"].astype(bool)]
    # Cells touching the border
    border_cells = unsafe_cell_df[unsafe_cell_df["is_touching_border"].astype(bool)]

    # Find border cells without overlap
    existing_tiles = set(unsafe_cell_df["tile_coordinates"].unique())
    no_overlap_cells = find_non_overlapping_border_cells(
        border_cells, existing_tiles, verbose
    )

    clean_cells = pd.concat((margin_cells, no_overlap_cells)).sort_index(
        ignore_index=True
    )

    return clean_cells


def _filter_overlapping_cells(
    cell_df: pd.DataFrame,
    intersection_threshold: float,
    verbose: int,
) -> pd.DataFrame:
    """Filter overlapping cells from provided DataFrame."""
    # Build cell polygon
    cell_polygons, cell_uids = build_cell_polygons(cell_df, verbose)
    if not cell_polygons:
        return pd.DataFrame(columns=cell_df.columns)

    # Precompute areas for each cell
    areas = [poly.area for poly in cell_polygons]

    # Place them in a spatial index R-tree for fast querying
    tree = strtree.STRtree(cell_polygons)

    # Initialize Union-Find data structure
    parent = list(range(len(cell_polygons)))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u

    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            parent[root_v] = root_u  # Union without rank for simplicity

    # Iterate over each cell to find overlaps and union overlapping cells
    for i in tqdm(
        range(len(cell_polygons)),
        desc="Filter overlapping cells (find-union)",
        leave=False,
        disable=bool(verbose == 0),
    ):
        current_polygon = cell_polygons[i]
        intersecting_indices = tree.query(current_polygon)
        for j in intersecting_indices:
            if j <= i:
                continue  # Process each pair only once (j > i)
            other_polygon = cell_polygons[j]
            intersection = current_polygon.intersection(other_polygon)

            if intersection.is_empty:
                continue

            area_i = areas[i]
            area_j = areas[j]
            intersection_area = intersection.area
            ratio_i = intersection_area / area_i
            ratio_j = intersection_area / area_j
            if ratio_i >= intersection_threshold or ratio_j >= intersection_threshold:
                union(i, j)

    # Group cells by their root component
    groups = defaultdict(list)
    for idx in tqdm(
        range(len(cell_polygons)),
        desc="Filter overlapping cells (grouping)",
        leave=False,
        disable=bool(verbose == 0),
    ):
        root = find(idx)
        # Retrieve UID using the polygon at index idx
        uid = cell_uids[cell_polygons[idx]]
        groups[root].append(uid)

    uid_to_area = {cell_uids[poly]: areas[i] for i, poly in enumerate(cell_polygons)}

    # Select the UID with the maximum area in each group
    selected_uids = set()
    for group in groups.values():
        if not group:
            continue
        selected_uid = max(group, key=lambda uid: uid_to_area[uid])
        selected_uids.add(selected_uid)

    # Filter the original DataFrame based on selected UIDs
    filtered_df = cell_df.loc[cell_df.index.isin(selected_uids)]

    return filtered_df
