"""Postprocess raw maps into instance prediction maps."""

import gc
import os
import pickle
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from histoplus.helpers.segmentor import Segmentor

from .predict import TMP_RAW_PREDICTIONS_SUBDIR


# Name of the temp subdirectory storing the raw predicted maps
TMP_FUSED_PREDICTIONS_SUBDIR = "fused_tile_predictions"


def postprocess_predictions(
    segmentor: Segmentor,
    tmp_dir: str,
    n_workers: int,
) -> None:
    """Combine raw maps into a single prediction instance map per tile."""
    predictions_dir = os.path.join(tmp_dir, TMP_FUSED_PREDICTIONS_SUBDIR)
    os.makedirs(predictions_dir, exist_ok=True)

    paths = list((Path(tmp_dir) / TMP_RAW_PREDICTIONS_SUBDIR).glob("*.npz"))
    # Sort paths to ensure consistent ordering
    paths.sort(key=lambda x: int(x.stem.split("_")[1]))

    n_jobs = min(n_workers, len(paths))
    postprocess_fn = segmentor.get_postprocess_fn()

    # Process files in order when running in parallel using their indices
    Parallel(n_jobs=n_jobs)(
        delayed(_fuse_raw_maps_into_prediction_maps)(path, tmp_dir, postprocess_fn, idx)
        for idx, path in enumerate(paths)
    )


def _fuse_raw_maps_into_prediction_maps(
    input_path, tmp_dir, postprocess_fn, index=None
):
    """Fuse raw maps into instance prediction maps."""
    raw_prediction_maps = np.load(input_path)

    fused_predictions = postprocess_fn(raw_prediction_maps)

    # Use the index in the save path to maintain order
    if index is not None:
        save_path = (
            Path(tmp_dir) / TMP_FUSED_PREDICTIONS_SUBDIR / f"ordered_{index:05d}.pkl"
        )
    else:
        save_path = Path(tmp_dir) / TMP_FUSED_PREDICTIONS_SUBDIR / input_path.name

    with open(save_path, "wb") as fout:
        pickle.dump(fused_predictions, fout)

    os.remove(input_path)
    del fused_predictions
    gc.collect()
