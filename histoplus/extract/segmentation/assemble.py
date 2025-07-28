"""Assemble tile predictions into global predictions."""

import pickle
from pathlib import Path

from tqdm import tqdm

from histoplus.helpers.types import TilePrediction

from .postprocess import TMP_FUSED_PREDICTIONS_SUBDIR


def assemble_tile_predictions(tmp_dir: str, verbose: int) -> list[TilePrediction]:
    """Concatenate batch-wise tile predictions into global tile predictions."""
    paths = list((Path(tmp_dir) / TMP_FUSED_PREDICTIONS_SUBDIR).glob("*.pkl"))
    # Sort paths to ensure consistent ordering
    paths.sort(key=lambda x: int(x.stem.split("_")[1]))

    tile_predictions = []

    for path in tqdm(
        paths,
        total=len(paths),
        desc="Assembling",
        leave=False,
        disable=bool(verbose == 0),
    ):
        with open(path, "rb") as fin:
            batch_predictions = pickle.load(fin)
        tile_predictions.extend(batch_predictions)

    return tile_predictions
