"""Loading utilities."""

import pickle

import torch


def load_weights_from_hub(
    weight_path, map_location, pickle_module=pickle, **pickle_load_args
):
    """Load weights from the Hub."""
    return torch.load(
        weights_path,
        map_location=map_location,
        pickle_module=pickle_module,
        **pickle_load_args,
    )
