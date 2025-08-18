"""Loading utilities."""

import os
import pickle
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError


ENV_HISTOWMICS_HOME = "HISTOWMICS_HOME"
DEFAULT_CACHE_DIR = Path("~/.histoplus").expanduser()

# Subfolder under HISTOWMICS_HOME
HF_SUBCACHE = "hf_cache"


class HistoPLUSAuthError(RuntimeError):
    """Raised when accessing a private Hub repo without proper auth (HF token)."""

    def __init__(self, repo_id: str):
        super().__init__(
            f"Authentication failed for repo '{repo_id}'. "
            "If the repository is private, set a valid token via the "
            "HUGGING_FACE_HUB_TOKEN environment variable (or `huggingface-cli login`)."
        )


class HistoPLUSNotFoundError(FileNotFoundError):
    """Raised when the request file/revision does not exist on the Hub."""

    def __init__(self, repo_id: str, filename: str, revision: Optional[str]):
        rev = f"@{revision}" if revision else ""
        super().__init__(
            f"File '{filename}' not found in repo '{repo_id}{rev}' on the Hugging Face Hub."
        )


def _get_cache_dir() -> Path:
    """Resolve the base cache directory (under HISTOWMICS_HOME/hf_cache)."""
    base = Path(os.getenv(ENV_HISTOWMICS_HOME, DEFAULT_CACHE_DIR)).expanduser()
    cache_dir = base / HF_SUBCACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_weights_from_hub(
    repo_id: str,
    filename: str,
    revision: Optional[str] = None,
    *,
    map_location: Optional[torch.device] = None,
    pickle_module=pickle,
    local_files_only: bool = False,
    **pickle_load_args,
):
    """
    Load model weights with torch.load, fetching from Hugging Face Hub if needed.

    Parameters
    ----------
    repo_id : str
        e.g. 'owkin/histoplus' (organization_or_user/repo_name).
    filename : str
        Path inside the repo, e.g. 'weights/model.pt'.
    revision : Optional[str]
        Branch, tag, or commit SHA. If None, uses repo default branch.
    map_location : Optional[torch.device]
        Where to map the loaded tensors (e.g., 'cpu').
    local_files_only : bool
        If True, do not attempt network access; use only local cache.

    Returns
    -------
    Any
        The object returned by `torch.load(...)`.
    """
    try:
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=str(_get_cache_dir()),
            local_files_only=local_files_only,
        )
    except LocalEntryNotFoundError as e:
        # Requested to stay offline but file not in cache
        raise HistoPLUSNotFoundError(repo_id, filename, revision) from e
    except HfHubHTTPError as e:
        # 401/403 -> auth; 404 -> not found; re-raise others verbatim
        if e.response is not None and e.response.status_code in (401, 403):
            raise HistoPLUSAuthError(repo_id) from e
        if e.response is not None and e.response.status_code == 404:
            raise HistoPLUSNotFoundError(repo_id, filename, revision) from e
        raise

    return torch.load(
        cached_path,
        map_location=map_location,
        pickle_module=pickle_module,
        **pickle_load_args,
    )
