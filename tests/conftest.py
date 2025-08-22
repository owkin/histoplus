"""Fixtures for the tests."""

import hashlib
import importlib.util
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import openslide
import pytest
from typer.testing import CliRunner

from histoplus.helpers.data import SegmentationPolygon, TileSegmentationData
from histoplus.helpers.nn.extractor import TimmExtractor
from histoplus.helpers.segmentor import CellViTSegmentor


MOCK_SLIDE_PATH = Path("./CMU-1-JP2K-33005.svs")
WSI_DOWNLOAD_URL = "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-JP2K-33005.svs"
WSI_EXPECTED_HASH = "9a1923cd9bcb260ba4d99d64f8d6e32550648c332ba48817f920662f3a513420"

HF_HUB_NOT_AVAILABLE = os.getenv("HUGGING_FACE_HUB_TOKEN") is None


def download_wsi_if_missing(
    file_path: Path, download_url: str, expected_hash: str = None
) -> bool:
    """
    Download WSI file if it doesn't exist locally.

    Args:
        file_path: Path where the file should be stored
        download_url: URL to download the file from
        expected_hash: Optional SHA256 hash for file verification

    Returns
    -------
        bool: True if file exists or was successfully downloaded, False otherwise
    """
    if file_path.exists():
        print(f"WSI file already exists at {file_path}")
        return True

    # Create parent directories if they don't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading WSI file from {download_url} to {file_path}")

    try:
        # Download with progress indication
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) / total_size)
                print(f"\rDownload progress: {percent:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(download_url, file_path, reporthook=report_progress)
        print()  # New line after progress

        # Verify file hash if provided
        if expected_hash:
            print("Verifying file integrity...")
            actual_hash = calculate_file_hash(file_path)
            if actual_hash.lower() != expected_hash.lower():
                file_path.unlink()  # Remove corrupted file
                raise ValueError(
                    f"File hash mismatch. Expected: {expected_hash}, Got: {actual_hash}"
                )
            print("File integrity verified.")

        print(f"Successfully downloaded WSI file to {file_path}")
        return True

    except urllib.error.URLError as e:
        print(f"Failed to download WSI file: {e}")
        return False
    except Exception as e:
        print(f"Error downloading WSI file: {e}")
        if file_path.exists():
            file_path.unlink()  # Clean up partial download
        return False


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


@pytest.fixture(scope="session")
def ensure_wsi_available():
    """Ensure WSI file is available for testing."""
    if not download_wsi_if_missing(
        MOCK_SLIDE_PATH, WSI_DOWNLOAD_URL, WSI_EXPECTED_HASH
    ):
        pytest.skip("WSI file is not available and could not be downloaded")
    return MOCK_SLIDE_PATH


@pytest.fixture
def slide_data():
    """Load slide for testing."""
    return openslide.open_slide(str(MOCK_SLIDE_PATH))


@pytest.fixture
def dummy_polygon():
    """Create a dummy polygon."""
    return SegmentationPolygon(
        cell_id=0,
        cell_type="dummy_polygon",
        confidence=0.5,
        coordinates=[[0, 0], [0, 1], [1, 1], [1, 0]],
        centroid=[0.5, 0.5],
    )


@pytest.fixture
def dummy_tile(dummy_polygon):
    """Create a dummy tile."""
    return TileSegmentationData(
        level=0,
        x=200,
        y=201,
        width=224,
        height=448,
        masks=[dummy_polygon],
    )


@pytest.fixture
def cellvit_segmentor():
    """Create a CellViT segmentor."""
    return CellViTSegmentor.from_histoplus(
        mixed_precision=True,
        mpp=0.25,
        inference_image_size=448,
    )


@pytest.fixture
def timm_extractor_vit_base():
    """Create a DINO ViT base (for AquaViT) and output is patch size."""
    return TimmExtractor(model="base_s14"), 14


def import_app():
    """Import the CLI app dynamically to avoid early imports."""
    # This allows us to test the app without importing all dependencies upfront
    try:
        from histoplus.cli.app import app

        return app
    except ImportError:
        # If direct import fails, try to locate the module path
        module_name = "histoplus.cli"
        for path in sys.path:
            potential_path = Path(path) / module_name.replace(".", "/") / "__init__.py"
            if potential_path.exists():
                spec = importlib.util.spec_from_file_location(
                    module_name, potential_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.app

        pytest.skip("Could not find the CLI module")


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def cli_app():
    """Import the CLI app using the lazy approach."""
    return import_app()
