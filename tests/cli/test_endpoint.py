"""Unit tests for the CLI."""

from unittest.mock import MagicMock

import pytest

from histoplus.cli.extract import extract_command
from histoplus.helpers.constants import (
    OutputFileType,
)
from histoplus.helpers.data import SlideSegmentationData

from ..conftest import HF_HUB_NOT_AVAILABLE, MOCK_SLIDE_PATH


TILE_SIZE = 224
N_TILES = 30
N_WORKERS = 4
BATCH_SIZE = 4
TUMOR_THRESHOLD = 0.1
SEED = 42


def create_mock_context():
    """Create a mock Typer context for testing."""
    mock_ctx = MagicMock()
    mock_ctx.params = {}
    return mock_ctx


@pytest.mark.skipif(HF_HUB_NOT_AVAILABLE, reason="Need HF token.")
def test_extract_cell_masks(tmp_path):
    """Test the extract_cell_masks CLI command."""
    slide_path = MOCK_SLIDE_PATH
    export_dir = tmp_path / "export"

    extract_command(
        create_mock_context(),
        slides=[str(slide_path)],
        export_dir=export_dir,
        tile_size=TILE_SIZE,
        n_tiles=N_TILES,
        n_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    # Check that the output file was created
    output_file = export_dir / slide_path.name / OutputFileType.JSON_CELL_MASKS.value
    assert output_file.exists(), f"Output file {output_file} was not created"

    # Load the saved SlideSegmentationData
    saved_data = SlideSegmentationData.load(output_file)

    # Check that the SlideSegmentationData object was correctly saved
    assert isinstance(saved_data, SlideSegmentationData)
