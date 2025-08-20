"""Fixtures for the tests."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import openslide
import pytest
from typer.testing import CliRunner

from histoplus.helpers.data import SegmentationPolygon, TileSegmentationData
from histoplus.helpers.nn.extractor import TimmExtractor
from histoplus.helpers.segmentor import CellViTSegmentor


def get_base_artifact_path():
    """Get base artifact path."""
    return Path(
        "/home/sagemaker-user/custom-file-systems/efs/fs-09913c1f7db79b6fd/abstra-ci/artifacts"
    )


SAGEMAKER_ARTIFACT_BASE_PATH = get_base_artifact_path()

MOCK_SLIDE_PATH = (
    SAGEMAKER_ARTIFACT_BASE_PATH
    / "TCGA-G2-A2EC-01Z-00-DX4.8E4382A4-71F9-4BC3-89AA-09B4F1B54985.svs"
)


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
