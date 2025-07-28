"""Fixtures for the tests."""

import numpy as np
import openslide
import pytest
from histoplus.helpers.data import SegmentationPolygon, TileSegmentationData
from histoplus.helpers.nn.common.extractor import TimmExtractor
from histoplus.helpers.segmentor.maskdino import MaskDINOSegmentor


SAGEMAKER_ARTIFACT_BASE_PATH = get_base_artifact_path()

MOCK_SLIDE_PATH = (
    SAGEMAKER_ARTIFACT_BASE_PATH
    / "TCGA-G2-A2EC-01Z-00-DX4.8E4382A4-71F9-4BC3-89AA-09B4F1B54985.svs"
)

PHIKON_FEATURES_ARRAY_PATH = (
    SAGEMAKER_ARTIFACT_BASE_PATH
    / "features_phikon"
    / "TCGA-G2-A2EC-01Z-00-DX4.8E4382A4-71F9-4BC3-89AA-09B4F1B54985.svs"
    / "features.npy"
)


@pytest.fixture
def slide_data():
    """Load slide for testing."""
    slide = openslide.open_slide(str(MOCK_SLIDE_PATH))
    features = np.load(PHIKON_FEATURES_ARRAY_PATH)
    return slide, features


@pytest.fixture
def dummy_square():
    """Create a dummy square polygon."""
    return SegmentationPolygon(
        cell_id=0,
        cell_type="unit_square",
        confidence=0.5,
        coordinates=[[0, 0], [0, 1], [1, 1], [1, 0]],
        centroid=[0.5, 0.5],
    )


@pytest.fixture
def dummy_circle():
    """Create a dummy circle polygon."""
    return SegmentationPolygon(
        cell_id=1,
        cell_type="unit_circle",
        confidence=0.3,
        coordinates=[[np.cos(t), np.sin(t)] for t in np.linspace(0, 2 * np.pi, 100)],
        centroid=[0, 0],
    )


@pytest.fixture
def dummy_ellipse():
    """Create a dummy ellipse polygon."""
    return SegmentationPolygon(
        cell_id=2,
        cell_type="ellipse",
        confidence=0.4,
        coordinates=[
            [1.4 * np.cos(t), 2.4 * np.sin(t)] for t in np.linspace(0, 2 * np.pi, 100)
        ],
        centroid=[0, 0],
    )


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
def maskdino_segmentor():
    """Create a MaskDINO segmentor."""
    return None  # TODO: implement


@pytest.fixture
def timm_extractor_vit_base():
    """Create a DINO ViT base (for AquaViT) and output is patch size."""
    return TimmExtractor(model="base_s14"), 14
