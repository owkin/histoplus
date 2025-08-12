"""Tests for SlideSegmentationData serialization/deserialization."""

from histoplus.helpers.data import SlideSegmentationData


def test_slide_segmentation_data_roundtrip(tmp_path, dummy_tile):
    """Ensure saving to JSON and loading back reconstructs the object."""
    original = SlideSegmentationData(
        model_name="unit-test-segmentor",
        inference_mpp=0.5,
        cell_masks=[dummy_tile],
    )

    out_path = tmp_path / "slide_data.json"
    original.save(out_path)

    loaded = SlideSegmentationData.load(out_path)

    # Dataclasses are frozen, default equality compares fields recursively
    assert loaded == original
