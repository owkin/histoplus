"""Test the CLI utilities."""

from unittest.mock import MagicMock, patch

import pytest

from histoplus.cli.utils import get_optimal_segmentor_for_slide

from ..conftest import HF_HUB_NOT_AVAILABLE


class TestGetOptimalSegmentorForSlide:
    """Test the get_optimal_segmentor_for_slide function."""

    @pytest.mark.skipif(HF_HUB_NOT_AVAILABLE, reason="Need access to a HF token")
    @patch("histoplus.cli.utils._get_best_available_mpp")
    def test_slide_with_25_and_5_mpp_using_histoplus(self, mock_get_mpp):
        """Test get_optimal_segmentor_for_slide with MPP 0.25 and 0.5 available using HistoPLUS.

        The function should return segmentor with MPP 0.25.
        """
        # Setup
        mock_slide = MagicMock()
        mock_get_mpp.return_value = 0.25  # Simulate slide with 0.25 MPP

        # Execute
        segmentor = get_optimal_segmentor_for_slide(mock_slide)

        # Assert
        assert segmentor.target_mpp == 0.25

    @pytest.mark.skipif(HF_HUB_NOT_AVAILABLE, reason="Need access to a HF token")
    @patch("histoplus.cli.utils._get_best_available_mpp")
    def test_slide_with_only_5_mpp_using_histoplus(self, mock_get_mpp):
        """Test get_optimal_segmentor_for_slide with only MPP 0.5 available using HistoPLUS.

        The function should return segmentor with MPP 0.5.
        """
        # Setup
        mock_slide = MagicMock()
        mock_get_mpp.return_value = 0.5  # Simulate slide with 0.5 MPP

        # Execute
        segmentor = get_optimal_segmentor_for_slide(mock_slide)

        # Assert
        assert segmentor.target_mpp == 0.5
