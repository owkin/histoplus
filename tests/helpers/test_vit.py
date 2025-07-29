"""Test to check the forward pass of the ViT."""

import pytest
import torch


@pytest.mark.parametrize("model_name", ["timm_extractor_vit_base"])
def test_positional_encoding_interpolation(model_name, request):
    """Test that ViT forward pass works correctly with different image sizes.

    This test verifies that the positional encoding interpolation works
    properly for both training-sized and inference-sized images.
    """
    model_fixture = request.getfixturevalue(model_name)
    model, patch_size = model_fixture

    def validate_output(image, expected_feature_dim=768):
        """Validate ViT feature map output dimensions for a given input image.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor of shape (batch_size, channels, height, width)
        expected_feature_dim : int, default=768
            Expected feature dimension of the output feature maps
        """
        batch_size, _, height, width = image.shape
        patches_h, patches_w = height // patch_size, width // patch_size

        # Extract feature maps from the model
        feature_maps = model(image)

        # Validate the shape of all feature maps
        for feature_map_name, feature_map in feature_maps.items():
            expected_shape = (batch_size, expected_feature_dim, patches_h, patches_w)
            assert feature_map.shape == expected_shape, (
                f"Feature map '{feature_map_name}' has shape {feature_map.shape}, "
                f"expected {expected_shape}"
            )

    # Test with training image size
    training_image = torch.randn(2, 3, 224, 224)
    validate_output(training_image)

    # Test with inference image size
    inference_image = torch.randn(2, 3, 896, 896)
    validate_output(inference_image)
