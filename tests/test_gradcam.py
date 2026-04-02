import torch
import numpy as np
import pytest
from PIL import Image
from src.model import build_model
from src.gradcam import GradCAM, overlay_heatmap


class TestGradCAM:

    def setup_method(self):
        """Setup model and dummy input before each test"""
        self.model = build_model(num_classes=7)
        self.model.eval()
        self.input_tensor = torch.randn(1, 1, 48, 48)
        self.gradcam = GradCAM(self.model, target_layer=self.model.block4)

    def test_heatmap_shape(self):
        """Heatmap must match block4 spatial output (3x3)"""
        heatmap, _ = self.gradcam.generate(self.input_tensor)
        assert heatmap.shape == (3, 3), \
            f"Expected (3, 3), got {heatmap.shape}"

    def test_heatmap_values_range(self):
        """Heatmap values must be in [0, 1]"""
        heatmap, _ = self.gradcam.generate(self.input_tensor)
        assert heatmap.min() >= 0.0, \
            f"Heatmap min below 0: {heatmap.min()}"
        assert heatmap.max() <= 1.0, \
            f"Heatmap max above 1: {heatmap.max()}"

    def test_predicted_class_valid(self):
        """Predicted class must be in range [0, 6]"""
        _, predicted_class = self.gradcam.generate(self.input_tensor)
        assert 0 <= predicted_class <= 6, \
            f"Invalid class index: {predicted_class}"

    def test_target_class_respected(self):
        """When class_idx is specified, it must be used"""
        for target in range(7):
            _, predicted_class = self.gradcam.generate(
                self.input_tensor, class_idx=target
            )
            assert predicted_class == target, \
                f"Expected class {target}, got {predicted_class}"

    def test_activations_captured(self):
        """Forward hook must capture activations"""
        self.gradcam.generate(self.input_tensor)
        assert self.gradcam.activations is not None, \
            "Activations not captured by forward hook"

    def test_gradients_captured(self):
        """Backward hook must capture gradients"""
        self.gradcam.generate(self.input_tensor)
        assert self.gradcam.gradients is not None, \
            "Gradients not captured by backward hook"

    def test_heatmap_not_nan(self):
        """Heatmap must not contain NaN values"""
        heatmap, _ = self.gradcam.generate(self.input_tensor)
        assert not np.isnan(heatmap).any(), \
            "Heatmap contains NaN values"

    def test_overlay_output_type(self):
        """overlay_heatmap must return a PIL Image"""
        heatmap, _ = self.gradcam.generate(self.input_tensor)
        original = Image.fromarray(
            np.uint8(np.random.rand(48, 48) * 255), mode="L"
        )
        result = overlay_heatmap(heatmap, original)
        assert isinstance(result, Image.Image), \
            f"Expected PIL Image, got {type(result)}"

    def test_overlay_output_size(self):
        """Overlay must match original image size"""
        heatmap, _ = self.gradcam.generate(self.input_tensor)
        original = Image.fromarray(
            np.uint8(np.random.rand(48, 48) * 255), mode="L"
        )
        result = overlay_heatmap(heatmap, original)
        assert result.size == original.size, \
            f"Expected size {original.size}, got {result.size}"

    def test_different_inputs_different_heatmaps(self):
        """Different inputs must produce different heatmaps"""
        input1 = torch.randn(1, 1, 48, 48)
        input2 = torch.randn(1, 1, 48, 48)
        heatmap1, _ = self.gradcam.generate(input1)
        heatmap2, _ = self.gradcam.generate(input2)
        assert not np.allclose(heatmap1, heatmap2), \
            "Different inputs produced identical heatmaps"