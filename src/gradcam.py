import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os


class GradCAM:
    def __init__(self, model, target_layer):
        """
        model        → your EmotionCNN instance
        target_layer → the layer to hook into (model.block4)
        """
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor → (1, 1, 48, 48) tensor
        class_idx    → target class (if None, uses predicted class)
        Returns      → heatmap as numpy array (H, W) values in [0, 1]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Use predicted class if not specified
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()

        # Pool gradients across channels
        pooled_gradients = self.gradients.mean(dim=[0, 2, 3])

        # Weight activations by pooled gradients
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        # Generate heatmap
        heatmap = activations.mean(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU

        # Normalize to [0, 1]
        if heatmap.max() != 0:
            heatmap /= heatmap.max()

        return heatmap, class_idx


def overlay_heatmap(heatmap, original_image, alpha=0.4):
    """
    Overlays Grad-CAM heatmap on original image.
    heatmap        → numpy array (H, W) in [0, 1]
    original_image → PIL Image (grayscale)
    Returns        → PIL Image (RGB with heatmap overlay)
    """
    # Resize heatmap to match image size
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_resized = Image.fromarray(heatmap_resized).resize(
        original_image.size, resample=Image.BILINEAR
    )
    heatmap_resized = np.array(heatmap_resized)

    # Apply colormap
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized / 255.0)
    heatmap_colored = np.uint8(heatmap_colored[:, :, :3] * 255)

    # Convert grayscale to RGB
    original_rgb = np.array(original_image.convert("RGB"))

    # Blend
    overlay = np.uint8(alpha * heatmap_colored + (1 - alpha) * original_rgb)
    return Image.fromarray(overlay)


def save_gradcam(model, input_tensor, original_image, emotions, save_path, class_idx=None):
    """
    Full pipeline → generate + overlay + save.
    model          → EmotionCNN instance
    input_tensor   → (1, 1, 48, 48) tensor
    original_image → PIL Image
    emotions       → list of emotion names from config
    save_path      → where to save the result
    """
    gradcam = GradCAM(model, target_layer=model.block4)
    heatmap, predicted_class = gradcam.generate(input_tensor, class_idx)

    overlay = overlay_heatmap(heatmap, original_image)

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM: {emotions[predicted_class]}")
    axes[1].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Grad-CAM saved → {save_path}")

    return predicted_class