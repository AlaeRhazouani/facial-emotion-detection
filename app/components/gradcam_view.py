import streamlit as st
from PIL import Image
from src.gradcam import GradCAM, overlay_heatmap


def render_gradcam(model, input_tensor, original_image: Image.Image, 
                   emotions: list, predicted_class: int):
    """
    Renders Grad-CAM heatmap overlay alongside original image.
    model          → EmotionCNN instance
    input_tensor   → (1, 1, 48, 48) tensor
    original_image → PIL Image
    emotions       → list of emotion names from config
    predicted_class → index of predicted emotion
    """
    st.subheader("Grad-CAM Visualization")
    st.caption("Heatmap shows which facial regions influenced the prediction")

    # Generate heatmap
    gradcam = GradCAM(model, target_layer=model.block4)
    heatmap, _ = gradcam.generate(input_tensor, class_idx=predicted_class)

    # Overlay heatmap on original image
    overlay = overlay_heatmap(heatmap, original_image)

    # Display side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original**")
        st.image(original_image, use_column_width=True)

    with col2:
        st.markdown(f"**Grad-CAM: {emotions[predicted_class].capitalize()}**")
        st.image(overlay, use_column_width=True)