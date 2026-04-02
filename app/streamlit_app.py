import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import torch
from PIL import Image
import yaml

from src.model import build_model
from app.components.webcam import load_config, preprocess_image, decode_prediction
from app.components.confidence_bar import render_confidence_bar
from app.components.gradcam_view import render_gradcam


def load_model(config):
    """Load trained model from checkpoint."""
    device = torch.device("cpu")
    model = build_model(
        num_classes=config["data"]["num_classes"],
        dropout=config["model"]["dropout"]
    ).to(device)

    checkpoint_path = os.path.join(config["paths"]["checkpoint_dir"], "best_model.pt")

    if not os.path.exists(checkpoint_path):
        return None, device

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, device


def main():
    # Page config
    st.set_page_config(
        page_title="Facial Emotion Detection",
        page_icon=":face_with_raised_eyebrow:",
        layout="wide"
    )

    st.title("Facial Emotion Detection")
    st.markdown("Capture your face using the webcam and the model will detect your emotion.")

    # Load config and model
    config = load_config()
    emotions = config["data"]["emotions"]

    model, device = load_model(config)

    if model is None:
        st.error("No trained model found at models/best_model.pt — please train the model first.")
        return

    st.success("Model loaded successfully.")

    # Webcam input
    st.subheader("Webcam Capture")
    camera_image = st.camera_input("Take a photo")

    if camera_image is not None:
        # Load and display captured image
        original_image = Image.open(camera_image)

        # Preprocess
        input_tensor = preprocess_image(original_image, config).to(device)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)

        predicted_emotion, scores = decode_prediction(output, emotions)
        st.write("Raw probabilities:", torch.softmax(output, dim=1).tolist())

        # Results layout
        st.markdown("---")
        st.markdown(f"### Predicted Emotion: **{predicted_emotion.upper()}**")

        col1, col2 = st.columns(2)

        with col1:
            render_confidence_bar(scores, predicted_emotion)

        with col2:
            # For gradcam we need gradients so no torch.no_grad()
            render_gradcam(
                model=model,
                input_tensor=input_tensor,
                original_image=original_image.convert("L").resize((48, 48)),
                emotions=emotions,
                predicted_class=list(scores.keys()).index(predicted_emotion)
            )


if __name__ == "__main__":
    main()