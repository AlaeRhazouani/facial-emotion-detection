import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import yaml


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_image(image: Image.Image, config: dict) -> torch.Tensor:
    size = config["data"]["image_size"]
    mean = config["data"]["mean"]
    std = config["data"]["std"]

    # face detection with new mediapipe API
    img_array = np.array(image.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)

    base_options = python.BaseOptions(model_asset_path="mediapipe_model/blaze_face_short_range.tflite")
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    results = detector.detect(mp_image)

    if results.detections:
        bbox = results.detections[0].bounding_box
        x, y = bbox.origin_x, bbox.origin_y
        w, h = bbox.width, bbox.height
        face = image.crop((x, y, x + w, y + h))
    else:
        face = image

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    tensor = transform(face)
    return tensor.unsqueeze(0)


def decode_prediction(output: torch.Tensor, emotions: list) -> tuple:
    """
    Takes raw model output (1, 7) and returns:
    - predicted emotion string
    - confidence scores dict {emotion: probability}
    """
    probabilities = torch.softmax(output, dim=1)[0]
    predicted_idx = probabilities.argmax().item()
    predicted_emotion = emotions[predicted_idx]

    scores = {
        emotion: round(probabilities[i].item() * 100, 2)
        for i, emotion in enumerate(emotions)
    }

    return predicted_emotion, scores