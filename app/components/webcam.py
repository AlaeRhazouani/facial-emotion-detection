import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import yaml


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_image(image: Image.Image, config: dict) -> torch.Tensor:
    """
    Takes a PIL Image (any mode/size) and returns
    a (1, 1, 48, 48) tensor ready for the model.
    """
    size = config["data"]["image_size"]
    mean = config["data"]["mean"]
    std = config["data"]["std"]

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

    tensor = transform(image)
    return tensor.unsqueeze(0)  # add batch dimension → (1, 1, 48, 48)


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