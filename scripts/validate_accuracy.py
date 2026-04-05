import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.dataset import load_config, get_dataloaders
from src.model import build_model


def validate(min_accuracy=0.60):
    config = load_config()
    device = torch.device("cpu")

    checkpoint_path = os.path.join(config["paths"]["checkpoint_dir"], "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No checkpoint found at {checkpoint_path}")
        sys.exit(1)

    model = build_model(
        num_classes=config["data"]["num_classes"],
        dropout=config["model"]["dropout"]
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    _, test_loader, _ = get_dataloaders(config)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Validation accuracy: {accuracy:.4f}")

    if accuracy < min_accuracy:
        print(f"FAILED: accuracy {accuracy:.4f} is below threshold {min_accuracy}")
        sys.exit(1)
    else:
        print(f"PASSED: accuracy {accuracy:.4f} meets threshold {min_accuracy}")
        sys.exit(0)


if __name__ == "__main__":
    validate()