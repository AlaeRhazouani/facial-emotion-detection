import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate(model, loader, criterion, device):
    """
    Called by train.py every epoch.
    Returns average loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / len(loader), correct / total


def plot_confusion_matrix(cm, emotions, results_dir):
    """
    Saves confusion matrix heatmap to docs/results/.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=emotions,
        yticklabels=emotions
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


def evaluate_model(model, loader, device, config):
    """
    Full evaluation after training.
    Generates confusion matrix + classification report.
    """
    emotions = config["data"]["emotions"]
    results_dir = config["paths"]["results_dir"]

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=emotions))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, emotions, results_dir)

    # Overall accuracy
    accuracy = (all_preds == all_labels).mean()
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")

    return accuracy, cm


def load_and_evaluate(config_path="configs/config.yaml"):
    """
    Standalone script entry point.
    Loads best checkpoint and runs full evaluation.
    """
    from src.model import build_model
    from src.dataset import get_dataloaders

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader, _ = get_dataloaders(config)

    model = build_model(
        num_classes=config["data"]["num_classes"],
        dropout=config["model"]["dropout"]
    ).to(device)

    checkpoint_path = os.path.join(config["paths"]["checkpoint_dir"], "best_model.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint from {checkpoint_path}")

    evaluate_model(model, test_loader, device, config)


if __name__ == "__main__":
    load_and_evaluate()