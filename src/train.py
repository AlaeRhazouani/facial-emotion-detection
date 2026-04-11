import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import yaml
from src.dataset import load_config, get_dataloaders
from src.model import EmotionCNN
from src.evaluate import evaluate


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def setup_wandb(config):
    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        config={
            "learning_rate": config["training"]["learning_rate"],
            "batch_size": config["training"]["batch_size"],
            "epochs": config["training"]["epochs"],
            "architecture": config["model"]["architecture"],
            "dropout": config["model"]["dropout"],
        }
    )

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader), correct / total

def train(config_path="configs/config.yaml"):
    config = load_config(config_path)
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader, test_loader, class_weights = get_dataloaders(config)
    class_weights = class_weights.to(device)

    model = EmotionCNN(
        num_classes=config["data"]["num_classes"],
        dropout=config["model"]["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    setup_wandb(config)

    best_val_acc = 0.0
    patience_counter = 0
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0]
        })

        print(f"Epoch {epoch+1}/{config['training']['epochs']} — train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"  Checkpoint saved — best val_acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    wandb.finish()
    print(f"Training complete. Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()