import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import yaml

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    

class FERDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        self.label_map = {emotion: idx for idx, emotion in enumerate(self.emotions)}
        
        self.image_paths = []
        self.labels = []
        
        for emotion in self.emotions:
            folder = os.path.join(self.data_dir, emotion)
            for img_file in os.listdir(folder):
                self.image_paths.append(os.path.join(folder, img_file))
                self.labels.append(self.label_map[emotion])
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def get_class_weights(self):
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.labels),
            y=self.labels
        )
        return torch.tensor(weights, dtype=torch.float)
    

def get_transforms(config, split="train"):
    mean = config["data"]["mean"]
    std = config["data"]["std"]
    size = config["data"]["image_size"]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(config["augmentation"]["random_rotation"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std])
        ])


def get_dataloaders(config):
    data_dir = config["data"]["data_dir"]
    batch_size = config["training"]["batch_size"]

    train_dataset = FERDataset(data_dir, split="train", transform=get_transforms(config, "train"))
    test_dataset = FERDataset(data_dir, split="test", transform=get_transforms(config, "test"))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.get_class_weights()