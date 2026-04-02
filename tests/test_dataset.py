import torch
import pytest
from unittest.mock import patch, MagicMock
from src.dataset import load_config, get_transforms, FERDataset, get_dataloaders


def test_load_config():
    config = load_config()
    assert "data" in config
    assert "training" in config
    assert "augmentation" in config


def test_get_transforms_train():
    config = load_config()
    transform = get_transforms(config, split="train")
    assert transform is not None


def test_get_transforms_test():
    config = load_config()
    transform = get_transforms(config, split="test")
    assert transform is not None


def test_transform_output_shape():
    config = load_config()
    transform = get_transforms(config, split="test")
    
    from PIL import Image
    import numpy as np
    dummy_img = Image.fromarray(np.uint8(np.random.rand(48, 48) * 255), mode="L")
    output = transform(dummy_img)
    
    assert output.shape == (1, 48, 48)


def test_transform_normalization():
    config = load_config()
    transform = get_transforms(config, split="test")
    
    from PIL import Image
    import numpy as np
    dummy_img = Image.fromarray(np.uint8(np.random.rand(48, 48) * 255), mode="L")
    output = transform(dummy_img)
    
    assert output.min() < 0 or output.max() <= 1.0