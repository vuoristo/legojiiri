from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(train: bool = False) -> transforms.Compose:
    common = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train:
        # Additional augmentations can be added here for robustness.
        return transforms.Compose(common)
    return transforms.Compose(common)


def create_dataloaders(dataset_root: Path, batch_size: int, val_split: float = 0.2) -> Tuple[DataLoader, DataLoader, dict[int, str]]:
    dataset = datasets.ImageFolder(dataset_root, transform=build_transforms(train=True))
    class_to_idx = dataset.class_to_idx
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Add labeled images before training.")
    val_size = max(1, int(len(dataset) * val_split)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_size else None
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return train_loader, val_loader, idx_to_class


def preprocess_image(path: Path) -> torch.Tensor:
    transform = build_transforms(train=False)
    image = datasets.folder.default_loader(path)
    return transform(image)
