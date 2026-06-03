"""
Custom PyTorch datasets for training and test-time inference on the Sargazo dataset.
Handles CSV parsing, image loading, augmentation, and missing file resilience.
"""

import os
import warnings
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def get_train_transforms():
    """Aggressive augmentation pipeline for training."""
    return transforms.Compose([
        transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_val_transforms():
    """Deterministic transforms for validation / inference."""
    return transforms.Compose([
        transforms.Resize(config.RESIZE_SIZE),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_tta_transforms():
    """Test-Time Augmentation: returns a list of transform variants."""
    base = [
        # Original
        transforms.Compose([
            transforms.Resize(config.RESIZE_SIZE),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize(config.RESIZE_SIZE),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]),
        # Slight rotation
        transforms.Compose([
            transforms.Resize(config.RESIZE_SIZE),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.RandomRotation(degrees=(10, 10)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]),
        # Vertical flip
        transforms.Compose([
            transforms.Resize(config.RESIZE_SIZE),
            transforms.CenterCrop(config.IMG_SIZE),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]),
    ]
    return base


class SargazoDataset(Dataset):
    """
    Training / validation dataset for sargassum level classification.
    
    Reads labels.csv, filters to only existing images, and applies
    the requested transforms.
    """

    def __init__(self, dataframe, images_dir, transform=None):
        """
        Args:
            dataframe: pandas DataFrame with columns ['image_name', 'label']
            images_dir: path to the directory containing images
            transform: torchvision transforms to apply
        """
        self.images_dir = images_dir
        self.transform = transform

        # Filter out rows whose images don't exist on disk
        valid_rows = []
        skipped = 0
        for _, row in dataframe.iterrows():
            img_path = os.path.join(images_dir, row["image_name"])
            if os.path.isfile(img_path):
                valid_rows.append(row)
            else:
                skipped += 1

        if skipped > 0:
            warnings.warn(
                f"SargazoDataset: {skipped} images not found in {images_dir}, skipped."
            )

        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)
        self.labels = self.data["label"].map(config.LABEL_TO_IDX).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.images_dir, row["image_name"])

        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_labels(self):
        """Return all labels (used for computing class weights and sampler)."""
        return self.labels

    def get_class_counts(self):
        """Return per-class sample counts."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))


class SargazoTestDataset(Dataset):
    """
    Test-time dataset (no labels). Reads test.csv for image names.
    """

    def __init__(self, csv_path, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform

        df = pd.read_csv(csv_path)
        self.image_names = df["image_name"].tolist()

        # Check which images exist
        existing = []
        missing = []
        for name in self.image_names:
            if os.path.isfile(os.path.join(images_dir, name)):
                existing.append(name)
            else:
                missing.append(name)

        if missing:
            warnings.warn(
                f"SargazoTestDataset: {len(missing)} test images not found. "
                f"First 5: {missing[:5]}"
            )

        # Keep all names (for submission) but mark missing ones
        self.existing_set = set(existing)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img_path = os.path.join(self.images_dir, name)

        if name in self.existing_set:
            image = Image.open(img_path).convert("RGB")
        else:
            # Return a blank image for missing files (will predict "nada" likely)
            image = Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, name
