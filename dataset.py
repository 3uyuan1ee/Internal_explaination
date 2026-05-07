"""
CUB-200-2011 dataset loader with attribute annotations.

Loads images, binary attribute labels, and class labels for the selected
subset of 24 bird species. Automatically filters attributes by variance.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

from config import (
    DATA_DIR, ATTR_NAMES_FILE, SELECTED_CLASSES, MIN_ATTRIBUTE_VARIANCE,
    RESIZE_SIZE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
)
from torchvision import transforms


def _parse_cub_annotations(data_dir: Path, selected_classes: dict):
    """Parse CUB annotation files and return filtered image-level data.

    Returns:
        image_paths:   list of (image_id, relative_path)
        class_labels:  dict {image_id: local_class_id}
        attributes:    dict {image_id: np.array of binary attribute values}
        attr_names:    list of attribute name strings
        valid_attr_indices: indices of attributes passing variance filter
        class_names:   dict {local_id: english_name}
    """
    # 1. Load class names
    class_names = {}
    with open(data_dir / "classes.txt") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            class_id = int(parts[0])
            if class_id in selected_classes:
                # Class name format: "001.Ovenbird" → extract "Ovenbird"
                name = parts[1].split(".")[-1].replace("_", " ")
                class_names[selected_classes[class_id]] = name

    # 2. Load image paths and filter by selected classes
    id_to_path = {}
    with open(data_dir / "images.txt") as f:
        for line in f:
            img_id, rel_path = line.strip().split(" ", 1)
            id_to_path[int(img_id)] = rel_path

    # 3. Load class labels and filter
    id_to_class = {}
    with open(data_dir / "image_class_labels.txt") as f:
        for line in f:
            img_id, class_id = line.strip().split()
            img_id, class_id = int(img_id), int(class_id)
            if class_id in selected_classes:
                id_to_class[img_id] = selected_classes[class_id]

    # 4. Load train/test split
    id_to_split = {}
    with open(data_dir / "train_test_split.txt") as f:
        for line in f:
            img_id, is_train = line.strip().split()
            id_to_split[int(img_id)] = int(is_train)

    # 5. Load attribute names (located at data/attributes.txt, outside CUB dir)
    attr_names = []
    attr_file = ATTR_NAMES_FILE
    if not attr_file.exists():
        attr_file = data_dir / "attributes" / "attributes.txt"
    with open(attr_file) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            # Format: "1 has_bill_shape::cone" → extract human-readable name
            name = parts[1].replace("has_", "").replace("::", " ").replace("_", " ")
            attr_names.append(name)

    # 6. Load binary attribute labels for filtered images
    num_total_attrs = len(attr_names)
    raw_attributes = {}  # {image_id: np.array}

    # Build set of valid image IDs first
    valid_ids = set(id_to_class.keys())

    # Initialize arrays
    for img_id in valid_ids:
        raw_attributes[img_id] = np.zeros(num_total_attrs, dtype=np.float32)

    with open(data_dir / "attributes" / "image_attribute_labels.txt") as f:
        for line in f:
            parts = line.strip().split()
            img_id = int(parts[0])
            if img_id not in valid_ids:
                continue
            attr_id = int(parts[1]) - 1  # 0-indexed
            is_present = int(parts[2])
            certainty = int(parts[3])
            # Use only confident annotations (certainty >= 3)
            if certainty >= 3:
                raw_attributes[img_id][attr_id] = float(is_present)

    # 7. Filter attributes by variance across the selected subset
    attr_matrix = np.stack([raw_attributes[i] for i in sorted(valid_ids)])
    variances = attr_matrix.var(axis=0)
    valid_attr_mask = variances > MIN_ATTRIBUTE_VARIANCE
    valid_attr_indices = np.where(valid_attr_mask)[0]

    # Filtered attribute names
    filtered_attr_names = [attr_names[i] for i in valid_attr_indices]

    print(f"[Dataset] Selected {len(id_to_class)} images from {len(class_names)} classes")
    print(f"[Dataset] Attributes: {num_total_attrs} total → {len(valid_attr_indices)} after variance filter")

    return (id_to_path, id_to_class, raw_attributes, id_to_split,
            filtered_attr_names, valid_attr_indices, class_names)


def get_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
            transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
            transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class CUBDataset(Dataset):
    """CUB-200-2011 dataset for Concept Bottleneck Model.

    Returns:
        image:        Tensor [3, 224, 224]
        concepts:     Tensor [num_valid_attrs] — binary attribute labels
        label:        int — local class index (0-23)
    """

    def __init__(self, train: bool = True):
        (self.id_to_path, self.id_to_class, self.raw_attributes,
         self.id_to_split, self.attr_names, self.valid_attr_indices,
         self.class_names) = _parse_cub_annotations(DATA_DIR, SELECTED_CLASSES)

        self.transform = get_transforms(train)

        # Filter by train/test split
        split_flag = 1 if train else 0
        self.image_ids = sorted(
            img_id for img_id in self.id_to_class
            if self.id_to_split.get(img_id, -1) == split_flag
        )

        self.num_concepts = len(self.valid_attr_indices)
        self.num_classes = len(SELECTED_CLASSES)
        self.images_dir = DATA_DIR / "images"

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        # Load image
        img_path = self.images_dir / self.id_to_path[img_id]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Load filtered attributes
        all_attrs = self.raw_attributes[img_id]
        concepts = torch.tensor(all_attrs[self.valid_attr_indices], dtype=torch.float32)

        # Class label
        label = self.id_to_class[img_id]

        return image, concepts, label


def get_dataloaders(batch_size: int = 32, num_workers: int = 2):
    """Create train and test dataloaders."""
    train_dataset = CUBDataset(train=True)
    test_dataset = CUBDataset(train=False)

    # Share filtered attribute info (same for both splits)
    assert train_dataset.valid_attr_indices is not None
    test_dataset.valid_attr_indices = train_dataset.valid_attr_indices
    test_dataset.attr_names = train_dataset.attr_names
    test_dataset.num_concepts = train_dataset.num_concepts

    pin_mem = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_mem, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem
    )

    return train_loader, test_loader, train_dataset
