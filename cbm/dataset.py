"""
CUB-200-2011 dataset loader for CBM.

Selects 50 classes (fixed seed), filters attributes by variance,
and returns InceptionV3-compatible 299x299 images.
"""
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms

from .config import (
    DATA_DIR, ATTR_NAMES_FILE, NUM_CLASSES, SEED,
    MIN_ATTRIBUTE_VARIANCE, MIN_CERTAINTY,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
)
import cbm.config as cfg


def select_50_classes() -> dict:
    """Select 50 of 200 CUB classes using fixed seed."""
    if cfg.SELECTED_CLASSES is not None:
        return cfg.SELECTED_CLASSES
    with open(DATA_DIR / "classes.txt") as f:
        all_class_ids = [int(line.strip().split()[0]) for line in f]
    rng = random.Random(SEED)
    selected = sorted(rng.sample(all_class_ids, NUM_CLASSES))
    result = {cid: idx for idx, cid in enumerate(selected)}
    cfg.SELECTED_CLASSES = result
    return result


def _parse_attribute_names() -> list[str]:
    """Load human-readable attribute names."""
    attr_names = []
    attr_file = ATTR_NAMES_FILE
    if not attr_file.exists():
        attr_file = DATA_DIR / "attributes" / "attributes.txt"
    with open(attr_file) as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            name = parts[1].replace("has_", "").replace("::", " ").replace("_", " ")
            attr_names.append(name)
    return attr_names


def _load_annotations(selected_classes: dict):
    """Parse all CUB annotations for the selected 50 classes."""
    # Class names
    class_names = {}
    with open(DATA_DIR / "classes.txt") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            cid = int(parts[0])
            if cid in selected_classes:
                name = parts[1].split(".")[-1].replace("_", " ")
                class_names[selected_classes[cid]] = name

    # Image paths
    id_to_path = {}
    with open(DATA_DIR / "images.txt") as f:
        for line in f:
            img_id, rel_path = line.strip().split(" ", 1)
            id_to_path[int(img_id)] = rel_path

    # Class labels (filter to selected)
    id_to_class = {}
    with open(DATA_DIR / "image_class_labels.txt") as f:
        for line in f:
            img_id, cid = line.strip().split()
            cid = int(cid)
            if cid in selected_classes:
                id_to_class[int(img_id)] = selected_classes[cid]

    # Train/test split
    id_to_split = {}
    with open(DATA_DIR / "train_test_split.txt") as f:
        for line in f:
            img_id, is_train = line.strip().split()
            id_to_split[int(img_id)] = int(is_train)

    # Attribute annotations
    attr_names = _parse_attribute_names()
    num_total_attrs = len(attr_names)
    valid_ids = set(id_to_class.keys())
    raw_attributes = {img_id: np.zeros(num_total_attrs, dtype=np.float32)
                      for img_id in valid_ids}

    with open(DATA_DIR / "attributes" / "image_attribute_labels.txt") as f:
        for line in f:
            parts = line.strip().split()
            img_id = int(parts[0])
            if img_id not in valid_ids:
                continue
            attr_id = int(parts[1]) - 1
            is_present = int(parts[2])
            certainty = int(parts[3])
            if certainty >= MIN_CERTAINTY:
                raw_attributes[img_id][attr_id] = float(is_present)

    # Variance filter (only training samples to avoid data leakage)
    train_ids = sorted(i for i in valid_ids if id_to_split.get(i, -1) == 1)
    attr_matrix = np.stack([raw_attributes[i] for i in train_ids])
    variances = attr_matrix.var(axis=0)
    valid_attr_indices = np.where(variances > MIN_ATTRIBUTE_VARIANCE)[0]
    filtered_attr_names = [attr_names[i] for i in valid_attr_indices]

    print(f"[Dataset] {len(id_to_class)} images, {len(class_names)} classes")
    print(f"[Dataset] {num_total_attrs} attrs -> {len(valid_attr_indices)} after variance filter")

    return (id_to_path, id_to_class, raw_attributes, id_to_split,
            filtered_attr_names, valid_attr_indices, class_names)


def get_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class CUBDataset(Dataset):
    """CUB-200-2011 dataset for CBM V2."""
    def __init__(self, train: bool = True, selected_classes: dict = None,
                 valid_attr_indices=None, filtered_attr_names=None):
        if selected_classes is None:
            selected_classes = select_50_classes()

        (self.id_to_path, self.id_to_class, self.raw_attributes,
         self.id_to_split, self.attr_names, self.valid_attr_indices,
         self.class_names) = _load_annotations(selected_classes)

        if valid_attr_indices is not None:
            self.valid_attr_indices = valid_attr_indices
            self.attr_names = filtered_attr_names

        self.selected_classes = selected_classes
        self.transform = get_transforms(train)

        split_flag = 1 if train else 0
        self.image_ids = sorted(
            img_id for img_id in self.id_to_class
            if self.id_to_split.get(img_id, -1) == split_flag
        )
        self.num_concepts = len(self.valid_attr_indices)
        self.num_classes = len(selected_classes)
        self.images_dir = DATA_DIR / "images"

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.images_dir / self.id_to_path[img_id]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        all_attrs = self.raw_attributes[img_id]
        concepts = torch.tensor(
            all_attrs[self.valid_attr_indices], dtype=torch.float32
        )
        label = self.id_to_class[img_id]
        return image, concepts, label, img_id


def get_dataloaders(batch_size: int = 32, num_workers: int = 2):
    """Create train and test dataloaders."""
    selected_classes = select_50_classes()  # sets cfg.SELECTED_CLASSES on first call
    train_ds = CUBDataset(train=True, selected_classes=selected_classes)
    test_ds = CUBDataset(
        train=False,
        selected_classes=selected_classes,
        valid_attr_indices=train_ds.valid_attr_indices,
        filtered_attr_names=train_ds.attr_names,
    )
    pin_mem = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_mem
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem
    )
    return train_loader, test_loader, train_ds
