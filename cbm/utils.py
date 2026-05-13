"""Utility classes for CBM V2 training."""
import random
import numpy as np
import torch


def set_seed(seed=42):
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_attribute_imbalance(concept_labels):
    """Compute per-attribute pos_weight for BCEWithLogitsLoss."""
    import torch
    n_pos = concept_labels.sum(axis=0).clip(min=1)
    n_neg = concept_labels.shape[0] - n_pos
    return torch.from_numpy((n_neg / n_pos).astype(np.float32))
