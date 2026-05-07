"""
Utility classes for training and evaluation.

AverageMeter ported from Koh et al. (ICML 2020) official CBM implementation.
"""

import numpy as np


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_attribute_imbalance(concept_labels):
    """Compute per-attribute positive/negative ratio for weighted loss.

    Args:
        concept_labels: np.array of shape (N, num_concepts) with binary values.

    Returns:
        pos_weight: torch.Tensor of shape (num_concepts,) — ratio of neg/pos per attribute.
    """
    import torch
    n_pos = concept_labels.sum(axis=0).clip(min=1)
    n_neg = concept_labels.shape[0] - n_pos
    pos_weight = (n_neg / n_pos).astype(np.float32)
    return torch.from_numpy(pos_weight)
