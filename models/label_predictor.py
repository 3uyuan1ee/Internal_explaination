"""
Label predictor: sparse linear layer from concepts to classes.

A single linear layer is used so that each class's prediction is a
weighted sum of concept activations — fully interpretable by reading
the weight matrix.
"""

import torch
import torch.nn as nn


class LabelPredictor(nn.Module):
    """Concepts → class logits via a sparse linear layer."""

    def __init__(self, num_concepts: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(num_concepts, num_classes)

    def forward(self, concepts):
        return self.linear(concepts)  # [B, num_classes]
