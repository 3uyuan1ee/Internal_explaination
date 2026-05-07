"""
Label predictor: concepts -> class logits.

Supports both linear and MLP modes. When expand_dim > 0, uses a 2-layer MLP
(inspired by Koh et al., ICML 2020 official CBM implementation).
When expand_dim == 0, falls back to a single interpretable linear layer.
"""

import torch
import torch.nn as nn


class LabelPredictor(nn.Module):

    def __init__(self, num_concepts: int, num_classes: int, expand_dim: int = 0):
        super().__init__()
        self.expand_dim = expand_dim
        if expand_dim > 0:
            self.fc1 = nn.Linear(num_concepts, expand_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(expand_dim, num_classes)
        else:
            self.linear = nn.Linear(num_concepts, num_classes)

    @property
    def weight_matrix(self):
        """Return the weight matrix for concept importance analysis."""
        if self.expand_dim > 0:
            return self.fc2.weight.data  # [num_classes, expand_dim]
        return self.linear.weight.data  # [num_classes, num_concepts]

    def forward(self, concepts):
        if self.expand_dim > 0:
            return self.fc2(self.relu(self.fc1(concepts)))
        return self.linear(concepts)
