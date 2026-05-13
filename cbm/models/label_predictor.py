"""
Label predictor: concepts -> class logits (linear, interpretable).
"""
import torch
import torch.nn as nn


class LabelPredictor(nn.Module):
    def __init__(self, num_concepts: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(num_concepts, num_classes)

    @property
    def weight_matrix(self):
        return self.linear.weight.data  # [num_classes, num_concepts]

    def forward(self, concepts):
        return self.linear(concepts)
