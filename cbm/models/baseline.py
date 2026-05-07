"""
Black-box baseline: ResNet-18 direct classification.
Used for Grad-CAM comparison with CBM.
"""

import torch
import torch.nn as nn
from torchvision import models


class BaselineModel(nn.Module):
    """Standard ResNet-18 fine-tuned for bird classification."""

    def __init__(self, num_classes: int = 24):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)
