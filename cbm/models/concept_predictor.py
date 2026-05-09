"""
Concept predictor: ResNet-18 encoder → concept bottleneck (sigmoid outputs).

Maps input images to a vector of concept probabilities, one per attribute.
Trained with binary cross-entropy using CUB attribute annotations.
"""

import torch
import torch.nn as nn
from torchvision import models


class ConceptPredictor(nn.Module):
    """Image → concept probabilities."""

    def __init__(self, num_concepts: int):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the final FC layer, keep everything up to avgpool
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        # Freeze pretrained backbone (official CBM pattern)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.flatten = nn.Flatten()
        # Concept prediction head (only trainable part)
        self.concept_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_concepts),
        )

    def forward(self, x):
        features = self.flatten(self.encoder(x))  # [B, 512]
        concept_logits = self.concept_head(features)  # [B, num_concepts]
        concept_probs = torch.sigmoid(concept_logits)  # [B, num_concepts]
        return concept_probs, concept_logits
