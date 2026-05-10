"""
Concept predictor: InceptionV3 (partial freeze) -> concept logits.

Outputs concept logits (pre-sigmoid) for BCEWithLogitsLoss.
"""
import torch
import torch.nn as nn
from torchvision import models


class ConceptPredictor(nn.Module):
    def __init__(self, num_concepts: int):
        super().__init__()
        self.backbone = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT,
            aux_logits=True,
        )
        # Replace classification head with identity
        self.backbone.fc = nn.Identity()

        # Freeze layers up to and including Mixed_6e
        freeze_until = "Mixed_6e"
        freezing = True
        for name, child in self.backbone.named_children():
            if freezing:
                for param in child.parameters():
                    param.requires_grad = False
            if name == freeze_until:
                freezing = False

        # Concept prediction head
        self.concept_head = nn.Linear(2048, num_concepts)

    def forward(self, x):
        if self.training:
            logits, _ = self.backbone(x)  # InceptionV3 returns (logits, aux) in train
        else:
            logits = self.backbone(x)      # Returns tensor in eval
        concept_logits = self.concept_head(logits)
        concept_probs = torch.sigmoid(concept_logits)
        return concept_probs, concept_logits
