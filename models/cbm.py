"""
Complete Concept Bottleneck Model: concept predictor + label predictor.

Supports concept-level intervention for causal analysis.
"""

import torch
import torch.nn as nn

from models.concept_predictor import ConceptPredictor
from models.label_predictor import LabelPredictor


class ConceptBottleneckModel(nn.Module):
    """Full CBM pipeline with intervention support."""

    def __init__(self, num_concepts: int, num_classes: int):
        super().__init__()
        self.concept_predictor = ConceptPredictor(num_concepts)
        self.label_predictor = LabelPredictor(num_concepts, num_classes)

    def forward(self, x, intervene_concepts=None):
        concept_probs, concept_logits = self.concept_predictor(x)

        if intervene_concepts is not None:
            concept_probs = intervene_concepts

        class_logits = self.label_predictor(concept_probs)
        return concept_probs, class_logits
