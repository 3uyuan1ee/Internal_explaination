"""
Quantitative evaluation: accuracy comparison, intervention experiments,
concept purity, and concept fidelity.

Usage:
    python evaluate.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from cbm.config import DEVICE, CHECKPOINT_DIR, FIGURE_DIR, BASELINE_BATCH_SIZE, TORCH_LOAD_KWARGS, LABEL_EXPAND_DIM
from cbm.dataset import get_dataloaders
from cbm.models.baseline import BaselineModel
from cbm.models.concept_predictor import ConceptPredictor
from cbm.models.label_predictor import LabelPredictor
from cbm.models.cbm import ConceptBottleneckModel


# ── Accuracy Comparison ────────────────────────────────────────────────

def evaluate_baseline(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, _, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
    return correct / total


def evaluate_cbm(model, loader, device):
    model.eval()
    correct, total = 0, 0
    concept_correct, concept_total = 0, 0
    with torch.no_grad():
        for images, concepts, labels in loader:
            images = images.to(device)
            concepts = concepts.to(device)
            labels = labels.to(device)
            pred_concepts, logits = model(images)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
            concept_correct += ((pred_concepts > 0.5).float() == concepts).sum().item()
            concept_total += concepts.numel()
    return correct / total, concept_correct / concept_total


# ── Intervention Experiments ───────────────────────────────────────────

def intervention_experiment(model, loader, device):
    """Systematic concept intervention to verify causality."""
    model.eval()
    results = {
        "baseline": [],
        "random_flip_10": [],
        "random_flip_50": [],
        "top5_correct": [],
        "all_correct": [],
    }

    torch.manual_seed(42)  # Reproducible random flips
    with torch.no_grad():
        for images, true_concepts, true_labels in tqdm(loader, desc="Intervention"):
            images = images.to(device)
            true_concepts = true_concepts.to(device)
            true_labels = true_labels.to(device)

            # 1. Baseline (no intervention)
            pred_concepts, pred_logits = model(images)
            pred_classes = pred_logits.argmax(1)
            results["baseline"].append(
                (pred_classes == true_labels).float().mean().item()
            )

            # 2. Perfect intervention (ground truth concepts)
            _, intervened_logits = model(images, intervene_concepts=true_concepts)
            results["all_correct"].append(
                (intervened_logits.argmax(1) == true_labels).float().mean().item()
            )

            # 3. Random flip 10%
            mask_10 = torch.rand_like(pred_concepts) < 0.1
            flipped_10 = torch.where(mask_10, 1 - pred_concepts, pred_concepts)
            _, logits_10 = model(images, intervene_concepts=flipped_10)
            results["random_flip_10"].append(
                (logits_10.argmax(1) == true_labels).float().mean().item()
            )

            # 4. Random flip 50%
            mask_50 = torch.rand_like(pred_concepts) < 0.5
            flipped_50 = torch.where(mask_50, 1 - pred_concepts, pred_concepts)
            _, logits_50 = model(images, intervene_concepts=flipped_50)
            results["random_flip_50"].append(
                (logits_50.argmax(1) == true_labels).float().mean().item()
            )

            # 5. Correct top-5 most important concepts
            W = model.label_predictor.weight_matrix  # [num_classes, num_concepts]
            assert W.shape[1] == pred_concepts.shape[1], (
                f"Weight dim {W.shape[1]} != concept dim {pred_concepts.shape[1]}. "
                f"Set LABEL_EXPAND_DIM=0 for interpretable weights."
            )
            class_weights = W[pred_classes]
            importance = pred_concepts * class_weights.abs()
            _, top5_idx = importance.topk(5, dim=1)

            corrected = pred_concepts.clone()
            for i in range(images.size(0)):
                corrected[i, top5_idx[i]] = true_concepts[i, top5_idx[i]]
            _, logits_top5 = model(images, intervene_concepts=corrected)
            results["top5_correct"].append(
                (logits_top5.argmax(1) == true_labels).float().mean().item()
            )

    return {k: np.mean(v) for k, v in results.items()}


# ── Concept Fidelity ───────────────────────────────────────────────────

def concept_fidelity(model, loader, device):
    """Measure whether removing top-k concepts changes the prediction."""
    model.eval()
    changed_count = 0
    total = 0

    with torch.no_grad():
        for images, _, true_labels in tqdm(loader, desc="Fidelity"):
            images = images.to(device)
            pred_concepts, pred_logits = model(images)
            pred_classes = pred_logits.argmax(1)

            W = model.label_predictor.weight_matrix  # [num_classes, num_concepts]
            assert W.shape[1] == pred_concepts.shape[1], (
                f"Weight dim {W.shape[1]} != concept dim {pred_concepts.shape[1]}. "
                f"Set LABEL_EXPAND_DIM=0 for interpretable weights."
            )
            for i in range(images.size(0)):
                class_weights = W[pred_classes[i]]
                importance = pred_concepts[i] * class_weights.abs()
                _, top5_idx = importance.topk(5)

                masked = pred_concepts[i].clone()
                masked[top5_idx] = 0.0

                _, new_logits = model(
                    images[i:i+1],
                    intervene_concepts=masked.unsqueeze(0),
                )
                new_class = new_logits.argmax(1).item()
                changed_count += int(new_class != pred_classes[i].item())
                total += 1

    return changed_count / total


# ── Main ───────────────────────────────────────────────────────────────

def main():
    train_loader, test_loader, dataset = get_dataloaders(
        batch_size=BASELINE_BATCH_SIZE
    )
    num_concepts = dataset.num_concepts
    num_classes = dataset.num_classes

    print("=" * 60)
    print("QUANTITATIVE EVALUATION")
    print("=" * 60)

    # 1. Load models
    baseline = BaselineModel(num_classes=num_classes).to(DEVICE)
    baseline.load_state_dict(
        torch.load(CHECKPOINT_DIR / "baseline_best.pth", map_location=DEVICE, **TORCH_LOAD_KWARGS)
    )

    cbm = ConceptBottleneckModel(num_concepts, num_classes, expand_dim=LABEL_EXPAND_DIM).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_DIR / "cbm_best.pth", map_location=DEVICE, **TORCH_LOAD_KWARGS)
    cbm.concept_predictor.load_state_dict(ckpt["concept_model"])
    cbm.label_predictor.load_state_dict(ckpt["label_model"])

    # 2. Accuracy comparison
    print("\n--- Accuracy Comparison ---")
    baseline_acc = evaluate_baseline(baseline, test_loader, DEVICE)
    cbm_acc, concept_acc = evaluate_cbm(cbm, test_loader, DEVICE)
    print(f"  Baseline (ResNet-18):  {baseline_acc:.4f}")
    print(f"  CBM Classification:   {cbm_acc:.4f}")
    print(f"  CBM Concept Accuracy: {concept_acc:.4f}")
    print(f"  Accuracy drop:        {baseline_acc - cbm_acc:.4f} ({(baseline_acc - cbm_acc) / baseline_acc:.2%})")

    # 3. Intervention experiments
    print("\n--- Intervention Experiments ---")
    interv = intervention_experiment(cbm, test_loader, DEVICE)
    for name, acc in interv.items():
        label = {
            "baseline": "No intervention",
            "random_flip_10": "Random flip 10%",
            "random_flip_50": "Random flip 50%",
            "top5_correct": "Correct top-5 concepts",
            "all_correct": "Perfect intervention (all correct)",
        }[name]
        print(f"  {label:35s}: {acc:.4f}")

    # 4. Concept fidelity
    print("\n--- Concept Fidelity ---")
    fidelity = concept_fidelity(cbm, test_loader, DEVICE)
    print(f"  Prediction change rate after removing top-5 concepts: {fidelity:.4f}")
    print(f"  (Higher = concepts are more faithfully used)")

    # 5. Save results
    results = {
        "baseline_acc": baseline_acc,
        "cbm_acc": cbm_acc,
        "concept_acc": concept_acc,
        "accuracy_drop": baseline_acc - cbm_acc,
        "intervention": interv,
        "concept_fidelity": fidelity,
    }
    torch.save(results, FIGURE_DIR / "evaluation_results.pth")
    print(f"\nResults saved to {FIGURE_DIR / 'evaluation_results.pth'}")


if __name__ == "__main__":
    main()
