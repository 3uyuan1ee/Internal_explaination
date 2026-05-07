"""
Visualization and analysis: generates all figures for the experiment report.

Produces:
1. Training curves (concept acc & classification acc)
2. Confusion matrices (CBM vs baseline)
3. Weight heatmap (label predictor W matrix)
4. Intervention curve
5. Accuracy-explainability Pareto plot

Usage:
    python analyze.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from sklearn.metrics import confusion_matrix
import itertools

from config import DEVICE, CHECKPOINT_DIR, FIGURE_DIR, BASELINE_BATCH_SIZE, TORCH_LOAD_KWARGS, LABEL_EXPAND_DIM
from dataset import get_dataloaders
from models.baseline import BaselineModel
from models.cbm import ConceptBottleneckModel

matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(cm, class_names, title, save_path, normalize=True):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels([class_names[i] for i in range(len(class_names))], rotation=90, fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels([class_names[i] for i in range(len(class_names))], fontsize=7)

    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def collect_predictions(model, loader, device, model_type="baseline"):
    """Collect all predictions and true labels."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, concepts, labels in loader:
            images = images.to(device)
            if model_type == "baseline":
                logits = model(images)
            else:
                _, logits = model(images)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_weight_heatmap(cbm, attr_names, class_names, save_path):
    """Visualize label predictor weight matrix W [classes x concepts]."""
    W = cbm.label_predictor.weight_matrix.cpu().numpy()

    # Only show top-20 concepts per class for readability
    # Compute importance per concept (max absolute weight across classes)
    concept_importance = np.abs(W).max(axis=0)
    top_concept_idx = np.argsort(concept_importance)[-30:]  # top 30 concepts
    W_subset = W[:, top_concept_idx]
    attr_subset = [attr_names[i][:20] for i in top_concept_idx]

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(W_subset, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(W_subset).max(), vmax=np.abs(W_subset).max())
    ax.set_xticks(range(len(attr_subset)))
    ax.set_xticklabels(attr_subset, rotation=90, fontsize=7)
    ax.set_yticklabels([class_names[i] for i in range(len(class_names))],
                       fontsize=8)
    ax.set_xlabel("Concept", fontsize=12)
    ax.set_ylabel("Bird Species", fontsize=12)
    ax.set_title("Label Predictor Weights (top-30 concepts)", fontsize=14)
    plt.colorbar(im, ax=ax, label="Weight value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_intervention_curve(interv_results, save_path):
    """Bar chart of intervention experiment results."""
    labels = [
        "No intervention",
        "Random flip\n10%",
        "Random flip\n50%",
        "Correct\ntop-5",
        "Perfect\n(all correct)",
    ]
    keys = ["baseline", "random_flip_10", "random_flip_50",
            "top5_correct", "all_correct"]
    values = [interv_results[k] for k in keys]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#3498db", "#e67e22", "#e74c3c", "#2ecc71", "#27ae60"]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Classification Accuracy", fontsize=13)
    ax.set_title("Intervention Experiment: Concept Causality Verification", fontsize=14)
    ax.set_ylim(0, max(values) * 1.12)
    ax.axhline(y=values[0], color="gray", linestyle="--", alpha=0.5, label="Baseline")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_accuracy_explainability_pareto(baseline_acc, cbm_acc, save_path):
    """Pareto frontier: accuracy vs explainability."""
    models = {
        "Black-box\n(ResNet-18)": (baseline_acc, 0.15),
        "Post-hoc\n(Grad-CAM)": (baseline_acc, 0.40),
        "CBM\n(Concept Bottleneck)": (cbm_acc, 0.90),
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (acc, explainability) in models.items():
        ax.scatter(explainability, acc, s=200, zorder=5)
        ax.annotate(name, (explainability, acc),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=11, fontweight="bold")

    # Draw Pareto frontier
    x_vals = [0.15, 0.40, 0.90]
    y_vals = [baseline_acc, baseline_acc, cbm_acc]
    ax.plot(x_vals, y_vals, "k--", alpha=0.3, label="Pareto frontier")

    ax.set_xlabel("Explainability Score (conceptual)", fontsize=13)
    ax.set_ylabel("Classification Accuracy", fontsize=13)
    ax.set_title("Accuracy–Explainability Trade-off", fontsize=14)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(min(cbm_acc, baseline_acc) - 0.1, max(baseline_acc, cbm_acc) + 0.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_global_explanation(cbm, attr_names, class_names, save_path, top_k=5):
    """Global explanation: top concepts per class as a grouped bar chart."""
    W = cbm.label_predictor.weight_matrix.cpu().numpy()
    num_classes = W.shape[0]

    # Select 6 representative classes for readability
    sample_classes = list(range(0, num_classes, num_classes // 6))[:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, c in zip(axes, sample_classes):
        top_idx = np.abs(W[c]).argsort()[-top_k:][::-1]
        names = [attr_names[i][:25] for i in top_idx]
        weights = [W[c, i] for i in top_idx]
        colors = ["#2ecc71" if w > 0 else "#e74c3c" for w in weights]

        ax.barh(range(top_k), weights, color=colors)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_title(class_names.get(c, f"Class {c}"), fontsize=12, fontweight="bold")
        ax.invert_yaxis()
        ax.axvline(x=0, color="black", linewidth=0.5)

    fig.suptitle("Global Explanation: Top Concepts per Species", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("Loading models and data...")
    train_loader, test_loader, dataset = get_dataloaders(
        batch_size=BASELINE_BATCH_SIZE
    )
    attr_names = dataset.attr_names
    class_names = dataset.class_names
    num_concepts = dataset.num_concepts
    num_classes = dataset.num_classes

    # Load models
    baseline = BaselineModel(num_classes=num_classes).to(DEVICE)
    baseline.load_state_dict(
        torch.load(CHECKPOINT_DIR / "baseline_best.pth", map_location=DEVICE, **TORCH_LOAD_KWARGS)
    )

    cbm = ConceptBottleneckModel(num_concepts, num_classes, expand_dim=LABEL_EXPAND_DIM).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_DIR / "cbm_best.pth", map_location=DEVICE, **TORCH_LOAD_KWARGS)
    cbm.concept_predictor.load_state_dict(ckpt["concept_model"])
    cbm.label_predictor.load_state_dict(ckpt["label_model"])

    # 1. Confusion matrices
    print("\n[1/5] Generating confusion matrices...")
    bl_preds, bl_labels = collect_predictions(baseline, test_loader, DEVICE, "baseline")
    cbm_preds, cbm_labels = collect_predictions(cbm, test_loader, DEVICE, "cbm")

    bl_cm = confusion_matrix(bl_labels, bl_preds, labels=list(range(num_classes)))
    cbm_cm = confusion_matrix(cbm_labels, cbm_preds, labels=list(range(num_classes)))

    plot_confusion_matrix(
        bl_cm, class_names,
        "Confusion Matrix: Black-box Baseline (ResNet-18)",
        FIGURE_DIR / "confusion_baseline.png",
    )
    plot_confusion_matrix(
        cbm_cm, class_names,
        "Confusion Matrix: Concept Bottleneck Model",
        FIGURE_DIR / "confusion_cbm.png",
    )

    # 2. Weight heatmap
    print("[2/5] Generating weight heatmap...")
    plot_weight_heatmap(cbm, attr_names, class_names,
                        FIGURE_DIR / "weight_heatmap.png")

    # 3. Intervention curve (from saved results or recompute)
    print("[3/5] Generating intervention curve...")
    results_path = FIGURE_DIR / "evaluation_results.pth"
    if results_path.exists():
        results = torch.load(results_path, map_location="cpu", **TORCH_LOAD_KWARGS)
        interv = results["intervention"]
        baseline_acc = results["baseline_acc"]
        cbm_acc = results["cbm_acc"]
    else:
        from evaluate import intervention_experiment, evaluate_baseline, evaluate_cbm
        baseline_acc = evaluate_baseline(baseline, test_loader, DEVICE)
        cbm_acc, _ = evaluate_cbm(cbm, test_loader, DEVICE)
        interv = intervention_experiment(cbm, test_loader, DEVICE)

    plot_intervention_curve(interv, FIGURE_DIR / "intervention_curve.png")

    # 4. Accuracy-explainability Pareto
    print("[4/5] Generating Pareto plot...")
    plot_accuracy_explainability_pareto(baseline_acc, cbm_acc,
                                        FIGURE_DIR / "pareto_plot.png")

    # 5. Global explanation
    print("[5/5] Generating global explanation...")
    plot_global_explanation(cbm, attr_names, class_names,
                            FIGURE_DIR / "global_explanation.png")

    print(f"\nAll figures saved to {FIGURE_DIR}/")


if __name__ == "__main__":
    main()
