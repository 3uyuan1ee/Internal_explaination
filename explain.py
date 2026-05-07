"""
Explanation generation: concept-level explanations + Grad-CAM heatmaps.

Produces side-by-side visualizations comparing CBM concept explanations
with Grad-CAM saliency maps for the same test images.

Usage:
    python explain.py [--num_images 8]
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from pathlib import Path

from config import (
    DEVICE, CHECKPOINT_DIR, FIGURE_DIR, DATA_DIR,
    IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, TORCH_LOAD_KWARGS,
)
from dataset import get_dataloaders
from models.baseline import BaselineModel
from models.cbm import ConceptBottleneckModel

matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def denormalize(tensor):
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def generate_concept_explanation(cbm, image_tensor, attr_names, class_names, top_k=5):
    """Generate natural language concept explanation for a single image."""
    cbm.eval()
    with torch.no_grad():
        concept_probs, class_logits = cbm(image_tensor)

    pred_class = class_logits.argmax(1).item()
    pred_confidence = torch.softmax(class_logits, 1).max().item()

    W = cbm.label_predictor.linear.weight.data[pred_class]
    importance = concept_probs.squeeze().cpu() * W.cpu().abs()
    topk_values, topk_indices = importance.topk(top_k)

    explanation = {
        "predicted_class": class_names.get(pred_class, f"Class {pred_class}"),
        "confidence": pred_confidence,
        "concepts": [],
    }

    for val, idx in zip(topk_values, topk_indices):
        activation = concept_probs.squeeze()[idx].item()
        weight = W[idx].item()
        explanation["concepts"].append({
            "name": attr_names[idx],
            "activation": activation,
            "weight": weight,
            "importance": val.item(),
        })

    return explanation


def format_explanation_text(explanation):
    """Format explanation dict into readable text."""
    lines = [f"Prediction: {explanation['predicted_class']} "
             f"(conf: {explanation['confidence']:.2%})"]
    lines.append("Key concepts:")
    for c in explanation["concepts"]:
        direction = "+" if c["weight"] > 0 else "-"
        lines.append(f"  {direction} {c['name']:30s} "
                     f"act={c['activation']:.2f}  w={c['weight']:.2f}")
    return "\n".join(lines)


def generate_gradcam(baseline, image_tensor):
    """Generate Grad-CAM heatmap for the baseline model."""
    try:
        from pytorch_grad_cam import GradCAM
    except ImportError:
        print("pytorch-grad-cam not installed. Skipping Grad-CAM.")
        return None

    cam = GradCAM(
        model=baseline,
        target_layers=[baseline.backbone.layer4[-1]],
    )
    grayscale_cam = cam(input_tensor=image_tensor)
    return grayscale_cam[0]  # [H, W]


def visualize_comparison(
    image_tensor, gradcam_map, explanation, attr_names, class_names, save_path
):
    """Create side-by-side visualization: original + Grad-CAM vs concept explanation."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    img = denormalize(image_tensor.squeeze().cpu())

    # Panel 1: Original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    # Panel 2: Grad-CAM heatmap
    if gradcam_map is not None:
        from pytorch_grad_cam.utils.image import show_cam_on_image
        visualization = show_cam_on_image(img, gradcam_map, use_rgb=True)
        axes[1].imshow(visualization)
        axes[1].set_title("Grad-CAM (Post-hoc)", fontsize=13)
    else:
        axes[1].text(0.5, 0.5, "Grad-CAM\nnot available",
                     ha="center", va="center", fontsize=14)
        axes[1].set_title("Grad-CAM (Post-hoc)", fontsize=13)
    axes[1].axis("off")

    # Panel 3: Concept explanation bar chart
    concepts = explanation["concepts"]
    names = [c["name"][:25] for c in concepts]
    importances = [c["importance"] for c in concepts]
    activations = [c["activation"] for c in concepts]

    colors = ["#2ecc71" if c["weight"] > 0 else "#e74c3c" for c in concepts]
    y_pos = range(len(names))
    axes[2].barh(y_pos, importances, color=colors)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(names, fontsize=10)
    axes[2].set_xlabel("Importance Score")
    axes[2].set_title(f"CBM Concepts → {explanation['predicted_class']}", fontsize=13)
    axes[2].invert_yaxis()

    # Add activation annotations
    for i, (imp, act) in enumerate(zip(importances, activations)):
        axes[2].text(imp + 0.01, i, f"act={act:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=8)
    args = parser.parse_args()

    # Load dataset once — get test split from returned loader
    train_loader, test_loader, dataset = get_dataloaders(batch_size=32)
    attr_names = dataset.attr_names
    class_names = dataset.class_names
    num_concepts = dataset.num_concepts
    num_classes = dataset.num_classes

    # Access the underlying test dataset for indexed access
    test_dataset = test_loader.dataset

    # Load models
    baseline = BaselineModel(num_classes=num_classes).to(DEVICE)
    baseline.load_state_dict(
        torch.load(CHECKPOINT_DIR / "baseline_best.pth", map_location=DEVICE, **TORCH_LOAD_KWARGS)
    )

    cbm = ConceptBottleneckModel(num_concepts, num_classes).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_DIR / "cbm_best.pth", map_location=DEVICE, **TORCH_LOAD_KWARGS)
    cbm.concept_predictor.load_state_dict(ckpt["concept_model"])
    cbm.label_predictor.load_state_dict(ckpt["label_model"])

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(min(args.num_images, len(test_dataset))):
        image, concepts, label = test_dataset[i]
        image_tensor = image.unsqueeze(0).to(DEVICE)

        # CBM concept explanation
        explanation = generate_concept_explanation(
            cbm, image_tensor, attr_names, class_names, top_k=5
        )
        print(f"\n--- Image {i+1} ---")
        print(format_explanation_text(explanation))

        # Grad-CAM
        gradcam_map = generate_gradcam(baseline, image_tensor)

        # Visualize
        save_path = FIGURE_DIR / f"explanation_{i+1:02d}.png"
        visualize_comparison(
            image_tensor, gradcam_map, explanation,
            attr_names, class_names, save_path,
        )


if __name__ == "__main__":
    main()
