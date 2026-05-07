"""
Adversarial robustness analysis: FGSM attack comparing Baseline vs CBM.

Produces:
1. Epsilon vs accuracy curve (Baseline vs CBM)
2. Side-by-side original vs perturbed image comparisons
3. Concept vulnerability ranking (sorted bar chart)
4. Concept perturbation heatmap
5. Defense recovery curve (concept intervention after attack)

FGSM attack core ported from code/fgsm.py.
Visualization patterns ported from code/laser_attach.py and code/stiker_attach.py.

Usage:
    python adversarial.py [--num_examples 6] [--epsilon 0.05]
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from pathlib import Path

from config import (
    DEVICE, CHECKPOINT_DIR, FIGURE_DIR, BASELINE_BATCH_SIZE,
    ADVERSARIAL_EPSILONS, ADVERSARIAL_DEFENSE_TOP_K,
    IMAGENET_MEAN, IMAGENET_STD, TORCH_LOAD_KWARGS,
)
from dataset import get_dataloaders
from models.baseline import BaselineModel
from models.cbm import ConceptBottleneckModel

matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers (ported from code/fgsm.py and code/laser_attach.py) ─────

def denormalize(tensor):
    """Reverse ImageNet normalization for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def fgsm_attack(image, epsilon, data_grad):
    """FGSM perturbation in normalized space. Ported from code/fgsm.py:47-53."""
    perturbed = image + epsilon * data_grad.sign()
    return perturbed.detach()


# ── Core Attack Functions ──────────────────────────────────────────────

def attack_baseline(model, image, label, epsilon, device):
    """FGSM attack on Baseline model. Returns perturbed image."""
    image = image.clone().detach().to(device)
    image.requires_grad = True
    logits = model(image.unsqueeze(0))
    loss = F.cross_entropy(logits, label.unsqueeze(0))
    model.zero_grad()
    loss.backward()
    perturbed = fgsm_attack(image, epsilon, image.grad.data)
    return perturbed


def attack_cbm(model, image, label, epsilon, device):
    """FGSM attack on CBM. Returns (perturbed_image, perturbed_concepts)."""
    image = image.clone().detach().to(device)
    image.requires_grad = True
    _, class_logits = model(image.unsqueeze(0))
    loss = F.cross_entropy(class_logits, label.unsqueeze(0))
    model.zero_grad()
    loss.backward()
    perturbed = fgsm_attack(image, epsilon, image.grad.data)
    with torch.no_grad():
        new_concepts, _ = model(perturbed.unsqueeze(0))
    return perturbed, new_concepts.squeeze()


# ── Batch Evaluation ──────────────────────────────────────────────────

def evaluate_robustness(baseline, cbm, loader, device, epsilons):
    """Sweep epsilon values and record accuracy for both models."""
    results = {"epsilons": epsilons, "baseline_accs": [], "cbm_accs": []}

    for eps in epsilons:
        bl_correct, cbm_correct, total = 0, 0, 0
        for images, concepts, labels in tqdm(loader, desc=f"Eps={eps:.3f}", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            concepts = concepts.to(device)
            batch_size = images.size(0)

            for i in range(batch_size):
                # Baseline attack
                if eps > 0:
                    perturbed = attack_baseline(baseline, images[i], labels[i], eps, device)
                    with torch.no_grad():
                        pred = baseline(perturbed.unsqueeze(0)).argmax(1).item()
                else:
                    with torch.no_grad():
                        pred = baseline(images[i:i+1]).argmax(1).item()
                bl_correct += int(pred == labels[i].item())

                # CBM attack
                if eps > 0:
                    perturbed_img, _ = attack_cbm(cbm, images[i], labels[i], eps, device)
                    with torch.no_grad():
                        _, cbm_logits = cbm(perturbed_img.unsqueeze(0))
                        pred = cbm_logits.argmax(1).item()
                else:
                    with torch.no_grad():
                        _, cbm_logits = cbm(images[i:i+1])
                        pred = cbm_logits.argmax(1).item()
                cbm_correct += int(pred == labels[i].item())

            total += batch_size

        results["baseline_accs"].append(bl_correct / total * 100)
        results["cbm_accs"].append(cbm_correct / total * 100)
        print(f"  Epsilon={eps:.3f}: Baseline={bl_correct/total*100:.2f}%, CBM={cbm_correct/total*100:.2f}%")

    return results


# ── Concept Stability Analysis ─────────────────────────────────────────

def concept_stability_analysis(cbm, loader, device, epsilon, attr_names):
    """Measure per-concept vulnerability under FGSM attack."""
    all_changes = []
    cbm.eval()

    for images, concepts, _ in tqdm(loader, desc="Stability", leave=False):
        images = images.to(device)
        for i in range(images.size(0)):
            with torch.no_grad():
                clean_concepts, _ = cbm(images[i:i+1])
                clean_concepts = clean_concepts.squeeze()

            perturbed, perturbed_concepts = attack_cbm(
                cbm, images[i], torch.tensor(0), epsilon, device
            )
            change = (perturbed_concepts - clean_concepts).abs().cpu().numpy()
            all_changes.append(change)

    all_changes = np.stack(all_changes)
    mean_change = all_changes.mean(axis=0)

    results = []
    for idx in range(len(attr_names)):
        results.append((idx, attr_names[idx], mean_change[idx]))
    results.sort(key=lambda x: x[2], reverse=True)
    return results


# ── Defense via Concept Intervention ───────────────────────────────────

def defense_intervention(cbm, loader, device, epsilon, top_k_values):
    """After FGSM attack, correct top-k perturbed concepts with ground truth."""
    results = {k: [] for k in top_k_values}

    for images, concepts, labels in tqdm(loader, desc="Defense", leave=False):
        images = images.to(device)
        concepts = concepts.to(device)
        labels = labels.to(device)

        for i in range(images.size(0)):
            # Get clean predicted concepts
            with torch.no_grad():
                clean_concepts, clean_logits = cbm(images[i:i+1])
                clean_concepts = clean_concepts.squeeze()

            # Attack
            perturbed_img, perturbed_concepts = attack_cbm(
                cbm, images[i], labels[i], epsilon, device
            )
            concept_diff = (perturbed_concepts - clean_concepts).abs()

            for k in top_k_values:
                if k == 0:
                    # No defense — use perturbed concepts directly
                    with torch.no_grad():
                        _, logits = cbm(perturbed_img.unsqueeze(0))
                    pred = logits.argmax(1).item()
                else:
                    # Correct top-k most perturbed concepts with ground truth
                    _, topk_idx = concept_diff.topk(min(k, len(concept_diff)))
                    corrected = perturbed_concepts.clone()
                    corrected[topk_idx] = concepts[i][topk_idx].to(device)
                    with torch.no_grad():
                        _, logits = cbm(perturbed_img.unsqueeze(0),
                                        intervene_concepts=corrected.unsqueeze(0))
                    pred = logits.argmax(1).item()

                results[k].append(int(pred == labels[i].item()))

    return {k: np.mean(v) * 100 for k, v in results.items()}


# ── Visualization Functions ────────────────────────────────────────────

def plot_epsilon_accuracy_curve(results, save_path):
    """Line plot: epsilon vs accuracy for Baseline and CBM."""
    fig, ax = plt.subplots(figsize=(10, 6))
    eps = results["epsilons"]
    ax.plot(eps, results["baseline_accs"], "o-", label="Baseline (ResNet-18)", color="#3498db", linewidth=2)
    ax.plot(eps, results["cbm_accs"], "s-", label="CBM", color="#e74c3c", linewidth=2)
    ax.set_xlabel("FGSM Epsilon", fontsize=13)
    ax.set_ylabel("Classification Accuracy (%)", fontsize=13)
    ax.set_title("Adversarial Robustness: Baseline vs CBM", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_attack_examples(baseline, cbm, test_dataset, device, epsilon,
                         class_names, save_path, num_examples=6):
    """Side-by-side: original vs attacked for both models.
    Color-coded titles from laser_attach.py pattern."""
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, num_examples * 3))

    for row in range(num_examples):
        image, concepts, label = test_dataset[row]
        img_tensor = image.to(device)
        true_class = class_names.get(label, f"Class {label}")

        # Panel 1: Original
        img = denormalize(image)
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"Original: {true_class}", fontsize=11, color="green")
        axes[row, 0].axis("off")

        # Panel 2: Baseline attacked
        with torch.no_grad():
            bl_clean = baseline(img_tensor.unsqueeze(0)).argmax(1).item()
        perturbed = attack_baseline(baseline, img_tensor, torch.tensor(label), epsilon, device)
        with torch.no_grad():
            bl_pred = baseline(perturbed.unsqueeze(0)).argmax(1).item()
            bl_conf = F.softmax(baseline(perturbed.unsqueeze(0)), 1).max().item()

        bl_correct = bl_pred == label
        axes[row, 1].imshow(denormalize(perturbed.cpu()))
        axes[row, 1].set_title(
            f"Baseline: {class_names.get(bl_pred, '?')} ({bl_conf:.1%})",
            fontsize=11, color="green" if bl_correct else "red")
        axes[row, 1].axis("off")

        # Panel 3: CBM attacked
        with torch.no_grad():
            _, cbm_clean = cbm(img_tensor.unsqueeze(0))
            cbm_clean_pred = cbm_clean.argmax(1).item()
        perturbed_cbm, _ = attack_cbm(cbm, img_tensor, torch.tensor(label), epsilon, device)
        with torch.no_grad():
            _, cbm_logits = cbm(perturbed_cbm.unsqueeze(0))
            cbm_pred = cbm_logits.argmax(1).item()
            cbm_conf = F.softmax(cbm_logits, 1).max().item()

        cbm_correct = cbm_pred == label
        axes[row, 2].imshow(denormalize(perturbed_cbm.cpu()))
        axes[row, 2].set_title(
            f"CBM: {class_names.get(cbm_pred, '?')} ({cbm_conf:.1%})",
            fontsize=11, color="green" if cbm_correct else "red")
        axes[row, 2].axis("off")

    plt.suptitle(f"FGSM Attack Examples (epsilon={epsilon})", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_concept_vulnerability(stability_results, save_path, top_k=20):
    """Sorted horizontal bar chart of most vulnerable concepts.
    Pattern from code/stiker_attach.py:31-48."""
    top = stability_results[:top_k]
    names = [r[1][:30] for r in top]
    values = [r[2] for r in top]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color="#e74c3c", height=0.8)
    ax.invert_yaxis()
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean Absolute Change", fontsize=12)
    ax.set_title(f"Top-{top_k} Most Vulnerable Concepts under FGSM", fontsize=14)

    for i, v in enumerate(values):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_concept_heatmap(cbm, loader, device, epsilon, attr_names, save_path, top_k=30):
    """Heatmap: clean vs perturbed concept activations for sample images."""
    cbm.eval()
    clean_all, perturbed_all = [], []
    n_samples = 8

    for idx, (images, concepts, _) in enumerate(loader):
        images = images.to(device)
        for i in range(min(images.size(0), n_samples - len(clean_all))):
            with torch.no_grad():
                clean_c, _ = cbm(images[i:i+1])
            _, perturbed_c = attack_cbm(cbm, images[i], torch.tensor(0), epsilon, device)
            clean_all.append(clean_c.squeeze().cpu().numpy())
            perturbed_all.append(perturbed_c.squeeze().cpu().numpy())
        if len(clean_all) >= n_samples:
            break

    clean_arr = np.stack(clean_all)
    perturbed_arr = np.stack(perturbed_all)
    diff = np.abs(perturbed_arr - clean_arr)

    # Select top-k concepts by mean change
    mean_diff = diff.mean(axis=0)
    top_idx = np.argsort(mean_diff)[-top_k:]

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(diff[:, top_idx].T, aspect="auto", cmap="Reds")
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([attr_names[i][:25] for i in top_idx], fontsize=7)
    ax.set_xlabel("Sample Image", fontsize=12)
    ax.set_ylabel("Concept", fontsize=12)
    ax.set_title(f"Concept Perturbation Heatmap (eps={epsilon})", fontsize=14)
    plt.colorbar(im, ax=ax, label="|Concept Change|")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_defense_recovery(defense_results, save_path, clean_acc, attacked_acc):
    """Line chart: number of corrected concepts vs recovered accuracy."""
    k_vals = sorted(defense_results.keys())
    accs = [defense_results[k] for k in k_vals]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_vals, accs, "o-", color="#2ecc71", linewidth=2, markersize=8)
    ax.axhline(y=clean_acc, color="blue", linestyle="--", alpha=0.5, label=f"Clean accuracy ({clean_acc:.1f}%)")
    ax.axhline(y=attacked_acc, color="red", linestyle="--", alpha=0.5, label=f"Attacked accuracy ({attacked_acc:.1f}%)")
    ax.set_xlabel("Number of Corrected Concepts", fontsize=13)
    ax.set_ylabel("Classification Accuracy (%)", fontsize=13)
    ax.set_title("Concept Intervention as Defense against FGSM", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=6)
    parser.add_argument("--epsilon", type=float, default=0.05)
    args = parser.parse_args()

    train_loader, test_loader, dataset = get_dataloaders(
        batch_size=BASELINE_BATCH_SIZE
    )
    attr_names = dataset.attr_names
    class_names = dataset.class_names
    num_concepts = dataset.num_concepts
    num_classes = dataset.num_classes
    test_dataset = test_loader.dataset

    # Load models
    baseline = BaselineModel(num_classes=num_classes).to(DEVICE)
    baseline.load_state_dict(
        torch.load(CHECKPOINT_DIR / "baseline_best.pth", map_location=DEVICE, **TORCH_LOAD_KWARGS)
    )
    baseline.eval()

    cbm = ConceptBottleneckModel(num_concepts, num_classes).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_DIR / "cbm_best.pth", map_location=DEVICE, **TORCH_LOAD_KWARGS)
    cbm.concept_predictor.load_state_dict(ckpt["concept_model"])
    cbm.label_predictor.load_state_dict(ckpt["label_model"])
    cbm.eval()

    print("=" * 60)
    print("ADVERSARIAL ROBUSTNESS ANALYSIS")
    print("=" * 60)

    # 1. Epsilon sweep
    print("\n[1/5] Epsilon sweep...")
    robust_results = evaluate_robustness(baseline, cbm, test_loader, DEVICE, ADVERSARIAL_EPSILONS)
    plot_epsilon_accuracy_curve(robust_results, FIGURE_DIR / "adversarial_epsilon_curve.png")

    # 2. Attack examples
    print("\n[2/5] Attack examples...")
    plot_attack_examples(baseline, cbm, test_dataset, DEVICE, args.epsilon,
                         class_names, FIGURE_DIR / "adversarial_examples.png",
                         num_examples=args.num_examples)

    # 3. Concept vulnerability
    print("\n[3/5] Concept vulnerability analysis...")
    stability = concept_stability_analysis(cbm, test_loader, DEVICE, args.epsilon, attr_names)
    plot_concept_vulnerability(stability, FIGURE_DIR / "adversarial_concept_vulnerability.png")

    # 4. Concept heatmap
    print("\n[4/5] Concept perturbation heatmap...")
    plot_concept_heatmap(cbm, test_loader, DEVICE, args.epsilon, attr_names,
                         FIGURE_DIR / "adversarial_concept_heatmap.png")

    # 5. Defense recovery
    print("\n[5/5] Defense via concept intervention...")
    defense = defense_intervention(cbm, test_loader, DEVICE, args.epsilon,
                                   ADVERSARIAL_DEFENSE_TOP_K)
    clean_acc = robust_results["cbm_accs"][0]  # epsilon=0
    attacked_acc = defense[0]  # k=0 means no correction
    plot_defense_recovery(defense, FIGURE_DIR / "adversarial_defense_recovery.png",
                          clean_acc, attacked_acc)

    # Save results
    torch.save({
        "robustness": robust_results,
        "defense": defense,
        "epsilon": args.epsilon,
    }, FIGURE_DIR / "adversarial_results.pth")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for i, eps in enumerate(robust_results["epsilons"]):
        print(f"  eps={eps:.3f}: Baseline={robust_results['baseline_accs'][i]:.2f}%, "
              f"CBM={robust_results['cbm_accs'][i]:.2f}%")
    print(f"\nDefense recovery (eps={args.epsilon}):")
    for k in sorted(defense.keys()):
        print(f"  Correct top-{k} concepts: {defense[k]:.2f}%")
    print(f"\nAll figures saved to {FIGURE_DIR}/")


if __name__ == "__main__":
    main()
