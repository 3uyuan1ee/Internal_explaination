"""
Stage 1: Train the concept predictor.

Uses binary cross-entropy loss with CUB attribute annotations.
Includes weighted loss for imbalanced attributes (ported from official CBM).

Usage:
    python train_concept.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from cbm.config import (
    DEVICE, CONCEPT_LR, CONCEPT_WEIGHT_DECAY,
    CONCEPT_EPOCHS, CONCEPT_BATCH_SIZE, CHECKPOINT_DIR, USE_AMP,
    EARLY_STOP_PATIENCE,
)
from cbm.dataset import get_dataloaders
from cbm.models.concept_predictor import ConceptPredictor
from cbm.utils import AverageMeter, compute_attribute_imbalance


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, concepts, _ in tqdm(loader, desc="Train", leave=False):
        images, concepts = images.to(device), concepts.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                probs, logits = model(images)
                loss = criterion(logits, concepts)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            probs, logits = model(images)
            loss = criterion(logits, concepts)
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        preds = (probs > 0.5).float()
        correct = (preds == concepts).sum().item()
        acc_meter.update(correct / concepts.numel() * 100, images.size(0))

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    acc_meter = AverageMeter()
    all_probs, all_targets = [], []

    for images, concepts, _ in loader:
        images, concepts = images.to(device), concepts.to(device)
        probs, _ = model(images)
        preds = (probs > 0.5).float()
        correct = (preds == concepts).sum().item()
        acc_meter.update(correct / concepts.numel() * 100, images.size(0))
        all_probs.append(probs.cpu())
        all_targets.append(concepts.cpu())

    concept_acc = acc_meter.avg
    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)
    per_concept_acc = ((all_probs > 0.5).float() == all_targets).float().mean(0)

    return concept_acc, per_concept_acc


def main():
    print(f"Device: {DEVICE}")
    train_loader, test_loader, dataset = get_dataloaders(
        batch_size=CONCEPT_BATCH_SIZE
    )
    num_concepts = dataset.num_concepts
    print(f"Train: {len(train_loader.dataset)} | Test: {len(test_loader.dataset)}")
    print(f"Num concepts: {num_concepts}")

    # Compute class imbalance for weighted loss (from official CBM)
    all_concepts = []
    for _, concepts, _ in train_loader:
        all_concepts.append(concepts)
    all_concepts = torch.cat(all_concepts).numpy()
    pos_weight = compute_attribute_imbalance(all_concepts).to(DEVICE)
    print(f"Attribute pos_weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")

    model = ConceptPredictor(num_concepts=num_concepts).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total:.1%})")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=CONCEPT_LR, weight_decay=CONCEPT_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=CONCEPT_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_acc = 0.0
    best_epoch = 0
    history = {"epochs": [], "train_loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(1, CONCEPT_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler
        )
        test_acc, per_concept = evaluate(model, test_loader, DEVICE)
        scheduler.step()

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch:3d}/{CONCEPT_EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Concept Acc: {train_acc:.2f}% | "
              f"Test Concept Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_concepts": num_concepts,
                "concept_acc": test_acc,
                "per_concept_acc": per_concept,
            }, CHECKPOINT_DIR / "concept_predictor_best.pth")
            print(f"  -> Saved best concept predictor (acc={best_acc:.2f}%)")

        # Early stopping
        if epoch - best_epoch >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch} (best was {best_epoch})")
            break

    torch.save(history, CHECKPOINT_DIR / "concept_history.pth")

    print(f"\nBest concept accuracy: {best_acc:.2f}%")

    # Compute per-concept AUC
    from sklearn.metrics import roc_auc_score
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for images, concepts, _ in test_loader:
            images = images.to(DEVICE)
            probs, _ = model(images)
            all_probs.append(probs.cpu())
            all_targets.append(concepts)
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()

    per_concept_auc = np.zeros(all_probs.shape[1])
    for j in range(all_probs.shape[1]):
        if len(np.unique(all_targets[:, j])) > 1:
            per_concept_auc[j] = roc_auc_score(all_targets[:, j], all_probs[:, j])
        else:
            per_concept_auc[j] = float("nan")
    mean_auc = np.nanmean(per_concept_auc)
    print(f"Mean per-concept AUC: {mean_auc:.4f}")

    # Save AUC in checkpoint
    ckpt = torch.load(CHECKPOINT_DIR / "concept_predictor_best.pth",
                       map_location="cpu", **{"weights_only": False})
    ckpt["per_concept_auc"] = per_concept_auc
    ckpt["mean_auc"] = mean_auc
    torch.save(ckpt, CHECKPOINT_DIR / "concept_predictor_best.pth")

    # Print top-5 easiest and hardest concepts (by AUC)
    _, per_concept = evaluate(model, test_loader, DEVICE)
    sorted_idx = per_concept.argsort()
    sorted_auc_idx = np.argsort(per_concept_auc)
    print("\n--- Easiest concepts (by AUC) ---")
    for i in sorted_auc_idx[-5:][::-1]:
        print(f"  {dataset.attr_names[i]:40s} AUC={per_concept_auc[i]:.3f}  acc={per_concept[i]:.3f}")
    print("--- Hardest concepts (by AUC) ---")
    for i in sorted_auc_idx[:5]:
        valid = " " if not np.isnan(per_concept_auc[i]) else " (single-class)"
        print(f"  {dataset.attr_names[i]:40s} AUC={per_concept_auc[i]:.3f}  acc={per_concept[i]:.3f}{valid}")


if __name__ == "__main__":
    main()
