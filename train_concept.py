"""
Stage 1: Train the concept predictor.

Uses binary cross-entropy loss with CUB attribute annotations.
Saves the best concept predictor checkpoint.

Usage:
    python train_concept.py
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from config import (
    DEVICE, CONCEPT_LR, CONCEPT_WEIGHT_DECAY,
    CONCEPT_EPOCHS, CONCEPT_BATCH_SIZE, CHECKPOINT_DIR, USE_AMP,
)
from dataset import get_dataloaders
from models.concept_predictor import ConceptPredictor


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

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

        total_loss += loss.item() * images.size(0)
        preds = (probs > 0.5).float()
        correct += (preds == concepts).sum().item()
        total += concepts.numel()

    return total_loss / len(loader.dataset), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_probs, all_targets = [], []

    for images, concepts, _ in loader:
        images, concepts = images.to(device), concepts.to(device)
        probs, _ = model(images)
        preds = (probs > 0.5).float()
        correct += (preds == concepts).sum().item()
        total += concepts.numel()
        all_probs.append(probs.cpu())
        all_targets.append(concepts.cpu())

    concept_acc = correct / total
    # Per-concept accuracy
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

    model = ConceptPredictor(num_concepts=num_concepts).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=CONCEPT_LR, weight_decay=CONCEPT_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=CONCEPT_EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_acc = 0.0
    for epoch in range(1, CONCEPT_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler
        )
        test_acc, per_concept = evaluate(model, test_loader, DEVICE)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{CONCEPT_EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Concept Acc: {train_acc:.4f} | "
              f"Test Concept Acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_concepts": num_concepts,
                "concept_acc": test_acc,
                "per_concept_acc": per_concept,
            }, CHECKPOINT_DIR / "concept_predictor_best.pth")
            print(f"  → Saved best concept predictor (acc={best_acc:.4f})")

    print(f"\nBest concept accuracy: {best_acc:.4f}")

    # Print top-5 easiest and hardest concepts
    _, per_concept = evaluate(model, test_loader, DEVICE)
    sorted_idx = per_concept.argsort()
    print("\n--- Easiest concepts ---")
    for i in sorted_idx[-5:].flip(0):
        print(f"  {dataset.attr_names[i]:40s} acc={per_concept[i]:.3f}")
    print("--- Hardest concepts ---")
    for i in sorted_idx[:5]:
        print(f"  {dataset.attr_names[i]:40s} acc={per_concept[i]:.3f}")


if __name__ == "__main__":
    main()
