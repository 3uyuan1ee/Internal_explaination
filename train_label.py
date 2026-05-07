"""
Stage 2: Train the label predictor (concepts → classes).

Freezes the concept predictor and trains a sparse linear classifier
on top of predicted concepts. L1 regularization encourages sparsity
for interpretability.

Usage:
    python train_label.py
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from config import (
    DEVICE, LABEL_LR, LABEL_WEIGHT_DECAY,
    LABEL_EPOCHS, LABEL_BATCH_SIZE, LABEL_L1_LAMBDA,
    CHECKPOINT_DIR, TORCH_LOAD_KWARGS, USE_AMP,
)
from dataset import get_dataloaders
from models.concept_predictor import ConceptPredictor
from models.label_predictor import LabelPredictor


def l1_penalty(model):
    return sum(p.abs().sum() for p in model.parameters())


def train_one_epoch(concept_model, label_model, loader, criterion, optimizer, device, scaler=None):
    concept_model.eval()
    label_model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, concepts, labels in tqdm(loader, desc="Train", leave=False):
        labels = labels.to(device)
        optimizer.zero_grad()

        with torch.no_grad():
            pred_concepts, _ = concept_model(images.to(device))

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = label_model(pred_concepts)
                loss = criterion(logits, labels) + LABEL_L1_LAMBDA * l1_penalty(label_model)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = label_model(pred_concepts)
            loss = criterion(logits, labels) + LABEL_L1_LAMBDA * l1_penalty(label_model)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(concept_model, label_model, loader, device):
    concept_model.eval()
    label_model.eval()
    correct, total = 0, 0

    for images, concepts, labels in loader:
        images, labels = images.to(device), labels.to(device)
        pred_concepts, _ = concept_model(images)
        logits = label_model(pred_concepts)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return correct / total


def main():
    print(f"Device: {DEVICE}")
    train_loader, test_loader, dataset = get_dataloaders(
        batch_size=LABEL_BATCH_SIZE
    )
    num_concepts = dataset.num_concepts
    num_classes = dataset.num_classes
    print(f"Train: {len(train_loader.dataset)} | Test: {len(test_loader.dataset)}")
    print(f"Concepts: {num_concepts} | Classes: {num_classes}")

    # Load pre-trained concept predictor
    concept_model = ConceptPredictor(num_concepts=num_concepts).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_DIR / "concept_predictor_best.pth", map_location=DEVICE, **TORCH_LOAD_KWARGS)
    concept_model.load_state_dict(ckpt["model_state_dict"])
    for param in concept_model.parameters():
        param.requires_grad = False
    print("Loaded and frozen concept predictor")

    # Train label predictor
    label_model = LabelPredictor(num_concepts, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(label_model.parameters(), lr=LABEL_LR, weight_decay=LABEL_WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_acc = 0.0
    for epoch in range(1, LABEL_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            concept_model, label_model, train_loader, criterion, optimizer, DEVICE, scaler
        )
        test_acc = evaluate(concept_model, label_model, test_loader, DEVICE)
        scheduler.step()

        # Sparsity: fraction of near-zero weights
        W = label_model.linear.weight.data
        sparsity = (W.abs() < 0.01).float().mean().item()

        print(f"Epoch {epoch:3d}/{LABEL_EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f} | "
              f"Sparsity: {sparsity:.2%}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "label_model": label_model.state_dict(),
                "concept_model": concept_model.state_dict(),
                "num_concepts": num_concepts,
                "num_classes": num_classes,
                "test_acc": test_acc,
                "weight_sparsity": sparsity,
            }, CHECKPOINT_DIR / "cbm_best.pth")
            print(f"  → Saved best CBM (acc={best_acc:.4f})")

    print(f"\nBest test accuracy: {best_acc:.4f}")

    # Print learned concept weights for top-5 concepts per class
    W = label_model.linear.weight.data.cpu()
    print("\n--- Top concepts per class (sample) ---")
    for c in range(min(6, num_classes)):
        _, top_idx = W[c].abs().topk(3)
        top_names = [dataset.attr_names[i] for i in top_idx]
        top_weights = [f"{W[c, i]:.2f}" for i in top_idx]
        print(f"  {dataset.class_names[c]:25s}: "
              f"{list(zip(top_names, top_weights))}")


if __name__ == "__main__":
    main()
