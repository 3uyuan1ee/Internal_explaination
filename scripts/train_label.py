"""
Stage 2: Train the label predictor (concepts -> classes).

Uses ground truth concept labels for training (oracle mode from official CBM).
Supports MLP label predictor with expand_dim > 0.

Usage:
    python train_label.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from cbm.config import (
    DEVICE, LABEL_LR, LABEL_WEIGHT_DECAY,
    LABEL_EPOCHS, LABEL_BATCH_SIZE, LABEL_L1_LAMBDA,
    LABEL_EXPAND_DIM, CHECKPOINT_DIR, TORCH_LOAD_KWARGS, USE_AMP,
    EARLY_STOP_PATIENCE,
)
from cbm.dataset import get_dataloaders
from cbm.models.concept_predictor import ConceptPredictor
from cbm.models.label_predictor import LabelPredictor
from cbm.utils import AverageMeter


def l1_penalty(model):
    return sum(p.abs().sum() for p in model.parameters())


def train_one_epoch(label_model, loader, criterion, optimizer, device, scaler=None):
    """Train label predictor with ground truth concepts (oracle mode)."""
    label_model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, concepts, labels in tqdm(loader, desc="Train", leave=False):
        concepts = concepts.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = label_model(concepts)
                loss = criterion(logits, labels) + LABEL_L1_LAMBDA * l1_penalty(label_model)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = label_model(concepts)
            loss = criterion(logits, labels) + LABEL_L1_LAMBDA * l1_penalty(label_model)
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item(), labels.size(0))
        correct = (logits.argmax(1) == labels).sum().item()
        acc_meter.update(correct / labels.size(0) * 100, labels.size(0))

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(concept_model, label_model, loader, device):
    """Evaluate full CBM: predicted concepts -> label predictor -> class."""
    concept_model.eval()
    label_model.eval()
    acc_meter = AverageMeter()

    for images, concepts, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        pred_concepts, _ = concept_model(images)
        logits = label_model(pred_concepts)
        correct = (logits.argmax(1) == labels).sum().item()
        acc_meter.update(correct / images.size(0) * 100, images.size(0))

    return acc_meter.avg


def main():
    print(f"Device: {DEVICE}")
    train_loader, test_loader, dataset = get_dataloaders(
        batch_size=LABEL_BATCH_SIZE
    )
    num_concepts = dataset.num_concepts
    num_classes = dataset.num_classes
    print(f"Train: {len(train_loader.dataset)} | Test: {len(test_loader.dataset)}")
    print(f"Concepts: {num_concepts} | Classes: {num_classes}")
    print(f"Label predictor expand_dim: {LABEL_EXPAND_DIM}")

    # Load pre-trained concept predictor (for evaluation only)
    concept_model = ConceptPredictor(num_concepts=num_concepts).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_DIR / "concept_predictor_best.pth", map_location=DEVICE, **TORCH_LOAD_KWARGS)
    concept_model.load_state_dict(ckpt["model_state_dict"])
    for param in concept_model.parameters():
        param.requires_grad = False
    print("Loaded and frozen concept predictor")

    # Train label predictor with GROUND TRUTH concepts (oracle mode)
    label_model = LabelPredictor(num_concepts, num_classes, expand_dim=LABEL_EXPAND_DIM).to(DEVICE)
    print(f"Label predictor parameters: {sum(p.numel() for p in label_model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(label_model.parameters(), lr=LABEL_LR, weight_decay=LABEL_WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_acc = 0.0
    best_epoch = 0
    history = {"epochs": [], "train_loss": [], "train_acc": [], "test_acc": [], "sparsity": []}
    for epoch in range(1, LABEL_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            label_model, train_loader, criterion, optimizer, DEVICE, scaler
        )
        test_acc = evaluate(concept_model, label_model, test_loader, DEVICE)
        scheduler.step()

        # Sparsity report
        if LABEL_EXPAND_DIM > 0:
            W = label_model.fc2.weight.data
        else:
            W = label_model.linear.weight.data
        sparsity = (W.abs() < 0.01).float().mean().item()

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["sparsity"].append(sparsity)

        print(f"Epoch {epoch:3d}/{LABEL_EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | "
              f"Sparsity: {sparsity:.2%}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save({
                "label_model": label_model.state_dict(),
                "concept_model": concept_model.state_dict(),
                "num_concepts": num_concepts,
                "num_classes": num_classes,
                "expand_dim": LABEL_EXPAND_DIM,
                "test_acc": test_acc,
                "weight_sparsity": sparsity,
            }, CHECKPOINT_DIR / "cbm_best.pth")
            print(f"  -> Saved best CBM (acc={best_acc:.2f}%)")

        # Early stopping
        if epoch - best_epoch >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch} (best was {best_epoch})")
            break

    torch.save(history, CHECKPOINT_DIR / "label_history.pth")

    print(f"\nBest test accuracy: {best_acc:.2f}%")

    # Print learned concept weights for top concepts per class
    if LABEL_EXPAND_DIM == 0:
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
