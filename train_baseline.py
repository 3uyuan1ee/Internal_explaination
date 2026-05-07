"""
Train the black-box baseline ResNet-18 classifier.

Usage:
    python train_baseline.py
"""

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from config import (
    DEVICE, BASELINE_LR, BASELINE_MOMENTUM, BASELINE_WEIGHT_DECAY,
    BASELINE_EPOCHS, BASELINE_BATCH_SIZE, CHECKPOINT_DIR, USE_AMP,
)
from dataset import get_dataloaders
from models.baseline import BaselineModel


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, _, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    for images, _, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return correct / total


def main():
    print(f"Device: {DEVICE}")
    train_loader, test_loader, dataset = get_dataloaders(
        batch_size=BASELINE_BATCH_SIZE
    )
    print(f"Train: {len(train_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = BaselineModel(num_classes=dataset.num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(), lr=BASELINE_LR,
        momentum=BASELINE_MOMENTUM, weight_decay=BASELINE_WEIGHT_DECAY,
    )
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_acc = 0.0
    for epoch in range(1, BASELINE_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, scaler
        )
        test_acc = evaluate(model, test_loader, DEVICE)
        scheduler.step()

        print(f"Epoch {epoch:3d}/{BASELINE_EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Test Acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), CHECKPOINT_DIR / "baseline_best.pth")
            print(f"  → Saved best model (acc={best_acc:.4f})")

    print(f"\nBest test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
