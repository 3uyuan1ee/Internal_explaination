"""Train ConceptPredictor (X -> C) using InceptionV3 on 50 CUB classes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from cbm.config import (
    DEVICE, CONCEPT_LR, CONCEPT_MOMENTUM, CONCEPT_WEIGHT_DECAY,
    CONCEPT_EPOCHS, CONCEPT_BATCH_SIZE, CHECKPOINT_DIR, USE_AMP,
    CONCEPT_SCHEDULER_STEP, CONCEPT_SCHEDULER_GAMMA, EARLY_STOP_PATIENCE,
    RESULTS_DIR,
)
from cbm.dataset import get_dataloaders
from cbm.models.concept_predictor import ConceptPredictor
from cbm.utils import AverageMeter, compute_attribute_imbalance, set_seed


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    for images, concepts, _, _ in tqdm(loader, desc="Train", leave=False):
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
        acc_meter.update((preds == concepts).float().mean().item() * 100, images.size(0))
    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    acc_meter = AverageMeter()
    all_probs, all_targets = [], []
    for images, concepts, _, _ in loader:
        images, concepts = images.to(device), concepts.to(device)
        probs, _ = model(images)
        acc_meter.update((probs > 0.5).float().eq(concepts).float().mean().item() * 100, images.size(0))
        all_probs.append(probs.cpu())
        all_targets.append(concepts.cpu())
    return acc_meter.avg, torch.cat(all_probs), torch.cat(all_targets)


def main():
    set_seed(42)
    print(f"Device: {DEVICE}")
    train_loader, test_loader, dataset = get_dataloaders(batch_size=CONCEPT_BATCH_SIZE)
    num_concepts = dataset.num_concepts
    print(f"Train: {len(train_loader.dataset)} | Test: {len(test_loader.dataset)} | Concepts: {num_concepts}")

    all_c = torch.cat([c for _, c, _, _ in train_loader]).numpy()
    pos_weight = compute_attribute_imbalance(all_c).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = ConceptPredictor(num_concepts).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total:.1%})")

    optimizer = SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONCEPT_LR, momentum=CONCEPT_MOMENTUM, weight_decay=CONCEPT_WEIGHT_DECAY,
    )
    scheduler = StepLR(optimizer, step_size=CONCEPT_SCHEDULER_STEP, gamma=CONCEPT_SCHEDULER_GAMMA)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    best_acc, best_epoch = 0.0, 0
    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, CONCEPT_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, scaler)
        test_acc, _, _ = evaluate(model, test_loader, DEVICE)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch:4d}/{CONCEPT_EPOCHS} | Loss {train_loss:.4f} | "
              f"Train {train_acc:.2f}% | Test {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc, best_epoch = test_acc, epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_concepts": num_concepts,
                "concept_acc": test_acc,
                "selected_classes": dataset.selected_classes,
                "valid_attr_indices": dataset.valid_attr_indices,
                "attr_names": dataset.attr_names,
            }, CHECKPOINT_DIR / "concept_predictor_best.pth")
            print(f"  -> Best: {best_acc:.2f}%")

        if epoch - best_epoch >= EARLY_STOP_PATIENCE:
            print(f"Early stop at epoch {epoch}")
            break

    torch.save(history, CHECKPOINT_DIR / "concept_history.pth")
    print(f"\nBest concept accuracy: {best_acc:.2f}% at epoch {best_epoch}")

    # Per-concept F1/Precision/Recall on test set
    _, test_probs, test_targets = evaluate(model, test_loader, DEVICE)
    test_preds = (test_probs > 0.5).numpy()
    test_targets_np = test_targets.numpy()
    per_concept = {
        "f1": f1_score(test_targets_np, test_preds, average=None, zero_division=0),
        "precision": precision_score(test_targets_np, test_preds, average=None, zero_division=0),
        "recall": recall_score(test_targets_np, test_preds, average=None, zero_division=0),
    }
    mean_f1 = per_concept["f1"].mean()
    print(f"Mean per-concept F1: {mean_f1:.4f}")

    # Save per-concept CSV
    import csv
    csv_path = RESULTS_DIR / "per_concept_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept_idx", "concept_name", "f1", "precision", "recall"])
        for j in range(num_concepts):
            name = dataset.attr_names[j] if j < len(dataset.attr_names) else f"concept_{j}"
            writer.writerow([j, name,
                             f"{per_concept['f1'][j]:.4f}",
                             f"{per_concept['precision'][j]:.4f}",
                             f"{per_concept['recall'][j]:.4f}"])
    print(f"Per-concept metrics saved to {csv_path}")


if __name__ == "__main__":
    main()
