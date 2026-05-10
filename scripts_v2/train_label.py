"""Train LabelPredictor (C_hat -> Y) using predicted concepts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from cbm_v2.config import (
    DEVICE, LABEL_LR, LABEL_WEIGHT_DECAY, LABEL_EPOCHS,
    LABEL_BATCH_SIZE, LABEL_SCHEDULER_STEP, LABEL_SCHEDULER_GAMMA,
    LABEL_L1_LAMBDA, CHECKPOINT_DIR, USE_AMP, EARLY_STOP_PATIENCE,
    TORCH_LOAD_KWARGS,
)
from cbm_v2.dataset import get_dataloaders
from cbm_v2.models.concept_predictor import ConceptPredictor
from cbm_v2.models.label_predictor import LabelPredictor
from cbm_v2.utils import AverageMeter


def main():
    print(f"Device: {DEVICE}")
    train_loader, test_loader, dataset = get_dataloaders(batch_size=LABEL_BATCH_SIZE)
    num_concepts = dataset.num_concepts
    num_classes = dataset.num_classes

    ckpt = torch.load(CHECKPOINT_DIR / "concept_predictor_best.pth",
                       map_location=DEVICE, **TORCH_LOAD_KWARGS)
    concept_model = ConceptPredictor(num_concepts).to(DEVICE)
    concept_model.load_state_dict(ckpt["model_state_dict"])
    concept_model.eval()
    print(f"Concept predictor loaded (acc={ckpt['concept_acc']:.2f}%)")

    @torch.no_grad()
    def get_concepts(loader):
        all_concepts, all_labels = [], []
        for images, _, labels, _ in tqdm(loader, desc="Extracting concepts"):
            probs, _ = concept_model(images.to(DEVICE))
            all_concepts.append(probs.cpu())
            all_labels.append(labels)
        return torch.cat(all_concepts), torch.cat(all_labels)

    print("Extracting training concepts...")
    train_concepts, train_labels = get_concepts(train_loader)
    print("Extracting test concepts...")
    test_concepts, test_labels = get_concepts(test_loader)

    label_model = LabelPredictor(num_concepts, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(label_model.parameters(), lr=LABEL_LR, weight_decay=LABEL_WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=LABEL_SCHEDULER_STEP, gamma=LABEL_SCHEDULER_GAMMA)

    train_concepts = train_concepts.to(DEVICE)
    train_labels = train_labels.to(DEVICE)

    best_acc, best_epoch = 0.0, 0
    batch_size = LABEL_BATCH_SIZE

    for epoch in range(1, LABEL_EPOCHS + 1):
        perm = torch.randperm(len(train_concepts))
        label_model.train()
        correct, total = 0, 0
        epoch_loss = 0.0

        for i in range(0, len(perm), batch_size):
            idx = perm[i:i+batch_size]
            c_batch = train_concepts[idx]
            y_batch = train_labels[idx]
            logits = label_model(c_batch)
            loss = criterion(logits, y_batch) + LABEL_L1_LAMBDA * label_model.linear.weight.abs().sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == y_batch).sum().item()
            total += len(idx)

        scheduler.step()
        train_acc = correct / total * 100

        label_model.eval()
        with torch.no_grad():
            test_logits = label_model(test_concepts.to(DEVICE))
            test_acc = (test_logits.argmax(1) == test_labels.to(DEVICE)).float().mean().item() * 100

        print(f"Epoch {epoch:4d}/{LABEL_EPOCHS} | Loss {epoch_loss/total:.4f} | "
              f"Train {train_acc:.2f}% | Test {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc, best_epoch = test_acc, epoch
            torch.save({
                "model_state_dict": label_model.state_dict(),
                "num_concepts": num_concepts,
                "num_classes": num_classes,
                "label_acc": test_acc,
                "weight_matrix": label_model.weight_matrix.cpu(),
            }, CHECKPOINT_DIR / "label_predictor_best.pth")
            print(f"  -> Best: {best_acc:.2f}%")

        if epoch - best_epoch >= EARLY_STOP_PATIENCE:
            print(f"Early stop at epoch {epoch}")
            break

    print(f"\nBest label accuracy: {best_acc:.2f}% at epoch {best_epoch}")


if __name__ == "__main__":
    main()
