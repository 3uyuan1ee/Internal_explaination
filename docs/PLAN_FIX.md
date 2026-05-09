# Fix Plan: CBM Pipeline Critical Fixes & Enhancements

## Background

Current evaluation results reveal CBM classification accuracy is only **30.4%** (expected ~80%+).
Root causes: label predictor trained with `expand_dim=128` (MLP mode) degraded on predicted concepts;
ConceptPredictor backbone unfrozen causing overfitting; insufficient training epochs.

## Task List (Execution Order)

### P0: Critical Bugs (Blocks Everything)

**Task 1: Freeze ConceptPredictor backbone**
- File: `cbm/models/concept_predictor.py`
- Action: Add `for param in self.encoder.parameters(): param.requires_grad = False` in `__init__`
- Verify: `sum(p.numel() for p in model.parameters() if p.requires_grad)` should only count concept_head params (~130K vs full ~11M)

**Task 2: Fix concept_stability_analysis label=0 bug**
- File: `scripts/adversarial.py` line 150
- Action: Change `torch.tensor(0)` to pass true labels through the loop
- Signature change: `concept_stability_analysis(cbm, loader, device, epsilon, attr_names)` already has access to labels from the loader

**Task 3: Adjust training epochs and early stopping**
- File: `cbm/config.py`
- Changes:
  - `BASELINE_EPOCHS = 50`
  - `CONCEPT_EPOCHS = 40`
  - `LABEL_EPOCHS = 50`
  - `EARLY_STOP_PATIENCE = 10` (allow early stopping to actually trigger)

### P1: Significant Issues

**Task 4: Fix Pareto plot hardcoded explainability scores**
- File: `scripts/analyze.py` function `plot_accuracy_explainability_pareto`
- Action: Replace hardcoded 0.15/0.40/0.90 with computed metrics:
  - Black-box: sparsity score (fraction of near-zero weights in a random probe)
  - Post-hoc: use concept fidelity score as proxy
  - CBM: use `(concept_fidelity + weight_sparsity) / 2` or similar composite
- Keep it simple: use 3 concrete metrics — concept fidelity, weight sparsity ratio, intervention sensitivity

**Task 5: Save training history for curves**
- Files: `scripts/train_baseline.py`, `scripts/train_concept.py`, `scripts/train_label.py`
- Action: Append (epoch, loss, accuracy) to a list each epoch, save as `{model}_history.pth` to CHECKPOINT_DIR
- Format: `{"epochs": [...], "train_loss": [...], "train_acc": [...], "test_acc": [...]}`

**Task 6: Add training curve plotting to analyze.py**
- File: `scripts/analyze.py`
- Action: Add `plot_training_curves()` function that loads all 3 history files and plots:
  - Panel 1: Baseline train/test accuracy vs epoch
  - Panel 2: Concept predictor accuracy vs epoch
  - Panel 3: CBM classification accuracy vs epoch (label predictor)
  - Panel 4: Weight sparsity vs epoch (label predictor)

**Task 7: Add per-concept AUC metric**
- File: `scripts/train_concept.py`
- Action: After training, compute per-concept ROC-AUC using sklearn, print top-5 easiest/hardest by AUC
- Also save AUC array in concept predictor checkpoint

### P2: Training Optimization

**Task 8: Verify frozen backbone training works correctly**
- File: `scripts/train_concept.py`
- Action: After freezing (Task 1), ensure optimizer only receives trainable params
- Add print: `f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"`

**Task 9: Add per-class accuracy breakdown**
- File: `scripts/evaluate.py`
- Action: Add per-class accuracy computation and printing
- Show which classes CBM < Baseline and vice versa

### P3: Selective Enhancements

**Task 10: Concept t-SNE visualization**
- File: `scripts/analyze.py`
- Action: New function `plot_concept_tsne()`:
  1. Run CBM on test set, collect concept_probs + true_labels
  2. Fit t-SNE on concept_probs (perplexity=30)
  3. Scatter plot colored by class
  4. Save to `outputs/figures/concept_tsne.png`

**Task 11: Update run_all.sh if needed**
- Ensure run_all.sh reflects any new scripts or argument changes

## Execution Dependencies

```
Task 1 ──┐
Task 2 ──┤
Task 3 ──┼──→ RETRAIN ALL MODELS ──→ Task 8 (verify) ──→ Tasks 4,5,6,7,9,10 (analysis)
          │
          └──→ code fixes can be done in parallel before retraining
```

## Key Risk: Retraining Required

All P0 fixes (Tasks 1-3) change training behavior. After applying code fixes,
the full pipeline must be retrained from scratch:
```bash
bash run_all.sh
```

## NOT in Scope (Deferred)

- SHAP analysis (high complexity, diminishing returns for course report)
- End-to-end CBM fine-tuning (adds complexity, not critical)
- Concept redundancy analysis / PCA (nice to have, not essential)
- FGSM batch vectorization (speed optimization, correctness is already fixed by Task 2)
