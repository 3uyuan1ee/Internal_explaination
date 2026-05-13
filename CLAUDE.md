# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Concept Bottleneck Model (CBM) intervention experiment for the Explainable AI course at BIT. Focuses on expert-in-the-loop concept intervention — quantifying how efficiently CBM intervention corrects errors on CUB-200-2011 bird dataset (50 selected species, InceptionV3 backbone).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full experiment pipeline (4 stages)
bash run_all.sh

# Individual stages
python scripts/train_concept.py       # Stage 1: Concept predictor (image → concepts)
python scripts/train_label.py         # Stage 2: Label predictor (concepts → classes)
python scripts/run_interventions.py   # Stage 3: Intervention experiments 1-4
python scripts/visualize.py           # Stage 4: Generate 6 figures
```

Outputs go to `outputs/` (checkpoints, figures, results).

## Project Structure

```
cbm/                    # Importable library package
  config.py             # Paths, hyperparameters, 50-class selection, device settings
  dataset.py            # CUB-200-2011 loader with variance-based attribute filtering
  utils.py              # AverageMeter, compute_attribute_imbalance
  models/
    concept_predictor.py # Image → concept probabilities (InceptionV3, partial freeze)
    label_predictor.py  # Concepts → class logits (linear, interpretable)
scripts/                # Executable entry points
  train_concept.py      # Train concept predictor
  train_label.py        # Train label predictor on predicted concepts
  run_interventions.py  # Experiments 1-4 (error attribution, strategies, minimal, noisy)
  visualize.py          # Generate 6 figures from experiment results
data/                   # CUB-200-2011 dataset (gitignored)
outputs/                # Training outputs (gitignored)
```

## Architecture

Two-stage Concept Bottleneck Model:

```
Image → [ConceptPredictor] → concept probs (sigmoid) → [LabelPredictor] → class logits
         InceptionV3 (partial freeze)  136 filtered attrs    Linear + L1 sparse
```

**ConceptPredictor** (`cbm/models/concept_predictor.py`): InceptionV3 backbone (pretrained, frozen up to Mixed_6e) → 2048-dim features → concept logits → sigmoid probabilities. Trained with BCEWithLogitsLoss (pos_weight for imbalance) on CUB attribute annotations.

**LabelPredictor** (`cbm/models/label_predictor.py`): Pure linear layer from concepts to class logits. L1 regularization for sparsity. Weight matrix [num_classes, num_concepts] is directly interpretable.

## Experiments

1. **Error Attribution**: Decompose errors into concept-attributable vs label predictor
2. **Strategy Comparison**: 5 intervention strategies (Random, Uncertainty, Importance, Greedy Oracle, Error-Targeted) across k values
3. **Minimal Intervention**: Greedy per-sample search for minimum concepts to correct
4. **Noisy Experts**: Robustness to noise levels and budget constraints

## Key Design Decisions

- **50 of 200 classes**: Selected with fixed seed=42 in `cbm/dataset.py` `select_50_classes()`
- **Attribute filtering**: 312 raw attributes filtered by variance (`MIN_ATTRIBUTE_VARIANCE=0.05`), yielding ~136 concepts
- **Confidence threshold**: Only CUB attribute annotations with certainty >= 3 are used
- **Sequential training**: Stage 1 trains X→C, Stage 2 trains Ĉ→Y (using predicted concepts, not GT)
- **GPU auto-detection**: `cbm/config.py` enables AMP and cuDNN benchmark when CUDA is available
- **Checkpoint loading**: Uses `weights_only=False` (PyTorch >= 2.6 compat)

## Data

Expects CUB-200-2011 dataset at `data/CUB_200_2011/` (or `data/CUB-200-2011/` as fallback). Attribute names at `data/attributes.txt`.
