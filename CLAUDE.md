# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Concept Bottleneck Model (CBM) implementation for the Explainable AI course at BIT. Implements intrinsic explainability via concept-based models on CUB-200-2011 bird dataset (24 selected species), compared against post-hoc Grad-CAM explanations on a black-box ResNet-18 baseline.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full experiment pipeline (all 7 stages sequentially)
bash run_all.sh

# Individual stages
python scripts/train_baseline.py          # Stage 1: Black-box ResNet-18
python scripts/train_concept.py           # Stage 2: Concept predictor (image → concepts)
python scripts/train_label.py             # Stage 3: Label predictor (concepts → classes)
python scripts/evaluate.py                # Stage 4: Accuracy + intervention experiments
python scripts/explain.py --num_images 8  # Stage 5: CBM concept explanations + Grad-CAM
python scripts/analyze.py                 # Stage 6: Confusion matrices, weight heatmaps, Pareto curves
python scripts/adversarial.py --num_examples 6 --epsilon 0.05  # Stage 7: FGSM robustness analysis
```

Outputs go to `outputs/` (checkpoints, figures, logs).

## Project Structure

```
cbm/                    # Importable library package
  config.py             # Paths, hyperparameters, selected 24 classes, device settings
  dataset.py            # CUB-200-2011 loader with attribute filtering
  utils.py              # AverageMeter, compute_attribute_imbalance
  models/               # Model architectures
    baseline.py         # ResNet-18 black-box
    concept_predictor.py # Image → concept probabilities (ResNet-18 encoder)
    label_predictor.py  # Concepts → class logits (sparse linear/MLP)
    cbm.py              # Full CBM pipeline with intervention support
scripts/                # Executable entry points (train, evaluate, explain, analyze, adversarial)
data/                   # CUB-200-2011 dataset (gitignored)
outputs/                # Training outputs (gitignored)
docs/                   # Experiment design documentation
presentation/           # Node.js PowerPoint generator
```

## Architecture

Two-stage Concept Bottleneck Model vs. end-to-end baseline:

```
Baseline:  Image → ResNet-18 → 24 class logits (black-box)
CBM:       Image → [ConceptPredictor] → concept probs (sigmoid) → [LabelPredictor] → class logits
                   ResNet-18 encoder         312→filtered attrs      Linear/MLP + L1 sparse
```

**ConceptPredictor** (`cbm/models/concept_predictor.py`): ResNet-18 backbone (pretrained, frozen) → 512-dim features → 256 hidden → concept logits → sigmoid probabilities. Trained with BCE loss on CUB attribute annotations.

**LabelPredictor** (`cbm/models/label_predictor.py`): Sparse linear layer (or MLP with `expand_dim=128`) from concepts to class logits. L1 regularization (`lambda=0.001`) for sparsity. Weight matrix is directly interpretable as concept importance per class.

**ConceptBottleneckModel** (`cbm/models/cbm.py`): Wraps both predictors. Supports `intervene_concepts` parameter for causal intervention experiments (replace predicted concepts with ground truth to measure causal impact).

## Key Design Decisions

- **24 of 200 classes**: Selected in `cbm/config.py` `SELECTED_CLASSES` mapping (original 1-indexed → local 0-indexed)
- **Attribute filtering**: 312 raw attributes filtered by variance (`MIN_ATTRIBUTE_VARIANCE=0.05`) in `cbm/dataset.py`, removing near-constant attributes
- **Confidence threshold**: Only CUB attribute annotations with certainty >= 3 are used
- **GPU auto-detection**: `cbm/config.py` enables AMP and cuDNN benchmark when CUDA is available; falls back to CPU
- **Checkpoint loading**: Uses `weights_only=False` (PyTorch >= 2.6 compat, see `TORCH_LOAD_KWARGS`)
- **sys.path setup**: Scripts in `scripts/` add the project root to `sys.path` so the `cbm` package is importable

## Data

Expects CUB-200-2011 dataset extracted at `data/CUB_200_2011/` (or `data/CUB-200-2011/` as fallback). Attribute name overrides at `data/attributes.txt`. The dataset loader (`cbm/dataset.py`) handles train/test splitting, attribute parsing, and variance filtering automatically.

## Presentation

`presentation/` contains a Node.js script (`create_pptx.js`) using PptxGenJS for generating slides from experiment results. Run with `node presentation/create_pptx.js` after installing `npm install` in that directory.
