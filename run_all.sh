#!/bin/bash
# CBM Intervention Experiment Pipeline
# Usage: bash run_all.sh

set -e

echo "=========================================="
echo "Stage 1: Train Concept Predictor (X -> C)"
echo "=========================================="
python scripts/train_concept.py

echo ""
echo "=========================================="
echo "Stage 2: Train Label Predictor (C -> Y)"
echo "=========================================="
python scripts/train_label.py

echo ""
echo "=========================================="
echo "Stage 3: Run Intervention Experiments 1-4"
echo "=========================================="
python scripts/run_interventions.py

echo ""
echo "=========================================="
echo "Stage 4: Generate Visualization Figures"
echo "=========================================="
python scripts/visualize.py

echo ""
echo "All done! Check outputs/ for results."
