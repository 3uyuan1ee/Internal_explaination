#!/bin/bash
# Run the full CBM experiment pipeline on GPU server
set -e

echo "=== Checking GPU ==="
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo ""
echo "=== Stage 1: Training Baseline ==="
python train_baseline.py

echo ""
echo "=== Stage 2: Training Concept Predictor ==="
python train_concept.py

echo ""
echo "=== Stage 3: Training Label Predictor ==="
python train_label.py

echo ""
echo "=== Stage 4: Evaluation ==="
python evaluate.py

echo ""
echo "=== Stage 5: Explanation Generation ==="
python explain.py --num_images 8

echo ""
echo "=== Stage 6: Analysis & Visualization ==="
python analyze.py

echo ""
echo "=== All experiments completed! ==="
ls -la outputs/figures/
