"""
Global configuration for CBM experiment (50 classes, InceptionV3).
"""
import os
import torch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
_CUB_DIR = BASE_DIR / "data" / "CUB_200_2011"
if not _CUB_DIR.exists():
    _CUB_DIR = BASE_DIR / "data" / "CUB-200-2011"
DATA_DIR = _CUB_DIR
ATTR_NAMES_FILE = BASE_DIR / "data" / "attributes.txt"
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

# CUB annotation files
IMAGE_LIST_FILE = DATA_DIR / "images.txt"
TRAIN_TEST_SPLIT_FILE = DATA_DIR / "train_test_split.txt"
CLASS_LABELS_FILE = DATA_DIR / "image_class_labels.txt"
CLASSES_FILE = DATA_DIR / "classes.txt"
ATTRIBUTES_FILE = DATA_DIR / "attributes" / "image_attribute_labels.txt"

# ── 50 Class Selection ────────────────────────────────────────────────
SEED = 42
NUM_CLASSES = 50
SELECTED_CLASSES = None  # {original_1indexed_id: local_0indexed_id}

# ── Attribute Filtering ───────────────────────────────────────────────
MIN_ATTRIBUTE_VARIANCE = 0.05
MIN_CERTAINTY = 3

# ── Image Preprocessing (InceptionV3 uses 299×299) ────────────────────
IMAGE_SIZE = 299
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Concept Predictor Training ────────────────────────────────────────
CONCEPT_LR = 0.01
CONCEPT_MOMENTUM = 0.9
CONCEPT_WEIGHT_DECAY = 4e-5
CONCEPT_EPOCHS = 1000
CONCEPT_BATCH_SIZE = 32
CONCEPT_SCHEDULER_STEP = 300
CONCEPT_SCHEDULER_GAMMA = 0.1
EARLY_STOP_PATIENCE = 100

# ── Label Predictor Training ──────────────────────────────────────────
# Elastic Net regularization: L2 via weight_decay + L1 via L1_LAMBDA
LABEL_LR = 0.001
LABEL_WEIGHT_DECAY = 5e-5       # L2 regularization (weight decay)
LABEL_EPOCHS = 500
LABEL_BATCH_SIZE = 32
LABEL_SCHEDULER_STEP = 500
LABEL_SCHEDULER_GAMMA = 0.1
LABEL_L1_LAMBDA = 0.0001        # L1 regularization for sparse, interpretable weights

# ── Intervention Experiments ──────────────────────────────────────────
INTERVENTION_K_VALUES = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, -1]
RANDOM_TRIALS = 10
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
NOISE_BUDGETS = [5, 10, 20]

# ── Device ────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
TORCH_LOAD_KWARGS = {"weights_only": False}

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print(f"[GPU] {torch.cuda.get_device_name(0)}")
else:
    print("[GPU] CUDA not available, using CPU")

# ── Output dirs ───────────────────────────────────────────────────────
for d in [OUTPUT_DIR, CHECKPOINT_DIR, FIGURE_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
