"""
Global configuration for Concept Bottleneck Model experiment.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
# Standard CUB-200-2011 tgz extracts to CUB_200_2011/
_CUB_DIR = BASE_DIR / "data" / "CUB_200_2011"
if not _CUB_DIR.exists():
    _CUB_DIR = BASE_DIR / "data" / "CUB-200-2011"  # fallback for renamed dir
DATA_DIR = _CUB_DIR
# Attribute names file is at data/ level (outside CUB subdir)
ATTR_NAMES_FILE = BASE_DIR / "data" / "attributes.txt"
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"

# CUB dataset annotation files
IMAGE_LIST_FILE = DATA_DIR / "images.txt"
TRAIN_TEST_SPLIT_FILE = DATA_DIR / "train_test_split.txt"
CLASS_LABELS_FILE = DATA_DIR / "image_class_labels.txt"
CLASSES_FILE = DATA_DIR / "classes.txt"
ATTRIBUTES_FILE = DATA_DIR / "attributes" / "image_attribute_labels.txt"
ATTRIBUTE_NAMES_FILE = DATA_DIR / "attributes" / "attributes.txt"

# ── Selected Classes (24 species) ──────────────────────────────────────
# Maps original 1-indexed class IDs to local 0-indexed IDs
SELECTED_CLASSES = {
    16: 0,   # Ovenbird
    17: 1,   # Groove-billed Ani
    22: 2,   # Sayornis
    36: 3,   # Northern Flicker
    47: 4,   # American Robin
    49: 5,   # European Starling
    55: 6,   # Purple Finch
    62: 7,   # American Goldfinch
    63: 8,   # House Sparrow
    68: 9,   # Cliff Swallow
    73: 10,  # Blue Jay
    85: 11,  # Northern Mockingbird
    96: 12,  # Northern Cardinal
    100: 13, # Brown-headed Cowbird
    104: 14, # Baltimore Oriole
    112: 15, # Great Crested Flycatcher
    122: 16, # White-breasted Nuthatch
    124: 17, # House Wren
    134: 18, # Cedar Waxwing
    161: 19, # Mourning Warbler
    166: 20, # American Redstart
    178: 21, # Blue-headed Vireo
    189: 22, # Pine Siskin
    196: 23, # House Finch
}

NUM_CLASSES = len(SELECTED_CLASSES)  # 24

# ── Attribute Filtering ────────────────────────────────────────────────
MIN_ATTRIBUTE_VARIANCE = 0.05  # Remove attributes with near-zero variance

# ── Image Preprocessing ────────────────────────────────────────────────
IMAGE_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Training Hyperparameters ───────────────────────────────────────────

# Baseline model
BASELINE_LR = 1e-3
BASELINE_MOMENTUM = 0.9
BASELINE_WEIGHT_DECAY = 1e-4
BASELINE_EPOCHS = 20
BASELINE_BATCH_SIZE = 16

# Concept predictor (Stage 1)
CONCEPT_LR = 1e-4
CONCEPT_WEIGHT_DECAY = 1e-4
CONCEPT_EPOCHS = 20
CONCEPT_BATCH_SIZE = 16

# Label predictor (Stage 2)
LABEL_LR = 1e-3
LABEL_WEIGHT_DECAY = 1e-4
LABEL_EPOCHS = 20
LABEL_BATCH_SIZE = 16
LABEL_L1_LAMBDA = 0.01  # L1 regularization for sparsity

# ── Device ─────────────────────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print(f"[GPU] Using {torch.cuda.get_device_name(0)}")
    print(f"[GPU] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("[GPU] CUDA not available, using CPU")

USE_AMP = torch.cuda.is_available()  # Mixed precision only on GPU

# PyTorch >= 2.6 defaults to weights_only=True; our checkpoints contain dicts
TORCH_LOAD_KWARGS = {"weights_only": False}
