"""
Naseej — central configuration.

Every tunable knob lives here so the rest of the codebase reads from a single
source of truth. Thresholds, in particular, are meant to be re-calibrated on
local data before any real use (see README §13).
"""

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")          # expects Benign/ and Malignant/
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
OUTPUT_DIR = os.path.join(ROOT, "outputs")           # plots, reports

# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #
# Index order matters: torchvision.ImageFolder assigns labels alphabetically,
# so "Benign" -> 0 and "Malignant" -> 1. Keep this consistent everywhere.
CLASS_NAMES = ["Benign", "Malignant"]
MALIGNANT_INDEX = 1
BENIGN_INDEX = 0

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
# One of: "resnet50", "resnet18", "efficientnet_b0"
BACKBONE = "resnet50"
PRETRAINED = True            # start from ImageNet weights (transfer learning)
FREEZE_BACKBONE = False      # if True, train only the new classification head

# --------------------------------------------------------------------------- #
# Image / preprocessing
# --------------------------------------------------------------------------- #
IMAGE_SIZE = 224
# ImageNet normalisation statistics (the pretrained backbones expect these).
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2              # fraction of TRAIN_DIR held out for validation
SEED = 42

# Recall-prioritised loss. A missed cancer is far costlier than a false alarm,
# so the malignant class is weighted higher in the cross-entropy loss.
# Order matches CLASS_NAMES: [benign_weight, malignant_weight].
CLASS_WEIGHTS = [1.0, 2.0]

# --------------------------------------------------------------------------- #
# Triage thresholds (applied to P(malignant))
# --------------------------------------------------------------------------- #
#   p >= URGENT_THRESHOLD                     -> URGENT
#   REVIEW_THRESHOLD <= p < URGENT_THRESHOLD  -> REVIEW
#   p <  REVIEW_THRESHOLD                     -> ROUTINE
URGENT_THRESHOLD = 0.70
REVIEW_THRESHOLD = 0.30

# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
# Guarded so this module stays importable even before torch is installed
# (e.g. scripts/get_data.py runs as a lightweight layout helper).
try:  # pragma: no cover - trivial environment branch
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:  # pragma: no cover
    DEVICE = "cpu"
