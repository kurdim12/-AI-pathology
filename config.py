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
THRESHOLDS_PATH = os.path.join(OUTPUT_DIR, "thresholds.json")  # calibrated triage cut-offs

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

# Two-phase fine-tuning: train only the new head for WARMUP_EPOCHS (backbone
# frozen) so the random head settles before we perturb the pretrained features,
# then unfreeze and fine-tune the whole network. Set to 0 to disable.
WARMUP_EPOCHS = 2

# Early stopping: stop if validation AUC hasn't improved for this many epochs.
# Set to 0 to disable and always run the full EPOCHS.
EARLY_STOP_PATIENCE = 5

# Mixed-precision training (only used when running on CUDA; ignored on CPU).
USE_AMP = True

# --------------------------------------------------------------------------- #
# Foundation-model upgrade path (optional)
# --------------------------------------------------------------------------- #
# When set, src.model.build_foundation_model can load an open pathology / vision
# foundation backbone via `timm` instead of a torchvision CNN. Examples:
#   FOUNDATION_MODEL = "timm:vit_base_patch16_224"
#   FOUNDATION_MODEL = "timm:convnext_tiny"
# FOUNDATION_WEIGHTS may point at local released weights (e.g. CTransPath/UNI).
FOUNDATION_MODEL = None
FOUNDATION_WEIGHTS = None

# --------------------------------------------------------------------------- #
# Triage thresholds (applied to P(malignant))
# --------------------------------------------------------------------------- #
#   p >= URGENT_THRESHOLD                     -> URGENT
#   REVIEW_THRESHOLD <= p < URGENT_THRESHOLD  -> REVIEW
#   p <  REVIEW_THRESHOLD                     -> ROUTINE
# These are sensible defaults. Run `python -m src.calibrate` after training to
# replace them with data-driven cut-offs that guarantee a sensitivity floor;
# calibrated values are written to THRESHOLDS_PATH and picked up automatically.
URGENT_THRESHOLD = 0.70
REVIEW_THRESHOLD = 0.30

# Calibration target: the minimum malignant sensitivity (recall) the REVIEW
# cut-off must guarantee on the validation set. A missed cancer is the failure
# that matters, so we hold recall high and let specificity float.
TARGET_SENSITIVITY = 0.95

# Test-time augmentation: average predictions over label-preserving views
# (flips / 90° rotations) for steadier probabilities on messy phone images.
TTA_ENABLED = False

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
