"""
Data loading and augmentation for Naseej.

The headline idea of this file is the **phone-capture augmentation pipeline**.
Commercial pathology AI assumes a pristine whole-slide scan. Naseej is built to
survive a photo taken down an ordinary microscope with a phone, so training
augmentations deliberately simulate those messy conditions: uneven lighting,
motion/defocus blur, colour/white-balance shift, rotation and the perspective
skew of a hand-held phone over an eyepiece.

Expected layout (see scripts/get_data.py)::

    data/train/Benign/*.png
    data/train/Malignant/*.png
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import config


def phone_capture_transforms(image_size: int = config.IMAGE_SIZE) -> transforms.Compose:
    """Training transforms that emulate phone-through-microscope capture.

    Each augmentation maps to a real-world nuisance:
      * RandomResizedCrop          -> framing / zoom variation, partial fields
      * H/V flips, rotation        -> arbitrary slide orientation on the stage
      * ColorJitter                -> staining variability + phone white balance
      * GaussianBlur               -> defocus / motion blur of a hand-held phone
      * RandomPerspective          -> shooting the eyepiece at a slight angle
      * RandomGrayscale (rare)     -> robustness to extreme colour loss
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomGrayscale(p=0.02),
            transforms.ToTensor(),
            transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
        ]
    )


def eval_transforms(image_size: int = config.IMAGE_SIZE) -> transforms.Compose:
    """Clean, deterministic transforms for validation / evaluation / inference."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
        ]
    )


def _check_layout(train_dir: str) -> None:
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            "Arrange the dataset as data/train/Benign/*.png and "
            "data/train/Malignant/*.png (or one folder per subtype for "
            "multi-class), then run `python -m scripts.get_data` to verify."
        )
    subdirs = [d for d in sorted(os.listdir(train_dir))
               if os.path.isdir(os.path.join(train_dir, d))]
    if len(subdirs) < 2:
        raise FileNotFoundError(
            f"Need at least two class subfolders under {train_dir} "
            f"(found {subdirs}). See `python -m scripts.get_data`."
        )


def build_dataloaders(
    train_dir: str = config.TRAIN_DIR,
    batch_size: int = config.BATCH_SIZE,
    val_split: float = config.VAL_SPLIT,
    num_workers: int = config.NUM_WORKERS,
    seed: int = config.SEED,
) -> Tuple[DataLoader, DataLoader, list]:
    """Build train/val dataloaders from a single labelled folder.

    The same images are wrapped by two ImageFolder views — one with the noisy
    phone-capture augmentations (train) and one with clean transforms (val) —
    and split by a seeded permutation so the val set never sees augmentation.

    Returns ``(train_loader, val_loader, class_names)``.
    """
    _check_layout(train_dir)

    train_view = datasets.ImageFolder(train_dir, transform=phone_capture_transforms())
    eval_view = datasets.ImageFolder(train_dir, transform=eval_transforms())

    # torchvision orders classes alphabetically. Reconcile with config:
    #   * exact match            -> nothing to do (the common binary case).
    #   * binary default but the folders are different / more numerous -> adopt
    #     them as a multi-class taxonomy (subtypes that contain a known
    #     malignant name are treated as malignant; otherwise the caller should
    #     set config.MALIGNANT_CLASSES first via config.use_multiclass).
    found = train_view.classes
    if found != config.CLASS_NAMES:
        is_default_binary = config.CLASS_NAMES == ["Benign", "Malignant"]
        if is_default_binary and found != ["Benign", "Malignant"]:
            # Heuristic malignant detection from common naming; the caller can
            # override beforehand for full control.
            mal = [c for c in found
                   if c in config.MALIGNANT_CLASSES
                   or any(k in c.lower() for k in ("malign", "carcinoma", "tumor", "cancer"))]
            config.use_multiclass(found, mal or [found[-1]])
            print(f"[naseej] detected {len(found)} classes -> multi-class mode; "
                  f"malignant = {config.MALIGNANT_CLASSES}")
        elif found != config.CLASS_NAMES:
            raise ValueError(
                f"Class folders {found} don't match the configured taxonomy "
                f"{config.CLASS_NAMES}. Set config.use_multiclass(...) or fix the "
                f"folder names."
            )

    n_total = len(train_view)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_ds = Subset(train_view, train_idx)
    val_ds = Subset(eval_view, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(config.DEVICE == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(config.DEVICE == "cuda"),
        drop_last=False,
    )

    return train_loader, val_loader, train_view.classes
