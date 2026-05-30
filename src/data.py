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
            "data/train/Malignant/*.png, then run `python -m scripts.get_data` "
            "to verify."
        )
    for cls in config.CLASS_NAMES:
        cls_dir = os.path.join(train_dir, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(
                f"Expected class folder missing: {cls_dir}\n"
                "Run `python -m scripts.get_data` for the expected layout."
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

    # Sanity check: torchvision orders classes alphabetically. Make sure that
    # matches the index convention the rest of the codebase relies on.
    if train_view.classes != config.CLASS_NAMES:
        raise ValueError(
            f"Class order mismatch. Found {train_view.classes}, "
            f"expected {config.CLASS_NAMES}. Rename the folders to match."
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
