"""
Dataset layout helper for Naseej.

This does **not** download anything — the anchor dataset (BreaKHis) requires
filling in an access request — but it prints where to get the data, the exact
layout Naseej expects, and verifies/counts what you have so far.

Run::

    python -m scripts.get_data
"""

from __future__ import annotations

import os
from glob import glob

import config

# --------------------------------------------------------------------------- #
# Where to request / download the data
# --------------------------------------------------------------------------- #
DATASETS = {
    "BreaKHis (anchor)": "https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/",
    "PatchCamelyon (PCam)": "https://github.com/basveeling/pcam",
    "LC25000": "https://github.com/tampapath/lung_colon_image_set",
}

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _count_images(folder: str) -> int:
    if not os.path.isdir(folder):
        return 0
    total = 0
    for ext in IMAGE_EXTS:
        total += len(glob(os.path.join(folder, f"*{ext}")))
        total += len(glob(os.path.join(folder, f"*{ext.upper()}")))
    return total


def print_layout() -> None:
    print("Expected layout:")
    print("  data/")
    print("  └── train/")
    for cls in config.CLASS_NAMES:
        print(f"      ├── {cls}/*.png")
    print()


def print_sources() -> None:
    print("Get the data (request access, then download):")
    for name, url in DATASETS.items():
        print(f"  • {name}: {url}")
    print()


def verify() -> bool:
    """Check the expected layout and report per-class image counts."""
    ok = True
    print(f"Checking: {config.TRAIN_DIR}")
    if not os.path.isdir(config.TRAIN_DIR):
        print("  ✗ data/train/ not found.")
        return False

    grand_total = 0
    for cls in config.CLASS_NAMES:
        cls_dir = os.path.join(config.TRAIN_DIR, cls)
        n = _count_images(cls_dir)
        grand_total += n
        if not os.path.isdir(cls_dir):
            print(f"  ✗ missing class folder: {cls}/")
            ok = False
        elif n == 0:
            print(f"  ✗ {cls}/ exists but contains no images.")
            ok = False
        else:
            print(f"  ✓ {cls}/  ->  {n} images")

    if ok:
        print(f"\nLayout looks good — {grand_total} images total. Ready to train:")
        print("  python -m src.train")
    else:
        print("\nLayout incomplete. Arrange images as shown above and re-run.")
    return ok


def main() -> None:
    print("=" * 60)
    print("Naseej · dataset helper")
    print("=" * 60)
    print_sources()
    print_layout()
    verify()


if __name__ == "__main__":
    main()
