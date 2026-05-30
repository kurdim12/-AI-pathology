"""
Automated dataset path for Naseej (no manual forms).

The anchor dataset (BreaKHis) is excellent but gated behind an access request,
which can't be automated. **PatchCamelyon (PCam)** is a fully public binary
histopathology dataset (tumor vs normal) that torchvision can download directly
— perfect for getting a real model training end-to-end today.

This script downloads PCam via ``torchvision.datasets.PCAM`` and materialises a
subset into the ImageFolder layout Naseej expects::

    data/train/Benign/*.png      (PCam label 0 = normal)
    data/train/Malignant/*.png   (PCam label 1 = tumor)

Run::

    python -m scripts.download_pcam                  # default subset
    python -m scripts.download_pcam --per-class 2000 # bigger sample
    python -m scripts.download_pcam --full           # everything (large!)

PCam is large; ``--per-class`` keeps the export quick for a first run.
"""

from __future__ import annotations

import argparse
import os

import config

# PCam: 0 = normal tissue -> Benign, 1 = tumor -> Malignant.
PCAM_TO_NASEEJ = {0: "Benign", 1: "Malignant"}


def export_subset(per_class: int | None, split: str = "train") -> None:
    try:
        from torchvision.datasets import PCAM
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "torchvision with PCAM support is required. Install requirements first."
        ) from exc

    raw_root = os.path.join(config.DATA_DIR, "_pcam_raw")
    os.makedirs(raw_root, exist_ok=True)

    print(f"[naseej] downloading PCam ({split}) to {raw_root} (this can be several GB)...")
    dataset = PCAM(root=raw_root, split=split, download=True)

    out_dirs = {}
    for label, cls in PCAM_TO_NASEEJ.items():
        d = os.path.join(config.TRAIN_DIR, cls)
        os.makedirs(d, exist_ok=True)
        out_dirs[label] = d

    saved = {0: 0, 1: 0}
    target = per_class if per_class is not None else float("inf")
    print(f"[naseej] exporting up to {per_class or 'all'} images per class -> {config.TRAIN_DIR}")

    for idx in range(len(dataset)):
        if saved[0] >= target and saved[1] >= target:
            break
        image, label = dataset[idx]  # PIL image, int label
        label = int(label)
        if saved[label] >= target:
            continue
        out_path = os.path.join(out_dirs[label], f"pcam_{label}_{saved[label]:06d}.png")
        image.save(out_path)
        saved[label] += 1
        if (saved[0] + saved[1]) % 500 == 0:
            print(f"  ...saved Benign={saved[0]} Malignant={saved[1]}")

    print(f"[naseej] done. Benign={saved[0]} Malignant={saved[1]} written to {config.TRAIN_DIR}")
    print("[naseej] verify with `python -m scripts.get_data`, then train with "
          "`python -m src.train`.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PatchCamelyon into Naseej's layout.")
    parser.add_argument("--per-class", type=int, default=1500,
                        help="images to export per class (default 1500)")
    parser.add_argument("--full", action="store_true",
                        help="export the entire split (overrides --per-class; large!)")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    export_subset(per_class=None if args.full else args.per_class, split=args.split)


if __name__ == "__main__":
    main()
