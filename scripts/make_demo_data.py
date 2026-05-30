"""
Generate a tiny SYNTHETIC dataset so the full pipeline runs with zero downloads.

This is a sanity / demo fixture, **not** real histopathology. It paints a
learnable but trivial signal — benign fields skew blue-green, malignant fields
skew red-purple, both with cellular-looking texture noise and blobs — into the
ImageFolder layout Naseej expects::

    data/train/Benign/*.png
    data/train/Malignant/*.png

It exists so you (or CI) can exercise train -> evaluate -> calibrate -> triage
end-to-end on any machine in seconds. **Do not read anything into the metrics**:
real performance only means something on real data (BreaKHis / PCam — see
`scripts/get_data.py` and `scripts/download_pcam.py`).

Run::

    python -m scripts.make_demo_data                 # 80 per class
    python -m scripts.make_demo_data --per-class 200
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

import config

# Class colour centres (RGB). Deliberately separable so a model can learn fast.
_CENTRES = {
    "Benign": np.array([70, 120, 175], dtype=np.float32),     # cool blue-green
    "Malignant": np.array([175, 70, 110], dtype=np.float32),  # red-purple
}


def _make_image(rng: np.random.Generator, centre: np.ndarray, size: int) -> Image.Image:
    """A noisy, blobby field of the given colour — vaguely tissue-like."""
    base = rng.normal(centre, 30.0, (size, size, 3))

    # Scatter a few darker "nuclei" blobs for texture.
    n_blobs = rng.integers(8, 18)
    ys = rng.integers(0, size, n_blobs)
    xs = rng.integers(0, size, n_blobs)
    yy, xx = np.mgrid[0:size, 0:size]
    for y, x in zip(ys, xs):
        r2 = (yy - y) ** 2 + (xx - x) ** 2
        blob = np.exp(-r2 / (2 * (rng.uniform(3, 7) ** 2)))
        base -= blob[..., None] * rng.uniform(40, 90)

    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


# Multi-class preset: 4 synthetic "subtypes", 2 benign + 2 malignant, each with
# its own colour centre so subtype grading is learnable. Names sort
# alphabetically (matching ImageFolder) and the malignant ones contain
# "carcinoma" so the taxonomy auto-detects them.
_SUBTYPE_CENTRES = {
    "adenosis_benign": np.array([70, 150, 120], dtype=np.float32),      # teal
    "ductal_carcinoma": np.array([185, 60, 90], dtype=np.float32),      # red
    "fibroadenoma_benign": np.array([70, 110, 190], dtype=np.float32),  # blue
    "lobular_carcinoma": np.array([180, 90, 170], dtype=np.float32),    # magenta
}


def _generate(centres: dict, per_class: int, size: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    total = 0
    for cls, centre in centres.items():
        out_dir = os.path.join(config.TRAIN_DIR, cls)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(per_class):
            _make_image(rng, centre, size).save(os.path.join(out_dir, f"{cls.lower()}_{i:04d}.png"))
            total += 1
    print(f"[naseej] wrote {total} synthetic images "
          f"({per_class}/class, {len(centres)} classes) -> {config.TRAIN_DIR}")
    print("[naseej] SANITY FIXTURE ONLY — not real histopathology. "
          "Verify with `python -m scripts.get_data`.")


def generate(per_class: int = 80, size: int = 96, seed: int = config.SEED,
             multiclass: bool = False) -> None:
    _generate(_SUBTYPE_CENTRES if multiclass else _CENTRES, per_class, size, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic demo dataset (not real data).")
    parser.add_argument("--per-class", type=int, default=80)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--multiclass", action="store_true",
                        help="generate 4 subtype folders (2 benign + 2 malignant) "
                             "to exercise multi-class grading")
    args = parser.parse_args()
    generate(per_class=args.per_class, size=args.size, seed=args.seed,
             multiclass=args.multiclass)


if __name__ == "__main__":
    main()
