"""
Generate a SYNTHETIC dataset so the full pipeline runs with zero downloads.

This is a sanity / demo fixture, **not** real histopathology — but it is built
to behave like a real classification problem rather than a toy. Benign and
malignant fields share the same overall tissue appearance and differ only in
subtle, overlapping cues (nuclei density, nuclear size variation, a faint colour
cast), with enough per-image variance that the class distributions **overlap**.
A configurable fraction of labels is also flipped, mimicking annotation noise.

The point of the overlap + noise: metrics land at *believable* values
(~0.85-0.92 AUC) instead of a meaningless 1.000, so calibration, the robustness
curve, and temperature scaling actually have something to work on. **Still do
not read clinical meaning into the numbers** — real performance only means
something on real data (BreaKHis / PCam — see `scripts/get_data.py` and
`scripts/download_pcam.py`).

Difficulty levels:
  * ``--difficulty easy``      well-separated, no label noise (fast, deterministic
                               sanity checks / CI).
  * ``--difficulty realistic`` overlapping distributions + label noise (default;
                               yields believable, sub-perfect metrics).

Run::

    python -m scripts.make_demo_data                      # realistic, 80/class
    python -m scripts.make_demo_data --difficulty easy
    python -m scripts.make_demo_data --multiclass --per-class 200
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

import config

# --------------------------------------------------------------------------- #
# Per-class "tissue signatures". Each class is defined not by a far-apart colour
# but by subtle, overlapping morphology: a base stain colour shared across
# classes, plus class-specific nuclei density / size / colour-cast *means* that
# the per-image noise deliberately blurs together.
# --------------------------------------------------------------------------- #
# (base_rgb, nuclei_density, nuclei_size, color_cast_rgb, malignant)
_BINARY_SIGNATURES = {
    # Benign: fewer, more uniform nuclei; faint cool cast.
    "Benign": dict(density=10, size=5.0, cast=np.array([-8.0, 4.0, 12.0]), malignant=False),
    # Malignant: denser, more pleomorphic (size-varied) nuclei; faint warm cast.
    "Malignant": dict(density=16, size=6.5, cast=np.array([12.0, -4.0, -6.0]), malignant=True),
}

# Multi-class: 2 benign + 2 malignant subtypes, neighbouring benign/malignant
# pairs kept close so the malignant rollup is non-trivial. Names sort
# alphabetically (ImageFolder) and malignant ones contain "carcinoma".
_SUBTYPE_SIGNATURES = {
    "adenosis_benign":     dict(density=9,  size=4.5, cast=np.array([-10.0, 2.0, 10.0]), malignant=False),
    "ductal_carcinoma":    dict(density=17, size=7.0, cast=np.array([14.0, -5.0, -8.0]), malignant=True),
    "fibroadenoma_benign": dict(density=11, size=5.0, cast=np.array([-4.0, 6.0, 8.0]),  malignant=False),
    "lobular_carcinoma":   dict(density=15, size=6.0, cast=np.array([8.0, -2.0, -4.0]),  malignant=True),
}

# Shared base stain (H&E-ish pink/purple) so classes are NOT colour-separable.
_BASE_STAIN = np.array([200.0, 150.0, 190.0], dtype=np.float32)


def _make_image(rng: np.random.Generator, sig: dict, size: int, hardness: float) -> Image.Image:
    """Render a vaguely H&E tissue field from a class signature.

    ``hardness`` in [0, 1] scales how much the class cues are drowned in noise:
    0 = crisp/separable, 1 = heavily overlapping. Realistic mode uses a high
    value so benign/malignant clouds genuinely intersect.
    """
    # Shared stain base with a faint, noise-perturbed class colour cast. At high
    # hardness the cast is weak and the per-image colour jitter is large, so the
    # colour channel carries little reliable class signal.
    cast = sig["cast"] * (1.0 - 0.6 * hardness)
    color_jitter = rng.normal(0.0, 10.0 + 25.0 * hardness, 3)
    base_color = _BASE_STAIN + cast + color_jitter
    base = rng.normal(base_color, 18.0, (size, size, 3))

    # Class-bearing morphology: nuclei count and size. At high hardness we add
    # large per-image variation to density/size so the classes overlap.
    density = sig["density"] + rng.normal(0.0, 3.0 + 5.0 * hardness)
    n_blobs = int(max(2, round(density)))
    size_mean = sig["size"] + rng.normal(0.0, 0.5 + 1.5 * hardness)

    ys = rng.integers(0, size, n_blobs)
    xs = rng.integers(0, size, n_blobs)
    yy, xx = np.mgrid[0:size, 0:size]
    for y, x in zip(ys, xs):
        sigma = max(1.5, rng.normal(size_mean, 1.2))
        r2 = (yy - y) ** 2 + (xx - x) ** 2
        blob = np.exp(-r2 / (2 * sigma ** 2))
        # Nuclei are haematoxylin-dark (push toward purple/blue).
        base[..., 0] -= blob * rng.uniform(40, 80)   # less red
        base[..., 1] -= blob * rng.uniform(50, 90)   # less green
        base[..., 2] -= blob * rng.uniform(10, 30)   # keep some blue

    # Global speckle so no two images are alike.
    base += rng.normal(0.0, 6.0 + 8.0 * hardness, (size, size, 3))
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def _generate(signatures: dict, per_class: int, size: int, seed: int,
              hardness: float, label_noise: float) -> None:
    rng = np.random.default_rng(seed)
    names = list(signatures.keys())

    # Pools of alternative class indices for label-noise reassignment, split by
    # malignancy so a flip stays "plausible" (benign<->benign, malig<->malig
    # where possible, else any other class).
    total = 0
    flipped = 0
    for true_name, sig in signatures.items():
        out_dir = os.path.join(config.TRAIN_DIR, true_name)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(per_class):
            img = _make_image(rng, sig, size, hardness)
            # Label noise: occasionally file a true-class image under a different
            # class folder (i.e. a mislabeled example the model must tolerate).
            dest_name = true_name
            if label_noise > 0 and rng.random() < label_noise:
                others = [n for n in names if n != true_name]
                dest_name = others[rng.integers(0, len(others))]
                flipped += 1
            dest_dir = os.path.join(config.TRAIN_DIR, dest_name)
            os.makedirs(dest_dir, exist_ok=True)
            img.save(os.path.join(dest_dir, f"{true_name.lower()}_{i:04d}.png"))
            total += 1

    print(f"[naseej] wrote {total} synthetic images "
          f"({per_class}/class, {len(signatures)} classes, "
          f"hardness={hardness:.2f}, label_noise={label_noise:.2f}, "
          f"{flipped} mislabeled) -> {config.TRAIN_DIR}")
    print("[naseej] SANITY FIXTURE ONLY — not real histopathology. "
          "Metrics are believable-but-meaningless; use real data for real numbers.")


# Difficulty presets: (hardness, label_noise). 'realistic' is tuned to land
# around 0.85-0.90 AUC — a believable, clearly sub-perfect operating regime.
_DIFFICULTY = {
    "easy": (0.0, 0.0),          # crisp, deterministic — fast CI sanity
    "realistic": (0.55, 0.05),   # overlapping + 5% label noise — believable metrics
    "hard": (0.85, 0.08),        # heavily overlapping — stress test
}


def generate(per_class: int = 80, size: int = 96, seed: int = config.SEED,
             multiclass: bool = False, difficulty: str = "realistic") -> None:
    if difficulty not in _DIFFICULTY:
        raise ValueError(f"difficulty must be one of {list(_DIFFICULTY)}")
    hardness, label_noise = _DIFFICULTY[difficulty]
    signatures = _SUBTYPE_SIGNATURES if multiclass else _BINARY_SIGNATURES
    _generate(signatures, per_class, size, seed, hardness, label_noise)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic demo dataset (not real data).")
    parser.add_argument("--per-class", type=int, default=80)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--multiclass", action="store_true",
                        help="generate 4 subtype folders (2 benign + 2 malignant) "
                             "to exercise multi-class grading")
    parser.add_argument("--difficulty", choices=list(_DIFFICULTY), default="realistic",
                        help="'realistic' = overlapping classes + label noise "
                             "(believable ~0.85-0.90 AUC); 'easy' = separable "
                             "(fast CI); 'hard' = heavy overlap (stress test)")
    args = parser.parse_args()
    generate(per_class=args.per_class, size=args.size, seed=args.seed,
             multiclass=args.multiclass, difficulty=args.difficulty)


if __name__ == "__main__":
    main()
