"""
Bundle a handful of sample slide images into assets/samples/ for the booth demo.

So a judge can click "Sample slides" in the Gradio app without having any
dataset on disk. These are the same SYNTHETIC fixtures the rest of the demo
uses (clearly not real histopathology) plus one deliberately-blank frame to show
the quality gate rejecting an unusable capture.

Run::

    python -m scripts.make_samples
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

import config
import scripts.make_demo_data as mdd

SAMPLES_DIR = os.path.join(config.ROOT, "assets", "samples")


def main() -> None:
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    rng = np.random.default_rng(7)

    # A couple of "benign" and "malignant" realistic-difficulty fields each.
    hardness, _ = mdd._DIFFICULTY["realistic"]
    plan = [
        ("benign_sample_1", mdd._BINARY_SIGNATURES["Benign"]),
        ("benign_sample_2", mdd._BINARY_SIGNATURES["Benign"]),
        ("malignant_sample_1", mdd._BINARY_SIGNATURES["Malignant"]),
        ("malignant_sample_2", mdd._BINARY_SIGNATURES["Malignant"]),
    ]
    for name, sig in plan:
        img = mdd._make_image(rng, sig, size=224, hardness=hardness)
        img.save(os.path.join(SAMPLES_DIR, f"{name}.png"))

    # A near-blank frame to demonstrate the quality gate (mostly bright slide).
    blank = np.full((224, 224, 3), 244, dtype=np.uint8)
    blank[100:120, 100:120] = 180  # a tiny smudge
    Image.fromarray(blank).save(os.path.join(SAMPLES_DIR, "blank_unusable.png"))

    print(f"[naseej] wrote 5 sample slides -> {SAMPLES_DIR}")
    print("[naseej] SANITY FIXTURES ONLY — not real histopathology.")


if __name__ == "__main__":
    main()
