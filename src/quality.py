"""
Slide-image quality gating.

A phone pointed down a microscope captures plenty of frames that aren't usable
tissue: blank/background fields (mostly white slide), out-of-focus blur, or
near-empty edges. Feeding those to the classifier produces confident-looking
nonsense. This module is a cheap, dependency-light pre-check that flags an image
as low-quality *before* it reaches the model, so the UI/queue can ask for a
re-capture instead of emitting a bogus triage call.

Heuristics (no learning, fast, explainable):
  * **tissue fraction** — share of pixels that are stained tissue rather than
    bright background. H&E tissue is darker and more saturated than the white
    slide, so we threshold on brightness + saturation.
  * **focus / sharpness** — variance of a Laplacian-like edge response; blurry
    fields have low edge energy.

These are intentionally simple and conservative: the goal is to catch obviously
unusable captures, not to be a learned quality model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

import config


@dataclass
class QualityResult:
    """Outcome of the pre-classification quality check."""

    usable: bool
    tissue_fraction: float    # 0..1 share of pixels that look like tissue
    focus_score: float        # edge-energy proxy; higher = sharper
    reason: str               # human-readable explanation when not usable


def _tissue_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels that look like stained tissue (not bright slide).

    Works in a light HSV-ish space: tissue is either reasonably saturated or
    not too bright. Bright, low-saturation pixels are background glass.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    brightness = maxc / 255.0
    saturation = np.where(maxc > 0, (maxc - minc) / np.maximum(maxc, 1e-6), 0.0)
    # Tissue = saturated enough OR dark enough to not be background glass.
    return (saturation > config.TISSUE_SAT_THRESHOLD) | (brightness < config.TISSUE_BRIGHTNESS_THRESHOLD)


def _focus_score(gray: np.ndarray) -> float:
    """Variance of a 4-neighbour Laplacian — a standard sharpness proxy."""
    g = gray.astype(np.float32)
    lap = (
        -4.0 * g
        + np.roll(g, 1, 0) + np.roll(g, -1, 0)
        + np.roll(g, 1, 1) + np.roll(g, -1, 1)
    )
    # Ignore the wrapped borders.
    return float(lap[1:-1, 1:-1].var())


def assess_quality(image: Image.Image) -> QualityResult:
    """Assess whether an image is usable tissue before classification."""
    rgb = np.asarray(image.convert("RGB"))
    tissue_fraction = float(_tissue_mask(rgb).mean())
    gray = rgb.mean(axis=2)
    focus = _focus_score(gray)

    usable, reason = True, "ok"
    if tissue_fraction < config.MIN_TISSUE_FRACTION:
        usable = False
        reason = (f"low tissue content ({tissue_fraction:.0%} < "
                  f"{config.MIN_TISSUE_FRACTION:.0%}) — mostly background; re-capture")
    elif focus < config.MIN_FOCUS_SCORE:
        usable = False
        reason = (f"image looks out of focus (sharpness {focus:.0f} < "
                  f"{config.MIN_FOCUS_SCORE:.0f}) — refocus and re-capture")

    return QualityResult(usable=usable, tissue_fraction=tissue_fraction,
                         focus_score=focus, reason=reason)
