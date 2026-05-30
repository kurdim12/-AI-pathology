"""
Smoke tests for Naseej.

These run on CPU with randomly initialised weights (``pretrained=False`` so no
download is needed) and without any dataset. They verify that the moving parts
fit together: every backbone builds, a forward pass produces 2-class logits,
Grad-CAM yields a normalised heatmap at input resolution, the augmentation
pipelines produce correctly shaped tensors, and the triage thresholds map
probabilities to the right priority.

Run::

    python -m pytest tests/ -q
    # or, without pytest:
    python -m tests.test_smoke
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

import config
from src.data import eval_transforms, phone_capture_transforms
from src.gradcam import GradCAM
from src.inference import (
    PRIORITY_REVIEW,
    PRIORITY_ROUTINE,
    PRIORITY_URGENT,
    analyze,
    triage_priority,
)
from src.model import SUPPORTED_BACKBONES, build_model, get_target_layer


def test_all_backbones_forward():
    x = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    for backbone in SUPPORTED_BACKBONES:
        model = build_model(backbone=backbone, pretrained=False).eval()
        logits = model(x)
        assert logits.shape == (2, len(config.CLASS_NAMES)), backbone


def test_gradcam_shape_and_range():
    model = build_model(backbone="resnet18", pretrained=False).eval()
    cam_engine = GradCAM(model, get_target_layer(model, "resnet18"))
    x = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    cam, logits = cam_engine(x, class_idx=config.MALIGNANT_INDEX)
    cam_engine.remove()

    assert cam.shape == (1, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert logits.shape == (1, len(config.CLASS_NAMES))
    assert float(cam.min()) >= 0.0 - 1e-6
    assert float(cam.max()) <= 1.0 + 1e-6


def test_transforms_output_shape():
    img = Image.fromarray(np.random.randint(0, 255, (300, 250, 3), dtype=np.uint8))
    for tfm in (phone_capture_transforms(), eval_transforms()):
        t = tfm(img)
        assert t.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)


def test_triage_thresholds():
    assert triage_priority(config.URGENT_THRESHOLD) == PRIORITY_URGENT
    assert triage_priority(1.0) == PRIORITY_URGENT
    assert triage_priority(config.REVIEW_THRESHOLD) == PRIORITY_REVIEW
    assert triage_priority((config.URGENT_THRESHOLD + config.REVIEW_THRESHOLD) / 2) == PRIORITY_REVIEW
    assert triage_priority(0.0) == PRIORITY_ROUTINE
    assert triage_priority(config.REVIEW_THRESHOLD - 1e-6) == PRIORITY_ROUTINE


def test_analyze_end_to_end():
    # End-to-end on an in-memory model (no checkpoint, no download).
    model = build_model(backbone="resnet18", pretrained=False).eval()
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    result = analyze(img, model=model, backbone="resnet18", trained=False)

    assert result.label in config.CLASS_NAMES
    assert 0.0 <= result.prob_malignant <= 1.0
    assert result.priority in (PRIORITY_URGENT, PRIORITY_REVIEW, PRIORITY_ROUTINE)
    assert result.overlay is not None and result.heatmap is not None
    assert result.overlay.size == (config.IMAGE_SIZE, config.IMAGE_SIZE)


def _run_all():
    """Allow running without pytest installed."""
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"\nAll {len(fns)} smoke tests passed.")


if __name__ == "__main__":
    _run_all()
