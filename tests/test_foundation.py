"""
Tests for the optional foundation-model loader (src/model.build_foundation_model).

The loader is an opt-in upgrade path that uses ``timm``. These tests verify its
*contract* (argument validation, error messages) without requiring timm, and —
if timm happens to be installed — that it builds a usable 2-class model whose
Grad-CAM target layer can be auto-detected.

Run::

    python -m pytest tests/test_foundation.py -q
"""

from __future__ import annotations

import importlib.util

import pytest

import config
from src.model import build_foundation_model

_HAS_TIMM = importlib.util.find_spec("timm") is not None


# ---- contract / validation (no timm needed) ------------------------------ #

def test_requires_a_model_name():
    saved = config.FOUNDATION_MODEL
    try:
        config.FOUNDATION_MODEL = None
        with pytest.raises(ValueError):
            build_foundation_model(name=None)
    finally:
        config.FOUNDATION_MODEL = saved


def test_rejects_non_timm_prefix():
    with pytest.raises(ValueError):
        build_foundation_model(name="torchvision:resnet50")


@pytest.mark.skipif(_HAS_TIMM, reason="timm installed; ImportError path not exercised")
def test_missing_timm_raises_clear_error():
    with pytest.raises(ImportError):
        build_foundation_model(name="timm:resnet18")


# ---- real build (only when timm is available) ---------------------------- #

@pytest.mark.skipif(not _HAS_TIMM, reason="timm not installed")
def test_builds_two_class_timm_model_and_gradcam_target():
    import torch

    from src.model import autodetect_target_layer

    # A tiny conv backbone keeps the test fast and avoids weight downloads.
    model = build_foundation_model(name="timm:resnet18", num_classes=2).eval()
    out = model(torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
    assert out.shape == (1, 2)
    # Grad-CAM target layer is discoverable on the timm model.
    layer = autodetect_target_layer(model)
    assert layer is not None
