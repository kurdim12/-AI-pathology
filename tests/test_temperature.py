"""
Tests for temperature scaling and its integration with inference.

CPU-only, synthetic tensors / random-init models — no dataset, no downloads.

Run::

    python -m pytest tests/test_temperature.py -q
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

import config
from src.inference import analyze, softmax_with_temperature
from src.model import build_model
from src.temperature import (
    calibration_report,
    fit_temperature,
)


def test_fit_temperature_reduces_overconfidence():
    # Build deliberately OVER-confident logits that are mostly correct: large
    # magnitude => softmax saturates near 1. Temperature should be > 1 to spread
    # them back out.
    torch.manual_seed(0)
    n = 400
    labels = torch.randint(0, 2, (n,))
    logits = torch.zeros(n, 2)
    for i, y in enumerate(labels):
        # Correct 85% of the time, but always with huge magnitude (overconfident).
        winner = int(y) if torch.rand(1).item() < 0.85 else 1 - int(y)
        logits[i, winner] = 8.0
    T = fit_temperature(logits, labels)
    assert T > 1.0, f"expected T>1 for overconfident logits, got {T}"


def test_temperature_preserves_argmax():
    # Scaling by T never changes which class wins -> labels/AUC unaffected.
    torch.manual_seed(1)
    logits = torch.randn(50, 2) * 5
    for T in (0.5, 1.0, 2.0, 5.0):
        base = logits.argmax(dim=1)
        scaled = (logits / T).argmax(dim=1)
        assert torch.equal(base, scaled)


def test_ece_improves_or_holds_after_scaling():
    torch.manual_seed(2)
    n = 300
    labels = torch.randint(0, 2, (n,))
    logits = torch.zeros(n, 2)
    for i, y in enumerate(labels):
        winner = int(y) if torch.rand(1).item() < 0.8 else 1 - int(y)
        logits[i, winner] = 6.0
    T = fit_temperature(logits, labels)
    rep = calibration_report(logits, labels, T)
    # Calibrated ECE should not be worse than uncalibrated.
    assert rep["ece_after"] <= rep["ece_before"] + 1e-6


def test_softmax_with_temperature_uses_attribute():
    model = build_model(backbone="resnet18", pretrained=False).eval()
    x = torch.randn(4, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    logits = model(x)

    model.temperature = 1.0
    p1 = softmax_with_temperature(logits, model)
    model.temperature = 3.0
    p3 = softmax_with_temperature(logits, model)

    # Higher temperature => softer (lower-max) probabilities.
    assert p3.max(dim=1).values.mean() <= p1.max(dim=1).values.mean() + 1e-6
    # Still valid distributions.
    assert torch.allclose(p1.sum(dim=1), torch.ones(4), atol=1e-5)


def test_analyze_default_temperature_is_noop():
    # A model with no fitted temperature behaves as T=1 (no crash, valid output).
    model = build_model(backbone="resnet18", pretrained=False).eval()
    img = Image.fromarray(np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8))
    r = analyze(img, model=model, backbone="resnet18", trained=False, with_heatmap=False)
    assert 0.0 <= r.prob_malignant <= 1.0
