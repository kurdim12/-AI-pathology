"""
Unit tests for src/train.py building blocks.

Covers the pieces that decide model quality and selection: validate() metrics
(binary + multiclass), seed determinism, the AMP scaler factory, two-phase
freeze/unfreeze, and temperature fitting + checkpoint round-trip — all on tiny
in-memory tensors, CPU-only, no dataset, no downloads.

Run::

    python -m pytest tests/test_train.py -q
"""

from __future__ import annotations

import os
import tempfile

import torch
import torch.nn as nn

import config
from src.model import build_model, get_head
from src.train import (
    _make_scaler,
    _set_backbone_trainable,
    set_seed,
    validate,
)


def _loader(n=8, n_classes=2):
    x = torch.randn(n, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    y = torch.arange(n) % n_classes
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=4)


def test_set_seed_is_deterministic():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_make_scaler_disabled_on_cpu():
    scaler = _make_scaler(enabled=False)
    assert scaler.is_enabled() is False


def test_validate_returns_sensible_metrics():
    model = build_model(backbone="resnet18", num_classes=2, pretrained=False).eval()
    model.temperature = 1.0
    crit = nn.CrossEntropyLoss()
    loss, auc, acc = validate(model, _loader(8, 2), crit, "cpu")
    assert loss >= 0.0
    assert (0.0 <= acc <= 1.0)
    # AUC is NaN only if a single class is present; loader has both.
    assert auc != auc or (0.0 <= auc <= 1.0)


def test_validate_binary_auc_uses_malignant_probability_multiclass():
    # In multiclass, validate must compute a *binary* AUC from summed malignant
    # probability — not crash on a 4-way head.
    try:
        config.use_multiclass(
            ["a_benign", "b_carcinoma", "c_benign", "d_carcinoma"],
            ["b_carcinoma", "d_carcinoma"],
        )
        model = build_model(backbone="resnet18", num_classes=4, pretrained=False).eval()
        model.temperature = 1.0
        crit = nn.CrossEntropyLoss()
        loss, auc, acc = validate(model, _loader(12, 4), crit, "cpu")
        assert loss >= 0.0 and 0.0 <= acc <= 1.0
        assert auc != auc or (0.0 <= auc <= 1.0)
    finally:
        config.use_multiclass(["Benign", "Malignant"], ["Malignant"])


def test_two_phase_freeze_then_unfreeze():
    model = build_model(backbone="resnet18", num_classes=2, pretrained=False)
    head_params = {id(p) for p in get_head(model, "resnet18").parameters()}

    # Freeze: only the head trains.
    _set_backbone_trainable(model, False)
    for p in model.parameters():
        if id(p) in head_params:
            assert p.requires_grad
        else:
            assert not p.requires_grad

    # Unfreeze: everything trains.
    _set_backbone_trainable(model, True)
    assert all(p.requires_grad for p in model.parameters())


def test_temperature_fit_and_checkpoint_roundtrip():
    # Fit T on overconfident logits, store in a checkpoint, reload via inference.
    from src.inference import load_model, softmax_with_temperature
    from src.temperature import fit_temperature

    model = build_model(backbone="resnet18", num_classes=2, pretrained=False).eval()
    torch.manual_seed(0)
    logits = torch.zeros(200, 2)
    labels = torch.randint(0, 2, (200,))
    for i, y in enumerate(labels):
        winner = int(y) if torch.rand(1).item() < 0.85 else 1 - int(y)
        logits[i, winner] = 8.0
    T = fit_temperature(logits, labels)
    assert T > 1.0

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        torch.save({
            "model_state": model.state_dict(),
            "backbone": "resnet18",
            "class_names": ["Benign", "Malignant"],
            "malignant_classes": ["Malignant"],
            "temperature": T,
        }, path)
        loaded, backbone, trained = load_model(checkpoint_path=path, device="cpu")
        assert trained is True
        assert abs(float(loaded.temperature) - T) < 1e-6
        # Temperature actually softens probabilities.
        x = torch.randn(4, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        raw = torch.softmax(loaded(x), dim=1).max(dim=1).values.mean()
        tempered = softmax_with_temperature(loaded(x), loaded).max(dim=1).values.mean()
        assert tempered <= raw + 1e-6
