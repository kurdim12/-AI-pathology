"""
Tests for multi-class subtype grading.

The whole codebase defaults to binary benign/malignant; multi-class mode adds
subtype grading that still rolls up to a benign/malignant triage decision. These
tests verify the taxonomy helpers, the P(malignant) rollup, and that switching
modes is correctly isolated (config is global, so each test restores it).

CPU-only, synthetic tensors, no dataset / no downloads.

Run::

    python -m pytest tests/test_multiclass.py -q
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

import config
from src.inference import TriageResult, analyze, malignant_probability
from src.model import build_model
from src.report import render_bilingual_text, render_html


# A 4-way taxonomy: 2 benign + 2 malignant (alphabetical, matching ImageFolder).
MC_CLASSES = ["adenosis_benign", "ductal_carcinoma", "fibroadenoma_benign", "lobular_carcinoma"]
MC_MALIGNANT = ["ductal_carcinoma", "lobular_carcinoma"]


def _restore_binary():
    config.use_multiclass(["Benign", "Malignant"], ["Malignant"])


# ---- taxonomy helpers ---------------------------------------------------- #

def test_binary_defaults():
    _restore_binary()
    assert not config.is_multiclass()
    assert config.malignant_indices() == [1]
    assert config.loss_class_weights() == list(config.CLASS_WEIGHTS)


def test_use_multiclass_sets_taxonomy():
    try:
        config.use_multiclass(MC_CLASSES, MC_MALIGNANT)
        assert config.is_multiclass()
        # malignant indices are positions 1 and 3.
        assert config.malignant_indices() == [1, 3]
        # benign subtypes weighted 1.0, malignant weighted MALIGNANT_CLASS_WEIGHT.
        w = config.loss_class_weights()
        assert w[0] == 1.0 and w[2] == 1.0
        assert w[1] == config.MALIGNANT_CLASS_WEIGHT and w[3] == config.MALIGNANT_CLASS_WEIGHT
    finally:
        _restore_binary()


# ---- P(malignant) rollup ------------------------------------------------- #

def test_malignant_probability_binary():
    _restore_binary()
    probs = torch.tensor([0.3, 0.7])
    assert abs(float(malignant_probability(probs)) - 0.7) < 1e-6


def test_malignant_probability_sums_malignant_subtypes():
    try:
        config.use_multiclass(MC_CLASSES, MC_MALIGNANT)
        # [benign, malig, benign, malig] -> P(malig) = 0.2 + 0.5 = 0.7
        probs = torch.tensor([0.2, 0.2, 0.1, 0.5])
        assert abs(float(malignant_probability(probs)) - 0.7) < 1e-6
        # batched
        batch = torch.tensor([[0.2, 0.2, 0.1, 0.5], [0.9, 0.05, 0.03, 0.02]])
        out = malignant_probability(batch)
        assert out.shape == (2,)
        assert abs(float(out[0]) - 0.7) < 1e-6 and abs(float(out[1]) - 0.07) < 1e-6
    finally:
        _restore_binary()


# ---- end-to-end analyze in multi-class ----------------------------------- #

def test_analyze_multiclass_predicts_subtype_and_rolls_up():
    try:
        config.use_multiclass(MC_CLASSES, MC_MALIGNANT)
        model = build_model(backbone="resnet18", num_classes=4, pretrained=False).eval()
        img = Image.fromarray(np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8))
        r = analyze(img, model=model, backbone="resnet18", trained=False, with_heatmap=True)
        assert r.label in MC_CLASSES                       # a subtype, not just Benign/Malignant
        assert 0.0 <= r.prob_malignant <= 1.0
        assert r.is_malignant_class == (r.label in MC_MALIGNANT)
        assert r.overlay is not None                       # Grad-CAM still works
    finally:
        _restore_binary()


# ---- bilingual report with subtypes -------------------------------------- #

def test_report_renders_subtype_with_rollup():
    res = TriageResult(label="lobular_carcinoma", prob_malignant=0.93, confidence=0.80,
                       priority="URGENT", trained=True, is_malignant_class=True)
    txt = render_bilingual_text(res)
    assert "lobular carcinoma" in txt          # underscores prettified
    assert "Malignant" in txt and "خبيث" in txt  # rollup shown in both languages
    html = render_html(res, "bilingual")
    assert "lobular carcinoma" in html
