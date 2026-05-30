"""
Tests for threshold calibration, TTA, and threshold plumbing.

All run on CPU with synthetic arrays / random-init models — no dataset, no
downloads — and assert the properties that actually matter for a recall-first
triage tool: the sensitivity floor is honoured, the bands stay ordered, TTA
yields valid probabilities, and calibrated thresholds round-trip through disk.

Run::

    python -m pytest tests/test_calibrate_tta.py -q
    # or
    python -m tests.test_calibrate_tta
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import torch
from PIL import Image

import config
from src.calibrate import (
    calibrate_thresholds,
    save_thresholds,
    sensitivity_specificity_at,
)
from src.inference import (
    PRIORITY_REVIEW,
    PRIORITY_ROUTINE,
    PRIORITY_URGENT,
    analyze,
    load_thresholds,
    triage_priority,
)
from src.model import build_model


def _toy_problem():
    # 10 benign (lowish probs), 10 malignant (higher, but 2 hard low ones).
    labels = np.array([0] * 10 + [1] * 10)
    probs = np.array(
        [0.05, 0.10, 0.12, 0.20, 0.22, 0.30, 0.35, 0.40, 0.45, 0.50,
         0.20, 0.25, 0.55, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    )
    return labels, probs


def test_calibration_honours_sensitivity_floor():
    labels, probs = _toy_problem()
    for target in (0.80, 0.90, 1.00):
        out = calibrate_thresholds(labels, probs, target_sensitivity=target)
        sens, _ = sensitivity_specificity_at(labels, probs, out["review_threshold"])
        assert sens >= target - 1e-9, (target, sens)
        # Bands ordered: urgent cut-off is never below review cut-off.
        assert out["urgent_threshold"] >= out["review_threshold"]


def test_higher_target_lowers_review_threshold():
    # Demanding more recall can only push the REVIEW cut-off down (or equal).
    labels, probs = _toy_problem()
    low = calibrate_thresholds(labels, probs, 0.80)["review_threshold"]
    high = calibrate_thresholds(labels, probs, 1.00)["review_threshold"]
    assert high <= low


def test_calibration_needs_both_classes():
    labels = np.zeros(8, dtype=int)
    probs = np.linspace(0, 1, 8)
    try:
        calibrate_thresholds(labels, probs, 0.95)
        assert False, "expected ValueError for single-class validation set"
    except ValueError:
        pass


def test_thresholds_save_and_load_roundtrip():
    labels, probs = _toy_problem()
    out = calibrate_thresholds(labels, probs, 0.90)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "thresholds.json")
        save_thresholds(out, path)
        urgent, review = load_thresholds(path)
        assert urgent == out["urgent_threshold"]
        assert review == out["review_threshold"]
        # File is valid JSON with the expected keys.
        with open(path) as fh:
            data = json.load(fh)
        assert "review_sensitivity" in data and "youden_j" in data


def test_load_thresholds_falls_back_to_config():
    urgent, review = load_thresholds("/no/such/file.json")
    assert urgent == config.URGENT_THRESHOLD
    assert review == config.REVIEW_THRESHOLD


def test_triage_priority_explicit_thresholds():
    assert triage_priority(0.9, 0.7, 0.3) == PRIORITY_URGENT
    assert triage_priority(0.5, 0.7, 0.3) == PRIORITY_REVIEW
    assert triage_priority(0.1, 0.7, 0.3) == PRIORITY_ROUTINE


def test_tta_produces_valid_probability():
    model = build_model(backbone="resnet18", pretrained=False).eval()
    img = Image.fromarray(np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8))
    r_plain = analyze(img, model=model, backbone="resnet18", trained=False,
                      tta=False, with_heatmap=False)
    r_tta = analyze(img, model=model, backbone="resnet18", trained=False,
                    tta=True, with_heatmap=False)
    for r in (r_plain, r_tta):
        assert 0.0 <= r.prob_malignant <= 1.0
        assert r.priority in (PRIORITY_URGENT, PRIORITY_REVIEW, PRIORITY_ROUTINE)


def test_analyze_respects_explicit_thresholds():
    model = build_model(backbone="resnet18", pretrained=False).eval()
    img = Image.fromarray(np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8))
    # Force everything URGENT (thresholds at 0) then ROUTINE (thresholds at >1).
    r_urgent = analyze(img, model=model, backbone="resnet18", trained=False,
                       with_heatmap=False, thresholds=(0.0, 0.0))
    r_routine = analyze(img, model=model, backbone="resnet18", trained=False,
                        with_heatmap=False, thresholds=(1.01, 1.01))
    assert r_urgent.priority == PRIORITY_URGENT
    assert r_routine.priority == PRIORITY_ROUTINE


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ✓ {fn.__name__}")
    print(f"\nAll {len(fns)} calibration/TTA tests passed.")


if __name__ == "__main__":
    _run_all()
