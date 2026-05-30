"""
Tests for real-world capabilities: quality gating, uncertainty/abstention,
and batched inference.

CPU-only, synthetic images, no dataset / no downloads.

Run::

    python -m pytest tests/test_capabilities.py -q
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

import config
from src.inference import (
    PRIORITY_REVIEW,
    PRIORITY_ROUTINE,
    PRIORITY_URGENT,
    _result_from_probs,
    analyze,
    predict_batch,
)
from src.model import build_model
from src.quality import assess_quality


# ---- quality gating ------------------------------------------------------ #

def test_blank_frame_rejected():
    blank = Image.fromarray(np.full((96, 96, 3), 245, dtype=np.uint8))  # bright/empty
    q = assess_quality(blank)
    assert not q.usable
    assert q.tissue_fraction < config.MIN_TISSUE_FRACTION


def test_tissue_frame_accepted():
    rng = np.random.default_rng(0)
    tissue = Image.fromarray(rng.integers(60, 180, (96, 96, 3), dtype=np.uint8))
    q = assess_quality(tissue)
    assert q.usable
    assert q.tissue_fraction >= config.MIN_TISSUE_FRACTION


def test_analyze_check_quality_short_circuits():
    model = build_model(backbone="resnet18", pretrained=False).eval()
    blank = Image.fromarray(np.full((96, 96, 3), 250, dtype=np.uint8))
    r = analyze(blank, model=model, backbone="resnet18", trained=False,
                with_heatmap=False, check_quality=True)
    assert r.quality_ok is False
    assert r.label == "Indeterminate"
    assert r.priority == PRIORITY_REVIEW         # never silently ROUTINE
    assert r.prob_malignant != r.prob_malignant  # NaN (no model call trusted)


# ---- uncertainty / abstention -------------------------------------------- #

def test_uncertain_band_bumps_routine_to_review():
    probs = torch.tensor([0.48, 0.52])  # P(malignant)=0.52, near 0.5
    # Thresholds that would otherwise make 0.52 ROUTINE.
    r = _result_from_probs(probs, 0.52, 1, True, (0.9, 0.8))
    assert r.uncertain is True
    assert r.priority == PRIORITY_REVIEW


def test_confident_call_not_uncertain():
    r = _result_from_probs(torch.tensor([0.03, 0.97]), 0.97, 1, True, (0.7, 0.3))
    assert r.uncertain is False
    assert r.priority == PRIORITY_URGENT


def test_uncertain_does_not_downgrade_urgent():
    # An uncertain case that's already URGENT stays URGENT (only ROUTINE is bumped).
    r = _result_from_probs(torch.tensor([0.49, 0.51]), 0.51, 1, True, (0.5, 0.3))
    assert r.uncertain is True
    assert r.priority == PRIORITY_URGENT


# ---- batched inference --------------------------------------------------- #

def test_predict_batch_matches_single_and_aligned():
    torch.manual_seed(0)
    model = build_model(backbone="resnet18", pretrained=False).eval()
    imgs = [Image.fromarray(np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8))
            for _ in range(10)]
    single = [analyze(im, model=model, backbone="resnet18", trained=False,
                      with_heatmap=False) for im in imgs]
    batched = predict_batch(imgs, model=model, backbone="resnet18", trained=False,
                            batch_size=4)
    assert len(batched) == len(imgs)
    for s, b in zip(single, batched):
        assert abs(s.prob_malignant - b.prob_malignant) < 1e-5
        assert s.label == b.label
        assert s.priority == b.priority


def test_predict_batch_empty():
    model = build_model(backbone="resnet18", pretrained=False).eval()
    assert predict_batch([], model=model, backbone="resnet18", trained=False) == []
