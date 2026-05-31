"""
Tests for phone-capture simulation, the robustness benchmark, and bilingual
reporting. CPU-only, synthetic images, no dataset / no downloads.

Run::

    python -m pytest tests/test_robustness_report.py -q
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.inference import TriageResult
from src.model import build_model
from src.phone_sim import PhoneConditions, apply_conditions, degrade
from src.report import render_bilingual_text, render_html, render_text
from src.robustness import evaluate_at_severity


def _img(seed: int = 0, size: int = 96) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


# ---- phone_sim ----------------------------------------------------------- #

def test_degrade_preserves_mode_and_size():
    img = _img().convert("RGB")
    for sev in (0.0, 0.5, 1.0):
        out = degrade(img, sev)
        assert out.mode == "RGB"
        assert out.size == img.size


def test_severity_clamped_and_monotone_bundle():
    low = PhoneConditions.from_severity(0.2)
    high = PhoneConditions.from_severity(0.9)
    # Worse severity => more blur, more rotation, lower JPEG quality.
    assert high.blur_radius > low.blur_radius
    assert high.rotate_deg > low.rotate_deg
    assert high.jpeg_quality < low.jpeg_quality
    # Out-of-range severity is clamped.
    assert PhoneConditions.from_severity(5.0).blur_radius == PhoneConditions.from_severity(1.0).blur_radius


def test_zero_conditions_is_near_identity():
    img = _img().convert("RGB")
    out = apply_conditions(img, PhoneConditions())
    assert np.array_equal(np.asarray(img), np.asarray(out))


# ---- robustness ---------------------------------------------------------- #

def test_evaluate_at_severity_shape():
    model = build_model(backbone="resnet18", pretrained=False).eval()
    # Build a tiny in-memory val set on disk-free paths by writing temp files.
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        paths, labels = [], []
        for i in range(6):
            p = os.path.join(d, f"img_{i}.png")
            _img(i).save(p)
            paths.append(p)
            labels.append(i % 2)  # alternate benign/malignant
        out = evaluate_at_severity(model, paths, labels, severity=0.5, device="cpu")
    for key in ("severity", "recall_at_review", "auc", "specificity_at_0.5", "n"):
        assert key in out
    assert out["n"] == 6
    assert 0.0 <= out["recall_at_review"] <= 1.0


# ---- report -------------------------------------------------------------- #

def _result(label="Malignant", priority="URGENT", trained=True) -> TriageResult:
    return TriageResult(label=label, prob_malignant=0.91, confidence=0.91,
                        priority=priority, trained=trained)


def test_english_report_contains_fields():
    txt = render_text(_result(), "en")
    assert "Malignant" in txt and "URGENT" in txt
    assert "Decision support" in txt  # disclaimer always present


def test_arabic_report_is_localised():
    txt = render_text(_result(), "ar")
    assert "خبيث" in txt        # Malignant
    assert "عاجل" in txt        # URGENT
    assert "نسيج" in txt        # Naseej


def test_bilingual_and_html():
    bi = render_bilingual_text(_result())
    assert "Malignant" in bi and "خبيث" in bi
    html = render_html(_result(label="Benign", priority="ROUTINE"), "bilingual")
    assert "<section" in html and "حميد" in html and "Benign" in html


def test_untrained_warning_shown():
    txt = render_text(_result(trained=False), "en")
    assert "WARNING" in txt
    ar = render_text(_result(trained=False), "ar")
    assert "تحذير" in ar


def test_invalid_language_raises():
    try:
        render_text(_result(), "fr")
        assert False, "expected ValueError"
    except ValueError:
        pass
