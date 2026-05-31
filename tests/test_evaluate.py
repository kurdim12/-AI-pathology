"""
Unit tests for src/evaluate.py — the metrics that decide whether the tool works.

These exercise compute_metrics / triage_breakdown / collect_predictions directly
on controlled inputs (no training needed), CPU-only, no dataset, no downloads.

Run::

    python -m pytest tests/test_evaluate.py -q
"""

from __future__ import annotations

import numpy as np
import torch

import config
from src.evaluate import (
    collect_predictions,
    compute_metrics,
    triage_breakdown,
)

# ---- compute_metrics ----------------------------------------------------- #

def test_compute_metrics_perfect_separation():
    # Labels and preds agree exactly; probs rank perfectly.
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(labels, probs, preds)
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["auc"] == 1.0
    assert m["support"] == 4


def test_compute_metrics_counts_false_negatives():
    # Two malignant, one missed (pred 0) -> sensitivity 0.5.
    labels = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 0, 0])
    probs = np.array([0.9, 0.4, 0.1, 0.2])
    m = compute_metrics(labels, probs, preds)
    assert m["sensitivity"] == 0.5      # 1 of 2 caught
    assert m["specificity"] == 1.0      # both benign correct
    cm = m["confusion_matrix"]
    # rows = true [benign, malignant], cols = pred [benign, malignant]
    assert cm[1, 0] == 1                # one malignant predicted benign (a miss)


def test_compute_metrics_single_class_auc_nan():
    labels = np.array([1, 1, 1])
    preds = np.array([1, 1, 0])
    probs = np.array([0.9, 0.8, 0.4])
    m = compute_metrics(labels, probs, preds)
    assert m["auc"] != m["auc"]         # NaN when only one class present


def test_compute_metrics_empty_raises():
    try:
        compute_metrics(np.array([]), np.array([]), np.array([]))
        assert False, "expected ValueError on empty input"
    except ValueError:
        pass


# ---- triage_breakdown ---------------------------------------------------- #

def test_triage_breakdown_recall_at_review(tmp_path, monkeypatch):
    # Point thresholds at a known operating point: urgent>=0.7, review>=0.3.
    import json

    thr = tmp_path / "thresholds.json"
    thr.write_text(json.dumps({"urgent_threshold": 0.7, "review_threshold": 0.3}))
    monkeypatch.setattr(config, "THRESHOLDS_PATH", str(thr))

    # 4 malignant: probs 0.9,0.5,0.4,0.1 -> 3 are >=review(0.3), 1 missed.
    # 2 benign: 0.2,0.05 -> both routine.
    labels = np.array([1, 1, 1, 1, 0, 0])
    probs = np.array([0.9, 0.5, 0.4, 0.1, 0.2, 0.05])
    b = triage_breakdown(labels, probs)
    assert b["malignant_recall_at_review"] == 0.75       # 3 of 4 flagged
    assert b["malignant_missed_as_routine"] == 1
    assert b["bands"]["URGENT"]["malignant"] == 1        # only 0.9 >= 0.7


# ---- collect_predictions (binary + multiclass collapse) ------------------ #

class _ConstModel(torch.nn.Module):
    """Returns fixed logits regardless of input — lets us assert exact probs."""

    def __init__(self, logits_row):
        super().__init__()
        self.row = torch.tensor(logits_row, dtype=torch.float)
        self.temperature = 1.0

    def forward(self, x):
        return self.row.unsqueeze(0).repeat(x.size(0), 1)


def _loader(n, n_classes):
    x = torch.zeros(n, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    y = torch.arange(n) % n_classes
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=4)


def test_collect_predictions_binary_shapes():
    model = _ConstModel([0.0, 5.0]).eval()   # always predicts malignant
    labels, probs, preds = collect_predictions(model, _loader(8, 2), "cpu")
    assert len(labels) == len(probs) == len(preds) == 8
    assert set(labels.tolist()) <= {0, 1}
    assert (preds == 1).all()                # malignant logit dominates
    assert (probs > 0.9).all()


def test_collect_predictions_multiclass_collapses_to_binary():
    try:
        # 4 classes; malignant = indices 1 and 3.
        config.use_multiclass(
            ["a_benign", "b_carcinoma", "c_benign", "d_carcinoma"],
            ["b_carcinoma", "d_carcinoma"],
        )
        # Logits favour class 3 (malignant) -> P(malignant) high, label binary 1.
        model = _ConstModel([0.0, 0.0, 0.0, 6.0]).eval()
        labels, probs, preds = collect_predictions(model, _loader(8, 4), "cpu")
        assert set(labels.tolist()) <= {0, 1}        # collapsed to binary
        assert (preds == 1).all()
        assert (probs > 0.9).all()                   # summed malignant prob
    finally:
        config.use_multiclass(["Benign", "Malignant"], ["Malignant"])
