"""
Threshold calibration for Naseej.

A triage tool lives or dies on its operating point. The README is explicit:
thresholds must be re-calibrated on local data before any real use. This module
does that, with the right objective for the problem — **a guaranteed sensitivity
floor**.

Given a trained model and a labelled validation set, we sweep every candidate
cut-off on P(malignant) and choose:

  * ``REVIEW_THRESHOLD`` = the *highest* probability cut-off whose malignant
    sensitivity is still >= ``TARGET_SENSITIVITY`` (default 0.95). Anything at or
    above this gets at least a REVIEW — so we catch the target share of cancers
    while sending as few benign cases to review as possible. Such a cut-off
    always exists, because sensitivity is 1.0 at threshold 0.
  * ``URGENT_THRESHOLD`` = the point that maximises Youden's J
    (sensitivity + specificity − 1), i.e. the most separating operating point,
    clamped to be >= the review cut-off so the bands stay ordered.

The result is written to ``outputs/thresholds.json`` and is picked up
automatically by inference / triage / the demo (see ``inference.load_thresholds``).

Run::

    python -m src.calibrate
    python -m src.calibrate --target-sensitivity 0.98
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

import config
from src.data import build_dataloaders
from src.evaluate import collect_predictions
from src.inference import load_model


def sensitivity_specificity_at(
    labels: np.ndarray, probs: np.ndarray, threshold: float
) -> Tuple[float, float]:
    """Malignant sensitivity and specificity if we call p >= threshold malignant."""
    preds = probs >= threshold
    pos = labels == config.MALIGNANT_INDEX
    neg = labels == config.BENIGN_INDEX

    tp = int(np.sum(preds & pos))
    fn = int(np.sum(~preds & pos))
    tn = int(np.sum(~preds & neg))
    fp = int(np.sum(preds & neg))

    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return sensitivity, specificity


def _candidate_thresholds(probs: np.ndarray) -> List[float]:
    """Distinct cut-offs to evaluate: just below each observed probability.

    Subtracting a tiny epsilon makes ``p >= t`` flip exactly at the observed
    values, so we consider every achievable (sensitivity, specificity) pair.
    """
    uniq = np.unique(probs)
    eps = 1e-6
    cands = [0.0] + [float(p - eps) for p in uniq] + [1.0]
    return sorted(set(c for c in cands if 0.0 <= c <= 1.0))


def calibrate_thresholds(
    labels: np.ndarray,
    probs: np.ndarray,
    target_sensitivity: float = config.TARGET_SENSITIVITY,
) -> Dict[str, float]:
    """Pick (review, urgent) cut-offs from labelled validation predictions.

    Returns a dict with the chosen thresholds and the operating characteristics
    achieved at each, suitable for serialising to JSON.
    """
    if len(set(labels.tolist())) < 2:
        raise ValueError(
            "Calibration needs both Benign and Malignant samples in the "
            "validation split; only one class was present."
        )

    candidates = _candidate_thresholds(probs)

    # REVIEW: highest cut-off that still meets the sensitivity floor.
    review_t = 0.0
    review_stats = (1.0, 0.0)
    for t in candidates:
        sens, spec = sensitivity_specificity_at(labels, probs, t)
        if sens >= target_sensitivity and t >= review_t:
            review_t, review_stats = t, (sens, spec)

    # URGENT: max Youden's J, but never below the review cut-off.
    best_j, urgent_t, urgent_stats = -1.0, review_t, review_stats
    for t in candidates:
        if t < review_t:
            continue
        sens, spec = sensitivity_specificity_at(labels, probs, t)
        if np.isnan(sens) or np.isnan(spec):
            continue
        j = sens + spec - 1.0
        if j > best_j:
            best_j, urgent_t, urgent_stats = j, t, (sens, spec)

    return {
        "review_threshold": round(float(review_t), 6),
        "urgent_threshold": round(float(urgent_t), 6),
        "target_sensitivity": float(target_sensitivity),
        "review_sensitivity": round(float(review_stats[0]), 6),
        "review_specificity": round(float(review_stats[1]), 6),
        "urgent_sensitivity": round(float(urgent_stats[0]), 6),
        "urgent_specificity": round(float(urgent_stats[1]), 6),
        "youden_j": round(float(best_j), 6),
        "n_val": int(len(labels)),
    }


def save_thresholds(thresholds: Dict[str, float], path: str = config.THRESHOLDS_PATH) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(thresholds, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate Naseej triage thresholds for a guaranteed sensitivity floor."
    )
    parser.add_argument("--target-sensitivity", type=float, default=config.TARGET_SENSITIVITY,
                        help="minimum malignant recall the REVIEW cut-off must guarantee")
    parser.add_argument("--checkpoint", default=config.BEST_MODEL_PATH)
    parser.add_argument("--data-dir", default=config.TRAIN_DIR,
                        help="labelled folder; its validation split is used to calibrate")
    parser.add_argument("--out", default=config.THRESHOLDS_PATH)
    args = parser.parse_args()

    model, backbone, trained = load_model(checkpoint_path=args.checkpoint)
    if not trained:
        print("[naseej] warning: no trained checkpoint — calibrating an "
              "un-fine-tuned model is meaningless. Run `python -m src.train` first.")

    _, val_loader, _ = build_dataloaders(train_dir=args.data_dir)
    labels, probs, _ = collect_predictions(model, val_loader, config.DEVICE)

    thresholds = calibrate_thresholds(labels, probs, args.target_sensitivity)
    save_thresholds(thresholds, args.out)

    print("\n================ Naseej · threshold calibration ================")
    print(f"  validation samples        : {thresholds['n_val']}")
    print(f"  target sensitivity floor  : {thresholds['target_sensitivity']:.2f}")
    print("  ----------------------------------------------------------")
    print(f"  REVIEW  threshold = {thresholds['review_threshold']:.4f}   "
          f"(sens {thresholds['review_sensitivity']:.3f}, "
          f"spec {thresholds['review_specificity']:.3f})")
    print(f"  URGENT  threshold = {thresholds['urgent_threshold']:.4f}   "
          f"(sens {thresholds['urgent_sensitivity']:.3f}, "
          f"spec {thresholds['urgent_specificity']:.3f}, "
          f"Youden J {thresholds['youden_j']:.3f})")
    print("  ----------------------------------------------------------")
    print(f"  written -> {args.out}")
    print("  inference, triage and the demo will use these automatically.")
    print("================================================================\n")


if __name__ == "__main__":
    main()
