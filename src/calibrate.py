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

import numpy as np

import config
from src.data import build_dataloaders
from src.evaluate import collect_predictions
from src.inference import load_model


def sensitivity_specificity_at(
    labels: np.ndarray, probs: np.ndarray, threshold: float
) -> tuple[float, float]:
    """Malignant sensitivity and specificity if we call p >= threshold malignant.

    ``labels`` are binary (1 = malignant, 0 = benign), as produced by
    ``collect_predictions`` / ``_degraded_predictions``.
    """
    preds = probs >= threshold
    pos = labels == 1
    neg = labels == 0

    tp = int(np.sum(preds & pos))
    fn = int(np.sum(~preds & pos))
    tn = int(np.sum(~preds & neg))
    fp = int(np.sum(preds & neg))

    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return sensitivity, specificity


def _candidate_thresholds(probs: np.ndarray) -> list[float]:
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
) -> dict[str, float]:
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


def save_thresholds(thresholds: dict[str, float], path: str = config.THRESHOLDS_PATH) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(thresholds, fh, indent=2)


def _degraded_predictions(model, data_dir: str, severity: float, device: str):
    """Score the val split under simulated phone degradation.

    Calibrating against these probabilities makes the chosen thresholds hold up
    in field conditions, not just on pristine images — a direct answer to the
    brittleness the robustness benchmark exposed.
    """
    import numpy as _np
    from PIL import Image

    from src.phone_sim import degrade
    from src.robustness import _score, _val_paths_labels

    # Pass the dataset dir explicitly: build_dataloaders' default is bound at
    # import, so mutating config.TRAIN_DIR would not redirect the split.
    paths, labels = _val_paths_labels(train_dir=data_dir)

    mal_idx = set(config.malignant_indices())
    bin_labels = [1 if int(y) in mal_idx else 0 for y in labels]

    probs = []
    for path in paths:
        img = Image.open(path)
        if severity > 0:
            img = degrade(img, severity)
        probs.append(_score(model, img, device))
    return _np.array(bin_labels), _np.array(probs)


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
    parser.add_argument("--robust", type=float, default=None, metavar="SEVERITY",
                        help="calibrate under simulated phone degradation at this "
                             "0..1 severity, so the sensitivity floor holds in the "
                             "field (not just on clean images)")
    args = parser.parse_args()

    model, backbone, trained = load_model(checkpoint_path=args.checkpoint)
    if not trained:
        print("[naseej] warning: no trained checkpoint — calibrating an "
              "un-fine-tuned model is meaningless. Run `python -m src.train` first.")

    if args.robust is not None:
        labels, probs = _degraded_predictions(model, args.data_dir, args.robust, config.DEVICE)
        print(f"[naseej] calibrating under phone degradation severity={args.robust:.2f}")
    else:
        _, val_loader, _ = build_dataloaders(train_dir=args.data_dir)
        labels, probs, _ = collect_predictions(model, val_loader, config.DEVICE)

    thresholds = calibrate_thresholds(labels, probs, args.target_sensitivity)
    if args.robust is not None:
        thresholds["calibrated_under_severity"] = float(args.robust)
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
