"""
Evaluation for Naseej.

For a triage tool the headline number is **sensitivity (recall)** — the share of
true malignancies the model flags. A missed cancer is the failure mode that
matters, so we lead with it, then report specificity, AUC, accuracy and the full
confusion matrix.

Run::

    python -m src.evaluate
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

import config
from src.data import build_dataloaders
from src.inference import (
    PRIORITY_REVIEW,
    PRIORITY_ROUTINE,
    PRIORITY_URGENT,
    load_model,
    load_thresholds,
    triage_priority,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Return ``(labels, probs_malignant, preds)`` as numpy arrays."""
    model.eval()
    all_labels, all_probs, all_preds = [], [], []
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, config.MALIGNANT_INDEX]
        preds = (probs >= 0.5).long()

        all_labels.extend(labels.tolist())
        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return np.array(all_labels), np.array(all_probs), np.array(all_preds)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> dict:
    """Compute triage metrics from labels/probabilities/predictions."""
    # Confusion matrix with fixed label order [Benign(0), Malignant(1)].
    cm = confusion_matrix(labels, preds, labels=[config.BENIGN_INDEX, config.MALIGNANT_INDEX])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")  # recall, malignant
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    accuracy = (tp + tn) / cm.sum() if cm.sum() else float("nan")
    auc = roc_auc_score(labels, probs) if len(set(labels.tolist())) > 1 else float("nan")

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc": auc,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "support": int(cm.sum()),
    }


def triage_breakdown(labels: np.ndarray, probs: np.ndarray) -> dict:
    """How the current triage thresholds bucket the validation set.

    Reports the per-band counts and, crucially, the malignant recall captured
    at the REVIEW cut-off — the share of cancers that get at least flagged.
    """
    urgent_t, review_t = load_thresholds()
    bands = {PRIORITY_URGENT: {"benign": 0, "malignant": 0},
             PRIORITY_REVIEW: {"benign": 0, "malignant": 0},
             PRIORITY_ROUTINE: {"benign": 0, "malignant": 0}}

    for label, p in zip(labels, probs):
        band = triage_priority(float(p), urgent_t, review_t)
        key = "malignant" if label == config.MALIGNANT_INDEX else "benign"
        bands[band][key] += 1

    n_malignant = int(np.sum(labels == config.MALIGNANT_INDEX))
    flagged = bands[PRIORITY_URGENT]["malignant"] + bands[PRIORITY_REVIEW]["malignant"]
    recall_at_review = flagged / n_malignant if n_malignant else float("nan")

    return {
        "urgent_threshold": urgent_t,
        "review_threshold": review_t,
        "bands": bands,
        "malignant_recall_at_review": recall_at_review,
        "malignant_missed_as_routine": bands[PRIORITY_ROUTINE]["malignant"],
    }


def save_confusion_matrix(cm: np.ndarray, out_path: str) -> None:
    """Save a labelled confusion-matrix figure (best-effort)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=config.CLASS_NAMES)
    ax.set_yticks([0, 1], labels=config.CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def print_report(metrics: dict, breakdown: dict | None = None) -> None:
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()
    print("\n================ Naseej · evaluation ================")
    print(f"  Samples evaluated : {metrics['support']}")
    print("  ---- triage metrics (lead with sensitivity) ----")
    print(f"  Sensitivity (recall) : {metrics['sensitivity']:.4f}   <-- missed cancers matter most")
    print(f"  Specificity          : {metrics['specificity']:.4f}")
    print(f"  AUC                  : {metrics['auc']:.4f}")
    print(f"  Accuracy             : {metrics['accuracy']:.4f}")
    print("  ---- confusion matrix (threshold 0.5) ----")
    print(f"                 pred Benign   pred Malignant")
    print(f"  true Benign        {tn:6d}          {fp:6d}")
    print(f"  true Malignant     {fn:6d}          {tp:6d}")
    if breakdown is not None:
        b = breakdown["bands"]
        print("  ---- triage bands (urgent ≥ {:.3f}, review ≥ {:.3f}) ----".format(
            breakdown["urgent_threshold"], breakdown["review_threshold"]))
        for band in (PRIORITY_URGENT, PRIORITY_REVIEW, PRIORITY_ROUTINE):
            print(f"  {band:<8} : benign {b[band]['benign']:4d}   malignant {b[band]['malignant']:4d}")
        print(f"  malignant recall captured at REVIEW : "
              f"{breakdown['malignant_recall_at_review']:.4f}")
        if breakdown["malignant_missed_as_routine"]:
            print(f"  ⚠ malignant cases sent to ROUTINE   : "
                  f"{breakdown['malignant_missed_as_routine']} (these are missed)")
    print("====================================================\n")


def _json_safe(metrics: dict, breakdown: dict) -> dict:
    out = {k: (None if isinstance(v, float) and np.isnan(v) else v)
           for k, v in metrics.items() if k != "confusion_matrix"}
    out["confusion_matrix"] = metrics["confusion_matrix"].tolist()
    out["triage"] = breakdown
    return out


def main(args: argparse.Namespace | None = None) -> None:
    checkpoint = getattr(args, "checkpoint", config.BEST_MODEL_PATH)
    data_dir = getattr(args, "data_dir", config.TRAIN_DIR)

    device = config.DEVICE
    model, backbone, trained = load_model(checkpoint_path=checkpoint, device=device)
    if not trained:
        print(
            "[naseej] warning: no trained checkpoint found "
            f"({checkpoint}). Evaluating an un-fine-tuned model; "
            "numbers below are not meaningful. Run `python -m src.train` first."
        )

    # Reuse the held-out validation split as the evaluation set.
    _, val_loader, _ = build_dataloaders(train_dir=data_dir)
    labels, probs, preds = collect_predictions(model, val_loader, device)
    metrics = compute_metrics(labels, probs, preds)
    breakdown = triage_breakdown(labels, probs)
    print_report(metrics, breakdown)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    cm_path = os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
    save_confusion_matrix(metrics["confusion_matrix"], cm_path)
    if os.path.exists(cm_path):
        print(f"[naseej] confusion matrix saved -> {cm_path}")

    metrics_path = os.path.join(config.OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(_json_safe(metrics, breakdown), fh, indent=2)
    print(f"[naseej] metrics written -> {metrics_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Naseej (lead with sensitivity).")
    parser.add_argument("--checkpoint", default=config.BEST_MODEL_PATH)
    parser.add_argument("--data-dir", default=config.TRAIN_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args())
