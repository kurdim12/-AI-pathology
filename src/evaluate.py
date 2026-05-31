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
    malignant_probability,
    softmax_with_temperature,
    triage_priority,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Return ``(binary_labels, probs_malignant, binary_preds)`` as numpy arrays.

    Works for binary *and* multi-class models: ``probs_malignant`` is the summed
    probability over malignant classes, and labels/preds are collapsed to the
    binary benign(0)/malignant(1) axis that triage cares about. Applies the
    model's fitted temperature so probabilities match inference/calibration.
    """
    model.eval()
    mal_idx = set(config.malignant_indices())
    all_labels, all_probs, all_preds = [], [], []
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        full_probs = softmax_with_temperature(model(images), model)
        probs = malignant_probability(full_probs)
        preds = (probs >= 0.5).long()

        # Collapse true class labels to binary malignant membership.
        bin_labels = [1 if int(y) in mal_idx else 0 for y in labels.tolist()]
        all_labels.extend(bin_labels)
        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    return np.array(all_labels), np.array(all_probs), np.array(all_preds)


@torch.no_grad()
def collect_class_predictions(model, loader, device):
    """Return ``(true_class, pred_class)`` over all classes — for the per-subtype
    confusion matrix in multi-class mode."""
    model.eval()
    true_c, pred_c = [], []
    for images, labels in tqdm(loader, desc="eval-cls", leave=False):
        probs = softmax_with_temperature(model(images.to(device)), model)
        true_c.extend(labels.tolist())
        pred_c.extend(probs.argmax(dim=1).cpu().tolist())
    return np.array(true_c), np.array(pred_c)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> dict:
    """Compute binary triage metrics from (binary) labels/probabilities/predictions."""
    if len(labels) == 0:
        raise ValueError(
            "No validation samples to evaluate. The validation split is empty — "
            "use more data or a larger VAL_SPLIT."
        )
    # Confusion matrix on the binary benign(0)/malignant(1) axis.
    cm = confusion_matrix(labels, preds, labels=[0, 1])
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

    # labels here are already binary (1 = malignant) from collect_predictions.
    for label, p in zip(labels, probs):
        band = triage_priority(float(p), urgent_t, review_t)
        key = "malignant" if int(label) == 1 else "benign"
        bands[band][key] += 1

    n_malignant = int(np.sum(labels == 1))
    flagged = bands[PRIORITY_URGENT]["malignant"] + bands[PRIORITY_REVIEW]["malignant"]
    recall_at_review = flagged / n_malignant if n_malignant else float("nan")

    return {
        "urgent_threshold": urgent_t,
        "review_threshold": review_t,
        "bands": bands,
        "malignant_recall_at_review": recall_at_review,
        "malignant_missed_as_routine": bands[PRIORITY_ROUTINE]["malignant"],
    }


def save_confusion_matrix(cm: np.ndarray, out_path: str, class_names=None,
                          title: str = "Confusion matrix") -> None:
    """Save a labelled confusion-matrix figure (best-effort).

    ``class_names`` defaults to the binary triage axis; pass subtype names for a
    multi-class matrix.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return

    names = class_names if class_names is not None else ["Benign", "Malignant"]
    n = len(names)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(4.5, n * 0.9), max(4, n * 0.85)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n), labels=names, rotation=45, ha="right")
    ax.set_yticks(range(n), labels=names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
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
    print("                 pred Benign   pred Malignant")
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
    save_confusion_matrix(metrics["confusion_matrix"], cm_path,
                          title="Triage confusion matrix (benign vs malignant)")
    if os.path.exists(cm_path):
        print(f"[naseej] confusion matrix saved -> {cm_path}")

    # In multi-class mode, also report the per-subtype confusion matrix.
    if config.is_multiclass():
        true_c, pred_c = collect_class_predictions(model, val_loader, device)
        n = len(config.CLASS_NAMES)
        cls_cm = confusion_matrix(true_c, pred_c, labels=list(range(n)))
        subtype_acc = float(np.trace(cls_cm) / cls_cm.sum()) if cls_cm.sum() else float("nan")
        print(f"  ---- multi-class grading ({n} subtypes) ----")
        print(f"  subtype accuracy : {subtype_acc:.4f}")
        cls_path = os.path.join(config.OUTPUT_DIR, "confusion_matrix_subtypes.png")
        save_confusion_matrix(cls_cm, cls_path, class_names=config.CLASS_NAMES,
                              title="Subtype confusion matrix")
        if os.path.exists(cls_path):
            print(f"[naseej] subtype confusion matrix saved -> {cls_path}")

    metrics_path = os.path.join(config.OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(_json_safe(metrics, breakdown), fh, indent=2)
    print(f"[naseej] metrics written -> {metrics_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Naseej (lead with sensitivity).")
    parser.add_argument("--checkpoint", default=config.BEST_MODEL_PATH)
    parser.add_argument("--data-dir", default=config.TRAIN_DIR)
    return parser.parse_args()


def _cli() -> None:
    """Console entry-point (naseej-eval)."""
    main(_parse_args())


if __name__ == "__main__":
    _cli()
