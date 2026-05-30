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

import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

import config
from src.data import build_dataloaders
from src.inference import load_model

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


def print_report(metrics: dict) -> None:
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()
    print("\n================ Naseej · evaluation ================")
    print(f"  Samples evaluated : {metrics['support']}")
    print("  ---- triage metrics (lead with sensitivity) ----")
    print(f"  Sensitivity (recall) : {metrics['sensitivity']:.4f}   <-- missed cancers matter most")
    print(f"  Specificity          : {metrics['specificity']:.4f}")
    print(f"  AUC                  : {metrics['auc']:.4f}")
    print(f"  Accuracy             : {metrics['accuracy']:.4f}")
    print("  ---- confusion matrix ----")
    print(f"                 pred Benign   pred Malignant")
    print(f"  true Benign        {tn:6d}          {fp:6d}")
    print(f"  true Malignant     {fn:6d}          {tp:6d}")
    print("====================================================\n")


def main() -> None:
    device = config.DEVICE
    model, backbone, trained = load_model(device=device)
    if not trained:
        print(
            "[naseej] warning: no trained checkpoint found "
            f"({config.BEST_MODEL_PATH}). Evaluating an un-fine-tuned model; "
            "numbers below are not meaningful. Run `python -m src.train` first."
        )

    # Reuse the held-out validation split as the evaluation set.
    _, val_loader, _ = build_dataloaders()
    labels, probs, preds = collect_predictions(model, val_loader, device)
    metrics = compute_metrics(labels, probs, preds)
    print_report(metrics)

    cm_path = os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
    save_confusion_matrix(metrics["confusion_matrix"], cm_path)
    if os.path.exists(cm_path):
        print(f"[naseej] confusion matrix saved -> {cm_path}")


if __name__ == "__main__":
    main()
