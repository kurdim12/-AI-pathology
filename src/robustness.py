"""
Phone-capture robustness benchmark — evidence for Naseej's central claim.

The project's thesis is that the model keeps working on messy phone photos, not
just clean scans. That claim has to be *measured*, not asserted. This benchmark
takes the held-out validation set, re-scores it at increasing phone-degradation
severities (``src/phone_sim.py``), and reports how the triage-critical metrics
hold up — above all the **malignant recall captured at the REVIEW threshold**,
i.e. the share of cancers still flagged as capture quality drops.

It writes ``outputs/robustness.json`` and a degradation curve to
``outputs/robustness.png``.

Run::

    python -m src.robustness
    python -m src.robustness --severities 0,0.25,0.5,0.75,1.0
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

import config
from src.data import build_dataloaders, eval_transforms
from src.inference import (
    load_model,
    load_thresholds,
    softmax_with_temperature,
    triage_priority,
)
from src.phone_sim import degrade

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def _val_paths_labels() -> tuple[list[str], list[int]]:
    """Recover the (path, label) pairs of the held-out validation split.

    Rebuilds the same seeded split ``build_dataloaders`` uses, but keeps the
    file paths so we can re-open and degrade each image at eval time.
    """
    train_loader, val_loader, _ = build_dataloaders()
    # val_loader wraps a Subset(ImageFolder) — pull paths via the underlying
    # samples and the subset indices.
    subset = val_loader.dataset
    base = subset.dataset           # ImageFolder
    indices = subset.indices
    paths = [base.samples[i][0] for i in indices]
    labels = [base.samples[i][1] for i in indices]
    return paths, labels


@torch.no_grad()
def _score(model, image: Image.Image, device: str) -> float:
    tensor = eval_transforms()(image.convert("RGB")).unsqueeze(0).to(device)
    probs = softmax_with_temperature(model(tensor), model).squeeze(0)
    return float(probs[config.MALIGNANT_INDEX].item())


def evaluate_at_severity(
    model, paths: List[str], labels: List[int], severity: float, device: str
) -> dict:
    """Re-score the val set at one degradation severity; return triage metrics."""
    urgent_t, review_t = load_thresholds()
    probs, y = [], []
    for path, label in zip(paths, labels):
        img = Image.open(path)
        if severity > 0:
            img = degrade(img, severity)
        probs.append(_score(model, img, device))
        y.append(int(label))

    probs_a, y_a = np.array(probs), np.array(y)
    pos = y_a == config.MALIGNANT_INDEX
    neg = ~pos

    # Recall at the (calibrated) REVIEW cut-off = cancers still flagged.
    flagged = probs_a[pos] >= review_t
    recall_at_review = float(np.mean(flagged)) if pos.any() else float("nan")

    # Sensitivity/specificity at the naive 0.5 line, plus AUC for ranking.
    sens = float(np.mean(probs_a[pos] >= 0.5)) if pos.any() else float("nan")
    spec = float(np.mean(probs_a[neg] < 0.5)) if neg.any() else float("nan")
    auc = float(roc_auc_score(y_a, probs_a)) if len(set(y)) > 1 else float("nan")

    missed = int(np.sum(probs_a[pos] < review_t)) if pos.any() else 0
    return {
        "severity": round(float(severity), 4),
        "recall_at_review": round(recall_at_review, 4),
        "sensitivity_at_0.5": round(sens, 4),
        "specificity_at_0.5": round(spec, 4),
        "auc": round(auc, 4),
        "malignant_missed_as_routine": missed,
        "n": int(len(y)),
    }


def save_curve(results: List[dict], out_path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sev = [r["severity"] for r in results]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(sev, [r["recall_at_review"] for r in results], "o-", label="malignant recall @ REVIEW", color="#b00020", linewidth=2)
    ax.plot(sev, [r["auc"] for r in results], "s--", label="AUC", color="#1b5e9b")
    ax.plot(sev, [r["specificity_at_0.5"] for r in results], "^:", label="specificity @ 0.5", color="#1b7a3d")
    ax.set_xlabel("phone-capture degradation severity")
    ax.set_ylabel("metric")
    ax.set_ylim(0, 1.02)
    ax.set_title("Naseej — triage robustness vs phone-capture degradation")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def print_report(results: List[dict]) -> None:
    print("\n============ Naseej · phone-capture robustness ============")
    print(f"  {'severity':>8}  {'recall@REVIEW':>13}  {'AUC':>6}  {'spec@.5':>7}  {'missed':>6}")
    for r in results:
        print(f"  {r['severity']:>8.2f}  {r['recall_at_review']:>13.3f}  "
              f"{r['auc']:>6.3f}  {r['specificity_at_0.5']:>7.3f}  "
              f"{r['malignant_missed_as_routine']:>6d}")
    base, worst = results[0], results[-1]
    drop = base["recall_at_review"] - worst["recall_at_review"]
    print("  ---------------------------------------------------------")
    print(f"  recall@REVIEW drop from clean -> worst: {drop:+.3f}")
    print("  (lower severity = cleaner capture; recall@REVIEW is the number")
    print("   that matters — cancers still flagged for a pathologist)")
    print("===========================================================\n")


def main(args: argparse.Namespace | None = None) -> None:
    severities = (args.severities if args else None) or [0.0, 0.25, 0.5, 0.75, 1.0]
    device = config.DEVICE

    model, backbone, trained = load_model()
    if not trained:
        print("[naseej] warning: no trained checkpoint — robustness numbers are "
              "meaningless on an un-fine-tuned model. Run `python -m src.train` first.")

    paths, labels = _val_paths_labels()
    results = [
        evaluate_at_severity(model, paths, labels, s, device)
        for s in tqdm(severities, desc="severities")
    ]
    print_report(results)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUT_DIR, "robustness.json"), "w") as fh:
        json.dump({"backbone": backbone, "trained": trained, "results": results}, fh, indent=2)
    curve_path = os.path.join(config.OUTPUT_DIR, "robustness.png")
    save_curve(results, curve_path)
    print(f"[naseej] robustness written -> {config.OUTPUT_DIR}/robustness.json"
          + (f" + {curve_path}" if os.path.exists(curve_path) else ""))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure triage robustness to phone-capture degradation.")
    parser.add_argument("--severities", type=str, default=None,
                        help="comma-separated 0..1 levels, e.g. 0,0.25,0.5,0.75,1.0")
    args = parser.parse_args()
    if args.severities:
        args.severities = [float(x) for x in args.severities.split(",")]
    return args


if __name__ == "__main__":
    main(_parse_args())
