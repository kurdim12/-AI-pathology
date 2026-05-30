"""
Training for Naseej — recall-prioritised fine-tuning.

We fine-tune an ImageNet-pretrained backbone on histopathology with a
class-weighted cross-entropy loss that penalises missed malignancies more than
false alarms (``config.CLASS_WEIGHTS``).

Production touches:
  * **Two-phase fine-tuning** — warm up the new head with the backbone frozen
    (``config.WARMUP_EPOCHS``) before unfreezing everything, so the random head
    doesn't wreck the pretrained features on the first step.
  * **Early stopping** on validation AUC (``config.EARLY_STOP_PATIENCE``).
  * **Mixed precision** on CUDA (``config.USE_AMP``).
  * **Logged history** -> ``outputs/history.json`` and a best-metrics summary.

The best checkpoint (by validation AUC — the metric that best reflects ranking
quality for a triage tool) is written to ``checkpoints/best_model.pt``.

Run::

    python -m src.train
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import config
from src.data import build_dataloaders
from src.model import build_model, get_head

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime
    def tqdm(iterable, **kwargs):
        return iterable


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_scaler(enabled: bool):
    """Create an AMP GradScaler across torch versions (no-op when disabled)."""
    try:  # torch >= 2.4
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None) -> float:
    model.train()
    running, n = 0.0, 0
    use_amp = scaler is not None and scaler.is_enabled()
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        running += loss.item() * images.size(0)
        n += images.size(0)
    return running / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Return ``(val_loss, val_auc, val_acc)``.

    ``val_auc`` is always the binary benign-vs-malignant AUC (the triage-relevant
    ranking metric used for checkpoint selection); in multi-class it is computed
    from the summed malignant probability. ``val_acc`` is class accuracy (subtype
    accuracy in multi-class, binary accuracy otherwise).
    """
    from src.inference import malignant_probability

    model.eval()
    mal_idx = set(config.malignant_indices())
    running, n = 0.0, 0
    mal_probs, bin_labels, class_correct = [], [], 0
    for images, labels in tqdm(loader, desc="val", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        running += criterion(logits, labels).item() * images.size(0)
        n += images.size(0)

        full = torch.softmax(logits, dim=1)
        mal_probs.extend(malignant_probability(full).cpu().tolist())
        bin_labels.extend([1 if int(y) in mal_idx else 0 for y in labels.cpu().tolist()])
        class_correct += int((full.argmax(dim=1) == labels).sum().item())

    val_loss = running / max(n, 1)
    acc = class_correct / max(n, 1)
    auc = float(roc_auc_score(bin_labels, mal_probs)) if len(set(bin_labels)) >= 2 else float("nan")
    return val_loss, auc, acc


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze everything except the classification head."""
    head_param_ids = {id(p) for p in get_head(model, config.BACKBONE).parameters()}
    for param in model.parameters():
        param.requires_grad = trainable or id(param) in head_param_ids


def _save_checkpoint(model, class_names, val_auc, epoch) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "backbone": config.BACKBONE,
            "class_names": class_names,
            "malignant_classes": list(config.MALIGNANT_CLASSES),
            "val_auc": val_auc,
            "epoch": epoch,
            "image_size": config.IMAGE_SIZE,
        },
        config.BEST_MODEL_PATH,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Naseej (recall-prioritised fine-tuning).")
    parser.add_argument("--backbone", default=config.BACKBONE,
                        help="resnet50 | resnet18 | efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help="max fine-tune epochs (phase 2)")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--data-dir", default=config.TRAIN_DIR)
    pre = parser.add_mutually_exclusive_group()
    pre.add_argument("--pretrained", dest="pretrained", action="store_true",
                     help="start from ImageNet weights (default)")
    pre.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                     help="random init (offline / smoke runs)")
    parser.set_defaults(pretrained=config.PRETRAINED)
    return parser.parse_args()


def _apply_overrides(args: argparse.Namespace) -> None:
    """Push CLI overrides into config so the whole module sees one source of truth."""
    config.BACKBONE = args.backbone
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.TRAIN_DIR = args.data_dir
    config.PRETRAINED = args.pretrained


def main(args: argparse.Namespace | None = None) -> None:
    if args is not None:
        _apply_overrides(args)

    set_seed()
    device = config.DEVICE
    print(f"[naseej] device={device} backbone={config.BACKBONE} "
          f"pretrained={config.PRETRAINED} epochs={config.EPOCHS}")

    train_loader, val_loader, class_names = build_dataloaders()
    print(
        f"[naseej] classes={class_names} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)}"
    )

    # Pass config values explicitly: build_model's defaults are bound at import
    # time, so CLI overrides pushed into config must be forwarded here.
    model = build_model(
        backbone=config.BACKBONE,
        num_classes=len(config.CLASS_NAMES),
        pretrained=config.PRETRAINED,
        freeze_backbone=config.FREEZE_BACKBONE,
    ).to(device)
    # Recall-prioritised weighting: malignant class(es) up-weighted (works for
    # both the binary default and multi-class subtype grading).
    weight = torch.tensor(config.loss_class_weights(), dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    use_amp = (device == "cuda") and config.USE_AMP
    scaler = _make_scaler(use_amp)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    history: list = []
    state = {"best_auc": -1.0, "best_epoch": -1, "no_improve": 0, "stop": False}
    global_epoch = 0

    def run_phase(name: str, num_epochs: int, lr: float) -> None:
        nonlocal global_epoch
        if num_epochs <= 0:
            return
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        for _ in range(num_epochs):
            global_epoch += 1
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_loss, val_auc, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()

            print(
                f"[{name} | epoch {global_epoch:02d}] "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_auc={val_auc:.4f} val_acc={val_acc:.4f}"
            )
            history.append(
                {
                    "epoch": global_epoch, "phase": name,
                    "train_loss": train_loss, "val_loss": val_loss,
                    "val_auc": val_auc, "val_acc": val_acc,
                }
            )

            improved = val_auc == val_auc and val_auc > state["best_auc"]
            if improved:
                state.update(best_auc=val_auc, best_epoch=global_epoch, no_improve=0)
                _save_checkpoint(model, class_names, val_auc, global_epoch)
                print(f"           ↳ saved new best (AUC={val_auc:.4f}) -> {config.BEST_MODEL_PATH}")
            else:
                state["no_improve"] += 1
                if config.EARLY_STOP_PATIENCE and state["no_improve"] >= config.EARLY_STOP_PATIENCE:
                    print(f"[naseej] early stopping (no AUC improvement for "
                          f"{config.EARLY_STOP_PATIENCE} epochs).")
                    state["stop"] = True
                    return

    # Phase 1 — head-only warm-up (only meaningful with a pretrained backbone).
    warmup = config.WARMUP_EPOCHS if config.PRETRAINED else 0
    if warmup > 0:
        print(f"[naseej] phase 1: warming up head for {warmup} epoch(s) (backbone frozen)")
        _set_backbone_trainable(model, False)
        run_phase("warmup", warmup, config.LEARNING_RATE)

    # Phase 2 — full fine-tune.
    if not state["stop"]:
        print(f"[naseej] phase 2: fine-tuning full network for up to {config.EPOCHS} epoch(s)")
        _set_backbone_trainable(model, True)
        run_phase("finetune", config.EPOCHS, config.LEARNING_RATE)

    # Persist training history + a best-metrics summary.
    with open(os.path.join(config.OUTPUT_DIR, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)

    if state["best_auc"] < 0:
        print("[naseej] warning: no checkpoint saved (could not compute a valid AUC).")
    else:
        # Fit temperature on the best checkpoint so stored probabilities are
        # calibrated (fixes overconfidence -> less brittle triage thresholds).
        temp_info = _fit_and_store_temperature(val_loader, device)

        summary = {"best_val_auc": state["best_auc"], "best_epoch": state["best_epoch"],
                   "backbone": config.BACKBONE, "epochs_run": global_epoch,
                   **(temp_info or {})}
        with open(os.path.join(config.OUTPUT_DIR, "train_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"[naseej] done. best val AUC={state['best_auc']:.4f} "
              f"(epoch {state['best_epoch']}). Run `python -m src.evaluate`.")


@torch.no_grad()
def _collect_logits(model, loader, device):
    """Gather raw (pre-softmax) logits and labels over a loader."""
    model.eval()
    logits_all, labels_all = [], []
    for images, labels in loader:
        logits_all.append(model(images.to(device)).cpu())
        labels_all.append(labels)
    return torch.cat(logits_all), torch.cat(labels_all)


def _fit_and_store_temperature(val_loader, device) -> dict | None:
    """Reload the best checkpoint, fit temperature on val, and re-save it."""
    from src.temperature import calibration_report, fit_temperature

    if not os.path.exists(config.BEST_MODEL_PATH):
        return None

    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=device, weights_only=False)
    best = build_model(backbone=config.BACKBONE, num_classes=len(config.CLASS_NAMES),
                       pretrained=False).to(device)
    best.load_state_dict(checkpoint["model_state"])

    logits, labels = _collect_logits(best, val_loader, device)
    temperature = fit_temperature(logits, labels)
    report = calibration_report(logits, labels, temperature)

    checkpoint["temperature"] = temperature
    torch.save(checkpoint, config.BEST_MODEL_PATH)
    print(f"[naseej] temperature scaling: T={report['temperature']:.3f} "
          f"(ECE {report['ece_before']:.3f} -> {report['ece_after']:.3f})")
    return report


if __name__ == "__main__":
    main(_parse_args())
