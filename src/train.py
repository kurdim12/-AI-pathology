"""
Training for Naseej — recall-prioritised fine-tuning.

We fine-tune an ImageNet-pretrained backbone on histopathology with a
class-weighted cross-entropy loss that penalises missed malignancies more than
false alarms (``config.CLASS_WEIGHTS``). The best checkpoint is selected by
validation AUC — the metric that best reflects ranking quality for a triage
tool — and written to ``checkpoints/best_model.pt``.

Run::

    python -m src.train
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import config
from src.data import build_dataloaders
from src.model import build_model

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


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running = 0.0
    n = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running += loss.item() * images.size(0)
        n += images.size(0)
    return running / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Return ``(val_loss, val_auc, val_acc)``."""
    model.eval()
    running = 0.0
    n = 0
    all_probs, all_labels = [], []
    for images, labels in tqdm(loader, desc="val", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        running += loss.item() * images.size(0)
        n += images.size(0)

        probs = torch.softmax(logits, dim=1)[:, config.MALIGNANT_INDEX]
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    val_loss = running / max(n, 1)
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    acc = float(np.mean([int(p == t) for p, t in zip(preds, all_labels)]))

    # AUC needs both classes present in the validation split.
    if len(set(all_labels)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(all_labels, all_probs))
    return val_loss, auc, acc


def main() -> None:
    set_seed()
    device = config.DEVICE
    print(f"[naseej] device={device} backbone={config.BACKBONE}")

    train_loader, val_loader, class_names = build_dataloaders()
    print(
        f"[naseej] classes={class_names} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)}"
    )

    model = build_model().to(device)

    weight = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_auc = -1.0

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"[epoch {epoch:02d}/{config.EPOCHS}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auc={val_auc:.4f} val_acc={val_acc:.4f}"
        )

        # Select on AUC; NaN (single-class val) never beats a real score.
        if val_auc == val_auc and val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "backbone": config.BACKBONE,
                    "class_names": class_names,
                    "val_auc": val_auc,
                    "image_size": config.IMAGE_SIZE,
                },
                config.BEST_MODEL_PATH,
            )
            print(f"           ↳ saved new best (AUC={best_auc:.4f}) -> {config.BEST_MODEL_PATH}")

    if best_auc < 0:
        print("[naseej] warning: no checkpoint saved (could not compute a valid AUC).")
    else:
        print(f"[naseej] done. best val AUC={best_auc:.4f}")


if __name__ == "__main__":
    main()
