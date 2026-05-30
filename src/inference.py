"""
Inference for Naseej: probability -> triage priority -> auditable heatmap.

This is where the pieces come together for a single slide image:

    image -> preprocess -> CNN -> P(malignant) -> triage priority
                                \-> Grad-CAM    -> heatmap overlay

The triage layer is the product. A raw probability is not actionable on a busy
bench; ``URGENT / REVIEW / ROUTINE`` re-orders the queue so the dangerous cases
surface first. Thresholds live in ``config.py`` and must be re-calibrated on
local data (README §13).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from src.data import eval_transforms
from src.gradcam import GradCAM
from src.model import build_model, get_target_layer

# Triage priority labels, most to least urgent.
PRIORITY_URGENT = "URGENT"
PRIORITY_REVIEW = "REVIEW"
PRIORITY_ROUTINE = "ROUTINE"


@dataclass
class TriageResult:
    """Everything the demo/UI needs to render one decision."""

    label: str                 # predicted class name
    prob_malignant: float      # P(malignant) in [0, 1]
    confidence: float          # P(predicted class) in [0, 1]
    priority: str              # URGENT / REVIEW / ROUTINE
    overlay: Optional[Image.Image] = None   # Grad-CAM blended on the slide
    heatmap: Optional[Image.Image] = None   # raw colourised heatmap
    trained: bool = True       # False -> running on un-fine-tuned weights


def load_model(
    checkpoint_path: str = config.BEST_MODEL_PATH,
    device: str = config.DEVICE,
) -> Tuple[torch.nn.Module, str, bool]:
    """Load the fine-tuned model, or fall back to a pretrained backbone.

    The fallback keeps the demo runnable at a booth before any training has
    happened — but ``trained=False`` is surfaced so predictions are clearly
    flagged as not meaningful yet.

    Returns ``(model, backbone, trained)``.
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        backbone = checkpoint.get("backbone", config.BACKBONE)
        model = build_model(backbone=backbone, pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        trained = True
    else:
        backbone = config.BACKBONE
        # ImageNet weights + random head: structurally valid, not yet meaningful.
        model = build_model(backbone=backbone, pretrained=True)
        trained = False

    model.to(device).eval()
    return model, backbone, trained


def triage_priority(prob_malignant: float) -> str:
    """Map P(malignant) to a triage priority using config thresholds."""
    if prob_malignant >= config.URGENT_THRESHOLD:
        return PRIORITY_URGENT
    if prob_malignant >= config.REVIEW_THRESHOLD:
        return PRIORITY_REVIEW
    return PRIORITY_ROUTINE


def _colorize(cam: np.ndarray) -> np.ndarray:
    """Turn a [H, W] heatmap in [0, 1] into an RGB uint8 image (jet colormap)."""
    try:
        from matplotlib import colormaps

        cmap = colormaps["jet"]
    except Exception:  # pragma: no cover - very old matplotlib
        from matplotlib import cm

        cmap = cm.get_cmap("jet")
    colored = cmap(np.clip(cam, 0.0, 1.0))[..., :3]  # drop alpha channel
    return (colored * 255).astype(np.uint8)


def overlay_heatmap(
    image: Image.Image, cam: np.ndarray, alpha: float = 0.45
) -> Tuple[Image.Image, Image.Image]:
    """Blend a Grad-CAM heatmap over the original image.

    Returns ``(overlay, heatmap)`` both as PIL RGB images sized to the model
    input resolution.
    """
    size = (config.IMAGE_SIZE, config.IMAGE_SIZE)
    base = image.convert("RGB").resize(size)
    base_arr = np.asarray(base).astype(np.float32)

    heat_arr = _colorize(cam).astype(np.float32)
    blended = (1 - alpha) * base_arr + alpha * heat_arr
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended), Image.fromarray(heat_arr.astype(np.uint8))


def analyze(
    image: Image.Image,
    model: Optional[torch.nn.Module] = None,
    backbone: str = config.BACKBONE,
    device: str = config.DEVICE,
    trained: bool = True,
    with_heatmap: bool = True,
) -> TriageResult:
    """Run the full pipeline on one PIL image and return a ``TriageResult``.

    If ``model`` is ``None`` it is loaded from the default checkpoint.
    """
    if model is None:
        model, backbone, trained = load_model(device=device)

    image = image.convert("RGB")
    tensor = eval_transforms()(image).unsqueeze(0).to(device)

    overlay_img: Optional[Image.Image] = None
    heatmap_img: Optional[Image.Image] = None

    if with_heatmap:
        # GradCAM does the forward pass (with gradients) and returns the logits,
        # so we read the probability from the very same pass that made the map.
        cam_engine = GradCAM(model, get_target_layer(model, backbone))
        try:
            cam, logits = cam_engine(tensor, class_idx=config.MALIGNANT_INDEX)
        finally:
            cam_engine.remove()
        cam_np = cam.squeeze(0).cpu().numpy()
        overlay_img, heatmap_img = overlay_heatmap(image, cam_np)
    else:
        with torch.no_grad():
            logits = model(tensor)

    probs = F.softmax(logits, dim=1).squeeze(0)
    prob_malignant = float(probs[config.MALIGNANT_INDEX].item())
    pred_idx = int(torch.argmax(probs).item())

    return TriageResult(
        label=config.CLASS_NAMES[pred_idx],
        prob_malignant=prob_malignant,
        confidence=float(probs[pred_idx].item()),
        priority=triage_priority(prob_malignant),
        overlay=overlay_img,
        heatmap=heatmap_img,
        trained=trained,
    )
