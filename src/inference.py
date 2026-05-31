r"""
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

import json
import os
from dataclasses import dataclass

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


def load_thresholds(path: str | None = None) -> tuple[float, float]:
    """Return ``(urgent_threshold, review_threshold)``.

    Prefers the calibrated values written by ``src.calibrate`` (so a freshly
    calibrated operating point is used automatically); falls back to the
    defaults in ``config`` when no calibration file exists. ``path`` is resolved
    from ``config.THRESHOLDS_PATH`` at call time (not bound at import).
    """
    if path is None:
        path = config.THRESHOLDS_PATH
    if os.path.exists(path):
        try:
            with open(path) as fh:
                data = json.load(fh)
            return float(data["urgent_threshold"]), float(data["review_threshold"])
        except (KeyError, ValueError, json.JSONDecodeError):
            pass  # malformed file -> fall back to config defaults
    return config.URGENT_THRESHOLD, config.REVIEW_THRESHOLD


@dataclass
class TriageResult:
    """Everything the demo/UI needs to render one decision."""

    label: str                 # predicted class name (subtype, in multi-class)
    prob_malignant: float      # P(malignant) in [0, 1] (summed over malignant classes)
    confidence: float          # P(predicted class) in [0, 1]
    priority: str              # URGENT / REVIEW / ROUTINE
    overlay: Image.Image | None = None   # Grad-CAM blended on the slide
    heatmap: Image.Image | None = None   # raw colourised heatmap
    trained: bool = True       # False -> running on un-fine-tuned weights
    is_malignant_class: bool = False        # predicted class is a malignant subtype
    uncertain: bool = False     # P(malignant) near 0.5 -> model is effectively guessing
    quality_ok: bool = True     # image passed the tissue/focus quality check
    quality_reason: str = "ok"  # why quality failed (when quality_ok is False)


def load_model(
    checkpoint_path: str = config.BEST_MODEL_PATH,
    device: str = config.DEVICE,
) -> tuple[torch.nn.Module, str, bool]:
    """Load the fine-tuned model, or fall back to a pretrained backbone.

    The fallback keeps the demo runnable at a booth before any training has
    happened — but ``trained=False`` is surfaced so predictions are clearly
    flagged as not meaningful yet.

    The fitted temperature (if any) is attached as ``model.temperature`` so the
    whole pipeline applies calibrated probabilities transparently.

    Returns ``(model, backbone, trained)``.
    """
    temperature = 1.0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        backbone = checkpoint.get("backbone", config.BACKBONE)

        # Restore the checkpoint's taxonomy so multi-class models load with the
        # right head size and the correct malignant grouping for P(malignant).
        class_names = checkpoint.get("class_names", config.CLASS_NAMES)
        mal_classes = checkpoint.get("malignant_classes", config.MALIGNANT_CLASSES)
        if class_names != config.CLASS_NAMES or list(mal_classes) != list(config.MALIGNANT_CLASSES):
            config.use_multiclass(class_names, mal_classes)

        model = build_model(backbone=backbone, num_classes=len(class_names), pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        temperature = float(checkpoint.get("temperature", 1.0))
        trained = True
    else:
        backbone = config.BACKBONE
        # No checkpoint: prefer ImageNet weights (structurally valid, not yet
        # meaningful — trained=False is surfaced). If the weights can't be
        # downloaded (offline lab / air-gapped / CI), degrade to random init
        # rather than crash; predictions are placeholders either way.
        try:
            model = build_model(backbone=backbone, pretrained=True)
        except Exception:
            model = build_model(backbone=backbone, pretrained=False)
        trained = False

    model.to(device).eval()
    model.temperature = temperature  # type: ignore[attr-defined]
    return model, backbone, trained


def softmax_with_temperature(logits: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """Softmax that applies the model's fitted temperature (default 1.0 = none).

    Temperature scaling only rescales confidences; it never changes which class
    wins, so labels/AUC are unaffected while the probabilities (and therefore
    the triage thresholds) become better calibrated.
    """
    temperature = float(getattr(model, "temperature", 1.0))
    return F.softmax(logits / temperature, dim=1)


def malignant_probability(probs: torch.Tensor) -> torch.Tensor:
    """Collapse per-class probabilities to a single P(malignant).

    Binary: this is just the malignant column. Multi-class (subtype grading):
    the sum of the probabilities over all malignant subtypes. Accepts ``[C]`` or
    ``[B, C]`` and returns a scalar tensor or ``[B]`` accordingly. This is the
    one place the binary-vs-multiclass distinction is resolved, so triage,
    calibration and robustness need no special-casing.
    """
    idx = config.malignant_indices()
    dim = probs.dim() - 1
    if not idx:
        # Degenerate taxonomy with no malignant class -> P(malignant) = 0.
        return probs.sum(dim=dim) * 0.0
    index = torch.tensor(idx, dtype=torch.long, device=probs.device)
    return probs.index_select(dim, index).sum(dim=dim)


def triage_priority(
    prob_malignant: float,
    urgent_threshold: float | None = None,
    review_threshold: float | None = None,
) -> str:
    """Map P(malignant) to a triage priority.

    With no thresholds passed, uses the calibrated values if present, else the
    config defaults. Explicit thresholds (e.g. from a config sweep) win.
    """
    if urgent_threshold is None or review_threshold is None:
        cal_urgent, cal_review = load_thresholds()
        urgent_threshold = cal_urgent if urgent_threshold is None else urgent_threshold
        review_threshold = cal_review if review_threshold is None else review_threshold

    if prob_malignant >= urgent_threshold:
        return PRIORITY_URGENT
    if prob_malignant >= review_threshold:
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
) -> tuple[Image.Image, Image.Image]:
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


@torch.no_grad()
def _tta_probs(model: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """Average softmax probabilities over label-preserving views.

    Histopathology has no canonical orientation, so horizontal/vertical flips
    and 90° rotations are all valid views of the same tissue. Averaging over
    them steadies predictions on noisy phone images — directly supporting the
    phone-capture robustness goal.
    """
    views = [
        tensor,
        torch.flip(tensor, dims=[3]),          # horizontal flip
        torch.flip(tensor, dims=[2]),          # vertical flip
        torch.rot90(tensor, k=1, dims=[2, 3]),  # 90°
        torch.rot90(tensor, k=3, dims=[2, 3]),  # 270°
    ]
    probs = [softmax_with_temperature(model(v), model) for v in views]
    return torch.stack(probs, dim=0).mean(dim=0)


def analyze(
    image: Image.Image,
    model: torch.nn.Module | None = None,
    backbone: str = config.BACKBONE,
    device: str = config.DEVICE,
    trained: bool = True,
    with_heatmap: bool = True,
    tta: bool = config.TTA_ENABLED,
    thresholds: tuple[float, float] | None = None,
    check_quality: bool = False,
) -> TriageResult:
    """Run the full pipeline on one PIL image and return a ``TriageResult``.

    Args:
        model: if ``None``, loaded from the default checkpoint.
        with_heatmap: also compute a Grad-CAM overlay (needs a backward pass).
        tta: average over flips/rotations for steadier probabilities. The
            Grad-CAM map is always computed on the canonical (un-augmented) view.
        thresholds: optional ``(urgent, review)`` cut-offs; defaults to the
            calibrated values if present, else config.
        check_quality: run the tissue/focus quality gate first; an unusable
            capture short-circuits to a REVIEW result flagged for re-capture
            rather than a (meaningless) confident classification.
    """
    if model is None:
        model, backbone, trained = load_model(device=device)

    image = image.convert("RGB")

    if check_quality:
        from src.quality import assess_quality

        q = assess_quality(image)
        if not q.usable:
            # Don't trust a model call on an unusable frame. Surface it for a
            # human (REVIEW) and explain why, rather than emitting a fake label.
            return TriageResult(
                label="Indeterminate",
                prob_malignant=float("nan"),
                confidence=0.0,
                priority=PRIORITY_REVIEW,
                trained=trained,
                uncertain=True,
                quality_ok=False,
                quality_reason=q.reason,
            )

    tensor = eval_transforms()(image).unsqueeze(0).to(device)

    overlay_img: Image.Image | None = None
    heatmap_img: Image.Image | None = None

    if with_heatmap:
        # Explain the most-probable malignant class (in binary that's simply the
        # malignant class). Pick it from a no-grad pass, then Grad-CAM on it so
        # the heatmap highlights the tissue driving the malignancy signal.
        with torch.no_grad():
            pre = softmax_with_temperature(model(tensor), model).squeeze(0)
        mal_idx = config.malignant_indices()
        cam_target = max(mal_idx, key=lambda i: float(pre[i])) if mal_idx else int(pre.argmax())

        cam_engine = GradCAM(model, get_target_layer(model, backbone))
        try:
            cam, logits = cam_engine(tensor, class_idx=cam_target)
        finally:
            cam_engine.remove()
        cam_np = cam.squeeze(0).cpu().numpy()
        overlay_img, heatmap_img = overlay_heatmap(image, cam_np)
        canonical_probs = softmax_with_temperature(logits, model)
    else:
        with torch.no_grad():
            canonical_probs = softmax_with_temperature(model(tensor), model)

    # TTA averages the *classification* probabilities; the heatmap stays on the
    # canonical view above (rotated heatmaps wouldn't align with the image).
    probs = (_tta_probs(model, tensor) if tta else canonical_probs).squeeze(0)
    prob_malignant = float(malignant_probability(probs).item())
    pred_idx = int(torch.argmax(probs).item())

    return _result_from_probs(
        probs, prob_malignant, pred_idx, trained, thresholds,
        overlay=overlay_img, heatmap=heatmap_img,
    )


def _result_from_probs(probs, prob_malignant, pred_idx, trained, thresholds,
                       overlay=None, heatmap=None) -> TriageResult:
    """Build a TriageResult, applying triage bands + the abstention rule.

    Shared by single-image ``analyze`` and ``predict_batch`` so the
    uncertainty/abstention behaviour is identical on both paths.
    """
    urgent_t, review_t = thresholds if thresholds is not None else (None, None)

    # Abstention: if P(malignant) sits within UNCERTAIN_MARGIN of 0.5 the model
    # is effectively guessing. Flag it and never let such a case fall to ROUTINE
    # — it gets at least a REVIEW so a human looks.
    uncertain = (config.UNCERTAIN_MARGIN > 0
                 and abs(prob_malignant - 0.5) < config.UNCERTAIN_MARGIN)
    priority = triage_priority(prob_malignant, urgent_t, review_t)
    if uncertain and priority == PRIORITY_ROUTINE:
        priority = PRIORITY_REVIEW

    return TriageResult(
        label=config.CLASS_NAMES[pred_idx],
        prob_malignant=prob_malignant,
        confidence=float(probs[pred_idx].item()),
        priority=priority,
        overlay=overlay,
        heatmap=heatmap,
        trained=trained,
        is_malignant_class=(pred_idx in config.malignant_indices()),
        uncertain=uncertain,
    )


@torch.no_grad()
def predict_batch(
    images,
    model: torch.nn.Module | None = None,
    backbone: str = config.BACKBONE,
    device: str = config.DEVICE,
    trained: bool = True,
    batch_size: int = config.BATCH_SIZE,
    thresholds: tuple[float, float] | None = None,
):
    """Triage many PIL images efficiently in batched forward passes.

    No Grad-CAM (that needs a per-image backward pass) — this is the fast path
    for queue triage where only the label/probability/priority is needed.
    Returns a list of ``TriageResult`` aligned with ``images``.
    """
    if model is None:
        model, backbone, trained = load_model(device=device)

    tfm = eval_transforms()
    results = []
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]
        batch = torch.stack([tfm(im.convert("RGB")) for im in chunk]).to(device)
        probs = softmax_with_temperature(model(batch), model)
        mal = malignant_probability(probs)
        for i in range(probs.size(0)):
            row = probs[i]
            results.append(_result_from_probs(
                row, float(mal[i].item()), int(row.argmax().item()),
                trained, thresholds,
            ))
    return results
