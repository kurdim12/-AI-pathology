"""
Model definition for Naseej.

Transfer learning from an ImageNet-pretrained CNN with a fresh 2-class head
(benign vs malignant). The backbone is selectable; ResNet-50 is the default.

The proprietary work of the project is *not* the architecture (these are
standard torchvision backbones) but the fine-tuning, the phone-robustness
augmentation pipeline (``src/data.py``) and the triage logic
(``src/inference.py``). A stub for swapping in an open pathology foundation
model is included as the documented upgrade path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

import config

# Backbones we know how to build a head for and where their Grad-CAM target
# layer lives.
SUPPORTED_BACKBONES = ("resnet50", "resnet18", "efficientnet_b0")


def _normalise_name(backbone: str) -> str:
    name = backbone.lower().replace("-", "_")
    if name in ("efficientnetb0", "efficientnet_b0"):
        return "efficientnet_b0"
    return name


def build_model(
    backbone: str = config.BACKBONE,
    num_classes: int = len(config.CLASS_NAMES),
    pretrained: bool = config.PRETRAINED,
    freeze_backbone: bool = config.FREEZE_BACKBONE,
) -> nn.Module:
    """Build a CNN classifier with a fresh ``num_classes`` head.

    Args:
        backbone: one of ``SUPPORTED_BACKBONES``.
        num_classes: size of the new classification head (2 for Naseej).
        pretrained: start from ImageNet weights (the whole point of transfer
            learning here — histopathology datasets are too small to train from
            scratch reliably).
        freeze_backbone: if True, freeze every layer except the new head. Useful
            for a quick warm-up before full fine-tuning.
    """
    name = _normalise_name(backbone)

    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unsupported backbone {backbone!r}. "
            f"Choose one of {SUPPORTED_BACKBONES}."
        )

    if freeze_backbone:
        _freeze_to_head(model, name)

    return model


def _freeze_to_head(model: nn.Module, name: str) -> None:
    """Freeze all parameters, then re-enable gradients on the new head only."""
    for param in model.parameters():
        param.requires_grad = False
    head = get_head(model, name)
    for param in head.parameters():
        param.requires_grad = True


def get_head(model: nn.Module, backbone: str = config.BACKBONE) -> nn.Module:
    """Return the classification head module for a given backbone."""
    name = _normalise_name(backbone)
    if name in ("resnet50", "resnet18"):
        return model.fc
    if name == "efficientnet_b0":
        return model.classifier
    raise ValueError(f"Unsupported backbone {backbone!r}.")


def get_target_layer(model: nn.Module, backbone: str = config.BACKBONE) -> nn.Module:
    """Return the convolutional layer Grad-CAM should hook.

    Grad-CAM uses the activations and gradients of the last convolutional
    block, where spatial information is still present but features are highly
    semantic.
    """
    name = _normalise_name(backbone)
    if name in ("resnet50", "resnet18"):
        return model.layer4[-1]
    if name == "efficientnet_b0":
        return model.features[-1]
    # Unknown backbone (e.g. a timm foundation model): find it by probing.
    return autodetect_target_layer(model)


def autodetect_target_layer(model: nn.Module, input_size: int = config.IMAGE_SIZE) -> nn.Module:
    """Find the last module that emits a 4D feature map for Grad-CAM.

    For backbones we don't recognise explicitly, run a dummy forward pass and
    pick the last leaf module whose output is ``[B, C, H, W]`` with spatial
    extent > 1 — the natural place to hook a class-activation map.
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:  # pragma: no cover - model with no params
        device = torch.device("cpu")

    candidates: list[nn.Module] = []
    handles = []

    def _hook(module, inputs, output):
        out = output[0] if isinstance(output, (tuple, list)) and output else output
        if torch.is_tensor(out) and out.dim() == 4 and out.shape[2] > 1 and out.shape[3] > 1:
            candidates.append(module)

    for module in model.modules():
        if len(list(module.children())) == 0:  # leaf modules only
            handles.append(module.register_forward_hook(_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, 3, input_size, input_size, device=device))
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    if not candidates:
        raise RuntimeError(
            "Could not autodetect a 4D feature map for Grad-CAM. "
            "Pass an explicit target layer for this backbone."
        )
    return candidates[-1]


def build_foundation_model(
    name: str | None = None,
    num_classes: int = len(config.CLASS_NAMES),
    weights_path: str | None = None,
) -> nn.Module:
    """Upgrade path (optional, real): swap the ImageNet CNN for an open
    vision / pathology foundation backbone via ``timm``.

    Models such as CTransPath, Phikon and UNI are pretrained on millions of
    histopathology patches and give state-of-the-art features with little extra
    data. This loader keeps ``timm`` out of the core dependencies — it is only
    imported when you actually opt in.

    Usage::

        # config.FOUNDATION_MODEL = "timm:convnext_tiny"
        model = build_foundation_model()                 # reads config
        model = build_foundation_model("timm:vit_base_patch16_224")

        # released pathology weights downloaded locally:
        model = build_foundation_model(
            "timm:swin_tiny_patch4_window7_224",
            weights_path="weights/ctranspath.pth",
        )

    Grad-CAM works on these models via :func:`autodetect_target_layer` (best on
    convolutional / hybrid backbones; pure ViTs expose no native spatial map).
    """
    name = name or config.FOUNDATION_MODEL
    weights_path = weights_path or config.FOUNDATION_WEIGHTS

    if not name:
        raise ValueError(
            "No foundation model specified. Set config.FOUNDATION_MODEL "
            '(e.g. "timm:convnext_tiny") or pass name=...'
        )
    if not name.startswith("timm:"):
        raise ValueError('Foundation model name must be of the form "timm:<model_name>".')

    try:
        import timm
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The foundation-model path needs `timm` (pip install timm). "
            "The default CNN backbones do not require it."
        ) from exc

    timm_name = name.split("timm:", 1)[1]
    # If loading released weights ourselves, don't also pull timm's pretrained.
    model = timm.create_model(
        timm_name, pretrained=(weights_path is None), num_classes=num_classes
    )

    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = state.get("model", state.get("state_dict", state))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[naseej] loaded foundation weights from {weights_path}: "
            f"{len(missing)} missing / {len(unexpected)} unexpected keys"
        )
    return model
