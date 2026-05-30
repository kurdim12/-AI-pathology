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
    raise ValueError(f"Unsupported backbone {backbone!r}.")


def build_foundation_model(name: str = "ctranspath", num_classes: int = len(config.CLASS_NAMES)):
    """Upgrade path (stub): swap the ImageNet backbone for an open pathology
    foundation model.

    Models such as CTransPath, Phikon and UNI are pretrained on millions of
    histopathology patches and give state-of-the-art features with very little
    extra data. To wire one in:

        1. ``pip install`` the relevant package / download the released weights.
        2. Load the encoder and expose its final feature map for Grad-CAM.
        3. Attach an ``nn.Linear(feature_dim, num_classes)`` head.
        4. Point ``get_target_layer`` at the encoder's last spatial block.

    This is intentionally left as a stub: it documents the path without pulling
    a heavy dependency into the core demo.
    """
    raise NotImplementedError(
        "Foundation-model backbone is the documented upgrade path. "
        "Load CTransPath / Phikon / UNI weights here and attach a 2-class head. "
        "See README §4 (Upgrade path) and src/model.py:build_foundation_model."
    )
