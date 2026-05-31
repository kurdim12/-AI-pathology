"""
Grad-CAM, implemented from scratch (no external CAM library).

Grad-CAM (Selvaraju et al., 2017) explains a CNN's decision by highlighting the
input regions that most increased the score for a chosen class. We:

  1. Hook the last convolutional block to capture its forward activations
     ``A`` and the gradients ``dY/dA`` flowing back from the class score.
  2. Global-average-pool the gradients to get a weight per channel.
  3. Take the ReLU of the weighted sum of activation maps.
  4. Normalise to [0, 1] and upsample to the input resolution.

The result is a heatmap a pathologist can audit: it shows *where* the model
looked, which is exactly the kind of accountability a triage tool needs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM for a single target convolutional layer.

    Example::

        cam_engine = GradCAM(model, target_layer)
        cam, logits = cam_engine(input_tensor, class_idx=1)   # heatmap for "malignant"
        cam_engine.remove()                                   # detach hooks when done
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        # In-place activations (e.g. EfficientNet's SiLU(inplace=True)) clash
        # with backward hooks — autograd refuses to track a view modified in
        # place. Temporarily switch them off; restored in remove().
        self._toggled_inplace = []
        for module in model.modules():
            if getattr(module, "inplace", False):
                module.inplace = False
                self._toggled_inplace.append(module)

        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    # -- hooks ------------------------------------------------------------- #
    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple; [0] is the gradient w.r.t. the layer output.
        self.gradients = grad_output[0].detach()

    # -- main -------------------------------------------------------------- #
    def __call__(
        self, input_tensor: torch.Tensor, class_idx: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the class-activation heatmap(s).

        Args:
            input_tensor: ``[B, C, H, W]`` normalised batch.
            class_idx: class to explain. If ``None``, uses each sample's
                predicted (arg-max) class.

        Returns:
            ``(cam, logits)`` where ``cam`` is ``[B, H, W]`` in [0, 1] and
            ``logits`` is the raw model output ``[B, num_classes]`` (handy so
            callers can read the probability from the same forward pass).
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        logits = self.model(input_tensor)  # [B, num_classes]

        if class_idx is None:
            target = logits.argmax(dim=1)
        else:
            target = torch.full(
                (logits.size(0),), int(class_idx), dtype=torch.long, device=logits.device
            )

        # Sum the chosen-class scores across the batch so a single backward call
        # populates per-sample gradients.
        score = logits.gather(1, target.unsqueeze(1)).sum()

        self.model.zero_grad(set_to_none=True)
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not fire. Is the target layer part of the "
                "model's forward pass?"
            )

        # Channel weights = global-average-pooled gradients.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, K, 1, 1]
        cam = (weights * self.activations).sum(dim=1)            # [B, h, w]
        cam = F.relu(cam)

        # Per-sample min-max normalisation to [0, 1].
        b = cam.size(0)
        flat = cam.view(b, -1)
        cam_min = flat.min(dim=1).values.view(b, 1, 1)
        cam_max = flat.max(dim=1).values.view(b, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Upsample to the input resolution.
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        return cam.detach(), logits.detach()

    def remove(self) -> None:
        """Detach the hooks and restore in-place activations. Call when done."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        for module in self._toggled_inplace:
            module.inplace = True
        self._toggled_inplace = []

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()
