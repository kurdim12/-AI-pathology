"""
Temperature scaling — post-hoc probability calibration.

The robustness benchmark exposed a real problem: the model is *overconfident*,
piling probabilities at 0 and 1. That makes the calibrated triage threshold
razor-thin and brittle — a small phone-capture perturbation tips a malignant
case from 0.999 to just under the cut-off and it silently drops to ROUTINE.

Temperature scaling (Guo et al., "On Calibration of Modern Neural Networks",
2017) is the standard fix. It learns a single scalar ``T > 1`` and divides the
logits by it before softmax, spreading the probabilities back out to match the
model's true accuracy — without changing which class wins (so accuracy/AUC are
untouched, only the *confidences* and thus the threshold behaviour improve).

``T`` is fit once on the validation set after training and stored in the
checkpoint; inference/evaluate/calibrate/triage all apply it automatically.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 100,
    lr: float = 0.01,
) -> float:
    """Fit the optimal temperature on held-out logits by minimising NLL.

    Args:
        logits: ``[N, C]`` raw (pre-softmax) model outputs on the val set.
        labels: ``[N]`` integer class labels.

    Returns:
        The fitted temperature ``T`` (a positive float; >1 means the model was
        overconfident). Optimised in log-space so ``T`` stays positive, and
        clamped to a sane range.
    """
    logits = logits.detach().float()
    labels = labels.detach().long()

    # Optimise log T so T = exp(log_T) > 0 for free.
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_t.exp().item())
    # Guard against degenerate fits on tiny/separable val sets.
    return float(min(max(temperature, 0.05), 100.0))


@torch.no_grad()
def expected_calibration_error(
    confidences: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error: gap between confidence and accuracy.

    A well-calibrated model has ECE near 0 (its 80%-confidence predictions are
    right ~80% of the time). Used to report calibration before/after scaling.
    """
    confidences = confidences.detach().float()
    correct = (predictions.detach() == labels.detach()).float()

    bin_edges = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        prop = in_bin.float().mean()
        if prop.item() > 0:
            acc_in_bin = correct[in_bin].mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += (acc_in_bin - conf_in_bin).abs() * prop
    return float(ece.item())


def calibration_report(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> dict:
    """ECE before vs after applying ``temperature`` — for logging."""
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    ece_before = expected_calibration_error(conf, pred, labels)

    probs_t = torch.softmax(logits / temperature, dim=1)
    conf_t, pred_t = probs_t.max(dim=1)
    ece_after = expected_calibration_error(conf_t, pred_t, labels)

    return {
        "temperature": round(float(temperature), 4),
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
    }
