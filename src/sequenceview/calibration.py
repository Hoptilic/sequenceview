from __future__ import annotations

import torch
from torch import nn


def expected_calibration_error(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    confidences, predictions = probabilities.max(dim=1)
    correct = (predictions == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    error = torch.zeros(())
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > low) & (confidences <= high)
        if in_bin.any():
            error += in_bin.float().mean() * (correct[in_bin].mean() - confidences[in_bin].mean()).abs()
    return float(error)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, steps: int = 200) -> float:
    """Fit a single temperature on held-out logits by minimising cross-entropy."""
    log_temperature = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=steps)
    loss_function = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_function(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().item())
