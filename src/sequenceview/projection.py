from __future__ import annotations

import torch


def pca_2d(vectors: list[list[float]]) -> tuple[list[list[float]], list[float]]:
    """Project row vectors onto their first two principal components.

    Returns the coordinates and the fraction of variance each component explains. Signs are fixed so
    results are stable between runs.
    """
    matrix = torch.tensor(vectors, dtype=torch.float64)
    centred = matrix - matrix.mean(dim=0, keepdim=True)
    _, singular_values, components = torch.linalg.svd(centred, full_matrices=False)

    count = min(2, components.shape[0])
    axes = components[:count]
    for index in range(count):
        if axes[index][axes[index].abs().argmax()] < 0:
            axes[index] = -axes[index]

    coordinates = centred @ axes.T
    if count < 2:
        coordinates = torch.cat([coordinates, torch.zeros(coordinates.shape[0], 2 - count, dtype=torch.float64)], dim=1)

    variance = singular_values.pow(2)
    total = float(variance.sum())
    explained = [float(variance[index] / total) if total > 0 and index < count else 0.0 for index in range(2)]
    return [[round(float(x), 4), round(float(y), 4)] for x, y in coordinates], [round(value, 4) for value in explained]
