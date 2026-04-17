from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


def resample_1d(array: np.ndarray, target_length: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if len(array) == target_length:
        return array.copy()
    if len(array) == 1:
        return np.repeat(array, target_length).astype(np.float32)
    old_positions = np.linspace(0.0, 1.0, num=len(array), dtype=np.float32)
    new_positions = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    return np.interp(new_positions, old_positions, array).astype(np.float32)


def batch_resample_1d(sequences: Iterable[torch.Tensor], target_length: int) -> torch.Tensor:
    resized = []
    for sequence in sequences:
        resized.append(
            F.interpolate(
                sequence.view(1, 1, -1),
                size=target_length,
                mode="linear",
                align_corners=False,
            ).view(-1)
        )
    return torch.stack(resized, dim=0)


def z_normalize(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    std = float(array.std())
    if std < eps:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - float(array.mean())) / std).astype(np.float32)


def rms_scale(array: np.ndarray, eps: float = 1e-6) -> float:
    value = float(np.sqrt(np.mean(np.square(array))))
    return max(value, eps)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    device = prediction.device
    horizon = prediction.shape[1]
    mask = torch.arange(horizon, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    squared_error = (prediction - target) ** 2
    return squared_error.masked_select(mask).mean()

