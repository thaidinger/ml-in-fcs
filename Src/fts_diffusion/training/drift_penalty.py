"""Torch utilities for the predictable-drift proxy penalty.

These functions compute the same fixed-feature moment as the evaluation
diagnostic on ordered return sequences. They are utility-only in this checkout:
the training pipeline source is unavailable, so the penalty is not integrated
into model optimization. Do not apply this across randomly shuffled batches and
label it as the true Alex statistic. If used on locally ordered reconstructed
segments, log it separately as a within-segment proxy such as
``drift_proxy_loss``.
"""

from __future__ import annotations

import torch


def _as_sequence_batch(returns: torch.Tensor) -> torch.Tensor:
    values = returns.float()
    if values.ndim == 1:
        values = values.unsqueeze(0)
    elif values.ndim == 2:
        pass
    elif values.ndim == 3 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    else:
        raise ValueError("returns must have shape (T,), (B, T), or (B, T, 1).")
    return values


def torch_drift_moment(
    returns: torch.Tensor,
    train_mean: float | torch.Tensor | None = None,
    train_std: float | torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return mean r_{t+1} omega_t for ordered return sequences."""
    values = _as_sequence_batch(returns)
    if values.shape[1] < 3:
        return torch.full((5,), float("nan"), dtype=values.dtype, device=values.device)

    if train_mean is None:
        mean = values.mean()
    else:
        mean = torch.as_tensor(train_mean, dtype=values.dtype, device=values.device)
    if train_std is None:
        std = values.std(unbiased=False)
    else:
        std = torch.as_tensor(train_std, dtype=values.dtype, device=values.device)

    z = (values - mean) / torch.clamp(torch.abs(std), min=eps)
    r_next = z[:, 2:]
    r_t = z[:, 1:-1]
    r_t_minus_1 = z[:, :-2]
    omega = torch.stack(
        [
            torch.ones_like(r_t),
            r_t,
            r_t_minus_1,
            torch.abs(r_t),
            torch.abs(r_t_minus_1),
        ],
        dim=-1,
    )
    return (r_next.unsqueeze(-1) * omega).reshape(-1, 5).mean(dim=0)


def torch_drift_penalty(
    returns: torch.Tensor,
    train_mean: float | torch.Tensor | None = None,
    train_std: float | torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return sum(moment ** 2), a differentiable proxy penalty."""
    moment = torch_drift_moment(returns, train_mean=train_mean, train_std=train_std, eps=eps)
    return torch.sum(moment**2)

