from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fts_diffusion.training.drift_penalty import torch_drift_moment, torch_drift_penalty


def test_torch_drift_penalty_is_sum_squared_moment() -> None:
    returns = torch.tensor([0.1, -0.2, 0.05, 0.3, -0.1], dtype=torch.float32)
    moment = torch_drift_moment(returns, train_mean=0.0, train_std=1.0)
    penalty = torch_drift_penalty(returns, train_mean=0.0, train_std=1.0)

    torch.testing.assert_close(penalty, torch.sum(moment**2))

