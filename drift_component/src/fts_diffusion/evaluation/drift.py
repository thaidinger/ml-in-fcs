"""Predictable-drift diagnostics for return series.

The diagnostic uses the fixed feature map
omega_t = (1, r_t, r_{t-1}, |r_t|, |r_{t-1}|) and reports
||mean_t r_{t+1} omega_t||_2. By default, returns are standardized before
forming both r_{t+1} and omega_t. This is a finite-dimensional diagnostic,
not a proof of a martingale or no-arbitrage property.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DRIFT_COMPONENTS = (
    "mean_r_next",
    "mean_r_next_r_t",
    "mean_r_next_r_t_minus_1",
    "mean_r_next_abs_r_t",
    "mean_r_next_abs_r_t_minus_1",
)


def as_1d_float_array(x: Any) -> np.ndarray:
    """Return a finite 1-D float NumPy array, dropping NaN and inf values."""
    if isinstance(x, pd.DataFrame):
        numeric = x.select_dtypes(include=["number"])
        if numeric.shape[1] != 1:
            raise ValueError("DataFrame input must have exactly one numeric column.")
        values = numeric.iloc[:, 0].to_numpy(dtype=float)
    elif isinstance(x, (pd.Series, pd.Index)):
        values = x.to_numpy(dtype=float)
    else:
        values = np.asarray(x, dtype=float)

    values = np.ravel(values).astype(float, copy=False)
    return values[np.isfinite(values)]


def train_standardize(
    series: Any, train_mean: float | None, train_std: float | None, eps: float = 1e-8
) -> np.ndarray:
    """Standardize a return series with training-split statistics."""
    values = as_1d_float_array(series)
    mean = float(np.mean(values)) if train_mean is None else float(train_mean)
    std = float(np.std(values, ddof=0)) if train_std is None else float(train_std)
    denom = std if abs(std) > eps else eps
    return (values - mean) / denom


def make_omega(
    returns: Any,
    standardize: bool = True,
    train_mean: float | None = None,
    train_std: float | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Build rows of omega_t aligned with r_{t+1}.

    For a return array of length n, the result has shape (max(n - 2, 0), 5)
    and rows [1, r_t, r_{t-1}, |r_t|, |r_{t-1}|] for t = 1 .. n - 2.
    """
    values = (
        train_standardize(returns, train_mean, train_std, eps=eps)
        if standardize
        else as_1d_float_array(returns)
    )
    if values.size < 3:
        return np.empty((0, 5), dtype=float)

    r_t = values[1:-1]
    r_t_minus_1 = values[:-2]
    return np.column_stack(
        [
            np.ones_like(r_t),
            r_t,
            r_t_minus_1,
            np.abs(r_t),
            np.abs(r_t_minus_1),
        ]
    )


def drift_moment(
    returns: Any,
    standardize: bool = True,
    train_mean: float | None = None,
    train_std: float | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Return mean_t r_{t+1} omega_t as a length-5 vector."""
    values = (
        train_standardize(returns, train_mean, train_std, eps=eps)
        if standardize
        else as_1d_float_array(returns)
    )
    if values.size < 3:
        return np.full(5, np.nan, dtype=float)

    omega = make_omega(values, standardize=False)
    r_next = values[2:]
    return np.mean(r_next[:, None] * omega, axis=0)


def drift_delta(
    returns: Any,
    standardize: bool = True,
    train_mean: float | None = None,
    train_std: float | None = None,
    eps: float = 1e-8,
) -> float:
    """Return the closed-form empirical drift violation."""
    moment = drift_moment(
        returns,
        standardize=standardize,
        train_mean=train_mean,
        train_std=train_std,
        eps=eps,
    )
    return float(np.linalg.norm(moment)) if np.all(np.isfinite(moment)) else float("nan")


def _standardization_stats(
    returns: Any, standardize: bool, train_mean: float | None, train_std: float | None
) -> tuple[float | None, float | None]:
    values = as_1d_float_array(returns)
    if not standardize:
        return train_mean, train_std
    mean = float(np.mean(values)) if train_mean is None else float(train_mean)
    std = float(np.std(values, ddof=0)) if train_std is None else float(train_std)
    return mean, std


def drift_report(
    returns: Any,
    standardize: bool = True,
    train_mean: float | None = None,
    train_std: float | None = None,
    eps: float = 1e-8,
) -> dict[str, float | int | None]:
    """Return n, delta, component moments, and standardization metadata."""
    values = as_1d_float_array(returns)
    mean, std = _standardization_stats(values, standardize, train_mean, train_std)
    moment = drift_moment(
        values,
        standardize=standardize,
        train_mean=mean,
        train_std=std,
        eps=eps,
    )
    report: dict[str, float | int | None] = {
        "n_obs": int(values.size),
        "delta": float(np.linalg.norm(moment)) if np.all(np.isfinite(moment)) else float("nan"),
        "train_mean": mean,
        "train_std": std,
    }
    report.update({name: float(value) for name, value in zip(DRIFT_COMPONENTS, moment)})
    return report


def rolling_drift_report(
    returns: Any,
    window_size: int = 252,
    step_size: int = 21,
    min_window: int = 64,
    standardize: bool = True,
    train_mean: float | None = None,
    train_std: float | None = None,
    eps: float = 1e-8,
) -> dict[str, Any]:
    """Compute pooled and rolling drift reports.

    If the series is shorter than window_size but has at least min_window
    observations, one whole-series window is evaluated.
    """
    values = as_1d_float_array(returns)
    if window_size < 3:
        raise ValueError("window_size must be at least 3.")
    if step_size < 1:
        raise ValueError("step_size must be positive.")
    if min_window < 3:
        raise ValueError("min_window must be at least 3.")

    pooled = drift_report(
        values,
        standardize=standardize,
        train_mean=train_mean,
        train_std=train_std,
        eps=eps,
    )
    windows: list[dict[str, Any]] = []
    n = values.size
    if n >= window_size:
        starts = range(0, n - window_size + 1, step_size)
        slices = [(start, start + window_size) for start in starts]
    elif n >= min_window:
        slices = [(0, n)]
    else:
        slices = []

    for start, end in slices:
        report = drift_report(
            values[start:end],
            standardize=standardize,
            train_mean=train_mean,
            train_std=train_std,
            eps=eps,
        )
        report["start"] = int(start)
        report["end"] = int(end)
        windows.append(report)

    deltas = np.asarray([window["delta"] for window in windows], dtype=float)
    finite = deltas[np.isfinite(deltas)]
    return {
        "pooled_delta": float(pooled["delta"]),
        "rolling_delta_mean": float(np.mean(finite)) if finite.size else float("nan"),
        "rolling_delta_std": float(np.std(finite, ddof=0)) if finite.size else float("nan"),
        "rolling_delta_min": float(np.min(finite)) if finite.size else float("nan"),
        "rolling_delta_max": float(np.max(finite)) if finite.size else float("nan"),
        "n_windows": int(len(windows)),
        "windows": windows,
        "pooled_report": pooled,
    }


def block_shuffle_returns(returns: Any, block_size: int = 21, seed: int | None = None) -> np.ndarray:
    """Shuffle contiguous blocks while preserving within-block order.

    The final shorter remainder block is kept as a block and shuffled together
    with the full blocks, so output length and the multiset of values are
    preserved deterministically for a fixed seed.
    """
    values = as_1d_float_array(returns)
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    if values.size == 0:
        return values.copy()
    blocks = [values[start : start + block_size] for start in range(0, values.size, block_size)]
    order = np.random.default_rng(seed).permutation(len(blocks))
    return np.concatenate([blocks[index] for index in order])


def null_drift_reports(
    returns: Any,
    n_reps: int = 20,
    block_size: int = 21,
    seed: int = 0,
    standardize: bool = True,
    train_mean: float | None = None,
    train_std: float | None = None,
    eps: float = 1e-8,
) -> list[dict[str, float | int | None]]:
    """Return drift reports for block-shuffled null replicates."""
    if n_reps < 0:
        raise ValueError("n_reps must be non-negative.")
    rng = np.random.default_rng(seed)
    reports: list[dict[str, float | int | None]] = []
    for rep in range(n_reps):
        shuffled = block_shuffle_returns(returns, block_size=block_size, seed=int(rng.integers(0, 2**32 - 1)))
        report = drift_report(
            shuffled,
            standardize=standardize,
            train_mean=train_mean,
            train_std=train_std,
            eps=eps,
        )
        report["rep"] = rep
        reports.append(report)
    return reports

