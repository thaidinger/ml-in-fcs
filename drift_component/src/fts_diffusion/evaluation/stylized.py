"""Lightweight stylized-fact metrics for return series."""

from __future__ import annotations

from typing import Any

import numpy as np

from .drift import as_1d_float_array


def summary_stats(returns: Any) -> dict[str, float | int]:
    """Return basic distributional summary statistics."""
    values = as_1d_float_array(returns)
    if values.size == 0:
        return {
            "n_obs": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "skew": float("nan"),
            "excess_kurtosis": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "q01": float("nan"),
            "q05": float("nan"),
            "q50": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
        }

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    centered = values - mean
    if std > 0:
        z = centered / std
        skew = float(np.mean(z**3))
        excess_kurtosis = float(np.mean(z**4) - 3.0)
    else:
        skew = float("nan")
        excess_kurtosis = float("nan")

    quantiles = np.quantile(values, [0.01, 0.05, 0.50, 0.95, 0.99])
    return {
        "n_obs": int(values.size),
        "mean": mean,
        "std": std,
        "skew": skew,
        "excess_kurtosis": excess_kurtosis,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
    }


def acf(x: Any, max_lag: int = 50) -> np.ndarray:
    """Autocorrelation for lags 0..max_lag using NumPy only."""
    values = as_1d_float_array(x)
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative.")
    result = np.full(max_lag + 1, np.nan, dtype=float)
    if values.size == 0:
        return result

    centered = values - np.mean(values)
    denom = float(np.dot(centered, centered))
    if denom <= 0:
        result[0] = 1.0
        return result

    result[0] = 1.0
    usable_lag = min(max_lag, values.size - 1)
    for lag in range(1, usable_lag + 1):
        result[lag] = float(np.dot(centered[:-lag], centered[lag:]) / denom)
    return result


def stylized_report(returns: Any, max_lag: int = 50) -> dict[str, float | int]:
    """Return compact stylized-fact metrics."""
    values = as_1d_float_array(returns)
    stats = summary_stats(values)
    acf_r = acf(values, max_lag=max_lag)
    acf_abs = acf(np.abs(values), max_lag=max_lag)
    upper = min(max_lag, max(values.size - 1, 0))
    if upper >= 1:
        acf_r_mean_abs = float(np.nanmean(np.abs(acf_r[1 : upper + 1])))
        acf_abs_mean_abs = float(np.nanmean(np.abs(acf_abs[1 : upper + 1])))
    else:
        acf_r_mean_abs = float("nan")
        acf_abs_mean_abs = float("nan")

    return {
        **stats,
        "acf_r_lag1": float(acf_r[1]) if max_lag >= 1 and acf_r.size > 1 else float("nan"),
        "acf_abs_r_lag1": float(acf_abs[1]) if max_lag >= 1 and acf_abs.size > 1 else float("nan"),
        "acf_r_l1_mean_abs": acf_r_mean_abs,
        "acf_abs_r_l1_mean_abs": acf_abs_mean_abs,
    }


def ks_2samp_report(x: Any, y: Any) -> dict[str, float] | None:
    """Optional SciPy-backed KS two-sample report."""
    try:
        from scipy import stats
    except Exception:
        return None
    result = stats.ks_2samp(as_1d_float_array(x), as_1d_float_array(y))
    return {"ks_statistic": float(result.statistic), "ks_pvalue": float(result.pvalue)}


def anderson_ksamp_report(samples: list[Any]) -> dict[str, float] | None:
    """Optional SciPy-backed Anderson-Darling k-sample report."""
    try:
        from scipy import stats
    except Exception:
        return None
    result = stats.anderson_ksamp([as_1d_float_array(sample) for sample in samples])
    return {
        "anderson_statistic": float(result.statistic),
        "anderson_pvalue": float(result.pvalue),
    }

