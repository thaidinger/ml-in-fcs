"""Evaluation utilities for FTS-Diffusion outputs."""

from .drift import (
    as_1d_float_array,
    block_shuffle_returns,
    drift_delta,
    drift_moment,
    drift_report,
    make_omega,
    null_drift_reports,
    rolling_drift_report,
    train_standardize,
)
from .stylized import acf, stylized_report, summary_stats

__all__ = [
    "acf",
    "as_1d_float_array",
    "block_shuffle_returns",
    "drift_delta",
    "drift_moment",
    "drift_report",
    "make_omega",
    "null_drift_reports",
    "rolling_drift_report",
    "stylized_report",
    "summary_stats",
    "train_standardize",
]

