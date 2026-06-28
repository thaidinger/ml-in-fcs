from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from fts_diffusion.evaluation.drift import (
    block_shuffle_returns,
    drift_delta,
    drift_moment,
    make_omega,
    rolling_drift_report,
)


def _ar1(seed: int, n: int, phi: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=n)
    values = np.empty(n, dtype=float)
    values[0] = noise[0]
    for idx in range(1, n):
        values[idx] = phi * values[idx - 1] + noise[idx]
    return values


def test_iid_has_smaller_drift_than_ar1() -> None:
    rng = np.random.default_rng(123)
    iid = rng.normal(size=6000)
    ar = _ar1(seed=123, n=6000, phi=0.5)

    iid_delta = drift_delta(iid)
    ar_delta = drift_delta(ar)

    assert iid_delta < 0.08
    assert ar_delta > 0.25
    assert ar_delta > iid_delta * 4


def test_closed_form_delta_equals_norm_of_moment() -> None:
    returns = np.array([0.1, -0.2, 0.3, 0.05, -0.1, 0.2])
    omega = make_omega(returns, standardize=False)
    manual_moment = np.mean(returns[2:, None] * omega, axis=0)

    np.testing.assert_allclose(drift_moment(returns, standardize=False), manual_moment)
    assert drift_delta(returns, standardize=False) == np.linalg.norm(manual_moment)


def test_block_shuffle_preserves_length_and_values() -> None:
    returns = np.arange(23, dtype=float)
    shuffled = block_shuffle_returns(returns, block_size=5, seed=7)

    assert len(shuffled) == len(returns)
    np.testing.assert_array_equal(np.sort(shuffled), returns)


def test_rolling_report_has_window_for_short_valid_series() -> None:
    returns = np.linspace(-1.0, 1.0, 80)
    report = rolling_drift_report(returns, window_size=252, step_size=21, min_window=64)

    assert report["n_windows"] == 1
    assert np.isfinite(report["pooled_delta"])
    assert np.isfinite(report["rolling_delta_mean"])


def test_cli_help_runs() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/evaluate_predictable_drift.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--real-csv" in result.stdout
    assert "--synthetic-input-type" in result.stdout
