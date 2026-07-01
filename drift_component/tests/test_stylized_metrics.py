from __future__ import annotations

import numpy as np

from fts_diffusion.evaluation.stylized import acf, stylized_report, summary_stats


def test_summary_stats_basic_values() -> None:
    values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    stats = summary_stats(values)

    assert stats["n_obs"] == 5
    assert stats["mean"] == 0.0
    assert stats["q50"] == 0.0
    assert stats["min"] == -2.0
    assert stats["max"] == 2.0


def test_acf_lag_zero_and_shape() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    result = acf(values, max_lag=3)

    assert result.shape == (4,)
    assert result[0] == 1.0
    assert np.isfinite(result[1])


def test_stylized_report_contains_expected_keys() -> None:
    rng = np.random.default_rng(321)
    report = stylized_report(rng.normal(size=200), max_lag=10)

    assert "acf_abs_r_lag1" in report
    assert "excess_kurtosis" in report
    assert "acf_r_l1_mean_abs" in report

