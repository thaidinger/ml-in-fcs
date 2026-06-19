#!/usr/bin/env python
"""NeurIPS-facing end-to-end diagnostics.

This script uses only artifacts available in the repository and the lightweight
drift metric implementation.  It produces three evidence blocks:

1. Real asset drift/null calibration from raw SP500, GOOG, and ZC=F price CSVs.
2. A controlled hidden-drift predictive-alpha experiment.
3. A compact summary of the stored end-to-end TATR protocol runs.

The goal is not to search for convenient numbers.  It is to produce paper-ready
tables that connect the diagnostic to raw data, to an exploitable failure mode,
and to the existing FTS-Diffusion replication outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPONENT_ROOT.parent

sys.path.insert(0, str(COMPONENT_ROOT / "src"))

from fts_diffusion.evaluation.drift import drift_delta, drift_report, null_drift_reports  # noqa: E402
from fts_diffusion.evaluation.stylized import stylized_report  # noqa: E402


ASSET_FILES = {
    "sp500": REPO_ROOT / "fts-diffusion-ref/data/sp500_timeseries.csv",
    "goog": REPO_ROOT / "fts-diffusion-ref/data/goog_timeseries.csv",
    "zcf": REPO_ROOT / "fts-diffusion-ref/data/zcf_timeseries.csv",
}


def load_price_returns(path: Path) -> np.ndarray:
    values: list[float] = []
    for raw in path.read_text(encoding="utf-8").splitlines()[1:]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            values.append(float(raw.split(",")[0]))
        except ValueError:
            continue
    prices = np.asarray(values, dtype=float)
    if prices.size < 3:
        raise ValueError(f"Not enough prices in {path}")
    return np.diff(prices) / prices[:-1]


def real_asset_benchmark(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    for asset, path in ASSET_FILES.items():
        returns = load_price_returns(path)
        split = int(math.floor(args.train_fraction * returns.size))
        train = returns[:split]
        test = returns[split:]
        train_mean = float(np.mean(train))
        train_std = float(np.std(train, ddof=0))
        report = drift_report(test, train_mean=train_mean, train_std=train_std)
        stylized = stylized_report(test, max_lag=50)
        null = null_drift_reports(
            test,
            n_reps=args.null_reps,
            block_size=args.block_size,
            seed=args.seed,
            train_mean=train_mean,
            train_std=train_std,
        )
        null_delta = np.asarray([row["delta"] for row in null], dtype=float)
        rows.append(
            {
                "asset": asset,
                "n_total": int(returns.size),
                "n_train": int(train.size),
                "n_test": int(test.size),
                "real_delta": report["delta"],
                "null_q05": float(np.quantile(null_delta, 0.05)),
                "null_q50": float(np.quantile(null_delta, 0.50)),
                "null_q95": float(np.quantile(null_delta, 0.95)),
                "inside_null_90": bool(
                    np.quantile(null_delta, 0.05) <= report["delta"] <= np.quantile(null_delta, 0.95)
                ),
                "acf_r_lag1": stylized["acf_r_lag1"],
                "acf_abs_r_lag1": stylized["acf_abs_r_lag1"],
                "excess_kurtosis": stylized["excess_kurtosis"],
            }
        )
        for idx, null_report in enumerate(null):
            null_rows.append(
                {
                    "asset": asset,
                    "rep": idx,
                    "delta": null_report["delta"],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(null_rows)


def simulate_garch(
    seed: int,
    n_obs: int,
    omega: float = 0.02,
    alpha: float = 0.08,
    beta: float = 0.90,
    df: int = 5,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    innovations = rng.standard_t(df, size=n_obs) / np.sqrt(df / (df - 2))
    returns = np.empty(n_obs, dtype=float)
    sigma2 = np.empty(n_obs, dtype=float)
    sigma2[0] = omega / (1.0 - alpha - beta)
    returns[0] = np.sqrt(sigma2[0]) * innovations[0]
    for idx in range(1, n_obs):
        sigma2[idx] = omega + alpha * returns[idx - 1] ** 2 + beta * sigma2[idx - 1]
        returns[idx] = np.sqrt(sigma2[idx]) * innovations[idx]
    return returns


def iid_signs(seed: int, n_obs: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n_obs) < 0.5, -1.0, 1.0)


def markov_signs(seed: int, n_obs: int, persistence: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    signs = np.empty(n_obs, dtype=float)
    signs[0] = -1.0 if rng.random() < 0.5 else 1.0
    for idx in range(1, n_obs):
        signs[idx] = signs[idx - 1] if rng.random() < persistence else -signs[idx - 1]
    return signs


def omega_targets(returns: np.ndarray, train_mean: float, train_std: float) -> tuple[np.ndarray, np.ndarray]:
    denom = train_std if abs(train_std) > 1e-8 else 1e-8
    z = (returns - train_mean) / denom
    x = np.column_stack(
        [
            np.ones(z.size - 2),
            z[1:-1],
            z[:-2],
            np.abs(z[1:-1]),
            np.abs(z[:-2]),
        ]
    )
    y = z[2:]
    return x, y


def fit_predictive_alpha(
    returns: np.ndarray,
    train_mean: float,
    train_std: float,
    ridge: float = 1e-3,
) -> dict[str, float]:
    x, y = omega_targets(returns, train_mean=train_mean, train_std=train_std)
    split = y.size // 2
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]
    coef = np.linalg.solve(x_train.T @ x_train + ridge * np.eye(x_train.shape[1]), x_train.T @ y_train)
    pred = x_test @ coef
    pnl = np.sign(pred) * y_test
    if np.std(pred) > 1e-12 and np.std(y_test) > 1e-12:
        corr = float(np.corrcoef(pred, y_test)[0, 1])
    else:
        corr = float("nan")
    return {
        "oos_corr": corr,
        "directional_accuracy": float(np.mean(np.sign(pred) == np.sign(y_test))),
        "strategy_edge": float(np.mean(pnl)),
        "strategy_sharpe": float(np.sqrt(252.0) * np.mean(pnl) / (np.std(pnl) + 1e-12)),
    }


def predictive_alpha_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows: list[dict[str, Any]] = []
    for seed in range(args.n_alpha_seeds):
        real = simulate_garch(args.seed + seed, args.n_train_alpha + args.n_eval_alpha)
        train = real[: args.n_train_alpha]
        eval_returns = real[args.n_train_alpha :]
        train_mean = float(np.mean(train))
        train_std = float(np.std(train, ddof=0))
        magnitudes = np.abs(eval_returns)
        samples: list[tuple[str, str, float | None, np.ndarray]] = [
            ("real_eval", "real", None, eval_returns),
            ("iid_signs", "iid_signs", None, magnitudes * iid_signs(args.seed + 10_000 + seed, args.n_eval_alpha)),
        ]
        for persistence in args.persistence:
            samples.append(
                (
                    f"markov_signs_p{persistence:.2f}",
                    "markov_signs",
                    persistence,
                    magnitudes
                    * markov_signs(args.seed + 20_000 + seed + int(round(persistence * 1000)), args.n_eval_alpha, persistence),
                )
            )
        for sample_id, family, persistence, sample in samples:
            alpha = fit_predictive_alpha(sample, train_mean=train_mean, train_std=train_std)
            stylized = stylized_report(sample, max_lag=20)
            raw_rows.append(
                {
                    "seed": seed,
                    "sample_id": sample_id,
                    "family": family,
                    "persistence": persistence,
                    "pooled_delta": drift_delta(sample, train_mean=train_mean, train_std=train_std),
                    "acf_r_lag1": stylized["acf_r_lag1"],
                    "acf_abs_r_lag1": stylized["acf_abs_r_lag1"],
                    **alpha,
                }
            )
    raw = pd.DataFrame(raw_rows)
    summary = (
        raw.groupby(["family", "sample_id", "persistence"], dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            pooled_delta_mean=("pooled_delta", "mean"),
            pooled_delta_std=("pooled_delta", "std"),
            oos_corr_mean=("oos_corr", "mean"),
            direction_acc_mean=("directional_accuracy", "mean"),
            strategy_edge_mean=("strategy_edge", "mean"),
            strategy_sharpe_mean=("strategy_sharpe", "mean"),
            strategy_sharpe_std=("strategy_sharpe", "std"),
            acf_r_lag1_mean=("acf_r_lag1", "mean"),
            acf_abs_r_lag1_mean=("acf_abs_r_lag1", "mean"),
        )
        .reset_index()
    )
    return raw, summary


def protocol_evidence() -> pd.DataFrame:
    tatr_path = REPO_ROOT / "reports/generated_outputs/01_sp500_downstream_replication/11_long_replication_batch/long_tatr_protocol_summary.csv"
    price_path = REPO_ROOT / "reports/generated_outputs/01_sp500_downstream_replication/11_long_replication_batch/long_synthetic_price_summary.csv"
    tatr = pd.read_csv(tatr_path)
    prices = pd.read_csv(price_path)
    continuous_price = prices[prices["protocol"].eq("continuous")].iloc[0].to_dict()
    independent_price = prices[prices["protocol"].eq("independent_blocks")].iloc[0].to_dict()
    rows: list[dict[str, Any]] = []
    for row in tatr.to_dict(orient="records"):
        protocol = row["protocol"]
        if protocol.startswith("continuous"):
            price = continuous_price
        else:
            price = independent_price
        rows.append(
            {
                "stage": row["stage"],
                "protocol": protocol,
                "ahead": row["ahead"],
                "seeds": row["seeds"],
                "best_pct_mean": row["best_pct_mean"],
                "final_pct_mean": row["final_pct_mean"],
                "synthetic_first_mean": price["first_mean"],
                "synthetic_last_mean": price["last_mean"],
                "synthetic_price_mean": price["price_mean"],
                "synthetic_price_std": price["std_mean"],
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    headers = [str(column) for column in df.columns]
    rows: list[list[str]] = []
    for row in df.itertuples(index=False, name=None):
        formatted: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                formatted.append(format(float(value), floatfmt))
            else:
                formatted.append(str(value))
        rows.append(formatted)
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]
    head = "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def write_report(
    output_dir: Path,
    real_summary: pd.DataFrame,
    alpha_summary: pd.DataFrame,
    protocol_summary: pd.DataFrame,
    raw_alpha: pd.DataFrame,
) -> None:
    alpha_compact = alpha_summary[
        [
            "sample_id",
            "n_seeds",
            "pooled_delta_mean",
            "oos_corr_mean",
            "direction_acc_mean",
            "strategy_sharpe_mean",
            "acf_r_lag1_mean",
        ]
    ].copy()
    real_compact = real_summary[
        [
            "asset",
            "n_test",
            "real_delta",
            "null_q05",
            "null_q50",
            "null_q95",
            "inside_null_90",
            "acf_abs_r_lag1",
            "excess_kurtosis",
        ]
    ].copy()
    protocol_compact = protocol_summary[
        [
            "stage",
            "protocol",
            "ahead",
            "best_pct_mean",
            "final_pct_mean",
            "synthetic_last_mean",
            "synthetic_price_std",
        ]
    ].copy()
    corr = float(raw_alpha[["pooled_delta", "strategy_sharpe"]].corr().iloc[0, 1])
    lines = [
        "# NeurIPS End-to-End Diagnostics",
        "",
        "## Real Asset Drift Null Calibration",
        "",
        markdown_table(real_compact),
        "",
        "## Hidden-Drift Predictive Alpha",
        "",
        markdown_table(alpha_compact),
        "",
        f"Across all controlled samples, the correlation between pooled delta and out-of-sample strategy Sharpe is `{corr:.4f}`.",
        "",
        "## Stored End-to-End TATR Protocol Evidence",
        "",
        markdown_table(protocol_compact),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports/generated_outputs/06_neurips_end_to_end_diagnostics")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--null-reps", type=int, default=500)
    parser.add_argument("--block-size", type=int, default=21)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--n-alpha-seeds", type=int, default=100)
    parser.add_argument("--n-train-alpha", type=int, default=4000)
    parser.add_argument("--n-eval-alpha", type=int, default=1000)
    parser.add_argument("--persistence", type=float, nargs="+", default=[0.65, 0.75, 0.85, 0.90])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    real_summary, real_null = real_asset_benchmark(args)
    alpha_raw, alpha_summary = predictive_alpha_experiment(args)
    protocol_summary = protocol_evidence()

    real_summary.to_csv(args.output_dir / "real_asset_drift_summary.csv", index=False)
    real_null.to_csv(args.output_dir / "real_asset_null_draws.csv", index=False)
    alpha_raw.to_csv(args.output_dir / "predictive_alpha_raw.csv", index=False)
    alpha_summary.to_csv(args.output_dir / "predictive_alpha_summary.csv", index=False)
    protocol_summary.to_csv(args.output_dir / "protocol_evidence_summary.csv", index=False)

    metadata = {
        "seed": args.seed,
        "null_reps": args.null_reps,
        "block_size": args.block_size,
        "train_fraction": args.train_fraction,
        "n_alpha_seeds": args.n_alpha_seeds,
        "n_train_alpha": args.n_train_alpha,
        "n_eval_alpha": args.n_eval_alpha,
        "persistence": args.persistence,
        "python": sys.executable,
        "notes": [
            "Real asset benchmark starts from raw price CSVs.",
            "Predictive-alpha experiment keeps magnitude paths fixed and changes only sign dependence.",
            "Protocol evidence summarizes stored end-to-end TATR runs generated by the reference sampler/LSTM pipeline.",
        ],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.output_dir, real_summary, alpha_summary, protocol_summary, alpha_raw)

    p85 = alpha_summary[alpha_summary["sample_id"].eq("markov_signs_p0.85")].iloc[0]
    iid = alpha_summary[alpha_summary["sample_id"].eq("iid_signs")].iloc[0]
    result = {
        "output_dir": str(args.output_dir),
        "real_assets_inside_90pct_null": int(real_summary["inside_null_90"].sum()),
        "real_assets_total": int(len(real_summary)),
        "p85_delta_mean": float(p85["pooled_delta_mean"]),
        "iid_delta_mean": float(iid["pooled_delta_mean"]),
        "p85_sharpe_mean": float(p85["strategy_sharpe_mean"]),
        "iid_sharpe_mean": float(iid["strategy_sharpe_mean"]),
        "delta_sharpe_corr": float(alpha_raw[["pooled_delta", "strategy_sharpe"]].corr().iloc[0, 1]),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
