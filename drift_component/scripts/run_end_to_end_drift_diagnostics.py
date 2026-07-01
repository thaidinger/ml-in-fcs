#!/usr/bin/env python
"""End-to-end predictable-drift diagnostics.

This script uses only artifacts available in the repository and the lightweight
drift metric implementation.  It produces three evidence blocks:

1. Real asset drift/null calibration from raw SP500, GOOG, and ZC=F price CSVs.
2. A controlled hidden-drift predictive-alpha experiment.
3. A compact summary of the stored end-to-end TATR protocol runs.

The goal is not to search for convenient numbers.  It is to produce reviewable
tables that connect the diagnostic to raw data, to an exploitable failure mode,
and to the existing FTS-Diffusion replication outputs.
"""

from __future__ import annotations

import argparse
import os
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPONENT_ROOT.parent
REF_ROOT = REPO_ROOT / "fts-diffusion-ref"

sys.path.insert(0, str(COMPONENT_ROOT / "src"))
sys.path.insert(0, str(REF_ROOT))

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


def price_returns(prices: np.ndarray) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    if prices.size < 2:
        return np.asarray([], dtype=float)
    return np.diff(prices) / prices[:-1]


def set_reference_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pooled_drift_report_from_parts(
    return_parts: list[np.ndarray],
    train_mean: float,
    train_std: float,
) -> dict[str, float | int]:
    """Pool drift moments over ordered parts without adding boundary transitions."""
    from fts_diffusion.evaluation.drift import DRIFT_COMPONENTS, drift_moment

    moments: list[np.ndarray] = []
    weights: list[int] = []
    block_deltas: list[float] = []
    for part in return_parts:
        values = np.asarray(part, dtype=float)
        if values.size < 3:
            continue
        moment = drift_moment(values, train_mean=train_mean, train_std=train_std)
        if np.all(np.isfinite(moment)):
            moments.append(moment)
            weights.append(values.size - 2)
            block_deltas.append(float(np.linalg.norm(moment)))
    if not moments:
        row: dict[str, float | int] = {
            "n_obs": 0,
            "n_effective": 0,
            "delta": float("nan"),
            "block_delta_mean": float("nan"),
            "block_delta_std": float("nan"),
        }
        row.update({component: float("nan") for component in DRIFT_COMPONENTS})
        return row

    weights_array = np.asarray(weights, dtype=float)
    pooled_moment = np.average(np.vstack(moments), axis=0, weights=weights_array)
    row = {
        "n_obs": int(sum(len(part) for part in return_parts)),
        "n_effective": int(sum(weights)),
        "delta": float(np.linalg.norm(pooled_moment)),
        "block_delta_mean": float(np.mean(block_deltas)),
        "block_delta_std": float(np.std(block_deltas, ddof=0)),
    }
    row.update({component: float(value) for component, value in zip(DRIFT_COMPONENTS, pooled_moment)})
    return row


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


def fts_protocol_drift_audit(args: argparse.Namespace, real_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate ordered FTS-Diffusion rollout samples with the fixed drift statistic."""
    from experiments.utils_downstream import get_downstream_data, init_first_segment
    from models.load_models import load_ftsdiffusion
    from models.model_params import prm_params
    from models.utils_sampling import sampling_inputs
    import torch
    from torch.nn import functional as F

    sp500_returns = load_price_returns(ASSET_FILES["sp500"])
    split = int(math.floor(args.train_fraction * sp500_returns.size))
    train = sp500_returns[:split]
    train_mean = float(np.mean(train))
    train_std = float(np.std(train, ddof=0))
    sp500_row = real_summary[real_summary["asset"].eq("sp500")].iloc[0]
    null_q95 = float(sp500_row["null_q95"])

    previous_cwd = Path.cwd()
    os.chdir(REF_ROOT)
    try:
        downstream_ts, segments_test, labels_test, lengths_test = get_downstream_data()
        init_state, init_segment = init_first_segment(segments_test, labels_test, lengths_test)
        model = load_ftsdiffusion()
        _, _, patterns = sampling_inputs()
        l_min = prm_params["l_min"]
        torch.set_num_threads(max(1, min(args.fts_torch_threads, os.cpu_count() or 1)))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        patterns_tensor = torch.as_tensor(patterns, dtype=torch.float32, device=device)
        init_state_tensor = init_state.clone().detach().to(device).float()
        init_segment_array = np.asarray(init_segment, dtype=np.float64)

        def evolve_states(states: torch.Tensor) -> torch.Tensor:
            pred = model["evolution"](states.to(device))
            n_patterns = model["evolution"].n_patterns
            range_length = model["evolution"].range_length
            pattern = torch.argmax(F.softmax(pred[:, :n_patterns], dim=1), dim=1).float().unsqueeze(1)
            length = (
                torch.argmax(F.softmax(pred[:, n_patterns : n_patterns + range_length], dim=1), dim=1)
                .float()
                .unsqueeze(1)
                + float(l_min)
            )
            magnitude = pred[:, n_patterns + range_length :].float()
            return torch.cat((pattern, length, magnitude), dim=1)

        def generate_segments(states: torch.Tensor) -> list[np.ndarray]:
            states = states.to(device)
            pattern_idx = states[:, 0].long()
            lengths = states[:, 1].long().detach().cpu().numpy()
            magnitudes = states[:, 2].float()
            pattern_batch = patterns_tensor[pattern_idx]
            generated, _ = model["generation"].generate(pattern_batch, lengths)
            generated = generated * magnitudes.unsqueeze(1)
            generated_np = generated.detach().cpu().numpy()
            return [
                np.asarray(generated_np[row_idx, : int(length)], dtype=np.float64)
                for row_idx, length in enumerate(lengths)
            ]

        def generate_paths(length: int, n_paths: int, seed: int) -> list[np.ndarray]:
            set_reference_seed(seed)
            states = init_state_tensor.repeat(n_paths, 1)
            paths = [list(init_segment_array) for _ in range(n_paths)]
            active = np.ones(n_paths, dtype=bool)
            while np.any(active):
                active_idx = np.flatnonzero(active)
                next_states = evolve_states(states[active_idx])
                segments = generate_segments(next_states)
                for local_idx, path_idx in enumerate(active_idx):
                    segment = segments[local_idx]
                    if segment.size == 0:
                        active[path_idx] = False
                        continue
                    segment = segment - segment[0] + paths[path_idx][-1]
                    remaining = length - len(paths[path_idx])
                    if remaining > 0:
                        paths[path_idx].extend(segment[:remaining])
                    states[path_idx] = next_states[local_idx]
                    active[path_idx] = len(paths[path_idx]) < length
            return [np.asarray(path[:length], dtype=np.float64) for path in paths]

        def generate_continuous(length: int, seed: int) -> np.ndarray:
            return generate_paths(length=length, n_paths=1, seed=seed)[0]

        def generate_independent_blocks(n_blocks: int, seed: int) -> list[np.ndarray]:
            return generate_paths(length=args.fts_block_length, n_paths=n_blocks, seed=seed)

        max_length = args.fts_blocks * args.fts_block_length
        rows: list[dict[str, Any]] = []
        components: list[dict[str, Any]] = []
        for seed in args.fts_seeds:
            continuous_prices = generate_continuous(max_length, seed)
            independent_blocks = generate_independent_blocks(args.fts_blocks, seed + 10_000)
            samples = [
                ("continuous_rollout", [price_returns(continuous_prices)], continuous_prices),
                (
                    "independent_blocks",
                    [price_returns(block) for block in independent_blocks],
                    np.concatenate(independent_blocks),
                ),
            ]
            for protocol, return_parts, prices in samples:
                pooled = pooled_drift_report_from_parts(return_parts, train_mean=train_mean, train_std=train_std)
                returns_flat = np.concatenate([part for part in return_parts if part.size])
                stylized = stylized_report(returns_flat, max_lag=50)
                row = {
                    "seed": seed,
                    "protocol": protocol,
                    "n_blocks": args.fts_blocks,
                    "n_price_obs": int(prices.size),
                    "n_return_obs": int(pooled["n_obs"]),
                    "delta": pooled["delta"],
                    "delta_over_sp500_null_q95": float(pooled["delta"]) / null_q95,
                    "violates_sp500_null_q95": bool(float(pooled["delta"]) > null_q95),
                    "block_delta_mean": pooled["block_delta_mean"],
                    "block_delta_std": pooled["block_delta_std"],
                    "first_price": float(prices[0]),
                    "last_price": float(prices[-1]),
                    "mean_price": float(np.mean(prices)),
                    "std_price": float(np.std(prices, ddof=0)),
                    "acf_r_lag1": stylized["acf_r_lag1"],
                    "acf_abs_r_lag1": stylized["acf_abs_r_lag1"],
                    "excess_kurtosis": stylized["excess_kurtosis"],
                }
                rows.append(row)
                components.append(
                    {
                        "seed": seed,
                        "protocol": protocol,
                        **{
                            key: pooled[key]
                            for key in (
                                "mean_r_next",
                                "mean_r_next_r_t",
                                "mean_r_next_r_t_minus_1",
                                "mean_r_next_abs_r_t",
                                "mean_r_next_abs_r_t_minus_1",
                            )
                        },
                    }
                )
    finally:
        os.chdir(previous_cwd)

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("protocol")
        .agg(
            seeds=("seed", "nunique"),
            delta_mean=("delta", "mean"),
            delta_min=("delta", "min"),
            delta_max=("delta", "max"),
            delta_over_q95_mean=("delta_over_sp500_null_q95", "mean"),
            violation_rate=("violates_sp500_null_q95", "mean"),
            block_delta_mean=("block_delta_mean", "mean"),
            first_price_mean=("first_price", "mean"),
            last_price_mean=("last_price", "mean"),
            mean_price_mean=("mean_price", "mean"),
            std_price_mean=("std_price", "mean"),
            acf_r_lag1_mean=("acf_r_lag1", "mean"),
            acf_abs_r_lag1_mean=("acf_abs_r_lag1", "mean"),
            excess_kurtosis_mean=("excess_kurtosis", "mean"),
        )
        .reset_index()
    )
    summary["sp500_real_delta"] = float(sp500_row["real_delta"])
    summary["sp500_null_q95"] = null_q95
    component_summary = (
        pd.DataFrame(components)
        .groupby("protocol")
        .mean(numeric_only=True)
        .reset_index()
    )
    return raw, summary.merge(component_summary, on="protocol", how="left", suffixes=("", "_component"))


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
    fts_drift_summary: pd.DataFrame,
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
        "# End-to-End Drift Diagnostics",
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
    if not fts_drift_summary.empty:
        fts_compact = fts_drift_summary[
            [
                "protocol",
                "seeds",
                "delta_mean",
                "delta_min",
                "delta_max",
                "delta_over_q95_mean",
                "violation_rate",
                "last_price_mean",
                "std_price_mean",
                "acf_abs_r_lag1_mean",
            ]
        ].copy()
        lines.extend(
            [
                "## Ordered FTS-Diffusion Drift Audit",
                "",
                markdown_table(fts_compact),
                "",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "reports/generated_outputs/06_end_to_end_drift_diagnostics")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--null-reps", type=int, default=500)
    parser.add_argument("--block-size", type=int, default=21)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--n-alpha-seeds", type=int, default=100)
    parser.add_argument("--n-train-alpha", type=int, default=4000)
    parser.add_argument("--n-eval-alpha", type=int, default=1000)
    parser.add_argument("--persistence", type=float, nargs="+", default=[0.65, 0.75, 0.85, 0.90])
    parser.add_argument("--run-fts-drift-audit", action="store_true")
    parser.add_argument("--fts-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46, 47])
    parser.add_argument("--fts-blocks", type=int, default=100)
    parser.add_argument("--fts-block-length", type=int, default=252)
    parser.add_argument("--fts-torch-threads", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    started_at = time.perf_counter()
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    real_summary, real_null = real_asset_benchmark(args)
    alpha_raw, alpha_summary = predictive_alpha_experiment(args)
    protocol_summary = protocol_evidence()
    fts_drift_error = None
    if args.run_fts_drift_audit:
        try:
            fts_drift_raw, fts_drift_summary = fts_protocol_drift_audit(args, real_summary)
        except Exception as exc:
            fts_drift_error = repr(exc)
            fts_drift_raw = pd.DataFrame()
            fts_drift_summary = pd.DataFrame()
    else:
        fts_drift_raw = pd.DataFrame()
        fts_drift_summary = pd.DataFrame()

    real_summary.to_csv(args.output_dir / "real_asset_drift_summary.csv", index=False)
    real_null.to_csv(args.output_dir / "real_asset_null_draws.csv", index=False)
    alpha_raw.to_csv(args.output_dir / "predictive_alpha_raw.csv", index=False)
    alpha_summary.to_csv(args.output_dir / "predictive_alpha_summary.csv", index=False)
    protocol_summary.to_csv(args.output_dir / "protocol_evidence_summary.csv", index=False)
    if not fts_drift_raw.empty:
        fts_drift_raw.to_csv(args.output_dir / "fts_protocol_drift_raw.csv", index=False)
        fts_drift_summary.to_csv(args.output_dir / "fts_protocol_drift_summary.csv", index=False)

    metadata = {
        "seed": args.seed,
        "null_reps": args.null_reps,
        "block_size": args.block_size,
        "train_fraction": args.train_fraction,
        "n_alpha_seeds": args.n_alpha_seeds,
        "n_train_alpha": args.n_train_alpha,
        "n_eval_alpha": args.n_eval_alpha,
        "persistence": args.persistence,
        "fts_seeds": args.fts_seeds,
        "fts_blocks": args.fts_blocks,
        "fts_block_length": args.fts_block_length,
        "fts_torch_threads": args.fts_torch_threads,
        "run_fts_drift_audit": args.run_fts_drift_audit,
        "fts_drift_error": fts_drift_error,
        "python": Path(sys.executable).name,
        "elapsed_seconds": time.perf_counter() - started_at,
        "notes": [
            "Real asset benchmark starts from raw price CSVs.",
            "Predictive-alpha experiment keeps magnitude paths fixed and changes only sign dependence.",
            "Protocol evidence summarizes stored end-to-end TATR runs generated by the reference sampler/LSTM pipeline.",
            "If enabled, ordered FTS-Diffusion drift audit regenerates synthetic paths from stored checkpoints and excludes independent-block boundary transitions.",
        ],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.output_dir, real_summary, alpha_summary, protocol_summary, alpha_raw, fts_drift_summary)

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
        "fts_drift_summary": fts_drift_summary[
            ["protocol", "delta_mean", "delta_over_q95_mean", "violation_rate"]
        ].to_dict(orient="records")
        if not fts_drift_summary.empty
        else [],
        "fts_drift_error": fts_drift_error,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
