#!/usr/bin/env python
"""Drift-incorporated synthetic augmentation selection benchmark.

The experiment asks whether the predictable-drift diagnostic helps when it is
used, not merely reported.  For each seed, we build a pool of candidate
synthetic augmentations that share the same evaluation-period magnitude path
but differ in sign dynamics.  A marginal stylized-fact selector sees very
similar volatility, tail, and magnitude-autocorrelation statistics across the
pool.  The drift-incorporated selector uses the same stylized score plus a
penalty when a candidate's drift statistic exceeds the real block-shuffle null
band.

Each selected candidate trains the same one-step ridge forecaster on a small
real initialization set plus the synthetic augmentation.  The forecaster is
then evaluated on held-out real returns.  This is a controlled analogue of a
TATR augmentation choice: artificial drift can look like useful signal in the
synthetic data, but it should not improve a real zero-alpha target.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COMPONENT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(COMPONENT_ROOT / "src"))

from fts_diffusion.evaluation.drift import drift_delta, null_drift_reports  # noqa: E402
from fts_diffusion.evaluation.stylized import stylized_report  # noqa: E402


STYLIZED_KEYS = (
    "std",
    "excess_kurtosis",
    "q01",
    "q05",
    "q50",
    "q95",
    "q99",
    "acf_abs_r_lag1",
    "acf_abs_r_l1_mean_abs",
)


def simulate_garch(
    seed: int,
    n_obs: int,
    omega: float = 0.02,
    alpha: float = 0.08,
    beta: float = 0.90,
    df: int = 5,
) -> np.ndarray:
    """Simulate a zero-mean heavy-tailed GARCH(1,1)-like return path."""
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


def stylized_vector(returns: np.ndarray) -> np.ndarray:
    report = stylized_report(returns, max_lag=20)
    return np.asarray([report[key] for key in STYLIZED_KEYS], dtype=float)


def scaled_distance(values: np.ndarray, reference: np.ndarray) -> float:
    scale = np.maximum(np.abs(reference), 0.05)
    return float(np.sqrt(np.mean(((values - reference) / scale) ** 2)))


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
    return x, z[2:]


def stack_predictor_rows(parts: list[np.ndarray], train_mean: float, train_std: float) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for part in parts:
        if part.size < 3:
            continue
        x_part, y_part = omega_targets(part, train_mean=train_mean, train_std=train_std)
        xs.append(x_part)
        ys.append(y_part)
    if not xs:
        raise ValueError("At least one training part must have length >= 3.")
    return np.vstack(xs), np.concatenate(ys)


def fit_and_score_predictor(
    train_parts: list[np.ndarray],
    test_returns: np.ndarray,
    train_mean: float,
    train_std: float,
    ridge: float,
) -> dict[str, float]:
    x_train, y_train = stack_predictor_rows(train_parts, train_mean=train_mean, train_std=train_std)
    coef = np.linalg.solve(x_train.T @ x_train + ridge * np.eye(x_train.shape[1]), x_train.T @ y_train)
    x_test, y_test = omega_targets(test_returns, train_mean=train_mean, train_std=train_std)
    pred = x_test @ coef
    mse = float(np.mean((pred - y_test) ** 2))
    normalized_mse = mse / float(np.var(y_test) + 1e-12)
    return {
        "normalized_mse": normalized_mse,
        "coef_norm": float(np.linalg.norm(coef)),
        "mean_abs_prediction": float(np.mean(np.abs(pred))),
    }


def make_candidate_specs(reps_per_family: int) -> list[tuple[str, float | None]]:
    specs: list[tuple[str, float | None]] = [("iid", None) for _ in range(reps_per_family)]
    for persistence in (0.55, 0.65, 0.75, 0.85, 0.90):
        specs.extend(("markov", persistence) for _ in range(reps_per_family))
    return specs


def candidate_pool_for_seed(
    seed: int,
    args: argparse.Namespace,
    magnitudes: np.ndarray,
    reference_stylized: np.ndarray,
    init_returns: np.ndarray,
    test_returns: np.ndarray,
    train_mean: float,
    train_std: float,
    null_q95: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(args.selector_seed + seed)
    specs = make_candidate_specs(args.reps_per_family)
    rng.shuffle(specs)
    rows: list[dict[str, Any]] = []
    for candidate_idx, (family, persistence) in enumerate(specs):
        if family == "iid":
            signs = iid_signs(args.candidate_seed + 100 * seed + candidate_idx, magnitudes.size)
            sample_id = "iid"
        else:
            assert persistence is not None
            signs = markov_signs(args.candidate_seed + 100 * seed + candidate_idx, magnitudes.size, persistence)
            sample_id = f"p{persistence:.2f}"
        sample = magnitudes * signs
        stylized_score = scaled_distance(stylized_vector(sample), reference_stylized)
        delta = drift_delta(sample, train_mean=train_mean, train_std=train_std)
        downstream = fit_and_score_predictor(
            [init_returns, sample],
            test_returns,
            train_mean=train_mean,
            train_std=train_std,
            ridge=args.ridge,
        )
        rows.append(
            {
                "candidate_idx": candidate_idx,
                "sample_id": sample_id,
                "family": family,
                "persistence": persistence,
                "stylized_score": stylized_score,
                "delta": delta,
                "delta_over_q95": delta / null_q95 if null_q95 > 0 else np.nan,
                "violates_real_null": bool(delta > null_q95),
                **downstream,
            }
        )
    return pd.DataFrame(rows)


def select_candidates(candidates: pd.DataFrame, null_q95: float, penalty_weight: float, rng: np.random.Generator) -> dict[str, pd.Series]:
    penalty = penalty_weight * np.maximum(0.0, candidates["delta"].to_numpy(dtype=float) / null_q95 - 1.0) ** 2
    penalized = candidates.assign(selection_score=candidates["stylized_score"].to_numpy(dtype=float) + penalty)
    random_idx = int(rng.integers(0, len(candidates)))
    return {
        "stylized_only": candidates.loc[candidates["stylized_score"].idxmin()],
        "drift_penalized": penalized.sort_values(["selection_score", "stylized_score", "delta"]).iloc[0],
        "oracle_test_mse": candidates.loc[candidates["normalized_mse"].idxmin()],
        "random": candidates.iloc[random_idx],
    }


def run_one_seed(seed: int, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    real = simulate_garch(
        seed=args.data_seed + seed,
        n_obs=args.n_train + args.n_val + args.n_test,
        omega=args.omega,
        alpha=args.alpha,
        beta=args.beta,
        df=args.df,
    )
    train = real[: args.n_train]
    val = real[args.n_train : args.n_train + args.n_val]
    test = real[args.n_train + args.n_val :]
    init = train[-args.n_init :]
    train_mean = float(np.mean(train))
    train_std = float(np.std(train, ddof=0))
    null = null_drift_reports(
        val,
        n_reps=args.null_reps,
        block_size=args.block_size,
        seed=args.null_seed + seed,
        train_mean=train_mean,
        train_std=train_std,
    )
    null_delta = np.asarray([row["delta"] for row in null], dtype=float)
    null_q95 = float(np.quantile(null_delta, 0.95))
    candidates = candidate_pool_for_seed(
        seed=seed,
        args=args,
        magnitudes=np.abs(val),
        reference_stylized=stylized_vector(val),
        init_returns=init,
        test_returns=test,
        train_mean=train_mean,
        train_std=train_std,
        null_q95=null_q95,
    )
    real_only = fit_and_score_predictor(
        [init],
        test,
        train_mean=train_mean,
        train_std=train_std,
        ridge=args.ridge,
    )
    rng = np.random.default_rng(args.selector_seed + 50_000 + seed)
    selected = select_candidates(candidates, null_q95=null_q95, penalty_weight=args.penalty_weight, rng=rng)

    selection_rows: list[dict[str, Any]] = []
    for selector, row in selected.items():
        selected_row = row.to_dict()
        selection_rows.append(
            {
                "seed": seed,
                "selector": selector,
                "real_null_q95": null_q95,
                "real_only_normalized_mse": real_only["normalized_mse"],
                "excess_normalized_mse_vs_real_only": selected_row["normalized_mse"] - real_only["normalized_mse"],
                **selected_row,
            }
        )
    candidate_rows = [{"seed": seed, "real_null_q95": null_q95, **row} for row in candidates.to_dict(orient="records")]
    return selection_rows, candidate_rows


def summarize_selection(selection: pd.DataFrame) -> pd.DataFrame:
    summary = (
        selection.groupby("selector")
        .agg(
            n_seeds=("seed", "nunique"),
            delta_mean=("delta", "mean"),
            delta_std=("delta", "std"),
            null_violation_rate=("violates_real_null", "mean"),
            high_persistence_rate=("persistence", lambda values: float(np.mean(np.asarray(values, dtype=float) >= 0.75))),
            stylized_score_mean=("stylized_score", "mean"),
            normalized_mse_mean=("normalized_mse", "mean"),
            normalized_mse_std=("normalized_mse", "std"),
            excess_mse_vs_real_only_mean=("excess_normalized_mse_vs_real_only", "mean"),
            coef_norm_mean=("coef_norm", "mean"),
            mean_abs_prediction=("mean_abs_prediction", "mean"),
        )
        .reset_index()
    )
    order = {"oracle_test_mse": 0, "drift_penalized": 1, "stylized_only": 2, "random": 3}
    summary["_order"] = summary["selector"].map(order)
    return summary.sort_values("_order").drop(columns="_order")


def paired_comparisons(selection: pd.DataFrame) -> pd.DataFrame:
    pivot = selection.pivot(index="seed", columns="selector", values="normalized_mse")
    rows: list[dict[str, Any]] = []
    for selector in ("drift_penalized", "oracle_test_mse", "random"):
        diff = pivot["stylized_only"] - pivot[selector]
        rows.append(
            {
                "comparison": f"stylized_only_minus_{selector}",
                "n_pairs": int(diff.notna().sum()),
                "win_rate": float(np.mean(diff > 0.0)),
                "mean_normalized_mse_reduction": float(np.mean(diff)),
                "median_normalized_mse_reduction": float(np.median(diff)),
                "q05_reduction": float(np.quantile(diff, 0.05)),
                "q95_reduction": float(np.quantile(diff, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def selected_counts(selection: pd.DataFrame) -> pd.DataFrame:
    counts = selection.groupby(["selector", "sample_id"]).size().reset_index(name="count")
    totals = counts.groupby("selector")["count"].transform("sum")
    counts["share"] = counts["count"] / totals
    return counts


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


def write_report(output_dir: Path, selection_summary: pd.DataFrame, paired: pd.DataFrame, counts: pd.DataFrame) -> None:
    compact_summary = selection_summary[
        [
            "selector",
            "n_seeds",
            "delta_mean",
            "null_violation_rate",
            "high_persistence_rate",
            "stylized_score_mean",
            "normalized_mse_mean",
            "excess_mse_vs_real_only_mean",
            "coef_norm_mean",
        ]
    ].copy()
    compact_counts = counts[counts["sample_id"].isin(["iid", "p0.75", "p0.85", "p0.90"])].copy()
    lines = [
        "# Drift-Incorporated Selection Benchmark",
        "",
        "Candidate augmentations share the same magnitude path within each seed; only sign dynamics differ.",
        "The drift-penalized selector uses the same marginal stylized score as the baseline plus a penalty above the real block-shuffle drift band.",
        "",
        "## Selector Summary",
        "",
        markdown_table(compact_summary),
        "",
        "## Paired Normalized-MSE Comparisons",
        "",
        markdown_table(paired),
        "",
        "## Selected Candidate Counts",
        "",
        markdown_table(compact_counts),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=COMPONENT_ROOT / "outputs/drift_incorporated_selection")
    parser.add_argument("--n-seeds", type=int, default=500)
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--n-init", type=int, default=252)
    parser.add_argument("--reps-per-family", type=int, default=8)
    parser.add_argument("--null-reps", type=int, default=200)
    parser.add_argument("--block-size", type=int, default=21)
    parser.add_argument("--penalty-weight", type=float, default=0.25)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--omega", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--beta", type=float, default=0.90)
    parser.add_argument("--df", type=int, default=5)
    parser.add_argument("--data-seed", type=int, default=5000)
    parser.add_argument("--null-seed", type=int, default=7000)
    parser.add_argument("--selector-seed", type=int, default=9000)
    parser.add_argument("--candidate-seed", type=int, default=100000)
    return parser


def main(argv: list[str] | None = None) -> int:
    started_at = time.perf_counter()
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selection_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for seed in range(args.n_seeds):
        selected, candidates = run_one_seed(seed, args)
        selection_rows.extend(selected)
        candidate_rows.extend(candidates)

    selection = pd.DataFrame(selection_rows)
    candidates = pd.DataFrame(candidate_rows)
    selection_summary = summarize_selection(selection)
    paired = paired_comparisons(selection)
    counts = selected_counts(selection)

    selection.to_csv(args.output_dir / "selection_raw.csv", index=False)
    candidates.to_csv(args.output_dir / "candidate_pool_raw.csv", index=False)
    selection_summary.to_csv(args.output_dir / "selection_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    counts.to_csv(args.output_dir / "selected_counts.csv", index=False)
    metadata = {
        "n_seeds": args.n_seeds,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,
        "n_init": args.n_init,
        "reps_per_family": args.reps_per_family,
        "null_reps": args.null_reps,
        "block_size": args.block_size,
        "penalty_weight": args.penalty_weight,
        "ridge": args.ridge,
        "python": Path(sys.executable).name,
        "elapsed_seconds": time.perf_counter() - started_at,
        "notes": [
            "The oracle_test_mse selector uses held-out test MSE and is included only as a reference upper bound.",
            "The stylized score excludes raw-return autocorrelation and the predictable-drift components.",
            "Candidate augmentations share the same magnitude path within seed; only signs differ.",
        ],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    write_report(args.output_dir, selection_summary, paired, counts)

    drift_row = selection_summary[selection_summary["selector"].eq("drift_penalized")].iloc[0]
    style_row = selection_summary[selection_summary["selector"].eq("stylized_only")].iloc[0]
    paired_row = paired[paired["comparison"].eq("stylized_only_minus_drift_penalized")].iloc[0]
    result = {
        "output_dir": str(args.output_dir),
        "n_seeds": args.n_seeds,
        "stylized_only_nmse": float(style_row["normalized_mse_mean"]),
        "drift_penalized_nmse": float(drift_row["normalized_mse_mean"]),
        "mean_nmse_reduction": float(paired_row["mean_normalized_mse_reduction"]),
        "win_rate": float(paired_row["win_rate"]),
        "stylized_only_high_persistence_rate": float(style_row["high_persistence_rate"]),
        "drift_penalized_high_persistence_rate": float(drift_row["high_persistence_rate"]),
        "stylized_only_null_violation_rate": float(style_row["null_violation_rate"]),
        "drift_penalized_null_violation_rate": float(drift_row["null_violation_rate"]),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
