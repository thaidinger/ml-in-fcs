#!/usr/bin/env python
"""Controlled predictable-drift advantage experiment.

The experiment isolates a failure mode that marginal magnitude and volatility
diagnostics cannot see.  For each simulated heavy-tailed GARCH path, it keeps
the evaluation-period absolute returns fixed and replaces only the signs:

* iid signs: no intentional sign predictability.
* Markov signs: persistent signs, hence predictable conditional mean.

Both families have the same magnitude path, so magnitude quantiles and
absolute-return autocorrelation are unchanged.  The predictable-drift statistic
should increase with sign persistence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COMPONENT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(COMPONENT_ROOT / "src"))

from fts_diffusion.evaluation.drift import drift_delta  # noqa: E402
from fts_diffusion.evaluation.stylized import stylized_report  # noqa: E402


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
    if not 0.0 <= persistence <= 1.0:
        raise ValueError("persistence must be in [0, 1].")
    rng = np.random.default_rng(seed)
    signs = np.empty(n_obs, dtype=float)
    signs[0] = -1.0 if rng.random() < 0.5 else 1.0
    for idx in range(1, n_obs):
        signs[idx] = signs[idx - 1] if rng.random() < persistence else -signs[idx - 1]
    return signs


def signed_stylized_vector(returns: np.ndarray) -> np.ndarray:
    report = stylized_report(returns, max_lag=20)
    keys = [
        "std",
        "skew",
        "excess_kurtosis",
        "q01",
        "q05",
        "q50",
        "q95",
        "q99",
        "acf_abs_r_lag1",
        "acf_abs_r_l1_mean_abs",
    ]
    return np.asarray([report[key] for key in keys], dtype=float)


def scaled_distance(x: np.ndarray, reference: np.ndarray) -> float:
    scale = np.maximum(np.abs(reference), 0.05)
    return float(np.sqrt(np.mean(((x - reference) / scale) ** 2)))


def run_one_seed(seed: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    real = simulate_garch(
        seed=args.base_seed + seed,
        n_obs=args.n_train + args.n_eval,
        omega=args.omega,
        alpha=args.alpha,
        beta=args.beta,
        df=args.df,
    )
    train = real[: args.n_train]
    eval_returns = real[args.n_train :]
    train_mean = float(np.mean(train))
    train_std = float(np.std(train, ddof=0))
    magnitudes = np.abs(eval_returns)
    reference_vector = signed_stylized_vector(eval_returns)

    rows: list[dict[str, Any]] = []

    def append_row(sample_id: str, family: str, sample: np.ndarray, persistence: float | None) -> None:
        report = stylized_report(sample, max_lag=20)
        rows.append(
            {
                "seed": seed,
                "sample_id": sample_id,
                "family": family,
                "persistence": persistence,
                "n_obs": int(sample.size),
                "pooled_delta": drift_delta(sample, train_mean=train_mean, train_std=train_std),
                "acf_r_lag1": report["acf_r_lag1"],
                "acf_abs_r_lag1": report["acf_abs_r_lag1"],
                "signed_stylized_distance": scaled_distance(
                    signed_stylized_vector(sample),
                    reference_vector,
                ),
                "magnitude_path_distance": 0.0,
            }
        )

    append_row("real_eval", "real", eval_returns, None)
    iid_sample = magnitudes * iid_signs(args.base_seed + 10_000 + seed, args.n_eval)
    append_row("iid_signs", "iid_signs", iid_sample, None)

    for persistence in args.persistence:
        sample = magnitudes * markov_signs(
            args.base_seed + 20_000 + seed + int(round(persistence * 1000)),
            args.n_eval,
            persistence,
        )
        append_row(f"markov_signs_p{persistence:.2f}", "markov_signs", sample, persistence)

    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["family", "sample_id", "persistence"], dropna=False)
    summary = grouped.agg(
        n_seeds=("seed", "nunique"),
        pooled_delta_mean=("pooled_delta", "mean"),
        pooled_delta_std=("pooled_delta", "std"),
        acf_r_lag1_mean=("acf_r_lag1", "mean"),
        acf_abs_r_lag1_mean=("acf_abs_r_lag1", "mean"),
        signed_stylized_distance_mean=("signed_stylized_distance", "mean"),
        signed_stylized_distance_std=("signed_stylized_distance", "std"),
        magnitude_path_distance_mean=("magnitude_path_distance", "mean"),
    )
    return summary.reset_index()


def paired_summary(rows: pd.DataFrame, hidden_id: str) -> dict[str, Any]:
    iid = rows[rows["sample_id"] == "iid_signs"].set_index("seed")
    hidden = rows[rows["sample_id"] == hidden_id].set_index("seed")
    joined = hidden[["pooled_delta", "signed_stylized_distance", "acf_r_lag1"]].join(
        iid[["pooled_delta", "signed_stylized_distance", "acf_r_lag1"]],
        lsuffix="_hidden",
        rsuffix="_iid",
        how="inner",
    )
    delta_ratio = joined["pooled_delta_hidden"] / joined["pooled_delta_iid"]
    return {
        "hidden_sample_id": hidden_id,
        "n_pairs": int(len(joined)),
        "hidden_delta_mean": float(joined["pooled_delta_hidden"].mean()),
        "iid_delta_mean": float(joined["pooled_delta_iid"].mean()),
        "delta_ratio_mean": float(delta_ratio.mean()),
        "delta_ratio_median": float(delta_ratio.median()),
        "hidden_delta_gt_iid_count": int((joined["pooled_delta_hidden"] > joined["pooled_delta_iid"]).sum()),
        "hidden_signed_distance_mean": float(joined["signed_stylized_distance_hidden"].mean()),
        "iid_signed_distance_mean": float(joined["signed_stylized_distance_iid"].mean()),
        "hidden_acf_r_lag1_mean": float(joined["acf_r_lag1_hidden"].mean()),
        "iid_acf_r_lag1_mean": float(joined["acf_r_lag1_iid"].mean()),
    }


def write_report(output_dir: Path, summary: pd.DataFrame, paired: dict[str, Any]) -> None:
    table = summary[
        [
            "sample_id",
            "n_seeds",
            "pooled_delta_mean",
            "pooled_delta_std",
            "acf_r_lag1_mean",
            "acf_abs_r_lag1_mean",
            "signed_stylized_distance_mean",
            "magnitude_path_distance_mean",
        ]
    ].copy()
    lines = [
        "# Sign-Drift Sweep",
        "",
        "Magnitude paths are fixed within each seed; only signs change.",
        "",
        markdown_table(table),
        "",
        "## Paired p=0.85 comparison",
        "",
        "```json",
        json.dumps(paired, indent=2, sort_keys=True),
        "```",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    formatted_rows: list[list[str]] = []
    for row in df.itertuples(index=False, name=None):
        formatted: list[str] = []
        for value in row:
            if isinstance(value, float):
                formatted.append(f"{value:.4f}")
            else:
                formatted.append(str(value))
        formatted_rows.append(formatted)

    headers = [str(column) for column in df.columns]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in formatted_rows))
        if formatted_rows
        else len(headers[idx])
        for idx in range(len(headers))
    ]
    header = "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in formatted_rows
    ]
    return "\n".join([header, sep, *body])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=COMPONENT_ROOT / "outputs/sign_drift_sweep")
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--omega", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--beta", type=float, default=0.90)
    parser.add_argument("--df", type=int, default=5)
    parser.add_argument(
        "--persistence",
        type=float,
        nargs="+",
        default=[0.55, 0.65, 0.75, 0.85, 0.90],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.n_seeds < 1:
        raise ValueError("n-seeds must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for seed in range(args.n_seeds):
        all_rows.extend(run_one_seed(seed, args))

    rows = pd.DataFrame(all_rows)
    summary = summarize(rows)
    paired = paired_summary(rows, hidden_id="markov_signs_p0.85")

    rows.to_csv(args.output_dir / "sign_drift_sweep_raw.csv", index=False)
    summary.to_csv(args.output_dir / "sign_drift_sweep_summary.csv", index=False)
    (args.output_dir / "paired_summary.json").write_text(
        json.dumps(paired, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    metadata = {
        "n_seeds": args.n_seeds,
        "base_seed": args.base_seed,
        "n_train": args.n_train,
        "n_eval": args.n_eval,
        "garch": {
            "omega": args.omega,
            "alpha": args.alpha,
            "beta": args.beta,
            "df": args.df,
        },
        "persistence": args.persistence,
        "interpretation": (
            "Magnitude paths are identical within seed; increasing Markov sign "
            "persistence creates predictable drift without changing absolute-return "
            "volatility diagnostics."
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_report(args.output_dir, summary, paired)
    print(f"Wrote sign-drift sweep to {args.output_dir}")
    print(json.dumps(paired, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
