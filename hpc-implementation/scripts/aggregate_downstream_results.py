#!/usr/bin/env python3
"""Aggregate run-level HPC downstream outputs and create plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("hpc-implementation/results/sp500_hpc_full"))
    return parser.parse_args()


def summarize(errors: np.ndarray) -> pd.DataFrame:
    rows = []
    for col in range(errors.shape[1]):
        values = np.asarray(errors[:, col], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            rows.append({"avg": np.nan, "min": np.nan, "max": np.nan})
            continue
        values = np.sort(values)
        percentile = int(np.ceil(len(values) * 0.025))
        if percentile == 0 or len(values) <= 2 * percentile:
            trimmed = values
            low = values[0]
            high = values[-1]
        else:
            trimmed = values[percentile:-percentile]
            low = values[percentile]
            high = values[-percentile]
        rows.append({"avg": float(np.mean(trimmed)), "min": float(low), "max": float(high)})
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, x_col: str, title: str, xlabel: str, path_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    x = summary[x_col].to_numpy()
    # Ensure numeric arrays and handle missing values in confidence bounds
    avg = pd.Series(summary["avg"]).astype(float)
    low = pd.Series(summary["min"]).astype(float)
    high = pd.Series(summary["max"]).astype(float)
    # Interpolate missing CI entries so the shaded band follows nearest data
    low = low.replace([np.inf, -np.inf], np.nan).interpolate(method="linear", limit_direction="both")
    high = high.replace([np.inf, -np.inf], np.nan).interpolate(method="linear", limit_direction="both")
    # If interpolation still leaves NaNs (all-NaN series), backfill as fallback
    low = low.fillna(method="bfill").fillna(method="ffill")
    high = high.fillna(method="bfill").fillna(method="ffill")

    ax.plot(x, avg.to_numpy(), marker="o", linewidth=1.8, markersize=3.8, color="#1f77b4", label="Average MAPE")
    # Hide CI where too few runs contributed
    min_runs_ci = 3
    if "n" in summary.columns:
        n = pd.Series(summary["n"]).astype(int)
        mask = n >= min_runs_ci
    else:
        mask = np.ones_like(x, dtype=bool)
    low_plot = np.where(mask, low.to_numpy(), np.nan)
    high_plot = np.where(mask, high.to_numpy(), np.nan)
    ax.fill_between(x, low_plot, high_plot, color="#1f77b4", alpha=0.18, label="95% CI")
    baseline = float(avg.iloc[0]) if not avg.isna().iloc[0] else float(np.nan)
    ax.axhline(baseline, color="0.45", linestyle="--", linewidth=1, label=f"Baseline (initial): {baseline:.3f}")
    # Annotate baseline value near the left of the axis
    try:
        ax.annotate(f"{baseline:.3f}", xy=(x[0], baseline), xytext=(10, 2), textcoords="offset points", color="0.25")
    except Exception:
        pass
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MAPE")
    # Add boxed labels for every 10th data point
    try:
        bbox_props = dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.3", linewidth=0.7)
        for idx in range(0, len(x), 10):
            yval = float(avg.iloc[idx])
            if np.isfinite(yval):
                ax.annotate(
                    f"{yval:.3f}",
                    xy=(x[idx], yval),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    bbox=bbox_props,
                    color="0.15",
                )
    except Exception:
        # If indexing fails or values missing, skip labels silently
        pass
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def aggregate_tatr(results_dir: Path) -> dict[str, object] | None:
    files = sorted((results_dir / "raw").glob("tatr_run_*.csv"))
    if not files:
        return None
    df = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    df.to_csv(results_dir / "tatr_long.csv", index=False)
    matrix = df.pivot_table(index="run_id", columns="augmentation_idx", values="mape", aggfunc="first").sort_index(axis=1)
    matrix.to_csv(results_dir / "tatr_matrix.csv")
    summary = summarize(matrix.to_numpy())
    summary.insert(0, "augmentation_idx", matrix.columns.astype(int))
    # Add per-level counts (number of runs contributing) so plotting can mask low-sample CI
    counts = matrix.notna().sum(axis=0)
    summary.insert(1, "n", [int(counts.loc[idx]) for idx in summary["augmentation_idx"]])
    if "synthetic_days" in df.columns:
        day_lookup = df.drop_duplicates("augmentation_idx").set_index("augmentation_idx")["synthetic_days"]
        summary.insert(2, "synthetic_days", [int(day_lookup.loc[idx]) for idx in summary["augmentation_idx"]])
    summary.to_csv(results_dir / "tatr_summary.csv", index=False)
    plot_summary(
        summary,
        "synthetic_days" if "synthetic_days" in summary.columns else "augmentation_idx",
        "SP500 TATR Downstream Test",
        "Synthetic days added",
        results_dir / "tatr_sp500_hpc",
    )
    best_idx = int(summary["avg"].idxmin())
    return {
        "runs": int(matrix.shape[0]),
        "levels": int(matrix.shape[1]),
        "baseline_mape": float(summary["avg"].iloc[0]),
        "best_level": int(summary["augmentation_idx"].iloc[best_idx]),
        "best_mape": float(summary["avg"].iloc[best_idx]),
        "final_mape": float(summary["avg"].iloc[-1]),
    }


def aggregate_tmtr(results_dir: Path) -> dict[str, object] | None:
    files = sorted((results_dir / "raw").glob("tmtr_run_*.csv"))
    if not files:
        return None
    df = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    df.to_csv(results_dir / "tmtr_long.csv", index=False)
    matrix = df.pivot_table(index="run_id", columns="synthetic_proportion_pct", values="mape", aggfunc="first").sort_index(axis=1)
    matrix.to_csv(results_dir / "tmtr_matrix.csv")
    summary = summarize(matrix.to_numpy())
    summary.insert(0, "synthetic_proportion_pct", matrix.columns.astype(int))
    counts = matrix.notna().sum(axis=0)
    summary.insert(1, "n", [int(counts.loc[idx]) for idx in summary["synthetic_proportion_pct"]])
    summary.to_csv(results_dir / "tmtr_summary.csv", index=False)
    plot_summary(
        summary,
        "synthetic_proportion_pct",
        "SP500 TMTR Downstream Test",
        "Synthetic proportion (%)",
        results_dir / "tmtr_sp500_hpc",
    )
    best_idx = int(summary["avg"].idxmin())
    return {
        "runs": int(matrix.shape[0]),
        "levels": int(matrix.shape[1]),
        "baseline_mape": float(summary["avg"].iloc[0]),
        "best_proportion_pct": int(summary["synthetic_proportion_pct"].iloc[best_idx]),
        "best_mape": float(summary["avg"].iloc[best_idx]),
        "final_mape": float(summary["avg"].iloc[-1]),
    }


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "results_dir": str(results_dir),
        "tatr": aggregate_tatr(results_dir),
        "tmtr": aggregate_tmtr(results_dir),
    }
    (results_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
