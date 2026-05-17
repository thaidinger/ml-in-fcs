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
        values = np.sort(errors[:, col].astype(float))
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
    ax.plot(x, summary["avg"], marker="o", linewidth=1.8, markersize=3.8, color="#1f77b4")
    ax.fill_between(x, summary["min"], summary["max"], color="#1f77b4", alpha=0.18)
    ax.axhline(summary["avg"].iloc[0], color="0.45", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MAPE")
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
    if "synthetic_days" in df.columns:
        day_lookup = df.drop_duplicates("augmentation_idx").set_index("augmentation_idx")["synthetic_days"]
        summary.insert(1, "synthetic_days", [int(day_lookup.loc[idx]) for idx in summary["augmentation_idx"]])
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
