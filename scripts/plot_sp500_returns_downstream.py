from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

COLORS = {
    "avg": "#1f77b4",
    "run": "#9aa7b2",
    "band": "#aec7e8",
    "best": "#d62728",
    "tmtr": "#2ca02c",
    "tatr": "#1f77b4",
}


def save(fig: plt.Figure, report_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(report_dir / f"{name}.png", dpi=220)
    fig.savefig(report_dir / f"{name}.pdf")
    plt.close(fig)


def load_outputs(report_dir: Path, datatype: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tatr_matrix = pd.read_csv(report_dir / f"tatr_{datatype}_matrix.csv")
    tmtr_matrix = pd.read_csv(report_dir / f"tmtr_{datatype}_matrix.csv")
    tatr_summary = pd.read_csv(report_dir / f"tatr_{datatype}_summary_authors_style.csv")
    tmtr_summary = pd.read_csv(report_dir / f"tmtr_{datatype}_summary_authors_style.csv")
    return tatr_matrix, tmtr_matrix, tatr_summary, tmtr_summary


def line_with_runs(
    matrix: pd.DataFrame,
    summary: pd.DataFrame,
    x_col: str,
    x_label: str,
    title: str,
    filename: str,
    color: str,
    report_dir: Path,
    datatype: str,
) -> None:
    x = summary[x_col].to_numpy()
    y_avg = summary["avg"].to_numpy()
    y_min = summary["min"].to_numpy()
    y_max = summary["max"].to_numpy()
    best_idx = int(np.nanargmin(y_avg))

    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    for _, row in matrix.iterrows():
        ax.plot(x, row.to_numpy(dtype=float), color=COLORS["run"], alpha=0.45, linewidth=1)
    ax.fill_between(x, y_min, y_max, color=color, alpha=0.16, label="authors-style band")
    ax.plot(x, y_avg, color=color, marker="o", linewidth=2.2, markersize=4.8, label="summary avg")
    ax.scatter([x[best_idx]], [y_avg[best_idx]], color=COLORS["best"], s=54, zorder=5, label="best avg")
    ax.axhline(y_avg[0], color="0.35", linestyle="--", linewidth=1, label="0% / no augmentation")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"MAPE on real {datatype}")
    ax.grid(alpha=0.24, linewidth=0.7)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    save(fig, report_dir, filename)


def heatmap(
    matrix: pd.DataFrame,
    x_labels: np.ndarray,
    title: str,
    x_label: str,
    filename: str,
    report_dir: Path,
) -> None:
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Run")
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels([str(i + 1) for i in range(values.shape[0])])
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("MAPE")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                f"{values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if values[i, j] > np.nanmedian(values) else "black",
            )
    save(fig, report_dir, filename)


def combined_summary(tatr_summary: pd.DataFrame, tmtr_summary: pd.DataFrame, report_dir: Path, datatype: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))

    ax = axes[0]
    x = tatr_summary["augmentation_idx"].to_numpy()
    y = tatr_summary["avg"].to_numpy()
    best = int(np.nanargmin(y))
    ax.plot(x, y, color=COLORS["tatr"], marker="o", linewidth=2.1)
    ax.fill_between(x, tatr_summary["min"], tatr_summary["max"], color=COLORS["tatr"], alpha=0.16)
    ax.scatter(x[best], y[best], color=COLORS["best"], s=54, zorder=5)
    ax.axhline(y[0], color="0.35", linestyle="--", linewidth=1)
    ax.set_title(f"TATR {datatype.title()}")
    ax.set_xlabel("Augmentation block")
    ax.set_ylabel("MAPE")
    ax.grid(alpha=0.24, linewidth=0.7)

    ax = axes[1]
    x = tmtr_summary["synthetic_proportion_pct"].to_numpy()
    y = tmtr_summary["avg"].to_numpy()
    best = int(np.nanargmin(y))
    ax.plot(x, y, color=COLORS["tmtr"], marker="o", linewidth=2.1)
    ax.fill_between(x, tmtr_summary["min"], tmtr_summary["max"], color=COLORS["tmtr"], alpha=0.16)
    ax.scatter(x[best], y[best], color=COLORS["best"], s=54, zorder=5)
    ax.axhline(y[0], color="0.35", linestyle="--", linewidth=1)
    ax.set_title(f"TMTR {datatype.title()}")
    ax.set_xlabel("Synthetic proportion (%)")
    ax.set_ylabel("MAPE")
    ax.grid(alpha=0.24, linewidth=0.7)

    fig.suptitle(f"SP500 {datatype.title()} Downstream Replication, Reduced Author-Style Run", y=1.03)
    save(fig, report_dir, f"combined_{datatype}_summary")


def improvement_bars(tatr_summary: pd.DataFrame, tmtr_summary: pd.DataFrame, report_dir: Path, datatype: str) -> None:
    tatr_base = float(tatr_summary["avg"].iloc[0])
    tmtr_base = float(tmtr_summary["avg"].iloc[0])
    tatr_delta = (tatr_summary["avg"] - tatr_base) / tatr_base * 100
    tmtr_delta = (tmtr_summary["avg"] - tmtr_base) / tmtr_base * 100

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=True)
    axes[0].bar(tatr_summary["augmentation_idx"], tatr_delta, color=COLORS["tatr"], alpha=0.82)
    axes[0].axhline(0, color="0.25", linewidth=1)
    axes[0].set_title("TATR vs no augmentation")
    axes[0].set_xlabel("Augmentation block")
    axes[0].set_ylabel("Avg MAPE change (%)")
    axes[0].grid(axis="y", alpha=0.24, linewidth=0.7)

    axes[1].bar(tmtr_summary["synthetic_proportion_pct"], tmtr_delta, width=7, color=COLORS["tmtr"], alpha=0.82)
    axes[1].axhline(0, color="0.25", linewidth=1)
    axes[1].set_title("TMTR vs 0% synthetic")
    axes[1].set_xlabel("Synthetic proportion (%)")
    axes[1].grid(axis="y", alpha=0.24, linewidth=0.7)
    save(fig, report_dir, f"relative_{datatype}_mape_change")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datatype", default="returns", choices=["prices", "returns"])
    args = parser.parse_args()

    report_dir = ROOT / "reports" / f"downstream_{args.datatype}_sp500"
    report_dir.mkdir(parents=True, exist_ok=True)
    tatr_matrix, tmtr_matrix, tatr_summary, tmtr_summary = load_outputs(report_dir, args.datatype)

    line_with_runs(
        tatr_matrix,
        tatr_summary,
        "augmentation_idx",
        "Augmentation block",
        f"SP500 TATR on {args.datatype.title()}",
        f"tatr_{args.datatype}_runs",
        COLORS["tatr"],
        report_dir,
        args.datatype,
    )
    line_with_runs(
        tmtr_matrix,
        tmtr_summary,
        "synthetic_proportion_pct",
        "Synthetic proportion (%)",
        f"SP500 TMTR on {args.datatype.title()}",
        f"tmtr_{args.datatype}_runs",
        COLORS["tmtr"],
        report_dir,
        args.datatype,
    )
    heatmap(
        tatr_matrix,
        tatr_summary["augmentation_idx"].to_numpy(),
        f"TATR {args.datatype.title()} MAPE by Run",
        "Augmentation block",
        f"tatr_{args.datatype}_heatmap",
        report_dir,
    )
    heatmap(
        tmtr_matrix,
        tmtr_summary["synthetic_proportion_pct"].to_numpy(),
        f"TMTR {args.datatype.title()} MAPE by Run",
        "Synthetic proportion (%)",
        f"tmtr_{args.datatype}_heatmap",
        report_dir,
    )
    combined_summary(tatr_summary, tmtr_summary, report_dir, args.datatype)
    improvement_bars(tatr_summary, tmtr_summary, report_dir, args.datatype)


if __name__ == "__main__":
    main()
