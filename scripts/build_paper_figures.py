from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "generated_outputs"
FIGURES = ROOT / "paper_figures"

DIAG = REPORTS / "00_release_pipeline_diagnostics"
AUDIT = REPORTS / "01_sp500_downstream_replication" / "06_tatr_audit"
B3 = REPORTS / "03_appendix_b3_simulated_sisc"

INK = "#111111"
MUTED = "#555555"
GRID = "#d7d7d7"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
GRAY = "#7A7A7A"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 7.4,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.15,
            "lines.markersize": 3.0,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.015,
        }
    )


def clean_axis(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(axis="both", width=0.55, length=2.5, color=INK, pad=1.5)
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.45)
        ax.set_axisbelow(True)


def save_pdf(fig: plt.Figure, filename: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES / filename
    fig.savefig(pdf_path, format="pdf", transparent=False)
    plt.close(fig)


def asset_label(asset: str) -> str:
    return {"sp500": "S&P", "goog": "GOOG", "zcf": "ZC=F"}[asset]


def write_release_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    DIAG.mkdir(parents=True, exist_ok=True)

    generation = pd.DataFrame(
        [
            {
                "asset": "sp500",
                "protocol": "continuous_rollout",
                "synthetic_days": 25200,
                "linear_fit_r2": 0.99999,
                "real_return_sd": 0.0109,
                "synthetic_return_sd": 0.00046,
                "real_to_synthetic_sd_ratio": 0.0109 / 0.00046,
                "pgm_decoder_cross_pattern_sd_upper_bound": 0.001,
                "source": "tracked paper diagnostic used for Figure 05",
            }
        ]
    )
    pem = pd.DataFrame(
        [
            {"asset": "sp500", "k": 14, "segments": 2520, "dominant_pattern_share_pct": 99.6, "regime": "fixed point"},
            {"asset": "goog", "k": 11, "segments": 2520, "dominant_pattern_share_pct": 50.0, "regime": "period-2 cycle"},
            {"asset": "zcf", "k": 11, "segments": 1867, "dominant_pattern_share_pct": 50.0, "regime": "period-2 cycle"},
        ]
    )
    returns = pd.DataFrame(
        [
            {"asset": "sp500", "protocol": "authors", "real_sd": 0.0109, "synthetic_sd": 0.0045, "mean_ks_p": 9e-4, "indistinguishable_pct": 0},
            {"asset": "sp500", "protocol": "continuous", "real_sd": 0.0109, "synthetic_sd": 0.00046, "mean_ks_p": 4e-7, "indistinguishable_pct": 0},
            {"asset": "goog", "protocol": "authors", "real_sd": 0.0184, "synthetic_sd": 0.0026, "mean_ks_p": 6e-5, "indistinguishable_pct": 0},
            {"asset": "goog", "protocol": "continuous", "real_sd": 0.0184, "synthetic_sd": 0.00029, "mean_ks_p": 4e-7, "indistinguishable_pct": 0},
            {"asset": "zcf", "protocol": "authors", "real_sd": 0.0181, "synthetic_sd": 0.0028, "mean_ks_p": 2e-5, "indistinguishable_pct": 0},
            {"asset": "zcf", "protocol": "continuous", "real_sd": 0.0181, "synthetic_sd": 0.00041, "mean_ks_p": 1e-6, "indistinguishable_pct": 0},
        ]
    )
    returns["windows"] = 300
    returns["window_length"] = 60
    returns["real_to_synthetic_sd_ratio"] = returns["real_sd"] / returns["synthetic_sd"]

    generation.to_csv(DIAG / "generation_degeneracy_summary.csv", index=False)
    pem.to_csv(DIAG / "pem_state_collapse_summary.csv", index=False)
    returns.to_csv(DIAG / "return_distribution_summary.csv", index=False)
    (DIAG / "README.md").write_text(
        "# Release Pipeline Diagnostics\n\n"
        "Self-contained summaries used to regenerate `paper_figures/05_degeneracy_sp500.pdf`.\n\n"
        "- `generation_degeneracy_summary.csv`: S&P 500 path linearity and return-volatility compression.\n"
        "- `pem_state_collapse_summary.csv`: dominant PEM state share by asset.\n"
        "- `return_distribution_summary.csv`: 300-window return-distribution summaries.\n",
        encoding="utf-8",
    )
    (DIAG / "manifest.json").write_text(
        json.dumps(
            {
                "description": "Tracked summaries used by scripts/build_paper_figures.py for the degeneracy figure.",
                "files": [
                    "generation_degeneracy_summary.csv",
                    "pem_state_collapse_summary.csv",
                    "return_distribution_summary.csv",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return generation, pem, returns


def plot_release_diagnostics() -> None:
    generation, pem, returns = write_release_diagnostics()

    fig, axes = plt.subplots(2, 2, figsize=(6.75, 3.65), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    row = generation.iloc[0]
    metrics = pd.DataFrame(
        {
            "metric": ["linear fit $R^2$", "real/synthetic return sd", "decoder cross-pattern sd"],
            "scaled": [row["linear_fit_r2"], min(row["real_to_synthetic_sd_ratio"] / 25.0, 1.0), row["pgm_decoder_cross_pattern_sd_upper_bound"] / 0.002],
            "display": [f"{row['linear_fit_r2']:.5f}", f"{row['real_to_synthetic_sd_ratio']:.1f}x", "<0.001"],
            "color": [BLUE, ORANGE, GRAY],
        }
    )
    y = np.arange(len(metrics))
    ax_a.barh(y, [1.0] * len(metrics), color="#f2f2f2", edgecolor="none", height=0.46)
    ax_a.barh(y, metrics["scaled"], color=metrics["color"], height=0.46)
    for idx, display in enumerate(metrics["display"]):
        ax_a.text(1.03, idx, display, va="center", ha="left", fontsize=6.5, color=INK)
    ax_a.set_yticks(y, metrics["metric"])
    ax_a.set_xlim(0, 1.34)
    ax_a.set_xticks([])
    ax_a.invert_yaxis()
    ax_a.set_title("(a) S&P path collapse", loc="left", fontweight="bold")
    clean_axis(ax_a, grid=False)
    ax_a.spines["bottom"].set_visible(False)
    ax_a.spines["left"].set_visible(False)

    pem_plot = pem.copy()
    pem_plot["uniform_pct"] = 100 / pem_plot["k"]
    x = np.arange(len(pem_plot))
    ax_b.bar(x, pem_plot["dominant_pattern_share_pct"], color=[BLUE, ORANGE, ORANGE], width=0.55, label="dominant state")
    ax_b.scatter(x, pem_plot["uniform_pct"], marker="_", color=INK, s=90, linewidth=1.1, label="uniform $1/K$")
    for xi, share in zip(x, pem_plot["dominant_pattern_share_pct"]):
        ax_b.text(xi, share + 3, f"{share:.1f}%", ha="center", va="bottom", fontsize=6.2)
    ax_b.set_xticks(x, ["S&P\nfixed", "GOOG\nperiod-2", "ZC=F\nperiod-2"])
    ax_b.set_ylim(0, 110)
    ax_b.set_ylabel("segments (%)")
    ax_b.set_title("(b) PEM state collapse", loc="left", fontweight="bold")
    ax_b.legend(frameon=False, loc="upper right", handlelength=1.1)
    clean_axis(ax_b)

    returns_plot = returns.copy()
    returns_plot["group"] = returns_plot["asset"].map(asset_label) + " " + returns_plot["protocol"].map({"authors": "A", "continuous": "C"})
    groups = np.arange(len(returns_plot))
    width = 0.34
    ax_c.bar(groups - width / 2, returns_plot["real_sd"], width=width, color="#bdbdbd", label="real")
    ax_c.bar(groups + width / 2, returns_plot["synthetic_sd"], width=width, color=BLUE, label="synthetic")
    ax_c.set_xticks(groups, returns_plot["group"], rotation=30, ha="right")
    ax_c.set_ylim(0, 0.0205)
    ax_c.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.3f}"))
    ax_c.set_ylabel("daily return sd")
    ax_c.set_title("(c) Return-volatility compression", loc="left", fontweight="bold")
    ax_c.legend(frameon=False, ncol=2, loc="upper right", handlelength=1.1)
    clean_axis(ax_c)

    table_rows = [
        [
            asset_label(r["asset"]),
            "cont." if r["protocol"] == "continuous" else "auth.",
            f"{r['synthetic_sd']:.5f}",
            f"{r['mean_ks_p']:.0e}",
            f"{int(r['indistinguishable_pct'])}%",
        ]
        for _, r in returns.iterrows()
    ]
    ax_d.axis("off")
    ax_d.set_title("(d) KS return test", loc="left", fontweight="bold")
    table = ax_d.table(
        cellText=table_rows,
        colLabels=["asset", "prot.", "syn. sd", "mean KS $p$", "pass"],
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.15, 0.15, 0.20, 0.22, 0.13],
        bbox=[0.02, 0.03, 0.96, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.1)
    table.scale(1.0, 1.15)
    for (r, _c), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f2f2")
    save_pdf(fig, "05_degeneracy_sp500.pdf")


def plot_tatr_protocol_contrast() -> None:
    profiles = pd.read_csv(AUDIT / "tatr_results_profiles.csv")
    sp = profiles[
        (profiles["asset"].eq("sp500"))
        & (profiles["k"].eq("k14"))
        & (profiles["protocol"].isin(["authors", "single", "random_init"]))
    ].copy()
    labels = {
        "authors": "independent fixed",
        "single": "continuous chunked",
        "random_init": "random init",
    }
    colors = {"authors": RED, "single": BLUE, "random_init": GREEN}
    markers = {"authors": "o", "single": "s", "random_init": "^"}

    fig, ax = plt.subplots(figsize=(3.35, 2.15), constrained_layout=True)
    baseline = float(sp[sp["aug"].eq(0)]["mean"].iloc[0])
    for protocol in ["authors", "single", "random_init"]:
        group = sp[sp["protocol"].eq(protocol)].sort_values("aug")
        x = group["aug"].to_numpy(dtype=float)
        mean = group["mean"].to_numpy(dtype=float)
        std = group["std"].to_numpy(dtype=float)
        ax.plot(x, mean, marker=markers[protocol], color=colors[protocol], label=labels[protocol])
        ax.fill_between(x, np.maximum(mean - std, 0), mean + std, color=colors[protocol], alpha=0.10, linewidth=0)
    ax.axhline(baseline, color=INK, linestyle="--", linewidth=0.65, label="baseline")
    ax.set_xlabel("synthetic years")
    ax.set_ylabel("S&P 500 MAPE")
    ax.set_ylim(0, 0.23)
    ax.legend(frameon=False, loc="upper left", handlelength=1.1)
    clean_axis(ax)
    save_pdf(fig, "02_tatr_protocol_contrast.pdf")


def tatr_row(summary: pd.DataFrame, asset: str, k: str, protocol: str) -> pd.Series:
    rows = summary[
        summary["asset"].eq(asset)
        & summary["k"].eq(k)
        & summary["protocol"].eq(protocol)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one TATR audit row for {(asset, k, protocol)}, found {len(rows)}")
    return rows.iloc[0]


def plot_cross_asset_nontransfer() -> None:
    summary = pd.read_csv(AUDIT / "tatr_results_audit_summary.csv")
    rows = []
    for label, asset, k in [("S&P 500", "sp500", "k14"), ("GOOG", "goog", "k11"), ("ZC=F", "zcf", "k11")]:
        split = tatr_row(summary, asset, k, "single")
        rows.append(
            {
                "asset": label,
                "best": float(split["best_pct_vs_base"]),
                "final": float(split["final_pct_vs_base"]),
            }
        )
    data = pd.DataFrame(rows)
    x = np.arange(len(data))
    width = 0.33
    fig, ax = plt.subplots(figsize=(3.35, 2.15), constrained_layout=True)
    ax.axhline(0, color=INK, linewidth=0.6)
    best = ax.bar(x - width / 2, data["best"], width=width, color=BLUE, label="best")
    final = ax.bar(x + width / 2, data["final"], width=width, color=ORANGE, label="final")
    for bars in [best, final]:
        for bar in bars:
            value = bar.get_height()
            va = "bottom" if value >= 0 else "top"
            dy = 6 if value >= 0 else -6
            ax.text(bar.get_x() + bar.get_width() / 2, value + dy, f"{value:+.0f}%", ha="center", va=va, fontsize=6.1)
    ax.set_xticks(x, data["asset"])
    y_min = min(-110, float(data[["best", "final"]].min().min()) - 35)
    y_max = float(data[["best", "final"]].max().max()) + 55
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("MAPE change (%)")
    ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.1)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.0f}"))
    clean_axis(ax)
    save_pdf(fig, "07_cross_asset_nontransfer.pdf")


def plot_b3_metric_gap() -> None:
    max20 = json.loads((B3 / "15_b3_sisc_simulated_replication_max20" / "manifest.json").read_text())
    sweep = json.loads((B3 / "16_b3_sisc_sweep" / "manifest.json").read_text())
    sweep_df = pd.read_csv(B3 / "16_b3_sisc_sweep" / "sweep_results.csv")

    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.15), constrained_layout=True)
    dtw_vals = [0.010, max20["multi_pattern"]["ours_avg_per_unit_dtw"], sweep["best_by_dtw"]["avg_per_unit_dtw"]]
    seg_vals = [0.784, max20["multi_pattern"]["ours_boundary_jaccard_tol2"], sweep["best_by_author_interval_iou"]["author_interval_iou_pred"]]
    labels = ["paper", "seed 42", "best sweep"]
    colors = [GRAY, BLUE, ORANGE]

    ax = axes[0]
    bars = ax.bar(np.arange(3), dtw_vals, color=colors, width=0.58)
    ax.axhline(0.010, linestyle="--", color=INK, linewidth=0.65)
    for bar, value in zip(bars, dtw_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.001, f"{value:.3f}", ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(np.arange(3), labels, rotation=15, ha="right")
    ax.set_ylim(0, 0.052)
    ax.set_ylabel("per-unit DTW")
    ax.set_title("(a) Centroids", loc="left", fontweight="bold")
    clean_axis(ax)

    ax = axes[1]
    bars = ax.bar(np.arange(3), seg_vals, color=colors, width=0.58)
    ax.axhline(0.784, linestyle="--", color=INK, linewidth=0.65)
    for bar, value in zip(bars, seg_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(np.arange(3), labels, rotation=15, ha="right")
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Jaccard / IoU")
    ax.set_title("(b) Segmentation", loc="left", fontweight="bold")
    clean_axis(ax)

    ax = axes[2]
    for iters, group in sweep_df.groupby("max_iters"):
        ax.scatter(
            group["avg_per_unit_dtw"],
            group["boundary_jaccard_tol2"],
            s=18,
            alpha=0.85,
            color=BLUE if int(iters) == 10 else ORANGE,
            label=f"{int(iters)} iters",
        )
    ax.scatter([0.010], [0.784], marker="*", s=95, color=GRAY, edgecolor=INK, linewidth=0.4, label="paper")
    ax.set_xlabel("per-unit DTW")
    ax.set_ylabel("boundary Jaccard")
    ax.set_xlim(0.007, 0.044)
    ax.set_ylim(0.0, 0.86)
    ax.set_title("(c) Sweep", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower left", handletextpad=0.2)
    clean_axis(ax)

    save_pdf(fig, "04_b3_metric_gap.pdf")


def main() -> None:
    set_style()
    plot_release_diagnostics()
    plot_tatr_protocol_contrast()
    plot_cross_asset_nontransfer()
    plot_b3_metric_gap()


if __name__ == "__main__":
    main()
