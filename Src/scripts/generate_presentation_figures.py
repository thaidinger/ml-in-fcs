from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "reports" / "generated_outputs"
OUT = GEN / "04_presentation_figures"


COLORS = {
    "ink": "#1f2933",
    "muted": "#64748b",
    "grid": "#d8dee9",
    "green": "#2f855a",
    "blue": "#2563eb",
    "orange": "#d97706",
    "red": "#c2410c",
    "purple": "#7c3aed",
    "teal": "#0f766e",
    "gray": "#6b7280",
}


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 240,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 16,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.8,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_caption(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, 0.015, text, ha="center", va="bottom", fontsize=9, color=COLORS["muted"])


def rounded_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, color: str, title: str, body: str) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=color,
        facecolor=mpl.colors.to_rgba(color, 0.09),
    )
    ax.add_patch(box)
    ax.text(xy[0] + 0.03, xy[1] + height - 0.09, title, fontsize=13, fontweight="bold", color=color, va="top")
    ax.text(xy[0] + 0.03, xy[1] + height - 0.18, body, fontsize=10.5, color=COLORS["ink"], va="top", linespacing=1.45)


def plot_replication_map() -> None:
    fig, ax = plt.subplots(figsize=(14, 7.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.03, 0.94, "FTS-Diffusion Replication: What Was Tested", fontsize=22, fontweight="bold")
    ax.text(
        0.03,
        0.895,
        "Three separate evidence tracks: downstream predictive utility, real-asset pattern libraries, and Appendix B.3 simulated SISC.",
        fontsize=11.5,
        color=COLORS["muted"],
    )

    boxes = [
        (
            (0.04, 0.56),
            0.27,
            0.24,
            COLORS["blue"],
            "S&P 500 Downstream",
            "Author-style TATR/TMTR did not reproduce a stable improvement.\nContinuous synthetic trajectories do reproduce a paper-like drop,\nbut that protocol differs from the released reference path.",
        ),
        (
            (0.365, 0.56),
            0.27,
            0.24,
            COLORS["teal"],
            "GOOG / ZC=F SISC",
            "K=11 SISC pattern libraries were generated for both real assets.\nThese outputs are visual and architectural checks, not B.3 metrics,\nbecause real prices do not have ground-truth labels.",
        ),
        (
            (0.69, 0.56),
            0.27,
            0.24,
            COLORS["orange"],
            "Appendix B.3 SISC",
            "Toy-data runs and a timed sweep do not match the paper numbers.\nBest multi-pattern DTW is 0.0321 vs paper 0.01;\nbest interval IoU is 0.6418 vs paper 0.784.",
        ),
    ]
    for args in boxes:
        rounded_box(ax, *args)

    rounded_box(
        ax,
        (0.20, 0.18),
        0.60,
        0.19,
        COLORS["red"],
        "Current Conclusion",
        "The released materials are not enough for an exact author-faithful replication.\nThe strongest evidence points to missing protocol/generation details rather than a single bad seed.",
    )

    for start_x in [0.175, 0.50, 0.825]:
        arrow = FancyArrowPatch(
            (start_x, 0.55),
            (0.50, 0.38),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.5,
            color=COLORS["muted"],
            alpha=0.75,
            connectionstyle="arc3,rad=0.08",
        )
        ax.add_patch(arrow)

    add_caption(fig, "Presentation figure generated from reports/generated_outputs; see EXPERIMENT_REGISTRY.md for exact settings.")
    save(fig, "01_replication_story_map")


def plot_tatr_protocol_contrast() -> None:
    path = GEN / "01_sp500_downstream_replication" / "11_long_replication_batch" / "long_tatr_curve_summary.csv"
    df = pd.read_csv(path)
    df = df[(df["ahead"] == 1) & (df["protocol"].isin(["continuous_chunked", "continuous_cross_refit_scaler", "independent_fixed"]))]

    labels = {
        "continuous_chunked": "Continuous trajectory",
        "continuous_cross_refit_scaler": "Continuous + scaler refit",
        "independent_fixed": "Author-style independent blocks",
    }
    colors = {
        "continuous_chunked": COLORS["green"],
        "continuous_cross_refit_scaler": COLORS["blue"],
        "independent_fixed": COLORS["red"],
    }

    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    endpoints: dict[str, tuple[float, float]] = {}
    for protocol, group in df.groupby("protocol"):
        group = group.sort_values("augmentation_blocks")
        x = group["augmentation_blocks"].to_numpy()
        y = group["mean_pct"].to_numpy()
        ymin = group["min_pct"].to_numpy()
        ymax = group["max_pct"].to_numpy()
        ax.plot(x, y, marker="o", linewidth=2.6, color=colors[protocol], label=labels[protocol])
        ax.fill_between(x, ymin, ymax, color=colors[protocol], alpha=0.13)
        endpoints[protocol] = (float(x[-1]), float(y[-1]))

    ax.axhline(0, color="#111827", linewidth=1)
    ax.axhline(-17.9, color=COLORS["purple"], linestyle="--", linewidth=1.8, label="Original S&P 100-year target: -17.9%")
    ax.annotate(
        "original Fig. 6(b)\nfinal target",
        xy=(8, -17.9),
        xytext=(18, -45),
        textcoords="data",
        color=COLORS["purple"],
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": COLORS["purple"], "linewidth": 1.1},
    )
    ax.set_title("TATR Is Not Recovered Uniquely Without Rollout Details", loc="left", fontweight="bold")
    ax.set_xlabel("Synthetic augmentation years (252-day blocks)")
    ax.set_ylabel("MAPE change vs baseline (%)")
    ax.grid(axis="y")
    ax.set_ylim(-105, 230)
    ax.legend(ncol=2, loc="upper left")
    endpoint_labels = {
        "continuous_chunked": ("continuous final\n-65.2%", (72, -54), COLORS["green"]),
        "continuous_cross_refit_scaler": ("scaler-refit final\n-68.2%", (54, -92), COLORS["blue"]),
        "independent_fixed": ("independent-block final\n+193.2%", (84, 210), COLORS["red"]),
    }
    for protocol, (text, xytext, color) in endpoint_labels.items():
        x_last, y_last = endpoints[protocol]
        ax.annotate(
            text,
            xy=(x_last, y_last),
            xytext=xytext,
            textcoords="data",
            color=color,
            fontsize=10.5,
            fontweight="bold",
            ha="left",
            va="center",
            arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.1},
        )
    add_caption(fig, "Six seeds from long replication batch. Shaded bands show min/max across seeds; negative means lower MAPE.")
    save(fig, "02_tatr_protocol_contrast")


def plot_tatr_summary_bars() -> None:
    path = GEN / "01_sp500_downstream_replication" / "11_long_replication_batch" / "long_tatr_protocol_summary.csv"
    df = pd.read_csv(path)
    df = df[df["stage"].isin(["tatr_continuous_chunked", "tatr_continuous_refit", "tatr_independent_fixed"])]
    labels = ["Continuous\ntrajectory", "Continuous\n+ scaler refit", "Author-style\nindependent blocks"]
    order = ["tatr_continuous_chunked", "tatr_continuous_refit", "tatr_independent_fixed"]
    df = df.set_index("stage").loc[order].reset_index()

    x = np.arange(len(df))
    width = 0.32
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    bars1 = ax.bar(x - width / 2, df["best_pct_mean"], width, color=COLORS["teal"], label="Best observed")
    bars2 = ax.bar(x + width / 2, df["final_pct_mean"], width, color=COLORS["orange"], label="Final at max augmentation")
    ax.axhline(0, color="#111827", linewidth=1)
    ax.axhline(-17.9, color=COLORS["purple"], linestyle="--", linewidth=1.8, label="Paper S&P target: -17.9%")
    ax.set_title("Best vs Final TATR Change by Protocol", loc="left", fontweight="bold")
    ax.set_ylabel("MAPE change vs baseline (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y")
    ax.legend(loc="upper left")
    ax.set_ylim(-105, 230)

    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_height()
            va = "bottom" if value >= 0 else "top"
            offset = 4 if value >= 0 else -4
            ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.1f}%", ha="center", va=va, fontsize=9)

    add_caption(fig, "Continuous variants can improve sharply; independent fixed blocks degrade in the six-seed long batch.")
    save(fig, "03_tatr_summary_bars")


def plot_b3_metric_gap() -> None:
    max20 = json.loads((GEN / "03_appendix_b3_simulated_sisc" / "15_b3_sisc_simulated_replication_max20" / "manifest.json").read_text())
    sweep = json.loads((GEN / "03_appendix_b3_simulated_sisc" / "16_b3_sisc_sweep" / "manifest.json").read_text())

    dtw_values = {
        "Paper": 0.01,
        "Seed 42,\n20 iters": max20["multi_pattern"]["ours_avg_per_unit_dtw"],
        "Best sweep\nby DTW": sweep["best_by_dtw"]["avg_per_unit_dtw"],
    }
    iou_values = {
        "Paper": 0.784,
        "Seed 42,\n20 iters": max20["multi_pattern"]["ours_boundary_jaccard_tol2"],
        "Best sweep\ninterval IoU": sweep["best_by_author_interval_iou"]["author_interval_iou_pred"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.8))
    specs = [
        (axes[0], dtw_values, "Centroid Recovery: Per-Unit DTW", "Lower is better", COLORS["blue"], 0.05),
        (axes[1], iou_values, "Segmentation Recovery: Jaccard / IoU", "Higher is better", COLORS["orange"], 0.9),
    ]
    for ax, values, title, ylabel, color, ylim in specs:
        names = list(values)
        vals = list(values.values())
        bar_colors = [COLORS["green"], color, COLORS["red"]]
        bars = ax.bar(names, vals, color=bar_colors, alpha=0.9)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        ax.set_ylim(0, ylim)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + ylim * 0.025, f"{val:.3f}", ha="center", fontsize=10)

    fig.suptitle("Appendix B.3 Replication Gap", fontsize=20, fontweight="bold", x=0.06, ha="left")
    add_caption(fig, "B.3 targets come from the paper. Local values use the shipped toy data and reference SISC with runtime fixes.")
    save(fig, "04_b3_metric_gap")


def plot_b3_sweep_frontier() -> None:
    path = GEN / "03_appendix_b3_simulated_sisc" / "16_b3_sisc_sweep" / "sweep_results.csv"
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(12.8, 7.2))

    for (lmax, iters), group in df.groupby(["l_max", "max_iters"]):
        label = f"l_max={lmax}, iters={iters}"
        marker = "o" if lmax == 20 else "s"
        color = COLORS["blue"] if iters == 10 else COLORS["orange"] if lmax == 20 else COLORS["teal"]
        ax.scatter(
            group["avg_per_unit_dtw"],
            group["author_interval_iou_pred"],
            s=95,
            alpha=0.82,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            label=label,
        )

    ax.scatter([0.01], [0.784], s=180, marker="*", color=COLORS["purple"], edgecolor="white", linewidth=0.9, label="Paper target")
    best_dtw = df.loc[df["avg_per_unit_dtw"].idxmin()]
    best_iou = df.loc[df["author_interval_iou_pred"].idxmax()]
    ax.annotate("Best DTW", (best_dtw["avg_per_unit_dtw"], best_dtw["author_interval_iou_pred"]), xytext=(18, -22), textcoords="offset points", fontsize=10, arrowprops={"arrowstyle": "->", "color": COLORS["muted"]})
    ax.annotate("Best IoU", (best_iou["avg_per_unit_dtw"], best_iou["author_interval_iou_pred"]), xytext=(18, 15), textcoords="offset points", fontsize=10, arrowprops={"arrowstyle": "->", "color": COLORS["muted"]})

    ax.set_title("B.3 Sweep Does Not Reach the Paper Target", loc="left", fontweight="bold")
    ax.set_xlabel("Average per-unit DTW (lower is better)")
    ax.set_ylabel("Author-style interval IoU (higher is better)")
    ax.grid(True)
    ax.set_xlim(0.005, max(df["avg_per_unit_dtw"].max() + 0.004, 0.045))
    ax.set_ylim(0.55, 0.82)
    ax.legend(loc="lower left")
    add_caption(fig, "Each point is one multi-pattern SISC run on the shipped toy data.")
    save(fig, "05_b3_sweep_frontier")


def plot_real_asset_pattern_summary() -> None:
    path = GEN / "02_goog_zcf_real_asset_sisc" / "12_sisc_pattern_library" / "pattern_library_summary.csv"
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), sharey=True)
    asset_names = {"goog": "GOOG", "zcf": "ZC=F"}
    asset_colors = {"goog": COLORS["blue"], "zcf": COLORS["teal"]}

    for ax, asset in zip(axes, ["goog", "zcf"]):
        sub = df[df["asset"] == asset].sort_values("pattern")
        x = np.arange(len(sub))
        bars = ax.bar(x, sub["segment_count"], color=asset_colors[asset], alpha=0.88)
        ax2 = ax.twinx()
        ax2.plot(x, sub["mean_length"], color=COLORS["orange"], marker="o", linewidth=2, label="Mean segment length")
        ax.set_title(f"{asset_names[asset]} SISC Pattern Usage", loc="left", fontweight="bold")
        ax.set_xlabel("Pattern index")
        ax.set_ylabel("Segment count")
        ax2.set_ylabel("Mean length")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in sub["pattern"]])
        ax.grid(axis="y")
        ax2.set_ylim(9, 19)
        for bar in bars:
            value = int(bar.get_height())
            if value >= 40:
                ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, str(value), ha="center", fontsize=9)

    handles = [
        mpl.lines.Line2D([0], [0], color=COLORS["blue"], linewidth=8, label="Segment count"),
        mpl.lines.Line2D([0], [0], color=COLORS["orange"], marker="o", linewidth=2, label="Mean length"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Real-Asset SISC Pattern Libraries, K=11", fontsize=20, fontweight="bold", x=0.06, ha="left")
    add_caption(fig, "Pattern counts summarize learned real-data SISC segments; these are not ground-truth B.3 scores.")
    save(fig, "06_real_asset_sisc_pattern_summary")


def plot_experiment_settings_grid() -> None:
    path = GEN / "experiment_registry.csv"
    df = pd.read_csv(path)
    primary = df[~df["experiment"].isin(["audit", "report", "diagnostic", "comparison"])]
    counts = pd.crosstab(primary["storyline"], primary["experiment"])

    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    im = ax.imshow(counts.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(counts.columns)))
    ax.set_yticks(np.arange(len(counts.index)))
    ax.set_xticklabels(counts.columns, rotation=25, ha="right")
    storyline_labels = {
        "sp500_downstream": "S&P downstream",
        "appendix_b3_simulated_sisc": "B.3 simulated SISC",
        "goog_zcf_real_asset_sisc": "GOOG/ZC=F real SISC",
        "presentation": "Presentation figures",
    }
    ax.set_yticklabels([storyline_labels.get(name, str(name)) for name in counts.index])
    ax.set_title("Completed Experiment Coverage", loc="left", fontweight="bold")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            val = counts.values[i, j]
            ax.text(j, i, str(val), ha="center", va="center", color="white" if val > counts.values.max() / 2 else COLORS["ink"], fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Number of completed output folders")
    add_caption(fig, "Use EXPERIMENT_REGISTRY.md before rerunning experiments with matching settings keys.")
    save(fig, "07_experiment_coverage_grid")


def load_centroids(asset: str) -> pd.DataFrame:
    path = ROOT / "fts-diffusion-ref" / "res" / f"sisc_{asset}_k11_l10-21_dba_kmpp_centroids.csv"
    return pd.read_csv(path, index_col=0)


def centered_unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    scale = np.nanmax(np.abs(x))
    return x / scale if scale > 0 else x


def diffusion_schedule(n_steps: int = 30, min_beta: float = 1e-4, max_beta: float = 0.02) -> np.ndarray:
    betas = np.linspace(min_beta, max_beta, n_steps)
    alphas = 1.0 - betas
    return np.cumprod(alphas)


def plot_pattern_forward_diffusion() -> None:
    centroids = load_centroids("goog")
    pattern = centered_unit(centroids.iloc[0].to_numpy())
    rng = np.random.default_rng(42)
    noise = rng.normal(size=pattern.shape)
    alpha_bars = diffusion_schedule()
    steps = [0, 4, 9, 14, 20, 29]
    x_axis = np.arange(len(pattern))

    fig, axs = plt.subplots(2, 3, figsize=(14.5, 7.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.12, hspace=0.42, wspace=0.20)
    axs = axs.ravel()
    for ax, step in zip(axs, steps):
        if step == 0:
            noisy = pattern
        else:
            a_bar = alpha_bars[step]
            noisy = np.sqrt(a_bar) * pattern + np.sqrt(1.0 - a_bar) * noise
        ax.plot(x_axis, pattern, color=COLORS["gray"], linewidth=2.2, alpha=0.45, label="conditioning pattern p")
        ax.plot(x_axis, noisy, color=COLORS["blue"], linewidth=2.6, label="diffused residual/state")
        ax.axhline(0, color="#111827", linewidth=0.7, alpha=0.45)
        ax.set_title(f"Diffusion step t={step}", loc="left", fontweight="bold", fontsize=13)
        ax.set_ylim(-1.8, 1.8)
        ax.grid(axis="y")
        if step == 0:
            ax.legend(loc="lower left", fontsize=9)
    fig.suptitle("Forward Diffusion: Clean Pattern Signal Is Gradually Mixed With Noise", fontsize=19, fontweight="bold", x=0.06, y=0.97, ha="left")
    fig.text(
        0.06,
        0.89,
        "Visualized with a learned GOOG K=11 SISC centroid and the paper-code beta schedule: x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) noise.",
        fontsize=10.5,
        color=COLORS["muted"],
    )
    add_caption(fig, "This is the exact forward noising equation used by the pattern-conditioned diffusion module; the sampled noise is fixed across panels.")
    save(fig, "08_pattern_forward_diffusion_goog")


def plot_pattern_reverse_denoising_schematic() -> None:
    centroids = load_centroids("zcf")
    pattern = centered_unit(centroids.iloc[2].to_numpy())
    rng = np.random.default_rng(7)
    noise = rng.normal(size=pattern.shape)
    x_axis = np.arange(len(pattern))
    levels = [1.00, 0.72, 0.46, 0.22, 0.00]
    labels = ["start: Gaussian residual", "denoise", "denoise", "denoise", "generated latent"]

    fig, axs = plt.subplots(1, len(levels), figsize=(15, 4.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.70, bottom=0.17, wspace=0.20)
    for ax, level, label in zip(axs, levels, labels):
        residual = level * noise + (1.0 - level) * 0.12 * rng.normal(size=pattern.shape)
        generated = pattern + residual
        ax.plot(x_axis, pattern, color=COLORS["gray"], linewidth=2.1, alpha=0.48)
        ax.plot(x_axis, generated, color=COLORS["green"], linewidth=2.6)
        ax.axhline(0, color="#111827", linewidth=0.7, alpha=0.35)
        ax.set_title(label, fontsize=11.5, fontweight="bold")
        ax.set_ylim(-2.1, 2.1)
        ax.grid(axis="y")
    for left, right in zip(axs[:-1], axs[1:]):
        start = left.transAxes.transform((1.02, 0.52))
        end = right.transAxes.transform((-0.10, 0.52))
        inv = fig.transFigure.inverted()
        arrow = FancyArrowPatch(
            inv.transform(start),
            inv.transform(end),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.5,
            color=COLORS["muted"],
        )
        fig.add_artist(arrow)

    fig.suptitle("Reverse Process: Denoise a Random Residual While Holding the Pattern Fixed", fontsize=18.5, fontweight="bold", x=0.06, y=0.97, ha="left")
    fig.text(
        0.06,
        0.83,
        "Schematic of the learned PCDM generation path: noise is iteratively removed, conditioned on the selected SISC pattern.",
        fontsize=10.5,
        color=COLORS["muted"],
    )
    add_caption(fig, "The reverse path is schematic for presentation clarity; the conditioning pattern is an actual ZC=F K=11 SISC centroid.")
    save(fig, "09_pattern_reverse_denoising_schematic")


def plot_pattern_diffusion_mechanism() -> None:
    goog = load_centroids("goog")
    zcf = load_centroids("zcf")
    motif_a = centered_unit(goog.iloc[0].to_numpy())
    motif_b = centered_unit(zcf.iloc[2].to_numpy())
    rng = np.random.default_rng(4)
    stages = [
        ("1. SISC library", "Real series is segmented into motifs and each segment receives a pattern label.", motif_a, COLORS["teal"]),
        ("2. Condition p", "One motif is selected as the conditioning shape for the generator.", motif_a, COLORS["blue"]),
        ("3. Add noise", "Training adds scheduled Gaussian noise to the residual around that motif.", motif_a + 0.50 * rng.normal(size=len(motif_a)), COLORS["orange"]),
        ("4. Denoise", "Sampling starts from noise and repeatedly predicts the noise to remove.", motif_b + 0.20 * rng.normal(size=len(motif_b)), COLORS["purple"]),
        ("5. Output segment", "The result is decoded, rescaled by magnitude, and stitched into the synthetic path.", motif_b, COLORS["green"]),
    ]

    fig = plt.figure(figsize=(15.2, 7.4))
    fig.text(0.045, 0.93, "What Gets Diffused in FTS-Diffusion?", fontsize=22, fontweight="bold")
    fig.text(
        0.045,
        0.885,
        "The SISC motif is the condition; the diffusion model learns how real segments vary around that condition.",
        fontsize=11.5,
        color=COLORS["muted"],
    )
    gs = fig.add_gridspec(2, 5, left=0.045, right=0.965, top=0.78, bottom=0.24, height_ratios=[1.15, 1.0], wspace=0.34, hspace=0.12)
    top_axes = []
    for idx, (title, body, series, color) in enumerate(stages):
        ax_plot = fig.add_subplot(gs[0, idx])
        x = np.arange(len(series))
        ax_plot.plot(x, series, color=color, linewidth=2.8)
        ax_plot.axhline(0, color="#111827", linewidth=0.7, alpha=0.35)
        ax_plot.set_xticks([])
        ax_plot.set_yticks([])
        ax_plot.set_ylim(-1.9, 1.9)
        ax_plot.set_title(title, loc="left", fontsize=13, fontweight="bold", color=color, pad=8)
        for spine in ax_plot.spines.values():
            spine.set_color(mpl.colors.to_rgba(color, 0.50))
            spine.set_linewidth(1.2)
        ax_plot.set_facecolor(mpl.colors.to_rgba(color, 0.055))
        top_axes.append(ax_plot)

        ax_text = fig.add_subplot(gs[1, idx])
        ax_text.set_axis_off()
        ax_text.text(0, 0.92, textwrap.fill(body, 26), va="top", ha="left", fontsize=10.7, linespacing=1.35, color=COLORS["ink"])

    for left, right in zip(top_axes[:-1], top_axes[1:]):
        start = left.transAxes.transform((1.04, 0.50))
        end = right.transAxes.transform((-0.12, 0.50))
        inv = fig.transFigure.inverted()
        fig.add_artist(
            FancyArrowPatch(
                inv.transform(start),
                inv.transform(end),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=1.5,
                color=COLORS["muted"],
            )
        )

    fig.text(
        0.16,
        0.13,
        "Speaker line: the model is not diffusing the pattern library itself. It keeps the pattern visible and denoises a residual/latent segment around it.",
        fontsize=12,
        color=COLORS["red"],
        fontweight="bold",
        ha="left",
    )
    add_caption(fig, "Use this slide for a non-technical audience before showing the mathematical diffusion equation.")
    save(fig, "10_pattern_diffusion_mechanism")


def plot_pattern_diffusion_animation() -> None:
    centroids = load_centroids("goog")
    pattern = centered_unit(centroids.iloc[0].to_numpy())
    rng = np.random.default_rng(42)
    noise = rng.normal(size=pattern.shape)
    alpha_bars = diffusion_schedule()
    x_axis = np.arange(len(pattern))

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.set_ylim(-1.9, 1.9)
    ax.set_xlim(0, len(pattern) - 1)
    ax.grid(axis="y")
    ax.set_title("Forward Diffusion of a GOOG SISC Pattern", loc="left", fontweight="bold")
    ax.set_xlabel("Segment index")
    ax.set_ylabel("Normalized value")
    ax.plot(x_axis, pattern, color=COLORS["gray"], linewidth=2.2, alpha=0.45, label="clean pattern")
    line, = ax.plot([], [], color=COLORS["blue"], linewidth=2.8, label="noisy state")
    step_text = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.legend(loc="lower left")

    def update(frame: int):
        if frame == 0:
            noisy = pattern
        else:
            a_bar = alpha_bars[frame]
            noisy = np.sqrt(a_bar) * pattern + np.sqrt(1.0 - a_bar) * noise
        line.set_data(x_axis, noisy)
        step_text.set_text(f"t = {frame}")
        return line, step_text

    ani = animation.FuncAnimation(fig, update, frames=list(range(len(alpha_bars))), interval=110, blit=True)
    ani.save(OUT / "11_pattern_forward_diffusion_goog.gif", writer=animation.PillowWriter(fps=9))
    plt.close(fig)


def plot_pattern_diffusion_story_animation() -> None:
    centroids = load_centroids("goog")
    pattern = centered_unit(centroids.iloc[0].to_numpy())
    rng = np.random.default_rng(123)
    shared_noise = rng.normal(size=pattern.shape)
    small_residual = 0.12 * rng.normal(size=pattern.shape)
    alpha_bars = diffusion_schedule()
    x_axis = np.arange(len(pattern))
    n_frames = 132
    terminal_noisy = np.sqrt(alpha_bars[-1]) * pattern + np.sqrt(1.0 - alpha_bars[-1]) * shared_noise

    fig, axs = plt.subplots(1, 3, figsize=(12.8, 5.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.72, bottom=0.22, wspace=0.20)
    fig.suptitle("Pattern-Conditioned Diffusion", fontsize=19, fontweight="bold", x=0.055, y=0.965, ha="left")
    subtitle = fig.text(
        0.055,
        0.87,
        "The motif stays fixed. The residual/latent state is noised during training and denoised during sampling.",
        fontsize=11.0,
        color=COLORS["muted"],
        ha="left",
    )
    phase_text = fig.text(0.055, 0.125, "", fontsize=12.5, color=COLORS["ink"], fontweight="bold", ha="left")
    progress_bg = mpl.lines.Line2D([0.055, 0.985], [0.080, 0.080], transform=fig.transFigure, color=COLORS["grid"], linewidth=7)
    progress_fg = mpl.lines.Line2D([0.055, 0.055], [0.080, 0.080], transform=fig.transFigure, color=COLORS["blue"], linewidth=7)
    fig.add_artist(progress_bg)
    fig.add_artist(progress_fg)

    titles = [
        ("Fixed condition", COLORS["teal"]),
        ("Forward noising", COLORS["blue"]),
        ("Reverse denoising", COLORS["green"]),
    ]
    lines = []
    pattern_lines = []
    for ax, (title, color) in zip(axs, titles):
        ax.set_title(title, loc="left", fontsize=13.5, fontweight="bold", color=color)
        ax.set_ylim(-2.0, 2.0)
        ax.set_xlim(0, len(pattern) - 1)
        ax.grid(axis="y")
        ax.axhline(0, color="#111827", linewidth=0.7, alpha=0.35)
        pattern_line, = ax.plot(x_axis, pattern, color=COLORS["gray"], linewidth=2.2, alpha=0.34)
        line, = ax.plot(x_axis, pattern, color=color, linewidth=3.0)
        lines.append(line)
        pattern_lines.append(pattern_line)

    axs[0].text(0.03, 0.08, "conditioning motif", transform=axs[0].transAxes, fontsize=10.5, color=COLORS["muted"])
    axs[1].text(0.03, 0.08, "training path", transform=axs[1].transAxes, fontsize=10.5, color=COLORS["muted"])
    axs[2].text(0.03, 0.08, "sampling path", transform=axs[2].transAxes, fontsize=10.5, color=COLORS["muted"])

    def smoothstep(x: float) -> float:
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def update(frame: int):
        f = frame / (n_frames - 1)
        forward_phase = smoothstep((f - 0.08) / 0.34)
        reverse_phase = smoothstep((f - 0.56) / 0.34)
        step = min(len(alpha_bars) - 1, int(round(forward_phase * (len(alpha_bars) - 1))))
        a_bar = alpha_bars[step]
        forward = np.sqrt(a_bar) * pattern + np.sqrt(1.0 - a_bar) * shared_noise
        generated = pattern + small_residual
        reverse = (1.0 - reverse_phase) * terminal_noisy + reverse_phase * generated

        lines[0].set_data(x_axis, pattern)
        lines[1].set_data(x_axis, forward)
        lines[2].set_data(x_axis, reverse)
        progress_fg.set_data([0.055, 0.055 + 0.93 * f], [0.080, 0.080])
        if f < 0.50:
            phase = "Training: gradually add noise to the segment representation"
            progress_fg.set_color(COLORS["blue"])
        else:
            phase = "Sampling: remove noise while keeping the pattern as condition"
            progress_fg.set_color(COLORS["green"])
        phase_text.set_text(phase)
        return [*lines, progress_fg, phase_text, subtitle, *pattern_lines]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=125, blit=False)
    ani.save(OUT / "12_pattern_conditioned_diffusion_story.gif", writer=animation.PillowWriter(fps=8))
    plt.close(fig)


def plot_pattern_diffusion_matched_endpoint() -> None:
    centroids = load_centroids("goog")
    pattern = centered_unit(centroids.iloc[0].to_numpy())
    rng = np.random.default_rng(123)
    shared_noise = rng.normal(size=pattern.shape)
    small_residual = 0.12 * rng.normal(size=pattern.shape)
    alpha_bars = diffusion_schedule()
    x_axis = np.arange(len(pattern))
    terminal_noisy = np.sqrt(alpha_bars[-1]) * pattern + np.sqrt(1.0 - alpha_bars[-1]) * shared_noise
    generated = pattern + small_residual

    panels = [
        ("1. Clean condition", pattern, COLORS["teal"], "SISC motif is held fixed"),
        ("2. Forward endpoint", terminal_noisy, COLORS["blue"], "same high-noise state"),
        ("3. Reverse start", terminal_noisy, COLORS["blue"], "exactly matches panel 2"),
        ("4. Denoised output", generated, COLORS["green"], "schematic generated segment"),
    ]

    fig, axs = plt.subplots(1, 4, figsize=(14.8, 5.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.70, bottom=0.22, wspace=0.22)
    fig.suptitle("Matched Diffusion Handoff", fontsize=19, fontweight="bold", x=0.055, y=0.96, ha="left")
    fig.text(
        0.055,
        0.865,
        "The terminal forward-noising state is reused as the initial reverse-denoising state.",
        fontsize=11,
        color=COLORS["muted"],
        ha="left",
    )

    for ax, (title, series, color, note) in zip(axs, panels):
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=color)
        ax.plot(x_axis, pattern, color=COLORS["gray"], linewidth=2.2, alpha=0.34, label="condition")
        ax.plot(x_axis, series, color=color, linewidth=3.0, label="state")
        ax.axhline(0, color="#111827", linewidth=0.7, alpha=0.35)
        ax.grid(axis="y")
        ax.set_xlim(0, len(pattern) - 1)
        ax.set_ylim(-2.0, 2.0)
        ax.text(0.03, 0.08, note, transform=ax.transAxes, fontsize=10.3, color=COLORS["muted"])

    for left, right in zip(axs[:-1], axs[1:]):
        start = left.transAxes.transform((1.04, 0.50))
        end = right.transAxes.transform((-0.12, 0.50))
        inv = fig.transFigure.inverted()
        fig.add_artist(
            FancyArrowPatch(
                inv.transform(start),
                inv.transform(end),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=1.5,
                color=COLORS["muted"],
            )
        )

    diff = float(np.max(np.abs(panels[1][1] - panels[2][1])))
    fig.text(
        0.055,
        0.115,
        f"Check: max absolute difference between forward endpoint and reverse start = {diff:.1e}",
        fontsize=11.5,
        color=COLORS["ink"],
        fontweight="bold",
    )
    add_caption(fig, "Forward endpoint and reverse start are numerically identical in this visualization.")
    save(fig, "12_pattern_conditioned_diffusion_matched_endpoint")


def extra_animation_files() -> list[str]:
    files = sorted(path.name for path in OUT.glob("13_specific_segment_diffusion_*.gif"))
    specific_manifest = OUT / "specific_segment_diffusion_manifest.json"
    if specific_manifest.exists():
        files.append(specific_manifest.name)
    return files


def write_readme(files: list[str]) -> None:
    rows = "\n".join(f"- `{name}.png` / `{name}.pdf`" for name in files)
    rows += "\n- `11_pattern_forward_diffusion_goog.gif`"
    rows += "\n- `12_pattern_conditioned_diffusion_story.gif`"
    for extra_file in extra_animation_files():
        rows += f"\n- `{extra_file}`"
    text = f"""# Presentation Figures

Slide-ready figures generated from the organized results in `reports/generated_outputs`.

## Files

{rows}

## Suggested Use

- `01_replication_story_map`: opening or conclusion slide.
- `02_tatr_protocol_contrast`: core explanation for why protocol details matter.
- `03_tatr_summary_bars`: compact result comparison for protocol outcomes.
- `04_b3_metric_gap`: Appendix B.3 replication failure in one slide.
- `05_b3_sweep_frontier`: evidence that seed/iteration/l_max sweeps did not reach the paper target.
- `06_real_asset_sisc_pattern_summary`: GOOG/ZC=F pattern-library overview.
- `07_experiment_coverage_grid`: methods/status slide showing what has already been run.
- `08_pattern_forward_diffusion_goog`: exact forward noising equation on a learned GOOG motif.
- `09_pattern_reverse_denoising_schematic`: slide-friendly reverse denoising schematic conditioned on a ZC=F motif.
- `10_pattern_diffusion_mechanism`: non-technical mechanism diagram for what is diffused.
- `11_pattern_forward_diffusion_goog`: short GIF animation of the forward process.
- `12_pattern_conditioned_diffusion_story`: slower complete GIF animation of conditioning, forward noising, and reverse denoising.
- `12_pattern_conditioned_diffusion_matched_endpoint`: static PDF/PNG proving the forward endpoint equals the reverse start.
- `13_specific_segment_diffusion_*`: optional specific-segment GIFs generated for concrete examples.

Generated by `scripts/generate_presentation_figures.py`.

Specific-segment GIFs are generated by `scripts/generate_specific_segment_diffusion_gif.py`.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup()
    plot_replication_map()
    plot_tatr_protocol_contrast()
    plot_tatr_summary_bars()
    plot_b3_metric_gap()
    plot_b3_sweep_frontier()
    plot_real_asset_pattern_summary()
    plot_experiment_settings_grid()
    plot_pattern_forward_diffusion()
    plot_pattern_reverse_denoising_schematic()
    plot_pattern_diffusion_mechanism()
    plot_pattern_diffusion_animation()
    plot_pattern_diffusion_story_animation()
    plot_pattern_diffusion_matched_endpoint()
    names = [
        "01_replication_story_map",
        "02_tatr_protocol_contrast",
        "03_tatr_summary_bars",
        "04_b3_metric_gap",
        "05_b3_sweep_frontier",
        "06_real_asset_sisc_pattern_summary",
        "07_experiment_coverage_grid",
        "08_pattern_forward_diffusion_goog",
        "09_pattern_reverse_denoising_schematic",
        "10_pattern_diffusion_mechanism",
        "12_pattern_conditioned_diffusion_matched_endpoint",
    ]
    write_readme(names)
    manifest = {
        "output": str(OUT),
        "source_tables": [
            "01_sp500_downstream_replication/11_long_replication_batch/long_tatr_curve_summary.csv",
            "01_sp500_downstream_replication/11_long_replication_batch/long_tatr_protocol_summary.csv",
            "03_appendix_b3_simulated_sisc/15_b3_sisc_simulated_replication_max20/manifest.json",
            "03_appendix_b3_simulated_sisc/16_b3_sisc_sweep/manifest.json",
            "03_appendix_b3_simulated_sisc/16_b3_sisc_sweep/sweep_results.csv",
            "02_goog_zcf_real_asset_sisc/12_sisc_pattern_library/pattern_library_summary.csv",
            "experiment_registry.csv",
            "fts-diffusion-ref/res/sisc_goog_k11_l10-21_dba_kmpp_centroids.csv",
            "fts-diffusion-ref/res/sisc_zcf_k11_l10-21_dba_kmpp_centroids.csv",
        ],
        "files": [f"{name}.{ext}" for name in names for ext in ("png", "pdf")]
        + ["11_pattern_forward_diffusion_goog.gif", "12_pattern_conditioned_diffusion_story.gif"]
        + extra_animation_files()
        + ["README.md", "manifest.json"],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
