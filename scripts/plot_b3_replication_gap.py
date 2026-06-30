"""Regenerate the Appendix B.3 SISC replication-gap figure (figures/04_b3_metric_gap.pdf).

Panels (a) and (b) reproduce the vetted summary bars already reported in the paper
and Table 5 (treated as fixed constants here, not re-derived). Panel (c) is built
directly from the 18-run sweep in
reports/generated_outputs/03_appendix_b3_simulated_sisc/16_b3_sisc_sweep/sweep_results.csv
and shows every run against the published target, using the two columns that are
fully reproducible from that file: per-unit DTW and boundary Jaccard (tol 2).

Run: .venv/bin/python scripts/plot_b3_replication_gap.py
"""

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SWEEP = (
    ROOT
    / "reports/generated_outputs/03_appendix_b3_simulated_sisc"
    / "16_b3_sisc_sweep/sweep_results.csv"
)
OUT = ROOT / "NeurIPS_Paper/figures/04_b3_metric_gap.pdf"

# Vetted summary values (paper text and Table 5); kept as fixed constants.
PAPER = "#2ca02c"
SEED42 = "#4C72B0"
BEST = "#DD8452"
TARGET_DTW = 0.01
TARGET_JACC = 0.784

dtw_vals = [0.010, 0.040, 0.032]  # paper, seed 42, best sweep
seg_vals = [0.784, 0.372, 0.642]  # paper (Jaccard), seed 42 (Jaccard), best sweep (interval IoU)
labels = ["Paper", "Seed 42", "Best sweep"]
colors = [PAPER, SEED42, BEST]

df = pd.read_csv(SWEEP)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

# (a) Centroid recovery: per-unit DTW (lower is better).
ax = axes[0]
bars = ax.bar(labels, dtw_vals, color=colors, width=0.62)
ax.axhline(TARGET_DTW, ls="--", lw=1, color=PAPER, alpha=0.7)
ax.set_title("(a) Centroid recovery")
ax.set_ylabel("per-unit DTW (lower is better)")
ax.set_ylim(0, max(dtw_vals) * 1.25)
for b, v in zip(bars, dtw_vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

# (b) Segmentation recovery: Jaccard / interval IoU (higher is better).
ax = axes[1]
bars = ax.bar(labels, seg_vals, color=colors, width=0.62)
ax.axhline(TARGET_JACC, ls="--", lw=1, color=PAPER, alpha=0.7)
ax.set_title("(b) Segmentation recovery")
ax.set_ylabel("Jaccard / interval IoU (higher is better)")
ax.set_ylim(0, 0.9)
for b, v in zip(bars, seg_vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

# (c) Full sweep against the published target (18 runs, raw columns).
ax = axes[2]
for iters, sub in df.groupby("max_iters"):
    ax.scatter(
        sub["avg_per_unit_dtw"],
        sub["boundary_jaccard_tol2"],
        s=34,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
        label=f"{int(iters)} iters",
    )
ax.scatter([TARGET_DTW], [TARGET_JACC], marker="*", s=240, color=PAPER, edgecolor="black", linewidth=0.6, zorder=5)
ax.annotate(
    "paper target",
    xy=(TARGET_DTW, TARGET_JACC),
    xytext=(TARGET_DTW + 0.004, TARGET_JACC - 0.12),
    fontsize=8,
    arrowprops=dict(arrowstyle="->", lw=0.8),
)
ax.set_title(f"(c) Sweep vs. target ({len(df)} runs)")
ax.set_xlabel("per-unit DTW (lower is better)")
ax.set_ylabel("boundary Jaccard, tol 2 (higher is better)")
ax.set_ylim(0, 0.9)
ax.legend(fontsize=7, frameon=False, loc="upper right")

fig.suptitle("Appendix B.3 SISC replication gap", y=1.02, fontsize=11)
fig.text(0.5, -0.04, "Targets are Huang et al. values; panel (c) compares every sweep run against the published per-unit DTW and boundary-Jaccard target.", ha="center", fontsize=7.5, color="0.35")
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
