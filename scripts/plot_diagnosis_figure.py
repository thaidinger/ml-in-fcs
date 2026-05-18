#!/usr/bin/env python3
"""Presentation figure: first 2 blocks of the FTS-Diffusion synthetic GOOG (K=11)
paths, authors vs split — makes it evident that every 252-day block is the same
motif (the generator collapses to a single pattern).

Output: FTS_diffusion_presentation/figures/diagnosis/goog_pattern_repetition.{pdf,png}
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "FTS_diffusion_presentation", "figures", "diagnosis")
ZOOM = 2 * 252  # first 2 blocks
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def load_runs(folder: str, pattern: str, n: int = 5):
    """Load up to n runs as 1-D arrays (authors blocks are concatenated)."""
    runs = []
    for f in sorted(glob.glob(os.path.join(folder, pattern)))[:n]:
        runs.append(np.load(f).astype(float).reshape(-1))
    return runs


def panel(ax, runs, title):
    for i, r in enumerate(runs):
        seg = r[:ZOOM]
        ax.plot(np.arange(len(seg)), seg, color=COLORS[i % len(COLORS)],
                lw=1.2, alpha=0.85, label=f"run {i:02d}")
    ax.axvline(252, color="gray", ls=":", lw=1.0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("synthetic day")
    ax.set_ylabel("price")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)


def main():
    os.makedirs(OUT, exist_ok=True)
    syn = os.path.join(ROOT, "synthetic", "goog", "k11")
    authors = load_runs(os.path.join(syn, "authors"), "run_*_blocks.npy")
    split = load_runs(os.path.join(syn, "split"), "run_*_continuous.npy")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    panel(axes[0], authors,
          "authors protocol  —  block resets, then replays the same motif")
    panel(axes[1], split,
          "split protocol  —  the same motif tiled continuously")
    fig.suptitle("FTS-Diffusion synthetic GOOG (K=11): every 252-day block is the "
                 "same motif  (first 2 blocks, 5 runs; dotted = block boundary)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"goog_pattern_repetition.{ext}"), dpi=130)
    plt.close(fig)
    print(f"written: {OUT}/goog_pattern_repetition.{{pdf,png}}")


if __name__ == "__main__":
    main()
