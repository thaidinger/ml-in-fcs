#!/usr/bin/env python3
"""Plot the real downstream series for the three assets, split into the
LSTM training set (initial real period, TATR year 0) and the LSTM test set.

Replicates get_downstream_data(): the downstream series is the concatenation
of the test 20% of the SISC subsequences (split_ratio=0.8). The train/test
boundary uses the notebook's adaptive rule init_fraction = 0.625, clamped.

Output: figures/downstream_data/downstream_train_test.png
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "downstream_data")

# asset -> (K used in the TATR experiments, pretty name)
ASSETS = [
    ("sp500", 14, "S&P 500"),
    ("goog", 11, "GOOG"),
    ("zcf", 11, "ZC=F (Corn futures)"),
]
SPLIT_RATIO = 0.8      # SISC segment train/test split (prepare_segments)
INIT_FRACTION = 0.625  # adaptive LSTM init fraction (notebook cell 16)


def downstream_series(asset: str, k: int) -> np.ndarray:
    """Concatenate the test-20% SISC subsequences -> downstream timeseries."""
    f = os.path.join(ROOT, "architectures", asset, f"k{k}", "res",
                     f"sisc_{asset}_k{k}_l10-21_dba_kmpp_subsequences.csv")
    sub = pd.read_csv(f).values[:, 1]
    segs = [np.array(s.strip("[]").split(), dtype=np.float64) for s in sub]
    n_test = int(len(segs) * SPLIT_RATIO)
    return np.concatenate(segs[n_test:])


def init_period(total_len: int) -> int:
    """Adaptive init period: 0.625 of the series, clamped to [252, len-252]."""
    p = int(total_len * INIT_FRACTION)
    p = min(p, total_len - 252)
    return max(p, 252)


def main():
    os.makedirs(OUT, exist_ok=True)

    # --- figure 1: full downstream series, LSTM train vs test ---
    fig, axes = plt.subplots(len(ASSETS), 1, figsize=(12, 10), squeeze=False)
    for ax, (asset, k, pretty) in zip(axes[:, 0], ASSETS):
        ds = downstream_series(asset, k)
        L = len(ds)
        cut = init_period(L)
        x = np.arange(L)
        ax.plot(x[:cut + 1], ds[:cut + 1], color="steelblue", lw=1.3,
                label=f"LSTM training set (real, year 0) — {cut} d / {cut/252:.2f} y")
        ax.plot(x[cut:], ds[cut:], color="#d62728", lw=1.3,
                label=f"LSTM test set — {L-cut} d / {(L-cut)/252:.2f} y")
        ax.axvline(cut, color="gray", ls="--", lw=1.0)
        ax.axhline(ds[:cut].max(), color="steelblue", ls=":", lw=0.8, alpha=0.7)
        ax.set_title(f"{pretty}  (k={k})  —  downstream series: {L} d "
                     f"({L/252:.2f} y);  price range [{ds.min():.1f}, {ds.max():.1f}]",
                     fontsize=11)
        ax.set_xlabel("downstream day")
        ax.set_ylabel("price")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Downstream data — LSTM training set vs. test set "
                 "(dotted = max training price)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p1 = os.path.join(OUT, "downstream_train_test.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    print(f"written: {p1}")

    # --- figure 2: LSTM test set per asset, with linear-trend fit ---
    fig, axes = plt.subplots(len(ASSETS), 1, figsize=(11, 9), squeeze=False)
    for ax, (asset, k, pretty) in zip(axes[:, 0], ASSETS):
        ds = downstream_series(asset, k)
        test = ds[init_period(len(ds)):]
        x = np.arange(len(test))
        ax.plot(x, test, color="#d62728", lw=1.5, label="LSTM test set (real)")
        slope, intercept = np.polyfit(x, test, 1)        # price per day
        ax.plot(x, intercept + slope * x, color="black", ls="--", lw=1.5,
                label="linear fit")
        pct_yr = slope * 252 / test.mean() * 100         # %/yr of mean price
        ax.set_title(f"{pretty}  (k={k})  —  LSTM test set: {len(test)} d "
                     f"({len(test)/252:.2f} y);  trend {slope*252:+.1f}/yr "
                     f"({pct_yr:+.1f}%/yr of mean price)", fontsize=11)
        ax.set_xlabel("test day")
        ax.set_ylabel("price")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("LSTM test set per asset — does it trend up?  (dashed = linear fit)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p2 = os.path.join(OUT, "test_sets_trend.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    print(f"written: {p2}")


if __name__ == "__main__":
    main()
