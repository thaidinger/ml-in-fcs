#!/usr/bin/env python3
"""Plot the synthetic price trajectories for the 'authors' and 'split' protocols.

For every experiment under synthetic/<asset>/k<K>/ that has 'authors' and/or
'split' data, produce one PNG with up to two panels:

  authors : run_*_blocks.npy      -> (n_blocks, 252), concatenated end-to-end.
            Each block is an INDEPENDENT 252-day trajectory restarted from the
            fixed initial segment, so the concatenation oscillates inside a
            narrow band around the init level.
  split   : run_*_continuous.npy  -> one continuous 25200-day trajectory.

All runs are drawn thin/transparent; the cross-run mean is drawn bold.
Output: figures/synthetic_trajectories/<asset>_k<K>.png
"""
from __future__ import annotations

import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYN = os.path.join(ROOT, "synthetic")
OUT = os.path.join(ROOT, "figures", "synthetic_trajectories")

# protocol -> (filename glob, loader). authors blocks are concatenated.
PROTOCOLS = {
    "authors": ("run_*_blocks.npy", "blocks"),
    "split":   ("run_*_continuous.npy", "continuous"),
}


def load_runs(res_dir: str, pattern: str, kind: str):
    """Return list of 1-D trajectories, one per run."""
    runs = []
    for f in sorted(glob.glob(os.path.join(res_dir, pattern))):
        arr = np.load(f).astype(np.float64)
        if kind == "blocks":          # (n_blocks, 252) -> flat concatenation
            arr = arr.reshape(-1)
        else:                          # already 1-D
            arr = arr.reshape(-1)
        if arr.size:
            runs.append(arr)
    return runs


def cross_run_mean(runs: list[np.ndarray]):
    """Mean across runs, trimmed to the shortest run length."""
    if not runs:
        return None
    n = min(len(r) for r in runs)
    return np.stack([r[:n] for r in runs], axis=0).mean(axis=0)


ZOOM_DAYS = 2 * 252  # first 2 blocks, so the per-block shape/resets are visible


def plot_full(ax, runs: list[np.ndarray], title: str):
    """Whole trajectory: every run faint, cross-run mean bold."""
    for r in runs:
        ax.plot(np.arange(len(r)), r, color="steelblue", lw=0.5, alpha=0.30)
    mean = cross_run_mean(runs)
    if mean is not None:
        ax.plot(np.arange(len(mean)), mean, color="darkblue", lw=2.0,
                label=f"mean of {len(runs)} runs")
        lo = min(float(r.min()) for r in runs)
        hi = max(float(r.max()) for r in runs)
        ax.set_title(f"{title} — full  (n={len(runs)};  range [{lo:.0f}, {hi:.0f}])",
                     fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.set_title(f"{title} — full  (no data)", fontsize=10)
    ax.set_xlabel("synthetic day")
    ax.set_ylabel("price")
    ax.grid(alpha=0.3)


def plot_zoom(ax, runs: list[np.ndarray], title: str):
    """First ~8 blocks of up to 5 individual runs, as distinct lines."""
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for i, r in enumerate(runs[:5]):
        seg = r[:ZOOM_DAYS]
        ax.plot(np.arange(len(seg)), seg, color=colors[i], lw=1.0,
                alpha=0.85, label=f"run {i:02d}")
    for b in range(1, ZOOM_DAYS // 252):
        ax.axvline(b * 252, color="gray", ls=":", lw=0.6, alpha=0.6)
    ax.set_title(f"{title} — zoom (first {ZOOM_DAYS // 252} blocks; "
                 f"dotted = 252-day boundaries)", fontsize=10)
    ax.set_xlabel("synthetic day")
    ax.set_ylabel("price")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)


def discover():
    """Yield (asset, k, {protocol: res_dir}) for every experiment."""
    for asset in sorted(os.listdir(SYN)):
        a_dir = os.path.join(SYN, asset)
        if not os.path.isdir(a_dir):
            continue
        for kdir in sorted(os.listdir(a_dir)):
            if not (kdir.startswith("k") and kdir[1:].isdigit()):
                continue
            present = {}
            for proto, (pat, _) in PROTOCOLS.items():
                d = os.path.join(a_dir, kdir, proto)
                if os.path.isdir(d) and glob.glob(os.path.join(d, pat)):
                    present[proto] = d
            if present:
                yield asset, kdir, present


def main():
    os.makedirs(OUT, exist_ok=True)
    made = []
    for asset, kdir, present in discover():
        protos = [p for p in ("authors", "split") if p in present]
        fig, axes = plt.subplots(2, len(protos), figsize=(7.6 * len(protos), 8.4),
                                 squeeze=False)
        for col, proto in enumerate(protos):
            pat, kind = PROTOCOLS[proto]
            runs = load_runs(present[proto], pat, kind)
            label = f"{asset} {kdir} — {proto}"
            plot_full(axes[0][col], runs, label)
            plot_zoom(axes[1][col], runs, label)
        fig.suptitle(f"Synthetic trajectories — {asset} {kdir}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = os.path.join(OUT, f"{asset}_{kdir}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        made.append((f"{asset}/{kdir}", protos, path))
        print(f"  {asset}/{kdir:5s}  protocols={protos}  -> {path}")

    print(f"\n{len(made)} figures written to {OUT}")


if __name__ == "__main__":
    main()
