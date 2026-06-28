from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REF_RES = ROOT / "fts-diffusion-ref" / "res"

ASSET_LABELS = {
    "goog": "GOOG",
    "zcf": "ZC=F Corn Futures",
}


def load_sisc(asset: str, k: int) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    prefix = REF_RES / f"sisc_{asset}_k{k}_l10-21_dba_kmpp"
    centroids = pd.read_csv(str(prefix) + "_centroids.csv", index_col=0)
    labels = pd.read_csv(str(prefix) + "_labels.csv", index_col=0).iloc[:, 0].astype(int)
    segmentation = pd.read_csv(str(prefix) + "_segmentation.csv", index_col=0).iloc[:, 0].astype(int)
    return centroids, labels, segmentation


def normalize_row(row: np.ndarray) -> np.ndarray:
    lo = np.nanmin(row)
    hi = np.nanmax(row)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.zeros_like(row, dtype=float)
    return (row - lo) / (hi - lo)


def plot_pattern_grid(asset: str, k: int, centroids: pd.DataFrame, counts: pd.Series, out: Path) -> None:
    title = ASSET_LABELS.get(asset, asset.upper())
    ncols = 4
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 6.8), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for idx in range(k):
        ax = axes[idx]
        y = normalize_row(centroids.iloc[idx].to_numpy(dtype=float))
        x = np.arange(len(y))
        ax.plot(x, y, color="#1f5f8b", linewidth=2)
        ax.fill_between(x, y, np.nanmin(y), color="#1f5f8b", alpha=0.13)
        ax.set_title(f"p{idx + 1}  n={int(counts.get(idx, 0))}", fontsize=10)
        ax.grid(alpha=0.22, linewidth=0.6)
        ax.set_ylim(-0.08, 1.08)

    for ax in axes[k:]:
        ax.axis("off")

    fig.suptitle(f"{title} SISC Pattern Library (K={k})", fontsize=14, fontweight="bold")
    fig.supxlabel("Normalized segment time")
    fig.supylabel("Normalized magnitude")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / f"{asset}_k{k}_pattern_library.png", dpi=220)
    fig.savefig(out / f"{asset}_k{k}_pattern_library.pdf")
    plt.close(fig)


def plot_pattern_counts(asset: str, k: int, counts: pd.Series, lengths: np.ndarray, out: Path) -> None:
    title = ASSET_LABELS.get(asset, asset.upper())
    x = np.arange(k)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    axes[0].bar(x + 1, [int(counts.get(i, 0)) for i in x], color="#2f6f4e")
    axes[0].set_title("Cluster assignment counts")
    axes[0].set_xlabel("Pattern")
    axes[0].set_ylabel("Segments")
    axes[0].set_xticks(x + 1)
    axes[0].grid(axis="y", alpha=0.24, linewidth=0.7)

    axes[1].hist(lengths, bins=np.arange(10, 23) - 0.5, color="#8a4f1d", alpha=0.86)
    axes[1].set_title("SISC segment length distribution")
    axes[1].set_xlabel("Segment length")
    axes[1].set_ylabel("Segments")
    axes[1].set_xticks(np.arange(10, 22))
    axes[1].grid(axis="y", alpha=0.24, linewidth=0.7)

    fig.suptitle(f"{title} Pattern Library Statistics", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out / f"{asset}_k{k}_pattern_stats.png", dpi=220)
    fig.savefig(out / f"{asset}_k{k}_pattern_stats.pdf")
    plt.close(fig)


def build_summary(asset: str, k: int, labels: pd.Series, segmentation: pd.Series) -> pd.DataFrame:
    lengths = np.diff(segmentation.to_numpy(dtype=int))
    rows = []
    for pattern_idx in range(k):
        assigned = labels.to_numpy(dtype=int) == pattern_idx
        assigned_lengths = lengths[assigned]
        rows.append(
            {
                "asset": asset,
                "k": k,
                "pattern": pattern_idx + 1,
                "label": pattern_idx,
                "segment_count": int(assigned.sum()),
                "mean_length": float(np.mean(assigned_lengths)) if len(assigned_lengths) else np.nan,
                "min_length": int(np.min(assigned_lengths)) if len(assigned_lengths) else np.nan,
                "max_length": int(np.max(assigned_lengths)) if len(assigned_lengths) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot stored SISC pattern libraries.")
    parser.add_argument("--assets", nargs="+", default=["goog", "zcf"], choices=sorted(ASSET_LABELS))
    parser.add_argument("--k", type=int, default=11)
    parser.add_argument("--output", default="reports/generated_outputs/12_sisc_pattern_library")
    args = parser.parse_args()

    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    manifest = []
    for asset in args.assets:
        centroids, labels, segmentation = load_sisc(asset, args.k)
        counts = labels.value_counts().sort_index()
        lengths = np.diff(segmentation.to_numpy(dtype=int))
        plot_pattern_grid(asset, args.k, centroids, counts, out)
        plot_pattern_counts(asset, args.k, counts, lengths, out)
        summaries.append(build_summary(asset, args.k, labels, segmentation))
        manifest.extend(
            [
                f"{asset}_k{args.k}_pattern_library.png",
                f"{asset}_k{args.k}_pattern_library.pdf",
                f"{asset}_k{args.k}_pattern_stats.png",
                f"{asset}_k{args.k}_pattern_stats.pdf",
            ]
        )

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out / "pattern_library_summary.csv", index=False)
    (out / "manifest.json").write_text(
        json.dumps({"assets": args.assets, "k": args.k, "files": sorted(manifest + ["pattern_library_summary.csv"])}, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({"output": str(out), "files": sorted(manifest + ["pattern_library_summary.csv"])}, indent=2))


if __name__ == "__main__":
    main()
