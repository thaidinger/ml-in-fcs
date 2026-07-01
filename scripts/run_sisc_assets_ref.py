from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "fts-diffusion-ref"

ASSETS = {
    "goog": {"ticker": "GOOG", "start": "2005-01-01", "end": "2020-01-01"},
    "zcf": {"ticker": "ZC=F", "start": "2001-01-01", "end": "2020-01-01"},
}


def patch_sisc_kmeanspp() -> None:
    """Patch the released k-means++ helper without editing reference code on disk."""
    from models.pattern_recognition_module import SISC

    def _weight_next_centroid(self, series, remain_idxs, centroids):
        weights = np.zeros(len(remain_idxs), dtype=float)
        for pos, idx in enumerate(remain_idxs):
            seq = series[idx : idx + self.l_max]
            weights[pos] = min(self._compute_scaled_dtw(seq, c) for c in centroids)
        total = weights.sum()
        if total <= 0 or not np.isfinite(total):
            return np.ones(len(remain_idxs), dtype=float) / len(remain_idxs)
        return weights / total

    def stop_criteria(self, new_centroids, new_loss, epsilon=1e-6):
        if not hasattr(self, "coverage"):
            self.coverage = 0
        for old, new in zip(self.centroids, new_centroids):
            dtw = self._compute_scaled_dtw(old, new)
            if dtw > epsilon:
                self.coverage = 0
                return False
        if abs(self.total_loss - new_loss) > epsilon:
            self.coverage = 0
            return False
        if self.coverage <= 3:
            self.coverage += 1
            return False
        return True

    SISC._weight_next_centroid = _weight_next_centroid
    SISC.stop_criteria = stop_criteria


def download_close(asset: str, ticker: str, start: str, end: str) -> Path:
    out = REF / "data" / f"{asset}_timeseries.csv"
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No rows downloaded for {asset} ({ticker})")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    if close.empty:
        raise RuntimeError(f"No usable close prices for {asset} ({ticker})")
    close.to_csv(out, index=False)
    return out


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def run_asset(asset: str, k: int, max_iters: int, seed: int) -> dict[str, object]:
    from models.pattern_recognition_module import train_recognition_module
    from utils.load_data import load_actual_fts

    spec = ASSETS[asset]
    data_path = download_close(asset, spec["ticker"], spec["start"], spec["end"])
    series = load_actual_fts(asset).squeeze()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    centroids, labels, subsequences, segmentation = train_recognition_module(
        series,
        dataname=asset,
        n_clusters=k,
        l_min=10,
        l_max=21,
        max_iters=max_iters,
        init_strategy="kmeans++",
        barycenter="dba",
        plot_progress=False,
        plot_loss=False,
        store_res=True,
    )

    prefix = REF / "res" / f"sisc_{asset}_k{k}_l10-21_dba_kmpp"
    artifact_paths = {
        "centroids": Path(str(prefix) + "_centroids.csv"),
        "labels": Path(str(prefix) + "_labels.csv"),
        "subsequences": Path(str(prefix) + "_subsequences.csv"),
        "segmentation": Path(str(prefix) + "_segmentation.csv"),
    }
    missing = [rel(path) for path in artifact_paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing expected SISC artifacts for {asset}: {missing}")

    return {
        "asset": asset,
        "ticker": spec["ticker"],
        "data_path": rel(data_path),
        "n_prices": int(len(series)),
        "k": k,
        "max_iters": max_iters,
        "n_centroids": int(len(centroids)),
        "n_labels": int(len(labels)),
        "n_subsequences": int(len(subsequences)),
        "n_segmentation_boundaries": int(len(segmentation)),
        "artifacts": {name: rel(path) for name, path in artifact_paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reference SISC artifacts for GOOG and ZC=F.")
    parser.add_argument("--assets", nargs="+", default=["goog", "zcf"], choices=sorted(ASSETS))
    parser.add_argument("--k", type=int, default=11)
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.chdir(REF)
    sys.path.insert(0, str(REF))
    patch_sisc_kmeanspp()

    results = [run_asset(asset, args.k, args.max_iters, args.seed) for asset in args.assets]
    out = REF / "res" / "sisc_goog_zcf_k11_replication_manifest.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
