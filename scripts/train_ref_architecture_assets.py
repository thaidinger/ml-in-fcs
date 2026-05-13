from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "fts-diffusion-ref"


def configure_asset(asset: str, k: int) -> None:
    from models import model_params
    from models.model_params import pem_params, pgm_params, prm_params

    prm_params["dataname"] = asset
    prm_params["k"] = k
    pgm_params["n_patterns"] = k
    pem_params["n_patterns"] = k

    # Keep modules that imported the parameter dictionaries via `from ... import *`
    # aligned with the same mutated dictionary objects.
    model_params.prm_params = prm_params
    model_params.pgm_params = pgm_params
    model_params.pem_params = pem_params


def expected_outputs(asset: str, k: int) -> dict[str, str]:
    from models.load_models import get_pem_name, get_pgm_name

    return {
        "pgm_state_dict": str(REF / "trained_models" / get_pgm_name(state_dict=True)),
        "pgm_full_model": str(REF / "trained_models" / get_pgm_name(state_dict=False)),
        "pem_state_dict": str(REF / "trained_models" / get_pem_name(state_dict=True)),
        "pem_full_model": str(REF / "trained_models" / get_pem_name(state_dict=False)),
        "sisc_centroids": str(REF / "res" / f"sisc_{asset}_k{k}_l10-21_dba_kmpp_centroids.csv"),
        "sisc_labels": str(REF / "res" / f"sisc_{asset}_k{k}_l10-21_dba_kmpp_labels.csv"),
        "sisc_subsequences": str(REF / "res" / f"sisc_{asset}_k{k}_l10-21_dba_kmpp_subsequences.csv"),
        "sisc_segmentation": str(REF / "res" / f"sisc_{asset}_k{k}_l10-21_dba_kmpp_segmentation.csv"),
    }


def train_asset(asset: str, k: int, seed: int) -> dict[str, object]:
    from models.train_ftsdiffusion import train_ftsdiffusion
    from utils.load_data import load_actual_fts

    configure_asset(asset, k)
    outputs = expected_outputs(asset, k)
    missing_sisc = [path for name, path in outputs.items() if name.startswith("sisc_") and not Path(path).exists()]
    if missing_sisc:
        raise RuntimeError(f"Missing SISC artifacts for {asset}: {missing_sisc}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    series = load_actual_fts(asset).squeeze()
    train_ftsdiffusion(series, store_model=True)

    missing_models = [path for name, path in outputs.items() if not name.startswith("sisc_") and not Path(path).exists()]
    if missing_models:
        raise RuntimeError(f"Missing trained model artifacts for {asset}: {missing_models}")

    return {
        "asset": asset,
        "k": k,
        "n_prices": int(len(series)),
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train reference FTS-Diffusion architecture for selected assets.")
    parser.add_argument("--assets", nargs="+", default=["goog", "zcf"], choices=["goog", "zcf", "sp500"])
    parser.add_argument("--k", type=int, default=11)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.chdir(REF)
    sys.path.insert(0, str(REF))

    results = [train_asset(asset, args.k, args.seed) for asset in args.assets]
    out = REF / "trained_models" / "goog_zcf_k11_architecture_manifest.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
