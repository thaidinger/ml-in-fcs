#!/usr/bin/env python3
"""Launch full FTS-Diffusion training (PGM + PEM) for S&P500 from the repo.

This script is a thin wrapper that prepares paths and calls the reference
training entrypoints in `fts-diffusion-ref` so the training runs in the
expected directory layout (trained_models/, res/, data/).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root containing fts-diffusion-ref")
    p.add_argument("--store-model", action="store_true", help="Save trained checkpoints")
    p.add_argument("--debug", action="store_true", help="Run fast smoke test (downloads data + 1 epoch)")
    p.add_argument("--recognition-max-iters", type=int, default=None, help="Override SISC recognition iterations")
    p.add_argument("--generation-epochs", type=int, default=None, help="Override PGM training epochs")
    p.add_argument("--evolution-epochs", type=int, default=None, help="Override PEM training epochs")
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    ref_root = repo_root / "fts-diffusion-ref"
    if not ref_root.exists():
        raise SystemExit(f"fts-diffusion-ref not found under {repo_root}; ensure payload unpacked")

    # Work from the reference tree so relative model/data paths match
    os.chdir(ref_root)
    sys.path.insert(0, str(ref_root))

    # Import reference training helpers
    from train_all import get_fts, load_actual_fts  # type: ignore
    from models.train_ftsdiffusion import train_ftsdiffusion

    print(f"[train_sp500] getting series and running training from {ref_root}")

    # Ensure data present (train_all/get_fts will download if needed)
    get_fts(ticker="^GSPC", fts_name="sp500", start_date="1980-01-01", end_date="2020-01-01")
    fts = load_actual_fts("sp500").squeeze()

    if args.debug:
        # Short-circuit: temporarily reduce all training stages for smoke test unless explicitly overridden.
        if args.recognition_max_iters is None:
            args.recognition_max_iters = 1
        if args.generation_epochs is None:
            args.generation_epochs = 1
        if args.evolution_epochs is None:
            args.evolution_epochs = 1

    train_ftsdiffusion(
        fts,
        train_test_split=0.8,
        store_model=args.store_model,
        recognition_max_iters=args.recognition_max_iters,
        generation_epochs=args.generation_epochs,
        evolution_epochs=args.evolution_epochs,
    )


if __name__ == "__main__":
    main()
