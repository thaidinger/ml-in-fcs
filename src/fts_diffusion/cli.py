from __future__ import annotations

import argparse

from fts_diffusion.training.pipeline import sample_from_run, train_from_path


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train FTS-Diffusion from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    args = parser.parse_args()
    train_from_path(args.config)


def sample_main() -> None:
    parser = argparse.ArgumentParser(description="Sample a synthetic time series from a trained run.")
    parser.add_argument("--run-dir", required=True, help="Training run directory.")
    parser.add_argument("--length", type=int, default=None, help="Desired terminal sequence length.")
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    args = parser.parse_args()
    sample_from_run(run_dir=args.run_dir, terminal_length=args.length, output_path=args.output)

