from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "fts-diffusion-ref"

sys.path.insert(0, str(REF))


def summarize_authors_style(errors: np.ndarray) -> pd.DataFrame:
    summary = np.zeros((3, errors.shape[1]))
    for col in range(errors.shape[1]):
        values = np.sort(errors[:, col].copy())
        percentile = int(np.ceil(len(values) * 0.025))
        if percentile == 0 or len(values) <= 2 * percentile:
            trimmed = values
            low = values[0]
            high = values[-1]
        else:
            trimmed = values[percentile:-percentile]
            low = values[percentile]
            high = values[-percentile]
        summary[0, col] = np.mean(trimmed)
        summary[1, col] = low
        summary[2, col] = high
    return pd.DataFrame({"avg": summary[0], "min": summary[1], "max": summary[2]})


def save_tatr_plots(out: Path, matrix: pd.DataFrame, summary: pd.DataFrame, aug_length: int) -> None:
    x_blocks = summary["augmentation_idx"].to_numpy()
    x_days = x_blocks * aug_length
    avg = summary["avg"].to_numpy()
    lo = summary["min"].to_numpy()
    hi = summary["max"].to_numpy()
    best_idx = int(np.argmin(avg))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for _, row in matrix.iterrows():
        ax.plot(x_blocks, row.to_numpy(dtype=float), color="#9aa7b2", alpha=0.42, linewidth=1)
    ax.fill_between(x_blocks, lo, hi, color="#1f77b4", alpha=0.16, label="authors-style band")
    ax.plot(x_blocks, avg, color="#1f77b4", marker="o", linewidth=2.2, label="summary avg")
    ax.scatter([x_blocks[best_idx]], [avg[best_idx]], color="#d62728", s=58, zorder=5, label="best avg")
    ax.axhline(avg[0], color="0.35", linestyle="--", linewidth=1, label="no augmentation")
    ax.set_title("SP500 TATR on Prices, author-style reduced run")
    ax.set_xlabel("Augmentation block (252 trading days each)")
    ax.set_ylabel("MAPE on real prices")
    ax.grid(alpha=0.24, linewidth=0.7)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "tatr_prices_author_style_runs.png", dpi=220)
    fig.savefig(out / "tatr_prices_author_style_runs.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    change = (avg - avg[0]) / avg[0] * 100
    ax.bar(x_blocks, change, color="#1f77b4", alpha=0.84)
    ax.axhline(0, color="0.25", linewidth=1)
    ax.set_title("TATR price MAPE change vs no augmentation")
    ax.set_xlabel("Augmentation block (252 trading days each)")
    ax.set_ylabel("Avg MAPE change (%)")
    ax.grid(axis="y", alpha=0.24, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out / "tatr_prices_author_style_relative_change.png", dpi=220)
    fig.savefig(out / "tatr_prices_author_style_relative_change.pdf")
    plt.close(fig)

    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_title("TATR price MAPE by run")
    ax.set_xlabel("Augmentation block")
    ax.set_ylabel("Run")
    ax.set_xticks(np.arange(len(x_blocks)))
    ax.set_xticklabels(x_blocks)
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels(np.arange(1, values.shape[0] + 1))
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("MAPE")
    fig.tight_layout()
    fig.savefig(out / "tatr_prices_author_style_heatmap.png", dpi=220)
    fig.savefig(out / "tatr_prices_author_style_heatmap.pdf")
    plt.close(fig)

    pd.DataFrame(
        {
            "augmentation_idx": x_blocks,
            "synthetic_days": x_days,
            "avg": avg,
            "min": lo,
            "max": hi,
            "pct_change_vs_no_aug": change,
        }
    ).to_csv(out / "tatr_prices_author_style_summary_with_change.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TATR-only author-style SP500 price replication using stored reference artifacts."
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--augmentations", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--aug-length", type=int, default=252)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--ahead", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--loss", default="mae", choices=["mae", "mse"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports/generated_outputs/01_sp500_downstream_replication/03_sp500_tatr_author_style_1day")
    args = parser.parse_args()

    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(REF)

    from experiments.predictor_lstm import separate_train_lstm_predictor, test_on_real
    from experiments.utils_downstream import (
        Timeseries2Dataset_Downstream,
        concat_datasets_downstream,
        get_downstream_data,
        init_first_segment,
    )
    from models.load_models import load_ftsdiffusion
    from models.model_params import prm_params
    from models.sampling import segment_generation_ftsdiffusion, state_evolution_ftsdiffusion
    from models.utils_sampling import sampling_inputs

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    start = time.time()

    downstream_timeseries, segments_test, labels_test, lengths_test = get_downstream_data()
    init_dataset, scaler = Timeseries2Dataset_Downstream(downstream_timeseries[: 252 * 5], args.window_size)
    test_dataset = Timeseries2Dataset_Downstream(downstream_timeseries[252 * 5 :], args.window_size, scaler)
    init_state, init_segment = init_first_segment(segments_test, labels_test, lengths_test)

    model = load_ftsdiffusion()
    _, _, patterns = sampling_inputs()
    l_min = prm_params["l_min"]

    def generate_cached(length: int) -> np.ndarray:
        state = init_state.clone().detach()
        timeseries = list(init_segment)
        curr_t = len(timeseries)
        while curr_t < length:
            state = state_evolution_ftsdiffusion(model["evolution"], state, l_min)
            segment = segment_generation_ftsdiffusion(model["generation"], state, patterns)
            segment = segment - segment[0] + timeseries[-1]
            timeseries.extend(segment)
            curr_t += len(segment)
        return np.asarray(timeseries[:length])

    errors = np.zeros((args.runs, args.augmentations + 1), dtype=np.float64)
    for run in range(args.runs):
        curr_dataset = init_dataset.clone().detach()
        for aug_idx in range(args.augmentations + 1):
            if aug_idx > 0:
                syn_timeseries = generate_cached(args.aug_length)
                syn_dataset = Timeseries2Dataset_Downstream(syn_timeseries, args.window_size, scaler)
                curr_dataset = concat_datasets_downstream(curr_dataset, syn_dataset)
            predictor = separate_train_lstm_predictor(
                args.epochs,
                curr_dataset,
                input_dim=1,
                hidden_dim=args.hidden_dim,
                output_dim=args.ahead,
                n_layers=2,
                criterion=args.loss,
                verbose=False,
            )
            errors[run, aug_idx] = test_on_real(predictor, test_dataset, scaler, criterion="mape")
            pd.DataFrame(errors).to_csv(out / "tatr_prices_author_style_matrix.csv", index=False)
            print(
                f"[TATR prices author-style] run={run + 1}/{args.runs} "
                f"aug={aug_idx}/{args.augmentations} mape={errors[run, aug_idx]:.7f}",
                flush=True,
            )

    matrix = pd.DataFrame(errors)
    summary = summarize_authors_style(errors)
    summary.insert(0, "augmentation_idx", np.arange(args.augmentations + 1))
    matrix.to_csv(out / "tatr_prices_author_style_matrix.csv", index=False)
    summary.to_csv(out / "tatr_prices_author_style_summary.csv", index=False)
    save_tatr_plots(out, matrix, summary, args.aug_length)

    metadata = {
        "asset": "sp500",
        "experiment": "TATR",
        "datatype": "prices",
        "reference_dir": str(REF.relative_to(ROOT)),
        "output_dir": str(out.relative_to(ROOT)),
        "used_stored_sisc_artifacts": True,
        "used_stored_pgm_pem_checkpoints": True,
        "preserved_author_settings": {
            "train_test_split": "reference prepare_segments default 0.8",
            "tatr_initial_real_period": "252*5 prices",
            "aug_length": args.aug_length,
            "n_epochs": args.epochs,
            "window_size": args.window_size,
            "ahead": args.ahead,
            "hidden_dim": args.hidden_dim,
            "lstm_layers": 2,
            "loss": args.loss,
            "optimizer": "Adam lr=1e-2 from reference predictor_lstm.py",
            "summary": "reference trimmed mean/min/max formula with small-n guard",
        },
        "resource_compromises": {
            "runs": f"{args.runs} instead of paper/reference 100",
            "augmentations": f"{args.augmentations} instead of paper/reference 100",
            "model_loading": "PGM/PEM loaded once and reused; generation math unchanged",
        },
        "elapsed_seconds": round(time.time() - start, 3),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    best = summary.loc[summary["avg"].idxmin()].to_dict()
    print(json.dumps({"metadata": metadata, "best": best}, indent=2), flush=True)


if __name__ == "__main__":
    main()
