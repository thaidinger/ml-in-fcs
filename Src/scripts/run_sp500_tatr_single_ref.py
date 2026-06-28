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


ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "fts-diffusion-ref"

sys.path.insert(0, str(REF))


def parse_eval_blocks(value: str) -> list[int]:
    blocks = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not blocks or blocks[0] != 0:
        raise argparse.ArgumentTypeError("eval blocks must be a comma-separated list starting with 0")
    if sorted(set(blocks)) != blocks:
        raise argparse.ArgumentTypeError("eval blocks must be sorted and unique")
    return blocks


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
    x_blocks = summary["augmentation_blocks"].to_numpy()
    x_days = summary["synthetic_days"].to_numpy()
    avg = summary["avg"].to_numpy()
    lo = summary["min"].to_numpy()
    hi = summary["max"].to_numpy()
    best_idx = int(np.argmin(avg))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for _, row in matrix.iterrows():
        ax.plot(x_blocks, row.to_numpy(dtype=float), color="#9aa7b2", alpha=0.42, linewidth=1)
    ax.fill_between(x_blocks, lo, hi, color="#2b6cb0", alpha=0.16, label="run band")
    ax.plot(x_blocks, avg, color="#2b6cb0", marker="o", linewidth=2.2, label="mean MAPE")
    ax.scatter([x_blocks[best_idx]], [avg[best_idx]], color="#c2410c", s=58, zorder=5, label="best")
    ax.axhline(avg[0], color="0.35", linestyle="--", linewidth=1, label="no augmentation")
    ax.set_title("SP500 TATR on Prices, single continuous trajectory")
    ax.set_xlabel("Synthetic blocks added (252 trading days each)")
    ax.set_ylabel("MAPE on real prices")
    ax.grid(alpha=0.24, linewidth=0.7)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "tatr_prices_single_runs.png", dpi=220)
    fig.savefig(out / "tatr_prices_single_runs.pdf")
    plt.close(fig)

    change = (avg - avg[0]) / avg[0] * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    colors = np.where(change <= 0, "#2f855a", "#c2410c")
    ax.bar(x_blocks, change, color=colors, alpha=0.86, width=6.5)
    ax.axhline(0, color="0.25", linewidth=1)
    ax.set_title("TATR single-protocol MAPE change vs no augmentation")
    ax.set_xlabel("Synthetic blocks added")
    ax.set_ylabel("Mean MAPE change (%)")
    ax.grid(axis="y", alpha=0.24, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out / "tatr_prices_single_relative_change.png", dpi=220)
    fig.savefig(out / "tatr_prices_single_relative_change.pdf")
    plt.close(fig)

    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_title("TATR single-protocol MAPE by run")
    ax.set_xlabel("Synthetic blocks added")
    ax.set_ylabel("Run")
    ax.set_xticks(np.arange(len(x_blocks)))
    ax.set_xticklabels(x_blocks)
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels(np.arange(1, values.shape[0] + 1))
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("MAPE")
    fig.tight_layout()
    fig.savefig(out / "tatr_prices_single_heatmap.png", dpi=220)
    fig.savefig(out / "tatr_prices_single_heatmap.pdf")
    plt.close(fig)

    pd.DataFrame(
        {
            "augmentation_blocks": x_blocks,
            "synthetic_days": x_days,
            "avg": avg,
            "min": lo,
            "max": hi,
            "pct_change_vs_no_aug": change,
        }
    ).to_csv(out / "tatr_prices_single_summary_with_change.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SP500 TATR replication with the non-author single continuous trajectory protocol."
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--eval-blocks", type=parse_eval_blocks, default=parse_eval_blocks("0,10,20,30,40,50,60,70,80,90,100"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--aug-length", type=int, default=252)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--ahead", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--loss", default="mae", choices=["mae", "mse"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports/generated_outputs/02_sp500_tatr_single_diagnostic")
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
    max_blocks = max(args.eval_blocks)
    max_length = max_blocks * args.aug_length

    def generate_continuous(length: int) -> np.ndarray:
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

    errors = np.zeros((args.runs, len(args.eval_blocks)), dtype=np.float64)
    synthetic_stats: list[dict[str, float | int]] = []
    for run in range(args.runs):
        run_seed = args.seed + run
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        synthetic_prices = generate_continuous(max_length) if max_length > 0 else np.asarray([])
        if synthetic_prices.size:
            synthetic_stats.append(
                {
                    "run": run,
                    "n_days": int(synthetic_prices.size),
                    "first": float(synthetic_prices[0]),
                    "last": float(synthetic_prices[-1]),
                    "min": float(np.min(synthetic_prices)),
                    "max": float(np.max(synthetic_prices)),
                    "mean": float(np.mean(synthetic_prices)),
                    "std": float(np.std(synthetic_prices)),
                }
            )

        for col, blocks in enumerate(args.eval_blocks):
            if blocks == 0:
                curr_dataset = init_dataset.clone().detach()
            else:
                prefix = synthetic_prices[: blocks * args.aug_length]
                syn_dataset = Timeseries2Dataset_Downstream(prefix, args.window_size, scaler)
                curr_dataset = concat_datasets_downstream(init_dataset.clone().detach(), syn_dataset)
            lstm_seed = args.seed + 100_000 + run * (max_blocks + 1) + blocks
            random.seed(lstm_seed)
            np.random.seed(lstm_seed)
            torch.manual_seed(lstm_seed)
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
            errors[run, col] = test_on_real(predictor, test_dataset, scaler, criterion="mape")
            pd.DataFrame(errors, columns=args.eval_blocks).to_csv(out / "tatr_prices_single_matrix.csv", index=False)
            print(
                f"[TATR prices single] run={run + 1}/{args.runs} "
                f"blocks={blocks}/{max_blocks} mape={errors[run, col]:.7f}",
                flush=True,
            )

    matrix = pd.DataFrame(errors, columns=args.eval_blocks)
    summary = summarize_authors_style(errors)
    summary.insert(0, "augmentation_blocks", args.eval_blocks)
    summary.insert(1, "synthetic_days", np.asarray(args.eval_blocks) * args.aug_length)
    matrix.to_csv(out / "tatr_prices_single_matrix.csv", index=False)
    summary.to_csv(out / "tatr_prices_single_summary.csv", index=False)
    pd.DataFrame(synthetic_stats).to_csv(out / "tatr_prices_single_synthetic_stats.csv", index=False)
    save_tatr_plots(out, matrix, summary, args.aug_length)

    best = summary.loc[summary["avg"].idxmin()].to_dict()
    metadata = {
        "asset": "sp500",
        "experiment": "TATR",
        "protocol": "single_continuous_trajectory",
        "datatype": "prices",
        "reference_dir": str(REF.relative_to(ROOT)),
        "output_dir": str(out.relative_to(ROOT)),
        "used_stored_sisc_artifacts": True,
        "used_stored_pgm_pem_checkpoints": True,
        "paper_settings_kept": {
            "train_test_split": "reference prepare_segments default 0.8",
            "tatr_initial_real_period": "252*5 prices",
            "window_size": args.window_size,
            "ahead": args.ahead,
            "hidden_dim": args.hidden_dim,
            "lstm_layers": 2,
            "loss": args.loss,
            "n_epochs": args.epochs,
            "optimizer": "Adam lr=1e-2 from reference predictor_lstm.py",
        },
        "protocol_change": (
            "Synthetic augmentation is one continuous trajectory per run and each evaluation uses a prefix. "
            "This differs from the reference authors protocol, which generates independent 252-day blocks."
        ),
        "resource_compromises": {
            "runs": args.runs,
            "eval_blocks": args.eval_blocks,
            "model_loading": "PGM/PEM loaded once and reused; generation math unchanged",
        },
        "elapsed_seconds": round(time.time() - start, 3),
        "best": best,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out / "README.md").write_text(
        "# SP500 TATR Single-Protocol Run\n\n"
        "This run keeps the paper downstream settings but switches the augmentation protocol to a "
        "single continuous FTS-Diffusion trajectory per run. Each column trains on the same real "
        "initial set plus a longer prefix of that trajectory.\n\n"
        "This is not the released reference-code TATR protocol; it is the diagnostic `single` setting "
        "used to test whether the paper-like downward trend appears when synthetic prices are allowed "
        "to drift continuously.\n"
    )

    print(json.dumps({"metadata": metadata, "best": best}, indent=2), flush=True)


if __name__ == "__main__":
    main()
