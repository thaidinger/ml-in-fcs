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
REPORTS = ROOT / "reports"

sys.path.insert(0, str(REF))


def pct_returns(series: np.ndarray) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    return (series[1:] - series[:-1]) / series[:-1]


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


def plot_summary(df: pd.DataFrame, x: np.ndarray, xlabel: str, title: str, path_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(x, df["avg"], color="#1f77b4", marker="o", linewidth=1.8, markersize=4)
    ax.fill_between(x, df["min"], df["max"], color="#1f77b4", alpha=0.18)
    ax.axhline(df["avg"].iloc[0], color="0.45", linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MAPE")
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduced author-style SP500 TMTR/TATR replication using stored reference checkpoints."
    )
    parser.add_argument("--datatype", default="returns", choices=["prices", "returns"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--tatr-augmentations", type=int, default=10)
    parser.add_argument("--tmtr-proportions", type=int, default=10)
    parser.add_argument("--aug-length", type=int, default=252)
    parser.add_argument("--mix-length", type=int, default=252 * 5)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--ahead", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--loss", default="mae", choices=["mae", "mse"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    out = ROOT / args.output if args.output else REPORTS / f"downstream_{args.datatype}_sp500"

    os.chdir(REF)

    from experiments.predictor_lstm import separate_train_lstm_predictor, test_on_real
    from experiments.utils_downstream import (
        Timeseries2Dataset_Downstream,
        concat_datasets_downstream,
        create_mix_dataset,
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

    out.mkdir(parents=True, exist_ok=True)
    start = time.time()

    downstream_ts, segments_test, labels_test, lengths_test = get_downstream_data()
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

    def transform(series: np.ndarray) -> np.ndarray:
        if args.datatype == "returns":
            return pct_returns(series)
        return np.asarray(series, dtype=np.float64).reshape(-1)

    # TATR: authors' 80/20 downstream split and 5-year initial real period.
    init_prices = downstream_ts[: 252 * 5]
    test_prices = downstream_ts[252 * 5 :]
    init_series = transform(init_prices)
    test_series = transform(test_prices)
    init_dataset, tatr_scaler = Timeseries2Dataset_Downstream(init_series, args.window_size)
    test_dataset = Timeseries2Dataset_Downstream(test_series, args.window_size, tatr_scaler)

    tatr_errors = np.zeros((args.runs, args.tatr_augmentations + 1), dtype=np.float64)
    for run in range(args.runs):
        curr_dataset = init_dataset.clone().detach()
        for aug_idx in range(args.tatr_augmentations + 1):
            if aug_idx > 0:
                syn_prices = generate_cached(args.aug_length)
                syn_series = transform(syn_prices)
                syn_dataset = Timeseries2Dataset_Downstream(syn_series, args.window_size, tatr_scaler)
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
            tatr_errors[run, aug_idx] = test_on_real(predictor, test_dataset, tatr_scaler, criterion="mape")
            pd.DataFrame(tatr_errors).to_csv(out / f"tatr_{args.datatype}_matrix.csv", index=False)
            print(
                f"[TATR {args.datatype}] run={run + 1}/{args.runs} aug={aug_idx}/{args.tatr_augmentations} "
                f"mape={tatr_errors[run, aug_idx]:.6g}",
                flush=True,
            )

    tatr_summary = summarize_authors_style(tatr_errors)
    tatr_summary.insert(0, "augmentation_idx", np.arange(args.tatr_augmentations + 1))
    tatr_summary.to_csv(out / f"tatr_{args.datatype}_summary_authors_style.csv", index=False)
    tatr_x = np.arange(args.tatr_augmentations + 1) * (
        args.aug_length - 1 if args.datatype == "returns" else args.aug_length
    )
    plot_summary(
        tatr_summary,
        tatr_x,
        f"Synthetic {args.datatype} observations added",
        f"SP500 TATR on {args.datatype.title()} (reduced author-style run)",
        out / f"tatr_{args.datatype}",
    )

    # TMTR: authors' mixture setup.
    real_prices = downstream_ts[: args.mix_length]
    tmtr_test_prices = downstream_ts[args.mix_length :]
    real_series = transform(real_prices)
    tmtr_test_series = transform(tmtr_test_prices)
    _, tmtr_scaler = Timeseries2Dataset_Downstream(tmtr_test_series, args.window_size)
    tmtr_test_dataset = Timeseries2Dataset_Downstream(tmtr_test_series, args.window_size, tmtr_scaler)
    mix_series_length = len(real_series)
    proportions = np.linspace(0, 100, args.tmtr_proportions + 1, dtype=int)

    tmtr_errors = np.zeros((args.runs, len(proportions)), dtype=np.float64)
    for run in range(args.runs):
        syn_prices = generate_cached(args.mix_length)
        syn_series = transform(syn_prices)
        for prop_idx, proportion_pct in enumerate(proportions):
            proportion = proportion_pct / 100
            mix_dataset = None
            if proportion == 0:
                mix_dataset = create_mix_dataset(real_series, mix_series_length, args.window_size, tmtr_scaler)
            elif proportion == 1:
                mix_dataset = create_mix_dataset(syn_series, mix_series_length, args.window_size, tmtr_scaler)
            else:
                syn_len = int(mix_series_length * proportion)
                real_len = mix_series_length - syn_len
                mix_real = create_mix_dataset(real_series, real_len, args.window_size, tmtr_scaler)
                mix_syn = create_mix_dataset(syn_series, syn_len, args.window_size, tmtr_scaler)
                mix_dataset = concat_datasets_downstream(mix_real, mix_syn)
            predictor = separate_train_lstm_predictor(
                args.epochs,
                mix_dataset,
                input_dim=1,
                hidden_dim=args.hidden_dim,
                output_dim=args.ahead,
                n_layers=2,
                criterion=args.loss,
                verbose=False,
            )
            tmtr_errors[run, prop_idx] = test_on_real(
                predictor, tmtr_test_dataset, tmtr_scaler, criterion="mape"
            )
            pd.DataFrame(tmtr_errors, columns=proportions).to_csv(out / f"tmtr_{args.datatype}_matrix.csv", index=False)
            print(
                f"[TMTR {args.datatype}] run={run + 1}/{args.runs} prop={proportion_pct}% "
                f"mape={tmtr_errors[run, prop_idx]:.6g}",
                flush=True,
            )

    tmtr_summary = summarize_authors_style(tmtr_errors)
    tmtr_summary.insert(0, "synthetic_proportion_pct", proportions)
    tmtr_summary.to_csv(out / f"tmtr_{args.datatype}_summary_authors_style.csv", index=False)
    plot_summary(
        tmtr_summary,
        proportions,
        "Synthetic proportion (%)",
        f"SP500 TMTR on {args.datatype.title()} (reduced author-style run)",
        out / f"tmtr_{args.datatype}",
    )

    metadata = {
        "asset": "sp500",
        "datatype": args.datatype,
        "reference_dir": str(REF.relative_to(ROOT)),
        "output_dir": str(out.relative_to(ROOT)),
        "used_stored_sisc_artifacts": True,
        "used_stored_pgm_pem_checkpoints": True,
        "author_style_settings_preserved": {
            "train_test_split": "reference prepare_segments default 0.8",
            "tatr_initial_real_period": f"252*5 prices, evaluated as {args.datatype}",
            "window_size": args.window_size,
            "ahead": args.ahead,
            "lstm_layers": 2,
            "lstm_optimizer": "Adam lr=1e-2 from reference predictor_lstm.py",
            "summary": "reference trimmed mean/min/max formula with small-n guard",
        },
        "downscaled_settings": vars(args),
        "elapsed_seconds": round(time.time() - start, 3),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
