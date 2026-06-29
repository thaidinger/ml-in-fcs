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


def parse_ints(value: str) -> list[int]:
    vals = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not vals or vals != sorted(set(vals)) or vals[0] != 0 or vals[-1] != 100:
        raise argparse.ArgumentTypeError("values must be sorted, unique, and span 0..100")
    return vals


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="TMTR smoke plot for folder 17 sampled-rollout diagnostics.")
    parser.add_argument("--input", default="reports/generated_outputs/17_sampled_rollout_tatr_smoke/sampled_rollout_prices.csv")
    parser.add_argument("--output", default="reports/generated_outputs/17_sampled_rollout_tatr_smoke")
    parser.add_argument("--proportions", type=parse_ints, default=parse_ints("0,25,50,75,100"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--mix-length", type=int, default=252 * 5)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--ahead", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--loss", default="mae", choices=["mae", "mse"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="tmtr_smoke", help="Output filename prefix.")
    args = parser.parse_args()

    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
    os.chdir(REF)

    from experiments.predictor_lstm import separate_train_lstm_predictor, test_on_real
    from experiments.utils_downstream import Timeseries2Dataset_Downstream, concat_datasets_downstream, get_downstream_data

    set_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    start = time.time()

    price_df = pd.read_csv(ROOT / args.input)
    protocol_map = {
        "independent_argmax": "independent_argmax",
        "continuous_argmax": "continuous_argmax",
        "independent_sampled": "independent_sampled",
        "continuous_sampled": "continuous_sampled",
    }

    downstream_timeseries, _, _, _ = get_downstream_data()
    real_timeseries = np.asarray(downstream_timeseries[: args.mix_length], dtype=np.float64)
    test_timeseries = np.asarray(downstream_timeseries[args.mix_length :], dtype=np.float64)
    _, scaler = Timeseries2Dataset_Downstream(test_timeseries, args.window_size)
    test_dataset = Timeseries2Dataset_Downstream(test_timeseries, args.window_size, scaler)

    def protocol_series(protocol_family: str) -> np.ndarray:
        sub = price_df[price_df["protocol_family"].eq(protocol_family)].copy()
        if sub.empty:
            raise ValueError(f"missing protocol_family in price file: {protocol_family}")
        sub = sub.sort_values(["path", "t"])
        return sub["price"].to_numpy(dtype=np.float64)

    def sample_window(series: np.ndarray, length: int, seed: int) -> np.ndarray:
        if len(series) <= length:
            return series[:length]
        rng = np.random.default_rng(seed)
        start_idx = int(rng.integers(0, len(series) - length + 1))
        return series[start_idx : start_idx + length]

    def make_mix_dataset(protocol_family: str, proportion_pct: int, seed: int) -> torch.Tensor:
        if proportion_pct == 0:
            return Timeseries2Dataset_Downstream(real_timeseries, args.window_size, scaler)
        if proportion_pct == 100:
            syn = sample_window(protocol_series(protocol_family), args.mix_length, seed)
            return Timeseries2Dataset_Downstream(syn, args.window_size, scaler)

        syn_length = int(args.mix_length * proportion_pct / 100)
        real_length = args.mix_length - syn_length
        real_part = sample_window(real_timeseries, real_length, seed + 1_000)
        syn_part = sample_window(protocol_series(protocol_family), syn_length, seed + 2_000)
        real_dataset = Timeseries2Dataset_Downstream(real_part, args.window_size, scaler)
        syn_dataset = Timeseries2Dataset_Downstream(syn_part, args.window_size, scaler)
        return concat_datasets_downstream(real_dataset, syn_dataset)

    rows: list[dict] = []
    for label, protocol_family in protocol_map.items():
        for prop in args.proportions:
            train_dataset = make_mix_dataset(protocol_family, prop, args.seed + prop + sum(ord(ch) for ch in label))
            set_seed(args.seed + 100_000 + prop + sum(ord(ch) for ch in label))
            t0 = time.time()
            predictor = separate_train_lstm_predictor(
                args.epochs,
                train_dataset,
                input_dim=1,
                hidden_dim=args.hidden_dim,
                output_dim=args.ahead,
                n_layers=2,
                criterion=args.loss,
                verbose=False,
            )
            mape = test_on_real(predictor, test_dataset, scaler, criterion="mape")
            fit_seconds = time.time() - t0
            row = {
                "protocol": label,
                "synthetic_proportion_pct": prop,
                "train_windows": int(train_dataset.shape[0]),
                "mape": float(mape),
                "fit_seconds": round(fit_seconds, 3),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(out / f"{args.name}_long.csv", index=False)
            print(
                f"[tmtr-smoke] {label} prop={prop}% windows={train_dataset.shape[0]} "
                f"mape={mape:.7f} fit={fit_seconds:.1f}s",
                flush=True,
            )

    results = pd.DataFrame(rows)
    baseline = results[results["synthetic_proportion_pct"].eq(0)].set_index("protocol")["mape"].to_dict()
    results["pct_change_vs_real_only"] = [
        (row.mape - baseline[row.protocol]) / baseline[row.protocol] * 100 for row in results.itertuples()
    ]
    results.to_csv(out / f"{args.name}_long.csv", index=False)

    summary_rows = []
    for protocol, group in results.groupby("protocol", sort=False):
        group = group.sort_values("synthetic_proportion_pct")
        best = group.loc[group["mape"].idxmin()]
        final = group.iloc[-1]
        summary_rows.append(
            {
                "protocol": protocol,
                "baseline_mape": float(group.iloc[0]["mape"]),
                "best_proportion_pct": int(best["synthetic_proportion_pct"]),
                "best_mape": float(best["mape"]),
                "best_pct_vs_real_only": float(best["pct_change_vs_real_only"]),
                "final_mape_100pct": float(final["mape"]),
                "final_pct_vs_real_only": float(final["pct_change_vs_real_only"]),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / f"{args.name}_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for protocol, group in results.groupby("protocol", sort=False):
        group = group.sort_values("synthetic_proportion_pct")
        ax.plot(group["synthetic_proportion_pct"], group["mape"], marker="o", linewidth=2, label=protocol)
    ax.set_title(f"S&P 500 TMTR smoke: argmax vs sampled PEM ({args.epochs} epochs)")
    ax.set_xlabel("Synthetic proportion (%)")
    ax.set_ylabel("MAPE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{args.name}_mape.png", dpi=220)
    fig.savefig(out / f"{args.name}_mape.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for protocol, group in results.groupby("protocol", sort=False):
        group = group.sort_values("synthetic_proportion_pct")
        ax.plot(
            group["synthetic_proportion_pct"],
            group["pct_change_vs_real_only"],
            marker="o",
            linewidth=2,
            label=protocol,
        )
    ax.axhline(0, color="0.25", linewidth=1)
    ax.set_title(f"S&P 500 TMTR smoke change vs real-only ({args.epochs} epochs)")
    ax.set_xlabel("Synthetic proportion (%)")
    ax.set_ylabel("MAPE change vs real-only (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{args.name}_pct_change.png", dpi=220)
    fig.savefig(out / f"{args.name}_pct_change.pdf")
    plt.close(fig)

    metadata = {
        "asset": "sp500",
        "purpose": "TMTR smoke plot using folder 17 argmax/sampled rollout price paths",
        "settings": vars(args),
        "elapsed_seconds": round(time.time() - start, 3),
    }
    (out / f"{args.name}_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(json.dumps({"out": str(out.relative_to(ROOT)), "summary": summary.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
