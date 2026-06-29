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


def ci95(values: pd.Series) -> float:
    n = int(values.count())
    if n <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1) / np.sqrt(n))


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-run TMTR confidence-band comparison for folder 17.")
    parser.add_argument("--input", default="reports/generated_outputs/17_sampled_rollout_tatr_smoke/sampled_rollout_prices.csv")
    parser.add_argument("--output", default="reports/generated_outputs/17_sampled_rollout_tatr_smoke")
    parser.add_argument("--name", default="tmtr_multirun_100epoch")
    parser.add_argument("--proportions", type=parse_ints, default=parse_ints("0,10,20,30,40,50,60,70,80,90,100"))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mix-length", type=int, default=252 * 5)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--ahead", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--loss", default="mae", choices=["mae", "mse"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
    os.chdir(REF)

    from experiments.predictor_lstm import separate_train_lstm_predictor, test_on_real
    from experiments.utils_downstream import Timeseries2Dataset_Downstream, concat_datasets_downstream, get_downstream_data

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    started = time.time()

    price_df = pd.read_csv(ROOT / args.input)
    protocols = ["independent_argmax", "continuous_argmax", "independent_sampled", "continuous_sampled"]

    downstream_timeseries, _, _, _ = get_downstream_data()
    real_timeseries = np.asarray(downstream_timeseries[: args.mix_length], dtype=np.float64)
    test_timeseries = np.asarray(downstream_timeseries[args.mix_length :], dtype=np.float64)
    _, scaler = Timeseries2Dataset_Downstream(test_timeseries, args.window_size)
    test_dataset = Timeseries2Dataset_Downstream(test_timeseries, args.window_size, scaler)
    real_dataset = Timeseries2Dataset_Downstream(real_timeseries, args.window_size, scaler)

    protocol_cache = {}
    for protocol in protocols:
        sub = price_df[price_df["protocol_family"].eq(protocol)].sort_values(["path", "t"])
        if sub.empty:
            raise ValueError(f"missing protocol_family in price file: {protocol}")
        protocol_cache[protocol] = sub["price"].to_numpy(dtype=np.float64)

    def sample_window(series: np.ndarray, length: int, seed: int) -> np.ndarray:
        if len(series) <= length:
            return series[:length]
        rng = np.random.default_rng(seed)
        start_idx = int(rng.integers(0, len(series) - length + 1))
        return series[start_idx : start_idx + length]

    def make_mix_dataset(protocol: str, proportion_pct: int, seed: int) -> torch.Tensor:
        if proportion_pct == 0:
            return real_dataset.clone().detach()
        if proportion_pct == 100:
            syn = sample_window(protocol_cache[protocol], args.mix_length, seed)
            return Timeseries2Dataset_Downstream(syn, args.window_size, scaler)

        syn_length = int(args.mix_length * proportion_pct / 100)
        real_length = args.mix_length - syn_length
        real_part = sample_window(real_timeseries, real_length, seed + 1_000)
        syn_part = sample_window(protocol_cache[protocol], syn_length, seed + 2_000)
        real_part_dataset = Timeseries2Dataset_Downstream(real_part, args.window_size, scaler)
        syn_part_dataset = Timeseries2Dataset_Downstream(syn_part, args.window_size, scaler)
        return concat_datasets_downstream(real_part_dataset, syn_part_dataset)

    def train_eval(dataset: torch.Tensor, seed: int) -> float:
        set_seed(seed)
        predictor = separate_train_lstm_predictor(
            args.epochs,
            dataset,
            input_dim=1,
            hidden_dim=args.hidden_dim,
            output_dim=args.ahead,
            n_layers=2,
            criterion=args.loss,
            verbose=False,
        )
        return float(test_on_real(predictor, test_dataset, scaler, criterion="mape"))

    rows: list[dict] = []
    for run in range(args.runs):
        run_seed = args.seed + run
        t0 = time.time()
        baseline_mape = train_eval(real_dataset, run_seed + 100_000)
        rows.append(
            {
                "run": run,
                "protocol": "real_only",
                "synthetic_proportion_pct": 0,
                "train_windows": int(real_dataset.shape[0]),
                "mape": baseline_mape,
                "baseline_mape": baseline_mape,
                "pct_change_vs_real_only": 0.0,
                "fit_seconds": round(time.time() - t0, 3),
            }
        )
        pd.DataFrame(rows).to_csv(out / f"{args.name}_long.csv", index=False)
        print(f"[tmtr-multirun] run={run} real_only mape={baseline_mape:.7f}", flush=True)

        for protocol in protocols:
            for prop in [p for p in args.proportions if p != 0]:
                dataset = make_mix_dataset(protocol, prop, run_seed + prop + sum(ord(ch) for ch in protocol))
                fit_seed = run_seed + 200_000 + prop + sum(ord(ch) for ch in protocol)
                t1 = time.time()
                mape = train_eval(dataset, fit_seed)
                row = {
                    "run": run,
                    "protocol": protocol,
                    "synthetic_proportion_pct": prop,
                    "train_windows": int(dataset.shape[0]),
                    "mape": mape,
                    "baseline_mape": baseline_mape,
                    "pct_change_vs_real_only": (mape - baseline_mape) / baseline_mape * 100,
                    "fit_seconds": round(time.time() - t1, 3),
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(out / f"{args.name}_long.csv", index=False)
                print(
                    f"[tmtr-multirun] run={run} {protocol} prop={prop}% "
                    f"mape={mape:.7f} change={row['pct_change_vs_real_only']:.1f}% "
                    f"fit={row['fit_seconds']:.1f}s",
                    flush=True,
                )

    long = pd.DataFrame(rows)
    long.to_csv(out / f"{args.name}_long.csv", index=False)

    curve = (
        long.groupby(["protocol", "synthetic_proportion_pct"], sort=False)
        .agg(
            n=("mape", "count"),
            mape_mean=("mape", "mean"),
            mape_std=("mape", "std"),
            mape_ci95=("mape", ci95),
            pct_mean=("pct_change_vs_real_only", "mean"),
            pct_std=("pct_change_vs_real_only", "std"),
            pct_ci95=("pct_change_vs_real_only", ci95),
        )
        .reset_index()
    )
    curve.to_csv(out / f"{args.name}_curve_summary.csv", index=False)

    final_rows = []
    for protocol, group in curve[curve["protocol"].ne("real_only")].groupby("protocol", sort=False):
        best = group.loc[group["mape_mean"].idxmin()]
        final = group[group["synthetic_proportion_pct"].eq(100)].iloc[0]
        final_rows.append(
            {
                "protocol": protocol,
                "runs": args.runs,
                "best_proportion_pct": int(best["synthetic_proportion_pct"]),
                "best_mape_mean": float(best["mape_mean"]),
                "best_pct_mean": float(best["pct_mean"]),
                "best_pct_ci95": float(best["pct_ci95"]),
                "final_mape_mean": float(final["mape_mean"]),
                "final_pct_mean": float(final["pct_mean"]),
                "final_pct_ci95": float(final["pct_ci95"]),
            }
        )
    final_summary = pd.DataFrame(final_rows)
    final_summary.to_csv(out / f"{args.name}_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for protocol, group in curve[curve["protocol"].ne("real_only")].groupby("protocol", sort=False):
        group = group.sort_values("synthetic_proportion_pct")
        x = group["synthetic_proportion_pct"].to_numpy(dtype=float)
        y = group["mape_mean"].to_numpy(dtype=float)
        band = group["mape_ci95"].fillna(0).to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2, label=protocol)
        ax.fill_between(x, y - band, y + band, alpha=0.16)
    real_mean = float(long[long["protocol"].eq("real_only")]["mape"].mean())
    ax.axhline(real_mean, color="0.25", linestyle="--", linewidth=1, label="real-only mean")
    ax.set_title(f"S&P 500 TMTR multi-run MAPE ({args.runs} runs, {args.epochs} epochs)")
    ax.set_xlabel("Synthetic proportion (%)")
    ax.set_ylabel("MAPE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{args.name}_mape_ci.png", dpi=220)
    fig.savefig(out / f"{args.name}_mape_ci.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for protocol, group in curve[curve["protocol"].ne("real_only")].groupby("protocol", sort=False):
        group = group.sort_values("synthetic_proportion_pct")
        x = group["synthetic_proportion_pct"].to_numpy(dtype=float)
        y = group["pct_mean"].to_numpy(dtype=float)
        band = group["pct_ci95"].fillna(0).to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2, label=protocol)
        ax.fill_between(x, y - band, y + band, alpha=0.16)
    ax.axhline(0, color="0.25", linewidth=1)
    ax.set_title(f"S&P 500 TMTR multi-run change vs real-only ({args.runs} runs, {args.epochs} epochs)")
    ax.set_xlabel("Synthetic proportion (%)")
    ax.set_ylabel("MAPE change vs same-run real-only (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{args.name}_pct_change_ci.png", dpi=220)
    fig.savefig(out / f"{args.name}_pct_change_ci.pdf")
    plt.close(fig)

    metadata = {
        "asset": "sp500",
        "purpose": "multi-run TMTR confidence interval band for sampled-rollout diagnostics",
        "settings": vars(args),
        "elapsed_seconds": round(time.time() - started, 3),
        "notes": "Bands are 95% normal-approximation confidence intervals over runs.",
    }
    (out / f"{args.name}_metadata.json").write_text(json.dumps(metadata, indent=2))

    readme = f"""# S&P 500 TMTR Multi-Run Confidence Bands

This run repeats the folder-17 TMTR comparison across `{args.runs}` runs with `{args.epochs}` LSTM epochs.

Important difference from the earlier smoke plots: the real-only baseline is trained once per run and shared across protocols, so `pct_change_vs_real_only` compares each synthetic mixture to the same run's real-only baseline.

Main files:

- `{args.name}_mape_ci.png`
- `{args.name}_pct_change_ci.png`
- `{args.name}_curve_summary.csv`
- `{args.name}_summary.csv`
- `{args.name}_long.csv`

Bands are 95% normal-approximation confidence intervals over runs.
"""
    (out / f"{args.name}_README.md").write_text(readme)

    print(json.dumps({"out": str(out.relative_to(ROOT)), "summary": final_summary.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
