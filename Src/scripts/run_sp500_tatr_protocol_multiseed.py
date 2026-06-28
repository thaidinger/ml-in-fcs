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


def parse_ints(value: str) -> list[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return values


def parse_blocks(value: str) -> list[int]:
    blocks = parse_ints(value)
    if blocks[0] != 0 or blocks != sorted(set(blocks)):
        raise argparse.ArgumentTypeError("blocks must be sorted, unique, and start with 0")
    return blocks


def stable_protocol_offset(protocol: str) -> int:
    offsets = {
        "continuous_chunked": 11_000,
        "independent_fixed": 22_000,
        "continuous_cross_refit_scaler": 33_000,
    }
    return offsets[protocol]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed SP500 TATR protocol robustness search using stored reference checkpoints."
    )
    parser.add_argument("--seeds", type=parse_ints, default=parse_ints("42,43,44"))
    parser.add_argument("--eval-blocks", type=parse_blocks, default=parse_blocks("0,30,50,70,100"))
    parser.add_argument(
        "--protocols",
        default="continuous_chunked,independent_fixed,continuous_cross_refit_scaler",
        help="Comma-separated subset of continuous_chunked, independent_fixed, continuous_cross_refit_scaler",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--aug-length", type=int, default=252)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--ahead", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--loss", default="mae", choices=["mae", "mse"])
    parser.add_argument("--output", default="reports/generated_outputs/10_protocol_multiseed")
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

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    start = time.time()
    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    allowed = {"continuous_chunked", "independent_fixed", "continuous_cross_refit_scaler"}
    unknown = sorted(set(protocols) - allowed)
    if unknown:
        raise ValueError(f"Unknown protocols: {unknown}")

    downstream_timeseries, segments_test, labels_test, lengths_test = get_downstream_data()
    init_prices = np.asarray(downstream_timeseries[: 252 * 5], dtype=np.float64)
    test_prices = np.asarray(downstream_timeseries[252 * 5 :], dtype=np.float64)
    init_dataset, fixed_scaler = Timeseries2Dataset_Downstream(init_prices, args.window_size)
    test_dataset_fixed_scaler = Timeseries2Dataset_Downstream(test_prices, args.window_size, fixed_scaler)
    init_state, init_segment = init_first_segment(segments_test, labels_test, lengths_test)

    model = load_ftsdiffusion()
    _, _, patterns = sampling_inputs()
    l_min = prm_params["l_min"]
    max_blocks = max(args.eval_blocks)
    max_length = max_blocks * args.aug_length

    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def generate_continuous(length: int, seed: int) -> np.ndarray:
        set_seed(seed)
        state = init_state.clone().detach()
        timeseries = list(init_segment)
        curr_t = len(timeseries)
        while curr_t < length:
            state = state_evolution_ftsdiffusion(model["evolution"], state, l_min)
            segment = segment_generation_ftsdiffusion(model["generation"], state, patterns)
            segment = segment - segment[0] + timeseries[-1]
            timeseries.extend(segment)
            curr_t += len(segment)
        return np.asarray(timeseries[:length], dtype=np.float64)

    def generate_independent_blocks(n_blocks: int, seed: int) -> list[np.ndarray]:
        return [generate_continuous(args.aug_length, seed + block_idx) for block_idx in range(n_blocks)]

    rows: list[dict] = []
    synthetic_rows: list[dict] = []
    long_path = out / "multiseed_protocol_long.csv"

    for seed in args.seeds:
        synthetic_cont = generate_continuous(max_length, seed)
        independent_blocks = generate_independent_blocks(max_blocks, seed + 10_000)
        independent_flat = np.concatenate(independent_blocks)
        synthetic_rows.extend(
            [
                {
                    "seed": seed,
                    "protocol": "continuous",
                    "n_days": len(synthetic_cont),
                    "first": float(synthetic_cont[0]),
                    "last": float(synthetic_cont[-1]),
                    "min": float(synthetic_cont.min()),
                    "max": float(synthetic_cont.max()),
                    "mean": float(synthetic_cont.mean()),
                    "std": float(synthetic_cont.std()),
                },
                {
                    "seed": seed,
                    "protocol": "independent_blocks",
                    "n_days": len(independent_flat),
                    "first": float(independent_blocks[0][0]),
                    "last": float(independent_blocks[-1][-1]),
                    "min": float(independent_flat.min()),
                    "max": float(independent_flat.max()),
                    "mean": float(independent_flat.mean()),
                    "std": float(independent_flat.std()),
                },
            ]
        )
        pd.DataFrame(synthetic_rows).to_csv(out / "multiseed_synthetic_price_stats.csv", index=False)

        def build_dataset(protocol: str, blocks: int):
            if protocol == "continuous_chunked":
                curr_dataset = init_dataset.clone().detach()
                for block_idx in range(blocks):
                    block = synthetic_cont[block_idx * args.aug_length : (block_idx + 1) * args.aug_length]
                    block_dataset = Timeseries2Dataset_Downstream(block, args.window_size, fixed_scaler)
                    curr_dataset = concat_datasets_downstream(curr_dataset, block_dataset)
                return curr_dataset, test_dataset_fixed_scaler, fixed_scaler

            if protocol == "independent_fixed":
                curr_dataset = init_dataset.clone().detach()
                for block_idx in range(blocks):
                    block_dataset = Timeseries2Dataset_Downstream(
                        independent_blocks[block_idx], args.window_size, fixed_scaler
                    )
                    curr_dataset = concat_datasets_downstream(curr_dataset, block_dataset)
                return curr_dataset, test_dataset_fixed_scaler, fixed_scaler

            if protocol == "continuous_cross_refit_scaler":
                if blocks == 0:
                    train_series = init_prices
                else:
                    train_series = np.concatenate([init_prices, synthetic_cont[: blocks * args.aug_length]])
                train_dataset, scaler = Timeseries2Dataset_Downstream(train_series, args.window_size)
                test_dataset = Timeseries2Dataset_Downstream(test_prices, args.window_size, scaler)
                return train_dataset, test_dataset, scaler

            raise ValueError(protocol)

        for protocol in protocols:
            for blocks in args.eval_blocks:
                train_dataset, test_dataset, scaler = build_dataset(protocol, blocks)
                lstm_seed = seed + stable_protocol_offset(protocol) + blocks
                set_seed(lstm_seed)
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
                mape = float(test_on_real(predictor, test_dataset, scaler, criterion="mape"))
                fit_seconds = time.time() - t0
                row = {
                    "seed": seed,
                    "protocol": protocol,
                    "augmentation_blocks": blocks,
                    "synthetic_days": blocks * args.aug_length,
                    "train_windows": int(train_dataset.shape[0]),
                    "mape": mape,
                    "fit_seconds": round(fit_seconds, 3),
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(long_path, index=False)
                print(
                    f"[multiseed] seed={seed} {protocol} blocks={blocks} "
                    f"windows={train_dataset.shape[0]} mape={mape:.7f} fit={fit_seconds:.1f}s",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    baselines = df[df["augmentation_blocks"].eq(0)][["seed", "protocol", "mape"]].rename(
        columns={"mape": "baseline_mape"}
    )
    df = df.merge(baselines, on=["seed", "protocol"], how="left")
    df["pct_change_vs_no_aug"] = (df["mape"] - df["baseline_mape"]) / df["baseline_mape"] * 100
    df.to_csv(long_path, index=False)

    per_seed_rows = []
    for (seed, protocol), group in df.groupby(["seed", "protocol"], sort=False):
        group = group.sort_values("augmentation_blocks")
        best = group.loc[group["mape"].idxmin()]
        final = group.iloc[-1]
        baseline = group.iloc[0]
        per_seed_rows.append(
            {
                "seed": seed,
                "protocol": protocol,
                "baseline_mape": baseline["mape"],
                "best_blocks": int(best["augmentation_blocks"]),
                "best_mape": best["mape"],
                "best_pct_vs_no_aug": best["pct_change_vs_no_aug"],
                "final_blocks": int(final["augmentation_blocks"]),
                "final_mape": final["mape"],
                "final_pct_vs_no_aug": final["pct_change_vs_no_aug"],
            }
        )
    per_seed = pd.DataFrame(per_seed_rows)
    per_seed.to_csv(out / "multiseed_protocol_per_seed_summary.csv", index=False)

    summary = (
        per_seed.groupby("protocol")
        .agg(
            seeds=("seed", "count"),
            baseline_mape_mean=("baseline_mape", "mean"),
            best_mape_mean=("best_mape", "mean"),
            best_pct_mean=("best_pct_vs_no_aug", "mean"),
            final_mape_mean=("final_mape", "mean"),
            final_pct_mean=("final_pct_vs_no_aug", "mean"),
            final_pct_min=("final_pct_vs_no_aug", "min"),
            final_pct_max=("final_pct_vs_no_aug", "max"),
        )
        .reset_index()
    )
    summary.to_csv(out / "multiseed_protocol_summary.csv", index=False)

    curve = (
        df.groupby(["protocol", "augmentation_blocks"])
        .agg(
            mean_pct=("pct_change_vs_no_aug", "mean"),
            min_pct=("pct_change_vs_no_aug", "min"),
            max_pct=("pct_change_vs_no_aug", "max"),
            mean_mape=("mape", "mean"),
            min_mape=("mape", "min"),
            max_mape=("mape", "max"),
        )
        .reset_index()
    )
    curve.to_csv(out / "multiseed_protocol_curve_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for protocol, group in curve.groupby("protocol", sort=False):
        group = group.sort_values("augmentation_blocks")
        ax.plot(group["augmentation_blocks"], group["mean_pct"], marker="o", linewidth=2, label=protocol)
        ax.fill_between(group["augmentation_blocks"], group["min_pct"], group["max_pct"], alpha=0.16)
    ax.axhline(0, color="0.25", linewidth=1)
    ax.set_title("SP500 TATR multi-seed protocol robustness")
    ax.set_xlabel("Synthetic 252-day blocks")
    ax.set_ylabel("MAPE change vs no augmentation (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "multiseed_protocol_pct_change.png", dpi=220)
    fig.savefig(out / "multiseed_protocol_pct_change.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for protocol, group in curve.groupby("protocol", sort=False):
        group = group.sort_values("augmentation_blocks")
        ax.plot(group["augmentation_blocks"], group["mean_mape"], marker="o", linewidth=2, label=protocol)
        ax.fill_between(group["augmentation_blocks"], group["min_mape"], group["max_mape"], alpha=0.16)
    ax.set_title("SP500 TATR multi-seed MAPE")
    ax.set_xlabel("Synthetic 252-day blocks")
    ax.set_ylabel("MAPE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "multiseed_protocol_mape.png", dpi=220)
    fig.savefig(out / "multiseed_protocol_mape.pdf")
    plt.close(fig)

    metadata = {
        "asset": "sp500",
        "experiment": "TATR",
        "purpose": "multi-seed robustness check for continuous-vs-independent protocol discrepancy",
        "settings": vars(args),
        "elapsed_seconds": round(time.time() - start, 3),
        "used_stored_sisc_artifacts": True,
        "used_stored_pgm_pem_checkpoints": True,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    readme = "# SP500 TATR Multi-Seed Protocol Robustness\n\n"
    readme += "Three-seed robustness run under the same paper-like downstream LSTM settings.\n\n"
    readme += "- `continuous_chunked`: one continuous synthetic trajectory per seed, split into 252-day chunks before windowing.\n"
    readme += "- `independent_fixed`: released reference-code structure, independent fixed-initialized 252-day blocks.\n"
    readme += "- `continuous_cross_refit_scaler`: continuous prefix with scaler refit on augmented train data.\n\n"
    readme += "See `multiseed_protocol_summary.csv` and `multiseed_protocol_pct_change.png` for the main result.\n"
    (out / "README.md").write_text(readme)

    print(json.dumps({"out": str(out.relative_to(ROOT)), "summary": summary.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
