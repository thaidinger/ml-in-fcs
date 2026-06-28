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


def parse_blocks(value: str) -> list[int]:
    blocks = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not blocks or blocks[0] != 0 or blocks != sorted(set(blocks)):
        raise argparse.ArgumentTypeError("blocks must be sorted, unique, and start with 0")
    return blocks


def pct_change(values: np.ndarray) -> np.ndarray:
    return (values - values[0]) / values[0] * 100


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Targeted SP500 TATR protocol search using stored reference checkpoints."
    )
    parser.add_argument("--eval-blocks", type=parse_blocks, default=parse_blocks("0,10,30,50,70,100"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--aug-length", type=int, default=252)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--ahead", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--loss", default="mae", choices=["mae", "mse"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--protocols",
        default="continuous_cross,continuous_chunked,independent_fixed,continuous_cross_refit_scaler",
        help=(
            "Comma-separated subset of: continuous_cross, continuous_chunked, "
            "independent_fixed, continuous_cross_refit_scaler"
        ),
    )
    parser.add_argument("--output", default="reports/generated_outputs/09_protocol_search")
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
    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    allowed = {
        "continuous_cross",
        "continuous_chunked",
        "independent_fixed",
        "continuous_cross_refit_scaler",
    }
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

    def generate_continuous(length: int, seed: int) -> np.ndarray:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
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
        blocks: list[np.ndarray] = []
        for block_idx in range(n_blocks):
            block_seed = seed + block_idx
            block = generate_continuous(args.aug_length, block_seed)
            blocks.append(block)
        return blocks

    synthetic_cont = generate_continuous(max_length, args.seed)
    independent_blocks = generate_independent_blocks(max_blocks, args.seed + 10_000)

    def build_dataset(protocol: str, blocks: int):
        if protocol == "continuous_cross":
            if blocks == 0:
                return init_dataset.clone().detach(), test_dataset_fixed_scaler, fixed_scaler
            prefix = synthetic_cont[: blocks * args.aug_length]
            syn_dataset = Timeseries2Dataset_Downstream(prefix, args.window_size, fixed_scaler)
            return concat_datasets_downstream(init_dataset.clone().detach(), syn_dataset), test_dataset_fixed_scaler, fixed_scaler

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
                block_dataset = Timeseries2Dataset_Downstream(independent_blocks[block_idx], args.window_size, fixed_scaler)
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

    rows: list[dict] = []
    for protocol in protocols:
        for blocks in args.eval_blocks:
            train_dataset, test_dataset, scaler = build_dataset(protocol, blocks)
            lstm_seed = args.seed + 100_000 + hash((protocol, blocks)) % 50_000
            random.seed(lstm_seed)
            np.random.seed(lstm_seed)
            torch.manual_seed(lstm_seed)
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
            elapsed = time.time() - t0
            row = {
                "protocol": protocol,
                "augmentation_blocks": blocks,
                "synthetic_days": blocks * args.aug_length,
                "train_windows": int(train_dataset.shape[0]),
                "mape": float(mape),
                "fit_seconds": round(elapsed, 3),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(out / "protocol_search_long.csv", index=False)
            print(
                f"[protocol-search] {protocol} blocks={blocks} windows={train_dataset.shape[0]} "
                f"mape={mape:.7f} fit={elapsed:.1f}s",
                flush=True,
            )

    df = pd.DataFrame(rows)
    baseline_by_protocol = df[df["augmentation_blocks"].eq(0)].set_index("protocol")["mape"].to_dict()
    df["pct_change_vs_no_aug"] = [
        (row.mape - baseline_by_protocol[row.protocol]) / baseline_by_protocol[row.protocol] * 100
        for row in df.itertuples()
    ]
    df.to_csv(out / "protocol_search_long.csv", index=False)

    summary_rows = []
    for protocol, group in df.groupby("protocol", sort=False):
        best = group.loc[group["mape"].idxmin()]
        final = group.sort_values("augmentation_blocks").iloc[-1]
        baseline = group.sort_values("augmentation_blocks").iloc[0]
        summary_rows.append(
            {
                "protocol": protocol,
                "baseline_mape": baseline["mape"],
                "best_blocks": int(best["augmentation_blocks"]),
                "best_mape": best["mape"],
                "best_pct_vs_no_aug": best["pct_change_vs_no_aug"],
                "final_blocks": int(final["augmentation_blocks"]),
                "final_mape": final["mape"],
                "final_pct_vs_no_aug": final["pct_change_vs_no_aug"],
                "total_fit_seconds": group["fit_seconds"].sum(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "protocol_search_summary.csv", index=False)

    synth_stats = pd.DataFrame(
        [
            {
                "protocol": "continuous",
                "n_days": len(synthetic_cont),
                "first": synthetic_cont[0],
                "last": synthetic_cont[-1],
                "min": synthetic_cont.min(),
                "max": synthetic_cont.max(),
                "mean": synthetic_cont.mean(),
                "std": synthetic_cont.std(),
            },
            {
                "protocol": "independent_blocks",
                "n_days": len(independent_blocks) * args.aug_length,
                "first": independent_blocks[0][0],
                "last": independent_blocks[-1][-1],
                "min": min(float(block.min()) for block in independent_blocks),
                "max": max(float(block.max()) for block in independent_blocks),
                "mean": float(np.mean(np.concatenate(independent_blocks))),
                "std": float(np.std(np.concatenate(independent_blocks))),
            },
        ]
    )
    synth_stats.to_csv(out / "synthetic_price_stats.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for protocol, group in df.groupby("protocol", sort=False):
        group = group.sort_values("augmentation_blocks")
        ax.plot(
            group["augmentation_blocks"],
            group["pct_change_vs_no_aug"],
            marker="o",
            linewidth=2,
            label=protocol,
        )
    ax.axhline(0, color="0.25", linewidth=1)
    ax.set_title("SP500 TATR protocol search")
    ax.set_xlabel("Synthetic 252-day blocks")
    ax.set_ylabel("MAPE change vs no augmentation (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "protocol_search_pct_change.png", dpi=220)
    fig.savefig(out / "protocol_search_pct_change.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for protocol, group in df.groupby("protocol", sort=False):
        group = group.sort_values("augmentation_blocks")
        ax.plot(group["augmentation_blocks"], group["mape"], marker="o", linewidth=2, label=protocol)
    ax.set_title("SP500 TATR protocol search MAPE")
    ax.set_xlabel("Synthetic 252-day blocks")
    ax.set_ylabel("MAPE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "protocol_search_mape.png", dpi=220)
    fig.savefig(out / "protocol_search_mape.pdf")
    plt.close(fig)

    metadata = {
        "asset": "sp500",
        "experiment": "TATR",
        "purpose": "targeted search for protocol choices that reproduce the paper-like downward curve",
        "settings": vars(args),
        "elapsed_seconds": round(time.time() - start, 3),
        "interpretation": (
            "If continuous_chunked improves similarly to continuous_cross, the important change is continuous "
            "state/price-level drift rather than cross-boundary windows. If refit_scaler changes results, "
            "scaler choice is a sensitivity. independent_fixed is the released reference TATR structure."
        ),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    readme = "# SP500 TATR Protocol Search\n\n"
    readme += "Targeted one-run search for settings that can reproduce the paper-like downward TATR curve.\n\n"
    readme += "- `continuous_cross`: one continuous trajectory, windows formed across the whole prefix.\n"
    readme += "- `continuous_chunked`: one continuous trajectory, split into 252-day chunks before windowing.\n"
    readme += "- `independent_fixed`: reference author-style independent 252-day blocks from the fixed first segment.\n"
    readme += "- `continuous_cross_refit_scaler`: continuous prefix with scaler refit on augmented train series.\n\n"
    readme += "See `protocol_search_summary.csv` for the headline comparison.\n"
    (out / "README.md").write_text(readme)

    print(json.dumps({"out": str(out.relative_to(ROOT)), "summary": summary.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
