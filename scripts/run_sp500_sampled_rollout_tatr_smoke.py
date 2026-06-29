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


def parse_blocks(value: str) -> list[int]:
    blocks = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not blocks or blocks[0] != 0 or blocks != sorted(set(blocks)):
        raise argparse.ArgumentTypeError("blocks must be sorted, unique, and start with 0")
    return blocks


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pct_change(values: pd.Series) -> pd.Series:
    return (values - values.iloc[0]) / values.iloc[0] * 100


def entropy(values: np.ndarray) -> float:
    _, counts = np.unique(values.astype(int), return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def checkpoint_path(kind: str) -> Path:
    model_dir = Path("trained_models")
    if kind == "pgm":
        base = "pgm-2_c48-80_sp500_k14_n30_lr4e-04_dw0.01_pw1_sw0.01"
    elif kind == "pem":
        base = "pem_sp500_k14_e196_h32_lr4e-04_pw0.05_lw0.01_mw0.94"
    else:
        raise ValueError(kind)
    candidates = [
        model_dir / f"{base}.pth",
        model_dir / f"{base}.pth.pth",
        model_dir / f"{base}.pt",
        model_dir / f"{base}.pth.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError([str(path) for path in candidates])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Small S&P 500 TATR comparison for argmax vs sampled PEM rollout."
    )
    parser.add_argument("--eval-blocks", type=parse_blocks, default=parse_blocks("0,10,30,50"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--aug-length", type=int, default=252)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--ahead", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--loss", default="mae", choices=["mae", "mse"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--protocols",
        default="independent_argmax,continuous_argmax_chunked,independent_sampled,continuous_sampled_chunked",
        help=(
            "Comma-separated subset of independent_argmax, continuous_argmax_chunked, "
            "independent_sampled, continuous_sampled_chunked."
        ),
    )
    parser.add_argument("--output", default="reports/generated_outputs/17_sampled_rollout_tatr_smoke")
    args = parser.parse_args()

    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
    os.chdir(REF)

    from experiments.predictor_lstm import separate_train_lstm_predictor, test_on_real
    from experiments.utils_downstream import (
        Timeseries2Dataset_Downstream,
        concat_datasets_downstream,
        get_downstream_data,
        init_first_segment,
    )
    from models.load_models import build_pem, build_pgm
    from models.model_params import prm_params
    from models.sampling import segment_generation_ftsdiffusion, state_evolution_ftsdiffusion
    from models.utils_sampling import sampling_inputs

    set_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    start = time.time()

    protocols = [item.strip() for item in args.protocols.split(",") if item.strip()]
    allowed = {
        "independent_argmax",
        "continuous_argmax_chunked",
        "independent_sampled",
        "continuous_sampled_chunked",
    }
    unknown = sorted(set(protocols) - allowed)
    if unknown:
        raise ValueError(f"Unknown protocols: {unknown}")

    device = torch.device("cpu")
    pgm = build_pgm(device)
    pem = build_pem(device)
    pgm.load_state_dict(torch.load(checkpoint_path("pgm"), map_location=device))
    pem.load_state_dict(torch.load(checkpoint_path("pem"), map_location=device))
    pgm.eval()
    pem.eval()
    model = {"generation": pgm, "evolution": pem}

    downstream_timeseries, segments_test, labels_test, lengths_test = get_downstream_data()
    init_prices = np.asarray(downstream_timeseries[: 252 * 5], dtype=np.float64)
    test_prices = np.asarray(downstream_timeseries[252 * 5 :], dtype=np.float64)
    init_dataset, fixed_scaler = Timeseries2Dataset_Downstream(init_prices, args.window_size)
    test_dataset_fixed_scaler = Timeseries2Dataset_Downstream(test_prices, args.window_size, fixed_scaler)
    init_state, init_segment = init_first_segment(segments_test, labels_test, lengths_test)
    _, _, patterns = sampling_inputs()
    l_min = prm_params["l_min"]
    n_patterns = pem.n_patterns
    range_length = pem.range_length

    def evolve_sampled(state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            pred = pem(state)
            pattern = torch.distributions.Categorical(logits=pred[:, :n_patterns]).sample().view(1, 1)
            length = (
                torch.distributions.Categorical(logits=pred[:, n_patterns : n_patterns + range_length])
                .sample()
                .view(1, 1)
                + l_min
            )
            magnitude = pred[:, n_patterns + range_length :].float()
            return torch.cat((pattern, length, magnitude), dim=1)

    def generate_chain(length: int, mode: str, seed: int, path_name: str) -> tuple[np.ndarray, pd.DataFrame]:
        set_seed(seed)
        state = init_state.clone().detach().float()
        timeseries = list(init_segment)
        states = [
            {
                "path": path_name,
                "segment_idx": 0,
                "pattern": int(state[0, 0].item()),
                "length": int(state[0, 1].item()),
                "magnitude": float(state[0, 2].item()),
            }
        ]
        while len(timeseries) < length:
            if mode == "argmax":
                state = state_evolution_ftsdiffusion(model["evolution"], state, l_min)
            elif mode == "sampled":
                state = evolve_sampled(state)
            else:
                raise ValueError(mode)
            segment = segment_generation_ftsdiffusion(model["generation"], state, patterns)
            segment = segment - segment[0] + timeseries[-1]
            timeseries.extend(segment)
            states.append(
                {
                    "path": path_name,
                    "segment_idx": len(states),
                    "pattern": int(state[0, 0].item()),
                    "length": int(state[0, 1].item()),
                    "magnitude": float(state[0, 2].item()),
                }
            )
        return np.asarray(timeseries[:length], dtype=np.float64), pd.DataFrame(states)

    max_blocks = max(args.eval_blocks)
    max_length = max_blocks * args.aug_length
    paths: dict[str, np.ndarray | list[np.ndarray]] = {}
    state_frames: list[pd.DataFrame] = []

    for mode in ("argmax", "sampled"):
        series, states = generate_chain(max_length, mode, args.seed, f"continuous_{mode}")
        paths[f"continuous_{mode}"] = series
        states["protocol_family"] = f"continuous_{mode}"
        state_frames.append(states)

        blocks = []
        for block_idx in range(max_blocks):
            block, block_states = generate_chain(
                args.aug_length,
                mode,
                args.seed + 10_000 + block_idx,
                f"independent_{mode}_block_{block_idx}",
            )
            blocks.append(block)
            block_states["block"] = block_idx
            block_states["protocol_family"] = f"independent_{mode}"
            state_frames.append(block_states)
        paths[f"independent_{mode}"] = blocks

    states_df = pd.concat(state_frames, ignore_index=True)
    states_df.to_csv(out / "sampled_rollout_states.csv", index=False)

    price_rows = []
    for family, generated in paths.items():
        if isinstance(generated, list):
            for block_idx, block in enumerate(generated):
                for t, price in enumerate(block):
                    price_rows.append(
                        {"protocol_family": family, "path": f"{family}_block_{block_idx}", "t": t, "price": price}
                    )
        else:
            for t, price in enumerate(generated):
                price_rows.append({"protocol_family": family, "path": family, "t": t, "price": price})
    price_df = pd.DataFrame(price_rows)
    price_df.to_csv(out / "sampled_rollout_prices.csv", index=False)

    def synthetic_dataset(protocol: str, blocks: int):
        if blocks == 0:
            return init_dataset.clone().detach()

        family = "sampled" if "sampled" in protocol else "argmax"
        if protocol.startswith("continuous"):
            series = np.asarray(paths[f"continuous_{family}"][: blocks * args.aug_length], dtype=np.float64)
            chunks = [series[i * args.aug_length : (i + 1) * args.aug_length] for i in range(blocks)]
        else:
            chunks = paths[f"independent_{family}"][:blocks]

        dataset = init_dataset.clone().detach()
        for chunk in chunks:
            chunk_dataset = Timeseries2Dataset_Downstream(np.asarray(chunk, dtype=np.float64), args.window_size, fixed_scaler)
            dataset = concat_datasets_downstream(dataset, chunk_dataset)
        return dataset

    rows: list[dict] = []
    for protocol in protocols:
        for blocks in args.eval_blocks:
            train_dataset = synthetic_dataset(protocol, blocks)
            lstm_seed = args.seed + 100_000 + sum(ord(ch) for ch in protocol) + blocks
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
            mape = test_on_real(predictor, test_dataset_fixed_scaler, fixed_scaler, criterion="mape")
            fit_seconds = time.time() - t0
            row = {
                "protocol": protocol,
                "augmentation_blocks": blocks,
                "synthetic_days": blocks * args.aug_length,
                "train_windows": int(train_dataset.shape[0]),
                "mape": float(mape),
                "fit_seconds": round(fit_seconds, 3),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(out / "tatr_smoke_long.csv", index=False)
            print(
                f"[sampled-smoke] {protocol} blocks={blocks} windows={train_dataset.shape[0]} "
                f"mape={mape:.7f} fit={fit_seconds:.1f}s",
                flush=True,
            )

    results = pd.DataFrame(rows)
    baseline = results[results["augmentation_blocks"].eq(0)].set_index("protocol")["mape"].to_dict()
    results["pct_change_vs_no_aug"] = [
        (row.mape - baseline[row.protocol]) / baseline[row.protocol] * 100 for row in results.itertuples()
    ]
    results.to_csv(out / "tatr_smoke_long.csv", index=False)

    summary_rows = []
    for protocol, group in results.groupby("protocol", sort=False):
        group = group.sort_values("augmentation_blocks")
        best = group.loc[group["mape"].idxmin()]
        final = group.iloc[-1]
        summary_rows.append(
            {
                "protocol": protocol,
                "baseline_mape": float(group.iloc[0]["mape"]),
                "best_blocks": int(best["augmentation_blocks"]),
                "best_mape": float(best["mape"]),
                "best_pct_vs_no_aug": float(best["pct_change_vs_no_aug"]),
                "final_blocks": int(final["augmentation_blocks"]),
                "final_mape": float(final["mape"]),
                "final_pct_vs_no_aug": float(final["pct_change_vs_no_aug"]),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "tatr_smoke_summary.csv", index=False)

    state_summary_rows = []
    for family, group in states_df.groupby("protocol_family"):
        counts = group["pattern"].value_counts().sort_index()
        state_summary_rows.append(
            {
                "protocol_family": family,
                "segments": int(len(group)),
                "unique_patterns": int(counts.size),
                "dominant_pattern_share": float(counts.max() / counts.sum()),
                "pattern_entropy_bits": entropy(group["pattern"].to_numpy()),
                "pattern_counts": json.dumps({int(k): int(v) for k, v in counts.items()}),
            }
        )
    state_summary = pd.DataFrame(state_summary_rows)
    state_summary.to_csv(out / "sampled_rollout_state_summary.csv", index=False)

    price_summary = (
        price_df.groupby("protocol_family")
        .agg(
            paths=("path", "nunique"),
            points=("price", "count"),
            first=("price", "first"),
            last=("price", "last"),
            min=("price", "min"),
            max=("price", "max"),
            mean=("price", "mean"),
            std=("price", "std"),
        )
        .reset_index()
    )
    price_summary.to_csv(out / "sampled_rollout_price_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for protocol, group in results.groupby("protocol", sort=False):
        group = group.sort_values("augmentation_blocks")
        ax.plot(group["augmentation_blocks"], group["pct_change_vs_no_aug"], marker="o", linewidth=2, label=protocol)
    ax.axhline(0, color="0.25", linewidth=1)
    ax.set_title(f"S&P 500 TATR smoke: argmax vs sampled PEM ({args.epochs} epochs)")
    ax.set_xlabel("Synthetic 252-day blocks")
    ax.set_ylabel("MAPE change vs no augmentation (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "tatr_smoke_pct_change.png", dpi=220)
    fig.savefig(out / "tatr_smoke_pct_change.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for protocol, group in results.groupby("protocol", sort=False):
        group = group.sort_values("augmentation_blocks")
        ax.plot(group["augmentation_blocks"], group["mape"], marker="o", linewidth=2, label=protocol)
    ax.set_title(f"S&P 500 TATR smoke MAPE ({args.epochs} epochs)")
    ax.set_xlabel("Synthetic 252-day blocks")
    ax.set_ylabel("MAPE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "tatr_smoke_mape.png", dpi=220)
    fig.savefig(out / "tatr_smoke_mape.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.0), sharex=False)
    for family in ("continuous_argmax", "continuous_sampled"):
        sub = price_df[price_df["protocol_family"].eq(family)]
        axes[0].plot(sub["t"], sub["price"], linewidth=1.5, label=family)
    axes[0].axhspan(float(test_prices.min()), float(test_prices.max()), color="gray", alpha=0.12, label="real test range")
    axes[0].set_title("Continuous synthetic price paths")
    axes[0].set_ylabel("price")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    for family in ("independent_argmax", "independent_sampled"):
        sub = price_df[price_df["protocol_family"].eq(family)]
        for path, group in sub.groupby("path"):
            axes[1].plot(group["t"], group["price"], linewidth=0.8, alpha=0.45, color="C0" if "argmax" in family else "C1")
        proxy = plt.Line2D([], [], color="C0" if "argmax" in family else "C1", label=family)
        axes[1].add_line(proxy)
    axes[1].axhspan(float(test_prices.min()), float(test_prices.max()), color="gray", alpha=0.12)
    axes[1].set_title("Independent 252-day synthetic blocks")
    axes[1].set_xlabel("day inside block")
    axes[1].set_ylabel("price")
    axes[1].grid(alpha=0.25)
    handles = [
        plt.Line2D([], [], color="C0", label="independent_argmax"),
        plt.Line2D([], [], color="C1", label="independent_sampled"),
    ]
    axes[1].legend(handles=handles, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sampled_vs_argmax_price_paths.png", dpi=220)
    fig.savefig(out / "sampled_vs_argmax_price_paths.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for family, group in states_df.groupby("protocol_family"):
        if family.startswith("continuous"):
            group = group.sort_values("segment_idx")
            ax.plot(group["segment_idx"], group["pattern"], marker=".", linewidth=1.2, label=family)
    ax.set_title("Continuous pattern paths: argmax collapse vs sampled PEM")
    ax.set_xlabel("generated segment")
    ax.set_ylabel("pattern")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sampled_vs_argmax_pattern_paths.png", dpi=220)
    fig.savefig(out / "sampled_vs_argmax_pattern_paths.pdf")
    plt.close(fig)

    metadata = {
        "asset": "sp500",
        "purpose": "old-vs-fixed smoke test for PEM sampled rollout in TATR",
        "settings": vars(args),
        "elapsed_seconds": round(time.time() - start, 3),
        "interpretation": (
            "This is a small downstream smoke test. It checks whether stochastic PEM rollout fixes motif "
            "collapse and how that changes TATR relative to deterministic argmax protocols."
        ),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    readme = f"""# S&P 500 Sampled-Rollout TATR Smoke Test

This folder compares the old deterministic PEM `argmax` rollout with a debugged stochastic PEM rollout that samples the learned pattern and length distributions.

Settings: one seed, blocks `{args.eval_blocks}`, `{args.epochs}` LSTM epochs, S&P 500 prices, one-day ahead, window `64`, hidden size `32`, MAE loss.

Main files:

- `tatr_smoke_summary.csv`: compact downstream summary.
- `tatr_smoke_long.csv`: full TATR curve rows.
- `sampled_rollout_state_summary.csv`: motif diversity diagnostics.
- `sampled_rollout_price_summary.csv`: generated price-level diagnostics.
- `sampled_vs_argmax_price_paths.png`: price paths for visual judgement.
- `sampled_vs_argmax_pattern_paths.png`: continuous pattern paths.

This is intentionally a smoke test, not the final statistical replication batch.
"""
    (out / "README.md").write_text(readme)

    print(json.dumps({"out": str(out.relative_to(ROOT)), "summary": summary.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
