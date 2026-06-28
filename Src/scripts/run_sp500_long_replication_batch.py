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
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def parse_blocks(value: str) -> list[int]:
    blocks = parse_ints(value)
    if blocks[0] != 0 or blocks != sorted(set(blocks)):
        raise argparse.ArgumentTypeError("blocks must be sorted, unique, and start with 0")
    return blocks


def parse_protocols(value: str) -> list[str]:
    protocols = [x.strip() for x in value.split(",") if x.strip()]
    if not protocols:
        raise argparse.ArgumentTypeError("expected comma-separated protocols")
    return protocols


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def protocol_seed_offset(protocol: str) -> int:
    return {
        "continuous_chunked": 11_000,
        "continuous_cross": 12_000,
        "continuous_cross_refit_scaler": 13_000,
        "independent_fixed": 22_000,
        "tmtr_reference": 31_000,
        "tmtr_continuous_offset30": 32_000,
        "tmtr_continuous_offset50": 33_000,
    }[protocol]


def save_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Long SP500 downstream replication batch using stored reference checkpoints."
    )
    parser.add_argument("--budget-hours", type=float, default=7.5)
    parser.add_argument("--reserve-minutes", type=float, default=10.0)
    parser.add_argument("--seeds", type=parse_ints, default=parse_ints("42,43,44,45,46,47"))
    parser.add_argument("--tatr-blocks", type=parse_blocks, default=parse_blocks("0,10,20,30,40,50,60,70,80,90,100"))
    parser.add_argument("--control-blocks", type=parse_blocks, default=parse_blocks("0,20,40,60,80,100"))
    parser.add_argument("--tmtr-proportions", type=parse_ints, default=parse_ints("0,10,20,30,40,50,60,70,80,90,100"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--aug-length", type=int, default=252)
    parser.add_argument("--mix-length", type=int, default=252 * 5)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--ahead", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--loss", default="mae", choices=["mae", "mse"])
    parser.add_argument(
        "--stages",
        type=parse_protocols,
        default=parse_protocols(
            "tatr_continuous_chunked,tatr_independent_fixed,tatr_continuous_refit,"
            "tatr_continuous_chunked_5ahead,tmtr_reference,tmtr_continuous_offsets"
        ),
    )
    parser.add_argument("--output", default="reports/generated_outputs/11_long_replication_batch")
    args = parser.parse_args()

    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
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

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    start = time.time()
    stop_after = start + args.budget_hours * 3600
    reserve_seconds = args.reserve_minutes * 60

    downstream_ts, segments_test, labels_test, lengths_test = get_downstream_data()
    downstream_ts = np.asarray(downstream_ts, dtype=np.float64)
    init_prices = downstream_ts[: 252 * 5]
    test_prices = downstream_ts[252 * 5 :]
    tmtr_real_prices = downstream_ts[: args.mix_length]
    tmtr_test_prices = downstream_ts[args.mix_length :]

    init_dataset, fixed_scaler = Timeseries2Dataset_Downstream(init_prices, args.window_size)
    test_dataset_fixed = Timeseries2Dataset_Downstream(test_prices, args.window_size, fixed_scaler)
    _, tmtr_scaler = Timeseries2Dataset_Downstream(tmtr_test_prices, args.window_size)
    tmtr_test_dataset = Timeseries2Dataset_Downstream(tmtr_test_prices, args.window_size, tmtr_scaler)
    init_state, init_segment = init_first_segment(segments_test, labels_test, lengths_test)

    model = load_ftsdiffusion()
    _, _, patterns = sampling_inputs()
    l_min = prm_params["l_min"]

    max_tatr_blocks = max(max(args.tatr_blocks), max(args.control_blocks))
    max_cont_length = max(max_tatr_blocks * args.aug_length, 80 * args.aug_length + args.mix_length)

    metadata = {
        "asset": "sp500",
        "experiment": "TATR/TMTR",
        "purpose": "multi-hour attempt to replicate author-like downstream results and isolate protocol sensitivity",
        "settings": vars(args),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "used_stored_sisc_artifacts": True,
        "used_stored_pgm_pem_checkpoints": True,
    }
    save_json(out / "metadata.json", metadata)

    def enough_time_for_next(estimated_seconds: float = 240.0) -> bool:
        return time.time() + estimated_seconds + reserve_seconds < stop_after

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

    tatr_rows: list[dict] = []
    tmtr_rows: list[dict] = []
    synth_rows: list[dict] = []
    tatr_path = out / "long_tatr_rows.csv"
    tmtr_path = out / "long_tmtr_rows.csv"

    if tatr_path.exists():
        tatr_rows = pd.read_csv(tatr_path).to_dict(orient="records")
    if tmtr_path.exists():
        tmtr_rows = pd.read_csv(tmtr_path).to_dict(orient="records")

    completed_keys = {
        (row["stage"], int(row["seed"]), row["protocol"], int(row["ahead"]), int(row["augmentation_blocks"]))
        for row in tatr_rows
    }
    completed_tmtr_keys = {
        (row["stage"], int(row["seed"]), row["protocol"], int(row["synthetic_proportion_pct"]))
        for row in tmtr_rows
    }

    def train_eval_tatr(
        stage: str,
        seed: int,
        protocol: str,
        blocks: int,
        ahead: int,
        synthetic_cont: np.ndarray,
        independent_blocks: list[np.ndarray] | None,
    ) -> None:
        key = (stage, seed, protocol, ahead, blocks)
        if key in completed_keys:
            return
        if not enough_time_for_next():
            raise TimeoutError("time budget reached before next TATR fit")

        if protocol == "continuous_chunked":
            curr_dataset = init_dataset.clone().detach()
            for block_idx in range(blocks):
                block = synthetic_cont[block_idx * args.aug_length : (block_idx + 1) * args.aug_length]
                block_dataset = Timeseries2Dataset_Downstream(block, args.window_size, fixed_scaler)
                curr_dataset = concat_datasets_downstream(curr_dataset, block_dataset)
            test_dataset = test_dataset_fixed
            scaler = fixed_scaler
        elif protocol == "continuous_cross":
            if blocks == 0:
                curr_dataset = init_dataset.clone().detach()
            else:
                prefix = synthetic_cont[: blocks * args.aug_length]
                syn_dataset = Timeseries2Dataset_Downstream(prefix, args.window_size, fixed_scaler)
                curr_dataset = concat_datasets_downstream(init_dataset.clone().detach(), syn_dataset)
            test_dataset = test_dataset_fixed
            scaler = fixed_scaler
        elif protocol == "continuous_cross_refit_scaler":
            if blocks == 0:
                train_series = init_prices
            else:
                train_series = np.concatenate([init_prices, synthetic_cont[: blocks * args.aug_length]])
            curr_dataset, scaler = Timeseries2Dataset_Downstream(train_series, args.window_size)
            test_dataset = Timeseries2Dataset_Downstream(test_prices, args.window_size, scaler)
        elif protocol == "independent_fixed":
            if independent_blocks is None:
                raise ValueError("independent blocks required")
            curr_dataset = init_dataset.clone().detach()
            for block_idx in range(blocks):
                block_dataset = Timeseries2Dataset_Downstream(independent_blocks[block_idx], args.window_size, fixed_scaler)
                curr_dataset = concat_datasets_downstream(curr_dataset, block_dataset)
            test_dataset = test_dataset_fixed
            scaler = fixed_scaler
        else:
            raise ValueError(protocol)

        lstm_seed = seed + protocol_seed_offset(protocol) + ahead * 100 + blocks
        set_seed(lstm_seed)
        t0 = time.time()
        predictor = separate_train_lstm_predictor(
            args.epochs,
            curr_dataset,
            input_dim=1,
            hidden_dim=args.hidden_dim,
            output_dim=ahead,
            n_layers=2,
            criterion=args.loss,
            verbose=False,
        )
        mape = float(test_on_real(predictor, test_dataset, scaler, criterion="mape"))
        fit_seconds = time.time() - t0
        row = {
            "stage": stage,
            "seed": seed,
            "protocol": protocol,
            "ahead": ahead,
            "augmentation_blocks": blocks,
            "synthetic_days": blocks * args.aug_length,
            "train_windows": int(curr_dataset.shape[0]),
            "mape": mape,
            "fit_seconds": round(fit_seconds, 3),
        }
        tatr_rows.append(row)
        completed_keys.add(key)
        pd.DataFrame(tatr_rows).to_csv(tatr_path, index=False)
        print(
            f"[long-tatr] {stage} seed={seed} {protocol} ahead={ahead} blocks={blocks} "
            f"windows={curr_dataset.shape[0]} mape={mape:.7f} fit={fit_seconds:.1f}s",
            flush=True,
        )

    def train_eval_tmtr(stage: str, seed: int, protocol: str, proportion_pct: int, syn_series: np.ndarray) -> None:
        key = (stage, seed, protocol, proportion_pct)
        if key in completed_tmtr_keys:
            return
        if not enough_time_for_next():
            raise TimeoutError("time budget reached before next TMTR fit")
        proportion = proportion_pct / 100
        mix_series_length = len(tmtr_real_prices)
        if proportion == 0:
            mix_dataset = create_mix_dataset(tmtr_real_prices, mix_series_length, args.window_size, tmtr_scaler)
        elif proportion == 1:
            mix_dataset = create_mix_dataset(syn_series, mix_series_length, args.window_size, tmtr_scaler)
        else:
            syn_len = int(mix_series_length * proportion)
            real_len = mix_series_length - syn_len
            mix_real = create_mix_dataset(tmtr_real_prices, real_len, args.window_size, tmtr_scaler)
            mix_syn = create_mix_dataset(syn_series, syn_len, args.window_size, tmtr_scaler)
            mix_dataset = concat_datasets_downstream(mix_real, mix_syn)

        lstm_seed = seed + protocol_seed_offset(protocol) + proportion_pct
        set_seed(lstm_seed)
        t0 = time.time()
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
        mape = float(test_on_real(predictor, tmtr_test_dataset, tmtr_scaler, criterion="mape"))
        fit_seconds = time.time() - t0
        row = {
            "stage": stage,
            "seed": seed,
            "protocol": protocol,
            "synthetic_proportion_pct": proportion_pct,
            "train_windows": int(mix_dataset.shape[0]),
            "mape": mape,
            "fit_seconds": round(fit_seconds, 3),
        }
        tmtr_rows.append(row)
        completed_tmtr_keys.add(key)
        pd.DataFrame(tmtr_rows).to_csv(tmtr_path, index=False)
        print(
            f"[long-tmtr] {stage} seed={seed} {protocol} prop={proportion_pct}% "
            f"windows={mix_dataset.shape[0]} mape={mape:.7f} fit={fit_seconds:.1f}s",
            flush=True,
        )

    def synthetic_stats(seed: int, synthetic_cont: np.ndarray, independent_blocks: list[np.ndarray]) -> None:
        independent_flat = np.concatenate(independent_blocks)
        synth_rows.extend(
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
        pd.DataFrame(synth_rows).to_csv(out / "long_synthetic_price_stats.csv", index=False)

    stopped_reason = "completed requested stages"
    try:
        for seed in args.seeds:
            if not enough_time_for_next(estimated_seconds=360.0):
                stopped_reason = "time budget reached before next seed"
                break

            synthetic_cont = generate_continuous(max_cont_length, seed)
            independent_blocks = generate_independent_blocks(max_tatr_blocks, seed + 10_000)
            synthetic_stats(seed, synthetic_cont, independent_blocks)

            if "tatr_continuous_chunked" in args.stages:
                for blocks in args.tatr_blocks:
                    train_eval_tatr(
                        "tatr_continuous_chunked",
                        seed,
                        "continuous_chunked",
                        blocks,
                        args.ahead,
                        synthetic_cont,
                        independent_blocks,
                    )

            if "tatr_independent_fixed" in args.stages:
                for blocks in args.control_blocks:
                    train_eval_tatr(
                        "tatr_independent_fixed",
                        seed,
                        "independent_fixed",
                        blocks,
                        args.ahead,
                        synthetic_cont,
                        independent_blocks,
                    )

            if "tatr_continuous_refit" in args.stages:
                for blocks in args.control_blocks:
                    train_eval_tatr(
                        "tatr_continuous_refit",
                        seed,
                        "continuous_cross_refit_scaler",
                        blocks,
                        args.ahead,
                        synthetic_cont,
                        independent_blocks,
                    )

            if "tatr_continuous_chunked_5ahead" in args.stages:
                for blocks in args.control_blocks:
                    train_eval_tatr(
                        "tatr_continuous_chunked_5ahead",
                        seed,
                        "continuous_chunked",
                        blocks,
                        5,
                        synthetic_cont,
                        independent_blocks,
                    )

            if "tmtr_reference" in args.stages:
                syn_reference = synthetic_cont[: args.mix_length]
                for proportion_pct in args.tmtr_proportions:
                    train_eval_tmtr("tmtr_reference", seed, "tmtr_reference", proportion_pct, syn_reference)

            if "tmtr_continuous_offsets" in args.stages:
                for offset_blocks, protocol in [(30, "tmtr_continuous_offset30"), (50, "tmtr_continuous_offset50")]:
                    start_idx = offset_blocks * args.aug_length
                    syn_offset = synthetic_cont[start_idx : start_idx + args.mix_length]
                    for proportion_pct in args.tmtr_proportions:
                        train_eval_tmtr("tmtr_continuous_offsets", seed, protocol, proportion_pct, syn_offset)

    except TimeoutError as exc:
        stopped_reason = str(exc)

    elapsed = time.time() - start
    metadata.update(
        {
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(elapsed, 3),
            "stopped_reason": stopped_reason,
        }
    )
    save_json(out / "metadata.json", metadata)
    print(json.dumps({"out": str(out.relative_to(ROOT)), "elapsed_seconds": round(elapsed, 3), "stopped_reason": stopped_reason}, indent=2))


if __name__ == "__main__":
    main()
