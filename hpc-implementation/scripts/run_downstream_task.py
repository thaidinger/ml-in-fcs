#!/usr/bin/env python3
"""Run one resumable SP500 TMTR/TATR downstream task on a cluster node.

Each Slurm array task handles one run id.  The companion aggregation script
combines all run-level CSV files into summary tables and publication-ready
plots under the experiment results directory.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("hpc-implementation/results/sp500_hpc_full"))
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--experiment", choices=["both", "tatr", "tmtr"], default="both")
    parser.add_argument("--datatype", choices=["prices", "returns"], default="prices")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--ahead", type=int, default=1)
    parser.add_argument("--loss", choices=["mae", "mse"], default="mae")
    parser.add_argument("--tatr-augmentations", type=int, default=100)
    parser.add_argument("--tatr-protocol", choices=["author_independent", "continuous_chunked"], default="author_independent")
    parser.add_argument("--tmtr-proportions", type=int, default=10)
    parser.add_argument("--aug-length", type=int, default=252)
    parser.add_argument("--mix-length", type=int, default=252 * 5)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--epoch-log-interval", type=int, default=25)
    return parser.parse_args()


def pct_returns(series: np.ndarray) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    return (series[1:] - series[:-1]) / series[:-1]


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def main() -> None:
    args = parse_args()
    run_id = args.run_id
    if run_id is None:
        run_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    run_id = int(run_id)

    repo_root = args.repo_root.resolve()
    ref_root = repo_root / "fts-diffusion-ref"
    out = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ref_root))
    os.chdir(ref_root)
    os.environ.setdefault("MPLBACKEND", "Agg")
    torch.set_num_threads(max(1, args.torch_threads))
    set_seed(args.seed_base + run_id)

    from experiments.predictor_lstm import LSTMPredictor, set_loss_fn, test_on_real
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

    started = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(
        "[task-start] "
        f"run_id={run_id} experiment={args.experiment} datatype={args.datatype} "
        f"device={device} out={out}"
    )
    log(
        "[task-settings] "
        f"epochs={args.epochs} hidden_dim={args.hidden_dim} window={args.window_size} "
        f"ahead={args.ahead} loss={args.loss} tatr_augmentations={args.tatr_augmentations} "
        f"tmtr_proportions={args.tmtr_proportions}"
    )

    log("[setup] loading downstream SP500 data and stored SP500 SISC artifacts")
    downstream_ts, segments_test, labels_test, lengths_test = get_downstream_data()
    downstream_ts = np.asarray(downstream_ts, dtype=np.float64)
    init_state, init_segment = init_first_segment(segments_test, labels_test, lengths_test)
    log(
        "[setup] "
        f"downstream_points={len(downstream_ts)} test_segments={len(segments_test)} "
        f"init_segment_len={len(init_segment)}"
    )
    log("[setup] loading stored SP500 PEM/PGM checkpoints")
    model = load_ftsdiffusion()
    _, _, patterns = sampling_inputs()
    l_min = prm_params["l_min"]
    log(f"[setup] model ready; patterns_shape={tuple(patterns.shape)} l_min={l_min}")

    generated_cache: dict[tuple[int, int], np.ndarray] = {}

    def generate_prices(length: int, seed: int) -> np.ndarray:
        length = int(length)
        seed = int(seed)
        key = (length, seed)
        if key in generated_cache:
            log(f"[sampling] cache hit length={length} seed={seed}")
            return generated_cache[key]
        log(f"[sampling] start length={length} seed={seed}")
        t0 = time.time()
        set_seed(seed)
        state = init_state.clone().detach()
        timeseries = list(init_segment)
        curr_t = len(timeseries)
        last_reported = curr_t
        while curr_t < length:
            state = state_evolution_ftsdiffusion(model["evolution"], state, l_min)
            segment = segment_generation_ftsdiffusion(model["generation"], state, patterns)
            segment = segment - segment[0] + timeseries[-1]
            timeseries.extend(segment)
            curr_t += len(segment)
            if curr_t - last_reported >= 2_500 or curr_t >= length:
                log(f"[sampling] progress seed={seed} generated={min(curr_t, length)}/{length}")
                last_reported = curr_t
        result = np.asarray(timeseries[:length], dtype=np.float64)
        generated_cache[key] = result
        log(f"[sampling] done length={length} seed={seed} seconds={time.time() - t0:.1f}")
        return result

    def transform(series: np.ndarray) -> np.ndarray:
        if args.datatype == "returns":
            return pct_returns(series)
        return np.asarray(series, dtype=np.float64).reshape(-1)

    def train_eval(
        dataset: torch.Tensor,
        test_dataset: torch.Tensor,
        scaler,
        seed: int,
        label: str,
    ) -> float:
        seed = int(seed)
        set_seed(seed)
        fit_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        predictor = LSTMPredictor(1, args.hidden_dim, args.ahead, 2, fit_device).to(fit_device)
        criterion = set_loss_fn(args.loss)
        optimizer = optim.Adam(predictor.parameters(), lr=1e-2)
        X = dataset[:, :-args.ahead].unsqueeze(-1).to(fit_device)
        y = dataset[:, -args.ahead :].to(fit_device)
        log(
            f"[{label}] train-start seed={seed} windows={dataset.shape[0]} "
            f"epochs={args.epochs} device={fit_device}"
        )
        for epoch in range(args.epochs):
            predictor.train()
            y_pred = predictor(X)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_num = epoch + 1
            if (
                epoch_num == 1
                or epoch_num == args.epochs
                or (args.epoch_log_interval > 0 and epoch_num % args.epoch_log_interval == 0)
            ):
                log(f"[{label}] epoch={epoch_num}/{args.epochs} loss={loss.item():.6f}")
        mape = float(test_on_real(predictor, test_dataset, scaler, criterion="mape"))
        log(f"[{label}] eval-complete mape={mape:.7f}")
        return mape

    if args.experiment in {"both", "tatr"}:
        tatr_path = raw / f"tatr_run_{run_id:03d}.csv"
        if tatr_path.exists():
            log(f"[tatr-prices] {tatr_path.name} already exists; skipping")
        else:
            log("[tatr-prices] setup start")
            init_prices = downstream_ts[: 252 * 5]
            test_prices = downstream_ts[252 * 5 :]
            init_series = transform(init_prices)
            test_series = transform(test_prices)
            init_dataset, tatr_scaler = Timeseries2Dataset_Downstream(init_series, args.window_size)
            test_dataset = Timeseries2Dataset_Downstream(test_series, args.window_size, tatr_scaler)
            log(
                "[tatr-prices] setup done "
                f"init_points={len(init_series)} test_points={len(test_series)} "
                f"init_windows={init_dataset.shape[0]} test_windows={test_dataset.shape[0]}"
            )

            rows = []
            curr_dataset = init_dataset.clone().detach()
            if args.tatr_protocol == "continuous_chunked":
                log("[tatr-prices] pre-generating continuous synthetic trajectory")
                cont_prices = generate_prices(args.tatr_augmentations * args.aug_length, args.seed_base + run_id)
            else:
                cont_prices = None

            for aug_idx in range(args.tatr_augmentations + 1):
                log(f"[tatr-prices] augmentation {aug_idx}/{args.tatr_augmentations} begin")
                if aug_idx > 0:
                    if args.tatr_protocol == "continuous_chunked":
                        start = (aug_idx - 1) * args.aug_length
                        syn_prices = cont_prices[start : start + args.aug_length]
                    else:
                        syn_prices = generate_prices(args.aug_length, args.seed_base + run_id * 10_000 + aug_idx)
                    syn_dataset = Timeseries2Dataset_Downstream(transform(syn_prices), args.window_size, tatr_scaler)
                    curr_dataset = concat_datasets_downstream(curr_dataset, syn_dataset)
                    log(
                        f"[tatr-prices] augmentation {aug_idx} synthetic block added; "
                        f"train_windows={curr_dataset.shape[0]}"
                    )

                fit_start = time.time()
                mape = train_eval(
                    curr_dataset,
                    test_dataset,
                    tatr_scaler,
                    args.seed_base + run_id * 1_000 + aug_idx,
                    label=f"tatr-prices run={run_id} aug={aug_idx}",
                )
                row = {
                    "run_id": run_id,
                    "experiment": "tatr",
                    "datatype": args.datatype,
                    "protocol": args.tatr_protocol,
                    "augmentation_idx": aug_idx,
                    "synthetic_days": aug_idx * args.aug_length,
                    "train_windows": int(curr_dataset.shape[0]),
                    "mape": mape,
                    "fit_seconds": round(time.time() - fit_start, 3),
                }
                rows.append(row)
                atomic_write_csv(pd.DataFrame(rows), tatr_path)
                log(
                    f"[tatr-prices] run={run_id} aug={aug_idx}/{args.tatr_augmentations} "
                    f"mape={mape:.7f} fit_seconds={time.time() - fit_start:.1f} saved={tatr_path}"
                )

    if args.experiment in {"both", "tmtr"}:
        tmtr_path = raw / f"tmtr_run_{run_id:03d}.csv"
        if tmtr_path.exists():
            log(f"[tmtr-prices] {tmtr_path.name} already exists; skipping")
        else:
            log("[tmtr-prices] setup start")
            real_prices = downstream_ts[: args.mix_length]
            test_prices = downstream_ts[args.mix_length :]
            real_series = transform(real_prices)
            test_series = transform(test_prices)
            _, tmtr_scaler = Timeseries2Dataset_Downstream(test_series, args.window_size)
            test_dataset = Timeseries2Dataset_Downstream(test_series, args.window_size, tmtr_scaler)
            log(
                "[tmtr-prices] setup done "
                f"real_points={len(real_series)} test_points={len(test_series)} "
                f"test_windows={test_dataset.shape[0]}"
            )
            log("[tmtr-prices] generating synthetic mix series")
            syn_series = transform(generate_prices(args.mix_length, args.seed_base + run_id))
            proportions = np.linspace(0, 100, args.tmtr_proportions + 1, dtype=int)
            mix_series_length = len(real_series)

            rows = []
            for proportion_pct in proportions:
                log(f"[tmtr-prices] proportion {proportion_pct}% begin")
                proportion = proportion_pct / 100
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
                log(
                    f"[tmtr-prices] proportion {proportion_pct}% dataset ready; "
                    f"train_windows={mix_dataset.shape[0]}"
                )

                fit_start = time.time()
                mape = train_eval(
                    mix_dataset,
                    test_dataset,
                    tmtr_scaler,
                    args.seed_base + run_id * 2_000 + proportion_pct,
                    label=f"tmtr-prices run={run_id} prop={proportion_pct}",
                )
                row = {
                    "run_id": run_id,
                    "experiment": "tmtr",
                    "datatype": args.datatype,
                    "synthetic_proportion_pct": int(proportion_pct),
                    "train_windows": int(mix_dataset.shape[0]),
                    "mape": mape,
                    "fit_seconds": round(time.time() - fit_start, 3),
                }
                rows.append(row)
                atomic_write_csv(pd.DataFrame(rows), tmtr_path)
                log(
                    f"[tmtr-prices] run={run_id} prop={proportion_pct}% "
                    f"mape={mape:.7f} fit_seconds={time.time() - fit_start:.1f} saved={tmtr_path}"
                )

    metadata = {
        "run_id": run_id,
        "repo_root": str(repo_root),
        "output_dir": str(out),
        "device": device,
        "settings": None,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    # Make settings JSON-safe (convert Path and other non-serializable types to strings)
    def _json_safe(v):
        if isinstance(v, Path):
            return str(v)
        try:
            json.dumps(v)
            return v
        except TypeError:
            return str(v)

    metadata["settings"] = {k: _json_safe(v) for k, v in vars(args).items()}
    (raw / f"task_{run_id:03d}_metadata.json").write_text(json.dumps(metadata, indent=2))
    log("[task-complete] " + json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
