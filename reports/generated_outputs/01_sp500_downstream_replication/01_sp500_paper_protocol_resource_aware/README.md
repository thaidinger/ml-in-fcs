# SP500 Paper-Protocol Resource-Aware Replication

This folder contains a resource-aware replication of the paper's original one-day-ahead SP500 downstream protocol using the reference code under `fts-diffusion-ref` and the stored SP500 SISC + PGM/PEM artifacts.

## Original Protocol Determination

From the paper:

- Downstream task: one-day-ahead prediction in Section 5.3.
- Predictor input: previous `64` historical values.
- Metric: MAPE averaged over multiple runs.
- Data split: first `80%` of each asset for FTS-Diffusion training, final `20%` held out for downstream evaluation.
- TATR: initialize with limited observed data, then append synthetic data and evaluate on real held-out data.
- TMTR: train on real/synthetic mixtures at varying synthetic proportions, test on held-out real data.

From the reference code:

- `setup_dowmstream_tatr()` uses the first `252*5` prices of the downstream split as the initial real training period; the rest is the real test set.
- `setup_dowmstream_tmtr()` uses `mix_length = 252*5` real prices for the real mixture component; the rest is the real test set.
- TATR generation calls `generate_timeseries_ftsdiffusion(252, fixed_init_state, fixed_init_segment)` for each appended block. This is an independent-block protocol from the same fixed initial state.
- TMTR generation calls `generate_timeseries_ftsdiffusion(252*5, fixed_init_state, fixed_init_segment)` once per run.
- Reference README paper-like command uses `n_runs=100`, `n_epochs=100`, `window_size=64`, `ahead=1`, `hidden_dim=32`, `loss=mae`.

Important reference-code caveat:

- `experiments/downstream.py` is a small wrapper using `n_runs=3`, `n_epochs=10`; it is not the paper-scale protocol.
- `plot_dowmstream_tatr()` hardcodes `252*10` in the x-axis and ignores its `aug_length` argument.

## Resource-Aware Run

Preserved:

- asset: SP500
- target: prices
- forecast horizon: `ahead=1`
- window size: `64`
- hidden size: `32`
- LSTM layers: `2`
- LSTM epochs: `100`
- LSTM loss: `MAE`
- optimizer: Adam, `lr=1e-2`
- TATR block length: `252`
- TMTR mix length: `252*5`
- synthetic generation math from the reference modules
- stored SP500 SISC artifacts and stored SP500 PGM/PEM checkpoints

Downscaled:

- runs: `5` instead of `100`
- TATR augmentation steps: `10` instead of `100`
- TMTR proportions: `0,10,...,100`
- PGM/PEM loaded once and reused; this avoids repeated checkpoint loading but does not change state evolution or segment generation math

Runtime: `1357.861` seconds.

## Results

TATR:

- baseline average MAPE: `0.057437`
- best average MAPE: `0.055006` at augmentation block `2`
- final average MAPE at block `10`: `0.087458`
- interpretation: baseline is on the paper scale and there is a small early improvement, but no sustained downward trend.

TMTR:

- baseline average MAPE: `0.102008`
- best average MAPE: `0.098000` at `10%` synthetic
- final average MAPE at `100%` synthetic: `0.484582`
- interpretation: high synthetic proportions strongly degrade performance, unlike the paper's stable FTS-Diffusion TMTR curve.

## Files

- `tatr_prices_matrix.csv`
- `tatr_prices_summary_authors_style.csv`
- `tmtr_prices_matrix.csv`
- `tmtr_prices_summary_authors_style.csv`
- `combined_resource_aware_summary.png`
- `tatr_resource_aware_runs.png`
- `tatr_resource_aware_heatmap.png`
- `tmtr_resource_aware_runs.png`
- `tmtr_resource_aware_heatmap.png`
- `metadata.json`
