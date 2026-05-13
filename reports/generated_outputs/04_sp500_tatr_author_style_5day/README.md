# SP500 TATR Five-Day-Ahead Author-Style Reduced Run

TATR-only replication targeting the page-21 / Figure 10 setting: five-day-ahead S&P 500 price prediction.

This run uses the reference code under `fts-diffusion-ref`, stored SP500 SISC artifacts, and stored SP500 PGM/PEM checkpoints.

Preserved settings:

- forecast horizon: `ahead=5`
- target: prices
- initial real training period: `252*5` prices
- augmentation block length: `252`
- LSTM epochs: `100`
- LSTM hidden size: `32`
- LSTM layers: `2`
- loss: `MAE`
- optimizer: Adam, `lr=1e-2`
- window size: `64`
- summary: authors-style trimmed mean/min/max

Resource compromise:

- runs: `3` instead of `100`
- augmentation blocks: `40` instead of `100`
- augmented-size coverage: up to `10080` synthetic days, enough to reach the first `10000`-day region visible in the paper figure
- PGM/PEM are loaded once and reused; generation math is unchanged

Runtime: `2626.495` seconds.

Result:

- no-augmentation baseline average MAPE: `0.078829`
- best average MAPE: `0.078722` at augmentation block `2` (`504` synthetic days)
- best change versus no augmentation: `-0.135%`
- final block average MAPE at `10080` synthetic days: `0.157251`

Interpretation:

- This run matches the page-21 forecast horizon and keeps the main downstream training settings.
- Under this reduced but targeted setting, the paper's downward TATR trend does not appear.
- Performance is roughly flat only at the first few augmentation blocks and then degrades as more synthetic data is appended.

Files:

- `tatr_prices_author_style_matrix.csv`
- `tatr_prices_author_style_summary.csv`
- `tatr_prices_author_style_summary_with_change.csv`
- `tatr_prices_author_style_runs.png`
- `tatr_prices_author_style_runs.pdf`
- `tatr_prices_author_style_relative_change.png`
- `tatr_prices_author_style_relative_change.pdf`
- `tatr_prices_author_style_heatmap.png`
- `tatr_prices_author_style_heatmap.pdf`
- `metadata.json`
