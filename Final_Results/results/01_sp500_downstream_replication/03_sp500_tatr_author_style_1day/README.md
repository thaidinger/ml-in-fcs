# SP500 TATR Author-Style Reduced Run

TATR-only replication using the reference code under `fts-diffusion-ref`, stored SP500 SISC artifacts, and stored SP500 PGM/PEM checkpoints.

This run uses prices, not returns. That is the comparable MAPE scale for the paper's reported values near `0.05`.

Preserved settings:

- initial real training period: `252*5` prices
- augmentation block length: `252`
- LSTM epochs: `100`
- LSTM hidden size: `32`
- LSTM layers: `2`
- loss: `MAE`
- optimizer: Adam, `lr=1e-2`
- window size: `64`
- ahead: `1`
- summary: authors-style trimmed mean/min/max

Resource compromises:

- runs: `10` instead of `100`
- augmentation blocks: `10` instead of `100`
- PGM/PEM are loaded once and reused; generation math is unchanged

Runtime: `1368.393` seconds.

Result:

- no augmentation baseline average MAPE: `0.064301`
- best average MAPE: `0.060812` at augmentation block `2` (`504` synthetic days)
- best change versus no augmentation: `-5.426%`

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
