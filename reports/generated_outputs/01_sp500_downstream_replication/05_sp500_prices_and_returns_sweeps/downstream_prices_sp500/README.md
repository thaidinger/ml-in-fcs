# SP500 Prices Downstream Replication

Reduced author-style TMTR/TATR rerun using the stored SP500 SISC artifacts and PGM/PEM checkpoints in `fts-diffusion-ref`.

This is the price-scale rerun. It is the comparable setup for the paper's MAPE values near `0.05`; the earlier returns rerun is not directly comparable because MAPE on returns is unstable when realized returns are near zero.

Settings:

- runs: 10
- LSTM epochs: 50
- hidden size: 32
- loss: MAE
- window size: 64
- ahead: 1
- TATR augmentation levels: 0..10 blocks
- TMTR synthetic proportions: 0..100% in 10% steps
- elapsed time: 1260.531 seconds

Best average MAPE in this reduced run:

- TATR prices: augmentation index 2, average MAPE 0.091616
- TMTR prices: 10% synthetic, average MAPE 0.141887

Main caveats:

- The paper-scale downstream setting uses many more independent runs; this run uses 10 because of the 30-minute budget.
- The paper uses 100 LSTM epochs in the reference downstream commands; this run uses 50 to fit both TATR and TMTR.
- The stored checkpoints in this checkout are local/reference SP500 checkpoints, not confirmed official author pretrained checkpoints.

Files:

- `tatr_prices_matrix.csv`
- `tatr_prices_summary_authors_style.csv`
- `tatr_prices.png`
- `tatr_prices.pdf`
- `tatr_prices_runs.png`
- `tatr_prices_runs.pdf`
- `tatr_prices_heatmap.png`
- `tatr_prices_heatmap.pdf`
- `tmtr_prices_matrix.csv`
- `tmtr_prices_summary_authors_style.csv`
- `tmtr_prices.png`
- `tmtr_prices.pdf`
- `tmtr_prices_runs.png`
- `tmtr_prices_runs.pdf`
- `tmtr_prices_heatmap.png`
- `tmtr_prices_heatmap.pdf`
- `combined_prices_summary.png`
- `combined_prices_summary.pdf`
- `relative_prices_mape_change.png`
- `relative_prices_mape_change.pdf`
- `metadata.json`
