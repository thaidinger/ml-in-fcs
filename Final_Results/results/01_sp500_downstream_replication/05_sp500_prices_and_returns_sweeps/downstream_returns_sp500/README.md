# SP500 Returns Downstream Replication

Reduced author-style TMTR/TATR rerun using the stored SP500 SISC artifacts and PGM/PEM checkpoints in `fts-diffusion-ref`.

Settings:

- runs: 3
- LSTM epochs: 25
- hidden size: 32
- loss: MAE
- window size: 64
- ahead: 1
- TATR augmentation levels: 0..10 blocks
- TMTR synthetic proportions: 0..100% in 10% steps
- elapsed time: 196.169 seconds

Best average MAPE in this reduced run:

- TATR returns: augmentation index 2, average MAPE 1.024199
- TMTR returns: 60% synthetic, average MAPE 1.058940

Files:

- `tatr_returns_matrix.csv`
- `tatr_returns_summary_authors_style.csv`
- `tatr_returns.png`
- `tatr_returns.pdf`
- `tatr_returns_runs.png`
- `tatr_returns_runs.pdf`
- `tatr_returns_heatmap.png`
- `tatr_returns_heatmap.pdf`
- `tmtr_returns_matrix.csv`
- `tmtr_returns_summary_authors_style.csv`
- `tmtr_returns.png`
- `tmtr_returns.pdf`
- `tmtr_returns_runs.png`
- `tmtr_returns_runs.pdf`
- `tmtr_returns_heatmap.png`
- `tmtr_returns_heatmap.pdf`
- `combined_returns_summary.png`
- `combined_returns_summary.pdf`
- `relative_mape_change.png`
- `relative_mape_change.pdf`
- `metadata.json`
