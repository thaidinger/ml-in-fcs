# S&P 500 Sampled-Rollout TATR Smoke Test

This folder compares the old deterministic PEM `argmax` rollout with a debugged stochastic PEM rollout that samples the learned pattern and length distributions.

Settings: one seed, blocks `[0, 10, 30, 50]`, `30` LSTM epochs, S&P 500 prices, one-day ahead, window `64`, hidden size `32`, MAE loss.

Main files:

- `tatr_smoke_summary.csv`: compact downstream summary.
- `tatr_smoke_long.csv`: full TATR curve rows.
- `sampled_rollout_state_summary.csv`: motif diversity diagnostics.
- `sampled_rollout_price_summary.csv`: generated price-level diagnostics.
- `sampled_vs_argmax_price_paths.png`: price paths for visual judgement.
- `sampled_vs_argmax_pattern_paths.png`: continuous pattern paths.

This is intentionally a smoke test, not the final statistical replication batch.
