# SP500 TATR Multi-Seed Protocol Robustness

Two complete seeds were run under the same paper-like downstream LSTM settings. The third seed was not started/completed because the run was stopped to stay near the requested one-hour laptop budget.

Protocols:

- `continuous_chunked`: one continuous synthetic trajectory per seed, split into 252-day chunks before windowing.
- `independent_fixed`: released reference-code structure, independent fixed-initialized 252-day blocks.
- `continuous_cross_refit_scaler`: continuous prefix with scaler refit on augmented train data.

Main files:

- `multiseed_protocol_summary.csv`: protocol-level summary across completed seeds.
- `multiseed_protocol_per_seed_summary.csv`: one row per seed/protocol.
- `multiseed_protocol_long_complete.csv`: full completed-seed curve data.
- `multiseed_protocol_pct_change.png`: mean percent-change plot with seed range.

Conclusion: continuous protocols reproduce the paper-like TATR drop across both completed seeds; the released independent-block protocol worsens after augmentation.
