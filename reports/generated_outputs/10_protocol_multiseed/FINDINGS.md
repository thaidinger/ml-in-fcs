# Multi-Seed Protocol Robustness Findings

This run used the same paper-like downstream LSTM settings as the earlier protocol search:

- S&P 500 prices
- one-day ahead
- window size `64`
- hidden size `32`
- 2-layer LSTM
- MAE loss
- `100` epochs
- initial real TATR period `252*5`
- synthetic block size `252`
- evaluation blocks `0, 30, 50, 70, 100`

The run attempted seeds `42, 43, 44` and was stopped after two complete seeds to stay near the requested one-hour laptop runtime.

## Summary Across Completed Seeds

| Protocol | Seeds | Baseline MAPE | Best MAPE | Best change | Final MAPE | Final change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `continuous_chunked` | 2 | `0.06286` | `0.00961` | `-84.67%` | `0.01833` | `-70.15%` |
| `continuous_cross_refit_scaler` | 2 | `0.06054` | `0.01344` | `-75.35%` | `0.02234` | `-63.28%` |
| `independent_fixed` | 2 | `0.06977` | `0.06977` | `0.00%` | `0.16365` | `+137.12%` |

## Interpretation

The multi-seed result confirms the earlier one-seed protocol search:

- Continuous synthetic trajectories reproduce the paper-like TATR improvement across both completed seeds.
- The released reference-style independent-block protocol worsens substantially across both completed seeds.
- Re-fitting the scaler does not remove the continuous-trajectory improvement.

This strengthens the conclusion that the discrepancy is protocol-level: the paper-like curve is reachable, but not from the released independent-block TATR implementation.
