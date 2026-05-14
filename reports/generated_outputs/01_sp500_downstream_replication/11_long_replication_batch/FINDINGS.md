# Long Replication Batch Findings

This batch used stored S&P 500 reference artifacts/checkpoints and paper-like downstream LSTM settings: prices, hidden size `32`, 2 LSTM layers, MAE loss, `100` epochs, window `64`, initial TATR real period `252*5`, synthetic block length `252`.

The full staged batch completed in the recorded runtime in `metadata.json`.

## TATR 1-Day Summary

| Protocol | Seeds | Baseline MAPE | Best MAPE | Best change | Final MAPE | Final change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `continuous_chunked` | 6 | `0.05892` | `0.00889` | `-84.85%` | `0.02038` | `-65.21%` |
| `continuous_cross_refit_scaler` | 6 | `0.06298` | `0.00873` | `-85.42%` | `0.02030` | `-68.25%` |
| `independent_fixed` | 6 | `0.06459` | `0.06459` | `0.00%` | `0.18956` | `193.20%` |

## TATR 5-Day Summary

| Protocol | Seeds | Baseline MAPE | Best MAPE | Best change | Final MAPE | Final change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `continuous_chunked`, ahead=5 | 6 | `0.08096` | `0.01180` | `-85.19%` | `0.01777` | `-77.46%` |

## Interpretation

- The paper-like TATR drop is strongly reproducible under continuous synthetic trajectory protocols.
- The released reference-code independent-block TATR protocol remains non-replicating and worsens substantially.
- Re-fitting the scaler does not remove the continuous-trajectory improvement.
- The 5-day ahead continuous protocol also reproduces a large drop, matching the qualitative appendix/page-21 behavior much better than the released independent-block protocol.
- TMTR remains noisy: low or mid synthetic proportions can help, but high synthetic proportions often degrade unless a favorable continuous offset is used. This is less clean than TATR.

## Cautious Conclusion

The authors' qualitative TATR result is replicable if their effective experiment used one long continuous synthetic trajectory, or chunks from such a trajectory. It is not replicable from the released independent-block TATR code path. The strongest explanation remains a protocol mismatch between the paper figure and the released reference implementation.
