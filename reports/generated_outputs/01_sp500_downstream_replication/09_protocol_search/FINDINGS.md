# Protocol Search Findings

Verdict: the paper-like S&P 500 TATR curve is reproducible only when synthetic augmentation is generated as a **continuous trajectory**. The released reference-code TATR protocol, which adds independent 252-day blocks initialized from the same first segment, does not reproduce it.

## Tested Settings

Common settings: S&P 500 prices, one-day ahead, window size `64`, hidden size `32`, 2-layer LSTM, MAE loss, `100` epochs, stored S&P SISC artifacts/checkpoints, initial real training period `252*5`, synthetic block size `252`.

Evaluation blocks: `0, 10, 30, 50, 70, 100`.

## Results

| Protocol | Description | Baseline MAPE | Best MAPE | Final MAPE | Final change |
| --- | --- | ---: | ---: | ---: | ---: |
| `continuous_cross` | One continuous trajectory, rolling windows across full prefix | `0.06524` | `0.00915` at block `70` | `0.01535` | `-76.47%` |
| `continuous_chunked` | One continuous trajectory, split into 252-day chunks before windowing | `0.05297` | `0.00992` at block `100` | `0.00992` | `-81.27%` |
| `independent_fixed` | Released reference-code structure: independent blocks from the fixed first segment | `0.05654` | `0.05654` at block `0` | `0.11111` | `+96.51%` |
| `continuous_cross_refit_scaler` | Continuous trajectory with scaler refit on augmented train series | `0.06488` | `0.00930` at block `70` | `0.00947` | `-85.41%` |

## Interpretation

- `continuous_chunked` improves almost as strongly as `continuous_cross`, so the improvement is **not** mainly caused by extra windows crossing synthetic block boundaries.
- `continuous_cross_refit_scaler` improves strongly too, so the result is **not** mainly caused by the fixed MinMax scaler.
- `independent_fixed` fails badly under the same downstream LSTM settings, so the released reference TATR protocol remains inconsistent with the paper-like downward trend.
- The synthetic price-level stats explain the mechanism:
  - continuous trajectory: mean about `4138`, final about `7086`
  - independent blocks: mean about `1214`, final about `1249`

The continuous trajectory drifts through and beyond the real test-period price level, while independent blocks remain near the early initial level. That is why continuous protocols can make the LSTM price forecast look dramatically better, while independent-block TATR hurts.

## Bottom Line

If the paper curve was produced from the released reference code exactly as written, our runs still do not replicate it. If the paper curve was produced from a continuous synthetic trajectory or from continuous trajectory chunks, then it is straightforward to reproduce the qualitative downward trend. Those are materially different TATR protocols.
