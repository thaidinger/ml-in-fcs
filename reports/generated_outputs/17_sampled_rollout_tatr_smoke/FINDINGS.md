# Sampled-Rollout TATR Smoke Findings

This run tested the suspected fix: replace deterministic PEM `argmax` rollout with stochastic sampling from the learned pattern and length distributions.

Settings:

- S&P 500 prices
- one seed, seed `42`
- TATR blocks `0, 10, 30, 50`
- LSTM epochs `30`
- window `64`, hidden size `32`, one-day ahead, MAE loss

## Downstream Result

| Protocol | Best change | Final change | Interpretation |
| --- | ---: | ---: | --- |
| `independent_argmax` | `0.0%` | `+74.1%` | Old released-style reset blocks fail. |
| `continuous_argmax_chunked` | `-88.9%` | `-72.2%` | Paper-like TATR win, despite motif collapse. |
| `independent_sampled` | `0.0%` | `+93.4%` | Sampling motifs does not fix reset blocks. |
| `continuous_sampled_chunked` | `-51.2%` | `-51.2%` | Sampling plus continuity helps, but less than collapsed argmax. |

## Motif-State Diagnosis

| Protocol family | Unique patterns | Dominant pattern share | Entropy |
| --- | ---: | ---: | ---: |
| `continuous_argmax` | `2` | `0.992` | `0.067` |
| `continuous_sampled` | `14` | `0.175` | `3.561` |
| `independent_argmax` | `2` | `0.600` | `0.971` |
| `independent_sampled` | `14` | `0.172` | `3.571` |

Sampling clearly fixes the motif-collapse symptom.

## Price-Path Diagnosis

The price-path plot is the most important artifact:

- `sampled_vs_argmax_price_paths.png`

`continuous_argmax` is almost entirely collapsed to pattern `12`, but it drifts from about `1253` to about `4138`, crossing the real test range and producing the strongest TATR score.

`continuous_sampled` has healthy motif diversity, but drifts more moderately, ending around `2638`. It still improves TATR, but not nearly as dramatically.

Both independent-block protocols stay near the initial price level around `1200`, far below the real test range, so both hurt downstream performance.

## Interpretation

The original problem is two-layered:

1. The generation scheduler is indeed broken/degenerate under deterministic `argmax`: it collapses to a tiny motif subset.
2. The paper-like TATR improvement is not explained by fixing that collapse. It is mostly a continuous price-level drift effect in a price-level forecasting benchmark.

So the debugged stochastic PEM rollout is more plausible as a generator, but the stronger paper-like downstream result comes from a less plausible collapsed continuous argmax path.
