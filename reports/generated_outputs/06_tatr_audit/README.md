# TATR Audit

Audit of stored TATR outputs under `results/tatr` and the notebook runner in `src/TATR_Reduced_Replication.ipynb`.

## Main Findings

The stored results do not support the paper's monotone-downward TATR claim for the `authors` protocol.

For SP500 K14:

- `authors`: baseline `0.061146`, final `0.188566` (`+208.39%`)
- `random_init`: baseline `0.061146`, final `0.102826` (`+68.17%`)
- `single`: baseline `0.061146`, best `0.013178` at column `50`, final `0.017342` (`-71.64%`)

Across all stored `authors` runs, the best point is usually the baseline (`0`) and the final point is worse. The only SP500 protocol with a strong improvement is `single`, which is not the authors' documented independent-block TATR protocol.

## Protocol Semantics From Notebook

The notebook defines:

- `authors`: independent `252`-day synthetic blocks, every block generated from the same fixed `(init_state, init_segment)`.
- `random_init`: independent `252`-day blocks, each from a random init segment sampled from the LSTM-train portion.
- `single`: one long continuous synthetic trajectory; evaluation uses prefixes of that trajectory.
- `split`: one long continuous trajectory split into `252`-day blocks and unfolded separately.
- `burn_in`: one long trajectory with an initial burn-in discard.

## Synthetic-Level Diagnosis

SP500 TATR price ranges:

- real TATR init period: about `1250` to `2272`
- real TATR test period: about `2239` to `3240`
- `authors` synthetic blocks: mean around `1214`, max around `1276`
- `single` synthetic trajectory: mean around `4138`, last values around `7085`

This explains the protocol behavior:

- `authors` keeps appending low-price, near-initial-level synthetic data while the real test period is much higher, so the LSTM is trained on increasingly mismatched data.
- `single` drifts upward and eventually covers/overshoots the test scale, which can accidentally help price-level prediction.
- `random_init` sits between those behaviors.

## Files

- `tatr_results_audit_summary.csv`: one row per stored result matrix.
- `tatr_results_profiles.csv`: mean/std by asset/K/protocol/augmentation column.
- `sp500_synthetic_protocol_stats_by_run.csv`: synthetic price-level diagnostics by protocol/run.
- `sp500_protocol_mean_curves.png`: SP500 mean MAPE curves for `authors`, `random_init`, and `single`.
- `final_change_all_results.png`: final MAPE change vs baseline across stored runs.
- `sp500_synthetic_price_level_by_protocol.png`: synthetic price-level audit by protocol.

## Interpretation

The run artifacts point to a generator/protocol mismatch rather than a downstream-LSTM failure. The author-style independent-block protocol does not produce a decreasing TATR curve in these stored outputs. The decreasing SP500 result appears only for a continuous-trajectory protocol, which is methodologically different from the authors' stated TATR block-augmentation setup.
