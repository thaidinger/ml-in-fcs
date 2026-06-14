# Experiment Registry

Use this registry before launching a new run. The `Settings key` column is the shortest practical fingerprint of a completed experiment. If a proposed run has the same key, it is a duplicate unless the code, data, or evaluation metric changed.

## S&P 500 Downstream Runs

Common settings unless noted: S&P 500 prices, stored reference SISC/PGM/PEM artifacts, chronological 80/20 split, downstream window `64`, LSTM hidden size `32`, 2 layers, MAE loss, Adam `lr=1e-2`, seed `42`, TATR initial real period `252*5`, synthetic block length `252`.

| Output folder | Settings key | What was varied | Do not rerun unless changing |
|---|---|---|---|
| `01_sp500_downstream_replication/01_sp500_paper_protocol_resource_aware` | `sp500-prices | TATR+TMTR | author_independent_blocks | ahead=1 | runs=5 | epochs=100 | grid=0..10 blocks | tmtr=0..100 step10 | seed=42` | Baseline resource-aware author-style TATR/TMTR. | Protocol, seed count, epochs, model, or split. |
| `01_sp500_downstream_replication/02_sp500_tatr_single_diagnostic` | `sp500-prices | TATR | single_continuous_trajectory | ahead=1 | runs=1_of_5 | epochs=100 | grid=0,10,...,100 | seed=42` | Diagnostic continuous trajectory prefixes. | Need more completed seeds or different continuous-generation rule. |
| `01_sp500_downstream_replication/03_sp500_tatr_author_style_1day` | `sp500-prices | TATR | author_independent_blocks | ahead=1 | runs=10 | epochs=100 | grid=0..10 blocks | seed=42` | Author-style one-day TATR-only replication. | Seed, number of runs, grid, or training protocol. |
| `01_sp500_downstream_replication/04_sp500_tatr_author_style_5day` | `sp500-prices | TATR | author_independent_blocks | ahead=5 | runs=3 | epochs=100 | grid=0..40 blocks | seed=42` | Author-style five-day horizon check. | Horizon, runs, grid, or protocol. |
| `01_sp500_downstream_replication/05_sp500_prices_and_returns_sweeps/downstream_prices_sp500` | `sp500-prices | TATR+TMTR | reduced_author_style | ahead=1 | runs=10 | epochs=50 | grid=0..10 blocks | tmtr=0..100 step10 | seed=42` | Earlier reduced price sweep. | Full epoch count, protocol, or downstream task. |
| `01_sp500_downstream_replication/05_sp500_prices_and_returns_sweeps/downstream_returns_sp500` | `sp500-returns | TATR+TMTR | reduced_author_style | ahead=1 | runs=3 | epochs=25 | grid=0..10 blocks | tmtr=0..100 step10 | seed=42` | Earlier reduced returns sweep. | Return transform, loss/metric, epochs, or protocol. |
| `01_sp500_downstream_replication/09_protocol_search` | `sp500-prices | TATR | protocol_search | ahead=1 | runs=1_each | epochs=100 | protocols=continuous_cross,continuous_chunked,independent_fixed,continuous_refit | grid=0,10,30,50,70,100 | seed=42` | One-run search across plausible generation protocols. | Add new protocol candidates or more seeds. |
| `01_sp500_downstream_replication/10_protocol_multiseed` | `sp500-prices | TATR | protocol_multiseed | ahead=1 | runs=2 seeds | epochs=100 | protocols=continuous_chunked,independent_fixed,continuous_refit | grid=0,30,50,70,100 | seeds=42,43` | Two-seed robustness check of promising protocol variants. | More seeds, different grid, or different protocol. |
| `01_sp500_downstream_replication/11_long_replication_batch` | `sp500-prices | TATR+TMTR | long_batch | ahead=1_and_5 | runs=6 | epochs=100 | TATR=0..100 blocks | controls=0,20,...,100 | TMTR=0..100 step10 | seeds=42..47` | Six-seed long batch for continuous vs independent TATR, five-day TATR, and TMTR checks. | More assets, more seeds, different model, or corrected author protocol. |

The audit/report folders `06_tatr_audit`, `07_replication_report`, and `08_replicability_diagnostic` are synthesis artifacts, not primary compute runs. Regenerate them only when upstream result folders change.

## GOOG and ZC=F Real-Asset SISC Runs

| Output folder | Settings key | What was varied | Do not rerun unless changing |
|---|---|---|---|
| `02_goog_zcf_real_asset_sisc/12_sisc_pattern_library` | `assets=GOOG,ZC=F | SISC | k=11 | l_min=10 | l_max=21 | init=kmeans++ | barycenter=dba | data=Yahoo adjusted close | dates=GOOG 2005-01-01..2020-01-01, ZC=F 2001-01-01..2020-01-01` | Generated and plotted real-asset pattern libraries. | Asset, date range, K, segment lengths, init, or barycenter. |
| `02_goog_zcf_real_asset_sisc/13_paper_comparison` | `assets=GOOG,ZC=F | comparison | depends_on=SISC k11 artifacts + PGM/PEM checkpoints` | Written comparison to paper setup. | Underlying SISC/architecture artifacts or paper-comparison framing. |

## Appendix B.3 Simulated SISC Runs

Common data unless noted: `fts-diffusion-ref/data/data_toy_l10-20_*`, series length `10000`, true `K=4`, ground-truth segments `668`, SISC init `kmeans++`, barycenter `dba`, runtime patches only for broken reference SISC execution.

| Output folder | Settings key | What was varied | Do not rerun unless changing |
|---|---|---|---|
| `03_appendix_b3_simulated_sisc/14_b3_sisc_simulated_replication` | `B3-toy | multi K=4 + derived one-pattern | l_min=10 | l_max=20 | max_iters=10 | seed=42 | metric=normalized per-unit DTW + boundary Jaccard tol2` | Initial B.3 end-to-end replication attempt. | Metric definition, one-pattern construction, seed, or iteration count. |
| `03_appendix_b3_simulated_sisc/15_b3_sisc_simulated_replication_max20` | `B3-toy | multi K=4 + derived one-pattern | l_min=10 | l_max=20 | max_iters=20 | seed=42 | metric=normalized per-unit DTW + boundary Jaccard tol2` | Same as previous with stricter max iteration count. | Metric definition, one-pattern construction, seed, or iteration count beyond 20. |
| `03_appendix_b3_simulated_sisc/16_b3_sisc_sweep` | `B3-toy | multi K=4 | seeds=0..9 for lmax20 iter10; seeds=0..4 for lmax20 iter20; seeds=0..2 for lmax21 iter20 | init=kmeans++ | barycenter=dba | metrics=boundary_Jaccard_tol0/1/2/5 + author_interval_IoU` | Timed sweep over seed, `l_max`, and iterations. | New seed range, new init/barycenter, new synthetic data, or exact author generator/code. |

## Known Non-Runs

- GOOG/ZC=F author-faithful downstream TATR/TMTR is not present here. The released S&P-style downstream split requires `252*5` held-out initialization points, which the GOOG/ZC=F 80/20 held-out windows do not provide.
- Exact Appendix B.3 author replication is not present here. The reference tree lacks the exact one-pattern data, synthetic generator, standard pattern arrays, random seed, and standalone B.3 reproduction script.

## Presentation Figure Pack

| Output folder | Settings key | What was varied | Do not rerun unless changing |
|---|---|---|---|
| `04_presentation_figures` | `presentation_figures | source=organized generated outputs | script=scripts/generate_presentation_figures.py | figures=13` | Slide-ready visual summaries, including pattern-diffusion mechanism visuals, matched-handoff PDF, and GIF animations. | Upstream result tables, presentation story, or visual style. |
