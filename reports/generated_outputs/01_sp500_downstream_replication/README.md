# S&P 500 Downstream Replication Outputs

This group contains the S&P 500 downstream replication artifacts: TMTR/TATR runs, protocol diagnostics, generated reports, and longer robustness checks.

Before rerunning anything in this group, check `../EXPERIMENT_REGISTRY.md` for the exact completed settings keys.

## Folder Map

| Folder | Description |
|---|---|
| `01_sp500_paper_protocol_resource_aware/` | Resource-aware one-day author-protocol TATR/TMTR run using stored S&P artifacts. |
| `02_sp500_tatr_single_diagnostic/` | Diagnostic TATR run using one continuous synthetic trajectory. |
| `03_sp500_tatr_author_style_1day/` | One-day TATR-only author-style run. |
| `04_sp500_tatr_author_style_5day/` | Five-day TATR-only author-style horizon check. |
| `05_sp500_prices_and_returns_sweeps/` | Earlier reduced TATR/TMTR sweeps for prices and returns. |
| `06_tatr_audit/` | Audit plots and tables comparing protocols and synthetic price-level behavior. |
| `07_replication_summary_tables/` | Retained TATR and reference summary tables used by the claim audit. |
| `08_replicability_diagnostic/` | Diagnostic verdict on whether the authors' trend is reproducible from released code/artifacts. |
| `09_protocol_search/` | Targeted search over plausible protocol variants. |
| `10_protocol_multiseed/` | Multi-seed robustness check contrasting continuous and independent-block protocols. |
| `11_long_replication_batch/` | Six-seed long batch with 1-day, 5-day, and TMTR checks. |

## Takeaway

The author-style released-code protocol did not reproduce a sustained TATR improvement trend. Continuous synthetic trajectories can reproduce a paper-like TATR drop, but that protocol differs from the released independent-block TATR reference path.
