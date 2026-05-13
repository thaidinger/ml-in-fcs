# Generated Outputs

Canonical outputs generated during the S&P 500 TATR/TMTR replication work.

This is the single stored report tree. Older unnumbered folders directly under `reports/` were byte-for-byte duplicates of the numbered folders here and were removed.

## Folder Map

- `01_sp500_paper_protocol_resource_aware`: final resource-aware one-day author-protocol TATR/TMTR run using stored S&P artifacts.
- `02_sp500_tatr_single_diagnostic`: diagnostic TATR run with one continuous synthetic trajectory (`single` setting).
- `03_sp500_tatr_author_style_1day`: one-day TATR-only author-style run.
- `04_sp500_tatr_author_style_5day`: five-day TATR-only author-style run, corresponding to the appendix/page-21 horizon check.
- `05_sp500_prices_and_returns_sweeps`: earlier reduced TATR/TMTR sweeps for prices and returns.
- `06_tatr_audit`: audit plots/tables comparing stored TATR protocols and diagnosing synthetic price-level behavior.
- `07_replication_report`: generated replication report, tables, figures, and build script.
- `08_replicability_diagnostic`: final diagnostic deciding whether the authors' trend is reproducible from the released code/artifacts.
- `09_protocol_search`: one-hour targeted search over plausible protocol variants; identifies continuous synthetic trajectory generation as the setting that reproduces the paper-like TATR drop.
- `10_protocol_multiseed`: two-complete-seed robustness check confirming continuous protocols improve while independent fixed blocks worsen.
- `11_long_replication_batch`: six-seed long batch showing continuous TATR robustly reproduces the paper-like drop while independent-block TATR fails; includes 1-day, 5-day, and TMTR checks.

`MANIFEST.txt` lists every file copied into this consolidated folder.

## Settings

- `SETTINGS_COMPARISON.md`: human-readable comparison table with run settings, protocol differences, and headline results.
- `settings_summary.csv`: machine-readable summary of the same comparison.

## Key Results

- The faithful author-style S&P one-day run did not reproduce a sustained downward TATR trend.
- The `single` continuous-trajectory diagnostic did show a strong MAPE drop, but that protocol differs from the released author reference code.
- The audit artifacts support a protocol/generator mismatch: independent synthetic blocks stay near the initial price level, while a continuous trajectory drifts across the test-period scale.
- The final diagnostic verdict is: not replicable with the released reference code and stored S&P artifacts under the tested author-style protocols.
- The protocol search shows the qualitative paper trend is reproducible with continuous synthetic trajectories, but not with the released independent-block TATR protocol.
- The multi-seed robustness run confirms the same contrast across seeds `42` and `43`.
- The long batch expands this to seeds `42..47`: continuous TATR best MAPE averages around `0.009`, while independent-block TATR ends far worse than baseline.
