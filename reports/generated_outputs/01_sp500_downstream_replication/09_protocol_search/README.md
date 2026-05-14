# SP500 TATR Protocol Search

Targeted one-run search for settings that can reproduce the paper-like downward TATR curve.

- `continuous_cross`: one continuous trajectory, windows formed across the whole prefix.
- `continuous_chunked`: one continuous trajectory, split into 252-day chunks before windowing.
- `independent_fixed`: reference author-style independent 252-day blocks from the fixed first segment.
- `continuous_cross_refit_scaler`: continuous prefix with scaler refit on augmented train series.

See `protocol_search_summary.csv` for the headline comparison.
