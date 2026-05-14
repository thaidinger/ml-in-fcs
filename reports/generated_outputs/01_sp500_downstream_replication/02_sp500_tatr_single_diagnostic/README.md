# SP500 TATR Single-Protocol Diagnostic

This diagnostic keeps the paper downstream LSTM settings and changes only the synthetic augmentation protocol to one continuous trajectory. The planned 5-run sweep was stopped after one completed trajectory because the large-prefix LSTM fits were too slow for the earlier 30-minute budget.

Use `tatr_prices_single_matrix.csv` and `tatr_prices_single_summary_with_change.csv` for the completed one-run curve.
