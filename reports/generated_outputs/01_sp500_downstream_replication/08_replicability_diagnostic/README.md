# Replicability Diagnostic

Verdict: **NOT REPLICABLE with the released code/artifacts under the tested protocols**.

This diagnostic does not establish cheating or intent. It says that, using the released reference protocol plus the stored S&P 500 artifacts/checkpoints available in this repo, the paper-like sustained downward TATR behavior is not reproduced by the author-style protocol.

## Main Checks

1. **Protocol check**: the reference paper-like settings are `prices`, `ahead=1`, `window_size=64`, `hidden_dim=32`, `loss=mae`, `epochs=100`; TATR adds independent 252-day synthetic blocks from a fixed initial segment.
2. **Author-style reruns**: all generated author-style S&P TATR curves end worse than their no-augmentation baseline.
3. **Stored audit**: stored SP500 `authors` protocol results also end worse, while stored SP500 `single` improves strongly.
4. **Protocol sensitivity**: the `single` continuous-trajectory diagnostic produces a large drop, but it is not the released author-code TATR protocol.
5. **Synthetic-level diagnosis**: independent blocks remain near the initial price level; continuous trajectories drift through the test-period scale.

## Files

- `curve_replicability_summary.csv`: one-row summary per generated curve.
- `stored_sp500_protocol_audit.csv`: stored SP500 protocol audit rows.
- `synthetic_protocol_price_level_summary.csv`: synthetic price-level summary by protocol.
- `diagnostic_conclusion.json`: machine-readable conclusion.
- `author_vs_single_tatr_pct_change.png`: author-style vs single TATR percent-change plot.
- `synthetic_price_levels_by_protocol.png`: synthetic price-level diagnostic plot.

## Bottom Line

The authors' reported trend is **not replicable from the released reference code and stored S&P artifacts we have**. The result becomes qualitatively paper-like only after switching to the `single` continuous-trajectory protocol, which is a different protocol from the released author TATR code.
