# GOOG and ZC=F Real-Asset SISC Outputs

This group contains outputs for the GOOG and corn futures (`ZC=F`) real-asset SISC replication work.

Before rerunning anything in this group, check `../EXPERIMENT_REGISTRY.md` for the exact completed settings keys.

## Folder Map

| Folder | Description |
|---|---|
| `12_sisc_pattern_library/` | Pattern-library plots and summary tables for GOOG and ZC=F with `K=11`. |
| `13_paper_comparison/` | Comparison notes explaining how these real-asset outputs relate to the paper setup. |

## Takeaway

The GOOG/ZC=F runs produce real-data SISC artifacts and visual pattern libraries. They are not Appendix B.3, and they cannot be evaluated with Appendix B.3 metrics because real market data has no ground-truth simulated labels, centroids, or segment boundaries.
