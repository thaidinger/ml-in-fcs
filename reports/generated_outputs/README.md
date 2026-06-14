# Generated Outputs

This directory contains generated replication artifacts and diagnostics. It is organized by storyline so reviewers can quickly distinguish the S&P downstream replication work from the GOOG/ZCF real-asset SISC work and the Appendix B.3 simulated-data SISC investigation.

## Navigation

| Folder | Purpose | Read first |
|---|---|---|
| `01_sp500_downstream_replication/` | S&P 500 TMTR/TATR downstream replication, protocol diagnostics, and long-batch checks. | `01_sp500_downstream_replication/README.md` |
| `02_goog_zcf_real_asset_sisc/` | GOOG and ZC=F real-asset SISC pattern-library artifacts and comparison notes. | `02_goog_zcf_real_asset_sisc/README.md` |
| `03_appendix_b3_simulated_sisc/` | Appendix B.3 simulated-data SISC replication attempts and sweep results. | `03_appendix_b3_simulated_sisc/README.md` |
| `04_presentation_figures/` | Slide-ready visual summaries generated from the organized result tables. | `04_presentation_figures/README.md` |

Top-level files:

- `EXPERIMENT_REGISTRY.md`: duplicate-run guard with settings keys for each completed experiment.
- `experiment_registry.csv`: machine-readable version of the experiment registry.
- `SETTINGS_COMPARISON.md`: original settings comparison for the S&P 500 downstream experiments.
- `settings_summary.csv`: machine-readable version of the settings comparison.
- `MANIFEST.txt`: grouped inventory of the retained output folders.

Before launching a new experiment, check `EXPERIMENT_REGISTRY.md`. If the proposed settings match an existing `Settings key`, treat it as already run unless the code, data, or evaluation metric has changed.

## Headline Results

### S&P 500 Downstream Replication

The released author-style protocol using stored S&P artifacts did not reproduce a sustained downward TATR trend. A continuous synthetic trajectory protocol can reproduce a paper-like drop, but that differs from the released reference-code independent-block TATR protocol. The long-batch run supports the same conclusion across seeds.

### GOOG and ZC=F Real-Asset SISC

GOOG and corn futures (`ZC=F`) SISC pattern libraries were generated with `K=11` and plotted. These real-asset pattern libraries cannot be scored against the paper's Appendix B.3 metrics because real market data has no ground-truth pattern labels or segment boundaries.

### Appendix B.3 Simulated SISC

The B.3-specific simulated-data replication attempts did not reproduce the paper's reported values. The best sweep result reached average per-unit DTW `0.0321` versus the paper's `0.01`; the best author-style interval IoU reached `0.6418` versus the paper's `0.784`.

## Hygiene

Author/reference source code remains outside this generated-output tree. OS metadata files such as `.DS_Store` were removed from this directory before regrouping.
