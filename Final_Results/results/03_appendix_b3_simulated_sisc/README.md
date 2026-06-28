# Appendix B.3 Simulated SISC Outputs

This group contains the attempts to reproduce Appendix B.3 of the paper using the simulated toy data shipped in the reference tree.

Before rerunning anything in this group, check `../EXPERIMENT_REGISTRY.md` for the exact completed settings keys.

## Folder Map

| Folder | Description |
|---|---|
| `14_b3_sisc_simulated_replication/` | Initial B.3 end-to-end run with `max_iters=10`, including Fig. 8/Fig. 9 style plots. |
| `15_b3_sisc_simulated_replication_max20/` | Stricter B.3 run with `max_iters=20`. |
| `16_b3_sisc_sweep/` | Timed sweep over seeds, `l_max=20/21`, and iteration counts; includes strict boundary Jaccard and reference-code-style interval IoU. |

## Paper Targets

| Experiment | Paper per-unit DTW | Paper Jaccard / IoU |
|---|---:|---:|
| Fig. 8 one-pattern | 0.009 | 0.938 |
| Fig. 9 multi-pattern | 0.01 | 0.784 |

## Best Observed Multi-Pattern Sweep Result

| Metric | Paper | Best observed |
|---|---:|---:|
| Average per-unit DTW | 0.01 | 0.0321 |
| Strict boundary Jaccard, tolerance 2 | 0.784 | 0.3608 |
| Reference-code-style interval IoU | 0.784 | 0.6418 |

## Takeaway

The available reference code/data is not sufficient to exactly reproduce Appendix B.3. The reference tree contains the multi-pattern toy CSVs and generic SISC/evaluation helpers, but not the exact one-pattern data, synthetic generator, standard pattern arrays, random seeds, or a B.3 reproduction script that yields the paper numbers.
