# Drift-Incorporated Selection Benchmark

Candidate augmentations share the same magnitude path within each seed; only sign dynamics differ.
The drift-penalized selector uses the same marginal stylized score as the baseline plus a penalty above the real block-shuffle drift band.

## Selector Summary

| selector        | n_seeds | delta_mean | null_violation_rate | high_persistence_rate | stylized_score_mean | normalized_mse_mean | excess_mse_vs_real_only_mean | coef_norm_mean |
| --------------- | ------- | ---------- | ------------------- | --------------------- | ------------------- | ------------------- | ---------------------------- | -------------- |
| oracle_test_mse | 1000    | 0.0999     | 0.4150              | 0.0050                | 0.2135              | 0.9979              | -0.0319                      | 0.0950         |
| drift_penalized | 1000    | 0.0958     | 0.3460              | 0.0020                | 0.0520              | 1.0087              | -0.0211                      | 0.1066         |
| stylized_only   | 1000    | 0.2780     | 0.7600              | 0.3930                | 0.0344              | 1.0452              | 0.0154                       | 0.1988         |
| random          | 1000    | 0.3400     | 0.8480              | 0.5020                | 0.3106              | 1.0579              | 0.0281                       | 0.2263         |

## Paired Normalized-MSE Comparisons

| comparison                          | n_pairs | win_rate | mean_normalized_mse_reduction | median_normalized_mse_reduction | q05_reduction | q95_reduction |
| ----------------------------------- | ------- | -------- | ----------------------------- | ------------------------------- | ------------- | ------------- |
| stylized_only_minus_drift_penalized | 1000    | 0.6080   | 0.0365                        | 0.0131                          | 0.0000        | 0.1400        |
| stylized_only_minus_oracle_test_mse | 1000    | 0.9760   | 0.0473                        | 0.0267                          | 0.0017        | 0.1521        |
| stylized_only_minus_random          | 1000    | 0.3840   | -0.0127                       | -0.0093                         | -0.1256       | 0.1082        |

## Selected Candidate Counts

| selector        | sample_id | count | share  |
| --------------- | --------- | ----- | ------ |
| drift_penalized | iid       | 531   | 0.5310 |
| drift_penalized | p0.75     | 2     | 0.0020 |
| oracle_test_mse | iid       | 577   | 0.5770 |
| oracle_test_mse | p0.75     | 3     | 0.0030 |
| oracle_test_mse | p0.85     | 1     | 0.0010 |
| oracle_test_mse | p0.90     | 1     | 0.0010 |
| random          | iid       | 139   | 0.1390 |
| random          | p0.75     | 154   | 0.1540 |
| random          | p0.85     | 185   | 0.1850 |
| random          | p0.90     | 163   | 0.1630 |
| stylized_only   | iid       | 214   | 0.2140 |
| stylized_only   | p0.75     | 157   | 0.1570 |
| stylized_only   | p0.85     | 125   | 0.1250 |
| stylized_only   | p0.90     | 111   | 0.1110 |
