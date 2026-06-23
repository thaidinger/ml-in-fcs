# NeurIPS End-to-End Diagnostics

## Real Asset Drift Null Calibration

| asset | n_test | real_delta | null_q05 | null_q50 | null_q95 | inside_null_90 | acf_abs_r_lag1 | excess_kurtosis |
| ----- | ------ | ---------- | -------- | -------- | -------- | -------------- | -------------- | --------------- |
| sp500 | 2018   | 0.0297     | 0.0222   | 0.0260   | 0.0303   | True           | 0.2165         | 3.2110          |
| goog  | 755    | 0.0575     | 0.0403   | 0.0511   | 0.0629   | True           | 0.0935         | 6.1142          |
| zcf   | 953    | 0.0173     | 0.0150   | 0.0202   | 0.0259   | True           | 0.1250         | 2.0247          |

## Hidden-Drift Predictive Alpha

| sample_id          | n_seeds | pooled_delta_mean | oos_corr_mean | direction_acc_mean | strategy_sharpe_mean | acf_r_lag1_mean |
| ------------------ | ------- | ----------------- | ------------- | ------------------ | -------------------- | --------------- |
| iid_signs          | 500     | 0.0941            | 0.0036        | 0.5020             | 0.0853               | -0.0029         |
| markov_signs_p0.65 | 500     | 0.2088            | 0.1482        | 0.5968             | 2.3338               | 0.1740          |
| markov_signs_p0.75 | 500     | 0.3435            | 0.2786        | 0.6952             | 4.7966               | 0.2892          |
| markov_signs_p0.85 | 500     | 0.5126            | 0.4112        | 0.7976             | 7.7221               | 0.4039          |
| markov_signs_p0.90 | 500     | 0.6164            | 0.4811        | 0.8483             | 9.4274               | 0.4613          |
| real_eval          | 500     | 0.0976            | 0.0004        | 0.5025             | 0.0701               | 0.0002          |

Across all controlled samples, the correlation between pooled delta and out-of-sample strategy Sharpe is `0.6921`.

## Stored End-to-End TATR Protocol Evidence

| stage                          | protocol                      | ahead | best_pct_mean | final_pct_mean | synthetic_last_mean | synthetic_price_std |
| ------------------------------ | ----------------------------- | ----- | ------------- | -------------- | ------------------- | ------------------- |
| tatr_continuous_chunked        | continuous_chunked            | 1     | -84.8547      | -65.2051       | 7085.0674           | 1701.5287           |
| tatr_continuous_chunked_5ahead | continuous_chunked            | 5     | -85.1853      | -77.4552       | 7085.0674           | 1701.5287           |
| tatr_continuous_refit          | continuous_cross_refit_scaler | 1     | -85.4196      | -68.2497       | 7085.0674           | 1701.5287           |
| tatr_independent_fixed         | independent_fixed             | 1     | 0.0000        | 193.2016       | 1248.8246           | 24.5283             |
