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
| iid_signs          | 100     | 0.1005            | 0.0017        | 0.5003             | 0.0822               | -0.0055         |
| markov_signs_p0.65 | 100     | 0.2152            | 0.1393        | 0.5909             | 2.1476               | 0.1690          |
| markov_signs_p0.75 | 100     | 0.3717            | 0.2845        | 0.6925             | 4.7842               | 0.2947          |
| markov_signs_p0.85 | 100     | 0.5383            | 0.4137        | 0.7945             | 7.6808               | 0.4036          |
| markov_signs_p0.90 | 100     | 0.6604            | 0.4886        | 0.8526             | 9.6466               | 0.4650          |
| real_eval          | 100     | 0.1049            | 0.0035        | 0.5021             | 0.0980               | 0.0000          |

Across all controlled samples, the correlation between pooled delta and out-of-sample strategy Sharpe is `0.6414`.

## Stored End-to-End TATR Protocol Evidence

| stage                          | protocol                      | ahead | best_pct_mean | final_pct_mean | synthetic_last_mean | synthetic_price_std |
| ------------------------------ | ----------------------------- | ----- | ------------- | -------------- | ------------------- | ------------------- |
| tatr_continuous_chunked        | continuous_chunked            | 1     | -84.8547      | -65.2051       | 7085.0674           | 1701.5287           |
| tatr_continuous_chunked_5ahead | continuous_chunked            | 5     | -85.1853      | -77.4552       | 7085.0674           | 1701.5287           |
| tatr_continuous_refit          | continuous_cross_refit_scaler | 1     | -85.4196      | -68.2497       | 7085.0674           | 1701.5287           |
| tatr_independent_fixed         | independent_fixed             | 1     | 0.0000        | 193.2016       | 1248.8246           | 24.5283             |
