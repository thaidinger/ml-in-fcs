# Sign-Drift Sweep

Magnitude paths are fixed within each seed; only signs change.

| sample_id          | n_seeds | pooled_delta_mean | pooled_delta_std | acf_r_lag1_mean | acf_abs_r_lag1_mean | signed_stylized_distance_mean | magnitude_path_distance_mean |
| ------------------ | ------- | ----------------- | ---------------- | --------------- | ------------------- | ----------------------------- | ---------------------------- |
| iid_signs          | 100     | 0.1005            | 0.0715           | -0.0055         | 0.1668              | 0.8097                        | 0.0000                       |
| markov_signs_p0.55 | 100     | 0.1258            | 0.0933           | 0.0607          | 0.1668              | 0.7158                        | 0.0000                       |
| markov_signs_p0.65 | 100     | 0.2152            | 0.1289           | 0.1690          | 0.1668              | 0.7253                        | 0.0000                       |
| markov_signs_p0.75 | 100     | 0.3717            | 0.2297           | 0.2947          | 0.1668              | 0.8335                        | 0.0000                       |
| markov_signs_p0.85 | 100     | 0.5383            | 0.2660           | 0.4036          | 0.1668              | 0.7499                        | 0.0000                       |
| markov_signs_p0.90 | 100     | 0.6604            | 0.3686           | 0.4650          | 0.1668              | 0.9535                        | 0.0000                       |
| real_eval          | 100     | 0.1049            | 0.0801           | 0.0000          | 0.1668              | 0.0000                        | 0.0000                       |

## Paired p=0.85 comparison

```json
{
  "delta_ratio_mean": 6.406928582344851,
  "delta_ratio_median": 5.784435848946543,
  "hidden_acf_r_lag1_mean": 0.4035516026620959,
  "hidden_delta_gt_iid_count": 100,
  "hidden_delta_mean": 0.5383291249287918,
  "hidden_sample_id": "markov_signs_p0.85",
  "hidden_signed_distance_mean": 0.7499310086251297,
  "iid_acf_r_lag1_mean": -0.005506039708128768,
  "iid_delta_mean": 0.10045498551347969,
  "iid_signed_distance_mean": 0.8096553555740081,
  "n_pairs": 100
}
```
