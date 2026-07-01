# Predictable-Drift Diagnostic

This repository includes a lightweight diagnostic for Alex's predictable-drift
extension. For returns \(r_t\), define the fixed feature map

```text
omega_t = (1, r_t, r_{t-1}, |r_t|, |r_{t-1}|)
```

and linear predictors \(g_b(F_t) = b^T omega_t\), with \(\|b\|_2 <= 1\).
The empirical drift violation has the closed form

```text
delta_hat = sup ||b||_2<=1 | mean_t r_{t+1} b^T omega_t |
          = || mean_t r_{t+1} omega_t ||_2
```

The implementation reports `delta_hat` and these component moments:

```text
mean r_{t+1}
mean r_{t+1} r_t
mean r_{t+1} r_{t-1}
mean r_{t+1} |r_t|
mean r_{t+1} |r_{t-1}|
```

By default, diagnostics are computed on standardized returns

```text
z_t = (r_t - train_mean) / train_std
```

where `train_mean` and `train_std` come from the chronological training split
only. The constant feature remains equal to 1.

The null comparison uses block-shuffled returns. Contiguous blocks preserve
within-block order, including the final shorter remainder block, and the block
order is shuffled with a deterministic seed. This preserves the return multiset
and some local dependence while disrupting longer chronological alignment.

Run the direct evaluator with real and generated CSVs:

```bash
python drift_component/scripts/evaluate_predictable_drift.py \
  --real-csv data/asset.csv \
  --synthetic-csv outputs/default/samples/generated.csv \
  --input-type return \
  --synthetic-input-type return \
  --output-dir outputs/default/drift_diagnostics
```

Run the light experiment wrapper:

```bash
python drift_component/scripts/run_light_drift_experiment.py \
  --run-dir outputs/default \
  --output-dir outputs/default/drift_diagnostics \
  --max-synthetic 5 \
  --null-reps 20
```

The report table includes real, synthetic, and block-shuffled null rows where
available. Lower delta means less predictable drift under this fixed feature
map; this checks a finite-dimensional implication of no predictable drift, not
a full martingale/no-arbitrage proof.

The optional `drift_penalty` config block is disabled by default. In this
checkout, the PyTorch penalty is provided as a utility only because the training
pipeline source is not present. If it is later wired into training, it should be
used only on correctly ordered return sequences or clearly logged as a
within-segment reconstruction proxy such as `drift_proxy_loss`.
