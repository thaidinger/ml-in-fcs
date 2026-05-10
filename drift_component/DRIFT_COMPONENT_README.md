# Predictable-Drift Component for FTS-Diffusion

## Central Point

FTS-Diffusion is good at matching the unconditional appearance of financial
returns, but it does not mathematically guarantee that generated returns satisfy
a basic financial condition: no predictable drift under a fixed information set.

In short:

```text
Matching stylised facts does not imply E[r_{t+1} | F_t] = 0.
```

The original FTS-Diffusion pipeline learns realistic-looking financial paths
through:

```text
SISC / DTW pattern recognition
-> conditional diffusion pattern generation
-> pattern evolution through a transition mechanism
```

Those modules target realistic motifs, distributions, volatility clustering,
and useful synthetic augmentation. They do not directly impose a martingale-
difference-style condition on returns. A generated series can have realistic
tails and volatility clustering while still containing artificial conditional
mean predictability.

This component implements the supervisor PDF's Step 2 diagnostic and provides a
conservative Step 3 penalty utility.

## Mathematical Diagnostic

Let `r_t` be returns. Fix the information vector:

```text
omega_t = (1, r_t, r_{t-1}, |r_t|, |r_{t-1}|)
```

Consider linear predictors:

```text
g_b(F_t) = b^T omega_t,    ||b||_2 <= 1
```

The predictable-drift violation is:

```text
delta = sup_{||b||_2 <= 1} | E[r_{t+1} b^T omega_t] |
```

Let:

```text
m = E[r_{t+1} omega_t]
```

Then by Cauchy-Schwarz:

```text
delta = ||m||_2
```

The empirical estimator implemented here is:

```text
m_hat = mean_t r_{t+1} omega_t
delta_hat = ||m_hat||_2
```

The component moments are also reported:

```text
mean r_{t+1}
mean r_{t+1} r_t
mean r_{t+1} r_{t-1}
mean r_{t+1} |r_t|
mean r_{t+1} |r_{t-1}|
```

Each component has a simple interpretation:

- `mean r_{t+1}` checks unconditional drift.
- `mean r_{t+1} r_t` checks first-lag return predictability.
- `mean r_{t+1} r_{t-1}` checks second-lag return predictability.
- `mean r_{t+1} |r_t|` checks whether recent volatility predicts next-return
  mean or sign.
- `mean r_{t+1} |r_{t-1}|` checks the same using the previous lag.

## Standardization

By default, the diagnostic is computed on standardized returns:

```text
z_t = (r_t - train_mean) / train_std
```

`train_mean` and `train_std` are computed from the chronological training split
only. This avoids leaking validation/test information into the diagnostic.

The constant feature remains equal to `1`.

## What To Compare

The supervisor PDF asks for the diagnostic on:

```text
real data
FTS-Diffusion synthetic data
simple null / block-shuffled real data
```

This component reports all three when generated samples are available. The null
uses block-shuffled real returns: within-block order is preserved, and the block
order is shuffled with a deterministic seed.

The report also includes stylised-fact summaries:

- mean, standard deviation, skew, excess kurtosis
- min, max, quantiles
- autocorrelation of `r_t`
- autocorrelation of `|r_t|`

The intended interpretation is modest:

```text
Lower delta means less predictable drift under this fixed feature map.
This is not a full martingale or no-arbitrage proof.
```

## Why This Targets A Real Limitation

Distributional tests such as KS or AD can say that two return samples have
similar marginal distributions. They do not rule out conditional drift.

For example:

```text
r_{t+1} = rho r_t + epsilon_{t+1}
```

can have a realistic-looking marginal distribution while still satisfying:

```text
E[r_{t+1} | r_t] = rho r_t
```

If `rho` is nonzero, tomorrow's return is predictable from today's return. A
marginal distribution test may miss this, but the drift diagnostic detects it
through the `mean r_{t+1} r_t` component.

This matters for generative finance because downstream prediction improvements
are not automatically proof of financial validity. They may indicate useful
augmentation, but they may also reflect artificial predictable structure.

## Optional Penalty Extension

The natural penalty suggested by the diagnostic is:

```text
L_new = L_FTS + lambda * delta_hat^2
```

where:

```text
delta_hat^2 =
|| mean_t r_{t+1} (1, r_t, r_{t-1}, |r_t|, |r_{t-1}|) ||_2^2
```

A defensible empirical claim would be:

```text
Adding one fixed predictable-drift penalty reduces synthetic drift under a
pre-specified diagnostic while preserving the main stylised facts of financial
returns.
```

This component includes:

```text
drift_component/src/fts_diffusion/training/drift_penalty.py
drift_component/configs/drift_penalty.yaml
```

The penalty is utility-only in this checkout. It is not wired into the original
training pipeline because the training source files are not present here. It
should not be computed across randomly shuffled mini-batches and called the true
Alex statistic. If used on ordered reconstructed segments, log it separately as
a proxy, for example `drift_proxy_loss`.

## Validation Rule For A Full Step 3 Experiment

If someone with compute integrates the penalty into training, use a fixed
validation rule:

```text
lambda in {0, 1e-4, 1e-3, 1e-2, 1e-1, 1}
```

For each `lambda`, train using training data only and compute on validation:

```text
delta_hat_val(lambda)
stylised-fact fidelity(lambda)
```

Choose:

```text
lambda* = argmin delta_hat_val(lambda)
```

subject to a pre-specified fidelity constraint, such as no more than 5-10%
degradation in tail/ACF/distributional metrics relative to the baseline.

Then apply `lambda*` once to the final hold-out and, if possible, to a second
asset. Do not select `lambda` using the final hold-out.

## Repository Layout

```text
drift_component/
  README.md                         quick start
  DRIFT_COMPONENT_README.md          conceptual and mathematical framing
  docs/predictable_drift.md          compact method note
  configs/drift_penalty.yaml         disabled penalty config snippet
  requirements.txt                   lightweight dependencies
  pyproject.toml                     standalone package metadata
  scripts/evaluate_predictable_drift.py
  scripts/run_light_drift_experiment.py
  src/fts_diffusion/evaluation/drift.py
  src/fts_diffusion/evaluation/stylized.py
  src/fts_diffusion/training/drift_penalty.py
  tests/
```

## Install

From the repository root:

```bash
python -m venv .venv-drift
source .venv-drift/bin/activate
pip install -r drift_component/requirements.txt
```

Or with `uv`:

```bash
uv run --python 3.11 --with numpy --with pandas --with pyyaml --with pytest \
  python -m pytest drift_component/tests -q
```

## Expected Inputs

The evaluator needs:

- a real CSV with ordered returns or prices
- one or more synthetic CSVs with ordered generated returns or prices

Accepted value columns are inferred from common names:

```text
return, returns, r, close, value, generated, synthetic
```

The `results/tatr/**/results_matrix.csv` files are benchmark summaries, not
ordered generated return paths. They should not be used as synthetic samples for
this diagnostic.

## Direct Evaluation

For real returns and generated returns:

```bash
python drift_component/scripts/evaluate_predictable_drift.py \
  --real-csv data/asset.csv \
  --synthetic-csv outputs/default/samples/generated.csv \
  --config configs/default.yaml \
  --input-type return \
  --synthetic-input-type return \
  --output-dir outputs/default/drift_diagnostics
```

For real prices and generated returns:

```bash
python drift_component/scripts/evaluate_predictable_drift.py \
  --real-csv fts-diffusion-ref/data/sp500_timeseries.csv \
  --synthetic-csv outputs/default/samples/generated.csv \
  --input-type close \
  --synthetic-input-type return \
  --output-dir outputs/default/drift_diagnostics
```

## Light Experiment Wrapper

This searches for existing generated samples under:

```text
outputs/**/samples/*.csv
outputs/**/*generated*.csv
outputs/**/*sample*.csv
```

If no generated samples are found, it runs a toy smoke experiment with:

- iid Gaussian returns
- AR(1) returns
- block-shuffled iid null

Run:

```bash
python drift_component/scripts/run_light_drift_experiment.py \
  --run-dir outputs/default \
  --output-dir outputs/default/drift_diagnostics \
  --max-synthetic 5 \
  --null-reps 20
```

## Outputs

The evaluator writes:

```text
drift_summary.csv
drift_components.csv
stylized_summary.csv
null_summary.csv
metadata.json
report.md
```

The report table includes:

```text
source_type | sample_id | n_obs | pooled_delta | rolling_delta_mean |
rolling_delta_std | acf_abs_r_lag1 | excess_kurtosis
```

## Report Framing

A concise contribution statement:

```text
We identify that FTS-Diffusion matches several unconditional properties of
financial returns but does not explicitly control conditional predictable drift.
We therefore add a fixed martingale-style diagnostic and a conservative
drift-penalty utility to test whether synthetic paths can remain realistic while
avoiding artificial predictability.
```

The main limitation statement:

```text
This diagnostic checks a fixed finite-dimensional implication of the
martingale-difference condition. It does not prove full no-arbitrage.
```

