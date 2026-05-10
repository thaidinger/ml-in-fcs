# Drift Component

This folder contains the predictable-drift diagnostic as a standalone component
for FTS-Diffusion outputs. It is separated from the original repo code so
someone with the Colab artifacts or more compute can run it without touching the
training implementation.

For the full mathematical framing and report-ready explanation, see
`DRIFT_COMPONENT_README.md`.

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
`return`, `returns`, `r`, `close`, `value`, `generated`, `synthetic`.

The `results/tatr/**/results_matrix.csv` files are benchmark summaries, not
ordered generated return paths, so they should not be used as synthetic samples
for this diagnostic.

## Run Direct Evaluation

```bash
python drift_component/scripts/evaluate_predictable_drift.py \
  --real-csv data/asset.csv \
  --synthetic-csv outputs/default/samples/generated.csv \
  --config configs/default.yaml \
  --output-dir outputs/default/drift_diagnostics
```

For price data:

```bash
python drift_component/scripts/evaluate_predictable_drift.py \
  --real-csv fts-diffusion-ref/data/sp500_timeseries.csv \
  --synthetic-csv outputs/default/samples/generated.csv \
  --input-type close \
  --synthetic-input-type return \
  --output-dir outputs/default/drift_diagnostics
```

## Run Light Experiment

This searches for generated CSVs under `outputs/`. If none are found, it runs a
toy smoke experiment so the pipeline can still be validated.

```bash
python drift_component/scripts/run_light_drift_experiment.py \
  --run-dir outputs/default \
  --output-dir outputs/default/drift_diagnostics \
  --max-synthetic 5 \
  --null-reps 20
```

## Outputs

The evaluator writes:

- `drift_summary.csv`
- `drift_components.csv`
- `stylized_summary.csv`
- `null_summary.csv`
- `metadata.json`
- `report.md`

The report table includes:

```text
source_type | sample_id | n_obs | pooled_delta | rolling_delta_mean |
rolling_delta_std | acf_abs_r_lag1 | excess_kurtosis
```

Lower delta means less predictable drift under the fixed feature map; this is
not a full martingale/no-arbitrage proof.
