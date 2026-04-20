# FTS-Diffusion (Reference Implementation)

This repository implements the paper's FTS-Diffusion pipeline for financial time series generation and evaluation.

## 1) Codebase Summary

### Top-level flow
- `main.py`: Calls training then experiments (`train_all()` then `run_experiments()`).
- `train_all.py`: Fetches S&P500 data and trains FTS-Diffusion modules.
- `experiments_all.py`: Runs qualitative, quantitative, and downstream experiments.

### Core model components (`models/`)
- `pattern_recognition_module.py`: SISC segmentation/clustering (pattern recognition).
- `pattern_generation_module.py` + `pattern_conditioned_diffusion.py` + `scaling_autoencoder.py`: pattern-conditioned generation.
- `pattern_evolution_module.py`: next-pattern/length/magnitude state evolution.
- `train_ftsdiffusion.py`: full training pipeline (recognition + generation + evolution).
- `sampling.py`: synthetic time-series generation from trained modules.
- `load_models.py`: deterministic model filenames and loading logic.
- `model_params.py`: default hyperparameters.

### Experiments (`experiments/`)
- `qual_analysis.py` + `stylized_facts.py`: stylized facts plots (heavy tails, QQ, autocorrelation).
- `quant_analysis.py` + `metrics_quant.py`: KS/AD distribution tests.
- `tmtr.py`: Train-on-Mixture-Test-on-Real downstream experiment.
- `tatr.py`: Train-on-Augmentation-Test-on-Real downstream experiment.
- `predictor_lstm.py`: LSTM forecaster for downstream tasks.
- `utils_downstream.py`: dataset setup, metrics aggregation, and plotting.

### Utilities (`utils/`)
- `load_data.py`: Yahoo Finance download + loading segmented artifacts.
- `prepare_data.py`: split and package segment-level train/test data.
- `markov_processing.py`, `series_processing.py`, `metrics_clustering.py`, etc.: preprocessing and clustering helpers.

## 2) Reproduce Paper Results

## Environment
Use Python 3.10 in conda (as you already did).

```bash
pip install -r requirements.txt
mkdir -p trained_models res figs data
```

## Step A: Download the target time series
```bash
python -c "from utils.load_data import get_fts; get_fts(ticker='^GSPC', fts_name='sp500', start_date='1980-01-01', end_date='2020-01-01')"
```

This writes:
- `data/sp500_timeseries.csv`

## Step B: Train FTS-Diffusion (all modules)
```bash
python - <<'PY'
from models.model_params import prm_params
from utils.load_data import load_actual_fts
from models.train_ftsdiffusion import train_ftsdiffusion

# Needed by recognition training in current code path
prm_params.setdefault('max_iters', 100)

fts = load_actual_fts(prm_params['dataname']).squeeze()
train_ftsdiffusion(fts, store_model=True)
PY
```

This produces:
- SISC artifacts in `res/` (centroids, labels, subsequences, segmentation)
- Trained model checkpoints in `trained_models/`

## Step C: Qualitative analysis (stylized facts)
```bash
python -c "from experiments.qual_analysis import qualitative_analysis; qualitative_analysis(store_res=True)"
```

Primary artifact:
- `figs/stylized_fact.pdf`

## Step D: Quantitative distribution tests
```bash
python -c "from experiments.quant_analysis import quantitative_analysis; quantitative_analysis()"
```

Output is printed to stdout as a dataframe (KS/AD mean and std).

## Step E: Downstream experiments (TMTR and TATR)
```bash
python - <<'PY'
from experiments.tmtr import TMTR
from experiments.tatr import TATR

# Paper-like defaults from downstream wrappers
TMTR(datatype='prices', n_runs=100, n_proportions=10, mix_length=252*5,
     n_epochs=100, window_size=64, ahead=1,
     lstm_hidden_dim=32, lstm_loss='mae', store_res=True)

TATR(datatype='prices', n_runs=100, n_augmentations=100, aug_length=252,
     n_epochs=100, window_size=64, ahead=1,
     lstm_hidden_dim=32, lstm_loss='mae', store_res=True)
PY
```

Primary artifacts:
- `res/res_tmtr_sp500-prices_1ahead_h32_mae.csv`
- `res/res_tatr_sp500-prices_1ahead_h32_mae.csv`

## Step F: Aggregate and plot downstream results
```bash
python - <<'PY'
from models.model_params import prm_params
from experiments.utils_downstream import summarize_results, plot_dowmstream_tmtr, plot_dowmstream_tatr

dataname = prm_params['dataname']

df_tmtr = summarize_results('tmtr', dataname, 1, 32, 'mae')
plot_dowmstream_tmtr(df_tmtr, store_fig=True)

df_tatr = summarize_results('tatr', dataname, 1, 32, 'mae')
plot_dowmstream_tatr(df_tatr, 252, store_fig=True)
PY
```

Primary artifacts:
- `res/res_tmtr_summary.csv`
- `res/res_tatr_summary.csv`
- `figs/res_tmtr.pdf`
- `figs/res_tatr.pdf`

## Notes on reproducibility
- Seeds are set in multiple files (`SEED=0` or `SEED=42`), but deep learning runs can still vary by hardware/backend.
- Runtime is substantial for full replication (`n_runs=100` in TMTR/TATR).
- GPU is used automatically if available (`cuda:0`), otherwise CPU.

## Optional quick smoke test (fast, non-paper settings)
Use smaller downstream settings before full runs:
```bash
python - <<'PY'
from experiments.tmtr import TMTR
from experiments.tatr import TATR
TMTR(datatype='prices', n_runs=2, n_proportions=2, mix_length=252, n_epochs=2, store_res=False)
TATR(datatype='prices', n_runs=2, n_augmentations=2, aug_length=50, n_epochs=2, store_res=False)
PY
```
