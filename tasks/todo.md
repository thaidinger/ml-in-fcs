# TMTR Reduced Replication notebook — plan

## Goal

Create `src/TMTR_Reduced_Replication.ipynb`, a standalone notebook mirroring the
style of `src/TATR_Reduced_Replication.ipynb`, that runs the **TMTR** (Train on
Mixture, Test on Real) downstream experiment from the FTS-Diffusion paper.

## Decisions (already approved)

1. **Protocols supported:** `authors`, `split`, `random_init` (no `single`, no
   `burn_in` — those were experimental for TATR only).
2. **Sweep:** authors-style — fixed `MIX_LENGTH = 252*5` (5 years), proportion
   sweep `{0%, 10%, …, 100%}`. No variable mix-length sweep.
3. **Storage:** results on Google Drive, same `PERSIST_ROOT` as TATR
   (`/content/drive/MyDrive/fts_diffusion`). Naming convention:
     - `synthetic/{ASSET}/k{K}/{PROTOCOL}/run_XX_*.npy`
     - `results/tmtr/{ASSET}/k{K}/{PROTOCOL}/run_XX.csv`
     - `figures/tmtr/{ASSET}/k{K}/{PROTOCOL}/{live,final}.{png,pdf}`

## TMTR semantics (from `fts-diffusion-ref/experiments/tmtr.py`)

```
real_timeseries = downstream[:MIX_LENGTH]      # first 5y (anchor pool)
test_timeseries = downstream[MIX_LENGTH:]      # held-out (~rest of history)

for proportion in [0.0, 0.1, ..., 1.0]:
    syn_length  = int(MIX_LENGTH * proportion)
    real_length = MIX_LENGTH - syn_length
    mix = sample(real_timeseries, real_length) ⊕ sample(syn_timeseries, syn_length)
    train LSTM on mix
    MAPE on test_timeseries
```

Total training-set length is **constant** (= MIX_LENGTH); only the *composition*
varies. This is the key difference from TATR (where total length grows).

## Notebook structure (mirrors TATR cell-by-cell)

| § | TATR section | TMTR adaptation |
|---|---|---|
| 0 | Title / overview markdown | Update title to "TMTR — Reduced Replication"; describe TMTR semantics |
| 1 | Colab/Drive mount + clone | **Identical** (copy as-is) |
| 2 | Imports + `EXPERIMENT CONFIG` | Set `EXPERIMENT='tmtr'`; **drop** `EVAL_YEARS`, **add** `MIX_LENGTH`, `PROPORTIONS=[0,10,…,100]`, `N_RUNS` |
| 3 | Phase 1 — SISC fit / load architectures | **Identical** (TMTR reuses the same trained FTS-Diffusion checkpoints) |
| 4 | Phase 2 — TATR loop | **Replaced** by Phase 2 — TMTR loop (see below) |
| 5 | Aggregation / summary CSVs | Adapted: x-axis is `proportion` instead of `eval_year`; output `summary.csv` and `results_matrix.csv` |
| 6 | Plots (paper-faithful + diagnostic enhanced) | Adapted: x = "Syn. Prop. (%)", y = "MAPE", baseline = `MAPE@0%`, comparison fig across protocols |

## Phase 2 — TMTR loop (new code)

### Setup function
```python
def setup_dowmstream_tmtr_adaptive(window_size, mix_length=MIX_LENGTH):
    """Returns (real_timeseries, test_timeseries, scaler, init_state, init_segment).
    Mirrors authors' setup_dowmstream_tmtr() but works for short assets too."""
    downstream_ts, segments_test, labels_test, lengths_test = get_downstream_data()
    assert len(downstream_ts) > mix_length + WINDOW_SIZE, "Asset too short for MIX_LENGTH"
    real_ts = downstream_ts[:mix_length]
    test_ts = downstream_ts[mix_length:]
    _, scaler = Timeseries2Dataset_Downstream(test_ts, window_size)
    test_dataset = Timeseries2Dataset_Downstream(test_ts, window_size, scaler)
    init_state, init_segment = init_first_segment(segments_test, labels_test, lengths_test)
    return real_ts, test_ts, test_dataset, scaler, init_state, init_segment
```

For GOOG / ZC=F (shorter histories), assert `len(downstream) > MIX_LENGTH + W`;
if not, fall back to a smaller `MIX_LENGTH` per-asset (logged clearly).

### Per-run function
```python
def run_single_tmtr(run_idx, proportions, seed_base=42, protocol=None):
    """One TMTR replicate.
       authors:      fresh MIX_LENGTH-day synthetic series per run, single shot
       split:        ONE long continuous trajectory (length N_RUNS*MIX_LENGTH or pre-budgeted),
                     this run takes a non-overlapping window of MIX_LENGTH days
       random_init:  fresh MIX_LENGTH-day synthetic series, init_segment sampled at random
                     from SEGMENTS_INIT (same distribution as TATR random_init)
    """
    ...
    for proportion_idx, prop in enumerate(proportions):
        syn_len  = int(MIX_LENGTH * prop / 100)
        real_len = MIX_LENGTH - syn_len
        mix_dataset = build_mix_dataset(real_ts, syn_ts, real_len, syn_len, scaler)
        mape = train_and_evaluate(mix_dataset, test_dataset, scaler, ...)
        results[prop] = mape
    save run_csv
```

### Per-protocol synthetic generation
- **authors**: call `generate_timeseries_ftsdiffusion(MIX_LENGTH, init_state, init_segment)` once per run.
- **split**: pre-generate ONE continuous trajectory of length `N_RUNS * MIX_LENGTH` and save to `SYN_DIR/continuous.npy`; run `r` reads slice `[r*MIX_LENGTH : (r+1)*MIX_LENGTH]`. Mirrors TATR `split` but sliced by run index, not by year.
- **random_init**: re-sample `init_segment` from `SEGMENTS_INIT` (using `seed_base + run_idx`) before each run, then generate `MIX_LENGTH` days.

### Mixing helper
```python
def build_mix_dataset(real_ts, syn_ts, real_len, syn_len, scaler):
    if real_len == MIX_LENGTH:
        return create_syn_dataset(real_ts[:real_len], WINDOW_SIZE, scaler, DATATYPE)
    if syn_len == MIX_LENGTH:
        return create_syn_dataset(syn_ts[:syn_len],  WINDOW_SIZE, scaler, DATATYPE)
    real_chunk = real_ts[:real_len]                  # first real_len days of anchor
    syn_chunk  = syn_ts[:syn_len]                    # first syn_len days of synthetic
    real_ds = create_syn_dataset(real_chunk, WINDOW_SIZE, scaler, DATATYPE)
    syn_ds  = create_syn_dataset(syn_chunk,  WINDOW_SIZE, scaler, DATATYPE)
    return concat_datasets_downstream(real_ds, syn_ds)
```

(Reuses existing helpers from the TATR notebook so we don't reinvent dataset
plumbing.)

## Aggregation & plots

- `summary.csv`: columns `[proportion_pct, mape_mean, mape_lo, mape_hi, n_runs]`
- `results_matrix.csv`: rows = runs, cols = proportions
- **Live plot** (during the loop): MAPE vs Syn. Prop. (%), one curve, with running CI band
- **Final paper-faithful plot**: matches `plot_dowmstream_tmtr` from
  `utils_downstream.py` (single curve + min/max band, dashed baseline at 0%)
- **Final enhanced plot**: bootstrap CI band, baseline at 0%, % diff annotation
- **Cross-protocol comparison** (NEW): if results for ≥2 protocols exist on Drive,
  overlay them on a single plot (authors vs split vs random_init)

## Files touched

- **NEW**: `src/TMTR_Reduced_Replication.ipynb` (~1500 lines, mirrors TATR)
- **NO changes** to existing TATR notebook
- **NO changes** to `fts-diffusion-ref/` (read-only reference)
- Auto-creates Drive directories under `PERSIST_ROOT/results/tmtr/...`,
  `PERSIST_ROOT/synthetic/...`, `PERSIST_ROOT/figures/tmtr/...`

## Acceptance criteria

- [ ] Notebook opens in Jupyter / Colab without JSON errors
- [ ] Cell order matches TATR notebook section-by-section
- [ ] `EXPERIMENT='tmtr'` is hardcoded; `PROTOCOL ∈ {authors, split, random_init}` switchable
- [ ] First end-to-end run on S&P 500 with `N_RUNS=2, n_proportions=4` produces
      a `summary.csv` and a `final.pdf` on Drive (smoke test, ~5 min)
- [ ] Re-running the same protocol detects existing `run_XX.csv` and skips
      completed runs (same idempotency as TATR)

## Open questions for the user (please confirm before I start writing)

1. **MIX_LENGTH per asset.** Paper uses `252*5 = 1260` days for S&P 500
   (full history ~10 088 days, so `test ≈ 35y`). For GOOG (~5k days) and
   ZC=F (~6k days), do you want:
   - (a) keep `MIX_LENGTH = 1260` (test will shrink to ~3.7y for GOOG, 4.5y for ZC=F);
   - (b) scale `MIX_LENGTH` proportionally (e.g. 0.25 × `len(downstream)`)?
   Recommendation: **(a)** — keeps the comparison directly to the paper.

2. **N_RUNS default.** TATR notebook uses `N_RUNS = 15`. Paper TMTR uses
   `n_runs = 20`. Suggest **15** for symmetry with our TATR results.

3. **Number of proportions.** Paper uses 11 points (`{0, 10, …, 100}`).
   Confirm **11** (gives ~5×5 = 25 min/run for 5y mix on a T4; ~6 h for
   `N_RUNS=15`).

4. **Plot style.** Want both "paper-faithful" (matches Liao et al. Fig. 6a)
   and "enhanced diagnostic" (with CI + baseline annotation), like TATR? Or
   only one of the two?

## Review

**Created:** `src/TMTR_Reduced_Replication.ipynb` (28 cells, 1266 lines of JSON).

### What was built
- 9 sections mirroring the TATR notebook order:
  Setup → Config → Phase 1 (FTS-Diffusion training/restore) → Phase 2 (TMTR
  loop) → Phase 3 (aggregation) → Phase 4 (figures) → Quality checks →
  Cross-protocol comparison → Resume guide.
- Three protocols implemented in `run_single_tmtr`:
  - `authors`: fresh `MIX_LENGTH=1260`-day synthetic per run, fixed `(init_state, init_segment)`
  - `split`: ONE shared continuous trajectory of length `N_RUNS × MIX_LENGTH`
    pre-generated to `SYN_DIR/_global_continuous.npy`; each run uses a
    non-overlapping slice
  - `random_init`: fresh 1260-day synthetic per run, init sampled from
    `SEGMENTS_INIT` (the SISC segments inside the real anchor)
- Paper-faithful TMTR LSTM hyperparameters (`hidden=64`, `epochs=200`,
  `loss=MSE`) — different from TATR which uses `hidden=32`, `100`, `MAE`
- Idempotent re-runs: per-run CSVs, partial-completion detection,
  proportion-level skip
- Drive layout: `results/tmtr/{ASSET}/k{K}/{PROTOCOL}/run_XX.csv`,
  shares `architectures/` with TATR
- Two final figures + cross-protocol overlay (matches TATR style)

### What was NOT built (out of scope per user)
- `single` and `burn_in` protocols (not requested)
- Variable `MIX_LENGTH` sweep (not requested)
- MC state-evolution analysis section (TATR-specific)

### Verification done
- JSON parses as valid `.ipynb` (28 cells, kernel `python3`)
- Every code cell parses as valid Python AST (0 syntax errors after
  stripping `%matplotlib` magic line)
- Escape handling for the AUTO-FIX patch cell verified manually
  (single-backslash `\n` inside Python string literals)

### Pending (cannot be tested without GPU / Drive)
- End-to-end smoke run on Colab with `N_RUNS=2` to confirm path/save logic
  works against the real Drive mount
- Confirm `get_downstream_data()` returns enough days for GOOG / ZC=F
  (assertion in `setup_dowmstream_tmtr_adaptive` will catch this at runtime
  with a clear error message)


# ============================================================================
# TimeGAN baseline (TATR + TMTR)
# ============================================================================

## Goal

Replicate the TATR and TMTR experiments from the FTS-Diffusion paper using
**TimeGAN (Yoon et al. 2019)** as the synthetic data generator, so we can
compare it head-to-head with FTS-Diffusion exactly as the paper authors did.

## Decisions (already approved)

1. **TimeGAN implementation**: `birdx0810/timegan-pytorch` (PyTorch port of
   the original Yoon et al. paper). Cloned at runtime in the notebooks.
2. **Notebook layout**: 2 separate notebooks
   (`TATR_TimeGAN_Replication.ipynb`, `TMTR_TimeGAN_Replication.ipynb`).
3. **Storage**: parallel to FTS-Diffusion on Drive, with a `generator` axis:
     - `synthetic/{ASSET}/timegan/run_XX_syn.npy`
     - `architectures/{ASSET}/timegan/model.pt`
     - `results/{tatr|tmtr}/{ASSET}/timegan/run_XX.csv`
     - `figures/{tatr|tmtr}/{ASSET}/timegan/{live,final}.{png,pdf}`
4. **Setup**: paper-faithful (univariate daily returns, window length = 21).

## Paper-faithful setup (from Appendix E.2)

- Input: univariate daily returns of closing prices (= `(p_t - p_{t-1}) / p_{t-1}`).
- Window length: **21** (= `l_max` of FTS-Diffusion's SISC).
- Train TimeGAN on sliding windows of 21 returns.
- Generate `T` days of synthetic prices: stitch `ceil(T/21)` independent
  21-day return windows, trim to T, then `prices = p_init * cumprod(1 + returns)`.

## TimeGAN protocols

TimeGAN does NOT have a Markov chain (each generated window comes from an
independent random `Z`), so the FTS-Diffusion protocols (authors / split /
random_init / burn_in) collapse into **one** TimeGAN protocol:
`timegan_default` = "stitch independent 21-day windows".

## TATR_TimeGAN_Replication.ipynb structure

1. Setup (Drive mount, clone our repo, clone `timegan-pytorch`, deps)
2. Configuration (ASSET, N_RUNS, EVAL_YEARS, TimeGAN hyperparams)
3. Phase 1 — Data loading + windowing
   - Load asset prices from Drive (shared with FTS-Diffusion notebooks)
   - Compute returns, build sliding 21-day windows
4. Phase 2 — Train TimeGAN (~30-60 min on A100)
   - Build `args` namespace, `model = TimeGAN(args)`, `timegan_trainer(...)`
   - Persist `args.pickle` and `model.pt` to Drive
5. Phase 3 — Synthetic data wrapper
   - `generate_timeseries_timegan(T, model, args, p_init)` → 1D price series
6. Phase 4 — TATR loop
   - Same shape as FTS-Diffusion TATR loop: per-run cache, idempotent,
     for `eval_year in EVAL_YEARS` build `init_dataset ⊕ syn[:eval_year*252]`,
     train LSTM, MAPE
7. Phase 5 — Aggregation + plots (paper-faithful + enhanced)
8. Phase 6 — Cross-generator comparison (FTS-Diffusion vs TimeGAN)

## TMTR_TimeGAN_Replication.ipynb structure

Same Phases 1–3, then:

4. Phase 4 — TMTR loop
   - For each run: generate `MIX_LENGTH = 252×5` days of synthetic prices once.
   - For each `proportion ∈ {0,…,100}%`:
     - mix = real[:real_len] ⊕ syn[:syn_len], total = MIX_LENGTH
     - train LSTM, MAPE
5. Phase 5 — Aggregation + plots
6. Phase 6 — Cross-generator comparison

## Key wrappers (shared between the two notebooks)

```python
def build_returns_windows(prices, window_size=21):
    \"\"\"prices: 1D array → (N_windows, window_size, 1) of sliding-window returns.\"\"\"

def make_timegan_args(asset, model_path, **overrides):
    \"\"\"Build the argparse-style Namespace required by birdx0810/timegan-pytorch.\"\"\"

def generate_timeseries_timegan(T, model, args, p_init):
    \"\"\"Generate T days of synthetic prices by stitching 21-day return windows.\"\"\"

def load_or_train_timegan(asset, train_data, train_time, args):
    \"\"\"Idempotent: load checkpoint if exists, else train and save.\"\"\"
```

## Acceptance criteria

- [x] Both notebooks open as valid `.ipynb` JSON (26 cells each)
- [x] Phase 1 cell clones `birdx0810/timegan-pytorch` into Colab `/content/`
- [x] All code cells parse as valid Python AST (0 syntax errors)
- [x] Per-run CSVs idempotent (re-runs skip done runs/sweep points)
- [x] Cross-generator plot reads FTS-Diffusion summary at
      `results/{exp}/{asset}/k{K}/{protocol}/summary.csv` and TimeGAN summary at
      `results/{exp}/{asset}/timegan/summary.csv`, overlays them on one figure
- [x] `models` package shadowing handled: cell 4 wipes any cached
      `models*` from `sys.modules` before importing TimeGAN, so a kernel
      that previously ran an FTS-Diffusion notebook can still import
      `from models.timegan import TimeGAN`

## Review

**Created:** `src/TATR_TimeGAN_Replication.ipynb` and
`src/TMTR_TimeGAN_Replication.ipynb` (26 cells each, mirror structure of the
FTS-Diffusion notebooks).

### What was built
- 10 sections each: Setup → Config → Phase 1 (data + return windows) →
  Phase 2 (TimeGAN training) → Phase 3 (price-stitching wrapper) →
  Phase 4 (TATR/TMTR loop) → Phase 5 (aggregation) → Phase 6 (final figs)
  → Phase 7 (cross-generator overlay) → Resume guide.
- Auto-clones `birdx0810/timegan-pytorch` into `/content/` (Colab) or
  `/tmp/` (local) on Phase 1.
- TimeGAN args pre-built via `SimpleNamespace`: matches the constructor
  signature of `birdx0810/timegan-pytorch/main.py`.
- Sliding-window construction: returns of length `TG_WINDOW=21`
  (paper-faithful per Appendix E.2), z-score standardised; inverted at
  generation time before `cumprod(1+r)` price stitching.
- Return clipping at ±50% during stitching to avoid pathological
  GAN-tail compounding (TimeGAN can occasionally output extreme returns).
- Single TimeGAN protocol (`timegan_default`); the cross-generator
  comparison cell loads any FTS-Diffusion summary (configurable K +
  protocol) and overlays on one plot.
- Idempotent: per-run CSVs, partial-completion detection, sweep-point skip.
- Drive layout uses `generator` axis: `…/{ASSET}/timegan/` next to
  `…/{ASSET}/k{K}/{PROTOCOL}/` for FTS-Diffusion. No refactor of existing
  FTS-Diffusion notebooks.

### What was NOT built
- Multi-protocol TimeGAN comparison (TimeGAN doesn't have an MC, so
  the protocol axis is meaningless for it).
- TimeGAN training-from-scratch hyperparameter sweep — defaults match
  the birdx0810 repo's defaults but reduced to 200/200/200 epochs (vs
  600/600/600) to fit Colab; bump them in cell 6 if quality is poor.

### Verification done
- JSON validity confirmed (26 cells, kernel python3)
- Every code cell parses as valid Python AST (0 syntax errors)
- TimeGAN repo API verified by cloning
  `birdx0810/timegan-pytorch` and reading `models/timegan.py`,
  `models/utils.py`, `models/dataset.py`, `main.py` — args namespace,
  `timegan_trainer` and `timegan_generator` signatures match
- `models` package shadowing: handled with explicit
  `sys.modules` cleanup in the imports cell

### Pending (cannot be tested without GPU + Drive)
- End-to-end smoke test on Colab (TimeGAN training takes ~30-60 min on A100
  even with reduced epochs)
- Cross-generator plot empirical validation (needs an existing
  FTS-Diffusion summary on the same Drive)



