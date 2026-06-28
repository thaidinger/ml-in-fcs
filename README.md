# FTS-Diffusion: Replication and Predictable-Drift Diagnostics

Course project replicating **FTS-Diffusion** and extending its evaluation with a
predictable-drift diagnostic.

Reference paper: Hongbin Huang, Minghua Chen, Xiao Qiao. *Generative Learning for
Financial Time Series with Irregular and Scale-Invariant Patterns*. ICLR 2024.
OpenReview: <https://openreview.net/forum?id=CdjnzWsQax>.

## What this repository contains

1. **Replication** of the paper's three-module pipeline (SISC pattern recognition,
   pattern-conditioned diffusion, pattern evolution) and of its downstream
   Train-on-Augmentation-Test-on-Real (TATR) evaluation, run against the authors'
   released code preserved under `fts-diffusion-ref/`.
2. **Predictable-drift extension**: a fixed, closed-form diagnostic that flags an
   exploitable conditional mean (artificial alpha) in synthetic returns, with
   real-asset calibration, a controlled predictive-alpha test, and an
   augmentation-selection benchmark. Code in `Src/fts_diffusion/`.
3. **Paper** with our findings (degeneracy of the released sampling pipeline,
   cross-asset non-transfer of the TATR gain, drift extension) in
   `Final_Results/paper/`.

## Repository structure

```
Config/            # YAML configs (default.yaml, drift_penalty.yaml)
Data/              # real price series (S&P 500, GOOG, ZC=F) + B.3 toy data
Src/
  fts_diffusion/   # our package: drift + stylized-fact diagnostics, drift penalty
  scripts/         # replication and drift experiment drivers
  tests/           # unit tests for the diagnostics
fts-diffusion-ref/ # authors' released code, preserved as-is (see note below)
notebooks/         # training, TATR, B.3, and drift Colab notebooks
Final_Results/
  paper/           # compiled paper (neurips_2026.pdf) + LaTeX source
  results/         # curated experiment summaries (CSV/JSON/Markdown)
  figures/         # generation-animation GIFs
docs/              # design notes and extension documentation
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # exposes the `fts_diffusion` package
```

## How to run

Predictable-drift diagnostics (self-contained, operate on return CSVs):

```bash
python Src/scripts/run_light_drift_experiment.py
python Src/scripts/run_neurips_end_to_end_diagnostics.py
```

Replication drivers (use the authors' code under `fts-diffusion-ref/`):

```bash
python Src/scripts/run_sisc_assets_ref.py
python Src/scripts/run_sp500_tatr_protocol_multiseed.py
```

Tests:

```bash
pytest Src/tests
```

## Data provenance

`Data/` holds the daily Yahoo Finance close-price series for S&P 500, GOOG, and
ZC=F, plus the Appendix B.3 toy series. The same files are kept inside
`fts-diffusion-ref/data/` so the authors' self-contained scripts run unchanged.

## Note on `fts-diffusion-ref/`

This directory is the authors' released implementation, preserved verbatim and
under its original license. It is the implementation our replication scripts call.
The standalone training CLI referenced by the original packaging
(`src/fts_diffusion/cli`) is not part of this deliverable; the two scripts
`Src/scripts/{train,sample}_fts_diffusion.py` that target it are kept for
reference only and are not runnable here.

## License

Our code is released under the MIT License (`LICENSE.md`). `fts-diffusion-ref/`
retains the authors' original license.
