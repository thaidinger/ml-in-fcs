# Comparison to FTS-Diffusion Paper

This note compares the current local outputs for GOOG and ZC=F against the paper claims in `docs/references/4634_Generative_Learning_for_F.pdf`.

## Matching Settings

The new local runs match the paper's reported real-data setup for the two assets:

| Item | Paper | Local output |
| --- | --- | --- |
| Assets | GOOG and ZC=F | GOOG and ZC=F |
| Date ranges | GOOG: 2005-01-01 to 2020-01-01; ZC=F: 2001-01-01 to 2020-01-01 | Same ranges downloaded from Yahoo Finance |
| Train/test split | 80/20 chronological split | Reference code uses the same split |
| SISC segment lengths | `l_min=10`, `l_max=21` | `l_min=10`, `l_max=21` |
| Number of clusters | S&P500: 14, GOOG: 11, ZC=F: 11 | GOOG: 11, ZC=F: 11 |
| Architecture stage | SISC + PGM + PEM | Completed for GOOG and ZC=F |

## Current Local Results

### SISC Artifacts

| Asset | Prices | K | Segments | Boundaries | Output prefix |
| --- | ---: | ---: | ---: | ---: | --- |
| GOOG | 3,775 | 11 | 258 | 259 | `fts-diffusion-ref/res/sisc_goog_k11_l10-21_dba_kmpp_*` |
| ZC=F | 4,763 | 11 | 321 | 322 | `fts-diffusion-ref/res/sisc_zcf_k11_l10-21_dba_kmpp_*` |

Pattern-library plots are stored in `reports/generated_outputs/12_sisc_pattern_library/`.

### Architecture Checkpoints

The reference PGM/PEM checkpoints now exist for both assets:

| Asset | PGM | PEM |
| --- | --- | --- |
| GOOG | `pgm-2_c48-80_goog_k11_n30_lr4e-04_dw0.01_pw1_sw0.01.{pth,pt}` | `pem_goog_k11_e196_h32_lr4e-04_pw0.05_lw0.01_mw0.94.{pth,pt}` |
| ZC=F | `pgm-2_c48-80_zcf_k11_n30_lr4e-04_dw0.01_pw1_sw0.01.{pth,pt}` | `pem_zcf_k11_e196_h32_lr4e-04_pw0.05_lw0.01_mw0.94.{pth,pt}` |

## What Can Be Compared Directly

The SISC configuration is consistent with the paper: same assets, dates, segment-length range, and K values.

The real-data pattern libraries cannot be numerically scored against the paper because the paper does not publish GOOG/ZC=F real-data centroids or labels. The paper reports quantitative SISC validation only for simulated data: average per-unit DTW error around `0.01` and Jaccard similarity `0.784` in the multi-pattern simulation. Those metrics require ground-truth segment boundaries and pattern labels, which do not exist for real GOOG/ZC=F prices.

## What Does Not Yet Match the Paper

The paper's main downstream TATR claim is that appending 100 years of FTS-Diffusion synthetic data reduces one-day-ahead MAPE by:

| Asset | Paper-reported TATR MAPE reduction |
| --- | ---: |
| S&P500 | 17.9% |
| GOOG | 15.3% |
| ZC=F | 17.4% |

The current GOOG/ZC=F work has not produced a valid author-faithful downstream TATR/TMTR rerun. The released SP500-style author TATR split consumes `252*5 = 1260` held-out points for initial downstream training before testing, but the 80/20 held-out blocks are shorter:

| Asset | Held-out downstream points | Points required by released TATR split | Valid author-faithful test window? |
| --- | ---: | ---: | --- |
| GOOG | 744 | 1260 | No |
| ZC=F | 963 | 1260 | No |

Existing stored GOOG/ZC=F TATR matrices in the report also do not match the paper's downward trend:

| Asset | Stored protocol | Baseline MAPE | Final MAPE | Change vs baseline |
| --- | --- | ---: | ---: | ---: |
| GOOG | `authors` | 0.0119 | 0.1308 | +1003.1% |
| GOOG | `single` | 0.0119 | 0.0244 | +105.6% |
| ZC=F | `authors` | 0.0113 | 0.0198 | +75.1% |
| ZC=F | `single` | 0.0113 | 0.0600 | +430.9% |

Those stored matrices are therefore inconsistent with the paper's claimed GOOG/ZC=F TATR improvements.

## Bottom Line

The local GOOG/ZC=F SISC and architecture replication now matches the paper's stated asset-level setup. The pattern libraries are plausible local artifacts, but not directly scoreable against the paper's real-data results.

The downstream paper claim has not been replicated for GOOG/ZC=F. A strict author-faithful run is blocked by the released split logic on these shorter assets. Any next downstream run should be explicitly labeled as an adaptive-protocol replication, not as an author-faithful replication.
