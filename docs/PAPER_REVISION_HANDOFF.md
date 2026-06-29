# Paper revision handoff (FTS-Diffusion replication)

Working notes so any future session can continue without re-deriving context.
Last updated: 2026-06-29.

## Current state in one paragraph

The canonical paper is `NeurIPS_Paper/neurips_2026.tex` (the old `paper/`
skeleton is retired). It is a **replication + diagnostic-extension** study of
FTS-Diffusion (Huang/Liao et al., ICLR 2024). Thesis: the released sampling
pipeline, as configured, is **degenerate** (collapses to a near-linear price
ramp from one repeated motif); the apparent S&P 500 TATR gain is a **scale
coincidence** that does not transfer to GOOG or ZC=F; the authors' own KS/AD
return test already catches the degeneracy. A predeclared **predictable-drift
diagnostic** (§6) is the extension. The paper compiles to **13 pages**, no
undefined refs. §4 is currently being rewritten by the user; §6 (extension) is
intentionally left mostly untouched.

## Branch layout (important)

The same revised paper lives on **both** branches and they differ in **only
three preamble lines**:

| Line | `main` (preprint) | `deli_work` (anonymous submission) |
|---|---|---|
| documentclass opt | `\usepackage[preprint]{neurips_2026}` | `\usepackage{neurips_2026}` |
| hyperref | `\hypersetup{hidelinks}` present | removed |
| author block | real names (Deli Lin, Eric Zhao, Lapo Linossi, Tom Haidinger) | `Anonymous Author(s)` |

The **body content is identical**. To sync a change across branches: commit on
one branch, `git checkout <other-branch>`, `git checkout <src-branch> --
NeurIPS_Paper/neurips_2026.tex NeurIPS_Paper/figures/*`, then re-apply the three
preamble lines for the target branch, rebuild, commit. Stage **explicitly** only
`NeurIPS_Paper/` files: the working tree has unrelated deletions/untracked files
that must not be swept into a commit.

## Writing style we must keep (supervisor feedback + house rules)

The supervisor flagged the draft as "clearly AI-generated." Keep these rules in
any future edit:

1. **No em-dashes or en-dashes in prose.** Use a period, a comma, parentheses,
   or a colon instead. (LaTeX `--` for numeric ranges and `---` are fine inside
   math/ranges, not as sentence punctuation.)
2. **Few sentences starting with "The".** The supervisor counted 38+ in 12
   pages and called it extreme. Reword openings (use the subject, "Because…",
   "A…", a verb-first clause). Captions count too.
3. **No "This is X, not Y" constructions** (e.g. the old "This is a replication
   study, not a new generator"). Rephrase to a plain claim ("We position this
   work as…").
4. **No vague phrases that sound good but say little** (e.g. the old
   "deliberately narrow"). State the concrete fact ("tests only the
   five-dimensional linear feature map ω_t").
5. **Define every term/acronym before first use; calibrate to the audience.**
   The supervisor caught "Jaccard" used 5 times, never defined. Now defined; see
   the changes list. When adding content, expand the acronym on first use.
6. Avoid generic AI tells: ceremonial summaries, "it is worth noting", reflexive
   hedging.

Genuinely simple terms do **not** need a gloss (the user explicitly said MAE
does not). Use judgment: define audience-specific jargon, not basics.

## Important changes made (this session, on both branches)

- **Protocol set fixed and reduced to three**, named by code identifier with the
  paper term: `authors` (independent fixed blocks), `split` (continuous
  chunked), `random_init` (independent blocks, random init segment). The
  `single` protocol (one continuous trajectory, un-chunked) was **removed from
  the paper** because it does not share the same data budget as the other three.
  The literal token `single` as a protocol name is gone; the English adjective
  ("a single pattern") is fine.
- **TATR table (`tab:tatr-summary`) is now 30-seed, 3 protocols** (was a stale
  6-seed / 4-row table with a phantom "Continuous + scaler refit" and a "5-day"
  row). Numbers verified from `results/tatr/sp500/k14/<proto>/summary.csv`
  (`n_runs=30`). Prose "six-seed" updated to "30-seed" in the intro contribution
  (ii), the Compute paragraph, and §5.
- **GOOG/ZC=F numbers aligned to `split`** (the continuous chunked rollout),
  since `single` is gone: `authors` finish +1003% (GOOG) and +75% (ZC=F);
  `split` never beats baseline and finishes +92% (GOOG) and +179% (ZC=F).
- **`sp500_us` retrain-from-scratch removed from the paper** (user: not needed).
  Numbers retained below for the record.
- **Checkpoint framing corrected.** The original authors ship **no** checkpoints
  (verified: `code_from_authors/` has no `trained_models`; `README_ref.md` tells
  you to train them). We trained our own at a reduced budget. GOOG and ZC=F
  **do** have full generative models (SISC + PGM + PEM) for several K (the proof:
  TATR results exist for them, and TATR cannot run without PGM+PEM). So the paper
  must NOT say "checkpoints only for S&P 500" or "GOOG/ZC=F only diagnostics".
  Current intro text: "the original authors release no trained checkpoints, so we
  train every module ourselves and at a reduced budget relative to their
  defaults."
- **Definitions added at first use:** PGM (pattern generation module), Scaling-AE
  (scaling autoencoder), PCDM (pattern-conditioned diffusion model), PEM
  (pattern-evolution module) in §2; TMTR = **Train on Mixture, Test on Real** in
  §5; IoU (intersection over union) in §5; ACF(|r|) and excess kurtosis in the
  §6 calibration caption; NMSE (normalized mean squared error) and Jaccard
  (`|A∩B|/|A∪B|`, tolerance-2 variant explained) in §5.
- **Figures.** `02_tatr_protocol_contrast.pdf` already updated to the three
  protocols / 30 seeds. **Added** `03_pem_collapse.pdf` (next to the PEM-collapse
  table) and `07_cross_asset_nontransfer.pdf` (GOOG/ZC=F non-transfer); both
  verified against the data. All figure PDFs in `NeurIPS_Paper/figures/` are
  tracked.
- Earlier session (already in the draft before this one): new §4 "Degeneracy of
  the released generation pipeline" with the three code choices, PEM-collapse
  table, KS/AD table, underspecification table, confounds; degeneracy figure
  `05_degeneracy_sp500.pdf`; cautious abstract/intro/conclusion.

## Verified facts and numbers (do not recompute)

1. **Degeneracy, three code choices** (present in BOTH `fts-diffusion-ref/` and
   the authors' own `code_from_authors/`, so they are the authors' choices):
   - `models/sampling.py:30,33,34` deterministic `argmax` PEM rollout → fixed
     point (S&P 500, pattern 12 used 99.6%) or period-2 cycle (GOOG/ZC=F, 50%).
   - `model_params.py` `sae_hidden_dim=1` + decoder LSTM
     (`scaling_autoencoder.py:58`) → decoder near-constant across patterns
     (cross-pattern std < 0.001, from `GENERATION_FAILURE_ANALYSIS.md`).
   - `sampling.py:53-54` magnitude range-normalization commented out (line 55 is
     the active un-normalized multiply) → β lost; decoder range ≈ 0.073.
   - Net effect: one tiny ≈+0.075 motif tiled ≈2520× = near-linear ramp,
     continuous-run **linear-fit R² ≈ 0.999999**.
2. **Configs:** ours PGM 30 epochs / PEM 60 / PCDM 30 steps; released (authors)
   400 / 1000 / 100; shared SISC k-means seed 42; `l_min=10`, `l_max=21`.
3. **30-seed S&P 500 TATR** (`results/tatr/sp500/k14`):

   | Protocol | baseline | best Δ | final Δ | per-seed final range |
   |---|---|---|---|---|
   | `authors` | 0.0611 | +0.0% | +208.4% | [+17.0%, +438.3%] |
   | `split` | 0.0620 | −77.6% (yr 60) | −72.2% | [−89.1%, −21.2%] |
   | `random_init` | 0.0611 | +0.0% | +68.2% | [+3.9%, +263.3%] |

   Cross-asset (continuous `split`, vs each asset's own baseline): GOOG +92%
   final / never beats; ZC=F +179% final / never beats. `authors`: GOOG +1003%,
   ZC=F +75%.
4. **KS/AD return test** mirrors `metrics_quant.py::distribution_tests`: mean KS
   p ~1e-4 to 1e-7, 0/300 windows indistinguishable, every asset/protocol.
   Real/synthetic daily-return sd ratios span ≈2.4× (`authors`) to ≈62×
   (continuous). The paper text says "4 to 60×"; the true floor is ≈2.4×
   (S&P `authors` row), so "2 to 60×" is more accurate (pending §4 rewrite).
5. **`sp500_us` retrain-from-scratch** (reduced budget 30/60/30, NOT full):
   PEM less collapsed (66.5% dominant over 4 patterns vs 99.6% over 2), but still
   degenerate (R² = 0.999995, return vol ≈27× too small, runs near-identical);
   TATR reproduces the pattern (`split` −72%, `authors` +175%). This rules out a
   single unlucky checkpoint as the cause. Removed from the paper but kept here.

## Open items / unverified

1. **§4 is being rewritten by the user.** The "4 to 60×" → "2 to 60×" fix and any
   wording there are theirs to finalize.
2. **TMTR numbers unverified.** No `results/tmtr/` on disk. The §5 TMTR figures
   (best +26.8%, finishes +359.7% worse, offset-30 +50%/−17.8%, offset-50 +752%)
   and the seed count could not be checked. The phrase "six-seed long batch" was
   neutralized to "In our TMTR runs"; confirm or re-run.
3. **Compute time (10,478 s)** is labeled "30-seed" but may be the old 6-seed
   timing. Re-measure for 30 seeds, or correct.
4. **§6 coherence gap** (still open, per "don't touch the extension"): the drift
   statistic δ does not flag the continuous rollout (δ=0.0263 inside the S&P 500
   null band q95=0.0303, 0/6 violations) because it standardizes returns (kills
   the constant drift) and uses only lag-1/lag-2 features (misses motif
   periodicity). Mild tension with §4 calling the rollout degenerate. Decide on a
   one-line bridge in §6 or leave as is.
5. **H1 full-budget retrain** (PGM 400 / PEM 1000 / PCDM 100, multi-seed) is the
   experiment that would upgrade "degenerate release" to "degenerate method" or
   falsify it. Falsifier: decoder cross-pattern std > 0.01.
6. `sp500_us` `single` and `random_init` TATR were never produced (only `authors`
   and `split` summaries exist), should the retrain be revived.

## Useful locations / how to re-run

- Degeneracy doc: `docs/GENERATION_FAILURE_ANALYSIS.md`
- Underspecs doc: `docs/paper_underspecifications.md`
- Authors' return tests: `fts-diffusion-ref/experiments/metrics_quant.py`
  (`distribution_tests`: KS/AD/Wasserstein on returns)
- Authors' original code (no checkpoints): `code_from_authors/codes/`
- Synthetic runs: `synthetic/<asset>/k<K>/<proto>/run_*_{syn,blocks,states}.npy`
  (`authors`/`random_init` = (100,252) blocks; `single`/`split` = (25200,) cont.;
  `sp500_us` uses `run_*_{continuous,blocks,states}.npy`)
- Real prices: `fts-diffusion-ref/data/sp500_timeseries.csv`,
  `architectures/{goog,zcf}/k11/data/*_timeseries.csv`
- TATR summaries: `results/tatr/<asset>/k<K>/<proto>/summary.csv` (`n_runs=30`),
  per-seed in `run_XX.csv`
- Trained checkpoints (ours): `fts-diffusion-ref/trained_models/` (active S&P
  path read by `load_ftsdiffusion()`); per-asset archives under
  `architectures/<asset>/k<K>/trained_models/`. Each model = a `pgm-2_*.pt`
  (~720 KB, Scaling-AE + diffusion) plus a `pem_*.pt` (~82 KB).
- Build: `cd NeurIPS_Paper && pdflatex … ; bibtex neurips_2026 ; pdflatex ×2`.
  Clean aux files after.
- **matplotlib gotcha:** system numpy 2.x but system matplotlib is built for
  numpy 1.x and crashes. Render figures in a venv:
  `python3 -m venv figvenv && figvenv/bin/pip install "numpy>=2" "matplotlib>=3.8" pandas`.
  To inspect a figure PDF: `pdftoppm -png -r 90 fig.pdf out` then view the png.
