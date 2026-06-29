# Paper revision handoff (FTS-Diffusion replication)

Working notes so any future session can continue without re-deriving context.
Last updated: 2026-06-29.

## Current state in one paragraph

The canonical paper is `NeurIPS_Paper/neurips_2026.tex` (the old `paper/`
skeleton is retired), titled **"Replicating FTS-Diffusion: Degeneracy of the
Released Generation Pipeline"**. It is a **replication / degeneracy** study of
FTS-Diffusion (Huang/Liao et al., ICLR 2024). Thesis: the released sampling
pipeline, as configured, is **degenerate** (collapses to a near-linear price
ramp from one repeated motif); the apparent S&P 500 TATR gain is a **scale
coincidence** that does not transfer to GOOG or ZC=F; the authors' own KS/AD
return test already catches the degeneracy. The **predictable-drift extension
was removed** on the user's instruction (see below); the paper is now a focused
replication study, **~9 pages**, compiles with no undefined refs.

## Branch layout (important)

`main` and `deli_work` now hold an **identical** paper: real author block,
`\usepackage[preprint]{neurips_2026}`, and `\hypersetup{hidelinks}` on both. The
earlier anonymized `deli_work` variant was dropped on the user's request.

To produce an anonymous NeurIPS submission later, change three preamble lines:
drop `[preprint]` (`\usepackage{neurips_2026}`), remove `\hypersetup{hidelinks}`,
and set the author block to `Anonymous Author(s)`.

To sync a change across branches: commit on one branch, `git checkout
<other-branch>`, `git checkout <src-branch> -- NeurIPS_Paper/...`, rebuild,
commit. While the branches match there is no preamble to re-apply. Stage
**explicitly** only `NeurIPS_Paper/` files: the working tree carries unrelated
changes (a modified `.gitignore`, an untracked
`results/tatr/goog/k11/split/summary.csv`) that must not be swept into a commit.
When switching to `main` they block the checkout, so `git stash push -u` first
and `git stash pop` after returning to `deli_work`.

## Writing style we must keep (supervisor feedback + house rules)

The supervisor flagged the draft as "clearly AI-generated." Keep these rules in
any future edit:

1. **No em-dashes or en-dashes in prose.** Use a period, a comma, parentheses,
   or a colon. (LaTeX `--` for numeric ranges is fine, not as sentence
   punctuation.)
2. **Few sentences starting with "The".** The supervisor counted 38+ in 12
   pages and called it extreme. Reword openings. Captions count too.
3. **No "This is X, not Y" / "not X but Y" constructions** and no overly
   balanced, symmetrical sentences. State a plain claim.
4. **No vague phrases that sound good but say little** ("deliberately narrow",
   "creates a sensitive interface", "well motivated", "usable samples",
   "failure mode", "A narrower reading fits"). State the concrete fact.
5. **No long parallel "we do X, check Y, and test Z" lists.** Break them up.
6. **No smooth/formulaic colon-explanations or generic transitions**
   ("Crucially,", "Under that lens", "What matters is that", "We note only
   that"). Cut or rephrase.
7. **Define every audience-specific term/acronym before first use.** The
   supervisor caught "Jaccard" used 5 times, never defined. Genuinely simple
   terms (e.g. MAE) do NOT need a gloss; use judgment.

Example rewrite the user endorsed: "This separation is useful, but it also
creates a sensitive interface: downstream results depend on..." becomes
"Although useful, this separation makes downstream results sensitive to how the
learned state chain is initialized, reset, and divided into synthetic training
blocks." Make **local** edits only; do not rewrite aggressively or reorder
arguments.

## Important changes made (most recent first)

- **Predictable-drift extension removed entirely** (user decision, 2026-06-29):
  the whole §6, its four tables, its figure, its equations, plus every mention
  in the title, abstract, intro, contributions (now four, was five), compute
  timings, limitations, and conclusion. The paper dropped from 13 to ~9 pages.
  Orphaned but left on disk: `figures/06_drift_extension_evidence.pdf`; the
  `\newcommand`s `\E`, `\F`, `\norm` are now unused but harmless. No paper text
  references any drift artifact anymore.
- **Two prose passes** removing inflated/generic/AI-sounding phrasing across
  abstract, intro, background, §4, §5, conclusion (see the style rules above).
- **Protocol set reduced to three**, by code identifier with the paper term:
  `authors` (independent fixed blocks), `split` (continuous chunked),
  `random_init` (independent blocks, random init). The `single` protocol (one
  continuous trajectory, un-chunked) was dropped because it does not share the
  same data budget; the literal protocol token `single` is gone (the English
  adjective "a single pattern" is fine).
- **TATR table is 30-seed, 3 protocols** (was a stale 6-seed / 4-row table).
  Numbers verified from `results/tatr/sp500/k14/<proto>/summary.csv`
  (`n_runs=30`). All "six-seed" wording updated to "30-seed".
- **GOOG/ZC=F numbers aligned to `split`**: `authors` +1003% (GOOG) / +75%
  (ZC=F); `split` never beats baseline, +92% (GOOG) / +179% (ZC=F).
- **Checkpoint framing corrected.** The original authors ship **no** checkpoints
  (verified: `code_from_authors/` has no `trained_models`; `README_ref.md` tells
  you to train them). We trained our own at a reduced budget. GOOG and ZC=F
  **do** have full generative models (SISC + PGM + PEM) for several K (proof:
  TATR results exist for them, which need PGM+PEM). So do NOT say "checkpoints
  only for S&P 500" or "GOOG/ZC=F only diagnostics".
- **Definitions added at first use:** PGM, Scaling-AE, PCDM, PEM (§2); TMTR =
  Train on Mixture, Test on Real (§5); IoU, Jaccard (§5). (ACF/NMSE definitions
  lived in the now-removed §6.)
- **Figures.** `02_tatr_protocol_contrast.pdf` (three protocols / 30 seeds),
  plus added `03_pem_collapse.pdf` and `07_cross_asset_nontransfer.pdf`; all
  verified against the data and tracked in git.

## Verified facts and numbers (do not recompute)

1. **Degeneracy, three code choices** (present in BOTH `fts-diffusion-ref/` and
   the authors' own `code_from_authors/`, so they are the authors' choices):
   - `models/sampling.py:30,33,34` deterministic `argmax` PEM rollout to a fixed
     point (S&P 500, pattern 12 used 99.6%) or period-2 cycle (GOOG/ZC=F, 50%).
   - `model_params.py` `sae_hidden_dim=1` + decoder LSTM
     (`scaling_autoencoder.py:58`) gives a decoder near-constant across patterns
     (cross-pattern std < 0.001, from `GENERATION_FAILURE_ANALYSIS.md`).
   - `sampling.py:53-54` magnitude range-normalization commented out (line 55 is
     the active un-normalized multiply); decoder range about 0.073.
   - Net effect: one tiny about +0.075 motif tiled about 2520x = near-linear
     ramp, continuous-run **linear-fit R^2 about 0.999999**.
2. **Configs:** ours PGM 30 epochs / PEM 60 / PCDM 30 steps; released (authors)
   400 / 1000 / 100; shared SISC k-means seed 42; `l_min=10`, `l_max=21`.
3. **30-seed S&P 500 TATR** (`results/tatr/sp500/k14`):

   | Protocol | baseline | best change | final change | per-seed final range |
   |---|---|---|---|---|
   | `authors` | 0.0611 | +0.0% | +208.4% | [+17.0%, +438.3%] |
   | `split` | 0.0620 | -77.6% (yr 60) | -72.2% | [-89.1%, -21.2%] |
   | `random_init` | 0.0611 | +0.0% | +68.2% | [+3.9%, +263.3%] |

   Cross-asset (continuous `split`, vs each asset's own baseline): GOOG +92%
   final / never beats; ZC=F +179% final / never beats. `authors`: GOOG +1003%,
   ZC=F +75%.
4. **KS/AD return test** mirrors `metrics_quant.py::distribution_tests`: mean KS
   p about 1e-4 to 1e-7, 0/300 windows indistinguishable, every asset/protocol.
   Real/synthetic daily-return sd ratios span about 2.4x (`authors`) to about
   62x (continuous). The paper text still says "4 to 60x"; the true floor is
   about 2.4x, so "2 to 60x" is more accurate (pending the §4 rewrite).
5. **`sp500_us` retrain-from-scratch** (reduced budget 30/60/30, NOT full):
   PEM less collapsed (66.5% dominant over 4 patterns vs 99.6% over 2), but still
   degenerate (R^2 = 0.999995, return vol about 27x too small, runs
   near-identical); TATR reproduces the pattern (`split` -72%, `authors` +175%).
   Rules out a single unlucky checkpoint as the cause. Removed from the paper but
   kept here for the record.

## Open items / unverified

1. **§4 may still be edited by the user.** The "4 to 60x" -> "2 to 60x" fix is
   theirs to finalize.
2. **TMTR numbers unverified.** No `results/tmtr/` on disk. The §5 TMTR figures
   (best +26.8%, finishes +359.7% worse, offset-30 +50%/-17.8%, offset-50 +752%)
   and the seed count could not be checked. The phrase was neutralized to "In
   our TMTR runs"; confirm or re-run.
3. **Compute time (10,478 s)** is labeled "30-seed" but may be the old 6-seed
   timing. Re-measure for 30 seeds, or correct.
4. **H1 full-budget retrain** (PGM 400 / PEM 1000 / PCDM 100, multi-seed) is the
   experiment that would upgrade "degenerate release" to "degenerate method" or
   falsify it. Falsifier: decoder cross-pattern std > 0.01.

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
  (about 720 KB, Scaling-AE + diffusion) plus a `pem_*.pt` (about 82 KB).
- Build: `cd NeurIPS_Paper && pdflatex ... ; bibtex neurips_2026 ; pdflatex x2`.
  Clean aux files after.
- **matplotlib gotcha:** system numpy 2.x but system matplotlib is built for
  numpy 1.x and crashes. Render figures in a venv:
  `python3 -m venv figvenv && figvenv/bin/pip install "numpy>=2" "matplotlib>=3.8" pandas`.
  Inspect a figure PDF with `pdftoppm -png -r 90 fig.pdf out` then view the png.
