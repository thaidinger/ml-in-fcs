# FTS-Diffusion presentation rebuild — todo

Mirror of `/home/deli/.claude/plans/tidy-nibbling-swing.md` (full plan there).

## Steps

- [x] **0.** Backup `ML_Finance_and_Complex_Systems_Presentation.tex` → `*.OLD.tex`.
- [x] **0b.** Verify historical year ranges from each timeseries CSV.
- [x] **1.** Write `main.tex` skeleton (56 lines).
- [x] **2.** `sections/00_title.tex`, `01_motivation.tex`, `02_architecture_overview.tex`.
- [x] **3.** `sections/03_sisc.tex`:
  - [x] port OLD lines 153–429;
  - [x] Step-1 overlap fix (centroid-slots label moved to far-left, arrow bend tightened);
  - [x] Step-3 iteration counter (one tick per full segment+assign+update cycle).
- [x] **4.** `sections/04_generation_module.tex`.
- [x] **5.** `sections/05_evolution_module.tex`.
- [x] **6.** `sections/06_data_and_training.tex` — table + inline TikZ bar chart (no external dependency).
- [x] **7.** `sections/07_tatr_protocols.tex` — 3 TikZ MC schemes.
- [x] **8.** `sections/08_results_sisc.tex`.
- [x] **9.** `sections/09_results_tatr.tex` — authors vs split (sp500/zcf) + GOOG split placeholder + random_init + checkpoint-vs-scratch placeholder.
- [x] **10.** `sections/10_results_extra.tex`.
- [x] **11.** `sections/11_limitations_extensions.tex`.
- [x] **12.** `sections/12_conclusions.tex`.
- [x] **13.** Built with `lualatex`; cleaned aux files; PDF 68 pages.

## Acceptance

- `main.tex` ≤100 lines, only preamble + `\input{...}`.
- Compiles without errors → single PDF.
- No overlap on any SISC frame; Step-3 counter advances once per full cycle.
- Placeholders clearly marked for TMTR / 5-day-ahead / scratch.
- Backup `*.OLD.tex` and theme files untouched.

## Out of scope

- ETH theme modifications.
- Invented TMTR / 5-day-ahead numbers.
- Restructuring `architectures/figures/results/synthetic/`.

## Review

- `presentation/main.tex` = 56 lines (preamble + 13 `\input{...}`); compiles with `lualatex` to 68-page `main.pdf` (≈900 KB).
- Modular layout under `presentation/sections/`: 13 files, 00–12.
- SISC fixes applied:
  - Step-1 (`03_sisc.tex` frame 1): centroid-slots label moved to far-left anchor; arrow bends increased (`bend right=22`) so they no longer sweep through the label region.
  - Step-3 (`03_sisc.tex` frame 3): counter overlays now hold at 20 across segment/assign/update phases of iteration 1, then tick to 19 only at end-of-cycle; new label *"one tick = one full cycle"* clarifies the semantics; progress bar advances one tile per completed cycle.
- New slides:
  - 06 — training-data table (sp500 1980–2020 / goog 2004–2020 / zcf 2000–2020) + inline TikZ bar chart showing raw-days vs post-SISC counts (10\,088 vs 2\,282 / 3\,776 vs 733 / 4\,764 vs 629).
  - 07 — three side-by-side MC TikZ schemes (authors / split / random_init), no formulae, condensed captions.
  - 09.5 — checkpoint-vs-scratch placeholder (authors checkpoint figure on the left, TODO fbox on the right).
  - 10 — TMTR + 5-day-ahead alertblock placeholders.
- GOOG/k11 split figure was missing (`figures/tatr/goog/k11/split/` contains only `live.png`), so slide 9.2 uses a placeholder fbox pointing to the existing CSVs under `results/tatr/goog/k11/split/`.
- Backup `ML_Finance_and_Complex_Systems_Presentation.OLD.tex` preserved.
- Aux files cleaned (no `.aux/.log/.toc/.out/.synctex.gz/.fls/.fdb_latexmk/.nav/.snm/.vrb` left).
- Remaining cosmetic overfulls (≤8pt) inside the SISC step-3 caption `\only` macros and one ≈57pt overflow inside the ETH-theme title box — both inherited from the OLD presentation; theme untouched per spec.
