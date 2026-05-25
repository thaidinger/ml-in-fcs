# NeurIPS 2026 paper sources

Reproducibility study of FTS-Diffusion (Huang et al., ICLR 2024).
Thesis confirmed 2026-05-25. Deadline 2026-07-01.

## Layout

```
paper/
├── main.tex                 # Master document; \input{}s sections/
├── neurips_2026.sty         # NeurIPS 2026 style file (do not edit)
├── checklist.tex            # NeurIPS reproducibility checklist (fill in week 5)
├── references.bib           # Bibliography (BibTeX)
├── Makefile                 # `make` to build; `make watch` for live rebuild
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_background.tex
│   ├── 04_reproducibility_findings.tex
│   ├── 05_evaluation_blindness.tex
│   ├── 06_stress_test_framework.tex   # stretch — decide week-2 in/out
│   ├── 07_discussion.tex
│   ├── 08_conclusion.tex
│   └── appendix.tex
├── figures/                 # PDF/PNG figures (currently empty)
└── tables/                  # Optional: standalone table files
```

## Build

Local: `make` (needs `pdflatex` + `bibtex`).
Live edit loop: `make watch` (needs `latexmk`).
Overleaf: upload the whole `paper/` directory.

## Where to find things while writing

- **\S4 reproducibility evidence**: `GENERATION_FAILURE_ANALYSIS.md` at repo root.
- **\S4 underspec table**: `paper_underspecifications.md` at repo root.
- **\S5 experimental data**:
  - existing HPC TATR/TMTR runs: `raw_2/raw_2/{tatr,tmtr}_run_*.csv` + `task_*_metadata.json`
  - synthetic trajectories: `synthetic/{asset}/k{K}/{protocol}/run_*_*.npy`
  - new metrics tables (E1, E2): plan to land in `results/distribution_tests/` and `results/baselines/linear_ramp/`
- **Architecture description for \S3**: `4634_Generative_Learning_for_F.pdf` at repo root (the FTS-Diffusion paper).
- **Outstanding work**: `paper_experiments_checklist.md` at repo root.

## Convention

Every `\todo{...}` marker in the LaTeX sources marks a writing or experimental
task that's not yet resolved. Grep for `todo` to find them all:

```
grep -rn "todo" paper/sections/
```
