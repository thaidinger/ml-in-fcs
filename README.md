# FTS-Diffusion Reproduction Package

This branch is the public-facing reproduction package for the FTS-Diffusion project. It is intended for ETH supervisor review and contains only the code, input artifacts, generated result tables, and final figure PDFs needed to inspect or rerun the paper results.

The manuscript itself is intentionally not included. The only retained manuscript artifacts are the final figure PDFs in `paper_figures/`.

Original paper being reproduced:

- Huang, Chen, and Qiao, "Generative Learning for Financial Time Series with Irregular and Scale-Invariant Patterns", ICLR 2024.
- OpenReview: <https://openreview.net/forum?id=CdjnzWsQax>
- Proceedings: <https://proceedings.iclr.cc/paper_files/paper/2024/hash/f90fc76b199fe6b0ec2a51aaf72c3277-Abstract-Conference.html>

## Repository Layout

```text
paper_figures/                 Final figure PDFs used in the submitted paper.
reports/generated_outputs/     Stored result tables, plots, manifests, and audit outputs.
scripts/                       Reproducible entrypoints for reruns and figure builds.
fts-diffusion-ref/             Preserved reference implementation, data, SISC artifacts, and checkpoints.
drift_component/               Standalone predictable-drift diagnostic code and tests.
docs/                          Short orientation notes for this cleaned branch.
requirements.txt               Combined Python dependency list for reruns.
```

Removed from this branch: paper TeX/PDF/submission zips, presentation deck files, exploratory notebooks, internal handoff notes, OS metadata, Python bytecode caches, and stale duplicate plotting scripts.

## Environment

Use Python 3.10 or 3.11. Some reruns train PyTorch models and can take hours.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For drift-component-only work:

```bash
pip install -r drift_component/requirements.txt
pip install pytest
python -m pytest drift_component/tests -q
```

## Start Here

1. Read `reports/generated_outputs/README.md` for the result map.
2. Read `reports/generated_outputs/EXPERIMENT_REGISTRY.md` before rerunning anything; it records completed settings and prevents duplicate runs.
3. Use `paper_figures/` for the final paper figures.
4. Use `scripts/build_paper_figures.py` to regenerate the figures from stored outputs:

```bash
python scripts/build_paper_figures.py
```

That script writes the retained paper figure PDFs:

- `paper_figures/02_tatr_protocol_contrast.pdf`
- `paper_figures/04_b3_metric_gap.pdf`
- `paper_figures/05_degeneracy_sp500.pdf`
- `paper_figures/07_cross_asset_nontransfer.pdf`

## Main Reproduction Entry Points

S&P 500 downstream replication and protocol diagnostics:

```bash
python scripts/run_sp500_tatr_ref.py
python scripts/run_sp500_tatr_single_ref.py
python scripts/run_sp500_tatr_protocol_search.py
python scripts/run_sp500_tatr_protocol_multiseed.py
python scripts/run_sp500_long_replication_batch.py
python scripts/run_replicability_diagnostic.py
```

GOOG and ZC=F SISC/architecture artifacts:

```bash
python scripts/run_sisc_assets_ref.py
python scripts/train_ref_architecture_assets.py
python scripts/plot_sisc_pattern_library.py
```

Appendix B.3 simulated SISC checks:

```bash
python scripts/run_b3_sisc_simulated_replication.py
python scripts/sweep_b3_sisc_replication.py
```

Claim audit and final figure rebuild:

```bash
python scripts/probe_pem_transition.py
python scripts/build_reproduction_failure_audit.py
python scripts/build_paper_figures.py
```

Predictable-drift diagnostics retained as related work:

```bash
python drift_component/scripts/run_sign_drift_sweep.py \
  --output-dir reports/generated_outputs/05_drift_diagnostic_advantage
python drift_component/scripts/run_end_to_end_drift_diagnostics.py
python drift_component/scripts/run_drift_incorporated_selection.py \
  --output-dir reports/generated_outputs/07_drift_incorporated_selection
```

## Stored Outputs

The canonical output tree is `reports/generated_outputs/`.

Key folders:

- `00_release_pipeline_diagnostics/`: compact summaries used for the degeneracy figure.
- `01_sp500_downstream_replication/`: TATR/TMTR reruns, protocol search, long-batch outputs, and replicability diagnostics.
- `02_goog_zcf_real_asset_sisc/`: GOOG and ZC=F SISC pattern libraries and comparison notes.
- `03_appendix_b3_simulated_sisc/`: Appendix B.3 reproduction attempts and sweep outputs.
- `05_original_claim_reproduction_audit/`: claim-by-claim audit of reproduced and unreproduced claims.
- `05_drift_diagnostic_advantage/`, `06_end_to_end_drift_diagnostics/`, `07_drift_incorporated_selection/`: retained drift-diagnostic experiments.

## Notes on Reference Code

`fts-diffusion-ref/` is preserved as reference code plus the specific data/checkpoint artifacts used by the rerun scripts. The public branch does not attempt to rewrite that implementation; wrapper scripts in `scripts/` set paths, seeds, and output locations around it.

The paper itself is not required to rerun the experiments. Figure PDFs are kept only as review artifacts and can be regenerated from the retained outputs.
