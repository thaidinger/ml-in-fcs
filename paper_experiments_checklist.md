# NeurIPS 2026 paper — experiments + writing checklist

**Thesis (confirmed 2026-05-25):** Reproducibility audit + evaluation-blindness
demonstration of FTS-Diffusion. See conversation log; in short: (i) the released
artifacts do not reproduce the paper's claims; (ii) the metrics used to validate
the paper fail to flag the resulting degeneracy on at least the upward-trending
S&P 500.

**Deadline:** 2026-07-01 14:00 (≈ 5 weeks from confirmation).
**Page budget:** 9 content pages + unlimited appendix.

---

## 0. Open scoping decisions (resolve in week 1)

These were defaulted but not explicitly confirmed. Lock them down before drafting.

- [ ] **Track**: main (double-blind) vs. position (single-blind methodological).
      Default: **main**. Position is a viable fallback if the constructive §6 is
      cut and the paper leans further into "metrics are blind."
- [ ] **Author contact (Huang, Chen, Qiao @ CityU HK)**: send email this week
      asking whether `sae_hidden_dim=1` is the value used for the paper's
      headline results, and whether trained checkpoints exist.
      Default: **send**. Calendar-sensitive — reply only useful if it lands
      before week 4.
- [ ] **Stretch §6 (alternative metrics)**: in or out.
      Default: **in**. Decision point at end of week 2 based on experiment
      progress. If §5 isn't done by end of week 2, cut §6.
- [ ] **Asset scope for H1 retraining**: S&P 500 only, or all three.
      Default: **S&P 500 only** for H1; reuse existing GOOG/ZC=F checkpoints
      for blindness demos with caveat.

---

## 1. Tier 1 — must-do, week 1

These are blockers for everything else. Without them §4 and §5 cannot be written.

- [ ] **E1. KS/AD on our generated returns** — all assets, all protocols
      we have saved trajectories for.
      *Produces*: a table mirroring paper Table 1 with our numbers.
      *Compare*: paper Table 1 (S&P 500 FTS-Diffusion KS=.327, AD=.128;
      similar for GOOG, ZC=F).
      *Estimate*: 1–2 days. *Pure scripting; no training.*
      *Artifact path*: `results/distribution_tests/<asset>_<k>_<protocol>.csv`

- [ ] **E2. Linear-ramp baseline** — generate the simplest possible degenerate
      synthetic series (a literal linear ramp from train-set start price to
      train-set end price, plus tiny iid Gaussian noise to make returns
      well-defined), apply the paper's metrics (KS/AD + TATR MAPE + TMTR).
      *Produces*: numbers showing how high the bar is to fail the paper's
      evaluation.
      *Estimate*: 1–2 days.
      *Artifact path*: `results/baselines/linear_ramp/*`

- [ ] **E3. Email FTS-Diffusion authors.** Subject: "Question about
      sae_hidden_dim in released code for FTS-Diffusion (ICLR 2024)."
      One paragraph: cite `model_params.py:33`, ask whether `sae_hidden_dim=1`
      is what produced the paper's results or whether a different config was
      used; ask whether trained checkpoints are available. Politely.
      *Estimate*: 30 minutes. *Calendar-critical — send this week.*

- [ ] **E4. Start H1 retraining in background.** Set
      `pgm_params['n_epochs']=400`, `pem_params['n_epochs']=1000`,
      `pgm_params['pcdm_n_steps']=100` in
      `fts-diffusion-ref/models/model_params.py`. Train PGM + PEM on S&P 500
      with same data path as current checkpoints. Save checkpoints to a
      separate directory (do not overwrite the 30-epoch ones — they are
      evidence in §4).
      *Produces*: `fts-diffusion-ref/trained_models_h1/{pgm.pt,pem.pt}` and
      `scripts/investigate_pgm.py` rerun output for the new checkpoints.
      *Estimate*: 3–5 GPU-days. Set it running and don't watch it.
      *See also*: GENERATION_FAILURE_ANALYSIS.md §Alternative hypotheses / H1.

---

## 2. Tier 2 — must-do, weeks 2–3

These complete the experimental backbone of §4 and §5.

- [ ] **E5. H1 result analysis.** When E4 completes: run
      `scripts/investigate_pgm.py` against the new checkpoints, regenerate
      synthetic trajectories with `generate_timeseries_ftsdiffusion`, recompute
      KS/AD (E1 pipeline), recompute TATR/TMTR on the new synthetic.
      *Falsifier criterion*: cross-pattern std of decoder output >0.01 at
      any timestep; per-pattern net range varies by >0.05 across patterns.
      *Estimate*: 1–2 days after E4 finishes.
      *Artifact path*: `results/h1_full_budget/*`

- [ ] **E6. TimeGAN alternate-generator reference.** Use existing
      `src/TATR_TimeGAN_Replication.ipynb` and `src/TMTR_TimeGAN_Replication.ipynb`
      outputs to populate side-by-side metric comparisons. If those notebooks
      haven't been run end-to-end on S&P 500, run them now.
      *Produces*: a table with FTS-Diffusion (ours, 30ep), FTS-Diffusion (H1,
      400ep), TimeGAN, linear-ramp baseline, all metrics, all assets.
      *Estimate*: 3–5 GPU-days if not yet run; ≤1 day if already done.

- [ ] **E7. Pattern-distribution comparison.** For each generated trajectory,
      run the segmentation back through SISC and tabulate the histogram of
      `(pattern, length, magnitude)` triples; compare to the real-data
      histogram. This is the diagnostic that most directly shows the PEM
      collapse to a fixed point. Likely already partly done in
      `figures/mc_analysis/`; verify and finalize as a paper-ready figure.
      *Estimate*: 1–2 days.
      *Artifact path*: `figures/pattern_distribution/<asset>.pdf` and
      `results/pattern_distribution/<asset>.csv`

- [ ] **E8. Per-asset robustness check.** Re-run E1 + paper-style TATR/TMTR
      on GOOG and ZC=F with our existing (degenerate) checkpoints, to test
      whether the metric blindness is S&P-500-specific (where the asset itself
      trends upward, masking the ramp) or generalizes.
      *Hypothesis to test*: metrics may catch the degeneracy on non-trending
      assets (GOOG, ZC=F) but miss it on the trending one (S&P 500).
      *Estimate*: 1–2 days using existing trajectories.

---

## 3. Tier 3 — stretch §6 (decide by end of week 2)

Only run these if §4 + §5 are landed and there is time for a constructive
contribution.

- [ ] **E9. Pattern-transition entropy metric.** Define
      `H(p_{t+1} | p_t)` over the synthetic chain, compute on real data,
      compute on each generator's output. A near-zero conditional entropy is
      the signature of a collapsed PEM that TATR/TMTR cannot see.
      *Produces*: a new diagnostic table.
      *Estimate*: 1–2 days implementation + 1 day validation.

- [ ] **E10. Returns second-moment stylized fact**: rolling-window volatility
      autocorrelation. The paper claims "decaying autocorrelation of absolute
      returns" but doesn't quantify the decay rate. Compute decay constants
      on real vs. each generator's output; show that a linear ramp produces
      degenerate (near-constant zero) absolute-return autocorrelation.
      *Estimate*: 1 day.

- [ ] **E11. Validation of proposed metrics on a non-FTS-Diffusion generator
      (TimeGAN).** Apply E9 + E10 to TimeGAN output to demonstrate the
      metrics aren't accidentally only calibrated against FTS-Diffusion's
      failure mode.
      *Estimate*: 1 day.

---

## 4. Tier 4 — appendix-only / nice-to-have

Include only if there is genuinely free time in week 5.

- [ ] **E12. H2 seed sensitivity.** Retrain PGM at H1 settings with
      `SEED ∈ {0, 1, 2}` (skip 42 since that's H1). Three runs in parallel
      if GPU permits. Appendix table comparing decoder-output cross-pattern
      std across seeds.
      *Estimate*: 5–7 GPU-days (parallelizable).
      *See also*: GENERATION_FAILURE_ANALYSIS.md §Alternative hypotheses / H2.

- [ ] **E13. H3 capacity ablation.** Retrain PGM with `sae_hidden_dim=16`
      and `sae_hidden_dim=64`, both at H1 epoch budget, S&P 500 only.
      *Estimate*: 5–7 GPU-days.
      *See also*: GENERATION_FAILURE_ANALYSIS.md §Alternative hypotheses / H3.

- [ ] **E14. Teacher-free rollout pilot.** Implement the DTW re-inference
      loop from SAMPLING_DESIGN_ANALYSIS.tex, run on one of the (H3) larger
      decoders. Appendix-only; do *not* sell as a contribution.
      *Estimate*: 3 days implementation + 1 day evaluation.

---

## 5. Writing milestones

- [ ] **W1 (end of week 1)**: §1 Introduction + §3 Background drafts in
      LaTeX. Doesn't need experiments. Mostly synthesized from existing docs.
- [ ] **W2 (end of week 2)**: §2 Related Work + §4 Reproducibility findings
      drafted. §4 uses GENERATION_FAILURE_ANALYSIS.md tables + paper_under-
      specifications.md.
- [ ] **W3 (end of week 3)**: §5 Evaluation blindness experiments drafted
      with E1, E2, E5, E6, E7, E8 figures and tables wired in.
- [ ] **W4 (end of week 4)**: §6 Stress-test framework (if stretch is in) +
      §7 Discussion + §8 Conclusion drafted. Appendix populated.
- [ ] **W5 (end of week 5)**: Full pass, checklist.tex completed, figures
      polished, page-count fit, references cleaned. **Submit.**

---

## 6. Calendar — fragile dependencies to track

- [ ] **2026-05-25 → 2026-06-01 (week 1)**: send author email (E3) by
      Wednesday 2026-05-27 at the latest.
- [ ] **2026-06-01 → 2026-06-08 (week 2)**: H1 (E4) checkpoints due. If H1
      fails to complete (e.g., GPU crash), restart immediately — losing more
      than 2 days here makes §4 weaker.
- [ ] **2026-06-15 (start of week 4)**: stretch §6 in-or-out final
      decision. If in, E9–E11 must complete by 2026-06-22.
- [ ] **2026-06-29**: writing freeze; only typo/figure fixes after this.
- [ ] **2026-07-01 14:00**: submission deadline.

---

## 7. Done log

When an item completes, move it here with date + one-line note of what
was produced. Keep the upper checklist clean.

(empty)
