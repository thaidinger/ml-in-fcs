# Original Claim Reproduction Audit

This folder is a compact audit of the original FTS-Diffusion claims that we could not reproduce from the released materials.

The original paper passages checked were the abstract/Section 5.3/Figure 6 for TMTR/TATR and Appendix B.3 for SISC toy-data metrics. The key TATR claim is that appending 100 synthetic years reduces one-day-ahead MAPE by 17.9%, 15.3%, and 17.4% on S&P 500, GOOG, and ZC=F. The key B.3 claims are one-pattern DTW/Jaccard 0.009/0.938 and multi-pattern DTW/Jaccard 0.01/0.784.

## Files

- `claim_reproduction_audit.csv`: row-level claim, closest local evidence, status, and failure reason.
- `audit_metrics.json`: machine-readable numeric summary used by the plot.
- `original_claim_reproduction_audit.png` / `.pdf`: visual summary for discussion or appendix material.

## Bottom Line

The failed reproduction is not one vague failure. It breaks at three concrete points:

- The S&P 500 TATR result is not uniquely recovered from released details: continuous rollouts improve, while released-code-style independent blocks worsen.
- GOOG and ZC=F author-faithful TATR is blocked by the released split logic and the stored author matrices worsen.
- Appendix B.3 needs unreleased toy-generator and metric details to recover the reported DTW/Jaccard values.

## Claim Rows

### TATR S&P 500

- Paper claim: Appending 100 synthetic years reduces one-day MAPE by 17.9%.
- Closest local evidence: Stored author matrix: +208.4%; six-seed independent fixed: +193.2%; continuous rollout final: -65.2% and best: -84.9%.
- Status: Not reproduced as an author-specified result.
- What failed: The released-code-style independent 252-day reset path worsens MAPE. Paper-like gains appear only when synthetic data are generated as one continuous pattern-evolution rollout, so the Markov rollout, initialization, block slicing, scaler fitting, and warm-up choices do not uniquely recover the paper curve.

### TATR GOOG

- Paper claim: Appending 100 synthetic years reduces one-day MAPE by 15.3%.
- Closest local evidence: Stored GOOG author matrix finishes +1003.1% versus baseline.
- Status: Not reproduced; strict rerun blocked by split logic.
- What failed: The released S&P-style downstream split consumes 1260 held-out initialization points, but the GOOG 80/20 held-out window has 744 points. The stored author matrix also trends in the wrong direction.

### TATR ZC=F

- Paper claim: Appending 100 synthetic years reduces one-day MAPE by 17.4%.
- Closest local evidence: Stored ZC=F author matrix finishes +75.1% versus baseline.
- Status: Not reproduced; strict rerun blocked by split logic.
- What failed: The released S&P-style downstream split consumes 1260 held-out initialization points, but the ZC=F 80/20 held-out window has 963 points. The stored author matrix also worsens rather than improves.

### TMTR robustness

- Paper claim: FTS-Diffusion maintains comparable prediction accuracy across synthetic mixing proportions.
- Closest local evidence: Six-seed S&P reference TMTR best -26.8% but final +359.7%; continuous offset-30 final -17.8%; offset-50 final +752.0%.
- Status: Only partially supported under our controls.
- What failed: TMTR is not stable across plausible generation offsets; some settings help, while others collapse at high synthetic proportions.

### Appendix B.3 one-pattern SISC

- Paper claim: One-pattern toy reaches per-unit DTW 0.009 and Jaccard 0.938.
- Closest local evidence: Best available one-pattern DTW 0.0202; best boundary-Jaccard@2 0.8693.
- Status: Not reproduced from released materials.
- What failed: The released tree does not provide the exact one-pattern generator, standard pattern arrays, seed state, or standalone metric implementation. Derived one-pattern runs on the shipped toy data remain below the reported target.

### Appendix B.3 multi-pattern SISC

- Paper claim: Four-pattern toy reaches average per-unit DTW 0.01 and Jaccard/IoU 0.784.
- Closest local evidence: Best sweep DTW 0.0321; best author-style interval IoU 0.6418; best strict boundary-Jaccard@2 0.3608.
- Status: Not reproduced from released materials.
- What failed: The gap persisted across seed, l_max, and iteration sweeps. Exact reproduction appears to need the unpublished synthetic generator details and metric conventions.

