# Paper underspecifications — FTS-Diffusion replication

A running log of places where the FTS-Diffusion paper (Liao et al., 2024) leaves a
design or hyperparameter choice unspecified, and what we picked in our replication.
This is an **append-only log**, not an argument: one short entry per underspec, with
file:line where the choice lives in our code, and a `See also:` pointer to
[GENERATION_FAILURE_ANALYSIS.md](GENERATION_FAILURE_ANALYSIS.md) when the
underspecification is also a load-bearing cause of the observed generation failures.

Entry format:

```
### N. <short title>
**Paper says:** <what is / isn't specified>
**We chose:** <what we picked, with file:line>
**Why:** <reason for our pick>
**See also:** GENERATION_FAILURE_ANALYSIS.md §<section>   ← only if also a failure cause
```

---

### 1. LSTM downstream train-split for GOOG and ZC=F
**Paper says:** the downstream LSTM split is specified for S&P 500 (~62.5% train);
GOOG and ZC=F splits are not stated.
**We chose:** 62.5% train across all three assets.
**Why:** consistency with the S&P 500 setup so cross-asset comparisons are
apples-to-apples; no asset-specific evidence for a different split.

---

### 2. Scaling-AE decoder hidden dimension
**Paper says:** the Scaling-AE is described as encoding/decoding variable-length
segments to/from a fixed-length representation; the dimension of the
encoder/decoder LSTM hidden state is not specified.
**We chose:** `sae_hidden_dim = 1`
([`fts-diffusion-ref/models/model_params.py:34`](fts-diffusion-ref/models/model_params.py#L34)),
the value shipped by the authors at
[`code_from_authors/codes/models/model_params.py:33`](code_from_authors/codes/models/model_params.py#L33).
**Why:** kept the authors' released value unchanged for replication.
**See also:** GENERATION_FAILURE_ANALYSIS.md §Problem 2 / *PGM investigation*.

---

### 3. Segment magnitude normalization at sampling time
**Paper says:** the generative process emits a unit-range shape and multiplies it by
the magnitude β to produce the segment. This requires normalizing the raw decoder
output by its own range before scaling.
**We chose:** kept the authors' released sampling path, which has the
range-normalization commented out and multiplies the un-normalized output by β
directly ([`fts-diffusion-ref/models/sampling.py:53-55`](fts-diffusion-ref/models/sampling.py#L53-L55)).
**Why:** replicate the released behavior exactly; uncommenting changes generation
semantics relative to authors' code.
**See also:** GENERATION_FAILURE_ANALYSIS.md §Problem 2 / *The lost magnitude*.

---

### 4. PEM rollout: argmax vs. sampling
**Paper says:** describes the pattern-evolution module as a Markov chain over
pattern/length/magnitude states. The rollout policy (greedy argmax vs. sampling
from the predicted distribution) is not specified.
**We chose:** deterministic argmax for pattern and length, raw regression mean for
magnitude — kept from authors' released code at
[`fts-diffusion-ref/models/sampling.py:30,33,34`](fts-diffusion-ref/models/sampling.py#L30-L34).
**Why:** released behavior unchanged; switching to multinomial sampling is one of
the suggested fixes (§Suggested fixes #1) but would be a deviation from the
release.
**See also:** GENERATION_FAILURE_ANALYSIS.md §Problem 1 / *Mechanism*.

---

### 5. Training epoch budgets (PGM, PEM)
**Paper says:** PGM and PEM training epochs are not specified.
**We chose:** `pgm_params['n_epochs']=30`
([`fts-diffusion-ref/models/model_params.py:48`](fts-diffusion-ref/models/model_params.py#L48)),
`pem_params['n_epochs']=60`
([`fts-diffusion-ref/models/model_params.py:65`](fts-diffusion-ref/models/model_params.py#L65)).
The authors' released defaults are 400 / 1000
([`code_from_authors/codes/models/model_params.py:47,64`](code_from_authors/codes/models/model_params.py#L47)) —
a 13× / 16× compute-driven reduction on our side.
**Why:** compute budget. **This is a confound for the failure analysis** and must
be retired before publishing a strong critical claim — see
GENERATION_FAILURE_ANALYSIS.md §Alternative hypotheses, H1.
**See also:** GENERATION_FAILURE_ANALYSIS.md §Alternative hypotheses / H1.

---

### 6. SISC k-means clustering seed
**Paper says:** k-means is used to cluster shape-invariant subsequences into k
patterns; no seed is specified.
**We chose:** `SEED = 42`, set globally at
[`fts-diffusion-ref/models/model_params.py:6`](fts-diffusion-ref/models/model_params.py#L6)
and used by the clustering in `pattern_recognition_module.py`. Same value as
authors'.
**Why:** kept the released value for replicability.
**See also:** GENERATION_FAILURE_ANALYSIS.md §Alternative hypotheses / H2.

---

### 7. PCDM diffusion steps
**Paper says:** the pattern-conditioned diffusion module is described as a standard
DDPM; the number of forward/reverse steps is not specified in the main text.
**We chose:** `pgm_params['pcdm_n_steps']=30` and `pgm_params['n_steps']=30`
([`fts-diffusion-ref/models/model_params.py:37,46`](fts-diffusion-ref/models/model_params.py#L37)).
The authors' released defaults are 100 / 100
([`code_from_authors/codes/models/model_params.py:36,45`](code_from_authors/codes/models/model_params.py#L36)).
**Why:** compute budget (companion to entry 5). Should be restored to 100 when
running the H1 falsification experiment.
**See also:** GENERATION_FAILURE_ANALYSIS.md §Alternative hypotheses / H1.

---

### 8. PGM loss weighting
**Paper says:** the PGM is trained with a reconstruction loss on the AE output and
a denoising loss on the PCDM; the weighting between the two is not specified.
**We chose:** `loss_weights=[0.98, 0.01]` (AE-dominant) plus `pad_weight=1`,
`scale_weight=0.01`
([`fts-diffusion-ref/models/model_params.py:50-52`](fts-diffusion-ref/models/model_params.py#L50-L52)).
Identical to authors' released values.
**Why:** kept the released values.

---

### 9. PEM loss weighting
**Paper says:** the PEM has three heads (pattern classification, length
classification, magnitude regression). The relative loss weights are not specified.
**We chose:** `loss_weights=[0.05, 0.01, 0.94]` — magnitude-dominant
([`fts-diffusion-ref/models/model_params.py:67`](fts-diffusion-ref/models/model_params.py#L67)).
Identical to authors' released values.
**Why:** kept the released values. Note: the pattern-classification weight of 0.05
is very low; it is worth checking (during H2) whether this weighting contributes
to the near-absorbing transition matrix referenced in
GENERATION_FAILURE_ANALYSIS.md §Problem 1.

---

### 10. Initial segment for autoregressive generation
**Paper says:** generation is started from an "initial segment"; the policy for
choosing this segment is not specified.
**We chose:** `get_init_state_by_index(sample_idx)` /
`init_first_segment(...)` selects the first segment of the test set (or a
random segment from the SISC pool in the `random_init` protocol).
[`fts-diffusion-ref/models/sampling.py:80`](fts-diffusion-ref/models/sampling.py#L80)
and per-protocol logic in the TATR/TMTR notebooks.
**Why:** mirrors authors' downstream experiment scripts. Different policies
(`authors` / `split` / `random_init`) are exposed as a protocol switch in our
notebooks; the paper does not distinguish them.

---

### 11. PEM hidden dimension and embedding dimension
**Paper says:** the PEM is described as a small network over `(pattern, length,
magnitude)` triples; specific dimensions are not stated.
**We chose:** `evo_embed_dim=196`, `evo_hidden_dim=32`
([`fts-diffusion-ref/models/model_params.py:61-62`](fts-diffusion-ref/models/model_params.py#L61)).
Identical to authors' released values.
**Why:** kept the released values.
