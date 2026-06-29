# FTS-Diffusion generation failure — PEM collapse + PGM degeneration

**Status:** confirmed from code inspection and from the team's own saved synthetic
trajectories (`synthetic/<asset>/k*/{split,single}/run_*`).
**Scope:** both failures are rooted in the **authors' released code / design**, not in
our experiment scripts (see *Provenance* below).

---

## TL;DR

At sampling time the FTS-Diffusion generator is **degenerate**: it emits essentially
**one segment, tiled thousands of times**. The synthetic price series is therefore a
near-perfect straight line (the "unrealistic linear drift"), and all 30 "runs" of an
experiment are nearly identical. Two independent failures combine:

1. **PEM collapse** — the pattern-evolution Markov chain is rolled out with `argmax`
   (deterministic) and converges to a fixed point (S&P 500) or a 2-cycle (GOOG, ZC=F).
2. **PGM degeneration** — conditioned on a pattern, the generation module does *not*
   reproduce that pattern: it outputs a near-flat segment with a tiny spurious upward
   slope, with the pattern shape and the magnitude `β` both lost.

The upward drift seen in every synthetic series is the composition of the two: the PGM
produces a small spurious `+Δ` per segment, the collapsed PEM repeats it without end.

---

## Problem 1 — PEM collapses (the Markov chain does not evolve)

### Mechanism
`state_evolution_ftsdiffusion`
([`fts-diffusion-ref/models/sampling.py:22-38`](../fts-diffusion-ref/models/sampling.py#L22-L38),
identical to authors' [`code_from_authors/codes/models/sampling.py:22-38`](../code_from_authors/codes/models/sampling.py#L22-L38))
predicts the next state with **`torch.argmax`** for the pattern and the length, and the
raw regression output for the magnitude — **no sampling**:

```python
# sampling.py:30  (pattern)
pred_pattern = torch.argmax(probabilities_pattern, dim=1).unsqueeze(0)
# sampling.py:33  (length)
pred_length  = torch.argmax(probabilities_length,  dim=1).unsqueeze(0) + l_min
# sampling.py:34  (magnitude — raw regression output, no noise)
pred_magnitude = pred[:, n_patterns+range_length:].float()
```

So the autoregressive rollout `next_state = f(state)` is a **deterministic map**.
Iterated from a fixed initial state it must converge to a fixed point or a short cycle.
This is amplified by the *learned* PEM weights: the transition network has learned a
near-absorbing self-transition for one pattern, so even with sampling that pattern would
dominate — `argmax` simply makes the collapse total.

### Evidence (team's saved `run_*_states.npy`, rows = `(pattern, length, magnitude)`)

| Asset | segments | dominant pattern | α | β | regime |
|---|---|---|---|---|---|
| sp500 k14 | 2520 | **p=12 used 99.6 %** | ≈10 (l_min) | ≈31.4 | fixed point |
| goog k11  | 2520 | p=9 used **exactly 50 %** | ≈10 | ≈0.8 | period-2 cycle |
| zcf k11   | 1867 | p=6 used **exactly 50 %** | ≈13.5 | ≈26.6 | period-2 cycle |

Runs 00/01/02 are identical — the dynamics are deterministic.

---

## Problem 2 — PGM does not reproduce the conditioning pattern

### The contradiction
On S&P 500 the collapsed pattern is **p=12**, whose SISC centroid is a strong
**downtrend** (`first 0.899 → last 0.040`, net **−0.859`). Pattern 11 is also a
downtrend (net −0.816). Yet every synthetic series **drifts upward** (≈1253 → 7085 over
25 200 days).

### Evidence — actual generated segments extracted from the trajectory

| segment | pattern | len | first | last | net | internal range |
|---|---|---|---|---|---|---|
| 50  | 12 | 10 | 1309.20 | 1311.56 | **+2.35** | 2.4 |
| 100 | 12 | 10 | 1426.25 | 1428.59 | **+2.34** | 2.3 |
| 500 | 12 | 10 | 2361.80 | 2364.14 | **+2.34** | 2.3 |

Every `p=12` segment is a **monotone tiny up-creep**: net `+2.34`, internal range ≈2.3.
It is **not** a downtrend and it does **not** have magnitude ≈31 (its own `β`). The PGM,
conditioned on a downtrend pattern of range ≈0.86, emits a degenerate near-constant
slightly-rising segment — pattern shape lost, magnitude lost.

### The lost magnitude
The segment range (≈2.3) is far below `β`=31.4 because the authors **commented out** the
range normalization in `segment_generation_ftsdiffusion`
([`fts-diffusion-ref/models/sampling.py:53-55`](../fts-diffusion-ref/models/sampling.py#L53-L55),
identical to authors' [`code_from_authors/codes/models/sampling.py:53-55`](../code_from_authors/codes/models/sampling.py#L53-L55)):

```python
# m_new = (max(new_segment) - min(new_segment))
# new_segment = (new_segment / m_new) * m_t.squeeze(0)
new_segment = new_segment * m_t.squeeze(0)
```

So `β` multiplies a diffusion output whose range is **not** unit-magnitude (here ≈0.073);
the final magnitude is `β × (whatever range the diffusion produced)`, not `β`.

### PGM investigation — the decoder is the degenerate component

Loaded the trained PGM and generated a segment for **every** SISC pattern (p0–p13),
inspecting the diffusion latent `z_` and the decoder output `x_` separately
(`scripts/investigate_pgm.py`):

- **Decoder output `x_` is constant across all 14 patterns**: net ≈ `+0.0745`,
  range ≈ `0.0745`, std across repeats < 0.001. An up-trend pattern and a down-trend
  pattern produce essentially the same segment:
  ```
  p1  (centroid UP,   net +0.87): 0.346 0.378 0.396 0.406 0.412 0.416 0.418 0.419 0.419 0.420
  p12 (centroid DOWN, net -0.86): 0.343 0.375 0.391 0.403 0.410 0.414 0.416 0.418 0.419 0.420
  ```
- The diffusion latent `z_` *does* carry weak pattern information (its net/range vary
  slightly with `p`), but the decoder discards it entirely.

**Root cause:** the Scaling-AE decoder is an LSTM with **`sae_hidden_dim = 1`** —
the LSTM is defined at
[`fts-diffusion-ref/models/scaling_autoencoder.py:58`](../fts-diffusion-ref/models/scaling_autoencoder.py#L58)
(`nn.LSTM(hidden_dim, hidden_dim, n_layers=2, batch_first=True)`), its `hidden_dim` is
fed from `pgm_params['sae_hidden_dim']` set to `1` at
[`fts-diffusion-ref/models/model_params.py:34`](../fts-diffusion-ref/models/model_params.py#L34)
(identical to authors' [`code_from_authors/codes/models/model_params.py:33`](../code_from_authors/codes/models/model_params.py#L33)).
A single hidden unit is essentially no capacity. It collapses to a fixed degenerate
output (a small monotone saturating ramp, net ≈ +0.075) that ignores both the
conditioning pattern *and* the diffusion latent.

So the generated segment is *always* the same ≈+0.075 ramp; `× β` (≈31.4) turns it into
the ≈+2.34 per-segment creep seen in the trajectory. The pattern's down/up shape never
reaches the output — which is exactly why a downtrend pattern (p12) yields an upward
series.

---

## Combined effect

```
collapsed PEM        ->  emits the SAME state (p=12, α=10, β=31.4) every segment
degenerate PGM       ->  turns that state into the SAME ~flat +2.34 segment
additive anchoring   ->  segment - segment[0] + timeseries[-1]  glues them end-to-end
result               ->  one tiny segment tiled ~2520x  =  perfectly linear ramp
```

This explains every odd observation:
- the synthetic price series is a near-perfect straight line;
- all 30 runs of a configuration are nearly identical;
- TATR `split` "works" on S&P 500 only by coincidence — the ramp happens to sweep the
  (also upward-trending) test-price range; it is not evidence that augmentation helps.

---

## Provenance — authors' code, not our error

Every degeneracy-causing line above is present in the authors' released code
(`code_from_authors/codes/`) and was carried into our working copy
(`fts-diffusion-ref/`) without functional change. Specifically:

| Claim | Authors' file:line | Our file:line |
|---|---|---|
| PEM `argmax` for pattern + length, raw regression for magnitude | [`code_from_authors/.../sampling.py:30,33,34`](../code_from_authors/codes/models/sampling.py#L30-L34) | [`fts-diffusion-ref/models/sampling.py:30,33,34`](../fts-diffusion-ref/models/sampling.py#L30-L34) |
| Magnitude range-normalization commented out | [`code_from_authors/.../sampling.py:53-55`](../code_from_authors/codes/models/sampling.py#L53-L55) | [`fts-diffusion-ref/models/sampling.py:53-55`](../fts-diffusion-ref/models/sampling.py#L53-L55) |
| `sae_hidden_dim = 1` in `pgm_params` | [`code_from_authors/.../model_params.py:33`](../code_from_authors/codes/models/model_params.py#L33) | [`fts-diffusion-ref/models/model_params.py:34`](../fts-diffusion-ref/models/model_params.py#L34) |
| Scaling-AE decoder is `nn.LSTM(hidden_dim, hidden_dim, 2)` | [`code_from_authors/.../scaling_autoencoder.py:58`](../code_from_authors/codes/models/scaling_autoencoder.py#L58) | [`fts-diffusion-ref/models/scaling_autoencoder.py:58`](../fts-diffusion-ref/models/scaling_autoencoder.py#L58) |
| Segment anchoring `segment - segment[0] + timeseries[-1]` | [`code_from_authors/.../sampling.py:92,140`](../code_from_authors/codes/models/sampling.py#L92) | [`fts-diffusion-ref/models/sampling.py:92,140`](../fts-diffusion-ref/models/sampling.py#L92) |

`sampling.py` is **byte-identical** between the two repos except line 7
(`from tqdm.notebook import tqdm` → `from tqdm import tqdm`, an environment fix). All
core sampling logic (lines 22–145) is unchanged.

### What is still ours to rule out

Two confounds prevent escalating "these lines are degenerate" to "the authors' published
method is broken":

1. **No checkpoints shipped.** `code_from_authors/codes/trained_models/` does not
   exist; the authors' release does not include pretrained PGM/PEM weights. Every
   reproduction depends on retraining from their training scripts. We cannot compare
   our generations against the authors' "official" outputs.
2. **We undertrained.** Our [`fts-diffusion-ref/models/model_params.py`](../fts-diffusion-ref/models/model_params.py)
   uses PGM `n_epochs=30` (line 48) and PEM `n_epochs=60` (line 65) vs. the authors'
   defaults of `n_epochs=400` and `n_epochs=1000` in
   [`code_from_authors/codes/models/model_params.py:47,64`](../code_from_authors/codes/models/model_params.py#L47).
   The 13×/16× gap is a compute-driven reduction; it is also the first thing a
   peer-reviewer will challenge.

These confounds are addressed by the falsification experiments in the next section.

---

## Alternative hypotheses — what would have to be true for our analysis to be wrong

Three concrete hypotheses could explain the observed degeneracy *without* the failure
being structural to the method. Each is paired with a falsification experiment.

### H1. Undertraining
**Claim under stress:** the PGM decoder collapse and the PEM fixed-point are training
artifacts of our reduced epoch budget (PGM 30/400, PEM 60/1000), not architectural.

**Falsification.**
- Change: in `fts-diffusion-ref/models/model_params.py`, set `pgm_params['n_epochs']=400`
  and `pem_params['n_epochs']=1000` (matching authors' defaults). Also restore
  `pgm_params['pcdm_n_steps']=100` (line 37 currently 30 vs. authors' 100 on line 36).
- Run: `python scripts/train_fts_diffusion.py` on the S&P 500 dataset (identical
  data path used for the current checkpoints), then `python scripts/investigate_pgm.py`.
- Observable: per-pattern decoder output `x_` across `p0..p13`.
- **Falsifier:** if the cross-pattern std of `x_[t]` exceeds **0.01** at any t,
  or if the per-pattern net (`x_[-1] - x_[0]`) varies by more than **0.05** across
  patterns, undertraining was the cause. If it stays at the current ≈ 0 across
  patterns, the collapse is architectural.

### H2. Seed sensitivity
**Claim under stress:** SISC k-means and PGM/PEM training both run under `SEED=42`
(hard-coded in [`fts-diffusion-ref/models/model_params.py:6`](../fts-diffusion-ref/models/model_params.py#L6) and
[`pattern_recognition_module.py`](../fts-diffusion-ref/models/pattern_recognition_module.py)).
The collapse could be one unlucky seed — a different SISC clustering might produce
patterns that the decoder *can* represent, or a different training seed might find a
non-degenerate minimum.

**Falsification.**
- Change: parametrize `SEED` (e.g. via env var) and run training three times with
  `SEED ∈ {0, 1, 2}`.
- Run: full PGM training × 3, then `scripts/investigate_pgm.py` per checkpoint.
- Observable: cross-pattern std of decoder `x_` per seed.
- **Falsifier:** if at least one seed yields cross-pattern std > 0.01, the collapse
  is seed-driven. If all three seeds yield ≈ 0 std across patterns, the collapse
  is not seed-driven.

### H3. Capacity is not the cause
**Claim under stress:** `sae_hidden_dim=1` is *not* the bottleneck — something else
(e.g. PCDM noise schedule, loss weighting) is responsible, and increasing decoder
capacity would not help.

**Falsification.**
- Change: in `fts-diffusion-ref/models/model_params.py:34`, set `sae_hidden_dim=16`
  (then a second run with `sae_hidden_dim=64`). Keep everything else at authors'
  defaults (per H1). Note: `sae_input_dim` and `sae_output_dim` stay at 1; only the
  hidden dimension of the encoder/decoder LSTMs changes.
- Run: retrain PGM (full epochs per H1), then `scripts/investigate_pgm.py`.
- Observable: per-pattern decoder output as in H1.
- **Falsifier:** if cross-pattern std *and* per-pattern net still ≈ 0 even at
  `sae_hidden_dim=64`, the decoder degeneracy is not capacity-driven and the
  "single hidden unit" explanation in §Problem 2 is wrong.

### A note on author contact
A 1-unit LSTM hidden state in the decoder of a published generative model is so
extreme that it is worth asking the authors directly whether `sae_hidden_dim=1` is the
value used for the paper's headline results, or whether it is a residual debug value
in the released config. The same goes for the commented-out magnitude normalization —
the released code path bypasses the formulation in the paper. A short email before
publishing a critical follow-up is cheaper than a contested correction afterward.

---

## Claims that need more evidence before publication

The analysis above contains some claims that are not yet supported at
publication-rebuttal strength. Listed here so they are not asserted as proven in any
write-up until the evidence is in.

- **"PGM does not reproduce the conditioning pattern."** The evidence
  (§Problem 2 / *PGM investigation*) was produced with a PGM trained for 30 epochs
  (vs. authors' 400). The claim is sound *for our checkpoint*, but to claim it for
  *the method* requires the H1 + H3 experiments above.
- **"PEM transition network has learned a near-absorbing self-transition for one
  pattern, so even with sampling that pattern would dominate"** (§Problem 1 /
  *Mechanism*). We have not actually run the multinomial-sampled rollout. The
  evidence supports "deterministic argmax produces a fixed point/2-cycle"; it does
  not yet support the stronger claim about the learned transition distribution.
  To support it, dump `probabilities_pattern` from the trained PEM at the collapsed
  state and report the actual distribution — if one entry is > 0.5 and dominates,
  the claim is justified; if the distribution is closer to uniform, `argmax` alone
  is responsible.
- **"Authors' method is broken"** (implicit in the TL;DR framing). Until H1 is run
  (and ideally H2/H3), the defensible claim is the narrower one: *the authors'
  released code, trained per the released configuration, does not in our hands
  produce realistic series.* This is a reproducibility finding; it becomes a
  method-failure finding only after the alternative hypotheses are ruled out.

---

## Suggested fixes (for the replication / limitations section)

1. **PEM** — sample the next state instead of `argmax`:
   `torch.multinomial(probabilities_pattern, 1)` for pattern and length; add noise to /
   model a distribution for the magnitude.
2. **PGM** — the Scaling-AE decoder needs real capacity: `sae_hidden_dim = 1` must be
   increased (e.g. 16–64) and the PGM retrained, otherwise the decoder cannot represent
   distinct pattern shapes. Also re-enable the range normalization (uncomment the
   `m_new` block) so the segment magnitude actually equals `β`.
3. Treat the current "linear drift" synthetic series as a **known artifact** — it should
   be flagged in the presentation's *Limitations* section, not used as evidence.
