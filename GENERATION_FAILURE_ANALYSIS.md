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
`state_evolution_ftsdiffusion` (`fts-diffusion-ref/models/sampling.py`, ~line 22)
predicts the next state with **`torch.argmax`** for the pattern and the length, and the
raw regression output for the magnitude — **no sampling**:

```python
pred_pattern = torch.argmax(probabilities_pattern, dim=1)   # deterministic
pred_length  = torch.argmax(probabilities_length,  dim=1)   # deterministic
pred_magnitude = pred[:, n_patterns+range_length:]          # regression mean
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
range normalization in `segment_generation_ftsdiffusion`:

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

**Root cause:** the Scaling-AE decoder is an LSTM with **`sae_hidden_dim = 1`**
(`pgm_params` in the authors' `model_params.py`) — a single hidden unit, essentially no
capacity. It collapses to a fixed degenerate output (a small monotone saturating ramp,
net ≈ +0.075) that ignores both the conditioning pattern *and* the diffusion latent.

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

- `state_evolution_ftsdiffusion` with `argmax` is **identical** in
  `code_from_authors/codes/models/sampling.py` and in `fts-diffusion-ref/models/sampling.py`.
- The commented-out magnitude normalization is in the authors' released code.

What is still ours to rule out: whether the **trained PGM/PEM checkpoints** we use are
degenerate because of our training run, or inherited from the authors' checkpoints.

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
