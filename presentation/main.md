# Motivation

:::::::: frame
The problem: financial time series are hard to generate

::::::: columns
:::: column
0.55

::: block
Empirical stylized facts

- Heavy-tailed returns; rare but consequential crises.

- Volatility clustering, long memory in $|r_t|$.

- Scale-invariant patterns at irregular timescales.

- Single realised history (no resampling).
:::
::::

:::: column
0.42

::: alertblock
Why standard generators fail Fixed-window models segment in [calendar
time]{.alert}, not [pattern time]{.alert}. The same shape at twice the
duration becomes a different sample.
:::
::::
:::::::
::::::::

# Architecture overview

::: frame
The modelling assumption: scale-invariant patterns
:::

::: frame
One pipeline, two phases: training and generation
:::

# SISC: three animated steps

:::::: frame
\<1-2\>DTW: Align in time before measuring distance

::::: columns
::: column
0.56\
:::

::: column
0.42

- **The Problem:** Pointwise distance would mistakenly conclude *"these
  are very different"* because their peaks misalign in time.

- **The Solution:** DTW permits non-linear mapping, concluding *"same
  shape, just stretched in time"* $\to$ small distance.

- **Application:** This property is exactly what we need for financial
  motifs that recur with varying durations.
:::
:::::
::::::

::: frame
SISC, Step 1: Initializing centroids (K-means++ for DTW)

**Sampling rule:**
$\Pr\!\bigl(p_{k+1}=X[t:t+l_{\max}]\bigr)\;\propto\;\min_{k'\le k}\mathop{\mathrm{DTW}}\!\bigl(X[t:t+l_{\max}],\,p_{k'}\bigr).$\
:::

::: frame
SISC, Step 2: Greedy segmentation $+$ DBA centroid update
:::

::: frame
SISC, Step 3: Iterate until convergence
:::

::: frame
SISC output: every segment becomes a labelled triplet
:::

::: frame
One SISC run builds the dataset for every downstream module
:::

:::::: frame
Limits of the SISC algorithm

::: block
Local minima risk SISC is a non-convex optimisation: greedy segmentation
followed by iterative centroid update. It can stall when motifs are
noisy or weakly separated.
:::

::: block
Initialization sensitivity Bad initial centroids yield a bad final
segmentation. This is exactly why we use K-means++ for the seeding step
(back-reference to the pipeline slide).
:::

::: alertblock
Hyperparameter sensitivity $K$ and $[l_{\min},l_{\max}]$ must be chosen
up front. Mis-set values either over-fragment (too many tiny motifs) or
under-segment (one motif swallows the series).
:::

*Inherent to any non-convex clustering procedure on time-series data: we
don't take a clean segmentation for granted.*
::::::

# Pattern Evolution module(PEM)

::::::::: frame
Pattern evolution network

#### Classification for patterns, regression for scales

::: block
Transition network
$$(\hat{p}_{m+1},\hat{\alpha}_{m+1},\hat{\beta}_{m+1}) = \phi(p_m,\alpha_m,\beta_m).$$
:::

::::::: columns
:::: column
0.48

::: alertblock
Outputs

- next pattern $p_{m+1}$;

- next duration scale $\alpha_{m+1}$;

- next magnitude scale $\beta_{m+1}$.
:::
::::

:::: column
0.48

::: alertblock
Learning tasks

- pattern: classification;

- duration: regression;

- magnitude: regression.
:::
::::
:::::::

$$\mathcal{L}(\phi)
=
\mathbb{E}_{x_m}\!\bigl[
\ell_{CE}(p_{m+1},\hat{p}_{m+1})
+\|\alpha_{m+1}-\hat{\alpha}_{m+1}\|_2^2
+\|\beta_{m+1}-\hat{\beta}_{m+1}\|_2^2\bigr].$$
:::::::::

# Generation module (PGM)

::: frame
Image Diffusion

#### Destroy an image with noise, then learn to reverse the process
:::

:::: frame
Generation Module Pipeline

::: block
Main Idea
$$p_\theta(\,\cdot\mid p,\alpha,\beta)\approx q(\,\cdot\mid p,\alpha,\beta)$$
Trying to learn the conditional distribution function
$q(\,\cdot\mid p,\alpha,\beta)$, where $p$ is the pattern type, $\alpha$
the duration scale, and $\beta$ the magnitude scale.
:::
::::

::::::::: frame
Scaling autoencoder

#### From variable-length segments to fixed-length latent sequences

::::::: columns
:::: column
0.48

::: alertblock
Encoder Maps a segment to a padded latent sequence: $$x_m
\xrightarrow[\text{packed using }\alpha_m]{\text{encoder}}
z_m \in \mathbb{R}^{l_{\max}\times 1}.$$

**Released code:**\
$\to$ 2-layer LSTM, hidden dim $1$\
$\to$ linear projection, dim $1$
:::
::::

:::: column
0.48

::: alertblock
Decoder Maps the latent sequence back to a segment: $$z_m
\xrightarrow[\text{unpacked using }\alpha_m]{\text{decoder}}
\hat{x}_m.$$

**Released code:**\
$\to$ 2-layer LSTM, hidden dim $1$\
$\to$ linear output, dim $1$
:::
::::
:::::::

::: block
Role of $\alpha_m$ $\alpha_m$ is the segment duration/length used for
packing the valid timesteps. It is not an additional learned latent
coordinate.
:::
:::::::::

:::: frame
Loss: reconstruction term

::: center
Pattern generation loss objective
:::
::::

:::::: frame
Latent residual for diffusion

::::: columns
::: column
0.70 **One Step Forward Process in Latent:**\
$$q(z^i\mid z^{i-1},p)
=
\mathcal{N}\!\left(
z^i;\sqrt{1-\gamma_i}\,(z^{i-1}-p),\gamma_i I
\right)$$
:::

::: column
0.28 $p$: reference pattern\
$\gamma_i$: diffusion noise at step $i$
:::
:::::

**Center at $p$ and diffuse the residual $z^{i-1}-p$.**

![image](./figures/generation_module/pgm_latent_residual_view.png){width="95%"}
::::::

::: frame
Schematic Visualization: Froward Process
![image](./figures/generation_module/pgm_forward_static_4stage_clean.png){width="93%"}
:::

:::::: frame
Pattern-conditioned diffusion: Backward Process

::::: columns
::: column
0.60 **One Step Backward Process in Latent:**\
$\displaystyle
p_\theta(z^{i-1}\mid z^i,p)
=
\mathcal{N}\!\left(
z^{i-1};\mu_\theta(z^i,i,p),\gamma_i I
\right)$
:::

::: column
0.38 $\mu_\theta$: `TCN` denoiser (released code)\
6 residual TCN blocks, kernel $3$\
channels: $48,64,80,80,64,48$\
dilations: $1,2,4,8,16,32$\
time embedding $32$, steps $30$
:::
:::::

![image](./figures/generation_module/pgm_backward_static_4stage_clean.png){width="91%"}
::::::

::::::::: frame
Deriving the diffusion loss

#### Train the denoiser to recover the noise that was added

::::::: columns
:::: column
0.58

::: block
1. Forward noising of the residual+closed form $$r_0 = z_m - p$$ $$r_i =
\sqrt{\bar a_i}\,r_0
+
\sqrt{1-\bar a_i}\,\epsilon$$ $$\epsilon\sim\mathcal{N}(0,I),
\qquad
\bar a_i=\prod_{s=1}^{i}a_s,
\qquad
a_s=1-\gamma_s.$$
:::
::::

:::: column
0.38

::: block
2. Predict noise $$\epsilon_\theta(r_i,i,p)\approx\epsilon$$ Target: the
exact Gaussian noise sampled in the forward process.
:::
::::
:::::::

::: block
Connection to the reverse mean $$\mu_\theta(r_i,i,p)
=
\frac{1}{\sqrt{a_i}}
\left(
r_i
-
\frac{1-a_i}{\sqrt{1-\bar a_i}}\,
\epsilon_\theta(r_i,i,p)
\right)$$ Predicting $\epsilon_\theta$ determines the mean used in the
reverse denoising step.
:::
:::::::::

:::: frame
Loss: diffusion term

::: center
Pattern generation loss objective
:::
::::

# Synthetic data generation

::: frame
Generation: the PEM directs, two modules execute

#### At sampling time only PEM, the diffusion model and the decoder run --- SISC is gone
:::

::: frame
One segment: three different roles for $(p,\alpha,\beta)$
:::

::::::: frame
Training data per asset

#### Strong size asymmetry across the three assets

*Yahoo Finance daily close, [raw prices]{.alert} (no return transform).
Same 80/20 SISC split, same 62.5/37.5 LSTM init/test split across all
assets.*

::: center
:::

::::: columns
::: column
0.55 **Data pipeline**

- **Source:** Yahoo Finance, daily close, no transform.

- **FTS-Diffusion:** fitted on the first 80 % ([SISC + PGM + PEM all see
  only this slice]{.alert}).

- **Downstream LSTM:** init / test carved *inside* the held-out 20 %
  (62.5 / 37.5).
:::

::: column
0.42
:::
:::::
:::::::

# TATR: the downstream benchmark

:::::: frame
TATR: Train on Augmentation, Test on Real

::::: columns
::: column
0.62

- **Train:** `real_init` $\;\oplus\;$ `synth`$[0:Y\!\cdot\!252]$.

- **Test:** real held-out (never seen during training).

- **Sweep:** $Y\in\{0,10,20,\dots,100\}$ years of synthetic.

- **Score:** MAPE on real test, mean over 15 seeds.
:::

::: column
0.36
:::
:::::
::::::

# TATR: synthetic-data protocols

::::::: frame
From synthetic days to LSTM training windows

#### Identical for every protocol --- establish it once

:::: center
::: {style="background-color: NavyBlue!9"}
**Same data budget.** Every protocol hands the LSTM *exactly* the same
$Y\times 189$ context windows --- only the Markov-chain rollout differs.
:::
::::

:::: center
::: minipage
\>0 $\bullet$ **`authors`** --- MC *restarts every year* from the *same*
state $(p_0,l_0,m_0)$. $\bullet$ **`authors`** --- MC *restarts every
year* from the *same* state $(p_0,l_0,m_0)$.

\>0 $\bullet$ **`split`** --- MC runs *once, continuously*; year edges
are only LSTM-window cuts. $\bullet$ **`split`** --- MC runs *once,
continuously*; year edges are only LSTM-window cuts.

\>0 $\bullet$ **`random_init`** --- MC restarts every year from a
*random* state $\sim$ `SEGMENTS_INIT`.
$\bullet$ **`random_init`** --- MC restarts every year from a *random*
state $\sim$ `SEGMENTS_INIT`.
:::
::::
:::::::

::::::: frame
Protocol 1 / 3 --- `authors`

#### Yearly restart from the *same* conditional state --- paper-faithful

:::: center
::: {style="background-color: NavyBlue!9"}
**Same data budget.** Every protocol hands the LSTM *exactly* the same
$Y\times 189$ context windows --- only the Markov-chain rollout differs.
:::
::::

:::: center
::: minipage
\>1 $\bullet$ **`authors`** --- MC *restarts every year* from the *same*
state $(p_0,l_0,m_0)$. $\bullet$ **`authors`** --- MC *restarts every
year* from the *same* state $(p_0,l_0,m_0)$.

\>1 $\bullet$ **`split`** --- MC runs *once, continuously*; year edges
are only LSTM-window cuts. $\bullet$ **`split`** --- MC runs *once,
continuously*; year edges are only LSTM-window cuts.

\>1 $\bullet$ **`random_init`** --- MC restarts every year from a
*random* state $\sim$ `SEGMENTS_INIT`.
$\bullet$ **`random_init`** --- MC restarts every year from a *random*
state $\sim$ `SEGMENTS_INIT`.
:::
::::
:::::::

::::::: frame
Protocol 2 / 3 --- `split`

#### One continuous chain generated up front, then cut --- our re-implementation

:::: center
::: {style="background-color: NavyBlue!9"}
**Same data budget.** Every protocol hands the LSTM *exactly* the same
$Y\times 189$ context windows --- only the Markov-chain rollout differs.
:::
::::

:::: center
::: minipage
\>2 $\bullet$ **`authors`** --- MC *restarts every year* from the *same*
state $(p_0,l_0,m_0)$. $\bullet$ **`authors`** --- MC *restarts every
year* from the *same* state $(p_0,l_0,m_0)$.

\>2 $\bullet$ **`split`** --- MC runs *once, continuously*; year edges
are only LSTM-window cuts. $\bullet$ **`split`** --- MC runs *once,
continuously*; year edges are only LSTM-window cuts.

\>2 $\bullet$ **`random_init`** --- MC restarts every year from a
*random* state $\sim$ `SEGMENTS_INIT`.
$\bullet$ **`random_init`** --- MC restarts every year from a *random*
state $\sim$ `SEGMENTS_INIT`.
:::
::::
:::::::

::::::: frame
Protocol 3 / 3 --- `random_init`

#### Yearly restart from a *random* state --- diagnostic control

:::: center
::: {style="background-color: NavyBlue!9"}
**Same data budget.** Every protocol hands the LSTM *exactly* the same
$Y\times 189$ context windows --- only the Markov-chain rollout differs.
:::
::::

:::: center
::: minipage
\>3 $\bullet$ **`authors`** --- MC *restarts every year* from the *same*
state $(p_0,l_0,m_0)$. $\bullet$ **`authors`** --- MC *restarts every
year* from the *same* state $(p_0,l_0,m_0)$.

\>3 $\bullet$ **`split`** --- MC runs *once, continuously*; year edges
are only LSTM-window cuts. $\bullet$ **`split`** --- MC runs *once,
continuously*; year edges are only LSTM-window cuts.

\>3 $\bullet$ **`random_init`** --- MC restarts every year from a
*random* state $\sim$ `SEGMENTS_INIT`.
$\bullet$ **`random_init`** --- MC restarts every year from a *random*
state $\sim$ `SEGMENTS_INIT`.
:::
::::
:::::::

# SISC: patterns on real data

::: frame
S&P 500: learned motifs ($K=14$)
![image](./figures/sisc/sp500/k14/patterns.pdf){width="88%"}
:::

::: frame
GOOG: learned motifs ($K=11$)
![image](./figures/sisc/goog/k11/patterns.pdf){width="88%"}
:::

::: frame
ZC$=$F (corn futures): learned motifs ($K=11$)
![image](./figures/sisc/zcf/k11/patterns.pdf){width="88%"}
:::

# TATR results: protocol comparisons

::: frame
What the paper reports --- TATR (Fig. 6b)
![image](./figures/paper/fig6_tatr.pdf){width="74%"}

FTS-Diffusion paper (Yu et al., ICLR 2024), Fig. 6(b): TATR prediction
error (MAPE) vs. augmented size --- FTS-Diffusion and three baselines
(RCGAN, TimeGAN, CSDI) on S&P 500, GOOG and ZC$=$F. Dashed line $=$
year-0 MAPE.
:::

:::::: frame
S&P 500, $K=14$: authors vs split

::::: columns
::: column
0.49 **authors**\
![image](./figures/tatr/sp500/k14/authors/final_enhanced.pdf){width="\\linewidth"}
:::

::: column
0.49 **split**\
![image](./figures/tatr/sp500/k14/split/final_enhanced.pdf){width="\\linewidth"}
:::
:::::

Same data budget and same #runs --- only the MC roll-out changes: yearly
re-init ([authors, $+208.6\%$]{style="color: Coral"}) vs one continuous
chain ([split, $-73.3\%$]{style="color: TealAcc"}).
::::::

:::::: frame
S&P 500, $K=14$: authors vs random init

::::: columns
::: column
0.49 **authors**\
![image](./figures/tatr/sp500/k14/authors/final_enhanced.pdf){width="\\linewidth"}
:::

::: column
0.49 **random init**\
![image](./figures/tatr/sp500/k14/random_init/final_enhanced.pdf){width="\\linewidth"}
:::
:::::

Replacing the deterministic re-init $(p_0,\alpha_0,\beta_0)$ with a
random MC state: [$+208.6\%$]{style="color: Coral"} $\rightarrow$
[$+67.7\%$]{style="color: NavyBlue"} --- the fixed seed *hurts*.
::::::

:::::: frame
Authors' checkpoint vs. trained-from-scratch (S&P 500, $K=14$)

#### Same authors protocol; only the PGM$+$PEM weights differ

::::: columns
::: column
0.49 **authors' checkpoint**\
![image](./figures/tatr/sp500/k14/authors/final_enhanced.pdf){width="\\linewidth"}
:::

::: column
0.49 **trained from scratch**\
![image](./figures/tatr/sp500_us/k14/authors/final_enhanced.pdf){width="\\linewidth"}
:::
:::::

Authors' checkpoint [$+208.6\%$]{style="color: Coral"} vs. our
from-scratch PGM$+$PEM weights [$+169.6\%$]{style="color: NavyBlue"}:
same qualitative degradation.
::::::

:::::: frame
GOOG, $K=11$: authors vs split

::::: columns
::: column
0.49 **authors**\
![image](./figures/tatr/goog/k11/authors/final_enhanced.pdf){width="\\linewidth"}
:::

::: column
0.49 **split**\
![image](./figures/tatr/goog/k11/split/final_enhanced.pdf){width="\\linewidth"}
:::
:::::

GOOG $K=11$: yearly re-init explodes
([$+981.7\%$]{style="color: Coral"}); the continuous chain contains it
([$+90.4\%$]{style="color: TealAcc"}).
::::::

:::::: frame
GOOG, $K=11$: authors vs random init

::::: columns
::: column
0.49 **authors**\
![image](./figures/tatr/goog/k11/authors/final_enhanced.pdf){width="\\linewidth"}
:::

::: column
0.49 **random init**\
![image](./figures/tatr/goog/k11/random_init/final_enhanced.pdf){width="\\linewidth"}
:::
:::::

A random MC state instead of the deterministic seed collapses the error:
[$+981.7\%$]{style="color: Coral"} $\rightarrow$
[$+48.3\%$]{style="color: NavyBlue"}.
::::::

:::::: frame
ZC$=$F, $K=11$: authors vs split

::::: columns
::: column
0.49 **authors**\
![image](./figures/tatr/zcf/k11/authors/final_enhanced.pdf){width="\\linewidth"}
:::

::: column
0.49 **split**\
![image](./figures/tatr/zcf/k11/split/final_enhanced.pdf){width="\\linewidth"}
:::
:::::

ZC$=$F $K=11$: here the continuous chain is *worse*
([$+174.6\%$]{style="color: TealAcc"}) than yearly re-init
([$+75.6\%$]{style="color: Coral"}) --- the ranking is asset-dependent.
::::::

:::::: frame
ZC$=$F, $K=11$: authors vs random init

::::: columns
::: column
0.49 **authors**\
![image](./figures/tatr/zcf/k11/authors/final_enhanced.pdf){width="\\linewidth"}
:::

::: column
0.49 **random init**\
![image](./figures/tatr/zcf/k11/random_init/final_enhanced.pdf){width="\\linewidth"}
:::
:::::

random init stays moderate ([$+49.4\%$]{style="color: NavyBlue"}),
slightly below yearly re-init ([$+75.6\%$]{style="color: Coral"}).
::::::

# TMTR

::: frame
What the paper reports --- TMTR (Fig. 6a)
![image](./figures/tmtr/new_tmtr.jpeg){width="100%"}

FTS-Diffusion paper (Yu et al., ICLR 2024), Fig. 6(a): TMTR prediction
error (MAPE) vs. synthetic proportion --- FTS-Diffusion
vs. RCGAN/TimeGAN/CSDI on S&P 500, GOOG and ZC$=$F.
:::

::: frame
TMTR: Train on Mixture, Test on Real

![image](./figures/tmtr/tmtr.jpeg){width="90%"}
:::

# Methodological observations

::: frame
The generation collapses to a single motif

#### Every 252-day block is the same pattern --- the synthetic data is not reliable

![image](./figures/diagnosis/goog_pattern_repetition.pdf){width="96%"}

First two blocks of the synthetic GOOG paths (5 runs). Each 252-day
block is an [identical copy]{.alert} of the previous one: under
`authors` the chain resets and replays the same motif, under `split` it
tiles it continuously --- the generator emits [one motif
forever]{.alert}. The rising downstream MAPE reflects this degenerate
synthetic data, not the protocol.
:::

# Limitations and Extensions

:::: frame
Limitation 1: motif discovery is fragile

#### The whole generator depends on the first segmentation step

::: block
Discrete motif extraction SISC turns a continuous time series into a
finite dictionary of motifs. This is powerful, but it creates several
sensitivities.
:::

- The number of clusters $K$ and length range $[l_{\min},l_{\max}]$ are
  important hyperparameters.

- Greedy segmentation can misplace boundaries between adjacent motifs.

- DTW captures shape similarity, but may ignore economically meaningful
  differences.

- Motifs are learned offline; new market regimes may require
  re-clustering.

[Implication:]{.alert} generation quality is only as good as the learned
motif vocabulary.
::::

:::::::: frame
Limitation 2: local patterns may miss long-range structure

#### Short motifs are not the same as full market dynamics

::::::: columns
:::: column
0.48

::: block
Captured well

- Local shape recurrence

- Heavy-tailed return behavior

- Volatility clustering proxies

- Segment-level scaling
:::
::::

:::: column
0.48

::: block
Harder to capture

- Long-memory dependence

- Regime persistence

- Macro or news conditioning

- Rare crises and structural breaks

- Cross-asset co-movement
:::
::::
:::::::

The learned transition is essentially a compressed state evolution:
$$(p_m,\alpha_m,\beta_m)
\mapsto
(p_{m+1},\alpha_{m+1},\beta_{m+1}).$$
::::::::

:::: frame
Why no predictable drift matters

#### Martingale intuition for financial returns

A standard benchmark assumption in financial market modelling is that
returns should not contain an easily exploitable conditional mean under
the available information:
$$\operatorname{E}\!\left[r_{t+1}\mid \mathcal{F}_t\right] = 0,$$ or,
equivalently, for every admissible test function $g(\mathcal{F}_t)$,
$$\operatorname{E}\!\left[r_{t+1} g(\mathcal{F}_t)\right] = 0 = \left(
\operatorname{Cov}(r_{t+1},g(\mathcal{F}_t))
+
\operatorname{E}[r_{t+1}]\operatorname{E}[g(\mathcal{F}_t)]
\right)..$$

::: block
Financial interpretation Given the information available at time $t$,
the next return should not be systematically predictable. Otherwise, the
generated data may contain [artificial alpha]{.alert}.
:::

- Real financial returns are noisy and close to unpredictable at short
  horizons.

- A good synthetic generator should match stylized facts without
  creating fake trading signals.

- Therefore, distributional realism alone is not enough: we also need a
  conditional-mean diagnostic.
::::

:::: frame
Extension 1: no-predictable-drift diagnostic

#### A fixed test for artificial conditional mean

Use a simple fixed information set: $$\omega_t =
\left(1,\ r_t,\ r_{t-1},\ |r_t|,\ |r_{t-1}|\right)^\top .$$

Define the drift-violation statistic: $$\delta
=
\sup_{\|b\|_2 \leq 1}
\left|
\widehat{\operatorname{E}}
\left[
r_{t+1} b^\top \omega_t
\right]
\right|.$$

For this linear class: $$\delta
=
\left\|
\widehat{\operatorname{E}}
\left[
r_{t+1}\omega_t
\right]
\right\|_2 .$$

::: block
Compare three objects $$\delta_{\text{real}},
\qquad
\delta_{\text{synthetic}},
\qquad
\delta_{\text{null}} .$$ The null can be built by block-shuffling
returns to preserve rough marginal behavior while breaking
predictability.
:::
::::

:::: frame
Extension 2: drift-regularized generation

#### Keep stylized facts, discourage artificial alpha

Modify training with one additional penalty:
$$\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{FTS}}
+
\lambda
\delta_{\text{synthetic}} .$$

- $\mathcal{L}_{\text{FTS}}$: original reconstruction, diffusion, and
  transition losses.

- $\delta_{\text{synthetic}}$: fixed no-predictable-drift statistic.

- $\lambda$: chosen once on validation data.

::: block
Evaluation question Does the penalty reduce predictable drift without
destroying:

- heavy-tailed returns;

- volatility clustering;

- distributional similarity to real data?
:::
::::

:::::::::: frame
Extension 3: broader model directions

::::::::: columns
::::: column
0.48

::: block
Multivariate FTS-Diffusion Generate vector returns:
$$r_t \in \mathbb{R}^d$$ and preserve cross-correlations, co-jumps, and
sector-level dependence.
:::

::: block
Regime-conditioned transitions Condition the transition module on market
state: $$\phi(p_m,\alpha_m,\beta_m,c_m).$$
:::
:::::

::::: column
0.48

::: block
Stress-aware generation Oversample rare but realistic regimes: crises,
crashes, liquidity shocks, and volatility spikes.
:::

::: block
Stronger validation Move beyond distribution matching:

- martingale-style diagnostics;

- trading-rule robustness;

- risk metrics such as VaR and ES;

- out-of-sample transfer to new assets.
:::
:::::
:::::::::
::::::::::

# Conclusions

::: frame
Conclusions

- 3-subsystem architecture is ambitious with many trade-offs

- Many open questions about specific implementations

- May lead to better interpretability of timeseries
:::

::: closingframe
Deli Lin, Eric Shao, Lapo Linossi, Tom Haidinger\
ETH Zurich\
Machine Learning in Finance and Complex Systems\
Supervisor: Alexander Arandjelovic

Thank you. *Questions?*
:::
