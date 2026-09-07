# Experiment 04: Controlled Comparison of Adaptive Conformal Prediction Geometries

## Revision history

- **v1.0 (2026-09-06):** Initial preregistered-style design specification.
- **v1.1 (2026-09-06, pre-implementation amendment):** The eight
  implementation ambiguities identified in the design review were resolved and
  frozen before any implementation:
  1. causal boundary-crossing sequence and window construction (Section 5–7);
  2. exact 40/8/40/8 noise geometry with globally indexed phase (Section 4);
  3. mutually exclusive regime definitions, ramp excluded from headline
     regime metrics (Sections 4, 7, 17);
  4. frozen Quantile LSTM architecture and training configuration following
     the Experiment 03 canonical precedent (Section 9);
  5. replication seed derivation `replication_seed = 1000 + replication_id`
     (Section 16);
  6. no learned-scale arm; D2 reserved for pre-specified diagnostics
     (Section 7);
  7. block-based calibration deferred to a future Experiment 05 (Section 20);
  8. frozen statistical conventions: ddof = 0, paired t-intervals, Spearman
     scale association, standard two-sided Winkler score (Sections 10, 18,
     19).

  No protocol change is permitted after test results are observed.

## Frozen protocol summary

| Quantity | Frozen value |
|---|---|
| Cycle length \(P\) | 96 = 40 low + 8 transition up + 40 high + 8 transition down |
| Warm-up | 128 observations discarded as targets; phase not reset |
| D1 | 5,952 targets (54 training + 8 validation cycles) |
| D2 | 2,400 targets (pre-specified diagnostics only; no learned-scale arm) |
| D3 | 1,152 targets (conformal calibration) |
| Test | 2,400 targets (final evaluation) |
| Sequence length \(L\) | 32 |
| \(\alpha\) | 0.10 (nominal coverage 0.90) |
| Replications \(R\) | 30 |
| Conformal rank \(k\) | \(\lceil 1153 \times 0.90 \rceil = 1038\) |
| Primary scale | \(s_t = \operatorname{std}(y_{t-32},\ldots,y_{t-1};\ \mathrm{ddof}=0)\), causal |
| Architecture | 1-layer LSTM (hidden 32) on the lag sequence; phase features concatenated with \(h_t\); `Linear(34 -> 2)` quantile head |
| Training | pinball loss at \(\tau \in \{0.10, 0.90\}\); Adam, lr \(10^{-3}\); full batch; 40 epochs (Experiment 03 configuration) |
| Seeds | `replication_seed = 1000 + replication_id` seeds DGP, model initialization, and training randomness |
| Methods | M0–M3 plus two ORACLE diagnostics |
| Primary comparison | Replacement − Additive under the same dataset, same Quantile LSTM, same local scale, same calibration set, and same test set |

## 1. Research question

The purpose of Experiment 04 is to determine whether center-based
replacement/relative conformal prediction provides better heteroskedastic
interval adaptation than additive CQR when the same base Quantile LSTM and
the same local scale estimator are used.

The central question is:

> Does replacement conformal geometry overcome the interval-width dilution
> inherent in additive CQR?

The answer must not be claimed in advance.

## 2. Scientific motivation

- Experiment 01 established the Quantile LSTM + CQR baseline.
- Experiment 02 demonstrated the limitation of standard CQR under
  heteroskedasticity.
- Experiment 03 studied local-scale conformal adaptation using a 1,000-point
  prototype. It showed that local scaling can produce regime-dependent
  interval widths, but additive geometry retains the raw quantile width as an
  additive component.

Experiment 04 is therefore a controlled comparison of interval geometry. The
primary scientific variable is:

    additive geometry
    versus
    replacement geometry

The base predictive model and the local scale estimator remain fixed between
the two methods.

## 3. Main hypotheses

- **H1.** Standard CQR improves marginal coverage but does not strongly adapt
  interval width between low- and high-noise regimes.
- **H2.** Local-scale scaled additive CQR adapts interval width to local noise
  level, but the raw Quantile LSTM interval creates an additive width
  component that limits the achievable width ratio.
- **H3.** Local-scale replacement CQR produces stronger heteroskedastic width
  adaptation than local-scale scaled additive CQR when both use the same base
  model and local scale.
- **H4.** Oracle replacement CQR provides a diagnostic upper benchmark for
  width adaptation.
- **H5.** Any observed differences in adaptive performance must be interpreted
  separately from temporal calibration validity because the data are dependent
  time-series observations.

These hypotheses are empirical and must not be presented as confirmed before
the experiment is run.

## 4. Data-generating process

Use the cyclic heteroskedastic DGP with cycle length \(P = 96\). For each
\(t\) (indexed over the entire generated series, including the warm-up):

\[
\phi_t = (t \bmod 96)\,/\,96 .
\]

Signal:

\[
f(\phi) = \sin(2\pi\phi) + 0.5\cos(4\pi\phi).
\]

Observations:

\[
y_t = f(\phi_t) + \sigma(\phi_t)\,\epsilon_t,
\qquad \epsilon_t \sim \mathcal{N}(0,1)
\]

independently across \(t\).

### Exact noise geometry (frozen in v1.1)

Within each cycle, with cycle-relative slot \(j = 0, \ldots, 95\):

| Slots | Regime | \(\sigma\) |
|---|---|---|
| \(j = 0\)–\(39\) (40 slots) | low plateau | 0.2 |
| \(j = 40\)–\(47\) (8 slots) | raised-cosine transition up | \(0.2 + 0.8\cdot\frac{1-\cos(\pi m/9)}{2},\ m = j-39\) |
| \(j = 48\)–\(87\) (40 slots) | high plateau | 1.0 |
| \(j = 88\)–\(95\) (8 slots) | raised-cosine transition down | \(1.0 - 0.8\cdot\frac{1-\cos(\pi m/9)}{2},\ m = j-87\) |

so that \(40 + 8 + 40 + 8 = 96\). The transition slots evaluate the
raised-cosine profile at the strictly interior fractions \(m/9\),
\(m = 1, \ldots, 8\), so no transition slot coincides with a plateau value and
the plateau/transition counts remain exact.

The phase is defined globally from the generated-series index; the warm-up
does not reset the phase. Because the warm-up is \(128 = 96 + 32\) and every
segment length is a multiple of 96, all four target segments begin at the same
cycle phase \(\phi = 32/96\), giving reproducible cycle alignment.

The approximate low/high noise ratio is:

\[
\sigma_H\,/\,\sigma_L = 1.0\,/\,0.2 = 5 .
\]

Regime labels are known from the DGP and are used ONLY for evaluation. They
must never be supplied to the non-oracle prediction methods.

## 5. Feature representation

Features are constructed from the single continuous generated series. For
every target \(y_t\), use:

\[
x_t = [\,y_{t-1},\ \ldots,\ y_{t-32},\ \sin(2\pi\phi_t),\ \cos(2\pi\phi_t)\,].
\]

Therefore:

- the lag history contains realized local variability;
- the phase features identify the deterministic cyclic structure and are the
  target-time features;
- the true future noise realization is unavailable.

The 32-lag window may reach back across segment boundaries and into the
warm-up block; it contains only observations strictly preceding the target
index (causal). Use sequence length 32. Do not provide true \(\sigma(\phi_t)\)
to the non-oracle methods.

## 6. Warm-up

Discard an initial 128 observations as warm-up. Document this explicitly. The
warm-up observations are not used as training/calibration/test targets. They
may serve as causal lag and local-scale context for the first valid targets,
and they do not reset the phase.

## 7. Chronological data splits

Use cycle-aligned chronological segments:

| Segment | Observations | Cycles | Purpose |
|---|---|---|---|
| D1 | 5,952 | 62 complete cycles (54 training + 8 validation) | fit the Quantile LSTM |
| D2 | 2,400 | 25 complete cycles | out-of-fold development segment; pre-specified diagnostics only (amended in v1.1, see below) |
| D3 | 1,152 | 12 complete cycles | conformal calibration |
| Test | 2,400 | 25 complete cycles | final evaluation |

The segment boundaries must remain chronological. Observations are never
randomly shuffled across train/calibration/test.

### Sequence and window construction (frozen in v1.1)

Only target observations are assigned to D1/D2/D3/Test. Historical context —
the 32-lag feature window and the 32-observation local-scale window — may
cross segment boundaries causally. For example, the first D3 target is allowed
to use observations from the end of D2. Those observations are already
observed at prediction time, so this is **not leakage**. This preserves
\(n_{\text{cal}} = 1152\) exactly and therefore \(k = 1038\).

### Regime composition

With the 40/8/40/8 cycle structure, each cycle contains exactly 40 low, 16
ramp (8 up + 8 down), and 40 high observations, so every segment preserves the
approximate regime composition:

- low ≈ 41.7% (exactly 40/96 per cycle)
- high ≈ 41.7% (exactly 40/96 per cycle)
- ramp ≈ 16.7% (exactly 16/96 per cycle)

For example, the test segment contains exactly 1,000 low, 400 ramp, and 1,000
high targets, and D3 contains exactly 480 low, 192 ramp, and 480 high
calibration targets.

### Role of D2 (amended in v1.1)

D2 is not used to fit a learned scale estimator in the primary Experiment 04
protocol; the primary experiment contains no learned-scale arm. D2 provides a
clean chronological separation between base-model development (D1) and
conformal calibration (D3), and it is reserved for pre-specified diagnostics
only. It is never used to select a method based on test performance. This
keeps the experiment focused on the same scale estimator with different
interval geometry, which is the actual research question.

## 8. Conformal calibration

Use:

\[
\alpha = 0.10,
\qquad \text{target nominal coverage } 1-\alpha = 0.90 .
\]

For \(n_{\text{cal}} = 1152\), use:

\[
k = \lceil (n_{\text{cal}} + 1)(1-\alpha) \rceil
  = \lceil 1153 \times 0.90 \rceil
  = 1038 .
\]

Use the 1038th ordered calibration score. This is exact because, under causal
boundary-crossing construction, all 1,152 D3 observations are calibration
targets. Verify this programmatically (assert \(n_{\text{cal}} = 1152\) and
\(k = 1038\)).

## 9. Base quantile model

Use one common Quantile LSTM for all methods. The architecture and training
configuration below are frozen (v1.1) and must not change between conformal
methods.

### Architecture (frozen in v1.1)

Following the Experiment 03 canonical precedent
(`03_clean_local_scale_cqr.ipynb`), with the phase-feature extension required
by this design:

- one `nn.LSTM` layer with `input_size = 1`, `hidden_size = 32`,
  `num_layers = 1`, `batch_first = True`, consuming the univariate lag
  sequence \(y_{t-32}, \ldots, y_{t-1}\);
- the last-step hidden state \(h_t\) (32-dimensional) is concatenated with the
  two target-time phase features \(\sin(2\pi\phi_t), \cos(2\pi\phi_t)\),
  giving a 34-dimensional vector;
- a linear quantile head `nn.Linear(34, 2)` maps this vector to
  \((Q_L, Q_U)\) at \(\tau = 0.10\) and \(\tau = 0.90\).

\[
h_t = \operatorname{LSTM}(y_{t-32:t-1}),
\qquad
[\,h_t,\ \sin(2\pi\phi_t),\ \cos(2\pi\phi_t)\,] \rightarrow (Q_L, Q_U).
\]

An alternative representation feeding
\([y_{t-j}, \sin(2\pi\phi_t), \cos(2\pi\phi_t)]\) at every timestep (input
dimension 3) was considered and not adopted: concatenating the phase after the
LSTM keeps the 32-lag sequence genuinely temporal and makes the architecture
unambiguous.

No input normalization is applied; the model operates on raw observations,
mirroring Experiment 03.

### Training (frozen in v1.1)

- Loss: pinball (quantile) loss at \(\tau = 0.10\) and \(\tau = 0.90\),
  averaged over both quantiles, identical to Experiment 03.
- Optimizer: Adam with learning rate \(10^{-3}\).
- 40 full-batch epochs — copied unchanged from the Experiment 03 canonical
  configuration.
- Train only on D1. D1's 8 validation cycles may be used for
  model-selection/early-stopping decisions if required; Experiment 03 itself
  used fixed epochs without early stopping. Any early-stopping rule must be
  specified before implementation and use only D1's validation cycles.
- Determinism settings mirrored from Experiment 03:
  `torch.use_deterministic_algorithms(True, warn_only=True)`,
  `torch.backends.cudnn.deterministic = True`,
  `torch.backends.cudnn.benchmark = False`; device recorded (Experiment 03
  precedent is CPU).
- Any deviation from the Experiment 03 training configuration must be fixed
  and recorded before implementation, never after observing test results. No
  test-based tuning is permitted.

Once the base model is fixed, generate predictions for D2, D3, and Test. All
methods must use identical base predictions within each Monte Carlo
replication.

## 10. Primary scale estimator

For the primary geometry comparison, use a local variability estimator derived
only from historical observations. Use the preceding 32 observations:

\[
s_t = \operatorname{std}(y_{t-32}, \ldots, y_{t-1};\ \mathrm{ddof}=0).
\]

The standard-deviation convention is frozen to the population convention
(`ddof = 0`) (v1.1). The estimator must be causal; like the feature windows,
the scale window may cross segment boundaries causally. Do not use \(y_t\),
future observations, or the true \(\sigma_t\) to construct the non-oracle
scale.

The purpose is to hold scale estimation fixed while comparing interval
geometry.

## 11. Method M0 — Raw Quantile Interval

Use:

\[
I_0(x) = [\,Q_L(x),\ Q_U(x)\,].
\]

No conformal correction. This is the base predictive interval.

## 12. Method M1 — Standard CQR

Define:

\[
r_i = \max\{Q_L(x_i) - y_i,\ \ y_i - Q_U(x_i),\ \ 0\}.
\]

Calibrate \(q\) using the 1038th ordered calibration score. Construct:

\[
I_1(x) = [\,Q_L(x) - q,\ \ Q_U(x) + q\,].
\]

## 13. Method M2 — Local-Variability Scaled Additive CQR

Use:

\[
S_i = r_i / s_i,
\]

where \(r_i\) is the standard CQR score. Calibrate \(q_A\) from the 1038th
ordered scaled score. Construct:

\[
I_2(x) = [\,Q_L(x) - q_A\,s(x),\ \ Q_U(x) + q_A\,s(x)\,].
\]

Call this method exactly **Local-Variability Scaled Additive CQR**. Do NOT
call it replacement CQR.

## 14. Method M3 — Local-Variability Replacement CQR

Define the center:

\[
c(x) = (Q_L(x) + Q_U(x))\,/\,2 .
\]

Define:

\[
S_i = |y_i - c(x_i)|\,/\,s_i .
\]

Calibrate \(q_R\) from the 1038th ordered replacement score. Construct:

\[
I_3(x) = [\,c(x) - q_R\,s(x),\ \ c(x) + q_R\,s(x)\,].
\]

Call this **Local-Variability Replacement CQR**. This is the primary proposed
method.

## 15. Oracle diagnostics

Implement two oracle methods. The oracle scale is:

\[
s_t = \sigma(\phi_t).
\]

This true scale is used ONLY for oracle diagnostics.

- **Oracle additive:** \([\,Q_L - q_{OA}\,\sigma,\ \ Q_U + q_{OA}\,\sigma\,]\)
- **Oracle replacement:** \([\,c - q_{OR}\,\sigma,\ \ c + q_{OR}\,\sigma\,]\)

Label these clearly as ORACLE. Do not mix oracle results into the primary
non-oracle comparison.

## 16. Monte Carlo replications

Run \(R = 30\) independent replications initially. Every replication must use
a different DGP seed.

### Seed policy (frozen in v1.1)

With replication id \(r = 0, \ldots, 29\):

    replication_seed = 1000 + replication_id

This single seed deterministically drives all randomness in the replication:
the DGP generation, NumPy, PyTorch, model initialization, and training
randomness. Each replication therefore represents a genuinely independent
experiment. Record the seed for every replication.

Within each replication:

- generate exactly one dataset;
- train exactly one base Quantile LSTM;
- produce one shared set of base predictions;
- evaluate every method on the same test observations.

This creates paired comparisons. Do not tune individual methods separately
using test results.

## 17. Primary metrics

For every method report:

1. Marginal coverage
2. Mean interval width
3. Median interval width
4. Low-regime coverage
5. High-regime coverage
6. Low-regime mean width
7. High-regime mean width
8. High/low width ratio

### Regime metric definitions (frozen in v1.1)

The three regimes are mutually exclusive: **Low** is the 40-point low
plateau, **High** is the 40-point high plateau, and **Ramp** is the union of
both 8-point transition regions. Headline regime metrics report low and high
only. Ramp observations are reported separately as a diagnostic and are never
included in either low- or high-regime coverage. This avoids ambiguous
threshold definitions.

Define:

\[
\text{width\_ratio} = \text{mean\_width\_high}\,/\,\text{mean\_width\_low}.
\]

The noise-scale ratio is exactly \(\sigma_H/\sigma_L = 5\). An ideal centered
Gaussian 90% prediction interval has width \(2 z_{0.95}\,\sigma\), so the
ideal width ratio of such intervals is also 5. The value 5 is therefore the
**oracle/ideal width-ratio benchmark** for this DGP — a diagnostic reference
point, NOT a requirement that any conformal method must achieve.

## 18. Secondary metrics

- Coverage error: \(|\text{coverage} - 0.90|\)
- Coverage imbalance: \(|\text{coverage}_{\text{high}} - \text{coverage}_{\text{low}}|\)
- Width-ratio error (deviation from the oracle/ideal benchmark):
  \(|\text{width\_ratio} - 5|\)
- Winkler score: the standard two-sided interval score for nominal 90%
  coverage,

\[
W_\alpha(L, U; y)
= (U - L)
+ \frac{2}{\alpha}(L - y)\,\mathbf{1}(y < L)
+ \frac{2}{\alpha}(y - U)\,\mathbf{1}(y > U),
\qquad \alpha = 0.10 .
\]

For adaptive methods, calculate the diagnostic association between the
estimated local scale and the true \(\sigma_t\) as the **Spearman
correlation** between \(s_t\) and \(\sigma_t\) on the test segment (frozen in
v1.1). Spearman is used because the question of interest is whether the
estimator correctly orders low-to-high noise levels, not whether the
relationship is linear. Clearly label scale association as a diagnostic. Do
NOT treat correlation(width, scale) as independent evidence of adaptivity
because adaptive interval width explicitly contains the scale term.

## 19. Statistical summary

Across replications report mean, standard deviation, and a 95% confidence
interval for each metric. Confidence intervals use the t-interval
\(\bar{m} \pm t_{0.975,29}\,\operatorname{sd}/\sqrt{30}\) (frozen in v1.1).

Because every method is evaluated on the same replication, use paired
differences. The primary paired comparison is **Replacement − Additive** for:

- coverage
- width
- width ratio
- coverage imbalance
- Winkler score

For each metric, form the per-replication paired differences, e.g.

\[
\Delta_r = R_{W,r}^{\text{replacement}} - R_{W,r}^{\text{additive}},
\qquad r = 0, \ldots, 29,
\]

and report the mean paired difference with the paired t-confidence interval
(29 degrees of freedom):

\[
\bar{\Delta} \pm t_{0.975,29}\,\operatorname{sd}(\Delta)\,/\,\sqrt{30}.
\]

Do not rely only on p-values.

## 20. Temporal calibration validity

The data are time series. Therefore ordinary iid split-conformal
finite-sample validity cannot automatically be claimed.

The primary experiment describes the chronological calibration protocol as an
empirical time-series evaluation. No primary method may be altered after
observing test results.

A block-based/time-series-aware calibration procedure is explicitly **out of
scope** for the Experiment 04 notebook (frozen in v1.1). After the primary
Experiment 04 results are frozen, it will be studied in a separate follow-up,
**Experiment 05: Temporal Calibration Robustness**. This keeps Experiment 04
a single clean protocol and separates interval geometry from temporal
calibration validity.

## 21. Leakage audit

The final notebook must explicitly verify:

- no test labels enter calibration;
- no future observations enter local-scale calculation;
- no true sigma enters non-oracle methods;
- no test metric is used for hyperparameter tuning;
- all methods use identical base predictions within a replication;
- regime labels are evaluation-only;
- calibration scores come only from D3.

Add assertions where possible. Under the frozen boundary-crossing
construction, window causality (every lag and scale input strictly precedes
its target index) must also be asserted.

## 22. Reproducibility

Record:

- Python version
- PyTorch version
- NumPy version
- operating environment
- device
- random seeds (including the per-replication derivation
  `replication_seed = 1000 + replication_id`)
- model configuration (as frozen in Section 9)
- DGP configuration (as frozen in Section 4)
- Git commit hash if available

Use deterministic PyTorch settings where practical, mirroring the Experiment
03 determinism settings listed in Section 9. Do not use unsafe OpenMP
workarounds such as `KMP_DUPLICATE_LIB_OK=TRUE`.

## 23. Output files

The eventual notebook should create:

    experiments/
        04_adaptive_cqr_controlled_comparison.ipynb

and:

    results/
        experiment_04/
            replication_results.csv
            summary_results.csv
            paired_comparisons.csv
            figures/

These files are not created at the design-document stage.

## 24. Figures to plan

The eventual experiment should produce at least:

1. **Figure 1:** Example test-series interval visualization for one
   replication.
2. **Figure 2:** Mean width by regime for all methods.
3. **Figure 3:** Coverage by regime for all methods.
4. **Figure 4:** Distribution of width ratios across Monte Carlo replications.
5. **Figure 5:** Paired difference in width ratio (replacement − additive).
6. **Figure 6:** Example local scale versus true sigma.

No figures are created at the design-document stage.

## 25. Interpretation rules

- Do not claim "replacement CQR is better" before results exist.
- Do not optimize hyperparameters separately to maximize width ratio.
- Do not use the oracle/ideal width-ratio benchmark of 5 as a tuning
  objective.
- Do not use test results to select the preferred method.
- If replacement CQR improves width adaptation but hurts coverage, report both
  facts.
- If additive CQR performs equally well, report that.
- If the hypothesis fails, the experiment is still scientifically useful.

## 26. Relation to Experiment 03

Experiment 03 remains the verified historical prototype. Experiment 04 is NOT
a continuation that overwrites Experiment 03.

- **Experiment 03:** 1,000-point prototype; local-scale adaptation; additive
  versus replacement formulation introduced.
- **Experiment 04:** controlled cyclic heteroskedastic DGP; repeated Monte
  Carlo evaluation; fixed base model; fixed local scale; direct additive
  versus replacement comparison.

Do not modify Experiment 03.

One deliberate methodological difference from Experiment 03 is documented in
Section 7: Experiment 03 constructed sequences independently inside each
split, whereas Experiment 04 assigns only target observations to segments and
lets causal lag/scale windows cross segment boundaries, preserving
\(n_{\text{cal}} = 1152\) and \(k = 1038\) exactly.

## 27. Design status

This is revision v1.1 of the preregistered-style design specification. The
eight implementation ambiguities from the design review were resolved and
frozen in this revision, before implementation. Numerical results are
intentionally absent at this stage. The implementation must follow this
protocol without method-specific tuning based on test performance.
