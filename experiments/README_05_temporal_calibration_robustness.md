# Experiment 05: Temporal Calibration Robustness

## Revision history

- **v1.0 (2026-09-07):** Initial frozen, preregistered-style design specification,
  written before any E05 implementation and before observing any E05 result.
  Experiment 04 (`experiments/README_04_controlled_comparison.md`, v1.1) and its
  executed notebook/results are the methodological baseline; E05 changes exactly one
  methodological factor: the conformal calibration's treatment of temporal dependence.
- **v1.1 (2026-09-07, pre-implementation confirmation):** The six ambiguities flagged
  in the design audit were resolved by the owner exactly as specified in this
  document, with no design change: (1) C1 applies only to M2 and M3; M0 is
  calibration-free and M1 plus the oracle diagnostics remain C0-only E04
  baselines/diagnostics; (2) the E05 M3 display name is "Local-Variability Scaled
  Center-Based Replacement CQR"; (3) paired contrasts use the six primary metrics
  plus the Winkler score; (4) the notebook path is
  `experiments/05_temporal_calibration_robustness.ipynb`; (5) coverage imbalance is
  reported once, as a primary metric; (6) E04's median-width metric is excluded from
  the frozen E05 summaries.

## Frozen protocol summary

| Quantity | Frozen value |
|---|---|
| Cycle length \(P\) | 96 = 40 low + 8 raised-cosine up + 40 high + 8 raised-cosine down |
| Warm-up | 128 observations discarded as targets; phase not reset |
| D1 | 5,952 targets (54 training + 8 validation cycles) |
| D2 | 2,400 targets (pre-specified diagnostics only; no learned-scale arm) |
| D3 | 1,152 targets (calibration; 12 complete chronological blocks of 96) |
| Test | 2,400 targets (final evaluation) |
| Sequence length \(L\) | 32 |
| \(\alpha\) | 0.10 (nominal coverage 0.90) |
| Pointwise conformal rank \(k\) | \(\lceil 1153 \times 0.90 \rceil = 1038\) (identical to E04) |
| Block length \(B\) | 96 (fixed; one complete heteroskedastic cycle) |
| Number of calibration blocks \(n_{\text{block}}\) | \(1152 / 96 = 12\) |
| Block conformal rank \(k_{\text{block}}\) | \(\lceil 13 \times 0.90 \rceil = 12\) |
| Replications \(R\) | 30, ids \(0,\ldots,29\), seeds \(1000 + \text{replication\_id}\) |
| Architecture / training | Exactly the E04 frozen configuration (Section 4) |
| Local scale | \(s_t = \operatorname{std}(y_{t-32},\ldots,y_{t-1};\ \mathrm{ddof}=0)\), causal |
| Experimental design | \(2 \times 2\): {M2, M3} \(\times\) {C0 pointwise, C1 block-aware} |
| Primary contrasts | M3 − M2 under C0 and under C1; C1 − C0 for M2 and for M3 |
| Outputs | `results/experiment_05/` (fully separate from `results/experiment_04/`) |

## 1. Research question

Test whether the replacement-versus-additive behavior observed in Experiment 04
remains when conformal calibration explicitly accounts for temporal dependence.

The primary comparison is:

- **M2** = Local-Variability Scaled Additive CQR
- **M3** = Local-Variability Scaled Center-Based Replacement CQR (the same method
  frozen as "Local-Variability Replacement CQR" in E04)

compared under:

- **C0** = ordinary pointwise calibration, exactly as in E04;
- **C1** = block-aware chronological calibration.

The scientific comparison must remain controlled: same DGP, same features, same
Quantile LSTM, same local scale estimator, same calibration set, same \(\alpha\),
same test observations, same replication seeds. **Only the calibration dependence
treatment changes.**

## 2. Relation to Experiment 04

Experiment 04 established the baseline empirical comparison between pointwise
calibration and replacement/additive interval geometry under the cyclic
heteroskedastic DGP. Its frozen specification is
`experiments/README_04_controlled_comparison.md` (v1.1) and its executed results live
in `results/experiment_04/`.

E05 asks whether that relationship is robust when calibration explicitly incorporates
temporal dependence through fixed 96-point chronological blocks. The only intended
methodological change from E04 is the calibration procedure. **Do not modify E04
retrospectively** — its notebook, README, and results remain untouched.

## 3. What must remain identical to E04

The following are frozen from E04 and must not change:

\[
P = 96,\quad \text{warm-up} = 128,\quad L = 32,\quad \alpha = 0.10,\quad
n_{\text{cal}} = 1152,\quad R = 30,
\]
\[
\text{replication ids} = 0,\ldots,29,\quad \text{seeds} = 1000 + \text{replication\_id}.
\]

Use the exact E04 DGP, including per 96-point cycle: 40 low plateau observations
(\(\sigma = 0.2\)), 8 raised-cosine transition observations (up to \(\sigma = 1.0\)),
40 high plateau observations (\(\sigma = 1.0\)), and 8 raised-cosine transition
observations (back to \(\sigma = 0.2\)), with the same global phase definition
\(\phi_t = (t \bmod 96)/96\) and the same chronological D1/D2/D3/Test construction.
Regime labels remain mutually exclusive (low plateau, high plateau, both transition
slots), evaluation-only, with headline regime metrics on low and high only and ramp
reported separately as a diagnostic.

Lag windows and local-scale windows may cross chronological segment boundaries,
provided they use only observations available strictly before the target. Warm-up
observations may provide causal context but are never evaluation targets.

## 4. Base quantile model

Use the canonical Experiment 03 / E04 configuration exactly:

```python
nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
```

- the 32 scalar lag observations \(y_{t-32},\ldots,y_{t-1}\) are fed into the LSTM;
- the final hidden state \(h_t\) is concatenated with the two target-time phase
  features \(\sin(2\pi\phi_t), \cos(2\pi\phi_t)\);
- the output head is `nn.Linear(34, 2)`;
- quantiles \(\tau = 0.10\) and \(\tau = 0.90\);
- pinball loss averaged over the two quantiles;
- training: Adam, learning rate \(10^{-3}\), 40 full-batch epochs on D1's 54 training
  cycles (D1's 8 validation cycles monitored as diagnostic only), no input
  normalization, deterministic CPU execution mirroring E04's determinism settings.

Do not introduce hyperparameter tuning, dropout, alternative architectures, or
normalization.

## 5. Local scale

Use exactly the E04 causal local scale estimator:

\[
s_t = \operatorname{std}(y_{t-32}, \ldots, y_{t-1};\ \mathrm{ddof}=0).
\]

The scale estimator must never use the current target \(y_t\) or future observations.
Do not introduce a learned-scale arm in E05.

## 6. Interval methods

Retain the E04 interval methods:

- **M0** = raw/unadjusted baseline \([Q_L, Q_U]\) (no calibration; identical under
  C0 and C1 by construction);
- **M1** = standard CQR;
- **M2** = local-variability scaled additive CQR;
- **M3** = local-variability scaled center-based replacement CQR.

The primary scientific comparison is **M2 vs M3**. Oracle additive and oracle
replacement diagnostics are retained exactly as in E04 (pointwise-calibrated, true
\(\sigma\) as scale), but must remain secondary diagnostics and must not enter the
primary inferential comparison.

**Scope of C1 (frozen in v1.1):** C1 applies only to M2 and M3. M0 is
calibration-free and is identical under C0 and C1 by construction. M1 and the oracle
diagnostics are evaluated under C0 pointwise calibration only, as E04 baselines and
diagnostics; no block-aware variants are computed for them.

## 7. Calibration C0: pointwise baseline

Implement the E04 pointwise split-conformal calibration exactly. For each calibration
observation:

\[
r_i = \max\{Q_{L,i} - y_i,\ \ y_i - Q_{U,i},\ \ 0\}.
\]

For M2:

\[
S_i = r_i / s_i,
\qquad
I_i = [\,Q_{L,i} - q\,s_i,\ \ Q_{U,i} + q\,s_i\,].
\]

For M3:

\[
c_i = (Q_{L,i} + Q_{U,i})/2,
\qquad
S_i = |y_i - c_i| / s_i,
\qquad
I_i = [\,c_i - q\,s_i,\ \ c_i + q\,s_i\,].
\]

Use the same finite-sample conformal rank convention as E04:

\[
k = \lceil (n_{\text{cal}} + 1)(1-\alpha) \rceil = \lceil 1153 \times 0.90 \rceil = 1038
\]

for \(n_{\text{cal}} = 1152\) and \(\alpha = 0.10\). Verify programmatically.

## 8. Calibration C1: block-aware

Use a fixed non-overlapping chronological block length:

\[
B = 96,
\]

which equals exactly one complete heteroskedastic cycle. Since
\(1152 / 96 = 12\), the calibration set contains exactly 12 complete blocks; block
\(b\) (\(b = 0,\ldots,11\)) is the consecutive run of 96 calibration targets
\([8480 + 96b,\ 8480 + 96(b+1))\) in global generated-series indices, so every block
spans exactly one full cycle and contains exactly the cycle's regime composition
(40 low, 16 ramp, 40 high targets).

Do NOT tune \(B\) after observing results. Do NOT test multiple block lengths in the
primary E05 experiment.

For each calibration block \(b\), first calculate the individual normalized conformity
scores using the same score definition as the corresponding method (M2 scaled scores
or M3 replacement scores). Then collapse the 96 scores within the block to one
conservative block score:

\[
S_b^{\text{block}} = \max_{i \in \text{block } b} S_i .
\]

Perform conformal quantile calibration over these 12 block scores rather than over
the 1152 individual scores, separately for M2 (giving \(q_{A,\text{block}}\)) and M3
(giving \(q_{R,\text{block}}\)).

This must be explicitly documented as a **deliberately conservative block-level
construction**. The purpose is to test robustness to temporal dependence, not to
claim that this particular block construction is universally optimal.

## 9. Block quantile / finite-sample convention

There are only \(n_{\text{block}} = 12\) block scores. Use the same finite-sample
rank convention:

\[
k_{\text{block}} = \lceil (n_{\text{block}} + 1)(1-\alpha) \rceil
                 = \lceil 13 \times 0.90 \rceil = \lceil 11.7 \rceil = 12 .
\]

Therefore the block calibration threshold is the maximum of the 12 block scores.
Freeze this convention explicitly.

An exact mathematical consequence must be documented and asserted in the
implementation: because the 12 non-overlapping blocks jointly cover all 1,152
calibration observations and \(k_{\text{block}} = 12\), the block-aware threshold
equals the **maximum individual calibration score over all 1,152 calibration
observations**,

\[
q_{\text{block}} = \max_{b=0,\ldots,11} S_b^{\text{block}}
                 = \max_{i \in D3} S_i ,
\]

computed separately for each score definition. The block-aware calibration threshold
is therefore extremely coarse: it is the largest conformity score the calibration set
contains. **This is a design property, not an implementation error**, and it makes C1
the most conservative threshold D3 can supply.

## 10. Application of block calibration

Once \(q_{\text{block}}\) is obtained, construct test intervals using exactly the same
M2/M3 interval geometry as above. **Only \(q\) changes.** Do not change the LSTM, the
local scale, the score definition, the interval geometry, the test set, or the nominal
\(\alpha\) between C0 and C1. For each test target, the interval must use only
information available at or before that target, apart from the model predictions
already generated under the fixed forecasting protocol.

The four inferential cells are therefore:

| Cell | Geometry | Calibration |
|---|---|---|
| C0-M2 | M2 scaled additive | pointwise, \(k = 1038\) |
| C0-M3 | M3 replacement | pointwise, \(k = 1038\) |
| C1-M2 | M2 scaled additive | block-aware, \(k_{\text{block}} = 12\) |
| C1-M3 | M3 replacement | block-aware, \(k_{\text{block}} = 12\) |

## 11. Role of D2

Retain D2 as a chronological diagnostic holdout exactly as in E04. D2 is NOT used to
tune: block length, model architecture, learning rate, training epochs, local scale,
or calibration strategy. No learned-scale arm is introduced. Any D2 diagnostics must
be pre-specified and must not influence the primary test results.

## 12. Hypotheses

Defined before implementation and before any E05 result exists:

- **H1.** Block-aware calibration changes empirical coverage relative to ordinary
  pointwise calibration under the temporally dependent evaluation process.
- **H2.** The replacement geometry retains an advantage in low/high regime coverage
  balance relative to additive geometry under block-aware calibration.
- **H3.** Block-aware calibration may produce wider intervals because each block
  threshold is based on the maximum conformity score.
- **H4.** Replacement geometry continues to provide stronger adaptation of interval
  width to the heteroskedastic regime than additive geometry.

These are empirical hypotheses, phrased so that the data may reject any of them. They
must not be presented as guaranteed outcomes.

## 13. Monte Carlo design

Use the same 30 paired replications and same seed ids as E04: replication id
\(r = 0,\ldots,29\), `replication_seed = 1000 + r` driving DGP, NumPy, PyTorch
initialization, and training randomness. Within a replication: one dataset, one base
Quantile LSTM, one shared set of base predictions for D2/D3/Test, and all calibration
variants evaluated on the same test observations.

## 14. Primary metrics

Retain the E04 primary metrics:

1. Overall test coverage
2. Mean interval width
3. Low-regime coverage
4. High-regime coverage
5. Low/high width ratio \(\;=\;\) mean width (high) / mean width (low)
6. Coverage imbalance, defined consistently with E04 as
   \(|\text{coverage}_{\text{high}} - \text{coverage}_{\text{low}}|\)

The primary paired comparison focuses on M3 − M2 under each calibration strategy;
the calibration contrast C1 − C0 is also evaluated for each of M2 and M3
(Section 16). The width-ratio benchmark of 5 keeps its E04 status: the oracle/ideal
reference for this DGP, never a tuning target or requirement.

## 15. Secondary metrics

Retain the relevant E04 diagnostics:

- coverage error \(|\text{coverage} - 0.90|\);
- low/high coverage difference (identical in definition to the primary coverage
  imbalance; reported once, under the primary metric);
- width-ratio error relative to the known scale ratio,
  \(|\text{width\_ratio} - 5|\);
- Winkler score (standard two-sided interval score, \(\alpha = 0.10\), E04 formula);
- ramp-regime coverage and ramp-regime width (diagnostic only);
- quantile-crossing rate of the raw quantile head;
- local-scale association with true \(\sigma\) (Spearman correlation, E04 convention);
- oracle additive/replacement diagnostics (C0, exactly as in E04).

Clearly distinguish mechanical diagnostics from inferential claims. In particular,
width-vs-scale association can be mechanically induced because the interval
construction itself contains \(s_t\); it must never be presented as independent
evidence of adaptive behaviour.

**Median width (frozen in v1.1):** E04's median-width metric is intentionally
excluded from the frozen E05 primary and secondary summaries.

## 16. Statistical analysis

For each replication calculate the four cells C0-M2, C0-M3, C1-M2, C1-M3, and the
pre-specified paired contrasts:

- M3 − M2 under C0;
- M3 − M2 under C1;
- C1 − C0 for M2;
- C1 − C0 for M3.

Each contrast is evaluated on the six primary metrics; the Winkler score is added as
a pre-specified secondary contrast (E04 precedent, and required by the
interpretation rules against selective omission). Use paired t-based 95% confidence
intervals across the 30 replications with 29 degrees of freedom
(\(t_{0.975,29} = 2.045229642132703\), the E04 frozen constant). For individual
across-replication metric summaries, retain the E04 convention of t-based 95%
intervals, \(\bar m \pm t_{0.975,29}\,\operatorname{sd}/\sqrt{30}\).

State clearly that these are across-replication uncertainty summaries, not a theorem
establishing temporal conformal validity.

## 17. Temporal validity caveat

This remains an empirical chronological time-series experiment.

- Do NOT claim iid split-conformal validity.
- Do NOT claim that taking block maxima automatically establishes formal coverage
  guarantees for the dependent process.
- The block-aware method is an empirical robustness construction designed to make
  temporal dependence explicit.
- Formal theoretical validity under dependent data is outside the scope of E05.

## 18. Leakage audit

The implementation must require explicit assertions verifying:

- no future observations enter lag windows;
- no future observations enter local-scale windows;
- calibration targets are strictly before test targets;
- block membership follows chronological order;
- calibration blocks do not contain test observations;
- block scores use only their own calibration observations;
- test labels are never used to calculate \(q\);
- no test-driven hyperparameter selection;
- warm-up is never counted as an evaluation target.

## 19. Single-replication validation

Before running all 30 replications, the eventual notebook must perform a
single-replication validation pass with assertions for:

- total generated length (12,032);
- segment boundaries ([128, 5312, 6080, 8480, 9632, 12032));
- phase alignment (every segment starts at cycle slot 32);
- 40/8/40/8 noise geometry;
- exact regime counts;
- causal lag windows;
- causal local-scale windows;
- \(n_{\text{cal}} = 1152\);
- 12 complete calibration blocks;
- block length = 96;
- \(k_{\text{pointwise}} = 1038\);
- \(k_{\text{block}} = 12\);
- no leakage (Section 18 checklist);
- correct pointwise score identities;
- correct block-max score identities (including the Section 9 identity
  \(q_{\text{block}} = \max_{i \in D3} S_i\));
- correct interval identities for all four cells;
- deterministic retraining.

Only after this validation passes may the full 30-replication experiment be run.

## 20. Reproducibility

Freeze replication ids \(0,\ldots,29\) with seed \(1000 + \text{replication\_id}\).
Record: Python version, PyTorch version, NumPy version, CPU/deterministic settings,
all frozen experiment constants, method definitions, calibration definitions, and
random seeds — stored in the notebook metadata and in `results/experiment_05/config.json`.

Do not use `KMP_DUPLICATE_LIB_OK`. If the MKL/PyTorch threading issue encountered in
E04 occurs, use the documented E04 environment fix `MKL_THREADING_LAYER=SEQUENTIAL`
(set before NumPy loads and recorded in the notebook metadata/configuration); it
changes no numerical result.

## 21. Output files

The eventual implementation should produce a dedicated directory:

    results/
        experiment_05/
            replication_results.csv
            summary_results.csv
            paired_comparisons.csv
            config.json
            figures/

and the notebook `experiments/05_temporal_calibration_robustness.ipynb`. Do not
overwrite Experiment 04 outputs; E05 outputs must remain completely separate from
`results/experiment_04/`.

## 22. Figures

Plan figures that directly answer the research question; each must clearly
distinguish C0 pointwise from C1 block-aware calibration:

1. Overall coverage by method and calibration strategy.
2. Low/high regime coverage comparison.
3. Width ratio comparison.
4. Coverage imbalance comparison.
5. Mean width comparison.
6. Calibration-strategy paired differences (C1 − C0).

Do not add figures merely because they look interesting after seeing results.

## 23. Interpretation rules

Pre-specify that:

- marginal coverage alone does not determine superiority;
- replacement should be evaluated primarily on regime-level adaptation and coverage
  balance;
- increased width must be reported as a trade-off;
- the Winkler score must be reported rather than selectively omitted;
- block calibration's coarse 12-block quantile must be treated as an important
  limitation;
- oracle results are diagnostics, not deployable methods;
- no post-hoc method modification is allowed based on observed results.

The experiment must explicitly distinguish the effect of (1) calibration strategy,
(2) interval geometry, and (3) local-scale estimation. Do not attribute a result to
replacement geometry if it is actually caused by the calibration strategy.

## 24. Design status

This is revision v1.1. Experiment 05 design is frozen before implementation and
before observing E05 results; the six design-audit ambiguities were resolved and
frozen in this revision, before implementation. No notebook has been created as part
of this task, no results have been generated, and no files outside this README have
been modified. The implementation must follow this protocol without method-specific
tuning based on test performance.
