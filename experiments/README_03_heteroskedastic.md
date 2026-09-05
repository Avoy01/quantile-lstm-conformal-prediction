# Experiment 03: Clean Local-Scale CQR Prototype

## Status

`03_clean_local_scale_cqr.ipynb` is the canonical clean implementation for
Experiment 03. The earlier `03_adaptive_cqr_heteroskedastic.ipynb` and
`03_adaptive_cqr_relative_scale.ipynb` are retained as historical prototype
artifacts; they are not the canonical method description.

## Design

The notebook reconstructs the historical prototype exactly:

- \(N=1000\), with \(\mu_t=\sin(2\pi t/50)\);
- Gaussian noise with hard alternating 100-observation blocks of
  \(\sigma_t=0.2\) and \(\sigma_t=1.0\);
- chronological raw splits: 600 training, 150 scale, 100 calibration, and
  150 test observations;
- independently constructed five-lag sequences, with effective sizes 595,
  145, 95, and 145 respectively;
- a one-layer, 32-hidden-unit Quantile LSTM, trained for 40 full-batch epochs
  to predict the 0.10 and 0.90 quantiles.

The true noise scale and regime labels are used only for evaluation and
explicitly labelled oracle diagnostics. They are not inputs to any non-oracle
adaptive interval.

## Conformal methods

For calibration target \(y_i\) and base quantiles \(Q_L(x_i),Q_U(x_i)\),
the standard CQR score is

\[
r_i=\max\{Q_L(x_i)-y_i,\;y_i-Q_U(x_i),\;0\}.
\]

All methods use \(\alpha=0.10\), 95 calibration targets, and the finite
sample rank

\[
k=\lceil(95+1)(1-0.10)\rceil=87.
\]

The notebook evaluates:

1. **Raw Quantile Interval:** \([Q_L,Q_U]\).
2. **Standard CQR:** \([Q_L-q,Q_U+q]\).
3. **Local-Variability Scaled Additive CQR:** define the positive local scale
   \(s_i=\operatorname{sd}(y_{i-5},\ldots,y_{i-1})\), using one documented
   standard-deviation convention. Median normalization from the scale segment
   is only a parameterization. The method is
   \[
   S_i=r_i/s_i,\qquad [Q_L-q s_i,Q_U+q s_i].
   \]
4. **Local-Variability Replacement CQR:** a distinct center-based geometry,
   \[
   c_i=(Q_L+Q_U)/2,\quad S_i=|y_i-c_i|/s_i,\quad[c_i-q s_i,c_i+q s_i].
   \]
5. **ORACLE Scaled-Additive CQR** and **ORACLE Replacement CQR**, which use
   known \(s_i=\sigma_i\) only as labelled diagnostics.

The local-variability scaled additive method is not replacement CQR. Scale
normalization is not a separate method because it rescales both scores and the
conformal threshold.

## Metrics and interpretation

For every method the notebook reports marginal coverage, coverage error from
90%, mean/median interval width, low/high-regime coverage, low/high-regime
mean width, and high/low width ratio. Local-scale versus true-sigma correlation
is a diagnostic only. Width-versus-scale correlation is not treated as
scientific evidence because the interval formula mechanically contains scale.

Calibration and test are chronological, dependent blocks with different regime
compositions. Ordinary iid/exchangeable split-conformal finite-sample coverage
guarantees therefore do not apply. These are empirical time-series coverage
measurements, not theorem-level validity claims.

## Historical prototype values

These are preserved prior outputs, not silently relabelled as outputs from the
clean notebook:

| Method | Coverage | Mean width | High/low width ratio |
|---|---:|---:|---:|
| Raw Quantile | 26.2069% | 0.8629 | — |
| Standard CQR | 88.2759% | 3.4914 | — |
| Local-variability scaled additive CQR | 93.1035% | 5.2007 | 2.9676 |
| Oracle scaled-additive CQR | 97.9310% | 4.9056 | 3.1971 |

The clean notebook presents an explicit comparison after a fresh run but does
not tune code to force agreement. Historical prototype results are preserved
separately. The cleaned notebook must be rerun before its numerical results
are treated as independently verified.
