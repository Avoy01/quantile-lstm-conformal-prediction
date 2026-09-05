# Experiment 2: Quantile LSTM and CQR under Heteroskedastic Noise

## Overview

This experiment investigates the behavior of neural quantile regression and
Conformalized Quantile Regression (CQR) when the underlying time series
exhibits heteroskedastic noise.

The experiment is designed as a controlled extension of the synthetic
baseline in Experiment 1. While Experiment 1 considers a homoskedastic
setting, where the noise level remains constant, this experiment introduces
a localized high-noise regime. This allows us to examine whether a standard
Quantile LSTM learns prediction intervals whose widths reflect changes in
the underlying uncertainty.

The experiment has two primary purposes:

1. To establish a heteroskedastic baseline for neural quantile prediction.
2. To determine whether the resulting prediction intervals exhibit
   uncertainty-adaptive behavior before introducing an adaptive conformal
   method.

The results provide the experimental motivation for the subsequent
development and evaluation of Adaptive CQR.

---

## 1. Experimental Setting

The dataset consists of 1,000 observations generated from a deterministic
periodic signal with time-varying Gaussian noise.

The signal is defined as

\[
f(t)=\sin\left(\frac{2\pi t}{50}\right),
\]

and the observed response is

\[
Y_t=f(t)+\sigma_t\epsilon_t,
\]

where

\[
\epsilon_t\sim\mathcal{N}(0,1).
\]

The noise standard deviation is piecewise constant:

\[
\sigma_t =
\begin{cases}
1.0, & 300\leq t\leq599,\\
0.2, & \text{otherwise}.
\end{cases}
\]

Consequently, the experiment contains two distinct uncertainty regimes.

The low-noise regime has

\[
\sigma_{\mathrm{low}}=0.2,
\]

while the high-noise regime has

\[
\sigma_{\mathrm{high}}=1.0.
\]

Thus, the standard deviation of the observation noise changes by a factor of

\[
\frac{1.0}{0.2}=5.
\]

The variance changes by a factor of

\[
\frac{1.0^2}{0.2^2}=25.
\]

This substantial difference provides a controlled setting for testing whether
prediction intervals respond to local changes in uncertainty.

---

## 2. Data Regimes

The high-noise region occupies observations

\[
t=300,\ldots,599,
\]

giving 300 high-noise observations.

The remaining 700 observations belong to the low-noise regime:

\[
t=0,\ldots,299
\]

and

\[
t=600,\ldots,999.
\]

The resulting data-generating structure is therefore:

| Region | Index Range | Noise SD | Number of Observations |
|---|---|---:|---:|
| Low noise | 0–299 | 0.2 | 300 |
| High noise | 300–599 | 1.0 | 300 |
| Low noise | 600–999 | 0.2 | 400 |
| **Total** | **0–999** | — | **1000** |

The high-noise regime is intentionally localized in the middle of the time
series rather than distributed randomly. This preserves a clear temporal
structure that can be analyzed directly.

---

## 3. Temporal Data Splitting

The observations are divided chronologically into training, calibration,
and test sets.

| Split | Raw Observations | Time Range |
|---|---:|---|
| Training | 700 | 0–699 |
| Calibration | 150 | 700–849 |
| Test | 150 | 850–999 |

No random shuffling is performed across these splits.

This chronological design is important for time-series prediction because it
prevents observations from the future from being used to construct the
training or calibration distributions.

The high-noise regime lies entirely within the training period:

\[
300\leq t\leq599.
\]

Consequently, the training set contains both noise regimes, whereas the
calibration and test sets contain only low-noise observations.

This characteristic of the experiment is intentional but also constitutes an
important limitation, discussed later.

---

## 4. Supervised Sequence Construction

The time series is transformed into a supervised sequence prediction problem
using a window length of five.

For each prediction target \(Y_t\), the input is

\[
X_t=
(Y_{t-5},Y_{t-4},Y_{t-3},Y_{t-2},Y_{t-1}).
\]

The objective is therefore to estimate the conditional distribution of
\(Y_t\) given the previous five observations.

Because five previous observations are required for each target, the effective
number of samples decreases by five in each split:

| Split | Raw Samples | Sequence Length | Effective Samples |
|---|---:|---:|---:|
| Training | 700 | 5 | 695 |
| Calibration | 150 | 5 | 145 |
| Test | 150 | 5 | 145 |

The final tensors therefore have the following dimensions:

\[
X_{\mathrm{train}}\in\mathbb{R}^{695\times5\times1},
\]

\[
X_{\mathrm{cal}}\in\mathbb{R}^{145\times5\times1},
\]

\[
X_{\mathrm{test}}\in\mathbb{R}^{145\times5\times1}.
\]

The corresponding target vectors have dimensions

\[
Y_{\mathrm{train}}\in\mathbb{R}^{695},
\]

\[
Y_{\mathrm{cal}}\in\mathbb{R}^{145},
\]

and

\[
Y_{\mathrm{test}}\in\mathbb{R}^{145}.
\]

---

## 5. Quantile LSTM Baseline

A recurrent neural network based on an LSTM architecture is used to estimate
conditional prediction quantiles.

For each input sequence \(X_t\), the model produces two outputs:

\[
\hat Q_{0.1}(X_t)
\]

and

\[
\hat Q_{0.9}(X_t).
\]

These represent estimates of the conditional 10th and 90th percentiles of the
target distribution.

The model output can therefore be written as

\[
f_\theta(X_t)
=
\left[
\hat Q_{0.1}(X_t),
\hat Q_{0.9}(X_t)
\right].
\]

The corresponding nominal 90% raw prediction interval is

\[
I_{\mathrm{raw}}(X_t)
=
\left[
\hat Q_{0.1}(X_t),
\hat Q_{0.9}(X_t)
\right].
\]

The raw interval width is

\[
W_{\mathrm{raw}}(X_t)
=
\hat Q_{0.9}(X_t)-\hat Q_{0.1}(X_t).
\]

---

## 6. Quantile Regression Objective

The Quantile LSTM is trained using the pinball loss.

For a target quantile \(\tau\), the pinball loss is

\[
L_\tau(y,\hat q_\tau)
=
\begin{cases}
\tau(y-\hat q_\tau), & y\geq\hat q_\tau,\\
(1-\tau)(\hat q_\tau-y), & y<\hat q_\tau.
\end{cases}
\]

The model simultaneously estimates the two quantiles

\[
\tau\in\{0.1,0.9\}.
\]

The training objective is the aggregate pinball loss over the two predicted
quantiles.

The calibration set is not used during model optimization.

---

## 7. Reproducibility and Training

The experiment fixes the random seeds used by both NumPy and PyTorch:

```text
NumPy seed: 42
PyTorch seed: 42