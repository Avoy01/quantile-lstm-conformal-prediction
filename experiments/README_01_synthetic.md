# Experiment 1: Synthetic AR(1) Quantile LSTM + Conformal Prediction

## 1. Objective

This experiment establishes the baseline performance of neural quantile
regression and conformal prediction on a synthetic homoskedastic time-series
dataset.

The experiment evaluates three prediction-interval approaches:

1. Raw Quantile Regression
2. Split Conformal Prediction (SCP)
3. Conformalized Quantile Regression (CQR)

The purpose of this experiment is to establish a controlled baseline before
introducing heteroskedastic noise and adaptive conformal prediction.

---

## 2. Data-Generating Process

The synthetic time series follows an AR(1) process:

\[
Y_t = 0.7Y_{t-1} + \epsilon_t
\]

where

\[
\epsilon_t \sim \mathcal{N}(0,1).
\]

The noise variance is constant throughout the dataset. Therefore, this
experiment represents a homoskedastic setting.

---

## 3. Input Construction

A sliding window of length 5 is used.

For each prediction target:

\[
X_t =
(Y_{t-5},Y_{t-4},Y_{t-3},Y_{t-2},Y_{t-1})
\]

and the prediction target is:

\[
Y_t.
\]

Thus, the model uses the previous five observations to predict the next
observation.

---

## 4. Data Split

The original 1000 observations are split chronologically:

| Split | Observations |
|---|---:|
| Training | 700 |
| Calibration | 150 |
| Test | 150 |

The sequence length is 5. After sequence construction, the effective sample
sizes become:

| Split | Sequence Samples |
|---|---:|
| Training | 695 |
| Calibration | 145 |
| Test | 145 |

The chronological structure is preserved throughout the experiment to avoid
temporal data leakage.

The training set is used to fit the Quantile LSTM. The calibration set is
used only for conformal calibration, while the test set is reserved for final
evaluation.

---

## 5. Quantile LSTM

A two-output LSTM is trained to estimate two conditional quantiles:

\[
\hat Q_{0.1}(X_t)
\]

and

\[
\hat Q_{0.9}(X_t).
\]

The model output is therefore:

\[
f_\theta(X_t)
=
\left[
\hat Q_{0.1}(X_t),
\hat Q_{0.9}(X_t)
\right].
\]

The corresponding raw prediction interval is:

\[
I_{\mathrm{raw}}(X_t)
=
[
\hat Q_{0.1}(X_t),
\hat Q_{0.9}(X_t)
].
\]

The nominal coverage level is:

\[
1-\alpha=0.90,
\]

with:

\[
\alpha=0.10.
\]

---

## 6. Quantile Regression Training

The model is trained using the pinball loss.

For a target quantile \(\tau\), the pinball loss is defined as:

\[
L_\tau(y,\hat q_\tau)
=
\begin{cases}
\tau(y-\hat q_\tau), & y\geq\hat q_\tau,\\
(1-\tau)(\hat q_\tau-y), & y<\hat q_\tau.
\end{cases}
\]

The model is trained simultaneously for:

\[
\tau\in\{0.1,0.9\}.
\]

The overall training objective is the sum or mean of the losses associated
with the lower and upper quantiles.

The trained model therefore attempts to estimate the conditional 10th and
90th percentiles of the response distribution.

---

## 7. Raw Quantile Prediction Interval

After training, the model directly provides:

\[
\hat Q_{0.1}(X_t)
\]

and:

\[
\hat Q_{0.9}(X_t).
\]

The raw interval is:

\[
I_{\mathrm{raw}}(X_t)
=
[
\hat Q_{0.1}(X_t),
\hat Q_{0.9}(X_t)
].
\]

The width of the raw interval is:

\[
W_{\mathrm{raw}}(X_t)
=
\hat Q_{0.9}(X_t)
-
\hat Q_{0.1}(X_t).
\]

No conformal correction is applied to this interval.

---

## 8. Split Conformal Prediction

Split Conformal Prediction (SCP) uses the trained model to produce a point
prediction.

The midpoint of the two predicted quantiles is used:

\[
\hat y_i
=
\frac{
\hat Q_{0.1}(X_i)
+
\hat Q_{0.9}(X_i)
}{2}.
\]

For each calibration sample, the absolute residual is calculated as:

\[
R_i
=
|Y_i-\hat y_i|.
\]

The calibration set contains:

\[
n_{\mathrm{cal}}=145
\]

samples.

For nominal coverage:

\[
1-\alpha=0.90,
\]

the finite-sample conformal order statistic is:

\[
k
=
\left\lceil
(n_{\mathrm{cal}}+1)(1-\alpha)
\right\rceil.
\]

Therefore:

\[
k
=
\left\lceil
(145+1)(0.90)
\right\rceil
=
\left\lceil
131.4
\right\rceil
=
132.
\]

The conformal residual quantile is denoted by:

\[
q_{\mathrm{SCP}}.
\]

The final SCP prediction interval is:

\[
I_{\mathrm{SCP}}(X)
=
[
\hat y-q_{\mathrm{SCP}},
\hat y+q_{\mathrm{SCP}}
].
\]

---

## 9. Conformalized Quantile Regression

CQR directly calibrates the predicted lower and upper quantiles.

For each calibration observation \(i\), the nonconformity score is:

\[
S_i
=
\max
\left\{
\hat Q_{0.1}(X_i)-Y_i,
Y_i-\hat Q_{0.9}(X_i),
0
\right\}.
\]

This score measures how far the true observation lies outside the predicted
quantile interval.

If:

\[
Y_i\in
[
\hat Q_{0.1}(X_i),
\hat Q_{0.9}(X_i)
],
\]

then:

\[
S_i=0.
\]

Otherwise, the score represents the amount by which the interval must be
expanded to contain the observation.

Using:

\[
n_{\mathrm{cal}}=145
\]

and:

\[
\alpha=0.10,
\]

the conformal order statistic is:

\[
k=132.
\]

The resulting conformal quantile is denoted by:

\[
q_{\mathrm{CQR}}.
\]

The final CQR interval is:

\[
I_{\mathrm{CQR}}(X)
=
[
\hat Q_{0.1}(X)-q_{\mathrm{CQR}},
\hat Q_{0.9}(X)+q_{\mathrm{CQR}}
].
\]

---

## 10. Evaluation Metrics

Two primary metrics are used to evaluate prediction intervals.

### 10.1 Empirical Coverage

For a prediction interval:

\[
I_i=[L_i,U_i],
\]

the empirical coverage is:

\[
\widehat{\mathrm{Coverage}}
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf{1}
\left\{
L_i\leq Y_i\leq U_i
\right\}.
\]

The desired nominal coverage is:

\[
90\%.
\]

---

### 10.2 Average Interval Width

For each test observation:

\[
W_i=U_i-L_i.
\]

The average interval width is:

\[
\overline W
=
\frac{1}{n}
\sum_{i=1}^{n}W_i.
\]

A useful prediction interval should ideally provide high coverage while
maintaining a small interval width.

---

## 11. Experimental Results

The final experiment produced the following results:

| Method | Coverage | Average Width |
|---|---:|---:|
| Raw Quantile | 78.62% | 2.6039 |
| SCP | 88.97% | 3.2863 |
| CQR | 88.97% | 3.2660 |

The nominal coverage level is:

\[
90\%.
\]

---

## 12. Interpretation of Results

The raw Quantile LSTM achieves:

\[
78.62\%
\]

empirical coverage, which is substantially below the desired 90% level.

This demonstrates that direct neural quantile regression does not necessarily
provide finite-sample calibrated prediction intervals.

SCP increases the empirical coverage to approximately:

\[
88.97\%.
\]

However, SCP produces relatively wide intervals:

\[
\overline W_{\mathrm{SCP}}
=
3.2863.
\]

CQR also achieves approximately:

\[
88.97\%
\]

coverage, while producing slightly narrower intervals:

\[
\overline W_{\mathrm{CQR}}
=
3.2660.
\]

Therefore, in this experiment, CQR provides a better width-coverage trade-off
than SCP while maintaining coverage close to the nominal level.

---

## 13. Comparison of the Three Methods

The progression can be summarized as:

\[
\text{Raw Quantile}
\rightarrow
\text{SCP}
\rightarrow
\text{CQR}.
\]

The raw method relies entirely on the learned conditional quantiles.

SCP calibrates a point-prediction-based interval using absolute residuals.

CQR instead calibrates the entire predicted quantile interval using
interval-based nonconformity scores.

The results show the benefit of conformal calibration for improving empirical
coverage.

---

## 14. Research Significance

This experiment establishes the first baseline for the project.

It verifies the complete experimental pipeline:

\[
\text{Synthetic Data}
\rightarrow
\text{Sequence Construction}
\rightarrow
\text{Quantile LSTM}
\rightarrow
\text{Pinball Loss}
\rightarrow
\text{Calibration}
\rightarrow
\text{Prediction Intervals}
\rightarrow
\text{Coverage/Width Evaluation}.
\]

The experiment also provides a controlled environment in which the behavior
of SCP and CQR can be studied before introducing heteroskedastic noise.

---

## 15. Reproducibility

The experiment uses fixed random seeds:

```text
NumPy seed = 42
PyTorch seed = 42