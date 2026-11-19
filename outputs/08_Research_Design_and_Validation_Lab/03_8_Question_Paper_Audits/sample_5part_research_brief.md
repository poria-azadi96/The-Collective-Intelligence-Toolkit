# 5-Part Research Brief: Greedy Conditional Transfer Entropy Under Non-Stationary Regime Shifts

**Anchored Classic:** 1. Schreiber (2000) - Transfer Entropy (Module 02)

## 1. Gap Analysis (Derived from Q7 Limitation)
Bivariate TE fails under hidden confounders, and static conditional TE assumes time-invariant transition densities.

## 2. Falsifiable Hypothesis (Q8 Extension)
Windowed adaptive KSG Conditional TE combined with BH-FDR (T >= T_80) suppresses confounder false positives across regime shifts.

## 3. Minimal Model Architecture
3-tier sliding-window KSG estimator with circular time-shift surrogates and Benjamini-Hochberg step-up thresholding.

## 4. Null Baseline & Sensitivity Validation Plan
Track A Monte Carlo calibration solving T_80 for 80% power alongside Track B shuffled-phase null model comparison (Cohen's d > 1.5).

## 5. Expected Defensible Claim
Conditioned multi-lag inference eliminates >90% of indirect cascade/common-driver artifacts while controlling FDP <= 0.05.
