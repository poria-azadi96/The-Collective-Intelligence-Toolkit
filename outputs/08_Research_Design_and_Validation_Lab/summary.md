# Module 08: Computational Research Design & Validation Lab

**Source Notebook:** `notebooks/08_research_design_lab.ipynb`

Track A multiple-testing Benjamini-Hochberg FDR & statistical power calibration, Track B mechanistic ABM vs. randomized null baseline with Cohen's d effect size, and the complete 8-Question Scientific Paper Audit Database across 6 foundational papers.

## Generated Classroom Figures & Outputs

| # | Subfolder / File | Scenario & Pedagogical Focus | Parameters | Console Output / Metrics |
| :--- | :--- | :--- | :--- | :--- |
| 01 | [`01_Track_A_Observational_FDR_and_Power/01_TrackA_LowSNR0.55_T100.png`](./01_Track_A_Observational_FDR_and_Power/01_TrackA_LowSNR0.55_T100.png) | Track A Notebook Default (SNR=0.55, T=100): uncorrected false positives while BH-FDR power=0% (underpowered) | `paper_choice=1. Schreiber (2000) - Transfer Entropy`, `track=Track A: Observational FDR & Power`, `sample_size=100` | 8-QUESTION PAPER AUDIT PROTOCOL | 1. Schreiber (2000) - Transfer Entropy <br> 1. Problem: Symmetric mutual information cannot distinguish directional information flow from shared static correlation. <br> 2. Hypothesis: Conditioning on target past history isolates directed predictive information T_{Y->X}. <br> 3. Model: Conditional mutual information I(Y_past; X_future | X_past) estimated via kernel or k-NN estimators. <br> 4. Assumptions: Local stationarity, Markov order k, and absence of unobserved common drivers. <br> 5. Key Result: Asymmetric coupling detection in coupled chaotic maps and physiological bivariate series. <br> 6. Reproducibility Target: Elimination of spurious links in Common Driver & Cascade motifs via Conditional TE. <br> 7. Limitations: Bivariate TE fails under confounders; high-dimensional conditioning suffers curse of dimensionality. <br> 8. Open Question: How does iterative greedy conditioning scale under non-stationary regime shifts? |
| 02 | [`01_Track_A_Observational_FDR_and_Power/02_TrackA_LowSNR0.55_T300.png`](./01_Track_A_Observational_FDR_and_Power/02_TrackA_LowSNR0.55_T300.png) | Track A Notebook Default (SNR=0.55, T=300): uncorrected TPR=25%, FPR=2.4%; BH-FDR stays conservative (power=0%) | `paper_choice=1. Schreiber (2000) - Transfer Entropy`, `track=Track A: Observational FDR & Power`, `sample_size=300` | 8-QUESTION PAPER AUDIT PROTOCOL | 1. Schreiber (2000) - Transfer Entropy <br> 1. Problem: Symmetric mutual information cannot distinguish directional information flow from shared static correlation. <br> 2. Hypothesis: Conditioning on target past history isolates directed predictive information T_{Y->X}. <br> 3. Model: Conditional mutual information I(Y_past; X_future | X_past) estimated via kernel or k-NN estimators. <br> 4. Assumptions: Local stationarity, Markov order k, and absence of unobserved common drivers. <br> 5. Key Result: Asymmetric coupling detection in coupled chaotic maps and physiological bivariate series. <br> 6. Reproducibility Target: Elimination of spurious links in Common Driver & Cascade motifs via Conditional TE. <br> 7. Limitations: Bivariate TE fails under confounders; high-dimensional conditioning suffers curse of dimensionality. <br> 8. Open Question: How does iterative greedy conditioning scale under non-stationary regime shifts? |
| 03 | [`01_Track_A_Observational_FDR_and_Power/03_TrackA_LowSNR0.55_T800.png`](./01_Track_A_Observational_FDR_and_Power/03_TrackA_LowSNR0.55_T800.png) | Track A Notebook Default (SNR=0.55, T=800): weak SNR remains underpowered under 50-test BH-FDR up to T=1200 | `paper_choice=1. Schreiber (2000) - Transfer Entropy`, `track=Track A: Observational FDR & Power`, `sample_size=800` | 8-QUESTION PAPER AUDIT PROTOCOL | 1. Schreiber (2000) - Transfer Entropy <br> 1. Problem: Symmetric mutual information cannot distinguish directional information flow from shared static correlation. <br> 2. Hypothesis: Conditioning on target past history isolates directed predictive information T_{Y->X}. <br> 3. Model: Conditional mutual information I(Y_past; X_future | X_past) estimated via kernel or k-NN estimators. <br> 4. Assumptions: Local stationarity, Markov order k, and absence of unobserved common drivers. <br> 5. Key Result: Asymmetric coupling detection in coupled chaotic maps and physiological bivariate series. <br> 6. Reproducibility Target: Elimination of spurious links in Common Driver & Cascade motifs via Conditional TE. <br> 7. Limitations: Bivariate TE fails under confounders; high-dimensional conditioning suffers curse of dimensionality. <br> 8. Open Question: How does iterative greedy conditioning scale under non-stationary regime shifts? |
| 04 | [`01_Track_A_Observational_FDR_and_Power/04_TrackA_Calibrated_SNR1.25_T1200.png`](./01_Track_A_Observational_FDR_and_Power/04_TrackA_Calibrated_SNR1.25_T1200.png) | Track A Calibrated SNR=1.25 (T=1200): BH-FDR achieves 87.5% power (>80% benchmark) with 0.0% false positives | `signal_snr=1.25`, `sample_size=1200`, `n_tests=50`, `true_edges=8` | [Supplementary Calibrated Run using Module 08 simulate_fdr_validation] SNR=1.25, T=1200 | Raw TPR=1.000, FPR=0.024 | BH-FDR TPR=0.875, FPR=0.000 |
| 05 | [`02_Track_B_Generative_ABM_vs_Null_Model/05_TrackB_ABM_vs_Null_Baseline.png`](./02_Track_B_Generative_ABM_vs_Null_Model/05_TrackB_ABM_vs_Null_Baseline.png) | Track B Generative ABM vs. Randomized Null Baseline & Cohen's d Falsifiability Profile | `paper_choice=5. Tero et al. (2010) - Physarum Tokyo Rail`, `track=Track B: Generative ABM vs Null Model`, `sample_size=300` | 8-QUESTION PAPER AUDIT PROTOCOL | 5. Tero et al. (2010) - Physarum Tokyo Rail <br> 1. Problem: Can a decentralized tubular organism design transport networks balancing cost, efficiency, and fault tolerance? <br> 2. Hypothesis: Positive feedback between flux Q_ij and tube conductance D_ij self-organizes near-optimal planar graphs. <br> 3. Model: Kirchhoff flow network coupled with adaptive ODE dD_ij/dt = |Q_ij|^gamma / (1 + |Q_ij|^gamma) - D_ij. <br> 4. Assumptions: Quasi-steady laminar flow and sequential random source-sink demand pairs. <br> 5. Key Result: Reconstructed networks match Tokyo rail efficiency and fault tolerance at comparable total length. <br> 6. Reproducibility Target: Multi-objective comparison (Cost, Global Efficiency, Fault Tolerance) against MST baseline. <br> 7. Limitations: Dynamic edge-cutting self-healing is an experimental model extension beyond static topology comparison. <br> 8. Open Question: How does non-uniform traffic demand weighting alter optimal feedback exponent gamma? |

## Full Console Logs

### 01. Track A Notebook Default (SNR=0.55, T=100): uncorrected false positives while BH-FDR power=0% (underpowered)
- **File:** [`01_Track_A_Observational_FDR_and_Power/01_TrackA_LowSNR0.55_T100.png`](./01_Track_A_Observational_FDR_and_Power/01_TrackA_LowSNR0.55_T100.png)
- **Parameters:** `{'paper_choice': '1. Schreiber (2000) - Transfer Entropy', 'track': 'Track A: Observational FDR & Power', 'sample_size': 100}`
```text
========================================================================================
8-QUESTION PAPER AUDIT PROTOCOL | 1. Schreiber (2000) - Transfer Entropy
========================================================================================
1. Problem: Symmetric mutual information cannot distinguish directional information flow from shared static correlation.
2. Hypothesis: Conditioning on target past history isolates directed predictive information T_{Y->X}.
3. Model: Conditional mutual information I(Y_past; X_future | X_past) estimated via kernel or k-NN estimators.
4. Assumptions: Local stationarity, Markov order k, and absence of unobserved common drivers.
5. Key Result: Asymmetric coupling detection in coupled chaotic maps and physiological bivariate series.
6. Reproducibility Target: Elimination of spurious links in Common Driver & Cascade motifs via Conditional TE.
7. Limitations: Bivariate TE fails under confounders; high-dimensional conditioning suffers curse of dimensionality.
8. Open Question: How does iterative greedy conditioning scale under non-stationary regime shifts?
========================================================================================
```

### 02. Track A Notebook Default (SNR=0.55, T=300): uncorrected TPR=25%, FPR=2.4%; BH-FDR stays conservative (power=0%)
- **File:** [`01_Track_A_Observational_FDR_and_Power/02_TrackA_LowSNR0.55_T300.png`](./01_Track_A_Observational_FDR_and_Power/02_TrackA_LowSNR0.55_T300.png)
- **Parameters:** `{'paper_choice': '1. Schreiber (2000) - Transfer Entropy', 'track': 'Track A: Observational FDR & Power', 'sample_size': 300}`
```text
========================================================================================
8-QUESTION PAPER AUDIT PROTOCOL | 1. Schreiber (2000) - Transfer Entropy
========================================================================================
1. Problem: Symmetric mutual information cannot distinguish directional information flow from shared static correlation.
2. Hypothesis: Conditioning on target past history isolates directed predictive information T_{Y->X}.
3. Model: Conditional mutual information I(Y_past; X_future | X_past) estimated via kernel or k-NN estimators.
4. Assumptions: Local stationarity, Markov order k, and absence of unobserved common drivers.
5. Key Result: Asymmetric coupling detection in coupled chaotic maps and physiological bivariate series.
6. Reproducibility Target: Elimination of spurious links in Common Driver & Cascade motifs via Conditional TE.
7. Limitations: Bivariate TE fails under confounders; high-dimensional conditioning suffers curse of dimensionality.
8. Open Question: How does iterative greedy conditioning scale under non-stationary regime shifts?
========================================================================================
```

### 03. Track A Notebook Default (SNR=0.55, T=800): weak SNR remains underpowered under 50-test BH-FDR up to T=1200
- **File:** [`01_Track_A_Observational_FDR_and_Power/03_TrackA_LowSNR0.55_T800.png`](./01_Track_A_Observational_FDR_and_Power/03_TrackA_LowSNR0.55_T800.png)
- **Parameters:** `{'paper_choice': '1. Schreiber (2000) - Transfer Entropy', 'track': 'Track A: Observational FDR & Power', 'sample_size': 800}`
```text
========================================================================================
8-QUESTION PAPER AUDIT PROTOCOL | 1. Schreiber (2000) - Transfer Entropy
========================================================================================
1. Problem: Symmetric mutual information cannot distinguish directional information flow from shared static correlation.
2. Hypothesis: Conditioning on target past history isolates directed predictive information T_{Y->X}.
3. Model: Conditional mutual information I(Y_past; X_future | X_past) estimated via kernel or k-NN estimators.
4. Assumptions: Local stationarity, Markov order k, and absence of unobserved common drivers.
5. Key Result: Asymmetric coupling detection in coupled chaotic maps and physiological bivariate series.
6. Reproducibility Target: Elimination of spurious links in Common Driver & Cascade motifs via Conditional TE.
7. Limitations: Bivariate TE fails under confounders; high-dimensional conditioning suffers curse of dimensionality.
8. Open Question: How does iterative greedy conditioning scale under non-stationary regime shifts?
========================================================================================
```

### 04. Track A Calibrated SNR=1.25 (T=1200): BH-FDR achieves 87.5% power (>80% benchmark) with 0.0% false positives
- **File:** [`01_Track_A_Observational_FDR_and_Power/04_TrackA_Calibrated_SNR1.25_T1200.png`](./01_Track_A_Observational_FDR_and_Power/04_TrackA_Calibrated_SNR1.25_T1200.png)
- **Parameters:** `{'signal_snr': 1.25, 'sample_size': 1200, 'n_tests': 50, 'true_edges': 8}`
```text
[Supplementary Calibrated Run using Module 08 simulate_fdr_validation] SNR=1.25, T=1200 | Raw TPR=1.000, FPR=0.024 | BH-FDR TPR=0.875, FPR=0.000
```

### 05. Track B Generative ABM vs. Randomized Null Baseline & Cohen's d Falsifiability Profile
- **File:** [`02_Track_B_Generative_ABM_vs_Null_Model/05_TrackB_ABM_vs_Null_Baseline.png`](./02_Track_B_Generative_ABM_vs_Null_Model/05_TrackB_ABM_vs_Null_Baseline.png)
- **Parameters:** `{'paper_choice': '5. Tero et al. (2010) - Physarum Tokyo Rail', 'track': 'Track B: Generative ABM vs Null Model', 'sample_size': 300}`
```text
========================================================================================
8-QUESTION PAPER AUDIT PROTOCOL | 5. Tero et al. (2010) - Physarum Tokyo Rail
========================================================================================
1. Problem: Can a decentralized tubular organism design transport networks balancing cost, efficiency, and fault tolerance?
2. Hypothesis: Positive feedback between flux Q_ij and tube conductance D_ij self-organizes near-optimal planar graphs.
3. Model: Kirchhoff flow network coupled with adaptive ODE dD_ij/dt = |Q_ij|^gamma / (1 + |Q_ij|^gamma) - D_ij.
4. Assumptions: Quasi-steady laminar flow and sequential random source-sink demand pairs.
5. Key Result: Reconstructed networks match Tokyo rail efficiency and fault tolerance at comparable total length.
6. Reproducibility Target: Multi-objective comparison (Cost, Global Efficiency, Fault Tolerance) against MST baseline.
7. Limitations: Dynamic edge-cutting self-healing is an experimental model extension beyond static topology comparison.
8. Open Question: How does non-uniform traffic demand weighting alter optimal feedback exponent gamma?
========================================================================================
```
