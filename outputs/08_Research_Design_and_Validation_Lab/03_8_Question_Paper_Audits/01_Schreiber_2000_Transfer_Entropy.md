# 1. Schreiber (2000) - Transfer Entropy

- 1. Problem: Symmetric mutual information cannot distinguish directional information flow from shared static correlation.
- 2. Hypothesis: Conditioning on target past history isolates directed predictive information T_{Y->X}.
- 3. Model: Conditional mutual information I(Y_past; X_future | X_past) estimated via kernel or k-NN estimators.
- 4. Assumptions: Local stationarity, Markov order k, and absence of unobserved common drivers.
- 5. Key Result: Asymmetric coupling detection in coupled chaotic maps and physiological bivariate series.
- 6. Reproducibility Target: Elimination of spurious links in Common Driver & Cascade motifs via Conditional TE.
- 7. Limitations: Bivariate TE fails under confounders; high-dimensional conditioning suffers curse of dimensionality.
- 8. Open Question: How does iterative greedy conditioning scale under non-stationary regime shifts?
