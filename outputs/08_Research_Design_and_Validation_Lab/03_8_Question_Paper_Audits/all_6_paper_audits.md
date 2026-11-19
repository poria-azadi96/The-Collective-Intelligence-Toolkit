# 8-Question Scientific Paper Audit Protocol — All 6 Core Papers

Extracted directly from `notebooks/08_research_design_lab.ipynb` (`PAPER_AUDITS`).

## 1. Schreiber (2000) - Transfer Entropy

- 1. Problem: Symmetric mutual information cannot distinguish directional information flow from shared static correlation.
- 2. Hypothesis: Conditioning on target past history isolates directed predictive information T_{Y->X}.
- 3. Model: Conditional mutual information I(Y_past; X_future | X_past) estimated via kernel or k-NN estimators.
- 4. Assumptions: Local stationarity, Markov order k, and absence of unobserved common drivers.
- 5. Key Result: Asymmetric coupling detection in coupled chaotic maps and physiological bivariate series.
- 6. Reproducibility Target: Elimination of spurious links in Common Driver & Cascade motifs via Conditional TE.
- 7. Limitations: Bivariate TE fails under confounders; high-dimensional conditioning suffers curse of dimensionality.
- 8. Open Question: How does iterative greedy conditioning scale under non-stationary regime shifts?

## 2. Hanson & Crutchfield (1997) - Rule 54 Particles

- 1. Problem: How to formally extract emergent computational structures from raw cellular automata spacetime patterns.
- 2. Hypothesis: Regular spacetime domains act as a vacuum; localized defects behave as information-carrying particles.
- 3. Model: Finite-State Transducer (FST) domain filters constructed from regular language grammar of Rule 54.
- 4. Assumptions: Known periodic ether background and deterministic synchronous lattice updates.
- 5. Key Result: Complete particle taxonomy (alpha, beta, gamma+, gamma-) and deterministic collision algebra.
- 6. Reproducibility Target: Filtering period-4 ether to reveal particle tracks and collision vertices.
- 7. Limitations: Domain filters are rule-specific and do not automatically generalize to arbitrary chaotic rules.
- 8. Open Question: How robust is particle collision logic under stochastic update noise?

## 3. Mitchell, Crutchfield & Hraber (1993/1994) - Evolved CA

- 1. Problem: Can evolutionary search discover decentralized local rules that solve a global density task?
- 2. Hypothesis: GA discovers particle-based information propagation, though symmetry breaking impedes search.
- 3. Model: Radius-3 1D CA (N=149) evolved via Genetic Algorithm on uniform-density initial configurations.
- 4. Assumptions: Fitness evaluated over finite random initial states without explicit global communication.
- 5. Key Result: Four evolutionary innovation epochs transitioning from naive block expansion to particle-based logic.
- 6. Reproducibility Target: Accuracy gap between Epoch 1 local majority and GKL particle logic across uniform vs unbiased ICs.
- 7. Limitations: No 2-state radius-3 CA achieves 100% accuracy on all configurations (Land & Belew impossibility).
- 8. Open Question: How does co-evolving the initial configuration distribution alter evolutionary escape from local optima?

## 4. Kauffman / Greil & Drossel (2005) - RBN Attractors

- 1. Problem: How do gene regulatory network topology (K) and update timing govern dynamical stability and cell fates?
- 2. Hypothesis: Critical networks (s = 2Kp(1-p) = 1) balance homeostatic stability with adaptive responsiveness.
- 3. Model: Random Boolean Networks with N nodes, in-degree K, Boolean bias p, and sync vs async updates.
- 4. Assumptions: Annealed approximation for Derrida curve; uniform random wiring without modular hierarchy.
- 5. Key Result: Superpolynomial attractor scaling in sync critical RBNs vs power-law scaling under random async updates.
- 6. Reproducibility Target: Derrida damage spreading curve and sync vs async attractor convergence comparison.
- 7. Limitations: Exhaustive state enumeration is restricted to small N; random sampling undercounts rare attractors.
- 8. Open Question: How do biologically realistic canalizing functions reshape asynchronous attractor stability?

## 5. Tero et al. (2010) - Physarum Tokyo Rail

- 1. Problem: Can a decentralized tubular organism design transport networks balancing cost, efficiency, and fault tolerance?
- 2. Hypothesis: Positive feedback between flux Q_ij and tube conductance D_ij self-organizes near-optimal planar graphs.
- 3. Model: Kirchhoff flow network coupled with adaptive ODE dD_ij/dt = |Q_ij|^gamma / (1 + |Q_ij|^gamma) - D_ij.
- 4. Assumptions: Quasi-steady laminar flow and sequential random source-sink demand pairs.
- 5. Key Result: Reconstructed networks match Tokyo rail efficiency and fault tolerance at comparable total length.
- 6. Reproducibility Target: Multi-objective comparison (Cost, Global Efficiency, Fault Tolerance) against MST baseline.
- 7. Limitations: Dynamic edge-cutting self-healing is an experimental model extension beyond static topology comparison.
- 8. Open Question: How does non-uniform traffic demand weighting alter optimal feedback exponent gamma?

## 6. Arthur (1994) & Challet-Zhang (1997) - El Farol & MG

- 1. Problem: How do boundedly rational agents coordinate when the optimal action depends on being in the minority?
- 2. Hypothesis: Inductive competition among heterogeneous predictors self-organizes attendance near capacity.
- 3. Model: Agents select highest-scoring local predictor from a private bag of S strategies using shared history.
- 4. Assumptions: Agents act independently without direct social communication or collusion in the baseline model.
- 5. Key Result: Attendance fluctuates around capacity (El Farol) and exhibits a phase transition in volatility at alpha_c.
- 6. Reproducibility Target: Attendance stabilization near L=60 and volatility scaling across control parameter alpha.
- 7. Limitations: Adding social network coupling breaks independence and triggers herding volatility.
- 8. Open Question: What network topologies prevent herding cascades while preserving distributed information aggregation?
