# 5. Tero et al. (2010) - Physarum Tokyo Rail

- 1. Problem: Can a decentralized tubular organism design transport networks balancing cost, efficiency, and fault tolerance?
- 2. Hypothesis: Positive feedback between flux Q_ij and tube conductance D_ij self-organizes near-optimal planar graphs.
- 3. Model: Kirchhoff flow network coupled with adaptive ODE dD_ij/dt = |Q_ij|^gamma / (1 + |Q_ij|^gamma) - D_ij.
- 4. Assumptions: Quasi-steady laminar flow and sequential random source-sink demand pairs.
- 5. Key Result: Reconstructed networks match Tokyo rail efficiency and fault tolerance at comparable total length.
- 6. Reproducibility Target: Multi-objective comparison (Cost, Global Efficiency, Fault Tolerance) against MST baseline.
- 7. Limitations: Dynamic edge-cutting self-healing is an experimental model extension beyond static topology comparison.
- 8. Open Question: How does non-uniform traffic demand weighting alter optimal feedback exponent gamma?
