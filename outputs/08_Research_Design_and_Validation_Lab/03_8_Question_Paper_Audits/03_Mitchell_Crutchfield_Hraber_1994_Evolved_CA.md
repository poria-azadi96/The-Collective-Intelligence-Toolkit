# 3. Mitchell, Crutchfield & Hraber (1993/1994) - Evolved CA

- 1. Problem: Can evolutionary search discover decentralized local rules that solve a global density task?
- 2. Hypothesis: GA discovers particle-based information propagation, though symmetry breaking impedes search.
- 3. Model: Radius-3 1D CA (N=149) evolved via Genetic Algorithm on uniform-density initial configurations.
- 4. Assumptions: Fitness evaluated over finite random initial states without explicit global communication.
- 5. Key Result: Four evolutionary innovation epochs transitioning from naive block expansion to particle-based logic.
- 6. Reproducibility Target: Accuracy gap between Epoch 1 local majority and GKL particle logic across uniform vs unbiased ICs.
- 7. Limitations: No 2-state radius-3 CA achieves 100% accuracy on all configurations (Land & Belew impossibility).
- 8. Open Question: How does co-evolving the initial configuration distribution alter evolutionary escape from local optima?
