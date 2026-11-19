# Module 02: Transfer Entropy, Delay Reconstruction & Network Inference

**Source Notebook:** `notebooks/02_transfer_entropy_and_network_inference.ipynb`

Schreiber Transfer Entropy, Frenzel-Pompe / KSG k-NN estimator, Wibral interaction delay scan, circular surrogates, BH-FDR network recovery, 3-Tier protocols, and 3 domain applications.

## Generated Classroom Figures (Baselines, 3-Tier Protocol & Real-World Applications)

| # | Category | File | Pedagogical Focus | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| 01 | Baseline Dashboard | [`01_Causal_Motifs/01_Direct_Coupling_c0.70_n450.png`](./01_Causal_Motifs/01_Direct_Coupling_c0.70_n450.png) | Direct Coupling (X->Y lag u=2): peak at u=2 survives conditioning | `motif_type=Direct_Coupling`, `coupling=0.7`, `n_samples=450`, `n_surrogates=60` |
| 02 | Baseline Dashboard | [`01_Causal_Motifs/02_Indirect_Cascade_c0.70_n450.png`](./01_Causal_Motifs/02_Indirect_Cascade_c0.70_n450.png) | Indirect Cascade (X->Z->Y): Conditional TE | Z filters indirect link | `motif_type=Indirect_Cascade`, `coupling=0.7`, `n_samples=450`, `n_surrogates=60` |
| 03 | Baseline Dashboard | [`01_Causal_Motifs/03_Common_Driver_c0.70_n450.png`](./01_Causal_Motifs/03_Common_Driver_c0.70_n450.png) | Common Driver (Z->X,Y): Conditional TE | Z removes spurious bivariate link | `motif_type=Common_Driver`, `coupling=0.7`, `n_samples=450`, `n_surrogates=60` |
| 04 | Baseline Dashboard | [`01_Causal_Motifs/04_Unobserved_Confounder_c0.70_n450.png`](./01_Causal_Motifs/04_Unobserved_Confounder_c0.70_n450.png) | Unobserved Confounder (hidden U->X,Y): irreducible observational false positive | `motif_type=Unobserved_Confounder`, `coupling=0.7`, `n_samples=450`, `n_surrogates=60` |
| 05 | Baseline Dashboard | [`02_Sensitivity_Coupling_and_SampleSize/05_Direct_Coupling_weak_c0.30_n450.png`](./02_Sensitivity_Coupling_and_SampleSize/05_Direct_Coupling_weak_c0.30_n450.png) | Weak Direct Coupling (c=0.30): reduced TE near surrogate null | `motif_type=Direct_Coupling`, `coupling=0.3`, `n_samples=450`, `n_surrogates=60` |
| 06 | Baseline Dashboard | [`02_Sensitivity_Coupling_and_SampleSize/06_Direct_Coupling_strong_c0.85_n450.png`](./02_Sensitivity_Coupling_and_SampleSize/06_Direct_Coupling_strong_c0.85_n450.png) | Strong Direct Coupling (c=0.85): sharp lag-2 peak well above null | `motif_type=Direct_Coupling`, `coupling=0.85`, `n_samples=450`, `n_surrogates=60` |
| 07 | Baseline Dashboard | [`02_Sensitivity_Coupling_and_SampleSize/07_Common_Driver_small_sample_n250.png`](./02_Sensitivity_Coupling_and_SampleSize/07_Common_Driver_small_sample_n250.png) | Small Sample Regime (n=250): higher surrogate threshold variance | `motif_type=Common_Driver`, `coupling=0.7`, `n_samples=250`, `n_surrogates=60` |
| 08 | Baseline Dashboard | [`02_Sensitivity_Coupling_and_SampleSize/08_Direct_Coupling_large_sample_n800.png`](./02_Sensitivity_Coupling_and_SampleSize/08_Direct_Coupling_large_sample_n800.png) | Large Sample Regime (n=800): tight surrogate cutoff & clean lag-2 peak | `motif_type=Direct_Coupling`, `coupling=0.7`, `n_samples=800`, `n_surrogates=60` |
| 09 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier1_Observation_Confounder_Cascade.png`](./03_3Tier_Protocol/Tier1_Observation_Confounder_Cascade.png) | Tier 1: Common_Driver & Indirect_Cascade filtered by Conditional TE | Z | `n_surrogates=70` |
| 10 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier2_Analysis_Wibral_Delay_Tau2.png`](./03_3Tier_Protocol/Tier2_Analysis_Wibral_Delay_Tau2.png) | Tier 2: Wibral scan u=1..5 recovers tau=2 + AIS+TE chain rule | `n_surrogates=70` |
| 11 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier3_StressTest_Unobserved_Confounder.png`](./03_3Tier_Protocol/Tier3_StressTest_Unobserved_Confounder.png) | Tier 3: Unobserved Confounder sample sweep (250->800) shows irreducible false positives | `n_surrogates=70` |
| 12 | Real-World Demo | [`04_Real_World_Applications/App1_Effective_Brain_Connectivity.png`](./04_Real_World_Applications/App1_Effective_Brain_Connectivity.png) | App 2.1: 4-region effective brain connectivity (THAL, PFC, PAR, MOT) via Cond TE + BH-FDR | `default` |
| 13 | Real-World Demo | [`04_Real_World_Applications/App2_Finance_Volatility_Contagion.png`](./04_Real_World_Applications/App2_Finance_Volatility_Contagion.png) | App 2.2: Multi-asset nonlinear volatility spillover & systemic risk hub discovery | `default` |
| 14 | Real-World Demo | [`04_Real_World_Applications/App3_Systems_Biology_Gene_Network.png`](./04_Real_World_Applications/App3_Systems_Biology_Gene_Network.png) | App 2.3: Hill-function transcriptional cascade reconstruction (G1->G2->G3, G1-|G4) | `default` |

## Classroom Animated GIFs

- [`animations/01_wibral_delay_scan_and_filtering.gif`](./animations/01_wibral_delay_scan_and_filtering.gif)
- [`animations/02_brain_connectome_neural_dynamics.gif`](./animations/02_brain_connectome_neural_dynamics.gif)
