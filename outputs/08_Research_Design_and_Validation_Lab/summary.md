# Module 08: Computational Research Design & Validation Lab

**Source Notebook:** `notebooks/08_research_design_lab.ipynb`

8-Question Paper Audit Engine across 6 classics, Track A BH-FDR power calibration (T_80), Track B null baseline testing (Cohen's d), 5-Part Research Brief builder, 3-Tier protocols, and 3 real-world apps.

## Generated Classroom Figures (Baselines, 3-Tier Protocol & Real-World Applications)

| # | Category | File | Pedagogical Focus | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| 01 | Baseline Dashboard | [`01_Track_A_Observational_FDR_and_Power/01_TrackA_LowSNR0.55_T100.png`](./01_Track_A_Observational_FDR_and_Power/01_TrackA_LowSNR0.55_T100.png) | Track A Underpowered (SNR=0.55, T=100): uncorrected false positives | `paper_choice=1. Schreiber (2000) - Transfer Entropy`, `track=Track A: Observational FDR & Power`, `sample_size=100`, `signal_snr=0.55` |
| 02 | Baseline Dashboard | [`01_Track_A_Observational_FDR_and_Power/02_TrackA_LowSNR0.55_T300.png`](./01_Track_A_Observational_FDR_and_Power/02_TrackA_LowSNR0.55_T300.png) | Track A Underpowered (SNR=0.55, T=300): BH-FDR prevents false discoveries | `paper_choice=1. Schreiber (2000) - Transfer Entropy`, `track=Track A: Observational FDR & Power`, `sample_size=300`, `signal_snr=0.55` |
| 03 | Baseline Dashboard | [`01_Track_A_Observational_FDR_and_Power/03_TrackA_LowSNR0.55_T800.png`](./01_Track_A_Observational_FDR_and_Power/03_TrackA_LowSNR0.55_T800.png) | Track A Low SNR (SNR=0.55, T=800): power shortfall under multiple testing | `paper_choice=1. Schreiber (2000) - Transfer Entropy`, `track=Track A: Observational FDR & Power`, `sample_size=800`, `signal_snr=0.55` |
| 04 | Baseline Dashboard | [`01_Track_A_Observational_FDR_and_Power/04_TrackA_Calibrated_SNR1.25_T1200.png`](./01_Track_A_Observational_FDR_and_Power/04_TrackA_Calibrated_SNR1.25_T1200.png) | Track A Calibrated (SNR=1.25, T=1200): crosses 80% BH-FDR power target | `paper_choice=1. Schreiber (2000) - Transfer Entropy`, `track=Track A: Observational FDR & Power`, `sample_size=1200`, `signal_snr=1.25` |
| 05 | Baseline Dashboard | [`02_Track_B_Generative_ABM_vs_Null_Model/05_TrackB_ABM_vs_Null_Baseline.png`](./02_Track_B_Generative_ABM_vs_Null_Model/05_TrackB_ABM_vs_Null_Baseline.png) | Track B Mechanistic Transition vs Null Baseline & Cohen's d separation | `paper_choice=5. Tero et al. (2010) - Physarum Tokyo Rail`, `track=Track B: Generative ABM vs Null Model`, `coupling_strength=0.7` |
| 06 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier1_Paper_Audit_Q7_to_Q8_Matrix.png`](./03_3Tier_Protocol/Tier1_Paper_Audit_Q7_to_Q8_Matrix.png) | Tier 1: 8-Question Paper Audit Inspector mapping Q7 -> Q8 across 6 classics | `default` |
| 07 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier2_TrackA_Power_Calibration_T80.png`](./03_3Tier_Protocol/Tier2_TrackA_Power_Calibration_T80.png) | Tier 2: Track A BH-FDR power curve solving minimum sample size T_80 | `default` |
| 08 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier3_TrackB_Null_CohensD.png`](./03_3Tier_Protocol/Tier3_TrackB_Null_CohensD.png) | Tier 3: Track B Null separation Cohen's d & 5-Part Research Brief export | `default` |
| 09 | Real-World Demo | [`04_Real_World_Applications/App1_Q1_Peer_Review_Stress_Suite.png`](./04_Real_World_Applications/App1_Q1_Peer_Review_Stress_Suite.png) | App 8.1: Automated Q7 failure-mode stress-test across workshop models | `default` |
| 10 | Real-World Demo | [`04_Real_World_Applications/App2_Industrial_RD_Power_Pipeline.png`](./04_Real_World_Applications/App2_Industrial_RD_Power_Pipeline.png) | App 8.2: Pre-experiment multi-SNR T_80 power & null Cohen's d pipeline | `default` |
| 11 | Real-World Demo | [`04_Real_World_Applications/App3_Cross_Module_Grand_Synthesis.png`](./04_Real_World_Applications/App3_Cross_Module_Grand_Synthesis.png) | App 8.3: Cross-module reproducibility scorecard & grand synthesis report | `default` |

## Classroom Animated GIFs

- [`animations/01_trackA_bh_fdr_power_calibration.gif`](./animations/01_trackA_bh_fdr_power_calibration.gif)
