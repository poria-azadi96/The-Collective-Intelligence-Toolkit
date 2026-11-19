# Module 07: Collective Wisdom, Bounded Confidence & El Farol Market

**Source Notebook:** `notebooks/07_el_farol_and_opinion_dynamics.ipynb`

Scott Page's DPT, Hegselmann-Krause vs DeGroot consensus, Brian Arthur's El Farol Bar, Challet-Zhang Minority Game with Networked Herding, 3-Tier protocols, and 3 real-world apps.

## Generated Classroom Figures (Baselines, 3-Tier Protocol & Real-World Applications)

| # | Category | File | Pedagogical Focus | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| 01 | Baseline Dashboard | [`01_Diversity_Prediction_Theorem/01_DPT_Low_Correlation_rho0.05.png`](./01_Diversity_Prediction_Theorem/01_DPT_Low_Correlation_rho0.05.png) | DPT Low Error Correlation (rho=0.05): strong crowd wisdom gain | `study=1. Diversity Prediction Theorem`, `rho=0.05` |
| 02 | Baseline Dashboard | [`01_Diversity_Prediction_Theorem/02_DPT_Moderate_Correlation_rho0.25.png`](./01_Diversity_Prediction_Theorem/02_DPT_Moderate_Correlation_rho0.25.png) | DPT Moderate Error Correlation (rho=0.25): partial diversity benefit | `study=1. Diversity Prediction Theorem`, `rho=0.25` |
| 03 | Baseline Dashboard | [`01_Diversity_Prediction_Theorem/03_DPT_High_Correlation_Groupthink_rho0.80.png`](./01_Diversity_Prediction_Theorem/03_DPT_High_Correlation_Groupthink_rho0.80.png) | DPT High Correlation / Groupthink (rho=0.80): diversity collapses | `study=1. Diversity Prediction Theorem`, `rho=0.8` |
| 04 | Baseline Dashboard | [`02_Bounded_Confidence_Opinion_Dynamics/04_HK_Fragmentation_eps0.08.png`](./02_Bounded_Confidence_Opinion_Dynamics/04_HK_Fragmentation_eps0.08.png) | Hegselmann-Krause Fragmentation (eps=0.05): isolated echo chambers | `study=2. Bounded Confidence (HK vs DeGroot)`, `eps=0.05` |
| 05 | Baseline Dashboard | [`02_Bounded_Confidence_Opinion_Dynamics/05_HK_Polarization_eps0.18.png`](./02_Bounded_Confidence_Opinion_Dynamics/05_HK_Polarization_eps0.18.png) | Hegselmann-Krause Polarization (eps=0.18): multipolar clusters | `study=2. Bounded Confidence (HK vs DeGroot)`, `eps=0.18` |
| 06 | Baseline Dashboard | [`02_Bounded_Confidence_Opinion_Dynamics/07_HK_Full_Consensus_eps0.40.png`](./02_Bounded_Confidence_Opinion_Dynamics/07_HK_Full_Consensus_eps0.40.png) | Hegselmann-Krause Global Consensus (eps=0.40): 1 unified cluster | `study=2. Bounded Confidence (HK vs DeGroot)`, `eps=0.4` |
| 07 | Baseline Dashboard | [`03_El_Farol_and_Minority_Game/08_El_Farol_and_MG_Independent_M3.png`](./03_El_Farol_and_Minority_Game/08_El_Farol_and_MG_Independent_M3.png) | El Farol Bar & Minority Game Independent Agents (below coin-flip) | `study=3. El Farol Bar & Minority Game`, `memory_m=3`, `herding_ext=False` |
| 08 | Baseline Dashboard | [`03_El_Farol_and_Minority_Game/09_El_Farol_and_MG_Herding_Extension_M3.png`](./03_El_Farol_and_Minority_Game/09_El_Farol_and_MG_Herding_Extension_M3.png) | El Farol & Minority Game with Networked Herding (q=0.35 spike) | `study=3. El Farol Bar & Minority Game`, `memory_m=3`, `herding_q=0.35`, `herding_ext=True` |
| 09 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier1_DPT_Identity_and_Rho_Sweep.png`](./03_3Tier_Protocol/Tier1_DPT_Identity_and_Rho_Sweep.png) | Tier 1: Page single-trial identity & rho correlation sweep | `default` |
| 10 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier2_HK_Bifurcation_vs_DeGroot.png`](./03_3Tier_Protocol/Tier2_HK_Bifurcation_vs_DeGroot.png) | Tier 2: HK presets (0.05, 0.18, 0.40) vs DeGroot linear consensus | `default` |
| 11 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier3_ElFarol_MinorityGame_Herding.png`](./03_3Tier_Protocol/Tier3_ElFarol_MinorityGame_Herding.png) | Tier 3: El Farol L=60, MG cusp alpha_c & networked herding volatility | `default` |
| 12 | Real-World Demo | [`04_Real_World_Applications/App1_Financial_Limit_Order_Book.png`](./04_Real_World_Applications/App1_Financial_Limit_Order_Book.png) | App 7.1: Chartist vs fundamentalist order book & flash-crash liquidity drought | `default` |
| 13 | Real-World Demo | [`04_Real_World_Applications/App2_Forecasting_Delphi_Ensemble.png`](./04_Real_World_Applications/App2_Forecasting_Delphi_Ensemble.png) | App 7.2: Delphi decorrelation & diversity-weighted forecasting pipeline | `default` |
| 14 | Real-World Demo | [`04_Real_World_Applications/App3_LLM_Multi_Agent_Debate.png`](./04_Real_World_Applications/App3_LLM_Multi_Agent_Debate.png) | App 7.3: Multi-agent AI/LLM debate accuracy vs sycophantic convergence | `default` |

## Classroom Animated GIFs

- [`animations/01_hk_opinion_bifurcation.gif`](./animations/01_hk_opinion_bifurcation.gif)
- [`animations/02_el_farol_and_minority_herding.gif`](./animations/02_el_farol_and_minority_herding.gif)
