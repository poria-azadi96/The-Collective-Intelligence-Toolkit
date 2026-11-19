# Module 07: Collective Wisdom, Bounded Confidence & El Farol Market

**Source Notebook:** `notebooks/07_el_farol_and_opinion_dynamics.ipynb`

Scott Page's Diversity Prediction Theorem & error correlation, Hegselmann-Krause bounded confidence fragmentation vs. consensus, Arthur's El Farol Bar (1994), and Challet-Zhang Minority Game (1997) with networked herding extension.

## Generated Classroom Figures & Outputs

| # | Subfolder / File | Scenario & Pedagogical Focus | Parameters | Console Output / Metrics |
| :--- | :--- | :--- | :--- | :--- |
| 01 | [`01_Diversity_Prediction_Theorem/01_DPT_Low_Correlation_rho0.05.png`](./01_Diversity_Prediction_Theorem/01_DPT_Low_Correlation_rho0.05.png) | DPT Low Error Correlation (rho=0.05): high diversity yields drastic collective error reduction | `study=1. Diversity Prediction Theorem`, `param_val=0.05`, `herding_ext=False` | Single-Trial Algebraic Identity Check: E_indiv (86.01) - Diversity (85.71) = E_coll (0.31) |
| 02 | [`01_Diversity_Prediction_Theorem/02_DPT_Moderate_Correlation_rho0.25.png`](./01_Diversity_Prediction_Theorem/02_DPT_Moderate_Correlation_rho0.25.png) | DPT Moderate Error Correlation (rho=0.25): partial diversity benefit | `study=1. Diversity Prediction Theorem`, `param_val=0.25`, `herding_ext=False` | Single-Trial Algebraic Identity Check: E_indiv (67.84) - Diversity (67.66) = E_coll (0.18) |
| 03 | [`01_Diversity_Prediction_Theorem/03_DPT_High_Correlation_Groupthink_rho0.80.png`](./01_Diversity_Prediction_Theorem/03_DPT_High_Correlation_Groupthink_rho0.80.png) | DPT High Correlation / Groupthink (rho=0.80): diversity collapses & collective error approaches individual error | `study=1. Diversity Prediction Theorem`, `param_val=0.8`, `herding_ext=False` | Single-Trial Algebraic Identity Check: E_indiv (22.70) - Diversity (18.04) = E_coll (4.66) |
| 04 | [`02_Bounded_Confidence_Opinion_Dynamics/04_HK_Fragmentation_eps0.08.png`](./02_Bounded_Confidence_Opinion_Dynamics/04_HK_Fragmentation_eps0.08.png) | Hegselmann-Krause Fragmentation (eps=0.08): multiple isolated echo chambers | `study=2. Bounded Confidence (HK vs DeGroot)`, `param_val=0.08`, `herding_ext=False` | HK Bounded Confidence eps=0.08 -> Converged into 4 distinct opinion cluster(s). |
| 05 | [`02_Bounded_Confidence_Opinion_Dynamics/05_HK_Polarization_eps0.18.png`](./02_Bounded_Confidence_Opinion_Dynamics/05_HK_Polarization_eps0.18.png) | Hegselmann-Krause Polarization (eps=0.18): bipolar / tripolar clustering | `study=2. Bounded Confidence (HK vs DeGroot)`, `param_val=0.18`, `herding_ext=False` | HK Bounded Confidence eps=0.18 -> Converged into 2 distinct opinion cluster(s). |
| 06 | [`02_Bounded_Confidence_Opinion_Dynamics/06_HK_Critical_Transition_eps0.25.png`](./02_Bounded_Confidence_Opinion_Dynamics/06_HK_Critical_Transition_eps0.25.png) | Hegselmann-Krause Near-Consensus Transition (eps=0.25): bifurcation toward consensus | `study=2. Bounded Confidence (HK vs DeGroot)`, `param_val=0.25`, `herding_ext=False` | HK Bounded Confidence eps=0.25 -> Converged into 1 distinct opinion cluster(s). |
| 07 | [`02_Bounded_Confidence_Opinion_Dynamics/07_HK_Full_Consensus_eps0.40.png`](./02_Bounded_Confidence_Opinion_Dynamics/07_HK_Full_Consensus_eps0.40.png) | Hegselmann-Krause Full Consensus (eps=0.40): converges to single DeGroot-like consensus | `study=2. Bounded Confidence (HK vs DeGroot)`, `param_val=0.4`, `herding_ext=False` | HK Bounded Confidence eps=0.40 -> Converged into 1 distinct opinion cluster(s). |
| 08 | [`03_El_Farol_and_Minority_Game/08_El_Farol_and_MG_Independent_M3.png`](./03_El_Farol_and_Minority_Game/08_El_Farol_and_MG_Independent_M3.png) | El Farol Bar (M=3) & Original Minority Game (independent agents beat coin-flip volatility) | `study=3. El Farol Bar & Minority Game`, `param_val=0.25`, `herding_ext=False` | El Farol Mean Attendance = 58.00 (Threshold=60) | MG Memory M=3 (alpha=0.079) |
| 09 | [`03_El_Farol_and_Minority_Game/09_El_Farol_and_MG_Herding_Extension_M3.png`](./03_El_Farol_and_Minority_Game/09_El_Farol_and_MG_Herding_Extension_M3.png) | El Farol (M=3) & Minority Game with Networked Herding Extension (q=0.35 spikes volatility) | `study=3. El Farol Bar & Minority Game`, `param_val=0.25`, `herding_ext=True` | El Farol Mean Attendance = 58.00 (Threshold=60) | MG Memory M=3 (alpha=0.079) |
| 10 | [`03_El_Farol_and_Minority_Game/10_El_Farol_and_MG_Long_Memory_M5_Herding.png`](./03_El_Farol_and_Minority_Game/10_El_Farol_and_MG_Long_Memory_M5_Herding.png) | El Farol Longer Memory (M=5) & Minority Game with Herding Overlay | `study=3. El Farol Bar & Minority Game`, `param_val=0.75`, `herding_ext=True` | El Farol Mean Attendance = 58.76 (Threshold=60) | MG Memory M=5 (alpha=0.317) |

## Full Console Logs

### 01. DPT Low Error Correlation (rho=0.05): high diversity yields drastic collective error reduction
- **File:** [`01_Diversity_Prediction_Theorem/01_DPT_Low_Correlation_rho0.05.png`](./01_Diversity_Prediction_Theorem/01_DPT_Low_Correlation_rho0.05.png)
- **Parameters:** `{'study': '1. Diversity Prediction Theorem', 'param_val': 0.05, 'herding_ext': False}`
```text
Single-Trial Algebraic Identity Check: E_indiv (86.01) - Diversity (85.71) = E_coll (0.31)
```

### 02. DPT Moderate Error Correlation (rho=0.25): partial diversity benefit
- **File:** [`01_Diversity_Prediction_Theorem/02_DPT_Moderate_Correlation_rho0.25.png`](./01_Diversity_Prediction_Theorem/02_DPT_Moderate_Correlation_rho0.25.png)
- **Parameters:** `{'study': '1. Diversity Prediction Theorem', 'param_val': 0.25, 'herding_ext': False}`
```text
Single-Trial Algebraic Identity Check: E_indiv (67.84) - Diversity (67.66) = E_coll (0.18)
```

### 03. DPT High Correlation / Groupthink (rho=0.80): diversity collapses & collective error approaches individual error
- **File:** [`01_Diversity_Prediction_Theorem/03_DPT_High_Correlation_Groupthink_rho0.80.png`](./01_Diversity_Prediction_Theorem/03_DPT_High_Correlation_Groupthink_rho0.80.png)
- **Parameters:** `{'study': '1. Diversity Prediction Theorem', 'param_val': 0.8, 'herding_ext': False}`
```text
Single-Trial Algebraic Identity Check: E_indiv (22.70) - Diversity (18.04) = E_coll (4.66)
```

### 04. Hegselmann-Krause Fragmentation (eps=0.08): multiple isolated echo chambers
- **File:** [`02_Bounded_Confidence_Opinion_Dynamics/04_HK_Fragmentation_eps0.08.png`](./02_Bounded_Confidence_Opinion_Dynamics/04_HK_Fragmentation_eps0.08.png)
- **Parameters:** `{'study': '2. Bounded Confidence (HK vs DeGroot)', 'param_val': 0.08, 'herding_ext': False}`
```text
HK Bounded Confidence eps=0.08 -> Converged into 4 distinct opinion cluster(s).
```

### 05. Hegselmann-Krause Polarization (eps=0.18): bipolar / tripolar clustering
- **File:** [`02_Bounded_Confidence_Opinion_Dynamics/05_HK_Polarization_eps0.18.png`](./02_Bounded_Confidence_Opinion_Dynamics/05_HK_Polarization_eps0.18.png)
- **Parameters:** `{'study': '2. Bounded Confidence (HK vs DeGroot)', 'param_val': 0.18, 'herding_ext': False}`
```text
HK Bounded Confidence eps=0.18 -> Converged into 2 distinct opinion cluster(s).
```

### 06. Hegselmann-Krause Near-Consensus Transition (eps=0.25): bifurcation toward consensus
- **File:** [`02_Bounded_Confidence_Opinion_Dynamics/06_HK_Critical_Transition_eps0.25.png`](./02_Bounded_Confidence_Opinion_Dynamics/06_HK_Critical_Transition_eps0.25.png)
- **Parameters:** `{'study': '2. Bounded Confidence (HK vs DeGroot)', 'param_val': 0.25, 'herding_ext': False}`
```text
HK Bounded Confidence eps=0.25 -> Converged into 1 distinct opinion cluster(s).
```

### 07. Hegselmann-Krause Full Consensus (eps=0.40): converges to single DeGroot-like consensus
- **File:** [`02_Bounded_Confidence_Opinion_Dynamics/07_HK_Full_Consensus_eps0.40.png`](./02_Bounded_Confidence_Opinion_Dynamics/07_HK_Full_Consensus_eps0.40.png)
- **Parameters:** `{'study': '2. Bounded Confidence (HK vs DeGroot)', 'param_val': 0.4, 'herding_ext': False}`
```text
HK Bounded Confidence eps=0.40 -> Converged into 1 distinct opinion cluster(s).
```

### 08. El Farol Bar (M=3) & Original Minority Game (independent agents beat coin-flip volatility)
- **File:** [`03_El_Farol_and_Minority_Game/08_El_Farol_and_MG_Independent_M3.png`](./03_El_Farol_and_Minority_Game/08_El_Farol_and_MG_Independent_M3.png)
- **Parameters:** `{'study': '3. El Farol Bar & Minority Game', 'param_val': 0.25, 'herding_ext': False}`
```text
El Farol Mean Attendance = 58.00 (Threshold=60) | MG Memory M=3 (alpha=0.079)
```

### 09. El Farol (M=3) & Minority Game with Networked Herding Extension (q=0.35 spikes volatility)
- **File:** [`03_El_Farol_and_Minority_Game/09_El_Farol_and_MG_Herding_Extension_M3.png`](./03_El_Farol_and_Minority_Game/09_El_Farol_and_MG_Herding_Extension_M3.png)
- **Parameters:** `{'study': '3. El Farol Bar & Minority Game', 'param_val': 0.25, 'herding_ext': True}`
```text
El Farol Mean Attendance = 58.00 (Threshold=60) | MG Memory M=3 (alpha=0.079)
```

### 10. El Farol Longer Memory (M=5) & Minority Game with Herding Overlay
- **File:** [`03_El_Farol_and_Minority_Game/10_El_Farol_and_MG_Long_Memory_M5_Herding.png`](./03_El_Farol_and_Minority_Game/10_El_Farol_and_MG_Long_Memory_M5_Herding.png)
- **Parameters:** `{'study': '3. El Farol Bar & Minority Game', 'param_val': 0.75, 'herding_ext': True}`
```text
El Farol Mean Attendance = 58.76 (Threshold=60) | MG Memory M=5 (alpha=0.317)
```
