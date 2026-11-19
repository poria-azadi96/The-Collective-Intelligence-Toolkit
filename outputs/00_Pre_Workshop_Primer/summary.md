# Module 00: Pre-Workshop Primer

**Source Notebook:** `foundations/00_pre_workshop_primer.ipynb`

Dynamical states, nonlinear logistic map feedback regimes (fixed point, period-doubling, chaos), lagged phase-portrait embeddings, and binary Shannon entropy.

## Generated Classroom Figures & Outputs

| # | Subfolder / File | Scenario & Pedagogical Focus | Parameters | Console Output / Metrics |
| :--- | :--- | :--- | :--- | :--- |
| 01 | [`01_fixed_point_r2.80_tau1_p0.10.png`](./01_fixed_point_r2.80_tau1_p0.10.png) | Fixed-point convergence (r=2.80) & skewed low-entropy binary source (p=0.10) | `r_growth=2.8`, `lag_tau=1`, `p_coin=0.1` | Lagged Covariance Cov(X_{t-1}, X_t) = 0.0005 | Shannon Entropy = 0.4690 bits |
| 02 | [`02_period2_oscillation_r3.20_tau1_p0.50.png`](./02_period2_oscillation_r3.20_tau1_p0.50.png) | Period-2 limit cycle (r=3.20) & maximum binary Shannon entropy (p=0.50) | `r_growth=3.2`, `lag_tau=1`, `p_coin=0.5` | Lagged Covariance Cov(X_{t-1}, X_t) = -0.0200 | Shannon Entropy = 1.0000 bits |
| 03 | [`03_period4_bifurcation_r3.52_tau2_p0.80.png`](./03_period4_bifurcation_r3.52_tau2_p0.80.png) | Period-4 oscillation (r=3.52), lag tau=2 embedding & symmetric entropy (p=0.80) | `r_growth=3.52`, `lag_tau=2`, `p_coin=0.8` | Lagged Covariance Cov(X_{t-2}, X_t) = 0.0387 | Shannon Entropy = 0.7219 bits |
| 04 | [`04_chaos_default_r3.75_tau1_p0.50.png`](./04_chaos_default_r3.75_tau1_p0.50.png) | Chaotic trajectory (r=3.75) & lag-1 parabolic attractor reconstruction | `r_growth=3.75`, `lag_tau=1`, `p_coin=0.5` | Lagged Covariance Cov(X_{t-1}, X_t) = -0.0410 | Shannon Entropy = 1.0000 bits |
| 05 | [`05_deep_chaos_folded_attractor_r3.95_tau2_p0.50.png`](./05_deep_chaos_folded_attractor_r3.95_tau2_p0.50.png) | Deep chaos (r=3.95) & lag-2 folded portrait showing nonlinear structure with near-zero linear correlation | `r_growth=3.95`, `lag_tau=2`, `p_coin=0.5` | Lagged Covariance Cov(X_{t-2}, X_t) = -0.0094 | Shannon Entropy = 1.0000 bits |

## Full Console Logs

### 01. Fixed-point convergence (r=2.80) & skewed low-entropy binary source (p=0.10)
- **File:** [`01_fixed_point_r2.80_tau1_p0.10.png`](./01_fixed_point_r2.80_tau1_p0.10.png)
- **Parameters:** `{'r_growth': 2.8, 'lag_tau': 1, 'p_coin': 0.1}`
```text
Lagged Covariance Cov(X_{t-1}, X_t) = 0.0005 | Shannon Entropy = 0.4690 bits
```

### 02. Period-2 limit cycle (r=3.20) & maximum binary Shannon entropy (p=0.50)
- **File:** [`02_period2_oscillation_r3.20_tau1_p0.50.png`](./02_period2_oscillation_r3.20_tau1_p0.50.png)
- **Parameters:** `{'r_growth': 3.2, 'lag_tau': 1, 'p_coin': 0.5}`
```text
Lagged Covariance Cov(X_{t-1}, X_t) = -0.0200 | Shannon Entropy = 1.0000 bits
```

### 03. Period-4 oscillation (r=3.52), lag tau=2 embedding & symmetric entropy (p=0.80)
- **File:** [`03_period4_bifurcation_r3.52_tau2_p0.80.png`](./03_period4_bifurcation_r3.52_tau2_p0.80.png)
- **Parameters:** `{'r_growth': 3.52, 'lag_tau': 2, 'p_coin': 0.8}`
```text
Lagged Covariance Cov(X_{t-2}, X_t) = 0.0387 | Shannon Entropy = 0.7219 bits
```

### 04. Chaotic trajectory (r=3.75) & lag-1 parabolic attractor reconstruction
- **File:** [`04_chaos_default_r3.75_tau1_p0.50.png`](./04_chaos_default_r3.75_tau1_p0.50.png)
- **Parameters:** `{'r_growth': 3.75, 'lag_tau': 1, 'p_coin': 0.5}`
```text
Lagged Covariance Cov(X_{t-1}, X_t) = -0.0410 | Shannon Entropy = 1.0000 bits
```

### 05. Deep chaos (r=3.95) & lag-2 folded portrait showing nonlinear structure with near-zero linear correlation
- **File:** [`05_deep_chaos_folded_attractor_r3.95_tau2_p0.50.png`](./05_deep_chaos_folded_attractor_r3.95_tau2_p0.50.png)
- **Parameters:** `{'r_growth': 3.95, 'lag_tau': 2, 'p_coin': 0.5}`
```text
Lagged Covariance Cov(X_{t-2}, X_t) = -0.0094 | Shannon Entropy = 1.0000 bits
```
