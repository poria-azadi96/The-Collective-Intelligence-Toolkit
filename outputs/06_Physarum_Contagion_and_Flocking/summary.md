# Module 06: Adaptive Networks (Physarum), Contagion & Vicsek Flocking

**Source Notebook:** `notebooks/06_physarum_and_contagion.ipynb`

Tero et al. (2010) Physarum Tokyo Rail optimization & trunk disruption recovery, Centola (2010) Simple vs. Complex Contagion on small-world networks, and Vicsek flocking.

## Generated Classroom Figures & Outputs

| # | Subfolder / File | Scenario & Pedagogical Focus | Parameters | Console Output / Metrics |
| :--- | :--- | :--- | :--- | :--- |
| 01 | [`01_Physarum_Tokyo_Rail_Solver/01_Physarum_fault_tolerant_mesh_gamma0.50.png`](./01_Physarum_Tokyo_Rail_Solver/01_Physarum_fault_tolerant_mesh_gamma0.50.png) | Physarum Fault-Tolerant Mesh (gamma=0.50): 100% single-edge fault tolerance & 1.29x efficiency | `mode=1. Physarum Tokyo Rail Solver`, `gamma_or_noise=0.5`, `disrupt_or_rewire=False` | Physarum Cost/MST = 2.93x | Efficiency/MST = 1.29x | Single-Edge Fault Tolerance = 100.0% (vs MST 0.0%) |
| 02 | [`01_Physarum_Tokyo_Rail_Solver/02_Physarum_balanced_backbone_gamma0.60.png`](./01_Physarum_Tokyo_Rail_Solver/02_Physarum_balanced_backbone_gamma0.60.png) | Physarum Balanced Backbone (gamma=0.60): matches MST global efficiency (1.03x) across planar loops | `mode=1. Physarum Tokyo Rail Solver`, `gamma_or_noise=0.6`, `disrupt_or_rewire=False` | Physarum Cost/MST = 2.08x | Efficiency/MST = 1.03x | Single-Edge Fault Tolerance = 0.0% (vs MST 0.0%) |
| 03 | [`01_Physarum_Tokyo_Rail_Solver/03_Physarum_weak_feedback_gamma0.70.png`](./01_Physarum_Tokyo_Rail_Solver/03_Physarum_weak_feedback_gamma0.70.png) | Physarum Moderate Pruning (gamma=0.70): Cost=1.05x MST with emergent arterial loops | `mode=1. Physarum Tokyo Rail Solver`, `gamma_or_noise=0.7`, `disrupt_or_rewire=False` | Physarum Cost/MST = 1.05x | Efficiency/MST = 0.69x | Single-Edge Fault Tolerance = 0.0% (vs MST 0.0%) |
| 04 | [`01_Physarum_Tokyo_Rail_Solver/04_Physarum_optimal_default_gamma1.40.png`](./01_Physarum_Tokyo_Rail_Solver/04_Physarum_optimal_default_gamma1.40.png) | Physarum Default Gamma (gamma=1.40): strong hub pruning around central Tokyo station | `mode=1. Physarum Tokyo Rail Solver`, `gamma_or_noise=1.4`, `disrupt_or_rewire=False` | Physarum Cost/MST = 0.19x | Efficiency/MST = 0.11x | Single-Edge Fault Tolerance = 0.0% (vs MST 0.0%) |
| 05 | [`01_Physarum_Tokyo_Rail_Solver/05_Physarum_strong_pruning_gamma2.00.png`](./01_Physarum_Tokyo_Rail_Solver/05_Physarum_strong_pruning_gamma2.00.png) | Physarum Strong Pruning (gamma=2.00): aggressive pruning toward minimal hub core | `mode=1. Physarum Tokyo Rail Solver`, `gamma_or_noise=2.0`, `disrupt_or_rewire=False` | Physarum Cost/MST = 0.20x | Efficiency/MST = 0.11x | Single-Edge Fault Tolerance = 0.0% (vs MST 0.0%) |
| 06 | [`01_Physarum_Tokyo_Rail_Solver/06_Physarum_hub_disruption_recovery_gamma0.60.png`](./01_Physarum_Tokyo_Rail_Solver/06_Physarum_hub_disruption_recovery_gamma0.60.png) | Physarum Hub Disruption Recovery (gamma=0.60, Cut=True): bypass rerouting after trunk severing | `mode=1. Physarum Tokyo Rail Solver`, `gamma_or_noise=0.6`, `disrupt_or_rewire=True` | Physarum Cost/MST = 2.01x | Efficiency/MST = 1.04x | Single-Edge Fault Tolerance = 0.0% (vs MST 0.0%) |
| 07 | [`01_Physarum_Tokyo_Rail_Solver/07_Physarum_hub_disruption_recovery_gamma1.40.png`](./01_Physarum_Tokyo_Rail_Solver/07_Physarum_hub_disruption_recovery_gamma1.40.png) | Physarum Hub Disruption Recovery (gamma=1.40, Cut=True): rerouting under strong feedback | `mode=1. Physarum Tokyo Rail Solver`, `gamma_or_noise=1.4`, `disrupt_or_rewire=True` | Physarum Cost/MST = 0.28x | Efficiency/MST = 0.17x | Single-Edge Fault Tolerance = 0.0% (vs MST 0.0%) |
| 08 | [`02_Simple_vs_Complex_Contagion/08_Contagion_Clustered_Lattice_p0.02.png`](./02_Simple_vs_Complex_Contagion/08_Contagion_Clustered_Lattice_p0.02.png) | Clustered Lattice (rewire p=0.02): high local clustering enables Complex Contagion spread | `mode=2. Simple vs. Complex Contagion`, `gamma_or_noise=1.4`, `disrupt_or_rewire=False` | INSIGHT: Random shortcuts accelerate Simple Contagion, but destroy local clustering needed for Complex Contagion! |
| 09 | [`02_Simple_vs_Complex_Contagion/09_Contagion_Small_World_Shortcuts_p0.35.png`](./02_Simple_vs_Complex_Contagion/09_Contagion_Small_World_Shortcuts_p0.35.png) | Small-World Shortcuts (rewire p=0.35): Simple Contagion accelerates while Complex Contagion collapses | `mode=2. Simple vs. Complex Contagion`, `gamma_or_noise=1.4`, `disrupt_or_rewire=True` | INSIGHT: Random shortcuts accelerate Simple Contagion, but destroy local clustering needed for Complex Contagion! |
| 10 | [`03_Vicsek_Collective_Flocking/10_Vicsek_Ordered_Flocking_low_noise_eta0.40.png`](./03_Vicsek_Collective_Flocking/10_Vicsek_Ordered_Flocking_low_noise_eta0.40.png) | Vicsek Ordered Flocking (low noise eta=0.40): coherent collective polarization (Phi -> 1) | `mode=3. Vicsek Collective Flocking`, `gamma_or_noise=0.2`, `disrupt_or_rewire=False` | Vicsek Flocking | Noise eta = 0.40 | Final Polarization Order Parameter Phi = 0.993 |
| 11 | [`03_Vicsek_Collective_Flocking/11_Vicsek_SliderMin_Flocking_eta1.00.png`](./03_Vicsek_Collective_Flocking/11_Vicsek_SliderMin_Flocking_eta1.00.png) | Vicsek Coherent Phase (slider min, noise eta=1.00): strong global alignment | `mode=3. Vicsek Collective Flocking`, `gamma_or_noise=0.5`, `disrupt_or_rewire=False` | Vicsek Flocking | Noise eta = 1.00 | Final Polarization Order Parameter Phi = 0.922 |
| 12 | [`03_Vicsek_Collective_Flocking/12_Vicsek_Transition_Regime_eta2.80.png`](./03_Vicsek_Collective_Flocking/12_Vicsek_Transition_Regime_eta2.80.png) | Vicsek Transition Regime (noise eta=2.80): partial local clusters & fluctuating order parameter | `mode=3. Vicsek Collective Flocking`, `gamma_or_noise=1.4`, `disrupt_or_rewire=False` | Vicsek Flocking | Noise eta = 2.80 | Final Polarization Order Parameter Phi = 0.646 |
| 13 | [`03_Vicsek_Collective_Flocking/13_Vicsek_Disordered_Gas_high_noise_eta4.40.png`](./03_Vicsek_Collective_Flocking/13_Vicsek_Disordered_Gas_high_noise_eta4.40.png) | Vicsek Disordered Gas (high noise eta=4.40): random headings & collapsed polarization | `mode=3. Vicsek Collective Flocking`, `gamma_or_noise=2.2`, `disrupt_or_rewire=False` | Vicsek Flocking | Noise eta = 4.40 | Final Polarization Order Parameter Phi = 0.181 |

## Full Console Logs

### 01. Physarum Fault-Tolerant Mesh (gamma=0.50): 100% single-edge fault tolerance & 1.29x efficiency
- **File:** [`01_Physarum_Tokyo_Rail_Solver/01_Physarum_fault_tolerant_mesh_gamma0.50.png`](./01_Physarum_Tokyo_Rail_Solver/01_Physarum_fault_tolerant_mesh_gamma0.50.png)
- **Parameters:** `{'mode': '1. Physarum Tokyo Rail Solver', 'gamma_or_noise': 0.5, 'disrupt_or_rewire': False}`
```text
Physarum Cost/MST = 2.93x | Efficiency/MST = 1.29x | Single-Edge Fault Tolerance = 100.0% (vs MST 0.0%)
```

### 02. Physarum Balanced Backbone (gamma=0.60): matches MST global efficiency (1.03x) across planar loops
- **File:** [`01_Physarum_Tokyo_Rail_Solver/02_Physarum_balanced_backbone_gamma0.60.png`](./01_Physarum_Tokyo_Rail_Solver/02_Physarum_balanced_backbone_gamma0.60.png)
- **Parameters:** `{'mode': '1. Physarum Tokyo Rail Solver', 'gamma_or_noise': 0.6, 'disrupt_or_rewire': False}`
```text
Physarum Cost/MST = 2.08x | Efficiency/MST = 1.03x | Single-Edge Fault Tolerance = 0.0% (vs MST 0.0%)
```

### 03. Physarum Moderate Pruning (gamma=0.70): Cost=1.05x MST with emergent arterial loops
- **File:** [`01_Physarum_Tokyo_Rail_Solver/03_Physarum_weak_feedback_gamma0.70.png`](./01_Physarum_Tokyo_Rail_Solver/03_Physarum_weak_feedback_gamma0.70.png)
- **Parameters:** `{'mode': '1. Physarum Tokyo Rail Solver', 'gamma_or_noise': 0.7, 'disrupt_or_rewire': False}`
```text
Physarum Cost/MST = 1.05x | Efficiency/MST = 0.69x | Single-Edge Fault Tolerance = 0.0% (vs MST 0.0%)
```

### 04. Physarum Default Gamma (gamma=1.40): strong hub pruning around central Tokyo station
- **File:** [`01_Physarum_Tokyo_Rail_Solver/04_Physarum_optimal_default_gamma1.40.png`](./01_Physarum_Tokyo_Rail_Solver/04_Physarum_optimal_default_gamma1.40.png)
- **Parameters:** `{'mode': '1. Physarum Tokyo Rail Solver', 'gamma_or_noise': 1.4, 'disrupt_or_rewire': False}`
```text
Physarum Cost/MST = 0.19x | Efficiency/MST = 0.11x | Single-Edge Fault Tolerance = 0.0% (vs MST 0.0%)
```

### 05. Physarum Strong Pruning (gamma=2.00): aggressive pruning toward minimal hub core
- **File:** [`01_Physarum_Tokyo_Rail_Solver/05_Physarum_strong_pruning_gamma2.00.png`](./01_Physarum_Tokyo_Rail_Solver/05_Physarum_strong_pruning_gamma2.00.png)
- **Parameters:** `{'mode': '1. Physarum Tokyo Rail Solver', 'gamma_or_noise': 2.0, 'disrupt_or_rewire': False}`
```text
Physarum Cost/MST = 0.20x | Efficiency/MST = 0.11x | Single-Edge Fault Tolerance = 0.0% (vs MST 0.0%)
```

### 06. Physarum Hub Disruption Recovery (gamma=0.60, Cut=True): bypass rerouting after trunk severing
- **File:** [`01_Physarum_Tokyo_Rail_Solver/06_Physarum_hub_disruption_recovery_gamma0.60.png`](./01_Physarum_Tokyo_Rail_Solver/06_Physarum_hub_disruption_recovery_gamma0.60.png)
- **Parameters:** `{'mode': '1. Physarum Tokyo Rail Solver', 'gamma_or_noise': 0.6, 'disrupt_or_rewire': True}`
```text
Physarum Cost/MST = 2.01x | Efficiency/MST = 1.04x | Single-Edge Fault Tolerance = 0.0% (vs MST 0.0%)
```

### 07. Physarum Hub Disruption Recovery (gamma=1.40, Cut=True): rerouting under strong feedback
- **File:** [`01_Physarum_Tokyo_Rail_Solver/07_Physarum_hub_disruption_recovery_gamma1.40.png`](./01_Physarum_Tokyo_Rail_Solver/07_Physarum_hub_disruption_recovery_gamma1.40.png)
- **Parameters:** `{'mode': '1. Physarum Tokyo Rail Solver', 'gamma_or_noise': 1.4, 'disrupt_or_rewire': True}`
```text
Physarum Cost/MST = 0.28x | Efficiency/MST = 0.17x | Single-Edge Fault Tolerance = 0.0% (vs MST 0.0%)
```

### 08. Clustered Lattice (rewire p=0.02): high local clustering enables Complex Contagion spread
- **File:** [`02_Simple_vs_Complex_Contagion/08_Contagion_Clustered_Lattice_p0.02.png`](./02_Simple_vs_Complex_Contagion/08_Contagion_Clustered_Lattice_p0.02.png)
- **Parameters:** `{'mode': '2. Simple vs. Complex Contagion', 'gamma_or_noise': 1.4, 'disrupt_or_rewire': False}`
```text
INSIGHT: Random shortcuts accelerate Simple Contagion, but destroy local clustering needed for Complex Contagion!
```

### 09. Small-World Shortcuts (rewire p=0.35): Simple Contagion accelerates while Complex Contagion collapses
- **File:** [`02_Simple_vs_Complex_Contagion/09_Contagion_Small_World_Shortcuts_p0.35.png`](./02_Simple_vs_Complex_Contagion/09_Contagion_Small_World_Shortcuts_p0.35.png)
- **Parameters:** `{'mode': '2. Simple vs. Complex Contagion', 'gamma_or_noise': 1.4, 'disrupt_or_rewire': True}`
```text
INSIGHT: Random shortcuts accelerate Simple Contagion, but destroy local clustering needed for Complex Contagion!
```

### 10. Vicsek Ordered Flocking (low noise eta=0.40): coherent collective polarization (Phi -> 1)
- **File:** [`03_Vicsek_Collective_Flocking/10_Vicsek_Ordered_Flocking_low_noise_eta0.40.png`](./03_Vicsek_Collective_Flocking/10_Vicsek_Ordered_Flocking_low_noise_eta0.40.png)
- **Parameters:** `{'mode': '3. Vicsek Collective Flocking', 'gamma_or_noise': 0.2, 'disrupt_or_rewire': False}`
```text
Vicsek Flocking | Noise eta = 0.40 | Final Polarization Order Parameter Phi = 0.993
```

### 11. Vicsek Coherent Phase (slider min, noise eta=1.00): strong global alignment
- **File:** [`03_Vicsek_Collective_Flocking/11_Vicsek_SliderMin_Flocking_eta1.00.png`](./03_Vicsek_Collective_Flocking/11_Vicsek_SliderMin_Flocking_eta1.00.png)
- **Parameters:** `{'mode': '3. Vicsek Collective Flocking', 'gamma_or_noise': 0.5, 'disrupt_or_rewire': False}`
```text
Vicsek Flocking | Noise eta = 1.00 | Final Polarization Order Parameter Phi = 0.922
```

### 12. Vicsek Transition Regime (noise eta=2.80): partial local clusters & fluctuating order parameter
- **File:** [`03_Vicsek_Collective_Flocking/12_Vicsek_Transition_Regime_eta2.80.png`](./03_Vicsek_Collective_Flocking/12_Vicsek_Transition_Regime_eta2.80.png)
- **Parameters:** `{'mode': '3. Vicsek Collective Flocking', 'gamma_or_noise': 1.4, 'disrupt_or_rewire': False}`
```text
Vicsek Flocking | Noise eta = 2.80 | Final Polarization Order Parameter Phi = 0.646
```

### 13. Vicsek Disordered Gas (high noise eta=4.40): random headings & collapsed polarization
- **File:** [`03_Vicsek_Collective_Flocking/13_Vicsek_Disordered_Gas_high_noise_eta4.40.png`](./03_Vicsek_Collective_Flocking/13_Vicsek_Disordered_Gas_high_noise_eta4.40.png)
- **Parameters:** `{'mode': '3. Vicsek Collective Flocking', 'gamma_or_noise': 2.2, 'disrupt_or_rewire': False}`
```text
Vicsek Flocking | Noise eta = 4.40 | Final Polarization Order Parameter Phi = 0.181
```
