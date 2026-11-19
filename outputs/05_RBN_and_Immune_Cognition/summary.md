# Module 05: Random Boolean Networks & Distributed Immune Selection

**Source Notebook:** `notebooks/05_rbn_and_immune_cognition.ipynb`

Kauffman RBN order-to-chaos phase transitions (Derrida annealing curves), synchronous vs. asynchronous attractors (Greil & Drossel, 2005), gene knockouts, and immune clonal selection with McKeithan kinetic proofreading.

## Generated Classroom Figures & Outputs

| # | Subfolder / File | Scenario & Pedagogical Focus | Parameters | Console Output / Metrics |
| :--- | :--- | :--- | :--- | :--- |
| 01 | [`01_RBN_Phase_Transitions_and_Derrida_Curves/01_Ordered_Phase_K1_p0.50.png`](./01_RBN_Phase_Transitions_and_Derrida_Curves/01_Ordered_Phase_K1_p0.50.png) | Ordered Regime (K=1, p=0.50, s=0.50 < 1): perturbations die out below diagonal | `mode=1. RBN Dynamics & Derrida Curve`, `k_inputs=1`, `bias_p=0.5`, `knockout_gene=False` | Regime: ORDERED (s < 1) | Sensitivity s = 0.500 | Sync Attractors = 1 | Async Point-Attractor Convergence = 100.0% |
| 02 | [`01_RBN_Phase_Transitions_and_Derrida_Curves/02_Ordered_Canalizing_Bias_K3_p0.15.png`](./01_RBN_Phase_Transitions_and_Derrida_Curves/02_Ordered_Canalizing_Bias_K3_p0.15.png) | Ordered via Canalizing Bias (K=3, p=0.15, s=0.765 < 1): bias stabilizes high connectivity | `mode=1. RBN Dynamics & Derrida Curve`, `k_inputs=3`, `bias_p=0.15`, `knockout_gene=False` | Regime: ORDERED (s < 1) | Sensitivity s = 0.765 | Sync Attractors = 1 | Async Point-Attractor Convergence = 100.0% |
| 03 | [`01_RBN_Phase_Transitions_and_Derrida_Curves/03_Critical_Edge_K2_p0.50.png`](./01_RBN_Phase_Transitions_and_Derrida_Curves/03_Critical_Edge_K2_p0.50.png) | Critical Edge of Chaos (K=2, p=0.50, s=1.00): Derrida tangent to neutral line | `mode=1. RBN Dynamics & Derrida Curve`, `k_inputs=2`, `bias_p=0.5`, `knockout_gene=False` | Regime: CRITICAL EDGE (s ~ 1) | Sensitivity s = 1.000 | Sync Attractors = 3 | Async Point-Attractor Convergence = 90.8% |
| 04 | [`01_RBN_Phase_Transitions_and_Derrida_Curves/04_Critical_Edge_Tuned_Bias_K3_p0.20.png`](./01_RBN_Phase_Transitions_and_Derrida_Curves/04_Critical_Edge_Tuned_Bias_K3_p0.20.png) | Near-Critical Tuned Bias (K=3, p=0.20, s=0.96): critical boundary at higher in-degree | `mode=1. RBN Dynamics & Derrida Curve`, `k_inputs=3`, `bias_p=0.2`, `knockout_gene=False` | Regime: CRITICAL EDGE (s ~ 1) | Sensitivity s = 0.960 | Sync Attractors = 1 | Async Point-Attractor Convergence = 100.0% |
| 05 | [`01_RBN_Phase_Transitions_and_Derrida_Curves/05_Chaotic_Phase_K3_p0.50.png`](./01_RBN_Phase_Transitions_and_Derrida_Curves/05_Chaotic_Phase_K3_p0.50.png) | Chaotic Regime (K=3, p=0.50, s=1.50 > 1): damage amplification & longer cycles | `mode=1. RBN Dynamics & Derrida Curve`, `k_inputs=3`, `bias_p=0.5`, `knockout_gene=False` | Regime: CHAOTIC (s > 1) | Sensitivity s = 1.500 | Sync Attractors = 2 | Async Point-Attractor Convergence = 72.5% |
| 06 | [`01_RBN_Phase_Transitions_and_Derrida_Curves/06_Deep_Chaos_K4_p0.50.png`](./01_RBN_Phase_Transitions_and_Derrida_Curves/06_Deep_Chaos_K4_p0.50.png) | Deep Chaos Regime (K=4, p=0.50, s=2.00): steep Derrida curve & long L=26 attractor | `mode=1. RBN Dynamics & Derrida Curve`, `k_inputs=4`, `bias_p=0.5`, `knockout_gene=False` | Regime: CHAOTIC (s > 1) | Sensitivity s = 2.000 | Sync Attractors = 2 | Async Point-Attractor Convergence = 52.5% |
| 07 | [`01_RBN_Phase_Transitions_and_Derrida_Curves/07_MultiAttractor_Chaos_K5_p0.50.png`](./01_RBN_Phase_Transitions_and_Derrida_Curves/07_MultiAttractor_Chaos_K5_p0.50.png) | Multi-Attractor Chaos (K=5, p=0.50, s=2.50): 7 coexisting basins & 25% async fixed-point convergence | `mode=1. RBN Dynamics & Derrida Curve`, `k_inputs=5`, `bias_p=0.5`, `knockout_gene=False` | Regime: CHAOTIC (s > 1) | Sensitivity s = 2.500 | Sync Attractors = 7 | Async Point-Attractor Convergence = 25.8% |
| 08 | [`02_Gene_Knockout_Attractor_Shifts/08_Knockout_Node0_Critical_K2_p0.50.png`](./02_Gene_Knockout_Attractor_Shifts/08_Knockout_Node0_Critical_K2_p0.50.png) | Gene Knockout (Node 0) in Critical K=2 RBN: robustness preserving 3 basins & cycle periods | `mode=1. RBN Dynamics & Derrida Curve`, `k_inputs=2`, `bias_p=0.5`, `knockout_gene=True` | Regime: CRITICAL EDGE (s ~ 1) | Sensitivity s = 1.000 | Sync Attractors = 3 | Async Point-Attractor Convergence = 90.8% |
| 09 | [`02_Gene_Knockout_Attractor_Shifts/09_Knockout_Node0_Chaotic_K3_p0.50.png`](./02_Gene_Knockout_Attractor_Shifts/09_Knockout_Node0_Chaotic_K3_p0.50.png) | Gene Knockout (Node 0) in Chaotic K=3 RBN: attractor cycle shrinks from L=16 to L=5 | `mode=1. RBN Dynamics & Derrida Curve`, `k_inputs=3`, `bias_p=0.5`, `knockout_gene=True` | Regime: CHAOTIC (s > 1) | Sensitivity s = 1.500 | Sync Attractors = 2 | Async Point-Attractor Convergence = 72.5% |
| 10 | [`02_Gene_Knockout_Attractor_Shifts/10_Knockout_Node0_DeepChaos_K4_p0.50.png`](./02_Gene_Knockout_Attractor_Shifts/10_Knockout_Node0_DeepChaos_K4_p0.50.png) | Gene Knockout (Node 0) in Deep Chaos K=4 RBN: dramatic basin split (1017->536) & cycle drop (L=26->4) | `mode=1. RBN Dynamics & Derrida Curve`, `k_inputs=4`, `bias_p=0.5`, `knockout_gene=True` | Regime: CHAOTIC (s > 1) | Sensitivity s = 2.000 | Sync Attractors = 2 | Async Point-Attractor Convergence = 52.5% |
| 11 | [`02_Gene_Knockout_Attractor_Shifts/11_Knockout_Node0_MultiAttractor_K5_p0.50.png`](./02_Gene_Knockout_Attractor_Shifts/11_Knockout_Node0_MultiAttractor_K5_p0.50.png) | Gene Knockout (Node 0) in K=5 RBN: 7 attractors reorganize into 3 long cycles (L=20,18,2) | `mode=1. RBN Dynamics & Derrida Curve`, `k_inputs=5`, `bias_p=0.5`, `knockout_gene=True` | Regime: CHAOTIC (s > 1) | Sensitivity s = 2.500 | Sync Attractors = 3 | Async Point-Attractor Convergence = 25.8% |
| 12 | [`03_Immune_Clonal_Selection_and_Proofreading/12_Immune_Clonal_Selection_and_Kinetic_Proofreading.png`](./03_Immune_Clonal_Selection_and_Proofreading/12_Immune_Clonal_Selection_and_Kinetic_Proofreading.png) | Distributed Immune Clonal Selection (affinity maturation) & McKeithan Kinetic Proofreading | `mode=2. Immune Clonal Selection & Proofreading`, `k_inputs=2`, `bias_p=0.5`, `knockout_gene=False` | Final Clonal Affinity = 0.870 | 8-Step Proofreading Specificity Gain = 110.2x |

## Full Console Logs

### 01. Ordered Regime (K=1, p=0.50, s=0.50 < 1): perturbations die out below diagonal
- **File:** [`01_RBN_Phase_Transitions_and_Derrida_Curves/01_Ordered_Phase_K1_p0.50.png`](./01_RBN_Phase_Transitions_and_Derrida_Curves/01_Ordered_Phase_K1_p0.50.png)
- **Parameters:** `{'mode': '1. RBN Dynamics & Derrida Curve', 'k_inputs': 1, 'bias_p': 0.5, 'knockout_gene': False}`
```text
Regime: ORDERED (s < 1) | Sensitivity s = 0.500 | Sync Attractors = 1 | Async Point-Attractor Convergence = 100.0%
```

### 02. Ordered via Canalizing Bias (K=3, p=0.15, s=0.765 < 1): bias stabilizes high connectivity
- **File:** [`01_RBN_Phase_Transitions_and_Derrida_Curves/02_Ordered_Canalizing_Bias_K3_p0.15.png`](./01_RBN_Phase_Transitions_and_Derrida_Curves/02_Ordered_Canalizing_Bias_K3_p0.15.png)
- **Parameters:** `{'mode': '1. RBN Dynamics & Derrida Curve', 'k_inputs': 3, 'bias_p': 0.15, 'knockout_gene': False}`
```text
Regime: ORDERED (s < 1) | Sensitivity s = 0.765 | Sync Attractors = 1 | Async Point-Attractor Convergence = 100.0%
```

### 03. Critical Edge of Chaos (K=2, p=0.50, s=1.00): Derrida tangent to neutral line
- **File:** [`01_RBN_Phase_Transitions_and_Derrida_Curves/03_Critical_Edge_K2_p0.50.png`](./01_RBN_Phase_Transitions_and_Derrida_Curves/03_Critical_Edge_K2_p0.50.png)
- **Parameters:** `{'mode': '1. RBN Dynamics & Derrida Curve', 'k_inputs': 2, 'bias_p': 0.5, 'knockout_gene': False}`
```text
Regime: CRITICAL EDGE (s ~ 1) | Sensitivity s = 1.000 | Sync Attractors = 3 | Async Point-Attractor Convergence = 90.8%
```

### 04. Near-Critical Tuned Bias (K=3, p=0.20, s=0.96): critical boundary at higher in-degree
- **File:** [`01_RBN_Phase_Transitions_and_Derrida_Curves/04_Critical_Edge_Tuned_Bias_K3_p0.20.png`](./01_RBN_Phase_Transitions_and_Derrida_Curves/04_Critical_Edge_Tuned_Bias_K3_p0.20.png)
- **Parameters:** `{'mode': '1. RBN Dynamics & Derrida Curve', 'k_inputs': 3, 'bias_p': 0.2, 'knockout_gene': False}`
```text
Regime: CRITICAL EDGE (s ~ 1) | Sensitivity s = 0.960 | Sync Attractors = 1 | Async Point-Attractor Convergence = 100.0%
```

### 05. Chaotic Regime (K=3, p=0.50, s=1.50 > 1): damage amplification & longer cycles
- **File:** [`01_RBN_Phase_Transitions_and_Derrida_Curves/05_Chaotic_Phase_K3_p0.50.png`](./01_RBN_Phase_Transitions_and_Derrida_Curves/05_Chaotic_Phase_K3_p0.50.png)
- **Parameters:** `{'mode': '1. RBN Dynamics & Derrida Curve', 'k_inputs': 3, 'bias_p': 0.5, 'knockout_gene': False}`
```text
Regime: CHAOTIC (s > 1) | Sensitivity s = 1.500 | Sync Attractors = 2 | Async Point-Attractor Convergence = 72.5%
```

### 06. Deep Chaos Regime (K=4, p=0.50, s=2.00): steep Derrida curve & long L=26 attractor
- **File:** [`01_RBN_Phase_Transitions_and_Derrida_Curves/06_Deep_Chaos_K4_p0.50.png`](./01_RBN_Phase_Transitions_and_Derrida_Curves/06_Deep_Chaos_K4_p0.50.png)
- **Parameters:** `{'mode': '1. RBN Dynamics & Derrida Curve', 'k_inputs': 4, 'bias_p': 0.5, 'knockout_gene': False}`
```text
Regime: CHAOTIC (s > 1) | Sensitivity s = 2.000 | Sync Attractors = 2 | Async Point-Attractor Convergence = 52.5%
```

### 07. Multi-Attractor Chaos (K=5, p=0.50, s=2.50): 7 coexisting basins & 25% async fixed-point convergence
- **File:** [`01_RBN_Phase_Transitions_and_Derrida_Curves/07_MultiAttractor_Chaos_K5_p0.50.png`](./01_RBN_Phase_Transitions_and_Derrida_Curves/07_MultiAttractor_Chaos_K5_p0.50.png)
- **Parameters:** `{'mode': '1. RBN Dynamics & Derrida Curve', 'k_inputs': 5, 'bias_p': 0.5, 'knockout_gene': False}`
```text
Regime: CHAOTIC (s > 1) | Sensitivity s = 2.500 | Sync Attractors = 7 | Async Point-Attractor Convergence = 25.8%
```

### 08. Gene Knockout (Node 0) in Critical K=2 RBN: robustness preserving 3 basins & cycle periods
- **File:** [`02_Gene_Knockout_Attractor_Shifts/08_Knockout_Node0_Critical_K2_p0.50.png`](./02_Gene_Knockout_Attractor_Shifts/08_Knockout_Node0_Critical_K2_p0.50.png)
- **Parameters:** `{'mode': '1. RBN Dynamics & Derrida Curve', 'k_inputs': 2, 'bias_p': 0.5, 'knockout_gene': True}`
```text
Regime: CRITICAL EDGE (s ~ 1) | Sensitivity s = 1.000 | Sync Attractors = 3 | Async Point-Attractor Convergence = 90.8%
```

### 09. Gene Knockout (Node 0) in Chaotic K=3 RBN: attractor cycle shrinks from L=16 to L=5
- **File:** [`02_Gene_Knockout_Attractor_Shifts/09_Knockout_Node0_Chaotic_K3_p0.50.png`](./02_Gene_Knockout_Attractor_Shifts/09_Knockout_Node0_Chaotic_K3_p0.50.png)
- **Parameters:** `{'mode': '1. RBN Dynamics & Derrida Curve', 'k_inputs': 3, 'bias_p': 0.5, 'knockout_gene': True}`
```text
Regime: CHAOTIC (s > 1) | Sensitivity s = 1.500 | Sync Attractors = 2 | Async Point-Attractor Convergence = 72.5%
```

### 10. Gene Knockout (Node 0) in Deep Chaos K=4 RBN: dramatic basin split (1017->536) & cycle drop (L=26->4)
- **File:** [`02_Gene_Knockout_Attractor_Shifts/10_Knockout_Node0_DeepChaos_K4_p0.50.png`](./02_Gene_Knockout_Attractor_Shifts/10_Knockout_Node0_DeepChaos_K4_p0.50.png)
- **Parameters:** `{'mode': '1. RBN Dynamics & Derrida Curve', 'k_inputs': 4, 'bias_p': 0.5, 'knockout_gene': True}`
```text
Regime: CHAOTIC (s > 1) | Sensitivity s = 2.000 | Sync Attractors = 2 | Async Point-Attractor Convergence = 52.5%
```

### 11. Gene Knockout (Node 0) in K=5 RBN: 7 attractors reorganize into 3 long cycles (L=20,18,2)
- **File:** [`02_Gene_Knockout_Attractor_Shifts/11_Knockout_Node0_MultiAttractor_K5_p0.50.png`](./02_Gene_Knockout_Attractor_Shifts/11_Knockout_Node0_MultiAttractor_K5_p0.50.png)
- **Parameters:** `{'mode': '1. RBN Dynamics & Derrida Curve', 'k_inputs': 5, 'bias_p': 0.5, 'knockout_gene': True}`
```text
Regime: CHAOTIC (s > 1) | Sensitivity s = 2.500 | Sync Attractors = 3 | Async Point-Attractor Convergence = 25.8%
```

### 12. Distributed Immune Clonal Selection (affinity maturation) & McKeithan Kinetic Proofreading
- **File:** [`03_Immune_Clonal_Selection_and_Proofreading/12_Immune_Clonal_Selection_and_Kinetic_Proofreading.png`](./03_Immune_Clonal_Selection_and_Proofreading/12_Immune_Clonal_Selection_and_Kinetic_Proofreading.png)
- **Parameters:** `{'mode': '2. Immune Clonal Selection & Proofreading', 'k_inputs': 2, 'bias_p': 0.5, 'knockout_gene': False}`
```text
Final Clonal Affinity = 0.870 | 8-Step Proofreading Specificity Gain = 110.2x
```
