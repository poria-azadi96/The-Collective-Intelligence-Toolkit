# Module 05: Random Boolean Networks & Distributed Immune Selection

**Source Notebook:** `notebooks/05_rbn_and_immune_cognition.ipynb`

Kauffman RBN criticality (Derrida curve), sync vs async attractors, gene knockouts, canalizing GRNs, immune clonal maturation, and kinetic proofreading.

## Generated Classroom Figures (Baselines, 3-Tier Protocol & Real-World Applications)

| # | Category | File | Pedagogical Focus | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| 01 | Baseline Dashboard | [`01_RBN_Phase_Transitions_and_Derrida_Curves/01_Ordered_Phase_K1_p0.50.png`](./01_RBN_Phase_Transitions_and_Derrida_Curves/01_Ordered_Phase_K1_p0.50.png) | Ordered Regime (K=1, p=0.50, s=0.50 < 1): perturbations die out | `mode=1. RBN Dynamics & Derrida Curve`, `k_inputs=1`, `bias_p=0.5`, `knockout_gene=False` |
| 02 | Baseline Dashboard | [`01_RBN_Phase_Transitions_and_Derrida_Curves/02_Ordered_Canalizing_Bias_K3_p0.15.png`](./01_RBN_Phase_Transitions_and_Derrida_Curves/02_Ordered_Canalizing_Bias_K3_p0.15.png) | Ordered via Canalizing Bias (K=3, p=0.15, s=0.765 < 1) | `mode=1. RBN Dynamics & Derrida Curve`, `k_inputs=3`, `bias_p=0.15`, `knockout_gene=False` |
| 03 | Baseline Dashboard | [`01_RBN_Phase_Transitions_and_Derrida_Curves/03_Critical_Edge_K2_p0.50.png`](./01_RBN_Phase_Transitions_and_Derrida_Curves/03_Critical_Edge_K2_p0.50.png) | Critical Edge of Chaos (K=2, p=0.50, s=1.00): tangent to 45-deg line | `mode=1. RBN Dynamics & Derrida Curve`, `k_inputs=2`, `bias_p=0.5`, `knockout_gene=False`, `seed=6` |
| 04 | Baseline Dashboard | [`01_RBN_Phase_Transitions_and_Derrida_Curves/04_Critical_Edge_Tuned_Bias_K3_p0.20.png`](./01_RBN_Phase_Transitions_and_Derrida_Curves/04_Critical_Edge_Tuned_Bias_K3_p0.20.png) | Near-Critical Tuned Bias (K=3, p=0.20, s=0.96) | `mode=1. RBN Dynamics & Derrida Curve`, `k_inputs=3`, `bias_p=0.2`, `knockout_gene=False` |
| 05 | Baseline Dashboard | [`01_RBN_Phase_Transitions_and_Derrida_Curves/05_Chaotic_Phase_K3_p0.50.png`](./01_RBN_Phase_Transitions_and_Derrida_Curves/05_Chaotic_Phase_K3_p0.50.png) | Chaotic Regime (K=3, p=0.50, s=1.50 > 1): damage amplification | `mode=1. RBN Dynamics & Derrida Curve`, `k_inputs=3`, `bias_p=0.5`, `knockout_gene=False` |
| 06 | Baseline Dashboard | [`01_RBN_Phase_Transitions_and_Derrida_Curves/06_Deep_Chaos_K4_p0.50.png`](./01_RBN_Phase_Transitions_and_Derrida_Curves/06_Deep_Chaos_K4_p0.50.png) | Deep Chaos Regime (K=4, p=0.50, s=2.00): steep Derrida curve | `mode=1. RBN Dynamics & Derrida Curve`, `k_inputs=4`, `bias_p=0.5`, `knockout_gene=False` |
| 07 | Baseline Dashboard | [`02_Gene_Knockout_Attractor_Shifts/08_Knockout_Node0_Critical_K2_p0.50.png`](./02_Gene_Knockout_Attractor_Shifts/08_Knockout_Node0_Critical_K2_p0.50.png) | Knockout Node 0 in Critical K=2 RBN: basin merging & robustness | `mode=1. RBN Dynamics & Derrida Curve`, `k_inputs=2`, `bias_p=0.5`, `knockout_gene=True`, `seed=6` |
| 08 | Baseline Dashboard | [`02_Gene_Knockout_Attractor_Shifts/09_Knockout_Node0_Chaotic_K3_p0.50.png`](./02_Gene_Knockout_Attractor_Shifts/09_Knockout_Node0_Chaotic_K3_p0.50.png) | Knockout Node 0 in Chaotic K=3 RBN: structural cycle collapse | `mode=1. RBN Dynamics & Derrida Curve`, `k_inputs=3`, `bias_p=0.5`, `knockout_gene=True` |
| 09 | Baseline Dashboard | [`02_Gene_Knockout_Attractor_Shifts/10_Knockout_Node0_DeepChaos_K4_p0.50.png`](./02_Gene_Knockout_Attractor_Shifts/10_Knockout_Node0_DeepChaos_K4_p0.50.png) | Knockout Node 0 in Deep Chaos K=4 RBN: dramatic basin split | `mode=1. RBN Dynamics & Derrida Curve`, `k_inputs=4`, `bias_p=0.5`, `knockout_gene=True` |
| 10 | Baseline Dashboard | [`03_Immune_Clonal_Selection_and_Proofreading/12_Immune_Clonal_Selection_and_Kinetic_Proofreading.png`](./03_Immune_Clonal_Selection_and_Proofreading/12_Immune_Clonal_Selection_and_Kinetic_Proofreading.png) | Immune Clonal Selection (45 Gens) & McKeithan Kinetic Proofreading | `mode=2. Immune Clonal Selection & Proofreading`, `proofreading_steps=8` |
| 11 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier1_Observation_Derrida_K1_to_K4.png`](./03_3Tier_Protocol/Tier1_Observation_Derrida_K1_to_K4.png) | Tier 1: Sweep K=1..4 at p=0.50 showing Derrida crossing s=1 | `default` |
| 12 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier2_Analysis_Knockout_Sync_vs_Async.png`](./03_3Tier_Protocol/Tier2_Analysis_Knockout_Sync_vs_Async.png) | Tier 2: Knockout Node 0 basin merging & Sync vs Async convergence | `default` |
| 13 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier3_Advanced_Immune_Proofreading.png`](./03_3Tier_Protocol/Tier3_Advanced_Immune_Proofreading.png) | Tier 3: 45-gen Clonal Maturation & N=1..10 exponential specificity | `default` |
| 14 | Real-World Demo | [`04_Real_World_Applications/App1_Cancer_Attractor_Landscape.png`](./04_Real_World_Applications/App1_Cancer_Attractor_Landscape.png) | App 5.1: Oncogenic mutations distorting epigenetic cell-fate basins | `default` |
| 15 | Real-World Demo | [`04_Real_World_Applications/App2_Immunotherapy_Specificity.png`](./04_Real_World_Applications/App2_Immunotherapy_Specificity.png) | App 5.2: T-cell tolerance & kinetic proofreading therapeutic window | `default` |
| 16 | Real-World Demo | [`04_Real_World_Applications/App3_Canalizing_Async_GRNs.png`](./04_Real_World_Applications/App3_Canalizing_Async_GRNs.png) | App 5.3: Canalizing truth tables stabilizing asynchronous GRNs | `default` |

## Classroom Animated GIFs

- [`animations/01_rbn_derrida_phase_transition.gif`](./animations/01_rbn_derrida_phase_transition.gif)
- [`animations/02_immune_clonal_selection_maturation.gif`](./animations/02_immune_clonal_selection_maturation.gif)
