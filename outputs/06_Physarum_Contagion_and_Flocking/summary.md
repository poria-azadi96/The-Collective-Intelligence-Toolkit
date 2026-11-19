# Module 06: Physarum Self-Organization, Complex Contagion & Vicsek Flocking

**Source Notebook:** `notebooks/06_physarum_and_contagion.ipynb`

Tero et al. (2010) sparse-Laplacian Physarum solver with live severing, Centola (2010) Simple vs Complex Contagion, Vicsek collective motion, 3-Tier protocols, and 3 real-world apps.

## Generated Classroom Figures (Baselines, 3-Tier Protocol & Real-World Applications)

| # | Category | File | Pedagogical Focus | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| 01 | Baseline Dashboard | [`01_Physarum_Tokyo_Rail_Solver/01_Physarum_fault_tolerant_mesh_gamma0.50.png`](./01_Physarum_Tokyo_Rail_Solver/01_Physarum_fault_tolerant_mesh_gamma0.50.png) | Physarum Fault-Tolerant Mesh (mu=0.50): resilient loop mesh | `mode=Physarum Tokyo Rail`, `mu=0.5`, `disrupt_edges=False` |
| 02 | Baseline Dashboard | [`01_Physarum_Tokyo_Rail_Solver/02_Physarum_balanced_backbone_gamma0.60.png`](./01_Physarum_Tokyo_Rail_Solver/02_Physarum_balanced_backbone_gamma0.60.png) | Physarum Balanced Backbone (mu=0.60): matches MST efficiency | `mode=Physarum Tokyo Rail`, `mu=0.6`, `disrupt_edges=False` |
| 03 | Baseline Dashboard | [`01_Physarum_Tokyo_Rail_Solver/04_Physarum_optimal_default_gamma1.40.png`](./01_Physarum_Tokyo_Rail_Solver/04_Physarum_optimal_default_gamma1.40.png) | Physarum Optimal Pruning (mu=1.40): hub-centered rail backbone | `mode=Physarum Tokyo Rail`, `mu=1.4`, `disrupt_edges=False` |
| 04 | Baseline Dashboard | [`01_Physarum_Tokyo_Rail_Solver/05_Physarum_strong_pruning_gamma2.00.png`](./01_Physarum_Tokyo_Rail_Solver/05_Physarum_strong_pruning_gamma2.00.png) | Physarum Strong Pruning (mu=2.00): minimal hub core | `mode=Physarum Tokyo Rail`, `mu=2.0`, `disrupt_edges=False` |
| 05 | Baseline Dashboard | [`01_Physarum_Tokyo_Rail_Solver/06_Physarum_hub_disruption_recovery_gamma0.60.png`](./01_Physarum_Tokyo_Rail_Solver/06_Physarum_hub_disruption_recovery_gamma0.60.png) | Physarum Live Trunk Severing Recovery (mu=0.60, Cut=True) | `mode=Physarum Tokyo Rail`, `mu=0.6`, `disrupt_edges=True` |
| 06 | Baseline Dashboard | [`01_Physarum_Tokyo_Rail_Solver/07_Physarum_hub_disruption_recovery_gamma1.40.png`](./01_Physarum_Tokyo_Rail_Solver/07_Physarum_hub_disruption_recovery_gamma1.40.png) | Physarum Live Trunk Severing Recovery (mu=1.40, Cut=True) | `mode=Physarum Tokyo Rail`, `mu=1.4`, `disrupt_edges=True` |
| 07 | Baseline Dashboard | [`02_Simple_vs_Complex_Contagion/08_Contagion_Clustered_Lattice_p0.02.png`](./02_Simple_vs_Complex_Contagion/08_Contagion_Clustered_Lattice_p0.02.png) | Clustered Lattice (p_rewire=0.02): Complex Contagion succeeds | `mode=Complex Contagion`, `rewire_prob=0.02`, `contagion_threshold=2` |
| 08 | Baseline Dashboard | [`02_Simple_vs_Complex_Contagion/09_Contagion_Small_World_Shortcuts_p0.35.png`](./02_Simple_vs_Complex_Contagion/09_Contagion_Small_World_Shortcuts_p0.35.png) | Small-World Shortcuts (p_rewire=0.35): Complex Contagion collapses | `mode=Complex Contagion`, `rewire_prob=0.35`, `contagion_threshold=2` |
| 09 | Baseline Dashboard | [`03_Vicsek_Collective_Flocking/10_Vicsek_Ordered_Flocking_low_noise_eta0.40.png`](./03_Vicsek_Collective_Flocking/10_Vicsek_Ordered_Flocking_low_noise_eta0.40.png) | Vicsek Ordered Flocking (eta=0.40): coherent polarization | `mode=Vicsek Flocking`, `vicsek_eta=0.4` |
| 10 | Baseline Dashboard | [`03_Vicsek_Collective_Flocking/12_Vicsek_Transition_Regime_eta2.80.png`](./03_Vicsek_Collective_Flocking/12_Vicsek_Transition_Regime_eta2.80.png) | Vicsek Transition Regime (eta=2.80): partial local alignment | `mode=Vicsek Flocking`, `vicsek_eta=2.8` |
| 11 | Baseline Dashboard | [`03_Vicsek_Collective_Flocking/13_Vicsek_Disordered_Gas_high_noise_eta4.40.png`](./03_Vicsek_Collective_Flocking/13_Vicsek_Disordered_Gas_high_noise_eta4.40.png) | Vicsek Disordered Gas (eta=4.40): collapsed polarization | `mode=Vicsek Flocking`, `vicsek_eta=4.4` |
| 12 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier1_Physarum_Tokyo_Rail.png`](./03_3Tier_Protocol/Tier1_Physarum_Tokyo_Rail.png) | Tier 1: Physarum sparse-Laplacian self-organization vs MST | `default` |
| 13 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier2_Complex_Contagion_Rewiring.png`](./03_3Tier_Protocol/Tier2_Complex_Contagion_Rewiring.png) | Tier 2: Watts-Strogatz rewiring sweep showing clustering collapse | `default` |
| 14 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier3_Vicsek_Transition_and_Severing.png`](./03_3Tier_Protocol/Tier3_Vicsek_Transition_and_Severing.png) | Tier 3: Vicsek phi(eta) phase curve & live Physarum edge severing | `default` |
| 15 | Real-World Demo | [`04_Real_World_Applications/App1_Resilient_Transit_Power_Grid.png`](./04_Real_World_Applications/App1_Resilient_Transit_Power_Grid.png) | App 6.1: Self-healing Tokyo grid under live severing & non-uniform demand | `default` |
| 16 | Real-World Demo | [`04_Real_World_Applications/App2_Public_Health_Diffusion.png`](./04_Real_World_Applications/App2_Public_Health_Diffusion.png) | App 6.2: Clustered vs dispersed seeding for T=2 behavioral adoption | `default` |
| 17 | Real-World Demo | [`04_Real_World_Applications/App3_Autonomous_Drone_Swarm.png`](./04_Real_World_Applications/App3_Autonomous_Drone_Swarm.png) | App 6.3: Decentralized 2D autonomous drone obstacle-avoidance swarm | `default` |

## Classroom Animated GIFs

- [`animations/01_physarum_tokyo_rail_self_healing.gif`](./animations/01_physarum_tokyo_rail_self_healing.gif)
- [`animations/02_vicsek_flocking_phase_transition.gif`](./animations/02_vicsek_flocking_phase_transition.gif)
- [`animations/03_autonomous_drone_swarm_obstacles.gif`](./animations/03_autonomous_drone_swarm_obstacles.gif)
