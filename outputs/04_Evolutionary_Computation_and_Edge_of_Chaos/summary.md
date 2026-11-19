# Module 04: Evolutionary Computation, Genetic Algorithms & Edge of Chaos

**Source Notebook:** `notebooks/04_evolutionary_computation_holland.ipynb`

128-bit Radius-3 CA density classification, GKL rule, Mitchell et al. epochs, Land-Belew boundary, Holland's Schema Theorem, 3-Tier protocols, and 3 domain applications.

## Generated Classroom Figures (Baselines, 3-Tier Protocol & Real-World Applications)

| # | Category | File | Pedagogical Focus | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| 01 | Baseline Dashboard | [`01_Innovation_Epochs_and_Density_Classification/01_Epoch1_Naive_Majority_rho0.54.png`](./01_Innovation_Epochs_and_Density_Classification/01_Epoch1_Naive_Majority_rho0.54.png) | Epoch 1 (Naive Majority, rho=0.54): frozen domain walls stall | `checkpoint=Epoch 1`, `init_density=0.54`, `run_short_ga=False` |
| 02 | Baseline Dashboard | [`01_Innovation_Epochs_and_Density_Classification/02_Epoch1_Naive_Majority_rho0.46.png`](./01_Innovation_Epochs_and_Density_Classification/02_Epoch1_Naive_Majority_rho0.46.png) | Epoch 1 (Naive Majority, rho=0.46): symmetric frozen stalling | `checkpoint=Epoch 1`, `init_density=0.46`, `run_short_ga=False` |
| 03 | Baseline Dashboard | [`01_Innovation_Epochs_and_Density_Classification/03_Epoch2_Block_Expander_rho0.54.png`](./01_Innovation_Epochs_and_Density_Classification/03_Epoch2_Block_Expander_rho0.54.png) | Epoch 2 (Asymmetric Block Expander, rho=0.54): expands toward ALL-1s | `checkpoint=Epoch 2 (Asymmetric Block Expander)`, `init_density=0.54`, `run_short_ga=False` |
| 04 | Baseline Dashboard | [`01_Innovation_Epochs_and_Density_Classification/04_Epoch2_Block_Expander_rho0.46_symmetry_bias.png`](./01_Innovation_Epochs_and_Density_Classification/04_Epoch2_Block_Expander_rho0.46_symmetry_bias.png) | Epoch 2 Failure Mode (rho=0.46): symmetry-breaking false positive | `checkpoint=Epoch 2 (Asymmetric Block Expander)`, `init_density=0.46`, `run_short_ga=False` |
| 05 | Baseline Dashboard | [`01_Innovation_Epochs_and_Density_Classification/05_Epoch3_4_GKL_Particle_Logic_rho0.54.png`](./01_Innovation_Epochs_and_Density_Classification/05_Epoch3_4_GKL_Particle_Logic_rho0.54.png) | Epoch 3/4 (GKL Particle Logic, rho=0.54): particle signals classify ALL-1s | `checkpoint=Epoch 3/4 (GKL)`, `init_density=0.54`, `run_short_ga=False` |
| 06 | Baseline Dashboard | [`01_Innovation_Epochs_and_Density_Classification/06_Epoch3_4_GKL_Particle_Logic_rho0.46.png`](./01_Innovation_Epochs_and_Density_Classification/06_Epoch3_4_GKL_Particle_Logic_rho0.46.png) | Epoch 3/4 (GKL Particle Logic, rho=0.46): particle signals classify ALL-0s | `checkpoint=Epoch 3/4 (GKL)`, `init_density=0.46`, `run_short_ga=False` |
| 07 | Baseline Dashboard | [`01_Innovation_Epochs_and_Density_Classification/07_Epoch3_4_GKL_Particle_Logic_rho0.70.png`](./01_Innovation_Epochs_and_Density_Classification/07_Epoch3_4_GKL_Particle_Logic_rho0.70.png) | Epoch 3/4 (GKL Particle Logic, rho=0.70): rapid high-density convergence | `checkpoint=Epoch 3/4 (GKL)`, `init_density=0.7`, `run_short_ga=False` |
| 08 | Baseline Dashboard | [`01_Innovation_Epochs_and_Density_Classification/08_Epoch3_4_GKL_Particle_Logic_rho0.30.png`](./01_Innovation_Epochs_and_Density_Classification/08_Epoch3_4_GKL_Particle_Logic_rho0.30.png) | Epoch 3/4 (GKL Particle Logic, rho=0.30): rapid low-density convergence | `checkpoint=Epoch 3/4 (GKL)`, `init_density=0.3`, `run_short_ga=False` |
| 09 | Baseline Dashboard | [`02_Live_GA_Evolution/09_Live_GA_12Gen_Trajectory_GKL_rho0.54.png`](./02_Live_GA_Evolution/09_Live_GA_12Gen_Trajectory_GKL_rho0.54.png) | Live 12-Generation GA Trajectory (Seed=7, N=49) | `checkpoint=Epoch 3/4 (GKL)`, `init_density=0.54`, `run_short_ga=True` |
| 10 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier1_Observation_Epochs_rho0.54.png`](./03_3Tier_Protocol/Tier1_Observation_Epochs_rho0.54.png) | Tier 1: Epoch 1 vs Epoch 2 vs GKL on rho=0.54 (N=149) with signal slope overlay | `init_density=0.54` |
| 11 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier2_Analysis_Epoch2_Symmetry_Bias.png`](./03_3Tier_Protocol/Tier2_Analysis_Epoch2_Symmetry_Bias.png) | Tier 2: Epoch 2 at rho=0.44 vs 0.56 exposing symmetry breaking & Land-Belew curve | `default` |
| 12 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier3_LiveGA_Schema_Theorem_Seed7.png`](./03_3Tier_Protocol/Tier3_LiveGA_Schema_Theorem_Seed7.png) | Tier 3: Live 12-Gen GA (Seed=7) & Holland Schema Theorem survival verification | `seed=7` |
| 13 | Real-World Demo | [`04_Real_World_Applications/App1_Protein_Directed_Evolution_NK.png`](./04_Real_World_Applications/App1_Protein_Directed_Evolution_NK.png) | App 4.1: Directed protein evolution on smooth (K=0) vs rugged (K=6) NK landscapes | `default` |
| 14 | Real-World Demo | [`04_Real_World_Applications/App2_AutoML_Neural_Architecture_Search.png`](./04_Real_World_Applications/App2_AutoML_Neural_Architecture_Search.png) | App 4.2: Discrete GA Neural Architecture Search (Accuracy vs FLOPs Pareto front) | `default` |
| 15 | Real-World Demo | [`04_Real_World_Applications/App3_Swarm_Robotics_Foraging.png`](./04_Real_World_Applications/App3_Swarm_Robotics_Foraging.png) | App 4.3: Decentralized 2D foraging swarm robotics rule evolution | `default` |

## Classroom Animated GIFs

- [`animations/01_gkl_density_classification_particles.gif`](./animations/01_gkl_density_classification_particles.gif)
- [`animations/02_live_ga_12gen_evolution.gif`](./animations/02_live_ga_12gen_evolution.gif)
- [`animations/03_swarm_robotics_foraging_evolution.gif`](./animations/03_swarm_robotics_foraging_evolution.gif)
