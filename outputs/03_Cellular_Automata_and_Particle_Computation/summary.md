# Module 03: Cellular Automata & Emergent Particle Computation

**Source Notebook:** `notebooks/03_ca_and_particle_computation.ipynb`

1D ECA, Wolfram Classes I-IV, Langton lambda, Hanson & Crutchfield 5-state FST domain filter, Rule 110 period-14 ether filter, XOR damage spreading, 3-Tier protocols, and 3 domain applications.

## Generated Classroom Figures (Baselines, 3-Tier Protocol & Real-World Applications)

| # | Category | File | Pedagogical Focus | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| 01 | Baseline Dashboard | [`01_Rule54_Domain_Filter_and_Collisions/01_Rule54_collision_setup_filtered.png`](./01_Rule54_Domain_Filter_and_Collisions/01_Rule54_collision_setup_filtered.png) | Rule 54 Collision Setup + 5-State Hanson-Crutchfield FST Domain Filter | `rule_number=54`, `init_type=collision_setup`, `show_filter=True` |
| 02 | Baseline Dashboard | [`01_Rule54_Domain_Filter_and_Collisions/02_Rule54_random_init_filtered.png`](./01_Rule54_Domain_Filter_and_Collisions/02_Rule54_random_init_filtered.png) | Rule 54 Random Init + FST Domain Filter (spontaneous ether & collisions) | `rule_number=54`, `init_type=random`, `show_filter=True` |
| 03 | Baseline Dashboard | [`01_Rule54_Domain_Filter_and_Collisions/03_Rule54_collision_setup_perturbation.png`](./01_Rule54_Domain_Filter_and_Collisions/03_Rule54_collision_setup_perturbation.png) | Rule 54 Collision Setup + Single-Bit Perturbation Damage Cone | `rule_number=54`, `init_type=collision_setup`, `show_filter=False` |
| 04 | Baseline Dashboard | [`01_Rule54_Domain_Filter_and_Collisions/04_Rule54_single_seed_perturbation.png`](./01_Rule54_Domain_Filter_and_Collisions/04_Rule54_single_seed_perturbation.png) | Rule 54 Single Seed + Perturbation Cone | `rule_number=54`, `init_type=single_seed`, `show_filter=False` |
| 05 | Baseline Dashboard | [`02_Wolfram_Classes_Comparison/05_Rule110_Turing_Universal_random.png`](./02_Wolfram_Classes_Comparison/05_Rule110_Turing_Universal_random.png) | Rule 110 (Class IV Universal) Random Init + Period-14 Ether Filter | `rule_number=110`, `init_type=random`, `show_filter=True` |
| 06 | Baseline Dashboard | [`02_Wolfram_Classes_Comparison/06_Rule110_Turing_Universal_single_seed.png`](./02_Wolfram_Classes_Comparison/06_Rule110_Turing_Universal_single_seed.png) | Rule 110 (Class IV) Asymmetric Evolution from Single Seed | `rule_number=110`, `init_type=single_seed`, `show_filter=False` |
| 07 | Baseline Dashboard | [`02_Wolfram_Classes_Comparison/07_Rule30_Chaotic_single_seed.png`](./02_Wolfram_Classes_Comparison/07_Rule30_Chaotic_single_seed.png) | Rule 30 (Class III Chaotic) Single Seed + Light-Cone Damage | `rule_number=30`, `init_type=single_seed`, `show_filter=False` |
| 08 | Baseline Dashboard | [`02_Wolfram_Classes_Comparison/08_Rule30_Chaotic_random.png`](./02_Wolfram_Classes_Comparison/08_Rule30_Chaotic_random.png) | Rule 30 (Class III Chaotic) Random Init + Rapid Butterfly Effect | `rule_number=30`, `init_type=random`, `show_filter=False` |
| 09 | Baseline Dashboard | [`02_Wolfram_Classes_Comparison/09_Rule90_Fractal_Linear_single_seed.png`](./02_Wolfram_Classes_Comparison/09_Rule90_Fractal_Linear_single_seed.png) | Rule 90 (Additive Sierpinski Fractal) Single Seed + XOR Cone | `rule_number=90`, `init_type=single_seed`, `show_filter=False` |
| 10 | Baseline Dashboard | [`02_Wolfram_Classes_Comparison/10_Rule90_Fractal_Linear_random.png`](./02_Wolfram_Classes_Comparison/10_Rule90_Fractal_Linear_random.png) | Rule 90 (Additive Linear) Random Init + Deterministic XOR Cone | `rule_number=90`, `init_type=random`, `show_filter=False` |
| 11 | Baseline Dashboard | [`02_Wolfram_Classes_Comparison/11_Rule108_ClassII_Periodic.png`](./02_Wolfram_Classes_Comparison/11_Rule108_ClassII_Periodic.png) | Rule 108 (Class II Periodic / Localized Structures) | `rule_number=108`, `init_type=random`, `show_filter=False` |
| 12 | Baseline Dashboard | [`02_Wolfram_Classes_Comparison/12_Rule254_ClassI_Homogeneous.png`](./02_Wolfram_Classes_Comparison/12_Rule254_ClassI_Homogeneous.png) | Rule 254 (Class I Homogeneous Fixed Point) | `rule_number=254`, `init_type=random`, `show_filter=False` |
| 13 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier1_Observation_Rule54_FST_Toggle.png`](./03_3Tier_Protocol/Tier1_Observation_Rule54_FST_Toggle.png) | Tier 1: Rule 54 FST Filter OFF vs ON exposing gamma+/gamma- gliders & collision vertices | `default` |
| 14 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier2_Analysis_Rule54_Density_Slopes.png`](./03_3Tier_Protocol/Tier2_Analysis_Rule54_Density_Slopes.png) | Tier 2: Rule 54 random-init particle density decay & velocity slope verification | `default` |
| 15 | 3-Tier Protocol | [`03_3Tier_Protocol/Tier3_DamageSpreading_90_30_110.png`](./03_3Tier_Protocol/Tier3_DamageSpreading_90_30_110.png) | Tier 3: XOR damage spreading cones & H(t) across Rule 90, Rule 30, and Rule 110 | `default` |
| 16 | Real-World Demo | [`04_Real_World_Applications/App1_Excitable_BZ_Spiral_Waves.png`](./04_Real_World_Applications/App1_Excitable_BZ_Spiral_Waves.png) | App 3.1: 2D Belousov-Zhabotinsky chemical spiral waves & cardiac reentry | `default` |
| 17 | Real-World Demo | [`04_Real_World_Applications/App2_Traffic_Rule184_Nagel_Schreckenberg.png`](./04_Real_World_Applications/App2_Traffic_Rule184_Nagel_Schreckenberg.png) | App 3.2: Rule 184 & Nagel-Schreckenberg fundamental diagram & phantom jams | `default` |
| 18 | Real-World Demo | [`04_Real_World_Applications/App3_Neuromorphic_Glider_Parity_Circuit.png`](./04_Real_World_Applications/App3_Neuromorphic_Glider_Parity_Circuit.png) | App 3.3: Decentralized clock-driven glider-collision XOR parity circuit | `default` |

## Classroom Animated GIFs

- [`animations/01_rule54_domain_filter_and_collisions.gif`](./animations/01_rule54_domain_filter_and_collisions.gif)
- [`animations/02_wolfram_classes_damage_cone.gif`](./animations/02_wolfram_classes_damage_cone.gif)
- [`animations/03_belousov_zhabotinsky_spiral_waves.gif`](./animations/03_belousov_zhabotinsky_spiral_waves.gif)
- [`animations/04_traffic_rule184_nagel_schreckenberg_jams.gif`](./animations/04_traffic_rule184_nagel_schreckenberg_jams.gif)
- [`animations/05_neuromorphic_glider_parity_circuit.gif`](./animations/05_neuromorphic_glider_parity_circuit.gif)
