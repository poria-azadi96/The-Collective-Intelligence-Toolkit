# The Collective Intelligence & Info-Dynamics Toolkit

**Computational Research Workshop on Information Processing & Collective Intelligence in Complex Systems**  
**Instructor:** Poria Azadi (PhD Candidate in Bioinformatics, Institute of Biochemistry and Biophysics, University of Tehran)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/poria-azadi96/The-Collective-Intelligence-Toolkit/blob/main/notebooks/01_information_theory_and_pid.ipynb)

## Core Epistemological Framework
This repository accompanies the 12-hour intensive computational workshop addressing two foundational questions in decentralized complex systems:
1. **Generative Mechanisms:** How do local rules and decentralized interactions give rise to computation, adaptation, and collective intelligence?
2. **Observational Inference:** How can we extract, quantify, and statistically validate directed information flows from empirical time-series data?

## Quick Start (Unified Interactive Dashboard & Headless Export)

```bash
pip install -r requirements.txt

# Launch the unified interactive Streamlit laboratories
streamlit run app.py        # Modules 01–04 (Information, TE, CA, Evolutionary GA)
streamlit run app_part2.py  # Modules 05–08 (RBN/Immune, Physarum/Swarms, Markets, Validation)

# Run verification & smoke tests across Modules 01–08, 24 Tiers, and 24 Real-World Demos
python tests/verify_all.py
python tests/verify_all_part2.py
pytest -q

# Export all publication-grade PNG figures, classroom animated GIFs, CSVs, and summaries
python scripts/run_modules_01_04.py
python scripts/run_modules_05_08.py
```

## Workshop Modules & Interactive Colab Labs

| Module | Topic & Core Literature | Interactive Notebook | Classroom Outputs |
| :--- | :--- | :--- | :--- |
| **00 (Primer)** | Complex Systems Foundations, Probability & Lagged Series | [Open Primer](https://colab.research.google.com/github/poria-azadi96/The-Collective-Intelligence-Toolkit/blob/main/foundations/00_pre_workshop_primer.ipynb) | [`outputs/00_Pre_Workshop_Primer/`](./outputs/00_Pre_Workshop_Primer/summary.md) |
| **Module 01** | Information Theory, Nonlinearity, Landauer Bound & Williams–Beer PID | [Open Module 01](https://colab.research.google.com/github/poria-azadi96/The-Collective-Intelligence-Toolkit/blob/main/notebooks/01_information_theory_and_pid.ipynb) | [`outputs/01_Information_Theory_and_PID/`](./outputs/01_Information_Theory_and_PID/summary.md) |
| **Module 02** | Transfer Entropy, KSG Estimator, Wibral Delay Scan & BH-FDR Inference | [Open Module 02](https://colab.research.google.com/github/poria-azadi96/The-Collective-Intelligence-Toolkit/blob/main/notebooks/02_transfer_entropy_and_network_inference.ipynb) | [`outputs/02_Transfer_Entropy_and_Network_Inference/`](./outputs/02_Transfer_Entropy_and_Network_Inference/summary.md) |
| **Module 03** | Elementary Cellular Automata, 5-State Hanson–Crutchfield FST & Damage Spreading | [Open Module 03](https://colab.research.google.com/github/poria-azadi96/The-Collective-Intelligence-Toolkit/blob/main/notebooks/03_ca_and_particle_computation.ipynb) | [`outputs/03_Cellular_Automata_and_Particle_Computation/`](./outputs/03_Cellular_Automata_and_Particle_Computation/summary.md) |
| **Module 04** | Radius-3 CA, GKL Particle Logic, Holland GA & Schema Theorem | [Open Module 04](https://colab.research.google.com/github/poria-azadi96/The-Collective-Intelligence-Toolkit/blob/main/notebooks/04_evolutionary_computation_holland.ipynb) | [`outputs/04_Evolutionary_Computation_and_Edge_of_Chaos/`](./outputs/04_Evolutionary_Computation_and_Edge_of_Chaos/summary.md) |
| **Module 05** | Random Boolean Networks & Distributed Immune Selection (Kauffman; Greil & Drossel) | [Open Module 05](https://colab.research.google.com/github/poria-azadi96/The-Collective-Intelligence-Toolkit/blob/main/notebooks/05_rbn_and_immune_cognition.ipynb) | [`outputs/05_RBN_and_Immune_Cognition/`](./outputs/05_RBN_and_Immune_Cognition/summary.md) |
| **Module 06** | Adaptive Networks (*Physarum* Tokyo Rail), Contagion & Vicsek Flocking (Tero et al., 2010; Centola, 2010; Vicsek, 1995) | [Open Module 06](https://colab.research.google.com/github/poria-azadi96/The-Collective-Intelligence-Toolkit/blob/main/notebooks/06_adaptive_spatial_networks.ipynb) | [`outputs/06_Physarum_Contagion_and_Flocking/`](./outputs/06_Physarum_Contagion_and_Flocking/summary.md) |
| **Module 07** | Collective Wisdom, Bounded Confidence & El Farol Market (Arthur, 1994; Page; Hegselmann–Krause; Challet–Zhang) | [Open Module 07](https://colab.research.google.com/github/poria-azadi96/The-Collective-Intelligence-Toolkit/blob/main/notebooks/07_el_farol_and_opinion_dynamics.ipynb) | [`outputs/07_Collective_Wisdom_Opinion_and_El_Farol/`](./outputs/07_Collective_Wisdom_Opinion_and_El_Farol/summary.md) |
| **Module 08** | Computational Research Design, Audit Protocol, BH-FDR $T_{80}$ Calibration & Null Baselines | [Open Module 08](https://colab.research.google.com/github/poria-azadi96/The-Collective-Intelligence-Toolkit/blob/main/notebooks/08_research_design_lab.ipynb) | [`outputs/08_Research_Design_and_Validation_Lab/`](./outputs/08_Research_Design_and_Validation_Lab/summary.md) |

## Repository Architecture
- `app.py` & `app_part2.py`: Unified Streamlit interactive laboratories with sidebar navigation across Modules 01–04 and Modules 05–08, 3-Tier protocols, real-world case studies, interactive research brief builder, and animated GIF galleries.
- `core/`: Modular, deterministic Python engines across Modules 01–08 (`info_theory_pid.py`, `transfer_entropy.py`, `cellular_automata.py`, `evolutionary_ga.py`, `rbn_immune.py`, `adaptive_spatial_networks.py`, `collective_markets.py`, `research_validation.py`, `real_world_demos.py`, `real_world_demos_part2.py`, `plotting_utils.py`).
- `src/`: Package exports (`info_dynamics`, `automata_rbn`, `metaheuristics`, `bio_swarms`, `socio_econ`).
- `notebooks/`: Interactive Jupyter / Google Colab labs (`01`–`08`, including `06_adaptive_spatial_networks.ipynb` and `06_physarum_and_contagion.ipynb`) with `ipywidgets`, 3-Tier experimental protocols, and domain applications.
- `outputs/`: Publication-grade classroom figures (`240 DPI` PNGs), looping classroom animations (`*.gif`), structured `summary.md` reports, `run_log.csv`, and JSON metrics.
- `scripts/`: Reproducible simulation runners (`run_modules_01_04.py`, `run_modules_05_08.py`, `export_classroom_outputs.py`, `build_notebooks_01_04.py`, `build_notebooks_05_08.py`).
- `tests/`: Automated mathematical & smoke verification suites (`verify_all.py`, `verify_all_part2.py`, `test_modules_01_04.py`, `test_modules_05_08.py`).
- `docs/`: Standardized 8-question scientific paper audit protocol and research brief template.

## All 24 Real-World Applications (Modules 01–08, 3 per Module)
- **Module 01:** (1) Cortical Population Coding (Synergistic Coincidence vs. Redundant Firing), (2) Genomics & Pure SNP Epistasis Beyond GWAS, (3) Thermodynamic Landauer Dissipation vs. Reversible Fredkin/Toffoli Computing.
- **Module 02:** (1) 4-Region Effective Brain Connectivity (`THAL`, `PFC`, `PAR`, `MOT`), (2) Multi-Asset Financial Volatility Contagion & Systemic Risk Hubs, (3) Hill-Function Transcriptional Gene Regulatory Network Reconstruction.
- **Module 03:** (1) 2D Excitable Media (Belousov–Zhabotinsky Chemical Spiral Waves & Cardiac Reentry), (2) Rule 184 & Nagel–Schreckenberg Traffic Jams, (3) Decentralized Neuromorphic Glider Parity Logic Array.
- **Module 04:** (1) Directed Protein Evolution on Kauffman $NK$ Rugged Landscapes, (2) AutoML Discrete Neural Architecture Search (Accuracy vs. FLOPs Pareto Front), (3) Decentralized Evolutionary Swarm Robotics Foraging.
- **Module 05:** (1) Cancer Systems Biology Epigenetic Attractor Distortion & Basin Hijacking, (2) Immunotherapy TCR Affinity Tolerance vs. Kinetic Proofreading Specificity, (3) Canalizing Gene Regulatory Functions Stabilizing Asynchronous GRNs.
- **Module 06:** (1) Self-Healing Transit & Power Grid Rerouting Under Live Disruption, (2) Public Health Vaccine Campaign Simple vs. Complex Contagion on Clustered Communities, (3) Autonomous Drone Swarm Corridor Flocking with Obstacle Avoidance.
- **Module 07:** (1) Financial Market Microstructure Liquidity Droughts & Endogenous Flash Crashes, (2) Diversity-Weighted Delphi Forecasting Outperforming Correlated Groupthink, (3) Multi-Agent AI / LLM Debate Consensus vs. Premature Sycophantic Collapse.
- **Module 08:** (1) Q1 Journal Peer-Review Audit & Mechanistic Statistical Stress-Test Suite, (2) Industrial R&D Monte Carlo Power & BH-FDR Sample-Size Calibration ($T_{80}$), (3) Cross-Module Grand Synthesis & Reproducibility Report Across Modules 01–08.
