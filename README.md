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

# Launch the unified interactive Streamlit laboratory (Modules 01-04)
streamlit run app.py

# Run verification & smoke tests across Modules 01-04, 12 Tiers, and 12 Real-World Demos
python tests/verify_all.py
pytest -q

# Export all publication-grade PNG figures, classroom animated GIFs, CSVs, and summaries
python scripts/run_modules_01_04.py
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
| **Module 06** | Adaptive Networks (*Physarum* Tokyo Rail) & Contagion (Tero et al., 2010; Vicsek) | [Open Module 06](https://colab.research.google.com/github/poria-azadi96/The-Collective-Intelligence-Toolkit/blob/main/notebooks/06_physarum_and_contagion.ipynb) | [`outputs/06_Physarum_Contagion_and_Flocking/`](./outputs/06_Physarum_Contagion_and_Flocking/summary.md) |
| **Module 07** | Collective Wisdom, Bounded Confidence & El Farol Market (Arthur, 1994; Page) | [Open Module 07](https://colab.research.google.com/github/poria-azadi96/The-Collective-Intelligence-Toolkit/blob/main/notebooks/07_el_farol_and_opinion_dynamics.ipynb) | [`outputs/07_Collective_Wisdom_Opinion_and_El_Farol/`](./outputs/07_Collective_Wisdom_Opinion_and_El_Farol/summary.md) |
| **Module 08** | Computational Research Design & Validation Lab | [Open Module 08](https://colab.research.google.com/github/poria-azadi96/The-Collective-Intelligence-Toolkit/blob/main/notebooks/08_research_design_lab.ipynb) | [`outputs/08_Research_Design_and_Validation_Lab/`](./outputs/08_Research_Design_and_Validation_Lab/summary.md) |

## Repository Architecture
- `app.py`: Unified Streamlit interactive laboratory with sidebar navigation across Modules 01–04, 3-Tier protocols, real-world case studies, and animated GIF galleries.
- `core/`: Modular, deterministic Python engines for Modules 01–04 (`info_theory_pid.py`, `transfer_entropy.py`, `cellular_automata.py`, `evolutionary_ga.py`, `real_world_demos.py`, `plotting_utils.py`).
- `src/`: Package exports (`info_dynamics`, `automata_rbn`, `metaheuristics`, `bio_swarms`, `socio_econ`).
- `notebooks/`: Interactive Jupyter / Google Colab labs (`01`–`08`) with `ipywidgets`, 3-Tier experimental protocols, and domain applications.
- `outputs/`: Publication-grade classroom figures (`240 DPI` PNGs), looping classroom animations (`*.gif`), structured `summary.md` reports, `run_log.csv`, and JSON metrics.
- `scripts/`: Reproducible simulation runners (`run_modules_01_04.py`, `export_classroom_outputs.py`, `build_notebooks_01_04.py`).
- `tests/`: Automated mathematical & smoke verification suite (`verify_all.py`, `test_modules_01_04.py`).
- `docs/`: Standardized 8-question scientific paper audit protocol and research brief template.

## Modules 01–04 Real-World Applications (12 Executable Case Studies)
- **Module 01:** (1) Cortical Population Coding (Synergistic Coincidence vs. Redundant Firing), (2) Genomics & Pure SNP Epistasis Beyond GWAS, (3) Thermodynamic Landauer Dissipation vs. Reversible Fredkin/Toffoli Computing.
- **Module 02:** (1) 4-Region Effective Brain Connectivity (`THAL`, `PFC`, `PAR`, `MOT`), (2) Multi-Asset Financial Volatility Contagion & Systemic Risk Hubs, (3) Hill-Function Transcriptional Gene Regulatory Network Reconstruction.
- **Module 03:** (1) 2D Excitable Media (Belousov–Zhabotinsky Chemical Spiral Waves & Cardiac Reentry), (2) Rule 184 & Nagel–Schreckenberg Traffic Jams, (3) Decentralized Neuromorphic Glider Parity Logic Array.
- **Module 04:** (1) Directed Protein Evolution on Kauffman $NK$ Rugged Landscapes, (2) AutoML Discrete Neural Architecture Search (Accuracy vs. FLOPs Pareto Front), (3) Decentralized Evolutionary Swarm Robotics Foraging.
