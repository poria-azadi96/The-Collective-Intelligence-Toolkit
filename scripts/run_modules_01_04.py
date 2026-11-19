"""Run and export complete classroom outputs, 3-Tier protocols, 12 real-world demos & GIFs for Modules 01-04."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import (
    cellular_automata as m03,
    evolutionary_ga as m04,
    info_theory_pid as m01,
    real_world_demos as demos,
    transfer_entropy as m02,
)
from core.plotting_utils import save_figure

OUTPUTS_ROOT = REPO_ROOT / "outputs"


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items() if str(k) not in ("frames", "snaps_gen1", "snaps_evolved")}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _record_run(
    records: list[dict[str, Any]],
    mod_dir: Path,
    rel_path: str,
    category: str,
    focus: str,
    fn: Callable[..., tuple[plt.Figure, Any]],
    kwargs: dict[str, Any] | None = None,
    dpi: int = 240,
) -> dict[str, Any]:
    kw = kwargs or {}
    fig, stats = fn(**kw)
    out_file = mod_dir / rel_path
    save_figure(fig, out_file, dpi=dpi)
    safe_stats = _json_safe(stats)
    rec = {
        "file": rel_path,
        "category": category,
        "focus": focus,
        "parameters": kw,
        "metrics": safe_stats,
    }
    records.append(rec)
    print(f"  [PNG] {mod_dir.name}/{rel_path}")
    return safe_stats


def _write_module_artifacts(
    mod_dir: Path,
    title: str,
    notebook_rel: str,
    description: str,
    records: list[dict[str, Any]],
    gif_paths: list[Path],
) -> None:
    lines = [
        f"# {title}",
        "",
        f"**Source Notebook:** `{notebook_rel}`",
        "",
        description,
        "",
        "## Generated Classroom Figures (Baselines, 3-Tier Protocol & Real-World Applications)",
        "",
        "| # | Category | File | Pedagogical Focus | Parameters |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for idx, rec in enumerate(records, start=1):
        p_str = ", ".join(f"`{k}={v}`" for k, v in rec["parameters"].items()) or "`default`"
        lines.append(f"| {idx:02d} | {rec['category']} | [`{rec['file']}`](./{rec['file']}) | {rec['focus']} | {p_str} |")

    if gif_paths:
        lines.extend(["", "## Classroom Animated GIFs", ""])
        for gp in gif_paths:
            rel_g = gp.relative_to(mod_dir).as_posix()
            lines.append(f"- [`{rel_g}`](./{rel_g})")

    (mod_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (mod_dir / "run_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "category", "file", "focus", "parameters"])
        writer.writeheader()
        for idx, rec in enumerate(records, start=1):
            writer.writerow(
                {
                    "index": idx,
                    "category": rec["category"],
                    "file": rec["file"],
                    "focus": rec["focus"],
                    "parameters": json.dumps(rec["parameters"], ensure_ascii=False),
                }
            )

    (mod_dir / "metrics.json").write_text(
        json.dumps(
            {
                "module": mod_dir.name,
                "title": title,
                "records": records,
                "gifs": [g.relative_to(mod_dir).as_posix() for g in gif_paths],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def export_module_01() -> tuple[list[dict[str, Any]], list[Path]]:
    print("\n=== Running Module 01: Information Theory & PID ===")
    mod_dir = OUTPUTS_ROOT / "01_Information_Theory_and_PID"
    records: list[dict[str, Any]] = []

    baselines = [
        ("01_Logic_Gates_Baseline/01_XOR_pure_synergy.png", "XOR Gate: zero pairwise MI yet 1.0 bit Pure Synergy", {"gate_type": "XOR", "input_corr": 0.0, "sigma": 0.0, "coupling": 1.0}),
        ("01_Logic_Gates_Baseline/02_AND_redundancy_and_synergy.png", "AND Gate: coexistence of Redundancy (0.311 b) & Synergy (0.500 b)", {"gate_type": "AND", "input_corr": 0.0, "sigma": 0.0, "coupling": 1.0}),
        ("01_Logic_Gates_Baseline/03_OR_redundancy_and_synergy.png", "OR Gate: symmetric coexistence of Redundancy & Synergy", {"gate_type": "OR", "input_corr": 0.0, "sigma": 0.0, "coupling": 1.0}),
        ("01_Logic_Gates_Baseline/04_COPY_X1_unique_information.png", "COPY Gate: 1.0 bit Unique Information in X1", {"gate_type": "COPY", "input_corr": 0.0, "sigma": 0.0, "coupling": 1.0}),
        ("01_Logic_Gates_Baseline/05_IDENTICAL_pure_redundancy.png", "IDENTICAL Gate (X1=X2=Y): 1.0 bit Pure Redundancy", {"gate_type": "IDENTICAL", "input_corr": 0.0, "sigma": 0.0, "coupling": 1.0}),
        ("01_Logic_Gates_Baseline/06_Continuous_Gaussian_MMI_PID.png", "Continuous Gaussian MMI-PID (rho=0.25, sigma=0.30, c=1.0)", {"gate_type": "Continuous_Gaussian", "input_corr": 0.25, "sigma": 0.30, "coupling": 1.0}),
        ("02_Correlation_and_Noise_Effects/05_COPY_X1_correlated_inputs_corr0.70.png", "Correlated COPY (rho=0.70): correlation converts Unique info into Redundancy", {"gate_type": "COPY", "input_corr": 0.70, "sigma": 0.0, "coupling": 1.0}),
        ("02_Correlation_and_Noise_Effects/06_XOR_correlated_inputs_corr0.70.png", "Correlated XOR (rho=0.70): reduced output entropy H(Y)", {"gate_type": "XOR", "input_corr": 0.70, "sigma": 0.0, "coupling": 1.0}),
        ("02_Correlation_and_Noise_Effects/07_XOR_anticorrelated_inputs_corr-0.70.png", "Anti-correlated XOR (rho=-0.70): symmetric synergy shift", {"gate_type": "XOR", "input_corr": -0.70, "sigma": 0.0, "coupling": 1.0}),
        ("02_Correlation_and_Noise_Effects/08_XOR_noisy_channel_noise0.20.png", "Noisy XOR (noise_prob=0.20): channel noise degrades Synergy", {"gate_type": "XOR", "input_corr": 0.0, "noise_prob": 0.20}),
        ("02_Correlation_and_Noise_Effects/09_AND_noisy_channel_noise0.15.png", "Noisy AND (noise_prob=0.15): noise attenuates Redundancy & Synergy", {"gate_type": "AND", "input_corr": 0.0, "noise_prob": 0.15}),
    ]
    for rel, focus, kw in baselines:
        _record_run(records, mod_dir, rel, "Baseline Dashboard", focus, m01.create_pid_dashboard_figure, kw)

    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier1_Observation_XOR_Calibration.png", "3-Tier Protocol", "Tier 1: XOR 100% Synergy (sigma=0.1, c=1.0), 4-gate calibration & circular nonlinearity", m01.run_module01_tier1, {"sigma": 0.1, "coupling": 1.0})
    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier2_Analysis_AND_Crossover.png", "3-Tier Protocol", "Tier 2: Correlation sweep (0->0.95) on AND showing R vs S & II=R-S zero-crossing", m01.run_module01_tier2, {"gate_type": "AND"})
    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier3_StressTest_Landauer_Noise.png", "3-Tier Protocol", "Tier 3: Channel noise sweep (0->1.0) across all 4 gates & Landauer heat dissipation", m01.run_module01_tier3, {"temperature_k": 300.0})

    _record_run(records, mod_dir, "04_Real_World_Applications/App1_Computational_Neuroscience.png", "Real-World Demo", "App 1.1: Cortical coincidence detection (Synergy) vs redundant population firing", demos.demo_m01_computational_neuroscience)
    _record_run(records, mod_dir, "04_Real_World_Applications/App2_Genomics_Epistasis.png", "Real-World Demo", "App 1.2: Pure epistatic SNP interaction (Y=X1^X2+noise) hidden from linear GWAS", demos.demo_m01_genomics_epistasis)
    _record_run(records, mod_dir, "04_Real_World_Applications/App3_Thermodynamic_Reversible_Computing.png", "Real-World Demo", "App 1.3: Irreversible CMOS reset vs reversible Fredkin/Toffoli dissipation", demos.demo_m01_thermodynamic_reversible_computing)

    gifs = m01.generate_module01_gifs(mod_dir)
    for gp in gifs:
        print(f"  [GIF] {mod_dir.name}/animations/{gp.name}")

    _write_module_artifacts(
        mod_dir=mod_dir,
        title="Module 01: Information Theory, Nonlinearity & Partial Information Decomposition (PID)",
        notebook_rel="notebooks/01_information_theory_and_pid.ipynb",
        description="Shannon metrics, Circular Nonlinearity proof, Landauer's thermodynamic bound, Williams & Beer 4-atom PID, 3-Tier protocols, and 3 domain applications.",
        records=records,
        gif_paths=gifs,
    )
    return records, gifs


def export_module_02() -> tuple[list[dict[str, Any]], list[Path]]:
    print("\n=== Running Module 02: Transfer Entropy & Network Inference ===")
    mod_dir = OUTPUTS_ROOT / "02_Transfer_Entropy_and_Network_Inference"
    records: list[dict[str, Any]] = []

    baselines = [
        ("01_Causal_Motifs/01_Direct_Coupling_c0.70_n450.png", "Direct Coupling (X->Y lag u=2): peak at u=2 survives conditioning", {"motif_type": "Direct_Coupling", "coupling": 0.70, "n_samples": 450, "n_surrogates": 60}),
        ("01_Causal_Motifs/02_Indirect_Cascade_c0.70_n450.png", "Indirect Cascade (X->Z->Y): Conditional TE | Z filters indirect link", {"motif_type": "Indirect_Cascade", "coupling": 0.70, "n_samples": 450, "n_surrogates": 60}),
        ("01_Causal_Motifs/03_Common_Driver_c0.70_n450.png", "Common Driver (Z->X,Y): Conditional TE | Z removes spurious bivariate link", {"motif_type": "Common_Driver", "coupling": 0.70, "n_samples": 450, "n_surrogates": 60}),
        ("01_Causal_Motifs/04_Unobserved_Confounder_c0.70_n450.png", "Unobserved Confounder (hidden U->X,Y): irreducible observational false positive", {"motif_type": "Unobserved_Confounder", "coupling": 0.70, "n_samples": 450, "n_surrogates": 60}),
        ("02_Sensitivity_Coupling_and_SampleSize/05_Direct_Coupling_weak_c0.30_n450.png", "Weak Direct Coupling (c=0.30): reduced TE near surrogate null", {"motif_type": "Direct_Coupling", "coupling": 0.30, "n_samples": 450, "n_surrogates": 60}),
        ("02_Sensitivity_Coupling_and_SampleSize/06_Direct_Coupling_strong_c0.85_n450.png", "Strong Direct Coupling (c=0.85): sharp lag-2 peak well above null", {"motif_type": "Direct_Coupling", "coupling": 0.85, "n_samples": 450, "n_surrogates": 60}),
        ("02_Sensitivity_Coupling_and_SampleSize/07_Common_Driver_small_sample_n250.png", "Small Sample Regime (n=250): higher surrogate threshold variance", {"motif_type": "Common_Driver", "coupling": 0.70, "n_samples": 250, "n_surrogates": 60}),
        ("02_Sensitivity_Coupling_and_SampleSize/08_Direct_Coupling_large_sample_n800.png", "Large Sample Regime (n=800): tight surrogate cutoff & clean lag-2 peak", {"motif_type": "Direct_Coupling", "coupling": 0.70, "n_samples": 800, "n_surrogates": 60}),
    ]
    for rel, focus, kw in baselines:
        _record_run(records, mod_dir, rel, "Baseline Dashboard", focus, m02.create_te_dashboard_figure, kw)

    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier1_Observation_Confounder_Cascade.png", "3-Tier Protocol", "Tier 1: Common_Driver & Indirect_Cascade filtered by Conditional TE | Z", m02.run_module02_tier1, {"n_surrogates": 70})
    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier2_Analysis_Wibral_Delay_Tau2.png", "3-Tier Protocol", "Tier 2: Wibral scan u=1..5 recovers tau=2 + AIS+TE chain rule", m02.run_module02_tier2, {"n_surrogates": 70})
    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier3_StressTest_Unobserved_Confounder.png", "3-Tier Protocol", "Tier 3: Unobserved Confounder sample sweep (250->800) shows irreducible false positives", m02.run_module02_tier3, {"n_surrogates": 70})

    _record_run(records, mod_dir, "04_Real_World_Applications/App1_Effective_Brain_Connectivity.png", "Real-World Demo", "App 2.1: 4-region effective brain connectivity (THAL, PFC, PAR, MOT) via Cond TE + BH-FDR", demos.demo_m02_brain_connectivity)
    _record_run(records, mod_dir, "04_Real_World_Applications/App2_Finance_Volatility_Contagion.png", "Real-World Demo", "App 2.2: Multi-asset nonlinear volatility spillover & systemic risk hub discovery", demos.demo_m02_finance_contagion)
    _record_run(records, mod_dir, "04_Real_World_Applications/App3_Systems_Biology_Gene_Network.png", "Real-World Demo", "App 2.3: Hill-function transcriptional cascade reconstruction (G1->G2->G3, G1-|G4)", demos.demo_m02_gene_regulatory_network)

    gifs = m02.generate_module02_gifs(mod_dir)
    for gp in gifs:
        print(f"  [GIF] {mod_dir.name}/animations/{gp.name}")

    _write_module_artifacts(
        mod_dir=mod_dir,
        title="Module 02: Transfer Entropy, Delay Reconstruction & Network Inference",
        notebook_rel="notebooks/02_transfer_entropy_and_network_inference.ipynb",
        description="Schreiber Transfer Entropy, Frenzel-Pompe / KSG k-NN estimator, Wibral interaction delay scan, circular surrogates, BH-FDR network recovery, 3-Tier protocols, and 3 domain applications.",
        records=records,
        gif_paths=gifs,
    )
    return records, gifs


def export_module_03() -> tuple[list[dict[str, Any]], list[Path]]:
    print("\n=== Running Module 03: Cellular Automata & Particle Computation ===")
    mod_dir = OUTPUTS_ROOT / "03_Cellular_Automata_and_Particle_Computation"
    records: list[dict[str, Any]] = []

    baselines = [
        ("01_Rule54_Domain_Filter_and_Collisions/01_Rule54_collision_setup_filtered.png", "Rule 54 Collision Setup + 5-State Hanson-Crutchfield FST Domain Filter", {"rule_number": 54, "init_type": "collision_setup", "show_filter": True}),
        ("01_Rule54_Domain_Filter_and_Collisions/02_Rule54_random_init_filtered.png", "Rule 54 Random Init + FST Domain Filter (spontaneous ether & collisions)", {"rule_number": 54, "init_type": "random", "show_filter": True}),
        ("01_Rule54_Domain_Filter_and_Collisions/03_Rule54_collision_setup_perturbation.png", "Rule 54 Collision Setup + Single-Bit Perturbation Damage Cone", {"rule_number": 54, "init_type": "collision_setup", "show_filter": False}),
        ("01_Rule54_Domain_Filter_and_Collisions/04_Rule54_single_seed_perturbation.png", "Rule 54 Single Seed + Perturbation Cone", {"rule_number": 54, "init_type": "single_seed", "show_filter": False}),
        ("02_Wolfram_Classes_Comparison/05_Rule110_Turing_Universal_random.png", "Rule 110 (Class IV Universal) Random Init + Period-14 Ether Filter", {"rule_number": 110, "init_type": "random", "show_filter": True}),
        ("02_Wolfram_Classes_Comparison/06_Rule110_Turing_Universal_single_seed.png", "Rule 110 (Class IV) Asymmetric Evolution from Single Seed", {"rule_number": 110, "init_type": "single_seed", "show_filter": False}),
        ("02_Wolfram_Classes_Comparison/07_Rule30_Chaotic_single_seed.png", "Rule 30 (Class III Chaotic) Single Seed + Light-Cone Damage", {"rule_number": 30, "init_type": "single_seed", "show_filter": False}),
        ("02_Wolfram_Classes_Comparison/08_Rule30_Chaotic_random.png", "Rule 30 (Class III Chaotic) Random Init + Rapid Butterfly Effect", {"rule_number": 30, "init_type": "random", "show_filter": False}),
        ("02_Wolfram_Classes_Comparison/09_Rule90_Fractal_Linear_single_seed.png", "Rule 90 (Additive Sierpinski Fractal) Single Seed + XOR Cone", {"rule_number": 90, "init_type": "single_seed", "show_filter": False}),
        ("02_Wolfram_Classes_Comparison/10_Rule90_Fractal_Linear_random.png", "Rule 90 (Additive Linear) Random Init + Deterministic XOR Cone", {"rule_number": 90, "init_type": "random", "show_filter": False}),
        ("02_Wolfram_Classes_Comparison/11_Rule108_ClassII_Periodic.png", "Rule 108 (Class II Periodic / Localized Structures)", {"rule_number": 108, "init_type": "random", "show_filter": False}),
        ("02_Wolfram_Classes_Comparison/12_Rule254_ClassI_Homogeneous.png", "Rule 254 (Class I Homogeneous Fixed Point)", {"rule_number": 254, "init_type": "random", "show_filter": False}),
    ]
    for rel, focus, kw in baselines:
        _record_run(records, mod_dir, rel, "Baseline Dashboard", focus, m03.create_ca_dashboard_figure, kw)

    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier1_Observation_Rule54_FST_Toggle.png", "3-Tier Protocol", "Tier 1: Rule 54 FST Filter OFF vs ON exposing gamma+/gamma- gliders & collision vertices", m03.run_module03_tier1)
    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier2_Analysis_Rule54_Density_Slopes.png", "3-Tier Protocol", "Tier 2: Rule 54 random-init particle density decay & velocity slope verification", m03.run_module03_tier2)
    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier3_DamageSpreading_90_30_110.png", "3-Tier Protocol", "Tier 3: XOR damage spreading cones & H(t) across Rule 90, Rule 30, and Rule 110", m03.run_module03_tier3)

    _record_run(records, mod_dir, "04_Real_World_Applications/App1_Excitable_BZ_Spiral_Waves.png", "Real-World Demo", "App 3.1: 2D Belousov-Zhabotinsky chemical spiral waves & cardiac reentry", demos.demo_m03_excitable_bz_spiral)
    _record_run(records, mod_dir, "04_Real_World_Applications/App2_Traffic_Rule184_Nagel_Schreckenberg.png", "Real-World Demo", "App 3.2: Rule 184 & Nagel-Schreckenberg fundamental diagram & phantom jams", demos.demo_m03_traffic_flow_rule184)
    _record_run(records, mod_dir, "04_Real_World_Applications/App3_Neuromorphic_Glider_Parity_Circuit.png", "Real-World Demo", "App 3.3: Decentralized clock-driven glider-collision XOR parity circuit", demos.demo_m03_neuromorphic_glider_logic)

    gifs = m03.generate_module03_gifs(mod_dir)
    for gp in gifs:
        print(f"  [GIF] {mod_dir.name}/animations/{gp.name}")

    _write_module_artifacts(
        mod_dir=mod_dir,
        title="Module 03: Cellular Automata & Emergent Particle Computation",
        notebook_rel="notebooks/03_ca_and_particle_computation.ipynb",
        description="1D ECA, Wolfram Classes I-IV, Langton lambda, Hanson & Crutchfield 5-state FST domain filter, Rule 110 period-14 ether filter, XOR damage spreading, 3-Tier protocols, and 3 domain applications.",
        records=records,
        gif_paths=gifs,
    )
    return records, gifs


def export_module_04() -> tuple[list[dict[str, Any]], list[Path]]:
    print("\n=== Running Module 04: Evolutionary Computation & Edge of Chaos ===")
    mod_dir = OUTPUTS_ROOT / "04_Evolutionary_Computation_and_Edge_of_Chaos"
    records: list[dict[str, Any]] = []

    baselines = [
        ("01_Innovation_Epochs_and_Density_Classification/01_Epoch1_Naive_Majority_rho0.54.png", "Epoch 1 (Naive Majority, rho=0.54): frozen domain walls stall", {"checkpoint": "Epoch 1", "init_density": 0.54, "run_short_ga": False}),
        ("01_Innovation_Epochs_and_Density_Classification/02_Epoch1_Naive_Majority_rho0.46.png", "Epoch 1 (Naive Majority, rho=0.46): symmetric frozen stalling", {"checkpoint": "Epoch 1", "init_density": 0.46, "run_short_ga": False}),
        ("01_Innovation_Epochs_and_Density_Classification/03_Epoch2_Block_Expander_rho0.54.png", "Epoch 2 (Asymmetric Block Expander, rho=0.54): expands toward ALL-1s", {"checkpoint": "Epoch 2 (Asymmetric Block Expander)", "init_density": 0.54, "run_short_ga": False}),
        ("01_Innovation_Epochs_and_Density_Classification/04_Epoch2_Block_Expander_rho0.46_symmetry_bias.png", "Epoch 2 Failure Mode (rho=0.46): symmetry-breaking false positive", {"checkpoint": "Epoch 2 (Asymmetric Block Expander)", "init_density": 0.46, "run_short_ga": False}),
        ("01_Innovation_Epochs_and_Density_Classification/05_Epoch3_4_GKL_Particle_Logic_rho0.54.png", "Epoch 3/4 (GKL Particle Logic, rho=0.54): particle signals classify ALL-1s", {"checkpoint": "Epoch 3/4 (GKL)", "init_density": 0.54, "run_short_ga": False}),
        ("01_Innovation_Epochs_and_Density_Classification/06_Epoch3_4_GKL_Particle_Logic_rho0.46.png", "Epoch 3/4 (GKL Particle Logic, rho=0.46): particle signals classify ALL-0s", {"checkpoint": "Epoch 3/4 (GKL)", "init_density": 0.46, "run_short_ga": False}),
        ("01_Innovation_Epochs_and_Density_Classification/07_Epoch3_4_GKL_Particle_Logic_rho0.70.png", "Epoch 3/4 (GKL Particle Logic, rho=0.70): rapid high-density convergence", {"checkpoint": "Epoch 3/4 (GKL)", "init_density": 0.70, "run_short_ga": False}),
        ("01_Innovation_Epochs_and_Density_Classification/08_Epoch3_4_GKL_Particle_Logic_rho0.30.png", "Epoch 3/4 (GKL Particle Logic, rho=0.30): rapid low-density convergence", {"checkpoint": "Epoch 3/4 (GKL)", "init_density": 0.30, "run_short_ga": False}),
        ("02_Live_GA_Evolution/09_Live_GA_12Gen_Trajectory_GKL_rho0.54.png", "Live 12-Generation GA Trajectory (Seed=7, N=49)", {"checkpoint": "Epoch 3/4 (GKL)", "init_density": 0.54, "run_short_ga": True}),
    ]
    for rel, focus, kw in baselines:
        _record_run(records, mod_dir, rel, "Baseline Dashboard", focus, m04.create_evolutionary_dashboard_figure, kw)

    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier1_Observation_Epochs_rho0.54.png", "3-Tier Protocol", "Tier 1: Epoch 1 vs Epoch 2 vs GKL on rho=0.54 (N=149) with signal slope overlay", m04.run_module04_tier1, {"init_density": 0.54})
    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier2_Analysis_Epoch2_Symmetry_Bias.png", "3-Tier Protocol", "Tier 2: Epoch 2 at rho=0.44 vs 0.56 exposing symmetry breaking & Land-Belew curve", m04.run_module04_tier2)
    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier3_LiveGA_Schema_Theorem_Seed7.png", "3-Tier Protocol", "Tier 3: Live 12-Gen GA (Seed=7) & Holland Schema Theorem survival verification", m04.run_module04_tier3, {"seed": 7})

    _record_run(records, mod_dir, "04_Real_World_Applications/App1_Protein_Directed_Evolution_NK.png", "Real-World Demo", "App 4.1: Directed protein evolution on smooth (K=0) vs rugged (K=6) NK landscapes", demos.demo_m04_protein_directed_evolution)
    _record_run(records, mod_dir, "04_Real_World_Applications/App2_AutoML_Neural_Architecture_Search.png", "Real-World Demo", "App 4.2: Discrete GA Neural Architecture Search (Accuracy vs FLOPs Pareto front)", demos.demo_m04_automl_nas)
    _record_run(records, mod_dir, "04_Real_World_Applications/App3_Swarm_Robotics_Foraging.png", "Real-World Demo", "App 4.3: Decentralized 2D foraging swarm robotics rule evolution", demos.demo_m04_swarm_robotics_foraging)

    gifs = m04.generate_module04_gifs(mod_dir)
    for gp in gifs:
        print(f"  [GIF] {mod_dir.name}/animations/{gp.name}")

    _write_module_artifacts(
        mod_dir=mod_dir,
        title="Module 04: Evolutionary Computation, Genetic Algorithms & Edge of Chaos",
        notebook_rel="notebooks/04_evolutionary_computation_holland.ipynb",
        description="128-bit Radius-3 CA density classification, GKL rule, Mitchell et al. epochs, Land-Belew boundary, Holland's Schema Theorem, 3-Tier protocols, and 3 domain applications.",
        records=records,
        gif_paths=gifs,
    )
    return records, gifs


def main() -> None:
    t0 = time.perf_counter()
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    m01_rec, m01_gifs = export_module_01()
    m02_rec, m02_gifs = export_module_02()
    m03_rec, m03_gifs = export_module_03()
    m04_rec, m04_gifs = export_module_04()

    print("\n=== Generating Real-World Application Animated GIFs (Modules 02-04) ===")
    rw_gifs = demos.generate_real_world_gifs(OUTPUTS_ROOT)
    for gp in rw_gifs:
        print(f"  [GIF] {gp.relative_to(OUTPUTS_ROOT)}")

    # Re-write summary & metrics for M02-M04 to include the real-world GIFs in their animations lists
    rw_by_mod: dict[str, list[Path]] = {
        "02_Transfer_Entropy_and_Network_Inference": m02_gifs + [g for g in rw_gifs if "02_Transfer_Entropy" in str(g)],
        "03_Cellular_Automata_and_Particle_Computation": m03_gifs + [g for g in rw_gifs if "03_Cellular_Automata" in str(g)],
        "04_Evolutionary_Computation_and_Edge_of_Chaos": m04_gifs + [g for g in rw_gifs if "04_Evolutionary_Computation" in str(g)],
    }
    meta_specs = [
        ("02_Transfer_Entropy_and_Network_Inference", "Module 02: Transfer Entropy, Delay Reconstruction & Network Inference", "notebooks/02_transfer_entropy_and_network_inference.ipynb", "Schreiber Transfer Entropy, Frenzel-Pompe / KSG k-NN estimator, Wibral interaction delay scan, circular surrogates, BH-FDR network recovery, 3-Tier protocols, and 3 domain applications.", m02_rec),
        ("03_Cellular_Automata_and_Particle_Computation", "Module 03: Cellular Automata & Emergent Particle Computation", "notebooks/03_ca_and_particle_computation.ipynb", "1D ECA, Wolfram Classes I-IV, Langton lambda, Hanson & Crutchfield 5-state FST domain filter, Rule 110 period-14 ether filter, XOR damage spreading, 3-Tier protocols, and 3 domain applications.", m03_rec),
        ("04_Evolutionary_Computation_and_Edge_of_Chaos", "Module 04: Evolutionary Computation, Genetic Algorithms & Edge of Chaos", "notebooks/04_evolutionary_computation_holland.ipynb", "128-bit Radius-3 CA density classification, GKL rule, Mitchell et al. epochs, Land-Belew boundary, Holland's Schema Theorem, 3-Tier protocols, and 3 domain applications.", m04_rec),
    ]
    for folder, title, nb_rel, desc, recs in meta_specs:
        _write_module_artifacts(OUTPUTS_ROOT / folder, title, nb_rel, desc, recs, rw_by_mod[folder])

    all_gifs = m01_gifs + m02_gifs + m03_gifs + m04_gifs + rw_gifs
    summary_payload = {
        "modules_01_04_figures": len(m01_rec) + len(m02_rec) + len(m03_rec) + len(m04_rec),
        "modules_01_04_gifs": len(all_gifs),
        "gif_files": [str(g.relative_to(OUTPUTS_ROOT).as_posix()) for g in all_gifs],
        "records": {
            "module_01": m01_rec,
            "module_02": m02_rec,
            "module_03": m03_rec,
            "module_04": m04_rec,
        },
    }
    (OUTPUTS_ROOT / "modules_01_04_metrics.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _update_outputs_readme(
        {
            "01_Information_Theory_and_PID": ("Module 01: Information Theory & Partial Information Decomposition (PID)", "Shannon metrics, Circular Nonlinearity proof, Landauer thermodynamic bound, Williams & Beer PID, 3-Tier protocols, and 3 real-world apps.", m01_rec, m01_gifs),
            "02_Transfer_Entropy_and_Network_Inference": ("Module 02: Transfer Entropy, Delay Reconstruction & Network Inference", "Schreiber Transfer Entropy, KSG k-NN estimator, Wibral delay scan, circular surrogates, BH-FDR network recovery, 3-Tier protocols, and 3 real-world apps.", m02_rec, rw_by_mod["02_Transfer_Entropy_and_Network_Inference"]),
            "03_Cellular_Automata_and_Particle_Computation": ("Module 03: Cellular Automata & Emergent Particle Computation", "1D ECA, Wolfram Classes I-IV, Langton lambda, 5-state Hanson-Crutchfield FST filter, damage spreading, 3-Tier protocols, and 3 real-world apps.", m03_rec, rw_by_mod["03_Cellular_Automata_and_Particle_Computation"]),
            "04_Evolutionary_Computation_and_Edge_of_Chaos": ("Module 04: Evolutionary Computation & Edge of Chaos", "128-bit Radius-3 CA, GKL particle logic, Mitchell epochs, Land-Belew boundary, Holland's Schema Theorem, 3-Tier protocols, and 3 real-world apps.", m04_rec, rw_by_mod["04_Evolutionary_Computation_and_Edge_of_Chaos"]),
        }
    )
    print(f"\nCompleted Modules 01-04 export: {summary_payload['modules_01_04_figures']} PNGs + {len(all_gifs)} GIFs in {time.perf_counter() - t0:.2f}s")


def _update_outputs_readme(m01_04_meta: dict[str, tuple[str, str, list[dict[str, Any]], list[Path]]]) -> None:
    """Update outputs/README.md with complete Modules 00-08 index including GIF animations."""
    mod_folders = sorted([p for p in OUTPUTS_ROOT.iterdir() if p.is_dir() and p.name[:2].isdigit()])
    lines = [
        "# Classroom Outputs, Teaching Figures & Animated GIFs Index",
        "",
        "Generated directly from **The Collective Intelligence & Info-Dynamics Toolkit** (`Modules 01–04` upgraded with 3-Tier Protocols, 12 Real-World Applications, and Classroom GIFs).",
        "",
        "## Module Overview",
        "",
        "| Module | Folder | PNGs | GIFs | Summary Link |",
        "| :--- | :--- | :---: | :---: | :--- |",
    ]
    total_png, total_gif = 0, 0
    for mf in mod_folders:
        n_png = len(list(mf.rglob("*.png")))
        n_gif = len(list(mf.rglob("*.gif")))
        total_png += n_png
        total_gif += n_gif
        title = m01_04_meta[mf.name][0] if mf.name in m01_04_meta else mf.name.replace("_", " ")
        lines.append(f"| **{title}** | `{mf.name}/` | {n_png} | {n_gif} | [summary.md](./{mf.name}/summary.md) |")

    lines.extend(["", f"**Total Assets:** {total_png} high-resolution PNG figures (`240 DPI`) and {total_gif} animated classroom GIFs.", "", "## Modules 01–04 Highlights & Animations", ""])
    for folder, (title, desc, recs, gifs) in m01_04_meta.items():
        lines.append(f"### {title}")
        lines.append(f"{desc} ([summary.md](./{folder}/summary.md) | [metrics.json](./{folder}/metrics.json))")
        lines.append("")
        lines.append("**Animated GIFs:**")
        for gp in gifs:
            rel_g = gp.relative_to(OUTPUTS_ROOT).as_posix()
            lines.append(f"- [`{gp.name}`](./{rel_g})")
        lines.append("")
        lines.append("**3-Tier Protocol & Real-World Applications:**")
        for rec in recs:
            if rec["category"] in ("3-Tier Protocol", "Real-World Demo"):
                lines.append(f"- [{rec['category']}] [`{rec['file']}`](./{folder}/{rec['file']}): {rec['focus']}")
        lines.append("")

    (OUTPUTS_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
