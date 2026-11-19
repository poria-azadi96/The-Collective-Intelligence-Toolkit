"""Run and export complete classroom outputs, 3-Tier protocols, 12 real-world demos & GIFs for Modules 05-08."""

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
    adaptive_spatial_networks as m06,
    collective_markets as m07,
    rbn_immune as m05,
    real_world_demos_part2 as demos2,
    research_validation as m08,
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
        return {
            str(k): _json_safe(v)
            for k, v in obj.items()
            if str(k) not in ("history_pops", "pos_hist", "angle_hist", "low_sim", "high_sim", "d_history")
        }
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


def export_module_05() -> tuple[list[dict[str, Any]], list[Path]]:
    print("\n=== Running Module 05: RBN & Distributed Immune Selection ===")
    mod_dir = OUTPUTS_ROOT / "05_RBN_and_Immune_Cognition"
    records: list[dict[str, Any]] = []

    baselines = [
        ("01_RBN_Phase_Transitions_and_Derrida_Curves/01_Ordered_Phase_K1_p0.50.png", "Ordered Regime (K=1, p=0.50, s=0.50 < 1): perturbations die out", {"mode": "1. RBN Dynamics & Derrida Curve", "k_inputs": 1, "bias_p": 0.50, "knockout_gene": False}),
        ("01_RBN_Phase_Transitions_and_Derrida_Curves/02_Ordered_Canalizing_Bias_K3_p0.15.png", "Ordered via Canalizing Bias (K=3, p=0.15, s=0.765 < 1)", {"mode": "1. RBN Dynamics & Derrida Curve", "k_inputs": 3, "bias_p": 0.15, "knockout_gene": False}),
        ("01_RBN_Phase_Transitions_and_Derrida_Curves/03_Critical_Edge_K2_p0.50.png", "Critical Edge of Chaos (K=2, p=0.50, s=1.00): tangent to 45-deg line", {"mode": "1. RBN Dynamics & Derrida Curve", "k_inputs": 2, "bias_p": 0.50, "knockout_gene": False, "seed": 6}),
        ("01_RBN_Phase_Transitions_and_Derrida_Curves/04_Critical_Edge_Tuned_Bias_K3_p0.20.png", "Near-Critical Tuned Bias (K=3, p=0.20, s=0.96)", {"mode": "1. RBN Dynamics & Derrida Curve", "k_inputs": 3, "bias_p": 0.20, "knockout_gene": False}),
        ("01_RBN_Phase_Transitions_and_Derrida_Curves/05_Chaotic_Phase_K3_p0.50.png", "Chaotic Regime (K=3, p=0.50, s=1.50 > 1): damage amplification", {"mode": "1. RBN Dynamics & Derrida Curve", "k_inputs": 3, "bias_p": 0.50, "knockout_gene": False}),
        ("01_RBN_Phase_Transitions_and_Derrida_Curves/06_Deep_Chaos_K4_p0.50.png", "Deep Chaos Regime (K=4, p=0.50, s=2.00): steep Derrida curve", {"mode": "1. RBN Dynamics & Derrida Curve", "k_inputs": 4, "bias_p": 0.50, "knockout_gene": False}),
        ("02_Gene_Knockout_Attractor_Shifts/08_Knockout_Node0_Critical_K2_p0.50.png", "Knockout Node 0 in Critical K=2 RBN: basin merging & robustness", {"mode": "1. RBN Dynamics & Derrida Curve", "k_inputs": 2, "bias_p": 0.50, "knockout_gene": True, "seed": 6}),
        ("02_Gene_Knockout_Attractor_Shifts/09_Knockout_Node0_Chaotic_K3_p0.50.png", "Knockout Node 0 in Chaotic K=3 RBN: structural cycle collapse", {"mode": "1. RBN Dynamics & Derrida Curve", "k_inputs": 3, "bias_p": 0.50, "knockout_gene": True}),
        ("02_Gene_Knockout_Attractor_Shifts/10_Knockout_Node0_DeepChaos_K4_p0.50.png", "Knockout Node 0 in Deep Chaos K=4 RBN: dramatic basin split", {"mode": "1. RBN Dynamics & Derrida Curve", "k_inputs": 4, "bias_p": 0.50, "knockout_gene": True}),
        ("03_Immune_Clonal_Selection_and_Proofreading/12_Immune_Clonal_Selection_and_Kinetic_Proofreading.png", "Immune Clonal Selection (45 Gens) & McKeithan Kinetic Proofreading", {"mode": "2. Immune Clonal Selection & Proofreading", "proofreading_steps": 8}),
    ]
    for rel, focus, kw in baselines:
        _record_run(records, mod_dir, rel, "Baseline Dashboard", focus, m05.create_module05_dashboard_figure, kw)

    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier1_Observation_Derrida_K1_to_K4.png", "3-Tier Protocol", "Tier 1: Sweep K=1..4 at p=0.50 showing Derrida crossing s=1", m05.run_module05_tier1)
    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier2_Analysis_Knockout_Sync_vs_Async.png", "3-Tier Protocol", "Tier 2: Knockout Node 0 basin merging & Sync vs Async convergence", m05.run_module05_tier2)
    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier3_Advanced_Immune_Proofreading.png", "3-Tier Protocol", "Tier 3: 45-gen Clonal Maturation & N=1..10 exponential specificity", m05.run_module05_tier3)

    _record_run(records, mod_dir, "04_Real_World_Applications/App1_Cancer_Attractor_Landscape.png", "Real-World Demo", "App 5.1: Oncogenic mutations distorting epigenetic cell-fate basins", demos2.demo_m05_cancer_attractor_landscape)
    _record_run(records, mod_dir, "04_Real_World_Applications/App2_Immunotherapy_Specificity.png", "Real-World Demo", "App 5.2: T-cell tolerance & kinetic proofreading therapeutic window", demos2.demo_m05_immunotherapy_specificity)
    _record_run(records, mod_dir, "04_Real_World_Applications/App3_Canalizing_Async_GRNs.png", "Real-World Demo", "App 5.3: Canalizing truth tables stabilizing asynchronous GRNs", demos2.demo_m05_canalizing_grns)

    gifs = m05.generate_module05_gifs(mod_dir)
    for gp in gifs:
        print(f"  [GIF] {mod_dir.name}/animations/{gp.name}")

    _write_module_artifacts(
        mod_dir=mod_dir,
        title="Module 05: Random Boolean Networks & Distributed Immune Selection",
        notebook_rel="notebooks/05_rbn_and_immune_cognition.ipynb",
        description="Kauffman RBN criticality (Derrida curve), sync vs async attractors, gene knockouts, canalizing GRNs, immune clonal maturation, and kinetic proofreading.",
        records=records,
        gif_paths=gifs,
    )
    return records, gifs


def export_module_06() -> tuple[list[dict[str, Any]], list[Path]]:
    print("\n=== Running Module 06: Physarum, Complex Contagion & Vicsek Flocking ===")
    mod_dir = OUTPUTS_ROOT / "06_Physarum_Contagion_and_Flocking"
    records: list[dict[str, Any]] = []

    baselines = [
        ("01_Physarum_Tokyo_Rail_Solver/01_Physarum_fault_tolerant_mesh_gamma0.50.png", "Physarum Fault-Tolerant Mesh (mu=0.50): resilient loop mesh", {"mode": "Physarum Tokyo Rail", "mu": 0.50, "disrupt_edges": False}),
        ("01_Physarum_Tokyo_Rail_Solver/02_Physarum_balanced_backbone_gamma0.60.png", "Physarum Balanced Backbone (mu=0.60): matches MST efficiency", {"mode": "Physarum Tokyo Rail", "mu": 0.60, "disrupt_edges": False}),
        ("01_Physarum_Tokyo_Rail_Solver/04_Physarum_optimal_default_gamma1.40.png", "Physarum Optimal Pruning (mu=1.40): hub-centered rail backbone", {"mode": "Physarum Tokyo Rail", "mu": 1.40, "disrupt_edges": False}),
        ("01_Physarum_Tokyo_Rail_Solver/05_Physarum_strong_pruning_gamma2.00.png", "Physarum Strong Pruning (mu=2.00): minimal hub core", {"mode": "Physarum Tokyo Rail", "mu": 2.00, "disrupt_edges": False}),
        ("01_Physarum_Tokyo_Rail_Solver/06_Physarum_hub_disruption_recovery_gamma0.60.png", "Physarum Live Trunk Severing Recovery (mu=0.60, Cut=True)", {"mode": "Physarum Tokyo Rail", "mu": 0.60, "disrupt_edges": True}),
        ("01_Physarum_Tokyo_Rail_Solver/07_Physarum_hub_disruption_recovery_gamma1.40.png", "Physarum Live Trunk Severing Recovery (mu=1.40, Cut=True)", {"mode": "Physarum Tokyo Rail", "mu": 1.40, "disrupt_edges": True}),
        ("02_Simple_vs_Complex_Contagion/08_Contagion_Clustered_Lattice_p0.02.png", "Clustered Lattice (p_rewire=0.02): Complex Contagion succeeds", {"mode": "Complex Contagion", "rewire_prob": 0.02, "contagion_threshold": 2}),
        ("02_Simple_vs_Complex_Contagion/09_Contagion_Small_World_Shortcuts_p0.35.png", "Small-World Shortcuts (p_rewire=0.35): Complex Contagion collapses", {"mode": "Complex Contagion", "rewire_prob": 0.35, "contagion_threshold": 2}),
        ("03_Vicsek_Collective_Flocking/10_Vicsek_Ordered_Flocking_low_noise_eta0.40.png", "Vicsek Ordered Flocking (eta=0.40): coherent polarization", {"mode": "Vicsek Flocking", "vicsek_eta": 0.40}),
        ("03_Vicsek_Collective_Flocking/12_Vicsek_Transition_Regime_eta2.80.png", "Vicsek Transition Regime (eta=2.80): partial local alignment", {"mode": "Vicsek Flocking", "vicsek_eta": 2.80}),
        ("03_Vicsek_Collective_Flocking/13_Vicsek_Disordered_Gas_high_noise_eta4.40.png", "Vicsek Disordered Gas (eta=4.40): collapsed polarization", {"mode": "Vicsek Flocking", "vicsek_eta": 4.40}),
    ]
    for rel, focus, kw in baselines:
        _record_run(records, mod_dir, rel, "Baseline Dashboard", focus, m06.create_module06_dashboard_figure, kw)

    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier1_Physarum_Tokyo_Rail.png", "3-Tier Protocol", "Tier 1: Physarum sparse-Laplacian self-organization vs MST", m06.run_module06_tier1)
    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier2_Complex_Contagion_Rewiring.png", "3-Tier Protocol", "Tier 2: Watts-Strogatz rewiring sweep showing clustering collapse", m06.run_module06_tier2)
    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier3_Vicsek_Transition_and_Severing.png", "3-Tier Protocol", "Tier 3: Vicsek phi(eta) phase curve & live Physarum edge severing", m06.run_module06_tier3)

    _record_run(records, mod_dir, "04_Real_World_Applications/App1_Resilient_Transit_Power_Grid.png", "Real-World Demo", "App 6.1: Self-healing Tokyo grid under live severing & non-uniform demand", demos2.demo_m06_resilient_transit_power_grid)
    _record_run(records, mod_dir, "04_Real_World_Applications/App2_Public_Health_Diffusion.png", "Real-World Demo", "App 6.2: Clustered vs dispersed seeding for T=2 behavioral adoption", demos2.demo_m06_public_health_diffusion)
    _record_run(records, mod_dir, "04_Real_World_Applications/App3_Autonomous_Drone_Swarm.png", "Real-World Demo", "App 6.3: Decentralized 2D autonomous drone obstacle-avoidance swarm", demos2.demo_m06_autonomous_drone_swarm)

    gifs = m06.generate_module06_gifs(mod_dir)
    for gp in gifs:
        print(f"  [GIF] {mod_dir.name}/animations/{gp.name}")

    _write_module_artifacts(
        mod_dir=mod_dir,
        title="Module 06: Physarum Self-Organization, Complex Contagion & Vicsek Flocking",
        notebook_rel="notebooks/06_physarum_and_contagion.ipynb",
        description="Tero et al. (2010) sparse-Laplacian Physarum solver with live severing, Centola (2010) Simple vs Complex Contagion, Vicsek collective motion, 3-Tier protocols, and 3 real-world apps.",
        records=records,
        gif_paths=gifs,
    )
    return records, gifs


def export_module_07() -> tuple[list[dict[str, Any]], list[Path]]:
    print("\n=== Running Module 07: Collective Wisdom, Bounded Confidence & El Farol Market ===")
    mod_dir = OUTPUTS_ROOT / "07_Collective_Wisdom_Opinion_and_El_Farol"
    records: list[dict[str, Any]] = []

    baselines = [
        ("01_Diversity_Prediction_Theorem/01_DPT_Low_Correlation_rho0.05.png", "DPT Low Error Correlation (rho=0.05): strong crowd wisdom gain", {"study": "1. Diversity Prediction Theorem", "rho": 0.05}),
        ("01_Diversity_Prediction_Theorem/02_DPT_Moderate_Correlation_rho0.25.png", "DPT Moderate Error Correlation (rho=0.25): partial diversity benefit", {"study": "1. Diversity Prediction Theorem", "rho": 0.25}),
        ("01_Diversity_Prediction_Theorem/03_DPT_High_Correlation_Groupthink_rho0.80.png", "DPT High Correlation / Groupthink (rho=0.80): diversity collapses", {"study": "1. Diversity Prediction Theorem", "rho": 0.80}),
        ("02_Bounded_Confidence_Opinion_Dynamics/04_HK_Fragmentation_eps0.08.png", "Hegselmann-Krause Fragmentation (eps=0.05): isolated echo chambers", {"study": "2. Bounded Confidence (HK vs DeGroot)", "eps": 0.05}),
        ("02_Bounded_Confidence_Opinion_Dynamics/05_HK_Polarization_eps0.18.png", "Hegselmann-Krause Polarization (eps=0.18): multipolar clusters", {"study": "2. Bounded Confidence (HK vs DeGroot)", "eps": 0.18}),
        ("02_Bounded_Confidence_Opinion_Dynamics/07_HK_Full_Consensus_eps0.40.png", "Hegselmann-Krause Global Consensus (eps=0.40): 1 unified cluster", {"study": "2. Bounded Confidence (HK vs DeGroot)", "eps": 0.40}),
        ("03_El_Farol_and_Minority_Game/08_El_Farol_and_MG_Independent_M3.png", "El Farol Bar & Minority Game Independent Agents (below coin-flip)", {"study": "3. El Farol Bar & Minority Game", "memory_m": 3, "herding_ext": False}),
        ("03_El_Farol_and_Minority_Game/09_El_Farol_and_MG_Herding_Extension_M3.png", "El Farol & Minority Game with Networked Herding (q=0.35 spike)", {"study": "3. El Farol Bar & Minority Game", "memory_m": 3, "herding_q": 0.35, "herding_ext": True}),
    ]
    for rel, focus, kw in baselines:
        _record_run(records, mod_dir, rel, "Baseline Dashboard", focus, m07.create_module07_dashboard_figure, kw)

    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier1_DPT_Identity_and_Rho_Sweep.png", "3-Tier Protocol", "Tier 1: Page single-trial identity & rho correlation sweep", m07.run_module07_tier1)
    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier2_HK_Bifurcation_vs_DeGroot.png", "3-Tier Protocol", "Tier 2: HK presets (0.05, 0.18, 0.40) vs DeGroot linear consensus", m07.run_module07_tier2)
    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier3_ElFarol_MinorityGame_Herding.png", "3-Tier Protocol", "Tier 3: El Farol L=60, MG cusp alpha_c & networked herding volatility", m07.run_module07_tier3)

    _record_run(records, mod_dir, "04_Real_World_Applications/App1_Financial_Limit_Order_Book.png", "Real-World Demo", "App 7.1: Chartist vs fundamentalist order book & flash-crash liquidity drought", demos2.demo_m07_financial_limit_order_book)
    _record_run(records, mod_dir, "04_Real_World_Applications/App2_Forecasting_Delphi_Ensemble.png", "Real-World Demo", "App 7.2: Delphi decorrelation & diversity-weighted forecasting pipeline", demos2.demo_m07_forecasting_delphi_ensemble)
    _record_run(records, mod_dir, "04_Real_World_Applications/App3_LLM_Multi_Agent_Debate.png", "Real-World Demo", "App 7.3: Multi-agent AI/LLM debate accuracy vs sycophantic convergence", demos2.demo_m07_llm_multi_agent_debate)

    gifs = m07.generate_module07_gifs(mod_dir)
    for gp in gifs:
        print(f"  [GIF] {mod_dir.name}/animations/{gp.name}")

    _write_module_artifacts(
        mod_dir=mod_dir,
        title="Module 07: Collective Wisdom, Bounded Confidence & El Farol Market",
        notebook_rel="notebooks/07_el_farol_and_opinion_dynamics.ipynb",
        description="Scott Page's DPT, Hegselmann-Krause vs DeGroot consensus, Brian Arthur's El Farol Bar, Challet-Zhang Minority Game with Networked Herding, 3-Tier protocols, and 3 real-world apps.",
        records=records,
        gif_paths=gifs,
    )
    return records, gifs


def export_module_08() -> tuple[list[dict[str, Any]], list[Path]]:
    print("\n=== Running Module 08: Computational Research Design & Validation Lab ===")
    mod_dir = OUTPUTS_ROOT / "08_Research_Design_and_Validation_Lab"
    records: list[dict[str, Any]] = []

    baselines = [
        ("01_Track_A_Observational_FDR_and_Power/01_TrackA_LowSNR0.55_T100.png", "Track A Underpowered (SNR=0.55, T=100): uncorrected false positives", {"paper_choice": "1. Schreiber (2000) - Transfer Entropy", "track": "Track A: Observational FDR & Power", "sample_size": 100, "signal_snr": 0.55}),
        ("01_Track_A_Observational_FDR_and_Power/02_TrackA_LowSNR0.55_T300.png", "Track A Underpowered (SNR=0.55, T=300): BH-FDR prevents false discoveries", {"paper_choice": "1. Schreiber (2000) - Transfer Entropy", "track": "Track A: Observational FDR & Power", "sample_size": 300, "signal_snr": 0.55}),
        ("01_Track_A_Observational_FDR_and_Power/03_TrackA_LowSNR0.55_T800.png", "Track A Low SNR (SNR=0.55, T=800): power shortfall under multiple testing", {"paper_choice": "1. Schreiber (2000) - Transfer Entropy", "track": "Track A: Observational FDR & Power", "sample_size": 800, "signal_snr": 0.55}),
        ("01_Track_A_Observational_FDR_and_Power/04_TrackA_Calibrated_SNR1.25_T1200.png", "Track A Calibrated (SNR=1.25, T=1200): crosses 80% BH-FDR power target", {"paper_choice": "1. Schreiber (2000) - Transfer Entropy", "track": "Track A: Observational FDR & Power", "sample_size": 1200, "signal_snr": 1.25}),
        ("02_Track_B_Generative_ABM_vs_Null_Model/05_TrackB_ABM_vs_Null_Baseline.png", "Track B Mechanistic Transition vs Null Baseline & Cohen's d separation", {"paper_choice": "5. Tero et al. (2010) - Physarum Tokyo Rail", "track": "Track B: Generative ABM vs Null Model", "coupling_strength": 0.70}),
    ]
    for rel, focus, kw in baselines:
        _record_run(records, mod_dir, rel, "Baseline Dashboard", focus, m08.create_module08_dashboard_figure, kw)

    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier1_Paper_Audit_Q7_to_Q8_Matrix.png", "3-Tier Protocol", "Tier 1: 8-Question Paper Audit Inspector mapping Q7 -> Q8 across 6 classics", m08.run_module08_tier1)
    _record_run(records, mod_dir, "03_3Tier_Protocol/Tier2_TrackA_Power_Calibration_T80.png", "3-Tier Protocol", "Tier 2: Track A BH-FDR power curve solving minimum sample size T_80", m08.run_module08_tier2)
    t3_stats = _record_run(records, mod_dir, "03_3Tier_Protocol/Tier3_TrackB_Null_CohensD.png", "3-Tier Protocol", "Tier 3: Track B Null separation Cohen's d & 5-Part Research Brief export", m08.run_module08_tier3)

    # Export sample 5-Part Research Brief Markdown & JSON
    brief_dir = mod_dir / "03_8_Question_Paper_Audits"
    brief_dir.mkdir(parents=True, exist_ok=True)
    if "sample_brief" in t3_stats:
        (brief_dir / "sample_5part_research_brief.md").write_text(t3_stats["sample_brief"]["markdown"], encoding="utf-8")
        (brief_dir / "sample_5part_research_brief.json").write_text(t3_stats["sample_brief"]["json"], encoding="utf-8")

    _record_run(records, mod_dir, "04_Real_World_Applications/App1_Q1_Peer_Review_Stress_Suite.png", "Real-World Demo", "App 8.1: Automated Q7 failure-mode stress-test across workshop models", demos2.demo_m08_q1_peer_review_stress_test)
    _record_run(records, mod_dir, "04_Real_World_Applications/App2_Industrial_RD_Power_Pipeline.png", "Real-World Demo", "App 8.2: Pre-experiment multi-SNR T_80 power & null Cohen's d pipeline", demos2.demo_m08_industrial_rd_power_pipeline)
    _record_run(records, mod_dir, "04_Real_World_Applications/App3_Cross_Module_Grand_Synthesis.png", "Real-World Demo", "App 8.3: Cross-module reproducibility scorecard & grand synthesis report", demos2.demo_m08_grand_synthesis_pipeline)

    gifs = m08.generate_module08_gifs(mod_dir)
    for gp in gifs:
        print(f"  [GIF] {mod_dir.name}/animations/{gp.name}")

    _write_module_artifacts(
        mod_dir=mod_dir,
        title="Module 08: Computational Research Design & Validation Lab",
        notebook_rel="notebooks/08_research_design_lab.ipynb",
        description="8-Question Paper Audit Engine across 6 classics, Track A BH-FDR power calibration (T_80), Track B null baseline testing (Cohen's d), 5-Part Research Brief builder, 3-Tier protocols, and 3 real-world apps.",
        records=records,
        gif_paths=gifs,
    )
    return records, gifs


def _update_full_outputs_readme(part2_meta: dict[str, tuple[str, str, list[dict[str, Any]], list[Path]]]) -> None:
    """Regenerate unified outputs/README.md covering Modules 00 through 08."""
    mod_folders = sorted([p for p in OUTPUTS_ROOT.iterdir() if p.is_dir() and p.name[:2].isdigit()])
    titles_00_04 = {
        "00_Pre_Workshop_Primer": "Module 00: Pre-Workshop Primer",
        "01_Information_Theory_and_PID": "Module 01: Information Theory & Partial Information Decomposition (PID)",
        "02_Transfer_Entropy_and_Network_Inference": "Module 02: Transfer Entropy, Delay Reconstruction & Network Inference",
        "03_Cellular_Automata_and_Particle_Computation": "Module 03: Cellular Automata & Emergent Particle Computation",
        "04_Evolutionary_Computation_and_Edge_of_Chaos": "Module 04: Evolutionary Computation & Edge of Chaos",
    }

    lines = [
        "# Classroom Outputs, Teaching Figures & Animated GIFs Index",
        "",
        "Generated directly from **The Collective Intelligence & Info-Dynamics Toolkit** (`Modules 00–08` complete with 3-Tier Experimental Protocols, 24 Real-World Applications across Modules 01–08, and Classroom GIFs).",
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
        if mf.name in part2_meta:
            title = part2_meta[mf.name][0]
        else:
            title = titles_00_04.get(mf.name, mf.name.replace("_", " "))
        lines.append(f"| **{title}** | `{mf.name}/` | {n_png} | {n_gif} | [summary.md](./{mf.name}/summary.md) |")

    lines.extend(["", f"**Total Assets:** {total_png} high-resolution PNG figures (`240 DPI`) and {total_gif} animated classroom GIFs.", ""])

    # Load M01-04 metrics.json if present for unified listing
    m01_04_file = OUTPUTS_ROOT / "modules_01_04_metrics.json"
    if m01_04_file.exists():
        lines.append("## Modules 01–04 Highlights & Animations")
        lines.append("")
        m14 = json.loads(m01_04_file.read_text(encoding="utf-8"))
        for mf_name, mod_key in [
            ("01_Information_Theory_and_PID", "module_01"),
            ("02_Transfer_Entropy_and_Network_Inference", "module_02"),
            ("03_Cellular_Automata_and_Particle_Computation", "module_03"),
            ("04_Evolutionary_Computation_and_Edge_of_Chaos", "module_04"),
        ]:
            lines.append(f"### {titles_00_04[mf_name]}")
            lines.append(f"[summary.md](./{mf_name}/summary.md) | [metrics.json](./{mf_name}/metrics.json)")
            anim_gifs = sorted((OUTPUTS_ROOT / mf_name / "animations").glob("*.gif"))
            if anim_gifs:
                lines.append("")
                lines.append("**Animated GIFs:**")
                for gp in anim_gifs:
                    lines.append(f"- [`{gp.name}`](./{gp.relative_to(OUTPUTS_ROOT).as_posix()})")
            lines.append("")

    lines.append("## Modules 05–08 Highlights & Animations")
    lines.append("")
    for folder, (title, desc, recs, gifs) in part2_meta.items():
        lines.append(f"### {title}")
        lines.append(f"{desc} ([summary.md](./{folder}/summary.md) | [metrics.json](./{folder}/metrics.json))")
        lines.append("")
        lines.append("**Animated GIFs:**")
        for gp in gifs:
            lines.append(f"- [`{gp.name}`](./{gp.relative_to(OUTPUTS_ROOT).as_posix()})")
        lines.append("")
        lines.append("**3-Tier Protocol & Real-World Applications:**")
        for rec in recs:
            if rec["category"] in ("3-Tier Protocol", "Real-World Demo"):
                lines.append(f"- [{rec['category']}] [`{rec['file']}`](./{folder}/{rec['file']}): {rec['focus']}")
        lines.append("")

    (OUTPUTS_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.perf_counter()
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    m05_rec, m05_gifs = export_module_05()
    m06_rec, m06_gifs = export_module_06()
    m07_rec, m07_gifs = export_module_07()
    m08_rec, m08_gifs = export_module_08()

    print("\n=== Generating Real-World Application Animated GIFs (Part 2) ===")
    rw_gifs = demos2.generate_real_world_gifs_part2(OUTPUTS_ROOT)
    for gp in rw_gifs:
        print(f"  [GIF] {gp.relative_to(OUTPUTS_ROOT)}")

    m06_all_gifs = m06_gifs + [g for g in rw_gifs if "06_Physarum" in str(g)]
    _write_module_artifacts(
        OUTPUTS_ROOT / "06_Physarum_Contagion_and_Flocking",
        "Module 06: Physarum Self-Organization, Complex Contagion & Vicsek Flocking",
        "notebooks/06_physarum_and_contagion.ipynb",
        "Tero et al. (2010) sparse-Laplacian Physarum solver with live severing, Centola (2010) Simple vs Complex Contagion, Vicsek collective motion, 3-Tier protocols, and 3 real-world apps.",
        m06_rec,
        m06_all_gifs,
    )

    all_gifs_p2 = m05_gifs + m06_all_gifs + m07_gifs + m08_gifs
    payload = {
        "modules_05_08_figures": len(m05_rec) + len(m06_rec) + len(m07_rec) + len(m08_rec),
        "modules_05_08_gifs": len(all_gifs_p2),
        "gif_files": [str(g.relative_to(OUTPUTS_ROOT).as_posix()) for g in all_gifs_p2],
        "records": {
            "module_05": m05_rec,
            "module_06": m06_rec,
            "module_07": m07_rec,
            "module_08": m08_rec,
        },
    }
    (OUTPUTS_ROOT / "modules_05_08_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _update_full_outputs_readme(
        {
            "05_RBN_and_Immune_Cognition": ("Module 05: Random Boolean Networks & Distributed Immune Selection", "Kauffman RBN criticality, sync vs async attractors, gene knockouts, canalizing GRNs, immune clonal maturation, and kinetic proofreading.", m05_rec, m05_gifs),
            "06_Physarum_Contagion_and_Flocking": ("Module 06: Physarum Self-Organization, Complex Contagion & Vicsek Flocking", "Sparse-Laplacian Physarum solver with live severing, Centola Complex Contagion, Vicsek flocking, and 3 real-world apps.", m06_rec, m06_all_gifs),
            "07_Collective_Wisdom_Opinion_and_El_Farol": ("Module 07: Collective Wisdom, Bounded Confidence & El Farol Market", "Page's DPT, Hegselmann-Krause vs DeGroot, El Farol Bar, Minority Game + Herding, and 3 real-world apps.", m07_rec, m07_gifs),
            "08_Research_Design_and_Validation_Lab": ("Module 08: Computational Research Design & Validation Lab", "8-Question Paper Audit Engine, Track A BH-FDR power (T_80), Track B null Cohen's d, 5-Part Research Brief, and 3 real-world apps.", m08_rec, m08_gifs),
        }
    )
    print(f"\nCompleted Modules 05-08 export: {payload['modules_05_08_figures']} PNGs + {len(all_gifs_p2)} GIFs in {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
