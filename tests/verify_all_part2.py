"""Comprehensive smoke & mathematical verification suite for Modules 05-08 (Part 2)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

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


def verify_module_05() -> None:
    print("[Module 05] Verifying Kauffman RBNs, Derrida criticality, knockouts, immune selection & 3 demos...")
    assert abs(m05.derrida_sensitivity(1, 0.50) - 0.50) < 1e-9
    assert abs(m05.derrida_sensitivity(2, 0.50) - 1.00) < 1e-9
    assert abs(m05.derrida_sensitivity(3, 0.50) - 1.50) < 1e-9

    fig, t1 = m05.run_module05_tier1()
    plt.close(fig)
    assert t1["sensitivities"][1] < 1.0 and t1["sensitivities"][2] == 1.0 and t1["sensitivities"][4] > 1.0

    fig, t2 = m05.run_module05_tier2()
    plt.close(fig)
    assert sum(t2["wt_basins"]) == 1024 and sum(t2["ko_basins"]) == 1024

    fig, t3 = m05.run_module05_tier3()
    plt.close(fig)
    assert t3["final_mean_affinity"] > 0.70 and t3["gain_step_10"] > t3["gain_step_1"] * 100.0

    for fn in (
        demos2.demo_m05_cancer_attractor_landscape,
        demos2.demo_m05_immunotherapy_specificity,
        demos2.demo_m05_canalizing_grns,
    ):
        fig, _ = fn()
        plt.close(fig)
    print("  -> Module 05 PASSED.")


def verify_module_06() -> None:
    print("[Module 06] Verifying Physarum sparse-Laplacian, Complex Contagion, Vicsek & 3 demos...")
    fig, t1 = m06.run_module06_tier1()
    plt.close(fig)
    assert t1["fault_tolerance"] >= t1["mst_fault_tolerance"]

    fig, t2 = m06.run_module06_tier2()
    plt.close(fig)
    assert t2["complex_reach_sweep"][0] > t2["complex_reach_sweep"][-1]

    fig, t3 = m06.run_module06_tier3()
    plt.close(fig)
    assert t3["vicsek_phi_low_noise"] > t3["vicsek_phi_high_noise"]

    for fn in (
        demos2.demo_m06_resilient_transit_power_grid,
        demos2.demo_m06_public_health_diffusion,
        demos2.demo_m06_autonomous_drone_swarm,
    ):
        fig, _ = fn()
        plt.close(fig)
    print("  -> Module 06 PASSED.")


def verify_module_07() -> None:
    print("[Module 07] Verifying Page's DPT, HK bounded confidence, El Farol, Minority Game & 3 demos...")
    dpt = m07.simulate_dpt_ensemble(n_agents=50, indiv_sigma=10.0, error_corr=0.25, seed=42)
    assert dpt["identity_residual"] < 1e-10, f"DPT algebraic identity failed: {dpt}"

    fig, t1 = m07.run_module07_tier1()
    plt.close(fig)
    assert t1["identity_residual"] < 1e-10

    fig, t2 = m07.run_module07_tier2()
    plt.close(fig)
    assert t2["eps=0.05"] > t2["eps=0.18"] >= t2["eps=0.40"] and t2["eps=0.40"] == 1

    fig, t3 = m07.run_module07_tier3()
    plt.close(fig)
    assert 50.0 < t3["el_farol_mean"] < 70.0 and t3["min_volatility_indep"] < 1.0

    for fn in (
        demos2.demo_m07_financial_limit_order_book,
        demos2.demo_m07_forecasting_delphi_ensemble,
        demos2.demo_m07_llm_multi_agent_debate,
    ):
        fig, _ = fn()
        plt.close(fig)
    print("  -> Module 07 PASSED.")


def verify_module_08() -> None:
    print("[Module 08] Verifying 8-Question Paper Audits, Track A T_80, Track B Cohen's d, Brief & 3 demos...")
    assert len(m08.PAPER_AUDITS) == 6

    fig, t1 = m08.run_module08_tier1()
    plt.close(fig)
    assert t1["n_papers"] == 6

    fig, t2 = m08.run_module08_tier2()
    plt.close(fig)
    assert t2["T_80"] is not None and 100.0 <= t2["T_80"] <= 1200.0, f"Expected solvable T_80, got {t2['T_80']}"

    fig, t3 = m08.run_module08_tier3()
    plt.close(fig)
    assert t3["focused_cohen_d"] > 1.5 and "1. Gap Analysis" in t3["sample_brief"]["markdown"]

    for fn in (
        demos2.demo_m08_q1_peer_review_stress_test,
        demos2.demo_m08_industrial_rd_power_pipeline,
        demos2.demo_m08_grand_synthesis_pipeline,
    ):
        fig, _ = fn()
        plt.close(fig)
    print("  -> Module 08 PASSED.")


def main() -> None:
    verify_module_05()
    verify_module_06()
    verify_module_07()
    verify_module_08()
    print("\nALL PART 2 MODULES (05-08), 12 TIERS, AND 12 REAL-WORLD APPLICATIONS VERIFIED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
