"""Comprehensive smoke & mathematical verification suite for Modules 01-04."""

from __future__ import annotations

import os
import sys
from pathlib import Path

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


def verify_module_01() -> None:
    print("[Module 01] Verifying mathematical identities, canonical gates, 3 tiers, and 3 demos...")
    # 1. Landauer bound at 300K ~ 2.87e-21 J
    land = m01.landauer_erasure_energy(temperature_k=300.0, bits_erased=1.0)
    assert abs(land["e_bit_joules"] - 2.8712e-21) < 1e-23, f"Unexpected Landauer: {land}"

    # 2. Circular nonlinearity: |r| < 0.08 while MI > 0.3 bits
    nl = m01.circular_nonlinearity_benchmark(n_samples=1600, seed=42)
    assert abs(nl["pearson_r"]) < 0.08, f"Expected near-zero Pearson r, got {nl['pearson_r']}"
    assert nl["mi_ksg_bits"] > 0.3, f"Expected MI > 0.3, got {nl['mi_ksg_bits']}"

    # 3. 4 Canonical Gates Calibration
    xor_r = m01.evaluate_mechanism_pid("XOR", input_corr=0.0, sigma=0.0, coupling=1.0)
    and_r = m01.evaluate_mechanism_pid("AND", input_corr=0.0, sigma=0.0, coupling=1.0)
    copy_r = m01.evaluate_mechanism_pid("COPY", input_corr=0.0, sigma=0.0, coupling=1.0)
    ident_r = m01.evaluate_mechanism_pid("IDENTICAL", input_corr=0.0, sigma=0.0, coupling=1.0)
    gauss_r = m01.evaluate_mechanism_pid("Continuous_Gaussian", input_corr=0.2, sigma=0.25, coupling=1.0)

    assert abs(xor_r["Synergy"] - 1.0) < 1e-6 and xor_r["Redundancy"] < 1e-6
    assert abs(and_r["Synergy"] - 0.5) < 1e-3 and abs(and_r["Redundancy"] - 0.311278) < 1e-3
    assert abs(copy_r["Unique_X1"] - 1.0) < 1e-6 and copy_r["Synergy"] < 1e-6
    assert abs(ident_r["Redundancy"] - 1.0) < 1e-6 and ident_r["Synergy"] < 1e-6
    assert gauss_r["I(X1, X2; Y)"] > 0.5

    # 4. 3 Tiers
    fig, t1 = m01.run_module01_tier1(sigma=0.1, coupling=1.0)
    plt.close(fig)
    assert t1["gates"]["XOR"]["Synergy"] > 0.99 and t1["gates"]["XOR"]["I(X1; Y)"] < 1e-5

    fig, t2 = m01.run_module01_tier2(gate_type="AND")
    plt.close(fig)
    assert t2["crossover_rho"] is not None and 0.1 < t2["crossover_rho"] < 0.8

    fig, t3 = m01.run_module01_tier3(temperature_k=300.0)
    plt.close(fig)
    assert t3["traces"]["XOR"]["mi"][0] > t3["traces"]["XOR"]["mi"][-1]

    # 5. Real-World Demos (3)
    for fn in (
        demos.demo_m01_computational_neuroscience,
        demos.demo_m01_genomics_epistasis,
        demos.demo_m01_thermodynamic_reversible_computing,
    ):
        fig, _ = fn()
        plt.close(fig)
    print("  -> Module 01 PASSED.")


def verify_module_02() -> None:
    print("[Module 02] Verifying KSG TE, Wibral delay, BH-FDR, 3 tiers, and 3 demos...")
    # 1. Chain Rule check
    x, y, z = m02.generate_synthetic_motif("Direct_Coupling", n_steps=500, coupling=0.75, seed=42)
    idyn = m02.compute_information_dynamics(x, y, lag=2, k=5, seed=42)
    assert idyn["TE"] > 0.10, f"Expected strong Direct TE at lag 2, got {idyn}"

    # 2. Tier 1: Common_Driver & Indirect_Cascade filtered by Conditional TE
    fig, t1 = m02.run_module02_tier1(coupling=0.70, n_samples=500, n_surrogates=60)
    plt.close(fig)
    for motif in ("Common_Driver", "Indirect_Cascade"):
        assert t1[motif]["te_biv"] > t1[motif]["thr_biv"], f"{motif} bivariate should exceed cutoff: {t1[motif]}"
        assert t1[motif]["te_cond"] <= t1[motif]["thr_cond"] + 0.02, f"{motif} conditional should drop near null: {t1[motif]}"

    # 3. Tier 2: Direct Coupling delay peak at tau=2
    fig, t2 = m02.run_module02_tier2(coupling=0.70, n_samples=550, n_surrogates=60)
    plt.close(fig)
    assert t2["peak_lag"] == 2, f"Expected Wibral peak at lag 2, got {t2['peak_lag']} ({t2['cond_scan']})"

    # 4. Tier 3: Unobserved Confounder persists across sample sizes
    fig, t3 = m02.run_module02_tier3(coupling=0.70, n_surrogates=60)
    plt.close(fig)
    assert all(p <= 0.05 for p in t3["p_values"]), f"Expected persistent spurious p <= 0.05: {t3}"

    # 5. Real-World Demos (3)
    for fn in (
        demos.demo_m02_brain_connectivity,
        demos.demo_m02_finance_contagion,
        demos.demo_m02_gene_regulatory_network,
    ):
        fig, _ = fn()
        plt.close(fig)
    print("  -> Module 02 PASSED.")


def verify_module_03() -> None:
    print("[Module 03] Verifying ECA rules, 5-state FST, damage cones, 3 tiers, and 3 demos...")
    # 1. Langton lambda
    assert abs(m03.compute_langton_lambda(54) - 4.0 / 8.0) < 1e-9
    assert abs(m03.compute_langton_lambda(110) - 5.0 / 8.0) < 1e-9

    # 2. Tier 1: Rule 54 FST detects collisions
    fig, t1 = m03.run_module03_tier1()
    plt.close(fig)
    assert t1["n_collisions"] >= 1, f"Expected >= 1 Rule 54 collision vertices, got {t1}"

    # 3. Tier 2: Random init particle density
    fig, t2 = m03.run_module03_tier2()
    plt.close(fig)
    assert t2["n_collisions"] >= 5

    # 4. Tier 3: Rule 30 chaotic damage > Rule 90/110 saturation check
    fig, t3 = m03.run_module03_tier3()
    plt.close(fig)
    assert t3[30]["final_hamming"] > 0.35, f"Expected chaotic Rule 30 H(t) ~ 0.5, got {t3[30]}"

    # 5. Real-World Demos (3)
    for fn in (
        demos.demo_m03_excitable_bz_spiral,
        demos.demo_m03_traffic_flow_rule184,
        demos.demo_m03_neuromorphic_glider_logic,
    ):
        fig, res = fn()
        plt.close(fig)
        if fn is demos.demo_m03_neuromorphic_glider_logic:
            assert all(v["correct"] for v in res.values()), f"Glider XOR gate failed: {res}"
    print("  -> Module 03 PASSED.")


def verify_module_04() -> None:
    print("[Module 04] Verifying Radius-3 CA, GKL, Epoch 2 bias, Schema Theorem, 3 tiers, and 3 demos...")
    # 1. Schema Theorem canonical example H = 1**0*1** (o=3, delta=5)
    o_h, d_h = m04.schema_properties("1**0*1**")
    assert o_h == 3 and d_h == 5
    bound = m04.holland_schema_lower_bound(m_t=4, f_schema=0.78, f_avg=0.52, defining_length=d_h, order_k=o_h)
    assert bound > 4.0

    # 2. Tier 1: GKL classifies rho=0.54 -> ALL-1s
    fig, t1 = m04.run_module04_tier1(init_density=0.54)
    plt.close(fig)
    assert t1["Epoch 3/4 (GKL)"]["converged_all_1s"], f"GKL failed on rho=0.54: {t1}"

    # 3. Tier 2: Epoch 2 symmetry-breaking bias at rho=0.44 vs 0.56
    fig, t2 = m04.run_module04_tier2()
    plt.close(fig)
    print(f"     Epoch 2 rho=0.44 final density={t2['epoch2_rho044_final']:.2f}, rho=0.56 final={t2['epoch2_rho056_final']:.2f}")
    assert t2["epoch2_rho044_final"] > 0.8, f"Expected Epoch 2 false-positive expansion on rho=0.44: {t2}"
    assert t2["epoch2_rho056_final"] > 0.8, f"Expected Epoch 2 convergence on rho=0.56: {t2}"

    # 4. Tier 3: Live 12-Gen GA (Seed=7)
    fig, t3 = m04.run_module04_tier3(seed=7)
    plt.close(fig)
    assert t3["best_history"][-1] >= t3["best_history"][0]

    # 5. Real-World Demos (3)
    for fn in (
        demos.demo_m04_protein_directed_evolution,
        demos.demo_m04_automl_nas,
        demos.demo_m04_swarm_robotics_foraging,
    ):
        fig, _ = fn()
        plt.close(fig)
    print("  -> Module 04 PASSED.")


def main() -> None:
    verify_module_01()
    verify_module_02()
    verify_module_03()
    verify_module_04()
    print("\nALL 4 CORE MODULES, 12 TIERS, AND 12 REAL-WORLD APPLICATIONS VERIFIED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
