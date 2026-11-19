"""Unified Interactive Web Dashboard (Modules 01-04) for The Collective Intelligence Toolkit."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from core import (
    cellular_automata as m03,
    evolutionary_ga as m04,
    info_theory_pid as m01,
    real_world_demos as demos,
    transfer_entropy as m02,
)

REPO_ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = REPO_ROOT / "outputs"


def _render_fig(fig: plt.Figure) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_gif_gallery(module_folder: str) -> None:
    anim_dir = OUTPUTS_ROOT / module_folder / "animations"
    if anim_dir.exists():
        gifs = sorted(anim_dir.glob("*.gif"))
        if gifs:
            st.subheader("Classroom Animated Visualizations (GIFs)")
            cols = st.columns(min(2, len(gifs)))
            for idx, gif_path in enumerate(gifs):
                with cols[idx % len(cols)]:
                    st.caption(gif_path.name)
                    st.image(str(gif_path), use_container_width=True)


def render_module_01() -> None:
    st.header("Module 01: Information Theory, Nonlinearity & Partial Information Decomposition (PID)")
    tab_lab, tab_tiers, tab_apps = st.tabs(
        ["Interactive PID Lab", "3-Tier Experimental Protocol", "Real-World Applications (3)"]
    )

    with tab_lab:
        c1, c2, c3, c4, c5 = st.columns(5)
        gate = c1.selectbox(
            "Gate / Mechanism",
            ["XOR", "AND", "OR", "COPY", "IDENTICAL", "Continuous_Gaussian"],
            index=0,
        )
        rho = c2.slider("Correlation (rho)", -0.95, 0.95, 0.0, 0.05)
        sigma = c3.slider("Channel Noise (sigma)", 0.0, 1.0, 0.0, 0.05)
        coupling = c4.slider("Coupling (c)", 0.0, 1.0, 1.0, 0.05)
        temp_k = c5.slider("Temperature (K)", 4.0, 400.0, 300.0, 10.0)

        fig, stats = m01.create_pid_dashboard_figure(
            gate_type=gate, input_corr=rho, sigma=sigma, coupling=coupling, temperature_k=temp_k
        )
        _render_fig(fig)
        pid = stats["pid"]
        m_cols = st.columns(5)
        m_cols[0].metric("Redundancy R", f"{pid['Redundancy']:.4f} b")
        m_cols[1].metric("Unique U1 / U2", f"{pid['Unique_X1']:.3f} / {pid['Unique_X2']:.3f} b")
        m_cols[2].metric("Synergy S", f"{pid['Synergy']:.4f} b")
        m_cols[3].metric("Co-Info (R - S)", f"{pid['Co_Information']:+.4f} b")
        m_cols[4].metric("Landauer Heat", f"{pid['Landauer_J']:.2e} J")

    with tab_tiers:
        tier_choice = st.radio(
            "Select Experimental Protocol Tier",
            [
                "Tier 1 (Observation): XOR Pure Synergy & Canonical Gates",
                "Tier 2 (Analysis): AND Redundancy-Synergy Crossover",
                "Tier 3 (Stress Test): Noise & Landauer Heat Sweep",
            ],
        )
        if tier_choice.startswith("Tier 1"):
            fig, res = m01.run_module01_tier1(sigma=0.1, coupling=1.0)
            _render_fig(fig)
            st.json(res["gates"]["XOR"])
        elif tier_choice.startswith("Tier 2"):
            fig, res = m01.run_module01_tier2(gate_type="AND")
            _render_fig(fig)
            st.success(f"Co-Information II = R - S crosses zero at correlation rho* = {res['crossover_rho']:.4f}")
        else:
            fig, _ = m01.run_module01_tier3(temperature_k=300.0)
            _render_fig(fig)
        _render_gif_gallery("01_Information_Theory_and_PID")

    with tab_apps:
        app_choice = st.selectbox(
            "Select Module 01 Real-World Case Study",
            [
                "1. Computational Neuroscience: Coincidence Detection vs. Redundant Coding",
                "2. Genomics & Epistasis: Pure Nonlinear SNP Synergy Beyond GWAS",
                "3. Thermodynamic & Reversible Computing: CMOS vs. Fredkin/Toffoli",
            ],
        )
        if app_choice.startswith("1."):
            fig, res = demos.demo_m01_computational_neuroscience()
            _render_fig(fig)
            st.json(res)
        elif app_choice.startswith("2."):
            fig, res = demos.demo_m01_genomics_epistasis()
            _render_fig(fig)
            st.json(res)
        else:
            fig, res = demos.demo_m01_thermodynamic_reversible_computing()
            _render_fig(fig)
            st.json(res)


def render_module_02() -> None:
    st.header("Module 02: Transfer Entropy, Delay Reconstruction & Network Inference")
    tab_lab, tab_tiers, tab_apps = st.tabs(
        ["Interactive TE Lab", "3-Tier Experimental Protocol", "Real-World Applications (3)"]
    )

    with tab_lab:
        c1, c2, c3, c4, c5 = st.columns(5)
        motif = c1.selectbox(
            "Causal Motif",
            ["Common_Driver", "Indirect_Cascade", "Direct_Coupling", "Unobserved_Confounder"],
            index=0,
        )
        n_samples = c2.slider("Sample Size (N)", 250, 800, 450, 50)
        k_nn = c3.slider("k-NN Neighbors (k)", 3, 10, 5, 1)
        n_surr = c4.slider("Surrogates (N_surr)", 50, 200, 60, 10)
        coupling = c5.slider("Coupling C", 0.2, 0.9, 0.7, 0.05)

        fig, stats = m02.create_te_dashboard_figure(
            motif_type=motif,
            coupling=coupling,
            n_samples=n_samples,
            k_neighbors=k_nn,
            n_surrogates=n_surr,
        )
        _render_fig(fig)
        m_cols = st.columns(4)
        m_cols[0].metric("Optimal Lag u*", f"{stats['best_lag']} steps")
        m_cols[1].metric("Bivariate TE", f"{stats['te_bivariate']:.4f} b (p={stats['p_bivariate']:.3f})")
        m_cols[2].metric("Conditional TE | Z", f"{stats['te_conditional']:.4f} b (p={stats['p_conditional']:.3f})")
        m_cols[3].metric("AIS A_Y + TE", f"{stats['info_dynamics']['AIS_plus_TE']:.4f} b")

    with tab_tiers:
        tier_choice = st.radio(
            "Select Experimental Protocol Tier",
            [
                "Tier 1 (Observation): Common Driver & Indirect Cascade Filtering",
                "Tier 2 (Delay Reconstruction): Wibral Scan & Chain-Rule Decomposition",
                "Tier 3 (Confounder Stress Test): Unobserved Confounder Sample Sweep",
            ],
        )
        if tier_choice.startswith("Tier 1"):
            fig, res = m02.run_module02_tier1()
            _render_fig(fig)
            st.json(res)
        elif tier_choice.startswith("Tier 2"):
            fig, res = m02.run_module02_tier2()
            _render_fig(fig)
            st.success(f"Recovered true interaction delay at u* = {res['peak_lag']} steps.")
        else:
            fig, res = m02.run_module02_tier3()
            _render_fig(fig)
            st.json(res)
        _render_gif_gallery("02_Transfer_Entropy_and_Network_Inference")

    with tab_apps:
        app_choice = st.selectbox(
            "Select Module 02 Real-World Case Study",
            [
                "1. Computational Neuroscience: 4-Region Effective Brain Connectivity",
                "2. Quantitative Finance: Asymmetric Volatility Contagion Network",
                "3. Systems Biology: Hill-Function Gene Regulatory Network",
            ],
        )
        if app_choice.startswith("1."):
            fig, res = demos.demo_m02_brain_connectivity()
            _render_fig(fig)
            st.write("True Edges:", res["true_edges"])
            st.write("BH-FDR Recovered Edges:", res["recovered_edges"])
        elif app_choice.startswith("2."):
            fig, res = demos.demo_m02_finance_contagion()
            _render_fig(fig)
            st.json(res["out_strength"])
        else:
            fig, res = demos.demo_m02_gene_regulatory_network()
            _render_fig(fig)
            st.write("True Edges:", res["true_edges"])
            st.write("BH-FDR Recovered Edges:", res["recovered_edges"])


def render_module_03() -> None:
    st.header("Module 03: Cellular Automata & Emergent Particle Computation")
    tab_lab, tab_tiers, tab_apps = st.tabs(
        ["Interactive ECA Lab", "3-Tier Experimental Protocol", "Real-World Applications (3)"]
    )

    with tab_lab:
        c1, c2, c3, c4, c5 = st.columns(5)
        rule = c1.selectbox("ECA Rule", [54, 110, 30, 90, 108, 254], index=0)
        init_type = c2.selectbox("Initial State", ["collision_setup", "random", "single_seed"], index=0)
        show_filter = c3.checkbox("Domain Filter (FST / R110)", value=True)
        width = c4.slider("Lattice Width (N)", 100, 300, 160, 20)
        steps = c5.slider("Time Steps (T)", 100, 250, 120, 10)

        fig, stats = m03.create_ca_dashboard_figure(
            rule_number=rule, init_type=init_type, show_filter=show_filter, width=width, steps=steps
        )
        _render_fig(fig)
        st.json(stats)

    with tab_tiers:
        tier_choice = st.radio(
            "Select Experimental Protocol Tier",
            [
                "Tier 1 (Observation): Rule 54 FST Filter OFF vs. ON & Collisions",
                "Tier 2 (Quantitative Analysis): Random Init Particle Density & Worldlines",
                "Tier 3 (Damage Spreading): Rule 90 vs. Rule 30 vs. Rule 110 Cones",
            ],
        )
        if tier_choice.startswith("Tier 1"):
            fig, res = m03.run_module03_tier1()
            _render_fig(fig)
            st.json(res)
        elif tier_choice.startswith("Tier 2"):
            fig, res = m03.run_module03_tier2()
            _render_fig(fig)
            st.json(res)
        else:
            fig, res = m03.run_module03_tier3()
            _render_fig(fig)
            st.json(res)
        _render_gif_gallery("03_Cellular_Automata_and_Particle_Computation")

    with tab_apps:
        app_choice = st.selectbox(
            "Select Module 03 Real-World Case Study",
            [
                "1. Excitable Media: 2D Belousov-Zhabotinsky Spiral Waves & Cardiac Reentry",
                "2. Traffic Flow Modeling: Rule 184 & Nagel-Schreckenberg Phantom Jams",
                "3. Neuromorphic Hardware: Decentralized Glider Parity Logic Array",
            ],
        )
        if app_choice.startswith("1."):
            fig, res = demos.demo_m03_excitable_bz_spiral()
            _render_fig(fig)
            st.metric("Final Excited Cell Fraction", f"{res['final_excited_fraction']:.3f}")
        elif app_choice.startswith("2."):
            fig, res = demos.demo_m03_traffic_flow_rule184()
            _render_fig(fig)
            st.write(f"Max Flux Rule 184: {res['max_flux_r184']:.3f} | Max Flux N-S: {res['max_flux_ns']:.3f}")
        else:
            fig, res = demos.demo_m03_neuromorphic_glider_logic()
            _render_fig(fig)
            st.json(res)


def render_module_04() -> None:
    st.header("Module 04: Evolutionary Computation, Genetic Algorithms & Edge of Chaos")
    tab_lab, tab_tiers, tab_apps = st.tabs(
        ["Interactive GA & DCT Lab", "3-Tier Experimental Protocol", "Real-World Applications (3)"]
    )

    with tab_lab:
        c1, c2, c3 = st.columns(3)
        ckpt = c1.selectbox(
            "Checkpoint Rule",
            ["Epoch 1", "Epoch 2 (Asymmetric Block Expander)", "Epoch 3/4 (GKL)"],
            index=2,
        )
        rho_0 = c2.slider("Init Density (rho_0)", 0.20, 0.80, 0.54, 0.02)
        run_ga = c3.checkbox("Run Live 12-Gen GA (Seed=7)", value=False)

        fig, stats = m04.create_evolutionary_dashboard_figure(
            checkpoint=ckpt, init_density=rho_0, run_short_ga=run_ga, seed=7
        )
        _render_fig(fig)
        m_cols = st.columns(3)
        m_cols[0].metric("Uniform U[0,1] Acc", f"{stats['fit_uniform']:.1%}")
        m_cols[1].metric("Unbiased (rho~0.5) Acc", f"{stats['fit_unbiased']:.1%}")
        m_cols[2].metric("Spacetime Final Density", f"{stats['final_density']:.2f}")

    with tab_tiers:
        tier_choice = st.radio(
            "Select Experimental Protocol Tier",
            [
                "Tier 1 (Observation): Epochs 1 -> 2 -> GKL on rho_0 = 0.54",
                "Tier 2 (Symmetry Bias): Epoch 2 at rho_0 = 0.44 vs. 0.56 & Land-Belew Curve",
                "Tier 3 (Live Evolution): 12-Gen GA (Seed=7) & Holland Schema Theorem",
            ],
        )
        if tier_choice.startswith("Tier 1"):
            fig, res = m04.run_module04_tier1(init_density=0.54)
            _render_fig(fig)
            st.json(res)
        elif tier_choice.startswith("Tier 2"):
            fig, res = m04.run_module04_tier2()
            _render_fig(fig)
            st.write(f"Epoch 2 at rho=0.44 final density: {res['epoch2_rho044_final']:.2f} | at rho=0.56: {res['epoch2_rho056_final']:.2f}")
        else:
            fig, res = m04.run_module04_tier3(seed=7)
            _render_fig(fig)
            st.json(res["canonical_schema_example"])
            st.dataframe(res["schema_records"])
        _render_gif_gallery("04_Evolutionary_Computation_and_Edge_of_Chaos")

    with tab_apps:
        app_choice = st.selectbox(
            "Select Module 04 Real-World Case Study",
            [
                "1. Computational Biology: Directed Protein Evolution on NK Landscapes",
                "2. Artificial Intelligence: AutoML Discrete Neural Architecture Search",
                "3. Swarm Robotics: Decentralized 2D Foraging Rule Evolution",
            ],
        )
        if app_choice.startswith("1."):
            fig, res = demos.demo_m04_protein_directed_evolution()
            _render_fig(fig)
            st.json(res)
        elif app_choice.startswith("2."):
            fig, res = demos.demo_m04_automl_nas()
            _render_fig(fig)
            st.json(res)
        else:
            fig, res = demos.demo_m04_swarm_robotics_foraging()
            _render_fig(fig)
            st.write(f"Gen 1 Collected: {res['gen1_collected']} | Evolved Swarm Collected: {res['evolved_collected']}")


def main() -> None:
    st.set_page_config(
        page_title="The Collective Intelligence Toolkit — Modules 01–04 Lab",
        layout="wide",
    )
    st.sidebar.title("Collective Intelligence Lab")
    st.sidebar.markdown("Interactive Complex Systems Laboratory (**Modules 01–04**)")
    page = st.sidebar.radio(
        "Navigate Module",
        [
            "Module 01: Info Theory & PID",
            "Module 02: Transfer Entropy",
            "Module 03: Cellular Automata",
            "Module 04: Evolutionary GA",
        ],
    )
    if page.startswith("Module 01"):
        render_module_01()
    elif page.startswith("Module 02"):
        render_module_02()
    elif page.startswith("Module 03"):
        render_module_03()
    else:
        render_module_04()


if __name__ == "__main__":
    main()
