"""Unified Interactive Web Dashboard (Modules 05-08) for The Collective Intelligence Toolkit."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from core import (
    adaptive_spatial_networks as m06,
    collective_markets as m07,
    rbn_immune as m05,
    real_world_demos_part2 as demos2,
    research_validation as m08,
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


def render_module_05() -> None:
    st.header("Module 05: Random Boolean Networks (RBN) & Distributed Immune Selection")
    tab_lab, tab_tiers, tab_apps = st.tabs(
        ["Interactive RBN & Immune Lab", "3-Tier Experimental Protocol", "Real-World Applications (3)"]
    )

    with tab_lab:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        mode = c1.selectbox(
            "Lab Section",
            ["1. RBN Dynamics & Derrida Curve", "2. Immune Clonal Selection & Proofreading"],
        )
        k_in = c2.slider("In-Degree K", 1, 4, 2, 1)
        bias_p = c3.slider("Bias p", 0.05, 0.95, 0.50, 0.05)
        ko_gene = c4.checkbox("Knockout Node 0 (GRN)", value=False)
        timing = c5.selectbox("Update Timing", ["Synchronous", "Asynchronous"])
        proof_n = c6.slider("Proofreading Steps N", 1, 10, 8, 1)

        fig, stats = m05.create_module05_dashboard_figure(
            mode=mode,
            k_inputs=k_in,
            bias_p=bias_p,
            knockout_gene=ko_gene,
            update_timing=timing,
            proofreading_steps=proof_n,
        )
        _render_fig(fig)
        st.json(stats)

    with tab_tiers:
        tier = st.radio(
            "Select Module 05 Experimental Protocol",
            [
                "Tier 1 (Observation): Derrida Criticality Sweep K=1..4",
                "Tier 2 (Knockout & Timing): Basin Merging & Sync vs Async",
                "Tier 3 (Immune & Proofreading): 45-Gen Affinity & N=1..10 Specificity",
            ],
        )
        if tier.startswith("Tier 1"):
            fig, res = m05.run_module05_tier1()
        elif tier.startswith("Tier 2"):
            fig, res = m05.run_module05_tier2()
        else:
            fig, res = m05.run_module05_tier3()
        _render_fig(fig)
        st.json(res)
        _render_gif_gallery("05_RBN_and_Immune_Cognition")

    with tab_apps:
        app_choice = st.selectbox(
            "Select Module 05 Real-World Case Study",
            [
                "1. Cancer Systems Biology: Oncogenic Attractor Landscape Distortion",
                "2. Immunotherapy: T-Cell Affinity & Kinetic Proofreading Specificity",
                "3. Canalizing Functions in Asynchronous Gene Regulatory Networks",
            ],
        )
        if app_choice.startswith("1."):
            fig, res = demos2.demo_m05_cancer_attractor_landscape()
        elif app_choice.startswith("2."):
            fig, res = demos2.demo_m05_immunotherapy_specificity()
        else:
            fig, res = demos2.demo_m05_canalizing_grns()
        _render_fig(fig)
        st.json(res)


def render_module_06() -> None:
    st.header("Module 06: Physarum Self-Organization, Complex Contagion & Vicsek Flocking")
    tab_lab, tab_tiers, tab_apps = st.tabs(
        ["Interactive Spatial Lab", "3-Tier Experimental Protocol", "Real-World Applications (3)"]
    )

    with tab_lab:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        mode = c1.selectbox(
            "Model Selector",
            ["Physarum Tokyo Rail", "Complex Contagion", "Vicsek Flocking"],
        )
        mu = c2.slider("Physarum Exponent mu", 0.5, 2.0, 1.4, 0.1)
        gamma = c3.slider("Physarum Decay gamma", 0.5, 1.5, 1.0, 0.1)
        disrupt = c4.checkbox("Dynamic Edge Severing", value=False)
        rewire = c5.slider("Rewiring p_rewire", 0.0, 1.0, 0.15, 0.05)
        thresh_or_eta = c6.slider("Contagion T / Noise eta", 0.2, 6.0, 2.0, 0.2)

        fig, stats = m06.create_module06_dashboard_figure(
            mode=mode,
            mu=mu,
            gamma=gamma,
            disrupt_edges=disrupt,
            rewire_prob=rewire,
            contagion_threshold=int(np.clip(round(thresh_or_eta), 1, 4)),
            vicsek_eta=float(thresh_or_eta),
        )
        _render_fig(fig)
        st.json(stats)

    with tab_tiers:
        tier = st.radio(
            "Select Module 06 Experimental Protocol",
            [
                "Tier 1 (Physarum Self-Organization): Tokyo Rail Backbone vs MST",
                "Tier 2 (Clustering vs Shortcuts): Complex Contagion Rewiring Sweep",
                "Tier 3 (Phase Transition & Live Repair): Vicsek Curve & Severing Recovery",
            ],
        )
        if tier.startswith("Tier 1"):
            fig, res = m06.run_module06_tier1()
        elif tier.startswith("Tier 2"):
            fig, res = m06.run_module06_tier2()
        else:
            fig, res = m06.run_module06_tier3()
        _render_fig(fig)
        st.json(res)
        _render_gif_gallery("06_Physarum_Contagion_and_Flocking")

    with tab_apps:
        app_choice = st.selectbox(
            "Select Module 06 Real-World Case Study",
            [
                "1. Resilient Transit & Power Grid: Live Severing & Non-Uniform Demand",
                "2. Public Health Diffusion: Clustered Communities & Seeding Optimization",
                "3. Decentralized Autonomous Drone Swarm Obstacle Avoidance",
            ],
        )
        if app_choice.startswith("1."):
            fig, res = demos2.demo_m06_resilient_transit_power_grid()
            _render_fig(fig)
            st.json(res)
        elif app_choice.startswith("2."):
            fig, res = demos2.demo_m06_public_health_diffusion()
            _render_fig(fig)
            st.json(res)
        else:
            fig, res = demos2.demo_m06_autonomous_drone_swarm()
            _render_fig(fig)
            st.json({"low_noise_phi": res["low_noise_phi"], "high_noise_phi": res["high_noise_phi"]})


def render_module_07() -> None:
    st.header("Module 07: Collective Wisdom, Bounded Confidence & El Farol Market")
    tab_lab, tab_tiers, tab_apps = st.tabs(
        ["Interactive Collective Lab", "3-Tier Experimental Protocol", "Real-World Applications (3)"]
    )

    with tab_lab:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        study = c1.selectbox(
            "3-Model Selector",
            [
                "1. Diversity Prediction Theorem",
                "2. Bounded Confidence (HK vs DeGroot)",
                "3. El Farol Bar & Minority Game",
            ],
        )
        rho = c2.slider("Param (rho)", 0.05, 0.90, 0.25, 0.05)
        eps = c3.slider("Param (eps)", 0.05, 0.45, 0.18, 0.01)
        mem_m = c4.slider("Memory M", 1, 6, 3, 1)
        herd_q = c5.slider("Social Herding q", 0.0, 0.70, 0.35, 0.05)
        herd_on = c6.checkbox("Enable Networked Herding Extension", value=False)

        fig, stats = m07.create_module07_dashboard_figure(
            study=study, rho=rho, eps=eps, memory_m=mem_m, herding_q=herd_q, herding_ext=herd_on
        )
        _render_fig(fig)
        st.json(stats)

    with tab_tiers:
        tier = st.radio(
            "Select Module 07 Experimental Protocol",
            [
                "Tier 1 (Diversity & Correlation): Page's Exact Identity & rho Sweep",
                "Tier 2 (Opinion Bifurcation): HK Presets (0.05, 0.18, 0.40) vs DeGroot",
                "Tier 3 (Inductive Market & Herding): El Farol, MG Cusp alpha_c & Herding",
            ],
        )
        if tier.startswith("Tier 1"):
            fig, res = m07.run_module07_tier1()
        elif tier.startswith("Tier 2"):
            fig, res = m07.run_module07_tier2()
        else:
            fig, res = m07.run_module07_tier3()
        _render_fig(fig)
        st.json(res)
        _render_gif_gallery("07_Collective_Wisdom_Opinion_and_El_Farol")

    with tab_apps:
        app_choice = st.selectbox(
            "Select Module 07 Real-World Case Study",
            [
                "1. Financial Market Microstructure: Liquidity Droughts & Flash Crashes",
                "2. Forecasting & Prediction Markets: Decorrelating Ensemble Judgments",
                "3. Multi-Agent AI & LLM Swarms: Debate & Sycophancy Protocols",
            ],
        )
        if app_choice.startswith("1."):
            fig, res = demos2.demo_m07_financial_limit_order_book()
        elif app_choice.startswith("2."):
            fig, res = demos2.demo_m07_forecasting_delphi_ensemble()
        else:
            fig, res = demos2.demo_m07_llm_multi_agent_debate()
        _render_fig(fig)
        st.json(res)


def render_module_08() -> None:
    st.header("Module 08: Computational Research Design, Paper Audits & Validation Lab")
    tab_lab, tab_tiers, tab_brief, tab_apps = st.tabs(
        ["Validation Lab", "3-Tier Protocol", "5-Part Research Brief Builder", "Real-World Applications (3)"]
    )

    with tab_lab:
        c1, c2, c3, c4 = st.columns(4)
        paper_choice = c1.selectbox("Paper Audit (6 Classics)", list(m08.PAPER_AUDITS.keys()))
        track = c2.selectbox("Validation Track", ["Track A: Observational FDR & Power", "Track B: Generative ABM vs Null Model"])
        sample_t = c3.slider("Sample T", 100, 1200, 300, 50)
        alpha = c4.selectbox("FDR alpha", [0.01, 0.05, 0.10], index=1)

        fig, stats = m08.create_module08_dashboard_figure(
            paper_choice=paper_choice, track=track, sample_size=sample_t, alpha=alpha
        )
        _render_fig(fig)
        audit = m08.PAPER_AUDITS[paper_choice]
        col_s, col_c = st.columns(2)
        with col_s:
            st.subheader("Setup (Q1–Q4)")
            for line in audit["setup"]:
                st.write(f"- {line}")
        with col_c:
            st.subheader("Critique (Q5–Q8)")
            for line in audit["critique"]:
                st.write(f"- {line}")

    with tab_tiers:
        tier = st.radio(
            "Select Module 08 Experimental Protocol",
            [
                "Tier 1 (Paper Audit Inspection): Q7 Limitations -> Q8 Extensions Matrix",
                "Tier 2 (Track A Power & FDR): Solving Minimum Sample Size T_80",
                "Tier 3 (Track B Null Baseline): Cohen's d Separation & Sample Brief",
            ],
        )
        if tier.startswith("Tier 1"):
            fig, res = m08.run_module08_tier1()
        elif tier.startswith("Tier 2"):
            fig, res = m08.run_module08_tier2()
        else:
            fig, res = m08.run_module08_tier3()
        _render_fig(fig)
        st.json({k: v for k, v in res.items() if k != "sample_brief"})
        _render_gif_gallery("08_Research_Design_and_Validation_Lab")

    with tab_brief:
        st.subheader("Interactive 1-Page 5-Part Research Brief Builder")
        paper_key = st.selectbox("Anchor Paper", list(m08.PAPER_AUDITS.keys()), key="brief_paper")
        audit = m08.PAPER_AUDITS[paper_key]
        title = st.text_input("Project Title", f"Extending {audit['short']}: {audit['q8_summary']}")
        gap = st.text_area("1. Gap Analysis (Q7)", audit["q7_summary"])
        hyp = st.text_area("2. Falsifiable Hypothesis (Q8)", f"Test whether {audit['q8_summary']} improves stability/specificity over baseline.")
        arch = st.text_area("3. Minimal Model Architecture", f"Minimal {audit['module']} simulator with single-parameter control sweep.")
        val = st.text_area("4. Null Baseline & Sensitivity Plan", "Track A BH-FDR power >= 80% (T >= T_80) and Track B null comparison with Cohen's d > 1.2.")
        claim = st.text_area("5. Expected Defensible Claim", "Quantified separation against randomized null baseline with controlled false discovery below 5%.")
        brief = m08.build_research_brief(title, paper_key, gap, hyp, arch, val, claim)
        st.markdown(brief["markdown"])
        st.download_button("Download Brief (Markdown)", data=brief["markdown"], file_name="research_brief.md")
        st.download_button("Download Brief (JSON)", data=brief["json"], file_name="research_brief.json")

    with tab_apps:
        app_choice = st.selectbox(
            "Select Module 08 Real-World Case Study",
            [
                "1. Q1 Peer-Review & Reproducibility Stress-Testing Suite",
                "2. Industrial R&D Experimental Power & Null-Model Simulator",
                "3. Cross-Module Grand Synthesis Research Brief Pipeline",
            ],
        )
        if app_choice.startswith("1."):
            fig, res = demos2.demo_m08_q1_peer_review_stress_test()
        elif app_choice.startswith("2."):
            fig, res = demos2.demo_m08_industrial_rd_power_pipeline()
        else:
            fig, res = demos2.demo_m08_grand_synthesis_pipeline()
        _render_fig(fig)
        st.json({k: v for k, v in res.items() if k != "brief"})


def main() -> None:
    st.set_page_config(
        page_title="The Collective Intelligence Toolkit — Modules 05–08 Lab",
        layout="wide",
    )
    st.sidebar.title("Collective Intelligence Lab (Part 2)")
    st.sidebar.markdown("Interactive Complex Systems Laboratory (**Modules 05–08**)")
    page = st.sidebar.radio(
        "Navigate Module",
        [
            "Module 05: RBN & Immune",
            "Module 06: Physarum & Swarms",
            "Module 07: Markets & Opinions",
            "Module 08: Research Validation",
        ],
    )
    if page.startswith("Module 05"):
        render_module_05()
    elif page.startswith("Module 06"):
        render_module_06()
    elif page.startswith("Module 07"):
        render_module_07()
    else:
        render_module_08()


if __name__ == "__main__":
    main()
