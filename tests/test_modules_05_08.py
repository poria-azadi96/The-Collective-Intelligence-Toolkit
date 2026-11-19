"""Pytest unit and integration tests for Modules 05-08 (Part 2)."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from core import (
    adaptive_spatial_networks as m06,
    collective_markets as m07,
    rbn_immune as m05,
    real_world_demos_part2 as demos2,
    research_validation as m08,
)


def test_module05_rbn_and_immune() -> None:
    assert abs(m05.derrida_sensitivity(2, 0.50) - 1.0) < 1e-9
    w, tt = m05.generate_rbn(10, 2, 0.50, seed=42)
    _, basins = m05.analyze_exact_state_space(w, tt)
    assert sum(basins) == 1024

    kp = m05.kinetic_proofreading_specificity(10)
    assert kp["specificity_gain"][-1] > kp["specificity_gain"][0] * 100.0


def test_module06_physarum_contagion_vicsek() -> None:
    coords, edges, adj_len = m06.build_tokyo_benchmark_graph()
    res = m06.solve_physarum_network(coords, edges, adj_len, mu=1.4, steps=80, disrupt_hub_edges=True, seed=42)
    assert res["disrupted_mask"].sum() == 2

    c_low = m06.simulate_contagion_comparison(rewire_prob=0.02, complex_threshold=2, seed=42)
    c_hi = m06.simulate_contagion_comparison(rewire_prob=0.50, complex_threshold=2, seed=42)
    assert c_low["final_complex"] >= c_hi["final_complex"]

    v_lo = m06.simulate_vicsek_flocking(noise_eta=0.3, steps=70, seed=42)
    v_hi = m06.simulate_vicsek_flocking(noise_eta=4.5, steps=70, seed=42)
    assert v_lo["mean_tail_phi"] > v_hi["mean_tail_phi"]


def test_module07_dpt_hk_markets() -> None:
    dpt = m07.simulate_dpt_ensemble(n_agents=50, indiv_sigma=10.0, error_corr=0.25, seed=42)
    assert dpt["identity_residual"] < 1e-10

    _, c_frag = m07.simulate_hk_opinions(80, confidence_eps=0.05, seed=42)
    _, c_cons = m07.simulate_hk_opinions(80, confidence_eps=0.40, seed=42)
    assert c_frag > c_cons and c_cons == 1

    mg_ind = m07.simulate_minority_game(101, memory_m=6, herding_coupling=0.0, seed=42)
    mg_hrd = m07.simulate_minority_game(101, memory_m=6, herding_coupling=0.45, seed=42)
    assert mg_ind["volatility"] < 1.0 < mg_hrd["volatility"]


def test_module08_audits_power_and_cohen_d() -> None:
    assert len(m08.PAPER_AUDITS) == 6
    cal = m08.calibrate_fdr_power_curve(50, 8, signal_snr=1.05, alpha=0.05, seed=42)
    assert cal["T_80"] is not None

    tb = m08.simulate_abm_vs_null(coupling_strength=0.75, n_seeds=20, seed=42)
    assert tb["focused_cohen_d"] > 1.5


def test_all_12_part2_real_world_demos_run() -> None:
    for fn in (
        demos2.demo_m05_cancer_attractor_landscape,
        demos2.demo_m05_immunotherapy_specificity,
        demos2.demo_m05_canalizing_grns,
        demos2.demo_m06_resilient_transit_power_grid,
        demos2.demo_m06_public_health_diffusion,
        demos2.demo_m06_autonomous_drone_swarm,
        demos2.demo_m07_financial_limit_order_book,
        demos2.demo_m07_forecasting_delphi_ensemble,
        demos2.demo_m07_llm_multi_agent_debate,
        demos2.demo_m08_q1_peer_review_stress_test,
        demos2.demo_m08_industrial_rd_power_pipeline,
        demos2.demo_m08_grand_synthesis_pipeline,
    ):
        fig, _ = fn()
        plt.close(fig)


def test_streamlit_app_part2_renders_all_modules() -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO_ROOT / "app_part2.py"), default_timeout=60)
    at.run()
    assert not at.exception
    for option in [
        "Module 05: RBN & Immune",
        "Module 06: Physarum & Swarms",
        "Module 07: Markets & Opinions",
        "Module 08: Research Validation",
    ]:
        at.sidebar.radio[0].set_value(option).run()
        assert not at.exception


def test_notebooks_05_to_08_headless_import() -> None:
    shim = types.ModuleType("ipywidgets")
    shim.interact = lambda *a, **kw: None
    shim.Dropdown = lambda *a, **kw: None
    shim.FloatSlider = lambda *a, **kw: None
    shim.IntSlider = lambda *a, **kw: None
    shim.Checkbox = lambda *a, **kw: None
    shim.fixed = lambda *a, **kw: None
    old = sys.modules.get("ipywidgets")
    sys.modules["ipywidgets"] = shim
    try:
        for nb_name in (
            "05_rbn_and_immune_cognition.ipynb",
            "06_physarum_and_contagion.ipynb",
            "06_adaptive_spatial_networks.ipynb",
            "07_el_farol_and_opinion_dynamics.ipynb",
            "08_research_design_lab.ipynb",
        ):
            nb_path = REPO_ROOT / "notebooks" / nb_name
            nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
            ns = {"__name__": f"test_{nb_path.stem}"}
            for cell in nb_data["cells"]:
                if cell["cell_type"] == "code":
                    src = "".join(cell["source"])
                    exec(compile(src, str(nb_path), "exec"), ns)
    finally:
        if old is not None:
            sys.modules["ipywidgets"] = old
        else:
            sys.modules.pop("ipywidgets", None)
