"""Build or regenerate interactive Jupyter/Colab notebooks for Modules 05-08."""

from __future__ import annotations

from pathlib import Path
import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / "notebooks"

BOOTSTRAP_CODE = """import subprocess
import sys
from pathlib import Path

def _ensure_toolkit_on_path():
    cwd = Path.cwd().resolve()
    for cand in (cwd, cwd.parent, cwd.parent.parent):
        if (cand / "core").is_dir():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    clone_dir = cwd / "The-Collective-Intelligence-Toolkit"
    if not clone_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/poria-azadi96/The-Collective-Intelligence-Toolkit.git", str(clone_dir)],
            check=True,
        )
    if str(clone_dir) not in sys.path:
        sys.path.insert(0, str(clone_dir))
    return clone_dir

REPO_ROOT = _ensure_toolkit_on_path()
"""


def _make_nb(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    }
    nb.cells = [
        nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text)
        for kind, text in cells
    ]
    return nb


def build_module_05_notebook() -> nbf.NotebookNode:
    md_intro = """# Module 05: Random Boolean Networks (RBN) & Distributed Immune Selection

**Core Mathematical & Biological Framework:**
- **Kauffman RBNs & Derrida Sensitivity:** $s = 2Kp(1 - p)$ separating **Ordered** ($s < 1$), **Critical** ($s = 1$), and **Chaotic** ($s > 1$) dynamics across $2^N$ cell-fate attractors.
- **Update Timing & Knockouts:** Synchronous deterministic cycles vs. Greil & Drossel asynchronous attractors and gene knockout basin merging.
- **Distributed Immune Cognition:** Clonal affinity maturation (45 generations) & McKeithan multistep kinetic proofreading ($N = 1 \\dots 10$)."""

    code_setup = BOOTSTRAP_CODE + """import matplotlib.pyplot as plt
import ipywidgets as widgets
from core.rbn_immune import (
    create_module05_dashboard_figure,
    run_module05_tier1,
    run_module05_tier2,
    run_module05_tier3,
)
from core.real_world_demos_part2 import (
    demo_m05_cancer_attractor_landscape,
    demo_m05_canalizing_grns,
    demo_m05_immunotherapy_specificity,
)

def run_module05_lab(
    mode='1. RBN Dynamics & Derrida Curve',
    k_inputs=2,
    bias_p=0.50,
    knockout_gene=False,
    update_timing='Synchronous',
    proofreading_steps=8,
):
    fig, stats = create_module05_dashboard_figure(
        mode=mode,
        k_inputs=k_inputs,
        bias_p=bias_p,
        knockout_gene=knockout_gene,
        update_timing=update_timing,
        proofreading_steps=proofreading_steps,
    )
    plt.show()
    if stats['mode'] == 'RBN':
        print(
            f"Regime: {stats['regime']} | Sensitivity s = {stats['sensitivity_s']:.3f} | "
            f"Sync Attractors = {stats['n_attractors']} | Async Fixed = {stats['async_fixed_fraction']:.1%}"
        )
    else:
        print(
            f"Final Mean Clonal Affinity = {stats['final_mean_affinity']:.3f} | "
            f"Proofreading Gain (N={proofreading_steps}) = {stats['selected_proofreading_gain']:.1f}x"
        )

widgets.interact(
    run_module05_lab,
    mode=widgets.Dropdown(options=['1. RBN Dynamics & Derrida Curve', '2. Immune Clonal Selection & Proofreading'], value='1. RBN Dynamics & Derrida Curve', description='Lab Section:'),
    k_inputs=widgets.IntSlider(min=1, max=4, step=1, value=2, description='In-Degree K:'),
    bias_p=widgets.FloatSlider(min=0.05, max=0.95, step=0.05, value=0.50, description='Bias p:'),
    knockout_gene=widgets.Checkbox(value=False, description='Knockout Node 0 (GRN)'),
    update_timing=widgets.Dropdown(options=['Synchronous', 'Asynchronous'], value='Synchronous', description='Timing:'),
    proofreading_steps=widgets.IntSlider(min=1, max=10, step=1, value=8, description='Proofread N:'),
);"""

    md_tiers = """## 3-Tier Experimental Protocol
- **Tier 1 (Observation):** Sweep $K = 1 \\dots 4$ at $p = 0.50$ and watch the Derrida curve cross the $45^\\circ$ line as $s$ exceeds $1$.
- **Tier 2 (Knockout & Timing):** Toggle `Knockout Node 0` at $K = 2$ to observe basin merging; contrast Synchronous Attractors vs. Asynchronous point-attractor convergence.
- **Tier 3 (Immune & Proofreading):** Inspect 45-generation clonal affinity maturation and exponential discrimination gain across $N = 1 \\dots 10$."""

    code_tiers = """def run_m05_tier_selector(tier='Tier 1: Derrida Sweep K=1..4'):
    if tier.startswith('Tier 1'):
        fig, res = run_module05_tier1()
    elif tier.startswith('Tier 2'):
        fig, res = run_module05_tier2()
    else:
        fig, res = run_module05_tier3()
    plt.show()
    print(res)

widgets.interact(
    run_m05_tier_selector,
    tier=widgets.Dropdown(
        options=[
            'Tier 1: Derrida Sweep K=1..4',
            'Tier 2: Knockout Basin Merging & Sync vs Async',
            'Tier 3: Immune Clonal Maturation & Proofreading',
        ],
        value='Tier 1: Derrida Sweep K=1..4',
        description='Protocol:',
    ),
);"""

    md_apps = """## Module 05 Real-World Applications
1. **Cancer Systems Biology & Oncology:** Oncogenic mutations distorting cell-fate basins.
2. **Immunotherapy & Affinity Optimization:** Maximizing tumor targeting while suppressing autoimmune cross-reactivity.
3. **Canalizing Functions in Asynchronous GRNs:** Stabilizing biological regulatory attractors."""

    code_apps = """def run_m05_real_world_demo(application='1. Cancer Attractor Landscape'):
    if application.startswith('1.'):
        fig, res = demo_m05_cancer_attractor_landscape()
    elif application.startswith('2.'):
        fig, res = demo_m05_immunotherapy_specificity()
    else:
        fig, res = demo_m05_canalizing_grns()
    plt.show()
    print(res)

widgets.interact(
    run_m05_real_world_demo,
    application=widgets.Dropdown(
        options=[
            '1. Cancer Attractor Landscape',
            '2. Immunotherapy Specificity & Proofreading',
            '3. Canalizing Functions in Async GRNs',
        ],
        value='1. Cancer Attractor Landscape',
        description='Case Study:',
    ),
);"""

    return _make_nb([("md", md_intro), ("code", code_setup), ("md", md_tiers), ("code", code_tiers), ("md", md_apps), ("code", code_apps)])


def build_module_06_notebook() -> nbf.NotebookNode:
    md_intro = """# Module 06: Physarum Self-Organization, Complex Contagion & Vicsek Collective Motion

**Core Mathematical Models:**
- **Tero et al. (2010) *Physarum* Sparse-Laplacian Solver:** Poiseuille flux $Q_{ij} = \\frac{D_{ij}}{L_{ij}}(p_i - p_j)$, Kirchhoff conservation $\\mathbf{L}(D)\\mathbf{p} = \\mathbf{b}$, and $\\frac{dD_{ij}}{dt} = \\frac{|Q_{ij}|^\\mu}{1 + |Q_{ij}|^\\mu} - \\gamma D_{ij}$.
- **Complex Contagion (Centola, 2010):** Threshold reinforcement ($T \\ge 2$) on clustered Watts–Strogatz graphs vs. random shortcut collapse.
- **Vicsek Collective Flocking:** 2D heading alignment with global order parameter $\\phi(t) = \\frac{1}{N v_0}\\|\\sum \\mathbf{v}_i(t)\\|$."""

    code_setup = BOOTSTRAP_CODE + """import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from core.adaptive_spatial_networks import (
    create_module06_dashboard_figure,
    run_module06_tier1,
    run_module06_tier2,
    run_module06_tier3,
)
from core.real_world_demos_part2 import (
    demo_m06_autonomous_drone_swarm,
    demo_m06_public_health_diffusion,
    demo_m06_resilient_transit_power_grid,
)

def run_module06_lab(
    mode='1. Physarum Tokyo Rail Solver',
    gamma_or_noise=1.4,
    disrupt_or_rewire=False,
    gamma_decay=1.0,
    contagion_threshold=2,
):
    if 'Physarum' in mode:
        m_key, mu_v, rew_v, eta_v = 'Physarum Tokyo Rail', gamma_or_noise, 0.15, 0.6
    elif 'Contagion' in mode:
        m_key, mu_v, rew_v, eta_v = 'Complex Contagion', 1.4, (0.35 if disrupt_or_rewire else 0.02), 0.6
    else:
        m_key, mu_v, rew_v, eta_v = 'Vicsek Flocking', 1.4, 0.15, min(max(gamma_or_noise * 2.0, 0.1), 5.5)

    fig, stats = create_module06_dashboard_figure(
        mode=m_key,
        mu=mu_v,
        gamma=gamma_decay,
        disrupt_edges=disrupt_or_rewire,
        rewire_prob=rew_v,
        contagion_threshold=contagion_threshold,
        vicsek_eta=eta_v,
    )
    plt.show()
    print(f'Module 06 Stats: {stats}')

widgets.interact(
    run_module06_lab,
    mode=widgets.Dropdown(options=['1. Physarum Tokyo Rail Solver', '2. Simple vs. Complex Contagion', '3. Vicsek Collective Flocking'], value='1. Physarum Tokyo Rail Solver', description='Model:'),
    gamma_or_noise=widgets.FloatSlider(min=0.5, max=2.2, step=0.1, value=1.4, description='mu / Noise:'),
    disrupt_or_rewire=widgets.Checkbox(value=False, description='Sever Edges / Rewire'),
    gamma_decay=widgets.FloatSlider(min=0.5, max=1.5, step=0.1, value=1.0, description='Decay gamma:'),
    contagion_threshold=widgets.IntSlider(min=1, max=4, step=1, value=2, description='Threshold T:'),
);"""

    md_tiers = """## 3-Tier Experimental Protocol
- **Tier 1 (*Physarum* Self-Organization):** Observe automatic pruning of redundant planar corridors vs. MST cost/efficiency/fault-tolerance.
- **Tier 2 (Clustering vs. Shortcuts):** Sweep $p_{\\text{rewire}} \\in [0, 1]$ on Complex Contagion ($T \\ge 2$) to observe clustering loss.
- **Tier 3 (Phase Transition & Live Repair):** Locate Vicsek $\\phi(\\eta)$ alignment transition and test live *Physarum* recovery after trunk severing."""

    code_tiers = """def run_m06_tier_selector(tier='Tier 1: Physarum Tokyo Rail Backbone'):
    if tier.startswith('Tier 1'):
        fig, res = run_module06_tier1()
    elif tier.startswith('Tier 2'):
        fig, res = run_module06_tier2()
    else:
        fig, res = run_module06_tier3()
    plt.show()
    print(res)

widgets.interact(
    run_m06_tier_selector,
    tier=widgets.Dropdown(
        options=[
            'Tier 1: Physarum Tokyo Rail Backbone',
            'Tier 2: Complex Contagion Rewiring Sweep',
            'Tier 3: Vicsek Phase Curve & Live Edge Severing',
        ],
        value='Tier 1: Physarum Tokyo Rail Backbone',
        description='Protocol:',
    ),
);"""

    md_apps = """## Module 06 Real-World Applications
1. **Resilient Transit & Power Grid Design:** Self-healing hydraulic routing under live severing & non-uniform demand.
2. **Public Health & Innovation Diffusion:** Clustered seeding strategies for high-threshold behavioral adoption.
3. **Decentralized Autonomous Drone Swarms:** Obstacle-avoidance flocking under noisy communications."""

    code_apps = """def run_m06_real_world_demo(application='1. Resilient Transit & Power Grid'):
    if application.startswith('1.'):
        fig, res = demo_m06_resilient_transit_power_grid()
    elif application.startswith('2.'):
        fig, res = demo_m06_public_health_diffusion()
    else:
        fig, res = demo_m06_autonomous_drone_swarm()
        res = {k: v for k, v in res.items() if not k.endswith('_sim')}
    plt.show()
    print(res)

widgets.interact(
    run_m06_real_world_demo,
    application=widgets.Dropdown(
        options=[
            '1. Resilient Transit & Power Grid',
            '2. Public Health & Innovation Diffusion',
            '3. Autonomous Drone Swarm Navigation',
        ],
        value='1. Resilient Transit & Power Grid',
        description='Case Study:',
    ),
);"""

    return _make_nb([("md", md_intro), ("code", code_setup), ("md", md_tiers), ("code", code_tiers), ("md", md_apps), ("code", code_apps)])


def build_module_07_notebook() -> nbf.NotebookNode:
    md_intro = """# Module 07: Collective Wisdom, Bounded Confidence & El Farol Market

**Core Mathematical & Socio-Economic Framework:**
- **Scott Page's Diversity Prediction Theorem (DPT):** $(\\bar{c} - \\theta)^2 = \\frac{1}{N}\\sum (c_i - \\theta)^2 - \\frac{1}{N}\\sum (c_i - \\bar{c})^2$
- **Hegselmann–Krause Bounded Confidence:** $x_i(t+1) = \\frac{1}{|N_i(t)|}\\sum_{j \\in N_i(t)} x_j(t)$ vs. DeGroot linear consensus.
- **Arthur (1994) El Farol & Challet–Zhang (1997) Minority Game:** Control parameter $\\alpha = 2^M / N$, critical cusp $\\alpha_c$, and Networked Herding ($q$)."""

    code_setup = BOOTSTRAP_CODE + """import matplotlib.pyplot as plt
import ipywidgets as widgets
from core.collective_markets import (
    create_module07_dashboard_figure,
    run_module07_tier1,
    run_module07_tier2,
    run_module07_tier3,
)
from core.real_world_demos_part2 import (
    demo_m07_financial_limit_order_book,
    demo_m07_forecasting_delphi_ensemble,
    demo_m07_llm_multi_agent_debate,
)

def run_module07_lab(
    study='1. Diversity Prediction Theorem',
    param_val=0.25,
    herding_ext=False,
    memory_m=3,
    herding_q=0.35,
):
    mem_eff = int(round(2 + param_val * 4)) if 'El Farol' in study and memory_m == 3 else memory_m
    fig, stats = create_module07_dashboard_figure(
        study=study,
        rho=param_val,
        eps=max(0.04, min(0.50, param_val)),
        memory_m=mem_eff,
        herding_q=herding_q,
        herding_ext=herding_ext,
    )
    plt.show()
    print(f'Module 07 Stats: {stats}')

widgets.interact(
    run_module07_lab,
    study=widgets.Dropdown(options=['1. Diversity Prediction Theorem', '2. Bounded Confidence (HK vs DeGroot)', '3. El Farol Bar & Minority Game'], value='1. Diversity Prediction Theorem', description='Model Lab:'),
    param_val=widgets.FloatSlider(min=0.05, max=0.90, step=0.05, value=0.25, description='Param (rho/eps):'),
    herding_ext=widgets.Checkbox(value=False, description='Networked Herding'),
    memory_m=widgets.IntSlider(min=1, max=6, step=1, value=3, description='Memory M:'),
    herding_q=widgets.FloatSlider(min=0.0, max=0.70, step=0.05, value=0.35, description='Herding q:'),
);"""

    md_tiers = """## 3-Tier Experimental Protocol
- **Tier 1 (Diversity & Correlation):** Sweep $\\rho \\in [0.05, 0.90]$ and confirm Page's single-trial algebraic identity.
- **Tier 2 (Opinion Bifurcation):** Contrast Hegselmann–Krause presets $\\epsilon \\in \\{0.05, 0.18, 0.40\\}$ against DeGroot consensus.
- **Tier 3 (Inductive Market & Herding):** Locate Minority Game cusp $\\alpha_c$ and toggle Networked Herding to induce excess volatility."""

    code_tiers = """def run_m07_tier_selector(tier='Tier 1: DPT Identity & Correlation Sweep'):
    if tier.startswith('Tier 1'):
        fig, res = run_module07_tier1()
    elif tier.startswith('Tier 2'):
        fig, res = run_module07_tier2()
    else:
        fig, res = run_module07_tier3()
    plt.show()
    print(res)

widgets.interact(
    run_m07_tier_selector,
    tier=widgets.Dropdown(
        options=[
            'Tier 1: DPT Identity & Correlation Sweep',
            'Tier 2: HK Opinion Bifurcation vs DeGroot',
            'Tier 3: El Farol, MG Cusp & Herding Spike',
        ],
        value='Tier 1: DPT Identity & Correlation Sweep',
        description='Protocol:',
    ),
);"""

    md_apps = """## Module 07 Real-World Applications
1. **Financial Market Microstructure:** Chartist vs. fundamentalist order book generating liquidity droughts & flash crashes.
2. **Forecasting & Prediction Markets:** Delphi decorrelation & diversity weighting against groupthink.
3. **Multi-Agent AI & LLM Swarms:** Debate protocols balancing cognitive diversity against sycophantic convergence."""

    code_apps = """def run_m07_real_world_demo(application='1. Financial Limit Order Book'):
    if application.startswith('1.'):
        fig, res = demo_m07_financial_limit_order_book()
    elif application.startswith('2.'):
        fig, res = demo_m07_forecasting_delphi_ensemble()
    else:
        fig, res = demo_m07_llm_multi_agent_debate()
    plt.show()
    print(res)

widgets.interact(
    run_m07_real_world_demo,
    application=widgets.Dropdown(
        options=[
            '1. Financial Limit Order Book',
            '2. Forecasting & Delphi Decorrelation',
            '3. Multi-Agent LLM Debate Protocols',
        ],
        value='1. Financial Limit Order Book',
        description='Case Study:',
    ),
);"""

    return _make_nb([("md", md_intro), ("code", code_setup), ("md", md_tiers), ("code", code_tiers), ("md", md_apps), ("code", code_apps)])


def build_module_08_notebook() -> nbf.NotebookNode:
    md_intro = """# Module 08: Computational Research Design, Paper Audits & Validation Lab

**Core Validation Pipeline:** $\\text{Question} \\to \\text{Model & Null Baseline} \\to \\text{Sensitivity & BH-FDR Test} \\to \\text{Defensible Claim}$
- **Track A (BH-FDR Power Calibration):** Solve minimum sample size $T_{80}$ for $80\\%$ detection power under Benjamini–Hochberg FDR control.
- **Track B (Null Baseline & Effect Size):** Standardized separation Cohen's $d = \\frac{\\mu_{\\text{mech}} - \\mu_{\\text{null}}}{s_{\\text{pooled}}}$.
- **8-Question Paper Audit Protocol & 5-Part Research Brief Builder** across all 6 workshop classics."""

    code_setup = BOOTSTRAP_CODE + """import matplotlib.pyplot as plt
import ipywidgets as widgets
from core.research_validation import (
    PAPER_AUDITS,
    build_research_brief,
    create_module08_dashboard_figure,
    run_module08_tier1,
    run_module08_tier2,
    run_module08_tier3,
    simulate_fdr_validation as _core_sim_fdr,
)
from core.real_world_demos_part2 import (
    demo_m08_grand_synthesis_pipeline,
    demo_m08_industrial_rd_power_pipeline,
    demo_m08_q1_peer_review_stress_test,
)

def simulate_fdr_validation(n_tests=50, true_edges=8, signal_snr=0.55, sample_size=300, seed=42):
    res = _core_sim_fdr(n_tests=n_tests, true_edges=true_edges, signal_snr=signal_snr, sample_size=sample_size, n_mc=60, seed=seed)
    return (res['raw_tpr'], res['raw_fdp']), (res['fdr_tpr'], res['fdr_fdp'])

def run_module08_lab(
    paper_choice='1. Schreiber (2000) - Transfer Entropy',
    track='Track A: Observational FDR & Power',
    sample_size=300,
    alpha=0.05,
):
    fig, stats = create_module08_dashboard_figure(
        paper_choice=paper_choice,
        track=track,
        sample_size=sample_size,
        alpha=alpha,
    )
    plt.show()
    audit = PAPER_AUDITS[paper_choice]
    print('=' * 88)
    print(f'8-QUESTION PAPER AUDIT PROTOCOL | {paper_choice}')
    print('=' * 88)
    for line in audit['setup'] + audit['critique']:
        print(line)
    print('=' * 88)
    print('Validation Metrics:', {k: v for k, v in stats.items() if k != 'calibration'})

widgets.interact(
    run_module08_lab,
    paper_choice=widgets.Dropdown(options=list(PAPER_AUDITS.keys()), value='1. Schreiber (2000) - Transfer Entropy', description='Paper Audit:'),
    track=widgets.Dropdown(options=['Track A: Observational FDR & Power', 'Track B: Generative ABM vs Null Model'], value='Track A: Observational FDR & Power', description='Validation:'),
    sample_size=widgets.IntSlider(min=100, max=1200, step=100, value=300, description='Sample T:'),
    alpha=widgets.Dropdown(options=[0.01, 0.05, 0.10], value=0.05, description='FDR alpha:'),
);"""

    md_tiers = """## 3-Tier Experimental Protocol
- **Tier 1 (Paper Audit Inspection):** Visualize how Q7 (Limitations) directly generates Q8 (Open Research Question) across all 6 Classics.
- **Tier 2 (Track A Calibration):** Solve minimum sample size $T_{80}$ reaching $80\\%$ power under BH-FDR.
- **Tier 3 (Track B Null Baseline & Research Brief):** Verify Cohen's $d$ separation and generate a 1-page 5-Part Research Brief."""

    code_tiers = """def run_m08_tier_selector(tier='Tier 1: Paper Audit Q7 -> Q8 Matrix'):
    if tier.startswith('Tier 1'):
        fig, res = run_module08_tier1()
    elif tier.startswith('Tier 2'):
        fig, res = run_module08_tier2()
    else:
        fig, res = run_module08_tier3()
        print(res['sample_brief']['markdown'])
        res = {'focused_cohen_d': res['focused_cohen_d'], 'max_cohen_d': res['max_cohen_d']}
    plt.show()
    print(res)

widgets.interact(
    run_m08_tier_selector,
    tier=widgets.Dropdown(
        options=[
            'Tier 1: Paper Audit Q7 -> Q8 Matrix',
            'Tier 2: Track A Power Calibration (T_80)',
            'Tier 3: Track B Null Baseline & Research Brief',
        ],
        value='Tier 1: Paper Audit Q7 -> Q8 Matrix',
        description='Protocol:',
    ),
);"""

    md_apps = """## Module 08 Real-World Applications
1. **Q1 Peer-Review Stress-Testing Suite:** Reproducing Q7 failure modes across all workshop models.
2. **Industrial R&D Power & Null-Model Simulator:** Pre-experiment $T_{80}$ & Cohen's $d$ validation pipeline.
3. **Cross-Module Grand Synthesis Report:** Unified reproducibility scorecard across Modules 01–07."""

    code_apps = """def run_m08_real_world_demo(application='1. Q1 Peer-Review Stress-Test Suite'):
    if application.startswith('1.'):
        fig, res = demo_m08_q1_peer_review_stress_test()
    elif application.startswith('2.'):
        fig, res = demo_m08_industrial_rd_power_pipeline()
    else:
        fig, res = demo_m08_grand_synthesis_pipeline()
        res = {'pillar_scores': res['pillar_scores']}
    plt.show()
    print(res)

widgets.interact(
    run_m08_real_world_demo,
    application=widgets.Dropdown(
        options=[
            '1. Q1 Peer-Review Stress-Test Suite',
            '2. Industrial R&D Power & Null Simulator',
            '3. Cross-Module Grand Synthesis Pipeline',
        ],
        value='1. Q1 Peer-Review Stress-Test Suite',
        description='Case Study:',
    ),
);"""

    return _make_nb([("md", md_intro), ("code", code_setup), ("md", md_tiers), ("code", code_tiers), ("md", md_apps), ("code", code_apps)])


def main() -> None:
    builders = {
        "05_rbn_and_immune_cognition.ipynb": build_module_05_notebook,
        "06_physarum_and_contagion.ipynb": build_module_06_notebook,
        "06_adaptive_spatial_networks.ipynb": build_module_06_notebook,
        "07_el_farol_and_opinion_dynamics.ipynb": build_module_07_notebook,
        "08_research_design_lab.ipynb": build_module_08_notebook,
    }
    for fname, fn in builders.items():
        nb = fn()
        target = NB_DIR / fname
        nbf.write(nb, str(target))
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
