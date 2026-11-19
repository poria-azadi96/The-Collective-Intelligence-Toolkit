"""Build or regenerate interactive Jupyter/Colab notebooks for Modules 01-04."""

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


def build_module_01_notebook() -> nbf.NotebookNode:
    md_intro = """# Module 01: Information Theory, Nonlinearity & Partial Information Decomposition (PID)

**Core Mathematical Framework:**
- **Self-Information & Shannon Entropy:** $h(x) = -\\log_2 P(x)$, $H(X) = -\\sum P(x)\\log_2 P(x)$
- **Mutual Information:** $I(X; Y) = H(X) - H(X \\mid Y) = D_{KL}(P(X, Y) \\parallel P(X)P(Y))$
- **Landauer's Thermodynamic Bound:** $E_{\\text{erasure}} \\ge k_B T \\ln 2 \\approx 2.87 \\times 10^{-21}\\text{ J}$ at $T = 300\\text{ K}$
- **Williams & Beer (2010) 4-Atom PID Lattice:** $I(X_1, X_2; Y) = R + U_1 + U_2 + S$ with Co-Information $II = R - S$"""

    code_setup = BOOTSTRAP_CODE + """import matplotlib.pyplot as plt
import ipywidgets as widgets
from core.info_theory_pid import (
    create_pid_dashboard_figure,
    run_module01_tier1,
    run_module01_tier2,
    run_module01_tier3,
)
from core.real_world_demos import (
    demo_m01_computational_neuroscience,
    demo_m01_genomics_epistasis,
    demo_m01_thermodynamic_reversible_computing,
)

def run_pid_dashboard(
    gate_type='XOR',
    input_corr=0.0,
    noise_prob=None,
    sigma=0.0,
    coupling=1.0,
    temperature_k=300.0,
):
    fig, stats = create_pid_dashboard_figure(
        gate_type=gate_type,
        input_corr=input_corr,
        sigma=sigma,
        coupling=coupling,
        noise_prob=noise_prob,
        temperature_k=temperature_k,
    )
    plt.show()
    r = stats['pid']
    print(
        f"H(Y)={r['H(Y)']:.4f} b | Joint MI={r['I(X1, X2; Y)']:.4f} b | "
        f"R={r['Redundancy']:.4f} b | U1={r['Unique_X1']:.4f} b | "
        f"U2={r['Unique_X2']:.4f} b | Synergy={r['Synergy']:.4f} b | "
        f"Landauer={r['Landauer_J']:.3e} J ({r['Landauer_kTln2']:.2f} kT ln2)"
    )

widgets.interact(
    run_pid_dashboard,
    gate_type=widgets.Dropdown(
        options=['XOR', 'AND', 'OR', 'COPY', 'IDENTICAL', 'Continuous_Gaussian'],
        value='XOR',
        description='Mechanism:',
    ),
    input_corr=widgets.FloatSlider(min=-0.95, max=0.95, step=0.05, value=0.0, description='Corr (rho):'),
    noise_prob=widgets.fixed(None),
    sigma=widgets.FloatSlider(min=0.0, max=1.0, step=0.05, value=0.0, description='Noise (sigma):'),
    coupling=widgets.FloatSlider(min=0.0, max=1.0, step=0.05, value=1.0, description='Coupling (c):'),
    temperature_k=widgets.FloatSlider(min=4.0, max=400.0, step=10.0, value=300.0, description='Temp (K):'),
);"""

    md_tiers = """## 3-Tier Experimental Protocol
- **Tier 1 (Observation):** Verify `XOR` ($\rho=0, \\sigma=0.1, c=1.0$) exhibits $100\\%$ Synergy ($S \\approx 1.0\\text{ bit}$) with $I(X_1;Y)\\approx I(X_2;Y)\\approx 0$, plus circular nonlinearity ($r \\approx 0, I > 0$).
- **Tier 2 (Analysis):** Sweep $\\rho \\in [0, 0.95]$ on `AND` to trace the Redundancy–Synergy crossover and $II = R - S$ crossing zero.
- **Tier 3 (Stress Test):** Sweep $\\sigma \\in [0, 1.0]$ across all 4 gates to quantify entropy production, MI decay, and Landauer heat dissipation."""

    code_tiers = """def run_m01_tier_selector(tier='Tier 1: XOR Synergy & Canonical Gates'):
    if tier.startswith('Tier 1'):
        fig, res = run_module01_tier1(sigma=0.1, coupling=1.0)
        plt.show()
        print('XOR Stats:', res['gates']['XOR'])
        print('Circular Nonlinearity r =', round(res['nonlinearity']['pearson_r'], 4), '| KSG MI =', round(res['nonlinearity']['mi_ksg_bits'], 4), 'bits')
    elif tier.startswith('Tier 2'):
        fig, res = run_module01_tier2(gate_type='AND')
        plt.show()
        print('Crossover rho* where II = R - S = 0:', res['crossover_rho'])
    else:
        fig, _ = run_module01_tier3(temperature_k=300.0)
        plt.show()

widgets.interact(
    run_m01_tier_selector,
    tier=widgets.Dropdown(
        options=[
            'Tier 1: XOR Synergy & Canonical Gates',
            'Tier 2: AND Redundancy-Synergy Crossover',
            'Tier 3: Noise & Landauer Heat Stress Test',
        ],
        value='Tier 1: XOR Synergy & Canonical Gates',
        description='Protocol:',
    ),
);"""

    md_apps = """## Module 01 Real-World Applications
1. **Computational Neuroscience & Brain Imaging:** Coincidence detection vs. redundant cortical firing.
2. **Genomics & Epistasis:** Discovering pure nonlinear SNP-SNP synergy where single-locus $r \\approx 0$.
3. **Thermodynamic & Reversible Computing:** Irreversible CMOS reset vs. reversible Fredkin/Toffoli logic."""

    code_apps = """def run_m01_real_world_demo(application='1. Computational Neuroscience'):
    if application.startswith('1.'):
        fig, res = demo_m01_computational_neuroscience()
    elif application.startswith('2.'):
        fig, res = demo_m01_genomics_epistasis()
    else:
        fig, res = demo_m01_thermodynamic_reversible_computing()
    plt.show()
    print(res)

widgets.interact(
    run_m01_real_world_demo,
    application=widgets.Dropdown(
        options=[
            '1. Computational Neuroscience',
            '2. Genomics & Epistasis',
            '3. Thermodynamic & Reversible Computing',
        ],
        value='1. Computational Neuroscience',
        description='Case Study:',
    ),
);"""

    return _make_nb([("md", md_intro), ("code", code_setup), ("md", md_tiers), ("code", code_tiers), ("md", md_apps), ("code", code_apps)])


def build_module_02_notebook() -> nbf.NotebookNode:
    md_intro = """# Module 02: Transfer Entropy, Delay Reconstruction & Network Inference

**Core Mathematical Framework:**
- **Schreiber (2000) Transfer Entropy:** $T_{Y \\to X} = I(Y_{\\text{past}}; X_{\\text{future}} \\mid X_{\\text{past}})$
- **Information Dynamics Chain Rule:** $I(X_{\\text{past}}, Y_{\\text{past}}; X_t) = A_X + T_{Y \\to X}$
- **Frenzel–Pompe / KSG k-NN Estimator:** $\\hat{I}(X; Y \\mid Z) = \\psi(k) - \\langle \\psi(n_{xz}+1) + \\psi(n_{yz}+1) - \\psi(n_z+1) \\rangle$
- **Statistical Rigor:** Circular time-shift surrogates ($p = \\frac{1 + \\#\\{T_{\\text{surr}} \\ge T_{\\text{emp}}\\\\}}{1 + N_{\\text{surr}}}$) + Benjamini-Hochberg FDR."""

    code_setup = BOOTSTRAP_CODE + """import matplotlib.pyplot as plt
import ipywidgets as widgets
from core.transfer_entropy import (
    create_te_dashboard_figure,
    run_module02_tier1,
    run_module02_tier2,
    run_module02_tier3,
)
from core.real_world_demos import (
    demo_m02_brain_connectivity,
    demo_m02_finance_contagion,
    demo_m02_gene_regulatory_network,
)

def run_te_laboratory(
    motif_type='Common_Driver',
    coupling=0.70,
    n_samples=450,
    k_neighbors=5,
    n_surrogates=60,
):
    fig, stats = create_te_dashboard_figure(
        motif_type=motif_type,
        coupling=coupling,
        n_samples=n_samples,
        k_neighbors=k_neighbors,
        n_surrogates=n_surrogates,
    )
    plt.show()
    print(
        f"Motif: {motif_type} | Lag u={stats['best_lag']} | "
        f"Bivariate TE={stats['te_bivariate']:.4f} (p={stats['p_bivariate']:.3f}) | "
        f"Conditional TE={stats['te_conditional']:.4f} (p={stats['p_conditional']:.3f})"
    )

widgets.interact(
    run_te_laboratory,
    motif_type=widgets.Dropdown(
        options=['Common_Driver', 'Indirect_Cascade', 'Direct_Coupling', 'Unobserved_Confounder'],
        value='Common_Driver',
        description='Causal Motif:',
    ),
    coupling=widgets.FloatSlider(min=0.2, max=0.9, step=0.1, value=0.7, description='Coupling C:'),
    n_samples=widgets.IntSlider(min=250, max=800, step=50, value=450, description='Sample Size:'),
    k_neighbors=widgets.IntSlider(min=3, max=10, step=1, value=5, description='k-NN (k):'),
    n_surrogates=widgets.IntSlider(min=50, max=200, step=25, value=60, description='Surrogates:'),
);"""

    md_tiers = """## 3-Tier Experimental Protocol
- **Tier 1 (Observation):** Verify conditioning on $Z$ suppresses spurious bivariate links in `Common_Driver` and `Indirect_Cascade`.
- **Tier 2 (Delay Reconstruction):** Wibral et al. (2013) scan $T(u)$ over $u = 1..5$ on `Direct_Coupling` ($\\tau=2$) + Chain-Rule decomposition.
- **Tier 3 (Confounder Stress Test):** Sweep sample size $250 \\to 800$ on `Unobserved_Confounder` to expose the irreducible observational limit."""

    code_tiers = """def run_m02_tier_selector(tier='Tier 1: Confounder & Cascade Filtering'):
    if tier.startswith('Tier 1'):
        fig, res = run_module02_tier1()
    elif tier.startswith('Tier 2'):
        fig, res = run_module02_tier2()
    else:
        fig, res = run_module02_tier3()
    plt.show()
    print(res)

widgets.interact(
    run_m02_tier_selector,
    tier=widgets.Dropdown(
        options=[
            'Tier 1: Confounder & Cascade Filtering',
            'Tier 2: Wibral Delay Reconstruction (tau=2)',
            'Tier 3: Unobserved Confounder Sample Sweep',
        ],
        value='Tier 1: Confounder & Cascade Filtering',
        description='Protocol:',
    ),
);"""

    md_apps = """## Module 02 Real-World Applications
1. **Computational Neuroscience:** 4-region effective brain connectivity (`THAL`, `PFC`, `PAR`, `MOT`).
2. **Quantitative Finance:** Multi-asset asymmetric volatility contagion & systemic risk hub discovery.
3. **Cellular & Systems Biology:** Hill-kinetics transcriptional cascade reconstruction."""

    code_apps = """def run_m02_real_world_demo(application='1. Effective Brain Connectivity'):
    if application.startswith('1.'):
        fig, res = demo_m02_brain_connectivity()
        print('True Edges:', res['true_edges'], '| Recovered:', res['recovered_edges'])
    elif application.startswith('2.'):
        fig, res = demo_m02_finance_contagion()
        print('Systemic Out-Strength:', res['out_strength'])
    else:
        fig, res = demo_m02_gene_regulatory_network()
        print('True Edges:', res['true_edges'], '| Recovered:', res['recovered_edges'])
    plt.show()

widgets.interact(
    run_m02_real_world_demo,
    application=widgets.Dropdown(
        options=[
            '1. Effective Brain Connectivity',
            '2. Quantitative Finance Contagion',
            '3. Systems Biology Gene Regulatory Network',
        ],
        value='1. Effective Brain Connectivity',
        description='Case Study:',
    ),
);"""

    return _make_nb([("md", md_intro), ("code", code_setup), ("md", md_tiers), ("code", code_tiers), ("md", md_apps), ("code", code_apps)])


def build_module_03_notebook() -> nbf.NotebookNode:
    md_intro = """# Module 03: Cellular Automata & Emergent Particle Computation

**Core Computational Concepts:**
- **1D Elementary Cellular Automata (ECA):** 256 rules, Wolfram behavioral classes I–IV, Langton's $\\lambda$.
- **Hanson & Crutchfield (1997) Finite-State Transducer (FST):** 5-state transducer $\\{q_0, q_1, q_2, q_3, q_{\\text{defect}}\\}$ stripping Rule 54's period-4 ether to expose gliders ($\\gamma^+, \\gamma^-$) and stationary defects ($\\alpha, \\beta$).
- **Damage Spreading:** Single-bit perturbation XOR cone $|s(t) - \\tilde{s}(t)|$ and normalized Hamming distance $H(t)$."""

    code_setup = BOOTSTRAP_CODE + """import matplotlib.pyplot as plt
import ipywidgets as widgets
from core.cellular_automata import (
    create_ca_dashboard_figure,
    run_module03_tier1,
    run_module03_tier2,
    run_module03_tier3,
)
from core.real_world_demos import (
    demo_m03_excitable_bz_spiral,
    demo_m03_neuromorphic_glider_logic,
    demo_m03_traffic_flow_rule184,
)

def run_ca_particle_lab(
    rule_number=54,
    init_type='collision_setup',
    show_filter=True,
    width=160,
    steps=120,
):
    fig, stats = create_ca_dashboard_figure(
        rule_number=rule_number,
        init_type=init_type,
        show_filter=show_filter,
        width=width,
        steps=steps,
    )
    plt.show()
    print(f'Module 03 Stats: {stats}')

widgets.interact(
    run_ca_particle_lab,
    rule_number=widgets.Dropdown(
        options=[('Rule 54 (Class IV Particle Logic)', 54), ('Rule 110 (Class IV Universal)', 110), ('Rule 30 (Class III Chaotic)', 30), ('Rule 90 (Fractal/Linear)', 90), ('Rule 108 (Class II)', 108), ('Rule 254 (Class I)', 254)],
        value=54,
        description='ECA Rule:',
    ),
    init_type=widgets.Dropdown(options=['collision_setup', 'random', 'single_seed'], value='collision_setup', description='Init State:'),
    show_filter=widgets.Checkbox(value=True, description='Domain Filter'),
    width=widgets.IntSlider(min=100, max=300, step=20, value=160, description='Width N:'),
    steps=widgets.IntSlider(min=100, max=250, step=10, value=120, description='Steps T:'),
);"""

    md_tiers = """## 3-Tier Experimental Protocol
- **Tier 1 (Observation):** Toggle Rule 54 FST Domain Filter OFF/ON on `collision_setup` to reveal colliding gliders and circled collision vertices.
- **Tier 2 (Quantitative Analysis):** Measure Rule 54 random-init particle density decay and verify stationary ($v=0$) vs. diagonal glider worldlines.
- **Tier 3 (Damage Spreading):** Contrast XOR cones and $H(t)$ across `Rule 90`, `Rule 30`, and `Rule 110`."""

    code_tiers = """def run_m03_tier_selector(tier='Tier 1: Rule 54 FST Domain Filter & Collisions'):
    if tier.startswith('Tier 1'):
        fig, res = run_module03_tier1()
    elif tier.startswith('Tier 2'):
        fig, res = run_module03_tier2()
    else:
        fig, res = run_module03_tier3()
    plt.show()
    print(res)

widgets.interact(
    run_m03_tier_selector,
    tier=widgets.Dropdown(
        options=[
            'Tier 1: Rule 54 FST Domain Filter & Collisions',
            'Tier 2: Random Init Particle Density & Worldlines',
            'Tier 3: Damage Spreading (Rule 90 vs 30 vs 110)',
        ],
        value='Tier 1: Rule 54 FST Domain Filter & Collisions',
        description='Protocol:',
    ),
);"""

    md_apps = """## Module 03 Real-World Applications
1. **Excitable Media:** 2D Belousov-Zhabotinsky chemical spiral waves & cardiac reentry arrhythmias.
2. **Traffic Flow Modeling:** Rule 184 & Nagel-Schreckenberg fundamental diagram and backward phantom jams.
3. **Neuromorphic Hardware:** Decentralized glider-collision XOR parity circuit without central CPU/RAM."""

    code_apps = """def run_m03_real_world_demo(application='1. 2D Excitable BZ Spiral Waves'):
    if application.startswith('1.'):
        fig, res = demo_m03_excitable_bz_spiral()
        print('Final excited fraction:', res['final_excited_fraction'])
    elif application.startswith('2.'):
        fig, res = demo_m03_traffic_flow_rule184()
        print('Max flux R184:', res['max_flux_r184'], '| N-S:', res['max_flux_ns'])
    else:
        fig, res = demo_m03_neuromorphic_glider_logic()
        print('Glider Parity Truth Table:', res)
    plt.show()

widgets.interact(
    run_m03_real_world_demo,
    application=widgets.Dropdown(
        options=[
            '1. 2D Excitable BZ Spiral Waves',
            '2. Traffic Flow (Rule 184 & Nagel-Schreckenberg)',
            '3. Neuromorphic Glider Parity Logic',
        ],
        value='1. 2D Excitable BZ Spiral Waves',
        description='Case Study:',
    ),
);"""

    return _make_nb([("md", md_intro), ("code", code_setup), ("md", md_tiers), ("code", code_tiers), ("md", md_apps), ("code", code_apps)])


def build_module_04_notebook() -> nbf.NotebookNode:
    md_intro = """# Module 04: Evolutionary Computation, Genetic Algorithms & Edge of Chaos

**Core Mathematical & Algorithmic Framework:**
- **Density Classification Task (DCT):** $N = 149$ periodic 1D CA with radius $r = 3$ encoded as a $2^7 = 128$-bit GA chromosome.
- **GKL Particle Rule ($r=3$):** Propagates checkerboard signals ($v = +1/3$) and $0$-domain walls ($v = -1$) to resolve global majority.
- **Holland's Schema Theorem:** $\\mathbb{E}[m(H, t+1)] \\ge m(H, t)\\frac{f(H)}{\\bar{f}}\\left[1 - p_c\\frac{\\delta(H)}{L-1}\\right](1-p_m)^{o(H)}$."""

    code_setup = BOOTSTRAP_CODE + """import matplotlib.pyplot as plt
import ipywidgets as widgets
from core.evolutionary_ga import (
    create_evolutionary_dashboard_figure,
    holland_schema_lower_bound,
    run_module04_tier1,
    run_module04_tier2,
    run_module04_tier3,
    schema_properties,
)
from core.real_world_demos import (
    demo_m04_automl_nas,
    demo_m04_protein_directed_evolution,
    demo_m04_swarm_robotics_foraging,
)

def run_evolutionary_lab(
    checkpoint='Epoch 3/4 (GKL)',
    init_density=0.54,
    run_short_ga=False,
):
    fig, stats = create_evolutionary_dashboard_figure(
        checkpoint=checkpoint,
        init_density=init_density,
        run_short_ga=run_short_ga,
        seed=7,
    )
    plt.show()
    o_h, d_h = schema_properties('1**0*1**')
    schema_next = holland_schema_lower_bound(m_t=4, f_schema=0.78, f_avg=0.52, defining_length=d_h, order_k=o_h)
    print(
        f"Selected Rule: {stats['checkpoint']} | Uniform Acc: {stats['fit_uniform']:.2%} | "
        f"Unbiased Acc: {stats['fit_unbiased']:.2%} | Final Density: {stats['final_density']:.2f}"
    )
    print(
        f"Holland Schema Theorem H=1**0*1** (o(H)={o_h}, delta={d_h}, m_t=4, f(H)/f_avg=1.50) "
        f"=> E[m(H, t+1)] >= {schema_next:.2f}"
    )

widgets.interact(
    run_evolutionary_lab,
    checkpoint=widgets.Dropdown(
        options=['Epoch 1', 'Epoch 2 (Asymmetric Block Expander)', 'Epoch 3/4 (GKL)'],
        value='Epoch 3/4 (GKL)',
        description='Checkpoint:',
    ),
    init_density=widgets.FloatSlider(min=0.20, max=0.80, step=0.02, value=0.54, description='Init Density:'),
    run_short_ga=widgets.Checkbox(value=False, description='Run Live 12-Gen GA'),
);"""

    md_tiers = """## 3-Tier Experimental Protocol
- **Tier 1 (Observation):** Switch `Epoch 1 -> Epoch 2 -> Epoch 3/4 (GKL)` at $\\rho_0 = 0.54$ to compare frozen blocks vs. particle signals.
- **Tier 2 (Symmetry & Bias Analysis):** Test `Epoch 2` at $\\rho_0 = 0.44$ vs. $0.56$ and plot accuracy across densities (Land-Belew boundary).
- **Tier 3 (Live Evolution & Schema Check):** Run `Live 12-Gen GA (Seed=7)` and verify Holland's Schema lower bound vs. observed counts."""

    code_tiers = """def run_m04_tier_selector(tier='Tier 1: Innovation Epochs on rho=0.54'):
    if tier.startswith('Tier 1'):
        fig, res = run_module04_tier1(init_density=0.54)
        print(res)
    elif tier.startswith('Tier 2'):
        fig, res = run_module04_tier2()
        print('Epoch 2 rho=0.44 final:', res['epoch2_rho044_final'], '| rho=0.56 final:', res['epoch2_rho056_final'])
    else:
        fig, res = run_module04_tier3(seed=7)
        print('Canonical Schema H=1**0*1**:', res['canonical_schema_example'])
        for rec in res['schema_records'][:5]:
            print(rec)
    plt.show()

widgets.interact(
    run_m04_tier_selector,
    tier=widgets.Dropdown(
        options=[
            'Tier 1: Innovation Epochs on rho=0.54',
            'Tier 2: Epoch 2 Symmetry Bias (0.44 vs 0.56)',
            'Tier 3: Live 12-Gen GA & Schema Check (Seed=7)',
        ],
        value='Tier 1: Innovation Epochs on rho=0.54',
        description='Protocol:',
    ),
);"""

    md_apps = """## Module 04 Real-World Applications
1. **Computational Biology & Protein Design:** Directed evolution on Kauffman $NK$ rugged landscapes.
2. **Artificial Intelligence & AutoML:** Discrete Neural Architecture Search (Accuracy vs. FLOPs Pareto front).
3. **Swarm & Evolutionary Robotics:** Decentralized 2D foraging robots without a central controller."""

    code_apps = """def run_m04_real_world_demo(application='1. Directed Protein Evolution (NK)'):
    if application.startswith('1.'):
        fig, res = demo_m04_protein_directed_evolution()
    elif application.startswith('2.'):
        fig, res = demo_m04_automl_nas()
    else:
        fig, res = demo_m04_swarm_robotics_foraging()
    plt.show()
    print({k: v for k, v in res.items() if not k.startswith('snaps')})

widgets.interact(
    run_m04_real_world_demo,
    application=widgets.Dropdown(
        options=[
            '1. Directed Protein Evolution (NK)',
            '2. AutoML Neural Architecture Search',
            '3. Swarm Robotics Foraging Evolution',
        ],
        value='1. Directed Protein Evolution (NK)',
        description='Case Study:',
    ),
);"""

    return _make_nb([("md", md_intro), ("code", code_setup), ("md", md_tiers), ("code", code_tiers), ("md", md_apps), ("code", code_apps)])


def main() -> None:
    builders = {
        "01_information_theory_and_pid.ipynb": build_module_01_notebook,
        "02_transfer_entropy_and_network_inference.ipynb": build_module_02_notebook,
        "03_ca_and_particle_computation.ipynb": build_module_03_notebook,
        "04_evolutionary_computation_holland.ipynb": build_module_04_notebook,
    }
    for fname, fn in builders.items():
        nb = fn()
        target = NB_DIR / fname
        nbf.write(nb, str(target))
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
