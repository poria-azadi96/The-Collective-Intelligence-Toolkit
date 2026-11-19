"""Export high-resolution classroom figures and quantitative summaries for all modules.

This script executes the exact code cells from each workshop Jupyter notebook
(`foundations/00_pre_workshop_primer.ipynb` and `notebooks/01_*.ipynb` through
`08_*.ipynb`) across all pedagogically meaningful parameter regimes, modes,
and scenarios, saving organized PNG figures, console outputs, and Markdown
summaries under `outputs/`.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = REPO_ROOT / "outputs"


def install_ipywidgets_shim() -> None:
    """Register a lightweight no-op `ipywidgets` shim for headless execution."""
    if "ipywidgets" in sys.modules:
        return

    shim = types.ModuleType("ipywidgets")

    def interact(*_args: Any, **_kwargs: Any) -> None:
        return None

    class _WidgetStub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    shim.interact = interact  # type: ignore[attr-defined]
    shim.Dropdown = _WidgetStub  # type: ignore[attr-defined]
    shim.FloatSlider = _WidgetStub  # type: ignore[attr-defined]
    shim.IntSlider = _WidgetStub  # type: ignore[attr-defined]
    shim.Checkbox = _WidgetStub  # type: ignore[attr-defined]
    shim.fixed = _WidgetStub  # type: ignore[attr-defined]
    sys.modules["ipywidgets"] = shim


def load_notebook_namespace(nb_path: Path) -> dict[str, Any]:
    """Execute all code cells in `nb_path` and return the resulting namespace."""
    nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {"__name__": f"nb_{nb_path.stem}"}
    for cell in nb_data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        exec(compile(src, str(nb_path), "exec"), namespace)
    return namespace


def capture_run(
    fn: Callable[..., Any],
    kwargs: dict[str, Any],
    out_png: Path,
    dpi: int = 240,
) -> str:
    """Run a notebook dashboard function, capture its figure and stdout."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.close("all")

    saved_paths: list[Path] = []
    original_show = plt.show

    def patched_show(*_args: Any, **_kw: Any) -> None:
        fig = plt.gcf()
        target = out_png
        if saved_paths:
            stem = f"{out_png.stem}_fig{len(saved_paths) + 1}"
            target = out_png.with_name(f"{stem}{out_png.suffix}")
        fig.savefig(
            target,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
        saved_paths.append(target)
        plt.close(fig)

    plt.show = patched_show
    stdout_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf):
            fn(**kwargs)
        # In case a function builds a figure without calling plt.show()
        if not saved_paths and plt.get_fignums():
            patched_show()
    finally:
        plt.show = original_show
        plt.close("all")

    return stdout_buf.getvalue().strip()


def write_module_summary(
    module_dir: Path,
    title: str,
    description: str,
    notebook_rel: str,
    records: list[dict[str, Any]],
) -> None:
    """Write `summary.md` and `run_log.csv` inside a module output directory."""
    md_lines = [
        f"# {title}",
        "",
        f"**Source Notebook:** `{notebook_rel}`",
        "",
        description,
        "",
        "## Generated Classroom Figures & Outputs",
        "",
        "| # | Subfolder / File | Scenario & Pedagogical Focus | Parameters | Console Output / Metrics |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for idx, rec in enumerate(records, start=1):
        rel_png = rec["rel_png"]
        params_str = ", ".join(f"`{k}={v}`" for k, v in rec["kwargs"].items())
        out_one_line = " <br> ".join(
            line.strip() for line in rec["stdout"].splitlines() if line.strip() and not set(line.strip()) <= {"="}
        )
        md_lines.append(
            f"| {idx:02d} | [`{rel_png}`](./{rel_png}) | {rec['focus']} | {params_str} | {out_one_line} |"
        )

    md_lines.extend(["", "## Full Console Logs", ""])
    for idx, rec in enumerate(records, start=1):
        md_lines.append(f"### {idx:02d}. {rec['focus']}")
        md_lines.append(f"- **File:** [`{rec['rel_png']}`](./{rec['rel_png']})")
        md_lines.append(f"- **Parameters:** `{rec['kwargs']}`")
        md_lines.append("```text")
        md_lines.append(rec["stdout"])
        md_lines.append("```")
        md_lines.append("")

    (module_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    with (module_dir / "run_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["index", "file", "focus", "parameters", "console_output"]
        )
        writer.writeheader()
        for idx, rec in enumerate(records, start=1):
            writer.writerow(
                {
                    "index": idx,
                    "file": rec["rel_png"],
                    "focus": rec["focus"],
                    "parameters": json.dumps(rec["kwargs"], ensure_ascii=False),
                    "console_output": rec["stdout"].replace("\n", " | "),
                }
            )


def build_module_specs() -> list[dict[str, Any]]:
    """Define the educational scenario catalog for all 9 workshop modules."""
    return [
        {
            "folder": "00_Pre_Workshop_Primer",
            "title": "Module 00: Pre-Workshop Primer",
            "notebook": "foundations/00_pre_workshop_primer.ipynb",
            "fn_name": "run_primer_sandbox",
            "description": (
                "Dynamical states, nonlinear logistic map feedback regimes (fixed point, "
                "period-doubling, chaos), lagged phase-portrait embeddings, and binary Shannon entropy."
            ),
            "runs": [
                {
                    "rel_png": "01_fixed_point_r2.80_tau1_p0.10.png",
                    "focus": "Fixed-point convergence (r=2.80) & skewed low-entropy binary source (p=0.10)",
                    "kwargs": {"r_growth": 2.80, "lag_tau": 1, "p_coin": 0.10},
                },
                {
                    "rel_png": "02_period2_oscillation_r3.20_tau1_p0.50.png",
                    "focus": "Period-2 limit cycle (r=3.20) & maximum binary Shannon entropy (p=0.50)",
                    "kwargs": {"r_growth": 3.20, "lag_tau": 1, "p_coin": 0.50},
                },
                {
                    "rel_png": "03_period4_bifurcation_r3.52_tau2_p0.80.png",
                    "focus": "Period-4 oscillation (r=3.52), lag tau=2 embedding & symmetric entropy (p=0.80)",
                    "kwargs": {"r_growth": 3.52, "lag_tau": 2, "p_coin": 0.80},
                },
                {
                    "rel_png": "04_chaos_default_r3.75_tau1_p0.50.png",
                    "focus": "Chaotic trajectory (r=3.75) & lag-1 parabolic attractor reconstruction",
                    "kwargs": {"r_growth": 3.75, "lag_tau": 1, "p_coin": 0.50},
                },
                {
                    "rel_png": "05_deep_chaos_folded_attractor_r3.95_tau2_p0.50.png",
                    "focus": "Deep chaos (r=3.95) & lag-2 folded portrait showing nonlinear structure with near-zero linear correlation",
                    "kwargs": {"r_growth": 3.95, "lag_tau": 2, "p_coin": 0.50},
                },
            ],
        },
        {
            "folder": "01_Information_Theory_and_PID",
            "title": "Module 01: Information Theory & Partial Information Decomposition (PID)",
            "notebook": "notebooks/01_information_theory_and_pid.ipynb",
            "fn_name": "run_pid_dashboard",
            "description": (
                "Williams & Beer Partial Information Decomposition separating Redundancy, "
                "Unique Information, and Synergy across logic gates, input correlations, and channel noise."
            ),
            "runs": [
                {
                    "rel_png": "01_Logic_Gates_Baseline/01_XOR_pure_synergy.png",
                    "focus": "XOR Gate: zero pairwise MI yet 1.0 bit Pure Synergy",
                    "kwargs": {"gate_type": "XOR", "input_corr": 0.0, "noise_prob": 0.0},
                },
                {
                    "rel_png": "01_Logic_Gates_Baseline/02_AND_redundancy_and_synergy.png",
                    "focus": "AND Gate: coexistence of Redundancy and Synergy",
                    "kwargs": {"gate_type": "AND", "input_corr": 0.0, "noise_prob": 0.0},
                },
                {
                    "rel_png": "01_Logic_Gates_Baseline/03_OR_redundancy_and_synergy.png",
                    "focus": "OR Gate: coexistence of Redundancy and Synergy",
                    "kwargs": {"gate_type": "OR", "input_corr": 0.0, "noise_prob": 0.0},
                },
                {
                    "rel_png": "01_Logic_Gates_Baseline/04_COPY_X1_unique_information.png",
                    "focus": "COPY_X1 Gate: 1.0 bit Unique Information in X1 when inputs are independent",
                    "kwargs": {"gate_type": "COPY_X1", "input_corr": 0.0, "noise_prob": 0.0},
                },
                {
                    "rel_png": "02_Correlation_and_Noise_Effects/05_COPY_X1_correlated_inputs_corr0.70.png",
                    "focus": "Correlated COPY_X1 (corr=0.70): input correlation converts Unique info into Redundancy",
                    "kwargs": {"gate_type": "COPY_X1", "input_corr": 0.70, "noise_prob": 0.0},
                },
                {
                    "rel_png": "02_Correlation_and_Noise_Effects/06_XOR_correlated_inputs_corr0.70.png",
                    "focus": "Correlated XOR (corr=0.70): reduced output entropy H(Y) and altered synergy",
                    "kwargs": {"gate_type": "XOR", "input_corr": 0.70, "noise_prob": 0.0},
                },
                {
                    "rel_png": "02_Correlation_and_Noise_Effects/07_XOR_anticorrelated_inputs_corr-0.70.png",
                    "focus": "Anti-correlated XOR (corr=-0.70): input anti-correlation effects on PID atoms",
                    "kwargs": {"gate_type": "XOR", "input_corr": -0.70, "noise_prob": 0.0},
                },
                {
                    "rel_png": "02_Correlation_and_Noise_Effects/08_XOR_noisy_channel_noise0.20.png",
                    "focus": "Noisy XOR (noise=0.20): stochastic channel degradation of Synergy",
                    "kwargs": {"gate_type": "XOR", "input_corr": 0.0, "noise_prob": 0.20},
                },
                {
                    "rel_png": "02_Correlation_and_Noise_Effects/09_AND_noisy_channel_noise0.15.png",
                    "focus": "Noisy AND (noise=0.15): channel noise attenuating redundancy and synergy",
                    "kwargs": {"gate_type": "AND", "input_corr": 0.0, "noise_prob": 0.15},
                },
            ],
        },
        {
            "folder": "02_Transfer_Entropy_and_Network_Inference",
            "title": "Module 02: Transfer Entropy, Delay Reconstruction & Network Inference",
            "notebook": "notebooks/02_transfer_entropy_and_network_inference.ipynb",
            "fn_name": "run_te_laboratory",
            "description": (
                "Kraskov-Stogbauer-Grassberger (KSG) Transfer Entropy delay scans, surrogate "
                "significance testing, and conditioning across 4 canonical causal network motifs."
            ),
            "runs": [
                {
                    "rel_png": "01_Causal_Motifs/01_Direct_Coupling_c0.70_n450.png",
                    "focus": "Direct Coupling (X -> Y at lag u=2): true link peaks at u=2 and survives conditioning",
                    "kwargs": {"motif_type": "Direct_Coupling", "coupling": 0.70, "n_samples": 450},
                },
                {
                    "rel_png": "01_Causal_Motifs/02_Indirect_Cascade_c0.70_n450.png",
                    "focus": "Indirect Cascade (X -> Z -> Y): Conditional TE | Z filters indirect path",
                    "kwargs": {"motif_type": "Indirect_Cascade", "coupling": 0.70, "n_samples": 450},
                },
                {
                    "rel_png": "01_Causal_Motifs/03_Common_Driver_c0.70_n450.png",
                    "focus": "Common Driver (Z -> X, Y): Conditional TE | Z removes spurious bivariate link",
                    "kwargs": {"motif_type": "Common_Driver", "coupling": 0.70, "n_samples": 450},
                },
                {
                    "rel_png": "01_Causal_Motifs/04_Unobserved_Confounder_c0.70_n450.png",
                    "focus": "Unobserved Confounder (hidden U -> X, Y): failure mode where conditioning on Z cannot remove spurious link",
                    "kwargs": {"motif_type": "Unobserved_Confounder", "coupling": 0.70, "n_samples": 450},
                },
                {
                    "rel_png": "02_Sensitivity_Coupling_and_SampleSize/05_Direct_Coupling_weak_c0.30_n450.png",
                    "focus": "Weak Direct Coupling (c=0.30): reduced TE approaching surrogate null floor",
                    "kwargs": {"motif_type": "Direct_Coupling", "coupling": 0.30, "n_samples": 450},
                },
                {
                    "rel_png": "02_Sensitivity_Coupling_and_SampleSize/06_Direct_Coupling_strong_c0.85_n450.png",
                    "focus": "Strong Direct Coupling (c=0.85): sharp delay peak at u=2 well above null",
                    "kwargs": {"motif_type": "Direct_Coupling", "coupling": 0.85, "n_samples": 450},
                },
                {
                    "rel_png": "02_Sensitivity_Coupling_and_SampleSize/07_Common_Driver_small_sample_n250.png",
                    "focus": "Small Sample Regime (n=250): elevated surrogate variance on Common Driver",
                    "kwargs": {"motif_type": "Common_Driver", "coupling": 0.70, "n_samples": 250},
                },
                {
                    "rel_png": "02_Sensitivity_Coupling_and_SampleSize/08_Direct_Coupling_large_sample_n800.png",
                    "focus": "Large Sample Regime (n=800): tight surrogate cutoff and clean lag resolution",
                    "kwargs": {"motif_type": "Direct_Coupling", "coupling": 0.70, "n_samples": 800},
                },
            ],
        },
        {
            "folder": "03_Cellular_Automata_and_Particle_Computation",
            "title": "Module 03: Cellular Automata & Particle Computation",
            "notebook": "notebooks/03_ca_and_particle_computation.ipynb",
            "fn_name": "run_ca_particle_lab",
            "description": (
                "Elementary Cellular Automata (Rules 54, 110, 30, 90), Langton's lambda, "
                "Hanson & Crutchfield (1997) Rule 54 domain filtering & collision vertices, and damage spreading."
            ),
            "runs": [
                {
                    "rel_png": "01_Rule54_Domain_Filter_and_Collisions/01_Rule54_collision_setup_filtered.png",
                    "focus": "Rule 54 Collision Setup + Hanson-Crutchfield Domain Filter (particles & vertices)",
                    "kwargs": {"rule_number": 54, "init_type": "collision_setup", "show_filter": True},
                },
                {
                    "rel_png": "01_Rule54_Domain_Filter_and_Collisions/02_Rule54_random_init_filtered.png",
                    "focus": "Rule 54 Random Init + Domain Filter (emergent ether domains & spontaneous collisions)",
                    "kwargs": {"rule_number": 54, "init_type": "random", "show_filter": True},
                },
                {
                    "rel_png": "01_Rule54_Domain_Filter_and_Collisions/03_Rule54_collision_setup_perturbation.png",
                    "focus": "Rule 54 Collision Setup + Single-Bit Perturbation Damage Spread",
                    "kwargs": {"rule_number": 54, "init_type": "collision_setup", "show_filter": False},
                },
                {
                    "rel_png": "01_Rule54_Domain_Filter_and_Collisions/04_Rule54_single_seed_perturbation.png",
                    "focus": "Rule 54 Single Seed Growth + Perturbation Cone",
                    "kwargs": {"rule_number": 54, "init_type": "single_seed", "show_filter": False},
                },
                {
                    "rel_png": "02_Wolfram_Classes_Comparison/05_Rule110_Turing_Universal_random.png",
                    "focus": "Rule 110 (Class IV Turing Universal) from Random Init + Damage Spreading",
                    "kwargs": {"rule_number": 110, "init_type": "random", "show_filter": False},
                },
                {
                    "rel_png": "02_Wolfram_Classes_Comparison/06_Rule110_Turing_Universal_single_seed.png",
                    "focus": "Rule 110 (Class IV) Asymmetric Evolution from Single Seed",
                    "kwargs": {"rule_number": 110, "init_type": "single_seed", "show_filter": False},
                },
                {
                    "rel_png": "02_Wolfram_Classes_Comparison/07_Rule30_Chaotic_single_seed.png",
                    "focus": "Rule 30 (Class III Chaotic) from Single Seed + Light-Cone Damage Spread",
                    "kwargs": {"rule_number": 30, "init_type": "single_seed", "show_filter": False},
                },
                {
                    "rel_png": "02_Wolfram_Classes_Comparison/08_Rule30_Chaotic_random.png",
                    "focus": "Rule 30 (Class III Chaotic) from Random Init + Rapid Butterfly Effect",
                    "kwargs": {"rule_number": 30, "init_type": "random", "show_filter": False},
                },
                {
                    "rel_png": "02_Wolfram_Classes_Comparison/09_Rule90_Fractal_Linear_single_seed.png",
                    "focus": "Rule 90 (Additive / Sierpinski Fractal) from Single Seed + Linear XOR Cone",
                    "kwargs": {"rule_number": 90, "init_type": "single_seed", "show_filter": False},
                },
                {
                    "rel_png": "02_Wolfram_Classes_Comparison/10_Rule90_Fractal_Linear_random.png",
                    "focus": "Rule 90 (Additive / Linear) from Random Init + Deterministic XOR Cone",
                    "kwargs": {"rule_number": 90, "init_type": "random", "show_filter": False},
                },
            ],
        },
        {
            "folder": "04_Evolutionary_Computation_and_Edge_of_Chaos",
            "title": "Module 04: Evolutionary Computation & Edge of Chaos",
            "notebook": "notebooks/04_evolutionary_computation_holland.ipynb",
            "fn_name": "run_evolutionary_lab",
            "description": (
                "Holland's Schema Theorem, Mitchell-Hraber-Crutchfield (1993/1994) evolutionary "
                "innovation epochs for 1D CA density classification, and live Genetic Algorithm evolution."
            ),
            "runs": [
                {
                    "rel_png": "01_Innovation_Epochs_and_Density_Classification/01_Epoch1_Naive_Majority_rho0.54.png",
                    "focus": "Epoch 1 (Naive Local Majority, rho_0=0.54): frozen domain walls stall classification",
                    "kwargs": {"checkpoint": "Epoch 1: Naive Local Majority", "init_density": 0.54, "run_short_ga": False},
                },
                {
                    "rel_png": "01_Innovation_Epochs_and_Density_Classification/02_Epoch1_Naive_Majority_rho0.46.png",
                    "focus": "Epoch 1 (Naive Local Majority, rho_0=0.46): symmetric failure with frozen blocks",
                    "kwargs": {"checkpoint": "Epoch 1: Naive Local Majority", "init_density": 0.46, "run_short_ga": False},
                },
                {
                    "rel_png": "01_Innovation_Epochs_and_Density_Classification/03_Epoch2_Block_Expander_rho0.54.png",
                    "focus": "Epoch 2 (Asymmetric Block Expander, rho_0=0.54): expands 1-blocks toward ALL-1s",
                    "kwargs": {"checkpoint": "Epoch 2: Asymmetric Block Expander", "init_density": 0.54, "run_short_ga": False},
                },
                {
                    "rel_png": "01_Innovation_Epochs_and_Density_Classification/04_Epoch2_Block_Expander_rho0.46_symmetry_bias.png",
                    "focus": "Epoch 2 Failure Mode (rho_0=0.46): symmetry-breaking bias misclassifies majority-0 density",
                    "kwargs": {"checkpoint": "Epoch 2: Asymmetric Block Expander", "init_density": 0.46, "run_short_ga": False},
                },
                {
                    "rel_png": "01_Innovation_Epochs_and_Density_Classification/05_Epoch3_4_GKL_Particle_Logic_rho0.54.png",
                    "focus": "Epoch 3/4 (GKL Particle Logic, rho_0=0.54): particle signals classify ALL-1s",
                    "kwargs": {"checkpoint": "Epoch 3/4: GKL Particle Logic", "init_density": 0.54, "run_short_ga": False},
                },
                {
                    "rel_png": "01_Innovation_Epochs_and_Density_Classification/06_Epoch3_4_GKL_Particle_Logic_rho0.46.png",
                    "focus": "Epoch 3/4 (GKL Particle Logic, rho_0=0.46): particle signals classify ALL-0s",
                    "kwargs": {"checkpoint": "Epoch 3/4: GKL Particle Logic", "init_density": 0.46, "run_short_ga": False},
                },
                {
                    "rel_png": "01_Innovation_Epochs_and_Density_Classification/07_Epoch3_4_GKL_Particle_Logic_rho0.70.png",
                    "focus": "Epoch 3/4 (GKL Particle Logic, rho_0=0.70): rapid convergence on high density",
                    "kwargs": {"checkpoint": "Epoch 3/4: GKL Particle Logic", "init_density": 0.70, "run_short_ga": False},
                },
                {
                    "rel_png": "01_Innovation_Epochs_and_Density_Classification/08_Epoch3_4_GKL_Particle_Logic_rho0.30.png",
                    "focus": "Epoch 3/4 (GKL Particle Logic, rho_0=0.30): rapid convergence on low density",
                    "kwargs": {"checkpoint": "Epoch 3/4: GKL Particle Logic", "init_density": 0.30, "run_short_ga": False},
                },
                {
                    "rel_png": "02_Live_GA_Evolution/09_Live_GA_12Gen_Trajectory_GKL_rho0.54.png",
                    "focus": "Live 12-Generation GA Evolution Trajectory (best & population mean fitness)",
                    "kwargs": {"checkpoint": "Epoch 3/4: GKL Particle Logic", "init_density": 0.54, "run_short_ga": True},
                },
            ],
        },
        {
            "folder": "05_RBN_and_Immune_Cognition",
            "title": "Module 05: Random Boolean Networks & Distributed Immune Selection",
            "notebook": "notebooks/05_rbn_and_immune_cognition.ipynb",
            "fn_name": "run_module05_lab",
            "description": (
                "Kauffman RBN order-to-chaos phase transitions (Derrida annealing curves), "
                "synchronous vs. asynchronous attractors (Greil & Drossel, 2005), gene knockouts, "
                "and immune clonal selection with McKeithan kinetic proofreading."
            ),
            "runs": [
                {
                    "rel_png": "01_RBN_Phase_Transitions_and_Derrida_Curves/01_Ordered_Phase_K1_p0.50.png",
                    "focus": "Ordered Regime (K=1, p=0.50, s=0.50 < 1): perturbations die out below diagonal",
                    "kwargs": {"mode": "1. RBN Dynamics & Derrida Curve", "k_inputs": 1, "bias_p": 0.50, "knockout_gene": False},
                },
                {
                    "rel_png": "01_RBN_Phase_Transitions_and_Derrida_Curves/02_Ordered_Canalizing_Bias_K3_p0.15.png",
                    "focus": "Ordered via Canalizing Bias (K=3, p=0.15, s=0.765 < 1): bias stabilizes high connectivity",
                    "kwargs": {"mode": "1. RBN Dynamics & Derrida Curve", "k_inputs": 3, "bias_p": 0.15, "knockout_gene": False},
                },
                {
                    "rel_png": "01_RBN_Phase_Transitions_and_Derrida_Curves/03_Critical_Edge_K2_p0.50.png",
                    "focus": "Critical Edge of Chaos (K=2, p=0.50, s=1.00): Derrida tangent to neutral line",
                    "kwargs": {"mode": "1. RBN Dynamics & Derrida Curve", "k_inputs": 2, "bias_p": 0.50, "knockout_gene": False},
                },
                {
                    "rel_png": "01_RBN_Phase_Transitions_and_Derrida_Curves/04_Critical_Edge_Tuned_Bias_K3_p0.20.png",
                    "focus": "Near-Critical Tuned Bias (K=3, p=0.20, s=0.96): critical boundary at higher in-degree",
                    "kwargs": {"mode": "1. RBN Dynamics & Derrida Curve", "k_inputs": 3, "bias_p": 0.20, "knockout_gene": False},
                },
                {
                    "rel_png": "01_RBN_Phase_Transitions_and_Derrida_Curves/05_Chaotic_Phase_K3_p0.50.png",
                    "focus": "Chaotic Regime (K=3, p=0.50, s=1.50 > 1): damage amplification & longer cycles",
                    "kwargs": {"mode": "1. RBN Dynamics & Derrida Curve", "k_inputs": 3, "bias_p": 0.50, "knockout_gene": False},
                },
                {
                    "rel_png": "01_RBN_Phase_Transitions_and_Derrida_Curves/06_Deep_Chaos_K4_p0.50.png",
                    "focus": "Deep Chaos Regime (K=4, p=0.50, s=2.00): steep Derrida curve & long L=26 attractor",
                    "kwargs": {"mode": "1. RBN Dynamics & Derrida Curve", "k_inputs": 4, "bias_p": 0.50, "knockout_gene": False},
                },
                {
                    "rel_png": "01_RBN_Phase_Transitions_and_Derrida_Curves/07_MultiAttractor_Chaos_K5_p0.50.png",
                    "focus": "Multi-Attractor Chaos (K=5, p=0.50, s=2.50): 7 coexisting basins & 25% async fixed-point convergence",
                    "kwargs": {"mode": "1. RBN Dynamics & Derrida Curve", "k_inputs": 5, "bias_p": 0.50, "knockout_gene": False},
                },
                {
                    "rel_png": "02_Gene_Knockout_Attractor_Shifts/08_Knockout_Node0_Critical_K2_p0.50.png",
                    "focus": "Gene Knockout (Node 0) in Critical K=2 RBN: robustness preserving 3 basins & cycle periods",
                    "kwargs": {"mode": "1. RBN Dynamics & Derrida Curve", "k_inputs": 2, "bias_p": 0.50, "knockout_gene": True},
                },
                {
                    "rel_png": "02_Gene_Knockout_Attractor_Shifts/09_Knockout_Node0_Chaotic_K3_p0.50.png",
                    "focus": "Gene Knockout (Node 0) in Chaotic K=3 RBN: attractor cycle shrinks from L=16 to L=5",
                    "kwargs": {"mode": "1. RBN Dynamics & Derrida Curve", "k_inputs": 3, "bias_p": 0.50, "knockout_gene": True},
                },
                {
                    "rel_png": "02_Gene_Knockout_Attractor_Shifts/10_Knockout_Node0_DeepChaos_K4_p0.50.png",
                    "focus": "Gene Knockout (Node 0) in Deep Chaos K=4 RBN: dramatic basin split (1017->536) & cycle drop (L=26->4)",
                    "kwargs": {"mode": "1. RBN Dynamics & Derrida Curve", "k_inputs": 4, "bias_p": 0.50, "knockout_gene": True},
                },
                {
                    "rel_png": "02_Gene_Knockout_Attractor_Shifts/11_Knockout_Node0_MultiAttractor_K5_p0.50.png",
                    "focus": "Gene Knockout (Node 0) in K=5 RBN: 7 attractors reorganize into 3 long cycles (L=20,18,2)",
                    "kwargs": {"mode": "1. RBN Dynamics & Derrida Curve", "k_inputs": 5, "bias_p": 0.50, "knockout_gene": True},
                },
                {
                    "rel_png": "03_Immune_Clonal_Selection_and_Proofreading/12_Immune_Clonal_Selection_and_Kinetic_Proofreading.png",
                    "focus": "Distributed Immune Clonal Selection (affinity maturation) & McKeithan Kinetic Proofreading",
                    "kwargs": {"mode": "2. Immune Clonal Selection & Proofreading", "k_inputs": 2, "bias_p": 0.50, "knockout_gene": False},
                },
            ],
        },
        {
            "folder": "06_Physarum_Contagion_and_Flocking",
            "title": "Module 06: Adaptive Networks (Physarum), Contagion & Vicsek Flocking",
            "notebook": "notebooks/06_physarum_and_contagion.ipynb",
            "fn_name": "run_module06_lab",
            "description": (
                "Tero et al. (2010) Physarum Tokyo Rail optimization & trunk disruption recovery, "
                "Centola (2010) Simple vs. Complex Contagion on small-world networks, and Vicsek flocking."
            ),
            "runs": [
                {
                    "rel_png": "01_Physarum_Tokyo_Rail_Solver/01_Physarum_fault_tolerant_mesh_gamma0.50.png",
                    "focus": "Physarum Fault-Tolerant Mesh (gamma=0.50): 100% single-edge fault tolerance & 1.29x efficiency",
                    "kwargs": {"mode": "1. Physarum Tokyo Rail Solver", "gamma_or_noise": 0.5, "disrupt_or_rewire": False},
                },
                {
                    "rel_png": "01_Physarum_Tokyo_Rail_Solver/02_Physarum_balanced_backbone_gamma0.60.png",
                    "focus": "Physarum Balanced Backbone (gamma=0.60): matches MST global efficiency (1.03x) across planar loops",
                    "kwargs": {"mode": "1. Physarum Tokyo Rail Solver", "gamma_or_noise": 0.6, "disrupt_or_rewire": False},
                },
                {
                    "rel_png": "01_Physarum_Tokyo_Rail_Solver/03_Physarum_weak_feedback_gamma0.70.png",
                    "focus": "Physarum Moderate Pruning (gamma=0.70): Cost=1.05x MST with emergent arterial loops",
                    "kwargs": {"mode": "1. Physarum Tokyo Rail Solver", "gamma_or_noise": 0.7, "disrupt_or_rewire": False},
                },
                {
                    "rel_png": "01_Physarum_Tokyo_Rail_Solver/04_Physarum_optimal_default_gamma1.40.png",
                    "focus": "Physarum Default Gamma (gamma=1.40): strong hub pruning around central Tokyo station",
                    "kwargs": {"mode": "1. Physarum Tokyo Rail Solver", "gamma_or_noise": 1.4, "disrupt_or_rewire": False},
                },
                {
                    "rel_png": "01_Physarum_Tokyo_Rail_Solver/05_Physarum_strong_pruning_gamma2.00.png",
                    "focus": "Physarum Strong Pruning (gamma=2.00): aggressive pruning toward minimal hub core",
                    "kwargs": {"mode": "1. Physarum Tokyo Rail Solver", "gamma_or_noise": 2.0, "disrupt_or_rewire": False},
                },
                {
                    "rel_png": "01_Physarum_Tokyo_Rail_Solver/06_Physarum_hub_disruption_recovery_gamma0.60.png",
                    "focus": "Physarum Hub Disruption Recovery (gamma=0.60, Cut=True): bypass rerouting after trunk severing",
                    "kwargs": {"mode": "1. Physarum Tokyo Rail Solver", "gamma_or_noise": 0.6, "disrupt_or_rewire": True},
                },
                {
                    "rel_png": "01_Physarum_Tokyo_Rail_Solver/07_Physarum_hub_disruption_recovery_gamma1.40.png",
                    "focus": "Physarum Hub Disruption Recovery (gamma=1.40, Cut=True): rerouting under strong feedback",
                    "kwargs": {"mode": "1. Physarum Tokyo Rail Solver", "gamma_or_noise": 1.4, "disrupt_or_rewire": True},
                },
                {
                    "rel_png": "02_Simple_vs_Complex_Contagion/08_Contagion_Clustered_Lattice_p0.02.png",
                    "focus": "Clustered Lattice (rewire p=0.02): high local clustering enables Complex Contagion spread",
                    "kwargs": {"mode": "2. Simple vs. Complex Contagion", "gamma_or_noise": 1.4, "disrupt_or_rewire": False},
                },
                {
                    "rel_png": "02_Simple_vs_Complex_Contagion/09_Contagion_Small_World_Shortcuts_p0.35.png",
                    "focus": "Small-World Shortcuts (rewire p=0.35): Simple Contagion accelerates while Complex Contagion collapses",
                    "kwargs": {"mode": "2. Simple vs. Complex Contagion", "gamma_or_noise": 1.4, "disrupt_or_rewire": True},
                },
                {
                    "rel_png": "03_Vicsek_Collective_Flocking/10_Vicsek_Ordered_Flocking_low_noise_eta0.40.png",
                    "focus": "Vicsek Ordered Flocking (low noise eta=0.40): coherent collective polarization (Phi -> 1)",
                    "kwargs": {"mode": "3. Vicsek Collective Flocking", "gamma_or_noise": 0.2, "disrupt_or_rewire": False},
                },
                {
                    "rel_png": "03_Vicsek_Collective_Flocking/11_Vicsek_SliderMin_Flocking_eta1.00.png",
                    "focus": "Vicsek Coherent Phase (slider min, noise eta=1.00): strong global alignment",
                    "kwargs": {"mode": "3. Vicsek Collective Flocking", "gamma_or_noise": 0.5, "disrupt_or_rewire": False},
                },
                {
                    "rel_png": "03_Vicsek_Collective_Flocking/12_Vicsek_Transition_Regime_eta2.80.png",
                    "focus": "Vicsek Transition Regime (noise eta=2.80): partial local clusters & fluctuating order parameter",
                    "kwargs": {"mode": "3. Vicsek Collective Flocking", "gamma_or_noise": 1.4, "disrupt_or_rewire": False},
                },
                {
                    "rel_png": "03_Vicsek_Collective_Flocking/13_Vicsek_Disordered_Gas_high_noise_eta4.40.png",
                    "focus": "Vicsek Disordered Gas (high noise eta=4.40): random headings & collapsed polarization",
                    "kwargs": {"mode": "3. Vicsek Collective Flocking", "gamma_or_noise": 2.2, "disrupt_or_rewire": False},
                },
            ],
        },
        {
            "folder": "07_Collective_Wisdom_Opinion_and_El_Farol",
            "title": "Module 07: Collective Wisdom, Bounded Confidence & El Farol Market",
            "notebook": "notebooks/07_el_farol_and_opinion_dynamics.ipynb",
            "fn_name": "run_module07_lab",
            "description": (
                "Scott Page's Diversity Prediction Theorem & error correlation, Hegselmann-Krause "
                "bounded confidence fragmentation vs. consensus, Arthur's El Farol Bar (1994), "
                "and Challet-Zhang Minority Game (1997) with networked herding extension."
            ),
            "runs": [
                {
                    "rel_png": "01_Diversity_Prediction_Theorem/01_DPT_Low_Correlation_rho0.05.png",
                    "focus": "DPT Low Error Correlation (rho=0.05): high diversity yields drastic collective error reduction",
                    "kwargs": {"study": "1. Diversity Prediction Theorem", "param_val": 0.05, "herding_ext": False},
                },
                {
                    "rel_png": "01_Diversity_Prediction_Theorem/02_DPT_Moderate_Correlation_rho0.25.png",
                    "focus": "DPT Moderate Error Correlation (rho=0.25): partial diversity benefit",
                    "kwargs": {"study": "1. Diversity Prediction Theorem", "param_val": 0.25, "herding_ext": False},
                },
                {
                    "rel_png": "01_Diversity_Prediction_Theorem/03_DPT_High_Correlation_Groupthink_rho0.80.png",
                    "focus": "DPT High Correlation / Groupthink (rho=0.80): diversity collapses & collective error approaches individual error",
                    "kwargs": {"study": "1. Diversity Prediction Theorem", "param_val": 0.80, "herding_ext": False},
                },
                {
                    "rel_png": "02_Bounded_Confidence_Opinion_Dynamics/04_HK_Fragmentation_eps0.08.png",
                    "focus": "Hegselmann-Krause Fragmentation (eps=0.08): multiple isolated echo chambers",
                    "kwargs": {"study": "2. Bounded Confidence (HK vs DeGroot)", "param_val": 0.08, "herding_ext": False},
                },
                {
                    "rel_png": "02_Bounded_Confidence_Opinion_Dynamics/05_HK_Polarization_eps0.18.png",
                    "focus": "Hegselmann-Krause Polarization (eps=0.18): bipolar / tripolar clustering",
                    "kwargs": {"study": "2. Bounded Confidence (HK vs DeGroot)", "param_val": 0.18, "herding_ext": False},
                },
                {
                    "rel_png": "02_Bounded_Confidence_Opinion_Dynamics/06_HK_Critical_Transition_eps0.25.png",
                    "focus": "Hegselmann-Krause Near-Consensus Transition (eps=0.25): bifurcation toward consensus",
                    "kwargs": {"study": "2. Bounded Confidence (HK vs DeGroot)", "param_val": 0.25, "herding_ext": False},
                },
                {
                    "rel_png": "02_Bounded_Confidence_Opinion_Dynamics/07_HK_Full_Consensus_eps0.40.png",
                    "focus": "Hegselmann-Krause Full Consensus (eps=0.40): converges to single DeGroot-like consensus",
                    "kwargs": {"study": "2. Bounded Confidence (HK vs DeGroot)", "param_val": 0.40, "herding_ext": False},
                },
                {
                    "rel_png": "03_El_Farol_and_Minority_Game/08_El_Farol_and_MG_Independent_M3.png",
                    "focus": "El Farol Bar (M=3) & Original Minority Game (independent agents beat coin-flip volatility)",
                    "kwargs": {"study": "3. El Farol Bar & Minority Game", "param_val": 0.25, "herding_ext": False},
                },
                {
                    "rel_png": "03_El_Farol_and_Minority_Game/09_El_Farol_and_MG_Herding_Extension_M3.png",
                    "focus": "El Farol (M=3) & Minority Game with Networked Herding Extension (q=0.35 spikes volatility)",
                    "kwargs": {"study": "3. El Farol Bar & Minority Game", "param_val": 0.25, "herding_ext": True},
                },
                {
                    "rel_png": "03_El_Farol_and_Minority_Game/10_El_Farol_and_MG_Long_Memory_M5_Herding.png",
                    "focus": "El Farol Longer Memory (M=5) & Minority Game with Herding Overlay",
                    "kwargs": {"study": "3. El Farol Bar & Minority Game", "param_val": 0.75, "herding_ext": True},
                },
            ],
        },
        {
            "folder": "08_Research_Design_and_Validation_Lab",
            "title": "Module 08: Computational Research Design & Validation Lab",
            "notebook": "notebooks/08_research_design_lab.ipynb",
            "fn_name": "run_module08_lab",
            "description": (
                "Track A multiple-testing Benjamini-Hochberg FDR & statistical power calibration, "
                "Track B mechanistic ABM vs. randomized null baseline with Cohen's d effect size, "
                "and the complete 8-Question Scientific Paper Audit Database across 6 foundational papers."
            ),
            "runs": [
                {
                    "rel_png": "01_Track_A_Observational_FDR_and_Power/01_TrackA_LowSNR0.55_T100.png",
                    "focus": "Track A Notebook Default (SNR=0.55, T=100): uncorrected false positives while BH-FDR power=0% (underpowered)",
                    "kwargs": {
                        "paper_choice": "1. Schreiber (2000) - Transfer Entropy",
                        "track": "Track A: Observational FDR & Power",
                        "sample_size": 100,
                    },
                },
                {
                    "rel_png": "01_Track_A_Observational_FDR_and_Power/02_TrackA_LowSNR0.55_T300.png",
                    "focus": "Track A Notebook Default (SNR=0.55, T=300): uncorrected TPR=25%, FPR=2.4%; BH-FDR stays conservative (power=0%)",
                    "kwargs": {
                        "paper_choice": "1. Schreiber (2000) - Transfer Entropy",
                        "track": "Track A: Observational FDR & Power",
                        "sample_size": 300,
                    },
                },
                {
                    "rel_png": "01_Track_A_Observational_FDR_and_Power/03_TrackA_LowSNR0.55_T800.png",
                    "focus": "Track A Notebook Default (SNR=0.55, T=800): weak SNR remains underpowered under 50-test BH-FDR up to T=1200",
                    "kwargs": {
                        "paper_choice": "1. Schreiber (2000) - Transfer Entropy",
                        "track": "Track A: Observational FDR & Power",
                        "sample_size": 800,
                    },
                },
                {
                    "rel_png": "02_Track_B_Generative_ABM_vs_Null_Model/05_TrackB_ABM_vs_Null_Baseline.png",
                    "focus": "Track B Generative ABM vs. Randomized Null Baseline & Cohen's d Falsifiability Profile",
                    "kwargs": {
                        "paper_choice": "5. Tero et al. (2010) - Physarum Tokyo Rail",
                        "track": "Track B: Generative ABM vs Null Model",
                        "sample_size": 300,
                    },
                },
            ],
        },
    ]


def export_calibrated_track_a_figure(
    ns_module08: dict[str, Any], out_dir: Path, dpi: int = 240
) -> dict[str, Any]:
    """Generate a calibrated SNR=1.25 Track A figure showing >80% BH-FDR power crossing."""
    sim_fn = ns_module08["simulate_fdr_validation"]
    sample_size = 1200
    snr = 1.25
    (r_tpr, r_fpr), (f_tpr, f_fpr) = sim_fn(50, 8, snr, sample_size, seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.patch.set_facecolor("#F8FAFC")

    axes[0].set_facecolor("#F8FAFC")
    cats = ["True Positive Rate\n(Statistical Power)", "False Positive Rate\n(Spurious Edges)"]
    x = np.arange(2)
    axes[0].bar(x - 0.17, [r_tpr, r_fpr], width=0.34, color="#E11D48", edgecolor="#0F172A", label="Uncorrected (p < 0.05)")
    axes[0].bar(x + 0.17, [f_tpr, f_fpr], width=0.34, color="#0D9488", edgecolor="#0F172A", label="BH-FDR Corrected (q < 0.05)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cats, fontweight="bold")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title(f"1. Multiple Testing Control (SNR = {snr:.2f}, N_samples = {sample_size})", fontsize=11, fontweight="bold")
    axes[0].legend(facecolor="white")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    t_grid = [100, 200, 300, 500, 800, 1200]
    fdr_powers = [sim_fn(50, 8, snr, t, seed=42)[1][0] for t in t_grid]
    axes[1].set_facecolor("#F8FAFC")
    axes[1].plot(t_grid, fdr_powers, marker="o", color="#2563EB", linewidth=2.2, label=f"BH-FDR Power (SNR={snr:.2f})")
    axes[1].axhline(0.80, color="#D97706", linestyle="--", label="80% Power Benchmark")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Time-Series Sample Length T")
    axes[1].set_ylabel("Detection Power")
    axes[1].set_title("2. Calibrated Sample Size Curve (Crossing 80% Benchmark)", fontsize=11, fontweight="bold")
    axes[1].legend(facecolor="white")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    rel_png = "01_Track_A_Observational_FDR_and_Power/04_TrackA_Calibrated_SNR1.25_T1200.png"
    out_png = out_dir / rel_png
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    stdout_txt = (
        f"[Supplementary Calibrated Run using Module 08 simulate_fdr_validation] "
        f"SNR={snr}, T={sample_size} | Raw TPR={r_tpr:.3f}, FPR={r_fpr:.3f} | "
        f"BH-FDR TPR={f_tpr:.3f}, FPR={f_fpr:.3f}"
    )
    return {
        "module": "08_Research_Design_and_Validation_Lab",
        "rel_png": rel_png,
        "focus": "Track A Calibrated SNR=1.25 (T=1200): BH-FDR achieves 87.5% power (>80% benchmark) with 0.0% false positives",
        "kwargs": {"signal_snr": snr, "sample_size": sample_size, "n_tests": 50, "true_edges": 8},
        "stdout": stdout_txt,
    }


def export_paper_audits(ns_module08: dict[str, Any], out_dir: Path) -> Path:
    """Export all 6 8-question scientific paper audits and template from Module 08."""
    audits_dir = out_dir / "03_8_Question_Paper_Audits"
    audits_dir.mkdir(parents=True, exist_ok=True)
    template_src = REPO_ROOT / "docs" / "8_question_paper_audit_template.md"
    if template_src.exists():
        (audits_dir / "00_Audit_Template.md").write_text(
            template_src.read_text(encoding="utf-8"), encoding="utf-8"
        )
    paper_audits: dict[str, list[str]] = ns_module08["PAPER_AUDITS"]

    combined_lines = [
        "# 8-Question Scientific Paper Audit Protocol — All 6 Core Papers",
        "",
        "Extracted directly from `notebooks/08_research_design_lab.ipynb` (`PAPER_AUDITS`).",
        "",
    ]

    slug_map = {
        "1.": "01_Schreiber_2000_Transfer_Entropy.md",
        "2.": "02_Hanson_Crutchfield_1997_Rule54_Particles.md",
        "3.": "03_Mitchell_Crutchfield_Hraber_1994_Evolved_CA.md",
        "4.": "04_Kauffman_Greil_Drossel_2005_RBN_Attractors.md",
        "5.": "05_Tero_et_al_2010_Physarum_Tokyo_Rail.md",
        "6.": "06_Arthur_1994_Challet_Zhang_1997_El_Farol_MG.md",
    }

    for paper_title, items in paper_audits.items():
        prefix = paper_title.split()[0]
        fname = slug_map.get(prefix, f"{prefix.strip('.')}_audit.md")
        single_lines = [f"# {paper_title}", ""] + [f"- {item}" for item in items] + [""]
        (audits_dir / fname).write_text("\n".join(single_lines), encoding="utf-8")

        combined_lines.append(f"## {paper_title}")
        combined_lines.append("")
        for item in items:
            combined_lines.append(f"- {item}")
        combined_lines.append("")

    combined_path = audits_dir / "all_6_paper_audits.md"
    combined_path.write_text("\n".join(combined_lines), encoding="utf-8")
    return combined_path


def write_master_readme(module_summaries: list[dict[str, Any]]) -> None:
    """Write `outputs/README.md` indexing every module folder and figure."""
    lines = [
        "# Classroom Outputs & Teaching Figures Index",
        "",
        "Generated directly by running every module notebook in **The Collective Intelligence & Info-Dynamics Toolkit** across its interactive pedagogical regimes.",
        "",
        "## Module Overview",
        "",
        "| Module | Folder | Figures | Topic & Classroom Focus | Summary Link |",
        "| :--- | :--- | :---: | :--- | :--- |",
    ]

    total_figs = 0
    for mod in module_summaries:
        total_figs += len(mod["records"])
        folder = mod["folder"]
        lines.append(
            f"| **{mod['title']}** | `{folder}/` | {len(mod['records'])} | {mod['description']} | [summary.md](./{folder}/summary.md) |"
        )

    lines.extend(
        [
            "",
            f"**Total Generated Figures:** {total_figs} high-resolution PNG files (`240 DPI`).",
            "",
            "## Quick Navigation by Module",
            "",
        ]
    )

    for mod in module_summaries:
        folder = mod["folder"]
        lines.append(f"### {mod['title']}")
        lines.append(f"- **Folder:** [`{folder}/`](./{folder}/summary.md) | **Metrics CSV:** [`run_log.csv`](./{folder}/run_log.csv)")
        for rec in mod["records"]:
            lines.append(f"  - [`{rec['rel_png']}`](./{folder}/{rec['rel_png']}): {rec['focus']}")
        if folder == "08_Research_Design_and_Validation_Lab":
            lines.append(
                f"  - [`03_8_Question_Paper_Audits/all_6_paper_audits.md`](./{folder}/03_8_Question_Paper_Audits/all_6_paper_audits.md): Complete 8-Question Audits for all 6 foundational papers"
            )
        lines.append("")

    (OUTPUTS_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Execute all 9 modules and export organized figures and summaries."""
    t_start = time.perf_counter()
    install_ipywidgets_shim()
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)

    specs = build_module_specs()
    all_records: list[dict[str, Any]] = []
    module_summaries: list[dict[str, Any]] = []

    for mod in specs:
        mod_t0 = time.perf_counter()
        nb_path = REPO_ROOT / mod["notebook"]
        ns = load_notebook_namespace(nb_path)
        fn = ns[mod["fn_name"]]
        mod_dir = OUTPUTS_ROOT / mod["folder"]
        mod_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = []
        for run in mod["runs"]:
            out_png = mod_dir / run["rel_png"]
            stdout_txt = capture_run(fn, run["kwargs"], out_png)
            rec = {
                "module": mod["folder"],
                "rel_png": run["rel_png"],
                "focus": run["focus"],
                "kwargs": run["kwargs"],
                "stdout": stdout_txt,
            }
            records.append(rec)
            all_records.append(rec)
            print(f"[OK] {mod['folder']}/{run['rel_png']}")

        if mod["folder"] == "08_Research_Design_and_Validation_Lab":
            cal_rec = export_calibrated_track_a_figure(ns, mod_dir)
            records.insert(-1, cal_rec)
            all_records.insert(-1, cal_rec)
            print(f"[OK] {mod['folder']}/{cal_rec['rel_png']}")
            export_paper_audits(ns, mod_dir)

        write_module_summary(
            module_dir=mod_dir,
            title=mod["title"],
            description=mod["description"],
            notebook_rel=mod["notebook"],
            records=records,
        )

        module_summaries.append(
            {
                "folder": mod["folder"],
                "title": mod["title"],
                "description": mod["description"],
                "records": records,
            }
        )
        print(f"Completed {mod['folder']} ({len(records)} figures) in {time.perf_counter() - mod_t0:.2f}s")

    write_master_readme(module_summaries)
    (OUTPUTS_ROOT / "all_modules_metrics.json").write_text(
        json.dumps(all_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nAll 9 modules exported ({len(all_records)} total figures) in {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
