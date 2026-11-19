"""Module 08: 8-Question Paper Audit Engine, BH-FDR Power Calibration, Cohen's d & Research Brief."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

from core.plotting_utils import (
    PALETTE,
    apply_publication_style,
    fig_to_pil_image,
    save_gif_from_frames,
    style_axes,
)

PAPER_AUDITS: dict[str, dict[str, Any]] = {
    "1. Schreiber (2000) - Transfer Entropy": {
        "short": "Schreiber (2000) TE",
        "module": "Module 02",
        "setup": [
            "Q1. Problem: Symmetric mutual information cannot distinguish directional flow from static shared correlation.",
            "Q2. Hypothesis: Conditioning on target past history isolates directed predictive information T_{Y->X}.",
            "Q3. Model: Conditional MI I(Y_past; X_future | X_past) via kernel or adaptive k-NN (KSG) estimators.",
            "Q4. Assumptions: Local stationarity, Markov order k, and absence of unobserved common drivers.",
        ],
        "critique": [
            "Q5. Key Result: Asymmetric coupling detection in coupled chaotic maps and physiological bivariate series.",
            "Q6. Reproducibility Target: Suppression of spurious links in Common Driver & Cascade motifs via Conditional TE.",
            "Q7. Limitations: Bivariate TE fails under confounders; high-dimensional conditioning suffers curse of dimensionality.",
            "Q8. Open Question: How does greedy iterative conditional TE scale under non-stationary regime shifts?",
        ],
        "q7_summary": "Bivariate TE fails under confounders",
        "q8_summary": "Greedy conditional TE under non-stationarity",
    },
    "2. Hanson & Crutchfield (1997) - Rule 54 Particles": {
        "short": "Hanson & Crutchfield (1997)",
        "module": "Module 03",
        "setup": [
            "Q1. Problem: How to formally extract emergent computational structures from raw CA spacetime patterns.",
            "Q2. Hypothesis: Regular spacetime domains act as a vacuum; localized defects act as information particles.",
            "Q3. Model: Finite-State Transducer (FST) domain filters built from the regular language grammar of Rule 54.",
            "Q4. Assumptions: Known periodic ether background and deterministic synchronous lattice updates.",
        ],
        "critique": [
            "Q5. Key Result: Complete particle taxonomy (alpha, beta, gamma+, gamma-) and collision algebra.",
            "Q6. Reproducibility Target: Filtering period-4 ether to expose particle tracks and collision vertices.",
            "Q7. Limitations: Rule-54 FST filter is grammar-specific and does not automatically generalize to chaotic rules.",
            "Q8. Open Question: How robust is particle collision logic under stochastic update noise?",
        ],
        "q7_summary": "Rule-54 FST filter is grammar-specific",
        "q8_summary": "Particle robustness under stochastic update noise",
    },
    "3. Mitchell, Crutchfield & Hraber (1993/1994) - Evolved CA": {
        "short": "Mitchell et al. (1994)",
        "module": "Module 04",
        "setup": [
            "Q1. Problem: Can evolutionary search discover decentralized local rules solving a global density task?",
            "Q2. Hypothesis: GA discovers particle-based propagation, though symmetry breaking impedes search.",
            "Q3. Model: Radius-3 1D CA (N=149) evolved via Genetic Algorithm on uniform-density initial configurations.",
            "Q4. Assumptions: Fitness evaluated over finite random ICs without explicit global communication.",
        ],
        "critique": [
            "Q5. Key Result: Four evolutionary innovation epochs transitioning from block expansion to particle logic.",
            "Q6. Reproducibility Target: Accuracy gap between Epoch 1 local majority and GKL particle logic.",
            "Q7. Limitations: GA symmetry breaking & density overfitting (no binary r=3 CA solves 100%, Land & Belew).",
            "Q8. Open Question: How does co-evolutionary hard-density curriculum design overcome symmetry breaking?",
        ],
        "q7_summary": "GA symmetry breaking & density overfitting",
        "q8_summary": "Co-evolutionary hard-density curriculum design",
    },
    "4. Kauffman / Greil & Drossel (2005) - RBN Attractors": {
        "short": "Kauffman / Greil & Drossel",
        "module": "Module 05",
        "setup": [
            "Q1. Problem: How do gene regulatory topology (K) and update timing govern stability and cell fates?",
            "Q2. Hypothesis: Critical networks (s = 2Kp(1-p) = 1) balance homeostatic stability with responsiveness.",
            "Q3. Model: Random Boolean Networks with N nodes, in-degree K, bias p, and sync vs async updates.",
            "Q4. Assumptions: Annealed approximation for Derrida curve; uniform random wiring without modularity.",
        ],
        "critique": [
            "Q5. Key Result: Superpolynomial attractors in sync critical RBNs vs power-law scaling under async updates.",
            "Q6. Reproducibility Target: Derrida damage spreading curve and sync vs async attractor convergence.",
            "Q7. Limitations: Sync limit cycles collapse into loose attractors under async timing; exhaustive scan limited to small N.",
            "Q8. Open Question: How do biologically realistic canalizing functions stabilize asynchronous GRNs?",
        ],
        "q7_summary": "Sync cycles vs async loose attractors",
        "q8_summary": "Canalizing functions in asynchronous GRNs",
    },
    "5. Tero et al. (2010) - Physarum Tokyo Rail": {
        "short": "Tero et al. (2010) Physarum",
        "module": "Module 06",
        "setup": [
            "Q1. Problem: Can a decentralized organism design transport networks balancing cost, efficiency, and resilience?",
            "Q2. Hypothesis: Positive feedback between Poiseuille flux Q_ij and conductance D_ij self-organizes planar graphs.",
            "Q3. Model: Kirchhoff Laplacian flow coupled with adaptive ODE dD_ij/dt = |Q_ij|^mu/(1+|Q_ij|^mu) - gamma*D_ij.",
            "Q4. Assumptions: Quasi-steady laminar flow and sequential random source-sink demand pairs.",
        ],
        "critique": [
            "Q5. Key Result: Reconstructed networks match Tokyo rail efficiency and fault tolerance at comparable length.",
            "Q6. Reproducibility Target: Multi-objective comparison (Cost, Efficiency, Fault Tolerance) vs MST baseline.",
            "Q7. Limitations: Baseline benchmark evaluates static fault tolerance rather than live severing under skewed demand.",
            "Q8. Open Question: How does dynamic edge severing & non-uniform demand shift optimal feedback exponent mu?",
        ],
        "q7_summary": "Static fault tolerance vs live repair",
        "q8_summary": "Dynamic edge severing & non-uniform demand",
    },
    "6. Arthur (1994) & Challet-Zhang (1997) - El Farol & MG": {
        "short": "Arthur (1994) & Minority Game",
        "module": "Module 07",
        "setup": [
            "Q1. Problem: How do boundedly rational agents coordinate when the optimal action is being in the minority?",
            "Q2. Hypothesis: Inductive competition among heterogeneous predictors self-organizes attendance near capacity.",
            "Q3. Model: Agents pick highest-scoring local predictor from private bag of S strategies on shared history.",
            "Q4. Assumptions: Agents act independently without direct social communication or herding in baseline.",
        ],
        "critique": [
            "Q5. Key Result: Attendance fluctuates around L=60 (El Farol) and shows volatility cusp alpha_c (Minority Game).",
            "Q6. Reproducibility Target: Attendance stabilization near L=60 and volatility scaling across alpha = 2^M / N.",
            "Q7. Limitations: Assumes independent non-communicating agents; breaks down under social coupling.",
            "Q8. Open Question: What network topologies prevent social herding cascades and excess volatility spikes?",
        ],
        "q7_summary": "Assumes independent non-communicating agents",
        "q8_summary": "Social network herding & volatility cascades",
    },
}


def benjamini_hochberg_fdr(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Apply Benjamini-Hochberg False Discovery Rate control to an array of p-values."""
    p_arr = np.asarray(p_values, dtype=np.float64)
    m = len(p_arr)
    order = np.argsort(p_arr)
    ranked_p = p_arr[order]
    thresholds = float(alpha) * (np.arange(1, m + 1) / float(m))
    passed = ranked_p <= thresholds
    reject = np.zeros(m, dtype=bool)
    if np.any(passed):
        max_k = int(np.max(np.where(passed)[0]))
        reject[order[: max_k + 1]] = True
    return reject


def simulate_fdr_validation(
    n_tests: int = 50,
    true_edges: int = 8,
    signal_snr: float = 1.05,
    sample_size: int = 300,
    alpha: float = 0.05,
    n_mc: int = 60,
    seed: int = 42,
) -> dict[str, float]:
    """Run Monte Carlo BH-FDR power & false-discovery simulation at sample size T."""
    rng = np.random.default_rng(seed)
    eff = float(signal_snr) * np.sqrt(float(sample_size) / 100.0)
    raw_tprs, raw_fdps, fdr_tprs, fdr_fdps = [], [], [], []

    for _ in range(max(1, int(n_mc))):
        z_true = rng.normal(loc=eff, scale=1.0, size=true_edges)
        z_null = rng.normal(loc=0.0, scale=1.0, size=n_tests - true_edges)
        z_all = np.concatenate([z_true, z_null])
        p_vals = 0.5 * erfc(z_all / np.sqrt(2.0))

        raw_rej = p_vals < float(alpha)
        fdr_rej = benjamini_hochberg_fdr(p_vals, alpha=float(alpha))

        raw_tprs.append(float(np.mean(raw_rej[:true_edges])))
        raw_fdps.append(float(np.sum(raw_rej[true_edges:])) / max(1.0, float(np.sum(raw_rej))))
        fdr_tprs.append(float(np.mean(fdr_rej[:true_edges])))
        fdr_fdps.append(float(np.sum(fdr_rej[true_edges:])) / max(1.0, float(np.sum(fdr_rej))))

    return {
        "sample_size": int(sample_size),
        "raw_tpr": float(np.mean(raw_tprs)),
        "raw_fdp": float(np.mean(raw_fdps)),
        "fdr_tpr": float(np.mean(fdr_tprs)),
        "fdr_fdp": float(np.mean(fdr_fdps)),
    }


def calibrate_fdr_power_curve(
    n_tests: int = 50,
    true_edges: int = 8,
    signal_snr: float = 1.05,
    alpha: float = 0.05,
    t_grid: list[int] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Sweep sample size T in [100, 1200] and solve minimum T_80 for 80% BH-FDR power."""
    if t_grid is None:
        t_grid = [100, 200, 300, 400, 500, 650, 800, 1000, 1200]
    rows = [
        simulate_fdr_validation(
            n_tests=n_tests, true_edges=true_edges, signal_snr=signal_snr, sample_size=t, alpha=alpha, n_mc=60, seed=seed + t
        )
        for t in t_grid
    ]
    fdr_powers = np.array([r["fdr_tpr"] for r in rows])
    t_arr = np.array(t_grid, dtype=np.float64)

    t_80: float | None = None
    hits = np.where(fdr_powers >= 0.80)[0]
    if len(hits) > 0:
        idx = int(hits[0])
        if idx == 0:
            t_80 = float(t_arr[0])
        else:
            p0, p1 = fdr_powers[idx - 1], fdr_powers[idx]
            t0, t1 = t_arr[idx - 1], t_arr[idx]
            t_80 = float(t0 + (0.80 - p0) * (t1 - t0) / max(p1 - p0, 1e-9))

    return {
        "t_grid": t_grid,
        "raw_tpr": [r["raw_tpr"] for r in rows],
        "raw_fdp": [r["raw_fdp"] for r in rows],
        "fdr_tpr": [r["fdr_tpr"] for r in rows],
        "fdr_fdp": [r["fdr_fdp"] for r in rows],
        "T_80": t_80,
    }


def compute_cohens_d(sample_mech: np.ndarray, sample_null: np.ndarray) -> float:
    """Compute exact standardized separation Cohen's d using pooled standard deviation."""
    a = np.asarray(sample_mech, dtype=np.float64)
    b = np.asarray(sample_null, dtype=np.float64)
    n1, n2 = len(a), len(b)
    s1_sq = float(np.var(a, ddof=1)) if n1 > 1 else 0.0
    s2_sq = float(np.var(b, ddof=1)) if n2 > 1 else 0.0
    s_pooled = np.sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / max(1, n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / max(s_pooled, 1e-9))


def simulate_abm_vs_null(
    coupling_strength: float = 0.70,
    n_seeds: int = 24,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare mechanistic phase transition against distribution-preserving randomized null baseline."""
    rng = np.random.default_rng(seed)
    params = np.linspace(0.0, 1.0, 11)
    model_means, model_stds, null_means, null_stds, cohens_ds = [], [], [], [], []
    focused_mech = np.array([])
    focused_null = np.array([])

    for p in params:
        m_vals = np.clip(1.0 / (1.0 + np.exp(-10.0 * (p - 0.45))) + rng.normal(0.0, 0.055, size=n_seeds), 0.0, 1.0)
        n_vals = np.clip(0.16 + 0.05 * p + rng.normal(0.0, 0.045, size=n_seeds), 0.0, 1.0)
        model_means.append(float(np.mean(m_vals)))
        model_stds.append(float(np.std(m_vals, ddof=1)))
        null_means.append(float(np.mean(n_vals)))
        null_stds.append(float(np.std(n_vals, ddof=1)))
        cohens_ds.append(compute_cohens_d(m_vals, n_vals))
        if abs(p - round(float(coupling_strength), 1)) < 0.06:
            focused_mech = m_vals
            focused_null = n_vals

    if focused_mech.size == 0:
        focused_mech = np.clip(1.0 / (1.0 + np.exp(-10.0 * (coupling_strength - 0.45))) + rng.normal(0.0, 0.055, size=n_seeds), 0.0, 1.0)
        focused_null = np.clip(0.16 + 0.05 * coupling_strength + rng.normal(0.0, 0.045, size=n_seeds), 0.0, 1.0)

    return {
        "params": params,
        "model_means": np.array(model_means),
        "model_stds": np.array(model_stds),
        "null_means": np.array(null_means),
        "null_stds": np.array(null_stds),
        "cohens_ds": np.array(cohens_ds),
        "focused_mech": focused_mech,
        "focused_null": focused_null,
        "focused_cohen_d": compute_cohens_d(focused_mech, focused_null),
    }


def build_research_brief(
    title: str,
    paper_key: str,
    gap_analysis: str,
    hypothesis: str,
    model_arch: str,
    validation_plan: str,
    expected_claim: str,
) -> dict[str, str]:
    """Generate a structured 1-page 5-Part Research Brief in Markdown and JSON formats."""
    audit = PAPER_AUDITS.get(paper_key, list(PAPER_AUDITS.values())[0])
    md_text = "\n".join(
        [
            f"# 5-Part Research Brief: {title}",
            "",
            f"**Anchored Classic:** {paper_key} ({audit['module']})",
            "",
            "## 1. Gap Analysis (Derived from Q7 Limitation)",
            gap_analysis,
            "",
            "## 2. Falsifiable Hypothesis (Q8 Extension)",
            hypothesis,
            "",
            "## 3. Minimal Model Architecture",
            model_arch,
            "",
            "## 4. Null Baseline & Sensitivity Validation Plan",
            validation_plan,
            "",
            "## 5. Expected Defensible Claim",
            expected_claim,
            "",
        ]
    )
    payload = {
        "title": title,
        "anchored_paper": paper_key,
        "gap_analysis": gap_analysis,
        "hypothesis": hypothesis,
        "model_architecture": model_arch,
        "validation_plan": validation_plan,
        "expected_claim": expected_claim,
    }
    return {"markdown": md_text, "json": json.dumps(payload, indent=2, ensure_ascii=False)}


def create_module08_dashboard_figure(
    paper_choice: str = "1. Schreiber (2000) - Transfer Entropy",
    track: str = "Track A: Observational FDR & Power",
    sample_size: int = 300,
    signal_snr: float = 1.05,
    alpha: float = 0.05,
    coupling_strength: float = 0.70,
    seed: int = 42,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Build Module 08 Validation Dashboard (Track A Power/FDR or Track B Null/Cohen's d)."""
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    fig.patch.set_facecolor(PALETTE["bg"])

    if track.startswith("Track A") or "FDR" in track:
        cur = simulate_fdr_validation(50, 8, signal_snr, sample_size, alpha=alpha, n_mc=60, seed=seed)
        cal = calibrate_fdr_power_curve(50, 8, signal_snr, alpha=alpha, seed=seed)

        ax0 = style_axes(axes[0], title=f"1. Multiple Testing Control ($T={sample_size}$, $\\alpha={alpha:.2f}$)", ylabel="Rate / Proportion", grid_axis="y")
        cats = ["Detection Power\n(TPR)", "False Discovery\nProportion (FDP)"]
        xp = np.arange(2)
        b_raw = ax0.bar(xp - 0.17, [cur["raw_tpr"], cur["raw_fdp"]], width=0.34, color=PALETTE["crimson"], edgecolor=PALETTE["ink"], label=f"Uncorrected (p < {alpha:.2f})")
        b_fdr = ax0.bar(xp + 0.17, [cur["fdr_tpr"], cur["fdr_fdp"]], width=0.34, color=PALETTE["teal"], edgecolor=PALETTE["ink"], label=f"BH-FDR (q < {alpha:.2f})")
        ax0.set_xticks(xp)
        ax0.set_xticklabels(cats, fontweight="bold")
        ax0.set_ylim(0, 1.15)
        for b in list(b_raw) + list(b_fdr):
            h = b.get_height()
            ax0.text(b.get_x() + b.get_width() / 2, h + 0.025, f"{h:.0%}", ha="center", fontweight="bold", fontsize=9.0)
        ax0.legend(loc="upper right", fontsize=8.3)

        ax1 = style_axes(axes[1], title=f"2. Power Calibration Curve (SNR={signal_snr:.2f})", xlabel="Sample Length $T$", ylabel="Detection Power (TPR)")
        ax1.plot(cal["t_grid"], cal["fdr_tpr"], marker="o", lw=2.4, color=PALETTE["cobalt"], label="BH-FDR Power")
        ax1.plot(cal["t_grid"], cal["raw_tpr"], marker="s", lw=1.8, linestyle=":", color=PALETTE["slate"], label="Uncorrected Power")
        ax1.axhline(0.80, color=PALETTE["amber"], linestyle="--", lw=1.8, label="80% Power Target")
        if cal["T_80"] is not None:
            ax1.axvline(cal["T_80"], color=PALETTE["teal"], linestyle="--", lw=1.6)
            ax1.scatter([cal["T_80"]], [0.80], color=PALETTE["crimson"], s=95, zorder=6, label=f"$T_{{80}} \\approx {cal['T_80']:.0f}$")
        ax1.set_ylim(0, 1.06)
        ax1.legend(loc="lower right", fontsize=8.2)

        fig.suptitle(f"Module 08 Track A — {paper_choice}", fontsize=12.5, fontweight="bold", y=1.01)
        fig.tight_layout()
        return fig, {"track": "Track A", "current": cur, "T_80": cal["T_80"], "calibration": cal}

    # Track B: Mechanistic ABM vs Null Baseline + Cohen's d
    tb = simulate_abm_vs_null(coupling_strength=coupling_strength, n_seeds=24, seed=seed)
    ax0 = style_axes(axes[0], title="1. Mechanistic Model vs. Randomized Null Baseline", xlabel="Control Coupling Parameter", ylabel="Emergent Macroscopic Metric")
    ax0.plot(tb["params"], tb["model_means"], marker="o", color=PALETTE["cobalt"], lw=2.4, label="Mechanistic Model (±1 SD)")
    ax0.fill_between(tb["params"], tb["model_means"] - tb["model_stds"], tb["model_means"] + tb["model_stds"], color=PALETTE["cobalt"], alpha=0.18)
    ax0.plot(tb["params"], tb["null_means"], marker="s", color=PALETTE["slate"], linestyle="--", lw=2.1, label="Randomized Null Baseline")
    ax0.fill_between(tb["params"], tb["null_means"] - tb["null_stds"], tb["null_means"] + tb["null_stds"], color=PALETTE["slate"], alpha=0.18)
    ax0.set_ylim(0, 1.12)
    ax0.legend(loc="upper left", fontsize=8.4)

    ax1 = style_axes(axes[1], title=f"2. Null vs. Mechanistic Distribution (Cohen's $d = {tb['focused_cohen_d']:.2f}$)", xlabel="Control Coupling Parameter", ylabel="Effect Size (Cohen's $d$)", grid_axis="y")
    ax1.bar(tb["params"], tb["cohens_ds"], width=0.072, color=PALETTE["teal"], edgecolor=PALETTE["ink"])
    ax1.axhline(0.8, color=PALETTE["amber"], linestyle="--", lw=1.4, label="Large Effect ($d = 0.8$)")
    ax1.legend(loc="upper left", fontsize=8.5)

    fig.suptitle(f"Module 08 Track B — {paper_choice}", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {
        "track": "Track B",
        "focused_cohen_d": tb["focused_cohen_d"],
        "max_cohen_d": float(np.max(tb["cohens_ds"])),
    }


def run_module08_tier1() -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 1: Paper Audit Matrix mapping Q7 Limitations -> Q8 Open Research Questions across all 6 Classics."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(14.4, 5.4))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.axis("off")

    papers = list(PAPER_AUDITS.values())
    y_positions = np.linspace(0.86, 0.14, len(papers))
    ax.text(0.02, 0.96, "Classic Paper (Module)", fontsize=10.5, fontweight="bold", color=PALETTE["ink"])
    ax.text(0.32, 0.96, "Q7: Core Limitation / Failure Mode", fontsize=10.5, fontweight="bold", color=PALETTE["crimson"])
    ax.text(0.68, 0.96, "Q8: Open Research Extension", fontsize=10.5, fontweight="bold", color=PALETTE["teal"])
    ax.plot([0.01, 0.99], [0.93, 0.93], color=PALETTE["ink"], lw=1.4)

    for y, info in zip(y_positions, papers):
        ax.text(0.02, y, f"{info['short']} ({info['module']})", fontsize=9.6, fontweight="bold", va="center", color=PALETTE["cobalt"])
        ax.text(0.32, y, info["q7_summary"], fontsize=9.2, va="center", color=PALETTE["ink"],
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#FFF1F2", edgecolor=PALETTE["crimson"], lw=1.0))
        ax.annotate("", xy=(0.665, y), xytext=(0.605, y),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.45", color=PALETTE["amber"], lw=2.0))
        ax.text(0.68, y, info["q8_summary"], fontsize=9.2, va="center", color=PALETTE["ink"],
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#ECFDF5", edgecolor=PALETTE["teal"], lw=1.0))

    ax.set_xlim(0, 1)
    ax.set_ylim(0.04, 1.02)
    fig.suptitle("Module 08 — Tier 1 (Paper Audit Protocol): Q7 Limitations Directly Generate Q8 Research Hypotheses", fontsize=12.5, fontweight="bold", y=0.98)
    fig.tight_layout()
    return fig, {"n_papers": len(papers), "papers": [p["short"] for p in papers]}


def run_module08_tier2(signal_snr: float = 1.05, alpha: float = 0.05) -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 2: Track A BH-FDR power calibration across T=100..1200 solving minimum T_80."""
    return create_module08_dashboard_figure(
        paper_choice="Track A Power & BH-FDR Calibration",
        track="Track A: Observational FDR & Power",
        sample_size=650,
        signal_snr=signal_snr,
        alpha=alpha,
    )


def run_module08_tier3(coupling_strength: float = 0.70) -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 3: Track B Null Baseline & Cohen's d + Example 5-Part Research Brief export."""
    fig, stats = create_module08_dashboard_figure(
        paper_choice="Track B Null Model vs. Mechanistic ABM",
        track="Track B: Generative ABM vs Null Model",
        coupling_strength=coupling_strength,
    )
    brief = build_research_brief(
        title="Greedy Conditional Transfer Entropy Under Non-Stationary Regime Shifts",
        paper_key="1. Schreiber (2000) - Transfer Entropy",
        gap_analysis="Bivariate TE fails under hidden confounders, and static conditional TE assumes time-invariant transition densities.",
        hypothesis="Windowed adaptive KSG Conditional TE combined with BH-FDR (T >= T_80) suppresses confounder false positives across regime shifts.",
        model_arch="3-tier sliding-window KSG estimator with circular time-shift surrogates and Benjamini-Hochberg step-up thresholding.",
        validation_plan="Track A Monte Carlo calibration solving T_80 for 80% power alongside Track B shuffled-phase null model comparison (Cohen's d > 1.5).",
        expected_claim="Conditioned multi-lag inference eliminates >90% of indirect cascade/common-driver artifacts while controlling FDP <= 0.05.",
    )
    stats["sample_brief"] = brief
    return fig, stats


def generate_module08_gifs(out_dir: str | Path) -> list[Path]:
    """Generate classroom-ready animated GIFs for Module 08."""
    apply_publication_style(dpi=110)
    anim_dir = Path(out_dir) / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    cal = calibrate_fdr_power_curve(50, 8, signal_snr=1.05, alpha=0.05, seed=42)
    t_grid = cal["t_grid"]
    frames = []
    for idx, t_val in enumerate(t_grid):
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
        fig.patch.set_facecolor(PALETTE["bg"])

        ax0 = style_axes(axes[0], title=f"Track A Rates at Sample T = {t_val}", ylabel="Rate / Proportion", grid_axis="y")
        xp = np.arange(2)
        ax0.bar(xp - 0.17, [cal["raw_tpr"][idx], cal["raw_fdp"][idx]], width=0.34, color=PALETTE["crimson"], edgecolor=PALETTE["ink"], label="Uncorrected (p<0.05)")
        ax0.bar(xp + 0.17, [cal["fdr_tpr"][idx], cal["fdr_fdp"][idx]], width=0.34, color=PALETTE["teal"], edgecolor=PALETTE["ink"], label="BH-FDR (q<0.05)")
        ax0.set_xticks(xp)
        ax0.set_xticklabels(["Power (TPR)", "False Disc. (FDP)"], fontweight="bold")
        ax0.set_ylim(0, 1.12)
        ax0.legend(loc="upper right", fontsize=8.2)

        ax1 = style_axes(axes[1], title="Power Calibration Trajectory (Target = 80%)", xlabel="Sample Length $T$", ylabel="Detection Power (TPR)")
        ax1.plot(t_grid[: idx + 1], cal["fdr_tpr"][: idx + 1], marker="o", lw=2.4, color=PALETTE["cobalt"], label="BH-FDR Power")
        ax1.axhline(0.80, color=PALETTE["amber"], linestyle="--", lw=1.6, label="80% Benchmark")
        ax1.set_xlim(60, 1240)
        ax1.set_ylim(0, 1.06)
        ax1.legend(loc="lower right", fontsize=8.2)

        fig.suptitle("Module 08 Live Track A Calibration: Solving Minimum Sample Size T_80", fontsize=12, fontweight="bold")
        fig.tight_layout()
        frames.append(fig_to_pil_image(fig, dpi=105))
        plt.close(fig)

    generated.append(save_gif_from_frames(frames, anim_dir / "01_trackA_bh_fdr_power_calibration.gif", duration_ms=420, hold_last_frames=4))
    return generated
