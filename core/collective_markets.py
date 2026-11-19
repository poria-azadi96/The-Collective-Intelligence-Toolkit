"""Module 07: Page's DPT, Hegselmann-Krause Bounded Confidence, El Farol & Minority Game."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from core.plotting_utils import (
    PALETTE,
    apply_publication_style,
    fig_to_pil_image,
    save_gif_from_frames,
    style_axes,
)


def simulate_dpt_ensemble(
    n_agents: int = 50,
    indiv_sigma: float = 10.0,
    error_corr: float = 0.25,
    n_trials: int = 400,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate Page's exact single-trial DPT identity and ex-ante correlated ensemble error."""
    rng = np.random.default_rng(seed)
    theta_true = 100.0
    rho = float(np.clip(error_corr, 0.0, 0.99))
    z_common = rng.normal(0.0, 1.0, size=(n_trials, 1))
    z_indep = rng.normal(0.0, 1.0, size=(n_trials, n_agents))
    errors = indiv_sigma * (np.sqrt(rho) * z_common + np.sqrt(1.0 - rho) * z_indep)
    preds = theta_true + errors

    mean_pred = np.mean(preds, axis=1, keepdims=True)
    e_coll = float(np.mean((mean_pred[:, 0] - theta_true) ** 2))
    e_indiv = float(np.mean(np.mean((preds - theta_true) ** 2, axis=1)))
    div_pred = float(np.mean(np.mean((preds - mean_pred) ** 2, axis=1)))

    # Single-trial exact algebraic identity verification
    p0 = preds[0]
    m0 = float(np.mean(p0))
    exact_coll = float((m0 - theta_true) ** 2)
    exact_indiv = float(np.mean((p0 - theta_true) ** 2))
    exact_div = float(np.mean((p0 - m0) ** 2))
    identity_residual = abs((exact_indiv - exact_div) - exact_coll)
    theory_coll = (indiv_sigma**2) * (1.0 / n_agents + (1.0 - 1.0 / n_agents) * rho)

    return {
        "e_coll": e_coll,
        "e_indiv": e_indiv,
        "div_pred": div_pred,
        "theory_coll": float(theory_coll),
        "exact_single_trial": (exact_coll, exact_indiv, exact_div),
        "identity_residual": float(identity_residual),
    }


def simulate_degroot_consensus(
    n_agents: int = 50,
    steps: int = 25,
    seed: int = 42,
) -> np.ndarray:
    """Simulate linear DeGroot social averaging baseline converging to global consensus."""
    rng = np.random.default_rng(seed)
    opinions = np.zeros((steps, n_agents), dtype=np.float64)
    opinions[0] = rng.uniform(0.0, 1.0, size=n_agents)
    # Row-stochastic strongly connected trust matrix
    raw_w = rng.uniform(0.1, 1.0, size=(n_agents, n_agents)) + 4.0 * np.eye(n_agents)
    W = raw_w / np.sum(raw_w, axis=1, keepdims=True)
    for t in range(steps - 1):
        opinions[t + 1] = W.dot(opinions[t])
    return opinions


def simulate_hk_opinions(
    n_agents: int = 80,
    confidence_eps: float = 0.18,
    social_noise_q: float = 0.0,
    steps: int = 32,
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    """Evolve continuous Hegselmann-Krause bounded-confidence opinions x_i in [0, 1]."""
    rng = np.random.default_rng(seed)
    opinions = np.zeros((steps, n_agents), dtype=np.float64)
    opinions[0] = rng.uniform(0.0, 1.0, size=n_agents)

    for t in range(steps - 1):
        curr = opinions[t]
        diff_mat = np.abs(curr[:, None] - curr[None, :])
        in_bound = (diff_mat <= float(confidence_eps)).astype(np.float64)
        counts = np.sum(in_bound, axis=1)
        next_op = in_bound.dot(curr) / np.maximum(counts, 1.0)
        if social_noise_q > 0.0:
            next_op = np.clip(next_op + rng.normal(0.0, 0.025 * social_noise_q, size=n_agents), 0.0, 1.0)
        opinions[t + 1] = next_op

    final_sorted = np.sort(opinions[-1])
    n_clusters = 1 + int(np.sum(np.diff(final_sorted) > 0.03))
    return opinions, n_clusters


def simulate_el_farol_bar(
    n_agents: int = 100,
    capacity: int = 60,
    memory_m: int = 4,
    n_strategies: int = 6,
    weeks: int = 110,
    seed: int = 42,
) -> np.ndarray:
    """Simulate Brian Arthur's (1994) inductive El Farol Bar problem."""
    rng = np.random.default_rng(seed)
    history = list(rng.integers(35, 85, size=max(10, memory_m + 6)))
    weights = rng.uniform(-0.8, 0.8, size=(n_agents, n_strategies, memory_m))
    biases = rng.uniform(20.0, 80.0, size=(n_agents, n_strategies))
    attendance_series = []

    for _ in range(weeks):
        recent = np.array(history[-memory_m:], dtype=np.float64)
        preds = np.clip(np.tensordot(weights, recent, axes=([2], [0])) + biases, 0.0, float(n_agents))

        strat_errors = np.zeros((n_agents, n_strategies), dtype=np.float64)
        for back in range(1, 5):
            past_window = np.array(history[-(memory_m + back) : -back], dtype=np.float64)
            actual_val = float(history[-back])
            past_preds = np.tensordot(weights, past_window, axes=([2], [0])) + biases
            strat_errors += np.abs(past_preds - actual_val)

        best_strat = np.argmin(strat_errors, axis=1)
        chosen_forecast = preds[np.arange(n_agents), best_strat]
        decisions = (chosen_forecast < capacity).astype(np.int32)
        total_att = int(np.sum(decisions))
        attendance_series.append(total_att)
        history.append(total_att)

    return np.array(attendance_series)


def simulate_minority_game(
    n_agents: int = 101,
    memory_m: int = 3,
    herding_coupling: float = 0.0,
    steps: int = 180,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulate Challet-Zhang (1997) Minority Game + optional Networked Social Herding."""
    rng = np.random.default_rng(seed)
    n_states = 1 << int(memory_m)
    strategies = rng.choice([-1, 1], size=(n_agents, 2, n_states))
    scores = np.zeros((n_agents, 2), dtype=np.float64)
    state = int(rng.integers(0, n_states))
    excess_demand = []

    clique_leader = (np.arange(n_agents) // 6) * 6
    for _ in range(steps):
        chosen_s = (scores[:, 1] > scores[:, 0]).astype(np.int32)
        actions = strategies[np.arange(n_agents), chosen_s, state].copy()

        if herding_coupling > 0.0:
            imitate_mask = rng.random(n_agents) < float(herding_coupling)
            # Networked herding cluster: follow local clique hub creating correlated demand cascades
            actions[imitate_mask] = actions[clique_leader[imitate_mask]]

        a_t = int(np.sum(actions))
        excess_demand.append(a_t)
        minority_sign = -1 if a_t > 0 else 1

        for s_idx in (0, 1):
            scores[:, s_idx] += strategies[:, s_idx, state] * minority_sign

        winning_bit = 1 if minority_sign == 1 else 0
        state = ((state << 1) | winning_bit) & (n_states - 1)

    norm_volatility = float(np.var(excess_demand[steps // 4 :])) / float(n_agents)
    alpha = float(n_states) / float(n_agents)
    return {
        "excess_demand": np.array(excess_demand),
        "volatility": norm_volatility,
        "alpha": alpha,
    }


def create_module07_dashboard_figure(
    study: str = "1. Diversity Prediction Theorem",
    rho: float = 0.25,
    eps: float = 0.18,
    memory_m: int = 3,
    herding_q: float = 0.35,
    herding_ext: bool = False,
    seed: int = 42,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Build classroom dashboard for Module 07 models."""
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    fig.patch.set_facecolor(PALETTE["bg"])

    if study.startswith("1.") or "Diversity" in study:
        dpt = simulate_dpt_ensemble(n_agents=50, indiv_sigma=10.0, error_corr=rho, n_trials=400, seed=seed)
        rhos = np.linspace(0.05, 0.90, 18)
        n_ag, sig = 50, 10.0
        theory_coll = (sig**2) * (1.0 / n_ag + (1.0 - 1.0 / n_ag) * rhos)
        emp_coll = [simulate_dpt_ensemble(n_ag, sig, float(r), 250, seed=seed)["e_coll"] for r in rhos]

        ax0 = style_axes(axes[0], title="1. Error Correlation $\\rho$ Drives Collective Error", xlabel="Pairwise Error Correlation $\\rho$", ylabel="Mean Squared Error")
        ax0.plot(rhos, theory_coll, color=PALETTE["cobalt"], lw=2.4, label="Theory $\\mathbb{E}[E_{\\text{coll}}] = \\sigma^2(\\frac{1}{N} + (1-\\frac{1}{N})\\rho)$")
        ax0.scatter(rhos, emp_coll, color=PALETTE["crimson"], s=42, zorder=5, label="Monte Carlo Ensemble")
        ax0.axhline(sig**2, color=PALETTE["slate"], linestyle="--", label="Single Member Error ($\\sigma^2 = 100$)")
        ax0.legend(loc="upper left", fontsize=8.1)

        ax1 = style_axes(axes[1], title=f"2. DPT Identity at $\\rho = {rho:.2f}$", ylabel="Squared Error Units", grid_axis="y")
        labels = ["Individual Error\n$E_{\\text{indiv}}$", "Collective Error\n$E_{\\text{coll}}$", "Prediction Diversity\n$D$"]
        vals = [dpt["e_indiv"], dpt["e_coll"], dpt["div_pred"]]
        bars = ax1.bar(labels, vals, color=[PALETTE["slate"], PALETTE["cobalt"], PALETTE["teal"]], width=0.5, edgecolor=PALETTE["ink"])
        ax1.set_ylim(0, 128)
        for b in bars:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2, h + 2.2, f"{h:.1f}", ha="center", fontweight="bold", fontsize=9.2)

        fig.suptitle("Module 07 Dashboard — Scott Page's Diversity Prediction Theorem", fontsize=12.5, fontweight="bold", y=1.01)
        fig.tight_layout()
        return fig, {"study": "DPT", **dpt}

    if study.startswith("2.") or "Bounded" in study:
        q_op = herding_q if herding_ext else 0.0
        opinions, n_clus = simulate_hk_opinions(n_agents=80, confidence_eps=eps, social_noise_q=q_op, steps=32, seed=seed)
        ax0 = style_axes(axes[0], title=f"1. Hegselmann-Krause Trajectories ($\\epsilon={eps:.2f}$ | Clusters={n_clus})", xlabel="Interaction Step $t$", ylabel="Opinion $x_i(t)$")
        for i in range(opinions.shape[1]):
            ax0.plot(opinions[:, i], color=PALETTE["cobalt"], alpha=0.35, lw=1.1)
        ax0.set_ylim(-0.03, 1.03)

        eps_grid = np.linspace(0.05, 0.45, 17)
        clus_counts = [simulate_hk_opinions(80, float(e), q_op, 32, seed=seed)[1] for e in eps_grid]
        ax1 = style_axes(axes[1], title="2. Opinion Fragmentation $\\to$ Consensus Bifurcation", xlabel="Confidence Bound $\\epsilon$", ylabel="Final Opinion Clusters")
        ax1.plot(eps_grid, clus_counts, marker="o", color=PALETTE["amber"], lw=2.3, label="Hegselmann-Krause")
        ax1.axhline(1.0, color=PALETTE["teal"], linestyle="--", lw=1.8, label="DeGroot Linear Consensus (1 Cluster)")
        ax1.scatter([eps], [n_clus], color=PALETTE["crimson"], s=95, zorder=6, label=f"Current ($\\epsilon={eps:.2f}$)")
        ax1.legend(loc="upper right", fontsize=8.3)

        fig.suptitle("Module 07 Dashboard — Bounded Confidence Opinion Dynamics", fontsize=12.5, fontweight="bold", y=1.01)
        fig.tight_layout()
        return fig, {"study": "HK", "eps": eps, "n_clusters": n_clus, "eps_grid": eps_grid.tolist(), "cluster_counts": clus_counts}

    # Model 3: El Farol Bar & Minority Game
    att = simulate_el_farol_bar(n_agents=100, capacity=60, memory_m=memory_m, weeks=105, seed=seed)
    ax0 = style_axes(axes[0], title=f"1. Arthur (1994) El Farol Bar ($M={memory_m}$, Mean={np.mean(att):.1f})", xlabel="Week $t$", ylabel="Bar Attendance $A(t)$")
    ax0.plot(att, color=PALETTE["cobalt"], lw=1.9, label="Attendance $A(t)$")
    ax0.axhline(60.0, color=PALETTE["crimson"], linestyle="--", lw=1.8, label="Capacity Threshold $L=60$")
    ax0.set_ylim(15, 100)
    ax0.legend(loc="upper right", fontsize=8.4)

    m_vals = [1, 2, 3, 4, 5, 6, 7]
    alphas, vols_indep, vols_herd = [], [], []
    for m in m_vals:
        r0 = simulate_minority_game(101, memory_m=m, herding_coupling=0.0, steps=170, seed=seed)
        rh = simulate_minority_game(101, memory_m=m, herding_coupling=herding_q, steps=170, seed=seed)
        alphas.append(r0["alpha"])
        vols_indep.append(r0["volatility"])
        vols_herd.append(rh["volatility"])

    cusp_idx = int(np.argmin(vols_indep))
    alpha_c = alphas[cusp_idx]

    ax1 = style_axes(axes[1], title=f"2. Minority Game Volatility Cusp ($\\alpha_c \\approx {alpha_c:.3f}$)", xlabel="Control Parameter $\\alpha = 2^M / N$ (Log)", ylabel="Volatility $\\sigma^2 / N$ (Log)")
    ax1.loglog(alphas, vols_indep, marker="o", color=PALETTE["teal"], lw=2.3, label="Independent MG")
    ax1.scatter([alpha_c], [vols_indep[cusp_idx]], color=PALETTE["amber"], s=110, zorder=6, label=f"Critical Cusp $\\alpha_c={alpha_c:.2f}$")
    if herding_ext:
        ax1.loglog(alphas, vols_herd, marker="s", color=PALETTE["crimson"], lw=2.2, linestyle="--", label=f"Networked Herding ($q={herding_q:.2f}$)")
    ax1.axhline(1.0, color=PALETTE["slate"], linestyle=":", lw=1.4, label="Coin-Flip Baseline ($\\sigma^2/N=1$)")
    y_lo = min(0.45, float(min(vols_indep)) * 0.72)
    y_hi = max(float(max(vols_indep)), float(max(vols_herd))) * 1.45
    ax1.set_ylim(y_lo, y_hi)
    ax1.legend(loc="upper right", fontsize=8.0)

    fig.suptitle("Module 07 Dashboard — Inductive Rationality, El Farol & Minority Game", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {
        "study": "ElFarol_MG",
        "el_farol_mean": float(np.mean(att)),
        "el_farol_std": float(np.std(att)),
        "alpha_c": float(alpha_c),
        "min_volatility_indep": float(vols_indep[cusp_idx]),
        "herding_volatility_at_M": float(vols_herd[min(memory_m - 1, len(vols_herd) - 1)]),
    }


def run_module07_tier1(seed: int = 42) -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 1: Sweep Error Correlation rho 0.05 -> 0.90 verifying Page's exact identity."""
    return create_module07_dashboard_figure(study="1. Diversity Prediction Theorem", rho=0.25, seed=seed)


def run_module07_tier2(seed: int = 42) -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 2: HK Bounded Confidence at eps=0.05, 0.18, 0.40 & DeGroot baseline comparison."""
    apply_publication_style()
    presets = [(0.05, "Fragmentation"), (0.18, "Polarization"), (0.40, "Global Consensus")]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.3))
    fig.patch.set_facecolor(PALETTE["bg"])
    summary = {}

    for ax, (eps, regime) in zip(axes[:3], presets):
        ops, n_c = simulate_hk_opinions(80, confidence_eps=eps, steps=32, seed=seed)
        style_axes(ax, title=f"HK $\\epsilon={eps:.2f}$ ({regime})\nClusters = {n_c}", xlabel="Step $t$", ylabel="Opinion $x_i$")
        for i in range(ops.shape[1]):
            ax.plot(ops[:, i], color=PALETTE["cobalt"], alpha=0.35, lw=1.0)
        ax.set_ylim(-0.03, 1.03)
        summary[f"eps={eps:.2f}"] = n_c

    dg_ops = simulate_degroot_consensus(n_agents=60, steps=22, seed=seed)
    style_axes(axes[3], title="DeGroot Linear Consensus\n(Always 1 Cluster)", xlabel="Step $t$", ylabel="Opinion $x_i$")
    for i in range(dg_ops.shape[1]):
        axes[3].plot(dg_ops[:, i], color=PALETTE["teal"], alpha=0.45, lw=1.0)
    axes[3].set_ylim(-0.03, 1.03)

    fig.suptitle("Module 07 — Tier 2 (Opinion Bifurcation): Bounded Confidence vs. DeGroot Consensus", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, summary


def run_module07_tier3(herding_q: float = 0.40, seed: int = 42) -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 3: El Farol stabilization, Minority Game cusp alpha_c & Networked Herding spike."""
    return create_module07_dashboard_figure(
        study="3. El Farol Bar & Minority Game", memory_m=3, herding_q=herding_q, herding_ext=True, seed=seed
    )


def generate_module07_gifs(out_dir: str | Path) -> list[Path]:
    """Generate classroom-ready animated GIFs for Module 07."""
    apply_publication_style(dpi=110)
    anim_dir = Path(out_dir) / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # GIF 1: Hegselmann-Krause Bifurcation Sweep eps = 0.04 -> 0.42
    eps_vals = np.linspace(0.04, 0.42, 20)
    clus_all = [simulate_hk_opinions(80, float(e), steps=30, seed=42) for e in eps_vals]
    frames1 = []
    for idx, eps in enumerate(eps_vals):
        ops, n_c = clus_all[idx]
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
        fig.patch.set_facecolor(PALETTE["bg"])

        ax0 = style_axes(axes[0], title=f"HK Opinion Trajectories ($\\epsilon={eps:.2f}$, Clusters={n_c})", xlabel="Step $t$", ylabel="Opinion $x_i$")
        for i in range(ops.shape[1]):
            ax0.plot(ops[:, i], color=PALETTE["cobalt"], alpha=0.35, lw=1.0)
        ax0.set_ylim(-0.03, 1.03)

        ax1 = style_axes(axes[1], title="Cluster Bifurcation Curve", xlabel="Confidence Bound $\\epsilon$", ylabel="Cluster Count")
        ax1.plot(eps_vals[: idx + 1], [c[1] for c in clus_all[: idx + 1]], marker="o", color=PALETTE["amber"], lw=2.2)
        ax1.axhline(1.0, color=PALETTE["teal"], linestyle="--", lw=1.5)
        ax1.set_xlim(0.02, 0.44)
        ax1.set_ylim(0, max(c[1] for c in clus_all) + 1)

        fig.suptitle("Module 07 Live Sweep: Echo Chamber Fragmentation to Consensus", fontsize=12, fontweight="bold")
        fig.tight_layout()
        frames1.append(fig_to_pil_image(fig, dpi=105))
        plt.close(fig)
    generated.append(save_gif_from_frames(frames1, anim_dir / "01_hk_opinion_bifurcation.gif", duration_ms=170))

    # GIF 2: Minority Game Volatility Spike Across Herding Coupling q = 0.0 -> 0.60
    q_vals = np.linspace(0.0, 0.60, 13)
    m_vals = [1, 2, 3, 4, 5, 6]
    base_res = [simulate_minority_game(101, memory_m=m, herding_coupling=0.0, steps=160, seed=42) for m in m_vals]
    alphas = [r["alpha"] for r in base_res]
    base_vols = [r["volatility"] for r in base_res]
    att = simulate_el_farol_bar(n_agents=100, capacity=60, memory_m=4, weeks=80, seed=42)

    frames2 = []
    for idx, q in enumerate(q_vals):
        h_vols = [simulate_minority_game(101, memory_m=m, herding_coupling=float(q), steps=160, seed=42)["volatility"] for m in m_vals]
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
        fig.patch.set_facecolor(PALETTE["bg"])

        ax0 = style_axes(axes[0], title=f"El Farol Attendance Wave (Mean={np.mean(att):.1f})", xlabel="Week $t$", ylabel="Attendance")
        ax0.plot(att, color=PALETTE["cobalt"], lw=1.9)
        ax0.axhline(60, color=PALETTE["crimson"], linestyle="--", lw=1.5)
        ax0.set_ylim(20, 95)

        ax1 = style_axes(axes[1], title=f"Minority Game + Social Herding ($q={q:.2f}$)", xlabel="$\\alpha = 2^M/N$", ylabel="Volatility $\\sigma^2/N$")
        ax1.loglog(alphas, base_vols, marker="o", color=PALETTE["teal"], lw=2.2, label="Independent ($q=0$)")
        ax1.loglog(alphas, h_vols, marker="s", color=PALETTE["crimson"], lw=2.3, linestyle="--", label=f"Herding ($q={q:.2f}$)")
        ax1.axhline(1.0, color=PALETTE["slate"], linestyle=":", lw=1.3)
        ax1.set_ylim(0.08, 12.0)
        ax1.legend(loc="lower right", fontsize=8.3)

        fig.suptitle("Module 07 Networked Herding Triggers Excess Market Volatility", fontsize=12, fontweight="bold")
        fig.tight_layout()
        frames2.append(fig_to_pil_image(fig, dpi=105))
        plt.close(fig)
    generated.append(save_gif_from_frames(frames2, anim_dir / "02_el_farol_and_minority_herding.gif", duration_ms=220))
    return generated
