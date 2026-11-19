"""Module 05: Random Boolean Networks (RBNs), Derrida Criticality & Immune Selection."""

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


def derrida_sensitivity(k_inputs: int, bias_p: float) -> float:
    """Compute Kauffman RBN annealed sensitivity s = 2 * K * p * (1 - p)."""
    return float(2.0 * int(k_inputs) * float(bias_p) * (1.0 - float(bias_p)))


def derrida_annealed_theory(d_grid: np.ndarray, k_inputs: int, bias_p: float) -> np.ndarray:
    """Compute theoretical annealed Derrida map d(t+1) = 2p(1-p) * [1 - (1 - d(t))^K]."""
    p = float(bias_p)
    return 2.0 * p * (1.0 - p) * (1.0 - np.power(1.0 - d_grid, int(k_inputs)))


def _make_canalizing_table(k_inputs: int, bias_p: float, rng: np.random.Generator) -> np.ndarray:
    """Construct a biologically plausible canalizing Boolean truth table of arity K."""
    n_rows = 1 << k_inputs
    if k_inputs <= 1:
        return (rng.random(n_rows) < bias_p).astype(np.uint8)
    can_var = int(rng.integers(0, k_inputs))
    can_in_val = int(rng.integers(0, 2))
    can_out_val = int(rng.random() < bias_p)
    table = (rng.random(n_rows) < bias_p).astype(np.uint8)
    for idx in range(n_rows):
        bit = (idx >> (k_inputs - 1 - can_var)) & 1
        if bit == can_in_val:
            table[idx] = can_out_val
    return table


def generate_rbn(
    n_nodes: int = 10,
    k_inputs: int = 2,
    bias_p: float = 0.5,
    canalizing: bool = False,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a heterogeneous RBN with N nodes, in-degree K, and Boolean bias p."""
    rng = np.random.default_rng(seed)
    k_eff = min(int(k_inputs), n_nodes - 1)
    wiring = np.zeros((n_nodes, k_eff), dtype=np.int32)
    for i in range(n_nodes):
        choices = np.delete(np.arange(n_nodes), i)
        wiring[i] = rng.choice(choices, size=k_eff, replace=False)

    if canalizing:
        truth_tables = np.vstack(
            [_make_canalizing_table(k_eff, bias_p, rng) for _ in range(n_nodes)]
        ).astype(np.uint8)
    else:
        truth_tables = (rng.random((n_nodes, 1 << k_eff)) < float(bias_p)).astype(np.uint8)
    return wiring, truth_tables


def step_rbn_sync(
    states: np.ndarray,
    wiring: np.ndarray,
    truth_tables: np.ndarray,
    knockout_node: int = -1,
) -> np.ndarray:
    """Vectorized synchronous update over batch of binary states (shape [batch, N])."""
    _, n_nodes = states.shape
    k_inputs = wiring.shape[1]
    next_states = np.zeros_like(states)
    weights = (1 << np.arange(k_inputs - 1, -1, -1)).astype(np.int32)
    for i in range(n_nodes):
        if i == knockout_node:
            next_states[:, i] = 0
            continue
        inp = states[:, wiring[i]]
        idx = inp.dot(weights)
        next_states[:, i] = truth_tables[i, idx]
    return next_states


def compute_derrida_curve(
    wiring: np.ndarray,
    truth_tables: np.ndarray,
    n_samples: int = 150,
    knockout_node: int = -1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical Derrida damage spreading curve H(t+1) vs H(t)."""
    rng = np.random.default_rng(seed)
    n_nodes = wiring.shape[0]
    d_in_list, d_out_list = [], []
    for num_flips in range(1, n_nodes + 1):
        s1 = rng.integers(0, 2, size=(n_samples, n_nodes), dtype=np.uint8)
        s2 = s1.copy()
        for b in range(n_samples):
            flip_idx = rng.choice(n_nodes, size=num_flips, replace=False)
            s2[b, flip_idx] ^= 1
        s1_next = step_rbn_sync(s1, wiring, truth_tables, knockout_node=knockout_node)
        s2_next = step_rbn_sync(s2, wiring, truth_tables, knockout_node=knockout_node)
        d_in = num_flips / float(n_nodes)
        d_out = float(np.mean(np.abs(s1_next.astype(np.int32) - s2_next.astype(np.int32))))
        d_in_list.append(d_in)
        d_out_list.append(d_out)
    return np.array(d_in_list), np.array(d_out_list)


def analyze_exact_state_space(
    wiring: np.ndarray,
    truth_tables: np.ndarray,
    knockout_node: int = -1,
) -> tuple[list[list[int]], list[int]]:
    """Exhaustively enumerate all 2^N states for exact synchronous attractors and basins."""
    n_nodes = wiring.shape[0]
    num_states = 1 << n_nodes
    all_states = (
        (np.arange(num_states)[:, None] & (1 << np.arange(n_nodes - 1, -1, -1))) > 0
    ).astype(np.uint8)
    if knockout_node >= 0:
        all_states[:, knockout_node] = 0
    next_states = step_rbn_sync(all_states, wiring, truth_tables, knockout_node=knockout_node)
    weights = (1 << np.arange(n_nodes - 1, -1, -1)).astype(np.int32)
    next_idx = next_states.dot(weights)

    visited = np.full(num_states, -1, dtype=np.int32)
    attractors: list[list[int]] = []
    basin_sizes: list[int] = []
    for start in range(num_states):
        if visited[start] != -1:
            continue
        path: list[int] = []
        curr = start
        while visited[curr] == -1:
            visited[curr] = -2
            path.append(curr)
            curr = int(next_idx[curr])
        if visited[curr] == -2:
            cycle_start = path.index(curr)
            cycle = path[cycle_start:]
            att_id = len(attractors)
            attractors.append(cycle)
            basin_sizes.append(len(path))
            for node in path:
                visited[node] = att_id
        else:
            att_id = int(visited[curr])
            basin_sizes[att_id] += len(path)
            for node in path:
                visited[node] = att_id

    # Order attractors from largest basin to smallest so bar charts highlight dominant fates consistently
    order = np.argsort(basin_sizes)[::-1]
    attractors = [attractors[i] for i in order]
    basin_sizes = [basin_sizes[i] for i in order]
    return attractors, basin_sizes


def sample_asynchronous_trajectories(
    wiring: np.ndarray,
    truth_tables: np.ndarray,
    n_trials: int = 120,
    steps: int = 160,
    knockout_node: int = -1,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulate random asynchronous updates (Greil & Drossel) to quantify point vs loose attractors."""
    rng = np.random.default_rng(seed)
    n_nodes, k_inputs = wiring.shape
    weights = (1 << np.arange(k_inputs - 1, -1, -1)).astype(np.int32)
    states = rng.integers(0, 2, size=(n_trials, n_nodes), dtype=np.uint8)
    if knockout_node >= 0:
        states[:, knockout_node] = 0

    for _ in range(steps):
        update_nodes = rng.integers(0, n_nodes, size=n_trials)
        for tr in range(n_trials):
            u = int(update_nodes[tr])
            if u == knockout_node:
                states[tr, u] = 0
                continue
            inp = states[tr, wiring[u]]
            idx = int(inp.dot(weights))
            states[tr, u] = truth_tables[u, idx]

    sync_next = step_rbn_sync(states, wiring, truth_tables, knockout_node=knockout_node)
    is_fixed = np.all(states == sync_next, axis=1)
    # Also measure distinct terminal bitstrings visited across trials (loose-attractor dispersion)
    unique_terminal = len({tuple(row.tolist()) for row in states})
    return {
        "fixed_fraction": float(np.mean(is_fixed)),
        "loose_fraction": float(1.0 - np.mean(is_fixed)),
        "unique_terminal_states": int(unique_terminal),
    }


def simulate_immune_selection(
    n_clones: int = 70,
    n_steps: int = 45,
    self_threshold: float = 0.72,
    mutation_std: float = 0.04,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulate distributed immune receptor negative selection & 45-gen clonal affinity maturation."""
    rng = np.random.default_rng(seed)
    dim = 16
    self_antigen = rng.random(dim)
    pathogen = 1.0 - self_antigen

    receptors_init = rng.random((n_clones, dim))
    self_aff_init = 1.0 - np.mean(np.abs(receptors_init - self_antigen), axis=1)
    surviving = receptors_init[self_aff_init < self_threshold]
    if len(surviving) < 5:
        surviving = receptors_init[:5]

    mean_path_aff, max_path_aff = [], []
    pop = surviving.copy()
    history_pops = []
    for _ in range(n_steps):
        aff = 1.0 - np.mean(np.abs(pop - pathogen), axis=1)
        mean_path_aff.append(float(np.mean(aff)))
        max_path_aff.append(float(np.max(aff)))
        history_pops.append(pop.copy())
        probs = np.exp(8.0 * aff)
        probs /= np.sum(probs)
        idx = rng.choice(len(pop), size=n_clones, p=probs)
        pop = np.clip(pop[idx] + rng.normal(0.0, mutation_std, size=(n_clones, dim)), 0.0, 1.0)

    final_self_aff = 1.0 - np.mean(np.abs(pop - self_antigen), axis=1)
    return {
        "mean_affinity": np.array(mean_path_aff),
        "max_affinity": np.array(max_path_aff),
        "n_purged_self": int(np.sum(self_aff_init >= self_threshold)),
        "final_mean_self_cross": float(np.mean(final_self_aff)),
        "self_antigen": self_antigen,
        "pathogen": pathogen,
        "history_pops": history_pops,
    }


def kinetic_proofreading_specificity(
    max_steps: int = 10,
    kp: float = 1.0,
    koff_foreign: float = 0.25,
    koff_self: float = 1.25,
) -> dict[str, Any]:
    """Compute McKeithan (1995) multistep kinetic proofreading ligand discrimination gain."""
    steps_n = np.arange(1, int(max_steps) + 1)
    sig_foreign = np.power(kp / (kp + koff_foreign), steps_n)
    sig_self = np.power(kp / (kp + koff_self), steps_n)
    specificity = sig_foreign / sig_self
    return {
        "steps": steps_n,
        "sig_foreign": sig_foreign,
        "sig_self": sig_self,
        "specificity_gain": specificity,
    }


def create_module05_dashboard_figure(
    mode: str = "1. RBN Dynamics & Derrida Curve",
    k_inputs: int = 2,
    bias_p: float = 0.50,
    knockout_gene: bool = False,
    update_timing: str = "Synchronous",
    proofreading_steps: int = 8,
    seed: int = 42,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Build Module 05 classroom dashboard supporting both RBN Dynamics and Immune Selection."""
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    fig.patch.set_facecolor(PALETTE["bg"])

    if mode.startswith("1.") or "RBN" in mode:
        wiring, tt = generate_rbn(n_nodes=10, k_inputs=k_inputs, bias_p=bias_p, seed=seed)
        ko_node = 0 if knockout_gene else -1
        d_in, d_out = compute_derrida_curve(wiring, tt, n_samples=150, knockout_node=ko_node, seed=seed)
        d_grid = np.linspace(0.0, 1.0, 120)
        d_theory = derrida_annealed_theory(d_grid, k_inputs=k_inputs, bias_p=bias_p)
        sens_s = derrida_sensitivity(k_inputs=k_inputs, bias_p=bias_p)

        ax0 = style_axes(
            axes[0],
            title=f"1. Derrida Curve (K={k_inputs}, p={bias_p:.2f} | s={sens_s:.2f})",
            xlabel="Initial Perturbation $H(t)$",
            ylabel="Next-Step Divergence $H(t+1)$",
        )
        ax0.plot(d_grid, d_grid, "k--", alpha=0.55, lw=1.4, label="Critical $45^\\circ$ Diagonal ($H(t+1)=H(t)$)")
        ax0.plot(d_grid, d_theory, color=PALETTE["teal"], lw=2.4, label="Annealed Theory")
        ax0.scatter(d_in, d_out, color=PALETTE["crimson"], s=52, zorder=5, label="Empirical RBN ($N=10$)")
        ax0.set_xlim(0.0, 1.02)
        ax0.set_ylim(0.0, 0.72)
        ax0.legend(loc="lower right", fontsize=8.4)

        atts_wt, basins_wt = analyze_exact_state_space(wiring, tt, knockout_node=-1)
        atts, basins = analyze_exact_state_space(wiring, tt, knockout_node=ko_node)
        async_stats = sample_asynchronous_trajectories(wiring, tt, n_trials=120, steps=160, knockout_node=ko_node, seed=seed)
        lengths = [len(a) for a in atts]

        ax1 = style_axes(
            axes[1],
            title=(
                f"2. {update_timing} Basins ({'KO Node 0' if knockout_gene else 'Wild-Type'} | "
                f"Sync={len(atts)}, Async Fixed={async_stats['fixed_fraction']:.0%})"
            ),
            xlabel="Attractor ID ($L$ = Sync Cycle Period)",
            ylabel="Basin Size (States out of 1024)",
            grid_axis="y",
        )
        xp = np.arange(1, len(basins) + 1)
        bar_col = PALETTE["amber"] if knockout_gene else PALETTE["cobalt"]
        bars = ax1.bar(xp, basins, color=bar_col, edgecolor=PALETTE["ink"], width=0.58)
        for bar, l_val in zip(bars, lengths):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 14, f"L={l_val}", ha="center", fontsize=8.5, fontweight="bold")
        ax1.set_ylim(0, 1140)
        ax1.text(
            0.97,
            0.94,
            f"WT Attractors: {len(atts_wt)}\nCurrent Attractors: {len(atts)}\nAsync Fixed: {async_stats['fixed_fraction']:.1%}\nAsync Loose: {async_stats['loose_fraction']:.1%}",
            transform=ax1.transAxes,
            ha="right",
            va="top",
            fontsize=8.4,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["card"], edgecolor=PALETTE["cobalt"]),
        )

        regime = "ORDERED (s < 1)" if sens_s < 0.95 else ("CHAOTIC (s > 1)" if sens_s > 1.05 else "CRITICAL EDGE (s ~ 1)")
        fig.suptitle(f"Module 05 Dashboard — {regime} | Timing={update_timing}", fontsize=12.5, fontweight="bold", y=1.01)
        fig.tight_layout()
        return fig, {
            "mode": "RBN",
            "sensitivity_s": sens_s,
            "regime": regime,
            "n_attractors_wt": len(atts_wt),
            "n_attractors": len(atts),
            "basin_sizes": basins,
            "cycle_lengths": lengths,
            "async_fixed_fraction": async_stats["fixed_fraction"],
        }

    # Mode 2: Distributed Immune Selection & Kinetic Proofreading
    imm = simulate_immune_selection(n_clones=70, n_steps=45, self_threshold=0.72, seed=seed)
    kp_res = kinetic_proofreading_specificity(max_steps=10)
    n_idx = max(1, min(10, int(proofreading_steps))) - 1

    ax0 = style_axes(
        axes[0],
        title="1. Clonal Selection & 45-Gen Affinity Maturation",
        xlabel="Generation",
        ylabel="Pathogen Binding Affinity",
    )
    gens = np.arange(1, len(imm["mean_affinity"]) + 1)
    ax0.plot(gens, imm["mean_affinity"], color=PALETTE["cobalt"], lw=2.4, label="Population Mean Affinity")
    ax0.plot(gens, imm["max_affinity"], color=PALETTE["amber"], lw=2.2, linestyle="--", label="Best Clone Affinity")
    ax0.set_ylim(0.45, 1.02)
    ax0.legend(loc="lower right", fontsize=8.6)

    ax1 = style_axes(
        axes[1],
        title=f"2. McKeithan Kinetic Proofreading ($N={proofreading_steps} \\rightarrow {kp_res['specificity_gain'][n_idx]:.1f}\\times$)",
        xlabel="Proofreading Steps $N$",
        ylabel="Foreign / Self Discrimination Ratio (Log)",
    )
    ax1.semilogy(kp_res["steps"], kp_res["specificity_gain"], marker="o", lw=2.4, color=PALETTE["crimson"], label="Discrimination $(k_p/(k_p+k_{\\text{off,f}}))^N / \\text{self}^N$")
    ax1.scatter([kp_res["steps"][n_idx]], [kp_res["specificity_gain"][n_idx]], color=PALETTE["teal"], s=110, zorder=6, label=f"Selected N={proofreading_steps} ({kp_res['specificity_gain'][n_idx]:.1f}x)")
    ax1.legend(loc="upper left", fontsize=8.2)

    fig.suptitle("Module 05 Dashboard — Distributed Immune Cognition & Kinetic Proofreading", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {
        "mode": "Immune",
        "initial_mean_affinity": float(imm["mean_affinity"][0]),
        "final_mean_affinity": float(imm["mean_affinity"][-1]),
        "final_max_affinity": float(imm["max_affinity"][-1]),
        "purged_self_clones": imm["n_purged_self"],
        "selected_proofreading_gain": float(kp_res["specificity_gain"][n_idx]),
    }


def run_module05_tier1(bias_p: float = 0.50, seed: int = 42) -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 1: Sweep In-Degree K from 1 to 4 at p=0.50 showing Derrida curve crossing 45-deg line."""
    apply_publication_style()
    k_vals = [1, 2, 3, 4]
    cols = {1: PALETTE["cobalt"], 2: PALETTE["teal"], 3: PALETTE["amber"], 4: PALETTE["crimson"]}
    d_grid = np.linspace(0.0, 1.0, 120)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax0 = style_axes(axes[0], title=f"Tier 1A: Derrida Transition Across $K=1..4$ ($p={bias_p:.2f}$)", xlabel="Initial $H(t)$", ylabel="Next-Step $H(t+1)$")
    ax0.plot(d_grid, d_grid, "k--", lw=1.4, alpha=0.6, label="$45^\\circ$ Critical Diagonal ($s=1$)")
    sens_map = {}
    for k in k_vals:
        w, tt = generate_rbn(n_nodes=10, k_inputs=k, bias_p=bias_p, seed=seed)
        din, dout = compute_derrida_curve(w, tt, n_samples=120, seed=seed)
        s = derrida_sensitivity(k, bias_p)
        sens_map[k] = s
        ax0.plot(d_grid, derrida_annealed_theory(d_grid, k, bias_p), color=cols[k], lw=2.1, label=f"K={k} ($s={s:.2f}$)")
        ax0.scatter(din, dout, color=cols[k], s=26, alpha=0.8)
    ax0.set_xlim(0.0, 1.0)
    ax0.set_ylim(0.0, 0.72)
    ax0.legend(loc="lower right", fontsize=8.3)

    ax1 = style_axes(axes[1], title="Tier 1B: Sensitivity $s = 2Kp(1-p)$ Crossing $s=1$", xlabel="In-Degree $K$", ylabel="Network Sensitivity $s$", grid_axis="y")
    bars = ax1.bar([str(k) for k in k_vals], [sens_map[k] for k in k_vals], color=[cols[k] for k in k_vals], width=0.5, edgecolor=PALETTE["ink"])
    ax1.axhline(1.0, color=PALETTE["ink"], linestyle="--", lw=1.5, label="Critical Boundary $s = 1$")
    ax1.set_ylim(0.0, 2.35)
    for b in bars:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 0.05, f"s={h:.2f}", ha="center", fontweight="bold", fontsize=9.2)
    ax1.legend(loc="upper left", fontsize=8.6)

    fig.suptitle("Module 05 — Tier 1 (Observation): Ordered (K=1), Critical (K=2) & Chaotic (K=3,4) Regimes", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {"sensitivities": sens_map}


def run_module05_tier2(k_inputs: int = 2, bias_p: float = 0.50, seed: int = 6) -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 2: Toggle Knockout Node 0 at K=2 & compare Synchronous vs Asynchronous attractors."""
    apply_publication_style()
    w, tt = generate_rbn(n_nodes=10, k_inputs=k_inputs, bias_p=bias_p, seed=seed)
    atts_wt, basins_wt = analyze_exact_state_space(w, tt, knockout_node=-1)
    atts_ko, basins_ko = analyze_exact_state_space(w, tt, knockout_node=0)
    async_wt = sample_asynchronous_trajectories(w, tt, n_trials=120, steps=160, knockout_node=-1, seed=seed)
    async_ko = sample_asynchronous_trajectories(w, tt, n_trials=120, steps=160, knockout_node=0, seed=seed)

    # Also evaluate K=1..4 sync attractor counts vs async point-attractor convergence
    k_grid = [1, 2, 3, 4]
    sync_counts, async_fracs = [], []
    for k in k_grid:
        wk, ttk = generate_rbn(n_nodes=10, k_inputs=k, bias_p=bias_p, seed=seed)
        ak, _ = analyze_exact_state_space(wk, ttk, knockout_node=-1)
        ask = sample_asynchronous_trajectories(wk, ttk, n_trials=100, steps=150, seed=seed)
        sync_counts.append(len(ak))
        async_fracs.append(ask["fixed_fraction"])

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax0 = style_axes(axes[0], title=f"Tier 2A: Basin Merging on Knockout Node 0 (K={k_inputs})", xlabel="Attractor ID", ylabel="Basin Size (out of 1024)", grid_axis="y")
    max_b = max(len(basins_wt), len(basins_ko))
    b_wt_pad = basins_wt + [0] * (max_b - len(basins_wt))
    b_ko_pad = basins_ko + [0] * (max_b - len(basins_ko))
    xp = np.arange(1, max_b + 1)
    ax0.bar(xp - 0.18, b_wt_pad, width=0.36, color=PALETTE["cobalt"], edgecolor=PALETTE["ink"], label=f"Wild-Type ({len(basins_wt)} basins)")
    ax0.bar(xp + 0.18, b_ko_pad, width=0.36, color=PALETTE["amber"], edgecolor=PALETTE["ink"], label=f"KO Node 0 ({len(basins_ko)} basins)")
    ax0.set_xticks(xp)
    ax0.legend(loc="upper right", fontsize=8.5)

    ax1 = style_axes(axes[1], title="Tier 2B: Sync Attractor Count vs. Async Point-Attractor Fraction", xlabel="In-Degree $K$", ylabel="Synchronous Attractors")
    ax1.bar(np.array(k_grid) - 0.16, sync_counts, width=0.32, color=PALETTE["teal"], edgecolor=PALETTE["ink"], label="Sync Attractor Count")
    ax1.set_ylim(0, max(sync_counts) * 1.28 + 1.0)
    ax1_r = ax1.twinx()
    ax1_r.plot(k_grid, np.array(async_fracs) * 100.0, marker="o", lw=2.4, color=PALETTE["crimson"], label="Async Fixed-Point %")
    ax1_r.set_ylabel("Async Fixed Convergence (%)", fontsize=9.8, fontweight="bold", color=PALETTE["crimson"])
    ax1_r.set_ylim(0, 118)
    ax1_r.spines["top"].set_visible(False)
    ax1.set_xticks(k_grid)
    ax1.legend(loc="upper left", fontsize=8.2)
    ax1_r.legend(loc="upper right", fontsize=8.2)

    fig.suptitle("Module 05 — Tier 2 (Knockout & Timing): Structural Basin Shifts & Sync vs. Async Attractors", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {
        "wt_attractors": len(atts_wt),
        "ko_attractors": len(atts_ko),
        "wt_basins": basins_wt,
        "ko_basins": basins_ko,
        "wt_async_fixed": async_wt["fixed_fraction"],
        "ko_async_fixed": async_ko["fixed_fraction"],
        "sync_counts_k1_4": sync_counts,
        "async_fixed_k1_4": async_fracs,
    }


def run_module05_tier3(seed: int = 42) -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 3: 45-generation Immune Clonal Maturation & N=1..10 Exponential Kinetic Proofreading."""
    apply_publication_style()
    imm = simulate_immune_selection(n_clones=70, n_steps=45, self_threshold=0.72, seed=seed)
    kp = kinetic_proofreading_specificity(max_steps=10)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax0 = style_axes(axes[0], title="Tier 3A: Distributed Clonal Affinity Maturation (45 Gens)", xlabel="Generation", ylabel="Binding Affinity")
    gens = np.arange(1, 46)
    ax0.plot(gens, imm["mean_affinity"], color=PALETTE["cobalt"], lw=2.4, label="Mean Clone Affinity")
    ax0.plot(gens, imm["max_affinity"], color=PALETTE["amber"], lw=2.2, linestyle="--", label="Best Clone Affinity")
    ax0.set_ylim(0.45, 1.02)
    ax0.legend(loc="lower right", fontsize=8.6)

    ax1 = style_axes(axes[1], title="Tier 3B: Exponential Kinetic Proofreading Gain ($N=1..10$)", xlabel="Proofreading Steps $N$", ylabel="Foreign / Self Ratio (Log)")
    ax1.semilogy(kp["steps"], kp["specificity_gain"], marker="o", lw=2.5, color=PALETTE["crimson"], label="Discrimination Ratio")
    ax1.annotate(f"N=10: {kp['specificity_gain'][-1]:.1f}x", xy=(10, kp["specificity_gain"][-1]), xytext=(6.6, kp["specificity_gain"][-1] * 0.45),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["ink"]), fontsize=9.0, fontweight="bold")
    ax1.set_xticks(kp["steps"])
    ax1.legend(loc="upper left", fontsize=8.6)

    fig.suptitle("Module 05 — Tier 3 (Advanced): Immune Tolerance, Somatic Hypermutation & Kinetic Proofreading", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {
        "final_mean_affinity": float(imm["mean_affinity"][-1]),
        "final_max_affinity": float(imm["max_affinity"][-1]),
        "gain_step_1": float(kp["specificity_gain"][0]),
        "gain_step_10": float(kp["specificity_gain"][-1]),
    }


def generate_module05_gifs(out_dir: str | Path) -> list[Path]:
    """Generate classroom-ready animated GIFs for Module 05."""
    apply_publication_style(dpi=110)
    anim_dir = Path(out_dir) / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # GIF 1: RBN Order -> Critical -> Chaos Transition across K=1..4 & Bias sweep
    d_grid = np.linspace(0.0, 1.0, 100)
    configs = [
        (1, 0.50, "Ordered (K=1, p=0.50)"),
        (3, 0.15, "Ordered via Bias (K=3, p=0.15)"),
        (2, 0.50, "Critical Edge (K=2, p=0.50)"),
        (3, 0.50, "Chaotic (K=3, p=0.50)"),
        (4, 0.50, "Deep Chaos (K=4, p=0.50)"),
    ]
    frames1 = []
    for k, p, label in configs:
        w, tt = generate_rbn(n_nodes=10, k_inputs=k, bias_p=p, seed=42)
        din, dout = compute_derrida_curve(w, tt, n_samples=120, seed=42)
        atts, basins = analyze_exact_state_space(w, tt)
        s = derrida_sensitivity(k, p)

        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
        fig.patch.set_facecolor(PALETTE["bg"])
        ax0 = style_axes(axes[0], title=f"Derrida Map: {label} (s={s:.2f})", xlabel="Initial $H(t)$", ylabel="Next $H(t+1)$")
        ax0.plot(d_grid, d_grid, "k--", lw=1.3, alpha=0.6)
        ax0.plot(d_grid, derrida_annealed_theory(d_grid, k, p), color=PALETTE["teal"], lw=2.4, label="Theory")
        ax0.scatter(din, dout, color=PALETTE["crimson"], s=45, label="Empirical")
        ax0.set_xlim(0, 1)
        ax0.set_ylim(0, 0.72)
        ax0.legend(loc="lower right", fontsize=8.4)

        ax1 = style_axes(axes[1], title=f"Attractor Basins (Total={len(atts)})", xlabel="Attractor ID", ylabel="Basin Size", grid_axis="y")
        ax1.bar(range(1, len(basins) + 1), basins, color=PALETTE["cobalt"], edgecolor=PALETTE["ink"])
        ax1.set_ylim(0, 1100)
        fig.suptitle("Module 05 RBN Order-to-Chaos Phase Transition", fontsize=12, fontweight="bold")
        fig.tight_layout()
        frames1.append(fig_to_pil_image(fig, dpi=105))
        plt.close(fig)
    generated.append(save_gif_from_frames(frames1, anim_dir / "01_rbn_derrida_phase_transition.gif", duration_ms=900, hold_last_frames=2))

    # GIF 2: 45-Gen Immune Clonal Selection & Affinity Maturation
    imm = simulate_immune_selection(n_clones=70, n_steps=45, self_threshold=0.72, seed=42)
    frames2 = []
    # Project 16D receptor space onto two fixed informative axes (affinity to self vs affinity to pathogen)
    for g in range(0, 45, 2):
        pop = imm["history_pops"][g]
        self_a = 1.0 - np.mean(np.abs(pop - imm["self_antigen"]), axis=1)
        path_a = 1.0 - np.mean(np.abs(pop - imm["pathogen"]), axis=1)
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
        fig.patch.set_facecolor(PALETTE["bg"])

        ax0 = style_axes(axes[0], title=f"B-Cell Repertoire (Gen {g + 1}/45)", xlabel="Self-Antigen Affinity", ylabel="Pathogen Affinity")
        ax0.axvline(0.72, color=PALETTE["crimson"], linestyle="--", lw=1.4, label="Self-Tolerance Cutoff (0.72)")
        ax0.scatter(self_a, path_a, color=PALETTE["teal"], s=48, alpha=0.75, edgecolor=PALETTE["ink"])
        ax0.set_xlim(0.25, 0.85)
        ax0.set_ylim(0.35, 0.98)
        ax0.legend(loc="upper right", fontsize=8.0)

        ax1 = style_axes(axes[1], title="Clonal Affinity Maturation Curve", xlabel="Generation", ylabel="Pathogen Affinity")
        xs = np.arange(1, g + 2)
        ax1.plot(xs, imm["mean_affinity"][: g + 1], color=PALETTE["cobalt"], lw=2.3, label="Mean Affinity")
        ax1.plot(xs, imm["max_affinity"][: g + 1], color=PALETTE["amber"], lw=2.1, linestyle="--", label="Best Clone")
        ax1.set_xlim(1, 45)
        ax1.set_ylim(0.45, 1.02)
        ax1.legend(loc="lower right", fontsize=8.3)

        fig.suptitle("Module 05 Live Immune Clonal Selection & Somatic Hypermutation", fontsize=12, fontweight="bold")
        fig.tight_layout()
        frames2.append(fig_to_pil_image(fig, dpi=105))
        plt.close(fig)
    generated.append(save_gif_from_frames(frames2, anim_dir / "02_immune_clonal_selection_maturation.gif", duration_ms=140))
    return generated
