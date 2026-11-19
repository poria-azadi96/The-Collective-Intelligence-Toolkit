"""Module 06: Physarum Sparse Laplacian Solver, Complex Contagion & Vicsek Flocking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from scipy.sparse.linalg import splu, spsolve
from scipy.spatial import Delaunay

from core.plotting_utils import (
    PALETTE,
    apply_publication_style,
    fig_to_pil_image,
    save_gif_from_frames,
    style_axes,
)

ACTIVE_CONDUCTANCE_THRESH = 0.06


def build_tokyo_benchmark_graph() -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    """Create a 24-station planar urban rail benchmark centered on Tokyo Hub (node 0)."""
    coords = np.array(
        [
            [0.52, 0.48],  # 0: Tokyo Central Hub
            [0.45, 0.52], [0.43, 0.45], [0.48, 0.40], [0.56, 0.42], [0.58, 0.54],
            [0.36, 0.58], [0.34, 0.47], [0.37, 0.36], [0.46, 0.30], [0.59, 0.31],
            [0.67, 0.43], [0.66, 0.58], [0.55, 0.65], [0.44, 0.66], [0.25, 0.68],
            [0.22, 0.50], [0.24, 0.32], [0.35, 0.20], [0.52, 0.18], [0.71, 0.24],
            [0.78, 0.46], [0.75, 0.68], [0.58, 0.78],
        ],
        dtype=np.float64,
    )
    tri = Delaunay(coords)
    n_nodes = len(coords)
    adj_len = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for idx_a in range(3):
            u, v = int(simplex[idx_a]), int(simplex[(idx_a + 1) % 3])
            dist = float(np.linalg.norm(coords[u] - coords[v]))
            if dist < 0.36:
                edge = (min(u, v), max(u, v))
                edges.add(edge)
                adj_len[u, v] = adj_len[v, u] = dist
    return coords, sorted(edges), adj_len


def solve_physarum_network(
    coords: np.ndarray,
    edges: list[tuple[int, int]],
    adj_len: np.ndarray,
    mu: float = 1.4,
    gamma: float = 1.0,
    steps: int = 150,
    disrupt_hub_edges: bool = False,
    nonuniform_demand: bool = False,
    seed: int = 42,
    record_history: bool = False,
) -> dict[str, Any]:
    """Solve Tero et al. (2010) sparse-Laplacian Physarum dynamics with optional edge severing."""
    rng = np.random.default_rng(seed)
    n_nodes = len(coords)
    n_edges = len(edges)
    D = np.ones(n_edges, dtype=np.float64) * 0.5
    lengths = np.array([adj_len[u, v] for (u, v) in edges], dtype=np.float64)
    disrupted_mask = np.zeros(n_edges, dtype=bool)
    d_history = []
    mean_flux_history = []

    # Optional non-uniform demand weights favoring central hub & eastern industrial corridor
    node_weights = np.ones(n_nodes, dtype=np.float64)
    if nonuniform_demand:
        node_weights[0] = 6.0
        node_weights[[11, 12, 21, 22]] = 3.0
    node_probs = node_weights / np.sum(node_weights)

    cut_step = steps // 2
    keep = list(range(1, n_nodes))
    n_pairs = 8
    flux_drive = 2.75
    edges_u = np.array([u for (u, _) in edges], dtype=np.int32)
    edges_v = np.array([v for (_, v) in edges], dtype=np.int32)

    for step in range(steps):
        if disrupt_hub_edges and step == cut_step:
            top_two = np.argsort(D)[-2:]
            disrupted_mask[top_two] = True
            D[top_two] = 1e-6

        conductance = D / lengths
        # Assemble sparse weighted graph Laplacian L(D)
        L_sparse = lil_matrix((n_nodes, n_nodes), dtype=np.float64)
        for e_idx, (u, v) in enumerate(edges):
            c = float(conductance[e_idx])
            L_sparse[u, u] += c
            L_sparse[v, v] += c
            L_sparse[u, v] -= c
            L_sparse[v, u] -= c

        # Ground reference node 0 (p[0] = 0) and solve n_pairs station-to-station Kirchhoff drives in one LU pass
        L_red = L_sparse.tocsr()[keep, :][:, keep].tolil()
        for k_idx in range(len(keep)):
            L_red[k_idx, k_idx] += 1e-9
        L_red_csc = L_red.tocsc()

        rhs_mat = np.zeros((len(keep), n_pairs), dtype=np.float64)
        for p_idx in range(n_pairs):
            if nonuniform_demand and p_idx < n_pairs // 2:
                src = int(rng.choice(n_nodes, p=node_probs))
                snk = int(rng.choice(n_nodes, p=node_probs))
            else:
                src = int(rng.integers(0, n_nodes))
                snk = int(rng.integers(0, n_nodes))
            while snk == src:
                snk = int(rng.integers(0, n_nodes))
            if src > 0:
                rhs_mat[src - 1, p_idx] += flux_drive
            if snk > 0:
                rhs_mat[snk - 1, p_idx] -= flux_drive

        try:
            p_red = splu(L_red_csc).solve(rhs_mat)
        except Exception:
            p_red = np.column_stack([spsolve(L_red_csc.tocsr(), rhs_mat[:, j]) for j in range(n_pairs)])

        pressures = np.zeros((n_nodes, n_pairs), dtype=np.float64)
        pressures[1:, :] = p_red
        flux = conductance * np.mean(np.abs(pressures[edges_u, :] - pressures[edges_v, :]), axis=1)
        f_pow = np.power(flux, float(mu))
        dt = 0.25
        D = D + dt * ((f_pow / (1.0 + f_pow)) - float(gamma) * D)
        D[disrupted_mask] = 1e-6
        D = np.clip(D, 1e-6, 5.0)
        mean_flux_history.append(float(np.mean(flux)))
        if record_history:
            d_history.append(D.copy())

    return {
        "D": D,
        "disrupted_mask": disrupted_mask,
        "cut_step": cut_step if disrupt_hub_edges else None,
        "d_history": d_history,
        "mean_flux_history": mean_flux_history,
    }


def compute_network_metrics(
    n_nodes: int,
    edges: list[tuple[int, int]],
    lengths: np.ndarray,
    active_mask: np.ndarray,
) -> tuple[float, float, float]:
    """Compute Total Length (Cost), Global Efficiency, and Single-Edge Fault Tolerance."""
    dist_mat = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    active_edges: list[tuple[int, int, float]] = []
    total_cost = 0.0
    for e_idx, (u, v) in enumerate(edges):
        if active_mask[e_idx]:
            dist_mat[u, v] = dist_mat[v, u] = lengths[e_idx]
            total_cost += float(lengths[e_idx])
            active_edges.append((u, v, float(lengths[e_idx])))

    sp = shortest_path(dist_mat, directed=False)
    inv_sp = np.divide(1.0, sp, out=np.zeros_like(sp, dtype=float), where=(sp > 0) & np.isfinite(sp))
    global_eff = float(np.sum(inv_sp) / (n_nodes * (n_nodes - 1)))

    if not active_edges:
        return total_cost, global_eff, 0.0
    connected_survivals = 0
    for drop_idx in range(len(active_edges)):
        sub_mat = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        for idx_e, (u, v, w) in enumerate(active_edges):
            if idx_e != drop_idx:
                sub_mat[u, v] = sub_mat[v, u] = w
        sub_sp = shortest_path(sub_mat, directed=False)
        if np.all(np.isfinite(sub_sp)):
            connected_survivals += 1
    fault_tol = float(connected_survivals) / float(len(active_edges))
    return total_cost, global_eff, fault_tol


def simulate_contagion_comparison(
    rewire_prob: float = 0.15,
    complex_threshold: int = 2,
    n_nodes: int = 120,
    steps: int = 35,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare Simple (SIR-like) vs Complex (Threshold T) Contagion on a Watts-Strogatz graph."""
    rng = np.random.default_rng(seed)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int32)
    for i in range(n_nodes):
        for offset in (1, 2):
            j = (i + offset) % n_nodes
            adj[i, j] = adj[j, i] = 1
    for i in range(n_nodes):
        for offset in (1, 2):
            if rng.random() < float(rewire_prob):
                j = (i + offset) % n_nodes
                adj[i, j] = adj[j, i] = 0
                new_j = int(rng.integers(0, n_nodes))
                if new_j != i:
                    adj[i, new_j] = adj[new_j, i] = 1

    G = nx.from_numpy_array(adj)
    clustering = float(nx.average_clustering(G))

    seed_count = max(3, complex_threshold + 1)
    simple_state = np.zeros(n_nodes, dtype=np.int32)
    complex_state = np.zeros(n_nodes, dtype=np.int32)
    simple_state[:seed_count] = 1
    complex_state[:seed_count] = 1
    simple_curve = [float(np.mean(simple_state))]
    complex_curve = [float(np.mean(complex_state))]

    for _ in range(steps):
        active_s = adj.dot(simple_state)
        new_s = (rng.random(n_nodes) < (1.0 - (1.0 - 0.18) ** active_s)).astype(np.int32)
        simple_state = np.maximum(simple_state, new_s)

        active_c = adj.dot(complex_state)
        new_c = (active_c >= int(complex_threshold)).astype(np.int32)
        complex_state = np.maximum(complex_state, new_c)

        simple_curve.append(float(np.mean(simple_state)))
        complex_curve.append(float(np.mean(complex_state)))

    return {
        "simple_curve": np.array(simple_curve),
        "complex_curve": np.array(complex_curve),
        "clustering": clustering,
        "final_simple": float(simple_curve[-1]),
        "final_complex": float(complex_curve[-1]),
    }


def simulate_vicsek_flocking(
    n_birds: int = 140,
    noise_eta: float = 0.6,
    box_l: float = 10.0,
    r_int: float = 1.3,
    v0: float = 0.25,
    steps: int = 90,
    seed: int = 42,
    record_history: bool = False,
) -> dict[str, Any]:
    """Simulate 2D self-propelled Vicsek flocking and return order parameter phi(t)."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, box_l, size=(n_birds, 2))
    angles = rng.uniform(-np.pi, np.pi, size=n_birds)
    order_hist = []
    pos_hist = []
    angle_hist = []

    for _ in range(steps):
        if record_history:
            pos_hist.append(pos.copy())
            angle_hist.append(angles.copy())
        dx = pos[:, 0:1] - pos[:, 0:1].T
        dy = pos[:, 1:2] - pos[:, 1:2].T
        dx -= box_l * np.round(dx / box_l)
        dy -= box_l * np.round(dy / box_l)
        dist_sq = dx**2 + dy**2
        neighbors = dist_sq <= (r_int**2)
        mean_sin = neighbors.dot(np.sin(angles))
        mean_cos = neighbors.dot(np.cos(angles))
        angles = np.arctan2(mean_sin, mean_cos) + rng.uniform(-noise_eta / 2.0, noise_eta / 2.0, size=n_birds)
        pos[:, 0] = (pos[:, 0] + v0 * np.cos(angles)) % box_l
        pos[:, 1] = (pos[:, 1] + v0 * np.sin(angles)) % box_l
        phi = float(np.sqrt(np.mean(np.cos(angles)) ** 2 + np.mean(np.sin(angles)) ** 2))
        order_hist.append(phi)

    return {
        "pos": pos,
        "angles": angles,
        "order_hist": np.array(order_hist),
        "final_phi": float(order_hist[-1]),
        "mean_tail_phi": float(np.mean(order_hist[-20:])),
        "pos_hist": pos_hist,
        "angle_hist": angle_hist,
    }


def _draw_physarum_panel(
    ax: plt.Axes,
    coords: np.ndarray,
    edges: list[tuple[int, int]],
    D_vals: np.ndarray,
    disrupted: np.ndarray,
    title: str,
) -> None:
    ax.set_facecolor(PALETTE["bg"])
    max_d = max(float(np.max(D_vals)), 1e-6)
    for e_idx, (u, v) in enumerate(edges):
        if disrupted[e_idx]:
            ax.plot([coords[u, 0], coords[v, 0]], [coords[u, 1], coords[v, 1]], color=PALETTE["crimson"], ls="--", lw=1.9, alpha=0.85)
        elif D_vals[e_idx] > ACTIVE_CONDUCTANCE_THRESH:
            lw = 0.8 + 3.4 * (D_vals[e_idx] / max_d)
            ax.plot([coords[u, 0], coords[v, 0]], [coords[u, 1], coords[v, 1]], color=PALETTE["teal"], lw=lw, alpha=0.92)
        else:
            ax.plot([coords[u, 0], coords[v, 0]], [coords[u, 1], coords[v, 1]], color=PALETTE["grid"], lw=0.6, alpha=0.7)
    ax.scatter(coords[1:, 0], coords[1:, 1], color=PALETTE["ink"], s=36, zorder=5, label="Stations")
    ax.scatter([coords[0, 0]], [coords[0, 1]], color=PALETTE["crimson"], s=130, marker="*", zorder=6, label="Tokyo Hub")
    ax.set_title(title, fontsize=10.8, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(loc="lower left", fontsize=8.2)


def create_module06_dashboard_figure(
    mode: str = "Physarum Tokyo Rail",
    mu: float = 1.4,
    gamma: float = 1.0,
    disrupt_edges: bool = False,
    rewire_prob: float = 0.15,
    contagion_threshold: int = 2,
    vicsek_eta: float = 0.6,
    n_birds: int = 140,
    seed: int = 42,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Build classroom dashboard for any of the three Module 06 adaptive/spatial models."""
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    fig.patch.set_facecolor(PALETTE["bg"])

    if "Physarum" in mode:
        coords, edges, adj_len = build_tokyo_benchmark_graph()
        lengths = np.array([adj_len[u, v] for (u, v) in edges])
        res = solve_physarum_network(
            coords, edges, adj_len, mu=mu, gamma=gamma, steps=150, disrupt_hub_edges=disrupt_edges, seed=seed
        )
        D_final, disrupted = res["D"], res["disrupted_mask"]
        active_phy = D_final > ACTIVE_CONDUCTANCE_THRESH
        mst_sparse = minimum_spanning_tree(adj_len).toarray()
        active_mst = np.array([(mst_sparse[u, v] > 0 or mst_sparse[v, u] > 0) for (u, v) in edges])

        c_phy, e_phy, ft_phy = compute_network_metrics(len(coords), edges, lengths, active_phy)
        c_mst, e_mst, ft_mst = compute_network_metrics(len(coords), edges, lengths, active_mst)
        c_raw, e_raw, ft_raw = compute_network_metrics(len(coords), edges, lengths, np.ones(len(edges), dtype=bool))

        _draw_physarum_panel(axes[0], coords, edges, D_final, disrupted, f"1. Physarum Backbone ($\\mu={mu:.2f}, \\gamma={gamma:.2f}$ | Cut={disrupt_edges})")
        ax1 = style_axes(axes[1], title="2. Tero et al. (2010) Multi-Objective Benchmark", ylabel="Ratio vs. MST / Fraction", grid_axis="y")
        cats = ["Normalized Cost", "Global Efficiency", "Fault Tolerance"]
        xp = np.arange(3)
        ax1.bar(xp - 0.22, [1.0, 1.0, ft_mst], width=0.22, color=PALETTE["slate"], edgecolor=PALETTE["ink"], label="MST Baseline")
        ax1.bar(xp, [c_phy / c_mst, e_phy / e_mst, ft_phy], width=0.22, color=PALETTE["teal"], edgecolor=PALETTE["ink"], label="Physarum Adaptive")
        ax1.bar(xp + 0.22, [c_raw / c_mst, e_raw / e_mst, ft_raw], width=0.22, color=PALETTE["amber"], edgecolor=PALETTE["ink"], label="Full Mesh")
        ax1.set_xticks(xp)
        ax1.set_xticklabels(cats, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=8.3)

        fig.suptitle("Module 06 Dashboard — Physarum Sparse-Laplacian Self-Organization", fontsize=12.5, fontweight="bold", y=1.01)
        fig.tight_layout()
        return fig, {
            "mode": "Physarum",
            "cost_ratio": c_phy / c_mst,
            "efficiency_ratio": e_phy / e_mst,
            "fault_tolerance": ft_phy,
            "mst_fault_tolerance": ft_mst,
        }

    if "Contagion" in mode:
        res = simulate_contagion_comparison(rewire_prob=rewire_prob, complex_threshold=contagion_threshold, seed=seed)
        ax0 = style_axes(axes[0], title=f"1. Adoption Dynamics ($p_{{\\mathrm{{rewire}}}}={rewire_prob:.2f}, T={contagion_threshold}$)", xlabel="Time Step $t$", ylabel="Adopter Fraction")
        ax0.plot(res["simple_curve"], color=PALETTE["cobalt"], lw=2.4, label="Simple Contagion")
        ax0.plot(res["complex_curve"], color=PALETTE["crimson"], lw=2.4, linestyle="--", label=f"Complex Contagion ($T \\geq {contagion_threshold}$)")
        ax0.set_ylim(-0.02, 1.05)
        ax0.legend(loc="lower right", fontsize=8.5)

        p_sweep = np.linspace(0.0, 1.0, 11)
        s_fin, c_fin, clusts = [], [], []
        for pv in p_sweep:
            r_sw = simulate_contagion_comparison(rewire_prob=float(pv), complex_threshold=contagion_threshold, seed=seed)
            s_fin.append(r_sw["final_simple"])
            c_fin.append(r_sw["final_complex"])
            clusts.append(r_sw["clustering"])

        ax1 = style_axes(axes[1], title="2. Shortcuts Destroy Clustering & Complex Contagion", xlabel="Rewiring Probability $p_{\\mathrm{rewire}}$", ylabel="Final Reach / Clustering")
        ax1.plot(p_sweep, s_fin, marker="o", lw=2.3, color=PALETTE["cobalt"], label="Simple Contagion Reach")
        ax1.plot(p_sweep, c_fin, marker="s", lw=2.3, color=PALETTE["crimson"], label="Complex Contagion Reach")
        ax1.plot(p_sweep, clusts, marker="^", lw=1.8, linestyle=":", color=PALETTE["amber"], label="Clustering Coefficient $C$")
        ax1.set_ylim(-0.02, 1.05)
        ax1.legend(loc="best", fontsize=8.2)

        fig.suptitle("Module 06 Dashboard — Simple vs. Complex Social Contagion (Centola, 2010)", fontsize=12.5, fontweight="bold", y=1.01)
        fig.tight_layout()
        return fig, {
            "mode": "Contagion",
            "clustering": res["clustering"],
            "final_simple": res["final_simple"],
            "final_complex": res["final_complex"],
            "p_sweep": p_sweep.tolist(),
            "complex_reach_sweep": c_fin,
        }

    # Vicsek Flocking
    v_res = simulate_vicsek_flocking(n_birds=n_birds, noise_eta=vicsek_eta, steps=90, seed=seed)
    ax0 = style_axes(axes[0], title=f"1. 2D Vicsek Flocking Snapshot ($\\eta={vicsek_eta:.2f}, N={n_birds}$)", xlabel="$x$", ylabel="$y$", grid_axis=None)
    pos, ang = v_res["pos"], v_res["angles"]
    ax0.quiver(pos[:, 0], pos[:, 1], np.cos(ang), np.sin(ang), color=PALETTE["cobalt"], scale=26)
    ax0.set_xlim(0, 10)
    ax0.set_ylim(0, 10)
    ax0.set_aspect("equal")

    etas = np.linspace(0.0, 2.0 * np.pi, 13)
    phis = [simulate_vicsek_flocking(n_birds=n_birds, noise_eta=float(e), steps=85, seed=seed)["mean_tail_phi"] for e in etas]
    ax1 = style_axes(axes[1], title=f"2. Collective Alignment Phase Transition ($\\phi={v_res['final_phi']:.3f}$)", xlabel="Angular Noise $\\eta$ (rad)", ylabel="Order Parameter $\\phi(\\eta)$")
    ax1.plot(etas, phis, marker="o", lw=2.4, color=PALETTE["teal"], label="$\\phi(\\eta)$ Phase Curve")
    ax1.scatter([vicsek_eta], [v_res["mean_tail_phi"]], color=PALETTE["crimson"], s=95, zorder=6, label=f"Current ($\\eta={vicsek_eta:.2f}$)")
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(loc="upper right", fontsize=8.5)

    fig.suptitle("Module 06 Dashboard — Vicsek Self-Propelled Collective Motion", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {"mode": "Vicsek", "final_phi": v_res["final_phi"], "etas": etas.tolist(), "phis": phis}


def run_module06_tier1(mu: float = 1.4, gamma: float = 1.0) -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 1: Physarum self-organization on Tokyo rail network comparing against MST."""
    return create_module06_dashboard_figure(mode="Physarum Tokyo Rail", mu=mu, gamma=gamma, disrupt_edges=False)


def run_module06_tier2(complex_threshold: int = 2) -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 2: Sweep Watts-Strogatz rewiring p_rewire 0.0 -> 1.0 showing clustering & cascade collapse."""
    return create_module06_dashboard_figure(mode="Complex Contagion", rewire_prob=0.35, contagion_threshold=complex_threshold)


def run_module06_tier3(seed: int = 42) -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 3: Vicsek phase transition sweep + Physarum live trunk edge severing under non-uniform demand."""
    apply_publication_style()
    etas = np.linspace(0.0, 2.0 * np.pi, 13)
    phis = [simulate_vicsek_flocking(n_birds=140, noise_eta=float(e), steps=85, seed=seed)["mean_tail_phi"] for e in etas]

    coords, edges, adj_len = build_tokyo_benchmark_graph()
    lengths = np.array([adj_len[u, v] for (u, v) in edges])
    res_cut = solve_physarum_network(
        coords, edges, adj_len, mu=1.30, gamma=1.0, steps=160, disrupt_hub_edges=True, nonuniform_demand=True, seed=seed
    )
    active_cut = res_cut["D"] > ACTIVE_CONDUCTANCE_THRESH
    c_cut, e_cut, ft_cut = compute_network_metrics(len(coords), edges, lengths, active_cut)
    mst_sparse = minimum_spanning_tree(adj_len).toarray()
    active_mst = np.array([(mst_sparse[u, v] > 0 or mst_sparse[v, u] > 0) for (u, v) in edges])
    c_mst, e_mst, _ = compute_network_metrics(len(coords), edges, lengths, active_mst)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax0 = style_axes(axes[0], title="Tier 3A: Vicsek Order-Disorder Phase Transition $\\phi(\\eta)$", xlabel="Angular Noise $\\eta$ (0 to $2\\pi$)", ylabel="Polarization $\\phi$")
    ax0.plot(etas, phis, marker="o", lw=2.5, color=PALETTE["cobalt"], label="Collective Alignment $\\phi(\\eta)$")
    ax0.axhline(0.5, color=PALETTE["amber"], linestyle="--", lw=1.3, label="Half-Polarization Threshold")
    ax0.set_ylim(-0.02, 1.05)
    ax0.legend(loc="upper right", fontsize=8.5)

    _draw_physarum_panel(
        axes[1],
        coords,
        edges,
        res_cut["D"],
        res_cut["disrupted_mask"],
        f"Tier 3B: Live Severing & Non-Uniform Rerouting (Eff={e_cut/e_mst:.2f}x MST)",
    )

    fig.suptitle("Module 06 — Tier 3 (Advanced): Flocking Criticality & Live Hydraulic Self-Healing", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {
        "vicsek_phi_low_noise": float(phis[0]),
        "vicsek_phi_high_noise": float(phis[-1]),
        "physarum_cut_cost_ratio": c_cut / c_mst,
        "physarum_cut_eff_ratio": e_cut / e_mst,
        "physarum_cut_fault_tolerance": ft_cut,
    }


def generate_module06_gifs(out_dir: str | Path) -> list[Path]:
    """Generate classroom-ready animated GIFs for Module 06."""
    apply_publication_style(dpi=110)
    anim_dir = Path(out_dir) / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # GIF 1: Live Physarum Self-Organization + Dynamic Hub Edge Severing at midpoint
    coords, edges, adj_len = build_tokyo_benchmark_graph()
    res = solve_physarum_network(
        coords, edges, adj_len, mu=1.1, gamma=1.0, steps=140, disrupt_hub_edges=True, nonuniform_demand=True, seed=42, record_history=True
    )
    d_hist = res["d_history"]
    disrupted = res["disrupted_mask"]
    cut_step = res["cut_step"] or 70

    frames1 = []
    for step in range(0, len(d_hist), 5):
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
        fig.patch.set_facecolor(PALETTE["bg"])
        cur_cut = disrupted if step >= cut_step else np.zeros_like(disrupted)
        phase_lbl = "Rerouting & Self-Healing!" if step >= cut_step else "Pruning Redundant Mesh"
        _draw_physarum_panel(axes[0], coords, edges, d_hist[step], cur_cut, f"Step {step}/{len(d_hist)}: {phase_lbl}")

        ax1 = style_axes(axes[1], title="Active Backbone Fraction & Mean Flux", xlabel="Iteration", ylabel="Mean Flux $|Q_{ij}|$")
        ax1.plot(range(step + 1), res["mean_flux_history"][: step + 1], color=PALETTE["cobalt"], lw=2.2, label="Mean Hydraulic Flux")
        if step >= cut_step:
            ax1.axvline(cut_step, color=PALETTE["crimson"], linestyle="--", lw=1.6, label="Trunk Edges Severed!")
        ax1.set_xlim(0, len(d_hist))
        ax1.set_ylim(0, max(res["mean_flux_history"]) * 1.15)
        ax1.legend(loc="upper right", fontsize=8.2)
        fig.suptitle("Module 06 Live Physarum Tokyo Rail & Dynamic Repair", fontsize=12, fontweight="bold")
        fig.tight_layout()
        frames1.append(fig_to_pil_image(fig, dpi=105))
        plt.close(fig)
    generated.append(save_gif_from_frames(frames1, anim_dir / "01_physarum_tokyo_rail_self_healing.gif", duration_ms=130))

    # GIF 2: Vicsek Flocking Ordered (eta=0.40) vs Disordered (eta=4.40)
    v_ord = simulate_vicsek_flocking(n_birds=120, noise_eta=0.40, steps=80, seed=42, record_history=True)
    v_dis = simulate_vicsek_flocking(n_birds=120, noise_eta=4.40, steps=80, seed=42, record_history=True)
    frames2 = []
    for t in range(0, 80, 3):
        fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2))
        fig.patch.set_facecolor(PALETTE["bg"])
        for ax, vo, lbl, col in [
            (axes[0], v_ord, f"Ordered Flock ($\\eta=0.40, \\phi={v_ord['order_hist'][t]:.2f}$)", PALETTE["teal"]),
            (axes[1], v_dis, f"Disordered Gas ($\\eta=4.40, \\phi={v_dis['order_hist'][t]:.2f}$)", PALETTE["crimson"]),
        ]:
            p = vo["pos_hist"][t]
            a = vo["angle_hist"][t]
            ax.quiver(p[:, 0], p[:, 1], np.cos(a), np.sin(a), color=col, scale=24)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.set_aspect("equal")
            style_axes(ax, title=lbl, xlabel="$x$", ylabel="$y$", grid_axis=None)
        ax2 = style_axes(axes[2], title=f"Order Parameter $\\phi(t)$ ($t={t}$)", xlabel="Step $t$", ylabel="$\\phi(t)$")
        ax2.plot(range(t + 1), v_ord["order_hist"][: t + 1], color=PALETTE["teal"], lw=2.2, label="$\\eta=0.40$")
        ax2.plot(range(t + 1), v_dis["order_hist"][: t + 1], color=PALETTE["crimson"], lw=2.2, label="$\\eta=4.40$")
        ax2.set_xlim(0, 80)
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc="center right", fontsize=8.2)
        fig.suptitle("Module 06 Vicsek Collective Motion Phase Regimes", fontsize=12, fontweight="bold")
        fig.tight_layout()
        frames2.append(fig_to_pil_image(fig, dpi=105))
        plt.close(fig)
    generated.append(save_gif_from_frames(frames2, anim_dir / "02_vicsek_flocking_phase_transition.gif", duration_ms=115))
    return generated
