"""Module 02: Transfer Entropy, KSG Estimator, Wibral Delay Scan & Network Inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

from core.plotting_utils import (
    PALETTE,
    apply_publication_style,
    fig_to_pil_image,
    save_gif_from_frames,
    style_axes,
)

LN2: float = float(np.log(2.0))


def _standardize_with_jitter(arr: np.ndarray, noise_jitter: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean unit-variance standardization with infinitesimal jitter to break KSG ties."""
    arr_2d = np.asarray(arr, dtype=np.float64)
    if arr_2d.ndim == 1:
        arr_2d = arr_2d.reshape(-1, 1)
    mean = np.mean(arr_2d, axis=0)
    std = np.std(arr_2d, axis=0) + 1e-12
    z = (arr_2d - mean) / std
    if noise_jitter > 0.0:
        z = z + rng.normal(0.0, noise_jitter, size=z.shape)
    return z


def ksg_mi(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    noise_jitter: float = 1e-8,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate Mutual Information I(X; Y) in bits via KSG Algorithm 1 (Chebyshev norm)."""
    if rng is None:
        rng = np.random.default_rng(42)
    n_samples = len(x)
    x_s = _standardize_with_jitter(x, noise_jitter, rng)
    y_s = _standardize_with_jitter(y, noise_jitter, rng)
    xy = np.hstack([x_s, y_s])

    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x_s)
    tree_y = cKDTree(y_s)

    dists, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = np.maximum(dists[:, k] - 1e-15, 1e-15)
    nx_p1 = np.maximum(1, tree_x.query_ball_point(x_s, r=eps, p=np.inf, return_length=True))
    ny_p1 = np.maximum(1, tree_y.query_ball_point(y_s, r=eps, p=np.inf, return_length=True))

    mi_nats = digamma(k) + digamma(n_samples) - np.mean(digamma(nx_p1) + digamma(ny_p1))
    return max(0.0, float(mi_nats / LN2))


def ksg_conditional_mi(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None,
    k: int = 5,
    noise_jitter: float = 1e-8,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate Conditional MI I(X; Y | Z) in bits via Frenzel-Pompe / KSG estimator."""
    if z is None or np.asarray(z).size == 0:
        return ksg_mi(x, y, k=k, noise_jitter=noise_jitter, rng=rng)
    if rng is None:
        rng = np.random.default_rng(42)

    x_s = _standardize_with_jitter(x, noise_jitter, rng)
    y_s = _standardize_with_jitter(y, noise_jitter, rng)
    z_s = _standardize_with_jitter(z, noise_jitter, rng)

    xyz = np.hstack([x_s, y_s, z_s])
    xz = np.hstack([x_s, z_s])
    yz = np.hstack([y_s, z_s])

    tree_xyz = cKDTree(xyz)
    tree_xz = cKDTree(xz)
    tree_yz = cKDTree(yz)
    tree_z = cKDTree(z_s)

    dists, _ = tree_xyz.query(xyz, k=k + 1, p=np.inf)
    eps = np.maximum(dists[:, k] - 1e-15, 1e-15)

    nxz_p1 = np.maximum(1, tree_xz.query_ball_point(xz, r=eps, p=np.inf, return_length=True))
    nyz_p1 = np.maximum(1, tree_yz.query_ball_point(yz, r=eps, p=np.inf, return_length=True))
    nz_p1 = np.maximum(1, tree_z.query_ball_point(z_s, r=eps, p=np.inf, return_length=True))

    cmi_nats = digamma(k) - np.mean(digamma(nxz_p1) + digamma(nyz_p1) - digamma(nz_p1))
    return max(0.0, float(cmi_nats / LN2))


def _build_lagged_vectors(
    source: np.ndarray,
    target: np.ndarray,
    conditioning: np.ndarray | None = None,
    lag: int = 1,
    cond_history: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct aligned (source_past, target_future, conditioning_past) arrays for lag u."""
    u = max(int(lag), 1)
    hist_c = max(int(cond_history), 1) if conditioning is not None else 1
    start_idx = max(u, hist_c)
    n = len(target)
    if start_idx >= n - 5:
        raise ValueError(f"Time series length {n} is too short for lag {u}.")

    t_future = target[start_idx:]
    t_past = target[start_idx - 1 : n - 1]
    s_past = source[start_idx - u : n - u]

    if conditioning is None:
        return s_past, t_future, t_past.reshape(-1, 1)

    c_arr = np.asarray(conditioning, dtype=np.float64)
    if c_arr.ndim == 1:
        c_arr = c_arr.reshape(-1, 1)

    cond_cols = [t_past.reshape(-1, 1)]
    for h in range(1, hist_c + 1):
        cond_cols.append(c_arr[start_idx - h : n - h])
    if u > hist_c:
        cond_cols.append(c_arr[start_idx - u : n - u])
    return s_past, t_future, np.hstack(cond_cols)


def compute_transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    conditioning: np.ndarray | None = None,
    lag: int = 1,
    k: int = 5,
    cond_history: int = 2,
    rng: np.random.Generator | None = None,
) -> float:
    """Compute Bivariate or Conditional Transfer Entropy T_{source -> target | cond}(u) in bits."""
    s_past, t_future, cond_stack = _build_lagged_vectors(
        source=source,
        target=target,
        conditioning=conditioning,
        lag=lag,
        cond_history=cond_history,
    )
    return ksg_conditional_mi(s_past, t_future, cond_stack, k=k, rng=rng)


def compute_information_dynamics(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    k: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """Compute Active Information Storage A_X, Transfer Entropy T_{Y->X}, and Joint Predictive Info."""
    rng = np.random.default_rng(seed)
    s_past, t_future, t_past = _build_lagged_vectors(source, target, conditioning=None, lag=lag)
    ais = ksg_mi(t_past, t_future, k=k, rng=rng)
    te = ksg_conditional_mi(s_past, t_future, t_past, k=k, rng=rng)
    joint_past = np.hstack([t_past, s_past.reshape(-1, 1)])
    joint_pred = ksg_mi(joint_past, t_future, k=k, rng=rng)
    return {
        "AIS": ais,
        "TE": te,
        "AIS_plus_TE": ais + te,
        "Total_Predictive_MI": joint_pred,
        "Chain_Rule_Residual": abs((ais + te) - joint_pred),
    }


def generate_synthetic_motif(
    motif_type: str = "Common_Driver",
    n_steps: int = 600,
    coupling: float = 0.65,
    nonlinear: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate 3-node canonical causal motifs (with optional hidden confounder U)."""
    rng = np.random.default_rng(seed)
    burn_in = 100
    total_n = n_steps + burn_in
    x, y, z, u_hid = [np.zeros(total_n, dtype=np.float64) for _ in range(4)]

    noise_x = rng.normal(0.0, 0.45, total_n)
    noise_y = rng.normal(0.0, 0.45, total_n)
    noise_z = rng.normal(0.0, 0.45, total_n)
    noise_u = rng.normal(0.0, 0.60, total_n)

    f = (lambda val: np.tanh(1.8 * val)) if nonlinear else (lambda val: val)
    for t in range(3, total_n):
        if motif_type == "Direct_Coupling":
            x[t] = 0.5 * x[t - 1] + noise_x[t]
            y[t] = 0.4 * y[t - 1] + coupling * f(x[t - 2]) + noise_y[t]
            z[t] = 0.5 * z[t - 1] + noise_z[t]
        elif motif_type == "Indirect_Cascade":
            x[t] = 0.5 * x[t - 1] + noise_x[t]
            z[t] = 0.4 * z[t - 1] + coupling * f(x[t - 1]) + noise_z[t]
            y[t] = 0.4 * y[t - 1] + coupling * f(z[t - 1]) + noise_y[t]
        elif motif_type == "Common_Driver":
            z[t] = 0.6 * z[t - 1] + noise_z[t]
            x[t] = 0.3 * x[t - 1] + coupling * f(z[t - 1]) + noise_x[t]
            y[t] = 0.3 * y[t - 1] + coupling * f(z[t - 2]) + noise_y[t]
        elif motif_type == "Unobserved_Confounder":
            u_hid[t] = 0.6 * u_hid[t - 1] + noise_u[t]
            x[t] = 0.3 * x[t - 1] + coupling * f(u_hid[t - 1]) + noise_x[t]
            y[t] = 0.3 * y[t - 1] + coupling * f(u_hid[t - 2]) + noise_y[t]
            z[t] = 0.5 * z[t - 1] + noise_z[t]
        else:
            raise ValueError(f"Unknown motif_type '{motif_type}'.")

    return x[burn_in:], y[burn_in:], z[burn_in:]


def surrogate_significance_test(
    source: np.ndarray,
    target: np.ndarray,
    conditioning: np.ndarray | None = None,
    lag: int = 1,
    n_surrogates: int = 80,
    k: int = 5,
    cond_history: int = 2,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Run circular time-shift surrogate significance test for Transfer Entropy."""
    rng = np.random.default_rng(seed)
    s_emp, t_future, cond_stack = _build_lagged_vectors(
        source=source,
        target=target,
        conditioning=conditioning,
        lag=lag,
        cond_history=cond_history,
    )
    y_s = _standardize_with_jitter(t_future, 1e-8, rng)
    z_s = _standardize_with_jitter(cond_stack, 1e-8, rng)
    yz = np.hstack([y_s, z_s])
    tree_yz = cKDTree(yz)
    tree_z = cKDTree(z_s)
    dg_k = float(digamma(k))

    def _eval_with_cached_yz(s_vec: np.ndarray) -> float:
        x_s = _standardize_with_jitter(s_vec, 1e-8, rng)
        xyz = np.hstack([x_s, yz])
        xz = np.hstack([x_s, z_s])
        dists, _ = cKDTree(xyz).query(xyz, k=k + 1, p=np.inf)
        eps = np.maximum(dists[:, k] - 1e-15, 1e-15)
        nxz_p1 = np.maximum(1, cKDTree(xz).query_ball_point(xz, r=eps, p=np.inf, return_length=True))
        nyz_p1 = np.maximum(1, tree_yz.query_ball_point(yz, r=eps, p=np.inf, return_length=True))
        nz_p1 = np.maximum(1, tree_z.query_ball_point(z_s, r=eps, p=np.inf, return_length=True))
        cmi_nats = dg_k - np.mean(digamma(nxz_p1) + digamma(nyz_p1) - digamma(nz_p1))
        return max(0.0, float(cmi_nats / LN2))

    te_emp = _eval_with_cached_yz(s_emp)
    n = len(source)
    surr_vals = np.zeros(n_surrogates, dtype=np.float64)
    low = max(5, int(0.15 * n))
    high = max(low + 1, int(0.85 * n))
    u = max(int(lag), 1)
    hist_c = max(int(cond_history), 1) if conditioning is not None else 1
    start_idx = max(u, hist_c)
    for i in range(n_surrogates):
        shift = int(rng.integers(low=low, high=high))
        rolled = np.roll(source, shift)
        surr_vals[i] = _eval_with_cached_yz(rolled[start_idx - u : n - u])
    p_val = (1.0 + float(np.sum(surr_vals >= te_emp))) / (1.0 + float(n_surrogates))
    cutoff_95 = float(np.percentile(surr_vals, 95))
    return float(te_emp), cutoff_95, float(p_val)


def benjamini_hochberg_fdr(p_values: np.ndarray | list[float], alpha: float = 0.05) -> dict[str, Any]:
    """Apply Benjamini-Hochberg FDR multiple-testing correction across M candidate edges."""
    p_arr = np.asarray(p_values, dtype=np.float64)
    m = len(p_arr)
    order = np.argsort(p_arr)
    sorted_p = p_arr[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    crit_line = (ranks / float(m)) * float(alpha)

    below = np.where(sorted_p <= crit_line)[0]
    rejected = np.zeros(m, dtype=bool)
    if len(below) > 0:
        max_k = int(below[-1])
        rejected[order[: max_k + 1]] = True

    # Step-up BH adjusted q-values
    q_sorted = np.minimum.accumulate((sorted_p * m / ranks)[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q_vals = np.empty(m, dtype=np.float64)
    q_vals[order] = q_sorted

    return {
        "rejected": rejected,
        "q_values": q_vals,
        "order": order,
        "sorted_p": sorted_p,
        "crit_line": crit_line,
        "ranks": ranks,
    }


def infer_directed_network(
    signals: dict[str, np.ndarray],
    lags: list[int] | tuple[int, ...] = (1, 2, 3, 4, 5),
    k: int = 5,
    n_surrogates: int = 60,
    alpha: float = 0.05,
    cond_history: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Infer a full directed network across all M = N(N-1) candidate pairs with BH-FDR."""
    names = list(signals.keys())
    n_nodes = len(names)
    biv_mat = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    cond_mat = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    lag_mat = np.zeros((n_nodes, n_nodes), dtype=int)
    biv_p_mat = np.ones((n_nodes, n_nodes), dtype=np.float64)
    cond_p_mat = np.ones((n_nodes, n_nodes), dtype=np.float64)

    edge_records: list[dict[str, Any]] = []
    rng_scan = np.random.default_rng(seed)
    m_edges = max(1, n_nodes * (n_nodes - 1))
    # Guarantee enough surrogate resolution to pass rank-1 BH-FDR critical line (alpha / M)
    n_surr_high = max(n_surrogates, int(np.ceil(m_edges / max(alpha, 1e-6))) + 25)

    for i, src_name in enumerate(names):
        for j, tgt_name in enumerate(names):
            if i == j:
                continue
            src = signals[src_name]
            tgt = signals[tgt_name]
            other_cols = [signals[names[m]] for m in range(n_nodes) if m not in (i, j)]
            cond_arr = np.column_stack(other_cols) if other_cols else None

            biv_by_lag = [
                compute_transfer_entropy(src, tgt, conditioning=None, lag=u, k=k, rng=rng_scan)
                for u in lags
            ]
            best_idx = int(np.argmax(biv_by_lag))
            best_u = int(lags[best_idx])
            n_surr_b = n_surr_high if biv_by_lag[best_idx] > 0.03 else n_surrogates
            te_biv, thr_biv, p_biv = surrogate_significance_test(
                src, tgt, conditioning=None, lag=best_u, n_surrogates=n_surr_b, k=k, seed=seed + i * 17 + j
            )
            te_cond_est = compute_transfer_entropy(
                src, tgt, conditioning=cond_arr, lag=best_u, k=k, cond_history=cond_history, rng=rng_scan
            )
            n_surr_c = n_surr_high if te_cond_est > 0.03 else n_surrogates
            te_cond, thr_cond, p_cond = surrogate_significance_test(
                src, tgt, conditioning=cond_arr, lag=best_u, n_surrogates=n_surr_c, k=k, cond_history=cond_history, seed=seed + 100 + i * 17 + j
            )

            biv_mat[i, j] = te_biv
            cond_mat[i, j] = te_cond
            lag_mat[i, j] = best_u
            biv_p_mat[i, j] = p_biv
            cond_p_mat[i, j] = p_cond
            edge_records.append(
                {
                    "source": src_name,
                    "target": tgt_name,
                    "i": i,
                    "j": j,
                    "best_lag": best_u,
                    "te_bivariate": te_biv,
                    "thr_bivariate": thr_biv,
                    "p_bivariate": p_biv,
                    "te_conditional": te_cond,
                    "thr_conditional": thr_cond,
                    "p_conditional": p_cond,
                }
            )

    biv_fdr = benjamini_hochberg_fdr([r["p_bivariate"] for r in edge_records], alpha=alpha)
    cond_fdr = benjamini_hochberg_fdr([r["p_conditional"] for r in edge_records], alpha=alpha)

    biv_mask = np.zeros((n_nodes, n_nodes), dtype=bool)
    cond_mask = np.zeros((n_nodes, n_nodes), dtype=bool)
    for idx, rec in enumerate(edge_records):
        rec["biv_significant_fdr"] = bool(biv_fdr["rejected"][idx])
        rec["biv_q_value"] = float(biv_fdr["q_values"][idx])
        rec["cond_significant_fdr"] = bool(cond_fdr["rejected"][idx])
        rec["cond_q_value"] = float(cond_fdr["q_values"][idx])
        biv_mask[rec["i"], rec["j"]] = rec["biv_significant_fdr"]
        cond_mask[rec["i"], rec["j"]] = rec["cond_significant_fdr"]

    return {
        "nodes": names,
        "bivariate_te": biv_mat,
        "conditional_te": cond_mat,
        "best_lags": lag_mat,
        "bivariate_p": biv_p_mat,
        "conditional_p": cond_p_mat,
        "biv_fdr_mask": biv_mask,
        "cond_fdr_mask": cond_mask,
        "biv_fdr": biv_fdr,
        "cond_fdr": cond_fdr,
        "edges": edge_records,
    }


def _motif_ground_truth_edges(motif_type: str) -> list[tuple[str, str]]:
    if motif_type == "Direct_Coupling":
        return [("X", "Y")]
    if motif_type == "Indirect_Cascade":
        return [("X", "Z"), ("Z", "Y")]
    if motif_type == "Common_Driver":
        return [("Z", "X"), ("Z", "Y")]
    if motif_type == "Unobserved_Confounder":
        return [("U (hidden)", "X"), ("U (hidden)", "Y")]
    return []


def _draw_motif_recovery_graph(
    ax: plt.Axes,
    motif_type: str,
    net_res: dict[str, Any],
) -> None:
    """Visualize ground truth vs spurious bivariate vs conditional BH-FDR recovered edges."""
    ax.set_facecolor(PALETTE["bg"])
    ax.axis("off")

    pos = {"X": (-1.0, -0.45), "Y": (1.0, -0.45), "Z": (0.0, 0.72), "U (hidden)": (0.0, 1.32)}
    nodes = ["X", "Y", "Z"]
    if motif_type == "Unobserved_Confounder":
        nodes.append("U (hidden)")

    g_viz = nx.DiGraph()
    g_viz.add_nodes_from(nodes)

    # Draw node circles cleanly
    for node in nodes:
        x_p, y_p = pos[node]
        is_hidden = "hidden" in node
        fc = "#FEF2F2" if is_hidden else "#EFF6FF"
        ec = PALETTE["crimson"] if is_hidden else PALETTE["cobalt"]
        ls = "--" if is_hidden else "-"
        circ = plt.Circle((x_p, y_p), 0.25, facecolor=fc, edgecolor=ec, lw=2.0, linestyle=ls, zorder=4)
        ax.add_patch(circ)
        ax.text(x_p, y_p, node if not is_hidden else "U\n(Hidden)", ha="center", va="center", fontsize=10.5 if not is_hidden else 8.8, fontweight="bold", color=PALETTE["ink"], zorder=5)

    # True edges in subtle gray background arrows if hidden U
    if motif_type == "Unobserved_Confounder":
        for u_src, v_tgt in _motif_ground_truth_edges(motif_type):
            ax.annotate(
                "",
                xy=pos[v_tgt],
                xytext=pos[u_src],
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.5", color=PALETTE["muted"], lw=1.6, ls="--", shrinkA=22, shrinkB=22),
                zorder=2,
            )

    # Spurious bivariate-only edges (rejected by Conditional TE + BH-FDR) in dashed crimson
    for rec in net_res["edges"]:
        src, tgt = rec["source"], rec["target"]
        biv_sig = rec["biv_significant_fdr"]
        cond_sig = rec["cond_significant_fdr"]
        if cond_sig:
            ax.annotate(
                "",
                xy=pos[tgt],
                xytext=pos[src],
                arrowprops=dict(
                    arrowstyle="-|>,head_width=0.4,head_length=0.6",
                    color=PALETTE["teal"],
                    lw=2.6,
                    connectionstyle="arc3,rad=0.14",
                    shrinkA=23,
                    shrinkB=23,
                ),
                zorder=3,
            )
            mx = 0.5 * (pos[src][0] + pos[tgt][0])
            my = 0.5 * (pos[src][1] + pos[tgt][1]) + (0.14 if src == "X" and tgt == "Y" else 0.0)
            ax.text(
                mx,
                my,
                f"T={rec['te_conditional']:.2f}b (u={rec['best_lag']})",
                fontsize=8.2,
                fontweight="bold",
                color=PALETTE["teal"],
                ha="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=PALETTE["teal"], alpha=0.9),
                zorder=6,
            )
        elif biv_sig and not cond_sig:
            ax.annotate(
                "",
                xy=pos[tgt],
                xytext=pos[src],
                arrowprops=dict(
                    arrowstyle="-|>,head_width=0.35,head_length=0.5",
                    color=PALETTE["crimson"],
                    lw=1.9,
                    ls=":",
                    connectionstyle="arc3,rad=-0.16",
                    shrinkA=23,
                    shrinkB=23,
                ),
                zorder=2,
            )

    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-0.95, 1.72 if motif_type == "Unobserved_Confounder" else 1.15)
    ax.set_title("1. Graph Recovery (Teal=Conditional FDR, Dotted Red=Filtered Spurious)", fontsize=10.8, fontweight="bold")


def create_te_dashboard_figure(
    motif_type: str = "Common_Driver",
    coupling: float = 0.70,
    n_samples: int = 450,
    k_neighbors: int = 5,
    n_surrogates: int = 60,
    seed: int = 42,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Build a 4-panel Transfer Entropy & Network Inference dashboard for Module 02."""
    apply_publication_style()
    x, y, z = generate_synthetic_motif(motif_type, n_steps=n_samples, coupling=coupling, seed=seed)
    rng = np.random.default_rng(seed)
    lags = [1, 2, 3, 4, 5]

    biv_scan = [compute_transfer_entropy(x, y, conditioning=None, lag=u, k=k_neighbors, rng=rng) for u in lags]
    cond_scan = [compute_transfer_entropy(x, y, conditioning=z, lag=u, k=k_neighbors, rng=rng) for u in lags]
    best_lag = int(lags[int(np.argmax(biv_scan))])

    te_biv, thr_biv, p_biv = surrogate_significance_test(
        x, y, conditioning=None, lag=best_lag, n_surrogates=n_surrogates, k=k_neighbors, seed=seed
    )
    te_cond, thr_cond, p_cond = surrogate_significance_test(
        x, y, conditioning=z, lag=best_lag, n_surrogates=n_surrogates, k=k_neighbors, seed=seed
    )
    info_dyn = compute_information_dynamics(x, y, lag=best_lag, k=k_neighbors, seed=seed)
    net_res = infer_directed_network(
        {"X": x, "Y": y, "Z": z},
        lags=lags,
        k=k_neighbors,
        n_surrogates=n_surrogates,
        alpha=0.05,
        seed=seed,
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.2))
    fig.patch.set_facecolor(PALETTE["bg"])

    # Panel 1: Directed Graph Recovery
    _draw_motif_recovery_graph(axes[0, 0], motif_type, net_res)

    # Panel 2: Wibral Interaction Delay Curve T(u)
    ax2 = style_axes(
        axes[0, 1],
        title=f"2. Wibral Delay Scan $T_{{X \\to Y}}(u)$ ({motif_type})",
        xlabel="Candidate Interaction Delay $u$ (steps)",
        ylabel="Transfer Entropy (Bits)",
    )
    ax2.plot(lags, biv_scan, marker="o", lw=2.4, color=PALETTE["crimson"], label="Bivariate $T(X \\to Y)$")
    ax2.plot(lags, cond_scan, marker="s", lw=2.4, color=PALETTE["teal"], label="Conditional $T(X \\to Y \\mid Z)$")
    ax2.axhline(thr_cond, color=PALETTE["slate"], linestyle="--", lw=1.4, label=f"95% Surrogate Null ({thr_cond:.3f} b)")
    ax2.set_xticks(lags)
    ax2.legend(loc="best", fontsize=8.8)

    # Panel 3: Bivariate vs Conditional TE Grouped Bar Chart + Information Dynamics Chain Rule
    ax3 = style_axes(
        axes[1, 0],
        title=f"3. Confounder Filtering & Chain Rule (Lag $u={best_lag}$)",
        ylabel="Information / TE (Bits)",
        grid_axis="y",
    )
    labels = [f"Bivariate TE\n($p={p_biv:.3f}$)", f"Conditional TE | Z\n($p={p_cond:.3f}$)"]
    xp = np.arange(2)
    b_est = ax3.bar(xp - 0.16, [te_biv, te_cond], width=0.30, color=[PALETTE["crimson"], PALETTE["cobalt"]], edgecolor=PALETTE["ink"], label="Estimated TE")
    b_thr = ax3.bar(xp + 0.16, [thr_biv, thr_cond], width=0.30, color=PALETTE["grid"], edgecolor=PALETTE["ink"], label="95% Surrogate Cutoff")
    ax3.set_xticks(xp)
    ax3.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
    ymax3 = max(0.25, max(te_biv, te_cond, thr_biv, thr_cond) * 1.35)
    ax3.set_ylim(0, ymax3)
    for b in list(b_est) + list(b_thr):
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width() / 2, h + 0.025 * ymax3, f"{h:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    ax3.text(
        0.97,
        0.95,
        f"AIS $A_Y = {info_dyn['AIS']:.3f}$ b\n$A_Y + T_{{X\\to Y}} = {info_dyn['AIS_plus_TE']:.3f}$ b",
        transform=ax3.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor=PALETTE["card"], edgecolor=PALETTE["cobalt"]),
    )
    ax3.legend(loc="upper left", fontsize=8.5)

    # Panel 4: Benjamini-Hochberg FDR Rank Plot across all M = 6 candidate edges
    ax4 = style_axes(
        axes[1, 1],
        title="4. Benjamini-Hochberg FDR Control ($M=6$ Directed Edges)",
        xlabel="Sorted Edge Rank $k$",
        ylabel="Surrogate $p$-value",
    )
    fdr = net_res["cond_fdr"]
    ranks = fdr["ranks"]
    ax4.plot(ranks, fdr["crit_line"], color=PALETTE["amber"], linestyle="--", lw=2.0, label="BH Critical $(k/M)\\alpha$ ($\\alpha=0.05$)")
    for idx_r, r_val in enumerate(ranks):
        orig_idx = int(fdr["order"][idx_r])
        rec = net_res["edges"][orig_idx]
        col = PALETTE["teal"] if rec["cond_significant_fdr"] else PALETTE["crimson"]
        ax4.scatter([r_val], [fdr["sorted_p"][idx_r]], color=col, s=65, zorder=5, edgecolor=PALETTE["ink"])
        ax4.annotate(
            f"{rec['source']}→{rec['target']}",
            (r_val, fdr["sorted_p"][idx_r]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8.0,
            fontweight="bold",
            color=col,
        )
    ax4.set_xticks(ranks)
    ax4.set_ylim(-0.04, 1.08)
    ax4.legend(loc="upper left", fontsize=8.6)

    fig.suptitle(
        f"Module 02 Dashboard — {motif_type} | N={n_samples}, c={coupling:.2f}, k-NN={k_neighbors}, N_surr={n_surrogates}",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, {
        "motif": motif_type,
        "best_lag": best_lag,
        "te_bivariate": te_biv,
        "thr_bivariate": thr_biv,
        "p_bivariate": p_biv,
        "te_conditional": te_cond,
        "thr_conditional": thr_cond,
        "p_conditional": p_cond,
        "info_dynamics": info_dyn,
        "network": net_res,
    }


def run_module02_tier1(
    coupling: float = 0.70,
    n_samples: int = 500,
    k_neighbors: int = 5,
    n_surrogates: int = 80,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 1: Common_Driver & Indirect_Cascade filter spurious bivariate TE below 95% cutoff."""
    apply_publication_style()
    results = {}
    for motif in ("Common_Driver", "Indirect_Cascade"):
        x, y, z = generate_synthetic_motif(motif, n_steps=n_samples, coupling=coupling, seed=42)
        rng = np.random.default_rng(42)
        lags = [1, 2, 3, 4, 5]
        biv_scan = [compute_transfer_entropy(x, y, conditioning=None, lag=u, k=k_neighbors, rng=rng) for u in lags]
        best_lag = int(lags[int(np.argmax(biv_scan))])
        te_b, thr_b, p_b = surrogate_significance_test(x, y, None, lag=best_lag, n_surrogates=n_surrogates, k=k_neighbors, seed=42)
        te_c, thr_c, p_c = surrogate_significance_test(x, y, z, lag=best_lag, n_surrogates=n_surrogates, k=k_neighbors, seed=42)
        results[motif] = {
            "best_lag": best_lag,
            "te_biv": te_b,
            "thr_biv": thr_b,
            "p_biv": p_b,
            "te_cond": te_c,
            "thr_cond": thr_c,
            "p_cond": p_c,
        }

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    fig.patch.set_facecolor(PALETTE["bg"])

    for ax, motif, desc in [
        (axes[0], "Common_Driver", "Common Driver ($Z \\to X, Z \\to Y$)"),
        (axes[1], "Indirect_Cascade", "Indirect Cascade ($X \\to Z \\to Y$)"),
    ]:
        r = results[motif]
        style_axes(ax, title=f"Tier 1: {desc}", ylabel="Transfer Entropy (Bits)", grid_axis="y")
        xp = np.arange(2)
        b1 = ax.bar(xp - 0.16, [r["te_biv"], r["te_cond"]], width=0.30, color=[PALETTE["crimson"], PALETTE["cobalt"]], edgecolor=PALETTE["ink"], label="Estimated TE")
        b2 = ax.bar(xp + 0.16, [r["thr_biv"], r["thr_cond"]], width=0.30, color=PALETTE["grid"], edgecolor=PALETTE["ink"], label="95% Surrogate Cutoff")
        ax.set_xticks(xp)
        ax.set_xticklabels([f"Bivariate\n(p={r['p_biv']:.3f})", f"Conditional | Z\n(p={r['p_cond']:.3f})"], fontweight="bold")
        ymax = max(r["te_biv"], r["thr_biv"], r["thr_cond"]) * 1.38
        ax.set_ylim(0, ymax)
        for b in list(b1) + list(b2):
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.025 * ymax, f"{h:.3f}", ha="center", fontsize=8.8, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8.5)

    fig.suptitle("Module 02 — Tier 1 (Observation): Conditioning Eliminates Spurious Bivariate Links", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, results


def run_module02_tier2(
    coupling: float = 0.70,
    n_samples: int = 550,
    k_neighbors: int = 5,
    n_surrogates: int = 80,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 2: Wibral et al. (2013) delay scan u=1..5 on Direct_Coupling (true tau=2)."""
    apply_publication_style()
    x, y, z = generate_synthetic_motif("Direct_Coupling", n_steps=n_samples, coupling=coupling, seed=42)
    lags = [1, 2, 3, 4, 5]
    biv_vals, biv_thrs, cond_vals, cond_thrs = [], [], [], []
    for u in lags:
        tb, thb, _ = surrogate_significance_test(x, y, None, lag=u, n_surrogates=n_surrogates, k=k_neighbors, seed=42 + u)
        tc, thc, _ = surrogate_significance_test(x, y, z, lag=u, n_surrogates=n_surrogates, k=k_neighbors, seed=142 + u)
        biv_vals.append(tb)
        biv_thrs.append(thb)
        cond_vals.append(tc)
        cond_thrs.append(thc)

    peak_lag = int(lags[int(np.argmax(cond_vals))])
    info_dyn = compute_information_dynamics(x, y, lag=peak_lag, k=k_neighbors, seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax0 = style_axes(axes[0], title="Tier 2A: Wibral Interaction Delay Reconstruction ($\\tau=2$)", xlabel="Candidate Delay $u$ (steps)", ylabel="Transfer Entropy (Bits)")
    ax0.plot(lags, biv_vals, marker="o", lw=2.4, color=PALETTE["crimson"], label="Bivariate $T(X \\to Y)$")
    ax0.plot(lags, cond_vals, marker="s", lw=2.4, color=PALETTE["teal"], label="Conditional $T(X \\to Y \\mid Z)$")
    ax0.plot(lags, cond_thrs, linestyle="--", lw=1.6, color=PALETTE["slate"], label="95% Surrogate Cutoff")
    ax0.scatter([peak_lag], [cond_vals[peak_lag - 1]], color=PALETTE["amber"], s=110, zorder=6, label=f"Recovered Peak $u^* = {peak_lag}$")
    ax0.set_xticks(lags)
    ax0.legend(loc="best", fontsize=8.6)

    ax1 = style_axes(axes[1], title="Tier 2B: Information Dynamics Chain Rule Decomposition", ylabel="Information (Bits)", grid_axis="y")
    names = ["Active Storage\n$A_Y = I(Y_{\\mathrm{past}}; Y_t)$", "Transfer Entropy\n$T_{X \\to Y}(u^*)$", "$A_Y + T_{X \\to Y}$", "Total Predictive\n$I(X_{\\mathrm{past}}, Y_{\\mathrm{past}}; Y_t)$"]
    vals = [info_dyn["AIS"], info_dyn["TE"], info_dyn["AIS_plus_TE"], info_dyn["Total_Predictive_MI"]]
    cols = [PALETTE["cobalt"], PALETTE["teal"], PALETTE["amber"], PALETTE["slate"]]
    bars = ax1.bar(names, vals, color=cols, width=0.52, edgecolor=PALETTE["ink"])
    ymax = max(vals) * 1.28
    ax1.set_ylim(0, ymax)
    for b in bars:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 0.025 * ymax, f"{h:.3f} b", ha="center", fontsize=8.8, fontweight="bold")

    fig.suptitle("Module 02 — Tier 2 (Analysis): Delay Reconstruction & Information Dynamics", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {"lags": lags, "biv_scan": biv_vals, "cond_scan": cond_vals, "peak_lag": peak_lag, "info_dynamics": info_dyn}


def run_module02_tier3(
    coupling: float = 0.70,
    k_neighbors: int = 5,
    n_surrogates: int = 80,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 3: Unobserved_Confounder sample-size sweep demonstrating irreducible false positives."""
    apply_publication_style()
    sample_sizes = [250, 350, 450, 600, 800]
    te_cond_list, thr_list, p_list = [], [], []
    for n in sample_sizes:
        x, y, z = generate_synthetic_motif("Unobserved_Confounder", n_steps=n, coupling=coupling, seed=42)
        te_c, thr_c, p_c = surrogate_significance_test(x, y, z, lag=1, n_surrogates=n_surrogates, k=k_neighbors, seed=42)
        te_cond_list.append(te_c)
        thr_list.append(thr_c)
        p_list.append(p_c)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax0 = style_axes(axes[0], title="Tier 3A: Persistent Spurious TE Under Hidden Confounder $U$", xlabel="Sample Size $N$", ylabel="Conditional TE $T(X \\to Y \\mid Z)$ (Bits)")
    ax0.plot(sample_sizes, te_cond_list, marker="o", lw=2.5, color=PALETTE["crimson"], label="Conditional TE $|Z$ (False Positive!)")
    ax0.plot(sample_sizes, thr_list, marker="s", lw=2.0, linestyle="--", color=PALETTE["slate"], label="95% Surrogate Null Threshold")
    ax0.fill_between(sample_sizes, thr_list, te_cond_list, where=np.array(te_cond_list) > np.array(thr_list), color=PALETTE["crimson"], alpha=0.14)
    ax0.legend(loc="best", fontsize=8.8)

    ax1 = style_axes(axes[1], title="Tier 3B: Surrogate $p$-value Stays Below $\\alpha=0.05$", xlabel="Sample Size $N$", ylabel="Surrogate $p$-value")
    ax1.plot(sample_sizes, p_list, marker="o", lw=2.5, color=PALETTE["amber"], label="False-Positive $p$-value")
    ax1.axhline(0.05, color=PALETTE["crimson"], linestyle="--", lw=1.8, label="$\\alpha = 0.05$ Significance Threshold")
    ax1.set_ylim(0.0, max(0.15, max(p_list) * 1.3))
    ax1.legend(loc="upper right", fontsize=8.8)

    fig.suptitle("Module 02 — Tier 3 (Stress Test): Irreducible Observational Limit of Unobserved Confounders", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {"sample_sizes": sample_sizes, "te_conditional": te_cond_list, "surrogate_95": thr_list, "p_values": p_list}


def generate_module02_gifs(out_dir: str | Path) -> list[Path]:
    """Generate classroom-ready animated GIFs for Module 02."""
    apply_publication_style(dpi=115)
    anim_dir = Path(out_dir) / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # GIF 1: Wibral Delay Scan u = 1..5 comparing Direct_Coupling (tau=2) vs Common_Driver
    x_d, y_d, z_d = generate_synthetic_motif("Direct_Coupling", n_steps=500, coupling=0.75, seed=42)
    x_c, y_c, z_c = generate_synthetic_motif("Common_Driver", n_steps=500, coupling=0.75, seed=42)
    lags = [1, 2, 3, 4, 5]
    rng = np.random.default_rng(42)
    dc_biv = [compute_transfer_entropy(x_d, y_d, None, lag=u, k=5, rng=rng) for u in lags]
    dc_cond = [compute_transfer_entropy(x_d, y_d, z_d, lag=u, k=5, rng=rng) for u in lags]
    cd_biv = [compute_transfer_entropy(x_c, y_c, None, lag=u, k=5, rng=rng) for u in lags]
    cd_cond = [compute_transfer_entropy(x_c, y_c, z_c, lag=u, k=5, rng=rng) for u in lags]
    _, thr_d, _ = surrogate_significance_test(x_d, y_d, z_d, lag=2, n_surrogates=50, k=5, seed=42)
    _, thr_c, _ = surrogate_significance_test(x_c, y_c, z_c, lag=1, n_surrogates=50, k=5, seed=42)

    frames = []
    for idx in range(len(lags)):
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
        fig.patch.set_facecolor(PALETTE["bg"])

        ax0 = style_axes(axes[0], title=f"Direct Coupling ($X \\to Y, \\tau=2$) | Scan u=1..{lags[idx]}", xlabel="Delay $u$", ylabel="TE (Bits)")
        ax0.plot(lags[: idx + 1], dc_biv[: idx + 1], marker="o", lw=2.3, color=PALETTE["crimson"], label="Bivariate")
        ax0.plot(lags[: idx + 1], dc_cond[: idx + 1], marker="s", lw=2.3, color=PALETTE["teal"], label="Conditional | Z")
        ax0.axhline(thr_d, color=PALETTE["slate"], linestyle="--", lw=1.3, label="95% Null")
        ax0.set_xlim(0.7, 5.3)
        ax0.set_ylim(0, max(dc_biv + dc_cond) * 1.25)
        ax0.set_xticks(lags)
        ax0.legend(loc="upper right", fontsize=8.4)

        ax1 = style_axes(axes[1], title=f"Common Driver ($Z \\to X, Y$) | Scan u=1..{lags[idx]}", xlabel="Delay $u$", ylabel="TE (Bits)")
        ax1.plot(lags[: idx + 1], cd_biv[: idx + 1], marker="o", lw=2.3, color=PALETTE["crimson"], label="Bivariate (Spurious)")
        ax1.plot(lags[: idx + 1], cd_cond[: idx + 1], marker="s", lw=2.3, color=PALETTE["teal"], label="Conditional | Z (Filtered)")
        ax1.axhline(thr_c, color=PALETTE["slate"], linestyle="--", lw=1.3, label="95% Null")
        ax1.set_xlim(0.7, 5.3)
        ax1.set_ylim(0, max(cd_biv + [thr_c]) * 1.35)
        ax1.set_xticks(lags)
        ax1.legend(loc="upper right", fontsize=8.4)

        fig.suptitle("Module 02 Progressive Wibral Delay Reconstruction & Confounder Filtering", fontsize=12, fontweight="bold")
        fig.tight_layout()
        frames.append(fig_to_pil_image(fig, dpi=110))
        plt.close(fig)

    generated.append(save_gif_from_frames(frames, anim_dir / "01_wibral_delay_scan_and_filtering.gif", duration_ms=520, hold_last_frames=4))
    return generated
