"""Module 01: Information Theory, Nonlinearity, Landauer Bound & Williams-Beer PID."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
from scipy.spatial import cKDTree
from scipy.special import digamma, erfc
from scipy.stats import pearsonr

from core.plotting_utils import (
    PALETTE,
    apply_publication_style,
    fig_to_pil_image,
    save_gif_from_frames,
    style_axes,
)

# Exact SI 2019 Boltzmann constant [J / K]
K_BOLTZMANN: float = 1.380649e-23
LN2: float = float(np.log(2.0))


def self_information(prob: np.ndarray | float) -> np.ndarray | float:
    """Compute self-information (surprise) h(x) = -log2 P(x) in bits."""
    arr = np.asarray(prob, dtype=np.float64)
    safe = np.clip(arr, 1e-15, 1.0)
    res = -np.log2(safe)
    return float(res) if np.ndim(prob) == 0 else res


def binary_entropy(p: np.ndarray | float) -> np.ndarray | float:
    """Compute binary Shannon entropy H_b(p) = -p log2 p - (1-p) log2(1-p) in bits."""
    arr = np.asarray(p, dtype=np.float64)
    arr_clip = np.clip(arr, 1e-15, 1.0 - 1e-15)
    h = -(arr_clip * np.log2(arr_clip) + (1.0 - arr_clip) * np.log2(1.0 - arr_clip))
    h = np.where((arr <= 1e-15) | (arr >= 1.0 - 1e-15), 0.0, h)
    return float(h) if np.ndim(p) == 0 else h


def shannon_entropy(p: np.ndarray) -> float:
    """Compute Shannon entropy H(X) = -sum P(x) log2 P(x) in bits."""
    p_arr = np.asarray(p, dtype=np.float64).ravel()
    total = float(np.sum(p_arr))
    if total <= 0.0:
        return 0.0
    p_norm = p_arr / total
    mask = p_norm > 1e-15
    return float(-np.sum(p_norm[mask] * np.log2(p_norm[mask])))


def joint_entropy(p_joint: np.ndarray) -> float:
    """Compute joint Shannon entropy H(X, Y, ...) in bits."""
    return shannon_entropy(p_joint)


def conditional_entropy(p_xy: np.ndarray, given_axis: int = 1) -> float:
    """Compute conditional entropy H(X | Y) = H(X, Y) - H(Y) in bits."""
    p_arr = np.asarray(p_xy, dtype=np.float64)
    p_arr = p_arr / np.sum(p_arr)
    sum_axes = tuple(i for i in range(p_arr.ndim) if i != given_axis)
    p_given = np.sum(p_arr, axis=sum_axes)
    return max(0.0, joint_entropy(p_arr) - shannon_entropy(p_given))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Kullback-Leibler divergence D_KL(P || Q) in bits."""
    p_arr = np.asarray(p, dtype=np.float64).ravel()
    q_arr = np.asarray(q, dtype=np.float64).ravel()
    p_arr = p_arr / np.sum(p_arr)
    q_arr = q_arr / np.sum(q_arr)
    mask = p_arr > 1e-15
    if np.any(q_arr[mask] <= 1e-15):
        return float("inf")
    return float(np.sum(p_arr[mask] * np.log2(p_arr[mask] / q_arr[mask])))


def mutual_information(p_xy: np.ndarray) -> dict[str, float]:
    """Compute I(X; Y) via entropy subtraction and KL divergence identity checks."""
    p_arr = np.asarray(p_xy, dtype=np.float64)
    p_arr = p_arr / np.sum(p_arr)
    p_x = np.sum(p_arr, axis=1)
    p_y = np.sum(p_arr, axis=0)
    h_x = shannon_entropy(p_x)
    h_y = shannon_entropy(p_y)
    h_xy = joint_entropy(p_arr)
    h_x_given_y = conditional_entropy(p_arr, given_axis=1)
    mi_sum = max(0.0, h_x + h_y - h_xy)
    mi_cond = max(0.0, h_x - h_x_given_y)
    mi_kl = max(0.0, kl_divergence(p_arr, np.outer(p_x, p_y)))
    return {
        "H(X)": h_x,
        "H(Y)": h_y,
        "H(X,Y)": h_xy,
        "H(X|Y)": h_x_given_y,
        "I(X;Y)": mi_sum,
        "I_cond": mi_cond,
        "I_KL": mi_kl,
    }


def landauer_erasure_energy(
    temperature_k: float = 300.0,
    bits_erased: float = 1.0,
    clock_hz: float = 3.0e9,
) -> dict[str, float]:
    """Compute Landauer's thermodynamic lower bound on irreversible bit erasure."""
    e_per_bit = K_BOLTZMANN * float(temperature_k) * LN2
    energy_j = float(bits_erased) * e_per_bit
    power_w = energy_j * float(clock_hz)
    return {
        "temperature_k": float(temperature_k),
        "bits_erased": float(bits_erased),
        "e_bit_joules": float(e_per_bit),
        "energy_joules": float(energy_j),
        "kT_ln2_multiples": float(bits_erased),
        "power_watts": float(power_w),
    }


def _ksg_mi_2d(x: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42) -> float:
    """Estimate continuous mutual information I(X; Y) in bits via KSG Algorithm 1."""
    rng = np.random.default_rng(seed)
    n = len(x)
    x_s = (x - np.mean(x)) / (np.std(x) + 1e-12) + rng.normal(0.0, 1e-8, n)
    y_s = (y - np.mean(y)) / (np.std(y) + 1e-12) + rng.normal(0.0, 1e-8, n)
    xy = np.column_stack([x_s, y_s])
    x_2d = x_s.reshape(-1, 1)
    y_2d = y_s.reshape(-1, 1)

    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x_2d)
    tree_y = cKDTree(y_2d)
    dists, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = np.maximum(dists[:, k] - 1e-15, 1e-15)
    nx = tree_x.query_ball_point(x_2d, r=eps, p=np.inf, return_length=True)
    ny = tree_y.query_ball_point(y_2d, r=eps, p=np.inf, return_length=True)
    mi_nats = digamma(k) + digamma(n) - np.mean(digamma(nx) + digamma(ny))
    return max(0.0, float(mi_nats / LN2))


def circular_nonlinearity_benchmark(
    n_samples: int = 2000,
    radius: float = 1.0,
    noise_std: float = 0.08,
    n_bins: int = 16,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate X^2 + Y^2 ~ R^2 + eps proving Pearson r ~ 0 while I(X; Y) > 0."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    x = radius * np.cos(theta) + rng.normal(0.0, noise_std, size=n_samples)
    y = radius * np.sin(theta) + rng.normal(0.0, noise_std, size=n_samples)

    pearson_r, pearson_p = pearsonr(x, y)

    # Quantile-binned histogram MI (finite-sample bias mitigated via reasonable bin count)
    x_edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    y_edges = np.quantile(y, np.linspace(0.0, 1.0, n_bins + 1))
    hist_xy, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    p_xy = hist_xy / np.sum(hist_xy)
    mi_binned = mutual_information(p_xy)["I(X;Y)"]
    mi_ksg = _ksg_mi_2d(x, y, k=5, seed=seed)

    return {
        "x": x,
        "y": y,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "mi_binned_bits": float(mi_binned),
        "mi_ksg_bits": float(mi_ksg),
        "radius": float(radius),
        "noise_std": float(noise_std),
    }


def effective_flip_probability(
    sigma: float = 0.0,
    coupling: float = 1.0,
    noise_prob: float | None = None,
) -> float:
    """Map channel noise sigma and coupling c to effective symmetric bit-flip probability."""
    if noise_prob is not None:
        return float(np.clip(noise_prob, 0.0, 0.5))
    c = float(np.clip(coupling, 0.0, 1.0))
    s = max(float(sigma), 0.0)
    if s <= 1e-12:
        p_channel = 0.0
    else:
        # BPSK antipodal threshold bit-error probability: Phi(-0.5 / sigma)
        p_channel = 0.5 * float(erfc(0.5 / (np.sqrt(2.0) * s)))
    return float(np.clip(0.5 * (1.0 - c) + c * p_channel, 0.0, 0.5))


def normalize_gate_name(gate_type: str) -> str:
    """Normalize gate/mechanism aliases cleanly."""
    key = gate_type.strip().upper()
    aliases = {
        "COPY": "COPY",
        "COPY_X1": "COPY",
        "XOR": "XOR",
        "AND": "AND",
        "OR": "OR",
        "IDENTICAL": "IDENTICAL",
        "CONTINUOUS_GAUSSIAN": "Continuous_Gaussian",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported mechanism '{gate_type}'.")
    return aliases[key]


def build_joint_distribution(
    gate_type: str = "XOR",
    input_corr: float = 0.0,
    noise_prob: float | None = None,
    sigma: float = 0.0,
    coupling: float = 1.0,
) -> np.ndarray:
    """Build exact 2x2x2 joint probability tensor P(X1, X2, Y) for discrete gates."""
    gate = normalize_gate_name(gate_type)
    if gate == "Continuous_Gaussian":
        raise ValueError("Use compute_gaussian_pid() for Continuous_Gaussian mechanism.")

    p_flip = effective_flip_probability(sigma=sigma, coupling=coupling, noise_prob=noise_prob)
    corr = float(np.clip(input_corr, -0.9999, 0.9999))
    if gate == "IDENTICAL":
        corr = 1.0

    p_joint = np.zeros((2, 2, 2), dtype=np.float64)
    for x1 in (0, 1):
        for x2 in (0, 1):
            if gate == "IDENTICAL":
                p_x1_x2 = 0.5 if x1 == x2 else 0.0
            else:
                sign = 1.0 if x1 == x2 else -1.0
                p_x1_x2 = 0.25 * (1.0 + sign * corr)

            if gate == "XOR":
                y_det = x1 ^ x2
            elif gate == "AND":
                y_det = x1 & x2
            elif gate == "OR":
                y_det = x1 | x2
            elif gate in ("COPY", "IDENTICAL"):
                y_det = x1
            else:
                y_det = x1 ^ x2

            for y in (0, 1):
                p_y_given_x = (1.0 - p_flip) if y == y_det else p_flip
                p_joint[x1, x2, y] = p_x1_x2 * p_y_given_x

    p_joint /= np.sum(p_joint)
    return p_joint


def compute_specific_information(p_joint_xy: np.ndarray, y_val: int) -> float:
    """Compute Williams & Beer specific information I(X; Y=y) in bits."""
    p_x = np.sum(p_joint_xy, axis=-1)
    p_y = np.sum(p_joint_xy, axis=tuple(range(p_joint_xy.ndim - 1)))
    if p_y[y_val] <= 1e-15:
        return 0.0
    p_x_given_y = p_joint_xy[..., y_val] / p_y[y_val]
    mask = (p_x_given_y > 1e-15) & (p_x > 1e-15)
    return float(np.sum(p_x_given_y[mask] * np.log2(p_x_given_y[mask] / p_x[mask])))


def compute_pid_williams_beer(p_joint: np.ndarray) -> dict[str, float]:
    """Compute exact 4-atom Williams & Beer (2010) PID for discrete P(X1, X2, Y)."""
    p_arr = np.asarray(p_joint, dtype=np.float64)
    p_arr = p_arr / np.sum(p_arr)

    p_x1_y = np.sum(p_arr, axis=1)
    p_x2_y = np.sum(p_arr, axis=0)
    p_y = np.sum(p_arr, axis=(0, 1))
    p_x1_x2 = np.sum(p_arr, axis=2)

    h_y = shannon_entropy(p_y)
    h_x1_x2 = shannon_entropy(p_x1_x2)
    h_joint = joint_entropy(p_arr)
    h_y_given_x = max(0.0, h_joint - h_x1_x2)

    mi_joint = max(0.0, h_x1_x2 + h_y - h_joint)
    mi_x1_y = max(0.0, shannon_entropy(np.sum(p_x1_y, axis=1)) + h_y - shannon_entropy(p_x1_y))
    mi_x2_y = max(0.0, shannon_entropy(np.sum(p_x2_y, axis=1)) + h_y - shannon_entropy(p_x2_y))

    redundancy = 0.0
    for y_val in range(p_y.shape[0]):
        if p_y[y_val] > 1e-15:
            i_spec_1 = compute_specific_information(p_x1_y, y_val)
            i_spec_2 = compute_specific_information(p_x2_y, y_val)
            redundancy += float(p_y[y_val]) * min(i_spec_1, i_spec_2)

    unique_x1 = max(0.0, mi_x1_y - redundancy)
    unique_x2 = max(0.0, mi_x2_y - redundancy)
    synergy = max(0.0, mi_joint - (redundancy + unique_x1 + unique_x2))
    co_info = mi_x1_y + mi_x2_y - mi_joint  # Exact R - S

    # Thermodynamic bit erasure from 2-input state compression + noise entropy
    bits_erased = max(0.0, h_x1_x2 - h_y + h_y_given_x)
    return {
        "H(Y)": h_y,
        "H(X1,X2)": h_x1_x2,
        "H(Y|X1,X2)": h_y_given_x,
        "I(X1; Y)": mi_x1_y,
        "I(X2; Y)": mi_x2_y,
        "I(X1, X2; Y)": mi_joint,
        "Redundancy": redundancy,
        "Unique_X1": unique_x1,
        "Unique_X2": unique_x2,
        "Synergy": synergy,
        "Co_Information": co_info,
        "Bits_Erased": bits_erased,
    }


def compute_gaussian_pid(
    rho: float = 0.0,
    sigma: float = 0.25,
    coupling: float = 1.0,
) -> dict[str, float]:
    """Compute exact MMI-PID (Barrett, 2015) for jointly Gaussian Y = X1 + c*X2 + sigma*eps."""
    r = float(np.clip(rho, -0.95, 0.95))
    c = float(np.clip(coupling, 0.0, 1.0))
    sig = max(float(sigma), 0.10)

    var_y = 1.0 + c**2 + 2.0 * c * r + sig**2
    cov_x1_y = 1.0 + c * r
    cov_x2_y = r + c
    r1_sq = min(0.999999, (cov_x1_y**2) / var_y)
    r2_sq = min(0.999999, (cov_x2_y**2) / var_y)

    mi_x1_y = max(0.0, -0.5 * float(np.log2(1.0 - r1_sq)))
    mi_x2_y = max(0.0, -0.5 * float(np.log2(1.0 - r2_sq)))
    mi_joint = max(0.0, 0.5 * float(np.log2(var_y / (sig**2))))

    redundancy = min(mi_x1_y, mi_x2_y)
    unique_x1 = max(0.0, mi_x1_y - redundancy)
    unique_x2 = max(0.0, mi_x2_y - redundancy)
    synergy = max(0.0, mi_joint - (redundancy + unique_x1 + unique_x2))
    co_info = redundancy - synergy
    h_y_diff = 0.5 * float(np.log2(2.0 * np.pi * np.e * var_y))
    h_y_given_x_diff = 0.5 * float(np.log2(2.0 * np.pi * np.e * sig**2))

    return {
        "H(Y)": h_y_diff,
        "H(X1,X2)": float(np.log2(2.0 * np.pi * np.e) + 0.5 * np.log2(1.0 - r**2)),
        "H(Y|X1,X2)": h_y_given_x_diff,
        "I(X1; Y)": mi_x1_y,
        "I(X2; Y)": mi_x2_y,
        "I(X1, X2; Y)": mi_joint,
        "Redundancy": redundancy,
        "Unique_X1": unique_x1,
        "Unique_X2": unique_x2,
        "Synergy": synergy,
        "Co_Information": co_info,
        "Bits_Erased": max(0.0, 1.0 - min(1.0, mi_joint / max(h_y_diff, 1.0))),
    }


def evaluate_mechanism_pid(
    gate_type: str = "XOR",
    input_corr: float = 0.0,
    sigma: float = 0.0,
    coupling: float = 1.0,
    noise_prob: float | None = None,
    temperature_k: float = 300.0,
) -> dict[str, float]:
    """Evaluate PID metrics and Landauer bound for any supported discrete/Gaussian mechanism."""
    gate = normalize_gate_name(gate_type)
    if gate == "Continuous_Gaussian":
        res = compute_gaussian_pid(rho=input_corr, sigma=sigma, coupling=coupling)
    else:
        p_joint = build_joint_distribution(
            gate_type=gate,
            input_corr=input_corr,
            noise_prob=noise_prob,
            sigma=sigma,
            coupling=coupling,
        )
        res = compute_pid_williams_beer(p_joint)

    thermo = landauer_erasure_energy(temperature_k=temperature_k, bits_erased=res["Bits_Erased"])
    res.update(
        {
            "Landauer_J": thermo["energy_joules"],
            "Landauer_kTln2": thermo["kT_ln2_multiples"],
            "Temperature_K": float(temperature_k),
        }
    )
    return res


def draw_information_venn(ax: plt.Axes, res: dict[str, float], gate_label: str) -> None:
    """Draw a clean 3-source/target Information Venn Diagram annotated with PID atoms."""
    ax.set_facecolor(PALETTE["bg"])
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.7, 1.9)
    ax.set_aspect("equal")
    ax.axis("off")

    c_x1 = Circle((-0.62, -0.18), 1.05, facecolor=PALETTE["cobalt"], alpha=0.14, edgecolor=PALETTE["cobalt"], lw=2.0)
    c_x2 = Circle((0.62, -0.18), 1.05, facecolor=PALETTE["amber"], alpha=0.14, edgecolor=PALETTE["amber"], lw=2.0)
    c_y = Circle((0.0, 0.56), 1.05, facecolor=PALETTE["teal"], alpha=0.14, edgecolor=PALETTE["teal"], lw=2.0)
    for patch in (c_x1, c_x2, c_y):
        ax.add_patch(patch)

    ax.text(-1.45, -1.10, "$X_1$", fontsize=12, fontweight="bold", color=PALETTE["cobalt"], ha="center")
    ax.text(1.45, -1.10, "$X_2$", fontsize=12, fontweight="bold", color=PALETTE["amber"], ha="center")
    ax.text(0.0, 1.72, "Target $Y$", fontsize=12, fontweight="bold", color=PALETTE["teal"], ha="center")

    # Atom value annotations inside intersections
    ax.text(-0.56, 0.36, f"$U_1$\n{res['Unique_X1']:.3f} b", ha="center", va="center", fontsize=9.5, fontweight="bold", color=PALETTE["cobalt"])
    ax.text(0.56, 0.36, f"$U_2$\n{res['Unique_X2']:.3f} b", ha="center", va="center", fontsize=9.5, fontweight="bold", color=PALETTE["amber"])
    ax.text(0.0, 0.16, f"$R$\n{res['Redundancy']:.3f} b", ha="center", va="center", fontsize=9.8, fontweight="bold", color=PALETTE["teal"])
    ax.text(0.0, 1.05, f"Synergy $S = {res['Synergy']:.3f}$ b\n(in $X_1 \\otimes X_2 \\to Y$)", ha="center", va="center", fontsize=9.2, fontweight="bold", color=PALETTE["crimson"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF1F2", edgecolor=PALETTE["crimson"], lw=1.1))

    co_info = res["Co_Information"]
    regime = "Synergy-Dominated ($II < 0$)" if co_info < -1e-4 else ("Redundancy-Dominated ($II > 0$)" if co_info > 1e-4 else "Balanced / Unique ($II = 0$)")
    ax.text(
        0.0,
        -1.50,
        f"Co-Information $II = R - S = {co_info:+.3f}$ b  |  {regime}",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=PALETTE["ink"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["card"], edgecolor=PALETTE["grid"]),
    )
    ax.set_title(f"3. Information Venn Diagram ({gate_label})", fontsize=11.5, fontweight="bold", pad=6)


def create_pid_dashboard_figure(
    gate_type: str = "XOR",
    input_corr: float = 0.0,
    sigma: float = 0.0,
    coupling: float = 1.0,
    noise_prob: float | None = None,
    temperature_k: float = 300.0,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Build a 4-panel classroom dashboard for Module 01."""
    apply_publication_style()
    gate = normalize_gate_name(gate_type)
    res = evaluate_mechanism_pid(
        gate_type=gate,
        input_corr=input_corr,
        sigma=sigma,
        coupling=coupling,
        noise_prob=noise_prob,
        temperature_k=temperature_k,
    )
    nonlin = circular_nonlinearity_benchmark(n_samples=1600, seed=42)

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2))
    fig.patch.set_facecolor(PALETTE["bg"])

    # Panel 1: Pairwise vs Joint Mutual Information + PID Atom Bar Chart
    ax1 = style_axes(
        axes[0, 0],
        title=f"1. Pairwise vs. Joint MI ({gate}, rho={input_corr:+.2f})",
        ylabel="Information (Bits)",
        grid_axis="y",
    )
    mi_labels = ["$I(X_1; Y)$", "$I(X_2; Y)$", "$I(X_1, X_2; Y)$", "$H(Y)$"]
    mi_vals = [res["I(X1; Y)"], res["I(X2; Y)"], res["I(X1, X2; Y)"], res["H(Y)"]]
    mi_colors = [PALETTE["slate"], PALETTE["muted"], PALETTE["cobalt"], PALETTE["teal"]]
    bars1 = ax1.bar(mi_labels, mi_vals, color=mi_colors, width=0.52, edgecolor=PALETTE["ink"], linewidth=1.1)
    y_top1 = max(1.18, max(mi_vals) * 1.22)
    ax1.set_ylim(0, y_top1)
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.025 * y_top1, f"{h:.3f} b", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    # Panel 2: Williams & Beer 4-Atom PID Decomposition
    ax2 = style_axes(
        axes[0, 1],
        title="2. Williams & Beer 4-Atom PID Decomposition",
        ylabel="Decomposed Information (Bits)",
        grid_axis="y",
    )
    pid_labels = ["Redundancy\n$R$", "Unique $X_1$\n$U_1$", "Unique $X_2$\n$U_2$", "Synergy\n$S$"]
    pid_vals = [res["Redundancy"], res["Unique_X1"], res["Unique_X2"], res["Synergy"]]
    pid_colors = [PALETTE["teal"], PALETTE["cobalt"], PALETTE["amber"], PALETTE["crimson"]]
    bars2 = ax2.bar(pid_labels, pid_vals, color=pid_colors, width=0.52, edgecolor=PALETTE["ink"], linewidth=1.1)
    y_top2 = max(1.18, max(pid_vals) * 1.25)
    ax2.set_ylim(0, y_top2)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.025 * y_top2, f"{h:.3f} b", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    # Panel 3: Information Venn Diagram
    draw_information_venn(axes[1, 0], res, gate)

    # Panel 4: Circular Nonlinearity Benchmark & Binary Entropy inset summary
    ax4 = style_axes(
        axes[1, 1],
        title="4. Nonlinearity Benchmark: $X^2 + Y^2 \\approx R^2 + \\epsilon$",
        xlabel="$X = R\\cos\\theta + \\epsilon_x$",
        ylabel="$Y = R\\sin\\theta + \\epsilon_y$",
    )
    ax4.scatter(nonlin["x"], nonlin["y"], s=9, alpha=0.55, color=PALETTE["cobalt"], edgecolors="none", label="Circular Samples ($N=1600$)")
    ax4.set_aspect("equal", adjustable="datalim")
    ax4.text(
        0.03,
        0.96,
        (
            f"Pearson $r = {nonlin['pearson_r']:+.4f}$ ($p={nonlin['pearson_p']:.2f}$)\n"
            f"MI (Binned) $= {nonlin['mi_binned_bits']:.3f}$ bits\n"
            f"MI (KSG $k$=5) $= {nonlin['mi_ksg_bits']:.3f}$ bits\n"
            f"Landauer ($T={temperature_k:.0f}$ K): {res['Landauer_J']:.2e} J"
        ),
        transform=ax4.transAxes,
        va="top",
        ha="left",
        fontsize=8.8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=PALETTE["card"], edgecolor=PALETTE["cobalt"], lw=1.0),
    )

    fig.suptitle(
        f"Module 01 Dashboard — {gate} | rho={input_corr:+.2f}, sigma={sigma:.2f}, c={coupling:.2f}, T={temperature_k:.0f} K",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, {"pid": res, "nonlinearity": nonlin}


def run_module01_tier1(
    sigma: float = 0.1,
    coupling: float = 1.0,
    temperature_k: float = 300.0,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 1: Verify XOR pure synergy (~1.0 bit) & zero marginal MI + 4-gate calibration."""
    apply_publication_style()
    gates = ["COPY", "IDENTICAL", "AND", "OR", "XOR"]
    table = {
        g: evaluate_mechanism_pid(g, input_corr=0.0, sigma=sigma, coupling=coupling, temperature_k=temperature_k)
        for g in gates
    }
    xor_res = table["XOR"]
    nonlin = circular_nonlinearity_benchmark()

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7))
    fig.patch.set_facecolor(PALETTE["bg"])

    # Panel A: 4 Canonical Gate Calibration Stacked Bar
    ax0 = style_axes(axes[0], title="Tier 1A: Canonical Logic Gate Calibration", ylabel="Information (Bits)", grid_axis="y")
    x_pos = np.arange(len(gates))
    bottoms = np.zeros(len(gates))
    for key, label, col in [
        ("Redundancy", "Redundancy $R$", PALETTE["teal"]),
        ("Unique_X1", "Unique $U_1$", PALETTE["cobalt"]),
        ("Unique_X2", "Unique $U_2$", PALETTE["amber"]),
        ("Synergy", "Synergy $S$", PALETTE["crimson"]),
    ]:
        vals = np.array([table[g][key] for g in gates])
        ax0.bar(x_pos, vals, bottom=bottoms, width=0.58, label=label, color=col, edgecolor=PALETTE["ink"], lw=0.9)
        bottoms += vals
    ax0.set_xticks(x_pos)
    ax0.set_xticklabels(gates, fontsize=9.5, fontweight="bold")
    ax0.set_ylim(0, 1.20)
    ax0.legend(loc="upper right", fontsize=8.3, ncol=2)

    # Panel B: Binary Entropy H_b(p) Curve + XOR output marker
    ax1 = style_axes(axes[1], title="Tier 1B: Binary Entropy $H_b(p)$ & XOR Check", xlabel="$P(Y=1)$", ylabel="Entropy (Bits)")
    p_grid = np.linspace(0.001, 0.999, 250)
    ax1.plot(p_grid, binary_entropy(p_grid), color=PALETTE["teal"], lw=2.5, label="$H_b(p)$")
    ax1.scatter([0.5], [xor_res["H(Y)"]], color=PALETTE["crimson"], s=90, zorder=5, label=f"XOR $H(Y)={xor_res['H(Y)']:.3f}$ b")
    ax1.annotate(
        f"$I(X_1;Y)={xor_res['I(X1; Y)']:.3f}$\n$I(X_2;Y)={xor_res['I(X2; Y)']:.3f}$\nJoint $I={xor_res['I(X1, X2; Y)']:.3f}$ b",
        xy=(0.5, xor_res["H(Y)"]),
        xytext=(0.14, 0.38),
        arrowprops=dict(arrowstyle="->", color=PALETTE["crimson"], lw=1.4),
        fontsize=8.8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF1F2", edgecolor=PALETTE["crimson"]),
    )
    ax1.legend(loc="lower center", fontsize=8.5)

    # Panel C: Circular Nonlinearity Proof
    ax2 = style_axes(axes[2], title="Tier 1C: Circular Nonlinearity ($r \\approx 0, I > 0$)", xlabel="$X$", ylabel="$Y$")
    ax2.scatter(nonlin["x"], nonlin["y"], s=8, alpha=0.5, color=PALETTE["cobalt"])
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.text(
        0.04,
        0.95,
        f"Pearson $r = {nonlin['pearson_r']:+.4f}$\nKSG MI $= {nonlin['mi_ksg_bits']:.3f}$ b",
        transform=ax2.transAxes,
        va="top",
        fontsize=9.2,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=PALETTE["card"], edgecolor=PALETTE["cobalt"]),
    )

    fig.suptitle(f"Module 01 — Tier 1 (Observation): XOR Pure Synergy & Canonical Gates (sigma={sigma:.2f}, c={coupling:.2f})", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {"gates": table, "nonlinearity": nonlin}


def run_module01_tier2(
    gate_type: str = "AND",
    sigma: float = 0.0,
    coupling: float = 1.0,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 2: Sweep Correlation rho from 0.0 to 0.95 on AND; plot R vs S crossover & II = R - S."""
    apply_publication_style()
    rhos = np.linspace(0.0, 0.95, 40)
    rdn, u1, u2, syn, co_info, mi_joint = [], [], [], [], [], []
    for rho in rhos:
        r = evaluate_mechanism_pid(gate_type=gate_type, input_corr=float(rho), sigma=sigma, coupling=coupling)
        rdn.append(r["Redundancy"])
        u1.append(r["Unique_X1"])
        u2.append(r["Unique_X2"])
        syn.append(r["Synergy"])
        co_info.append(r["Co_Information"])
        mi_joint.append(r["I(X1, X2; Y)"])

    rdn_arr = np.asarray(rdn)
    syn_arr = np.asarray(syn)
    co_arr = np.asarray(co_info)

    # Locate exact crossover rho* where II = R - S crosses zero
    sign_changes = np.where(np.diff(np.sign(co_arr)) != 0)[0]
    rho_star = None
    if len(sign_changes) > 0:
        idx = int(sign_changes[0])
        c0, c1 = co_arr[idx], co_arr[idx + 1]
        rho_star = float(rhos[idx] - c0 * (rhos[idx + 1] - rhos[idx]) / (c1 - c0))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax0 = style_axes(axes[0], title=f"Tier 2A: {gate_type} Redundancy vs. Synergy Crossover", xlabel="Input Correlation $\\rho$", ylabel="Information (Bits)")
    ax0.plot(rhos, rdn_arr, color=PALETTE["teal"], lw=2.6, label="Redundancy $R(\\rho)$")
    ax0.plot(rhos, syn_arr, color=PALETTE["crimson"], lw=2.6, label="Synergy $S(\\rho)$")
    ax0.plot(rhos, mi_joint, color=PALETTE["cobalt"], lw=1.8, linestyle="--", label="Joint MI $I(X_1, X_2; Y)$")
    if rho_star is not None:
        r_cross = float(np.interp(rho_star, rhos, rdn_arr))
        ax0.axvline(rho_star, color=PALETTE["amber"], linestyle=":", lw=1.8)
        ax0.scatter([rho_star], [r_cross], color=PALETTE["amber"], s=85, zorder=6, label=f"Crossover $\\rho^*={rho_star:.3f}$")
    ax0.legend(loc="best", fontsize=9.0)

    ax1 = style_axes(axes[1], title="Tier 2B: Co-Information $II(\\rho) = R - S$ Zero-Crossing", xlabel="Input Correlation $\\rho$", ylabel="Co-Information $II$ (Bits)")
    ax1.axhline(0.0, color=PALETTE["ink"], lw=1.2, linestyle="-")
    ax1.fill_between(rhos, co_arr, 0.0, where=(co_arr < 0), color=PALETTE["crimson"], alpha=0.14, label="Synergy-Dominated ($II < 0$)")
    ax1.fill_between(rhos, co_arr, 0.0, where=(co_arr >= 0), color=PALETTE["teal"], alpha=0.14, label="Redundancy-Dominated ($II > 0$)")
    ax1.plot(rhos, co_arr, color=PALETTE["cobalt"], lw=2.5)
    if rho_star is not None:
        ax1.scatter([rho_star], [0.0], color=PALETTE["amber"], s=95, zorder=6, label=f"Zero Crossing $\\rho^*={rho_star:.3f}$")
    ax1.legend(loc="upper left", fontsize=8.8)

    fig.suptitle(f"Module 01 — Tier 2 (Analysis): Redundancy–Synergy Crossover on {gate_type} Gate", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {
        "rhos": rhos.tolist(),
        "redundancy": rdn,
        "synergy": syn,
        "co_info": co_info,
        "crossover_rho": rho_star,
    }


def run_module01_tier3(
    temperature_k: float = 300.0,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 3: Sweep sigma 0.0->1.0 across all 4 gates; plot entropy production, MI decay & Landauer heat."""
    apply_publication_style()
    sigmas = np.linspace(0.0, 1.0, 35)
    gates = ["XOR", "AND", "OR", "COPY"]
    gate_colors = {
        "XOR": PALETTE["crimson"],
        "AND": PALETTE["cobalt"],
        "OR": PALETTE["teal"],
        "COPY": PALETTE["amber"],
    }

    traces: dict[str, dict[str, list[float]]] = {
        g: {"mi": [], "h_cond": [], "heat_j": [], "heat_kt": []} for g in gates
    }
    for g in gates:
        for sig in sigmas:
            res = evaluate_mechanism_pid(g, input_corr=0.0, sigma=float(sig), coupling=1.0, temperature_k=temperature_k)
            traces[g]["mi"].append(res["I(X1, X2; Y)"])
            traces[g]["h_cond"].append(res["H(Y|X1,X2)"])
            traces[g]["heat_j"].append(res["Landauer_J"])
            traces[g]["heat_kt"].append(res["Landauer_kTln2"])

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.7))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax0 = style_axes(axes[0], title="Tier 3A: Joint Mutual Information Decay", xlabel="Channel Noise $\\sigma$", ylabel="$I(X_1, X_2; Y)$ (Bits)")
    ax1 = style_axes(axes[1], title="Tier 3B: Channel Entropy Production", xlabel="Channel Noise $\\sigma$", ylabel="Noise Entropy $H(Y \\mid X_1, X_2)$ (Bits)")
    ax2 = style_axes(axes[2], title=f"Tier 3C: Landauer Heat Bound ($T={temperature_k:.0f}$ K)", xlabel="Channel Noise $\\sigma$", ylabel="Min Heat ($10^{-21}$ J / op)")

    for g in gates:
        col = gate_colors[g]
        ax0.plot(sigmas, traces[g]["mi"], color=col, lw=2.3, label=g)
        ax1.plot(sigmas, traces[g]["h_cond"], color=col, lw=2.3, label=g)
        ax2.plot(sigmas, np.array(traces[g]["heat_j"]) * 1e21, color=col, lw=2.3, label=f"{g} (bits={traces[g]['heat_kt'][-1]:.2f} $k_BT\\ln 2$)")

    ax2_r = ax2.twinx()
    y_min_j, y_max_j = ax2.get_ylim()
    unit_kt_zJ = landauer_erasure_energy(temperature_k=temperature_k, bits_erased=1.0)["e_bit_joules"] * 1e21
    ax2_r.set_ylim(y_min_j / unit_kt_zJ, y_max_j / unit_kt_zJ)
    ax2_r.set_ylabel("Multiples of $k_B T \\ln 2$", fontsize=9.5, fontweight="bold", color=PALETTE["slate"])
    ax2_r.spines["top"].set_visible(False)

    for ax in (ax0, ax1, ax2):
        ax.legend(loc="best", fontsize=8.5)

    fig.suptitle("Module 01 — Tier 3 (Stress Test): Channel Noise & Landauer Thermodynamic Erasure Bound", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {"sigmas": sigmas.tolist(), "traces": traces, "temperature_k": temperature_k}


def generate_module01_gifs(out_dir: str | Path) -> list[Path]:
    """Generate classroom-ready animated GIFs for Module 01."""
    apply_publication_style(dpi=115)
    anim_dir = Path(out_dir) / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # GIF 1: AND-gate Correlation Crossover (rho: 0.0 -> 0.95)
    rhos = np.linspace(0.0, 0.95, 28)
    all_res = [evaluate_mechanism_pid("AND", input_corr=float(r)) for r in rhos]
    r_vals = [d["Redundancy"] for d in all_res]
    s_vals = [d["Synergy"] for d in all_res]
    co_vals = [d["Co_Information"] for d in all_res]

    frames_1 = []
    for idx, rho in enumerate(rhos):
        d = all_res[idx]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
        fig.patch.set_facecolor(PALETTE["bg"])

        ax0 = style_axes(axes[0], title=f"AND Gate PID Atoms ($\\rho = {rho:.2f}$)", ylabel="Bits", grid_axis="y")
        labels = ["Redundancy $R$", "Unique $U_1$", "Unique $U_2$", "Synergy $S$"]
        vals = [d["Redundancy"], d["Unique_X1"], d["Unique_X2"], d["Synergy"]]
        cols = [PALETTE["teal"], PALETTE["cobalt"], PALETTE["amber"], PALETTE["crimson"]]
        bars = ax0.bar(labels, vals, color=cols, width=0.55, edgecolor=PALETTE["ink"], lw=1.0)
        ax0.set_ylim(0, 1.08)
        for b in bars:
            h = b.get_height()
            ax0.text(b.get_x() + b.get_width() / 2, h + 0.025, f"{h:.3f}", ha="center", fontsize=9, fontweight="bold")

        ax1 = style_axes(axes[1], title="Crossover Sweep: $R(\\rho)$ vs $S(\\rho)$ & $II = R - S$", xlabel="Correlation $\\rho$", ylabel="Information (Bits)")
        ax1.axhline(0.0, color=PALETTE["ink"], lw=1.0, alpha=0.6)
        ax1.plot(rhos[: idx + 1], r_vals[: idx + 1], color=PALETTE["teal"], lw=2.5, label="$R(\\rho)$")
        ax1.plot(rhos[: idx + 1], s_vals[: idx + 1], color=PALETTE["crimson"], lw=2.5, label="$S(\\rho)$")
        ax1.plot(rhos[: idx + 1], co_vals[: idx + 1], color=PALETTE["cobalt"], lw=2.0, linestyle="--", label="$II = R - S$")
        ax1.scatter([rho, rho, rho], [r_vals[idx], s_vals[idx], co_vals[idx]], color=[PALETTE["teal"], PALETTE["crimson"], PALETTE["cobalt"]], s=55, zorder=5)
        ax1.set_xlim(-0.02, 0.98)
        ax1.set_ylim(-0.30, 1.05)
        ax1.legend(loc="upper left", fontsize=8.5)

        fig.suptitle("Module 01 Live Sweep: Correlation Converts Synergy into Redundancy", fontsize=12, fontweight="bold")
        fig.tight_layout()
        frames_1.append(fig_to_pil_image(fig, dpi=110))
        plt.close(fig)

    generated.append(save_gif_from_frames(frames_1, anim_dir / "01_pid_correlation_crossover_AND.gif", duration_ms=130))

    # GIF 2: Channel Noise & Landauer Heat Sweep (sigma: 0.0 -> 1.0)
    sigmas = np.linspace(0.0, 1.0, 26)
    gates = ["XOR", "AND", "OR", "COPY"]
    cols_g = {"XOR": PALETTE["crimson"], "AND": PALETTE["cobalt"], "OR": PALETTE["teal"], "COPY": PALETTE["amber"]}
    sweep = {g: [evaluate_mechanism_pid(g, sigma=float(s)) for s in sigmas] for g in gates}

    frames_2 = []
    for idx, sig in enumerate(sigmas):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
        fig.patch.set_facecolor(PALETTE["bg"])

        ax0 = style_axes(axes[0], title=f"Mutual Information Decay ($\\sigma = {sig:.2f}$)", xlabel="Channel Noise $\\sigma$", ylabel="Joint MI (Bits)")
        ax1 = style_axes(axes[1], title="Landauer Erasure Heat ($T = 300$ K)", ylabel="Minimum Heat ($\\times 10^{-21}$ J / op)", grid_axis="y")

        for g in gates:
            mi_trace = [d["I(X1, X2; Y)"] for d in sweep[g][: idx + 1]]
            ax0.plot(sigmas[: idx + 1], mi_trace, color=cols_g[g], lw=2.3, label=g)
            ax0.scatter([sig], [mi_trace[-1]], color=cols_g[g], s=50)
        ax0.set_xlim(-0.02, 1.02)
        ax0.set_ylim(0.0, 1.08)
        ax0.legend(loc="upper right", fontsize=8.5)

        heats = [sweep[g][idx]["Landauer_J"] * 1e21 for g in gates]
        bars = ax1.bar(gates, heats, color=[cols_g[g] for g in gates], width=0.52, edgecolor=PALETTE["ink"], lw=1.0)
        ax1.set_ylim(0.0, 6.5)
        for b in bars:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2, h + 0.14, f"{h:.2f} zJ", ha="center", fontsize=8.8, fontweight="bold")

        fig.suptitle("Module 01 Thermodynamic Stress Test: Noise, Erasure & Landauer Heat", fontsize=12, fontweight="bold")
        fig.tight_layout()
        frames_2.append(fig_to_pil_image(fig, dpi=110))
        plt.close(fig)

    generated.append(save_gif_from_frames(frames_2, anim_dir / "02_landauer_noise_stress_gates.gif", duration_ms=130))
    return generated
