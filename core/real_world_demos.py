"""All 12 Domain-Specific Real-World Applications for Modules 01-04."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import pearsonr

from core.cellular_automata import (
    HansonCrutchfieldFST,
    evolve_eca_from_state,
)
from core.info_theory_pid import (
    K_BOLTZMANN,
    LN2,
    compute_pid_williams_beer,
    landauer_erasure_energy,
    shannon_entropy,
)
from core.plotting_utils import (
    PALETTE,
    apply_publication_style,
    fig_to_pil_image,
    save_gif_from_frames,
    style_axes,
)
from core.transfer_entropy import (
    compute_transfer_entropy,
    infer_directed_network,
)


# ==============================================================================
# MODULE 01 REAL-WORLD APPLICATIONS (3 DEMOS)
# ==============================================================================


def demo_m01_computational_neuroscience(
    n_trials: int = 1800,
    noise_rate: float = 0.06,
    seed: int = 42,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Demo 1.1: Cortical population coincidence detection (Synergy) vs redundant firing (Redundancy)."""
    apply_publication_style()
    rng = np.random.default_rng(seed)

    def _empirical_pid(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> dict[str, float]:
        p = np.zeros((2, 2, 2), dtype=np.float64)
        for a, b, c in zip(x1, x2, y):
            p[int(a), int(b), int(c)] += 1.0
        p /= np.sum(p)
        return compute_pid_williams_beer(p)

    # Regime A: Synergistic coincidence detection (stimulus Y encoded by relative phase / XOR coincidence)
    x1_syn = rng.integers(0, 2, size=n_trials)
    x2_syn = rng.integers(0, 2, size=n_trials)
    flips_a = (rng.random(n_trials) < noise_rate).astype(int)
    y_syn = np.bitwise_xor(np.bitwise_xor(x1_syn, x2_syn), flips_a)
    pid_syn = _empirical_pid(x1_syn, x2_syn, y_syn)

    # Regime B: Redundant population firing (both cortical pools track stimulus Y in parallel)
    y_red = rng.integers(0, 2, size=n_trials)
    x1_red = np.bitwise_xor(y_red, (rng.random(n_trials) < noise_rate).astype(int))
    x2_red = np.bitwise_xor(y_red, (rng.random(n_trials) < noise_rate).astype(int))
    pid_red = _empirical_pid(x1_red, x2_red, y_red)

    # Continuous firing rates (spikes/sec) associated with the binary burst states for visualization
    fr1_syn = 12.0 + 34.0 * x1_syn + rng.normal(0.0, 4.0, n_trials)
    fr2_syn = 12.0 + 34.0 * x2_syn + rng.normal(0.0, 4.0, n_trials)

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.8))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax0 = style_axes(
        axes[0],
        title="1. Cortical Coincidence Firing ($Y = X_1 \\oplus X_2$)",
        xlabel="Population $X_1$ Firing Rate (Hz)",
        ylabel="Population $X_2$ Firing Rate (Hz)",
    )
    mask0 = y_syn[:400] == 0
    mask1 = y_syn[:400] == 1
    ax0.scatter(fr1_syn[:400][mask0], fr2_syn[:400][mask0], s=22, alpha=0.65, color=PALETTE["cobalt"], label="Stimulus $Y=0$")
    ax0.scatter(fr1_syn[:400][mask1], fr2_syn[:400][mask1], s=22, alpha=0.65, color=PALETTE["crimson"], label="Stimulus $Y=1$")
    ax0.legend(loc="upper center", fontsize=8.3)

    ax1 = style_axes(axes[1], title="2. PID Comparison: Coincidence vs. Redundant Coding", ylabel="Information (Bits)", grid_axis="y")
    atoms = ["Redundancy $R$", "Unique $U_1$", "Unique $U_2$", "Synergy $S$"]
    vals_syn = [pid_syn["Redundancy"], pid_syn["Unique_X1"], pid_syn["Unique_X2"], pid_syn["Synergy"]]
    vals_red = [pid_red["Redundancy"], pid_red["Unique_X1"], pid_red["Unique_X2"], pid_red["Synergy"]]
    xp = np.arange(4)
    ax1.bar(xp - 0.17, vals_syn, width=0.34, color=PALETTE["crimson"], edgecolor=PALETTE["ink"], label="Synergistic Regime")
    ax1.bar(xp + 0.17, vals_red, width=0.34, color=PALETTE["teal"], edgecolor=PALETTE["ink"], label="Redundant Regime")
    ax1.set_xticks(xp)
    ax1.set_xticklabels(atoms, fontsize=8.8, fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper center", fontsize=8.2)

    # Trial sample convergence of Synergy S vs trial count
    ax2 = style_axes(axes[2], title="3. Finite-Sample PID Estimation Across Trials", xlabel="Trial Count $N$", ylabel="Estimated Synergy / Redundancy (Bits)")
    trial_grid = [100, 250, 500, 900, 1400, n_trials]
    syn_curve = [_empirical_pid(x1_syn[:m], x2_syn[:m], y_syn[:m])["Synergy"] for m in trial_grid]
    red_curve = [_empirical_pid(x1_red[:m], x2_red[:m], y_red[:m])["Redundancy"] for m in trial_grid]
    ax2.plot(trial_grid, syn_curve, marker="o", lw=2.3, color=PALETTE["crimson"], label="Synergistic Circuit $S$")
    ax2.plot(trial_grid, red_curve, marker="s", lw=2.3, color=PALETTE["teal"], label="Redundant Circuit $R$")
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc="lower right", fontsize=8.5)

    fig.suptitle("Module 01 Real-World App 1: Computational Neuroscience & Cortical Population Coding", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {"synergistic_regime": pid_syn, "redundant_regime": pid_red}


def demo_m01_genomics_epistasis(
    n_individuals: int = 2400,
    noise_rate: float = 0.05,
    seed: int = 42,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Demo 1.2: Pure epistatic SNP interaction (Y = X1 XOR X2 + noise) invisible to single-locus GWAS."""
    apply_publication_style()
    rng = np.random.default_rng(seed)
    snp1 = rng.integers(0, 2, size=n_individuals)
    snp2 = rng.integers(0, 2, size=n_individuals)
    flips = (rng.random(n_individuals) < noise_rate).astype(int)
    phenotype = np.bitwise_xor(snp1, snp2) ^ flips

    r1, p1 = pearsonr(snp1, phenotype)
    r2, p2 = pearsonr(snp2, phenotype)

    p_joint = np.zeros((2, 2, 2), dtype=np.float64)
    for g1, g2, y in zip(snp1, snp2, phenotype):
        p_joint[int(g1), int(g2), int(y)] += 1.0
    p_joint /= np.sum(p_joint)
    pid = compute_pid_williams_beer(p_joint)

    penetrance = np.zeros((2, 2), dtype=np.float64)
    for i in (0, 1):
        for j in (0, 1):
            mask = (snp1 == i) & (snp2 == j)
            penetrance[i, j] = float(np.mean(phenotype[mask]))

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.8))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax0 = style_axes(axes[0], title="1. Single-Locus GWAS Misses Epistasis ($r \\approx 0$)", ylabel="Absolute Correlation $|r|$", grid_axis="y")
    bars0 = ax0.bar(["SNP 1 ($X_1$)", "SNP 2 ($X_2$)"], [abs(r1), abs(r2)], color=[PALETTE["slate"], PALETTE["muted"]], width=0.45, edgecolor=PALETTE["ink"])
    ax0.axhline(0.05, color=PALETTE["crimson"], linestyle="--", lw=1.4, label="Weak Effect Ref ($|r|=0.05$)")
    ax0.set_ylim(0, 0.25)
    for b, p_val in zip(bars0, [p1, p2]):
        h = b.get_height()
        ax0.text(b.get_x() + b.get_width() / 2, h + 0.008, f"|r|={h:.4f}\n(p={p_val:.2f})", ha="center", fontsize=8.8, fontweight="bold")
    ax0.legend(loc="upper right", fontsize=8.3)

    im = axes[1].imshow(penetrance, cmap="RdYlBu_r", vmin=0.0, vmax=1.0)
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(["SNP2=0", "SNP2=1"], fontweight="bold")
    axes[1].set_yticklabels(["SNP1=0", "SNP1=1"], fontweight="bold")
    axes[1].set_title("2. 2-Locus Disease Penetrance $P(Y=1 \\mid X_1, X_2)$", fontsize=11, fontweight="bold")
    for i in (0, 1):
        for j in (0, 1):
            axes[1].text(j, i, f"{penetrance[i, j]:.1%}", ha="center", va="center", fontsize=12, fontweight="bold", color=PALETTE["ink"])
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    ax2 = style_axes(axes[2], title="3. PID Uncovers Hidden Genetic Synergy", ylabel="Information (Bits)", grid_axis="y")
    labels2 = ["$I(X_1;Y)$", "$I(X_2;Y)$", "Redundancy $R$", "Synergy $S$"]
    vals2 = [pid["I(X1; Y)"], pid["I(X2; Y)"], pid["Redundancy"], pid["Synergy"]]
    cols2 = [PALETTE["slate"], PALETTE["muted"], PALETTE["teal"], PALETTE["crimson"]]
    b2 = ax2.bar(labels2, vals2, color=cols2, width=0.52, edgecolor=PALETTE["ink"])
    ax2.set_ylim(0, 1.05)
    for b in b2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.025, f"{h:.3f} b", ha="center", fontsize=8.8, fontweight="bold")

    fig.suptitle("Module 01 Real-World App 2: Genomics & Pure Epistasis Discovery Beyond Linear GWAS", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {"r_snp1": float(r1), "p_snp1": float(p1), "r_snp2": float(r2), "p_snp2": float(p2), "pid": pid}


def demo_m01_thermodynamic_reversible_computing(
    n_gates: float = 1e9,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Demo 1.3: Irreversible CMOS reset vs Reversible Fredkin/Toffoli gate dissipation across T and clock f."""
    apply_publication_style()

    # Verify exact information loss delta H = H(X_in) - H(Y_out) for 3-bit gates:
    # 1) Irreversible Bit-Reset / 2-to-1 AND+Zeroing: 8 equiprobable 3-bit inputs -> AND maps 2->1 + resets 2 bits
    #    Explicit 3-bit CMOS irreversible reset: (a,b,c) -> (a & b, 0, 0): loses H(X_in) - H(Y_out) bits!
    p_uniform_3 = np.full(8, 1.0 / 8.0)
    h_in = shannon_entropy(p_uniform_3)  # 3.0 bits

    # Toffoli (CCNOT): (a, b, c) -> (a, b, c ^ (a & b)) — exact bijection on {0..7}
    toffoli_outputs = set()
    fredkin_outputs = set()
    cmos_reset_counts: dict[tuple[int, int, int], int] = {}
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                toffoli_outputs.add((a, b, c ^ (a & b)))
                fredkin_outputs.add((a, c if a == 1 else b, b if a == 1 else c))
                y_cmos = (a & b, 0, 0)
                cmos_reset_counts[y_cmos] = cmos_reset_counts.get(y_cmos, 0) + 1

    h_toffoli = shannon_entropy(np.full(len(toffoli_outputs), 1.0 / 8.0))  # 3.0 bits -> 0 lost
    h_fredkin = shannon_entropy(np.full(len(fredkin_outputs), 1.0 / 8.0))  # 3.0 bits -> 0 lost
    p_cmos = np.array(list(cmos_reset_counts.values()), dtype=np.float64) / 8.0
    h_cmos_out = shannon_entropy(p_cmos)
    erased_cmos = h_in - h_cmos_out  # > 2.18 bits erased per 3-bit irreversible reset
    erased_1bit_reset = 1.0

    temps = np.linspace(4.0, 400.0, 100)
    e_1bit_zJ = np.array([landauer_erasure_energy(t, bits_erased=1.0)["energy_joules"] * 1e21 for t in temps])
    e_cmos3_zJ = np.array([landauer_erasure_energy(t, bits_erased=erased_cmos)["energy_joules"] * 1e21 for t in temps])
    e_rev_zJ = np.zeros_like(temps)

    freqs = np.logspace(6, 10, 60)  # 1 MHz to 10 GHz
    e_bit_300 = K_BOLTZMANN * 300.0 * LN2
    power_1bit_w = n_gates * e_bit_300 * freqs
    power_cmos3_w = n_gates * (erased_cmos * e_bit_300) * freqs

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.8))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax0 = style_axes(axes[0], title="1. Entropy Compression $\\Delta H$ per Logic Step", ylabel="Erased Entropy $\\Delta H$ (Bits)", grid_axis="y")
    gate_names = ["1-Bit Reset\n($2\\to 1$)", "CMOS AND+Reset\n(3-bit)", "Reversible\nFredkin", "Reversible\nToffoli"]
    erased_vals = [erased_1bit_reset, erased_cmos, h_in - h_fredkin, h_in - h_toffoli]
    bars = ax0.bar(gate_names, erased_vals, color=[PALETTE["amber"], PALETTE["crimson"], PALETTE["teal"], PALETTE["cobalt"]], width=0.52, edgecolor=PALETTE["ink"])
    ax0.set_ylim(0, 2.75)
    for b in bars:
        h = b.get_height()
        ax0.text(b.get_x() + b.get_width() / 2, h + 0.06, f"{h:.2f} b", ha="center", fontsize=8.8, fontweight="bold")

    ax1 = style_axes(axes[1], title="2. Minimum Heat vs. Temperature $T$", xlabel="Temperature $T$ (K)", ylabel="Min Heat per Op ($10^{-21}$ J)")
    ax1.plot(temps, e_cmos3_zJ, color=PALETTE["crimson"], lw=2.3, label=f"Irreversible 3-Bit CMOS ($\\Delta H={erased_cmos:.2f}$ b)")
    ax1.plot(temps, e_1bit_zJ, color=PALETTE["amber"], lw=2.3, label="1-Bit Reset ($k_B T \\ln 2$)")
    ax1.plot(temps, e_rev_zJ, color=PALETTE["teal"], lw=2.5, linestyle="--", label="Reversible Fredkin / Toffoli ($0$ J)")
    ax1.axvline(300.0, color=PALETTE["slate"], linestyle=":", lw=1.3)
    ax1.scatter([300.0], [landauer_erasure_energy(300.0, 1.0)["energy_joules"] * 1e21], color=PALETTE["ink"], s=55, zorder=5)
    ax1.annotate("300 K: 2.87 zJ", xy=(300.0, 2.87), xytext=(175.0, 4.3), arrowprops=dict(arrowstyle="->", color=PALETTE["ink"]), fontsize=8.6, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8.0)

    ax2 = style_axes(axes[2], title="3. Landauer Heat Power ($10^9$ Gates @ 300 K)", xlabel="Clock Frequency $f$ (Hz)", ylabel="Min Power Dissipation (W)")
    ax2.loglog(freqs, power_cmos3_w, color=PALETTE["crimson"], lw=2.4, label="Irreversible CMOS Array")
    ax2.loglog(freqs, power_1bit_w, color=PALETTE["amber"], lw=2.3, label="1-Bit Reset Array")
    ax2.axhline(1e-5, color=PALETTE["teal"], linestyle="--", lw=2.0, label="Reversible Limit ($\\to 0$ W)")
    ax2.legend(loc="upper left", fontsize=8.2)

    fig.suptitle("Module 01 Real-World App 3: Thermodynamic & Reversible Computing Limits", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {
        "toffoli_bijective": len(toffoli_outputs) == 8,
        "fredkin_bijective": len(fredkin_outputs) == 8,
        "erased_bits_cmos_3bit": float(erased_cmos),
        "landauer_300k_1bit_j": landauer_erasure_energy(300.0, 1.0)["energy_joules"],
    }


# ==============================================================================
# MODULE 02 REAL-WORLD APPLICATIONS (3 DEMOS)
# ==============================================================================


def _simulate_brain_connectome(n_steps: int = 550, seed: int = 42) -> tuple[dict[str, np.ndarray], list[tuple[str, str]]]:
    """Simulate 4-region effective brain connectivity network (THAL -> PFC, PAR; PFC -> PAR -> MOT)."""
    rng = np.random.default_rng(seed)
    burn = 100
    tot = n_steps + burn
    thal, pfc, par, mot = [np.zeros(tot, dtype=np.float64) for _ in range(4)]
    for t in range(4, tot):
        thal[t] = 0.55 * thal[t - 1] + rng.normal(0.0, 0.45)
        pfc[t] = 0.35 * pfc[t - 1] + 0.75 * np.tanh(1.6 * thal[t - 1]) + rng.normal(0.0, 0.40)
        par[t] = 0.30 * par[t - 1] + 0.72 * np.tanh(1.6 * pfc[t - 2]) + 0.65 * np.tanh(1.6 * thal[t - 2]) + rng.normal(0.0, 0.38)
        mot[t] = 0.35 * mot[t - 1] + 0.80 * np.tanh(1.7 * par[t - 3]) + rng.normal(0.0, 0.38)
    signals = {"THAL": thal[burn:], "PFC": pfc[burn:], "PAR": par[burn:], "MOT": mot[burn:]}
    true_edges = [("THAL", "PFC"), ("THAL", "PAR"), ("PFC", "PAR"), ("PAR", "MOT")]
    return signals, true_edges


def demo_m02_brain_connectivity(n_steps: int = 550, n_surrogates: int = 50, seed: int = 42) -> tuple[plt.Figure, dict[str, Any]]:
    """Demo 2.1: 4-region effective brain connectivity recovery using Conditional TE + BH-FDR."""
    apply_publication_style()
    signals, true_edges = _simulate_brain_connectome(n_steps=n_steps, seed=seed)
    net = infer_directed_network(signals, lags=(1, 2, 3, 4), k=5, n_surrogates=n_surrogates, alpha=0.05, seed=seed)
    names = net["nodes"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))
    fig.patch.set_facecolor(PALETTE["bg"])

    # Panel 1: Sample LFP/Neural Time Series
    ax0 = style_axes(axes[0], title="1. Multi-Region Neural Activity (First 120 ms)", xlabel="Time $t$ (ms)", ylabel="Normalized Signal")
    for idx, name in enumerate(names):
        ax0.plot(signals[name][:120] + idx * 3.0, lw=1.4, label=name)
    ax0.set_yticks([i * 3.0 for i in range(4)])
    ax0.set_yticklabels(names, fontweight="bold")

    # Panel 2: Bivariate vs Conditional TE Heatmap
    im = axes[1].imshow(net["conditional_te"], cmap="Blues", vmin=0.0)
    axes[1].set_xticks(range(4))
    axes[1].set_yticks(range(4))
    axes[1].set_xticklabels(names, fontweight="bold")
    axes[1].set_yticklabels(names, fontweight="bold")
    axes[1].set_xlabel("Target Region")
    axes[1].set_ylabel("Source Region")
    axes[1].set_title("2. Conditional TE Matrix (Bits) + BH-FDR (*)", fontsize=10.8, fontweight="bold")
    for i in range(4):
        for j in range(4):
            if i != j:
                sig_star = " *" if net["cond_fdr_mask"][i, j] else ""
                val = net["conditional_te"][i, j]
                col = "white" if val > 0.55 * np.max(net["conditional_te"]) else PALETTE["ink"]
                axes[1].text(j, i, f"{val:.2f}{sig_star}", ha="center", va="center", fontsize=8.8, fontweight="bold", color=col)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Reconstructed Directed Brain Connectome
    ax2 = axes[2]
    ax2.set_facecolor(PALETTE["bg"])
    ax2.axis("off")
    pos = {"THAL": (0.0, 0.9), "PFC": (-0.95, 0.15), "PAR": (0.15, -0.45), "MOT": (1.05, 0.25)}
    for name, (xp, yp) in pos.items():
        ax2.add_patch(plt.Circle((xp, yp), 0.25, facecolor="#EFF6FF", edgecolor=PALETTE["cobalt"], lw=2.0, zorder=4))
        ax2.text(xp, yp, name, ha="center", va="center", fontsize=10, fontweight="bold", zorder=5)
    recovered_edges = []
    for rec in net["edges"]:
        s, t = rec["source"], rec["target"]
        if rec["cond_significant_fdr"]:
            recovered_edges.append((s, t))
            ax2.annotate(
                "",
                xy=pos[t],
                xytext=pos[s],
                arrowprops=dict(arrowstyle="-|>,head_width=0.38,head_length=0.55", color=PALETTE["teal"], lw=2.4, shrinkA=22, shrinkB=22, connectionstyle="arc3,rad=0.1"),
                zorder=3,
            )
            mx, my = 0.5 * (pos[s][0] + pos[t][0]), 0.5 * (pos[s][1] + pos[t][1])
            ax2.text(mx, my + 0.08, f"u={rec['best_lag']}ms", fontsize=8.2, fontweight="bold", color=PALETTE["teal"], ha="center",
                     bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor=PALETTE["teal"], alpha=0.85), zorder=6)
        elif rec["biv_significant_fdr"]:
            ax2.annotate(
                "",
                xy=pos[t],
                xytext=pos[s],
                arrowprops=dict(arrowstyle="-|>,head_width=0.3,head_length=0.4", color=PALETTE["crimson"], lw=1.5, ls=":", shrinkA=22, shrinkB=22, connectionstyle="arc3,rad=-0.15"),
                zorder=2,
            )
    ax2.set_xlim(-1.4, 1.45)
    ax2.set_ylim(-0.9, 1.3)
    ax2.set_title("3. Recovered Connectome (Teal=Cond FDR, Red=Filtered)", fontsize=10.5, fontweight="bold")

    fig.suptitle("Module 02 Real-World App 1: Computational Neuroscience Effective Brain Connectivity", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {"true_edges": true_edges, "recovered_edges": recovered_edges, "network": net}


def demo_m02_finance_contagion(n_days: int = 550, n_surrogates: int = 50, seed: int = 42) -> tuple[plt.Figure, dict[str, Any]]:
    """Demo 2.2: Multi-asset nonlinear volatility contagion & systemic hub inference."""
    apply_publication_style()
    rng = np.random.default_rng(seed)
    burn = 80
    tot = n_days + burn
    bank, tech, energy, crypto, bond = [np.zeros(tot, dtype=np.float64) for _ in range(5)]

    for t in range(3, tot):
        crisis_amp = 2.1 if (280 <= t <= 420) else 1.0
        bank[t] = 0.20 * bank[t - 1] + crisis_amp * rng.normal(0.0, 0.55)
        shock_lag1 = np.tanh(1.6 * bank[t - 1]) + 0.45 * (bank[t - 1] ** 2 - 0.3)
        tech[t] = 0.25 * tech[t - 1] + 0.65 * shock_lag1 + rng.normal(0.0, 0.42)
        energy[t] = 0.25 * energy[t - 1] + 0.60 * np.tanh(1.5 * bank[t - 2]) + rng.normal(0.0, 0.42)
        crypto[t] = 0.20 * crypto[t - 1] + 0.70 * np.tanh(1.6 * tech[t - 1]) + rng.normal(0.0, 0.48)
        bond[t] = 0.35 * bond[t - 1] + rng.normal(0.0, 0.40)  # Safe-haven decoupled asset

    signals = {
        "BANK": bank[burn:],
        "TECH": tech[burn:],
        "ENERGY": energy[burn:],
        "CRYPTO": crypto[burn:],
        "BOND": bond[burn:],
    }
    net = infer_directed_network(signals, lags=(1, 2, 3), k=5, n_surrogates=n_surrogates, alpha=0.05, seed=seed)
    names = net["nodes"]
    out_strength = np.sum(net["conditional_te"] * net["cond_fdr_mask"], axis=1)
    in_strength = np.sum(net["conditional_te"] * net["cond_fdr_mask"], axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax0 = style_axes(axes[0], title="1. Asset Log-Returns & Crisis Volatility Regime", xlabel="Trading Day", ylabel="Log-Return (arb.)")
    ax0.axvspan(200, 340, color=PALETTE["crimson"], alpha=0.10, label="Crisis Shock Window")
    ax0.plot(signals["BANK"], color=PALETTE["crimson"], lw=1.1, alpha=0.85, label="BANK (Hub)")
    ax0.plot(signals["TECH"] + 3.5, color=PALETTE["cobalt"], lw=1.0, alpha=0.85, label="TECH (+3.5)")
    ax0.plot(signals["BOND"] - 3.5, color=PALETTE["teal"], lw=1.0, alpha=0.85, label="BOND (-3.5)")
    ax0.legend(loc="upper right", fontsize=8.0)

    im = axes[1].imshow(net["conditional_te"], cmap="OrRd", vmin=0.0)
    axes[1].set_xticks(range(5))
    axes[1].set_yticks(range(5))
    axes[1].set_xticklabels(names, rotation=25, fontweight="bold")
    axes[1].set_yticklabels(names, fontweight="bold")
    axes[1].set_title("2. Asymmetric Lead-Lag TE Matrix (* = BH-FDR)", fontsize=10.5, fontweight="bold")
    for i in range(5):
        for j in range(5):
            if i != j:
                star = "*" if net["cond_fdr_mask"][i, j] else ""
                v = net["conditional_te"][i, j]
                axes[1].text(j, i, f"{v:.2f}{star}", ha="center", va="center", fontsize=8.0, fontweight="bold",
                             color="white" if v > 0.55 * np.max(net["conditional_te"]) else PALETTE["ink"])
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    ax2 = style_axes(axes[2], title="3. Systemic Contagion Hub Profile (Out vs. In TE)", ylabel="Total Significant TE (Bits)", grid_axis="y")
    xp = np.arange(5)
    ax2.bar(xp - 0.17, out_strength, width=0.34, color=PALETTE["crimson"], edgecolor=PALETTE["ink"], label="Out-Flow (Risk Sender)")
    ax2.bar(xp + 0.17, in_strength, width=0.34, color=PALETTE["cobalt"], edgecolor=PALETTE["ink"], label="In-Flow (Risk Receiver)")
    ax2.set_xticks(xp)
    ax2.set_xticklabels(names, rotation=20, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8.3)

    fig.suptitle("Module 02 Real-World App 2: Quantitative Finance Asymmetric Contagion Network", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {"out_strength": dict(zip(names, out_strength.tolist())), "network": net}


def demo_m02_gene_regulatory_network(n_steps: int = 550, n_surrogates: int = 50, seed: int = 42) -> tuple[plt.Figure, dict[str, Any]]:
    """Demo 2.3: Nonlinear Hill-function gene regulatory cascade (G1->G2->G3, G1-|G4) inference."""
    apply_publication_style()
    rng = np.random.default_rng(seed)
    burn = 80
    tot = n_steps + burn
    g1, g2, g3, g4, g5 = [np.ones(tot, dtype=np.float64) * 0.5 for _ in range(5)]

    def hill_act(x: float, k_half: float = 0.5, n_hill: float = 4.0) -> float:
        xc = max(x, 0.01)
        return float((xc**n_hill) / (k_half**n_hill + xc**n_hill))

    for t in range(3, tot):
        g1[t] = 0.55 * g1[t - 1] + 0.45 * rng.uniform(0.0, 1.0) + rng.normal(0.0, 0.12)
        g2[t] = 0.35 * g2[t - 1] + 0.85 * hill_act(g1[t - 2]) + rng.normal(0.0, 0.10)
        g3[t] = 0.35 * g3[t - 1] + 0.85 * hill_act(g2[t - 2]) + rng.normal(0.0, 0.10)
        g4[t] = 0.35 * g4[t - 1] + 0.85 * (1.0 - hill_act(g1[t - 2])) + rng.normal(0.0, 0.10)
        g5[t] = 0.55 * g5[t - 1] + rng.normal(0.5, 0.18)

    signals = {"G1": g1[burn:], "G2": g2[burn:], "G3": g3[burn:], "G4": g4[burn:], "G5": g5[burn:]}
    net = infer_directed_network(signals, lags=(1, 2, 3, 4), k=5, n_surrogates=n_surrogates, alpha=0.05, seed=seed)
    names = net["nodes"]
    true_edges = {("G1", "G2"), ("G2", "G3"), ("G1", "G4")}
    recovered = {(r["source"], r["target"]) for r in net["edges"] if r["cond_significant_fdr"]}

    # Naive undirected Pearson correlation matrix for comparison
    corr_mat = np.abs(np.corrcoef([signals[n] for n in names]))
    np.fill_diagonal(corr_mat, 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))
    fig.patch.set_facecolor(PALETTE["bg"])

    im0 = axes[0].imshow(corr_mat, cmap="Greys", vmin=0, vmax=1)
    axes[0].set_xticks(range(5))
    axes[0].set_yticks(range(5))
    axes[0].set_xticklabels(names, fontweight="bold")
    axes[0].set_yticklabels(names, fontweight="bold")
    axes[0].set_title("1. Undirected $|r|$ (Fails Direction & Indirects)", fontsize=10.5, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(net["conditional_te"], cmap="YlGnBu", vmin=0.0)
    axes[1].set_xticks(range(5))
    axes[1].set_yticks(range(5))
    axes[1].set_xticklabels(names, fontweight="bold")
    axes[1].set_yticklabels(names, fontweight="bold")
    axes[1].set_title("2. Conditional TE Matrix (* = BH-FDR)", fontsize=10.5, fontweight="bold")
    for i in range(5):
        for j in range(5):
            if i != j:
                star = "*" if net["cond_fdr_mask"][i, j] else ""
                v = net["conditional_te"][i, j]
                axes[1].text(j, i, f"{v:.2f}{star}", ha="center", va="center", fontsize=8.0, fontweight="bold",
                             color="white" if v > 0.55 * np.max(net["conditional_te"]) else PALETTE["ink"])
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    ax2 = axes[2]
    ax2.set_facecolor(PALETTE["bg"])
    ax2.axis("off")
    pos = {"G1": (0.0, 0.8), "G2": (-0.65, 0.05), "G3": (-0.65, -0.75), "G4": (0.65, 0.05), "G5": (0.65, -0.75)}
    for n, (xp, yp) in pos.items():
        ax2.add_patch(plt.Circle((xp, yp), 0.20, facecolor="#ECFDF5", edgecolor=PALETTE["teal"], lw=2.0, zorder=4))
        ax2.text(xp, yp, n, ha="center", va="center", fontsize=10.5, fontweight="bold", zorder=5)
    for rec in net["edges"]:
        s, t = rec["source"], rec["target"]
        if rec["cond_significant_fdr"]:
            is_tp = (s, t) in true_edges
            col = PALETTE["teal"] if is_tp else PALETTE["amber"]
            ax2.annotate("", xy=pos[t], xytext=pos[s],
                         arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.5", color=col, lw=2.4, shrinkA=18, shrinkB=18), zorder=3)
            mx, my = 0.5 * (pos[s][0] + pos[t][0]), 0.5 * (pos[s][1] + pos[t][1])
            ax2.text(mx + 0.12, my, f"u={rec['best_lag']}", fontsize=8.2, fontweight="bold", color=col)
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.05, 1.1)
    ax2.set_title("3. Reconstructed GRN (Hill Cascade + Lag $\\tau=2$)", fontsize=10.5, fontweight="bold")

    fig.suptitle("Module 02 Real-World App 3: Systems Biology Hill-Kinetics Gene Network Reconstruction", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {"true_edges": sorted(true_edges), "recovered_edges": sorted(recovered), "network": net}


# ==============================================================================
# MODULE 03 REAL-WORLD APPLICATIONS (3 DEMOS)
# ==============================================================================


def simulate_excitable_bz_media(size: int = 84, steps: int = 80, n_refractory: int = 6) -> list[np.ndarray]:
    """Simulate 2D Greenberg-Hastings excitable medium generating spiral waves."""
    grid = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    # Broken planar wavefront: excited bar + refractory tail on one side creates free spiral tip
    grid[mid - 1 : mid + 1, 8 : mid + 6] = 1
    grid[mid + 1 : mid + 4, 8 : mid + 6] = 2

    frames = [grid.copy()]
    for _ in range(steps):
        excited = (grid == 1).astype(np.uint8)
        nb_count = (
            np.roll(excited, 1, 0) + np.roll(excited, -1, 0)
            + np.roll(excited, 1, 1) + np.roll(excited, -1, 1)
            + np.roll(np.roll(excited, 1, 0), 1, 1) + np.roll(np.roll(excited, 1, 0), -1, 1)
            + np.roll(np.roll(excited, -1, 0), 1, 1) + np.roll(np.roll(excited, -1, 0), -1, 1)
        )
        next_g = np.zeros_like(grid)
        # Resting (0) becomes excited (1) if >= 2 excited Moore neighbors
        next_g[(grid == 0) & (nb_count >= 2)] = 1
        # Excited (1) -> Refractory (2), Refractory k -> k+1 mod (n_refractory+1)
        ref_mask = grid >= 1
        next_g[ref_mask] = (grid[ref_mask] + 1) % (n_refractory + 1)
        grid = next_g
        frames.append(grid.copy())
    return frames


def demo_m03_excitable_bz_spiral(size: int = 84, steps: int = 80) -> tuple[plt.Figure, dict[str, Any]]:
    """Demo 3.1: Belousov-Zhabotinsky chemical waves & cardiac spiral reentry arrhythmias."""
    apply_publication_style()
    frames = simulate_excitable_bz_media(size=size, steps=steps, n_refractory=6)
    excited_frac = [float(np.mean(f == 1)) for f in frames]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7))
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax, step_idx, title in [
        (axes[0], 12, "1. Broken Wavefront Curling ($t=12$)"),
        (axes[1], 55, "2. Mature Spiral Reentry Rotor ($t=55$)"),
    ]:
        ax.imshow(frames[step_idx], cmap="plasma", vmin=0, vmax=6, interpolation="nearest")
        ax.set_title(title, fontsize=10.8, fontweight="bold")
        ax.set_xlabel("Space $x$")
        ax.set_ylabel("Space $y$")

    ax2 = style_axes(axes[2], title="3. Excited Cell Fraction (Spiral Periodicity)", xlabel="Time Step $t$", ylabel="Excited Fraction")
    ax2.plot(range(len(excited_frac)), excited_frac, color=PALETTE["crimson"], lw=2.2)
    ax2.fill_between(range(len(excited_frac)), 0, excited_frac, color=PALETTE["crimson"], alpha=0.14)

    fig.suptitle("Module 03 Real-World App 1: 2D Excitable Media (BZ Chemical Waves & Cardiac Arrhythmia)", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {"peak_excited_fraction": float(max(excited_frac)), "final_excited_fraction": float(excited_frac[-1]), "frames": frames}


def _simulate_nagel_schreckenberg(
    length: int = 160,
    density: float = 0.28,
    v_max: int = 5,
    p_brake: float = 0.25,
    steps: int = 140,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Simulate 1D ring-road Nagel-Schreckenberg traffic (Rule 184 when v_max=1, p_brake=0)."""
    rng = np.random.default_rng(seed)
    n_cars = max(1, int(round(density * length)))
    pos = np.sort(rng.choice(length, size=n_cars, replace=False))
    vel = np.minimum(v_max, np.ones(n_cars, dtype=int))

    spacetime = np.full((steps, length), -1, dtype=np.int16)
    flux_sum = 0.0
    for t in range(steps):
        spacetime[t, pos] = vel
        gaps = (np.roll(pos, -1) - pos - 1) % length
        # 1. Acceleration
        vel = np.minimum(vel + 1, v_max)
        # 2. Braking to avoid crash
        vel = np.minimum(vel, gaps)
        # 3. Stochastic randomization
        if p_brake > 0.0:
            brake_mask = (rng.random(n_cars) < p_brake) & (vel > 0)
            vel[brake_mask] -= 1
        # 4. Advance cars
        pos = (pos + vel) % length
        if t >= steps // 3:
            flux_sum += float(np.sum(vel)) / float(length)
    mean_flux = flux_sum / float(steps - steps // 3)
    return spacetime, mean_flux


def demo_m03_traffic_flow_rule184(seed: int = 42) -> tuple[plt.Figure, dict[str, Any]]:
    """Demo 3.2: Rule 184 & Nagel-Schreckenberg phantom traffic jams & fundamental diagram."""
    apply_publication_style()
    densities = np.linspace(0.05, 0.95, 19)
    flux_r184 = [_simulate_nagel_schreckenberg(120, float(d), v_max=1, p_brake=0.0, steps=90, seed=seed)[1] for d in densities]
    flux_ns = [_simulate_nagel_schreckenberg(120, float(d), v_max=5, p_brake=0.28, steps=110, seed=seed)[1] for d in densities]

    st_jam, _ = _simulate_nagel_schreckenberg(length=150, density=0.30, v_max=5, p_brake=0.25, steps=130, seed=seed)
    occ_jam = (st_jam >= 0).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0))
    fig.patch.set_facecolor(PALETTE["bg"])

    axes[0].imshow(occ_jam, cmap="binary", interpolation="nearest", aspect="auto")
    axes[0].set_title("1. Nagel-Schreckenberg ($\\rho=0.30, v_{\\max}=5, p=0.25$)\nBackward-Propagating Phantom Jams", fontsize=10.8, fontweight="bold")
    axes[0].set_xlabel("Ring Road Site $i$ (Flow $\\to$ Right)")
    axes[0].set_ylabel("Time Step $t$ (Top $\\to$ Bottom)")

    ax1 = style_axes(axes[1], title="2. Fundamental Traffic Diagram: Flux $J(\\rho)$", xlabel="Vehicle Density $\\rho$", ylabel="Mean Traffic Flux $J$ (veh/step)")
    ax1.plot(densities, flux_r184, marker="o", lw=2.3, color=PALETTE["cobalt"], label="Rule 184 ($v_{\\max}=1, p=0$)")
    ax1.plot(densities, flux_ns, marker="s", lw=2.3, color=PALETTE["crimson"], label="Nagel-Schreckenberg ($v_{\\max}=5, p=0.28$)")
    ax1.legend(loc="upper right", fontsize=8.6)

    fig.suptitle("Module 03 Real-World App 2: Vehicular Traffic Phase Transitions & Phantom Jams", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {
        "densities": densities.tolist(),
        "flux_rule184": flux_r184,
        "flux_nagel_schreckenberg": flux_ns,
        "max_flux_r184": float(max(flux_r184)),
        "max_flux_ns": float(max(flux_ns)),
    }


def _run_glider_xor_gate(a_bit: int, b_bit: int, width: int = 88, steps: int = 56) -> tuple[np.ndarray, np.ndarray, int]:
    """Execute decentralized Rule 54 glider-collision XOR parity gate (A XOR B -> Output Bit)."""
    row = np.tile(np.array([1, 1, 1, 0], dtype=np.uint8), width // 4)
    # Inject left input glider gamma+ if A=1, right input glider gamma- if B=1
    if a_bit == 1:
        row[16:20] = [0, 0, 1, 0]  # gamma+ moving right
    if b_bit == 1:
        row[64:68] = [1, 0, 0, 0]  # gamma- moving left
    grid = evolve_eca_from_state(54, row, steps=steps)
    fst = HansonCrutchfieldFST()
    mask, _ = fst.filter(grid)
    # Readout: at t >= 30, if A^B == 1 one surviving glider remains; if A==B (0,0 or 1,1 annihilation) zero particles remain!
    out_bit = 1 if int(np.sum(mask[-8:])) > 0 else 0
    return grid, mask, out_bit


def demo_m03_neuromorphic_glider_logic() -> tuple[plt.Figure, dict[str, Any]]:
    """Demo 3.3: Clock-driven neuromorphic glider collision circuit computing A XOR B without CPU/RAM."""
    apply_publication_style()
    cases = [(0, 0), (0, 1), (1, 0), (1, 1)]
    results = {}
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.3))
    fig.patch.set_facecolor(PALETTE["bg"])

    for ax, (a, b) in zip(axes, cases):
        _, mask, out_bit = _run_glider_xor_gate(a, b)
        expected = a ^ b
        results[f"({a},{b})"] = {"output": out_bit, "expected": expected, "correct": out_bit == expected}
        ax.imshow(mask, cmap="magma", interpolation="nearest", aspect="auto")
        if a == 1 and b == 1:
            ax.scatter([41], [25], s=140, facecolors="none", edgecolors="#38BDF8", lw=2.0, label="Annihilation")
            ax.legend(loc="lower right", fontsize=7.8)
        ax.set_title(f"In=({a},{b}) $\\to$ Out={out_bit} (XOR={expected})", fontsize=10.2, fontweight="bold",
                     color=PALETTE["teal"] if out_bit == expected else PALETTE["crimson"])
        ax.set_xlabel("Bus Site $i$")
        ax.set_ylabel("Clock $t$")

    fig.suptitle("Module 03 Real-World App 3: Decentralized Neuromorphic Glider Parity Logic (Rule 54 Bus)", fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, results


# ==============================================================================
# MODULE 04 REAL-WORLD APPLICATIONS (3 DEMOS)
# ==============================================================================


class NKLandscape:
    """Kauffman NK rugged fitness landscape for protein sequence directed evolution."""

    def __init__(self, n: int = 20, k: int = 6, seed: int = 42) -> None:
        self.n = n
        self.k = k
        rng = np.random.default_rng(seed)
        self.epistasis_partners = [
            rng.choice([j for j in range(n) if j != i], size=k, replace=False) if k > 0 else np.array([], dtype=int)
            for i in range(n)
        ]
        self.tables = rng.random((n, 2 ** (k + 1)))

    def fitness(self, seq: np.ndarray) -> float:
        total = 0.0
        for i in range(self.n):
            idx = int(seq[i])
            for p in self.epistasis_partners[i]:
                idx = (idx << 1) | int(seq[p])
            total += float(self.tables[i, idx])
        return total / float(self.n)


def demo_m04_protein_directed_evolution(seed: int = 42) -> tuple[plt.Figure, dict[str, Any]]:
    """Demo 4.1: Directed evolution of protein sequences on smooth (K=0) vs rugged (K=6) NK landscapes."""
    apply_publication_style()
    n_res, gens = 20, 35
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9))
    fig.patch.set_facecolor(PALETTE["bg"])

    for ax, k_epi, title in [
        (axes[0], 0, "1. Smooth Binding Landscape ($N=20, K=0$)"),
        (axes[1], 6, "2. Rugged Epistatic Landscape ($N=20, K=6$)"),
    ]:
        landscape = NKLandscape(n=n_res, k=k_epi, seed=seed)
        rng = np.random.default_rng(seed + k_epi)

        # Greedy single-mutant hill climber
        seq_hc = rng.integers(0, 2, size=n_res, dtype=np.uint8)
        hc_hist = [landscape.fitness(seq_hc)]
        for _ in range(gens - 1):
            cur_f = hc_hist[-1]
            best_cand, best_f = seq_hc, cur_f
            for pos in range(n_res):
                cand = seq_hc.copy()
                cand[pos] ^= 1
                f_c = landscape.fitness(cand)
                if f_c > best_f:
                    best_f, best_cand = f_c, cand
            seq_hc = best_cand
            hc_hist.append(best_f)

        # Genetic Algorithm (Pop=30, Crossover + Mutation + Elitism)
        pop = rng.integers(0, 2, size=(30, n_res), dtype=np.uint8)
        ga_best, ga_mean = [], []
        for _ in range(gens):
            fits = np.array([landscape.fitness(ind) for ind in pop])
            ga_best.append(float(np.max(fits)))
            ga_mean.append(float(np.mean(fits)))
            elites = np.argsort(fits)[-3:]
            nxt = [pop[i].copy() for i in elites]
            probs = (fits - np.min(fits) + 1e-3)
            probs /= np.sum(probs)
            while len(nxt) < 30:
                p1, p2 = pop[rng.choice(30, size=2, p=probs)]
                cut = int(rng.integers(1, n_res))
                c = np.concatenate([p1[:cut], p2[cut:]])
                c[rng.random(n_res) < 0.06] ^= 1
                nxt.append(c)
            pop = np.array(nxt, dtype=np.uint8)

        style_axes(ax, title=title, xlabel="Evolution Cycle", ylabel="Binding Affinity $F(s)$")
        ax.plot(range(1, gens + 1), ga_best, color=PALETTE["teal"], lw=2.5, marker="o", ms=4, label="GA Best Sequence")
        ax.plot(range(1, gens + 1), ga_mean, color=PALETTE["cobalt"], lw=1.8, linestyle="--", label="GA Population Mean")
        ax.plot(range(1, gens + 1), hc_hist, color=PALETTE["crimson"], lw=2.3, marker="s", ms=4, label="Greedy Hill Climber")
        ax.set_ylim(0.45, 0.90)
        ax.legend(loc="lower right", fontsize=8.4)
        results[f"K={k_epi}"] = {"ga_final_best": ga_best[-1], "hill_climber_final": hc_hist[-1]}

    fig.suptitle("Module 04 Real-World App 1: Computational Biology & In-Silico Protein Directed Evolution", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, results


def _eval_nas_chromosome(chrom: np.ndarray) -> tuple[float, float, float]:
    """Evaluate discrete neural architecture genome -> (val_accuracy, mflops, scalar_fitness)."""
    depth = int(chrom[0]) * 2 + int(chrom[1]) + 2  # 2..5 blocks
    widths = [16 * (2 ** int(chrom[2 + (i % 4)])) for i in range(depth)]  # 16 or 32 per block
    act_idx = int(chrom[6]) * 2 + int(chrom[7])  # 0:Tanh, 1:ReLU, 2:SiLU, 3:GELU
    skip_flags = [int(chrom[8 + (i % 4)]) for i in range(depth)]
    norm_flag = int(chrom[12])

    act_bonus = (0.0, 0.032, 0.048, 0.052)[act_idx]
    active_w = widths
    flops = sum(active_w[i] * (active_w[i - 1] if i > 0 else 16) for i in range(depth)) * 0.08
    flops += float(sum(skip_flags)) * 4.5

    # Non-differentiable architectural interaction: deep nets require skip connections & norm
    capacity = 0.72 + 0.045 * np.log2(max(np.mean(active_w), 8.0)) + 0.025 * min(depth, 4)
    vanishing_penalty = 0.08 * max(0, depth - 2 - int(sum(skip_flags)))
    norm_bonus = 0.025 if norm_flag == 1 else -0.015
    acc = float(np.clip(capacity + act_bonus + norm_bonus - vanishing_penalty, 0.50, 0.985))
    fitness = acc - 0.0012 * flops
    return acc, float(flops), float(fitness)


def demo_m04_automl_nas(generations: int = 24, seed: int = 7) -> tuple[plt.Figure, dict[str, Any]]:
    """Demo 4.2: Discrete GA Neural Architecture Search optimizing Accuracy vs. FLOPs Pareto front."""
    apply_publication_style()
    rng = np.random.default_rng(seed)
    pop_size, chrom_len = 28, 13
    pop = rng.integers(0, 2, size=(pop_size, chrom_len), dtype=np.uint8)

    gen0_pts = [_eval_nas_chromosome(ind) for ind in pop]
    best_fit_hist, best_acc_hist = [], []

    for _ in range(generations):
        evals = [_eval_nas_chromosome(ind) for ind in pop]
        fits = np.array([e[2] for e in evals])
        accs = np.array([e[0] for e in evals])
        best_fit_hist.append(float(np.max(fits)))
        best_acc_hist.append(float(accs[int(np.argmax(fits))]))

        elites = np.argsort(fits)[-4:]
        nxt = [pop[i].copy() for i in elites]
        probs = np.maximum(fits - np.min(fits) + 1e-3, 1e-4)
        probs /= np.sum(probs)
        while len(nxt) < pop_size:
            p1, p2 = pop[rng.choice(pop_size, size=2, p=probs)]
            cut = int(rng.integers(1, chrom_len))
            c = np.concatenate([p1[:cut], p2[cut:]])
            c[rng.random(chrom_len) < 0.08] ^= 1
            nxt.append(c)
        pop = np.array(nxt, dtype=np.uint8)

    gen_final_pts = [_eval_nas_chromosome(ind) for ind in pop]
    best_arch = max(gen_final_pts, key=lambda x: x[2])

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9))
    fig.patch.set_facecolor(PALETTE["bg"])

    ax0 = style_axes(axes[0], title="1. GA Architecture Search Progress", xlabel="Generation", ylabel="Score")
    ax0.plot(range(1, generations + 1), best_acc_hist, marker="o", lw=2.3, color=PALETTE["teal"], label="Best Validation Accuracy")
    ax0.plot(range(1, generations + 1), best_fit_hist, marker="s", lw=2.2, color=PALETTE["cobalt"], label="Accuracy - FLOPs Penalty")
    ax0.legend(loc="lower right", fontsize=8.5)

    ax1 = style_axes(axes[1], title="2. Accuracy vs. Compute Cost (Gen 1 vs. Evolved Gen 24)", xlabel="Compute Cost (MFLOPs)", ylabel="Validation Accuracy")
    ax1.scatter([p[1] for p in gen0_pts], [p[0] for p in gen0_pts], color=PALETTE["muted"], s=45, alpha=0.8, label="Gen 1 Random Architectures")
    ax1.scatter([p[1] for p in gen_final_pts], [p[0] for p in gen_final_pts], color=PALETTE["cobalt"], s=55, edgecolor=PALETTE["ink"], label="Gen 24 Evolved Population")
    ax1.scatter([best_arch[1]], [best_arch[0]], color=PALETTE["crimson"], s=130, marker="*", zorder=6, label=f"Best ({best_arch[0]:.1%}, {best_arch[1]:.1f} MFLOPs)")
    ax1.legend(loc="lower right", fontsize=8.3)

    fig.suptitle("Module 04 Real-World App 2: AutoML Discrete Neural Architecture Search (GA-NAS)", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {"best_accuracy": best_arch[0], "best_mflops": best_arch[1], "best_fitness": best_arch[2]}


def _simulate_robot_swarm(
    rule_16: np.ndarray,
    arena_size: int = 28,
    n_robots: int = 12,
    n_food: int = 36,
    steps: int = 55,
    seed: int = 7,
) -> tuple[float, list[dict[str, Any]]]:
    """Simulate decentralized 2D foraging robots controlled purely by a 16-entry sensor lookup table."""
    rng = np.random.default_rng(seed)
    food = np.zeros((arena_size, arena_size), dtype=bool)
    # Two dense food clusters + scattered tokens
    for _ in range(n_food):
        fx = int(rng.integers(2, arena_size - 2))
        fy = int(rng.integers(2, arena_size - 2))
        food[fx, fy] = True
    food[5:9, 5:9] = True
    food[arena_size - 9 : arena_size - 5, arena_size - 9 : arena_size - 5] = True

    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # E, S, W, N
    rx = rng.integers(3, arena_size - 3, size=n_robots)
    ry = rng.integers(3, arena_size - 3, size=n_robots)
    rd = rng.integers(0, 4, size=n_robots)

    collected = 0
    collisions = 0
    snapshots: list[dict[str, Any]] = []

    for _ in range(steps):
        snapshots.append({"rx": rx.copy(), "ry": ry.copy(), "rd": rd.copy(), "food": food.copy(), "collected": collected})
        for i in range(n_robots):
            dx, dy = dirs[int(rd[i])]
            lx, ly = dirs[(int(rd[i]) - 1) % 4]
            nx, ny = int(rx[i]) + dx, int(ry[i]) + dy
            wall_ahead = int(nx <= 0 or nx >= arena_size - 1 or ny <= 0 or ny >= arena_size - 1)
            food_ahead = int((not wall_ahead) and food[nx, ny])
            # Sense food in 3-cell forward/left quadrant
            food_near = 0
            for sx in range(-2, 3):
                for sy in range(-2, 3):
                    tx, ty = int(rx[i]) + sx, int(ry[i]) + sy
                    if 0 <= tx < arena_size and 0 <= ty < arena_size and food[tx, ty]:
                        food_near = 1
                        break
                if food_near:
                    break
            fl_x, fl_y = int(rx[i]) + lx, int(ry[i]) + ly
            food_left = int(0 <= fl_x < arena_size and 0 <= fl_y < arena_size and food[fl_x, fl_y])

            sensor_idx = (wall_ahead << 3) | (food_ahead << 2) | (food_left << 1) | food_near
            action = int(rule_16[sensor_idx])  # 0: fwd, 1: turn_left, 2: turn_right, 3: step_and_sweep
            if action == 1:
                rd[i] = (rd[i] - 1) % 4
            elif action == 2:
                rd[i] = (rd[i] + 1) % 4
            else:
                if wall_ahead:
                    collisions += 1
                    rd[i] = (rd[i] + (1 if action == 3 else 2)) % 4
                else:
                    rx[i], ry[i] = nx, ny
                    if food[rx[i], ry[i]]:
                        food[rx[i], ry[i]] = False
                        collected += 1
                    if action == 3:
                        rd[i] = (rd[i] + 1) % 4

    fitness = float(collected) - 0.12 * float(collisions)
    return fitness, snapshots


def demo_m04_swarm_robotics_foraging(generations: int = 20, seed: int = 7) -> tuple[plt.Figure, dict[str, Any]]:
    """Demo 4.3: Evolving decentralized 2D foraging robot lookup rules via GA."""
    apply_publication_style()
    rng = np.random.default_rng(seed)
    pop_size = 22
    pop = rng.integers(0, 4, size=(pop_size, 16), dtype=np.uint8)

    gen0_best_rule = pop[0].copy()
    best_hist, mean_hist = [], []
    best_rule_final = pop[0].copy()

    for g in range(generations):
        fits = np.array([_simulate_robot_swarm(ind, steps=45, seed=7 + g % 3)[0] for ind in pop], dtype=np.float64)
        best_idx = int(np.argmax(fits))
        best_hist.append(float(fits[best_idx]))
        mean_hist.append(float(np.mean(fits)))
        best_rule_final = pop[best_idx].copy()
        if g == 0:
            gen0_best_rule = pop[int(np.argmin(fits))].copy()

        elites = np.argsort(fits)[-3:]
        nxt = [pop[i].copy() for i in elites]
        probs = np.maximum(fits - np.min(fits) + 1.0, 0.1)
        probs /= np.sum(probs)
        while len(nxt) < pop_size:
            p1, p2 = pop[rng.choice(pop_size, size=2, p=probs)]
            cut = int(rng.integers(1, 16))
            c = np.concatenate([p1[:cut], p2[cut:]])
            mut = rng.random(16) < 0.12
            c[mut] = rng.integers(0, 4, size=int(np.sum(mut)), dtype=np.uint8)
            nxt.append(c)
        pop = np.array(nxt, dtype=np.uint8)

    fit0, snaps0 = _simulate_robot_swarm(gen0_best_rule, steps=55, seed=7)
    fit_ev, snaps_ev = _simulate_robot_swarm(best_rule_final, steps=55, seed=7)

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.9))
    fig.patch.set_facecolor(PALETTE["bg"])

    for ax, snaps, title, col in [
        (axes[0], snaps0, f"1. Gen 1 Random Rule (Collected={snaps0[-1]['collected']})", PALETTE["crimson"]),
        (axes[1], snaps_ev, f"2. Evolved Swarm Rule (Collected={snaps_ev[-1]['collected']})", PALETTE["teal"]),
    ]:
        s = snaps[-1]
        fy, fx = np.where(s["food"])
        ax.scatter(fx, fy, marker="s", s=34, color=PALETTE["amber"], label="Remaining Food")
        ax.scatter(s["ry"], s["rx"], marker="o", s=75, color=col, edgecolor=PALETTE["ink"], label="Robots")
        ax.set_xlim(0, 27)
        ax.set_ylim(0, 27)
        ax.set_aspect("equal")
        style_axes(ax, title=title, xlabel="Arena $x$", ylabel="Arena $y$")
        ax.legend(loc="upper right", fontsize=7.8)

    ax2 = style_axes(axes[2], title="3. Decentralized Swarm Fitness Evolution", xlabel="Generation", ylabel="Foraging Fitness Score")
    ax2.plot(range(1, generations + 1), best_hist, marker="o", lw=2.4, color=PALETTE["teal"], label="Best Rule Fitness")
    ax2.plot(range(1, generations + 1), mean_hist, marker="s", lw=2.0, linestyle="--", color=PALETTE["cobalt"], label="Swarm Pop Mean")
    ax2.legend(loc="lower right", fontsize=8.4)

    fig.suptitle("Module 04 Real-World App 3: Evolutionary Swarm Robotics (Decentralized Foraging)", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {
        "gen1_collected": snaps0[-1]["collected"],
        "evolved_collected": snaps_ev[-1]["collected"],
        "gen1_fitness": fit0,
        "evolved_fitness": fit_ev,
        "snaps_gen1": snaps0,
        "snaps_evolved": snaps_ev,
    }


def generate_real_world_gifs(outputs_root: str | Path) -> list[Path]:
    """Generate classroom-ready animated GIFs for selected real-world demos across Modules 02-04."""
    apply_publication_style(dpi=110)
    root = Path(outputs_root)
    generated: list[Path] = []

    # 1. Module 02 GIF: Brain Connectome Neural Time-Series & Network Discovery
    signals, _ = _simulate_brain_connectome(n_steps=480, seed=42)
    net_brain = infer_directed_network(signals, lags=(1, 2, 3, 4), k=5, n_surrogates=40, alpha=0.05, seed=42)
    names = ["THAL", "PFC", "PAR", "MOT"]
    pos = {"THAL": (0.0, 0.9), "PFC": (-0.95, 0.15), "PAR": (0.15, -0.45), "MOT": (1.05, 0.25)}
    frames_brain = []
    for t_win in range(30, 181, 10):
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
        fig.patch.set_facecolor(PALETTE["bg"])
        ax0 = style_axes(axes[0], title=f"4-Region Neural Oscillations ($t = 0..{t_win}$ ms)", xlabel="Time (ms)", ylabel="Region")
        for idx, nm in enumerate(names):
            ax0.plot(np.arange(t_win), signals[nm][:t_win] + idx * 3.0, lw=1.6, label=nm)
        ax0.set_xlim(0, 185)
        ax0.set_yticks([i * 3.0 for i in range(4)])
        ax0.set_yticklabels(names, fontweight="bold")

        ax1 = axes[1]
        ax1.set_facecolor(PALETTE["bg"])
        ax1.axis("off")
        for nm, (xp, yp) in pos.items():
            ax1.add_patch(plt.Circle((xp, yp), 0.24, facecolor="#EFF6FF", edgecolor=PALETTE["cobalt"], lw=2.0, zorder=4))
            ax1.text(xp, yp, nm, ha="center", va="center", fontsize=10, fontweight="bold", zorder=5)
        for rec in net_brain["edges"]:
            if rec["cond_significant_fdr"]:
                s, t = rec["source"], rec["target"]
                ax1.annotate("", xy=pos[t], xytext=pos[s],
                             arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.5", color=PALETTE["teal"], lw=2.3, shrinkA=21, shrinkB=21), zorder=3)
        ax1.set_xlim(-1.4, 1.45)
        ax1.set_ylim(-0.9, 1.3)
        ax1.set_title("Recovered Conditional TE + BH-FDR Connectome", fontsize=10.8, fontweight="bold")
        fig.suptitle("Module 02 Real-World Demo: Effective Brain Connectivity Discovery", fontsize=12, fontweight="bold")
        fig.tight_layout()
        frames_brain.append(fig_to_pil_image(fig, dpi=100))
        plt.close(fig)
    generated.append(save_gif_from_frames(frames_brain, root / "02_Transfer_Entropy_and_Network_Inference" / "animations" / "02_brain_connectome_neural_dynamics.gif", duration_ms=160))

    # 2. Module 03 GIF A: 2D Belousov-Zhabotinsky Chemical Spiral Waves
    bz_frames = simulate_excitable_bz_media(size=84, steps=76, n_refractory=6)
    frames_bz = []
    for t_idx in range(0, len(bz_frames), 2):
        fig, ax = plt.subplots(figsize=(5.4, 5.2))
        fig.patch.set_facecolor(PALETTE["bg"])
        ax.imshow(bz_frames[t_idx], cmap="plasma", vmin=0, vmax=6, interpolation="nearest")
        ax.set_title(f"2D Excitable BZ Spiral / Reentry Rotor (Step $t={t_idx}$)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Space $x$")
        ax.set_ylabel("Space $y$")
        fig.tight_layout()
        frames_bz.append(fig_to_pil_image(fig, dpi=105))
        plt.close(fig)
    generated.append(save_gif_from_frames(frames_bz, root / "03_Cellular_Automata_and_Particle_Computation" / "animations" / "03_belousov_zhabotinsky_spiral_waves.gif", duration_ms=100))

    # 3. Module 03 GIF B: Ring-Road Traffic Jams (Nagel-Schreckenberg)
    st_tr, _ = _simulate_nagel_schreckenberg(length=120, density=0.30, v_max=5, p_brake=0.25, steps=96, seed=42)
    thetas = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    frames_tr = []
    for t_idx in range(0, 96, 2):
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
        fig.patch.set_facecolor(PALETTE["bg"])
        # Left: Circular Ring Road
        ax0 = axes[0]
        ax0.set_facecolor(PALETTE["bg"])
        ring_x, ring_y = np.cos(thetas), np.sin(thetas)
        ax0.plot(ring_x, ring_y, color=PALETTE["grid"], lw=8, zorder=1)
        cars = np.where(st_tr[t_idx] >= 0)[0]
        speeds = st_tr[t_idx, cars]
        ax0.scatter(ring_x[cars], ring_y[cars], c=speeds, cmap="RdYlGn", vmin=0, vmax=5, s=48, edgecolor=PALETTE["ink"], zorder=3)
        ax0.set_xlim(-1.25, 1.25)
        ax0.set_ylim(-1.25, 1.25)
        ax0.set_aspect("equal")
        ax0.axis("off")
        ax0.set_title(f"Ring Road Cars (Green=Fast, Red=Jam) | $t={t_idx}$", fontsize=10.5, fontweight="bold")

        # Right: Progressive Spacetime Diagram
        view = np.zeros_like(st_tr, dtype=np.uint8)
        view[: t_idx + 1] = (st_tr[: t_idx + 1] >= 0).astype(np.uint8)
        axes[1].imshow(view, cmap="binary", interpolation="nearest", aspect="auto")
        axes[1].axhline(t_idx + 0.5, color=PALETTE["crimson"], lw=1.4)
        axes[1].set_title("Spacetime (Phantom Jams Travel Backward!)", fontsize=10.5, fontweight="bold")
        axes[1].set_xlabel("Road Site $i$")
        axes[1].set_ylabel("Time Step $t$")
        fig.suptitle("Module 03 Real-World Demo: Nagel-Schreckenberg Traffic Flow & Phantom Jams", fontsize=12, fontweight="bold")
        fig.tight_layout()
        frames_tr.append(fig_to_pil_image(fig, dpi=105))
        plt.close(fig)
    generated.append(save_gif_from_frames(frames_tr, root / "03_Cellular_Automata_and_Particle_Computation" / "animations" / "04_traffic_rule184_nagel_schreckenberg_jams.gif", duration_ms=110))

    # 4. Module 03 GIF C: Neuromorphic Glider Parity Logic Execution
    cases = [(0, 0), (0, 1), (1, 0), (1, 1)]
    masks = [_run_glider_xor_gate(a, b, width=88, steps=56)[1] for a, b in cases]
    frames_neuro = []
    for t_cur in range(4, 57, 2):
        fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.8))
        fig.patch.set_facecolor(PALETTE["bg"])
        for ax, (a, b), m in zip(axes, cases, masks):
            v = np.zeros_like(m)
            v[:t_cur] = m[:t_cur]
            ax.imshow(v, cmap="magma", interpolation="nearest", aspect="auto")
            ax.axhline(t_cur - 0.5, color="#38BDF8", lw=1.4)
            ax.set_title(f"In=({a},{b}) $\\to$ XOR={a^b}", fontsize=10.2, fontweight="bold")
            ax.set_xlabel("Bus Site")
            ax.set_ylabel("Clock $t$")
        fig.suptitle(f"Module 03 Neuromorphic Glider Parity Gate (t={t_cur}/56)", fontsize=12, fontweight="bold")
        fig.tight_layout()
        frames_neuro.append(fig_to_pil_image(fig, dpi=100))
        plt.close(fig)
    generated.append(save_gif_from_frames(frames_neuro, root / "03_Cellular_Automata_and_Particle_Computation" / "animations" / "05_neuromorphic_glider_parity_circuit.gif", duration_ms=115))

    # 5. Module 04 GIF: Evolved Swarm Robotics Foraging Arena (Gen 1 vs Evolved)
    _, swarm_meta = demo_m04_swarm_robotics_foraging(generations=20, seed=7)
    plt.close("all")
    s1 = swarm_meta["snaps_gen1"]
    s2 = swarm_meta["snaps_evolved"]
    frames_sw = []
    for idx in range(0, min(len(s1), len(s2)), 2):
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))
        fig.patch.set_facecolor(PALETTE["bg"])
        for ax, snap, title, rcol in [
            (axes[0], s1[idx], f"Gen 1 Uncoordinated (Collected={snap['collected']})" if False else f"Gen 1 Random (Step {idx}, Food={s1[idx]['collected']})", PALETTE["crimson"]),
            (axes[1], s2[idx], f"Gen 20 Evolved Swarm (Step {idx}, Food={s2[idx]['collected']})", PALETTE["teal"]),
        ]:
            fy, fx = np.where(snap["food"])
            ax.scatter(fx, fy, marker="s", s=34, color=PALETTE["amber"], label="Food")
            ax.scatter(snap["ry"], snap["rx"], marker="o", s=80, color=rcol, edgecolor=PALETTE["ink"], label="Robots")
            ax.set_xlim(0, 27)
            ax.set_ylim(0, 27)
            ax.set_aspect("equal")
            style_axes(ax, title=title, xlabel="Arena $x$", ylabel="Arena $y$")
            ax.legend(loc="upper right", fontsize=8.0)
        fig.suptitle("Module 04 Real-World Demo: Decentralized Evolutionary Swarm Robotics", fontsize=12, fontweight="bold")
        fig.tight_layout()
        frames_sw.append(fig_to_pil_image(fig, dpi=105))
        plt.close(fig)
    generated.append(save_gif_from_frames(frames_sw, root / "04_Evolutionary_Computation_and_Edge_of_Chaos" / "animations" / "03_swarm_robotics_foraging_evolution.gif", duration_ms=120))

    return generated
