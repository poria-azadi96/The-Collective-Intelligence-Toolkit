"""Module 03: 1D Elementary Cellular Automata, Hanson-Crutchfield FST & Damage Spreading."""

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

WOLFRAM_CLASSES: dict[int, dict[str, str]] = {
    254: {"class": "Class I", "label": "Fixed Point / Homogeneous", "color": PALETTE["slate"]},
    108: {"class": "Class II", "label": "Periodic / Localized Structures", "color": PALETTE["cobalt"]},
    90: {"class": "Class II/Fractal", "label": "Additive Linear (Sierpinski XOR)", "color": PALETTE["amber"]},
    30: {"class": "Class III", "label": "Chaotic / Pseudo-Random", "color": PALETTE["crimson"]},
    54: {"class": "Class IV", "label": "Complex Particle Computation", "color": PALETTE["teal"]},
    110: {"class": "Class IV", "label": "Turing Universal Gliders", "color": PALETTE["teal"]},
}


def rule_to_table(rule_number: int) -> np.ndarray:
    """Return 8-bit Wolfram lookup table ordered by (left<<2 | center<<1 | right)."""
    return np.array([(int(rule_number) >> i) & 1 for i in range(8)], dtype=np.uint8)


def compute_langton_lambda(rule_number: int) -> float:
    """Compute Langton's activity parameter lambda = (# of 1s in rule table) / 8."""
    return float(np.sum(rule_to_table(rule_number))) / 8.0


def build_initial_state(
    rule_number: int,
    width: int = 160,
    init_type: str = "collision_setup",
    seed: int = 42,
) -> np.ndarray:
    """Construct clean reproducible 1D initial lattice state for any ECA rule."""
    rng = np.random.default_rng(seed)
    w = int(width)
    if init_type == "single_seed":
        row = np.zeros(w, dtype=np.uint8)
        row[w // 2] = 1
        return row

    if init_type == "collision_setup" and rule_number == 54:
        # Period-4 Rule 54 ether (1110...) with gamma+ (0010), central alpha/beta (0011), and gamma- (1000)
        tile = np.array([1, 1, 1, 0], dtype=np.uint8)
        row = np.tile(tile, w // 4 + 1)[:w]
        p_left = max(4, ((w // 6) // 4) * 4)
        p_mid = max(p_left + 12, ((w // 2) // 4) * 4)
        p_right = min(w - 8, ((5 * w // 6) // 4) * 4)
        row[p_left : p_left + 4] = [0, 0, 1, 0]   # gamma+ (v = +1 site/step)
        row[p_mid : p_mid + 4] = [0, 0, 1, 1]     # central stationary particle
        row[p_right : p_right + 4] = [1, 0, 0, 0] # gamma- (v = -1 site/step)
        return row

    if init_type == "collision_setup" and rule_number == 110:
        # Period-14 Rule 110 ether (11111000100110) perturbed to emit colliding Cook gliders
        ether_14 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0], dtype=np.uint8)
        row = np.tile(ether_14, w // 14 + 1)[:w]
        p1 = max(14, (w // 4 // 14) * 14)
        p2 = min(w - 14, (3 * w // 4 // 14) * 14)
        row[p1 : p1 + 5] = [0, 0, 0, 1, 1]
        row[p2 : p2 + 6] = [1, 0, 0, 0, 1, 0]
        return row

    return rng.integers(0, 2, size=w, dtype=np.uint8)


def evolve_eca_from_state(rule_number: int, init_row: np.ndarray, steps: int = 120) -> np.ndarray:
    """Evolve a 1D ECA from a specified initial row under periodic boundaries."""
    rule_bits = rule_to_table(rule_number)
    width = len(init_row)
    grid = np.zeros((steps, width), dtype=np.uint8)
    grid[0] = np.asarray(init_row, dtype=np.uint8)
    for t in range(steps - 1):
        left = np.roll(grid[t], 1)
        center = grid[t]
        right = np.roll(grid[t], -1)
        idx = (left << 2) | (center << 1) | right
        grid[t + 1] = rule_bits[idx]
    return grid


def evolve_eca(
    rule_number: int,
    width: int = 160,
    steps: int = 120,
    init_type: str = "random",
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Evolve a 1D Elementary Cellular Automaton and return (spacetime_grid, lambda)."""
    init_row = build_initial_state(rule_number, width=width, init_type=init_type, seed=seed)
    grid = evolve_eca_from_state(rule_number, init_row, steps=steps)
    return grid, compute_langton_lambda(rule_number)


class HansonCrutchfieldFST:
    """Explicit 5-state Finite-State Transducer {q_0, q_1, q_2, q_3, q_defect} for Rule 54."""

    STATES = ("q_0", "q_1", "q_2", "q_3", "q_defect")
    PHASE_A_WORDS: dict[tuple[int, int, int, int], int] = {
        (1, 1, 1, 0): 0,
        (1, 1, 0, 1): 1,
        (1, 0, 1, 1): 2,
        (0, 1, 1, 1): 3,
    }
    PHASE_B_WORDS: dict[tuple[int, int, int, int], int] = {
        (0, 0, 0, 1): 0,
        (0, 0, 1, 0): 1,
        (0, 1, 0, 0): 2,
        (1, 0, 0, 0): 3,
    }

    def filter_row(self, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run FST across a 1D lattice row; return (particle_mask, state_codes 0..4)."""
        w = len(row)
        a_hits = sum(
            1 for i in range(w)
            if (int(row[i]), int(row[(i + 1) % w]), int(row[(i + 2) % w]), int(row[(i + 3) % w])) in self.PHASE_A_WORDS
        )
        b_hits = sum(
            1 for i in range(w)
            if (int(row[i]), int(row[(i + 1) % w]), int(row[(i + 2) % w]), int(row[(i + 3) % w])) in self.PHASE_B_WORDS
        )
        active_words = self.PHASE_A_WORDS if a_hits >= b_hits else self.PHASE_B_WORDS

        # State index 4 represents q_defect; 0..3 represent q_0..q_3 in the period-4 ether cycle
        state_codes = np.full(w, 4, dtype=np.uint8)
        for i in range(w):
            word = (
                int(row[i]),
                int(row[(i + 1) % w]),
                int(row[(i + 2) % w]),
                int(row[(i + 3) % w]),
            )
            if word in active_words:
                base_q = active_words[word]
                for offset in range(4):
                    state_codes[(i + offset) % w] = (base_q + offset) % 4

        particle_mask = (state_codes == 4).astype(np.uint8)
        return particle_mask, state_codes

    def filter(self, spacetime_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filter an entire spacetime diagram; return (particle_mask, fst_state_codes)."""
        steps, width = spacetime_grid.shape
        particle_mask = np.zeros((steps, width), dtype=np.uint8)
        state_grid = np.full((steps, width), 4, dtype=np.uint8)
        for t in range(steps):
            pm, st = self.filter_row(spacetime_grid[t])
            particle_mask[t] = pm
            state_grid[t] = st
        return particle_mask, state_grid


def apply_rule54_domain_filter(spacetime_grid: np.ndarray) -> np.ndarray:
    """Strip Rule 54's period-4 ether via Hanson-Crutchfield FST, leaving particles as 1."""
    fst = HansonCrutchfieldFST()
    particle_mask, _ = fst.filter(spacetime_grid)
    return particle_mask


def apply_rule110_domain_filter(spacetime_grid: np.ndarray) -> np.ndarray:
    """Strip Rule 110's period-14 spatial ether (11111000100110) to reveal Cook gliders."""
    base_14 = (1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0)
    valid_14 = {tuple(np.roll(base_14, s)) for s in range(14)}
    steps, width = spacetime_grid.shape
    in_domain = np.zeros((steps, width), dtype=bool)
    for t in range(steps):
        row = spacetime_grid[t]
        for i in range(width):
            seg = tuple(int(row[(i + k) % width]) for k in range(14))
            if seg in valid_14:
                for k in range(14):
                    in_domain[t, (i + k) % width] = True
    return (~in_domain).astype(np.uint8)


def extract_collision_catalog(particle_mask: np.ndarray) -> dict[str, Any]:
    """Identify localized collision vertices, density trace, and particle interaction stats."""
    steps, width = particle_mask.shape
    defect_fraction = float(np.mean(particle_mask))
    density_over_time = np.mean(particle_mask, axis=1)

    raw_pts: list[tuple[int, int]] = []
    for t in range(1, steps - 1):
        for i in range(2, width - 2):
            patch_sum = int(np.sum(particle_mask[t - 1 : t + 2, i - 1 : i + 2]))
            row_span = int(np.sum(particle_mask[t, i - 2 : i + 3]))
            if particle_mask[t, i] == 1 and patch_sum >= 7 and row_span >= 3:
                raw_pts.append((t, i))

    # Cluster nearby detections within Manhattan radius 4 into distinct collision vertices
    vertices: list[tuple[float, float]] = []
    used = [False] * len(raw_pts)
    for idx, (t0, i0) in enumerate(raw_pts):
        if used[idx]:
            continue
        cluster = [(t0, i0)]
        used[idx] = True
        for jdx in range(idx + 1, len(raw_pts)):
            if not used[jdx]:
                tj, ij = raw_pts[jdx]
                if abs(tj - t0) <= 4 and abs(ij - i0) <= 5:
                    cluster.append((tj, ij))
                    used[jdx] = True
        vertices.append((float(np.mean([p[0] for p in cluster])), float(np.mean([p[1] for p in cluster]))))

    return {
        "defect_fraction": defect_fraction,
        "density_over_time": density_over_time,
        "raw_collision_points": raw_pts,
        "collision_vertices": vertices,
    }


def compute_damage_spreading(
    rule_number: int,
    width: int = 160,
    steps: int = 120,
    init_type: str = "random",
    seed: int = 42,
) -> dict[str, Any]:
    """Flip 1 center bit (Delta = 1 bit) and compute XOR difference cone and Hamming H(t)."""
    grid, lam = evolve_eca(rule_number, width=width, steps=steps, init_type=init_type, seed=seed)
    init_pert = grid[0].copy()
    init_pert[width // 2] ^= 1
    grid_pert = evolve_eca_from_state(rule_number, init_pert, steps=steps)
    xor_cone = np.bitwise_xor(grid, grid_pert)
    hamming_t = np.mean(xor_cone, axis=1).astype(np.float64)
    return {
        "rule": rule_number,
        "lambda": lam,
        "grid": grid,
        "grid_perturbed": grid_pert,
        "xor_cone": xor_cone,
        "hamming_t": hamming_t,
        "final_hamming": float(hamming_t[-1]),
        "mean_hamming": float(np.mean(hamming_t)),
    }


def create_ca_dashboard_figure(
    rule_number: int = 54,
    init_type: str = "collision_setup",
    show_filter: bool = True,
    width: int = 160,
    steps: int = 120,
    seed: int = 42,
) -> tuple[plt.Figure, dict[str, Any]]:
    """Build a 3-panel classroom dashboard for Module 03."""
    apply_publication_style()
    dmg = compute_damage_spreading(rule_number, width=width, steps=steps, init_type=init_type, seed=seed)
    grid = dmg["grid"]
    lam = dmg["lambda"]
    cls_info = WOLFRAM_CLASSES.get(rule_number, {"class": "ECA", "label": f"Rule {rule_number}"})

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.9))
    fig.patch.set_facecolor(PALETTE["bg"])

    # Panel 1: Raw Spacetime Diagram
    axes[0].imshow(grid, cmap="binary", interpolation="nearest", aspect="auto")
    axes[0].set_title(f"1. Raw Spacetime (Rule {rule_number} | {cls_info['class']}, $\\lambda={lam:.3f}$)", fontsize=10.8, fontweight="bold")
    axes[0].set_xlabel("Lattice Site $i$")
    axes[0].set_ylabel("Time Step $t$ (Top $\\to$ Bottom)")

    metrics: dict[str, Any] = {
        "rule": rule_number,
        "lambda": lam,
        "class": cls_info["class"],
        "final_hamming": dmg["final_hamming"],
        "mean_hamming": dmg["mean_hamming"],
    }

    # Panel 2: FST Domain Filter or Perturbation Cone
    if show_filter and rule_number in (54, 110):
        if rule_number == 54:
            fst = HansonCrutchfieldFST()
            p_mask, _ = fst.filter(grid)
            filter_name = "Hanson-Crutchfield FST Filter (Rule 54)"
        else:
            p_mask = apply_rule110_domain_filter(grid)
            filter_name = "Period-14 Ether Filter (Rule 110)"
        cat = extract_collision_catalog(p_mask)
        axes[1].imshow(p_mask, cmap="magma", interpolation="nearest", aspect="auto")
        if cat["collision_vertices"]:
            vt, vi = zip(*cat["collision_vertices"])
            axes[1].scatter(vi, vt, s=95, facecolors="none", edgecolors="#38BDF8", linewidths=1.8, label=f"Collision Vertices ({len(vt)})")
            axes[1].legend(loc="upper right", fontsize=8.4)
        axes[1].set_title(f"2. {filter_name}", fontsize=10.8, fontweight="bold")
        axes[1].set_xlabel("Lattice Site $i$")
        axes[1].set_ylabel("Time Step $t$")
        metrics["defect_fraction"] = cat["defect_fraction"]
        metrics["n_collisions"] = len(cat["collision_vertices"])
    else:
        axes[1].imshow(dmg["xor_cone"], cmap="Reds", interpolation="nearest", aspect="auto")
        axes[1].set_title("2. Single-Bit Perturbation XOR Cone ($\\Delta = 1$ bit)", fontsize=10.8, fontweight="bold")
        axes[1].set_xlabel("Lattice Site $i$")
        axes[1].set_ylabel("Time Step $t$")

    # Panel 3: Normalized Hamming Damage Curve H(t)
    ax2 = style_axes(
        axes[2],
        title="3. Normalized Hamming Distance $H(t)$",
        xlabel="Time Step $t$",
        ylabel="Hamming Divergence $H(t)$",
    )
    ax2.plot(np.arange(steps), dmg["hamming_t"], color=PALETTE["crimson"], lw=2.3, label=f"Rule {rule_number} $H(t)$")
    ax2.axhline(0.5, color=PALETTE["slate"], linestyle="--", lw=1.2, label="Max Mixing ($H=0.5$)")
    ax2.set_ylim(-0.02, 0.58)
    ax2.legend(loc="best", fontsize=8.5)

    fig.suptitle(
        f"Module 03 Dashboard — Rule {rule_number} ({cls_info['label']}) | Width={width}, Steps={steps}, Init={init_type}",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    return fig, metrics


def run_module03_tier1(width: int = 128, steps: int = 90) -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 1: Toggle Hanson-Crutchfield FST Domain Filter OFF vs ON on Rule 54 collision_setup."""
    apply_publication_style()
    grid, lam = evolve_eca(54, width=width, steps=steps, init_type="collision_setup", seed=42)
    fst = HansonCrutchfieldFST()
    p_mask, state_grid = fst.filter(grid)
    cat = extract_collision_catalog(p_mask)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))
    fig.patch.set_facecolor(PALETTE["bg"])

    axes[0].imshow(grid, cmap="binary", interpolation="nearest", aspect="auto")
    axes[0].set_title(f"Tier 1A: Filter OFF — Raw Rule 54 Spacetime ($\\lambda={lam:.3f}$)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Lattice Site $i$")
    axes[0].set_ylabel("Time Step $t$")

    axes[1].imshow(p_mask, cmap="magma", interpolation="nearest", aspect="auto")
    if cat["collision_vertices"]:
        vt, vi = zip(*cat["collision_vertices"])
        axes[1].scatter(vi, vt, s=130, facecolors="none", edgecolors="#38BDF8", linewidths=2.0, label=f"Collision Vertices ({len(vt)})")
    axes[1].set_title("Tier 1B: Filter ON — FST Strips Ether & Reveals Glider Collisions", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Lattice Site $i$")
    axes[1].set_ylabel("Time Step $t$")
    axes[1].legend(loc="upper right", fontsize=8.8)

    fig.suptitle("Module 03 — Tier 1 (Observation): Rule 54 Emergent Particle Computation", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {
        "defect_fraction": cat["defect_fraction"],
        "n_collisions": len(cat["collision_vertices"]),
        "collision_vertices": cat["collision_vertices"],
        "fst_defect_state_pct": float(np.mean(state_grid == 4)),
    }


def run_module03_tier2(width: int = 180, steps: int = 140, seed: int = 42) -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 2: Rule 54 random init particle density decay and glider slope verification."""
    apply_publication_style()
    grid, _ = evolve_eca(54, width=width, steps=steps, init_type="random", seed=seed)
    p_mask = apply_rule54_domain_filter(grid)
    cat = extract_collision_catalog(p_mask)
    density_t = cat["density_over_time"]

    # Also build clean single-glider velocity slope verification reference
    ref_grid, _ = evolve_eca(54, width=128, steps=60, init_type="collision_setup", seed=42)
    ref_mask = apply_rule54_domain_filter(ref_grid)

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.8))
    fig.patch.set_facecolor(PALETTE["bg"])

    axes[0].imshow(p_mask, cmap="magma", interpolation="nearest", aspect="auto")
    if cat["collision_vertices"]:
        vt, vi = zip(*cat["collision_vertices"][:25])
        axes[0].scatter(vi, vt, s=65, facecolors="none", edgecolors="#38BDF8", linewidths=1.3)
    axes[0].set_title("Tier 2A: Rule 54 Random Init (FST Filtered)", fontsize=10.8, fontweight="bold")
    axes[0].set_xlabel("Lattice Site $i$")
    axes[0].set_ylabel("Time Step $t$")

    ax1 = style_axes(axes[1], title="Tier 2B: Particle Density Annihilation $\\rho(t)$", xlabel="Time Step $t$", ylabel="Defect Density")
    ax1.plot(np.arange(steps), density_t, color=PALETTE["teal"], lw=2.3, label="Particle Density")
    ax1.fill_between(np.arange(steps), 0, density_t, color=PALETTE["teal"], alpha=0.14)
    ax1.legend(loc="upper right", fontsize=8.6)

    axes[2].imshow(ref_mask, cmap="magma", interpolation="nearest", aspect="auto")
    t_line = np.arange(0, 36)
    axes[2].plot(16 + t_line, t_line, color="#38BDF8", lw=1.6, linestyle="--", label="$\\gamma^+$ ($v=+1$ / $+\\frac{1}{2}$ pair)")
    axes[2].plot(104 - t_line, t_line, color="#FBBF24", lw=1.6, linestyle="--", label="$\\gamma^-$ ($v=-1$ / $-\\frac{1}{2}$ pair)")
    axes[2].axvline(61, color="#34D399", lw=1.4, linestyle=":", label="$\\alpha/\\beta$ ($v=0$)")
    axes[2].set_title("Tier 2C: Particle Worldline Slope Verification", fontsize=10.8, fontweight="bold")
    axes[2].set_xlabel("Lattice Site $i$")
    axes[2].set_ylabel("Time Step $t$")
    axes[2].legend(loc="lower right", fontsize=7.8)

    fig.suptitle("Module 03 — Tier 2 (Analysis): Particle Density Dynamics & Worldline Velocities", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, {
        "initial_density": float(density_t[0]),
        "final_density": float(density_t[-1]),
        "n_collisions": len(cat["collision_vertices"]),
    }


def run_module03_tier3(width: int = 141, steps: int = 110, seed: int = 42) -> tuple[plt.Figure, dict[str, Any]]:
    """Tier 3: Damage spreading comparison across Rule 90, Rule 30, and Rule 110."""
    apply_publication_style()
    rules = [90, 30, 110]
    dmg_map = {
        r: compute_damage_spreading(r, width=width, steps=steps, init_type="random", seed=seed)
        for r in rules
    }
    colors = {90: PALETTE["amber"], 30: PALETTE["crimson"], 110: PALETTE["teal"]}
    titles = {90: "Rule 90 (Linear Fractal)", 30: "Rule 30 (Class III Chaotic)", 110: "Rule 110 (Class IV Universal)"}

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8))
    fig.patch.set_facecolor(PALETTE["bg"])
    panel_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]

    for ax, r in zip(panel_axes, rules):
        ax.imshow(dmg_map[r]["xor_cone"], cmap="Reds", interpolation="nearest", aspect="auto")
        ax.set_title(f"{titles[r]} | $\\lambda={dmg_map[r]['lambda']:.3f}$, $H_f={dmg_map[r]['final_hamming']:.3f}$", fontsize=10.8, fontweight="bold")
        ax.set_xlabel("Lattice Site $i$")
        ax.set_ylabel("Time Step $t$")

    ax_h = style_axes(axes[1, 1], title="Normalized Hamming Divergence $H(t)$ Across Wolfram Classes", xlabel="Time Step $t$", ylabel="Hamming Distance $H(t)$")
    for r in rules:
        ax_h.plot(np.arange(steps), dmg_map[r]["hamming_t"], color=colors[r], lw=2.2, label=titles[r])
    ax_h.axhline(0.5, color=PALETTE["slate"], linestyle="--", lw=1.2, label="Chaotic Saturation ($H=0.5$)")
    ax_h.set_ylim(-0.02, 0.58)
    ax_h.legend(loc="best", fontsize=8.6)

    fig.suptitle("Module 03 — Tier 3 (Damage Spreading): Linear vs. Chaotic vs. Edge-of-Chaos Cones", fontsize=12.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, {
        r: {"final_hamming": dmg_map[r]["final_hamming"], "mean_hamming": dmg_map[r]["mean_hamming"]}
        for r in rules
    }


def generate_module03_gifs(out_dir: str | Path) -> list[Path]:
    """Generate classroom-ready animated GIFs for Module 03."""
    apply_publication_style(dpi=115)
    anim_dir = Path(out_dir) / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # GIF 1: Progressive Rule 54 Raw Spacetime vs Hanson-Crutchfield FST Domain Filter
    steps, width = 80, 128
    grid54, _ = evolve_eca(54, width=width, steps=steps, init_type="collision_setup", seed=42)
    mask54 = apply_rule54_domain_filter(grid54)
    cat54 = extract_collision_catalog(mask54)
    verts = cat54["collision_vertices"]

    frames_1 = []
    for t_cur in range(4, steps + 1, 3):
        fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
        fig.patch.set_facecolor(PALETTE["bg"])

        raw_view = np.zeros_like(grid54)
        raw_view[:t_cur] = grid54[:t_cur]
        flt_view = np.zeros_like(mask54)
        flt_view[:t_cur] = mask54[:t_cur]

        axes[0].imshow(raw_view, cmap="binary", interpolation="nearest", aspect="auto")
        axes[0].axhline(t_cur - 0.5, color=PALETTE["cobalt"], lw=1.5)
        axes[0].set_title(f"Raw Rule 54 Spacetime ($t = {t_cur}/{steps}$)", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Lattice Site $i$")
        axes[0].set_ylabel("Time Step $t$")

        axes[1].imshow(flt_view, cmap="magma", interpolation="nearest", aspect="auto")
        axes[1].axhline(t_cur - 0.5, color="#38BDF8", lw=1.5)
        vis_verts = [(vt, vi) for vt, vi in verts if vt <= t_cur]
        if vis_verts:
            v_t, v_i = zip(*vis_verts)
            axes[1].scatter(v_i, v_t, s=115, facecolors="none", edgecolors="#38BDF8", linewidths=1.8)
        axes[1].set_title("Hanson-Crutchfield FST Filter (Gliders & Collisions)", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Lattice Site $i$")
        axes[1].set_ylabel("Time Step $t$")

        fig.suptitle("Module 03 Live Execution: Stripping Period-4 Ether Reveals Glider Computation", fontsize=12, fontweight="bold")
        fig.tight_layout()
        frames_1.append(fig_to_pil_image(fig, dpi=105))
        plt.close(fig)

    generated.append(save_gif_from_frames(frames_1, anim_dir / "01_rule54_domain_filter_and_collisions.gif", duration_ms=120))

    # GIF 2: Wolfram Classes XOR Damage Spreading Comparison (Rule 90, Rule 30, Rule 110)
    steps2, width2 = 96, 129
    d90 = compute_damage_spreading(90, width=width2, steps=steps2, init_type="random", seed=42)
    d30 = compute_damage_spreading(30, width=width2, steps=steps2, init_type="random", seed=42)
    d110 = compute_damage_spreading(110, width=width2, steps=steps2, init_type="random", seed=42)

    frames_2 = []
    for t_cur in range(4, steps2 + 1, 4):
        fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.9))
        fig.patch.set_facecolor(PALETTE["bg"])
        for ax, d_obj, name in [
            (axes[0], d90, "Rule 90 (Fractal XOR)"),
            (axes[1], d30, "Rule 30 (Chaotic Cone)"),
            (axes[2], d110, "Rule 110 (Edge of Chaos)"),
        ]:
            v = np.zeros_like(d_obj["xor_cone"])
            v[:t_cur] = d_obj["xor_cone"][:t_cur]
            ax.imshow(v, cmap="Reds", interpolation="nearest", aspect="auto", vmin=0, vmax=1)
            ax.set_title(name, fontsize=10, fontweight="bold")
            ax.set_xlabel("Site $i$")
            ax.set_ylabel("Time $t$")

        ax_h = style_axes(axes[3], title=f"Damage $H(t)$ ($t={t_cur}$)", xlabel="Time $t$", ylabel="$H(t)$")
        ax_h.plot(range(t_cur), d90["hamming_t"][:t_cur], color=PALETTE["amber"], lw=2.0, label="R90")
        ax_h.plot(range(t_cur), d30["hamming_t"][:t_cur], color=PALETTE["crimson"], lw=2.0, label="R30")
        ax_h.plot(range(t_cur), d110["hamming_t"][:t_cur], color=PALETTE["teal"], lw=2.0, label="R110")
        ax_h.axhline(0.5, color=PALETTE["slate"], linestyle="--", lw=1.0)
        ax_h.set_xlim(0, steps2)
        ax_h.set_ylim(-0.02, 0.58)
        ax_h.legend(loc="upper left", fontsize=7.8)

        fig.suptitle("Module 03 Damage Spreading Across Wolfram Behavioral Classes", fontsize=12, fontweight="bold")
        fig.tight_layout()
        frames_2.append(fig_to_pil_image(fig, dpi=105))
        plt.close(fig)

    generated.append(save_gif_from_frames(frames_2, anim_dir / "02_wolfram_classes_damage_cone.gif", duration_ms=120))
    return generated
