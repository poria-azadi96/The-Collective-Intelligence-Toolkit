"""Bio-Inspired Swarms, Adaptive Networks & Immune Cognition package (Modules 05 & 06)."""

from core.adaptive_spatial_networks import (
    build_tokyo_benchmark_graph,
    compute_network_metrics,
    simulate_contagion_comparison,
    simulate_vicsek_flocking,
    solve_physarum_network,
)
from core.rbn_immune import (
    kinetic_proofreading_specificity,
    simulate_immune_selection,
)

__all__ = [
    "build_tokyo_benchmark_graph",
    "compute_network_metrics",
    "kinetic_proofreading_specificity",
    "simulate_contagion_comparison",
    "simulate_immune_selection",
    "simulate_vicsek_flocking",
    "solve_physarum_network",
]
