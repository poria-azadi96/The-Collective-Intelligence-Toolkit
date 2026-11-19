"""Automata & Random Boolean Networks package (Modules 03 & 05)."""

from core.cellular_automata import (
    WOLFRAM_CLASSES,
    HansonCrutchfieldFST,
    apply_rule110_domain_filter,
    apply_rule54_domain_filter,
    build_initial_state,
    compute_damage_spreading,
    compute_langton_lambda,
    evolve_eca,
    evolve_eca_from_state,
    extract_collision_catalog,
    rule_to_table,
)
from core.rbn_immune import (
    analyze_exact_state_space,
    compute_derrida_curve,
    derrida_annealed_theory,
    derrida_sensitivity,
    generate_rbn,
    sample_asynchronous_trajectories,
    step_rbn_sync,
)

__all__ = [
    "HansonCrutchfieldFST",
    "WOLFRAM_CLASSES",
    "analyze_exact_state_space",
    "apply_rule110_domain_filter",
    "apply_rule54_domain_filter",
    "build_initial_state",
    "compute_damage_spreading",
    "compute_derrida_curve",
    "compute_langton_lambda",
    "derrida_annealed_theory",
    "derrida_sensitivity",
    "evolve_eca",
    "evolve_eca_from_state",
    "extract_collision_catalog",
    "generate_rbn",
    "rule_to_table",
    "sample_asynchronous_trajectories",
    "step_rbn_sync",
]
