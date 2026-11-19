"""Automata & Particle Computation package (Module 03): 1D ECA, FST Domain Filter & Damage Spreading."""

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

__all__ = [
    "HansonCrutchfieldFST",
    "WOLFRAM_CLASSES",
    "apply_rule110_domain_filter",
    "apply_rule54_domain_filter",
    "build_initial_state",
    "compute_damage_spreading",
    "compute_langton_lambda",
    "evolve_eca",
    "evolve_eca_from_state",
    "extract_collision_catalog",
    "rule_to_table",
]
