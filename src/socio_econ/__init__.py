"""Socio-Economic Dynamics, Markets & Research Validation package (Modules 07 & 08)."""

from core.collective_markets import (
    simulate_degroot_consensus,
    simulate_dpt_ensemble,
    simulate_el_farol_bar,
    simulate_hk_opinions,
    simulate_minority_game,
)
from core.research_validation import (
    PAPER_AUDITS,
    benjamini_hochberg_fdr,
    build_research_brief,
    calibrate_fdr_power_curve,
    compute_cohens_d,
    simulate_abm_vs_null,
    simulate_fdr_validation,
)

__all__ = [
    "PAPER_AUDITS",
    "benjamini_hochberg_fdr",
    "build_research_brief",
    "calibrate_fdr_power_curve",
    "compute_cohens_d",
    "simulate_abm_vs_null",
    "simulate_degroot_consensus",
    "simulate_dpt_ensemble",
    "simulate_el_farol_bar",
    "simulate_fdr_validation",
    "simulate_hk_opinions",
    "simulate_minority_game",
]
