"""
Cercus Framework — Analysis Package
===================================
Individual- and population-level statistical analyses.

All circular statistics use ``circ_mean_rad`` / circular distance functions;
linear means are never applied to angular data.
"""

from cercus.analysis.individual import (
    bootstrap_by_animal,
    loo_robustness,
    run_individual_checks,
    second_order_analysis,
    trial_counts_per_animal,
    trial_escape_angles,
    wallraff_by_animal,
)

__all__ = [
    "trial_escape_angles",
    "trial_counts_per_animal",
    "second_order_analysis",
    "loo_robustness",
    "bootstrap_by_animal",
    "wallraff_by_animal",
    "run_individual_checks",
]
