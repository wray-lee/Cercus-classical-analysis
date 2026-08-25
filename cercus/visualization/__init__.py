"""
Cercus Framework — Visualization Package
========================================
Publication-grade plotting for behavioral neuroscience.
All plot_* functions re-exported at package level.
"""

from cercus.visualization.trajectories import (
    plot_global_trajectory_overlay_fixed,
    plot_trajectory_overlay,
)
from cercus.visualization.kinetics import (
    plot_population_spaghetti_kinetics,
    plot_population_speed_kinetics,
    plot_single_trial_kinetics,
    plot_spaghetti_kinetics,
    plot_speed_kinetics,
)
from cercus.visualization.heatmaps import (
    plot_spaghetti_kinetics_heatmap,
    plot_trial_stacked_heatmap,
)
from cercus.visualization.vmax import (
    plot_population_vmax_gmm,
    plot_population_vmax_moving_gmm,
    plot_population_vmax_response,
    plot_vmax_distribution,
)
from cercus.visualization.behavior import (
    plot_behavior_probability,
    plot_habituation_curve,
    plot_population_behavior_probability,
    plot_population_habituation,
    plot_population_prewalk_integration,
    plot_prewalk_stillness,
)
from cercus.visualization.polar import (
    plot_escape_angle_distribution,
    plot_population_polar_histogram,
)
from cercus.visualization.individual import (
    plot_loo,
    plot_second_order,
    plot_trial_counts,
)
from cercus.visualization._circstats import (
    circ_dist_rad,
    circ_mean_rad,
    rayleigh_p,
    wallraff_test,
    wallraff_test_with_ref,
    watson_williams_test,
)

__all__ = [
    "plot_trajectory_overlay",
    "plot_global_trajectory_overlay_fixed",
    "plot_speed_kinetics",
    "plot_population_speed_kinetics",
    "plot_population_spaghetti_kinetics",
    "plot_spaghetti_kinetics",
    "plot_single_trial_kinetics",
    "plot_spaghetti_kinetics_heatmap",
    "plot_trial_stacked_heatmap",
    "plot_vmax_distribution",
    "plot_population_vmax_gmm",
    "plot_population_vmax_moving_gmm",
    "plot_population_vmax_response",
    "plot_behavior_probability",
    "plot_habituation_curve",
    "plot_population_habituation",
    "plot_population_behavior_probability",
    "plot_population_prewalk_integration",
    "plot_prewalk_stillness",
    "plot_escape_angle_distribution",
    "plot_population_polar_histogram",
    "plot_trial_counts",
    "plot_second_order",
    "plot_loo",
    "circ_mean_rad",
    "circ_dist_rad",
    "rayleigh_p",
    "watson_williams_test",
    "wallraff_test",
    "wallraff_test_with_ref",
]
