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
    plot_population_vmax_response,
    plot_vmax_distribution,
)
from cercus.visualization.behavior import (
    plot_behavior_probability,
    plot_habituation_curve,
    plot_population_behavior_probability,
    plot_population_habituation,
    plot_prewalk_stillness,
)
from cercus.visualization.polar import (
    plot_escape_angle_distribution,
    plot_population_polar_histogram,
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
    "plot_population_vmax_response",
    "plot_behavior_probability",
    "plot_habituation_curve",
    "plot_population_habituation",
    "plot_population_behavior_probability",
    "plot_prewalk_stillness",
    "plot_escape_angle_distribution",
    "plot_population_polar_histogram",
]
