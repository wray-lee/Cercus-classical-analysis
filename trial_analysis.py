"""
Cercus Framework — Backward-Compatibility Shim
================================================
Re-exports the public API surface of the original monolithic ``trial_analysis.py``
so that downstream consumers (e.g. ``Statistical Analysis/feature_extractor.py``)
continue to work without modification.

All canonical implementations now live in ``pipeline/`` submodules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Re-export from pipeline submodules ──
from pipeline.constants import (
    COLOR_CONTROL,
    COLOR_ESCAPE,
    COLOR_LEFT,
    COLOR_NO_RESPONSE,
    COLOR_OSCI_HW,
    COLOR_OSCI_VIS,
    COLOR_PREWALK,
    COLOR_RIGHT,
    DETAILS_KEYS,
    ESCAPE_START_THRESHOLD,
    ESCAPE_VMAX_THRESHOLD,
    ESCAPE_WINDOW_MS,
    LEGACY_TRIAL_DURATION_MS,
    POST_STIM_BUFFER_MS,
    PREWALK_THRESHOLD,
    PREWALK_WINDOW_MS,
    RADIUS_MM,
    SCALE_BAR_MM,
    SPEED_WINDOW_MS,
    TRAJECTORY_MAX_RADIUS_MM,
    TRAJECTORY_STEP_MM,
    _apply_publication_style,
    _get_unified_side,
)
from pipeline.io import (
    _parse_details,
    export_summary_metrics,
    load_and_concat_sessions,
    load_events,
    load_kinematics,
    scan_and_pair_sessions,
)
from pipeline.kinematics import (
    _compute_theoretical_ttc_ms,
    _integrate_trial,
    _slice_kinematics_by_window,
    compute_escape_latency as _compute_escape_latency_raw,
    preprocess,
)
from pipeline.classifier import classify_trial as _classify_trial, label_trials as _label_trials
from pipeline.visualization import (
    _add_threshold_lines,
    _draw_side_arrows,
    _draw_standardized_grid,
    plot_behavior_probability,
    plot_habituation_curve,
    plot_single_trial_kinetics as plot_single_escape_trial,
    plot_spaghetti_kinetics,
    plot_speed_kinetics,
    plot_trajectory_overlay,
    plot_vmax_distribution,
)


# ══════════════════════════════════════════════════════════════════════
# Backward-Compatibility Wrapper for compute_escape_latency
# ══════════════════════════════════════════════════════════════════════
#
# The original returned ``{"is_escaped": bool, "v_max": float, "latency_ms": float}``
# with a baseline veto.  The new implementation drops the baseline check and
# ``is_escaped``.  This wrapper restores the old return shape so
# ``feature_extractor.py`` (which reads ``is_escaped``) keeps working.
#


def compute_escape_latency(t_rel: np.ndarray, speed: np.ndarray) -> dict:
    """
    Backward-compatible wrapper: adds ``is_escaped`` key by checking
    ``latency_ms`` is not NaN, which matches the original semantics
    (baseline veto removed — the classifier now handles that).
    """
    result = _compute_escape_latency_raw(t_rel, speed)
    return {
        "is_escaped": not np.isnan(result["latency_ms"]),
        "v_max": result["v_max"],
        "latency_ms": result["latency_ms"],
    }


# ── Backward-compat aliases for internal functions ──
_classify_trial = _classify_trial
_label_trials = _label_trials


# ── Suppress "unused import" warnings in linters ──
__all__ = [
    # Style
    "_apply_publication_style",
    # Colors
    "COLOR_LEFT", "COLOR_RIGHT", "COLOR_CONTROL",
    "COLOR_OSCI_VIS", "COLOR_OSCI_HW",
    "COLOR_ESCAPE", "COLOR_PREWALK", "COLOR_NO_RESPONSE",
    # Constants
    "DETAILS_KEYS", "SPEED_WINDOW_MS", "SCALE_BAR_MM",
    "LEGACY_TRIAL_DURATION_MS", "RADIUS_MM",
    "ESCAPE_VMAX_THRESHOLD", "ESCAPE_START_THRESHOLD",
    "PREWALK_THRESHOLD", "PREWALK_WINDOW_MS",
    "POST_STIM_BUFFER_MS", "ESCAPE_WINDOW_MS",
    "TRAJECTORY_MAX_RADIUS_MM", "TRAJECTORY_STEP_MM",
    # Helpers
    "_get_unified_side", "_parse_details",
    # I/O
    "load_events", "load_kinematics",
    "scan_and_pair_sessions", "load_and_concat_sessions",
    "export_summary_metrics",
    # Kinematics
    "preprocess", "compute_escape_latency",
    # Classifier
    "_classify_trial", "_label_trials",
    # Visualization
    "_draw_standardized_grid", "_draw_side_arrows", "_add_threshold_lines",
    "plot_trajectory_overlay", "plot_speed_kinetics", "plot_spaghetti_kinetics",
    "plot_behavior_probability", "plot_habituation_curve",
    "plot_vmax_distribution", "plot_single_escape_trial",
]
