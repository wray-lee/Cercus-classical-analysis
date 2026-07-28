"""
Cercus Framework — Constants Package
=====================================

Backward-compatible re-exports of all constants.

All values are now loaded dynamically from YAML configuration files
(config/defaults/*.yaml) and can be overridden in the project root
config.yaml.

Usage:
    from cercus.constants import ESCAPE_VMAX_THRESHOLD, RADIUS_MM, COLOR_ESCAPE
"""

from __future__ import annotations

# ── Thresholds ──
from cercus.constants.thresholds import (
    ESCAPE_START_THRESHOLD,
    ESCAPE_VMAX_THRESHOLD,
    ESCAPE_WINDOW_MS,
    POST_STIM_BUFFER_MS,
    PREWALK_THRESHOLD,
    PREWALK_WINDOW_MS,
)

# ── Geometry ──
from cercus.constants.geometry import (
    HEATMAP_SPEED_MAX,
    HEATMAP_VMAX,
    LEGACY_TRIAL_DURATION_MS,
    RADIUS_MM,
    SPEED_WINDOW_MS,
    TRAJECTORY_MAX_RADIUS_MM,
    TRAJECTORY_STEP_MM,
)

# ── Colors ──
from cercus.constants.colors import (
    COLOR_CONTROL,
    COLOR_ESCAPE,
    COLOR_LEFT,
    COLOR_NO_RESPONSE,
    COLOR_NO_STILLNESS,
    COLOR_OSCI_HW,
    COLOR_OSCI_VIS,
    COLOR_PREWALK,
    COLOR_RIGHT,
    COLOR_WITH_STILLNESS,
    NPG_PALETTE,
)

__all__ = [
    # Thresholds
    "ESCAPE_START_THRESHOLD",
    "ESCAPE_VMAX_THRESHOLD",
    "ESCAPE_WINDOW_MS",
    "PREWALK_THRESHOLD",
    "PREWALK_WINDOW_MS",
    "POST_STIM_BUFFER_MS",
    # Geometry
    "RADIUS_MM",
    "TRAJECTORY_MAX_RADIUS_MM",
    "TRAJECTORY_STEP_MM",
    "SPEED_WINDOW_MS",
    "LEGACY_TRIAL_DURATION_MS",
    "HEATMAP_VMAX",
    "HEATMAP_SPEED_MAX",
    # Colors
    "COLOR_LEFT",
    "COLOR_RIGHT",
    "COLOR_CONTROL",
    "COLOR_OSCI_VIS",
    "COLOR_OSCI_HW",
    "COLOR_ESCAPE",
    "COLOR_PREWALK",
    "COLOR_NO_RESPONSE",
    "COLOR_WITH_STILLNESS",
    "COLOR_NO_STILLNESS",
    "NPG_PALETTE",
]
