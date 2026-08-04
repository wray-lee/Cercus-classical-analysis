"""
Cercus Framework — Geometry & Timing Constants
===============================================
Physical geometry and timing constants.

⚠️  These values are now loaded from config/defaults/geometry.yaml
    and can be overridden in the project root config.yaml.
"""

from __future__ import annotations

from cercus.config import get_geometry

# ═══════════════════════════════════════════════════════════════════════
# Arena Geometry
# ═══════════════════════════════════════════════════════════════════════

_RADIUS_MM = get_geometry().RADIUS_MM
RADIUS_MM: float = float(_RADIUS_MM)

# ═══════════════════════════════════════════════════════════════════════
# Trajectory Plot Constants
# ═══════════════════════════════════════════════════════════════════════

_TRAJECTORY_MAX_RADIUS_MM = get_geometry().TRAJECTORY_MAX_RADIUS_MM
TRAJECTORY_MAX_RADIUS_MM: float = float(_TRAJECTORY_MAX_RADIUS_MM)

_TRAJECTORY_STEP_MM = get_geometry().TRAJECTORY_STEP_MM
TRAJECTORY_STEP_MM: float = float(_TRAJECTORY_STEP_MM)

# ═══════════════════════════════════════════════════════════════════════
# Speed Calculation
# ═══════════════════════════════════════════════════════════════════════

_SPEED_WINDOW_MS = get_geometry().SPEED_WINDOW_MS
SPEED_WINDOW_MS: float = float(_SPEED_WINDOW_MS)

# ═══════════════════════════════════════════════════════════════════════
# Legacy Settings
# ═══════════════════════════════════════════════════════════════════════

_LEGACY_TRIAL_DURATION_MS = get_geometry().LEGACY_TRIAL_DURATION_MS
LEGACY_TRIAL_DURATION_MS: float = float(_LEGACY_TRIAL_DURATION_MS)

# ═══════════════════════════════════════════════════════════════════════
# Visualization Defaults
# ═══════════════════════════════════════════════════════════════════════

_HEATMAP_VMAX = get_geometry().HEATMAP_VMAX
HEATMAP_VMAX: float = float(_HEATMAP_VMAX)

_HEATMAP_SPEED_MAX = get_geometry().HEATMAP_SPEED_MAX
HEATMAP_SPEED_MAX: float = float(_HEATMAP_SPEED_MAX)

# ═══════════════════════════════════════════════════════════════════════
# Heatmap Window Config
# ═══════════════════════════════════════════════════════════════════════

_heatmap_cfg = get_geometry().heatmap

HEATMAP_T_WINDOW_TTC: tuple[float, float] = tuple(_heatmap_cfg.t_window_ttc)
HEATMAP_T_WINDOW_ONSET: tuple[float, float] = tuple(_heatmap_cfg.t_window_onset)
HEATMAP_T_BIN_S: float = float(_heatmap_cfg.t_bin_s)
HEATMAP_GAMMA: float = float(_heatmap_cfg.gamma)

# ═══════════════════════════════════════════════════════════════════════
# Backward Compatibility Aliases
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "RADIUS_MM",
    "TRAJECTORY_MAX_RADIUS_MM",
    "TRAJECTORY_STEP_MM",
    "SPEED_WINDOW_MS",
    "LEGACY_TRIAL_DURATION_MS",
    "HEATMAP_VMAX",
    "HEATMAP_SPEED_MAX",
    "HEATMAP_T_WINDOW_TTC",
    "HEATMAP_T_WINDOW_ONSET",
    "HEATMAP_T_BIN_S",
    "HEATMAP_GAMMA",
]
