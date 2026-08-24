"""
Cercus Framework — Shared Constants, Colors & Publication Style
===============================================================
*** DEPRECATED *** — Import from ``cercus.constants`` or ``cercus.config`` instead.
Retained for backward compatibility.
"""

from __future__ import annotations

import enum
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

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
from cercus.constants.geometry import (
    HEATMAP_SPEED_MAX,
    HEATMAP_VMAX,
    LEGACY_TRIAL_DURATION_MS,
    RADIUS_MM,
    SPEED_WINDOW_MS,
    TRAJECTORY_MAX_RADIUS_MM,
    TRAJECTORY_STEP_MM,
)
from cercus.constants.thresholds import (
    ESCAPE_START_THRESHOLD,
    ESCAPE_VMAX_THRESHOLD,
    ESCAPE_WINDOW_MS,
    POST_STIM_BUFFER_MS,
    PREWALK_THRESHOLD,
    PREWALK_WINDOW_MS,
)

_log = logging.getLogger(__name__)

# ── YAML Configuration Loader ──

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


def _load_config() -> dict:
    """Load ``config.yaml`` from the project root, with silent fallback."""
    if not _CONFIG_PATH.is_file():
        _log.info("config.yaml not found at %s — using defaults.", _CONFIG_PATH)
        return {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        _log.info("Loaded config from %s", _CONFIG_PATH)
        return cfg
    except Exception as exc:
        _log.warning("Failed to parse config.yaml: %s — using defaults.", exc)
        return {}


_cfg = _load_config()
_traj_cfg = _cfg.get("trajectory", {})
_escape_cfg = _cfg.get("escape", {})
_viz_cfg = _cfg.get("visualization", {})


# ── Visualization Style Enum ──

class BarLabelStyle(enum.Enum):
    INLINE = "inline"
    AXIS = "axis"


BAR_LABEL_STYLE: BarLabelStyle = BarLabelStyle(_viz_cfg.get("bar_label_style", "inline"))


# ── Trajectory Configuration ──

TRAJ_USE_Z_DEGREE: bool = bool(_traj_cfg.get("use_z_degree_to_draw", True))
TRAJ_USE_RIGID_ROTATION: bool = bool(_traj_cfg.get("use_rigid_rotation", False))
TRAJ_USE_ESCAPE_ONSET_HEADING: bool = bool(_traj_cfg.get("use_escape_onset_heading", True))
TRAJ_USE_ESCAPE_ONSET_ONLY_XY: bool = bool(_traj_cfg.get("use_escape_onset_only_xy", True))
TRAJ_USE_ANGULAR_VELOCITY_OFFSET: bool = bool(_traj_cfg.get("use_angular_velocity_offset", False))
# ponytail: global ring-angle delta fixing wind nozzle offset; 0 is legacy, non-zero rotates (dx,dy) CCW by delta (arena CW)
try:
    _wind_off = float(_traj_cfg.get("wind_angle_offset_deg", 0.0))
except Exception:
    _wind_off = 0.0
WIND_ANGLE_OFFSET_DEG: float = _wind_off

# ponytail: array controlling which analysis modules apply wind angle correction (e.g. ["trajectory"], ["polar"], ["individual"])
_raw_targets = _traj_cfg.get("wind_angle_offset_targets", ["trajectory", "polar", "individual"])
if isinstance(_raw_targets, str):
    _raw_targets = [_raw_targets]
WIND_ANGLE_OFFSET_TARGETS: set[str] = {str(t).strip().lower() for t in _raw_targets}

_DZ_RANGE_VALID = {"full_trial", "escape_interval", "trial_to_onset", "escape_angular_peak", "escape_onset_heading", "peak_bracket"}
DZ_INTEGRATION_RANGE: str = _traj_cfg.get("dz_integration_range", "escape_interval")
if DZ_INTEGRATION_RANGE not in _DZ_RANGE_VALID:
    _log.warning("Invalid dz_integration_range=%r, falling back to 'escape_interval'", DZ_INTEGRATION_RANGE)
    DZ_INTEGRATION_RANGE = "escape_interval"


# ── Publication-Grade Global Style ──


def _apply_publication_style() -> None:
    """Inject publication-grade rcParams for Nature/Science/Cell standards."""
    rc = plt.rcParams
    rc["font.family"] = "sans-serif"
    rc["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    rc["svg.fonttype"] = "none"
    rc["pdf.fonttype"] = 42

    rc["font.size"] = 10
    rc["axes.titlesize"] = 14
    rc["axes.titleweight"] = "bold"
    rc["axes.labelsize"] = 12
    rc["axes.labelweight"] = "bold"
    rc["legend.fontsize"] = 10
    rc["xtick.labelsize"] = 10
    rc["ytick.labelsize"] = 10

    rc["lines.linewidth"] = 1.5
    rc["axes.linewidth"] = 1.2
    rc["axes.spines.top"] = False
    rc["axes.spines.right"] = False
    rc["axes.edgecolor"] = "#000000"

    rc["xtick.direction"] = "out"
    rc["ytick.direction"] = "out"
    rc["xtick.major.size"] = 4
    rc["ytick.major.size"] = 4
    rc["xtick.major.width"] = 1.2
    rc["ytick.major.width"] = 1.2
    rc["xtick.minor.size"] = 2
    rc["ytick.minor.size"] = 2

    rc["legend.frameon"] = False
    rc["legend.borderaxespad"] = 0

    rc["figure.dpi"] = 150
    rc["savefig.dpi"] = 300
    rc["savefig.bbox"] = "tight"
    rc["savefig.transparent"] = True
    rc["axes.grid"] = False


# ── Event / Trial Metadata Keys ──

DETAILS_KEYS: tuple[str, ...] = (
    "type", "target_ttc_ms", "wind_dir", "screen_side",
    "lv_ratio_ms", "init_half_angle_deg", "direction", "side",
)


# ── Helpers ──


def _get_unified_side(data) -> str:
    """Extract and normalize direction identifier to 'left' or 'right'."""
    if isinstance(data, pd.DataFrame):
        row = data.iloc[0]
    elif isinstance(data, pd.Series):
        row = data
    else:
        return ""
    # ponytail: wind trials must use wind_dir first; visual screen_side can be stale/NaN for wind-only hardware offset calibration
    _ttype = ""
    if "type" in row:
        try:
            _ttype = str(row["type"]).lower()
        except Exception:
            _ttype = ""
    _is_wind = "wind" in _ttype
    _priority = ["wind_dir", "screen_side", "direction", "side"] if _is_wind else ["screen_side", "wind_dir", "direction", "side"]
    for col in _priority:
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip().lower()
            if val in ("left", "l"):
                return "left"
            if val in ("right", "r"):
                return "right"
    return ""
