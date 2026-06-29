"""
Cercus Framework — Shared Constants, Colors & Publication Style
===============================================================
Centralised configuration for thresholds, geometry, and Nature/Science rcParams.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Publication-Grade Global Style (Nature / Science / Cell)
# ──────────────────────────────────────────────────────────────────────


def _apply_publication_style() -> None:
    """Inject Nature/Science/Cell compliant rcParams."""
    rc = plt.rcParams
    rc["font.family"] = "sans-serif"
    rc["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    rc["svg.fonttype"] = "none"
    rc["pdf.fonttype"] = 42
    rc["font.size"] = 7
    rc["axes.titlesize"] = 9
    rc["axes.labelsize"] = 8
    rc["legend.fontsize"] = 7
    rc["xtick.labelsize"] = 7
    rc["ytick.labelsize"] = 7
    rc["lines.linewidth"] = 1.0
    rc["axes.linewidth"] = 0.75
    rc["axes.spines.top"] = False
    rc["axes.spines.right"] = False
    rc["xtick.direction"] = "in"
    rc["ytick.direction"] = "in"
    rc["xtick.major.size"] = 3
    rc["ytick.major.size"] = 3
    rc["xtick.major.width"] = 0.75
    rc["ytick.major.width"] = 0.75
    rc["xtick.minor.size"] = 1.5
    rc["ytick.minor.size"] = 1.5
    rc["legend.frameon"] = False
    rc["legend.borderaxespad"] = 0
    rc["figure.dpi"] = 150
    rc["savefig.dpi"] = 300
    rc["savefig.transparent"] = True


# ──────────────────────────────────────────────────────────────────────
# Colour Palette (Lancet / Cell style)
# ──────────────────────────────────────────────────────────────────────

COLOR_LEFT: str = "#00468B"          # Navy Blue
COLOR_RIGHT: str = "#ED0000"         # Crimson Red
COLOR_CONTROL: str = "#7C878E"       # Slate Grey
COLOR_OSCI_VIS: str = "#ADB6B6"      # Cool Grey (visual stimulus background)
COLOR_OSCI_HW: str = "#E69F00"       # Sand Orange (hardware stimulus background)
COLOR_ESCAPE: str = "#ED0000"
COLOR_PREWALK: str = "#00468B"
COLOR_NO_RESPONSE: str = "#7C878E"


# ──────────────────────────────────────────────────────────────────────
# Event / Trial Metadata Keys
# ──────────────────────────────────────────────────────────────────────

DETAILS_KEYS: tuple[str, ...] = (
    "type",
    "target_ttc_ms",
    "wind_dir",
    "screen_side",
    "lv_ratio_ms",
    "init_half_angle_deg",
    "direction",
    "side",
)


# ──────────────────────────────────────────────────────────────────────
# Geometry & Timing Constants
# ──────────────────────────────────────────────────────────────────────

SPEED_WINDOW_MS: float = 100.0
SCALE_BAR_MM: float = 5.0
LEGACY_TRIAL_DURATION_MS: float = 5829.6
RADIUS_MM: float = 30.0


# ──────────────────────────────────────────────────────────────────────
# Physical-Threshold Constants
# ──────────────────────────────────────────────────────────────────────

ESCAPE_VMAX_THRESHOLD: float = 50.0       # mm/s — burst floor for valid escape
ESCAPE_START_THRESHOLD: float = 10.0      # mm/s — latency onset anchor
PREWALK_THRESHOLD: float = 10.0           # mm/s — pre-stimulus spontaneous activity
PREWALK_WINDOW_MS: float = 1000.0         # ms — pre-stimulus validation window
POST_STIM_BUFFER_MS: float = 50.0         # ms — post-stimulus tail buffer
ESCAPE_WINDOW_MS: float = 250.0           # ms — post-stimulus burst detection window


# ──────────────────────────────────────────────────────────────────────
# Trajectory Plot Constants
# ──────────────────────────────────────────────────────────────────────

TRAJECTORY_MAX_RADIUS_MM: float = 80.0
TRAJECTORY_STEP_MM: float = 10.0


# ──────────────────────────────────────────────────────────────────────
# Angular Velocity Constants
# ──────────────────────────────────────────────────────────────────────

ANGULAR_VELOCITY_WINDOW_MS: float = 100.0   # smoothing window for angular velocity


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _get_unified_side(data: pd.Series | pd.DataFrame) -> str:
    """Extract and normalize direction identifier to ``'left'`` or ``'right'``."""
    row = data.iloc[0] if isinstance(data, pd.DataFrame) else data
    for col in ["screen_side", "wind_dir", "direction", "side"]:
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip().lower()
            if val in ("left", "l"):
                return "left"
            if val in ("right", "r"):
                return "right"
    return ""
