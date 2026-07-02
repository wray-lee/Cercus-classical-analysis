"""
Cercus Framework — Shared Constants, Colors & Publication Style
===============================================================
Centralised configuration for thresholds, geometry, and Nature/Science rcParams.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

_log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# YAML Configuration Loader
# ──────────────────────────────────────────────────────────────────────

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

TRAJ_USE_Z_DEGREE: bool = bool(_traj_cfg.get("use_z_degree_to_draw", True))
TRAJ_USE_RIGID_ROTATION: bool = bool(_traj_cfg.get("use_rigid_rotation", False))
TRAJ_USE_ESCAPE_ONSET_ONLY: bool = bool(_traj_cfg.get("use_escape_onset_only", True))


# ──────────────────────────────────────────────────────────────────────
# Publication-Grade Global Style (Nature / Science / Cell)
# ──────────────────────────────────────────────────────────────────────


def _apply_publication_style() -> None:
    """Inject publication-grade rcParams for Nature/Science/Cell standards.

    Covers four global constraints:
    1. Canvas: dpi=300, white background, hidden top/right spines.
    2. Typography: Helvetica/Arial sans-serif, hierarchical font sizes.
    3. Rendering: tight_layout-compatible defaults, high-contrast edges.
    4. Colour: NPG palette defined separately as ``NPG_PALETTE``.
    """
    rc = plt.rcParams
    # ── Font family (sans-serif, Helvetica/Arial preferred) ──
    rc["font.family"] = "sans-serif"
    rc["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    rc["svg.fonttype"] = "none"
    rc["pdf.fonttype"] = 42

    # ── Hierarchical font sizes ──
    rc["font.size"] = 10
    rc["axes.titlesize"] = 14
    rc["axes.titleweight"] = "bold"
    rc["axes.labelsize"] = 12
    rc["axes.labelweight"] = "bold"
    rc["legend.fontsize"] = 10
    rc["xtick.labelsize"] = 10
    rc["ytick.labelsize"] = 10

    # ── Line widths & spine visibility ──
    rc["lines.linewidth"] = 1.5
    rc["axes.linewidth"] = 1.2
    rc["axes.spines.top"] = False
    rc["axes.spines.right"] = False
    rc["axes.edgecolor"] = "#000000"

    # ── Ticks ──
    rc["xtick.direction"] = "out"
    rc["ytick.direction"] = "out"
    rc["xtick.major.size"] = 4
    rc["ytick.major.size"] = 4
    rc["xtick.major.width"] = 1.2
    rc["ytick.major.width"] = 1.2
    rc["xtick.minor.size"] = 2
    rc["ytick.minor.size"] = 2

    # ── Legend ──
    rc["legend.frameon"] = False
    rc["legend.borderaxespad"] = 0

    # ── Canvas: high DPI, transparent background ──
    rc["figure.dpi"] = 150
    rc["savefig.dpi"] = 300
    rc["savefig.bbox"] = "tight"
    rc["savefig.transparent"] = True
    # rc["figure.facecolor"] = "#FFFFFF"
    # rc["axes.facecolor"] = "#FFFFFF"
    # rc["savefig.facecolor"] = "#FFFFFF"
    rc["axes.grid"] = False


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


# ── NPG (Nature Publishing Group) colour palette ─────────────────────
# Colour-blind friendly, widely used in top-tier journals.

NPG_PALETTE: list[str] = [
    "#E64B35",   # Red
    "#4DBBD5",   # Cyan
    "#00A087",   # Teal
    "#3C5488",   # Navy Blue
    "#F39B7F",   # Salmon
    "#8491B4",   # Slate Blue
    "#91D1C2",   # Mint
    "#DC0000",   # Dark Red
]


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
