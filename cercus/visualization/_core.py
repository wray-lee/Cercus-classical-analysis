"""
Cercus Framework — Core Drawing Helpers
=======================================
Shared matplotlib helpers used across all visualization modules.
Trajectory integration physics now lives in
cercus/core/kinematics/trajectory_integration.py.
"""

from __future__ import annotations

import logging

import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cercus.core.kinematics.trajectory_integration import (  # noqa: F401
    body_to_traj,
    build_angular_peak_dz_mask,
    integrate_body_trajectory,
)
from pipeline.constants import (
    COLOR_LEFT,
    COLOR_OSCI_HW,
    COLOR_OSCI_VIS,
    COLOR_RIGHT,
    ESCAPE_START_THRESHOLD,
    ESCAPE_VMAX_THRESHOLD,
)

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Drawing Helpers
# ══════════════════════════════════════════════════════════════════════


def draw_standardized_grid(
    ax: plt.Axes, max_radius: float = 50.0, step: float = 10.0
) -> None:
    """Draw standardized physical coordinate system with concentric distance rings."""
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_aspect("equal")

    ax.axhline(0, color="black", lw=0.6, alpha=0.5, zorder=1)
    ax.axvline(0, color="black", lw=0.6, alpha=0.5, zorder=1)

    for r in np.arange(step, max_radius + step, step):
        circle = plt.Circle(
            (0, 0), r, color="gray", fill=False, ls="--", lw=0.5, alpha=1, zorder=1
        )
        ax.add_patch(circle)
        radius_dis = ax.text(
            r * 0.707,
            r * 0.707,
            f"{int(r)} mm",
            color="gray",
            fontsize=6,
            ha="left",
            va="bottom",
            alpha=1,
            fontweight="regular",
        )
        radius_dis.set_path_effects(
            [path_effects.withStroke(linewidth=0.5, foreground="#0F172A", alpha=1)]
        )

    ax.set_xticks([])
    ax.set_yticks([])


def draw_side_arrows(
    ax: plt.Axes,
    left_color: str = COLOR_LEFT,
    right_color: str = COLOR_RIGHT,
) -> None:
    """Draw minimalist vector arrows on LEFT and RIGHT edges."""
    arrow_style = dict(
        arrowstyle="]->, lengthA=0.01, widthA=10",
        color=None,
        lw=1.0,
        mutation_scale=8,
    )

    ax.annotate(
        "",
        xy=(0.04, 0.5),
        xytext=(-0.02, 0.5),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={**arrow_style, "color": left_color},
    )
    ax.text(
        0.09,
        0.45,
        "Left Stimulus",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color=left_color,
    )

    ax.annotate(
        "",
        xy=(0.96, 0.5),
        xytext=(1.02, 0.5),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={**arrow_style, "color": right_color},
    )
    ax.text(
        0.91,
        0.45,
        "Right Stimulus",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color=right_color,
    )


def add_threshold_lines(ax: plt.Axes) -> None:
    """Draw horizontal threshold lines at ESCAPE_VMAX (50) and ESCAPE_START (10)."""
    ax.axhline(
        y=ESCAPE_VMAX_THRESHOLD,
        color="k",
        linestyle="--",
        linewidth=0.75,
        alpha=0.7,
    )
    ax.text(
        ax.get_xlim()[1] * 0.98,
        ESCAPE_VMAX_THRESHOLD + 1.0,
        f"Vmax Threshold ({ESCAPE_VMAX_THRESHOLD:.0f})",
        ha="right",
        va="bottom",
        fontsize=6,
        color="k",
        alpha=0.7,
    )

    ax.axhline(
        y=ESCAPE_START_THRESHOLD,
        color="0.5",
        linestyle="--",
        linewidth=0.5,
        alpha=0.5,
    )
    ax.text(
        ax.get_xlim()[1] * 0.98,
        ESCAPE_START_THRESHOLD + 1.0,
        f"Start Threshold ({ESCAPE_START_THRESHOLD:.0f})",
        ha="right",
        va="bottom",
        fontsize=6,
        color="0.5",
        alpha=0.5,
    )


def draw_oscilloscope_channels(ax: plt.Axes, df: pd.DataFrame, cond: str) -> None:
    """Draw dual-channel oscilloscope waveforms (visual + wind) on *ax*."""
    vis_baseline = 1.0
    wind_baseline = 3.0

    if "visual" in cond.lower() or "looming" in cond.lower():
        stim_t_rel = df["t_rel"]
        t_loom_start = stim_t_rel.min() if not stim_t_rel.empty else df["t_rel"].min()
        t_loom = np.array([t_loom_start, 0.0])
        ax.fill_between(
            t_loom,
            vis_baseline,
            vis_baseline + 1.0,
            step="mid",
            color=COLOR_OSCI_VIS,
            alpha=0.6,
            label="Visual (looming)",
        )

    subset = df[df["type"] == cond]
    if not subset.empty:
        first_tid = subset["global_trial_id"].iloc[0]
        grp = subset[subset["global_trial_id"] == first_tid].sort_values("t_rel")

        if (
            "wind" in cond.lower()
            or "puff" in cond.lower()
            or grp["stim_state"].max() > 0
        ):
            t_wind = grp["t_rel"].values
            stim = grp["stim_state"].values.astype(float)
            dt_last = t_wind[-1] - t_wind[-2] if len(t_wind) > 1 else 1.0
            t_wind_ext = np.append(t_wind, t_wind[-1] + dt_last)
            stim_ext = np.append(stim, stim[-1])
            ax.fill_between(
                t_wind_ext,
                wind_baseline,
                wind_baseline + stim_ext,
                step="post",
                color=COLOR_OSCI_HW,
                alpha=0.6,
                label="Wind (stim_state)",
            )

    ax.set_ylim(0, 5)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("Time relative to TTC (ms)")
    ax.grid(False)


# ══════════════════════════════════════════════════════════════════════
# Backward-compatibility aliases (for old-style pipeline.visualization imports)
# ══════════════════════════════════════════════════════════════════════

_draw_standardized_grid = draw_standardized_grid
_draw_side_arrows = draw_side_arrows
_add_threshold_lines = add_threshold_lines
_draw_oscilloscope_channels = draw_oscilloscope_channels
