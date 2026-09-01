"""
Cercus Framework — Trajectory Overlay Plots
============================================
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.constants import (
    COLOR_CONTROL,
    COLOR_LEFT,
    COLOR_RIGHT,
    DZ_INTEGRATION_RANGE,
    TRAJECTORY_MAX_RADIUS_MM,
    TRAJECTORY_STEP_MM,
    TRAJ_USE_ESCAPE_ONSET_HEADING,
    TRAJ_USE_ESCAPE_ONSET_ONLY_XY,
    TRAJ_USE_RIGID_ROTATION,
    TRAJ_USE_Z_DEGREE,
    _get_unified_side,
)
from cercus.visualization._core import (
    compute_trajectory_masks,
    draw_side_arrows,
    draw_standardized_grid,
)

log = logging.getLogger(__name__)

# Multisensory comparison color
COLOR_MULTISENSORY = "#009E73"


def plot_multisensory_trajectory_comparison(
    df_bv: pd.DataFrame,
    df_bw: pd.DataFrame,
    df_ms: pd.DataFrame,
    figsize: tuple[float, float] = (6.0, 6.0),
    alpha: float = 0.8,
    lw: float = 0.5,
    bv_color: str = COLOR_LEFT,
    bw_color: str = COLOR_RIGHT,
    ms_color: str = COLOR_MULTISENSORY,
    USE_Z_DEGREE_TO_DRAW_TRAJECTORY: bool = TRAJ_USE_Z_DEGREE,
    USE_RIGID_ROTATION: bool = TRAJ_USE_RIGID_ROTATION,
    USE_ESCAPE_ONSET_ONLY_XY: bool = TRAJ_USE_ESCAPE_ONSET_ONLY_XY,
    dz_integration_range: str = DZ_INTEGRATION_RANGE,
) -> plt.Figure:
    """Multisensory comparison: baselines mirrored to −x, multisensory to +x.

    Parameters
    ----------
    df_bv : baseline-visual trials
    df_bw : baseline-wind trials
    df_ms : multisensory trials
    """
    fig, ax = plt.subplots(figsize=figsize)

    def _plot_dataset(df: pd.DataFrame, color: str, mirror_to_neg: bool) -> int:
        """Plot all trials in *df*. Returns count drawn.

        mirror_to_neg=True  → all trajectories end up on −x side
        mirror_to_neg=False → all trajectories end up on +x side
        """
        group_cols = (
            ["subject_id", "global_trial_id"]
            if "subject_id" in df.columns
            else ["global_trial_id"]
        )
        count = 0
        for _keys, grp in df.groupby(group_cols):
            grp = grp.sort_values("t_rel")

            _onset_ms = (
                grp["interval_onset_ms"].iloc[0]
                if "interval_onset_ms" in grp.columns
                else np.nan
            )
            _offset_ms = (
                grp["interval_offset_ms"].iloc[0]
                if "interval_offset_ms" in grp.columns
                else np.nan
            )

            result = compute_trajectory_masks(
                grp, _onset_ms, _offset_ms,
                use_escape_onset_only_xy=USE_ESCAPE_ONSET_ONLY_XY,
                use_escape_onset_heading=TRAJ_USE_ESCAPE_ONSET_HEADING,
                use_z_degree=USE_Z_DEGREE_TO_DRAW_TRAJECTORY,
                use_rigid_rotation=USE_RIGID_ROTATION,
                dz_integration_range=dz_integration_range,
            )
            if result is None:
                continue
            traj_x, traj_y, *_rest = result
            if traj_x is None or len(traj_x) < 2:
                continue

            traj_x = np.asarray(traj_x, dtype=float)
            traj_y = np.asarray(traj_y, dtype=float)

            # Determine original side and mirror as needed
            ss = _get_unified_side(grp)
            if mirror_to_neg:
                # Baselines → all to −x.  Right-side trials need x-flip.
                if ss == "right":
                    traj_x = -traj_x
            else:
                # Multisensory → all to +x.  Left-side trials need x-flip.
                if ss == "left":
                    traj_x = -traj_x

            ax.plot(traj_x, traj_y, color=color, alpha=alpha, lw=lw)
            count += 1
        return count

    n_bv = _plot_dataset(df_bv, bv_color, mirror_to_neg=True)
    n_bw = _plot_dataset(df_bw, bw_color, mirror_to_neg=True)
    n_ms = _plot_dataset(df_ms, ms_color, mirror_to_neg=False)

    draw_standardized_grid(
        ax, max_radius=TRAJECTORY_MAX_RADIUS_MM, step=TRAJECTORY_STEP_MM
    )

    # Legend: simple color patches
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=bv_color, lw=1.5, label=f"Baseline Visual (n={n_bv})"),
        Line2D([0], [0], color=bw_color, lw=1.5, label=f"Baseline Wind (n={n_bw})"),
        Line2D([0], [0], color=ms_color, lw=1.5, label=f"Multisensory (n={n_ms})"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.8)

    # Side labels
    ax.text(
        0.08, 0.5, "← Baselines",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=8, color="0.4", rotation=90,
    )
    ax.text(
        0.92, 0.5, "Multisensory →",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=8, color=ms_color, rotation=-90,
    )

    ax.set_title("Multisensory Trajectory Comparison", fontweight="bold")
    fig.tight_layout(pad=1.0)
    return fig


def plot_trajectory_overlay(
    df: pd.DataFrame,
    alpha: float = 0.7,
    control_type: str = "baseline_visual_test",
    left_color: str = COLOR_LEFT,
    right_color: str = COLOR_RIGHT,
    figsize_per_ax: tuple[float, float] = (4.0, 4.0),
    USE_Z_DEGREE_TO_DRAW_TRAJECTORY: bool = TRAJ_USE_Z_DEGREE,
    USE_RIGID_ROTATION: bool = TRAJ_USE_RIGID_ROTATION,
    USE_ESCAPE_ONSET_ONLY_XY: bool = TRAJ_USE_ESCAPE_ONSET_ONLY_XY,
    dz_integration_range: str = DZ_INTEGRATION_RANGE,
) -> plt.Figure:
    """One subplot per trial type. Left stimuli in NPG blue, right in NPG red."""
    all_types = sorted(df["type"].dropna().unique())
    if not all_types:
        log.warning("No trial types found for trajectory overlay.")
        fig, _ = plt.subplots(figsize=figsize_per_ax)
        return fig

    n = len(all_types)
    fig, axes = plt.subplots(
        1, n, figsize=(figsize_per_ax[0] * n, figsize_per_ax[1]), squeeze=False
    )
    axes = axes[0]

    for idx, ttype in enumerate(all_types):
        ax = axes[idx]
        subset = df[df["type"] == ttype]

        for _tid, grp in subset.groupby("global_trial_id"):
            grp = grp.sort_values("t_rel")

            _onset_ms = (
                grp["interval_onset_ms"].iloc[0]
                if "interval_onset_ms" in grp.columns
                else np.nan
            )
            _offset_ms = (
                grp["interval_offset_ms"].iloc[0]
                if "interval_offset_ms" in grp.columns
                else np.nan
            )

            result = compute_trajectory_masks(
                grp, _onset_ms, _offset_ms,
                use_escape_onset_only_xy=USE_ESCAPE_ONSET_ONLY_XY,
                use_escape_onset_heading=TRAJ_USE_ESCAPE_ONSET_HEADING,
                use_z_degree=USE_Z_DEGREE_TO_DRAW_TRAJECTORY,
                use_rigid_rotation=USE_RIGID_ROTATION,
                dz_integration_range=dz_integration_range,
            )
            if result is None:
                continue
            rot_x, rot_y, *_rest = result
            if rot_x is None or len(rot_x) < 2:
                continue

            ss = _get_unified_side(grp)
            if ttype == control_type:
                color = COLOR_CONTROL
            elif ss == "left":
                color = left_color
            elif ss == "right":
                color = right_color
            else:
                color = COLOR_CONTROL

            ax.plot(rot_x, rot_y, color=color, alpha=alpha, lw=0.8)

        ax.set_title(ttype, fontweight="bold")
        draw_standardized_grid(
            ax, max_radius=TRAJECTORY_MAX_RADIUS_MM, step=TRAJECTORY_STEP_MM
        )
        draw_side_arrows(ax, left_color=left_color, right_color=right_color)

    fig.tight_layout(pad=1.0)
    return fig


def plot_global_trajectory_overlay_fixed(
    df: pd.DataFrame,
    TRAJECTORY_MAX_RADIUS_MM: float = 200.0,
    TRAJECTORY_STEP_MM: float = 20.0,
    figsize: tuple[float, float] = (5.0, 5.0),
    alpha: float = 1.0,
    lw: float = 0.3,
    left_color: str = COLOR_LEFT,
    right_color: str = COLOR_RIGHT,
    USE_Z_DEGREE_TO_DRAW_TRAJECTORY: bool = TRAJ_USE_Z_DEGREE,
    USE_RIGID_ROTATION: bool = TRAJ_USE_RIGID_ROTATION,
    USE_ESCAPE_ONSET_ONLY_XY: bool = TRAJ_USE_ESCAPE_ONSET_ONLY_XY,
    dz_integration_range: str = DZ_INTEGRATION_RANGE,
) -> plt.Figure:
    """Unified trajectory overlay — all paradigms on one axes."""
    fig, ax = plt.subplots(figsize=figsize)

    group_cols = (
        ["subject_id", "global_trial_id"]
        if "subject_id" in df.columns
        else ["global_trial_id"]
    )

    for _keys, grp in df.groupby(group_cols):
        grp = grp.sort_values("t_rel")

        _onset_ms = (
            grp["interval_onset_ms"].iloc[0]
            if "interval_onset_ms" in grp.columns
            else np.nan
        )
        _offset_ms = (
            grp["interval_offset_ms"].iloc[0]
            if "interval_offset_ms" in grp.columns
            else np.nan
        )

        result = compute_trajectory_masks(
            grp, _onset_ms, _offset_ms,
            use_escape_onset_only_xy=USE_ESCAPE_ONSET_ONLY_XY,
            use_escape_onset_heading=TRAJ_USE_ESCAPE_ONSET_HEADING,
            use_z_degree=USE_Z_DEGREE_TO_DRAW_TRAJECTORY,
            use_rigid_rotation=USE_RIGID_ROTATION,
            dz_integration_range=dz_integration_range,
        )
        if result is None:
            continue
        traj_x, traj_y, *_rest = result
        if traj_x is None:
            continue

        ss = _get_unified_side(grp)
        if ss == "left":
            color = left_color
        elif ss == "right":
            color = right_color
        else:
            color = COLOR_CONTROL

        ax.plot(traj_x, traj_y, color=color, alpha=alpha, lw=lw)

    draw_standardized_grid(
        ax, max_radius=TRAJECTORY_MAX_RADIUS_MM, step=TRAJECTORY_STEP_MM
    )
    draw_side_arrows(ax)

    n_trials = df.groupby(group_cols).ngroups
    ax.set_title(
        f"Unified Trajectory Overlay(n={n_trials} trials)", fontweight="bold"
    )

    fig.tight_layout(pad=1.0)
    return fig