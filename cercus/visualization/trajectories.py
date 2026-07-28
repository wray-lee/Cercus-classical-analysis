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
    RADIUS_MM,
    TRAJECTORY_MAX_RADIUS_MM,
    TRAJECTORY_STEP_MM,
    TRAJ_USE_ESCAPE_ONSET_HEADING,
    TRAJ_USE_ESCAPE_ONSET_ONLY_XY,
    TRAJ_USE_RIGID_ROTATION,
    TRAJ_USE_Z_DEGREE,
    _get_unified_side,
)
from cercus.visualization._core import (
    body_to_traj,
    build_angular_peak_dz_mask,
    draw_side_arrows,
    draw_standardized_grid,
)

log = logging.getLogger(__name__)


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
            t_vals = grp["t_rel"].values

            _response_type = (
                grp["response_type"].iloc[0]
                if "response_type" in grp.columns
                else ""
            )
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

            _is_escape = (
                _response_type in ("Escape", "PreWalk")
                and pd.notna(_onset_ms)
                and pd.notna(_offset_ms)
            )
            if _is_escape:
                onset_idx = int(np.argmin(np.abs(t_vals - _onset_ms)))
                offset_idx = int(np.argmin(np.abs(t_vals - _offset_ms)))
                if onset_idx >= offset_idx:
                    offset_idx = min(onset_idx + 1, len(t_vals) - 1)
                escape_mask = np.zeros(len(t_vals), dtype=bool)
                escape_mask[onset_idx:offset_idx] = True
            else:
                escape_mask = None

            full_mask = np.ones(len(t_vals), dtype=bool)
            mask_xy = (
                escape_mask
                if (USE_ESCAPE_ONSET_ONLY_XY and escape_mask is not None)
                else full_mask
            )

            _macro_yaw_override = None
            if _is_escape and dz_integration_range == "escape_interval":
                mask_z = escape_mask
            elif _is_escape and dz_integration_range == "trial_to_onset":
                mask_z = np.zeros(len(t_vals), dtype=bool)
                mask_z[:onset_idx] = True
            elif _is_escape and dz_integration_range == "escape_angular_peak":
                av = (
                    grp["angular_velocity"].values
                    if "angular_velocity" in grp.columns
                    else None
                )
                if av is not None:
                    mask_z = build_angular_peak_dz_mask(
                        av, onset_idx, offset_idx, len(t_vals)
                    )
                else:
                    mask_z = escape_mask
            elif _is_escape and dz_integration_range == "escape_onset_heading":
                mask_z = escape_mask
                _macro_yaw_override = (
                    np.cumsum(grp["dz"].fillna(0).values)[onset_idx] / RADIUS_MM
                )
            else:
                mask_z = full_mask

            _heading_dz_mask = (
                escape_mask if (_is_escape and escape_mask is not None) else None
            )
            _heading_offset = 0.0
            if (
                TRAJ_USE_ESCAPE_ONSET_HEADING
                and _is_escape
                and dz_integration_range != "escape_onset_heading"
            ):
                _heading_offset = (
                    np.cumsum(grp["dz"].fillna(0).values)[onset_idx] / RADIUS_MM
                )

            if not np.any(mask_xy):
                continue

            rot_x, rot_y = body_to_traj(
                grp,
                mask_xy,
                use_z=USE_Z_DEGREE_TO_DRAW_TRAJECTORY,
                use_rigid_rotation=USE_RIGID_ROTATION,
                mask_z=mask_z,
                heading_dz_mask=_heading_dz_mask,
                macro_yaw_override=_macro_yaw_override,
                heading_offset=_heading_offset,
            )
            if rot_x is None:
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
        t_vals = grp["t_rel"].values

        _response_type = (
            grp["response_type"].iloc[0] if "response_type" in grp.columns else ""
        )
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

        _is_escape = (
            _response_type in ("Escape", "PreWalk")
            and pd.notna(_onset_ms)
            and pd.notna(_offset_ms)
        )
        if _is_escape:
            onset_idx = int(np.argmin(np.abs(t_vals - _onset_ms)))
            offset_idx = int(np.argmin(np.abs(t_vals - _offset_ms)))
            if onset_idx >= offset_idx:
                offset_idx = min(onset_idx + 1, len(t_vals) - 1)
            escape_mask = np.zeros(len(t_vals), dtype=bool)
            escape_mask[onset_idx:offset_idx] = True
        else:
            escape_mask = None

        full_mask = np.ones(len(t_vals), dtype=bool)
        mask_xy = (
            escape_mask
            if (USE_ESCAPE_ONSET_ONLY_XY and escape_mask is not None)
            else full_mask
        )

        _macro_yaw_override = None
        if _is_escape and dz_integration_range == "escape_interval":
            mask_z = escape_mask
        elif _is_escape and dz_integration_range == "trial_to_onset":
            mask_z = np.zeros(len(t_vals), dtype=bool)
            mask_z[:onset_idx] = True
        elif _is_escape and dz_integration_range == "escape_angular_peak":
            av = (
                grp["angular_velocity"].values
                if "angular_velocity" in grp.columns
                else None
            )
            if av is not None:
                mask_z = build_angular_peak_dz_mask(
                    av, onset_idx, offset_idx, len(t_vals)
                )
            else:
                mask_z = escape_mask
        elif _is_escape and dz_integration_range == "escape_onset_heading":
            mask_z = escape_mask
            _macro_yaw_override = (
                np.cumsum(grp["dz"].fillna(0).values)[onset_idx] / RADIUS_MM
            )
        else:
            mask_z = full_mask

        _heading_dz_mask = (
            escape_mask if (_is_escape and escape_mask is not None) else None
        )
        _heading_offset = 0.0
        if (
            TRAJ_USE_ESCAPE_ONSET_HEADING
            and _is_escape
            and dz_integration_range != "escape_onset_heading"
        ):
            _heading_offset = (
                np.cumsum(grp["dz"].fillna(0).values)[onset_idx] / RADIUS_MM
            )

        if not np.any(mask_xy):
            continue

        traj_x, traj_y = body_to_traj(
            grp,
            mask_xy,
            use_z=USE_Z_DEGREE_TO_DRAW_TRAJECTORY,
            use_rigid_rotation=USE_RIGID_ROTATION,
            mask_z=mask_z,
            heading_dz_mask=_heading_dz_mask,
            macro_yaw_override=_macro_yaw_override,
            heading_offset=_heading_offset,
        )
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