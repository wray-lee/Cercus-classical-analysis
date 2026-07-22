"""
Cercus Framework — Publication-Grade Visualization
===================================================
All Matplotlib plotting functions with Nature/Science global style.
"""

from __future__ import annotations

import logging

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import circmean, gaussian_kde

from .constants import (
    BAR_LABEL_STYLE,
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
    DZ_INTEGRATION_RANGE,
    ESCAPE_START_THRESHOLD,
    ESCAPE_VMAX_THRESHOLD,
    PREWALK_THRESHOLD,
    PREWALK_WINDOW_MS,
    RADIUS_MM,
    TRAJECTORY_MAX_RADIUS_MM,
    TRAJECTORY_STEP_MM,
    TRAJ_USE_ESCAPE_ONSET_HEADING,
    TRAJ_USE_ESCAPE_ONSET_ONLY_XY,
    TRAJ_USE_RIGID_ROTATION,
    TRAJ_USE_Z_DEGREE,
    _get_unified_side,
)
from .kinematics import _refine_offset_by_angular_velocity

log = logging.getLogger(__name__)



# ══════════════════════════════════════════════════════════════════════
# Shared Drawing Helpers
# ══════════════════════════════════════════════════════════════════════


def _draw_standardized_grid(ax: plt.Axes, max_radius: float = 50.0, step: float = 10.0) -> None:
    """Draw standardized physical coordinate system with concentric distance rings."""
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_aspect("equal")

    ax.axhline(0, color="black", lw=0.6, alpha=0.5, zorder=1)
    ax.axvline(0, color="black", lw=0.6, alpha=0.5, zorder=1)

    for r in np.arange(step, max_radius + step, step):
        circle = plt.Circle((0, 0), r, color="gray", fill=False, ls="--", lw=0.5, alpha=1, zorder=1)
        ax.add_patch(circle)
        radius_dis = ax.text(
            r * 0.707, r * 0.707, f"{int(r)} mm",
            color="gray", fontsize=6, ha="left", va="bottom", alpha=1,fontweight="regular",
        )
        radius_dis.set_path_effects([
            path_effects.withStroke(linewidth=0.5, foreground="#0F172A", alpha=1)
        ])

    ax.set_xticks([])
    ax.set_yticks([])


def _draw_side_arrows(ax: plt.Axes, left_color: str = COLOR_LEFT, right_color: str = COLOR_RIGHT) -> None:
    """Draw minimalist vector arrows on LEFT and RIGHT edges."""
    arrow_style = dict(arrowstyle="]->, lengthA=0.01, widthA=10", color=None, lw=1.0, mutation_scale=8)

    ax.annotate(
        "", xy=(0.04, 0.5), xytext=(-0.02, 0.5),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops={**arrow_style, "color": left_color},
    )
    ax.text(0.09, 0.45, "Left Stimulus", transform=ax.transAxes,
            ha="center", va="top", fontsize=7, color=left_color)

    ax.annotate(
        "", xy=(0.96, 0.5), xytext=(1.02, 0.5),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops={**arrow_style, "color": right_color},
    )
    ax.text(0.91, 0.45, "Right Stimulus", transform=ax.transAxes,
            ha="center", va="top", fontsize=7, color=right_color)


def _add_threshold_lines(ax: plt.Axes) -> None:
    """Draw horizontal threshold lines at ESCAPE_VMAX (50) and ESCAPE_START (10)."""
    ax.axhline(y=ESCAPE_VMAX_THRESHOLD, color="k", linestyle="--", linewidth=0.75, alpha=0.7)
    ax.text(
        ax.get_xlim()[1] * 0.98, ESCAPE_VMAX_THRESHOLD + 1.0,
        f"Vmax Threshold ({ESCAPE_VMAX_THRESHOLD:.0f})",
        ha="right", va="bottom", fontsize=6, color="k", alpha=0.7,
    )

    ax.axhline(y=ESCAPE_START_THRESHOLD, color="0.5", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.text(
        ax.get_xlim()[1] * 0.98, ESCAPE_START_THRESHOLD + 1.0,
        f"Start Threshold ({ESCAPE_START_THRESHOLD:.0f})",
        ha="right", va="bottom", fontsize=6, color="0.5", alpha=0.5,
    )


def _integrate_body_trajectory(
    burst_dx: np.ndarray,
    burst_dy: np.ndarray,
    burst_dz: np.ndarray,
    *,
    use_heading: bool = True,
    use_rigid_rotation: bool = True,
    macro_yaw_override: float | None = None,
    heading_dz: np.ndarray | None = None,
    heading_offset: float = 0.0,
    src_idx: np.ndarray | None = None,
    dst_idx: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """双阶映射算法核心：保留局部本征曲率的同时，锁定全局逃避散布象限。

    Stage 1 — Build an *inner trajectory* via dynamic dz integration so every
              frame's local curvature and S-turns are preserved.
              Controlled by *use_heading*.
    Stage 2 — Apply curvature-thresholded rigid macro rotation to the curved
              trajectory, producing the correct left/right fan-shaped
              dispersion without flattening the natural bends.
              Controlled by *use_rigid_rotation* (implies Stage 1).

    Parameters
    ----------
    burst_dx, burst_dy, burst_dz : np.ndarray
        Pre-extracted body-frame micro-displacements (already masked/sliced).
    use_heading : bool
        Whether to apply Stage 1 per-frame heading integration from dz.
    use_rigid_rotation : bool
        Whether to apply Stage 2 macro rotation (implies Stage 1).
    macro_yaw_override : float | None
        If provided, overrides the total yaw computed from ``sum(burst_dz)``
        for Stage 2 rigid rotation.  Used by ``escape_onset_heading`` to
        rotate by the cumulative heading at escape onset instead of the
        integrated dz from the selected range.
    heading_dz : np.ndarray | None
        Optional separate dz array for Stage 1 per-frame heading integration.
        When provided, Stage 1 uses this (typically the full escape-interval dz)
        to preserve trajectory curvature, while Stage 2 uses *burst_dz* for
        the rigid macro rotation angle.  Used by ``escape_angular_peak`` to
        decouple curvature from rotation range.
    src_idx, dst_idx : np.ndarray | None
        Frame indices for heading interpolation when dz and dx arrays have
        different lengths.  *src_idx* maps to heading_dz/burst_dz, *dst_idx*
        maps to burst_dx.  Defaults to uniform stretch when None.
    """
    if len(burst_dx) == 0:
        return None, None

    # Stage 1 heading integration is controlled by use_heading (from use_z_degree_to_draw).
    # Stage 2 rigid rotation is controlled by use_rigid_rotation, independently of Stage 1.
    # heading_dz allows decoupling: Stage 1 uses heading_dz for curvature, Stage 2 uses burst_dz for yaw.

    _heading_dz = heading_dz if heading_dz is not None else burst_dz

    if use_heading:
        # ── Stage 1: build curved trajectory via per-frame heading integration ──
        local_heading = np.cumsum(_heading_dz) / RADIUS_MM
        local_heading -= local_heading[0]
        local_heading += heading_offset  # initial heading from trial start to escape onset

        # Interpolate heading onto the xy grid if lengths differ.
        if len(local_heading) != len(burst_dx):
            if src_idx is None:
                src_idx = np.arange(len(_heading_dz))
            if dst_idx is None:
                dst_idx = np.linspace(0, len(_heading_dz) - 1, len(burst_dx))
            if len(src_idx) >= 2:
                local_heading = np.interp(dst_idx, src_idx, local_heading)
            else:
                local_heading = np.full(len(burst_dx), local_heading[0])

        dx_inner = burst_dx * np.cos(local_heading) - burst_dy * np.sin(local_heading)
        dy_inner = burst_dx * np.sin(local_heading) + burst_dy * np.cos(local_heading)

        x_inner = np.cumsum(dx_inner)
        y_inner = np.cumsum(dy_inner)
    else:
        # Straight accumulation without per-frame heading rotation
        x_inner = np.cumsum(burst_dx)
        y_inner = np.cumsum(burst_dy)

    # ── Stage 2: rigid macro rotation (with curvature thresholding) ──
    if use_rigid_rotation:
        total_yaw_rad = np.sum(burst_dz) / RADIUS_MM
        macro_yaw = macro_yaw_override if macro_yaw_override is not None else total_yaw_rad

        # ── Curvature Thresholding ──
        # High-intrinsic-curvature trajectories can suffer start-tangent
        # distortion and quadrant reversal when the full compensation
        # angle is applied.  We attenuate comp_angle in two steps:
        #   1. Gradual decay when |macro_yaw| exceeds MACRO_YAW_THRESHOLD.
        #   2. Hard clamp at COMP_ANGLE_MAX to cap the forced distortion.
        MACRO_YAW_THRESHOLD = np.pi / 2   # 90° — high intrinsic curvature
        COMP_ANGLE_MAX = np.pi / 2        # 90° — max allowed compensation

        comp_angle = macro_yaw

        yaw_abs = abs(macro_yaw)
        if yaw_abs > MACRO_YAW_THRESHOLD:
            decay_factor = max(0.0, 1.0 - (yaw_abs - MACRO_YAW_THRESHOLD) / MACRO_YAW_THRESHOLD)
            comp_angle *= decay_factor

        comp_angle = np.clip(comp_angle, -COMP_ANGLE_MAX, COMP_ANGLE_MAX)

        cos_yaw = np.cos(comp_angle)
        sin_yaw = np.sin(comp_angle)

        traj_x = x_inner * cos_yaw - y_inner * sin_yaw
        traj_y = x_inner * sin_yaw + y_inner * cos_yaw
    else:
        traj_x = x_inner
        traj_y = y_inner

    # ── Origin alignment: start from (0, 0) ──
    traj_x -= traj_x[0]
    traj_y -= traj_y[0]

    return traj_x, traj_y


def _build_angular_peak_dz_mask(
    angular_velocity: np.ndarray,
    onset_idx: int,
    offset_idx: int,
    total_len: int,
) -> np.ndarray:
    """Build a dz mask spanning [onset, angular-velocity peak → zero-crossing].

    Reuses :func:`_refine_offset_by_angular_velocity` to locate the first
    zero-crossing after the angular-velocity peak within the escape interval.
    Only dz frames in this direction-consistent range are included, filtering
    out the rebound phase of the air-floating ball.
    """
    mask = np.zeros(total_len, dtype=bool)
    refined = _refine_offset_by_angular_velocity(
        np.arange(total_len, dtype=float), angular_velocity, onset_idx, offset_idx,
    )
    end = refined if refined is not None else offset_idx
    mask[onset_idx:end + 1] = True
    return mask


def _body_to_traj(
    grp: pd.DataFrame, mask_xy: np.ndarray, *, use_z: bool,
    use_rigid_rotation: bool = False,
    mask_z: np.ndarray | None = None,
    heading_dz_mask: np.ndarray | None = None,
    macro_yaw_override: float | None = None,
    heading_offset: float = 0.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """DataFrame wrapper around :func:`_integrate_body_trajectory`.

    Extracts body-frame dx/dy/dz from *grp* using boolean masks, then
    delegates to the shared integration core.
    """
    if mask_z is None:
        mask_z = mask_xy

    dx_body = -grp["dx"].fillna(0).values
    dy_body = -grp["dy"].fillna(0).values
    dz_body = grp["dz"].fillna(0).values

    burst_dx = dx_body[mask_xy]
    burst_dy = dy_body[mask_xy]
    burst_dz = dz_body[mask_z]

    heading_dz = dz_body[heading_dz_mask] if heading_dz_mask is not None else None

    src_ref = heading_dz_mask if heading_dz_mask is not None else mask_z
    src_idx = np.flatnonzero(src_ref) if len(heading_dz if heading_dz is not None else burst_dz) != len(burst_dx) else None
    dst_idx = np.flatnonzero(mask_xy) if len(heading_dz if heading_dz is not None else burst_dz) != len(burst_dx) else None

    return _integrate_body_trajectory(
        burst_dx, burst_dy, burst_dz,
        use_heading=use_z,
        use_rigid_rotation=use_rigid_rotation,
        macro_yaw_override=macro_yaw_override,
        heading_dz=heading_dz,
        heading_offset=heading_offset,
        src_idx=src_idx, dst_idx=dst_idx,
    )


def _draw_oscilloscope_channels(ax: plt.Axes, df: pd.DataFrame, cond: str) -> None:
    """Draw dual-channel oscilloscope waveforms (visual + wind) on *ax*."""
    vis_baseline = 1.0
    wind_baseline = 3.0

    if "visual" in cond.lower() or "looming" in cond.lower():
        stim_t_rel = df["t_rel"]
        t_loom_start = stim_t_rel.min() if not stim_t_rel.empty else df["t_rel"].min()
        t_loom = np.array([t_loom_start, 0.0])
        ax.fill_between(
            t_loom, vis_baseline, vis_baseline + 1.0, step="mid",
            color=COLOR_OSCI_VIS, alpha=0.6, label="Visual (looming)",
        )

    subset = df[df["type"] == cond]
    if not subset.empty:
        first_tid = subset["global_trial_id"].iloc[0]
        grp = subset[subset["global_trial_id"] == first_tid].sort_values("t_rel")

        if "wind" in cond.lower() or "puff" in cond.lower() or grp["stim_state"].max() > 0:
            t_wind = grp["t_rel"].values
            stim = grp["stim_state"].values.astype(float)
            dt_last = t_wind[-1] - t_wind[-2] if len(t_wind) > 1 else 1.0
            t_wind_ext = np.append(t_wind, t_wind[-1] + dt_last)
            stim_ext = np.append(stim, stim[-1])
            ax.fill_between(
                t_wind_ext, wind_baseline, wind_baseline + stim_ext, step="post",
                color=COLOR_OSCI_HW, alpha=0.6, label="Wind (stim_state)",
            )

    ax.set_ylim(0, 5)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("Time relative to TTC (ms)")
    ax.grid(False)


# ══════════════════════════════════════════════════════════════════════
# Plot 1 — Trajectory Overlay
# ══════════════════════════════════════════════════════════════════════


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
    """
    One subplot per trial type. Left stimuli in NPG blue, right in NPG red.

    Each trial is aligned so the physical position at the backward-search
    latency onset is centered at (0, 0).
    """
    all_types = sorted(df["type"].dropna().unique())
    if not all_types:
        log.warning("No trial types found for trajectory overlay.")
        fig, _ = plt.subplots(figsize=figsize_per_ax)
        return fig

    n = len(all_types)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_ax[0] * n, figsize_per_ax[1]), squeeze=False)
    axes = axes[0]

    for idx, ttype in enumerate(all_types):
        ax = axes[idx]
        subset = df[df["type"] == ttype]

        for _tid, grp in subset.groupby("global_trial_id"):
            grp = grp.sort_values("t_rel")
            t_vals = grp["t_rel"].values

            # ── 读取当前试次的分类与 interval 信息 ──
            _response_type = grp["response_type"].iloc[0] if "response_type" in grp.columns else ""
            _onset_ms = grp["interval_onset_ms"].iloc[0] if "interval_onset_ms" in grp.columns else np.nan
            _offset_ms = grp["interval_offset_ms"].iloc[0] if "interval_offset_ms" in grp.columns else np.nan

            # ── Build escape-onset mask from pre-computed interval ──
            _is_escape = (_response_type in ("Escape", "PreWalk") and pd.notna(_onset_ms) and pd.notna(_offset_ms))
            if _is_escape:
                onset_idx = int(np.argmin(np.abs(t_vals - _onset_ms)))
                offset_idx = int(np.argmin(np.abs(t_vals - _offset_ms)))
                if onset_idx >= offset_idx:
                    offset_idx = min(onset_idx + 1, len(t_vals) - 1)
                escape_mask = np.zeros(len(t_vals), dtype=bool)
                escape_mask[onset_idx:offset_idx] = True
            else:
                escape_mask = None

            # ── Determine render window (xy and z independently) ──
            full_mask = np.ones(len(t_vals), dtype=bool)

            mask_xy = escape_mask if (USE_ESCAPE_ONSET_ONLY_XY and escape_mask is not None) else full_mask

            _macro_yaw_override = None
            if _is_escape and dz_integration_range == "escape_interval":
                mask_z = escape_mask
            elif _is_escape and dz_integration_range == "trial_to_onset":
                mask_z = np.zeros(len(t_vals), dtype=bool)
                mask_z[:onset_idx] = True
            elif _is_escape and dz_integration_range == "escape_angular_peak":
                av = grp["angular_velocity"].values if "angular_velocity" in grp.columns else None
                if av is not None:
                    mask_z = _build_angular_peak_dz_mask(av, onset_idx, offset_idx, len(t_vals))
                else:
                    mask_z = escape_mask
            elif _is_escape and dz_integration_range == "escape_onset_heading":
                mask_z = escape_mask
                _macro_yaw_override = np.cumsum(grp["dz"].fillna(0).values)[onset_idx] / RADIUS_MM
            else:
                mask_z = full_mask
            # Stage 1 always uses full escape interval dz for curvature;
            # mask_z (controlled by dz_integration_range) only affects Stage 2 yaw.
            _heading_dz_mask = escape_mask if (_is_escape and escape_mask is not None) else None
            # use_escape_onset_heading: add initial heading offset to Stage 1.
            # Skip for "escape_onset_heading" range — theta_init is already the Stage 2 angle.
            _heading_offset = 0.0
            if TRAJ_USE_ESCAPE_ONSET_HEADING and _is_escape and dz_integration_range != "escape_onset_heading":
                _heading_offset = np.cumsum(grp["dz"].fillna(0).values)[onset_idx] / RADIUS_MM

            if not np.any(mask_xy):
                continue

            burst = grp[mask_xy]



# --------------------------- Way for drawing trajectory -----------

            rot_x, rot_y = _body_to_traj(
                grp, mask_xy, use_z=USE_Z_DEGREE_TO_DRAW_TRAJECTORY,
                use_rigid_rotation=USE_RIGID_ROTATION, mask_z=mask_z,
                heading_dz_mask=_heading_dz_mask,
                macro_yaw_override=_macro_yaw_override,
                heading_offset=_heading_offset,
            )
            if rot_x is None:
                continue

# --------------------------------------------------------------------

            # ── Color & Plot ──
            ss = _get_unified_side(burst)
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
        _draw_standardized_grid(ax, max_radius=TRAJECTORY_MAX_RADIUS_MM, step=TRAJECTORY_STEP_MM)
        _draw_side_arrows(ax, left_color=left_color, right_color=right_color)

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Plot 2 — Speed Kinetics with Dual-Threshold Lines
# ══════════════════════════════════════════════════════════════════════


def plot_speed_kinetics(
    df: pd.DataFrame,
    control_type: str = "baseline_visual",
    stim_type: str = "looming_wind",
    figsize: tuple[float, float] = (10, 6),
    y_col: str = "speed",
    y_label: str = "Escape Speed (mm/s)",
) -> plt.Figure:
    """
    Two-panel figure (4:1 height ratio) with shared X axis.
    Upper panel: y_col (mean ± SEM) per condition + threshold lines.
    Lower panel: Oscilloscope-style dual-channel waveforms.
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.08)
    ax_main = fig.add_subplot(gs[0])
    ax_stim = fig.add_subplot(gs[1], sharex=ax_main)

    t_bin = 5.0
    t_min = df["t_rel"].min()
    t_max = df["t_rel"].max()
    bins = np.arange(t_min, t_max + t_bin, t_bin)
    df_binned = df.copy()
    df_binned["t_bin"] = pd.cut(df_binned["t_rel"], bins=bins, labels=bins[:-1], include_lowest=True)
    df_binned["t_bin"] = df_binned["t_bin"].astype(float)

    cond_colors: dict[str, str] = {}
    for ttype in df["type"].dropna().unique():
        if ttype == control_type:
            cond_colors[ttype] = COLOR_CONTROL
        else:
            sample = df[df["type"] == ttype].iloc[0]
            ss = _get_unified_side(sample)
            cond_colors[ttype] = COLOR_LEFT if ss == "left" else COLOR_RIGHT if ss == "right" else COLOR_LEFT

    for cond, color in cond_colors.items():
        subset = df_binned[df_binned["type"] == cond]
        if subset.empty:
            continue
        trial_means = subset.groupby(["global_trial_id", "t_bin"])[y_col].mean().reset_index()
        agg = trial_means.groupby("t_bin")[y_col]
        mean = agg.mean()
        sem = agg.sem().fillna(0)
        t_vals = mean.index.values
        ax_main.plot(t_vals, mean.values, color=color, lw=1.0, label=cond)
        ax_main.fill_between(t_vals, (mean - sem).values, (mean + sem).values,
                             color=color, alpha=0.2, edgecolor="none")

    ax_main.set_ylabel(y_label)
    ax_main.legend(loc="upper right", frameon=False)
    ax_main.set_xlabel("")
    plt.setp(ax_main.get_xticklabels(), visible=False)

    if y_col == "speed":
        _add_threshold_lines(ax_main)
    else:
        ax_main.axhline(y=0, color="0.5", linestyle="--", linewidth=0.5, alpha=0.4)

    # ── Lower panel: oscilloscope waveforms ──
    for cond in df["type"].dropna().unique():
        _draw_oscilloscope_channels(ax_stim, df, cond)
    ax_stim.legend(loc="upper right", frameon=False, ncol=2)

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Plot 3 — Spaghetti Kinetics with Dual-Threshold Lines
# ══════════════════════════════════════════════════════════════════════


def plot_spaghetti_kinetics(
    df: pd.DataFrame,
    control_type: str = "baseline_visual",
    stim_type: str = "looming_wind",
    figsize_per_col: float = 4.5,
    row_height: float = 5.0,
    y_col: str = "speed",
    y_label: str = "Escape Speed (mm/s)",
) -> plt.Figure:
    """Multi-panel spaghetti plot with per-condition spatial decoupling."""
    cond_color_map: dict[str, str] = {}
    for ttype in df["type"].dropna().unique():
        if ttype == control_type:
            cond_color_map[ttype] = COLOR_CONTROL
        else:
            sample = df[df["type"] == ttype].iloc[0]
            ss = _get_unified_side(sample)
            cond_color_map[ttype] = COLOR_LEFT if ss == "left" else COLOR_RIGHT if ss == "right" else COLOR_LEFT

    conditions = sorted(cond_color_map.keys())
    n_conds = len(conditions)
    if n_conds == 0:
        fig, ax = plt.subplots()
        return fig

    fig = plt.figure(figsize=(figsize_per_col * n_conds, row_height * 2))
    gs = gridspec.GridSpec(2, n_conds, height_ratios=[4, 1], hspace=0.1, wspace=0.15, figure=fig)

    ax_upper: list[plt.Axes] = []
    ax_lower: list[plt.Axes] = []
    for j in range(n_conds):
        sharey = ax_upper[0] if ax_upper else None
        ax_u = fig.add_subplot(gs[0, j], sharey=sharey)
        ax_upper.append(ax_u)
        ax_l = fig.add_subplot(gs[1, j], sharex=ax_u)
        ax_lower.append(ax_l)

    show_latency = y_col != "speed"  # ponytail: angular velocity plots show onset markers

    for j, cond in enumerate(conditions):
        ax = ax_upper[j]
        subset = df[df["type"] == cond]
        cond_color = cond_color_map[cond]

        if subset.empty:
            ax.set_title(cond, fontweight="bold")
            continue

        for _tid, grp in subset.groupby("global_trial_id"):
            grp_sorted = grp.sort_values("t_rel")
            ax.plot(grp_sorted["t_rel"], grp_sorted[y_col], color=cond_color, lw=0.5, alpha=0.25)

        t_bin = 5.0
        t_min = subset["t_rel"].min()
        t_max = subset["t_rel"].max()
        bins = np.arange(t_min, t_max + t_bin, t_bin)
        binned = subset.copy()
        binned["t_bin"] = pd.cut(binned["t_rel"], bins=bins, labels=bins[:-1], include_lowest=True)
        binned["t_bin"] = binned["t_bin"].astype(float)

        trial_means = binned.groupby(["global_trial_id", "t_bin"])[y_col].mean().reset_index()
        agg = trial_means.groupby("t_bin")[y_col]
        mean = agg.mean()
        sem = agg.sem().fillna(0)
        t_vals = mean.index.values

        ax.plot(t_vals, mean.values, color="white", lw=4.0, alpha=0.8, solid_capstyle="round")
        ax.plot(t_vals, mean.values, color=cond_color, lw=2.0, alpha=1.0, label=cond)
        ax.fill_between(t_vals, (mean - sem).values, (mean + sem).values,
                        color=cond_color, alpha=0.2, edgecolor="none")

        # ── Escape-onset markers (angular velocity plots only) ──
        if show_latency:
            trial_lats = subset.groupby("global_trial_id")["latency_ms"].first().dropna()
            for lat in trial_lats.values:
                ax.axvline(x=lat, color=cond_color, ls="--", lw=0.4, alpha=0.15)
            if not trial_lats.empty:
                mean_lat = trial_lats.mean()
                ax.axvline(x=mean_lat, color=cond_color, ls="--", lw=1.2, alpha=0.8)

        ax.set_title(cond, fontweight="bold")
        if j == 0:
            ax.set_ylabel(y_label)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.legend(loc="upper right", frameon=False)

        if y_col == "speed":
            _add_threshold_lines(ax)
        else:
            ax.axhline(y=0, color="0.5", linestyle="--", linewidth=0.5, alpha=0.4)

    # ── Oscilloscope channels ──
    for j, cond in enumerate(conditions):
        _draw_oscilloscope_channels(ax_lower[j], df, cond)
        if j == 0:
            ax_lower[j].legend(loc="upper right", frameon=False, ncol=2)

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Plot 4 — Behavior Probability Distribution
# ══════════════════════════════════════════════════════════════════════


def plot_behavior_probability(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of Escape / PreWalk / NoResponse proportions."""
    counts = df.groupby("global_trial_id")["response_type"].first().value_counts()
    total = counts.sum()

    categories = ["Escape", "PreWalk", "NoResponse"]
    values = [counts.get(c, 0) / total if total > 0 else 0.0 for c in categories]
    colors = [COLOR_ESCAPE, COLOR_PREWALK, COLOR_NO_RESPONSE]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    bars = ax.bar(categories, values, color=colors, width=0.55, edgecolor="none", alpha=0.85)

    for bar, val in zip(bars, values):
        if val > 0.02:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.05)
    ax.set_title("Behavior Probability Distribution", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Plot 5 — Habituation Decay Curve
# ══════════════════════════════════════════════════════════════════════


def plot_habituation_curve(df: pd.DataFrame) -> plt.Figure:
    """Scatter + line of V_max per trial across the global trial sequence."""
    trial_agg = df.groupby("global_trial_index").agg(
        v_max=("v_max", "first"),
        response_type=("response_type", "first"),
    ).reset_index()
    trial_agg = trial_agg.sort_values("global_trial_index")

    x = trial_agg["global_trial_index"].values
    y = trial_agg["v_max"].values

    color_map = {"Escape": COLOR_ESCAPE, "PreWalk": COLOR_PREWALK, "NoResponse": COLOR_NO_RESPONSE}
    point_colors = [color_map.get(rt, COLOR_NO_RESPONSE) for rt in trial_agg["response_type"].values]

    fig, ax = plt.subplots(figsize=(8, 3.5))

    ax.plot(x, y, color="0.7", lw=0.8, alpha=0.6, zorder=1)
    ax.scatter(x, y, c=point_colors, s=28, edgecolors="white", linewidths=0.4, zorder=2)

    ax.axhline(y=ESCAPE_VMAX_THRESHOLD, color="k", linestyle="--", linewidth=0.75, alpha=0.7)
    ax.text(x[-1] + 0.3, ESCAPE_VMAX_THRESHOLD, f"{ESCAPE_VMAX_THRESHOLD:.0f}",
            ha="left", va="center", fontsize=6, color="k", alpha=0.7)

    ax.axhline(y=ESCAPE_START_THRESHOLD, color="0.5", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.text(x[-1] + 0.3, ESCAPE_START_THRESHOLD, f"{ESCAPE_START_THRESHOLD:.0f}",
            ha="left", va="center", fontsize=6, color="0.5", alpha=0.5)

    ax.set_xlabel("Global Trial Index")
    ax.set_ylabel("V$_{max}$ (mm/s)")
    ax.set_title("Habituation Curve", fontweight="bold")
    ax.set_xlim(0.5, len(x) + 0.5)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=5, label=l)
        for l, c in [("Escape", COLOR_ESCAPE), ("PreWalk", COLOR_PREWALK), ("NoResponse", COLOR_NO_RESPONSE)]
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=6)

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Plot 6 — V_max Frequency Distribution (Diagnostic)
# ══════════════════════════════════════════════════════════════════════


def plot_vmax_distribution(df: pd.DataFrame, figsize: tuple[float, float] = (5.0, 3.5)) -> plt.Figure:
    """Histogram + KDE of per-trial V_max for Escape trials."""
    df_resp = df[df["response_type"] == "Escape"].copy()
    if df_resp.empty:
        log.warning("No Escape trials for V_max distribution plot.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No Escape trials", ha="center", va="center", transform=ax.transAxes)
        return fig

    trial_vmax = df_resp.groupby("global_trial_id")["v_max"].first().dropna().values

    if len(trial_vmax) < 2:
        log.warning("All V_max values are NaN — skipping distribution plot.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No valid V_max", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    bins = np.linspace(0, 400, 51)
    ax.hist(trial_vmax, bins=bins, density=True, color="#D0D0D0", edgecolor="white",
            linewidth=0.4, alpha=0.85, label="Histogram", zorder=2)

    kde = gaussian_kde(trial_vmax, bw_method="scott")
    x_kde = np.linspace(0, 400, 500)
    ax.plot(x_kde, kde(x_kde), color="black", lw=1.2, alpha=0.9, label="KDE", zorder=3)

    ax.axvline(ESCAPE_VMAX_THRESHOLD, color="black", ls="--", lw=0.75, alpha=0.7, zorder=4)
    ax.text(ESCAPE_VMAX_THRESHOLD + 3, ax.get_ylim()[1] * 0.92,
            f"Vmax\n{ESCAPE_VMAX_THRESHOLD:.0f}", fontsize=6, color="black", va="top")

    ax.axvline(ESCAPE_START_THRESHOLD, color="0.5", ls="--", lw=0.75, alpha=0.7, zorder=4)
    ax.text(ESCAPE_START_THRESHOLD + 3, ax.get_ylim()[1] * 0.92,
            f"Start\n{ESCAPE_START_THRESHOLD:.0f}", fontsize=6, color="0.5", va="top")

    ax.set_xlim(0, 400)
    ax.set_xlabel("$V_{max}$ (mm/s)")
    ax.set_ylabel("Probability Density")
    ax.set_title("$V_{max}$ Distribution — Threshold Diagnostic", fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=6)

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Plot 6b — Population V_max Distribution (All Subjects)
# ══════════════════════════════════════════════════════════════════════


def _get_trial_vmax(
    df: pd.DataFrame,
    response_filter: list[str] | None = None,
) -> np.ndarray:
    """Extract per-trial V_max values, optionally filtered by response_type."""
    work = df if response_filter is None else df[df["response_type"].isin(response_filter)]
    return (
        work.groupby(["subject_id", "global_trial_id"])["v_max"]
        .first().dropna().values
    )


def _setup_vmax_axes(
    trial_vmax: np.ndarray,
    x_upper_override: float | None = None,
    figsize: tuple[float, float] = (5.5, 4.0),
) -> tuple[plt.Figure, plt.Axes, float]:
    """Create figure + axes with histogram + KDE.  Returns (fig, ax, x_upper)."""
    fig, ax = plt.subplots(figsize=figsize)

    x_upper = x_upper_override if x_upper_override else max(
        np.percentile(trial_vmax, 99) * 1.15, 200
    )

    bins = np.linspace(0, x_upper, 51)
    ax.hist(trial_vmax, bins=bins, density=True, color="#D0D0D0", edgecolor="white",
            linewidth=0.4, alpha=0.85, label="Histogram", zorder=2)

    kde = gaussian_kde(trial_vmax, bw_method="scott")
    x_kde = np.linspace(0, x_upper, 500)
    ax.plot(x_kde, kde(x_kde), color="black", lw=1.2, alpha=0.9, label="KDE", zorder=3)

    return fig, ax, x_upper


def _draw_gmm_thresholds(
    ax: plt.Axes,
    x_upper: float,
    gmm_start_threshold: float | None = None,
    gmm_escape_threshold: float | None = None,
    iqr_gmm_threshold: float | None = None,
) -> None:
    """Draw adaptive GMM threshold lines with staggered labels."""
    label_y = [0.80, 0.70, 0.60]
    idx = 0

    if gmm_start_threshold is not None:
        ax.axvline(gmm_start_threshold, color="#E69F00", ls="--", lw=1.0, alpha=0.9, zorder=4)
        ax.text(gmm_start_threshold + x_upper * 0.01, ax.get_ylim()[1] * label_y[idx],
                f"GMM start\n{gmm_start_threshold:.0f}", fontsize=6, color="#E69F00", va="top")
        idx += 1

    if gmm_escape_threshold is not None:
        ax.axvline(gmm_escape_threshold, color="#DC0000", ls="--", lw=1.0, alpha=0.9, zorder=4)
        ax.text(gmm_escape_threshold + x_upper * 0.01, ax.get_ylim()[1] * label_y[idx],
                f"GMM escape\n{gmm_escape_threshold:.0f}", fontsize=6, color="#DC0000", va="top")
        idx += 1

    if iqr_gmm_threshold is not None:
        ax.axvline(iqr_gmm_threshold, color="#3C5488", ls="--", lw=1.0, alpha=0.9, zorder=4)
        ax.text(iqr_gmm_threshold + x_upper * 0.01, ax.get_ylim()[1] * label_y[idx],
                f"IQR-GMM\n{iqr_gmm_threshold:.0f}", fontsize=6, color="#3C5488", va="top")


def plot_population_vmax_gmm(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (5.5, 4.0),
    gmm_start_threshold: float | None = None,
    gmm_escape_threshold: float | None = None,
    iqr_gmm_threshold: float | None = None,
    draw_fixed_thresholds: bool = True,
) -> plt.Figure:
    """V_max distribution of **all trials** with GMM threshold markers.

    Purpose: threshold determination — shows the full distribution including
    NoResponse / PreWalk / Escape so the three GMM components are visible.
    """
    trial_vmax = _get_trial_vmax(df)  # all response types

    if len(trial_vmax) < 2:
        log.warning("Insufficient V_max values for GMM plot — skipping.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No valid V$_{max}$", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        return fig

    # Dynamic x-range: cover data + all threshold markers
    all_t = [t for t in [gmm_start_threshold, gmm_escape_threshold, iqr_gmm_threshold] if t]
    x_upper = max(np.percentile(trial_vmax, 99) * 1.15,
                  max(all_t) * 1.1 if all_t else 0, 200)

    fig, ax, x_upper = _setup_vmax_axes(trial_vmax, x_upper, figsize)

    # Fixed reference thresholds
    if draw_fixed_thresholds:
        ax.axvline(ESCAPE_VMAX_THRESHOLD, color="black", ls="--", lw=0.75, alpha=0.7, zorder=4)
        ax.text(ESCAPE_VMAX_THRESHOLD + x_upper * 0.01, ax.get_ylim()[1] * 0.92,
                f"Vmax\n{ESCAPE_VMAX_THRESHOLD:.0f}", fontsize=6, color="black", va="top")

        ax.axvline(ESCAPE_START_THRESHOLD, color="0.5", ls="--", lw=0.75, alpha=0.7, zorder=4)
        ax.text(ESCAPE_START_THRESHOLD + x_upper * 0.01, ax.get_ylim()[1] * 0.92,
                f"Start\n{ESCAPE_START_THRESHOLD:.0f}", fontsize=6, color="0.5", va="top")

    _draw_gmm_thresholds(ax, x_upper, gmm_start_threshold, gmm_escape_threshold, iqr_gmm_threshold)

    ax.set_xlim(0, x_upper)
    ax.set_xlabel("$V_{max}$ (mm/s)")
    ax.set_ylabel("Probability Density")
    ax.set_title("$V_{max}$ Distribution — All Trials (Threshold Determination)", fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=6)

    n_subjects = df["subject_id"].nunique()
    ax.text(0.97, 0.70, f"n = {len(trial_vmax)} trials\n({n_subjects} subjects)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8"))

    fig.tight_layout(pad=1.0)
    return fig


def plot_population_vmax_response(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (5.5, 4.0),
    auto_threshold: float | None = None,
    draw_fixed_thresholds: bool = True,
) -> plt.Figure:
    """V_max distribution of **Escape + PreWalk** trials only.

    Purpose: effective-response inspection — excludes NoResponse noise to
    focus on the shape of actual movement responses.
    """
    trial_vmax = _get_trial_vmax(df, response_filter=["Escape", "PreWalk"])

    if len(trial_vmax) < 2:
        log.warning("Insufficient Escape+PreWalk V_max values — skipping.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No valid V$_{max}$", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        return fig

    x_upper = max(np.percentile(trial_vmax, 99) * 1.15,
                  auto_threshold * 1.1 if auto_threshold else 0, 200)

    fig, ax, x_upper = _setup_vmax_axes(trial_vmax, x_upper, figsize)

    # Fixed reference thresholds
    if draw_fixed_thresholds:
        ax.axvline(ESCAPE_VMAX_THRESHOLD, color="black", ls="--", lw=0.75, alpha=0.7, zorder=4)
        ax.text(ESCAPE_VMAX_THRESHOLD + x_upper * 0.01, ax.get_ylim()[1] * 0.92,
                f"Vmax\n{ESCAPE_VMAX_THRESHOLD:.0f}", fontsize=6, color="black", va="top")

    # Active threshold used for tagging
    if auto_threshold is not None:
        ax.axvline(auto_threshold, color="#DC0000", ls="--", lw=1.0, alpha=0.9, zorder=4)
        ax.text(auto_threshold + x_upper * 0.01, ax.get_ylim()[1] * 0.80,
                f"Threshold\n{auto_threshold:.0f}", fontsize=6, color="#DC0000", va="top")

    # Per-type coloring in the info box
    n_esc = len(_get_trial_vmax(df, response_filter=["Escape"]))
    n_pw = len(_get_trial_vmax(df, response_filter=["PreWalk"]))
    n_subjects = df["subject_id"].nunique()
    ax.text(0.97, 0.70, f"Escape: {n_esc}  PreWalk: {n_pw}\n({n_subjects} subjects)",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8"))

    ax.set_xlim(0, x_upper)
    ax.set_xlabel("$V_{max}$ (mm/s)")
    ax.set_ylabel("Probability Density")
    ax.set_title("$V_{max}$ Distribution — Escape + PreWalk (Effective Responses)", fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=6)

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Plot 6c — Population Habituation / Fatigue Curve (All Subjects)
# ══════════════════════════════════════════════════════════════════════

_NPG8 = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488",
         "#F39B7F", "#8491B4", "#91D1C2", "#DC0000"]


def plot_population_habituation(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (10, 4.5),
) -> plt.Figure:
    """Population fatigue curve: per-subject lines + mean±SEM ribbon."""
    trial_df = (
        df.groupby(["subject_id", "global_trial_index"])
        .agg(v_max=("v_max", "first"), response_type=("response_type", "first"))
        .reset_index()
    )
    trial_df = trial_df.sort_values(["subject_id", "global_trial_index"])

    subjects = sorted(trial_df["subject_id"].unique())
    all_indices = sorted(trial_df["global_trial_index"].unique())

    # ── Per-subject lines ──
    subject_pivots: list[pd.Series] = []
    for subj in subjects:
        sdf = trial_df[trial_df["subject_id"] == subj].set_index("global_trial_index")["v_max"]
        subject_pivots.append(sdf)

    fig, ax = plt.subplots(figsize=figsize)

    for i, (subj, sdf) in enumerate(zip(subjects, subject_pivots)):
        color = _NPG8[i % len(_NPG8)]
        ax.plot(sdf.index, sdf.values, color=color, lw=0.7, alpha=0.35, zorder=1)

    # ── Population mean ± SEM ribbon ──
    pivot_df = pd.DataFrame({subj: sdf for subj, sdf in zip(subjects, subject_pivots)})
    pivot_df = pivot_df.reindex(all_indices)
    mean = pivot_df.mean(axis=1)
    sem = pivot_df.sem(axis=1)

    ax.plot(mean.index, mean.values, color="black", lw=2.0, alpha=0.9, zorder=3,
            label="Mean ± SEM")
    ax.fill_between(mean.index, (mean - sem).values, (mean + sem).values,
                    color="black", alpha=0.12, zorder=2)

    # ── Threshold lines ──
    x_max = max(all_indices)
    ax.axhline(y=ESCAPE_VMAX_THRESHOLD, color="k", linestyle="--", linewidth=0.75, alpha=0.7)
    ax.text(x_max + 0.3, ESCAPE_VMAX_THRESHOLD, f"{ESCAPE_VMAX_THRESHOLD:.0f}",
            ha="left", va="center", fontsize=6, color="k", alpha=0.7)

    ax.axhline(y=ESCAPE_START_THRESHOLD, color="0.5", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.text(x_max + 0.3, ESCAPE_START_THRESHOLD, f"{ESCAPE_START_THRESHOLD:.0f}",
            ha="left", va="center", fontsize=6, color="0.5", alpha=0.5)

    ax.set_xlabel("Global Trial Index")
    ax.set_ylabel("$V_{max}$ (mm/s)")
    ax.set_title("Population Habituation Curve", fontweight="bold")
    ax.set_xlim(0.5, x_max + 0.5)

    # ── Per-subject legend (small, in a box) ──
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=_NPG8[i % len(_NPG8)], lw=1.0, label=s)
               for i, s in enumerate(subjects)]
    handles.append(Line2D([0], [0], color="black", lw=2.0, label="Mean ± SEM"))
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=5,
              ncol=max(1, len(subjects) // 4 + 1))

    n_subjects = len(subjects)
    n_trials = len(trial_df)
    ax.text(0.02, 0.97, f"{n_subjects} subjects, {n_trials} trials",
            transform=ax.transAxes, ha="left", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8"))

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Plot 6d — Population Behavior Probability Distribution
# ══════════════════════════════════════════════════════════════════════


def plot_population_behavior_probability(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (4.5, 3.5),
    bar_label_style: str | None = None,
) -> plt.Figure:
    """Bar chart of Escape / PreWalk / NoResponse proportions across all subjects.

    Adds individual scatter points (jittered) showing each subject's
    probability for each behavior type to visualize individual variation.

    Parameters
    ----------
    bar_label_style :
        Override the default label placement.  If ``None``, the value from
        ``config.yaml`` (``visualization.bar_label_style``) is used.
        ``"inline"``  → label inside the bar (intuitive).
        ``"axis"``    → label via dashed line to y-axis (publication style).
    """
    trial_level = (
        df.groupby(["subject_id", "global_trial_index"])["response_type"]
        .first().reset_index()
    )

    # ── Overall proportions ──
    counts = trial_level["response_type"].value_counts()
    total = counts.sum()
    categories = ["Escape", "PreWalk", "NoResponse"]
    values = [counts.get(c, 0) / total if total > 0 else 0.0 for c in categories]
    colors = [COLOR_ESCAPE, COLOR_PREWALK, COLOR_NO_RESPONSE]

    # ── Per-subject proportions ──
    subject_probs = (
        trial_level.groupby("subject_id")["response_type"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(categories, values, color=colors, width=0.55, edgecolor="none", alpha=0.85)

    # ── Scatter individual subject probabilities ──
    rng = np.random.default_rng(42)
    scatter_size = 20
    scatter_alpha = 0.35
    jitter_width = 0.15

    for i, (cat, color) in enumerate(zip(categories, colors)):
        if cat in subject_probs.columns:
            subj_vals = subject_probs[cat].values
        else:
            subj_vals = np.zeros(len(subject_probs))

        jitter = rng.uniform(-jitter_width, jitter_width, size=len(subj_vals))
        ax.scatter(
            np.full(len(subj_vals), i) + jitter,
            subj_vals,
            s=scatter_size,
            c=color,
            alpha=scatter_alpha,
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
        )

    _style = bar_label_style if bar_label_style is not None else BAR_LABEL_STYLE.value

    for bar, val in zip(bars, values):
        if val > 0.02:
            if _style == "axis":
                # Publication style: dashed line from bar tip to y-axis + label on left
                ax.plot([0, bar.get_x() + bar.get_width() / 2],
                        [val, val], "k--", lw=0.5, alpha=0.4, zorder=1)
                ax.text(0.02, val, f"{val:.1%}", ha="left", va="center",
                        fontsize=7, color="black", fontweight="bold", zorder=10)
            else:
                # Inline style: label inside bar near bottom (away from scatter cloud at top).
                ax.text(bar.get_x() + bar.get_width() / 2, 0.03,
                        f"{val:.1%}", ha="center", va="bottom", fontsize=8,
                        color="white", fontweight="bold", zorder=10)

    n_subjects = df["subject_id"].nunique()
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.05)
    ax.set_title("Behavior Probability Distribution", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=1.0)
    fig.text(1.02, 0.5, f"{total} trials\n({n_subjects} subjects)",
             transform=ax.transAxes, ha="left", va="center", fontsize=7,
             color="0.4")
    return fig


# ══════════════════════════════════════════════════════════════════════
# Plot 6b — PreWalk Stillness Before Escape Onset
# ══════════════════════════════════════════════════════════════════════


def plot_prewalk_stillness(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (4.5, 3.5),
    stillness_threshold: float = ESCAPE_START_THRESHOLD,
    bar_label_style: str | None = None,
) -> plt.Figure:
    """Proportion of PreWalk trials with a stillness point (speed < threshold)
    in the 1-second window before escape onset.

    For each PreWalk trial, checks whether any frame in
    [interval_onset_ms - 1000, interval_onset_ms) has speed below
    *stillness_threshold* (default: ESCAPE_START_THRESHOLD = 10 mm/s).

    Parameters
    ----------
    bar_label_style :
        Override the default label placement.  If ``None``, the value from
        ``config.yaml`` (``visualization.bar_label_style``) is used.
        ``"inline"``  → label inside the bar (intuitive).
        ``"axis"``    → label via dashed line to y-axis (publication style).
    """
    trial_level = (
        df.groupby(["subject_id", "global_trial_index"])
        .agg(
            response_type=("response_type", "first"),
            interval_onset_ms=("interval_onset_ms", "first"),
        )
        .reset_index()
    )
    prewalk_trials = trial_level[trial_level["response_type"] == "PreWalk"].copy()

    if prewalk_trials.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No PreWalk trials", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="0.5")
        return fig

    # ── Detect stillness per trial ──
    has_stillness: dict[tuple, bool] = {}
    prewalk_keys = set(zip(prewalk_trials["subject_id"], prewalk_trials["global_trial_index"]))

    for (subj, tidx), grp in df.groupby(["subject_id", "global_trial_index"]):
        if (subj, tidx) not in prewalk_keys:
            continue
        onset_ms = grp["interval_onset_ms"].iloc[0]
        if pd.isna(onset_ms):
            continue
        window_mask = (grp["t_rel"] >= onset_ms - PREWALK_WINDOW_MS) & (grp["t_rel"] < onset_ms)
        window_speed = grp.loc[window_mask, "speed"].dropna()
        has_stillness[(subj, tidx)] = bool((window_speed < stillness_threshold).any())

    prewalk_trials["has_stillness"] = prewalk_trials.apply(
        lambda r: has_stillness.get((r["subject_id"], r["global_trial_index"]), False), axis=1
    )

    # ── Aggregate ──
    n_with = prewalk_trials["has_stillness"].sum()
    n_without = len(prewalk_trials) - n_with
    n_total = len(prewalk_trials)
    prop_with = n_with / n_total if n_total > 0 else 0.0

    # Per-subject proportions
    subject_props = (
        prewalk_trials.groupby("subject_id")["has_stillness"]
        .mean()
        .reset_index()
        .rename(columns={"has_stillness": "prop_stillness"})
    )

    # ── Plot ──
    fig, ax = plt.subplots(figsize=figsize)
    categories = ["With Stillness", "Without Stillness"]
    values = [prop_with, 1.0 - prop_with]
    bar_colors = [COLOR_WITH_STILLNESS, COLOR_NO_STILLNESS]  # deep navy vs neutral rock-grey

    bars = ax.bar(categories, values, color=bar_colors, width=0.55, edgecolor="none", alpha=0.85)

    # Scatter per-subject proportions (subtle, behind labels)
    rng = np.random.default_rng(42)
    jitter_width = 0.15
    for i, col in enumerate(["prop_stillness"]):
        subj_vals = subject_props[col].values
        jitter = rng.uniform(-jitter_width, jitter_width, size=len(subj_vals))
        ax.scatter(
            np.full(len(subj_vals), i) + jitter,
            subj_vals,
            s=20,
            c=bar_colors[i],
            alpha=0.35,
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
        )
    # Mirror: "Without" = 1 - "With"
    subj_without = 1.0 - subject_props["prop_stillness"].values
    jitter2 = rng.uniform(-jitter_width, jitter_width, size=len(subj_without))
    ax.scatter(
        np.full(len(subj_without), 1) + jitter2,
        subj_without,
        s=20,
        c=bar_colors[1],
        alpha=0.35,
        edgecolors="white",
        linewidths=0.5,
        zorder=5,
    )

    _style = bar_label_style if bar_label_style is not None else BAR_LABEL_STYLE.value

    for bar, val, n in zip(bars, values, [n_with, n_without]):
        if val > 0.02:
            if _style == "axis":
                # Publication style: dashed line from bar tip to y-axis + label on left
                ax.plot([0, bar.get_x() + bar.get_width() / 2],
                        [val, val], "k--", lw=0.5, alpha=0.4, zorder=1)
                ax.text(0.02, val, f"{val:.1%} ({n})", ha="left", va="center",
                        fontsize=7, color="black", fontweight="bold", zorder=10)
            else:
                # Inline style: label inside bar near bottom (away from scatter cloud at top).
                ax.text(bar.get_x() + bar.get_width() / 2, 0.03,
                        f"{val:.1%}\n({n})", ha="center", va="bottom", fontsize=7,
                        color="white", fontweight="bold", zorder=10)

    n_subjects = prewalk_trials["subject_id"].nunique()
    ax.set_ylabel("Proportion of PreWalk Trials")
    ax.set_ylim(0, 1.05)
    ax.set_title("PreWalk: Stillness Before Escape Onset", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=1.0)
    fig.text(1.02, 0.5, f"{n_total} PreWalk trials\n({n_subjects} subjects)",
             transform=ax.transAxes, ha="left", va="center", fontsize=7,
             color="0.4")
    return fig


# ══════════════════════════════════════════════════════════════════════
# Plot 7 — Single-Trial Speed Kinetics (Escape or PreWalk)
# ══════════════════════════════════════════════════════════════════════


def plot_single_trial_kinetics(
    trial: pd.DataFrame,
    latency_ms: float,
    v_max: float,
    global_trial_index: int,
    response_type: str = "Escape",
    figsize: tuple[float, float] = (6.0, 3.5),
    y_col: str = "speed",
    y_label: str = "Escape Speed (mm/s)",
    interval_onset_ms: float = np.nan,
    interval_offset_ms: float = np.nan,
) -> plt.Figure:
    """
    Single-trial kinetics (speed or angular velocity) with latency marker.
    Works for both Escape and PreWalk trials.
    """
    fig, ax = plt.subplots(figsize=figsize)

    t = trial["t_rel"].values
    y_vals = trial[y_col].values

    curve_color = COLOR_ESCAPE if response_type == "Escape" else COLOR_PREWALK

    ax.plot(t, y_vals, color=curve_color, lw=1.0, alpha=0.85, label=y_label)

    # ── Escape onset marker (blue dot on 10 mm/s crossing) ──
    onset_ms = latency_ms if np.isnan(interval_onset_ms) else interval_onset_ms
    if not np.isnan(onset_ms):
        ax.axvline(x=onset_ms, color="#3C5488", ls="--", lw=0.9, alpha=0.9,
                   label=f"Escape onset = {onset_ms:.1f} ms")
        onset_idx = np.argmin(np.abs(t - onset_ms))
        ax.scatter([onset_ms], [y_vals[onset_idx]], c="#3C5488", s=30, zorder=5,
                   edgecolors="white", linewidths=0.5)

    # ── Interval shading (directly from onset / offset) ──
    if y_col == "speed" and not np.isnan(interval_onset_ms) and not np.isnan(interval_offset_ms):
        ax.axvspan(interval_onset_ms, interval_offset_ms, alpha=0.10, color="#E64B35", zorder=0)

    # ── Reference lines ──
    if y_col == "speed":
        ax.axhline(y=ESCAPE_VMAX_THRESHOLD, color="k", linestyle="--", linewidth=0.75, alpha=0.6)
        ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02,
                ESCAPE_VMAX_THRESHOLD + 1.5,
                f"Vmax ({ESCAPE_VMAX_THRESHOLD:.0f})", fontsize=6, color="k", alpha=0.6)
        ax.axhline(y=ESCAPE_START_THRESHOLD, color="0.5", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02,
                ESCAPE_START_THRESHOLD + 1.5,
                f"Start ({ESCAPE_START_THRESHOLD:.0f})", fontsize=6, color="0.5", alpha=0.4)
    else:
        ax.axhline(y=0, color="0.5", linestyle="--", linewidth=0.5, alpha=0.4)

    # ── Labels & annotation ──
    ax.set_xlabel("Time relative to TTC (ms)")
    ax.set_ylabel(y_label)
    ax.set_title(f"Trial {global_trial_index} — {response_type}  (V$_{{max}}$={v_max:.1f} mm/s)", fontweight="bold")

    latency_str = f"{latency_ms:.1f}" if not np.isnan(latency_ms) else "N/A"
    if not np.isnan(interval_onset_ms) and not np.isnan(interval_offset_ms):
        interval_str = f"{interval_offset_ms - interval_onset_ms:.1f}"
    else:
        interval_str = "N/A"
    info_text = f"Latency: {latency_str} ms\nV$_{{max}}$: {v_max:.1f} mm/s\nInterval: {interval_str} ms"
    ax.text(0.98, 0.95, info_text, transform=ax.transAxes, fontsize=6,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.9))

    ax.legend(loc="upper left", frameon=False, fontsize=6)
    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Plot 9 — Global Unified Trajectory Overlay (Fixed Local Coordinates)
# ══════════════════════════════════════════════════════════════════════

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
    """
    Unified trajectory overlay — all paradigms on one axes, with left/right
    stimulus-side colouring and local heading alignment.

    [修复版]:
    1. 引入 subject_id 隔离跨动物的 global_trial_id 碰撞（修复放射状色块）。
    2. 丢弃全局 x/y 累计坐标，每次起步强制使用 dx, dy, dz 重新积分，确保局部 0 rad 起步。
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 修复1: 加入 subject_id 分组条件，防止不同动物的同名 Trial 互相串台导致坐标剧烈跳跃
    group_cols = ["subject_id", "global_trial_id"] if "subject_id" in df.columns else ["global_trial_id"]

    for _keys, grp in df.groupby(group_cols):
        grp = grp.sort_values("t_rel")
        t_vals = grp["t_rel"].values

        # ── 读取当前试次的分类与 interval 信息 ──
        _response_type = grp["response_type"].iloc[0] if "response_type" in grp.columns else ""
        _onset_ms = grp["interval_onset_ms"].iloc[0] if "interval_onset_ms" in grp.columns else np.nan
        _offset_ms = grp["interval_offset_ms"].iloc[0] if "interval_offset_ms" in grp.columns else np.nan

        # ── Build escape-onset mask from pre-computed interval ──
        _is_escape = (_response_type in ("Escape", "PreWalk") and pd.notna(_onset_ms) and pd.notna(_offset_ms))
        if _is_escape:
            onset_idx = int(np.argmin(np.abs(t_vals - _onset_ms)))
            offset_idx = int(np.argmin(np.abs(t_vals - _offset_ms)))
            if onset_idx >= offset_idx:
                offset_idx = min(onset_idx + 1, len(t_vals) - 1)
            escape_mask = np.zeros(len(t_vals), dtype=bool)
            escape_mask[onset_idx:offset_idx] = True
        else:
            escape_mask = None

        # ── Determine render window (xy and z independently) ──
        full_mask = np.ones(len(t_vals), dtype=bool)

        mask_xy = escape_mask if (USE_ESCAPE_ONSET_ONLY_XY and escape_mask is not None) else full_mask

        _macro_yaw_override = None
        if _is_escape and dz_integration_range == "escape_interval":
            mask_z = escape_mask
        elif _is_escape and dz_integration_range == "trial_to_onset":
            mask_z = np.zeros(len(t_vals), dtype=bool)
            mask_z[:onset_idx] = True
        elif _is_escape and dz_integration_range == "escape_angular_peak":
            av = grp["angular_velocity"].values if "angular_velocity" in grp.columns else None
            if av is not None:
                mask_z = _build_angular_peak_dz_mask(av, onset_idx, offset_idx, len(t_vals))
            else:
                mask_z = escape_mask
        elif _is_escape and dz_integration_range == "escape_onset_heading":
            mask_z = escape_mask
            _macro_yaw_override = np.cumsum(grp["dz"].fillna(0).values)[onset_idx] / RADIUS_MM
        else:
            mask_z = full_mask
        # Stage 1 always uses full escape interval dz for curvature;
        # mask_z (controlled by dz_integration_range) only affects Stage 2 yaw.
        _heading_dz_mask = escape_mask if (_is_escape and escape_mask is not None) else None
        # use_escape_onset_heading: add initial heading offset to Stage 1.
        # Skip for "escape_onset_heading" range — theta_init is already the Stage 2 angle.
        _heading_offset = 0.0
        if TRAJ_USE_ESCAPE_ONSET_HEADING and _is_escape and dz_integration_range != "escape_onset_heading":
            _heading_offset = np.cumsum(grp["dz"].fillna(0).values)[onset_idx] / RADIUS_MM

        if not np.any(mask_xy):
            continue

        burst = grp[mask_xy]

        traj_x, traj_y = _body_to_traj(
            grp, mask_xy, use_z=USE_Z_DEGREE_TO_DRAW_TRAJECTORY,
            use_rigid_rotation=USE_RIGID_ROTATION, mask_z=mask_z,
            heading_dz_mask=_heading_dz_mask,
            macro_yaw_override=_macro_yaw_override,
            heading_offset=_heading_offset,
        )
        if traj_x is None:
            continue

        # ── 颜色分配与绘图 ──
        ss = _get_unified_side(burst)
        if ss == "left":
            color = left_color
        elif ss == "right":
            color = right_color
        else:
            color = COLOR_CONTROL

        ax.plot(traj_x, traj_y, color=color, alpha=alpha, lw=lw)

    # 绘制背景和参考图例
    _draw_standardized_grid(ax, max_radius=TRAJECTORY_MAX_RADIUS_MM, step=TRAJECTORY_STEP_MM)
    _draw_side_arrows(ax)

    n_trials = df.groupby(group_cols).ngroups
    ax.set_title(
        f"Unified Trajectory Overlay(n={n_trials} trials)",
        fontweight="bold",
    )

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Plot 10 — Escape Angle Distribution (Escape vs PreWalk)
# ══════════════════════════════════════════════════════════════════════


def plot_escape_angle_distribution(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (6.0, 4.0),
    bins: int = 36,
) -> plt.Figure:
    """Histogram + KDE of final escape angles for Escape and PreWalk trials.

    The escape angle is defined as the angle (in degrees) from the origin
    to the trajectory endpoint: ``atan2(traj_y[-1], traj_x[-1])``.
    This reveals the directional dispersion pattern of escape responses.
    """
    group_cols = ["subject_id", "global_trial_id"] if "subject_id" in df.columns else ["global_trial_id"]
    escape_types = ["Escape", "PreWalk"]
    df_esc = df[df["response_type"].isin(escape_types)].copy()

    if df_esc.empty:
        log.warning("No Escape/PreWalk trials for angle distribution plot.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No Escape / PreWalk trials", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        return fig

    angles_by_type: dict[str, list[float]] = {"Escape": [], "PreWalk": []}

    for keys, grp in df_esc.groupby(group_cols):
        grp = grp.sort_values("t_rel")
        response_type = grp["response_type"].iloc[0]
        if response_type not in angles_by_type:
            continue

        t_vals = grp["t_rel"].values
        _onset_ms = grp["interval_onset_ms"].iloc[0] if "interval_onset_ms" in grp.columns else np.nan
        _offset_ms = grp["interval_offset_ms"].iloc[0] if "interval_offset_ms" in grp.columns else np.nan

        _is_valid = pd.notna(_onset_ms) and pd.notna(_offset_ms)
        if _is_valid:
            onset_idx = int(np.argmin(np.abs(t_vals - _onset_ms)))
            offset_idx = int(np.argmin(np.abs(t_vals - _offset_ms)))
            if onset_idx >= offset_idx:
                offset_idx = min(onset_idx + 1, len(t_vals) - 1)
            escape_mask = np.zeros(len(t_vals), dtype=bool)
            escape_mask[onset_idx:offset_idx] = True
        else:
            escape_mask = None

        full_mask = np.ones(len(t_vals), dtype=bool)
        mask_xy = escape_mask if (TRAJ_USE_ESCAPE_ONSET_ONLY_XY and escape_mask is not None) else full_mask

        _macro_yaw_override = None
        if _is_valid and DZ_INTEGRATION_RANGE == "escape_interval":
            mask_z = escape_mask
        elif _is_valid and DZ_INTEGRATION_RANGE == "trial_to_onset":
            mask_z = np.zeros(len(t_vals), dtype=bool)
            mask_z[:onset_idx] = True
        elif _is_valid and DZ_INTEGRATION_RANGE == "escape_angular_peak":
            av = grp["angular_velocity"].values if "angular_velocity" in grp.columns else None
            if av is not None:
                mask_z = _build_angular_peak_dz_mask(av, onset_idx, offset_idx, len(t_vals))
            else:
                mask_z = escape_mask
        elif _is_valid and DZ_INTEGRATION_RANGE == "escape_onset_heading":
            mask_z = escape_mask
            _macro_yaw_override = np.cumsum(grp["dz"].fillna(0).values)[onset_idx] / RADIUS_MM
        else:
            mask_z = full_mask

        _heading_dz_mask = escape_mask if (_is_valid and escape_mask is not None) else None
        _heading_offset = 0.0
        if TRAJ_USE_ESCAPE_ONSET_HEADING and _is_valid and DZ_INTEGRATION_RANGE != "escape_onset_heading":
            _heading_offset = np.cumsum(grp["dz"].fillna(0).values)[onset_idx] / RADIUS_MM

        if not np.any(mask_xy):
            continue

        traj_x, traj_y = _body_to_traj(
            grp, mask_xy, use_z=TRAJ_USE_Z_DEGREE,
            use_rigid_rotation=TRAJ_USE_RIGID_ROTATION, mask_z=mask_z,
            heading_dz_mask=_heading_dz_mask,
            macro_yaw_override=_macro_yaw_override,
            heading_offset=_heading_offset,
        )
        if traj_x is None or len(traj_x) < 2:
            continue

        angle_deg = float(np.degrees(np.arctan2(traj_y[-1], traj_x[-1])))
        angles_by_type[response_type].append(angle_deg)

    esc_angles = np.array(angles_by_type["Escape"])
    pw_angles = np.array(angles_by_type["PreWalk"])

    if len(esc_angles) == 0 and len(pw_angles) == 0:
        log.warning("No valid trajectory endpoints for angle distribution.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No valid escape angles", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        return fig

    fig, ax = plt.subplots(figsize=figsize)
    bin_edges = np.linspace(-180, 180, bins + 1)

    # Optional KDE x-grid
    x_kde = np.linspace(-180, 180, 300)

    # Cell style high-contrast palette
    color_esc = COLOR_ESCAPE   # #ED0000 (Crimson Red)
    color_pw = COLOR_PREWALK   # #00468B (Navy Blue)

    # 独立控制透明度：面透明(35%)，边框不透明(100%)
    fc_esc = mcolors.to_rgba(color_esc, 0.35)
    fc_pw = mcolors.to_rgba(color_pw, 0.35)

    # 中心参考线
    ax.axvline(0, color="#9CA3AF", linestyle="--", linewidth=1.0, alpha=0.5, zorder=0)

    if len(esc_angles) > 0:
        # 直方图：每个 Bin 都有清晰的不透明边框
        ax.hist(esc_angles, bins=bin_edges, density=True,
                facecolor=fc_esc, edgecolor=color_esc, linewidth=1.2,
                label=f"Escape (n={len(esc_angles)})", zorder=1)
        # KDE 密度曲线
        if len(esc_angles) > 1:
            kde_esc = gaussian_kde(esc_angles, bw_method="scott")
            ax.plot(x_kde, kde_esc(x_kde), color=color_esc, lw=2.5, alpha=0.9, zorder=3)

    if len(pw_angles) > 0:
        # 直方图：每个 Bin 都有清晰的不透明边框
        ax.hist(pw_angles, bins=bin_edges, density=True,
                facecolor=fc_pw, edgecolor=color_pw, linewidth=1.2,
                label=f"PreWalk (n={len(pw_angles)})", zorder=1)
        # KDE 密度曲线
        if len(pw_angles) > 1:
            kde_pw = gaussian_kde(pw_angles, bw_method="scott")
            ax.plot(x_kde, kde_pw(x_kde), color=color_pw, lw=2.5, alpha=0.9, zorder=3)

    ax.set_xlabel("Escape Angle (°)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Escape Angle Distribution", fontweight="bold")
    ax.set_xlim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 45))
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    # 极简背景网格：使用超浅实线取代虚线，符合现代顶刊规范
    ax.yaxis.grid(True, color="#E5E7EB", linestyle="-", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Plot 8 — Aligned Population Polar Direction Histogram
# ══════════════════════════════════════════════════════════════════════


def plot_population_polar_histogram(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (6.0, 4.8),
    bins: int = 36,
) -> plt.Figure:
    """
    360° polar rose of population escape-direction dispersion.

    All stimuli are mirrored to the right (Ipsi). Escape / PreWalk endpoint
    bearings are overlaid as proportional rose histograms (percent of trials
    per bin) with boundary-corrected wrapped-KDE outlines; arrows mark each
    group's circular mean direction, their length proportional to the mean
    resultant length R.
    """
    group_cols = ["subject_id", "global_trial_id"] if "subject_id" in df.columns else ["global_trial_id"]
    escape_types = ["Escape", "PreWalk"]
    df_esc = df[df["response_type"].isin(escape_types)].copy()

    angles_by_type: dict[str, list[float]] = {"Escape": [], "PreWalk": []}

    for keys, grp in df_esc.groupby(group_cols):
        grp = grp.sort_values("t_rel")
        response_type = grp["response_type"].iloc[0]

        t_vals = grp["t_rel"].values
        _onset_ms = grp["interval_onset_ms"].iloc[0] if "interval_onset_ms" in grp.columns else np.nan
        _offset_ms = grp["interval_offset_ms"].iloc[0] if "interval_offset_ms" in grp.columns else np.nan

        _is_valid = pd.notna(_onset_ms) and pd.notna(_offset_ms)
        if _is_valid:
            onset_idx = int(np.argmin(np.abs(t_vals - _onset_ms)))
            offset_idx = int(np.argmin(np.abs(t_vals - _offset_ms)))
            if onset_idx >= offset_idx:
                offset_idx = min(onset_idx + 1, len(t_vals) - 1)
            escape_mask = np.zeros(len(t_vals), dtype=bool)
            escape_mask[onset_idx:offset_idx] = True
        else:
            escape_mask = None

        full_mask = np.ones(len(t_vals), dtype=bool)
        mask_xy = escape_mask if (TRAJ_USE_ESCAPE_ONSET_ONLY_XY and escape_mask is not None) else full_mask

        # 获取 dz 掩码与宏观旋转参数
        _macro_yaw_override = None
        if _is_valid and DZ_INTEGRATION_RANGE == "escape_interval":
            mask_z = escape_mask
        elif _is_valid and DZ_INTEGRATION_RANGE == "trial_to_onset":
            mask_z = np.zeros(len(t_vals), dtype=bool)
            mask_z[:onset_idx] = True
        elif _is_valid and DZ_INTEGRATION_RANGE == "escape_angular_peak":
            av = grp["angular_velocity"].values if "angular_velocity" in grp.columns else None
            if av is not None:
                mask_z = _build_angular_peak_dz_mask(av, onset_idx, offset_idx, len(t_vals))
            else:
                mask_z = escape_mask
        elif _is_valid and DZ_INTEGRATION_RANGE == "escape_onset_heading":
            mask_z = escape_mask
            _macro_yaw_override = np.cumsum(grp["dz"].fillna(0).values)[onset_idx] / RADIUS_MM
        else:
            mask_z = full_mask

        _heading_dz_mask = escape_mask if (_is_valid and escape_mask is not None) else None
        _heading_offset = 0.0
        if TRAJ_USE_ESCAPE_ONSET_HEADING and _is_valid and DZ_INTEGRATION_RANGE != "escape_onset_heading":
            _heading_offset = np.cumsum(grp["dz"].fillna(0).values)[onset_idx] / RADIUS_MM

        if not np.any(mask_xy):
            continue

        traj_x, traj_y = _body_to_traj(
            grp, mask_xy, use_z=TRAJ_USE_Z_DEGREE,
            use_rigid_rotation=TRAJ_USE_RIGID_ROTATION, mask_z=mask_z,
            heading_dz_mask=_heading_dz_mask,
            macro_yaw_override=_macro_yaw_override,
            heading_offset=_heading_offset,
        )
        if traj_x is None or len(traj_x) < 2:
            continue

        # 核心对齐：统一将刺激映射到右侧 (Ipsi)
        ss = _get_unified_side(grp)
        if ss == "left":
            traj_x = -traj_x  # X轴水平镜像

        # 计算极坐标夹角：arctan2(X, Y) 确保正前方(Y轴)为0度，右侧为90度，左侧为-90度
        angle_rad = float(np.arctan2(traj_x[-1], traj_y[-1]))
        angles_by_type[response_type].append(angle_rad)

    esc = np.asarray(angles_by_type["Escape"])
    pw = np.asarray(angles_by_type["PreWalk"])

    if esc.size == 0 and pw.size == 0:
        log.warning("No valid trials for polar direction histogram.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No valid escape trials", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="0.5")
        return fig

    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    # ── Canvas: single overlay rose (0° = forward, 90° = Ipsi, clockwise) ──
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # set_thetagrids, not set_xticks: the unordered tick list would collapse
    # the theta view interval to a quarter circle and clip the outer spine
    ax.set_thetagrids([0, 90, 180, 270],
                      ["0°", "90°\nIpsi", "±180°", "−90°\nContra"],
                      fontsize=9, color="0.2")
    ax.tick_params(axis="x", pad=6)

    # Recessive solid hairline grid — never dashed
    ax.grid(True, color="0.88", linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)

    bin_edges = np.linspace(-np.pi, np.pi, bins + 1)
    bin_width = 2 * np.pi / bins

    series = [
        (name, angles, color)
        for name, angles, color in
        [("Escape", esc, COLOR_ESCAPE), ("PreWalk", pw, COLOR_PREWALK)]
        if angles.size > 0
    ]

    # Percent of trials per bin → common scale across groups
    radii_by_name: dict[str, np.ndarray] = {}
    for name, angles, _color in series:
        counts, _ = np.histogram(angles, bins=bin_edges)
        radii_by_name[name] = counts / angles.size * 100.0
    rmax = max(r.max() for r in radii_by_name.values()) * 1.30

    legend_handles: list[Patch] = []
    for name, angles, color in series:
        # Rose: translucent wash fill, adjacent bins separated by surface gaps
        ax.bar(bin_edges[:-1], radii_by_name[name], width=bin_width, align="edge",
               facecolor=mcolors.to_rgba(color, 0.30),
               edgecolor="white", linewidth=0.7, zorder=2)

        # Wrapped-KDE outline (corrected for the ±180° boundary)
        if angles.size > 1 and np.std(angles) > 1e-6:
            kde = gaussian_kde(angles, bw_method="scott")
            x_kde = np.linspace(-np.pi, np.pi, 361)
            dens = kde(x_kde) + kde(x_kde - 2 * np.pi) + kde(x_kde + 2 * np.pi)
            dens[0] = dens[-1] = 0.5 * (dens[0] + dens[-1])
            ax.plot(x_kde, dens * bin_width * 100.0, color=color, lw=2.0,
                    solid_capstyle="round", zorder=4)

        # Circular mean vector — straight radial shaft, length ∝ resultant R.
        # Journal-style marker: a hairline shaft with a thin single white halo
        # (not the heavy capsule the default `-|>` halo gives), straight (no
        # Bézier curl), and a small filled head. annotate handles the head size
        # in display pt so it stays ~9pt regardless of R or the radial scale.
        mu = float(circmean(angles, high=np.pi, low=-np.pi))
        resultant = float(np.abs(np.exp(1j * angles).mean()))
        arrow = ax.annotate(
            "", xy=(mu, resultant * rmax), xytext=(mu, 0.03 * rmax),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.4,
                mutation_scale=10,
                shrinkA=0, shrinkB=0,
                connectionstyle="arc3,rad=0",  # straight shaft, no Bézier curl
            ),
            zorder=5,
        )
        arrow.arrow_patch.set_path_effects([
            path_effects.withStroke(linewidth=2.6, foreground="white")
        ])

        legend_handles.append(Patch(
            facecolor=mcolors.to_rgba(color, 0.30), edgecolor=color, linewidth=1.2,
            label=f"{name} (n = {angles.size})\nμ = {np.degrees(mu):.0f}°, R = {resultant:.2f}",
        ))

    # ── Radial scale: percent of trials, labels on the emptiest spoke ──
    ax.set_ylim(0, rmax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="lower"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}%"))
    combined = np.sum(np.stack([radii_by_name[name] for name, _a, _c in series]), axis=0)
    spokes = np.arange(0, 360, 45)
    spoke_bins = [min(int(((np.deg2rad(s) + np.pi) % (2 * np.pi)) / bin_width), bins - 1)
                  for s in spokes]
    spoke_mass = [combined[max(b - 1, 0):b + 2].sum() for b in spoke_bins]
    ax.set_rlabel_position(float(spokes[int(np.argmin(spoke_mass))]))
    ax.tick_params(axis="y", labelsize=7, colors="0.45", pad=1)
    for lbl in ax.get_yticklabels():
        lbl.set_path_effects([
            path_effects.withStroke(linewidth=2.0, foreground="white")
        ])

    ax.set_title("Escape Direction Distribution", fontweight="bold", pad=22)

    fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.99, 0.99),
               frameon=False, fontsize=8, handlelength=1.4, labelspacing=0.6)

    return fig
