"""
Cercus Framework — Publication-Grade Visualization
===================================================
All Matplotlib plotting functions with Nature/Science global style.
"""

from __future__ import annotations

import logging

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from .constants import (
    COLOR_CONTROL,
    COLOR_ESCAPE,
    COLOR_LEFT,
    COLOR_NO_RESPONSE,
    COLOR_OSCI_HW,
    COLOR_OSCI_VIS,
    COLOR_PREWALK,
    COLOR_RIGHT,
    ESCAPE_START_THRESHOLD,
    ESCAPE_VMAX_THRESHOLD,
    RADIUS_MM,
    TRAJECTORY_MAX_RADIUS_MM,
    TRAJECTORY_STEP_MM,
    _get_unified_side,
)
from .kinematics import compute_escape_latency

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
        circle = plt.Circle((0, 0), r, color="gray", fill=False, ls="--", lw=0.5, alpha=0.5, zorder=1)
        ax.add_patch(circle)
        ax.text(
            r * 0.707, r * 0.707, f"{int(r)} mm",
            color="gray", fontsize=6, ha="left", va="bottom", alpha=0.8,
        )

    ax.set_xticks([])
    ax.set_yticks([])


def _draw_side_arrows(ax: plt.Axes, left_color: str = COLOR_LEFT, right_color: str = COLOR_RIGHT) -> None:
    """Draw minimalist vector arrows on LEFT and RIGHT edges."""
    arrow_style = dict(arrowstyle="-|>", color=None, lw=1.0, mutation_scale=8)

    ax.annotate(
        "", xy=(0.04, 0.5), xytext=(-0.02, 0.5),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops={**arrow_style, "color": left_color},
    )
    ax.text(0.01, 0.45, "Left Stimulus", transform=ax.transAxes,
            ha="center", va="top", fontsize=7, color=left_color)

    ax.annotate(
        "", xy=(0.96, 0.5), xytext=(1.02, 0.5),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops={**arrow_style, "color": right_color},
    )
    ax.text(0.99, 0.45, "Right Stimulus", transform=ax.transAxes,
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


# ══════════════════════════════════════════════════════════════════════
# Plot 1 — Trajectory Overlay
# ══════════════════════════════════════════════════════════════════════


def plot_trajectory_overlay(
    df: pd.DataFrame,
    control_type: str = "baseline_visual_test",
    left_color: str = COLOR_LEFT,
    right_color: str = COLOR_RIGHT,
    figsize_per_ax: tuple[float, float] = (4.0, 4.0),


    #------------- Trajectory Drawing Options -------------
    # USE_Z_DEGREE_TO_DRAW_TRAJECTORY = False
    USE_Z_DEGREE_TO_DRAW_TRAJECTORY = True
    # -------------------------------------------------
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
            speed_vals = grp["speed"].values

            # Pass wind-onset offset so burst detection uses the correct window
            # for multimodal trials (t_rel=0 is TTC, not wind onset).
            _ttc = grp["target_ttc_ms"].iloc[0] if "target_ttc_ms" in grp.columns else np.nan
            esc = compute_escape_latency(
                t_vals, speed_vals,
                stim_onset_t_rel=float(_ttc) if pd.notna(_ttc) else None,
            )

            # ── Determine render window ──
            if not np.isnan(esc["latency_ms"]):
                latency_ms = esc["latency_ms"]
                # Choose drawng window
                # render_start_ms = latency_ms - timewindow_ms
                # use all trial as trajectory analysis
                render_start_ms = float(t_vals.min())
                render_start_idx = int(np.argmin(np.abs(t_vals - render_start_ms)))
                actual_start_ms = t_vals[render_start_idx]
                burst_end_ms = latency_ms + 500.0
            else:
                render_start_idx = int(np.argmin(np.abs(t_vals - 0.0)))
                actual_start_ms = 0.0
                burst_end_ms = 500.0

            burst_mask = (t_vals >= actual_start_ms) & (t_vals <= burst_end_ms)

            # Trim at first sub-threshold drop after the burst peak
            if np.any(burst_mask):
                burst_speed = speed_vals[burst_mask]
                burst_t = t_vals[burst_mask]
                if np.all(np.isnan(burst_speed)):
                    continue
                peak_in_burst = int(np.nanargmax(burst_speed))
                post_peak_speed = burst_speed[peak_in_burst:]
                post_peak_t = burst_t[peak_in_burst:]
                below_rest = post_peak_speed < ESCAPE_START_THRESHOLD
                if np.any(below_rest):
                    rest_idx = int(np.argmax(below_rest))
                    actual_end_ms = post_peak_t[rest_idx]
                    burst_mask = (t_vals >= actual_start_ms) & (t_vals <= actual_end_ms)

            if not np.any(burst_mask):
                continue

            burst = grp[burst_mask]


# --------------------------- Way for drawing trajectory -----------

            # ── Dynamic Origin Translation ──
            x_origin = burst["x"].iloc[0]
            y_origin = burst["y"].iloc[0]
            burst_x = burst["x"].values - x_origin
            burst_y = burst["y"].values - y_origin

            if not USE_Z_DEGREE_TO_DRAW_TRAJECTORY:
                # ── Dynamic Vector Alignment (Reverse Rotation) ──
                raw_heading = grp["dz"].cumsum().values / RADIUS_MM
                theta = -raw_heading[render_start_idx]
                rot_x = burst_x * np.cos(theta) - burst_y * np.sin(theta)
                rot_y = burst_x * np.sin(theta) + burst_y * np.cos(theta)

            if USE_Z_DEGREE_TO_DRAW_TRAJECTORY:
                rot_x = burst_x
                rot_y = burst_y

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

            ax.plot(rot_x, rot_y, color=color, alpha=0.4, lw=0.8)

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
) -> plt.Figure:
    """
    Two-panel figure (4:1 height ratio) with shared X axis.
    Upper panel: speed (mean ± SEM) per condition + threshold lines.
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
        trial_means = subset.groupby(["global_trial_id", "t_bin"])["speed"].mean().reset_index()
        agg = trial_means.groupby("t_bin")["speed"]
        mean = agg.mean()
        sem = agg.sem().fillna(0)
        t_vals = mean.index.values
        ax_main.plot(t_vals, mean.values, color=color, lw=1.0, label=cond)
        ax_main.fill_between(t_vals, (mean - sem).values, (mean + sem).values,
                             color=color, alpha=0.2, edgecolor="none")

    ax_main.set_ylabel("Escape Speed (mm/s)")
    ax_main.legend(loc="upper right", frameon=False)
    ax_main.set_xlabel("")
    plt.setp(ax_main.get_xticklabels(), visible=False)

    _add_threshold_lines(ax_main)

    # ── Lower panel: oscilloscope waveforms ──
    vis_baseline = 1.0
    wind_baseline = 3.0
    drawn_vis = False
    drawn_wind = False

    for cond in df["type"].dropna().unique():
        subset = df[df["type"] == cond]
        if subset.empty:
            continue

        if ("visual" in cond.lower() or "looming" in cond.lower()) and not drawn_vis:
            stim_t_rel = subset["t_rel"]
            t_loom_start = stim_t_rel.min() if not stim_t_rel.empty else df["t_rel"].min()
            t_loom = np.array([t_loom_start, 0.0])
            ax_stim.fill_between(
                t_loom, vis_baseline, vis_baseline + 1.0, step="mid",
                color=COLOR_OSCI_VIS, alpha=0.6, label="Visual (looming)",
            )
            drawn_vis = True

        first_tid = subset["global_trial_id"].iloc[0]
        grp = subset[subset["global_trial_id"] == first_tid].sort_values("t_rel")

        if ("wind" in cond.lower() or "puff" in cond.lower() or grp["stim_state"].max() > 0) and not drawn_wind:
            t_wind = grp["t_rel"].values
            stim = grp["stim_state"].values.astype(float)

            dt_last = t_wind[-1] - t_wind[-2] if len(t_wind) > 1 else 1.0
            t_wind_ext = np.append(t_wind, t_wind[-1] + dt_last)
            stim_ext = np.append(stim, stim[-1])

            ax_stim.fill_between(
                t_wind_ext, wind_baseline, wind_baseline + stim_ext, step="post",
                color=COLOR_OSCI_HW, alpha=0.6, label="Wind (stim_state)",
            )
            drawn_wind = True

    ax_stim.set_ylim(0, 5)
    ax_stim.set_yticks([])
    ax_stim.set_ylabel("")
    ax_stim.set_xlabel("Time relative to TTC (ms)")
    ax_stim.legend(loc="upper right", frameon=False, ncol=2)
    ax_stim.grid(False)

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

    for j, cond in enumerate(conditions):
        ax = ax_upper[j]
        subset = df[df["type"] == cond]
        cond_color = cond_color_map[cond]

        if subset.empty:
            ax.set_title(cond, fontweight="bold")
            continue

        for _tid, grp in subset.groupby("global_trial_id"):
            grp_sorted = grp.sort_values("t_rel")
            ax.plot(grp_sorted["t_rel"], grp_sorted["speed"], color=cond_color, lw=0.5, alpha=0.25)

        t_bin = 5.0
        t_min = subset["t_rel"].min()
        t_max = subset["t_rel"].max()
        bins = np.arange(t_min, t_max + t_bin, t_bin)
        binned = subset.copy()
        binned["t_bin"] = pd.cut(binned["t_rel"], bins=bins, labels=bins[:-1], include_lowest=True)
        binned["t_bin"] = binned["t_bin"].astype(float)

        trial_means = binned.groupby(["global_trial_id", "t_bin"])["speed"].mean().reset_index()
        agg = trial_means.groupby("t_bin")["speed"]
        mean = agg.mean()
        sem = agg.sem().fillna(0)
        t_vals = mean.index.values

        ax.plot(t_vals, mean.values, color="white", lw=4.0, alpha=0.8, solid_capstyle="round")
        ax.plot(t_vals, mean.values, color=cond_color, lw=2.0, alpha=1.0, label=cond)
        ax.fill_between(t_vals, (mean - sem).values, (mean + sem).values,
                        color=cond_color, alpha=0.2, edgecolor="none")

        ax.set_title(cond, fontweight="bold")
        if j == 0:
            ax.set_ylabel("Escape Speed (mm/s)")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.legend(loc="upper right", frameon=False)

        _add_threshold_lines(ax)

    # ── Oscilloscope channels ──
    for j, cond in enumerate(conditions):
        ax = ax_lower[j]
        subset = df[df["type"] == cond]
        vis_baseline = 1.0
        wind_baseline = 3.0

        if "visual" in cond.lower() or "looming" in cond.lower():
            stim_t_rel = subset["t_rel"]
            t_loom_start = stim_t_rel.min() if not stim_t_rel.empty else df["t_rel"].min()
            t_loom = np.array([t_loom_start, 0.0])
            ax.fill_between(t_loom, vis_baseline, vis_baseline + 1.0, step="mid",
                            color=COLOR_OSCI_VIS, alpha=0.6, label="Visual (looming)")

        if not subset.empty:
            first_tid = subset["global_trial_id"].iloc[0]
            grp = subset[subset["global_trial_id"] == first_tid].sort_values("t_rel")

            if "wind" in cond.lower() or "puff" in cond.lower() or grp["stim_state"].max() > 0:
                t_wind = grp["t_rel"].values
                stim = grp["stim_state"].values.astype(float)

                dt_last = t_wind[-1] - t_wind[-2] if len(t_wind) > 1 else 1.0
                t_wind_ext = np.append(t_wind, t_wind[-1] + dt_last)
                stim_ext = np.append(stim, stim[-1])

                ax.fill_between(t_wind_ext, wind_baseline, wind_baseline + stim_ext,
                                step="post", color=COLOR_OSCI_HW, alpha=0.6, label="Wind (stim_state)")

        ax.set_ylim(0, 5)
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("Time relative to TTC (ms)")
        ax.grid(False)
        if j == 0:
            ax.legend(loc="upper right", frameon=False, ncol=2)

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
# Plot 7 — Single-Trial Speed Kinetics (Escape or PreWalk)
# ══════════════════════════════════════════════════════════════════════


def plot_single_trial_kinetics(
    trial: pd.DataFrame,
    latency_ms: float,
    v_max: float,
    global_trial_index: int,
    response_type: str = "Escape",
    figsize: tuple[float, float] = (6.0, 3.5),
) -> plt.Figure:
    """
    Speed kinetics for a single trial with latency marker and threshold lines.

    Works for both Escape and PreWalk trials.
    """
    fig, ax = plt.subplots(figsize=figsize)

    t = trial["t_rel"].values
    spd = trial["speed"].values

    # Color by response type
    curve_color = COLOR_ESCAPE if response_type == "Escape" else COLOR_PREWALK

    ax.plot(t, spd, color=curve_color, lw=1.0, alpha=0.85, label="Speed")

    # ── Latency marker ──
    if not np.isnan(latency_ms):
        ax.axvline(x=latency_ms, color="#3C5488", ls="--", lw=0.9, alpha=0.9,
                   label=f"Latency = {latency_ms:.1f} ms")
        lat_idx = np.argmin(np.abs(t - latency_ms))
        ax.scatter([latency_ms], [spd[lat_idx]], c="#3C5488", s=30, zorder=5,
                   edgecolors="white", linewidths=0.5)

    # ── Threshold reference lines ──
    ax.axhline(y=ESCAPE_VMAX_THRESHOLD, color="k", linestyle="--", linewidth=0.75, alpha=0.6)
    ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02,
            ESCAPE_VMAX_THRESHOLD + 1.5,
            f"Vmax ({ESCAPE_VMAX_THRESHOLD:.0f})", fontsize=6, color="k", alpha=0.6)

    ax.axhline(y=ESCAPE_START_THRESHOLD, color="0.5", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02,
            ESCAPE_START_THRESHOLD + 1.5,
            f"Start ({ESCAPE_START_THRESHOLD:.0f})", fontsize=6, color="0.5", alpha=0.4)

    # ── Labels & annotation ──
    ax.set_xlabel("Time relative to TTC (ms)")
    ax.set_ylabel("Escape Speed (mm/s)")
    ax.set_title(f"Trial {global_trial_index} — {response_type}  (V$_{{max}}$={v_max:.1f} mm/s)", fontweight="bold")

    latency_str = f"{latency_ms:.1f}" if not np.isnan(latency_ms) else "N/A"
    info_text = f"Latency: {latency_str} ms\nV$_{{max}}$: {v_max:.1f} mm/s"
    ax.text(0.98, 0.95, info_text, transform=ax.transAxes, fontsize=6,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.9))

    ax.legend(loc="upper left", frameon=False, fontsize=6)
    fig.tight_layout(pad=1.0)
    return fig
