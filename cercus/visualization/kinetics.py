"""
Cercus Framework — Kinetics Plots (Speed & Angular Velocity)
=============================================================
"""

from __future__ import annotations

import logging

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.constants import (
    COLOR_CONTROL,
    COLOR_ESCAPE,
    COLOR_LEFT,
    COLOR_NO_RESPONSE,
    COLOR_PREWALK,
    COLOR_RIGHT,
    ESCAPE_START_THRESHOLD,
    ESCAPE_VMAX_THRESHOLD,
    _get_unified_side,
)
from cercus.visualization._core import (
    add_threshold_lines,
    draw_oscilloscope_channels,
)

log = logging.getLogger(__name__)


def plot_speed_kinetics(
    df: pd.DataFrame,
    control_type: str = "baseline_visual",
    stim_type: str = "looming_wind",
    figsize: tuple[float, float] = (10, 6),
    y_col: str = "speed",
    y_label: str = "Escape Speed (mm/s)",
) -> plt.Figure:
    """Two-panel figure (4:1 height ratio) with shared X axis."""
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.08)
    ax_main = fig.add_subplot(gs[0])
    ax_stim = fig.add_subplot(gs[1], sharex=ax_main)

    t_bin = 5.0
    t_min = df["t_rel"].min()
    t_max = df["t_rel"].max()
    bins = np.arange(t_min, t_max + t_bin, t_bin)
    df_binned = df.copy()
    df_binned["t_bin"] = pd.cut(
        df_binned["t_rel"], bins=bins, labels=bins[:-1], include_lowest=True
    )
    df_binned["t_bin"] = df_binned["t_bin"].astype(float)

    cond_colors: dict[str, str] = {}
    for ttype in df["type"].dropna().unique():
        if ttype == control_type:
            cond_colors[ttype] = COLOR_CONTROL
        else:
            sample = df[df["type"] == ttype].iloc[0]
            ss = _get_unified_side(sample)
            cond_colors[ttype] = (
                COLOR_LEFT if ss == "left" else COLOR_RIGHT if ss == "right" else COLOR_LEFT
            )

    for cond, color in cond_colors.items():
        subset = df_binned[df_binned["type"] == cond]
        if subset.empty:
            continue
        trial_means = (
            subset.groupby(["global_trial_id", "t_bin"])[y_col].mean().reset_index()
        )
        agg = trial_means.groupby("t_bin")[y_col]
        mean = agg.mean()
        sem = agg.sem().fillna(0)
        t_vals = mean.index.values
        ax_main.plot(t_vals, mean.values, color=color, lw=1.0, label=cond)
        ax_main.fill_between(
            t_vals,
            (mean - sem).values,
            (mean + sem).values,
            color=color,
            alpha=0.2,
            edgecolor="none",
        )

    ax_main.set_ylabel(y_label)
    ax_main.legend(loc="upper right", frameon=False)
    ax_main.set_xlabel("")
    plt.setp(ax_main.get_xticklabels(), visible=False)

    if y_col == "speed":
        add_threshold_lines(ax_main)
    else:
        ax_main.axhline(y=0, color="0.5", linestyle="--", linewidth=0.5, alpha=0.4)

    for cond in df["type"].dropna().unique():
        draw_oscilloscope_channels(ax_stim, df, cond)
    ax_stim.legend(loc="upper right", frameon=False, ncol=2)

    fig.tight_layout(pad=1.0)
    return fig


def plot_population_speed_kinetics(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (10, 6),
    y_col: str = "speed",
    y_label: str = "Speed (mm/s)",
    t_window: tuple[float, float] = (-1000.0, 500.0),
) -> plt.Figure:
    """Population-level speed kinetics split by response type."""
    df = df[
        (df["t_rel"] >= t_window[0]) & (df["t_rel"] <= t_window[1])
    ].copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "No data in requested time window",
            ha="center", va="center", transform=ax.transAxes, fontsize=10, color="0.5",
        )
        return fig

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.08)
    ax_main = fig.add_subplot(gs[0])
    ax_stim = fig.add_subplot(gs[1], sharex=ax_main)

    t_bin = 5.0
    t_min = df["t_rel"].min()
    t_max = df["t_rel"].max()
    bins = np.arange(t_min, t_max + t_bin, t_bin)
    df_binned = df.copy()
    df_binned["t_bin"] = pd.cut(
        df_binned["t_rel"], bins=bins, labels=bins[:-1], include_lowest=True
    )
    df_binned["t_bin"] = df_binned["t_bin"].astype(float)

    response_colors = {
        "Escape": COLOR_ESCAPE,
        "PreWalk": COLOR_PREWALK,
        "NoResponse": COLOR_NO_RESPONSE,
    }

    for response_type, color in response_colors.items():
        subset = df_binned[df_binned["response_type"] == response_type]
        if subset.empty:
            continue
        trial_means = (
            subset.groupby(["subject_id", "global_trial_id", "t_bin"])[y_col]
            .mean()
            .reset_index()
        )
        agg = trial_means.groupby("t_bin")[y_col]
        mean = agg.mean()
        sem = agg.sem().fillna(0)
        t_vals = mean.index.values
        ax_main.plot(t_vals, mean.values, color=color, lw=1.2, label=response_type)
        ax_main.fill_between(
            t_vals,
            (mean - sem).values,
            (mean + sem).values,
            color=color,
            alpha=0.15,
            edgecolor="none",
        )

    ax_main.set_ylabel(y_label)
    ax_main.legend(loc="upper right", frameon=False)
    ax_main.set_xlabel("")
    plt.setp(ax_main.get_xticklabels(), visible=False)

    if y_col == "speed":
        add_threshold_lines(ax_main)
    else:
        ax_main.axhline(y=0, color="0.5", linestyle="--", linewidth=0.5, alpha=0.4)

    for cond in df["type"].dropna().unique():
        draw_oscilloscope_channels(ax_stim, df, cond)
    ax_stim.legend(loc="upper right", frameon=False, ncol=2)

    fig.tight_layout(pad=1.0)
    return fig


def plot_population_spaghetti_kinetics(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (12, 8),
    y_col: str = "speed",
    y_label: str = "Speed (mm/s)",
    t_window: tuple[float, float] = (-1000.0, 500.0),
) -> plt.Figure:
    """Population-level spaghetti plot split by response type."""
    df = df[
        (df["t_rel"] >= t_window[0]) & (df["t_rel"] <= t_window[1])
    ].copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5, "No data in requested time window",
            ha="center", va="center", transform=ax.transAxes, fontsize=10, color="0.5",
        )
        return fig

    response_types = ["Escape", "PreWalk", "NoResponse"]
    response_colors = {
        "Escape": COLOR_ESCAPE,
        "PreWalk": COLOR_PREWALK,
        "NoResponse": COLOR_NO_RESPONSE,
    }

    n_panels = len(response_types)
    fig = plt.figure(figsize=(figsize[0] * n_panels / 3, figsize[1]))
    gs = gridspec.GridSpec(
        2, n_panels, height_ratios=[4, 1], hspace=0.1, wspace=0.15, figure=fig
    )

    ax_upper: list[plt.Axes] = []
    ax_lower: list[plt.Axes] = []
    for j in range(n_panels):
        sharey = ax_upper[0] if ax_upper else None
        ax_u = fig.add_subplot(gs[0, j], sharey=sharey)
        ax_upper.append(ax_u)
        ax_l = fig.add_subplot(gs[1, j], sharex=ax_u)
        ax_lower.append(ax_l)

    for j, response_type in enumerate(response_types):
        ax = ax_upper[j]
        subset = df[df["response_type"] == response_type]
        color = response_colors[response_type]

        if subset.empty:
            ax.set_title(response_type, fontweight="bold")
            continue

        for (_subj, _tid), grp in subset.groupby(
            ["subject_id", "global_trial_id"]
        ):
            grp_sorted = grp.sort_values("t_rel")
            ax.plot(
                grp_sorted["t_rel"],
                grp_sorted[y_col],
                color=color,
                lw=0.3,
                alpha=0.15,
            )

        t_bin = 5.0
        t_min = subset["t_rel"].min()
        t_max = subset["t_rel"].max()
        bins = np.arange(t_min, t_max + t_bin, t_bin)
        binned = subset.copy()
        binned["t_bin"] = pd.cut(
            binned["t_rel"], bins=bins, labels=bins[:-1], include_lowest=True
        )
        binned["t_bin"] = binned["t_bin"].astype(float)

        trial_means = (
            binned.groupby(["subject_id", "global_trial_id", "t_bin"])[y_col]
            .mean()
            .reset_index()
        )
        agg = trial_means.groupby("t_bin")[y_col]
        mean = agg.mean()
        sem = agg.sem().fillna(0)
        t_vals = mean.index.values

        ax.plot(
            t_vals, mean.values, color="white", lw=4.0, alpha=0.8, solid_capstyle="round"
        )
        ax.plot(
            t_vals, mean.values, color=color, lw=2.0, alpha=1.0, label=response_type
        )
        ax.fill_between(
            t_vals,
            (mean - sem).values,
            (mean + sem).values,
            color=color,
            alpha=0.2,
            edgecolor="none",
        )

        ax.set_title(response_type, fontweight="bold")
        if j == 0:
            ax.set_ylabel(y_label)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.legend(loc="upper right", frameon=False)

        if y_col == "speed":
            add_threshold_lines(ax)
        else:
            ax.axhline(y=0, color="0.5", linestyle="--", linewidth=0.5, alpha=0.4)

    for j in range(n_panels):
        for cond in df["type"].dropna().unique():
            draw_oscilloscope_channels(ax_lower[j], df, cond)
        if j == 0:
            ax_lower[j].legend(loc="upper right", frameon=False, ncol=2)

    fig.tight_layout(pad=1.0)
    return fig


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
            cond_color_map[ttype] = (
                COLOR_LEFT
                if ss == "left"
                else COLOR_RIGHT if ss == "right" else COLOR_LEFT
            )

    conditions = sorted(cond_color_map.keys())
    n_conds = len(conditions)
    if n_conds == 0:
        fig, ax = plt.subplots()
        return fig

    fig = plt.figure(figsize=(figsize_per_col * n_conds, row_height * 2))
    gs = gridspec.GridSpec(
        2, n_conds, height_ratios=[4, 1], hspace=0.1, wspace=0.15, figure=fig
    )

    ax_upper: list[plt.Axes] = []
    ax_lower: list[plt.Axes] = []
    for j in range(n_conds):
        sharey = ax_upper[0] if ax_upper else None
        ax_u = fig.add_subplot(gs[0, j], sharey=sharey)
        ax_upper.append(ax_u)
        ax_l = fig.add_subplot(gs[1, j], sharex=ax_u)
        ax_lower.append(ax_l)

    show_latency = y_col != "speed"

    for j, cond in enumerate(conditions):
        ax = ax_upper[j]
        subset = df[df["type"] == cond]
        cond_color = cond_color_map[cond]

        if subset.empty:
            ax.set_title(cond, fontweight="bold")
            continue

        for _tid, grp in subset.groupby("global_trial_id"):
            grp_sorted = grp.sort_values("t_rel")
            ax.plot(
                grp_sorted["t_rel"],
                grp_sorted[y_col],
                color=cond_color,
                lw=0.5,
                alpha=0.25,
            )

        t_bin = 5.0
        t_min = subset["t_rel"].min()
        t_max = subset["t_rel"].max()
        bins = np.arange(t_min, t_max + t_bin, t_bin)
        binned = subset.copy()
        binned["t_bin"] = pd.cut(
            binned["t_rel"], bins=bins, labels=bins[:-1], include_lowest=True
        )
        binned["t_bin"] = binned["t_bin"].astype(float)

        trial_means = (
            binned.groupby(["global_trial_id", "t_bin"])[y_col]
            .mean()
            .reset_index()
        )
        agg = trial_means.groupby("t_bin")[y_col]
        mean = agg.mean()
        sem = agg.sem().fillna(0)
        t_vals = mean.index.values

        ax.plot(
            t_vals, mean.values, color="white", lw=4.0, alpha=0.8, solid_capstyle="round"
        )
        ax.plot(
            t_vals, mean.values, color=cond_color, lw=2.0, alpha=1.0, label=cond
        )
        ax.fill_between(
            t_vals,
            (mean - sem).values,
            (mean + sem).values,
            color=cond_color,
            alpha=0.2,
            edgecolor="none",
        )

        if show_latency:
            trial_lats = (
                subset.groupby("global_trial_id")["latency_ms"].first().dropna()
            )
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
            add_threshold_lines(ax)
        else:
            ax.axhline(y=0, color="0.5", linestyle="--", linewidth=0.5, alpha=0.4)

    for j, cond in enumerate(conditions):
        draw_oscilloscope_channels(ax_lower[j], df, cond)
        if j == 0:
            ax_lower[j].legend(loc="upper right", frameon=False, ncol=2)

    fig.tight_layout(pad=1.0)
    return fig


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
    """Single-trial kinetics with latency marker."""
    fig, ax = plt.subplots(figsize=figsize)

    t = trial["t_rel"].values
    y_vals = trial[y_col].values

    curve_color = COLOR_ESCAPE if response_type == "Escape" else COLOR_PREWALK

    ax.plot(t, y_vals, color=curve_color, lw=1.0, alpha=0.85, label=y_label)

    onset_ms = latency_ms if np.isnan(interval_onset_ms) else interval_onset_ms
    if not np.isnan(onset_ms):
        ax.axvline(
            x=onset_ms,
            color="#3C5488",
            ls="--",
            lw=0.9,
            alpha=0.9,
            label=f"Escape onset = {onset_ms:.1f} ms",
        )
        onset_idx = np.argmin(np.abs(t - onset_ms))
        ax.scatter(
            [onset_ms],
            [y_vals[onset_idx]],
            c="#3C5488",
            s=30,
            zorder=5,
            edgecolors="white",
            linewidths=0.5,
        )

    if y_col == "speed" and not np.isnan(interval_onset_ms) and not np.isnan(interval_offset_ms):
        ax.axvspan(
            interval_onset_ms, interval_offset_ms, alpha=0.10, color="#E64B35", zorder=0
        )

    if y_col == "speed":
        ax.axhline(
            y=ESCAPE_VMAX_THRESHOLD, color="k", linestyle="--", linewidth=0.75, alpha=0.6
        )
        ax.text(
            ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02,
            ESCAPE_VMAX_THRESHOLD + 1.5,
            f"Vmax ({ESCAPE_VMAX_THRESHOLD:.0f})",
            fontsize=6,
            color="k",
            alpha=0.6,
        )
        ax.axhline(
            y=ESCAPE_START_THRESHOLD, color="0.5", linestyle="--", linewidth=0.5, alpha=0.4
        )
        ax.text(
            ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02,
            ESCAPE_START_THRESHOLD + 1.5,
            f"Start ({ESCAPE_START_THRESHOLD:.0f})",
            fontsize=6,
            color="0.5",
            alpha=0.4,
        )
    else:
        ax.axhline(y=0, color="0.5", linestyle="--", linewidth=0.5, alpha=0.4)

    ax.set_xlabel("Time relative to TTC (ms)")
    ax.set_ylabel(y_label)
    ax.set_title(
        f"Trial {global_trial_index} — {response_type}  (V$_{{max}}$={v_max:.1f} mm/s)",
        fontweight="bold",
    )

    latency_str = f"{latency_ms:.1f}" if not np.isnan(latency_ms) else "N/A"
    if not np.isnan(interval_onset_ms) and not np.isnan(interval_offset_ms):
        interval_str = f"{interval_offset_ms - interval_onset_ms:.1f}"
    else:
        interval_str = "N/A"
    info_text = f"Latency: {latency_str} ms\nV$_{{max}}$: {v_max:.1f} mm/s\nInterval: {interval_str} ms"
    ax.text(
        0.98,
        0.95,
        info_text,
        transform=ax.transAxes,
        fontsize=6,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.9
        ),
    )

    ax.legend(loc="upper left", frameon=False, fontsize=6)
    fig.tight_layout(pad=1.0)
    return fig
