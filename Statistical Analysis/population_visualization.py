"""
Population Visualization — Group-Level Kinematics Plots
========================================================
Reads population_timeseries.parquet from aggregator.py and produces
three NPG-spec publication-grade figures:

    1. Population Trajectory Overlay — all (x, y) spatial trajectories
    2. Population Speed Kinetics — mean ± SEM with oscilloscope waveforms
    3. Population Spaghetti Kinetics — individual traces + population mean

Usage:
    python population_visualization.py --input population_timeseries.parquet --output results/
    python population_visualization.py --input timeseries.parquet --output figs/ --stim-type looming_wind
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Publication Style & NPG Palette
# ──────────────────────────────────────────────────────────────────────

def _apply_publication_style():
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


_apply_publication_style()

# NPG palette
COLOR_LEFT = "#4DBBD5"
COLOR_RIGHT = "#E64B35"
COLOR_CONTROL = "#999999"
COLOR_OSCI_VIS = "#8491B4"
COLOR_OSCI_HW = "#F39B7F"

SCALE_BAR_MM = 5.0


def _resolve_color(screen_side: str, trial_type: str, control_type: str = "baseline_visual") -> str:
    """Resolve trial to NPG color based on screen_side and type."""
    if trial_type == control_type:
        return COLOR_CONTROL
    side = str(screen_side).strip().lower()
    if side == "left":
        return COLOR_LEFT
    if side == "right":
        return COLOR_RIGHT
    return COLOR_CONTROL


# ══════════════════════════════════════════════════════════════════════
# Drawing Helpers
# ══════════════════════════════════════════════════════════════════════

def _draw_cross_axes(ax: plt.Axes, scale_bar_val: float = SCALE_BAR_MM):
    """Draw cross-shaped origin axes with arrowheads and a minimalist scale bar."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.annotate("", xy=(xlim[1], 0), xytext=(xlim[0], 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.75))
    ax.annotate("", xy=(0, ylim[1]), xytext=(0, ylim[0]),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.75))

    sb_x = xlim[1] * 0.65
    sb_y = ylim[0] * 0.85
    ax.plot([sb_x, sb_x + scale_bar_val], [sb_y, sb_y], "k-", lw=1.0, solid_capstyle="butt")
    ax.text(sb_x + scale_bar_val / 2, sb_y - (ylim[1] - ylim[0]) * 0.02,
            f"{scale_bar_val:.0f} mm", ha="center", va="top", fontsize=7)


def _draw_side_arrows(ax: plt.Axes, left_color: str = COLOR_LEFT, right_color: str = COLOR_RIGHT):
    """Draw minimalist vector arrows on LEFT and RIGHT edges."""
    arrow_style = dict(arrowstyle="-|>", color=None, lw=1.0, mutation_scale=8)

    ax.annotate("", xy=(0.04, 0.5), xytext=(-0.02, 0.5),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops={**arrow_style, "color": left_color})
    ax.text(0.01, 0.45, "Left Stimulus",
            transform=ax.transAxes, ha="center", va="top", fontsize=7, color=left_color)

    ax.annotate("", xy=(0.96, 0.5), xytext=(1.02, 0.5),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops={**arrow_style, "color": right_color})
    ax.text(0.99, 0.45, "Right Stimulus",
            transform=ax.transAxes, ha="center", va="top", fontsize=7, color=right_color)


# ══════════════════════════════════════════════════════════════════════
# Plot 1 — Population Trajectory Overlay
# ══════════════════════════════════════════════════════════════════════

def plot_population_trajectory_overlay(
    ts: pd.DataFrame,
    control_type: str = "baseline_visual",
    figsize_per_ax: tuple[float, float] = (4.0, 4.0),
) -> plt.Figure:
    """
    One subplot per trial type. All trials from all subjects overlaid.
    Left stimuli in NPG blue, right in NPG red, control in neutral gray.
    Cross-shaped axes with minimalist side-arrow indicators.
    """
    all_types = sorted(ts["type"].dropna().unique())
    if not all_types:
        log.warning("No trial types found for trajectory overlay.")
        fig, _ = plt.subplots(figsize=figsize_per_ax)
        return fig

    n = len(all_types)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_ax[0] * n, figsize_per_ax[1]),
                             squeeze=False)
    axes = axes[0]

    for idx, ttype in enumerate(all_types):
        ax = axes[idx]
        subset = ts[ts["type"] == ttype]

        # Group by unique trial (subject_id + global_trial_id)
        trial_key = ["subject_id", "global_trial_id"] if "subject_id" in subset.columns \
            else ["global_trial_id"]
        for _, grp in subset.groupby(trial_key):
            grp_sorted = grp.sort_values("sys_time")
            ss = str(grp_sorted["screen_side"].iloc[0]).strip().lower() \
                if pd.notna(grp_sorted["screen_side"].iloc[0]) else ""
            color = _resolve_color(ss, ttype, control_type)
            ax.plot(grp_sorted["x"], grp_sorted["y"], color=color, alpha=0.3, lw=0.6)

        ax.set_aspect("equal")
        ax.set_title(ttype, fontweight="bold")
        _draw_cross_axes(ax)
        _draw_side_arrows(ax)

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Plot 2 — Population Speed Kinetics
# ══════════════════════════════════════════════════════════════════════

def _two_step_aggregate(
    subset: pd.DataFrame,
    t_bin: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-step aggregation to avoid pseudoreplication:
        Step 1: per-subject mean within each time bin (averaging across trials)
        Step 2: population mean ± SEM across subjects

    Returns (t_vals, mean, sem)
    """
    t_min = subset["t_rel"].min()
    t_max = subset["t_rel"].max()
    bins = np.arange(t_min, t_max + t_bin, t_bin)

    binned = subset.copy()
    binned["t_bin"] = pd.cut(binned["t_rel"], bins=bins, labels=bins[:-1], include_lowest=True)
    binned["t_bin"] = binned["t_bin"].astype(float)

    # Step 1: per-subject mean within each time bin
    group_key = ["subject_id", "t_bin"] if "subject_id" in binned.columns else ["global_trial_id", "t_bin"]
    subject_means = (binned.groupby(group_key)["speed"]
                     .mean()
                     .reset_index())

    # Step 2: population mean ± SEM across subjects
    agg = subject_means.groupby("t_bin")["speed"]
    mean = agg.mean()
    sem = agg.sem()

    t_vals = mean.index.values
    return t_vals, mean.values, sem.values


def plot_population_speed_kinetics(
    ts: pd.DataFrame,
    control_type: str = "baseline_visual",
    stim_type: str = "looming_wind",
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """
    Two-panel figure (4:1 height ratio) with shared X axis.

    Upper panel: Escape speed (mean ± SEM) per condition, two-step aggregated.
    Lower panel: Oscilloscope-style dual-channel waveforms.
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.08)
    ax_main = fig.add_subplot(gs[0])
    ax_stim = fig.add_subplot(gs[1], sharex=ax_main)

    # Condition color mapping
    cond_colors = {}
    for ttype in ts["type"].dropna().unique():
        if ttype == control_type:
            cond_colors[ttype] = COLOR_CONTROL
        else:
            # Use screen_side from first occurrence
            sample = ts[ts["type"] == ttype].iloc[0]
            ss = str(sample.get("screen_side", "")).strip().lower()
            if ss == "left":
                cond_colors[ttype] = COLOR_LEFT
            elif ss == "right":
                cond_colors[ttype] = COLOR_RIGHT
            else:
                cond_colors[ttype] = COLOR_LEFT  # default

    for cond, color in cond_colors.items():
        subset = ts[ts["type"] == cond]
        if subset.empty:
            continue

        t_vals, mean, sem = _two_step_aggregate(subset)
        ax_main.plot(t_vals, mean, color=color, lw=1.0, label=cond)
        ax_main.fill_between(t_vals, mean - sem, mean + sem,
                             color=color, alpha=0.2, edgecolor="none")

    ax_main.set_ylabel("Escape Speed (mm/s)")
    ax_main.legend(loc="upper right", frameon=False)
    ax_main.set_xlabel("")
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # ── Lower panel: oscilloscope waveforms ──
    vis_baseline = 1.0
    wind_baseline = 3.0

    # Channel 1: Visual looming
    stim_t_rel = ts.loc[ts["type"] == stim_type, "t_rel"]
    t_loom_start = stim_t_rel.min() if not stim_t_rel.empty else ts["t_rel"].min()
    t_loom = np.array([t_loom_start, 0.0])
    ax_stim.fill_between(t_loom, vis_baseline, vis_baseline + 1.0,
                         step="mid", color=COLOR_OSCI_VIS, alpha=0.6,
                         label="Visual (looming)")

    # Channel 2: Wind stim_state from first trial of stim_type
    stim_subset = ts[ts["type"] == stim_type]
    if not stim_subset.empty:
        # Pick first trial
        trial_key = ["subject_id", "global_trial_id"] if "subject_id" in stim_subset.columns \
            else ["global_trial_id"]
        first_trial = stim_subset.groupby(trial_key).first().index[0]
        if isinstance(first_trial, tuple):
            grp = stim_subset[(stim_subset["subject_id"] == first_trial[0]) &
                              (stim_subset["global_trial_id"] == first_trial[1])]
        else:
            grp = stim_subset[stim_subset["global_trial_id"] == first_trial]
        grp = grp.sort_values("t_rel")
        t_wind = grp["t_rel"].values
        stim = grp["stim_state"].values.astype(float)

        if len(t_wind) > 1:
            dt_last = t_wind[-1] - t_wind[-2]
        else:
            dt_last = 1.0
        t_wind_ext = np.append(t_wind, t_wind[-1] + dt_last)
        stim_ext = np.append(stim, stim[-1])

        ax_stim.fill_between(t_wind_ext, wind_baseline, wind_baseline + stim_ext,
                             step="post", color=COLOR_OSCI_HW, alpha=0.6,
                             label="Wind (stim_state)")

    ax_stim.set_ylim(0, 5)
    ax_stim.set_yticks([])
    ax_stim.set_ylabel("")
    ax_stim.set_xlabel("Time relative to TTC (ms)")
    ax_stim.legend(loc="upper right", frameon=False, ncol=2)
    ax_stim.grid(False)

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Plot 3 — Population Spaghetti Kinetics
# ══════════════════════════════════════════════════════════════════════

def plot_population_spaghetti_kinetics(
    ts: pd.DataFrame,
    control_type: str = "baseline_visual",
    stim_type: str = "looming_wind",
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """
    Two-panel figure (4:1 height ratio) with shared X axis.

    Upper panel: Background layer with all individual speed traces (alpha=0.05, lw=0.2),
    foreground layer with population mean ± SEM (lw=1.5).
    Lower panel: Oscilloscope-style dual-channel stimulus waveforms.
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.08)
    ax_main = fig.add_subplot(gs[0])
    ax_stim = fig.add_subplot(gs[1], sharex=ax_main)

    # Condition color mapping
    cond_colors = {}
    for ttype in ts["type"].dropna().unique():
        if ttype == control_type:
            cond_colors[ttype] = COLOR_CONTROL
        else:
            sample = ts[ts["type"] == ttype].iloc[0]
            ss = str(sample.get("screen_side", "")).strip().lower()
            if ss == "left":
                cond_colors[ttype] = COLOR_LEFT
            elif ss == "right":
                cond_colors[ttype] = COLOR_RIGHT
            else:
                cond_colors[ttype] = COLOR_LEFT

    for cond, color in cond_colors.items():
        subset = ts[ts["type"] == cond]
        if subset.empty:
            continue

        # Background layer: individual trial traces
        trial_key = ["subject_id", "global_trial_id"] if "subject_id" in subset.columns \
            else ["global_trial_id"]
        for _, grp in subset.groupby(trial_key):
            grp_sorted = grp.sort_values("t_rel")
            ax_main.plot(grp_sorted["t_rel"], grp_sorted["speed"],
                         color=color, lw=0.2, alpha=0.05)

        # Foreground layer: population mean ± SEM (two-step aggregation)
        t_vals, mean, sem = _two_step_aggregate(subset)
        ax_main.plot(t_vals, mean, color=color, lw=1.5, alpha=1.0, label=cond)
        ax_main.fill_between(t_vals, mean - sem, mean + sem,
                             color=color, alpha=0.2, edgecolor="none")

    ax_main.set_ylabel("Escape Speed (mm/s)")
    ax_main.legend(loc="upper right", frameon=False)
    ax_main.set_xlabel("")
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # ── Lower panel: oscilloscope waveforms ──
    vis_baseline = 1.0
    wind_baseline = 3.0

    stim_t_rel = ts.loc[ts["type"] == stim_type, "t_rel"]
    t_loom_start = stim_t_rel.min() if not stim_t_rel.empty else ts["t_rel"].min()
    t_loom = np.array([t_loom_start, 0.0])
    ax_stim.fill_between(t_loom, vis_baseline, vis_baseline + 1.0,
                         step="mid", color=COLOR_OSCI_VIS, alpha=0.6,
                         label="Visual (looming)")

    stim_subset = ts[ts["type"] == stim_type]
    if not stim_subset.empty:
        trial_key = ["subject_id", "global_trial_id"] if "subject_id" in stim_subset.columns \
            else ["global_trial_id"]
        first_trial = stim_subset.groupby(trial_key).first().index[0]
        if isinstance(first_trial, tuple):
            grp = stim_subset[(stim_subset["subject_id"] == first_trial[0]) &
                              (stim_subset["global_trial_id"] == first_trial[1])]
        else:
            grp = stim_subset[stim_subset["global_trial_id"] == first_trial]
        grp = grp.sort_values("t_rel")
        t_wind = grp["t_rel"].values
        stim = grp["stim_state"].values.astype(float)

        if len(t_wind) > 1:
            dt_last = t_wind[-1] - t_wind[-2]
        else:
            dt_last = 1.0
        t_wind_ext = np.append(t_wind, t_wind[-1] + dt_last)
        stim_ext = np.append(stim, stim[-1])

        ax_stim.fill_between(t_wind_ext, wind_baseline, wind_baseline + stim_ext,
                             step="post", color=COLOR_OSCI_HW, alpha=0.6,
                             label="Wind (stim_state)")

    ax_stim.set_ylim(0, 5)
    ax_stim.set_yticks([])
    ax_stim.set_ylabel("")
    ax_stim.set_xlabel("Time relative to TTC (ms)")
    ax_stim.legend(loc="upper right", frameon=False, ncol=2)
    ax_stim.grid(False)

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════

def run_visualization_pipeline(
    timeseries_path: str | Path,
    output_dir: str | Path | None = None,
    control_type: str = "baseline_visual",
    stim_type: str = "looming_wind",
) -> tuple[plt.Figure, plt.Figure, plt.Figure]:
    """
    Full population visualization pipeline.

    Parameters
    ----------
    timeseries_path : path to population_timeseries.parquet
    output_dir : directory to save figures
    control_type : trial type label for control condition
    stim_type : trial type label for stimulus condition

    Returns
    -------
    (fig_trajectory, fig_speed, fig_spaghetti)
    """
    ts = pd.read_parquet(timeseries_path)
    log.info("Loaded population_timeseries.parquet: %d frames, %d subjects",
             len(ts), ts["subject_id"].nunique() if "subject_id" in ts.columns else 0)

    fig_traj = plot_population_trajectory_overlay(ts, control_type=control_type)
    fig_speed = plot_population_speed_kinetics(ts, control_type=control_type, stim_type=stim_type)
    fig_spaghetti = plot_population_spaghetti_kinetics(ts, control_type=control_type, stim_type=stim_type)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        fig_traj.savefig(out / "population_trajectory_overlay.png", dpi=300, bbox_inches="tight")
        fig_speed.savefig(out / "population_speed_kinetics.png", dpi=300, bbox_inches="tight")
        fig_spaghetti.savefig(out / "population_spaghetti_kinetics.png", dpi=300, bbox_inches="tight")
        log.info("Figures saved to %s", out)

    return fig_traj, fig_speed, fig_spaghetti


# ══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Population visualization — group-level trajectory & kinetics plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to population_timeseries.parquet")
    p.add_argument("--output", default=".", help="Output directory for figures (default: current dir)")
    p.add_argument("--control-type", default="baseline_visual",
                   help="Trial type for control condition (default: baseline_visual)")
    p.add_argument("--stim-type", default="looming_wind",
                   help="Trial type for stimulus condition (default: looming_wind)")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    figs = run_visualization_pipeline(
        timeseries_path=args.input,
        output_dir=args.output,
        control_type=args.control_type,
        stim_type=args.stim_type,
    )
    return figs


if __name__ == "__main__":
    main()
