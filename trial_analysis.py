"""
Cercus Framework — Behavioral Neuroscience Data Analysis & Visualization
=========================================================================
Dual-track recording system: Events (low-freq) + Kinematics (high-freq)
aligned by lifecycle events (trial_start / trial_stop / phase_transition).
Time axis zeroed to the absolute Collision_TTC0 system timestamp extracted
from phase_transition events, with lv_ratio-based fallback for control trials.

Usage:
    python cercus_analysis.py --events path/to/events.csv --kinematics path/to/kinematics.csv
    python cercus_analysis.py --events e.csv --kinematics k.csv --save figures/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Publication-Grade Global Style (Nature / Science / Cell)
# ──────────────────────────────────────────────────────────────────────

def _apply_publication_style():
    """Inject Nature/Science/Cell compliant rcParams."""
    rc = plt.rcParams
    # Font
    rc["font.family"] = "sans-serif"
    rc["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    rc["svg.fonttype"] = "none"
    rc["pdf.fonttype"] = 42
    # Font sizes
    rc["font.size"] = 7
    rc["axes.titlesize"] = 9
    rc["axes.labelsize"] = 8
    rc["legend.fontsize"] = 7
    rc["xtick.labelsize"] = 7
    rc["ytick.labelsize"] = 7
    # Lines & spines
    rc["lines.linewidth"] = 1.0
    rc["axes.linewidth"] = 0.75
    rc["axes.spines.top"] = False
    rc["axes.spines.right"] = False
    # Ticks
    rc["xtick.direction"] = "in"
    rc["ytick.direction"] = "in"
    rc["xtick.major.size"] = 3
    rc["ytick.major.size"] = 3
    rc["xtick.major.width"] = 0.75
    rc["ytick.major.width"] = 0.75
    rc["xtick.minor.size"] = 1.5
    rc["ytick.minor.size"] = 1.5
    # Legend
    rc["legend.frameon"] = False
    rc["legend.borderaxespad"] = 0
    # Figure
    rc["figure.dpi"] = 150
    rc["savefig.dpi"] = 300
    rc["savefig.transparent"] = True


# NPG (Nature Publishing Group) palette
COLOR_LEFT = "#4DBBD5"       # Nature blue
COLOR_RIGHT = "#E64B35"      # Nature red
COLOR_CONTROL = "#999999"    # Neutral gray
COLOR_OSCI_VIS = "#8491B4"   # Oscilloscope visual channel
COLOR_OSCI_HW = "#F39B7F"    # Oscilloscope hardware channel


_apply_publication_style()

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
DETAILS_KEYS = ("type", "target_ttc_ms", "wind_dir", "screen_side",
                "lv_ratio_ms", "init_half_angle_deg")
SPEED_WINDOW_MS = 100          # rolling-mean smoothing window (ms)
SCALE_BAR_MM = 5.0             # scale bar for trajectory plots
LEGACY_TRIAL_DURATION_MS = 5829.6  # fallback when trial_stop is missing


# ══════════════════════════════════════════════════════════════════════
# L0 — Data Loading
# ══════════════════════════════════════════════════════════════════════

def _parse_details(raw: Any) -> dict:
    """Parse a single 'details' cell — may be JSON string, NaN, or dict."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    cleaned = raw.replace('\\"', '"').replace('""', '"').strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        log.debug("Failed to parse details JSON: %s", cleaned[:120])
        return {}


def load_events(path: str | Path) -> tuple[pd.DataFrame, list[dict], dict[Any, float]]:
    """
    Load events CSV. Extract trial time windows, metadata, and TTC anchors.

    Returns
    -------
    trial_meta : DataFrame
        One row per trial with global_trial_id + DETAILS_KEYS.
    trial_windows : list of dict
        Each dict has keys: global_trial_id, t_start, t_stop.
        t_stop falls back to t_start + LEGACY_TRIAL_DURATION_MS if trial_stop missing.
    ttc_anchors : dict
        Mapping global_trial_id → absolute system timestamp of Collision_TTC0
        extracted from phase_transition events.
    """
    df = pd.read_csv(path)
    required = {"event_name", "timestamp", "global_trial_id", "details"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Events CSV missing columns: {missing}")

    starts = df[df["event_name"] == "trial_start"].copy()
    if starts.empty:
        log.warning("No 'trial_start' events found — stimulus metadata unavailable.")
        empty_meta = pd.DataFrame(columns=["global_trial_id", *DETAILS_KEYS])
        return empty_meta, [], {}

    stops = df[df["event_name"] == "trial_stop"].copy()

    # ── Trial boundaries: strict pairing via global_trial_id ──
    stop_lookup: dict[Any, float] = {}
    if not stops.empty:
        stop_lookup = dict(zip(stops["global_trial_id"], stops["timestamp"]))

    trial_windows: list[dict] = []
    for _, row in starts.iterrows():
        tid = row["global_trial_id"]
        t_start = float(row["timestamp"])
        if tid in stop_lookup:
            t_stop = float(stop_lookup[tid])
        else:
            t_stop = t_start + (LEGACY_TRIAL_DURATION_MS / 1000.0)
            log.debug("Trial %s: no trial_stop found, using fallback t_stop=%.1f",
                       tid, t_stop)
        trial_windows.append({
            "global_trial_id": tid,
            "t_start": t_start,
            "t_stop": t_stop,
        })

    # ── TTC anchors from phase_transition → Collision_TTC0 ──
    ttc_anchors: dict[Any, float] = {}
    transitions = df[df["event_name"] == "phase_transition"].copy()
    for _, row in transitions.iterrows():
        details = _parse_details(row["details"])
        if details.get("to_phase") == "Collision_TTC0":
            tid = row["global_trial_id"]
            ttc_anchors[tid] = float(row["timestamp"])
    log.info("Extracted %d Collision_TTC0 anchors from phase_transition events.",
             len(ttc_anchors))

    # ── Metadata from trial_start details ──
    parsed = starts["details"].apply(_parse_details)
    details_df = pd.DataFrame(parsed.tolist(), index=starts.index)
    for k in DETAILS_KEYS:
        if k not in details_df.columns:
            details_df[k] = np.nan

    trial_meta = starts[["global_trial_id"]].join(details_df[list(DETAILS_KEYS)])
    trial_meta = trial_meta.drop_duplicates(subset="global_trial_id", keep="first")
    return trial_meta.reset_index(drop=True), trial_windows, ttc_anchors


def load_kinematics(path: str | Path) -> pd.DataFrame:
    """Load kinematics CSV."""
    df = pd.read_csv(path)
    required = {"sys_time", "dx", "dy", "dz", "stim_state", "global_trial_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kinematics CSV missing columns: {missing}")
    return df


# ══════════════════════════════════════════════════════════════════════
# L0 — Preprocessing: Timestamp-Based Slicing & Integration
# ══════════════════════════════════════════════════════════════════════

def _slice_kinematics_by_window(
    kin: pd.DataFrame,
    window: dict,
) -> pd.DataFrame:
    """
    Boolean-mask slice of kinematics for one trial using [t_start, t_stop].

    Falls back to global_trial_id matching when timestamp-based slice yields
    no rows (e.g. kinematics sys_time uses a different clock than events).
    """
    tid = window["global_trial_id"]
    t_start = window["t_start"]
    t_stop = window["t_stop"]

    # Primary: timestamp-based boolean mask
    mask = (kin["sys_time"] >= t_start) & (kin["sys_time"] <= t_stop)
    trial_kin = kin.loc[mask].copy()

    # Fallback: match by global_trial_id if timestamp slice yields nothing
    if trial_kin.empty:
        trial_kin = kin[kin["global_trial_id"] == tid].copy()
        if not trial_kin.empty:
            log.debug("Trial %s: timestamp slice empty, fell back to global_trial_id match "
                       "(%d frames)", tid, len(trial_kin))

    if trial_kin.empty:
        log.warning("Trial %s: no kinematics data found (t_start=%.1f, t_stop=%.1f). "
                     "Skipping.", tid, t_start, t_stop)

    # Ensure global_trial_id is set consistently
    trial_kin["global_trial_id"] = tid
    return trial_kin


def _integrate_trial(grp: pd.DataFrame, t_zero_sys: float) -> pd.DataFrame:
    """
    Per-trial integration with lifecycle-anchored TTC time axis and denoised speed.

    Time axis zero-point is the absolute Collision_TTC0 system timestamp
    (or a lv_ratio-derived fallback for control trials):
        t_rel = (sys_time - t_zero_sys) * 1000
    t_rel < 0 is before collision, t_rel = 0 is collision moment.

    Speed denoising:
      1. cumsum dx/dy → absolute position, origin at (0,0)
      2. Savitzky-Golay filter on accumulated x and y positions
      3. Speed via central finite differences on smoothed positions
      4. Leading/trailing edge frames trimmed by half the S-G window
    """
    df = grp.copy()

    if len(df) < 2:
        df["t_rel"] = 0.0
        df["x"] = 0.0
        df["y"] = 0.0
        df["speed"] = np.nan
        return df

    # Time: zero-point anchored to lifecycle-derived TTC
    df["t_rel"] = (df["sys_time"] - t_zero_sys) * 1000.0

    # Position: cumulative sum of displacements, origin at (0,0)
    df.loc[df.index[:2], "dx"] = 0.0
    df.loc[df.index[:2], "dy"] = 0.0

    # Force-flip hardware X axis to align with real left-right space
    x_raw = (-df["dx"]).cumsum().values
    y_raw = df["dy"].cumsum().values

    # Savitzky-Golay smoothing on accumulated positions
    median_dt = df["sys_time"].diff().replace(0, np.nan).median()
    if pd.notna(median_dt) and median_dt > 0:
        win = max(5, int(round((SPEED_WINDOW_MS / 1000.0) / median_dt)))
    else:
        win = 11
    # Window must be odd and >= 3
    if win % 2 == 0:
        win += 1
    poly_order = min(3, win - 1)

    n = len(x_raw)
    if n >= win:
        x_smooth = savgol_filter(x_raw, window_length=win, polyorder=poly_order, deriv=0)
        y_smooth = savgol_filter(y_raw, window_length=win, polyorder=poly_order, deriv=0)
    else:
        # Too few points for S-G filter; fall back to raw positions
        x_smooth = x_raw
        y_smooth = y_raw

    df["x"] = x_smooth
    df["y"] = y_smooth

    # Speed via central finite differences on smoothed positions
    dt_sec = df["sys_time"].diff().values  # seconds
    dx_smooth = np.diff(x_smooth, prepend=x_smooth[0])
    dy_smooth = np.diff(y_smooth, prepend=y_smooth[0])
    # For index 0 the prepend gives 0 displacement; use actual first diff instead
    if n > 1:
        dx_smooth[0] = x_smooth[1] - x_smooth[0]
        dy_smooth[0] = y_smooth[1] - y_smooth[0]
    speed = np.sqrt(dx_smooth ** 2 + dy_smooth ** 2) / dt_sec  # mm/s
    speed[0] = np.nan  # dt[0] is NaN from diff()

    # Trim unstable boundary frames (half-window on each side)
    half_win = win // 2
    if half_win > 0:
        speed[:half_win] = np.nan
        speed[-half_win:] = np.nan

    df["speed"] = speed

    return df


def _compute_theoretical_ttc_ms(lv_ratio_ms: float, init_half_angle_deg: float) -> float:
    """
    Compute theoretical TTC (ms) from looming parameters for control trials
    that never triggered a Collision_TTC0 phase_transition.

    TTC = lv_ratio_ms / (1 - sin(init_half_angle_deg * π / 180))
    """
    import math
    rad = math.radians(init_half_angle_deg)
    denom = 1.0 - math.sin(rad)
    if denom <= 0:
        log.warning("init_half_angle_deg=%.1f yields sin≥1; TTC undefined, returning lv_ratio_ms.",
                     init_half_angle_deg)
        return lv_ratio_ms
    return lv_ratio_ms / denom


def preprocess(
    trials_meta: pd.DataFrame,
    trial_windows: list[dict],
    ttc_anchors: dict[Any, float],
    kin: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-trial integration anchored to lifecycle-derived TTC timestamps.

    For each trial:
        • If global_trial_id is in ttc_anchors → use the Collision_TTC0
          system timestamp directly as t_zero_sys.
        • Otherwise (control / baseline trials) → compute theoretical
          t_col_ms from lv_ratio_ms + init_half_angle_deg, then set
          t_zero_sys = t_start + t_col_ms / 1000.
    """
    parts: list[pd.DataFrame] = []
    for window in trial_windows:
        tid = window["global_trial_id"]
        trial_kin = _slice_kinematics_by_window(kin, window)
        if trial_kin.empty:
            continue

        # Merge metadata for this trial
        meta_row = trials_meta[trials_meta["global_trial_id"] == tid]
        if not meta_row.empty:
            for col in DETAILS_KEYS:
                trial_kin[col] = meta_row[col].iloc[0]

        # ── Determine t_zero_sys ──
        if tid in ttc_anchors:
            t_zero_sys = ttc_anchors[tid]
        else:
            # Fallback: derive theoretical TTC from looming parameters
            lv_ratio = np.nan
            init_angle = np.nan
            if not meta_row.empty:
                lv_ratio = meta_row["lv_ratio_ms"].iloc[0] if "lv_ratio_ms" in meta_row.columns else np.nan
                init_angle = meta_row["init_half_angle_deg"].iloc[0] if "init_half_angle_deg" in meta_row.columns else np.nan

            if pd.notna(lv_ratio) and pd.notna(init_angle):
                t_col_ms = _compute_theoretical_ttc_ms(float(lv_ratio), float(init_angle))
                t_zero_sys = window["t_start"] + (t_col_ms / 1000.0)
                log.debug("Trial %s: no TTC anchor, fallback t_col_ms=%.1f ms", tid, t_col_ms)
            else:
                # Last resort: use trial midpoint
                t_zero_sys = (window["t_start"] + window["t_stop"]) / 2.0
                log.warning("Trial %s: no TTC anchor and no lv_ratio/init_angle; "
                            "using trial midpoint as zero.", tid)

        try:
            parts.append(_integrate_trial(trial_kin, t_zero_sys))
        except Exception as exc:
            log.warning("Skipping trial %s during integration: %s", tid, exc)

    if not parts:
        raise RuntimeError("No valid trials after preprocessing.")
    return pd.concat(parts, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════
# Plot 1 — Trajectory Overlay
# ══════════════════════════════════════════════════════════════════════

def _draw_cross_axes(ax: plt.Axes, scale_bar_val: float = SCALE_BAR_MM):
    """Draw cross-shaped origin axes with arrowheads and a minimalist scale bar."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Horizontal axis (y=0)
    ax.annotate("", xy=(xlim[1], 0), xytext=(xlim[0], 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.75))
    # Vertical axis (x=0)
    ax.annotate("", xy=(0, ylim[1]), xytext=(0, ylim[0]),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.75))

    # Scale bar — minimal solid black line, text tight and centered
    sb_x = xlim[1] * 0.65
    sb_y = ylim[0] * 0.85
    ax.plot([sb_x, sb_x + scale_bar_val], [sb_y, sb_y], "k-", lw=1.0, solid_capstyle="butt")
    ax.text(sb_x + scale_bar_val / 2, sb_y - (ylim[1] - ylim[0]) * 0.02,
            f"{scale_bar_val:.0f} mm", ha="center", va="top", fontsize=7)


def _draw_side_arrows(ax: plt.Axes, left_color: str = COLOR_LEFT, right_color: str = COLOR_RIGHT):
    """
    Draw minimalist vector arrows on LEFT and RIGHT edges with text labels below.
    Uses axes-fraction coords. Arrows point inward (airflow toward center).
    """
    arrow_style = dict(
        arrowstyle="-|>",
        color=None,   # set per arrow
        lw=1.0,
        mutation_scale=8,
    )

    # Left arrow → points right (airflow from left toward center)
    ax.annotate(
        "", xy=(0.04, 0.5), xytext=(-0.02, 0.5),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops={**arrow_style, "color": left_color},
    )
    ax.text(0.01, 0.45, "Left Stimulus",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=7, color=left_color)

    # Right arrow → points left (airflow from right toward center)
    ax.annotate(
        "", xy=(0.96, 0.5), xytext=(1.02, 0.5),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops={**arrow_style, "color": right_color},
    )
    ax.text(0.99, 0.45, "Right Stimulus",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=7, color=right_color)


def plot_trajectory_overlay(
    df: pd.DataFrame,
    control_type: str = "baseline_visual",
    left_color: str = COLOR_LEFT,
    right_color: str = COLOR_RIGHT,
    figsize_per_ax: tuple[float, float] = (4.0, 4.0),
):
    """
    One subplot per trial type. Left stimuli in NPG blue, right in NPG red,
    control in neutral gray. Cross-shaped axes with minimalist side-arrow indicators.
    """
    all_types = sorted(df["type"].dropna().unique())
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
        subset = df[df["type"] == ttype]

        for _tid, grp in subset.groupby("global_trial_id"):
            ss = str(grp["screen_side"].iloc[0]).strip().lower() \
                if pd.notna(grp["screen_side"].iloc[0]) else ""
            if ttype == control_type:
                color = COLOR_CONTROL
            elif ss == "left":
                color = left_color
            elif ss == "right":
                color = right_color
            else:
                color = COLOR_CONTROL
            ax.plot(grp["x"], grp["y"], color=color, alpha=0.3, lw=0.6)

        ax.set_aspect("equal")
        ax.set_title(ttype, fontweight="bold")
        _draw_cross_axes(ax)
        _draw_side_arrows(ax, left_color=left_color, right_color=right_color)

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Plot 2 — Speed Kinetics & Oscilloscope Dual-Channel Waveforms
# ══════════════════════════════════════════════════════════════════════

def plot_speed_kinetics(
    df: pd.DataFrame,
    control_type: str = "baseline_visual",
    stim_type: str = "looming_wind",
    figsize: tuple[float, float] = (10, 6),
):
    """
    Two-panel figure (4:1 height ratio) with shared X axis.

    Upper panel: Escape speed (mean ± SEM) per condition.
    Lower panel: Oscilloscope-style dual-channel waveforms:
        • Channel 1 — Visual looming: adaptive coverage ending at t=0
        • Channel 2 — Wind stim_state: real hardware timing via fill_between
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.08)
    ax_main = fig.add_subplot(gs[0])
    ax_stim = fig.add_subplot(gs[1], sharex=ax_main)

    # ── Upper panel: speed mean ± SEM (two-step aggregation) ──
    t_bin = 5.0  # ms
    t_min = df["t_rel"].min()
    t_max = df["t_rel"].max()
    bins = np.arange(t_min, t_max + t_bin, t_bin)
    df_binned = df.copy()
    df_binned["t_bin"] = pd.cut(df_binned["t_rel"], bins=bins,
                                labels=bins[:-1], include_lowest=True)
    df_binned["t_bin"] = df_binned["t_bin"].astype(float)

    cond_colors = {control_type: COLOR_CONTROL, stim_type: COLOR_LEFT}
    for cond, color in cond_colors.items():
        subset = df_binned[df_binned["type"] == cond]
        if subset.empty:
            continue
        # Step 1: per-trial mean within each time bin
        trial_means = (subset.groupby(["global_trial_id", "t_bin"])["speed"]
                       .mean()
                       .reset_index())
        # Step 2: cross-trial mean and SEM
        agg = trial_means.groupby("t_bin")["speed"]
        mean = agg.mean()
        sem = agg.sem()
        t_vals = mean.index.values
        ax_main.plot(t_vals, mean.values, color=color, lw=1.0, label=cond)
        ax_main.fill_between(t_vals, (mean - sem).values, (mean + sem).values,
                             color=color, alpha=0.2, edgecolor="none")

    ax_main.set_ylabel("Escape Speed (mm/s)")
    ax_main.legend(loc="upper right", frameon=False)
    ax_main.set_xlabel("")
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # ── Lower panel: oscilloscope waveforms (no grid, no outlines) ──
    vis_baseline = 1.0
    wind_baseline = 3.0

    # Channel 1: Visual looming — high pulse from condition trial start to TTC
    stim_t_rel = df.loc[df["type"] == stim_type, "t_rel"]
    t_loom_start = stim_t_rel.min() if not stim_t_rel.empty else df["t_rel"].min()
    t_loom = np.array([t_loom_start, 0.0])
    ax_stim.fill_between(t_loom, vis_baseline, vis_baseline + 1.0,
                         step="mid", color=COLOR_OSCI_VIS, alpha=0.6,
                         label="Visual (looming)")

    # Channel 2: Wind stim_state from real hardware data
    stim_subset = df[df["type"] == stim_type]
    if not stim_subset.empty:
        first_tid = stim_subset["global_trial_id"].iloc[0]
        grp = stim_subset[stim_subset["global_trial_id"] == first_tid].sort_values("t_rel")
        t_wind = grp["t_rel"].values
        stim = grp["stim_state"].values.astype(float)

        # Append trailing point to fix step truncation
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


def plot_spaghetti_kinetics(
    df: pd.DataFrame,
    control_type: str = "baseline_visual",
    stim_type: str = "looming_wind",
    figsize: tuple[float, float] = (10, 6),
):
    """
    Two-panel figure (4:1 height ratio) with shared X axis.

    Upper panel: Single-trial speed traces (spaghetti) with population mean ± SEM.
    Lower panel: Oscilloscope-style dual-channel stimulus waveforms.
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.08)
    ax_main = fig.add_subplot(gs[0])
    ax_stim = fig.add_subplot(gs[1], sharex=ax_main)

    # ── Upper panel: spaghetti + mean ± SEM ──
    cond_colors = {control_type: COLOR_CONTROL, stim_type: COLOR_LEFT}
    for cond, color in cond_colors.items():
        subset = df[df["type"] == cond]
        if subset.empty:
            continue

        # Background layer: individual trial traces
        for _tid, grp in subset.groupby("global_trial_id"):
            grp_sorted = grp.sort_values("t_rel")
            ax_main.plot(grp_sorted["t_rel"], grp_sorted["speed"],
                         color=color, lw=0.3, alpha=0.15)

        # Foreground layer: population mean ± SEM (binned two-step aggregation)
        t_bin = 5.0
        t_min = subset["t_rel"].min()
        t_max = subset["t_rel"].max()
        bins = np.arange(t_min, t_max + t_bin, t_bin)
        binned = subset.copy()
        binned["t_bin"] = pd.cut(binned["t_rel"], bins=bins,
                                 labels=bins[:-1], include_lowest=True)
        binned["t_bin"] = binned["t_bin"].astype(float)

        trial_means = (binned.groupby(["global_trial_id", "t_bin"])["speed"]
                       .mean()
                       .reset_index())
        agg = trial_means.groupby("t_bin")["speed"]
        mean = agg.mean()
        sem = agg.sem()
        t_vals = mean.index.values

        ax_main.plot(t_vals, mean.values, color=color, lw=1.5, alpha=1.0, label=cond)
        ax_main.fill_between(t_vals, (mean - sem).values, (mean + sem).values,
                             color=color, alpha=0.2, edgecolor="none")

    ax_main.set_ylabel("Escape Speed (mm/s)")
    ax_main.legend(loc="upper right", frameon=False)
    ax_main.set_xlabel("")
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # ── Lower panel: oscilloscope waveforms ──
    vis_baseline = 1.0
    wind_baseline = 3.0

    stim_t_rel = df.loc[df["type"] == stim_type, "t_rel"]
    t_loom_start = stim_t_rel.min() if not stim_t_rel.empty else df["t_rel"].min()
    t_loom = np.array([t_loom_start, 0.0])
    ax_stim.fill_between(t_loom, vis_baseline, vis_baseline + 1.0,
                         step="mid", color=COLOR_OSCI_VIS, alpha=0.6,
                         label="Visual (looming)")

    stim_subset = df[df["type"] == stim_type]
    if not stim_subset.empty:
        first_tid = stim_subset["global_trial_id"].iloc[0]
        grp = stim_subset[stim_subset["global_trial_id"] == first_tid].sort_values("t_rel")
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
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cercus framework — trajectory & speed analysis (TTC-aligned)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--events", required=True, help="Path to *_events.csv")
    p.add_argument("--kinematics", required=True, help="Path to *_kinematics.csv")
    p.add_argument("--control-type", default="baseline_visual",
                   help="Trial type for control condition (default: baseline_visual)")
    p.add_argument("--stim-type", default="looming_wind",
                   help="Trial type for stimulus condition (default: looming_wind)")
    p.add_argument("--save", default=None,
                   help="Directory to save PNG figures. Omit to show interactively.")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

    # ── Load events (extract trial windows, metadata, TTC anchors) ──
    log.info("Loading events: %s", args.events)
    meta, trial_windows, ttc_anchors = load_events(args.events)
    log.info("  → %d trials with metadata, %d time windows, %d TTC anchors",
             len(meta), len(trial_windows), len(ttc_anchors))

    log.info("Loading kinematics: %s", args.kinematics)
    kin = load_kinematics(args.kinematics)
    log.info("  → %d frames total", len(kin))

    # ── Preprocess: lifecycle-anchored TTC integration ──
    df = preprocess(meta, trial_windows, ttc_anchors, kin)

    types_present = set(df["type"].dropna().unique())
    log.info("Trial types in data: %s", types_present)

    # ── Plot ──
    fig_traj = plot_trajectory_overlay(df, control_type=args.control_type)

    fig_speed = plot_speed_kinetics(
        df,
        control_type=args.control_type,
        stim_type=args.stim_type,
    )

    fig_spaghetti = plot_spaghetti_kinetics(
        df,
        control_type=args.control_type,
        stim_type=args.stim_type,
    )

    # ── Save or show ──
    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        fig_traj.savefig(out / "trajectory_overlay.png", dpi=300, bbox_inches="tight")
        fig_speed.savefig(out / "speed_kinetics.png", dpi=300, bbox_inches="tight")
        fig_spaghetti.savefig(out / "spaghetti_kinetics.png", dpi=300, bbox_inches="tight")
        log.info("Figures saved to %s", out)
    else:
        plt.show()

    return df, fig_traj, fig_speed, fig_spaghetti


if __name__ == "__main__":
    main()
