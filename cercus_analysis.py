"""
Cercus Framework — Behavioral Neuroscience Data Analysis & Visualization
=========================================================================
Dual-track recording system: Events (low-freq) + Kinematics (high-freq)
aligned by trial timestamps (trial_start / trial_stop) with legacy fallback.
Time axis zeroed to visual TTC (t_col).

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

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
DETAILS_KEYS = ("type", "target_ttc_ms", "wind_dir", "screen_side",
                "lv_ratio_ms", "init_half_angle_deg")
SPEED_WINDOW_MS = 100          # rolling-mean smoothing window (ms)
DEFAULT_LV_RATIO_MS = 100.0    # looming expansion ratio (ms)
DEFAULT_INIT_ANGLE_DEG = 2.0   # initial looming half-angle (degrees)
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


def load_events(path: str | Path) -> tuple[pd.DataFrame, list[dict]]:
    """
    Load events CSV. Extract trial time windows and metadata.

    Returns
    -------
    trial_meta : DataFrame
        One row per trial with global_trial_id + DETAILS_KEYS.
    trial_windows : list of dict
        Each dict has keys: global_trial_id, t_start, t_stop.
        t_stop falls back to t_start + LEGACY_TRIAL_DURATION_MS if trial_stop missing.
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
        return empty_meta, []

    stops = df[df["event_name"] == "trial_stop"].copy()

    # Build lookup: global_trial_id → t_stop timestamp
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

    # Extract metadata from details JSON
    parsed = starts["details"].apply(_parse_details)
    details_df = pd.DataFrame(parsed.tolist(), index=starts.index)
    for k in DETAILS_KEYS:
        if k not in details_df.columns:
            details_df[k] = np.nan

    trial_meta = starts[["global_trial_id"]].join(details_df[list(DETAILS_KEYS)])
    trial_meta = trial_meta.drop_duplicates(subset="global_trial_id", keep="first")
    return trial_meta.reset_index(drop=True), trial_windows


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

def compute_t_col(lv_ratio_ms: float, init_half_angle_deg: float) -> float:
    """
    Compute theoretical time from trial start to visual collision (ms).
    t_col = (lv_ratio_ms / 1000) / tan(radians(init_half_angle_deg)) * 1000
    """
    return (lv_ratio_ms / 1000.0) / np.tan(np.radians(init_half_angle_deg)) * 1000.0


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


def _integrate_trial(grp: pd.DataFrame, t_col_ms: float) -> pd.DataFrame:
    """
    Per-trial integration with TTC-aligned time axis and denoised speed.

    Time axis zero-point is the visual TTC: t_rel = (sys_time - sys_time[0]) - t_col.
    This means t_rel < 0 is before collision, t_rel = 0 is collision moment.

    Speed denoising:
      1. cumsum dx/dy → absolute position, origin at (0,0)
      2. v = sqrt(dx² + dy²) / dt  (real frame-to-frame dt)
      3. First 2 frames forced to NaN (residual displacement from prior phase)
      4. Rolling 100ms center-smoothed mean
    """
    df = grp.copy()

    if len(df) < 2:
        df["t_rel"] = -t_col_ms
        df["x"] = 0.0
        df["y"] = 0.0
        df["speed"] = np.nan
        return df

    # Time: zero-point aligned to visual TTC
    df["t_rel"] = (df["sys_time"] - df["sys_time"].iloc[0]) * 1000.0 - t_col_ms

    # Position: cumulative sum of displacements, origin at (0,0)
    # 强制清空前两帧的残余位移 (防止 ITI 期间的传感器漂移引发瞬移)
    df.loc[df.index[:2], "dx"] = 0.0
    df.loc[df.index[:2], "dy"] = 0.0

    # 强制翻转硬件 X 轴读数以对齐真实左右空间
    df["x"] = (-df["dx"]).cumsum()
    df["y"] = df["dy"].cumsum()

    # Real frame-to-frame dt (ms)
    dt = df["sys_time"].diff()
    dt = dt.replace(0, np.nan)

    # Euclidean displacement per frame
    dist = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    raw_speed = dist / dt  # mm/s

    # First 2 frames: force NaN to kill residual displacement artifacts
    raw_speed.iloc[:2] = np.nan

    # Rolling mean smoothing — adaptive window → ~100 ms
    median_dt = dt.median()
    if pd.notna(median_dt) and median_dt > 0:
        win = max(1, int(round((SPEED_WINDOW_MS / 1000.0) / median_dt)))
    else:
        win = 5
    df["speed"] = raw_speed.rolling(window=win, min_periods=1, center=True).mean()

    return df


def preprocess(
    trials_meta: pd.DataFrame,
    trial_windows: list[dict],
    kin: pd.DataFrame,
    t_col_ms: float,
) -> pd.DataFrame:
    """
    Slice kinematics by trial time windows, merge metadata, and integrate.

    For each trial:
      1. Boolean-mask kinematics using [t_start, t_stop] from events.
      2. Attach trial metadata (type, screen_side, etc.).
      3. Compute t_rel, position, and denoised speed.
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

        try:
            parts.append(_integrate_trial(trial_kin, t_col_ms))
        except Exception as exc:
            log.warning("Skipping trial %s during integration: %s", tid, exc)

    if not parts:
        raise RuntimeError("No valid trials after preprocessing.")
    return pd.concat(parts, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════
# Plot 1 — Trajectory Overlay
# ══════════════════════════════════════════════════════════════════════

def _draw_cross_axes(ax: plt.Axes, scale_bar_val: float = SCALE_BAR_MM):
    """Draw cross-shaped origin axes with arrowheads and a scale bar."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Horizontal axis (y=0)
    ax.annotate("", xy=(xlim[1], 0), xytext=(xlim[0], 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
    # Vertical axis (x=0)
    ax.annotate("", xy=(0, ylim[1]), xytext=(0, ylim[0]),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))

    # Scale bar at bottom-right
    sb_x = xlim[1] * 0.65
    sb_y = ylim[0] * 0.85
    ax.plot([sb_x, sb_x + scale_bar_val], [sb_y, sb_y], "k-", lw=2)
    ax.text(sb_x + scale_bar_val / 2, sb_y - (ylim[1] - ylim[0]) * 0.04,
            f"{scale_bar_val:.0f} mm", ha="center", va="top", fontsize=8)


def _draw_side_arrows(ax: plt.Axes, left_color: str = "blue", right_color: str = "red"):
    """
    Draw thick short arrows on LEFT and RIGHT edges with text labels below.
    Uses axes-fraction coords. Arrows point inward (airflow toward center).
    """
    arrow_style = dict(
        facecolor=None,  # set per arrow
        edgecolor="none",
        width=6, headwidth=14, headlength=12, shrink=0,
    )

    # Blue arrow on left edge → points right (airflow from left toward center)
    ax.annotate(
        "", xy=(0.04, 0.5), xytext=(-0.02, 0.5),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops={**arrow_style, "facecolor": left_color},
    )
    ax.text(0.01, 0.45, "Left Stimulus",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color=left_color)

    # Red arrow on right edge → points left (airflow from right toward center)
    ax.annotate(
        "", xy=(0.96, 0.5), xytext=(1.02, 0.5),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops={**arrow_style, "facecolor": right_color},
    )
    ax.text(0.99, 0.45, "Right Stimulus",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color=right_color)


def plot_trajectory_overlay(
    df: pd.DataFrame,
    control_type: str = "baseline_visual",
    left_color: str = "blue",
    right_color: str = "red",
    figsize_per_ax: tuple[float, float] = (4.0, 4.0),
):
    """
    One subplot per trial type. Left stimuli in blue, right in red, control in gray.
    Cross-shaped axes with thick short side-arrow indicators and text labels.
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
                color = "gray"
            elif ss == "left":
                color = left_color
            elif ss == "right":
                color = right_color
            else:
                color = "gray"
            ax.plot(grp["x"], grp["y"], color=color, alpha=0.5, lw=0.8)

        ax.set_aspect("equal")
        ax.set_title(ttype, fontsize=10, fontweight="bold")
        _draw_cross_axes(ax)
        _draw_side_arrows(ax, left_color=left_color, right_color=right_color)

    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# Plot 2 — Speed Kinetics & Oscilloscope Dual-Channel Waveforms
# ══════════════════════════════════════════════════════════════════════

def plot_speed_kinetics(
    df: pd.DataFrame,
    t_col_ms: float,
    control_type: str = "baseline_visual",
    stim_type: str = "looming_wind",
    figsize: tuple[float, float] = (10, 6),
):
    """
    Two-panel figure (8:2 height ratio) with shared X axis.

    Upper panel (80%): Escape speed (mean ± SEM) per condition.
    Lower panel (20%): Oscilloscope-style dual-channel waveforms:
        • Channel 1 — Visual looming (light blue): high pulse in [-t_col, 0]
        • Channel 2 — Wind stim_state (orange-red): real hardware timing via fill_between
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.08)
    ax_main = fig.add_subplot(gs[0])
    ax_stim = fig.add_subplot(gs[1], sharex=ax_main)

    # ── Upper panel: speed mean ± SEM ──
    t_bin = 5.0  # ms
    t_min = df["t_rel"].min()
    t_max = df["t_rel"].max()
    bins = np.arange(t_min, t_max + t_bin, t_bin)
    df_binned = df.copy()
    df_binned["t_bin"] = pd.cut(df_binned["t_rel"], bins=bins,
                                labels=bins[:-1], include_lowest=True)
    df_binned["t_bin"] = df_binned["t_bin"].astype(float)

    cond_colors = {control_type: "gray", stim_type: "blue"}
    for cond, color in cond_colors.items():
        subset = df_binned[df_binned["type"] == cond]
        if subset.empty:
            continue
        grp = subset.groupby("t_bin")["speed"]
        mean = grp.mean()
        sem = grp.sem()
        t_vals = mean.index.values
        ax_main.plot(t_vals, mean.values, color=color, lw=1.5, label=cond)
        ax_main.fill_between(t_vals, (mean - sem).values, (mean + sem).values,
                             color=color, alpha=0.2)

    ax_main.set_ylabel("Escape Speed (mm/s)")
    ax_main.legend(loc="upper right", fontsize=8)
    ax_main.set_xlabel("")
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # ── Lower panel: oscilloscope waveforms ──
    loom_color = "#A0C8E8"   # light blue
    wind_color = "#F4A460"   # sandy orange
    vis_baseline = 1.0
    wind_baseline = 3.0

    # Channel 1: Visual looming — high pulse in [-t_col, 0]
    t_loom = np.array([-t_col_ms, 0.0])
    ax_stim.fill_between(t_loom, vis_baseline, vis_baseline + 1.0,
                         step="mid", color=loom_color, alpha=0.8,
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
                             step="post", color=wind_color, alpha=0.8,
                             label="Wind (stim_state)")

    ax_stim.set_ylim(0, 5)
    ax_stim.set_yticks([])
    ax_stim.set_ylabel("")
    ax_stim.set_xlabel("Time relative to visual TTC (ms)")
    ax_stim.legend(loc="upper right", fontsize=7, ncol=2)

    # Remove lower panel background grid
    ax_stim.grid(False)
    ax_stim.set_facecolor("white")

    fig.tight_layout()
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
    p.add_argument("--lv-ratio-ms", type=float, default=DEFAULT_LV_RATIO_MS,
                   help="Looming expansion ratio in ms (default: 100)")
    p.add_argument("--init-angle", type=float, default=DEFAULT_INIT_ANGLE_DEG,
                   help="Initial looming half-angle in degrees (default: 2.0)")
    p.add_argument("--save", default=None,
                   help="Directory to save PNG figures. Omit to show interactively.")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

    # ── Load events (extract trial windows + metadata) ──
    log.info("Loading events: %s", args.events)
    meta, trial_windows = load_events(args.events)
    log.info("  → %d trials with metadata, %d time windows extracted",
             len(meta), len(trial_windows))

    log.info("Loading kinematics: %s", args.kinematics)
    kin = load_kinematics(args.kinematics)
    log.info("  → %d frames total", len(kin))

    # ── Compute TTC ──
    t_col_ms = compute_t_col(args.lv_ratio_ms, args.init_angle)
    log.info("Visual collision time: t_col = %.1f ms", t_col_ms)

    # ── Preprocess: slice by timestamps, integrate per-trial ──
    df = preprocess(meta, trial_windows, kin, t_col_ms)

    types_present = set(df["type"].dropna().unique())
    log.info("Trial types in data: %s", types_present)

    # ── Plot ──
    fig_traj = plot_trajectory_overlay(df, control_type=args.control_type)

    fig_speed = plot_speed_kinetics(
        df,
        t_col_ms=t_col_ms,
        control_type=args.control_type,
        stim_type=args.stim_type,
    )

    # ── Save or show ──
    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        fig_traj.savefig(out / "trajectory_overlay.png", dpi=300, bbox_inches="tight")
        fig_speed.savefig(out / "speed_kinetics.png", dpi=300, bbox_inches="tight")
        log.info("Figures saved to %s", out)
    else:
        plt.show()

    return df, fig_traj, fig_speed


if __name__ == "__main__":
    main()
