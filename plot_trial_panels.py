"""
Cercus Framework — Per-Trial Composite Panel Plot
==================================================
Standalone script: for each trial, draw a single figure containing
linear speed, angular velocity, escape trajectory, and stimulus paradigm.

Output structure:
    <save>/response/   trial_<N>_escape.png
    <save>/prewalk/    trial_<N>_prewalk.png
    <save>/no_response/ trial_<N>_noresponse.png

Usage:
    python plot_trial_panels.py --input-dir path/to/data/ --save figures/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.classifier import label_trials
from pipeline.constants import (
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
    _apply_publication_style,
    _get_unified_side,
)
from pipeline.io import load_and_concat_sessions, scan_and_pair_sessions
from pipeline.kinematics import compute_escape_latency, preprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_apply_publication_style()


# ══════════════════════════════════════════════════════════════════════
# Drawing Helpers (reused from pipeline.visualization)
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
        txt = ax.text(
            r * 0.707, r * 0.707, f"{int(r)} mm",
            color="gray", fontsize=6, ha="left", va="bottom", alpha=1, fontweight="regular",
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=0.5, foreground="#0F172A", alpha=1)])
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_side_arrows(ax: plt.Axes) -> None:
    """Draw minimalist vector arrows on LEFT and RIGHT edges."""
    arrow_style = dict(arrowstyle="]->, lengthA=0.01, widthA=10", color=None, lw=1.0, mutation_scale=8)
    ax.annotate(
        "", xy=(0.04, 0.5), xytext=(-0.02, 0.5),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops={**arrow_style, "color": COLOR_LEFT},
    )
    ax.text(0.09, 0.45, "Left Stimulus", transform=ax.transAxes,
            ha="center", va="top", fontsize=7, color=COLOR_LEFT)
    ax.annotate(
        "", xy=(0.96, 0.5), xytext=(1.02, 0.5),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops={**arrow_style, "color": COLOR_RIGHT},
    )
    ax.text(0.91, 0.45, "Right Stimulus", transform=ax.transAxes,
            ha="center", va="top", fontsize=7, color=COLOR_RIGHT)


def _add_threshold_lines(ax: plt.Axes) -> None:
    """Draw horizontal threshold lines at ESCAPE_VMAX (50) and ESCAPE_START (10)."""
    ax.axhline(y=ESCAPE_VMAX_THRESHOLD, color="k", linestyle="--", linewidth=0.75, alpha=0.7)
    ax.text(
        ax.get_xlim()[1] * 0.98, ESCAPE_VMAX_THRESHOLD + 1.0,
        f"Vmax ({ESCAPE_VMAX_THRESHOLD:.0f})",
        ha="right", va="bottom", fontsize=6, color="k", alpha=0.7,
    )
    ax.axhline(y=ESCAPE_START_THRESHOLD, color="0.5", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.text(
        ax.get_xlim()[1] * 0.98, ESCAPE_START_THRESHOLD + 1.0,
        f"Start ({ESCAPE_START_THRESHOLD:.0f})",
        ha="right", va="bottom", fontsize=6, color="0.5", alpha=0.5,
    )


# ══════════════════════════════════════════════════════════════════════
# Composite Panel Figure
# ══════════════════════════════════════════════════════════════════════

# Layout — left column shares time axis (X), trajectory on the right:
# ┌──────────────────────┬──────────┐
# │  Speed Kinetics      │          │
# ├──────────────────────┤ Trajectory
# │  AngVel Kinetics     │ (spatial)│
# ├──────────────────────┤          │
# │  Stimulus Oscilloscope          │
# └──────────────────────┴──────────┘


def plot_trial_panel(
    trial: pd.DataFrame,
    latency_ms: float,
    v_max: float,
    global_trial_index: int,
    response_type: str = "Escape",
    use_z_heading: bool = True,
) -> plt.Figure:
    """Single composite figure for one trial: speed, angular velocity, stimulus, trajectory."""
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[3, 3, 1], width_ratios=[3, 1],
                           hspace=0.25, wspace=0.25)

    ax_speed = fig.add_subplot(gs[0, 0])
    ax_angvel = fig.add_subplot(gs[1, 0], sharex=ax_speed)
    ax_stim = fig.add_subplot(gs[2, 0], sharex=ax_speed)
    ax_traj = fig.add_subplot(gs[:, 1])

    t = trial["t_rel"].values
    spd = trial["speed"].values
    ang_vel = trial["angular_velocity"].values

    # ── Color by response type ──
    if response_type == "Escape":
        curve_color = COLOR_ESCAPE
    elif response_type == "PreWalk":
        curve_color = COLOR_PREWALK
    else:
        curve_color = COLOR_NO_RESPONSE

    # ── Panel 1: Speed Kinetics ──
    ax_speed.plot(t, spd, color=curve_color, lw=1.0, alpha=0.85, label="Speed")
    if not np.isnan(latency_ms):
        ax_speed.axvline(x=latency_ms, color="#3C5488", ls="--", lw=0.9, alpha=0.9)
        lat_idx = np.argmin(np.abs(t - latency_ms))
        ax_speed.scatter([latency_ms], [spd[lat_idx]], c="#3C5488", s=30, zorder=5,
                         edgecolors="white", linewidths=0.5)
    _add_threshold_lines(ax_speed)
    ax_speed.set_ylabel("Speed (mm/s)")
    ax_speed.set_xlabel("")
    plt.setp(ax_speed.get_xticklabels(), visible=False)
    ax_speed.set_title("Linear Velocity", fontweight="bold")

    # ── Panel 2: Angular Velocity Kinetics (shares X with speed) ──
    ax_angvel.plot(t, ang_vel, color=curve_color, lw=1.0, alpha=0.85, label="Angular Vel.")
    if not np.isnan(latency_ms):
        ax_angvel.axvline(x=latency_ms, color="#3C5488", ls="--", lw=0.9, alpha=0.9)
        lat_idx = np.argmin(np.abs(t - latency_ms))
        ax_angvel.scatter([latency_ms], [ang_vel[lat_idx]], c="#3C5488", s=30, zorder=5,
                          edgecolors="white", linewidths=0.5)
    ax_angvel.set_ylabel("Angular Velocity (rad/s)")
    ax_angvel.set_xlabel("")
    plt.setp(ax_angvel.get_xticklabels(), visible=False)
    ax_angvel.set_title("Angular Velocity", fontweight="bold")

    # ── Panel 3: Stimulus Oscilloscope ──
    vis_baseline = 1.0
    wind_baseline = 3.0
    trial_type = trial["type"].iloc[0] if "type" in trial.columns else ""

    if "visual" in str(trial_type).lower() or "looming" in str(trial_type).lower():
        t_loom_start = t.min()
        t_loom = np.array([t_loom_start, 0.0])
        ax_stim.fill_between(
            t_loom, vis_baseline, vis_baseline + 1.0, step="mid",
            color=COLOR_OSCI_VIS, alpha=0.6, label="Visual (looming)",
        )

    if "stim_state" in trial.columns and trial["stim_state"].max() > 0:
        stim = trial["stim_state"].values.astype(float)
        dt_last = t[-1] - t[-2] if len(t) > 1 else 1.0
        t_ext = np.append(t, t[-1] + dt_last)
        stim_ext = np.append(stim, stim[-1])
        ax_stim.fill_between(
            t_ext, wind_baseline, wind_baseline + stim_ext, step="post",
            color=COLOR_OSCI_HW, alpha=0.6, label="Wind (stim_state)",
        )

    ax_stim.set_ylim(0, 5)
    ax_stim.set_yticks([])
    ax_stim.set_ylabel("")
    ax_stim.set_xlabel("Time relative to TTC (ms)")
    ax_stim.legend(loc="upper right", frameon=False, ncol=2)
    ax_stim.grid(False)
    ax_stim.set_title("Stimulus Paradigm", fontweight="bold")

    # ── Panel 4: Trajectory Overlay (whole trial, origin at first frame) ──
    dx_body = -np.nan_to_num(trial["dx"].values)
    dy_body = -np.nan_to_num(trial["dy"].values)
    dz_body = np.nan_to_num(trial["dz"].values)

    if use_z_heading:
        local_heading = np.cumsum(dz_body) / RADIUS_MM
        local_heading -= local_heading[0]
        dx_global = dx_body * np.cos(local_heading) - dy_body * np.sin(local_heading)
        dy_global = dx_body * np.sin(local_heading) + dy_body * np.cos(local_heading)
        traj_x = np.cumsum(dx_global)
        traj_y = np.cumsum(dy_global)
    else:
        traj_x = np.cumsum(dx_body)
        traj_y = np.cumsum(dy_body)

    # Subtract first frame so origin = start position
    traj_x -= traj_x[0]
    traj_y -= traj_y[0]

    # Color trajectory by stimulus side
    ss = _get_unified_side(trial)
    if ss == "left":
        traj_color = COLOR_LEFT
    elif ss == "right":
        traj_color = COLOR_RIGHT
    else:
        traj_color = COLOR_CONTROL

    ax_traj.plot(traj_x, traj_y, color=traj_color, lw=0.8, alpha=0.85)
    # ax_traj.scatter([traj_x[0]], [traj_y[0]], c="green", s=30, zorder=5,
    #                 edgecolors="white", linewidths=0.5, label="Start")
    # ax_traj.scatter([traj_x[-1]], [traj_y[-1]], c="red", s=30, zorder=5,
    #                 edgecolors="white", linewidths=0.5, label="End")

    _draw_standardized_grid(ax_traj, max_radius=TRAJECTORY_MAX_RADIUS_MM, step=TRAJECTORY_STEP_MM)
    _draw_side_arrows(ax_traj)
    ax_traj.legend(loc="upper right", frameon=False, fontsize=6)
    ax_traj.set_title("Escape Trajectory", fontweight="bold")

    # ── Super title ──
    latency_str = f"{latency_ms:.1f}" if not np.isnan(latency_ms) else "N/A"
    fig.suptitle(
        f"Trial {global_trial_index} — {response_type}  |  "
        f"V$_{{max}}$ = {v_max:.1f} mm/s  |  Latency = {latency_str} ms",
        fontweight="bold", fontsize=10, y=0.98,
    )

    fig.tight_layout(pad=1.0, rect=[0, 0, 1, 0.95])
    return fig


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Per-trial composite panel: speed + angular velocity + trajectory + stimulus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input-dir", required=True,
                   help="Directory containing session CSV files.")
    p.add_argument("--save", required=True,
                   help="Directory to save output figures.")
    return p


def _export_trials(
    df_slice: pd.DataFrame,
    output_dir: Path,
    response_type: str,
    label: str,
) -> None:
    """Generate composite panel figures for each trial in a response-type slice."""
    if df_slice.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    for tid, grp in df_slice.groupby("global_trial_index"):
        trial_data = grp.sort_values("t_rel")
        row = grp.iloc[0]
        lat = float(row["latency_ms"])
        vmax = float(row["v_max"])

        fig = plot_trial_panel(
            trial_data, lat, vmax, int(tid), response_type=response_type,
        )
        fig.savefig(
            output_dir / f"trial_{int(tid)}_{response_type.lower()}.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close(fig)

    n = df_slice["global_trial_index"].nunique()
    log.info("Exported %d %s trial panels to %s", n, label, output_dir)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_dir = Path(args.input_dir)
    save_dir = Path(args.save)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"--input-dir does not exist: {input_dir}")

    subjects = scan_and_pair_sessions(input_dir)
    if not subjects:
        log.error("No valid (events, kinematics) pairs found in %s", input_dir)
        return

    for subject_name, sessions in subjects.items():
        log.info("═══ Processing subject: %s ═══", subject_name)

        all_meta, all_windows, all_anchors, all_kin, _ = load_and_concat_sessions(sessions)
        df = preprocess(all_meta, all_windows, all_anchors, all_kin)
        df["global_trial_index"] = df["global_trial_id"]
        df = label_trials(df)

        subject_dir = save_dir / subject_name

        df_escape = df[df["response_type"] == "Escape"].copy()
        df_prewalk = df[df["response_type"] == "PreWalk"].copy()
        df_no_response = df[df["response_type"] == "NoResponse"].copy()

        _export_trials(df_escape, subject_dir / "response", "Escape", "Escape")
        _export_trials(df_prewalk, subject_dir / "prewalk", "PreWalk", "PreWalk")
        _export_trials(df_no_response, subject_dir / "no_response", "NoResponse", "NoResponse")

    log.info("All subjects processed.")


if __name__ == "__main__":
    main()
