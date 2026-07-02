"""
Cercus Framework — Per-Trial Composite Panel Plot
==================================================
Standalone script: for each trial, draw a single figure containing
linear speed, angular velocity, escape trajectory, and stimulus paradigm.

Output structure:
    <save>/response/   trial_<N>_escape.svg
    <save>/prewalk/    trial_<N>_prewalk.svg
    <save>/no_response/ trial_<N>_noresponse.svg

Usage:
    python plot_trial_panels.py --input-dir path/to/data/ --save figures/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.gridspec as gridspec
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
    RADIUS_MM,
    TRAJECTORY_MAX_RADIUS_MM,
    TRAJECTORY_STEP_MM,
    TRAJ_USE_ESCAPE_ONSET_ONLY_XY,
    TRAJ_USE_ESCAPE_ONSET_ONLY_Z,
    TRAJ_USE_RIGID_ROTATION,
    TRAJ_USE_Z_DEGREE,
    _apply_publication_style,
    _get_unified_side,
)
from pipeline.io import load_and_concat_sessions, scan_and_pair_sessions
from pipeline.kinematics import preprocess
from pipeline.visualization import (
    _add_threshold_lines,
    _draw_side_arrows,
    _draw_standardized_grid,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_apply_publication_style()


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
    use_z_heading: bool = TRAJ_USE_Z_DEGREE,
    use_rigid_rotation: bool = TRAJ_USE_RIGID_ROTATION,
    use_escape_onset_only_xy: bool = TRAJ_USE_ESCAPE_ONSET_ONLY_XY,
    use_escape_onset_only_z: bool = TRAJ_USE_ESCAPE_ONSET_ONLY_Z,
    interval_ms: float = np.nan,
) -> plt.Figure:
    """Single composite figure for one trial: speed, angular velocity, stimulus, trajectory."""
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(3, 2, height_ratios=[3, 3, 1], width_ratios=[2, 1.5],
                           hspace=0.25, wspace=0.3)

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
    if not np.isnan(interval_ms) and not np.isnan(latency_ms):
        offset_t = latency_ms + interval_ms
        ax_speed.axvspan(latency_ms, offset_t, alpha=0.10, color="#E64B35", zorder=0)
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

    # ── Panel 4: Trajectory Overlay ──
    t_vals = trial["t_rel"].values
    speed_vals = trial["speed"].values

    # ── Build escape-onset mask (shared logic) ──
    _is_escape = (response_type == "Escape" and not np.isnan(latency_ms))
    if _is_escape:
        lat_idx = int(np.argmin(np.abs(t_vals - latency_ms)))
        post_onset_speed = speed_vals[lat_idx:]
        below_mask = post_onset_speed < ESCAPE_START_THRESHOLD
        if np.any(below_mask):
            end_idx = lat_idx + int(np.argmax(below_mask))
        else:
            end_idx = len(speed_vals) - 1
        if lat_idx >= end_idx:
            end_idx = min(lat_idx + 1, len(speed_vals) - 1)
        escape_start, escape_end = lat_idx, end_idx
    else:
        escape_start, escape_end = 0, len(speed_vals)

    if use_rigid_rotation:
        # ── Rigid global rotation mode: use dx/dy/dz body-frame integration ──
        dx_body = -np.nan_to_num(trial["dx"].values)
        dy_body = -np.nan_to_num(trial["dy"].values)
        dz_body = np.nan_to_num(trial["dz"].values)

        # Independent masks for xy displacement and z heading
        if use_escape_onset_only_xy and _is_escape:
            xy_start, xy_end = escape_start, escape_end
        else:
            xy_start, xy_end = 0, len(dx_body)

        if use_escape_onset_only_z and _is_escape:
            z_start, z_end = escape_start, escape_end
        else:
            z_start, z_end = 0, len(dz_body)

        burst_dx = dx_body[xy_start:xy_end]
        burst_dy = dy_body[xy_start:xy_end]
        burst_dz = dz_body[z_start:z_end]

        if len(burst_dx) > 0:
            # Stage 1: inner trajectory with dynamic heading
            local_heading = np.cumsum(burst_dz) / RADIUS_MM
            local_heading -= local_heading[0]

            # Align heading length to displacement length via interpolation
            if len(local_heading) != len(burst_dx):
                src_idx = np.arange(len(burst_dz))
                dst_idx = np.linspace(0, len(burst_dz) - 1, len(burst_dx))
                if len(src_idx) >= 2:
                    local_heading = np.interp(dst_idx, src_idx, local_heading)
                else:
                    local_heading = np.full(len(burst_dx), local_heading[0])

            dx_inner = burst_dx * np.cos(local_heading) - burst_dy * np.sin(local_heading)
            dy_inner = burst_dx * np.sin(local_heading) + burst_dy * np.cos(local_heading)
            x_inner = np.cumsum(dx_inner)
            y_inner = np.cumsum(dy_inner)

            # Stage 2: rigid macro rotation
            total_yaw_rad = np.sum(burst_dz) / RADIUS_MM
            cos_yaw = np.cos(total_yaw_rad)
            sin_yaw = np.sin(total_yaw_rad)
            traj_x = x_inner * cos_yaw - y_inner * sin_yaw
            traj_y = x_inner * sin_yaw + y_inner * cos_yaw
        else:
            traj_x = np.array([])
            traj_y = np.array([])
    else:
        # ── Default: global x/y coordinates with optional initial heading rotation ──
        full_x_raw = trial["x"].values
        full_y_raw = trial["y"].values

        if use_z_heading:
            theta_init = (trial["dz"].cumsum().values / RADIUS_MM)[0]
            full_x = full_x_raw * np.cos(theta_init) + full_y_raw * np.sin(theta_init)
            full_y = -full_x_raw * np.sin(theta_init) + full_y_raw * np.cos(theta_init)
        else:
            full_x = full_x_raw
            full_y = full_y_raw

        if use_escape_onset_only_xy and _is_escape:
            traj_x = full_x[escape_start:escape_end]
            traj_y = full_y[escape_start:escape_end]
        else:
            traj_x = full_x
            traj_y = full_y

    # ── Origin alignment: start from (0, 0) ──
    if len(traj_x) > 0:
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
    interval_str = f"{interval_ms:.1f}" if not np.isnan(interval_ms) else "N/A"
    fig.suptitle(
        f"Trial {global_trial_index} — {response_type}  |  "
        f"V$_{{max}}$ = {v_max:.1f} mm/s  |  Latency = {latency_str} ms  |  Interval = {interval_str} ms",
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
        interval = float(row.get("escape_interval_ms", np.nan))

        fig = plot_trial_panel(
            trial_data, lat, vmax, int(tid), response_type=response_type,
            interval_ms=interval,
        )
        fig.savefig(
            output_dir / f"trial_{int(tid)}_{response_type.lower()}.svg",
            bbox_inches="tight",
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
