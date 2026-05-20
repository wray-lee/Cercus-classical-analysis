"""Stateless matplotlib figure generator for Cercus Analysis.

Receives data dicts, returns figure objects. No file I/O.
All functions return Figure objects; callers handle display or saving.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.models import EscapeBout, FreezingEpisode, PSTHResult, SessionResults

# Dark-theme-aware color constants
_COLOR_PRIMARY = "#4A9EFF"  # Bright blue (visible on dark bg)
_COLOR_SECONDARY = "#FF6B6B"  # Soft red
_COLOR_ACCENT = "#4ADE80"  # Green
_COLOR_FILL = "#4A9EFF"  # Match primary for SEM fill
_COLOR_STIMULUS = "#FBBF24"  # Yellow stimulus marker
_COLOR_FREEZE = "#4ADE80"  # Green freezing overlay
_COLOR_DX = "#4A9EFF"      # Lateral axis (blue)
_COLOR_DY = "#FF6B6B"      # Forward/back axis (red)
_COLOR_DZ = "#4ADE80"      # Rotational axis (green)
_ALPHA_FILL = 0.25
_LINEWIDTH_TIMESERIES = 0.7
_LINEWIDTH_PSTH = 1.6
_GRID_ALPHA = 0.3


def create_trajectory_heatmap(
    traj_x: np.ndarray,
    traj_y: np.ndarray,
    session_id: str,
    bins: int = 60,
) -> Figure:
    """Panel A: 2D Spatial Trajectory + Probability Density Heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"Session: {session_id}", fontsize=13, fontweight="bold")

    # --- Trajectory plot ---
    ax_traj: Axes = axes[0]
    # Color trajectory by time progression
    points = ax_traj.scatter(
        traj_x,
        traj_y,
        c=np.arange(len(traj_x)),
        cmap="viridis",
        s=2,
        alpha=0.6,
        edgecolors="none",
    )
    fig.colorbar(points, ax=ax_traj, label="Sample index", shrink=0.8)
    ax_traj.plot(
        traj_x[0],
        traj_y[0],
        "o",
        color=_COLOR_ACCENT,
        markersize=10,
        zorder=5,
        label="Start",
    )
    ax_traj.plot(
        traj_x[-1],
        traj_y[-1],
        "s",
        color=_COLOR_SECONDARY,
        markersize=10,
        zorder=5,
        label="End",
    )
    ax_traj.set_xlabel("X displacement (mm)")
    ax_traj.set_ylabel("Y displacement (mm)")
    ax_traj.set_title("2D Trajectory (color = time)")
    ax_traj.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.invert_yaxis()
    ax_traj.grid(True, alpha=_GRID_ALPHA)

    # --- Heatmap ---
    ax_heat: Axes = axes[1]
    h = ax_heat.hist2d(
        traj_x,
        traj_y,
        bins=bins,
        cmap="inferno",
        density=True,
    )
    fig.colorbar(h[3], ax=ax_heat, label="Probability Density", shrink=0.8)
    ax_heat.set_xlabel("X displacement (mm)")
    ax_heat.set_ylabel("Y displacement (mm)")
    ax_heat.set_title("Spatial Probability Density")
    ax_heat.set_aspect("equal", adjustable="datalim")
    ax_heat.invert_yaxis()

    fig.tight_layout()
    return fig


def create_speed_timeseries(
    time: np.ndarray,
    speed: np.ndarray,
    session_id: str,
    freezing_episodes: Optional[List[FreezingEpisode]] = None,
    escape_bouts: Optional[List[EscapeBout]] = None,
) -> Figure:
    """Panel B: Global Speed vs. Time timeseries with ethological overlays."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(
        time, speed, linewidth=_LINEWIDTH_TIMESERIES, color=_COLOR_PRIMARY, alpha=0.9
    )

    # Overlay freezing episodes
    if freezing_episodes:
        for i, ep in enumerate(freezing_episodes):
            ax.axvspan(
                ep.start_time,
                ep.end_time,
                alpha=0.25,
                color=_COLOR_FREEZE,
                label="Freezing" if i == 0 else None,
            )

    # Overlay escape bout peaks
    if escape_bouts:
        escape_times = [b.peak_time for b in escape_bouts]
        escape_vels = [b.peak_velocity for b in escape_bouts]
        ax.scatter(
            escape_times,
            escape_vels,
            marker="v",
            s=50,
            c=_COLOR_SECONDARY,
            zorder=5,
            edgecolors="white",
            linewidths=0.5,
            label="Escape peaks",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (mm/s)")
    ax.set_title(f"Speed Timeseries — {session_id}", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=_GRID_ALPHA)
    fig.tight_layout()
    return fig


def create_psth_plot(
    psth_results: Dict[str, PSTHResult],
    session_id: str,
    show_angular_velocity: bool = True,
) -> Figure:
    """Panel C: Stimulus-Aligned Locomotor Response (Mean Speed ± SEM).

    If show_angular_velocity is True, adds a secondary axis for angular velocity.
    """
    n_events = len(psth_results)
    if n_events == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(
            0.5,
            0.5,
            "No stimulus events found",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color="gray",
        )
        ax.set_axis_off()
        return fig

    fig, axes = plt.subplots(
        n_events,
        1,
        figsize=(10, 4.5 * n_events),
        squeeze=False,
    )
    fig.suptitle(f"PSTH — {session_id}", fontsize=13, fontweight="bold")

    for i, (evt_type, psth) in enumerate(psth_results.items()):
        ax: Axes = axes[i, 0]

        # Speed trace
        ax.plot(
            psth.time_axis,
            psth.mean_speed,
            linewidth=_LINEWIDTH_PSTH,
            color=_COLOR_PRIMARY,
            label="Mean Speed",
        )
        ax.fill_between(
            psth.time_axis,
            psth.mean_speed - psth.sem_speed,
            psth.mean_speed + psth.sem_speed,
            alpha=_ALPHA_FILL,
            color=_COLOR_FILL,
            label="±SEM",
        )

        ax.axvline(
            0,
            color=_COLOR_STIMULUS,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="Stimulus Onset",
        )
        ax.set_ylabel("Speed (mm/s)", color=_COLOR_PRIMARY)
        ax.tick_params(axis="y", labelcolor=_COLOR_PRIMARY)
        ax.set_title(f"{evt_type} (n={psth.n_trials} trials)")

        # Angular velocity on secondary axis
        if show_angular_velocity:
            ax2 = ax.twinx()
            ax2.plot(
                psth.time_axis,
                psth.mean_angular_velocity,
                linewidth=1.0,
                color=_COLOR_SECONDARY,
                alpha=0.7,
                linestyle="-.",
                label="Ang. Vel.",
            )
            ax2.fill_between(
                psth.time_axis,
                psth.mean_angular_velocity - psth.sem_angular_velocity,
                psth.mean_angular_velocity + psth.sem_angular_velocity,
                alpha=0.15,
                color=_COLOR_SECONDARY,
            )
            ax2.set_ylabel("Angular Velocity", color=_COLOR_SECONDARY)
            ax2.tick_params(axis="y", labelcolor=_COLOR_SECONDARY)

        if i == n_events - 1:
            ax.set_xlabel("Time relative to stimulus (s)")

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        if show_angular_velocity:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc="upper right",
                fontsize=7,
                framealpha=0.8,
            )
        else:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

        ax.grid(True, alpha=_GRID_ALPHA)

    fig.tight_layout()
    return fig


def create_angular_velocity_histogram(
    angular_velocity: np.ndarray,
    session_id: str,
    bins: int = 120,
) -> Figure:
    """Panel D: Angular Velocity histogram with statistics overlay."""
    fig, ax = plt.subplots(figsize=(9, 4.5))

    counts, bin_edges, patches = ax.hist(
        angular_velocity,
        bins=bins,
        color=_COLOR_PRIMARY,
        alpha=0.7,
        edgecolor="#1E1E1E",
        linewidth=0.2,
        density=True,
    )

    # Mark statistics
    mean_val = float(np.mean(angular_velocity))
    std_val = float(np.std(angular_velocity))
    median_val = float(np.median(angular_velocity))

    ax.axvline(
        mean_val,
        color=_COLOR_SECONDARY,
        linestyle="-",
        linewidth=1.5,
        label=f"Mean: {mean_val:.2f}",
    )
    ax.axvline(
        median_val,
        color=_COLOR_ACCENT,
        linestyle="--",
        linewidth=1.5,
        label=f"Median: {median_val:.2f}",
    )
    ax.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    # Text box with stats
    stats_text = f"σ = {std_val:.2f}\nn = {len(angular_velocity):,}"
    ax.text(
        0.98,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        color="#D4D4D4",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#2D2D30",
            edgecolor="#4A4A4E",
            alpha=0.9,
        ),
    )

    ax.set_xlabel("Angular Velocity (units/s)")
    ax.set_ylabel("Density")
    ax.set_title(f"Angular Velocity Distribution — {session_id}", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=_GRID_ALPHA)
    fig.tight_layout()
    return fig


def create_decoupled_velocity_panel(
    time: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    session_id: str = "",
) -> Figure:
    """Calibration Panel: Decoupled velocity outputs with unified Y-axis.

    Uses sharey=True so all three subplots share an identical Y-scale.
    This visually compresses cross-talk noise to a flat baseline relative
    to the primary movement signal, preventing matplotlib auto-scaling
    from amplifying ~% cross-talk as false coupling.

    The Y-axis is inverted to match spherical treadmill convention
    (positive displacement = forward/right visual movement).
    """
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 8), sharex=True, sharey=True,
    )
    fig.suptitle(
        f"Decoupled Axis Outputs — {session_id}" if session_id
        else "Decoupled Axis Outputs",
        fontsize=13, fontweight="bold",
    )

    # Compute global symmetric limits from all three axes
    global_max = max(
        float(np.max(np.abs(dx))) if len(dx) > 0 else 1.0,
        float(np.max(np.abs(dy))) if len(dy) > 0 else 1.0,
        float(np.max(np.abs(dz))) if len(dz) > 0 else 1.0,
    )
    # Add 5% margin so primary peaks don't clip against the frame
    ylim_hi = global_max * 1.05
    ylim_lo = -ylim_hi

    traces = [
        (dx, _COLOR_DX, "Lateral (X-axis)", "dx (displacement)"),
        (dy, _COLOR_DY, "Forward/Back (Y-axis)", "dy (displacement)"),
        (dz, _COLOR_DZ, "Rotational (Z-axis / Yaw)", "dz (displacement)"),
    ]

    for ax, (data, color, title, ylabel) in zip(axes, traces):
        ax.plot(time, data, linewidth=0.5, color=color, alpha=0.9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_ylim(ylim_lo, ylim_hi)
        ax.invert_yaxis()
        ax.grid(True, alpha=_GRID_ALPHA)

    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def create_steering_sine_panel(
    time: np.ndarray,
    heading_angle: np.ndarray,
    steering_sine: np.ndarray,
    session_id: str = "",
) -> Figure:
    """Steering Angle Panel: Sine of integrated heading angle over time.

    The heading angle θ is the cumulative sum of Z-axis rotational
    displacement. sin(θ) produces a bounded [-1, 1] waveform that
    directly encodes the animal's instantaneous steering direction
    for PI (path integration) analysis.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(
        f"Steering Angle — {session_id}" if session_id
        else "Steering Angle",
        fontsize=13, fontweight="bold",
    )

    # Top panel: raw heading angle (cumulative yaw)
    ax_theta: Axes = axes[0]
    ax_theta.plot(time, heading_angle, linewidth=0.7, color=_COLOR_DZ, alpha=0.9)
    ax_theta.set_ylabel("Heading θ (rad)")
    ax_theta.set_title("Integrated Heading Angle  (θ = Σ dz)")
    ax_theta.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_theta.grid(True, alpha=_GRID_ALPHA)

    # Bottom panel: sin(θ) — bounded steering signal
    ax_sin: Axes = axes[1]
    ax_sin.plot(time, steering_sine, linewidth=0.7, color=_COLOR_PRIMARY, alpha=0.9)
    ax_sin.set_ylabel("sin(θ)")
    ax_sin.set_xlabel("Time (s)")
    ax_sin.set_title("Steering Sine Wave  (sin(θ))")
    ax_sin.set_ylim(-1.1, 1.1)
    ax_sin.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_sin.axhline(1, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
    ax_sin.axhline(-1, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
    ax_sin.grid(True, alpha=_GRID_ALPHA)

    fig.tight_layout()
    return fig


def generate_all_panels(
    result: SessionResults,
) -> Dict[str, Figure]:
    """Generate all four visualization panels for a single session."""
    # Apply dark theme before generating figures
    try:
        from src.gui.theme import apply_mpl_theme

        apply_mpl_theme()
    except ImportError:
        pass  # Running without GUI (e.g., headless export)

    kd = result.kinematic_data
    panels: Dict[str, Figure] = {
        "trajectory_heatmap": create_trajectory_heatmap(
            kd.trajectory_x, kd.trajectory_y, result.session_id
        ),
        "speed_timeseries": create_speed_timeseries(
            kd.time,
            kd.speed,
            result.session_id,
            freezing_episodes=result.freezing_episodes,
            escape_bouts=result.escape_bouts,
        ),
        "psth": create_psth_plot(
            result.psth_results,
            result.session_id,
            show_angular_velocity=True,
        ),
        "angular_velocity_histogram": create_angular_velocity_histogram(
            kd.angular_velocity, result.session_id
        ),
    }

    # Decoupled velocity panel (requires raw decoupled axes)
    if kd.decoupled_dx is not None:
        panels["decoupled_velocity"] = create_decoupled_velocity_panel(
            kd.time,
            kd.decoupled_dx,
            kd.decoupled_dy,
            kd.decoupled_dz,
            result.session_id,
        )

    # Steering angle sine wave panel (requires integrated heading)
    if kd.heading_angle is not None and kd.steering_sine is not None:
        panels["steering_sine"] = create_steering_sine_panel(
            kd.time,
            kd.heading_angle,
            kd.steering_sine,
            result.session_id,
        )

    return panels


def export_figures(
    figures: Dict[str, Figure],
    output_dir: Path,
    session_id: str,
    formats: Tuple[str, ...] = ("png", "pdf"),
) -> List[Path]:
    """Export figures to disk. Returns list of saved file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for name, fig in figures.items():
        for fmt in formats:
            filepath = output_dir / f"{session_id}_{name}.{fmt}"
            fig.savefig(filepath, dpi=150, bbox_inches="tight", format=fmt)
            saved.append(filepath)
        plt.close(fig)
    return saved
