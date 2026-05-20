"""Calibration Visualizer: Hardware matrix validation for spherical treadmill.

Independent from the main ethological analysis pipeline. Reads raw
decoupled kinematics CSVs to validate orthogonal decoupling, yaw drift,
and stationary jitter.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Dark-theme-aware color constants
_COLOR_DX = "#4A9EFF"      # Bright blue
_COLOR_DY = "#FF6B6B"      # Soft red
_COLOR_DZ = "#4ADE80"      # Green
_COLOR_FILL = "#4A9EFF"
_GRID_ALPHA = 0.3


def create_crosstalk_panel(
    time: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    session_id: str = "",
) -> Figure:
    """Cross-talk Panel: 3-row subplot of dx, dy, dz vs time.

    Uses sharey=True to force an identical Y-axis on all three subplots.
    This visually compresses cross-talk noise to a flat baseline relative
    to the primary movement, preventing matplotlib auto-scaling from
    amplifying ~1% residual coupling as false severity.

    Y-axis is inverted to match spherical treadmill convention.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, sharey=True)
    fig.suptitle(
        f"Cross-talk Inspection — {session_id}" if session_id else "Cross-talk Inspection",
        fontsize=13,
        fontweight="bold",
    )

    # Compute global symmetric limits from all three axes
    global_max = max(
        float(np.max(np.abs(dx))) if len(dx) > 0 else 1.0,
        float(np.max(np.abs(dy))) if len(dy) > 0 else 1.0,
        float(np.max(np.abs(dz))) if len(dz) > 0 else 1.0,
    )
    ylim_hi = global_max * 1.05
    ylim_lo = -ylim_hi

    # dx trace
    ax_dx: Axes = axes[0]
    ax_dx.plot(time, dx, linewidth=0.5, color=_COLOR_DX, alpha=0.9)
    ax_dx.set_ylabel("dx (displacement)")
    ax_dx.set_title("Lateral (X-axis)")
    ax_dx.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_dx.set_ylim(ylim_lo, ylim_hi)
    ax_dx.invert_yaxis()
    ax_dx.grid(True, alpha=_GRID_ALPHA)

    # dy trace
    ax_dy: Axes = axes[1]
    ax_dy.plot(time, dy, linewidth=0.5, color=_COLOR_DY, alpha=0.9)
    ax_dy.set_ylabel("dy (displacement)")
    ax_dy.set_title("Forward/Back (Y-axis)")
    ax_dy.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_dy.grid(True, alpha=_GRID_ALPHA)

    # dz trace
    ax_dz: Axes = axes[2]
    ax_dz.plot(time, dz, linewidth=0.5, color=_COLOR_DZ, alpha=0.9)
    ax_dz.set_ylabel("dz (displacement)")
    ax_dz.set_xlabel("Time (s)")
    ax_dz.set_title("Rotational (Z-axis / Yaw)")
    ax_dz.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_dz.grid(True, alpha=_GRID_ALPHA)

    fig.tight_layout()
    return fig


def create_steering_sine_panel(
    time: np.ndarray,
    dz: np.ndarray,
    session_id: str = "",
) -> Figure:
    """Steering Angle Panel: Sine of integrated heading angle.

    Integrates the Z-axis rotational displacement to obtain the heading
    angle θ = cumsum(dz), then plots sin(θ). The sine wave produces a
    bounded [-1, 1] signal encoding instantaneous steering direction,
    as required for path integration (PI) analysis.
    """
    heading_angle = np.cumsum(dz)
    steering_sine = np.sin(heading_angle)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(
        f"Steering Angle — {session_id}" if session_id else "Steering Angle",
        fontsize=13,
        fontweight="bold",
    )

    # Top panel: cumulative heading angle
    ax_theta: Axes = axes[0]
    ax_theta.plot(time, heading_angle, linewidth=0.7, color=_COLOR_DZ, alpha=0.9)
    ax_theta.set_ylabel("Heading θ (rad)")
    ax_theta.set_title("Integrated Heading Angle  (θ = Σ dz)")
    ax_theta.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_theta.grid(True, alpha=_GRID_ALPHA)

    # Bottom panel: sin(θ) — bounded steering signal
    ax_sin: Axes = axes[1]
    ax_sin.plot(time, steering_sine, linewidth=0.7, color=_COLOR_DX, alpha=0.9)
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


def create_yaw_drift_panel(
    time: np.ndarray,
    dz: np.ndarray,
    session_id: str = "",
) -> Figure:
    """Yaw Drift Panel: Cumulative sum of dz over time.

    Used to verify:
    - 360-degree rotation accuracy (cumulative sum should match known rotations)
    - Zero-drift during stationary phases (flat line when not rotating)
    """
    cumulative_yaw = np.cumsum(dz)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(time, cumulative_yaw, linewidth=0.8, color=_COLOR_DZ, alpha=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Yaw (displacement units)")
    title = "Yaw Drift Analysis"
    if session_id:
        title += f" — {session_id}"
    ax.set_title(title, fontweight="bold")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=_GRID_ALPHA)

    # Annotate start and end drift
    if len(cumulative_yaw) > 0:
        start_val = cumulative_yaw[0]
        end_val = cumulative_yaw[-1]
        total_drift = end_val - start_val
        ax.annotate(
            f"Start: {start_val:.2f}",
            xy=(time[0], start_val),
            xytext=(time[0] + (time[-1] - time[0]) * 0.05, start_val),
            fontsize=9,
            color="#D4D4D4",
            arrowprops=dict(arrowstyle="->", color="#A0A0A5"),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#2D2D30", edgecolor="#4A4A4E", alpha=0.9),
        )
        ax.annotate(
            f"End: {end_val:.2f}\nDrift: {total_drift:.2f}",
            xy=(time[-1], end_val),
            xytext=(time[-1] - (time[-1] - time[0]) * 0.15, end_val),
            fontsize=9,
            color="#D4D4D4",
            arrowprops=dict(arrowstyle="->", color="#A0A0A5"),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#2D2D30", edgecolor="#4A4A4E", alpha=0.9),
        )

    fig.tight_layout()
    return fig


def create_jitter_histogram(
    speed: np.ndarray,
    session_id: str = "",
    bins: int = 200,
    stationary_percentile_threshold: float = 10.0,
) -> Figure:
    """Jitter Histogram: Speed distribution when system is supposedly stationary.

    Calculates and annotates the 95th and 99th percentiles (in mm/s).
    Used to calibrate the downstream Freezing threshold.

    Parameters
    ----------
    speed : np.ndarray
        Speed array (mm/s).
    session_id : str
        Session identifier for title.
    bins : int
        Histogram bin count.
    stationary_percentile_threshold : float
        Speed values below this percentile are considered "stationary"
        for jitter analysis. Default 10th percentile.
    """
    # Define "stationary" as speeds below a low percentile
    stationary_cutoff = np.percentile(speed, stationary_percentile_threshold)
    stationary_speed = speed[speed <= stationary_cutoff]

    if len(stationary_speed) == 0:
        # Fallback: use all speed if filtering yields nothing
        stationary_speed = speed
        stationary_cutoff = 0.0

    # Calculate percentiles
    p95 = np.percentile(stationary_speed, 95)
    p99 = np.percentile(stationary_speed, 99)
    mean_jitter = np.mean(stationary_speed)
    std_jitter = np.std(stationary_speed)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Histogram
    counts, bin_edges, patches = ax.hist(
        stationary_speed, bins=bins,
        color=_COLOR_DX, alpha=0.7,
        edgecolor="#1E1E1E", linewidth=0.2,
        density=True,
        label="Stationary speed",
    )

    # Percentile lines
    ax.axvline(p95, color=_COLOR_DY, linestyle="--", linewidth=1.5, label=f"95th percentile: {p95:.3f} mm/s")
    ax.axvline(p99, color="#e53e3e", linestyle="-.", linewidth=1.5, label=f"99th percentile: {p99:.3f} mm/s")
    ax.axvline(mean_jitter, color=_COLOR_DZ, linestyle="-", linewidth=1.2, label=f"Mean: {mean_jitter:.3f} mm/s")

    # Stats text box
    stats_text = (
        f"Stationary cutoff: ≤{stationary_cutoff:.3f} mm/s\n"
        f"n = {len(stationary_speed):,} samples\n"
        f"σ = {std_jitter:.4f} mm/s\n"
        f"95th: {p95:.3f} mm/s\n"
        f"99th: {p99:.3f} mm/s"
    )
    ax.text(
        0.98, 0.95, stats_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="right",
        color="#D4D4D4",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#2D2D30", edgecolor="#4A4A4E", alpha=0.9),
        family="monospace",
    )

    # Suggested freezing threshold
    suggested_threshold = p99 * 1.5
    ax.axvline(
        suggested_threshold, color="orange", linestyle=":",
        linewidth=1.5, alpha=0.8,
        label=f"Suggested threshold: {suggested_threshold:.3f} mm/s",
    )

    ax.set_xlabel("Speed (mm/s)")
    ax.set_ylabel("Density")
    title = "Stationary Jitter Analysis"
    if session_id:
        title += f" — {session_id}"
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=_GRID_ALPHA)

    fig.tight_layout()
    return fig


def compute_energy_ratio(
    primary: np.ndarray,
    cross_a: np.ndarray,
    cross_b: np.ndarray,
) -> float:
    """Compute Signal-to-Noise Energy Ratio for decoupled axes.

    SNR = Var(primary) / (Var(cross_a) + Var(cross_b))

    Replaces legacy Pearson correlation for calibration assessment.
    A high ratio indicates the primary axis carries dominant signal
    energy while cross-talk axes contribute only noise-floor variance.
    """
    var_primary = float(np.var(primary))
    var_cross = float(np.var(cross_a) + np.var(cross_b))
    if var_cross == 0.0:
        return float("inf") if var_primary > 0.0 else 0.0
    return var_primary / var_cross


def generate_calibration_report(
    df: pd.DataFrame,
    session_id: str = "",
) -> Dict[str, Figure]:
    """Generate all calibration panels from a raw kinematics DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: sys_time, dx, dy, dz.
    session_id : str
        Identifier for titles.

    Returns
    -------
    Dict[str, Figure]
        Keys: "cross_talk", "yaw_drift", "jitter_histogram", "steering_sine".
    """
    # Apply dark theme before generating figures
    try:
        from src.gui.theme import apply_mpl_theme
        apply_mpl_theme()
    except ImportError:
        pass  # Running without GUI (e.g., headless export)

    time = df["sys_time"].values.astype(np.float64)
    dx = df["dx"].values.astype(np.float64)
    dy = df["dy"].values.astype(np.float64)
    dz = df["dz"].values.astype(np.float64)

    # Compute speed from raw displacements
    dt = np.median(np.diff(time))
    if dt <= 0:
        dt = 1.0
    speed = np.sqrt(dx**2 + dy**2) / dt

    # Compute and log SNR energy ratios (replaces Pearson correlation)
    snr_x = compute_energy_ratio(dx, dy, dz)
    snr_y = compute_energy_ratio(dy, dx, dz)
    snr_z = compute_energy_ratio(dz, dx, dy)
    logger.info(
        "Calibration SNR — X: %.1f, Y: %.1f, Z: %.1f",
        snr_x, snr_y, snr_z,
    )

    return {
        "cross_talk": create_crosstalk_panel(time, dx, dy, dz, session_id),
        "yaw_drift": create_yaw_drift_panel(time, dz, session_id),
        "jitter_histogram": create_jitter_histogram(speed, session_id),
        "steering_sine": create_steering_sine_panel(time, dz, session_id),
    }


def export_calibration_report(
    figures: Dict[str, Figure],
    output_dir: Path,
    session_id: str = "",
) -> Path:
    """Export calibration figures as a single high-res PDF.

    Filename: Calibration_Report_<timestamp>.pdf

    Parameters
    ----------
    figures : Dict[str, Figure]
        Figures from generate_calibration_report.
    output_dir : Path
        Directory to save PDF.
    session_id : str
        Optional session ID for filename.

    Returns
    -------
    Path
        Path to saved PDF.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if session_id:
        filename = f"Calibration_Report_{session_id}_{timestamp}.pdf"
    else:
        filename = f"Calibration_Report_{timestamp}.pdf"
    filepath = output_dir / filename

    # Save all figures into a single multi-page PDF
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(filepath) as pdf:
        for fig in figures.values():
            pdf.savefig(fig, dpi=300, bbox_inches="tight")
        # Set PDF metadata
        d = pdf.infodict()
        d["Title"] = "Cercus Treadmill Calibration Report"
        d["Author"] = "Cercus Analysis"
        d["Subject"] = f"Hardware validation — {session_id}" if session_id else "Hardware validation"

    return filepath
