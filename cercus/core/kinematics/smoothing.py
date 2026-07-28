"""
Cercus Core — Smoothing Utilities
==================================
Savitzky-Golay smoothing for kinematic data.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def savgol_smooth_xy(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    window_length: int,
    polyorder: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth x/y trajectories with Savitzky-Golay filter."""
    n = len(x_raw)
    if n >= window_length:
        x_smooth = savgol_filter(x_raw, window_length=window_length, polyorder=polyorder, deriv=0)
        y_smooth = savgol_filter(y_raw, window_length=window_length, polyorder=polyorder, deriv=0)
    else:
        x_smooth = x_raw.copy()
        y_smooth = y_raw.copy()
    return x_smooth, y_smooth


def savgol_smooth_heading(
    heading_raw: np.ndarray,
    window_length: int,
    polyorder: int = 3,
) -> np.ndarray:
    """Smooth heading angle with Savitzky-Golay filter."""
    n = len(heading_raw)
    if n >= window_length:
        return savgol_filter(heading_raw, window_length=window_length, polyorder=polyorder, deriv=0)
    return heading_raw.copy()


def compute_speed_from_xy(
    x_smooth: np.ndarray,
    y_smooth: np.ndarray,
    dt: float,
    half_win: int = 0,
) -> np.ndarray:
    """Compute instantaneous speed from smoothed x/y trajectories."""
    n = len(x_smooth)
    dx_smooth = np.diff(x_smooth, prepend=x_smooth[0])
    dy_smooth = np.diff(y_smooth, prepend=y_smooth[0])
    if n > 1:
        dx_smooth[0] = x_smooth[1] - x_smooth[0]
        dy_smooth[0] = y_smooth[1] - y_smooth[0]

    speed = np.sqrt(dx_smooth**2 + dy_smooth**2) / dt
    speed[0] = np.nan

    if half_win > 0:
        speed[:half_win] = np.nan
        speed[-half_win:] = np.nan

    return speed


def compute_angular_velocity(
    heading_smooth: np.ndarray,
    dt: float,
    half_win: int = 0,
) -> np.ndarray:
    """Compute angular velocity from smoothed heading."""
    angular_velocity = np.diff(heading_smooth, prepend=heading_smooth[0]) / dt
    angular_velocity[0] = np.nan
    if half_win > 0:
        angular_velocity[:half_win] = np.nan
        angular_velocity[-half_win:] = np.nan
    return angular_velocity
