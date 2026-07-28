"""
Cercus Core — Velocity Computation
===================================
Pure functions for computing velocity from body-frame micro-displacements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cercus.constants.geometry import RADIUS_MM
from cercus.core.kinematics.smoothing import (
    compute_angular_velocity,
    compute_speed_from_xy,
    savgol_smooth_heading,
    savgol_smooth_xy,
)


def compute_global_coordinates(
    dx_body: np.ndarray,
    dy_body: np.ndarray,
    heading_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate body-frame dx/dy into global coordinates."""
    dx_global = dx_body * np.cos(heading_rad) - dy_body * np.sin(heading_rad)
    dy_global = dx_body * np.sin(heading_rad) + dy_body * np.cos(heading_rad)
    x_raw = np.cumsum(dx_global)
    y_raw = np.cumsum(dy_global)
    return x_raw, y_raw


def compute_heading_from_dz(dz: np.ndarray) -> np.ndarray:
    """Compute cumulative heading angle from body-frame dz."""
    return dz.cumsum() / RADIUS_MM


def compute_speed_and_angular_velocity(
    df: pd.DataFrame,
    speed_window_ms: float,
    median_dt: float | None = None,
) -> pd.DataFrame:
    """Add speed and angular_velocity columns to a trial DataFrame."""
    df = df.copy()
    n = len(df)

    dx_body = -df["dx"].values
    dy_body = -df["dy"].values
    heading_rad = compute_heading_from_dz(df["dz"].values)

    x_raw, y_raw = compute_global_coordinates(dx_body, dy_body, heading_rad)

    if median_dt is None or np.isnan(median_dt) or median_dt <= 0:
        median_dt = df["sys_time"].diff().replace(0, np.nan).median()
    if pd.notna(median_dt) and median_dt > 0:
        win = max(5, int(round((speed_window_ms / 1000.0) / median_dt)))
    else:
        win = 11
    if win % 2 == 0:
        win += 1
    poly_order = min(3, win - 1)

    half_win = win // 2
    valid_dt = median_dt if (pd.notna(median_dt) and median_dt > 0) else 0.005

    x_smooth, y_smooth = savgol_smooth_xy(x_raw, y_raw, win, poly_order)

    speed = compute_speed_from_xy(x_smooth, y_smooth, valid_dt, half_win)
    df["speed"] = speed

    heading_raw = compute_heading_from_dz(-df["dz"].values)
    heading_smooth = savgol_smooth_heading(heading_raw, win, poly_order)
    angular_velocity = compute_angular_velocity(heading_smooth, valid_dt, half_win)
    df["angular_velocity"] = angular_velocity

    return df
