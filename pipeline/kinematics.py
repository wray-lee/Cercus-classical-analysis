"""
Cercus Framework — Pure Physics: Kinematics Integration & Latency Detection
============================================================================
Handles time-axis alignment, Savitzky-Golay smoothing, coordinate rotation,
and burst-feature extraction.  **No state-classification logic lives here.**
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .constants import (
    DETAILS_KEYS,
    RADIUS_MM,
    SPEED_WINDOW_MS,
    TRAJ_USE_ANGULAR_VELOCITY_OFFSET,
)
from cercus.core.kinematics.latency import (
    compute_escape_interval as _core_compute_escape_interval,
    compute_escape_latency as _core_compute_escape_latency,
    refine_offset_by_angular_velocity as _refine_offset_by_angular_velocity,
)

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Timestamp-Based Slicing & Per-Trial Integration
# ══════════════════════════════════════════════════════════════════════


def _slice_kinematics_by_window(kin: pd.DataFrame, window: dict) -> pd.DataFrame:
    """Boolean-mask slice of kinematics for one trial using ``[t_start, t_stop]``."""
    tid = window["global_trial_id"]
    t_start = window["t_start"]
    t_stop = window["t_stop"]

    mask = (kin["sys_time"] >= t_start) & (kin["sys_time"] <= t_stop)
    trial_kin = kin.loc[mask].copy()

    if trial_kin.empty:
        trial_kin = kin[kin["global_trial_id"] == tid].copy()
        if not trial_kin.empty:
            log.debug("Trial %s: timestamp slice empty, fell back to global_trial_id match.", tid)

    if trial_kin.empty:
        log.warning("Trial %s: no kinematics data found. Skipping.", tid)

    trial_kin["global_trial_id"] = tid
    return trial_kin


def _integrate_trial(grp: pd.DataFrame, t_zero_sys: float) -> pd.DataFrame:
    """Per-trial integration with lifecycle-anchored TTC time axis and denoised speed."""
    df = grp.copy()

    if len(df) < 2:
        df["t_rel"] = 0.0
        df["x"] = 0.0
        df["y"] = 0.0
        df["speed"] = np.nan
        return df

    df["t_rel"] = (df["sys_time"] - t_zero_sys) * 1000.0

    df.loc[df.index[:2], "dx"] = 0.0
    df.loc[df.index[:2], "dy"] = 0.0

    dx_body = -df["dx"].values
    dy_body = -df["dy"].values

    heading_rad = df["dz"].cumsum().values / RADIUS_MM

    dx_global = dx_body * np.cos(heading_rad) - dy_body * np.sin(heading_rad)
    dy_global = dx_body * np.sin(heading_rad) + dy_body * np.cos(heading_rad)

    x_raw = np.cumsum(dx_global)
    y_raw = np.cumsum(dy_global)

    median_dt = df["sys_time"].diff().replace(0, np.nan).median()
    if pd.notna(median_dt) and median_dt > 0:
        win = max(5, int(round((SPEED_WINDOW_MS / 1000.0) / median_dt)))
    else:
        win = 11
    if win % 2 == 0:
        win += 1
    poly_order = min(3, win - 1)

    n = len(x_raw)
    if n >= win:
        x_smooth = savgol_filter(x_raw, window_length=win, polyorder=poly_order, deriv=0)
        y_smooth = savgol_filter(y_raw, window_length=win, polyorder=poly_order, deriv=0)
    else:
        x_smooth = x_raw
        y_smooth = y_raw

    # Anchor spatial origin at the frame closest to t_rel = 0
    zero_idx = int(np.argmin(np.abs(df["t_rel"].values)))
    df["x"] = x_smooth - x_smooth[zero_idx]
    df["y"] = y_smooth - y_smooth[zero_idx]

    valid_dt = median_dt if (pd.notna(median_dt) and median_dt > 0) else 0.005

    dx_smooth = np.diff(x_smooth, prepend=x_smooth[0])
    dy_smooth = np.diff(y_smooth, prepend=y_smooth[0])
    if n > 1:
        dx_smooth[0] = x_smooth[1] - x_smooth[0]
        dy_smooth[0] = y_smooth[1] - y_smooth[0]

    # 分母由波动数组替换为常数
    speed = np.sqrt(dx_smooth**2 + dy_smooth**2) / valid_dt
    speed[0] = np.nan

    half_win = win // 2
    if half_win > 0:
        speed[:half_win] = np.nan
        speed[-half_win:] = np.nan

    df["speed"] = speed

    # ── Angular velocity (rad/s) from dz ──
    # Sign convention: positive = rightward turn, negative = leftward turn.
    # The raw dz integration yields the opposite sign, so we negate.
    heading_raw = -df["dz"].cumsum().values / RADIUS_MM
    if n >= win:
        heading_smooth = savgol_filter(heading_raw, window_length=win, polyorder=poly_order, deriv=0)
    else:
        heading_smooth = heading_raw
    angular_velocity = np.diff(heading_smooth, prepend=heading_smooth[0]) / valid_dt
    angular_velocity[0] = np.nan
    if half_win > 0:
        angular_velocity[:half_win] = np.nan
        angular_velocity[-half_win:] = np.nan
    df["angular_velocity"] = angular_velocity

    return df


# ══════════════════════════════════════════════════════════════════════
# Physical-Threshold Feature Extraction
# ══════════════════════════════════════════════════════════════════════


def compute_escape_latency(
    t_rel: np.ndarray,
    speed: np.ndarray,
    stim_onset_t_rel: float | None = None,
    trial_type: str | None = None,
    angular_velocity: np.ndarray | None = None,
    dz: np.ndarray | None = None,
    use_angular_onset_refinement: bool | None = None,
) -> dict[str, float | bool]:
    """Pure geometric burst-feature extraction — no baseline veto.

    Delegates to ``cercus.core.kinematics.latency.compute_escape_latency``.
    """
    return _core_compute_escape_latency(
        t_rel, speed, stim_onset_t_rel, trial_type,
        angular_velocity, dz, use_angular_onset_refinement,
    )


def compute_escape_interval(
    t_rel: np.ndarray,
    speed: np.ndarray,
    latency_ms: float,
    trial_type: str | None = None,
    stim_onset_t_rel: float | None = None,
    angular_velocity: np.ndarray | None = None,
    latency_coarse_ms: float | None = None,
) -> dict[str, float]:
    """Compute the escape interval (onset to offset in ms).

    Delegates to ``cercus.core.kinematics.latency.compute_escape_interval``,
    passing ``TRAJ_USE_ANGULAR_VELOCITY_OFFSET`` from the global config.
    """
    return _core_compute_escape_interval(
        t_rel, speed, latency_ms,
        trial_type=trial_type,
        stim_onset_t_rel=stim_onset_t_rel,
        angular_velocity=angular_velocity,
        use_angular_velocity_offset=TRAJ_USE_ANGULAR_VELOCITY_OFFSET,
        latency_coarse_ms=latency_coarse_ms,
    )


# ══════════════════════════════════════════════════════════════════════
# High-Level Preprocessing Pipeline
# ══════════════════════════════════════════════════════════════════════


def _compute_theoretical_ttc_ms(lv_ratio_ms: float, init_half_angle_deg: float) -> float:
    """Compute theoretical TTC (ms) from looming parameters for control trials."""
    rad = math.radians(init_half_angle_deg)
    denom = 1.0 - math.sin(rad)
    if denom <= 0:
        log.warning("init_half_angle_deg=%.1f yields sin≥1; TTC undefined.", init_half_angle_deg)
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
    """
    parts: list[pd.DataFrame] = []
    for window in trial_windows:
        tid = window["global_trial_id"]
        trial_kin = _slice_kinematics_by_window(kin, window)
        if trial_kin.empty:
            continue

        meta_row = trials_meta[trials_meta["global_trial_id"] == tid]
        if not meta_row.empty:
            for col in DETAILS_KEYS:
                trial_kin[col] = meta_row[col].iloc[0]

        # ── Determine t_zero_sys ──
        wind_active = trial_kin[trial_kin["stim_state"] > 0]
        has_wind = not wind_active.empty

        # Fetch trial metadata for multimodal detection.
        # NOTE: these are already in trial_kin via DETAILS_KEYS (line 221).
        lv_ratio = meta_row.get("lv_ratio_ms", pd.Series([np.nan])).iloc[0] if not meta_row.empty else np.nan
        init_angle = meta_row.get("init_half_angle_deg", pd.Series([np.nan])).iloc[0] if not meta_row.empty else np.nan
        trial_type = meta_row.get("type", pd.Series([None])).iloc[0] if not meta_row.empty else None
        target_ttc = meta_row.get("target_ttc_ms", pd.Series([np.nan])).iloc[0] if not meta_row.empty else np.nan

        # Broad multimodal detection — catch all trials that carry both wind
        # and looming stimuli, even when init_half_angle_deg is missing.
        # Criteria (any one suffices):
        #   1. trial_type == "looming_wind"   (explicit type tag)
        #   2. has_wind AND lv_ratio present  (looming params partially present)
        #   3. has_wind AND target_ttc_ms != 0 (non-zero TTC delay implies looming)
        _is_multimodal = has_wind and (
            (trial_type is not None and "looming" in str(trial_type))
            or pd.notna(lv_ratio)
            or (pd.notna(target_ttc) and target_ttc != 0)
        )

        # Strict looming flag — both params present, needed for theoretical TTC.
        has_full_looming = pd.notna(lv_ratio) and pd.notna(init_angle)

        # Multimodal (looming + wind): align t=0 to TTC, NOT wind onset.
        # `target_ttc_ms` is the signed wind-vs-TTC offset (negative = wind before TTC).
        # Setting t_zero_sys = wind_onset - target_ttc_ms/1000 places:
        #   t_rel = 0               → TTC moment
        #   t_rel = -target_ttc_ms  → wind hardware onset (stim_state > 0)
        # Examples: target_ttc_ms = -373 → wind at ~-373 ms;
        #           target_ttc_ms =    0 → wind at ~0 ms;
        #           target_ttc_ms = +200 → wind at ~+200 ms.
        if _is_multimodal:
            wind_onset_sys = float(wind_active.iloc[0]["sys_time"])
            if pd.notna(target_ttc):
                t_zero_sys = wind_onset_sys - (float(target_ttc) / 1000.0)
                log.info(
                    "Trial %s: MULTIMODAL, aligned to TTC (wind onset shifted by %+.0f ms).",
                    tid, -float(target_ttc),
                )
            else:
                # target_ttc_ms missing — fall back to wind onset
                t_zero_sys = wind_onset_sys
                log.warning(
                    "Trial %s: MULTIMODAL but target_ttc_ms is NaN; falling back to wind onset.", tid,
                )
        elif tid in ttc_anchors:
            t_zero_sys = ttc_anchors[tid]
            log.debug("Trial %s: using ttc_anchor=%.4f", tid, t_zero_sys)
        elif has_full_looming:
            t_col_ms = _compute_theoretical_ttc_ms(float(lv_ratio), float(init_angle))
            t_zero_sys = window["t_start"] + (t_col_ms / 1000.0)
            log.debug("Trial %s: pure looming, t_col_ms=%.1f ms", tid, t_col_ms)
        elif has_wind:
            t_zero_sys = float(wind_active.iloc[0]["sys_time"])
            log.info("Trial %s: pure wind, aligned to stim_state onset.", tid)
        else:
            t_zero_sys = (window["t_start"] + window["t_stop"]) / 2.0
            log.warning("Trial %s: absolute baseline (no visual, no wind); using trial midpoint as zero.", tid)

        try:
            parts.append(_integrate_trial(trial_kin, t_zero_sys))
        except Exception as exc:
            log.warning("Skipping trial %s during integration: %s", tid, exc)

    if not parts:
        raise RuntimeError("No valid trials after preprocessing.")
    return pd.concat(parts, ignore_index=True)
