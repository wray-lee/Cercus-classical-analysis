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
    ESCAPE_START_THRESHOLD,
    ESCAPE_VMAX_THRESHOLD,
    ESCAPE_WINDOW_MS,
    RADIUS_MM,
    SPEED_WINDOW_MS,
    TRAJ_USE_ANGULAR_VELOCITY_OFFSET,
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
) -> dict[str, float | bool]:
    """
    Pure geometric burst-feature extraction — **no baseline veto**.

    1. Measure ``v_max`` in the burst window:
       - For baseline_visual: search the entire stimulus period ``[onset, +∞)``
       - For others: ``[onset, onset + ESCAPE_WINDOW_MS]``
       where ``onset = stim_onset_t_rel`` (defaults to 0 when *None*).
    2. If ``v_max > ESCAPE_VMAX_THRESHOLD``: locate the first frame exceeding
       50 mm/s, then search **backwards** through the full time series for the
       last frame below 10 mm/s.  The frame immediately following is the true
       latency (in t_rel coordinates, relative to TTC).
    3. Otherwise return ``latency_ms = NaN``.

    Parameters
    ----------
    t_rel : np.ndarray
        Time axis in ms, relative to TTC (t_rel=0 → TTC).
    speed : np.ndarray
        Instantaneous speed in mm/s.
    stim_onset_t_rel : float or None
        Stimulus onset on the t_rel axis (ms).  For multimodal trials this
        equals ``target_ttc_ms`` (the wind-vs-TTC signed offset); for pure
        visual or pure wind trials leave as *None* to default to 0.
    trial_type : str or None
        Trial type string (e.g. ``"baseline_visual"``, ``"looming_wind"``).
        When ``"baseline_visual"``, the burst window covers the entire
        post-onset period instead of the fixed ``ESCAPE_WINDOW_MS`` window.

    Returns
    -------
    dict with keys: ``v_max`` (float), ``latency_ms`` (float or NaN)
    """
    # ── Burst window ──
    # For multimodal trials, stim_onset_t_rel shifts the window to match the
    # actual wind onset (e.g. -373 ms for the 30-degree paradigm), so
    # wind-triggered escapes before TTC are still detected.
    onset = 0.0 if stim_onset_t_rel is None else stim_onset_t_rel

    _is_baseline_visual = (trial_type is not None and "baseline_visual" in str(trial_type))

    if _is_baseline_visual:
        # ── Baseline visual: search from trial start to TTC (t_rel <= 0) ──
        # 1. Find the peak (>50 mm/s) with maximum speed in [trial_start, t=0]
        # 2. Search BACKWARD from peak for nearest < 10 mm/s → latency
        # 3. Search FORWARD from peak for nearest < 10 mm/s → escape end
        burst_mask = t_rel <= 0
        if not np.any(burst_mask):
            return {"v_max": np.nan, "latency_ms": np.nan}

        burst_speed = speed[burst_mask]
        if np.all(np.isnan(burst_speed)):
            return {"v_max": np.nan, "latency_ms": np.nan}

        v_max = float(np.nanmax(burst_speed))
        if np.isnan(v_max) or v_max <= ESCAPE_VMAX_THRESHOLD:
            return {"v_max": v_max, "latency_ms": np.nan}

        # Find peak index in the full array
        burst_indices = np.where(burst_mask)[0]
        peak_local = int(np.nanargmax(burst_speed))
        peak_idx = burst_indices[peak_local]

        # Search BACKWARD from peak for nearest < 10 mm/s
        # latency = reaction time = first frame >= 10 mm/s (escape onset)
        search_back = speed[:peak_idx + 1]
        below_back = search_back < ESCAPE_START_THRESHOLD
        if np.any(below_back):
            last_below_idx = int(np.where(below_back)[0][-1])
            latency_ms = float(t_rel[last_below_idx + 1])
        else:
            latency_ms = float(t_rel[0])

        return {"v_max": v_max, "latency_ms": latency_ms}

    # ── Non-baseline-visual: fixed window [onset, onset + ESCAPE_WINDOW_MS] ──
    burst_mask = (t_rel >= onset) & (t_rel <= onset + ESCAPE_WINDOW_MS)

    if not np.any(burst_mask):
        return {"v_max": np.nan, "latency_ms": np.nan}

    burst_speed = speed[burst_mask]
    if np.all(np.isnan(burst_speed)):
        return {"v_max": np.nan, "latency_ms": np.nan}

    v_max = float(np.nanmax(burst_speed))

    if np.isnan(v_max) or v_max <= ESCAPE_VMAX_THRESHOLD:
        return {"v_max": v_max, "latency_ms": np.nan}

    # ── Backward-search latency (full-history, pierces t=0) ──
    burst_indices = np.where(burst_mask)[0]
    first_exceed_local = int(np.argmax(burst_speed > ESCAPE_VMAX_THRESHOLD))
    first_exceed_global = burst_indices[first_exceed_local]

    # Search backwards from this point through the ENTIRE array (including t < 0)
    search_speed = speed[: first_exceed_global + 1]
    below_mask = search_speed < ESCAPE_START_THRESHOLD

    if np.any(below_mask):
        last_below_idx = int(np.where(below_mask)[0][-1])
        latency_ms = float(t_rel[last_below_idx + 1])
    else:
        latency_ms = float(t_rel[0])

    return {"v_max": v_max, "latency_ms": latency_ms}


def _refine_offset_by_angular_velocity(
    t_rel: np.ndarray,
    angular_velocity: np.ndarray,
    onset_idx: int,
    offset_idx: int,
) -> float | None:
    """Refine escape offset to the first angular-velocity zero-crossing after its peak.

    Searches the angular velocity within ``[onset_idx, offset_idx]``, locates
    the peak (by absolute value), then scans forward for the first point where
    the signal crosses zero.  Returns the refined offset time in ms, or
    ``None`` if no valid peak / zero-crossing is found.
    """
    segment = angular_velocity[onset_idx:offset_idx + 1].copy()
    # Treat NaN as 0 for peak / crossing detection
    valid = ~np.isnan(segment)
    if not np.any(valid):
        return None
    seg_abs = np.abs(segment)
    peak_local = int(np.nanargmax(seg_abs))
    # Search forward from peak for sign change (zero crossing)
    for i in range(peak_local, len(segment) - 1):
        a, b = segment[i], segment[i + 1]
        if np.isnan(a) or np.isnan(b):
            continue
        # Zero crossing: sign changes, or either value is exactly 0
        if a * b < 0 or a == 0.0 or b == 0.0:
            return float(t_rel[onset_idx + i + (0 if a == 0.0 else 1)])
    return None


def compute_escape_interval(
    t_rel: np.ndarray,
    speed: np.ndarray,
    latency_ms: float,
    trial_type: str | None = None,
    stim_onset_t_rel: float | None = None,
    angular_velocity: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute the escape interval.

    For baseline_visual: interval = [peak前最近<10mm/s, peak后最近<10mm/s].
    Onset is constrained to [stim_onset_t_rel, ttc=0]; offset is unconstrained after peak.
    For others: interval = [latency, 第一个<10mm/s].

    When *TRAJ_USE_ANGULAR_VELOCITY_OFFSET* is enabled and *angular_velocity*
    is provided, the offset is refined to the first zero-crossing of angular
    velocity after its peak within the escape interval (i.e. the point where
    the rotational burst subsides).

    Parameters
    ----------
    t_rel : np.ndarray
        Time axis in ms, relative to TTC.
    speed : np.ndarray
        Instantaneous speed in mm/s.
    latency_ms : float
        Latency (reaction time) in ms, as returned by ``compute_escape_latency``.
    trial_type : str or None
        Trial type string.
    stim_onset_t_rel : float or None
        Stimulus onset on the t_rel axis (ms). For baseline_visual, the onset
        search is constrained to [stim_onset_t_rel, t_rel=0].
    angular_velocity : np.ndarray or None
        Per-frame angular velocity in rad/s. Required for the angular-velocity
        offset refinement; ignored when the feature is disabled.

    Returns
    -------
    dict with keys: ``interval_ms``, ``onset_ms``, ``offset_ms``.
    All NaN if latency_ms is NaN or speed never drops back below threshold.
    """
    nan_result = {"interval_ms": np.nan, "onset_ms": np.nan, "offset_ms": np.nan}

    if np.isnan(latency_ms):
        return nan_result

    _is_baseline_visual = (trial_type is not None and "baseline_visual" in str(trial_type))

    if _is_baseline_visual:
        # For baseline_visual: find peak in t_rel <= 0
        mask = t_rel <= 0
        if not np.any(mask):
            return nan_result
        burst_speed = speed[mask]
        peak_local = int(np.nanargmax(burst_speed))
        peak_idx = np.where(mask)[0][peak_local]

        # Search BACKWARD from peak for nearest < 10 mm/s → interval onset
        # Constrain to [stim_onset_t_rel, peak_idx] if stim_onset_t_rel is given
        if stim_onset_t_rel is not None:
            lower_bound_idx = int(np.searchsorted(t_rel, stim_onset_t_rel))
        else:
            lower_bound_idx = 0
        search_back = speed[lower_bound_idx:peak_idx + 1]
        below_back = search_back < ESCAPE_START_THRESHOLD
        if np.any(below_back):
            interval_onset_idx = lower_bound_idx + int(np.where(below_back)[0][-1])
        else:
            interval_onset_idx = lower_bound_idx

        # Search FORWARD from peak for nearest < 10 mm/s → interval offset
        # Unconstrained: search the entire trial after the peak
        search_fwd = speed[peak_idx:]
        below_fwd = search_fwd < ESCAPE_START_THRESHOLD
        if np.any(below_fwd):
            offset_local = int(np.argmax(below_fwd))
            interval_offset_idx = peak_idx + offset_local
        else:
            return nan_result

        onset_ms = float(t_rel[interval_onset_idx])
        offset_ms = float(t_rel[interval_offset_idx])

        # ── Angular velocity offset refinement ──
        if TRAJ_USE_ANGULAR_VELOCITY_OFFSET and angular_velocity is not None:
            refined = _refine_offset_by_angular_velocity(
                t_rel, angular_velocity, interval_onset_idx, interval_offset_idx,
            )
            if refined is not None:
                offset_ms = refined

        return {"interval_ms": offset_ms - onset_ms, "onset_ms": onset_ms, "offset_ms": offset_ms}
    else:
        # For others: interval from latency to first <10mm/s after latency
        onset_idx = int(np.argmin(np.abs(t_rel - latency_ms)))
        post_onset_speed = speed[onset_idx:]
        below_mask = post_onset_speed < ESCAPE_START_THRESHOLD
        if np.any(below_mask):
            offset_local = int(np.argmax(below_mask))
            offset_idx = onset_idx + offset_local
            onset_ms = float(t_rel[onset_idx])
            offset_ms = float(t_rel[offset_idx])

            # ── Angular velocity offset refinement ──
            if TRAJ_USE_ANGULAR_VELOCITY_OFFSET and angular_velocity is not None:
                refined = _refine_offset_by_angular_velocity(
                    t_rel, angular_velocity, onset_idx, offset_idx,
                )
                if refined is not None:
                    offset_ms = refined

            return {"interval_ms": offset_ms - onset_ms, "onset_ms": onset_ms, "offset_ms": offset_ms}

    return nan_result


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
