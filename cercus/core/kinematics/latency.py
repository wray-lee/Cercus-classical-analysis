"""
Cercus Core — Escape Latency & Interval Detection
==================================================
Pure functions for backward-search latency detection and escape interval
computation. No matplotlib dependency.
"""

from __future__ import annotations

import numpy as np

from cercus.constants.thresholds import (
    ESCAPE_START_THRESHOLD,
    ESCAPE_VMAX_THRESHOLD,
    ESCAPE_WINDOW_MS,
)


def compute_escape_latency(
    t_rel: np.ndarray,
    speed: np.ndarray,
    stim_onset_t_rel: float | None = None,
    trial_type: str | None = None,
) -> dict[str, float]:
    """Pure geometric burst-feature extraction — no baseline veto.

    1. Measure ``v_max`` in the burst window:
       - For baseline_visual: search the entire stimulus period ``[onset, +∞)``
       - For others: ``[onset, onset + ESCAPE_WINDOW_MS]``
    2. If ``v_max > ESCAPE_VMAX_THRESHOLD``: locate the first frame exceeding
       50 mm/s, then search **backwards** through the full time series for the
       last frame below 10 mm/s.
    3. Otherwise return ``latency_ms = NaN``.
    """
    onset = 0.0 if stim_onset_t_rel is None else stim_onset_t_rel
    _is_baseline_visual = (trial_type is not None and "baseline_visual" in str(trial_type))

    if _is_baseline_visual:
        burst_mask = t_rel <= 0
        if not np.any(burst_mask):
            return {"v_max": np.nan, "latency_ms": np.nan}

        burst_speed = speed[burst_mask]
        if np.all(np.isnan(burst_speed)):
            return {"v_max": np.nan, "latency_ms": np.nan}

        v_max = float(np.nanmax(burst_speed))
        if np.isnan(v_max) or v_max <= ESCAPE_VMAX_THRESHOLD:
            return {"v_max": v_max, "latency_ms": np.nan}

        burst_indices = np.where(burst_mask)[0]
        peak_local = int(np.nanargmax(burst_speed))
        peak_idx = burst_indices[peak_local]

        search_back = speed[:peak_idx + 1]
        below_back = search_back < ESCAPE_START_THRESHOLD
        if np.any(below_back):
            last_below_idx = int(np.where(below_back)[0][-1])
            latency_ms = float(t_rel[last_below_idx + 1])
        else:
            latency_ms = float(t_rel[0])

        return {"v_max": v_max, "latency_ms": latency_ms}

    # Non-baseline-visual: fixed window [onset, onset + ESCAPE_WINDOW_MS]
    burst_mask = (t_rel >= onset) & (t_rel <= onset + ESCAPE_WINDOW_MS)

    if not np.any(burst_mask):
        return {"v_max": np.nan, "latency_ms": np.nan}

    burst_speed = speed[burst_mask]
    if np.all(np.isnan(burst_speed)):
        return {"v_max": np.nan, "latency_ms": np.nan}

    v_max = float(np.nanmax(burst_speed))

    if np.isnan(v_max) or v_max <= ESCAPE_VMAX_THRESHOLD:
        return {"v_max": v_max, "latency_ms": np.nan}

    burst_indices = np.where(burst_mask)[0]
    first_exceed_local = int(np.argmax(burst_speed > ESCAPE_VMAX_THRESHOLD))
    first_exceed_global = burst_indices[first_exceed_local]

    search_speed = speed[:first_exceed_global + 1]
    below_mask = search_speed < ESCAPE_START_THRESHOLD

    if np.any(below_mask):
        last_below_idx = int(np.where(below_mask)[0][-1])
        latency_ms = float(t_rel[last_below_idx + 1])
    else:
        latency_ms = float(t_rel[0])

    return {"v_max": v_max, "latency_ms": latency_ms}


def refine_offset_by_angular_velocity(
    t_rel: np.ndarray,
    angular_velocity: np.ndarray,
    onset_idx: int,
    offset_idx: int,
) -> int | None:
    """Return the index of the first angular-velocity zero-crossing after its peak."""
    segment = angular_velocity[onset_idx:offset_idx + 1].copy()
    valid = ~np.isnan(segment)
    if not np.any(valid):
        return None

    peak_local = None
    for i in range(len(segment) - 1):
        a, b = segment[i], segment[i + 1]
        if np.isnan(a) or np.isnan(b):
            continue
        if abs(a) > 0 and abs(a) >= abs(b):
            peak_local = i
            break
    if peak_local is None:
        peak_local = int(np.nanargmax(np.abs(segment)))
    escape_sign = np.sign(segment[peak_local])
    if escape_sign == 0:
        return None

    for i in range(peak_local, len(segment) - 1):
        a, b = segment[i], segment[i + 1]
        if np.isnan(a) or np.isnan(b):
            continue
        if a * b < 0 or a == 0.0 or b == 0.0:
            return onset_idx + i + (0 if a == 0.0 else 1)
    return None


def compute_escape_interval(
    t_rel: np.ndarray,
    speed: np.ndarray,
    latency_ms: float,
    trial_type: str | None = None,
    stim_onset_t_rel: float | None = None,
    angular_velocity: np.ndarray | None = None,
    use_angular_velocity_offset: bool = False,
) -> dict[str, float]:
    """Compute the escape interval (onset to offset in ms).

    For baseline_visual: interval = [peak前最近<10mm/s, peak后最近<10mm/s].
    For others: interval = [latency, 第一个<10mm/s].
    """
    nan_result = {"interval_ms": np.nan, "onset_ms": np.nan, "offset_ms": np.nan}

    if np.isnan(latency_ms):
        return nan_result

    _is_baseline_visual = (trial_type is not None and "baseline_visual" in str(trial_type))

    if _is_baseline_visual:
        mask = t_rel <= 0
        if not np.any(mask):
            return nan_result
        burst_speed = speed[mask]
        peak_local = int(np.nanargmax(burst_speed))
        peak_idx = np.where(mask)[0][peak_local]

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

        search_fwd = speed[peak_idx:]
        below_fwd = search_fwd < ESCAPE_START_THRESHOLD
        if np.any(below_fwd):
            offset_local = int(np.argmax(below_fwd))
            interval_offset_idx = peak_idx + offset_local
        else:
            return nan_result

        onset_ms = float(t_rel[interval_onset_idx])
        offset_ms = float(t_rel[interval_offset_idx])

        if use_angular_velocity_offset and angular_velocity is not None:
            refined_idx = refine_offset_by_angular_velocity(
                t_rel, angular_velocity, interval_onset_idx, interval_offset_idx,
            )
            if refined_idx is not None:
                offset_ms = float(t_rel[refined_idx])

        return {"interval_ms": offset_ms - onset_ms, "onset_ms": onset_ms, "offset_ms": offset_ms}
    else:
        onset_idx = int(np.argmin(np.abs(t_rel - latency_ms)))
        post_onset_speed = speed[onset_idx:]
        below_mask = post_onset_speed < ESCAPE_START_THRESHOLD
        if np.any(below_mask):
            offset_local = int(np.argmax(below_mask))
            offset_idx = onset_idx + offset_local
            onset_ms = float(t_rel[onset_idx])
            offset_ms = float(t_rel[offset_idx])

            if use_angular_velocity_offset and angular_velocity is not None:
                refined_idx = refine_offset_by_angular_velocity(
                    t_rel, angular_velocity, onset_idx, offset_idx,
                )
                if refined_idx is not None:
                    offset_ms = float(t_rel[refined_idx])

            return {"interval_ms": offset_ms - onset_ms, "onset_ms": onset_ms, "offset_ms": offset_ms}

    return nan_result
