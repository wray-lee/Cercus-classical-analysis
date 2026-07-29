"""
Cercus Framework — Ternary State Classifier
============================================
Receives physical features from ``kinematics.py`` and routes each trial into
one of three response categories: **Escape**, **PreWalk**, or **NoResponse**.

Classification is orthogonal to measurement — this module never re-derives
geometric features; it only reads ``v_max`` / ``latency_ms`` and applies
baseline-state logic.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .constants import (
    ESCAPE_START_THRESHOLD,
    PREWALK_THRESHOLD,
    PREWALK_WINDOW_MS,
)
from .kinematics import compute_escape_interval, compute_escape_latency

log = logging.getLogger(__name__)


def classify_trial(trial: pd.DataFrame) -> dict[str, str | float]:
    """
    Ternary classification with **orthogonal** physical measurement.

    Priority order (orthogonal routing):

    1. **No-burst intercept** — no valid escape burst detected (``v_max`` ≤ 50
       or ``latency_ms`` is NaN).  Returns ``NoResponse`` with measured ``v_max``
       but ``latency_ms`` forced to NaN.  This is the *absolute veto*: regardless
       of pre-stimulus baseline speed, failing to reach 50 mm/s in [0, 250 ms]
       is always NoResponse.

    2. **PreWalk intercept** — a valid burst exists **and** pre-stimulus
       spontaneous activity exceeds ``PREWALK_THRESHOLD`` in the 1-s window
       before stimulus onset.

    3. **Escape** — a valid burst exists **and** baseline speed at t=0 is
       quiescent (< ``ESCAPE_START_THRESHOLD``).

    4. **Fallback** — NoResponse (e.g. burst exists but baseline ≥ 10 mm/s
       and no pre-walk activity).

    Returns
    -------
    dict with keys: ``response_type``, ``v_max``, ``latency_ms``, ``escape_interval_ms``,
    ``interval_onset_ms``, ``interval_offset_ms``
    """
    if trial.empty:
        return {"response_type": "NoResponse", "v_max": np.nan, "latency_ms": np.nan, "escape_interval_ms": np.nan, "interval_onset_ms": np.nan, "interval_offset_ms": np.nan}

    speed_vals = trial["speed"].values
    t_vals = trial["t_rel"].values
    ang_vel_vals = trial["angular_velocity"].values if "angular_velocity" in trial.columns else None

    # ── Determine stimulus onset in t_rel coordinates ──
    # For multimodal trials t_rel=0 is TTC; wind onset = target_ttc_ms on the
    # t_rel axis.  Passing this to compute_escape_latency shifts the burst
    # detection window to the real wind trigger so early-wind escapes are found.
    _ttc = trial["target_ttc_ms"].iloc[0] if "target_ttc_ms" in trial.columns else np.nan
    stim_onset: float | None = float(_ttc) if pd.notna(_ttc) else None
    onset = stim_onset if stim_onset is not None else 0.0

    # ── Extract trial type for baseline_visual special handling ──
    _trial_type = trial["type"].iloc[0] if "type" in trial.columns else None

    # ── 1. Physical measurement (always runs, never vetoed) ──
    result = compute_escape_latency(t_vals, speed_vals, stim_onset_t_rel=stim_onset, trial_type=_trial_type)
    v_max: float = result["v_max"]  # type: ignore[assignment]
    latency_ms: float = result["latency_ms"]  # type: ignore[assignment]

    has_burst: bool = not np.isnan(latency_ms)

    # ── Escape interval: 10 mm/s onset → 10 mm/s offset ──
    interval_result = compute_escape_interval(t_vals, speed_vals, latency_ms, trial_type=_trial_type, stim_onset_t_rel=stim_onset, angular_velocity=ang_vel_vals) if has_burst else {"interval_ms": np.nan, "onset_ms": np.nan, "offset_ms": np.nan}
    interval_ms = interval_result["interval_ms"]
    interval_onset_ms = interval_result["onset_ms"]
    interval_offset_ms = interval_result["offset_ms"]

    # ── 2. Priority 1 — No-burst absolute veto ──
    if not has_burst:
        return {"response_type": "NoResponse", "v_max": v_max, "latency_ms": np.nan, "escape_interval_ms": np.nan, "interval_onset_ms": np.nan, "interval_offset_ms": np.nan}

    # ── 3. Unified PreWalk detection anchored to escape onset ──
    # Anchor is the escape-onset time (interval_onset_ms), not stimulus onset.
    # This checks walking in [escape_onset - 1000, escape_onset - 50ms) — the
    # 50ms gap avoids edge contamination from the escape burst itself.
    _is_baseline_visual = (_trial_type is not None and "baseline_visual" in str(_trial_type))
    anchor = interval_onset_ms if pd.notna(interval_onset_ms) else latency_ms
    if pd.isna(anchor):
        log.warning("classify_trial: no valid anchor (interval_onset_ms and latency_ms both NaN)")
    else:
        pre_mask = (t_vals >= anchor - PREWALK_WINDOW_MS) & (t_vals < anchor - 50.0)
        if np.any(pre_mask):
            pre_slice = speed_vals[pre_mask]
            pre_slice = pre_slice[~np.isnan(pre_slice)]
            if len(pre_slice) > 0:
                pre_max = float(np.nanmax(pre_slice))
                frac_above = float(np.mean(pre_slice > PREWALK_THRESHOLD))
                if pre_max > PREWALK_THRESHOLD and frac_above > 0.15:
                    return {"response_type": "PreWalk", "v_max": v_max, "latency_ms": latency_ms, "escape_interval_ms": interval_ms, "interval_onset_ms": interval_onset_ms, "interval_offset_ms": interval_offset_ms}

    # ── 4. Escape — burst detected, no pre-walk activity ──
    # A valid burst with no walking in the pre-window is always Escape.
    return {"response_type": "Escape", "v_max": v_max, "latency_ms": latency_ms, "escape_interval_ms": interval_ms, "interval_onset_ms": interval_onset_ms, "interval_offset_ms": interval_offset_ms}


def label_trials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``response_type``, ``v_max``, ``latency_ms``, ``escape_interval_ms``,
    ``interval_onset_ms``, and ``interval_offset_ms`` columns to a preprocessed DataFrame.
    """
    classify_map: dict = {}
    v_max_map: dict = {}
    latency_map: dict = {}
    interval_map: dict = {}
    interval_onset_map: dict = {}
    interval_offset_map: dict = {}

    for tid, grp in df.groupby("global_trial_id"):
        result = classify_trial(grp)
        classify_map[tid] = result["response_type"]
        v_max_map[tid] = result["v_max"]
        latency_map[tid] = result["latency_ms"]
        interval_map[tid] = result["escape_interval_ms"]
        interval_onset_map[tid] = result["interval_onset_ms"]
        interval_offset_map[tid] = result["interval_offset_ms"]

    df = df.copy()
    df["response_type"] = df["global_trial_id"].map(classify_map)
    df["v_max"] = df["global_trial_id"].map(v_max_map)
    df["latency_ms"] = df["global_trial_id"].map(latency_map)
    df["escape_interval_ms"] = df["global_trial_id"].map(interval_map)
    df["interval_onset_ms"] = df["global_trial_id"].map(interval_onset_map)
    df["interval_offset_ms"] = df["global_trial_id"].map(interval_offset_map)

    n_escape = sum(1 for v in classify_map.values() if v == "Escape")
    n_prewalk = sum(1 for v in classify_map.values() if v == "PreWalk")
    n_none = sum(1 for v in classify_map.values() if v == "NoResponse")
    n_total = len(classify_map)
    log.info(
        "Ternary classification: Escape=%d, PreWalk=%d, NoResponse=%d (total=%d)",
        n_escape, n_prewalk, n_none, n_total,
    )
    return df
