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
    dict with keys: ``response_type``, ``v_max``, ``latency_ms``, ``escape_interval_ms``
    """
    if trial.empty:
        return {"response_type": "NoResponse", "v_max": np.nan, "latency_ms": np.nan, "escape_interval_ms": np.nan}

    speed_vals = trial["speed"].values
    t_vals = trial["t_rel"].values

    # ── Determine stimulus onset in t_rel coordinates ──
    # For multimodal trials t_rel=0 is TTC; wind onset = target_ttc_ms on the
    # t_rel axis.  Passing this to compute_escape_latency shifts the burst
    # detection window to the real wind trigger so early-wind escapes are found.
    _ttc = trial["target_ttc_ms"].iloc[0] if "target_ttc_ms" in trial.columns else np.nan
    stim_onset: float | None = float(_ttc) if pd.notna(_ttc) else None
    onset = stim_onset if stim_onset is not None else 0.0

    # ── 1. Physical measurement (always runs, never vetoed) ──
    result = compute_escape_latency(t_vals, speed_vals, stim_onset_t_rel=stim_onset)
    v_max: float = result["v_max"]  # type: ignore[assignment]
    latency_ms: float = result["latency_ms"]  # type: ignore[assignment]

    has_burst: bool = not np.isnan(latency_ms)

    # ── Escape interval: 10 mm/s onset → 10 mm/s offset ──
    interval_ms = compute_escape_interval(t_vals, speed_vals, latency_ms) if has_burst else np.nan

    # ── 2. Priority 1 — No-burst absolute veto ──
    if not has_burst:
        return {"response_type": "NoResponse", "v_max": v_max, "latency_ms": np.nan, "escape_interval_ms": np.nan}

    # ── 3. Priority 2 — PreWalk intercept: pre-stimulus spontaneous activity ──
    # Window is relative to stimulus onset (wind for multimodal, TTC otherwise).
    pre_mask = (t_vals >= onset - PREWALK_WINDOW_MS) & (t_vals < onset)
    if np.any(pre_mask):
        pre_slice = speed_vals[pre_mask]
        if not np.all(np.isnan(pre_slice)):
            pre_v_max = float(np.nanmax(pre_slice))
            if pre_v_max > PREWALK_THRESHOLD:
                return {"response_type": "PreWalk", "v_max": v_max, "latency_ms": latency_ms, "escape_interval_ms": interval_ms}

    # ── 4. Priority 3 — Escape: baseline quiescent at stimulus onset ──
    zero_idx = int(np.argmin(np.abs(t_vals - onset)))
    baseline_speed = speed_vals[zero_idx]

    # When the exact-onset speed is NaN (common in pure-wind trials where the
    # kinematics stream has gaps around t_rel=0), fall back to the nearest
    # non-NaN PRE-STIMULUS value within 2 s before onset.
    # (Pure-wind paradigm guarantees the animal is stationary for ≥2 s before wind.)
    if np.isnan(baseline_speed):
        pre_onset_mask = (t_vals >= onset - 2000.0) & (t_vals < onset)
        if np.any(pre_onset_mask):
            pre_speeds = speed_vals[pre_onset_mask]
            valid = pre_speeds[~np.isnan(pre_speeds)]
            if len(valid) > 0:
                # Take the value closest to onset (last in the pre-stimulus window)
                baseline_speed = valid[-1]

    # If baseline is still NaN (no valid pre-stimulus data), assume quiescent —
    # the animal was stationary before the stimulus.  This is the common case
    # for pure-wind paradigms where kinematics data only starts after wind onset.
    if np.isnan(baseline_speed):
        baseline_speed = 0.0

    if baseline_speed < ESCAPE_START_THRESHOLD:
        return {"response_type": "Escape", "v_max": v_max, "latency_ms": latency_ms, "escape_interval_ms": interval_ms}

    # ── 5. Fallback ──
    return {"response_type": "NoResponse", "v_max": v_max, "latency_ms": np.nan, "escape_interval_ms": np.nan}


def label_trials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``response_type``, ``v_max``, ``latency_ms``, and ``escape_interval_ms``
    columns to a preprocessed DataFrame.
    """
    classify_map: dict = {}
    v_max_map: dict = {}
    latency_map: dict = {}
    interval_map: dict = {}

    for tid, grp in df.groupby("global_trial_id"):
        result = classify_trial(grp)
        classify_map[tid] = result["response_type"]
        v_max_map[tid] = result["v_max"]
        latency_map[tid] = result["latency_ms"]
        interval_map[tid] = result["escape_interval_ms"]

    df = df.copy()
    df["response_type"] = df["global_trial_id"].map(classify_map)
    df["v_max"] = df["global_trial_id"].map(v_max_map)
    df["latency_ms"] = df["global_trial_id"].map(latency_map)
    df["escape_interval_ms"] = df["global_trial_id"].map(interval_map)

    n_escape = sum(1 for v in classify_map.values() if v == "Escape")
    n_prewalk = sum(1 for v in classify_map.values() if v == "PreWalk")
    n_none = sum(1 for v in classify_map.values() if v == "NoResponse")
    n_total = len(classify_map)
    log.info(
        "Ternary classification: Escape=%d, PreWalk=%d, NoResponse=%d (total=%d)",
        n_escape, n_prewalk, n_none, n_total,
    )
    return df
