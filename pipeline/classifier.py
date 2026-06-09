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
from .kinematics import compute_escape_latency

log = logging.getLogger(__name__)


def classify_trial(trial: pd.DataFrame) -> dict[str, str | float]:
    """
    Ternary classification with **orthogonal** physical measurement.

    Priority order:

    1. **PreWalk intercept** — pre-stimulus spontaneous activity.
       If any speed in ``[-PREWALK_WINDOW_MS, 0)`` exceeds ``PREWALK_THRESHOLD``,
       the trial is PreWalk.  ``v_max`` and ``latency_ms`` are always populated
       from the burst measurement (never vetoed).

    2. **Escape** — baseline quiescent at t=0 (speed < 10 mm/s) **and** a valid
       burst detected (``latency_ms`` is not NaN).

    3. **NoResponse** — fallback.

    Returns
    -------
    dict with keys: ``response_type``, ``v_max``, ``latency_ms``
    """
    if trial.empty:
        return {"response_type": "NoResponse", "v_max": np.nan, "latency_ms": np.nan}

    speed_vals = trial["speed"].values
    t_vals = trial["t_rel"].values

    # ── 1. Physical measurement (always runs, never vetoed) ──
    result = compute_escape_latency(t_vals, speed_vals)
    v_max: float = result["v_max"]  # type: ignore[assignment]
    latency_ms: float = result["latency_ms"]  # type: ignore[assignment]

    # ── 2. PreWalk intercept: pre-stimulus spontaneous activity ──
    pre_mask = (t_vals >= -PREWALK_WINDOW_MS) & (t_vals < 0)
    if np.any(pre_mask):
        pre_slice = speed_vals[pre_mask]
        if not np.all(np.isnan(pre_slice)):
            pre_v_max = float(np.nanmax(pre_slice))
            if pre_v_max > PREWALK_THRESHOLD:
                # PreWalk — but still propagate the burst measurement
                return {"response_type": "PreWalk", "v_max": v_max, "latency_ms": latency_ms}

    # ── 3. Escape: baseline quiescent + valid burst ──
    zero_idx = int(np.argmin(np.abs(t_vals)))
    baseline_speed = speed_vals[zero_idx]
    if (not np.isnan(baseline_speed)) and (baseline_speed < ESCAPE_START_THRESHOLD) and (not np.isnan(latency_ms)):
        return {"response_type": "Escape", "v_max": v_max, "latency_ms": latency_ms}

    # ── 4. Fallback ──
    return {"response_type": "NoResponse", "v_max": v_max, "latency_ms": latency_ms}


def label_trials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``response_type``, ``v_max``, and ``latency_ms`` columns to a preprocessed DataFrame.
    """
    classify_map: dict = {}
    v_max_map: dict = {}
    latency_map: dict = {}

    for tid, grp in df.groupby("global_trial_id"):
        result = classify_trial(grp)
        classify_map[tid] = result["response_type"]
        v_max_map[tid] = result["v_max"]
        latency_map[tid] = result["latency_ms"]

    df = df.copy()
    df["response_type"] = df["global_trial_id"].map(classify_map)
    df["v_max"] = df["global_trial_id"].map(v_max_map)
    df["latency_ms"] = df["global_trial_id"].map(latency_map)

    n_escape = sum(1 for v in classify_map.values() if v == "Escape")
    n_prewalk = sum(1 for v in classify_map.values() if v == "PreWalk")
    n_none = sum(1 for v in classify_map.values() if v == "NoResponse")
    n_total = len(classify_map)
    log.info(
        "Ternary classification: Escape=%d, PreWalk=%d, NoResponse=%d (total=%d)",
        n_escape, n_prewalk, n_none, n_total,
    )
    return df
