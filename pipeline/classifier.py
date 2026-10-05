"""
Cercus Framework — Ternary State Classifier
============================================
Receives physical features from ``kinematics.py`` and routes each trial into
response categories: **Escape**, **PreEscape**, **PreWalk**, or **NoResponse**
(PreEscape is switch-controlled — ``classification.use_preescape``, default
true — and applies to wind trials whose measured onset precedes wind).

Classification is orthogonal to measurement — this module never re-derives
geometric features; it only reads ``v_max`` / ``latency_ms`` and applies
baseline-state logic.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .constants import (
    ESCAPE_VMAX_THRESHOLD,
    PREESCAPE_BUFFER_MS,
    PREWALK_THRESHOLD,
    PREWALK_WINDOW_MS,
    USE_PRE_ESCAPE,
)
from .kinematics import compute_escape_interval, compute_escape_latency
from cercus.config import get_thresholds
from cercus.core.kinematics.distance import compute_reaction_and_distance
from cercus.core.kinematics.latency import measure_pause_response, measure_stopping

log = logging.getLogger(__name__)


def classify_trial(
    trial: pd.DataFrame,
    use_angular_onset_refinement: bool | None = None,
) -> dict[str, str | float | bool]:
    """Measure burst/escape interval, then route NoResponse → PreEscape → PreWalk → Escape.

    Wind classification and escape RT share the observed acquisition-clock
    onset. Compatibility PreWalk RT measures stopping; descriptive latency
    and interval fields remain separate for distance and trajectory analysis.
    Neither short-RT QC nor missing stillness changes class eligibility.
    """
    pause = {
        "pause_stopping_time_ms": np.nan, "pause_to_escape_time_ms": np.nan,
        "pause_reaction_time_ms": np.nan, "pause_status": "not_applicable",
        "pause_escape_status": "not_applicable", "pause_baseline_status": "not_applicable",
        "pause_moving_fraction": np.nan, "pause_moving_eligible": False,
    }
    if trial.empty:
        return {"response_type": "NoResponse", "v_max": np.nan, "latency_ms": np.nan, "escape_interval_ms": np.nan, "interval_onset_ms": np.nan, "interval_offset_ms": np.nan, "reaction_time_ms": np.nan, "escape_onset_ms": np.nan, "escape_reaction_time_ms": np.nan, "stillness_reaction_time_ms": np.nan, "stillness_status": "invalid_data", "stillness_presence": "unobserved", "stop_to_escape_interval_ms": np.nan, "stillness_baseline_status": "unobserved", "stillness_window_start_ms": np.nan, "short_rt": False, "distance_mm": np.nan, "distance_500ms_mm": np.nan, **pause}

    speed_vals = trial["speed"].values
    t_vals = trial["t_rel"].values
    ang_vel_vals = trial["angular_velocity"].values if "angular_velocity" in trial.columns else None
    dz_vals = trial["dz"].values if "dz" in trial.columns else None

    # ── Determine stimulus onset in t_rel coordinates ──
    # For multimodal trials t_rel=0 is TTC; wind onset = target_ttc_ms on the
    # t_rel axis.  Passing this to compute_escape_latency shifts the burst
    # detection window to the real wind trigger so early-wind escapes are found.
    _ttc = trial["target_ttc_ms"].iloc[0] if "target_ttc_ms" in trial.columns else np.nan
    stim_onset: float | None = float(_ttc) if pd.notna(_ttc) else None
    onset = stim_onset if stim_onset is not None else 0.0

    # ── Extract trial type for baseline_visual special handling ──
    _trial_type = trial["type"].iloc[0] if "type" in trial.columns else None
    is_wind = _trial_type is not None and "wind" in str(_trial_type).lower()
    rt_cfg = get_thresholds().reaction_time
    # Calibration changes the moving-history and latency reference; burst and
    # PreEscape windows stay hardware-anchored. Zero delay is not measured arrival.
    rt_anchor = onset + float(rt_cfg.wind_arrival_delay_ms) if is_wind else onset

    # ── 1. Physical measurement (always runs, never vetoed) ──
    result = compute_escape_latency(
        t_vals, speed_vals, stim_onset_t_rel=stim_onset, trial_type=_trial_type,
        angular_velocity=ang_vel_vals, dz=dz_vals,
        use_angular_onset_refinement=use_angular_onset_refinement,
    )
    v_max: float = result["v_max"]  # type: ignore[assignment]
    latency_ms: float = result["latency_ms"]  # type: ignore[assignment]
    # Coarse 10 mm/s onset — interval / PreWalk anchors must not shift when
    # latency_ms is refined earlier by the angular-velocity onset.
    latency_coarse_ms: float = result.get("latency_coarse_ms", latency_ms)  # type: ignore[assignment]

    has_burst: bool = bool(np.isfinite(v_max) and v_max > ESCAPE_VMAX_THRESHOLD)

    # ── Escape interval: 10 mm/s onset → 10 mm/s offset ──
    interval_result = compute_escape_interval(
        t_vals, speed_vals, latency_ms, trial_type=_trial_type,
        stim_onset_t_rel=stim_onset, angular_velocity=ang_vel_vals,
        latency_coarse_ms=latency_coarse_ms,
    ) if has_burst else {"interval_ms": np.nan, "onset_ms": np.nan, "offset_ms": np.nan}
    interval_ms = interval_result["interval_ms"]
    interval_onset_ms = interval_result["onset_ms"]
    interval_offset_ms = interval_result["offset_ms"]

    # ── Stimulus-anchored RT + escape distance (NaN-safe; no-burst ⇒ NaN) ──
    # 刺激锚定反应时与行程：wind trial 锚在 target_ttc，其余锚在 t_rel=0。
    rt_dist = compute_reaction_and_distance(
        t_vals, speed_vals, interval_onset_ms, interval_offset_ms,
        anchor_ms=rt_anchor,
    )
    escape_rt = rt_dist["reaction_time_ms"]
    rt_dist.update(
        escape_onset_ms=np.nan,
        escape_reaction_time_ms=escape_rt,
        stillness_reaction_time_ms=np.nan,
        stillness_status="not_applicable",
        stillness_presence="not_applicable",
        stop_to_escape_interval_ms=np.nan,
        stillness_baseline_status="not_applicable",
        stillness_window_start_ms=np.nan,
        short_rt=bool(is_wind and 0.0 <= escape_rt < float(rt_cfg.short_escape_ms)),
    )

    has_acquisition = "speed_raw" in trial and "t_acquisition_rel" in trial
    if is_wind:
        if has_acquisition:
            pause = measure_pause_response(
                trial["t_acquisition_rel"].to_numpy(float),
                trial["speed_raw"].to_numpy(float), rt_anchor, str(_trial_type),
                hardware_onset_ms=onset,
            )
        else:
            pause.update(pause_status="missing_acquisition_clock", pause_baseline_status="unobserved")
    causal_onset = float(pause.pop("escape_onset_ms", np.nan))
    stopping_cap = float(pause.pop("stopping_cap_ms", np.nan))
    rt_dist.update(pause)

    # ── 2. Priority 1 — No-burst absolute veto ──
    if not has_burst:
        return {"response_type": "NoResponse", "v_max": v_max, "latency_ms": np.nan, "escape_interval_ms": np.nan, "interval_onset_ms": np.nan, "interval_offset_ms": np.nan, **rt_dist}

    # ── 3. Shared wind onset: classification, escape RT and stopping cap ──
    # Acquisition tables never substitute a centered onset for an unobserved
    # causal transition. Legacy tables retain their descriptive timing basis.
    # 居中区间只用于行程/轨迹；风试次判类与时间测量共用采集时钟起点。
    classification_onset = causal_onset if has_acquisition else interval_onset_ms
    if is_wind and has_acquisition:
        escape_rt = causal_onset - rt_anchor
        rt_dist.update(
            escape_onset_ms=causal_onset,
            reaction_time_ms=escape_rt,
            escape_reaction_time_ms=escape_rt,
            short_rt=bool(
                np.isfinite(escape_rt) and escape_rt >= 0.0
                and escape_rt < float(get_thresholds().reaction_time.short_escape_ms)
            ),
        )
    if (
        USE_PRE_ESCAPE
        and is_wind
        and pd.notna(classification_onset)
        and classification_onset < onset - PREESCAPE_BUFFER_MS
    ):
        if np.isfinite(causal_onset):
            rt_dist["reaction_time_ms"] = causal_onset - rt_anchor
            rt_dist["escape_reaction_time_ms"] = causal_onset - rt_anchor
            rt_dist["short_rt"] = False
        return {"response_type": "PreEscape", "v_max": v_max, "latency_ms": latency_ms, "escape_interval_ms": interval_ms, "interval_onset_ms": interval_onset_ms, "interval_offset_ms": interval_offset_ms, **rt_dist}

    # A negative PreEscape value is a lead time, not a wind reaction.
    # For post-wind responses, reject an onset before calibrated arrival.
    if is_wind and escape_rt < 0.0:
        rt_dist["reaction_time_ms"] = np.nan
        rt_dist["escape_reaction_time_ms"] = np.nan

    # ── 4. PreWalk detection — wind shares active-at-reference eligibility with RT ──
    prewalk = False
    if is_wind:
        prewalk = bool(pause["pause_moving_eligible"])
        rt_dist["stillness_baseline_status"] = pause["pause_baseline_status"]
        if has_acquisition:
            acquisition_t = trial["t_acquisition_rel"].to_numpy(float)
            acquisition_onset = causal_onset
            stopping = measure_stopping(
                acquisition_t, trial["speed_raw"].to_numpy(float),
                stopping_cap, rt_anchor,
            )
            stillness_rt = float(stopping["onset_ms"]) - rt_anchor
            rt_dist.update(
                stillness_reaction_time_ms=stillness_rt,
                stillness_status=stopping["status"],
                stillness_presence=stopping["presence"],
                stillness_window_start_ms=stillness_rt - float(get_thresholds().stillness.speed_window_ms),
            )
            stop_ms = float(stopping["onset_ms"])
            if np.isfinite(stop_ms) and np.isfinite(acquisition_onset):
                stop_matches = np.flatnonzero(acquisition_t == stop_ms)
                if len(stop_matches):
                    stop_index = int(stop_matches[0])
                    escape_matches = np.flatnonzero(acquisition_t == acquisition_onset)
                    escape_index = int(escape_matches[0]) if len(escape_matches) else stop_index
                    clock_segment = acquisition_t[stop_index:escape_index + 1]
                    if (len(clock_segment) >= 2 and np.all(np.isfinite(clock_segment))
                            and np.all(np.diff(clock_segment) > 0)):
                        rt_dist["stop_to_escape_interval_ms"] = acquisition_onset - stop_ms
        else:
            rt_dist["stillness_status"] = "missing_acquisition_clock"
            rt_dist["stillness_presence"] = "unobserved"
    else:
        # Visual-only trials keep their escape-anchored activity rule. There is
        # no source-clock stimulus baseline to claim the paper's wind cohort.
        anchor = interval_onset_ms if pd.notna(interval_onset_ms) else latency_ms
        if pd.isna(anchor):
            log.warning("classify_trial: no valid PreWalk anchor")
        else:
            pre_mask = (t_vals >= anchor - PREWALK_WINDOW_MS) & (t_vals < anchor - 50.0)
            pre_slice = speed_vals[pre_mask]
            pre_slice = pre_slice[~np.isnan(pre_slice)]
            if len(pre_slice):
                prewalk = bool(np.nanmax(pre_slice) > PREWALK_THRESHOLD
                               and np.mean(pre_slice > PREWALK_THRESHOLD) > 0.15)
    if prewalk:
        if is_wind:
            rt_dist["reaction_time_ms"] = rt_dist["stillness_reaction_time_ms"]
        return {"response_type": "PreWalk", "v_max": v_max, "latency_ms": latency_ms, "escape_interval_ms": interval_ms, "interval_onset_ms": interval_onset_ms, "interval_offset_ms": interval_offset_ms, **rt_dist}

    # ── 5. Escape — residual burst class, not evidence of a stationary baseline ──
    return {"response_type": "Escape", "v_max": v_max, "latency_ms": latency_ms, "escape_interval_ms": interval_ms, "interval_onset_ms": interval_onset_ms, "interval_offset_ms": interval_offset_ms, **rt_dist}


def label_trials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``response_type``, ``v_max``, ``latency_ms``, ``escape_interval_ms``,
    ``interval_onset_ms``, ``interval_offset_ms``, ``reaction_time_ms``,
    ``escape_reaction_time_ms``, ``stillness_reaction_time_ms``, ``short_rt``,
    ``distance_mm``, and ``distance_500ms_mm`` columns to a preprocessed DataFrame.
    """
    classify_map: dict = {}
    v_max_map: dict = {}
    latency_map: dict = {}
    interval_map: dict = {}
    interval_onset_map: dict = {}
    interval_offset_map: dict = {}
    rt_map: dict = {}
    escape_rt_map: dict = {}
    stillness_rt_map: dict = {}
    stillness_status_map: dict = {}
    stillness_presence_map: dict = {}
    stop_to_escape_map: dict = {}
    stillness_baseline_map: dict = {}
    stillness_window_map: dict = {}
    short_rt_map: dict = {}
    dist_map: dict = {}
    dist500_map: dict = {}
    pause_maps = {col: {} for col in (
        "escape_onset_ms", "pause_stopping_time_ms", "pause_to_escape_time_ms", "pause_reaction_time_ms",
        "pause_status", "pause_escape_status", "pause_baseline_status",
        "pause_moving_fraction", "pause_moving_eligible",
    )}

    for tid, grp in df.groupby("global_trial_id"):
        result = classify_trial(grp)
        classify_map[tid] = result["response_type"]
        v_max_map[tid] = result["v_max"]
        latency_map[tid] = result["latency_ms"]
        interval_map[tid] = result["escape_interval_ms"]
        interval_onset_map[tid] = result["interval_onset_ms"]
        interval_offset_map[tid] = result["interval_offset_ms"]
        rt_map[tid] = result["reaction_time_ms"]
        escape_rt_map[tid] = result["escape_reaction_time_ms"]
        stillness_rt_map[tid] = result["stillness_reaction_time_ms"]
        stillness_status_map[tid] = result["stillness_status"]
        stillness_presence_map[tid] = result["stillness_presence"]
        stop_to_escape_map[tid] = result["stop_to_escape_interval_ms"]
        stillness_baseline_map[tid] = result["stillness_baseline_status"]
        stillness_window_map[tid] = result["stillness_window_start_ms"]
        short_rt_map[tid] = result["short_rt"]
        dist_map[tid] = result["distance_mm"]
        dist500_map[tid] = result["distance_500ms_mm"]
        for col, values in pause_maps.items():
            values[tid] = result[col]

    df = df.copy()
    df["response_type"] = df["global_trial_id"].map(classify_map)
    df["v_max"] = df["global_trial_id"].map(v_max_map)
    df["latency_ms"] = df["global_trial_id"].map(latency_map)
    df["escape_interval_ms"] = df["global_trial_id"].map(interval_map)
    df["interval_onset_ms"] = df["global_trial_id"].map(interval_onset_map)
    df["interval_offset_ms"] = df["global_trial_id"].map(interval_offset_map)
    df["reaction_time_ms"] = df["global_trial_id"].map(rt_map)
    df["escape_reaction_time_ms"] = df["global_trial_id"].map(escape_rt_map)
    df["stillness_reaction_time_ms"] = df["global_trial_id"].map(stillness_rt_map)
    df["stillness_status"] = df["global_trial_id"].map(stillness_status_map)
    df["stillness_presence"] = df["global_trial_id"].map(stillness_presence_map)
    df["stop_to_escape_interval_ms"] = df["global_trial_id"].map(stop_to_escape_map)
    df["stillness_baseline_status"] = df["global_trial_id"].map(stillness_baseline_map)
    df["stillness_window_start_ms"] = df["global_trial_id"].map(stillness_window_map)
    df["short_rt"] = df["global_trial_id"].map(short_rt_map).fillna(False).astype(bool)
    df["distance_mm"] = df["global_trial_id"].map(dist_map)
    df["distance_500ms_mm"] = df["global_trial_id"].map(dist500_map)
    for col, values in pause_maps.items():
        df[col] = df["global_trial_id"].map(values)

    n_escape = sum(1 for v in classify_map.values() if v == "Escape")
    n_preescape = sum(1 for v in classify_map.values() if v == "PreEscape")
    n_prewalk = sum(1 for v in classify_map.values() if v == "PreWalk")
    n_none = sum(1 for v in classify_map.values() if v == "NoResponse")
    n_total = len(classify_map)
    log.info(
        "Classification: Escape=%d, PreEscape=%d, PreWalk=%d, NoResponse=%d (total=%d)",
        n_escape, n_preescape, n_prewalk, n_none, n_total,
    )
    return df
