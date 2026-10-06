"""
Cercus Framework — Independent Pre-Stimulus Baseline Measurement
================================================================
Measures motion in the one-second window *before the first stimulus*, on the
raw full-session kinematics. Host ``sys_time`` only *locates* the documented
event's sample; the acquisition clock ``ard_time`` carries the history and its
averaging margin, and the ONLY speed denominator is the acquisition interval
(``hypot(dx, dy) / Δard_time``). The classifier's cropped trial slice and its
boundary-reset displacements are never used, so remote-wind and NoResponse
trials still get a real baseline.

Reference scheme (``prestim_reference_kind``):

* ``visual_event_sample`` — visual / multisensory trials. The documented
  ``Looming`` phase-transition event (the FIRST stimulus) mapped to the last
  observed session sample at or before it. A missing event stays ``unobserved``
  — it never falls back to a later wind onset, trial start, TTC, or escape.
  A mapping farther than one acquisition gap is remote and also stays
  ``unobserved``.
* ``calibrated_wind`` — wind-only trials. The observed first binary valve frame
  inside the trial window plus ``reaction_time.wind_arrival_delay_ms`` (the same
  primary reference as the ``pause_*`` measurement); this reference need not be
  a sampled time.
* ``unobserved`` — no visual and no wind stimulus, or an unmeasurable history.

The history is the maximal contiguous valid-acquisition component around the
reference sample: the walk stops at the first invalid, non-increasing, or
over-gap interval, so a full-session history may reach into the previous trial
id but never across a session or an invalid timestamp. Threshold logic is not
reimplemented — the shared :func:`_causal_displacement_speed` /
:func:`_movement_history_metrics` are reused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cercus.config import get_thresholds
from cercus.core.kinematics.latency import (
    _causal_displacement_speed,
    _movement_history_metrics,
)

KIND_VISUAL = "visual_event_sample"
KIND_WIND = "calibrated_wind"
KIND_UNOBSERVED = "unobserved"

#: Columns attached per trial (all classes, including NoResponse).
PRESTIM_COLUMNS: tuple[str, ...] = (
    "prestim_reference_kind",
    "prestim_reference_offset_ms",
    "prestim_reference_sample_offset_ms",
    "prestim_status",
    "prestim_moving_fraction",
    "prestim_reference_speed_mm_s",
)

_UNOBSERVED_ROW: dict = {
    "prestim_reference_kind": KIND_UNOBSERVED,
    "prestim_reference_offset_ms": np.nan,
    "prestim_reference_sample_offset_ms": np.nan,
    "prestim_status": KIND_UNOBSERVED,
    "prestim_moving_fraction": np.nan,
    "prestim_reference_speed_mm_s": np.nan,
}


def _session_speed_gap(
    ard_ms: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    factor: float,
) -> tuple[np.ndarray, float]:
    """Optical speed on the acquisition clock, plus the session gap scale (ms).

    The denominator is the acquisition interval only. ``speed`` is ``NaN``
    wherever the interval is missing, non-positive, or larger than the gap
    scale, so a diff across a session boundary can never manufacture a value.
    """
    dt = np.diff(ard_ms)
    positive = dt[np.isfinite(dt) & (dt > 0)]
    gap = float(np.median(positive) * factor) if len(positive) and np.isfinite(factor) and factor > 0 else np.nan
    speed = np.full(ard_ms.shape, np.nan)
    valid = np.isfinite(dt) & (dt > 0) & (dt <= gap)
    idx = np.flatnonzero(valid) + 1
    speed[idx] = np.hypot(dx[idx], dy[idx]) * 1000.0 / dt[valid]
    return speed, gap


def _component_bounds(
    source: np.ndarray, mapped: int, gap: float, ref_ms: float,
    window_ms: float, width_ms: float,
) -> tuple[int, int]:
    """Contiguous valid-acquisition bounds around *mapped*, trimmed to what the
    history actually reads: left to one observation before the averaging margin
    (``ref − window − width``), right to the first observation at/after the
    calibrated reference. Never bridged across a gap or invalid timestamp.
    """
    def _step(i: int, j: int) -> bool:
        return np.isfinite(source[j]) and 0.0 < source[j] - source[i] <= gap

    start = ref_ms - window_ms
    left = mapped
    while left > 0 and _step(left - 1, left) and source[left] > start:
        left -= 1
    # The observation carried across the left boundary needs its OWN averaging
    # margin; irregular sampling can put that margin before start - width.
    target = min(start, source[left]) - width_ms
    while left > 0 and _step(left - 1, left) and source[left] > target:
        left -= 1
    if left > 0 and _step(left - 1, left) and source[left - 1] <= target:
        left -= 1
    right = mapped
    while right + 1 < len(source) and _step(right, right + 1) and source[right + 1] <= ref_ms:
        right += 1
    # Cover a calibrated reference that falls between samples (delay > 0).
    if right + 1 < len(source) and _step(right, right + 1):
        right += 1
    return left, right


def measure_prestim_baseline(
    times: np.ndarray,
    speeds: np.ndarray,
    reference_ms: float,
    window_ms: float,
) -> tuple[str, float, float]:
    """Causal history on one local acquisition component → ``(status, fraction, speed)``.

    ``unobserved`` (with ``NaN`` metrics) whenever the component cannot cover
    the history window with its preceding averaging margin, or a gap/invalid
    frame falls inside it. Reuses the shared causal average and history
    classifier — no threshold logic lives here.
    """
    if not np.isfinite(reference_ms) or times.size < 2 or times.size != speeds.size:
        return KIND_UNOBSERVED, np.nan, np.nan
    averaged, _valid, local_gap = _causal_displacement_speed(times, speeds)
    status, fraction, reference_speed = _movement_history_metrics(
        times, averaged, reference_ms, window_ms, local_gap,
    )
    if not np.isfinite(fraction):
        return KIND_UNOBSERVED, np.nan, np.nan
    return status, fraction, reference_speed


def _trial_kind(trial_type: object) -> str:
    """Map a trial type tag to a baseline reference kind (visual wins over wind)."""
    ttype = str(trial_type).lower() if trial_type is not None else ""
    if "visual" in ttype or "looming" in ttype:
        return KIND_VISUAL
    if "wind" in ttype:
        return KIND_WIND
    return KIND_UNOBSERVED


def build_session_blocks(kin: pd.DataFrame) -> dict:
    """Per-session raw arrays (original row order) keyed by ``session_id``.

    Sessions are restricted FIRST and never sorted or bridged — a corrupt
    acquisition clock stays local to its session.
    """
    if kin is None or kin.empty:
        return {}
    if not {"sys_time", "ard_time", "dx", "dy", "stim_state"}.issubset(kin.columns):
        return {}
    factor = float(get_thresholds().stillness.max_frame_gap_factor)
    sys_t = pd.to_numeric(kin["sys_time"], errors="coerce").to_numpy(float)
    ard = pd.to_numeric(kin["ard_time"], errors="coerce").to_numpy(float)
    dx = pd.to_numeric(kin["dx"], errors="coerce").to_numpy(float)
    dy = pd.to_numeric(kin["dy"], errors="coerce").to_numpy(float)
    stim = pd.to_numeric(kin["stim_state"], errors="coerce").to_numpy(float)
    groups = (
        kin.groupby("session_id", sort=False).indices
        if "session_id" in kin.columns else {None: np.arange(len(kin))}
    )
    blocks: dict = {}
    for sid, pos in groups.items():
        idx = np.asarray(pos, dtype=int)
        session_ard = ard[idx]
        speed, gap = _session_speed_gap(session_ard, dx[idx], dy[idx], factor)
        blocks[sid] = {
            "sys_time": sys_t[idx],
            "ard_time": session_ard,
            "stim_state": stim[idx],
            "speed": speed,
            "max_gap": gap,
        }
    return blocks


def _measure_block(
    block: dict,
    kind: str,
    *,
    visual_onset_sys: float,
    t_start: float,
    t_stop: float,
    t_zero_sys: float,
    window_ms: float,
    width_ms: float,
    delay_ms: float,
) -> dict:
    """Resolve the reference sample on the raw session, then measure the history."""
    row = dict(_UNOBSERVED_ROW)
    row["prestim_reference_kind"] = kind
    sys_t, ard, stim = block["sys_time"], block["ard_time"], block["stim_state"]
    gap = block["max_gap"]
    if not np.isfinite(gap):
        return row

    if kind == KIND_VISUAL:
        if not np.isfinite(visual_onset_sys):
            return row
        cands = np.flatnonzero(np.isfinite(sys_t) & (sys_t <= visual_onset_sys))
        if not len(cands):
            return row
        mapped = int(cands[-1])
        ref_host_s = float(sys_t[mapped])
        sample_offset_ms = (float(visual_onset_sys) - ref_host_s) * 1000.0
        row["prestim_reference_sample_offset_ms"] = sample_offset_ms
        if sample_offset_ms > gap:
            # Remote mapping: no observed sample within one gap of the event.
            return row
        reference_ms = float(ard[mapped])
    else:  # KIND_WIND — first observed binary valve frame inside the trial window
        mask = np.isfinite(sys_t) & (sys_t >= t_start) & (sys_t <= t_stop) & (stim == 1.0)
        cands = np.flatnonzero(mask)
        if not len(cands):
            return row
        mapped = int(cands[0])
        ref_host_s = float(sys_t[mapped])
        # The calibrated arrival delay is part of the wind reference itself.
        reference_ms = float(ard[mapped]) + delay_ms

    if not np.isfinite(reference_ms):
        return row
    if np.isfinite(t_zero_sys):
        row["prestim_reference_offset_ms"] = (ref_host_s - t_zero_sys) * 1000.0 + (
            0.0 if kind == KIND_VISUAL else delay_ms
        )

    left, right = _component_bounds(ard, mapped, gap, reference_ms, window_ms, width_ms)
    times = ard[left:right + 1]
    speeds = block["speed"][left:right + 1]
    status, fraction, ref_speed = measure_prestim_baseline(times, speeds, reference_ms, window_ms)
    row["prestim_status"] = status
    row["prestim_moving_fraction"] = fraction
    row["prestim_reference_speed_mm_s"] = ref_speed
    return row


def measure_window_prestim(
    blocks: dict,
    window: dict,
    trial_type: object,
    t_zero_sys: float,
    cfg=None,
) -> dict:
    """Independent ``prestim_*`` row for one trial window.

    Additive only. A missing source (no block, no config, no stimulus event)
    stays ``unobserved`` — never silently substituted.
    """
    cfg = cfg or get_thresholds()
    window_ms = float(cfg.prewalk.window_ms)
    width_ms = float(cfg.stillness.speed_window_ms)
    delay_ms = float(cfg.reaction_time.wind_arrival_delay_ms)
    kind = _trial_kind(trial_type)
    row = dict(_UNOBSERVED_ROW)
    row["prestim_reference_kind"] = kind
    if kind == KIND_UNOBSERVED or not blocks:
        return row
    sid = window.get("session_id")
    block = blocks.get(sid) if sid in blocks else blocks.get(None)
    if block is None:
        return row
    if not np.all(np.isfinite([window_ms, width_ms, delay_ms])) or min(window_ms, width_ms) <= 0:
        return row
    return _measure_block(
        block, kind,
        visual_onset_sys=float(window.get("visual_onset_sys", np.nan)),
        t_start=float(window["t_start"]), t_stop=float(window["t_stop"]),
        t_zero_sys=float(t_zero_sys),
        window_ms=window_ms, width_ms=width_ms, delay_ms=delay_ms,
    )


__all__ = [
    "PRESTIM_COLUMNS",
    "KIND_VISUAL",
    "KIND_WIND",
    "KIND_UNOBSERVED",
    "measure_prestim_baseline",
    "build_session_blocks",
    "measure_window_prestim",
]
