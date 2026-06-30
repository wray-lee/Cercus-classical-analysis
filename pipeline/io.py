"""
Cercus Framework — Data I/O & Session Management
=================================================
CSV loading, cross-session auto-pairing, concatenation, and summary export.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import DETAILS_KEYS, LEGACY_TRIAL_DURATION_MS

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Low-Level CSV Parsing
# ══════════════════════════════════════════════════════════════════════


def _parse_details(raw: Any) -> dict:
    """Parse a single 'details' cell — may be JSON string, NaN, or dict."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    cleaned = raw.replace('\\"', '"').replace('""', '"').strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        log.debug("Failed to parse details JSON: %s", cleaned[:120])
        return {}


def load_events(path: str | Path) -> tuple[pd.DataFrame, list[dict], dict[Any, float]]:
    """
    Load events CSV. Extract trial time windows, metadata, and TTC anchors.

    Returns
    -------
    trial_meta : DataFrame
        One row per trial with ``global_trial_id`` + ``DETAILS_KEYS``.
    trial_windows : list of dict
        Each dict has keys: ``global_trial_id``, ``t_start``, ``t_stop``.
    ttc_anchors : dict
        Mapping ``global_trial_id`` → absolute system timestamp of ``Collision_TTC0``.
    """
    df = pd.read_csv(path)
    required = {"event_name", "timestamp", "global_trial_id", "details"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Events CSV missing columns: {missing}")

    starts = df[df["event_name"] == "trial_start"].copy()
    if starts.empty:
        log.warning("No 'trial_start' events found — stimulus metadata unavailable.")
        empty_meta = pd.DataFrame(columns=["global_trial_id", *DETAILS_KEYS])
        return empty_meta, [], {}

    stops = df[df["event_name"] == "trial_stop"].copy()

    stop_lookup: dict[Any, float] = {}
    if not stops.empty:
        stop_lookup = dict(zip(stops["global_trial_id"], stops["timestamp"]))

    trial_windows: list[dict] = []
    for _, row in starts.iterrows():
        tid = row["global_trial_id"]
        t_start = float(row["timestamp"])
        if tid in stop_lookup:
            t_stop = float(stop_lookup[tid])
        else:
            t_stop = t_start + (LEGACY_TRIAL_DURATION_MS / 1000.0)
        trial_windows.append({"global_trial_id": tid, "t_start": t_start, "t_stop": t_stop})

    ttc_anchors: dict[Any, float] = {}
    transitions = df[df["event_name"] == "phase_transition"].copy()
    for _, row in transitions.iterrows():
        details = _parse_details(row["details"])
        if details.get("to_phase") == "Collision_TTC0":
            tid = row["global_trial_id"]
            ttc_anchors[tid] = float(row["timestamp"])
    log.info("Extracted %d Collision_TTC0 anchors.", len(ttc_anchors))

    parsed = starts["details"].apply(_parse_details)
    details_df = pd.DataFrame(parsed.tolist(), index=starts.index)
    for k in DETAILS_KEYS:
        if k not in details_df.columns:
            details_df[k] = np.nan

    trial_meta = starts[["global_trial_id"]].join(details_df[list(DETAILS_KEYS)])
    trial_meta = trial_meta.drop_duplicates(subset="global_trial_id", keep="first")
    return trial_meta.reset_index(drop=True), trial_windows, ttc_anchors


def load_kinematics(path: str | Path) -> pd.DataFrame:
    """Load kinematics CSV."""
    df = pd.read_csv(path)
    required = {"sys_time", "dx", "dy", "dz", "stim_state", "global_trial_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kinematics CSV missing columns: {missing}")
    return df


# ══════════════════════════════════════════════════════════════════════
# Cross-Session Auto-Pairing & Concatenation
# ══════════════════════════════════════════════════════════════════════

_RE_EVENTS = re.compile(r"^(.+?)_session_(\d+)_events\.csv$", re.IGNORECASE)
_RE_KINEMATICS = re.compile(r"^(.+?)_session_(\d+)_kinematics\.csv$", re.IGNORECASE)


def scan_and_pair_sessions(input_dir: Path) -> dict[str, list[dict]]:
    """
    Scan *input_dir* for CSV files matching the naming convention and pair
    events ↔ kinematics by ``(subject, session_id)``.

    Returns
    -------
    subjects : dict
        ``{subject_name: [{"session_id": int, "events": Path, "kinematics": Path}, …]}``
        Sessions sorted ascending by *session_id*.
    """
    event_files: dict[tuple[str, int], Path] = {}
    kin_files: dict[tuple[str, int], Path] = {}

    for f in input_dir.iterdir():
        if not f.is_file():
            continue
        m = _RE_EVENTS.match(f.name)
        if m:
            event_files[(m.group(1), int(m.group(2)))] = f
            continue
        m = _RE_KINEMATICS.match(f.name)
        if m:
            kin_files[(m.group(1), int(m.group(2)))] = f

    paired_keys = set(event_files) & set(kin_files)
    orphans_ev = set(event_files) - paired_keys
    orphans_kin = set(kin_files) - paired_keys
    for key in orphans_ev:
        log.warning("Unpaired events file (no matching kinematics): %s", event_files[key].name)
    for key in orphans_kin:
        log.warning("Unpaired kinematics file (no matching events): %s", kin_files[key].name)

    subjects: dict[str, list[dict]] = {}
    for subject, sid in sorted(paired_keys):
        subjects.setdefault(subject, []).append(
            {"session_id": sid, "events": event_files[(subject, sid)], "kinematics": kin_files[(subject, sid)]}
        )

    for subject in subjects:
        subjects[subject].sort(key=lambda s: s["session_id"])

    log.info("Auto-paired %d sessions across %d subject(s).", len(paired_keys), len(subjects))
    for subj, sessions in subjects.items():
        log.info("  Subject '%s': %d session(s) → ids %s", subj, len(sessions), [s["session_id"] for s in sessions])

    return subjects


def load_and_concat_sessions(
    sessions: list[dict],
) -> tuple[pd.DataFrame, list[dict], dict[Any, float], pd.DataFrame, dict[Any, int]]:
    """
    Load all sessions for one subject, preprocess each, and concatenate into
    a single DataFrame with a continuous ``global_trial_index``.

    Returns
    -------
    all_meta : concatenated trial metadata
    all_windows : concatenated trial windows (with remapped global_trial_id)
    all_anchors : concatenated TTC anchors (remapped)
    all_kin : concatenated raw kinematics (remapped)
    trial_to_global : mapping ``(session_id, original_id)`` → new ``global_trial_index``
    """
    all_meta_parts: list[pd.DataFrame] = []
    all_windows: list[dict] = []
    all_anchors: dict[Any, float] = {}
    all_kin_parts: list[pd.DataFrame] = []

    global_index = 0
    trial_to_global: dict[Any, int] = {}

    for sess in sessions:
        sid = sess["session_id"]
        log.info("  Loading session %d: %s / %s", sid, sess["events"].name, sess["kinematics"].name)
        meta, windows, ttc_anchors = load_events(sess["events"])
        kin = load_kinematics(sess["kinematics"])

        session_tids = sorted(meta["global_trial_id"].unique()) if not meta.empty else []
        for tid in session_tids:
            global_index += 1
            trial_to_global[(sid, tid)] = global_index

        kin_only_tids = set(kin["global_trial_id"].unique()) - set(session_tids)
        for tid in sorted(kin_only_tids):
            global_index += 1
            trial_to_global[(sid, tid)] = global_index

        if not meta.empty:
            meta = meta.copy()
            meta["global_trial_id"] = meta["global_trial_id"].map(
                lambda t, _s=sid: trial_to_global.get((_s, t), t)
            )
            meta["session_id"] = sid
            all_meta_parts.append(meta)

        for w in windows:
            new_id = trial_to_global.get((sid, w["global_trial_id"]), w["global_trial_id"])
            all_windows.append({**w, "global_trial_id": new_id, "session_id": sid})

        for tid, ts in ttc_anchors.items():
            new_id = trial_to_global.get((sid, tid), tid)
            all_anchors[new_id] = ts

        kin = kin.copy()
        kin["global_trial_id"] = kin["global_trial_id"].map(
            lambda t, _s=sid: trial_to_global.get((_s, t), t)
        )
        kin["session_id"] = sid
        all_kin_parts.append(kin)

    all_meta = (
        pd.concat(all_meta_parts, ignore_index=True)
        if all_meta_parts
        else pd.DataFrame(columns=["global_trial_id", *DETAILS_KEYS])
    )
    all_kin = pd.concat(all_kin_parts, ignore_index=True) if all_kin_parts else pd.DataFrame()

    all_windows.sort(key=lambda w: w["global_trial_id"])

    log.info("  Concatenated: %d total trials, %d kinematics frames.", len(all_windows), len(all_kin))

    return all_meta, all_windows, all_anchors, all_kin, trial_to_global


# ══════════════════════════════════════════════════════════════════════
# Summary Metrics CSV Export
# ══════════════════════════════════════════════════════════════════════


def export_summary_metrics(
    df: pd.DataFrame,
    output_path: Path,
    groupby: str | list[str] = "global_trial_index",
) -> Path:
    """
    Export per-trial summary metrics to a CSV file for downstream statistical analysis.

    Parameters
    ----------
    df : DataFrame
        Fully labelled preprocessed DataFrame (after ``label_trials``).
    output_path : Path
        Full path for the output CSV file.
    groupby : str or list of str
        Column(s) to group by. Default ``"global_trial_index"``;
        pass ``["subject_id", "global_trial_index"]`` for population export.

    Returns
    -------
    Path to the written CSV file.
    """
    agg_spec: dict[str, tuple[str, str] | tuple[str, Any]] = {
        "global_trial_id": ("global_trial_id", "first"),
        "session_id": ("session_id", "first"),
        "type": ("type", "first"),
        "response_type": ("response_type", "first"),
        "latency_ms": ("latency_ms", "first"),
        "v_max": ("v_max", "first"),
    }
    for col in ("wind_dir", "screen_side", "direction", "side", "target_ttc_ms", "lv_ratio_ms", "init_half_angle_deg"):
        if col in df.columns:
            agg_spec[col] = (col, "first")

    trial_agg = df.groupby(groupby).agg(**agg_spec).reset_index()

    float_cols = ["latency_ms", "v_max", "target_ttc_ms", "lv_ratio_ms", "init_half_angle_deg"]
    for col in float_cols:
        if col in trial_agg.columns:
            trial_agg[col] = pd.to_numeric(trial_agg[col], errors="coerce")

    present_float_cols = [c for c in float_cols if c in trial_agg.columns]
    trial_agg[present_float_cols] = trial_agg[present_float_cols].round(2)

    sort_cols = groupby if isinstance(groupby, list) else [groupby]
    trial_agg = trial_agg.sort_values(sort_cols).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trial_agg.to_csv(output_path, index=False, float_format="%.2f")
    log.info("Summary metrics exported: %d trials → %s", len(trial_agg), output_path)
    return output_path
