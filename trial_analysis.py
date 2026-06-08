"""
Cercus Framework — Behavioral Neuroscience Data Analysis & Visualization
=========================================================================
Cross-session unified analysis pipeline. Scans a directory for per-session
event/kinematics CSV pairs, concatenates them by subject, and generates
publication-grade figures with physical-threshold escape classification.

Usage:
    python trial_analysis.py --input-dir path/to/data/ --save figures/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Publication-Grade Global Style (Nature / Science / Cell)
# ──────────────────────────────────────────────────────────────────────


def _apply_publication_style():
    """Inject Nature/Science/Cell compliant rcParams."""
    rc = plt.rcParams
    rc["font.family"] = "sans-serif"
    rc["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    rc["svg.fonttype"] = "none"
    rc["pdf.fonttype"] = 42
    rc["font.size"] = 7
    rc["axes.titlesize"] = 9
    rc["axes.labelsize"] = 8
    rc["legend.fontsize"] = 7
    rc["xtick.labelsize"] = 7
    rc["ytick.labelsize"] = 7
    rc["lines.linewidth"] = 1.0
    rc["axes.linewidth"] = 0.75
    rc["axes.spines.top"] = False
    rc["axes.spines.right"] = False
    rc["xtick.direction"] = "in"
    rc["ytick.direction"] = "in"
    rc["xtick.major.size"] = 3
    rc["ytick.major.size"] = 3
    rc["xtick.major.width"] = 0.75
    rc["ytick.major.width"] = 0.75
    rc["xtick.minor.size"] = 1.5
    rc["ytick.minor.size"] = 1.5
    rc["legend.frameon"] = False
    rc["legend.borderaxespad"] = 0
    rc["figure.dpi"] = 150
    rc["savefig.dpi"] = 300
    rc["savefig.transparent"] = True


# Lancet / Cell 风格（展示）
COLOR_LEFT = "#00468B"        # Navy Blue (深海军蓝)
COLOR_RIGHT = "#ED0000"       # Crimson Red (深绛红)
COLOR_CONTROL = "#7C878E"     # Slate Grey (石板灰)
COLOR_OSCI_VIS = "#ADB6B6"    # Cool Grey (冷灰底色)
# COLOR_OSCI_HW = "#F2B880"     # Sand Orange (沙橙色底色)
COLOR_OSCI_HW = "#E69F00"
COLOR_ESCAPE = "#ED0000"      
COLOR_PREWALK = "#00468B"     
COLOR_NO_RESPONSE = "#7C878E"

# Neuron 风格 （文章）
# COLOR_LEFT = "#008B8B"        # Dark Cyan / Teal (深青色，冷静且深邃)
# COLOR_RIGHT = "#E05A47"       # Coral Red (珊瑚红，醒目但不刺眼)
# COLOR_CONTROL = "#8A9A9A"     # Cool Ash Grey (冷灰，降低控制组的视觉存在感)
# COLOR_OSCI_VIS = "#A5C8C8"    # Pale Teal (极浅青色，用作底部示波器背景，防粘连)
# # COLOR_OSCI_HW = "#F4C4B7"     # Pale Coral (极浅珊瑚色，用作背景)
# COLOR_OSCI_HW = "#E69F00"
# COLOR_ESCAPE = "#E05A47"
# COLOR_PREWALK = "#008B8B"
# COLOR_NO_RESPONSE = "#8A9A9A"

_apply_publication_style()

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
DETAILS_KEYS = (
    "type",
    "target_ttc_ms",
    "wind_dir",
    "screen_side",
    "lv_ratio_ms",
    "init_half_angle_deg",
    "direction",
    "side",
)


def _get_unified_side(data: pd.Series | pd.DataFrame) -> str:
    """Extract and normalize direction identifier to 'left' or 'right'."""
    row = data.iloc[0] if isinstance(data, pd.DataFrame) else data
    for col in ["screen_side", "wind_dir", "direction", "side"]:
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip().lower()
            if val in ["left", "l"]: return "left"
            if val in ["right", "r"]: return "right"
    return ""
SPEED_WINDOW_MS = 100
SCALE_BAR_MM = 5.0
LEGACY_TRIAL_DURATION_MS = 5829.6
RADIUS_MM = 30.0

# ──────────────────────────────────────────────────────────────────────
# Physical-Threshold Escape Detection (Module 2)
# ──────────────────────────────────────────────────────────────────────
ESCAPE_VMAX_THRESHOLD = 50.0       # mm/s — absolute peak floor for valid escape response
ESCAPE_START_THRESHOLD = 10.0      # mm/s — latency onset anchor
PREWALK_THRESHOLD = 10.0           # mm/s — pre-stimulus spontaneous activity threshold
PREWALK_WINDOW_MS = 1000.0         # ms — pre-stimulus validation window (1 s)
POST_STIM_BUFFER_MS = 50.0         # ms — post-stimulus tail extension buffer
ESCAPE_WINDOW_MS = 250.0           # ms — post-stimulus burst detection window

TRAJECTORY_MAX_RADIUS_MM = 30.0      # mm — physical radius limit for trajectory plots
TRAJECTORY_STEP_MM = 5.0             # mm — step size for concentric rings in trajectory plots

def compute_escape_latency(t_rel: np.ndarray, speed: np.ndarray) -> dict:
    """
    Backward-search escape latency detection per literature standards.

    1. Baseline check: speed at t=0 must be < ESCAPE_START_THRESHOLD (10 mm/s).
    2. Burst check: max speed in [0, ESCAPE_WINDOW_MS] must be > ESCAPE_VMAX_THRESHOLD (50 mm/s).
    3. Latency: locate the first frame exceeding 50 mm/s within [0, ESCAPE_WINDOW_MS],
       then search backwards through the **full** time series (including negative time)
       for the last frame below 10 mm/s.  The frame immediately following is the true
       latency — which may be negative relative to stimulus onset.

    Returns dict: is_escaped (bool), v_max (float), latency_ms (float or NaN)
    """
    # ── Baseline check at t=0 ──
    zero_idx = int(np.argmin(np.abs(t_rel)))
    baseline_speed = speed[zero_idx]
    if not np.isnan(baseline_speed) and baseline_speed >= ESCAPE_START_THRESHOLD:
        return {"is_escaped": False, "v_max": np.nan, "latency_ms": np.nan}

    # ── Burst window [0, ESCAPE_WINDOW_MS] ──
    burst_mask = (t_rel >= 0) & (t_rel <= ESCAPE_WINDOW_MS)
    if not np.any(burst_mask):
        return {"is_escaped": False, "v_max": np.nan, "latency_ms": np.nan}

    burst_speed = speed[burst_mask]
    if np.all(np.isnan(burst_speed)):
        return {"is_escaped": False, "v_max": np.nan, "latency_ms": np.nan}
    v_max = float(np.nanmax(burst_speed))

    if np.isnan(v_max) or v_max <= ESCAPE_VMAX_THRESHOLD:
        return {"is_escaped": False, "v_max": v_max, "latency_ms": np.nan}

    # ── Backward-search latency (full-history, pierces t=0) ──
    # Global index of the first frame in [0, ESCAPE_WINDOW_MS] exceeding 50 mm/s
    burst_indices = np.where(burst_mask)[0]
    first_exceed_local = int(np.argmax(burst_speed > ESCAPE_VMAX_THRESHOLD))
    first_exceed_global = burst_indices[first_exceed_local]

    # Search backwards from this point through the ENTIRE array (including t < 0)
    search_speed = speed[:first_exceed_global + 1]
    below_mask = search_speed < ESCAPE_START_THRESHOLD

    if np.any(below_mask):
        last_below_idx = int(np.where(below_mask)[0][-1])
        latency_ms = float(t_rel[last_below_idx + 1])
    else:
        latency_ms = float(t_rel[0])

    return {"is_escaped": True, "v_max": v_max, "latency_ms": latency_ms}


# ══════════════════════════════════════════════════════════════════════
# Module 1 — Cross-Session Auto-Pairing & Concatenation
# ══════════════════════════════════════════════════════════════════════

# Regex: <subject>_session_<id>_events.csv / <subject>_session_<id>_kinematics.csv
_RE_EVENTS = re.compile(r"^(.+?)_session_(\d+)_events\.csv$", re.IGNORECASE)
_RE_KINEMATICS = re.compile(r"^(.+?)_session_(\d+)_kinematics\.csv$", re.IGNORECASE)


def scan_and_pair_sessions(input_dir: Path) -> dict[str, list[dict]]:
    """
    Scan *input_dir* for CSV files matching the naming convention and pair
    events ↔ kinematics by (subject, session_id).

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

    # Pair by (subject, session_id)
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

    # Sort sessions within each subject by session_id
    for subject in subjects:
        subjects[subject].sort(key=lambda s: s["session_id"])

    log.info("Auto-paired %d sessions across %d subject(s).", len(paired_keys), len(subjects))
    for subj, sessions in subjects.items():
        log.info("  Subject '%s': %d session(s) → ids %s", subj, len(sessions), [s["session_id"] for s in sessions])

    return subjects


def _load_single_session(events_path: Path, kinematics_path: Path) -> tuple[pd.DataFrame, list[dict], dict[Any, float], pd.DataFrame]:
    """Load one session's events + kinematics. Thin wrapper around existing loaders."""
    meta, trial_windows, ttc_anchors = load_events(events_path)
    kin = load_kinematics(kinematics_path)
    return meta, trial_windows, ttc_anchors, kin


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
    trial_to_global : mapping original global_trial_id → new global_trial_index
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
        meta, windows, ttc_anchors, kin = _load_single_session(sess["events"], sess["kinematics"])

        # Build global_trial_id → global_trial_index mapping for this session
        session_tids = sorted(meta["global_trial_id"].unique()) if not meta.empty else []
        for tid in session_tids:
            global_index += 1
            trial_to_global[(sid, tid)] = global_index

        # Also map any trial_ids that only appear in kinematics (no meta row)
        kin_only_tids = set(kin["global_trial_id"].unique()) - set(session_tids)
        for tid in sorted(kin_only_tids):
            global_index += 1
            trial_to_global[(sid, tid)] = global_index

        # Remap meta to ensure downstream preprocess matches correctly
        if not meta.empty:
            meta = meta.copy()
            meta["global_trial_id"] = meta["global_trial_id"].map(
                lambda t, _s=sid: trial_to_global.get((_s, t), t)
            )
            meta["session_id"] = sid
            all_meta_parts.append(meta)

        # Remap windows
        for w in windows:
            new_id = trial_to_global.get((sid, w["global_trial_id"]), w["global_trial_id"])
            all_windows.append({**w, "global_trial_id": new_id, "session_id": sid})

        # Remap anchors
        for tid, ts in ttc_anchors.items():
            new_id = trial_to_global.get((sid, tid), tid)
            all_anchors[new_id] = ts

        # Remap kinematics
        kin = kin.copy()
        kin["global_trial_id"] = kin["global_trial_id"].map(
            lambda t, _s=sid: trial_to_global.get((_s, t), t)
        )
        kin["session_id"] = sid
        all_kin_parts.append(kin)

    all_meta = pd.concat(all_meta_parts, ignore_index=True) if all_meta_parts else pd.DataFrame(columns=["global_trial_id", *DETAILS_KEYS])
    all_kin = pd.concat(all_kin_parts, ignore_index=True) if all_kin_parts else pd.DataFrame()

    # Sort windows by global_trial_id for consistent ordering
    all_windows.sort(key=lambda w: w["global_trial_id"])

    log.info("  Concatenated: %d total trials, %d kinematics frames.", len(all_windows), len(all_kin))

    return all_meta, all_windows, all_anchors, all_kin, trial_to_global


# ══════════════════════════════════════════════════════════════════════
# L0 — Data Loading (unchanged API)
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
        One row per trial with global_trial_id + DETAILS_KEYS.
    trial_windows : list of dict
        Each dict has keys: global_trial_id, t_start, t_stop.
    ttc_anchors : dict
        Mapping global_trial_id → absolute system timestamp of Collision_TTC0.
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
# L0 — Preprocessing: Timestamp-Based Slicing & Integration
# ══════════════════════════════════════════════════════════════════════


def _slice_kinematics_by_window(kin: pd.DataFrame, window: dict) -> pd.DataFrame:
    """Boolean-mask slice of kinematics for one trial using [t_start, t_stop]."""
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
    """
    Per-trial integration with lifecycle-anchored TTC time axis and denoised speed.
    """
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

    # 严格将 t_rel 逼近 0 的时刻锚定为空间原点
    zero_idx = int(np.argmin(np.abs(df["t_rel"].values)))
    df["x"] = x_smooth - x_smooth[zero_idx]
    df["y"] = y_smooth - y_smooth[zero_idx]

    dt_sec = df["sys_time"].diff().values
    dx_smooth = np.diff(x_smooth, prepend=x_smooth[0])
    dy_smooth = np.diff(y_smooth, prepend=y_smooth[0])
    if n > 1:
        dx_smooth[0] = x_smooth[1] - x_smooth[0]
        dy_smooth[0] = y_smooth[1] - y_smooth[0]
    speed = np.sqrt(dx_smooth**2 + dy_smooth**2) / dt_sec
    speed[0] = np.nan

    half_win = win // 2
    if half_win > 0:
        speed[:half_win] = np.nan
        speed[-half_win:] = np.nan

    df["speed"] = speed
    return df


def _compute_theoretical_ttc_ms(lv_ratio_ms: float, init_half_angle_deg: float) -> float:
    """Compute theoretical TTC (ms) from looming parameters for control trials."""
    import math

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

        if tid in ttc_anchors:
            t_zero_sys = ttc_anchors[tid]
        else:
            # Fallback: derive theoretical TTC from looming parameters
            lv_ratio = meta_row.get("lv_ratio_ms", pd.Series([np.nan])).iloc[0] if not meta_row.empty else np.nan
            init_angle = meta_row.get("init_half_angle_deg", pd.Series([np.nan])).iloc[0] if not meta_row.empty else np.nan

            if pd.notna(lv_ratio) and pd.notna(init_angle):
                t_col_ms = _compute_theoretical_ttc_ms(float(lv_ratio), float(init_angle))
                t_zero_sys = window["t_start"] + (t_col_ms / 1000.0)
                log.debug("Trial %s: no TTC anchor, fallback t_col_ms=%.1f ms", tid, t_col_ms)
            elif not wind_active.empty:
                # 纯风刺激/触觉刺激兜底：优先锚定硬件真实吹风的瞬间
                t_zero_sys = float(wind_active.iloc[0]["sys_time"])
                log.debug("Trial %s: aligned to hardware wind onset.", tid)
            else:
                # 极端兜底：既无视觉碰撞也无风刺激（如绝对空白对照）
                t_zero_sys = (window["t_start"] + window["t_stop"]) / 2.0
                log.warning("Trial %s: absolute baseline (no visual, no wind); using trial midpoint as zero.", tid)

        try:
            parts.append(_integrate_trial(trial_kin, t_zero_sys))
        except Exception as exc:
            log.warning("Skipping trial %s during integration: %s", tid, exc)

    if not parts:
        raise RuntimeError("No valid trials after preprocessing.")
    return pd.concat(parts, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════
# Module 2 — Physical-Threshold Ternary Classifier
# ══════════════════════════════════════════════════════════════════════


def _classify_trial(trial: pd.DataFrame) -> dict:
    """
    Ternary classification with backward-search latency detection.

    Priority order:
        1. PreWalk intercept  — pre-stimulus spontaneous activity vetoes the trial.
        2. Baseline check     — speed at t=0 must be < 10 mm/s.
        3. Burst threshold    — max speed in [0, 250] ms must be > 50 mm/s.
        4. Backward latency   — first frame > 50 mm/s, search back for last < 10 mm/s.
        5. NoResponse fallback.

    Returns dict with: response_type, v_max, latency_ms
    """
    if trial.empty:
        return {"response_type": "NoResponse", "v_max": np.nan, "latency_ms": np.nan}

    speed_vals = trial["speed"].values
    t_vals = trial["t_rel"].values

    # ── 1. PreWalk intercept: pre-stimulus spontaneous activity ──
    pre_mask = (t_vals >= -PREWALK_WINDOW_MS) & (t_vals < 0)
    if np.any(pre_mask):
        pre_slice = speed_vals[pre_mask]
        if not np.all(np.isnan(pre_slice)):
            pre_v_max = float(np.nanmax(pre_slice))
            if pre_v_max > PREWALK_THRESHOLD:
                return {"response_type": "PreWalk", "v_max": pre_v_max, "latency_ms": np.nan}

    # ── 2–4. Escape detection via backward-search latency ──
    result = compute_escape_latency(t_vals, speed_vals)

    if result["is_escaped"]:
        return {"response_type": "Escape", "v_max": result["v_max"], "latency_ms": result["latency_ms"]}

    return {"response_type": "NoResponse", "v_max": result["v_max"], "latency_ms": np.nan}


def _label_trials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``response_type``, ``v_max``, and ``latency_ms`` columns to a preprocessed DataFrame.
    Replaces the old binary ``is_escaped`` labeling.
    """
    classify_map: dict = {}
    v_max_map: dict = {}
    latency_map: dict = {}

    for tid, grp in df.groupby("global_trial_id"):
        result = _classify_trial(grp)
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


# ══════════════════════════════════════════════════════════════════════
# Plot 1 — Trajectory Overlay
# ══════════════════════════════════════════════════════════════════════


def _draw_standardized_grid(ax: plt.Axes, max_radius: float = 50.0, step: float = 10.0):
    """绘制统一尺度的物理坐标系与同心距离环"""
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 锁定物理坐标范围
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_aspect("equal")

    # 绘制原点十字准星
    ax.axhline(0, color="black", lw=0.6, alpha=0.5, zorder=1)
    ax.axvline(0, color="black", lw=0.6, alpha=0.5, zorder=1)

    # 绘制同心距离环与标尺文本
    for r in np.arange(step, max_radius + step, step):
        circle = plt.Circle((0, 0), r, color="gray", fill=False, ls="--", lw=0.5, alpha=0.5, zorder=1)
        ax.add_patch(circle)
        ax.text(r * 0.707, r * 0.707, f"{int(r)} mm", color="gray", fontsize=6,
                ha="left", va="bottom", alpha=0.8)

    ax.set_xticks([])
    ax.set_yticks([])


def _draw_side_arrows(ax: plt.Axes, left_color: str = COLOR_LEFT, right_color: str = COLOR_RIGHT):
    """Draw minimalist vector arrows on LEFT and RIGHT edges."""
    arrow_style = dict(arrowstyle="-|>", color=None, lw=1.0, mutation_scale=8)

    ax.annotate("", xy=(0.04, 0.5), xytext=(-0.02, 0.5),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops={**arrow_style, "color": left_color})
    ax.text(0.01, 0.45, "Left Stimulus", transform=ax.transAxes,
            ha="center", va="top", fontsize=7, color=left_color)

    ax.annotate("", xy=(0.96, 0.5), xytext=(1.02, 0.5),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops={**arrow_style, "color": right_color})
    ax.text(0.99, 0.45, "Right Stimulus", transform=ax.transAxes,
            ha="center", va="top", fontsize=7, color=right_color)


def plot_trajectory_overlay(
    df: pd.DataFrame,
    control_type: str = "baseline_visual",
    left_color: str = COLOR_LEFT,
    right_color: str = COLOR_RIGHT,
    figsize_per_ax: tuple[float, float] = (4.0, 4.0),
):
    """
    One subplot per trial type. Left stimuli in NPG blue, right in NPG red.

    Each trial is aligned so the physical position at the backward-search
    latency onset is centered at (0, 0).  The burst window extends from
    latency_ms to +500 ms or until speed drops below the resting threshold.
    """
    all_types = sorted(df["type"].dropna().unique())
    if not all_types:
        log.warning("No trial types found for trajectory overlay.")
        fig, _ = plt.subplots(figsize=figsize_per_ax)
        return fig

    n = len(all_types)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_ax[0] * n, figsize_per_ax[1]), squeeze=False)
    axes = axes[0]

    for idx, ttype in enumerate(all_types):
        ax = axes[idx]
        subset = df[df["type"] == ttype]

        for _tid, grp in subset.groupby("global_trial_id"):
            grp = grp.sort_values("t_rel")
            t_vals = grp["t_rel"].values
            speed_vals = grp["speed"].values

            # Backward-search latency for this trial
            esc = compute_escape_latency(t_vals, speed_vals)

            # ── Determine render window ──
            if esc["is_escaped"] and not np.isnan(esc["latency_ms"]):
                # Escape trial: expand 250 ms before latency to capture pre-sprint pivot
                latency_ms = esc["latency_ms"]
                render_start_ms = latency_ms - 250.0
                render_start_idx = int(np.argmin(np.abs(t_vals - render_start_ms)))
                actual_start_ms = t_vals[render_start_idx]
                burst_end_ms = latency_ms + 500.0
            else:
                # NoResponse fallback: standard baseline window [0, 500] ms
                render_start_idx = int(np.argmin(np.abs(t_vals - 0.0)))
                actual_start_ms = 0.0
                burst_end_ms = 500.0

            burst_mask = (t_vals >= actual_start_ms) & (t_vals <= burst_end_ms)

            # Trim at first sub-threshold drop after the burst peak
            if np.any(burst_mask):
                burst_speed = speed_vals[burst_mask]
                burst_t = t_vals[burst_mask]
                if np.all(np.isnan(burst_speed)):
                    continue
                peak_in_burst = int(np.nanargmax(burst_speed))
                post_peak_speed = burst_speed[peak_in_burst:]
                post_peak_t = burst_t[peak_in_burst:]
                below_rest = post_peak_speed < ESCAPE_START_THRESHOLD
                if np.any(below_rest):
                    rest_idx = int(np.argmax(below_rest))
                    actual_end_ms = post_peak_t[rest_idx]
                    burst_mask = (t_vals >= actual_start_ms) & (t_vals <= actual_end_ms)

            if not np.any(burst_mask):
                continue

            burst = grp[burst_mask]

            # ── 2. Dynamic Origin Translation ──
            # Center at the first frame of the expanded burst
            x_origin = burst["x"].iloc[0]
            y_origin = burst["y"].iloc[0]
            burst_x = burst["x"].values - x_origin
            burst_y = burst["y"].values - y_origin

            # ── 3. Dynamic Vector Alignment (Reverse Rotation) ──
            # Rotate so the heading at the render start frame points North (0 rad)
            raw_heading = grp["dz"].cumsum().values / RADIUS_MM
            theta = -raw_heading[render_start_idx]
            rot_x = burst_x * np.cos(theta) - burst_y * np.sin(theta)
            rot_y = burst_x * np.sin(theta) + burst_y * np.cos(theta)

            # ── 4. Color & Plot ──
            ss = _get_unified_side(burst)
            if ttype == control_type:
                color = COLOR_CONTROL
            elif ss == "left":
                color = left_color
            elif ss == "right":
                color = right_color
            else:
                color = COLOR_CONTROL

            ax.plot(rot_x, rot_y, color=color, alpha=0.4, lw=0.8)

        ax.set_title(ttype, fontweight="bold")
        _draw_standardized_grid(ax, max_radius=TRAJECTORY_MAX_RADIUS_MM, step=TRAJECTORY_STEP_MM)
        _draw_side_arrows(ax, left_color=left_color, right_color=right_color)

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Plot 2 — Speed Kinetics with Dual-Threshold Lines
# ══════════════════════════════════════════════════════════════════════


def _add_threshold_lines(ax: plt.Axes):
    """Draw horizontal threshold lines at ESCAPE_VMAX (50 mm/s) and ESCAPE_START (10 mm/s)."""
    ax.axhline(y=ESCAPE_VMAX_THRESHOLD, color="k", linestyle="--", linewidth=0.75, alpha=0.7)
    ax.text(ax.get_xlim()[1] * 0.98, ESCAPE_VMAX_THRESHOLD + 1.0,
            f"Vmax Threshold ({ESCAPE_VMAX_THRESHOLD:.0f})", ha="right", va="bottom", fontsize=6, color="k", alpha=0.7)

    ax.axhline(y=ESCAPE_START_THRESHOLD, color="0.5", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.text(ax.get_xlim()[1] * 0.98, ESCAPE_START_THRESHOLD + 1.0,
            f"Start Threshold ({ESCAPE_START_THRESHOLD:.0f})", ha="right", va="bottom", fontsize=6, color="0.5", alpha=0.5)


def plot_speed_kinetics(
    df: pd.DataFrame,
    control_type: str = "baseline_visual",
    stim_type: str = "looming_wind",
    figsize: tuple[float, float] = (10, 6),
):
    """
    Two-panel figure (4:1 height ratio) with shared X axis.
    Upper panel: Escape speed (mean ± SEM) per condition + threshold lines.
    Lower panel: Oscilloscope-style dual-channel waveforms.
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.08)
    ax_main = fig.add_subplot(gs[0])
    ax_stim = fig.add_subplot(gs[1], sharex=ax_main)

    t_bin = 5.0
    t_min = df["t_rel"].min()
    t_max = df["t_rel"].max()
    bins = np.arange(t_min, t_max + t_bin, t_bin)
    df_binned = df.copy()
    df_binned["t_bin"] = pd.cut(df_binned["t_rel"], bins=bins, labels=bins[:-1], include_lowest=True)
    df_binned["t_bin"] = df_binned["t_bin"].astype(float)

    cond_colors = {}
    for ttype in df["type"].dropna().unique():
        if ttype == control_type:
            cond_colors[ttype] = COLOR_CONTROL
        else:
            sample = df[df["type"] == ttype].iloc[0]
            ss = _get_unified_side(sample)
            cond_colors[ttype] = COLOR_LEFT if ss == "left" else COLOR_RIGHT if ss == "right" else COLOR_LEFT

    for cond, color in cond_colors.items():
        subset = df_binned[df_binned["type"] == cond]
        if subset.empty:
            continue
        trial_means = subset.groupby(["global_trial_id", "t_bin"])["speed"].mean().reset_index()
        agg = trial_means.groupby("t_bin")["speed"]
        mean = agg.mean()
        sem = agg.sem().fillna(0)
        t_vals = mean.index.values
        ax_main.plot(t_vals, mean.values, color=color, lw=1.0, label=cond)
        ax_main.fill_between(t_vals, (mean - sem).values, (mean + sem).values,
                             color=color, alpha=0.2, edgecolor="none")

    ax_main.set_ylabel("Escape Speed (mm/s)")
    ax_main.legend(loc="upper right", frameon=False)
    ax_main.set_xlabel("")
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # Threshold lines
    _add_threshold_lines(ax_main)

    # ── Lower panel: oscilloscope waveforms (Dynamic multi-condition render) ──
    vis_baseline = 1.0
    wind_baseline = 3.0

    drawn_vis = False
    drawn_wind = False

    for cond in df["type"].dropna().unique():
        subset = df[df["type"] == cond]
        if subset.empty:
            continue

        # Channel 1: Visual looming — 动态感知当前存在的视觉条件
        if ("visual" in cond.lower() or "looming" in cond.lower()) and not drawn_vis:
            stim_t_rel = subset["t_rel"]
            t_loom_start = stim_t_rel.min() if not stim_t_rel.empty else df["t_rel"].min()
            t_loom = np.array([t_loom_start, 0.0])
            ax_stim.fill_between(
                t_loom, vis_baseline, vis_baseline + 1.0, step="mid",
                color=COLOR_OSCI_VIS, alpha=0.6, label="Visual (looming)"
            )
            drawn_vis = True  # 避免多条件叠加导致颜色加深及图例重复

        # Channel 2: Wind stim_state — 动态感知当前存在的硬件风控
        first_tid = subset["global_trial_id"].iloc[0]
        grp = subset[subset["global_trial_id"] == first_tid].sort_values("t_rel")

        if ("wind" in cond.lower() or "puff" in cond.lower() or grp["stim_state"].max() > 0) and not drawn_wind:
            t_wind = grp["t_rel"].values
            stim = grp["stim_state"].values.astype(float)

            if len(t_wind) > 1:
                dt_last = t_wind[-1] - t_wind[-2]
            else:
                dt_last = 1.0
            t_wind_ext = np.append(t_wind, t_wind[-1] + dt_last)
            stim_ext = np.append(stim, stim[-1])

            ax_stim.fill_between(
                t_wind_ext, wind_baseline, wind_baseline + stim_ext, step="post",
                color=COLOR_OSCI_HW, alpha=0.6, label="Wind (stim_state)"
            )
            drawn_wind = True

    ax_stim.set_ylim(0, 5)
    ax_stim.set_yticks([])
    ax_stim.set_ylabel("")
    ax_stim.set_xlabel("Time relative to TTC (ms)")
    ax_stim.legend(loc="upper right", frameon=False, ncol=2)
    ax_stim.grid(False)

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Plot 3 — Spaghetti Kinetics with Dual-Threshold Lines
# ══════════════════════════════════════════════════════════════════════


def plot_spaghetti_kinetics(
    df: pd.DataFrame,
    control_type: str = "baseline_visual",
    stim_type: str = "looming_wind",
    figsize_per_col: float = 4.5,
    row_height: float = 5.0,
):
    """
    Multi-panel spaghetti plot with per-condition spatial decoupling.
    Threshold lines drawn on each upper panel.
    """
    cond_color_map = {}
    for ttype in df["type"].dropna().unique():
        if ttype == control_type:
            cond_color_map[ttype] = COLOR_CONTROL
        else:
            sample = df[df["type"] == ttype].iloc[0]
            ss = _get_unified_side(sample)
            cond_color_map[ttype] = COLOR_LEFT if ss == "left" else COLOR_RIGHT if ss == "right" else COLOR_LEFT

    conditions = sorted(cond_color_map.keys())
    n_conds = len(conditions)
    if n_conds == 0:
        fig, ax = plt.subplots()
        return fig

    fig = plt.figure(figsize=(figsize_per_col * n_conds, row_height * 2))
    gs = gridspec.GridSpec(2, n_conds, height_ratios=[4, 1], hspace=0.1, wspace=0.15, figure=fig)

    ax_upper = []
    ax_lower = []
    for j in range(n_conds):
        sharey = ax_upper[0] if ax_upper else None
        ax_u = fig.add_subplot(gs[0, j], sharey=sharey)
        ax_upper.append(ax_u)
        ax_l = fig.add_subplot(gs[1, j], sharex=ax_u)
        ax_lower.append(ax_l)

    for j, cond in enumerate(conditions):
        ax = ax_upper[j]
        subset = df[df["type"] == cond]
        cond_color = cond_color_map[cond]

        if subset.empty:
            ax.set_title(cond, fontweight="bold")
            continue

        for _tid, grp in subset.groupby("global_trial_id"):
            grp_sorted = grp.sort_values("t_rel")
            # 统一使用条件主色，降维线宽至 0.5，透明度至 0.25 形成背景数据云
            ax.plot(grp_sorted["t_rel"], grp_sorted["speed"], color=cond_color, lw=0.5, alpha=0.25)

        t_bin = 5.0
        t_min = subset["t_rel"].min()
        t_max = subset["t_rel"].max()
        bins = np.arange(t_min, t_max + t_bin, t_bin)
        binned = subset.copy()
        binned["t_bin"] = pd.cut(binned["t_rel"], bins=bins, labels=bins[:-1], include_lowest=True)
        binned["t_bin"] = binned["t_bin"].astype(float)

        trial_means = binned.groupby(["global_trial_id", "t_bin"])["speed"].mean().reset_index()
        agg = trial_means.groupby("t_bin")["speed"]
        mean = agg.mean()
        sem = agg.sem().fillna(0)
        t_vals = mean.index.values

        ax.plot(t_vals, mean.values, color="white", lw=4.0, alpha=0.8, solid_capstyle="round")
        ax.plot(t_vals, mean.values, color=cond_color, lw=2.0, alpha=1.0, label=cond)
        ax.fill_between(t_vals, (mean - sem).values, (mean + sem).values,
                        color=cond_color, alpha=0.2, edgecolor="none")

        ax.set_title(cond, fontweight="bold")
        if j == 0:
            ax.set_ylabel("Escape Speed (mm/s)")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.legend(loc="upper right", frameon=False)

        # Threshold lines
        _add_threshold_lines(ax)

    # ── Row 1: Oscilloscope channels ──
    for j, cond in enumerate(conditions):
        ax = ax_lower[j]
        subset = df[df["type"] == cond]
        vis_baseline = 1.0
        wind_baseline = 3.0

        # Channel 1: Visual looming — 仅在条件包含 visual 或 looming 时绘制
        if "visual" in cond.lower() or "looming" in cond.lower():
            stim_t_rel = subset["t_rel"]
            t_loom_start = stim_t_rel.min() if not stim_t_rel.empty else df["t_rel"].min()
            t_loom = np.array([t_loom_start, 0.0])
            ax.fill_between(t_loom, vis_baseline, vis_baseline + 1.0, step="mid",
                            color=COLOR_OSCI_VIS, alpha=0.6, label="Visual (looming)")

        # Channel 2: Wind stim_state — 仅在包含 wind/puff 或硬件触发值大于0时绘制
        if not subset.empty:
            first_tid = subset["global_trial_id"].iloc[0]
            grp = subset[subset["global_trial_id"] == first_tid].sort_values("t_rel")

            if "wind" in cond.lower() or "puff" in cond.lower() or grp["stim_state"].max() > 0:
                t_wind = grp["t_rel"].values
                stim = grp["stim_state"].values.astype(float)

                if len(t_wind) > 1:
                    dt_last = t_wind[-1] - t_wind[-2]
                else:
                    dt_last = 1.0
                t_wind_ext = np.append(t_wind, t_wind[-1] + dt_last)
                stim_ext = np.append(stim, stim[-1])

                ax.fill_between(t_wind_ext, wind_baseline, wind_baseline + stim_ext,
                                step="post", color=COLOR_OSCI_HW, alpha=0.6, label="Wind (stim_state)")

        ax.set_ylim(0, 5)
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("Time relative to TTC (ms)")
        ax.grid(False)
        if j == 0:
            ax.legend(loc="upper right", frameon=False, ncol=2)

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Module 3 — Behavior Probability Distribution
# ══════════════════════════════════════════════════════════════════════


def plot_behavior_probability(df: pd.DataFrame) -> plt.Figure:
    """
    Bar chart showing the proportion of Escape / PreWalk / NoResponse across
    all trials for this subject.

    Returns the matplotlib Figure.
    """
    counts = df.groupby("global_trial_id")["response_type"].first().value_counts()
    total = counts.sum()

    categories = ["Escape", "PreWalk", "NoResponse"]
    values = [counts.get(c, 0) / total if total > 0 else 0.0 for c in categories]
    colors = [COLOR_ESCAPE, COLOR_PREWALK, COLOR_NO_RESPONSE]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    bars = ax.bar(categories, values, color=colors, width=0.55, edgecolor="none", alpha=0.85)

    for bar, val in zip(bars, values):
        if val > 0.02:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.05)
    ax.set_title("Behavior Probability Distribution", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Module 4 — Habituation Decay Curve
# ══════════════════════════════════════════════════════════════════════


def plot_habituation_curve(df: pd.DataFrame) -> plt.Figure:
    """
    Scatter + line plot of V_max per trial across the global trial sequence,
    showing habituation (neural desensitisation) slope.

    X-axis: global_trial_index (1 … N)
    Y-axis: V_max (mm/s)
    Horizontal anchors at ESCAPE_VMAX_THRESHOLD (50 mm/s) and ESCAPE_START_THRESHOLD (10 mm/s).

    Returns the matplotlib Figure.
    """
    # Aggregate: one row per trial
    trial_agg = df.groupby("global_trial_index").agg(
        v_max=("v_max", "first"),
        response_type=("response_type", "first"),
    ).reset_index()
    trial_agg = trial_agg.sort_values("global_trial_index")

    x = trial_agg["global_trial_index"].values
    y = trial_agg["v_max"].values

    # Color each point by its response_type
    color_map = {"Escape": COLOR_ESCAPE, "PreWalk": COLOR_PREWALK, "NoResponse": COLOR_NO_RESPONSE}
    point_colors = [color_map.get(rt, COLOR_NO_RESPONSE) for rt in trial_agg["response_type"].values]

    fig, ax = plt.subplots(figsize=(8, 3.5))

    # Line underneath
    ax.plot(x, y, color="0.7", lw=0.8, alpha=0.6, zorder=1)
    # Scatter on top
    ax.scatter(x, y, c=point_colors, s=28, edgecolors="white", linewidths=0.4, zorder=2)

    # Threshold anchors
    ax.axhline(y=ESCAPE_VMAX_THRESHOLD, color="k", linestyle="--", linewidth=0.75, alpha=0.7)
    ax.text(x[-1] + 0.3, ESCAPE_VMAX_THRESHOLD, f"{ESCAPE_VMAX_THRESHOLD:.0f}", ha="left", va="center", fontsize=6, color="k", alpha=0.7)

    ax.axhline(y=ESCAPE_START_THRESHOLD, color="0.5", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.text(x[-1] + 0.3, ESCAPE_START_THRESHOLD, f"{ESCAPE_START_THRESHOLD:.0f}", ha="left", va="center", fontsize=6, color="0.5", alpha=0.5)

    ax.set_xlabel("Global Trial Index")
    ax.set_ylabel("V$_{max}$ (mm/s)")
    ax.set_title("Habituation Curve", fontweight="bold")
    ax.set_xlim(0.5, len(x) + 0.5)

    # Minimalist legend for response types
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=5, label=l)
               for l, c in [("Escape", COLOR_ESCAPE), ("PreWalk", COLOR_PREWALK), ("NoResponse", COLOR_NO_RESPONSE)]]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=6)

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Diagnostic — V_max Frequency Distribution (Bimodal Threshold Tuning)
# ══════════════════════════════════════════════════════════════════════


def plot_vmax_distribution(df: pd.DataFrame, figsize: tuple[float, float] = (5.0, 3.5)) -> plt.Figure:
    """
    Histogram + KDE of per-trial V_max for Escape trials.

    Used to visually verify the escape threshold placement relative to the
    observed V_max distribution.

    Returns the matplotlib Figure.
    """
    # Only keep Escape trials
    df_resp = df[df["response_type"] == "Escape"].copy()
    if df_resp.empty:
        log.warning("No Escape trials for V_max distribution plot.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No Escape trials", ha="center", va="center", transform=ax.transAxes)
        return fig

    # One v_max per trial (first value per global_trial_id is identical)
    trial_vmax = df_resp.groupby("global_trial_id")["v_max"].first().dropna().values

    if len(trial_vmax) == 0:
        log.warning("All V_max values are NaN — skipping distribution plot.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No valid V_max", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    # ── Histogram (density-normalised) ──
    bins = np.linspace(0, 400, 51)  # 50 bins over 0–400 mm/s
    ax.hist(trial_vmax, bins=bins, density=True, color="#D0D0D0", edgecolor="white",
            linewidth=0.4, alpha=0.85, label="Histogram", zorder=2)

    # ── KDE overlay ──
    kde = gaussian_kde(trial_vmax, bw_method="scott")
    x_kde = np.linspace(0, 400, 500)
    ax.plot(x_kde, kde(x_kde), color="black", lw=1.2, alpha=0.9, label="KDE", zorder=3)

    # ── Threshold reference lines ──
    ax.axvline(ESCAPE_VMAX_THRESHOLD, color="black", ls="--", lw=0.75, alpha=0.7, zorder=4)
    ax.text(ESCAPE_VMAX_THRESHOLD + 3, ax.get_ylim()[1] * 0.92,
            f"Vmax\n{ESCAPE_VMAX_THRESHOLD:.0f}", fontsize=6, color="black", va="top")

    ax.axvline(ESCAPE_START_THRESHOLD, color="0.5", ls="--", lw=0.75, alpha=0.7, zorder=4)
    ax.text(ESCAPE_START_THRESHOLD + 3, ax.get_ylim()[1] * 0.92,
            f"Start\n{ESCAPE_START_THRESHOLD:.0f}", fontsize=6, color="0.5", va="top")

    ax.set_xlim(0, 400)
    ax.set_xlabel("$V_{max}$ (mm/s)")
    ax.set_ylabel("Probability Density")
    ax.set_title("$V_{max}$ Distribution — Threshold Diagnostic", fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=6)

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Module 5 — Individual Escape Trial Export
# ══════════════════════════════════════════════════════════════════════


def plot_single_escape_trial(
    trial: pd.DataFrame,
    latency_ms: float,
    v_max: float,
    global_trial_index: int,
    figsize: tuple[float, float] = (6.0, 3.5),
) -> plt.Figure:
    """
    Speed kinetics for a single Escape trial with latency marker and threshold lines.

    Parameters
    ----------
    trial : DataFrame
        Kinematics rows for one trial (must contain ``t_rel`` and ``speed``).
    latency_ms : float
        System-computed response latency (ms).
    v_max : float
        Peak speed (mm/s) within the escape window.
    global_trial_index : int
        Trial index used in the figure title and filename.
    figsize : tuple
        Figure dimensions.

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    t = trial["t_rel"].values
    spd = trial["speed"].values

    # ── Speed curve ──
    ax.plot(t, spd, color=COLOR_ESCAPE, lw=1.0, alpha=0.85, label="Speed")

    # ── Latency marker ──
    if not np.isnan(latency_ms):
        ax.axvline(x=latency_ms, color="#3C5488", ls="--", lw=0.9, alpha=0.9, label=f"Latency = {latency_ms:.1f} ms")
        # Scatter at the latency point — interpolate speed at that time
        lat_idx = np.argmin(np.abs(t - latency_ms))
        ax.scatter([latency_ms], [spd[lat_idx]], c="#3C5488", s=30, zorder=5, edgecolors="white", linewidths=0.5)

    # ── Threshold reference lines ──
    ax.axhline(y=ESCAPE_VMAX_THRESHOLD, color="k", linestyle="--", linewidth=0.75, alpha=0.6)
    ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02,
            ESCAPE_VMAX_THRESHOLD + 1.5,
            f"Vmax ({ESCAPE_VMAX_THRESHOLD:.0f})", fontsize=6, color="k", alpha=0.6)

    ax.axhline(y=ESCAPE_START_THRESHOLD, color="0.5", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02,
            ESCAPE_START_THRESHOLD + 1.5,
            f"Start ({ESCAPE_START_THRESHOLD:.0f})", fontsize=6, color="0.5", alpha=0.4)

    # ── Labels & annotation ──
    ax.set_xlabel("Time relative to TTC (ms)")
    ax.set_ylabel("Escape Speed (mm/s)")
    ax.set_title(f"Trial {global_trial_index} — Escape  (V$_{{max}}$={v_max:.1f} mm/s)", fontweight="bold")

    # Text box with key metrics
    info_text = f"Latency: {latency_ms:.1f} ms\nV$_{{max}}$: {v_max:.1f} mm/s"
    ax.text(0.98, 0.95, info_text, transform=ax.transAxes, fontsize=6,
            ha="right", va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.9))

    ax.legend(loc="upper left", frameon=False, fontsize=6)
    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# CLI Entry Point — Module 1: --input-dir
# ══════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cercus framework — cross-session behavioral analysis with ternary classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-dir", required=True,
        help="Directory containing *_session_*_events.csv and *_session_*_kinematics.csv files.",
    )
    p.add_argument("--control-type", default="baseline_visual", help="Trial type for control condition.")
    p.add_argument("--stim-type", default="looming_wind", help="Trial type for stimulus condition.")
    p.add_argument("--save", default=None, help="Directory to save PNG figures. Omit to show interactively.")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"--input-dir does not exist: {input_dir}")

    # ── Module 1: scan, pair, concatenate ──
    subjects = scan_and_pair_sessions(input_dir)
    if not subjects:
        log.error("No valid (events, kinematics) pairs found in %s", input_dir)
        return

    for subject_name, sessions in subjects.items():
        log.info("═══ Processing subject: %s ═══", subject_name)

        all_meta, all_windows, all_anchors, all_kin, trial_to_global = load_and_concat_sessions(sessions)

        # ── Preprocess with global_trial_id (which now IS the global_trial_index) ──
        df = preprocess(all_meta, all_windows, all_anchors, all_kin)

        # Inject global_trial_index into the preprocessed DataFrame
        # The windows were remapped so global_trial_id == global_trial_index
        df["global_trial_index"] = df["global_trial_id"]

        # ── Module 2: ternary classification ──
        df = _label_trials(df)

        types_present = set(df["type"].dropna().unique())
        log.info("Trial types in data: %s", types_present)

        # ── Module 3: output routing ──
        df_response = df[df["response_type"] == "Escape"].copy()
        df_no_response = df[df["response_type"] == "NoResponse"].copy()

        n_resp = df_response["global_trial_id"].nunique()
        n_nr = df_no_response["global_trial_id"].nunique()
        log.info("Split: %d Escape, %d NoResponse", n_resp, n_nr)

        # Peak diagnostic for discarded trials
        if not df_no_response.empty:
            diag_window = df_no_response[(df_no_response["t_rel"] > 0) & (df_no_response["t_rel"] <= POST_STIM_BUFFER_MS)]
            if not diag_window.empty:
                peaks = diag_window.groupby("global_trial_id")["speed"].max()
                log.info("====== NoResponse peak diagnostic (0–%.0f ms) ======", POST_STIM_BUFFER_MS)
                for tid, pmax in peaks.items():
                    log.info("  Trial %s peak: %.1f mm/s", tid, pmax)
                log.info("  Mean peak: %.1f mm/s", peaks.mean())
                log.info("════════════════════════════════════════════════")

        if args.save:
            out = Path(args.save)
            subject_dir = out / subject_name

            # ── Response figures ──
            if not df_response.empty:
                resp_dir = subject_dir / "response"
                resp_dir.mkdir(parents=True, exist_ok=True)

                fig_traj = plot_trajectory_overlay(df_response, control_type=args.control_type)
                fig_traj.savefig(resp_dir / "trajectory_overlay.png", dpi=300, bbox_inches="tight")
                plt.close(fig_traj)

                fig_speed = plot_speed_kinetics(df_response, control_type=args.control_type, stim_type=args.stim_type)
                fig_speed.savefig(resp_dir / "speed_kinetics.png", dpi=300, bbox_inches="tight")
                plt.close(fig_speed)

                fig_spaghetti = plot_spaghetti_kinetics(df_response, control_type=args.control_type, stim_type=args.stim_type)
                fig_spaghetti.savefig(resp_dir / "spaghetti_kinetics.png", dpi=300, bbox_inches="tight")
                plt.close(fig_spaghetti)

                log.info("Response figures saved to %s", resp_dir)

            # ── No-response figures ──
            if not df_no_response.empty:
                nr_dir = subject_dir / "no_response"
                nr_dir.mkdir(parents=True, exist_ok=True)

                fig_traj_nr = plot_trajectory_overlay(df_no_response, control_type=args.control_type)
                fig_traj_nr.savefig(nr_dir / "trajectory_overlay.png", dpi=300, bbox_inches="tight")
                plt.close(fig_traj_nr)

                fig_speed_nr = plot_speed_kinetics(df_no_response, control_type=args.control_type, stim_type=args.stim_type)
                fig_speed_nr.savefig(nr_dir / "speed_kinetics.png", dpi=300, bbox_inches="tight")
                plt.close(fig_speed_nr)

                fig_spaghetti_nr = plot_spaghetti_kinetics(df_no_response, control_type=args.control_type, stim_type=args.stim_type)
                fig_spaghetti_nr.savefig(nr_dir / "spaghetti_kinetics.png", dpi=300, bbox_inches="tight")
                plt.close(fig_spaghetti_nr)

                log.info("No-response figures saved to %s", nr_dir)

            # ── Module 3: Behavior probability distribution ──
            fig_prob = plot_behavior_probability(df)
            fig_prob.savefig(subject_dir / "behavior_probability_distribution.png", dpi=300, bbox_inches="tight")
            plt.close(fig_prob)
            log.info("Behavior probability saved to %s", subject_dir)

            # ── Module 4: Habituation curve ──
            fig_hab = plot_habituation_curve(df)
            fig_hab.savefig(subject_dir / "habituation_curve.png", dpi=300, bbox_inches="tight")
            plt.close(fig_hab)
            log.info("Habituation curve saved to %s", subject_dir)

            # ── Diagnostic: V_max distribution for threshold tuning ──
            fig_vmax = plot_vmax_distribution(df)
            fig_vmax.savefig(subject_dir / "vmax_distribution_diagnostic.png", dpi=300, bbox_inches="tight")
            plt.close(fig_vmax)
            log.info("V_max distribution diagnostic saved to %s", subject_dir)

            # ── Module 5: Individual Escape trial export ──
            if not df_response.empty:
                indiv_dir = subject_dir / "individual_escapes"
                indiv_dir.mkdir(parents=True, exist_ok=True)

                for tid, grp in df_response.groupby("global_trial_index"):
                    trial_data = grp.sort_values("t_rel")
                    row = grp.iloc[0]
                    lat = float(row["latency_ms"])
                    vmax = float(row["v_max"])

                    fig_trial = plot_single_escape_trial(trial_data, lat, vmax, int(tid))
                    fig_trial.savefig(indiv_dir / f"trial_{int(tid)}_escape.png", dpi=300, bbox_inches="tight")
                    plt.close(fig_trial)

                n_exported = df_response["global_trial_index"].nunique()
                log.info("Exported %d individual Escape trial figures to %s", n_exported, indiv_dir)

        else:
            plt.show()

    log.info("All subjects processed.")


if __name__ == "__main__":
    main()
