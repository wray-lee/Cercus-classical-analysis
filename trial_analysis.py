"""
Cercus Framework — Behavioral Neuroscience Data Analysis & Visualization
=========================================================================
Cross-session unified analysis pipeline. Scans a directory for per-session
event/kinematics CSV pairs, concatenates them by subject, and generates
publication-grade figures with dual-threshold escape classification.

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


# NPG (Nature Publishing Group) palette
COLOR_LEFT = "#4DBBD5"
COLOR_RIGHT = "#E64B35"
COLOR_CONTROL = "#999999"
COLOR_OSCI_VIS = "#8491B4"
COLOR_OSCI_HW = "#F39B7F"
COLOR_STARTLE = "#E64B35"
COLOR_WALK = "#4DBBD5"
COLOR_PRE_ACTIVE = "#CCCCCC"
COLOR_NO_RESPONSE = "#999999"

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
)
SPEED_WINDOW_MS = 100
SCALE_BAR_MM = 5.0
LEGACY_TRIAL_DURATION_MS = 5829.6
RADIUS_MM = 30.0

# ──────────────────────────────────────────────────────────────────────
# Dual-Threshold Escape Detection (Module 2)
# ──────────────────────────────────────────────────────────────────────
EVOKED_WALK_THRESHOLD = 15.0       # mm/s — evoked crawl baseline
STARTLE_ESCAPE_THRESHOLD = 100.0    # mm/s — explosive startle baseline
ESCAPE_SPEED_THRESHOLD = 10.0      # mm/s — sustained speed for consecutive-frame check
ESCAPE_CONSECUTIVE_K = 3           # frames above threshold to qualify
ESCAPE_WINDOW_MS = 250.0           # post-stimulus window (ms)

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

        # Remap meta
        if not meta.empty:
            meta = meta.copy()
            meta["global_trial_index"] = meta["global_trial_id"].map(
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

    dx_global = dx_body * np.cos(heading_rad) + dy_body * np.sin(heading_rad)
    dy_global = -dx_body * np.sin(heading_rad) + dy_body * np.cos(heading_rad)

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
# Module 2 — Dual-Threshold Ternary Classifier
# ══════════════════════════════════════════════════════════════════════


def _classify_trial(trial: pd.DataFrame) -> dict:
    """
    Ternary classification for a single trial using dual velocity thresholds.

    Criteria (post-stimulus, 0 < t_rel ≤ 250 ms):
        1. Consecutive-frame gate: ≥ ESCAPE_CONSECUTIVE_K frames > ESCAPE_SPEED_THRESHOLD (10 mm/s)
        2. If gate passes and V_max ≥ STARTLE_ESCAPE_THRESHOLD (30.0) → "Startle"
        3. If gate passes and EVOKED_WALK_THRESHOLD (15.0) ≤ V_max < 30.0 → "Walk"
        4. Otherwise → "NoResponse"

    Returns dict with: response_type, v_max, latency_ms
    """
    # ── Pre-stimulus rest period guard ──
    # Flag trials where the animal was already moving before stimulus onset,
    # to prevent residual momentum from triggering false Startle/Walk classifications.
    pre_stim = trial[(trial["t_rel"] >= -500.0) & (trial["t_rel"] < 0)]
    if not pre_stim.empty:
        pre_v_max = np.nanmax(pre_stim["speed"].values)
        if not np.isnan(pre_v_max) and pre_v_max >= EVOKED_WALK_THRESHOLD:
            # Extract v_max from the 0-250ms post-stimulus window; fallback to pre_v_max
            post_window = trial[(trial["t_rel"] > 0) & (trial["t_rel"] <= ESCAPE_WINDOW_MS)]
            if not post_window.empty:
                v_max_post = np.nanmax(post_window["speed"].values)
                v_max = v_max_post if not np.isnan(v_max_post) else float(pre_v_max)
            else:
                v_max = float(pre_v_max)
            return {"response_type": "Pre_Active", "v_max": float(v_max), "latency_ms": np.nan}

    post = trial[trial["t_rel"] > 0].copy()
    if post.empty:
        return {"response_type": "NoResponse", "v_max": np.nan, "latency_ms": np.nan}

    in_window = post[post["t_rel"] <= ESCAPE_WINDOW_MS]
    if in_window.empty:
        return {"response_type": "NoResponse", "v_max": np.nan, "latency_ms": np.nan}

    speed_vals = in_window["speed"].values
    t_vals = in_window["t_rel"].values

    v_max = np.nanmax(speed_vals)
    if np.isnan(v_max):
        return {"response_type": "NoResponse", "v_max": np.nan, "latency_ms": np.nan}

    # Consecutive-frame gate
    above = speed_vals > ESCAPE_SPEED_THRESHOLD
    consec = 0
    gate_passed = False
    latency_ms = np.nan
    for i, val in enumerate(above):
        if val:
            consec += 1
            if consec >= ESCAPE_CONSECUTIVE_K and not gate_passed:
                gate_passed = True
                latency_ms = float(t_vals[i - ESCAPE_CONSECUTIVE_K + 1])
        else:
            consec = 0

    if not gate_passed:
        return {"response_type": "NoResponse", "v_max": float(v_max), "latency_ms": np.nan}

    # Ternary classification by V_max
    if v_max >= STARTLE_ESCAPE_THRESHOLD:
        response_type = "Startle"
    elif v_max >= EVOKED_WALK_THRESHOLD:
        response_type = "Walk"
    else:
        response_type = "NoResponse"

    return {"response_type": response_type, "v_max": float(v_max), "latency_ms": latency_ms}


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

    n_startle = sum(1 for v in classify_map.values() if v == "Startle")
    n_walk = sum(1 for v in classify_map.values() if v == "Walk")
    n_pre = sum(1 for v in classify_map.values() if v == "Pre_Active")
    n_none = sum(1 for v in classify_map.values() if v == "NoResponse")
    n_total = len(classify_map)
    log.info(
        "Quaternary classification: Startle=%d, Walk=%d, Pre_Active=%d, NoResponse=%d (total=%d)",
        n_startle, n_walk, n_pre, n_none, n_total,
    )
    return df


# ══════════════════════════════════════════════════════════════════════
# Plot 1 — Trajectory Overlay
# ══════════════════════════════════════════════════════════════════════


def _draw_cross_axes(ax: plt.Axes, scale_bar_val: float = SCALE_BAR_MM):
    """Draw cross-shaped origin axes with arrowheads and a minimalist scale bar."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.annotate("", xy=(xlim[1], 0), xytext=(xlim[0], 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.75))
    ax.annotate("", xy=(0, ylim[1]), xytext=(0, ylim[0]),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.75))

    sb_x = xlim[1] * 0.65
    sb_y = ylim[0] * 0.85
    ax.plot([sb_x, sb_x + scale_bar_val], [sb_y, sb_y], "k-", lw=1.0, solid_capstyle="butt")
    ax.text(sb_x + scale_bar_val / 2, sb_y - (ylim[1] - ylim[0]) * 0.02,
            f"{scale_bar_val:.0f} mm", ha="center", va="top", fontsize=7)


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
    """One subplot per trial type. Left stimuli in NPG blue, right in NPG red."""
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
            # 1. 提取物理意义上的有效反应段 (t_rel ≥ 0，且硬上限限制在 300 ms 内)
            burst = grp[(grp["t_rel"] >= 0) & (grp["t_rel"] <= 300)].copy()
            if burst.empty:
                continue

            # 2. 锁定爆发峰值点
            peak_idx = burst["speed"].idxmax()
            if pd.isna(peak_idx):
                continue

            # 3. 动态截断：从峰值点向后寻找，一旦速度跌破 EVOKED_WALK_THRESHOLD (15.0)，即视为爆发动作结束
            post_peak = burst.loc[peak_idx:]
            stop_frames = post_peak[post_peak["speed"] < EVOKED_WALK_THRESHOLD]

            if not stop_frames.empty:
                end_idx = stop_frames.index[0]
                burst = burst.loc[:end_idx]

            # 4. 获取颜色并绘图
            ss = str(burst["screen_side"].iloc[0]).strip().lower() if pd.notna(burst["screen_side"].iloc[0]) else ""
            if ttype == control_type:
                color = COLOR_CONTROL
            elif ss == "left":
                color = left_color
            elif ss == "right":
                color = right_color
            else:
                color = COLOR_CONTROL

            # 调整线宽与透明度，使有效轨迹更清晰
            ax.plot(burst["x"], burst["y"], color=color, alpha=0.4, lw=0.8)

        ax.set_aspect("equal")
        ax.set_title(ttype, fontweight="bold")
        _draw_cross_axes(ax)
        _draw_side_arrows(ax, left_color=left_color, right_color=right_color)

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Plot 2 — Speed Kinetics with Dual-Threshold Lines
# ══════════════════════════════════════════════════════════════════════


def _add_threshold_lines(ax: plt.Axes):
    """Draw horizontal threshold lines at 30 mm/s (startle) and 15 mm/s (walk)."""
    ax.axhline(y=STARTLE_ESCAPE_THRESHOLD, color="k", linestyle="--", linewidth=0.75, alpha=0.7)
    ax.text(ax.get_xlim()[1] * 0.98, STARTLE_ESCAPE_THRESHOLD + 1.0,
            "Startle/Jump Threshold", ha="right", va="bottom", fontsize=6, color="k", alpha=0.7)

    ax.axhline(y=EVOKED_WALK_THRESHOLD, color="0.5", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.text(ax.get_xlim()[1] * 0.98, EVOKED_WALK_THRESHOLD + 1.0,
            "Walk Threshold", ha="right", va="bottom", fontsize=6, color="0.5", alpha=0.5)


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
            ss = str(sample.get("screen_side", "")).strip().lower()
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
            ss = str(sample.get("screen_side", "")).strip().lower()
            cond_color_map[ttype] = COLOR_LEFT if ss == "left" else COLOR_RIGHT if ss == "right" else COLOR_LEFT

    conditions = sorted(cond_color_map.keys())
    n_conds = len(conditions)
    if n_conds == 0:
        fig, ax = plt.subplots()
        return fig

    cmap = plt.cm.get_cmap("tab20")
    trial_idx = 0

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
            trial_color = cmap(trial_idx % 20)
            trial_idx += 1
            ax.plot(grp_sorted["t_rel"], grp_sorted["speed"], color=trial_color, lw=0.75, alpha=0.6)

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
    Bar chart showing the proportion of Startle / Walk / NoResponse across
    all trials for this subject.

    Returns the matplotlib Figure.
    """
    counts = df.groupby("global_trial_id")["response_type"].first().value_counts()
    total = counts.sum()

    categories = ["Startle", "Walk", "Pre_Active", "NoResponse"]
    values = [counts.get(c, 0) / total if total > 0 else 0.0 for c in categories]
    colors = [COLOR_STARTLE, COLOR_WALK, COLOR_PRE_ACTIVE, COLOR_NO_RESPONSE]

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
    Y-axis: V_max (mm/s) within 0–250 ms response window
    Horizontal anchors at Y=30.0 (black dashed) and Y=15.0 (gray dashed).

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
    color_map = {"Startle": COLOR_STARTLE, "Walk": COLOR_WALK, "Pre_Active": COLOR_PRE_ACTIVE, "NoResponse": COLOR_NO_RESPONSE}
    point_colors = [color_map.get(rt, COLOR_NO_RESPONSE) for rt in trial_agg["response_type"].values]

    fig, ax = plt.subplots(figsize=(8, 3.5))

    # Line underneath
    ax.plot(x, y, color="0.7", lw=0.8, alpha=0.6, zorder=1)
    # Scatter on top
    ax.scatter(x, y, c=point_colors, s=28, edgecolors="white", linewidths=0.4, zorder=2)

    # Threshold anchors
    ax.axhline(y=STARTLE_ESCAPE_THRESHOLD, color="k", linestyle="--", linewidth=0.75, alpha=0.7)
    ax.text(x[-1] + 0.3, STARTLE_ESCAPE_THRESHOLD, f"{STARTLE_ESCAPE_THRESHOLD:.0f}", ha="left", va="center", fontsize=6, color="k", alpha=0.7)

    ax.axhline(y=EVOKED_WALK_THRESHOLD, color="0.5", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.text(x[-1] + 0.3, EVOKED_WALK_THRESHOLD, f"{EVOKED_WALK_THRESHOLD:.0f}", ha="left", va="center", fontsize=6, color="0.5", alpha=0.5)

    ax.set_xlabel("Global Trial Index")
    ax.set_ylabel("V$_{max}$ (mm/s)")
    ax.set_title("Habituation Curve", fontweight="bold")
    ax.set_xlim(0.5, len(x) + 0.5)

    # Minimalist legend for response types
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=5, label=l)
               for l, c in [("Startle", COLOR_STARTLE), ("Walk", COLOR_WALK), ("Pre_Active", COLOR_PRE_ACTIVE), ("NoResponse", COLOR_NO_RESPONSE)]]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=6)

    fig.tight_layout(pad=1.0)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Diagnostic — V_max Frequency Distribution (Bimodal Threshold Tuning)
# ══════════════════════════════════════════════════════════════════════


def plot_vmax_distribution(df: pd.DataFrame, figsize: tuple[float, float] = (5.0, 3.5)) -> plt.Figure:
    """
    Histogram + KDE of per-trial V_max for response trials (Startle + Walk).

    Used to visually identify the bimodal distribution valley that separates
    evoked walk from startle/jump, enabling objective threshold refinement.

    Returns the matplotlib Figure.
    """
    # Only keep trials with an actual response (exclude NoResponse)
    df_resp = df[df["response_type"].isin(["Startle", "Walk"])].copy()
    if df_resp.empty:
        log.warning("No response trials for V_max distribution plot.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No response trials", ha="center", va="center", transform=ax.transAxes)
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
    ax.axvline(EVOKED_WALK_THRESHOLD, color="0.5", ls="--", lw=0.75, alpha=0.7, zorder=4)
    ax.text(EVOKED_WALK_THRESHOLD + 3, ax.get_ylim()[1] * 0.92,
            f"Walk\n{EVOKED_WALK_THRESHOLD:.0f}", fontsize=6, color="0.5", va="top")

    ax.axvline(STARTLE_ESCAPE_THRESHOLD, color="black", ls="--", lw=0.75, alpha=0.7, zorder=4)
    ax.text(STARTLE_ESCAPE_THRESHOLD + 3, ax.get_ylim()[1] * 0.92,
            f"Startle\n{STARTLE_ESCAPE_THRESHOLD:.0f}", fontsize=6, color="black", va="top")

    ax.set_xlim(0, 400)
    ax.set_xlabel("$V_{max}$ (mm/s)")
    ax.set_ylabel("Probability Density")
    ax.set_title("$V_{max}$ Distribution — Threshold Diagnostic", fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=6)

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
        df_response = df[df["response_type"].isin(["Startle", "Walk"])].copy()
        df_no_response = df[df["response_type"] == "NoResponse"].copy()

        n_resp = df_response["global_trial_id"].nunique()
        n_nr = df_no_response["global_trial_id"].nunique()
        log.info("Split: %d response (Startle+Walk), %d no-response", n_resp, n_nr)

        # Peak diagnostic for discarded trials
        if not df_no_response.empty:
            diag_window = df_no_response[(df_no_response["t_rel"] > 0) & (df_no_response["t_rel"] <= ESCAPE_WINDOW_MS)]
            if not diag_window.empty:
                peaks = diag_window.groupby("global_trial_id")["speed"].max()
                log.info("====== NoResponse peak diagnostic (0–250 ms) ======")
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

        else:
            plt.show()

    log.info("All subjects processed.")


if __name__ == "__main__":
    main()
