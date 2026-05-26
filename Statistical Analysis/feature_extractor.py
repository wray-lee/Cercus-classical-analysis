"""
Feature Extractor — Per-Trial Scalar Extraction & Behavior Classification
==========================================================================
Reads preprocessed kinematics data (output of trial_analysis.py) and extracts
per-trial scalar features: escape detection, behavior typing (Jump/Run),
and escape direction.

Outputs:
    session_summary.csv      — one row per trial with scalar features
    session_timeseries.parquet — filtered high-frequency time series

Usage:
    python feature_extractor.py --events path/to/events.csv --kinematics path/to/kinematics.csv
    python feature_extractor.py --events e.csv --kinematics k.csv --output results/
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# Reuse the preprocessing pipeline from trial_analysis
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trial_analysis import (
    load_events,
    load_kinematics,
    preprocess,
    _apply_publication_style,
    DETAILS_KEYS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Escape Detection Parameters
# ──────────────────────────────────────────────────────────────────────
ESCAPE_SPEED_THRESHOLD = 10.0      # mm/s — sustained speed threshold
ESCAPE_V_MAX_THRESHOLD = 50.0      # mm/s — peak speed within escape window
ESCAPE_CONSECUTIVE_K = 3           # frames above threshold to qualify
ESCAPE_WINDOW_MS = 250.0           # post-stimulus window to search (ms)

# Behavior Classification Parameters
JUMP_ACCEL_THRESHOLD = 500.0       # mm/s² — peak acceleration for Jump classification
JUMP_DAMPING_RATIO = 0.3           # ratio of second peak to first peak (< this → Jump)


# ══════════════════════════════════════════════════════════════════════
# Per-Trial Feature Extraction
# ══════════════════════════════════════════════════════════════════════

def _detect_escape(trial: pd.DataFrame) -> dict:
    """
    Escape detection for a single trial.

    Criteria (post-stimulus, t_rel > 0):
        - speed > ESCAPE_SPEED_THRESHOLD for ESCAPE_CONSECUTIVE_K consecutive frames
        - max speed within the escape window > ESCAPE_V_MAX_THRESHOLD

    Returns dict with: is_escaped (bool), latency_ms (float or NaN)
    """
    post = trial[trial["t_rel"] > 0].copy()
    if post.empty:
        return {"is_escaped": False, "latency_ms": np.nan}

    # Restrict to escape window
    window_end = ESCAPE_WINDOW_MS
    in_window = post[post["t_rel"] <= window_end]
    if in_window.empty:
        return {"is_escaped": False, "latency_ms": np.nan}

    speed_vals = in_window["speed"].values
    t_vals = in_window["t_rel"].values

    # Check max speed
    v_max = np.nanmax(speed_vals)
    if np.isnan(v_max) or v_max <= ESCAPE_V_MAX_THRESHOLD:
        return {"is_escaped": False, "latency_ms": np.nan}

    # Find consecutive frames above threshold
    above = speed_vals > ESCAPE_SPEED_THRESHOLD
    consec = 0
    for i, val in enumerate(above):
        if val:
            consec += 1
            if consec >= ESCAPE_CONSECUTIVE_K:
                # Latency = time of the first frame in this consecutive run
                latency_idx = i - ESCAPE_CONSECUTIVE_K + 1
                return {"is_escaped": True, "latency_ms": float(t_vals[latency_idx])}
        else:
            consec = 0

    return {"is_escaped": False, "latency_ms": np.nan}


def _classify_behavior(trial: pd.DataFrame) -> str:
    """
    Behavior classification for escaped trials.

    Extracts peak acceleration (a_max) and post-peak damping profile.
    If a_max > threshold and smooth exponential damping → 'Jump', else → 'Run'.
    """
    speed = trial["speed"].values
    t = trial["t_rel"].values
    dt = np.diff(t) / 1000.0  # convert ms to seconds

    if len(speed) < 3 or len(dt) < 1:
        return "Run"

    # Acceleration via finite differences
    accel = np.diff(speed) / dt
    # Only look post-stimulus
    post_mask = t[1:] > 0
    if not np.any(post_mask):
        return "Run"

    accel_post = accel[post_mask]
    a_max = np.nanmax(accel_post)

    if a_max < JUMP_ACCEL_THRESHOLD:
        return "Run"

    # Check damping: find first peak, then look at subsequent peaks
    peak_idx = np.nanargmax(accel_post)
    speed_post = speed[1:][post_mask]
    t_post = t[1:][post_mask]

    # Time-decoupled decay: find the first sample >= 50ms after the peak
    t_peak = t_post[peak_idx]
    decay_mask = t_post >= (t_peak + 50.0)
    if np.any(decay_mask):
        decay_idx = np.argmax(decay_mask)
    else:
        decay_idx = len(speed_post) - 1  # fallback to last frame

    peak_speed = speed_post[peak_idx]
    decay_speed = speed_post[decay_idx]

    if peak_speed > 0:
        damping_ratio = decay_speed / peak_speed
    else:
        damping_ratio = 1.0

    # If damping is smooth (ratio < threshold) → Jump (ballistic burst + rapid decay)
    # If speed stays elevated → Run (sustained locomotion)
    if damping_ratio < JUMP_DAMPING_RATIO:
        return "Jump"
    return "Run"


def _extract_escape_direction(trial: pd.DataFrame) -> float:
    """
    Extract escape direction as polar angle theta = arctan2(dy, dx).

    Uses the displacement in a short window around the escape onset.
    Returns angle in radians.
    """
    post = trial[trial["t_rel"] > 0].copy()
    if post.empty or len(post) < 2:
        return np.nan

    # Use first 100ms of post-stimulus displacement
    window = post[post["t_rel"] <= 100.0]
    if window.empty or len(window) < 2:
        window = post.head(max(2, min(10, len(post))))

    dx_total = window["x"].iloc[-1] - window["x"].iloc[0]
    dy_total = window["y"].iloc[-1] - window["y"].iloc[0]

    if dx_total == 0 and dy_total == 0:
        return np.nan

    return math.atan2(dy_total, dx_total)


def extract_trial_features(trial: pd.DataFrame) -> dict:
    """
    Extract all scalar features from a single trial DataFrame.

    Returns dict with: global_trial_id, type, screen_side, is_escaped,
    latency_ms, behavior_type, escape_direction_rad
    """
    tid = trial["global_trial_id"].iloc[0] if "global_trial_id" in trial.columns else np.nan
    trial_type = trial["type"].iloc[0] if "type" in trial.columns else np.nan
    screen_side = trial["screen_side"].iloc[0] if "screen_side" in trial.columns else np.nan

    # Escape detection
    esc = _detect_escape(trial)

    # Behavior classification (only for escaped trials)
    if esc["is_escaped"]:
        behavior = _classify_behavior(trial)
        direction = _extract_escape_direction(trial)
    else:
        behavior = "NoEscape"
        direction = np.nan

    return {
        "global_trial_id": tid,
        "type": trial_type,
        "screen_side": screen_side,
        "is_escaped": esc["is_escaped"],
        "latency_ms": esc["latency_ms"],
        "behavior_type": behavior,
        "escape_direction_rad": direction,
    }


# ══════════════════════════════════════════════════════════════════════
# Session-Level Pipeline
# ══════════════════════════════════════════════════════════════════════

def extract_session(
    events_path: str | Path,
    kinematics_path: str | Path,
    output_dir: str | Path | None = None,
    session_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full feature extraction pipeline for one session.

    Parameters
    ----------
    events_path : path to events CSV
    kinematics_path : path to kinematics CSV
    output_dir : directory to write session_summary.csv and session_timeseries.parquet
    session_id : identifier for this session (defaults to stem of kinematics file)

    Returns
    -------
    summary : DataFrame — one row per trial with scalar features
    timeseries : DataFrame — full preprocessed time series with trial-level annotations
    """
    if session_id is None:
        session_id = Path(kinematics_path).stem.replace("_kinematics", "")

    log.info("Processing session: %s", session_id)

    # Load and preprocess using existing pipeline
    meta, windows, ttc_anchors = load_events(events_path)
    kin = load_kinematics(kinematics_path)
    df = preprocess(meta, windows, ttc_anchors, kin)

    log.info("  Preprocessed: %d frames across %d trials",
             len(df), df["global_trial_id"].nunique())

    # Extract per-trial features
    records = []
    for tid, grp in df.groupby("global_trial_id"):
        features = extract_trial_features(grp)
        features["session_id"] = session_id
        records.append(features)

    summary = pd.DataFrame(records)
    escaped_n = summary["is_escaped"].sum()
    log.info("  Escaped: %d / %d trials (%.1f%%)",
             escaped_n, len(summary), 100 * escaped_n / len(summary) if len(summary) > 0 else 0)

    # Add session_id to timeseries
    timeseries = df.copy()
    timeseries["session_id"] = session_id

    # Save outputs
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        summary_path = out / "session_summary.csv"
        summary.to_csv(summary_path, index=False)
        log.info("  Saved: %s", summary_path)

        ts_path = out / "session_timeseries.parquet"
        timeseries.to_parquet(ts_path, index=False)
        log.info("  Saved: %s", ts_path)

    return summary, timeseries


# ══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Feature extractor — per-trial escape detection & behavior classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--events", required=True, help="Path to *_events.csv")
    p.add_argument("--kinematics", required=True, help="Path to *_kinematics.csv")
    p.add_argument("--session-id", default=None, help="Session identifier (default: from filename)")
    p.add_argument("--output", default=".", help="Output directory (default: current dir)")
    return p


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    summary, timeseries = extract_session(
        events_path=args.events,
        kinematics_path=args.kinematics,
        output_dir=args.output,
        session_id=args.session_id,
    )
    return summary, timeseries


if __name__ == "__main__":
    main()
