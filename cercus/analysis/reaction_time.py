"""Shared endpoint selection for descriptive latency summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_escape_latency(trials: pd.DataFrame) -> pd.Series:
    """Keep causal moving RTm missing rather than borrowing centered timing.

    Current wind tables exclude intermittent/unobserved histories from the
    primary latency summary. Below-threshold histories retain the existing
    centered escape measurement, not a validated causal paper RT. PreEscape
    retains its diagnostic lead time. Legacy/nonwind tables keep their basis.
    """
    col = "escape_reaction_time_ms" if "escape_reaction_time_ms" in trials else "reaction_time_ms"
    rt = pd.to_numeric(trials[col], errors="coerce").copy()
    if {"type", "pause_baseline_status"}.issubset(trials.columns):
        wind = trials["type"].astype(str).str.contains("wind", case=False, na=False)
        if "response_type" in trials:
            wind &= ~trials["response_type"].eq("PreEscape")
        baseline = trials["pause_baseline_status"]
        rt.loc[wind & ~baseline.eq("stationary")] = np.nan
        moving = wind & baseline.eq("continuous_moving")
        if "pause_reaction_time_ms" in trials:
            rt.loc[moving] = pd.to_numeric(trials.loc[moving, "pause_reaction_time_ms"], errors="coerce")
    if "response_type" in trials:
        rt.loc[trials["response_type"].eq("NoResponse")] = np.nan
    return rt
