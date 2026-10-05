"""Classifier-driven endpoint selection for descriptive latency summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_escape_latency(trials: pd.DataFrame) -> pd.Series:
    """Use final response classes, never reselect trials from motion history.

    Wind PreWalk uses nonnegative causal RTm with missing endpoints left missing; Escape
    keeps its descriptive escape timing, PreEscape its lead time, and NoResponse
    has no primary RT. Unlabeled/nonwind/legacy tables keep their timing basis.
    Endpoint availability does not redefine the classifier cohort.
    """
    col = "escape_reaction_time_ms" if "escape_reaction_time_ms" in trials else "reaction_time_ms"
    rt = pd.to_numeric(trials[col], errors="coerce").copy()
    if "response_type" in trials:
        if "type" in trials and any(c in trials for c in (
            "pause_reaction_time_ms", "pause_stopping_time_ms", "pause_to_escape_time_ms",
            "pause_baseline_status", "pause_moving_eligible",
        )):
            wind = trials["type"].astype(str).str.contains("wind", case=False, na=False)
            prewalk = wind & trials["response_type"].eq("PreWalk")
            # A missing causal endpoint is not replaced by centered/stopping RT.
            rt.loc[prewalk] = np.nan
            if "pause_reaction_time_ms" in trials:
                causal = pd.to_numeric(trials["pause_reaction_time_ms"], errors="coerce")
                rt.loc[prewalk] = causal.where(causal >= 0.0).loc[prewalk]
        rt.loc[trials["response_type"].eq("NoResponse")] = np.nan
    return rt.where(np.isfinite(rt))
