"""MCMC binarization / KM event semantics after PreEscape became a class.

T1: escape_only 排除 PreEscape（不等风，拉偏 sigmoid）
T2: escape_prewalk 含 PreEscape（任何有 burst 的反应都是逃逸事件）
T3: KM 生存分析把 PreEscape 记为事件而非删失
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from pipeline.mcmc import prepare_mcmc_data, prepare_survival_data
from cercus.constants.response_types import BURST_CLASSES, ESCAPE_CLASSES


def _frame(tid, trial_type, ttc, response, latency_ms, t_last=500.0):
    t = np.array([-2000.0, t_last])
    return pd.DataFrame({
        "global_trial_id": tid,
        "type": trial_type,
        "target_ttc_ms": ttc,
        "response_type": response,
        "latency_ms": latency_ms,
        "t_rel": t,
    })


def _mixed_df() -> pd.DataFrame:
    """4 个多模态 trial：Escape / PreEscape / PreWalk / NoResponse 各一。"""
    return pd.concat(
        [
            _frame(1, "looming_wind", -373.0, "Escape", 100.0),
            _frame(2, "looming_wind", -373.0, "PreEscape", -600.0),
            _frame(3, "looming_wind", -373.0, "PreWalk", 80.0),
            _frame(4, "looming_wind", -373.0, "NoResponse", np.nan),
        ],
        ignore_index=True,
    )


def test_escape_only_excludes_preescape():
    _, escape, _, _ = prepare_mcmc_data(_mixed_df(), "escape_only")
    assert escape.sum() == 1  # 仅 Escape


def test_escape_prewalk_includes_preescape():
    _, escape, _, _ = prepare_mcmc_data(_mixed_df(), "escape_prewalk")
    assert escape.sum() == 3  # Escape + PreEscape + PreWalk


def test_km_counts_preescape_as_event():
    df = _mixed_df()
    surv, _start = prepare_survival_data(df)
    esc = surv[surv["stim_condition"].str.startswith("looming_wind")]
    assert esc["Event_Observed"].sum() == 2  # Escape + PreEscape
    pre = esc.loc[esc["TTC_at_event"] == -600.0]
    assert pre["Event_Observed"].iloc[0] == 1


def test_class_lists_are_switch_aware():
    assert "PreEscape" in BURST_CLASSES and "PreWalk" in BURST_CLASSES
    assert "NoResponse" not in BURST_CLASSES
    assert set(ESCAPE_CLASSES) == {"Escape", "PreEscape"}
